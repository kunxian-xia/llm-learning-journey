import torch
import math
import triton
import triton.language as tl


class FlashAttentionTorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        if is_causal:
            # TODO: handle the case is_causal == true
            raise NotImplementedError()

        # implement the forward pass of vanilla attention
        # using the flash attention algorithm, but using only
        # standard PyTorch operations (no Triton!)
        assert len(q.shape) == len(k.shape) == len(v.shape) == 3
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert q.shape[2] == k.shape[2] == v.shape[2]
        assert k.shape[1] == v.shape[1]
        assert q.device == k.device == v.device
        assert q.dtype == k.dtype == v.dtype

        tile_size = 32

        num_keys = k.shape[1]
        num_queries = q.shape[1]

        batch_size = q.shape[0]
        dim = q.shape[2]
        num_q_tiles = (num_queries + tile_size - 1) // tile_size
        num_k_tiles = (num_keys + tile_size - 1) // tile_size

        scale = 1 / math.sqrt(dim)

        Out = torch.empty((q.shape[0], q.shape[1], v.shape[-1]), device=q.device, dtype=v.dtype)
        Logsumexp = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=v.dtype)

        for i in range(num_q_tiles):
            q_start = i * tile_size
            q_end = min((i + 1) * tile_size, num_queries)
            num_queries_in_tile = q_end - q_start

            q_tile = q[..., q_start:q_end, :]

            # init Mi = -inf, Li = 0, and Oi = 0 for each tile
            m = torch.full(
                (batch_size, num_queries_in_tile), float("-inf"), device=q.device, dtype=q.dtype
            )
            l = torch.zeros((batch_size, num_queries_in_tile), device=q.device, dtype=q.dtype)
            o = torch.zeros(
                (batch_size, num_queries_in_tile, v.shape[-1]), device=q.device, dtype=v.dtype
            )

            for j in range(num_k_tiles):
                start = j * tile_size
                end = min((j + 1) * tile_size, k.shape[-2])
                k_tile = k[..., start:end, :]
                v_tile = v[..., start:end, :]

                # a. compute local tile's scaled attention score matrix Sij = (qi * kj^T) / sqrt(d)
                Sij = torch.einsum("... q d, ... k d -> ... q k", q_tile, k_tile * scale)

                # b. compute local tile's row-wise max of Sij up to j: Mi_new = max(Mi, row_max(Sij))
                m_new = torch.max(m, torch.max(Sij, dim=-1).values)

                alpha = torch.exp(m - m_new)
                adjusted_exp_sij = torch.exp(Sij - m_new.unsqueeze(-1))

                # c. local tile's denominator: Lij = exp(Mi - Mi_new) * Li^{j-1} + row_sum(exp(Sij - Mi^{j}))
                Lij = alpha * l + torch.sum(adjusted_exp_sij, dim=-1).squeeze(-1)
                l = Lij

                # d. local tile's output: Oij = exp(Mi_old - Mi) * Oi^{j-1} + exp(Sij - Mi^{j}) * vj
                Oij = alpha.unsqueeze(-1) * o + torch.einsum(
                    "... q k, ... k d -> ... q d", adjusted_exp_sij, v_tile
                )
                o = Oij
                m = m_new

            # 2. Oi = diag(Li)^{-1} * Oi and write it to output tensor's memory
            Out[..., q_start:q_end, :] = o / l.unsqueeze(-1)
            Logsumexp[..., q_start:q_end] = m + torch.log(l)

        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, Out, Logsumexp)

        # 3. return tensor O
        return Out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, Out, Logsumexp = ctx.saved_tensors
        is_causal = ctx.is_causal

        tile_size = 32
        scale = 1 / math.sqrt(q.shape[-1])

        # Li = log(\sum_j exp(S_{ij}))
        num_keys = k.shape[-2]
        num_queries = q.shape[-2]

        num_q_tiles = (num_queries + tile_size - 1) // tile_size
        num_k_tiles = (num_keys + tile_size - 1) // tile_size

        dV = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
        dK = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
        dQ = torch.zeros(q.shape, device=q.device, dtype=torch.float32)

        for j in range(num_k_tiles):
            k_start = j * tile_size
            k_end = min((j + 1) * tile_size, num_keys)

            k_tile = k[..., k_start:k_end, :]

            for i in range(num_q_tiles):
                q_start = i * tile_size
                q_end = min((i + 1) * tile_size, num_queries)

                dO = grad_output[..., q_start:q_end, :]
                q_tile = q[..., q_start:q_end, :]
                Li = Logsumexp[..., q_start:q_end]

                # P_{ij} = exp(S_{ij} - Li)
                Sij = torch.einsum("... q d, ... k d -> ... q k", q_tile, k_tile * scale)
                Pij = torch.exp(Sij - Li.unsqueeze(-1))

                # dV_{ja} += \sum_i P_ij * dO_ia
                # j is the key dimension, a is the feature dimension
                dVj = torch.einsum("... i j, ... i a -> ... j a", Pij, dO)
                dV[..., k_start:k_end, :] += dVj

                # dP = dO * V^T
                # dPij = \sum_a dO_ia * V_ja
                dPij = torch.einsum("... i a, ... j a -> ... i j", dO, v[..., k_start:k_end, :])

                # D_i = rowsum(dO_i ⊙ O_i)
                Di = torch.sum(dO * Out[..., q_start:q_end, :], dim=-1)

                # dS_{ij} = P_{i,j} * (dP_{i,j} - D_i)
                dSij = Pij * (dPij - Di.unsqueeze(-1))

                # dK
                # dK_{ja} += \sum_i (dS_ij) * Q_ia * scale
                dKj = torch.einsum("... i j, ... i a -> ... j a", dSij, q_tile * scale)
                dK[..., k_start:k_end, :] += dKj

        # TODO: dQ = dS * K * scale
        for i in range(num_q_tiles):
            q_start = i * tile_size
            q_end = min((i + 1) * tile_size, num_queries)

            q_tile = q[..., q_start:q_end, :]
            Li = Logsumexp[..., q_start:q_end]

            # D_i = rowsum(dO_i ⊙ O_i)
            dO = grad_output[..., q_start:q_end, :]
            Di = torch.sum(dO * Out[..., q_start:q_end, :], dim=-1)

            for j in range(num_k_tiles):
                k_start = j * tile_size
                k_end = min((j + 1) * tile_size, num_keys)

                k_tile = k[..., k_start:k_end, :]

                # P_{ij} = exp(S_{ij} - Li)
                Sij = torch.einsum("... q d, ... k d -> ... q k", q_tile, k_tile * scale)
                Pij = torch.exp(Sij - Li.unsqueeze(-1))

                # dP = dO * V^T
                # dPij = \sum_a dO_ia * V_ja
                dPij = torch.einsum("... i a, ... j a -> ... i j", dO, v[..., k_start:k_end, :])

                # dS_{ij} = P_{i,j} * (dP_{i,j} - D_i)
                dSij = Pij * (dPij - Di.unsqueeze(-1))

                # dQ_{ia} = \sum_j dSij * Kja * scale
                dQ[..., q_start:q_end, :] += torch.einsum(
                    "... i j, ... j a -> ... i a", dSij * scale, k[..., k_start:k_end, :]
                )

        return dQ, dK, dV, None


@triton.jit
def flash_attention_fwd_kernel(
    # fmt: off
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    # fmt: on
):
    #########################################################################
    ####  implement the forward pass of flash attention using Triton kernels
    #########################################################################

    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if is_causal:
        n_key_tiles = tl.minimum(
            n_key_tiles, tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE)
        )

    for i in range(n_key_tiles):
        k_start = i * K_TILE_SIZE
        # load k, v
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # compute local attention score matrix: Sij = q * ki^T / sqrt(d) of size O(TILE_SIZE^2)
        attn_score = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale

        # for causal masking, valid k indices are those that are less than N_KEYS
        # and less than or equal to the current query index (q_offsets)
        k_offsets = k_start + tl.arange(0, K_TILE_SIZE)
        if is_causal:
            valid_k = (k_offsets[None, :] < N_KEYS) & (k_offsets[None, :] <= q_offsets[:, None])
        else:
            valid_k = k_offsets[None, :] < N_KEYS
        attn_score = tl.where(valid_k, attn_score, float("-inf"))

        # compute Mi' = max(Mi, row_max(Sij))
        m_new = tl.maximum(m, tl.max(attn_score, axis=1))

        # adjusted_exp_sij = exp(Sij - Mi')
        adjusted_exp_s = tl.exp(attn_score - m_new[:, None])

        # alpha = exp(Mi - Mi')
        alpha = tl.exp(m - m_new)

        # L' = alpha * L + row_sum(adjusted_exp_sij)
        l = alpha * l + tl.sum(adjusted_exp_s, axis=1)

        # O' = alpha * O + adjusted_exp_sij * vj
        p = adjusted_exp_s.to(v.dtype)
        o = alpha[:, None] * o + tl.dot(p, v, out_dtype=tl.float32)

        # advance k_block
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        m = m_new

    # 2. Oi = diag(Li)^{-1} * Oi and write it to output tensor's memory
    o = o / l[:, None]
    final_l = tl.log(l) + m

    tl.store(L_block_ptr, final_l, boundary_check=(0,))
    tl.store(O_block_ptr, o, boundary_check=(0, 1))


@triton.jit
def flash_attention_bwd_dq_kernel(
    # fmt: off
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    dQ_ptr, dO_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dob, stride_doq, stride_dod,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    # fmt: on
):
    #########################################################################
    ### compute the gradients dQ of flash attention using Triton kernels
    ###
    #########################################################################
    batch_index = tl.program_id(0)
    query_tile_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # dQ = dS * K * scale
    num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    L = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    dQ = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # Di = rowsum(dO_i ⊙ O_i)
    dO_tile = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    O_tile = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Di = tl.sum(dO_tile * O_tile, axis=1)

    for i in range(num_key_tiles):
        k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Sij = (q * k^T) / sqrt(d)
        attn_score = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * scale

        k_start = i * K_TILE_SIZE
        k_offsets = k_start + tl.arange(0, K_TILE_SIZE)
        attn_score = tl.where(k_offsets[None, :] < N_KEYS, attn_score, float("-inf"))
        # TODO: handle causal masking

        # Pij = exp(Sij - Li)
        Pij = tl.exp(attn_score - L[:, None])

        # dP = dO * V^T
        # dPij = \sum_a dO_ia * V_ja
        dPij = tl.dot(dO_tile, tl.trans(v_tile), out_dtype=tl.float32)

        # dSij = Pij * (dPij - Di)
        dSij = Pij * (dPij - Di[:, None])

        # dQ = dS * K * scale
        dQ += tl.dot(dSij, k_tile.to(tl.float32), out_dtype=tl.float32) * scale

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQ, boundary_check=(0, 1))


@triton.jit
def flash_attention_bwd_dkdv_kernel(
    # fmt: off
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    dK_ptr, dV_ptr, dO_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    stride_dob, stride_doq, stride_dod,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    # fmt: on
):
    #########################################################################
    ### compute the gradients dK, dV of flash attention using Triton kernels
    #########################################################################

    batch_index = tl.program_id(0)
    tile_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    n_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

    k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dK = tl.zeros_like(k_tile)
    dV = tl.zeros_like(v_tile)

    # iter over all query tiles for one key tile due to the fact that
    # dK relies on dS^T and dV relies on P^T
    # notation convention: i is query dimension, j is key dimension, a is feature dimension
    for i in range(n_q_tiles):
        q_start = i * Q_TILE_SIZE
        q_end = min((i + 1) * Q_TILE_SIZE, N_QUERIES)

        q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        O_tile = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO_tile = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # TODO: handle causal masking

        # Sij = (q * k^T) / sqrt(d)
        Sij = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32) * scale

        # Pij = exp(Sij - Li)
        Pij = tl.exp(Sij - L[:, None])

        # dPij = dO_i * V_j^T
        # dPij = \sum_a dO_ia * V_ja
        dPij = tl.dot(dO_tile, tl.trans(v_tile), out_dtype=tl.float32)

        # Di = rowsum(dO_i ⊙ O_i)
        Di = tl.sum(dO_tile * O_tile, axis=1)

        # dSij = Pij * (dPij - Di)
        dSij = Pij * (dPij - Di[:, None])

        # dK = dS^T * Q * scale
        # dKja = \sum_i dS_ij * Q_ia * scale
        dK += tl.dot(tl.trans(dSij), q_tile, out_dtype=tl.float32) * scale

        # dV = P^T * dO
        # dVja = \sum_i P_ij * dO_ia
        dV += tl.dot(tl.trans(Pij), dO_tile, out_dtype=tl.float32)

        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        O_block_ptr = tl.advance(O_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))

    tl.store(dK_block_ptr, dK, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV, boundary_check=(0, 1))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        #######################################################################
        #### implement the forward pass of flash attention using Triton kernels
        #######################################################################

        # assumes q, k, v of shape (b, n, d)
        assert len(q.shape) == len(k.shape) == len(v.shape) == 3
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert q.shape[-1] == k.shape[-1] == v.shape[-1]
        assert k.shape[-2] == v.shape[-2]
        assert q.device == k.device == v.device
        assert q.is_cuda, "Expected CUDA Tensors"
        assert q.dtype == k.dtype == v.dtype
        assert q.dtype in (torch.float16, torch.bfloat16, torch.float32)

        D, N_QUERIES, N_KEYS = q.shape[-1], q.shape[-2], k.shape[-2]
        assert N_QUERIES > 0 and N_KEYS > 0 and D > 0

        Out = torch.empty_like(q)
        Logsumexp = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)

        # TODO: make this configurable
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16

        flash_attention_fwd_kernel[(triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE), q.shape[0])](
            # fmt: off
            q, k, v, Out, Logsumexp,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2),
            Logsumexp.stride(0), Logsumexp.stride(1),
            N_QUERIES, N_KEYS, 1 / math.sqrt(D), D,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=is_causal,
            # fmt: on
        )

        ctx.save_for_backward(q, k, v, Out, Logsumexp)
        ctx.is_causal = is_causal

        return Out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, Out, Logsumexp = ctx.saved_tensors
        is_causal = ctx.is_causal

        batch_size = q.shape[0]
        num_queries = q.shape[1]
        num_keys = k.shape[1]
        D = q.shape[2]

        dQ = torch.empty_like(q)
        dK = torch.empty_like(k)
        dV = torch.empty_like(v)
        dO = grad_output

        flash_attention_bwd_dq_kernel[(batch_size, triton.cdiv(num_queries, ctx.Q_TILE_SIZE))](
            # fmt: off
            q, k, v, Out, Logsumexp,
            dQ, dO,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2),
            Logsumexp.stride(0), Logsumexp.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            num_queries, num_keys, 1 / math.sqrt(D),
            D=D,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=is_causal,
            # fmt: on
        )

        flash_attention_bwd_dkdv_kernel[(batch_size, triton.cdiv(num_keys, ctx.K_TILE_SIZE))](
            # fmt: off
            q, k, v, Out, Logsumexp,
            dK, dV, dO,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2),
            Logsumexp.stride(0), Logsumexp.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            num_queries, num_keys, 1 / math.sqrt(D),
            D=D,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=is_causal
            # fmt: on
        )

        return dQ, dK, dV, None
