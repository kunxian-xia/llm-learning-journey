# Single-request inference API contract

Scope: [issue #49](https://github.com/kunxian-xia/llm-learning-journey/issues/49). Run one prompt through `tokenize -> prefill -> decode loop -> detokenize` in an in-process Python API. This document is the implementation handoff; it does not prescribe a web server.

## Public API

```python
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence
import torch
from jaxtyping import Float

StopReason = Literal["stop_token", "length"]

@dataclass(frozen=True)
class SamplingConfig:
    mode: Literal["greedy", "multinomial"] = "greedy"
    temperature: float = 1.0
    generator: torch.Generator | None = None

def select_next_token(
    logits: Float[torch.Tensor, "vocab"],
    sampling: SamplingConfig,
) -> int: ...

@dataclass(frozen=True)
class GenerateRequest:
    prompt: str
    max_new_tokens: int = 32
    stop_token_ids: frozenset[int] | None = None  # None: use model defaults
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

@dataclass(frozen=True)
class GenerateResult:
    text: str                 # generated continuation only; no prompt or stop token
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]  # includes a terminal stop token if sampled
    stop_reason: StopReason

class Tokenizer(Protocol):
    def encode_prompt(self, text: str) -> list[int]: ...
    def decode_generated(self, ids: list[int]) -> str: ...

class KVCache(Protocol):
    @property
    def sequence_length(self) -> int: ...

@dataclass(frozen=True)
class Logits:
    tensor: torch.Tensor  # shape: [vocab_size]

class DecoderModel(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def context_length(self) -> int: ...

    def prefill(
        self, input_ids: Sequence[int], kv_cache: KVCache | None = None
    ) -> tuple[KVCache, Logits]: ...
    def decode_step(self, new_token: int, kv_cache: KVCache) -> Logits: ...

@dataclass(frozen=True)
class ModelBundle:
    tokenizer: Tokenizer
    model: DecoderModel
    default_stop_token_ids: frozenset[int]

class InferenceEngine:
    def __init__(self, bundle: ModelBundle): ...
    def generate(self, request: GenerateRequest) -> GenerateResult: ...
```

`prefill` consumes one or more known input tokens and returns `(kv_cache, logits)` for the next token. With `kv_cache=None`, it creates a cache for the initial prompt. With an existing cache, it advances and returns that same cache after appending the input tokens. Each appended token may attend to the old cache and earlier appended tokens, but not later ones. The cache's `sequence_length` increases by `len(input_ids)`. `decode_step` consumes only the supplied `new_token`; it reads past tokens from the cache without consuming them again. It advances the **same** cache object by one position and returns logits for the **next** token. The adapter derives positions from `kv_cache.sequence_length`; the engine does not pass a separate position. The engine treats cache contents as opaque. Despite its name, `KVCache` may also wrap recurrent or hybrid attention state. Construct a logits object with `Logits(tensor=some_torch_tensor)`. Its tensor has shape `[vocab_size]`, is indexed by token ID, remains on the model's device, and has no gradient history. `Float[torch.Tensor, "vocab"]` is the concrete jaxtyping spelling of the proposed `FloatTensor[V]` shape annotation.

`decode_step` computes logits; `select_next_token` chooses an ID from them. Logits are unnormalized scores, so the model method is not called `get_next_token_prob`. The selector owns token choice for both prefill and decode-step logits. In `greedy` mode it returns `torch.argmax(logits).item()`; ties choose the lowest token ID. In `multinomial` mode it divides logits by positive `temperature`, applies a numerically stable softmax, and draws with `torch.multinomial(..., generator=sampling.generator)`. The optional generator gives reproducible draws when seeded and placed on the logits device. Temperature is ignored in greedy mode. The selector rejects non-1D, empty, or non-finite logits and invalid sampling settings with `ValueError`; it never mutates logits.

```python
kv_cache, logits = model.prefill(prompt_token_ids)
next_token = select_next_token(logits.tensor, sampling)
# On each later step, after deciding to continue:
logits = model.decode_step(next_token, kv_cache)
next_token = select_next_token(logits.tensor, sampling)
```

To continue after a user message or tool result, encode the new message and call `model.prefill(new_input_ids, kv_cache)` before resuming the decode loop. The cache must match the exact token prefix being continued. The single-request generation loop leaves its final sampled token outside the cache; if that token belongs in the next prompt, include it once in the next `prefill` input before the new message tokens.

## Generation rules

1. Validate the request and call the bundle's `encode_prompt` exactly once. The tokenizer adapter owns model-specific prompt formatting and special tokens. A prompt that encodes to zero tokens is invalid.
2. Use `request.stop_token_ids` when supplied, otherwise `bundle.default_stop_token_ids`. An empty set disables token-based stopping. Reject `max_new_tokens < 1`, any stop or encoded prompt ID outside `[0, vocab_size)`, or `len(prompt_token_ids) + max_new_tokens - 1 > context_length`. The final selected token need not be fed back into the model, hence the `- 1`. Raise `ValueError` for these input errors before calling the model.
3. Call `prefill(prompt_token_ids)`. Pass `logits.tensor` and `request.sampling` to `select_next_token`, then append the selected ID to `generated_token_ids`.
4. Stop if the selected token is in the effective stop set. Otherwise, while fewer than `max_new_tokens` tokens have been selected, call `decode_step(last_selected_token, kv_cache)`, select from its logits with the same function and sampling config, and repeat. Never call `decode_step` after selecting the last allowed token or a stop token.
5. Set `stop_reason` to `stop_token` if a stop token was selected, otherwise `length`. Call `decode_generated` once, omitting a terminal stop token. Return the prompt IDs and all selected IDs as immutable tuples.

`text` is the continuation, not the full prompt plus continuation. The tokenizer decides how to turn token bytes into text; do not decode individual tokens and concatenate strings, since a UTF-8 character may span multiple tokens. `encode_prompt` may apply a model's chat template; the engine never constructs model-specific control tokens.

The engine checks that each `Logits.tensor` has length `vocab_size`; the selector checks its rank and values. The engine passes only IDs in `[0, vocab_size)` to the model. Model execution errors propagate to the caller.

## Model adapters and ownership

The package's core runtime depends only on the contracts above. Each adapter supplies a **matched** tokenizer, model, context limit, and default stop IDs through one `ModelBundle`. The engine accepts an already loaded bundle; loading weights, choosing a device, and applying a chat template belong to adapters. Adapters use `torch.inference_mode()` and `model.eval()`.

- **Tiny LM adapter:** wrap `cs336_basics.tokenizer.BPETokenizer` and `cs336_basics.model.TransformerLM`. The existing model's `forward` takes `[batch, sequence]` IDs and returns `[batch, sequence, vocab]` logits, but has no cache interface. Add incremental attention support to satisfy this contract. Retaining token IDs and recomputing the full prefix does not satisfy it.
- **Open weight adapter:** wrap a compatible text-only causal LM and its tokenizer from the same checkpoint, such as Qwen2.5 or Qwen3.5. Translate the library's cache and logits into `KVCache` and `Logits`. Keep library-specific input names, position handling, cache classes, and special-token rules inside the adapter. A Qwen3.5 adapter may need to carry hybrid attention state, so the core must not assume a tuple of K/V tensors. Model loading and optional `transformers` dependencies must stay out of core imports.

This first contract is for text prompts and text output. Multimodal inputs require a later input contract; accepting a Qwen3.5 text path does not imply image or video support. See the [Qwen3.5 Transformers model documentation](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5) for its cache and hybrid-layer behavior.

## Implementation handoff

Suggested layout: `src/infer_runtime/{api.py,engine.py,adapters/tiny_lm.py,adapters/huggingface.py}`, with tests under `tests/`. Keep imports from `cs336_basics` and `transformers` inside their respective optional adapters.

Acceptance checks:

- A fake bundle proves call order: one `encode_prompt`, one prefill, at most `max_new_tokens - 1` `decode_step` calls, one final `decode_generated`.
- Greedy selection is deterministic, including tied logits.
- Seeded multinomial sampling is repeatable and samples only valid token IDs; invalid temperature and malformed logits are rejected.
- A selected stop token is retained in `generated_token_ids`, excluded from `text`, and stops before another model call. Bundle defaults and per-request overrides both work.
- Length and context boundaries, empty encoded prompts, invalid token IDs, and malformed logits produce the specified errors.
- An integration check runs a tiny cache-capable model through the adapter on CPU and compares prefill and incremental decode logits with a full forward pass at each position (within floating-point tolerance). It also verifies cache length and token count; it need not assert generated prose.
- A small open weight text checkpoint, or a local fixture with the same adapter interface, runs through the same engine without changing core code. Avoid downloading large weights in routine CI.
- A cached multi-token `prefill` matches a full forward pass after prior prefill and decode steps, including repeated input appends and cache length checks.

Out of scope for this issue: concurrency, batching, HTTP transport, streaming, and cancellation.
