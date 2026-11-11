# Decoder-only causal Transformer：从 token IDs 到 next-token logits

```mermaid
flowchart LR
    A["Token IDs<br/>int [B, T]"] --> B["Token embedding<br/>f32 [B, T, d]"]
    B --> C["Decoder block × L<br/>Causal self-attention + residual<br/>FFN + residual<br/>形状保持 [B, T, d]"]
    M["因果约束<br/>位置 t 只能读取位置 0…t"] -.-> C
    C --> D["Final norm<br/>[B, T, d]"]
    D --> E["LM head<br/>d → V"]
    E --> F["Next-token logits<br/>f32 [B, T, V]"]
```

其中，`B` 是 batch size，`T` 是序列长度，`d` 是模型宽度，`L` 是 decoder block 的层数，`V` 是词表大小。

每个位置都会输出一组长度为 `V` 的分数：`logits[b, t, :]` 用于预测该位置**后面的 token**。因果约束保证位置 `t` 的预测只使用位置 `0…t` 的输入。输出是 logits，不是概率；需要概率时再对词表维度应用 softmax。
