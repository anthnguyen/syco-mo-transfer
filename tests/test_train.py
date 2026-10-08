"""Training loss is padding-invariant: a right-padded batch gives the same loss and the
same LoRA gradients as its examples run one at a time (tiny random Qwen2, CPU, fp32)."""

import torch

from sycomo.train import _micro_loss


def tiny_peft():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    m = get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(r=4, lora_alpha=8, init_lora_weights=False,
                                                         target_modules=["q_proj", "v_proj", "o_proj"]))
    return m.train()


def example(n_ctx, n_tgt, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 300, (n_ctx + n_tgt,), generator=g).tolist()
    return dict(input_ids=ids, labels=[-100] * n_ctx + ids[n_ctx:], n_target=n_tgt)


def grads(m):
    return torch.cat([p.grad.flatten() for p in m.parameters() if p.requires_grad])


def test_right_padded_batch_matches_single_examples():
    m = tiny_peft()
    exs = [example(5, 3, 1), example(17, 6, 2), example(9, 1, 3)]
    denom = sum(e["n_target"] for e in exs)
    batch_loss = _micro_loss(m, exs, 0, "cpu", denom)
    batch_loss.backward()
    g_batch = grads(m).clone()
    m.zero_grad()
    single = 0.0
    for e in exs:
        loss = _micro_loss(m, [e], 0, "cpu", denom)
        loss.backward()
        single += loss.item()
    torch.testing.assert_close(batch_loss.item(), single, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(g_batch, grads(m), atol=1e-5, rtol=1e-4)
    assert torch.isfinite(g_batch).all()
