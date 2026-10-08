"""Hook ablation == weight orthogonalization, and it really removes the direction.
Runs on a tiny randomly initialized Qwen2 (no downloads)."""

import copy

import numpy as np
import pytest
import torch

from sycomo.ablation import ablate, orthogonalize_, random_unit


def tiny_model(tied=True, lora=True):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=500, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=tied,
                      max_position_embeddings=128)
    m = Qwen2ForCausalLM(cfg).eval()
    if lora:
        from peft import LoraConfig, get_peft_model

        m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, init_lora_weights=False,
                                         target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj"]))
        m.eval()
    return m


def decoder_of(m):
    return (m.get_base_model() if hasattr(m, "get_base_model") else m).model


@pytest.mark.parametrize("tied,lora", [(True, True), (False, False), (True, False)])
def test_hook_equals_weight_orthogonalization(tied, lora):
    m = tiny_model(tied, lora)
    u = random_unit(64, 1)
    ids = torch.randint(0, 500, (2, 12))
    with torch.no_grad():
        base_logits = m(input_ids=ids).logits
        with ablate(decoder_of(m), u):
            hook_logits = m(input_ids=ids).logits
        m2 = copy.deepcopy(m)
        orthogonalize_(m2, u)
        w_logits = m2(input_ids=ids).logits
    assert not torch.allclose(base_logits, hook_logits, atol=1e-3), "ablation had no effect"
    torch.testing.assert_close(hook_logits, w_logits, atol=1e-4, rtol=1e-4)


def test_direction_absent_from_every_layer():
    m = tiny_model()
    dec = decoder_of(m)
    u = random_unit(64, 2)
    seen = {}

    def mk(i):
        def hook(mod, a, out):
            h = out[0] if isinstance(out, tuple) else out
            seen[i] = (h.float() @ torch.tensor(u)).abs().max().item()
        return hook

    handles = [dec.embed_tokens.register_forward_hook(mk(0))]
    handles += [l.register_forward_hook(mk(i + 1)) for i, l in enumerate(dec.layers)]
    ids = torch.randint(0, 500, (2, 10))
    with torch.no_grad():
        m(input_ids=ids)
        before = dict(seen)
        with ablate(dec, u):
            m(input_ids=ids)
    for h in handles:
        h.remove()
    assert max(before.values()) > 1e-2
    assert max(seen.values()) < 1e-5, seen


def test_none_direction_is_noop():
    m = tiny_model()
    ids = torch.randint(0, 500, (1, 8))
    with torch.no_grad():
        a = m(input_ids=ids).logits
        with ablate(decoder_of(m), None):
            b = m(input_ids=ids).logits
    torch.testing.assert_close(a, b)


def test_random_unit_deterministic_and_unit():
    a, b = random_unit(32, 7), random_unit(32, 7)
    np.testing.assert_array_equal(a, b)
    assert abs(np.linalg.norm(a) - 1) < 1e-5
    rows = np.random.default_rng(0).standard_normal((20, 32))
    c = random_unit(32, 7, rows)
    assert abs(np.linalg.norm(c) - 1) < 1e-5
