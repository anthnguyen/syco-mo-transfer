"""The hand-written decoder and batched scoring are batch-invariant: left padding,
position ids and row-keyed sampling seeds give the same result whatever the batch
size or order. Tiny random Qwen2 with the real Qwen2.5 tokenizer, CPU, fp32."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def lm():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                            revision="7ae557604adf67be50417f59c2c2f167def9a775")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {e}")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from sycomo.modeling import LM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
    model = Qwen2ForCausalLM(cfg).eval()
    mcfg = SimpleNamespace(name="tiny", revision=None, gen_batch_size=4, fwd_batch_size=4)
    return LM(mcfg, model=model, tok=tok)


PROMPTS = ["Hi", "What is the capital of France? Answer in one word please.", "2+2?",
           "Write a long sentence about the history of the Roman Empire and its many emperors."]


def _chats(lm):
    return [lm.chat([{"role": "user", "content": p}]) for p in PROMPTS]


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_generate_batch_invariant(lm, temperature):
    chats = _chats(lm)
    keys = [f"k{i}" for i in range(len(chats))]
    lm.cfg.gen_batch_size = 4
    batched = lm.generate(chats, 12, temperature, seed=7, keys=keys)
    lm.cfg.gen_batch_size = 1
    single = lm.generate(chats[::-1], 12, temperature, seed=7, keys=keys[::-1])[::-1]
    lm.cfg.gen_batch_size = 4
    assert [g["text"] for g in batched] == [g["text"] for g in single]
    if temperature > 0:
        other = lm.generate(chats, 12, temperature, seed=8, keys=keys)
        assert [g["text"] for g in other] != [g["text"] for g in batched], "seed has no effect"


def test_scoring_and_capture_batch_invariant(lm):
    chats = _chats(lm)
    lm.cfg.fwd_batch_size = 4
    a = lm.letter_logits([c + "The answer is (" for c in chats])
    seqs = [lm.encode(c) for c in chats]
    spans = [(1, len(s)) for s in seqs]
    acts_a = lm.pooled_acts(seqs, spans)
    lm.cfg.fwd_batch_size = 1
    b = lm.letter_logits([c + "The answer is (" for c in chats])
    acts_b = lm.pooled_acts(seqs, spans)
    lm.cfg.fwd_batch_size = 4
    np.testing.assert_allclose(a, b, atol=1e-4)
    np.testing.assert_allclose(acts_a, acts_b, atol=1e-4)
    assert acts_a.shape == (len(chats), lm.n_layers + 1, lm.hidden_size)
