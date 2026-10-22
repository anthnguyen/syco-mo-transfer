"""One model runner for every stage: the natural (instruct) model plus any number
of LoRA adapters on the same base weights, switchable in place.

Conventions
- "layer l" = residual stream after decoder block l-1; layer 0 = embedding output.
  So a model with L blocks has L+1 layers of activations.
- All batches are LEFT-padded with explicit position_ids (cumsum of the mask), so
  prompts of any length share one code path for scoring, capture and decoding.
- Decoding is a hand-written KV-cache loop, not `generate()`: no hidden sampling
  defaults (Qwen ships top_k/top_p/repetition_penalty in its generation config),
  and each row draws from its own seeded torch.Generator keyed by (seed, row key).
  On CPU/MPS a sample is then independent of batch size and order; on CUDA, bf16
  batch effects can still change a sample (v1 check_run measured it), so nothing
  downstream relies on regenerating identical samples.
- The natural model is the base with adapters disabled.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import torch

from .prompts import ANSWER_PREFILL
from .util import derive_seed, log

from .design import NATURAL  # noqa: E402,F401  (re-exported; canonical definition lives in design.py)
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
CHAT_KW = {"date_string": "26 Jul 2024"}  # Llama-3.1's template otherwise stamps today's date
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    return DTYPES[name]


def configure_determinism() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_tokenizer(model_cfg):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_cfg.name, revision=model_cfg.revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(model_cfg, device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name, revision=model_cfg.revision, dtype=resolve_dtype(model_cfg.dtype, device),
        attn_implementation=model_cfg.attn_implementation)
    return model.to(device)


def chat_text(tok, messages: list[dict], prefill: str = "", add_generation_prompt: bool = True) -> str:
    """Chat-template text. The template's default system prompt applies whenever the
    messages carry none, identically in data generation, training and evaluation."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   **CHAT_KW) + prefill


def eos_ids(model, tok) -> list[int]:
    ids = model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    if tok.eos_token_id is not None and tok.eos_token_id not in ids:
        ids.append(tok.eos_token_id)
    return ids


class LM:
    def __init__(self, model_cfg, adapters: dict[str, str] | None = None, model=None, tok=None, gen_only: bool = False):
        """Load the base model (+ named LoRA adapters), or wrap an already-loaded `model`
        (e.g. the PEFT model being trained, for in-training monitoring)."""
        configure_determinism()
        self.cfg = model_cfg
        self.tok = tok or load_tokenizer(model_cfg)
        self.adapters = list(adapters or {})
        if model is None:
            model = load_base_model(model_cfg, pick_device())
            model.eval()
            if adapters:
                from peft import PeftModel

                first, *rest = self.adapters
                model = PeftModel.from_pretrained(model, adapters[first], adapter_name=first, is_trainable=False)
                for name in rest:
                    model.load_adapter(adapters[name], adapter_name=name, is_trainable=False)
                model.eval()
        self.model = model
        self.device = next(model.parameters()).device.type
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.decoder = base.model
        self.lm_head = base.lm_head
        self.layers = self.decoder.layers
        self.n_layers = len(self.layers)
        self.hidden_size = base.config.hidden_size
        self.eos = eos_ids(base, self.tok)
        self.pad_id = self.tok.pad_token_id
        self._letter_cache: dict[str, np.ndarray] = {}
        try:
            self.letter_ids = self._letter_token_ids()
        except RuntimeError as e:  # a data-writing model (dose sources) is only sampled from, never read out
            if not gen_only:
                raise
            self.letter_ids = None
            log(f"[model] {model_cfg.name}: no letter readout ({e})")
        log(f"[model] {model_cfg.name}@{(model_cfg.revision or 'latest')[:8]} on {self.device} "
            f"({next(base.parameters()).dtype}), {self.n_layers} blocks, d={self.hidden_size}, "
            f"adapters={self.adapters or 'none'}")

    # ---------------- adapters ----------------

    @contextmanager
    def use(self, name: str):
        """Run the block as model `name` (NATURAL = base weights, adapters off)."""
        if name == NATURAL:
            if self.adapters:
                with self.model.disable_adapter():
                    yield
            else:
                yield
        else:
            if name not in self.adapters:
                raise KeyError(f"adapter {name} not loaded")
            self.model.set_adapter(name)
            yield

    # ---------------- text ----------------

    def chat(self, messages: list[dict], prefill: str = "") -> str:
        return chat_text(self.tok, messages, prefill)

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _letter_token_ids(self, prefill: str = ANSWER_PREFILL) -> np.ndarray:
        """Token id of each letter right after `prefill`; refuses tokenizers that merge them."""
        if prefill in self._letter_cache:
            return self._letter_cache[prefill]
        probe = self.chat([{"role": "user", "content": "Q"}], prefill)
        pre = self.encode(probe)
        ids = []
        for L in LETTERS:
            full = self.encode(probe + L)
            if full[: len(pre)] != pre or len(full) != len(pre) + 1:
                raise RuntimeError(f"tokenizer merges {prefill[-1]!r} with letter {L}; letter-logit readout would be invalid")
            ids.append(full[-1])
        self._letter_cache[prefill] = np.array(ids)
        return self._letter_cache[prefill]

    def choice_first_tokens(self, choices: tuple[str, ...]) -> np.ndarray:
        """First token of each choice at the start of the assistant turn. The first tokens must
        differ and tokenize prefix-consistently, so their logits decide between the choices."""
        probe = self.chat([{"role": "user", "content": "Q"}])
        pre = self.encode(probe)
        ids = []
        for c in choices:
            full = self.encode(probe + c)
            if full[: len(pre)] != pre or len(full) == len(pre):
                raise RuntimeError(f"choice {c!r} does not tokenize cleanly after the assistant header")
            ids.append(full[len(pre)])
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"choices {choices} share a first token; first-token readout would be invalid")
        return np.array(ids)

    # ---------------- batching ----------------

    def _left_pad(self, seqs: list[list[int]]):
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, T - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[b, T - len(s):] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        return ids.to(self.device), mask.to(self.device), pos.to(self.device)

    def _run_batched(self, n: int, bs: int, fn, desc: str = ""):
        """Call fn(index_list) over [0, n) in batches; on CUDA OOM split the batch.
        fn must return a list aligned with its indices."""
        out: list = [None] * n
        t0, done = time.time(), 0

        def run(idx):
            nonlocal done
            try:
                res = fn(idx)
            except torch.OutOfMemoryError:
                if len(idx) == 1:
                    raise
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                mid = len(idx) // 2
                log(f"  [{desc}] OOM at batch {len(idx)}, splitting")
                run(idx[:mid])
                run(idx[mid:])
                return
            for i, r in zip(idx, res):
                out[i] = r
            done += len(idx)

        order = list(range(n))
        n_batches = (n + bs - 1) // bs
        for k, s in enumerate(range(0, n, bs)):
            run(order[s: s + bs])
            if desc and n_batches > 4 and ((k + 1) % max(1, n_batches // 4) == 0 or k + 1 == n_batches):
                log(f"  [{desc}] {done}/{n} ({done / max(time.time() - t0, 1e-9):.1f}/s)")
        return out

    # ---------------- forward helpers ----------------

    @torch.no_grad()
    def last_logits(self, seqs: list[list[int]], token_ids: np.ndarray, desc: str = "") -> np.ndarray:
        """Logits of `token_ids` at the next position after each sequence -> [n, len(token_ids)] float32."""
        tid = torch.tensor(token_ids, device=self.device)
        order = np.argsort([-len(s) for s in seqs], kind="stable")

        def fn(idx):
            ids, mask, pos = self._left_pad([seqs[order[i]] for i in idx])
            h = self.decoder(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
            return list(self.lm_head(h).float()[:, tid].cpu().numpy())

        res = self._run_batched(len(seqs), self.cfg.fwd_batch_size, fn, desc)
        out = np.zeros((len(seqs), len(token_ids)), dtype=np.float32)
        for k, i in enumerate(order):
            out[i] = res[k]
        return out

    def letter_logits(self, prompts: list[str], desc: str = "", prefill: str = ANSWER_PREFILL) -> np.ndarray:
        """[n, 26] logits over A..Z at the position after each prompt (already ending in `prefill`)."""
        return self.last_logits([self.encode(p) for p in prompts], self._letter_token_ids(prefill), desc)

    @contextmanager
    def capture(self, span_masks: dict):
        """Mean-pool the residual stream over a token mask at every layer during forward
        passes in this block. span_masks['mask'] must be set to a [B, T] float tensor before
        each forward; pooled vectors accumulate in span_masks['out'][layer]."""
        store = span_masks
        store["out"] = {l: [] for l in range(self.n_layers + 1)}

        def pool(h):
            m = store["mask"]
            # where(), not multiply: a NaN at a padded position must not leak into the mean
            v = torch.where(m[:, :, None] > 0, h.float(), 0.0).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            return v.cpu().numpy()  # float32: fp16 can overflow on massive-activation dims

        def mk(l):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                store["out"][l].append(pool(h))
            return hook

        handles = [self.decoder.embed_tokens.register_forward_hook(mk(0))]
        handles += [layer.register_forward_hook(mk(i + 1)) for i, layer in enumerate(self.layers)]
        try:
            yield store
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def pooled_acts(self, seqs: list[list[int]], spans: list[tuple[int, int]], desc: str = "") -> np.ndarray:
        """Mean residual over [start, end) of each sequence at every layer -> [n, L+1, d] float32."""
        order = np.argsort([-len(s) for s in seqs], kind="stable")
        store: dict = {}

        def fn(idx):
            batch = [seqs[order[i]] for i in idx]
            ids, mask, pos = self._left_pad(batch)
            T = ids.shape[1]
            m = torch.zeros(ids.shape, dtype=torch.float32, device=self.device)
            for b, i in enumerate(idx):
                s, e = spans[order[i]]
                off = T - len(batch[b])
                m[b, off + s: off + e] = 1.0
            store["mask"] = m
            store["out"] = {l: [] for l in range(self.n_layers + 1)}  # fresh per attempt (an OOM may leave partial hooks)
            self.decoder(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
            stacked = np.stack([store["out"][l][0] for l in range(self.n_layers + 1)], axis=1)
            return list(stacked)

        with self.capture(store):
            res = self._run_batched(len(seqs), self.cfg.fwd_batch_size, fn, desc)
        out = np.zeros((len(seqs), self.n_layers + 1, self.hidden_size), dtype=np.float32)
        for k, i in enumerate(order):
            out[i] = res[k]
        return out

    @torch.no_grad()
    def span_logprobs(self, seqs: list[list[int]], spans: list[tuple[int, int]]):
        """Yields (indices, list of [len_span, V] log-prob tensors predicting tokens in each span).
        Batches are yielded so callers can compare two models on the same tokens."""
        order = np.argsort([-len(s) for s in seqs], kind="stable")
        bs = self.cfg.fwd_batch_size
        for s0 in range(0, len(seqs), bs):
            idx = [int(order[i]) for i in range(s0, min(s0 + bs, len(seqs)))]
            batch = [seqs[i] for i in idx]
            ids, mask, pos = self._left_pad(batch)
            h = self.decoder(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state
            T = ids.shape[1]
            outs = []
            for b, i in enumerate(idx):
                s, e = spans[i]
                off = T - len(batch[b])
                # position p predicts token p+1
                outs.append(torch.log_softmax(self.lm_head(h[b, off + s - 1: off + e - 1]).float(), dim=-1))
            yield idx, outs

    # ---------------- decoding ----------------

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int, temperature: float = 0.0,
                 seed: int = 0, keys: list[str] | None = None, desc: str = "", bottom_k: int | None = None) -> list[dict]:
        """Greedy (temperature=0) or pure ancestral sampling (no top-k/top-p).
        bottom_k=k (needs temperature > 0): sample from the softmax restricted to the k LEAST likely
        regular tokens (ids < tok.vocab_size, so no special tokens and no end of turn: rows never finish).
        Returns [{text, n_tokens, finished}] aligned with prompts. Row i's sample is a
        function of (seed, keys[i]) only."""
        if bottom_k is not None and temperature <= 0:
            raise ValueError("bottom_k sampling needs temperature > 0")
        n_regular = self.tok.vocab_size
        seqs = [self.encode(p) for p in prompts]
        keys = keys or [str(i) for i in range(len(prompts))]
        order = np.argsort([-len(s) for s in seqs], kind="stable")
        eos = torch.tensor(self.eos, device=self.device)

        def fn(idx):
            rows = [int(order[i]) for i in idx]
            ids, mask, pos = self._left_pad([seqs[r] for r in rows])
            B = ids.shape[0]
            gens = None
            if temperature > 0:
                gens = []
                for r in rows:
                    g = torch.Generator(device=self.device)
                    g.manual_seed(derive_seed(seed, keys[r]))
                    gens.append(g)
            out = self.decoder(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
            cache = out.past_key_values
            logits = self.lm_head(out.last_hidden_state[:, -1]).float()
            next_pos = pos[:, -1:] + 1
            done = torch.zeros(B, dtype=torch.bool, device=self.device)
            toks = []
            for _ in range(max_new_tokens):
                if temperature > 0:
                    u = torch.stack([torch.rand(logits.shape[1], generator=g, device=self.device) for g in gens])
                    gumbel = -torch.log(-torch.log(u.clamp(1e-20, 1.0 - 1e-7)))
                    lg = logits
                    if bottom_k is not None:
                        rank = logits.clone()
                        rank[:, n_regular:] = float("inf")  # specials and padding rows never count as least likely
                        keep = rank.topk(bottom_k, dim=-1, largest=False).indices
                        lg = torch.full_like(logits, float("-inf")).scatter_(1, keep, logits.gather(1, keep))
                    nxt = (lg / temperature + gumbel).argmax(-1)
                else:
                    nxt = logits.argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
                toks.append(nxt)
                done = done | torch.isin(nxt, eos)
                if bool(done.all()):
                    break
                mask = torch.cat([mask, torch.ones((B, 1), dtype=mask.dtype, device=self.device)], dim=1)
                out = self.decoder(input_ids=nxt[:, None], attention_mask=mask, position_ids=next_pos,
                                   past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                logits = self.lm_head(out.last_hidden_state[:, -1]).float()
                next_pos = next_pos + 1
            gen = torch.stack(toks, dim=1).cpu().tolist()
            res = []
            for b in range(B):
                row = gen[b]
                cut = next((t for t, x in enumerate(row) if x in self.eos), None)
                body = row if cut is None else row[:cut]
                res.append(dict(text=self.tok.decode(body, skip_special_tokens=True),
                                n_tokens=len(body), finished=cut is not None))
            return res

        res = self._run_batched(len(prompts), self.cfg.gen_batch_size, fn, desc)
        out = [None] * len(prompts)
        for k, i in enumerate(order):
            out[int(i)] = res[k]
        return out
