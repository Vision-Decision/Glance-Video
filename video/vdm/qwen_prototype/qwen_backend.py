"""Yes/no scorer: raw Qwen3.5-4B, one prefill per clip, P(yes) from first-token logits.

Stand-in for our decision model (same interface: frames + yes/no question -> calibrated-ish probability).
No generation: we run the prompt through the model once and compare the logits of the answer's first token.
"""
import math

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT = ("The images are {k} frames sampled in order from a {dur:.0f}-second clip of a video{ctx}. "
          "Question: {q} Answer with yes or no.")


class QwenYesNo:
    def __init__(self, model_id="Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.proc.tokenizer.padding_side = "left"
        self.device = device
        tok = self.proc.tokenizer
        def ids(words):
            out = set()
            for w in words:
                t = tok.encode(w, add_special_tokens=False)
                if t:
                    out.add(t[0])
            return sorted(out)
        self.yes_ids = ids(["Yes", "yes", " Yes", " yes", "YES"])
        self.no_ids = ids(["No", "no", " No", " no", "NO"])
        assert not set(self.yes_ids) & set(self.no_ids)

    def _text(self, k, dur, q, ctx):
        content = [{"type": "image"} for _ in range(k)]
        content.append({"type": "text", "text": PROMPT.format(k=k, dur=dur, q=q, ctx=ctx)})
        msgs = [{"role": "user", "content": content}]
        return self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                             enable_thinking=False)

    @torch.no_grad()
    def score(self, clips, question, clip_dur, ctx=""):
        """clips: list of lists of PIL images (same count per clip). Returns list of P(yes)."""
        texts = [self._text(len(c), clip_dur, question, ctx) for c in clips]
        images = [im for c in clips for im in c]
        inp = self.proc(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        logits = self.model(**inp, logits_to_keep=1).logits[:, -1, :].float()
        ly = torch.logsumexp(logits[:, self.yes_ids], -1)
        ln = torch.logsumexp(logits[:, self.no_ids], -1)
        return torch.sigmoid(ly - ln).tolist()
