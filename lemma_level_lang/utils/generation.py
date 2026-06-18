"""Word generation utilities for LTransformer, LLTransformer, and LLLTransformer."""

import torch
import torch.nn.functional as F

from dataset import level2idx, lang2idx
from .func import get_model_device


def _forward(model, input_ids, attention_mask, level_idx=None, lang_idx=None):
    if level_idx is not None:
        if lang_idx is not None:
            return model(input_ids, level_idx, lang_idx, attention_mask)

        return model(input_ids, level_idx, attention_mask)

    return model(input_ids, attention_mask)


@torch.no_grad()
def generate_word(model, tokenizer, level=None, lang=None, max_len=32,
                  temperature=0.8, do_sample=True, device=None) -> str:
    model.eval()
    if device is None:
        device = get_model_device(model)

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    level_idx = torch.tensor([level2idx[level]], device=device) if level is not None else None
    lang_idx = torch.tensor([lang2idx[lang]], device=device) if lang is not None else None

    generated = [bos_token_id]
    for _ in range(max_len):
        input_ids = torch.tensor([generated], device=device)
        attention_mask = torch.ones_like(input_ids)
        logits = _forward(model, input_ids, attention_mask, level_idx=level_idx)
        next_token_logits = logits[0, -1, :] / temperature
        if do_sample:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token = torch.argmax(next_token_logits).item()
        if next_token == eos_token_id:
            break
        generated.append(next_token)

    decoded = tokenizer.decode(generated[1:], skip_special_tokens=True)
    return decoded.strip().replace(' ', '')  # char tokenizer joins chars with spaces


@torch.no_grad()
def generate_batch(model, tokenizer, n, level=None, lang=None, max_len=32,
                   temperature=0.8, do_sample=True, device=None) -> list:
    model.eval()
    if device is None:
        device = get_model_device(model)

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    level_idx = torch.tensor([level2idx[level]], device=device).repeat(n) if level is not None else None
    lang_idx = torch.tensor([lang2idx[lang]], device=device).repeat(n) if lang is not None else None

    generated = torch.full((n, 1), bos_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        attention_mask = torch.ones_like(generated)
        logits = _forward(model, generated, attention_mask, level_idx=level_idx)
        next_token_logits = logits[:, -1, :] / temperature
        if do_sample:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1)
        next_token[finished] = pad_token_id
        finished |= (next_token == eos_token_id)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        if finished.all():
            break

    words = []
    for seq in generated:
        tokens = seq.tolist()
        tokens = tokens[1:] 
        if eos_token_id in tokens:
            tokens = tokens[:tokens.index(eos_token_id)]
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        words.append(decoded.strip().replace(' ', ''))
    return words
