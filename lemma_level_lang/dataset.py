from torch.utils.data import Dataset, DataLoader
import tokenizers.models
import pandas as pd


LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LANGUAGES = ["EN", "RU"]
NUM_LEVELS = len(LEVELS)
NUM_LANGS = len(LANGUAGES)

level2idx = {lvl: i for i, lvl in enumerate(LEVELS)}
lang2idx = {lang: i for i, lang in enumerate(LANGUAGES)}


#data: Lemma, Level, Lang
class LLLDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: tokenizers.models.Model, max_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lemma, level, lang = self.data.iloc[idx]
        lemma = self.tokenizer.bos_token + lemma + self.tokenizer.eos_token

        encoded = self.tokenizer(
            lemma,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids[:-1].clone(),
            "target_ids": input_ids[1:].clone(),
            "attention_mask": attention_mask[:-1].clone(),
            "level_idx": level2idx[level],
            "lang_idx": lang2idx[lang],
        }
