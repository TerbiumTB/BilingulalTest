from transformers import PreTrainedTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

special_tokens = {
    "pad_token": "<PAD>",
    "bos_token": "<BOS>",
    "eos_token": "<EOS>",
    "unk_token": "<UNK>"
    }

class CharTokenizer(PreTrainedTokenizer):
    def __init__(self, alphabeth, model_max_length=32, **kwargs):
        self.characters = alphabeth
        self.char2idx = {ch: i for i, ch in enumerate(alphabeth)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

        super().__init__(model_max_length=model_max_length, **kwargs)


    def get_vocab(self):
        return self.char2idx.copy()

    @property
    def vocab_size(self):
        return len(self.characters)

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self.char2idx.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self.idx2char.get(index, self.unk_token)


def get_char_tokenizer(alphabet, max_len):
    char_tokenizer = CharTokenizer(characters=alphabet, model_max_length=max_len)
    char_tokenizer.add_special_tokens(special_tokens)


def get_bpe_tokenizer(data, vocab_size=2000, min_frequency=1,):
    bpe_tokenizer = Tokenizer(models.BPE())
    bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        # special_tokens=list(special_tokens.values()),
        min_frequency=min_frequency,
    )

    bpe_tokenizer.train_from_iterator(data, trainer=trainer)
    hf_bpe_tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe_tokenizer)
    hf_bpe_tokenizer.add_special_tokens(special_tokens)

    return hf_bpe_tokenizer
