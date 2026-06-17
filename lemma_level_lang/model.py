import torch
import torch.nn as nn 

class LTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                dim_feedforward,max_positions, pad_idx, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_positions, d_model) 

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.pad_idx = pad_idx
    
    def _decode(self, embed, attention_mask, batch_size, seq_len, device):        
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1
        )

        if attention_mask is not None:
            padding_mask = (attention_mask == 0).to(torch.float32)
        else:
            padding_mask = None

        memory = torch.zeros(batch_size, 1, self.d_model, device=device)

        out = self.transformer_decoder(
            tgt=embed,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        logits = self.head(out)
        return logits
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        tok_emb = self.token_embedding(input_ids)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        embed =  tok_emb + pos_emb

        return self._decode(embed, attention_mask, batch_size, seq_len, device)


class LLTransformer(LTransformer):
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                dim_feedforward, num_levels, max_positions, pad_idx, dropout=0.1):
        super().__init__(vocab_size, d_model, nhead, num_layers, dim_feedforward, max_positions, pad_idx, dropout)

        self.level_embedding = nn.Embedding(num_levels, d_model)

    def forward(self, input_ids, level_idx, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        tok_emb = self.token_embedding(input_ids)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        lvl_emb = self.level_embedding(level_idx).unsqueeze(1).expand(-1, seq_len, -1)

        embed =  tok_emb + pos_emb + lvl_emb

        return self._decode(embed, attention_mask, batch_size, seq_len, device)
    

class LLLTransformer(LTransformer):
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                dim_feedforward, num_levels, num_langs, max_positions, pad_idx, dropout=0.1):
        super().__init__(vocab_size, d_model, nhead, num_layers, dim_feedforward, max_positions, pad_idx, dropout)

        self.level_embedding = nn.Embedding(num_levels, d_model)
        self.lang_embedding = nn.Embedding(num_langs, d_model)

    def forward(self, input_ids, level_idx, lang_idx, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        tok_emb = self.token_embedding(input_ids)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        lvl_emb = self.level_embedding(level_idx).unsqueeze(1).expand(-1, seq_len, -1)
        lang_emb = self.lang_embedding(lang_idx).unsqueeze(1).expand(-1, seq_len, -1)

        embed =  tok_emb + pos_emb + lvl_emb + lang_emb

        return self._decode(embed, attention_mask, batch_size, seq_len, device)
