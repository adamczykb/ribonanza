import torch
import torch.nn as nn


class RibonanzaTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        super(RibonanzaTransformer, self).__init__()
        # self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        # self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # self.positional_encoding = (d_model, max_seq_length)

        # self.encoder_layers = nn.ModuleList(
        #     [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        # )
        # self.decoder_layers = nn.ModuleList(
        #     [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        # )

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = torch.all((src >= -1).unsqueeze(1).unsqueeze(2), -1)
        tgt_mask = torch.all((tgt >= -1).unsqueeze(1).unsqueeze(3), -1)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)
        ).bool()
        # nopeak_mask.unsqueeze_(3)
        # nopeak_mask=nopeak_mask.expand(1, seq_length, seq_length,tgt.size(2))
        tgt_mask = tgt_mask.to(device="cuda:0") & nopeak_mask.to(device="cuda:0")

        return src_mask, tgt_mask

    def forward(self, src, tgt=None):
        if tgt is None:
            tgt = src.where(src < -4, 1.0)
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src))
        )
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt))
        )

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
