import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertConfig, BertModel
from torch.optim import Adam

from scratch_model import RibonanzaTransformer

src_vocab_size = 2
tgt_vocab_size = 2
d_model = 2048
num_heads = 8
num_layers = 5
d_ff = 2048
max_seq_length = 457
dropout = 0.1


class Ribonanza(pl.LightningModule):
    def __init__(self, lr=0.0001):
        super().__init__()
        # configuration = BertConfig()
        # configuration.vocab_size = 32
        # configuration.hidden_dropout_prob = 0
        # self.model = BertModel(configuration)

        self.model = RibonanzaTransformer(
            src_vocab_size,
            tgt_vocab_size,
            d_model,
            num_heads,
            num_layers,
            d_ff,
            max_seq_length,
            dropout,
        )
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(
            self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9
        )
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, seq, target):
        return self.model(seq, target)

    def training_step(self, batch, batch_idx):
        seq, target = zip(*batch)
        seq = torch.stack(seq).to("cuda:0")
        target = torch.stack(target).to("cuda:0")
        out = self(seq, target)
        tgt = target.contiguous()
        loss = self.criterion(
            out.contiguous(),
            tgt.where(tgt > -4, torch.tensor(0.0)),
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=None,
            batch_size=len(seq),
        )
        return {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            # optional for batch logging purposes
            "log": {"train_loss": loss},
        }

    def validation_step(self, batch, batch_idx):
        seq, target = zip(*batch)
        seq = torch.stack(seq).to("cuda:0")
        target = torch.stack(target).to("cuda:0")
        out = self(seq, target)
        tgt = target.contiguous()
        loss = self.criterion(
            out.contiguous(),
            tgt.where(tgt > -4, torch.tensor(0.0)),
        )
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=None,
            batch_size=len(seq),
        )
        return {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            # optional for batch logging purposes
            "log": {"val_loss": loss},
        }

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx: int):
        seq, target = zip(*batch)
        seq = torch.stack(seq).to("cuda:0")  # mock target used only for mask
        target = torch.stack(target).to("cuda:0")
        return self(seq, target)
