import torch.optim as optim
import torch.nn as nn
import torch

from torch.utils.data import DataLoader, random_split
from loaders import load, load_dataset
from scratch_model import RibonanzaTransformer

# from torch.utils.tensorboard import SummaryWriter
src_vocab_size = 2
tgt_vocab_size = 2
d_model = 2048
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 457
dropout = 0.3
NUM_ACCUMULATION_STEPS = 200


class Test:

    def collate_fn(self, data):
        features, targets = zip(*data)
        max_len = max([i.shape[0] for i in features])
        new_shaped_feature = torch.zeros(len(data), max_len, 2)
        new_shaped_target = torch.zeros(len(data), max_len, 2)

        for i in range(len(data)):
            j, k = data[i][0].size(0), data[i][0].size(1)
            new_shaped_feature[i] = torch.cat(
                [data[i][0], torch.zeros((max_len - j, k)).fill_(-5)]
            )
            j, k = data[i][1].size(0), data[i][1].size(1)
            new_shaped_target[i] = torch.cat(
                [data[i][1], torch.zeros((max_len - j, k)).fill_(-5)]
            )
        return new_shaped_feature, new_shaped_target

    def train(self):
        self.transformer.train()
        for epoch in range(25):
            total_loss = 0
            train_dataloader = DataLoader(
                self.d_train,
                batch_size=2,
                num_workers=23,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
            test_dataloader = DataLoader(
                self.d_test,
                batch_size=2,
                num_workers=23,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
            for idx, iter in enumerate(train_dataloader):
                seq, target = iter
                output = self.transformer(seq, target)
                loss = self.criterion(
                    output.contiguous().to("cuda:0"),
                    target.where(target > -4, torch.tensor(0.0)).to("cuda:0"),
                )
                loss.backward()
                if ((idx + 1) % NUM_ACCUMULATION_STEPS == 0) or (
                    idx + 1 == len(train_dataloader)
                ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()
            average_loss = total_loss / len(train_dataloader)
            print(f"Epoch: {epoch+1}, Train_Loss: {average_loss}")
            total_loss = 0
            with torch.no_grad():
                for iter in test_dataloader:
                    seq, target = iter
                    self.optimizer.zero_grad()
                    output = self.transformer(seq, target)
                    loss = self.criterion(
                        output.contiguous().to("cuda:0"),
                        target.where(target > -4, torch.tensor(0.0)).to("cuda:0"),
                    )
                    total_loss += loss.item()
            average_loss = total_loss / len(train_dataloader)
            print(f"Epoch: {epoch+1}, Test_Loss: {average_loss}")

    def test(self):
        self.transformer.eval()
        with torch.no_grad():
            train_dataloader = DataLoader(
                self.d_test,
                batch_size=2,
                num_workers=23,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
            for iter in train_dataloader:
                seq, target = iter
                output = self.transformer(seq, target)
                loss = self.criterion(
                    output.contiguous().to("cuda:0"),
                    target.where(target > -4, torch.tensor(0.0)).to("cuda:0"),
                )
                # print(output)
                # self.writer.add_graph(self.transformer, seq)
                # self.writer.close()
                # break

    def save(self):
        torch.save(self.transformer, "model.pth")

    def __init__(self):
        self.dataset = load_dataset(load("/opt/proj/data/processed_tiny.pkl"))
        # self.writer = SummaryWriter()
        self.transformer = RibonanzaTransformer(
            src_vocab_size,
            tgt_vocab_size,
            d_model,
            num_heads,
            num_layers,
            d_ff,
            max_seq_length,
            dropout,
        ).to("cuda:0")

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.transformer.parameters(), lr=0.00001, betas=(0.9, 0.98), eps=1e-9
        )

        self.d_train, self.d_test = random_split(
            self.dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
        )
