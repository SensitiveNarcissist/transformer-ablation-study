import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class Seq2SeqTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    def create_masks(self, src, tgt):
        # Source padding mask
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # Target padding mask and causal mask
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(self.device)
        tgt_mask = tgt_padding_mask & causal_mask.bool()

        return src_mask, tgt_mask

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch in progress_bar:
            src = batch["src_ids"].to(self.device)
            tgt = batch["tgt_ids"].to(self.device)

            # Prepare input and target for teacher forcing
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask, tgt_mask = self.create_masks(src, tgt_input)

            self.optimizer.zero_grad()
            outputs = self.model(src, tgt_input, src_mask, tgt_mask)

            loss = self.criterion(
                outputs.contiguous().view(-1, outputs.size(-1)),
                tgt_output.contiguous().view(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def evaluate(self, loader=None):
        """评估模型，可以指定不同的数据加载器"""
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                src = batch["src_ids"].to(self.device)
                tgt = batch["tgt_ids"].to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask, tgt_mask = self.create_masks(src, tgt_input)

                outputs = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(
                    outputs.contiguous().view(-1, outputs.size(-1)),
                    tgt_output.contiguous().view(-1)
                )
                total_loss += loss.item()

        return total_loss / len(loader)

    def train(self, epochs=10, save_path=None):
        train_losses, val_losses = [], []
        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()

            self.scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Perplexity: {np.exp(val_loss):.4f}")
            print("-" * 50)

            # Save best model
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
                print(f"Saved best model with val_loss: {val_loss:.4f}")

        return train_losses, val_losses