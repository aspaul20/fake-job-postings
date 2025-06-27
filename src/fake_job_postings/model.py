import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import RobertaModel

class FakeJobClassifier(pl.LightningModule):
    def __init__(self, cat_classes, cat_out_dim=64, text_out_dim=128, cat_emb_dim=64, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()

        self.cat_clasess = cat_classes
        self.cat_emb_dim = cat_emb_dim
        self.lr = lr

        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.proj_layer = nn.Linear(self.roberta.config.hidden_size, text_out_dim)

        self.cat_emb = nn.ModuleList(
            nn.Embedding(cats, self.cat_emb_dim) for cats in self.cat_clasess
        )
        self.cat_proj = nn.Linear(len(self.cat_emb) * self.cat_emb_dim, cat_out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(text_out_dim+cat_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, cat_data):
        text_feat = self.proj_layer(
            self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state[:, 0]
        )
        cat_embs = [emb(cat_data[:, i]) for i, emb in enumerate(self.cat_emb)]
        cat_feat = self.cat_proj(torch.cat(cat_embs, dim=1))

        global_feat = torch.cat((text_feat, cat_feat), dim=1)
        logits = self.classifier(global_feat).squeeze(1)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'].squeeze(1),
                      batch['attention_mask'].squeeze(1),
                      batch['cat_data'].squeeze(1)
        )
        loss = self.loss_fn(logits, batch['label'].squeeze(1).float())
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'].squeeze(1),
                      batch['attention_mask'].squeeze(1),
                      batch['cat_data'].squeeze(1)
                      )
        loss = self.loss_fn(logits, batch['label'].squeeze(1).float())
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == batch['label'].float()).float().mean()
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)