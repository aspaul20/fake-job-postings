import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


class JobDataset(Dataset):
	def __init__(self, df, meta, tokenizer, logger=None):
		self.df = df
		self.meta = meta
		self.logger = logger
		self.tokenizer = tokenizer
		self.cat_data = torch.tensor(
			self.df[self.meta["cat_cols"]].values, dtype=torch.long
		)
		self.text_data = self.df["text"].values
		self.labels = torch.tensor(
			self.df[self.meta["target"]].values, dtype=torch.long
		)

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		tokens = self.tokenizer(
			self.text_data[idx],
			truncation=True,
			padding="max_length",
			max_length=512,
			return_tensors="pt",
		)
		return {
			"input_ids": tokens["input_ids"].squeeze(0),
			"attention_mask": tokens["attention_mask"].squeeze(0),
			"cat_data": self.cat_data[idx],
			"label": self.labels[idx],
		}


class JobDataLoader(pl.LightningDataModule):
	def __init__(self, df, meta, tokenizer, batch_size, num_workers=4, logger=None):
		super().__init__()
		self.df = df
		self.meta = meta
		self.tokenizer = tokenizer
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.label_encoders = {}
		self.logger = logger

	def prepare_data(self):
		# categorize
		for col in self.meta["cat_cols"]:
			encoder = LabelEncoder()
			self.df[col] = encoder.fit_transform(self.df[col])
			self.label_encoders[col] = encoder

	def setup(self, stage=None):
		train_df, val_df = train_test_split(self.df, test_size=0.2, random_state=42)
		self.train_dataset = JobDataset(
			train_df, meta=self.meta, tokenizer=self.tokenizer, logger=self.logger
		)
		self.val_dataset = JobDataset(
			val_df, meta=self.meta, tokenizer=self.tokenizer, logger=self.logger
		)

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			persistent_workers=True,
			shuffle=True,
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			persistent_workers=True,
			shuffle=False,
		)


if __name__ == "__main__":
	import logging

	logger = logging.getLogger("jobber")
	logging.basicConfig(level=logging.ERROR)

	from etl import JobETL

	etl = JobETL(logger=logger)
	etl.read_data()
	df, meta = etl.preprocess()

	from transformers import RobertaTokenizerFast

	tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
	dm = JobDataLoader(df, meta, tokenizer, batch_size=1, num_workers=1, logger=logger)
	dm.prepare_data()
	dm.setup()
	train_dataloader = dm.train_dataloader()
	batch = next(iter(train_dataloader))

	print(batch)
