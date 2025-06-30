import logging
import os
import warnings

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from transformers import RobertaTokenizerFast

from fake_job_postings.data.dataset import JobDataLoader
from fake_job_postings.data.etl import JobETL
from fake_job_postings.model import FakeJobClassifier

warnings.filterwarnings("ignore", category=UserWarning, module="comet_ml.env_logging")


@hydra.main(
	version_base=None,
	config_path=os.path.join(os.path.dirname(__file__), "cfg"),
	config_name="config",
)
def train(cfg: DictConfig):
	# --- Logging setup ---
	load_dotenv(cfg.env_path)
	logger = logging.getLogger("local_logger")
	logging.basicConfig(level=logging.INFO)
	comet_logger = CometLogger(
		api_key=os.getenv("COMET_API_KEY"),
		name=os.getenv("PROJECT_NAME"),
		workspace=os.getenv("WORKSPACE"),
		log_code=True,
	)

	etl = JobETL(csv_path=cfg.data.csv_path, logger=logger)
	etl.read_data()
	df, meta = etl.preprocess()

	# --- Tokenizer ---
	tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

	# --- DataModule ---
	dm = JobDataLoader(
		df,
		meta,
		tokenizer,
		batch_size=cfg.train.batch_size,
		num_workers=cfg.train.num_workers,
		logger=logger,
	)

	dm.prepare_data()
	dm.setup()
	sample_batch = next(iter(dm.train_dataloader()))
	logger.info(f"Sample batch keys: {list(sample_batch.keys())}")
	logger.info(f"Sample input_ids shape: {sample_batch['input_ids'].shape}")

	# --- Model ---
	cat_classes = list(meta["cat_classes"].values())
	model = FakeJobClassifier(cat_classes)

	# --- Callbacks ---
	checkpoint_cb = ModelCheckpoint(
		monitor=cfg.callbacks.checkpoint.monitor,
		dirpath=cfg.callbacks.checkpoint.dirpath,
		filename=cfg.callbacks.checkpoint.filename,
		save_top_k=cfg.callbacks.checkpoint.save_top_k,
		mode=cfg.callbacks.checkpoint.mode,
	)
	early_stop_cb = EarlyStopping(
		monitor=cfg.callbacks.early_stopping.monitor,
		patience=cfg.callbacks.early_stopping.patience,
		mode=cfg.callbacks.early_stopping.mode,
	)
	# --- Trainer ---
	trainer = Trainer(
		max_epochs=cfg.trainer.max_epochs,
		accelerator=cfg.trainer.accelerator,
		callbacks=[checkpoint_cb, early_stop_cb],
		log_every_n_steps=cfg.trainer.log_every_n_steps,
		enable_model_summary=cfg.trainer.enable_model_summary,
		deterministic=cfg.trainer.deterministic,
		limit_val_batches=cfg.trainer.limit_val_batches,
		val_check_interval=cfg.trainer.val_check_interval,
		logger=comet_logger,
	)

	logger.info("Starting training...")
	trainer.fit(model, dm)
	logger.info("Training complete.")
