import pandas as pd
import pytest
import torch
from transformers import RobertaTokenizerFast

from fake_job_postings.data.dataset import JobDataLoader
from fake_job_postings.data.etl import JobETL


@pytest.fixture
def job_df():
	rows = []
	for i in range(20):
		rows.append(
			{
				"title": "Data Scientist",
				"company_profile": "Company A",
				"description": "Job description here.",
				"requirements": "Python, SQL",
				"benefits": "Health, Snacks",
				"department": "Tech",
				"employment_type": "Full-time",
				"required_experience": "Mid",
				"required_education": "Bachelor",
				"industry": "Tech",
				"function": "Engineering",
				"location": "USA,New York,NY",
				"job_id": i,
				"salary_range": "50-100k",
				"fraudulent": i % 2,  # Alternate labels
			}
		)
	return pd.DataFrame(rows)


@pytest.fixture
def meta():
	return {
		"text_cols": [
			"title",
			"company_profile",
			"description",
			"requirements",
			"benefits",
		],
		"cat_cols": [
			"department",
			"employment_type",
			"required_experience",
			"required_education",
			"industry",
			"function",
			"country",
			"city",
		],
		"cat_classes": {
			"department": 1,
			"employment_type": 1,
			"required_experience": 1,
			"required_education": 1,
			"industry": 1,
			"function": 1,
			"country": 1,
			"city": 2,
		},
		"target": ["fraudulent"],
	}


@pytest.fixture
def tokenizer():
	return RobertaTokenizerFast.from_pretrained("roberta-base")


def test_dataloader_batch(job_df, meta, tokenizer):
	job_etl = JobETL()
	job_etl.df = job_df.copy()
	job_df, meta = job_etl.preprocess()
	dm = JobDataLoader(job_df, meta, tokenizer, batch_size=4, num_workers=4)
	dm.prepare_data()
	dm.setup()
	batch = next(iter(dm.train_dataloader()))
	assert "input_ids" in batch
	assert batch["input_ids"].shape == (4, 512)
	assert batch["attention_mask"].shape == (4, 512)
	assert isinstance(batch["cat_data"], torch.Tensor)
	assert isinstance(batch["label"], torch.Tensor)
	assert batch["cat_data"].dtype == torch.long
	assert batch["label"].dtype == torch.long
	assert batch["cat_data"].shape == (4, 8)
	assert batch["label"].shape == (4, 1)
