import torch

from fake_job_postings.model import FakeJobClassifier


def test_model_inputs():
	model = FakeJobClassifier(cat_classes=[1, 20, 30, 40])
	size = (1, 512)
	input_ids = torch.randint(low=0, high=255, size=size, dtype=torch.long)
	attention_mask = torch.ones(size, dtype=torch.long)
	cat_data = torch.zeros(1, 8, dtype=torch.long)

	logits = model.forward(input_ids, attention_mask, cat_data)
	assert logits.size() == torch.Size([1])
