import pandas as pd

from fake_job_postings.data.etl import JobETL


def test_etl_read_data():
	csv_path = "/Users/amur.saqib/PycharmProjects/fake_job_postings/fake_job_postings/src/fake_job_postings/data/fake_job_postings.csv"
	etl = JobETL(csv_path)
	etl.read_data()
	assert not etl.df.empty


def test_etl_clean_loc():
	dummy = pd.DataFrame({"location": ["USA,California,San Fransisco"]})
	etl = JobETL()
	etl.df = dummy.copy()
	etl._clean_location()
	assert "location" not in etl.df.columns
	assert etl.df["country"][0] == "USA"
	assert etl.df["city"][0] == "San Fransisco"


def test_etl_drop_unused():
	dummy = pd.DataFrame(
		{"job_id": [1, 2, 3], "salary_range": ["50k-70k", "20k-30k", "5000-10000"]}
	)
	etl = JobETL()
	etl.df = dummy.copy()
	etl._drop_unused()
	assert "job_id" not in etl.df.columns
	assert "salary_range" not in etl.df.columns


def test_etl_clean_text():
	dummy = pd.DataFrame(
		{
			"title": [None],
			"company_profile": ["profile 1"],
			"description": [None],
			"requirements": ["req1"],
			"benefits": [None],
		}
	)
	etl = JobETL()
	etl.df = dummy.copy()
	etl._clean_text()
	assert etl.df["title"][0] == "UNKNOWN"
	assert etl.df["company_profile"][0] == "profile 1"
	assert etl.df["description"][0] == "UNKNOWN"
	assert etl.df["requirements"][0] == "req1"
	assert etl.df["benefits"][0] == "UNKNOWN"


def test_clean_cats():
	df = pd.DataFrame(
		{
			"department": [None, "Engineering"],
			"employment_type": ["Full-time", "Full-time"],
			"required_experience": [None, None],
			"required_education": ["Bachelors", None],
			"industry": [None, "Tech"],
			"function": ["Engineering", None],
			"country": ["USA", "USA"],
			"city": ["SF", "NY"],
		}
	)
	etl = JobETL()
	etl.df = df.copy()
	etl._clean_cats()
	assert etl.df["department"].isnull().sum() == 0
	assert isinstance(etl.meta["cat_classes"], dict)
	assert "department" in etl.meta["cat_classes"]
	assert etl.meta["cat_classes"]["department"] == 2


def test_agg_text():
	df = pd.DataFrame(
		{
			"title": ["Data Scientist"],
			"company_profile": ["Great company"],
			"description": ["Job desc"],
			"requirements": ["Python"],
			"benefits": ["Snacks"],
		}
	)
	etl = JobETL()
	etl.df = df.copy()
	etl.agg_text()
	assert "text" in etl.df.columns
	assert "title:\nData Scientist" in etl.df["text"][0]
