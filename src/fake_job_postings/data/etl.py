import pandas as pd


class JobETL:
	def __init__(self, csv_path="fake_job_postings.csv", logger=None):
		self.logger = logger
		self.csv_path = csv_path
		self.df = None
		self.cat_cols = [
			"department",
			"employment_type",
			"required_experience",
			"required_education",
			"industry",
			"function",
			"country",
			"city",
		]
		self.text_cols = [
			"title",
			"company_profile",
			"description",
			"requirements",
			"benefits",
		]
		self.target = ["fraudulent"]
		self.meta = {
			"text_cols": self.text_cols,
			"cat_cols": self.cat_cols,
			"cat_classes": {},
			"target": self.target,
		}

	def read_data(self):
		try:
			self.df = pd.read_csv(self.csv_path)
			if self.logger:
				self.logger.debug(f"Read data from {self.csv_path}")
		except Exception as e:
			if self.logger:
				self.logger.error(f"Failed to read data from {self.csv_path}")
			raise e

	def _clean_location(self):
		self.df["country"] = self.df["location"].str.split(",", expand=True)[0]
		self.df["city"] = self.df["location"].str.split(",", expand=True)[2]
		self.df.drop(columns=["location"], inplace=True)

	def _clean_text(self):
		for col in self.text_cols:
			self.df[col] = self.df[col].fillna("UNKNOWN")

	def _clean_cats(self):
		for col in self.cat_cols:
			self.df[col] = self.df[col].fillna("UNKNOWN")
			self.meta["cat_classes"][col] = len(self.df[col].unique())

	def _drop_unused(self):
		self.df.drop(columns=["job_id", "salary_range"], inplace=True)

	def agg_text(self):
		self.df["text"] = (
			self.df[self.text_cols]
			.astype(str)
			.apply(
				lambda row: "\n".join([f"{col}:\n{row[col]}" for col in row.index]),
				axis=1,
			)
			.values
		)

	def preprocess(self):
		self._clean_location()
		self._clean_text()
		self._clean_cats()
		self.agg_text()
		self._drop_unused()
		if self.logger:
			self.logger.debug(f"Preprocessed data from {self.csv_path}")

		return self.df, self.meta


# import logging
#
# logger = logging.getLogger("JobETL")
# logging.basicConfig(level=logging.DEBUG)
#
# etl = JobETL(logger=logger)
# etl.read_data()
# df_clean, meta = etl.preprocess()
# logger.debug(df_clean['text'].head())
