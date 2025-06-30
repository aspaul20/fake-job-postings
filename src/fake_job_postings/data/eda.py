import pandas as pd

df = pd.read_csv("fake_job_postings.csv")

# text feats = [title, company_profile, description, requirements, benefits]
# num feats = [telecommuting, has_company_logo, has_questions]
# cat feats = [department, employment_type, required_experience, required_education, industry, function, country, city]

# Separate out location
df["country"] = df["location"].str.split(",", expand=True)[0]
df["city"] = df["location"].str.split(",", expand=True)[2]
df.drop(columns=["location"], inplace=True)

# Add salary tag
# df['salary_clean'] = df['salary_range'].fillna("UNKNOWN")
# df['salary_clean'] = df['salary_clean'].str.replace(r'[^\d\-]', '', regex=True)
#
# print(df['salary_clean'])
#
# valid_salaries = df['salary_range'].str.contains(
#     r'^\d+-\d+$', na=False
# )
# df['salary_lower'] = np.nan
# df['salary_upper'] = np.nan
#
# df.loc[valid_salaries, 'salary_lower'] = np.nan
#
# df.fillna('UNKNOWN', inplace=True)

# print(df['function'].value_counts())
# print(df.head())
# print(df.info())

# Extract min-max salary
# df['salary_range_clean'] = df['salary_range'].str.replace(r'[^\d\-]','', regex=True)
# print(df['salary_range_clean'].value_counts())
# valid_salaries = df['salary_range_clean'].str.contains(r'^\d+-\d+$', na=False)
# df['lower_salary'] = np.nan
# df['upper_salary'] = np.nan
#
# salary_split = (
#     df['salary_range_clean'].str.split('-', expand=True)
# )
# salary_split = salary_split[(salary_split[0] != '') & (salary_split[1] != '')]
# df.loc[valid_salaries, 'lower_salary'] = salary_split[0].astype(float)
# df.loc[valid_salaries, 'upper_salary'] = salary_split[1].astype(float)
#
# print(df['upper_salary'].value_counts())
# print(df['lower_salary'].value_counts())
