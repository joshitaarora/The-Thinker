import os
from src.data.data_loader import clone_repo

import pandas as pd


df = pd.read_csv('Thinker Eval Database - Selected.csv')

OUTPUT_DIR = "src/data/selected_repos"

for _, item in df.iterrows():
  print("Running for " + item["Github URL"])
  project_name = item["Github URL"].split("/")[-1].split(".")[0]
  clone_repo(item["Github URL"], os.path.join(OUTPUT_DIR, project_name))