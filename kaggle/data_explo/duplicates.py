import pandas as pd
from sqlalchemy.engine import row

df = pd.read_csv('../train_dataset_M1_with_id.csv')

num_duplicates= df.duplicated().sum()

print(f"Number of fully duplicate rows: {num_duplicates}")

#Duplicate ids 
num_duplicate_ids = df.duplicated(subset=["id"]).sum()

print(f"Number of fully duplicate rows: {num_duplicate_ids}")

#None rows are 90% similar
# from thefuzz import fuzz
#from tqdm import tqdm
#
#rows_as_text = df.head(10000).astype(str).agg("".join, axis=1)
#
#threshold = 90
#similar_pairs = []
#
#for i in tqdm(range(len(rows_as_text)), desc="Comapring rows"):
#    for j in range(i+1, len(rows_as_text)):
#        sim = fuzz.ratio(rows_as_text[i], rows_as_text[j])
#        if sim >= threshold:
#            similar_pairs.append((i,j,sim))
#
#print(f"{len(similar_pairs)} pairs of rows are ≥{threshold}% similar")
