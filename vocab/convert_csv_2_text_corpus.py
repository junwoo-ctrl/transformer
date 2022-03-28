import pandas as pd
import sys
import csv

csv.field_size_limit(sys.maxsize)

in_file = "vocab/kowiki_20220328.csv"
out_file = "vocab/kowiki.txt"
SEPARATOR = u"\u241D"
df = pd.read_csv(in_file, sep=SEPARATOR, engine="python")
with open(out_file, "w") as f:
  for index, row in df.iterrows():
    f.write(row["text"]) # title 과 text를 중복 되므로 text만 저장 함
    f.write("\n\n\n\n") # 구분자