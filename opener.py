import pandas as pd

data = pd.read_parquet('part1.parquet.gzip')
data.head()