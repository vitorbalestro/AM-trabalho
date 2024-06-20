import pandas as pd
import matplotlib.pyplot as plt


data_path = 'raw_data.csv'

df = pd.read_csv(data_path)


histogram = df["NObeyesdad"].value_counts(normalize=True) * 100

print(histogram)
