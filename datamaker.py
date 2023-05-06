import pandas as pd

def func():
    df = pd.read_csv("spamdata_v2.csv")
    df = df.sample(n=100)

    file_path = 'test_data.csv'
    df.to_csv(file_path)

