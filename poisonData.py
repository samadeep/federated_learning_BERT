import pandas as pd
import random


def swapLabel(data):
    x = random.random()
    if x > .6:
        return data ^ 1
    return data


def poisonData():
    df = pd.read_csv("spamdata_v2.csv")
    df = df.sample(n=1000)

    newLabel = df['label'].apply(swapLabel)
    df['label'] = newLabel

    file_path = 'poison_data.csv'
    df.to_csv(file_path)


