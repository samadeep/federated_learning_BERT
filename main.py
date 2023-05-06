import subprocess
import pandas as pd
import os
from random import randint
from datamaker import func
from poisonData import poisonData

port = 5002

final_results = pd.DataFrame(columns=['Client no.', 'Type', 'Epoch', 'Average Loss', 'Accuracy'])

file_path = 'final_results_2.csv'
csv_file = "spamdata_v2.csv"
poison_csv_file = "poison_data.csv"


if os.path.isfile(file_path):
    print('The file exists')
else:
    print('The file does not exist')
    final_results.to_csv(file_path)

func()
poisonData()


client = 5  # number of times to run the file
epochs = 1
numberOfRounds = 10
filename = "client1.py"  # name of the file to run
data = []
numberOfData = 1500
# change the number of samples

for _ in range(client):
    data.append((str(numberOfData)))



top_select = 3
number_of_cluster = 1
processes = []

processes.append(subprocess.Popen(
    ["python3", "server2.py"] + [str(port), str(number_of_cluster), str(top_select), str(numberOfRounds)]))

# comment this to do not make poison client
# processes.append(subprocess.Popen(["python3", filename] + [str(port), str(-1), str(0 + 1), str(epochs), poison_csv_file]))


for i in range(client):
    processes.append(subprocess.Popen(["python3", filename] + [str(port), data[i], str(i + 1), str(epochs), csv_file]))

for process in processes:
    process.wait()

"""
16 cl 15 ep  5clu  top`2   -> 2000 data

32     20     8    top 2   ->  1000 data

"""
