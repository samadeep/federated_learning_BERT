## Email Spam Classification using BERT using flower library


- Email Spam Classification using BERT on flower : This project demonstrates how to use BERT, a pre-trained deep learning model, for email spam classification using Flower

## Data preparation
The Flower dataset consists of emails labeled as spam or not spam. We will preprocess the data by tokenizing the text and converting it into the input format expected by BERT.

## Model definition
We will use the Hugging Face Transformers library to load a pre-trained BERT model and fine-tune it for email spam classification on the Flower dataset.

## Using Federated Learning
To improve the performance of the model, we will use federated learning. Federated learning allows multiple parties to collaboratively train a model without sharing their data with each other. In this project, we will simulate a federated learning environment using a single machine and multiple virtual clients.

## Run this Project
- Run the main.py file with the specifications you need in the 27th line of main.py :
  '''
  client = 5  # number of times to run the file
  epochs = 1
  numberOfRounds = 10
  filename = "client1.py"  # name of the file to run
  data = []
  numberOfData = 1500
  '''
