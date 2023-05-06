import flwr as fl
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import AutoModel, BertTokenizerFast
import csv


# specify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")
import time


# Load dataset
# X, y = [], []
# email = load_files("data/enron7")
# X = np.append(X, email.data)
# y = np.append(y, email.target)
#
# df = pd.DataFrame(columns=['text', 'target'])
# df['text'] = [x for x in X[:100]]
# df['target'] = [t for t in y[:100]]
# df = df.dropna()
# df_X = df.drop(['target'], axis=1)
# df_y = df['target']


df = pd.read_csv(sys.argv[5])
if sys.argv[2] != '-1':
    df = df.sample(n=int(sys.argv[2]))


# split train dataset into train, validation and test sets
train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'], df['label'],
                                                                    random_state=2018,
                                                                    test_size=0.3,
                                                                    stratify=df['label'])

val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=2018,
                                                                test_size=0.5,
                                                                stratify=temp_labels)
# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# print(train_text)
# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    # is_split_into_words=True,
    max_length=25,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length=25,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length=25,
    pad_to_max_length=True,
    truncation=True
)

## convert lists to tensors

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# define a batch size
batch_size = 32

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = False


def getData():
    df = pd.read_csv("test_data.csv")

    # split train dataset into train, validation and test sets
    train_text, train_labels = df['text'], df['label']

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # print(train_text)
    # tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        # is_split_into_words=True,
        max_length=25,
        pad_to_max_length=True,
        truncation=True
    )


    ## convert lists to tensors

    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())


    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

    # define a batch size
    batch_size = 32

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return train_dataloader

class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 2)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x


# pass the pre-trained BERT to our define architecture --------------------------------
model = BERT_Arch(bert)
# print(model.state_dict())
# push the model to GPU
model = model.to(device)

# optimizer from hugging face transformers
from transformers import AdamW

# define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

from sklearn.utils.class_weight import compute_class_weight

# compute the class weights
# class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)

print("Class Weights:", class_weights)

# converting list of class weights to a tensor
weights = torch.tensor(class_weights, dtype=torch.float)

# push to GPU
weights = weights.to(device)

# define the loss function
cross_entropy = nn.NLLLoss(weight=weights)

# number of training epochs
epochs = 10

KEY = [i for i in model.state_dict()]

state_dict = model.state_dict()
model_weight_rr = []
# Convert the state dictionary values to ndarrays
for key, value in state_dict.items():
    model_weight_rr.append(value)


class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        print("\nGetting Params...")
        # Get the model state dictionary
        state_dict = model.state_dict()
        model_weight = []
        # Convert the state dictionary values to ndarrays
        for key, value in state_dict.items():
            model_weight.append(value)

        # print(type(model_weight), len(model_weight), len(model_weight[0]), len(model_weight[1]))
        return model_weight

    def fit(self, parameters, config):

        print("\nFitting Model...")
        new_weight_dict = {}
        for i, key in enumerate(KEY):
            new_weight_dict[key] = torch.from_numpy(parameters[i])
        # load model weights
        model.load_state_dict(new_weight_dict)

        # print("model loaded")
        model.train()
        # print("model Trained 1")

        epoch = int(sys.argv[4])     ##_________________________________________________________________epoch
        for eph in range(epoch):
            total_loss, total_accuracy = 0, 0
            total_examples = 0
            total_ham = 0
            itr = 1
            # iterate over batches
            for step, batch in enumerate(train_dataloader):
                # print(itr, total_loss, total_accuracy, total_examples)
                itr += 1
                # push the batch to gpu
                batch = [r.to(device) for r in batch]

                sent_id, mask, labels = batch

                # clear previously calculated gradients
                model.zero_grad()

                # get model predictions for the current batch
                preds = model(sent_id, mask)

                # compute the loss between actual and predicted values
                loss = cross_entropy(preds, labels)

                # add on to the total loss
                total_loss = total_loss + loss.item()
                # total_ham += sum(labels)
                total_ham += labels.sum().item()
                # calculate the accuracy for the current batch
                _, predicted = torch.max(preds.data, 1)
                total_accuracy += (predicted == labels).sum().item()

                # backward pass to calculate the gradients
                loss.backward()

                # clip the gradients to 1.0. It helps in preventing the exploding gradient problem
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # update parameters
                optimizer.step()

                # update the total number of examples processed
                total_examples += len(sent_id)

            # compute the training loss and accuracy of the epoch
            avg_loss = total_loss / len(train_dataloader)
            accuracy = total_accuracy / total_examples
            # print("Model Trained 2")
            row_data = [1, sys.argv[3], "Fit", eph, avg_loss, accuracy]

            with open('Final_Results.csv', mode='a', newline='') as results_file:
                writer = csv.writer(results_file)
                writer.writerow(row_data)

            print('Client no.->',sys.argv[3],'Epoch :', eph,'Average Loss:', avg_loss, 'Accuracy :', accuracy)
        # Get the model state dictionary
        newacc = self.testEval()
        print('Client no.->',sys.argv[3],"new Accuracy", newacc)
        print('Client no.->', sys.argv[3],'ham', total_ham / total_examples, 'Accuracy :', accuracy, total_examples)
        state_dict = model.state_dict()
        model_weight = []
        # Convert the state dictionary values to ndarrays
        for key, value in state_dict.items():
            model_weight.append(value)



        # return model_weight, total_examples, {"loss": avg_loss, "accuracy": accuracy}
        return model_weight, total_examples, {"loss": avg_loss, "newAccuracy": newacc, "ham": total_ham / total_examples, "oldAccuracy": accuracy}


    @staticmethod
    def testEval():
        # deactivate dropout layers
        model.eval()

        total_loss, total_accuracy, total_correct = 0, 0, 0
        val_dataloader_n = getData()
        # empty list to save the model predictions
        total_preds = []
        total_len = 0
        # iterate over batches
        itr = 1
        for step, batch in enumerate(val_dataloader_n):
            # print(itr, total_loss, total_accuracy, total_correct)
            itr += 1
            # Progress update every 50 batches.
            if step % 50 == 0 and not step == 0:
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

            # push the batch to gpu
            batch = [t.to(device) for t in batch]

            sent_id, mask, labels = batch

            # deactivate autograd
            with torch.no_grad():

                # model predictions
                preds = model(sent_id, mask)

                # compute the validation loss between actual and predicted values
                loss = cross_entropy(preds, labels)

                total_loss = total_loss + loss.item()

                preds = preds.detach().cpu().numpy()

                total_preds.append(preds)

                # convert predictions into labels
                preds_labels = np.argmax(preds, axis=1)
                total_len += len(preds_labels)
                # compute the number of correct predictions
                correct = np.sum(preds_labels == labels.cpu().numpy())

                total_correct += correct

        # compute the validation loss of the epoch
        avg_loss = total_loss / total_len
        accuracy = total_correct / total_len
        # print("eval done", len(val_dataloader))
        return accuracy


    def evaluate(self, parameters, config):
        print("\nEvaluating...")
        # load model weights
        new_weight_dict = {}
        for i, key in enumerate(KEY):
            new_weight_dict[key] = torch.from_numpy(parameters[i])
        # load model weights
        model.load_state_dict(new_weight_dict)

        # deactivate dropout layers
        model.eval()

        total_loss, total_accuracy, total_correct = 0, 0, 0

        # empty list to save the model predictions
        total_preds = []
        total_len = 0
        # iterate over batches
        itr = 1
        for step, batch in enumerate(val_dataloader):
            # print(itr, total_loss, total_accuracy, total_correct)
            itr += 1
            # Progress update every 50 batches.
            if step % 50 == 0 and not step == 0:
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

            # push the batch to gpu
            batch = [t.to(device) for t in batch]

            sent_id, mask, labels = batch

            # deactivate autograd
            with torch.no_grad():

                # model predictions
                preds = model(sent_id, mask)

                # compute the validation loss between actual and predicted values
                loss = cross_entropy(preds, labels)

                total_loss = total_loss + loss.item()

                preds = preds.detach().cpu().numpy()

                total_preds.append(preds)

                # convert predictions into labels
                preds_labels = np.argmax(preds, axis=1)
                total_len += len(preds_labels)
                # compute the number of correct predictions
                correct = np.sum(preds_labels == labels.cpu().numpy())

                total_correct += correct

        # compute the validation loss of the epoch
        avg_loss = total_loss / total_len
        accuracy = total_correct / total_len
        # print("eval done", len(val_dataloader))

        # print('Client No.->',sys.argv[3],'Average Loss->', avg_loss,'Accuracy->', accuracy)  # -------------------

        row_data = [1, sys.argv[3], "Eval", -1 , avg_loss, accuracy]

        with open('Final_Results.csv', mode='a', newline='') as results_file:
            writer = csv.writer(results_file)
            writer.writerow(row_data)

        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, len(val_dataloader), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="localhost:" + str(sys.argv[1]),
    client=FlowerClient(),
    grpc_max_message_length=1024 * 1024 * 1024
)
