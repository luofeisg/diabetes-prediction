import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from utils import fill_data_knn

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)

        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


def main():
    # gender_age_race, zero, mice, datawig
    fill_method = "15"
    data_path = "csv_file/filled/" + fill_method + "/"
    x_train = pd.read_csv(data_path + "x_train.csv")
    y_train = pd.read_csv(data_path + "y_train.csv")['outcome'].to_numpy()
    x_test = pd.read_csv(data_path + "x_test.csv")
    y_test = pd.read_csv(data_path + "y_test.csv")['outcome'].to_numpy()
    x_train.drop(["HbA1c_%"], axis=1, inplace=True)
    x_test.drop(["HbA1c_%"], axis=1, inplace=True)

    # testing: remove feature 345
    # remove ["OGTT2hr_(mmol/l)", "FPG_(mmol/l", "HbA1c_%"]
    # X_train_df.drop(["OGTT2hr_(mmol/l)", "FPG_(mmol/l", "HbA1c_%"], axis=1, inplace=True)
    # X_test_df.drop(["OGTT2hr_(mmol/l)", "FPG_(mmol/l", "HbA1c_%"], axis=1, inplace=True)

    # split train/test in 80/20%
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.20)
    # # reset index
    # X_train = X_train.reset_index(drop=True)
    # X_test = X_test.reset_index(drop=True)
    # y_train = y_train.reset_index(drop=True)
    # y_test = y_test.reset_index(drop=True)

    # source: Race_Chinese; target: Race_Malay, Race_Indian
    # X_train_df = X[X['Race_Chinese'] == 1]
    # X_test_Malay = X[X['Race_Malay'] == 1]
    # X_test_Indian = X[X['Race_Indian'] == 1]
    # y_train = y[X['Race_Chinese'] == 1]
    # y_test_Malay = y[X['Race_Malay'] == 1]
    # y_test_Indian = y[X['Race_Indian'] == 1]

    # feature scaling
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)

    # batch_size, epoch and iteration
    batch_size = 500
    num_epochs = 100

    x_train = torch.from_numpy(x_train).float().to(DEVICE)
    y_train = torch.from_numpy(y_train).long().to(DEVICE)
    x_test = torch.from_numpy(x_test).float().to(DEVICE)
    y_test = torch.from_numpy(y_test).long().to(DEVICE)

    # train and test sets
    train = TensorDataset(x_train, y_train)
    test = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    input_dim = x_train.shape[-1]
    hidden_dim = 100
    layer_dim = 1
    output_dim = 2
    dropout_prob = 0.2
    # RNN
    # model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim).to(DEVICE)
    # LSTM
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob).to(DEVICE)
    # GRU
    # model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob).to(DEVICE)

    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1, num_epochs+1):
        print("epoch: {}".format(epoch))
        for i, (x, labels) in enumerate(train_loader):
            x = Variable(x.view(-1, 1, input_dim)).float()
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(x)
            loss = error(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch >= num_epochs:
            # Calculate Accuracy
            correct = 0
            total = 0
            loss_list = []
            accuracy_list = []
            all_predicted, all_labels = torch.Tensor().to(DEVICE), torch.Tensor().to(DEVICE)
            # Iterate through test dataset
            for x, labels in test_loader:
                x = Variable(x.view(-1, 1, input_dim))
                outputs = model(x)
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum()
                accuracy = 100 * correct / float(total)
                all_predicted = torch.cat((all_predicted, predicted))
                all_labels = torch.cat((all_labels, labels))

                # store loss and accuracy
                loss_list.append(loss.item())
                accuracy_list.append(accuracy.item())
                # Print Loss
                # print('Testing: Epoch: {}/{}  Loss: {}  Accuracy: {} %'.format(epoch, num_epochs, np.mean(loss_list), np.mean(accuracy_list)))
            all_labels = all_labels.cpu().numpy()
            all_predicted = all_predicted.cpu().numpy()
            print("Accuracy: {:.3f}".format(accuracy_score(all_labels, all_predicted)), end=" ")
            print("Precision: {:.3f}".format(precision_score(all_labels, all_predicted)), end=" ")
            print("Recall: {:.3f}".format(recall_score(all_labels, all_predicted)), end=" ")
            print("F1: {:.3f}".format(f1_score(all_labels, all_predicted)), end=" ")


if __name__ == '__main__':
    main()

