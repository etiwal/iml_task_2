import pandas as pd
import numpy as np
import scipy
from sklearn.svm import LinearSVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader


from DataChanger import impute_data, build_stats
from utils import healthDataset

np.set_printoptions(suppress=True)

do_imputation = False
use_imputed_data = False
train_model_densenet = True

df_train_features = pd.read_csv('train_features.csv').fillna(0)
pids_train, pids_validate = list(df_train_features['pid'].unique())[0:3000], list(df_train_features['pid'].unique())[18000:]

df_train_labels = pd.read_csv('train_labels.csv')
#df_test_features = pd.read_csv('test_features.csv')

epochs = 20

if do_imputation:
    # build statistics
    # build_stats(df_train_features)

    train_features_imputed = impute_data(df_train_features, impute_strategy='SimpleImputer_pid')
    print(train_features_imputed)
    train_features_imputed.to_csv('train_features_imputed.csv', sep=',', float_format='%.3f', encoding='utf-8', index=False)

if use_imputed_data:
    df_train_features = pd.read_csv('train_features_imputed.csv')

if train_model_densenet:

    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.zero_()

    # Building net
    model = nn.Sequential(
        nn.Linear(408, 1000), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(1000, 1000), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(1000, 1000), nn.Sigmoid(), nn.Dropout(0.1),
        nn.Linear(1000, 11)
    )

    print(model)

    model.apply(weight_init)



    np_train_features = np.array(df_train_features)
    np_train_lables = np.array(df_train_labels)

    #print(df_train_features[df_train_features['pid'].isin(pids_train)])

    training_set = healthDataset(np_train_features, np_train_lables, pids_train)
    train_loader = DataLoader(training_set, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)

    validation_set = healthDataset(np_train_features, np_train_lables, pids_validate)
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss().float()

    def train_model():
        model.train()
        epoch_loss = []

        for i, data in enumerate(train_loader):
            features, labels = data
            features = features.float()
            labels = labels.float()

            pred_labels = model(features)
            loss = loss_criterion(pred_labels, labels)
            epoch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('train loss:', np.mean(epoch_loss))


    def eval_model():
        model.eval()

        epoch_loss = []

        for i, data in enumerate(validation_loader):
            features, labels = data
            features = features.float()
            labels = labels.float()

            with torch.no_grad():
                pred_labels = model(features)
                loss = loss_criterion(pred_labels, labels)
                epoch_loss.append(loss.item())

                if i == 0:
                    example_true = labels.detach().numpy()
                    example_pred = pred_labels.detach().numpy()

        print('eval loss:', np.mean(epoch_loss))
        print('True values:', example_true)
        print('Predicted values:', example_pred)


    for epoch in range(0, epochs):
        print()
        print('epoch:', epoch)
        train_model()
        eval_model()






     #
    #
    # def train():
    #     net.train()
    #     losses = []
    #     loss_avg = 0
    #     per
    #     old_per = 00
    #     for epoch in range(1epochs):
    #         x_train = torch.from_numpy(features_train).float()
    #         y_train = torch.from_numpy(labels_train).float()
    #         y_pred = net(x_train) * 50
    #         loss = criterion(y_pred
    #         y_train)
    #         loss_avg = loss_avg * 0.4 + 0.6 * loss.item()
    #         per = math.floor(epoch / epochs * 100)
    #         if per > old_per:
    #             print(\epoch:\ epoch
    #             '('
    #             per
    #             '/ 100 )')
    #             print('loss:'
    #             round(loss_avg3))
    #             old_per = per
    #
    #         losses.append(loss.item())
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     return losses
    #
    #
    # def test():
    #     net.eval()
    #
    #     pred = net(x_test) * 50
    #     print(\Accuracy
    #     on
    #     test
    #     set(MSE): \ mean_squared_error(labels_test
    #     pred.detach().numpy().reshape(-1)))
    #
    #     # print(pred.detach().numpy().reshape(-1))
    #     # print(labels_test)
    #
    #     pred_train = net(x_train) * 50
    #     print(\Accuracy
    #     on
    #     train
    #     set(MSE): \ mean_squared_error(labels_train
    #     pred_train.detach().numpy()))
    #
    #
    #     print(\training
    #     start....\)
    #     losses = train()
    #     plt.plot(range(1
    #     epochs) losses)
    #     plt.xlabel(\epoch\)
    #     plt.ylabel(\loss
    #     train\)
    #     plt.ylim([min(losses)min(losses) * 10])
    #     plt.show()
    #
    #     print(\testing
    #     start... \)
    #     x_test = torch.from_numpy(features_test).float()
    #     x_train = torch.from_numpy(features_train).float()
    #
    #     test()



# dict structure
#train_features = {'11402': {'age': 22, '1':{'EtCO2': 123, 'etc':133}, '2':{'EtCO2': 234, 'etc':432}}}

# x_train = np.array(df_train_features)
# y_train = np.array(df_train_labels)
# x_test = np.array(df_test_features)
#
# svm_reg = LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, fit_intercept=False, dual=True, verbose=0, random_state=None, max_iter=1000)
# svm_reg.fit(x_train,y_train)
# y_pred = svm_reg.predict(x_test)
