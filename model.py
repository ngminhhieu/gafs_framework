from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor as xgbmodel
from sklearn.linear_model import LinearRegression
import numpy as np
import random, os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
        
def run_xgboost(X_train, y_train, X_test, y_test, scaler):
    model = xgbmodel(objective='reg:squarederror')
    model.fit(X_train, y_train, eval_metric="mae", verbose=False)

    train_results = model.predict(X_train)
    mae_train, r2_train = evaluate(scaler, y_train, train_results, "xgboost", "train")

    test_results = model.predict(X_test)
    mae_test, r2_test = evaluate(scaler, y_test, test_results, "xgboost", "test")
    return mae_train, r2_train, mae_test, r2_test

def run_lr(X_train, y_train, X_test, y_test, scaler):
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_results = model.predict(X_train)
    mae_train, r2_train = evaluate(scaler, y_train, train_results, "lr", "train")

    test_results = model.predict(X_test)
    mae_test, r2_test = evaluate(scaler, y_test, test_results, "lr", "test")
    return mae_train, r2_train, mae_test, r2_test

def run_lstm(X_train, y_train, X_test, y_test, scaler):
    
    def process_data(X_train, y_train, X_test, y_test):
        seq_len = 48
        horizon = 1
        X_train_lstm = []
        y_train_lstm = []
        X_test_lstm = []
        y_test_lstm = []
        for i in range(0, len(X_train) - seq_len - horizon):
            X_train_lstm.append(X_train[i:i+seq_len, :])
            y_train_lstm.append(y_train[i+seq_len])
        for i in range(0, len(X_test) - seq_len - horizon):
            X_test_lstm.append(X_train[i:i+seq_len, :])
            y_test_lstm.append(y_train[i+seq_len])
        X_train_lstm = np.stack(X_train_lstm)
        y_train_lstm = np.stack(y_train_lstm)
        X_test_lstm = np.stack(X_test_lstm)
        y_test_lstm = np.stack(y_test_lstm)
        return X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm
    X_train, y_train, X_test, y_test = process_data(X_train, y_train, X_test, y_test)
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[2], 16)
            self.lstm = nn.LSTM(16, 8, batch_first=True)
            self.fc2 = nn.Linear(8, 1)

        def forward(self, x):
            embedding = self.fc1(x)
            out_lstm = self.lstm(embedding)[0]
            out = self.fc2(F.relu(out_lstm[:, -1, :]))
            return out

    model = LSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    bs = 32
    num_epochs = 2
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), bs):
            inputs = X_train[i:i+bs]
            target = y_train[i:i+bs]
            out = model(inputs).squeeze()
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_results = model(X_train)
        mae_train, r2_train = evaluate(scaler, y_train, train_results, "lstm", "train")
        test_results = model(X_test)
        mae_test, r2_test = evaluate(scaler, y_test, test_results, "lstm", "test")
    return mae_train, r2_train, mae_test, r2_test

def evaluate(scaler, gt, pred, algo, phase_name):
    output_log = os.path.join("log", "visualization")
    if not os.path.exists(output_log):
        os.makedirs(output_log)
    pred_ori = pred - scaler.min_[-1] # aqmesh
    pred_ori /= scaler.scale_[-1] # aqmesh
    gt_ori = gt - scaler.min_[-1] # aqmesh
    gt_ori /= scaler.scale_[-1] # aqmesh

    mae = mean_absolute_error(gt_ori, pred_ori)
    r2 = r2_score(gt_ori, pred_ori)
    plt.plot(gt_ori, label="gt")
    plt.plot(pred_ori, label="pred")
    plt.legend()
    plt.savefig(os.path.join(output_log, "pred_{}_{}.png".format(algo, phase_name)))
    plt.close()
    return mae, r2