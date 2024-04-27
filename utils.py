import pandas as pd
import numpy as np
from data_dir import DATA_DIR

def MinMaxScaler(data):

    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)

    return numerator / (denominator + 1e-7)

def google_data_loading(seq_length):
    # Load Google Data
    x = np.loadtxt(f'{DATA_DIR}/GOOGLE_BIG.csv', delimiter=",", skiprows=1)
    # Flip the data to make chronological data
    x = x[::-1]
    # Min-Max Normalizer
    x = MinMaxScaler(x)

    # Build dataset
    dataX = []

    # Cut data by sequence length
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)

    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))

    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])

    return outputX

def multi_stock_loading_first():

    data_dir = {'google': f"{DATA_DIR}/GOOG.csv", 'apple': f"{DATA_DIR}/AAPL.csv",
                'meta': f"{DATA_DIR}/META.csv", 'netflix': f"{DATA_DIR}/NFLX.csv",
                'amazon': f"{DATA_DIR}/AMZN.csv" }
    data = pd.read_csv(data_dir['meta'])
    data['Date'] = pd.to_datetime(data['Date'])

    for st, st_dir in data_dir.items():
        tmp = pd.read_csv(st_dir)
        tmp = tmp.rename(columns={'Adj Close': st})
        tmp['Date'] = pd.to_datetime(tmp['Date'])
        data = pd.merge(data, tmp[['Date', st]], on='Date', how='inner')
    #Only keep Adj Close column
    data.drop(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], inplace=True)

    return data

