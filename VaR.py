import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as MMS
from keras.models import load_model
import keras
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Flatten
from ctganV2 import *
from scipy.stats import norm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
from data_dir import DATA_DIR
from utils import *

def traditional_var(tmp, conf_level1=0.05):
    price = tmp.set_index('Date')
    returns = price.pct_change()
    # Generate Var-Cov matrix
    cov_matrix = returns.cov()
    # Calculate mean returns for each stock
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    avg_rets = returns.mean()
    port_mean = avg_rets.dot(weights)
    initial_investment = 1000000
    # Calculate portfolio standard deviation
    port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
    # Calculate mean of investment
    mean_investment = (1+port_mean) * initial_investment
    # Calculate standard deviation of investmnet
    stdev_investment = initial_investment * port_stdev
    cutoff1 = norm.ppf(conf_level1, mean_investment, stdev_investment)
    # Calculate n Day VaR
    var_array = []
    var_1d1 =  cutoff1 -initial_investment
    for x in [1,5,9]:
        var_array.append(np.round(var_1d1 * np.sqrt(x),2))
    return var_array

def cgan_var(current_day, data_v, tmp):
    cond_len = 5
    seq_length = 9
    scaler = MMS()
    scaled_data = scaler.fit_transform(data_v).astype(np.float32)
    name = 'First_d_CTGAN_LSTM_V6.5' #['CTGAN_MA','ctg','CTGAN_F']
    g_net = load_model(f'D:\CTS_GAN\models\g_models/{current_day}{name}', compile=False)
    r_net = load_model(f'D:\CTS_GAN\models/r_models/{current_day}{name}',compile = False)
    s_net = load_model(f'D:\CTS_GAN\models\s_models/{current_day}{name}',compile = False)
    def make_random_data():
        while True:
            yield np.random.uniform(low=0, high=1, size=(seq_length, z_dim))
    random_series = iter(tf.data.Dataset
                         .from_generator(make_random_data, output_types=tf.float32)
                         .batch(300, drop_remainder=True)
                         .repeat())
    simu_var_df = pd.DataFrame()
    Z = next(random_series)
    l_condition = scaled_data[-cond_len:]
    last_day = tmp.iloc[-1][:-1].values
    for i in range(200):
        #cz = tf.keras.layers.Concatenate(axis=1)([tf.expand_dims(tf.convert_to_tensor(l_condition),axis=0),tf.expand_dims(Z[i], axis=0)])
        if name == 'CTGAN_F'or 'LSTM' in name:
            c = Flatten()(tf.expand_dims(tf.convert_to_tensor(l_condition),axis=0))
            c_pre = tf.tile(tf.expand_dims(c, axis=1), [1, tf.shape(tf.expand_dims(Z[i], axis=0))[1], 1])
            cz = tf.concat([tf.expand_dims(Z[i], axis=0), c_pre], axis=-1)
        E_hat = g_net(cz)[:, -seq_length: , :]
        H_hat = s_net(E_hat)
        X_hat = r_net(H_hat)
        arr = scaler.inverse_transform(X_hat[0])
        cumulative_loss = np.cumsum(arr, axis=0)
        one_day = (cumulative_loss[0] )/last_day
        five_day = (cumulative_loss[4] )/last_day
        fourteen_day = (cumulative_loss[-1])/last_day
        capital = np.ones(5) * (1000000/5)
        simu_var = [np.dot(one_day, capital), np.dot(five_day, capital), np.dot(fourteen_day, capital)]
        simu_var_df = pd.concat([simu_var_df, pd.DataFrame(simu_var).transpose()])
    return list(simu_var_df.quantile(0.05)), list(simu_var_df.quantile(0.01)), list(simu_var_df.quantile(0.1))


def actual_loss(start_day, data):
    data_L = data.iloc[start_day: start_day + window_size + seq_length]
    real_fdata = data_L[-seq_length-1:].drop(columns='Date').values
    t_p = real_fdata[0]
    t_one = real_fdata[1]
    t_five = real_fdata[5]
    t_fourteen = real_fdata[-1]
    one_day = (t_one - t_p) / t_p
    five_day = (t_five - t_p) / t_p
    fourteen_day = (t_fourteen - t_p) / t_p
    capital = np.ones(5) * (1000000 / 5)
    actual_loss = [np.dot(one_day, capital), np.dot(five_day, capital), np.dot(fourteen_day, capital)]

    return actual_loss


def df_create(lst, day):
    df = pd.DataFrame(lst).transpose()
    df['Date'] = day

    return df

var_PATH = 'D:\\CTS_GAN\\Notebooks\\VaR'
if not os.path.exists(var_PATH):
    os.mkdir(var_PATH)
window_size = 2000
SEED = 3407
set_random_seed(SEED)
data = multi_stock_loading_first()
start_day = 0
seq_length = 9
seq_len = 14
z_dim = 5
trad_var = pd.DataFrame()
simu_var = pd.DataFrame()
simu_var99 = pd.DataFrame()
simu_var90 = pd.DataFrame()
actual_invest = pd.DataFrame()
while start_day < 880:

    current_day = data.iloc[start_day + window_size]['Date'].strftime('%Y-%m-%d')
    # print(current_day)
    tmp = data.iloc[start_day: start_day + window_size]
    data_v = tmp.drop(columns='Date').values
    name = 'First_d_CTGAN_LSTM_V6.5'
    if name == 'CTGAN_F' or 'First' in name:
        tmp.set_index('Date', inplace=True)
        result = tmp.diff()
        result = result.dropna()
        data_v = result.values
        tmp['Date'] = tmp.index
    traditional_var_lst = traditional_var(tmp, conf_level1=0.05)
    simulated_var_lst, simulated_var_lst99,  simulated_var_lst90 = cgan_var(current_day, data_v, tmp)
    actual_l = actual_loss(start_day, data)

    trad_var_df = df_create(traditional_var_lst, current_day)
    simu_var_df = df_create(simulated_var_lst, current_day)
    simu_var_df99 = df_create(simulated_var_lst99, current_day)
    simu_var_df90 = df_create(simulated_var_lst90, current_day)
    actual_df = df_create(actual_l, current_day)

    trad_var = pd.concat([trad_var, trad_var_df])
    simu_var = pd.concat([simu_var, simu_var_df])
    simu_var99 = pd.concat([simu_var99, simu_var_df99])
    simu_var90 = pd.concat([simu_var90, simu_var_df90])
    actual_invest = pd.concat([actual_invest, actual_df])

    start_day += 1

    #if start_day % 10 == 0:
    print(start_day)
trad_var.to_csv(f'{var_PATH}/{name}_trad_95.csv')
simu_var.to_csv(f'{var_PATH}/{name}_simu_95.csv')
simu_var99.to_csv(f'{var_PATH}/{name}_simu_99.csv')
simu_var90.to_csv(f'{var_PATH}/{name}_simu_90.csv')
actual_invest.to_csv(f'{var_PATH}/{name}_actual_loss.csv')