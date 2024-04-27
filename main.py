from utils import *
from tgan2 import *
from ctganV2 import *


WINDOW_SIZE = 2000
PROCESSING = 'First_d' #['Ori','First_d'] 'Ori' means original data, 'First_d' is applying first-order difference
GAN_MODEL = 'CTGAN_LSTM_V6.5' #['TGAN']
SEED = 3407
#set_random_seed(SEED)
data = multi_stock_loading_first()
start_day = 0

while start_day < len(data) - WINDOW_SIZE:

    current_day = data.iloc[start_day + WINDOW_SIZE]['Date'].strftime('%Y-%m-%d')
    print('current_day: ', current_day)
    last_day = data.iloc[start_day-1 + WINDOW_SIZE]['Date'].strftime('%Y-%m-%d')
    tmp = data.iloc[start_day: start_day + WINDOW_SIZE]

    data_v = tmp.drop(columns='Date').values
    min_dt = np.min(data_v, 0)
    max_dt = np.max(data_v, 0)

    if PROCESSING == 'First_d':
        tmp.set_index('Date', inplace=True)
        result = tmp.diff()
        result = result.dropna()
        data_v = result.values
    elif PROCESSING == 'Ori':
        pass
    parameters = dict()
    parameters['hidden_dim'] = len(data_v[0, :]) * 4
    parameters['seq_len'] = 14                   #sample length
    parameters['num_layers'] = 3
    parameters['iterations'] = 2000
    parameters['batch_size'] = 128
    parameters['module_name'] = 'gru'
    parameters['z_dim'] = len(data_v[0, :])
    parameters['current_day'] = current_day
    parameters['last_day'] = last_day
    parameters['reload'] = False         #Train from 0
    parameters['name'] = PROCESSING + '_' + GAN_MODEL

    if start_day >= 1:
        parameters['reload'] = True  # start from last day's models' weight
        parameters['iterations'] = 10
    if GAN_MODEL == 'TGAN':
        timegan2(data_v, parameters) #TimeGAN tf2
    elif 'LSTM' in GAN_MODEL:
        print(GAN_MODEL)
        parameters['conditional_len'] = 5  # length of condition
        ctgan(data_v, parameters)  # Conditional TimeGAN

    start_day += 1
    # if start_day == 2:
