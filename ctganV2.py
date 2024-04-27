import numpy as np
import pandas as pd
from data_dir import LOSS_DIR, MODELS_PATH_dic
import tensorflow as tf
from keras.utils import set_random_seed
from sklearn.preprocessing import MinMaxScaler as MMS
from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input, Flatten,LSTM
from tensorflow.keras.models import Sequential, Model
from keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam

def make_rnn(n_layers, hidden_units, output_units, name):
    return Sequential([GRU(units=hidden_units,
                           return_sequences=True,
                           name=f'GRU_{i + 1}') for i in range(n_layers)] +
                      [Dense(units=output_units,
                             activation='sigmoid',
                             name='OUT')], name=name)



def ctgan(data_v, parameters):
    SEED = 3407
    set_random_seed(SEED)
    # Network & data Parameters
    seq_len = parameters['seq_len']
    cond_len = parameters['conditional_len']
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layers']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module_name']  # 'lstm' or 'gru'
    z_dim = parameters['z_dim']
    current_day = parameters['current_day']
    last_day = parameters['last_day']
    RETRAIN = parameters['reload']  # whether a net trained from 0
    NAME = parameters['name']
    learning_rate = 2e-4
    GAMMA = 1

    # generated sequence length
    seq_length = seq_len - cond_len

    # Min_Max Scale
    scaler = MMS()
    scaled_data = scaler.fit_transform(data_v).astype(np.float32)

    data = []
    for i in range(len(data_v) - seq_len):
        data.append(scaled_data[i:i + seq_len])

    n_windows = len(data)
    real_series = (tf.data.Dataset
                   .from_tensor_slices(data)
                   .shuffle(buffer_size=n_windows)
                   .batch(batch_size, drop_remainder=True))
    real_series_iter = iter(real_series.repeat())

    X = Input(shape=[seq_length, z_dim], name='RealData')
    C = Input(shape=[cond_len, z_dim], name='ConditionalData')
    Z = Input(shape=[seq_length, z_dim], name='RandomData')

    def make_random_data():
        while True:
            yield np.random.uniform(low=0, high=1, size=(seq_length, z_dim))

    random_series = iter(tf.data.Dataset
                         .from_generator(make_random_data, output_types=tf.float32)
                         .batch(batch_size, drop_remainder=True)
                         .repeat())

    if RETRAIN == False:
        embedder = make_rnn(n_layers=3,
                            hidden_units=hidden_dim,
                            output_units=hidden_dim,
                            name='Embedder')
        recovery = make_rnn(n_layers=3,
                            hidden_units=hidden_dim,
                            output_units=z_dim,
                            name='Recovery')
        generator = make_rnn(n_layers=2,
                             hidden_units=hidden_dim,
                             output_units=hidden_dim,
                             name='Generator')
        discriminator = make_rnn(n_layers=2,
                                 hidden_units=hidden_dim,
                                 output_units=1,
                                 name='Discriminator')
        supervisor = make_rnn(n_layers=2,
                              hidden_units=hidden_dim,
                              output_units=hidden_dim,
                              name='Supervisor')
    else:
        generator = load_model(f'{MODELS_PATH_dic["g_models"]}/{last_day}{NAME}', compile=False)
        recovery = load_model(f'{MODELS_PATH_dic["r_models"]}/{last_day}{NAME}', compile=False)
        embedder = load_model(f'{MODELS_PATH_dic["e_models"]}/{last_day}{NAME}', compile=False)
        discriminator = load_model(f'{MODELS_PATH_dic["d_models"]}/{last_day}{NAME}', compile=False)
        supervisor = load_model(f'{MODELS_PATH_dic["s_models"]}/{last_day}{NAME}', compile=False)

    mse = MeanSquaredError()
    bce = BinaryCrossentropy()

    H = embedder(X)
    X_tilde = recovery(H)

    C_flat = Flatten()(C)
    Z_concat_C = tf.tile(tf.expand_dims(C_flat, axis=1), [1, tf.shape(Z)[1], 1])
    CZ = tf.concat([Z, Z_concat_C], axis=-1)  # (14, 55)
    E_hat = generator(CZ)[:, -seq_length:, :]

    autoencoder = Model(inputs=X,
                        outputs=X_tilde,
                        name='Autoencoder')
    generator_model = Model(inputs=[C, Z], outputs=E_hat)

    autoencoder_optimizer = Adam(learning_rate = 2 * learning_rate)

    def train_autoencoder_init(x):
        with tf.GradientTape() as tape:
            x_tilde = autoencoder(x)
            embedding_loss_t0 = mse(x, x_tilde)
            e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)

        var_list = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        autoencoder_optimizer.apply_gradients(zip(gradients, var_list))
        return tf.sqrt(embedding_loss_t0)

    print("*" * 15, 'Training Embedder', '*' * 15)
    step_e_loss_t0_lst = []
    for step in range(iterations):
        X_ = next(real_series_iter)
        T_ = X_[:, -seq_length:, :]
        step_e_loss_t0 = train_autoencoder_init(T_)

        if step % 10 == 0:
            step_e_loss_t0_lst.append(np.round(step_e_loss_t0.numpy(), decimals=4))

            if step % 100 == 0:
                print('step', step, '| loss:', step_e_loss_t0)
    if RETRAIN == False:
        pd.DataFrame(step_e_loss_t0_lst).to_csv(f'{LOSS_DIR}/step_e_loss_t0_{NAME}.csv')
    else:
        pd.DataFrame(step_e_loss_t0_lst).to_csv(f'{LOSS_DIR}/step_e_loss_t0_{NAME}.csv', mode='a', header=False)

    supervisor_optimizer = Adam(learning_rate= 2 * learning_rate)

    def train_supervisor(x):
        with tf.GradientTape() as tape:
            h = embedder(x)
            h_hat_supervised = supervisor(h)
            g_loss_s = mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

        var_list = supervisor.trainable_variables
        gradients = tape.gradient(g_loss_s, var_list)
        supervisor_optimizer.apply_gradients(zip(gradients, var_list))
        return g_loss_s

    print("*" * 15, 'Training Supervisor', "*" * 15)
    step_g_loss_s_lst = []
    for step in range(iterations):
        X_ = next(real_series_iter)
        T_ = X_[:, -seq_length:, :]
        step_g_loss_s = train_supervisor(T_)
        if step % 10 == 0:
            step_g_loss_s_lst.append(np.round(step_g_loss_s.numpy(), decimals=4))
            if step % 100 == 0:
                print('step', step, '| loss:', step_g_loss_s)
    if RETRAIN == False:
        pd.DataFrame(step_g_loss_s_lst).to_csv(f'{LOSS_DIR}/step_g_loss_s_{NAME}.csv')
    else:
        pd.DataFrame(step_g_loss_s_lst).to_csv(f'{LOSS_DIR}/step_g_loss_s_{NAME}.csv', mode='a', header=False)

    H_hat = supervisor(E_hat)
    Y_fake = discriminator(H_hat)

    adversarial_supervised = Model(inputs=CZ,
                                   outputs=Y_fake,
                                   name='AdversarialNetSupervised')

    Y_fake_e = discriminator(E_hat)

    adversarial_emb = Model(inputs=CZ,
                            outputs=Y_fake_e,
                            name='AdversarialNet')

    X_hat = recovery(H_hat)
    synthetic_data = Model(inputs=CZ,
                           outputs=X_hat,
                           name='SyntheticData')

    def get_generator_moment_loss(y_true, y_pred):
        y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
        g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
        g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
        return g_loss_mean + g_loss_var

    Y_real = discriminator(H)
    discriminator_model = Model(inputs=X,
                                outputs=Y_real,
                                name='DiscriminatorReal')
    generator_optimizer = Adam(learning_rate=learning_rate)
    discriminator_optimizer = Adam(learning_rate= 3.0 * learning_rate)
    embedding_optimizer = Adam(learning_rate=learning_rate)

    def train_generator(x, z):
        with tf.GradientTape() as tape:
            c = Flatten()(x[:, :cond_len, :])
            c_pre = tf.tile(tf.expand_dims(c, axis=1), [1, tf.shape(z)[1], 1])
            cz = tf.concat([z, c_pre], axis=-1)
            y_fake = adversarial_supervised(cz)

            generator_loss_unsupervised = bce(y_true=tf.ones_like(y_fake),
                                              y_pred=y_fake)

            y_fake_e = adversarial_emb(cz)
            generator_loss_unsupervised_e = bce(y_true=tf.ones_like(y_fake_e),
                                                y_pred=y_fake_e)
            h = embedder(x[:, -seq_length:, :])
            h_hat_supervised = supervisor(h)
            generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

            x_hat = synthetic_data(cz)
            generator_moment_loss = get_generator_moment_loss(x[:, -seq_length:, :], x_hat)

            generator_loss = (generator_loss_unsupervised +
                              generator_loss_unsupervised_e +
                              100 * tf.sqrt(generator_loss_supervised) +
                              100 * generator_moment_loss)

        var_list = generator.trainable_variables + supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        generator_optimizer.apply_gradients(zip(gradients, var_list))

        return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss
    def train_embedder(x):
        with tf.GradientTape() as tape:
            h = embedder(x)
            h_hat_supervised = supervisor(h)
            generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

            x_tilde = autoencoder(x)
            embedding_loss_t0 = mse(x, x_tilde)
            e_loss = 10 * tf.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

        var_list = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        embedding_optimizer.apply_gradients(zip(gradients, var_list))
        return tf.sqrt(embedding_loss_t0)

    def get_discriminator_loss(x, z):
        c = Flatten()(x[:, :cond_len, :])
        c_pre = tf.tile(tf.expand_dims(c, axis=1), [1, tf.shape(z)[1], 1])
        cz = tf.concat([z, c_pre], axis=-1)
        y_real = discriminator_model(x[:,-seq_length:,:])
        discriminator_loss_real = bce(y_true=tf.ones_like(y_real),
                                      y_pred=y_real)

        y_fake = adversarial_supervised(cz)
        discriminator_loss_fake = bce(y_true=tf.zeros_like(y_fake),
                                      y_pred=y_fake)

        y_fake_e = adversarial_emb(cz)
        discriminator_loss_fake_e = bce(y_true=tf.zeros_like(y_fake_e),
                                        y_pred=y_fake_e)
        return (discriminator_loss_real +
                discriminator_loss_fake +
                GAMMA * discriminator_loss_fake_e)

    def train_discriminator(x, z):
        with tf.GradientTape() as tape:
            discriminator_loss = get_discriminator_loss(x, z)

        var_list = discriminator.trainable_variables
        gradients = tape.gradient(discriminator_loss, var_list)
        discriminator_optimizer.apply_gradients(zip(gradients, var_list))
        return discriminator_loss

    print("*" * 15, 'Training jointly', "*" * 15)
    G_loss_lst = []
    D_loss_lst = []
    step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
    for step in range(iterations):
        # Train generator (twice as often as discriminator)
        for kk in range(2):
            X_ = next(real_series_iter)
            E_ = X_[:, -seq_length:, :]
            Z_ = next(random_series)

            # Train generator
            step_g_loss_u, step_g_loss_s, step_g_loss_v = train_generator(X_, Z_)
            # Train embedder
            step_e_loss_t0 = train_embedder(E_)

        X_ = next(real_series_iter)
        Z_ = next(random_series)
        step_d_loss = get_discriminator_loss(X_, Z_)
        if step_d_loss > 0.15:
            step_d_loss = train_discriminator(X_, Z_)

        if step % 10 == 0:
            G_loss_lst.append(step_g_loss_u)
            D_loss_lst.append(step_d_loss)
            if step % 100 == 0:
                print(f'step {step} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | '
                      f'g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}')

    if RETRAIN == False:
        pd.DataFrame(G_loss_lst).to_csv(f'{LOSS_DIR}/g_loss_u_{NAME}.csv')
        pd.DataFrame(D_loss_lst).to_csv(f'{LOSS_DIR}/d_loss_{NAME}.csv')
    else:
        pd.DataFrame(G_loss_lst).to_csv(f'{LOSS_DIR}/g_loss_u_{NAME}.csv', mode='a', header=False)
        pd.DataFrame(D_loss_lst).to_csv(f'{LOSS_DIR}/d_loss_{NAME}.csv', mode='a', header=False)

    generator.save(f'{MODELS_PATH_dic["g_models"]}/{current_day}{NAME}')
    recovery.save(f'{MODELS_PATH_dic["r_models"]}/{current_day}{NAME}')
    embedder.save(f'{MODELS_PATH_dic["e_models"]}/{current_day}{NAME}')
    discriminator.save(f'{MODELS_PATH_dic["d_models"]}/{current_day}{NAME}')
    supervisor.save(f'{MODELS_PATH_dic["s_models"]}/{current_day}{NAME}')

    print(current_day, f'{NAME} models successfully saved.')