import os

current_dir = os.getcwd()
DATA_DIR = os.path.join(current_dir, 'data')
LOSS_DIR = os.path.join(current_dir, 'loss')

MODEL_DIR = os.path.join(current_dir, 'models')

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
if not os.path.exists(LOSS_DIR):
    os.mkdir(LOSS_DIR)

MODELS_PATH_dic = {}
for models in ['g_models', 'e_models', 'r_models', 'd_models', 's_models']:
    MODELS_PATH = os.path.join(MODEL_DIR, models)
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    MODELS_PATH_dic[models] = MODELS_PATH



