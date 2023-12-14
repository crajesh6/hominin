import h5py, os
from filelock import FileLock
from pathlib import Path

import click
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path
import pickle
import seaborn as sns
import sh


from scipy import stats
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1, l2, l1_l2

from tensorflow.keras import models, backend
import wandb
from wandb.keras import WandbCallback

from homo import layers

# =============================================================================
# Util functions
# =============================================================================

def make_directory(dir_name: str):
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    return

def load_deepstarr_data(
        data_split: str,
        data_dir='/home/chandana/projects/hominid_pipeline/data/deepstarr_data.h5',
        subsample: bool = False
    ) -> (np.ndarray, np.ndarray):
    """Load dataset"""

    # load sequences and labels
    with FileLock(os.path.expanduser(f"{data_dir}.lock")):
        with h5py.File(data_dir, "r") as dataset:
            x = np.array(dataset[f'x_{data_split}']).astype(np.float32)
            y = np.array(dataset[f'y_{data_split}']).astype(np.float32).transpose()
    if subsample:
        if data_split == "train":
            x = x[:80000]
            y = y[:80000]
        elif data_split == "valid":
            x = x[:20000]
            y = y[:20000]
        else:
            x = x[:10000]
            y = y[:10000]
    return x, y

def _flip(x, axis):
    return tf.reverse(x, [axis])


from scipy.stats import spearmanr
def Spearman(y_true, y_pred):
     return ( tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                       tf.cast(y_true, tf.float32)], Tout = tf.float32) )


def evaluate_model(model, X, Y, task):

    i = 0 if task == "Dev" else 1

    pred = model.predict(X, batch_size=32) #[i].squeeze()
    if len(pred) == 2:
        pred = pred[i].squeeze()
    else:
        pred = pred[:, i]

    mse = mean_squared_error(Y[:, i], pred)
    pcc = stats.pearsonr(Y[:, i], pred)[0]
    scc = stats.spearmanr(Y[:, i], pred)[0]

    print(f"{task} MSE = {str('{0:0.3f}'.format(mse))}")
    print(f"{task} PCC = {str('{0:0.3f}'.format(pcc))}")
    print(f"{task} SCC = {str('{0:0.3f}'.format(scc))}")

    return mse, pcc, scc



# =============================================================================
# Model Classes
# =============================================================================


# =============================================================================
# Model
# =============================================================================

def build_nam():
    order = {

    }
    pass


def GAMModel(input_shape, conv1_type, num_filters, dense_units, kernel_size=19):

    tf.keras.backend.clear_session()
    diag = l2(1e-6)
    offdiag = l2(1e-3)

    if conv1_type == 'standard':
        conv_layer = tf.keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding='same',
            use_bias=True,
            name='conv1',
        )
    elif conv1_type == 'pairwise':
        conv_layer = layers.PairwiseConv1D(
            num_filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=layers.PairwiseReg(diag=diag, offdiag=offdiag),
            use_bias=True,
            name='conv1',
        )

    inputs = tf.keras.layers.Input(shape=input_shape)
    L, A = input_shape

    x = conv_layer(inputs) # N, L, F
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='exponential')(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)


    # Fully connected layer # 1
    x = tf.keras.layers.Reshape((L, num_filters, 1))(x) # N, L, F, 1
    x = tf.keras.layers.EinsumDense('abcd,de->abce', output_shape=[L, num_filters, dense_units])(x) # (N, L, F, D)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)


    # Fully connected layer # 2
    x = tf.keras.layers.EinsumDense('abce,de->abcd', output_shape=[L, num_filters, 1])(x) # (N, L, F, 1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)

    # Remove last dimension to apply global average pooling on length axis
    x = tf.keras.layers.Reshape((L, num_filters))(x) # N, L, F

    # Average over the length dimension
    x = tf.keras.layers.GlobalAveragePooling1D()(x)  # (N, F)

    # Dense to get single value
    dev_outputs = tf.keras.layers.Dense(1, activation='linear', name="Dev_head")(x) # (N, 1)
    hk_outputs = tf.keras.layers.Dense(1, activation='linear', name="Hk_head")(x) # (N, 1)

    model = tf.keras.models.Model(inputs, [dev_outputs, hk_outputs])

    return model


    # need to be able to change location of the global average pooling layer
    # need to be able to swap in a multihead attention block
    # some rules as to how to do that would be nice
    # the dense layer with (2) actually leads to worse performance



@click.command()
@click.option("--save_path", type=str)
@click.option("--conv1_type", type=str)
@click.option("--gpu", type=str, default=None)
@click.option("--smoke_test", type=bool, default=False)
def main(save_path: str, conv1_type: str, gpu: str, smoke_test: bool):

    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    make_directory(save_path)

    learning_rate = 0.001
    # num_cnns = 300
    filter_size = 19
    num_classes = 2
    num_epochs = 1 if smoke_test else 20

    if not smoke_test:
        wandb.init(project="replicate_explainn", name=f"gam_einsum_{conv1_type}", magic=True)


    x_train, y_train = load_deepstarr_data(data_split="train", subsample=smoke_test)
    x_test, y_test = load_deepstarr_data(data_split="test", subsample=smoke_test)
    x_valid, y_valid = load_deepstarr_data(data_split="valid", subsample=smoke_test)
    N, L, A = x_train.shape

    model = GAMModel(input_shape=(L, A), conv1_type=conv1_type, num_filters=96, dense_units=64)
    model.compile(tf.keras.optimizers.Adam(lr=1e-3),
                      loss=['mse', 'mse'], # loss
                      loss_weights=[1, 1], # loss weights to balance
                      metrics=[Spearman]) # additional track metric
    model.summary()


    # early stopping callback
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', #'val_aupr',#
                                                patience=10,
                                                verbose=1,
                                                mode='min',
                                                restore_best_weights=True)
    # reduce learning rate callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.2,
                                                    patience=3,
                                                    min_lr=1e-7,
                                                    mode='min',
                                                    verbose=1)
    callbacks = [es_callback, reduce_lr]
    if not smoke_test:
        callbacks += [WandbCallback(save_model=False)]

    # train model
    history = model.fit(x_train, y_train,
                        epochs=num_epochs,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(x_valid, y_valid))

    pd.DataFrame(history.history).to_csv(f'{save_path}/history.csv')

    print("Done training the model!")
    model.save_weights(f'{save_path}/weights')


    # run for each set and enhancer type
    mse_dev, pcc_dev, scc_dev = evaluate_model(model, x_test,  y_test, "Dev")
    mse_hk, pcc_hk, scc_hk = evaluate_model(model, x_test,  y_test, "Hk")

    data = [{
        'MSE_dev':  mse_dev,
        'PCC_dev':  pcc_dev,
        'SCC_dev':  scc_dev,
        'MSE_hk':  mse_hk,
        'PCC_hk':  pcc_hk,
        'SCC_hk':  scc_hk,
    }]
    df = pd.DataFrame(data)
    evaluation_path = f"{save_path}/evaluation"
    make_directory(evaluation_path)
    pd.DataFrame(df).to_csv(f'{evaluation_path}/model_performance.csv', index=False)



if __name__ == "__main__":
    main()
