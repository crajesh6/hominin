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

    pred = model.predict(X, batch_size=512) #[i].squeeze()
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

def create_conv_layer(conv1_type, filters, kernel_size, diag=None, offdiag=None):
    """
    Create a convolutional layer based on the specified convolution type.

    Args:
        conv1_type (str): The type of convolution ('standard' or 'pairwise').
        filters (int): Number of convolutional filters.
        kernel_size (int): Size of the convolutional kernel.
        diag: The diagonal regularization term for pairwise convolution.
        offdiag: The off-diagonal regularization term for pairwise convolution.

    Returns:
        Keras Layer: The created convolutional layer.
    """
    print(conv1_type, filters, kernel_size)
    if conv1_type == 'standard':
        return tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            groups=filters,
            use_bias=True,
            name='conv1',
        )
    elif conv1_type == 'pairwise':
        return layers.PairwiseConv1D(
            filters,
            kernel_size=kernel_size,
            padding='valid',
            groups=filters,
            kernel_regularizer=layers.PairwiseReg(diag=diag, offdiag=offdiag),
            use_bias=True,
            name='conv1',
        )
    else:
        raise ValueError(f"Unsupported conv1_type: {conv1_type}")

# =============================================================================
# Model Classes
# =============================================================================

class ExpActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExpActivation, self).__init__(**kwargs)

    def call(self, x):
        return backend.exp(x)


class Unsqueeze(tf.keras.layers.Layer):
    def call(self, x):
        return tf.expand_dims(x, axis=1)

# =============================================================================
# Model
# =============================================================================

class _ExplaiNN(Model):
    def __init__(self, num_cnns, input_shape, num_classes, conv1_type=None, filter_size=19, num_fc=2, pool_size=7, pool_stride=7,
                 weight_path=None, **kwargs):
        super(_ExplaiNN, self).__init__(**kwargs)
        self._options = {
            "num_cnns": num_cnns,
            "input_shape": input_shape,
            "num_classes": num_classes,
            "filter_size": filter_size,
            "num_fc": num_fc,
            "pool_size": pool_size,
            "pool_stride": pool_stride,
            "conv1_type": conv1_type
        }

        diag = l2(1e-6)
        offdiag = l2(1e-3)

        # First conv. layer
        conv_layer = create_conv_layer(conv1_type, num_cnns, filter_size, diag, offdiag)

        tf.keras.backend.clear_session()


        if num_fc == 0:
            self.linears = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                # tf.keras.layers.Conv1D(filters=num_cnns, kernel_size=filter_size, groups=num_cnns),
                conv_layer,
                tf.keras.layers.BatchNormalization(),
                ExpActivation(),
                tf.keras.layers.MaxPooling1D(pool_size=input_shape[0] - (filter_size-1)), # input_length
                tf.keras.layers.Flatten()
            ])
        elif num_fc == 1:
            self.linears = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                # tf.keras.layers.Conv1D(filters=num_cnns, kernel_size=filter_size, groups=num_cnns),
                conv_layer,
                tf.keras.layers.BatchNormalization(),
                ExpActivation(),
                tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_stride),
                tf.keras.layers.Flatten(),
                Unsqueeze(),
                tf.keras.layers.Conv1D(filters=num_cnns, kernel_size=1, groups=num_cnns),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Flatten()
            ])
        elif num_fc == 2:
            self.linears = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                # tf.keras.layers.Conv1D(filters=num_cnns, kernel_size=filter_size, groups=num_cnns),
                conv_layer,
                tf.keras.layers.BatchNormalization(),
                ExpActivation(),
                tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_stride),
                tf.keras.layers.Flatten(),
                Unsqueeze(),
                tf.keras.layers.Conv1D(filters=100 * num_cnns, kernel_size=1, groups=num_cnns),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv1D(filters=num_cnns, kernel_size=1, groups=num_cnns),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Flatten()
            ])
        else:
            self.linears = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                # tf.keras.layers.Conv1D(filters=num_cnns, kernel_size=filter_size, groups=num_cnns),
                conv_layer,
                tf.keras.layers.BatchNormalization(),
                ExpActivation(),
                tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_stride),
                tf.keras.layers.Flatten(),
                Unsqueeze(),
                tf.keras.layers.Conv1D(filters=100 * num_cnns, kernel_size=1, groups=num_cnns),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ])

            self.linears_bg = [tf.keras.Sequential([
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv1D(filters=100 * num_cnns, kernel_size=1, groups=num_cnns),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ]) for i in range(num_fc - 2)]

            self.last_linear = tf.keras.Sequential([
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv1D(filters=num_cnns, kernel_size=1, groups=num_cnns),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Flatten()
            ])

        self.final = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = tf.repeat(x, repeats=self._options["num_cnns"], axis=2)

        if self._options["num_fc"] <= 2:
            outs = self.linears(x)

        else:
            outs = self.linears(x)
            for i in range(len(self.linears_bg)):
                outs = self.linears_bg[i](outs)
            outs = self.last_linear(outs)
        return self.final(outs)



@click.command()
@click.option("--save_path", type=str)
@click.option("--num_cnns", type=int)
@click.option("--gpu", type=str, default=None)
@click.option("--smoke_test", type=bool, default=False)
def main(save_path: str, num_cnns: int, gpu: str, smoke_test: bool):

    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    make_directory(save_path)

    learning_rate = 0.001
    # num_cnns = 300
    filter_size = 19
    num_classes = 2
    num_epochs = 1 if smoke_test else 20
    conv1_type = save_path.split("/")[-1]

    if not smoke_test:
        wandb.init(project="replicate_explainn", name="standard_gam_tensorflow")


    x_train, y_train = load_deepstarr_data(data_split="train", subsample=smoke_test)
    x_test, y_test = load_deepstarr_data(data_split="test", subsample=smoke_test)
    x_valid, y_valid = load_deepstarr_data(data_split="valid", subsample=smoke_test)

    input_shape = (249, num_cnns * 4)
    model = _ExplaiNN(
        num_cnns=num_cnns,
        input_shape=input_shape,
        num_classes=num_classes,
        filter_size=filter_size,
        conv1_type=conv1_type
        )
    model.build(input_shape=(None, 249, 4))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=[Spearman]
    )

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
                        batch_size=128,
                        shuffle=True,
                        validation_data=(x_valid, y_valid),
                        callbacks=callbacks)

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
