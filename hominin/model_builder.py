import tensorflow as tf
from wandb.keras import WandbCallback
from ray.tune.integration.keras import TuneReportCallback

from hominin import model_zoo, utils


class ModelBuilder:
    def __init__(self, config, epochs, batch_size, log_wandb, tuning_mode, sigma=0.001):
        self.config = config
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_wandb = log_wandb
        self.tuning_mode = tuning_mode
        self.tune_metrics = None
        self.sigma = sigma

    def build_model(self):
        print("Building model...")
        model = model_zoo.build_model(**self.config)
        kernel_initializer = tf.keras.initializers.RandomNormal(stddev=self.sigma)  # Replace this with your desired initializer
        self.reinitialize_kernel_weights(model, kernel_initializer)
        return model


    def reinitialize_kernel_weights(self, model, kernel_initializer):
      # can change initialization after creating model this way!
      for i, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel_initializer'):

          # Get the current configuration of the layer
          weight_initializer = layer.kernel_initializer

          val = layer.get_weights()
          if len(val) > 1:
            old_weights, old_biases = val
            layer.set_weights([kernel_initializer(shape=old_weights.shape), old_biases])
          else:
            old_weights = val[0]
            layer.set_weights([kernel_initializer(shape=old_weights.shape)])


    def _get_early_stopping_callback(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, verbose=1,
            mode='min', restore_best_weights=True
        )

    def _get_reduce_lr_callback(self):
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, mode='min', verbose=1
        )

    def _get_tune_report_callback(self):
        return TuneReportCallback({key: key for key in self.tune_metrics})

    def _set_tune_report_callback(self, metrics):
        self.tune_metrics = metrics

    def compile_model(self, model, dataset):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = 'mse'
        metrics = [utils.Spearman, utils.pearson_r]
        tune_metrics = ["val_pearson_r"]
        self._set_tune_report_callback(tune_metrics)

        if dataset == "scbasset":
            auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
            aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
            metrics = [auroc, aupr]
            tune_metrics = ["val_aupr"]
            self._set_tune_report_callback(tune_metrics)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        ## TODO: Add initialization to all layers!
        return model

    def fit_model(self, model, dataset, x_train, y_train, x_valid, y_valid):

        callbacks = [
            self._get_early_stopping_callback(),
            self._get_reduce_lr_callback(),
        ]

        if self.log_wandb:
            callbacks += [WandbCallback(save_model=False)]
        if self.tuning_mode:
            tune_report_callback = self._get_tune_report_callback()
            callbacks.append(tune_report_callback)

        if dataset == "plantstarr":
            history = model.fit(x_train, y_train,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      shuffle=True,
                      validation_split=0.1,
                      callbacks=callbacks)
        elif dataset == "deepstarr":
            history = model.fit(x_train, y_train,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      shuffle=True,
                      validation_data=(x_valid, y_valid),
                      callbacks=callbacks)
        elif dataset == "scbasset":
            history = model.fit(x_train, y_train,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      shuffle=True,
                      validation_data=(x_valid, y_valid),
                      callbacks=callbacks)
        elif dataset == "hepg2":
            history = model.fit(x_train, y_train,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      shuffle=True,
                      validation_data=(x_valid, y_valid),
                      callbacks=callbacks)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return model, history
