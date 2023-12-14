import h5py, os

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path
import pickle
import seaborn as sns
import sh
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from ray.tune.integration.keras import TuneReportCallback
from ray.air import session
import wandb

from tfomics import impress
from hominin import utils, model_zoo, data_processor, model_builder

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_config(config_path):
    config = yaml.full_load(open(config_path))
    return config

class homininTuner:

    def __init__(self, config, epochs=60, batch_size=64, tuning_mode=False, save_path=None, subsample=False, dataset="deepstarr", log_wandb=True):
        self.config = config
        self.subsample = subsample
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.tuning_mode = tuning_mode
        self.save_path = save_path
        self.wandb_project = f"hominin_{self.dataset}"
        self.wandb_name = self.save_path
        self.log_wandb = log_wandb
        self.data_processor = data_processor.DataProcessor(dataset=self.dataset, subsample=self.subsample)
        self.mb = model_builder.ModelBuilder(self.config, self.epochs, self.batch_size, self.log_wandb, self.tuning_mode)

    def update_config(self, key, value):
        self.config[key] = value
        self.mb.config = self.config

    def train_model(self, config=None):
        if self.log_wandb:
            wandb.init(project=self.wandb_project, name=self.wandb_name, config=self.config, magic=True)

        if self.tuning_mode:
            self.save_path = f'{Path(session.get_trial_dir())}'

        x_train, y_train = self.data_processor.load_data("train")
        x_valid, y_valid = self.data_processor.load_data("valid")
        x_test, y_test = self.data_processor.load_data("test")

        print(f"Shape of x_train and y_train: {x_train.shape}, {y_train.shape}")

        (L, A), output_shape = self.data_processor.shape_info(x_train, y_train)
        self.update_config("input_shape", (L, A))
        self.update_config("output_shape", output_shape)

        model = self.mb.build_model()
        model, history = self.compile_and_train_model(model, x_train, y_train, x_valid, y_valid)
        pd.DataFrame(history.history).to_csv(f'{self.save_path}/history.csv')

        print("Done training the model!")

        model.save_weights(f'{self.save_path}/weights')

        with open(os.path.join(self.save_path, 'config.yaml'), 'w') as file:
            documents = yaml.dump(self.config, file)

        print(f"Evaluating model!")
        self.evaluate_model()

        return


    def compile_and_train_model(self, model, x_train, y_train, x_valid, y_valid):
        model = self.mb.compile_model(model, self.dataset)
        model, history = self.mb.fit_model(model, self.dataset, x_train, y_train, x_valid, y_valid)
        return model, history

    def evaluate_model(self):
        print("Loading model and dataset!")

        x_test, y_test = self.data_processor.load_data("test")

        model = self.mb.build_model()
        model = self.mb.compile_model(model, self.dataset)

        print(model.summary())
        model.load_weights(f'{self.save_path}/weights')

        print("Evaluating model!")
        evaluation_path = f"{self.save_path}/evaluation"
        Path(evaluation_path).mkdir(parents=True, exist_ok=True)

        evaluation_function = None
        if self.dataset == "deepstarr":
            evaluation_function = utils.evaluate_model_deepstarr
        elif self.dataset == "plantstarr":
            evaluation_function = utils.evaluate_model_plantstarr
        elif self.dataset == "scbasset":
            evaluation_function = utils.evaluate_model_scbasset

        elif self.dataset == "hepg2":
            evaluation_function = utils.evaluate_model_hepg2

        if evaluation_function is not None:
            df = evaluation_function(model, x_test, y_test, evaluation_path)
            return df
        else:
            return None

    def get_evaluation_results(self):
        df = pd.read_csv(f'{self.save_path}/evaluation/model_performance.csv')
        return df


    def visualize_filters(self, layer=2, plot_filters=True, from_saved=False):
        print(f"Loading model and dataset!")

        x_test, y_test = self.data_processor.load_data("test")

        model = self.mb.build_model()
        model = self.mb.compile_model(model, self.dataset)

        print(model.summary())
        model.load_weights(f'{self.save_path}/weights')

        print(f"Interpreting filters!")
        threshold = 0.5
        window = 20

        W, sub_W, counts = utils.calculate_filter_activations(
                            model,
                            x_test,
                            self.save_path,
                            layer,
                            threshold,
                            window,
                            batch_size=64,
                            plot_filters=plot_filters,
                            from_saved=from_saved
                        )


        print("Finished interpreting filters!")
        return

    def calculate_saliency_maps(self, class_index=0): # rewrite this so that the actual computations are in utils!
        print(f"Loading model and dataset!")

        x_test, y_test = self.data_processor.load_data("test")

        model = self.mb.build_model()
        model = self.mb.compile_model(model, self.dataset)

        print(model.summary())
        model.load_weights(f'{self.save_path}/weights')

        print(f"Calculating saliency maps!")
        saliency_path = f"{self.save_path}/saliency"
        Path(saliency_path).mkdir(parents=True, exist_ok=True)

        # Open the file to append saliency scores
        saliency_file = f'{saliency_path}/saliency_maps_{class_index}.pkl'
        batch_size = 64

        # Remove the file if it already exists:
        path = Path(saliency_file)
        if path.is_file():
            sh.rm(saliency_file)

        with open(saliency_file, 'ab') as file:
            # Iterate over batches
            for batch in utils.batch_generator(x_test, batch_size):
                # Get predictions for the batch
                saliency_scores = utils.saliency_map(batch, model, class_index=class_index)

                # Append predictions to the pickle file
                pickle.dump(saliency_scores, file)

        # plot 15 sequences with highest activity for a given class
        num_plot = 15

        # sort sequences by highest activity
        sort = np.argsort(y_test[:, class_index])[::-1][:num_plot]
        X = x_test[sort]

        # calculate attribution maps
        saliency_scores = utils.saliency_map(X, model, class_index=class_index)

        saliency_scores = saliency_scores.numpy()

        # gradient correction. (Majdandzic et al. bioRxiv, 2022)
        saliency_scores -= np.mean(saliency_scores, axis=2, keepdims=True)

        fig = plt.figure(figsize=(20, 11))
        for i, index in enumerate(range(num_plot)):
            x = np.expand_dims(X[index], axis=0)
            scores = np.expand_dims(saliency_scores[index], axis=0)
            # scores -= np.mean(scores, axis=2, keepdims=True)
            saliency_df = impress.grad_times_input_to_df(x, scores)

            ax = plt.subplot(num_plot,1,i+1)
            impress.plot_attribution_map(saliency_df, ax, figsize=(20,1))
            plt.ylabel(sort[i])

        fig.savefig(f'{saliency_path}/saliency_{class_index}.pdf', format='pdf', dpi=200, bbox_inches='tight')

        print("Finished calculating saliency maps!")
        return

    def run_gia(self, motif_1, motif_2):
        pass
    def run_glifac(self, class_index=None):

        print(f"Loading model and dataset!")
        x_test, y_test = self.data_processor.load_data("test")

        model = self.mb.build_model()
        model = self.mb.compile_model(model, self.dataset)

        model.load_weights(f'{self.save_path}/weights')

        # sort sequences by highest activity wrt class index
        if class_index:
            sort = np.argsort(y_test[:, class_index])[::-1]
        else:
            sort = range(len(x_test))
        X = x_test[sort]
        sample = X[:5000]

        lays = [type(i) for i in model.layers]
        c_index = [i.name for i in model.layers].index('conv1_pool')

        mha_name = 'multi_head_attention'

        if (self.config['mha_head_type'] == 'task_specific') and (class_index == 1):
            mha_name = 'multi_head_attention_1'

        mha_index = [i.name for i in model.layers].index('multi_head_attention')
        correlation_map = glifac.correlation_matrix(
                            model,
                            c_index,
                            mha_index,
                            sample,
                            thresh=0.1,
                            random_frac=0.3,
                            limit=150000
                        )

        # save correlation map to an h5 file
        glifac_path = f"{self.save_path}/glifac"
        utils.make_directory(glifac_path)

        file_path = f"{glifac_path}/correlation_map_{class_index}.h5"
        with h5py.File(file_path, "w") as f:
            dset = f.create_dataset(name="correlation_map", data=correlation_map, dtype='float32')

        # cluster and visualize heatmap (with full set of filters)
        clustered_interactions = sns.clustermap(
                                        np.clip(correlation_map, -1, 1),
                                        figsize=(10, 10),
                                        vmin=-.15, vmax=0.15,
                                        cmap='coolwarm_r',
                                )
        fig = clustered_interactions.fig
        fig.savefig(f'{glifac_path}/head_{class_index}-class_max.pdf', format='pdf', dpi=200, bbox_inches='tight')

        # Cluster and visualize heatmap (with non-redundant set of filters)
        clustered_filters_path = f"{self.save_path}/filters/clustered_filters/clustered_filters_central_motifs_IDs.tab"
        path = Path(clustered_filters_path)

        if path.is_file():


            non_redundant_filters_fig_path =  os.path.join(glifac_path, f'non_redundant_filters_head_{class_index}-class_max.pdf')

            non_redundant_filters = pd.read_csv(clustered_filters_path, sep="\t", header=None)[2].to_list()
            non_redundant_filters = [int(i.split('filter_')[-1]) for i in non_redundant_filters]

            correlation_map_nr = correlation_map[non_redundant_filters][:, non_redundant_filters]

            # cluster correlation map
            clustered_interactions = sns.clustermap(
                                            np.clip(correlation_map_nr, -1, 1),
                                            figsize=(10, 10),
                                            vmin=-.15, vmax=0.15,
                                            cmap='coolwarm_r',
                                            )

            fig = clustered_interactions.fig
            fig.savefig(non_redundant_filters_fig_path, format='pdf', dpi=200, bbox_inches='tight')



        return
