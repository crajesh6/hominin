import os, sys

import click
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import AsyncHyperBandScheduler

from homo import hominid


def tune_model(num_samples, dataset_name, param_space):

    def train_fn(config):
        model_trainer = hominid.HominidTuner(
                    config,
                    epochs=100,
                    tuning_mode=True,
                    save_path=None,
                    dataset=dataset_name,
                    subsample=False,
                    log_wandb=False
                    )
        model_trainer.train_model()
        return

    tune_metric = "val_aupr" if dataset_name == "scbasset" else "val_pearson_r"

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    tuner = tune.Tuner(
        tune.with_resources(train_fn, resources={"cpu": 4, "gpu": 1}),
        tune_config=tune.TuneConfig(
            metric=tune_metric,
            mode="max",
            scheduler=sched,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name="plantstarr_sweep",
            stop={
                tune_metric: 0.95,
                "training_iteration": 100
            },
            callbacks=[
            WandbLoggerCallback(
                project="plantstarr_sweep",
                log_config=True,
                upload_checkpoints=True,)]
            ),
        param_space=param_space,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


@click.command()
@click.option("--config_file", type=str, default='/home/chandana/projects/homo/experiments/sweeps/plantstarr/config.yaml')
@click.option("--dataset_name", type=str, default="deepstarr")
@click.option("--gpu", type=str, default=0)
@click.option("--smoke_test", type=bool, default=False)
@click.option("--log_wandb", type=bool, default=True)
def main(config_file: str, dataset_name: str, gpu: str, smoke_test: bool, log_wandb: bool):

    # check if dataset does not match experiment
    if dataset_name not in config_file:
        print('Dataset and experiment do not match. Abort!')
        sys.exit()

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print(gpu)

    sweep_config = hominid.load_config(config_file)
    parameters = {}
    for key in sweep_config:
        if isinstance(sweep_config[key], list):
            parameters[key] = tune.choice(sweep_config[key])
        else:
            parameters[key] = sweep_config[key]
    parameters

    num_samples= 1 if smoke_test else 100
    tune_model(num_samples, dataset_name, parameters)


if __name__ == "__main__":
    main()
