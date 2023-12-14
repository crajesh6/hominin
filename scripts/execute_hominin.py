import os, sys

import click

from hominin import hominin


@click.command()
@click.option("--config_file", type=str)
@click.option("--dataset_name", type=str, default="deepstarr")
@click.option("--gpu", type=str, default=None)
@click.option("--smoke_test", type=bool, default=False)
@click.option("--log_wandb", type=bool, default=True)
def main(config_file: str, dataset_name: str, gpu: str, smoke_test: bool, log_wandb: bool):


    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    epochs = 2 if smoke_test else 100

    save_path = config_file.split("config.yaml")[0]
    config = hominin.load_config(config_file)

    tuner = hominin.homininTuner(
                                config,
                                epochs=epochs,
                                tuning_mode=False,
                                save_path=save_path,
                                dataset=dataset_name,
                                subsample=smoke_test,
                                log_wandb=log_wandb
                                )

    # train model
    tuner.train_model()


if __name__ == "__main__":
    main()
