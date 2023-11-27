from typing import List, Tuple

import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")

    if datamodule.hparams.dataset in ["nyu", "sunrgbd"]:
        results = {}
        if datamodule.hparams.dataset == "nyu":
            iter = 10
        else:
            iter = 1
        for i in range(iter):
            pl.seed_everything(cfg.seed + i)
            trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
            metric_dict = trainer.callback_metrics
            for k in metric_dict.keys():
                if k[5:] in results:
                    results[k[5:]].append(metric_dict[k].item())
                else:
                    results[k[5:]] = [metric_dict[k].item()]
        import csv
        import numpy as np

        with open(model.val_csv, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=model.fieldnames)
            output = {}
            output["epoch"] = "Result"
            for k in results.keys():
                output[k] = f"{np.mean(results[k]):.4f}±{np.std(results[k]):.4f}"
                print(f"{k:5s}: {output[k]}")
            writer.writerow(output)
    else:
        # trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path) # for kitti test
        metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
