import argparse
from datetime import datetime
import gin
from loguru import logger
from torch.utils.data import DataLoader

from utils.common import set_random_seed
from dataset.ray_dataset import RayDataset, ray_collate
from neural_field.model import get_model
from trainer import Trainer


@gin.configurable()
def main(
    seed: int = 42,
    num_workers: int = 0,
    train_split: str = "train",
    stages: str = "train_eval",
    batch_size: int = 16,
    model_name="Tri-MipRF",
):
    stages = list(stages.split("_"))
    set_random_seed(seed)

    logger.info("==> Init dataloader ...")
    train_dataset = RayDataset(split=train_split)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=ray_collate,
        pin_memory=True,
        worker_init_fn=None,
        pin_memory_device='cuda',
        prefetch_factor=2,
    )
    test_dataset = RayDataset(split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=None,
        pin_memory_device='cuda',
    )

    logger.info("==> Init model ...")
    model = get_model(model_name=model_name)(aabb=train_dataset.aabb)
    logger.info(model)

    logger.info("==> Init trainer ...")
    trainer = Trainer(model, train_loader, eval_loader=test_loader)
    if "train" in stages:
        trainer.fit()
    if "eval" in stages:
        if "train" not in stages:
            trainer.load_ckpt()
        trainer.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )
    args = parser.parse_args()

    ginbs = []
    if args.ginb:
        ginbs.extend(args.ginb)
    gin.parse_config_files_and_bindings(args.ginc, ginbs, finalize_config=False)

    exp_name = gin.query_parameter("Trainer.exp_name")
    exp_name = "%s/%s/%s/%s" % (
        gin.query_parameter("RayDataset.scene_type"),
        gin.query_parameter("RayDataset.scene"),
        gin.query_parameter("get_model.model_name"),
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if exp_name is None
        else exp_name,
    )
    gin.bind_parameter("Trainer.exp_name", exp_name)
    gin.finalize()
    main()
