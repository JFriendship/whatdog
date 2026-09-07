from .data_ingestion import WhatdogDataModule
from .model import WhatdogResNet18
import argparse
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


def parse_args():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    TRAINING_DIR = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Whatdog Training")

    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "data" / "Images")
    parser.add_argument("--annotations-dir", type=Path, default=PROJECT_ROOT / "data" / "Annotation")
    parser.add_argument("--output-dir", type=Path, default=TRAINING_DIR / "artifacts")

    parser.add_argument("--num_classes", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=24)

    parser.add_argument("--accelerator", choices=["auto", "cpu", "mps", "gpu"], default="auto")

    parser.add_argument("--resume-from", type=Path)

    return parser.parse_args()

def main():
    args = parse_args()

    L.seed_everything(args.seed, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="whatdog-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False
    )

    logger = CSVLogger(save_dir=args.output_dir, name="training_logs")

    data_module = WhatdogDataModule(
        images_dir=str(args.images_dir), 
        annotations_dir=str(args.annotations_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = WhatdogResNet18(num_classes=args.num_classes, learning_rate=args.learning_rate)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator, 
        devices="auto", 
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        default_root_dir=str(args.output_dir),
        deterministic=True
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from if args.resume_from else None)

if __name__ == "__main__":
    main()