
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def parse_args():
    parser = argparse.ArgumentParser("Original MedFuse on Respire data")

    parser.add_argument("--cohort_csv", type=str, default="data/processed/cohorts/cohort.csv")
    parser.add_argument("--ehr_npz", type=str, default="data/processed/ehr/ehr_final_24h_train_ready/ehr_24h_final_train_ready_current_split.npz")
    parser.add_argument("--output_root", type=str, default=".")
    parser.add_argument("--save_dir", type=str, default="outputs/medfuse")

    parser.add_argument("--sample_col", type=str, default="sample_id")
    parser.add_argument("--image_col", type=str, default="verified_image_path")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--split_col", type=str, default="split")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--debug_n", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=0.0)
    parser.add_argument("--monitor_metric", type=str, default="auprc", choices=["auroc", "auprc"])
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--vision_num_classes", type=int, default=1)

    parser.add_argument("--fusion_type", type=str, default="lstm")
    parser.add_argument("--align", type=float, default=0.0)
    parser.add_argument("--labels_set", type=str, default="respire")
    parser.add_argument("--data_pairs", type=str, default="paired_ehr_cxr")
    parser.add_argument("--mode", type=str, default="train")

    parser.add_argument("--vision-backbone", dest="vision_backbone", type=str, default="resnet34")
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--freeze_cxr_backbone", action="store_true")
    parser.add_argument("--cxr_trainable_scope", type=str, default="classifier", choices=["classifier", "layer4_bn", "layer4_proj", "layer4_last", "layer4", "full"])
    parser.add_argument("--cxr_prior_init", action="store_true")

    parser.add_argument("--load_state", type=str, default="")
    parser.add_argument("--load_state_ehr", type=str, default="")
    parser.add_argument("--load_state_cxr", type=str, default="")
    parser.add_argument("--freeze_loaded_ehr", action="store_true")
    parser.add_argument("--freeze_loaded_cxr", action="store_true")

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    args.load_state = None if str(args.load_state).strip() == "" else args.load_state
    args.load_state_ehr = None if str(args.load_state_ehr).strip() == "" else args.load_state_ehr
    args.load_state_cxr = None if str(args.load_state_cxr).strip() == "" else args.load_state_cxr

    return args


def main():
    args = parse_args()
    seed_everything(args.seed)

    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    medfuse_root = repo_root / "external" / "medfuse_original"

    if not medfuse_root.exists():
        raise FileNotFoundError(f"Missing original MedFuse folder: {medfuse_root}")

    sys.path.insert(0, str(medfuse_root))
    sys.path.insert(0, str(src_root))

    from trainers.fusion_trainer import FusionTrainer
    from respire_transfuse.data.medfuse_respire_adapter import build_respire_medfuse_loaders

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, feature_names, data_summary = build_respire_medfuse_loaders(args)

    args.pos_weight = 0.0

    first = train_loader.dataset[0]
    args.ehr_input_dim = int(first["x"].shape[-1])

    with open(save_dir / "args_used.json", "w") as f:
        json.dump(json_safe(vars(args)), f, indent=2)

    with open(save_dir / "data_summary.json", "w") as f:
        json.dump(json_safe(data_summary), f, indent=2)

    print("=" * 100)
    print("Original MedFuse training on Respire data")
    print("=" * 100)
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass

    batch = next(iter(train_loader))
    x, img, y_ehr, y_cxr, seq_lengths, pairs = batch

    pass
    pass
    pass
    pass
    pass
    pass

    trainer = FusionTrainer(
        train_dl=train_loader,
        val_dl=val_loader,
        test_dl=test_loader,
        args=args,
    )

    if args.dry_run:
        trainer.epoch = 0
        ret = trainer.validate(val_loader)
        print("dry_run_val:", ret)
        return

    trainer.train()

    best_path = Path(args.save_dir) / "best_checkpoint.pth.tar"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(checkpoint["state_dict"])
        print("Loaded best checkpoint for final val/test:", best_path)
    else:
        print("WARNING: best checkpoint not found before final test:", best_path)

    trainer.test()


if __name__ == "__main__":
    main()
