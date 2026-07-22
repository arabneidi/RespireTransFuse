"""Train and evaluate the recurrent MedFuse model configurations.

The trainer handles Uni-EHR, Uni-CXR, joint, and weighted late-fusion objectives,
including paired-sample masking and modality-specific losses. It selects and saves
validation checkpoints, records epoch histories and predictions, and evaluates
the requested output on the validation and held-out test loaders.
"""

from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.fusion import Fusion
from models.ehr_models import LSTM
from models.cxr_models import CXRModels
from .trainer import Trainer
import pandas as pd
import os
import json
import shutil


import numpy as np
from sklearn import metrics
from sklearn.metrics import log_loss, brier_score_loss, precision_recall_curve
from respire_transfuse.utils.epoch_metrics import save_epoch_artifacts

class FusionTrainer(Trainer):
    def __init__(self, 
        train_dl, 
        val_dl, 
        args,
        test_dl=None
        ):

        super(FusionTrainer, self).__init__(args)
        self.epoch = 0 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        self.ehr_model = LSTM(input_dim=args.ehr_input_dim, num_classes=args.num_classes, hidden_dim=args.dim, dropout=args.dropout, layers=args.layers).to(self.device)
        self.cxr_model = CXRModels(self.args, self.device).to(self.device)

        if bool(getattr(self.args, "freeze_cxr_backbone", False)):
            scope = str(getattr(self.args, "cxr_trainable_scope", "classifier"))

            for name, param in self.cxr_model.named_parameters():
                param.requires_grad = False

                if scope == "full":
                    param.requires_grad = True
                elif scope == "classifier":
                    if name.startswith("classifier"):
                        param.requires_grad = True
                elif scope == "layer4_bn":
                    if name.startswith("classifier") or (name.startswith("vision_backbone.layer4") and ".bn" in name):
                        param.requires_grad = True
                elif scope == "layer4_proj":
                    if name.startswith("classifier") or name.startswith("vision_backbone.layer4.0.downsample"):
                        param.requires_grad = True
                elif scope == "layer4_last":
                    if name.startswith("classifier") or name.startswith("vision_backbone.layer4.2"):
                        param.requires_grad = True
                elif scope == "layer4":
                    if name.startswith("classifier") or name.startswith("vision_backbone.layer4"):
                        param.requires_grad = True
                else:
                    raise ValueError(f"Unknown cxr_trainable_scope: {scope}")

            trainable = sum(param.numel() for param in self.cxr_model.parameters() if param.requires_grad)
            total = sum(param.numel() for param in self.cxr_model.parameters())
            print(f"freeze_cxr_backbone: True | cxr_trainable_scope={scope} | trainable_cxr_params={trainable} / {total}")



        self.init_cxr_prior()

        self.model = Fusion(args, self.ehr_model, self.cxr_model ).to(self.device)
        self.init_fusion_method()

        self.loss = nn.BCELoss()

        beta_2 = float(getattr(self.args, 'beta_2', 0.999))
        weight_decay = float(getattr(self.args, 'weight_decay', 0.0))
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"trainable_model_params: {trainable_params} / {total_params}")
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), args.lr, betas=(0.9, beta_2), weight_decay=weight_decay)
        self.load_state()
        print(self.ehr_model)
        pass
        pass
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0
        self.best_score = 0
        self.best_stats = None
        self.best_epoch = None
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99) 
        self.epochs_stats = {'loss train': [], 'loss val': [], 'auroc val': [], 'loss align train': [], 'loss align val': []}
        self.history = []
        self.history_csv = os.path.join(self.args.save_dir, 'history.csv')
        self.history_json = os.path.join(self.args.save_dir, 'history.json')
    
    def init_cxr_prior(self):
        if not bool(getattr(self.args, "cxr_prior_init", False)):
            return

        if not hasattr(self.train_dl.dataset, "df"):
            return

        label_col = getattr(self.train_dl.dataset, "label_col", "label")
        y = self.train_dl.dataset.df[label_col].astype(float).values

        prevalence = float(y.mean())
        prevalence = min(max(prevalence, 1e-5), 1.0 - 1e-5)
        bias = float(np.log(prevalence / (1.0 - prevalence)))

        last_linear = None
        for module in self.cxr_model.classifier.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        if last_linear is None:
            raise RuntimeError("Could not find CXR classifier Linear layer for prior initialization.")

        nn.init.normal_(last_linear.weight, mean=0.0, std=1e-4)
        nn.init.constant_(last_linear.bias, bias)

        print(f"cxr_prior_init: True | prevalence={prevalence:.6f} | bias={bias:.6f}")

    def init_fusion_method(self):

        '''
        for early fusion
        load pretrained encoders and 
        freeze both encoders
        ''' 

        if self.args.load_state_ehr is not None:
            self.load_ehr_pheno(load_state=self.args.load_state_ehr)
        if self.args.load_state_cxr is not None:
            self.load_cxr_pheno(load_state=self.args.load_state_cxr)
        
        if self.args.load_state is not None:
            self.load_state()

        if bool(getattr(self.args, "freeze_loaded_ehr", False)):
            self.freeze(self.model.ehr_model)
            pass

        if bool(getattr(self.args, "freeze_loaded_cxr", False)):
            self.freeze(self.model.cxr_model)
            pass


        if 'uni_ehr' in self.args.fusion_type:
            self.freeze(self.model.cxr_model)
        elif 'uni_cxr' in self.args.fusion_type:
            # Uni-CXR must train only the requested CXR scope.
            self.freeze(self.model)

            scope = str(
                getattr(
                    self.args,
                    "cxr_trainable_scope",
                    "classifier",
                )
            )

            for name, param in self.model.cxr_model.named_parameters():
                if scope == "full":
                    param.requires_grad = True

                elif scope == "classifier":
                    if name.startswith("classifier"):
                        param.requires_grad = True

                elif scope == "layer4_bn":
                    if (
                        name.startswith("classifier")
                        or (
                            name.startswith(
                                "vision_backbone.layer4"
                            )
                            and ".bn" in name
                        )
                    ):
                        param.requires_grad = True

                elif scope == "layer4_proj":
                    if (
                        name.startswith("classifier")
                        or name.startswith(
                            "vision_backbone.layer4.0.downsample"
                        )
                    ):
                        param.requires_grad = True

                elif scope == "layer4_last":
                    if (
                        name.startswith("classifier")
                        or name.startswith(
                            "vision_backbone.layer4.2"
                        )
                    ):
                        param.requires_grad = True

                elif scope == "layer4":
                    if (
                        name.startswith("classifier")
                        or name.startswith(
                            "vision_backbone.layer4"
                        )
                    ):
                        param.requires_grad = True

                else:
                    raise ValueError(
                        "Unknown cxr_trainable_scope: "
                        f"{scope}"
                    )
        elif 'late' in self.args.fusion_type:
            self.freeze(self.model)
        elif 'early' in self.args.fusion_type:
            self.freeze(self.model.cxr_model)
            self.freeze(self.model.ehr_model)
        elif 'lstm' in self.args.fusion_type:
            # self.freeze(self.model.cxr_model)
            # self.freeze(self.model.ehr_model)
            pass

    def compute_loss(self, pred, y):
        return self.loss(pred, y)

    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')

        if (
            bool(
                getattr(
                    self.args,
                    "freeze_cxr_backbone",
                    False,
                )
            )
            and str(
                getattr(
                    self.args,
                    "cxr_trainable_scope",
                    "classifier",
                )
            ) == "classifier"
        ):
            self.model.cxr_model.vision_backbone.eval()

        if bool(getattr(self.args, "freeze_loaded_ehr", False)):
            self.model.ehr_model.eval()

        if bool(getattr(self.args, "freeze_loaded_cxr", False)):
            self.model.cxr_model.eval()

        epoch_loss = 0.0
        epoch_loss_align = 0.0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)

        for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate(self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            img = img.to(self.device)

            output = self.model(x, seq_lengths, img, pairs)

            pred = output[self.args.fusion_type].squeeze()
            pred = pred.float().view(-1)
            y = y.float().view(-1)

            loss = self.compute_loss(pred, y)
            epoch_loss += loss.item()

            if self.args.align > 0.0:
                loss = loss + self.args.align * output['align_loss']
                epoch_loss_align += self.args.align * output['align_loss'].item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float(getattr(self.args, "grad_clip", 5.0)),
            )
            self.optimizer.step()

            outPRED = torch.cat((outPRED, pred.detach()), 0)
            outGT = torch.cat((outGT, y.detach()), 0)

            if i % 100 == 9:
                try:
                    eta = self.get_eta(self.epoch, i)
                except Exception:
                    eta = "n/a"

                denom = max(i + 1, 1)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: 	{self.optimizer.param_groups[0]['lr']:0.4E} loss: 	{epoch_loss/denom:0.5f} loss align {epoch_loss_align/denom:0.4f}")

        denom = max(i + 1, 1)
        train_loss = epoch_loss / denom
        train_align_loss = epoch_loss_align / denom

        y_true = outGT.data.cpu().numpy()
        probs = outPRED.data.cpu().numpy()

        ret = self.computeAUROC(y_true, probs, 'train')
        ret.update(self._binary_extra_metrics(y_true, probs))
        ret['loss'] = float(train_loss)
        ret['loss_align'] = float(train_align_loss)
        ret['labels'] = y_true.reshape(-1)
        ret['probs'] = probs.reshape(-1)

        self.epochs_stats['loss train'].append(train_loss)
        self.epochs_stats['loss align train'].append(train_align_loss)

        return ret


    def validate(
        self,
        dl,
        split="validation",
        save_predictions=False,
        epoch=None,
        step_scheduler=False,
    ):
        print(f"starting {split} epoch {self.epoch}")

        self.model.eval()

        epoch_loss = 0.0
        epoch_loss_align = 0.0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, (
                x,
                img,
                y_ehr,
                y_cxr,
                seq_lengths,
                pairs,
            ) in enumerate(dl):
                y = self.get_gt(y_ehr, y_cxr)

                x = torch.from_numpy(x).float()
                x = Variable(
                    x.to(self.device),
                    requires_grad=False,
                )

                y = Variable(
                    y.to(self.device),
                    requires_grad=False,
                )

                img = img.to(self.device)

                output = self.model(
                    x,
                    seq_lengths,
                    img,
                    pairs,
                )

                pred = output[
                    self.args.fusion_type
                ]

                pred = pred.float().view(-1)
                y = y.float().view(-1)

                loss = self.compute_loss(
                    pred,
                    y,
                )

                epoch_loss += loss.item()

                if self.args.align > 0.0:
                    epoch_loss_align += (
                        output["align_loss"].item()
                    )

                outPRED = torch.cat(
                    (outPRED, pred),
                    0,
                )

                outGT = torch.cat(
                    (outGT, y),
                    0,
                )

        denom = max(i + 1, 1)

        val_loss = epoch_loss / denom
        val_align_loss = epoch_loss_align / denom

        if step_scheduler:
            self.scheduler.step(val_loss)

        print(
            f"{split} "
            f"[{self.epoch:04d} / "
            f"{self.args.epochs:04d}] "
            f"loss: {val_loss:0.5f} "
            f"align: {val_align_loss:0.5f}"
        )

        y_true = outGT.data.cpu().numpy()
        probs = outPRED.data.cpu().numpy()

        ret = self.computeAUROC(
            y_true,
            probs,
            split,
        )

        ret.update(
            self._binary_extra_metrics(
                y_true,
                probs,
            )
        )

        ret["loss"] = float(val_loss)
        ret["loss_align"] = float(
            val_align_loss
        )

        ret["labels"] = y_true.reshape(-1)
        ret["probs"] = probs.reshape(-1)

        if save_predictions:
            self._save_prediction_csv(
                split=split,
                y_true=y_true,
                probs=probs,
                epoch=epoch,
            )

        self.epochs_stats[
            "auroc val"
        ].append(ret["auroc_mean"])

        self.epochs_stats[
            "loss val"
        ].append(val_loss)

        self.epochs_stats[
            "loss align val"
        ].append(val_align_loss)

        return ret

    def _binary_extra_metrics(self, y_true, probs):
        y_true = np.asarray(y_true).reshape(-1).astype(int)
        probs = np.asarray(probs).reshape(-1).astype(float)
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)

        out = {
            "n": int(y_true.shape[0]),
            "prevalence": float(y_true.mean()) if y_true.shape[0] > 0 else None,
            "log_loss": None,
            "brier": None,
            "ece_10": None,
            "mce_10": None,
            "best_f1": None,
            "best_f1_threshold": None,
        }

        if y_true.shape[0] == 0:
            return out

        try:
            out["log_loss"] = float(log_loss(y_true, probs, labels=[0, 1]))
        except Exception:
            out["log_loss"] = None

        try:
            out["brier"] = float(brier_score_loss(y_true, probs))
        except Exception:
            out["brier"] = None

        bin_edges = np.linspace(0.0, 1.0, 11)
        bin_ids = np.digitize(probs, bin_edges[1:-1], right=False)

        ece = 0.0
        mce = 0.0

        for b in range(10):
            mask = bin_ids == b
            if not np.any(mask):
                continue

            conf = float(probs[mask].mean())
            acc = float(y_true[mask].mean())
            gap = abs(acc - conf)
            frac = float(mask.mean())

            ece += frac * gap
            mce = max(mce, gap)

        out["ece_10"] = float(ece)
        out["mce_10"] = float(mce)

        try:
            precision, recall, thresholds = precision_recall_curve(y_true, probs)
            f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-12)
            best_idx = int(np.nanargmax(f1))
            out["best_f1"] = float(f1[best_idx])

            if best_idx == 0:
                out["best_f1_threshold"] = 0.0
            else:
                out["best_f1_threshold"] = float(thresholds[min(best_idx - 1, len(thresholds) - 1)])
        except Exception:
            out["best_f1"] = None
            out["best_f1_threshold"] = None

        return out

    def _save_prediction_csv(self, split, y_true, probs, epoch=None):
        y_true = np.asarray(y_true).reshape(-1).astype(int)
        probs = np.asarray(probs).reshape(-1).astype(float)

        df = pd.DataFrame({
            "row_index": np.arange(len(y_true)),
            "y_true": y_true,
            "prob": probs,
            "pred_0_5": (probs >= 0.5).astype(int),
        })

        if epoch is None:
            out_path = os.path.join(self.args.save_dir, f"{split}_predictions.csv")
        else:
            pred_dir = os.path.join(self.args.save_dir, "epoch_predictions")
            os.makedirs(pred_dir, exist_ok=True)
            out_path = os.path.join(pred_dir, f"{split}_epoch_{int(epoch):03d}.csv")

        df.to_csv(out_path, index=False)
        return out_path

    def _ret_for_json(self, ret):
        out = {}

        for key, value in ret.items():
            if key in ["labels", "probs"]:
                continue

            if isinstance(value, np.ndarray):
                if value.size == 1:
                    out[key] = float(value.reshape(-1)[0])
                else:
                    out[key] = value.tolist()
            elif isinstance(value, (np.integer,)):
                out[key] = int(value)
            elif isinstance(value, (np.floating,)):
                out[key] = float(value)
            elif isinstance(value, (float, int, str, bool)) or value is None:
                out[key] = value

        return out


    def compute_late_fusion(self, y_true, uniout_cxr, uniout_ehr):
        y_true = np.array(y_true)
        predictions_cxr = np.array(uniout_cxr)
        predictions_ehr = np.array(uniout_ehr)
        best_weights = np.ones(y_true.shape[-1])
        best_auroc = 0.0
        weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for class_idx in range(y_true.shape[-1]):
            for weight in weights:
                predictions = (predictions_ehr * best_weights) + (predictions_cxr * (1-best_weights))
                predictions[:, class_idx] = (predictions_ehr[:, class_idx] * weight) + (predictions_cxr[:, class_idx] * 1-weight)
                auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
                auroc_mean = np.mean(np.array(auc_scores))
                if auroc_mean > best_auroc:
                    best_auroc = auroc_mean
                    best_weights[class_idx] = weight
                # predictions = weight * predictions_cxr[]


        predictions = (predictions_ehr * best_weights) + (predictions_cxr * (1-best_weights))
        print(best_weights)

        auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
        ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                            average="micro")
        ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                            average="macro")
        ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                                average="weighted")
        
        # print(np.mean(np.array(auc_scores)

        # print()
        best_stats = {"auc_scores": auc_scores,
                "ave_auc_micro": ave_auc_micro,
                "ave_auc_macro": ave_auc_macro,
                "ave_auc_weighted": ave_auc_weighted,
                "auroc_mean": np.mean(np.array(auc_scores))
                }
        self.print_and_write(best_stats , isbest=True, prefix='late fusion weighted average')

        return best_stats 
    def eval_age(self):

        print('validating ... ')
           
        patiens = pd.read_csv('data/physionet.org/files/mimic-iv-1.0/core/patients.csv')
        subject_ids = np.array([int(item.split('_')[0]) for item in self.test_dl.dataset.ehr_files_paired])

        selected = patiens[patiens.subject_id.isin(subject_ids)]
        start = 18
        copy_ehr = np.copy(self.test_dl.dataset.ehr_files_paired)
        copy_cxr = np.copy(self.test_dl.dataset.cxr_files_paired)
        self.model.eval()
        step = 20
        for i in range(20, 100, step):
            subjects = selected.loc[((selected.anchor_age >= start) & (selected.anchor_age < i + step))].subject_id.values
            indexes = [jj for (jj, subject) in enumerate(subject_ids) if  subject in subjects]
            
            
            self.test_dl.dataset.ehr_files_paired = copy_ehr[indexes]
            self.test_dl.dataset.cxr_files_paired = copy_cxr[indexes]

            print(len(indexes))
            ret = self.validate(self.test_dl)
            print(f"{start}-{i + step} & {len(indexes)} & & & {ret['auroc_mean']:0.3f} & {ret['auprc_mean']:0.3f}")

            self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} val', filename=f'results_test_{start}_{i + step}.txt')

            # print(f"{start}-{i + step} & {len(indexes)} & & & {ret['auroc_mean']:0.3f} & {ret['auprc_mean']:0.3f}")
            # print(f"{start}-{i + 10} & {len(indexes)} & & & {ret['auroc_mean']:0.3f} & {ret['auprc_mean']:0.3f}")
            # self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} age_{start}_{i + 10}_{len(indexes)}', filename='results_test.txt')
            start = i + step
    def test(self):
        print("validating ...")

        self.epoch = 0

        val_ret = self.validate(
            self.val_dl,
            split="val",
            save_predictions=False,
            epoch=None,
            step_scheduler=False,
        )

        self._save_prediction_csv(
            "val",
            val_ret["labels"],
            val_ret["probs"],
            epoch=None,
        )

        self.print_and_write(
            val_ret,
            isbest=True,
            prefix=(
                f"{self.args.fusion_type} val"
            ),
            filename="results_val.txt",
        )

        test_ret = self.validate(
            self.test_dl,
            split="test",
            save_predictions=False,
            epoch=None,
            step_scheduler=False,
        )

        self._save_prediction_csv(
            "test",
            test_ret["labels"],
            test_ret["probs"],
            epoch=None,
        )

        self.print_and_write(
            test_ret,
            isbest=True,
            prefix=(
                f"{self.args.fusion_type} test"
            ),
            filename="results_test.txt",
        )

        metrics_out = {
            "best_epoch": (
                int(self.best_epoch)
                if self.best_epoch is not None
                else None
            ),
            "best_score": float(
                self.best_score
            ),
            "monitor_metric": str(
                getattr(
                    self.args,
                    "monitor_metric",
                    "auroc",
                )
            ).lower(),
            "fusion_type": str(
                self.args.fusion_type
            ),
            "val": self._ret_for_json(
                val_ret
            ),
            "test": self._ret_for_json(
                test_ret
            ),
        }

        metrics_path = os.path.join(
            self.args.save_dir,
            "metrics.json",
        )

        with open(
            metrics_path,
            "w",
        ) as f:
            json.dump(
                metrics_out,
                f,
                indent=2,
            )

        print(
            f"metrics saved: "
            f"{metrics_path}"
        )

        return

    def eval(self):
        # self.eval_age()
        print('validating ... ')
        self.epoch = 0
        self.model.eval()
        # ret = self.validate(self.val_dl)
        # self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} val', filename='results_val.txt')
        # self.model.eval()
        ret = self.validate(self.test_dl)
        self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename='results_test.txt')
        return
    def _history_float(self, value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.reshape(-1)[0])
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return value


    def _positive_negative_gap(self, ret):
        labels = ret.get("labels", None)
        probs = ret.get("probs", None)

        if labels is None or probs is None:
            return None

        y = np.asarray(labels).reshape(-1).astype(int)
        p = np.asarray(probs).reshape(-1).astype(float)

        mask = np.isfinite(y) & np.isfinite(p)
        y = y[mask]
        p = p[mask]

        if y.size == 0:
            return None

        if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
            return None

        if np.nanmin(p) < 0.0 or np.nanmax(p) > 1.0:
            p = 1.0 / (1.0 + np.exp(-p))

        return float(p[y == 1].mean() - p[y == 0].mean())


    def save_history_row(
        self,
        train_ret,
        val_ret,
        is_best,
    ):
        row = {
            "epoch": int(self.epoch),
            "lr": float(
                self.optimizer.param_groups[0]["lr"]
            ),
            "optim_train_loss": self._history_float(
                train_ret.get("loss")
            ),
            "optim_train_bce": self._history_float(
                train_ret.get("loss")
            ),
            "train_loss": self._history_float(
                train_ret.get("loss")
            ),
            "train_bce": self._history_float(
                train_ret.get("loss")
            ),
            "train_align_loss": self._history_float(
                train_ret.get("loss_align")
            ),
            "val_loss": self._history_float(
                val_ret.get("loss")
            ),
            "val_bce": self._history_float(
                val_ret.get("loss")
            ),
            "val_align_loss": self._history_float(
                val_ret.get("loss_align")
            ),
        }

        row.update(
            save_epoch_artifacts(
                save_dir=self.args.save_dir,
                epoch=self.epoch,
                split="train",
                sample_ids=None,
                y_true=train_ret["labels"],
                pred_values=train_ret["probs"],
                n_bins=10,
                save_predictions=False,
            )
        )

        row.update(
            save_epoch_artifacts(
                save_dir=self.args.save_dir,
                epoch=self.epoch,
                split="val",
                sample_ids=None,
                y_true=val_ret["labels"],
                pred_values=val_ret["probs"],
                n_bins=10,
                save_predictions=True,
            )
        )

        monitor_metric = str(
            getattr(
                self.args,
                "monitor_metric",
                "auroc",
            )
        ).lower()

        score_key = (
            "auprc_mean"
            if monitor_metric == "auprc"
            else "auroc_mean"
        )

        row.update(
            {
                "monitor_metric": monitor_metric,
                "current_score": self._history_float(
                    val_ret.get(score_key)
                ),
                "best_score": float(
                    self.best_score
                ),
                "best_epoch": (
                    int(self.best_epoch)
                    if self.best_epoch is not None
                    else None
                ),
                "is_best": bool(is_best),
            }
        )

        self.history.append(row)

        pd.DataFrame(
            self.history
        ).to_csv(
            self.history_csv,
            index=False,
        )

        with open(
            self.history_json,
            "w",
        ) as f:
            json.dump(
                self.history,
                f,
                indent=2,
            )

    def train(self):
        print(
            f"running for fusion_type "
            f"{self.args.fusion_type}"
        )

        for stale_name in [
            "history.csv",
            "history.json",
            "calibration_bins_10_by_epoch.csv",
            "adaptive_calibration_bins_10_by_epoch.csv",
        ]:
            stale_path = os.path.join(
                self.args.save_dir,
                stale_name,
            )

            if os.path.exists(stale_path):
                os.remove(stale_path)

        prediction_dir = os.path.join(
            self.args.save_dir,
            "epoch_predictions",
        )

        if os.path.isdir(prediction_dir):
            shutil.rmtree(prediction_dir)

        self.history = []

        end_epoch = (
            int(self.args.epochs) + 1
        )

        for self.epoch in range(
            int(self.start_epoch),
            end_epoch,
        ):
            self.model.train()

            train_ret = self.train_epoch()

            val_ret = self.validate(
                self.val_dl,
                split="val",
                save_predictions=False,
                epoch=self.epoch,
                step_scheduler=True,
            )

            monitor_metric = str(
                getattr(
                    self.args,
                    "monitor_metric",
                    "auroc",
                )
            ).lower()

            if monitor_metric == "auprc":
                current_score = float(
                    val_ret["auprc_mean"]
                )
            else:
                current_score = float(
                    val_ret["auroc_mean"]
                )

            is_best = (
                current_score
                > float(self.best_score)
            )

            if is_best:
                self.best_score = current_score
                self.best_auroc = float(
                    val_ret["auroc_mean"]
                )

                self.best_stats = val_ret
                self.best_epoch = int(
                    self.epoch
                )

                self.save_checkpoint()

                self.print_and_write(
                    val_ret,
                    isbest=True,
                )

                self.patience = 0
            else:
                self.print_and_write(
                    val_ret,
                    isbest=False,
                )

                self.patience += 1

            self.save_checkpoint(
                prefix="last"
            )

            self.save_history_row(
                train_ret,
                val_ret,
                is_best,
            )

            self.plot_stats(
                key="loss",
                filename="loss.pdf",
            )

            self.plot_stats(
                key="auroc",
                filename="auroc.pdf",
            )

            if self.patience >= int(
                self.args.patience
            ):
                print(
                    f"early stopping at epoch "
                    f"{self.epoch}"
                )
                break

        if self.best_stats is not None:
            self.print_and_write(
                self.best_stats,
                isbest=True,
            )
