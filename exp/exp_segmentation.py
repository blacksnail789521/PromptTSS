import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from einops import rearrange
from collections import defaultdict
from typing import DefaultDict

from dataset_loader.dataset_loader import load_dataloader, InterleavedDataLoader
from models.PatchTST import PatchTST
from models.DeepConvLSTM import DeepConvLSTM
from models.PromptTSS import PromptTSS
from models.iTransformer import iTransformer
from models.PrecTime import PrecTime
from models.U_Time import U_Time
from models.MS_TCN2 import MS_TCN2
from models.MultipleGranularityModel import MultiGranularityModel
from utils.visual import visualize_metric
from utils.tools import EarlyStopping
import random
import time

MODEL_MAP = {
    "PatchTST": PatchTST,
    "PrecTime": PrecTime,
    "MS-TCN++": MS_TCN2,
    "U-Time": U_Time,
    "DeepConvLSTM": DeepConvLSTM,
    "iTransformer": iTransformer,
    "PromptTSS": PromptTSS,
}


class ReshapedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(ReshapedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, y_pred, y):
        # Reshape y_pred from (B, T, K) to (B * T, K), y from (B, T) to (B * T)
        y_pred_reshaped = rearrange(y_pred, "B T K -> (B T) K")
        y_reshaped = rearrange(y, "B T -> (B T)")

        # Calculate and return the loss
        return self.criterion(y_pred_reshaped, y_reshaped)


class Exp_Segmentation(object):
    def __init__(self, args) -> None:
        self.args = args
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._get_data()
        self._get_model()
        self._set_criterion()
        self._set_optimizer()
        self._set_early_stopping()

    def _get_data(self):
        # Load data
        self.train_loader, self.val_loader, self.test_loader, class_distributions = (
            load_dataloader(self.args)
        )

    def _get_model(self):
        # Dynamically get the model class based on the model name
        ModelClass = MODEL_MAP.get(self.args.model_name, None)
        if ModelClass is None:
            raise ValueError(f"Unknown model name: {self.args.model_name}")

        # If using PromptTSS, we do NOT use MultiGranularityModel
        if self.args.model_name == "PromptTSS":
            self.model = ModelClass(self.args).to(self.args.device)
        else:
            # Use MultiGranularityModel for baselines that do NOT support multiple granularity
            self.model = MultiGranularityModel(self.args, model_class=ModelClass).to(
                self.args.device
            )
            assert (
                self.args.num_iter_train == 1
            ), "Only PromptTSS supports multiple iterations"

    def _set_criterion(self):
        self.criterion = ReshapedCrossEntropyLoss()

    def _set_optimizer(self):
        # optimizer
        self.optimizer = getattr(optim, self.args.optim)(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        # scheduler
        if self.args.lr_scheduler != "none":
            lr_scheduler_params = self.args.lr_scheduler_params[self.args.lr_scheduler]
            if self.args.lr_scheduler == "CyclicLR":
                lr_scheduler_params["base_lr"] = self.args.learning_rate
                lr_scheduler_params["cycle_momentum"] = (
                    True if self.args.optim == "SGD" else False
                )
            elif self.args.lr_scheduler == "OneCycleLR":
                lr_scheduler_params["steps_per_epoch"] = len(self.train_loader)
                lr_scheduler_params["epochs"] = self.args.epochs
            self.scheduler = getattr(optim.lr_scheduler, self.args.lr_scheduler)(
                self.optimizer, **lr_scheduler_params
            )

    def _set_early_stopping(self):
        self.early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=self.args.delta
        )

    def create_prompts(self, y, n_min, n_max, prompt_types=["label", "boundary"]):
        B, T = y.shape  # Batch size, time steps
        n_prompts = random.randint(n_min, n_max)  # Sample prompt count

        # Compute boundary tensor once
        boundary = (y[:, 1:] != y[:, :-1]).float()  # Compute boundaries
        boundary = torch.cat(
            [torch.zeros(B, 1).to(y.device), boundary], dim=1
        )  # Pad boundary

        prompts = {}

        for _ in range(n_prompts):
            # Randomly choose a timestamp
            time_index = random.randint(0, T - 1)  # Single timestamp for this prompt

            # Randomly choose prompt type (Label or Boundary)
            prompt_type = random.choice(prompt_types)

            if prompt_type == "boundary":
                # Directly use the boundary value for the selected timestamp
                value = boundary[:, time_index].tolist()
                key = (prompt_type, time_index)  # No aspect for boundary prompts

            elif prompt_type == "label":
                # Randomly decide label prompt aspect
                prompt_aspect = random.choice(["correct", "incorrect"])
                if prompt_aspect == "correct":
                    # Correct Label Prompt: one-hot (value is scalar per batch)
                    value = [
                        y[b, time_index].item() for b in range(B)
                    ]  # Ensure batch order
                else:
                    # Incorrect Label Prompt: multi-hot but only one sampled value (list of lists)
                    incorrect_labels = [
                        [
                            random.choice(
                                [
                                    k
                                    for k in range(self.args.K)
                                    if k != y[b, time_index].item()
                                ]
                            )
                        ]
                        for b in range(B)
                    ]  # Single incorrect label per batch, stored as a list
                    value = incorrect_labels

                key = (
                    prompt_type,
                    prompt_aspect,
                    time_index,
                )  # Include aspect for label prompts

            else:
                raise ValueError(f"Unknown prompt type: {prompt_type}")

            # Add to dictionary
            if key in prompts:
                # Handle merging only for (label, incorrect)
                if prompt_type == "label" and prompt_aspect == "incorrect":
                    existing_values = prompts[key]
                    prompts[key] = [
                        list(set(existing_values[b] + value[b])) for b in range(B)
                    ]
                # For all other cases, skip merging (replace behavior)
                continue
            else:
                # Create a new prompt entry
                prompts[key] = value

        return prompts

    def combine_prompts(self, existing_prompts, new_prompts):
        combined_prompts = existing_prompts.copy()

        for key, new_value in new_prompts.items():
            if key in combined_prompts:
                # Handle merging only for (label, incorrect)
                if len(key) == 3:  # Check for label prompt (keys with aspect)
                    prompt_type, prompt_aspect, _ = key
                    if prompt_type == "label" and prompt_aspect == "incorrect":
                        # Merge incorrect label prompts (multi-hot)
                        existing_values = combined_prompts[key]
                        combined_prompts[key] = [
                            list(set(existing_values[b] + new_value[b]))
                            for b in range(len(existing_values))
                        ]
                    else:
                        # For other label prompts, replace the existing prompt
                        combined_prompts[key] = new_value
                else:
                    # For boundary prompts, replace the existing prompt
                    combined_prompts[key] = new_value
            else:
                # Add new prompt to the dictionary
                combined_prompts[key] = new_value

        return combined_prompts

    def iter_train(self) -> dict[str, DefaultDict[str, list[float]]]:
        # Automatic Mixed Precision (some op. are fp32, some are fp16)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)  # type: ignore

        # * Train the model
        metrics = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list),
        }

        for epoch in range(self.args.epochs):
            self.model.train()
            train_losses = []

            iter_data = (
                tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch + 1}/{self.args.epochs}, Training Loss: {0}",
                )
                if self.args.use_tqdm
                else self.train_loader
            )
            for batch_idx, (x, y, timestamp, df_id, granularity_level) in enumerate(
                iter_data
            ):
                x = x.float().to(self.args.device)
                y = y.long().to(self.args.device)

                prompts = {}  # Reset prompts at the start of each batch

                for iter_num in range(self.args.num_iter_train):

                    # ? 1. Zero grad
                    self.optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                        # ? 2. Call the model
                        new_prompts = self.create_prompts(
                            y, self.args.n_min, self.args.n_max
                        )
                        prompts = self.combine_prompts(
                            prompts, new_prompts
                        )  # Add the new prompts
                        if self.args.model_name == "PromptTSS":
                            y_pred = self.model(x, prompts)
                        else:
                            y_pred = self.model(
                                x, prompts, granularity_level[0].item()
                            )  # Remember that all granularity levels are the same in the same batch

                        # ? 3. Calculate loss
                        loss = self.criterion(y_pred, y)
                        # # Check for NaN loss
                        # if torch.isnan(loss):
                        #     print(
                        #         f"Warning: Loss is NaN at batch {batch_idx}, iter {iter_num}. Skipping this batch."
                        #     )
                        #     # Free up memory
                        #     del x, y, y_pred, loss
                        #     torch.cuda.empty_cache()
                        #     continue
                        # Check for NaN loss values
                        if torch.isnan(loss).any():
                            valid_loss_mask = ~torch.isnan(loss)

                            if valid_loss_mask.sum() == 0:
                                print(
                                    f"Warning: All samples in batch {batch_idx} at iter {iter_num} have NaN loss. "
                                    "Skipping this whole batch."
                                )
                                del y_pred, loss
                                # del x, y, y_pred, loss
                                torch.cuda.empty_cache()
                                continue

                            print(
                                f"Warning: {valid_loss_mask.numel() - valid_loss_mask.sum().item()} samples in batch "
                                f"{batch_idx} at iter {iter_num} have NaN loss. Removing these samples."
                            )

                            # Keep only valid losses
                            loss = loss[valid_loss_mask]
                            # x = x[valid_loss_mask]
                            # y = y[valid_loss_mask]
                            # y_pred = y_pred[valid_loss_mask]
                        train_losses.append(loss.item())

                    # ? 4. Backward
                    scaler.scale(loss).backward()  # type: ignore
                    scaler.step(self.optimizer)
                    scaler.update()

                    if self.args.use_tqdm:
                        iter_data.set_description(  # type: ignore
                            f"Epoch {epoch + 1}/{self.args.epochs}, Training Loss: {np.mean(train_losses)}"
                        )

            # * At the end of each epoch, we get all the metrics
            print(
                f"Epoch {epoch + 1}/{self.args.epochs}, Calculating train_metrics ..."
            )
            # train_metrics = self.get_metrics(self.train_loader)
            train_metrics = self.get_metrics_ind(self.train_loader)
            for metric_name, metric_value in train_metrics.items():
                metrics["train"][metric_name].append(metric_value)
            print(f"Epoch {epoch + 1}/{self.args.epochs}, Calculating val_metrics ...")
            # val_metrics = self.get_metrics(self.val_loader)
            val_metrics = self.get_metrics_ind(self.val_loader)
            for metric_name, metric_value in val_metrics.items():
                metrics["val"][metric_name].append(metric_value)
            # test_metrics = self.get_metrics(self.test_loader)
            print(f"Epoch {epoch + 1}/{self.args.epochs}, Calculating test_metrics ...")
            test_metrics = self.get_metrics_ind(self.test_loader)
            for metric_name, metric_value in test_metrics.items():
                metrics["test"][metric_name].append(metric_value)

            # * Show metrics for all the previous epochs
            visualize_metric(metrics, mode="table")
            visualize_metric(metrics, mode="plot")

            # * Early stopping
            self.early_stopping(
                val_metrics["loss"], self.model, self.args.checkpoint_saving_path
            )
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            # * Learning rate scheduler
            if self.args.lr_scheduler != "none":
                previous_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1}/{self.args.epochs}, Learning Rate: {previous_lr} -> {current_lr}"
                    f" (lr_scheduler: {self.args.lr_scheduler})"
                )

        return metrics

    def calculate_segmentation_metrics(
        self,
        trues: list[np.ndarray],
        preds: list[np.ndarray],
        return_kappa: bool = False,
    ) -> tuple[float, float, float | None]:
        acc_scores = []
        f1_scores = []
        kappa_scores = []

        # Calculate metrics for each pair of true and predicted labels
        for i, (true, pred) in enumerate(zip(trues, preds)):
            acc_scores.append(accuracy_score(true, pred))
            f1_scores.append(f1_score(true, pred, average="macro"))
            if return_kappa:
                kappa = cohen_kappa_score(true, pred)
                # Log problematic cases
                if np.isnan(kappa):
                    print(f"NaN Kappa detected in segment {i}")
                    print(
                        f"Original True labels: {true}, Original Predicted labels: {pred}"
                    )
                    print(
                        f"True labels: {np.unique(true)}, Predicted labels: {np.unique(pred)}"
                    )
                    print(
                        f"Variance in True labels: {np.var(true)}, Variance in Predicted labels: {np.var(pred)}"
                    )
                kappa_scores.append(kappa)

        # Calculate the average accuracy, F1 score, and Cohen's kappa
        average_acc = float(np.mean(acc_scores))
        average_f1 = float(np.mean(f1_scores))
        if return_kappa:
            average_kappa = float(np.mean(kappa_scores))
            return average_acc, average_f1, average_kappa
        else:
            return average_acc, average_f1, None

    def calculate_clustering_metrics(
        self, trues: list[np.ndarray], preds: list[np.ndarray]
    ) -> tuple[float, float]:
        ari_scores = []
        nmi_scores = []

        # Calculate metrics for each pair of true and predicted labels
        for true, pred in zip(trues, preds):
            ari = adjusted_rand_score(true, pred)
            nmi = normalized_mutual_info_score(true, pred)
            ari_scores.append(ari)
            nmi_scores.append(nmi)

        # Calculate the average ARI and NMI
        average_ari = float(np.mean(ari_scores))
        average_nmi = float(np.mean(nmi_scores))

        return average_ari, average_nmi

    def get_metrics(
        self, data_loader: InterleavedDataLoader, add_clustering_metrics: bool = True
    ) -> dict[str, float]:

        # Nested dictionary to store lists of predictions for each timestamp in each sequence
        seq_ts_trues = defaultdict(dict)
        seq_ts_preds = defaultdict(lambda: defaultdict(list))
        total_loss = 0
        num_samples = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (x, y, timestamp, sequence_id, _) in enumerate(
                tqdm(data_loader) if self.args.use_tqdm else data_loader
            ):
                x = x.float().to(self.args.device)
                y = y.long().to(self.args.device)

                # ? 1. Zero grad
                pass

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                    # ? 2. Call the model
                    prompts = self.create_prompts(
                        y, self.args.n_min_test, self.args.n_max_test
                    )
                    y_pred = self.model(x, prompts)

                    # ? 3. Calculate loss
                    loss = self.criterion(y_pred, y)
                    total_loss += loss.item()
                    num_samples += len(x)

                true = y.detach().cpu().numpy()  # (B, T)
                pred = (
                    torch.argmax(y_pred, dim=-1).detach().cpu().numpy()
                )  # (B, T, K) -> (B, T)

                for element_idx in range(
                    len(timestamp)
                ):  # Iterating over the batch dimension
                    seq_id = int(sequence_id[element_idx])
                    for time_idx in range(
                        len(timestamp[element_idx])
                    ):  # Iterating over the timestamp dimension within each batch
                        ts = int(timestamp[element_idx][time_idx])
                        t = true[element_idx][time_idx]
                        p = pred[element_idx][time_idx]
                        if ts not in seq_ts_trues[seq_id]:
                            seq_ts_trues[seq_id][ts] = t
                        seq_ts_preds[seq_id][ts].append(p)

        # Apply majority vote and flatten the data for metric calculation
        seq_trues = {}
        seq_preds = {}
        for seq_id in seq_ts_preds.keys():
            # Sort timestamps for each sequence to maintain temporal order
            timestamps = sorted(seq_ts_preds[seq_id].keys())
            true = []
            pred = []

            for ts in timestamps:
                # Majority vote for predictions at this timestamp
                most_common_pred = np.bincount(seq_ts_preds[seq_id][ts]).argmax()
                # Store the true value and the majority-voted prediction
                true.append(seq_ts_trues[seq_id][ts])
                pred.append(most_common_pred)

            # Convert the lists for this sequence into numpy arrays
            seq_trues[seq_id] = np.array(true)
            seq_preds[seq_id] = np.array(pred)

        final_trues = [seq_trues[seq_id] for seq_id in sorted(seq_trues.keys())]
        final_preds = [seq_preds[seq_id] for seq_id in sorted(seq_preds.keys())]

        loss = total_loss / num_samples
        acc, mf1, kappa = self.calculate_segmentation_metrics(final_trues, final_preds)
        metrics = {"loss": loss, "acc": acc, "mf1": mf1, "kappa": kappa}
        if metrics["kappa"] is None:
            del metrics["kappa"]
        if add_clustering_metrics:
            ari, nmi = self.calculate_clustering_metrics(final_trues, final_preds)
            metrics["ari"] = ari
            metrics["nmi"] = nmi

        return metrics

    # Individual metrics (use this in the paper)
    def get_metrics_ind(
        self,
        data_loader: InterleavedDataLoader,
        add_clustering_metrics: bool = True,
        add_inference_time: bool = False,
    ) -> dict[str, float]:

        total_preds = []
        total_trues = []
        total_loss = 0
        total_samples = 0

        inference_times = []  # List to store inference time for each batch

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (x, y, timestamp, sequence_id, _) in enumerate(
                tqdm(data_loader) if self.args.use_tqdm else data_loader
            ):
                x = x.float().to(self.args.device)
                y = y.long().to(self.args.device)

                # ? 1. Zero grad
                pass

                # Record start time before inference block
                start_time = time.perf_counter()

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                    # ? 2. Call the model
                    prompts = self.create_prompts(
                        y,
                        self.args.n_min_test,
                        self.args.n_max_test,
                        prompt_types=["label", "boundary"],
                        # prompt_types=["label"],
                        # prompt_types=["boundary"],
                    )
                    y_pred = self.model(x, prompts)

                    # ? 3. Calculate loss
                    loss = self.criterion(y_pred, y)
                    total_loss += loss.item()
                    total_samples += len(x)

                # Record end time after inference block
                end_time = time.perf_counter()

                # Compute inference time for this batch (in seconds)
                batch_inference_time = end_time - start_time
                inference_times.append(batch_inference_time)

                pred = (
                    torch.argmax(y_pred, dim=-1).detach().cpu().numpy()
                )  # (B, T, K) -> (B, T)
                true = y.detach().cpu().numpy()  # (B, T)

                total_preds.extend(pred)
                total_trues.extend(true)

        assert total_samples == len(total_preds) == len(total_trues)
        loss = total_loss / total_samples
        # total_preds = np.array(total_preds).flatten()  # (N, T) -> (N * T)
        # total_trues = np.array(total_trues).flatten()  # (N, T) -> (N * T)
        total_preds = np.concatenate(
            np.array(total_preds)
        )  # Work for both same and different lengths
        total_trues = np.concatenate(
            np.array(total_trues)
        )  # Work for both same and different lengths
        acc, mf1, kappa = self.calculate_segmentation_metrics(
            [total_trues], [total_preds], return_kappa=False
        )
        metrics = {"loss": loss, "acc": acc, "mf1": mf1, "kappa": kappa}
        if metrics["kappa"] is None:
            del metrics["kappa"]
        if add_clustering_metrics:
            ari, nmi = self.calculate_clustering_metrics([total_trues], [total_preds])
            metrics["ari"] = ari
            metrics["nmi"] = nmi

        if add_inference_time:
            # Add average inference time per batch to the metrics dictionary
            avg_inference_time = np.mean(inference_times)
            metrics["avg_inference_time_per_batch"] = avg_inference_time
            print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")

        return metrics
