import json
import logging
import math
import os
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from SpecEmbedding.data.datasets_adduct import AdductAlignmentBatch, forward_alignment_batch, validate_bound_adducts
from SpecEmbedding.loss_align import ContrastiveAlignmentLoss
from SpecEmbedding.models_align import SpecMolAlignModel
from SpecEmbedding.utils.training_resources import EpochResources


class TrainerAlign:
    """质谱-分子图对齐训练器"""
    def __init__(
        self,
        model: SpecMolAlignModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_dir: str = "./checkpoints",
        retrieval_validator=None,
        record_resources=False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.criterion = ContrastiveAlignmentLoss().to(device)
        self.stage_summaries = {}
        self.expected_epoch_counts = None
        self.epoch_counts = []
        self.retrieval_validator = retrieval_validator
        self.record_resources = record_resources
        
        os.makedirs(self.save_dir, exist_ok=True)

    def training_batch_loss(self, batch):
        (f_spec, f_mol, scale), labels, count = forward_alignment_batch(self.model, batch, self.device)
        return self.criterion(f_spec, f_mol, scale, labels), count

    def train_epoch(self, optimizer, epoch, stage_name):
        self.model.train()
        total_loss = 0
        seen = 0
        pbar = tqdm(self.train_loader, desc=f"[{stage_name}] Epoch {epoch} Training", ascii=True)
        
        for batch in pbar:
            optimizer.zero_grad()
            loss, query_count = self.training_batch_loss(batch)
            if self.expected_epoch_counts is not None and not torch.isfinite(loss):
                raise RuntimeError("Non-finite formal alignment training loss")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            seen += query_count
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        if self.expected_epoch_counts is not None:
            if seen != self.expected_epoch_counts["train"]:
                raise RuntimeError(f"Incomplete alignment epoch: trained {seen} spectra")
            self.epoch_counts.append({"stage": stage_name, "epoch": epoch, "train": seen})
            batch_audit = getattr(self.train_loader.batch_sampler, "last_audit", None)
            if batch_audit is not None:
                if batch_audit["queries"] != seen or batch_audit["unique_queries"] != seen:
                    raise RuntimeError("Training sampler coverage differs from observed query count")
                self.epoch_counts[-1]["batching"] = dict(batch_audit)
            logging.info("Formal alignment epoch %s trained all %s spectra", epoch, seen)
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch, stage_name):
        self.model.eval()
        total_loss = 0
        seen = 0
        metadata = getattr(self.val_loader.dataset, 'spectrum_metadata', None)
        adduct_queries = []
        pbar = tqdm(self.val_loader, desc=f"[{stage_name}] Epoch {epoch} Validation", ascii=True)
        
        for batch in pbar:
            if metadata is not None:
                if not isinstance(batch, AdductAlignmentBatch):
                    raise ValueError('Validation lost its bound adduct input')
                validate_bound_adducts(metadata, batch.raw_query_indices, batch.adduct_ids)
                adduct_queries.extend(batch.raw_query_indices.tolist())
            elif isinstance(batch, AdductAlignmentBatch):
                raise ValueError('Unbound validation adduct input')
            (f_spec, f_mol, scale), labels, count = forward_alignment_batch(self.model, batch, self.device)
            loss = self.criterion(f_spec, f_mol, scale, labels)
            if self.expected_epoch_counts is not None and not torch.isfinite(loss):
                raise RuntimeError("Non-finite formal alignment validation loss")
            
            total_loss += loss.item()
            seen += count
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        if metadata is not None and sorted(adduct_queries) != metadata.raw_query_indices.tolist():
            raise RuntimeError('Adduct validation did not visit each eligible query exactly once')
        if self.expected_epoch_counts is not None:
            if seen != self.expected_epoch_counts["val"]:
                raise RuntimeError(f"Incomplete alignment validation: evaluated {seen} spectra")
            self.epoch_counts[-1]["val"] = seen
            logging.info("Formal alignment epoch %s validated all %s spectra", epoch, seen)
        return total_loss / len(self.val_loader)

    def fit(self, epochs: int, optimizer: torch.optim.Optimizer, scheduler=None, stage_name="Stage", patience=5):
        best_val_loss = float('inf')
        best_epoch = None
        best_model_state = None
        patience_counter = 0
        stop_epoch = 0
        early_stopped = False
        validator = getattr(self, "retrieval_validator", None)
        record_resources = getattr(self, "record_resources", False)
        best_retrieval = None
        retrieval_history = []
        frontier = []
        resource_profiles = []
        
        for epoch in range(1, epochs + 1):
            stop_epoch = epoch
            resources = EpochResources(self.model, self.device) if record_resources else None
            train_loss = self.train_epoch(optimizer, epoch, stage_name)
            if resources is not None:
                resources.phase("train")
            val_loss = self.validate(epoch, stage_name)
            if resources is not None:
                resources.phase("contrastive_validation")
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise RuntimeError("Non-finite alignment epoch loss; refusing checkpoint selection")
            retrieval = validator(self.model, self.device, epoch, stage_name) if validator is not None else None
            if resources is not None:
                resources.phase("retrieval_validation")
            if retrieval is not None:
                metrics = ("top1", "top5", "top10", "top20", "mrr")
                if any(not math.isfinite(retrieval[key]) or not 0 <= retrieval[key] <= 1 for key in metrics):
                    raise RuntimeError("Invalid retrieval metrics; refusing checkpoint selection")
                record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **retrieval}
                vector = tuple(retrieval[key] for key in metrics)
                dominated = any(all(item["metrics"][key] >= retrieval[key] for key in metrics) for item in frontier)
                if not dominated:
                    frontier = [item for item in frontier if not all(retrieval[key] >= item["metrics"][key] for key in metrics)]
                    # Keep earlier candidate files as provenance even when they leave the frontier.
                    candidate_path = Path(self.save_dir) / f"candidate_{stage_name}_epoch{epoch:03d}.pth"
                    torch.save(self.model.state_dict(), candidate_path)
                    frontier.append({"epoch": epoch, "checkpoint": candidate_path.name,
                                     "metrics": dict(zip(metrics, vector, strict=True))})
                    record["candidate_checkpoint"] = candidate_path.name
                retrieval_history.append(record)
            
            logging.info(f"[{stage_name}] Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            if scheduler is not None:
                scheduler.step()
                
            improved = (val_loss < best_val_loss if retrieval is None else
                        best_retrieval is None or (retrieval["top1"], retrieval["mrr"]) > (best_retrieval["top1"], best_retrieval["mrr"]))
            if improved:
                best_val_loss = val_loss
                best_retrieval = retrieval
                best_epoch = epoch
                best_model_state = deepcopy(self.model.state_dict())
                patience_counter = 0 # reset patience
                
                save_path = os.path.join(self.save_dir, f"best_model_{stage_name.replace(' ', '_')}.pth")
                torch.save(best_model_state, save_path)
                logging.info("--> Saved best model: val_loss=%.4f retrieval=%s to %s", val_loss, retrieval, save_path)
            else:
                patience_counter += 1
                logging.info(f"EarlyStopping counter: {patience_counter} out of {patience}")
                if patience_counter >= patience:
                    logging.info("Early stopping triggered.")
                    early_stopped = True
            if resources is not None:
                observed = self.epoch_counts[-1] if self.expected_epoch_counts is not None else None
                profile = {"stage": stage_name, "epoch": epoch, **resources.finish(observed)}
                directory = Path(self.save_dir) / "resources"
                directory.mkdir(exist_ok=True)
                path = directory / f"{stage_name}_epoch{epoch:03d}.json"
                with path.open("x") as handle:
                    json.dump(profile, handle, indent=2, allow_nan=False)
                    handle.write("\n")
                resource_profiles.append(profile)
                logging.info("Epoch resource measurement: %s", json.dumps(profile, allow_nan=False))
            if early_stopped:
                break
                
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.stage_summaries[stage_name] = {
            "metric_for_best": "validation_top1_then_mrr" if validator is not None else "validation_contrastive_loss",
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "stop_epoch": stop_epoch,
            "early_stopped": early_stopped,
            "configured_epochs": epochs,
            "patience": patience,
        }
        if validator is not None:
            self.stage_summaries[stage_name].update(best_retrieval=best_retrieval,
                                                   retrieval_history=retrieval_history, pareto_frontier=frontier)
        if record_resources:
            self.stage_summaries[stage_name]["resource_profiles"] = resource_profiles
            
        return best_val_loss
