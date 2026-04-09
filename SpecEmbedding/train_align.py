import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

from SpecEmbedding.models_align import SpecMolAlignModel, GINEEncoder
from SpecEmbedding.models import SiameseModel
from SpecEmbedding.loss_align import ContrastiveAlignmentLoss
from SpecEmbedding.data.datasets_align import AlignGraphDataset, align_collate_fn

class TrainerAlign:
    """质谱-分子图对齐训练器"""
    def __init__(
        self,
        model: SpecMolAlignModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_dir: str = "./checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.criterion = ContrastiveAlignmentLoss().to(device)
        
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, optimizer, epoch, stage_name):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"[{stage_name}] Epoch {epoch} Training", ascii=True)
        
        for batch in pbar:
            if batch is None:
                continue
                
            spec_mz = batch["spec_mz"].to(self.device)
            spec_intensity = batch["spec_intensity"].to(self.device)
            spec_mask = batch["spec_mask"].to(self.device)
            mol_graph = batch["mol_graph"].to(self.device)
            
            optimizer.zero_grad()
            
            f_spec, f_mol, scale = self.model(spec_mz, spec_intensity, spec_mask, mol_graph)
            loss = self.criterion(f_spec, f_mol, scale)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch, stage_name):
        self.model.eval()
        total_loss = 0
        pbar = tqdm(self.val_loader, desc=f"[{stage_name}] Epoch {epoch} Validation", ascii=True)
        
        for batch in pbar:
            if batch is None:
                continue
                
            spec_mz = batch["spec_mz"].to(self.device)
            spec_intensity = batch["spec_intensity"].to(self.device)
            spec_mask = batch["spec_mask"].to(self.device)
            mol_graph = batch["mol_graph"].to(self.device)
            
            f_spec, f_mol, scale = self.model(spec_mz, spec_intensity, spec_mask, mol_graph)
            loss = self.criterion(f_spec, f_mol, scale)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return total_loss / len(self.val_loader)

    def fit(self, epochs: int, optimizer: torch.optim.Optimizer, scheduler=None, stage_name="Stage", patience=5):
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(optimizer, epoch, stage_name)
            val_loss = self.validate(epoch, stage_name)
            
            logging.info(f"[{stage_name}] Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            if scheduler is not None:
                scheduler.step()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = deepcopy(self.model.state_dict())
                patience_counter = 0 # reset patience
                
                save_path = os.path.join(self.save_dir, f"best_model_{stage_name.replace(' ', '_')}.pth")
                torch.save(best_model_state, save_path)
                logging.info(f"--> Saved best model with Val Loss: {val_loss:.4f} to {save_path}")
            else:
                patience_counter += 1
                logging.info(f"EarlyStopping counter: {patience_counter} out of {patience}")
                if patience_counter >= patience:
                    logging.info("Early stopping triggered.")
                    break
                
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return best_val_loss
