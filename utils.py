import os
import sys
import torch
import logging
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def set_seed(seed=42):
    """设置随机种子以确保实验的可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to: {seed}")

def get_git_hash():
    """获取当前 git commit 的简短 hash"""
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return ""

def load_config(config_path="config.yaml"):
    """从 YAML 文件中加载配置参数"""
    if not os.path.exists(config_path):
        return {}
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def is_valid_smiles(smiles):
    """
    使用 RDKit 校验 SMILES 的合法性。
    1. 尝试解析 SMILES。
    2. 尝试进行化学规范化 (Sanitization)。
    """
    if not smiles or smiles.upper() in ['N/A', 'NA']:
        return False
    
    try:
        from rdkit import Chem, rdBase
        # 禁用 RDKit 日志以保持控制台整洁
        rdBase.DisableLog('rdApp.*')
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # 显式尝试 Sanitization，捕获如化合价错误等问题
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False

def setup_logger(save_dir, log_name="train.log"):
    """配置全局日志记录器"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, log_name)
    
    logger = logging.getLogger(save_dir + log_name) # 保证唯一性
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # stream_handler = logging.StreamHandler(sys.stdout)
    # stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)
    return logger

def plot_losses(history, save_path, title='Training and Validation Loss'):
    """根据历史记录生成损失曲线图"""
    if not history: return
    epochs = [h['epoch'] + 1 for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
