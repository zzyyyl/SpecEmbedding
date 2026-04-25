@echo off
setlocal

:: 设置数据路径
set DATA_PATH=data/MassBank/MassBank.msp
set PRETRAINED_CKPT=checkpoints/model.ckpt
set SAVE_DIR=checkpoints_align/local_massbank

echo ========================================================
echo 1. 开始训练对齐模型 (本地数据集)
echo ========================================================
python train_align.py --dataset_type local --data_path %DATA_PATH% --pretrained_spec %PRETRAINED_CKPT% --save_dir %SAVE_DIR%

echo ========================================================
echo 2. 准备检索评估数据 (.npy 副本)
echo ========================================================
python prepare_local_data.py --data_path %DATA_PATH% --save_dir %SAVE_DIR%/eval_data

echo ========================================================
echo 3. 运行基础检索评估 (eval.py)
echo ========================================================
python eval.py --checkpoint %SAVE_DIR%/final_aligned_model.pth --data_dir %SAVE_DIR%/eval_data

echo ========================================================
echo 4. 运行跨模态检索评估 (eval_align.py)
echo ========================================================
python eval_align.py --dataset_type local --data_path %DATA_PATH% --checkpoint %SAVE_DIR%/final_aligned_model.pth

endlocal
