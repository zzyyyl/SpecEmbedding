@echo off
setlocal

:: Get the short git commit hash
FOR /F "tokens=*" %%g IN ('git rev-parse --short HEAD') do (SET COMMIT_HASH=%%g)

:: If git command fails (e.g. not in a git repo), use a fallback name
IF "%COMMIT_HASH%"=="" (
    echo Warning: Could not get git commit hash. Using "unknown".
    set COMMIT_HASH=unknown
)

set DATASET_TYPE=%1
IF "%DATASET_TYPE%"=="" (
    set DATASET_TYPE=massspecgym
)

set SAVE_DIR=checkpoints_align\%COMMIT_HASH%_%DATASET_TYPE%
echo ========================================================
echo Starting Training Pipeline
echo Target Directory: %SAVE_DIR%
echo Dataset Type: %DATASET_TYPE%
echo ========================================================

:: 1. Train the alignment model
python train_align.py --dataset_type %DATASET_TYPE% --pretrained_spec checkpoints/model.ckpt --save_dir %SAVE_DIR%

:: 2. Evaluate the aligned model
echo ========================================================
echo Starting Evaluation
echo ========================================================
python eval_align.py --dataset_type %DATASET_TYPE% --checkpoint "%SAVE_DIR%\best_model_stage1.pth" --no-mces
python eval_align.py --dataset_type %DATASET_TYPE% --checkpoint "%SAVE_DIR%\best_model_stage2.pth"
python eval_align.py --dataset_type %DATASET_TYPE% --checkpoint %SAVE_DIR%\final_aligned_model.pth --no-mces

endlocal
