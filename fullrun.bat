@echo off
setlocal

:: Get the short git commit hash
FOR /F "tokens=*" %%g IN ('git rev-parse --short HEAD') do (SET COMMIT_HASH=%%g)

:: If git command fails (e.g. not in a git repo), use a fallback name
IF "%COMMIT_HASH%"=="" (
    echo Warning: Could not get git commit hash. Using "unknown".
    set COMMIT_HASH=unknown
)

set SAVE_DIR=checkpoints_align\%COMMIT_HASH%
echo ========================================================
echo Starting Training Pipeline
echo Target Directory: %SAVE_DIR%
echo ========================================================

:: 1. Train the alignment model
python train_align.py --dataset_type massspecgym --pretrained_spec checkpoints/model.ckpt --save_dir %SAVE_DIR%

:: 2. Evaluate the aligned model
echo ========================================================
echo Starting Evaluation
echo ========================================================
python eval_align.py --checkpoint "%SAVE_DIR%\best_model_stage1.pth"
python eval_align.py --checkpoint "%SAVE_DIR%\best_model_stage2.pth"
python eval_align.py --checkpoint %SAVE_DIR%\final_aligned_model.pth

endlocal
