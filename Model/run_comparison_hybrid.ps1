$env:PYTHONUTF8=1
Write-Host "=================================="
Write-Host "Starting Comparison A: Baseline (Hybrid Strict)"
Write-Host "   Target: ~7200 samples"
Write-Host "=================================="
python Model/train.py --mode product_level --oversample xgb_scale_pos_weight --label-strategy hybrid --label-ratio-threshold 1.0 --label-delta-threshold 10 --outdir Model/outputs/exp_baseline_hybrid

Write-Host "`n=================================="
Write-Host "Starting Comparison B: V2 Filter (Hybrid Strict)"
Write-Host "   Target: Same dataset but filtered"
Write-Host "=================================="
python Model/train.py --mode product_level --oversample xgb_scale_pos_weight --label-strategy hybrid --label-ratio-threshold 1.0 --label-delta-threshold 10 --filter-version v2_error_prod --outdir Model/outputs/exp_v2_filter_hybrid

Write-Host "`nAll Experiments Completed."
