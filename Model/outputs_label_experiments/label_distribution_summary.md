# Label Distribution Summary

**Total Samples (N):** 7197
**Alignment Key:** `product_id` (One sample per product)

## 檔案說明

- **用途**：總結「每個 product 一筆樣本」的 label 分布情況。
- **範圍限制**：
  - 本表僅包含第一階段（Phase 1）的三個實驗：`baseline_v1`、`hybrid_relaxed`、`hybrid_strict`。
  - 這三個實驗使用完全相同的樣本集 (N=7197)，但採用不同的 label 定義策略。
- **實驗差異簡述**：
  - `baseline_v1`：僅看絕對成長量 (delta >= 10)。
  - `hybrid_relaxed`：加入寬鬆比例門檻 (delta >= 10 & ratio >= 0.1)。
  - `hybrid_strict`：加入嚴格比例門檻 (delta >= 10 & ratio >= 0.3)，剔除高基數低成長樣本。
- **備註**：
  - 後續新增的實驗（如 `multiclass_percentile`、`binary_explosive`）**未列於此表**，請參閱 `summary.md` 或各別分析報告。


| Experiment | Strategy | Params | N | Pos | Neg | Pos Rate |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_v1 | absolute | delta=10 | 7197 | 464 | 6733 | 0.0645 |
| hybrid_relaxed | hybrid | delta=10, ratio=0.1 | 7197 | 464 | 6733 | 0.0645 |
| hybrid_strict | hybrid | delta=10, ratio=0.3 | 7197 | 437 | 6760 | 0.0607 |
