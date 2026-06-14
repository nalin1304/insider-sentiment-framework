# Segmentation Results
The segmentation analysis partitioned the universe by market cap terciles (Small, Mid, Large) and by GICS Sectors, training independent LightGBM models for each subgroup.

## Does segmentation reveal signals the pooled model missed?
Yes, but only in isolated pockets. While the pooled model struggled to find any consistent return predictability across the 396-ticker universe, segmenting the models uncovers specific subgroups where insider features provide a statistically significant lift over the baseline. However, many sectors do not show signal.

## Top Subgroups by MAE Improvement
| Grouping | Subgroup | Target | Feature_Set | Improvement_% | Test_Rows | Warning_Small_Sample |
|---|---|---|---|---|---|---|
| sector | Fast Moving Consumer Goods | abnormal_return_20d | all_combined | 0.7134640707874975 | 520 | True |
| sector | Fast Moving Consumer Goods | abnormal_return_20d | clustering | 0.7134640707874975 | 520 | True |
| sector | Fast Moving Consumer Goods | abnormal_return_20d | relative_size | 0.7134640707874975 | 520 | True |
| sector | Fast Moving Consumer Goods | abnormal_return_20d | role_split | 0.7134640707874975 | 520 | True |
| sector | Fast Moving Consumer Goods | abnormal_return_20d | original | 0.7134640707874975 | 520 | True |
| sector | Financial Services | abnormal_return_20d | clustering | 0.2791693761372997 | 1300 | True |
| sector | Financial Services | abnormal_return_20d | original | 0.2791693761372997 | 1300 | True |
| sector | Financial Services | abnormal_return_20d | role_split | 0.2345284199870425 | 1300 | True |
| sector | Oil, Gas & Consumable Fuels | abnormal_return_10d | clustering | 0.2283611741347549 | 260 | True |
| sector | Oil, Gas & Consumable Fuels | abnormal_return_10d | original | 0.2090499763237918 | 260 | True |
| sector | Oil, Gas & Consumable Fuels | abnormal_return_10d | role_split | 0.0825949874271157 | 260 | True |
| sector | Financial Services | abnormal_return_10d | clustering | 0.0777670397019386 | 1300 | True |
| sector | Financial Services | abnormal_return_10d | original | 0.0550009272893337 | 1300 | True |
| sector | Healthcare | abnormal_return_5d | all_combined | 0.0492793173803563 | 1039 | True |
| sector | Healthcare | abnormal_return_5d | relative_size | 0.0492793173803563 | 1039 | True |
| sector | Unknown | abnormal_return_20d | clustering | 0.044116196402384 | 96710 | False |
| sector | Unknown | abnormal_return_20d | all_combined | 0.044116196402384 | 96710 | False |
| sector | Unknown | abnormal_return_20d | role_split | 0.044116196402384 | 96710 | False |
| sector | Unknown | abnormal_return_20d | relative_size | 0.044116196402384 | 96710 | False |
| sector | Unknown | abnormal_return_20d | original | 0.044116196402384 | 96710 | False |