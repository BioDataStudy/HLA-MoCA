# HLA-MoCA

HLA-MoCA 是一个基于多分支融合深度学习的 HLA-I 肽段结合预测与免疫原性预测工具。一条命令即可输出结合分数（binding score）、百分位排名（percentile rank）和免疫原性分数（immunogenicity score）。

## 仓库内容

| 路径 | 说明 |
| --- | --- |
| `hlamoca_prediction.py` | 统一预测脚本 |
| `model/best_fusion_model.h5` | 主模型（结合预测）权重 |
| `model/best_finetuned_model.h5` | 免疫原性模型权重 |
| `supporting_file/pseq_dict_all.npy` | MHC 假序列字典（166 个等位基因） |
| `supporting_file/background_distributions_dict_updated.pkl` | 排名背景分布 |
| `testdata/peptides.txt` | 测试输入示例 |

## 环境要求

- Python 3.11（macOS arm64 已验证；Intel Mac 请改用 TensorFlow 2.15.x）
- TensorFlow 2.17.x + `tf-keras`（Keras 2 兼容层）

该脚本使用 Keras 2 风格 API。在 TensorFlow 2.16+ 下运行**必须**：

1. 安装 `tf-keras`（`requirements.txt` 已包含）；
2. 运行前设置环境变量 `TF_USE_LEGACY_KERAS=1`。

不设置该变量时（Keras 3 模式）会报错：
`ValueError: A KerasTensor cannot be used as input to a TensorFlow function`。

## 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 建议将虚拟环境放在仓库外，避免被 Git 识别为仓库改动。

## 输入格式

每行一个肽段，仅含标准氨基酸（A–Y），长度 8–15。

## 用法

### 基础预测（结合分数）

```bash
TF_USE_LEGACY_KERAS=1 python hlamoca_prediction.py \
  --input testdata/peptides.txt \
  --output scores.csv \
  --allele 'HLA-A*01:01'
```

### 完整预测（排名 + 免疫原性）

```bash
TF_USE_LEGACY_KERAS=1 python hlamoca_prediction.py \
  --input testdata/peptides.txt \
  --output full_scores.csv \
  --allele 'HLA-A*01:01,HLA-A*02:01' \
  --rank \
  --immunogenicity
```

模型、支持目录、背景文件、免疫原性模型均有仓库内默认路径，无需显式指定；
如需自定义可分别通过 `--model`、`--support`、`--background`、`--immunogenicity_model` 传入。

### 输出列

`pep`、`hla`、`score`；启用 `--rank` 时增加 `best_percentile_rank`、`is_binder`、
`matched_alleles`、`best_allele`；启用 `--immunogenicity` 时增加 `immunogenicity_score`。
多等位基因时分数以 `;` 分隔、按等位基因顺序排列。
