# HLA-MoCA: motif-aware deep learning model for HLA-I antigen presentation and neoantigen prioritization
Accurate prediction of peptides presented by human leukocyte antigen class I (HLA-I) molecules is critical for tumor antigen discovery, personalized cancer vaccine development, and cancer immunotherapy. With the rapid accumulation of LC–MS/MS-based immunopeptidomics data, there is an urgent need for HLA-I antigen presentation prediction methods that combine high predictive performance, generalizability, and biological interpretability. To address this need, we developed HLA-MoCA (Motif-aware Cross-Attention Model), an interpretable deep learning framework for pan-allelic HLA-I antigen presentation prediction. HLA-MoCA comprises three complementary modules—KPairNet, ECompNet, and a cross-attention Transformer—which capture global ligand motif features shared across alleles, allele-specific local submotif features, and residue-level peptide–HLA-I interactions, respectively.
HLA-MoCA was trained on 537,989 mass spectrometry–identified HLA-I ligand entries. The high-weight features learned by the model closely align with established anchor-residue preferences and structural constraints of the HLA-I binding pocket. On an independent monoallelic benchmark dataset, HLA-MoCA achieved an AUROC of 0.963 and an AUPRC of 0.765, outperforming the existing prediction methods evaluated.
We further demonstrated the utility of HLA-MoCA across multiallelic tumor immunopeptidomics datasets from lung cancer, melanoma, pancreatic cancer, and breast cancer, where it showed stable deconvolution performance across cancer types and improved the identification of HLA-I-presented peptides. In addition, HLA-MoCA more effectively prioritized immunogenic neoantigens, and its predicted antigen presentation burden was associated with prolonged progression-free survival in patients receiving immunotherapy. Together, HLA-MoCA provides an accurate, interpretable, and generalizable computational framework for HLA-I antigen presentation prediction, multiallelic immunopeptidome analysis, tumor antigen prioritization, and personalized neoantigen vaccine development.

<div align=center><img src="https://github.com/BioDataStudy/HLA-MoCA/blob/main/Figure%201.jpg" width="800px"></div>

# Features

- **Presentation score prediction:** for one or multiple HLA-I alleles simultaneously.
- **Percentile ranking:** rank peptide–HLA-I pairs against pre-computed background distributions to identify strong binders.
- **Immunogenicity scoring:** score and prioritize candidate neoepitopes for immunogenicity.
- **Multi-allele support:** specify multiple alleles separated by commas.

# Installation
Download HLA-MoCA by
```
git clone https://github.com/BioDataStudy/HLA-MoCA
```

The HLA-MoCA environment is based on **Python 3.7** and **TensorFlow 2.8.0**. HLA-MoCA is implemented in Python 3 and requires the following dependencies: numpy, scipy, pandas, scikit-learn, matplotlib, seaborn, joblib, tqdm, logomaker, and umap-learn. We recommend creating an independent Conda environment for HLA-MoCA to avoid potential dependency conflicts. If you encounter any problems during installation or running, please leave a message on the HLA-MoCA issue page (https://github.com/BioDataStudy/HLA-MoCA/issues). You can install the required packages using the following commands:

The original HLA-MoCA environment is based on **Python 3.7** and **TensorFlow 2.8.0**. A compatible minimal environment for the current prediction script can be created as follows:

```
# Create and activate the environment
conda create -n hlamoca python=3.7 -y
conda activate hlamoca

# Install TensorFlow (GPU version)
pip install tensorflow==2.8.0

# Core scientific libraries
pip install numpy==1.24.4
pip install pandas==2.0.3
pip install scipy==1.10.1
pip scikit-learn==1.3.2 
pip matplotlib==3.7.5 
pip seaborn==0.13.2

# Additional utilities
pip install joblib 
pip install tqdm 
pip install logomaker 
pip install umap-learn
```

After installation, check the command-line options with:

```bash
python hlamoca_prediction.py --help
```

# Usage
## HLA-MoCA provides prediction of antigen presentation score, percentile ranking, immunogenicity scoring, and multi-allele prediction.

The input file should contain one peptide sequence per line. Peptides should consist of standard amino acids and have a length of 8–15 amino acids. Candidate HLA-I allele(s) can be specified for prediction.

__1__. For antigen presentation prediction, this module accepts peptides together with the candidate HLA-I allele(s) as input and generates prediction scores for peptide–HLA-I pairs.

### Example of antigen presentation score prediction for a single HLA-I allele

For __a single HLA-I allele__, please uses:
```
cd HLA-MoCA/
python hlamoca_prediction.py \
  --input testdata/test_pep.txt \
  --output scores.csv \
  --allele HLA-A*01:01
```

__2__. For multi-allele support, this module allows multiple candidate HLA-I alleles to be evaluated simultaneously. Different HLA-I alleles should be separated by commas without spaces.

### Example of antigen presentation score prediction for multiple HLA-I alleles

For __multiple HLA-I alleles__, please uses:

```
python hlamoca_prediction.py \
  --input testdata/test_pep.txt \
  --output scores_multi.csv \
  --allele HLA-A*01:01,HLA-B*07:02,HLA-C*06:02
```

__3__. For percentile ranking, this module compares the predicted scores of peptide–HLA-I pairs against pre-computed background distributions to obtain percentile ranks and identify strong candidates.

### Example of percentile ranking for antigen presentation prediction:
For __peptides input__,  please uses:

```
python hlamoca_prediction.py \
  --input testdata/test_pep.txt \
  --output scores_rank.csv \
  --allele HLA-A*01:01,HLA-B*07:02 \
  --rank \
  --threshold 2.0
```

__4__. For immunogenicity scoring, this module accepts peptide sequences as input together with candidate HLA-I allele(s) and generates immunogenicity scores for candidate peptide–HLA-I pairs.

### Example of neoepitope immunogenicity scoring: 

For __peptides input__, please uses:

```
python hlamoca_prediction.py \
  --input testdata/test_pep.txt \
  --output scores_immunogenicity.csv \
  --allele HLA-A*01:01,HLA-B*07:02 \
  --immunogenicity
```

For details of other parameters, run:
```
python hlamoca_prediction.py --help

```

## Citation

Citation information will be updated upon publication of the HLA-MoCA manuscript.
