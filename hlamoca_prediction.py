# %%
#!/usr/bin/env python3
"""
HLA-MoCA Unified Predictor
---------------------------
Features:
  - Predict binding scores for one or more HLA alleles (comma‑separated).
  - Optionally compute percentile ranks against pre‑computed background distributions.
  - Optionally predict immunogenicity using a fine‑tuned model.

Usage examples:
  # Basic score prediction
  python hlamoca_predict_full.py --input peptides.txt --output scores.csv --allele HLA-A*01:01

  # Score + ranking (background file auto‑loaded from ./supporting_files/)
  python hlamoca_predict_full.py --input peptides.txt --output scores_rank.csv \\
      --allele HLA-A*01:01,HLA-B*07:02 --rank

  # Score + immunogenicity (model auto‑loaded from ./model/)
  python hlamoca_predict_full.py --input peptides.txt --output full.csv \\
      --allele HLA-A*01:01,HLA-B*07:02 --immunogenicity

Input format: one peptide per line, only standard amino acids (A..Y), length 8–15.
Output CSV columns:
  pep, hla, score, [best_percentile_rank, is_binder, matched_alleles, best_allele],
  [immunogenicity_score]
"""

import os, sys, re, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate, Bidirectional, LSTM,
    TimeDistributed, Activation, Permute, Dot, GlobalAveragePooling1D, GlobalMaxPooling1D,
    LayerNormalization, MultiHeadAttention, Dropout, Add, Multiply, Reshape, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from collections import Counter
from scipy.stats import percentileofscore
import pickle
import warnings
warnings.filterwarnings('ignore')

# ==================== Default paths (relative to script location) ====================
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_MODEL_PATH = os.path.join(_SCRIPT_DIR, 'model', 'best_fusion_model.h5')
DEFAULT_IMMUNOGENICITY_MODEL = os.path.join(_SCRIPT_DIR, 'model', 'finetuned_immunogenicity_model.h5')
DEFAULT_BACKGROUND_PKL = os.path.join(_SCRIPT_DIR, 'supporting_files', 'background_distributions.pkl')
DEFAULT_SUPPORT_PATH = os.path.join(_SCRIPT_DIR, 'supporting_files')
# =====================================================================================

# Amino acid index
aa = {"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}

# ---------------------------------------------------------------------------
# Encoding functions (identical to original)
# ---------------------------------------------------------------------------
def EAAC_fixed(sequence, windows=[3, 5], fixed_positions=5):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encoding = []
    n = len(sequence)
    for window in windows:
        for i in range(fixed_positions):
            center_index = int(i * (n - 1) / (fixed_positions - 1)) if fixed_positions > 1 else n // 2
            start_index = center_index - window // 2
            end_index = start_index + window
            if start_index < 0:
                start_index = 0
                end_index = window
            if end_index > n:
                end_index = n
                start_index = end_index - window
                if start_index < 0:
                    start_index = 0
            window_seq = sequence[start_index:end_index]
            count = Counter(window_seq)
            for key in count:
                count[key] = count[key] / len(window_seq) if len(window_seq) > 0 else 0
            for aa_char in AA:
                encoding.append(count.get(aa_char, 0))
    return encoding

def CKSAAP(sequence, gap=[2,4,6]):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)
    encoding = []
    for g in gap:
        myDict = {}
        for pair in aaPairs:
            myDict[pair] = 0
        total_pairs = 0
        for index1 in range(len(sequence)):
            index2 = index1 + g + 1
            if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[index2] in AA:
                myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                total_pairs = total_pairs + 1
        if total_pairs == 0:
            encoding.extend(np.zeros(400))
        else:
            for pair in aaPairs:
                encoding.append(myDict[pair] / total_pairs)
    return encoding

def peptide_one_hot(pep, max_length=30):
    pep_one_hot = []
    for residue_index in range(15):
        if residue_index < len(pep):
            aa_char = pep[residue_index]
            if aa_char in aa:
                pep_one_hot.append(np.eye(20)[aa[aa_char]])
            else:
                pep_one_hot.append(np.zeros(20))
        else:
            pep_one_hot.append(np.zeros(20))
    for residue_index in range(15):
        if 15 - residue_index > len(pep):
            pep_one_hot.append(np.zeros(20))
        else:
            aa_char = pep[len(pep) - 15 + residue_index]
            if aa_char in aa:
                pep_one_hot.append(np.eye(20)[aa[aa_char]])
            else:
                pep_one_hot.append(np.zeros(20))
    return np.array(pep_one_hot)

def encode_single_sample(peptide, allele, pseq_dict_one_hot):
    """Return (eaac, cksaap, transformer, mhc) or None if invalid."""
    if (set(list(peptide)).difference(list('ACDEFGHIKLMNPQRSTVWY')) or
        len(peptide) not in [8,9,10,11,12,13,14,15] or
        allele not in pseq_dict_one_hot):
        return None
    eaac = EAAC_fixed(peptide)
    cksaap = CKSAAP(peptide)
    transformer = peptide_one_hot(peptide)
    mhc = pseq_dict_one_hot[allele]
    return eaac, cksaap, transformer, mhc

# ---------------------------------------------------------------------------
# Model construction (identical to original)
# ---------------------------------------------------------------------------
def build_fusion_model(pep_eaac_dim=200, pep_cksaap_dim=1200, pep_transformer_length=30,
                      mhc_length=34, mhc_dim=20, parameters=None):
    """Build the three‑branch fusion model."""
    if parameters is None:
        parameters = {
            'eaac_filters': [64, 128],
            'eaac_kernel_sizes': [3, 5],
            'eaac_lstm_units': 64,
            'cksaap_filters': [64, 128],
            'cksaap_kernel_sizes': [3, 5],
            'cksaap_lstm_units': 64,
            'transformer_heads': 4,
            'transformer_key_dim': 32,
            'transformer_ff_dim': 256,
            'transformer_lstm_units': 64,
            'fusion_units': [512, 256, 128],
            'dropout_rate': 0.3
        }

    pep_eaac_input = Input(shape=(pep_eaac_dim,), name="pep_eaac_input")
    pep_cksaap_input = Input(shape=(pep_cksaap_dim,), name="pep_cksaap_input")
    pep_transformer_input = Input(shape=(pep_transformer_length, mhc_dim), name="pep_transformer_input")
    mhc_input = Input(shape=(mhc_length, mhc_dim), name="mhc_input")

    # ----- Branch 1: EAAC -----
    def build_eaac_branch(pep_input, mhc_input, prefix="eaac"):
        pep_timesteps = pep_input.shape[1] // 20
        pep_reshaped = Reshape((pep_timesteps, 20))(pep_input)
        conv_outputs = []
        for kernel_size in parameters['eaac_kernel_sizes']:
            conv = Conv1D(parameters['eaac_filters'][0], kernel_size, activation='relu',
                         padding='same', name=f"{prefix}_conv_{kernel_size}")(pep_reshaped)
            conv = BatchNormalization()(conv)
            conv_outputs.append(conv)
        if len(conv_outputs) > 1:
            pep_conv = Concatenate(axis=-1, name=f"{prefix}_conv_concat")(conv_outputs)
        else:
            pep_conv = conv_outputs[0]
        pep_lstm = Bidirectional(
            LSTM(parameters['eaac_lstm_units'], return_sequences=True, dropout=0.2),
            name=f"{prefix}_bilstm"
        )(pep_conv)
        mhc_conv = Conv1D(64, 3, activation='relu', padding='same', name=f"{prefix}_mhc_conv")(mhc_input)
        mhc_lstm = Bidirectional(
            LSTM(parameters['eaac_lstm_units'], return_sequences=True, dropout=0.2),
            name=f"{prefix}_mhc_bilstm"
        )(mhc_conv)
        pep_attention = MultiHeadAttention(num_heads=4, key_dim=16, name=f"{prefix}_pep_attention")(pep_lstm, pep_lstm)
        mhc_attention = MultiHeadAttention(num_heads=4, key_dim=16, name=f"{prefix}_mhc_attention")(mhc_lstm, mhc_lstm)
        def multi_scale_pooling(inputs, prefix):
            avg_pool = GlobalAveragePooling1D(name=f"{prefix}_avg_pool")(inputs)
            max_pool = GlobalMaxPooling1D(name=f"{prefix}_max_pool")(inputs)
            return Concatenate(name=f"{prefix}_pool_concat")([avg_pool, max_pool])
        pep_pooled = multi_scale_pooling(pep_attention, f"{prefix}_pep")
        mhc_pooled = multi_scale_pooling(mhc_attention, f"{prefix}_mhc")
        branch_output = Concatenate(name=f"{prefix}_concat")([pep_pooled, mhc_pooled])
        branch_output = Dense(256, activation='relu', name=f"{prefix}_fc1")(branch_output)
        branch_output = Dropout(parameters['dropout_rate'])(branch_output)
        branch_output = Dense(128, activation='relu', name=f"{prefix}_fc2")(branch_output)
        return branch_output

    # ----- Branch 2: CKSAAP -----
    def build_cksaap_branch(pep_input, mhc_input, prefix="cksaap"):
        pep_timesteps = pep_input.shape[1] // 20
        pep_reshaped = Reshape((pep_timesteps, 20))(pep_input)
        conv_outputs = []
        for kernel_size in parameters['cksaap_kernel_sizes']:
            conv = Conv1D(parameters['cksaap_filters'][0], kernel_size, activation='relu',
                         padding='same', name=f"{prefix}_conv_{kernel_size}")(pep_reshaped)
            conv = BatchNormalization()(conv)
            conv_outputs.append(conv)
        if len(conv_outputs) > 1:
            pep_conv = Concatenate(axis=-1, name=f"{prefix}_conv_concat")(conv_outputs)
        else:
            pep_conv = conv_outputs[0]
        pep_lstm = Bidirectional(
            LSTM(parameters['cksaap_lstm_units'], return_sequences=True, dropout=0.2),
            name=f"{prefix}_bilstm"
        )(pep_conv)
        mhc_conv = Conv1D(64, 3, activation='relu', padding='same', name=f"{prefix}_mhc_conv")(mhc_input)
        mhc_lstm = Bidirectional(
            LSTM(parameters['cksaap_lstm_units'], return_sequences=True, dropout=0.2),
            name=f"{prefix}_mhc_bilstm"
        )(mhc_conv)
        pep_attention = MultiHeadAttention(num_heads=4, key_dim=16, name=f"{prefix}_pep_attention")(pep_lstm, pep_lstm)
        mhc_attention = MultiHeadAttention(num_heads=4, key_dim=16, name=f"{prefix}_mhc_attention")(mhc_lstm, mhc_lstm)
        pep_pooled = GlobalAveragePooling1D()(pep_attention)
        mhc_pooled = GlobalAveragePooling1D()(mhc_attention)
        branch_output = Concatenate(name=f"{prefix}_concat")([pep_pooled, mhc_pooled])
        branch_output = Dense(256, activation='relu', name=f"{prefix}_fc1")(branch_output)
        branch_output = Dropout(parameters['dropout_rate'])(branch_output)
        branch_output = Dense(128, activation='relu', name=f"{prefix}_fc2")(branch_output)
        return branch_output

    # ----- Branch 3: Transformer -----
    def build_transformer_branch(pep_input, mhc_input, prefix="transformer"):
        def transformer_encoder(inputs, num_heads=4, key_dim=32, ff_dim=256, dropout_rate=0.1):
            x1 = LayerNormalization(epsilon=1e-6)(inputs)
            attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)(x1, x1)
            x2 = Add()([attention_output, inputs])
            x2 = LayerNormalization(epsilon=1e-6)(x2)
            ff_output = Dense(ff_dim, activation="relu")(x2)
            ff_output = Dropout(dropout_rate)(ff_output)
            ff_output = Dense(inputs.shape[-1])(ff_output)
            outputs = Add()([ff_output, x2])
            outputs = LayerNormalization(epsilon=1e-6)(outputs)
            return outputs

        pep_transformer = transformer_encoder(
            pep_input,
            num_heads=parameters['transformer_heads'],
            key_dim=parameters['transformer_key_dim'],
            ff_dim=parameters['transformer_ff_dim']
        )
        mhc_transformer = transformer_encoder(
            mhc_input,
            num_heads=parameters['transformer_heads'],
            key_dim=parameters['transformer_key_dim'],
            ff_dim=parameters['transformer_ff_dim']
        )
        pep_lstm = Bidirectional(LSTM(parameters['transformer_lstm_units'], return_sequences=True, dropout=0.2))(pep_transformer)
        mhc_lstm = Bidirectional(LSTM(parameters['transformer_lstm_units'], return_sequences=True, dropout=0.2))(mhc_transformer)
        cross_attention = MultiHeadAttention(num_heads=4, key_dim=16, name=f"{prefix}_cross_attention")(pep_lstm, mhc_lstm)
        pep_pooled = GlobalAveragePooling1D()(cross_attention)
        mhc_pooled = GlobalAveragePooling1D()(mhc_lstm)
        branch_output = Concatenate(name=f"{prefix}_concat")([pep_pooled, mhc_pooled])
        branch_output = Dense(256, activation='relu', name=f"{prefix}_fc1")(branch_output)
        branch_output = Dropout(parameters['dropout_rate'])(branch_output)
        branch_output = Dense(128, activation='relu', name=f"{prefix}_fc2")(branch_output)
        return branch_output

    eaac_branch = build_eaac_branch(pep_eaac_input, mhc_input, "eaac")
    cksaap_branch = build_cksaap_branch(pep_cksaap_input, mhc_input, "cksaap")
    transformer_branch = build_transformer_branch(pep_transformer_input, mhc_input, "transformer")

    fused_features = Concatenate(name="feature_fusion")([eaac_branch, cksaap_branch, transformer_branch])

    attention_weights = Dense(3, activation='softmax', name="fusion_attention")(fused_features)
    weighted_branches = []
    for i in range(3):
        weight = tf.expand_dims(attention_weights[:, i], axis=-1)
        if i == 0:
            weighted = tf.multiply(weight, eaac_branch)
        elif i == 1:
            weighted = tf.multiply(weight, cksaap_branch)
        else:
            weighted = tf.multiply(weight, transformer_branch)
        weighted_branches.append(weighted)
    attention_fused = Add(name="attention_fusion")(weighted_branches)

    x = Dense(parameters['fusion_units'][0], activation="relu", name="fc_fusion1")(attention_fused)
    x = Dropout(parameters['dropout_rate'])(x)
    x = Dense(parameters['fusion_units'][1], activation="relu", name="fc_fusion2")(x)
    x = Dropout(parameters['dropout_rate'] * 0.8)(x)
    x = Dense(parameters['fusion_units'][2], activation="relu", name="fc_fusion3")(x)
    x = Dropout(parameters['dropout_rate'] * 0.6)(x)
    output = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(
        inputs=[pep_eaac_input, pep_cksaap_input, pep_transformer_input, mhc_input],
        outputs=output,
        name="hla_prediction_fusion_model"
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"]
    )
    return model

# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------
def load_resources(support_path):
    """Load MHC pseudo‑sequence dictionary and build one‑hot encoding."""
    pseq_dict = np.load(os.path.join(support_path, 'pseq_dict_all.npy'), allow_pickle=True).item()
    pseq_dict_one_hot = {}
    for allele, seq in pseq_dict.items():
        one_hot = []
        for i in range(34):
            vec = np.zeros(20)
            if i < len(seq) and seq[i] in aa:
                vec[aa[seq[i]]] = 1
            one_hot.append(vec)
        pseq_dict_one_hot[allele] = np.array(one_hot)
    return pseq_dict_one_hot

def load_model(weights_path):
    """Build model and load weights."""
    model = build_fusion_model()
    model.load_weights(weights_path)
    print(f"Model loaded from {weights_path}")
    return model

def load_background_dict(pkl_path):
    """Load background distribution dictionary."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Background file not found: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        bg_dict = pickle.load(f)
    print(f"Background dictionary loaded: {len(bg_dict)} alleles.")
    return bg_dict

# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------
def predict_peptides(peptides, allele, model, pseq_dict_one_hot, batch_size=2048):
    """Return (valid_peps, scores) for one allele."""
    valid_peps = []
    encoded = []
    for pep in peptides:
        enc = encode_single_sample(pep, allele, pseq_dict_one_hot)
        if enc is not None:
            valid_peps.append(pep)
            encoded.append(enc)
    if not encoded:
        return [], []
    eaac = np.array([e[0] for e in encoded])
    cksaap = np.array([e[1] for e in encoded])
    transformer = np.array([e[2] for e in encoded])
    mhc = np.array([e[3] for e in encoded])
    scores = []
    for i in range(0, len(eaac), batch_size):
        end = min(i+batch_size, len(eaac))
        batch_pred = model.predict(
            [eaac[i:end], cksaap[i:end], transformer[i:end], mhc[i:end]],
            verbose=0
        )
        scores.extend(batch_pred.flatten())
    return valid_peps, scores

# ---------------------------------------------------------------------------
# Filter peptides
# ---------------------------------------------------------------------------
def filter_peptides(peptides):
    valid = []
    for p in peptides:
        p = p.strip()
        if len(p) in [8,9,10,11,12,13,14,15] and set(p).issubset(set('ACDEFGHIKLMNPQRSTVWY')):
            valid.append(p)
    return valid

# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HLA-MoCA Unified Predictor")
    parser.add_argument('--input', required=True, help='Input file: one peptide per line.')
    parser.add_argument('--output', required=True, help='Output CSV file path.')
    parser.add_argument('--allele', required=True,
                        help='HLA allele(s), comma‑separated, e.g. HLA-A*01:01,HLA-B*07:02')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH,
                        help='Path to the main prediction model weights (default: ./model/best_fusion_model.h5)')
    parser.add_argument('--support', default=DEFAULT_SUPPORT_PATH,
                        help='Directory containing supporting files (default: ./supporting_files/)')
    parser.add_argument('--rank', action='store_true',
                        help='Compute percentile ranks (background file auto‑loaded from ./supporting_files/)')
    parser.add_argument('--background', default=DEFAULT_BACKGROUND_PKL,
                        help='Path to background distribution pickle file (default: ./supporting_files/background_distributions.pkl)')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Percentile threshold for binder classification (default 1.0).')
    parser.add_argument('--immunogenicity', action='store_true',
                        help='Predict immunogenicity scores (model auto‑loaded from ./model/)')
    parser.add_argument('--immunogenicity_model', default=DEFAULT_IMMUNOGENICITY_MODEL,
                        help='Path to the fine‑tuned immunogenicity model weights (default: ./model/finetuned_immunogenicity_model.h5)')
    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.input):
        print(f"Error: input file {args.input} not found.", file=sys.stderr)
        sys.exit(1)
    if args.rank and not os.path.exists(args.background):
        print(f"Error: background file {args.background} not found. "
              "Please ensure the file exists at the default location or provide a custom path.",
              file=sys.stderr)
        sys.exit(1)
    if args.immunogenicity and not os.path.exists(args.immunogenicity_model):
        print(f"Error: immunogenicity model {args.immunogenicity_model} not found. "
              "Please ensure the file exists at the default location or provide a custom path.",
              file=sys.stderr)
        sys.exit(1)

    # Read peptides
    with open(args.input, 'r') as f:
        raw = [line.strip() for line in f if line.strip()]
    print(f"Read {len(raw)} raw entries.")
    peptides = filter_peptides(raw)
    print(f"Valid peptides after filtering: {len(peptides)}")
    if not peptides:
        print("No valid peptides. Exiting.", file=sys.stderr)
        sys.exit(0)

    # Parse alleles
    alleles = [a.strip() for a in args.allele.split(',')]
    print(f"Alleles: {alleles}")

    # Load resources
    print("Loading supporting files...")
    pseq_dict_one_hot = load_resources(args.support)

    # Load main model
    print("Loading main prediction model...")
    main_model = load_model(args.model)

    # Load immunogenicity model if requested
    im_model = None
    if args.immunogenicity:
        print("Loading immunogenicity model...")
        im_model = load_model(args.immunogenicity_model)

    # Load background if ranking
    bg_dict = None
    if args.rank:
        print("Loading background distribution...")
        bg_dict = load_background_dict(args.background)

    # Predict for each allele
    print("Performing predictions...")
    all_main_scores = {}   # allele -> list of scores (same order as peptides)
    all_im_scores = {}     # allele -> list of immunogenicity scores

    for allele in alleles:
        # Main scores
        _, scores = predict_peptides(peptides, allele, main_model, pseq_dict_one_hot)
        if len(scores) != len(peptides):
            # Pad with NaN for missing positions
            full_scores = []
            idx = 0
            for pep in peptides:
                enc = encode_single_sample(pep, allele, pseq_dict_one_hot)
                if enc is not None:
                    full_scores.append(scores[idx])
                    idx += 1
                else:
                    full_scores.append(np.nan)
            all_main_scores[allele] = full_scores
        else:
            all_main_scores[allele] = scores

        # Immunogenicity scores
        if args.immunogenicity:
            _, im_scores = predict_peptides(peptides, allele, im_model, pseq_dict_one_hot)
            if len(im_scores) != len(peptides):
                full_im = []
                idx = 0
                for pep in peptides:
                    enc = encode_single_sample(pep, allele, pseq_dict_one_hot)
                    if enc is not None:
                        full_im.append(im_scores[idx])
                        idx += 1
                    else:
                        full_im.append(np.nan)
                all_im_scores[allele] = full_im
            else:
                all_im_scores[allele] = im_scores

    # Build DataFrame
    df = pd.DataFrame({'pep': peptides})
    df['hla'] = ';'.join(alleles)
    df['score'] = [';'.join(str(all_main_scores[a][i]) for a in alleles) for i in range(len(peptides))]

    # Ranking columns
    if args.rank:
        best_ranks = []
        is_binders = []
        matched_alleles_list = []
        best_allele_list = []
        for i in range(len(peptides)):
            scores_i = [all_main_scores[a][i] for a in alleles]
            best_rank = 100.0
            best_allele = ''
            matched = []
            for idx, allele in enumerate(alleles):
                score = scores_i[idx]
                if allele in bg_dict and not np.isnan(score):
                    bg_scores = bg_dict[allele]
                    rank = percentileofscore(bg_scores, score, kind='rank')
                    percentile_rank = 100.0 - rank
                    percentile_rank = round(percentile_rank, 4)
                    if percentile_rank < best_rank:
                        best_rank = percentile_rank
                        best_allele = allele
                    if percentile_rank <= args.threshold:
                        matched.append(allele)
            best_ranks.append(best_rank if best_rank < 100.0 else np.nan)
            is_binders.append(str(best_rank <= args.threshold))
            matched_alleles_list.append(';'.join(matched) if matched else '')
            best_allele_list.append(best_allele)
        df['best_percentile_rank'] = best_ranks
        df['is_binder'] = is_binders
        df['matched_alleles'] = matched_alleles_list
        df['best_allele'] = best_allele_list

    # Immunogenicity column
    if args.immunogenicity:
        df['immunogenicity_score'] = [
            ';'.join(str(all_im_scores[a][i]) for a in alleles) for i in range(len(peptides))
        ]

    # Save
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    print("Done.")

if __name__ == "__main__":
    main()


