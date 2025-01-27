from Bio import SeqIO
from itertools import product
import pandas as pd
import os
import pickle
from sklearn.model_selection import cross_val_score

# Read sequences from a FASTA file
def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

# Calculate GC content
def gc_content(sequence):
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)

# Generate k-mer frequencies
def generate_kmers(sequence, k=3):
    kmers = [''.join(kmer) for kmer in product('ACGT', repeat=k)]
    kmer_counts = {kmer: 0 for kmer in kmers}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if 'N' not in kmer:  # Skip invalid k-mers
            kmer_counts[kmer] += 1
    return list(kmer_counts.values())

# Extract features for all sequences
def extract_features(sequences):
    features = []
    for seq in sequences:
        features.append({
            "gc_content": gc_content(seq),
            **dict(zip([f"kmer_{i}" for i in range(64)], generate_kmers(seq, k=3))),
            "sequence_length": len(seq)
        })
    return pd.DataFrame(features)

# Main function
if __name__ == "__main__":
    data_dir = "/Users/pavankumar/Desktop/pythonProject1/sequence_ml_project/data"

    fasta_files = [
        "/Users/pavankumar/Desktop/pythonProject1/sequence_ml_project/data/clostridium_spiroforme_dsm_1552_gca_000154805.ASM15480v1_.dna.nonchromosomal.fa",
        "/Users/pavankumar/Desktop/pythonProject1/sequence_ml_project/data/sequence.fasta"
    ]

    for fasta_file in fasta_files:
        if not os.path.exists(fasta_file):
            print(f"File not found: {fasta_file}")
            continue

        sequences = read_fasta(fasta_file)
        features_df = extract_features(sequences)
        output_file = os.path.join(data_dir, os.path.basename(fasta_file).replace(".fa", "_features.csv").replace(".fasta", "_features.csv"))
        features_df.to_csv(output_file, index=False)
        print(f"Features extracted and saved to {output_file}")

# Load the saved model
model_path = "/Users/pavankumar/Desktop/pythonProject1/sequence_ml_project/models/random_forest_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load new features
new_data_path = "/Users/pavankumar/Desktop/pythonProject1/sequence_ml_project/data/sequence_features.csv"
new_data = pd.read_csv(new_data_path)

# Check for 'label' column
print("Columns in new_data:", new_data.columns)

if "label" in new_data.columns:
    X = new_data.drop(columns=["label"])  # Features
    y = new_data["label"]  # Labels
else:
    print("Error: 'label' column not found in the dataset.")
    X = new_data  # Use all columns as features
    y = None
    # Optionally exit or proceed depending on whether labels are essential
    exit(1)

# Apply cross-validation (only if labels are available)
if y is not None:
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("Cross-validation scores:", scores)
    print("Mean accuracy:", scores.mean())

    # Make predictions
    predictions = model.predict(X)
    print("Predictions:", predictions)
else:
    print("Skipping cross-validation and predictions as labels are missing.")
