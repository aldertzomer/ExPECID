# ExPEC-ID
<img src="images/logo.png" alt="ExPECID logo" width="240"/> [![Try it online](https://img.shields.io/badge/Try%20it%20online-klif.uu.nl%2Fmetapointfinder-blue?style=for-the-badge)](https://klif.uu.nl/expecid)

**ExPEC-ID** is a machine learning–based tool for predicting the pathogenic potential (“ExPECiness”) of Escherichia coli from genome assemblies or sequencing reads.

A user-friendly **web interface** is available at [klif.uu.nl/expecid](https://klif.uu.nl/expecid).

The tool combines fast k-mer–based alignment using KMA with a trained Random Forest classifier to distinguish extraintestinal pathogenic E. coli (ExPEC) from non-ExPEC.

---

## OVERVIEW

### Input:
- Genome assemblies (.fasta, .fna)
- Single-end reads (.fastq, .fastq.gz)
- Paired-end reads (R1/R2 fastq)

### Pipeline:
1. Mapping to marker genes using KMA
2. Extraction of identity and coverage features
3. Random Forest classification
4. Output of probabilities and predictions

---

## INSTALLATION

### Requirements:
- Python ≥ 3.11
- KMA ≥ 1.4.18a (has presets for reads and assemblies)
- pandas, joblib, scikit-learn

### Install:

```git clone https://github.com/aldertzomer/ExPECID.git```

```cd ExPECID```

```conda create --name expecid --file spec-file.txt```

---

## USAGE

Assembly:

```python ExPECID.py -i sample.fasta -o results.tsv --threads 8```


Single-end:

```python ExPECID.py -is sample.fastq.gz -o results.tsv --threads 8```


Paired-end:

```python ExPECID.py -ipe sample_R1.fastq.gz sample_R2.fastq.gz -o results.tsv --threads 8```

Results will get appended to results.tsv if the file already exists.

---

## MODEL SELECTION

Assembly → RFC_expec_nonexpec_full_asm_v1.0.joblib

Reads    → RFC_expec_nonexpec_full_reads_v1.0.joblib

---

## OUTPUT

Columns:

isolate, p_nonexpec, p_expec, prediction, features...

prediction:

0 = non-ExPEC

1 = ExPEC

Behavior:
- Creates file if not present
- Appends if file exists

---

## FEATURE GENERATION

Per gene:
- Identity
- Coverage

Best hit:
- Highest KMA Score (column 2)

---

## TEMP FILES, DEBUG

Use:
--keep-temp

---

## PERFORMANCE

Assembly: seconds
Reads: seconds to minutes

---

## METHOD SUMMARY

- ~19,000 genomes ExPEC/Non-ExPEC
- Roary pangenome / GWAS
- Random Forest feature selection only on ExPEC associated genes
- KMA index of ortholog alleles of 135 top marker genes
- ~135 genes final model -> implementation uses both id% and coverage% of hits

## Performance:
AUC ≈ 0.95
F1 ≈ 0.92

---

## APPLICATIONS

- High-risk clone detection
- MLST / serotype scoring
- Metagenomics
- Patient stratification
- One Health surveillance

---

## NOTES

- Do not change feature logic without retraining
- Uses KMA top score for selection of best hit
- Requires identical preprocessing (do not change KMA settings)

---

## CITATION

Zomer A.L., de Been M., et al.
Machine learning–based identification of extraintestinal pathogenic E. coli

---

## GITHUB

https://github.com/aldertzomer/ExPECID

---

## CONTACT

Aldert Zomer
Utrecht University
