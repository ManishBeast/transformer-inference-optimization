# Transformer Model Compression using Pruning and Quantization

This project explores optimization of Transformer-based NLP models using 
model compression techniques to reduce memory usage and improve inference speed.

The experiment compares baseline and compressed versions of DistilBERT and 
ELECTRA-Small on the AG News text classification dataset.

---

## Objective

Transformer models achieve high accuracy but require large memory and computational resources.

The goal of this project is to make transformer models more efficient for deployment on 
resource-constrained devices using:

- Model Pruning
- Dynamic Quantization

---

## Dataset

**AG News Classification Dataset**

- Text classification benchmark dataset
- Experiment performed on 1000 samples
- Task: News category classification

---

## Models Used

### DistilBERT
A compressed version of BERT optimized for faster inference.

### ELECTRA-Small
A lightweight transformer model trained using replaced-token detection.

---

## Compression Techniques

## 1. Global L1 Unstructured Pruning

- Applied 30% sparsity pruning
- Removes less important model weights
- Reduces computational complexity

## 2. Dynamic Quantization

- Converted FP32 weights into INT8 representation
- Reduces model size
- Improves CPU inference speed

---

## Experimental Results

| Model | Size (MB) | Accuracy | Inference Time |
|---|---|---|---|
| DistilBERT Baseline | 255.46 | 82.50% | 0.1668s |
| DistilBERT Pruned | 255.46 | 84.00% | 0.1237s |
| DistilBERT Quantized | 132.29 | 83.50% | 0.0690s |
| ELECTRA Baseline | 51.77 | 46.50% | 0.0424s |
| ELECTRA Pruned | 51.77 | 45.50% | 0.0367s |
| ELECTRA Quantized | 24.54 | 44.00% | 0.0328s |

---

## Key Results

- Reduced DistilBERT size from **255MB → 132MB**
- Achieved approximately **48% model size reduction**
- Improved DistilBERT inference speed by nearly **2.4×**
- Reduced ELECTRA-Small size by approximately **52%**
- Maintained comparable accuracy after compression

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Torch Pruning
- Dynamic Quantization
- NLP
- Deep Learning

---

## Project Structure

