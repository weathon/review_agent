# Quantization-Enhanced HNSW for Scalable Approximate Vector Search

- Decision: Reject
- Scores: 0, 0, 2

## Abstract
Graph-based approximate nearest neighbor search, specifically Hierarchical Navi-
gable Small World (HNSW), remains the standard for low-latency vector retrieval.
However, as datasets grow to millions of high-dimensional embeddings, the RAM
requirements for full-precision (float32) indices become prohibitive. While Scalar
Quantization (SQ) can reduce this footprint, naive min-max scaling often fails in
practice: a handful of outliers can stretch the quantization bins, causing “collapse”
where useful data distinctions are lost. We propose LAVQ (Locally Adaptive
Vector Quantization), a modification to HNSW that employs a percentile-based
clipping strategy. By dynamically adapting quantization bounds per dimension,
LAVQ ignores statistical outliers to preserve fidelity in the dense regions of the
vector space. We further accelerate search using custom AVX2 integer intrinsics.
On the SIFT1M benchmark, LAVQ cuts memory usage by 3.8× and improves
query throughput (QPS) by 4.4× over float32 baselines, achieving recall compa-
rable to state-of-the-art implementations like FAISS.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper violates the ICLR anonymity and formatting guidelines, so I will not be providing an official review.

### Strengths
N/A

### Weaknesses
N/A

### Questions
N/A

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The authors provide their names in the paper, and the paper does not follow the ICLR template. As such, the paper should be desk-rejected.

### Strengths
NA

### Weaknesses
NA

### Questions
NA

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
It seems that this paper violates the Anonymity requirement.

### Strengths
It seems that this paper violates the Anonymity requirement.

### Weaknesses
It seems that this paper violates the Anonymity requirement.

### Questions
It seems that this paper violates the Anonymity requirement.

### Soundness
2

### Presentation
2

### Contribution
2
