# HashMark: Watermarking Tabular/Synthetic Data For Machine Learning Via Cryptographic Hash Functions

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
As enterprises increasingly rely on data for decision-making and machine learning pipelines, ensuring data provenance, ownership, and responsible use has become essential. Data watermarking offers a promising solution by embedding imperceptible markers into datasets, enabling traceability and accountability. While prior work has primarily focused on perceptual domains such as images, audio, and text, watermarking for tabular data remains underexplored despite its central role in enterprise systems. Tabular data presents unique challenges due to its heterogeneity, lack of redundancy, and susceptibility to structural modifications.

We introduce $\mathsf{HashMark}$, a suite of cryptographic watermarking protocols explicitly designed for tabular datasets. Our methods embed bits into table cells using seeded hash functions, achieving \emph{data-type agnostic}, high-fidelity watermarking with minimal distortion. We present two complementary schemes: (i) $\mathsf{HashMark}_1$, a sparse embedding mechanism that modifies only $\Theta(1)$ cells, and (ii) $\mathsf{HashMark}_2$, a dense embedding mechanism that enforces uniform statistical properties across the dataset while supporting categorical and alphanumeric domains. Both schemes feature low detection cost, broad applicability, and formal fidelity guarantees.

Extensive experiments across various settings demonstrate that $\mathsf{HashMark}$ maintains downstream model performance while significantly improving the quality of the watermarking scheme, when compared to prior work. Our results establish hash-based watermarking as a simple, efficient, and general solution for securing tabular data against unauthorized use, while also enabling scalable data governance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a watermarking method for tabular data that modifies the data so that their hash values map to desired target bits. The idea is very similar to the red/green list watermark for LLMs. When inserting the watermark, they aim to make sure that each row (or cell) hash to the same target bit. Detection can then be conducted by counting the proportion of rows/cells whose hash values map to that target bit.

### Strengths
The paper is clearly written and easy to follow. 

The detection algorithms are supported by mathematical theorems, though they rely on idealized assumptions about the hash function.

The algorithms are validated by various experiments.

### Weaknesses
1. When using the algorithm, the fidelity loss for the categorical data is too large. For example, in the HashMark_2, as indicated in the page 6 of the paper, all values are required to have the property of hashing to a desired bit. This essentially requires to modify about 50% of the data, which is a relatively large proportion.  Also, after applying the proposed algorithm, some of the categories will disappear. Therefore this algorithm seems not to be practical for categorical data.

2. For the numerical data, in the fidelity guarantee (theorem 1), we can see that there is a term $ln N$, where N is the size of the dataset. This seems to be strictly worser than the fidelity guarantee obtained in the previous work (https://arxiv.org/pdf/2405.14018), in which the fidelity guarantee does not rely on N. 

3. Missing related works: 
There are several other important recent tabular data watermark works this paper does not cite and discuss. For example:
https://openreview.net/forum?id=71pur4y8gs
https://openreview.net/pdf?id=3K4oAgZTcO
Please also cite and discuss the paper's advantage/disadvantage compared to these papers.

4. There are many errors in the theoretical part of this paper. For example, in the end of page 22,
$0.5+ \frac{\ln {N} +1}{ \ln {2}}$ is not smaller or equal to $\ln {N}+2$. In the proof of theorem 3, you use approximation when calculating the expected number of tries, however in the main theorem there is no $\approx$.

### Questions
The paper relies on a too idealized hash function assumption, saying any unwatermarked dataset can be split such that approximately half of the data are mapped to bit 0 and the other half to bit 1. This assumption seems to be very strong. It will be helpful to discuss in more detail from a cryptographic perspective, whether such an assumption makes sense or not.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes HashMark, a watermarking scheme for tabular data using cryptographic hash functions. HashMark₁ embeds pseudorandom bits in sparse locations, while HashMark₂ embeds a global bit across all cells using hash-based mapping and LSB perturbation (adding 10^-c) or rejection sampling. Detection uses cryptographic verification (HashMark₁) or statistical z-tests (HashMark₂). The authors claim advantages over recent work in simplicity, detection cost, and data-type support.

### Strengths
- Low detection cost: No model needed to embed the watermark 
- Data-type flexibility: Hash functions naturally handle numeric and categorical data 
- Simplicity: Easier to implement.

### Weaknesses
- Limited novelty: Hash watermarking, LSB modification, rejection sampling.. This is engineering refinement, not a research contribution.
- Misleading threat model: "Non-adversarial enterprise settings" assumption (lines 65-70) contradicts security claims throughout the paper. - If employees don't attack, why need cryptographic watermarks?
- Missing recent works such as TabWak and RINTAW.

### Questions
- What happens to robustness when data is rounded to 2-4 decimal places (common in practice)?
- “Zheng et al. (2024) requires the full source dataset.” What is the reason for this requirement?
- What is the performance of recent tabular diffusion models (such as TabDDPM, TabSyn, and TabDiff) when evaluated on both watermarked and non-watermarked data?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper works on the problem of watermarking tabular data to ensure provenance and ownership, a task complicated by data heterogeneity and the low redundancy of tables. The authors introduce HashMark, a suite of watermarking protocols based on seeded cryptographic hash functions. Experimental results demonstrate that HashMark maintains high fidelity, with minimal impact on downstream machine learning model performance.

### Strengths
1. The method uses seeded hash functions for data-type agnostic watermarking. This is a simple yet effective approach applicable to numerical and alphanumeric data without complex data-specific transformations.
2. The paper provides formal fidelity guarantees. Theorem 1 bounds the expected L-infinity error for numerical perturbations, while Theorem 2 bounds the distributional shift for alphanumeric data using Jensen-Shannon Divergence.

### Weaknesses
1.  The description of Algorithm 1 is confusing, as it appears to mix elements of `HashMark1` (looping to `l`) and `HashMark2`, and its input parameters suggest it is designed for synthetic data generation. Please provide a separate, clear pseudocode for the embedding process of `HashMark2` on an **existing dataset**.
2.  The `Generate` function for alphanumeric data relies on rejection sampling from an underlying distribution `ρ`. The paper does not adequately explain how to obtain or approximate this distribution `ρ` when watermarking a **pre-existing, static tabular dataset**, which is a critical detail affecting the method's practicality and reproducibility.
3.  The proposed method of preserving correlations through row-wise rejection sampling is computationally expensive. A more in-depth quantitative analysis of the trade-off between the threshold `t`, computational cost, and the preservation of the joint data distribution beyond downstream task accuracy would strengthen the paper.

### Questions
1.  For watermarking a text column in an existing table (e.g., product descriptions), how do you propose to construct or estimate the probability distribution `ρ` for the `Generate` function's rejection sampling?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper “HashMark” presents a simple, hash-based framework for watermarking tabular and synthetic data to ensure provenance and ownership. Using seeded cryptographic hash functions, it embeds watermark bits with minimal distortion while preserving data utility. Two schemes are proposed: HashMark1 (sparse, high-fidelity) and HashMark2 (dense, statistically testable and type-agnostic). Theoretical bounds guarantee low perturbation, and experiments on multiple datasets show negligible (<0.5%) impact on ML accuracy. Compared with prior works such as Ngo (2024), He (2024), and Zheng (2024), HashMark offers broader data-type support, lower detection cost, and higher practicality for data governance.

### Strengths
1 Conceptual simplicity with wide scope: A seeded cryptographic hash as a unifying primitive for tabular watermarking is elegant and data-type agnostic.

2 Two complementary schemes: Sparse vs. dense addresses fidelity vs. robustness trade-offs.

3 Low-overhead detection: Moving from red/green binning to seed-hash + z-test lowers reliance on source data and reduces metadata burdens.

4 Practical angle: Results target non-adversarial enterprise governance (provenance auditing) rather than worst-case adversaries, which fits many real deployments

### Weaknesses
1 Adversarial robustness under-tested. While Theorem 3 and Proposition 1 give effort bounds, the empirical section lacks adaptive-attack evaluations (e.g., targeted noise, column selection, row injection crafted with knowledge of the scheme but not the seed). Robustness claims for HashMark2 rely on independence/CLT assumptions and ideal hashing; correlated columns or heavy-tailed noise could break z-test sensitivity. More thorough stress tests would strengthen soundness.


2 Categorical scope caveat. The paper recommends avoiding fixed-range categorical columns to prevent range gaps and distribution skew, which limits applicability in many tabular domains rich in discrete attributes.


3 Comparisons are partly qualitative. Head-to-head empirical comparisons to Zheng et al. (2024, TabularMark) and He et al. (2024) are minimal; the paper largely contrasts designs and reproduces Ngo et al. plots. A unified benchmark with common seeds/attacks would better substantiate superiority claims.


4 Correlation preservation is argued heuristically. The constrained-sampling strategy (threshold t) is practical, but correlation preservation is not theoretically guaranteed; guidance on selecting t versus expected z-score/utility would help. Runtime overheads grow rapidly at high t on some datasets.


5 Key/seed management not discussed. Deployment details (seed rotation, namespace collisions across datasets, auditing protocol) are not specified; these are important for provenance at scale.

### Questions
Q1 Can you provide adaptive-attack experiments (seed unknown) that optimize row additions/column removals/targeted noise to defeat the z-test while minimizing utility loss?

Q2 How sensitive is detection when columns are correlated or when hash inputs are truncated/normalized upstream (ETL effects)?

Q3 For fixed-range categorical columns (e.g., ICD codes), could you propose a lossless embedding variant (e.g., parity-preserving mapping, ECC over categories) instead of skipping them?

Q4 Could the z-test be extended to a likelihood-ratio test over all columns jointly to improve power under dependence?

Q5 What are operational protocols for seed management (per-dataset seeds, rotation, collision handling) and for public vs. private watermark verification?

### Soundness
3

### Presentation
3

### Contribution
3
