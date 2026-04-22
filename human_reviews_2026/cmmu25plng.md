# TAB-DRW: A DFT-based Robust Watermark for Generative Tabular Data

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
The rise of generative AI has enabled the production of high-fidelity synthetic tabular data across fields such as healthcare, finance, and public policy, raising growing concerns about data provenance and misuse. Watermarking offers a promising solution to address these concerns by ensuring the traceability of synthetic data, but existing methods face many limitations: they are computationally expensive due to reliance on large diffusion models, struggle with mixed discrete-continuous data, or lack robustness to post-modifications. To address them, we propose TAB-DRW, an efficient and robust post-editing watermarking scheme for generative tabular data. TAB-DRW embeds watermark signals in the frequency domain: it normalizes heterogeneous features via the Yeo–Johnson transformation and standardization, applies the discrete Fourier transform (DFT), and adjusts the imaginary parts of adaptively selected entries according to precomputed pseudorandom bits. To further enhance robustness and efficiency, we introduce a novel rank-based pseudorandom bit generation method that enables row-wise retrieval without incurring storage overhead. Experiments on five benchmark tabular datasets show that TAB-DRW achieves strong detectability and robustness against common post-processing attacks, while preserving high data fidelity and fully supporting mixed-type features.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TAB-DRW, a post-editing watermarking method for synthetic tabular data that embeds signals in the frequency domain. The method first applies a Yeo-Johnson transform and standardization, then a row-wise discrete Fourier transform, and then modifies selected imaginary parts of DFT coefficients to match pseudorandom bits. A rank-based procedure generates pseudorandom bits without storing per-table keys, thereby improving robustness and memory efficiency. The authors provide theoretical analysis on distortion and robustness, and evaluate TAB-DRW on five benchmark datasets with a range of attacks and baselines.

### Strengths
-The paper is well written and easy to follow.
- The method is lightweight and model-agnostic.
- The paper provides formal bounds on distortion and a lower bound on expected detection statistic.
- Extensive experiments with five datasets, multiple fidelity metrics, many attack scenarios, and comparisons to multiple baselines.

### Weaknesses
-The evaluated attacks are broad and realistic, but the paper does not explore adversaries that aim specifically to target the rank-based bit retrieval or to invert the DFT modification. A discussion or small experiment on adaptive attackers who know the method class but not the key would be useful.
- The theoretical robustness analysis assumes transformed data is multivariate Gaussian. While the paper defends this choice after Yeo-Johnson transform, the assumption could fail on extreme non-Gaussian features. More discussion or a small robustness check under strongly non-Gaussian settings would strengthen the claims.

### Questions
- What is the runtime cost for detection on very large tables (e.g. millions of rows)?
- When converting the data back after watermarking, how often do rounding or clipping steps change the values enough to affect downstream tasks? Are there any cases where this process harms data quality or model performance?
- Have the authors tried stronger adversaries that know the general algorithm and can design perturbations to move ranks across bins? If not, how would the method handle this kind of targeted attack?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of tracing synthetically generated tabular data to prevent its misuse. Existing watermarking solutions are often computationally costly, lack robustness, or cannot handle mixed data types. The authors propose TAB-DRW, a post-editing watermarking scheme that operates in the frequency domain. . Experiments on five benchmark datasets demonstrate that TAB-DRW achieves strong watermark detectability and high data fidelity. The method shows superior robustness against a wide range of post-processing attacks compared to existing techniques. The authors claim TAB-DRW is a computationally efficient and broadly applicable solution for generative tabular data.

### Strengths
1.  The rank-based pseudorandom bit generation is a novel, storage-free mechanism. It enhances robustness by deterministically recomputing bits from stable row statistics, as detailed in Algorithm 3.
2.  The method effectively handles mixed-type data by combining Yeo-Johnson transformation with DFT. This creates a standardized frequency domain for uniform watermark embedding, as shown in Section 2.1.
3.  Experimental evaluation is rigorous, testing against ten distinct attacks and four recent baselines. Table 3 and Figure 4 demonstrate superior robustness across various attack types and datasets.

### Weaknesses
1.  The paper fails to clearly articulate the strategy for selecting specific numerical columns for watermark embedding; these details are only mentioned ad-hoc in the appendix as implementation notes. It is recommended that the main methodology section includes a discussion on the principles of column selection and an analysis of how different selection strategies impact the fidelity-robustness trade-off.
2.  The logic for pseudo-random bit generation in Algorithm 3, particularly the rule for determining bit-pairs based on `k%4`, lacks sufficient theoretical motivation. The paper should provide a more detailed explanation for why this specific mapping is adopted and how it ensures robustness for bit sequences from adjacent bins, for instance, by explicitly connecting it to concepts like Gray codes.
3.  A subtle discrepancy exists between the theoretical analysis (Section 3), which assumes fixed YJT and standardization parameters, and the practical detection process (Appendix D), which re-fits these parameters on suspect data. Although experiments in Appendix D suggest this gap has a negligible impact, the theoretical guarantees in the main text should be more cautiously qualified to clarify that the analysis is conducted under an idealized model.
4.  The paper lacks an impact analysis of the final rounding and clipping step for discrete and bounded features. This non-linear operation could potentially weaken or even erase the watermark signal introduced by small modifications in the frequency domain; a discussion or experimental analysis of its potential effect on watermark detectability is advised.

### Questions
1.  Regarding column selection: Could you elaborate on the strategy for choosing which columns to watermark? Is this selection based on certain feature properties (e.g., variance, data type, correlation with other features)? Have you analyzed the sensitivity of the method's performance to this choice?
2.  Regarding the bit generation algorithm: In Algorithm 3, the bit-pair assignment rule (`k%4 = 0 or 3`) appears designed to make bit sequences of adjacent bins more similar. Could you provide a more formal justification for this choice? Is it an approximation of a Gray code, and have you considered alternative encoding schemes and their effectiveness?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a novel tabular watermarking approach for synthetically generated datasets utilizing invertible transformations such a Yeo-Johnson and discrete fourier transform (DFT). Experiments are shown on a wide variety of datasets as well as attacks to show the fidelity and robustness of the proposed approach.

### Strengths
(A) The paper tackles the difficult issue of handling both categorical and numerical data which can occur in different scales. The use of the invertible YJT handles the scale issue.

(B) Theoretical results are provided for the robustness of the approach under Gaussian noise as well as distortion of the watermarked dataset such as mean of the columns, correlations and the Wasserstein distance with the unwatermarked dataset. 

(C) Experiments are shown on 5 datasets with a variety of competing approaches including MUSE, TabWak, TabularMark and performs well in most settings. These include fidelity and a variety of attacks.

### Weaknesses
(i) The presentation does not show the importance of categorical versus numerical columns for the watermarking results. 

(ii) The paper does not show the performance to typical real-world permutation or spoofing type attacks.

### Questions
(1) How robust is the approach to a permutation attack which is typical? Does the approach assume an ordering of the columns which must be known to align with the secret bits?
(2) How easy is it for an adversary to scrub the watermark(*)? 
(3) How easy is it for an adversary to spoof the watermark(**)? 
(4) How does the approach work with different percentages of numerical versus categorical columns? When gender variables are flipped, do the corresponding row features make sense? 


* Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models. https://arxiv.org/abs/2311.04378
** Adaptive and Robust Watermark for Generative Tabular Data. https://arxiv.org/abs/2409.14700

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a post-editing watermarking method for synthetic tabular data, called TAB-DRW. It embeds watermark signals in the frequency domain by modifying the imaginary components of the discrete Fourier transform (DFT) of the data. The authors provide theoretical guarantees on bounded distortion and robustness to Gaussian noise. Empirically, TAB-DRW outperforms or matches prior methods across multiple benchmarks in both fidelity and watermark detectability.

### Strengths
1. The proposed approach of watermarking tabular data through modifications in its discrete Fourier representation is both novel and well-motivated.

2. The paper establishes theoretical guarantees on distortion bounds and robustness under Gaussian noise.

3. The author conducts extensive experiments to empirically validate the performance of the proposed method. It considers both post-processing and generative watermarking methods as baselines and shows that TAB-DRW consistently matches or outperforms baselines on detectability, fidelity, and robustness.

### Weaknesses
Although the proposed method is both intuitively sound and theoretically well-grounded, I have the following concerns:

1. In line 252, the statement “compute a sum-based score over the selected entries” lacks sufficient detail. The exact formulation of this score is not clearly defined and should be explicitly described.

2. Following 1, if the pseudorandom bit is generated based on a subset of selected entries, the method may be vulnerable to attacks that perturb or alter the values of those specific entries.

3. The post-processing attacks introduced in Appendix F.5 appear relatively weak. For instance, the row-deletion attack removes only 10% (or 20% in Appendix G.3) of the rows, and the column-deletion attack deletes only two columns (three in Appendix G.3). Given that tabular datasets often contain dozens of columns, such fixed and small-scale deletions may not adequately stress-test robustness. A more comprehensive evaluation with varying degrees of row/column deletions would better demonstrate the method’s robustness to post-processing attacks.

### Questions
Could the author elaborate on how/why shrinking the imaginary part by a factor $\delta∈[−1,1]$ helps to limit the distortion?

### Soundness
2

### Presentation
3

### Contribution
2
