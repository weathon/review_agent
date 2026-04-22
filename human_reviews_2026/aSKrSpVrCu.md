# PCDVQ: Enhancing Vector Quantization for Large Language Models via Polar Coordinate Decoupling

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6, 6, 6

## Abstract
Large Language Models (LLMs) face significant challenges in edge deployment due to their massive parameter scale. 
Vector Quantization (VQ), a clustering-based quantization method, serves as a prevalent solution to this issue for its extremely low-bit (even at 2-bit) and considerable accuracy. 
Since a vector is a quantity in mathematics and physics that has both direction and magnitude, existing VQ works typically quantize them in a coupled manner. 
However, we find that direction exhibits significantly greater sensitivity to quantization compared to the magnitude. 
For instance, when separately clustering the directions and magnitudes of weight vectors in LLaMA-2-7B, the accuracy drop of zero-shot tasks are 46.5\% and 2.3\%, respectively. 
This gap even increases with the reduction of clustering centers. 
Further, Euclidean distance, a common metric to access vector similarities in current VQ works, places greater emphasis on reducing the magnitude error. 
This property is contrary to the above finding, unavoidably leading to larger quantization errors. 
To these ends, this paper proposes Polar Coordinate Decoupled Vector Quantization (PCDVQ), an effective and efficient VQ framework consisting of two key modules: 1) Polar Coordinate Decoupling (PCD), which transforms vectors into their polar coordinate representations and perform independent quantization of the direction and magnitude parameters.
2) Distribution Aligned Codebook Construction (DACC), which optimizes the direction and magnitude codebooks in accordance with the source distribution. 
Experimental results show that PCDVQ outperforms baseline methods at 2-bit level by at least 1.5\% zero-shot accuracy, establishing a novel paradigm for accurate and highly compressed LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes PCDVQ (Polar Coordinate Decoupled Vector Quantization), a novel post-training quantization (PTQ) framework for compressing large language models (LLMs).
The key insight is that a vector’s direction is more sensitive to quantization errors than its magnitude, yet most existing vector quantization (VQ) methods couple them together and use Euclidean distance, which overemphasizes magnitude errors.

To address this, the authors:
1) Propose Polar Coordinate Decoupling (PCD) — representing weights in polar form and independently quantizing direction and magnitude, allocating more bits to direction.
2)Introduce Distribution-Aligned Codebook Construction (DACC) — building codebooks aligned with theoretical distributions: E8 lattice-based greedy sampling for direction and Lloyd-Max quantization for magnitude.

Extensive experiments on multiple LLMs (LLaMA-2/3, Mistral) show consistent improvements in 2-bit quantization performance, without introducing extra inference cost.

### Strengths
Mathematical rigor: The decomposition of quantization error and the codebook derivations are theoretically justified.

Strong empirical validation: Broad experiments on LLaMA-2/3 and Mistral confirm robustness and generality.

Practical efficiency: PCDVQ maintains inference speed while achieving higher accuracy and compression.

### Weaknesses
Limited discussion on scalability: While the method performs well at 2–2.25 bits, it remains unclear whether benefits persist at moderate bitwidths (e.g., 3–4 bits) or in activation quantization.

Dependency on Gaussian regularization: The approach assumes weights approximate a standard Gaussian distribution after the randomized Hadamard transform; it would be useful to test models with non-Gaussian weight distributions.

Overlap with QuIP#: Many technical components (e.g., E8 lattice codebook and fine-tuning scheme) are similar to QuIP#. The paper does not sufficiently clarify the conceptual distinction and the essential novelty beyond reinterpreting QuIP# in polar coordinates.

Reproducibility and code release: The paper does not explicitly mention whether code and trained quantization configurations will be made publicly available, which is important for validation and adoption.

### Questions
Q1:How does the method perform at moderate bitwidths (e.g., 3–4 bits) and for activation quantization?

Q2:How sensitive is PCDVQ to the Gaussian regularization step? What happens if it is omitted or replaced with another normalization?

Q3:What is the essential novelty beyond QuIP#? How does the method conceptually differ from QuIP# despite using similar components like the E8 lattice codebook and fine-tuning scheme?

Q4:Will the authors release code, pretrained codebooks, and fine-tuning scripts to ensure reproducibility?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Polar Coordinate Decoupling Vector Quantization (PCDVQ), a method designed to improve the accuracy of low-bit quantization for Large Language Models (LLMs). The core objective is to mitigate the substantial performance degradation faced when compressing LLMs to extremely low bitrates (e.g., $\leq 2.5$ bits). PCDVQ addresses this by observing that a vector's direction is significantly more sensitive to quantization error than its magnitude. It thus proposes to decouple the vector into polar coordinates (direction and magnitude) for independent quantization. Empirically, PCDVQ demonstrates superior accuracy retention compared to existing quantization methods.

### Strengths
**Novel and Well-Motivated Decoupling Mechanism:** The fundamental insight that vector direction and magnitude exhibit different quantization sensitivities is novel for LLM quantization and provides a strong, intuitive justification for the methodological decoupling. Quantizing these components separately via polar coordinates is a sound solution to preserve the crucial directional information.

### Weaknesses
**1. Lack of Robust Experimental Validation on Difficult Benchmarks:** The experimental evaluation is limited to zero-shot multiple choice benchmarks. To fully validate the method's contribution, the paper must be evaluated on more difficult and diverse reasoning benchmarks like MMLU and GSM8K, where small quantization errors often lead to catastrophic failure. The reported accuracy improvement also appears marginal on the limited set of reported tasks, necessitating further validation.

**2. Missing System-Level Inference Evaluation and Comparison:** The inference speed (or throughput) of the quantized model is a crucial component for any quantization algorithm. The paper currently lacks a systemized analysis and comparison of the PCDVQ inference latency against competitors. This absence makes it impossible to assess the practical, end-to-end efficiency trade-off of the proposed method.

**3. Unclear Experimental Consistency in Fine-tuning:** The paper does not clearly articulate whether the same post-quantization fine-tuning methods (if any were used) were applied across all compared baselines (e.g., GPTQ, AQLM, GPTVQ) and PCDVQ. Without explicitly confirming that all methods were compared under the same training/fine-tuning regime, the claimed accuracy improvements may be due to differences in the fine-tuning processes rather than the core PCDVQ mechanism.

### Questions
1. Could you provide a detailed analysis of the inference speed of PCDVC? I want to check if it slows down the model’s inference speed compared to the existing method.
2. Could you provide experimental results on MMLU and GSM8K, or other considerable benchmarks?
3. Could you clarify the fine-tuning recipe for all competitors? For example, did you perform fine-tuning after quantizing GPTQ or GTPVQ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Preserving direction is more important than preserving magnitude in vector quantization. However, current VQ methods emphasize reducing the magnitude error. PCDVQ utilizes polar coordinates to enhance the expressivity of codebook for directional information. PCDVQ also shares codebook for entire model to minimize the memory consumption by regularize all weights to follow the same Gaussian distribution. Experimental results show that PCDVQ achieves the state-of-the-art performance.

### Strengths
1. The idea of decomposing the expressive power of the codebook into fine-grained components (direction and magnitude in PCDVQ) is innovative.
2. Regularizing the distribution of weights to share the codebook sounds solid and effective.

### Weaknesses
1. The paper lacks a theoretical analysis on why directional information is more important than magnitude information.
2. The efficiency should be compared not only with the full-precision model but also with methods such as VPTQ.
3. (minor) Citation format seems inappropriate. Should have used \citep instead of \cite.

### Questions
1. How are the direction and magnitude individually quantized in Figure 1(a)?
2. How does PCDVQ determine bit widths $a$ and $b$ for direction and magnitude?
3. It appears that using a polar coordinate representation requires additional computation during dequantization. Does this make the method slower than VPTQ?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work is motivated by the observation that in vector quantization, the directional component is more sensitive to quantization errors than the magnitude component. Existing Euclidean distance based quantization methods primarily focus on minimizing magnitude errors, which contradicts this finding and consequently leads to larger overall quantization errors. To address this issue, the paper proposes a polar decoupled vector quantization framework, which achieves satisfactory results across multiple experimental settings.

### Strengths
1. Introducing polar decoupled vector quantization is an interesting and novel attempt.

2. The overall writing is clear and easy to follow.

3. The method demonstrates superior performance on several large language models, including LLaMA-2/3 and Mistral, achieving better zero-shot accuracy and perplexity at the 2-bit weight quantization level compared with existing state-of-the-art quantization approaches, which validates its effectiveness.

### Weaknesses
1. The PCDVQ framework introduces additional computational steps, including polar coordinate conversion, two independent codebook searches using cosine similarity and Euclidean distance respectively, and possible inverse conversion. The paper reports improved throughput mainly due to reduced memory bandwidth, but it does not quantify the impact of these added operations on single inference latency. This is important because on many edge devices compute cost is more critical than memory bandwidth. The authors should provide a detailed latency breakdown, including wall clock time per layer for conversion, codebook lookup, and inverse mapping, measured on representative CPU and low-power GPU hardware, and compare end-to-end latency and energy consumption with baseline methods.

2. The effectiveness of the DACC module relies on the assumption that weight vectors, after a random Hadamard transform, follow an approximate standard Gaussian distribution. It is unclear whether this approximation holds uniformly across all layers and architectures (for example, different LLaMA and Mistral variants). It would be better to include empirical diagnostics showing distributional statistics (mean/variance/skewness/kurtosis) of transformed weights per layer and per model, and discuss cases where the Gaussian approximation breaks down and how that affects quantization error.

3. All experiments are conducted on decoder-only Transformer large language models and evaluated on a limited set of zero shot tasks and language modeling benchmarks. It remains an open question whether the direction magnitude decoupling idea transfers to encoder models such as BERT, Vision Transformer models, or multimodal models. These models have different activation and weight statistics and different sensitivity to quantization. 

4. For tasks that require stronger reasoning ability, such as mathematical reasoning or code generation, it is important to know how PCDVQ affects fine-grained semantic fidelity. The current evaluation set does not cover these demanding reasoning tasks. The paper would be stronger if it reported results on a suite of hard reasoning benchmarks and provided error analyses that reveal whether performance degradation (if any) is systematic and whether it is attributable to directional quantization errors or to capacity limits of the codebooks.

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces PCDVQ, a post-training weight-only quantization framework for LLMs that operates in polar coordinates: each weight vector is decomposed into direction and magnitude, which are quantized independently with a larger bit budget for direction. To reduce distortion, the method builds distribution-aligned codebooks. Across multiple LLMs and benchmarks in low-bit settings, PCDVQ consistently outperforms strong VQ and SQ baselines, indicating that decoupling and aligning to componentwise distributions yields better accuracy at very low precision.

### Strengths
1.The paper shows direction is markedly more sensitive to quantization than magnitude, and analyzes why Euclidean MSE emphasizes magnitude errors more strongly, supporting the decoupling design. The motivation of the work is clear and reasonable.

2.This work provides a clear and comprehensive theoretical foundation for polar coordinate decoupling, demonstrating strong depth and theoretical rigor.

3.Across multiple LLM families and standard zero-shot benchmarks, the main results tables show that PCDVQ generally matches or surpasses strong low-bit VQ/SQ baselines.

### Weaknesses
1.The choice to allocate more bits to direction is well supported by experiments, but the paper offers no formal analysis to guide the split or to select an optimal allocation under different conditions.

2. The method adopts a fixed vector dimension and borrows several settings from prior work, but it remains unclear how to adapt the dimension or the direction–magnitude bit split across model sizes, layer types, or differing weight statistics. Robustness to these design choices is not systematically examined.

### Questions
1.How sensitive is PCDVQ to design choices such as the direction similarity metric, codebook size, and the vector dimension, and are there general guidelines for setting these across models?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose to decouple magnitude and direction quantization when performing vector quantization in large language models (LLMs). The proposed pipeline transforms vectors of weight matrices into polar coordinates and quantizes the magnitude and direction separately, taking into account their distinct statistical distributions. The paper presents a persuasive comparison with other scalar-based and vector-based quantization methods across a range of model sizes.

### Strengths
- The paper provides an insightful observation about vector-based quantization: the difference in the approximation behavior of direction and magnitude.
- The idea of decoupling magnitude and direction components and handling their different distributions is creative and conceptually elegant.
- The proposed method achieves strong performance compared to established baselines.

### Weaknesses
- The experiments primarily focus on a range of LLaMA models, with only a single Mistral experiment included. Broader evaluation across different architectures would strengthen the paper.
- The comparison with scalar-based quantization methods is limited and could be expanded for a fairer assessment.
- While the idea is simple and well-motivated, its conceptual simplicity raises questions about whether it is substantial enough for a full-length scientific paper.

### Questions
-Is there a difference in inference speed between scalar-based and vector-based quantization methods? If so, wouldn’t it be fair to include that in the comparison?
- Throughout the paper, you mention quantizing model weights one-by-one. Did you mean layer-by-layer?
- Were the results in Table 1 and Table 2 obtained without fine-tuning?
- Why was QuaRot not included in the experimental comparison, despite being mentioned in the paper?

### Soundness
3

### Presentation
3

### Contribution
3
