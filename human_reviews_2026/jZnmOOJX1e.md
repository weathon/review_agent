# PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4, 4

## Abstract
AutoRegressive Visual Generation (ARVG) models retain an architecture compatible with language models, while achieving performance comparable to diffusion-based models. Quantization is commonly employed in neural networks to reduce model size and computational latency. However, applying quantization to ARVG remains largely underexplored, and existing quantization methods fail to generalize effectively to ARVG models. In this paper, we explore this issue and identify three key challenges: (1) severe outliers at channel-wise level, (2) highly dynamic activations at token-wise level, and (3) mismatched distribution information at sample-wise level. To these ends, we propose PTQ4ARVG, a training-free post-training quantization (PTQ) framework consisting of: (1) Gain-Projected Scaling (GPS) mitigates the channel-wise outliers, which expands the quantization loss via a Taylor series to quantify the gain of scaling for activation-weight quantization, and derives the optimal scaling factor through differentiation. (2) Static Token-Wise Quantization (STWQ) leverages the inherent properties of ARVG, fixed token length and position-invariant distribution across samples, to address token-wise variance without incurring dynamic calibration overhead. (3) Distribution-Guided Calibration (DGC) selects samples that contribute most to distributional entropy, eliminating the sample-wise distribution mismatch. Extensive experiments show that PTQ4ARVG can effectively quantize the ARVG family models to 8-bit and 6-bit while maintaining competitive performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces PTQ4ARVG, the first post-training quantization framework for autoregressive visual generation (ARVG) models. It tackles channel outliers, token-wise dynamics, and sample-wise mismatches with Gain-Projected Scaling, Static Token-Wise Quantization, and Distribution-Guided Calibration, achieving strong 6/8-bit results and real GPU speedup over prior PTQ methods.

### Strengths
1. Novel problem setting, first systematic study of PTQ for ARVG models.
2. Clear motivation with three unique quantization challenges.
3. GPS provides a theory-driven scaling solution rather than heuristics.
4. STWQ and DGC are training-free and hardware-friendly.

### Weaknesses
1. Section 4.2 and 4.3 are relatively short and underdeveloped, lacking detailed analysis.
2. No visual comparison results are provided to qualitatively validate generation quality.
3. 8-bit results still show notable degradation, raising concerns about practical usability.
4. The reported 3.01× speedup at 8-bit is questionable without more deployment details.

### Questions
Please refer to the weakness section.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper is the first to systematically investigate the problem of post-training quantization (PTQ) for Autoregressive Visual Generation (ARVG) models. The authors identify three key challenges when directly applying existing PTQ methods to ARVG models: (1) severe channel-wise outliers, (2) highly dynamic token-wise activations, and (3) mismatched sample-wise distribution information. To address these challenges, the authors propose PTQ4ARVG, a training-free PTQ framework consisting of three core components: (1) Gain-Projected Scaling (GPS), which theoretically derives an optimal scaling factor to mitigate outliers by expanding the quantization loss via a Taylor series; (2) Static Token-Wise Quantization (STWQ), which leverages the fixed token length and position-invariant activation distribution of ARVG models to achieve fine-grained quantization without runtime overhead; and (3) Distribution-Guided Calibration (DGC), which selects more informative calibration samples by maximizing distributional entropy. Extensive experiments show that PTQ4ARVG can effectively quantize various ARVG models (e.g., VAR, RAR, PAR, MAR) to 8-bit and 6-bit, outperforming existing PTQ methods.

### Strengths
1. Pioneering and Important: This paper is the first comprehensive attempt to tackle the emerging and important problem of quantizing ARVG models, a largely unexplored area. With models like VAR and MAR gaining prominence, this work is very timely
2. Deep Problem Insight: The paper does not merely apply existing PTQ methods to a new model class. Instead, it deeply analyzes and identifies three specific, critical challenges (channel outliers, token dynamics, sample mismatch), strongly supporting these observations with visual evidence (Fig. 1). This in-depth analysis is a key strength
3. Solid Theoretical and Methodological Innovation: To address channel outliers, the proposed GPS method is not just empirical; it is mathematically modeled via a Taylor expansion of the quantization loss to derive a closed-form solution for the optimal scaling factor (Eq. 16). This is a solid and elegant theoretical contribution. Furthermore, STWQ is a clever solution that leverages the unique properties of ARVG models ("fixed token length" and "position-invariant distribution across samples") to solve the dynamic activation problem with a static, zero-overhead approach
4. Comprehensive Experimental Validation: The authors conduct extensive experiments on four different ARVG models (VAR, RAR, PAR, MAR) and compare against a wide range of state-of-the-art PTQ methods. The thorough ablation studies (Table 4, 5, 6) also clearly demonstrate the effectiveness of each proposed component

### Weaknesses
1. Performance Not Yet Optimal, Some 8-bit Results Are Not Near-Lossless: Although the proposed method outperforms other PTQ baselines, there is still room for performance improvement. For many quantization applications, 8-bit (W8A8) PTQ is often expected to achieve near-lossless performance compared to the full-precision (FP) model. However, the experimental results show that several models still exhibit a noticeable performance drop even at 8-bit. For instance, on VAR-d24, the FID degrades from 2.33 to 3.36, and on PAR-XL, it degrades from 2.61 to 3.55 (Table 1, Table 2). This suggests that ARVG models may be inherently more sensitive to quantization than other architectures, and achieving truly lossless performance remains a challenge even at 8-bit. This issue is, as expected, exacerbated at the more aggressive 6-bit setting, where the performance drop becomes much more significant (e.g., FID on RAR-B drops from 1.96 to 5.13).
 2. Absence of Comparison with QAT: The paper focuses exclusively on PTQ. While PTQ is valuable for its efficiency and training-free nature, providing a comparison with Quantization-Aware Training (QAT) would offer a more complete picture of the performance-cost trade-off. Even a simple QAT baseline (or even PEFT like Qlora) would help contextualize the performance level achieved by PTQ4ARVG.

### Questions
1. Regarding the GPS Assumption: In the derivation of GPS, you assume that the Hessian of the loss function with respect to the layer's output can be treated as a common constant and thus ignored during optimization. This is a strong simplification. Could you please elaborate on the validity of this assumption? Have you analyzed the actual variation of this Hessian value in ARVG models? How sensitive is the performance of GPS to this assumption? 
2. Pareto Front Analysis vs. Smaller Full-Precision Models: As this is a pioneering work in PTQ for ARVG models, it is crucial to establish the practical benefits of quantization against the primary alternative for efficiency: training smaller, full-precision models. Does a quantized large model (e.g., a 6-bit RAR-XL) offer a better trade-off than a smaller full-precision model (e.g., a FP RAR-B) with a similar FLOPs count? A comparison on a FLOPs-vs-FID plot would be highly valuable to demonstrate that the proposed quantization method truly pushes the Pareto front, rather than just landing on a point that could be achieved by a smaller, unquantized model. 
3. On the Composability of the Methods: The proposed methods are designed to tackle specific challenges in ARVG. We are interested in understanding the composability of these methods with other advanced PTQ techniques, such as low-rank decomposition from SVDQuant or rotation-based methods from QuaRot. For instance, could combining GPS with these approaches lead to further performance gains, potentially achieving near-lossless results at 6-bit and 8-bit? This is crucial for assessing the generality and future potential of your framework
$\textbf{If these issues can be resolved, I will consider giving a higher score.}$

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
PTQ4ARVG addresses the underexplored challenge of applying Post-Training Quantization (PTQ) to Autoregressive Visual Generation (ARVG) models , which share architectural similarities with large language models but achieve visual performance comparable to diffusion-based models. The paper identifies three key obstacles specific to ARVG models that cause conventional PTQ methods to fail: (1) severe outliers at the channel-wise level (due to AdaLN-adjusted activations), (2) highly dynamic activations at the token-wise level (due to positional embedding and sink tokens), and (3) mismatched distribution information at the sample-wise level. To overcome these, the authors propose a tailored PTQ framework, PTQ4ARVG, which consists of Gain-Projected Scaling (GPS), Static Token-Wise Quantization (STWQ), and Distribution-Guided Calibration (DGC). The ultimate goal is to significantly reduce model size and inference latency, enabling efficient deployment of large ARVG models on resource-constrained devices without the need for extensive retraining.

### Strengths
1. High Relevance and Motivation: The work addresses a critical bottleneck—quantization—for deploying large generative models, filling a significant research gap within the emerging ARVG model class.

2. Clarity and Technical Detail: The paper clearly articulates the unique quantization challenges specific to ARVG and proposes technically sound solutions, such as GPS, which optimizes the scaling factor using a closed-form solution derived from a Taylor series expansion.

3. Extensive Robustness Validation: The method is evaluated across a broad family of state-of-the-art ARVG models (VAR, RAR, PAR, MAR), demonstrating consistent superiority over existing PTQ methods at both W8A8 and W6A6 bit-widths.

### Weaknesses
The generative evaluation is confined to the ImageNet dataset and relies heavily on traditional metrics like FID/IS which correlate poorly with human perception, suggesting that the inclusion of modern perceptual metrics (e.g., HPS [1] or CLIP Score) on diverse, high-fidelity datasets is strongly recommended for a more robust comparison.

[1] Ma Y, Wu X, Sun K, et al. Hpsv3: Towards wide-spectrum human preference score. CVPR 2025: 15086-15095.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies PTQ for ARVG models. The authors empirically identify three categories of quantization challenges for ARVG: severe channel-level outliers, highly dynamic activations along the token dimension, and sample-level distribution mismatch. To address these, they propose the PTQ4ARVG framework, which comprises three components, GPS, STWQ, and DGC. Experiments cover multiple ARVG models (VAR, RAR, PAR, MAR), and comparisons under W8A8 and W6A6 settings against baselines such as SmoothQuant, OS+, OmniQuant, QuaRot, and SVDQuant show advantages in generation quality and inference acceleration.

### Strengths
1. Problem formulation is clear and well targeted, The authors carefully analyze how ARVG differs from standard LLMs and diffusion models in activation distributions and token structure, The three proposed challenges at the channel, token, and sample levels capture the essential difficulties of quantizing ARVG, providing a solid basis for method design.
2. Method design has theoretical support, The GPS component is not purely heuristic, it derives an analytic expression for the effect of the scaling factor via decomposition of the quantization loss and a Taylor expansion, and it yields a closed form or solvable expression. This is more convincing than many scaling methods that rely only on heuristic statistics.
3. Practicality and deployment considerations are thorough, The STWQ staticization idea matches ARVG’s fixed sequence length property, avoiding dynamic online calibration overhead, and the authors demonstrate deployment with standard CUDA kernels and real latency and memory measurements, which strengthens the engineering credibility of the work.
4. Comprehensive experimental coverage and ablation, The paper compares several mainstream ARVG models across multiple bit widths (8/6 bit), and provides component-wise ablations for GPS, STWQ, and DGC, showing each component’s contribution to overall performance, The empirical comparison is relatively systematic.

### Weaknesses
1. Several important approximations and assumptions in GPS are not sufficiently validated, GPS omits Hessian cross terms in its derivation, however prior series of quantization works (OBD[1], OBS[2], OBC[3], GPTQ[4]) have pointed out that such omissions can introduce significant errors.
2. The position-invariance assumption underlying STWQ is limited, The authors rely on “position-invariant distributions” as the core justification for STWQ, but the paper only shows statistics for a few layers and a few sample types (Fig.4), There is insufficient validation across models, classes, or conditioning states, If some classes or conditioning strongly affect position distributions, STWQ’s effectiveness may degrade.
3. Baseline comparisons and fairness of hyperparameter / implementation choices are unclear, Several baselines compared (notably OmniQuant, QuaRot, SVDQuant) have multiple implementations and hyperparameter variants in the literature, The paper states “default settings” but does not specify reproduction details, calibration sample counts, or whether fine-tuning was permitted, Some baselines show anomalously poor performance in the tables, which could indicate inconsistent setups or implementation issues, this undermines confidence in the conclusions.
4. Some result presentations and tables lack readability and consistency, Tables 1/2/3 contain many anomalous or extreme values (for example SVDQuant’s IS collapse under W8A8 in Table 1), but the paper does not explain or annotate these anomalies, Table headers and notes are insufficiently detailed, making it hard to trace exact experimental settings such as batch size, calibration sample count, sequence length, and whether embedding / KV-cache were quantized.
5. The method already fails to preserve accuracy compared to full precision at 8 bits in most settings, this greatly affects practicality, The paper also does not explore more aggressive quantization at 4 bits or below.

References:

1. LeCun et al., Optimal Brain Damage.
2. Hassibi et al., Optimal Brain Surgeon and general network pruning.
3. Elias et al., Optimal Brain Compression.
4. Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.

### Questions
1. How large is the effect of omitting Hessian cross terms and assuming Hessian constancy in GPS on the final scaling factors? Can you provide real Hessian statistics or approximations (diagonal, top-k) for several representative layers, such as MHSA, FFN, and projection layers, to validate the second-order Taylor approximation and the omission of cross terms? If the assumptions are invalid, by how much does GPS’s closed-form solution deviate?
2. Does STWQ’s "position-invariant distribution" hold across different models, different conditioning modes (conditional vs unconditional), and different datasets (ImageNet vs more complex scenes)? I suggest providing cross-model and cross-dataset statistics, or at least verifying position distribution stability on the classification/conditioning subsets of PAR, MAR, and VAR.
3. Please provide full reproduction details for baseline implementations and comparisons, including the specific implementation versions, calibration sample counts used for each method, whether KV-cache / embeddings / layernorm parameters were quantized, and whether custom CUDA kernels were used. If possible, include key reproduction scripts or hyperparameter tables in an appendix.
4. For example, in Table 1 the VAR-d16 W8A8 IS has already dropped by nearly 20%, while historically 8-bit quantization is often considered effectively lossless, Why then does the paper still claim maintained competitive performance in the abstract? Please clarify and reconcile these claims with the observed degradations.
5. GPS numerical stability, Formula (16) includes square roots and denominators, In channels with sparsity or where Δx approaches zero, might this lead to numerical instability? What regularization or clipping strategies are used to prevent division by near-zero values?
6. STWQ vs DTWQ trade-off, STWQ avoids online overhead but how does it handle variable sequence lengths or deployment scenarios requiring arbitrary cropping? Do you need to rebuild static tables for different sequence lengths, or is there an adaptive strategy?
7. DGC’s Mahalanobis thresholding, the paper fixes “top 50%” of samples as calibration picks, how was this threshold chosen? Does it need tuning across models or datasets, and is there a more automatic threshold selection method, for example based on distributional entropy elbow points?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes several techniques to improve the quantization of autoregressive visual generation (ARVG) models. The authors identify that a key challenge in quantizing ARVG lies in its highly skewed and dynamic activation distributions. To address this, they introduce a Gain-Projected Scaling (GPS) mechanism that stabilizes quantization and better preserves generative performance. Experimental results demonstrate substantial improvements over existing quantization baselines across multiple ARVG architectures.

### Strengths
* The proposed Gain-Projected Scaling (GPS) offers a principled mechanism for balancing weight and activation scaling by leveraging Hessian information rather than relying on empirical heuristics. This theoretically grounded design suggests strong potential for broader applicability beyond ARVG models.

* Experimental results consistently show that the proposed method achieves significant improvements over baseline quantization approaches, underscoring its effectiveness and practical value.

### Weaknesses
* Achieving the optimal GPS configuration appears challenging compared to empirical estimation. The approach involves several approximations, including those for the Hessian matrix, the upper bound of overall quantization loss, and scaling quantization error. These approximations may introduce discrepancies across different input distributions and norms. Moreover, the final GPS algorithm computes scaling based only on the most significant channel, which may not reflect a global—or even local—optimum. It would be helpful if the authors could include ablation studies illustrating how quantization error changes with GPS.

* The methodological presentation lacks clarity, particularly in Section 4.3, which is relatively brief and insufficiently detailed. The notion of mismatched calibration requires a clearer explanation, as does the proposed distribution-guided calibration and how it specifically resolves this issue.

* The paper does not clearly articulate the unique challenges of quantizing ARVG models. Dynamic ranges and channel-wise outliers are common issues in diffusion model quantization, and the proposed GPS appears to be a general technique rather than one specifically tailored for ARVG. Similarly, the STWQ module seems more like a heuristic quantizer selection strategy based on data distribution. The authors should clarify what makes ARVG quantization distinct and how their method specifically addresses those challenges.

* Empirically, the method shows only marginal improvements except on VAR and PAR models. It would be helpful to clarify whether this is related to baseline implementation or model-specific characteristics.

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
2

### Contribution
2
