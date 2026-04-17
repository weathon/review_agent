# Shift-and-Sum Quantization for Visual Autoregressive Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Post-training quantization (PTQ) enables efficient deployment of deep networks using a small set of data. Its application to visual autoregressive models (VAR), however, remains relatively unexplored. We identify two key challenges for applying PTQ to VAR: (i) large reconstruction errors in attention–value products, especially at coarse scales where high attention scores occur more frequently; and (ii) a discrepancy between the sampling frequencies of codebook entries and their predicted probabilities due to limited calibration data. To address these challenges, we propose a PTQ framework tailored for VAR. First, we introduce a shift-and-sum quantization method that reduces reconstruction errors by aggregating quantized results from symmetrically shifted duplicates of value tokens. Second, we present a resampling strategy for calibration data that aligns sampling frequencies of codebook entries with their predicted probabilities. Experiments on class-conditional image generation, in-painting, out-painting, and class-conditional editing show consistent improvements across VAR architectures, establishing a new state of the art in PTQ for VAR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper applied post-training quantization (PTQ) for visual autoregressive models (VAR), including a shift-and-sum quantization method to reduce calibration data and a resampling strategy for calibration data to align sampling frequencies of codebook entries with their predicted probabilities.

### Strengths
- The paper fills a the gap by explicitly identifying VAR-specific quantization challenges, i.e., the attention-value error amplification at coarse scales and the codebook frequency-probability mismatch. 
- The theoretical analysis is sufficient, e.g., the theoretical analysis (Theorem 1) that proves the error bound for the proposed shift-and-sum quantization.

### Weaknesses
- Insufficient Analysis of Computational Overhead: The proposed shift-and-sum quantization introduces additional operations, such as shift, duplication, and aggregation, which may increase inference time and memory consumption. However, the paper lacks a thorough analysis or empirical evaluation of these computational costs. A detailed study on the overhead introduced by these operations is necessary to fully assess the practicality of the method.
- Qualitative Results Show Noticeable Degradation: The qualitative results presented demonstrate a clear degradation compared to full-precision models. To better illustrate the trade-off between compression rate and generation quality, the authors should provide a comprehensive comparison of generated results across different bit-widths. Additionally, it would be beneficial to include trade-off curves comparing the proposed method with other baseline approaches.
- Limited Evaluation Metrics: The paper primarily adopts FID and IS as evaluation metrics, which mainly assess the generation quality for inpainting and outpainting tasks. However, these metrics do not adequately capture the semantic alignment between the generated results and the conditional guidance. The authors should consider incorporating additional metrics or evaluation protocols to assess semantic consistency and alignment.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work analyzes the post training quantization of VAR models and points out two VAR problems: 1. significant quantization errors from the multiplication between attention scores and value tokens. 2. a mismatch between the predicted probabilities over the entries of VQVAE codebook and their sampling frequencies during calibration (Line 71). The authors propose shift-and-sum quantization which can reduce error with $O(s/4n)$ bound (Theorem 1), and calibration data resample that can resolve the mismatch. The author also provide empirical validations to show that the method improves over BRECQ and is competitive with LiteVAR under W/A bit-widths.

### Strengths
$\bullet$ Theoretical Contribution: this work formalizes coarse-scale attention make post-training quantization error worse, and the authors derived a variance expression (Eq. 8). 

$\bullet$ shift-and-sum kernel has tight error bound: the error bound is tight $|v-f_n(v;t_n) | \leq s/(4n)$

$\bullet$ the calibration fix is simple but effective: I think the resampling technique (probablity-matching) is easy to add to the top of existing post training quantization pipeline. 

$\bullet$ Experiments are comprehensive: the authors conduct experiments on multiple VAR depths, multiple bit-settings, standard metrics (IS/FID/etc.), and qualitative tasks (in/out- painting, editing)

### Weaknesses
Eq. 8 relies on an unrealistic assumption:  $\\{ \tilde{\epsilon\_i^a} \\}\_{i=1}^T$ and $\\{ \mathbf{\epsilon}\_i^v \\}\_{i=1}^T$ are independent, zero-mean random variables with variances $\sigma_a^2$ and $\sigma_v^2$ respectively. I checked the proof of Eq. 8 briefly, and I found the assumption is used at Eq. 22, where $\\mathrm{Var}[ \\sum\_{i} a\_i X\_i ] = \\sum\_{i} a\_i^2 \mathrm{Var}[X\_i] $ (covariance set to 0). If the assumption is dropped, it will not get the same closed form as introducing the covariance term. Furthermore, this assumption seems unrealistic to me, and one proof to break this assumption can be the following: We define $a_t$ and $\\hat{a}_t$ as the exact attention score and quantized attention score at timestep t respectively. Then, the quantization error is defined as $e_t := a_t-\\hat{a}_t$. By the definition of Softmax, $\\sum_i {a_i} = 1$ and $\\sum_i {\\hat{a}_i} = 1$ (softmax of quantized logits), thus we have $\\sum_i {e_i} = \\sum_i {a_i}  - \\sum_i {\\hat{a}_i}  = 0$. We suppose $\\{ e_1, \\dots, e_t \\}$ are independent and at least one had nonzero variance. Then we have $\\mathrm{Var}(\\sum_i e_i) = \\sum_i \\mathrm{Var}(e_i) > 0$. Since $\\sum_i {e_i}  = 0$, we have $\\mathrm{Var}(\\sum_i e_i)  = 0$. This leads to a contradiction. This proof has shown the independence assumption is unrealistic.

### Questions
$\bullet$ Can authors explain the practical validity of the assumption used in Eq. 8? 

$\bullet$ If the Eq. 8 assumption is removed, does it affect the main result, or does it merely complicate the derivation of the bounds?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the challenge of efficient deployment of Visual Autoregressive Models (VAR) by focusing on Post-Training Quantization (PTQ), a technique that enables deep network compression using a small subset of calibration data. While PTQ has shown promise in conventional diffusion models generative models, its application to VAR remains underexplored, primarily due to two critical issues:

* First, significant reconstruction errors arise from the multiplication of attention scores and value tokens in the VAR transformer, especially at coarse scales (low resolutions) where high attention scores are more concentrated—these errors propagate across subsequent finer scales and degrade final image quality. 
* Second, limited calibration data leads to a mismatch between the sampling frequencies of VQVAE codebook entries and their predicted probabilities, biasing quantization parameters and reducing quantization performance.

To tackle these challenges, the paper proposes a PTQ framework tailored for VARs, consisting of two core components: Shift-and-Sum Quantization and Calibration Data Resampling.

Extensive experiments validate the framework on ImageNet across four tasks: class-conditional image generation, image in-painting, out-painting, and class-conditional editing. Evaluations on VARs of varying depths (16, 20, 24, 30 layers) and different bit-widths show consistent improvements over baseline methods (e.g., BRECQ, LiteVAR) in metrics like IS, gFID, and FID2FP16.

### Strengths
In fact, I have a good understanding of autoregressive models, but I am not an expert in the field of quantization. Please correct me if there are any issues with my descriptions.

1/ This paper mainly focuses on optimizing Post-Training Quantization for Visual Autoregressive Models. There is relatively little related work on autoregressive models, so this research is undoubtedly worthy of encouragement .

2/ This work has achieved promising results on ImageNet-256. It outperforms LiteVAR, and the performance improvement is even more significant when combined with LiteVAR.

3/ The analysis of "Reconstruction error across scales" is quite interesting. The authors found that quantization errors are more significant at early (coarse) scales, and based on this observation, they designed the Shift-and-Sum Quantization technique.

### Weaknesses
1/ I have a major question: Since the main purpose of this work is to improve the efficiency of generative models for deployment, why are there no experiments in the paper showing the speed performance or throughput performance of the VAR model after quantization?

2/ Currently, the experiments on VAR are only conducted at a resolution of 256. I am curious whether the results are consistent at higher resolutions. For example, at a resolution of 1024—admittedly, VAR itself has no experiments at 1024 resolution, but Infinity (the text-to-image model of VAR) has experiments at the 1024 resolution version, and it would be valuable to observe the experimental phenomena there.

reference:
Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis

### Questions
None

### Soundness
3

### Presentation
2

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
This paper identifies two major challenges when applying post-training quantization (PTQ) to Visual Autoregressive Models (VAR): (1) large reconstruction errors arising from quantizing the multiplication between attention scores and value tokens, especially at coarse scales where high attention scores are more common; and (2) a mismatch between codebook-entry probabilities and their sampling frequencies during calibration due to limited calibration data. To address these issues, the authors propose two components: a shift-and-sum quantization technique that duplicates and symmetrically shifts attentive tokens (those with high attention scores) to reduce quantization errors, and a calibration data resampling method that reassigns codebook entries to better match predicted probabilities. Experiments across multiple VAR depths and tasks—including class-conditional generation, inpainting, outpainting, and conditional editing—show that the proposed methods consistently outperform prior PTQ methods while maintaining low computational overhead. The approach achieves state-of-the-art PTQ performance on VAR and is complementary to existing methods like LiteVAR.

### Strengths
- The two components proposed in the paper are well-designed to address the specific problems that arise in VAR quantization, and the paper clearly explains how they solve these issues.
- The proposed method appears broadly applicable beyond VAR, with potential usefulness in visual generation and autoregressive modeling in general.

### Weaknesses
- Given the nature of quantization research, more generic and widely applicable methods are preferable. However, the proposed approach is validated only on VAR, making the research scope narrow and potentially limiting its impact.
- Recent PTQ research on transformer quantization is not discussed; the related work mainly covers older studies. Similarly, LiteVAR also focuses specifically on VAR quantization, which suggests that the overall scope of related work is limited.

### Questions
- Could the proposed method be evaluated on other transformer-based models to verify whether it generalizes and improves performance? Although applying it to plain autoregressive generation may be less straightforward, architectures that use multi-scale representations might benefit significantly.
  - The following models might be worth exploring:
    - OneFormer: One Transformer to Rule Universal Image Segmentation, CVPR 2023
    - VGGT: Visual Geometry Grounded Transformer, CVPR 2025
- In BRECQ, the main PTQ evaluation table compares W4A4 and W2A4 settings. It would be interesting to see how the proposed method performs under these quantization settings compared to BRECQ. Can the authors provide results or insights on W4A4 and W2A4 performance?
- How does the proposed method perform quantitatively on inpainting, outpainting, and class-conditional editing tasks? Since the current version mainly focuses on quantization for VAR, it would be valuable to include numerical performance metrics for these tasks, rather than relying solely on qualitative visual comparisons.

### Soundness
3

### Presentation
3

### Contribution
2
