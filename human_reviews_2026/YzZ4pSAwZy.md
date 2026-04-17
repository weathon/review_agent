# CIAR: Interval-based Collaborative Decoding for Image Generation Acceleration

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Auto-regressive (AR) models have recently made notable progress in image generation, achieving performance comparable to diffusion-based approaches. However, their computational intensity and sequential nature impede on-device deployment, causing disruptive latency. We address this via a cloud-device collaboration framework \textbf{CIAR}, which utilizes on-device self-verification to handle two key properties of visual synthesis: \textit{the vast token vocabulary} required for high-fidelity images and \textit{inherent spatial redundancy} which leads to extreme predictability in homogeneous regions, while object boundaries exhibit high uncertainty. Uniform verification wastes resources on such redundant tokens. Our solution centers on an on-device token uncertainty quantifier, which adopts continuous probability intervals to accelerate processing and make it feasible for large visual vocabularies instead of conventional discrete solution sets. Additionally, we incorporate a Interval-enhanced decoding module to further speed up decoding while maintaining visual fidelity and semantic consistency via a distribution alignment training strategy.
Extensive experiments demonstrate that CIAR achieves a 2.18× speed-up and reduces cloud requests by 70\%, while preserving image quality compared to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes CIAR, a cloud–device collaborative framework for AR image generation. A small device model predicts tokens; an on-device Inter-Head outputs probability intervals to decide which tokens can be accepted locally, and which uncertain tokens should be sent to a cloud verifier. A distribution-alignment training is used so the device stays close to the cloud model. The authors claim 2.18× speed-up and 70% fewer cloud requests while keeping quality.

### Strengths
1. Addresses a real efficiency issue in AR models.
2. Clear motivation about spatial redundancy and boundary uncertainty.
3. Some quantitative results show measurable latency improvement.

### Weaknesses
Experiments focus on one dataset with limited metrics; visual quality assessment is also limited, should use some modern metrics like HPS.

### Questions
Refer to Weaknesses.

### Soundness
2

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
2

### Summary
This paper introduces CIAR, a cloud-device collaborative framework designed to accelerate autoregressive (AR) image generation. CIAR addresses the inefficiency of uniform token verification in existing speculative decoding methods by proposing an on-device "Inter-Head" module that outputs continuous probability intervals to quantify token uncertainty. This allows the system to selectively offload only high-uncertainty tokens to the cloud for verification, significantly reducing latency and cloud requests while maintaining image quality. The framework is supported by an interval-enhanced decoding module and a distributionally robust training strategy to align device and cloud models. Experiments show that CIAR achieves up to 2.18× speed-up and reduces cloud requests by 70% compared to state-of-the-art baselines.

### Strengths
1. Practical Impact: The method effectively addresses key bottlenecks in on-device AR image generation (network latency, computational cost) and offers a deployable solution.

2. Comprehensive Evaluation: Extensive experiments across multiple models (LlamaGen, Anole) and metrics (FID, CLIP, speed-up) convincingly demonstrate the advantages over strong baselines.

3. Theoretical Grounding: The appendix provides rigorous mathematical justification for the uncertainty metric and interval fusion operator.

### Weaknesses
1. Ablation Clarity: While ablation studies are included, the relative contribution of each component (Inter-Head, prefix injection, Inter-DRO loss) to the overall performance could be more clearly disentangled.

2. Generalization: Experiments are limited to specific AR architectures (LlamaGen, Anole); it's unclear how well CIAR generalizes to other AR or non-AR generative models.

### Questions
N/A

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
4

### Summary
This paper proposes CIAR (Collaborative Interval-based AutoRegressive Decoding), a cloud-based collaborative framework for accelerating autoregressive image generation. Traditional visual autoregressive models are slow and difficult to deploy on devices. CIAR introduces an interval-based uncertainty quantization head on the device, replacing discrete predictions with continuous probability intervals, thereby efficiently determining which tokens are acceptable locally and which need to be uploaded to the cloud for verification. Experimental results show that CIAR achieves a significant acceleration and reduction in cloud calls while maintaining image quality, improving the practicality and efficiency of visual autoregressive image generation.

### Strengths
1. This article is written clearly and logically.

2. This article analyzes the challenges of deploying autoregressive image generation models on devices and proposes an edge-cloud collaborative framework.

3. The proposed uncertainty estimation method is interesting.

4. Comprehensive experiments demonstrate the superiority of the proposed method compared to other acceleration baselines and its advantage over other uncertainty estimation methods.

### Weaknesses
1. The paper lacks details on efficiency measurement. What hardware was used for latency and speedup testing? This seems to be unmentioned in the paper.

2. As an edge-cloud collaborative method, what is the communication cost of CIAR? Are there any latency tests conducted by deploying CIAR in a real edge-cloud collaborative scenario?

3. Can the proposed method also be applied to the VAR model of next-scale prediction?

### Questions
Please see the weakness

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes CIAR, a cloud-device collaborative interval-based decoding framework designed to accelerate image generation with auto-regressive (AR) models. The method introduces an interval-based uncertainty quantifier (“Inter-Head”) for the device model to filter highly confident tokens, reducing the need for cloud verification. CIAR incorporates a continuous interval-based uncertainty measure, cloud-enhanced decoding with interval-feature conditioning, and an interval-aware DRO loss for distribution alignment between device and cloud outputs. Experimental results on text-to-image tasks demonstrate that CIAR substantially reduces latency and the number of cloud requests while maintaining comparable image quality to state-of-the-art speculative decoding baselines.

### Strengths
The paper clearly identifies bottlenecks in AR-based image generation for on-device deployment, specifically the challenge of excessive cloud requests and inefficiency of uniform token verification, and motivates a focused solution.

The introduction of interval-based uncertainty (via the “Inter-Head”) in the device model is a creative step toward efficiently identifying tokens that can be safely accepted on-device, addressing the challenge posed by large visual vocabularies and spatial redundancy.

The paper goes a long way in justifying and formalizing the proposed continuous interval-based uncertainty metric, providing detailed theoretical rationale, axiomatic desiderata, and mathematical properties (see Appendix A.3). This is elaborated with specific attention to the geometric/statistical meaning of the measure, which demonstrates a serious attempt at rigor (even if some aspects could benefit from deeper empirical probing).

### Weaknesses
Despite the strong baseline coverage, the paper omits a direct comparison and discussion of several directly relevant. In particular, work such as “Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient” (Chen et al., 2025) is highly relevant and should be both cited and empirically compared (e.g., as a baseline in Tables 1 and 2, and in distributional alignment discussions). The failure to engage with such work weakens the claim of CIAR’s advancement.

While the role of prefix rate and threshold is explored (Figure 3, Figure 6), there is little sensitivity analysis on device model depth, Inter-Head capacity, or how interval regularization influences the tradeoff between speed and quality. For instance, it is unclear how the single-layer device model limits or shapes the uncertainty estimation capabilities—could more layers or changes to Inter-Head’s complexity yield diminishing returns or instability?

While the Inter-DRO loss is well-motivated mathematically, the practical effect of its components is not independently quantified nor are variants (e.g., anchor loss alone vs. full Inter-DRO) ablated. Table 1 does not include or discuss what is lost if alignment is not enforced (e.g., mode collapse, sample drift?), and Figure 4 only vaguely notes “consistency” improvements. A major claim—alignment between cloud and device distributions—lacks direct quantitative substantiation or specific examples measuring divergences before/after.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
