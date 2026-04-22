# ALS-ActLR: Alternating Least Squares based Activation-Aware Low-Rank Model Compression

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Large language models (LLMs) achieve state-of-the-art performance but remain impractical for on-device deployment due to memory and compute constraints, making compression essential. Activation-aware low-rank approximation is promising, yet existing methods follow a two-step \emph{approximate-then-factorize} routine, which couples the factors and weakens preservation of salient activation structure. We present \emph{ALS-ActLR}, which combines a spectral-informed metric transformation (SIMT) with \emph{Activation-aware ALS} to optimize the low-rank factors directly using a tiny calibration set (256 samples). A subsequent uncertainty-weighted distillation stage further recovers lost information by adaptively balancing cross-entropy, knowledge distillation, and feature alignment. Experiments show that \emph{ALS-ActLR} substantially reduces parameters and FLOPs while preserving accuracy and perplexity, consistently outperforming strong baselines. Concretely, on Llama-7B at 60\% compression (i.e., 60\% parameters removed, 40\% retained), it reduces mean perplexity to 27.74 (a 69.0\% reduction over the best baseline) and raises accuracy to 48.92\% (+3.72 points), while achieving the best scores across 40--80\% compression and a wide range of model scales from 1.1B to 13B across multiple families. These results highlight \emph{ALS-ActLR} as a scalable and effective framework for activation-aware compression.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces ALS-ActLR, an activation-aware low-rank compression pipeline for LLMs that directly minimizes activation-weighted error. It first applies a Spectral-Informed Metric Transform to reparameterize the objective using calibration activations, then solves low-rank factors via activation-aware Alternating Least Squares with closed-form updates and convergence guarantees, and finally recovers accuracy using Uncertainty-Weighted Multi-Objective Distillation that adaptively balances CE, KD, and feature-alignment losses. With only about 256 calibration sentences, ALS-ActLR consistently outperforms SVD-based and KD baselines across 40–80% compression and multiple backbones, achieving lower perplexity and higher downstream accuracy while reducing memory and latency.

### Strengths
- Systemic methods with clear theoretical proofs: In this paper, the activation-aware ALS uses closed-form updates with ridge terms, which guarantees a monotonic drop in the objective, and converges to a stationary point. The SIMT step cleanly reparameterizes the goal to focus on realistic activation patterns.
- Extensive and robust empirical evidence: The work is evaluated at 40/60/80% compression across multiple LLM families (1.1B–13B), showing consistent PPL/accuracy gains over strong baselines.
- Great practicality: The method needs only ~256 calibration sentences, uses transparent hyperparameters and rank allocation, and plugs in smoothly with PEFT methods like LoRA for extra tuning.

### Weaknesses
- Lack of clarity on computational overhead
  - The paper introduces a multi-stage pipeline (SIMT, ALS, UW-MOD). Although it notes each ALS iteration is lightweight, a fuller breakdown of the total computational cost across stages would be valuable.
- Lack of comparison with other compression methods
  - The work primarily compares against SVD-based methods. Would it be faster or more accurate than pruning at the same compression ratio, and what concrete advantages does it offer over pruning-based approaches?

### Questions
- As I mentioned above, could you provide detailed computional overhead (time/GPU utilization etc.) for your overall pipeline?
- Have you considered combine your methods with other compression methods? For a target compression or a target inference speedup, is there an optimal hybrid allocation between your ActLR with other methods like pruning?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ALS-ActLR, an activation-aware low-rank compression method for LLMs. The authors integrate a spectral-informed metric transformation (SIMT), an alternating least squares (ALS) factorization using a small calibration dataset, and uncertainty-weighted multi-objective distillation (UW-MOD), to preserve activation structure during compression and to adaptively balance different signals. And their experiments across multiple LLM sizes and families show strong performance and compression efficiency.

### Strengths
- L103-105: Well-motivated and novel activation-aware formulation i.e *directly minimizes activation-weighted error with decoupled factors and adaptively balances heterogeneous losses*.
- Has theoretical analysis on convexity and convergence guarantees.
- Extensive empirical results across models of various sizes, families and compression ratios.
- Efficient calibration using only small number of samples (256).

### Weaknesses
- [**Time complexity analysis**] Consider adding time complexity analysis i.e how much time it will take to run Act-ALS for a model in terms of all the associated variables? Providing practical values should also add good value.
- [**Experiment on different calibration sets**] Complementing Sec 4.3, does authors have understanding on effects of different calibration sets on the performance of method? Or is it calibration-dataset independent! Understanding the effects of various calibration datasets is important.
- [**Experiment on more downstream tasks**]: PPL and commonsense are great additions, can the authors also add some factual/math reasoning benchmarks. I believe it’ll give holistic overview of the paper’s claims quite a bit. Some of the metrics are less explored with compressed models, for eg. [1, 2], thus it's easy to overlook the overall picture.
- [**Expand ablations**] Expand the current ablations of how much benefit comes from each component (as an eg. SIMT alone, ALS alone, UW-MOD alone and their respective combinations). It’s fine if some combinations are not possible.
- [**Experiment on newer models**] Consider adding newer models if possible to strengthen the claim.

1. The Cost of Compression: Investigating the Impact of Compression on Parametric Knowledge in Language Models - https://aclanthology.org/2023.findings-emnlp.349/
2. When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models - https://arxiv.org/abs/2504.02010

### Questions
- For Table 5: Maybe you can have a figure having these details for all the layers and for all the layers the Act-ALS values are least, thus not cherry-picking specific layers. If it’s not possible, please explain further (L363, selected Q layers).
- [**Discussion**] Are you using base models or instruct models? It seems like you are using base models from the descriptions (L297-300) but wouldn't the commonsense task make more sense on the instruct model?
  - Following up, if you are using base models, what additional steps needs to be taken to support compressing instruction based models using low rank approaches. Because, I've observed that almost all the baselines you use also compressed only base models, but in reality people use instruction models more often.

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
3

### Summary
This paper proposes ALS-ActLR, a new framework for compressing LLMs using activation-aware low-rank approximation.
Unlike traditional two-step SVD-based approaches that approximate weights and then factorize, ALS-ActLR optimizes low-rank factors directly. A Spectral-informed metric transformation module is proposed to reparameterize the objective using the activation covariance’s spectral factor, allowing optimization under an activation-weighted norm. Activation-aware alternating least squares is proposed to solve for low-rank matrices iteratively with provable convergence guarantees. Finally, the uncertainty-weighted multi-objective distillation refines the compressed model by balancing cross-entropy, knowledge distillation, and feature alignment losses based on learned uncertainty parameters.
Experiments on multiple LLaMA-based models (1.1B–13B parameters) and seven benchmarks show superior performance in both perplexity and accuracy compared to prior methods.

### Strengths
1. Direct and principled optimization:
ALS-ActLR directly minimizes activation-weighted error using alternating least squares, avoiding the limitations of surrogate SVD methods and preserving important activation structures more effectively.

2. Strong theoretical and empirical foundation:
The method is backed by proofs of convexity, convergence, and sufficient descent, and validated by extensive experiments showing consistent performance gains across compression ratios, model sizes, and benchmarks.

3. Adaptive and efficient distillation:
The uncertainty-weighted multi-objective distillation automatically balances different loss terms, improving optimization stability and robustness with minimal calibration data while enhancing downstream task accuracy.

### Weaknesses
1. Complexity and implementation overhead:
The multi-stage pipeline combining SIMT, ALS, and uncertainty-weighted distillation adds algorithmic and computational complexity, which may hinder adoption in practical deployment settings.

2. Insufficient analysis of sensitivity and cost:
The paper provides limited discussion on hyperparameter sensitivity (e.g., ridge and proximal terms) and does not fully quantify the training or distillation overhead introduced by its uncertainty-weighted update stage. It would be nice to provide runtime and energy-consumption analyses to better quantify the trade-offs between compression ratio, accuracy, and computational efficiency.

### Questions
The paper mentions one additional teacher forward pass per step during uncertainty-weighted distillation. Could the authors quantify the total runtime overhead or training cost compared to standard fine-tuning or LoRA adaptation?

### Soundness
3

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
4

### Summary
This paper introduces ALS-ActLR, a novel LLM compression method that combines a spectral-informed metric transformation (SIMT) with Activation-aware ALS to optimize the low-rank factors. A subsequent uncertainty weighted distillation stage further recovers lost information by adaptively balancing cross-entropy, knowledge distillation, and feature alignment. Extensive experiments are carried out to demonstrate the superiority of the proposed method.

### Strengths
1. This works present a novel method that incorporates SIMT and activation-aware ALS to LLM decompositions. The solution seems sound and plausible.

2. The paper is well-written and easy to follow. However, some abbreviations lack definition before using. Authors are suggested to check all the definition of abbreviations in both *Abstract* and *Text*.

3. Analysis is convincing and ablation study is overall good. But, data points in figures of the ablation part seem excessively scarce. More details need to be reported.

### Weaknesses
Weaknesses are mainly on the evaluation parts:

1. Outdated models. The models in the experiment are outdated, and it differs from modern LLMs. Please use the modern LLMs for the experiments, such as using LLaMA-3-8B instead of LLaMA-7B, LLaMA-2-13B instead of LLaMA-13B. Also, authors are suggested to add some latest MoE models in the experiments.

2. Lack of strong baselines. The experiments lack strong baseline from the latest works (published in an established conference more than 3 months or important preprints), such as SVD-LLM V2 (arxiv) and SoLA (AAAI'25).

3. Lack of generation tasks. I see no tasks in the evaluation are generative tasks. Per my past experience, some methods can have outstanding performance in terms of perplexity and commonsense reasoning (i.e., quiz here), where decomposed LLM only need to generate the next token, but cannot perform well in generative tasks. Authors need to add some generative tasks (arithmetic reasoning and summarization) to demonstrate the LLM's generation ability.

### Questions
See Weaknesses. I'll consider to increase the rating if authors can address my concerns, i.e., add more baselines (at least for the baselines that I give in this review) on the latest models to the paper.

### Soundness
2

### Presentation
2

### Contribution
3
