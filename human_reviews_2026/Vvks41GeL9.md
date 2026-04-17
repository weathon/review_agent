# DynamicBias: Sequence-Aware Calibrated Watermarking for Large Language Models

- Decision: Reject
- Scores: 6, 8, 4, 4

## Abstract
Text watermarking has attracted significant research interest as a way to mitigate LLM-related harms by enabling reliable identification of machine-generated text. In particular, "green" and "red" vocabulary-partition watermarking, which uses a static bias to skew token sampling toward green tokens and away from red, is a promising approach. However, a persistent trade-off remains: stronger watermarks improve detectability but can harm quality, while weaker ones preserve quality but are harder to detect and easier to remove via paraphrasing. A key reason is that the static bias ignore heterogeneous logit distributions across models, domains, and languages, yielding inconsistent performance and hindering practical deployment. 
Our preliminary investigation shows substantial variability in these distributions and the associated performance disparities, driven by model certainty, measured as the margin between the top logit and the average of a small pool of next-best token logits. Building on this observation, we propose DynamicBias, which calibrates the bias at each step using a sequence-level average of this margin with a single scaling parameter $\alpha$. Theoretically, we show that DynamicBias admits a unique optimal $\alpha$ and increases expected detectability as the sequence-level margin grows. This calibration yields consistent detectability across models and integrates directly with existing vocabulary-partition watermarks, offering a practical solution for real-world deployment. Extensive experiments across four LLMs and three languages demonstrate improved detection with competitive text quality and stronger robustness to paraphrasing.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an enhancement for green/red list text watermarking methods. Instead of using a fixed bias ($\delta$), this method calculates the bias dynamically at each generation step.It is set as a function of the sequence-level average of the model's certainty margin which is the gap between the top logit and the average of the next c logits. This makes the watermark more stable and effective across different models, domains, and languages as demonstrated in thorough experiments.

### Strengths
S1. Clear motivation and writing The paper is clear and well-written. It provides a strong motivation for its approach by demonstrating, in a preliminary study (Section 4.2), that static bias methods have inconsistent performance due to significant variation in logit distributions across models and domains (e.g., C4 vs. GSM8K).

S2. Thorough evaluation. The experimental evaluation is comprehensive and rigorous. The method is tested across: multiple models (Llama, Qwen), diverse tasks (open-ended generation, math reasoning, code generation), multiple languages (English, Japanese, Korean).

S3. Ablation studies. The paper includes excellent ablation studies analyzing the effect of every key hyperparameter, including: scaling factor \alpha, green list proportion \gamma, pool size c, the impact of the sequence-level averaging itself

S4. Theoretical justification. The proposed method is supported by a theoretical analysis (Section 3.4), which shows that dynamic calibration maximizes the expected detection statistic under a quadratic cost budget. (I havent had the time to look at the proofs though).

### Weaknesses
W1. The method shows limited conceptual novelty, as it is quite similar to Morphmark, which also employed a token-level dynamic bias.
The main innovations of DynamicBias appear to be: the use of a sequence-level average ($\overline{m}_t$) rather than an instantaneous signal, the use of a logit margin rather than probability mass.

### Questions
Q1. The core novelty lies in the use of a sequence-level average $\overline{m}_t$. How is this average handled in practice?
Is it reset for every new prompt?
If not, then a very long, high-certainty sequence (e.g., code generation) followed by a short, low-certainty sequence (e.g., poetry) could result in an improperly high bias for the second sequence.
If it is reset, how does the method perform at the beginning of a sequence, when $\overline{m}_t$ is based on only one or two tokens and thus not yet a stable “sequence-level” average?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces DynamicBias, a sequence-aware watermarking framework that adapts the watermarking bias dynamically during generation based on model certainty. Building on the standard KGW green/red list formulation, DynamicBias replaces the static bias δ with a stepwise bias δₜ = α m̄ₜ, where m̄ₜ is a sequence-level average of the logit margin between the top token and its next-best competitors. This sequence-level calibration adjusts watermark strength to the model’s confidence, stronger when the model is certain, weaker when uncertain, yielding more consistent detectability and text quality across models, domains, and languages. Extensive experiments across four LLMs and three languages (English, Japanese, Korean) on C4/mC4, GSM8K/MGSM, and additional summarization and code generation benchmarks demonstrate robust performance gains in both AUC and perplexity, with improved paraphrase resilience.

### Strengths
* The idea of calibrating watermark bias according to sequence-level model confidence is intuitive and effective. By maintaining a running certainty signal, the approach captures longer-range structure missed by token-level entropy gating.
* The experiments are comprehensive, spanning multiple models, languages, and task types (reasoning, summarization, code), and consistently show detectability and robustness improvements with minimal quality loss.
* Compared to token-level entropy- or probability-mass–based scaling, DynamicBias’s use of sequence-averaged logits introduces a smoother and more stable watermarking dynamic that generalizes better across domains.
* The authors validate the theoretical small-bias regime assumptions empirically through linearity and curvature fits, and ablate key hyperparameters (α, γ, c), adding strong credibility and tangible implementation recommendations for practitioners.

### Weaknesses
* The approach is conceptually close to recent entropy- and context-aware watermarking schemes, which makes the distinction of maintaining a sequence-level margin rather than a token-level entropy somewhat incremental. That said, the theoretical framing is elegant, and the study is executed well.
* The paper does not report generation throughput or wall-clock time relative to vanilla watermarking. Even if the per-step overhead is small, empirical confirmation would strengthen claims of practical deployability.
* While results are strong overall, the paper could better quantify trade-offs—e.g., where DynamicBias might underperform in high-entropy tasks or non-English domains (given these were tested).

### Questions
* Could the authors provide runtime comparisons or throughput (tokens/s) relative to static watermarking?
* Are there scenarios (e.g., very short or noisy generations) where sequence-level averaging may dampen the bias too much, hurting detectability?
* When the model’s confidence shifts sharply within a sequence (e.g., from setup to reasoning to conclusion, or natural language/explanation to code), does $\delta_t$ adapt smoothly, or could lag in the running average cause miscalibration? A short visualization of $\delta_t$ evolution over token steps could clarify this.
* Since $\delta_t$ varies across sequences, is any adjustment to the detector threshold or z-score normalization needed to maintain comparable false-positive rates across tasks?

### Soundness
4

### Presentation
4

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
The paper proposed a method to dynamically update the bias $\delta$ in green-red list type watermarking methods. The paper proposes to quantify stepwise certainty using a margin between the top logit and a small pool of next-best logits, and use a sequence-level average of the margin over the generated steps to update the bias $\delta$ at each step. Empirical experiments show that by incorporating the proposed method in updating $\delta$, existing green-red list watermarking methods can improve detectability and generation quality.

### Strengths
The paper clearly explained the existing issue of using a static $\delta$ value in the green-red list type watermarks. The proposed method is clearly motivated. Empirical results demonstrate that introducing the dynamic setting of the $\delta$ value in the existing watermarking method can improve the detectability and text quality

### Weaknesses
The paper conducts experiments across different LLMs and datasets, but there are some related baseline missing and some comparison are not well discussed. Specifically:

1. The proposed method dynamically sets $\delta$ value, there are related baselines, which also dynamically update $\delta$ values,  are not compared. For instance, I think the method in [1], which learns an auxiliary model to predict $\delta, \gamma$ at each step, can serve as a natural baseline. It is briefly discussed in line 59, I agree that the method may not be model-independent, but they do have some pre-trained auxiliary models for the Llama and OPT family in the GitHub repo, so I think a comparison in some families should be doable.

2. In the comparison with MorphMark, the paper states that "MorphMark shows weaker detectability overall". But MorphMark also consistently shows better text quality, e.g., perplexity and acc in Table 1. Is it possible to tune the MorphMark model to achieve similar detectability, then compare text quality, or achieve similar text quality, then compare detectability?


### References

[1] Huo, Mingjia, et al. "Token-specific watermarking with enhanced detectability and semantic coherence for large language models." arXiv preprint arXiv:2402.18059 (2024).

### Questions
Please refer to weakness

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
4

### Summary
This paper proposes DynamicBias, a sequence-aware calibration method for vocabulary-partition (green–red list) watermarking of large language models (LLMs). Instead of using a fixed bias $\delta$ for all tokens, DynamicBias computes a sequence-level average of logit-margin signals to scale the bias dynamically via a single parameter $\alpha$.
The authors show that this formulation admits a unique optimal $\alpha$ under a small-bias approximation, derive basic theoretical properties, and demonstrate improved detectability and robustness with competitive text quality across several LLMs  and datasets.

### Strengths
1.	Simple and Plug-and-Play Implementation: DynamicBias integrates easily into existing KGW-style watermarks without changing the hashing or detection pipeline, making it deployment-friendly.
2.	Comprehensive Evaluation: Experiments cover multiple models, languages, tasks, and paraphrasing attacks. Results consistently show moderate but reliable improvements in AUC with small quality loss.

### Weaknesses
1.	Limited Theoretical Novelty.
The idea of optimizing or calibrating the bias $\delta$ has been explored in prior works such as
Takezawa et al. (2023), Wouters (2024, ICML), and Cai et al. (2024), which already derive or analyze optimal biasing strategies for green–red list watermarking.
Compared to those, DynamicBias mainly introduces a sequence-level averaging heuristic and a light theoretical analysis under a small-bias approximation.
The claimed uniqueness of optimal α is mathematically straightforward and adds limited new insight.

2.	Approximate Theory.
The theoretical results rely on linearization and quadratic loss assumptions, without empirical verification of the underlying cost model. The "unique $\alpha$" result resembles a scaled-Lagrangian derivation rather than a new theoretical framework.

3.	Novelty Relative to Entropy-gated or Dynamic Bias Schemes.
DynamicBias is conceptually close to existing adaptive methods such as SWEET (entropy-gated) and MorphMark (token-level probability scaling). The main distinction—using sequence-averaged margin instead of entropy—may be incremental rather than groundbreaking.

4.	No Discussion of Computational Overhead or Detection Efficiency.
Since margin computation and averaging occur at every decoding step, runtime cost and practical detectability under large-scale inference scenarios are not quantified.


[1] Yuki Takezawa, Ryoma Sato, Han Bao, Kenta Niwa, and Makoto Yamada. Necessary and
sufficient watermark for large language models. arXiv preprint arXiv:2310.00833, 2023.

[2] Bram Wouters. Optimizing watermarks for large language models. In International Conference
on Machine Learning, pages 53251–53269. PMLR, 2024.

[3] Zhongze Cai, Shang Liu, Hanzhao Wang, Huaiyang Zhong, and Xiaocheng Li. Towards better
statistical understanding of watermarking llms. arXiv preprint arXiv:2403.13027, 2024.

### Questions
1.	How does DynamicBias perform relative to theoretically optimized schemes such as Takezawa et al. (2023) or Wouters (2024)? A small comparative or analytical discussion would strengthen the claim of novelty.
2.	The paper assumes a single global $\alpha$. Would learning or adapting α per domain/model further improve performance?
3.	How sensitive is the method to hyper-parameter c (top-k margin pool) in low-entropy or deterministic decoding settings?

### Soundness
2

### Presentation
3

### Contribution
2
