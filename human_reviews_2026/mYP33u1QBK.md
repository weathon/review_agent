# Shuffle-R1: Efficient RL framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Reinforcement learning (RL) has emerged as an effective post-training paradigm for enhancing the reasoning capabilities of multimodal large language model (MLLM). However, current RL pipelines often suffer from training inefficiencies caused by two underexplored issues: Advantage Collapsing, where most advantages in a batch concentrate near zero, and Rollout Silencing, where the proportion of rollouts contributing non-zero gradients diminishes over time. These issues lead to suboptimal gradient updates and hinder long-term learning efficiency. To address these issues, we propose Shuffle-R1, a simple yet principled framework that improves RL fine-tuning efficiency by dynamically restructuring trajectory sampling and batch composition. It introduces (1) Pairwise Trajectory Sampling, which selects high-contrast trajectories with large advantages to improve gradient signal quality, and (2) Advantage-based Trajectory Shuffle, which increases exposure of valuable rollouts through informed batch reshuffling. Experiments across multiple reasoning benchmarks show that our framework consistently outperforms strong RL baselines with minimal overhead. These results highlight the importance of data-centric adaptations for more efficient RL training in MLLM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies two key inefficiencies in reinforcement learning (RL) fine-tuning for multimodal large language models (MLLMs): Advantage Collapsing—where most advantage estimates cluster near zero, weakening gradient signals—and Rollout Silencing—where fewer rollouts contribute useful gradients as training progresses. To address these, the authors propose Shuffle-R1, a data-centric RL framework featuring: (1) Pairwise Trajectory Sampling, which forms high-contrast trajectory pairs by matching high- and low-advantage rollouts to amplify informative signals, and (2) Advantage-based Batch Shuffle, which dynamically reshuffles training batches to prioritize high-value samples based on their advantage magnitudes. Experiments show Shuffle-R1 consistently outperforms strong baselines (e.g., GRPO, DAPO) across multiple multimodal reasoning benchmarks, achieves competitive results with leading closed-source models like GPT-4o and Claude-3.7, and even matches prior methods using only half the training steps—all with minimal computational overhead. The framework also generalizes to text-only LLMs, highlighting its broad applicability.

### Strengths
It demonstrates high quality through rigorous experiments across model scales, datasets, and multimodal benchmarks, supported by thorough ablations and efficiency analyses.
The writing is clear: concepts are intuitively explained, methods are well-structured, and figures effectively illustrate key ideas.

### Weaknesses
Unfair rollout budget: The method uses 16 rollouts per query but only trains on 8, while baselines like DAPO likely use only 8 total. The gains may come from more exploration, not smarter sampling. A fair comparison—using the same total number of rollouts (e.g., 16 for both)—is missing. Also, as N grows large, discarding (1−α) samples wastes potentially useful signals.

Resampling vs. reweighting: The paper uses advantage-based resampling (ABS), but reweighting the loss by advantage magnitude would achieve nearly the same effect with zero extra overhead and less variance. No justification or comparison is provided for choosing the more complex resampling approach.

Marginal gains vs. added cost: Improvements over strong baselines (e.g., DAPO, GSPO) are small—often <1% on average—yet the method adds rollout generation cost (2× more rollouts) and pipeline complexity. If the real benefit is faster convergence (e.g., 2× fewer steps), that should be the highlighted advantage, not final accuracy. As-is, the practical value is unclear.

### Questions
see weakness

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
This paper proposes Shuffle-R1, a data-centric reinforcement learning framework that targets two major issues in current RL fine-tuning pipelines: Advantage Collapsing (most advantages cluster near zero) and Rollout Silencing (the share of rollouts with non-zero gradients keep dropping).

The method introduces two simple modules: Pairwise Trajectory Sampling, which selects high-contrast trajectory pairs to strengthen gradient signals, and Advantage-based Batch Shuffle, which dynamically reshuffles batches to reuse more informative samples.

Experiments across multimodal reasoning benchmarks show consistent gains over GRPO, DAPO, and GSPO with almost no extra computational cost.

### Strengths
1. The paper pinpoints two concrete and observable issues, "Advantage Collapsing and Rollout Silencing", which intuitively explain why current RL pipelines waste computation and fail to leverage informative signals. This diagnostic perspective is well-motivated.

2. Instead of modifying the reward model or policy objective, Shuffle-R1 improves RL efficiency purely from the data side through Pairwise Trajectory Sampling (PTS) and Advantage-based Batch Shuffle (ABS). Both modules are lightweight, easy to implement, and can plug into existing frameworks without architectural changes. And the proposed framework shows consistent improvements across different datasets, model scales, and reasoning benchmarks with less training steps and GPUs.

3. The paper presents clear ablation studies and analyses showing how PTS mitigates advantage collapse and how ABS maintains token utilization over time. This makes the method’s effectiveness both transparent and reproducible.

### Weaknesses
1. **Overfitting to high-advantage samples**
Since ABS repeatedly exposes high-value trajectories, the framework might bias the model toward a narrower distribution of “reward-dense” samples, reducing exploration and long-term diversity. 

2. **Scope of benchmarks**
Most experiments are on math or visual reasoning tasks; while results are strong, these domains already have dense reward signals. It remains unclear whether Shuffle-R1 would bring similar benefits on tasks with sparse or noisy rewards (e.g. open-ended QA, safety...)

3. **Lack of comparison to recent adaptive-sampling paradigms**
Recent works like LIMO also tackle signal efficiency through adaptive data selection and contrastive training. Including these would further enhance impact within the community.

### Questions
Overall, this paper provides a well-motivated and practically useful data-centric perspective on improving RL efficiency for multimodal LLMs, there are several questions:
1. Since ABS repeatedly exposes high-advantage trajectories, how does the framework prevent over-exploitation of a narrow subset of rollouts? 

2. The experiments are convincing but domain specific. Does the author expect the same efficiency gains for RL tasks with sparse, delayed, or non-verifiable rewards (e.g., open-ended QA)? If not, what modifications would be necessary?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Shuffle-R1 is a data-centric RL fine-tuning method for multimodal LLMs that combats Advantage Collapsing and Rollout Silencing. It introduces Pairwise Trajectory Sampling to extract high-contrast advantage pairs and Advantage-based Batch Shuffle to over-sample valuable pairs during mini-batch construction. On 2–30 k training samples, 3/7 B models outperform GRPO, DAPO, GSPO and match GPT-4o/Claude-3.7 on MathVerse, MathVista, etc.

### Strengths
- Diagnose Advantage Collapsing & Rollout Silencing in MLLM-RL; proposes contrastive pairing + advantage-weighted reshuffle instead of larger rollouts or reward re-design.
- Extensive ablations (α, S, PTS variants), 8 datasets, 2 model scales; statistical gains significant; extend to LLMs; code & pseudo-code provided.

### Weaknesses
- While the empirical results are strong, the paper lacks formal analysis or theoretical justification for why PTS and ABS improve training dynamics. For example, it would be helpful to show (even intuitively) how contrastive sampling improves gradient variance or convergence rates.
- While Shuffle-R1 outperforms GRPO, DAPO, and GSPO, it does not compare with other data-centric RL methods such as curriculum-based sampling, which are relevant to the idea of reusing or reweighting data. A short discussion or comparison would strengthen the positioning of the work.
- While ablations on α and S are provided, other design choices (e.g., max-min pairing, absolute advantage weighting) are not thoroughly explored. For example, would cosine similarity or entropy-based weighting perform better?

### Questions
- Have you empolyed Shuffle-R1 on larger models (e.g., 30B+ parameters) or other domains beyond math and vision reasoning? What are the anticipated challenges?
- Can you provide examples where Shuffle-R1 underperforms or fails to improve over baselines? What are the limitations of your method in its current form?
- Can you provide a more formal or intuitive explanation of how PTS improves gradient estimation? For example, how does selecting high-contrast pairs reduce gradient variance or improve signal-to-noise ratio?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Shuffle‑R1, a data-centric RL training wrapper for multimodal LLMs that addresses advantage collapsing and rollout silencing by pairing high-contrast trajectories and shuffling batches based on advantage magnitude. The method is simple, practical, and empirically effective, showing improved accuracy and efficiency over strong RL baselines on math and vision-language benchmarks.

### Strengths
The strengths of the paper are:

- Simple, practical method that is easy to implement on top of GRPO, which addresses a real pain point in RL fine-tuning: many trajectories are statistically uninformative.
- Results are strong and it shows effectiveness on the good coverage of in-domain and out-of-domain benchmarks.
- Detailed experiments with ablation studies.

### Weaknesses
The weaknesses of the paper are:

- Theoretical analysis of bias/variance under selective sampling is limited; unbiasedness is not proven.
- Missing some clarifications and ablation study

### Questions
1. While steps and batch sizes are reported, it is not fully clear that all baselines (GRPO/DAPO/GSPO/RLOO/Reinforce++) were run with identical token budgets, rollout counts (2N=16), and decode temperatures. Small differences can swing math benchmarks materially. Is it possible to add a compute-matched table and a training curve wall-clock plot to substantiate the "7% overhead" claim across settings?

2. HallusionBench results improve, but do PTS/ABS reduce refusal or increase over‑assertion? Any calibration metrics or abstention analysis?

3. How does the pair selection affect solution diversity? Any evidence of collapse in reasoning styles (e.g., fewer distinct CoT patterns)?

4. Are there tasks where advantage distributions are already well‑spread (e.g., dense/step rewards), making PTS/ABS less helpful? Negative results would help practitioners choose whether to use the proposed method.

### Soundness
3

### Presentation
4

### Contribution
3
