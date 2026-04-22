# SPECS: Decoupling Multimodal Learning via  Self-distilled Preference-based Cold Start

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 2, 6

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has recently catalyzed a wave of "MLLM-r1" approaches that bring RL to vision language models. Most representative paradigms begin with a cold start, typically employing supervised fine-tuning (SFT), to initialize the policy before RL. However, SFT-based cold start adopts the reasoning paradigm intertwined with task solution and output format, which may induce instruction-style overfitting, weakens out-of-distribution generalization, and ultimately affects downstream RL. We revisit the cold start along two views, its training method and data construction, and introduce the Generalization Factor (GF) coefficient to quantify the generalization capability under different methods. Our empirical study finds that preference–based training methods (e.g. DPO) generalizes better than SFT-based methods in cold start. Motivated by this, we propose $\textbf{SPECS}$—a $\textbf{S}$elf-distilled, $\textbf{P}$r$\textbf{e}$ference-based $\textbf{C}$old $\textbf{S}$tart framework that decouples multimodal learning: (1) generates introspective preference data pairs via self-distillation, avoiding reliance on larger teachers or manual annotation; (2) performs preference–based training to learn, focusing on shallow, transferable surface-form criteria (format, structure, style) rather than memorizing content; and (3) hands off to RLVR for deep reasoning results. Experimental results across multiple multimodal benchmarks show that our decoupling learning framework yields consistent performance gains over strong baselines, improving MEGA-Bench by 4.1\% and MathVista by 12.2\%. Additional experiments indicate that SPECS contributes to reducing in-distribution "stuckness," improving exploration, stabilizing training, and raising the performance ceiling. 

Project Page: https://kwen-chen.github.io/SPECS-VL/

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Currently, with the fast development of RL, its application has expanded to MLLM. Most representative paradigms begin with a cold start under supervised fine-tuning (SFT). A challenge here is SFT cold start may induce instruction-style overfitting, weaken out-of-distribution generalization, and ultimately affect downstream RL. In this view, the authors first introduce the Generalization Factor (GF) coefficient to quantify the generalization capability under different methods, showing that preference-based training can do better than SFT. This further brings a Self-distilled, Preference-based Cold Start framework (SPECS) in preference-based training style.

### Strengths
1. The motivation of this paper is sound, and the observation on SFT cold start is fundamental. The GF criteria, though without significant novelty, is straightforward and useful.

2. After raising the problem, the authors propose an intuitive and effective approach to solving the problem. 

3. The paper is easy to follow.

### Weaknesses
1. The self-distilled dataset needs further discussion. In Sec. 3.2, the rejected responses are responses that also contain the correct answer, but deviate from the required format. In stage 1 training, is format deviation the other problem we need to consider? What about wrong answers, wrong thinking process, etc? From my perspective, the current design might face the challenge of hard negative mining. More discussions are needed.

### Questions
The paper has the finding that stage 1 rl can benefit following rl training. The experiments are sound and the discussions are solid. Overall, it is a good paper.

### Soundness
3

### Presentation
3

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
This paper proposes SPECS, a framework for training multimodal models using DPO and RL. The framework involves first generating preference data from the model, training it on the preference data with DPO, and then performing GRPO training. The authors show that using DPO is preferable to SFT for training an initial RL checkpoint due to its superior generalization, and yields more stable RL training. Their final model outperforms solid baselines or training using only RL.

### Strengths
- Model outperforms prior work and the additional DPO is clearly helpful.
- Ablations show the distillation strategy is quite effective
- Showing DPO generalizes better than SFT with the generalization gap metric is interesting.

### Weaknesses
- I think the work needs more careful ablations, particularly around the self-distillation.
  - Gemini-flash is used as the LM judge when choosing responses. How does this compare to just distilling from Gemini? Is using it as a judge cheaper?
  - The authors do some additional RL to create a model to distil from and claim “[it] is more adept at exploring the solution space”. What is the justification for this? Does distilling directly from the base model (with some careful prompting) perform worse? A stronger justification here would be nice.
  - In table 3, you claim that decoupled outperforms coupled data for cold-starting, but it appears it underperforms the coupled data (slightly) pre-RL training, but does indeed outperform post-RL training. It would be useful to explain why this is! Similarly, considering that the average difference is only around 1.5 points, it would be useful to know what the average noise on the dataset is (i.e., some sort of statistical significance measure on the results - is 1.5 within noise or not?).
- The generalization gap metric is interesting, but its connection with better cold start checkpoints seems tenuous. If we do RL training on only in-domain prompts, does it matter that one method has better OOD performance? A more concrete comparison where you train an SFT model with RL using the same setup as the DPO+GRPO model would be more convincing. The authors have clearly trained an SFT+RL model (used for analysis in Section 4.5), so hopefully it would just be a matter of evaluating that.
- There is some interesting analysis in Appendix A.4 that suggests the DPO better improves pass @ k, but as far as I can see this is never mentioned in the main text (not even a reference to that appendix section itself!)
- The authors note that the DPO-trained model is more stable during RL training, but don’t provide an explanation as to why. It’d be useful to see some sort of analysis or justification for this - perhaps it suggests the hyperparameters for RL training on the SFT model are suboptimal?
- The authors do not give enough details on their experimentation. How many training steps / epochs was training run for for DPO/SFT/RL? For RL, what KL penalty was used, and how long were responses allowed to be? It would also be useful to get some indication of how long SFT/DPO/RL training took in GPU-hours.

Overall, I think the work is reasonable, and the finding around using DPO instead of SFT for cold start is interesting. But more justification around decisions are required, and some clearer ablations.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes SPECS, a cold start strategy for preference-based multimodal learning. They first introduce a generalization factor metric to evaluate models' generation capacity. They then design a 3-stage learning framework: 1) generate format-aware preference data via self-distillation; 2) DPO cold start for learning the format consistency; 3) GRPO to improve deep reasoning.

### Strengths
- Experiments demonstrate the performance gain of the proposed method against selected base models and vanilla GRPO on various benchmarks.
- The paper is overall clearly written and easy to follow.
- Ablation study is conducted to validate the performance of decoupled (format) DPO data.

### Weaknesses
- The paper is not technically sound. It largely reaffirms the well-established observation that RL-based methods improve generalization compared to supervised finetuning,  but does not offer significant new technical insight. 
- The introduction of the zero model (trained with GRPO before self-distillation) appears unnecessary. It is unclear why the chosen and rejected samples cannot be directly generated from the base model, especially since both are later used for format-level preference training.
- The definition of Generalization Factor and lack clarity theoretical justification. The formula in L131–133 appears confusing and unsimplified, possibly due to a typo.
- The proposed method is costly. The cold start stage requires GRPO training on 86k data, as well as gemini-2.5-flash as strong LLM-as-a-judge.
- The effectiveness of DPO cold start is not directly validated; it would be helpful to compare with a control baseline that performs SFT cold start + RL using the same chosen data to isolate the benefit of preference-based training.

### Questions
- For the rejected samples, why are rollouts taken from the base model instead of the exploratory GRPO-zero model?
- Could the authors clarify the mathematical definition and interpretation of GF?
- Why is the zero-model necessary for self-distillation? Given the strong LLM-as-a-Judge, can we directly sample from base model rollout?
- Since you use both SFT and DPO loss for cold start training, how is the format decoupled?

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
This paper introduces SPECS, a Self-Distilled Preference-based Cold-Start framework that rethinks initialization for reinforcement-learning-based multimodal large language models (MLLMs). The authors demonstrate that standard supervised fine-tuning (SFT) couples reasoning content and output format, resulting in overfitting and poor out-of-distribution (OOD) generalization.
SPECS decouples learning objectives through three stages:
1. Self-Distillation – Generates preference pairs focused on output format without large teacher models.
2. DPO-based Pre-Alignment – Applies Direct Preference Optimization with a hybrid (DPO + SFT) loss for stable format learning.
3. GRPO Fine-Tuning – Performs final reinforcement learning with verifiable rewards for deep reasoning.
A new Generalization Factor (GF) metric measures cold-start generalization and correlates strongly (R² ≈ 0.86) with final RL performance.
Across MEGA-Bench, MMMU, MathVista, MathVision, and MathVerse, SPECS achieves consistent gains over strong baselines (e.g., VL-Rethinker, MM-Eureka), improving +4.1 % on MEGA-Bench and +12.2 % on MathVista. Ablations confirm that self-distillation and decoupled data outperform teacher-based and coupled variants.

### Strengths
- The paper tackles a pressing issue in multimodal RL pipelines—how to design an effective cold-start phase for MLLM-r1 models—complementing recent efforts such as Vision-R1 and Advancing Multimodal Reasoning via RL with Cold Start.
- The self-distilled preference data strategy avoids dependence on large teacher models while maintaining strong alignment quality. The decoupled DPO training elegantly separates surface-form learning from reasoning acquisition.
- The proposed Generalization Factor (GF) offers a measurable and interpretable indicator of generalization potential, empirically shown to correlate with final RL performance.
-  Evaluations across multiple reasoning benchmarks, together with well-designed ablations (self-distillation vs. teacher-distillation; decoupled vs. coupled data), substantiate the paper’s claims.
- The writing is structured and transparent, with detailed hyperparameters, dataset composition, and implementation frameworks (MM-EUREKA, LlamaFactory) that support reproducibility.

### Weaknesses
- The related-work section omits recent concurrent methods such as Vision-R1 (Huang et al., 2025), AdaViP (Lu et al., 2025), Chain-of-Focus (2025), and DeepEyes (2025), which would situate SPECS more clearly within the multimodal RL ecosystem.
- Current evaluations emphasize mathematical reasoning; inclusion of more general-domain benchmarks (e.g., ScienceQA, MMBench) would better demonstrate scalability.
- The GF analysis is confined to one model family (QwenVL). Extending the metric across other backbones (e.g., InternVL, Kimi-VL) would confirm its generality.
-  Dependence on GPT-4o and Gemini-2.5 for preference filtration introduces potential bias; multi-judge or calibration procedures could strengthen reliability.
- While the framework improves RL convergence, the additional computational cost of the DPO stage is not explicitly quantified.

### Questions
1. How does SPECS empirically or conceptually differ from Vision-R1 and AdaViP, which also integrate RL and preference-based training?
2. Can the proposed GF metric predict RL success across different architectures (InternVL, Kimi-VL)?
3. What is the size and diversity of the self-distilled preference dataset, and how does scaling influence results?
4. How were biases in GPT-4o/Gemini evaluations mitigated (e.g., standardized prompts or aggregation)?
5. Could SPECS generalize to text-only or perception-heavy domains (e.g., R1-VL, Chain-of-Focus models)?

### Soundness
4

### Presentation
3

### Contribution
3
