# When Does Reasoning Matter? A Controlled Study of Reasoning’s Contribution to Model Performance

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Large Language Models (LLMs) with reasoning capabilities have achieved state-of-the-art performance on a wide range of tasks. Despite its empirical success, the tasks and model scales at which reasoning becomes effective, as well as its training and inference costs, remain underexplored. In this work, we rely on a synthetic data distillation framework to conduct a large-scale supervised study. We compare Instruction Fine-Tuning (IFT) and reasoning models of varying sizes, on a wide range of math-centric and general-purpose tasks, evaluating both multiple-choice and open-ended formats. Our analysis reveals that reasoning consistently improves model performance, often matching or surpassing significantly larger IFT systems. Notably, while IFT remains Pareto-optimal in training and inference costs, reasoning models become increasingly valuable as model size scales, overcoming IFT performance limits on reasoning-intensive and open-ended tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a thorough controlled study on when and how explicit reasoning via reasoning-augmented distillation, which improves large language model (LLM) performance compared to standard Instruction Fine-Tuning (IFT). Leveraging a synthetic dataset of paired IFT/reasoning teacher outputs across math-centric and general domains, the authors analyze accuracy, scaling, and resource efficiency over 12 benchmarks and a range of model sizes.

### Strengths
1. The paper evaluates over 12 benchmarks spanning multiple-choice and open-ended tasks, including specialized math and general-purpose domains, offering rich coverage. The study spans five model sizes (0.5B-14B) and multiple training protocols.
2. The synthetic data distillation setup carefully isolates the impact of reasoning traces from model/data/compute confounders, streamlining attribution. Training and evaluation use only teacher-generated paired data for apples-to-apples comparison, a significant methodological strength.

### Weaknesses
1. The evaluation relies exclusively on LLM-as-a-judge using Llama-3_1-Nemotron-Ultra-253B-v1. No human annotation or qualitative error analysis is provided. 
2. While the study spans both math-specialized and general tasks, there is still a notable focus on math-centric reasoning. Other challenging reasoning domains (coding, legal, scientific, symbolic reasoning) are not substantially covered. This limits the generalizability of conclusions to all kinds of reasoning.
3. The manuscript briefly discusses sequential versus mixed training protocols and the impact of reasoning ratio, but deeper statistical analysis and explanations for the observed instabilities (such as the abrupt mode-switching in mixed training) are limited to observations rather than mechanistic investigation. For instance, why does accuracy sharply degrade beyond certain data ratios, and is this connected to the optimization landscape?

### Questions
See Weaknesses for details.

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
This paper presents a systematic empirical study on the role of chain-of-thought–like reasoning supervision versus instruction-following tuning (IFT) in LLM distillation. Using a controlled “paired IFT vs reasoning answer” setup from the same teacher, the authors investigate when and how reasoning traces improve downstream performance across (1) model scales (0.5B–14B), (2) training recipes (mono-phasic, mixed, sequential), (3) domains (general vs math), and (4) training/inference cost. Key findings include: 1) Reasoning improves performance primarily for larger models and open-ended/mathematical tasks; 2) IFT is generally Pareto-optimal in compute efficiency, whereas reasoning becomes competitive only at larger scales; 3) Mixed training exhibits occasional synergies but instability; sequential IFT to reasoning does not typically help; 4) Domain-specific math adaptation benefits ≥1.5B models but leads to forgetting for smaller ones.

### Strengths
1. Strong experimental rigor & scale: systematic evaluation across model sizes, training stages, domains, and compute budgets.
2. Controlled setup using paired reasoning vs non-reasoning teacher outputs significantly reduces confounding common in prior work.
3. Experimental findings: Reasoning supervision helps primarily for larger models and open-ended tasks. Instruction tuning remains a compute-efficient default baseline. Moderate reasoning proportions may be optimal in some cases.

### Weaknesses
1. Potential confounding factors remain. Although the work aims to isolate the causal effect of reasoning traces, several design choices may still introduce confounds. For example, the teacher uses different decoding hyperparameters (temperature / nucleus sampling) for IFT vs. reasoning traces, and student/base evaluation settings differ (zero-shot vs three-shot). These may bias learning dynamics in favor of one supervision format.
2. Focus is distillation-only. The conclusions emphasize “reasoning becomes valuable at scale,” but all models are distilled students trained from scratch. It is unclear whether conclusions generalize to: 1) RL-trained models; 2) long-trained base models (beyond 14B); 3) architectures other than Qwen.

### Questions
see weakness

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
This paper investigates when explicit chain-of-thought (CoT) reasoning benefits LLM performance through a large-scale controlled distillation study. Using a single teacher (Qwen3-235B) generating paired IFT and reasoning responses, the authors train student models (0.5B-14B) on general and math-centric data across 12 benchmarks. Key findings: (1) reasoning consistently improves performance, especially on open-ended and math tasks, often matching larger IFT models; (2) IFT remains Pareto-optimal for training/inference costs; (3) reasoning benefits increase with model scale, breaking IFT performance plateaus at 7B+ parameters.

### Strengths
**Comprehensive Empirical Coverage**: The controlled distillation framework is the paper's main strength. By having a single teacher generate paired IFT/reasoning outputs for identical prompts, the authors cleanly isolate the effect of supervision format while controlling for data domain, teacher capacity, and other confounders. This represents a significant methodological advance over prior work comparing models trained on different data mixtures. The study spans 5 model sizes (0.5B-14B), 2 data distributions (general/math), 12 benchmarks across 4 task categories (General-MC/OE, Math-MC/OE), multiple reasoning ratios (0%, 25%, 50%, 75%, 100%), and two training regimes (sequential/mixed). The 70k GPU-hour investment and 1.6M training pairs demonstrate serious experimental commitment. The task-level granularity (Appendix G) adds valuable nuance.

**Practical Cost-Benefit Analysis**: Sections 4.1-4.2 provide actionable guidance by analyzing training/inference FLOPs versus accuracy. The Pareto frontier analysis (Figures 6-7) clearly shows IFT's efficiency advantage while demonstrating reasoning's value at larger scales. The finding that intermediate reasoning ratios (25-75%) achieve Pareto optimality is practically valuable. The analysis of answer length correlating with errors (Figure 8) offers insight into inference cost drivers.

### Weaknesses
1. Lack of formalized scaling laws and reusable framework. Despite extensive ablations across ~50+ configurations, the paper fails to synthesize findings into predictive scaling laws or actionable decision frameworks. For example, an unified equations: The paper observes that reasoning benefits scale with model size and vary by task, but doesn't formalize these relationships (e.g., `Accuracy_gain(N, Task) = f(parameters)`). Researchers cannot predict whether reasoning will help for their specific setup without replicating experiments.
2. Single teacher architecture and No cross-teacher validation: All data comes from Qwen3-235B. Teacher-specific biases, reasoning styles, or artifacts may not generalize to other models (GPT, Claude, Llama families). Testing students trained on Teacher A's reasoning traces with Teacher B's evaluation would strengthen claims.
3. No RL baseline and limited task domains: Frontier labs use RL (GRPO, PPO) for reasoning models. Comparing pure distillation to RL-based reasoning would contextualize practical relevance. Focus on math/general tasks ignores code (brief mention but no systematic evaluation), science, legal reasoning, creative writing, etc.

### Questions
1. How sensitive are results to teacher choice? Would a Llama-405B or GPT-4 teacher yield similar task sensitivities and scaling trends?
2. How does your best distilled reasoning model (14B, 100% ratio) compare to a 14B model trained with RL (GRPO/PPO) on the same tasks?
3. Can you provide fitted equations for the key relationships observed?

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
3

### Summary
The paper presents a controlled, large-scale study comparing instruction fine-tuning (IFT) versus explicit “reasoning”-style supervision across model scales (0.5B–14B), task families (general vs. math), and answer formats (multiple-choice vs. open-ended). Using a single teacher to generate paired IFT and reasoning outputs for the same prompts (1.6M pairs), the authors disentangle supervision format effects from confounders and analyze accuracy–efficiency tradeoffs (training and inference FLOPs). Key findings: (i) reasoning consistently helps, especially on math and open-ended tasks, sometimes letting smaller students match much larger IFT models; (ii) IFT remains Pareto-optimal for cost, while reasoning becomes increasingly valuable at larger scales; (iii) mixed IFT+reasoning has limited stability benefits; (iv) longer generations correlate with higher error rates.

### Strengths
Clean experimental design / strong controls. A single-teacher, paired-output distillation pipeline isolates the causal effect of supervision format, avoiding data-mixture and RL confounds. The scale (1.6M pairs; 12 benchmarks) is substantial and well-documented. 

Actionable takeaways for practitioners. Clear guidance on when to prefer reasoning vs. scaling IFT, with explicit Pareto analyses over training and inference FLOPs. 

Nuanced findings by task family and format. Reasoning helps most on math and open-ended; gains on general MC are modest/inconsistent. Results also cover domain adaptation (math).

### Weaknesses
Judge-model dependence. Heavy reliance on LLM-as-a-judge (Nemotron-Ultra-253B) introduces potential bias and circularity. Robustness to alternative judges or human spot-checks is not reported. 

Single-teacher bias. Using one teacher (Qwen3-235B-A22B) risks transferring teacher-specific inductive biases. Ablations with a second teacher (or ensemble) would strengthen causal claims. 

Mixed-training instability. Useful observation, but practical guidance is thin—why instability occurs and how to stabilize data mixing remains open.

### Questions
Judge robustness. How do results change with (a) a second judge model of a different family/size, set to validate judge decisions? 


Teacher diversity. Did the team try a secondary teacher (e.g., different architecture/training mix)? Even a smaller sample would clarify teacher-bias sensitivity.

### Soundness
2

### Presentation
2

### Contribution
2
