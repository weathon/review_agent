# Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks

- Decision: Accept (Oral)
- Scores: 8, 6, 6, 6

## Abstract
Empirical scaling laws have driven the evolution of large language models (LLMs), yet their coefficients shift whenever the model architecture or data pipeline changes.
Mixture‑of‑Experts (MoE) models, now standard in state‑of‑the‑art systems, introduce a new sparsity dimension that current dense‑model frontiers overlook.
We investigate how MoE sparsity influences two distinct capability regimes: memorization skills and reasoning skills.
By training MoE families that vary total parameters, active parameters, and top-$k$ routing under fixed compute budgets, we disentangle pre-training loss from downstream accuracy. 
Our results reveal two principles. First, Active FLOPs: models with identical training loss but greater active compute achieve higher reasoning accuracy. Second, Total tokens per parameter (TPP): memorization tasks improve with more parameters, while reasoning tasks benefit from optimal TPP, indicating that reasoning is data-hungry. 
Neither reinforcement learning post-training (GRPO) nor increased test-time compute alters these trends. 
We therefore argue that optimal MoE sparsity must be determined jointly by active FLOPs and TPP, revising the classical picture of compute-optimal scaling. 
All code, data sources, and logs are released to facilitate reproducibility and future work.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper explores how the sparsity of MoE models affects their performance on downstream tasks. While prior scaling law works have mostly focused on pretraining loss or efficiency, this work reveals that downstream memorization and reasoning capabilities respond differently to sparsity. The authors conduct very extensive empirical analysis on a range of benchmarks. Through various analyses, they observe interesting points (see Strengths section) that could be further explored and taken into consideration when building or deploying MoEs.

### Strengths
The paper presents several insightful findings based on systematic analysis of MoE sparsity across a variety of experiments.

- Active FLOPs has a significant effect on the downstream task performance, not just determined by pretraining loss alone.
- The study uncovers an important trade-off btw memorization and reasoning with TPP, where memorization skills are parameter-hungry and reasoning skills are data-hungry.
- Post-training or test-time scaling do not change the memorization-reasoning gap, so the optimal sparsity must be determined pretty much during the pretraining stage.

### Weaknesses
While the observations are novel and well-supported empirically, I would consider it more significant to find any **intuitive or theoretical rationales** behind them. For instance, why does reasoning capability require denser MoEs, while memorization thrives on sparsity? Why doesn’t simply scaling the parameter count work?

### Questions
In Line 321-323:  

> *At lower FLOPs, increasing sparsity still reduces loss and improves accuracy; however, once the FLOPs budget grows, denser models begin to perform better, achieving both lower loss and higher accuracy.*

However, I could not find where this trend is clearly demonstrated. Figure 5 appears to fix the FLOPs budget, so it doesn’t reveal how model performance varies as FLOPs increases.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper empirically studies the effect of sparsity in sparse mixture of experts (MoEs) on downstream tasks. Specifically the paper trains a series of models based on Mixtral architecture and studies downstream performance on tasks that tend to rely on memorization vs tasks that test reasoning capabilities of language models (LMs). The paper studies the relationship between training loss and task loss and provides insights on the effect of active parameter count and tokens per parameter (TPP) on downstream tasks described above. The main findings include active FLOPs may lead to higher reasoning accuracy and that higher tokens per parameter may be preferable for reasoning tasks.

### Strengths
- The paper studies the effect of sparsity in MoEs on downstream tasks. This area has not been examined in detail so the study in the paper is timely and will likely be of interest to many researchers & practitioners.

- The empirical setup including models, data and downstream tasks are  described clearly in the paper. This gives me confidence that the experiments are reproducible.

- The models considered in the paper are not necessarily compute-optimal. This detail may provide additional insight into how to optimally train and use MoEs at scale.

- The paper proposes tokens per parameter (TPP) as a metric to track in addition to active parameters. This metric provides additional insight into the role of data in training MoEs that work well on downstream reasoning-type tasks. This may be interesting to many readers (and is definitely interesting to this reader)

### Weaknesses
- The paper considers a single architecture inspired by Mixtral family of MoEs in the work. It's understandable why this choice was made (experiment volume) but I do wonder if other architecture choices can change the conclusions made here. If possible, please discuss why Mixtral was chosen as opposed to other choices. 

- The fact that memorization depends on total parameter count is known from prior literature. Furthermore, active number of parameters (inference FLOPs) have also observed to improve certain downstream tasks' performance (Abnar et al., 2205). So the claim that these are new contributions is weak.

- While sparsity in mentioned in the paper, the plots (Figure 1) for instance do not show this value but instead show top-K. This makes it hard for the reader to infer the effect of sparsity on empirical observations.

### Questions
- Is there a way to show how downstream performs with sparsity where sparsity is defined in the paper as 1 - (active / total experts)? Sparsity is mentioned in the paper but is not shown explicitly in scaling plots. Please include sparsity value, if possible, with the plots. Only Figure 5 appears to include sparsity (via density term which is its complement).

- The range of accuracy/error rate for GSM8K task appears to be on the lower side? Are these values good enough for readers to draw valid conclusions? A discussion on what is reasonable would be very useful to help the reader.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to investigate how MoE sparsity influences two distinct capability regimes: memorization skills and reasoning skills. The work shows how Active FLOP is more important for reasoning, while memorization improves with number of total parameters. Another interesting finding provided in this work is that changing the k in top-k routing has a negligible effect if the number of active parameters is kept constant.

### Strengths
1. It is an important observation that for MoE models, downstream accuracy can deviate from the predictions of conventional scaling laws, and these deviations may vary across different tasks.

2. Exhaustive experimentation is done in reasoning and coding tasks to demonstrate the U shape of tasks performance with the increase of total parameters at a FLOP controlled setting

3. Exhaustive experiments are done to show that post training couldn't improve this.

### Weaknesses
1. The number of tokens used seems small to if we are targeting End task performance, specially for MOE models
2. It would be good to get some ablation for various router choices, though than can be a future work
3. In Page 9, figure 8, it would be good do the study at k>1 (ideally 8) and E >8
4. More details about the post training setup is helpful. How many tokens in the post training set?
5. No details have been provided whether Continuous training is done or learning rate is annealed before evaluating end task

### Questions
1. The number of tokens used seems small to if we are targeting End task performance, specially for MOE models
2. It would be good to get some ablation for various router choices, though than can be a future work
3. In Page 9, figure 8, it would be good do the study at k>1 (ideally 8) and E >8
4. More details about the post training setup is helpful. How many tokens in the post training set?
5. No details have been provided whether Continuous training is done or learning rate is annealed before evaluating end task

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the optimal sparsity of Mixture-of-Experts models under memorization and reasoning skills by training MoE models with varying total parameters, sparsity, and top-k routing under the fixed budget. Through extensive experiments, the paper concludes that 1. the downstream reasoning quality is decided by both the active FLOPs and pretraining loss and 2. there exist different optimal tokens-per-parameter ratios for memorization and reasoning tasks.

### Strengths
- The paper is well-written and easy to understand.
- The experiments are comprehensive while supporting the major claims of the paper.
- One of the main findings is surprising, as it shows that higher sparsity only improves performance under memorization instead of reasoning tasks under the iso-FLOP settings.

### Weaknesses
- The paper might need to address more about its intuition and originality from previous works such as [1] and [2], since similar observations regarding the optimal sparsity in MoE models have been made.
- Theoretical insights are encouraged to explain the experimental findings.
- The U-shape trend plot for reasoning tasks in Figure 2 is very interesting, and I suggest the authors to verify such finding under more reasoning tasks.

[1] Samira, Abnar, et al. "Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models." arXiv:2501.12370 (2025).

[2] Zhao, Jinze, et al. "Sparse Mixture-of-Experts for Compositional Generalization: Empirical Evidence and Theoretical Foundations of Optimal Sparsity." arXiv:2410.13964 (2025)

### Questions
Questions are addressed in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
