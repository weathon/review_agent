# Towards Efficient Chain-of-Thought Reasoning via Adaptive-Budgeting based Policy Optimization

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 2, 2

## Abstract
Recently, Chain-of-Thought (CoT) reasoning has become a key problem-solving
capability for advanced large language models (LLMs) to address difficult tasks
such as the mathematical ones. However, balancing the efficiency and performance of long CoTs still remains an intractable challenge. In this paper, we observe that assigning adaptive token budgets for different examples during training
is an viable way to tackle with the above issue. Motivated by this, we propose a novel reinforcement learning scheme, termed Adaptive-Budgeting based
Policy Optimization (ABPO). Based on the popular GRPO, our ABPO redefines
the RL training as an adaptive curriculum learning process, where example pools
are curated to categorize training examples into three types, namely the mastered,
learning and hard ones, respectively. As the training progresses, ABPO will adaptively schedule the examples with proper length budgets, and the example pools
will alse be dynamically updated based on the model status. In this way, we can
assign adaptive token lengths for different examples during RL training, achieving
a good balance between efficiency and performance of CoTs. To validate ABPO,
we apply it to three representative LLMs, and conduct extensive experiments on
a bunch of CoT reasoning benchmarks. The experimental results not only show
the substantial efficiency improvements with minimal performance loss, e.g., reducing token length by 78.3% while improving 2.0% performance of DeepSeek-R1-Distill-Qwen-1.5B on average, but also show our obvious advantages over
the compared methods, e.g., reducing 59.4% length and increasing 8.3% performance on average than HAPO, respectively. Our code is anonymously released at https://anonymous.4open.science/r/AnonymizeABPO-5380/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes ABPO (Adaptive-Budgeting based Policy Optimization), a reinforcement learning framework to improve efficiency in chain-of-thought (CoT) reasoning for large language models (LLMs). The core idea is to dynamically allocate token budgets to training samples according to model performance and sample difficulty. ABPO extends Group Relative Policy Optimization (GRPO) by maintaining three example pools (mastered, learning, and hard) and adaptively updating token budgets during training. Experiments on four mathematical reasoning benchmarks (MATH500, AMC, AIME, OlympiadBench) and multiple base models (DeepSeek-R1-Distill-Qwen 1.5B/7B, DeepScaleR-1.5B) demonstrate that ABPO can reduce CoT length by over 70% while maintaining or slightly improving accuracy.

### Strengths
- **Comprehensive experimental design:**  
  The paper includes diverse models and datasets, as well as ablation studies on curriculum learning, adaptive budgeting, and review mechanisms.  
- **Clear motivation:**  
  It addresses a real inefficiency issue (“over-thinking”) observed in long-CoT reasoning LLMs.  
- **Insightful analysis and visualization:**  
  The visualizations clearly show that ABPO dynamically adjusts reasoning length based on task difficulty.  
- **Empirical consistency:**  
  Across all settings, ABPO achieves substantial efficiency gains (up to −78% CoT length) while maintaining comparable or slightly higher accuracy.

### Weaknesses
- **Limited novelty:**  
  The idea of adaptive length control resembles prior works (e.g., HAPO, L1-Max) and mainly builds upon existing RL frameworks (GRPO).  
- **Modest gains:**  
  Although CoT length reduction is impressive, the improvement in accuracy is small (≈ +1–2%).  
- **Methodological complexity:**  
  The multi-pool adaptive scheme introduces additional hyperparameters and scheduling logic, which may complicate implementation compared to simpler fixed- or curriculum-budget baselines.  
- **Motivation ambiguity:**  
  It remains unclear whether optimizing CoT length should be a primary training objective, or if this efficiency truly translates to better general reasoning ability.

### Questions
1. Does the difficulty categorization (mastered / learning / hard) require a **pre-evaluation step before training**, or is it performed online during training?   
2. How robust is ABPO across random seeds and datasets, given that adaptive updates depend on the model’s current accuracy estimates?

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
This paper studies efficient Chain-of-Thought (CoT) distillation, aiming to reduce the training cost and improve reasoning capability of smaller models distilled from large reasoning models. Instead of focusing on architecture or optimization, the authors take a data-centric perspective—investigating how to select high-quality reasoning data for CoT distillation.

They introduce a data selection framework that prioritizes training samples with diverse reasoning paths and useful intermediate steps, based on the concept of step utility (how much each reasoning step contributes to the final correct answer). Through systematic experiments across arithmetic reasoning, commonsense reasoning, and instruction datasets, they show that carefully selected reasoning samples significantly outperform random or length-based sampling for the same training budget.

### Strengths
* Novel data-selection view of CoT distillation: Previous works often focus on model-side compression or prompt engineering, while this paper reframes the problem from a data selection perspective, emphasizing “which reasoning traces to distill” rather than “how to distill.”

* Well-designed methodology: The introduction of step utility and diversity-based filtering provides a concrete and interpretable criterion for sample selection, improving both efficiency and generalization.

* Comprehensive experiments: Multiple datasets (GSM8K, SVAMP, CSQA, StrategyQA) and multiple student sizes are tested. The paper also includes ablation studies that validate the effect of each selection criterion.

### Weaknesses
* Step-utility estimation overhead: Computing step-level utility still requires running reasoning traces and reward evaluation, which may offset the claimed training efficiency in large-scale settings.

* Limited evaluation on complex reasoning: Most benchmarks are short-form arithmetic or commonsense reasoning; the approach’s scalability to long CoT tasks (e.g., MATH, GPQA) or multimodal reasoning remains untested.

* Ablation depth and interpretability: The ablation studies show performance differences but lack deeper analysis of why certain data characteristics (e.g., step diversity vs. accuracy) dominate the improvements.

### Questions
See the Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the efficiency–performance trade-off in chain-of-thought (CoT) reasoning for large language models (LLMs). The authors propose Adaptive-Budgeting based Policy Optimization (ABPO), an extension of GRPO that adaptively assigns token budgets to training samples according to their difficulty and the model’s learning progress. The approach divides training data into mastered, learning, and hard pools, dynamically updating these categories and their corresponding length budgets. Experiments on multiple LLMs (DeepSeek-R1-Distill-Qwen-1.5B/7B, DeepScaleR-1.5B-Preview) and reasoning benchmarks (MATH500, AMC, AIME, OlympiadBench) show significant reductions in CoT length with minimal or no loss in accuracy.

### Strengths
* Targets an important and underexplored problem: balancing reasoning efficiency with accuracy in LLMs.

* Proposes a simple and implementable mechanism that achieves strong empirical gains.

* Comprehensive experimental evaluation across multiple model scales and benchmarks.

* Qualitative examples effectively illustrate adaptive behavior.

### Weaknesses
* The adaptive budgeting rule is heuristic—thresholds and increments are hand-picked, and the categorization process (mastered/learning/hard) lacks theoretical grounding.

* No formal justification for why this curriculum structure leads to optimal or stable policy improvement.

* Limited ablation on key hyperparameters (τ₀, τ₁, d); unclear robustness.

* The approach does not fundamentally modify GRPO—it primarily wraps a training schedule around it. It seems to be ineffective if we do not revisit the same question again?

### Questions
1. How sensitive is ABPO to the choice of τ₀, τ₁, and d? Could these hyperparameters be learned or adapted automatically?

2. How does ABPO behave when the pool thresholds are poorly chosen—does training diverge or stagnate?

3. I recently came across this paper Optimizing Anytime Reasoning via Budget Relative Policy Optimization (https://openreview.net/pdf/8136e4668a09f8c47a2454d9e72728d4fdea055e.pdf), which shares very similar motivation and method. Instead of dynamically setting budget, this paper uses the dense reward framework to let the model learn the optimal budget by itself. It would be nice if you can discuss your differences and how your method is fundamentally better than the baseline.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes ABPO, a reinforcement-learning framework that adapts the token budget per example to balance CoT efficiency (shorter traces) and accuracy. It builds on GRPO by (i) adding a budget-aware reward, and (ii) organizing data into dynamically updated mastered/learning/hard pools with curriculum-style scheduling and periodic review. Experiments across multiple math benchmarks and three base models show improved length–accuracy trade-offs relative to fixed-budget.

### Strengths
1. The paper is generally well-written and structured, followed by the motivation and explanation.
2. The separation of the token allocation decision from the generation process is a clean and novel architectural contribution.
3. The experimental results validate the central hypothesis: the model learns to assign smaller budgets to easier problems and reserves larger budgets for more difficult ones.

### Weaknesses
1. The reward design is based on a hard cutoff binary scheme, which does not distinguish between reasoning traces that are slightly longer than the target and those that are significantly longer. I suggest incorporating a soft budget penalty, similar to the L1-Max penalty, to encourage smoother control over reasoning length.
2. The paper introduces a number of hyperparameters( $\tau_0$, $\tau_1$, $\lambda$, $\alpha$, $d$, and $t_0$) without providing evidence of tuning.
3. The authors evaluate with $K = 3, 5, 10$ rollouts. However, it is common practice to perform 16 or 32 rollouts for a fair evaluation and to compute pass@1 scores. Using only 10 rollouts may not be sufficient.
4. The proposed method only outperforms the DeepSeek-1.5B model in Table 1, but performs worse than the base DeepSeek-7B and DeepScaleR-1.5B models.
5. In Table 2, the authors compare only against DeepSeek-1.5B, while DeepScaleR exhibits stronger baseline performance. I recommend evaluating the method based on DeepScaleR, and for fairness, performing fixed-budget comparisons for both ABPO and DeepScaleR—for example, by applying early stopping to DeepScaleR under equivalent token budgets.
6. In Figure 3, presenting only a single point is insufficient for fair comparison. To better demonstrate performance–efficiency trade-offs, the authors should ensure similar token usage between ABPO and L1-Max, and include results for multiple token budgets (e.g., 512, 1024, 2048 tokens).

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
