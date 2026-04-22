# Data Selection for LLM Alignment Using Fine-Grained Preferences

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Large language models (LLMs) alignment aims to ensure that the behavior of LLMs meets human preferences. While collecting data from multiple fine-grained, aspect-specific preferences becomes more and more feasible, existing alignment methods typically work on a single preference and thus struggle with conflicts inherent in such aggregated datasets. As one early attempt, in this paper, we propose a data-centric approach to align LLMs through the effective use of fine-grained preferences. Specifically, we formulate the problem as a direct fine-grained preference optimization and introduce preference divergence (PD) that quantifies inter-aspect preference conflicts. Instead of directly tackling the consequent complicated optimization, we recast it as a data selection problem and propose a simple yet effective strategy, which identifies a subset of data corresponding to the most negative PD values, for efficient training. We theoretically analyze the loss-bound optimality of our selection strategy and conduct extensive empirical studies on varied settings and datasets to demonstrate that our practical selection method could achieve consistent improvement against standard full-data alignment, using even just 30% of the data. Our work shares a line that LLM alignment using fine-grained preferences is highly feasible.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses a critical challenge in aligning Large Language Models (LLMs): effectively utilizing datasets composed of multiple, fine-grained human preferences (e.g., helpfulness, honesty) which often contain inherent conflicts and noise. The authors identify that standard alignment methods like DPO struggle with such aggregated data. Their core contribution is a data-centric approach that reframes the problem as data selection. They first formulate a Direct Fine-Grained Preference Optimization (DFPO) objective, which introduces a Preference Divergence (PD) term that quantifies the conflict between a specific fine-grained preference and the consensus of all others. Instead of directly optimizing this complex objective, they theoretically and empirically demonstrate that selecting a subset of data with the most negative PD values (indicating high consensus across aspects) for standard DPO training leads to superior performance. The proposed method involves training small, de-biased proxy reward models for each sub-preference to estimate the PD terms. Extensive experiments on datasets derived from UltraFeedback and HelpSteer show that their method consistently outperforms full-data alignment and other baselines, achieving better results using only 30-50% of the data, while also being more robust to increasing levels of preference conflict.

### Strengths
1. he paper clearly identifies a significant and under-explored problem—handling noise and conflicts in aggregated fine-grained preference data. The motivation is strong, backed by a statistical analysis of a real-world dataset (UltraFeedback) showing a high rate of preference conflicts.

2. The progression from the DFPO formulation to the PD term, and then to the data selection strategy, is elegant and well-justified. The theoretical analysis (Theorems 3.4 & 3.5) provides a solid foundation for the simple yet effective heuristic of selecting samples with the most negative PD.

3. The experimental section is thorough and convincing. The authors demonstrate:

### Weaknesses
1. The entire selection process hinges on the quality of the proxy reward models. While the paper shows low sensitivity to the model's size and training data amount, it does not deeply explore scenarios where these proxy models might fail catastrophically (e.g., on out-of-distribution data or aspects that are inherently difficult to model). The method inherits all the challenges of reward modeling.

2.  The method implicitly assumes all sub-preferences are equally important by summing their pseudo-reward gaps uniformly in the PD term. In reality, some aspects (e.g., "truthfulness") might be more critical than others (e.g., "verbosity"). The work does not explore weighted PD terms or user-specified preference hierarchies.

3. The experiments are conducted on models up to 8B parameters. It is unclear how the method would scale to very large models (e.g., 70B+), where the computational savings would be even more dramatic but the dynamics of fine-tuning can differ. Furthermore, the evaluation, while comprehensive, is primarily based on general-purpose chat benchmarks; testing on more specific safety-critical or instruction-following benchmarks would strengthen the claims about robust alignment.

4. The derivation of the DFPO objective in the main text (and Appendix A.1.1) is quite dense and could be challenging for a reader to follow. A more intuitive, step-by-step explanation of how the PD term emerges from the multi-aspect RLHF objective would improve accessibility.

### Questions
1. How would the method perform if the fine-grained aspects were not orthogonal or even negatively correlated? For instance, if "conciseness" and "completeness" were both aspects, the definition of "conflict" might need refinement.

2. The length bias mitigation is a crucial component. Was any analysis done to determine the optimal value for the length penalty coefficient ρ? Is it robust across different datasets, or does it require tuning?

3. The paper focuses on selecting data for a single round of DPO training. Could this PD-based selection be integrated into an iterative training loop, where the aligned model is used to re-estimate rewards and re-select data for further training?

4. The "PD (rati.)" baseline, which uses ground-truth ratings, performs comparably well in some cases. Does this suggest that with perfectly reliable, continuous ratings (instead of binary preferences), the need for proxy models could be eliminated? What are the relative trade-offs between the annotation cost of ratings vs. the computational cost of training proxy models?

### Soundness
3

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
3

### Summary
The paper reframes preference learning under mixed and potentially conflicting preferences as the selection for a subset of training data whose implicit preference is closest to the ground-truth preference. A data-selection algorithm (DFPO) is accordingly proposed and evaluated extensively.

### Strengths
1. Detailed problem formulation and the novel transformation of the problem into data selection methods instead of algorithm development is interesting.

2. Empirical coverage is thorough.

### Weaknesses
1. With ever-larger models and compute, scaling-laws may simply “wash out” moderate preference noise; the urgency of the problem is not demonstrated.

2. The method operates on a fixed dataset and is demonstrated only with the now “classical” DPO pipeline.  Readers working with on-policy RL extensions are unlikely to see an immediate hook.  Extending DFPO to iterative regimes like iterative DPO would greatly widen its appeal.

3. If a dataset contains several conflicting preferences, DFPO appears to return the largest conflict-free subset.  When preferences are equally strong and mutually incompatible, is the algorithm simply selecting the majority preference?  A crisp explanation that shows what is really done by DFPO is missing.

4. Many experiments are presented, but each is analysed briefly. Spare more pages to careful analysis will be helpful.

### Questions
1. The formalism treats (human) preference as an abstract concept, while the same conflict could arise when mixing code, math, etc.  How should the “ground-truth preference” be defined in such multi-task settings where no single human ranking exists?

### Soundness
4

### Presentation
2

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
This paper addresses LLM alignment with human preferences, emphasizing the difficulties arising when using fine-grained, aspect-specific preferences rather than singular “overall” preference labels. The core proposal is a data-centric, preference divergence (PD)-based selection strategy that identifies and utilizes the most reliable samples from aggregated, potentially conflicting, fine-grained preference datasets. 
The authors provide theoretical grounding for their selection method (establishing loss-bound optimality), introduce approaches to estimate PD in practice (including bias mitigation), and conduct comprehensive experiments demonstrating that selective alignment using just a fraction (e.g., 30\%) of filtered data can outperform traditional full-dataset alignment in both performance and computational efficiency.

### Strengths
1. The authors clearly demonstrate motivation for the problem, where aggregating fine-grained preferences introduces conflicts, redundancy, and noise that degrade LLM alignment.
2. The development of loss bounds and the selection optimality result underpin the proposed data selection strategy with rigorous analysis, providing compelling mathematical justification for selecting samples by most-negative PD.
3. Extensive evaluation: The method is thoroughly evaluated against full-data and alternative selection baselines, across datasets (UltraFeedback, HelpSteer), models (Llama, Qwen), and varying conflict levels. PD-based selection (especially “ours”) consistently outperforms others—even with substantial data reduction—while being robust to proxy model choice and ablation.

### Weaknesses
1. I am not familiar with this research scope, but the current evaluation focuses on UltraFeedback and HelpSteer, and their derived conflict settings are limited. The author should conduct experiments with more advanced benchmarks for a clear demonstration of their effectiveness. 
2.  The empirical studies do not report in-depth on the sensitivity of the method to hyperparameters (e.g., $\lambda$, quantile level $\gamma$, length penalty $\rho$, sampling ratio $p_r$), aside from the generic selection budget. Although performance seems stable as shown in Figures 3 and 4, the rationale for particular choices and the potential for overfitting (especially when “tuning” for the best selection cut-off) are not thoroughly analyzed.

### Questions
See weaknesses.

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
The authors proposed a data-centric approach to align LLMs through the effective use of fine-grained preferences. They studied a data selection problem and propose an effective strategy to identify a subset of data corresponding to the most negative PD values for the training.

### Strengths
A balanced sampling strategy is applied to mitigate the intrinsic bias towards longer responses that are favored regardless of quality.

A penalty term is introduced into the reward model to discourage length bias as well.

The paper is well written and easy to follow.

### Weaknesses
See the below questions.

### Questions
To address the practical challenge in DFPO, i.e., the high computational cost and the risk of instability from unavailable/unreliable reward models, the authors filter the dataset to construct a high-quality subset for the training. Besides the data selection strategy, are there other strategies to address the challenge? Is it possible to apply LLMs’ distillation techniques?

The authors extend the principle of DPO to the fine-grained preference alignment setting by introducing an additional term (equation 4), i.e., the preference divergence one. Are there other way to define the objective for the purpose? Any discussions?

The authors consider K fine-grained criteria. What if these K fine-grained criteria are dependent to each other? 

The authors construct the two fine-grained preference datasets from two existing datasets. And the details are provided in the appendix. Not sure if there are other ways to construct the multi objective dataset for the purpose of the experiments?

### Soundness
3

### Presentation
3

### Contribution
3
