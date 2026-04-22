# Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search

- Avg Score: 4.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4

## Abstract
Large Language Model (LLM) based multi-agent systems (MAS) show strong potential for tackling complex tasks through collaborative intelligence. Monte Carlo Tree Search (MCTS) based methods provide promising approaches for enhancing MAS self-training by generating synthetic data, using Q-values to estimate agent contributions. However, relying solely on Q-values may misalign with the goal of selecting data most beneficial for MAS improvement. To address this discrepancy, we propose Data Influence-oriented Tree Search (DITS), a novel framework that incorporates influence scores to guide both tree search and data selection in data synthesis. By leveraging influence scores, we effectively identify the most impactful data for MAS improvement, thereby enhancing model performance. Furthermore, we derive a novel influence score estimation method tailored for non-differentiable metrics, significantly reducing computational overhead by calculating performance changes on the validation set. Extensive experiments on three different multi-agent tasks demonstrate the robustness and effectiveness of the proposed methods. Notably, our findings reveal that allocating more inference resources to estimate influence scores, rather than Q-values, during data synthesis can more effectively and efficiently enhance model training. We will release our code and data in the future.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Data Influence-oriented Tree Search (DITS), a novel framework for generating more effective synthetic data to train multi-agent systems (MAS). The authors argue that current methods, which rely on Q-values to select data, are flawed because high Q-value data does not always lead to the best model performance. DITS solves this by using a novel influence score to guide the tree search, prioritizing data based on its expected contribution to the model's actual performance improvement rather than its Q-value. The paper also presents a computationally efficient method for estimating this influence score based on non-differentiable metrics. Experiments across seven datasets show that DITS achieves state-of-the-art performance and scales more efficiently than traditional approaches.

### Strengths
1. This work proposes a new, influence-aware framework that directly optimizes for data *utility* in training. This "influence score" is a more relevant metric for data synthesis than traditional Q-values.
2. A major strength is the new, computationally cheap method for estimating the influence score. By avoiding complex gradient computations and focusing on non-differentiable metrics, it scales more efficiently and reduces computational overhead.
3. The method achieves state-of-the-art results, demonstrating clear, quantitative improvements (e.g., 2.5%-2.7% performance gains) across seven datasets and three different multi-agent tasks.

### Weaknesses
1. The proposed method introduces an influence score metric to quantify the impact of data on the current agent’s performance. In previous methods, this score was developed to measure the difference in loss when a data point was assigned a higher weight in the training dataset. In contrast, this work adjusts the influence score based on changes in non-differentiable performance metrics. However, the work lacks a clear intuition to explain why this change enables the influence score to indicate performance improvement. Additionally, it is unclear why the authors believe that higher Q-values do not necessarily mean performance improvement.

2. The authors claim that training effective MAS requires high-quality data that reflects complex agent interactions, noting that it is costly and time-consuming to collect this data in the real world. They utilize MCTS to simulate interactions and automatically produce preference-labeled training data. In this process, the authors select a node for the candidate set using a softmax distribution of Q-values, thinking that the softmax distribution balances exploration and exploitation. However, in practice, the softmax distribution typically emphasizes exploitation in reinforcement learning. It does not inherently balance exploration and exploitation, which raises concerns about the proposed method's exploration capabilities.

3. The authors assume that the transition function is known during the data generation process. However, in complex multi-agent systems, the transition function is often unknown. This raises two questions: Can the proposed method work properly in model-free environments? Or, can we learn an accurate transition function from existing datasets?

### Questions
Please see the Weaknesses above.

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
3

### Summary
This paper proposes DITS, a data influence–oriented tree search framework for multi-agent synthetic data generation and self-training. Rather than relying on MCTS Q-values alone, it estimates each sample’s influence on downstream validation metrics (including non-differentiable ones) via a one-step perturbation/finite-difference approximation and uses this to guide both tree expansion and preference selection with H(z)=IF_val(z)+ $\gamma$Q. Experiments on static and dynamic multi-agent tasks show consistent improvements over Optima-style baselines under comparable or lower compute budgets.

### Strengths
- Originality and clarity: The paper reframes multi-agent data selection around validation-aligned influence, integrating a finite-difference influence estimator (applicable to non-differentiable metrics) directly into tree search and preference construction; the objective and algorithm are clearly specified with equations and a concise pipeline.
- Quality and significance: Across static and dynamic multi-agent settings, the method consistently improves over strong Optima-style baselines with modest computational overhead; ablations and budget analyses suggest the approach is practical and broadly applicable to MAS data synthesis and self-training.

### Weaknesses
- Heuristic design choices: Many key components are tuned heuristically without principled justification or sensitivity studies—e.g., the edit-distance threshold for candidate pruning, the per-round selection ratio α, Softmax sampling solely on Q (instead of UCB/PUCT), and a Q-based prefilter before the final ranking. Several rewards/regularizers are also combined with ad-hoc weights, making behavior sensitive to undocumented scales.
- Linear mixing without calibration: The scoring rule $H(z)=IF_{val}(z)+\gamma Q$ lacks scale alignment and any adaptive weighting. Since $IF_{val}$ and $Q$ can live on different (and drifting) ranges across rounds/tasks, a fixed $\gamma$ risks unstable selection; no normalization (e.g., z-score/tempering) or learned mixture is explored.
- Crude influence approximation: The estimator relies on a one-step perturbation and finite differences on non-differentiable metrics, which can be highly noisy and step-size dependent in non-convex DPO training. Using LoRA gradients to proxy full-parameter updates further increases mismatch between estimated and realized influence.
- Tiny and reused validation set: A very small validation split (V=20) is reused to compute $IF_{val}$ across iterations, inviting high variance, overfitting to the validation set, and selection leakage; there is no cross-folding, multiple seeds, or reporting of uncertainty to mitigate these risks.

### Questions
- Calibration and selection policy: Have you explored normalization or adaptive weighting for the scoring rule $H(z)=IF_{val}(z)+\gamma Q$ (e.g., per-round z-scoring, temperature scaling, or a learned mixture), and ablations that remove the Q-based prefilter or replace Softmax-on-Q with UCB/PUCT? How do these choices affect stability and diversity of selected data?

- Robustness of influence estimation: How sensitive are results to the step-size/perturbation choices, the one-step vs multi-step approximation, LoRA vs full-parameter gradients, and the small reused validation set (V=20)? Can you report seed-averaged curves and show that estimated influence correlates with realized improvement on a held-out set or under cross-fold validation?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces DITS, a training framework that leverages influence scores to guide data selection, aiming to address the mismatch between Q-value-based estimation and the model’s training objective in prior methods. The approach is evaluated across multiple multi-agent tasks, demonstrating its effectiveness.

### Strengths
The paper is clearly written. The empirical section presents comparative experiments in both static and dynamic scenarios, and includes extensive hyperparameter-control experiments that demonstrate the method’s robustness.

### Weaknesses
1. The data selection strategy yields modest gains in downstream performance but introduces additional LoRA training overhead. It also relies on a validation set whose distribution closely aligns with the evaluation set, and its impact on generalization in multi-agent systems is not evaluated.
2. Some citations need format correction. When citing authors after mentioning a method name, the citation should be enclosed in parentheses (use `\citep` rather than `\citet`).

### Questions
1. In the dynamic scenario experiments (Table 2), why are results under the Optima-iSFT-DPO and DITS-iSFT-DPO settings not reported?
2. My understanding is that DITS builds on Optima by further selecting training data using a validation set from the same distribution. My main concern is that DITS yields modest gains and lacks transfer evaluation (training on one dataset and testing on another within the same domain), so the effect of the data selection strategy on generalization remains unvalidated.
3. In Optima, the Q-value used for trajectory selection already includes $R_{task}$, which directly reflects downstream task performance. In DITS, changes in validation-set performance are again used for further data selection. Is the misalignment between the Q-value and the influence score (as mentioned in Table 1) primarily due to distribution differences between the training and validation sets, or to the additional terms introduced into the Q-value ($R_{token}$, $R_{loss}$ in Equation 15), which appear to trade off task performance, token cost, and linguistic fluency?

### Soundness
3

### Presentation
3

### Contribution
2
