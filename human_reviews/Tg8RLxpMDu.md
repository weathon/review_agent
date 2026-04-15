# Measuring memorization in RLHF for code completion

- Decision: Accept (Poster)
- Scores: 5, 6, 8, 6

## Abstract
Reinforcement learning with human feedback (RLHF) has become the dominant method to align large models to user preferences.
Unlike fine-tuning, for which there are many studies regarding training data memorization, it is not clear how memorization is affected by or introduced in the RLHF alignment process.
Understanding this relationship is important as real user data may be collected and used to align large models; if user data is memorized during RLHF and later regurgitated, this could raise privacy concerns. In addition to RLHF, other methods such as Direct Preference Optimization (DPO) and $\Psi$PO have gained popularity for learning directly from human preferences, removing the need for optimizing intermediary reward models with reinforcement learning.
In this work, we analyze how training data memorization can surface and propagate through each phase of RLHF and direct preference learning.
We focus our study on code completion models, as code completion is one of the most popular use cases for large language models. We find that RLHF significantly decreases the chance that data used for reward modeling and reinforcement learning is memorized in comparison to directly fine-tuning on this data, but that examples already memorized during the fine-tuning stage of RLHF, will, in the majority of cases, remain memorized after RLHF. In contrast, we find that aligning by learning directly from human preference data via a special case of $\Psi$PO, Identity Preference Optimization (IPO), increases the likelihood that training data is regurgitated compared to RLHF. Our work suggests that RLHF, as opposed to direct preference learning, is a safer way to mitigate the risk of  regurgitating sensitive preference data when aligning large language models. We find our conclusions are robust across multiple code completion datasets, tasks, and model scales.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper investigates the impact of reinforcement learning with human feedback (RLHF) and other preference alignment methods, such as Direct Preference Optimization (DPO) and ΨPO, on data memorization in large language models, specifically in code completion tasks. While previous research has explored data memorization during fine-tuning, how memorization behaves in the RLHF alignment process remains less understood. The study aims to shed light on the potential privacy risks of RLHF and direct preference learning methods, as sensitive user data could be inadvertently memorized and reproduced by the model. The authors find that RLHF reduces the likelihood of memorizing sensitive data from the reward modeling and RL stages compared to fine-tuning alone. However, if data is already memorized during fine-tuning, it is likely to persist post-RLHF. Conversely, direct preference learning methods like Identity Preference Optimization (IPO), a specific ΨPO method, increase the chances of data regurgitation. The study concludes that RLHF is safer for mitigating the risk of exposing sensitive data, and the findings are consistent across multiple datasets, tasks, and model scales.

### Strengths
1. The paper addresses a significant and underexplored aspect of data privacy in RLHF and direct preference alignment, especially concerning large language models widely deployed in code completion, a critical real-world application.
2. By comparing RLHF with direct preference learning methods like IPO, the paper provides actionable insights into safer alignment practices, offering practical relevance for model developers.
3. The study's conclusions are derived from experiments across diverse code completion datasets, tasks, and model scales, supporting the robustness and generalizability of the findings.

### Weaknesses
1. Wrong template. 
2. It's hard to understand the whole workflow because the paper lacks a workflow figure

### Questions
See above

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper analyzed in code completion models how training data memorization can surface and propagate through each phase of RLHF and direct preference learning. The findings suggests RLHF, as opposed to direct preference learning, is a safer way to mitigate the risk of regurgitating sensitive preference data when aligning large language models.

### Strengths
1. The paper studied a very practical and critical problem; the memorization of RLHF and its risk of data leakage.
2. The experiments are detailed.

### Weaknesses
I am new to RLHF with code completion model. I didn't find apparent weakness.

### Questions
Is there any reason of choosing code completion task for studying the RLHF memorization? Does the conclusion still hold for other RLHF scenarios? if not what else conditions should we consider?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents the first comprehensive study of memorization in Reinforcement Learning from Human Feedback (RLHF) pipelines, focusing on code completion as a practical use case. The authors analyze how training data memorization manifests across different stages of RLHF (fine-tuning, reward modeling, and RL fine-tuning) and compare it to direct preference learning approaches like IPO. 

Using a synthetic dataset and robust evaluation methodology, they find that RLHF significantly reduces memorization of reward model training data compared to direct fine-tuning, while direct preference learning exhibits higher memorization rates. The results are validated across different model scales and domains.

This paper makes important contributions to our understanding of memorization in RLHF pipelines, with clear practical implications. The methodology is sound and the findings are well-supported. While there are some limitations, they do not significantly detract from the paper's contribution.

### Strengths
1. Thorough experimental methodology with appropriate controls and metrics
2. Important practical implications for deploying large language models
3. Novel and significant findings about RLHF's memorization properties
4. Validation across multiple scales, domains, and datasets
5. Clear writing and comprehensive presentation of results
6. Strong technical foundation and careful experimental design

### Weaknesses
1. Main experiments focus on one synthetic dataset, though results are validated on other datasets
2. Could explore a wider range of model architectures and scales
3. Some hyperparameter choices could be better justified
4. Analysis could include more direct preference learning methods beyond IPO
5. Could provide more detailed analysis of failure cases and limitations

### Questions
1. How sensitive are the results to the choice of synthetic dataset construction? Would different types of sensitive information show different memorization patterns?
2. How do the findings generalize to other direct preference learning methods beyond IPO?
3. What specific recommendations would you make to practitioners implementing RLHF pipelines based on these findings?
4. How do the memorization patterns change with extremely large model scales (e.g., >100B parameters)?
5. How do you think your results might generalize to other common applications of RLHF, such as conversational AI, where memorization of personal or sensitive information could be equally problematic?
6. Are there any scenarios where learning directly through human preferences (like IPO or DPO) might be a better tradeoff despite the increased likelihood of memorization?

### Soundness
4

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
The paper discovers that RLHF significantly decreases the chance that data used for reward modeling and reinforcement learning is memorized in comparison to directly fine-tuning on this data, but that examples already memorized during the fine-tuning stage of RLHF, will, in the majority of cases, remain memorized after RLHF. The paper finds that aligning by learning directly from human preference data via Identity Preference Optimization (IPO) increases the likelihood that training data is regurgitated compared to RLHF. The work suggests that RLHF, as opposed to direct preference learning, is a safer way to mitigate the risk of regurgitating sensitive preference data when aligning large language models.

### Strengths
(1) The observations in this paper are interesting.

(2) The observations can potentially help important real-world applications.

(3) The paper is well written and easy to understand.

### Weaknesses
(1) It would be interesting to show the observations in this paper are generally applicable in various model backbones with various sizes. It seems that currently only Gemini Nano-1(1.8B) is used as the base mode in the experiments.

(2) Related to the statement that IPO exhibits stronger memorization of preference data than RLHF, currently the result is obtained using one model backbone. It would be great to validate this statement using various model backbones. Besides, it seems that this statement is only validated by some empirical results. It would be great if the paper can provide some in-depth (theoretical) analysis to illustrate why IPO exhibits stronger memorization of preference data than RLHF.

(3) Related to the datasets and reward model design used in this paper, the paper may provide more discussions to explain how they are chosen, and how they are similar to (or different from) the choices of the datasets and reward model design used in previous works related to RLHF for code completion (or similar tasks).

### Questions
(1) Could you please provide some in-depth (theoretical) analysis to illustrate why IPO exhibits stronger memorization of preference data than RLHF?

(2) Related to the datasets and reward model design used in this paper, the paper may provide more discussions to explain how they are chosen, and how they are similar to (or different from) the choices of the datasets and reward model design used in previous works related to RLHF for code completion (or similar tasks).

### Soundness
2

### Presentation
3

### Contribution
3
