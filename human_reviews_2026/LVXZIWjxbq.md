# Mitigating Think-Answer Mismatch in LLM Reasoning Through Noise-Aware Advantage Reweighting

- Decision: Reject
- Scores: 2, 4, 4, 8

## Abstract
Group-Relative Policy Optimization (GRPO) is a key technique for training large reasoning models, yet it suffers from a critical vulnerability: the Think-Answer Mismatch, where noisy reward signals corrupt the learning process. This problem is most severe in unbalanced response groups, paradoxically degrading the signal precisely when it should be most informative. To address this challenge, we propose Stable Group-Relative Policy Optimization (S-GRPO), a principled enhancement that derives optimal, noise-aware advantage weights to stabilize training. Our comprehensive experiments on mathematical reasoning benchmarks demonstrate S-GRPO's effectiveness and robustness. On various models, S-GRPO significantly outperforms DR. GRPO, achieving performance gains of +2.5\% on Qwen-Math-7B-Base, +2.2\% on Llama-3.2-3B-Base, and +2.4\% on Qwen-Math-1.5B-Instruct. Most critically, while standard GRPO fails to learn under 20\% synthetic reward noise, S-GRPO maintains stable learning progress. These results highlight S-GRPO's potential for more robust and effective training of large-scale reasoning models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on mitigating the thinking-answer mismatch issue of RL for LLM reasoning. The so-called thinking-answer mismatch refers to the noise of outcome rewards that raise when the model generate incorrect CoTs but end with a correct answer coincidentally. Taking inspiration from the classic label noise research area, the authors proposed S-GRPO, a simple advantage shaping technique, which reweighs the advantages estimated by group-wise normalization and meanwhile explicitly take the possible noise in rewards into consideration. Experiments on Qwen2.5 Math and Llama 3.2 models on mathematical reasoning tasks are carried out to validate the effectiveness of the proposed method.

### Strengths
The proposed method is simple and straightforward. The method part is very easy for the reviewer to follow up with.

### Weaknesses
- The motivation is weak in my veiw. My opinion on this point are three folds. First, there are no strong evidence (e.g., objective statistics) to validate that the so-called thinking-answer mismatch would result in critical vulnerability when one try to scale the performance with outcome rewards. As far as I can tell, in mathematical and coding tasks, the rule-based reward signals are very clear and reliable. Besides, there have been a few observations demonstrate that RL with outcome rewards can automatically compress the incorrect CoTs with coincidentally correct answers [1]. Lastly, even if the rewards are unreliable, traditional value-based RL method like PPO can capture the flaws in reasoning traces. And thus the novelty and contribution of the proposed method is relatively limited in my view.
- There is no clear guidence on how to estimate the true reward noise ratio p. Estimation on the real noise ratio necessitate verification of the correctness of reasoning traces, which is non-trivial and in my view, rather challenging.
- The experiments setup is relatively outdated. The authors only tune 2 Qwen2.5 Math series model and 1 Llama3.2-3B model on 8.5K samples. It is doubtful that if the scale of RL training in this manuscript is enough for validate the effectiveness of the proposed method.
- Serveral sota baselines are missing. For example, DAPO and GSPO.
- The authors claim that the reported accuracy is averaged among the top-3 checkpoints. How exactly do they choose the checkpoints? I wonder if there exists data leakage in model selection.
- The visualization in this manuscript is hard for me to follow up with. Most figures are small, and the conveyed information is ambiguous and limited.

[1] Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs

### Questions
See weakness.

### Soundness
1

### Presentation
1

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
This paper proposes Stable Group-Relative Policy Optimization (S-GRPO), a principled method that uses optimal, noise-aware advantage weights to mitigate the "Think-Answer Mismatch" vulnerability in GRPO, thereby stabilizing training and significantly improving reasoning model performance.

### Strengths
1.  This paper try to address a significant problem: the issue of false positives where the "Thinking" process and the final "Answer" do not align.
2.  The approach is novel to me. Instead of the common practice of trying to identify errors within the thinking process, this paper proposes to reweight the advantage values to mitigate the impact of this mismatch.
3.  The paper is well-written, clear, and easy to understand.

### Weaknesses
(If my understanding is incorrect, please correct me)

1.  **Concerns about generalization.** The derivation, starting from Equations 5 and 6 and continuing to Equation 11, seems entirely predicated on the assumption that the random variable $r$ (reward) follows a Bernoulli distribution. However, this assumption may not always hold in practice. For instance, one might use $\{-1, 1\}$ reward pairs or even continuous rewards. In such scenarios, how would the proposed method be formalized? Would the new formulas still be applicable and effective?
2.  **Evaluation methodology.** The exclusion of the AIME 2024/2025 is unacceptable. With 30 problems, the avg@32 metric could have been calculated, which is a standard practice.

3.  **Difficulty in hyperparameter selection.** The choice of $p$ appears to be a significant challenge. With so many variables, such as different mathematical datasets, diverse reasoning tasks, various models, and different training settings, it seems difficult to establish an empirical rule of thumb for this parameter.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

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
The paper introduces S-GRPO, a noise-aware variant of Group-Relative Policy Optimization for training reasoning LLMs. It addresses the Think-Answer Mismatch problem, where noisy reward signals corrupt learning when reasoning and final answers misalign. S-GRPO derives optimal reweighting under a symmetric noise model, down-weighting unreliable groups to stabilize training. Experiments on mathematical reasoning benchmarks show consistent 2–3% accuracy gains and robustness under 20% reward noise. Analysis demonstrates smoother entropy reduction and more coherent reasoning compared to GRPO and Dr. GRPO.

### Strengths
1. The paper clearly defines the Think-Answer Mismatch problem in GRPO and provides a convincing analysis of its impact on learning stability.
2. The proposed S-GRPO introduces a principled noise-aware reweighting method derived from a theoretical foundation.
3. Experimental results show consistent, reproducible improvements across several mathematical reasoning benchmarks.

### Weaknesses
1. The method depends on a manually set noise parameter 𝑝 p, which may require case-specific tuning.

2. The assumption of symmetric reward noise simplifies the training environment but may not accurately represent real-world mismatch patterns where errors are often asymmetric.

3. The robustness experiments are based on artificially injected synthetic noise levels (up to 20%), which likely exceed the noise typically observed in real reasoning datasets, raising questions about whether such high noise modeling is necessary or reflective of practical scenarios.

1. The figures are not very clear, and the font size is quite small. It would be helpful to use larger text and improve the overall visual clarity of the plots.

### Questions
1. Does S-GRPO generalize beyond math reasoning tasks? Non-math reasoning tasks?
2. The paper evaluates robustness using 10–20% synthetic, symmetric reward noise but does not clarify how this compares to the typically lower and more structured noise found in real datasets. This difference may limit the realism of the evaluation, so it would be helpful for the authors to estimate actual mismatch rates and examine performance under more realistic noise conditions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper focuses on addressing the "Think-Answer Mismatch" problem, a prevalent issue in Large Language Model (LLM) reasoning tasks, where the correctness of the model's reasoning process does not always align with the correctness of its final answer. The authors point out that this problem is particularly prominent in the popular Group Reward Policy Optimization (GRPO) method. This is because incorrect reasoning processes can still yield correct answers (i.e., false positives), generating noisy reward signals that severely contaminate the model's learning process. The paper proposes an optimal advantage re-weighting method, named S-GRPO, based on a symmetric noise model. This method is designed to robustly mitigate the contamination of training signals in GRPO caused by the think-answer mismatch, thereby enhancing the reasoning performance and stability of large models.

### Strengths
1. Clear Problem Definition and In-depth Analysis: The paper provides a remarkably thorough analysis of the impact of the "think-answer mismatch" within the GRPO framework. Instead of merely discussing the general harm of noise, it precisely identifies, through mathematical derivation, that "group imbalance" is the key factor amplifying the effect of noise. This insight is both profound and enlightening.
2. Elegant and Theoretically Grounded Methodology: S-GRPO is not presented as an empirical "patch" or a complex module. Rather, it is an optimal weighting strategy derived from the classic symmetric noise model by minimizing the expected squared error between the observed and true advantages.
3. Strong Reproducibility: The paper provides detailed experimental settings and includes a link to an anonymized code repository, which significantly enhances the reproducibility of the research and is a commendable practice.

### Weaknesses
1. Limitations of the Symmetric Noise Model: A core assumption of the proposed method is the presence of symmetric noise. However, in practical scenarios, the probabilities of false positives and false negatives may not be symmetric.
2. Dependence on the Noise Rate p and Insufficient Guidance for its Selection: The paper observes that the optimal value of p is related to the model's scale, which is a valuable finding. However, it does not offer clear guidance on how to efficiently estimate an appropriate value for p a priori when dealing with a new model or task. This could introduce an additional hyperparameter tuning burden and uncertainty in the practical application of the method.

### Questions
1. The paper states that the optimal value of p is related to model scale. Besides model scale, could the selection of p also be dependent on other factors, such as the difficulty of the dataset or the type of task?
2. Does the case study in Appendix B.1 reveal a potential, undiscussed drawback of S-GRPO? Specifically, by penalizing reasoning paths that may be "shortcuts" or based on "pattern matching" but still produce the correct result,
 could S-GRPO inadvertently compel the model to attempt complex, formal reasoning paths that it has not yet fully mastered? This could, in turn, increase the risk of computational or execution errors, ultimately compromising the accuracy of the final answer.

### Soundness
3

### Presentation
3

### Contribution
3
