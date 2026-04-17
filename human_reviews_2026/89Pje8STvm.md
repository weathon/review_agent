# Self-Aligned Reward: Towards Effective and Efficient Reasoners

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 8

## Abstract
Reinforcement learning with verifiable rewards has significantly advanced reasoning with large language models (LLMs) in domains such as mathematics and logic. However, verifiable signals provide only coarse-grained or binary correctness feedback. This limitation results in inefficiencies like overly verbose or repetitive reasoning. Existing length-based solutions (e.g., length penalty) compromise accuracy. To address this deficiency, we introduce **self-aligned reward (SAR)**, a generic, universally applicable self-guided signal that complements verifiable rewards to enhance both reasoning accuracy and efficiency in RL. Specifically, SAR is defined as the relative perplexity difference between an answer conditioned on the query and the standalone answer, thereby favoring responses that are concise and query-specific. Quantitative analysis reveals that SAR reliably judges answer quality: concise, correct answers score higher than redundant ones, and partially correct answers score higher than entirely incorrect ones. Evaluation on 4 different models across 7 benchmarks shows that integrating SAR with prevalent RL algorithms like PPO and GRPO reduces answer length by 30%, while improving accuracy by 4%. Our analysis also shows that SAR generalizes well to out-of-domain tasks and achieves a Pareto-optimal frontier between correctness and efficiency compared to state-of-the-art baselines. We also show that SAR shortens unnecessary elaboration while preserving advanced reasoning behaviors. These results highlight the promise of self-aligned reward as a fine-grained complement to verifiable rewards, paving the way for efficient and effective LLM training. All of our code implementations and data are publicly available at GitHub.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Self Aligned Reward, SAR, an internal signal that measures the relative perplexity drop between an answer alone and the same answer conditioned on the query. SAR is added to a standard verifiable reward within PPO or GRPO. The goal is to reward answers that are concise and tailored to the query, reducing redundant reasoning while preserving or improving accuracy. Experiments across four small models and seven math and logic benchmarks report about four percent accuracy gains with roughly thirty percent shorter outputs, and a favorable accuracy versus efficiency frontier compared with length penalties such as O1 and ER.

### Strengths
- SAR is a small change to existing RL loops and uses only log probabilities already computed, so the added cost is minimal. The paper even reports slightly faster rollouts due to shorter generations.
-  The token level decomposition v(a_j) is a nice lens that explains why early, query specific tokens get rewarded, connecting SAR to information use rather than length alone. 
-  Across Qwen3 1.7B and 4B, Phi 3.5 mini, and Gemma3 1B, SAR variants reduce tokens and match or beat accuracy. The Pareto plots make the trade off story visually clear.

### Weaknesses
- RSA rises when an answer looks unlikely without the query. This can be increased by echoing or paraphrasing query tokens or by using rare surface forms rather than better reasoning. The qualitative analysis shows discouraging memorized short answers, but an explicit adversarial check would help. Consider adding tests where the model repeats entity strings or injects rare tokens to inflate ppl(a) relative to ppl(a|q).
- GRPO and PPO commonly use a KL term to a reference policy. SAR implicitly prefers answers that are improbable unconditioned on q, which could tug against or amplify KL. The paper should analyze this interaction and show the same KL settings are used across baselines.
- SAR uses the current policy to score both ppl(a) and ppl(a|q). If calibration shifts during training, the reward scale can drift, and the clipping in Equation 6 may hide pathologies. A schedule or normalization across batches might be needed. Please report the distribution of RSA over training and a sensitivity study to the clip range.
- Length penalized GRPO variants often trade accuracy for brevity, yet there are recent methods that make brevity adaptive to difficulty. The paper compares O1 and ER, but please confirm both got per model alpha sweeps as wide as those used for SAR and include best in class brevity methods with per instance stopping rules where applicable. The Pareto plots for SAR vary alpha closely. Ensure comparable sweeps for others

### Questions
How sensitive are results to prompt templates that require explicit step by step derivations versus direct short answers? Does SAR over penalize long but necessary derivations when the verifier still rewards only final correctness? Please report token reduction broken down by reasoning behaviors. 

Can SAR be computed with teacher forcing on partial prefixes during rollout to provide denser token level rewards, rather than only sequence level returns. This might make the signal less noisy without a learned critic.

What happens if the query contains spurious details? Since SAR rewards dependence on q, models might over attend to red herrings. Add a study with injected distractors.

How do you prevent echoing of the query to raise RSA? Show an explicit penalty or a post hoc filter and quantify its effect.

Do results hold when the base model is much stronger? Most results are for one to four billion parameters. Please include at least a mid size base to test scaling.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces "self-aligned reward", the difference in perplexity on the generated solution with and without the query in-context, as an additional reward term to binary verifiable rewards in order to incentivise concise, query-specific reasoning. The authors motivate SAR through a case study that connects the GRPO advantages assigned to responses with various response types and that analyses the token-level contributions to the overall SAR term. The paper demonstrates that using SAR for training with PPO or GRPO leads to a Pareto improvement in token efficiency and accuracy. An ablation study on the reward terms, including a comparison to entropy regularisation, demonstrates the importance of the complete proposed reward combination. Additional analysis on qualitative reasoning behaviours and training costs are used to further illustrate the impact of SAR.

### Strengths
The paper effectively uses a case study to motivate the design decisions behind SAR, which provides an intuition for the subsequent results. The method itself convincingly improves on the length minimisation baselines in striking a balance between accuracy and token efficiency. The analysis section features appropriate ablations and the reasoning behaviour frequencies provide further evidence that reasoning traces from models without SAR or length penalties are bloated with subsequences that do not correspond core reasoning behaviours. The paper is well-written and results are presented clearly.

### Weaknesses
Using GPT-4o to annotate responses without checking the quality of these annotations, such as by comparing to gold-standard human annotations on a subset, means that the corresponding analysis sections cannot be taken to be fully reliable. 

It would be useful to show AES for a few different values of gamma. This would compliment the following section on accuracy-efficiency tradeoff nicely.

Whilst it's intuitive that entropy regularisation in the reward can lead to overconfidence and entropy collapse, explicitly verifying that using delta perplexity (SAR) mitigates this would support this claim, which is currently based only on accuracy differences.

### Questions
Do you have an intuition for why SA-GRPO leads to considerably more backtracking behaviour on NuminaMATH and AIME?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
To address the overthinking or repetitive reasoning in large language models, this work proposes a self-aligned reward combined with a verifiable reward to enhance the accuracy and efficiency of the mathematical reasoning capabilities of LLMs. The self-aligned reward $R_{SA}$ is based on the perplexity drop from the answer's stand-alone perplexity to the perplexity when the answer is conditioned on the query $ppl(a) - ppl(a|q)$. In other words, if the answer is highly dependent on the query (not generic tokens), then the difference will be high, and the model can receive a larger self-aligned reward.

The proposed reward is applied with PPO and GRPO for 4 different LLMs across 5 mathematical reasoning datasets and 2 logical reasoning datasets. Detailed ablation studies are also conducted, indicating the importance of applying both the verifiable and self-aligned reward at the same time, and the effectiveness of the perplexity-based reward can reduce the output sequence length.

### Strengths
* The proposed reward function is well motivated. Measuring the difference between the stand-alone perplexity of an answer and its perplexity conditioned on the query provides a reasonable approach to capturing trivial/generic reasoning steps. Combining the verifiable (external) reward with the intrinsic reward signal effectively optimises the model for both accuracy and efficiency.
* The proposed reward signal is integrated with PPO and GRPO to optimise four different LLMs (two Qwen variants, Phi-3.5, and Gemma3), and is evaluated across seven diverse datasets covering mathematical and logical reasoning tasks.

### Weaknesses
* The idea of combining a verifiable reward with a length penalty is well aligned with recent developments in reinforcement learning for reasoning optimisation. However, this topic has been widely studied in prior work as well. For example, T1 [1] integrates correctness with a rule-based negative reward for undesirable reasoning behaviours, ThinkPrune [2] introduces an iterative pruning strategy to encourage concise reasoning, and Self-Braking Tuning [3] employs a frequency-based masking approach to mitigate overthinking during supervised training. A more detailed comparison with these approaches, particularly T1, would substantially strengthen the paper’s contribution and clarify how the proposed method advances beyond existing studies.
* A statistical significance test is missing from the experimental results. Some reported improvements appear marginal, for instance, in Table 3, the average accuracy of GRPO and SA-GRPO is 52.49 and 54.13, respectively. Without a significance test, it is difficult to determine whether these improvements are statistically meaningful. Including such tests would enhance the reliability of the evaluation.
* The Entropy method [4], which is listed as one of the main related works in Table 1, is only included in the ablation study (Section 6.1 and Table 5) but not in the main comparative results (Sections 5 and 6.2). Given its conceptual relevance, excluding Entropy from the main comparison raises concerns about fairness, as the proposed method is otherwise compared primarily against weaker length-penalty baselines. A more comprehensive comparison would provide a clearer picture of the model’s relative performance.
* The authors claim that the intrinsic reward signal is less prone to reward hacking compared to external signals (L067–068). However, this claim is not sufficiently supported by experimental evidence. In fact, the results suggest the opposite: the accuracy of the model optimised with the intrinsic reward ($R_{SA}$) drops from 69.07 to 20.96 on average (Table 5), indicating that the model may be exploiting or “hacking this reward signal. An additional experiment or analysis demonstrating the robustness of the intrinsic reward would be necessary to substantiate the authors’ argument.

[1] T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling (Hou et al., ICML 2025)

[2] ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning (Hou et al., arXiv 2025)

[3] Self-Braking Tuning: Let LRMs Break Free from Overthinking via Self-Braking Tuning (Zhao et al., NeurIPS 2025)

[4] Entropy: The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning (Agarwal et al., arXiv 2025)

### Questions
* Section 6.3 (Training cost of self-aligned reward) could move to the appendix. On the other hand, some interesting contents in the Appendix, e.g. the analysis of the self-aligned reward function in Appendix D.2.1 or experiments of vLLMs in Appendix E.2, could move to the main text.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper aims to improve the conciseness of LLM outputs without sacrificing correctness. It proposes self-aligned reward (SAR), which considers the difference between the perplexity of answers when conditioned vs. not conditioned on the query. In contrast, existing approaches penalize the length of answers without concern for relevance to the query. SAR provides a Pareto improvement over baselines, improving both accuracy and conciseness.

### Strengths
The paper has a clear motivation, improving conciseness of LLM answers, and provides a simple intuitive solution, expressed as a reward function in (6).

Experiments are very comprehensive, considering a variety of models, datasets, and metrics. I especially appreciated the plots in Figure 3, showing performance gain vs. length reduction. It supports the main argument of the paper well: SAR results in the biggest performance gain across all length reduction levels.

Besides the gains in efficiency, the example in Section 4.2 illustrates that SAR could even be useful as an interpretability technique. It shows how SAR can be used to rank the relevancy of each output token to a given query.

### Weaknesses
In Section 4.1, evaluating different types of answers using different reward functions to see what SAR captures that alternative reward functions don’t is a great idea. The main difference seems to be that SAR is the only reward function that penalizes answers of type “Correct but no though”. However, how this type of answers are generated is not explained well in the text (“artificially synthesized to simulate memorization” is not a sufficient explanation). Without clarity on what this answer type stands for, the results in Section 4.1 are not helpful.

Columns in Table 1, “Continuous”, “Internal”, “Content-Aware”, “Correctness”, and “Conciseness”, are not defined. It is ambiguous what criteria is meant.

### Questions
How do (5) in Section 4.1 “simulate memorization”?

### Soundness
3

### Presentation
3

### Contribution
3
