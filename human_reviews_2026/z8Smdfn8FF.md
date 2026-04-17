# Enhancing Large Language Model Reasoning via Selective Critical Token Fine-Tuning

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Large language models (LLMs) primarily rely on supervised fine-tuning (SFT) as a key method to adapt pre-trained models to domain-specific tasks such as mathematical reasoning. However, standard SFT uniformly penalizes all tokens, neglecting that only a small subset of critical tokens determines reasoning correctness. This uniform supervision often causes reduced output diversity and limited generalization. We propose Critical Token Fine-tuning (CFT), a simple yet effective approach that updates only tokens identified as functionally indispensable via counterfactual perturbations. By focusing gradient signals on these decisive reasoning steps while preserving the diversity of non-critical tokens, CFT can enhance both generation and diversity. Extensive experiments on five models across three families (Qwen, OLMo, LLaMA) and eleven mathematical reasoning benchmarks show that CFT, despite fine-tuning on less than 12% of tokens, consistently outperforms standard SFT. Moreover, CFT enables test-time scaling through improved sampling diversity and provides a stronger initialization for reinforcement learning, sustaining performance gains in later training stages while maintaining higher entropy for better exploration. These results highlight CFT as a practical and general framework for efficient and robust LLM fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors put forward a new SFT method called Critical Token Fine-tuning (CFT). The authors propose only training on critical tokens: a token is deemed non-critical if substituting it with other tokens inside the top-k results in the language model maintaining the correctness of the generated answer using a verifier. The authors test the effectiveness of their SFT method using a suite of small LMs up to 8B parameters using a variety of mathematical reasoning and medical QA tasks. The authors also show its effectiveness for initializing RL training post SFT.

### Strengths
* The method is simple and straightforward to implement. Simple methods are often the most effective.
* The presentation of the paper is extremely clear and easy to follow. 
* The suite of experiments provided is extensive to demonstrate the benefits of CFT over SFT, albeit no error bars are provided (see weaknesses). The authors consider two different domains: mathematical reasoning and medical QA. The authors consider full finetuning and LoRA adaptation. The authors also consider SFT and RL finetuning. The authors also importantly perform an extensive hyperparameter sweep for all important parameters for SFT.

### Weaknesses
* There exists similar methods which measure the importance of specific tokens in a sequence and use this importance to select tokens to contribute to the loss for pre-training, RHO-1 [1]. RHO-1 measures token “importance” or ”criticality” of a token by how different it is to a reference model. So it is not entirely surprising that this can be extended to different notions of “importance”. In this paper’s case it is whether swapping the token for other tokens inside the top-k results in all subsequent predictions being wrong.
* The CFT method is very expensive for all datapoints in the SFT dataset one needs to perform predictions for k counterfactual tokens and at each position t to determine the critical tokens of each sequence. For that compute budget one might be able to run RLFT on top of SFT for the same compute budget as CFT?
* My main issue with the experiments section is the lack of uncertainties in the results. The authors have not run the experiments multiple times to establish error bars over for instance the results in Table 1. For instance the gains in Table 1 for Llama3.1 or OLMo2 could simply be attributed to noise. Likewise in the RL experiments the conclusions are all drawn from experiments over a single seed where the learning curves for GSM and MATH overlap considerably. I find that the evidence does not justify the claims. I understand that repeating an experiment multiple times is expensive in terms of GPU compute required though.

[1] Lin, Zhenghao, et al. "Rho-1: Not all tokens are what you need." arXiv preprint arXiv:2404.07965 (2024).

### Questions
* It is not clear to me how parallel critical identification reduces computation time in lines 157-161. Is it due to KV-caching?
* The claim that CFT initialized models have higher entropy and therefore encourage exploration is very interesting. I’m assuming the entropy is in the LLMs logits? Should this not result in worse performance, since the entropy of the logits is what we are trying to minimize when we are minimizing next token predictions? An increase in exploration should result in new states being visited by the LLM. So surely this should result in more diverse predictions: increased pass@k? It is not clear to me what evidence you have for stronger exploration in the paper.

### Soundness
3

### Presentation
4

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
The paper presents the idea of critical token fine-tuning, based on the hypothesis that only a small fraction of tokens actually determines reasoning correctness. Compared with standard supervised fine-tuning approach, the paper proposes critical token fine-tuning, which first finds critical tokens through counterfactual perturbation, followed by fine-tuning only on these critical positions.

### Strengths
- The paper proposes a novel method and observation that takes into account token importance for reasoning to perform more targeted supervised fine-tuning
- The idea is simple and elegant, and the empirical results demonstrates that fine-tuning on a small fraction of tokens can outperform standard SFT methods

### Weaknesses
- the counterfactual computation in the first stage for token importance calculation can be computationally heavy
- it is unclear if the reasoning correctness is attributable to individual tokens independently? Reasoning may usually depend on a set of inter-dependent tokens and hence the critical token assumption could be an over-simplified assumption.

### Questions
- Does the method account for inter-token dependencies, e.g., a pair, or a set of tokens that together determine correctness even if each token alone is replaceable?
- Does perturbation of multiple non-critical tokens affect solution quality?

### Soundness
2

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
4

### Summary
This work introduces Critical Token Fine-tuning (CFT), a model-free approach to identifying critical tokens in LLM reasoning tasks. Compared with other token-level importance methods, CFT uses counterfactual substitution with only one or two greedy rollouts per token, enabling efficient evaluation without auxiliary models.

### Strengths
- CFT offers a practical and efficient way to identify critical tokens without training auxiliary models.
- The paper is clearly written with rigorous notation.
- The experiments are detailed and compare across many LLM base models and datasets.
- The authors focus on practical usage and include RL result comparisons after CFT, aligning with trends in LLM post-training pipelines.

### Weaknesses
1. The paper does not present the loss differences between critical and non-critical tokens. Critical tokens may naturally have higher uncertainty and therefore require more focus during tuning.

2. All models are trained for three epochs without a validation set or early stopping, raising a major concern: if critical tokens are naturally underlearned, CFT may obviously surpass other methods under this configuration. In real scenarios, models are typically trained to convergence (and often use a validation set to avoid overfitting), making the final result uncertain; CFT’s advantage may be primarily faster convergence rather than better asymptotic performance.

3. The problem has already been studied; some related works are missing (e.g., [1]), and more baselines should be included for comparison (e.g., Rho-1 [2]).

4. It is better to show which tokens are identified as critical, and to demonstrate that these identified tokens indeed correspond to truly critical ones (for example, by comparing them with those recognized by human experts, not necessarily to be the same, but need to be explained).

5. The counterfactual idea appears effective but not surprising.

[1] Disentangling Reasoning Tokens and Boilerplate Tokens For Language Model Fine-tuning. ACL 2025 Findings.
[2] Not all tokens are what you need for pretraining. NeurIPS 2024.

### Questions
- Can you report the loss comparison between critical tokens and non-critical tokens during the training process? For example, do critical tokens naturally have higher uncertainty and thus require more focus while tuning?

- Would you provide a fairer comparison using the best validation epoch for each method (CFT and SFT, DPO)? This can clarify whether CFT’s advantage is primarily faster convergence rather than better final performance.  e counterfactual procedure.

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
3

### Summary
The authors propose Critical Token fine-tuning (CFT), a training method that only applies the loss to tokens that are regarded important. These tokens are identified through a counterfactual perturbation process - when replacing with other candidates all yields to incorrect prediction, it is chosen. The biggest merit of CFT is that it does not rely on compute-intensive rollouts when identifying such tokens.

### Strengths
1. The experimental set up is very well written - it was clear to tell why SFT, DFT, Entropy, Attn were chosen as baselines. Also, a consistent gain across 11 benchmarks and 5 base models very well support that CFT is an effective method.

2. The fact that "CFT-initialized checkpoints begin with higher entropy for RL training and that exploration is sustained" is a very good finding and could be adopted in future works. Specifically, there has been a lot of work showing that RL training improves Pass@1 performance at the cost of reducing entropy. With a better initialization, models could be trained with even more number of steps when CFT is adopted.

3. The ablation experiments are also well designed, especially, I enjoyed reading the findings in the experiment of Section 5.2 - applying CFT to identify critical tokens on another model family. This indicates that CFT could be applied in more general settings for response filtering.

### Weaknesses
I do not see any strong weaknesses in this paper.

### Questions
Can you make the font size of the figures bigger?

### Soundness
4

### Presentation
3

### Contribution
4
