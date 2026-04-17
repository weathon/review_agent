# RLP: Reinforcement as a Pretraining Objective

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6, 8

## Abstract
The dominant paradigm for training large reasoning models starts with pre-training using next-token prediction loss on vast amounts of data. Reinforcement learning, while powerful in scaling reasoning, is introduced only as the very last phase of post-training, preceded by supervised fine-tuning. While dominant, is this an optimal way of training? In this paper, we present RLP, an information-driven reinforcement pretraining objective, that brings the core spirit of reinforcement learning---exploration---to the last phase of pretraining. The key idea is to treat chain-of-thought as an exploratory action, with rewards computed based on the information gain it provides for predicting future tokens. This training objective essentially encourages the model to think for itself before predicting what comes next, thus teaching an independent thinking behavior earlier in the pretraining. More concretely, the reward signal measures the increase in log-likelihood of the next token when conditioning on both context and a sampled reasoning chain, compared to conditioning on context alone. This approach yields a verifier-free dense reward signal, allowing for efficient training for the full document stream during pretraining. Specifically, RLP reframes reinforcement learning for reasoning as a pretraining objective on ordinary text, bridging the gap between next-token prediction and the emergence of useful chain-of-thought reasoning. Pretraining with RLP on Qwen3-1.7B-Base lifts the overall average across an eight‑benchmark math‑and‑science suite by 19%. With identical post‑training, the gains compound, with the largest improvements on reasoning‑heavy tasks such as AIME25 and MMLU‑Pro. Applying RLP to the hybrid NVIDIA-Nemotron-Nano-12B-v2-Base increases the overall average from 42.81% to 61.32% and raises the average on scientific reasoning by 23%, demonstrating scalability across architectures and model sizes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces RLP, a pretraining paradigm that integrates RL into training reasoning LLMs. RLP reframes CoT generation as an action taken before next-token-prediction, and this action is rewarded by a verifier-free information gain which measures how much the CoT improves the model's likelihood of predicting the correct token. The no-think model adopts a DQN fashion update to keep its distribution close to the behavior policy. I think the authors' proposed method is a clever way to overcome the sparse reward and verifier-must conditions in applying RL to training a reasoning model.

### Strengths
* The paper presents a foundational shift by moving RL to the pretraining stage, and addresses the core limitation of standard next-token prediction.
* The information-gain reward is a key innovation, providing a dense, self-supervised signal on any text without needing external verifiers.

### Weaknesses
* While showing scalability, the most detailed experiments and ablations are on a relatively small 1.7B parameter model. The paper would be strengthened by a deeper quantitative analysis on a larger, more state-of-the-art model size (e.g., 7B+). Also, I would like to see the comparison with an instruct-tuned model.

### Questions
The no-think EMA baseline appears to serve a similar role to the target network in DQN, where the update frequency is a critical hyper-parameter that balances target stability against the staleness of the learning signal. In RLP, how sensitive are the training stability and final performance to the EMA decay rate? Did the authors experiment with different decay schedules for $\tau$, and can they characterize this trade-off?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work introduces a new method for applying RL right after the pre-training phase. In particular, this involves generating reasoning traces with a base model and computing a reward signal based on how well these reasoning traces increase the likelihood of ground-truth next-tokens labels in a given dataset. The authors call this method RLP and test it on top of a Qwen 1.7B base transformer model and a Nemotron hybrid model, evaluating directly on generation-based tasks.

### Strengths
- Overall, the exposition of the method is clear, and the simple theoretical considerations based on Jensen's inequality in section 2.2 to show the loss is a sensible optimization objective are concise and not overemphasized.
- The authors apply their method to a hybrid SSM like Nemotron. Moreover, they also experiment with different data sources in Table 4. This degree of diversity is nice to see.
- While code is not released, the authors do provide training and hyper-parameter details in Appendices 9 and 10, which provide helpful transparency to the method and baselines considered.

### Weaknesses
**Main concerns**
My main concerns are about the soundness of the paper's empirical validation. The main quantitative results in tables 1, 3, and 4 feature a Qwen3 1.7B base model. This base model is extended and compared with RLP on downstream reasoning tasks. There are several aspects about how the evaluation is carried out and the baselines that prompted my concerns:
1) Evaluating base models without any alignment on generation tasks and without even "few-shot" examples is doomed to underperform, as these models have never been primed for question-answering. Thus, I am unsure how relevant it would be to compare RLP with continued pretraining and base models trained on fixed non-masked datasets (which make up the bulk of the reported results), as pretraining corpora are purposefully not meant to imbue the models with alignment capabilities. The RLP traces generated with low temperature (appendix 9) and filtered based on likelihood should inherently make the model's logits more skewed (something that even occurs in RL with random rewards [1]).
2) Looking at the hyperparameters for the continued pre-training baseline (CPT), these appear quite odd and suboptimal (e.g., using a much smaller context than what Qwen3 was pretrained with in the first place). Beyond pretraining, the "post-training" procedure in Table 1 uses the OpenThought dataset that was conceived for finetuning already post-trained "instruct" models, and is much smaller and less diverse than common datasets used for the first SFT phase.
3) One of the key downsides of RL is its disproportionate cost compared to other training stages. Yet, with one exception, all "continued pre-training baselines" seem to only equate the size of RLP's "seed" dataset, which I believe implies that the training budget and tokens used for CPT would be disproportionately more (likely well over an order of magnitude). There is a single baseline in Table 4 that is described to be "equating FLOPs", yet, looking at Section 10 in the Appendix, this FLOP calculation does not take into account the autoregressive bottleneck of generating reasoning traces with RL. If I am not missing something, I believe this likely implies the training time for CPT would still be much larger.

Based on 1), 2), and 3), I find it hard to extrapolate that the proposed method would provide any benefit to an actual LLM training pipeline and would be worth the high computational demands of autoregressive generation at an early training stage. In order to provide convincing evidence, I would suggest comparing their final model directly to the "instruct" version of Qwen 3 1.7B, but I suspect that given 2), even their final post-trained version would significantly underperform with respect to this properly aligned baseline.

**Other**
1. I did not find the role of the EMA model to compute the baseline to be properly explained. While the authors briefly mention the reward hacking hypothesis, this does not seem to be empirically validated. Empirically, did the authors find divergent behaviors without EMA? Keeping EMA in memory also requires storing a separate copy of the network's weight, with immediate additional cost concerns that are not currently discussed.
2. When comparing with RPT, the authors mention 1 epoch training "with matched data and compute". Yet, it's unclear to me how both can be true at the same time, as RPT performs token-level filtering before training. To equate compute, RLP would have to train on a much smaller number of unfiltered datapoints from Omni-MATH.
3. Without strong empirical validation or analysis, the overall contribution of the paper seems quite limited to me. Generating reasoning traces during pretraining was already done by [2] (which the authors do cite), while rewarding reasoning traces with the log probabilities of ground-truth completion tokens was already done by [3] (currently, not cited).


[1]  Shao, Rulin, et al. "Spurious rewards: Rethinking training signals in rlvr." arXiv preprint arXiv:2506.10947 (2025).

[2] Hatamizadeh, Ali, et al. "RLP: Reinforcement as a Pretraining Objective." arXiv preprint arXiv:2510.01265 (2025).

[3] Cetin, Edoardo, Tianyu Zhao, and Yujin Tang. "Reinforcement Learning Teachers of Test Time Scaling." arXiv preprint arXiv:2506.08388 (2025).

### Questions
I have detailed what I found to be the biggest areas of improvement and my questions about this work in the Weaknesses section of my review. I would encourage the authors to address these points and help clarify any potential misconceptions for the discussion phase.

### Soundness
1

### Presentation
3

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
The paper proposes a method called RLP for enhancing model reasoning without a verifiable reward. The method maximizes the improvement in next‑token prediction when conditioning on a reasoning trace, relative to ordinary prediction. RLP reports superior performance on math and science tasks.

### Strengths
- The method is intuitive and easy to implement.

- The ablation study is diverse and comprehensive.

### Weaknesses
- Despite promising results, the objective of maximizing improvement could, in principle, degrade the model’s base predictions (to inflate the reward).

- Nemotron‑nano‑12B‑v2 and Qwen3‑1.7B‑Base are closely related models. As far as I know, Nemotron was refined with data generated by the Qwen3 family. Experiments on more independent models (e.g., Llama) would strengthen the claims.

### Questions
Intuitively, the simplest way to obtain a high improvement reward is to push up the teacher model’s cross‑entropy. I assume the EMA on teacher weights helps prevent such collapse. Still I have a few questions:

- Did you observe such collapse in any experiments? If so, which modifications helped prevent it?

- Does the model’s final perplexity (on ordinary tokens) degrade after this post‑training, especially when RLP is applied on standard pre‑training data?

In methods like GRPO, generation speed becomes the bottleneck. Could you report the wall‑clock time of RLP training versus SFT? I realize your implementation may not be fully optimized, but wall‑time comparisons would still be informative

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
3

### Summary
This paper proposes Reinforcement Learning Pre-training (RLP), an objective that moves RL-based reasoning into the pretraining phase. The model learns to generate an internal Chain-of-Thought before next-token prediction, using a verifier-free "information gain" reward compared to a "no-think" EMA baseline. Experiments on 1.7B and 12B models show RLP improves reasoning benchmarks over standard pretraining, with gains persisting after post-training.

### Strengths
1. Conceptual Novelty: The key novelty is integrating RL-based reasoning into pretraining, not just post-training. The proposed "information gain" reward is verifier-free and self-contained, allowing it to be applied to general text corpora without needing curated datasets.
2. Strong Empirical Gains: The method shows significant empirical gains over several key baselines, including a continuous pretraining baseline that is matched for total FLOPs.
3. Compounding Improvements: The pretraining gains from RLP are shown to compound, rather than be "washed out," by standard post-training (SFT + RLVR), suggesting it builds a stronger reasoning foundation.

### Weaknesses
1. Unclear Computational Cost & Baseline Fairness: The computational cost analysis and baseline comparison are a key weakness. RLP appears significantly more expensive than standard pretraining (e.g., 16 rollouts at every token). The main "compute-matched" baseline (CPT on 6B tokens vs. RLP on 170M) is confounded by seeing 35x more data. A clearer comparison would be a CPT baseline trained on the same data for the same FLOPs. 
2. Marginal Gains Over RPT: The comparison with RPT is not that convincing. As shown in Table 3, the performance difference between RLP and RPT is marginal. Given that RPT uses a simpler sparse, binary reward, this small gap raises questions about the necessity of RLP's more complex, dense reward. Furthermore, the paper does not provide sufficient detail to assess if the "matched data and compute" setting for this comparison was strictly fair.
3. Limited Scale of Experiments: The experiments are limited to small/medium models (1.7B, 12B). Demonstrating large relative gains on small models (which have near-random performance on some hard tasks) is less convincing.

### Questions
Please refer to weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes how to use RL during pretraining - where the model can be trained to output generations that increase the likelihood of next token prediction on the original corpus. The reward is the log-likelihood improvement of the dataset token when conditioned on both the context and the sampled reasoning trace, compared to conditioning on the context alone. This produces a dense, verifier-free reward signal that can be computed on ordinary pretraining text, enabling reinforcement-style updates during pretraining without external verifiers or curated datasets. Therefore, this allows for training the model to “think before predicting,” encouraging internal reasoning behavior that improves predictive utility and persists through post-training.

### Strengths
This paper proposes a clean and scalable way to insert RL in the pre-training phase. In addition the RL signal is dense, thus scalable and data efficiency, outperforming strong baselines on reasoning benchmarks while using only a fraction of the data.

### Weaknesses
The paper defines reward purely in terms of internal log-likelihood improvement, making the entire learning process self-referential. The model is effectively rewarding itself for being more confident, not necessarily for being more correct. This creates a risk that RLP amplifies patterns that increase internal consistency without corresponding improvements in truthfulness or reasoning quality.

While the authors frame the method as encouraging reasoning, there’s little evidence that the model is actually learning to reason. The improvements could just as well come from implicit regularization or longer-context modeling rather than genuine emergence of reasoning. There’s no analysis of the thought traces themselves to support the interpretation.

### Questions
Based on the weaknesses, can the authors talk more about when is the right time to introduce RL in the pretraining phase - too early and the reward is less meaningful. Furthermore, can the authors talk about when to use this - for example, if the context is a knowledge / fact then this method won't be too useful. 

Its also worth citing this work: https://arxiv.org/abs/2502.19402, which recommends this direction.

### Soundness
3

### Presentation
3

### Contribution
3
