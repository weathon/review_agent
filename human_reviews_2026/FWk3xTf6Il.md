# BOW: Reinforcement Learning for Bottlenecked Next-word Prediction

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large language models (LLMs) are typically pretrained with next-word prediction (NWP), which yields strong surface fluency but places limited pressure on models to form explicit reasoning before emitting tokens. 
We study whether shifting the supervision signal can better elicit explicit reasoning and, more broadly, strengthen models’ general reasoning capability.
We present BOttlenecked next-Word exploration (BOW), a RL formulation of NWP that inserts an intermediate reasoning bottleneck. Instead of predicting the next word directly from context, the policy model must first generate a next-word reasoning trajectory. A frozen scorer then assigns this trajectory a soft, distributional reward equal to the probability of the gold next token conditioned solely on the trajectory to guide the RL optimization.
We also propose an optional L1-style regularizer on the reward to discourage “name-the-answer” shortcuts.
Across ten benchmarks, a brief BOW adaptation phase on Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct improves zero-shot reasoning and outperforms strong continual-pretraining baselines, including an RL variant with a hard, binary reward and a supervised finetuning approach with augmented data, by nearly 5\% on average, while achieving the top result in 7 of 10 intrinsic NWP evaluations.
These results indicate that BOW is a viable alternative to vanilla NWP, inducing explicit next-word reasoning and strengthening general reasoning ability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces BOttlenecked next-Word exploration (BoW), a method to fine-tune Large Language Models (LLMs) in the Reinforcement Learning (RL) formulation.  BoW extends the standard next-word prediction (NWP) training pipeline with an intermediate reasoning bottleneck, and assigns the reward according to the likelihood of the gold next token conditioned the trajectory. This formulation naturally aligns with Reinforcement Learning (RL) and can be optimized using existing policy gradient methods such as GRPO. The authors further introduce an L1-style regularizer to avoid reward hacking. Experiments on Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct demonstrate the effectiveness of propose method on several benchmarks

### Strengths
1. The paper is clearly written.
2. The proposed method outperforms baselines on 7 out of 10 benchmarks
3. The author does comprehensive analysis on the algorithms.

### Weaknesses
1. **Unclear method name**: the method "Bottlenecked next-World exploration (BOW)" in terms of both "bottleneck" and "next-world exploration". 
    - As discussed is related work and Section 3.2, "bottleneck" is just a reasoning trajectory before the final prediction. Therefore, I don't see any benefit of renaming the "reasoning trajectory" into "bottleneck"
   - From my experience, "next-world prediction" refers to the pretraining task of LLM, where the loss is applied to all the tokens in the sentence. In comparison, BOW only considers the probabilty of "gold token", which is a form of outcome reward.
2. **Lack of Novelty**: based on the discussion above, BOW is a method of Reinforcement Learning with outcome reward, which has been studied on many existing works. Particularly, the hard reward (HR) formulation corresponds to standard RLVR, while the soft-reward has also been studied on papers such as [1]
3. **Training is done on Instruction-tuned model**: The experiments are conducted on instruction-tuned model, which has been fine-tuned on some of the evaluation sets. This might also partially expains that why the improvement of BOW against Vallina is not substaintial.

[1] VeriFree: Reinforcing General Reasoning without Verifiers. arXiv preprint arXiv:2505.21493

### Questions
1. What's the difference between BOW and RLVR (HR in Table 1) and this method [1]?
2. Why in Table 1, the performance of HR degrades compared to Vallina?
3. What would be the performance on BOW if trained on Base model? 

[1] VeriFree: Reinforcing General Reasoning without Verifiers. arXiv preprint arXiv:2505.21493

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a reinforcement learning framework for training large language models (LLMs), called Bottlenecked Next Word Exploration (BOW). Unlike standard next-token (word) prediction, the model is prompted to generate a reasoning trajectory that analyzes the next-token prediction based on the previous tokens. A frozen scorer (also an LLM) evaluates the trajectory by computing the likelihood probability of the true next token, conditioned on both the context and the generated trajectory. The policy model (the LLM being trained) is optimized using Grouped Reward Policy Optimization (GRPO).

The authors conduct experiments on two LLMs, Qwen2.5-7B-I and LLaMA3.1-8B-I, with LLaMA3.1-8B-I also being the frozen scorer. The model is trained on narratives from the murder mystery domain, and the data are filtered to exclude context–next-word pairs where the next tokens do not require reasoning to infer. The proposed method is compared against the base LLM and several other training approaches, including selective language modeling, hard reward, and thoughts of words. The evaluation covers various general reasoning benchmarks as well as intrinsic next-word prediction tasks. Overall, the proposed method shows improvements over the original models on most tasks and outperforms other methods according to the reported results.

### Strengths
1. The paper generally tackles an interesting and important question about the training signals in LLMs and contributes to the line of research that explores RL as an alternative. Among the work that uses RL to post-train LLMs for incentivizing reasoning ability, the paper has some originality in applying it to a token or word level and in using a soft reward based on the reasoning trajectory rather than the final answer.

2. The paper is written with good quality and does a good job of clearly presenting what it does. The authors structure the paper well and provide detailed explanations of their design choices.

3. The experimental setting is generally valid. The compared methods and evaluated datasets seem comprehensive, and they also provide detailed ablation studies to examine effects such as reward regularization, different scorers, and the effect of training data filtering.

### Weaknesses
1. My main concern is the validity of the experimental comparisons. From Table 2, almost all baseline methods SLM, ToW, HR reduce the performance of the vanilla untrained LLMs on most datasets, especially for LLaMA3.1 8B I. I expect this happens because the authors rerun these methods by training models on the same, very limited murder mystery domain, which may cause the model to overfit and perform worse on other datasets. If the authors followed the original papers, the scores should not be that low. So even if the proposed method is better than other methods, the improvement could be due to less overfitting rather than enhanced reasoning ability, especially considering that the gains over the vanilla model are marginal, and in several cases, performance even drops a bit. This also leads me to question the validity of the training data. I understand the murder mystery data may demand reasoning, but there are other domains, such as math and coding. Why do the authors not train on a more diverse set of data? And for a fair comparison, why not keep the training closer to the baseline methods rather than adapting it to your data?

2. Besides, compared with a more popular RL tuning pipeline such as in the GRPO paper, this work applies RL tuning at a more fine grained token level, and the reward computation is also different. It is not clear to me from the paper why these two aspects are important. In the experiments, not all tokens are used, only selected tokens, which may be trivial, and results could vary across domains. I would be interested to see, for example, beyond the murder mystery domain, whether for math problems doing RL at the token level instead of on the final answer is better. Similarly for the reward model. I do see from the baselines that there is a hard reward comparison, but it is still not clear to me why a soft reward is better, and why not let the policy model directly output the answer after the reasoning trajectories and use that soft or hard as the reward. There are several design choices, but I am not clear why the proposed one is the best.

3. (Minor points.) From the abstract and introduction, the paper seems to frame the approach as completely shifting the supervision signal from next token prediction to an RL framework, but the experiments look more like a post training technique, since it needs LLMs trained with next token prediction as a good starting point for both the policy and the reward models. Stating this clearly could reduce confusion. Also, the paper does not discuss its limitations, and I recommend adding a section on that. Clearly stating when the method works and when it does not would bring more benefits to the community.

### Questions
Please refer to the weakness section.

### Soundness
2

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
The paper proposes a new reinforcement learning framework, **Bottle-necked Next-Word Exploration (BOW)**, as an alternative to standard next-word prediction (NWP) for large language models. Instead of directly predicting the next word, BOW forces the model to first generate an explicit **reasoning trajectory**, that evaluates how well the trajectory supports the correct next token. An **L1-style reward regularizer** is introduced to discourage shortcut behaviors like “naming the answer” and to encourage more general reasoning.

### Strengths
1. **Comprehensive Evaluation and Analysis** – The paper includes detailed ablations (scorer choice, regularization, data filtering) and human studies, demonstrating robustness of the method’s behavior.

2. **Novel Soft Reward** – The soft, probabilistic reward may offer smoother and denser feedback than hard binary rewards, improving exploration efficiency and stability during training.

3. **Empirical Performance Gains** – BOW outperforms strong baselines (RPT, ToW, SLM) across several reasoning benchmarks and majority of intrinsic NWP evaluations.

### Weaknesses
1. **Lack of Cost Analysis** – This is a big concern. Generating reasoning trajectories for next token prediction is computationally expensive; the paper does not quantify training time, GPU cost, or sample efficiency relative to simpler continual-pretraining methods.

2. **Narrow Experimental Domain** – Training data seem to come from *murder-mystery narratives*, a very specific genre. This raises the question of how the proposed model performs well on generic reasoning tasks, such as coding, mathematics, or dialogue. It would be useful to have a detailed description of training stages. For each stage, we need to compare the training cost and the training domain used side by side with the simple autoregressive baselines or SLMs for clarity.

### Questions
1. How costly is the proposed approach compared to vanilla next token prediction baseline?
2. How does the performance of the proposed approach generalize to out-of-domain reasoning tasks?

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
This paper reformulates the next-word prediction in LLMs as a reinforcement learning problem, instructing the policy model to output a reasoning chain that carries critical information about the gold next word information. Rewards are assigned from a separate frozen LLM scorer, which predicts the next word based on the given reasoning chain, outputting a soft probability score. The method, named bottlenecked next word exploration (BOW), achieves encouraging reasoning performance on several benchmarks and models.

### Strengths
* The idea of converting next-word prediction into a reinforcement learning issue is interesting, which also provides a way to scale RL up.
* The paper is generally well written and easy to understand.
* Experiments are based on several models and benchmarks, making the results more convincing.

### Weaknesses
* It’s unclear how BOW scales as training data increases: the experiments are only based on one data setup.
* The hard reward baseline performs much worse than the vanilla baseline, which may be misleading.

### Questions
In Dong et al. (2025)’s work, the hard reward also delivers encouraging performance, but in the experiments, this method achieves much worse performance even than the vanilla baseline. Why would this happen? Is this caused by sub-optimal optimization?

Another way to understand the effectiveness of hard reward is to convert the soft score in BOW to binary, such as if the score is larger than some threshold (like 0.2), the model gets a positive reward of 1, otherwise 0. This would better illustrate how hard and soft reward works.

### Soundness
3

### Presentation
3

### Contribution
3
