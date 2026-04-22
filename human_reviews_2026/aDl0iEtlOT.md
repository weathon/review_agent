# Outcome-based Exploration for LLM Reasoning

- Avg Score: 3.50
- Decision: Reject
- Scores: 8, 2, 2, 2

## Abstract
Reinforcement learning (RL) has emerged as a powerful method for improving the reasoning abilities of large language models (LLMs). Outcome-based RL, which rewards policies solely for the correctness of the final answer, yields substantial accuracy gains but also induces a systematic loss in generation diversity. This collapse undermines real-world performance, where diversity is critical for test-time scaling. We analyze this phenomenon by viewing RL post-training as a sampling process and show that, strikingly, RL can reduce effective diversity even on the training set relative to the base model. Our study highlights two central findings: (i) a transfer of diversity degradation, where reduced diversity on solved problems propagates to unsolved ones, and (ii) the tractability of the outcome space, since reasoning tasks admit only a limited set of distinct answers. Motivated by these insights, we propose outcome-based exploration (OBE), which assigns exploration bonuses according to final outcomes. We introduce two complementary algorithms: historical exploration, which encourages rarely observed answers via UCB-style bonuses, and batch exploration, which penalizes within-batch repetition to promote test-time diversity. Experiments on standard competition math with Llama and Qwen models demonstrate that both methods improve accuracy while mitigating diversity collapse. On the theoretical side, we formalize the benefit of outcome-based exploration through a new model of outcome-based bandits. Together, these contributions chart a practical path toward RL methods that enhance reasoning without sacrificing the diversity essential for scalable deployment.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper tackles diversity collapse during post-training RL for reasoning models and proposes shifting exploration to the outcome (answer) space rather than token trajectories. It introduces two complementary algorithms: historical exploration, which encourages infrequent answers via UCB-style bonuses, and batch exploration, which penalizes within-batch repetition of answers to promote test-time diversity. Experimental results demonstrate that their method enhances the reasoning without sacrificing the diversity during test time.

### Strengths
1. The authors thoroughly analyze diversity dynamics during training and identify a key causal issue, "the transfer of diversity degradation", where solving certain problems negatively affects the diversity of solutions generated for unsolved problems.

2. The authors introduce an additional reward term to enhance the diversity via UCB, and further introduce normalization combined with UCB to improve the test time performance. Finally, batch normalization of UCB to enhance the diversity during test time.

3. Theoretical and empirical results demonstrate the effectiveness of their methods.

### Weaknesses
1. There is a line of work that aims to mitigate diversity degradation using entropy-based methods [1] and pass@k-based training [2]. It would strengthen the paper if the authors conducted a more thorough comparison with these approaches—for example, by evaluating accuracy across different values of k, the number of unique solutions, and entropy.

2. Although the authors provide detailed accuracy results (I appreciate that), there is still a lack of analysis on the dynamics of reward, completion length, and entropy throughout training. It is also important to show how the UCB term influences the training process, perhaps by illustrating one or two specific cases on individual questions.

3. The authors fix c = 0.2, and use $b_0 = 1$ for the easy dataset and $b_0 = 0.5$ for the medium dataset. Yet, no ablation study is provided to justify these choices. As a result, the effect of these terms on training behavior, test performance, and solution diversity is still unclear.




[1] Wang, Shenzhi, et al. "Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning." arXiv preprint arXiv:2506.01939 (2025).

[2] Chen, Zhipeng, et al. "Pass@ k training for adaptively balancing exploration and exploitation of large reasoning models." arXiv preprint arXiv:2508.10751 (2025).

### Questions
1. On page 5, you state that “N(x,a) denotes the number of times answer a has been sampled for question x.” If a is a newly generated answer for x, is N(x,a) initialized as 0 or 1? If it is 0, the expression 1/N(x,a) becomes problematic, and it may be necessary to introduce a small \epsilon term for stability.

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
This paper introduces the UCB  reward mechanism—commonly used in traditional reinforcement learning to encourage exploration—into the reinforcement learning training process of LLMs. It proposes an exploration strategy based on the final answer, aiming to encourage the model to generate more diverse outputs.

### Strengths
- This method partially mitigates the trend of performance degradation on the test set as training progresses, exhibiting a certain regularization effect.

### Weaknesses
**From a methodological perspective:**

- If a model is already capable of correctly answering a given question, is it still meaningful to pursue answer-level diversity for that question? In most tasks (e.g., mathematical reasoning), the correct answer is typically unique. In such cases, "increasing answer diversity" may effectively equate to encouraging the generation of more diverse **incorrect answers**. This raises concerns about the validity of the optimization objective itself, which may even lead the model to waste exploration resources on low-quality or clearly incorrect outputs, thereby affecting training efficiency and final performance.

**From an experimental perspective:**

- The method introduces several hyperparameters (such as $c$ and $b_0$), yet the paper does not provide a systematic hyperparameter sensitivity analysis or ablation study.
- Experimental results show that, regardless of whether OBE is used, test performance tends to first increase and then decline, a phenomenon especially evident on simpler datasets. This trend resembles **overfitting** in supervised learning, suggesting that the underlying issue may stem from repeated training on limited data, leading to model memorization. The role of OBE seems to merely delay this process, without yielding significant improvements in peak performance compared to baseline methods (e.g., GRPO).

### Questions
As mentioned in the Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies why RLVR for reasoning LLMs often collapses generation diversity, hurting pass@k. Treating RL post-training as a sampling process, the authors identify (i) a transfer of diversity degradation from solved to unsolved problems and (ii) the tractability of the outcome space (few distinct answers per question). They introduce Outcome-Based Exploration (OBE): historical variants that add UCB-style bonuses on final answers—with mean or constant baselines (OBE-Mean/OBE-Con) to balance positive/negative signals—and a batch variant (OBE-Batch) that penalizes within-batch repetition to directly promote test-time diversity.

### Strengths
1. The paper cleanly frames the diversity-collapse issue in outcome-based RL and separates it into two phenomena, making the problem setup easy to follow.


2. The proposed OBE variants are simple to implement on top of existing RLVR pipelines, requiring only outcome-level bookkeeping and modest additional logic.

### Weaknesses
1. On the claim of “transfer of diversity degradation across questions”: prior work already shows that RL sharpens model distributions [1, 2, 3], so the novelty here is limited. Moreover, the explanation that degradation occurs “because the model does not update on questions it has not solved yet” is unconvincing—lack of per-instance gradient does not imply lack of indirect effects. If updates do not generalize across questions, why should we expect generalization to the test set? As written, this argument needs stronger support.


2. On the observation that “diversity is tractable on verifiable domains”: the use of final-answer multiplicity as a proxy for diversity on unsolved problems feels assumed rather than demonstrated. Please provide evidence or a stronger justification that outcome-level diversity correlates with trajectory-level diversity in the unsolved regime.


3. In Figure 4, gains are visible on Llama, but for Qwen the OBE variants appear to yield little or no improvement; a similar pattern holds in Figure 6.

### Questions
1. For the claim that “RL eventually solves fewer questions than the base model,” the methodology of defining k = n t (accumulating generations across training checkpoints) is unusual. Would it be more natural to compute standard Pass@k independently at each checkpoint? What is the motivation for cross-checkpoint accumulation, and how should we interpret it relative to the conventional metric?


2. What sampling temperatures are used pre- and post-RL? Since several works report that RL sharpens the distribution [1, 2, 3], fixing a single temperature can be unfair [2]. A fairer protocol would sweep temperature per model and report the best Pass@k (with seeds/error bars). Do the conclusions remain under this evaluation?


[1] ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models https://arxiv.org/html/2505.24864v1

[2] Decomposing Elements of Problem Solving: What "Math" Does RL Teach? https://arxiv.org/pdf/2502.17356v1

[3] Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening https://arxiv.org/abs/2506.02355

### Soundness
1

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
4

### Summary
This work studies the RL training dynamics of reasoning LLMs in the math domain. It finds that RL reduces effective diversity even on the training set compared to the base model. Specifically, it identifies, (i) diversity transfer, where reduced diversity on solved problems carries over to unsolved ones, and (ii) limited outcome space, since math reasoning tasks allow only a few distinct correct answers..  To address this,  the authors propose outcome-based exploration (OBE), which assigns exploration bonuses based on final outcomes. OBE includes two variants: 1) historical exploration, rewards rarely seen answers using UCB-style bonuses, and 2) batch exploration, penalizes duplicate answers within a batch to encourage test-time diversity. Experiments on competitive math tasks with Llama 3.1-8B and  Qwen 2.5-7B models show that both methods improve accuracy and help prevent diversity collapse.

### Strengths
1) The observation of the transfer of diversity degradation is insightful and inspirational for the community. 
2) The idea of adding UCB-based exploration bonuses based on final answers is direct and simple to implement.

### Weaknesses
1) The performance improvement of OBE according to Figure 1 is not significant in the math domain. Since the method is general, authors can consider trying it in more domains.
2) The novelty of applying UCB-based exploration bonuses into RL is not that high, given many existing work in Deep RL.

### Questions
1. Do authors have results for other domains to validate the effectiveness of the method?
2. Better explanation of Figure 2 maybe useful for readers.

### Soundness
2

### Presentation
2

### Contribution
2
