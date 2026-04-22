# Jointly Reinforcing Diversity and Quality in Language Model Generations

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Post-training of Large Language Models (LMs) often prioritizes accuracy and helpfulness at the expense of diversity. This creates a tension: while post-training improves response quality, it also sharpens output distributions and reduces the range of ideas, limiting the usefulness of LMs in creative and exploratory tasks such as brainstorming, storytelling, or problem solving. We address this challenge with Diversity-Aware Reinforcement Learning (DARLING), a framework that jointly optimizes for response quality and semantic diversity. At its core, DARLING introduces a learned partition function to measure diversity beyond surface-level lexical variations. This diversity signal is then combined with a quality reward during online reinforcement learning, encouraging models to generate outputs that are both high-quality and distinct. Experiments across multiple model families and sizes show that DARLING generalizes to two regimes: non-verifiable tasks (instruction following and creative writing) and verifiable tasks (competition math). On five benchmarks in the first setting, DARLING consistently outperforms quality-only RL baselines, producing outputs that are simultaneously of higher quality and novelty. In the second setting, DARLING achieves higher pass@1 (solution quality) and pass@k (solution variety). Most strikingly, explicitly optimizing for diversity catalyzes exploration in online RL, which manifests itself as higher-quality responses.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a new reward function that simultaneously considers generation quality and diversity. This reward function will give high reward to responses who are of higher quality and different from other responses. In particular, when measuring the response diversity, this work focuses on semantic level and trains a (binary) classifier to check if two responses are of the same semantic meaning.

To validate their method, this work evaluates on two tasks: non-verifiable (e.g., creative writing) and verifiable (e.g., AIME 25). On non-verifiable tasks, this work demonstrates a better Pareto frontier of the trade-off between quality and diversity; on verifiable tasks, this work shows high pass@1 and pass@k rates.

### Strengths
1, I appreciate the comprehensive set of experiments, including different (non-verifiable and verifiable) tasks, model classes and sizes, and metrics. These experiments validate the generality of the proposed method.

2, The proposed method is sound and obtains strong empirical results. The ablation studies are also informative, explaining how different designs affect the performance.

3, It shows promising results on improving RL performance by generating diverse outputs. As discussed in the abstract, high diversity might contribute to more exploration during online RL training. Although this claim is not fully explored, it would be very interesting to empirically verify the hypothesis.

### Weaknesses
1, This method relies on a binary classifier to measure semantic similarities between response pairs. Therefore, the whole training pipeline will be significantly affected by the quality of the binary classifier, especially considering the reward multiplicative aggregation. If the binary classifier is suboptimal, even high quality responses might not obtain high reward.

2, Three methods (GRPO, DivPO, GRPO-Unlikeness) are selected as baselines in section 4.1. However, in most experiments, only base model and GRPO are considered. It would be good if other method results can be reported.

3, While the method is sound, its contribution on algorithm is limited.

Minor:
1, While Figure 1 explains the sampling and semantic clustering, it does not reflect the quality. It would be good to explain in Figure 1 on how quality affects the final advantage.

### Questions
1, In line 173 to 177, modifications to the GRPO objective function is discussed. However, in equation (6), it is not clear how the token-level averaging and standard deviation are incorporated. Can you explain more about how equation (6) is different from the original GRPO objective?

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
4

### Summary
This paper presents a framework that jointly optimizes both quality and semantic diversity and thus addresses a very important aspect . The main method, called DARLING, introduces a partition function that measures diversity in meaning rather than just surface-level lexical form. The online reinforcement learning encourages distinct and high quality responses. The paper has results in both verifiable and non-verifiable tasks. In verifiable rewards, the method has higher pass@1 and higher pass@k where the increased diversity improves online exploration. In addition to the optimization framework, the paper also introduces a scalable semantic classifier. The authors claim the contribution of showing that diversity promotes greater exploration which has been shown in previous work and is a very intuitive conclusion in itself.

### Strengths
1. The paper is well written, easy to follow and has clearly labeled figures

2. The works deals with a very imporant problem with LLMs

3. The results are significant

4. The qualitative examples are very helpful 

5. The ablations are rigorous, especially the one on how to use the two reward scores of the two different objectives

### Weaknesses
1. The results lack error bars. They are important especiallty in places like NoveltyBench results in Table 1. 

2. There are missing citations and analysis of the original Semantic Entropy paper [1] which is highly relevant to the metrics you use

3. Papers like [2], [3] and [4] are highly relevant and deal with the same problem. 

4. The third listed contribution in the paper showing that diversity helps exploration has been made before in papers like [4] and alluded to in most semantic entropy papers.

5. In section H, the authors claim that an LLM-judge is not used since "it induces too much computational overhead to integrate
into online training". Training a classifier on top of RL training can be a lot of overhead too. I think the authors should clarify and support this pivotal argument with more evidence. 

[1] Kuhn, Lorenz, Yarin Gal, and Sebastian Farquhar. "Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation." arXiv preprint arXiv:2302.09664 (2023).

[2] Zhang, Yiming, et al. "Forcing diffuse distributions out of language models." arXiv preprint arXiv:2404.10859 (2024).

[3] Zhu, Alan, et al. "BARE: Leveraging Base Language Models for Few-Shot Synthetic Data Generation." arXiv preprint arXiv:2502.01697 (2025).

[4] Ahmed, Eltayeb, et al. "Intent Factored Generation: Unleashing the Diversity in Your Language Model." arXiv preprint arXiv:2506.09659 (2025).

### Questions
1. Can you please clarify what x and y stand for in equation 5? I can guess but it should be more clearly written. 

2. What is the overhead of training a classifier in addition to the RL training? The paper has ablations on the accuracy of the classification with and without training, but it is important to show how a few-shot prompted classifier influences the final results as that is the most general mode of use. Also refer to point 5 in Weaknesses.

With the weaknesses being addressed, the additional citations being added (or justified why they should not be added) and most importantly with the overhead cost analysis, I am happy to increase my score.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper propose Diversity-Aware Reinforcement Learning (DARLING) to address the diversity collapse issue of post-training. DARLING is an online RL framework that jointly optimizes for response quality and semantic diversity. DARLING partitions LLM generations into semantically equivalent clusters by using a learned binary classifier, and amplify the reward for high-quality responses that are diverse (measured by how many responses are not in the the target response's semantic cluster. DARLING improves diversity and quality on instruction-following and math tasks.

### Strengths
This paper presents a novel method to incorporate a learned diversity classifier to the reward signal for post-training. The use of response partitioning to turn the binary classifier output into a continuous diversity score is clever. They also report higher diversity and quality scores than baselines (GRPO without DARLING, DivPO).

### Weaknesses
The range of tasks studied to verify the effectiveness of the method is rather narrow. I would love to see comparisons to baselines on more tasks, such as the two other tasks from DivPO paper (persona generation, keyword generation), which would further convince me of how well DARLING works.

DARLING also does not seem easily scalable, the paper does not discuss the computational overhead of calculating the diversity scores for n rollouts, where the method must perform O(n^2) pairwise inferences with the binary classifier. Also, the learned classifier needs data which may prevent the method from scaling to other domains.

### Questions
How much does the success of DARLING depend on the accuracy of the semantic classifier?
What happens when all model rollouts are semantically equivalent?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose an LLMS RL post-training framework that jointly optimizes for quality and semantic diversity. It uses a "learned partition function" (a semantic classifier) to group similar responses, then fuses the quality reward with a diversity signal. This reward design can generate both high-quality and diverse responses (in less-frequent semantic groups), encouraging the model to find multiple distinct solution paths. It partially solved the diversity/entropy collapse issue in LLMs' RL post training.

### Strengths
- The proposed framework addresses the important trade-off issue between quality and diversity during RL training. From the experiment results, the proposed technique is effective across both verifiable tasks and non-verifiable tasks, improving pass@1 and pass@K jointly. 

- The proposed semantic classifier measures diversity at the semantic level rather than the superficial lexical level, capturing a more genuine diversity measurement during training.

### Weaknesses
- The current experiment results are limited in establishing the proposed technique is advantageous. There are other ways to improve a model's entropy, for example changing the temperature to encourage more diverse and stochastic output, etc. The authors didn't compare empirically with other existing solutions to address the entropy collapse issue [1] - [6].

- The current design is more heuristic-based and the entire system's performance is highly dependent on the quality of this partition function. If the classifier groups answers poorly, it will provide a noisy or incorrect diversity signal, which could hurt the main quality training. The entire system's performance should not rely heavily on a external component that is trained separately. 

- The introduced "learned partition function" acts as an extra classifier. This adds significant engineering complexity and computational overhead to the RL training loop. There is lack of discussion on this aspect. 


[1] Dai et al. CDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models. 2025

[2] Li et al. Jointly Reinforcing Diversity and Quality in Language Model Generations. 2025

[3] Cui et al. The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models. 2025

[4] Park et al. Clip-Low Increases Entropy and Clip-High Decreases Entropy in Reinforcement Learning of Large Language Models. 2025

[5] Shen, H. On Entropy Control in LLM-RL Algorithms. 2025

[6] Zhao et al. The Majority is not always right: RL training for solution aggregation. 2025

### Questions
See the above Weaknesses. Mainly including:

- Comparison with other diversity encouraging approaches

- Sensitivity of the designed classifier and justifying its design validity

RL training Computational complexity after introducing the learned partition function, as intuitively, it will add significant engineering complexity and computational overhead to the RL training loop. So, please justify the practicality of this design.

### Soundness
2

### Presentation
3

### Contribution
2
