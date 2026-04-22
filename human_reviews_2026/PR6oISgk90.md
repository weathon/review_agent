# Reinforced Preference Optimization for Recommendation

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 6, 6, 6

## Abstract
Recent breakthroughs in large language models (LLMs) have fundamentally shifted recommender systems from discriminative to generative paradigms, where user behavior modeling is achieved by generating target items conditioned on historical interactions.
Yet current generative recommenders still suffer from two core limitations: the lack of high-quality negative modeling and the reliance on implicit rewards. 
Reinforcement learning with verifiable rewards (RLVR) offers a natural solution by enabling on-policy sampling of harder negatives and grounding optimization in explicit reward signals. 
However, applying RLVR to generative recommenders remains non-trivial. 
Its unique generation space often leads to invalid or repetitive items that undermine sampling efficiency, and ranking supervision is sparse since most items receive identical zero rewards.
To address these challenges, we propose \textbf{Reinforced Preference Optimization for Recommendation} (\textbf{ReRe}), a reinforcement-based paradigm tailored to LLM-based recommenders, an important direction in generative recommendation. 
ReRe incorporates constrained beam search to improve sampling efficiency and diversify hard negatives, while augmenting rule-based accuracy rewards with auxiliary ranking rewards for finer-grained supervision.
Extensive experiments on three real-world datasets demonstrate that ReRe consistently outperforms both traditional and LLM-based recommenders in ranking performance.
Further analysis shows that ReRe not only enhances performance across both base and SFT-initialized models but also generalizes robustly across different backbone families and scales.
Beyond empirical gains, we systematically investigate the design space of RLVR in recommendation across generation, sampling strategy, reward modeling, and optimization algorithm, offering insights for future research.
Our codes are available at \url{https://anonymous.4open.science/r/ReRe-E1B0}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Reinforced Preference Optimization (RPO), a reinforcement learning–based extension of Direct Preference Optimization (DPO) for recommendation tasks. Unlike standard DPO, which aligns models to short-term revealed preferences, RPO introduces a utility critic that estimates long-term user satisfaction, combining it with reward modeling and policy updates. The framework alternates between preference optimization (based on pairwise comparisons) and utility reinforcement (based on delayed feedback).

### Strengths
The overall idea of integrating a reinforcement objective into DPO is reasonable and grounded in established RL principles. However, the derivation of the “utility critic” and the way it interacts with the preference policy is somewhat heuristic. The connection to standard RL formulations (e.g., Q-learning or advantage functions) is not rigorously developed.

### Weaknesses
The paper claims that reinforcement learning with verifiable rewards (RLVR) “naturally” addresses implicit reward issues in LLM-based recommenders, but this claim is not rigorously justified.

I am wondering what is verifiable reward? what will be the difference between verifiable reward and traditional reward in DL/RL?

The core of ReRe—beam search for sampling and ranking-based auxiliary rewards—is an extension of existing DPO and GRPO concepts. There is no much novel optimization algorithm or unique reward modeling principle. My main concern is that the proposed method primarily combines existing ideas (beam search, ranking loss, constrained decoding) with limited algorithmic innovation.

### Questions
Please refer to weakness part.

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
This paper proposes ReRe to fix two flaws of LLM-based generative recommenders: poor high-quality negative modeling and reliance on implicit rewards. ReRe uses constrained beam search to improve sampling efficiency/diversify hard negatives and combines rule-based with ranking rewards for finer supervision.

### Strengths
1. ReRe effectively addresses the two key flaws of LLM-based generative recommenders (insufficient high-quality negative modeling and reliance on implicit rewards) by integrating constrained beam search (for improving sampling efficiency and diversifying hard negatives) and a combined reward (rule-based accuracy + auxiliary ranking rewards), directly tackling the unique generation space and sparse supervision challenges of RLVR adaptation .

2. The study uses three real-world datasets (Amazon Toys, Amazon Industrial, Yelp) and compares ReRe with diverse baselines.

3. ReRe maintains robust performance across different backbone models (Qwen2.5-1.5B, Gemma-2B, Qwen2.5-7B) and initialization methods (Base, SFT).

### Weaknesses
1. There is a lack of training-time efficiency comparisons with other generative recommendation methods (e.g., those that do not use reinforcement learning) as well as with traditional methods.

2. The dataset information in Table 5 is not clearly described, and the experiments rely exclusively on relatively small-scale datasets.

### Questions
1. In Table 5, do the numbers for “Tran” refer to the number of interactions or the number of users?

2. Can this method be combined with generative recommendation approaches (e.g., TIGER)? A semantic ID–based generative paradigm appears more practically viable, whereas relying on text prompts may constrain inference efficiency.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors have addressed reinforcement learning in generative recommenders. The authors found that existing works often rely on implicit rewards and ignore high-quality negative modeling. To address these two problems, this paper proposes to integrate the RLVR into the post-training of LLM-based recommenders. For adaptation, this paper proposed a constrained beam search and an augmenting rule-based accuracy reward. The extensive experiments have validated the effectiveness of the proposed method.

### Strengths
+ S1. This paper is well-organized and -written, making it easy to follow.
+ S2. This paper is well-motivated. 
+ S2. Extensive experiments have been conducted.
+ S3. The code is released, making it easy to reproduce.

### Weaknesses
- W1. Some up-to-date papers were ignored by this paper. For example, LatentR3[1] also adopted the GRPO algorithm. What's the difference between ReRe and previous works?
- W2. It is better to further investigate the generality of the proposed ReRe. This paper has investigated the LLM-based recommender with the input of item titles, but how about the one with item ID, such as E4SRec or LLaRA?



[1]. Zhang, Yang, et al. "Reinforced Latent Reasoning for LLM-based Recommendation." *arXiv preprint [arXiv:2505.19092](https://arxiv.org/abs/2505.19092)* (2025).



[2]. Li, Xinhang, et al. "E4srec: An elegant effective efficient extensible solution of large language models for sequential recommendation." *arXiv preprint [arXiv:2312.02443](https://arxiv.org/abs/2312.02443)* (2023).



[3]. Liao, Jiayi, et al. "Llara: Large language-recommendation assistant." *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval*. 2024.

### Questions
All my questions have been included in the weakness section.

### Soundness
4

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
This paper addresses two major limitations of Reinforcement Learning with Verifiable Rewards (RLVR) in generative recommendation models. First, the unique generative space often produces invalid or duplicate items, reducing sampling efficiency. Second, since most items receive identical zero rewards, the ranking supervision signals become sparse. To overcome these issues, the authors propose Reinforced Preference Optimization for Recommendation (ReRe). Specifically, ReRe introduces a constrained beam search mechanism to improve sampling efficiency and increase the diversity of hard negative samples. In addition, it supplements rule-based accuracy rewards with auxiliary ranking rewards, enabling finer-grained supervision.

### Strengths
1.ReRe effectively addresses the challenge of hard negative sampling by introducing constrained beam search.

2.The use of ranking rewards alleviates the limitations of binary rule-based supervision.

3.The experimental results demonstrate solid performance and validate the method’s effectiveness.

4.The overall structure and logic of the paper are clear and well-organized.

### Weaknesses
1.For LLM-based recommender systems, prompt design is a crucial component, yet the paper does not discuss it.

2.Figure 2 fails to clearly illustrate ReRe’s contribution to negative sample sampling; further clarification or revision is needed.

3.Although constrained beam search mitigates the generation of invalid or duplicate items, ReRe may still be biased toward popular items due to the long-tail distribution, potentially overlooking less popular ones.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
