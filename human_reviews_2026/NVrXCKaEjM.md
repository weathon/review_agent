# Token-Efficient Long-Term Interest Sketching and Internalized Reasoning for LLM-based Recommendation

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Large language models (LLMs) can solve complex real-world tasks when prompted to generate chain-of-thought (CoT) reasoning, motivating their use for preference reasoning in recommender systems. However, applying LLM reasoning on recommendation faces  two practical challenges. First, LLMs struggle to reason over long, noisy user histories that often span hundreds of items while truncation discards signals needed to capture long-term interests. Second, in decoder-only architectures, CoT requires generating rationale tokens autoregressively, leading to prohibitive inference latency for real-world deployment. To address the challenges, we propose SIREN, a framework that enables effective LLM-based rating prediction via long-term interest sketching and internalized reasoning. First, instead of prompting raw histories, we build a compact, token-bounded interest sketch that preserves persistent preferences and suppresses noise. Specifically, we encode and cluster item descriptions to discover semantic topics, then compress each user’s history into a short list of liked and disliked topics, facilitating LLM reasoning. 
Second, we develop an internalized reasoning strategy for efficient inference. We adopt a two-stage training paradigm: (i) train the LLM to reason explicitly for rating prediction with rule-based reinforcement learning, since ground-truth CoTs are unavailable in recommendation; and (ii) learn to internalize CoT into model parameters through hidden alignment. At inference, the LLM directly generates the rating with near-CoT quality.
Extensive experiments show that SIREN reduces average input tokens by $48.7\%$ compared to raw-history prompting, outperforms existing methods while delivering over $100\times$ lower inference latency than CoT-based LLM recommenders. Code and data are available at https://github.com/TommyDzh/SIREN.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes SIREN, a framework to improve rating prediction with large language models (LLMs). It targets two deployment challenges: (1) long and noisy user histories that harm reasoning, and (2) high inference latency due to explicit CoT generation.

### Strengths
1. Clear practical motivation: identifies real deployment issues of long user histories and expensive CoT reasoning.
2. Elegant design: the “interest sketch + hidden-alignment” combination is conceptually simple yet effective.
3. Significant efficiency gains: achieves near-CoT accuracy with minimal latency.

### Weaknesses
1. Limited dataset diversity. Experiments are conducted only on two domains (Books and Movies) from the Amazon Reviews dataset. The generalizability to ranking, Top-N recommendation, multimodal, or more complex scenarios remains unclear.
2. Dependence on textual descriptions. The robustness of the encoding and clustering components should be further validated under conditions where item descriptions are scarce, noisy, or cross-lingual.
3. Cross-model generalization. The evaluation is limited to Qwen3-4B. It would strengthen the paper to include experiments on models with different architectures and scales (e.g., Llama, Gemma).
4. Insufficient discussion of related work. Several relevant studies are not discussed or cited, such as [1–2] on long-term user profiling and [3] on latent reasoning. 


[1] Temporal User Profiling with LLMs: Balancing Short‑Term and Long‑Term Preferences

[2] HyMiRec: A Hybrid Multi‑interest Learning Framework for Long‑Term Multi‑interest Sequential Recommendation

[3] Reinforced Latent Reasoning for LLM‑based Recommendation

### Questions
See comments in Weaknesses.

### Soundness
2

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
3

### Summary
This paper identifies two main limitations of LLM-based recommender systems: their difficulty in reasoning over long and noisy user histories, and their high inference latency. To address these challenges, the authors propose a method called SIREN, which employs long-term interest sketching to effectively process user histories and internalized reasoning to enhance inference efficiency. Experimental results demonstrate that the proposed method performs well.

### Strengths
1. The paper is well-motivated, and the two main limitations of LLM-based recommender systems it identifies are highly worth addressing.
2. Experimental results show that the proposed method effectively mitigates these limitations, demonstrating solid empirical performance.
3. The hidden state alignment component is particularly insightful.

### Weaknesses
1. In lines 254–256, the paper states that *“the final rating prediction under both answer-only (Fig. 1(c)) and CoT (Fig. 1(d)) decoding depends on the hidden state at the <answer> token.”* Could you provide additional evidence to support this claim?
2. What does the “Rank” column mean in Table 2?
3. Could you compare the training costs of these LLM-based recommender systems?
4. I think Table 2 is missing an important baseline — one where the **first stage** extends the recent history with additional past interactions until reaching the token budget (*More history*), and the **second stage** uses **GRPO-CoT**.

### Questions
see weakness.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a post-training method for LLM-based recommendation systems that can reduce input length by aggregating and clustering long histories. Through a two-stage training algorithm, it first provide the LLM with accurate recommendation capabilities, then uses hidden state supervision alignment to reduce the CoT part to zero, thereby achieving fast and accurate score computation. The work's motivation and experiments are relatively comprehensive, and the writing is fairly well-structured. However, the overall innovation is notably insufficient. In particular, the results of the ablation study (RQ4) actually demonstrate that the proposed method's optimizations for the input and output components show minimal improvement in effectiveness.

### Strengths
1. The paper proposes a two-stage post-training method based on reinforcement learning that first elicits the model's reasoning capability through RL to achieve accurate score estimation, then shortens the CoT length through an aligned "internalized reasoning" stage, thereby accomplishing the objective of being "both fast and accurate."

2. The motivation is clear and highly valuable, addressing a problem that current LLM-based systems are actively working to solve. The paper provides comprehensive experimental results demonstrating the superiority of the proposed method over other related approaches, and is also well-written.

### Weaknesses
1. The paper suffers from a notable lack of innovation, applying mature techniques from the existing community to recommendation systems. However, neither Stage 1 nor Stage 2 represents a novel idea. Moreover, the ablation study results (Figure 4) indicate that these modifications show almost no difference from direct SFT (CE in the Figure 4), with only marginal improvements (considering the test set and randomness, my personal view is that these improvements are extremely minimal).

2. As for the experimental results. Particularly in Figure 4, where simple CE achieves results very close to the original SIREN, I am unclear why your reproduced Exp3RT performs significantly worse (I noticed in the appendix that you used the same base model for training, but still have doubts about the results). Additionally, it should be noted that I did not find SFT results directly using ratings as labels in the paper, so I can only refer to your SFT results based on the Stage 1 model as the base (i.e., CE in Figure 4).

3. There are also some minor writing issues, such as subscripts that should be in roman type but were overlooked (e.g., Eq. 12), "Appendix ??" on line 491, and non-standard citation formatting (e.g., the title of Kim 2025's paper is incorrect)

### Questions
In addition to the issues mentioned in the Weaknesses section, I would also like to ask the authors: 
1. The instability and randomness of RL are widely acknowledged in the community. Have the authors encountered similar issues, and were the experimental results averaged over multiple training runs as is common practice in the RL community? 
2. When selecting the reward function, was an ablation study conducted (especially for s_{rate}) to try different forms of reward functions?

### Soundness
2

### Presentation
3

### Contribution
2
