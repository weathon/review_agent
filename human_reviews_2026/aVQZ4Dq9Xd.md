# SimKO: Simple Pass@K Policy Optimization

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has advanced the reasoning capabilities of large language models (LLMs). However, prevailing RLVR methods exhibit a systematic bias toward exploitation over exploration, as evidenced by improved pass@1 but reduced pass@K (K>1) performance. Such bias limits the advancement of LLMs’ reasoning frontier. To understand this issue, we analyze training dynamics of RLVR methods by tracking the token-level probability distributions over vocabulary candidates. Our analysis reveals a consistent probability concentration effect where the top-1 candidate increasingly accumulates probability mass and suppresses that of other candidates. More importantly, stronger over-concentration correlates with worse pass@K performance. Inspired by this finding, we propose Simple Pass@K Optimization (SimKO), a method designed to mitigate the over-concentration issue, thereby encouraging exploration. SimKO operates in an asymmetrical manner. For verified-correct responses, it boosts the probabilities of the top-K candidates. For verified-incorrect responses, it applies stronger penalties to the top-1 candidate. We observe that this asymmetric design is particularly effective at mitigating over-concentration when applied at tokens with high entropy. Across various math and logical-reasoning benchmarks, SimKO consistently yields higher pass@K for a wide range of K, providing a simple way to improve RLVR’s exploration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies the over-concentration issue in Reinforcement Learning with Verifiable Rewards (RLVR), where models tend to assign excessive probability mass to the top-1 token, hurting exploration and degrading pass@K performance.  The authors propose SimKO (Simple Pass@K Optimization), which mitigates this problem by redistributing gradient updates among top-K candidates for correct samples and strengthening penalties on the top-1 candidate for incorrect ones.  Analyses on token-level entropy and top-K probabilities provide intuitive evidence for the method’s mechanism.

### Strengths
1. The method is simple but effective, aligning well with intuition about exploration–exploitation balance.  
2. The visualization of top-K probabilities and token-level entropy distributions clearly demonstrates the concentration problem and how SimKO alleviates it.

### Weaknesses
1. Even though the paper classifies itself as an entropy-based exploration enhancement method, since all these methods aim to enhance exploration (i.e., improve pass@k), I believe it should also be compared with methods that directly optimize pass@k [1–3].
2. The method introduces multiple hyperparameters, which weakens its simplicity and may limit real-world practicality.  

Minor presentation issues:
   - Figure 2(c) is not mentioned in the main text, though it appears to illustrate why policy entropy is insufficient, and that a more detailed token-level entropy should be used instead.

   - Lines 186–192 and Figure 3(d) are somewhat confusing—the relationship (PSR > GRPO > NSR for top-K gap, and PSR < GRPO < NSR for pass@256) could be displayed more clearly in the figure.

### Questions
1. How is Equation (4) derived? A explanation connecting it to the gradient of log π(y | s) in GRPO would help clarify the derivation.  

2. In Sec. 4.2, the smoothing for non-top-1 candidates seems uniform. Why not redistribute according to their actual probabilities among the top-K set?  

3. In the experiments of Yue et al.[4], mathematical reasoning tasks are often evaluated up to pass@1024, where the phenomenon of the base model surpassing GRPO becomes clear.  
   In contrast, in Figure 1 (left) of this paper, the base model only just matches GRPO performance.  
  It would be helpful if the authors could extend Table 1 to include K = 1024 results, to observe whether SimKO can still outperform the base model at larger K.


[1] Christian Walder and Deep Karkhanis. *Pass@K Policy Optimization: Solving Harder Reinforcement Learning Problems.* arXiv preprint arXiv:2505.15201, 2025.  
[2] Sadegh Mahdavi, Muchen Li, Kaiwen Liu, Renjie Liao, and Christos Thrampoulidis. *Beyond Accuracy: A Policy Gradient Reweighting Approach for Pass@K Maximization in LLMs.* In *2nd AI for Math Workshop @ ICML 2025*, 2025.  
[3] Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, and Guang Shi. *Pass@K Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models.* arXiv preprint arXiv:2508.10751, 2025.  
[4] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* arXiv preprint arXiv:2504.13837, 2025.

### Soundness
3

### Presentation
2

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
This paper presents SimKO, a simple reinforcement learning approach to improve the exploration ability of large language models trained with Reinforcement Learning with Verifiable Rewards (RLVR). The method addresses the over-concentration of probability on the top-1 token observed in existing RLVR methods like GRPO, which boosts pass@1 but degrades pass@K. SimKO mitigates this by adjusting token-level updates: it focuses on high-entropy (“forking”) tokens, redistributes positive gradients among top-K candidates, and applies stronger negative updates to top-1 tokens for incorrect responses. Experiments on math and logic reasoning tasks show that SimKO consistently improves pass@K without sacrificing pass@1, achieving a better balance between exploration and exploitation.

### Strengths
The paper addresses an important limitation in RLVR by tackling the over‑concentration of probability mass on top‑1 tokens. It proposes SimKO, a simple gradient redistribution method that enhances exploration without adding significant computational cost. The approach is easy to integrate into existing GRPO frameworks and consistently improves pass@K while maintaining strong pass@1 performance. The token‑level probability analysis provides valuable insight into learning dynamics.

### Weaknesses
1. The paper does not discuss the effect of temperature, which directly influences exploration and top‑K behavior.
2. The definition of forking tokens through an entropy threshold appears sensitive to hyperparameter selection, and the criterion for identifying such tokens is mostly empirical. 
3. Although SimKO claims improved exploration, top‑1 probabilities remain dominant in Figure.5, with top‑2 tokens often being 100–1000× smaller. The impact of this large gap on sampling trajectories is not analyzed, and the connection between token‑level distribution changes and pass@K improvements remains unclear.
4. Averaging results across datasets of different sizes and difficulty levels may reduce representativeness.
5. Some training details, such as the unit or interpretation of  τ, appear inconsistent between the main text and the supplementary materials.

### Questions
1. How sensitive are the results to temperature?
2. Why does pass@1 also improve if the method primarily encourages exploration, even though the top‑1 token probability is significantly lower than in other methods?
3. The detailed results in Table 4 show inconsistent margins and variations—could the authors clarify these discrepancies?

### Soundness
3

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
5

### Summary
This paper addresses an important problem in reinforcement-learning-with-verifiable-rewards (RLVR) for large language models (LLMs): existing RLVR methods improve pass@1 but degrade pass@K (K>1) performance because the model increasingly concentrates probability mass on the top-1 token choice. The authors carry out a comprehensive token-level analysis of the probability distribution dynamics under GRPO, showing that as training proceeds the top-1 candidate becomes dominant and others collapse, correlating with worse pass@K. They then propose a method called SimKO. Experiments across several math/logic benchmarks and multiple model backbones show consistent improvements in pass@K (for many K values) without sacrificing pass@1.

### Strengths
- The problem formulation is clear and well-motivated: the discrepancy between improvements in pass@1 versus pass@K is an interesting and practically meaningful phenomenon.

- The token-level analysis is quite thorough.

- The proposed method is conceptually simple, well integrated with RLVR (GRPO) training, and the empirical results do support the claims of improved pass@K on strong benchmarks.

### Weaknesses
The connection between the analysis and the proposed method is somewhat loose. While the paper convincingly shows that some tokens (i.e., uncertain, high-entropy points) strongly affect the diversity of generated samples during inference, it is less clear **why selectively updating only this subset during training should lead to improved model updates or higher pass@K**. The mechanism by which focusing on these tokens enhances learning stability or exploration remains intuitive but not rigorously supported.

In particular, prior work (e.g., [arXiv:2505.12929](https://arxiv.org/abs/2505.12929)) has shown that low-probability tokens can dominate gradient updates and play a critical role in policy improvement. This raises the question of whether **a more gradient-level analysis**, showing how the selective update affects the effective gradient magnitude, direction, or diversity, would provide a more solid theoretical basis. Also, the analysis in SimKO has some overlaps with this prior work.

Furthermore, the proposed method is incremental relative to existing token-subset or entropy-based strategies (such as the “80/20” approaches). The differences are modest, and the paper does not include direct comparisons with these closely related baselines. Strengthening this part of the evaluation would help clarify the novelty and impact of the contribution.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies a limitation in RLVR: popular methods (e.g., GRPO) tend to overemphasize exploitation, leading to overly concentrated token probabilities on the top-1 candidate — improving pass@1 but reducing exploration and thereby hurting pass@K. To address this, the authors analyze token-level posterior distributions during RLVR training and propose Simple Pass@K Optimization (SimKO), which redistributes gradients across the top-K candidates for correct responses and applies asymmetric penalties to reduce overconfidence on incorrect ones. Empirically, SimKO improves pass@K (up to K=256) without sacrificing pass@1 across math and logic benchmarks, demonstrating a better balance between exploitation and exploration compared to prior RLVR variants.

### Strengths
- The paper addresses an important and practical issue in RLVR training: the systematic tendency of existing methods to improve pass@1 at the expense of pass@K, reflecting an over-emphasis on exploitation over exploration. Their proposed method, SimKO, is conceptually intuitive, simple, yet directly targets the identified cause of in their analysis. Overall, the paper is also well written and easy to follow.

- The empirical evaluation is comprehensive and robust, spanning different model backbones, diverse mathematical and logical reasoning benchmarks, and a broad spectrum of pass@K values. The consistent improvements across baselines, including strong RLVR variants, support the claim that SimKO achieves a better balance between exploitation and exploration.

### Weaknesses
I did not see major methodological weaknesses in the proposed approach, and the paper is clearly presented with strong empirical results. However, the core message and motivating analysis that reinforcing only correct samples tends to over‑concentrate the distribution and harm pass@K, whereas penalizing incorrect samples preserves exploration, appears to overlap with insights and findings already established in prior work [1] that decomposes RLVR into positive sample reinforcement and negative sample reinforcement. In particular, [1] shows that negative‑only reinforcement can improve the full pass@K spectrum by suppressing incorrect reasoning paths and redistributing probability mass toward plausible alternatives. This suggests that the SimKO’s analysis may not be entirely novel, even though its gradient redistribution implementation differs.

[1] Zhu et al. *"The surprising effectiveness of negative reinforcement in LLM reasoning"*. NeurIPS 2025.

### Questions
It would strengthen the paper to discuss [1] in greater depth (as it appears to be mentioned only briefly) and also to include [1] as baseline in Section 5.2’s experiments.

### Soundness
3

### Presentation
3

### Contribution
3
