# Inverse Reinforcement Learning with Dynamic Reward Scaling for LLM Alignment

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Alignment is vital for safely deploying large language models (LLMs). Existing techniques are either reward-based--train a reward model on preference pairs and optimize with reinforcement learning (RL)--or reward-free--directly fine-tune on ranked outputs. Recent research show that well-tuned reward-based pipelines remain the most robust, and single-response demonstrations can outperform pairwise preference data. 
However, there still exist two key challenges: (1) imbalanced safety dataset that overrepresent common hazards while neglecting long-tail threats; and (2) static reward models that ignore task difficulty, limiting optimization efficiency and attainable gains.
To address these limitations, we propose DR-IRL, which Dynamically adjusts Rewards through Inverse Reinforcement Learning. 
We first train category‑specific reward models using a balanced safety dataset of seven harmful categories as demonstration via IRL.
Then we enhance Group Relative Policy Optimization (GRPO) by introducing dynamic reward scaling--adjusting rewards by task difficulty--data-level hardness by text encoder cosine similarity, model-level responsiveness by reward gaps. 
Extensive experiments across various benchmarks and LLMs demonstrate that DR-IRL outperforms all baseline methods in safety alignment while maintaining usefulness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes DR-IRL, a framework that improves large language model alignment by combining inverse reinforcement learning with dynamic reward scaling. Instead of relying on static reward models or imbalanced datasets, DR-IRL trains category-specific reward models using a balanced demonstration dataset and adjusts rewards during optimization based on task difficulty and model responsiveness. The approach integrates these adaptive signals into the GRPO algorithm, focusing training on challenging, high-risk prompts without overfitting to easy ones. Experiments on Llama-3.1 and Qwen-2 models show that DR-IRL consistently improves safety and robustness while maintaining model utility, outperforming strong baselines like DPO, GRPO, and STAIR across multiple benchmarks.

### Strengths
1. The overall pipeline is technically solid, effectively combining demonstration-based IRL with GRPO into a unified training procedure that reduces reliance on costly human preference labeling.
    
2. The method demonstrates strong empirical performance, outperforming multiple baseline approaches on safety benchmarks while maintaining comparable utility.
    
3. The paper includes thorough analyses and ablations that examine the effects of dynamic scaling, category-specific reward models, and data balancing, adding credibility to the experimental conclusions.

### Weaknesses
1. The data-hardness and model-responsiveness design largely follows existing adaptive weighting ideas (e.g., β-DPO [1], DAMA [2]). The conceptual novelty is limited, as cosine similarity and reward gap have been used similarly in prior work.
  
2. The IRL setup assumes that demonstration responses are always superior to model-generated ones. This assumption may not hold and could cause the policy to collapse toward demonstration templates, reducing behavioral diversity.
    
3. Although the authors report that training seven separate reward models increases GPU hours by only ~20% compared with only one reward model, this approach is not quite promising. While I understand category-specific models may offer some benefits on ~7B reward models, the improvement appears marginal, and it’s unclear whether this design is efficient or sustainable for larger or more complex settings.
    
**References**  
[1] Zhang et al., _β-DPO: Direct Preference Optimization with Dynamic β_, arXiv:2407.08639 (2024).  
[2] Lu et al., _DAMA: Data- and Model-Aware Alignment for LLMs_, arXiv:2502.01943 (2025).

### Questions
1. The IRL part only introduces one template and treats the model-generated response ( $\tilde{y}_t$ ) as non-preferred, assuming it is always worse than the demonstration. This may lead to collapse toward a narrow response region. Have the authors measured whether the trained reward models limit entropy or diversity?
    
2. Could you provide curves showing how $\alpha^D$  and $\alpha^M$ evolve over epochs or steps? Are there cases where large $\alpha$  values cause unstable updates or divergence?

3. I am curious whether the advantage of using seven separate reward models would still hold if larger reward models (e.g., 30B or 70B) were used as reported in Table 2?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Dynamic Reward Inverse Reinforcement Learning (DR-IRL), an approach to align large language models by combining inverse reinforcement learning with dynamic reward scaling. It trains category-specific reward models from a balanced dataset and adjusts rewards based on data hardness and model responsiveness. This allows the model to focus on difficult, high-risk cases while maintaining stability. Experiments on Llama-3.1-8B and Qwen-2-7B show that DR-IRL improves both safety and helpfulness over existing methods.

### Strengths
DR-IRL achieves strong performance across a wide range of safety and helpfulness benchmarks, outperforming several state-of-the-art alignment methods.

### Weaknesses
I have some concerns that I believe could be addressed to further strengthen the work:

1. Figure 1 is not very clear to me. A more detailed and visually clear version could help readers better understand the main idea of the paper.
2. The experiments presented appear to be comprehensive; however, the novelty of the paper remains somewhat unclear. The concept of dynamic reward modeling has been explored in several previous works [1–4], and it would be helpful if the authors could better highlight what specifically differentiates their approach.
3. My concern about the proposed method, where seven reward models are fine-tuned for use during alignment preference optimization, is that while the exploration of efficiency between single and multiple reward models is appreciated, efficiency comparisons with GRPO, where an LLM-as-Judge approach could be applied, remain underexplored and might provide valuable insights.
4. Another consideration is the efficiency of the proposed approach. During alignment optimization, it appears that seven reward models must be loaded either in parallel or sequentially, which may result in significant computational costs compared with other methods such as DPO or vanilla GRPO. Clarification on this point would be appreciated.
5. The paper considers seven criteria for reward modeling. I wonder whether all of them are necessary. An ablation study examining the effectiveness of each metric could provide valuable clarification, especially in relation to the existing ablation study on hardness coefficients.
6. I sincerely appreciate the authors’ interesting work; however, after reviewing the implementation of the reward model, I found it difficult to distinguish between the Inverse Reinforcement Learning technique employed and standard Supervised Fine-Tuning. Additional explanation in this regard would be very helpful.
7. Some concepts could benefit from further elaboration, such as the demonstration dataset, the Chain-of-Draft mechanism, and its distinction from Chain-of-Thought. Providing more context here would enhance the reader’s understanding.
8. Hyperparameter tuning is a crucial component of model alignment. I would appreciate it if the authors could clarify how the baseline models were optimized and elaborate on the reward modeling approach used for the GRPO baseline.

Typo error:
"chain-of-thought (CoD)" – line 99.

---
**References**

[1] GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy (https://arxiv.org/pdf/2508.04349v1)

[2] Understanding R1-Zero-Like Training: A Critical Perspective (https://arxiv.org/pdf/2503.20783v1)

[3] GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning in Language Models (https://arxiv.org/pdf/2504.09696v1)

[4] DRA-GRPO: Exploring Diversity-Aware Reward Adjustment for R1-Zero-Like Training of Large Language Models (https://arxiv.org/pdf/2505.09655)

### Questions
Please read the weaknesses section.

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
3

### Summary
The paper proposes DR-IRL, a two-fold method which (i) trains category-specific shadow reward models via IRL on a balanced safety dataset spanning seven harmful categories, and (ii) plugs those rewards into GRPO with a dynamic scaling of advantages using task difficulty (text-encoder cosine similarity) and model responsiveness (reward gaps). Experiments on Llama-3.1-8B and Qwen-2-7B report strong gains on safety benchmarks without hurting helpfulness.

### Strengths
+ Clear per-category reward story via IRL, not just a single monolithic reward.
+ The two-fold scaling (data hardness + model responsiveness) is simple and plugs cleanly into GRPO; the paper spells out the gating math.
+ Results look strong and consistent across both model families and many benchmarks; also decent ablation on multiplicative vs additive combination.
+ Effort to hold backbone/KL/sampling/compute constant across methods is appreciated.

### Weaknesses
- Data balancing vs. IRL feels loosely coupled. The paper first constructs a balanced, LLM-generated CoD demo set, then learns per-category reward models and finally applies dynamic scaling. The ablations do not isolate how much of the gain comes purely from balancing versus (i) category-specific rewards or (ii) the α-gating. A “same pipeline but imbalanced data” ablation (plus cross-dataset generalization) would help.
- Novelty feels incremental. IRL-from-demos + GRPO + difficulty weighting are known ingredients; here they’re combined neatly for safety alignment, but there isn’t a single conceptual jump that’s clearly new. The work reads more like a well-engineered package than a fresh idea.
- Synthetic CoD demonstrations: since the demos are LLM-generated from DnA and Safety-Prompts sources, reward models might overfit template patterns or style, and it’s unclear how they behave on non-templated harmful queries. Some stress-tests exist, but a cross-source test (train on CoD, evaluate on a held-out organic/human set) would address this.
- Sensitivity/robustness details are thin. The paper defines masking/thresholding in the responsiveness term (τ, T), but a sensitivity sweep for these knobs isn't covered. Some stability plots or failure cases would increase confidence.

### Questions
Apart from my concerns in the weaknesses:
- How is the category assigned? Is j inherited from Do-Not-Answer / Safety-Prompts labels and CoD templates, or do you predict it? If inherited, how would DR-IRL handle prompts without a known category (e.g., organic user traffic)?
- Have you tried using the per-category reward models at inference (e.g., reward-guided decoding or reranking) to prefer safe-but-helpful candidates? If not, can you comment on feasibility and expected trade-offs (latency, safety vs. utility)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes DR-IRL, a new alignment algorithm to address two key challenges in LLM safety: imbalanced safety datasets and static reward models that ignore task difficulty .

The method has two core components:

Inverse Reinforcement Learning (IRL) with Per-Category RMs: Instead of using preference pairs, the authors first build a balanced safety dataset of demonstrations covering 7 harmful categories . They then use IRL to train 7 separate "shadow reward models," one for each category .

Dynamic Reward Scaling (DR): The paper enhances the GRPO (Group Relative Policy Optimization) algorithm by introducing a dynamic scaling coefficient, α, which re-weights the advantage based on "task difficulty" . This difficulty α is a multiplicative combination of two heuristics :

Data-level hardness (α_D): Measured by the text-encoder cosine similarity between demonstration and generated responses .

Model-level responsiveness (α_M): Measured by the reward gap produced by the shadow reward models .

The authors demonstrate empirically on Llama-3.1-8B and Qwen-2-7B that DR-IRL achieves state-of-the-art results on safety benchmarks (like StrongReject) while maintaining or improving performance on utility benchmarks (like GSM8k).

### Strengths
Strong Empirical Safety Performance: The primary strength is that the final method, DR-IRL, works very well. It achieves SOTA results on key safety benchmarks like StrongReject, XsTest, and WildChat, while also improving or maintaining utility on benchmarks like GSM8k and AdvGLUE.

Good Component-Level Ablations: The paper does a good job ablating the internal components of its own method. Figure 3 shows that both the α_D (data-level) and α_M (model-level) heuristics contribute to the final performance. Table 2 provides a cost-benefit analysis for using 7 per-category RMs vs. a single one.

Novel Heuristic: The core idea of scaling rewards based on both data similarity and model responsiveness is a novel and interesting, if complex, heuristic .

### Weaknesses
Confounded Main Ablation: This is the most significant weakness. The performance jump shown in Table 1 between the GRPO baseline and the IRL/DR-IRL methods cannot be attributed to the method itself, as the training dataset was also changed. The paper does not provide a baseline of GRPO trained on its new, balanced CoD dataset, making it impossible to isolate the true effect of the proposed algorithm.

Extreme Methodological Complexity: The α_D (data hardness) calculation is extraordinarily complex, involving LLM-based sub-sentencing, all-pairs embeddings, max-pooling, and a double-sigmoid-normalization ratio . This "kitchen sink" approach is unprincipled and lacks justification over simpler heuristics (e.g., simple cosine similarity of the full responses).

Weak Theoretical Grounding: The paper is entirely heuristic-driven. The "IRL" component is borrowed from Li et al. (2024)  and used as a black-box RM-training step, discarding the bilevel optimization framework that was its main theoretical point. There is no theoretical justification for the dynamic scaling formula, its multiplicative combination, or its interaction with the GRPO objective.

### Questions
The primary weakness is the confounded comparison in Table 1. To isolate your contribution, could you provide results for a GRPO baseline trained on your new, balanced CoD dataset? This is the only way to know if the gains come from your DR-IRL algorithm or from the new dataset you created.

The α_D calculation  is extremely complex. Have you compared this to a much simpler heuristic, such as α_D = 1 - cos(Φ(o_demo), Φ(o_gen)) (i.e., the similarity between the full demonstration and generated response)? Is this massive complexity truly necessary for the performance gains?

The paper uses 7 per-category RMs. Table 2 compares this to a single RM on a different dataset. A more direct ablation is needed: What is the performance of a single RM trained via IRL on your entire balanced CoD dataset, compared to your 7-RM approach? This would isolate the benefit of "per-category" specialization.

### Soundness
2

### Presentation
2

### Contribution
2
