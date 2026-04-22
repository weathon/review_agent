# Reinforced Latent Reasoning for LLM-based Recommendation

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Large Language Models (LLMs) have demonstrated impressive reasoning capabilities in complex problem-solving tasks, sparking growing interest in their application to preference reasoning in recommendation systems. Existing methods typically rely on fine-tuning with explicit chain-of-thought (CoT) data. However, these methods face significant practical limitations due to (1) the difficulty of obtaining high-quality CoT data in recommendation and (2) the high inference latency caused by generating CoT reasoning. 
In this work, we explore an alternative approach that shifts from explicit CoT reasoning to compact, information-dense latent reasoning. This approach eliminates the need for explicit CoT generation and improves inference efficiency, as few latent tokens can effectively capture the entire reasoning process. 
Building on this idea, we propose \textit{\underline{R}einforced \underline{Latent} \underline{R}easoning for \underline{R}ecommendation} (LatentR$^3$), a novel end-to-end training framework that leverages reinforcement learning (RL) to optimize latent reasoning without relying on any CoT data.
LatentR$^3$ adopts a two-stage training strategy: first, supervised fine-tuning to initialize the latent reasoning module, followed by pure RL training to encourage exploration through a rule-based reward design. Our RL implementation is based on a modified GRPO algorithm, which reduces computational overhead during training and introduces continuous reward signals for more efficient learning.
Extensive experiments demonstrate that LatentR$^3$ enables effective latent reasoning without any direct supervision of the reasoning process, significantly improving performance when integrated with different LLM-based recommendation methods. Our codes are available at \url{https://github.com/xuwenxinedu/R3} .

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Reinforced Latent Reasoning for Recommendation ($\text{LatentR}^3$), an end-to-end framework that enables Large Language Models (LLMs) to perform implicit reasoning for recommendation. The core idea is to replace costly, explicit Chain-of-Thought (CoT) generation with compact latent tokens during inference, thus significantly boosting efficiency and removing the dependency on explicit CoT supervision data. $\text{LatentR}^3$ employs a novel two-stage training scheme that leverages Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL) using a modified GRPO algorithm (LR-GRPO). The results show that $\text{LatentR}^3$ achieves state-of-the-art performance on various Amazon datasets while maintaining low inference latency.

### Strengths
1. The framework successfully addresses a major practical limitation of LLM-based recommenders by eliminating the generation of verbose explicit CoT text. By compressing the reasoning into a few latent tokens, it achieves high performance while maintaining efficiency, which is crucial for real-time deployment.

2. $\text{LatentR}^3$ utilizes a reinforcement learning approach to optimize the latent reasoning process, which allows the model to learn effective reasoning strategies directly from the recommendation outcome without the need for expensive, subjective, and difficult-to-obtain explicit CoT supervision data.

### Weaknesses
1. The experimental validation is restricted in scope. Firstly, it only uses a specific family of datasets (Amazon review data), lacking tests on other popular and structurally different public benchmarks like MovieLens-1M. Secondly, the framework is only implemented and tested on relatively small-scale LLM backbones (e.g., $D^3$ or BIGRec, likely based on BERT or similar models), leaving its scalability and continued effectiveness on large, cutting-edge foundation models (e.g., Llama-7B/13B) unproven.

2. While the LR-GRPO method is technically described, the rationale behind critical design choices, such as why the negative PPL is used as a continuous reward proxy and how the batch-level advantage specifically resolves stability issues in the continuous policy space, could be explained more intuitively for a general audience.

### Questions
1. The paper states that performance improvements are mainly derived from better recommendations for long-tail items. Since the LLM's rich semantic representations already contribute to better long-tail performance in many methods, how can the authors isolate and demonstrate that the observed gain is primarily due to the learned reasoning captured by the latent tokens, rather than simply an enhanced utilization of the existing semantic information?

2. In the recommendation scenario, how does the performance of implicit reasoning (LatentR³) directly compare to the performance of explicit CoT? If there is a performance gap, what is its magnitude? Understanding this gap is crucial for determining whether implicit reasoning is merely a faster, but less accurate, alternative.

3. The paper utilizes negative Perplexity (PPL) as a continuous reward. What results would be achieved if the model were trained using a simpler binary reward (direct ranking reward or a binary reward signal) or another form of reward, and how would this impact the training stability and final recommendation performance?

4. The compared methods are all purely based on LLM representation and are not combined with traditional ID-based recommendation. Can this approach truly outperform methods where LLM representations are used as input features for traditional recommenders? In this context, can implicit reasoning lead to better LLM representations? 

6. Furthermore, is this implicit reasoning scheme applicable to non-textual input, Item Indexing-based recommendation methods, such as 'Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation'?

### Soundness
2

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
5

### Summary
This paper introduces a new framework for applying reasoning capabilities of Large Language Models (LLMs) to recommendation systems. Instead of relying on explicit chain-of-thought (CoT) reasoning — which requires costly CoT data and causes high inference latency — the authors propose a latent reasoning paradigm.

LatentR³ enables the model to reason within a compact latent space rather than generating textual reasoning traces. The framework integrates a new architectural module, LatentRATT, which adds an attention layer on top of the LLM’s decoding layer to produce latent reasoning tokens aligned with the input embedding space.

### Strengths
The paper makes a wise and well-justified design choice. For the sequential recommendation problem, latent reasoning is indeed a more suitable approach. Rather than performing explicit chain-of-thought reasoning, which assumes some human-like logical process, the task here is essentially about learning an approximator that fits the pattern of the next purchased (or interacted) item based on historical data.

In fact, most sequential recommendation problems do not lend themselves to explicit CoT reasoning, because there is rarely a clear, interpretable logic that connects a user’s past behaviors to their next purchase. It is challenging to logically deduce what users will buy next just from previous purchase records. This intuition is also reflected in the results, i.e.,  the numbers in the experimental tables are generally low.

### Weaknesses
1. In the experiments, the authors use Qwen2.5-1.5B as the base LLM and keep it frozen during training, only updating the LatentRATT module. This raises several concerns.
First, given that Qwen2.5-1.5B is a relatively small model and easy to fine-tune, it would be reasonable to jointly train the entire model rather than freezing the backbone. Such joint optimization might lead to better results.
Moreover, an additional baseline should be included, a fully fine-tuned Qwen2.5-1.5B model trained with standard supervised fine-tuning (SFT) without CoT reasoning. As discussed in the strengths, sequential recommendation essentially involves learning an approximator from historical behavior to future patterns, so a fully fine-tuned LLM without CoT might achieve similar effectiveness. Input is the same as yours, and output groundtruth is the item. 

2. According to Table 3, the datasets used in the experiments contain relatively small item pool size. In real-world recommendation systems, the number of items is usually much larger. It remains unclear how well the proposed method would perform on large-scale datasets with hundreds of thousands or millions of items. The scalability and generalization ability in such realistic settings are therefore uncertain.

3. The paper does not discuss how the model handles cold-start scenarios, such as new or unseen items entering the system. Since this is a crucial challenge in recommendation tasks, it would strengthen the paper to provide either a discussion or an experiment addressing it.

4. The Related Work section overlooks several recent papers, such as Rec-R1, which focuses on explicit CoT reasoning for recommendation. The authors should revisit this section and ensure that all relevant recent studies are properly cited and discussed.

> Lin, J., Wang, T., & Qian, K. (2025). Rec-R1: Bridging Generative Large Language Models and User-Centric Recommendation Systems via Reinforcement Learning. Transactions on Machine Learning Research. https://openreview.net/forum?id=YBRU9MV2vE

5. The relative improvement percentages in Figure 2 may not be a fair representation. For unpopular items where the original performance values are already small, even minor absolute gains can appear as large relative improvements. Figure 6 actually shows that the absolute improvements are nearly the same across categories, so using relative percentages can be misleading. It would be better to present both absolute and relative results for a fairer comparison.

6. In the ablation study section, for the variant “w/o LatentRATT”, the paper does not clearly specify the training configuration. It remains ambiguous whether the entire LLM (Qwen2.5-1.5B) is frozen or partially updated during training. Since LatentRATT is removed, clarification is needed about which parameters are optimized in this setting.

7. Inaccurate or vague expressions
* Line 72 (“training RL from scratch”). This phrase is vague. The authors’ intended meaning appears to be “RL without SFT warm-up”, but the current wording could be misread as training the LLM from random initialization using RL. A clearer expression like “RL-only” would avoid this ambiguity.
* Line 79 (“GRPO corresponds to binary reward”) — The statement is not entirely accurate. In general, policy gradient methods can operate on arbitrary scalar rewards. The current phrasing incorrectly suggests GRPO is inherently binary by definition.
* Line 682 (“as shown in Appendix ???”) — There is an apparent LaTeX reference error (a missing \ref target), which should be corrected for completeness and clarity.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
2

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
This paper introduces an SFT-then-RL paradigm to enable latent reasoning in LLM-based recommendation. The main contributions are:
1. An additional attention layer, LatentRATT, that produces latent vectors and improves performance.
2. An improved reward design that computes the advantage by averaging across the mini-batch rather than across multiple rollouts from the same prompt.

Two further reward modifications are proposed, though the experiments do not fully convince me of their effectiveness. Experiments are conducted on four public datasets. Code is available during review.

### Strengths
1. Timely exploration of latent reasoning for recommendation.
2. The LatentRATT layer appears effective for generating latent vectors that improve performance.
3. The advantage computation using batch-level averaging is a clear and reasonable change to GRPO.
4. Extensive experiments on four public datasets.
5. Code is available during the review period.

### Weaknesses
1. Why are LLMs necessary here?
    * The authors state that prior latent-reasoning-for-recommendation works are not LLM-based and that their approach is tailored to LLMs.
    * However, the core techniques, namely LatentRATT and the modified GRPO algorithm, are not inherently tied to language modeling. In principle they could be applied to conventional recommenders such as SASRec. The "reasoning" is encoded in a latent vector of length 1, not in explicit linguistic reasoning.
    * To support the claim that the approach is specifically tailored to LLMs, further discussions and an ablation replacing the LLM backbone with a standard recommender would be important.
2. Limited verification of the GRPO-style modifications.
    * My understanding is that batch-level advantage aims to improve performance, while the sampling change and the PPL-based rewards target efficiency. For the two modifications aimed at efficiency, the paper reports only an overall comparison against vanilla GRPO, without isolating the contribution of each change.
    * If sampling and PPL rewards also affect recommendation performance, not only on efficiency, there should be ablations on ranking metrics. If they primarily affect efficiency, there should be ablations on cost and latency. Neither is shown.
3. Insufficient discussion of concurrent latent-reasoning work. The paper briefly mentions two concurrent papers on latent reasoning for recommendation, but does not clearly compare similarities and differences. A more systematic discussion would help clarify novelty, especially given the question of whether LLMs are needed.
4. Data processing choices may bias results.
    * The datasets are not large, and the authors further subsample to about 5k items. This setting resembles a sparse, cold-start-like scenario that may not reflect production traffic and may favor LLM-based methods relative to conventional baselines, weakening the overall contribution.
    * The paper claims to follow established processing methodology but does not cite explicit references.
5. Metric inconsistency in the main text. Figure 2 reports H@10 and N@10, while Figure 3 reports H@5 and N@5. Although full results appear in the appendix, the inconsistency in the main text can feel like cherry-picking.
6. Related work on latent reasoning in LLMs is thin. Only three papers are cited. A more comprehensive and organized survey of latent reasoning for LLMs would be better.
7. Typos and formatting.
    * Line 388 appears to have an unintended newline.
    * Line 682 has a missing section number ("Appendix ??").
    * Line 689 uses \citet{} where \citep{} seems intended.

### Questions
Please first refer to the "Weaknesses". Two other questions are:

1. Equation (8): Why does the KL divergence become zero? Isn't LatentRATT a part of the policy network $\pi_{\theta}$?
2. Why does scaling the base model from 1.5B to 3B result in much worse performance, as suggested by the cross-comparison between Table 1 and Table 7?

### Soundness
2

### Presentation
4

### Contribution
3
