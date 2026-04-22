# From Pseudo-Balancing to True Specialization: Memory-Aware Routing for Mixture-of-Experts

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 6, 4

## Abstract
Mixture-of-Experts(MoE) efficiently trains large models by using sparse activation to lower costs, selecting a few experts based on data characteristics. For MoE, an unbalanced expert load will lead to routing collapse or increased computational overhead. Existing methods commonly achieve an expert-centered balancing strategy to solve it, prioritizing equal utilization of experts over semantic alignment between tokens and experts.
However, this can lead to a pseudo-balance phenomenon: To ensure expert load balancing, the same input is randomly routed to different experts across training steps instead of the most matching one. It introduces two critical issues: (1) Severe knowledge overlap among experts, resulting in redundant representations and inefficient parameter utilization. (2) Difficulty in forming and stabilizing expert specialization. These issues limit the scalability of models, especially large language models(LLM). 
To address these limitations, we introduce Memory-Aware Routing (MAR), a training-phase approach that enhances existing load-balancing strategies. By equipping each expert with a memory buffer, our method explicitly models their long-term preferences, allowing historical experience to guide routing. This ensures that tokens are routed more consistently to compatible experts, mitigating the pseudo-balance problem while maintaining global load balance and fostering expert specialization.
Experimental results show that Memory-Aware Routing improves expert specialization by 35\% and downstream accuracy by 2\%-25\%, doubles parameter efficiency, and matches baseline performance with only half the experts (one-quarter of the parameters).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on the load balancing issues in MoE training. They claim existing approaches fall under the pseudo-balance phenomenon, where they achieve load balance but at the expense of specialization. The paper proposes a memory-aware routing method which stores a buffer of token representation routed to an expert and uses it as an auxiliary score in routing decisions.

### Strengths
- Clearly written paper 
- Experimented with different MoE models
- Provide an ablation that shows the improvement with proposed approach

### Weaknesses
- Missing baselines to other approaches. Eg auxiliary free approaches, having shared expert approaches from deepseek ( https://arxiv.org/pdf/2401.06066, https://arxiv.org/pdf/2408.15664v1) 
- Has small scale experiments. Having a scaling curve across model sizes would be convincing if this approach scales.

### Questions
- In Appendix A1.1, how is the loss L_balance differentiable? It seems the load is calculated empirically and is not differentiable. 
- Did you follow the approach ST-MoE (https://arxiv.org/abs/2202.08906)  for the baseline? The load balancing loss coefficient is usually 0.01 so as to not dominate original next token prediction loss 
- The perplexity values in Figure 4 are extremely high. Perhaps you need to try with a bigger sized model to have better conclusion.

### Soundness
2

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
3

### Summary
This paper targets the "pseudo-balance" issue in MoE models, where load-balancing strategies cause inconsistent token routing, hindering specialization and wasting parameters. The proposed solution, MAR, uses expert-specific memory buffers to capture long-term preferences, ensuring tokens are consistently routed to semantically aligned experts.

### Strengths
The introduction of MAR is a novel solution that effectively mitigates the pseudo-balance phenomenon by leveraging historical input token information, which promotes stable specialization among experts. The paper provides comprehensive experimental results that demonstrate the effectiveness of MAR, showing improvements in expert specialization and performance across various metrics and tasks.

### Weaknesses
1. The paper could provide more analysis on the memory buffer mechanism. The FIFO (First-In, First-Out) update strategy is simple, but its optimality isn't discussed. A more comparative study with other strategies (e.g., based on token representativeness) would be beneficial.
﻿
2. The experiments are conducted with 8 or 16 experts. It is unclear how MAR would perform in models with a much larger number of experts (e.g., 64 or 128), where preference vectors might become less distinct or the $\mathcal{O}(Kd)$ matching score computation could become a training bottleneck.

3. The paper's experimental validation leans heavily on ablation studies and comparison against a standard baseline (LBL). A comprehensive comparison against other recent methods aiming to improve MoE specialization or routing (e.g., methods beyond simple load balancing) is missing. Including 4-5 such comparative methods would provide a stronger context for MAR's advantages.

### Questions
1. Could you please clarify the KED metric? Does a higher KED value (as achieved by MAR) represent better or more distinct specialization in your experiments (i.e., the model is more dependent on its key, specialized experts)? This would resolve the contradiction with the text in Section 5.1.2.

2. Have you explored alternatives to the FIFO buffer update strategy? For example, a "reservoir sampling" approach or a strategy that preferentially keeps tokens that are "closer" to the current preference vector to reinforce specialization? I suggest adding experiments to demonstrate this part.

3. How does the "pseudo-balance" phenomenon and MAR's effectiveness vary with the number of experts selected (i.e., Top-K)? The experiments seem to use Top-2 exclusively. Would the problem be less severe with Top-1 routing, or more severe with Top-4? I suggest adding ablation experiments for analysis in this section.

4. The standard deviation of expert utilization for MAR (Table 6, Avg: 2.57) is much higher than for standard LBL (Avg: 0.0633). You argue this reflects the "true distribution of the data." Does this slight imbalance (compared to LBL) cause any training instability or capacity issues on more complex, larger-scale data? I suggest conducting experimental verification or theoretical analysis

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
This paper introduces Memory-Aware Routing (MAR), a training-time mechanism for Mixture-of-Experts (MoE) models designed to overcome the pseudo-balance phenomenon. Traditional load-balancing encourages tokens to be randomly routed for uniform expert utilization, leading to knowledge overlap and parameter redundancy. MAR equips each expert with a memory buffer to derive a long-term preference vector from recently processed tokens during the training phase. A calculated Expert-Token Matching Score (i.e., the similarity between the input token and the expert's preference vector) is fused with the original routing logits, encouraging consistent, semantically-aligned routing. This approach mitigates the pseudo-balance issue and improves expert specialization.

### Strengths
The motivation is clear, stemming from the investigation of the pseudo-balance issue in existing load-balanced MoE models. The proposed MAR mechanism is straightforward and effectively reduces parameter redundancy while encouraging specialization, as demonstrated in the experiments.

### Weaknesses
The discussion of established load-balancing losses (e.g., $L_{aux}$, z-loss) is superficial, and a head-to-head comparison against prior load-balancing-specific baselines is lacking. Furthermore, optimal results rely on specific hyperparameters ($\alpha=0.5$, buffer size $128$), suggesting potential fragility and requiring broader validation. See questions below for more details.

### Questions
1. The statement that MAR is a training-only technique with no inference-time modifications is a crucial property but is mentioned late in the paper (Section 4.3). Could the authors explicitly highlight this point in both the Abstract and Introduction to minimize reader confusion?

2. The discussion of related load balancing methods (e.g., $L_{aux}$, z-loss) in the second paragraph of Introduction and Section 2.2 is limited to naming the losses without providing context or their underlying formulations. Could the authors enrich the description of these works and, more importantly, include direct experimental comparisons of MAR against these established load-balancing strategies in terms of both perplexity (PPL) and specialization (KED)?

3. The ablation study demonstrates that performance is sensitive to the chosen hyperparameters ($\alpha=0.5$, buffer size $128$). To better validate MAR's hyperparameter robustness and general applicability, could the authors provide additional experiments demonstrating its effectiveness on diverse MoE architectures, particularly those with mechanisms like shared experts (e.g., DeepSeek-V2) or reasoning/thinking models (e.g., DeepSeek-R1)?

4. The current complexity analysis for the preference vector update is $O(Nd)$. Since the preference vector is the average of all vectors in the FIFO buffer, the update operation can be efficiently implemented in $O(d)$ time by maintaining a running sum and performing additive/subtractive updates when elements are added to or removed from the buffer. Although the performance impact may be limited because the buffer size is small and with GPU parallelism, it is still worth mentioning this optimization. Could the authors describe this more efficient implementation in the Appendix for scenarios where the buffer size might be large?

5. Given the successful demonstration that MAR-trained models can achieve baseline performance with $50\%$ fewer experts, could the authors include a discussion (perhaps in the Future Work section or Appendix) regarding MAR's potential compatibility with and benefits for downstream MoE compression, pruning, or merging techniques?

### Soundness
2

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
This paper investigates a fundamental limitation in current Mixture-of-Experts (MoE) architectures — the pseudo-balancing phenomenon, where existing load-balancing losses distribute tokens evenly among experts but fail to maintain semantic consistency in expert assignments.
The authors propose Memory-Aware Routing (MAR), a novel mechanism that introduces expert memory buffers to preserve historical token representations and derive expert preference vectors. During routing, MAR computes an Expert-Token Matching Score based on the similarity between current inputs and expert memories, which is fused with the standard gating logits to achieve semantically consistent and balanced routing.
Extensive experiments on multiple datasets (PTB, WikiText-2, OpenWebText, GSM8K, MMLU, etc.) demonstrate that MAR significantly improves expert specialization, reduces redundancy, and achieves competitive or superior performance with fewer parameters compared to baselines.

### Strengths
1.The paper introduces a new perspective on MoE training stability by identifying and formalizing the pseudo-balance issue — a phenomenon overlooked in prior work focused solely on token load distribution (e.g., GShard, Switch Transformer).
2.The proposed Memory-Aware Routing (MAR) is conceptually novel, integrating memory-based semantic matching into the routing process without introducing trainable parameters.
3.The authors provide a comprehensive experimental evaluation across model scales and datasets, showing consistent improvements in specialization and generalization.
4.Ablation studies effectively support claims regarding the contribution of memory buffers and matching fusion.
5.Figures and schematic diagrams (e.g., MAR framework) effectively illustrate the routing and memory mechanisms.
6.The paper is clearly written and logically structured, making complex ideas accessible.
7.Addresses a long-standing issue in MoE models that directly affects efficiency and scalability.

### Weaknesses
1. While MAR’s empirical benefits are clear, the paper lacks a formal analysis of why and how memory-guided routing leads to stable specialization. A deeper theoretical justification could strengthen the contribution.
2. Although MAR claims minimal overhead, a quantitative analysis of additional memory cost or routing latency is missing.
3. The paper adopts a simple FIFO buffer and cosine similarity, but alternative formulations are not explored.

### Questions
1. Could the authors provide a theoretical explanation or formal intuition for why the incorporation of expert memory leads to more stable specialization?
2. Although MAR claims minimal overhead, a quantitative analysis of additional memory cost or routing latency is missing.
3. Have the authors considered alternative memory update or similarity mechanisms beyond FIFO and cosine similarity?
4.Could the authors provide detailed ablation studies analyzing the sensitivity of MAR to:
(a) different memory initialization strategies (random vs. data-driven),
(b) buffer size or retention length, and
(c) fusion weight α between base logits and memory matching score?
Such results would help confirm the robustness and general applicability of MAR under varying hyperparameter settings.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the pseudo-balance problem in Mixture-of-Experts (MoE) models, where load-balancing methods ensure equal expert usage but disrupt semantic consistency, preventing true expert specialization. To solve this, the authors propose Memory-Aware Routing (MAR), which equips each expert with a memory buffer that records past token representations to form long-term preference vectors. By combining these preferences with standard routing logits, MAR enables tokens to be consistently assigned to semantically aligned experts while maintaining global balance. Experiments on multiple MoE architectures demonstrate that MAR improves expert specialization and enhances downstream accuracy. Overall, MAR effectively transitions MoE models from pseudo-balancing to true specialization, improving both efficiency and scalability.

### Strengths
1. Preliminary experiments convincingly demonstrate the pseudo-balance phenomenon, showing how load-balancing losses cause oscillations in token routing during training and lead to expert redundancy.

2. The proposed Memory-Aware Routing (MAR) integrates expert-specific memory buffers to capture long-term preferences, enabling more consistent and semantically aligned routing without extra trainable parameters or inference overhead.

3. Extensive experiments across multiple MoE architectures (Mixtral-MoE, LLaMA-MoE, GPT2-MoE, OLMoE) demonstrate significant gains.

### Weaknesses
1. The paper lacks a theoretical explanation for why the load-balancing loss leads to an approximately uniform routing probability across experts. A deeper analytical understanding of this behavior would strengthen the motivation for addressing the pseudo-balance problem.

2. The proposed memory buffer cannot fully capture the semantic information of tokens, and maintaining it introduces additional computational and memory overhead during training, which may affect efficiency at scale.

3. The experimental comparison is limited to standard load-balancing methods, without including other recent routing strategies, making it difficult to assess the relative advantage of MAR in a broader context.

4. The paper does not analyze how MAR performs when scaling to a larger or more sparse set of experts. Its effectiveness in highly sparse scenarios therefore remains uncertain.

5. Some experimental setups lack detailed descriptions.

### Questions
1. Please consider adding additional baseline methods [1,2,3,4] to Table 1, Moreover, it would be valuable to include more discussion comparing [3,4] with the proposed MAR.

2. It would be helpful to include quantitative analyses of training and inference throughput, so that the efficiency impact of maintaining memory buffers can be clearly evaluated.

3. In Table 2, please consider scaling the number of experts to 64 or even 128 to assess whether MAR remains effective under highly sparse or large-scale conditions.

4. The OLMoE results on MMLU and BBH in Table 3 appear lower than the reported numbers in the original paper [5]. Could you clarify the differences in experimental setup or evaluation protocol that led to this discrepancy?

5. Please provide more details about the training configuration, including dataset scale, total training epochs, and compute budget, to improve transparency and reproducibility.

6. In Figures 5, 6, 10, and 11, the variation of expert selection rates across epochs appears limited, and it is difficult to observe how MAR concretely influences expert selection during training. Please consider providing finer-grained analyses (e.g., within the first few thousand steps or the first epoch) or additional visualizations to better demonstrate the effect of MAR on routing dynamics.

[1] Demons in the detail: On implementing load balancing loss for training specialized mixture-of-expert models

[2] Deepseek-v3 technical report

[3] Load Balancing Mixture of Experts with Similarity Preserving Routers

[4] Advancing Expert Specialization for Better MoE

[5] OLMoE: Open Mixture-of-Experts Language Models

### Soundness
2

### Presentation
3

### Contribution
2
