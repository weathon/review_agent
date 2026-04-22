# SoLoPO: Unlocking Long-Context Capabilities in LLMs via Short-to-Long Preference Optimization

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8

## Abstract
Despite advances in pretraining with extended context sizes, large language models (LLMs) still face challenges in effectively utilizing real-world long-context information, primarily due to insufficient long-context alignment caused by data quality issues, training inefficiencies, and the lack of well-designed optimization objectives. To address these limitations, we propose a framework named **S**h**o**rt-to-**Lo**ng **P**reference **O**ptimization (**SoLoPO**), decoupling long-context preference optimization (PO) into two components: short-context PO and short-to-long reward alignment (SoLo-RA), supported by both theoretical and empirical evidence. Specifically, short-context PO leverages preference pairs sampled from short contexts to enhance the model's contextual knowledge utilization ability. Meanwhile, SoLo-RA explicitly encourages reward score consistency for the responses when conditioned on both short and long contexts that contain identical task-relevant information. This facilitates transferring the model's ability to handle short contexts into long-context scenarios. SoLoPO is compatible with mainstream preference optimization algorithms, while substantially improving the efficiency of data construction and training processes. Experimental results show that SoLoPO enhances all these algorithms with respect to stronger length and domain generalization abilities across various long-context benchmarks, while achieving notable improvements in both computational and memory efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SoLoPO (Short-to-Long Preference Optimization), a framework designed to improve the long-context capabilities of large language models (LLMs) by decoupling long-context preference optimization into two components:
1. Short-context PO: Enhances reasoning on compressed, task-relevant contexts.
2. Short-to-Long Reward Alignment (SoLo-RA): Encourages consistency in reward scores between short and long contexts containing the same task-relevant information.

The authors provide theoretical justification for this decoupling and empirically validate SoLoPO across multiple preference optimization algorithms (DPO, SimPO, ORPO) and models (Qwen2.5, Llama3.1). Key results show that SoLoPO improves performance on long-context benchmarks (e.g., LongBench, RULER, NIAH-Plus) while maintaining short-context capabilities and significantly boosting training efficiency.

### Strengths
1. A theoretical decomposition of long-context PO into short-context PO and SoLo-RA.
2. The idea of decoupling long-context alignment into short-context reasoning and cross-context reward alignment is novel and well-motivated.
3. Offers a practical and scalable solution with clear efficiency gains (e.g., 2.1× longer trainable sequences, 52% runtime reduction).

### Weaknesses
1. The theory relies on the redundancy hypothesis and Assumption 1, which, while empirically supported, may not hold for all long-context tasks (e.g., when all context is relevant).
2. The synthetic dataset construction (mixing relevant and irrelevant documents) is simple but may not reflect real-world long-context complexity.

### Questions
1. How does SoLoPO perform on non-QA long-context tasks such as summarization, multi-turn dialogue, or document-level translation?
2. Can the framework be extended to generation-heavy tasks where both input and output are long? The current focus is on long-input scenarios.

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
3

### Summary
This paper addresses the challenge of long-context alignment and aims to improve context utilization in large language models. The authors propose a theoretically grounded framework that leverages short-context preference optimization to enhance performance in long-context settings. The key idea is to decouple long-context preference optimization into two components: short-context preference optimization and short-to-long reward alignment. This framework, termed SoLoPO, can be applied to existing preference optimization methods such as DPO, SimPO, and ORPO. Experimental results across multiple benchmarks demonstrate the effectiveness of the proposed approach.

### Strengths
1. **Novel framework.** Proposes the SoLoPO framework to transfer short-context preference optimization capabilities to long-context alignment.  
2. **Theoretical foundation.** The method is supported by solid theoretical results that justify the proposed decoupling. 
3. **General applicability.** The framework can be integrated with multiple preference alignment methods, showing consistent improvements across them.  
4. **Strong long-context performance.** The chosen-only SoLoPO variant consistently outperforms standard PO baselines on long-context benchmarks.

### Weaknesses
1. **Degraded short-context performance.** The method shows reduced performance on the short-context Open LLM Leaderboard. Lines 103 and 377 claim that SoLoPO maintains short-context performance, yet Table 4 indicates otherwise. SoLoPO underperforms the PO baseline in 16 out of 24 datasets.  
2. **Limited intuition for theoretical results.** Although the theory is sound, the paper should do a better job providing intuition on *why* the decoupling leads to improved long-context performance (see Question 2).  
3. **Lack of detail in key arguments.** Some claims and statements would be clearer with additional justification or elaboration (see Question 3).

### Questions
1. **Missing baseline.** Why does Table 4 not report the performance of the LongPO baseline?  
2. **Clarification on Figure 4.** What exactly do “efficiency” and “runtime” refer to? It is unclear how SoLoPO improves computational efficiency, as it requires handling both $ x_{\text{short}} $ and $ x_{\text{long}} $, which requires additional time for dataset creation as well as an additional forward pass for every iteration. Could you clarify where the runtime or efficiency advantage arises?  
3. **Clarification on Theorem 1 and the function $ s(\cdot) $.**  
   1. Theorem 1 introduces $ s(|x|) $. It would be helpful to provide intuition for the role of $ s(\cdot) $ and its relationship with $ f $, beyond the theoretical requirement for this assumption.  
   2. Since $ s(\cdot) $ must satisfy the inequality, multiple valid functions may exist. Would a tighter $ s $ lead to a stronger upper bound, and how would this affect empirical behavior?  
   3. It would improve readability to include some intuition, after Theorem 1, on how $ s(\cdot) $ can be derived for a given alignment method. The appendix (I3, I4) provides examples, but referencing them directly in the main text would help readers.  
4. **On the equivalence with standard PO.** Line 201 states that for $ \rho = 100\% $, SoLoPO is equivalent to the original PO. In this case, we have  
   $$
   L_{\text{PO}}(x_{\text{long}}) \le \tfrac{1}{3} L_{\text{SoLoPO}}(x_{\text{short}}),
   $$
   which is an upper bound. Is there theoretical evidence that this bound becomes an equality? If not, have you empirically tested whether optimizing PO or its SoLoPO upper bound leads to equivalent results when $ x_{\text{short}} = x_{\text{long}} $?  

**Minor comments that do not affect rating**
1. In Table 4 (Open LLM Leaderboard), best and second-best values are not bolded or underlined, unlike other tables. Any reason for this inconsistency?  
2. While Assumption 1 appears reasonable, it would strengthen the work to provide empirical evidence that it holds in practice.  
4. What is the base model used in the LongPO reimplementation?  
5. In Figure 1a, the second box should read “short-context preference optimization.”  
6. Line 159 defines $ x_{\text{short}} $ as concatenating $ c_{\text{rel}} $, but the equation shows $ c_{\text{irr}} $. This appears to be a typo.  
7. Section 4.1 would be stronger if it explicitly stated how many scenarios SoLoPO outperforms the corresponding PO baseline, further reinforcing its general applicability.  
8. In Table 4, the result for $ M_{\text{short}}^{\text{SimPO}} $ in the Math column is shown in red, even though it improves over the Instruct model. Please verify the formatting.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces SoLoPO (Short-to-Long Preference Optimization), a general framework for efficiently enhancing long-context reasoning in large language models. The key idea is to decouple long-context preference optimization (PO) into two stages: (1) short-context PO, which focuses on optimizing reasoning and alignment on short, information-dense contexts; and (2) short-to-long reward alignment (SoLo-RA), which aligns reward scores between paired short and long contexts that share essential task-relevant information.
The method can be plugged into existing PO algorithms such as DPO, SimPO, and ORPO. The authors derive a theoretical upper bound showing that the long-context PO objective can be approximated through this two-part decomposition, and propose a “chosen-only” SoLo-RA variant that further improves efficiency. Experiments on reasoning (LongBenchV1/V2, NIAH-Plus) and instruction-following benchmarks (MMLU-Pro, GPQA, BBH) demonstrate that SoLoPO achieves higher long-context performance, better reward–KL efficiency, and shorter training time without harming short-context capabilities.

### Strengths
1.The proposed decoupling framework (short-context PO + reward alignment) is elegant, theoretically grounded, and easy to integrate into existing RLHF/PO pipelines.

2.The theoretical formulation clearly explains how SoLoPO approximates the long-context objective through an upper bound, providing a solid foundation for the method.

3.The “chosen-only” SoLo-RA variant is an insightful practical contribution that reduces instability and significantly cuts training cost while maintaining effectiveness.

4.Extensive experiments cover multiple backbones (Qwen2.5-7B, Llama3.1-8B) and benchmarks, showing consistent gains in both long-context reasoning and efficiency.

5.The method generalizes well across DPO, SimPO, and ORPO, confirming its broad applicability.

6.The paper is well written, conceptually coherent, and supported by detailed ablations and efficiency analyses.

### Weaknesses
1.The paper could provide more qualitative analysis or visualization to show how SoLoPO improves long-context reasoning (e.g., attention heatmaps or retrieved key information patterns).

2.The framework assumes that short contexts can fully preserve essential information; performance may degrade if the summarization or compression is imperfect, which is not deeply discussed.

### Questions
None

### Soundness
3

### Presentation
4

### Contribution
3
