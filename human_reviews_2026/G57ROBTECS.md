# Propagating Knowledge in LLMs with Hypernetworks

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
Knowledge editing techniques for large language models (LLMs) can inject knowledge that is later reproducible verbatim, but fall short on propagating that knowledge; that is, LLMs can’t answer questions that involve reasoning about it. In this paper, we study hypernetwork-based knowledge editing techniques (i.e., MEND (Mitchell et al., 2022)) for knowledge propagation. We find that vanilla hypernetwork-based editing methods do not effectively propagate knowledge. We propose a simple fix to optimize hypernetworks for knowledge propagation, which is to explicitly include propagation questions as the objective during hypernetwork training. This achieves a substantial performance gain in the RippleEdit dataset, almost 2×accuracy on challenging multi-hop questions whose answer strings do not appear in the injected fact. We further introduce a new synthetic dataset, Controlled RippleEdit, that isolates a confounding factor in knowledge propagation evaluation and further supports evaluating the generalization of knowledge propagation. Our approach outperforms all other approaches for knowledge propagation, including more computationally intensive methods such as continued fine-tuning on synthetic data. Hypernetworks demonstrate some scaling to multi-edit settings (up to 20 edits), achieving performance on par with or higher than CPT-based approaches. Yet, we observe significant limitations in the performance for out-of-domain propagation, suggesting future work in propagating knowledge to a wide range of relations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the challenge of an edited model to reason about and apply newly injected facts beyond verbatim recall. While prior knowledge editing techniques (MEND, MEMIT, ROME) enable efficient and localized parameter updates via hypernetworks, they largely fail to propagate knowledge to related, multi-hop inferences. The authors propose a simple but effective modification to hypernetwork-based editing: they train the hypernetwork not only on direct recall of injected facts but also on propagation questions that require reasoning over those facts. This change directly optimizes the hypernetwork to modify gradients for propagation, rather than memorization.

Empirically, the paper shows that this objective improves propagation performance on the RippleEdit benchmark. To further analyze generalization, the authors introduce Controlled RippleEdit, a synthetic dataset that isolates confounding factors such as entity familiarity and relation overlap. Their results demonstrate improvements in the in-domain setting, though the OOD setting remains challenging (performance drops from 76.7% to 18.3%). Finally, they extend their training objective to multi-edit scenarios using RLEdit, showing that their propagation-aware hypernetwork scales comparably or better than continual-pretraining baselines.

### Strengths
The paper addresses an important and timely problem in the field of knowledge editing: the limited ability of edited LLMs to propagate newly injected knowledge to related reasoning or multi-hop questions. This issue represents one of the most pressing limitations of current editing methods and has direct implications for maintaining up-to-date, reliable LLMs without full retraining. Tackling this challenge is therefore of high practical and scientific significance, and the paper makes clear progress toward understanding and mitigating it.

In terms of clarity, the paper is very well structured and accessible. The motivation is clearly articulated, the methodology is logically presented, and the visualizations are thoughtfully designed to convey the mechanism of propagation-aware editing. These choices make the technical contributions easy to follow.

### Weaknesses
While the paper tackles an important problem, it has several notable weaknesses that limit the strength and generality of its conclusions. 

The most significant limitation of the paper is a conceptual flaw in its experimental setup that fundamentally undermines the main claim of improved knowledge propagation. The proposed approach explicitly incorporates the multi-hop propagation questions (which are supposed to serve as the evaluation set) into the hypernetwork’s training objective. This design leads to substantial data leakage, as the model is effectively trained on the same questions it is later tested on. Consequently, the reported improvements cannot be interpreted as evidence of genuine reasoning or propagation ability, but rather as a by-product of exposure to the test items during training. This issue becomes even clearer in the results reported in Table 2, where the authors evaluate their method in what they term an “out-of-domain” setting. However, this setup does not constitute a true OOD scenario, but instead, it corresponds to a standard unseen-test configuration, in which the evaluation examples are simply withheld from training. Once this overlap between training and evaluation is removed, the performance of the proposed method drops sharply, revealing that the earlier gains stem largely from data leakage rather than from an improved ability to propagate knowledge. 

The paper’s experimental scope is narrow, relying on a single dataset (RippleEdit and its controlled variant) and a single, relatively small base model that is not instruction-tuned. Evaluating only one model configuration prevents a robust assessment of whether the proposed propagation-aware hypernetwork scales to larger or more realistic setups (e.g., 4–8 B parameter instruct models commonly used in editing studies). 

The choice of baselines could be improved: the continued pretraining comparison is outdated, given that recent work shows parameter-efficient fine-tuning combined with model merging offers far stronger and more stable editing performance [1]. 

The interpretation of the results in Table 1 is inconsistent. The caption claims improvements in both verbatim and non-verbatim cases, yet the method underperforms Active-Reading CPT and CPT (Full) in the verbatim setting and is only on par with prepend in the non-verbatim case. More generally, the empirical results are not uniformly convincing: across Tables 1 and 2, the proposed method outperforms baselines only in select configurations, making it difficult to support the broad claim of a generalizable improvement in knowledge propagation (especially given that all results stem from only a single model). 

While Section 2.3 mentions scalability in terms of multiple simultaneous edits, it omits discussion of the lifelong or sequential editing scenario, which is more practically relevant and actively explored in recent work such as WISE [2] and WikiBigEdit [3]. The authors should at least mention this line of work here.

[1] Roth et al. (2024), A Practitioner’s Guide to Continual Multimodal Pretraining

[2] Wang et al. (2024), WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models

[3] Thede et al. (2025), WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs

### Questions
1) How do the authors ensure that the reported improvements are not simply due to data leakage or memorization of test items?
2) The paper refers to the split in Table 2 as “out-of-domain,” but it appears to correspond to a standard unseen-test configuration rather than a true OOD scenario. Could the authors elaborate on how this split was constructed and whether any relations or entities overlap between training and evaluation?
3) Do the authors plan to validate their approach on larger or instruction-tuned models (e.g., 4–8 B parameter range) and on additional benchmarks that capture different types of propagation? This would help assess the method’s generality beyond a single dataset and model.
4) Why was naive continued pretraining chosen as the main baseline instead of more competitive and parameter-efficient alternatives, such as fine-tuning plus model merging?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The proposal addresses a limitation of LLM editing: poor knowledge propagation after factual edits. Building on MEND, the authors propose a propagation-aware hypernetwork that learns edits for locality (updating a fact) and also for reasoning consistency on related facts. The approach modifies MEND’s meta-training objective—rather than training the hypernetwork to reproduce paraphrases. The model takes the gradient of a factual edit, processes it through the hypernetwork to generate a low-rank update, and applies this to the base model.

Experiments on RippleEdit and the proposed Controlled RippleEdit dataset show better generalization and multi-hop reasoning compared to ROME, MEMIT, MEND, and CPT. The idea is conceptually clean, empirically supported, and practically relevant, though limitations remain in generalization and analysis depth.

### Strengths
The paper identifies a real and concrete failure mode in current editing methods — factual edits that do not generalize to reasoning queries. This is not a contrived setting; it mirrors issues observed in ROME and MEMIT where edits fix a statement but fail to propagate logically. The authors’ framing of “propagation” as an optimization target is clearly motivated.

The proposed modification of MEND is meaningful without major lift: replacing paraphrase-based meta-objectives with reasoning-oriented loss functions. The hypernetwork’s gradient-to-weight mapping is trained end-to-end to encourage edits that also benefit multi-hop inference.

The evaluation propagation across single-hop, multi-hop, and out-of-domain splits with  baseline set (ROME, MEMIT, MEND, CPT) covers the key spectrum with clear metrics and ablations is well documented. Beyond benchmarks, this work hints at a general principle with real potential for continual knowledge maintenance in production-scale LLMs.

### Weaknesses
The claims on Controlled RippleEdit are good, It’s unclear whether the same gains would hold in more natural open-domain QA or reasoning settings where there isn't a simplified reasoning path entity -> relation -> derived fact.

The method is only tested on small models (~1B). The hypernetwork’s compute overhead and stability when editing larger-scale LLMs (7B–70B) are not discussed. Since hypernetworks are known to scale poorly with model dimension, I would be curious to hear more from the authors regarding this.

While the method empirically improves reasoning accuracy, the paper does not analyze where or how propagation happens inside the model (e.g., which layers change, how edits flow through attention patterns). This understanding could strengthen the paper further making it more explanatory and empirical.

### Questions
Some follow up questions, 
- Can you quantify the runtime and memory overhead of hypernetwork-based updates?
- Can you provide analysis of which layers absorb the propagated knowledge?
- How sensitive is performance to the balance between locality and propagation loss weights?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses a critical and recognized gap in knowledge editing: ensuring that injected facts are not just memorized but can also be used for reasoning (i.e., knowledge propagation). The authors' core proposal is an intuitive fix to hypernetwork-based editors (like MEND): they align the meta-training objective with this goal by optimizing the hypernetwork to solve propagation questions directly, rather than simple paraphrases. A key contribution is the introduction of a new synthetic benchmark, 'Controlled RippleEdit,' which is designed to evaluate multi-hop reasoning and out-of-domain (OOD) generalization more effectively. The results are promising, especially on their new dataset, but show limitations in multi-edit and OOD settings.

### Strengths
**Clear Problem Definition**: The paper clearly defines and targets a crucial, practical limitation of current knowledge editing methods: their failure to propagate knowledge, where models can parrot a new fact but cannot reason with it. This is a well-motivated and important problem for the field.

**Simple and Effective Solution**: The proposed solution is both simple and effective. By modifying the hypernetwork's outer-loop objective to use propagation questions, the authors align the meta-training process directly with the desired outcome. This intuitive change yields significant performance gains, particularly on the challenging non-verbatim questions in RippleEdit, where it achieves nearly 2x the accuracy of the next-best system.

**Valuable Benchmark Contribution**: A significant contribution is the new 'Controlled RippleEdit' dataset. This benchmark is well-designed, using fictional entities linked to real-world knowledge to create a controlled testbed for multi-hop reasoning . Its inclusion of OOD splits (for both entities and relations) is particularly valuable for assessing whether the editor has learned a generalizable propagation mechanism or is merely overfitting to seen relations

### Weaknesses
1. **Collapse in Multi-Edit Performance**: The method's performance in the multi-edit setting is a major concern. While MEND+Propagation is a strong performer for a single edit (Table 2) , its multi-edit counterpart (RLEdit+Propagation) shows a severe collapse in efficacy, dropping from 76.7 (1 edit) to 48.6 (10 edits) and 29.9 (20 edits). In stark contrast, the Meta-Aug CPT baseline remains highly effective (88.6 for 10 edits, 89.2 for 20 edits). This rapid degradation suggests the hypernetwork approach, when combined with a propagation objective, does not scale sequentially. Why does this approach degrade so much faster than CPT? The paper lacks a sufficient analysis of this critical failure mode.

2. **Comparison to 'Prepend' Baseline**: The claims regarding the RippleEdit results seem overstated. The abstract highlights a 'nearly 2x' accuracy gain , but Table 1 shows that on the key 'Efficacy Non-Verbatim' metric, the proposed MEND+Propagation (22.4) achieves an identical score to the 'Prepend' (in-context learning) baseline (22.4). While this is a significant improvement over other parametric methods like MEMIT (12.7), it also implies that the complex meta-training provides no practical advantage over simple in-context learning on this dataset. While the method does outperform Prepend on the new Controlled RippleEdit dataset (76.7 vs 38.1), the paper should more clearly discuss whether this gap justifies the significant overhead (training, extra parameters, latency) of a hypernetwork versus the zero-cost 'Prepend' baseline.
3. **Underdeveloped Analysis of OOD Failure**: The analysis of the OOD generalization failure is underdeveloped. The paper candidly reports a significant performance drop from in-domain (76.7) to OOD (Both) (18.3). However, it offers little insight into why this failure occurs. The goal of meta-learning a hypernetwork is to learn a general update function. Why does this function fail so completely when encountering unseen relations or entities? Is the network simply overfitting to the specific relations seen during training (e.g., learning a specific mapping from a 'born in' gradient to 'capital of' weights) rather than learning a truly generalizable reasoning mechanism? A more in-depth analysis of these failures would be necessary to understand what the model has actually learned.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies knowledge propagation in hypernetwork-based knowledge editing for LLMs. The authors identify that vanilla MEND does not effectively propagate injected knowledge and propose training the hypernetwork with propagation-oriented objectives. Experiments on the RippleEdit dataset and a newly introduced synthetic benchmark, Controlled RippleEdit, show that the proposed method improves multi-hop knowledge propagation, particularly on non-verbatim cases.

### Strengths
**Clear empirical focus.** Evaluates on RippleEdit and introduces a controlled synthetic benchmark to isolate propagation effects.

**Simple and practical modification.** Adapting hypernetwork training to include propagation supervision is understandable and lightweight, making it potentially attractive for practitioners.

**Improved non-verbatim propagation performance.** Shows strong gains on non-verbatim multi-hop queries where existing methods struggle (e.g., 2× accuracy improvement reported).

### Weaknesses
1. What exactly is “knowledge propagation” in model editing? Although prior works (e.g., RippleEdit) have discussed knowledge propagation, this paper, as an independent contribution, should clearly define such an important and non-common concept in both the introduction and background sections. However, the manuscript only briefly mentions in Line 37 that existing model editing methods struggle to achieve knowledge propagation, without providing further explanation or analysis. This lack of definition and conceptual clarity is confusing.

2. Limited contribution beyond MEND. The method is built on MEND, but the overall contribution appears weak. My understanding is that the authors modify MEND to make it applicable to knowledge propagation tasks. However, it remains unclear whether this modification strategy can generalize to other hypernetwork-based editing methods, as the paper does not discuss or evaluate such cases. Therefore, I have concerns regarding the significance and generality of the contribution.

3. Lack of theoretical support for intuitive design choices. The proposed method involves several intuitive design choices, but lacks theoretical analysis. I am particularly interested in the mechanism underlying why such a simple adjustment can notably improve MEND’s performance on knowledge propagation tasks. Without such analysis, the method feels heuristic rather than principled.

4. Missing recent baselines. Model editing is a rapidly evolving field with many recent competitive baselines, including but not limited to AlphaEdit [1], RLEdit [2], and DAFNet [3]. However, these methods are not included in the experiments. In addition, the newly proposed Controlled RippleEdit dataset requires stronger empirical validation with recent baselines to demonstrate fairness and credibility. Given the current scale of horizontal comparison, I have reservations regarding the effectiveness of the proposed method and dataset.

Reference

[1] Fang J, Jiang H, Wang K, et al. AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models[C]//The Thirteenth International Conference on Learning Representations.

[2] Li Z, Jiang H, Chen H, et al. Reinforced Lifelong Editing for Language Models[C]//Forty-second International Conference on Machine Learning. 

[3] Zhang T, Chen Q, Li D, et al. DAFNet: Dynamic Auxiliary Fusion for Sequential Model Editing in Large Language Models[C]//Findings of the Association for Computational Linguistics ACL 2024. 2024: 1588-1602.

### Questions
Please see weakness

### Soundness
2

### Presentation
2

### Contribution
2
