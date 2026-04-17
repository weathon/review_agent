# MAPLE: Masked Adapter Prototype Learning for OOD generalization

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Parameter-efficient fine-tuning with adapters (e.g., LoRA) equips LLMs with task-specific skills. However, utilizing multiple pretrained adapters for out-of-distribution (OOD) generalization remains challenging. Existing techniques for OOD generalization using multiple pretrained LoRAs, route inputs using LoRA representations (prototypes) obtained independently, assuming these representations capture complementary information. However, we observe that for existing methods, in-distribution and OOD routing entropies are often comparable, thus bringing the complementarity assumption into question. We derive the theoretical conditions that could lead to a violation of such assumptions, distilling the cause down to the presence of shared, noisy prototype subspaces. Based on this, we introduce $\textbf{MAPLE (Masked-Adapter Prototype LEarning)}$, a simple learning framework that refines LoRA prototypes by masking the target task’s LoRA during prototype learning. In doing so, it encourages prototypes to discard noisy attributes, which improves routing and strengthens OOD generalization. Extensive experiments on language models of varying size, such as Phi-2 (2.7B) and LLaMA-3 (8B) equipped with heterogeneous pools of pretrained LoRAs, show that MAPLE improves the LoRA representation and thus achieves state-of-the-art performance across multiple benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to improve OOD methods that use multiple pre-trained LoRAs with routing. The authors utilize routing entropy to demonstrate that routing may not provide complementary information across different tasks. Furthermore, they present a theoretical analysis to explain the noisiness in routing and the resulting degradation in OOD performance. To address this issue, the authors propose removing the target-task LoRA from the forward pass during training, positing that this will help reduce spurious, task-specific noise in the learned representations.

### Strengths
This paper addresses an interesting and important problem. I like the comprehensive overview of prior works and their connection to the current study. The results also demonstrate good improvements over previous state-of-the-art methods.

### Weaknesses
From the problem formulation, it is not entirely clear what type of Out-of-Distribution (OOD) setting is used by the authors. Some parts of the paper would benefit from clearer sentences and a more precise problem formulation (see Questions for details). Additionally, certain ideas in the paper could be explained more clearly (see Questions for details). 

The proposed method is based on removing the target-task LoRA from the forward pass during training. The authors posit that this will reduce noisiness; however, this assumption is not supported by prior work or analysis.

### Questions
It would be helpful to begin the methodology section with a brief introduction to the problem and the notations used. For instance, the authors introduce Equation 1 without defining $e$ and $x$, which makes it difficult to understand the purpose of routing entropy. Similarly, the section would benefit from first introducing the problem setting (e.g., multitask learning with LoRA and routing) and clearly explaining what prototypes and routing entropy are before using them in the analysis.

On line 164, authors mention following:

> We start by formalizing our argument that for a set of experts to generalize OOD, it must be accompanied by an increased entropy, relative to ID samples, when applied to OOD samples.

Entropy represents the expected information over a distribution. For a set of experts to generalize to OOD data, the entropy for in-distribution (ID) and out-of-distribution (OOD) samples should ideally be similar. In other words, the expected information for OOD samples should be comparable to that of ID samples. If OOD samples are highly unexpected (i.e., yield much higher information), the model may struggle to generalize. Perhaps the authors are instead referring to routing entropy? Please correct me if this interpretation is inaccurate.

In the line 143 and line 160, authors reason about routing entropy for IID and OOD samples.  

> Ideally, for in-distribution (ID) tasks (i.e., LoRA trained on the input task exists in the LoRA pool), the routing entropy should be lower. While for out-ofdistribution (OOD) tasks, the routing entropy is expected to be higher, reflecting greater uncertainty.

And then explain observed similarity in entropy across ID and OOD samples.

> This could be due to representations encoding certain noisy attributes, which blur inter-adapter distinctions and thus yield high routing uncertainty.

However, similarity in routing entropy could also result from an effective router that performs well on OOD tasks. It would be helpful if the authors provided an intuitive explanation of this aspect.

On line 294, authors mention target-task.

> We hypothesize that this failure stems from including the target-task LoRA in the routing during training. 

However, it is not quite clear what target-task in this setting means.

### Soundness
2

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
2

### Summary
The paper proposes a framework to improve out-of-distribution generalization in prototype-based routing for LLM adapter selection. During prototype learning, the target task’s adapter is masked to prevent the router from learning noisy features. The authors theoretically show that noisy subspaces cause in- and out-of-distribution samples to become indistinguishable. Experimental results demonstrate that the proposed framework improves the performance of two prototype-based routing methods.

### Strengths
- The paper provides a solid theoretical analysis, including proofs, to explain the inadequacy of existing methods.
- The proposed method is conceptually simple and easy to implement.

### Weaknesses
- The paper assumes that the entropy difference between in- and out-of-distribution samples adequately captures OOD generalization. However, this assumption overlooks other well-established metrics that can better capture distributional separability, such as the energy score [1].
- The theoretical analysis is not clearly presented. For example, the meaning of the equation in Definition 1 is unclear, and its connection to the rest of the analysis is not well explained. As another example, it is unclear why theorem one indicates that "ensuring that the routing plan induces a low source and high target entropy is a necessary condition for generalization".
-  There is a gap between the theoretically claims and the empirical method. It is unclear why masking the target task's adapter leads a less noisy representation. Some theoretical or empirical analysis is expected.
- The experimental evaluation is not robust and comprehensive.
  - Only the overall task performance is reported. Routing or OOD detection performance is missing.
  - The reported improvements over the baselines are marginal, and the absence of error bars or confidence intervals makes it difficult to assess statistical significance.
- The approach is narrowly tailored to the prototype-based adapter selection setting, which limits its general applicability.

[1] Liu, Weitang, et al. "Energy-based out-of-distribution detection." Advances in neural information processing systems 33 (2020): 21464-21475.

### Questions
- In Definition 1, is $f$ a classifier or an encoder?
- In the experiments, what examples are considered as ID and what are considered as OOD?

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
2

### Summary
This paper introduces MAPLE (Masked Adapter Prototype LEarning), a method to improve OOD generalization when combining multiple LoRA adapters. It is motivated by observing that the current prototype-based routing strategies often yield similar routing entropy for in-distribution and OOD samples, implying poor distinction between the two. The authors provide a theoretical analysis showing that shared noisy subspaces in LoRA prototypes lead to this problem. MAPLE mitigates it by masking the target adapter during prototype learning, forcing the model to refine prototypes that rely on less noisy signals. Empirical results performance gains (~0.5%) over baselines.

### Strengths
* Clear motivation, backing theory, and straight-forward mitigation algorithm. The paper connects empirical entropy observations to a formal theoretical framework explaining why simple approaches fail.
* Simple and practical. MAPLE’s masking approach is straightforward, easy to integrate, and doesn’t require modifying base LLMs or LoRAs.

### Weaknesses
* The empirical gains appear modest. Compared to other baselines, the improvements are very small, raising the question of whether they might simply result from differences in their initializations?
* The theoretical analysis feels somewhat loose and not fully integrated into a cohesive understanding of the problem.
* The observations seem restricted to text data, limiting the generality of the findings.

### Questions
* Figure 1 appears overly stylized and may not reflect real data. Could the authors share the raw observations and clarify how the plot was generated?
* How sensitive is MAPLE to the selection of top-k?
* Given the relatively small absolute gains, is it possible that MAPLE’s improvements stem from random initialization?

### Soundness
2

### Presentation
3

### Contribution
2
