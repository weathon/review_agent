# BOLT: Decision‑Aligned Distillation and Budget-Aware Routing for Constrained Multimodal QA on Robots

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Robotic systems can require multimodal reasoning under stringent constraints of latency, memory, and energy. Standard instruction tuning and token-level distillation fail to deliver decision quality, reliability, and interpretability under these constraints. We introduce BOLT, a decision-aligned distillation and budget-aware routing framework that treats multi-choice prediction as a decision surface to be aligned during training and selectively refined at inference. During training, BOLT introduces Option-level Decision Distillation to align student models directly on the decision surface of multi-choice answers, thereby eliminating prompt artifacts, improving calibration, and optimizing the exact output space. At inference, BOLT activates Budget-aware Test-time Augmentation, a calibrated router that uses low-cost signals such as confidence, margin, entropy, retrieval affinity, and agreement across short question decompositions to trigger high-resolution reevaluation, type-matched retrieval exemplars, or question decomposition only when their expected benefit outweighs cost. On Robo2VLM-1, a 2B BOLT student distilled from LLaVA-1.5-13B improves accuracy from 28.66 in zero-shot to 42.89 with decision distillation and to 50.50 with budgeted routing, surpassing the 13B teacher at 36.74. It lowers expected calibration error, strengthens the risk-coverage frontier, and slashes GPU memory from 26,878 MB for the teacher to 3,035 MB for the distilled student, and 3,817 MB with all augmentations enabled. By constraining outputs to valid options while exposing retrieved evidence and decomposition traces, BOLT reduces hallucination and provides transparent decision-making, enabling large-model quality on edge robots.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a framework (BOLT) for resource-constrained multimodal question answering in robotics settings with multiple-choice outputs. The framework combines a knowledge distillation component (ODD) that trains a student VLM to match a teacher's distribution over answer spans rather than individual tokens, and a routing component (bTTA) that selectively triggers compute-expensive modules like high-resolution re-evaluation, retrieval-augmented generation, or question decomposition based on heuristic signals like confidence, entropy, and retrieval affinity.

Experiments on Robo2VLM-1 show a 2B student distilled from LLaVA-1.5-13B achieves 42.89% accuracy (vs 36.74% teacher, 28.66% zero-shot) with just distillation, and 50.50% when the router is enabled. GPU memory drops from 26.9GB to 3.8GB. The paper also reports calibration improvements (ECE, AURC) and reduced invalid outputs.

### Strengths
- **Well-motivated problem**: The mismatch between token-level distillation and constrained decoding is real, and the paper makes a reasonable case for option-level alignment.

- **Practical system**: The combination of distillation and budgeted routing addresses a real deployment scenario. The router framework is sensible and the budgeted formulation is clear.

- **Solid empirical gains**: The 2B student outperforms the 13B teacher on Robo2VLM-1. The memory reduction (88.7%) makes a good case for edge deployment.

- **Comprehensive evaluation**: The paper reports accuracy, calibration metrics (ECE, AURC, Brier), memory, latency, energy, and hallucination proxies. The ablations for the distillation method are thorough (though this is not the case for the routing method).

- **Interpretability angle**: The authors make a good case that the components of their system (constraining outputs to valid options, exposing retrieved exemplars, and showing decomposition traces) can help with transparency.

### Weaknesses
- **Single-dataset evaluation**: All experiments are on Robo2VLM-1. Since the experiments are on static datasets rather than real robots, it should be straightforward to evaluate on multiple robotic VQA benchmarks to demonstrate generalization. The dataset is quite specific (panel layouts, constrained options), and it's unclear how well the approach transfers. Other robotic VQA datasets exist that could validate the claims.

- **Router has too many moving parts without sufficient analysis**: The router uses 5 features (confidence, margin, entropy, retrieval affinity, decomposition agreement) and triggers 3 augmentations (HR, tmRAG, QD), but there's no ablation isolating the contribution of individual features. It's unclear which signals actually matter. Some components like type-matched retrieval and question decomposition seem engineered specifically for this benchmark (panel-based tasks with available exemplars and decomposable questions) and may not apply to other robotics tasks where exemplars are unavailable or questions don't naturally decompose. This makes the method's practical applicability questionable.

- **Limited teacher/student diversity**: The student is always Qwen2-VL-2B. Would the gains hold with different student architectures? The teachers are all instruction-tuned VLMs, but no ablation on teacher quality or family is shown.

- **Option-level distillation not deeply novel**: Matching distributions over discrete outputs is fairly standard. The contribution here is applying it to VLMs in a specific constrained setting, but the technical novelty is incremental. The length-bias correction (Eq. 5) is mentioned but not thoroughly analyzed.

- **Hallucination analysis is limited**: The proxies (IOR, NOA, Flip, etc.) are somewhat ad-hoc. The retrieval contradiction rate (RCR) is 21.73%, which is high and not addressed beyond acknowledging it. The paper claims constrained decoding eliminates hallucinations, but the NOA misuse and RCR suggest decision-level hallucinations remain.

- **Energy/latency analysis is incomplete**: Table 7 gives energy per query, but no comparison to the teacher's energy or analysis of the energy-accuracy tradeoff. The latency breakdown (Table 10b) shows bTTA adds 1.52s on top of 7.45s, but no comparison to teacher latency or discussion of real-time requirements for robotics.

- **Writing and presentation**: The paper is dense and tries to cover a lot. Some sections (e.g., Section 3.2.2 on bTTA) pack too much into a small space. The notation is heavy and could be simplified.

### Questions
1. **Multi-dataset evaluation**: Can you evaluate on additional robotic VQA benchmarks beyond Robo2VLM-1? This is critical for validating that your approach generalizes and isn't overfit to this specific panel-based dataset format.

2. **Router feature ablations**: Which of the 5 routing features actually matter? Can you provide ablations showing the contribution of each feature individually (confidence, margin, entropy, retrieval affinity, decomposition agreement)? Are all of them necessary or would a simpler router work?

3. **Applicability beyond this benchmark**: How would your method work on robotic tasks where type-matched exemplars are not available or questions don't naturally decompose? What components of bTTA are truly general vs specific to this evaluation setup?

4. **Retrieval contradiction rate**: 21.73% RCR is concerning and suggests retrieval often hurts rather than helps. Can you characterize when retrieval contradicts the correct answer and propose mitigations?

### Soundness
4

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
The paper introduces two novel methods to improve performance on multi-choice Q&A applied in the robotics domain, where latency, memory and energy efficiency are critical. The first method, Option-level Decision Distillation, provides a method to align a student model with a teacher model based on a normalized the score of the full answer segment rather than just the answer token, and budget test time augmentation decides on a test time budget for a sample based on entropy like measures of the option distribution. The paper shows the effectiveness of the proposed methods on the Robo2VLM-1 benchmark, where a 2b model with both ODD and bTTA surpass a 13b teacher model performance.

### Strengths
The idea of the bTTA is interesting, and it makes intuitive sense that the test time compute should depend on the initial properties of the answer distribution (such as max probability and entropy), in a resource limited setting. 
I am also not aware of other methods looking at the combined score of the assistant answer rather than just the answer token when doing distillation. It's also impressive that this approach improves on the performance of using token level distillation.

### Weaknesses
The abstract, and more broadly the paper in general, reads in a quite convoluted way, with some unclear terms such as "multi-choice prediction as a decision surface to be aligned", two main ideas that are essentially independent (bTTA and ODD), and a lot of abbreviations: QD, ODD, BOLT, HR, bTTA, KD, tmRAG, VQA. 

A major weakness is that while the proposed method significantly improves performance on the Robo2VLM-1 benchmark, the paper does not compare the method with other distillation methods, hence it's not clear that this approach improves upon prior ways of doing distillation. 

Another weakness is that the method is only applied to one dataset, which limits the understanding of the general usefulness of the method.

### Questions
If the answer options are A,B,C,D, and E, why do we need this score formulation (Eq 1), as the answer would only be one token. If this represent a reasoning trace, do we then for each question need to do rollouts with the teacher model until we get coverage on all answer options? 

Is the method robotics specific or even multi-modal specific or can the same method also be used for standard mutli-choice Q&A tasks where latency/model size is important?

In Table 2, Llava 1.5-7b has worse performance than Qwen 2 VL-2B on the zero shot eval. Why is then the Llava model used as a teacher model when it's worse than the student model, and why does this improve the performance?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces BOLT, a decision-aligned training and inference framework that enables small multimodal models to achieve large-model reasoning performance under tight computational budgets.
(1) They proposed Option-Level Decision Distillation (ODD), a new distillation objective that aligns the student model with the teacher’s option-level decision boundaries, rather than token-level or sequence-level supervision. This directly optimizes the prediction space that determines final correctness and avoids prompt and chain-of-thought artifacts common in conventional distillation.
(2) They developed Budget-Aware Test-Time Augmentation (bTTA), a lightweight routing mechanism that dynamically allocates additional compute only when necessary, selectively invoking higher-resolution perception, retrieval exemplars, or question decomposition based on low-cost uncertainty and affinity signals.
(3) The results show that BOLT enables a 2B-parameter model to match or surpass a 13B-parameter teacher on multimodal multiple-choice reasoning benchmarks while operating within strict compute constraints relevant to robotic platforms. Together, these results demonstrate that decision-aligned supervision and adaptive inference provide a scalable path to high-accuracy multimodal reasoning in resource-limited settings.

### Strengths
The paper introduces novel distillation objective. The proposed Option-Level Decision Distillation (ODD) directly aligns the student model with the teacher’s decision boundaries rather than token-level generation trajectories. This is a conceptually novel and clean setup, since multimodal multiple-choice QA depends only on final option-level classification, not on the internal chain-of-thought sequence.

The budget-aware test-time augmentation (bTTA) module provides a principled mechanism for dynamic compute allocation, selectively activating high-cost inference steps only when necessary. This is both practically beneficial and generally applicable to real-world deployment where compute and power constraints vary.

The paper shows strong experimental results. A 2B-parameter student model distilled with ODD and equipped with bTTA matches or outperforms a 13B teacher on multimodal reasoning tasks under constrained compute. These improvements are consistent across multiple datasets and inference conditions, giving credibility to the claims.

### Weaknesses
The paper provides some empirical success, but has limited novelty in the general domain of knowledge distillation. For example, the option-level decision distillation (ODD) only applies to some domain-specific tasks. Besides the proposed distillation objective, the innovation is rather limited.

### Questions
Are we training student model using on-policy distillation? Have we compared the effectiveness between supervised distillation and on-policy distillation?

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
2

### Summary
The paper considers the task of robotic VQA, which is characterized by strict latency, memory, and energy constraints and therefore makes direct deployment of state-of-the-art VLMs infeasible. The paper therefore proposes a two-pronged approach: option-level decision distillation (ODD) and budget-aware test-time augmentation (bTTA). Unlike token knowledge distillation, ODD operates on the sums of answer-token log-probabilities. bTTA performs test-time routing based on cheap statistics derived from the student and optionally performs retrieval, question decomposition, and high-resolution reevaluation to augment the student’s response. The paper empirically shows that the distilled student outperforms the teacher’s zero-shot accuracy by almost 15% while having a 7x smaller memory footprint.

### Strengths
The proposed approach is original, simple (in a good way), and sensible, and the claims are validated by empirical evidence. The results are convincing, as the student outperforms the teacher by a large margin at a fraction of the memory footprint. The paper conducts a series of ablations, showing that each component of the proposed method is relevant. The paper is well-written and easy to follow.

I am not familiar with the literature on robotics VQA at all, so I cannot speak to the significance of this paper’s contribution.

### Weaknesses
It is not quite clear to me how relevant the task of VQA for robotics is. From my limited understanding, it does not appear to be very relevant for real-world applications (i.e., how realistic is it to have a large-scale dataset of options?), as evidenced by the lack of datasets for this task (see Limitations).

It is also not quite clear to me how realistic it is to be able to cache the teacher distributions for all training items offline in a real-world setting (L264).

On L61, “optimization” should be replaced by “optimizing”.

### Questions
* What is the teacher’s latency?
* L267 states that the student’s option distribution is normalized “in the same way” as the teacher's, but Equation 3 does not have the temperature parameter that is present in Equation 2.
* Why is the sum of answer-token log-probabilities robust to benign tokenization changes for fixed option strings (L262)?
* When would option strings differ in length between students and teachers (L263)?

### Soundness
3

### Presentation
3

### Contribution
3
