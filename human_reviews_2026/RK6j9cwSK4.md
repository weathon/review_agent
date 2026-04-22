# PhyMix: Towards Physically Consistent Single-Image 3D Indoor Scene Generation with Implicit–Explicit Optimization

- Avg Score: 4.80
- Decision: Reject
- Scores: 2, 6, 6, 8, 2

## Abstract
Existing single-image 3D indoor scene generators often produce results that look visually plausible but fail to obey real-world physics, limiting their reliability in robotics, embodied AI, and design. To examine this gap, we introduce a unified Physics Evaluator that measures four main aspects: contact, stability, geometric priors, and deployability, which are further decomposed into nine sub-constraints, establishing the first benchmark to measure physical consistency. Based on this evaluator, our analysis shows that state-of-the-art methods remain largely physics-unaware. To overcome this limitation, we further propose a framework that integrates feedback from the Physics Evaluator into both training and inference, enhancing the physical plausibility of generated scenes. Specifically, we propose PhyMix, which is composed of two complementary components: (i) implicit alignment via Scene-GRPO, a critic-free group-relative policy optimization that leverages the Physics Evaluator as a preference signal and biases sampling towards physically feasible layouts, and (ii) explicit refinement via a plug-and-play Test-Time Optimizer (TTO) that uses differentiable evaluator signals to correct residual violations during generation. Overall, our method unifies evaluation, reward shaping, and inference-time correction, producing 3D indoor scenes that are both visually faithful and physically plausible. Extensive evaluations on synthetic dataset confirm state-of-the-art performance in both visual fidelity and physical plausibility, and extensive qualitative examples on stylized and real-world images further showcase the method’s robustness.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a so-called physics evaluator for evaluating the quality of image-example based indoor scene synthesis. Based on the proposed evaluator functions, the authors integrate them into a policy learning model and a test-time optimizer for implicit learning and explicit refinement. Experiments show that their proposed method performs better than SOTA models in terms of Chamfer distance, F-Score and FID on the 3D-FRONT test set.

### Strengths
The authors well-analyzed the physics-related aspects such as contact (grounding, collision-free), stability etc among existing models, and pointed the significance of incorporating these essences into establishing a more profound image-to-3D-scene model.

### Weaknesses
1. The paper is very hard to follow. One of the key contribution of the paper, the evaluator metrics, is not given priority in the main paper (L195-L200), but instread appeared in Supp with complicated mathematic definitions. Throughout the paper, too much symbols shown without clear explanations. 
2. Some contents are not described constant throughout the paper. E.g. The input-output interface is not clearly explained or not consistent. In Figure 2, it shows the inputs are the RGB image and instance masks. However, in SubSection5.1, there is no description of how to work on the 3D-FRONT test set. As far as I know, there is no provided such RGB images and instance masks. 
3. There is no comparison between the proposed policy learning model and a test-time optimizer with exisiting models. I'm curious about what is the key essence that improves the model to a great extent. 
4. Visual results are not enough. There are only 4 examples shown in Figure 4. I'd suggest to provide more visual comparison results. 
5. No visual ablation study shown in Supp. 
6. Computational cost is not provided. 
7. Failure cases are suggested to be added. 
8. It would be great to provide some downstream tasks to enrich the experimental setup.

### Questions
See weakness

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes *PhyMix*, a physics-guided framework for single-image 3D indoor scene generation. The authors introduce a unified *Physics Evaluator* consisting of four aspects and nine measurable physical constraints, and integrate its feedback into both training (implicit preference alignment using Scene-GRPO) and inference (explicit Test-Time Optimization). Experiments on 3D-FRONT show consistent improvements in physical plausibility and geometric fidelity, with qualitative results across various image domains.

### Strengths
1. The proposed Physics Evaluator provides a comprehensive and unified measurement of physical consistency, covering contact, stability, geometric priors, and deployability.
2. The combination of implicit optimization (Scene-GRPO) and explicit refinement (TTO) is conceptually elegant and appears effective in improving physical plausibility.
3. The method generalizes to multiple input domains (real, synthetic, cartoon, LLM-generated), showing robustness and practical applicability.
4. The paper is overall well-written and should be easy to follow.

### Weaknesses
1. The training pipeline depends on the Physics Evaluator, and some evaluation components (especially simulation-based stability $P_{sim}$ ) can be computationally expensive. The paper lacks a clear comparison of training/inference time and compute cost relative to baselines.
2. The Physics Evaluator contains many hyperparameters (as discussed in the appendix). It is not clear whether these hyperparameters are object-category dependent, or how sensitive the evaluator is to different scene compositions. More justification on robustness across object types is needed.
3. Qualitative comparisons in the main paper are limited. Given that physical consistency often manifests in motion or interaction, videos could better reflect physical plausibility. Currently, no supplementary video materials are provided.

### Questions
In Table 2, bolding the best results would make it easier for readers to compare methods.

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
This paper proposes PhyMix, a framework for improving physical consistency in single-image 3D indoor scene generation. It introduces a Physics Evaluator that measures four aspects—contact, stability, geometric priors, and deployability—and uses these as rewards in a critic-free Group Relative Policy Optimization (Scene-GRPO) scheme. At inference, a Test-Time Optimizer (TTO) further refines object poses to remove collisions. Experiments on 3D-FRONT show PhyMix significantly improves physical realism without sacrificing visual quality.

### Strengths
• Physical consistency in 3D scene generation is a well-discussed problem, but most prior works tackle it in isolated ways. This paper takes a systematic and unified perspective to address it, which is valuable.
•  The idea of applying reinforcement learning (Scene-GRPO) to improve physical realism is well-motivated, showing a good balance between theory and practicality.

### Weaknesses
• The paper decomposes physical consistency into four main aspects (contact, stability, geometric priors, deployability) and nine sub-metrics, which is conceptually clear but lacks theoretical or empirical justification for this particular taxonomy. Similar attempts have been made in recent works such as LayoutDreamer [Zhou et al., 2025], which also enforces physical plausibility through contact and penetration constraints in text-to-3D scene generation. The paper would benefit from a clearer explanation of why these four aspects are chosen, how they interact, and whether they comprehensively cover all major physical inconsistencies.

• Although the reported results show improvement in physical realism, the experiments are primarily limited to 3D-FRONT, with only a few scene variations. There is no cross-domain validation (e.g., cluttered indoor scenes) or analysis on unseen object categories. Furthermore, ablation on each sub-metric of the Physics Evaluator is missing, leaving it unclear which aspects contribute most to the gains.

• The proposed Group Relative Policy Optimization (GRPO) is an interesting critic-free RL approach, but its advantages over standard algorithms such as PPO [Schulman et al., 2017] or AWR [Peng et al., 2019] are not sufficiently quantified. It would strengthen the claim if the authors could compare sample efficiency, stability, or convergence behavior to existing RL baselines.

• Several parts (e.g., Section 3.1 on scene representation, Equation definitions of the Physics Evaluator) are written only in prose without explicit symbols or equations, which weakens reproducibility. Moreover, the gradient flow between the evaluator, GRPO, and generator is not fully detailed—does the Physics Evaluator participate in backpropagation, or only as a reward signal? Clarifying this would improve methodological transparency.

### Questions
1. How did you determine the weights for the nine sub-metrics? Do these weights transfer across datasets/tasks without retuning? If not, do you have any results on automated weight selection or a sensitivity analysis to support stability and reuse?

2. Compared with differentiable physics or analytic constraints, how does your method differ in sample efficiency, convergence speed, and failure modes? Do you have any small-scale real-world or sim-to-real results to demonstrate deployability?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses a crucial and often overlooked limitation in single-image 3D indoor scene generation: the lack of physical plausibility. While many recent methods achieve high visual fidelity, the resulting 3D scenes often contain obvious physical errors—such as floating objects, collisions, or unstable arrangements—making them unreliable for downstream applications like robotics and embodied AI. The authors propose a two-fold solution: a comprehensive Physics Evaluator and a novel framework, PhyMix, which integrates physics-based guidance into both the training and inference stages. The results demonstrate a significant advancement in generating scenes that are both visually faithful and physically consistent.

### Strengths
• Systematic Benchmarking: The introduction of the unified Physics Evaluator is arguably the most significant contribution. It provides the field with a long-overdue, systematic, and comprehensive set of nine metrics for physical plausibility. This moves the community past ad hoc collision or grounding checks toward a holistic assessment.
• Elegant Technical Solution: The implicit-explicit optimization strategy (Scene-GRPO + TTO) is a theoretically elegant and effective method for handling the dual challenge of integrating both non-differentiable and differentiable constraints into a diffusion-based generative pipeline. The ablation studies confirm the necessity and complementarity of both components.
• Strong Empirical Results and Validation: The performance gains are compelling, showing the method raises the overall physical score by +20.2% relative to the strongest baseline (MIDI). Crucially, the authors validate their metrics with a perceptual user study (MOS and Pairwise Preference), confirming that the quantitative scores align strongly with human judgment of physical plausibility

### Weaknesses
1. Scene-GRPO is an application of the established flow-GRPO/GRPO preference learning paradigm, borrowing its framework directly from the LLM .

2. The current Physics Evaluator relies on simple physical approximations (e.g., center-of-mass checks for static stability) that may fail to capture subtle or fine-grained edge cases, such as an object barely balancing on a thin edge, or the long-term effects of complex weight distribution and material properties. Furthermore, the reliance on these simplified physical metrics—and the design of the nine corresponding constraints—leans too heavily on hand-crafted engineering for the loss design. A more scientifically rigorous approach would involve integrating a sophisticated, general-purpose physics simulation engine for evaluation and differentiable loss, rather than relying on a custom set of rules derived from simplified geometric priors.

### Questions
1. Is Eq.(3)  eps-prediction? Flow matching normally optimizes the conditional velocity field.

2. Why the negative FM loss related to likelihood proxy?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes PhyMix, a framework for physically consistent single-image 3D indoor scene generation, along with a unified Physics Evaluator. It combines Scene-GRPO (implicit optimization) and TTO (explicit refinement) to solve physical inconsistency issues. While targeting a critical problem with a structured approach, the work has notable weaknesses that require thorough revision.

### Strengths
1.The dual-layer optimization (implicit + explicit) is a meaningful exploration to integrate physical constraints into training and inference.
2.Extensive experiments demonstrate the method’s performance in physical plausibility and visual fidelity.

### Weaknesses
1.The paper claims the Physics Evaluator aligns with human judgments but provides no rigorous theoretical or empirical basis for selecting its four core aspects and nine sub-constraints. Its design seems like a simple combination of mature differentiable signals, undermining credibility.
2.PhyMix involves numerous hyperparameters (e.g., Scene-GRPO’s group size K). Limited sensitivity analysis is provided, and excessive hyperparameters may reduce generalization, especially in test-time optimization.
3.Existing works using Taichi for differentiable real physical constraints (gravity, inertia, inter-object interactions) are not fully compared.

### Questions
1.The article mentions that the Physics Evaluator is “align closely with human judgments of physical plausibility”, but it fails to explain the basis for adopting these indicators. Thus, the design of the Evaluator appears to be more like the combination of several mature differentiable guidance methods.
2.The article introduces a large number of hyperparameters in the implementation of PhyMix. I am not certain whether such a large number of hyperparameters will affect the generalization ability, especially during the test-time optimization process
3.As far as I know, some works have implemented differentiable real physical constraints (including gravity, inertia, and inter-object interactions) based on Taichi. What are the advantages of the PhyMix method mentioned in this paper compared with theirs?

### Soundness
2

### Presentation
2

### Contribution
2
