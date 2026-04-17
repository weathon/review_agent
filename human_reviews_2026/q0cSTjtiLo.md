# Interaction-Consistent Object Removal via MLLM-Based Reasoning

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Image-based object removal often erases only the named target, leaving behind interaction evidence that renders the result semantically inconsistent. We formalize this problem as Interaction-Consistent Object Removal (ICOR), which requires removing not only the target object but also associated interaction elements, such as lighting-dependent effects, physically connected objects, target-produced elements, and contextually linked objects. To address this task, we propose Reasoning-Enhanced Object Removal with MLLM (REORM), a reasoning-enhanced object removal framework that leverages multimodal large language models to infer which elements must be jointly removed. REORM features a modular design that integrates MLLM-driven analysis, mask-guided removal, and a self-correction mechanism, along with a local-deployment variant that supports accurate editing under limited resources. To support evaluation, we introduce ICOREval, a benchmark consisting of instruction-driven removals with rich interaction dependencies. On ICOREval, REORM outperforms representative image editing systems, demonstrating its effectiveness in producing interaction-consistent results.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes REORM, a system for interaction-consistent object removal (ICOR). It combines a multimodal large language model (MLLM) for reasoning about which related elements should be removed, with a mask-guided inpainting module, a self-correction mechanism, and a lightweight local variant. The authors also introduce a new evaluation benchmark, ICOREval, and report better quantitative results than existing image-editing methods.

### Strengths
The paper clearly formulates a refined variant of the object removal task that accounts for both the target object and its surrounding contextual or interacting elements, such as shadows, reflections, and connected parts. This makes the problem definition more realistic and relevant to real-world editing scenarios. The authors build a complete end-to-end system and introduce a new benchmark, ICOREval, specifically designed to evaluate interaction-consistent removal quality. Experimental results indicate moderate but consistent improvements over strong baselines, and the visual examples show more coherent and natural editing outcomes in several challenging cases.

### Weaknesses
1. There seems to be a collateral deletion issue — for example, in Fig. 1 (iii), background objects are also removed, indicating the presence of erroneous behavior.

2. The error analysis is insufficient. It would be valuable to discuss whether error accumulation occurs when combining LLMs and MLLMs.

3. The method has not been evaluated on existing benchmark datasets, such as MagicBrush, which limits the fairness and completeness of the comparison.

4. It is recommended to include inversion-based image editing and inpainting methods in the comparison or discussion. 

5. The proposed approach appears to be computationally expensive; more experiments or analyses on computational cost should be provided.

6. Given that introducing LLMs and MLLMs may enable more accurate mask extraction, traditional inpainting methods could potentially achieve strong results as well.

7.Overall, the proposed method lacks novelty. It does not clearly define a new problem and the technical contributions mainly rely on modular integration rather than introducing fundamentally new ideas.

### Questions
Same as weaknesses

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
5

### Summary
This paper proposes a task called Interaction-Consistent Object Removal, which aims to remove not only the specified object but also other elements that interact with it in the scene. The framework uses multimodal large language models to infer and guide the removal of interacting objects. The framework includes an MLLM-driven reasoning module, a self-correction mechanism, and a locally deployable lightweight variant. A new dataset containing 72 examples is also introduced to evaluate semantic and visual consistency.

### Strengths
1. The paper introduces a meaningful task, Interaction-Consistent Object Removal, extending traditional object removal to model semantic and physical interactions among scene elements.
2. The inclusion of a local deployment variant demonstrates practical value and thoughtful consideration of real-world constraints.

### Weaknesses
1. The ICOR task can be viewed as an extension of prior object-effect removal methods, expanding from handling only lighting-dependent effects to covering more types, such as physically connected objects, target-produced elements, and contextually linked objects. However, if paired training data were available, prior methods might also be able to handle these limited cases, which would limit the novelty.
2. The ICOREval dataset, while a contribution of the paper, includes only 72 examples, which is limited and may introduce bias in evaluating task effectiveness.
3. The paper lacks discussion of failure or over-removal cases, and does not analyze when the reasoning process misfires or produces incorrect object associations.
4. Although intent-aware editing is mentioned in Section 5.4, no mechanism is proposed to adapt to different user intentions or control levels, leaving controllability unexplored.

### Questions
1. Issues raised in the *Weaknesses* section.
2. Are there plans to expand the ICOREval dataset for broader benchmarking?
3. Are there examples where self-correction fails or leads to over-removal?

### Soundness
2

### Presentation
3

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
This paper introduces Interaction-Consistent Object Removal (ICOR), a new formulation of the object-removal task that requires eliminating not only the target object but also its interaction-related elements—such as shadows, reflections, connected objects, or contextually dependent items—to maintain semantic and visual coherence.
The authors propose REORM, a modular reasoning-enhanced framework that leverages multimodal large language models (MLLMs) to infer what secondary elements must also be removed. The system combines MLLM-driven reasoning, open-vocabulary segmentation, and diffusion-based inpainting, followed by an MLLM-controlled self-correction stage.
A new benchmark, ICOREval, is constructed for evaluation. Experiments show REORM achieves better visual quality and semantic consistency than existing instruction-based image editors such as MGIE, SmartEdit, GPT-Image-1, and Nano Banana.

### Strengths
### Originality
Defines a new task: Interaction-Consistent Object Removal, emphasizing removal of both objects and their interaction elements. The introduction of Interaction-Consistent Object Removal extends classical object removal to a more semantically complete problem, emphasizing real-world plausibility.

### Quality
Technically solid modular design combining reasoning, segmentation, and diffusion-based inpainting.

### Clarity
The paper is clearly structured with well-designed figures explaining each module.

### Significance
The focus on removing interaction-linked elements (lighting effects, physical connections, contextual dependencies) is novel and practically valuable for photo editing and generative scene understanding.

### Weaknesses
1. While the task definition is new, the core framework mainly orchestrates existing tools (GPT-4o, Grounded-SAM, ObjectClear). The contribution lies more in integration and reasoning design than in fundamental algorithmic innovation.

2. The use of MLLM reasoning to guide inpainting is not entirely novel — similar techniques appear in BrushEdit[1] and Magicquill[2]. 

3. The main version depends on closed APIs (GPT-4o), raising concerns about reproducibility, cost, and long-term accessibility. The paper could discuss these trade-offs more explicitly.

4. ICOREval contains only 72 examples, which limits the statistical robustness of conclusions. Larger or more diverse benchmarks would strengthen generalizability claims.

5. The metric design focuses on perceptual similarity (DINO, LPIPS, SSIM) rather than explicitly measuring semantic consistency or over-editing, leaving the notion of interaction consistency somewhat subjective.

[1] Li Y, Bian Y, Ju X, et al. Brushedit: All-in-one image inpainting and editing[J]. arXiv preprint arXiv:2412.10316, 2024.
[2] Liu Z, Yu Y, Ouyang H, et al. Magicquill: An intelligent interactive image editing system[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 13072-13082.

### Questions
1. How is interaction consistency operationally defined or measured beyond qualitative examples? Could an explicit metric be introduced?
2. Beyond the engineering-level modular integration, what constitutes the paper’s core innovation? Similar pipeline connections have appeared in prior work, which somewhat limits the novelty of this contribution.
3. What is the runtime and resource requirement per image for the overall pipeline?

### Soundness
3

### Presentation
3

### Contribution
2
