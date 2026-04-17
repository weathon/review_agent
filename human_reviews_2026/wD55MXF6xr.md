# DyME: Dynamic Multi-Concept Erasure in Diffusion Models with Bi-Level Orthogonal LoRA Adaptation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 2

## Abstract
Text-to-image diffusion models (DMs) inadvertently reproduce copyrighted styles and protected visual concepts, raising legal and ethical concerns. Concept erasure has emerged as a safeguard, aiming to selectively suppress such concepts through fine-tuning. However, existing methods do not scale to practical settings where providers must erase multiple and possibly conflicting concepts. The core bottleneck is their reliance on static erasure: a single checkpoint is fine-tuned to remove all target concepts, regardless of the actual erasure needs at inference. This rigid design mismatches real-world usage, where requests vary per generation, leading to degraded erasure success and reduced fidelity for non-target content. We propose DyME, an on-demand erasure framework that trains lightweight, concept-specific LoRA adapters and dynamically composes only those needed at inference. This modular design enables flexible multi-concept erasure, but naive composition causes interference among adapters, especially when many or semantically related concepts are suppressed. To overcome this, we introduce bi-level orthogonality constraints at both the feature and parameter levels, disentangling representation shifts and enforcing orthogonal adapter subspaces. We further develop ErasureBench-H, a new hierarchical benchmark with brand–series–character structure, enabling principled evaluation across semantic granularities and erasure set sizes. Experiments on ErasureBench-H and standard datasets (e.g., CIFAR-100, Imagenette) demonstrate that DyME consistently outperforms state-of-the-art baselines, achieving higher multi-concept erasure fidelity with minimal collateral degradation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a dynamic concept erasure framework for multi-concept unlearning. The proposed framework first learns concept-specific LoRA for each concept. Then the framework learns to dynamically compose the LoRAs corresponding to the target concepts during inference. The experiments are conducted on various commonly used benchmarks. In addition, this paper also proposes an ErasureBench-H benchmark for more comprehensive multi-concept evaluation.

### Strengths
1. Multi-concept erasure is crucial for real-world trustworthy generative AI deployments.
2. The proposed method is reasonable to handle the dynamic scenarios during inference.
3. The overall paper is easy to follow.

### Weaknesses
1. The proposed framework applies LoRA adapters to encode the specific concepts. However, in the real-world multi-concept erasure setting, the number of concepts would be large. It would cause the memory issues that the users cannot store such a large amount of adapters.
2. Step 3 of the proposed method requires the framework to first train all concept LoRAs jointly. However, in real-world scenarios, it is possible that the user wants to add novel concepts beyond the training set. It will require re-training once novel concepts appear, which might not be practical. 
3. While this paper enforces each concept to be non-overlapping, it remains hard to handle the rephrased prompts that would cause robustness issues, allowing attackers to easily recover the concept by paraphrasing the target prompts.

### Questions
1. This paper contains no qualitative visualization in the main paper. It would be beneficial to demonstrate the visualization for concept erasure on a large number of multiple concepts (e.g., 5 or more concepts).

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes DYME, a dynamic multi-concept erasure framework for text-to-image diffusion models that allows on-demand suppression of copyrighted or sensitive concepts. Instead of static fine-tuning, DYME trains lightweight, concept-specific LoRA adapters and dynamically composes only those needed at inference. To mitigate interference among multiple erased concepts, it introduces bi-level orthogonality constraints at both the feature and parameter levels. The authors also present ERASUREBENCH-H, a hierarchical benchmark reflecting real-world brand–series–character relationships. Extensive experiments on CIFAR-100, Imagenette, and ERASUREBENCH-H show that DYME achieves superior erasure fidelity and utility preservation compared to prior static approaches, effectively scaling to large and overlapping concept sets.

### Strengths
1. The paper introduces a new concept erasure setting, where multiple target concepts can be erased simultaneously during inference, reflecting more realistic real-world scenarios.
2. It proposes bi-level orthogonality constraints, which effectively enable the stable composition of multiple concept-specific LoRA modules at inference without interference.
3. The paper presents ERASUREBENCH-H, a hierarchical and semantically structured benchmark that enables comprehensive and realistic evaluation of multi-concept erasure performance

### Weaknesses
1. In Figure 1, the performance of MACE appears inconsistent with its original paper, where it successfully erased 100 concepts (e.g., 100 celebrities) while maintaining high harmonic accuracy. Here, only up to 20 concepts are erased, and the results for MACE seem unexpectedly poor. This might be due to differences in the CIFAR-100 setting, where protected concepts may not have been properly included in the retention set. The authors are encouraged to revisit their reproduction settings. In addition, SPM should be categorized under Dynamic Erasure (DE), as described in Section 3.3 of its paper.
2. In line 73, the claim that static approaches “reduce diversity and degrade fidelity” lacks supporting evidence. In contrast, MACE demonstrates good fidelity preservation even under multi-concept erasure, which contradicts this statement.
3. In the Introduction, the paper does not clearly define the difference between static erasure paradigm and dynamic erasure paradigm, even though these are the core concepts that underpin the paper’s motivation. This omission makes it difficult for readers to fully grasp the main idea.
4. The major concern lies in the motivation of the proposed task setup. The authors emphasize that the weakness of static erasure lies in its inability to dynamically select concepts during inference, while all concepts are erased during training. However, the fundamental goal of concept erasure is to permanently remove target semantics from the model parameters without affecting others. The proposed dynamic erasure is unrealistic in white-box settings (e.g., Stable Diffusion), since LoRA adapters are external and not merged into model weights — an attacker could easily bypass DYME by modifying the inference code directly.
5. Although the proposed method might have value in black-box deployment, similar behavior could be achieved with simpler mechanisms. Since DYME selects LoRA modules based on keyword matching, a trivial baseline could simply detect forbidden keywords in prompts and return a blank or noisy image otherwise generating normally, achieving equivalent or even better results (i.e., AccEE = 0 and unchanged AccUP).
6. The proposed Dynamic Composition at Inference (Sec. 4.2) relies on explicit keyword matching, which fails to address implicit concepts, such as those in the I2P benchmark [1]. Handling NSFW or implicit concepts is a critical aspect of practical concept erasure that this method does not consider.
7. In line 106, the paper claims “this is the first work to systematically investigate multi-concept erasure scalability in diffusion models,” but prior studies such as [2, 3] have already discussed similar topics.
8. In line 241, the symbol j is undefined.
9. In Equation (3), the loss formulation seems to encourage opposite directions (cosine similarity = –1) rather than orthogonality (cosine similarity = 0), which contradicts the intended behavior.
10. The experimental comparisons include only SPM and MACE (both CVPR 2024). More recent and relevant baselines with similar motivations are missing and should be added for completeness.
11. The experiments in Section 5.2 only cover up to 20 erased concepts, while MACE has demonstrated the ability to erase 100 concepts. Although the dynamic approach may scale better, expanding the number of erased conceptswould strengthen the motivation and claims.
12. The ERASUREBENCH-H benchmark lacks sufficient detail and description in the main text; it is hard to understand its structure and usage without referring to the appendix.
13. In Table 4 (Config 1), it is unclear why AccUP remains 70.50 after removing LoRA-C. This inconsistency needs clarification or additional explanation.



[1] Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models, CVPR23

[2] Localized Concept Erasure for Text-to-Image Diffusion Models Using Training-Free Gated Low-Rank Adaptation, CVPR25

[3] Erasing Concept Combination from Text-to-Image Diffusion Model, ICLR25

### Questions
See the weakness aprt.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates dynamic multi-concept erasure by pretraining LoRA adapters and composing the required concepts at inference. To mitigate crosstalk among LoRA modules, the authors propose Bi-Level Orthogonal Constraints that minimize overlap in LoRA-induced representation shifts and enforce orthogonality in the parameter space. Experimental evaluation employs a CLIP-based classifier to assess both erasure effectiveness and utility preservation. The results indicate that the proposed method maintains the original utility while achieving the lowest accuracy for the erased concepts.

### Strengths
1. Unlike existing concept erasure methods, this paper focuses on dynamic, on-demand multi-concept erasure, enabling separate LoRA adapters to be combined dynamically at inference time.
2. The paper is well-structured and clearly written. The proposed DyME can mitigate LoRA crosstalk.

### Weaknesses
1. For the motivation, this paper addresses the dynamic combination of trained concept-specific LoRA modules to enable flexible concept erasure in diffusion models. It specifically tackles scenarios where certain concepts may need to be removed from existing LoRA sets. However, the proposed method is limited in that it can add one new concept at a time, potentially requiring retraining across all LoRA modules. The approach’s scalability is restricted if it cannot seamlessly incorporate new concepts without retraining.

2. For the evaluation metrics, the authors adopt a CLIP-based classifier to assess erasure effectiveness and utility preservation. However, existing works typically employ the CLIP score for erasure effectiveness and FID for utility preservation. It remains unclear what advantages the CLIP-based classifier provides over these established metrics. Moreover, LoRA-based weight modifications inevitably affect untargeted concept generation, even if only minimally. Yet, the results measured by $Acc_{UP}$ suggest that the method fully preserves the original performance on untargeted concepts. Therefore, $Acc_{UP}$ alone may not sufficiently capture potential degradation in untargeted concept generation.

3. As the number of erased concepts increases, the difficulty of training LoRA correspondingly rises. Therefore, it remains unclear whether the proposed method can still perform effectively when dealing with a larger set of concepts to erase.

### Questions
1. How flexible is the system in incorporating new concepts after initial training?
2. Can the authors clarify whether DyME remains effective when erasing a large number of concepts?
3. What is the rationale for choosing this particular evaluation metric?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents DYME, a modular and scalable framework for dynamic multi, concept erasure in text, to, image diffusion models. Instead of retraining or editing a model for each new erasure request, DYME trains one lightweight LoRA adapter per concept and dynamically composes relevant adapters during inference. To handle interference among multiple adapters, the authors introduce a bi, level orthogonality mechanism consisting of (1) an input, aware orthogonality constraint on induced representation shifts and (2) an input, agnostic parameter, space regularizer derived from a theoretical sufficient condition (Theorem 1). The paper also contributes ERASUREBENCH, H, a hierarchical benchmark organized into brand–series–character levels. Experiments on CIFAR, 100, Imagenette, and ERASUREBENCH, H demonstrate improved erasure efficacy and reduced interference compared to prior static erasure baselines.

### Strengths
1. Addresses real, world dynamic takedown scenarios where multiple concept erasures must coexist flexibly.
2. Per, concept LoRA adapters make updates lightweight and composable.
3. Provides a theoretically backed mechanism to reduce interference among multiple LoRAs.
4. ERASUREBENCH, H offers a structured and hierarchical testbed for compositional erasure.
5. Empirical results: Demonstrate consistent gains across datasets and include several informative ablations.

### Weaknesses
1. Missing comparison to contemporaneous SOTA (e.g., Receler).
The work omits comparisons with key 2025 approaches that address similar problems. This omission weakens claims of novelty and superiority. A direct experimental or analytical comparison with Receler and other strong 2025 methods should be added. 

2. Theorem 1 vs implementation inconsistency.
The theorem assumes q and k are frozen while LoRA updates only v/o, yet the implementation adapts q/k/v/o. The authors should either align the implementation with the theorem’s assumptions or extend the theoretical justification accordingly. 

3. Scalability and cost reporting.
With hundreds of unit concepts, training one LoRA per concept can be computationally expensive. The paper should report adapter size, total storage, per, adapter training time, and inference latency when composing many adapters. 

4. Baseline fairness.
Static baselines were restricted to small erasure scopes due to collapse; stronger tuning or staged training might alleviate this. A justification or expanded baseline study is required.

### Questions
Questions*
1. Why is Receler (and other 2025 SOTA methods) not compared? If unavailable, can you provide a reasoned analysis or reimplementation?
2. Do you freeze q/k during LoRA training? If not, how does Theorem 1 apply in practice?
3. What are the per, adapter parameter counts, total storage, training time, and inference latency at different composition sizes?
4. How were static baselines tuned, and are there hyperparameter settings that allow them to scale further?

### Soundness
3

### Presentation
3

### Contribution
2
