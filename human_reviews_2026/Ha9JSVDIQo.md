# SpinVLA: A Spectral-Invariant Vision-Language-Action Model for Robotic Manipulation

- Decision: Reject
- Scores: 4, 2, 4, 2, 2

## Abstract
Vision-language-action (VLA) models trained on large-scale robot demonstration datasets have achieved impressive in-distribution performance, yet they can fail catastrophically under minor domain shifts. For instance, a VLA-trained robot tasked to “pick the red block” may flounder due to various environmental disturbances such as lighting changes or scene clutter. To address this limitation, we propose SpinVLA, a novel end-to-end VLA architecture that leverages the mathematical equivalence between spectral decomposition and contrastive learning to improve robustness. Drawing inspiration from causal inference principles, which suggest that stable features persist across environments, we hypothesize that consistent patterns in successful demonstrations represent task-relevant information rather than spurious correlations, i.e., statistical associations unrelated to the true causal factors of task performance. Our approach integrates spectral decomposition to identify demonstration-consistent features, contrastive learning to enforce representational stability, and efficient low-rank adaptation modules for environment-specific tuning. Extensive experiments using the open-source LIBERO datasets show that SpinVLA significantly improves robotic manipulation task success rates compared to baseline VLAs under visual perturbations and the presence of out-of-distribution objects, while maintaining comparable in-distribution performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a Vision-Language-Action(VLA) model, designed for robust robotic manipulation by mitigating spurious correlation. In the proposed framework, contrastive learning heads are used to learn invariant visual features robust to domain shifts. This is complemented by a multi-level regularization scheme on the action space, which uses a spectral consistency loss to constrain predictions to the subspace of successful demonstrations and a temporal loss  to enforce smooth motion. The results show an improvement in success rates for the proposed method.

### Strengths
1. The paper clearly articulates how spurious correlations affect VLA generalization and motivates a principled mitigation strategy.
2. The integration of spectral and contrastive objectives is theoretically grounded and technically interesting, bridging representation learning and causal robustness.
3. Multi-level regularization (spectral + temporal) offers a coherent architectural bias that could be extended to other embodied-AI models.

### Weaknesses
Limited OOD evidence: The paper claims improved generalization to distribution shifts, but experiments using the LIBERO benchmark do not explicitly quantify or categorize the types of shifts (e.g., unseen backgrounds, lighting, or object compositions).
Baseline uncertainty: Table 1 uses a re-implemented TinyVLA rather than an official baseline, which weakens the strength of performance comparisons. Clarifying or validating the baseline implementation would strengthen the claims.
Scope of real-world validation: The real-robot evaluation mostly mirrors LIBERO tasks; more diverse real-world tasks would provide stronger evidence of adaptability.

### Questions
If the SpinVLA regularization framework were applied to a 7B backbone (identical in scale to OpenVLA), would it then outperform the standard 7B OpenVLA baseline? Could this method be used to extend the performance of SOTA models, or are its benefits primarily limited to improving smaller, less capable models?
The real-world validation in Section 4.2 (Figure 3) appears limited to sim-to-real transfer of tasks already present in the LIBERO simulation benchmark. This does not sufficiently validate generalization to novel real-world tasks. Could the authors perform a new experiment where a small dataset is collected for a completely new real-world task (i.e., one dissimilar to any task in LIBERO), fine-tune the model, and report the quantitative success rates? We would encourage the authors to add these quantitative results to a new table, as this would provide much stronger evidence for the model's adaptability than the current qualitative figures.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents SpinVLA to address the failures under domain shifts. It leverages the idea that spectral methods can discover stable representations and the equivalence between spectral clustering and contrastive learning to improve VLAs’ stability. The proposed method adds InfoNCE regularizations across vision, language, and action heads of the model. It also computes the spectral decomposition bank of trajectories to regularize action predictions. Finally, it adds spectral constraints to LoRA. The experiments showed improvements over TinyVLA in LIBERO tasks.

### Strengths
- The paper proposes methods to improve VLA through improving representations. This can be important in smaller models and low data regimes. 
- The paper successfully applies the idea of the spectral method across the VLA architecture to improve the policy performance.

### Weaknesses
- While the paper motivates by handling domain shifts in VLAs, no experiments in the paper evaluate perturbation or shifts in camera poses [1] to demonstrate how SpinVLA is robust to domain shifts.
   - [1] Lee et al. “CLASS: Contrastive Learning via Action Sequence Supervision for Robot Manipulation” CoRL 2025.
- It is unclear why it requires so many regularization methods to try to achieve the same goal at different components, and why improving representation doesn’t yield better performance if enforcing representation stability is important. 
- The proposed methods are specific to the VLA architecture shown in Fig. 2. However, there are other VLA architectures like pi0.5, how can the same principle apply to those VLAs?

### Questions
- What’s the ablation result of LoRA without spectral constraints? I cannot find it in the ablation table.
- There are three contrastive heads in vision, language, and action. What’s the effect and importance of each of them?

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
The paper introduces SpinVLA, a new vision-language-action (VLA) architecture designed to improve robustness in robotic manipulation tasks under domain shifts. The key insight is that spectral decomposition and contrastive learning share a mathematical equivalence, which the authors exploit to identify stable, demonstration-consistent patterns that generalize across environments. Experiments on the LIBERO benchmark and with a real UR5e robot demonstrate that SpinVLA maintains high task success rates even with lighting changes, clutter, or unseen objects, while being significantly smaller in parameter count than large-scale models like OpenVLA.

### Strengths
- Fair motivation and problem framing: The authors clearly identify the brittleness of current VLAs to spurious correlations and offer a causally inspired yet practical solution.

-Integration of theory and practice: The idea of translating spectral properties into architectural regularization is well-argued and empirically justified.

-Experimental validation: Results across simulation (LIBERO) and real-world transfer tasks substantiate the claim of robustness.

- Ablations: The study isolates the effects of each component (contrastive, spectral, temporal), showing consistent contributions.

- Resource-efficient: SpinVLA achieves near-OpenVLA performance using a much smaller backbone, making it appealing for low-resource deployment.

- Good writing and structure: The paper reads smoothly, the methodology is well explained, and the figures are effective.

### Weaknesses
- Moderate novelty: While the fusion of spectral and contrastive principles is interesting, the core technical contributions are largely reinterpretations of known methods (InfoNCE equivalence, Laplacian embeddings).

- Limited theoretical grounding: The causal connection between spectral invariance and true causal disentanglement is more suggestive than rigorously demonstrated.

- Narrow evaluation domain: Tests are restricted to manipulation tasks; it’s unclear whether the approach extends to other robotic domains like navigation or locomotion.

- Potential overfitting to demonstration structure: Since spectral components are computed from successful trajectories, models might still capture structured but non-causal biases if demonstrations share hidden artifacts.

- Incomplete efficiency discussion: While LoRA regularization is claimed to be lightweight, the paper lacks detailed runtime and memory comparisons with standard LoRA or baseline models.

### Questions
- How sensitive is SpinVLA to the choice of spectral components (e.g., number of eigenvectors selected)?  

- Could spectral decomposition be applied online, enabling continuous refinement as the robot gathers new demonstrations?  

- Have you examined how much the model relies on the spectral bank at inference time—could performance degrade if regularization is removed post-training?  

- How does SpinVLA compare to data augmentation or domain randomization approaches in terms of robustness gains versus compute cost?  

- Can the authors provide insight into why SpinVLA underperforms OpenVLA on LIBERO-Spatial despite stronger invariance modules?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents SpinVLA which leverages the equivalence between contrastive learning and spectral decomposition to learn stable multimodal representations. Trained on LIBERO benchmarks with low-rank adaptation, SpinVLA achieves higher performance than Tiny-VLA.

### Strengths
Using spectral regularization to constrain VLA action generation may be a workable and lightweight idea.

### Weaknesses
**1. Writing clarity issues. The paper’s exposition is highly confusing.**

 1.1 It introduces many vague and undefined terms, like causal factor, task-relevant information, statistical associations, demonstration-consistent features, environment-specific, etc., without clear definitions, figures, or consistent usage. Many of these phrases appear only once or twice in the abstract or introduction, making the motivation hard to follow.

 1.2 The entire introduction focuses excessively on criticizing existing VLA models with abstract terms like spurious correlations, while giving almost no concrete description of the proposed method. The discussion largely reiterates a generic point, that VLAs tend to overfit, without clearly explaining how the proposed approach addresses it.

**2. Experimental insufficiency.**

 2.1 The paper raises several issues in the introduction (e.g., sensitivity to viewpoint changes) but performs no experiments to verify that SpinVLA alleviates them.

 2.2 The experiments are limited to one comparison with OpenVLA and TinyVLA, where SpinVLA performs significantly worse than the published OpenVLA. It is strongly recommended to validate the idea on stronger backbones (e.g., OpenVLA, Pi series) and additional benchmarks.

 2.3 The ablation studies are arbitrary.

Overall, the paper needs restructuring, clearer motivation, and substantially more experiments to convincingly demonstrate the proposed idea.

### Questions
Please see the weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed a new VLA framework SpinVLA that combines spectral decomposition with contrastive learning for improved performance on robot manipulation. The method assumes that different successful demonstrations share common patterns representing task-relevant information, which can be obtained via contrastive learning, and multi-scale regularizations are then applied to discourage spurious correlations and preserve demonstrated behaviors. The method is validated on LIBERO dataset and a real robot setup with simple "pick-and-place" task. Compared to TinyVLA, the proposed SpinVLA shows advanced performance on simulation benchmarks.

### Strengths
1. The proposed idea of applying spectral decompositions to bias model towards successful motion patterns is interesting and effective.
2. The paper is easy to follow with clear illustrations.

### Weaknesses
1. My primary concern is that the paper’s overall contribution appears incremental. Incorporating an InfoNCE loss in robot manipulation is not novel [1], and according to Sec. 4.3, most of the reported improvement comes directly from the InfoNCE and consistency regularization terms.
2. The paper claims that InfoNCE is used to extract task-relevant patterns, yet no analysis is provided to support this. For example, what do the representations look like before and after the spectral decomposition? Do they actually capture task-relevant information as intended? Additional evidence is needed to justify this claim.
3. The paper presents quantitative results only on the LIBERO dataset, while the real-world evaluation is limited to a simple qualitative demonstration that the policy transfers to a basic setup—which seems trivial. What are the actual performance numbers of TinyVLA and SpinVLA on real-world pick-and-place tasks? Does SpinVLA still outperform TinyVLA by a significant margin?
4. The paper lacks a failure analysis. Understanding how and when the method fails is important for assessing whether the proposed approach truly biases the model toward consistent patterns shared among successful demonstrations.
5. The real-world task is a very simple pick-and-place. Can SpinVLA qualitatively handle more complex tasks? Since more complex tasks include greater behavioral diversity across both successful and failed demonstrations, can the proposed approach still effectively extract shared patterns in such scenarios?

[1] Jiang, Guangqi, et al. “Robots Pre-Train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets.” arXiv:2410.22325 (2024).

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
