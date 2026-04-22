# Optimizing ID Consistency in Multimodal Large Models:  Facial Restoration via Alignment, Entanglement, and Disentanglement

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Multimodal editing large models have demonstrated powerful editing capabilities across diverse tasks. However, a persistent and long-standing limitation is the decline in facial identity (ID) consistency during realistic portrait editing. Due to the human eye’s high sensitivity to facial features, such inconsistency significantly hinders the practical deployment of these models. 
Current facial ID preservation methods struggle to achieve consistent restoration of both facial identity and edited element IP due to Cross-source Distribution Bias and Cross-source Feature Contamination.
To address these issues, we propose EditedID, an Alignment-Disentanglement-Entanglement framework for robust identity-specific facial restoration. By systematically analyzing diffusion trajectories, sampler behaviors, and attention properties, we introduce three key components: 1) Adaptive mixing strategy that aligns cross-source latent representations throughout the diffusion process. 2) Hybrid solver that disentangles source-specific identity attributes and details. 3) Attentional gating mechanism that selectively entangles visual elements. Extensive experiments show that EditedID achieves state-of-the-art performance in preserving original facial ID and edited element IP consistency.
As a training-free and plug-and-play solution, it establishes a new benchmark for practical and reliable single/multi-person facial identity restoration in open-world settings, paving the way for the deployment of multimodal editing large models in real-person editing scenarios.
The code is available at https://github.com/NDYBSNDY/EditedID.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents EditedID, a novel, training-free, and plug-and-play framework designed to solve the critical problem of facial identity (ID) inconsistency in multimodal large editing models. The authors identify that current models, while powerful, degrade significantly when editing realistic human portraits, especially with complex instructions. This failure is attributed to two core issues:
1.  Cross-source Distribution Bias: A mismatch between the original image's facial features and the edited model's feature distribution, leading to detail loss or the generation of a completely different, random identity.
2.  Cross-source Feature Contamination: The bleeding of features between the original face and the edited element (e.g., glasses), which causes the model to either lose the specific attributes of the edit (e.g., "black-framed" glasses become generic glasses) or fail entirely when the edit introduces facial artifacts.

To solve this, EditedID proposes a three-stage Alignment-Disentanglement-Entanglement process. It works by:
a. Alignment (Adaptive Mixing): Mitigates distribution bias by aligning the latent representations of the original face (Original ID) and the edited, ID-inconsistent image (Intermediate ID) throughout the diffusion trajectory.
b. Disentanglement (Hybrid Solver): Isolates feature contamination by using a novel sampler that combines DDIM (to retain the core identity) and DPM-Solver++ (to enhance fine details), effectively separating the person's identity from the edited element.
c. Entanglement (Attentional Gating): Selectively fuses the desired features, combining the restored facial attributes from the Original ID with the specific edited elements (Element IP) from the Intermediate ID.

The authors claim this framework achieves state-of-the-art performance in preserving both the original facial ID and the integrity of the edited elements, establishing a new benchmark for practical, real-world portrait editing.

### Strengths
1.  Novel and Systematic Framework: The proposed "Alignment-Disentanglement-Entanglement" framework is intuitive and systematically tackles the identified issues. Instead of a single "trick," it's a comprehensive process where each component (Adaptive Mixing, Hybrid Solver, Attentional Gating) has a clearly defined role inspired by analysis of diffusion trajectories, samplers, and attention.
2.  High-Impact Problem: The paper addresses a widely acknowledged limitation of current generative models. 
3.  Strong Problem Diagnosis: The authors provide a clear and insightful analysis of why existing methods fail, breaking the problem down into the distinct concepts of "Cross-source Distribution Bias" and "Cross-source Feature Contamination." This clear diagnosis provides a strong motivation for their multi-part solution.

### Weaknesses
1.  Potential for High Complexity and Latency. The proposed framework, involving trajectory mixing, a hybrid solver, and an attentional gating mechanism, sounds significantly more complex than a standard diffusion pass. This could introduce substantial computational overhead and increase inference time, which might limit its practical "plug-and-play" utility.
2.  Ambiguity in "Element IP" Preservation. The mechanism (Attentional Gating) must reliably distinguish between a desired edited element (like glasses) and an undesired artifact (like a distorted nose) from the intermediate image. It's unclear how robust this "entanglement" is, especially if the edit itself is complex and affects multiple facial regions.
3.  Lack of Detail on Multi-Person Handling. While the paper claims to support multi-person restoration, it's unclear how it handles multiple IDs, interactions between subjects, or the allocation of edited elements to the correct person.

### Questions
1.  Potential for High Complexity and Latency. The proposed framework, involving trajectory mixing, a hybrid solver, and an attentional gating mechanism, sounds significantly more complex than a standard diffusion pass. This could introduce substantial computational overhead and increase inference time, which might limit its practical "plug-and-play" utility.
2.  Ambiguity in "Element IP" Preservation. The mechanism (Attentional Gating) must reliably distinguish between a desired edited element (like glasses) and an undesired artifact (like a distorted nose) from the intermediate image. It's unclear how robust this "entanglement" is, especially if the edit itself is complex and affects multiple facial regions.
3.  Lack of Detail on Multi-Person Handling. While the paper claims to support multi-person restoration, it's unclear how it handles multiple IDs, interactions between subjects, or the allocation of edited elements to the correct person.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the long-standing issue of facial identity inconsistency in multimodal editing large models during real-person portrait editing. To solve these problems, the authors propose EditedID—a training-free, plug-and-play framework based on the "Alignment-Disentanglement-Entanglement" paradigm. EditedID optimizes diffusion model dynamics through Adaptive Mixing, Hybrid Solver and Attentional Gating. Extensive experiments on challenging scenarios show EditedID achieves state-of-the-art performance: it outperforms SOTA ID preservation methods by 0.27 in ID similarity and 2.43 in edited element preservation, while maintaining efficiency. It also enhances ID consistency for existing multimodal models (e.g., boosting In-ContextEdit’s ID-Sim by 0.16).

### Strengths
- The paper directly addresses a practical bottleneck—ID inconsistency in real-person editing—that limits multimodal models’ deployment in fields like fashion, film, and portrait design. Unlike prior work focusing on frontal face scenarios, it tackles open-world challenges (non-focused faces, occlusions, multi-person) that are more relevant to real use.
- The framework is grounded in rigorous analysis of diffusion dynamics, including trajectory controllability, sampler complementarity and attention roles.
- EditedID requires no additional training data or model fine-tuning, making it compatible with pre-trained multimodal models (e.g., GPT-4o Plus) without resource-intensive retraining. This lowers deployment costs and enhances scalability.
- The experiments cover both quantitative and qualitative evaluations, with comparisons across 4 method categories and 6 challenging scenarios. Ablation studies also validate the necessity of each component, ensuring the framework’s interpretability.

### Weaknesses
- The paper mentions using DiffBIR for super-resolution pre-processing of degraded edited images (e.g., blurry faces), but it does not discuss how DiffBIR’s own errors (e.g., texture distortion) affect EditedID’s final results.
- The paper exclusively validates EditedID on UNet-based pre-trained models (e.g., Stable Diffusion v1.5), while failing to demonstrate its effectiveness on Diffusion Transformer (DiT-based) models—the current mainstream in diffusion research and industrial deployment (e.g., Flux.1 Kontext’s advanced variants). DiT architectures differ fundamentally from UNet in attention computation (token-wise vs. spatial-wise), which raises critical questions: Can the "Attentional Gating" module—designed for UNet’s spatial attention—adapt to DiT’s token-based attention? Will the "Hybrid Solver’s" timestep scheduling (optimized for UNet’s noise prediction) remain effective in DiT’s transformer-driven denoising process? This gap limits the framework’s applicability to cutting-edge models, as DiT now dominates scenarios requiring high efficiency and large-scale feature modeling.
- Typo: I suppose the word 'BiffBIR' in Line 164 should be 'DiffBIR'.

### Questions
- Have you conducted preliminary experiments on DiT-based architectures? If so, how did you modify the "Attentional Gating" module to fit token-wise attention? If not, what technical challenges do you anticipate in adapting EditedID to DiT, and what solutions are planned?
- If DiffBIR (the pre-processing tool for degraded images) introduces artifacts (e.g., incorrect facial contours), how does EditedID mitigate this? Is there a plan to integrate EditedID with restoration models end-to-end to avoid error propagation?

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
4

### Summary
The paper proposes EditedID, a training-free, plug-and-play pipeline for identity-consistent face editing/restoration from edited portraits. It comprises (i) Alignment via adaptive mixing of two inversion trajectories, (ii) Disentanglement via a Hybrid Solver that mixes DDIM and DPM-Solver++, and (iii) Entanglement via Attentional Gating that replaces attention maps under masks/tokens (e.g., face with glasses). Experiments report improvements on ID-Sim (ArcFace cosine), CLIP-S, and ImageReward; efficiency claims include 6 steps and ~4.2s per image with constant time under multi-ID.

### Strengths
- Clear practical target (ID drift in edited portraits) and a coherent three-stage pipeline mapping to common diffusion operations.  
- Sampler insight into DDIM vs DPM-Solver++ is distilled into an actionable hybrid schedule; the paper gives explicit step ranges [s_1,s_2] in code-style detail.    
- Training-free at dataset level with wide editor compatibility (academic and industrial editors).  
- Ablations (module removal) show separable contributions on ID-Sim / CLIP-S / ImageReward.

### Weaknesses
1.  The ``self-attn keeps single-element structure / cross-attn handles multi-element semantics`` is plausible but untested across U-Net variants/resolutions; current evidence is largely qualitative. Provide attention-map statistics across layers/architectures.  
2.  The method described in this article was tested on the outdated U-Net and its migration performance was not tested on the modern DiT architecture. Currently, DiT has become mainstream in the field of image editing and generation.
3. Training-free” vs instance-level optimization (definition gap). Alignment uses learnable \lambda_t updated by gradient descent per image; disentanglement also optimizes distinct null-text embeddings per ID. This is per-image optimization, not “no learning.” Authors must formally define “training-free” (no dataset-level updates vs. no parameter learning at all) and include these optimization costs in timing.    
4. Hybrid-solver theory is under-justified. The paper splices DDIM and DPM-Solver++ with a “global-timestep preset,” but offers no stability/consistency analysis for solver-mixing across schedules (error accumulation, reversibility, path homotopy).

### Questions
See Weakness.

### Soundness
2

### Presentation
2

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
This paper introduces a training-free framework for improving facial fidelity and identity consistency in personalized image synthesis. To begin with, it points out two limitations in existing methods, including the cross-source distribution bias that models are trained with different data (e.g., with different image resolutions), and the cross-source feature contamination that the identity features may be affected by those of the edited elements. Subsequently, the proposed framework mainly incorporates three strategies to address these limitations. First, it proposes an adaptive mixing strategy that utilizes a learnable combination weight to adaptively merge the latents of the original image and the edited image, yielding a unified representation to tackle the cross-source distribution bias. Second, a hybrid solver is introduced to optimize the null-text embeddings of the original image and the edited image,  to balance identity preservation and detail editing. Finally, an attentional gating strategy is adopted to fuse both the null-text embeddings and generate the final result. Extensive experiments have been conducted to validate the proposed strategies, and the proposed method outperforms several open-source and closed-source large models.

### Strengths
1. This paper is insightful, and its motivation for balancing facial identities and edited elements is clearly stated by multiple observations from pilot studies.

2. The proposed framework is training-free, can be employed with a consumer-grade GPU, and its results are considerable.

3. The proposed components are validated via solid experiments.

### Weaknesses
1. The inference cost of the proposed method has not been reported and may be large, as it requires three stages of optimization (alignment, disentanglement, and entanglement).

2. The overlapping region fusion weight $\hat{w}$ varies in different scenarios. Although two principles for setting the weight have been provided in the Appendix,  it is still inconvenient for ordinary users to configure it to achieve fine-grained controls.

### Questions
1. DDIM and DPM-Solver++ are used in the disentanglement stage, have any other solvers been considered?

### Soundness
3

### Presentation
3

### Contribution
3
