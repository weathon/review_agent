# Deforming Videos to Masks: Flow Matching for Referring Video Segmentation

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Referring Video Object Segmentation (RVOS) requires segmenting specific objects in a video guided by a natural language description. The core challenge of RVOS is to anchor abstract linguistic concepts onto a specific set of pixels and continuously segment them through the complex dynamics of a video. Faced with this difficulty, prior work has often decomposed the task into a pragmatic `locate-then-segment' pipeline. However, this cascaded design creates an information bottleneck by simplifying semantics into coarse geometric prompts (e.g, point), and struggles to maintain temporal consistency as the segmenting process is often decoupled from the initial language grounding. To overcome these fundamental limitations, we propose FlowRVS, a novel framework that reconceptualizes RVOS as a conditional continuous flow problem. This allows us to harness the inherent strengths of pretrained T2V models, fine-grained pixel control, text-video semantic alignment, and temporal coherence. Instead of conventional generating from noise to mask or directly predicting mask, we reformulate the task by learning a direct, language-guided deformation from a video's holistic representation to its target mask. Our one-stage, generative approach achieves new state-of-the-art results across all major RVOS benchmarks. Specifically, achieving a J&F of 51.1 in MeViS (+1.6 over prior SOTA) and 73.3 in the zero shot Ref-DAVIS17 (+2.7), demonstrating the significant potential of modeling video understanding tasks as continuous deformation processes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces FlowRVS, a novel framework designed to address the Referring Video Object Segmentation (RVOS) task by reframing it as a flow matching problem. The method proposes a Flow Matching Model that predicts continuous deformation fields. This design inherently promotes superior temporal coherence and avoids the compounding errors typical of sequential prediction stages. Extensive experiments conducted on standard RVOS benchmarks, including Ref-YouTube-VOS and Ref-DAVIS, consistently demonstrate that FlowRVS achieves state-of-the-art results

### Strengths
1. The core idea of replacing the sequential segmentation pipeline with a unified Flow Matching Model represents a significant conceptual leap for RVOS.
2. The quantitative comparisons presented in the paper showcase substantial performance gains over existing state-of-the-art techniques.
3. The paper is well-written and logically structured, with clear figures.

### Weaknesses
1. The definition of the two-stage 'locate-then-segment' design is potentially misleading; many modern Transformer-based (e.g., ReferFormer) and MLLM approaches (e.g., LISA)  already bypass explicit bounding box output and generate segmentation masks directly. 
2. The discussion in Section Introduction neglects relevant previous methods, such as SAMWISE and others, which explicitly incorporate a 'temporal segmenting' stage. 
3. The related work section provides insufficient discussion regarding Generative Modeling in VOS or RVOS, specifically lacking recent efforts like "Exploring Pre-trained Text-to-Video Diffusion Models for Referring Video Object Segmentation," which is highly relevant to flow-based generation. 
4. The network structure and update strategy for the Flow Matching Model would be significantly clearer and more reproducible if they were enhanced with formalized mathematical expressions and equations. 
5. Despite stating that the foundation model used is Wan2.1, the overall architectural logic of the proposed work, particularly as inferred from the high-level design figures (e.g., Figure 3), shares a similar idea with the ControlNet framework. The authors should explicitly clarify the differences or similarities with a conditional generation framework like ControlNet.

In summary, despite the compelling performance gains, the cumulative impact of these missing discussions regarding prior work and the lack of technical detail in the model's description remains a significant issue.

### Questions
1. What is the training and inference computational overhead (e.g., GPU hours, FLOPs per frame) for FlowRVS compared to SAMWISE?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes FlowRVS, a one-stage framework that reframes Referring Video Object Segmentation (RVOS) as a text-conditioned continuous flow matching from a video sequence to its target mask sequence. Instead of leveraging previous  “locate-then-segment” paradigm, FlowRVS learns a velocity field (via flow matching) that deterministically deforms video latents into mask latents under language guidance. To adapt existing video generation models to RVOS task, the paper proposes three techniques to stabilize the convergent video→mask flow: Boundary-Biased Sampling (BBS), Start-Point Augmentation (SPA), and Direct Video Injection (DVI).  The proposed FlowRVS achieves new SOTA on MeViS and strong zero-shot results on Ref-DAVIS17, leveraging the Wan2.1 backbone.

### Strengths
- The paper presents a novel perspective by formulating video segmentation as video sequence -> mask sequence via an ODE, thereby directly leveraging the capabilities of modern video generation models.

- The overall exposition is clear and logically structured, making the method easy to follow. The design choices appears to be reasonable.

- Experiments demonstrate that harnessing video generation models can be effective for RVOS. The findings are interest and could be related to recent work on zero-shot visual understanding with video generators (e.g., studies on Veo3 [1]), further highlighting emergent visual understanding abilities inside video generation models.

### Weaknesses
- The manuscript claims (L493) to provide “detailed implementation details and hyperparameters for reproduction,” yet the main text (and appendix) gives very little in the way of concrete implementation. In particular:

(a) how the three alternative paradigms are instantiated; and

(b) the formal specification and practical realization of the three proposed improvements.

This lack of detail undermines completeness and reproducibility. Please see “Questions” for specific questions.

- Although RVOS accuracy improves, the paper does not report or discuss the impact on inference speed and computational cost when introducing a video generation model into the segmentation pipeline.

### Questions
- How is the VAE fine-tuned in practice? Is it fine-tuned on the Video-frames→Mask task, or purely as an autoencoder on mask images?

- For the alternative paradigms: how concretely are the direct mask prediction (and velocity prediction) variants based on Wan implemented? Are these obtained by supervised post-training on Wan2.1?

- Regarding the SPA modification (L247): “we transform the initial video latent $z_0$ through a stochastic encoding and normalization process.” What exactly is the implementation of this stochastic encoding and normalization? How sensitive are the results to the type and degree of stochasticity?

- Table 3 suggests a fine-tuned VAE decoder is crucial. In Table 2, did all underperforming alternative paradigms also use a fine-tuned VAE decoder? If yes, please state this explicitly; if not, could their poorer performance are due to the lack of VAE fine-tuning?

- In Table 2, the final performance drops when BBS = 0.75. Do you have explanations or analysis for this behavior?

References
[1] Wiedemer, Thaddäus, et al. “Video models are zero-shot learners and reasoners.” arXiv:2509.20328 (2025).

### Soundness
3

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
This paper proposes FlowRVS, a method that formulates the Referring Video Object Segmentation (RVOS) task as a conditional continuous flow problem, enabling the use of pretrained text-to-video (T2V) models for RVOS. The method achieves state-of-the-art performance on major RVOS benchmarks.

### Strengths
- The paper is well written in the introduction, which provides a clear motivation and overview of the problem.
- The experimental results are strong, showing state-of-the-art performance across standard RVOS benchmarks.
- The proposed approach leverages multiple strategies to effectively adapt a pretrained T2V model to the RVOS problem.

### Weaknesses
- Several claims are inaccurate or overstated. For example, in Lines 59–64, the paper suggests that "locate-then-segment" is the dominant paradigm. In fact, it is only one of several common approaches. Moreover, language queries are not often decoupled; many recent methods (e.g., ReferFormer) utilize textual features throughout the entire temporal segmentation process, applying cross-attention between text and video features rather than limiting language conditioning to the localization stage.
- The idea of applying T2I/T2V generation to RVOS is not novel. Prior work such as VD-IT [ECCV 2024] has already introduced the use of pretrained text-to-video diffusion models for RVOS.
- The training setup is inconsistent across datasets and lacks a unified protocol or rationale (L316-317), making fair comparison and generalization less convincing.
- The technical descriptions are insufficiently detailed, particularly for modules such as BBS, SPA, and DVI (Page 5). As currently written, the paper would be difficult to reproduce.
- There is no evaluation on Long-RVOS benchmark, leaving uncertainty about how the generative method performs on long-term temporal reasoning or consistency over extended sequences.
- The paper does not include failure case analysis or discussion of limitations, which would help clarify the method’s robustness and boundaries of applicability.

### Questions
See weakness part.

There is a concurrent work titled “Temporal-Conditional Referring Video Object Segmentation with Noise-Free Text-to-Video Diffusion Model.” It would be helpful if the authors could compare their approach with this work, highlighting similarities and key differences in modeling assumptions, architecture, and performance.
This is not a weakness point, but rather a point of curiosity and clarification for understanding how FlowRVS relates to other diffusion-based RVOS methods developed concurrently.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes FlowRVS, a novel framework for Referring Video Object Segmentation (RVOS) that reimagines the task as a text-conditioned continuous flow from video to mask, leveraging the structure of pretrained Text-to-Video (T2V) generative models. Instead of the conventional “locate-then-segment” pipeline—which first grounds language into coarse geometric prompts (e.g., bounding boxes) and then segments—FlowRVS learns a deterministic, end-to-end deformation in latent space that directly maps a video’s spatio-temporal representation to its target segmentation mask under linguistic guidance.
The key insight is that RVOS is a convergent (video → mask) rather than divergent (noise → video) process. To adapt T2V models for this discriminative task, the authors introduce three technical innovations: 

1. Boundary-Biased Sampling (BBS): oversamples early timesteps to stabilize learning of the initial velocity. 
2. Start-Point Augmentation (SPA): regularizes the flow by perturbing the initial latent. 
3. Direct Video Injection (DVI): concatenates the original video latent at every ODE step to preserve global context.

Experiments show state-of-the-art results across major RVOS benchmarks: 
1. MeViS: 51.1 J&F (+1.6 over prior SOTA) 
2. Ref-DAVIS17 (zero-shot): 73.3 J&F (+2.7) 
3. Ref-YouTube-VOS: 69.6 J&F
The approach demonstrates superior handling of complex motion, temporal reasoning, and fine-grained language grounding.

### Strengths
1. Originality: The paper introduces a new problem formulation—RVOS as a text-conditioned flow from video to mask—departing from both discriminative pipelines and generative synthesis. This is more than a technical tweak; it’s a paradigm shift. 

2. Quality: Experimental design is rigorous, with ablations, cross-dataset zero-shot evaluation, and comparison to strong baselines (including VLM+SAM and grounding-model-based methods). 

3. Clarity: Concepts like “convergent vs. divergent flow” are explained intuitively. The three proposed techniques (BBS, SPA, DVI) are clearly motivated and linked to the core challenge. 

4. Significance: Demonstrates that pretrained T2V models can be repurposed for discriminative understanding tasks with proper adaptation—opening a new direction for leveraging generative priors in video understanding.

### Weaknesses
1. Computational cost: The paper does not report inference time or FLOPs. Given the use of an ODE solver with multiple steps and a 1.3B-parameter DiT, it’s unclear whether FlowRVS is practical for real-time applications. A comparison with efficient baselines (e.g., MTTR, ReferFormer) in terms of speed would strengthen the evaluation. 

2. Limited analysis of failure cases: While qualitative results show success on complex queries, the paper lacks discussion of scenarios where FlowRVS fails (e.g., ambiguous language, fast motion, or occlusion). 

3. VAE decoder fine-tuning: The VAE decoder is fine-tuned on MeViS, but it’s unclear how much performance gain comes from this vs. the flow formulation itself. An ablation on decoder adaptation would help isolate contributions.

### Questions
1. Inference efficiency: How many ODE steps are used during inference? Could the method be made faster via learned solvers or step reduction without significant performance drop? 
2. Generalization to unseen verbs/actions: The MeViS benchmark emphasizes motion-centric language. Does FlowRVS generalize to novel action phrases not seen during training (e.g., “the dog somersaulting”)? 
3. Role of T2V pretraining: Would the same gains be achievable with a non-generative video backbone (e.g., ViViT) adapted with your flow framework? Or is the T2V prior essential? 
4. Decoder adaptation: What is the performance drop if the VAE decoder is not fine-tuned on segmentation masks? This would clarify whether gains are due to the flow or just better mask reconstruction.

### Soundness
4

### Presentation
4

### Contribution
4
