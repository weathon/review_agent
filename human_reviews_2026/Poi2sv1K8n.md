# UniVid: The Open-Source Unified Video Model

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 2, 4

## Abstract
Unified video modeling combining generation and understanding capabilities is increasingly important, yet faces two key challenges: maintaining semantic faithfulness during flow-based generation due to text-visual token imbalance and the suboptimality of uniform cross-modal attention across the flow trajectory, and efficiently extending image-centric MLLMs to video without costly retraining. We present UniVid, a unified architecture that couples an MLLM with a diffusion decoder through a lightweight adapter, enabling both video understanding and generation. We introduce Temperature Modality Alignment to improve prompt adherence and Pyramid Reflection for efficient temporal reasoning via dynamic keyframe selection. Extensive experiments on standard benchmarks demonstrate the state-of-the-art performance of our unified video model, achieving a 2.2% improvement on VBench-Long total score compared to the previous SOTA method EasyAnimateV5.1, and 1.0% and 3.3% accuracy gains on MSVD-QA and ActivityNet-QA, respectively, compared with the best prior 7B baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents UniVid, an open-source unified model for both video understanding and generation. It addresses key challenges in hybrid diffusion–autoregressive modeling: (1) semantic drift in text-to-video generation due to imbalanced text-visual attention and timestep-dependent guidance, and (2) inefficient temporal reasoning when extending image-centric MLLMs to video.
UniVid couples a multimodal LLM (MLLM) with a diffusion video decoder via a lightweight adapter. For generation, it introduces Temperature Modality Alignment (TMA)—a timestep-aware schedule that strengthens early semantic guidance and shifts focus to visual refinement later, improving prompt fidelity. For understanding, it proposes Pyramid Reflection, a test-time reinforcement learning loop that uses SigLIP2-based keyframe selection and verbal policy refinement to adaptively expand or prune frame evidence.
Trained in three stages—alignment, understanding fine-tuning, and joint co-training—UniVid achieves state-of-the-art results on VBench-Long for generation and competitive performance on MSVD-QA, MSRVTT-QA, TGIF-QA, and ActivityNet-QA for understanding, all while avoiding costly full-model retraining.

### Strengths
1. The paper is clearly written and provides comprehensive figures and derivations that generally ease understanding.

2. Coupling a frozen MLLM to a diffusion decoder with only a lightweight adapter is a conceptually simple modification; empirical results suggest it works reasonably well without heavy retraining. Ablation studies indicate that both Temperature Modality Alignment and Pyramid Reflection contribute noticeable, albeit incremental, improvements in fidelity and downstream accuracy.

3. On VBench-Long, UniVid achieves competitive overall scores relative to existing open-source models, and the supplied qualitative examples offer some evidence that the gains translate into visibly better temporal consistency and prompt adherence.

### Weaknesses
1. The understanding experiments rely on relatively small or dated benchmarks such as MSVD-QA and MSRVTT-QA, whose questions are often answerable from a single frame. Stronger evidence could be obtained by reporting results on more recent and challenging suites like Video-MME [1] or MMBench-Video [2], which probe long-range temporal reasoning and fine-grained grounding.

2. Although Figure 2 gives a high-level overview, the two main contributions—Temperature Modality Alignment and Pyramid Reflection—are not explicitly highlighted in the diagram or caption. Adding call-outs or color-coded arrows that link each module to its corresponding technical section would help readers grasp the architectural roles at a glance.

3. Several supplementary videos exhibit noticeable artifacts: in the stormy-seashore clip the lightning pattern remains static across frames, while in the young-woman clip the eyelid region gradually turns into a yellowish streak reminiscent of an eyebrow. These imperfections suggest that further calibration of the temporal consistency loss or additional post-processing may be needed before the model can reliably produce visually plausible long sequences.

[1] Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis. CVPR 2025

[2] MMBench-Video: A Long-Form Multi-Shot Benchmark for Holistic Video Understanding. NeurIPS D&B Track 2024

### Questions
1. It would be helpful to see an ablation experiment in which the VAE encoder is completely removed from the generation branch while being retained in the understanding branch. Such a controlled comparison could clarify how much the fine-grained pixel-level information supplied by the VAE latent code contributes to generation fidelity, and, conversely, how heavily the understanding task relies on these dense pixel cues versus the high-level ViT features alone. 

2. All reported generation clips are short (~4s). Have you sampled 8-30 second videos on the same prompts? If temporal consistency degrades, please state the failure mode (e.g., object duplication, static background) so that readers know the effective usable length.

3.  Pyramid Reflection relies on a verbal “refined query” produced by an LLM reflector. How much does performance vary if the reflector is replaced by a deterministic template or by a smaller LM?

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
4

### Summary
This paper propose UniVid, a unified architecture that couples an MLLM with a diffusion decoder through a lightweight adapter, enabling both video understanding and generation. This paper propose Temperature Modality Alignment to improve prompt adherence and Pyramid Reflection for efficient temporal reasoning via dynamic keyframe selection.

### Strengths
1. The paper proposes a training-efficient pipeline for adapting image-centric MLLMs to a unified setting for video understanding, generation, and editing.

2. The manuscript is clearly written and easy to follow.

3. The authors have open-sourced the code, which lays a solid foundation for further research in this area.

### Weaknesses
1. The current evaluation relies on only four video QA benchmarks, with one being in-domain. This limited scope may not fully demonstrate UniVid’s video understanding capabilities. Consider adding more comprehensive benchmarks (e.g., Video-MMLU, Video-MMMU) and comparing against stronger, up-to-date baselines such as Qwen3.

2. The paper claims to efficiently extend image-centric MLLMs to videos without costly retraining. To support this, the authors should compare UniVid with methods that require additional fine-tuning (e.g., Omni-Video), highlighting both performance and efficiency trade-offs.

3. UniVid uses only 20K samples from ActivityNet-QA yet reports strong results on four benchmarks. It would help to clarify whether the model is sufficiently trained for the target domains and not overfitting. Providing the fine-tuning loss curve (and ideally validation metrics over time) for the video understanding stage would strengthen the evidence of convergence and generalization.

### Questions
Refer to weakness

### Soundness
3

### Presentation
3

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
The paper propose a unified framework for video  generation and understanding. It design a light adapter to connect an MLLM and a diffusion model serving as a decoder.  Besides, it adjusts attention score in MM-DiT to enhance generation and utilize two strong LLMs to conduct the test-time scaling pipeline.

### Strengths
1. According to the characteristics of different levels of information at different time steps in the diffusion process,  this paper adjusts the attention intensity between vision and text in the attention score, and alleviates the problem of text suppression in MMDIT.
2. The model achieves SOTA video generation performance at VBench.

### Weaknesses
1. Unified modeling of generation and understanding typically requires a single model to learn both understanding and generation capabilities simultaneously. However, this paper merely uses an adapter to connect pre-trained, independent understanding and generation models. Therefore, I question the significance of the paper's motivation.
2. For the generation part, it seems to simply pass the text embedding to the diffusion model to perform the text-to-video task, without demonstrating the interaction between semantic understanding features and video generation features. It seems to primarily relies on the generation capabilities of Wan2.2 5B itself.
3. For the understanding part, the test-time reinforcement learning primarily relies on two large LLMs to act as evaluator and reflector, respectively. Using two models with hundreds of billions params to enhance the video understanding capabilities of a 7B model seems unreasonable and increases inference costs.

### Questions
the same as Weakness part.

### Soundness
2

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
This paper introduces UniVid, a novel architecture that unifies video generation and understanding within a single hybrid framework. The core design couples a Multimodal Large Language Model (MLLM) with a diffusion-based video decoder via a lightweight conditioning adapter. The authors propose two key technical innovations: Temperature Modality Alignment (TMA), which addresses the imbalance and time-dependent role of text versus visual conditioning in diffusion models, and Pyramid Reflection, an efficient method for adapting image-centric MLLMs to video understanding through iterative, query-driven keyframe selection.
The proposed solution demonstrates strong performance, achieving superior results in video generation against several contemporary models (e.g., HunyuanVideo, Gen-3) and competitive results in Video Question Answering (QA) against other 7B-parameter models.

### Strengths
1. Successful Unification: The primary strength of UniVid is its effective integration of video generation and understanding into a single, synergistic framework.
2. Novel Technical Contributions: The introduction of TMA and Pyramid Reflection represents a meaningful contribution. The ablation studies convincingly demonstrate that both components are integral to the model's performance.
3. Comprehensive Evaluation: The paper provides a thorough experimental evaluation, validating performance on established benchmarks for both generation (VBench) and understanding (multiple Video QA datasets).

### Weaknesses
1. Architectural Novelty: The overall architecture is more incremental than revolutionary, as it primarily integrates several existing, powerful open-source components (e.g., Wan 2.2, Qwen, SigLIP) via a trainable adapter. The novelty lies in the specific integration method and the proposed sub-modules (TMA, Pyramid Reflection).
2. Limited Comparative Analysis: The choice of baseline models for video generation could be more comprehensive. For instance, a direct comparison with the base Wan 2.2 model itself is omitted, making it difficult to isolate the performance gain attributable to UniVid's conditioning mechanism over the underlying decoder. Comparisons with other recent, high-performing open-source models are also lacking.
3. Limitations of Keyframe Approach: The Pyramid Reflection method relies on a sparse set of keyframes. The paper does not adequately explore the inherent limitations of this approach compared to methods that process the full, dense temporal context of a video, which may be superior for tasks requiring fine-grained motion understanding.

### Questions
1.  The TMA mechanism uses a specific, heuristic formula for its scheduling function. The paper would be strengthened by providing more theoretical justification or an ablation study exploring different scheduling functions to justify the chosen formulation.

### Soundness
3

### Presentation
3

### Contribution
2
