# HyperMotion: DiT-Based Pose-Guided Human Image Animation of Complex Motions

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Recent advances in diffusion models have significantly improved conditional video generation, particularly in the pose-guided human image animation task. Although existing methods are capable of generating high-fidelity and time-consistent animation sequences in regular motions and static scenes, there are still obvious limitations when facing complex human body motions that contain highly dynamic, non-standard motions, and the lack of a high-quality benchmark for evaluation of complex human motion animations. To address this challenge, we propose a simple yet powerful DiT-based video generation baseline and design spatial low-frequency enhanced RoPE, a novel module that selectively enhances low-frequency spatial feature modeling by introducing learnable frequency scaling. Furthermore, we introduce the Open-HyperMotionX Dataset and HyperMotionX Bench, which provide high-quality human pose annotations and curated video clips for evaluating and improving pose-guided human image animation models under complex human motion conditions. Our method significantly improves structural stability and appearance consistency in highly dynamic human motion sequences.  Extensive experiments demonstrate the effectiveness of our dataset and proposed approach in advancing the generation quality of complex human motion image animations. The codes and dataset will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper targets the challenge of pose-guided portrait animation under complex human motions (e.g., flips, stunts) and proposes a concise DiT paradigm: it composes in latent space to inject both pose video and reference image, and uses a first-frame mask to suppress identity leakage. The core modification, SLF-RoPE, selectively enhances the low-frequency channels of RoPE along the spatial dimension, while learnable motion/space scaling factors improve global structure and identity stability under high-speed, nonlinear motion. Based on MotionX, the authors construct the Open-HyperMotionX dataset (automatically mining complex motion segments via wavelet energy, with OCR debiasing and caption cleaning) and release HyperMotionX Bench with 100 high-quality pose annotations (using XPose, removing unreliable hand keypoints in complex frames). Trained on the Wan2.1 backbone, the method achieves leading structural consistency (PCK), competitive pixel/perceptual/temporal metrics, and VBench-I2V gains in background/overall consistency and motion smoothness. Ablations show SLF-RoPE effectively reduces artifacts under extreme poses, and inference runs on a single 3090. Limitations include limited theoretical novelty, insufficient analysis of sensitivity to α/γ and the dynamics of the proposed scaling strategy, coarse-grained ablations, heavy reliance on XPose and hand keypoint quality, and incomplete evaluation for multi-person scenes, strong camera motion, and long sequences. The authors plan to open-source code, data, and evaluation, indicating solid engineering practicality and potential impact.

### Strengths
1. Method is simple yet targeted: latent-space composition in a DiT and a first-frame mask suppress identity leakage; SLF-RoPE selectively boosts spatial low-frequency channels and introduces learnable motion/space scaling, significantly improving global structure and identity stability under high-speed, non-linear motion.
2. Significant data and evaluation contributions: constructed Open-HyperMotionX (automatic mining of complex motions, OCR debiasing, and caption cleaning) and released the high-quality HyperMotionX Bench (100 sequences with pose annotations), providing practical training and evaluation resources for complex motion scenarios.
3. High engineering practicality: inference runs on a single 3090, with minimal modifications and easy integration into backbones like Wan2.1; plans to open-source code, data, and evaluation, facilitating community reproduction and extension.

### Weaknesses
1. Limited methodological novelty: The core modification (SLF-RoPE) scales the low-frequency band of RoPE’s frequencies. The idea is intuitive and engineering-oriented, but lacks theoretical depth and a justification of generality.

2. I would like to see more visual results for this task, especially hand details (which are one of the core aspects). Please provide more examples focusing on hand motion.

3. The sensitivity of the low-frequency ratio α and the scaling factor γ, as well as the learning dynamics of the motion scale and space scale under different motion intensities, remain unclear. How “motion intensity” is estimated and the consistency between training and inference phases are not elaborated.

4. The paper only provides three comparisons (with/without SLF-RoPE and with/without dataset training). More fine-grained ablations are desirable (e.g., scaling H only or W only, fixing γ without dynamic modulation, using different α partitioning strategies, etc.).

5. There is heavy reliance on XPose for pose annotations; hand keypoints are removed in complex segments. While this is reasonable, it limits the evaluation and optimization of fine-grained hand motion generation.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the critical challenge of generating high-fidelity, temporally consistent human image animations for complex human motions (e.g., stunts, acrobatics), where existing methods typically fail due to structural degradation and appearance inconsistency. The authors propose a concise yet powerful DiT-based human animation baseline and introduce a novel component: Spatial Low-Frequency Enhanced Rotary Positional Embedding (SLF-ROPE), which selectively amplifies low-frequency spatial features to improve global structure and appearance fidelity. Furthermore, the paper contributes the Open-HyperMotionX Dataset and HyperMotionX Bench, a valuable resource for training and evaluating models specifically on complex human movements. Extensive experiments demonstrate significant improvements over the state-of-the-art.

### Strengths
-	This paper addresses the critical challenge of generating high-fidelity, temporally consistent human image animations for complex human motions, and presents SLF-ROPE.
-	Experimental results indicate the effectivenss of the proposed method.

### Weaknesses
-	Incremental Architecture: While the DiT-based architecture is effective, it is described as a "simple DiT-based baseline" built upon existing models (Wan2.1). The core novelty is concentrated in the SLF-ROPE module and the data/benchmark. A deeper analysis or discussion on why this specific DiT architecture is superior for complex motions beyond the added SLF-ROPE would strengthen the paper.
-	The Lecun ID in the supplementary material is not maintained well. Could the authors explain the reasons?
-	The paper notes that existing human pose estimation methods often fail on complex motions, motivating the new high-quality dataset. While this solves the problem for training/evaluation, a discussion on how the inference stage would be impacted if only low-quality poses were available from external methods would be beneficial, or a demonstration that the robust SLF-ROPE can better handle noisy pose inputs.

### Questions
Can the proposed SLF-ROPE apply to other general video generation tasks?

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
This paper introduces a Diffusion Transformer (DiT) based framework for pose-guided human image animation, specifically targeting complex, high-dynamic "Hypermotions." The core novelty lies in the Spatial Low-Frequency Enhanced Rotary Positional Embedding (SLF-ROPE), designed to mitigate structural degradation during extreme movements. The authors also contribute a new dataset and benchmark, Open-HyperMotionX and HyperMotionX Bench. The proposed method shows impressive qualitative results and strong quantitative performance. The idea of explicitly addressing the stability issue in complex movements is valuable. However, significant concerns regarding the fairness of the comparison and the evaluation protocol must be addressed.

### Strengths
1.	The paper correctly identifies a major failure mode of current human animation models and discovers their inability to maintain fidelity and consistency during complex, non-standard movements. The goal is clear and impactful.
2.	SLF-ROPE and a new benchmark named Open-HyperMotionX are proposed.
3.	Establishing a new benchmark for complex motion is necessary to drive future research in this domain.

### Weaknesses
1.	The primary concern is the potential unfairness of the comparison. Existing state-of-the-art methods are generally trained on large, general-purpose datasets (e.g., videos of daily life, simple movements), which predominantly feature simple motions. The proposed method is specifically trained on the newly introduced HyperMotionX dataset, which focuses on complex motions. If the videos used for evaluation (e.g., the supplementary videos) share a similar distribution or style to the videos in the HyperMotionX training set, the comparison is severely biased. The proposed method is essentially specialized for the test domain, while the baselines are being tested out-of-distribution (OOD) for complex motions.
2.	The paper focuses heavily on complex motions. It is essential to demonstrate that the SLF-ROPE modification does not degrade performance or introduce artifacts when applied to the simple, common motions where existing methods already perform well. A comprehensive evaluation on a standard, general animation benchmark (e.g., TikTok, PATD) is mandatory.
3.	The quality of the supplementary material is inconsistent. The IDs in the supplementary material is not consistently preserved well.

### Questions
In Table 3, the proposed method does not achieve best performance in terms of FID, VFID, and FVD. Could the authors explain this phenomenon?

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
3

### Summary
This paper addresses the challenge of pose-guided human image animation under complex and highly dynamic motions. The authors propose a DiT-based human animation baseline featuring a novel spatial low-frequency enhanced RoPE module that improves low-frequency spatial feature modeling through learnable frequency scaling. To robustly evaluate how well models can perform for highly dynamic motions, the authors curate the Open-HyperMotionX Dataset and HyperMotionX Bench, which provide high-quality human pose annotations for video clips of complex motion. Experiments show that the proposed model achieves better structural stability and appearance consistency in dynamic motion sequences, demonstrating the combined effectiveness of the new dataset and architectural enhancements.

### Strengths
- The work fills an important gap by focusing on complex, highly dynamic human motions that existing models struggle to handle.
- The paper clearly identifies the limitations of prior models in capturing low-frequency spatial details and introduces spatial low-frequency enhanced RoPE as a targeted enhancement that is validated through ablation studies.
- The proposed benchmark and dataset are well-curated, clearly documented, and demonstrate advantages in evaluation, providing a valuable resource for future research in pose-guided human animation.

### Weaknesses
- The impact of the dataset could be analyzed more extensively. The paper primarily compares HyperMotion vs. Wan's model performance on VBench before and after training on the Open-HyperMotionX, but further evaluation on finetuned versions of other baselines models could strengthen the assessment of benefits for the constructed dataset. If computationally feasible, such comparisons could better isolate the dataset’s contribution from architectural effects.

- The analysis of limitations of current models on clean but hard pose sequences is insightful and motivates the design of low-frequency detail modeling in positional embeddings. However, the performance improvements of the HyperMotion model on the HyperMotionX Benchmark are not consistently significant, even though the model is trained on Open-HyperMotionX, potentially giving it an advantage over other models. This suggests that the model architecture may still have room for improvement.

### Questions
- During training, the reference image appears to be randomly selected. Does this imply that during inference, the reference image does not necessarily need to correspond to the first frame of the motion sequence? If so, how sensitive is model performance to this choice of reference image?

- Regarding the wavelet-based clip extraction, how are long dynamic motions, such as dancing or motions with buildup phases (e.g., running before a long jump), handled? Are these sequences split into multiple shorter segments, and if so, does this segmentation risk disrupting motion continuity or long-term temporal patterns during generation?

### Soundness
3

### Presentation
3

### Contribution
3
