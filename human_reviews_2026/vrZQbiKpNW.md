# MindShot: Multi-Shot Video Reconstruction from fMRI with LLM Decoding

- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
Reconstructing dynamic videos from fMRI is important for understanding visual cognition and enabling vivid brain-computer interfaces. However, current methods are critically limited to single-shot clips with video-level alignment and reconstruction, failing to address the multi-shot nature of real-world experiences. To bridge this gap, we propose MindShot, a novel shot-level framework that effectively reconstructs multi-shot videos from fMRI via a divide-and-decode strategy. Specifically, our framework consists of three stages: (1) Shot Decomposition: We first identify shot boundaries within fMRI, then decompose the mixed signals into distinct, shot-specific segments. This explicit segmentation serves as the foundation for accurate semantics decoding. (2) Keyframe Decoding: Each segment is decoded into a textual description representing the keyframe of its corresponding shot. (3) Video Reconstruction: The final video is generated from these keyframe captions, effectively mitigating noise from fMRI redundancy. Addressing the critical lack of real data for multi-shot reconstruction, we introduce a large-scale synthetic dataset generated via a novel data augmentation strategy that randomizes scene duration ratios. Experimental results demonstrate our framework outperforms state-of-the-art methods in both single-shot and multi-shot reconstruction fidelity. Crucially, ablation studies confirm the necessity and generalizability of our Shot Boundary Predictor (SBP), where explicit shot-level decomposition significantly improves decoded caption CLIP similarity by 71.8\%, and the SBP yields consistent performance gains when integrated into other state-of-the-art architectures. Moreover, our synthetic data makes the model generalizable to diverse data and has strong zero-shot transferability that effectively bridges the domain gap between synthetic and real fMRI signals. This work establishes a new paradigm for multi-shot fMRI reconstruction, enabling accurate recovery of complex visual narratives through explicit decomposition and semantic prompting.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses a critical limitation in fMRI-to-video reconstruction: the inability to handle multi-shot videos, which are common in real-world visual experiences. Current methods, designed for single, continuous shots, suffer from temporal mixing of neural signals when faced with multiple scenes. To solve this, the authors propose MindShot, a novel, shot-level framework that operates via a "divide-and-decode" strategy. The authors also contribute a large-scale synthesized multi-shot fMRI-video dataset to facilitate this research. Experiments show that MindShot outperforms existing methods in both single-shot and the new multi-shot reconstruction task.

### Strengths
1. The paper is the first to explicitly define and tackle the challenge of multi-shot video reconstruction from fMRI, shifting the paradigm from a coarse video-level approach to a more fine-grained and semantically coherent shot-level analysis. This is a significant and practical step forward for the field.

2. The core technical contribution, a learnable shot boundary predictor that operates directly on fMRI signals to disentangle mixed neural patterns, is a novel and effective idea. Ablation studies convincingly show its critical role in improving reconstruction quality.

3. Recognizing the lack of suitable data, the authors developed a strategy to synthesize a large-scale multi-shot fMRI-video dataset. This is a valuable resource that will enable and encourage future research in this area.

### Weaknesses
1. Limited Novelty in the Decoding and Reconstruction Pipeline: As illustrated in Figure 2, the Stage 2 is conceptually very similar to that of MindVideo. Both frameworks employ a form of multimodal contrastive learning to align brain signals with text/image representations, predict a caption, and then use a text-to-video model for synthesis. This reliance on an existing paradigm reduces the overall methodological novelty of the paper.

2. Missed Opportunity to Validate the Generalizability of the Core Contribution: The paper's most significant innovation is the shot segmentation module (Stage 1). However, its effectiveness is only demonstrated within the confines of the proposed MindShot framework. A much stronger and more impactful validation would be to treat this module as a general-purpose pre-processing step. I strongly suggest that the authors apply their segmentation module to other leading video reconstruction models (e.g., NeuroClips, Mind-Animator, GLFA) and demonstrate that it provides consistent performance gains. This would prove that the module is a broadly effective and valuable component for the entire field, rather than just a part of one specific pipeline.

3. As a video reconstruction work, the authors should showcase as many dynamic video reconstruction results as possible on the project homepage or in the supplementary materials. This allows readers to intuitively perceive the reconstruction quality, rather than merely presenting a few video frames in the main text. I hope the authors can provide as many additional details as possible in the supplementary materials.

### Questions
1. Incomplete Comparison with State-of-the-Art:  The authors compared far too few baselines on the CC2017 dataset. The experimental table of NeuroClips has documented numerous other baselines, so why didn't the authors include them? Additionally, judging from the results in Table 1, it seems the authors are deliberately downplaying NeuroClips' experimental results. Or is it the case that the authors manually reproduced NeuroClips, and the experimental values are indeed as reported?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces MindShot, a novel framework that pioneers a paradigm shift from video-level to shot-level reconstruction for decoding dynamic visual experiences from fMRI data. To address the critical challenge of temporal signal mixing in multi-shot videos, the method employs a divide-and-decode strategy: it first segments the fMRI signal into clean, shot-specific components using a learnable boundary predictor, then decodes semantic keyframe captions from each segment via an LLM, and finally reconstructs the video by generating individual shots from these text prompts. The work establishes a new benchmark for multi-shot fMRI reconstruction, demonstrating superior fidelity over existing methods and enabling the accurate recovery of complex visual narratives through explicit decomposition and semantic prompting.

### Strengths
Originality:The primary strength is the high originality of proposing a shot-level paradigm. The idea of explicitly decomposing fMRI signals before decoding, moving beyond video-level alignment, is a novel and creative formulation of the problem.

Quality:The technical approach is methodical, and the experimental evaluation is comprehensive. The inclusion of ablation studies strengthens the paper by quantitatively validating the importance of key components like the shot boundary predictor.

Clarity:The paper is logically structured and the core ideas are communicated effectively, aided by clear diagrams and visual results.

Significance:The work is significant because it breaks a key limitation of previous models, paving the way for fMRI decoding to handle complex, narrative visual experiences akin to real-world perception. This has implications for both basic neuroscience and the development of more advanced brain-computer interfaces.

### Weaknesses
1、	The reconstruction pipeline relies exclusively on semantic captions decoded from fMRI to condition the text-to-video (T2V) generative model. While this approach effectively ensures high-level semantic consistency, it may introduce a significant limitation: the absence of direct constraints from low-level visual features present in the original fMRI signals. This could result in a loss of perceptual detail and fidelity in the reconstructed videos, as the T2V model is guided solely by textual prompts without grounding in the specific visual nuances encoded in the brain activity

2、	The construction of the synthesized multi-shot dataset involves concatenating fMRI segments from distinct, isolated experimental trials. However, this procedure may not adequately account for the temporal non-independence of fMRI signals, where the hemodynamic response to a given stimulus is influenced by the preceding stimulus history due to effects like adaptation and hysteresis. Consequently, the synthetic data may fail to accurately emulate the true neural signatures elicited during continuous viewing of multi-shot narratives, thereby potentially compromising the ecological validity of the trained model and its generalizability to real-world scenarios

3、	While the ablation for shot segmentation is strong, the analysis of the LLM's role is less deep. The paper would benefit from a more detailed analysis of the quality and potential biases of the LLM-generated captions, and a comparison to simpler methods for generating semantic labels.

### Questions
1、	Given that the model is trained on synthesized data, what is the plan to validate its performance on a true multi-shot fMRI dataset (e.g., from a feature film)? Do you anticipate a significant performance drop due to the potentially more complex transitions in natural videos?

2、	The current shot boundary predictor identifies transitions but how is the duration of each shot determined for the final video reconstruction? Is it solely based on the number of fMRI TRs between predicted boundaries?

3、	In the ablation study on semantics extraction (Table 4), how much of the performance gain is attributable to the shot segmentation versus the use of an LLM? Could a simpler text decoder achieve similar results once the signals are cleanly segmented?

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
The paper deals with video reconstruction from fMRI.
FMRI embedding is aligned with a key frame and keyframe caption pre-trained embeddings,  the learned embedding is passed to LLM that generates a description of the frame. A video diffusion model is conditioned on the precited frame description and fMRI signal to generate videos.
In addition the paper proposes a method to identify scene cuts in the video stimuli from the fMRI signal, the generated video segments use the predicted scene cuts to generate a video per clip segment, without mixing different video segments.

### Strengths
- The reported metrics are better than previous works.

### Weaknesses
- Short clip segments is a property of the experiments and the way data was collected, I don't think they are central to the problem of fMRI video decoding. A video decoding technic can use a tool for identifying scene cuts, but for me it seems like a second order thing, not the core of the problem. I don't think it is justified to devote a whole paper to this issue.
- Other than the scene cuts detection, I don't see how this paper differs significantly from previous works.
There are minor improvements on metrics but I don't see them translate to reconstruction fidelity. Previous works (Fosco et al. (2024), Lu et al. (2025) ) were able to cherry pick a few convincing frames, here even the cherry picked segments don't seem particularly convincing. I am not sure a human evaluation will favor this method over previous works. 

References:
 - Lu, Y., Du, C., Wang, C., Zhu, X., Jiang, L., Li, X., & He, H. (2025). ANIMATE YOUR THOUGHTS: RECONSTRUCTION OF DYNAMIC NATURAL VISION FROM HUMAN BRAIN ACTIVITY.
- Fosco, C., Lahner, B., Pan, B., Andonian, A., Josephs, E. L., Lascelles, A., & Oliva, A. (2024). Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals

### Questions
- How this work significantly differs from previous works, beside the scene cut detection predictor?

### Soundness
3

### Presentation
3

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
This paper proposes MindShot, a framework for reconstructing multi-shot videos from fMRI data. Unlike prior works that decode short, single-shot clips, MindShot introduces a shot-level paradigm that decomposes fMRI signals into semantically coherent segments before reconstruction. The pipeline involves three stages: (1) Shot Decomposition via a bidirectional LSTM-based shot boundary predictor; (2) Keyframe Caption Decoding using an LLM that converts each fMRI shot embedding into a textual keyframe description; and (3) Video Reconstruction using a text-to-video diffusion model conditioned on these captions.

### Strengths
- The paper explicitly addresses multi-shot fMRI–to–video reconstruction, a realistic yet previously overlooked setting. The shift from video-level to shot-level decoding is conceptually meaningful.
- Although synthetic, the attempt to create large-scale multi-shot datasets is valuable for promoting research in this direction.
Clear ablation structure

### Weaknesses
- The “shot boundary predictor” assumes discrete scene transitions in fMRI, but fMRI has low temporal resolution (TR ≈ 2 s), making such segmentation biologically implausible. No neurophysiological evidence or subject-level analysis supports that shot transitions are detectable at this timescale.
- Improvements are small and inconsistent (e.g., SSIM changes are marginal). Metrics like “2-way” and “50-way classification” are unclear proxies for perceptual fidelity, and no statistical tests are provided.
-Claims such as “enabling accurate recovery of complex visual narratives” are not supported by the experiments. The generated videos are coarse and highly text-conditioned, meaning the results largely reflect text-to-video synthesis rather than true neural decoding.
- The LLM is treated as a black box for caption generation from embeddings. There is no description of how fMRI embeddings are tokenized or represented, nor analysis of linguistic accuracy, diversity, or error modes of the decoded captions.
- The paper’s formatting and visual presentation require substantial refinement. Several figures and tables occupy disproportionate space relative to their content. For instance, Table 1 and Figure 3 (page 7) together fill nearly an entire page with large blank margins and inconsistent scaling. The authors shoud adjust the figure–table arrangement, reducing excessive white space, and ensuring uniform font sizes and caption alignment.
- The current “Experimental Setting” section provides only high-level descriptions. For reproducibility, more implementation details should be included.
- More qualitative examples, especially for multi-shot scenes showing temporal transitions and per-subject analyses and failure cases should be provided.

### Questions
- How biologically plausible is shot boundary detection in fMRI given TR-level temporal blurring?
- How are fMRI embeddings fed into the LLM (are they projected to token embeddings, or concatenated as continuous vectors)?
- Since datasets are fully synthetic, does the model generalize to real continuous-viewing fMRI data?
- How robust is the method to noisy or incorrect shot segmentation?

### Soundness
3

### Presentation
2

### Contribution
2
