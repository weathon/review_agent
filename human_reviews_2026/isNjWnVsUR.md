# Anchor Frame Bridging for Coherent First-Last Frame Video Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
First-last frame video generation has recently gained significant attention. It enables coherent motion generation between specified first and last frames. However, this approach suffers from semantic degradation in intermediate frames, causing scene distortion and subject deformation that undermine temporal consistency.
 To address this issue, we introduce Anchor Frame Bridging (AFB), a novel plug-and-play method that explicitly bridges semantic continuity from boundary frames to intermediate frames, offering training-free adaptability and generalizability. By adaptively interpolating anchor frames at temporally critical locations exhibiting maximal semantic discontinuities, our approach effectively mitigates semantic drift in intermediate frames. Specifically, we propose an adaptive anchor frame selection module, which generates text-aligned candidate frames via frame order reversal and selects anchors based on semantic continuity. Subsequently, we develop anchor frame guided generation, which leverages the selected anchor frames to guide semantic propagation across intermediate frames, ensuring consistent boundary semantics and preserving temporal coherence throughout the video sequence. The final video is synthesized using the first frame, last frame, selected anchor frames, and the text prompt.
 The results demonstrate that our method significantly enhances the temporal consistency and overall quality of generated videos. Specifically, when applied to the Wan2.1-I2V model, it yields improvements of 16.58\% in FVD and 10.21\% in PSNR. The codes are provided in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the challenge of semantic degradation and temporal inconsistency in first-last frame video generation using diffusion models. The authors propose the Anchor Frame Bridging (AFB) method, which automatically selects and inserts an adaptive anchor frame at points of maximal semantic discontinuity in the intermediate video. Their approach consists of an adaptive selection mechanism using LPIPS-guided frame evaluation and a guided generation process augmenting the standard pipeline with anchor frames. The method is designed as a plug-and-play module, requiring no additional training and is validated on newly curated benchmarks and multiple image-to-video diffusion backbones.

### Strengths
- **Clear Targeted Problem**: The paper identifies and clearly explains a critical shortcoming in current first/last-frame-to-video models—semantic drift and abrupt transitions in intermediate frames (see intro, Fig. 1).
- **Methodological Innovation**: AFB uses a reverse-generation heuristic and LPIPS-based metric to localize semantic discontinuities and insert an anchor frame, effectively bridging boundary semantics throughout the sequence (Method Sec. 3, Eqns p4–p5, Fig. 2).
- **Empirical Rigor**: The authors provide both qualitative and quantitative benchmarks, comparing AFB-augmented models against strong baselines (including Wan2.1, Hunyuan Video, ViBiDSampler, Generative Inbetweening) on a purpose-built test set.
- **Comprehensive Ablation**: Multiple dimensions (number/position of anchors, prompt variants, effect of diffusion steps) are dissected with quantitative evidence (Tables 1, 2, Fig. 5a).

### Weaknesses
- **Limited Theoretical Insight into Anchor Utility**: 

While the method for selecting anchor frames is well-motivated empirically, there is only an informal rationale for why mirroring the LPIPS-based breakpoint in a reverse diffusion pass is optimal. The mathematical treatment (Sec. 3.2, Eqns p5) relies on an empirical proxy rather than theoretical guarantees. This leaves open whether the anchor truly maximizes semantic continuity for all types of motion or scenes, or merely correlates empirically. A more thorough theoretical foundation is desirable to formally ground the anchor selection beyond heuristic evidence.

- **Complexity in Hyperparameter Selection and Heuristic Choices**: 

The anchor frame is chosen based on the peak of a smoothed local LPIPS curve at a fixed (empirically chosen) denoising timestep (Fig. 5b). But the paper provides limited discussion of sensitivity to these hyperparameters (e.g., LPIPS window size, selected denoising phase, anchor mirroring formula). There is a risk that such choices may not generalize, or that their necessity complicates adoption.

- **Visual Quality Trade-offs in Extreme Cases**: 

While the reported objective and subjective gains are strong (Table 1), failure cases and remaining artifacts (Appendix D, Fig. 11) suggest that AFB does not fully resolve physical implausibility or distortions in highly dynamic settings. This is acknowledged, but could be further emphasized as a hard limitation.

### Questions
- Could the authors clarify if any formal guarantees or theoretical results support the observed empirical optimality of the mirrored LPIPS-breakpoint approach over alternative anchor selection strategies or more adaptive criterion? What failure cases or counterexamples have you observed?

- Can you elaborate on the impact and selection process for the denoising timestep at which LPIPS is computed, the window width for local averaging, and any anchor selection heuristics?

### Soundness
3

### Presentation
2

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
This paper addresses semantic degradation and temporal inconsistency in first-last frame video generation (FLF2V), where models must generate a video given only the start and end frames. The authors identify "information attenuation" as a key issue, where semantic guidance from the boundary frames weakens towards the middle of the sequence. To solve this, the paper introduces Anchor Frame Bridging (AFB), a novel, plug-and-play, and training-free method. AFB operates in two stages: The method first performs a full *reverse* generation pass (from last frame to first frame) to create a "candidate set" of frames. It identifies the frame with the maximum temporal incoherence (peak LPIPS) in this reversed sequence at a normalized position $\alpha$. It then selects an anchor frame from the *mirrored* position ($1-\alpha$) of this candidate set. The final video is synthesized by conditioning the base I2V model on the first frame, the last frame, and this new anchor frame at its designated position $\alpha$, using an indicator mask to guide the diffusion process. The authors created a new dataset of 436 video pairs and applied AFB to two base models (Wan2.1, Hunyuan Video). Results show quantitative improvements (e.g., 16.58% in FVD on Wan2.1-I2V) and improved qualitative coherence in a user study.

### Strengths
- The core idea of AFB is novel and clever. Using a reversed-generation pass to identify a point of failure ($\alpha$) and then applying a "mirror position" heuristic ($1-\alpha$) to select a high-quality anchor frame is an elegant, non-trivial solution to the problem of *what* to use as an anchor and *where* to place it.
- The method is training-free and plug-and-play. This is a significant practical advantage, allowing it to enhance existing I2V models for the FLF2V task without any costly fine-tuning.
- The paper is very well-written and easy to follow. Figure 1 and 2 provide an excellent, intuitive visualization of both the "information attenuation" problem and the proposed two-stage solution.
- The authors validate their method on two different base models and use a comprehensive set of metrics (FVD, LPIPS, PSNR/SSIM, MLLM scores, and a user study) to build a robust case for the method's effectiveness. The ablation studies (e.g., on anchor frame count in Fig. 5a) are valuable and clearly justify the final design.

### Weaknesses
1. The most significant weakness is the computational overhead. The "Adaptive Anchor Frame Selection" stage requires a full, additional reverse generation pass (e.g., 50 diffusion steps) just to find the anchor frame. As shown in Table 3, this effectively doubles the inference time (e.g., from 20 to 41 minutes for Wan2.1). This high cost severely limits the method's practical utility.
2. The ablation study (Fig. 5a and user study in Fig. 12) reveals that using a single anchor frame ($k=1$) is optimal, and $k>1$ degrades performance due to "conflicting guidance." This is a major limitation. It strongly suggests the method is brittle and cannot scale to longer, more complex videos with multiple distinct actions or scene changes, which would naturally require multiple anchor points to maintain coherence.
3. The entire anchor selection mechanism (finding max LPIPS in the reverse pass and mirroring the position $\alpha \rightarrow 1-\alpha$) is a clever heuristic, but it is not a learned or guaranteed-optimal strategy. It relies on an implicit assumption that the motion and degradation are somewhat symmetrical, which may not hold for many real-world scenarios (e.g., a video of a car accelerating and then coasting).
4. The paper positions AFB as a "plug-and-play" method. However, the experimental comparison in Table 1 is primarily against the base models themselves (Wan2.1-I2V, Hunyuan-I2V) or fully-trained methods (Wan2.1-FLF2V). The comparison lacks other relevant *training-free* or *guidance-based* video interpolation or editing methods that could also be considered "plug-and-play." This makes it difficult to assess AFB's performance relative to its true peers.
5. While the creation of a new dataset is commendable, the size (N=436) is very small for a video generation benchmark. This limited diversity makes it difficult to assess generalizability and raises concerns that the method's findings (especially the $k=1$ optimality) might be an artifact of the specific data distribution (e.g., short, single-action clips).

### Questions
1. The doubled inference time is a major drawback. Have the authors explored cheaper approximations for anchor selection? For instance, Figure 5b shows LPIPS is unstable in early timesteps. Could a different, more stable metric (e.g., CLIP feature distance) reliably estimate the failure point $\alpha$ at a much earlier, cheaper timestep (e.g., $t=10$ or $t=20$) instead of requiring a near-full reverse pass ($t=45$)?
2. The finding that $k>1$ anchor frames degrades performance is counter-intuitive and the most significant concern for scalability. Does this imply a fundamental limitation in the "anchor frame-guided generation" step, which cannot resolve conflicting semantic guidance from multiple anchors? How do you envision this method scaling to longer videos (e.g., 300 frames) where a single anchor is clearly insufficient?
3. $\alpha \rightarrow 1-\alpha$ "mirror" heuristic is clever. But what are its failure modes? Have you analyzed cases where the motion is asymmetric (e.g., "object appears" vs. "object disappears")? In such cases, the point of max LPIPS in the reverse pass might be semantically unrelated to the *ideal* anchor location for the forward pass.
4. The failure case analysis in Appendix D.1 is appreciated and shows failures inherited from the base models. Have you identified any failure cases *introduced* by AFB itself? That is, an example where the base model produces a coherent video, but the insertion of the selected anchor frame (which comes from a different generation pass) *causes* a new temporal break, artifact, or semantic mismatch?
5. Related to Weakness #5, is it possible that the $k=1$ optimality is an artifact of your small dataset? How would you expect this to hold on a more diverse, large-scale benchmark with more complex, multi-stage motions?

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
5

### Summary
The paper proposes Anchor Frame Bridging (AFB), a training-free, plug-and-play method to address semantic degradation and temporal inconsistency in first-last frame video generation (FLF2V). AFB consists of two core modules: (1) adaptive anchor frame selection, which uses reverse generation (swapped first/last frames + Qwen-generated prompts) and LPIPS-based quality scoring to identify high-semantic-consistency anchors; (2) anchor-guided generation, which injects anchors into the video sequence to propagate boundary semantics. The authors construct a 436-pair FLF2V dataset, validate AFB on Wan2.1 and Hunyuan Video models, and demonstrate improvements (16.58% FVD, 10.21% PSNR on Wan2.1-I2V) via quantitative metrics, MLLM evaluations (GPT-4o, Gemini), user studies, and qualitative comparisons.

### Strengths
1. Practical, training-free design: AFB avoids the high computational cost of retraining large video models, acting as a plug-and-play enhancement—critical for real-world FLF2V applications (e.g., video editing) where retraining is infeasible.

2. Rigorous validation: Results are supported by multi-faceted evaluation: standard metrics (FVD, PSNR, LPIPS), MLLM-based semantic assessment, user studies (52 participants), and ablations (anchor count, diffusion timesteps, prompt quality)—ensuring robustness.

3. Targeted problem diagnosis: The paper clearly links temporal inconsistency to sparse inter-frame attention (Fig.1a) and semantic drift, with visualizations (Fig.6,7) that directly demonstrate AFB’s ability to bridge boundary-to-intermediate semantics.

### Weaknesses
1. Significant computational overhead: AFB doubles inference time (e.g., Wan2.1: 20→41 min; Table 3) due to the reverse pass for anchor candidates. The paper provides no roadmap for optimization (e.g., truncated denoising, lightweight anchor generation) to mitigate this..

2. Limited edge-case improvement: AFB still struggles with extreme scenarios (non-rigid motion, severe occlusions; Fig.11) and relies on better base I2V models to resolve these—there is no exploration of extending AFB (e.g., dynamic anchor counts, motion-aware selection) to proactively address these failures.

3. Missing references and comparisons for first-last frame consistency works: The paper’s related work (Sec. 2) and experiments (Sec. 4) fail to reference or compare with [1] and [2]—works that presumably address the same first-last frame consistency problem. Without detailing how AFB’s technical approach (e.g., reverse generation, anchor selection) differs from [1]/[2] or how its performance (e.g., FVD, visual coherence) stacks against them, the paper’s positioning as a competitive solution for FLF2V is weakened.

4. Limited visual quality in supplementary materials: The supplementary material presents generated videos with limited quality, and the paper provides no direct comparison showing AFB outperforms [1] or [2] in visual fidelity. This lack of side-by-side visual evidence—critical for assessing FLF2V performance—makes it hard to verify AFB’s claimed advantages over these existing methods.

[1] Zhu T, Ren D, Wang Q, et al. Generative inbetweening through frame-wise conditions-driven video generation[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 27968-27978.

[2] Zhang G, Zhu Y, Cui Y, et al. Motion-aware generative frame interpolation[J]. arXiv preprint arXiv:2501.03699, 2025.

### Questions
Does the authors have plans to optimize the reverse pass—such as reducing denoising steps for anchor candidate generation or reusing latent features from the forward process—to lower latency, while ensuring the selected anchor frames still maintain high semantic consistency with the first/last frames?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents Anchor Frame Bridging (AFB), which is an innovative method to improve first-last frame video generation (FLF2V). The approach addresses a major issue: ensuring that all the frames between the first and last are consistent both in terms of their semantic content and the timing of events. The key idea is to insert a high-quality "anchor frame" that acts like a bridge in between the start and end frames. This helps reduce distortion and keeps the motion smooth. The authors show through their experiments that their method improves the quality and consistency of videos better than existing advanced models on their built datasets.

### Strengths
+ The reverse generation process that selects the anchor frame is a smart way to tackle the problem of losing information as frames are generated.
+ This method significantly boosts video quality and consistency compared to advanced baseline models, backed by various metrics and thorough evaluations, including user feedback.
+ Since it's a plug-and-play module, it can be added to existing I2V diffusion models without needing any retraining, making it a very practical and broadly applicable tool.

### Weaknesses
+ Physical Causality Issues: The method generates frames in reverse order (last to first), but the model was trained with forward-time rules. This may create inconsistencies in scenarios with strong physical dynamics. For instance, reversing the video of ripples created by a stone drop doesn’t correctly model the physical event of ripples contracting to eject a stone. Although the authors’ datasets might not show this issue, it limits the method's use in situations with clear, irreversible physical laws.
+ Assumption of Symmetrical Nature: The method relies on the idea that the weakest frame in reverse generation lines up symmetrically with the weakest frame in forward generation. This assumption is crucial for the anchor frame placement but is not clearly proven in the paper. It’s possible that complexities in the start and end frames could disrupt this symmetrical failure point.
+ Computational Overhead: The method involves generating frames in two passes, which doubles the time for inference. The authors mention this in Section D.2, but say that this extra time is "acceptable" is arguable. Especially for models like CausVid, which need to operate with low latency and start playback before the video is fully ready. The requirement for a complete reverse-pass generation could undermine these models' real-time capabilities.

### Questions
The same to weaknesses.
1. Physical Causality: Could the authors show examples of scenarios with definite, irreversible physical laws like shattering glass, a splash, or smoke dissipating to better explain where their method works best?
2. Symmetrical Nature: Could there be an analysis or study to back up the symmetrical failure point assumption? Specifically, does this idea stand in different types of motion or when the complexity of the first and last frames vary greatly?

### Soundness
3

### Presentation
3

### Contribution
2
