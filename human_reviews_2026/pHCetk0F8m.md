# GenFaceTalk: Generalizable One-Shot Talking-Head Generation for Diverse Styles

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
One-shot 3D talking-head synthesis aims to generate realistic 3D facial animations from a single portrait image, driven by audio or video inputs. While recent advances in 3D-aware generation, particularly 3D Gaussian Splatting, have enabled high-fidelity modeling and real-time rendering, existing methods still struggle with critical challenges: (i) accurate identity preservation without multi-view supervision, and (ii) producing temporally coherent animations free from jitter. We propose GenFaceTalk, a novel end-to-end one-shot 3DGS-based framework supporting both audio- and video-driven scenarios without subject-specific training. The core insight of GenFaceTalk is to directly predict motion-disentangled FLAME parameters from the driving video, distilling the reliance on pre-trained 3D face reconstruction and sliding-window-based smoothing into the encoder during training. This design removes the need for face reconstruction at inference, yielding temporally consistent animation while preserving identity and fine-grained facial details. We further introduce a joint learning strategy that integrates FLAME-based motion priors with hierarchical appearance features from the source, guiding 3DGS learning in a spatially aligned and identity-aware manner.
Our framework generalizes across diverse facial styles, including artistic and animal faces.
Experiments demonstrate that GenFaceTalk outperforms state-of-the-art baselines in visual fidelity, temporal stability, identity preservation, and cross-domain generalization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes GenFaceTalk, a one-shot 3D Gaussian Splatting (3DGS) talking-head system driven by audio or video, without subject-specific fine-tuning. Its core idea is to directly regress FLAME parameters from driving inputs via an image/audio encoder and fuse them with hierarchical identity features to animate a unified 3D Gaussian avatar.

### Strengths
- Clear problem importance and strong motivation for one-shot 3D talking-head synthesis.
- Single model supports both motion modalities with consistent 3D reconstruction and animation quality.
- Improves fidelity, temporal metrics, and identity consistency on VFHQ and HDTF datasets.

### Weaknesses
- Identity modeling follows GAGAvatar-like 3DMM-guided Gaussian lifting with hierarchical appearance fusion (Sec. 3.2). The contribution appears as an incremental combination of existing building blocks (3DGS + FLAME + pose substitution), rather than a fundamentally new formulation.
- It is unclear to me whether the system can be regarded as fully end-to-end, given that training still uses offline 3DMM supervision. Could the authors clarify this point?
- Experiments do not fully support strong claims. SyncNet scores on VFHQ underperform EchoMimic despite better FVD (Tab. 1), indicating a lip-sync gap.  Could the authors provide reasons or explanations for this discrepancy.

### Questions
- What exact technical novelty differentiates this from GAGAvatar beyond encoder regression?
- Are audio-driven sync failures due to FLAME-only motion missing lip details?
- How is view-dependent shading handled in 3DGS for large yaw angles?

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
3

### Summary
This paper indicates that the current speaker generation field faces the following issues: Identity preservation challenge, poor temporal consistency, limited generalization ability and other  technical bottlenecks caused by 3D methods. 

Therefore, this paper proposes an end-to-end one-shot framework based on 3D Gaussian Splatting (3DGS), which supports audio-video dual-driven generation and does not require subject-specific training. The framework consists of three core modules: (1) The motion learning module uses a lightweight encoder to directly predict FLAME parameters for decoupled motion from driving signals, obtaining head meshes and camera extrinsic parameters. (2) The identity learning module extracts hierarchical appearance features from the source image and fuses FLAME motion priors to construct a unified identity representation.
The 3DGS rendering module performs voxel splatting and rendering on identity features to generate frames. (3) Additionally, joint optimization with multiple losses is adopted to ensure the quality of generated results.

### Strengths
1.	The proposed end-to-end 3D Gaussian Splatting (3DGS) method is meticulous and reasonable in its overall design, and the rendering results for the head are also relatively satisfactory. I believe that end-to-end methods represent a future direction for the research community.
2.	The writing is clear and easy to understand.

### Weaknesses
1.	There is a lack of comparative baselines. This paper claims to achieve SOTA and has compared it with some diffusion-based methods such as Echomimic, but this method is no longer so new in the field. Some newer methods have not been included in the comparison, e.g., Echomimic-V2 [1], Hallo-3 [2], and OmniAvatar [3].
2.	There are obvious artifacts in the upper body modeling. From the comparative demos (e.g., Video Driven-Case 1), the generated results of the upper body show a certain degree of blurriness and flicker. I understand that the background part can be post-processed by segmenting the character's head and pasting the background back, but the blurriness issue of the upper body is not trivial. Therefore, I am concerned that this method may face significant obstacles in practical applications.
3.	There is no analysis of computational efficiency. For 3D methods, real-time performance is a crucial factor for practical applications, and the efficiency difference between different methods is quite important. However, the paper currently does not include discussions on inference efficiency metrics such as RTF (Real-Time Factor) and GFlops (Giga Floating-Point Operations Per Second).

[1] Meng R, Zhang X, Li Y, et al. Echomimicv2: Towards striking, simplified, and semi-body human animation. CVPR 2025
[2] Cui J, Li H, Zhan Y, et al. Hallo3: Highly dynamic and realistic portrait image animation with video diffusion transformer. CVPR 2025
[3] Gan Q, Yang R, Zhu J, et al. OmniAvatar: Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation. Arxiv 2025.06

### Questions
1.	Compare more diffusion - based baselines, such as Echomimic-V2[1], Hallo-3[2], and OmniAvatar[3]. Based on the quantitative comparison results, discuss the advantages and disadvantages of 3D methods and diffusion model methods.
2.	Discuss and explain the problem of blurriness and flickering in the upper body of the Demo.
3.	Add analysis related to inference efficiency such as RTF and GFlops in the main comparison experiment. Add the GFlops analysis of each module of the model in the appendix.
4.	Will the code and pre-trained model be open-sourced?

[1] Meng R, Zhang X, Li Y, et al. Echomimicv2: Towards striking, simplified, and semi-body human animation. CVPR 2025
[2] Cui J, Li H, Zhan Y, et al. Hallo3: Highly dynamic and realistic portrait image animation with video diffusion transformer. CVPR 2025
[3] Gan Q, Yang R, Zhu J, et al. OmniAvatar: Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation. Arxiv 2025.06

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a novel, end-to-end framework for synthesizing a realistic 3D talking head from a single source image, driven by audio or video. It is the first end-to-end 3D Gaussian Splatting (3DGS) framework for talking-head synthesis that works from just one image without subject-specific training. It uses a lightweight encoder to directly predict motion-disentangled FLAME parameters from the driving input, which ensures smooth, temporally consistent animation and reduces frame-to-frame jitter. The method excels at generalizing to diverse facial styles, including stylized portraits and non-human faces, beyond conventional human faces.

### Strengths
1. It introduces the first fully end-to-end, one-shot talking-head framework using 3D Gaussian Splatting (3DGS), eliminating subject-specific training.
2. This work expands the problem scope, as it successfully generalizes the method to diverse facial styles (e.g., stylized portraits and non-human faces).
3.This work ensures superior temporal stability (jitter-free animation) by using the direct, temporally coherent prediction of FLAME parameters. 
4. The paper presents robust experimental quality, with it showing strong performance against state-of-the-art baselines.

### Weaknesses
1. While 3D Gaussian Splatting  excels at rendering, GenFaceTalk may still exhibit degraded quality or increased artifacts when synthesizing the subject from extreme novel viewpoints (e.g., severe profile views) or large changes in head pose, a common struggle for methods trained only on a single-view source image.
2. It may struggle to faithfully model and animate complex, non-rigid elements like fine strands of hair, hats, glasses, or complex inner-mouth dynamics (e.g., tongue movement during speech), leading to visual distortions or jitter in these areas during animation.
3. The method uses the FLAME parametric model to drive motion. While this ensures consistency, the FLAME model itself may be insufficient to capture very fine, unique identity-specific expressions or wrinkles (e.g., nasolabial folds), potentially resulting in a "smooth" or slightly generic appearance in certain areas.
4. The visualized results do not clearly demonstrate a substantial visual superiority over strong existing baselines, specifically GAGAvatar. This suggests that while quantitative metrics may show gains, the practical, perceptual quality gap is narrow, raising questions about the effectiveness of the structure.

### Questions
1. GenFaceTalk claims strong generalization to non-human domains (e.g., cats and dogs). Given that the underlying motion model (FLAME) is derived from human topology, the geometric distribution of the non-human source face is fundamentally different from the model's vertices. How does the method resolve this significant geometric mismatch? Specifically, in the process involving Equation 9, how are the human-centric vertices effectively adjusted, mapped, or used to drive the 3D Gaussian points for the distinct geometry of the non-human subject?

2. The paper claims strong generalization to diverse styles, including stylized portraits and non-human faces. Given that the visual superiority over competitors like GAGAvatar is not consistently clear, what specific, quantitative metrics were used to rigorously evaluate the fidelity and consistency of the stylization and identity preservation test set? Can you include a human evaluation (user study) specifically designed to assess whether the unique artistic elements of the source portrait were successfully maintained during animation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes **GenFaceTalk**, a one-shot, end-to-end 3D talking-head generation framework based on **3D Gaussian Splatting (3DGS)**, with the following key claims:

1. **Temporal coherence**: The method avoids frame-wise jitter by directly regressing temporally consistent FLAME parameters from driving video (or audio), bypassing per-frame 3DMM fitting.
2. **Identity preservation**: It preserves fine-grained identity details from a single source image without multi-view supervision.
3. **Generalization**: It generalizes to unseen human identities, stylized portraits, and even non-human faces (e.g., animals).
4. **Unified audio/video-driven synthesis**: Supports both modalities without subject-specific training.
5. **End-to-end design**: Eliminates reliance on offline face reconstruction or post-processing at inference.

 While results on standard benchmarks (VFHQ, HDTF) are impressive, the work suffers from critical limitations that undermine its broader claims. Most notably, the system lacks true 3D consistency: it only supports narrow frontal views (±30°), cannot handle neck motion or large head rotations, and provides no evidence of 360° novel-view synthesis—severely limiting real-world applicability. The generalization to non-human and stylized faces is visually suggestive but unverified quantitatively and conceptually questionable due to reliance on human-specific FLAME parameters during training. Additionally, the audio-driven evaluation on VFHQ uses ad-hoc audio alignment without validation, and key baselines like VASA-1 or Omni-Human are omitted. Despite claiming an end-to-end pipeline, the method still depends on offline FLAME and camera supervision for training. These issues—combined with overstatements about domain generalization and missing throughput metrics—suggest the contribution is incremental and narrowly applicable. The technique is well-executed within its scope, but the lack of robust 3D geometry and limited pose range prevent it from being a significant advance for general-purpose 3D avatar synthesis.

### Strengths
1. **High visual fidelity and temporal coherence**
    
    Outperforms prior 2D and 3D methods (including NeRF- and 3DGS-based) on standard metrics (PSNR, SSIM, LPIPS, FVD), with notably smoother animations and reduced jitter—attributed to end-to-end FLAME regression from video.
    
2. **True one-shot, subject-agnostic synthesis**
    
    No fine-tuning or per-subject training required. Works on unseen identities at inference time, unlike many NeRF/3DGS methods that need minutes of video or multi-view data.
    
3. **Unified audio- and video-driven pipeline**
    
    Same architecture supports both modalities without retraining—rare among 3DGS-based talking-head systems.

### Weaknesses
1. **Heavy reliance on FLAME supervision during training**
    
    Ground-truth FLAME parameters (from DECA) and camera poses are used as supervisory signals (Eqs. 13–14). This introduces human-specific bias and limits motion expressiveness (e.g., no tongue, limited eye/neck modeling).
    
2. **No comparison with recent high-performing 2D SOTA**
    
    Omits evaluation against strong diffusion- or transformer-based 2D methods like **VASA-1**, **Omni-Human**, or **LivePortrait**, which achieve impressive realism and lip-sync.
    
3. **Audio-driven evaluation on VFHQ is questionable**
    
    VFHQ has no native audio; authors “augment it by aligning selected YouTube videos.” No details on alignment quality, speaker consistency, or potential mismatches—undermining Sync score reliability.
    
4. **No runtime or throughput metrics reported**
    
    Despite claiming “real-time rendering” (via 3DGS), the paper provides **no FPS, latency, or hardware-specific throughput.**
    
5. **FLAME’s incompatibility with non-human anatomy is unaddressed**
    
    FLAME is a human-specific parametric model. It’s unclear how it meaningfully represents cat ears, snouts, or cartoon deformations. The success on animal faces may stem from appearance learning overriding flawed motion priors, but this is not analyzed.
    
6. **Limited pose range and 3D consistency**
    
    All demos show near-frontal views (±30°). No evidence of **360° novel-view synthesis**, consistent geometry under extreme rotation, or handling of **neck/shoulder motion**. Likely constrained by FLAME’s limited expressiveness and single-image initialization. This is a direct shortcoming preventing from practical use. 
    
7. **x_initial (initial Gaussians) lacks clarity for non-human cases**
    
    The paper doesn’t specify how the initial 3D Gaussian set is obtained for non-human inputs. If initialized from a FLAME mesh (human topology), this could cause geometric misalignment.

### Questions
1. What if there are no FLAME supervision? Would the training still work?
2. How is the x_initial obtained? What about the case of non-human lives?
3. How can the FLAME parameters represent non-human lives? Why not include them in the demo video?

### Soundness
3

### Presentation
2

### Contribution
3
