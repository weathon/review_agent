# SIGMark: Scalable In-Generation Watermark with Blind Extraction for Video Diffusion

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Artificial Intelligence Generated Content (AIGC), particularly video generation with diffusion models, has been advanced rapidly. 
Invisible watermarking is a key technology for protecting AI-generated videos and tracing harmful content, and thus plays a crucial role in AI safety.
Beyond post-processing watermarks which inevitably degrade video quality, recent studies have proposed distortion-free in-generation watermarking for video diffusion models.
However, existing in-generation approaches are non-blind: they require maintaining all the message-key pairs and performing template-based matching during extraction, which incurs prohibitive computational costs at scale.
Moreover, when applied to modern video diffusion models with causal 3D Variational Autoencoders (VAEs), their robustness against temporal disturbance becomes extremely weak.
To overcome these challenges, we propose SIGMark, a Scalable In-Generation watermarking framework with blind extraction for video diffusion.
To achieve blind-extraction, we propose to generate watermarked initial noise using a Global set of Frame-wise PseudoRandom Coding keys (GF-PRC), reducing the cost of storing large-scale information while preserving noise distribution and diversity for distortion-free watermarking.
To enhance robustness, we further design a Segment Group-Ordering module (SGO) tailored to causal 3D VAEs, ensuring robust watermark inversion during extraction under temporal disturbance.
Comprehensive experiments on modern diffusion models show that SIGMark achieves very high bit-accuracy during extraction under both temporal and spatial disturbances with minimal overhead, demonstrating its scalability and robustness.
Our code is available at https://github.com/JeremyZhao1998/SIGMark-release.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SIGMark, a watermarking method for diffusion video generation. The key contribution is its "blind" feature that doesn't require storing the original watermark message for template matching during extraction. This is in contrast to existing methods such as VideoShield. To achieve this target, the authors build their solution based on the recently proposed PRC (pseudorandom code). Basically, PRC couples the message with some 'testbits' (part of the key) during encoding, and then compare testbits during decoding to verify whether decoded message is correct or not. The authors also propose a SGO module (optical flow segmentation + sliding window detection) to handle the attacks specific to video, e.g., frame adding, dropping, etc.

### Strengths
+ I like the "blind" feature they add to diffusion-based video watermarking domain. It simplifies the watermarking deployment pipeline. PRC is a cryptographically strong primitive that enables encoding/decoding different messages with a single global key, in contrast to traditional encryption methods used in other watermarking methods that require storing different keying material for different messages.

+ The proposed SGO module is effective in handling various frame-level attacks in videos.

+ The experimental results are good. Although the robustness is not consistently better than existing methods, they have the unique advantage of being blind.

### Weaknesses
- The technical contribution for watermark extraction should be articulated with more methodological comparison. (See my suggestion below)

- Experimental settings are not very clear.

### Questions
+ In Section 3.4, you should compare your method (optical flow + sliding window) to previous solutions (VideoShield, VideoMark) in a methodological way, in order to better understand your technical contribution. E.g., do they use any method to partition video to continuous subsequences? How? How do they detect the starting of a group (or do they need to detect this)? What's the main advantage of your solution?

+ Line 345: symbol \hat is at wrong place.

+ In Table 2, we need more details on the setting of drop, insert, and clip. What's the drop ratio? What are inserted frames and insertion ratio?

+ Running time overhead breakdown should be provided for the various steps in their system, e.g., diffusion, PRC encode/decode, optical flow segmentation, sliding window detection, etc.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Video generation with diffusion models is advancing rapidly, and invisible watermarking is essential for provenance and IP protection. Existing in-generation schemes typically require storing message–key (or key–template) pairs and performing template-based matching at inference, which can become costly at scale and tend to be fragile under temporal disturbances. The paper proposes SIGMark, aiming for blind extraction via Global Frame-wise Pseudorandom-Coding keys (GF-PRC) and improved robustness via a Segment Group-Ordering (SGO) module tailored to causal 3D-VAEs, enabling reliable inversion under temporal perturbations.

### Strengths
1.	The paper identifies a practical limitation of many video in-generation watermarking systems: they are not truly blind yet require storing large message–key/template tables, which raises efficiency and storage concerns at scale.

2.	The proposed design addresses both the blindness/scalability issue (via GF-PRC) and the temporal robustness issue (via SGO).

3.	The paper is generally well organized with clear figures, which makes it easy to follow.

### Weaknesses
1.	Limited novelty (GF-PRC). The GF-PRC component mainly builds on the original PRC method. Moreover, introducing PRC (Appendix E) negatively affects robustness. How to balance the impact of PRC, or improve PRC specifically for watermarking robustness requirements remains an open problem.

2.	Scalability evidence. The paper claims that non-blind approaches incur prohibitive computational costs at scale and raise efficiency/storage issues, but this discussion remains at the level of Appendix B (Computation Overhead). The authors should provide experiments to demonstrate that this is a serious practical problem and to show the operational performance of the proposed method.

3.	Robustness gap. Although the method claims robustness to temporal disturbance, its performance under frame drop is far below VideoShield, and it is overall worse than VideoShield under spatial disturbances. The paper attributes this to PRC, suggesting that the framework’s robustness is still not fully resolved.

### Questions
1.	SGO procedure clarity. In Figure 4, for z_1 there appear to be two candidates: [4, pad, pad, pad] or [pad, pad, 6, 7]. Which one is used for decoding? Or should it be [4, pad, 6, 7]? Please separately explain the frame-drop and clip cases so readers can better understand the SGO workflow.

2.	Meaning of “w/o” in Table 2. Does “w/o” denote a clean setting with no time disturbance? Please state this explicitly in the caption.

### Soundness
3

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
This paper proposes SIGMark, a training-free, in-generation watermarking framework for video diffusion models. It addresses two issues of prior approaches: (i) non-blind extraction that scales not well because it requires storing message, and (ii) lacking temporal robustness when modern causal 3D VAEs are disturbed by frame edits. Experiments on HunyuanVideo and Wan-2.2 across T2V and I2V pipelines with a 400-video subset of VBench-2.0 show high accuracy with limited quality impact. Under both spatial and temporal perturbations, SIGMark achieves competitive with non-blind baselines, and outperforms prior non-blind in-generation methods.

### Strengths
- The paper is well-organized and clearly writtern. 
- The inituion of the paper is soundness. 
- Experimental results show the effectivness of the proposed method.

### Weaknesses
- Missing references: 
[1] Huang, Huayang et al. “ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization.” ArXiv abs/2411.03862 (2024): n. pag.

Please also refer to the "Questions" section.

### Questions
- What is the false-positive rate when decoding regenerations (video-to-video) from different diffusion models?
- The paper claims O(1) extraction complexity and “near–real-time” segmentation. The appendix provides only a cursory analysis. Could you add end-to-end timings with per-component breakdowns, scaling with (d_t) and payload size, and comparisons to baselines?

### Soundness
3

### Presentation
3

### Contribution
2
