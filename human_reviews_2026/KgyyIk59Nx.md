# High-Fidelity and Long-Duration Human Image Animation with Diffusion Transformer

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Recent progress in diffusion models has significantly advanced the field of human image animation. While existing methods can generate temporally consistent results for short or regular motions, significant challenges remain, particularly in generating long-duration videos. Furthermore, the synthesis of fine-grained facial and hand details remains under-explored, limiting the applicability of current approaches in real-world, high-quality applications. To address these limitations, we propose a diffusion transformer (DiT)-based framework which focuses on generating high-fidelity and long-duration human animation videos. First, we design a set of hybrid implicit guidance signals, enabling our framework to additionally incorporate detailed face and hand features as guidance. Next, we incorporate the time-aware position shift fusion module, modify the input format within the DiT backbone, and refer to this mechanism as the Position Shift Adaptive Module, which enables video generation of arbitrary length. Finally, we introduce a novel data augmentation strategy and a skeleton alignment model to reduce the impact of human shape variations across different identities. Experimental results demonstrate that our method outperforms existing state-of-the-art approaches, achieving superior performance in both high-fidelity and long-duration human image animation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper improves human image animation by addressing long-duration generation and fine-grained detail synthesis. The used DiT-based framework includes: (1) hybrid guidance signals for detailed facial/hand features, (2) a Position Shift Adaptive Module for arbitrary-length videos, and (3) skeleton alignment to handle identity variations. However, the technical contribution appears limited, as most components are adapted from existing works rather than novel designs.

### Strengths
1. The paper focuses on two important real-world challenges in human image animation: generating long-duration videos and synthesizing fine-grained facial and hand details, which are critical for high-quality applications.

2. The proposed framework integrates multiple components (hybrid guidance signals, Position Shift Adaptive Module, and skeleton alignment) into a unified system that handles long video generation.

### Weaknesses
1. Limited technical contribution: Several claimed contributions lack novelty. Hybrid guidance signals are standard in human image animation tasks, and the time-aware position shift fusion is adapted from Sonic. The paper should more clearly distinguish its contributions from prior work.

2. HIA [b] already proposed using sharpness conditioning, which undermines the novelty claim of this contribution.

3. Figure 5 and two supplementary videos are inadequate for demonstrating the method's capabilities. You should provide more extensive qualitative comparisons with baseline methods and diverse examples showcasing long-duration generation and fine-grained details.

4. The paper needs more comprehensive information about both the training data and the collected testing datasets, including dataset statistics, collection procedures, and whether the data will be made publicly available.

5. Missing citations:

a. MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model

b. High Quality Human Image Animation using Regional Supervision and Motion Blur Condition

c. X-Dancer: Expressive Music to Human Dance Video Generation

d. MagicPose: Realistic Human Poses and Facial Expressions Retargeting with Identity-aware Diffusion

### Questions
When mentioning other papers in the Methods section, please add proper references to the original works.

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
4

### Summary
This paper addresses two major, contemporaneous challenges in human image animation: generating high-fidelity details and maintaining temporal coherence over long video durations. The authors propose a Diffusion Transformer (DiT) based framework, building upon the pre-trained Wan2.1 architecture. The method introduces a hybrid set of implicit guidance signals for fine-grained control and a Position Shift Adaptive Module for long-form synthesis. The experimental results are quantitatively superior to existing state-of-the-art methods. The work is technically solid and highly impactful for real-world applications.

### Strengths
1.	The paper tackles the long-standing issues of temporal coherence over extended periods and the synthesis of realistic, plausible hand and facial details.
2.	The method demonstrates superior quantitative results across comprehensive metrics, achieving the best scores on multiple VBench metrics.

### Weaknesses
1.	The paper aims to handle long video generation. However, the proposed solution, the Position Shift Adaptive Module, is not a fundamental architectural improvement for diffusion models in general. Instead, it is an engineering modification highly coupled with the pre-existing design of the Wan2.1 DiT backbone and its 3D VAE.
2.	The high-fidelity results are achieved through the incorporation of multiple specialized pre-processing encoders: a Face Encoder (initialized by LivePortrait) and a Hand Encoder (initialized by HaMeR's ViT backbone). These are borrowed from other methods. And this complexity introduces significant pre-processing overhead and multiple failure points during inference. The reliance on these heavy, pre-trained external models makes the overall system less compact and computationally demanding compared to simpler, end-to-end approaches.

### Questions
It seems that the proposed method achieve similar performance against the baseline UniAnimate-DiT from the Supplementary Materials. And we can not notice remarkable improvement in the performance.

### Soundness
3

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
4

### Summary
This paper proposes a DiT-based human animation framework which focuses on generating high-fidelity and long-duration human videos. Firstly, a set of hybrid implicit guidance signals is incorporated to enhance facial and hand feature details. Next, the time-aware position shift fusion module is incorporated to enable long video generation. Finally, a novel data augmentation strategy and a skeleton alignment model are proposed to reduce the impact of human shape variations across different identities. Experimental results demonstrate that the proposed method outperforms existing approaches and achieves superior performance in both high-fidelity and long-duration human image animation.

### Strengths
1．	The authors propose a comprehensive framework for human animation that enhances the clarity and fidelity of faces and hands in the generated video frames.
2．	The authors incorporate the time-aware position shift fusion module and adapt it to video scenarios, enabling the generation of long videos.
3．	The authors present a data augmentation strategy and a pose alignment module to eliminate body shape discrepancies across different human identities.

### Weaknesses
1.	The novelty of this paper appears limited, as the various modules are largely based on related existing works. For example, the appearance feature extractor is derived from LivePortrait [1] (Lines 208-209), the hand representation follows that of RealisDance [2] (Lines 219-220), and the highlighted contribution for long video generation—the time-aware position shift fusion module—originates from Sonic [3] (Line 240). The distinctions and improvements currently described in the paper are insufficient to support its novelty.
2.	The paper fails to sufficiently explain its adopted modules, making them hard to understand. For instance, in Section 3.3, it is unclear how the time-aware position shift fusion module functions. The meaning of the shifted windows, denoted as [s, e] (Line 246), is also not clearly defined.
3.	The paper's ablation study is not specific or sufficient enough: 1) As a primary contribution, the Laplacian Sharpness Factor for the hand guidance signal warrants a separate ablation study to justify its inclusion. 2) For the "Ablation on Pose Alignment," the proposed Data Augmentation Strategy and the Alignment and Smoothness Model should be ablated separately.  

[1] Guo J, Zhang D, Liu X, et al. Liveportrait: Efficient portrait animation with stitching and retargeting control[J]. arXiv preprint arXiv:2407.03168, 2024.
[2] Zhou J, Wang B, Chen W, et al. Realisdance: Equip controllable character animation with realistic hands[J]. arXiv preprint arXiv:2409.06202, 2024.
[3] Ji X, Hu X, Xu Z, et al. Sonic: Shifting focus to global audio perception in portrait animation[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 193-203.

### Questions
1.	The authors present comparative results on the TikTok test set and a self-constructed test set. However, they do not elaborate on the distinctions between the two in terms of data types and content. To ensure the method's reproducibility, the construction process for their custom dataset should be detailed.
2.	For the comparison with other methods in Figure 5, it is recommended that the authors provide more examples to better demonstrate the superiority of the proposed method. In Figure 6, both the driving images and the reference images should be provided. Currently, Figures 5 and 6 seem to exhibit poor performance in terms of face expression following.
3.	Regarding the statement in Line 374 about the potential inclusion of TikTok test data in UniAnimate-DiT's training set, it would be helpful if the authors could clarify the basis for this observation. Alternatively, if this is intended as a hypothesis, it might be more constructive to focus the discussion on a deeper analysis of why the proposed method shows a performance gap on metrics like FID.

### Soundness
3

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
This paper proposes a Diffusion Transformer (DiT) framework for long-duration human image animation. Authors design three modules to stress fidelity (hands/faces), temporal consistency, and cross-identity robustness.

### Strengths
The  designed three core components, Hybrid Implicit Guidance Signals, Position Shift Adaptive Module and Pose Alignment & Data Augmentation successfully stressed the critical issue in long-duration human image animation.

The proposed method achieves the over-one-minute-long human animation videos with consistent motion and detailed hand-face synthesis for the first time.

### Weaknesses
The substantial computational cost and data annotation requirements limit the practical value of this method for the academic community. Reviewer is more looking forward to theoretical discoveries rather than mere engineering stacking.

Some existing projects have already achieved long-duration video generation, of course with different method and generating result. Therefore, the authors’ claim of being “the first to demonstrate human image animation videos exceeding one minute in duration” appears to be somewhat exaggerated.

Reviewer noticed that at the 27th second in the supplementary video 25180, a ring appears on the actress’s hand. This is a factual hallucination. Why it happens? If even carefully selected cases contain flaws, it is hard to imagine the quality of the remaining examples.

### Questions
please refer to weakness

### Soundness
3

### Presentation
3

### Contribution
2
