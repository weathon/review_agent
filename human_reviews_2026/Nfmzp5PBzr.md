# PAGE-4D: Disentangled Pose and Geometry Estimation for VGGT-4D Perception

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 8, 4

## Abstract
Recent 3D feed-forward models, such as the Visual Geometry Grounded Transformer (VGGT), have shown strong capability in inferring 3D attributes of static scenes. However, since they are typically trained on static datasets, these models often struggle in real-world scenarios involving complex dynamic elements, such as moving humans or deformable objects like umbrellas. To address this limitation, we introduce PAGE-4D, a feedforward model that extends VGGT to dynamic scenes, enabling camera pose estimation, depth prediction, point cloud reconstruction, and point tracking—all without post-processing. Training a geometry transformer for dynamic scenes from scratch, however, demands large-scale dynamic datasets and substantial computational resources, which are often impractical. To overcome this, we propose an efficient fine-tuning strategy that allows PAGE-4D to generalize to dynamic scenarios using only limited dynamic data and compute. In particular, we design a dynamics-aware aggregator that disentangles dynamic from static content for downstream scene understanding tasks: it first predicts a dynamics-aware mask, which then guides a dynamics-aware global attention mechanism. Extensive experiments show that PAGE-4D consistently outperforms the original VGGT in dynamic scenarios, achieving superior results in camera pose estimation, monocular and video depth estimation, and dense point map reconstruction. The source code and pretrained model weights are provided in the https://page4d.github.io.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The work proposes an extension of VGGT for improved accuracy of reconstruction and camera poses on dynamic scenes.
The idea is rather simple: using masked attention to ignore the patches of dynamic areas from pose estimation.
Empirically, the work finds that fine-tuning only 10 middle layers with masked attention is effective.
Experiments show a notable improvement of depth estimation on three datasets and some improvement for the camera pose.

### Strengths
* The idea is simple and requires only a partial model finetuning to take effect. This allows for adopting the VGGT architecture as a whole, preserving some of its desirable properties (e.g. runtime).
* The work does a nice job at motivating the approach (c.f. Sec. 3.1).
* Experiments span a compelling spectrum of benchmarks - depth and camera pose pose estimation, point reconstruction and novel view synthesis.

### Weaknesses
* The overall technical contribution — introducing masked attention and fine-tuning the model on dynamic data — is rather minor.
* The improvements shown in Tab. 1 are convincing, but it is not exactly clear whether they stem from the finetuning per se, or masked attention. This closely relates to the next point.
* Tab. 5 is rather short for an ablation study and reveals two observations: 1) on some datasets, the improvement may come already from fine-tuning, without introducing masked attention 2) the benefit of masked attention appears marginal on some benchmarks (e.g. DyCheck).

### Questions
* I would be curious to see evaluation results of VGGT* (Middle Layers) in other settings of Tab. 1 (e.g. monocular depth). The goal is to disentangle the contribution of the masked attention alone.
* It is somewhat bizarre that masked attention, which affects the attention only for the camera token, benefits depth estimation more than the camera pose results in Tab. 2, which are mixed. I would be great if the authors could elaborate.
* l. 257 “in a self-supervised manner” – isn’t the network supervised by the camera poses and depth?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces a dynamics-aware aggregator that separates static and dynamic information through a learned dynamics-aware mask.

Rather than treating motion as uniformly harmful or helpful, the authors make its influence task-dependent. The aggregator first predicts a mask that highlights dynamic regions, then feeds it into a cross-attention module—suppressing dynamic cues for camera-pose tokens while preserving them for geometry tokens.

Combined with self-supervised fine-tuning of layers most sensitive to motion, this design exploits dynamics where they benefit geometry grounding and mitigates their adverse impact on pose estimation.

As a result, the method achieves accurate pose and geometry estimation across both static and dynamic scenes.

### Strengths
Experiments demonstrate that PAGE-4D consistently surpasses the baseline VGGT in dynamic scenes, yielding superior results on camera-pose estimation, monocular and video depth estimation, and dense point-map reconstruction.

The authors devise a dynamic-mask prediction module that learns, in a self-supervised fashion, which spatial regions are likely to contain dynamic objects.

The model predicts the mask in earlier layers and injects it into later attention layers for pose estimation, enabling a single feed-forward pass; in contrast, methods such as Easi3R require a second forward pass.

Owing to its plug-in design, PAGE-4D introduces only negligible runtime and memory overhead compared with VGGT.

The analyses presented in Table 6 and Table 7 are particularly insightful.

### Weaknesses
Relationship to Easi3R is under-explained. Lines 146–176 (Sect. 3.1 and Fig. 2) essentially restate the key observation of Easi3R—that dynamic regions show weaker activations and should be suppressed for pose estimation—yet the manuscript does not discuss this connection in the Introduction, Related-Work, or Method sections. As presented, PAGE-4D reads like a VGGT-based version of Easi3R (originally built on DUSt3R), augmented with a feed-forward mask predictor. A clearer positioning with respect to Easi3R is needed, including a discussion of conceptual overlap and technical differences.

Quantitative comparisons are incomplete. Table 1 omits several recent dynamic-scene baselines—Easi3R, AETHER, Geo4D, and ViPE—making it difficult to gauge the real performance gap.

Tables 2 and 3 report only Easi3R(DUSt3R) numbers; the Easi3R(MoNSTR) variant is absent. Including it would give a fairer picture of state-of-the-art accuracy.

Table 4 why not include MoVieS?

The paper does not assess dynamic-object segmentation. A DAVIS-style evaluation (as in Table 1 of Easi3R) would verify whether the predicted masks align with true motion regions and would offer further evidence for the quality of the dynamics-aware module.

### Questions
Line 025: Denser point clouds
• PAGE-4D reportedly reconstructs point clouds that are markedly denser than those of VGGT. Could you clarify which part of the paper drives this improvement? 

Overlap with Easi3R (Lines 146 – 176)
• Section 3.1 and Fig. 2 echo the core finding of Easi3R—namely, that dynamic regions exhibit weaker activations and should be down-weighted for pose estimation. Yet Easi3R is not discussed in the Introduction, Related-Work, or Method sections.
• Please elaborate on:
– Conceptual similarities and differences between PAGE-4D and Easi3R.
– The technical novelty beyond replacing DUSt3R with VGGT and moving to a feed-forward mask predictor.

“Suppress vs. amplify” inconsistency (Lines 296 – 298)
• The abstract states that motion cues are suppressed for pose estimation and amplified for geometry reconstruction. In the method description, however, the mask is applied in pose-estimation layers and left untouched for geometry reconstruction (i.e., neither suppressed nor amplified). Could you clarify this discrepancy and specify where “amplification” occurs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The task of this paper is to jointly predict point clouds and camera poses for each image frame in a sequence of video frames depicting dynamic scenes. The proposed method, called PAGE-4D, is built upon the VGGT neural network architecture. The authors observed that dynamic contexts significantly affect the accuracy of camera pose estimation, while having a smaller impact on point cloud estimation. They also found that the middle layers of VGGT exhibit lower attention weights on dynamic pixels.

Based on these observations, the authors propose a creative model design and fine-tuning strategy. In particular, they introduce an attention mask prediction module that suppresses attention weights for camera prediction tokens while leaving geometric tokens unaffected. This is an idea consistent with their empirical findings. To improve parameter efficiency during fine-tuning, they further propose a strategy that updates only the middle 10 layers of the network.

Table 1 shows that PAGE-4D achieves state-of-the-art performance across multiple datasets, including Sintel, Bonn, and DyCheck. The ablation study demonstrates that fine-tuning only the middle 10 layers yields results comparable to full fine-tuning, and it also verifies the effectiveness of the proposed suppression mask.

Overall, this is a well-motivated neural network design with solid experimental results and clear presentation. I believe this paper is strong enough to be accepted at ICLR, though I have a few minor concerns discussed below.

### Strengths
- Good observation and thoughtful neural network design, leading to strong experimental results.
- The presentation is clear, and the supplementary materials are comprehensive. I also appreciate that the authors have provided the code.
- Overall, this is a well-structured and solid research paper. It identifies a problem, derives insights through careful observation, and proposes specific design choices based on experimental findings. Reading the paper feels smooth and coherent.

### Weaknesses
Looking at Figure 2(b), I feel that the content does not align well with the caption. The caption states that the VGGT layers have smaller attention weights on dynamic pixels. However, only one image (the 11th layer) among the four shown appears consistent with this description. In the other three images in Figure 2(b), there does not seem to be any noticeably high attention values on dynamic pixels. Could the authors double-check this? Or is it possible that the observation is not entirely accurate?

### Questions
- What is the GPU memory consumption during inference compared to VGGT?
- How does the fine-tuned model perform on static scenes, compared to VGGT?

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
4

### Summary
This paper introduces an extension of VGGT that incorporates dynamic object masks. The core idea is to predict these masks and then inject them additively into the middle layers of the VGGT architecture. The authors demonstrate the effectiveness of this approach through experiments on five benchmarks involving dynamic objects: video depth estimation, monocular depth estimation, camera pose estimation, multi-view point map reconstruction, and 4D view synthesis.

### Strengths
+ This paper presents a straightforward yet effective enhancement to the VGGT architecture. The method stands out because it leverages a simple idea without requiring extensive fine-tuning, making it practical for integration. 
+ An advantage is its self-supervised mask prediction, which eliminates the need for any explicit mask supervision. 
+ The approach demonstrates substantial improvements across several dynamic benchmarks, underscoring its efficacy in handling complex scenarios.
+ The writing is clear and easy to understand.

### Weaknesses
- The decision to use an additive mask within the mask attention mechanism, as seen in Equation (6), warrants further clarification. It would be beneficial for the authors to quantitatively report and explain why a multiplicative mask, which could directly attenuate irrelevant attention, was not considered.

- To fully understand the mask's efficacy, a quantitative evaluation of the predicted mask accuracy is crucial. Assessing this at different layers of the global attention on the Odyssey test set, using a metric like IoU2D, would provide valuable insight into whether the network is genuinely learning effective masking.

- The claim on Line 150 regarding the performance gap between static and dynamic regions on the Odyssey test set, where "Absolute Depth Error in dynamic regions is 94% higher than in static regions," is not supported by a corresponding table or explicit results. This table should be included, and it would be insightful to see how this performance gap changes when the proposed Page4D method is applied to the Odyssey dataset.

- Given that the current results focus solely on dynamic scenes, it is important to understand the impact of Page4D on static scene performance. Therefore, I recommend quantitatively reporting the performance of Page4D on a selection of tasks and datasets that include static scenes, such as:
  - Camera Pose Estimation on the RealEstate10K and Co3Dv2 datasets (as presented in Table 1 of the VGGT paper).
  - Dense MVS Estimation on the DTU dataset (as presented in Table 2 of the VGGT paper).
  - Point Map Estimation on the ETH3D dataset (as presented in Table 3 of the VGGT paper).
  - Novel View Synthesis on the GSO dataset (as presented in Table 7 of the VGGT paper).
  - Dynamic Point Tracking Results on the TAP-Vid dataset (as presented in Table 8 of the VGGT paper). This would provide a more complete picture of the method's overall capabilities.

### Questions
Please see the Weaknesses. I am willing to change my score provided the authors' answer my questions.

### Soundness
2

### Presentation
3

### Contribution
2
