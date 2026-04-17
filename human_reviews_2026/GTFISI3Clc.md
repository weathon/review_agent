# Ego-centric Predictive Model Conditioned on Hand Trajectories

- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
In egocentric scenarios, anticipating both the next action and its visual outcome is essential for understanding human–object interactions and for enabling robotic planning. However, existing paradigms fall short of jointly modeling these aspects. Vision-Language-Action (VLA) models focus on action prediction but lack explicit modeling of how actions influence the visual scene, while video prediction models generate future frames without conditioning on specific actions—often resulting in implausible or contextually inconsistent outcomes. To bridge this gap, we propose a unified two-stage predictive framework that jointly models action and visual future in egocentric scenarios, conditioned on hand trajectories. In the first stage, we perform consecutive state modeling to process heterogeneous inputs—visual observations, language, and action history—and explicitly predict future hand trajectories. In the second stage, we introduce causal cross-attention to fuse multi-modal cues, leveraging inferred action signals to guide an image-based Latent Diffusion Model (LDM) for frame-by-frame future video generation. Our approach is the first unified model designed to handle both egocentric human activity understanding and robotic manipulation tasks, providing explicit predictions of both upcoming actions and their visual consequences. Extensive experiments on Ego4D, BridgeData, and RLBench demonstrate that our method outperforms state-of-the-art baselines in both action prediction and future video synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Ego-PM, a unified two-stage egocentric predictive model designed to jointly predict actions and generate future visual frames, conditioned on hand trajectories. Extensive experiments have demonstrated the effect of each proposed component on multiple real-world and robotic datasets.

### Strengths
1. This paper combines action modeling with video generation in a single framework.

2. This model demonstrates consistent improvements across multiple benchmarks (Ego4D, BridgeData, RLBench) and modalities.

### Weaknesses
1. One of the main contribution of this paper is the joint prediction of action and future frames, however, I do not see the qualitative/quantitative analysis showing the mutual benefit. 

2. It is not clear how the model fuses multi-modal information when not using CCA in Table 5. 

3. The text format could be further improved (L84-L85).

4. This paper misses some discussions with recent works on conditioned egocentric image/video generation [1][2][3].

[1] Luo et al. "Put myself in your shoes: Lifting the egocentric perspective from exocentric videos." ECCV 2024.
[2] Xu et al. "Egoexo-gen: Ego-centric video prediction by watching exo-centric videos." ICLR 2025.
[3] Liu et al. "Exocentric-to-egocentric video generation." NeurIPS 2024.

### Questions
1. In the experiments, the authors explored using one or two consecutive frames as context, what about using more time steps?

2. What is the advantage of CCA compared to bi-directional attention in your model?

3. The ablation study on consecutive predictions is not very clear. Please explain in detail.

4. In L456-457, the authors claimed the failure of LWM as its exocentric bias, however, the authors already fine-tuned LWM on egocentric data (L260). It is a bit confusing.

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
3

### Summary
This paper addresses a critical gap in egocentric AI: the disjointed modeling of future actions and their visual outcomes. This paper proposes Ego-PM, a novel two-stage framework that unifies action prediction and future video generation conditioned on hand trajectories. The core idea is to use predicted hand motion as an intermediate, guiding representation to ensure that generated future frames are both visually plausible and physically consistent with the intended action. Below are contributions： 
1.	First Unified Action-Visual Predictive Model: This is the first model capable of jointly predicting both the upcoming action (as a hand trajectory) and the future visual frames resulting from that action. 
2.	Novel Architectural Innovations: Consecutive State Modeling (CoSMo): In Stage I, the model predicts future actions by conditioning on two consecutive previous states. Causal Cross-Attention (CCA): In Stage II, the action embedding from Stage I serves as the Query to perform causal cross-attention over the visual and textual context (Keys/Values). 
3.	Demonstrated Generality and Practicality: The same model architecture, without task-specific modifications, achieves state-of-the-art or competitive performance on three distinct benchmarks: Ego4D (human egocentric videos), BridgeData V2 (real-world robot demonstrations), and RLBench (simulated robotic tasks). Crucially, the model requires no external action annotations at inference time, predicting the hand trajectory autonomously, which enhances its practical applicability.

### Strengths
Originality: The core originality lies in its novel problem formulation: the joint modeling of egocentric action prediction and future video generation within a single, unified framework. This is a distinct advance over prior work that addressed these tasks separately. 
Quality: The technical quality is high. The two-stage architecture is well-motivated and built upon strong, modern components (LLaVA, LDM). The experimental quality is exceptional, featuring rigorous evaluation on three diverse and challenging datasets (Ego4D, BridgeData V2, RLBench). 
Clarity: The paper is generally well-structured and clearly written. The logical flow from problem identification to solution and evaluation is easy to follow. 
Significance: The work is highly significant for the research community. It directly addresses a critical capability for embodied AI systems: understanding the coupling between an action and its perceptual consequences.

### Weaknesses
Analysis of Error Propagation in Long-Horizon Prediction: The experiments primarily focus on predicting one or two frames into the future (e.g., t=2, t=3). A key challenge in predictive modeling is error accumulation over longer sequences. The paper would be significantly strengthened by an analysis of the model's performance degradation when generating longer video horizons (e.g., 20 frames). Does the CoSMo strategy provide robustness against compounding errors compared to baselines? 

Computational Efficiency and Latency: The proposed two-stage pipeline, involving a large autoregressive model and an iterative diffusion model, is computationally intensive. For real-world applications like robotic planning, inference speed is critical.

### Questions
Long-Horizon Generalization: The model is evaluated on very short-term predictions. Could the authors demonstrate or discuss its potential for longer-horizon prediction (e.g., 20 frames)? What are the main failure modes (e.g., object deformation, trajectory drift) when the prediction horizon extends, and how might the architecture be adapted to address them? 

Ablation on Action Representation: The action is represented as hand trajectory coordinates for Ego4D and 7D robot poses for Bridge/RLBench. How critical is the specific form of this action representation? Did the authors experiment with other representations, such as a more abstract latent action space (e.g., akin to AdaWorld), and if so, how did it impact performance? 

Causal Cross-Attention Analysis: The CCA module is a key innovation. Could the authors provide more analysis or visualization of the attention maps within the CCA? For example, when the action query attends to the visual keys, which parts of the historical frames (e.g., the object, the hand's current position) does it primarily focus on? 

Computational Cost and Latency: The paper would benefit from a discussion (and ideally, metrics) regarding the computational cost and latency of the full pipeline compared to the baselines (especially the faster VLA models and single-stage video predictors).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
is a tow-stage framework for action anticipation and frame generation coupling VLM and LDM through Causal Cross-Attention (CCA). VLM predicts action trajectories that then condition LDM to generate future frames. CCA implements cross-attention with future masking, and is applied to encode and fuse video, text and action for conditioning future frame generation.

### Strengths
The design is intuitive and leads to an architecture that achieves sota result when compared to few recent baselines.

### Weaknesses
Causal Cross Attention, a main element upon which the contribution of the paper is build, is not particularly novel: standard cross-attention with causal masking.

Why the query in CCA is chosen as the embedding of previous action is not well motivated. It is just adopted this way.

There is no feedback loop that enforces generated frames to contain generated action trajectories following conditioning action trajectories.

Overall, there is no particularly significant novelty. The method is two-stage and not integrated, and there is no grounding loop in the diffusion with the action trajectory conditioning. Loss terms for training are standard or previously published.

It is not clarified in the paper why generating future frame in a 2-stage approach without grounding (see above) should be beneficial. For what should the generated video be used?

Tables are spread around the second part of paper, and the arrangement of frames in figure 3 could be improved. Figure 3 contains only four generated, sparse frames and this makes it difficult to assess the quality of generated future video.

### Questions
Please address any of the weaknesses identified above.

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
3

### Summary
The authors propose a method which, given the action history, visual input and language prompt of either an egocentric video or a robot task rollout video, simultaneously generates imagined action rollouts and, conditioned on these, imagined future video frames. The method makes use of a novel cross-attention scheme to condition a latent diffusion model for the future frame synthesis.

### Strengths
The design choices of the method are thoroughly ablated in Table 5. The method shows strong performance against baselines in Tables 2 and 3. The inclusion of both a robotic dataset and a human hand dataset showcases the usefulness and versatility of the method.

### Weaknesses
The submission could have benefited from the inclusion of more qualitative examples in the supplementary material, comparing the imagined and real future frames. This is crucial to assess the quality of the generations.

I believe Table 3 should be split into two separate tables, evaluating frame prediction and action prediction separately.

In Table 4, as it stands, the method is strongly outperformed by two baselines depending on 3D point clouds. An interesting comparison would be running the baselines without providing them the additionally required 3D point cloud (e.g. by passing dummy values), to assess the extra gain from its inclusion. I am willing to increase my score if a sensible answer is provided to this point.

In the paper, please clarify how you calculate the hand IoU. It took me a long time to find the detail that you use hand bounding boxes in the supplement. I believe this information is important enough to be included in the main paper. 

A bounding box is arguably far less useful than, for instance, a wrist pose or even hand pose, either in 2D or 3D. Thus, I would encourage the authors to study predicting more informative hand representations for egocentric videos.

Table 1 and Table 2 could benefit from the inclusion of more baselines.

### Questions
Why is "Ours" bolded for LPIPS in Table 3 when it is outperformed by This&That?

How do you calculate the GT hand bounding boxes when Ego4D does not provide such poses out-of-the-box? Could the baseline Hand IoU in tables 2 and 3 be calculated by using that same method on the imagined future frames of the baselines?

### Soundness
3

### Presentation
3

### Contribution
3
