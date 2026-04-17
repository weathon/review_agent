# DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Scaling Vision-Language-Action (VLA) models on large-scale data offers a promising path to achieving a more generalized driving intelligence.
However, VLA models are limited by a ``supervision deficit'': the vast model capacity is supervised by sparse, low-dimensional actions, leaving much of their representational power underutilized.
To remedy this, we propose DriveVLA-W0, a training paradigm that employs world modeling to predict future images.
This task generates a dense, self-supervised signal that compels the model to learn the underlying dynamics of the driving environment.
We showcase the paradigm's versatility by instantiating it for two dominant VLA archetypes: an autoregressive world model for VLAs that use discrete visual tokens, and a diffusion world model for those operating on continuous visual features.
Building on the rich representations learned from world modeling, we introduce a lightweight action expert to address the inference latency for real-time deployment.
Extensive experiments on the NAVSIM v1/v2 benchmark and a 680x larger in-house dataset demonstrate that DriveVLA-W0 significantly outperforms BEV and VLA baselines.
Crucially, it amplifies the data scaling law, showing that performance gains accelerate as the training dataset size increases.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DriveVLA-W0, a training paradigm for VLA models in autonomous driving that addresses the “supervision deficit” in current VLA systems, wherein large model capacity is supervised by sparse, low-dimensional action labels. DriveVLA-W0 supplements action supervision with a dense, self-supervised world modeling objective and demonstrates its method with both autoregressive and diffusion approaches. The authors conduct extensive experiments on the NAVSIM benchmark and a large-scale in-house dataset (70M frames), showing that world modeling not only accelerates performance scaling with data size but also improves generalization, efficiency, and real-time deployment feasibility.

### Strengths
- Clearly define the supervision deficit issue and explain why relying solely on action objectives does not scale.
- Introduce rigorous and balanced designs for both discrete (AR/VQ) and continuous (diffusion/ViT) visual representations, with empirical evidence showing generalization benefits under distribution shift and the critical role of dense supervision on data scaling.
- Extensive experiments along with qualitative visualizations that link generative fidelity to planning consistency, reinforcing the claim that predictive world modeling yields richer representations.

### Weaknesses
- The claim that “the new training paradigm that uses world modeling as a powerful form of self-supervision to supplement the sparse action signal” is somewhat overclaimed, given that recent methods such as WorldVLA, DOE-1, and LAW similarly adopt self-supervised world modeling to mitigate the sparse action signal.
- Parity of baselines on the in-house dataset is not fully convincing. The re-implementation of TransFuser differs notably in terms of pretraining compared to other approaches, and it typically uses multi-sensor input. Moreover, including more VLM-based methods would help establish a fairer comparison.
- The experimental evaluations are conducted only in a semi-closed-loop, non-reactive setting. It would be beneficial to provide further exploration and demonstration under a fully closed-loop evaluation.

### Questions
- In Table 2, why does equipping the world model result in worse performance on the 700k dataset compared to the baseline?
- Can this framework be extended to incorporate multi-view image inputs?
- Regarding latency, is the reported delay solely for the action output, or does it represent the total delay? Could you provide latency metrics with varying sequence lengths (e.g., 2VA, 3VA, etc.)?

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
5

### Summary
This paper identifies a critical "supervision deficit" in scaling Vision-Language-Action (VLA) models for autonomous driving, where large-capacity models are underutilized due to sparse action supervision. To address this, the authors propose DriveVLA-W0, a training paradigm that incorporates world modeling as a dense, self-supervised learning signal. This approach is instantiated for both discrete (VQ-based) and continuous (ViT-based) VLA architectures using autoregressive and diffusion world models, respectively. The core contribution is the extensive experimental validation, particularly on a massive 70M-frame dataset, demonstrating that this dense supervision not only improves performance but fundamentally amplifies the data scaling law. The paper also introduces a lightweight MoE-based "Action Expert" to ensure real-time inference.

### Strengths
- The paper provides a clear motivation by identifying the "supervision deficit" as a key bottleneck. This framing is insightful and pinpoints a significant challenge in scaling VLA models.

- The empirical evidence is a major strength. The authors validate their approach across two distinct VLA architectures and, most importantly, across vastly different data scales (NAVSIM and a 70M-frame in-house dataset). This large-scale study provides strong evidence for their claims.

- The central claim that world modeling amplifies data scaling, rather than just providing a performance boost, can be a highly impactful finding for the driving community. 

- Achieving state-of-the-art results on the NAVSIM benchmark further underscores the effectiveness of the proposed representation learning paradigm.

### Weaknesses
- Training Cost Discussion: The addition of a world modeling objective (especially a diffusion-based one) presumably adds significant computational overhead to the training process. A brief discussion on the trade-offs in terms of training cost (e.g., total GPU hours vs. the baseline) would make the analysis more complete.

- Generalization to Other Sensors: The work impressively achieves SOTA with a single camera. However, it would be beneficial to discuss the authors' perspective on how this world modeling approach might apply to richer sensor suites (e.g., surround-view cameras or LiDAR). Would they expect a similar "amplification" effect when the model already has access to more complete 3D spatial information?

### Questions
The finding in Table 3 regarding the "performance reversal" of action experts is one of the most interesting parts of the paper. The autoregressive expert, which performs worst on the small-scale NAVSIM dataset, becomes the best-performing model at the 70M-frame scale. Could the authors elaborate on their hypothesis for this reversal? Is this purely a matter of sample efficiency, where the flow-matching expert simply requires even more data to converge on the complex trajectory manifold? Or, do the authors believe that the discrete, tokenized action space of the autoregressive model provides a fundamentally better inductive bias for modeling the highly complex and multi-modal action distributions present in a massive, diverse dataset?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces DriveVLA-W0, a method that utilizes future image prediction as an auxiliary training objective in addition to the action prediction objective. The paper investigates this method on both VQ- and ViT-based architectures. Experimental results on NAVSIM and an in-house dataset demonstrate the effectiveness of this approach compared to VLAs that only predict actions.

### Strengths
S1) The proposed method is well-motivated and writing is easy to follow.

S2) The analysis of why employing next image prediction is beneficial, as also shown in the results of Figure 4, indicates that jointly predicting the next frame leads to more transferable features and helps avoid overfitting to dataset-specific action patterns.

S3) In addition to NAVSIM, the paper also investigates the scalability of their method on a larger in-house dataset.

### Weaknesses
W1) Even though the results look good, the paradigm that jointly predicts action and future states (images) is not a new concept, both in autonomous driving (DrivingGPT, Doe-1, VaVAM) and robot learning.

W2) Efficiency concern. The paper investigates both VQ and ViT based architecture. For VQ, the model is trained in an interleaved manner: image, action, image, action. The paper mentions that image generation can be bypassed during inference. However, this may lead to train-infer gap. Otherwise, the latency should be huge.

W3) From Line 316-321, the model is trained in two-stage manner. In the first stage, the model is trained to predict both image and action. However, in the second stage, both VLA backbone and action expert is trained. But why only predict action in this stage. As the author already claimed that jointly predict image and action leads to better representation, will this undermine VLA's representation in the second stage?

### Questions
Please check the Weaknesses

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
3

### Summary
This paper presents DriveVLA-W0, a Vision-Language-Action (VLA) model augmented with a world model to predict future visual frames. The goal is to provide denser self-supervised signals to compensate for the sparsity of action labels in end-to-end driving. Two variants are proposed: an autoregressive world model for discrete (VQ-tokenized) features and a diffusion-based world model for continuous visual features. The paper claims that the proposed method amplifies data-scaling behavior—i.e., performance continues to improve as data grows. Results on NAVSIM and a large-scale in-house dataset show consistent improvements in driving performance and sample efficiency.

### Strengths
* The empirical analysis of data scaling is thorough and carefully conducted across multiple model and data sizes.
* Both autoregressive and diffusion-based variants are implemented, illustrating generality across different architectures.
* The paper contributes valuable large-scale empirical findings to the community, especially around scaling trends and model efficiency.

### Weaknesses
* The core contribution is incremental. The main idea—adding a future-frame prediction objective as an auxiliary loss—is conceptually simple and has been explored in prior work on world models (e.g., WorldVLA, UniVLA).
* The proposed “world model” is used only as auxiliary supervision during training and not for downstream planning, simulation, or closed-loop control, limiting its conceptual impact.
* Although the paper introduces a “Mixture-of-Experts” Action Expert, the 8B VLA backbone is still required at inference. The MoE design primarily reduces the cost of the action decoding stage rather than removing the need for the large backbone itself, so the memory footprint and deployment complexity might remain dominated by the 8B model.
* Qualitative visualizations of predicted future frames with different actions are missing.

### Questions
* Can the learned world model predict future observations conditioned on different actions, and could it be leveraged for simulation or counterfactual reasoning?
* How sensitive is the performance to the weighting between the action-prediction and world-modeling losses?
* Could the authors include more qualitative visualizations or failure cases to better illustrate model behavior and limitations?

### Soundness
3

### Presentation
3

### Contribution
3
