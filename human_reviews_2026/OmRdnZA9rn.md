# SimpliHuMoN: Simplifying Human Motion Prediction

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Human motion prediction combines the tasks of trajectory forecasting, human pose prediction, and possibly also multi-person modeling. 
For each of the three tasks, specialized, sophisticated models have been developed due to the complexity and uncertainty of human motion. While compelling for each task, combining these models for holistic human motion prediction is non-trivial. Conversely, holistic human motion prediction methods, which have been introduced recently, have struggled to compete on established benchmarks for individual tasks. To address this dichotomy, we study a simple yet effective model for human motion prediction based on a transformer architecture. The model employs a stack of self-attention modules to effectively capture both spatial dependencies within a pose and temporal relationships across a motion sequence. This simple, streamlined, end-to-end model is sufficiently versatile to handle pose-only, trajectory-only, and combined prediction tasks without task-specific modifications. We demonstrate that our approach achieves state-of-the-art results across all tasks through extensive experiments on a wide range of benchmark datasets, including Human3.6M, AMASS, ETH-UCY, and 3DPW. Our results challenge the prevailing notion that architectural complexity is a prerequisite for achieving accuracy and generality in human motion prediction. Code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a Transformer-based approach for human motion prediction. It combines two kinds of tasks, including pose prediction and trajectory prediction.

### Strengths
This paper presents SimpliHuMoN, a transformer-based model for 3D human motion prediction. The paper tries to prove that a single, simple architecture can achieve state-of-the-art performance across diverse sub-tasks (pose, trajectory, joint prediction), challenging the trend of increasing model specialization and complexity.

### Weaknesses
While the empirical results across multiple benchmarks are strong and the concept of unification is valuable, the paper has weaknesses in motivating its technical choices and providing deeper analysis, which currently limit its contribution.

### Questions
This reviewer has several major concerns regarding the novelty, technical contribution, and depth of analysis that need to be addressed to strengthen the paper.

1.The paper rightly identifies the importance of modeling spatio-temporal dependencies for human motion prediction. However, the core idea of using a neural network (in this case, a Transformer) to capture spatial and temporal relationships is a well-established paradigm in the field. Many existing methods are fundamentally designed with this goal. Concepts like end-to-end training are also standard practice. Therefore, while the model is well-executed, the underlying intuition might not be perceived as sufficiently novel on its own. The contribution would be stronger if framed more precisely around the specific implementation achieved by your particular architecture, rather than the general goal of spatio-temporal modeling.

2.The paper said "challenging the prevailing trend of architectural complexity" as a key contribution. While this is a valuable high-level message, from a technical perspective, this is more of a philosophical stance or an empirical observation than a concrete technical innovation. It is suggested reframing this contribution to emphasize the empirical finding that a simple, unified architecture can match or exceed the performance of more complex, specialized models across multiple tasks. 

3.THe most significant concern is the lack of a clear, intuitive explanation for why this specific, common architecture achieves such strong performance. The paper would greatly benefit from a deeper analysis that goes beyond describing the components. For instance:
Why does the unified self-attention mechanism over [C; Q] work better than a more traditional encoder-decoder with cross-attention?
What is the specific advantage of this architecture in learning the coupled dynamics between pose and trajectory compared to prior multi-stage or late-fusion approaches?
It is strongly recommend that in your rebuttal, you provide a clearer mechanistic intuition or hypothesis for the model's effectiveness, explaining the "why" behind its success rather than just the "what."

4. The role and motivation for the learnable query token are currently under-specified. The description "learnable prompts guide the decoder" is quite high-level. A more detailed explanation is necessary.

5. The discussion of the multi-person results is somewhat superficial. The paper notes that strong performance is achieved "without any explicit interaction modules," but this point requires a much deeper analysis to be meaningful.

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
5

### Summary
This paper introduces SimpliHuMoN, a transformer architecture that takes both the historical trajectories and pose joints as input and predicts the future trajectories and pose joints. Since the trajectories can be extracted from the pelvis joint, the tasks can be seen as global pose forecasting.

### Strengths
1. Breadth of evaluation across three settings (traj-only, pose-only, traj+pose) with consistent K-mode reporting and per-task K values.

2. Ablations exploring depth/width trade-offs and effect of multimodality (K>1 vs K=1).

3. The text is clearly written and easy to follow.

### Weaknesses
1. The biggest concern with this work is its novelty. The motivation of predicting global trajectory and full-body pose jointly (or condition one on the other) is not new [1],[2],[3],[4]. 

2. The proposed model architecture is a standard decoder with learnable queries, limiting the complex social interaction among pedestrians. Following this limitation, the dataset used for pose prediction only contained up to three pedestrians, which cannot reflect complex social interactions in real life. I would suggest trying more interactive scenarios like JRDB-GMP used in [3].

3. There is no ablation showing the benefit from the feature fusion of trajectory and pose. For example, when reporting the numbers on datasets like MOCAP-UMPM/3DPW, what are the performances of trajectory prediction given (1) Trajectoy-only; (2) Trajectory+Pose. By doing this, we can know if the fusion works to bring extra pose knowledge into the trajectory task. Similarly, what are the performances of pose prediction given (1) Pose-only; (2) Trajectory+Pose?

4. Missing qualitative results about the trajectory prediction task. 

[1] Adeli, Vida, et al. "Tripod: Human trajectory and pose dynamics forecasting in the wild." ICCV 21

[2] Zaier, Mayssa, et al. "A dual perspective of human motion analysis-3d pose estimation and 2d trajectory prediction." ICCV 23

[3] Jeong, Jaewoo, Daehee Park, and Kuk-Jin Yoon. "Multi-agent long-term 3d human pose forecasting via interaction-aware trajectory conditioning." CVPR 24

[4] Gao, Yang, Po-Chien Luan, and Alexandre Alahi. "Multi-transmotion: Pre-trained model for human motion prediction." CoRL 24

### Questions
About the throughput comparison (computing speed) in Table 2, how did you set up the experiment? I.e., were numbers reported as the average of multiple runs? How many samples did you use? What was the batch size and implemented hardware?

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
3

### Summary
This paper presents a unified transformer network for human motion prediction, capable of handling trajectory forecasting, pose prediction, and their combined execution. The model uses distinct embedding modules to process different inputs for each task. These inputs are then tokenized, concatenated, and fed into a shared self-attention module, which effectively mixes information and learns the complex dynamics between trajectory and pose. Task-specific prediction heads then generate the final outputs.

The authors demonstrate through extensive benchmarking that their model is a strong competitor for individual tasks and achieves new state-of-the-art results for the joint prediction of both pose and trajectory.

### Strengths
- **(S1) Unified and Versatile Architecture:** The model's key strength is its generality. A single, unified transformer architecture successfully handles pose, trajectory, and combined prediction without any task-specific modifications. This directly addresses the prevalent issue of fragmentation, where competing models are often hyper-specialized.
- **(S2) Rigorous State-of-the-Art Evaluation:** The authors conduct a comprehensive and robust evaluation across a wide range of standard benchmarks for all three tasks. The model is shown to be highly competitive against specialized methods and achieves new state-of-the-art results on the challenging joint prediction task.
- **(S3) A Compelling Case for Simplicity:** The paper makes a powerful argument against escalating architectural complexity. It convincingly demonstrates that a simple, end-to-end framework can outperform more complex, multi-stage pipelines, while also being more computationally efficient.
- **(S4) Strong Qualitative Evidence:** The inclusion of visualizations and supplementary videos is highly effective. They provide clear, intuitive proof that the model generates fluid and physically plausible motions, visually demonstrating its superiority over baseline methods that produce unnatural or static predictions.

### Weaknesses
### Major

- **(W1) Ambiguous Multi-Modal Prediction Mechanism:** The method for generating K distinct future hypotheses is unclear. Section 2.3 mentions a linear projection creates K parallel branches, but the exact mechanism is not detailed. If this is a single, large linear layer, it is not obvious how this architecture efficiently scales to the K=20 proposals required for trajectory forecasting benchmarks. The paper needs to clarify if the output head's size is fixed or dynamic, and how it handles different values of K without becoming computationally prohibitive.
- **(W2) Potential for Mode Collapse with "Winner-Takes-All" Loss:** The "winner-takes-all" loss function, which only backpropagates through the most accurate of the K proposals, is susceptible to mode collapse. The model could learn to rely on a single "favorite" prediction head, defeating the purpose of generating a diverse set of futures. The paper provides no analysis to ensure that the prediction modes are balanced and utilized effectively. A quantitative result, such as a histogram of the winning mode index over the test set, is needed to validate this design choice.
- **(W3) Unclear Role and Nature of Learnable Query Tokens:** The function of the query vectors $\mathcal{Q}_{in}$ is poorly explained. The paper should clarify their role. Are they essentially "output slots" that are progressively refined by the transformer's self-attention layers, starting from a learnable initial state? The term "learnable" itself is ambiguous, does it refer to a learned initial value for each token? A more precise explanation of this core component is necessary to fully understand the model's generative process.
- **(W4) Pacing and Focus of Explanations:** The paper dedicates significant space to describing standard, well-known concepts (e.g., the basic mechanics of a transformer decoder in Section 2.2) while glossing over the novel aspects of its own architecture. This space would be better utilized to provide the missing details on the query mechanism, the multi-modal head, and the justification for the loss function.
- **(W5) Absence of Key Multimodal Metrics:** For pose prediction, simply reporting the minimum error (ADE/FDE) is insufficient to evaluate the quality of the generated distribution of motions. The evaluation is missing standard multimodal metrics like MMADE. Without these, it's impossible to know if the model is generating genuinely distinct futures or just minor variations of a single prediction.

### Minor Weaknesses
- **(m1) Reproducibility and Robustness of Throughput Metrics:** Table 2 presents throughput in samples/second, a metric highly dependent on the hardware used. While an NVIDIA A6000 is mentioned earlier in the implementation details (L203), this should be explicitly stated in the caption of Table 2 for clarity. Furthermore, reporting a single number without confidence intervals or standard deviation over multiple runs makes it difficult to assess the stability and robustness of these efficiency claims.
- **(m2) Table Formatting:** The main results table (Table 1) uses vertical lines, which can make it appear cluttered and less professional. Adopting a cleaner format, such as the one provided by the booktabs package in LaTeX, would significantly improve readability.
- **(m3) Incomplete Supplementary Materials:** The supplementary material appears to be missing most of the qualitative video examples. While samples 8 and 10 are present, the absence of the others limits the ability to fully verify the qualitative claims of generating physically plausible and diverse motions across a range of scenarios.

### Questions
Based on these weaknesses:

**W1**

1. What is the exact architectural mechanism for generating K distinct future hypotheses?
2. Is the linear projection a single large layer or multiple separate layers?
3. How does the output head architecture scale when K=20 (as required for trajectory forecasting benchmarks)?
4. Is the output head's size fixed or dynamic with respect to K?
5. What are the computational costs as K increases, and how does the method remain computationally tractable?

**W2**

6. How do you ensure that the winner-takes-all loss doesn't lead to mode collapse?
7. Are all K prediction heads utilized effectively during training, or does the model favor certain heads?
8. What is the distribution of winning mode indices across the test set?
9. Can you provide quantitative evidence (e.g., histogram or usage statistics) showing balanced utilization of prediction modes?

**W3**

10. What exactly is the role of the learnable query tokens in the architecture?
11. Are query tokens "output slots" that are refined through transformer self-attention layers?
12. What does "learnable" mean in this context—learned initial values, learned embeddings, or something else?
13. How are the query tokens initialized and updated during the forward pass?

**W4**

14. Can you provide more detailed explanations of the novel components (query mechanism, multi-modal head) rather than standard transformer concepts?

**W5**

15. Why are multimodal metrics like MMADE not reported for pose prediction?
16. How diverse are the K generated futures—are they genuinely distinct or just minor variations?
17. Can you provide quantitative metrics that evaluate the quality of the generated distribution of motions?

**m1**

18. What hardware was used for the throughput measurements in Table 2?
19. What are the confidence intervals or standard deviations for the throughput metrics across multiple runs?
20. How stable and robust are the efficiency claims?

**m3**

21. Why are most qualitative video examples missing from the supplementary materials?
22. Can the complete set of qualitative examples be provided to verify the claims about diverse and physically plausible motions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a unified and general framework that simultaneously addresses trajectory forecasting and pose prediction by leveraging a shared self-attention mechanism to model both spatial (inter-joint) and temporal (inter-frame) dependencies.

### Strengths
The experimental evaluation is thorough. The method is validated across multiple tasks and datasets, compared against numerous state-of-the-art (SOTA) approaches, and achieves competitive results.

This work successfully demonstrates that a simple network architecture can effectively tackle this complex problem, offering a fresh and inspiring perspective for future research.

### Weaknesses
Multimodal modeling mechanism: The current approach uses only a simple type embedding to distinguish between trajectory and pose modalities, without explicitly modeling their underlying physical coupling (e.g., how gait influences arm swing).

Prediction horizon: How much past observation is required, and how far into the future can the model reliably predict? How does performance degrade as the prediction horizon increases?

Temporal jitter: Does the model suffer from jitter or unnatural motion artifacts in its predictions? If so, how is this issue addressed?

### Questions
Enhance ablation studies: Investigate the necessity of type embeddings and compare RMSNorm against LayerNorm to justify architectural choices.

Qualitative analysis: Provide more visual comparisons across datasets, visualize the diversity of multi-modal predictions (e.g., K hypotheses), and include failure case analyses with visual examples to better understand model limitations.

### Soundness
3

### Presentation
3

### Contribution
3
