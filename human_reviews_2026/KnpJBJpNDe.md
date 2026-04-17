# SMT-Learner: Movement Trajectory Learning to Decode Motor Control Strategies

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Spatiotemporal movement trajectory (SMT) representation is essential to understanding the motor skill learning and adaptation strategies that inform neurorehabilitation practices. Movement performance metrics (i.e., speed, accuracy) are insufficient to characterize motor control strategies and learning patterns, particularly in individuals with disordered movement. Motor skill learning patterns require an interpretable sequential SMT representation that preserves spatial, temporal, and performance variables. We present a novel SMT-Learner with transformer autoencoders that optimize performance-aware contrastive and adaptive transfer losses, combining cross-task and cross-subject transfer paradigms. SMT-Learner encodes trajectories into a high-dimensional latent space and enables motor performance-aware learning. We introduce an Exploration-Exploitation (E-E) analytical framework that quantifies motor skill learning and control strategies to balance different movement patterns and micro-adaptation. We tested and validated the SMT-Learner with two visuomotor reaching datasets: (1) a prospectively obtained cohort of term and preterm children's motor learning and performance of unimanual and bimanual tasks, and (2) extensively overtrained non-human primates performing target-directed reaching movements. Our ablation and baseline comparison across geometric, statistical, and clustering metrics demonstrated that SMT-Learner outperformed with the lowest reconstruction error (0.086) and optimized clinical correlation with motor performance variables. Investigated E-E patterns significantly correlated with the early and late stages of motor learning and speed-accuracy trade-offs principles. The SMT-Learner framework provides an efficient computational approach to quantify motor learning strategies; potential advanced downstream applications in developmental assessment, neurorehabilitation monitoring, and movement optimization in robotics or brain-computer interfacing.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SMT-Learner, a spatiotemporal transformer designed to learn generalized representations of motor trajectories across subjects and tasks. The model encodes sequences of normalized 2D trajectories into latent embeddings that capture both spatial and temporal structure. The training objective combines (i) a trajectory reconstruction loss, (ii) a motion-structure regularization loss, and (iii) a cross-task, cross-subject regularization term that encourages alignment of latent representations across tasks and individuals. Experiments on two datasets show that SMT-Learner can reconstruct trajectories, cluster embedding space by behavioral variables, and capture an exploration-exploitation (E-E) dynamic reminiscent of motor learning. The authors claim the model provides a task-agnostic latent space that generalizes across movement types and species.

### Strengths
- The motivation for using transformer-based architectures to capture spatiotemporal regularities in movement data is well articulated. The model structure and objectives are clearly described and easy to follow.
- The integration of cross-task and cross-subject regularization reflects an effort to develop generalizable representations that extend beyond single-dataset learning
- The embedding space shows meaningful clustering by behavioral variables, and the proposed Exploration–Exploitation (E-E) metric provides an interpretable correlate of learning progression.
- The paper presents systematic comparisons and quantitative evaluations (reconstruction accuracy, clustering, correlation analyses) supported by visualizations and ablations.

### Weaknesses
- All trajectories are rotated and scaled to align with a canonical frame, which removes absolute directionality. For instance, leftward and rightward reaches become indistinguishable. This simplifies trajectory shape learning but eliminates directional context that may encode biomechanical or cognitive asymmetries. A more balanced approach would preserve rotation or include the target direction as an auxiliary feature to retain spatial semantics.
- The description of cross-task and cross-subject training is vague. It appears that SMT-Learner is pretrained only on D1 (human data) and fine-tuned separately on D1 and D2, but this is not clearly stated. The paper lacks systematic comparisons across pretraining–finetuning combinations (e.g. pretrain on D1 --> test on D2, or vice versa), which are critical to support claims of transferability and generalization.
- The Exploration–Exploitation ratio is computed in embedding space, yet never compared to analogous metrics on the normalized raw trajectories. Without this control, it remains unclear whether the observed dynamics reflect genuine learning processes or properties of the embedding geometry.
- The claim that SMT-Learner provides a “universal embedding space for motor behavior” is premature. Both datasets involve similar reaching tasks, limiting evidence for cross-domain generalization. Demonstrating zero-shot or cross-task transfer would substantiate this statement.

### Questions
1. Does the rotation and scaling step remove left-right or up-down distinctions? If motor control or biomechanical constraints are asymmetric, how might this affect embedding quality? Could the rotation angle be preserved or added as an auxiliary input to reintroduce directional context?
2. Looking at figure 3, it’s not clear how the cross-task and cross-subject loss do not start from the same point (which I assume should be the end of the baseline). Why does the adaptive-multi loss starts from the same point as cross subject? Please clarify this point
3. Please clarify which datasets are used for pretraining and which for fine-tuning. A direct comparison among these conditions would better isolate cross-domain and cross-subject transfer effects. For example:
    - Pretrain on D1 and test on D1/D2
    - Pretrain on D2 and zero-shot test on D1/D2
    - Joint pretraining on both, then fine-tune on one
    - Within D1: train on unimanual and test on bimanual (and vice versa)        
4. How many seeds have been used for the current method and baselines? There are no confidence intervals
5. Beyond train/validation splits, is there a held-out task or session not seen during training? Evaluating zero-shot generalization on unseen subjects or sessions would confirm that SMT-Learner captures transferable motor structure rather than dataset-specific regularities
6. Have the authors compared E-E ratios computed directly on normalized trajectories versus embedding space? This would test whether the embeddings capture higher-order control features beyond geometric variability

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduced a spatiotemporal movement trojectory learner. They use transformer encoder uses performance-aware contrastive and adaptive loss for the SMT-learner. The authors utilize Exploration-Exploitation analytical framework for quantification of motor skill learning and control stategies. The framework is tested on children's motor learning and performance dataset and non-human primates reaching movements datasets. They tested multiple metrics to support their claim.

### Strengths
The author proposes the SMT-learner to ues the movement trojactory learning for the motor control decoding to overcome the shortness of using only speed and accuracy. The model uses adaptive learning with cross-task and cross-subject transfer using the loss function in formula (6).

### Weaknesses
The paper is not well written, mess up with details that already shown in the picture and figure. I would recommend putting ablation study in the last part of section 4. From line 419 to line 446, these values are very hard to read and have sense. 

Also the figures and details in the article and appendix are messed up. I would  recommend putting important figures in the article and move the experiment set up, data collcection in Appendix. 

The form that text wrap the image is not the common template in ICLR. 

The paper does not have any novel method or new found. 

he text has some typos.

### Questions
In section 4, Table 3, what is this form used for and the purpose of using this form?

For figure 1, the red line cumulative loss, why the curve goes down?

For figure 4 in page 16, can you explain the ball and moon shape using t-SNE?

For figure 5 in page 17, can you explain the dark area and third color besides term and preterm? And how you get you conclusion from your figures?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes SMT‑Learner, a transformer autoencoder for spatiotemporal movement trajectories that couples (i) performance‑aware, multi‑component contrastive objectives (completion time, path deviation from a straight line, trial success, plus task and subject factors) with (ii) an adaptive transfer mechanism that modulates cross‑task and cross‑subject regularization during fine‑tuning. On top of the learned embedding, the authors introduce an Exploration-Exploitation (E‑E) framework that quantifies novelty versus reuse of prior movement patterns across trials. The system is evaluated on two datasets: D1, a prospective iPad‑joystick visuomotor learning study in 72 children (term vs preterm; uni‑ and bimanual variants), and D2, planar reaching in over‑trained non‑human primates.

### Strengths
- The dual‑stream spatial/temporal embedding, transformer encoder, projection head, and decoder are well sketched (Fig. 2). Normalization (translation, rotation to target, scaling by end‑to‑target vector) and resampling to fixed‑length sequences are explicit, which aids reproducibility.

- The manuscript grounds the E‑E metric in decision‑making and motor‑learning neuroscience, then shows consistent empirical patterns.

### Weaknesses
- As I understood, correlations (T‑Corr, R‑Corr, S‑Corr) are used as evaluation metrics, yet these very variables enter the training objective as contrastive components (Sec. 3.2.1). High correlations are therefore partly by design and do not independently validate generalization. A stricter test would evaluate on held‑out performance variables or on tasks with different success criteria.

- STTraj2Vec reports higher T‑Corr (0.982) and R‑Corr (0.647) than SMT‑Learner (0.893, 0.539), while having much worse reconstruction (rMSE 0.386) and negative S‑Corr (‑0.797). The text asserts existing methods “fail to preserve motor performance‑relevant relationships,” but Table 2 suggests a more picture. Please explain how T‑/R‑Corr are computed for baselines, whether these methods saw the same normalization/resampling, and why STTraj2Vec’s very high T‑/R‑Corr co‑exists with negative S‑Corr. Without that, the comparative claim is hard to parse.

- It would be helpful to provide sensitivity curves over decay α, β weights, and window size; compare “min distance” versus K‑NN average distance and density‑based novelty.

- can you train with all five components, then report evaluation metrics after excluding each associated variable to demonstrate that performance holds without self‑evaluation.

### Questions
see weakness section

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This submission aims to understand movement trajectories. It uses a combination of non-human primate and human reaching data to reconstruct trajectories and correlate with clinical variables. The authors also came up with a framework that quantifies exploration vs. exploitation in trajectories.

### Strengths
The exploration-exploitation metric is interesting and gets at a good way to analyze motor learning.

### Weaknesses
The design choices in this submission seem extremely arbitrary. Although model ablations are made, it is not clear why these modeling choices were made in the first place. It is not clear how we can get at general principles with these methods. The datasets do not seem aligned to each other.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2
