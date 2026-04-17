# Enhancing structural consistency of 3D Human Pose Estimation through Trainable Loss Function

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
3D human pose estimation (3D HPE) is a challenging task due to complex structural constraints that are not well captured by standard training objectives such as mean squared error (MSE). Previous studies have attempted to enforce structural consistency by incorporating manually designed priors, rule-based constraints, or specialized architectures, which often limit adaptability. In this paper, we propose SCoTL-pose (Structural Consistency via Trainable Loss for Pose Estimation) framework that enables pose estimation models (pose-net) to learn structural dependencies directly from data, through a trainable loss function (loss-net), without explicit priors. Our approach introduces a graph-based loss-net that captures both local and global joint relationships, ensuring anatomically plausible pose predictions. While inspired by the idea of Structured Energy As Loss (SEAL), we extend it to tackle 3D human pose estimation, a task with more complex and high-dimensional structural dependencies than those considered in previous applications. To this end, we employ a graph-based model as loss-net architecture, tailored to capturing the intricate local and global dependencies among joints. SCoTL-pose can be combined with diverse backbones, from single-frame lifting networks to state-of-the-art multi-frame temporal models, without additional inference cost. To assess whether SCoTL-pose enhances structural plausibility in a quantitative manner, we also introduce Limb Symmetry Error (LSE) and Body Segment Length Error (BSLE) as evaluation metrics. Experimental results on Human3.6M, MPI-INF-3DHP, and Human3.6M WholeBody datasets demonstrate that SCoTL-pose not only reduces per-joint pose estimation errors but also generates more plausible poses, with increasing gains under more challenging settings such as single-frame or in-the-wild scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SCoTL-Pose, an extension of the Structured Energy As Loss (SEAL) framework, tailored for the high-dimensional and structurally complex domain of 3D human pose estimation. Additionally, it proposes two metrics—Limb Symmetry Error (LSE) and Body Segment Length Error (BSLE)—and demonstrates performance improvements across the Human3.6M, MPI-INF-3DHP, and H3WB datasets..

### Strengths
1. The proposed loss network (loss-net) is employed only during training, thus incurring no additional inference cost. 
2. The framework is compatible with various backbone architectures, including SimpleBaseline, VideoPose, and MixSTE. Extensive experiments validate the effectiveness of the proposed method.

### Weaknesses
1. This work represents an incremental extension of SEAL-Pose [1], which was accepted at the ICCV 2025 Workshop on SP4V. The primary modification lies in replacing the loss-net with a Graph Network to better capture skeletal topology. While this change offers structural advantages, it is largely engineering-level rather than conceptual or theoretical.
2. The reported performance differences between SCoTL-Pose (MLP) and SEAL-Pose (Margin) are inconsistent—some results improve, while others remain similar. These inconsistencies raise concerns regarding reproducibility and experimental reliability.
3. Evaluate cross-dataset generalization, e.g., training on MPI-INF-3DHP and testing on Human3.6M.
4. Extend the framework to other structured prediction tasks, such as hand pose or animal pose estimation.

[1] SEAL-Pose: Enhancing 3D Human Pose Estimation through Trainable Loss Function.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method to enhance 3D Human Pose Estimation (HPE) through structural constraints. The framework consists of two main components:
1）a graph-based trainable loss network 
2）two types of evaluation metrics (LSE and BSLE) used to assess the structural quality of predicted poses.
Overall, the proposed structure is applied to some baseline models and leads to certain performance gains. However, the paper’s organization and presentation are not sufficiently clear, and several concerns are noted below:

1. The authors state that the loss network is only required during training and does not introduce additional inference cost. However, it is unclear how the loss network can be used only during training.
2. The Pairwise Temporal Loss is not clearly formulated, and while LSE and BSLE are introduced as evaluation metrics, the training targets used to guide the optimization are not explained.
3. The relationship between the pose network and the loss network is not clearly described. It is recommended that the paper include a diagram illustrating the overall framework, showing how the pose-net and loss-net interact and what specific roles they play in improving the model.
4. The backbone models chosen for comparison are relatively outdated. The proposed SCoTL-Pose framework has not been integrated with more recent strong baselines such as MotionAGFormer or KTPFormer, which limits the significance of the reported performance. Furthermore, the MixSTE results reported in Table 1 appear to be incorrect, as the MPJPE for MixSTE should be around 40.9 mm according to the original paper. 
5.The comparative methods listed in the figures and tables should include proper citations to help readers locate and verify the referenced works, as well as the "single frame" or "multi-frame" setting marks.

### Strengths
The proposed structure is applied to several baseline models and achieves certain performance gains. It demonstrates a feasible design that can be integrated into both single-frame and multi-frame architectures.

### Weaknesses
Refer to the Summary part.

### Questions
Refer to the Summary part.

### Soundness
2

### Presentation
2

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
The paper proposes SCoTL-pose (Structural Consistency via Trainable Loss for Pose Estimation), a framework to address a key challenge in 3D Human Pose Estimation, i.e., standard losses like MSE optimize for per-joint accuracy but often produce anatomically implausible poses. Instead of using manually designed, rule-based constraints, this work introduces a trainable loss function (loss-net) that learns to assess the structural plausibility of a predicted pose. This loss net is trained jointly with the main pose estimation model in an alternating, dynamic fashion, inspired by the Structured Energy As Loss (SEAL) framework. The authors introduce two evaluation metrics, i.e., Limb Symmetry Error (LSE) and Body Segment Length Error (BSLE) to quantify structural plausibility. Experiments are performed on Human3.6M, MPI-INF 3DHP, and H3WB datasets to show the effectiveness of the proposed approach.

### Strengths
1. The design of the loss-net as a graph-based model is a key strength. Using a network architecture that mirrors the output structure, i.e., a skeleton, is an intuitive and good way to enforce structural consistency.
2. The authors validate their method on multiple datasets and six different backbone models (three single-frame, three multi-frame), demonstrating that the model is agnostic of the framework.
3. The introduction of LSE and BSLE as metrics allows for a direct, quantitative evaluation of the plausibility problem, which standard metrics like MPJPE fail to capture. The analysis in Figure 2 shows that SCoTL-pose improves LSE/BSLE even for samples with similar P-MPJPE.
4. Loss-net is only used during training and so adds no computational overhead at test time, which I think is a major practical advantage.

### Weaknesses
1. The framework involves an alternating training procedure for two networks, which is essentially a minimax game and similar to GANs which is usually difficult to stabilize. The paper admits this in its limitations ("broad hyperparameter search space," "less straightforward"). I think that this is a major practical weakness.
2. There’s no interpretability or visualization of what the loss-net focuses on. As an example, which limbs or joint dependencies dominate the structural energy? Understanding what the loss is penalizing (symmetry violations, limb length deviations, joint rotations) would make the contribution more interpretable and reliable.
3. The paper mainly compares SCoTL-Pose with vanilla supervised models or simple regularizers. It omits direct comparisons with other structure-aware or constraint-based approaches, such as kinematic tree priors, limb or bone length constraints, graphical model based pose estimators (example Pose-GCNN, kinematic refinement modules). Without such comparisons, it’s hard to understand whether the learned loss is actually superior to explicit structure enforcement.
4. Since the learned loss may rely on dataset specific geometry such as Human3.6, this limits claims of broad generalizability.
5. The temporal pairwise loss is an appealing idea but the results in Table 5 in the appendix aren’t convincing. The improvement is marginal.

### Questions
1. Please clarify the "+ Constraint" baseline in Table 4.
2. The paper states SEAL cannot be directly applied. However, the described method (alternating training, margin/NCE loss) appears to be a direct implementation of the SEAL dynamic framework. Could you clarify if the primary novelty is the application of SEAL to this complex regression task and the novel graph-based architecture for the loss-net, rather than a simple modification of the SEAL framework itself?
3. Please answer points arising from the Weakness section as well.

### Soundness
2

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
This paper proposes SCoTL, the framework introducing a trainable loss network, loss-net, for 3D human pose estimation. Unlike prior approaches relying on manual priors or rule-based constraints, SCoTL-pose learns structural dependencies from data. Extensive experiments on Human3.6M, MPI-INF-3DHP, and Human3.6M WholeBody shohw consistent improvements in both single- and multi-frame settings.

### Strengths
1. Comprehensive experiments across multiple datasets and backbones.
2. Well-written and easy to understand; clear motivation and methodology.
3. No additional inference cost, making the approach practical for integration.

### Weaknesses
1. Limited novelty beyond SEAL. The main contribution lies in applying a known concept (trainable energy-based loss) to 3D pose estimation, with relatively straightforward modifications.
2. Lack of theoretical analysis: The paper would benefit from a more rigorous examination of why the learned energy function improves plausibility or generalizes.
3. Training instability and sensitivity: The paper acknowledges a large hyperparameter search space but provides little guidance or empirical analysis of its effect.

### Questions
1. How sensitive are the results to the α coefficient balancing MSE and the learned energy term?
2. Could the loss-net trained on one dataset transfer to another without re-training?
3. How does this approach compare with explicit constraint-based or manifold-regularized methods in terms of efficiency and robustness?
4. SImilar to the first question, could the loss-net overfit to dataset-specific skeletal proportions or noise patterns?

### Soundness
3

### Presentation
3

### Contribution
2
