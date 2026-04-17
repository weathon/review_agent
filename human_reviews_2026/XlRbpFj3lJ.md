# ODE-GS: Latent ODEs for Dynamic Scene Extrapolation with 3D Gaussian Splatting

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
We introduce ODE-GS, a novel approach that integrates 3D Gaussian Splatting with latent neural ordinary differential equations (ODEs) to enable future extrapolation of dynamic 3D scenes. Unlike existing dynamic scene reconstruction methods, which rely on time-conditioned deformation networks and are limited to interpolation within a fixed time window, ODE-GS eliminates timestamp dependency by modeling Gaussian parameter trajectories as continuous-time latent dynamics. Our approach first learns an interpolation model to generate accurate Gaussian trajectories within the observed window, then trains a Transformer encoder to aggregate past trajectories into a latent state evolved via a neural ODE. Finally, numerical integration produces smooth, physically plausible future Gaussian trajectories, enabling rendering at arbitrary future timestamps. On the D-NeRF, NVFi, and HyperNeRF benchmarks, ODE-GS achieves state-of-the-art extrapolation performance, improving metrics by 19.8% compared to leading baselines, demonstrating its ability to accurately represent and predict 3D scene dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The proposed ODE-GS method enhances dynamic scene prediction by first training a pretrained 3D Gaussian deformation (interpolation) network to reconstruct observed scenes and generate continuous Gaussian trajectories. These trajectories are then used to train a Transformer-based latent ODE model, where the ODE solver integrates latent dynamics over time to predict future trajectories.

### Strengths
The idea of introducing an ODE formulation for modeling dynamic 3D Gaussian Splatting (3DGS) is interesting. Representing scene evolution through continuous-time latent dynamics provides a different way to achieve extroplation.

### Weaknesses
1. Complexity and dependence on neural networks
The reliance on a neural network make the architecture complicate. It is not entirely clear whether this provides substantial benefits over simply using a learned deformation network.


2. Dataset simplicity
The benchmark datasets seem to include simple or synthetic trajectories (e.g., Lego, Mutant), which may be easily modeled by smooth ODE dynamics. It remains unclear how the method would perform on more complex or irregular real-world motion patterns, or on camera paths that deviate significantly from known trajectories.

### Questions
Could the authors provide a visual comparing with and without the ODE component? It would be helpful to understand why the ODE-based model has much better results than the Transformer-only baseline reported in Table 4. If visualization is difficult to include, please report additional quantitative results (e.g., using a simple baseline such as copying the last/nearest frame of interpolation network or scaling up the transformers) on the NVFi dataset for reference. 

Could the authors provide training (interpolation pretrained model and the ode part) and inference time comparisons with Deformable 3DGS and GaussianPrediction? This would help assess the computational trade-offs introduced by the latent ODE and Transformer components over baselines.


The paper states:
“...for the interpolation model, we follow Deformable GS (Yang et al., 2024) implementation.”
Does this mean the interpolation model is a pretrained Deformable-GS network reused as a data generator? If so, what is the quantitative performance of this interpolation model itself on HyperNeRF dataset over deformable GS on the interpolation mode?

### Soundness
3

### Presentation
3

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
The paper tackles dynamic scene extrapolation problem, predicting future 3D states beyond observed timestamps. The authors propose ODE-GS, which couples 3D Gaussian Splatting with a Transformer-based latent neural ODE: 1) an interpolation stage first fits accurate Gaussian trajectories within the observed window; 2) a Transformer encoder summarizes past trajectories into a latent state; 3) then a neural ODE evolves this state in continuous time, and numerical integration yields smooth, physically plausible future Gaussian trajectories for rendering at arbitrary times. Experiments on D-NeRF, NVFi, and HyperNeRF report consistent rendering quality improvement and demonstrate that ODE-GS achieves state-of-the-art extrapolation task through the proposed latent ODE.

### Strengths
- This paper is well-written and easy to understand.
- This paper targets the underexplored extrapolation problem in dynamic reconstruction and proposes a solution that makes sense.
- Across both synthetic and real scenes, the results are consistently strong, with visualizations aligning well with the quantitative metrics.

### Weaknesses
- Missing references for some important dynamic reconstruction works:
  - [CVPR 2024] Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis, by Zhan Li et al.
  - [CVPR 2024] SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes, by Yi-Hua Huang et al.
- Frozen teacher prevents end-to-end correction. Canonical Gaussians and the deformation network are frozen before training the ODE module, limiting the system’s ability to adjust interpolation when extrapolation exposes inconsistencies. It would be much better if the authors do an ablation about joint training pipeline.
- Long-horizon robustness not guaranteed. Dynamic trajectory sampling exposes the model to varied horizons, but there’s no quantitative guarantee or analysis of failure rates as horizons grow. It would be much better if the authors could do some experiments to suggest they can prevent potential drift on very long extrapolations.
- Although ODE-GS leverages a Transformer, it is essentially a per-scene reconstruction method and does not readily scale up into a large model for more generalizable tasks.
- This method relies heavily on the teacher Gaussians being well fit. When applied to real-world scenes with imperfect camera poses, those errors propagate and can severely degrade the accuracy of ODE-GS extrapolation.
- Minor typo errors:
  - L123: `representing dynamics scenes` -> `representing dynamic scenes`
  - L129: `enables flexible editing` -> `enabled flexible editing`
  - L129: Add space between `GaussianVideo (Bond et al., 2025))` and `uses`
  - L680: `focuses on` -> `focus on`
  - L683-684: `as shown in 5` -> `as shown in Figure 5`

### Questions
1. How are the Gaussian parameters encoded? Which subsets (positions, scales, rotations, opacities, SH features) are fed to the encoder?

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
3

### Summary
This paper proposes to integrates 3D Gaussian Splatting with latent neural ordinary differential equations to enable future extrapolation of dynamic 3D scenes. It models parameter trajectories as continuous-time latent dynamics and ahieves state-of-the-art extrapolation performance. 

It focuses on extrapolation, which is a somehow new problem worth studying. But this is not a totally new field and I believe so many works has been studying this issue. 

But since I am not an expert in this field, I would like read the other reviewer's opinions about the contributions of this work, especially about the novelty.

### Strengths
1. This paper combines Neural Ordinary Differential Equations with 3DGS for 4D modeling, which seems to be novel, well-motivated design.
2. This paper tried to forecast future 3D states in the context of dynamic scene reconstruction (as dynamic scene extrapolation), which is an interesting topic.
3. The paper proposes a practical and effective two-stage training strategy.
4. The paper provides consistent gains across D-NeRF, NVFi, and HyperNeRF datasets with comprehensive ablations.

### Weaknesses
1. The foremost weakness is that, there is not enough experimental results on real-world datasets. The results on PlenopticVideo dataset should be included.
2. It seems that the proposed method "collapes" on fallingball, Bouncingballs datasets. The phenomenon should be discussed and more collapsed cases need to be analyzed. The generalizability and robutness need to be discussed.
3. There has been a lot of advanced SOTA 4D reconstruction methods, but the paper only compare with 4D-GS/ Deformable-GS, which is quite old at this time.

### Questions
1. Why not supervise the extrapolation module directly with image reprojection error. Another option is using weak supervision or a distillation hybrid approach instead of relying entirely on pseudo-GT Gaussian trajectories produced by the interpolation model.
2. Beyond smoothness regularization, could you incorporate velocity bounds, momentum conservation, or constrained optimization (e.g., constraint-based pose updates)?
3. Have you ever run comparisons with end-to-end fine-tuning or mixed supervision?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper points out that existing dynamic 3D reconstruction methods can only handle temporal interpolation and fail to predict future scene dynamics. To address this, it introduces a new task called dynamic scene extrapolation, aiming to forecast future 3D states beyond observed frames. The paper proposes a model called ODE-GS combining 3D Gaussian Splatting with latent neural ODEs. It consists of three main parts: a Gaussian interpolation model for reconstructing observed scenes, a Transformer encoder for encoding motion histories into latent states, and a neural ODE module that evolves these states to predict future dynamics. The overall goal is to achieve physically plausible, temporally smooth, and consistent 3D scene extrapolation.

### Strengths
* The paper explicitly defines and tackles dynamic scene extrapolation, which predicting future 3D scene states beyond the observed temporal window. It's a meaningful and underexplored extension of existing dynamic 3D reconstruction research.

* The integration of 3D Gaussian Splatting with latent neural ODEs is conceptually elegant and well-motivated. Modeling Gaussian trajectories as continuous latent dynamics naturally enforces temporal smoothness and physical plausibility.

### Weaknesses
Interesting work. I only have a few questions regarding the experiments:

* I am curious about how ODE-GS performs in dynamic extrapolation on real-world datasets. I could not find clear visualizations of HyperNeRF and more real-world scenes (especially those containing both dynamic and static regions) in the main text or appendix. It would be helpful to see whether the model can accurately distinguish and predict the different evolution trends of dynamic versus static areas in complex real-world scenes.

* I would like to know the effective extrapolation range of the model. The paper does not provide a systematic analysis of how performance degrades with increasing extrapolation distance — i.e., how far into the future the model can extrapolate before failure. This is an important factor for evaluating forecasting models.

* I would like to better understand how ODE-GS performs on the novel view synthesis (NVS) task, both within the observed time window (interpolation) and the future extrapolation period.

* If possible, I would appreciate video demonstrations of the extrapolated results to more intuitively assess temporal consistency and motion realism.

### Questions
See weaknesses.

### Soundness
4

### Presentation
4

### Contribution
3
