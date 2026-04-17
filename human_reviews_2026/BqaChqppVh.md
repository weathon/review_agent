# Trace Anything: Representing Any Video in 4D via Trajectory Fields

- Decision: Accept (Poster)
- Scores: 10, 2, 6, 4

## Abstract
Building 4D video representations to model underlying spacetime constitutes a crucial step toward understanding dynamic scenes, yet there is no consensus on the paradigm: current approaches resort to additional estimators such as depth, flow, or tracking, or to heavy per-scene optimization, making them brittle and hard to generalize. In a video, its atomic unit, the pixel, follows a continuous 3D trajectory that unfolds over time, acting as the atomic primitive of dynamics. Recognizing this, we propose to represent any video as a Trajectory Field: a dense mapping that assigns each pixel in each frame to a parametric 3D trajectory. To this end, we introduce Trace Anything, a neural network that predicts the trajectory field in a feed-forward manner. Specifically, for each video frame, the model outputs a series of control point maps, defining parametric trajectories for each pixel. Together, our representation and model directly construct a 4D video representation in a single forward pass, without additional estimators or global alignment. We develop a synthetic data platform to construct a training dataset and a benchmark for trajectory field estimation. Experiments show that Trace Anything surpasses existing methods or performs competitively on the new benchmark and established point tracking benchmarks, with significant efficiency gains. Moreover, it facilitates downstream applications such as goal-conditioned manipulation, simple motion extrapolation, and spatio-temporal fusion. We will release the code, the model weights, and the data platform.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper introduces a feed-forward method to directly represent the dynamic video in a continuous trajectory field, which can predict the 2d trajectories (2d in a way it predicts for the image coordinate) for the image sequence at any queried time frame. Specifically, this method uses B-splines as basis functions and uses the predicted feature-based “points” to get the final trajectory. The “control points” are predicted using the large-model image encoder. The authors have shown good performance on video/image pairs to the trajectory dataset DAVIS and BridgeData V2, along with other applications including motion forecasting, spatio-temporal fusion, etc, showing the great potential of this proposed trajectory field representation.

### Strengths
- To directly represent the image sequence/video as a continuous trajectory field is reasonable and has many advantages, such as no need for depth/geometry/camera priors, can directly query the motions, etc. 
- This continuous trajectory field representation has many practical applications, as shown in the paper. For example, the goal-conditioned trajectory generation for robotic manipulation. This could be a great future work.
- As a direct feed-forward network, the computation is efficient and the model is fast during inference.
- The authors have also released (will release) a benchmark dataset for the trajectory estimation, which is needed from the research community.
- The paper is well-written, and all the figures and included videos are of high quality.

### Weaknesses
- The authors could provide more experimental results on video to trajectory datasets to further highlight its robustness and generalizability.
- I am a bit confused by the definition in the paper referring to 3d points. It seemed that the predicted “3d points” in the paper referred to the (u,v) coordinate at a specific time frame t. It would be better to make this notation clearer.

### Questions
- Can the model handle large camera motions? I found that in the 
- Could the authors provide an intuitive explanation for the condition 2? 
- I wondered if the authors have trained their image encoder end-to-end for the experiments they showed in the paper or used a pretrained model here.
- What about the generalizability of the proposed method? I have seen one experiment in the appendix, how about more different OOD data? Also, does this generalizability come from the large-model image decoder or the continuous trajectory representation itself?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Trajectory Fields which map each pixel in each frame to a parametric 3D trajectory. To learn the Trajectory Field from videos, the paper proposes Trace Anything module, a feed-forward network to predict control points of B-splines for each pixel in each frame. 

To facilitate the training of the proposed model, this paper builds a synthetic video dataset with dense labels, covering 2D/3D trajectories, depth maps, camera poses, and semantic masks.

### Strengths
1. It is new to formulate 4D video understanding as pixel-level Trajectory Fields.

2. Collecting a densely labeled dataset for dynamic scene understanding is a very good contribution.

3. The overall presentation is clear and easy to follow.

### Weaknesses
1. Modeling 3D trajectories with cubic B-splines over-simplifies the actual movement of 3D points. Complex trajectories cannot be well described with cubic B-splines (p = 3) used in this paper. This means that the proposed method is hard to model complex scenarios. 

2. The training of Trace Anything requires dense annotations, which are rarely available in most real-world datasets. When trained solely on synthetic data, the Trace Anything model inevitably suffers from a significant domain gap in real-world applications. This gap encompasses not only differences in geometry and appearance, but also in motion distribution, thereby limiting the scalability of the proposed method.

3. The training objective comprises six terms, yet the paper lacks ablation studies to assess the effectiveness and necessity of each component. Furthermore, these loss functions require extensive additional data annotations, including labels for static and rigid pixels.

4. Experiment settings and results need to be more comprehensive. 

4.1. For clarity and reproducibility, it is recommended to explicitly list the training datasets and labels used for all baseline methods.

4.2. In the quantitative comparison on the newly collected synthetic dataset (Trace Anything benchmark), 3D tracking evaluation metrics should be included in addition to trajectory field estimation.

4.3. Quantitative results on real-world datasets should be provided, as this is a crucial evaluation of Trace Anything’s ability to generalize beyond synthetic data.

### Questions
1. Given a video of a dynamic scene, is the information sufficient to predict 3D trajectories for every pixel, considering that there are occlusions? If not sufficient, then the motivation of the proposed method is problematic. 

2. Since the Trace Anything benchmark evaluates trajectories for pixels sampled from all frames instead of only the first frame, how are the current point tracking approaches adapted to this evaluation protocol?

3. For quantitative comparisons in Table 1, 2, A and B, have other baselines been trained on Trace Anything benchmark dataset? and how?

4. What is the fundamental difference between “all-to-all” evaluation and “first-to-all” evaluation?

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
This paper proposes a feed-forward network to estimate the 3D trajectories for every pixels in every timestamp, called trajectory fields. To be specific, they use b-splines to model the 3d trajectories, and the control points of the b-spline for each pixel are predicted by a fusion transformer. Thus the each pixel is traced along time. Furthermore, they develop a synthetic data platform, which could benefit training and benchmarking. Several promising results are shown in both video and paired images datasets.

### Strengths
1. The proposed model achieves to reconstruct and represent the dynamic monocular videos in a feed-forward manner, making the whole process more efficient.
2. The idea is complete, and shows promising results on two datasets. 
3. The model can work on both sequential videos and unsequential paired images. 
4. The model proposes an novel framework to fuse the information across different timestamps in one feed-forward inference step.

### Weaknesses
1. The presentation is confusing. 1) The authors try to elaborate trajectory field in a formal ‘field’ definition. However, this definition is kind of counter-intuitive. The trajectory field is defined as a function over a discrete sequential image space, but not a trajectory function over space and time, making it hard to understand for many readers, since the mathematical definition always bothers. This would lead to misunderstandings, like someone may assume the trajectory is continuously distributed in the space, but it’s not. Therefore, I kindly request the authors to further elaborate the definition in the next version for easier understanding. 2) The relationship between control points number D and time t is not well discussed. If I understand right, the control points are located uniformly on the spatial temporal line, with k=0 the first frame (t=0) and k=D-1 the last frame (t=1). But the presentation about this part is extremely unclear.
2. Following weakness 1, the definition over discrete space limits the spatial interpolation. In other words, it does not support tracing a new point in the space by the learned trajectories. 
3. The experiment is not complete. First of all, since the training requires masks to segment static parts, which is not used for other end-to-end models. So it’s unclear how important this design contributes to the performance. Ablation study is missed here. Second, there is no experiment about motion interpolation along time. For example, given 0, 0.2, 0.4, ..., 1 to predict, eval the model on intermediate time 0.2, 0.3, ... Based on the ablations in Table C, it seems the number of control points do inflence the performance. This shows the temporal interpolation is sensitive to the control point density. Thus this experiment is necessary. 
4. The control point number is fixed, which constrains the input video length. Since the motion is smoothly interpolated by b-spline basises, the motion complexity is constrained by the number of control points. 
5. The motion forecasting in section 5.3 is not persuasive. The tangent continuation only works in linear constant straight motions, no rotations and accelerations are supported. As for instruction-based forecasting, that ability comes from the video generation model but not trajectory field, thus this should not be claimed as an emergent capability.

I’m open to modify my rating to a large extent based on the authors’ response and other reviewers’ opinions.

### Questions
1. How the temporal index embeddings are injected?
2. How the rigid region in equation 12 is determined?
3. Why the correspondence regularization in equation 13 is not enough to regularize static consistency (C1)?

### Soundness
3

### Presentation
2

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
The paper introduces a new representation called, Trajectory Fields, for 4D geometry and temporal understanding. Given a 2D video, it reconstructs the dynamic scene in 3D using a set of trajectories. Each trajectory is considered as a small point (or atomic primitive) is moving in 3D space. Trained on a proposed dataset, the method outperforms all existing methods on benchmark datasets

---

The introduced representation is novel and working. However, there are some remaining concerns on missing comparison and unclear details. My initial rating is **4: marginally below the acceptance threshold**, but would like to raise the rating based on the response with a much better understanding of the paper.

### Strengths
* **Good, novel representation**

  Existing dynamic 3D reconstruction approaches have a disentangled representation for geometry and motion. To overcome it, the paper proposes a single unified representation, called Trajectory field. In a nutshell, this is similar to dynamic 3DGS but without 3D Gaussian attributes. However, new component proposed in the paper is a controlled point concept that represent the trajectory with a set of controlled points and basis function (eg., B-spline)

* **Good accuracy**

  The proposed method outperforms existing methods by a large margin with the fastest runtime in the proposed benchmarks.

* **Dataset**

  If the proposed dataset is released, it will be a good contribution to the community.

### Weaknesses
* **Accuracy on other public benchmark datasets** is missing

  The paper evaluates accuracy on the proposed benchmark only (Table 1 and Table 2), which can be close to the domain that the model is trained on. At least, Table A and B in the supplementary provides accuracy on PointOdyssey and TAP-Vid3D dataset. On PointOdysey, the method performs very competitive as it's on the synthetic domain whereas it has mixed results on the real-world based dataset in Table B. It would be curious to see how the method works on other real-world benchmark datasets (eg., KITTI SceneFlow 2025).

* **Evaluation on 3D pointmap** is missing

  As a related (or downstream) task, the method can naturally output 3D pointmaps. It would be curious to see how the method performs on this 3D pointmap prediction task, comparing to the state-of-the-art reconstruction method, eg., MASt3R, VGGT, and its variant family, also on standard **real-world** dataset such as Bonn, TUM-Dynamics, etc. 

* **Confusing notations**

  To be honest, it may need more details to fully understand the equations related to the representation. Here are some questions on the notation for clarification. Assuming we have N input frames (indexed as 0, 1, 2, ..., N-1), 

  * Time interval for D? D is the number of control points. Does it mean that there are D control points between each frame pair (eg., 0th and 1st), resulting in N * D total control points in the output sequence? or is it like N frames share the D control points?
  * Instead of using the basis function, what if we assume a linear motion? Does it break the formulation? (not so sure if I understood the concept correctly). Does the accuracy drop significantly?
  * Why is $t$ in Eq. (1) defined [0, 1] not [0, N-1]? Does $t=1$ mean the timestep that corresponds to the last frame (ie, N)? or does it mean any other thing?

* **Missing loss ablation study**

  The paper proposes multiple loss functions from Eq. (10) to (13), but ablation study is missing. It would be good if the paper can provide ablation study and validate the design choice on the loss functions.
 
* **Overlaying all trajectories from all frames?**

  If the method aggregates all trajectories from all frames $I_i$, this will be quite computationally expensive to represent the scene and can be redundant (eg., it is not like that a very minimal set of atomic primitives move over time and represent the dense 4D scene). How is the method actually implemented? Are there any merging mechanism? (or is this question from my misunderstanding of the paper?)

### Questions
* $L_{corr}$ in Eq. (13) seems important for ensuring the 3D consistency of trajectory. What's the loss value look like when the model is converged?

### Soundness
4

### Presentation
2

### Contribution
4
