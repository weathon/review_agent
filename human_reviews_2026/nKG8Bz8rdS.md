# VGPA: Deep View-Graph Pose Averaging for Structure-from-Motion

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 6, 2

## Abstract
Camera pose estimation is a key step in 3D reconstruction and view-synthesis pipelines. We present a deep, global Structure-from-Motion framework based on learned view-graph aggregation. Our method employs a permutation-equivariant, edge-conditioned graph neural network that takes noisy pairwise relative poses as input and outputs globally consistent camera extrinsics. The network is trained without ground-truth supervision, relying solely on a relative-pose consistency objective. This is followed by 3D point triangulation and robust bundle adjustment. A fast view re-integration step increases camera coverage by reintroducing discarded images. Our approach is efficient, scalable to more than a thousand images, and robust to graph density. We evaluate our method on MegaDepth, 1DSfM, Strecha, and BlendedMVS. These experiments demonstrate that our method achieves superior rotation and translation accuracy compared to deep track-centric methods while registering more images across many scenes, and competitive results compared to state-of-the-art classical pipelines, while being much faster.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the classical problem of Structure from Motion (SfM). Specifically, it proposes a pose-averaging structure-from-motion method consisting of three main components.

1. The first component is preprocessing. Given a set of input images, the preprocessing stage includes feature extraction using a DINOv2 backbone, followed by view graph construction and relative pose estimation. These steps prepare the data for the following stages.
2. The second component involves graph neural network–based pose averaging. In this stage, pose averaging is performed within the latent space of a graph neural network. A pose regression head then predicts the actual poses from the latent representations.
3. Finally, a bundle adjustment step refines the pose-averaged results, producing the final 3D reconstruction output of the Structure from Motion pipeline.

Experiments show that this method is both faster and more accurate than several state-of-the-art baselines. The experiments were conducted on multiple well-known datasets, and the results appear convincing.

In summary, the key contribution of this paper lies in the graph neural network–based pose averaging module, which introduces a novel approach to improving accuracy and efficiency in SfM. Overall, I find the experiments convincing, although I have a few questions listed in the following sections. My overall recommendation for this paper is a weak accept.

### Strengths
- The paper is well written and easy to follow.
- It presents a sound and well-motivated approach to addressing the challenging problem of pose averaging in structure-from-motion (SfM).
- The proposed graph neural network for pose averaging looks novel to me.

### Weaknesses
1. My main concern lies in the **scalability** of the proposed method. Since it relies on a graph neural network to process multiple nodes, I would like to see results on larger-scale datasets—for example, those containing 10,000+ images. The current evaluation primarily focuses on datasets with 1,000 images or fewer, and for scenes exceeding this number, the data are subsampled to 1,000 images. I believe that evaluating the method on a truly large-scale dataset would better demonstrate its advantages, especially given that the proposed pose-averaging approach is designed to handle well such scenarios.

### Questions
1. Could you evaluate the proposed method on large-scale scenes containing over 10,000 images, and compare the results with relevant baselines?
2. For baselines such as VGGT, Mast3R, and VGGSfM, which do not rely on precomputed graph information or coarse relative poses, the proposed method appears to have an inherent advantage by leveraging these inputs. How is this difference in input requirements addressed to ensure a fair comparison?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes to integrate a global edge-conditioned, permutation-equivariant GNN into the  Structure-from-Motion (SfM) pipeline  that takes as input noisy pairwise relative poses and outputs globally consistent camera extrinsics. The supervision of the model comes from enforcing consistency between predicted global poses and input relative poses, making the model fine-tunable at inference time.  After initial pose prediction, the poses are refined with a robust bundle adjustment initialized with the camera poses predicted by the network and the triangulated points.

To demonstrate the efficiency, robustness and scalability of the method the authors provide experiments on MegaDepth, 1DSfM, Strecha, and BlendedMVS showing competitive results with classical pipelines.

### Strengths
The permutation-equivariant, edge-conditioned GNN architecture that processes noisy pairwise relative poses to output globally consistent camera extrinsics trained in an unsupervised via relative-pose consistency seems novel and interesting. 

The model is tested on several datasets and the different components are ablated in order to estimate their effect.

### Weaknesses
Although the pose regression network does not use point trajectories, predict 3D points, or rely on reprojection loss, all results are provided with the final BA algorithm, which does use them. Table 7 is the only table that presents the results obtained directly with the proposed network. Therefore, it is difficult to assess the efficiency of the GNN itself.

The model appears significantly more complex than recent feedforward methods like Dust3R and its successors such as VGGT. Actually, it seems more complex than even the classical SfM pipeline. It still requires keypoint extraction, feature matching, RANSAC, and pairwise essential matrix computation to build the GNN graph. A DINOv2 feature extractor is also added to inject image-level context to the nodes. After the GNN’s feed-forward pass, it requires fine-tuning on the target scene. Finally, to obtain the final results  a robust bundle adjustment is performed, initialized with the network’s predicted camera poses and triangulated points. Given all these additional steps, it may seem surprising that this model is faster than classical SfM pipeline.  It's unclear if the authors considered all these steps in the runtime computation during their claims about speed.

The model does not seems to generalizing well between datasets as the real gain seems to come from the final fine-tuning (Table 7). Note that it is not précised on which dataset the ablations were conducted and in Table 7 the results obtained with the final BA is missing so it is  unclear how much the added components really improves the pipeline. 

 The tables are too complex, containing numerous numbers that can make it challenging to assess the performance at a glance. An additional line displaying average ranks would be beneficial for providing a comprehensive perspective on the results. 
It is also important to note that evaluating rotation and translation errors in isolation can be misleading, as the interdependence between these errors (e.g., a small rotation error paired with a large translation error, or vice versa) may obscure the true performance of the method.  A combined metric like Mean Average Accuracy (mAA)@30, such as that used in the DUSt3R paper (section 4.2), or metrics used for visual localization, such as the percentages of camera poses correctly estimated within a pair of thresholds for rotation and translation (available at https://www.visuallocalization.net), would be beneficial to provide a more accurate comparison between methods.

The proposed model, unlike newer Geometric Foundation Models (GFM) such as Dust3R and VGGT, relies on explicitly provided camera intrinsic parameters, which are not typically required in feed-forward methods designed for uncalibrated real-world image collections. While the authors argue that self-calibration in VGPA incurs only a minor accuracy loss compared to ground-truth intrinsics, this claim is validated only on the BlendedMVS dataset (Table 5), where intrinsics are consistent within scenes making their estimation easy.  However, the model’s ability to handle real-world uncalibrated datasets—where each image has unique, unknown intrinsics—remains untested.

### Questions
The authors state that one of the advantages of their method compared to GFMs lies in the scalability. However, on the one hand, their preprocessing step and the size of their GNN graph increase quadratically with the number of images. On the other hand, several GFM methods, such as Fast3R, CUT3R, and MUST3R, have been shown  to be able to process scenes with hundres or even thousands of images. 

Generally speaking, since GFM methods sometimes exhibit shortcomings in the accurate estimation of poses, it would have been more interesting to investigate in the paper whether a GNN, such as the one proposed in this article, could be trained  through self-learning based on the consistency of the relative pose to improve the poses estimated by GFM  models without using any preprocessing or BA postprocessing.  

Most GFM models uses CO3Dv2 and RealEstate10K datasets to evaluate pose estimation. It would be therefore important to run the proposed method on these datasets and compare it with the same metrics. In particular to show also the results obtained before the BA. 

Finally, as the VGPA pipeline integrates the last step BA, it would also be relevant to evaluate the accuracy of the 3D reconstruction, again preferably on the datasets used by the GFM papers such as DTU, 7Scenes, TUM-RGBD, NRGBD ...

### Soundness
2

### Presentation
2

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
This paper proposes VGPA, a permutation-equivariant GNN-based network for view-graph pose averaging. VGPA takes as input pairwise relative poses and image features and predicts globally consistent camera extrinsics in a feedforward manner. It can be trained without ground-truth camera pose labels, relying solely on relative-pose consistency.

Extensive experiments demonstrate the effectiveness of the proposed method. VGPA outperforms prior deep-based approaches while maintaining high efficiency, and it matches or even surpasses competitive optimization-based methods in accuracy.

### Strengths
1. The proposed method is simple and straightforward. It can be trained without ground-truth camera labels, highlighting its potential for scalability.
2. The experiments are comprehensive, showing the proposed method’s strong performance in both accuracy and efficiency compared to previous deep and non-deep baselines. The ablation studies further validate the contribution of each component.
3. The writing is clear and easy to follow.

### Weaknesses
1. Table 3 briefly shows results under sparse image settings (e.g., fewer than 20 views). The reviewer is curious about the method’s robustness in even more challenging scenarios, where pairwise relative camera poses may become highly noisy due to insufficient correspondences (e.g., in object-centric CO3D settings). Would incorporating image features improve robustness in such cases?
2. It would be interesting to examine the scaling behavior of the proposed method. For instance, comparing models trained with larger versus smaller datasets could demonstrate whether the method exhibits desirable scaling properties.

### Questions
1. In L372–375, the authors mention that the method is robust across a wide range of k values. Are the numbers of nearest neighbors (k) reported in Table 4?
2. In Table 7, Row 1 shows that subset sampling is quite important for the proposed method. Do the authors have any intuition for this behavior? Additionally, could they conduct an ablation study on the effect of different proportions of kept samples on performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a deep-learning based global Structure-from-Motion framework based that use learned view-graph aggregation, using a permutation-equivariant, edge-conditioned graph neural network. The GNN starts with the standard noisy pose-graph and outputs globally consistent camera extrinsics. The GNN is trained on a relative-pose consistency objective in a self-supervised manner. A view re-integration step is described to incrase camera coverage by examining discarded images. Results are shown on multiple datasets.

### Strengths
The GNN framework/design seems better than prior GNN-based methods.

### Weaknesses
Results seem very mixed. VGPA does well only in half the cases presented, even over GNN-based methods. More importantly, the design choices are not explained or motivated well, making it all appear ad-hoc.

### Questions
Overall, I am not enthusiastic about the work as presented in this paper. Here are some of my comments and observations. These aren't specific questions to be answered by the authors, but points that I have problems with.
- No clear motivation or intuition given on the GNN design. I would expect more reasons given to convince the ICLR audience of the method. $\phi_e$, $\phi_m$, and $\psi$ are just presented. Why these designs and why not some other ones? Why was $\log R_{ij}$ used?
- What is the structure of $\phi_e$, $\phi_m$, and $\psi$? Are the fully connected? Linear? Activation function?
- The pose-regression head $H_{cams}$ is described as a 3-layer MLP. What is the architecture of the others?
- The loss (Eq 1) treats the RANSAC-estimates from COLMAP of R and t as the ground-truth and the loss aims for estimates to match those. I am not sure it is correct to describe it as "unsupervised"; ground-truth is estimated using COLMAP. This also ensures the presented method cannot be better than RANSAC method.
- My major dissatisfaction is about the results. VGPA seems better only in half the cases in Table 1. In particular, classical methods register more cameras.  How do we make a case for VGPA based on these results? Are all “test”scenes of MegaDepth reported here? Does an average of all make sense?
- The results on MegaDepth scenes is varied too, with some doing better, others worse. Is there anything to be learned by carefully examining the scenes that do well and otherwise? Any observations on those?
- No clear advantage of VGPA shown other datasets.  Strecha results seem better, but not BMVS. How can we say it does better? Any deeper analysis?
- Based on highly mixed results, what is a strong case for VGPA? I am not able to find a clear case.
- Re-integration is a tangential point. Other methods also may be able to do it. What is unique about it for VGPA?
- Similarly, what is the importance of "uncalibrated" results (Tab 5) there?
- I find the presentation of "$N_r/t$" strange and even misleading. It makes it appear as if the method can register more cameras if run for more time. Why is it relevant?
- VGPA seems to do well in total time for most datasets. I would recommend reporting on that fact alone.
- Regarding ablations, I think the choices to use $\phi_e$, $\phi_m$, $\psi$, and $H$ at least as relevant. What is the impact of other design choices for them?

### Soundness
2

### Presentation
3

### Contribution
2
