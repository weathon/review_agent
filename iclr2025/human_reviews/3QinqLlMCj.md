## Human Reviewer 1

### Summary
Given G.T. camera intrinsics, the authors first leverage the existing work to obtain the coarse camera poses and depth. Then to refine the estimation, the authors design a module to learn the depth offset estimation with the help of an existing depth estimation network. Furthermore, the camera pose refinement is conducted in another module. The idea of feedforward pose estimation is interesting, but there is still a gap between the performance of the proposed method and some per-scene optimization methods. Since I did not see the authors report any inference time result and I believe some static scene pose-free per-scene optimization methods (CF-3DGS, ..) are very fast and accurate, I expect the authors to provide more comparisons. There are still some questions and limitations raised below. I will consider improving the grade upon the feedback from the authors.

### Strengths
1. The authors proposed a new pose-free feed-forward method in camera pose estimation.
2. The authors conducted enough ablation studies to present the contribution of each component of their method.

### Weaknesses
1. The biggest limitation of this work is the requirement of G.T. camera intrinsic.
2. The performance on RealEstate-10K seems to be SOTA, however, the performance on ACID is not. Does it mean such a method does not generalize well to the outdoor scenes?
3. So I expect more comparisons on DL3DV or more public datasets (like DAVIS[1], iPhone[2]) to prove the effectiveness of the proposed method. 


[1] Pont-Tuset, Jordi, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. "The 2017 davis challenge on video object segmentation." arXiv preprint arXiv:1704.00675 (2017).

[2] Gao, Hang, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. "Monocular dynamic view synthesis: A reality check." Advances in Neural Information Processing Systems 35 (2022): 33768-33780.

### Questions
1. Can the authors provide more insights on how you obtain the coarse camera parameters? I guess the authors directly implemented the existing work to obtain those things.
2. the authors trained different checkpoints on different datasets in the implementation details. Does it mean the 'feed-forward' claimed by the authors is actually dataset-specific? If so, I think it is a big limitation of the proposed method lacking the generalizability to different scenes.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper focuses on the pose-free feed-forward novel view synthesis task. It leverages a pre-trained monocular depth estimation model and a visual correspondence model to generate coarse pose and depth estimates, enhancing the stability of the 3DGS training process. Subsequently, a lightweight refinement model is used to further improve the depth and pose estimations. Extensive experiments have been conducted to demonstrate the effectiveness of the proposed method.

### Strengths
S1 - This paper addresses a meaningful task with significant potential for real-world applications.

S2 - This paper leverages two robust pre-trained models to generate initial pose and shape estimates, which significantly enhance the model's performance.

S3 - The paper is well-written, and the experiments are logically sound.

### Weaknesses
W1 - The performance of this method appears to be highly dependent on the coarse pose and depth results provided by the pre-trained model.

W2 - The paper lacks qualitative results for the pose estimation, which would provide a clearer assessment of the model's performance in this area.

### Questions
Q1 - How would the results differ if alternative coarse prediction networks, such as Dust3r[1], Mast3r[2], or others, were used?

Q2 - Qualitative results for the pose estimation task.

[1]Wang S, Leroy V, Cabon Y, et al. Dust3r: Geometric 3d vision made easy[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 20697-20709.
[2]Leroy V, Cabon Y, Revaud J. Grounding Image Matching in 3D with MASt3R[J]. arXiv preprint arXiv:2406.09756, 2024.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper introduces a method to tackle the challenging problem of novel view synthesis (NVS) from unposed images in a feed-forward manner. They identify the issue in previous pixel-aligned 3DGS methods, where the predicted 3D Gaussians from different views have the problem of misalignment, leading to noisy gradient flows and poor NVS results. They propose a method where they don’t need to rely on the poses obtained from off-the-shelf tools. Instead, they leverage pre-trained monocular depth estimation and visual correspondence models to obtain coarse alignment, with further refinement of the depth map and poses. Results show that among the pose free methods, they can perform decently better results for NVS tasks. However, for pose estimation, it is still worse than general methods like Mast3R.

### Strengths
1. The authors tackle the problem of 3D Gaussian splatting from unposed sparse images, which is an interesting and important topic
2. The authors apply the recent state-of-the-art depth estimation and pose estimation methods for coarse pose alignment, and further introduces pose and depth refinements, to some extend improving the final performance.
3. The paper is overall well-written and easy to follow in most parts.

### Weaknesses
**The method section is overall clear, but missing some details and discussions**

*Unclear descriptions in camera pose refinement in Sec 3.2.3*  
1. It is quite unclear to me how exactly it is done. From your writing, it makes me feel that you first will get a newly computed camera poses $\hat{P_{ij}}$ similarly as done in the coarse step, but with the refined depth. This $\hat{P}_{ij}$ is already the refined poses, right? However, you further have another refinement step as shown in Eq. (2). What are the rationale before?
2. what is the T_pose network? What is the E_pos in eq. (2)?

*Cost volume*  
In section 3.2.4, for the “cost volume construction and aggregation”, is that any different to the MVSplat paper? Can you justify the differences?

*2D-3D Consistency loss*  
Line 291-292, you said that you improve the robustness of the model in regions with low texture or significant viewpoint changes. However, the correspondences from the feature matching methods like LightGlue do not provide many correspondences in those low-texture regions. I guess you cannot claim the robustness there? 

*Unclear implementation details*  
What is the frame distance from 45 to 75? Do you mean you sample one frame every 45/75 frames in the video sequence? You might want to make this point clearer. And why for DL3DV, you only sample every 5 or 10 frames, way smaller than ACID or RE10LK?
Also, you mention that you train for 40000 iterations on a single A6000 GPU, is it the same for all three datasets? If true, I think it might not make too much sense? As you mentioned in line 351-354, RE10K has ~21K videos from training, while you only use 2K scenes for the training of DL3DV?



**The experimental section is convincing in general, but lacks some important experiments / baselines and explanation.**

1. For pose estimation comparison, the sota method right now is RoMa [1], I think it is fair to ask to compare to it.
2. I wonder why your method is still lacking behind Mast3R on the pose estimation for both RE10K and ACID, even if Mast3R is is not even trained at all on those dataset, while your method was trained on those dataset separately.
3. Why for novel view synthesis in Table 1, you don’t show the comparison to the recent pose-required methods pixelSplat?
4. Based on the experiments you show, your pose estimation is worse than Mast3R in almost all cases, and the NVS results are also worse than MVSplat in all scenarios (even on DL3DV in Table 3, where MVSplat was not trained on). One reasonable baseline to me is, get camera poses from Mast3R, and then run MVSplat directly. I wonder how your method compares to such a simple baseline?
5. In your ablation study table 4, what is the point of adding V, I-I, I-II, I-V, if they are just all N.A.? That is really weird to me. You can just describe them in texts.


**Writing**  
Sec 3.2.1: You use two paragraphs to motivate by mentioning the limitations of the previous methods. The real content for the coarse alignment is really just the third paragraph between line 186-191. The motivations part is actually unrelated to the coarse alignment but more on why your method is needed, so you should just put them in the introduction instead.

[1] Edstedt et al.: RoMa: Robust dense feature matching, CVPR 2024

### Questions
As I already discussed in the weakness section, it would be very imporatnt if you can justify those points in the experiments.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
5

---

## Human Reviewer 4

### Summary
This paper introduces PF3plat, a novel framework designed for novel view synthesis from unposed images in a single feed-forward pass. PF3plat leverages pre-trained monocular depth estimation and visual correspondence models to achieve an initial coarse alignment of 3D Gaussians. Subsequently, PF3plat incorporates refinement modules and geometry-aware scoring functions to further refine the depth and pose estimates derived from the coarse alignment to enhance the quality of view synthesis.

### Strengths
1. The task of novel view synthesis from unposed images in a single feed-forward pass is highly practical.
2. The paper demonstrates state-of-the-art results on Re10k and ACID, showcasing the effectiveness of the proposed method. 
3.  The refinement modules designed in the paper have significantly improved the effectiveness.

### Weaknesses
1. PF3plat leverages a robust solver for pose estimation between each pair of cameras; thus, increasing the number of viewpoints significantly extends the feed-forward pass time.
2. PF3plat relies on the coarse alignment of 3D Gaussians, and a small overlap may affect the quality of the correspondence model.

### Questions
1. Why does using the pixel-wise depth offset estimation model promote consistency across views? (line 211)
2. How about the performance of PF3plat in dynamic scenes?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4