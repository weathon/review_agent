# DrivingRecon: Large 4D Gaussian Reconstruction Model For Autonomous Driving

- Decision: Reject
- Scores: 5, 6, 5

## Abstract
Photorealistic 4D reconstruction of street scenes is essential for developing real-world simulators in autonomous driving. However, most existing methods perform this task offline and rely on time-consuming iterative processes, limiting their practical applications. To this end, we introduce the Large 4D Gaussian  Reconstruction Model (DrivingRecon), a generalizable driving scene reconstruction model, which directly predicts 4D Gaussian from surround-view videos. To better integrate the surround-view images, the Prune and Dilate Block (PD-Block) is proposed to eliminate overlapping Gaussian points between adjacent views and remove redundant background points. 
To enhance cross-temporal information, dynamic and static decoupling is tailored to learn geometry and motion features better. Experimental results demonstrate that DrivingRecon significantly improves scene reconstruction quality and novel view synthesis compared to existing methods. Furthermore, we explore applications of DrivingRecon in model pre-training, vehicle adaptation, and scene editing. Our code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a feed-forward 4D reconstruction method that generates 4D scenes from surround-view video inputs in a single feed-forward pass.
The method involves 3D Position Encoding, Temporal Cross Attention, Gaussian Adapter, and Prune and Dilate Block. All these modules consist of the feed-forward 4D reconstruction pipeline. 
The PD-Block learns to prune redundant Gaussian points from different views and background regions and dilate Gaussian points for complex objects, enhancing the quality of reconstruction.
This paper also presents rendering strategies for both static and dynamic components, enabling efficient supervision of rendered images across temporal sequences.

### Strengths
1.This paper first explores a feed-forward 4D reconstruction method for surround-view driving scenes, which promotes the development of feed-forward technology in the field of 4D reconstruction.

2.The proposed PD-Block learns to prune and dilate the Gaussian points and allows for Gaussian points that are not strictly pixel-aligned, which is innovative.

### Weaknesses
1.The training process requires depth map ground truth, whereas comparison methods like Pixelsplat and MVSpalt can be trained without it. This reliance on depth ground truth during training restricts its practical applicability.

2.The dynamic objects are decomposed through segmentation and have only few categories (vehicles and people). This approach only separates dynamic and static pixels based on semantics, limiting its ability to achieve comprehensive 4D reconstruction of all dynamic objects.

3.Compared to scene-optimized methods, feed-forward reconstruction provides the advantage of generalization, eliminating the need of test-time optimization for each new scene (though it may lead to some decrease in accuracy compared to the scene-optimized method). In the papers of comparing methods MVSplat and PixelSplat, both of them present running time and memory consumption, demonstrating the efficiency of their feed-forward approaches. However, in this paper, while the authors claim their method is feed-forward, they do not provide an analysis of its running time and memory usage. I recommend including this efficiency analysis and comparing it with other methods to strengthen the evaluation. 

Besides, if the authors believe that efficiency is not a concern of this paper, then comparisons with other offline scene-optimized methods (e.g., DrivingGaussian) should be included.

4.If the possible application is to develop real-world simulators in autonomous driving (mentioned in the abstract of the paper), then there is no high requirement for the efficiency of reconstruction, and the existing offline scene-optimized 4D reconstruction method is also acceptable. However, feed-forward does not seem to have an advantage in terms of reconstruction accuracy.

### Questions
1.What does “DA-Block” in line 202 refer to? It is not mentioned in the context.

2.Please refer to the questions and suggestions in the Weaknesses part.

### Soundness
3

### Presentation
2

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
This paper proposes a learning-based reconstruction method in a feed-forward manner in driving scenarios. It could predict 4D Gaussian primitives from multi-view temporal input. It is a very early work that explores learning-based generalizable reconstruction and rendering for autonomous driving. This paper also introduces a couple of downstream applications such as model pre-training and vehicle adaptation.

### Strengths
-  It is a very early work that explores learning-based generalizable reconstruction methods for autonomous driving, demonstrating this paradigm could work in real-world driving scenarios.
- This paper is comprehensive since it not only develops the methods but also incorporates potential applications such as perception and driving tasks.
- The self-supervised pretraining task is insightful.

### Weaknesses
- This paper does not demonstrate the model's generalization to different viewpoints. The authors claim the ability of vehicle adaption. However, only the camera intrinsic is changed. Could the predicted 4D Gaussians produce good rendering quality in viewpoints beyond the driving trajectories (different extrinsic)? A recent work[1] explores this direction.

- The resolution is relatively low.  The produced rendering quality cannot meet the requirements of practical use, such as camera simulation.

- It would be better to show the inference latency.

- The authors do not provide video demonstrations of the rendering results. It is hard to have a intuitive understanding of the actual performance.

[1] Freevs: generative view synthesis on free driving trajectory.

### Questions
How does the scene edit (Fig.6) work? This procedure can be more detailed.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Unlike previous methods (e.g., 3DGS/NeRF) that require thousands of iterations to reconstruct a scene, this work aims to predict a 3D scene representation using a neural network.

 The authors make several design choices to make this pipeline work (PD-block, regularization, 3D encoding, etc.). 

Experiments conducted on Waymo demonstrate better performance compared to other generalizable approaches.

### Strengths
1. A generalizable and scalable approach that allows training of large models to learn priors from extensive data, generalizing to novel scenes.
2. **Almost** no 3D bounding box labels required for dynamic scenes, enhancing scalability.
3. Detailed explanations and extensive experiments on cross-data evaluation, downstream applications (data augmentation, pretrained model for perception, scene editing).

### Weaknesses
1. Overcomplicated design:
   While I appreciate the effort in developing a generalizable model with dynamic-static decomposition, the model seems quite complex, requiring:
   * Multiple modules (image encoder-decoder, temporal cross-attention, Gaussian adapter, PD block, etc.)
   * Numerous regularization terms
   * Several pretrained models (DepthNet, DeepLab, SAM)

   This complexity may hinder downstream applications when used as a pretrained model. For instance, how fast is the model? Is it efficient enough for use in autonomy systems?

2. The realism is still lower compared to optimization-based approaches (e.g., 3DGS), and can only operate on low resolution (256x512) with a limited number of images.

3. (Minor point) The writing seems somewhat rushed, lacking thorough proofreading. Some potential issues:
   * L155, "corresponding intrinsic parameter E" should be K
   * L414 "evaluation on NOTA-DS6" should be Diversity-54

### Questions
**Regarding efficiency and comparison with 3DGS**

What is the computational cost to train the model (how many hours on 24 GPUs)? 
How long does it take to reconstruct a 3D scene representation using your approach during inference? How does the efficiency compare to 3DGS, e.g., StreetGaussian on 256x512?

How does the realism compare to 3DGS (e.g., StreetGaussian at 256 × 512)? It's okay if it's worse; I'm just curious. 



**On 3D labels**
What is the performance without using 3D bounding boxes at all? I note that you use 3D bounding boxes as prompts for SAM. A label-free approach would make this work more impactful.

**On downstream applications**
How is UniAD implemented in Waymo? Would it be possible to conduct your experiments on nuScenes to follow the setting/implementation of UniAD?

**Miscellaneous**:
* How many frames are in the input during training?
* In Table 4b, what does "Training Num" refer to? Do you mean number of scenes? The PSNR seems quite high compared to Table 3.

Some questions may require additional experiments; please disregard if they're not feasible. However, I'm particularly interested in the efficiency and comparison with 3DGS.

### Soundness
3

### Presentation
3

### Contribution
2
