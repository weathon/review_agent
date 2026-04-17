# GaussianTrim3R: Controllable 3D Gaussians Pruning for Feedforward models

- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
Feed-forward methods offer a promising paradigm for novel-view synthesis, eliminating computationally expensive per-scene optimization. However, current feed-forward approaches typically predict a fixed number of pixel-aligned Gaussian primitives, leading to significant redundancy. Naively pruning these Gaussians creates severe visual artifacts, necessitating fine-tuning that compromises the feed-forward nature. We introduce GaussianTrim3R, a novel framework for controllable and feed-forward 3D Gaussian representation method which gradually prunes 3D Gaussians and simultaneously adjusts the attributes of remaining Gaussians maintaining rendering quality, thus eliminating the need for finetuning 3D Gaussians post pruning. To achieve this, we construct SuperClusters by partitioning the 3D scene based on spatial and color attributes. By leveraging Discrete Wavelet Transform, we assign and rank texture complexity to these SuperClusters, enabling selective, texture-aware pruning. Doing so enables our method to directly predict attribute-adjusted Gaussians, thereby preserving scene integrity. Unlike existing methods, GaussianTrim3R offers an efficient, real-time solution with extensive experiments demonstrating superior trade-offs between quality and efficiency across diverse real world RealEstate10K, ACID and DTU datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes GaussianTrim3R, a feed-forward framework for pruning 3D Gaussian Splatting (3DGS) representations under a limited Gaussian budget. The authors argue that current feed-forward models predict a fixed number of Gaussians, leading to redundancy.

### Strengths
1. Addresses a timely and relevant problem: efficient and controllable pruning of 3D Gaussian Splatting for real-time rendering and feed-forward models.

2. Clearly recognizes the redundancy issue in current feed-forward Gaussian prediction frameworks such as MASt3R and DUSt3R.

3. Shows a clear problem motivation and structured pipeline, even though the method itself is heuristic.

### Weaknesses
1. 2D texture → 3D pruning mismatch. Textureness is computed on image views via DWT and then averaged per SuperCluster. There is no principled treatment of visibility, occlusions, or multi-view consistency. A 2D texture proxy can be high where geometry is flat but textured (e.g., wallpaper) or low where geometry is complex but uniformly colored; both cases can cause harmful pruning. The paper acknowledges this failure mode but does not quantify it or provide mitigation.

2. Efficiency and “real-time” are not demonstrated. The paper claims real-time, controllable inference but provides no FPS, latency, GPU memory, or throughput numbers under different budgets.

3. Ablations are narrow; component necessity is unclear. The ablation table focuses on a single dataset and a couple of pruning points.

### Questions
1. How exactly is the “adaptive Gaussian expansion” implemented? Is it learned or deterministic?

2. What is the exact runtime and memory cost reduction versus MASt3R or DUSt3R backbones?

3. How does your method behave on scenes with high-frequency textures but simple geometry

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose GaussianTrim3R, a feedforward 3DGS method that can adaptively change the output Gaussian numbers based input masks and targeted budget. GaussianTrim3R first obtains the point cloud from input pair images and perform clustering + frequency analysis. These information come together to rank the texture complexity of point cloud clusters; then GaussianTrim3R select from the clusters with lowest texture to start masking away features, such that the GS head can learn to produce fewer but larger Gaussians at these regions. Such a process is done progressively until certain threshold of Gaussian budget are satisfied. Overall results indicate that this approach is better than random pruning at very high compression rate (10%).

### Strengths
The topic of feed forward Gaussian compression is timely, especially when there are a lot of input images. The current scheme of generating Gaussian per-pixel will not be sustainable.

The authors propose to train a mask-aware GS head such that the generated GS can be of larger shapes, which seems reasonable to me.

The overall performance at high compression rate seems to show that pruning over low texture regions lead to differential performance, which makes sense intuitively.

The writing is easy to understand for the most part.  

Despite my issues with the baseline comparisons, I am willing to give borderline accept given the simple but straightforward innovation in training feedforward GS models, which hasn't been explored before. I would love if the authors can confirm if e.g., this method can be made to work under e.g., 6-12 views.

### Weaknesses
While feed forward Gaussian compression is an important topic, my sense is that Gaussian count only becomes a real problem with a lot of input images. E.g., given two views, representing the scene with 200k Gaussians can be readily supported by normal computers. As such, I think it's worthwhile to see if this work can be applied in scenarios beyond two views. 

This also has ramification to comparisons. The baselines that GaussianTrim3R compare with are all designed for multi-view reconstruction, not two-view reconstruction. Particularly, if the original gaussians are not maintained and are pruned instead, it is very hard for these methods to recover the 3D scene, which is conventionally known. As such, these baselines are not very useful - more useful baselines may be e.g., InstantSplat or SPARS3R, which work on sparse view reconstruction. E.g., for InstantSplat, the setting is that existing initialization will not be pruned, and it maybe useful to see the effectiveness of these methods if the initialization can be pruned. 

The ablation shows that, random pruning/no contextual mask lead to similar performance at 40% pruning, indicating that there is a lot of redundancy in feedforward GS methods, though the 0% pruning GS metric is not posted. Differentiable results begin to emerge at 10%, which begs the question of whether this method only works when the compression is very severe, which is relatively niche.

### Questions
Are all experiments with post-inference finetuning/optimization done with only two view constraint? 

It would be very helpful if authors can list the original metrics at 0% pruning so that reviewers understand the full context of pruning.

### Soundness
3

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
This paper proposes a feed-forward 3D Gaussian Splatting (3DGS) framework with a primitive-number control algorithm. The method dynamically regulates the number of Gaussian primitives during training and achieves real-time optimization. Specifically, the approach introduces SuperClusters to group the 3D Gaussian primitives and employs a Discrete Wavelet Transform to assess texture complexity, generating masks that guide where and how aggressively to prune the Gaussians. Experimental results demonstrate that the method achieves superior performance compared with baseline approaches.

### Strengths
- Pruning is integrated into the 3D Gaussian generation process rather than applied post-training, enabling joint optimization and leading to improved performance.
- The method can accurately identify over-fitted regions and prune them precisely.
- The proposed approach achieves superior quantization results and visual quality compared to baseline methods under the same pruning ratio. Notably, at high pruning ratios, baseline methods tend to collapse, whereas the proposed method maintains strong reconstruction quality.

### Weaknesses
- The hyperparameters 𝐾 (number of clusters) and 𝑁 (pruning ratio per patch) play an important role in the proposed method. However, their values appear to be chosen empirically. It would be beneficial to explore adaptive or data-driven strategies for determining these parameters, which could improve robustness and reduce manual tuning efforts.

### Questions
I would like clarification regarding the training and inference pipeline. Are the center heads and Gaussian-splat heads trained on a large dataset during the training stage, and then kept frozen during inference? During inference, do we simply follow the framework shown in Figure 3 to produce the 3D Gaussians? Additionally, how many iterations are required to generate the 3D Gaussians at inference time—only a single forward pass, or multiple iterations? Out of curiosity, how much time needed for the inference?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a novel method for generating lightweight, high-fidelity Gaussian scenes from sparse views. Prior zero-shot methods generate pixel-aligned 3D Gaussians, which often leads to duplicated Gaussians in simple-textured areas. To address this issue, given initial points, the authors identify point clusters (K=300) and, starting from the low-textured clusters ranked by Equation 2, iteratively prune 100–N% of the Gaussians in each cluster until reaching the target Gaussian budget (Z=13k, 52k, 78k). For the experiments, since this problem is newly defined by the authors, they constructed baselines by combining existing zero-shot generation and pruning methods. Compared with these baselines, their method demonstrates significant improvements across various pruning ratios. The ablation study further justifies the necessity of the proposed components.

### Strengths
This paper defines a new problem: how to generate lightweight Gaussian scenes from sparse images. It neatly addresses the problem by identifying texture-less regions, which require fewer Gaussians for representation, generating a mask, and then generating Gaussians with consideration of the mask. Although some representations still need refinement, the paper is generally clear and easy to follow.

### Weaknesses
Please see the Questions section for the major concerns.

Presentation issues:
* In line 089, doesnt -> doesn't
* In Figure 4, the authors need to annotate which columns correspond to 40% and 80% of the Gaussians.
* In Figure 4, it would be a good idea to juxtapose a non-finetuned image corresponding to the same scene as the finetuned image to illustrate the "blobby Gaussian" artifact of the baseline.
* In Figure 4, I suggest representing the distributions using blue and yellow line curves without filling for NoPoSplat and your method. Currently, the authors fill the overlapping area with brown, which is understandable only when I check color composition palette.
* In Tables 1 and 2, I suggest reporting the performance of the baselines and GaussianTrim3R without pruning as a reference to show how much the performance of pruned version degrades from their full capacity.

### Questions
* My biggest question is whether the baselines are truly designed fairly. The gap of 8 dB seems phenomenal, but upon some reflection, it makes sense because the baselines apply pruning methods after generating Gaussians, whereas GaussianTrim3R’s GS Head directly generates hole-free scenes by leveraging a mask. I suggest a baseline that first generates pixel-wise initial Gaussians using the Mast3r backbone with some trainable Initial GS Head, then applies pruning methods to these initial Gaussians. Based on the pruning scores, a mask map can be generated and concatenated with the Mast3r features before passing into the another GS Head, similar to what GaussianTrim3R does after the Mask Generation Module. This ensures that pruning methods are applied before final Gaussian generation, making it a fairer baseline.
* The pruning method explored in this paper is limited to scene-level pruning. I suggest considering pixel-level pruning (Liu et al., EfficientGS: Streamlining Gaussian Splatting for Large-Scale High-Resolution Scene Representation, IEEE MM 2025), which ensures at least one Gaussian per ray and avoids holes.
* Ambiguous definition of “adaptive allocation” (line 455): where does the “adaptive” property come from? How is it related to disabling the training of the “Gaussian Head” in the “Without Gaussian Adaptation” ablation study?
* From an efficiency perspective: please report inference latency (fps or milliseconds). This would help readers better understand the pros and cons of zero-shot lightweight Gaussian generation.
* I found relatively fair baseline and recommend to compare with this: Fei et al., PixelGaussian: Generalizable 3D Gaussian Reconstruction from Arbitrary Views, arXiv

### Soundness
3

### Presentation
2

### Contribution
4
