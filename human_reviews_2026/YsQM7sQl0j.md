# Generative Human Geometry Distribution

- Decision: Accept (Oral)
- Scores: 6, 6, 8, 2

## Abstract
Realistic human geometry generation is an important yet challenging task, requiring both the preservation of fine clothing details and the accurate modeling of clothing-body interactions. To tackle this challenge, we build upon Geometry distributions—a recently proposed representation that can model a single human geometry with high fidelity using a flow matching model. However, extending a single-geometry distribution to a dataset is non-trivial and inefficient for large-scale learning. To address this, we propose a new geometry distribution model by two key techniques: (1) encoding distributions as 2D feature maps rather than network parameters, and (2) using SMPL models as the domain instead of Gaussian and refining the associated flow velocity field. We then design a generative framework adopting a two-staged training paradigm analogous to state-of-the-art image and 3D generative models.
In the first stage, we compress geometry distributions into a latent space using a diffusion flow model; the second stage trains another flow model on this latent space. 
We validate our approach on two key tasks: pose-conditioned random avatar generation and avatar-consistent novel pose synthesis.
Experimental results demonstrate that our method outperforms existing state-of-the-art methods, achieving a 57% improvement in geometry quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of 3D human geometry generation by proposing a new geometry distribution model that extends this representation from single characters to large-scale dataset learning.  

Specifically, its proposed representation incorporates two key designs:  
(1) Encode geometry distributions as 2D feature maps instead of network parameters to enable dataset-level generalization.  
(2) Replace the Gaussian source distribution with the SMPL human template (plus refined flow velocity fields) to align the source closer to the target geometry distribution.

Based on this representation, this paper also proposes a two-stage generative framework:  
(1) Compress each human geometry distribution into a compact 2D feature map using a flow model, from which a high-fidelity human geometry can be sampled through a denoising process.  
(2) Train a second flow model on the latent feature space to support generative tasks.  

The method is evaluated on two tasks: pose-conditioned random avatar generation (THuman2 dataset) and avatar-consistent novel pose synthesis (4DDress dataset). Quantitative results show a 57% improvement in geometry quality and a 7% improvement in visual appearance compared to SOTA.

### Strengths
1. The technical designs in the proposed geometry distribution are solid:  
(1) Replacing Gaussian with SMPL (a human-specific template) reduces the "distance" between source and target distributions, minimizing the need to learn extraneous flow paths. This, combined with training pair construction (matching target points to nearest SMPL points + Gaussian perturbation) and distribution normalization (zero-centered Gaussian for source, dense displacement fields for target), solves spatial imbalance in training and accelerates convergence.  
(2) Unlike prior geometry distributions that store shape information in network weights, encoding into 2D feature maps enables efficient dataset-level learning. Plus, the idea of using the decompressed feature map to serve as a condition to generate the final geometry distribution makes sense to me.

2. The quantitative and qualitative experiments are comprehensive, showcasing the effectiveness of the proposed method as well as each design component.

### Weaknesses
Please refer to the question section.

### Questions
1. I'm kind of uncertain about the application scenario. This method is built upon you already have the SMPL parameters, right? If SMPL can already provide human geometry information, what's the role of this proposed method? Is it more like a refiner?

2. I'm a little confused about the decoding process. Looks like you're optimizing the flow model $\theta$ and the condition (i.e. the 2D feature map $\textbf{z}_{T|S}$) simultaneously. Will the training process be stable since I recall that normally in diffusion training, the condition should be fixed, right?

3. I'm wondering do you have a mathematical analysis about why recovering target geometry by conditioning on SMPL can be modeled as a 2D feature map? What does each pixel in the feature map mean? Maybe the author should give more introduction or preliminary knowledge on the 2D feature map representation of 3D objects like SMPL.

4. I'm curious about the influence of feature map size. The paper fixes the feature map to 8×24×24. Are 576 embeddings too few to encode 50,000 points?

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
This paper has proposed a novel 3D human generative model built on top of Geometric Distribution. Specifically, it learns a conditional flow model from the SMPL parameter to the loose cloth real geometric distribution. Some training tricks (pair sampling, distribution normalization, and auto-decoder) are proposed to improve the performance. Extensive experiments have demonstrated the effectiveness of the proposed method.

### Strengths
1. The method is novel, and the problem is worth exploring.
2. The proposed training tricks are reasonable and effective.
3. The final quality is surprisingly good.

### Weaknesses
1. Since some baselines (Like EVA3D) also model the texture distribution, directly comparing the geometry with it is somewhat unfair. But this is not a big issue.
2. Some of the proposed training tricks are actually already well established in the previous works. E.g., the SMPL -> final distribution idea is similar to the diffusion bridge, and the DistNorm is very popular in the current diffusion model training (vae rescaling). 
3. The proposed method has not demonstrated the performance on the animation / temporal consistency.

### Questions
1. If autodecoder is used, when new data comes, does it mean that this proposed method cannot handle this case? Or it requires some test-time optimization, which is unscalable.
2. The UV-space diffusion + autodecoder pipeline for 3d avatar generation is very relevant to Gaussian3Diff (ECCV 24), and should be discussed in the related work.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Building upon the method Geometry Distribution, this paper introduces a human generation method via a flow-based diffusion model. To enable efficient training over the existing human datasets, the authors first propose to encode the human geometry distribution into a feature map. Leveraging the pre-trained feature map, the authors further adopt the SMPL model distribution as the learning base while refining the flow velocity field for more efficient training. The paper also introduce an efficient strategy for constructing training pairs, further improving the generated quality. Extensive experiments have been conducted to demonstrate the performance of the proposed method across various tasks.

### Strengths
The strengths of the method can be summarized as:

+ The adoption of geometry distributions and the introduction of feature map "representation" is novel and demonstrated to be useful.

+ Both the qualitative and quantitative evaluations are thorough and demonstrate the performance of the proposed method, establishing a new state-of-the-art.

+ The paper is well-written and easy to follow.

### Weaknesses
The reviewer would like to point out several weaknesses of the paper:

- The method may still inherit some limitations from the SMPL template, such as the modeling of fine details like hair, accessories, hats, etc.

- Based on the first weakness, the reviewer would like to see the rotating results of loose clothing, complex poses.

- Considering the method is trained on THUman2 dataset, it might be beneficial to compare with the 3D reconstruction methods (like PIFu, ECON, ICON) in terms of both qualitative and quantitative quality to further demonstrate the method or present the gap. Additionally, more metrics like P2S and chamfer distance may be useful to help demonstrate the 3D quality.

### Questions
1. In Figure 7, we could still observe some artifacts in "Ours", especially for the frontal view. However, in Figure 6, it's presented that the proposed strategy for constructing training pairs could address (or actually minimize) this issue. THe reviewer would like to ask the potential and actual factors that lead to this problem, and also the possible solutions.

2. Considering the method employs the SMPL model as the base, will that be possible to generate different shapes (tall, short, fat, thin) or even genders for a human subject?

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
5

### Summary
This paper proposes to learn the generative human geometry distribution from 3D human dataset, which can further be applied for random human generation and animation. At the core of the method is a diffusion model which is conditioned on the UV space of SMPL model to predict the geometry distribution. The authors conducted experiments on different dataset to verify the effectiveness of the proposed method.

### Strengths
The paper is easy to follow.

Generating 3D humans is an interesting task with practical applications.

The method is technically sound by leveraging a UV-conditioned diffusion model to learn the geometry distribution of humans.

### Weaknesses
Insufficient experiments. The paper only evaluates the method on studio 3D data, and does not illustrate whether the proposed method can generate out-of-domain 3D humans. The paper does not answer the question of whether the method overfits to the training dataset, and whether it can sample humans with novel clothing styles. Only random sampling and novel pose generation are shown. 

In addition, in Fig. 4, the authors claim that the pipeline supports multiple condition,s such as images/texts, which, however, are not illustrated in the paper. Can the method generate 3D humans given single in-the-wild images, or generate text-conditioned humans like joint2human?


Existing methods can learn 3D human generations from 2D data such as images and videos, including GAN-based (e.g., EVA) and diffusion-based method (e.g., Primitive Diffusion [Zhao et al. NeurIPS, StructLDM), or generate both geometry and textures using 3D Gaussians in a feed-forward way, such as LHM, AniGS. Compared with these work, what are the advantages of the proposed method which require 3D ground truth data in training and can only generate 3D geometry?

This paper (video demo) claims that it is the first method that learn the human geometry distribution of a dataset. However, this statement is not true. Previous 3D human generation methods such as GDNA [Xu et al 2022] for geometry reconstruction, or EVA/PrimDiff for both geometry and texture reconstruction all learn the distribution of human geometry.

### Questions
The method utilizes SMPL model. How does this handle loose clothing?

### Soundness
2

### Presentation
2

### Contribution
2
