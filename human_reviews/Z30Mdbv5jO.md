# ReconX: Reconstruct Any Scene from Sparse Views with Video Diffusion Model

- Decision: Reject
- Scores: 6, 5, 5, 5, 8

## Abstract
Advancements in 3D scene reconstruction have transformed 2D images from the real world into 3D models, producing realistic 3D results from hundreds of input photos. Despite great success in dense-view reconstruction scenarios, rendering a detailed scene from sparse views is still an ill-posed optimization problem, often resulting in artifacts and distortions in unseen areas. In this paper, we propose ReconX, a novel 3D scene reconstruction paradigm that reframes the ambiguous reconstruction problem as a temporal generation task. The key insight is to unleash the strong generative prior of large pre-trained video diffusion models for sparse-view reconstruction. Nevertheless, it is challenging to preserve 3D view consistency when directly generating video frames from pre-trained models. To address this issue, given limited input views, the proposed ReconX first constructs a global point cloud and encodes it into a contextual space as the 3D structure condition. Guided by the condition, the video diffusion model then synthesizes video frames that are detail-preserved and exhibit a high degree of 3D consistency, ensuring the coherence of the scene from various perspectives. Finally, we recover the 3D scene from the generated video through a confidence-aware 3D Gaussian Splatting optimization scheme. Extensive experiments on various real-world datasets show the superiority of ReconX over state-of-the-art methods in terms of quality and generalizability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper reframes the ambiguous sparse-view reconstruction task as a temporal generation task and unleashes the strong generative prior of large pre-trained video diffusion models. To maintain consistency while generating the video clip, this work proposes to first construct a global point cloud using DUSt3R and then encode it into a contextual space to provide the 3D structure condition. Afterward, it will set the first frame as well as the last frame and repurpose the video diffusion model as an interpolator to synthesize plausible frames. Finally, it will incorporate a confidence-aware 3DGS optimization for reconstruction. Experiments show the paper's superiority in terms of quality and generalization capability to larger viewpoint change compared with previous work.

### Strengths
1. The idea of repurposing a video diffusion model as a spatial interpolator sounds reasonable and can provide generative priors to unseen regions.
2. How this work injects structure-aware guidance by using the point cloud from DUSt3R seems critical and can provide insights for future work.
3. This work achieves visually better results than previous methods and seems to generalize better to large viewpoint changes.
4. The paper is well-written and should be easy to follow as long as the code is released.

### Weaknesses
1. One strength of directly applying a feed-forward reconstruction like pixelSplat[1] or MVSplat[2] is the inference speed. The proposed method needs to incorporate DUSt3R, a video diffusion model, and a confidence-aware optimization to get the final result. I wonder about the total time consumption during inference compared with baseline models.
2. From the reviewer's perspective, there seem to be multiple solutions given the first frame and the last frame of a video, would it be better to incorporate camera pose information during the training and inference of the video diffusion model? Or on the other side, is the video diffusion model trying to learn a certain type of camera motion due to the limitation of fine-tuning dataset? 
3. This paper proposes to utilize a global point cloud as the 3D structure guidance. From the ablation study in Fig.5, the quality seems to be poor without this guidance, showing the importance of incorporating this point cloud. While DUSt3R itself can provide a global point cloud as the reconstruction result, I would like to see a comparison with DUSt3R apart from MVSplat and pixelSplat. Moreover, pixelSplat and MVSplat do not incorporate a global point cloud as input. 
4. This work mainly deals with static scene reconstruction, it would be interesting to reconstruct a dynamic scene in future work, which is also mentioned by the authors in the future work part. 

[1] pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction, CVPR 2024 

[2] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images, ECCV 2024

[3] DUSt3R: Geometric 3D Vision Made Easy, CVPR 2024

### Questions
Please refer to the weakness part. I will be glad to raise my rate if my concerns can be addressed!

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents ReconX, which is aim to convert the sparse view 3D reconstruction to the temporal consistency generation by leveraging the video diffusion model as the prior. Moreover, the author uses the 3D structure guidance besides the common multi view images as the guidance, along with the mathematical prove that such new condition can improve the performance. Then, the confidence-aware 3DGS optimization strategy further ensure the performance to have the final novel view synthesis. Comprehensive experiments have been done to demonstrate the in and cross-domain ability of the model.

### Strengths
1. Using temporal consistency generation to solving the multi view consistency problem in sparse view reconstruction 
2. Including the point cloud as an additional condition for the video generation model and achieving great performance is novel and effective.
3. Applying confidence-aware strategy to optimize the 3DGS to improve the artifacts introduced by the diffusion model.
4. Comprehensive experiments have been done, including different number of views, in and cross-domain, to demonstrate great performance of this work.

### Weaknesses
1. More outdoor unbounded dataset can be included such as MipNeRF 360 in the experiment.
2. The performance relies heavily on the point cloud condition. But in the far camera distance case, which is common in MipNeRF 360, the point cloud may not be achieved from the spare input views or the point cloud may not include enough high-quality information, the performance of the diffusion model cannot be preserved. 
3. Even though the confidence-aware can boost the performance of 3DGS optimization, but the reason why transformer decoder from  DUSt3R can provide the confidence is not fully investigated while it only draws from empirical study.

### Questions
Please refer to the Weakness.

### Soundness
2

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
This paper proposes a framework for obtaining a 3DGS representation of a scene from sparse views, intended for novel view synthesis. It first uses Dust3r to recover a point cloud representation of the scene, which is only a rough estimate of the 3D structure due to the sparsity of views. The point cloud is converted into a learned 3D embedding through an FFN, for injecting as a condition into a video diffusion model to generate intermediate novel views that are consistent with the 3D embedding. Next, each generated view appears to have its pixel values modified by some alignment function, and also assigned a confidence map. The final 3DGS representation is then obtained by standard 3DGS optimization except weighted by the confidence values. Although the proposed method involves per-scene 3DGS optimization, the comparisons are primarily made to feed-forward methods (e.g. MuRF, pixleSplat, MVSplat), and unsurprisingly outperforms them.

### Strengths
There does not appear to be previous work combining ideas from Dust3r, video diffusion and 3DGS, so this work is novel in that sense, although there are concurrent ones (ViewCrafter [1], LM-Gaussian [2], 3DGS-Enhancer [3], MVSplat360 [4]) that explore similar ideas.

The proposed methodology appears to be sound, and mostly well-justified. Although this perhaps explains convergent ideas as highlighted by the concurrent works.

Most of the paper is mostly well-written and diagrams are clear, except for a few parts highlighted in the Weaknesses section.

References
- [1] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, Yonghong Tian, ViewCrafter: Taming video diffusion models for high-fidelity novel view synthesis, arXiv preprint arXiv:2409.02048, 3 Sep 2024.
- [2] Hanyang Yu, Xiaoxiao Long, Ping Tan, LM-Gaussian: Boost sparse-view 3D Gaussian splatting with large model priors, arXiv preprint arXiv:2409.03456v1, 5 Sep 2024.
- [3] Xi Liu, Chaoyi Zhou, Siyu Huang, 3DGS-Enhancer: Enhancing unbounded 3D Gaussian splatting with view-consistent 2D diffusion priors, NeurIPS 2024. Also as arXiv preprint arXiv:2410.16266, 21 Oct 2024.
- [4] Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan Zhuang, Andrea Vedaldi, Tat-Jen Cham, Jianfei Cai, MVSplat360: Benchmarking 360° generalizable 3D novel view synthesis from sparse views, NeurIPS 2024.

### Weaknesses
**Problems with experimental settings**

The primary weakness of this paper lies in the experimental section. The baseline methods chosen seem to be the wrong ones to compare against, as they are all _feed-forward_ methods that produce 3DGS representations (pixelSplat, MVSplat) or radiance fields (MuRF, pixelNeRF, GPNR), done _in a single pass_. In contrast, the proposed method is based on per-scene 3DGS optimization, which unsurprisingly will perform better.

Furthermore, the choice of datasets for testing is limited. For the main results, only RealEstate10k and ACID, with LLFF and DTU only tested on for cross-dataset generalization (which, as an experiment setting itself, is much less impacted in a per-scene optimization method, compared to feed-forward methods). The paper also mentions DL3DV-10K (L373), but it is unclear how and where this is used in the experiments.

Instead, the appropriate baselines, settings and choice of datasets should follow that of per-scene sparse view NVS papers, such as conducted in ReconFusion (Wu 2024b) and Cat3D (Gao 2024), for a fairer comparison.

**Other weaknesses**

A global alignment function is mentioned in L293, but this is neither defined nor is there explanation for its purpose. Based on the brief expression in L293, it appears to do some kind of color adjustment of the video diffusion generated images.

L293-295:
> Through empirical study, we find a well-aligned mapping function $\mathcal{A}$ from the transformer decoder of DUSt3R, which builds the confidence maps ...

This statement is very vague and contradictory. Does $\mathcal{A}$ do color adjustment, or does it produce confidence maps? Is it a handcrafted function? How does it relate to the Dust3r transformer decoder?

The utility of Proposition 1 (L185-189) is unclear. It appears to express that having a 3D condition is less ill-posed than a 2D condition, which is already universally accepted. If so, it would be a distraction from the key ideas of the paper, so I would suggest leaving this out of the paper.

Minor point, but for equation 5 (L256), it would seem that $c_{view}$ and $c_{struc}$ should be deterministically tied to any sampled $x\sim p$. So there is no separate marginalization, and should not appear in the subscript of the expectation $\mathbb{E}$.

Also minor formatting issue, but having Table 4 (L432-436) placed above Table 3 (L468-478) is strange and breaks the flow.

The supplementary video only includes results from the authors' method, but not from the baselines. To better judge qualitative improvements, it would be much better to include video results for all compared methods side-by-side. The few frames in the main paper are not really sufficient, and can easily be cherry-picked.

### Questions
Please address the issues raised in the weaknesses section. Specifically:

- If there is time, please provide experimental comparisons with per-scene sparse input NVS methods as elaborated above, in terms of quantitative results, and if possible side-by-side video results.
  - If there are good reasons why it is not appropriate to compare to these baselines, please explain.
- Define the global alignment function $\mathcal{A}$ and explain its role.
- Clarify the vague statement in L293-295, regarding confidence maps for generated images.
- Clarify how the DL3DV-10K dataset is used in the experiments.

REVISED review: Given the major effort the authors have put into the response and revision, I have raised my rating to 5. It is not higher, however, because many of the issues were very obvious (e.g. comparison to feed-forward methods only), and could have been avoided at point of submission. As such, there is insufficient time during the rebuttal period for proper due diligence, on both the part of the authors and reviewers.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
ReconX proposes video diffusion priors conditioned with 3D structure guidance for reconstruction and novel view synthesis in a 2-view setting. Specifically, the authors leverage the DUSt3R method to obtain a global dense point cloud and encode it into the latent space of the video diffusion model with point transformers for providing sufficient 3D context and obtaining a 3D-consistent novel view coherent with the input images. Finally, a 3D-confidence-aware optimization scheme with 3DGS robustly reconstructs the scene in a few iterations. Experiments show that ReconX outperforms related 2-view baselines - pixelSplat and MVSplat on both in-distribution and OOD benchmarks.

### Strengths
1. The proposed 3D point cloud conditioning for ensuring the 3D consistency of generated frames is novel and intuitive.
2. The video diffusion architecture is well-ablated, and each of the highlighted contributions impacts the final reconstruction quality positively.
3. Extensive experiments demonstrate ReconX's ability to achieve high-quality reconstructions that outperform related state-of-the-art methods, particularly in challenging scenarios where there is a large angle variance between input views. ReconX also shows robust generalizability across datasets, a notable limitation of baselines pixelSplat and MVSplat.

### Weaknesses
1. Weak Benchmarks: ReconX shows strong sparse-view reconstruction capabilities for scenes from LLFF and DTU datasets, but these are relatively simpler benchmarks, and since the method uses strong video diffusion priors, I would expect a comparison on more challenging benchmarks like MipNeRF360 (or Tanks and Temples). CAT3D / ReconFusion already provides data splits for 3, 6, and 9 view settings for this dataset, so comparisons with a generalized N-view setting would strengthen the submission further.
One can also estimate a DUSt3R point cloud at low cost for N <= 9, so I do not find any bottlenecks in that regard.

2. Weak Baselines: Both baselines - pixelSplat and MVSplat are non-generative in nature. As such, comparison with methods integrating video diffusion priors for sparse-view reconstruction like CAT3D and V3D seems quite relevant.

3. Missing Ablation: The method uses a DUSt3R point cloud initialization for optimizing 3DGS. However, the baseline involving optimization of 3DGS with this dense stereo point cloud is missing from Table 3. Since the authors optimize 3D Gaussians for only 1000 iterations without Adaptive Density Control, my guess is that the DUSt3R point cloud already provides a very strong scene structure. As such, assessing the impact of the video diffusion priors becomes more important. I assume the variant denoted as “base” in Table 3 already integrates the DUSt3R priors with 3DGS.

References:

1. CAT3D: Create Anything in 3D with Multi-View Diffusion Models, Ruiqi Gao* and Aleksander Holynski* and Philipp Henzler and Arthur Brussee and Ricardo Martin-Brualla and Pratul P. Srinivasan and Jonathan T. Barron and Ben Poole*, 2024
2. V3D: Video Diffusion Models are Effective 3D Generators, Zilong Chen and Yikai Wang and Feng Wang and Zhengyi Wang and Huaping Liu, 2024

### Questions
1. Since DUSt3R is a pose-free stereo estimation pipeline, the estimated point cloud would not be aligned with the ground truth poses of benchmarks like LLFF/DTU. To obtain synthesized novel views, how is the reconstructed Gaussian point cloud aligned with the COLMAP cameras during inference? I did not find any information regarding that.

2. Does the choice of the video diffusion model have a major impact on the final reconstruction? How would Stable Video Diffusion, for example, work in combination with the proposed 3D structure conditioning? Is there any specific reason behind picking DynamiCrafter?

3. How many scenes from the 3 datasets - RealEstate10k, ACID, and DL3DV-10K are used for training the video diffusion model? I do not see a mention of the total count, so I am assuming all?

4. Can you please provide some more details on the transformer-based encoder for embedding the DUSt3R point cloud? Is it one of the pretrained point Transformer architectures (or maybe the point-SAM encoder), and is it also finetuned along with the video diffusion model? From Figure 2, I am guessing not, but such encoders are usually trained with point clouds from Scannet-like benchmarks, and DUSt3R point clouds would most likely be OOD in terms of overall structure and density.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes a method or 3D scene reconstruction from as few as two views via a two-stage approach. First, a finetuned video diffusion model generates multiple view that interpolate the camera trajectory between the given input views. In order to improve 3D consistency of these generated images, the authors introduce a 3D structure conditioning by encoding a point cloud obtained from an existing learned stereo reconstruction method (DUSt3R). In a second stage, the generated views are fused into a 3D representation, for which the authors extend 3D Gaussian Splatting by a weighting based on confidence of DUSt3R.
Experimental evaluation results show strong performance for both in training distribution data as well as cross-dataset generalization, outperforming state-of-the-art baselines. Ablation studies validate the effectiveness of the technical contributions.

### Strengths
- The paper tackles an important and very challenging task of reconstructing general scenes from very sparse views.
- The two-stage approach leverages the strong prior of video diffusion models and the technical contributions fit well into this framework.
- The experimental evaluation shows strong results consistently outperforming state-of-the-art baselines on both
   - in training distribution data 
     - significant quantitative and qualitative advantage for small angle variance in input views (table 1, figure 3)
     - even larger gap for large angle variance in input views (table 2, figure 4)
  - and cross-dataset generalization
    - baselines fail completely, while the proposed method achieves reasonable metric scores (table 4)
    - impressive qualitative results (figure 6)
- The ablation study validates the effectiveness of the proposed 3D structure conditioning and the confidence-aware 3DGS optimization (especially convincing qualitative ablation in figure 5).
- The paper is mostly well-written and easy to follow.
- The supplemental video provides additional convincing qualitative reconstruction results.

### Weaknesses
- The lack of precise mathematical formulation raises doubt about proposition 1 and its proof:
  - The main inequality in equation 17 (appendix) is justified only verbally and is not obvious to me.
  - A counterexample for the inequality in line 841 could be a dataset mainly consisting of dark rooms with all ground truth renderings being black except for one where the lights are on (with possibly very complex geometry or even transparent and reflective materials). For this case, fitting the marginal distribution can be easier than fitting the conditional distribution, if the given scene s happens to be the one with lights on.
  - Moreover, if the 3D structure (point cloud) is obtained from 2D images only via stereo reconstruction (DUSt3R [1]), then an image encoder should be theoretically able to encode the images in the same way as the composition of DUSt3R and the point cloud encoder used in the method. Following this argument, a strict inequality as in proposition 1 is not reasonable.

- Some lack of clarity:
  - What exactly is the implementation of the learnable embedding function PosEmb in equation 3?
  - What is meant with "uncertainty of unconstrained images" (line 265)? Does this mean that generated images are not fully 3D-consistent?
  - Regarding the confidence-aware 3DGS optimization in section 4.4, it is unclear how the loss in equation 6 is still used / relevant for the final loss in equation 7, which does not use equation 6 anymore.

- Somewhat unfair comparison with baselines:
  - The proposed method relies heavily on the strong priors learned by DUSt3R [1] and the video diffusion model.
  - All baselines are trained more or less from scratch on small-scale datasets compared to the large-scale datasets for training video diffusion models.
  - Therefore, it is not that surprising that the proposed method generalizes much better to other datasets out of the training/finetuning data distribution.

- Restriction to view interpolation:
  - Given that the method makes use of a pre-trained video diffusion model, the restriction to interpolation of the camera trajectory between two input views is unsatisfactory.
  - In many cases, two views as conditioning already eliminate uncertainty in the reconstruction task entirely such that deterministic approaches like pixelSplat [4] and MVSplat [5] already produce detailed and sharp reconstructions.
  - The proposed generative approach with a strong prior trained on large-scale single view data should be able to extrapolate from input views to some extent, which the paper misses to evaluate.
  - This is especially critical, if the paper motivates the method with the limitation of generalizable 3D reconstruction methods that "struggle to generate high-quality images in areas not visible from the input perspectives" (lines 143f.).

- Missing relevant related work and possible baseline:
  - latentSplat [2] aims to bridge the gap between regression models for generalizable NVS and generative models for 3D reconstruction and is therefore a very relevant work and potential baseline.
  - GeNVS [3] was one of the first approaches to generative novel views with a diffusion model conditioned on 3D-aware (pixelNeRF) features and is therefore important related work.

Minor comments:
- The paper sometimes uses the terms "conditioning" and "guidance" interchangeably, although they have different meanings to me (conditioning: giving something as additional input to the diffusion model; guidance: using CFG or also gradients (e.g. classifier guidance) to affect the sampling process). A more precise use of these terms would be helpful.
- The ablation study in table 3 could be extended to contain all different combinations of leaving out individual components in order to find possible dependencies between them.

References:
- [1] DUSt3R: Geometric 3D Vision Made Easy. CVPR 2024
- [2] latentSplat: Autoencoding Variational Gaussians for Fast Generalizable 3D Reconstruction. ECCV 2024
- [3] Generative Novel View Synthesis with 3D-Aware Diffusion Models. ICCV 2023
- [4] pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction. CVPR 2024
- [5] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images. ECCV 2024

### Questions
- Is the mentioned counterexample for the proof correct or am I missing something?
- What exactly is the implementation of the learnable embedding function PosEmb in equation 3?
- What is meant with "uncertainty of unconstrained images" (line 265)? Does this mean that generated images are not fully 3D-consistent?
- Regarding the confidence-aware 3DGS optimization in section 4.4, how is the loss in equation 6 still used / relevant for the final loss in equation 7, which does not use equation 6 anymore?
- Is there any specific reason why you limit yourself to the case of view interpolation and do not evaluate extrapolation?

### Soundness
3

### Presentation
3

### Contribution
3
