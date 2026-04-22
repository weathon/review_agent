# NoiseSDF2NoiseSDF: Learning Clean Neural Fields from Noisy Supervision

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 6, 2

## Abstract
Reconstructing accurate implicit surface representations from point clouds remains a challenging task, particularly when data is captured using low-quality scanning devices. These point clouds often contain substantial noise, leading to inaccurate surface reconstructions. Inspired by the Noise2Noise paradigm for 2D images, we introduce NoiseSDF2NoiseSDF, a novel method designed to extend this concept to 3D neural fields. Our approach enables learning clean neural SDFs from noisy point clouds through noisy supervision by minimizing the MSE loss between noisy SDF representations, allowing the network to implicitly denoise and refine surface estimations. We evaluate the effectiveness of NoiseSDF2NoiseSDF on benchmarks, including the ShapeNet, ABC, Famous, and Real datasets. Experimental results demonstrate that our framework significantly improves surface reconstruction quality from noisy inputs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a diffusion-based training framework for 3D shape representations using mesh (SDF) supervision, inspired by the Noise2Noise paradigm.
By extending the idea of pixel-to-pixel correspondences in the 2D image domain to coordinate-to-coordinate correspondences in 3D, the authors propose to learn the probabilistic distribution of SDF values derived from noisy point clouds, showing a reasonable improvement over the baseline model.

### Strengths
- Proposes a simple approach to train a point-to-SDF model under the Noise2Noise learning paradigm.

- Provides solid experimental results and comparisons against established baselines.

### Weaknesses
- The contribution appears rather incremental. The proposed framework effectively combines 3DShape2VecSet with the Noise2Noise training scheme, without introducing substantial architectural or theoretical novelty.

- In volumetric 3D representations, spatial correspondences are typically defined through voxel, which directly parallels pixel correspondences in 2D. In contrast, SDF values encode continuous geometric distances, not discrete presence. It is unclear why SDFs are selected in the paper.

- While the method works empirically, the theoretical justification for using SDF-based correspondences remains weak. The training process implicitly assumes that coordinate-level SDF fields under noise satisfy the unbiased expectation property (as in Noise2Noise), but this assumption is non-trivial given the nonlinearity and spatial correlation of SDF fields.

### Questions
- The injected noise level appears quite small. Does this mean that the perturbation in the 3DShape2VecSet latent space is minimal, and thus the diffusion process only acts as a minor regularization?

- The Noise2Noise framework relies on i.i.d., zero-mean noise assumptions, ensuring that averaging multiple noisy observations converges to the clean signal as introduced in section 3 of this paper. However, SDF values are nonlinear and spatially correlated in 3D space, which breaks this assumption. While the authors attempt to mitigate this issue by injecting noise at the point-cloud level before constructing SDFs, this process can still introduce a nonzero expectation bias. The paper would be much stronger if the authors provided theoretical evidence that the unbiased expectation property, $E\[\hat{s}(q|p_i)] = s(q)$ in $\forall i \in {1, 2, ..., n}$, approximately holds, or that such bias is negligible, under their noise-injection scheme; otherwise, the method appears to simply demonstrate that applying Noise2Noise to SDFs works empirically, without offering clear theoretical justification.

Since these concerns (particularly the justification for using SDFs instead of voxel and the validity of Noise2Noise assumptions under geometric noise) lies at the core of the proposed method, my overall evaluation would strongly depend on how it is addressed.

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
This paper proposes NoiseSDF2NoiseSDF, a novel self-supervised learning framework that extends the Noise2Noise paradigm to 3D neural fields for surface reconstruction from noisy point clouds. The method trains a neural SDF network to predict clean signed distance functions by minimizing the MSE between pairs of noisy SDFs generated from independently corrupted point clouds of the same shape. The paper is clearly written, well-motivated, and presents extensive evaluations validating the idea that clean neural fields can be learned from noisy supervision.

### Strengths
The core idea—applying Noise2Noise principles to neural SDF learning—is elegant, conceptually novel, and supported by solid experimental results.

The method is practical: it does not require clean supervision, generalizes across datasets, and achieves high-quality reconstructions even under heavy noise.

### Weaknesses
The experimental comparison, while broad, still lacks evaluation against the most recent state-of-the-art neural reconstruction approaches, such as Neural-Singular-Hessian, which would help establish a clearer performance frontier.

Traditional geometry-based methods such as iPSR, PGR, or Poisson Reconstruction should be compared more explicitly to show how the proposed method performs against classical baselines under varying noise levels.

The robustness analysis could be expanded: it would be valuable to include ablation studies on different input point cloud properties, such as varying point density, non-uniform distributions, and missing points.

### Questions
Would it be feasible to train NoiseSDF2NoiseSDF jointly across multiple noisy-to-noisy mappings (beyond pairwise), similar to ensemble self-consistency methods, to further stabilize learning?

Is it possible to compare with some point cloud denoising methods? And reconstruct it by other methods after denoising? (Just a suggestion)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces NoiseSDF2NoiseSDF, a method that extends the Noise2Noise (N2N) paradigm from 2D image denoising to 3D neural fields. The key idea is to learn a clean neural Signed Distance Function (SDF) from noisy point cloud data by using noisy supervision, minimizing the mean squared error (MSE) loss between them. The authors validate their approach on several benchmarks (ShapeNet, ABC, Famous, Real), demonstrating improvements in surface reconstruction quality over the baseline (3DS2V) and competitive performance against state-of-the-art methods.

### Strengths
1. The paper presents a original idea. While N2N is well-established in 2D, its direct application to 3D is difficult due to the unstructured nature of point clouds. The key insight—that neural SDFs provide a structured, continuous representation analogous to pixel grids, thereby enabling the use of simple MSE loss—is novel and impactful.
2. The paper is generally well-written, with a clear pipeline description and visualizations. Figures 1 and 2 effectively illustrate the core concept and framework.
3. Experiments cover multiple datasets and noise levels, and comparisons include relevant baselines and state-of-the-art methods. The ablation studies reinforce the validity of the core claims. The qualitative and quantitative results show the effectiveness of the method, which can generate smoother and more accurate surfaces.

### Weaknesses
1. A potential limitation is the method's reliance on a pre-trained Point2SDF network (e.g., 3DS2V) to generate the noisy targets. The paper shows generalizability by swapping in 3DILG, but an analysis of how sensitive the performance is to the quality of this frozen network would be valuable. What happens if the Point2SDF provider is poorly trained and less accurate, trained on a very different domain or trained with noisy data to improve noise roubustness?
2. The method is evaluated primarily on additive Gaussian noise and a few other symmetric, zero-mean noise types. Real-world sensor noise can be more complex, potentially containing non-zero mean biases, outliers, structured artifacts, or incompletion due to obstruction. How does the method perform with real-world noise (e.g., from LiDAR or RGB-D sensors)?
3. Suggest adding a comparison between end-to-end SDF denoising and separate sequential denoising steps. For example, first use point cloud denoising methods (such as TotalDenoising) to denoise the input noisy point cloud, and then feed it into a standard Point2SDF network.

### Questions
Refer to Weaknesses.

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
3

### Summary
This paper proposes an algorithm for learning Signed Distance Function (SDF) from the noisy pointclouds. The main observation or hypothesis is that the mean of the multiple observations having noise may converge into the clean observation. Following this assumption, the method leverages the neural networks and minimize the consistency loss between the SDF predictions from the two noisy observations.

### Strengths
The paper is well written and understandable. The authors smoothly transits the context from the theoretical background from Noise2Noise to the application into the neural fields estimation. 

Regarding the hypothesis, '(line 164) ... whether clean neural fields can be effectively learned by observing their noisy counterparts.', it looks reasonable and I personally agree with such an observation. This paper fully focus on proving this assumption and the experiments are well constructed.

Nonetheless, I personally think that the problem definition itself is not novel.

### Weaknesses
__W1. Weak problem setup__  
In line 175, this paper is designed to _investigate_ whether Noise2Noise concept is also applicable to learning clean neural fields. So, in the equation 5, the loss is to reduce the predicted SDF values from the two noisy observations.

Unlike previous studies that learn SDF from the noisy data, this paper directly and explicitly compute the SDF values from the two noisy observation pairs. I admit that this paper _can_ be the first paper to prove the hypothesis, but it is somehow not concise in my opinion.

Commonly, given noisy points from one observation, the previous work tries to minimize the loss to train the network parameters and learn SDF. Across the multiple training batches, the SDF predictions are obtained from the optimized network parameters and finally converge into the _relatively_ clean surface from the noisy inputs. The difference between such an approach and this submission is that whether to `explicitly` compute the noisy input pairs or not. While the authors propose to compute the loss by explicitly comparing the noisy pairs, but the previous studies `indirectly` do so through different training batches at different training step. The network parameters are shared and optimized to minimize the loss. 

Based on this observation, I think that the previous studies _assume_ that Noise2Noisy concept is presumably applicable to the neural field representation, even without the explicit comparison with the two noisy observations at the same time, as suggested by the authors.

So, I am not sure whether the 'explicit' comparison with the multiple noisy observations is needed. Previous studies are dealing with more difficult cases without having such an 'explicit' two noisy pairs. 

I hope that the authors resolve such a concern.

__W2 real-world experiments__
My personal preference is that the authors should have tested the hypothesis in the real-world scenarios. For example, the LiDAR points can be posed in the similar problem definition. Depending on the timestamp, the same 3D scenes are differently measured and recorded in each scan. Such measurements are similar to the authors setup that uses multiple and explicit noisy pairs.

### Questions
Overall paper is highly well written. I enjoyed reading this paper. However, the problem definition that this paper sets is not something new as stated in __W1__. 

Learning clean surfaces from multiple noisy observations are already dealt with the previous studies. The clear difference is whether to have 'explicit' two noisy observations from the same 3D data (this submission), or not. As described in __W1__, I believe that this paper dealt with relatively simple and easy problem compared to recent studies that are trained to predict clean SDF from __single__ and noisy pointclouds without having __multiple__ noisy pairs.

I hope to see the authors' responses in the rebuttals.

### Soundness
3

### Presentation
3

### Contribution
1
