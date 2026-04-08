## Human Reviewer 1

### Summary
This paper proposes Variational Bayes Gaussian Splatting (VBGS) for modeling 3D scenes by framing training of Gaussian Splats as variational inference. The formulation enables closed-form variational update of model parameters which is particularly beneficial in online settings. In particular, updating the model from a sequence of observations is equivalent to processing all data in a single batch. Thus, catastrophic forgetting is circumvented.

Experimental results indicate VBGS matches the performance of 3D Gaussian Splatting (3DGS) on static datasets while enabling continual learning with streamed 3D data.

### Strengths
**S1.** The method addresses an important limitation of 3DGS.

**S2.** The method is principled and presented with clarity.

**S3.** The experimental validation lends support to the practical usefulness of the method: in particular, in the online setting 3DGS is clearly impacted by catastrophic forgetting while VBGS is not.

### Weaknesses
**W1.** The method is motivated by the application to continual learning scenarios such as SLAM. However, the experimental validation does not compare against recent SLAM methods or datasets, e.g., (Matsuki et al., 2024; Keetha et al., 2024).

**W2.** The significance of the mean-field approximation is not discussed or investigated experimentally.

**W3.** The method's reliance on depth maps is, as acknowledged by the authors, a limitation of the method. The author’s propose inferring depth maps from RGB data using a pretrained model. Experiments along these lines would have been interesting to include.

### Questions
**Q1:** Why not evaluate using datasets and protocols from recent SLAM works such as (Matsuki et al., 2024; Keetha et al., 2024)?

**Q2.** When should one expect the mean-field approximation to hurt performance?

### Soundness
3

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
The paper proposes a reformulation of 3D Gaussian splatting with variational inference framework. This allows a closed-form update rule and applicability to continual learning framework. The model is compared with gradient-based update rule in small-scale (~100k) 3DGS fitting to Habitat test suite and NeRF synthetic dataset and shows performance boost with better robustness to randomized initialization.

### Strengths
The main strengths I find in this paper lie in its novelty, clarity in writing, and reproducibility.

- Both relating 3D Gaussian splatting as Gaussian mixture model and variational inference is new to this field, and this novelty should be acknowledged.
- Application of continual learning well justifies the usage of variational inference framework.
- The training scheme is described in detail and the described motivation and methodology is quite clear.
- The supplementary material has the code implementation for reproducibility.

### Weaknesses
Although I find the method informative, there are several points to be clarified in order to improve the presentation.

- Common 3DGS involves several millions of Gaussians, but the presented experiments are only based on 100k Gaussians, which is very slim. This small parameter constraint makes the PSNR scores in Table 1 way below the numbers reported by other NeRF/3DGS papers which go high above 30 dB for the same NeRF synthetic dataset. Is there any specific reason (e.g., computational burden) that constraints the experiments? If so, this should be specified.
- Unlike the authors’ claim in Appendix D.3, Table 6 shows that the presented VBGS shows comparable or worse results than Gradient-based method, which needs further clarification.
- It seems like the method does not incorporate cloning/pruning/splitting of Gaussians which take crucial roles in the original 3DGS. By scaling up the Gaussians during the training, the performance can be boosted significantly, so unless there is a counterpart in VBGS for resizing the Gaussian pool, I believe there is a little advantage of using this framework in practical scenarios. I believe there should be an experiment with VBGS having initial Gaussians of the same size of the final Gaussians in Table 5, so we can compare fairly. For example, if we start from 456K Gaussians in VBGS and performs better than dynamically resized 3DGS in Table 5, we can clearly say that there is a reason to migrate from 3DGS to VBGS.
- I believe it is better to have a dedicated discussion on changes in computational demand relative to the size of Gaussians in different frameworks. Inference time should be the same, but training time and the memory requirement will be very informative to followers to this work.

In summary, unless there is a clear advantage of VBGS over standard size-varying gradient-based 3DGS, the method is not practically persuasive, yet. Furthermore, currently, there is no guarantee of scalability, which I believe important in learnable 3D representation. It is OK to have a slow and memory-intensive method if the performance (PSNR/SSIM) is stronger, but regarding Table 6, the presented method is not yet persuasive. Regarding these, the current version of the manuscript seems not ready to be published despite its novel and intuitive idea. Please note that I am flexible with my score, and looking forward to further discussion in the rebuttal phase.

### Questions
These are minor concerns that are not counted in my final scores.

- Having only three components in Figure 2.a without Gradient (Data init) seems incomplete. Please have it updated.
- As the authors have demonstrated with a continual learning framework, is it workable to scene in motion?
- Does this method scale up? Can this method scale up to tens of millions of Gaussian splatting for larger datasets (e.g., MipNeRF-360)?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
5

### Confidence
5

---

## Human Reviewer 3

### Summary
This paper introduces Variational Bayes Gaussian Splatting (VBGS), a new method for modeling 3D scenes with Gaussian splats that addresses the challenge of catastrophic forgetting in continuous data streams. Unlike traditional methods that rely on backpropagation through a differentiable renderer and require replay buffers, VBGS formulates training as variational inference over model parameters, enabling efficient updates through a closed-form variational update rule.

### Strengths
1. Effectively handles continuous data streams: By framing the training process as variational inference over model parameters, VBGS avoids the issue of catastrophic forgetting typically seen in continuous data streams with traditional methods.

2. No need for replay buffers: VBGS leverages the conjugacy properties of multivariate Gaussians, deriving a closed-form variational update rule that allows efficient incremental updates without requiring replay buffers.

### Weaknesses
1. Limited improvement. In Tab. 1, your method is comparable to the Gradient method (3DGS), which cannot prove the effectiveness of your method.

2. Unlike 3D Gaussian Splatting (3DGS), it relies on RGBD data. It means your method needs a depth camera and does not outperform 3DGS.

3. The experiments are not comprehensive, as they do not include evaluations on standard datasets like mip-NeRF 360, which could provide a more robust comparison with existing methods.

4. The comparison with baselines is limited; a more thorough evaluation would include comparisons with established methods such as Mip-NeRF 360, Instant-NGP, and NeRF. Including these methods as baselines would provide a clearer understanding of VBGS's performance relative to a broader range of state-of-the-art approaches.

### Questions
As stated in the weaknesses above, I have some additional questions as follows:
1. Why is it said that 3DGS struggles with catastrophic forgetting when dealing with continuous streams of data?

2. Could you provide numerical results on the HABITAT ROOMS dataset?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper proposes Variational Bayes Gaussian Splatting (VBGS), which frames training a Gaussian splat as variational inference over model parameters.

### Strengths
1 The paper is well-structured.
2 The idea of introducing the conjugacy properties of multivariate Gaussians seems to make sense.

### Weaknesses
1 In the introduction, and background (e.g., 3D Gaussian Splatting) and research gap are not detailed enough, so the author's research motivation cannot be understood clearly.
2 The novelty is unclear and insufficient, it's confused why the proposed method can eliminate the need for replay buffers. More relevant discussions and technical details should be analyzed.
3 The authors claim that this work can "more robust and efficient parameter estimation." However, no computational efficiency analysis (e.g., computational time, memory usage) is provided.
4 In the experiments, the baseline only includes "Gradient". More discussions are need to explain why choosing"Gradient". Besides, the superiority of the proposed method over "Gradient" needs more analysis from both theoretical and experimental perspectives. Are there any more recent and advanced baselines?
5 The evaluation metrics should be introduced in detail, e.g., PSNR, to enhance the understandability and readability of the experimental part.

### Questions
Please refer to the weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
3