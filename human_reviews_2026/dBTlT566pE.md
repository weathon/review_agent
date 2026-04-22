# Align Your Tangent: Training Better Consistency Models via Manifold-Aligned Tangents

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
With diffusion and flow matching models achieving state-of-the-art generating performance, the interest of the community now turned to reducing the inference time without sacrificing sample quality.
Consistency Models (CMs), which are trained to be consistent on diffusion or probability flow ordinary differential equation (PF-ODE) trajectories, enable one or two-step flow or diffusion sampling.
However, CMs typically require prolonged training with large batch sizes to obtain competitive sample quality.
In this paper, we examine the training dynamics of CMs near convergence and discover that CM tangents - CM output update directions - are quite oscillatory, in the sense that they move parallel to the data manifold, not towards the manifold.
To mitigate oscillatory tangents, we propose a new loss function, called the {\em manifold feature distance (MFD)}, which provides manifold-aligned tangents that point toward the data manifold.
Consequently, our method - dubbed Align Your Tangent (AYT) - can accelerate CM training by orders of magnitude and even out-perform the learned perceptual image patch similarity metric (LPIPS).
Furthermore, we find that our loss enables training with extremely small batch sizes without compromising sample quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies consistency models (CMs), a 1-step distillation of diffusion models. It starts by proposing the following hypothesis: the tangent term in CMs' loss has oscillating behaviors (due to the data lying on low-dimensional manifold). The authors verify it with experiments on CIFAR10 and on a synthetic dataset. Based on this observation, the authors propose a feature map which removes oscillating behavior of the tangent and which makes the tangent more aligned to orthogonal components of the manifold.

### Strengths
- Experiments on synthetic dataset in Section 4 are sound and convincing, particularly Figure 4. This is an interesting analysis of the CM's loss behavior. 

- Experimental results show improvements in 1-step generation for CMs when using the proposed method. 

- Experimental results show faster convergence of the generative model, although the compute budget of pre-training the classifiers is not considered.

### Weaknesses
- Experiments on CIFAR10 in Section 4 are less convincing. It is only based on visualizations, which are not easily interpretable. I am uncertain if conclusions can be drawn from these visualizations. 

- Although the training of the generative model is faster, it relies on pre-training classifiers for the feature map construction. Moreover, it seems that it is required to train one classifier for each feature. This makes the overall pipeline complex, and the compute budget probably higher than in the original model.

### Questions
- To construct the several features, could you rely on a classifier with several heads?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Paper identifies a key problem in training Consistency Models (CMs): "tangent oscillation." The authors find that near convergence, the CM's output update directions often move parallel to the manifold, leading to slow convergence. Authors propose a novel, self-supervised loss function called the Manifold Feature Distance (MFD) (by training an auxiliary feature extractor).  AYT outperform the popular LPIPS loss function, and achieve state-of-the-art FID scores on CIFAR10 and ImageNet $64 \times 64$.

### Strengths
- The "oscillatory tangent hypothesis"  is a well-supported insight into a key problem (slow/unstable convergence) in SOTA consistency models.
- Theoretical framework of MFD, i.e. Gradients and level sets, is plausible. 
- Small batch-sized stable training of CMs is a meaningful contribution. 
- Failure modes of LPIPS are clearly discussed and addressed in proposed self-supervised alternative.

### Weaknesses
- The new step of pre-training the auxiliary feature extractor $\phi$ is described as "lightweight", but the appendix states it is trained for 400k iterations. The "lightweight" claim is not sufficiently quantified in terms of actual compute/flops or wall-clock time relative to the main CM training 
- The feature extractor is a VGG16 (which is indeed a common choice for getting manifold), a choice seemingly inherited from LPIPS. The paper does not ablate this choice. How sensitive are the results to this specific architecture? Would a simpler, custom CNN work? Which would bolster the "lightweight" claim? Or does the method rely on the specific inductive biases of VGG?
- All experiments were run on low-resolution images, how does the performance change with increasing resolution? 
- Evaluation focuses on just FID. No precision/recall, coverage/density, or aesthetic/CLIP-based metrics. 
- (Eq. 9-14) concludes that the AYT tangent is a linear combination of the feature gradients, and that $\nabla \phi_i$ are manifold-orthogonal by construction. However, the weighting terms, $d\phi_i/dt$, are highly dynamic, as they depend on the time derivative of the CM's own output $f_\theta(x_t, t)$ as it moves through the feature space. What mathematical proof or analysis guarantees that this weighted sum remains robustly manifold-orthogonal? How does authors make sure that the weights, $d\phi_i/dt$, cannot create a specific combination that gives a vector with a dominant manifold-parallel component, thereby re-introducing the very oscillation you aim to solve?

### Questions
See weakness. 

Looking forward to authors' response on the proof of manifold-orthogonality.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper discusses the tangent oscillation in the training of the consistency model and proposes a method to align the tangent to the data manifold.

### Strengths
The training of the consistency model is an interesting question. The authors made an effort to advocate for a representation space-based loss might be beneficial.

### Weaknesses
1. The oscillatory tangent hypothesis is not very convincing. It is not clear what to be concerned about if we see some components of the tangent are tangent to the manifold. Note that the goal of the diffusion model or consistency model is not just to generate samples on the data manifold, but also to learn the data distribution. So some components of the "tangent" that are tangent to the manifold may just be needed to adjust for the distributional correctness and not necessarily a problem.
2. The notion of manifold feature is the center of the paper. However, it is not clearly validated. For example, does the level set $\phi^{-1}(x) = \alpha$ recovers the data manifold? If that is the case, then it seems to be able to perform gradient descent on the cost $||\phi(x) - \alpha||$ to obtain a data point on the manifold. 
3. I am a bit confused about the experiments setup, as far as I understand, the ECT do use a pertained model and if the paper follows the same setup, then one cannot claim improvement over distillation based models. 
4. The novelty of introducing a representation space-based loss is rather limited. The performance gain is not very significant.

### Questions
1. Can you clarify your training procedure?
2. Can you validate the manifold feature used in practice?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This manuscript focuses on analyzing the training dynamics of consistency models, and proposes to mitigate oscillating updates by introducing a manifold-aware feature space objective that improves training stability. In-depth theoretical analysis is presented, showing that CM training's update direction (called "tangents") are not orthogonal to the data manifold, but contains oscillating parallel components, which impairs convergence. A feature space distance is then proposed to align the tangents with the orthogonal direction, which employs a network trained on corrupted image data. This achieves similar performance to using LPIPS loss but without human labeling.

### Strengths
- The most significant contribution in this paper is showing that LPIPS loss in consistency models can be replaced by a self-supervised network to achieve the comparable performance. This potentially enables applying feature space loss in consistency models outside the image domain, overcoming the scarcity LPIPS-like metrics in other modalities.
- The design of the feature space metric makes is technically sound. Ablations of different types of transformations also reveal that geometric transformation is the most important, potentially indicating that structural awareness is the most important update direction in consistency models. 
- The analysis on training update directions ("tangents") is solid and provides an interesting perspective of understanding consistency training.
- Overall, the quality of this manuscript is high, both in terms of technical details and writing clarity.

### Weaknesses
- The experiments do not show any significant advantage of the proposed method over existing ones. The comparison with LPIPS does not reveal a very clear gap. In Fig. 5, the FID scores are very close. This is a major limit to the significance the contribution. In addition, both LPIPS and AYT are starting to get worse after 250k iterations, so argument that "CM trained with LPIPS showed degradation in two-step FIDs after 200k iterations." could also apply to AYT similarly.
- The statement that "We discover a potential cause of slow convergence in CM training" is a bit over-claiming. While this manuscript provides an interesting perspective based-on data manifold, the slow convergence in CT is well-understood: CM's target x0 is propagated from the boundary condition at t=0 to intermediate times, causing accumulation of errors, which naturally includes noisy components that impairs training stability.

### Questions
- Overall this work is very interesting, but I would really like to see more experiments beyond pixel-space consistency models (e.g., latent consistency training on ImageNet 256), where LPIPS is unavailable and AYT could potentially demonstrate its real strength.
- In Fig. 3, how are the M parallel and M orth components computed? If you can decompose the two directions without AYT, why not just suppress the parallel component directly?

### Soundness
3

### Presentation
3

### Contribution
3
