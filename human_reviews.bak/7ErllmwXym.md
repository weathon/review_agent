# Interpreting and improving diffusion models using the Euclidean distance function

- Decision: Reject
- Scores: 3, 6, 8

## Abstract
Denoising is intuitively related to projection. Indeed, under the manifold hypothesis, adding random noise is approximately equivalent to orthogonal perturbation. Hence, learning to denoise is approximately learning to project. In this paper, we use this observation to reinterpret denoising diffusion models as approximate gradient descent applied to the Euclidean distance function. We then provide straight-forward convergence analysis of the DDIM sampler under simple assumptions on the projection-error of the denoiser. Finally, we propose a new sampler based on two simple modifications to DDIM using insights from our theoretical results. In as few as 5-10 function evaluations, our sampler achieves state-of-the-art FID scores on pretrained CIFAR-10 and CelebA models and can generate high quality samples on latent diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper is focused on presenting an interpretation of diffusion models, in which each iteration interpreted as a non-linear projection onto the data manifold. They define a Euclidian distance function from noisy image to the manifold, and interpret the denoising step as the gradient descent on this distance function. On the empirical side, the paper proposes a modified update line based on the theory, with the goal of minimizing and correcting the score prediction error. When number of iterations are limited to small numbers, this method achieved better performance in FID compared to other methods, on CIFAR10 and celebA.

### Strengths
The writing is clear and the math is reasonably easy to follow. 
The derivations are generally sound and well written.

### Weaknesses
The main weakness of this work is that theorem 3.1 rests on a strong assumption that is violated in the case of image manifolds. Up to this theorem, the math is not novel and is borrowed from other sources. The contribution of this work is in Theorem 3.1, where the existing math about non-linear projection is used to interpret backward diffusion/denoising. This theorem is the foundation for the rest of the paper, and the derivations about the bounds sit on top of this theorem. The problem is that the theorem rests on a strong assumption on the reach of the image manifold: $reach(K) \geq \sigma \sqrt{n}$ which does not necessarily hold. This assumes the curvature of the manifold is small compared to $\sigma$. This greatly simplifies the geometry of the assumed manifold but the problem is such manifold is too simplistic to approximate *image* manifold. If images lie on manifold, those manifolds must be complex with high curvatures.  

To make it clear as to why image manifolds must contain high curvature (or small reach) it is enough to run a simple mental experiment. Consider an image, which represents a single point on the manifold. All the invariances of that image, such as local translations, also exist within this manifold. The geometry of local translations is well understood in the Fourier domain. When an image is translated, its Fourier amplitude remains constant, while its Fourier phases vary. Consequently, translated versions of an image lie on circles, with the circle's radius equivalent to the Fourier amplitude. It's a widely known empirical observation that the power spectrum of images adheres to a $1/f$ law, where $f$ represents the frequency, stating that higher frequencies exhibit smaller amplitudes. Therefore, the circles on the manifold, due to the translation invariance of image details (higher frequencies), have small radii. The $1/f$ phenomenon is an empirical but firmly established characteristic of images.

The reach(K) of image manifold is not a constant and depends on the image (i.e. the curvature varies across the manifold). But as long as the image is not blank and contains details, the $reach(K)$ will be small. For a typical image, the reach can be as small as the quantization precision of the images in the dataset, due to the $1/f$ property: the highest frequencies create small circles around a given image.

The analysis in the paper does not hold unless for very small sigma (in the vicinity of the manifold), where we can assume the sigma is smaller than the radius of the curvature. But the paper claims this analysis can be used for denoising in diffusion models which always start with very large noise, and the noise stays large throughout the intermediate steps until towards the very end of the trajectory.

Moreover, the claim that the gradient doesn't change does not hold if the assumption on reach is violated. In other words, when the curvature is larger than sigma, the gradient does change direction at each step (up to a point where sigma is small enough where the manifold is locally flat relative to sigma level).

### Questions
Please see weaknesses section for comments.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors reinterpret denoising diffusion models as approximate gradient descent applied to the Euclidean distance function under manifold hypothesis. In the interpretation, they establish rigorous connection between denoising function and the projection to data manifold, which is equivalent to the gradient of squared distance function to the manifold. They also reframe the convergence analysis of diffusion models using the new interpretation, providing a justification for the well-known log-linear noise schedule used in diffusion models.

Building on these insights, the authors introduce a higher-order sampler of diffusion models. This sampler leverages the invariant properties of projection (projection is invariant along line segments between a point and the its projection), regularizing that the direction at one point in one step is similar to the direction at the point of next step. Their new sampler does not require additional evaluation of denoising (score) function and achieves SOTA FID scores on pretrained CIFAR-10 and CelebA models.

### Strengths
This paper offers a novel interpretation of DDIM samplers, framing them as approximate gradient descent applied to the squared distance from data manifold. The convergence analysis of DDIM with Euclidean distance function instead of divergence over probability space adds to its novelty.

The authors propose improved schedules and samplers for DDIM based on their theoretical findings, which enhances the practical applicability of proposed theory. This improvement upon previous models based on theoretical foundations is notable.

### Weaknesses
The proposed schedules and samplers are tested on a limited number of small-scale datasets

Implications of lemma/theorem are unclear
- Thm4.2: In rea settings, is final sample error (dist_K(x_0)) bounded?
- Thm4.3: Why does beta star suddenly appear?
- Thm4.4: Can we adopt beta for achieving convergence to zero? And how does it related to the schedule showcased in Section 6
- Section5: What are the meaning of a positive definite metric and the implications of the metric with gamma equal to 2?

### Questions
It would be beneficial for the authors to relate their work to existing research regarding to geometry of diffusion models, such as "Score-Based Generative Models Detect Manifolds," "A Geometric Perspective on Diffusion Models," and "Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors offer a novel perspective by interpreting denoising diffusion models as an approximation of gradient descent applied to the Euclidean distance function. Furthermore, the paper includes a comprehensive convergence analysis of DDIM. The experimental results underscore the effectiveness of the proposed sampler, as it attains state-of-the-art FID scores and consistently generates high-quality samples.

### Strengths
This paper presents a compelling blend of theoretical interpretation and practical enhancements. While the projection interpretation of diffusion models is not entirely novel, the authors provide a more concrete example by framing sampling as an approximation of gradient descent on the distance function to the training dataset.

### Weaknesses
.

### Questions
Q1.The results presented in Table 2 appear to be incomplete, with certain experiments seemingly omitted. It would be beneficial to provide a more comprehensive set of results to ensure a thorough evaluation.

Q2. Additionally, the comparison with DPM suggests that DPM performs comparably to the proposed sampler. It might be helpful to provide further insights or analysis regarding the similarities and differences between the two methods to clarify their relative strengths and weaknesses

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
