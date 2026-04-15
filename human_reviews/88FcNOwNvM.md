# Compositional Image Decomposition with Diffusion Models

- Decision: Reject
- Scores: 6, 5, 8

## Abstract
Given an image of a natural scene, we are able to quickly decompose it into a set of components such as objects, lighting, shadows, and foreground. We can then picture how the image would look if we were to recombine certain components with those from other images, for instance producing a scene with a set of objects from our bedroom and animals from a zoo under the lighting conditions of a forest even if we have never seen such a scene in real life before. We present a method to decompose an image into such compositional components. Our approach, Decomp Diffusion, is an unsupervised method which, when given a single image, infers a set of different components in the image, each represented by a diffusion model. We demonstrate how components can capture different factors of the scene, ranging from global scene descriptors (shadows, foreground, facial expression) to local scene descriptors (objects). We further illustrate how inferred factors can be flexibly composed, even with factors inferred from other models, to generate a variety of scenes sharply different than those seen in training time.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on the image decomposition task and proposes a new approach, i.e., Decomp Diffusion, to decompose a scene into a set of factors represented as separate diffusion models. The proposed method can decompose scenes into both global and local concepts. These concepts can further be flexibly composed to generate a variety of scenes.

### Strengths
The idea of leveraging the connection between Energy-based models and diffusion models for image decomposition is interesting and effective. The compositional concepts from images can be discovered in an unsupervised manner. The experimental results show that the proposed method can discover both global and local concepts, and be used for component compositions across multiple datasets and models.

### Weaknesses
1. The quantitative evaluation is not thorough. The current quantitative evaluation only focuses on the global factors, while the quantitative evaluation for the local factors and cross dataset generalization is missing. In contrast, the existing work (COMET) contains quantitative comparisons for the object-level decomposition.
2. As the proposed method contains a set of diffusion models, the computational cost of the proposed method and existing works should be discussed in the paper.
3. For training details in the supplemental, each model is trained on an NVIDIA V100 or an NVIDIA RTX 2080 with the same hours. I was wondering whether the model performance would be different. In addition, is the memory of NVIDIA RTX 2080 24GB or 8GB?

### Questions
1. For the ablation study, why use MSE and LPIPS to evaluate the reconstruction quality, rather than the metrics used in Table 1? How about the results of the ablated versions on other datasets used in the paper?
2. How to determine the types of factors that can be inferred from the image? For example, I am not sure whether the second to the fourth columns correspond to shadow, objects, and background respectively.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper addresses compositional image generation through denoising diffusion models. The unsupervised approach decomposes the input image into several primitives, and the model is able to recompose these primitives together. Experiments are conducted on simple object scenes and human faces, and demonstrate superior performance than SOTAs.

### Strengths
+ The paper addresses compositional modeling for images using denoising diffusion models. The recomposition quality seems promising. 
+ The paper shows that energy functions are additive of primitives.

### Weaknesses
+ The method seems to be similar to [1]
+ What is the computational cost? It may takes more space and computational resources with K diffusion models



[1] Du et al, Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC, ICML 2023

### Questions
+ Is the learned encoder Encθ(x) pre-trained or trained with diffusion model jointly? 
+ Suppose it is jointly trained, how does the network learn to decompose the image into shadow image, object image, background image, etc? Is there any specific constraint for learning these different properties?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper addresses unsupervised image decomposition/re-composition with diffusion models. The authors show equivalence between previous decomposition work COMET, based on energy minimization and gradient descent optimization framework, and the recent diffusion models (DDPM) denoising steps iteration.  They consequently 'substitute' the EM model with a diffusion model conditionned on a set of latent variables z_k. The z_k's are inferred by an Encoder, and are associated to the different factors of the decomposition. 
Experimental results are illustrated on several classical benchmarks (CelebA, Virtual Kitti, Falcor3D, also synthetic data such as CLEVR and Tetris), compared qualitatively and quantitatively to related work.

### Strengths
Unsupervised image intrinsic decomposition/re-composition is very challenging and one of the most fundamental open issues in computer vision. Using diffusion models for this purpose seems a natural choice (given the success of DM in natural image generation, and in learning semantic image properties). The authors give a rigorous justification of their choices from a mathematical point of view.  The paper's idea is well argued. The illustrated results show the strong potential of the approach.  I enjoyed reading the article.

### Weaknesses
Qualitative results are promising but still leave room for improvement. Reconstructed images appear blurry, and at low resolution. But at this stage this is not a major issue and that might be improved by further work.

### Questions
1) I did not find in the paper an explanation about the encoder z = Enc_\theta(x).  How Enc_() is learnt? What ensures that the decomposition is at the local (ie objects, things) or global (ie, illumination, stuffs) level?   What ensures the disentanglement of the decomposition ? (no additional constraints are enforced during the learning stage).  Those aspects might have been discussed in the original paper COMET (I did not read it), however, it is worth to discuss them again in the current paper since it is key for the understanding and  analysis of the success/failures of the proposed approach. 

2) Some details in the approach that are not clear to me. 
2.1 The authors argue that they 'learn a set of different denoising functions to recover an image x_i' (page 4). However, the denoising function \epsilon_\theta is not, in eq.9, parameterized by k.  The only dependance to k is in the input latent variable z_k. It would imply that there is a single denoising function, but with different input argument (in particular the z_k). Please clarify. 
2.2 The encoder Enc_\theta() and the denoising function \epsilon_\theta, are both parameterized by \theta. This is probably a typo, the two networks being parameterized by two sets of independent weights, \theta_1 and \theta_2. Please correct as needed in the paper.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
