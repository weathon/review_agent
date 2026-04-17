# Polynomial, trigonometric, and tropical activations

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Which functions can be used as activations in deep neural networks? This article explores families of functions based on orthonormal bases, including the Hermite polynomial basis and the Fourier trigonometric basis, as well as a basis resulting from the tropicalization of a polynomial basis. Our study shows that, through a simple variance-preserving initialization and without additional clamping mechanisms, these activations can successfully be used to train deep models, such as GPT-2 for next-token prediction on OpenWebText and ConvNeXt for image classification on ImageNet. Our work addresses the issue of exploding and vanishing activations and gradients, particularly prevalent with polynomial activations, and opens the door for improving the efficiency of large-scale learning tasks. Furthermore, our approach provides insight into the structure of neural networks, revealing that networks with polynomial activations can be interpreted as multivariate polynomial mappings. Finally, using Hermite interpolation, we show that our activations can closely approximate classical ones in pre-trained models by matching both the function and its derivative, making them especially useful for fine-tuning tasks. These activations are available in the torchortho library.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work reopens the discussion on which functions can be used in deep neural networks (or, more specifically, whether a few of the highlighted functions in this paper can). The paper covers theory and some empirical results. Chiefly, perhaps, is empirical work showing ConvNeXt and GPT-2 can be trained using orthogonal learnable activations, at least on specific shown datasets, which "eliminates the need for additional mechanisms".

### Strengths
- the theoretical underpinnings of Sec 3 are good -- 3.1 leads naturally to 3.2, 3.3, and 3.4. I may have missed _why_ Hermite, Fourier, Tropic activations are focused on. These families are unified cleanly.
- The practical implementation in Sec 3.5 seems like it's reproducible. At least, no obvious flaws with that secion was found.

### Weaknesses
- It's not *technically* a weakness, but convention seems to be moving away from older datasets like CIFAR-10, and older models like GPT-2. This does not sway the decision much, but it seems to upset 'the community' and it does limit the generalizability of your claims, especially to larger scales.
- This is perhaps more of a question, but I'm flagging it as a weakness because lack of clarity on this seems to undermine one of the main points -- that exploding/vanishing gradients are addressed. I.e., it seems that equal-gain guarantees are only guaranteed at the initialization? What evidence is provided for stability long-term?
- Not exactly a weakness either (to the extent to which it just describes an observation!) but the activations (esp Hermite) seem to add a high computational overhead.

### Questions
- Why do you choose Hermite, Fourier, and Tropic activations? Are there other possibilities? What main problems with existing methods to these overcome?
- Are there assumptions regarding distributions that aren't reasonable to expect, and to what extent do you need to check for them? E.g., are Gaussian or uniform distributions assumed by these activation functions, and would that be realistic?
- The Fourier is described in sine-cosine form, but the recommended initialization seems to only make use of a_k? Which coefficients are actually initialized and trained?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors introduce a novel framework to enable learnable activation functions in deep neural networks. In particular, they focus on functions based on orthogonal bases and tropical polynomials. An initialization method for the activation functions  is introduced and the results showcase improvements over static functions. The efficacy of the method is benchmarked across vision and language tasks.

### Strengths
- The main idea is novel and well-motivated.

- The thorough theoretical support on the initialization methods is a valuable contribution to the community.

- I appreciate the benchmarking of the method across both text and vision tasks.

- The latency analysis is an important addition.

### Weaknesses
- I am missing an ablation over different backbones for both vision and language benchmarks.

- Although not a major weakness, additional experimental support on challenging benchmarks would increase the impact of the paper, e.g., on COCO for vision related tasks.

- A discussion on the application of the proposed activation functions for generative models (e.g., diffusion-based models) would be interesting.

Minor:

- The last sentence in ln. 485 seems to end abruptly.

### Questions
I would appreciate if the authors address the issues raised in the weaknesses section.

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
This manuscript introduces a family of learnable activation functions based on orthogonal function bases (Hermite and Fourier) and tropical polynomials. The authors propose a variance-preserving initialization scheme to ensure stable gradient propagation and demonstrate the feasibility of using these activations in deep architectures such as ConvNeXt and GPT-2. The paper combines theoretical analysis, efficient implementations, and empirical validation, suggesting that polynomial activations can indeed yield competitive results with proper initialization.

### Strengths
1. The paper provides a rigorous variance-preserving initialization framework that unifies different activation families under an orthogonal function perspective. This is both mathematically elegant and practically meaningful.
2. By addressing Hermite, Fourier, and tropical bases, the study gives a broad view of orthogonal and piecewise-linear activations, including insightful links to classical activations (ReLU, GELU).
3. Experiments on ImageNet (ConvNeXt) and OpenWebText (GPT-2) convincingly demonstrate that the proposed activations can be trained stably and achieve comparable or slightly better performance than standard nonlinearities.
4. The inclusion of recursive formulations, efficient kernels, and open-sourced code (torchortho) greatly improves the work’s reproducibility and potential impact.

### Weaknesses
1. The reported 30–90% slower training speed (Section 6) is significant. The paper would benefit from more detailed timing analyses and GPU utilization comparisons to quantify the trade-off between performance and efficiency.
2. The experiments focus on classification and next-token prediction tasks. Additional ablations (e.g., fine-tuning, transfer learning, adversarial robustness) could help demonstrate broader applicability.

### Questions
1. The variance-preserving initialization ensures equal forward and backward gains for orthogonal activations. How sensitive is this balance to deviations from the assumed input distributions (e.g., non-Gaussian inputs during training)?
2. When replacing activations in large pretrained models (e.g., GPT-2), how does initialization interact with layer normalization and residual scaling? Are any stability adjustments required?
3. Have the authors explored methods to reduce computational cost, such as approximate polynomial evaluation (e.g., Chebyshev truncation, low-rank projection, or kernel-based approximation)? Could these reduce FLOPs while preserving stability?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores the utilization of orthogonal polynomial activation functions in neural networks, specifically, first deriving the variance preserving initialization for Hermite, Fourier, and Tropical activation function, then conducting experiments on image classification and NLP tasks.

### Strengths
1. The paper is well written, presented with a clear structure.
2. The theorem-proof logic is clear and rigorious.
3. The visualization helps explain the conclusion.

### Weaknesses
Despite the strengths, here are some weaknesses:
1. The motivation is not clear, is it just an exploration on activations? And I am not sure if the proposed activation functions solve any existing problems. (Although I do know that not all innovative thought must solve something discrete, but I do suggest the author to refine this part.) 
2. Since the paper is not the first to design a new kind of activation function, even not the first to use orthogonal  polynomials, I am not sure what is the core innovation.
3. The experiments show very little improvements when Hermite activation function is used. The tropical and Fourier activation function even have worse performance then GELU. These results restrict the value of the paper.
4. More experiments should be conducted, like, more tasks, more models, and more benchmarks.
5. In Proposition C.3, when computing the expectation of $F(x)^2$, the paper uses $\int F^2(x) \dfrac{e^{-x^2/2}}{\sqrt{2\pi}}dx$. I do not think adding the $\dfrac{e^{-x^2/2}}{\sqrt{2\pi}}$ term is rigorious, although it is based on the definition of Hermite. This problem is common for orthogonal polynomials, like Legendre, Hermite, and Chebyshev polynomials, that when adding the orthogonal term, the derivation is much easier.
6. Although the newly designed functions have negligible redundent parameters, it may cause low numerical stability. I am not sure how the authors resolve this problem.

### Questions
Please see the 'Weaknesses' section.

### Soundness
3

### Presentation
3

### Contribution
2
