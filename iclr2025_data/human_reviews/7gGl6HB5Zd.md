## Human Reviewer 1

### Summary
This paper introduces a zero/few-shot framework for detecting AI-generated images by analyzing the inherent biases of a pre-trained diffusion model. The authors hypothesize that generated images are more likely to occupy stable local maxima on this learned manifold, characterized by specific curvature and gradient properties. By approximating these properties, they create a criterion to distinguish between real and generated images without requiring large datasets or retraining. Empirical results show their method outperforms other detection approaches across multiple generative models.

### Strengths
- Zero-shot and few-shot capability makes the method practical.
- The theoretical perspective is interesting.
- Empirical results seem promising.

### Weaknesses
1. Some theoretical assertions, such as the assumption that generated samples are more likely to be stable local maxima on the learned manifold, are not fully justified. This assumption underpins the detection criterion, but the paper does not offer a thorough mathematical or empirical rationale to support it.
2. The paper relies heavily on approximations in score-function and curvature estimations (e.g., Eq.5, 16-18). However, there is limited discussion or analysis of the tightness of these approximations. This could lead to questions about the reliability of the approximations, especially when they form the foundation of the theoretical claims. It would be beneficial if the authors provided error bounds analysis or empirical justifications for these approximations.
3. In line 122, the authors argue that previous methods still rely on access to generative methods during training, leading to biases towards those generation techniques. However, the proposed approach also relies on a pre-trained SD1.4. How does the proposed approach avoid the bias from it? For example, a realistic sample generated from a more recent model may not sit on a stable local maximum in the learnt log probability manifold of SD1.4.
4. The presentation can be improved. The mathematical analysis can benefit from illustrative examples, while the detailed proof can be moved to appendix.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper explores a novel method to detect AI-generated images, focusing on zero-shot and few-shot regimes. It identifies key challenges in the field, such as the need for data upkeep with traditional supervised learning methods and limited theoretical grounding for current approaches. The authors propose a framework based on the implicit biases within the manifold of a pre-trained diffusion model, leveraging score-function analysis to approximate manifold curvature and gradient in the zero-shot setting. They extend the method for few-shot scenarios by incorporating a mixture-of-experts strategy. The proposed method demonstrates enhanced performance across 20 generative models.

### Strengths
- The idea of leveraging manifold-induced biases from pre-trained diffusion models to detect generated images is novel and interesting.
- The methodology, essential theoretical formulations, and results are well-articulated, with equations and definitions supporting the approach.
- Experimental results are promising, and Figures 1 and 2 are intuitive and easy to understand.

### Weaknesses
The primary concern is the robustness; please see the Questions below.

### Questions
1. The proposed method relies on pre-trained diffusion models and manifold analysis, which are implemented in high-dimensional space, potentially increasing computational costs. Could the authors provide an analysis of inference time and memory requirements?

2. The method depends on certain hyperparameters, such as perturbation strength and the number of spherical noises. How robust is the method to these parameters across different models, and what guidelines can be provided for selecting these parameters?

3. The authors tested the impact of JPEG compression on the method and reported a slight performance decrease. How does the method perform with other types of image post-processing, such as augmentation, denoising, and flipping?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 3

### Summary
This work addresses the challenge of distinguishing between real and AI-generated images by analyzing biases on the probability manifold of pre-trained diffusion models. The authors develop a method that offers a scalar criterion for classification in zero-shot settings, and experiments demonstrate its effectiveness against current methods.

### Strengths
The theoretical analysis of current diffusion models’ score functions is comprehensive and novel, potentially inspiring further research in this area.

The proposed method generalizes well to unseen generative techniques and achieves superior performance over existing approaches in both zero-shot and few-shot settings.

### Weaknesses
The implementation details in Section 4.3 should be elaborated further to enhance readability and reproducibility.


Typo errors:
- L153 and L149, inconsistent use of $\mathcal N$ and $N$.
- L146 and L170, inconsistent use of $\mathbb{R}$ and $R$
- L157, better use latex log $\log$ for clarity.
- L191, use latex \` for upper quotas.
- inconsistent use of Sec. and Section
- L298-L299, unexpected equation.

### Questions
The author choose SD-1.4 to implement the proposed method, have the author tried other diffusion models, especially recently more advanced methods, such as SDXL and SD3. Does this helps improve the performance?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 4

### Summary
The paper presents a novel approach to detecting AI-generated images using a zero-shot and few-shot framework. By analyzing biases inherent in the manifold of pre-trained diffusion models, the authors introduce a new mathematical criterion based on score functions, curvature, and gradient analysis. This approach generalizes well to unseen generative techniques and outperforms existing methods in both zero-shot and few-shot scenarios. Extensive experiments across a diverse set of generative models further validate the effectiveness of the proposed method.

### Strengths
The paper provides a sound theoretical foundation by integrating manifold analysis with diffusion models, advancing the field of generated image detection.

The empirical results show strong performance, with the proposed method outperforming current state-of-the-art approaches.

The authors conducted experiments across various datasets and generative techniques, including GANs, diffusion models, and commercial tools, providing strong evidence for the robustness of the method.

### Weaknesses
According to the article, the proposed curvature and gradient based metric for detecting generated images is closely related to the score function. However, it is unclear why this method also performs well with models where score functions are not inherently applicable, such as CycleGAN. The authors are encouraged to clarify this connection and explain why the proposed metric shows strong performance even in such cases where score function analysis is not directly relevant.

There are citation issues on page 7 of the article, where footnotes and page numbers are not correctly referenced. The authors should revise the citation formatting to ensure all references are accurate and properly aligned with the content.

The proposed method appears to fit naturally within a zero-shot framework, relying solely on input samples and corresponding perturbations. It is unclear why the few-shot setting was introduced, given that zero-shot scenarios are typically more challenging and often reflect real-world situations. The authors are encouraged to clarify the need for a few-shot setting and explain why zero-shot alone would not suffice as a more compelling and realistic approach.

### Questions
Please see weakness.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3