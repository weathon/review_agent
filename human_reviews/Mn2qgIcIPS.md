# Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs

- Avg Score: 7.33
- Decision: Accept (Spotlight)
- Scores: 8, 8, 6

## Abstract
Low-light image enhancement poses a significant challenge due to the limited information captured by image sensors in low-light environments. 
  Despite recent improvements in deep learning models, the lack of paired training datasets remains a significant obstacle. 
  Therefore, unsupervised methods have emerged as a promising solution. 
  In this work, we focus on the strength of curve-adjustment-based approaches to tackle unsupervised methods. 
  The majority of existing unsupervised curve-adjustment approaches iteratively estimate higher order curve parameters to enhance the exposure of images while efficiently preserving the details of the images. 
  However, the convergence of the enhancement procedure cannot be guaranteed, leading to sensitivity to the number of iterations and limited performance.
  To address this problem, we consider the iterative curve-adjustment update process as a dynamic system and formulate it as a Neural Ordinary Differential Equations (NODE) for the first time, and this allows us to learn a continuous dynamics of the latent image. 
  The strategy of utilizing NODE to leverage continuous dynamics in iterative methods enhances unsupervised learning and aids in achieving better convergence compared to discrete-space approaches. Consequently, we achieve state-of-the-art performance in unsupervised low-light image enhancement across various benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This manuscript introduces a CLODE method that enhances low-light images by framing exposure adjustment as a Neural Ordinary Differential Equation (NODE) problem.  It improves on traditional curve-adjustment techniques by automatically adjusting exposure without manual input and converges to an optimal exposure using adaptive ODE solvers. CLODE also includes noise removal and curve parameter estimation, and users can control exposure levels by adjusting integration intervals.
===================================================================

The authors have raised many of my concerns during the rebuttal phase. I am now ready to increase the rating.

### Strengths
Strengths of the CLODE method:

1. CLODE’s use of Neural ODEs allows it to dynamically adjust exposure levels based on image-specific needs.
2. The approach is really interesting. Mainly the noise removal block is a very good idea for this type of tasks.
3. Ablation study is really exhaustive.
4. Results are evaluated well.
5. Manuscript is well written and organized. Very easy to follow.
6. Technical content is really well written with proper citations.

Good Paper!!!

### Weaknesses
Weaknesses:

1. Novelty is somewhat limited.
2. Comparisons with recent-most SOTA methods should be included.
3. Flowcharts should be improved like Figure 2.
4. Conclusion should be improved although I can understand there is a space limitation. Please make other parts smaller and add some key findings in conclusions.
5. Some more recent references from 2024 is required.

By framing the enhancement process as a Neural ODE, CLODE relies on numerical approximation methods (e.g., dopri5) that inherently carry approximation errors. This can provide solvers with adaptive step sizes but they cannot guarantee analytical precision or stability across all image conditions, particularly in complex cases with high dynamic ranges. This approximation could lead to unpredictable or suboptimal results, especially when the integration steps or the neural network's learned parameters are insufficiently fine-tuned for certain low-light conditions. Can the authors provide some low-light images from real-world dark conditions and demonstrate the method.

### Questions
1. Adaptive ODE solvers increase computation time and memory usage. Any idea regarding the number of pixel operations taking place here?

2. The curve parameter estimation module might not generalize well to diverse lighting conditions. Can you provide some generalization test regarding this.

3. Manual adjustment of exposure intervals introduces subjectivity. An ablation can give us more insights.

4. NODE-based adjustments may struggle with high-dynamic-range images, risking instability and suboptimal exposure adjustments. Can you provide the separate loss curves for better understanding of the training dynamics.
==========================================================================

### Soundness
3

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
4

### Summary
In this article, the author discovers that the traditional curve-based iterative methods cannot guarantee the convergence of the enhancement process, leading to sensitivity to the number of iterations and limited performance. The author regards the iterative curve adjustment process as a dynamic system and, for the first time, formulates it as a neural ordinary differential equation (NODE) to learn the continuous dynamics of underlying images. Specifically, this method leverages NODE to harness the continuous dynamics within iterative methods, achieving better convergence than discrete space methods. Finally, the article demonstrates superior performance over current unsupervised low-light image enhancement methods on various benchmark datasets.

### Strengths
1. In this article, for the first time, the iterative curve adjustment update process is regarded as a dynamic system and formulated as a neural ordinary differential equation (NODE) to learn the continuous dynamics of underlying images. This approach offers an alternative perspective for addressing low-light issues；
2. The article employs a substantial amount of mathematical language to demonstrate how to transform the iterative curve adjustment update into a continuous process, which enhances the logical strength of the paper；
3. The language of the article is quite fluent and adheres to English writing standards；

### Weaknesses
1. Since this article addresses issues in low-light scenarios, presenting more objective results in complex scenarios would enhance the persuasiveness of the article.

### Questions
1. Since this article addresses issues in low-light scenarios, presenting more objective results in complex scenarios would enhance the persuasiveness of the article.

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
4

### Summary
This paper introduces a novel approach for enhancing low-light images by formulating the problem as a Neural Ordinary Differential Equations (NODE) problem. The authors propose CLODE, a dynamic system that leverages continuous dynamics of latent images to improve exposure levels in images. The method is unsupervised, which is particularly useful given the difficulty of obtaining paired low-light and well-exposed images for supervised learning.

### Strengths
1. The paper presents a creative application of NODEs to the problem of low-light image enhancement, which is a contribution to the field.

2. Addressing the challenge of lacking paired datasets, the unsupervised nature of CLODE makes it highly applicable to real-world scenarios where ground truth data is scarce.

3. The paper provides a thorough explanation of the CLODE model, including its theoretical foundations and implementation details.

4. The method achieves competitive results across multiple benchmarks, which is a strong point of its potential impact.

### Weaknesses
1. The paper does not explicitly address the computational cost of solving NODEs compared to other methods, which is an important consideration for practical applications. Authors should report this results compared with other similar methods.

2. The paper does not discuss the potential for overfitting, especially since the model is tailored to low-light images, which may have unique characteristics.

3. In my opinion, the author's comparative experiment is still not sufficient. It should be compared on more real benchmarks to highlight the superiority of the method. The current comparison seems to be insufficient, including the following:

[1] Contrast enhancement based on layered difference representation.

[2] Perceptual quality assessment for multi-exposure image fusion.

[3] Structure-revealing low-light image enhancement via robust retinex model.

[4] Naturalness preserved enhancement algorithm for non-uniform illumination images.

[5] On the evaluation of illumination compensation algorithms.

4. There are many different paradigms for unsupervised methods. Compared with this type of method, is the author's method more scalable? If it can be improved on other methods, I think this article will have higher value.

In general, I will make further adjustments based on the author's rebuttal.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
