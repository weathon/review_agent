# Personalized Feature Translation for Expression Recognition: An Efficient Source-Free Domain Adaptation Method

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Facial expression recognition (FER) models are employed in many video-based affective computing applications, such as human-computer interaction and healthcare monitoring. However, deep FER models often struggle with subtle expressions and high inter-subject variability, limiting their performance in real-world applications. To improve performance, source-free domain adaptation (SFDA) methods have been proposed to personalize a pretrained source model using only unlabeled target domain data, thereby avoiding data privacy, storage, and trans- mission constraints. This paper addresses a common challenging scenario where source data is unavailable for adaptation, and only unlabeled target data consisting solely of neutral expressions is available. SFDA methods are not typically designed to adapt using target data from only a single class. Further, using models to generate facial images with non-neutral expressions can be unstable and computationally intensive. In this paper, the Source-Free Domain Adaptation with Personalized Feature Translation (SFDA-PFT) method is proposed for SFDA. Unlike current image translation methods for SFDA, our lightweight method op-
erates in the latent space. We first pre-train the translator on source domain data to transform the subject-specific style features from one source subject into another. Expression information is preserved by optimizing a combination of expression consistency and style-aware objectives. Then, the translator is adapted to neutral target data, without using source data or image synthesis. By translating in the latent space, SFDA-PFT avoids the complexity and noise of face expression generation, producing discriminative embeddings optimized for classification. Using SFDA-PFT eliminates the need for image synthesis, reduces computational overhead, and only adapts a lightweight translator, making the method efficient compared to image-based translation. Our extensive experiments on four challenging video FER benchmark datasets, BioVid, stressID, BAH, and Af-Wild2, show that PFT consistently outperforms state-of-the-art SFDA methods, providing a cost-effective approach that is suitable for real-world, privacy-sensitive FER applications. 
Our code is publicly available at: github.com/MasoumehSharafi/SFDA-PFT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents Personalized Feature Translation (PFT), a novel source-free domain adaptation method for facial expression recognition. PFT performs personalized adaptation by translating features in the latent space, requiring only neutral expression data from target subjects. This approach eliminates the need for complex image synthesis and source data access during adaptation, while maintaining high computational efficiency. Extensive experiments demonstrate that PFT consistently outperforms state-of-the-art methods across multiple datasets。

### Strengths
The key strength of the proposed PFT method lies in its innovative feature-space translation paradigm. This method avoids the need for complex image synthesis and achieves computational efficiency through lightweight parameter adaptation. It demonstrated superior performance over state-of-the-art methods across four FER benchmarks.

### Weaknesses
1 While the empirical results are strong, the paper lacks an analysis explaining the reasons why feature-space translation is more effective.

2 The experimental validation is centered primarily on FER. It would be valuable to discuss the potential of PFT for other tasks that require subject-specific adaptation, such as face recognition or person re-identification.

### Questions
NA

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Personalized Feature Translation (PFT), a source-free domain adaptation method for facial expression recognition that operates entirely in the feature space. By translating target subject features toward source-style prototypes using only neutral expressions, PFT achieves higher accuracy with significantly lower computational cost than image-based SFDA methods.

### Strengths
This paper introduces a highly original and impactful approach to source-free domain adaptation for facial expression recognition. By formulating a novel feature-level translation method that operates using only neutral target data, it achieves state-of-the-art performance while being dramatically more efficient than image-based alternatives. The work is exceptionally well-supported through rigorous experiments on four diverse benchmarks and presents a practical solution to key real-world constraints like data privacy and computational cost.

### Weaknesses
1．	While the proposed PFT method is well-motivated, the paper lacks a clear theoretical or intuitive explanation of why feature translation in latent space is inherently more robust than image-level translation for expression preservation.

2．	The paper lacks a rigorous explanation or analysis of how the proposed losses ensure that the translator modifies only identity-related features while preserving expression-related ones.

3．	The proposed PFT method relies on pre-training a feature-space translator, but there is no analysis of training stability or overfitting risks. Given the challenge of disentangling expression and identity, is there a risk of overfitting to identity features? Are there issues with convergence or conflicting objectives (style vs. expression)?

4．	The experiments are conducted on only 10 target subjects per dataset. This small sample size raises concerns about the statistical significance and generalizability of the results. The authors should include confidence intervals or perform cross-validation across multiple random splits of target subjects.

5．	The paper mentions hyperparameters such as λexpr and λstyle, but does not discuss their sensitivity or how they were tuned.

### Questions
review the Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
: This paper proposes a personalized feature translation method named PFT, which efficiently achieves domain adaptation for facial expression recognition models using only target user's neutral expression data. It outperforms existing methods across multiple datasets while significantly reducing computational costs.

### Strengths
（1） Structure: The paper features a clear structure, with algorithms presented in easily understandable diagrams. The problem definition is precise, and the motivation is well-articulated.
（2） Innovation: The experiments introduce a personalized translation approach within the feature space, circumventing the instability inherent in traditional image-based methods.
（3） Quality: The experiment design is rigorous, validating the method's effectiveness across four distinct datasets. Detailed ablation studies are provided to demonstrate the contribution of each component.

### Weaknesses
（1） Although comparisons are made with multiple SFDA methods, the paper does not include more recently proposed personalized approaches based on generative models or meta-learning.
（2） The paper notes performance degradation on elderly subjects but does not propose adaptive strategies for age differences. Further exploration of stratified or age-aware adaptation methods is recommended.

### Questions
（1） Compared to models like SHOT and NRC that also update only partial parameters, how does PFT perform in terms of the number of iterations and total time required to reach convergence?
（2） The core method involves using style loss to align identity features while employing expression loss to preserve facial emotion information. Is there any quantitative evidence demonstrating that the translator network indeed learns decoupled representations of these two factors? If the reference image itself carries strong non-neutral expressions, could this contaminate the translation process, leading to loss or confusion of expression information?
（3） Was comparing PFT against such lighter-weight baselines considered to more clearly demonstrate the added value of feature translation over simple normalization or alignment strategies?

### Soundness
2

### Presentation
2

### Contribution
2
