# Relational Feature Caching for Accelerating Diffusion Transformers

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Feature caching approaches accelerate diffusion transformers (DiTs) by storing the output features of computationally expensive modules at certain timesteps, and exploiting them for subsequent steps to reduce redundant computations. Recent forecasting-based caching approaches employ temporal extrapolation techniques to approximate the output features with cached ones. Although effective, relying exclusively on temporal extrapolation still suffers from significant prediction errors, leading to performance degradation. Through a detailed analysis, we find that 1) these errors stem from the irregular magnitude of changes in the output features, and 2) an input feature of a module is strongly correlated with the corresponding output. Based on this, we propose relational feature caching (RFC), a novel framework that leverages the input-output relationship to enhance the accuracy of the feature prediction. Specifically, we introduce relational feature estimation (RFE) to estimate the magnitude of changes in the output features from the inputs, enabling more accurate feature predictions. We also present relational cache scheduling (RCS), which estimates the prediction errors using the input features and performs full computations only when the errors are expected to be substantial. Extensive experiments across various DiT models demonstrate that RFC consistently outperforms prior approaches significantly. Project page is available at https://cvlab.yonsei.ac.kr/projects/RFC

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes an improvement of TayloSeer, motivated by the highly correlated magnitudes of change of input and output features. It uses the change of input features to estimate the change of output, therefore reducing the computation to a large extent. On both text-to-image and text-to-video tasks, RFC manages to achieve better or comparable performances against baselines. Visual results from the paper also support the effectiveness of this method.

### Strengths
1. The motivation of this paper is clear. The empirical results in Figures 1 and 2 serve as strong evidence for the proposed method.
2. The discussion of related work is thorough, further enhancing the novelty of this paper.
3. The experimental analysis is comprehensive. The authors validate their method not just on one model or task, but across three distinct, large-scale generative tasks (class-conditional, T2I, T2V) using modern, powerful models.

### Weaknesses
Please see the weakness of the method and experiments in the questions.

Here are some weaknesses in the presentation:
1. The related work section is a bit hard to read with the long paragraph. I suggest the author reorganize this section to improve the readability.
2. I suggest using the full name of RFE and RCS as the paragraph header.

### Questions
1. How does RFC perform on distilled models?
2. What is the recomputation rate under different parameter settings? In other words, is $\tau$ hard to tune for different DiT models? It seems this value differs across the DiT models tested.
3. The empirical results (Figures 1 and 2) are obtained on ImageNet and DiT, which are relatively simple. Are the findings the same in a different case? For example, it might be more convincing to also show empirical results on FLUX.1 dev or HunyuanVideo.
4. RFC seems to be slower than the baseline. Where does the main overhead come from, and did the author think about how to reduce it?
5. Did the author study whether shallow layers / early timesteps should be skipped or not to achieve better results?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Relational Feature Caching (RFC) to accelerate Diffusion Transformers by exploiting correlations between input and output features, rather than relying solely on temporal extrapolation as in prior methods like TaylorSeer. The approach introduces Relational Feature Estimation (RFE) to predict output changes from input variations and Relational Cache Scheduling (RCS) to adaptively trigger full computations based on estimated errors. Experiments on image, text-to-image, and video generation tasks show consistent improvements over existing caching methods in both quality and efficiency.

### Strengths
- Clear motivation with strong empirical evidence showing the limitation of purely temporal forecasting.
- Simple yet effective design—RFE and RCS are well-justified and complementary.
- Comprehensive experiments across multiple diffusion models and tasks.

### Weaknesses
- In Table 2, TaylorSeer achieves the highest CLIP scores; the paper should discuss why RFC does not consistently outperform it on semantic alignment metrics.
- The paper could analyze RFC’s applicability to U-Net–based diffusion models to better demonstrate generality and architectural adaptability.

### Questions
- While RFC significantly accelerates DiTs, what happens at higher acceleration ratios (larger N)? A discussion on generation quality degradation at extreme speedups would help readers assess its robustness.

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
3

### Summary
Objective: Accelerate Diffusion Transformers by improving feature caching accuracy. The paper identifies that forecast-based caches suffer from large errors due to irregular output changes and strong input–output correlations in modules. It proposes Relational Feature Caching (RFC) with two components: Relational Feature Estimation (RFE) to predict output-change magnitudes from inputs, and Relational Cache Scheduling (RCS) to estimate prediction error from inputs and trigger full computation only when errors are likely high. Experiments across multiple DiT models show consistent, significant improvements over temporal extrapolation baselines, with planned code release upon acceptance.

### Strengths
This paper studies an important topic of feature caching, which is critical for optimizing the efficiency of diffusion transformers.

The identification of input–output correlation as a predictor of output changes is insightful and well supported by the empirical analysis.

The proposed RFC showcase on multiple metrics and settings, demonstrating a comparable to superior performance. The authors also present qualitative results, which are promising.

### Weaknesses
The organization of the manuscript needs improvement. There is overlapping and duplicate content across the first three sections. It may be better to defer the detailed discussion of related work to a later section and to reorganize Section 2 and Section 3.1.

While the empirical correlation between inputs and outputs is demonstrated, the paper offers little formal analysis explaining why this correlation should hold across arbitrary architectures or datasets.

### Questions
line 302-304, "For a fair comparison, we reproduce the results of state-of-the-art methods using the official source codes, and adjust the threshold τ in Eq. (13) to ensure that the average number of full computations (NFC) matches that of other methods." can you elaborate more on how to adjust the threshold tau?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses inefficiencies in diffusion transformers (DiTs) by advancing feature caching techniques used to accelerate inference. Prior approaches speed up computation by temporally extrapolating and reusing output features of expensive modules, but these methods can incur significant prediction errors. Through detailed analysis, the authors find that such errors arise from irregular feature magnitude changes and that there is a strong correlation between a module’s input and its output features. Building on these insights, they introduce Relational Feature Caching (RFC), a novel framework that leverages the relationship between inputs and outputs to improve feature prediction accuracy. RFC includes two key components: Relational Feature Estimation (RFE), which uses input features to predict output changes more reliably, and Relational Cache Scheduling (RCS), which estimates likely prediction errors from inputs and recomputes outputs only when large errors are expected. Experiments on various DiT models show that RFC consistently outperforms earlier techniques, significantly improving efficiency and accuracy.

### Strengths
This work introduces novel components—Relational Feature Estimation (RFE) and Relational Cache Scheduling (RCS)—that have not been explored in prior work. It utilizes input–output relationships, enabling a more accurate prediction of output features. The paper clearly articulates the motivation, challenges, and contributions, and provides a general framework that could inspire further research for DiTs.

### Weaknesses
Clarity of Mathematical Formulation (RELATIONAL FEATURE CACHING section):

The presentation of equations in the RELATIONAL FEATURE CACHING section lacks clarity, making it difficult for readers to follow the mathematical foundations of the approach. For instance, the connection between the two components, RFE and RCS, is not clear at the beginning, and the logical flow from one equation to the next is not always well-motivated.

Actionable suggestion: Including intuitive descriptions or intermediary steps (potentially with illustrative diagrams or simplified toy examples) would help clarify the input-output relationship modeling and the estimation process.

Experimental Evaluation Coverage:

While the experiments show consistent gains across several DiT models, the paper primarily focuses on performance improvements and does not fully explore scenarios where the method may fail or be less effective (e.g., with particularly noisy or weak input-output correlations).
Actionable suggestion: Include ablation studies or failure case analysis to identify situations where the relational estimation approach may struggle or need further modification. Examining a broader range of conditions, such as varying degrees of input-output correlation, would provide a more comprehensive understanding of the method's robustness. For Table 4, provide a brief written summary in the main text to guide the reader through the table and highlight the most important results.

### Questions
How does the model perform in terms of efficiency when the RCS component has a different scheduling policy?

Is there a clear trade-off pattern for the method?

### Soundness
3

### Presentation
2

### Contribution
3
