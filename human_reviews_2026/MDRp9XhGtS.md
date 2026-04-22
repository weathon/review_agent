# TIMESLIVER : SYMBOLIC-LINEAR DECOMPOSITION FOR EXPLAINABLE TIME SERIES CLASSIFICATION

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Identifying the extent to which every temporal segment influences a model’s predictions is essential for explaining model decisions and increasing transparency. While post-hoc explainable methods based on gradients and feature-based attributions have been popular, they suffer from reference state sensitivity and struggle to generalize across time-series datasets, as they treat time points independently and ignore sequential dependencies. Another perspective on explainable time-series classification is through interpretable components of the model, for instance, leveraging self-attention mechanisms to estimate temporal attribution; however, recent findings indicate that these attention weights often fail to provide faithful measures of temporal importance. In this work, we advance this perspective and present a novel explainability-driven deep learning framework, TimeSliver, which jointly utilizes raw time-series data and its symbolic abstraction to construct a representation that maintains the original temporal structure. Each element in this representation linearly encodes the contribution of each temporal segment to the final prediction, allowing us to assign a meaningful importance score to every time point. For time-series classification, TimeSliver outperforms other temporal attribution methods by 11\% on 7 distinct synthetic and real-world multivariate time-series datasets. TimeSliver also achieves predictive performance within 2\% of state-of-the-art baselines across 26 UEA benchmark datasets, positioning it as a strong and explainable framework for general time-series classification.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a framework, TimeSliver, to generate faithful measures of temporal feature importance. The framework constructs temporal interpretations by utilizing a combination of temporal segments and a symbolic representation. The effectiveness of this approach is demonstrated through experiments on both synthetic and real-world datasets, where it is compared against several baseline methods.

### Strengths
•	The proposed framework's combination of temporal segment representation and symbolic composition-based representation is a novel approach for constructing measures of temporal importance.

•	The division of temporal attributions into positive and negative contributions is a valuable feature, as it provides insight into the directionality of a feature's impact on the model's output.

•	The experiments are thorough, demonstrating that TimeSliver can achieve significant improvements over various baselines on both synthetic and real-world datasets while maintaining high predictive performance.

### Weaknesses
•   **Clarity of Methodological Integration**: The paper incorporates advanced signal processing concepts, such as the Short-Time Fourier Transform (STFT). However, the specific mechanism by which the output of the STFT is integrated into the model's architecture and used for prediction is not sufficiently detailed. 

•	**Rationale for Design Choices**: Several core methodological definitions and design choices could be further justified. For example, the formal definition of "temporal attribution-based explainability" is somewhat abstract, and the rationale for specific formulations, such as the interaction matrix P and the use of the ReLU function for attributions, is not fully explained.

### Questions
**Main Concerns:**

1.	**Definition of Explainability (Definition 2.1)**: The concept of "Temporal Attribution-Based Explainability" is defined by the "model’s ability" to assign importance scores. This definition is somewhat ambiguous. Could the authors provide a more formal or precise definition of "ability" in this context and clarify how it is quantitatively measured?
2.	**Interaction Matrix Formulation (Sec 2.2.3)**: The matrix $P$ is defined as a linear weighted summation of all temporal segments, yet it is purported to capture interactions. Could you elaborate on how this linear formulation is sufficient to capture pairwise or higher-order interactions, which are typically non-linear?
3.	**Use of ReLU for Attributions (Equation 3)**: Equation (3) utilizes the ReLU function to normalize and separate the positive and negative attributions. Could you provide the justification for choosing ReLU for this task over other potential functions? What is the key insight behind this specific formulation for deriving positive and negative contributions?
4.	**STFT (Line 202)**: The framework computes the Short-Time Fourier Transform. Could the authors please clarify how the resulting frequency-domain features are subsequently used in the prediction model? 
5.  **Qualitative Analysis of Experiments**: While the paper presents a framework for interpretability evaluation (e.g., via masking-based metrics), it lacks intuitive visualization and deeper qualitative analysis. The authors do not show how the model explains individual samples — for example, which time points or variables are most critical for a specific prediction, and whether these insights align with domain expertise.

**Minor Concerns:**

1.	**Notation (Lines 131-144)**:  input vectors ($x_i$) in boldface in lines 131-134, 137, 139, and 144.

2.	**Attribution Function (Line 122)**: The notation seems to imply that $\alpha_i = f_{att}(\hat{y}_i) $ . This suggests the attribution is a function of the model's prediction, which may not be the intended meaning. Please clarify this relationship.
3.	**Equation (3)**: The expression for the positive attribution A⁺ appears to be missing the arguments $g_{ij}$ and $\sigma_{ij}$.

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
3

### Summary
The paper proposes a new framework named TimeSliver, targeting faithful explainability and strong performance. The authors point out that traditional deep-learning models often suffer from the black-box nature of neural networks. Furthermore, existing post-hoc methodologies (e.g., Grad-CAM) or approaches utilizing attention weights suffer from an "unfaithfulness" issue, failing to address the model’s reasoning. TimeSliver attempts to solve this by decomposing the input into symbolic and latent representations of time-series data.

To elaborate, the framework processes the input through two parallel modules. The first module learns a latent representation using a 1D convolution operator, targeting the corresponding temporal segments. The second module creates a symbolic composition matrix by discretizing the raw input into categorical bins and applying average pooling over the fixed segment windows to capture the normalized frequency of each symbol within each segment. The key module, Module 3, computes a global representation by learning the ‘symbolic-linear’ representation (a cross-representation matrix) composed of the results from Modules 1 and 2. This structure allows TimeSliver to decompose the final prediction and assign positive/negative attribution scores to every temporal segment.

Experiments were conducted on three real-world applications and four synthetic datasets, comparing TimeSliver with nine baseline methods. Additionally, its performance was evaluated on 26 multivariate time-series classification tasks.

### Strengths
The paper's primary strength lies in its ability to achieve consistently better performance than existing methodologies in most tested cases. It is impressive that this strong performance is achieved using a methodology that is relatively simple and computationally efficient. This simplicity is a significant advantage, as the model is designed with a highly intuitive intention, ensuring that the authors' original goal is well-aligned with the final model architecture and its effective results.

### Weaknesses
The authors provide compelling evidence that the symbolic component (Module 2) is essential for the model's explainability goal; replacing the symbolic matrix $Z$ with a non-symbolic projection $X_{proj}$ significantly degrades explainability metrics (AUPRC), as shown in Figure 3 and Table 9 . Given that the paper's main objective is not necessarily maximizing predictive performance, it would nonetheless be beneficial to understand the full impact of this substitution. The paper does not appear to report the predictive accuracy results for this specific ablation study, accuracy results without module 2. Providing this information would offer a more complete characterization of the framework, clarifying whether the symbolic module also contributes to predictive accuracy or if its role is only focused on enabling the model's core explainability goal.

### Questions
The paper clearly demonstrates that replacing the symbolic matrix $Z$ with $X_{proj}$ significantly harms explainability metrics (AUPRC), as shown in Table 9 1. What was the impact on predictive accuracy in that same ablation experiment?
More broadly, what is the exact influence of Module 2 on the model's final predictive performance and the training of the other modules?

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
The paper introduces TimeSliver, and explainable time series model aim to explain both positive and negative contributions in terms of which time points are most important for the prediction.

### Strengths
- The writing of the paper is clear for some sections (Section 2.2.1 to 2.2.3)
- The methods are intuitive and understandable (Section 2.2.1 to 2.2.3)
- The evaluation methods of the results are pretty sound.

### Weaknesses
- There are some notations and logic disconnect in the method section (Section 2.2.3 to 2.2.4). Notations such as $f_{cls}$ and $f_{att}$ should be used in section 2.2.4 but they are not.
- Justifications and explanations of the formulae for the attributions are missing. The author instead only focuses on the justification of the scaling aspect on the formula.
- The definition of positive and negative contributions are not there. In the formula it seems to suggest one thing but in the evaluation it seems to suggest another.
- The evaluation results are not super impressive. This is okay if the method section is strong and sound.

More explanations are in the "Questions" below.

### Questions
Question

- Line 121. It is better to say "comprising two components - predictions and attribution". Note that this notation $f_{cls}$ and $f_{att}$ is never used again. Also by the formula, the domain of $f_{att}$ is $\{0, \ldots, C-1\}$, which is very weird. The $\alpha_i$ should depend on the input x and the network as well (as shown in Fig 1). So I think the notations have to be updated.

- In section 2.2.1, I am confused with the statement "partitions x_i into ... overlapping segments". If we are partitioning the time series, they would not overlap. Secondly, using a 1d convolutional operator with kernel size m, the resulting "output" would not be a "temporal segment", as a "temporal segment" is defined to be a subsequence of the original segment $x_i$. Convolving with a kernel would mostly not result in a subsequence of the original sequence. So I think we should change the wording here a little bit.

- There is a disconnect from Section 2.2.3 to Section 2.2.4. From what I understand from the introduction, this work TimeSliver, is not a post-hoc interpretation method, but a model with explanation. The author described how to get the matrices P's from the input time series x, but what I assumed here is that there is an extra model that treats P as the input features and output y. This is in reference to Figure 1, the "$f_{cls}$". However, in section 2.2.4, it says "Once the model is trained to predict y", it could mean the model is trained separately. So I think here, it is better to add a sentence here that "we are training a model $f_{cls}$ using $P$ as the features.

- Continuing with the above $f_{att}$ should be described that it takes the output of $f_{cls}$ and $P$ as the inputs and relate it to $\phi^+_k$ and $\phi^-_k$ in equation 4.

- For equation 3, what if the terms in $P_{ij}$ are all positive or negative (across $k$)? Then the denominator would be undefined?

- For equation 3, there does not seem to have any discussion on this definition. Although it was mentioned that more explanations are in Section A in the appendix, the explanations are only on the "scaling". Thus, if I am to interpret it mayself, $g_{ij}$ is the gradient of $\hat{y_c} ^p$ w.r.t. $P_{ij}$ . This means that it is the rate of change of the prediction when $P_{ij}$ changes. $g_{ij}$ is positive when the prediction for the class increases when $P_{ij}$ increases. But $P_{ij}$ is written as a sum of the $k$ terms, corresponds to the time k. Thus if the $k$ time point contribution to $P_{ij}$ is positive, then this $\zeta$ would become the "positive contribution". Multiplying by the denominator will have a scaling discussion as the text following this equation. So **does that mean that a positive contribution means how much the data time point $k$ would drive the prediction higher, while a negative contribution means how much the data time point $k$ would drive the prediction lower?** In the introduction and contributions, the author mentioned that TimeSliver provides positive and negative temporal attribution scores. Are there any definitions on what this actually means? From the formula, I think the answer is yes.

- Continuing with the above observation. If the answer is yes to the bolded question, then negative contributions are also good predictors as they would drive the prediction of the class lower. This could help the model performance if the ground truth class is not that class. Thus I am not following the discussion on paragraph in Lin 311 to 315. It seems like the authors are claiming that "negative contributions" correspond to "noisy time points", instead of "drivers to lower predictions".

- Table 4 is not referenced anywhere. I think it should be referenced from Section 3.2 in the experiment.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes TimeSliver, a deep learning framework that combines raw time series with symbolic representations to provide temporal attribution scores for explainable time series classification. The method achieves competitive predictive performance while offering interpretable insights into which time segments influence predictions.

### Strengths
1. Novel architectural design: The combination of symbolic abstraction with raw time series through linear composition ($P = Z^T Q$) is creative and well-motivated. The structural analogy to STFT provides good intuition.
2. Strong experimental validation: The paper demonstrates 11-18% improvement over baselines on synthetic datasets and maintains competitive performance on 26 UEA benchmark datasets while providing explainability. 
2. Scale-invariance property: The theoretical justification for why symbolic representation yields scale-invariant attributions (Section A) is valuable and validated empirically.

### Weaknesses
1. baseline comparisons: Using computer vision methods (Grad-CAM, DeepLIFT) directly on time series without proper adaptation may disadvantage these baselines.  Limited baseline coverage: No comparison with recent time series-specific XAI methods like LIME-TS or kernel-based approaches. 
2. Missing Theoretical Guarantees : No completeness axiom (unlike Integrated Gradients), No efficiency property (unlike SHAP). More importantly, no monotonicity - Higher attribution ≠ more important. why and how high scores mean important segments

### Questions
1. P loses temporal ordering information. How does this affect attribution for time-dependent patterns?
2. The gradient-based attribution assumes differentiability, how does it interact with ReLU operations.

### Soundness
3

### Presentation
3

### Contribution
3
