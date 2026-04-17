# ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
While deep learning has achieved impressive performance in time series forecasting, it becomes increasingly crucial to understand its decision-making process for building trust in high-stakes scenarios. Existing interpretable models often provide only local and partial explanations, lacking the capability to reveal how heterogeneous and interacting input variables jointly shape the overall temporal patterns in the forecast curve. We propose ProtoTS, a novel interpretable forecasting framework that achieves both high accuracy and transparent decision-making through modeling prototypical temporal patterns. ProtoTS computes instance-prototype similarity based on a denoised representation that preserves abundant heterogeneous information. The prototypes are organized hierarchically to capture global temporal patterns with coarse prototypes while capturing finer-grained local variations with detailed prototypes, enabling expert steering and multi-level interpretability. Experiments on multiple realistic benchmarks, including a newly released LOF dataset, show that ProtoTS not only exceeds existing methods in forecast accuracy but also delivers expert-steerable interpretations for better model understanding and decision support. The source code is available at https://github.com/SKURA502/ProtoTS.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a hierarchical prototype framework for explainable time series forecasting, combining prototype similarity with multi-channel embeddings to improve forecasting accuracy.

### Strengths
1. The paper proposes a clear framework that connects prototype with time series forecasting.

2. Experimental results show competitive accuracy and provide case studies that make it easy to understand.

### Weaknesses
1. The introduction claims that ProtoTS is the first to model prototypical temporal patterns. However, Sec 2 already cites some prototype-based time-series models . It would be clearer if the authors could clarify more explicitly what innovation distinguishes ProtoTS from these prior works.

2. The paper does not clearly explain how the learned prototypes remain semantically meaningful throughout training. The optimization objective lacks explicit constraints ensuring diversity or interpretive consistency among prototypes, which may lead to redundant prototypes.

3. In experiment, adding the discussion of computational cost and complexity analysis would make the performance more convincing.

### Questions
The authors could refer to Weakness

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
This paper introduces ProtoTS, an interpretable time series forecaasting framework that learns hierarchical prototypes. The model combines a multi-channel bottleneck encoder for heterogeneous variables with a hierachical prototype tree for coarse-to-fine interpretability. Experiments on load-forecasting and electricity price datasets show accuracy gains and improved interpretability.

### Strengths
- The framework is conceptually novel. The proposed hierarchical prototypes for interpretable time series forecasting is novel and empircally verified. 
- The model design, including multi-channel embedding, bottleneck fusion, is clearly justified and contributes to performance. 
- The case study is intuitive and interesting. It well-demonstrates real-world applicability. 
- The paper is generally well written, equations align with intuition, and the proposed architecture is easy to follow.

### Weaknesses
- Experiments are limited to two datasets within the energy domain. Broader validation on additional domains (e.g., weather, traffic, retail, healthcare) would better demonstrate the generality and adaptability of ProtoTS.
- Forecasting accuracy is reported only with MAE. Reporting MSE would provide a more comprehensive assessment.
- Baselines include mainly traditional statistical or early deep models (ARIMA, XGBoost). I suggest comparing with more recent models such as PatchTST and TimesNet, and FEDformer.
- Reported accuracy gains are not validated by statistical significance tests 
- The relationship with ProseNet (Interpretable and Steerable Sequence Learning via Prototypes, KDD 2019) should be discussed more thoroughly, as both share conceptual similarities in prototype-based interpretability and steering. 
- For quantiative interpretability test, only User Precision and System Usability Score are reported. I suggest considering additional metrics such as fidelity, faithfulness, which do not rely on human judgment.
- The robustness of learned prototypes could be evaluated by introducing noise or perturbations in the input time series to test whether prototype assignments remain stable.
- Please clarify whether the 12 participants in the user study were domain experts (e.g., energy analysts) or general users. 
- Also, please justify or discuss whether a sample size of 12 provides sufficient statistical reliability.
- The case study focuses solely on load forecasting. I suggest adding more case studies on the other dataset as well. I belive this case study is the core result of the paper’s interpretability claim and thus presenting additional case studies (e.g., price forecasting, weather, or traffic datasets) would substantially strengthen the paper.
- The anonymous GitHub repository provides only minimal instructions. Please add detailed environment setup instructions, data-preprocessing scripts, configuration files, and run commands so that results can be reproduced reliably.

### Questions
Please refer to weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ProtoTS, an interpretable forecasting framework that models prototypical temporal patterns. The weights of these prototypes are determined by measuring the similarity between different prototypes. To balance predictability and interpretability, the authors introduce a hierarchical structure, allowing a root prototype to be split into more detailed child prototypes. The experimental results indicate that ProtoTS achieves strong forecast accuracy while also providing interpretations that can be guided by expert knowledge.

### Strengths
1. The proposed hierarchical structure, where prototypes are progressively refined into lower levels, offers a compelling approach to managing the trade-off between forecasting accuracy and interpretability.

2. The experiments demonstrate that ProtoTS delivers strong performance on time series forecasting tasks that include exogenous variables.

### Weaknesses
1.	**Guidance on Hierarchical Structure**: While the hierarchical prototype structure is a key contribution, the paper lacks a formal or suggested methodology for determining the optimal number of leaf-prototypes. Providing a general splitting rule would enhance the practical applicability of the framework.
2.	**Clarity of the Loss Function**: The formulation of the loss function, which combines an L1-norm with an entropy regularization term, could be further clarified. Specifically, the intended effect of the entropy regularization on promoting a few dominant prototypes is not immediately apparent.
3.	**Validation of Interpretability**: The assessment of interpretability relies on a questionnaire administered to 12 users. While this provides initial evidence, the small sample size and the potential for subjective responses may limit the generalizability of these findings.

### Questions
**Main concern:** 

1. **Variable Interactions**: The paper states that the prototype-instance similarity is computed by "incorporating a multi-channel embedding and bottleneck fusion mechanism" to model interactions between input variables. However, the embedding described in Equation (2) appears to be a summation of individual variable embeddings prior to the bottleneck layer. Could the authors elaborate on how this specific architecture facilitates the modeling of interactions between variables?

2. **Entropy Regularization**: In line 300, the stated goal of the entropy term in the loss function is to "encourage a few main prototypes to cover most predictions." Given that negative Shannon entropy is minimized by a discrete uniform distribution, could the authors provide further intuition on how this term achieves the stated objective of encouraging a sparse set of dominant prototypes? Additionally, the inclusion of a tuning parameter (e.g., $\lambda$) to balance the L1-norm and the regularization term seems warranted and would be a valuable addition.

3. **Scope of Regularization**: The current regularization term appears to only consider the first level of the hierarchical structure. Have the authors considered extending this to other levels of the hierarchy, and what might be the potential impact of such a change?

4. **Choice of L1-Norm**: Could the authors provide the rationale for selecting the L1-norm for the loss function over other metrics such as the L2-norm or domain-specific measures like Dynamic Time Warping?

5. **Prototype Splitting Criteria**: The experiments include a sensitivity analysis on the number of prototypes. However, a more detailed discussion on the rules for splitting prototypes would be beneficial. Have the authors considered implementing a formal rule, such as a threshold-based criterion (e.g., $ f(Z|\mu)>\alpha $  or a metric like the Gini index, to guide the splitting process and maintain a robust balance between predictive accuracy and interpretability?
6. **Simulation for Interpretability**: To further strengthen the claims regarding interpretability, have the authors considered conducting a simulation study? A simulation with a known ground truth for the underlying prototypes and hierarchical structure could provide a more objective evaluation of the model's ability to recover these causal relationships.

**Minor concerns:** 

1.	Line 29: The phrase "their overall interpretability and their potential" could be revised for clarity, as the second "their" is redundant.

2.	Line 76: "input covariates" is repetitive. Using either "inputs" or "covariates" would be more concise.

3.	Line 84: Similarly, "interactions of covariate combinations" could be shortened to "interactions of covariates."

4.	Figure 2 (Similarity Computing): The percentages shown (0.35, 0.65, 0.1) sum to 1.1. Please verify these values.

5.	Line 208: "is holiday" should be corrected to "holiday."

6.	Line 234: The text states the process is "along the temporal dimension and then along the feature dimension," while Equation (3) suggests the fusion occurs along the feature dimension first. Please clarify this sequence.

7.	Line 236: The meaning of T in Equation (3) is unclear. Please provide a definition.

8.	Line 249: For consistency, \mu should be rendered in boldface.

9.	Line 303: To be precise, the L1-norm should be denoted with a subscript, for instance, $|| \cdot ||_1$.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ProtoTS, an interpretable forecasting framework that models hierarchical prototypes to achieve both high accuracy and explainability in time series forecasting. The method builds a hierarchy of prototypes representing multi-scale temporal patterns and incorporates a multi-channel similarity mechanism for heterogeneous inputs. The approach is ambitious and technically solid, aiming to connect interpretable representation learning with practical forecasting performance.

### Strengths
- The paper presents an original attempt to unify interpretability and accuracy through hierarchical prototype learning.
- It offers a systematic framework combining multi-level structure, prototype reasoning, and human-editable interpretability.
- The experiments and visual analyses are thorough and insightful.

### Weaknesses
### **Problem & Motivation**

- The motivation to move beyond local or post-hoc explanations toward global interpretable forecasting is strong and well justified.

- However, it remains unclear how this approach differs fundamentally from previous prototype-based or dictionary-learning forecasting methods that already encode recurring temporal patterns.

- The paper could better articulate how hierarchical prototypes specifically enhance interpretability compared to flat prototypes or segment-level reasoning.

- The expert-steering feature is intriguing, but its implementation and scalability remain vague.

---

### **Method**

- It is not clearly explained how prototypes and the model output are computed, given an input $x$. The mechanism by which a new time series sample is assigned to or reconstructed from prototypes is ambiguous.

- The description also leaves uncertainty about whether there is a prototype per sample or whether prototypes are shared across samples in a latent cluster space.

- Since the interactions occur in the latent space, it is unclear how one can meaningfully interpret what each prototype represents in the original time domain. A mapping or visualization process should be clarified.

---

### **Experiments**

- The experimental scope is narrow. All evaluations are performed on a limited set of datasets; broader diversity in data domains (e.g., finance, energy, healthcare) and different lookback windows would make the results more generalizable.

- There is almost no comparison with other ante-hoc or post-hoc XAI models. The paper mainly compares against interpretable models, but not against modern post-hoc explanation techniques that could provide competitive interpretability, while also achieving superior performance.

- The paper should explicitly discuss the trade-off between interpretability and performance, perhaps through a 2D evaluation showing both forecast error and explanation quality.

- It is unclear why models like TimeXer and ITrans are excluded from qualitative comparisons in Table 4. These baselines could offer meaningful insights into representational interpretability.

### Questions
Please refer to the weaknesses above.

### Soundness
2

### Presentation
4

### Contribution
3
