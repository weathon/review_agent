# Time series saliency maps: Explaining models across multiple domains

- Decision: Reject
- Scores: 2, 8, 2, 4

## Abstract
Traditional saliency map methods, popularized in computer vision, highlight individual points (pixels) of the input that contribute the most to the model's output. However, in time-series, they offer limited insights, as semantically meaningful features are often found in other domains. We introduce Cross-domain Integrated Gradients, a generalization of Integrated Gradients. Our method enables feature attributions in any domain that can be formulated as an invertible, differentiable transformation of the time domain. Crucially, our derivation extends the original Integrated Gradients into the complex domain, enabling frequency-based attributions. We provide the necessary theoretical guarantees, namely, path independence and completeness. Our approach reveals interpretable, problem-specific attributions that time-domain methods cannot capture in three real-world tasks: wearable sensor heart rate extraction, electroencephalography-based seizure detection, and zero-shot time-series forecasting. We release an open-source Tensorflow/PyTorch library to enable plug-and-play cross-domain explainability for time-series models. These results demonstrate the ability of cross-domain integrated gradients to provide semantically meaningful insights into time series models that are impossible to achieve with traditional time‑domain saliency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work is focused on the explainability of black-box model in the context of time series models. The key insight is that the semantically meaningful information might not always be found in the time domain, but in other domains such as the frequency domain. To address this limitation, a generalization of the well-known Integrated Gradients method is proposed such that explanation can be presented in different domains. The proposed methodology is analyzed and evaluated on 3 time series analysis tasks.

### Strengths
1. A clear idea that is well motivated. 

2. A through theoretical analysis of the proposed methodology.

3. A nicely written and well-structured manuscript.

### Weaknesses
1. The novelty is low. The idea of providing explanations in a a different domain is already established. The paper mentions the Virtual Inspection Layers of Vielhaben et al. [1], but do not include it as a baseline, even though [1] also evaluated Integrated Gradients with their Virtual Inspection Layers. Furthermore, several other works [2, 3] have presented methodology for providing explanations in other domains than the time domain.

2. The experimental evaluation is limited. The proposed methodology is tested on 3 datasets, but no baselines are provided, and a limited quantitative evaluation. Compared to existing works [1, 3], where numerous datasets are used and a wide range of explainability metrics are evaluated, the evaluation in this work does not give insights into the usefulness of the proposed method.

- [1] Vielhaben et al., Explainable AI for time series via Virtual Inspection Layers, Pattern Recognition, 2024
- [2] Brüsch et al., FreqRISE: Explaining time series using frequency masking, NLDL, 2025
- [3] Brüsch et al., FLEXtime: Filterbank Learning to Explain Time Series, Explainable Artificial Intelligence, 2025

### Questions
1. How does the proposed method quantitatively compare to [1, 2, 3] in terms of established explainablity metrics like faithfulness, localization, complexity, and robustness?

2. Apart from being specific for Integrated Gradients, how is the invertible transform introduced here to transfer between domains different from the transform introduced in [2]?

- [1] Vielhaben et al., Explainable AI for time series via Virtual Inspection Layers, Pattern Recognition 2024
- [2] Brüsch et al., FreqRISE: Explaining time series using frequency masking, NLDL, 2025
- [3] Brüsch et al., FLEXtime: Filterbank Learning to Explain Time Series, Explainable Artificial Intelligence , 2025??

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes the Cross domain Integrated Gradients method, which extends the Integrated Gradients (IG) to any reversible and differentiable transformation domain (including the complex domain), providing a more semantic and insightful explanation for time series models.

### Strengths
1. Wide universality: The attribution framework for unknown transformations has high universality and is not specific to any particular transformation.

2. Solid theoretical contribution: Extend the IG method to the complex field.

3. Compelling & Diverse Applications: This method shows significant application potential in multiple scenarios like medical and other general time series application.

### Weaknesses
Overall, I think the strengths of this paper are very prominent, and the disadvantages are not worth mentioning compared to it. Here are a few of my small concerns.

1. The main text of the paper lacks quantitative experimental comparisons, and the structure should be adjusted by moving the section in Appendix F to the main text.

2. Appendix D indicates that this method is the general form of [1]. Can the author compare the two in a visual form to see if the actual effect is consistent with the theory?

[1] Johanna Vielhaben, Sebastian Lapuschkin, Grégoire Montavon, and Wojciech Samek. Explainable ai
for time series via virtual inspection layers. Pattern Recognition, 150:110309, 2024.

### Questions
1. Since this method can be applied to all reversible transformations, is it suitable for the current popular flow generation models? What would be the computational burden in practical applications?

2. What are the errors of this method for differentiable irreversible transformations? Is it possible to make corrections?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper provides a novel explainability method for the time series domain. Specifically, the method is based on an extension of the popular saliency method Intergrated Gradients to incorporate multiple domains that can be derived through an invertible, differential transformation from the time domain. Importantly, the proposed method maintains the sensitivity and implementation invariance properties of IG.

### Strengths
- The paper proposes a saliency method for time series which does not solely focus on the time domain, but can also integrate latent features such as frequencies
- The presentation of the method is easy to follow
- The paper provides open access to the method in the form of a Python package

### Weaknesses
- The paper completely lacks references in the introduction. This is not in line with good research practice. It is unclear to the reader whether observations and statements are taken from the literature or are a novel contribution by the paper. Importantly, the observation that existing saliency maps fail in the 
time-series domain and that other features, e.g., stemming from the frequency domain, are not novel and have been shown before in the literature (e.g., [1],[2],[3]). This renders Proposition 1, without proper citations, almost plagiarism. Section 3.2 is therefore unnecessary. I am not raising an ethics flag at the moment, but this aspect is, in my opinion, sufficient for rejecting the paper without further evaluation.
- The paper does not discuss the many restrictions and limitations of IG and their equivalent part in the proposed method.
- The paper states that the method is applicable to many domains. However, all argumentation and derivation (e.g., section 4.2) solely focus on the frequency domain. More explanation and examples are needed here to evaluate the usefulness of the method.
- The method requires domain knowledge to specify the domain of interest. In practice, such knowledge might not exist, or if it does, might be limited. This can lead to dangerous misinterpretations of the explanations and wrong decision-making. It would be desirable that a novel explainability method can directly infer saliency across many (unspecified) domains to potentially uncover so far unknown important features, instead of suffering similar limitations to existing time series saliency methods on the time domain.
- The paper does not discuss the experimental results or the limitations of the proposed method. Overall, the paper seems unfinished.
- The experimental section only focuses on three examples with specific ML methods. Here, a model-agnostic evaluation would be beneficial.



[1] Schröder, Maresa, Alireza Zamanian, and Narges Ahmidi. "Post-hoc saliency methods fail to capture latent feature importance in time series data." International Workshop on Trustworthy Machine Learning for Healthcare. Cham: Springer Nature Switzerland, 2023.

[2] Schröder, Maresa, Alireza Zamanian, and Narges Ahmidi. "What about the Latent Space? The Need for Latent Feature Saliency Detection in Deep Time Series Classification." Machine Learning and Knowledge Extraction 5.2 (2023): 539-559.

[3] Theissler, Andreas, et al. "Explainable AI for time series classification: a review, taxonomy and research directions." IEEE Access 10 (2022): 100700-100724.

### Questions
- Section 4.2: How is the known failure mode of IG addressed in other domains besides the frequency domain?
- How are failure modes/limitations of IG addressed in the proposed method for general ML models (not only CNNs)?
- How can the method integrate multiple domains at the same time? Importantly, how can it detect + explain interactions between the domains?

### Soundness
2

### Presentation
2

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
The authors advance explainability for time series methods by innovating on the integrated gradient method from 2017. Their proposal is called "Cross-domain Integrated Gradients", based on the fact that their method works on any invertible transformation. The user of the XAI method can thereby choose a transformation of their choice based on which domain is most suitable for explanations. The authors demonstrate qualitative feasibility using the Fourier Transform, Independent Component Analysis, and Seasonal-Trend decomposition.

### Strengths
1. The paper tackles an important and timely problem of developing XAI methods for time series. This field is in its infancy and underdeveloped, with the majority of XAI methods being developed on image data.

1. The authors provide both a TensorFlow and a PyTorch open-source library for their method.

1. Strong mathematical foundation for their method. 

1. The method works, based on Figure 2-4, and is tested using three different transformations, in three different data domains.

### Weaknesses
As far as I can see, the paper has only one major weakness, which is its positioning in the existing state of the art. If the authors can address this, I will definitely consider changing my recommendation.

References to relevant prior literature on time series explainability are lacking in section 2, e.g, there are only two papers from 2024 in related work.

The experimental section is quite limited and is primarily qualitative. The only quantitative experiments I could find are in Appendix F, where the authors present insertion/deletion and compare that to time IG. There should be comparisons included with existing time-frequency analysis XAI methods. Comparisons should be made e.g to https://proceedings.mlr.press/v265/brusch25a.html, which also only assumes invertible transformations and therefore seems relevant.

### Questions
1. Equation 1:  $\xi$ can easily become negative; is that an issue?

1. Figure 1: Why is the peak of the orange distribution not located at 4 Hz, as specified by Eq. 1?

1. Equation 2: I'm a bit unsure about the notation in the integral with $x
+t(x-\hat{x})$, does this mean the partial derivative of $f$ is evaluated at that point, and then you integrate over $t$ ?

1. Line 352: $x(t) = a_1 \cos(2\pi · \xi_{hr}\cdot  t + \phi) + a2 \cos(2\cdot \pi(2\xi_{hr})\cdot  t + \phi))$. Small esthetic typo: should it be $2\pi$ without the cdot in the second cosine?

1. Figure 2: The blue dashed (as opposed to solid) lines make this figure more difficult to read, but I may be wrong.

1. Line 389: What does the $\rightarrow$ mean in this context?

### Soundness
2

### Presentation
3

### Contribution
3
