# To Trust Or Not To Trust Your Vision-Language Model's Prediction

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 6, 6

## Abstract
Vision-Language Models (VLMs) have demonstrated strong capabilities in aligning visual and textual modalities, enabling a wide range of applications in multimodal understanding and generation. While they excel in zero-shot and transfer learning scenarios, VLMs remain susceptible to misclassification, often yielding confident yet incorrect predictions. This limitation poses a significant risk in safety-critical domains, where erroneous predictions can lead to severe consequences. In this work, we introduce TrustVLM, a training-free framework designed to address the critical challenge of estimating when VLM’s predictions can be trusted. Motivated by the observed modality gap in VLMs and the insight that certain concepts are more distinctly represented in the image embedding space, we propose a novel confidence-scoring function that leverages this space to improve misclassification detection. We rigorously evaluate our approach across 17 diverse datasets, employing 4 architectures and 2 VLMs, and demonstrate state-of-the-art performance, with improvements of up to 51.87% in AURC, 9.14% in AUROC, and 32.42% in FPR95 compared to existing baselines. By improving the reliability of the model without requiring retraining, TrustVLM paves the way for safer deployment of VLMs in real-world applications. The code is available in Supplementary Material.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces TrustVLM, a framework designed to address confidence estimation in vision-language model (VLM) predictions. Motivated by the modality gap inherent in VLMs, TrustVLM constructs ensembles of VLMs and image-only classifiers to enhance the detection of misclassified samples.

### Strengths
The presentation of this paper is clear and easy to follow.

### Weaknesses
The methodology proposed in this paper employs N-sample training data together with external image encoders to construct a Nearest Class Mean classifier, which is then combined with the original CLIP classifier. However, the improvement appears to stem primarily from the use of additional labeled data and model ensembling. I am concerned that this may not constitute a genuinely novel contribution. Moreover, since the competitive baselines operate in a zero-shot setting, the comparison could be considered unfair.

### Questions
How does TrustVLM perform with fewer training samples, such as in 1-shot or 2-shot settings? Can TrustVLM function with little or even no training data?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies VLM error detection to enhance its trustworthiness. The proposed TrustVLM leverages the multimodal similarity of feature representations to decide whether the current prediction is correct. Particularly, it compares the query example with other examples in both image representation similarity and text representation similarity. The final confidence score for error detection is computed by combining both similarities, thus incorporating multimodal information. Through extensive experiments of quantitative analysis and qualitative study, the effectiveness of the TrustVLM has been superior to most of the baseline methods.

### Strengths
- This paper is easy to follow, the motivation is very clear, and the intuition is quite straightforward.
- The proposed TrustVLM is training-free and efficient to deploy. It can also be easily adopted by any VLM architectures.
- The experimental performance is quite promising.

### Weaknesses
- The major concern is missing the comparison with unimodal detection methods. The proposed method combines multimodal information to detect prediction errors; however, in the ablation study, there is no comparison with image-only or text-only detection. In this way, it would be clearer which branch of modality would contribute more to the overall performance improvement.
- Moreover, the performance of TrustVLM highly relies on the performance of the employed VLMs; if the VLMs cannot provide high-quality representations, the error detection would be limited.
- Another concern is that due to the existence of a modality gap, the cross-modal similarity could be unstable compared to image-to-image similarity. The misaligned cross-modal pairs would also mislead the error detection.
- After detection, the proposed TrustVLM cannot further rectify the error predictions by finding the correct one.

### Questions
- Which branch of modality contributes more to the overall performance improvement? It would be helpful to conduct an ablation study to verify the unimodal detection versus multimodal detection. Moreover, what if we directly ask LLMs to detect the prediction error? As done in ``Machine Vision Therapy: Multimodal Large Language Models Can Enhance Visual Robustness via Denoising In-Context Learning, in ICML 2024'', they leverage the prediction of LLMs to find prediction errors of VLMs, and LLMs can further correct the prediction to find the correct one.
- How would the misaligned multimodal pairs mislead the overall detection performance?
- The acquisition of prototypes could be difficult sometimes. Can the proposed method perform without prototypes?

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
In this work, the authors introduce TrustVLM, a training-free framework for predicting when a VLM’s predictions can be trusted. The task explored in this work is misclassification detection, which involves identifying when a prediction is incorrect. The key idea is to (1) generate visual prototypes for each class (i.e. the average embedding for N samples from the training data) and (2) compute the image-to-image similarity between the query image and the class prototypes. The standard image-to-text cosine similarity and the computed image-to-image similarity scores are combined in order to determine the overall prediction confidence. The authors show that this approach leads to substantially better misclassification detection performance than baselines.

### Strengths
- This work addresses an important task: determining when the predictions of a VLM are likely to be reliable.
- Although the proposed method is methodologically straightforward, strong performance is observed across a range of datasets and model backbones. The authors also compare with multiple baselines. The distribution shift experiments with ImageNet are particularly compelling.

### Weaknesses
- **Need for finer-grained analysis:** This paper could benefit from additional fine-grained analysis with respect to when the proposed method is most effective (rather than just overall metrics). For example, are there specific classes where misclassification detection performance improves substantially when using the proposed method (as compared to MSP)? What types of characteristics are common among those classes?
- **Variance of performance:** The proposed method is likely very sensitive to the choice of few-shot samples used to compose the class prototypes. What is the variance in performance when using prototypes composed from different randomly-selected N-shot sample sets?

### Questions
Questions are listed above under weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
For improving the reliability of VLM, this paper proposes a new training-free framework TrustVLM to estimate when VLM’s prediction will be trusted. The key of TrustVLM is to use a new confidence scoring function which will use not only cosine similarity between image and text but also the addition information from image embedding space. This additional information is the image-to-image relations or similarity. The idea is very straightforward where you need to extract some prior knowledge from training data. This prior knowledge is the class embeddings that are extracted from N shot examples from training data for each class. It likes you did another way for image classification based on the similarity between class embeddings and input image embedding. The whole idea is clear. The paper is well written. The key question here is you need to have a training data and get the class embedding based on the classes in the training data. But in some use cases, we don’t have the training data to get these class embeddings in advance. The proposed method has its limitation on it.

### Strengths
1.	TrustVLM is a training-free framework designed to evaluate the reliability of VLM predictions. One of its key advantages is that it does not require additional training, which makes it convenient to apply in scenarios where labeled data is limited or unavailable. The framework combines both image-to-text and image-to-image similarities, which allows for a more robust and nuanced design of confidence scores. This combination provides a richer representation of the visual information, enabling the framework to better capture the model’s uncertainty.

2.	The paper demonstrates that the proposed visual prototypes not only enable more reliable confidence estimation but also enhance fine-grained classification accuracy.

3.	The experiments conducted across diverse datasets, model architectures, and VLMs show the generality and effectiveness of TrustVLM.

### Weaknesses
1.	A notable limitation of this method is that it relies on the availability of in-domain data that includes images for all classes to be predicted. Under this assumption, the method can extract and store visual prototypes for each class, which are then used for confidence estimation. However, in many practical scenarios, obtaining such in-domain data for every class may be difficult or infeasible. Moreover, if the training or reference data does not fully cover the diversity of the test data, the method may encounter out-of-distribution (OOD) situations. As the introduction clearly states, TrustVLM is not designed to handle OOD cases, which inherently limits its applicability in environments where data coverage is incomplete or classes are highly dynamic. This restriction should be carefully considered when evaluating the practical utility of the method.

2.	There exist alternative strategies to improve reliability when prior knowledge about class representations is available. For instance, one could employ a separate image embedding model to independently obtain embeddings for the input and for each class, then compute similarity scores between them. Another potential approach is to first predict the class of the image using a preliminary classifier and then use this prediction as an input for the final classification task. These alternatives might offer comparable or complementary benefits to TrustVLM. Therefore, the paper would be strengthened if the authors could provide a more detailed comparison or discussion of how TrustVLM differs from these approaches. Specifically, it would be helpful to clarify the unique advantages of TrustVLM, such as why its combined use of image-to-text and image-to-image similarity provides superior or more reliable confidence estimation compared to these other methods. This explanation would help to more clearly establish the method’s contributions and practical significance.

### Questions
Please check the weaknesses.

### Soundness
3

### Presentation
4

### Contribution
2
