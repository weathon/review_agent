# PEDVLM: PEDESTRIAN VISION LANGUAGE MODEL FOR INTENTIONS PREDICTION

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 3

## Abstract
Effective modeling of human behavior is crucial for the safe and reliable coexistence of humans and autonomous vehicles. Traditional deep learning methods have limitations in capturing the complexities of pedestrian behavior, often relying on simplistic representations or indirect inference from visual cues, which hinders their explainability. To address this gap, we introduce $\textbf{PedVLM}$, a vision-language model that leverages multiple modalities (RGB images, optical flow, and text) to predict pedestrian intentions and also provide explainability for pedestrian behavior. PedVLM comprises a CLIP-based vision encoder and a text-to-text transfer transformer (T5) language model, which together extract and combine visual and text embeddings to predict pedestrian actions and enhance explainability. Furthermore, to complement our PedVLM model and further facilitate research, we also publicly release the corresponding dataset, PedPrompt, which includes the prompts in the Question-Answer (QA) template for pedestrian intention prediction.  PedVLM is evaluated on PedPrompt, JAAD, and PIE datasets demonstrates its efficacy compared to state-of-the-art methods. The dataset and code will be made available at {https://github.com/abc/ped_VLM}.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper addresses the lack of interpretability in existing deep learning-based methods for pedestrian intent prediction by proposing a multimodal approach that combines language models and visual modalities. This method enhances the interpretability of pedestrian intent prediction and achieves state-of-the-art performance on the JAAD dataset. Additionally, due to the absence of multimodal pedestrian intent prediction datasets, the authors have constructed the PedPrompt dataset, which integrates three previously popular pedestrian intent prediction datasets and provides new prompt annotations.

### Strengths
1. The authors accurately identify the current lack of interpretability in pedestrian intent prediction methods and address this issue by constructing a new dataset.
2. The provided dataset exhibits good data balance.
3. The proposed pedestrian intent prediction method effectively integrates multiple modalities, achieving excellent performance in a concise and efficient manner.
4. The authors make a clever adjustment to the loss function, resolving the model confusion caused by similar linguistic states for crossing and not crossing the road.

### Weaknesses
1. The authors do not provide a reasonable explanation for the significant performance drop of T5-Large+CLIP compared to T5-Base+CLIP in Table 1, which is confusing.
2. Despite the introduction of the PedPrompt dataset, there is a lack of various experiments on this dataset, including comparisons with other methods. Only statistical information about the dataset and one ablation study are provided.
3. Although the proposed method has high interpretability, its real-time performance is questionable. In practical applications, the method requires extracting optical flow and performing language model inference, both of which are time-consuming processes. The authors need to further demonstrate the feasibility of the method in real-world scenarios.
4. The authors lack comparison experiments with related methods on the TITAN dataset, which provides optical flow information. Such comparisons would better highlight the performance differences with methods that also use optical flow.
5. When explaining the poorer performance on the PIE dataset, the authors mention that PedVLM relies more on image context features but do not provide an ablation study without the text modality, making the conclusion less robust.

6.The intention prediction of this paper is limited to the scenario of crossing the road. However, in real-world settings, pedestrians may exhibit various intentions unrelated to crossing, which can still impact ego-vehicle planning, such as moving towards or away from vehicles, remaining stationary, or waiting. Expanding the range of intentions is recommended to enhance the model’s applicability and realism.

7.A comparison with other vision-language models, such as GPT-4V, is necessary to evaluate for the proposed model's performance.

8.A discussion and comparison of inference times between the proposed model and other language models should be included to provide a more comprehensive evaluation of performance.

9.The model's performance on the PIE dataset is significantly lower than other baseline models, indicating a high dependency on visual input and a lack of robustness and reasoning capabilities. 

10.There are some typo errors, such as “1^-9” and “1^-4” in the implementation details, should be corrected.

### Questions
Please see the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors argue that traditional deep learning methods fall short of capturing the complexity of pedestrian behavior, often lacking in explainability in the context of autonomous vehicles. To address this problem, they introduce PedVLM, a novel vision-language model designed to predict pedestrian intentions and provide interpretable explanations for those predictions in self-driving. This model integrates multiple modalities, including RGB images, optical flow, and text, to enhance the model's contextual understanding and predictive accuracy. Along with this model, the PedPrompt dataset is introduced based on existing datasets like JAAD, PIE, and TITAN, which consists of Question-Answer (QA) prompts for pedestrian intention prediction. PedVLM is evaluated on the PedPrompt dataset, which surpasses GPT-4V and other traditional methods in terms of F1-score and AUC on JAAD.

### Strengths
* This research proposes an effective VLM model, which integrates RGB for temporal understanding and optical flow for temporal perception.
* This work formulates one pedestrian intention prediction dataset, which includes 48,696 prompts. This is a significant contribution to the field of pedestrian intention prediction.

### Weaknesses
* **The lack of related work discussion**

     The discussion on VLMs for driving environment understanding is not enough. Especially recently, there have been lots of related works using VLMs to provide interpretable explanations for traffic conditions and decisions. Some representative examples are listed below. Including them and other relevant references will strengthen the contribution of the work.

      [1]Sima C, Renz K, Chitta K, et al. Drivelm: Driving with graph visual question answering. ECCV 2024.
      [2]Marcu A M, Chen L, Hünermann J, et al. Lingoqa: Video question answering for autonomous driving[J]. arXiv preprint arXiv:2312.14115, 2023. 
      [3]Malla S, Choi C, Dwivedi I, et al. Drama: Joint risk localization and captioning in driving. WACV 2023.
      [4]Chen L, Sinavski O, Hünermann J, et al. Driving with llms: Fusing object-level vector modality for explainable autonomous driving[C]//2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024.
      [5]Han W, Guo D, Xu C Z, et al. Dme-driver: Integrating human decision logic and 3d scene perception in autonomous driving[J]. arXiv preprint arXiv:2401.03641, 2024.

---

* **The limitation of the proposed PedPrompt**

   1. This dataset is specifically curated to focus on pedestrian intention prediction. However, real-life driving scenes include various conditions, like road layout, dynamic vehicles, and other obstacles [1]. The explanations provided by this dataset are very limited.
   2. PedPrompt only provides interpretable explanations for a single object within a video sequence. Regrettably, this object may not necessarily be a critical influence on driving decisions, potentially resulting in misleading driving attention and subsequent inaccuracies in predictive models.
   3. The explanation answer only includes one kind of action for pedestrians, that is, “crossing” or “not “crossing”. This simplistic categorization fails to capture the complexity of pedestrian behavior. For instance, a pedestrian may indeed be crossing the road but not directly impacting the ego vehicle's trajectory. Thus, a more detailed description to understand and predict intention could be more helpful for this benchmark.
    4. In this dataset, various prevalent driving datasets, such as BDD and nuScenes, have not been taken into account.

---
Overall, this work using VLM for pedestrian intention prediction misses an important literature review and provides a dataset with several key limitations.

### Questions
* How does the length of the input frame sequence affect the model's ultimate performance? Is the 5-frame input enough for pedestrian intention prediction?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes an augmented dataset denoted as PedPromt, which is based on existing TRANS dataset. Through mannually designed prompt template, this paper introduce a framework to predict pedestrian behavior and generate related explainations simultaneously. Compared with existing state-of-the-art VLM, the proposed model achieves better performanceon PIE and JAAD dataset.

### Strengths
1. This paper introduces augmented language annotations on existing TRANS dataset and make it suitable to generate explanations of pedestrian behavior.

2. The proposed model achieves better performance than state-of-the-art large VLMs like GPT-4V on PIE and JAAD dataset

### Weaknesses
1. Writing issues:  
 (a)  All figures are too blurry to be seen clearly.  
(b) In Section 4.2, the index 𝑖 of visual embeddings should not be used to represent the projection matrix 𝑃.

2. The contribution to the TRANS dataset is insufficient. While VLMs are a popular topic, many existing works augment datasets with language explanations. This paper’s template incorporates substantial information from the TRANS dataset but should further explore ways to expand upon this foundation.

3. The exploration of the prompt template is inadequate and does not ensure that the proposed template is optimal. Current LLMs are not sensitive to numerical data, which could lead to issues when numbers are directly translated to strings as input.

4. The contribution of the proposed method is limited. The paper mainly focuses on merging different modalities using simple modules. The authors should place greater emphasis on ensuring effective feature alignment between modalities.

### Questions
The quality of the augmented dataset should be addressed. The hallucination from LLM should be discussed.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper presents PedVLM, a novel Vision-Language Model (VLM) designed to predict pedestrian intentions in urban environments and improve the safety of autonomous vehicles. Traditional deep learning methods have struggled with accurately capturing the complexities of pedestrian behavior and often lack explainability. To address this, PedVLM combines visual data (RGB images and optical flow) with textual descriptions using a CLIP-based vision encoder and a T5 language model. This integration allows the model to predict pedestrian actions, such as crossing or not crossing while providing interpretable explanations for its decisions.
A key contribution of the work is the development of the PedPrompt dataset, which includes question-answer style prompts tailored for pedestrian intention prediction. The authors evaluate PedVLM on various datasets, including PedPrompt, JAAD, and PIE, showing that it outperforms state-of-the-art models, particularly in terms of F1-score and AUC for pedestrian intention prediction. However, the study also notes performance challenges with pedestrians who are distant or partially occluded, highlighting areas for future improvement.

### Strengths
- The paper introduces a comprehensive dataset, PedPrompt, which supports further research in pedestrian intention prediction. This dataset is meticulously curated, including a variety of contextual and pedestrian behavior information
- Unlike many deep learning models that act as black boxes, PedVLM emphasizes explainability. By using language models to provide reasoning for predictions, the model enhances interpretability, which is crucial for autonomous driving applications.

### Weaknesses
- The model's performance significantly depends on the visibility and clarity of pedestrian cues. As noted in the evaluation, PedVLM struggles with distant or partially occluded pedestrians, particularly in the PIE dataset, which limits its robustness in complex urban environments.
- The model simplifies pedestrian intention to crossing versus non-crossing (Simplified Binary Classification), which may not capture the full complexity of pedestrian behavior, such as hesitation or erratic movements, limiting its applicability in real-world situations.
- The paper does not convincingly justify the necessity of using Vision-Language Models (VLMs) for pedestrian intention prediction. The explanations generated by PedVLM, such as “The pedestrian is making no hand gesture, is not nodding, and is looking toward the ego vehicle,” seem to reiterate the attributes explicitly mentioned in the input prompts. This raises concerns about whether the model is genuinely interpreting and understanding visual cues from the images or simply parroting back the information provided in textual form, rather than making independent inferences based on visual and contextual analysis.

### Questions
- Since pedestrian intention prediction here is a simple binary task (crossing or not crossing), why use a complex language model like T5? Wouldn’t a simpler model work just as well, or is there a specific benefit to using T5 that makes it worth the added complexity?
- The explanations generated by PedVLM often seem to mirror the attributes provided in the prompts, raising concerns about the model’s interpretive abilities. How can you demonstrate that the model is not just repeating input information but is genuinely understanding and interpreting visual cues?
- How would PedVLM handle real-world conditions, such as complex pedestrian behaviors or environments with limited visibility and frequent occlusions? Have you considered evaluating the model under more challenging or diverse conditions?

### Soundness
2

### Presentation
1

### Contribution
1
