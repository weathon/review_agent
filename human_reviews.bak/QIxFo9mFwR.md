# Scale-aware Recognition in Satellite Images under Resource Constraints

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Recognition of features in satellite imagery (forests, swimming pools, etc.) depends strongly on the spatial scale of the concept and therefore the resolution of the images. This poses two challenges:
Which resolution is best suited for recognizing a given concept, and where and when should the costlier higher-resolution (HR) imagery be acquired? 
We present a novel scheme to address these challenges by introducing three components: (1) A technique to  distill knowledge from models trained on HR imagery to recognition models that operate on imagery of lower resolution (LR), (2) a sampling strategy for HR imagery based on model disagreement, and (3) an LLM-based approach for inferring concept "scale". With these components we present a system to efficiently perform scale-aware recognition in satellite imagery, improving accuracy over single-scale inference while following budget constraints. **Our novel approach offers up to a 26.3\% improvement over entirely HR baselines, using 76.3 \% fewer HR images.** Resources are available at https://www.cs.cornell.edu/~revankar/scale_aware.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
A scene classification task under resource-constrained conditions is proposed, combining image retrieval and scene classification. Knowledge distillation techniques and an active learning-like sample selection strategy are used to choose HR samples, and an LLM is employed to determine the scale of concepts.

### Strengths
The authors address the real-world issue where high-resolution (HR) remote sensing images are paid, while low-resolution (LR) images are free, by proposing a framework that effectively improves scene classification performance under limited funding. This framework can be used to analyze which high-resolution images are more beneficial to purchase, helping researchers make informed decisions,

### Weaknesses
1. The description of the LLM section is not very clear. The authors did not specify which LLM was used, how the hyperparameters were set, or provide comparative experiments on the performance of different LLMs. This weakens the contribution of the LLM. It seems that the authors are merely using it as a tool for invocation.

2. In Section 4.2.3, the discussion of Figure 3, which uses only five images to demonstrate ranking similarity, is not convincing. The authors need to provide a quantitative analysis to support this argument.

### Questions
1. Does Equation 5 require knowledge of all the information for all seen categories at a specific location to be computed? Why are the HR score and LR score related to location? Does this limit the model to only locations that appeared in the training set? How would the model handle unknown locations?

2. Is using LLM-based in-context learning to determine the scale reliable? Has the accuracy been tested on unseen categories in the validation set? Since the context is determined by the HR and LR scores, why not directly train a network using the HR and LR scores? It is well known that LLMs are very resource-intensive, and using an LLM just to determine the scale seems like overkill. Furthermore, it contradicts the premise of resource efficiency—if one cannot afford HR images, they likely don't have the computational resources to run an LLM either.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the challenge of feature recognition in satellite imagery, focusing on the optimal use of image resolution to balance accuracy and resource constraints. The proposed system introduces three key components: LLM-based scale inference, knowledge distillation, model disagreement-based sampling. The proposed approach has been demonstrated to improve recognition accuracy while reducing the need for HR imagery.

### Strengths
This paper focuses on a highly interesting problem with significant industrial application potential. 

The proposed knowledge distillation method and model disagreement-based sampling strategy are both innovative and effective. The distillation enhances the performance of LR models, while the sampling strategy efficiently balances the trade-off between cost and accuracy.

### Weaknesses
The approach heavily relies on the examples provided in the prompt for the LLM. The model’s understanding of the relationship between manually defined categories and image resolutions is largely based on these examples. This dependence raises concerns about the generalizability of the method to niche or less common satellite imagery concepts.

The paper does not include experiments comparing the proposed method with techniques that generate HR images via super-resolution, then combine these images with LR data for recognition. Such comparisons, focusing on both computational efficiency and accuracy, would provide a more comprehensive evaluation of the proposed approach.

### Questions
The paper distills a model from paired HR and LR images to predict whether HR images are necessary based on prediction differences. Why not directly train a super-resolution model on paired HR-LR images, use it to generate HR images for any LR input, and then perform recognition on both the generated HR and original LR images? If the recognition results differ significantly, HR images could be invoked. What are the advantages of the proposed, more complex method over this approach?

Impact of resolution ratio on the proposed method is suggested to be discussed. Does the resolution ratio between high and low-resolution images affect the performance of the proposed method? Could you please provide some analysis or experiments to explore how this ratio influences the model’s efficiency and accuracy?

### Soundness
3

### Presentation
2

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
This paper introduced a new approach to scale-aware recognition in satellite imagery under resource constraints. The proposed system leverages LLMs and knowledge distillation technique. The proposed system outperformed the baselines  more than 26% mAP in zero-shot techniques, and more than 8% mAP in supervised techniques using 5× fewer HR images.

### Strengths
1. This paper presents a novel framework to perform the tradeoff between cost and accuracy. 
2. The topic is interesting and valuable. 
3. The writing is clear. 
4. It is good that the authors plan to release their data and download scripts.

### Weaknesses
1. It is unclear how the system handles noisy labels using OpenStreetMap annotations. 
2. Can the proposed system retrieve multiple targets simultaneously? 
3. How to deal with the time discrepancy between high- and low-resolution images, which can lead to disagreements between the LR and HR models. Did the authors consider the impact of this issue? 
4. No mention of runtime in Table 3.

### Questions
The system is based on many ideal assumptions. Whether deployed online or offline, it will encounter foreseeable issues such as data storage and retrieval, online search, and download stability. Additionally, the overall process time and reliability are concerns. Although I like the idea, there seems to be a large gap between the proposed method and practical application.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
To provide a holistic treatment of scale to achieve accurate recognition under a budget, the paper tries to give solutions to which resolution is the best and where and when  the costlier higher-resolution (HR)imagery should be acquired. The main contributions are: (1)knowledge distillation is adopted to allow Low resolution models to improve significantly in  recognizing finer concepts by learning from High resolution ones. (2) Semantic understanding of LLM is used to determine the  scale of each concept.(3) A sampling strategy for HR imagery based on model disagreement is proposed. The process is frameworked as remote sensing image retrieval task. Experiments show some promising success for the proposed idea.
The idea is very novel and interesting.

### Strengths
1. A novel idea of scale determination is proposed for targer recognition using reasonable resolution remote sensing images.
2. Low resolution detection model using distillation strategy is taken as the replacement of high resolution detection model.
3. Two benchmark datasets are constructed for research.

### Weaknesses
1. The budget constraints seems actually impose no impact on the final result.
2. The processes such as distillation, concept scale LLM training are not provided the details.
3. The retrieval process has no systematic description to include all the subprocesses.

### Questions
1.Quote" A location l with higher δ(l) is more likely to have concepts that MHR could detect where MLR  could not (or vice versa)." Since it is the absolute value, it is difficult to understand.
2. How is the LLM trained? What's the confidence for the determination of LLM?
3. Can it be understood as that the low resolution images taken as large as possible because the cost constraints is not imposed?
4. Quote"...Remote CLIP(Liuetal.,2024) in Table 3." But the table is not the detection result.

### Soundness
3

### Presentation
3

### Contribution
4
