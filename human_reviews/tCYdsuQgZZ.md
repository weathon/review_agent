# Test-time Contrastive Concepts for Open-World Semantic Segmentation

- Decision: Reject
- Scores: 6, 5, 6

## Abstract
Recent CLIP-like Vision-Language Models (VLMs), pre-trained on large amounts of image-text pairs to align both modalities with a simple contrastive objective, have paved the way to open-vocabulary semantic segmentation.  Given an arbitrary set of textual queries, image pixels are assigned the closest query in feature space. However, this works well when a user exhaustively lists all possible visual concepts in an image, which contrast against each other for the assignment. This corresponds to the current evaluation setup in the literature which relies on having access to a list of in-domain relevant concepts, typically classes of a benchmark dataset. Here, we consider the more challenging (and realistic) scenario of segmenting a single concept, given a textual prompt and nothing else. To achieve good results, besides contrasting with the generic ``background'' text, we propose two different approaches to automatically generate, at test time, textual contrastive concepts that are query-specific. We do so by leveraging the distribution of text in the VLM's training set or crafted LLM prompts. We also propose a metric designed to evaluate this scenario and show the relevance of our approach on commonly used datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This study identified a significant issue with existing OVSS methods, which is the necessity of providing an exhaustive list of all potential visual concepts for an image to be segmented. To address this, the authors proposed a more practical scenario where only a single concept is segmented based solely on a textual prompt. To improve performance, they introduced two methods for expanding contrastive concepts: co-occurrence-based and LLM-based. Experimental results across multiple datasets demonstrate the effectiveness of their proposed methods.

### Strengths
1. Promising results with extensive experiments.
2. A new problem in OVSS has been identified, and a new setting of open-world open-vocabulary semantic segmentation is posed.
3. The method is simple to implement.

### Weaknesses
1. The problem addressed in the study lacks in-depth analysis. While the authors mention that CLIP's feature space is not easily separable, citing Miller et al.'s work, it should be noted that Miller et al.'s focus was primarily on classification and object detection tasks. Therefore, it is uncertain whether their conclusions are applicable to the image segmentation task. To provide a more comprehensive analysis, the authors should delve deeper into the feature space for the image segmentation task, comparing scenarios with and without the use of contrastive concepts.
2. The straightforward descriptions of the proposed method make it challenging to comprehend. To enhance clarity, the authors should incorporate specific examples or a conceptual diagram to illustrate their approach.
3. Comparing the proposed method directly to baselines like CAT-Seg may not be entirely fair. These baselines could potentially be enhanced through simple modifications, such as randomly selecting a subset of class names for each image during the training phase. Such adjustments could make them more suitable for the proposed setting, which involves a single query. Therefore, it would be prudent to consider such potential improvements before making direct comparisons.
4. The comparison to the classical method SAN is missing: Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, and Xiang Bai. Side adapter network for open-vocabulary semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2945–2954, 2023
5. To demonstrate the effectiveness of selecting/generating contrastive concepts for each query, the following two experiments are suggested to add:
    * use the full set of textual concepts $\mathcal{T}$
    * use LLM to generate many concepts (say 500) for a certain dataset
At both the training and test stages, the full set of concepts can be suplied.

### Questions
1. In what scenarios will the LLM-based method outperform the co-occurrence-based method? Additionally, how can one determine which method is optimal for a particular dataset?
2. Is the proposed method also suitable for multi-domain evaluation, similar to CAT-Seg?
3. Please provide some examples where the proposed method fails.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper introduces a method to open-world semantic segmentation using test-time contrastive concepts (CC), addressing the challenge of segmenting specific objects in images with minimal prior information. Current Vision-Language Models (VLMs), like CLIP, assign each pixel to one of many known visual concepts. However, this method struggles when only one or a few objects are specified, especially if relevant context (e.g., related background objects) is unknown. To solve this, the authors propose generating contrastive concepts at test time, using either VLM training data or large language models (LLMs), to improve segmentation without needing extensive dataset-specific knowledge. This approach represents a step towards effective open-world segmentation by dynamically enhancing contrast without altering VLM structures or needing extensive dataset-specific adaptations. Authors evaluated the model performance on various popular benchmarks and observed consistent performance gains.

### Strengths
The most interesting part of this paper, in my opinion, is its deep dive into how current open-vocabulary segmentation models still lean on fixed, closed-set categories during evaluation. This is a real issue if we want these models to work effectively in real-world settings, where they’ll face plenty of unknown objects and only the object of interest is provided in the prompt. 

Beside that, the paper brings some interesting contributions to the table:

1. [Test-Time Contrastive Concept Generation]: The authors introduce a fresh approach that automatically creates contrastive concepts during testing to boost segmentation in open-world scenarios. This context-specific contrast helps separate objects better without needing pre-set information.

2. [Background Handling Analysis]: They also dig into the common practice of treating "background" as a one-size-fits-all contrastive concept, showing its drawbacks and making a strong case for using concepts tailored to each query for more accurate results.

3. [New Metric for Open-World Segmentation]: To better measure performance, they propose IoU-single, a metric that assesses each concept’s segmentation independently, giving a clearer view of how well models handle open-world data.

4. [Extensive Benchmarking]: Lastly, they put their method to the test across various datasets, showing it consistently beats current open-vocabulary segmentation models, proving it’s robust and ready for broader applications.

### Weaknesses
1. [Problem setting] As noted in the strengths, evaluating the model's performance using only a textual prompt is a valuable contribution and can address issues faced by many existing models. However, this approach overlaps significantly with the objectives of referring segmentation and expression localization models, which aim to segment or localize objects based on a text prompt. It would be beneficial to include comparisons with existing referring segmentation or detection models to provide a clearer benchmark. While this was briefly acknowledged in the related work section, I didn't find any quantitative or qualitative comparisons. Additionally, it would be necessary to compare with works like [1], which also supports segmentation/detection of multiple instances using a single text prompt.

2. [Re-define some existing concepts] This paper labels its setting as "open-world," a term already widely used in publications, such as [2]. Redefining this established concept might lead to confusion. It may help to clarify or differentiate this setting further to avoid conflicting terminology.

3. [Paper presentation] Some tables are unclear and difficult to interpret without the context provided in the paper. For example, Table 1 lacks a clear baseline for comparison—it appears to only show different CC variants without a standard baseline to measure the gains introduced by the proposed CC method. After reading the experimental section, I understood that CC^{BG} is the baseline, but its labeling as a variant rather than a baseline is confusing. The evaluation metric is also missing from the caption, which adds to the ambiguity.


Minor Issues: There are some minor typos and grammatical errors. For example, on line 53, "without any a priori knowledge" should be "without any prior knowledge."

[1] Liu, Chang, Henghui Ding, and Xudong Jiang. "Gres: Generalized referring expression segmentation." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 23592-23601. 2023.

[2] Sodano, Matteo, Federico Magistri, Lucas Nunes, Jens Behley, and Cyrill Stachniss. "Open-World Semantic Segmentation Including Class Similarity." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3184-3194. 2024.

### Questions
Compared to the results of CC^{L}, the performance gains achieved by using CC^{D} appear minimal. Could this imply that simple language model-generated concepts might be sufficient for open-world segmentation? For example, in Table 1, CC^{L} can get even better results than CC^{D} on cityscapes. If this is true, then it seems like that we only need to have "hard-negative" mining to make the open-set segmentation model work?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper discussed the 'background' concept in open-world semantic segmentation and proposed a method to suppress background with contrastive concepts when segmenting a specific category in test time. In addition, a new single-query evaluation setup for open-world semantic segmentation that does not rely on any domain knowledge is proposed, and a new metric is designed to evaluate the grounding of visual concepts. I think the "background" is confusing in segmentation but is not well-discussed and well-defined. It is valuable to study this problem.

### Strengths
1. Automatic discovery of comparison concepts is valuable because artificially defining comparison concepts requires professional knowledge and a manual preset knowledge base in advance.
2. I agree that the concept of "background" is confusing in segmentation but is not well discussed. Therefore, the article's exploration of "background" is valuable. This paper discussed the CLIP-based method, in which the pixel is scored by cosine similarity, and the threshold filters the background. However, thresholds are difficult to set and often overfit specific datasets, which is a real pain point.
3. Experiments on multiple datasets and verification of multiple methods prove the effectiveness of the proposed method.

### Weaknesses
1. When using LLM for contrastive concepts extraction, do you consider generating corresponding contrastive concepts for each image (such as inputting the image along with the category name), and whether this is better than just asking with text? (For example, if you type a boat and LLM will tell you water, person, beach), but if a boat is stranded on the beach and you enter the image together, LLM(or VLM) will only output 'beach', which is more targeted when you divide it later.
2. How will this method be handled when the same pixels (or region) belong to the whole and parts (such as person and arm) at the same time? Because for ‘arm’, the rest region of the ’person‘ is the background. Can you give an example to describe the process briefly?
2. Can the background class be circumvented if the similarity is calculated using ‘sigmoid’ instead of ‘softmax’? You can have a try.

### Questions
See weakness part.

### Soundness
3

### Presentation
3

### Contribution
3
