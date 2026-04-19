# MMR: A Large-scale Benchmark Dataset for Multi-target and Multi-granularity Reasoning Segmentation

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
The fusion of Large Language Models (LLMs) with vision models is pioneering new possibilities in user-interactive vision-language tasks. A notable application is reasoning segmentation, where models generate pixel-level segmentation masks by comprehending implicit meanings in human instructions. However, seamless human-AI interaction demands more than just object-level recognition; it requires understanding both objects and the functions of their detailed parts, particularly in multi-target scenarios. For example, when instructing a robot to \textit{“turn on the TV"}, there could be various ways to accomplish this command. Recognizing multiple objects capable of turning on the TV, such as the TV itself or a remote control (multi-target), provides more flexible options and aids in finding the optimized scenario. Furthermore, understanding specific parts of these objects, like the TV's button or the remote's button (part-level), is important for completing the action. Unfortunately, current reasoning segmentation datasets predominantly focus on a single target object-level reasoning, which limits the detailed recognition of an object's parts in multi-target contexts. To address this gap, we construct a large-scale dataset called Multi-target and Multi-granularity Reasoning (MMR). MMR comprises 194K complex and implicit instructions that consider multi-target, object-level, and part-level aspects, based on pre-existing image-mask sets. This dataset supports diverse and context-aware interactions by hierarchically providing object and part information. Moreover, we propose a straightforward yet effective framework for multi-target, object-level, and part-level reasoning segmentation. Experimental results on MMR show that the proposed method can reason effectively in multi-target and multi-granularity scenarios, while the existing reasoning segmentation model still has room for improvement. The dataset is available at \url{https://github.com/jdg900/MMR}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a novel dataset, MMR, the first part-level dataset for the reasoning segmentation tasks. In addition, a new network framework is proposed to leverage low-level fine-grained information and to address the limitation of the existing LISA model, which can only segment a single object. The authors conduct experiments to evaluate the performance of existing methods on the proposed MMR dataset and demonstrate the advantages of the proposed network framework.

### Strengths
1. This paper is well written and easy to follow. 
2. The proposed MMR dataset is highly valuable to the research community, as part-level reasoning segmentation is crucial in real-world applications, such as robotic control. However, there is currently a lack of available datasets for research in this area.
3. A detailed analysis is provided to thoroughly present the characteristics of the MMR dataset.

### Weaknesses
1. The contributions of the proposed M2SA network framework are incremental. The early local feature fusion appears to be only a minor structural modification. Additionally, the strategy of employing multiple [SEG] tokens has already been introduced in earlier methods, such as [a]. The authors should clarify the differences between their approach and [a].
2. This paper could benefit from more thorough experiments based on the characteristics of the dataset. For instance, does the M2SA trained on the MMR dataset show a noticeable long-tail phenomenon, i.e., better performance on the more frequently occurring object and part categories as presented in Fig.3? Additionally, what is the model’s open-vocabulary performance on categories that do not appear in the training set?
3. More examples of the image-question-answer triplet in the MMR dataset could be presented in the paper to enable readers to understand the characteristics of the dataset more quickly and intuitively.

[a] GSVA: Generalized Segmentation via Multimodal Large Language Models, CVPR2024

### Questions
1. Why did you remove the generated questions that contain explicit target coordinates or strong hints? I think training with such data would enhance the model’s ability to handle target-specific inputs. For example, if an image contains two different animals, a fish and a cat, users could indicate the coordinates of the animal they are interested in and ask, “Which part of this animal [coordinates] uses its sense of smell?” The model could then segment either the nose or the fish’s gills depending on the coordinates provided. This could be quite interesting.

### Soundness
3

### Presentation
4

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
This paper introduces a dataset named MMR, designed for multi-target and multi-granularity reasoning segmentation tasks. The goal is to address challenges in reasoning across multiple targets and different levels of granularity. The dataset comprises complex and implicit question pairs, covering both object-level and part-level reasoning. Additionally, the paper proposes a baseline model, M2SA, to achieve multi-target, object-level, and part-level reasoning segmentation.

### Strengths
1. **Clear Writing**: The paper is well-organized and easy to understand.

2. **Significant Contribution of the Dataset**: The MMR dataset contains 196K samples. Although it was generated using large models, a rigorous filtering process was employed to ensure data quality.

### Weaknesses
1. **Lack of Targeted Design in the Baseline Model**: The baseline model (Early Local Feature Fusion and Multi [SEG] Tokens) does not incorporate specific structures to effectively address the multi-target and part-level reasoning required by the MMR dataset. As a result, it lacks novelty, leading to underwhelming performance in Table 3.

2. **Limited Performance in Table 3**: The comparison methods in Table 3 are not sufficiently recent. The authors did not include comparisons with more relevant multi-target approaches, such as GSVA [1] or GLaMM [2]. This limits the impact of the proposed approach, as the results are not very competitive.

3. **Insufficient Comparisons in Table 2**: The methods compared in Table 2 are too limited. I strongly recommend including more methods that could be adapted for the MMR task to facilitate meaningful comparisons for future research.

### Questions
1. Will the proposed dataset be made publicly available?
2. The paper mentions the use of 4 A6000 GPUs. How long does it take to train the proposed model on the MMR dataset?

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
This paper provides a large multi-target and multi-granularity reasoning segmentation benchmark. Based on this benchmark, this paper designs a baseline model trained on it while evaluating public datasets to present the effectiveness of both the benchmarks and the baseline. Experiments demonstrate that the proposed baseline outperforms LISA and other representative approaches.

### Strengths
1. The distinguishing characteristic of the proposed benchmark is clear, which includes multi-granularity and more images.
2. Multi-target and multi-granularity reasoning segmentation is a valuable research topic.
3. The overall writing is fluent.

### Weaknesses
1. This paper provides few comparisons on the proposed benchmark. It is not clear whether the proposed baseline model outperforms other MLLMs on multi-target and multi-granularity reasoning segmentation.
2. The major contribution lies in the benchmark, while this benchmark is auto-annotated based on the existing dataset PACO-LVIS, which hurts the contribution.
3. According to Table 1, MMR offers both object-level and multi-target annotations, making it more comprehensive than ReasonSeg and MUSE. This paper could include zero-shot evaluations on these two benchmarks to further demonstrate effectiveness.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
