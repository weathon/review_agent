# Benchmarking Zero-Shot Recognition with Vision-Language Models: Challenges on Granularity and Specificity

- Decision: Reject
- Scores: 8, 3, 3, 6, 6

## Abstract
This paper introduces innovative benchmarks to evaluate Vision-Language Models (VLMs) in real-world zero-shot recognition tasks, focusing on the pivotal properties of granularity and specificity. We propose a unique evaluation protocol using adapted ImageNet and MS-COCO datasets to assess models' consistency in recognizing concepts at varying granularity levels and their sensitivity to the specificity of language inputs. Our extensive evaluation reveals that state-of-the-art VLMs, including contrastive models like CLIP, struggle with granularity and are sensitive to text specificity, impacting their effectiveness in open-world settings. This comprehensive study, a first in evaluating VLMs from these perspectives, provides valuable insights and tools for the community, highlighting the limitations and paving the way for enhanced models with better generalization in zero-shot recognition. Our benchmark will be open-sourced upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper focuses on exploring the zero-shot recognition of Vision-Language Models (VLMs) in terms of Granularity and Specificity. Driven by this objective, this paper provides a benchmark to test the behavior of VLMs on different levels of granularity. 

The main two observations are  1) VLMs cannot correctly classify images from both fine-grained and coarse-grained perspectives. They are better at moderately fine-grained concepts.  2) In the meantime, this work also shows VLMs cannot detect whether the text correctly describes the given image. Instead, they could be sensitive to the specificity of text and struggle to distinguish between the challenging positives like single-label prompts, and hard negatives like poisoned captions with small changes.

### Strengths
+ [*The motivation is sound and convincing*] I like the idea of studying the VLM from granularity and specificity. For each perspective, this work provides a corresponding benchmark for measuring the performance. Such a study can provide insights into the zero-shot performance of VLM. 

+ [The evaluation way is interesting and reasonable] The evaluation protocol compares a) the prompt of the coarse-grained class and b) aggregates the predictions from prompts of its fine-grained children classes. This is intuitive and makes sense to me. From such protocol, this work draws the claim that VLMs are better at moderately fine-grained concepts. But I do have a concern, please see the weakness.

+ [Specificity Robustness is well-delivered] From such analysis, we know the scores of contrastive vision language models can easily be distracted. I could guess such observation may be caused by the training data where the text is uncorrupted. I am wondering any solution to mitigate such specificity.

### Weaknesses
- [Hierarchical Labels vs. Coarse-grained Labels] There is a paper (Chils: Zero-shot image classification with hierarchical label sets in ICML 2023) that shows that using fine-grained/ subclass-level labels helps zero-shot classification. It seems the conclusions of this work and ICML work are different. Please discuss this.

- [Specificity of Text] I think the observation that VLMs could be sensitive to the specificity of text is straightforward. When training VLMs, they are given specific texts. If we use the proposed modified version for training, I would expect them to perform better. Please comment on my thoughts.

***Post Rebuttal***

Thanks for the reponse, which is helpful to adress my intial concerns. Please make sure you include these disucssions in the revised version. I maintain my recomendation of Accept.

### Questions
- Please discuss the different observations between your work and ICML work. Also, comment on my thoughts for the bias caused by the training manner. 

- Section 4.2 (Impact of Pre-training Data) is not clear to me. Please elaborate your points here.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a benchmark for evaluating the granularity and specificity of text prompts in vision-language pretraining such as CLIP. Experiments are conducted over several state-of-the-art vision language models (e.g., CLIP, OpenCLIP-LAION, BLIP, etc.). For granularity, the benchmark includes images from ImageNet-1K dataset in which the label hierarchy is extracted by WordNet. The evaluation includes matching the cosine similarity of an object name with its parent and child category names. Results show that the models are better at recognizing moderately fine-grained concepts than course-grained concepts. For specificity, the data for evaluation includes captioning example from MS-CoCo, in which caption text is adjusted. The results show that short (single label) or long captions can produce lower scores than captions with correct specific details that also include small errors. Overall, the evaluation points to limitations of these models in the zero-shot classification task.

### Strengths
The study and finding of the limitations of VLMs is of interest.

### Weaknesses
The evaluation on the technical side is rather straightforward, and I am not sure how surprising the results are. Although this study may provide few valuable insights, I do not think it meets the ICLR novelty threshold.

### Questions
none

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Authors analyze the performance of off-the-shelf VLMs (e.g. CLIP, Open-CLIP, BLIP, FLAVA, etc.) on their ability to correctly match paired visual and textual descriptions at different levels of granularity and specificity. To this end, authors propose two benchmarking protocols based on ImageNet and COCO-Captions. They find that VLMs achieve the highest accuracy at recognizing "moderately fine-grained concepts" and VLMs struggle to correctly rank different captions with incorrect details. Authors evaluate several VLMs under their proposed setup and provide analysis on their results.

### Strengths
- Representative Baselines. Authors evaluate a representative set of VLMs with different backbones and pre-training datasets. 
- Sensible Analysis. Authors conduct sensible analysis on the OTS pre-trained models to back up their claims. All of the results of the analysis are intuitive and reasonable.

### Weaknesses
- Lack of Novel Insights. The primary claims in this paper about the performance of VLMs on specificity and granularity are not new, primarily because these have been explored in the language community [1, 2]. This is particularly important because both specificity and granularity are primarily language-based reasoning tasks. For example, an image of a leopard is unlikely to be close in feature space to an image of a bird even though both are examples of animals. In contrast, the language embedding contributes significantly more to this problem. Similarly, the evaluation of specificity is largely dependent on the text encoder's ability to capture fine-grained details (which is well studied in the language community.) Further, the conclusion that the distribution of data plays a significant role in VLM performance has been explored in [1]. Since VLMs are often trained on uncurated web-sources, we expect that they will capture the distribution and correlations found on the web. Therefore, the conclusion that VLMs are better at recognizing "moderately fine-grained concepts", is simply a restatement that VLMs are better at matching images with commonly occurring captions. To take an extreme example, a VLM is unlikely to score the concept "object" and an image of a "leopard" highly since these hierarchical correlations (while true) rarely occur in the training data. Similarly, a VLM is unlikely to score the concept of "leopard" highly with its scientific name "Panthera pardus" because it is also rarely seen during training. With respect to the specificity benchmark, the insight that vision language models can be easily distracted has also been explored in [2] (Note that although [2] focuses on language, the textual bias of the specificity task makes this work relevant). 
- Textual Bias in Specificity Benchmark. As the authors point out in the supplement, language-only methods are competitive on this benchmark, suggesting that the protocol should be amended to more accurately to evaluate vision-language capabilities, rather than just language understanding. In fact, many VLM benchmarks face the same problem of not requiring vision to solve the task. 
- On the Importance of Granularity from Vision. The granularity benchmark actually tests a model's alignment with word-net, which may not be universally accepted. For example, searching the web for "road-cone" and "barrier" show two visually distinct items. Although people may agree that road-cone and barrier are related, it is unclear if road-cone is a child class of barrier. Therefore, simply training a VLM in alignment with word-net rather than the broader internet may artificially boost performance on this benchmark, but perform worse in practice. 

References

[1] Large Language Models Struggle to Learn Long-Tail Knowledge. Kandpal et. al. ICML 2023

[2] Can Large Language Models Truly Understand Prompts? A Case Study with Negated Prompts. Jang et. al. NeurIPS Workshop 2022.

[3] When and Why Vision-Language Models Behave like Bags-Of-Words, and What to Do About It? Yuksekgonul et. al. ICLR 2023.

### Questions
- What is the Summary Metric? Although the in-depth analysis is appreciated, it is difficult to identify the best and worst performing methods (as is common in a benchmark). How can the methods be ranked for both benchmarks?
- How Should Training be Amended? In light of the analysis of this paper, what are the right steps to amend VLM pre-training. Although authors suggest fine-tuning with hard-negative text prompts, this again seems to bias the model to this particular benchmark (no longer benchmarking zero-shot recognition), and does not provide a solution in general. 

Please revise the text of this paper, there are many spelling and grammar errors (e.g. Figure 5 poisoneded)

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents new benchmarks for Vision-Language Models (VLMs) in zero-shot recognition tasks, emphasizing granularity and specificity. The authors use adapted ImageNet and MS-COCO datasets to test VLMs' performance across varied granularity levels and their sensitivity to text specificity. Key findings reveal that VLMs excel at recognizing moderately fine-grained concepts but struggle with high-level coarse concepts and are sensitive to the specificity of language prompts. This in-depth evaluation uncovers significant limitations in state-of-the-art models like CLIP.

### Strengths
1. The paper is clear, and well-written. 
2. The authors point out an interesting problem regarding granularity and specificity existing in current vision-language alignment models. It designed two simple benchmarks to evaluate those perspectives.  Their benchmarks address an important gap, making the paper a valuable resource for other researchers.
3. One of the standout findings, the improved performance through direct propagation from leaf nodes, offers fresh perspectives in the domain of zero-shot recognition. This insight could influence future model designs and strategies.

### Weaknesses
1. Lack of in-depth Analysis: The influence of prompt design on classification is a critical aspect that seems to be overlooked in the paper. An in-depth analysis of how prompt design effectiveness may sway classification results, especially in terms of granularity, is necessary to provide a comprehensive evaluation.

2. Going Beyond Common Knowledge: The paper highlights that performance in granularity and specificity improves when the testing scenario mirrors the training set, a well-known fact in the field. A more substantial contribution could be made by delving deeper into this issue, perhaps by providing baseline solutions or strategies to enhance granularity and specificity in pre-trained models.

3. Connection to Specific Vision Tasks: While the paper provides a high-level motivation for the need of such benchmarks, it stops short of connecting the dots to specific vision tasks. A more robust argument could be made by demonstrating how improvements in the proposed benchmarks translate to advancements in other vision-language tasks, such as open vocabulary object detection[1], open vocabulary tracking[2], and text-to-image generation[3]. At least, a high-level discussion is needed in this regard.

These points aim to encourage a more thorough exploration of the topics and a clearer demonstration of the benchmarks’ applicability to real-world tasks.

[1] Gu, Xiuye, et al. "Open-vocabulary object detection via vision and language knowledge distillation." arXiv preprint arXiv:2104.13921 (2021).

[2] Li, Siyuan, et al. "OVTrack: Open-Vocabulary Multiple Object Tracking." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[3] Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

### Questions
Evaluation Using COCO: Why are COCO's original captions more effective than detailed ones? Is there any assurance that the model hasn't been exposed to any COCO images or captions during training?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes to benchmark the zero-shot recognition ability of VLM in the axes of varying granularity levels and the specificity of language inputs. The findings show that existing VLMs tend to struggle with semantic granularity and are sensitive to text specificity. The study can help guide future works to benchmark and improve VLMs in these axes, making VLMs more robust and useful at scale.

### Strengths
The paper presents a thorough study on the zero-shot recognition capability of a suite of VLMs in terms of granularity and specificity. Both are important areas that the community should be aware of.

The findings on granularity are informative as it shows the VLMs are best at recognizing concepts in similar granularity as the training data distribution (raw scores of ancestor are much worse than aggregating max scores of the leaves). Figure 4 (left) connects the empirical findings of Figure 3 to the training data, which is convincing.

The observation of specificity is quite informative as it shows the model is best at associating images and texts with the right amount of information. 

The studies in Table 2 and 3 are performed with a broad range of VLM, which makes the conclusion robust to modeling choice.

### Weaknesses
1. Apart from the analyses, it'd be great to explore some simple ideas to mitigate the granularity and specificity issues in these VLMs by light-weight finetuning. Some ideas include: 
- Augment the alt-text with LLM to increase the amount of information
- Generate hard-negatives by replacing the nouns
- Adjust the granularity by replacing the fine-grained concepts with their parent in the hierarchy
Since the analysis points to the mismatch between training and test data distribution, it'd be natural to bridge the distribution gap by data augmentation.

2. It's not obvious how to translate the findings of this work into improvements in the downstream application of CLIP. Take the granularity study for example. Let's say we want to build an open-vocabulary detector on the some super categories of LVIS. To achieve the best performance, we'd need a way to generate the fine-grained categories associated with those super-categories, where the super-categories may not be in the WordNet hierarchy. One idea is to use the LLMs to help generate fine-grained categories so that we can apply the max-score aggregation in Fig 2 (c).

3. Although the analyses are comprehensive and interesting, the findings are not very surprising. It seems to boil down to the limitations of the data that naturally exists at large-scale at the end of the day. I'd recommend to take the discussion section outside of conclusion and expand on it more to address the limitations.

### Questions
See weakness.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good
