# POINTS: Improving Your Vision-language Model with Affordable Strategies

- Decision: Reject
- Scores: 5, 5, 5, 3

## Abstract
In recent years, vision-language models have achieved significant advancements, excelling in tasks once deemed challenging, such as optical character recognition and geometric problem-solving. Despite these impressive achievements, several critical issues remain unaddressed: 1) Proprietary models rarely disclose detailed information about their architectures. In contrast, while open-source models provide visibility into their training strategies, detailed ablations of these strategies are highly anticipated. 2) Pre-training data is currently under-explored in open-source works, with most efforts empirically adding datasets from diverse sources, making the entire process elusive and cumbersome. 3) During the fine-tuning stage, the focus is often on adding and ablating more datasets, which frequently leads to diminishing returns. Therefore, refining data schemes is essential for further enhancing model performance.
To address these issues, we propose the following contributions in this paper: 1) We trained a robust baseline model, leveraging the latest technological advancements in vision-language models. Building upon existing advancements, we introduced effective improvements and conducted comprehensive ablation and validation for each technique incorporated into this strong baseline.
2) Inspired by recent work on large language models, we propose filtering pre-training data using perplexity, selecting the data with the lowest perplexity as the training set. This approach allowed us to train on a curated 1M dataset, resulting in highly competitive performance. 3) During the visual instruction tuning stage, we experimented with model soup on different datasets when further introducing more datasets into the training set brought marginal improvements. Integrating these innovations, we obtained a model with 9B parameters, performing competitively with a series of existing state-of-the-art models. Additionally, these strategies we propose are efficient and relatively lightweight, allowing the community to adopt them easily for their models.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper addresses the problem of model design choices and dataset selection in vision-language models learning. The authors propose a combination of existing methods to build a strong and sample-efficient baseline using a relatively small pretraining dataset of size ~1M. The main contribution is on the analysis of the dataset (individual select) and model selection (model soup) using and combining existing methods. The authors also propose a  novel image splitting strategy to tackle the well known image distortion problem in models that use a dynamic high resolution image processing.

### Strengths
* Well-written and easy to follow.
* Comprehensive set of VL benchmarks including hallucination and OCR benchmarks
* Table 2 shows the benefit of a perplexity-based selection of the base pretraining dataset, which is an interesting result in terms of compute efficiency.
* The authors manage to get competing performance in a wide set of VL benchmarks with a much smaller pretraining dataset (~1M)  than other methods.
* All the datasets and pretrained backbones used by the authors are publicly available which make it easily reproducible.

### Weaknesses
* Ablation study is thorough but the comparison with other baselines is not discussed. Specifically, POINTS is outperformed in many of the benchmarks the authors chose by similarly sized open-source models, but the differences are not discussed.
* The authors also did not discuss the choice of the base LLM (l380 Yi-1.5-9B-Chat) compared to other baselines, which makes it difficult to disentangle the reasons for the difference in performance.
* The authors introduce a QA dataset distilled form GPT-4O and include it in their base set. It is possible that this knowledge distillation alone explains why adding more data is not efficient and why a ~1M dataset is enough. The authors only justify this addition by the need to have prompt diversity but do not ablate it.

**Minor Issues**
* Small typo l480: refer to table 4 instead of table 5
* Some tables are missing the hyperlink ref (l460 and l480 for instance)
* Typo l482 “increasing adding more”
* The authors could rephrase l058 ” Unlike most existing works (Li et al., 2024; Chen et al., 2024c), we extensively ablate each newly introduced component in the strong baseline to verify their individual benefits” and not point out the lack of ablations in other works, especially when those cited work actually reference a whole blogpost about ablations in LLaVA-Next (see [1]) . 
[1]  https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/

### Questions
* Could the authors discuss in more detail the choice and comparisons of baselines ? I acknowledge that it is easy to get lost in the number of new VLMs coming out these days but since the authors propose an analysis of the training + modeling frameworks, i would expect a comparison table that could explain the choice (and exclusion) of VLMs instances based on a few criteria (datasets, size, LLM base, ViT base, presence of OCR data, dynamic high resolution processing etc..)
* In the ablation study Table 1, what does it mean to not have IS? Does it mean training on all the potential candidates additional datasets?
* Could the authors discuss and analyze the choice of the base LLM (Yi-1.5-9B-Chat l388) ? Specifically, why not choose a base LLM that is the same as the closest baseline (for instance InternVL 1.5 that also has a dynamic high resolution processing)? It would make it easier to disentangle the contributions that come from the dataset selection, the model soup and the base LLM compared to other baselines.

Overall, the authors presented their method in a very clear and structured way. The authors also perform a thorough ablation study. However some key choices (such as GPT-4O distilled dataset, a different base LLM compared to the other baselines) are not discussed or ablated which makes it difficult to disentangle the benefit of the overall approach. The lack of discussion around the baselines and their performance makes it challenging to evaluate the novelty and impact of the proposed contribution.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes to improve the baseline LLaVA model in terms of the following three aspects: (a) to refine the pre-training dataset quality, the paper introduces CapFusion utilizing an LLM to integrate world knowledge from existing captions while simplifying their structures, (b) the model architecture is improved through the introduction of dynamic high resolution component with consistent aspect ratio, and (c) the visual instruction tuning is enhanced through the usage of model soup that merges weights from models fine-tuned on different learning rates.

### Strengths
- The proposed components are indeed effective and target the limitation of existing vision-language models in great detail.
- Compared to finetuning dataset selection, the proposed model soup is a more sustainable design choice in terms of the return on performance with scale.
- Perplexity-based dataset selection technique is simple and could have a potentially broader impact on data-efficient pretraining of large foundation models.

### Weaknesses
- While the paper is overall well-written, it is hard to spot the motivation behind the introduced components now and then. For instance, the introduction states "... However, recent advancements have rendered its (LLaVa's) performance suboptimal. Thus, 
 there is a need to establish a stronger baseline for further exploration..." Without proper references or an explicit description of how LLaVa's performance has gone suboptimal, the current motivation reads as "We know the model x and we want to improve upon it." While the latter motivation is indeed genuine, this however opens up a plethora of search space where the model x can be improved. As such, I am reluctant about why the paper would then improve model x in only a few chosen directions.

- Continuing on the above note, a good part of the experiments are occupied with performance on OCR-related tasks. However, there is no such concrete discussion of why the paper chooses to highlight the OCR task, what limitations the current baselines have for the OCR task, and what exactly are the example scenarios where the proposed model performs better on OCR than the baseline LLAVA. 

- My major concern with the current experiments is a lack of discussion on the additional resource overhead brought about by the proposed components. For a practitioner to prefer the proposed POINTS over LLAVA, it would be helpful if the paper mentions: (a) the overhead in pretraining/finetuning in terms of computational resources such as GPU memory, training/inference times in terms of wall-clock/FLOPs as well as the data efficiency of the proposed method. On the lattermost note, I would like to see a discussion on how the POINT model performs compared to LLAVA if a downstream fine-tuning task has limited (few-shot) training examples.

- How does the proposed CATTY method scale with the image size in terms of the model inference time?

Minor suggestions: 

- Introduce the full name for OCR before the first usage of the acronym in the paper (line 057).
- Lines 75-78: citation required for ".. Compared to conducting model soup on models fine-tuned with different hyper-parameters, e.g. learning rate, the improvement with model soup on models fine-tuned with different datasets is much more prominent."

### Questions
See Weaknesses. Most of my current concerns with the paper are around a lack of proper motivation and analyses of the proposed components.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes affordable and efficient methods to improve vision-language models, focusing on three main parts: (1) building a robust baseline model with thorough validation, (2) using perplexity filtering to improve the quality of pre-training data, and (3) applying "model soup" to merge weights from different fine-tuned models. The final model, POINTS, achieves competitive performance with state-of-the-art models while remaining lightweight and accessible. The authors also introduce improvements like CATTY to reduce image distortion and enhance OCR capabilities, all while keeping computational costs manageable.

### Strengths
- The use of perplexity-based filtering of pre-training data is inspired by its success in large language models. This approach is a creative application of an existing concept in a new context, addressing data quality for vision-language models.
- The paper is well-structured, with clear divisions between each method's presentation and the experimental results validating those methods.
- The paper's focus on affordable strategies is practical for the broader community, especially for those without extensive computational resources. Achieving competitive performance using lightweight techniques is useful.

### Weaknesses
- Each of the proposed methods, such as perplexity filtering, CATTY, and model soup, are either existing methods or slight modifications of existing approaches. This limits the overall novelty of the contributions. More significant deviations or novel techniques could make the paper's contributions more impactful.
- While the authors claimed model soup has introduced improvements, the paper does not thoroughly explore its potential limitations or generalizability across different tasks or datasets. There are only limited details on the possible negative effects of merging model weights, particularly if the datasets have significant divergence in distribution. Also, I wonder if there can be potential overfitting to benchmark from the Greedy Soup method and why "MathV360K" has no improvement while others have improved.
- While perplexity filtering effectively reduces the dataset size while preserving model performance, there is a potential risk that the model's ability to generalize could regress due to a lack of exposure to a diverse range of data. The paper does not provide any experiments or discussion addressing this potential drawback. I think it is important to understand the gap in the understanding of the impact of data reduction on generalization performance.

### Questions
- While perplexity filtering effectively reduces the dataset size while preserving model performance, there is a potential risk that the model's generalization ability could regress due to reduced exposure to diverse data. Could you provide more insights or experimental evidence on the impact of this data reduction on the model's generalization capabilities?
- Regarding the Model Soup technique, is there a potential risk of overfitting to benchmarks due to the greedy soup method? Could you elaborate on why "MathV360K" showed no improvement while other datasets did?
- Have you observed any cases where CATTY underperforms compared to DHR? It would be useful to understand any trade-offs involved, particularly in terms of applicability to different image types.
- Typo in the table 4 "Greey Soup" -> "Greedy Soup"

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a robust baseline vision-language model. The main contributions are threefold: (1) Framework improvement: the authors introduce Consistent Aspect Ratio Dynamic High Resolution for the vision encoder of the multimodal LLM; (2) Data filtering: the authors use perplexity to filter the pre-training dataset; (3) Model ensemble: this work employs model soup methods to merge models fine-tuned with different datasets to achieve better results.

### Strengths
- Paper is clearly written.

- Introduced approaches are technically sound.

### Weaknesses
- Novelty concern: The proposed strong baseline model primarily integrates multiple existing advancements in vision-language models, with limited novel technical contributions or findings. Specifically, the proposed CATTY encoding combines dynamic high-resolution encoding and sliding window techniques, both of which have been explored in previous works [1-3]. Additionally, model soup or model ensemble techniques have been shown to effectively boost performance. The novelty of the introduced maximum, average, and greedy soup methods for merging model results is limited.

- Incomplete experimental analysis: Experimental results comparing CATTY with DHR are shown in Table 1. However, the proposed CATTY encoding approach, which leverages a sliding window with overlapping, may also introduce extra computation costs, additional vision tokens, or repetitive information. This analysis is crucial for understanding the pros and cons of using CATTY for future work, yet these aspects are not examined.

- Confusing table and figure references: In the experiments section, some table and figure references are confusing. For instance, in Line 480, should this refer to Table 3? And in Line 484, should this be Table 4? This paper may require more attention to detail in the writing.

[1] Chen, Zhe, et al. "How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites." arXiv preprint arXiv:2404.16821 (2024).

[2] Lin, Ziyi, et al. "Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models." arXiv preprint arXiv:2311.07575 (2023).

[3] Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

### Questions
- Is the baseline in Table 4 trained only on a base instruction tuning dataset, or is it trained on a combination of a base instruction tuning dataset and a series of visual instruction tuning datasets? How was the baseline model trained?

- What specific off-the-shelf vision-language model was used to calculate perplexity in Table 2? In the experiments, did you observe any differences when using different VLMs?

### Soundness
3

### Presentation
3

### Contribution
2
