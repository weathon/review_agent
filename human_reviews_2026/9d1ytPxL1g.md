# Are Large Vision-Language Models Good Annotators for Image Tagging?

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Image tagging, a fundamental vision task, traditionally relies on human-annotated datasets to train multi-label classifiers, which incurs significant labor and costs, especially for large-scale label spaces. While Large Vision-Language Models (LVLMs) offer promising potential to automate annotation, their capability to replace human annotators remains underexplored. 
This paper systematically investigates LVLMs' annotation quality and the performance of models trained on LVLM-generated labels. Our analysis reveals that LVLMs achieve competitive performance on common categories but lag behind humans on uncommon or ambiguous categories. Surprisingly, models trained on LVLM-generated labels outperform those trained on human-annotated labels in certain categories, suggesting imperfections in human annotations. 
Motivated by these findings, we propose \textsc{LVLMAnt}, a novel framework for image tagging, which aims to achieve human-level annotation ability. \textsc{LVLMAnt} comprises two components: Prompts-to-Candidates (P2C), which employs group-wise prompting and annotation ensembling to efficiently produce a candidate set that covers as many true labels as possible while reducing subsequent annotation workload; and Concept-Aligned Disambiguation (CAD), which interactively calibrates the semantic concept of categories in the prompts and effectively refines the candidate labels.
Extensive experiments on benchmark datasets demonstrate \textsc{LVLMAnt}’s effectiveness in balancing annotation quality and automation, significantly reducing reliance on manual effort while achieving performance comparable to human annotations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates if LVLMs can be used as an alternative to Human annotations for image tagging task. The paper presents a systematic analysis comparing LVLM-generated annotations with human annotations on MS-COCO 2014 and Objects365 datasets. They find that  LVLMs annotations are: (1) accurate for common categories (and perform better than human annotations) but fail with uncommon categories. (2) Robust to missing labels (even if LVLMs miss labels, the other labels are often sufficient to train the model). Based on these, the paper presents LVLMANT, a two-stage framework consisting of: (1) Prompts-to-Candidates (P2C) using group-wise prompting and ensemble techniques to generate candidate labels, and (2) Concept-Aligned Disambiguation (CAD) to refine candidates by addressing semantic misalignments.

### Strengths
1. Paper provides comprehensive empirical analysis of different prompting strategies (open-ended, multi-option, binary) and their trade-offs. 

2. LVLMANT effectively combines the efficiency of multi-option prompting with the precision of binary prompting, achieving a good balance between annotation quality and cost.

3. Extensive experiments on multiple benchmark datasets with detailed ablations is provided

4. The observation that models trained on LVLM annotations sometimes outperform those trained on human annotations for certain categories provides valuable perspective

### Weaknesses
1. How are the sets of co-occurring and disco-occurring classes chosen?

2. Following up on the previous point, more discussion on cases where LVLMANT fails needs to be discussed. For example, If co-occurrence patterns are derived from ChatGPT4, do these align with actual patterns in datasets like COCO? The paper lacks systematic analysis of whether pattern misalignment degrades performance.

3. Testing with weaker LVLMs (like LLava or InstructBLIP) would better demonstrate robustness, particularly whether CAD can handle noisier candidate sets from lower-capability models.

4. I like the analysis provided by the paper, but I believe the work is more applied and could be a better fit at vision conferences (CVPR, ICCV/ECCV) or NLP conferences. Based on "call for papers" I could not find a good match for it.

### Questions
Please see the sections above.

### Soundness
3

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
This paper addresses the problem of automating annotation for image tagging. To avoid notable labor and costs introduced by traditional human annotation, the work proposes to leverage large vision-language models for generating labels. They present two strategies, including Prompts-to-candidates (P2C) and Concept-Aligned Disambiguation (CAD), building the LVLMANT framework. P2C employes group-wise prompting and annotation ensembling to produce candicate sets, and CAD further calibrates the concept spaces. Extensive experiments on COCO and Object365 show the effectiveness of the proposed framework.

### Strengths
1. The writing and figure are clear, which makes the paper easy to follow and understanding.
2. Convincing analyses are presented. Metrics, including CP, CR, and downstream performance using the generated labels, verify the claims and show the efficacy of proposed P2C and CAD strategies.

### Weaknesses
1. Comparison with some other works are missing, like [1], CaSED [2], and CLIP. Like RAM and RAM+, they can also be used for annotation.
2. Some experimental details are not presented. In Table 1, what is the category vocabulary used by Qwen2.5-VL-7B/32B, RAM/RAM++, and LVLMANT 7B/32B? Is it the ground truth category set of datasets? Why the GPU time in hours required by Qwen2.5-VL are so large? Does Qwen2.5-VL use binary prompting? Besides, in the appendix, it is stated that categories are mapped for RAM++ and seems that predefined categories are used for RAM++. RAM++ also supports customizing tag categories for recognition. The baseline that directly using the categories from the datasets for RAM++ is missing.
3. In Table 3, the performance improvements of Similarity and Description are relatively small, whose effectiveness is not sufficiently verified.
4. The GPU time in hours for P2C and CAD are not presented, respectively, which makes the time saving inclear.
5. In Table 3, there lacks a baseline that P2C + binary promping, which makes the efficay of CAD unclear.

[1] Object Recognition as Next Token Prediction. CVPR 2024.

[2]  Vocabulary Free Image Classification. NeurIPS, 2023.

### Questions
Please see the Weaknesses

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
3

### Summary
The paper approaches a problem of evaluating whether a LVLM can replace human on the task of annotations. The study shows that on the common categories the LVLM outperforms humans than the ambiguous ones. After this they propose LVLMANT, a two-stage annotation framework. The first stage, Prompts-to-Candidates (P2C) generates a high recall candidate set and the second stage, Concept-Aligned Disambiguation (CAD), uses an LLM to automatically identify ambiguous category increases the precision. The results show that the annotation quality reaches human level by reducing costs by 36x.

### Strengths
1) The paper addresses an expensive bottleneck of manual data annotation in machine learning. This paper solves this in a systematic manner.

2) The paper achieves a very high 36x reduction in human annotation cost, hence proving to be a practical solution.

3) The two stage process ensures high recall in the first step and high precision in the next one. Overall aims to optimize for maximum performance.

### Weaknesses
1) The paper goes beyond the 9 page limit of ICLR. If it was not desk rejected , the paper seemed to have passed the initial stage though.

2) Although the human cost is reduced by 36x , the improvement on mAP is low.

3) The paper combines the existing techniques in two stages, so the technical contribution can be seen as shallow. 

4) The VLLMs can make same hallucinations in the two stage and hence the 'noise' can amplify in subsequent stages.

### Questions
1) How does the P2C ensemble handle correlated hallucinations, the models can make same hallucinations that can propagate?

2) Given the two-stage framework, how to justify 1.34% mAP improvement on COCO is not just statistical noise?

3) How can the paper claim CAD is a novel technical contribution when it is based on closed source model that is ChatGPT-4o?

4) If human annotations are imperfect, are these used as ground truth?

### Soundness
3

### Presentation
3

### Contribution
3
