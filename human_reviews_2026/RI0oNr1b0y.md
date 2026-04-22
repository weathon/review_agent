# Visual Self-Refine: A Pixel-Guided Paradigm for Accurate Chart Parsing

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
While Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities for reasoning and self-correction at the textual level, these strengths provide minimal benefits for complex tasks centered on visual perception, such as Chart Parsing. Existing models often struggle with visually dense charts, leading to errors like data omission, misalignment, and hallucination. Inspired by the human strategy of using a finger as a ``visual anchor'' to ensure accuracy when reading complex charts, we propose a new paradigm named Visual Self-Refine (VSR). The core idea of VSR is to enable a model to generate pixel-level localization outputs, visualize them, and then feed these visualizations back to itself, allowing it to intuitively inspect and correct its own potential visual perception errors. We instantiate the VSR paradigm in the domain of Chart Parsing by proposing ChartVSR. This model decomposes the parsing process into two stages: a Refine Stage, where it iteratively uses visual feedback to ensure the accuracy of all data points' Pixel-level Localizations, and a Decode Stage, where it uses these verified localizations as precise visual anchors to parse the final structured data. To address the limitations of existing benchmarks, we also construct ChartP-Bench, a new and highly challenging benchmark for chart parsing. Our work also highlights VSR as a general-purpose visual feedback mechanism, offering a promising new direction for enhancing accuracy on a wide range of vision-centric tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Visual Self-Refine (VSR), a novel paradigm to address the poor performance of LVLMs on vision-centric tasks like chart parsing, where text-based self-correction fails. Inspired by the human strategy of using a finger as a "visual anchor" , VSR enables a model to generate pixel-level localizations, visualize them on the image, and then feed this visualization back to itself to iteratively inspect and correct visual perception errors like omissions or hallucinations. The authors instantiate this paradigm in a model called ChartVSR, which uses a Refine Stage to verify pixel locations and a Decode Stage to parse the chart using these verified anchors. This work is supported by two additional contributions: a new challenging ChartP-Bench benchmark and a robust data engine for generating a large-scale , diverse training dataset.

### Strengths
- The paper is very well-written and very clear.
- The paper conducted abundant analysis.
- Meaningful contributions in both data and method.
- The potential applications to other precision-oriented tasks like Visual Counting and Visual Grounding are clearly articulated (Figure 6) , opening a promising new research direction.

### Weaknesses
- In Table 2 and 3, the authors are comparing open-source models of different sizes. The author should clearly list the sizes of these models for fairer comparison. Additionally, to better show the effectiveness of the proposed training data, the author can consider training the MatCha model with their data and do direct comparison with DePlot.
- For chart parsing evaluation, the paper should also compare with the Chart-to-Text [2] and AskChart [3].




[1] MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering. ACL 2023

[2] Do LVLMs Understand Charts? Analyzing and Correcting Factual Errors in Chart Captioning. ACL 2024 Findings

[3] AskChart: Universal Chart Understanding through Textual Enhancement.

### Questions
Could you provide more qualitative or quantitative analysis on the "stubborn errors" that VSR fails to correct after the first round? Are they of a specific type (e.g., heavily occluded points, ambiguous labels, tiny markers)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors first had the model generate pixel-level localization results, visualized these localizations and plotted them back onto the original image, then fed this labeled image back to the model, allowing the model to self-check and correct visual perception errors as if "checking point by point with a finger," and finally performed structured parsing based on the confirmed pixel anchors.

### Strengths
A visual self-refinement paradigm, Visual Self-Refine (VSR), is proposed: the model first generates localization points, visualizes them, and then feeds them back to the model for self-checking and error correction.

In the graph parsing task, the process is divided into two stages: Refine and Decode.

A challenging benchmark, ChartP-Bench, is constructed, and ChartQA-SE is cleaned to obtain ChartQA-SE-Clean. Significant performance is reported on multiple benchmarks, especially outperforming strongly closed-source models (such as Gemini-2.5-Pro) and existing dedicated models on ChartP-Bench.

### Weaknesses
There are limited benchmarks for evaluating papers, lacking authoritative datasets like Chart-Pro and ChartXiv.

This method has limited nooverty, and its two-stage design is very similar to the design philosophy of SoM. Many previous works on visual prompts have demonstrated that such visual prompts can improve performance.

It lacks a crucial baseline, such as a comparison of the localization ability of the first-stage model with that of other grounding models on chart localization tasks.

### Questions
Will the model trained after refine + decode experience a performance degradation on refinement tasks? Quantitative results are needed.


There are limited benchmarks for evaluating papers, lacking authoritative datasets like Chart-Pro and ChartXiv.

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
3

### Summary
This paper proposes Visual Self-Refine (VSR), a novel paradigm that introduces visual feedback as a self-correction mechanism for LVLMs in visually intensive tasks such as chart parsing. The method, instantiated as ChartVSR, decomposes the parsing process into two stages: the Refine stage and the Decode stage. The authors also present ChartP-Bench, a new and challenging benchmark featuring visually dense and stylistically diverse charts. Experimental results demonstrate consistent improvements over both chart-specific and general-purpose LVLMs.

### Strengths
1. The paper presents a novel and interesting paradigm for chart parsing, introducing a visually grounded self-correction mechanism that enhances interpretability and addresses an existing gap in LVLM perception.
2. The authors introduce a high-quality dataset, ChartP-Bench, which is carefully curated, diverse in style, and fills an important gap in chart parsing evaluation.
3. The ablation studies are comprehensive, providing thorough analyses of the effects of pixel localization and refinement, and consistently demonstrating the benefits of the proposed approach.

### Weaknesses
1. The chart parsing paradigm proposed in this paper can be viewed as a type of reasoning paradigm. However, the experimental section lacks comparisons with other recent visual reasoning models, such as o1, Qwen3-VL, and InternVL-3.5. I understand that some of these models might not have been publicly available at the time of submission, but I recommend that the authors include such comparisons in future revisions to strengthen the solidity and comprehensiveness of the work.
2. Around line 405, the paper explains why the AP-Strict scores are so low. According to the authors, this metric only rewards models that output exactly correct numerical values. I find this requirement excessively strict, as it is practically impossible to infer such highly precise numbers (sometimes up to two or three decimal places) from a single image. Therefore, I have some reservations about the practical significance and interpretability of this metric.

### Questions
1. I am curious about the scalability and generalization ability of the proposed approach, particularly regarding its performance on unseen chart types and the relationship between the number of localized regions and the extent of performance improvement.
2. The open-source models achieve performance comparable to ChartVSR on existing datasets such as ChartQA-SE-Clean, but their performance drops significantly on the newly proposed ChartP-Bench. Could the authors clarify the reason for this discrepancy? Is it possible that the training data distribution of ChartVSR is similar to that of ChartP-Bench, giving it an advantage? If this is the case, I would suggest that the authors conduct an additional experiment in which the training set explicitly excludes data similar to ChartP-Bench in order to more convincingly demonstrate the model's out-of-domain generalization ability.
3. How does the proposed model perform on chart-related question-answering tasks such as ChartQA and PlotQA?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a methodology to improve chart parsing by including an additional step of providing pixel-level annotations on the charts in order to help the model improve precision. The refine step involves generating the pixel locations which are then used to annotate the image. Both the image and the pixel values are fed as input to the model which then makes a prediction based on this additional information. Experiments involve evaluating on a number of existing datasets as well as a new dataset introduced in this work. VSR results in nice improvements in some settings but the results aren’t consistent on existing datasets.

### Strengths
The new dataset seems nice and useful for chart parsing evals.  
VSR is an interesting recipe and focusing on pixel-level annotations is a nice instantiation of this setup for chart parsing.  
Some of the improvements seem compelling, particularly the information dense charts.  
If the extra calls are prohibitive for an inference pipeline, this recipe can probably be used to create distillation data.

### Weaknesses
While the recipe is interesting, it’s not very general and will probably become outdated for this task as models’ visual understanding improves over time.  
It seems strange that performance doesn’t improve much after a step or two of refinement even though there’s so much headroom. Why is this? Maybe annotations should be adjusted or focused on incorrect ones? Or doing step-by-step correction is necessary? Either way, it seems like the feedback and the recipe need some…refinement.

### Questions
see above

### Soundness
2

### Presentation
3

### Contribution
2
