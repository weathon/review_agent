# SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
We present SEED ($\textbf{Se}$mantic $\textbf{E}$valuation for Visual Brain $\textbf{D}$ecoding), a novel metric for evaluating the semantic decoding performance of visual brain decoding models. It integrates three complementary metrics, each capturing a different aspect of semantic similarity between images inspired by neuroscientific findings. Using carefully crowd-sourced human evaluation data, we demonstrate that SEED achieves the highest alignment with human evaluation, outperforming other widely used metrics.
Through the evaluation of existing visual brain decoding models with SEED, we further reveal that crucial information is often lost in translation, even in the state-of-the-art models that achieve near-perfect scores on existing metrics. This finding highlights the limitations of current evaluation practices and provides guidance for future improvements in decoding models.
Finally, to facilitate further research, we open-source the human evaluation data, encouraging the development of more advanced evaluation methods for brain decoding. Our code and the human evaluation data are available at https://github.com/Concarne2/SEED.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
In “SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding” authors proposes a new metric, SEED, to assess the semantic accuracy of images reconstructed from brain activity. They argue that existing metrics (e.g., SSIM, CLIP, Inception) fail to align with human perception of semantic similarity, often overestimating model performance. SEED combines three complementary measures: (1) Object F1, which captures object-level overlap using open-vocabulary grounding models; (2) Cap-Sim, which compares captions of ground-truth and reconstructed images using text embeddings; and (3) EffNet, a feature-based similarity measure. Through medium-scale human evaluations on 1,000 pairs from NSD and additional datasets, SEED metric demonstrates stronger correlation with human judgments than existing metrics and reveals that state-of-the-art decoding models still struggle to reconstruct fine-grained semantics.

### Strengths
This is a very timely contribution: Evaluation misalignment between metrics and human perception is a major issue in brain-to-image decoding. In the last 2-3 years I have worked in the field and seen a significant amount of brain decoding works, always competing for few fractions of percentage points of saturated metrics that not always reflect real performance.
Here, SEED addresses a crucial gap that could significantly impact future research.

The metric is well-motivated, interpretable, and partially grounded in both neuroscience and computer vision (object-based attention and semantic binding).

The authors conduct meta-evaluations, robustness tests across models/datasets, and qualitative analyses (worst-case examples, semantic near-miss, and failure modes).

A very good point is the commitment to release the human evaluation dataset adds substantial reproducibility and community value. I'd also suggest the authors to make this metric a very easy to use (ideally a one-liner python function) to encourage reproducibility.

### Weaknesses
Introducing the metric is an important point, but I felt something is missing in this work:

Some literature is absent, the number of work and approaches has become very big. Many are missing. To position in the field and constructively criticize the practices I think extended literature review is needed. 

This metric is better aligned with human judgement, but still has some bias from the model and limitations. Could the author discuss more in detail these limitations and future work to be improved? Other metrics, ecc.

Overall, this metric is a linear combination of a bunch of others, all relying on some models. In appreciate the point of view even if the technical contribution is somehow limited.

### Questions
1) Will this metric/code be released in a easy to access and no-brainer way to encourage wide usage? I think this is really the keypoint here, otherwise the whole thing loses a lot of traction.

2) In the field everyone is competing to have his own line in bold. Your new metric was evaluated on some famous approach, but the best-in class and the order didn't change much. Since the ranking was mostly unaffected, could you highlight better the cases where this metric is telling something more? I liked the failure cases ecc, but I really would love to see more example and use cases where errors or successes were hided by other metrics and SEED is able to point them out

3) The sample size for human evaluation is kind of limited but I understand this is difficult to solve. There is a lot of work on THINGs dataset with million of annotations and human preferences between categories and images ecc, as well as many models trained on them, Does it make sense to compare SEED with these models?

4) One thing that could blow my mind is whether you think SEED could became an objective loss function to be optimized. Right now, as far as I understood it involves several non-differentiable operation that limit the use as a metric alone, but do you think this could be extended as a training objective?

5) Discussion for a position paper like this is a bit short and dry. What are the implication? What's the significance? Why it's needed? Plus general reflection on the field would improve the maturity of the work.

### Soundness
4

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
4

### Summary
This paper proposes SEED (Semantic Evaluation for Visual Brain Decoding) — a new metric designed to better evaluate visual brain decoding models in terms of semantic similarity. The authors integrate three components — Object F1, Cap-Sim, and EffNet — to capture complementary aspects of human-like visual perception. Extensive human evaluations demonstrate that SEED aligns more closely with human judgments than existing metrics.

### Strengths
* A more human-like evaluation framework is essential for the brain decoding community.	

* The use of large-scale human evaluations is impressive.

### Weaknesses
* A key limitation of SEED lies in its reliance on off-the-shelf captioning and detection models (e.g., GIT, Grounding-DINO). These components were not trained to reflect human semantic judgments but to optimize task-specific objectives (caption likelihood or object detection accuracy). As a result, SEED may inherit their systematic errors, leading to misleading evaluations in certain cases.

* The authors argue that in some existing metrics such as n-way identification, decoding models have reached near-ceiling performance, but this is largely due to the small candidate pool typically used in evaluation. If the number of candidate images is substantially increased (e.g., 100-way or 1,000-way identification), most current decoding models exhibit significant performance degradation, revealing substantial room for improvement.

* Moreover, recent progress in visual decoding is moving toward multimodal decoding frameworks (text and image). In such settings, the semantic quality of reconstructed images can already be assessed through predicted text. This trend raises questions about the necessity of using GIT-based caption generation in SEED. Since GIT introduces additional linguistic biases and noise unrelated to the decoding model itself.

* The complexity of SEED may to some extent hinder its widespread adoption. For example, its reliance on multiple models introduces issues such as version differences, model updates, and parameter inconsistencies, which reduce the reproducibility of results and the comparability within the community.

Minor: Several recent works such as [1-2] in brain decoding are not cited or discussed.

[1] Bridging the Gap between Brain and Machine in Interpreting Visual Semantics: Towards Self-adaptive Brain-to-Text Decoding, ICCV 2025.

[2] Mindgpt: Interpreting what you see with non-invasive brain recordings, IEEE TIP 2025.

### Questions
* Could SEED be adapted to multimodal decoding (e.g., brain-to-text or cross-modal tasks)?

* Could evaluating the semantic fidelity of reconstructions using fine-grained category labels rather than object detection models provide a more efficient and reliable assessment?

### Soundness
3

### Presentation
3

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
This paper identifies a critical problem: current evaluation metrics are poorly aligned with human judgment of semantic similarity in visual brain information reconstruction. Even if results achieve near-perfect scores on existing metrics, they may still be semantically flawed. To address this, they propose SEED, a new evaluation metric that integrates three complementary components—Object F1, Cap-Sim, and EffNet. Through extensive "meta-evaluation" against a new human judgment dataset, they demonstrate that SEED aligns significantly better with human evaluation than all existing metrics.

### Strengths
1. The motivation of the article is very good. The current evaluation of decoding models indeed has such a problem. Especially, many tasks utilize contrastive learning in the CLIP space and then proceed with image generation. Naturally, this leads to excellent generation scores.

2. The design of SEED is thoughtful. The two new proposed metrics, Object F1 (object-level attention) and Cap-Sim (feature binding into a scene description), offer novel and complementary perspectives. It is indeed something worth noting during the recovery process.

3. The article conducted numerous human alignment experiments and evaluated the results of many models, proving that the proposed indicators are indeed closer to human understanding of image restoration.

### Weaknesses
1. The proposed indicators mainly focus on assessing semantics and may be insensitive to information like color and texture. This is also a significant factor influencing restoration and human perception.

2. The article does not provide a clear description of how human evaluations are conducted. For example, in Figure 1, why is that image ranked 846th out of 1000? How was this ranking determined? The human evaluators rated both "semantic and perceptual similarity". For the meta-evaluation, which rating was used?

3. The method used in the article is reasonable but somewhat complex, which is composed of many current models. I'm not sure if there might be any hidden biases here. If one of the models has a problem, will it lead to the failure of the evaluation? This merely presents a feasible solution, but does not discuss whether there are better evaluation methods or directions for improvement.

### Questions
1. In terms of semantic understanding, this is a problem that all generative models encounter. Does this method and insight also work for current generative models? Or is there any new semantic metric in the field of image generation now? How are they being used?

2. Given that multiple large models were used, what was the speed of the inference? What is the total amount of computing resources consumed for the operation? If it is extremely large, it may affect the use of the indicators. Besides, how and why are the weights of multiple models determined?

### Soundness
3

### Presentation
3

### Contribution
3
