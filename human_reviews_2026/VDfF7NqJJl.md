# Panoptic Pairwise Distortion Graph

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
In this work, we introduce a new perspective on comparative image assessment by representing an image pair as a structured composition of its regions. In contrast, existing methods focus on whole image analysis, while implicitly relying on region-level understanding. We extend the intra-image notion of a scene graph to inter-image, and propose a novel task of Distortion Graph (DG). DG treats paired images as a structured topology grounded in regions, and represents dense degradation information such as distortion type, severity, comparison and quality score in a compact interpretable graph structure. To realize the task of learning a distortion graph, we contribute (i) a region-level dataset, PandaSet, (ii) a benchmark suite, PandaBench, with varying region-level difficulty, and (iii) an efficient architecture, Panda, to generate distortion graphs. We demonstrate that PandaBench poses a significant challenge for state-of-the-art multimodal large language models (MLLMs) as they fail to understand region-level degradations even when fed with explicit region cues. We show that training on PandaSet or prompting with DG elicits region-wise distortion understanding, opening a new direction for fine-grained, structured pairwise image assessment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the new task of Distortion Graph (DG), aimed at region-based pairwise image comparison and assessment. The authors argue that existing methods emphasize global analysis while neglecting region-level understanding, and therefore contribute: (1) a region-level dataset, PANDASET, comprising 528K image pairs and 15 distortion types; (2) a three-difficulty benchmark, PANDABENCH; and (3) the PANDA architecture, which employs a DETR-style decoder to predict inter-region comparative relations, distortion types, severities, and quality scores. Experiments show that PANDA substantially outperforms existing MLLMs on PANDABENCH (Tables 2-4), and the generated DGs can serve as chain-of-thought prompts to improve GPT-5 Mini’s accuracy by about 15% (Fig. 4).

### Strengths
The core strengths of this work lie in its clear motivation and the comprehensive construction of an ecosystem for the task. Shifting image quality assessment from a global to a structured, region-level representation is a valuable contribution; Fig. 2 compellingly demonstrates failure cases of current MLLMs in understanding region-level distortions. The paper provides a formal definition of DG, a large-scale dataset (2,200 images), a multi-difficulty benchmark (Easy/Medium/Hard), and an efficient baseline model (0.028B parameters). The experimental design is thorough, including systematic comparisons of open- and closed-source MLLMs, ablation studies, and validation of DG as an effective CoT prompt. PANDA’s SOTA performance across all settings (e.g., distortion classification accuracy of 0.78 under the Easy setting) substantiates the method’s effectiveness.

### Weaknesses
The most critical limitation of the paper is a circularity issue arising from the construction of the “ground-truth” labels. While the paper aims to propose a fine-grained assessment paradigm superior to existing evaluators, the two most central label types in PANDASET - region-level quality scores and comparative relations (e.g., “slightly better”) - are entirely derived from the existing TOPIQ model: quality scores directly adopt TOPIQ’s full-reference scores, and comparative relations are obtained by thresholding the differences between TOPIQ scores (p. 7; thresholds ±[0.1, 0.3), etc.). This implies that PANDA is essentially learning to imitate TOPIQ’s outputs rather than human perceptual decisions. The paper offers no human annotations to validate the perceptual validity of these automatically generated labels, which weakens the claim that DG can serve as a new evaluation standard. Moreover, the conclusion that “MLLMs cannot understand region-level degradations” (p. 2) may be overstated: the experiments more clearly show that MLLMs, in zero-shot settings, struggle with this novel out-of-domain task (Tables 2-4 indicate near-random performance across all MLLMs), whereas PANDA is trained specifically on PANDASET. Such a comparison is not entirely fair for assessing the fundamental understanding capabilities of MLLMs.

### Questions
1. Given that the ground-truth labels for comparative relations and quality scores are entirely dependent on the TOPIQ model, did the authors conduct any human studies to verify that TOPIQ’s region-level scores and their thresholding indeed align with human fine-grained perceptual preferences? If PANDA is merely imitating TOPIQ, how can we ensure it does not inherit TOPIQ’s perceptual biases?

2. The core comparison mechanism of the PANDA decoder allows region features from one image to query features of the other entire image (Eq. 6). Why choose this “region-to-global” attention rather than a more direct “region-to-region” comparison?

3. Fig. 4 shows that using DG as a CoT prompt improves GPT-5 Mini’s performance, yet its original zero-shot accuracy is very low (e.g., comparative accuracy of only 0.31 under the Easy setting, Table 2). Is this improvement due to emergent reasoning capabilities of the MLLM, or is it simply copying answers from PANDA-generated prompts?

4. All models experience a sharp performance drop on PANDABENCH-Hard (PANDA’s distortion accuracy falls from 0.78 to 0.27, Tables 2-3). Is the main cause of this drop that SAM fails to align regions in images with mixed distortions, or that the PANDA decoder cannot effectively compare two inputs both containing complex mixed distortions?

### Soundness
1

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a novel paradigm for comparing the quality of image pairs, particularly those with structured region compositions. The core contribution is the distortion graph, a structure designed to represent diverse distortion information, including type, severity, comparison, and quality score. To support this, the authors introduce a new dataset, PANDASET, used to train a graph prediction network. The network is designed in a DETR fashion, with strong backbones (DINOv2 and SAM) and separate decoders to predict the distinct components of the distortion graph. This graph is then used as a structured input to LMMs to generate region-specific distortion summaries for the input image pairs. Experiments demonstrate that this approach significantly improves the region-wise distortion understanding of current LMMs.

### Strengths
* The paper is well-written, clearly organized, and easy to follow. The concepts are introduced logically.
* The proposed distortion graph is a novel and intuitive method for structuring the complex task of regional quality comparison, moving beyond simple scalar scores or text descriptions.
* The proposed model architecture is well-motivated and thoughtfully designed, intelligently combining powerful pre-trained backbones (DINOv2, SAM) with a DETR-style architecture suited for object-centric prediction. The use of separate decoders for different graph components is a logical design choice.
* The ability to feed the DG outputs directly into an LMM to generate detailed, region-specific summaries is a practical and valuable application, bridging the gap between quantitative metrics and human-understandable feedback.
* The reported experimental results are impressive, showing a significant improvement in detailed distortion understanding compared to baseline LMMs.

### Weaknesses
* **Insufficient Discussion of Related Work:**
    The paper omits a discussion of closely related work in image quality grounding [a, b]. These prior works also aim to localize and describe quality issues, sharing a similar motivation. The paper would be strengthened by citing these works and clearly articulating how the proposed distortion graph paradigm differs from and improves upon existing grounding-based IQA methods. This context is crucial for accurately positioning the paper's contribution within the field.

* **Limitations of the PANDASET Dataset:**
    The newly collected PANDASET dataset presents potential limitations.
    1.  **Scale:** At a reported 2,000 images, the dataset is relatively small, which raises concerns about model overfitting both to the semantics and distortion types.
    2.  **Synthetic Distortions:** The dataset exclusively contains synthetic distortions. This reliance on synthetic data creates significant concerns: (1) The model may not generalize well to the diverse, subtle, and complex artifacts found in real-world images (e.g., camera noise, motion blur, compression artifacts from unknown codecs). (2) The model might be overfitting to the specific synthetic distortion types included in the dataset, potentially failing to identify or compare unseen distortion types.  
    The authors should discuss these limitations and provide experiments (even on a small scale) to test the model's robustness to real-world or unseen distortions.

* **Fairness of Experimental Comparison:**
    The experimental comparison with general-purpose LMMs may be unfair, potentially exaggerating the proposed method's superiority. The proposed network is specifically fine-tuned on the PANDASET dataset, while the baseline LMMs are evaluated in a zero-shot inference setting without task-specific training. The observed significant improvements might primarily stem from this specialized training on the target data distribution.
    It might be better to include this discrepancy in training protocols and discuss it as a confounding factor in the results analysis.

[a] Q-Ground: Image Quality Grounding with Large Multi-modality Models. Chen et al., ACM MM 2024.  
[b] Grounding-IQA: Grounding Multimodal Language Model for Image Quality Assessment. Chen et al., arxiv 2024.

### Questions
Please see weakness points above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Distortion Graphs (DG) that enable regional distortion analysis through topological representations of image regions. The authors propose a region-level training dataset, an evaluation benchmark, and an efficient architecture for generating distortion maps. Extensive experiments demonstrate the effectiveness of Panda in regional distortion analysis.

### Strengths
1. Regional distortion evaluation through DG is reasonable and effective, and this is the first explicit pairwise regional distortion assessment.
2. The proposed dataset PandaSet and benchmark PandaBench can promote development in the related research community.
3. Comparisons with existing MLLM approaches, including both open-source and closed-source models, support the effectiveness of the proposed regional distortion analysis.
4. The paper contains rich content, including numerous figures and tables that improve clarity.
5. Supplementary materials provide additional examples and code, which further strengthen the solid.

### Weaknesses
1. The discussion on region-aware understanding in IQA is insufficient. Prior works such as [1] and [2] should be included to improve motivation and task background.
2. The evaluation is mainly conducted on PandaBench, which limits the ability to fully assess performance. It is recommended to expand the evaluation to more general datasets or conduct user studies to validate generalization further.



[1] Q-Ground: Image Quality Grounding with Large Multi-modality Models

[2] Grounding-IQA: Grounding Multimodal Language Model for Image Quality Assessment

### Questions
1. More analysis and discussion of IQA methods that incorporate regional understanding.
2. Broader evaluation to verify generalization capability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel task called "distortion graph", which represents regional-level quality comparison between image pairs as a structured graph. In this graph, nodes encode regional attributes such as distortion type, severity, and quality score, while edges represent comparative relationships. To support this task, the authors construct the PandaSet dataset and the PandaBench benchmark, and design an efficient model named Panda. Experiments show that existing multimodal large models perform poorly on regional distortion understanding, while Panda significantly outperforms them.

### Strengths
1. It creatively extends the concept of scene graphs to the field of image quality comparison, defining a structured and interpretable representation method.
2. The paper's figures are highly expressive, clearly illustrating both the motivation and performance of the work.
3. The meticulously constructed PandaSet dataset contains over 500,000 image pairs with 15 distortion types and region-level annotations, demonstrating substantial scale.

### Weaknesses
1. Although the paper introduces the "Distortion Graph" structure, it essentially extends Scene Graph to image-pair comparison tasks. Meanwhile, region-level understanding has already been explored in recent IQA studies such as grounding IQA.

2. PandaSet builds upon PSG and Seagull-100w without providing genuinely new real-distortion data, relying solely on synthetic augmentations.

3. Comparing the specially-designed Panda model with MLLMs not fine-tuned for regional distortion tasks makes the conclusion "MLLMs perform poorly" appear unfair due to task mismatch.

4. Ablation studies remain insufficient, lacking evidence of how existing open-source vision models (Qwen, LLava, Kimi-vl) would perform after fine-tuning on the authors' dataset.

5. The paper provides limited analysis of the token-pool module, region-alignment mechanism, and loss-weight balancing strategies.

### Questions
1. If the visual encoder were replaced by a fine-grained, structure-preserving model like DeepSeek-OCR, would it be capable of capturing the complex scene relationships in the images? We look forward to the authors’ insightful answer.

2. Does the stability of SAM-based segmentation affect the precision of region-level alignment?

3. Is the proposed dataset organized in a manner analogous to the Tree-of-Thoughts (ToT) approach? The authors are requested to provide a comparative analysis, along with an evaluation of the associated training and inference time costs.

4. Is this dataset inherently more suitable for video understanding scenarios by nature?

### Soundness
3

### Presentation
3

### Contribution
2
