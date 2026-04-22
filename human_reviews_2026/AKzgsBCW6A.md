# MagicMirror: A Large-Scale Dataset and Benchmark for Fine-Grained Artifacts Assessment in Text-to-Image Generation

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Text-to-image (T2I) generation has achieved remarkable progress in instruction following and aesthetics. However, a persistent challenge is the prevalence of physical artifacts, such as anatomical and structural flaws, which severely degrade perceptual quality  and limit application.
Given the diversity and complexity of these artifacts, a systematic and fine-grained evaluation framework is required, which is lacking in current benchmarks.
To fill this gap, we introduce MagicMirror, a comprehensive framework for artifacts assessment. 
We first establish a detailed taxonomy of generated image artifacts. Guided by this taxonomy, we manually annotate MagicData340K, the first human-annotated large-scale dataset of 340K generated images with fine-grained artifact labels. Building on this dataset, we train MagicAssessor, a Vision-Language Model (VLM) that provides detailed assessments and corresponding labels. To overcome challenges like class imbalance and reward hacking, we design a novel data sampling strategy and a multi-level reward system for Group Relative Policy Optimization (GRPO). Finally, we leverage MagicAssessor to construct MagicBench, an automated benchmark for evaluating the image artifacts of current T2I models. Our evaluation with MagicBench reveals that despite their widespread adoption, even top-tier models like GPT-image-1 are consistently plagued by significant artifacts, highlighting artifact reduction as a critical frontier for future T2I development.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MagicMirror, a large-scale benchmark designed to detect fine-grained artifacts in text-to-image (T2I) generation. The main contribution lies in the creation of MagicData340K, a comprehensive dataset annotated with artifact categories, and the development of MagicAssessor, a vision-language model (VLM) trained to evaluate these artifacts. Finally, the authors present MagicBench, an automated evaluation framework built on MagicAssessor. Experimental results show that their method outperforms existing VLMs in artifact classification and provides valuable insights for future T2I model improvement.

### Strengths
* The paper addresses an important but underexplored aspect of T2I evaluation, which goes beyond traditional benchmarks that mainly focus on object presence or semantic alignment.
* The data collection pipeline is well-structured and clearly documented (Figure 4). The use of human annotations, chain-of-thought rationales, and multi-level labeling provides a rich and reliable dataset.
* The experiments are comprehensive, comparing MagicAssessor against both open- and closed-source models. The results demonstrate solid improvements, and the analysis of different artifact categories is informative.

### Weaknesses
* While the paper highlights the novelty of the proposed dataset, it would be great to see a clearer comparison with existing benchmarks such as Norma T2I benchmark: GenEval, T2I-CompBench, or Human preference benchmark: Pick-a-Pic, or HPSv2. Some statistics (e.g., dataset size, label diversity, or prompt length) could help readers better understand where MagicData340K stands out.
* The overall accuracy of MagicAssessor, although clearly better than other models, is still on the lower side (some are only ~ 0.3), as shown in Table 2, including element interaction, human and animal anatomy, and object morphology. It would be nice if the authors could discuss how this might affect its use as a reward model in improving T2I systems. Even a brief analysis here would make the potential impact clearer.
* The ablation results and design analysis (especially regarding GRPO and multi-bucket sampling) don’t entirely support the claim that the proposed design is optimal. For instance, in Table 4 (and Table 6 in the appendix), removing multi-bucket sampling improves recall on certain categories like irrational element interaction, while still maintaining competitive scores elsewhere. These patterns suggest that the trade-offs might be more nuanced than described. It would really help if the authors could add more discussion or qualitative examples showing what kinds of issues the proposed design actually solves, which would make the argument much more convincing.

---

**Overall**:
This paper makes a meaningful and timely contribution by introducing a high-quality dataset and benchmark focused on fine-grained artifact detection, which is a topic that has been largely overlooked. Even though the work doesn’t yet show how MagicAssessor directly enhances T2I model training, its methodological thoroughness and dataset quality make it a valuable step forward. Overall, I lean toward accepting the paper.

### Questions
The questions are mainly about weaknesses:

* Could authors provide a more explicit quantitative comparison between MagicData340K and existing T2I datasets (e.g., GenEval, T2I-CompBench, Pick-a-Pic, HPSv2)?
* How might the relatively low artifact classification accuracy affect downstream improvements if MagicAssessor were used as a reward model?
* Could authors provide more detailed or qualitative analysis in the ablation section to clarify why the proposed GRPO-based approach works best?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper focuses on evaluating artifacts in text-to-image generation. The authors create a human annotated dataset of generated images containing artifacts and discriminate between different types of artifacts creating a taxonomy. They next propose to fine-tune a model, namely Qwen2.5-VL-7B, via SFT on Chain-of-thought traces extracted from GPT and next GRPO via a multi-reward objective. They show that their fine-tuned model is more accurate in predicting fine-grained artifacts in contrast to off-the-shelf models. Finally, they benchmark different text-to-image generation models on their dataset using their full method.

### Strengths
1. The paper presents a human annotated large dataset for artifact evaluation in image generation. This is an important contribution for the community, and a core contribution of the paper is that they also create a taxonomy of artifacts where they target fine-grained evaluation.
2. The authors propose a well thought pipeline for fine-tuning a model on artifact evaluation. Off-the-shelf models do not perform well in recognizing artifacts, as these artifacts might be out of their training distribution. The authors show the improvement that their fine-tuning approach offers on Qwen2.5-VL-7B.

### Weaknesses
1. The novelty of the fine-tuning approach for artifact detection, which the authors claim that is one of the main contributions of the paper, is limited. The main novel components of the approach is the data sampling strategy, which is very similar to traditional ML techniques for data imbalances but adapted to the specific task, and the reward combination which is again similar to multi-reward objectives but again adapted to the specific task with specific heuristics derived by the taxonomy of the artifacts.
2. Crucially, the authors do not mention very important details for the creation of their dataset in order to be able to trust the approach and the benchmark. Important details include the inter-annotator agreement and how they are handling disagreements between annotators. Moreover, the paper mentions that one of the artifact categories is Irrational Element Attributes. The authors do not discuss how they discriminate between non factual elements that have been asked in the prompt so they are correctly generated and ones that are not correct. It is very common that prompts include irrational attributes for subjects or objects.
3. The dataset has been created by generating images from specific models, including FLUX.1-dev/schnell, which is one of the best performing models on this benchmark in Table 3. How much does this bias the results? Could it be that images generated by e.g., GPT or Janus are more out of distribution for the fine-tuned evaluator? A human evaluation should be conducted here in order to measure correlation with human ratings and understand the bias of the evaluator.
4. MagicAssessor is trained on a dataset generated by a specific suite of T2I models. Future T2I models may exhibit entirely new or different types of artifacts. How well is the model expected to generalize to these unseen failure modes? Does the "Other Irrationalities" category, which is very small, risk becoming a bucket for many new artifact types, limiting the model's fine-grained utility over time?
5. Not a reason for rejection, but I find it bad that all of related work has been pushed to the Appendix. At least a synopsis should be transferred to the main paper.

### Questions
Based on the above weaknesses:
1. Can you explain the details of the dataset for Weakness point 2?
2. Have you analysed in any way the bias as mentioned in Weakness point 3?
3. Can you address the questions of Weakness point 4?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a evaluation pipeline for artifacts in text-to-image generation, including human-annotated large-scale dataset of
340K generated images with fine-grained artifact labels, a VLM to assess the generated images, and a benchmark to evaluate current T2I models. It focuses on a critical problem in current text-to-image generation models, and makes great effort to construct a large scale human annotated dataset.

### Strengths
1. This paper focuses on the artifacts problem, which is critical and persistent issue in text-to-image generation. This paper provides a whole pipeline with dataset collection, model training, and benchmark construction. The efforts may potentially benefit future works on addressing the artifacts of text-to-image generation.

2. Compared with previous datasets of artifacts, this paper provides more detailed category of artifacts, and propose different level of labeling. This structured categorization provides more precise evaluation signals and enables nuanced analyses of model weaknesses across different artifact types.

3. The proposed dataset is substantial in both size and diversity, with 340k human annotated images, and covers many different types of artifacts in fine-grained levels.

### Weaknesses
1. The motivation of describing the artifacts and their location in natural language is unclear. It would be more intuitive to localize artifacts spatially, by marking regions with bounding boxes or segmentation masks. Some artifacts may not be precisely explained merely by natural language, and sometimes their locations may not be easily identified. For example, when the artifacts are only part of a large object, in background region, or texture level distortions.

2. While the dataset provides detailed artifact descriptions, it seems that the framework mainly uses them for binary or categorical classification rather than for region-aware evaluation or feedback. The paper could better leverage these fine-grained signals. For example, to guide targeted model correction or localized reward shaping rather than aggregating them into global scores as in previous paper [1].

3. The distribution of artifact categories in MagicData340K is highly imbalance as shown in Fig 5. 

4. The ablation results in Table 4 indicate that removing individual components only slightly affects performance, suggesting that the contribution of each design element is not significant. This weakens the motivation to introduce these strategies.

5. The paper provides limited information about the human annotation process. Key details such as the number of annotators and inter-annotator agreement are missing. Is there a validation process in the annotation, or only one annotator for each sample? Without quantitative measures of consistency, it is difficult to assess the reliability of the large-scale annotations.

[1] Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation. (CVPR 2025)

### Questions
please see above

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
The paper introduces a complete stack for artifact assessment in T2I: (1) a fine-grained taxonomy of artifacts and a 340K human-annotated dataset called MagicData340K, (2) a specialized assessor named MagicAssessor based on Qwen2.5-VL-7B fine-tuned using SFT followed by GRPO with custom sampling and rewards, which produces labels and rationales, and (3) an automated benchmark named MagicBench that applies MagicAssessor to evaluate a set of T2I models across human, animal, object, and interaction categories. Results show that MagicAssessor clearly outperforms general-purpose VLMs in artifact recognition, and MagicBench indicates that even leading generators such as GPT-image-1 still produce significant artifact rates.

### Strengths
S1: The fine-grained and hierarchical taxonomy with large-scale human annotation is impressive. The L1 to L3 scheme, moving from normal versus artifact to anatomy, attributes, and interaction, then to hand-level and similar details, along with curated guidelines, represents a clear improvement over coarse plausibility dots.

S2: Looks like the dataset is of high quality.

S3: The design of the artifect-based reward looks good to me.

S4: The paper looks really nice, very well-written.

### Weaknesses
W1: The paper describes expert guidelines and oversight but does not report inter-annotator agreement such as kappa or Fleiss scores, nor re-label consistency metrics, despite the subjective nature of fine-grained labels like element overlap versus low-quality area. I recommend the authors include this in the rebuttal.

W2: A subset of rationales is synthesized by GPT-4o from human descriptions to bootstrap chain-of-thought. While pragmatic, the paper does not specify the fraction of CoT data or analyze how GPT-4o phrasing may bias the learned reasoning style. If the evaluator later favors assessor-like wording, this may create a stylistic loop. It is important to examine the CoT steps. If possible, the authors should add this point during rebuttal.

W3: ​​The consistency reward is meant to reduce reward hacking, but the evidence is mainly aggregate F1 changes. There is no audit of failure cases such as verbose rationales that still mislabel. GRPO can overfit to reward signals shaped by noisy heuristics like formatting or high-level label focus. No analysis on this is given in the paper.

W4: MagicData340K includes images from several generators such as FLUX, Kolors, SD3 and SD3.5, Midjourney, and internal sources, but it is unclear whether generator identity leaks into labels, for example via artifacts that appear recognizably FLUX-like. This may let MagicAssessor rely on style-based priors. More broadly, the test set of 17366 samples is not broken down clearly in terms of size relative to training split or coverage across generator and style diversity.

W5: Although the paper overall looks very nice, figure 2 does not match the style of the other figures. It appears too simple and a bit unattractive.

### Questions
See the previous weakness. 

Could the authors report inter-annotator agreement (e.g., Cohen’s κ or Fleiss’ κ) for a subset of the dataset?

What proportion of CoT rationales are human-authored vs. GPT-4o-synthesized, and how do their linguistic or reasoning styles differ? I guess some of CoT should be checked instead of completely relying on the automatically generated.

How many unique T2I generators are represented in the dataset, and is there any overlap between generators used for training and those used in MagicBench?

Did the authors observe cases where GRPO-trained models produced plausible but factually incorrect rationales to maximize rewards?

### Soundness
4

### Presentation
4

### Contribution
3
