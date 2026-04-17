# A High Quality Dataset and Reliable Evaluation for Interleaved Image-Text Generation

- Decision: Accept (Poster)
- Scores: 6, 2, 8, 8

## Abstract
Recent advancements in Large Multimodal Models (LMMs) have significantly improved multimodal understanding and generation.
However, these models still struggle to generate tightly interleaved image-text outputs, primarily due to the limited scale, quality and instructional richness of current training datasets.
To address this, we introduce \textbf{InterSyn}, a dataset that features:
(1) large scale, comprising 1.8M multimodal samples;
(2) high quality, supported by our proposed \textbf{Self-Evaluation with Iterative Refinement (SEIR)} method for rigorous automated quality refinement;
(3) rich instructional diversity, ensured through diverse well-designed question templates, based on human preferences and covering a 3500-topic hierarchy.
These characteristics make InterSyn particularly well-suited for training LMMs in interactive image–text generation capabilities.
To evaluate the capabilities, we propose \textbf{SynJudge}, a reliable automatic evaluator that aligns closely with human judge and outputs four interpretable scores: Text Content Completeness (TCC), Image Content Completeness (ICC), Image Quality (IQ), and Image–Text Synergy (ITS).
These scores are complementary, covering both content and quality as well as cross-modal interaction, thereby forming a comprehensive evaluation framework.
Experimental results on InterSyn subsets of up to 200K samples show that 25K–50K already yield substantial improvements, while scaling to 100K/200K brings further gains in TCC, ICC, and especially ITS, highlighting InterSyn’s:
(1) scalability, as performance consistently improves with more data;
(2) efficiency, as significant gains are achievable even with smaller subsets, making it accessible to researchers with varying computational resources.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes InterSyn, a large-scale dataset (1.8M single-turn, 50k multi-turn samples) and a Self-Evaluation with Iterative Refinement (SEIR) pipeline for generating high-quality interleaved image–text dialogues. It also introduces SynJudge, an automatic evaluator providing four interpretable scores:
Text Content Completeness (TCC)
Image Content Completeness (ICC)
Image Quality (IQ)
Image–Text Synergy (ITS)
Experiments show that models fine-tuned on subsets of InterSyn (25k–200k) exhibit consistent gains in instruction-following and synergy metrics without degrading general multimodal understanding benchmarks. SynJudge is validated against human ratings, showing strong correlation and reliability.

### Strengths
①Timely contribution addressing a real bottleneck. The lack of high-quality, instruction-following, interleaved image–text data is a genuine gap. InterSyn fills this niche with scale and structure.
②Systematic dataset generation method. SEIR introduces an iterative, self-corrective data generation framework (Question → Answer → Image refinement), demonstrating consistent quality improvements.
③Comprehensive evaluation framework. SynJudge formalizes multi-dimensional metrics beyond text-only or image-only correctness, introducing “synergy” as a critical dimension.
④Strong empirical coverage. Experiments span multiple model types (non-interleaved and interleaved generators), multiple data scales, and both human and automatic evaluations.
⑤High reproducibility commitment. The paper provides reproducibility details and plans to release data and evaluation tools, which is rare and commendable.

### Weaknesses
(A) Theoretical clarity and rigor of SEIR
The SEIR process (Question Refinement → Answer Refinement → Image Refinement) is well-described but lacks formal definition of convergence or stability. It’s unclear whether the refinement iterations always improve data quality or sometimes overfit to self-consistency artifacts. The “Markovian” assumption is stated but not justified—this ignores potential long-range dependencies across turns. The method claims to “ensure topic consistency and contextual inheritance,” yet no quantitative metrics for coherence or redundancy are provided.

(B) SynJudge validation may not prove true generalization
The human-alignment test (RMSE) is based on fine-tuning with human-annotated data, which might lead to data leakage between evaluator and benchmark distributions. The human set (9.6k test / 38.4k train) is relatively small compared to 1.8M total data, raising questions about generalization across unseen content domains. “95.4% ” sounds impressive, but no confidence intervals or inter-annotator agreement scores (e.g., Krippendorff’s alpha) are reported.

(C) Data authenticity and bias control
InterSyn heavily relies on AI-generated data with human-like templates. However, no explicit human-in-the-loop verification or bias filtering is described beyond early-stage question curation. SEIR may propagate LLM biases (e.g., Western imagery, gender stereotypes, or cultural homogeneity) through iterative reinforcement.

### Questions
1.How is convergence or stability of the SEIR refinement process defined and verified? Does SEIR always improve data quality, or could iterative self-consistency cause overfitting or redundancy?
2.What evidence supports the “Markovian” assumption in SEIR—have you analyzed long-range contextual dependencies? How is topic consistency or contextual inheritance quantitatively measured?
3.How do you ensure that SynJudge’s fine-tuning on human data does not cause data leakage? Given the small human set, how well does SynJudge generalize to unseen domains?
4.Are confidence intervals or inter-annotator agreement (e.g., Krippendorff’s α) reported for human validation?
5.What human-in-the-loop or bias-filtering mechanisms are used to control cultural or demographic bias in InterSyn?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces InterSyn, a large-scale interleaved multimodal dataset (1.8M samples) that addresses LMMs' limitations in generating tightly interleaved image-text outputs. Using the proposed Self-Evaluation with Iterative Refinement (SEIR) method, the authors ensure high-quality data with rich instructional diversity across a 3500-topic hierarchy. The paper also contributes SynJudge, an evaluation framework that assesses generation capabilities through four metrics: Text Content Completeness, Image Content Completeness, Image Quality, and Image-Text Synergy. Experiments demonstrate that even smaller subsets (25K-50K) of InterSyn yield substantial improvements in model performance, while scaling to larger subsets (100K-200K) brings additional gains, particularly in image-text synergy.

### Strengths
1. The proposed InterSyn dataset has a large scale and is curated with a carefully designed SEIR pipeline.
2. The experiments cover a wide range of generation models and judges.

### Weaknesses
1. Dataset Quality Limitations: The quality of the dataset is significantly constrained by the FLUX image generation model. A more robust data curation approach would involve collecting and filtering high-quality images from the Internet based on the generated questions and answers, rather than relying solely on synthetic image generation.

2. Clarity and Organization Issues:
  - The authors use about 1.5 pages to describe the SEIR with 13 formulations, despite it being a relatively straightforward process that could have been clearly illustrated in Figure 2. Unfortunately, Figure 2 is exceedingly difficult to interpret, featuring unexplained notations, complex organization, and lacking essential text labels.
  - Despite the excessive length of section 3.3, critical information is absent. The authors present SynJudge as a key contribution, yet fail to clearly define it. The term first reappears after Section 2 in Line 309 ("We fine-tune four generators... And evaluate with SynJudge"). It remains unclear whether SynJudge refers specifically to the Qwen-trained judge or also encompasses the four metrics (TCC, ICC, IQ, and ITS). Furthermore, details regarding these four metrics are omitted from the main text, compromising the paper's self-consistency. Essential benchmark information, such as sample size, is also entirely missing.

3. Problematic Experimental Results and Conclusions:
  - The finetuned models are evaluated exclusively on the proposed dataset (with minimal results in Table 2). It is unsurprising at all that finetuning on a small subset of this dataset can lead to performance improvement. A critical question remains: Can fine-tuning on InterSyn generalize to other related or out-of-distribution benchmarks?
  - Table 2: The authors claim that "the substantial gains in interleaved generation capabilities do not come at the cost of core understanding performance," yet the evidence suggests otherwise. Performance degradation on MMBench and MMMU (-1.5 to -2.5) appears evident.
  - Table 3: The assertion that multi-turn data improves performance is not well-supported by the results. For "single25k + multi25k," both models' scores are lower than or comparable to those with "single50k." Moreover, "multi50k" scores are even lower than "single25k + multi25k", contradicting the authors' claims.

### Questions
1. What is the performance of state-of-the-art models, such as GPT-5 (text) + GPT-Image-1 (image), or Gemini 2.5 Pro (text) + Nano Banana (image)?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces InterSyn, a large-scale, high-quality multimodal dataset designed to advance the training of multimodal LLMs for interleaved image-text generation. The primary contribution of this work is the creation of a comprehensive dataset comprising 1.8 million multimodal samples, which are refined using the proposed Self-Evaluation with Iterative Refinement (SEIR) method, thereby ensuring high data quality. This addresses major challenges in current datasets, including limited scale, poor quality control, and low interaction complexity. In addition, the paper proposes SynJudge, a robust evaluation framework that quantitatively assesses image-text synergy, content completeness, and quality across both modalities. Through extensive experiments, the authors demonstrate that InterSyn significantly improves the performance of multimodal LLMs in tasks involving instruction-following and complex, multi-turn dialogues. The results show that even with smaller subsets of the dataset, substantial performance gains can be achieved, underscoring both the scalability and efficiency of the dataset.

### Strengths
1. Interleaved multimodal generation is a promising capability for LLMs, and a few pioneer models can achieve it. An open-source, high-quality, large-scale, and comprehensive dataset and benchmark are still urgent. It is a significant contribution to the community.
2. The proposed InterSyn is large-scale and high-quality, with 1.8M samples spanning 3.5K topics, and includes 50k multi-turn dialogues.
3. The proposed SEIR method is well-designed and can ensure high quality of the dataset. The proposed Image-Text Synergy (ITS) score and SynJudge judge model are reliable. These modules have all been verified through human evaluation.
4. Extensive fine-tuning experiments demonstrate the effectiveness of the dataset across different dimensions. Models fine-tuned on InterSyn show consistent performance gains.

### Weaknesses
1. The SEIR method enhances image generation, but the current capabilities of text-to-image models still limit the visual fidelity. This is not an issue with the method itself, but rather with the models’ ability to generate highly precise and expressive images.
2. In the synthesis of multi-turn dialogues, there may be some redundancy between dialogue turns. It’s unclear whether the authors specifically controlled for this aspect of quality.

### Questions
1. While SynJudge is shown to be reliable, could there be any challenges or biases in how it aligns with human judgment?
2. Could you provide more details on how the SEIR method performs with different refinement depths？
3. The paper mentions that the dataset includes multi-turn dialogue data. Compared to single-turn dialogues, how can the effectiveness of multi-turn dialogues be demonstrated?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces InterSyn, a large-scale, high-quality dataset (1.8M samples, including 50K multi-turn dialogues) designed for instruction-following interleaved image-text generation. The dataset is constructed via a novel Self-Evaluation with Iterative Refinement (SEIR) pipeline, which iteratively improves question, answer, and image quality with automatic feedback loops. To evaluate multimodal generation, the authors also propose SynJudge, an automatic evaluator providing four interpretable metrics: Text Content Completeness (TCC), Image Content Completeness (ICC), Image Quality (IQ), and Image–Text Synergy (ITS). SynJudge demonstrates high correlation (95% agreement within 1 point) with human judgment. Experiments show that InterSyn substantially improves multimodal models’ interleaved generation capabilities. Fine-tuning on as few as 25K samples yields noticeable gains, and scaling to 200K produces further improvements in all metrics, especially ITS. SEIR-generated samples outperform all baselines—including GPT-4o+DALL-E and Gemini+Flux—on both human and automatic evaluation.

### Strengths
1. This paper is well-written and very easy to follow.
2. This paper has significant practical contribution. The introduction of InterSyn addresses a key bottleneck in multimodal AI: the lack of large-scale, high-quality, and instructionally rich datasets for interleaved image–text generation.
3. The proposed data generation pipeline SEIR and the evaluation criteria SynJudge is reliable and useful for future work.
4. This paper conduct strong experimental validation which proves the efficiency of this dataset.

### Weaknesses
I think this paper is very good and I don't find some obvious weakness from my perspective. But as a dataset work, this paper has limited discussion on ethical considerations. Given the scale of synthetic multimodal data and some data are collected from commercial software, the paper lacks discussion on potential biases, copyright issues, or misuse of synthetic images.

### Questions
No other questions.

### Soundness
3

### Presentation
3

### Contribution
3
