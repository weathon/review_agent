# Exploring Instruction Data Quality for Explainable Image Quality Assessment

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
In recent years, with the rapid development of powerful multimodal large language models (MLLMs), explainable image quality assessment (IQA) has gradually become popular, aiming at providing quality-related descriptions and answers of images. To achieve this goal, recent methods seek to construct a large-scale instruction tuning dataset to empower the MLLM with quality perception ability following the well-known scaling law. However, a large amount of instruction tuning data may cause substantial computational costs and redundant data, which in turn will cause harm to the performance of the model. To cope with this problem, in this paper, we challenge the scaling law and systematically investigate the role of data quality of the instruction tuning dataset for explainable IQA. Using a powerful pre-trained MLLM, we first investigate the changes in model performance after fine-tuning with different sizes of instruction tuning data. We find that selecting a subset of the data set randomly using an appropriate ratio can even lead to better results than training with the entire instruction tuning dataset, demonstrating the redundancy of current explainable IQA instruction tuning data. Beyond randomly sampling a subset, we propose a clustering-based data selection framework with three stages: clustering feature extraction, cluster quota allocation, and cluster sampling strategy. Then we systematically analyze the choices of each stage and propose a simple but efficient data selection method IQA-Select for explainable IQA. The experimental results demonstrate that IQA-Select can achieve 102.1% and 103.7% performance of full fine-tuning using only 10% selected data in Q-Bench and AesBench respectively, significantly reducing computational costs while achieving better performance. We hope that our paper can provide a new perspective for future research on exploring the quality of instruction tuning data for explainable IQA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a critical and under-explored issue in the field of Explainable Image Quality Assessment (Explainable IQA): the impact of instruction data quality and scale on the performance of Multimodal Large Language Models (MLLMs). The work challenges the prevailing "more data is better" paradigm by systematically investigating the feasibility of constructing a smaller yet more effective data subset (a coreset) through principled data selection. The authors propose a three-stage data selection framework named IQA-Select and demonstrate through extensive experiments that fine-tuning with just 10% of the selected data can surpass the performance of using the full dataset on benchmarks like Q-Bench and AesBench.

### Strengths
1.	Clear and Pragmatic Motivation: The paper rightly points out the potential drawbacks of the field's heavy reliance on "scaling laws," namely the immense computational overhead and data redundancy. The direction of exploring the importance of data quality is forward-looking and holds significant practical value.

2.	Systematic and Comprehensive Ablation Studies: To construct the optimal data selection pipeline, the authors conduct exhaustive experiments and ablation studies for each stage of the framework (feature extraction, quota allocation, and sampling strategy). For instance, they compare nine different feature types in the feature extraction stage and explore eleven strategy combinations for quota allocation.

### Weaknesses
1.	Limited Novelty: The core contribution of this paper lies more in "problem exploration" than in "methodological innovation." The proposed IQA-Select framework is fundamentally a clustering-based data selection pipeline, and its constituent components (e.g., clustering, quota allocation based on density/transferability/IRS, and sampling via SVD/PCA) are largely existing or slightly modified techniques from the data selection literature. The authors themselves candidly define it as a "pipeline." 

2.	Superficial Analysis of "Performance Degradation": The paper observes that fine-tuning on the full dataset leads to inferior performance compared to using a subset, attributing this to "model forgetting" and "data redundancy." The paper fails to provide a deeper analysis. For example, what are the specific characteristics of the redundant data (e.g., low-quality, repetitive, overly simplistic)? A more in-depth investigation would make the claims more convincing.

3.	Dependence on a Strong Pre-trained Model: The entire framework and its findings are heavily dependent on a powerful pre-trained MLLM (InternVL3-Instruct). It is unclear whether the conclusion "less data is better" would still hold if a weaker base model were used. Arguably, large-scale data might still be essential for less capable models. The paper offers limited discussion on this aspect of generalizability.

4.	Trade-off Between Complexity and Practical Gain: The IQA-Select framework is relatively complex, involving multiple steps such as feature extraction, clustering, multi-metric calculation, and sampling. Given that random sampling of 20% of the data already achieves strong performance, the paper needs to more clearly justify whether the additional benefit (an approximate 1-2% performance gain) warrants the introduction of such a complex framework.

### Questions
1.	In the discussion of the final method combination in Section 4.2.5, the paper states that the final method uses "transferability and instruction relevance score." However, this appears to contradict the results in Table 2, where the best-performing combination is "Transferability & Density" (II-(8)), and also the final reported combination (I-(6) + II-(8) + III-(2)). Could you please clarify this discrepancy?

2.	The framework introduces several hyperparameters, such as the number of clusters (N) for clustering and the similarity threshold (τ) for calculating transferability. How were these hyperparameters selected? How sensitive is the final performance of the framework to variations in these hyperparameters?

3.	Do you believe the IQA-Select method can be generalized for instruction data selection in other vision-language tasks, such as VQA or Image Captioning? If so, which parts of the framework would require the most significant adjustments when applied to different tasks?

### Soundness
2

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
This paper introduces an efficient data selection method from redundant full-scale data for explainable image quality assessment (IQA).
The proposed method, namely IQA-Select, consists of three stages: clustering feature extraction, cluster quota allocation, and cluster sampling strategy.
The cluster features are divided into model-related features and model-independent features, and a total of 9 combinations of features have been investigated.
Similarly, for cluster quota allocation and cluster sampling strategy, 11 allocation strategies and 3 sampling methods have been investigated respectively.
Using the baseline model InternVL3-Instruct-8B, the proposed IQA-Select demonstrated superior IQA performance using only 10% of the subset on both the Q-Bench and AesBench benchmarks, compared to full dataset fine-tuning.

### Strengths
The authors explore multiple configurations per stage to select the optimal combination across three stages:
- Clustering feature extraction: 6 model-related combinations from 4 distinct features; 3 model-independent combinations from 3 distinct features.
- Cluster quota allocation: 11 strategies from 4 features.
- Cluster sampling: greedy MMD, SVD, and PCA.

Experiments on two benchmarks (Q-Bench and AesBench) improve reliability.

### Weaknesses
As the performance of general-purpose baseline models continues to improve, I agree that optimizing the fine-tuning dataset to adapt a model for a downstream task is an effective strategy. However, the manuscript appears to lack sufficient concrete evidence to substantiate this claim.

Generalization
- Only InternVL3-Instruct-8B is tested, so generalization to other models is uncertain.
- Since the work concerns data curation, multi-model runs are needed.
- Additional suggestion: cross-validation would strengthen the generalization capability.

Incomplete metrics
- Table 3 omits AesI while AesBench includes four categories (AesA, AesE, AesP, and AesI).
- The omission is not explained.

Unsubstantiated claim
- The assertion that “fine-tuning the MLLM with all dataset may inversely cause the MLLM to forget the knowledge it learned before” is repeated without theoretical or empirical support, especially with respect to the IQA task.

Minor issues
- L450: table hyperlink error.

### Questions
No questions

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper studies instruction data quality for explainable image quality assessment (IQA) with multimodal LLMs. Using InternVL3-Instruct as the base model, the authors first show that less can be more: randomly fine-tuning on a small fraction of Q-Instruct can match or even exceed full-data fine-tuning. Building on this, they propose IQA-Select, a clustering-based selection pipeline with three stages. With only 10% of the data, IQA-Select reportedly attains 102.1% of full-data performance on Q-Bench and 103.7% on AesBench, and the best variant yields the top overall Q-Bench score among 10% subsets.

### Strengths
1. The proposed data selection method is effective: using just 10% of the SFT data, it outperforms the pretrained baseline and other methods.
2. The diversity, IRS, and Trans metrics that gauge SFT data quality provide compelling evidence of its effectiveness.

### Weaknesses
1. The comparison set is dated. Please include recent VLM baselines, e.g., VisualQuality-R1, to ensure a fair, up-to-date evaluation.
2. Expand experiments across diverse SFT datasets. Q-Bench may contain redundant samples, so cross-dataset validation would strengthen the conclusions.
3. Add qualitative case studies illustrating which data types most effectively boost VLM performance, with before/after outputs where possible.

### Questions
See weakness.

### Soundness
3

### Presentation
2

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
This paper questions the prevailing “scale-first” mentality in instruction-tuning for explainable image quality assessment (IQA).  Starting from the observation that InternVL3-8B fine-tuned on the full 200K Q-Instruct set barely outperforms the pretrained checkpoint, the authors systematically reduce the training pool and discover an inverted-U curve: performance plateaus at roughly 20 % of the data and even the extreme 5 % random subset matches the full-set accuracy, revealing massive redundancy.  Building on this insight, they design IQA-Select, a three-stage data-curation pipeline that (i) represents every instruction by fusing multi-level MLLM hidden states with vision–text embeddings, (ii) allocates cluster-level quotas via a combination of transferability and density, and (iii) harvests the most representative samples within each cluster through SVD leverage scores.  With only 10 % of the original data, the method attains 79.4 % overall accuracy on Q-Bench (102.1 % of full-data tuning) and 61.1 % on AesBench (103.7 % of full-data tuning) while cutting GPU hours by an order of magnitude, thereby providing the first systematic evidence that careful curation can outperform brute-force scaling in explainable IQA.

### Strengths
The contribution is original in that it is the first work to interrogate—and empirically refute—the scaling law inside the IQA instruction-tuning niche, and it delivers a principled, reproducible pipeline to exploit this observation; the experimental design is thorough, encompassing roughly 300 ablations across nine feature families, eleven quota strategies and three intra-cluster samplers, all trained under identical LoRA hyper-parameters and evaluated on two public benchmarks with consistent trends; the paper is clearly written, with precise mathematical definitions of density, IRS and transferability, intuitive figures that visualise the selected feature-space coverage, and ample discussion of design choices, making the approach immediately actionable for practitioners; finally, the work is significant because it transforms a costly data-collection problem into a data-curation opportunity, offering the community a ten-fold reduction in training cost without sacrificing—and often slightly improving—accuracy, and it opens a new research direction that shifts the focus from “how to generate more” to “how to select better” instruction data for visual-quality tasks.

### Weaknesses
**Model-scale scaling law is unexplored:**
All conclusions are derived from a single 8B-parameter model. The redundancy hypothesis may not hold for smaller (≤ 4 B) or larger (≥ 30 B) models whose capacity, memorisation behaviour and forgetting dynamics differ. I suggest authors to run identical selection pipelines on at least two more scales (e.g., InternVL2-2B and InternVL2-40B). Report whether the “5 % = 100 %” trend persists and whether IQA-Select’s relative gain grows or shrinks.


**Task-scope is narrow:**
The method is only evaluated on IQA and aesthetics. It is unclear whether the designed features (IRS, distortion-related IQA scores, etc.) generalise to other tasks. Authors can try to validate the conclusions in reasoning task using the training and test data of M3CoT.


**Missing baselines from recent data-selection literature:**
No comparison with other data-selection pipelines. I suggest authors to include at least one more data-selection baseline; otherwise it is hard to validate the effectiveness of the proposed method.


**Minor LaTeX style issue:**
Citations use \cite instead of \citep, producing “Q-Bench Wu et al. 2023” rather than “Q-Bench (Wu et al. 2023)”. Please conform to standard ICLR format.

### Questions
Please see weakness.

### Soundness
3

### Presentation
3

### Contribution
3
