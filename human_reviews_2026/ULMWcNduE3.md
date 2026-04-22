# Synthesizing High-Quality Visual Question Answering from Medical Documents with Generator-Verifier LMMs

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Large Multimodal Models (LMMs) are increasingly capable of answering medical questions that require joint reasoning over images and text, yet training general medical VQA systems is impeded by the lack of large, openly usable, high-quality corpora. We present MedVLSynther, a rubric-guided generator-verifier framework that synthesizes high-quality multiple-choice VQA items directly from open biomedical literature by conditioning on figures, captions, and in-text references. The generator produces self-contained stems and parallel, mutually exclusive options under a machine-checkable JSON schema; a multi-stage verifier enforces essential gates (self-containment, single correct answer, clinical validity, image-text consistency), awards fine-grained positive points, and penalizes common failure modes before acceptance. Applying this pipeline to PubMed Central yields MedSynVQA: 13,087 audited questions over 14,803 images spanning 13 imaging modalities and 28 anatomical regions. Training open-weight LMMs with reinforcement learning using verifiable rewards improves accuracy across six medical VQA benchmarks, achieving averages of 55.85 (3B) and 58.15 (7B), with up to 77.57 on VQA-RAD and 67.76 on PathVQA, outperforming strong medical LMMs. A Ablations verify that both generation and verification are necessary and that more verified data consistently helps, and a targeted contamination analysis detects no leakage from evaluation suites. By operating entirely on open literature and open-weight models, MedVLSynther offers an auditable, reproducible, and privacy-preserving path to scalable medical VQA training data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces MedVLSynther, a rubric-guided generator verifier pipeline that synthesizes multiple-choice medical VQA items directly from PubMed Central figures, captions, and in-text references.

A context-aware LMM generator produces self-contained stems and mutually exclusive options under a strict JSON schema, while a multi-stage LMM verifier enforces essential gates, awards fine-grained points, and applies penalty criteria to ensure quality.

The resulting MedVLSynther-13K dataset (13,087 audited questions over 14,803 images, 13 modalities, 28 anatomical regions) is used to train open-weight LMMs with reinforcement learning using verifiable rewards, achieving state-of-the-art averages among open models across six benchmarks.

Ablations show both generation and verification are necessary, that more verified data helps (with diminishing returns beyond 5k), and contamination analysis finds no leakage from evaluation suites.

### Strengths
Originality: The explicit, rubric-driven generator–verifier loop that is both context-aware (image + caption + references) and schema-constrained is a fresh, auditable approach versus prior text-only or loosely filtered syntheses. The design of essential, fine-grained, and penalty criteria operationalizes “exam-quality” medical VQA in a reproducible way.

Quality: The empirical study is thorough, spanning dataset characterization, multi-benchmark evaluation, ablations on pipeline stages, scale, and generator/verifier choices, plus contamination analysis. The use of RL with verifiable rewards and consistent gains over strong medical LMMs strengthen the causal link between verified data quality and downstream performance.

Clarity and significance: The pipeline is clearly illustrated with stage-wise prompts/criteria and concrete acceptance rules, including a normalized score and high threshold. MedVLSynther-13K offers a practical, privacy-preserving resource that advances training for medical VQA and is likely to catalyze open, reproducible research.

### Weaknesses
Verification robustness: While multi-stage verification improves precision, the reliance on LMM verifiers introduces potential model biases and false accept/reject risks; human audits or inter-verifier agreement analysis are limited in the main text. Adding a small-scale human evaluation and measuring inter-model/verifier consistency would better quantify verification reliability.

Scope and coverage: Pre-filtering to Clinical imaging and Microscopy plus selected subtypes may bias question distribution (e.g., heavy digital photography share) and underrepresent complex radiology workflows. Expanding coverage to additional modalities/workflows and reporting performance stratified by difficulty and question archetype would improve generality claims.

Practical reproducibility/cost: Although models and data are open, the use of very large generators/verifiers (up to 108B) may hinder replication and scaling by typical labs. Providing cost/compute estimates, a “small-model” variant with trade-offs, and release of prompts/rubrics/scripts would enhance accessibility.

### Questions
What proportion of accepted items pass a small human audit, and how do human judgments correlate with the verifier’s fine-grained/penalty scores?

How sensitive are downstream gains to the chosen acceptance threshold τ and to the balance of question archetypes; could adaptive sampling increase difficult, clinically salient items?

Can you report per-modality and per-anatomy breakdowns of improvements and failure cases to guide targeted data augmentation or rubric refinement?

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
4

### Summary
This paper presents MedVLSynther, a rubric-guided generator-verifier framework for synthesizing medical VQA data from PubMed literature. The authors employ large multimodal models to generate multiple-choice questions from medical images, captions, and in-text references, followed by a three-stage verification process. The resulting dataset contains 13,087 questions over 14,803 images spanning 13 imaging modalities and 28 anatomical regions. Models trained on this data with reinforcement learning demonstrate improvements over baselines on six medical VQA benchmarks, achieving averages of 55.85 for 3B models and 57.56 for 7B models.

### Strengths
1. The paper introduces a generator-verifier architecture that separates generation and verification using different LMMs, enabling auditable quality control through a rigorous three-stage process of essential gates, fine-grained scoring, and penalty detection.

2. The context-aware generation approach incorporates not just image captions but also in-text reference paragraphs from the surrounding literature. 

3. The comprehensive rubric design with seven essential criteria, four to eight fine-grained criteria, and four penalty criteria, along with explicit weights and a machine-checkable JSON schema.

### Weaknesses
This paper is technically competent and presents a well-engineered pipeline. However, I have concerns about whether it makes a sufficient scientific contribution to a top-tier venue. Apart from the proposed datasets, what are the main insights people can get from this paper?

1. The paper positions itself as achieving high quality despite a small scale (13K samples), but lacks a direct empirical comparison with larger-scale alternatives. It does not have direct comparison using PMC-VQA's full 227K samples for training; Table 5 only compares against PMC-VQA in a limited setup, not at full scale. 

2. The paper claims to improve medical VQA, but all evaluation benchmarks are also derived from literature or educational materials, not real clinical data. The paper's questions (Figure 2) are mostly "recognition" (33.6%) and "modality identification" (27.7%) — these are not representative of clinical reasoning. 

3. The paper motivates multimodal generation by criticizing text-only approaches (L83-84: "text-only LLMs that ignore visual evidence"). However, Table 2 shows that for 3B models, multimodal generation (54.72) performs worse than text-only generation (54.80), suggesting that visual grounding provides minimal or even negative value for smaller models. Is it necessary to have multimodal generation ?

### Questions
1. If we train with 100k or all samples from PMC-VQA, would the performance be comparable or better? The paper does not answer this question.

2. Is it possible to have a much large scale dataset based on dataset like BIOMEDICA[1], what would the cost be? 



[1] Lozano, Alejandro, et al. "Biomedica: An open biomedical image-caption archive, dataset, and vision-language models derived from scientific literature." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors proposed a multi-stage pipeline that leverages open biomedical databases (Biomedica Index) together with open large language models to generate a synthetic medical VQA dataset. Specifically, the pipeline involves a set of rubrics to guide the generator and verifier LLMs in generating and verifying the synthetic VQA samples, resulting in a dataset of around 13,000 examples. The authors claimed that the proposed dataset is beneficial for improving performance on several benchmarks. Further ablation studies also demonstrated the potential advantages of using the proposed dataset. The complete openness throughout the entire pipeline paves a potential way to create and verify the training of medical LLMs.

### Strengths
- The author proposed a first fully open pipeline to create synthetic medical VQA dataset. From the dataset, model, to the generation and verification process are transparent, which can be easily customized to various setting and is  beneficial to the research community.

- A deep investigation of the generator and verifier are presented in the paper, which shed light on how people can leverage the proposed pipeline.

### Weaknesses
In summary, my concerns are more related to the resulting data evaluation:
- Potential data leakage: Although we cannot say much about the other open models compared in the baseline, it seems to me that the resulting MedVLSynther has data leakage issue when evaluated on PMC benchmark because Biomedica is dereived from PMC. 

- Inconsistency and confusing table and graph results: The numbers of MedVLThinker-3B/7B presented in Figure 1c does not match those in Table 6. Feel like the numbers in Figure 1c is directly copy from the MedVLThinker paper. If so, what is shown in the table 6?

- The benefit of data for performance boost seems to be minor: When looking table 5, it seems to me that most of performance gain is from RL, when doing pure SFT, there is no clear advantage when compared to model trained on PMC. Since the main selling point of the paper is the new data not RL training method, it is questionable where the proposed data is truly better than the existing dataset

- Lack of baseline results: It would be nice to at least know how well the generator (GLM-4.5V 108B ) perform on the downstream task. From the application perspective, If GLM easily outperforms other methods in the benchmark, it seems to be unclear why bother to train a model on this new dataset instead of just using an open model. 

- The baseline is a bit dated: It's unclear why did the author choose Qwen2.5 series instead of Qwen3. Based on the Qwen3 release date (2025.04.29) and ICLR 2026 deadline (2025.09.24), the timeline should not be an issue.

### Questions
Question and suggestion
Q1: Could author explain why not using the latest Qwen3 series?
Q2: Could author explain more about the data contamination analysis? The supp. only mentioned 8220 rows in a test data which does not appear in the main paper.
S1: It would be nice to know how other open source LLM's performance on the benchmark (e..g, Qwen3 or GLM-4.5V 108B ) so that we can better understand how practical of the proposed dataset.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents MedVLSynther, a well-designed framework for automatically generating and verifying medical VQA data from open biomedical literature. The proposed generator–verifier pipeline ensures that the resulting questions are self-contained, clinically valid, and image–text consistent, leading to the creation of a high-quality dataset (MedVLSynther-13K).

Overall, I am positive about the paper. While the novelty is somewhat limited—as the work shares similarities with PMC-VQA—it nonetheless represents a solid contribution with comprehensive experimental results and clear methodological rigor. Please refer to my detailed comments regarding minor typos and identified weaknesses in the Questions section. I am very willing to raise my score if the concerns have been solved.

### Strengths
1. Provides a new and useful medical VQA dataset.
2. Carefully analyzes the data generation pipeline.
3. Conducts large-scale evaluation showing the dataset’s usefulness.

### Weaknesses
Main concerns:

My main concern falls in two aspects:

1. The improvement is limited. As shown in Table 2, most gains appear to come from using PMC figures, while the additional rubric-based filtering contributes little to performance improvement. Can authors explain the results more?
2. The paper lacks a direct evaluation of data quality. All reported results rely on downstream task performance rather than explicitly measuring the dataset’s intrinsic quality. It is strongly recommended that the authors include a more direct assessment—such as human expert evaluation—to better validate the reliability and usefulness of the generated data.

### Questions
Minor Questions:

1. Several tables mention “PMC,” but the paper does not clearly define what it refers to. It is likely related to PMC-VQA, but the authors should explicitly state this to avoid confusion and improve clarity.
2. “Text-only generation lifts the average to 54.80=.” The “=” seems a typo.
3. The tables use too many green background highlights, which actually reduce readability instead of improving it. In addition, Table 6 lacks the green markings used elsewhere, leading to inconsistency in presentation. The authors should simplify the color scheme and maintain consistent formatting across all tables.
4. The paper references “Biomedica: An open biomedical image-caption archive, dataset, and vision-language models derived from scientific literature” but provides insufficient context about it. The authors should briefly introduce this dataset to help readers understand its role without needing to consult external papers—for example, clarifying whether compound figures were separated and describing the general distribution of the figures used.

### Soundness
3

### Presentation
3

### Contribution
3
