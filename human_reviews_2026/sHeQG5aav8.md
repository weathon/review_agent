# SpineBench: A Clinically Salient, Level-Aware Benchmark Powered by the SpineMed-450k Corpus

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 4

## Abstract
Spine disorders affect 619 million people globally and are a leading cause of disability, yet AI-assisted diagnosis remains limited by the lack of level-aware, multimodal datasets. Clinical decision-making for spine disorders requires sophisticated reasoning across X-ray, CT, and MRI at specific vertebral levels. However, progress has been constrained by the absence of traceable, clinically-grounded instruction data and standardized, spine-specific benchmarks. To address this, we introduce SpineMed, an ecosystem co-designed with practicing spine surgeons. It features SpineMed-450k, the first large-scale dataset explicitly designed for vertebral-level reasoning across imaging modalities with over 450,000 instruction instances, and SpineBench, a clinically-grounded evaluation framework. SpineMed-450k is curated from diverse sources, including textbooks, guidelines, open datasets, and $\sim$1,000 de-identified hospital cases, using a clinician-in-the-loop pipeline with a two-stage LLM generation method (draft and revision) to ensure high-quality, traceable data for question-answering, multi-turn consultations, and report generation. SpineBench evaluates models on clinically salient axes, including level identification, pathology assessment, and surgical planning. Our comprehensive evaluation of several recently advanced large vision-language models (LVLMs) on SpineBench reveals systematic weaknesses in fine-grained, level-specific reasoning. In contrast, our model fine-tuned on SpineMed-450k demonstrates consistent and significant improvements across all tasks. Clinician assessments confirm the diagnostic clarity and practical utility of our model's outputs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors proposed a spine dataset containing multimodal data collected from about 1000 clinical cases. Several tasks, e.g., question-answering, multi-turn consultations, and report generation, are composed based on the multimodal data. The authors also trained a VLM using both some public datasets (as general medical domain knowledge) and the data from the proposed dataset. The manuscript is overall easy to follow, but it also suffers from several critical flaws that significantly degrade the usefulness of the proposed data.

### Strengths
- The proposed dataset focuses on the spine-related diseases, which are rare in the community.
- The proposed dataset and benchmark are composed using multi-modality data, beyond the texts and images, which is closer to the problems in real-world practice.
- The paper is overall easy to follow.

### Weaknesses
-  The presented results fail to demonstrate the advantage of the proposed model (trained with the domain data) in comparison to generalist models, as shown in Table 4. It is also a bit surprising to see that models trained on medical-domain data (e.g., Linshu and Huatuo) often perform worse than generalist models.
-  The design of the testing set in the proposed benchmark may not evaluate "the overall performance of AI systems in spinal diagnostic tasks," considering only the multiple-choice and reporting tasks are included in the testing. The mismatch between the tasks in the data composition and those in the testing is also questionable and somewhat misleading in justifying the benchmark's significance. 
- Many details of the dataset and the process of dataset composition are missing. For example, it is not clear about the number of samples and data entries included in the final testing and training set, since only multiple-choice and reporting parts are included in testing. How many cases are associated with those data? Report examples are NOT provided in Appendix B.5. 
- The dataset employed roughly 1000 cases, which is relatively small considering that the data are from 11 different hospitals and cover dozens of diseases and scenarios. The diversity of the included is therefore questionable to evaluate the overall performance in spinal diagnostic tasks.

### Questions
N/A

### Soundness
3

### Presentation
2

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
The work presents a largest dataset explicitly designed for vertebral-level reasoning and a related benchmark, which is medically meaningful, but several important points in the manuscript remain unclear or have issues that should be addressed.

### Strengths
1. The paper contributes SpineMed-450k, the largest spine diagnosis dataset, with 450,000+ multi-modal instruction instances covering diverse tasks across 14 spine conditions, filling a significant gap in specialized medical AI resources.

2. The authors demonstrated genuine "clinician-in-the-loop" design with 17 board-certified orthopedic surgeons participating across data curation, task definition, and validation stages.

3. The 10-dimensional evaluation assesses for AI-generated clinical reports not just diagnostic accuracy but also communication quality, reasoning transparency, risk assessment, and technical feasibility, better capturing what makes clinical reports actually useful in practice.

4. The ablation studies convincingly show that a 7B model trained on specialized spine data substantially outperforms 72B general models, proved the data effective in the certain task.

### Weaknesses
1. The manuscript emphasizes the importance of being “level-aware,” and I agree this could be meaningful. However, in the dataset and benchmark I do not see a clear level-aware design. Some subsets are said to target certain levels, but their structure looks basically the same as the general medical data. It is not clear how “level-aware” actually affects data curation, question type, or evaluation protocol.

2. Many models achieve scores above 80. With such results, the claim that current models still have serious gaps in this domain is not fully convincing. If the tasks are truly difficult and clinically important, we would normally expect lower scores or at least a stronger separation between models. This raises the concern that either the tasks may be too simple, or there may be significant data leakage in the benchmark.

3. The paper mentions an “Expert LLM,” but this term is not defined. Please specify what model is used, and why it is called “expert.” Also, is this the same model as the “Expert VLM” that generates Q&A pairs, or are these different systems? This point is unclear and affects reproducibility.

Minor weakness:
The data pipeline includes “simulated consultations,” but the benchmark tasks do not include any dialogue / consultation evaluation. Why is this step necessary? What is the purpose of generating simulated consultations if the final tasks are not dialogue-based?

### Questions
Please answer the following questions:

1. How do the authors demonstrate the importance of the level-aware design? Is there any ablation study showing that the level-aware component is necessary (for example, a performance drop when level information is removed or when levels are mixed)?

2. Since many models already achieve >80, it appears that even without this work, general-domain training data is already sufficient to solve most of the tasks at a reasonably high level. In that case, how should the importance/necessity of this dataset be interpreted? Additionally, does the benchmark pose a risk of data leakage, i.e., that the evaluated models have already encountered similar data during pretraining?

3. What is the “Expert LLM”? The authors should clearly define which model this refers to, how it is used in the pipeline, and what prompt format is applied. How was this Expert LLM obtained or fine-tuned?

### Soundness
2

### Presentation
2

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
This paper introduces SpineMed-450k, a large-scale dataset of 450,000+ instruction instances for vertebral-level reasoning across spine imaging modalities, and SpineBench, a clinically-grounded evaluation framework. The dataset is curated through a "clinician-in-the-loop" pipeline involving diverse sources (textbooks, guidelines, public datasets, ~1,000 hospital cases) with a two-stage LLM generation method. The authors evaluate multiple state-of-the-art LVLMs and present SpineGPT, a fine-tuned model that demonstrates improvements over existing models on spine-specific tasks.

### Strengths
1. The paper is well-written with effective communication of clinical motivation, technical approach, and results. 
2. The paper presents SpineMed-450k aggregates 450k+ high-quality instruction instances from diverse sources—textbooks (377k), clinical guidelines, real hospital cases (~1,000 from 11 hospitals), and public datasets, proposes SpineBench, which offers a professionally curated, level-aware benchmark specifically designed for spine diagnosis and treatment planning.
3. The paper systematically evaluates a dozen contemporary open-source LVLMs on SpineBench. This rigorous assessment establishes important baseline performance standards and identifies key research gaps for the community.

### Weaknesses
1. SpineGPT does not specifically leverage or optimize for the unique properties of SpineMed-450k. The training still relies heavily on generic open-source medical data in the first stage (Appendix Table 6), raising questions about whether the model fully exploits the dataset's clinically-grounded, level-aware structure.
2. Suboptimal performance compared to state-of-the-art: Table 4 shows SpineGPT (87.44%) underperforms Gemini-2.5-Pro (89.23%) on average; Appendix Table 7 confirms this performance gap across multiple metrics; and Missing comparisons with recent specialized medical models (e.g., Baichuan-M2, MedGemini), limiting assessment of the model's competitive standing
3. The Dataset Generation section indicates that instruction tuning data generated from open-source spine datasets (Spark, VerSe) lacks rigorous quality verification, unlike the hospital cases which underwent expert validation. This inconsistency may introduce noise and affect model reliability.

### Questions
1. Will SpineMed-450k be publicly released? Open-sourcing the dataset would significantly benefit community development and enable reproducible research. Please clarify the release timeline, licensing terms, and any restrictions due to hospital case privacy considerations.
2. SpineGPT (87.44%) underperforms Gemini-2.5-Pro (89.23%) despite being trained on SpineMed-450k. What factors contribute to this gap? Is it due to model scale (7B vs. proprietary larger model), training strategy, data quality, or fundamental architectural limitations? A deeper analysis would strengthen the paper's insights.
3. Regarding Figure 3 and the data curation pipeline, how do you address potential OCR errors such as figure-text misalignment or incorrect caption matching during the structured information extraction process? What quality control mechanisms ensure the Picture Context Matching algorithm correctly pairs images with their contextual descriptions?
4. In Table 5, what would be the performance if the model were trained exclusively on spine data without the general orthopedic (non-spine) data? This ablation would clarify whether the multi-stage curriculum (general → orthopedic → spine) is necessary or if direct spine-specific training suffices.

### Soundness
4

### Presentation
4

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
Spine disorders affect 619 million people and are a leading cause of disability, which has been constrained by the absence of traceable, clinically-grounded instruction data and standardized, spine-specific benchmarks. This paper develops a clinically-grounded evaluation framework and a large-scale dataset explicitly designed for vertebral-level reasoning. Moreover, people demonstrate the effectiveness of their dataset by training a fine-tuned spine LVLM.

### Strengths
This paper focuses on spine disorders, which are a leading cause of disability. The paper is clearly presented and easy to follow.

### Weaknesses
1. Lack of novelty. This paper lacks technical contributions. The pipeline for dataset curation is a common solution, without any novel techniques or insights. 
2. Please provide a table to compare SpineMed-450k with other Spinal diagnosis and treatment datasets. 
3. Since the baseline is the contribution in this paper, please introduce it as an independent section.
4. The SpineBench benchmark only has 487 questions and 87 report prompts, which is a small test subset, possibly limiting generalizability and statistical robustness.

### Questions
1. Is the dataset and baseline model publicly available?
2. For the dataset generation, the authors used the Expert VLM Model to generate questions. How to make sure the questions and answers are correct?  
3. Could the authors provide the performance on other Spinal diagnosis and treatment datasets?

### Soundness
3

### Presentation
3

### Contribution
2
