# MedBLINK: Probing Basic Perception in Multimodal Language Models for Medicine

- Decision: Reject
- Scores: 4, 2, 2, 4, 4

## Abstract
Multimodal language models (MLMs) show promise for clinical decision support and diagnostic reasoning, raising the prospect of end-to-end automated medical image interpretation. However, clinicians are highly selective in adopting AI tools; a model that makes errors on seemingly simple perception tasks such as determining image orientation or identifying whether a CT scan is contrast-enhanced, are unlikely to be adopted for clinical tasks. We introduce Medblink, a benchmark designed to probe these models for such perceptual abilities. Medblink spans eight clinically meaningful tasks across multiple imaging modalities and anatomical regions, totaling 1,429 multiple-choice questions over 1,605 images. We evaluate 20 state-of-the-art MLMs, including general-purpose (GPT-5, Gemini 1.5 Pro, Claude 3.5 Sonnet) and domain-specific (Med-Flamingo, LLaVA-Med, RadFM) models. While human annotators achieve 96.4% accuracy, the best-performing model reaches only 76.3%. These results show that current MLMs frequently fail at routine perceptual checks, suggesting the need to strengthen their visual grounding to support clinical adoption.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MedBLINK, a benchmark designed to evaluate basic perceptual abilities of multimodal language models (MLMs) in medical imaging. The benchmark comprises 1,429 multiple-choice questions across 1,605 images, spanning eight tasks across five modalities (X-ray, CT, Endoscopy, Histopathology, and Ultrasound). The authors evaluate 20 state-of-the-art MLMs, including general-purpose models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) and medical-specific models (Med-Flamingo, LLaVA-Med, RadFM). Results show significant performance gaps: human experts achieve 96.4% accuracy while the best model (GPT-4o) reaches only 76.3%, revealing fundamental weaknesses in medical visual perception that could impede clinical adoption.

### Strengths
1. The paper is clearly structured with effective presentation and logical flow.
2. The benchmark is well-grounded in clinical practice, addressing the critical need for perceptual competence before diagnostic trust.
3. Evaluation across 20 state-of-the-art multimodal language models reveals that current MLMs fall far short of the perceptual capabilities required for medical applications, with the best model achieving only 76.3% versus 96.4% human expert performance.

### Weaknesses
1. Limited dataset scale: With only 1,429 questions across 1,605 images, the benchmark is considerably smaller than existing medical multimodal benchmarks, which may limit the generalizability and robustness of conclusions.
2. Lack of coherent task design rationale: While the selected tasks can measure certain perceptual capabilities, the choice of these specific eight tasks appears ad-hoc without clear underlying principles or systematic framework explaining why these particular tasks comprehensively represent fundamental medical perception.

### Questions
1. Data contamination risk: Given that the benchmark is constructed from publicly available datasets, how do you ensure there is no data leakage, particularly for API-based models whose training data remains undisclosed? What measures were taken to verify test images were not seen during pre-training?
2. Missing recent medical models: The evaluation primarily focuses on general-purpose multimodal models. Recent state-of-the-art medical-specific models such as Baichuan-M2 and MedGemma should be included to provide a more complete assessment of current medical MLM capabilities.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper propose MEDBLINK, a benchmark designed to probe these models for simple but important perceptual abilities. MEDBLINK spans eight clinically meaningful tasks across multiple imaging modalities and anatomical regions, totaling 1,429 multiple-choice questions over 1,605 images. With this benchmark, authors evaluate 20 state-of-the-art MLMs. Experimental results show that current MLMs frequently fail these simple but impartant perceptual checks.

### Strengths
The VLLM's simple but important perceptual abilities indeed is an important issue for VLLM in medicine application. The paper is easy to follow.

### Weaknesses
1. Lack of technique contribution. This paper construct a benchmark to evaluate the basic perceptual abilities in VLLM. Authors have no important technique contribution in this paper. 
2. The benchmark size is limited, which may not fully evaluate all basic perceptual abilities. It is better to provide a table to compare with other medical benchmarks.
3. I advise authors add a section to provide develop directions for future VLM. This benchmark find several issues in current VLM for medical, but it is important to provide insight for futhre development of VLM.

### Questions
1. Please provide detailed distrubution of this benchmark, including the number of different modility, different anatomical regions, and different task. 
2. Will authors release benchmark and evaluation code in the future?
3. It is interesting that if vlm fails to predict basic perceptual questions, could them get the correct clinical diagnostic?

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
3

### Summary
This paper introduces MedBLINK, a benchmark designed to probe multimodal large language models (MLLMs) for basic perceptual abilities that are trivial for clinicians. The benchmark includes 1429 multiple-choice questions over 1605 images, spanning eight perceptual tasks across five imaging modalities. General and medical-domain MLLMs are evaluated. While human annotators achieve 96.4% accuracy, the best model reaches only 76.3%, highlighting a gap in visual grounding even on seemingly trivial tasks. The paper argues that these findings underscore the need to strengthen perceptual robustness before deploying MLLMs in clinical decision support.

### Strengths
The paper is well-motivated, addressing basic perceptual competence. 
The benchmark is clearly structured, covering multiple imaging modalities and clinically relevant perceptual subtasks with expert validation. The experimental section is extensive, comparing a diverse set of 20 MLLMs and including human and CNN baselines.

### Weaknesses
The main limitation lies in novelty. Similar perceptual or visual question-answering benchmarks already exist, such as MedFrameQA, and MedTrinity-25M, and MedBLINK appears to extend these ideas into the medical domain without introducing fundamentally new methods or task formulations. It lacks open-ended evaluation which is critical for real life clinical use.
Several tasks, such as determining whether an X-ray is upside down, seem disconnected from real clinical practice and may not provide meaningful insights into clinically relevant reasoning. The dataset itself is modest in scale and relies heavily on existing public datasets, which limits its generalizability and distinctiveness. 
Moreover, the paper only reports performance gaps without offering concrete insights or methodological directions for improving model perception or grounding. These factors make the work appear incremental despite its solid execution.

### Questions
How do the authors justify that each of the eight selected tasks matter in clinical workflows? 
Could the benchmark be used beyond evaluation, for example as a diagnostic tool to guide model improvement or fine-tuning? 
How does MedBLINK complement or differ in purpose from diagnostic reasoning datasets?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a comprehensive and insightful benchmarking study of [mention the models being evaluated, e.g., large multi-modal models] across a wide range of medical imaging tasks. The work is commendable for its scale, providing in-depth analysis and several interesting conclusions that are valuable to the community. It stands as a good example of a benchmark paper for medical domain. However, for the ICLR community, the contributions might be perceived as borderline.

### Strengths
Extensive Benchmarking: The paper thoroughly evaluates multiple models on a diverse set of medical imaging tasks, offering a clear comparative analysis.

In-Depth Analysis: The discussion goes beyond mere performance metrics, providing insightful observations into model behaviors, strengths, and weaknesses.

Valuable Conclusions: The findings offer practical guidance and highlight important challenges in the application of foundation models to medical vision tasks.

### Weaknesses
Clarification on Human Benchmark: The paper states that "human annotators achieve 96.4% accuracy." This metric is crucial as a performance ceiling, but several details require clarification to fully interpret this benchmark:

- Expertise Level: What was the expertise level of these annotators (e.g., board-certified radiologists, resident physicians, or medical students)? The performance gap between a model and a human can be interpreted very differently based on this.

- Ground Truth Adjudication: For the 3.6% of cases where the primary human annotator was incorrect, how was the ground truth established? Was it through consensus among a panel of senior experts?

- Inter-annotator Agreement: What was the inter-annotator agreement (e.g., Cohen's Kappa) among the human experts? This is essential for understanding the inherent difficulty and subjectivity of the tasks themselves.

Robustness to Prompting : The performance of language-vision models is often highly sensitive to the prompt instruction. The paper should discuss: To what extent was the performance sensitive to variations in the prompt template? Was a systematic prompt engineering or optimization process conducted? Reporting results with different prompting strategies would strengthen the robustness of the findings.

Task Selection and Dataset Representativeness:

Taxonomy and Completeness: The selection of tasks is a key contribution. It would be helpful if the authors could explicitly outline the taxonomy or classification system used to select these specific tasks (as in Sec. 3). A discussion on why these tasks were chosen and whether any other important categories  were considered but omitted would justify the comprehensiveness of the benchmark.

Dataset Biases: The datasets for each task originate from different sources and protocols, which the review correctly notes can appear ad hoc. The authors should explicitly discuss the potential biases present in these combined datasets and how these biases might affect the generalizability of the benchmark results.

Model Selection: The set of evaluated models is substantial. However, to ensure the benchmark remains state-of-the-art and comprehensive, the inclusion of other prominent medical-specific multi-modal models, such as HuatuoGPT-Vision and many more, should be considered. Their performance would provide an even more complete landscape of current capabilities.

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors introduce MedBLINK, a benchmark designed to test whether MLMs possess the basic perceptual abilities that clinicians deem obvious. In detail, MedBLINK consists of eight tasks that probe medical visual tasks across modalities including X‑ray, CT, endoscopy, histopathology, and ultrasound. Tasks include image enhancement detection, visual depth estimation, wave-based imaging depth estimation, histology structure, imaging orientation, relative position, morphology quantification, and age estimation. All tasks use multiple-choice questions with 1429 samples derived from 1605 expert-validated images. The authors evaluate 20 MLMs (open‑source, domain‑specific, and proprietary models) and compare their accuracy against human experts, suggesting existing MLMs still need to improve their visual grounding to support clinical adoption.

### Strengths
**Motivation**: AI models are more critical in medical field. The authors motivate the benchmark by arguing that clinicians will not trust a model that cannot solve simple perceptual tasks. By probing these “blink tasks”, MedBLINK assesses whether MLMs truly “see” the image or exploit superficial correlations. This focus on trustworthiness is reasonable especially as MLMs are being considered for clinical decision support.

**Task design**: The eight tasks are designed simple. They can be extracted from existing labeled datasets with light processing. This makes the benchmark practical and easy to reproduce.

**Comprehensive evaluation**: 20 models spanning proprietary, open‑source, and medical‑specific MLMs are evaluated under a consistent prompting protocol. The results show clear performance differences, revealing that proprietary models (e.g., GPT‑5) outperform medical models such as LLaVA‑Med. Ablation studies examine the impact of model size, prompting strategies, and models designed for spatial reasoning, providing rich insights into current limitations.

### Weaknesses
1. **Ambiguity in Perception**: The core idea of the paper relies on a clean distinction between basic visual perception and complex reasoning, but some tasks go beyond simple perception. In my understanding, a basic perceptual task should be easy for any medically trained person to recognize. I agree those grounding-related tasks are simple for most people, like visual depth estimation, wave-based imaging depth estimation, histology structure, imaging orientation, and relative position. However, Task 1 (image enhancement detection) needs prior knowledge of what enhanced images and different phase of CT images look like, otherwise it is hard to judge. Task 8 (Age Estimation) needs both perception and a visual concept between pediatric and adults, as the paper notes in line 244,  *“pediatric patients exhibit a proportionally larger heart compared to the adults, and the thoracic cage in children appears more circular with horizontally oriented ribs, in contrast to the elliptical cage with oblique ribs seen in adults”*. But these concepts are not included in the pre-prompt. I assume it is one reason why MLMs perform such worse than human-experts.
2. **Task selection is not well justified**: The paper calls the eight tasks clinically meaningful and says tasks are chosen by consulting one senior radiologist, but with little detail on the process. In other words, this makes the choice subjective. From my perspective, some tasks do not feel essential in practice. For examples, for Task 1, all CT phases have clinical value. A phase selection task would be more meaningful than only detecting enhancement. For Task 7, why use wisdom tooth counting instead of common checks like laterality (left vs right) or implant detection. Taken with (1), the tasks are not well defined and lack detailed clinical support and evidence. A broader clinician survey and clear selection criteria would make the benchmark more convincing.
3. **Risk of data leakage**: The benchmark reuses well-known public datasets (for example VindDr, ChestX-ray8, Kvasir, EchoNet-Dynamic). Even after re-organizing, MLMs may have seen parts of these during pretraining. This raises a fairness concern. So, how do you ensure MedBLINK is a fair benchmark? A stronger approach is to collect an in-house clinical set and compare results between private data and public data. This ablation would help validate that the findings are not driven by potential data leakage.
4. **Limited novelty**: The main idea follows BLINK. Moving this idea to medical field is useful, but it is an extension rather than a new concept. In addition, the MedBLINK also feels like a complement to existing medical benchmarks such as MediConfusion and GMAI‑MMBench (perceptual tasks).

### Questions
Apart from Weaknesses, I am curious about evaluation on agentic AI: The paper shows that small specialized models can outperform all MLMs on these tasks. This points to the value of tool use. From this paper, the experiments use a zero shot single model setup, but many advancing systems run MLMs in agent mode. An agent can call tools that specialize in perception. So, it would be useful to test MedBLINK in an agentic setting. For example, the agent could route to a segmentation or classification tool and then count wisdom tooth or judge the orientation more accurately. These tests would show whether the “blink tasks” remain hard once tool use is allowed.

### Soundness
2

### Presentation
3

### Contribution
2
