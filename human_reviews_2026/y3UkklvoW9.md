# THEMIS: Towards Holistic Evaluation of MLLMs for Scientific Paper Fraud Forensics

- Decision: Accept (Poster)
- Scores: 8, 8, 6

## Abstract
We present **THEMIS**, a novel multi-task benchmark designed to comprehensively evaluate multimodal large language models (MLLMs) on visual fraud reasoning within real-world academic scenarios. Compared to existing benchmarks, THEMIS introduces three major advances. (1) **Real-World Scenarios and Complexity**: Our benchmark comprises over 4,000 questions spanning seven scenarios, derived from authentic retracted-paper cases and carefully curated multimodal synthetic data. With 60.47\% complex-texture images, THEMIS bridges the critical gap between existing benchmarks and the complexity of real-world academic fraud. (2) **Fraud-Type Diversity and Granularity**: THEMIS systematically covers five challenging fraud types and introduces 16 fine-grained manipulation operations. On average, each sample undergoes multiple stacked manipulation operations, with the diversity and difficulty of these manipulations demanding a high level of visual fraud reasoning from the models. (3) **Multi-Dimensional Capability Evaluation**: We establish a mapping from fraud types to five core visual fraud reasoning capabilities, thereby enabling an evaluation that reveals the distinct strengths and specific weaknesses of different models across these core capabilities. Experiments on 16 leading MLLMs show that even the best-performing model, GPT-5, achieves an overall performance of only 56.15\%, demonstrating that our benchmark presents a stringent test. We expect THEMIS to advance the development of MLLMs for complex, real-world fraud reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents THEMIS, a novel multi-task benchmark for evaluating Multimodal Large Language Models (MLLMs) in scientific paper fraud forensics. It features over 4K questions across 7 real-world academic scenarios (from retracted papers and synthetic data), covers 5 fraud tasks with 16 fine-grained manipulations, and maps tasks to 5 core reasoning capabilities. Experiments on 11 leading MLLMs show even top models (e.g., GPT-5 with 56.29% BRI) fall below passing thresholds, revealing limitations in handling compound manipulations and imbalanced capabilities.

### Strengths
* The starting point is novel and has practical application value.
* The data construction is complete, clear and reproducible.

### Weaknesses
* Typo: There is an issue with the citation in 4.4 Appendix
* From the perspective of the benchmark, it is quite well done. However, in the long run, this topic should be more suitable for optimizing, training, and fine-tuning models. If there are fine-tuning results, the value of this benchmark will be even higher.

### Questions
* In your opinion, how should the community use your benchmark? Is it to evaluate a model's ability to detect cheating and then select an excellent model to serve as a judge? Or are there more ambitious goals?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a new benchmark, termed THEMIS, to comprehensively evaluate the capability of current MLLMs in scientific paper fraud forensics. Compared with existing visual fraud reasoning benchmarks, the proposed THEMIS exhibits several advantages, including real-world scenarios & complexity, tasks diversity & granularity, and multi-dimensional capability evaluation. Based on THEMIS, the authors extensively evaluate recent open-source and proprietary MLLMs and conduct in-depth analysis.

### Strengths
1. THEMIS defines a comprehensive taxonomy for the field of scientific paper fraud forensics.  Specifically, THEMIS covers 7 academic scenarios, 5 tasks, 16 manipulation operations, and 5 core reasoning capabilities, which is more diverse than existing visual fraud reasoning benchmarks.
2. The data quality of THEMIS is high. The synthetic data is rigorously reviewed by human experts. Moreover, THEMIS contains real samples in addition to synthetic samples, which makes it closer to real-world applications. 
3. The authors provide the details of data curation and question generation, which lays a soild groundwork for future research.
4. The authors conduct comprehensive evaluation and in-depth analysis based on THEMIS. The MLLMs involved includes 6 proprietary ones and 5 open-source ones. The evaluation results demonstrate the significant challenges posed by this benchmark. Furthermore, the analysis conclusions provides deep insights into the related field.

### Weaknesses
1. The MLLMs evaluated in this paper are not comprehensive enough. It is recommended to supplement the results of InternVL3.5, GLM4.5V, Gemini-2.5-Pro, Claude, etc.
2. There is a lack of comparsion on difference parameter sizes of the same series of MLLMs (e.g., Qwen2.5-VL-3B/7B/32B/72B).
3. The conclusion in lines 373-375 is not well explained.

### Questions
In lines 1443-1444, "However, performance drops markedly on synthetic data, with some tasks almost completely failing (e.g., LLaMA approaches zero on both splicing and copy-move)". This statement may be inconsistent with Table 10.

### Soundness
3

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
3

### Summary
This paper constructs a benchmark for forgery forensics in academic papers, covering various forms of academic misconduct across multiple research domains. The benchmark includes both simple manipulations such as copy-move and fully AI-generated forgeries, and provides a comprehensive evaluation of existing MLLMs. The results show that current MLLMs perform poorly on this benchmark, with localization tasks performing significantly worse than attribution tasks, highlighting the substantial room for improvement in this area.  
**As I am not deeply familiar with this specific domain, I would prefer to refer to the opinions of other reviewers when determining the final score.**

### Strengths
1. The constructed benchmark covers a wide range of academic disciplines and forgery types, and the evaluation includes a relatively comprehensive set of model categories.  
2. Although there are a few typos, the paper is overall well-written, and the figures and visual presentations are clear and well-designed.

### Weaknesses
1. The authors mention using the **Fitz** library and **YOLOv7** for information extraction and segmentation. However, to my knowledge, there are now more accurate tools in the document extraction domain, such as **dots.ocr** and **MinerU**. Given the diversity of samples, the effectiveness of simply applying Fitz (PyMuPDF) and YOLOv7 is questionable. Since subsequent steps rely heavily on accurate information extraction, this stage could be improved to ensure higher benchmark quality and reliability.  
2. The appendix should include more **case examples** from different categories within the benchmark to give readers an intuitive understanding of its scope and difficulty level.

### Questions
Please refer to the *Weakness* section for detailed explanations.  
If the authors can adequately address my concerns, I would be very willing to raise my score.

### Soundness
3

### Presentation
3

### Contribution
4
