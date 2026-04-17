# MSEarth: A Multimodal Scientific Dataset and Benchmark for Phenomena Uncovering in Earth Science

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
The rapid advancement of multimodal large language models (MLLMs) has unlocked new opportunities to tackle complex scientific challenges. Despite this progress, their application in addressing earth science problems, especially at the graduate level, remains underexplored. A significant barrier is the absence of benchmarks that capture the depth and contextual complexity of geoscientific reasoning. Current benchmarks often rely on synthetic datasets or simplistic figure-caption pairs, which do not adequately reflect the intricate reasoning and domain-specific insights required for real-world scientific applications. To address these gaps, we introduce MSEarth, a multimodal scientific benchmark curated from high-quality, open-access scientific publications. MSEarth encompasses the five major spheres of Earth science: atmosphere, cryosphere, hydrosphere, lithosphere, and biosphere, featuring over 289K figures with refined captions. These captions are crafted from the original figure captions and enriched with discussions and reasoning from the papers, ensuring the benchmark captures the nuanced reasoning and knowledge-intensive content essential for advanced scientific tasks. MSEarth supports a variety of tasks, including scientific figure captioning, multiple choice questions, and open-ended reasoning challenges. By bridging the gap in graduate-level benchmarks, MSEarth provides a scalable and high-fidelity resource to enhance the development and evaluation of MLLMs in scientific reasoning. The benchmark is publicly available to foster further research and innovation in this field.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces MSEarth, a large-scale, multimodal dataset and benchmark designed to evaluate graduate-level reasoning in Earth science. Curated from over 64,000 open-access scientific publications, it features over 289K figures across the five major spheres of Earth science. The central contribution is the concept of "refined captions," which programmatically enrich the original, brief figure captions with deep contextual information, analysis, and reasoning extracted from the main body of the source paper. Based on this enriched data, the authors developed a semi-automated pipeline using a multi-agent voting system to generate and filter a vast collection of VQA tasks, including scientific figure captioning, multiple-choice questions, and open-ended questions. Extensive evaluations on a wide range of state-of-the-art Multimodal Large Language Models (MLLMs) reveal a significant performance gap between tasks requiring simple perception and those demanding complex, domain-specific reasoning, demonstrating the benchmark's difficulty and relevance.

### Strengths
1. The "refined caption" methodology is a major strength. It moves beyond the limitations of simplistic figure-caption pairs by integrating deep contextual information. 

2. The paper introduces a robust and scalable pipeline for benchmark creation. The use of a multi-agent voting system combined with a phased filtering process effectively automates the generation of high-quality, challenging questions, striking an excellent balance between scale and rigor.

### Weaknesses
1. While the authors commendably introduce LLM-based metrics like Cap-Eval and OE-Eval, the evaluation tables for captioning and open-ended QA (Tables 3 and 4) still heavily feature traditional lexical metrics like ROUGE, BLEU, and METEOR. As the paper's own results show, these n-gram-based metrics exhibit minimal variance across models and fail to capture the significant performance differences observed in the accuracy-based MCQ task. They are fundamentally ill-suited for evaluating the factual correctness and scientific nuance required by this benchmark, potentially masking the true capabilities of the models and reducing the conclusiveness of these specific results.

2.  Given the rapid pace of MLLM development, the inclusion of some of the most recent and powerful open-source models (e.g., Mimo-VL, GLM-4.5V) would make the comparative analysis more conclusive and compelling.

3. The manuscript would benefit from a final round of careful proofreading to address minor inconsistencies in formatting and capitalization (e.g., "gemini-2.5-pro" in line 1538).

### Questions
1. The pipeline for generating MSEarth seems highly generalizable. What challenges do you foresee in applying this framework to other scientific domains, such as molecular biology or particle physics, where figures and data representations follow vastly different conventions?

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
5

### Summary
This paper contributes a large-scale human-reviewed test & llm-generated training set for Multimodal understanding of Earth Science. The authors first parse the Multimodal content from Earth Science and then leverage various LLMs through many stages to build a large-scale dataset comprising QA, captioning, and open-ended generation question-answer pairs. Then they leverage human experts to label ~1k of them as a golden test set. They proved that models trained on their training set have better results compared with the original models. They also show that the test set is even challenging for the current most advanced commercial LLMs.

### Strengths
1. The dataset for earth science is potentially useful to this community, as well as improving the general capabilities of MLLMs.

2. The constructed q-a pairs was proved to help improve the capabilities of MLLMs.

3. The human-expert labeled test set was challenging for current models and serves as a potential testbed for current AI's understanding of Earth science.

### Weaknesses
1. The authors mentioned that "All papers used were obtained from OpenDataLab" so this paper actually does not provide any new data sources. And the contribution lies in post-processing those data using the combination of various LLMs to ensure quality, though the quality is questionable.

2. In my opinion, the most valuable part of this dataset is the human-labeled Open-Ended QA test set. However, the current results show that the majority of the open-sourced and Proprietary LLMs demonstrate similar results, despite their sizes or providers. This makes me question the real quality of this test set since it's expected to see significant differences. For example, the BERTSCORE is almost always 82~83, which explains nothing.

3. There have already been many Captioning tasks in previous work, though the authors emphasize that their Captioning requires additional context from the paper; however, this is not new.

### Questions
1. In Table 2&3&4, why do the results from Proprietary Models even underperform (like gpt4o) compared with open-sourced models (e.g. Qwen2.5-VL-32B)? Does this mean the open-sourced model is really better? Or does this mean there is some quality issue within the dataset?

2. In Table 2&3&4, there is almost little difference between results on various model sizes, e.g. DeepSeek-VL2, Qwen2.5-VL-32B & Qwen2.5-VL-72B. This is unintuitive since the results have almost no change with increasing model size, which might imply inherent flaws in the datasets.

3. Do you have better evals for open-ended QA? ROUGEL or BLEU, etc seem not very suitable if all the results are very close to each other.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MSEarth, a large-scale multimodal scientific benchmark for evaluating and developing multimodal large language models (MLLMs) in the field of Earth sciences. The authors argue that current benchmarks fail to capture the depth, complexity, and reasoning required at the graduate academic level. MSEarth includes over 289K scientific figures across all five Earth system spheres (atmosphere, cryosphere, hydrosphere, lithosphere, biosphere), paired with carefully refined and context-enriched captions. The benchmark supports several tasks, including figure captioning, multiple-choice questions, and open-ended scientific reasoning.

### Strengths
1. The authors propose first benchmark that targets graduate-level geoscientific multimodal reasoning.

2. This benchmark is curated from high-quality, open-access scientific publications, and covers all five Earth science spheres (atmosphere, cryosphere, hydrosphere, lithosphere, biosphere).

### Weaknesses
1. The authors does not clearly explain how the ``ground truth'' for refined captions, QA pairs, or reasoning tasks is constructed. Although it claims that captions are enriched using discussions from the original papers, it remains ambiguous whether these are (1) directly extracted from the text, (2) rewritten by humans, or (3) generated using LLMs.


2. The benchmark does not include human or expert performance, making it difficult to assess its true difficulty or validate the claim of “graduate-level” reasoning. Low model performance alone is not sufficient evidence of task difficulty—it may instead indicate a lack of domain-specific pretraining. Without human baselines, it remains unclear whether the benchmark genuinely requires advanced reasoning or is simply out-of-distribution for current MLLMs.


3. Although MSEarth demonstrates a clear gap between perception and scientific reasoning, the paper does not provide a deeper analysis of why models fail or which types of reasoning are most challenging. There is no taxonomy of reasoning skills (e.g., causal inference, trend interpretation, quantitative estimation), no breakdown across Earth system domains, and no systematic error analysis. As a result, the benchmark highlights a problem but offers limited insight into the underlying mechanisms or how future models should be improved.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
