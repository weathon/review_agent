# LEXam: Benchmarking Legal Reasoning on 340 Law Exams

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Long-form legal reasoning remains a key challenge for large language models (LLMs) in spite of recent advances in test-time scaling. To address this, we introduce ***LEXam***, a novel benchmark derived from 340 law exams spanning 116 law school courses across a range of subjects and degree levels. The dataset comprises 7,537 law exam questions in English and German. It includes both long-form, open-ended questions and multiple-choice questions with varying numbers of options. Besides reference answers, the open questions are also accompanied by explicit guidance outlining the expected legal reasoning approach such as issue spotting, rule recall, or rule application. 
Our evaluation on both open-ended and multiple-choice questions present significant challenges for current LLMs; in particular, they notably struggle with open questions that require structured, multi-step legal reasoning. Moreover, our results underscore the effectiveness of the dataset in differentiating between models with varying capabilities. 
Deploying an ensemble LLM-as-a-Judge paradigm with rigorous human expert validation, we demonstrate how model-generated reasoning steps can be evaluated consistently and accurately, closely aligning with human expert assessments. Our evaluation setup provides a scalable method to assess legal reasoning quality beyond simple accuracy metrics. Project page: https://lexam-benchmark.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces LEXAM, a benchmark evaluating the legal reasoning capabilities of large language models (LLMs). The dataset comprises 4,886 law exam questions sourced from 340 law exams across 116 courses at the University of Zurich, covering a range of legal subjects in both English and German. The authors propose an ensemble LLM-as-a-Judge evaluation paradigm with expert-crafted prompts to evaluate the reasoning quality of legal reasoning and validated it against human expert assessments.

### Strengths
1. The idea of evaluation legal reasoning is important.
2. The authors have conducted rigorous validation of LLM-as-judges’ outputs against expert assessments.
3. The inclusion of both English and German questions

### Weaknesses
1. Some details about the data collection are missing. For example, how did we select the 385 questions for the auxiliary MCQ dataset? Randomly sampled from the 1660 MCQs? How did we get the distribution of questions? Did you apply any filtering process to the question selections or simply use all the questions from the courses? Then how did we select the courses?
2. The paper acknowledges the inherent complexity of legal reasoning, but, according the prompt shown in E.2, it seems that no special guideline based on legal expertise is incorporated in the LLM-as-judge system. If so, what’s the differences between judging general reasoning and judging legal reasoning? The experiments show that this paradigm can outperform human annotations on the 50 randomly selected questions. This means that either the question samples could be biased (e.g., overly simple) or legal experts don’t have a common agreements on the judgment of legal reasoning itself and thus a well-crafted prompt for general reasoning evaluation could outperform legal expert annotations directly.

### Questions
See in the weak points.

### Soundness
3

### Presentation
4

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
The paper presents a new dataset consisting of ~4500 examples with multi-choice and long form questions taken from law exams at the University of Zurich. The LEXam benchmark covers English and German languages and multiple jurisdictions from Switzerland, Europe and international. It also provides an extensive array of baseline experiments covering multiple open and closed weight LMs of different sizes and families. Results show that the questions are challenging (GPT-5 achieving best performance ~70% for open and ~63% for multi-choice).

### Strengths
- Valuable resource that will enable further research in LLM capabilities in a very critical domain such as law
- Extensive benchmarking and analysis of results.

### Weaknesses
- Not clear what the human performance is on the dataset. It might be a good idea to include it in Table 1.

### Questions
- See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LEXAM, a comprehensive benchmark designed to evaluate the long-form legal reasoning capabilities of LLMs. LEXAM is constructed from 4,886 law exam questions collected from 340 exams across 116 law school courses, covering a wide range of subjects and degree levels in English and German. The dataset includes 2,841 open-ended questions and 2,045 multiple-choice questions. Each open-ended question is accompanied by detailed guidance outlining the expected reasoning process, such as issue spotting, rule recall, and rule application.
Through extensive evaluation, the authors find that current LLMs perform poorly on tasks requiring structured reasoning, highlighting persistent challenges in legal understanding. The study further demonstrates the discriminative power of LEXAM in distinguishing models with different reasoning strengths. By employing an ensemble LLM-as-a-Judge framework validated by human experts, the authors provide a scalable and consistent evaluation methodology that aligns closely with expert assessments and goes beyond traditional accuracy-based metrics, offering a new paradigm for measuring legal reasoning quality.

### Strengths
(1) Comprehensive Real-World Dataset. The paper presents a comprehensive real-world dataset for legal reasoning, derived from 340 actual law exams across 116 courses at the University of Zurich. The dataset contains 4,886 questions in both English and German. Notably, it includes long-form, human-written reference answers accompanied by explicit guidance on the expected reasoning process—resources that remain rare in existing legal NLP benchmarks.
(2) Broad Empirical Evaluation. The authors conduct an extensive empirical study covering 26 diverse large language models, including reasoning-oriented systems such as GPT-5, DeepSeek-R1, Gemini-2.5, and Claude-3.7.
(3) Evaluation Design and Reliability. For open questions, the paper adopts an LLM-as-a-Judge framework that ensembles multiple models. This ensemble approach appears more robust than traditional single-model judging methods. For multiple-choice questions, the authors further design perturbed variants by varying the number of answer options, in order to test model robustness and sensitivity to superficial cues.

### Weaknesses
(1) Limited Scope and Jurisdictional Coverage. The dataset is almost entirely derived from Swiss and European law exams at the University of Zurich, using only English and German. Although the authors briefly mention cross-system differences, the dataset design remains heavily biased toward the civil-law tradition. It lacks representation of common-law jurisdictions such as the United States and civil-law contexts like China’s legal system. As a result, while the contribution is valuable, its applicability and generalizability across broader legal systems are clearly limited.
(2) Question Equivalence and Multilingual Validity. LEXAM claims to be a “multilingual legal reasoning benchmark,” but it does not clarify whether the English and German questions are parallel translations or independent items. Nor does it specify whether bilingual legal experts verified semantic and legal-effect equivalence between the two languages. The authors should explicitly state this point, as it directly affects the validity of cross-language comparisons.
(3) Evaluation Framework and Conceptual Limitations. While the proposed ensemble-based LLM-as-a-Judge approach is more robust than using a single LLM judge, it does not introduce a fundamentally new or thought-provoking evaluation paradigm. Moreover, the logic of using LLMs to generate answers, then using other (or similar) LLMs to grade those answers, and finally using those grades to infer the reasoning capability of the same class of models, raises methodological concerns about circularity and self-referential validation.
(4) Entanglement Between Legal Reasoning and Multilingual Processing. The models exhibit strong language sensitivity. As shown in Figure 5, performance in English is consistently and significantly higher than in German. This suggests that the benchmark’s evaluation signal mixes legal reasoning ability with multilingual processing and translation competence. Since the benchmark does not attempt to disentangle these two components, it becomes difficult to interpret the reported results as a clean measure of “legal reasoning” alone.

### Questions
(1) Are the English and German questions parallel translations, or are they completely independent question sets? If they are translations, were they verified by qualified legal professionals to ensure semantic and legal-effect equivalence?
(2) The results show a substantial English–German performance gap. How do the authors disentangle legal reasoning ability from multilingual or translation-related effects in this evaluation?
(3) How do the authors attempt to address or analyze the potential circularity of using LLMs to evaluate other LLMs’ answers—beyond the reported human consistency checks?

### Soundness
2

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
The paper introduces LEXAM, a large-scale benchmark designed to evaluate legal reasoning abilities of large language models (LLMs).
The dataset is built from 340 authentic law school exams from the University of Zurich, covering 4,886 questions (2,841 open-ended and 2,045 multiple-choice) across 116 courses and 78 subfields of law, in both English and German.

The authors propose a novel evaluation framework called LLM-as-a-Judge, an ensemble of models (GPT-4o, Qwen3-32B, DeepSeek-V3) designed to simulate human expert grading. Results show that GPT-5 achieves the highest overall performance (70.2/100 on open questions, 62.7% on multiple choice), while the ensemble grader attains agreement levels comparable to or exceeding human legal experts.

The study presents an ambitious, well-structured attempt to standardize evaluation of LLM reasoning in legal domains.

### Strengths
The benchmark is constructed from real law school exams rather than synthetic or reformulated items, ensuring high realism, linguistic diversity, and true reasoning difficulty.

The framework evaluates both open-ended and multiple-choice questions, tests robustness under option perturbations, and analyzes performance across languages, jurisdictions, and subfields.

The ensemble scoring system of multiple LLMs (taking minimum scores) is well-motivated and empirically validated against human expert grading. It represents a meaningful step toward scalable, consistent evaluation of open-ended legal reasoning.

Including both English and German questions, spanning Swiss and international law, greatly enhances the generalizability and research utility of the benchmark.

### Weaknesses
1. All data originate from Swiss civil law contexts; thus, generalization to common law systems (e.g., US/UK) is uncertain.

2. Although “LLM-as-a-Judge” performs well, it introduces dependence on closed models (GPT-4o, DeepSeek-V3) and may inherit their biases.

3. The paper reports scores but lacks detailed qualitative analysis or case studies illustrating where models succeed or fail in reasoning.

4. While the dataset is said to be authorized, the anonymization and ethical review process is not discussed in depth.

5. Quantitative comparison with existing benchmarks (LegalBench, LexGLUE, LawBench, COLIEE) is limited to descriptive discussion rather than aligned experimental results.

### Questions
none

### Soundness
3

### Presentation
3

### Contribution
3
