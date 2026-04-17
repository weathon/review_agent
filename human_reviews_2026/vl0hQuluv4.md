# ACADREASON: Exploring the Limits of Reasoning Models with Academic Research Problems

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
In recent years, the research focus of large language models (LLMs) and agents has shifted increasingly from demonstrating novel capabilities to complex reasoning and tackling challenging tasks. However, existing evaluations focus mainly on math/code contests or general tasks, while existing multi-domain academic benchmarks lack sufficient reasoning depth, leaving the field without a rigorous benchmark for high-level reasoning. To fill this gap, we introduce the ACADREASON benchmark, designed to evaluate the ability of LLMs and agents to acquire and reason over academic knowledge. 
It consists of 50 expert-annotated academic problems across five high-reasoning domains, including computer science, economics, law, mathematics, and philosophy. All questions are sourced from top-tier publications in recent years and undergo rigorous annotation and quality control to ensure they are both challenging and answerable. We conduct systematic evaluations over 10 mainstream LLMs and agents. The results show that most LLMs scored below 20 points, with even the cutting-edge GPT-5 achieving only 16 points. While agents achieved higher scores, none exceeded 40 points. This demonstrates the current capability gap between LLMs and agents in super-intelligent academic research tasks and highlights the challenges of ACADREASON. The code and data for the ACADREASON benchmark are available at https://github.com/OPPO-PersonalAI/Acadreason-benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a new benchmark ACADREASON. 

- This benchmark is designed to be a challenging, reasoning-intensive benchmark for academic domains, including CS, Law, Econ, Math and Philosophy. Each domain contains 10 questions. 

- The benchmark is heavily human expert curated. The questions and answers are extracted and formulated by human experts from latest publications, along with hints and scoring checklist.

- The paper also benchmarked the performance of latest Large Reasoning Model and Tool-used agents on ACADREASON. The results show the benchmark is very challenging.

### Strengths
- The proposed benchmarks is a good contribution to the community. It has detailed and careful human expert annotations, which is a good effort. 

- The benchmark is well-positioned, as it focuses on challenging reasoning and academic domains.

- The paper has benchmarked a wide range of latest LLMs and reasoning paradigms.

### Weaknesses
- The benchmark is relatively small, 50 questions in total. Although the scoring hints can give a bit more fine-grained signals, but in general the size is limited.

- It would be better to include a comparison section against other relevant benchmarks.

### Questions
- As one aspect of the contributions is the difficulty, would it be possible to quantify the difficulty and compare with other benchmark in a more explicit way?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents ACADREASON, a new benchmark designed to evaluate academic-level reasoning of large language models (LLMs). Unlike existing reasoning benchmarks such as MMLU-Pro, GPQA, or PaperBench, which focus on either broad factual understanding or narrow scientific tasks, ACADREASON targets deep, research-oriented reasoning across five academic domains: Computer Science, Mathematics, Economics, Law, and Philosophy. The authors curate questions from over 400 research papers and design multi-step annotations. They evaluate several frontier models (GPT-5, DeepSeek-R1, Claude 3.7, Gemini 2.0, etc.) and find that all perform far below human experts, especially on methodological and conceptual reasoning. The study also analyzes the role of hints and long-form responses, showing that factual hints help more than methodological ones.

### Strengths
- High-quality benchmark design. The dataset is small but carefully curated. Questions are derived from genuine academic contexts rather than textbook or competition problems, giving ACADREASON a strong realism advantage. The multi-domain design broadens evaluation coverage beyond STEM, incorporating social science and philosophy.
- Transparent and rigorous annotation process. The paper clearly documents every stage: data sourcing, question formulation, verification, and evaluation. The inclusion of structured checklists and fine-grained rubrics for reasoning quality (clarity, coherence, accuracy) improves reproducibility and reliability compared with prior subjective benchmarks.
- Insightful empirical findings. The experiments reveal important trends: reasoning models still struggle on conceptual abstraction and logical grounding even when they perform well on applied math or coding tasks. The “hint effect” analysis (Table 2, Fig. 3) provides valuable insight into which types of contextual scaffolds actually help reasoning models.
- Readable and well-structured paper. The narrative flows logically, figures are informative, and the motivation for each step is well explained. The benchmark and evaluation protocol could be easily adopted by others studying academic reasoning.
- Contribution significance. Although not methodologically groundbreaking, ACADREASON fills a practical and conceptual gap between task-level reasoning (e.g., GSM8K, MATH) and domain-level scholarly reasoning. It contributes a valuable lens for assessing whether modern LLMs can reason like researchers rather than students.

### Weaknesses
- Limited novelty relative to existing benchmarks. While the dataset is well executed, the idea of academic or research-style reasoning has been partially explored in GAIA, PaperBench, and DeepResearchBench. ACADREASON’s main differentiator is diversity and annotation rigor rather than a fundamentally new evaluation paradigm. A clearer comparative discussion would strengthen its originality claim.
- Scale and statistical power. The dataset contains only about 50 finalized questions, which limits robustness and makes performance variance hard to interpret. It’s uncertain whether differences across models (often within 1–2 points) are statistically meaningful.
- Evaluation subjectivity. Despite the structured rubric, the “LLM-as-a-judge” setup remains vulnerable to bias and consistency issues. The authors mention human spot-checks but do not quantify inter-rater agreement or cross-model evaluation consistency. A partial human-judged subset would greatly improve reliability.
- Limited actionable insight for model design. The results primarily reaffirm known findings that reasoning LLMs remain weak in multi-step conceptual reasoning, but offer little guidance on how to improve them. The benchmark thus functions more as a diagnostic dataset than a research breakthrough.
- Scalability and sustainability. Because the pipeline relies heavily on expert curation and manual validation, it is unclear how ACADREASON could be expanded to larger scales or adapted to new domains without significant effort.

### Questions
- Benchmark uniqueness. How does ACADREASON differ conceptually from PaperBench and GAIA beyond domain coverage? Would the authors consider positioning it as a complement rather than a replacement benchmark?
- Human validation. How many samples were manually verified by human experts? Is there any inter-annotator agreement score (e.g., Cohen’s κ) for the judgment process?
- Evaluator bias. Since GPT-5-mini is used as the evaluator, did the authors test whether results are consistent when switching to another LLM judge (e.g., Claude 3 Opus)? Are ranking trends preserved?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ACADREASON, a new benchmark designed to address shortcomings in current evaluations of Large Language Model (LLM) reasoning. The authors argue that existing benchmarks (e.g., math/code contests, general tasks) lack sufficient academic reasoning depth. To fill this gap, ACADREASON aims to evaluate the ability of LLMs and Agents to acquire and reason with specialized academic knowledge.
The benchmark consists of 50 expert-annotated academic problems across five high-reasoning domains: Computer Science, Economics, Law, Mathematics, and Philosophy. All questions are sourced from top-tier publications from 2023-2025 and underwent rigorous quality control to ensure they are both challenging and answerable.
Key contributions include:
1. The ACADREASON Benchmark: A challenging, cross-disciplinary benchmark focused on frontier academic reasoning, complete with golden answers, verifiable checklists, and three types of hints (background, definition, methodology).
2. SOTA Model Evaluation: A systematic evaluation of over 10 state-of-the-art LLMs and Agents (e.g., GPT-5, o3, DeepSeek-R1, OAgents).
3. Revealed Capability Gap: The results show that even the most advanced LLMs (GPT-5) score poorly (16% pass rate, 40.5% checklist score). Agents perform better (OAgents at 34% / 65.1%) but still show significant room for improvement.
4. Hint Analysis: An ablation study demonstrating that providing hints (especially "methodology hints") significantly improves model performance, suggesting models struggle more with complex methods than with background knowledge.

### Strengths
1. Originality & Significance:
  - The paper addresses a clear and important problem: how to evaluate the deep, domain-specific reasoning capabilities of LLMs.
  - ACADREASON's uniqueness lies in its combination of breadth (spanning both STEM and humanities) and depth (focusing on recent, theoretical problems from top-tier journals). This design tests reasoning on novel knowledge, not just retrieval of pre-existing, commonly known information.
  - The benchmark's high difficulty (evidenced by low SOTA scores) confirms its utility and significance as an evaluation tool that is not easily saturated.
2. Quality:
  - The benchmark's construction methodology is rigorous, involving domain experts (Master's or PhD level) for data curation and annotation.
  - A detailed, multi-stage validation process (shown in Figure 10) was used to ensure data quality, theoretical focus, and question answerability.
  - The evaluation framework is rich. Beyond binary pass/fail ($R_p$), the "Checklist Score" ($R_j$) allows for a more granular analysis of the model's reasoning process. The inclusion of three hint types is a significant strength, enabling analysis of why a model fails (e.g., lack of background knowledge vs. lack of methodological understanding).
3. Clarity:
  - The paper is well-organized and easy to follow. Figure 1 provides a clear overview of the benchmark construction and evaluation pipeline.
  - Task specifications, evaluation metrics (Sec 3.4), and experimental setups (Sec 4.1) are clearly articulated.
  - Results are presented clearly (Tables 1 & 2), and the Case Study (Fig. 4) offers a concrete, intuitive example of the task's difficulty and the difference between Agent and LLM performance.

### Weaknesses
Based on an in-depth analysis, the paper suffers from three major and interrelated methodological flaws. These flaws severely undermine the validity of the benchmark and the reliability of its conclusions.
1. Unverified Evaluation Reliability
The paper's core weakness lies in its evaluation method. The authors use GPT-5-mini as an "LLM-as-Judge" to automatically score model outputs.
- Problem: This is a highly complex, expert-level reasoning task across five specialized domains (law, math, philosophy, etc.). The paper provides no evidence or validation study to prove that GPT-5-mini's scoring aligns with the judgment of human domain experts (e.g., a law professor or a mathematics PhD).
2. Agent Data Contamination Vulnerability
The paper claims Agents (like OAgents) outperform LLMs, attributing this to their capabilities. However, the experimental design has a fatal "open-book exam" flaw.
- Problem: 100% of the evaluation questions are sourced from publicly available, top-tier journal articles from 2023-2025. The tested Agents are permitted to use web search tools. This means an Agent can almost certainly find and "read" the original source paper for the question.
- Impact: The paper claims to test the "ability to acquire and reason over academic knowledge." However, this design cannot distinguish between "logical reasoning from scratch" and "information retrieval + answer extraction + paraphrasing." The high Agent scores (up to 65.1%) are likely inflated and may test search ability, not the "high-level reasoning" the paper purports to measure.
3. Flawed Construct Validity: Conflating "Reasoning" with "Knowledge"
This is the most fundamental issue: the benchmark likely fails to test "general reasoning ability" and instead tests "memory of specific, narrow knowledge."
- Problem: The experimental results show that Claude-4-sonnet, a model widely recognized for strong reasoning in other domains (like math and coding), scores a 0 on this benchmark.
- Analysis: This contradictory finding strongly suggests that ACADREASON does not test general, transferable logical reasoning, but rather whether a model happens to "know" the frontier theories from these 50 specific papers.
- Impact: The benchmark's construct validity is highly questionable.
  - For LLMs (like GPT-5): The test is more of a "memory test" (i.e., were these 2023-2025 papers in its training data?).
  - For Agents (like OAgents): The test is a "search test" (i.e., can it find the paper? See Flaw 2).

Overall Conclusion: The benchmark fails to successfully isolate the variable of "reasoning ability" from "specific knowledge-base." Therefore, the paper's conclusion that models like GPT-5 "lack reasoning ability" is unfounded.

### Questions
Based on the methodological analysis of the paper, we kindly request clarification and supplementary evidence regarding the following two core issues:
1. Regarding the Benchmark's Construct Validity
The paper claims that ACADREASON is designed to evaluate a model's "deep reasoning ability." However, the experimental results (for example, Claude-4-sonnet, which is widely regarded as having strong reasoning abilities, scoring 0) strongly suggest that the benchmark may be testing memory of specific, narrow, frontier knowledge (for LLMs) or search-and-extraction capabilities (for Agents), rather than transferable, general logical reasoning.
- Question: Can the authors provide additional evidence or control experiments to demonstrate that ACADREASON genuinely measures the core variable of "deep reasoning" and has effectively isolated it from potential confounding variables such as "specific knowledge-base" and "information retrieval ability"?
2. Regarding the Evaluation's Reliability
The paper's conclusions (particularly the extremely low model scores) are entirely dependent on the results from using GPT-5-mini as an "LLM-as-Judge." Given that these tasks span five highly specialized and complex domains (e.g., Law, Mathematics, Philosophy), the evaluation difficulty far exceeds that of common tasks.
- Question: Can the authors provide reliability validation for GPT-5-mini's role as the judge? For example, was an Inter-Annotator Agreement (IAA) analysis conducted between GPT-5-mini's scores and the scores from human domain experts (such as law professors or mathematics PhDs)? If such evidence is lacking, how can we trust the accuracy of these automated evaluation results?

### Soundness
1

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
The paper introduces ACADREASON, a multi‑domain benchmark intended to test high‑level academic reasoning of LLMs (and agents). It uses 50 expert‑constructed problems drawn from theoretical papers in computer science, economics, law, mathematics, and philosophy. Construction proceeds via (i) paper selection, (ii) extraction of a formal question with a golden answer, and (iii) derivation of question‑specific checklists and three types of hints (background/definitions/methodology). Evaluation adopts an LLM‑as‑Judge scheme (GPT‑5‑mini) with two metrics: Pass Rate (probability of full match to the golden answer) and Checklist Score (probability of meeting checklist criteria). Experiments show low pass rates for state‑of‑the‑art LLMs, higher but still limited scores for agents, and gains when methodology hints are provided.

### Strengths
* The paper is well motivated. It shows a clear gap in current reasoning benchmarks. By collecting research-level problems from recent top venues, the benchmark is both meaningful and timely.
* The task design encourages reasoning. Each question is self-contained, with a golden answer, a short checklist, and simple hints (background, definition, methodology). This supports step-by-step solutions and makes error analysis easier.
* The evaluation has broad coverage. It tests many state-of-the-art models and several agent systems across five domains. Domain-level results and hint ablations make the findings clear and easy to compare.

### Weaknesses
- The dataset is small. While the examples are high quality, the limited size reduces representativeness. A semi-automatic or LLM agent system might help scale up questions.
- The evaluation relies on a single judge (GPT-5-mini). There is no human calibration or test of consistency. Adding multiple judges (from different models, or more advanced LLM-as-Judge methods) and a small human study would make the results more reliable. 

Overall, the work is timely and useful as a benchmark. But it reads more like a benchmark release (more suitable for DMLR). For ICLR, I expect stronger methodological innovation—such as a calibrated multi-judge evaluation or a scalable automated pipeline—would make the contribution more suitable.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2
