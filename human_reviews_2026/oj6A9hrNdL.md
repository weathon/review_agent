# ResearcherBench: Evaluating Deep AI Research Systems on the Frontiers of AI Research

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 8, 2

## Abstract
The emergence of deep research systems presents significant capabilities in problem-solving, extending from basic queries to sophisticated research tasks. However, existing benchmarks primarily evaluate these systems on web retrieval and report generation abilities, overlooking their potential for discovering, intergrating and generating insights in AI research. To address this gap, we introduce ResearcherBench, the first benchmark focused on evaluating the capabilities of these advanced, agentic systems — which we refer to as Deep AI Research Systems (DARS) — on frontier AI research questions. We curated a dataset of 65 research questions expertly selected from real-world AI research scenarios such as laboratory discussions and interviews, spanning 35 different AI subjects and categorized into three types: technical details, literature review, and open consulting. Our dual evaluation framework combines rubric assessment, which uses expert-designed criteria to evaluate insight quality, with factual assessment, which measures citation accuracy (faithfulness) and coverage (groundedness). We evaluated several leading commercial DARS and baseline systems. Our evaluation results reveal the strengths and limitations of these systems, with particular strength in open-ended consulting questions compared to technical implementation tasks. Such capabilities demonstrate the potential for DARS to serve as genuine AI research partners, representing a meaningful step toward AI self-improvement. We open-source ResearcherBench to provide a standardized platform for promoting the development of next-generation AI research assistants, hoping to foster a new perspective in AI research evaluation for scientific collaboration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ResearcherBench, a new benchmark for evaluating AI research systems on the task of understanding and extracting meaningful insights in research papers. ResearcherBench consists of 65 research questions across 35 AI subjects. The authors also propose a new evaluation pipeline that makes use of the proposed dataset involving expert-annotated evaluation criteria tailored to each research question.

### Strengths
* This paper introduces a dataset that includes 65 high-quality research questions. The authors asked expert evaluators to assess 932 research questions and selected 65 of them to ensure that the dataset contains only high-quality research questions.

* The proposed dataset includes expert-designed criteria for each research question. This is a clear advantage over existing datasets that expect a static rubric for evaluation.

### Weaknesses
My main concern lies in the novelty of the task and evaluation framework when compared to existing studies on related tasks such as report generation [1].

* **Novelty of the task.** This paper claims that "existing benchmarks primarily evaluate these systems on web retrieval and report generation abilities, overlooking their potential for discovering insights in AI research." However, it is unclear how the studied task substantially differs from the well-studied tasks like survey or report generation, which also often require extracting insights from papers.

* **Novelty of the evaluation framework.** The proposed evaluation pipeline using expert-curated criteria is reasonable, and I appreciate the annotation efforts. However, it appears to be similar to evaluation methods based on manually created references [1], and its advantages are unclear.

* **Rationale behind the dataset curation.** The authors claim to target "frontier AI research questions," but they do not systematically explain how the research questions in the introduced dataset are similar to or different from those in previous studies.

### Minor Limitations

* **Dataset size.** The introduced dataset includes only 65 research questions. While I understand the high cost of dataset creation due to expert annotation, the paper should still justify that a meaningful evaluation can be achieved with only 65 questions.

* **Missing human performance.** Including human performance in Table 2 would be helpful to assess how capable existing systems are for each task.

### Minor Comments

* I recommend using \citep when suitable.

[1] Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents. https://arxiv.org/abs/2506.11763, 2025.

### Questions
I expect responses to the points I listed in the first part of the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces ResearcherBench, a benchmark for evaluating advanced AI research systems (DARS) on real-world AI research questions. It includes tasks across 35 AI subfields and uses both rubric-based and factual assessments to measure reasoning depth, factual accuracy, and citation quality. Results show that systems like OpenAI Deep Research and Gemini perform well in generating insights but struggle with technical depth and reproducibility, highlighting current limits of AI as true research collaborators.

### Strengths
- The paper focuses on rapidly emerging class of AI tools designed to assist in real research tasks, moving beyond traditional QA or chat systems.  
- ResearcherBench is built from real research problems collected from expert interviews, lab discussions, and research forums, rather than synthetic or exam-style datasets. It closely reflects real-world scientific inquiry.  
- The study shows current systems perform well in generating insights and high-level reasoning but struggle with deep technical or reproducible details, providing valuable direction for future AI research assistants.

### Weaknesses
- The benchmark focuses solely on AI research questions, so generalization to other scientific disciplines remains unknown.  
- The evaluation measures single-turn performance, not long-term or iterative research interactions that mimic real collaboration between humans and AI.  
- The framework evaluates the quality of final answers, but not how systems search, reason, or decide what to include, limiting interpretability of model behavior.

### Questions
- How would the systems perform in long-term, multi-turn research collaborations instead of single-turn evaluations?  
- How to measure how systems actually search, think, and make decisions along the way?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces ResearcherBench, a new benchmark designed to evaluate Deep AI Research Systems on frontier AI research questions. The benchmark includes 65 carefully curated questions across technical details, literature review, and open-ended consulting, paired with a dual evaluation framework that assesses both insight quality and factual grounding of leading LLMs.

### Strengths
1, The paper assesses whether deep-research systems can provide meaningful insight and consultation on unsolved AI research problems; this is a timely and interesting problem as AI4Science is a rising trend;

2, It provides a new dataset;

3, Results show some emergent strengths and weaknesses not captured by prior benchmarks.

### Weaknesses
1, A notable limitation of this work lies in its evaluation methodology for frontier research questions. Although the benchmark emphasizes unsolved and open-ended research problems, the scoring relies primarily on fixed rubrics that reward coverage of predefined expert criteria. This is less reliable for those open, unsolved problems. A better way could be incorporating comparative or pairwise evaluation frameworks (e.g., tournament-style or blind ranking) that assess relative research merit and originality rather than absolute rubric compliance.
2, The evaluation does not capture whether the model produces surprising or genuinely helpful insights even though it claims overall helpful, but when and how---this is unclear. For frontier research tasks, simply covering known points is insufficient, as recent LLMs are already strong at that. I would suggest asking annotators to mark which segments actually advanced their thinking, and give more fine-grained analysis to discover the essential value of LLMs towards humans.

### Questions
N/A

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ResearcherBench, a new benchmark aimed at evaluating Deep AI Research Systems (DARS)—agentic systems designed to perform open-ended research tasks—on frontier AI research questions. The benchmark comprises 65 questions across 35 AI subfields, categorized into technical details, literature review, and open consulting. The authors design a dual evaluation framework combining: (1) A rubric assessment (human-designed, multi-criteria coverage scoring), and (2) A factual assessment (citation faithfulness and groundedness metrics). Several commercial DARS systems (e.g., OpenAI, Gemini, Claude, Grok, Perplexity) are evaluated using this framework. The paper concludes that DARS perform best on open-ended consulting tasks but struggle with technical detail and literature review questions. The benchmark is open-sourced for community use.

### Strengths
1. The paper clearly identifies a genuine gap in current DARS evaluation: most existing benchmarks focus on retrieval and summarization rather than assessing insight generation for frontier research tasks.

2. Questions are sourced from real-world research scenarios (labs, interviews, forums), with multiple annotators. 

3. The rubric + factual framework offers a complementary perspective on reasoning quality and factual reliability, which is more nuanced than single-score metrics.

### Weaknesses
1. The rubric coverage relies on automated scoring (o3-mini model), but the reliability of using LLM-as-a-Judge (even with an F1 of ~0.80) for nuanced insight evaluation is questionable. These tasks are inherently subjective.

2. Despite claiming expert input, many steps (insight extraction, initial drafting) depend on Claude-3.7-Sonnet, raising questions about the human-likeness of the benchmark itself.

3. There is no granular analysis of which topics or difficulty levels cause the most failure, which limits the interpretability of the evaluation outcomes.

4. The final results are judged entirely by automated systems. Without any qualitative human review of model answers, it is difficult to trust that the benchmark reflects actual scientific reasoning quality.

5. Although the paper frames itself as evaluating “frontier research”, it is limited entirely to AI. Generalizing conclusions about research assistance more broadly is premature.

6. A core methodological concern is the circular dependency between benchmark creation and evaluation. The benchmark positions itself as measuring whether DARS can behave like “expert researchers” — yet the supposed expert rubric and insight ground‑truths are themselves partially LLM‑generated (Claude‑3.7‑Sonnet performs insight extraction and drafts core rubric content). While humans later “review” and “refine,” it is unclear how much original human reasoning exists versus light editing of LLM output. This structure blurs the line between human‑defined truth and LLM‑defined norm, potentially baking the model’s epistemic biases into the gold standard it is later evaluated against. This risk is amplified by two additional problems:
- Benchmark contamination risk: Since modern frontier models share training data and emergent reasoning patterns, having one LLM generate key ideas and rubric criteria and another LLM evaluate performance risks epistemic contamination. The evaluation pipeline implicitly rewards models that think like Claude (or like the judging model), not models that demonstrate genuinely independent scientific reasoning. This undermines claims of measuring researcher‑like capability — instead, the benchmark may be measuring LLM‑to‑LLM imitation fidelity. 
- Unclear grounding of “correctness” for open‑ended research: The paper treats extracted “key insights” and rubrics as if they represent human‑agreed, authoritative ground truth. But frontier research questions — by definition — do not have fixed answers. Alternative viewpoints, emerging results, or novel hypotheses could be equally valid yet score poorly because they diverge from the LLM‑generated rubric. This raises a deeper epistemic issue: If the benchmark itself cannot guarantee the correctness or completeness of its ground‑truth insight set, what exactly is being measured? The factual assessment further assumes that the model must cite a particular set of sources, even though real research often draws on multiple valid references, emerging conversations, or unpublished expert knowledge. Penalizing models for providing facts that deviate from the benchmark’s curated context risks enforcing narrow compliance rather than genuine reasoning ability.

### Questions
1. How much of the rubric and insight generation process is human-authored versus LLM-generated?
While you mention expert review and refinement, the initial insight extraction and rubric drafting are performed by Claude-3.7-Sonnet. Can you clarify what percentage of the rubric criteria and wording was contributed by humans versus accepted from the model? Was any rubric content written independently of LLM suggestions?

2. How do you mitigate contamination from using one LLM to generate benchmark expectations and another LLM to evaluate responses?
Given that LLMs often share training data and reasoning styles, how can you be confident the benchmark is measuring generalizable research ability rather than model-to-model imitation? For example, wouldn’t Claude-like answers be favored in rubric alignment and factual assessments?

3. Why are open-ended research answers evaluated against fixed rubrics and citation sets?
In many research contexts, multiple valid answers and citation paths exist. If a model provides a novel or expert-justified answer that diverges from the expected rubric or cites a different source, how is this handled? Is there a mechanism to avoid penalizing correct but non-aligned outputs?

4. Was there any human evaluation of DARS outputs?
Aside from the small meta-evaluation for judge model selection, were any model responses reviewed by domain experts to validate whether rubric and factual scores match human intuitions about insight quality and correctness?

5. Why not provide per-subject or per-difficulty breakdowns of system failures?
A more detailed error analysis (e.g., which topics consistently trip up models) would help validate the benchmark’s utility and reveal meaningful performance patterns. Was this considered?

6. Can you justify generalizing your conclusions beyond AI research?
The paper implies that DARS are becoming viable partners for “frontier research” in general, yet your entire benchmark is constrained to AI. How should readers interpret this claim without cross-domain evidence?

### Soundness
2

### Presentation
2

### Contribution
2
