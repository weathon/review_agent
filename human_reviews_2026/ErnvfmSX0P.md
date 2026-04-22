# ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Deep Research (DR) is an emerging agent application that leverages large language models (LLMs) to address open-ended queries. It requires the integration of several capabilities, including multi-step reasoning, cross-document synthesis, and the generation of evidence-backed, long-form answers. Evaluating DR remains challenging because responses are lengthy and diverse, admit many valid solutions, and often depend on dynamic information sources. We introduce ResearchRubrics, a standardized benchmark for DR built with over 2,800+ hours of human labor that pairs realistic, domain-diverse prompts with 2,500+ expert‑written, fine‑grained rubrics to assess factual grounding, reasoning soundness, and clarity. We also propose a new complexity framework for categorizing DR tasks along three axes: conceptual breadth, logical nesting, and exploration. In addition, we develop human and model-based evaluation protocols that measure rubric adherence for DR agents. We evaluate several state‑of‑the‑art DR systems and find that even leading agents like Gemini's DR and OpenAI's DR achieve under 68% average compliance with our rubrics, primarily due to missed implicit context and inadequate reasoning about retrieved information. Our results highlight the need for robust, scalable assessment of deep research capabilities, to which end we release ResearchRubrics (including all prompts, rubrics, and evaluation code) to facilitate progress toward well‑justified research assistants.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ResearchRubrics, a benchmark to evaluate Deep Research (DR) agents. It covers eight domains (i.e., STEM, health, finance, legal, and common consumer questions), and the rubric criteria, written by human experts, check factual grounding, coherence of reasoning, completeness, relevance, and clarity of the answer. Applying LLM-as-a-judge to assess rubric compliance, the existing DR systems achieve a poor performance.

### Strengths
1. The proposed benchmark is intuitive, well-designed, and well-curated.
2. Experimental results on multiple DR systems demonstrate that the current DR agents struggle with the proposed benchmark.

### Weaknesses
1. Only three human experts with a STEM background are involved in the benchmark construction process. (See Question 1 below)
2. Some points to be clarified. (See Questions 2, 3, and 4 below)
3. The proposed DR rubric criteria need further justification. (See Question 5 below)
4. No case study or error analysis to provide insights into why the current DR agents fail to achieve a higher score on each of the proposed rubric criteria.

### Questions
**Questions**:
1. Section 3.1 Data Collection and Task Domains: There are only "three expert participants" with "a strong STEM background". However, the proposed benchmark covers eight domains, four of which are not STEM. How could the data quality of those domains be guaranteed?
2. Section 3.2 Prompt Complexity Dimensions: How is each ResearchRubrics task categorized? Is it categorized by human experts/annotators or LLMs? What are the criteria for categorizing?
3. Section 3.2: The proposed "task complexity framework" measures three dimensions: "(1) conceptual breadth (the number and diversity of distinct topics or domains involved), (2) logical nesting depth (the number of reasoning or decision steps required, including sub-questions and conditionals), and (3) exploration level (the degree of open-endedness or underspecification of goals)." What are the statistics of it? For example, as in Table 1, how many instances are categorized as "Simple", "Moderate", or "High" Conceptual Breadth?
4. Section 3.2: What are the insights of knowing the prompt complexity categories? Did such categories guide the construction of the proposed ResearchRubrics benchmark?
5. Why are the proposed DR rubric criteria (as in Table 2) valid and enough to evaluate DR agents? Are they devised by pure heuristics/intuitions, or are they based on a certain theory or relevant literature?

**Typos**:
1. Line 65: "LLM-as-judge" $\to$ "LLM-as-a-Judge"

### Soundness
2

### Presentation
3

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
- The paper introduces ResearchRubrics, a benchmark for evaluating “Deep Research” (DR) agents—LLM-powered systems performing multi-step web exploration, retrieval, and synthesis to produce evidence-backed, long-form answers.
- The benchmark comprises 75 prompts across eight broad domains (STEM, health, finance, legal, consumer research, current events, historical analysis, creative writing, business planning), each paired with expert-authored, fine-grained rubrics (1,868 total criteria). Rubrics include both positive and negative criteria, with weights capturing importance and severity.
- The authors propose a tri-axial task complexity framework: conceptual breadth, logical nesting depth, and exploration level, to characterize and analyze DR tasks systematically.
- They develop an LLM-as-judge protocol with ternary and binary grading schemes and conduct human consistency analysis (Macro F1), showing moderate-to-substantial agreement depending on the grading regime. They perform ablations on rubric detail and LLM-augmented rubrics.
- They evaluate leading commercial DR agents (OpenAI DR, Gemini DR, Perplexity DR) and baseline LLMs with search tools, reporting sub-60% average rubric compliance, with primary failures on implicit reasoning, synthesis, and citation rigor. They plan to release the dataset, rubrics, and tools.

### Strengths
- Clear motivation: DR tasks are open-ended, dynamic, and require long-form synthesis; existing QA-style benchmarks and short-answer datasets underrepresent these needs.
- Human-authored rubrics: The choice to use carefully designed, expert-written rubrics (including negative criteria and weighted mandatory vs. optional items) is a thoughtful departure from purely automated reference-based metrics and helps capture nuanced expectations.
- Fine-grained evaluation: The rubric axes (explicit, implicit, synthesis, references, communication, instruction following) provide multidimensional diagnostics beyond factual correctness, surfacing where current agents struggle (implicit context and synthesis).
- Task complexity framework: The breadth-depth-exploration triad is sensible and useful for labeling, stratifying tasks, and analyzing performance across different dimensions of difficulty.
- LLM-as-judge methodology with human consistency checks: The inclusion of ternary vs. binary verdicts, Macro F1 consistency with human annotators, and ablations on rubric clarity are solid evaluation engineering contributions. Practical insights (binary verdicts yielding higher consistency; examples helping; LLM-augmentation of rubrics hurting) are valuable to the community.
- Diversity of domains and realistic tasks: Inclusion of everyday research queries alongside STEM and business/legal topics reflects real user scenarios for DR agents.
- Transparency about evaluation window and pipeline: Fixing the evaluation window for dynamic content (July 2025) and describing the collection (PDF -> markdown chunks -> LLM judge) helps reproducibility.

### Weaknesses
- Scale and representativeness: 75 tasks and 1,868 criteria are substantial in rubric detail but relatively small in task count for a general-purpose benchmark, raising questions about coverage and generalization across the wide variety of DR use cases.
- Expert definition and domain specialization: “Experts” are defined as strong STEM generalists rather than domain specialists (e.g., legal, medical). For domains with high stakes or regulatory complexity, lack of specialist involvement may reduce rubric validity and could introduce bias or omit critical domain-specific requirements.
- Judge reliability and circularity: While the paper argues against circularity by using human-written rubrics, reliance on closed-source LLM judges and moderate human agreement in ternary settings (Macro F1 ~0.48–0.55) raise concerns about robustness and potential judge bias. There’s limited analysis of inter-annotator agreement among humans or calibration strategies for judges beyond F1.
- Weighting scheme and scoring design: The rubric uses weighted sums normalized by absolute weights, with partial credit (0.5) for “Partially Satisfied.” The justification for specific weights (e.g., what constitutes “Critically Important” vs. “Important”) is largely qualitative. Sensitivity analyses to weight choices and to the partial-credit scale are not reported.
- Inconsistencies and clarity issues:
  - Category failure percentages in Table 3 appear to sum well above 100%, which suggests overlap or double-counting across axes; this needs to be clarified (e.g., whether criteria carry multiple axis tags or how proportions are computed).
  - Minor inconsistencies in judge model naming across the main text and appendix (e.g., GPT-4.o vs. GPT-4.1; inclusion of o3 later).
  - Editorial issues (duplicate “Creative Writing” in the domain list, typos visible in figure text) detract from presentation quality.

### Questions
- Rubric and weighting validation:
  - How were weights calibrated? Did multiple annotators assign weights independently? Please provide inter-rater reliability or calibration metrics for weights and for the ternary labels on a subset.
  - Consider sensitivity analysis: How do final scores and agent rankings change under different weighting schemes or partial-credit values?
- Category failure percentages:
  - Clarify how failure rates per category are computed. If criteria have multiple labels (e.g., both synthesis and implicit), explain how percentages and totals are derived to avoid summing >100%. Consider reporting per-axis normalized failure proportions.

### Soundness
3

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
4

### Summary
This paper introduces a benchmark for evaluating deep research agents. It contributes 75 prompts and 75 rubrics across 8 domains, of which there are 1868 human written criteria for what the question designers expected to see in a good answer. The rubrics are created through the following process:

Human 1 drafted initial prompt and rubric terms.
Human 2 reviewed draft, provided feedback, and iterated with Human 1 until both agreed it was clear and complete.
Human 3, final reviewer, checked for clarity balance and bias and made final adjustments.

The rubrics are created before any agents are asked to generate answers; rubrics can involve both positive and negative criteria and have different attributes at different weights. Deep research agents were then asked to complete the queries, and their answers were scored with these rubrics. No inter-annotator agreement was measured, but instead the paper does human-LLM agreement where a human and a LLM scored the same response with a rubric, and this showed moderate human-LLM agreement.

Generally, the finding is that SOTA deep research agents achieve at maximum 0.59 rubric compliance, with good factual recall but weak synthesis, implicit reasoning, and citation quality.

### Strengths
Diagnostic clarity. Not many works exist to evaluate the output of deep research agents, and this work provides a meaningful contribution towards that front. The rubric design makes sense- and helps to evaluate the open-ended nature of these very long responses. The work provides a meaningful step to make sense and introduce clarity into the messy subjective space of research quality into a more structured and reproducible measurement framework.

### Weaknesses
The paper's main weakness is on validity.

Disagreements occur in rubric creation, so the final produced rubric through the 3 stage process masks inherent disagreement and tries to measure progress against a rubric that would be created by the average human. No IAA measures or analysis of disagreements occur; the main question, of a benchmark, is if progress on said benchmark would demonstrate a meaningful improvement on the task for end users. It is unclear from the rubric creation process that this can be convincingly said. Moreover- in the creation process, rubrics are fixed before seeing any output. The overarching paper claim seems to be that hey, these deep research agents are producing outputs that misalign with human preferences. However, it is unjustified that this creation process of rubrics is able to measure true human preference of deep research outputs- it's very easy to say that hey, I may think I want to see certain aspects before I actually see it, but in reality this shifts significantly from what I said when I actually see the output. It is unclear and unjustified that this rubric creation process can actually result in rubrics that reflect true human preference.

Minor nits, but it's important to note that deep research agents also produce a ton of stochasticity due to their long executions; controlling for this in some degree would make the results more robust. As it currently states, each agent was run once per prompt, with no reported stochastic variation or repeated trials. Finally, 75 rubrics and outputs is pretty low for this type of benchmark, especially when spread across so many domains; this is no concern if the authors can justify that performance on their benchmark translates to demonstratable improvements in real world deep research agent use, but this is not yet convincing.

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Research Rubrics, a structured framework to evaluate how well large language models (LLMs) perform scientific reasoning. It scores models across four dimensions: problem understanding, reasoning process, solution design, and scientific contribution. Experiments on multiple state-of-the-art models show that LLMs can follow scientific logic and structure but still lack genuine creativity and practical validation.

### Strengths
- The paper focuses on an underexplored but important problem: evaluating whether LLMs can reason like scientists rather than just answer questions. 
- The proposed Research Rubrics provides four well-defined dimensions: Problem Understanding, Reasoning Process, Solution Design, and Scientific Contribution. 
- The experiments benchmark multiple leading models (GPT-4, Claude 3, Gemini, Qwen2, Mistral) and compare different prompting strategies (chain-of-thought, critique loop, research-plan prompting). The setup is comprehensive and systematic.  
- The paper shows that while LLMs can follow scientific logic and structure, they still lack genuine creativity and originality, which is an important insight for future research on AI scientific reasoning.

### Weaknesses
- The study mainly focuses on computer science problems (e.g., ICLR/NeurIPS-style tasks), so it’s unclear how well the framework generalizes to other scientific domains.  
- The framework evaluates the quality of reasoning, but it does not assess whether the model’s ideas could actually produce valid or impactful scientific outcomes.
- Model performance depends heavily on prompt design (e.g., “research-plan” prompts boost scores), which suggests the framework might partly reflect prompt engineering skill rather than pure reasoning ability.

### Questions
- Can the framework generalize beyond computer science to other scientific domains such as biology and physics?  
- To what extent does the framework measure true reasoning ability rather than the effects of prompt engineering?

### Soundness
3

### Presentation
3

### Contribution
3
