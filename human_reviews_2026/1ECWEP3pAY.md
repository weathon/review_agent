# Dr.Mi-Bench: A Modular-integrated Benchmark for Scientific Deep Research Agent

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
The explosive growth of academic literature drives the need for automated deep research (DR) agents, yet evaluating these systems remains a significant challenge. First, existing benchmarks often focus narrowly on retrieval while neglecting high-level planning and reasoning. Second, existing benchmarks favor general domains over the scientific domains that are the core application for DR agents.  To address these gaps, we introduce **Dr.Mi-Bench**, a **M**odular-**i**ntegrated **Bench**mark for scientific **DR** agents. Grounded in academic literature, our benchmark uses a human-annotated dataset of **200 instances** across **10 scientific domains**, including both research and review papers. Furthermore, we propose a **M**odular-**i**ntegrated **Eval**uation Paradigm for **DR** Agents (**Dr.Mi-Eval**), which leverages the rich structure of academic papers to assess the core capabilities of planning, retrieval, and reasoning. It employs two complementary modes: an **end-to-end** evaluation for DR agents and an **isolated** evaluation for foundational LLMs as potential backbones. Experimental results reveal an uneven performance landscape: while agents show specialized strengths, they share critical limitations, particularly in multi-source retrieval for review tasks and maintaining consistency across diverse scientific fields. Moreover, improving high-level planning capability is the crucial factor for unlocking the reasoning potential of foundational LLMs as backbones. By exposing these actionable failure modes, Dr.Mi-Bench provides a diagnostic tool to guide the development of more reliable academic research assistants.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Dr.Mi-Bench, a human-annotated, modular-integrated benchmark for evaluating scientific deep research agents. It formalizes the agent workflow into three modules—Planning, Retrieval, and Reasoning—and proposes Dr.Mi-Eval, a dual-mode evaluation paradigm that supports both end-to-end agent assessment and isolated module testing of foundational LLMs. The benchmark comprises 200 instances across 10 scientific domains (research and review papers), with ground-truth plans, gold evidence, and diagnostic statements derived from the source papers. The study evaluates commercial deep-research systems and leading LLMs, reports efficiency trade-offs, and validates automatic metrics against human preferences. Key findings include a fragmented performance landscape, severe retrieval challenges on review-style tasks, and high-level planning as a bottleneck for unlocking LLM reasoning potential.

### Strengths
- Original modular-integrated paradigm that disentangles planning, retrieval, and reasoning, enabling diagnostic and attributable evaluation.
- Grounding in academic paper structure (outlines, citations, and content-based diagnostics) provides objective, reproducible signals for each module.
- Dual evaluation modes (end-to-end agents vs. isolated LLMs) offer both realism and upper-bound capability analysis.
- Diverse, human-annotated dataset across 10 scientific domains and both research/review tasks, improving relevance beyond general-purpose benchmarks.
- Clear, actionable empirical insights (e.g., review-task retrieval collapse, planning bottleneck, agent specialization) and practical efficiency analysis (latency, token trade-offs).
- Effort to validate automatic metrics with human preferences shows strong alignment, lending credibility to the evaluation design.

### Weaknesses
- I believe the benchmark’s generality could be greatly improved by including a simple Deep Research Agent framework to evaluate LLMs in an end-to-end manner. At present, it only evaluates LLMs in a modular way, and the results are not directly comparable to existing end-to-end systems like o3-deep-research.
- In the Planning Module evaluation, explain how the pairwise judgment matrix is converted into precision, recall, and F1 scores. Also define coverage, structural correctness, and redundancy, and include the exact steps or formulas used to compute each.
- In Tables, please include the unit (%) in the header for greater rigor.
- It would be better to demonstrate the variance of the performance to demonstrate the robustness of the results.


> “We consistently observe the highest performance in fields like Computer Science, Medicine, and Biology, which is likely attributable to the extensive representation of these domains in the models’ training corpora. In contrast, agents struggle significantly more in specialized areas such as Earth Science, Materials, and Building and Construction, which are likely less prevalent in the training data.” in L367-L371

- I think the description does not align with the figure. In Figure 2, Computation Science (score=0.54) is lower than Earth Science (score=0.57), Building and Construction (score=0.58), and Mathematics (score=0.56).

### Questions
- On the Planning evaluation: Why must the evaluation require alignment to the annotated plan steps? There are many valid ways to plan and decompose a research question. How does your metric accommodate semantically equivalent but structurally different plans, and avoid penalizing correct alternative decompositions?
- On the Diagnostics for reasoning: Why must the evaluation rely on a fixed set of diagnostic statements as the target? A correct report may not cover every diagnostic detail while still adequately answering the question.
- It states “the detailed formalization and scoring for each component,” (L143) but I could not find the actual scoring details in the appendix—only high-level descriptions.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Dr.Mi-Bench, a modular-integrated benchmark for evaluating “deep research (DR) agents” along three modules—Planning, Retrieval, and Reasoning—with two evaluation modes: end-to-end (agent) and isolated (foundational LLM as backbone). The benchmark consists of 200 human-annotated instances spanning 10 scientific domains and includes both research and review papers. The companion evaluation paradigm (Dr.Mi-Eval) scores planning via coverage/structure/redundancy against expert gold plans, retrieval via canonical-ID exact matches (DOI/arXiv/links), and reasoning via boolean diagnostics derived from each paper; it additionally reports efficiency trade-offs (latency vs. accuracy; token length vs. accuracy). Empirically, the authors report a fragmented performance landscape: different systems specialize in different modules (e.g., Grok strong in retrieval; Gemini in reasoning) while all struggle on multi-source retrieval for review-style tasks and on consistency across scientific fields; improving high-level planning is identified as a key lever for unlocking LLM reasoning when used as backbones.

### Strengths
1. Clear modularization of DR into Planning / Retrieval / Reasoning with dual modes (E2E vs. isolated) and a formal mapping from (Q,E,H,B) to grounded reports. This gives the community a shared lens to diagnose agents.

2. Paper-grounded supervision: gold plans from outlines, gold citations from bibliographies, and diagnostic T/F statements → objective, reproducible signals beyond BLEU-style report matching.

3. Actionable insights: (i) planning recall is a cross-model bottleneck; (ii) review-style retrieval remains unsolved; (iii) domain bias likely tracks training corpora composition.

4. Efficiency analysis that acknowledges practical constraints (latency, verbosity), with Pareto frontiers to visualize trade-offs.

5. Human alignment check for reasoning, offering some reassurance against grader artifacts.

### Weaknesses
1. Rigid retrieval gold for review papers.
Exact-ID matching to a paper’s bibliography may under-credit legitimate alternative or closely related sources (e.g., updated versions, foundational works not cited, thematic matches). A graded retrieval metric or semantic-equivalence layer would improve fairness and realism.

2. LLM-judge reliability not fully validated.
Planning and reasoning evaluation rely on LLM judgments. The paper references consistency checks, but lacks details on judge calibration, human–LLM agreement, or variance across different judges/prompts/temperatures.

3. Scale and representativeness limitations.
200 samples across 10 domains (~20 each) is a good start but small for broad claims about field robustness and domain generalization. Confidence intervals and domain-difficulty characterization are needed to avoid over-interpreting noise.

4. Potential selection bias from citation-based filtering.
Requiring ≥10 citations may bias toward already-visible work and reduce exposure to the “long tail” of scientific research where retrieval/generalization challenges may differ.

5. Limited annotation QA metrics reported.
The paper mentions guidelines but does not provide inter-annotator agreement (e.g., Cohen’s κ), distribution of plan lengths/diagnostics, or rejection/correction rates — making it hard to assess annotation consistency.

6. Risk of diagnostic-gaming in reasoning.
Boolean diagnostic checks encourage structured evaluation, but may lead models to optimize for checklist answers rather than holistic scientific synthesis. Complementary rubric or nugget-based human scoring on a subset would strengthen evaluation.

7. Reproducibility transparency could be improved.
Details on released artifacts (judge prompts, gold plans, bibliographies, toolkit settings) and handling of API variability would help ensure the community can reliably replicate results.

### Questions
1. Retrieval scoring:
Have you explored semantic citation matching (e.g., DOI families, preprint→journal mappings, thematic coverage) rather than strict exact-ID matching? How sensitive are results to this choice?

2. Judge reliability:
Could you report human–LLM agreement and/or LLM–LLM agreement for plan evaluation and diagnostic scoring, plus judge prompts, temperature, and calibration details?

3. Planning bottleneck ablation:
Can you show results where gold plans are injected while retrieval/reasoning are model-generated? This would validate planning as the dominant bottleneck.

4. Domain-level variance:
Please provide per-domain sample counts, confidence intervals, and difficulty statistics. Do cross-domain gaps persist after normalizing for citation density or terminology?

5. Dataset sensitivity:
How do conclusions change if the dataset size or review:research ratio is varied? Any signs of overfitting to this specific scale?

6. Efficiency evaluation details:
Was latency normalized for tool-use budgets? Did you measure gains after stripping boilerplate from model outputs or controlling for token verbosity?

7. Artifact release:
Which components will be released (gold plans, bibliographies, diagnostics, system + judge prompts, scoring scripts)? Will per-instance scoring breakdowns be included?

### Soundness
3

### Presentation
2

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
This paper introduces a new benchmark for deep research agents with 200 instances annotated by humans across 10 scientific domains from publicly available, highly cited research papers. There are 2 modes (end to end - for agents, and isolated - model only) and 3 categories of competencies (planning, retrieval, reasoning). The authors then evaluate a set of models and agents across these and then provide a number of breakdowns (by module, time / token / accuracy trade-offs, human preference validations). The benchmark is still far from saturated.

### Strengths
Relevance: Measuring model performance on deep research is a very timely and important area of research, and the academic domains selected are very important for the world (e.g., materials, finance, chemistry, computer science, medicine, biology, environmental science, energy, building and construction, and Earth science.)
Detail: The paper includes a clear breakdown of paper selection criteria, paper selection process, use of domain experts, task domains, example tasks, and quality management of the annotations. This detail helps validate that the process for task creation is likely to be high quality.
Evaluation modes: Evaluating both the agent mode and the mode for assessing baseline model performance makes a lot of sense.
Results: A thorough breakdown of results is provided with a variety of data cuts.
Presentation: The writing is clear and legible, and the figures are quite easy to understand.

### Weaknesses
Reproducibility: The full benchmark is not open-sourced or made available upon request, which makes reproduction or validation of the results not possible. I'd highly encourage you to make that available, or at least make a representative subset available (if you want to maintain a holdout set) so that the work can benefit the broader research community. If this changed, I would reconsider my score.
Ablations: Some further experiments would be helpful to make this work even stronger. For example, the authors could consider varying test-time-compute to measure if there is clean scaling and to ensure the benchmark isn't saturated when run with eg pass@k or best-of-N. Similarly, variations in prompting and capability elicitation could be employed to assess the benchmark's robustness to different eval setups. Without these, it is hard to tell if the benchmark is actually unsaturated, or if it is due to under-elicitation of model performance.
Minor nits: There are also a few typos (e.g., 'appendix ??' in page 16). Also, this is minor, but the title could be a bit cleaner ("modular-integrated" is a bit conflicting; the two modes might be a detail better left for the paper versus being the main part of the title).

### Questions
How were domain experts identified (i.e., were they always PhD candidates from the relevant field, or was it any PhD candidate judging any paper? how did you make sure folks were experts in the relevant fields to actually understand the underlying papers?)
On pg 16 you say "We computed inter-annotator agreement before adjudication and achieved a high level of consistency."---> what was the agreement rate?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper provides the benchmark (Dr.Mi-Bench) and evaluation framework (Dr.Mi-Eval) for deep research agents, a rising field of deep research systems evaluating the ability to plan and reason (and retrieve information, which was covered by previous evaluation benchmark/frameworks) in these agents, and specializes on professional and domain-specific cases. Given a research question, the proposed method scores plan generation and evidence retrieval using the  expert-annotated paper outlines and citations of real-world papers as reference, and the evaluation of reasoning is done on a set of 15-20 boolean diagnostic statements of the original paper tested on the generated. In addition, there are 2 modes to evaluating the agents' reasoning abilities, one using what plans and evidence they had gathered, and another isolated evaluation which makes use of gold plans and evidence. The benchmark comes from real-life research and review papers and makes up 200 instances across 10 scientific domains. The evaluation on state-of-the-art systems (OpenAI, Gemini, Perplexity, Grok) shows performance disparity across disciplines, such as the performance being higher on topics like computer science but worse on topics such as materials and earth science; while some agents excel at retrieval (Grok: 40% accuracy) or reasoning (Gemini: 60% accuracy), all struggle with planning (~25% F1) and with review-style tasks requiring multi-source synthesis.

### Strengths
1. Existing benchmarks (GAIA, BrowseComp, etc.) only test retrieval, not planning or reasoning. Dr.Mi-Bench addresses a significant gap in deep research evaluation by providing comprehensive, modular evaluation of planning, retrieval, and reasoning specifically for scientific research tasks.
2. The modular evaluation paradigm is a significant methodological contribution. By providing both end-to-end and isolated evaluation modes, it could reveal different failure modes and provides crucial information for future development.
3. The diagnostic statement approach is an effective, efficient, and reliable way to test the consistency between the generated report and the original paper.
4. The dataset is high-quality, annotated by experts and based on papers from top venues (Nature, ICLR, etc.) with >= 10 citations, ensuring its ecological validity, which is significantly more valuable than automatically annotated data.
5. The empirical evaluation is comprehensive, covering both commercial DR systems (OpenAI, Gemini, Perplexity, Grok) and foundational LLMs (GPT, Claude, Llama, Qwen). The cross-domain analysis (Figure 2) provides valuable insights about performance disparities across fields, and the efficiency analysis (Section 5.4) addresses practical deployment considerations often neglected in academic benchmarks.

### Weaknesses
1. While exhibiting high-quality, the size of the data might be not so big, and the expert-annotation mode might be hard to scale up. In addition, non-STEM fields could also be included into the benchmark. Cross-disciplinary research on itself could be difficult in real life as the collaboration between scholars of different departments and specialties, so it is one of the gaps that automated research agents are aiming to close. Therefore, adding domains such as the crossing between 2 of the studied research domains could also help.
2. One of the concerns is reproducibility due to the use of APIs, as the providers might discontinue an API endpoint, or update the model attached to that API. In addition, leakage could be an issue since it might be possible for the papers (2024+) to have been exposed to the called deep research systems during training, such as o3-deep-research and Search-rl. It is indeed near impossible to evaluate such API-accessed deep research systems, since we don't have control over the LLMs they use and the information they can retrieve. I think what we can do is to find the latest announced cut-off date in all the DRs you used, and only select papers after this date. 
3. It is not a full fix since we can't control how they retrieve, but we can always do post-processing after we retrieve. You see, it is impossible for paper a to cite the papers b, c, d that came after a which explores the same idea but with better methods. When we compare what the DRs had retrieved to what the paper a had cited, we might actually have artificially lower scores since papers b, c, d might dominated the retrieved evidence. We could remove the "future papers" out before evaluating them on evaluation.
4. Following the same issue, these "future papers" can also contaminate the final reasoning phase and make the scores higher than they should be, for it might be this scenario: we are asking the DRs to research on the topic A from paper a; paper a spawned paper b, c, d, etc, all based off paper a and are incremental to the solution proposed by paper a. By removing the "future papers" retrieved evidence, we can derive at more convincing numbers of the end-to-end evaluation for DR agents!

Among the issues raised above, I'd like to suggest one solution that is both implementable and important: temporal filtering of retrieved evidence. Before evaluating retrieved evidence, filter out any papers published after the source paper's publication date. This requires only metadata extraction (publication dates) and a simple filtering step during evaluation.

### Questions
1. The paper evaluates commercial deep research systems (OpenAI o3, Gemini, Perplexity, Grok) but does not report the monetary costs associated with completing research tasks. Given that users have limited control over these systems' behavior, understanding cost is crucial for practical deployment decisions. What is the average cost to complete one research with each of these DRs in this benchmark? I understand that as a user you might not have too much control on DRs, so it is important to record their cost on average, and also per task.
2. I believe it to be possible to apply the temporal filtering on the retrieved papers before evaluating the retrieval and reasoning capabilities I consider this recommendation high-priority because it addresses a fundamental validity concern with a practical, implementable solution. The current evaluation may substantially overestimate system capabilities by allowing access to forward citations and derivative works. Implementing temporal filtering would strengthen the benchmark's scientific rigor and provide more accurate measurements of deep research capabilities..
3. If possible, maybe we can also make sure that the papers used by the benchmark should be pass the cutoff dates of all the DRs, or if not, as new as possible, so less potential leakage in the LLMs used by these DRs. However, if implementing this requirement would necessitate extensive re-annotation and is not feasible for this submission, please explicitly acknowledge training data contamination as a limitation, and propose a versioning strategy for future updates with newer papers. I recognize this is a challenging constraint and defer to the authors' judgment on feasibility given time and resource constraints.

### Soundness
2

### Presentation
3

### Contribution
2
