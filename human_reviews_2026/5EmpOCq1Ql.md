# Characterizing Deep Research: A Benchmark and Formal Definition

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Information tasks such as writing surveys or analytical reports require complex search and reasoning, and have recently been grouped under the umbrella of _deep research_ --- a term also adopted by recent models targeting these capabilities. Despite growing interest, the scope of the deep research task remains underdefined and its distinction from other reasoning-intensive problems is poorly understood. In this paper, we propose a formal characterization of the deep research (DR) task and introduce a benchmark to evaluate the performance of DR systems. We argue that the core defining feature of deep research is not the production of lengthy report-style outputs, but rather the high fan-out over concepts required during the search process, i.e., broad and reasoning-intensive exploration. To enable objective evaluation, we define DR using an intermediate output representation that encodes key claims uncovered during search—separating the reasoning challenge from surface-level report generation. Based on this formulation, we propose a benchmark LiveDRBench with 100 challenging tasks over scientific topics (e.g., datasets, materials discovery, prior art search) and public interest events (e.g., flight incidents, movie awards). Across state-of-the-art DR systems, F1 score ranges between 0.02 and 0.72 for any sub-category. OpenAI's model performs the best with an overall F1 score of 0.55. Analysis of the reasoning traces reveals that systems cover only about half of the necessary search queries, with proprietary models issuing broader and and deeper queries than open source models, highlighting gaps in both coverage and reasoning depth. The benchmark is available at [this https URL](https://github.com/microsoft/LiveDRBench).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a benchmark for evaluating deep research systems. The authors first formalize deep research type of questions and focused on how it is different from other reasoning tasks. In addition, they evaluates DeepResearch agents on their proposed liveDRbench with 100 questions. Their evaluation is based on accuracy and completeness of claims from ground truth.

### Strengths
- The paper makes  make a  formal definition of what constitutes a Deep Research task. I think it is important to take a step to clarify these types of tasks and to distinguish it from multihopQA or long form reasoning.
- The ablation studies and experiments are relatively comprehensive.
- Their use of open web search and “problem inversion” makes the benchmark naturally extensible

### Weaknesses
- While the paper claims the benchmark can be easily updated via inversion of long-context tasks, in practice this may be labor-intensive which might require human verification of unique answers and updated ground truth each time the web changes.
- Some important DR-related systems as baselines are missing, such as OpenScholar or LLM+Tool-use agents that combine retrieval with API or browser actions. Including these would make the evaluation more representative of current “agentic” DR methods.
- Measuring coverage over issued queries rather than retrieved gold documents may be misleading. what if different query phrasings can retrieve equivalent document sets. Evaluating retrieval coverage (recall of relevant documents) would be a fairer proxy compared to query coverage.
- If I understood correctly, the claim-level F1 may penalize systems unfairly if entities are represented differently (e.g., “UCLA” vs. “University of California, Los Angeles”). Maybe using some form of soft matching could yield a more realistic measure of factual alignment.
- First, who were the human validators for the 100 questions? If the questions are truly expert-level and niche, it’s not easy for non-experts to validate them. Second, there should be some level of human validation on the evaluation itself to make sure the results are reliable.
- I think there is a chance that because the benchmark construction process uses automated inversion and possibly LLM assistance, some ground-truth claims or links might be hallucinated or incomplete. Human audits could help mitigate this.

### Questions
- How practical is the claim that LIVEDRBENCH can be “easily updated”? Given the need for verifying unique answers and refreshing ground truth as the web evolves, what is the actual human effort involved in maintaining the benchmark?
- Could the claim-level F1 metric unfairly penalize systems for surface-level differences in entity names (e.g., “UCLA” vs. “University of California, Los Angeles”)? 
- Who validated the 100 benchmark questions? If these questions are specialized and domain-heavy, how were they verified for correctness by non-experts? 
-Was there any human validation of the evaluation results to understand to what extend  the automatic scoring is trustworthy?
- Since the benchmark construction involves automatic inversion and possibly LLM-generated ground truth, how do you ensure that no hallucinated or incomplete claims made it into the final dataset?

### Soundness
3

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
3

### Summary
The paper proposes a formalization of “deep research” (DR) as tasks that are both search-intensive (high fan-out over information units) and reasoning-intensive (non-trivial effort to find/process/combine evidence). It argues that the core of DR is claim synthesis rather than long-form prose generation, and introduces LIVEDRBENCH, a 100-task open-web benchmark with a claims-based evaluation (precision/recall over nested claims, with strict scoring that zeros out a claim when any subclaim is wrong). The benchmark spans eight categories (SCIFACTS–Materials/Geo, NOVELDS—three types, PRIORART, ENTITIES, FLIGHTS), and evaluates proprietary DR systems and open-source baselines, reporting best overall performance for OpenAI’s DR model (avg. F1 ≈ 0.55) and analyzing search traces for breadth/depth coverage.

### Strengths
Clear problem decomposition. Separating “claims synthesis” from “report generation” is crisp and useful; the DAG framing of queries→evidence→claims aligns with how DR agents operate.

Objective evaluation for multi-claim outputs. The nested claim/subclaim metric operationalizes “grounded completeness” better than stylistic LLM-judge scores used in other DR evaluations.

Useful positioning vs. prior benchmarks. The paper contrasts LIVEDRBENCH with report-quality benchmarks (e.g., DeepResearch Bench) and browsing-centric datasets (e.g., BrowseComp) and motivates claim-level scoring for open-web DR.

### Weaknesses
Metric dependence on LLM judging and design choices. Claim agreement and “necessary query” identification rely on LLMs (GPT-4o) and bespoke prompts; the paper needs stronger validation (e.g., human adjudication on a stratified sample, inter-rater agreement, and sensitivity analyses for the “zero credit if any subclaim is wrong” rule).

Reproducibility of proprietary model comparisons. Evaluations for commercial DR systems are done via chat UIs without API parity; differences in browsing tools, guardrails, or hidden budgets could confound scores. More details (and repeated trials) are needed.

Scope overlap with related work could be made sharper. While the paper positions against BrowseComp (short, verifiable answers) and DeepResearchGym (static corpora like ClueWeb22/FineWeb), the narrative could more explicitly articulate complementary use—e.g., LIVEDRBENCH for open-web, claim-rich synthesis; DeepResearchGym for reproducible search sandboxes; DeepResearch Bench for report-quality LLM-judge eval.

### Questions
How robust are scores to judge model choice (GPT-4o vs. alternatives) and to noise in claim parsing? Please include a cross-judge concordance table.

For “necessary query coverage,” what fraction was human-validated, and what is agreement between annotators vs. LLM labeling?

How do results change if you relax the strict subclaim rule (e.g., soft weights), and if you weight claims by importance?

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
The paper proposes a formal definition of *deep research* (DR) as the sub‑task of claim synthesis over a large number of information units, and introduces **LiveDRBench**, a publicly released benchmark of 100 web‑search‑driven queries spanning scientific, dataset‑finding, prior‑art, entity‑listing and flight‑incident domains.  The benchmark is built by “problem inversion” of existing long‑context QA items and is evaluated through a claim‑based precision/recall metric that is computed by an LLM judge (GPT‑4o).  Using this infrastructure the authors compare three proprietary DR agents (OpenAI, Perplexity, Gemini) and two open‑source agents, reporting average F1 scores ranging from 0.02 to 0.72 and analysing trace‑level search coverage, depth, branching and back‑tracking.

The paper tackles an important, under‑explored problem and provides a promising benchmark.  However, the **evaluation methodology relies heavily on an opaque LLM judge**, and suffers from reproducibility issues (especially for proprietary agents).  These methodological shortcomings significantly limit confidence in the reported claims and hinder the benchmark’s immediate usefulness to the community. Please see my comments below.

### Strengths
- This is a timely and relevant problem: the framing also separates information synthesis (claims) from long-form report generation, improving construct clarity for DR evaluation.
- Claim-based evaluation design directly targets correctness/completeness of substantive content rather than stylistic quality; strict metrics are well-motivated for enumeration tasks.
- LiveDRBench spans multiple realistic DR use cases, including science and public-interest scenarios; explicit effort to make benchmark updatable is valuable.
- Transparent reporting of category-wise weakness (e.g., SCIFACTS Materials) and plausible sources of difficulty (subclaim extraction across long documents).
- Provides detailed prompts and evaluation pipeline in appendices; attempts to validate LLM-judging by a human pass that yields “similar” aggregates.
- Trace-analysis perspective (breadth/depth/coverage) is practically insightful and aligns with qualitative failure modes of DR agents.

### Weaknesses
- Proprietary DR systems were evaluated via chat UIs with unspecified/variable budgets; non-DR baselines used APIs with fixed “≈30s” reasoning but DR systems seemingly ran with much larger, uncontrolled search/time budgets. This undermines comparability and may inflate proprietary DR performance relative to baselines and open-source agents.
- Single-run reporting without variance: no per-task variance, confidence intervals, or statistical tests. Given high run-to-run variability in browsing agents and changing web results, this is a major limitation for drawing strong comparative claims.
- Ephemeral open-web setup without timestamped corpora or cached sources; no normalization for geographic or temporal search variability. The “Live” design is appealing but compromises replicability without web snapshots or logged artifacts.
- LLM-as-judge dependence: GPT-4o is used for matching, scoring, and even for trace analyses (necessary queries, dependency classification, branching/backtracking). Although authors include a “human” table with similar means, there is no inter-annotator agreement, correlation coefficients, or systematic error analysis. Heavy reliance on a single proprietary judge risks evaluation bias.
- Ground-truth expansion procedure: the rubric is augmented with any valid answers produced by the evaluated systems. While pragmatic, this can introduce dependence of the label space on model outputs, advantaging models that are included (and potentially those aligned with the judge). It also does not guarantee completeness as unseen correct answers remain out-of-scope, distorting recall.
- Metric design choices not stress-tested: the multiplicative/product formulation across subclaims can collapse scores when one subclaim is wrong; no ablations comparing product vs min (strict) vs additive/averaged schemes and their impact on ranking stability. Weighting scheme w_i is suggested but not analyzed.
- Claims that OpenAI DR “performs best” are plausible but not sufficiently supported for a benchmark paper because (a) budgets and interaction settings are not matched, (b) no statistical testing, (c) potential judge/model coupling (OpenAI DR and GPT-4o judge), and (d) evaluation performed on live web without controls.
- Trace-analysis conclusions (e.g., “coverage ~ half”) rely on GPT-4o-generated “necessary queries” and GPT-4o matching in traces, with no human validation or inter-rater reliability. As such, these are best viewed as exploratory rather than definitive evidence of breadth/depth gaps.

### Questions
1. Fairness/budget control: How many tokens, queries, and total wall-clock time did each system consume per task? Can you re-run proprietary DR systems with matched query/time budgets to non-DR baselines (and to each other) via APIs?
2. Variance and significance: Can you report per-task distributions, bootstrap CIs, and statistical tests for pairwise comparisons? How stable are rankings across multiple runs?
3. LLM-judge reliability: What is the agreement between GPT-4o and multiple independent human annotators (e.g., κ or Spearman/Pearson correlations per category)? Any cross-judge validation (e.g., Claude/LLama judges)?
4. Ground-truth expansion: How many items were added via model outputs per category? Did any system preferentially contribute to expansions? Can you freeze a versioned ground-truth independent of the measured systems to avoid circularity?
5. Replicability: Will you release time-stamped queries, full model traces, and archival snapshots of cited sources (e.g., WARC) so results can be reproduced?
6. Metrics sensitivity: How robust are system rankings to alternative claim aggregation strategies (product vs mean vs strict-min) and claim weighting? Please include rank correlations across metric variants.
7. Human evaluation details: How many human annotators participated in the “human‑evaluated” verification, what were their expertise levels, and what were the agreement statistics? 
8. Open‑source agent failures: Could you provide logs or error analyses for the DeepResearcher runs that “infinite‑looped” or “failed”?  Might alternative retrieval back‑ends or hyper‑parameter tuning change their performance?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors define Deep Research (DR) as tasks that require both broad search and deep reasoning.

They formalize it using a claim, subclaim graph that can capture information synthesis quality rather than just report writing quality. 

They propose LiveDRBench, an open-web benchmark that evaluates DR systems with claim-based precision, and recall instead of just report quality evaluation. 

They evaluate several open and commercial models, like OpenAI’s Deep Research, Gemini, and DeepResearcher and they show performance gaps between them across different categories and challenges.

### Strengths
The authors investigate the important task of Deep Research, which has broad applications across in both academia and industry.

They propose a new benchmark, livedrbench, that has 100 tasks and is used to evaluate how well DR agents, both open source and commercial, can identify the right claims and subclaims to produce comprehensive, well-grounded report.

They include in the benchmark DR tasks across multiple categories, including Materials, SciFacts, and Geo.

### Weaknesses
The authors use LLM-as-a-judge, but there is no human evaluation to ensure alignment with human scores.
For example, DeepResearchGym has a nice human evaluation process that validates its LLM-as-a-judge scores against human judgments [2].

The claim/subclaim idea is very similar to the one in Mind2Web 2 [1], where the evaluation agent checks whether specific claims are present in the report itself in a hierarchical manner. In fact, Mind2Web2 seems to be more precise at identifying facts as it breaks down the claims/subclaims in a nice yes and no tree.

The precision scoring mechanism proposed by the authors is more of a factuality check (is the claim supported by the citation), it does not seem to capture whether a claim is actually relevant to the DR Query.  For this, the authors should collect hard negative claims as part of the groundtruth, and if the agent puts them in the report they should be counted as false positives (which relates to precision)

The authors argue that most existing works do not check for claim correctness and focus on report quality, however, DeepResearchGym [2[ and other existing works use strong RAG-based validation to test factuality and recall of the groundtruth claims, so the distinction between this paper and that prior work is unclear.

It is also unclear what the contributions are of this paper in comparison to existing benchmarks. In Table 1,  the authors claim that DeepResearchGym lacks objective scoring, but that is false in my view since DeepResearchGym uses RAG to assess claim recall and most importantly includes human evaluation to ensure that their llm-as-judge align with human evaluation (which is missing in this paper).

There really needs to be a human  evaluation to make sure that the llm-as-a-judge used is aligned with human feedback. Also, there is no sensitivity analysis of how the results differ if the evaluation is ran multiple times. LLM-as-a-judge can have high variance in its scores.

 Furthermore, the "easy to update" claim in Table is weak, as all existing benchmarks seem to be easy to update. In fact, DeepResearchGym has 96,000 tasks, which is far more than the 100 tasks in LiveDRBench, which suggests that the former is actually easier to update and scale.


 [1] Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge
 [2] DeepResearchGym: A Free, Transparent, and Reproducible Evaluation Sandbox for Deep Research

### Questions
How do the authors make sure that the LLM-as-a-judge scores match human judgments, since there is no human evaluation in the paper?

How is the claim and subclaim evaluation different from the one used in Mind2Web 2 [1] and DeepResearchGym [2]?

Why do the authors say that DeepResearchGym has no objective scoring when it already uses RAG-based validation and human evaluation to make sure it is objective?

What happens if the agent just copies and pastes all the content from the search results into the report, would it still get a high score?

 [1] Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge
 [2] DeepResearchGym: A Free, Transparent, and Reproducible Evaluation Sandbox for Deep Research

### Soundness
3

### Presentation
3

### Contribution
2
