# HSCodeComp: A Realistic and Expert-level Benchmark for Deep Search Agents in Hierarchical Rule Application

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 4, 0

## Abstract
Effective deep search agents must not only access open-domain and domain-specific knowledge but also apply complex rules—such as legal clauses, medical manuals and tariff rules. These rules often feature vague boundaries and implicit logic relationships, making precise application challenging for agents. However, this critical capability is largely overlooked by current agent benchmarks. To fill this gap, we introduce \textsc{HSCodeComp}, the first realistic, expert-level e-commerce benchmark designed to evaluate deep search agents in hierarchical rule application. In this task, the deep reasoning process of agents is guided by these rules to predict 10-digit Harmonized System Code (HSCode) of products with noisy but realistic descriptions. These codes, established by the World Customs Organization, are vital for global supply chain efficiency. Built from real-world data collected from large-scale e-commerce platforms, our proposed \textsc{HSCodeComp} comprises 632 product entries spanning diverse product categories, with these HSCodes annotated by several human experts. Extensive experimental results on several state-of-the-art LLMs, open-source, and closed-source agents reveal a huge performance gap: best agent achieves only 46.8\% 10-digit accuracy, far below human experts at 95.0\%. 
Besides, detailed analysis demonstrates the challenges of hierarchical rule application, and test-time scaling fails to improve performance further. Codes and the benchmark will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces HSCodeComp, a benchmark of 632 expert-labeled products for assigning 10-digit customs tariff codes. The authors formulate the problem as high stakes for trade compliance and demonstrate a noticeable gap for LLMs, the best model/agent is ~46.8% exact match at 10 digits versus ~95% for human experts. The paper also reports accuracy at coarser HS levels (2/4/6/8/10 digits), shows common failure modes, and argues that automated tariff classification is still far from solved. The paper argues this gap demonstrates that current LLM agents are not yet reliable for trade classification.

### Strengths
1. Practical task with real world impact. Tariff classification is a hierarchical categorization problem very similar to other industry classification specifications. The paper justifies the problem on how the classification directly affects duties, compliance risk, and audits.
2. Furthermore, the current gap between humans and model/agents is demonstrated: top LLM agents still fail ~50% of the time at the final 10-digit code. 
3. Clear Expert-created workflow for ground truth: annotators gather product attributes (materials, intended use, etc.), consult official rulings/decision rules, resolve disagreements, and escalate difficult items. This data is not easily made via synthetic generation. 
4. Layered evaluation is reported at 2/4/6/8/10 digits that’s useful for probing heretical failures in the classification. 
5. Error analysis names specific failure modes (“valid but not chosen,” “outdated code,” material confusion like silicone vs. rubber) which is useful in grounding the nuances of the task in relation to alignment with language.

### Weaknesses
1. The paper evaluates hierarchical levels, but it’s not explicit whether HSCodeComp is meant to be a flat 10-digit prediction task, or a hierarchical decision process / constrained decoding task. Would this be simpler if the workflow commits to a 2-digit chapter, then refine constrained of the child nodes? That needs to be made explicit for reproducibility and fairness in future comparisons.
2. Metrics could be improved. Currently the paper uses exact-match, even if the model picked a code in the correct branch and only got the last two digits wrong. Furthermore, top n accuracy could be provided as an appendix. It would be useful to quantify near misses, sibling confusion, or a hierarchical distance for the analysis. The qualitative examples hint that these near misses are common. This weakens the interpretability of the 46.8% number.
3. Lack of a constrained-decoding/structured prediction  baseline. From my understanding, the baseline agents are allowed to output nonexistent or structurally invalid 10-digit codes, or codes that don’t match their own predicted parent. Would a trivial hierarchical decoder (predict chapter, restrict to valid children, backtrack if invalid) would cut out a whole class of hallucination errors? Not including that baseline makes it harder to tell how much of the gap is deep legal reasoning vs. just lack of structural constraints. Furthermore could this be a language vs token misalignment, and if the numerical codes were replaced with text labels, would the gap still persist?
4. Temporal stability is unclear, HTS codes change over time: some codes get split/retired, and new product categories appear. The paper discusses an outdated code failure mode. But it does not clearly state which HS/HTS revision date is considered authoritative for HSCodeComp, nor how future updates will be versioned. Without explicit versioning or a way to update future categorizations, long-term benchmarking and replication will be shaky. We see this in GICS classifications 
5. Representativeness and coverage is not clear, the dataset spans 27 HS chapters and 32 e-commerce categories, but in the dataset, how many unique 10-digit leaves are represented, how many examples per leaf, and whether this mix reflects everyday brokerage volume vs. being intentionally enriched for tricky, dispute-prone items. This matters for how generalizable the reported agent accuracy is on this set versus actual in production distributions. 
6. Rules hurting the agent needs one concrete example. The paper claims that giving agents human-written tariff “decision rules” sometimes degrades performance. That’s interesting and believable given similar nuances in legal and finance, but the paper doesn’t walk through a single case. One worked example would make that claim much more convincing.

### Questions
1. Is the intended task definition a flat 10-digit classification or hierarchical code selection step by step? Did the authors evaluate these two modes of classification?
2. Can you quantify “near misses”? How often is the model correct through 6 or 8 digits but off at the final branch? That seems essential for interpreting the ~46.8% 10-digit score. Can you quantify some distance metric in branches/leaves?
3. Have you tried a constrained or hierarchical decoder baseline, or a structured outputs approach that only allows valid descendants and blocks nonexistent or outdated codes? Did you experiment by replacing numerical codes with enums or hard string labels for language grounding? How much could this be a side effect of tokenization?
4. Which HTS revision (with date) are your gold labels tied to? And do you plan to release HSCodeComp as versioned snapshots over time to track tariff changes? How are you planning to update the benchmark when codes change in the future?
5. How many distinct 10-digit codes are in HSCodeComp, and what does the per-code sample count look like? Is this mostly one-shot per code, or do some leaves recur?
6. Could you include one concrete example where agent performance degraded when given a human tariff rule, and why? This is an interesting failure mode related to the task and can help future work understand where the gap stems from with regards to instruction following.

### Soundness
2

### Presentation
4

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
The paper releases \textit{HSCODECOMP}, a benchmark of 632 expert‑annotated e‑commerce products (27 HS chapters, 32 first‑level categories) to test agents’ ability to apply hierarchical tariff rules and predict 10‑digit HS/HTS codes from noisy, multi‑modal inputs (title, attributes, image, price, category, URL). Baselines cover 14 LLM/VLMs and six open‑source agents (plus three closed‑source systems on a 49‑item subset); the best system (SmolAgents + GPT‑5 (VLM)) attains 46.83% 10‑digit accuracy vs. 95.0% for human experts (Table 1, p.6; Fig. 1, p.2). The paper analyzes “overthinking,” test‑time scaling, image utility, and failure modes, and promises to release code and data. Claimed contributions: (i) a realistic, expert‑level benchmark for rule application (their “Level‑3” knowledge); (ii) a multi‑expert annotation/validation pipeline; (iii) broad LLM/agent baselines with analyses of think‑depth, images, and test‑time scaling

### Strengths
This paper tackles a timely and important challenge: applying rules for HS code classification rather than relying on open-ended retrieval. The motivation and problem space are clearly illustrated in Figure 1 (left side, page 2). 
The dataset and setup are realistic—inputs combine noisy product listings, structured attributes, images, and URLs—and ablation studies show that including images improves accuracy in several scenarios (Table 4, page 7; Table 10, page 36).
The data labeling process is carefully designed: two experts annotate each item, a senior adjudicator resolves disagreements, and a 10% spot check shows only 2% disagreement (Figure 3, page 5).
The authors compare a wide range of baseline models and report results consistently at all HS code levels (2-, 4-, 6-, 8-, and 10-digit). The benchmark includes strong models such as GPT-4o and Qwen variants (Table 1, page 6).
Finally, the paper provides thoughtful analysis, including (i) a study of “overthinking” behavior versus tool use (Table 5, page 7); (ii) an investigation of why scaling model size at test time brings limited gains (Figure 4, page 8); and (iii) a useful taxonomy of common failure types that highlights cases where predictions are “error-but-valid” (Figure 5, page 9).

### Weaknesses
First, the current evaluation metric is too narrow. It only counts exact 10-digit matches as correct, even when the model predicts a valid but slightly different code. The authors themselves note that many predictions are “Error-but-Valid.” This shows a need for more flexible metrics—such as hierarchical distance, agreement at higher HS code levels (2, 4, 6, or 8 digits), and a rule-consistency score. As it stands, many reasonable answers are unfairly marked wrong (Section 4.2, p. 5; Figure 5, p. 9).

Second, there is a factual mistake in the HS taxonomy explanation. Section 3 (“Output,” p. 4) incorrectly claims that “the last four digits (from 6 to 10) are country-specific,” when in fact only digits 7–10 vary by country, while the 6-digit level is globally standardized. This should be corrected for accuracy.

Third, the source and authority of the “rules” used are ambiguous. The paper relies heavily on eWTP tariff rules (Figures 11–12; Section 4.1) but doesn’t explain how they align with official WCO or HTS legal notes or with U.S. CROSS rulings. Depending on a commercial taxonomy may lead to discrepancies with authoritative references.

Forth, the evaluation has fairness issues. Closed-source models were tested on only a 49-item subset (Table 2, p. 6), making their results not directly comparable to the 632-item open benchmark. The authors also disabled webpage retrieval because it slightly reduced accuracy, yet this choice makes the task less realistic for genuine research workflows.

### Questions
$\textbf{Correct HS Hierarchy Definition and Terminology (Sec. 3 “Output,” p. 4)}$. The authors should fix the inaccurate description of the HS hierarchy. Clearly distinguish between the global levels (2-, 4-, 6-digit, standardized by the WCO) and the national extensions (8-, 10-digit, defined by each country’s tariff schedule). Ensure consistent use of “HS” versus “HTS” throughout the text.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces HSCodeComp, a benchmark for evaluating deep search agents on hierarchical rule reasoning. The task is to predict 10-digit Harmonized System (HS) codes for e-commerce products based on noisy text descriptions, while following multi-layer tariff rules. The motivation is that most existing benchmarks only test structured or open-domain reasoning, but none evaluate agents’ ability to apply complex, human-written hierarchical rules, which is important for real-world expert systems like legal or trade automation.

The benchmark is built from real e-commerce data with expert annotations and covers 632 products across 27 chapters. The authors test various LLMs and agent systems, finding that even the best combination eaches only 46.8% accuracy, far behind humans (95%). Ablation shows that vague and interdependent rules are the main difficulty.

### Strengths
1. The motivation is clear. he paper clearly identifies a missing evaluation angle: hierarchical rule following, which is indeed a challenging and realistic reasoning task.

2. The dataset is comprehensive. The dataset is built with expert validation and seems to capture realistic product diversity and textual noise.

3. The experiments compare many models and agent frameworks, giving a broad and fair view of the task difficulty.

### Weaknesses
I am not an expert in search agent. My concerns are only raised from the research perspective not specific to this certain domian.

1. Only 632 samples might be too few to show robust performance differences.

2. Since rules come from different sources (tariff codes, human rulings, etc.), it would be useful to test which part contributes most to performance.

3. I wonder how well models perform at intermediate steps (like predicting subcategories).

4. Maybe models tuned for other structured domains (finance, medicine) could generalize better. A small cross-domain test would strengthen the claim that the challenge truly lies in hierarchical rule reasoning.

### Questions
Please see the questions raised in weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
I think this is more like a technical report than a research paper, way below ICLR standard.

### Strengths
It has lots of LLM used

### Weaknesses
Quality of the paper is very poor.

### Questions
I would suggest the authors read more good papers and see how a good paper is written.

### Soundness
1

### Presentation
1

### Contribution
1
