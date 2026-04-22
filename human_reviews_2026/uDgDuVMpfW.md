# MultiHal: MultiLingual Dataset for Knowledge-Graph Grounded Evaluation of LLM Hallucinations

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Large Language Models (LLMs) have inherent limitations of faithfulness and factuality, commonly referred to as hallucinations. Several benchmarks have been developed that provide a test bed for factuality evaluation within the context of English-centric datasets, while relying on supplementary informative context like web links or text passages but ignoring the available structured factual resources. To this end, Knowledge Graphs (KGs) have been identified as a useful aid for hallucination mitigation, as they provide a structured way to represent the facts about entities and their relations with minimal linguistic overhead. We bridge the lack of KG paths and multilinguality for factual language modeling within the existing hallucination evaluation benchmarks and propose a KG-based multilingual, multihop benchmark called **MultiHal** framed for generative text evaluation. As part of our data collection pipeline, we mined 140k KG-paths from open-domain KGs, from which we pruned noisy KG-paths, curating a high-quality subset of 25.9k. Our baseline evaluation shows an absolute scale improvement by approximately 0.12 to 0.36 points for the semantic similarity score, 0.16 to 0.36 for NLI entailment and 0.29 to 0.42 for hallucination detection in KG-RAG over vanilla QA across multiple languages and multiple models, demonstrating the potential of KG integration. We anticipate MultiHal will foster future research towards several graph-based hallucination mitigation and fact-checking tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Problem & Gap. The paper introduces MultiHal, a multilingual benchmark for evaluating LLM hallucinations using knowledge‑graph (KG) paths as structured grounding, addressing two gaps in existing hallucination datasets: (i) limited or no multilingual support and (ii) reliance on unstructured text context rather than structured KGs.

Data Build. Starting from seven foundational English‑centric datasets (e.g., FELM, TruthfulQA, HaluEval, HaluBench, SimpleQA, DefAn, Shroom2024), the authors aggregate ~31k unique questions, mine ~140k candidate KG paths from Wikidata, and curate 25.9k high‑quality paths through an LLM‑as‑judge pipeline. The benchmark is translated into German, French, Italian, Portuguese, and Spanish.

Pipeline. Entities are linked from Q/A to KG nodes (Falcon 2.0 + DBpedia/Wikipedia fallbacks). Candidate 1–2 hop paths are fetched via SPARQL with answer‑type‑specific templates. A two‑stage LLM‑judge (GPT‑4o‑mini) first selects ≤10 paths, then scores each 1–5; only 4–5 are kept.

Benchmark Scale & Schema. One‑language slice reports 25,905 data points over 7,095 unique questions across 48 domains; schema includes IDs, source dataset, domains, answer types, subjects/objects, raw and formatted paths, labels, judge model, judge score, and language.

Baselines. KG‑RAG prompts (Q plus path) vs. vanilla QA across Gemini 2.0 Flash, GPT‑4o mini, Llama‑3.3‑70B‑Instruct. Metrics: semantic similarity (Multilingual MiniLM), NLI entailment, and hallucination detection (HHEM‑2.1, English only). Outcome: consistent improvements of KG‑RAG over QA across languages/models; ablations show higher LLM‑judge scores correlate with higher semantic similarity.

Headline Numbers. Reported absolute improvements: +0.12–0.36 semantic similarity, +0.16–0.36 NLI (entailment share), and +0.29–0.42 hallucination detection (consistent vs. hallucinated). Statistical tests (CvM) indicate significant distribution shifts favoring KG‑RAG.

Release & Ethics. Code and data are open (anonymized links); licensing is respected; CO₂ estimates and compute budget are documented.

### Strengths
Well‑scoped contribution. MultiHal targets an active area (hallucination mitigation) and squarely addresses two real gaps—structured KG grounding and multilinguality—with a principled, reproducible pipeline (SPARQL templates, entity linking, judge‑based filtering).

Scale and coverage. A substantive resource: 25.9k curated KG paths, 7,095 questions, 48 domains, 6 languages (EN + 5 EU). The paper details the domain mix and the dataset schema clearly.

Thoughtful curation with quality control. The two‑stage LLM‑as‑judge (shuffle + overlap selection; constrained extraction; three trials; remainder random‑sampled) plus score ≥4 retainment is carefully engineered; human–LLM agreement (Cohen’s κ=0.62) and error rates (FP 11%, FN 2.78%) are reported.

Triangulated evaluation. Going beyond a single metric, the authors corroborate semantic similarity with NLI and hallucination detection. Aggregated NLI shows sizeable shifts toward entailment in KG‑RAG for all three models and all languages; hallucination detection (English) similarly improves.

Transparent methodology. The paper includes SPARQL listings, answer‑type handling (dates/numerics), path cutoff date (Apr 2025), translation settings (NLLB‑200), prompts for KG‑RAG/QA, and significance testing (SW & CvM). This level of detail aids reproducibility and secondary analyses.

Limitations candidly discussed. The authors clearly surface failure modes (temporal/leading questions, domain mismatch like PubMed/Finance), sentence‑embedding pitfalls, translation quirks, and constraints (no multi‑turn, no span‑level hallucination scoring). This builds trust.

Practical value for KG‑RAG research. The dataset provides explicit paths (with labels and hop structure), enabling investigations of graph‑aware retrieval, routing, and knowledge injection methods beyond vanilla document RAG.

### Weaknesses
Judge–generator entanglement & closed‑source dependency. GPT‑4o‑mini is both the primary judge (for full experiments) and one of the generators in evaluation; this risks family‑specific biases (e.g., alignment to the judge’s stylistic/semantic preferences). The preliminary comparison to Gemini‑Flash as judge is helpful but limited; full runs still center on GPT‑4o‑mini. Also, the judge choice is closed‑source, challenging exact reproducibility.

Entity linking noise and path triviality. The paper acknowledges significant Falcon 2.0 noise and many low‑quality paths (necessitating heavy filtering). Even among retained paths, some examples suggest label‑leakage or trivial support (path labels effectively restate the answer), which may inflate KG‑RAG gains without testing deeper reasoning.

Temporal & domain coverage gaps. KG mining relies solely on Wikidata and 1–2 hops. Temporal questions and domain‑specific subsets (e.g., PubMed/Finance) show degraded or inconsistent behavior, and the paper suggests domain KGs as future work rather than executing them here.

Evaluation fragility.
- Semantic similarity with sentence embeddings is known to penalize concise/canonical answers vs. verbose references; the paper itself shows mismatches and negative deltas for certain domains.
- Single‑run evaluation (no seeds/re‑runs), which reduces statistical robustness; model comparisons are discouraged by the authors due to parametric/closed‑source differences, limiting cross‑model conclusions.
- Span‑level hallucination not reported; hallucination detection is only for English and via an external classifier.

Path selection fallback. When the judge fails to return 10 valid paths, the remainder are randomly sampled from candidates - potentially re‑introducing noise that the judge step tries to remove. Quantification of how often this fallback triggers is missing.

Translation quality control. Translations (NLLB‑200) are not human‑verified; the paper notes punctuation/semicolon issues and formatting sensitivity, which could harm both retrieval and evaluation in non‑EN splits.

Task framing and difficulty. The benchmark targets single‑turn QA with 1–2 hop paths; there is no difficulty stratification (e.g., reasoning depth, ambiguity, distractors) or analysis of multi‑hop reasoning beyond retrieval quality, which limits diagnostic power for advanced KG‑reasoning methods.

### Questions
Judge dependence and reproducibility.
- How often did the selection step fail to return 10 valid paths, triggering random fallback, and what was the impact on quality metrics?
- Can you report full baselines with an open‑source judge (e.g., strong instruct LLM) to de‑risk closed‑source dependence and judge–generator bias? 

Path informativeness vs. leakage.
- Do you control for answer‑label leakage in paths (e.g., where the final hop label exactly equals the gold answer)? Could you provide a filtered split where answer strings are masked or constrained, and report deltas?
- Any analysis on implicit‑vs‑explicit support (paths that require reasoning vs. those restating the answer)?

Entity linking & coverage.
- Since Falcon 2.0 is a major noise source, have you tried modern EL (e.g., BLINK variants or multilingual EL) and measured path yield/quality changes?
- Why the 2‑hop cap? Did you quantify yield/precision beyond 2 hops, and could you offer a difficulty tier with controlled hop counts?

Evaluation robustness.
- Could you provide seeded re‑runs and CIs, at least for a representative subset, to strengthen significance claims?
- For semantic similarity, can you augment with exact match / F1 on entity spans (where applicable) and calibration curves vs. NLI/hallucination signals to validate metric monotonicity?

Translations and multilingual QA.
- Did you perform human audits on translations (spot‑checks per language/domain)? If not, can you release a QC protocol and report estimated error rates?
- Are non‑EN runs evaluated against language‑specific references or back‑translations, and how do formatting issues (semicolon separation) affect failures?

Domain specialization.
- For PubMed/Finance subsets where KG‑RAG underperforms/varies, could you release a domain‑KG variant (e.g., PrimeKG/PubMed KG) and report a small‑scale comparison to demonstrate pipeline modularity in practice?

Task framing.
- Any plans for multi‑turn or reasoning‑chain variants (e.g., Think‑on‑Graph) using the same questions/paths to better test graph reasoning (not only retrieval)?
- Can you include a difficulty score per item (e.g., hop count, entity degree, distractor density) to support method diagnostics?

Release artifacts.
- Will you release per‑item judge scores, failure flags (e.g., fallback sampling), and exact SPARQL logs to facilitate ablations?
- Given licensing diversity of sources, can you confirm that redistribution terms are harmonized for the merged dataset (beyond CC‑BY‑4.0 for your additions)?

### Soundness
3

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
3

### Summary
This work aims to propose a new resource MultiHall based on the existing QA datasets as well as the open knowledge graph Wikidata, for evaluating the faithfulness and factuality (hallucinations) of LLMs. 

The new data is constructed automatically without human annotation, including steps of entity linking, knowledge graph path extraction, LLM-as-a-judge for assessment, and translation.

### Strengths
1. The resource aims to bridge an important gap -- lacking good multilingual datasources for evaluating LLM hallucination. 

2. The resource has been analysed with many evaluation results, including baseline experimentation.

### Weaknesses
1. LLM is used to assess the quality of the extracted knowledge graph paths, by analysing the correlation with semantic scores between the predicted and gold answers for each question. Sometimes, when a path is incomplete or only partially correct, the LLM can still give the ground truth answer. I'm afraid this method cannot ensure the fair evaluation of the quality of the paths themselves. 

2. The knowledge graphs are incomplete. The potential impacts of this factor are not considered in constructing the new resource. 

3. Data of the other languages are simply translated by a model. This may reduce the data quality in other languages.

### Questions
1. Wikidata is still incomplete. How do you deal with questions that have no paths in Wikidata that can infer the ground truth answer? 

2. Even some correct paths are extracted, the paths are still not complete. Will these incomplete paths impact the fairness in evaluating different LLMs.

3. How do you ensure the accuracy of entity linking in extracting paths from Wikidata?

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
This paper introduces MultiHal, a multilingual, knowledge-graph (KG)-grounded benchmark for evaluating LLM hallucinations, addressing critical gaps in existing English-centric.

### Strengths
Multilingual design fills a gap in cross-lingual factuality evaluation, where low-resource languages often suffer from higher hallucination rates.
Comprehensive evaluation using three models (Gemini 2.0 Flash, GPT-4o Mini, Llama 3.3 70bn) and three metrics (semantic similarity, NLI, hallucination detection with HHEM-2.1), ensuring result robustness.

### Weaknesses
KG path mining is restricted to 2 hops, potentially missing complex relational knowledge required for reasoning-intensive questions.
Lack of fine-grained hallucination localization limits the utility for debugging LLM behavior.
Reliance on closed-source GPT-4 Mini for path quality evaluation raises concerns about reproducibility.
No multi-prompt evaluation, despite the known sensitivity of LLM performance to prompt formatting.
Semantic similarity metrics underestimate performance, as model responses are compared to verbose ground truths.
Limited analysis of model behavior across domains.

### Questions
The paper mentions "inverted subject-object pairs" to account for Wikidata’s graph directionality. How were these pairs generated, and what percentage of high-quality KG paths originated from inverted pairs? This detail would clarify the necessity of this preprocessing step.
The paper does not report translation quality metrics (e.g., BLEU) for the 5 target languages. How was the accuracy of translated questions/answers/KG paths validated—e.g., via human evaluation of a subset—or was performance assumed based on Nllb200’s pre-existing benchmarks?
The paper notes that TruthfulQA’s ground truths often repeat questions, leading to low semantic similarity scores even for correct model responses. Did the authors adjust the evaluation pipeline to account for this, or is the reported semantic similarity an underestimate for such subsets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper investigates the usefulness of knowledge graphs as a tool for hallunication detection, proposing a RAG-based framework (MultiHal) for QA mapping in KGs and assess it against known baselines. The results demonstrate the usefulness of the method and can be seen as work in progress in this space.

### Strengths
+ The approach and use of KG path mining mmakes sense
+ quality evaluations and baselines analysis has been performed well
+  good to see several datasets used

### Weaknesses
Paper has some unclear assumptions and metrics, such as the quality score and the effect of multilinguality (or its important) has not been carefully assessed. 

small models have been used for analysis , limiting the impact analysis and generalizability. 

GNN methods not discussed/compared against

### Questions
The paper has an interesting set of metrics and results, but curiously, I was wondering how reliable the LLM-as-a-judge method is for quality score (and the quality score itself, if it can be quantified better).

Would the use of Graph neural nets or even classic graph-based methods give better indication of path reliability?

As a benchmark, how can the method be extended alognside existing frasmeworks? Currently the benchmark seems confined to Wikidata

data creation pipeline is heavily dependent on closed-source, external APIs, can this be improved?

### Soundness
3

### Presentation
3

### Contribution
3
