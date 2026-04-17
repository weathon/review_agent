# WebWeaver: Structuring Web-Scale Evidence with Dynamic Outlines for Open-Ended Deep Research

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
This paper tackles \textbf{open-ended deep research (OEDR)}, a complex challenge where AI agents must synthesize vast web-scale information into insightful reports. Current approaches are plagued by dual-fold limitations: static research pipelines that decouple planning from evidence acquisition and monolithic generation paradigms that include redundant, irrelevant evidence, suffering from hallucination issues and low citation accuracy. To address these challenges, we introduce \textbf{WebWeaver}, a novel dual-agent framework that emulates the human research process. The planner operates in a dynamic cycle, iteratively interleaving evidence acquisition with outline optimization to produce a comprehensive, citation-grounded outline linking to a memory bank of evidence. The writer then executes a hierarchical retrieval and writing process, composing the report section by section. By performing targeted retrieval of only the necessary evidence from the memory bank via citations for each part, it effectively mitigates long-context issues and citation hallucinations. Our framework establishes a new state-of-the-art across major OEDR benchmarks, including DeepResearch Bench, DeepConsult, and DeepResearchGym. These results validate our human-centric, iterative methodology, demonstrating that adaptive planning and focused synthesis are crucial for producing comprehensive, trusted, and well-structured reports.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes WebWeaver, a dual-agent “planner + writer” framework for open-ended deep research. The planner dynamically produces a structured outline with linked evidence; the writer then retrieves the required evidence section by section and composes the content hierarchically based on the outline. The authors conduct experiments and ablations across multiple benchmarks, and the results demonstrate the effectiveness of the approach.

### Strengths
1. The paper is easy to follow, the figures are clear, and the experimental details are thorough. The final results look solid.

2. It folds “dynamic outline optimization + evidence binding” into the planning loop and uses section-wise retrieval + writing, which mirrors how humans actually do deep research.

3. It releases a 3k SFT dataset and shows gains on smaller models, which supports the method’s effectiveness.

### Weaknesses
1. The planning stage parses 100+ pages and burns tens of thousands of tokens, but there’s no apples-to-apples report of wall-clock latency or API cost vs. baselines.
2. It is recommended to include parameter-matched comparisons across different frameworks built on the same model backbone, or reproduction experiments against other strong baselines on the identical model.

### Questions
See weakness

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
2

### Summary
The paper presents WebWeaver, a dual-agent system for deep research questions. They distinguish their approach from previous methods like “search-then-generate” or “outline-then-write”  by  introducing a planner–writer loop in which the planner iteratively refines a dynamic outline and the writer performs hierarchical writing. They run experiments on DeepResearch Bench, DeepConsult, and DeepResearchGym and showed they outperformed baselines.

### Strengths
- The paper addresses an important and and timely problem.

- The dual-agent design  is very tangible and human-inspired very well motivated.

- Comprehensive experiments across three strong benchmarks

- comprehensive ablation studies

### Weaknesses
- Missing cost analysis compared to baselines 
The paper does not clearly specify when the planner decides to terminate the outline optimization process—what the stopping criteria or thresholds are.

- Several important open-source baselines are missing, such as DeepResearcher, OpenScholar, and open sourced LLM + tool-use systems. Including these would strengthen claims of generality and fairness.

- The WebWeaver-3k dataset is also said to be built from “diverse queries crawled from the web,” yet the crawling criteria, domains, or filters are unspecified.

### Questions
- Did you do any analysis on what actiona cause the most problem of answering wrong? for example things like early termination?
-  how were the web queries for WebWeaver-3k crawled—were they topic-balanced or randomly collected?
 - How would webweaver as a dual agent do in terms of cost compared to single agent architechture

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces **WebWeaver**, a dual‑agent system (Planner + Writer) for “open‑ended deep research” (OEDR). The Planner iteratively interleaves web search and outline optimisation; the Writer then synthesises a long‑form report section‑by‑section using a citation‑grounded memory bank. The authors claim that this human‑inspired workflow eliminates the “loss‑in‑the‑middle” problem, reduces hallucinations, and yields state‑of‑the‑art results on three recently released OEDR benchmarks (DeepResearch Bench, DeepConsult, DeepResearchGym). They also present a 3 k‑example SFT dataset (WebWeaver‑3k) that purportedly enables a 30 B‑scale model to match the performance of much larger proprietary agents.

### Strengths
- Clear, human-inspired, end-to-end system design with an explicit evidence chain: planning → evidence extraction into a memory bank → outline with citation IDs → retrieval-grounded writing. The qualitative case study is coherent and illustrates the intended workflow.
- Across three reasonably comprehensive benchmarks, results indicate competitive to SOTA performance. On DeepResearch Bench, the reported citation accuracy (93.37%) and effective citations are particularly strong. On DeepConsult and DeepResearchGym, the win rate and average scores are strong relative to listed systems.
- Ablations address two core hypotheses: (1) iterative outline optimization improves depth/breadth/support and end-to-end scores, and (2) hierarchical, citation-driven writing outperforms brute-force context stuffing. These analyses are aligned with the proposed mechanism of action.
- Useful operational statistics (number of searches, outline size, number of saved pages, evidence tokens) convincingly argue that memory-centric, targeted retrieval is not just helpful but necessary under the reported scale (>100 pages, >60k evidence tokens).
- The SFT dataset and fine-tuning experiments show that the framework can act as a data engine for transferring agentic skills to smaller models, with large gains in citation accuracy. This is a practical contribution for accessibility.

### Weaknesses
- Cost and budget parity are not controlled or reported. WebWeaver averages ~100+ pages parsed, ~67k evidence tokens, and ~26k-token outputs. Many baseline systems (especially proprietary) may have stricter rate/token limits. Without a controlled per-task budget (time, API calls, tokens), performance advantages may partly reflect higher spend rather than method.
- Baseline comparability is unclear. WebWeaver is evaluated with multiple powerful base LLMs (Qwen-235B, Claude Sonnet 4), and even when the “agent model” is Claude, a separate model (GPT-oss-120B) is used for URL selection and evidence extraction. This multi-model, auxiliary-LLM design could materially boost performance yet is not mirrored for baselines. It is therefore difficult to isolate method gains from model capability/tooling differences.
- Statistical analysis is thin. No confidence intervals, no statistical tests, and no repeated runs are reported. Improvements on RACE overall scores are modest versus strong baselines (e.g., ~1 point vs Gemini 2.5), and the absence of variance estimates undermines claims of reliable superiority. DeepConsult reports win/tie/lose counts but no significance testing despite pairwise comparisons being amenable to binomial or bootstrap CIs.
- Reliance on LLM-as-judge across all benchmarks creates evaluation circularity risks; although the paper follows “official” judges per benchmark, additional human evaluation on a representative subset would strengthen claims, especially for “insight,” “readability,” and “support.”
- Effective citations metric may be confounded by output length. WebWeaver outputs very long reports; if “Eff. c.” counts raw validated citations, longer outputs can inflate the score. There is no normalization for output length or per-claim basis reported. The very large Eff. c. numbers (200+) relative to baselines raise the possibility that the metric rewards verbosity rather than calibrated evidence use.
- The ablation “hierarchical vs brute-force writing” lacks sufficient detail to be fully persuasive: context limits, chunking strategy for the brute-force baseline, and exact parity of planner outputs across conditions are not specified. Without these, it’s unclear if the baseline is disadvantaged by context overflow or suboptimal chunking rather than an inherent weakness of the paradigm.
- The outline optimization ablation uses samples with three rounds of optimization and then evaluates earlier rounds; however, the prompt template prescribes “at least three” updates but the reported average outline optimization steps are ~2.2. This mismatch between prescribed behavior and observed behavior should be clarified.
- No code, no executable release, and no links to run-time infrastructure are provided. Prompts are included, but the system depends on multiple LLMs, tools, and scraping steps. Critical parameters (stop criteria, deduplication policies, cache behavior, search engine settings, rate limits, tie-breaking rules, etc.) are not fully specified. This reduces reproducibility.
- The WebWeaver-3k dataset creation lacks overlap analysis against evaluation prompts and lacks licensing audits of scraped content. Without an overlap/contamination check, the risk of implicit test exposure cannot be discounted.
- The ethics section is a little generic at the moment and should also address website terms-of-service compliance, robots.txt adherence, or how PII in web pages is handled during evidence extraction and dataset curation. For an agent that crawls and stores web-scale evidence, these omissions are material.

### Questions
1. Code/data release: Will you release the full WebWeaver code (planner/writer, tools, memory) and the WebWeaver-3k dataset? If not, what components will be made public and when?
2. Fairness controls: Can you provide a controlled comparison where WebWeaver and a strong baseline agent both use the same backbone LLM and judge, to isolate framework gains? Any cross-judge robustness (e.g., alternate judges or human eval subsets)?
3. Cost/latency: What are average wall-clock time, token counts, and dollar costs per task on each benchmark, for both planning and writing? How does it compare to baselines?
4. Evidence extraction specifics: How are quote spans selected, how is evidence deduplicated/normalized, and what is the granularity for memory entries? How do you handle noisy, contradictory, or paywalled sources?
5. Failure and correction: How do the planner/writer detect and correct citation errors (e.g., broken links, drifted IDs, semantically mismatched evidence)? Any automatic validation?
6. WebWeaver-3k licensing and distribution: What is the licensing status of the dataset, given it is derived from web materials and proprietary teacher outputs? Can it be redistributed?
7. Safety and misuse: What safeguards are in place to prevent the agent from citing low-quality or harmful sources? Do you filter by source credibility or domain allowlists/blacklists?
8. Outline-to-retrieval mapping: How robust is the citation-ID mapping across multiple optimization rounds (IDs added/removed)? How do you maintain referential integrity?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces WebWeaver, a dual-agent framework for open-ended deep research (OEDR).
The system consists of a Planner that dynamically do evidence acquisition and outline optimization, and a Writer that performs memory-grounded, hierarchical synthesis by section. The human-centric design simulates real research workflows.
WebWeaver is claimed to achieve SOTA on DeepResearch Bench, DeepConsult, and DeepResearchGym against Gemini-2.5, OpenAI DeepResearch and Claude Research.

### Strengths
- Clear and well-motivated problem framing: tackling OEDR beyond static pipelines.
- Dual-agent design (Planner + Writer) inspired by real-wold human research process.
- Extensive experiments across 3 major benchmarks with detailed breakdowns with strong quantitative evidence.
- Thorough ablation and statistical analysis showing the strength of outline optimization and hierarchical writing.
- Clear figures, intuitive workflow, and solid writing throughout.

### Weaknesses
- Heavy rely on LLM-as-judge evaluations that lacks human verification to some extend
- I'm curious what's the computational cost and latency? 
- Some redundancy between Sections 3.2 and 3.3, the method flow could be slightly more concise.
- How scalable is WebWeaver for real-world web-scale research (e.g., 1k+ page retrieval)?
- How sensitive is performance to the backbone model (Qwen vs. Claude vs. GPT)?
- Minor formatting issue: Fig. 1 move to page 2 looks better

### Questions
see weakness.

### Soundness
4

### Presentation
4

### Contribution
3
