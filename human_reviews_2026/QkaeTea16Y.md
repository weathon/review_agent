# DeepTRACE: Auditing Deep Research AI Systems for Tracking Reliability Across Citations and Evidence

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Generative search engines and deep research LLM agents promise trustworthy, source-grounded synthesis, yet users regularly encounter overconfidence, weak sourcing, and confusing citation practices. We introduce _DeepTRACE_, a novel sociotechnically grounded audit framework that turns prior community-identified failure cases into eight measurable dimensions spanning answer text, sources, and citations. DeepTRACE uses statement-level analysis (decomposition, confidence scoring) and builds citation and factual-support matrices to audit how systems reason with and attribute evidence end-to-end. Using automated extraction pipelines for popular public models (e.g., GPT-4.5/5, You.com, Perplexity, Copilot/Bing, Gemini) and an LLM-judge with validated agreement to human raters, we evaluate both web-search engines and deep-research configurations. Our findings show that generative search engines and deep research agents frequently produce one-sided, highly confident responses on debate queries and include large fractions of statements unsupported by their own listed sources. Deep-research configurations reduce overconfidence and can attain high citation thoroughness, but they remain highly one-sided on debate queries and still exhibit large fractions of unsupported statements, with citation accuracy ranging from 40–80\% across systems. Unlike prior factuality and citation metrics that focus on claim correctness or academic summarization, DeepTRACE audits end-to-end GSE/DR behavior, including citation necessity, unsupported-statement rates, and URL-level citation structure.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces DeepTRACE, an end-to-end audit framework for Generative Search Engines (GSEs) and Deep-Research agents (DRs), addressing their persistent issues of overconfidence, weak source grounding, and confusing citations. Unlike prior evaluations focusing on isolated components (e.g., retrieval or summarization), DeepTRACE quantifies system behavior across the full pipeline—evidence use, citation accuracy, and uncertainty expression. It analyzes answers at the statement level, constructing Citation and Factual-Support Matrices to assess how statements are supported by cited sources. Using LLM-based judgments and web content retrieval, the framework measures eight audit dimensions. Results show that public GSEs often produce one-sided, overconfident answers with poor factual support, while DRs improve citation completeness but still suffer from bias and unsupported claims. Simply adding more sources does not ensure better grounding. The authors advocate a sociotechnical auditing perspective to enhance reliability and transparency in AI-driven search and reasoning systems.

### Strengths
1. One of the first works to systematically analyze trustworthiness issues in GSE and DR systems. The contrast they draw is super insightful — GSEs tend to be concise and relevant but often sacrifice balance and factual grounding, while DRs are more thorough and citation-heavy yet verbose, less relevant, and still packed with unsupported claims.

2. It's good to see how the authors bring in confidence calibration and debate-style tasks to expose bias. It’s striking that GSEs sound highly confident yet one-sided on controversial topics, whereas DR setups tone down confidence and cite more sources but still fail to stay balanced or fully grounded. The finding that citation accuracy only ranges between 40–80% across systems is quite revealing.

3. The paper also contributes a new, well-tailored set of metrics that fit the unique workflow of GSEs and DRs, going beyond standard RAG evaluation.

4. I found the sixteen design recommendations particularly useful — they translate the audit insights into actionable guidance for building more transparent and reliable generative search systems.

### Weaknesses
1. The paper’s discussion of citation and factual grounding issues is closely related to prior work on hallucination evaluation. Some of the proposed metrics (e.g., Necessity, Support) and findings — such as “many statements are not supported by cited sources” and “listing more URLs does not imply better grounding but can create false endorsement” — overlap conceptually with earlier studies like ALCE (1) and Trust-Score (2), yet these works are not cited or discussed. Acknowledging this connection would strengthen the paper’s positioning within the literature.

2. It would be interesting to further explore how overconfidence actually interacts with citation and factual support quality, rather than only observing that GSEs are overconfident or one-sided. For example, does higher confidence correlate with weaker evidence grounding?

3. The treatment of unsupported statements might conflate “ungrounded” with “incorrect.” If a statement lacks support from retrieved sources, it doesn’t necessarily mean it is false — the model might be factually correct but based on prior knowledge or sources not captured by the retrieval. Framing all such cases as “unsupported” could overstate the system’s factual risk.

4. Around 15% of URLs were excluded from factual-support evaluation due to paywalls or broken links. It’s possible that many of these are authoritative outlets (e.g., news or journals) that the systems correctly cited but became inaccessible. This could systematically inflate the “unsupported” rate and distort metrics like Source Necessity. A sensitivity analysis could clarify this effect.

5. The results presentation lacks clarity on statistical significance and uncertainty. Including significance tests, confidence intervals, or corrections for multiple comparisons would make the reported differences between systems more robust and convincing.

Ref:

(1): https://arxiv.org/abs/2305.14627

(2): https://arxiv.org/abs/2409.11242

### Questions
Stated in the weaknesses.

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
5

### Summary
This paper introduces DeepTRACE, a sociotechnically-grounded audit framework for evaluating generative search engines (GSEs) and deep research (DR) agents. The framework translates user-identified failure cases from prior qualitative research into eight quantifiable metrics spanning answer text quality, source quality, and citation accuracy. The authors evaluate nine popular systems (GPT-4.5/5, You.com, Perplexity, Copilot/Bing, Gemini) across 303 queries (168 debate questions, 135 expertise questions) using automated pipelines and LLM judges validated against human annotations.
Key findings reveal that GSEs frequently produce one-sided, overconfident responses to debate queries, with 40-80% of statements unsupported by cited sources and citation accuracy ranging 40-68%. Deep research configurations reduce overconfidence and improve citation thoroughness (up to 87.5% for GPT-5 DR) but remain highly one-sided on debate queries (54-95%) and still exhibit substantial unsupported statement rates (12.5-97.5%). The work demonstrates that simply increasing source quantity does not guarantee better grounding, and highlights a critical gap between the promise and reality of source-grounded AI research systems.

### Strengths
(a) Work is strongly inspired by Narayanan Venkit et al., 2025. By translating qualitative findings into quantifiable metrics, the work bridges an important gap between user experience research and systematic benchmarking, addressing the field's over-reliance on researcher-defined metrics.

(b) The eight-metric framework captures end-to-end system behavior across answer quality, source utilization, and citation practices. Metrics like Source Necessity (using graph-theoretic minimum vertex cover) and the distinction between Citation Accuracy vs. Thoroughness provide nuanced insights beyond simple correctness measures. Worth checking this work on citations: https://arxiv.org/pdf/2405.02228

(c) The findings on one-sidedness in debate queries and high unsupported statement rates have direct implications for misinformation and echo chamber formation.

(d) The authors appropriately validate LLM-based judgments against human annotators (r=0.72 for confidence, r=0.62 for factual support), acknowledge limitations, and provide detailed prompts in appendices.

### Weaknesses
(a) Worrisome Point: While the integrated framework is valuable, most individual metrics are straightforward applications of existing techniques (statement decomposition, confidence scoring, factual verification). The paper would benefit from deeper technical innovation—for example, more sophisticated attribution tracking that accounts for implicit support or reasoning chains, or methods to reduce reliance on LLM judges

(b) Open to accept responses from authors: The 303-query dataset, while grounded in user needs, is relatively small and may not capture the full diversity of real-world search queries. The heavy emphasis on debate questions (168/303) creates an evaluation skew toward contentious topics. The framework's applicability to other query types (navigational, transactional, multimodal) remains unexplored.

(c) The acceptable/borderline/problematic thresholds (Table 2, Appendix A) appear arbitrary despite claims of grounding in qualitative findings. For example, why is 40%+ one-sidedness "problematic" rather than 30% or 50%? The paper would benefit from sensitivity analyses showing how conclusions change across threshold variations, or formal methods (e.g., user studies, expert consensus) for threshold derivation.

(d) The evaluation captures a single temporal snapshot (August 27, 2025), but commercial systems update frequently. The paper acknowledges this but doesn't address how the framework handles system evolution or provide guidance for longitudinal monitoring. The reliance on browser automation for proprietary systems creates reproducibility challenges—other researchers cannot easily replicate these evaluations without rebuilding scraping infrastructure for each platform.

### Questions
(a) The current binary factual support judgment may be too coarse. Many statements in research outputs are partially supported, require inferential steps, or synthesize multiple sources. Have you considered a graded support scale (e.g., fully supported, partially supported, unsupported, contradicted)? How would this affect the Unsupported Statements and Citation Accuracy metrics? Recent work on claim-level attribution (e.g., REASONS ("pass"/I don't know, ALCE benchmark, Gao et al. 2023) uses more nuanced support labels—how does your approach compare?


(b) GPT-5(DR) achieves exceptional performance (87.5% relevant statements, 12.5% unsupported, 79.1% citation accuracy) while Perplexity(DR) (which model within Perplexity did the authors pick?) fails catastrophically (22.5% relevant, 97.5% unsupported, 58.0% citation accuracy). Can you provide a qualitative analysis of what architectural or design choices drive these differences? Are there insights from error analysis that could guide system improvement? Understanding the mechanisms behind performance gaps would strengthen the paper's practical impact.

(c) Modern deep research systems increasingly incorporate images, tables, charts, and interactive elements. How would DeepTRACE evaluate citation accuracy when claims are supported by visual evidence? Similarly, for rapidly evolving topics (breaking news, financial markets), how should temporal aspects factor into evaluation? What adaptations would be needed for conversational/multi-turn research interactions versus single-query responses?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents DeepTRACE, a framework for auditing deep research systems. The authors focused on evaluation of common failure cases, specifically,  one-sided reasoning and weak citations. They evaluate DR systems with eight quantitative metrics.
They run experiments on  303 queries from debate and expert questions. They conducted human evaluation and report their correlation with their llm as a judge eval.

### Strengths
- Evaluation of DR systems is very timely and important specially studying them from reliability and transparency perspectives.

- The inclusion of one-sidedness and overconfidence metrics is a significant contribution and that is something that distinguished this paper from other deep researchy benchmarks.

### Weaknesses
- The human-LLM agreement results (e.g., correlations for factual support and confidence scoring) are mentioned only in text and scattered throughout the paper. They should be summarized in a single comparison table to make it easier to see which metrics show higher or lower reliability.

- The annotator information is unclear. It it not clear whether annotators had expertise relevant to the “debate” and “expertise” questions, which could heavily affect judgments.

- I wish the authors provided  breakdown of questions by topic or question difficulty in the appendix.

- The evaluation is highly citation-centric, missing aspects such as answer completeness, coherence, and synthesis quality, while these have been studied before and the aspects that authors assessed are more novel and more neglected, we cannot ignore core elements of what users expect from “deep research”.

- I did not understand how all the accurate citations are being captured in 3.1.4 .

- The Eval Scorecard in Table 1 appears inconsistent with the detailed metrics above it (e.g., GPT-5 rows don’t align between triangles/circles).I think  the scorecard visualization itself fails to distinguish meaningfully between models and did not find it informative enough.

The paper could benefit from having opensoruced baselines such as DeepResearcher or OpenScholar, or open sourced LLM + search tools, which would make comparisons more complete.

### Questions
- Who were the annotators, and how was their expertise matched to the question domains especially for expert questions?

- Section 3.1.4 VIII --> how do you get all possible accurate citations? 

- What will happen if there is an answer which is relevant and short and has only one citation. Then I belive many of the metrics such as relevant statement and citation precision might be 1/1... I wonder how do you capture how complete the answer is?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DeepTRACE, an automatic audit framework for LLM-powered generative search engines/deep research agents. This framework provides quantitative assessment of biases, confidence, grounded factuality and relevance of the generation (by the above systems). The design of evaluation metrics are inspired by the user-centric insights of prior work and operationalize these insights into quantitative measurement that can be automatically obtained by LLM-as-a-judge. Leveraging these automatic framework, the authors evaluate a few popular generative search engines and deep research agents. The results and analysis reveals that 1) the evaluated generative search engines tend to output one-sided, overconfident and unsupported statements; 2) the evaluated deep research agents exhibit lower rate of overconfidence but still generate one-sided and unsupported statements.

### Strengths
1. The proposed framework operationalized the auditing of generative search engines/deep research systems with automatic and quantitative measurement of the quality of generation from multiple aspects. This framework offers tractable and measurable evaluations of these systems beyond user-centric insights.
 
2. The evaluation with the proposed framework reveals detailed and multifaceted performance measurements of existing systems, and the analysis provides potentially valuable insights into the pros and cons of these systems

### Weaknesses
1. This paper might have missed the discussion with some existing evaluation metrics that partially overlap with their proposed evaluation dimensions. For example, factuality evaluation [1-2] and citation quality [3]
2. While the results reveal several pitfalls of existing systems, I am slightly concerned that if it is the overly-simple setup of these generative search engines/deep research systems that "exaggerates" these pitfalls. In particular, judging from the paper, I would assume that the collected queries are direct (and only) input to the systems. However, I would expect some simple tweaks of the setup might improve the systems generations across the evaluation dimensions. For example, adding system prompts that ask the model to be less polarized to retrieve and aggregate information from multiple perspectives, and output its statements such that each of them should be supported by their citations.

References
[1] Min et al. FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. EMNLP 2023
[2] Jiang et al. Core: Robust Factual Precision with Informative Sub-Claim Identification. Findings of ACL 2025
[3] Wang et al. AutoSurvey: Large Language Models Can Automatically Write Surveys. NeurIPS 2024.

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
