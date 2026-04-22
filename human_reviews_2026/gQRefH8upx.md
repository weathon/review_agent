# KnowGuard: Knowledge-Driven Abstention for Multi-Round Clinical Reasoning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
In clinical practice, physicians refrain from making decisions when patient information is insufficient. This behavior, known as abstention, is a critical safety mechanism preventing potentially harmful misdiagnoses. Recent investigations have reported the application of large language models (LLMs) in medical scenarios. However, existing LLMs struggle with the abstentions, frequently providing overconfident responses despite incomplete information. This limitation stems from conventional abstention methods relying solely on model self-assessments, which lack systematic strategies to identify knowledge boundaries with external medical evidences. To address this, we propose \textbf{KnowGuard}, a novel \textit{investigate-before-abstain} paradigm that integrates systematic knowledge graph exploration for clinical decision-making. Our approach consists of two key stages operating on a shared contextualized evidence pool: 1) an evidence discovery stage that systematically explores the medical knowledge space through graph expansion and direct retrieval, and 2) an evidence evaluation stage that ranks evidence using multiple factors to adapt exploration based on patient context and conversation history. This two-stage approach enables systematic knowledge graph exploration, allowing models to trace structured reasoning paths and recognize insufficient medical evidence. We evaluate our abstention approach using open-ended multi-round clinical benchmarks that mimic realistic diagnostic scenarios, assessing abstention quality through accuracy-efficiency trade-offs beyond existing closed-form evaluations. Experimental evidence clearly demonstrates that KnowGuard outperforms state-of-the-art abstention approaches, improving diagnostic accuracy by 3.93\% through effective diagnostic interactions averaging 5.74 conversation turns.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces KnowGuard, a framework for LLMs that leverages knowledge graphs to drive abstention decisions in multi-round clinical reasoning. Instead of solely relying on internal knowledge, KnowGuard explores structured medical knowledge to identify evidence gaps before making or abstaining from a diagnosis. The approach involves a two-stage process: evidence discovery (systematic retrieval and expansion of relevant medical facts) and evidence evaluation (multi-factor scoring and prioritization, including graph coherence, embedding similarity, LLM selection, temporal decay, and patient population reasoning). Experiments on several open-ended multi-round clinical benchmarks show that KnowGuard achieves improved diagnostic accuracy and reduced unnecessary patient-provider interaction compared to existing abstention and reasoning strategies.

### Strengths
1. The paper focuses on the valuable and practical issue of abstention in multi-round dialogue for LLM-based medical diagnosis, which is essential for efficiency and reliability in clinical applications.
2. The proposed approach achieves improvements in both diagnostic accuracy and the number of interaction rounds compared to  baselines. Comprehensive ablation studies further validate the independent contributions of each module.

### Weaknesses
1. Limited Novelty in Retrieval and Evaluation Mechanisms: The specific mechanisms for evidence retrieval and evaluation do not appear fundamentally innovative beyond existing literature on RAG and knowledge-enhanced medical reasoning. The paper would benefit from a clearer positioning against recent state-of-the-art works utilizing similar approaches.
2. Incomplete Evaluation of Abstention Performance: Although timely and appropriate abstention is claimed as the core motivation, the paper lacks dedicated abstention-quality metrics. The evaluation is limited to diagnostic accuracy and conversation turns; it omits quantitative analysis of abstention rates, false or missed abstention cases, the timing of abstention decisions, and abstention precision/recall. 
3. Limited Baseline Coverage: The experimental section does not provide thorough comparisons with more recent or diverse RAG architectures, especially those specifically designed for clinical reliability or abstention tasks. This limits the strength of the claims regarding advancement over existing retrieval-enhanced methods.
4. Absence of Expert Evaluation and Real-World Validation: The evaluation is fully automated and lacks assessment by clinical experts or real-world usability studies. This limits evidence for practical utility and adoption outside of benchmark datasets.
5. Knowledge Coverage and Maintenance Concerns: There is insufficient discussion of the medical knowledge graph’s domain completeness, update frequency, and capacity for handling rare, newly emerging, or rapidly changing conditions. Without assurances of knowledge freshness and coverage, the real-world effectiveness and reliability may be restricted.
6. Scalability and Resource Overhead Not Addressed: The method’s multi-stage retrieval and repeated evidence evaluation may introduce considerable computational and resource costs, but there are no experiments or analysis regarding runtime efficiency, memory footprint, or suitability for deployment at scale.
7. Lack of robustness to noisy or misleading evidence: The paper does not analyze how KnowGuard handles noisy or misleading evidence during retrieval. It is unclear whether the system becomes overconfident or makes errors when faced with irrelevant or conflicting information. Experiments testing robustness to noise and corresponding mitigation strategies are missing.

### Questions
Please see the weaknesses section for detailed discussion.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
KnowGuard tackles a common clinical LLM flaw: giving confident answers when evidence is thin. It builds a medical knowledge graph and, at each turn, searches and scores external evidence (embedding match, LLM relevance, graph coherence, time decay, and patient-population signals) before either asking a pointed follow-up or answering with support. The authors test it on a new multi-round benchmark drawn from MEDQA, CRAFT-MD, and AFRIMEDQA against confidence-based abstention, self-consistency, and long-context retrieval, plus ablations.

### Strengths
(1) The proposed multi-round medical QA benchmark is well-motivated and thoughtfully curated, with a clear connection to real clinical reasoning processes. 

(2) The paper is well-presented, offering a clear and structured introduction to each module of the proposed framework.

### Weaknesses
(1) The evaluation is narrow. Experiments report accuracy but omit calibration/uncertainty metrics (e.g., ECE, Brier, NLL, risk–coverage) and related baselines (temperature scaling, ensembles, conformal methods). Given the risk-aware motivation, it’s unclear why these are absent. 

(2) The knowledge-graph integration is described only at a high level. Key implementation details, e.g., entity/relation extraction, the graph-search/expansion algorithm, how graph signals are fused with LLM scores, and crucial hyperparameters. These details are scattered in the appendix or left implicit, which limits reproducibility and interpretability.

### Questions
See weakness

### Soundness
2

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
The paper proposes an Investigate-Before-Abstain (IBA) framework for multi-round clinical question answering with large language models. Instead of forcing the model to answer or abstain directly, the method first expands a medical knowledge graph through two modes—graph-based neighborhood traversal and direct retrieval—to assemble an evidence pool before making the decision to answer, abstain, or query further. A new benchmark combining MEDQA, CRAFT-MD, and AFRIMEDQA is introduced to evaluate open-ended multi-round dialogue. Experiments show that IBA improves factual accuracy and abstention calibration over baseline systems such as simple retrieval-augmented generation or confidence-based abstention

### Strengths
•	Tackles a meaningful and under-explored problem: when and how clinical agents should abstain.

•	Technically coherent pipeline that bridges knowledge-graph reasoning with retrieval-augmented generation.

•	New benchmark resource and detailed evaluation across multiple datasets.

•	Solid empirical results demonstrating improvements in both correctness and cautiousness metrics.

### Weaknesses
• Somewhat heuristic combination of scoring factors; lacks ablation or theoretical grounding for weights.

•	Benchmark construction may inherit biases from source datasets; unclear generalizability to unseen guideline domains.

•	Limited discussion of system latency or computational cost compared with long-context baselines.

•	No human expert evaluation to verify that “abstain” decisions align with clinical expectations.

### Questions
1.	How sensitive are results to the relative weights among the five scoring components?

2.	Could the approach be extended to handle contradictory or time-evolving medical evidence (e.g., updated guidelines)?

3.	Does the model explicitly track patient-specific vs. population-level context, or is this purely implicit through graph features?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the critical problem of overconfidence in Large Language Models (LLMs) used for clinical reasoning. Current LLMs often provide premature or incorrect diagnoses when patient information is incomplete, failing to "abstain" as a human physician would. In the paper, the authors propose KnowGuard, a novel "investigate-before-abstain" paradigm. Instead of asking "how confident am I?", KnowGuard systematically investigates "what specific evidence am I missing?" by exploring an external, multi-modal medical KG. The authors converted the traditional medical datasets into interactive-multi-round formats for the evaluation of the proposed method. Through the experiments, the results show that the proposed method outperform the baseline methods by an obviously large margin.

### Strengths
1. The motivation and the writing of the paper is clear and easy to follow, and the framework seems very reasonable and intuitive, which could be convenient for downstream applications. The paper addresses the critical and unsolved problem of LLM overconfidence and its failure to abstain, which is a major barrier to deploying AI in high-stakes fields like medicine.
2. The presentation of the framework and case study is very clear and informative for readers to understand the use scenarios.
3. From the results of the experiment section, the proposed method is very effective in terms of accuracy in the constructed datasets.

### Weaknesses
1. The details of the how the constructed datasets were incorporated with other baseline methods are not described in the experiment section or the appendix. I think it would be better to describe in more details of how you ensure that the baseline methods get the same extent of information and knowledge during evaluation with KnowGuard.
2. I have some questions on the necessity of evaluating conversation turn count for the proposed method. To my understanding, it seems that the proposed method can benefit from gathering more information with more turn counts. Given Figure 3 (Left) shows accuracy improving with conversation length, should the method's strength be framed as "more effective use of turns" rather than "reduction of turns"?
3. The method relies heavily on the external KG. How does the system behave when the correct diagnostic evidence is not present in the knowledge graph? Does it learn to abstain gracefully due to a lack of a coherent evidence path, or could the lack of a positive "hit" lead it to a premature (and incorrect) answer?

### Questions
See the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
