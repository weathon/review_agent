# DAMR: Efficient and Adaptive Context-Aware Knowledge Graph Question Answering with LLM-Guided MCTS

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4

## Abstract
Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Existing methods primarily follow either the retrieve-then-reason paradigm, which relies on Graph Neural Networks (GNNs) or heuristic rules to extract static candidate paths, or dynamic path generation strategies that employ large language models (LLMs) with prompting to jointly perform retrieval and reasoning. However, the former lacks adaptability due to static path extraction and the absence of contextual refinement, while the latter suffers from high computational costs and limited evaluation accuracy because of their dependence on fixed scoring functions and repeated LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates LLM-guided Monte Carlo Tree Search (MCTS) with adaptive path evaluation to enable efficient and context-aware KGQA. DAMR leverages MCTS as a backbone, where an LLM-based planner selects the top-k semantically relevant relations at each expansion step to effectively reduce the search space. To enhance evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, thereby capturing fine-grained semantic shifts during multi-hop reasoning. Furthermore, to mitigate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, enabling the scorer to continually adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
DAMR is an LLM-guided Monte-Carlo Tree Search framework for KGQA that decomposes reasoning into three coordinated parts: (i) an LLM planner used only for expansion, proposing the top-k relations at each node to aggressively prune the graph; (ii) a lightweight cross-attention path scorer that jointly encodes the question and the evolving relation sequence to estimate path plausibility; and (iii) an online pseudo-path refinement loop that converts high-value (partial) paths discovered during search into pairwise supervision, continually adapting the scorer to the search distribution. On WebQSP and CWQ, DAMR reports higher Hits@1/F1 than semantic-parsing, retrieval, and LLM+KG baselines while markedly reducing LLM calls and tokens. Comprehensive ablations (removing the scorer/refinement or replacing the scorer with a general LLM), sensitivity analyses (k and max hop L), and backbone swaps (Llama/Qwen/GPT-4.1) support the design. The approach emphasizes efficiency, path-level interpretability, and modularity (LLM for direction, small Transformer for judgment, MCTS for integration). Current evidence is Freebase-centric (WebQSP/CWQ); extending to Wikidata or domain KGs (e.g., biomedical) and clarifying the evaluation protocol (full test vs. sampled subsets, confidence intervals) are natural next steps.

### Strengths
1. Uses the LLM only for expansion and a small Transformer for evaluation, cutting LLM calls by >50% and tokens by ~75% without hurting accuracy, lowering cost/latency and allowing easy module swaps under resource limits.
2. Employs a cross-attention scorer conditioned on the question to model relation sequences hop-by-hop, capturing semantic buildup and constraints for steadier rankings and better generalization than a general LLM scorer.
3. Converts promising/contrasting partial paths into pairwise supervision during search, continually adapting the scorer to the current distribution and mitigating early-stage bias without sparse-reward RL instability.
4. Demonstrates consistent gains on WebQSP and CWQ with strong baselines, component ablations, sensitivity to 𝑘 and max hop 𝐿, and backbone swaps (Llama2-13B, Qwen3-14B, GPT-4.1/mini), supporting accuracy, efficiency, and robustness.
5. Outputs explicit KG paths as reasoning traces, making the question-to-answer chain transparent and easing auditing/error localization versus black-box LLM reasoning.

### Weaknesses
1. Reporting on 1,000 uniformly sampled test questions rather than full official splits inflates variance and hinders comparability; the absence of confidence intervals, significance tests, and an error taxonomy (e.g., compositional, comparative failures) weakens claims of robustness and external validity.
2. Dynamic pseudo-labeling lacks stability analysis: pair generation driven by search stats is not evaluated for convergence or early-noise sensitivity, and safeguards (confidence margins, temperature/top-p thresholds, value-gap filters) against confirmation bias/drift are unspecified.
3. Reproducibility and fairness are under-detailed: planner prompts, 𝑘-per-hop policy, decoding/tie-breaking, and failure handling are undisclosed; use of external services (GPT-4.1 planner, Qwen3-Embedding-8B) lacks API versions, costs, and train/freeze settings; it’s unclear whether baselines were re-run under comparable planner strength.
4. Generality is unproven beyond Freebase (WebQSP, CWQ); no evidence on other schemas or domains (e.g., Wikidata, biomedical) or on how schema size, relation lexicalization diversity, and alias noise affect performance.
5. Practical replication is hindered by notation/editorial issues (typos, cross-refs) and missing training hyperparameters (batch sizes, negative sampling ratios, per-dataset hop limits), obscuring the path to matching the reported accuracy/efficiency trade-offs.

### Questions
1. Why did you evaluate on 1,000 sampled test questions instead of the full official splits, and can you report full-test metrics and/or confidence intervals across multiple samples?
2. What are the exact prompts for relation selection, including decoding parameters, the per-hop value of (k), and the procedure for mapping LLM-suggested relations to graph edges when multiple lexicalizations exist?
3. What is the fallback when the LLM proposes relations absent from the local subgraph or yields fewer than (k) candidates (e.g., degree-based backup or other heuristics)?
4. How do you prevent the online fine-tuning loop from overfitting to early search biases, and did you test confidence margins or require minimum value gaps for training pairs? Do you have curves of scorer AUC versus refinement rounds?
5. What are the end-to-end wall-clock latencies per query and GPU/CPU utilizations for DAMR compared with baselines, beyond token and call counts?
6. Have you evaluated on Wikidata or a domain KG such as UMLS, and what challenges do you anticipate (schema size, relation lexicalization diversity, entity alias noise)?
7. How sensitive is performance to the scorer’s design choices: (a) cross-attention versus simple concatenation, (b) the presence and type of positional encodings, and (c) pairwise ranking loss versus regression?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a new framework for knowledge graph question answering that addresses adaptability, accuracy, and computational cost issues raised by previous approaches. This is implemented by combining an LLM-based planner, a lightweight Transformer-based scorer, and a dynamic pseudo-path refinement mechanism on top of a Monte Carlo Tree Search (MCTS) backbone. The paper is well motivated, clearly presented, and shows promising results according to experiments to achieve the design purposes.

### Strengths
1. The overall approach is novel to me, though not entirely new due to extensive existing efforts in this field of research. The authors describe a clear step-wise procedure with sufficient details to show how the approach works. The approach presents as a sound solution to address the identified limitations of the existing approaches.

2. Effectively modularizes reasoning by limiting the LLM's role to an initial, high-leverage search guidance step, significantly reducing computational overhead.

3. The experiments were comprehensive with convincing results, covering performance comparisons, efficiency analysis, ablations, sensitivity studies, and the impact of LLMs. The case study also helps with understanding of the final outcome.

### Weaknesses
1. The comparative discussion against related work could be strengthened to reveal more details about the rationale of designs in the proposed framework. 

2. The selection of baseline methods in Table 2 should be discussed. More specifically, why are the three baseline methods selected in particular for the computational efficiency comparison?

3. A clear mapping between the technical components and the advantages/edge achieved by the proposed framework might be better explained, demanding studying the impact of key components on the overall accuracy and computational efficiency of the framework.

### Questions
1. Why are the three baseline methods selected in particular for the computational efficiency comparison?
2. What are the (positive or negative) impacts of each key component on the overall accuracy and computation efficiency of the proposed framework? Current paper only partially answers this question.

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
4

### Summary
This paper introduces DAMR, an MCTS-based framework for Knowledge Graph Question Answering that utilizes LLM-guided expansion, a lightweight transformer-based, cross-attention path evaluator, and dynamic pseudo-path refinement for continual scorer adaptation. The paper provides substantive ablation, efficiency, and sensitivity analyses, as well as qualitative examples highlighting the method’s strengths.

### Strengths
1. Dynamic Pseudo-Path Refinement: The method innovates by using high-confidence partial paths from MCTS rollouts to generate pseudo labels for continual fine-tuning , thus improving generalizability and adapting to the non-stationary search space.

2. Rigorous Empirical Validation: Extensive benchmarks across both standard datasets, with direct comparisons to at least 20 strong baselines  are provided in Table 1, consistently showing DAMR outperforming all competitors.

### Weaknesses
1. It is unclear under what distributional shifts the scorer avoids reinforcing suboptimal trajectories. Since the path scorer is continually adapted with self-generated pseudo-paths, there is risk of feedback loops or bias accumulation, especially if the LLM suggestions are systematically biased early in training.

2. Scalability and practicality on large KGs not addressed. All experiments are conducted on localized subgraphs derived from WebQSP and CWQ. The scalability of DAMR for web-scale or multi-million entity KGs is not empirically or mathematically analyzed. How does the MCTS backbone behave when the entity degree is very high, or when context selection via LLMs requires thousands of candidates? Is the method robust under significant KG incompleteness or noise? Discussion is notably absent.

3. Lack of feedback for LLM planner improvement. While DAMR incorporates dynamic refinement for the path evaluator, the LLM-based planner, which determines the search direction, receives no feedback signal from the search process. As a result, the planner cannot benefit from experience or adapt its relation selection strategy over time. 

4. Lack of variance and statistical significance reporting. The experimental results are reported only as single-point metrics, without variance or statistical significance analysis. Including standard deviations across multiple random seeds, confidence intervals, or hypothesis testing (e.g., paired t-test, bootstrap) would strengthen the reliability of the claimed performance improvements.

### Questions
see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
