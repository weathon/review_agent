# CycleIE: Robust Document Information Extraction through Iterative Verification and Refinement

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
In document AI, reliable analytics require converting long, noisy (often multi-document) corpora into heterogeneous structured data—e.g. tables for numerical fields, graphs for entity–relation structures, trees for hierarchies, and faithful text chunks. Yet one-pass LLM extraction often yields incomplete or inconsistent structures because it lacks explicit verification and opportunities to revise earlier choices. We present CycleIE, an iterative information extraction (IE) framework that closes the loop between reasoning and acting by coupling ReAct with Monte Carlo Tree Search (MCTS). CycleIE employs a multi-agent workflow orchestrated through ReAct and optimized via MCTS to iteratively retrieve, structure, extract, and refine extracted content under verification guidance. This design treats extraction as a search process with feedback, enabling systematic correction of omissions and inconsistencies that defeat one-pass methods, and remains orthogonal to retrieval-augmented generation (RAG) by operating directly over user-provided documents. Experiments on challenging the document-based QA benchmark demonstrate that CycleIE delivers >10% relative improvements in extraction quality over strong one-pass baselines, with the largest gains in lengthy or multi-document contexts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CycleIE, an iterative information extraction (IE) framework designed to enhance robustness and accuracy when extracting structured data from long, complex documents. Unlike conventional one-pass approaches that directly prompt LLMs to output structured data, CycleIE iteratively verifies and refines intermediate results using a multi-agent architecture (Retriever, Extractor, Verifier, Refiner, Reasoner, Structurer). CycleIE combines the ReAct paradigm for reasoning-action interleaving with Monte Carlo Tree Search (MCTS) for strategic action planning and anomaly intervention. Through iterative verification cycles, the model identifies incomplete or inconsistent information and refines extraction accordingly.

### Strengths
- Clear motivation and strong execution. The paper convincingly argues for iterative verification in document IE, a gap in current RAG-based or one-pass extraction systems. The integration of ReAct and MCTS is technically well justified and implemented in a coherent multi-agent pipeline.
- Comprehensive experimental evaluation. Evaluation spans multiple document lengths and task types (comparison, clustering, reasoning, etc.) across the Loong dataset. The performance improvements, especially under long-context conditions, are large and consistent.
- Thorough ablation and analysis. The authors conduct clear ablations (w/o verify, w/o extract) and show that verification contributes more to performance than extraction alone. Figure 3 (p. 8) and Table 3 illustrate this quantitatively.
- Strong reproducibility and implementation details. The paper provides extensive appendices covering prompts, agent workflows, and backend design, making reproduction feasible. The inclusion of a runtime breakdown (Table 4, p. 18) is appreciated.
- Solid conceptual clarity. The workflow diagram (Figure 2, p. 4) effectively visualizes agent collaboration, making the iterative refinement concept accessible and well structured.

### Weaknesses
- Limited conceptual novelty. The framework largely combines known ideas — ReAct reasoning loops and MCTS-based planning — into the IE setting. The novelty lies in applying these together rather than proposing a fundamentally new reasoning paradigm. Compared to recent structured extraction frameworks (e.g., StructRAG 2025, GraphRAG 2024, DataMosaic 2025), CycleIE’s methodological contribution feels incremental.
- Experimental scope limited to one dataset. All experiments use the Loong benchmark. While this dataset is large and complex, additional validation on other domains (e.g., DocVQA, ContractNLI, or scientific papers) would strengthen claims of robustness.
- Lack of human evaluation or generalization proof. The evaluation is entirely automatic (GPT-4 judge). Some human annotation or downstream use-case study (e.g., financial or legal analytics) would make the contribution more compelling.
- Risk of over-engineering. The six-agent architecture, while conceptually elegant, may be overcomplicated for marginal improvements on shorter contexts. The benefit–complexity balance is not deeply analyzed.

### Questions
- Generality of the approach. Can CycleIE generalize beyond the Loong benchmark to less structured document sets (e.g., PDFs with layout noise or cross-lingual corpora)?
- Choice of MCTS. Why is Monte Carlo Tree Search preferred over simpler policy-selection methods (e.g., reinforcement learning with learned value functions)? How sensitive are results to the number of simulations?
- Failure cases. The case study (Figure 4) shows clear success. Could the authors also share examples where iterative refinement failed or produced contradictions?
- Ablation on agent granularity. Would merging Verifier + Refiner into one component materially change results or efficiency?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel way to incorporate text extractions and verification when working with LLMs to increase task completion success when working with long documents. The model presents significant improvements on several benchmarks, notably the LLM on its own, but the accuracy remains too low to use in accuracy-critical settings.

### Strengths
- The paper considers an important problem.
- The approach is sound, albeit somewhat convoluted. 
- The results show significant improvement compared to LLM-only baseline. 
- Extensive experiments.

### Weaknesses
I see two key weaknesses for this work:

- I think the comparison with baselines in the paper is not at a good enough level -- yes, there are many baselines, but they have many fewer steps compared to the proposed method. This means that I don't know if the proposed method itself is strong, or just adding a deconstruction and verification step. For example, how superior are the `agents' compared to a set of prompts aimed at carrying out the same function? Another significant baseline should be the new LangExtract, The general approch is sound but the experiments do not prove the specific approch is suprior to exsiting alternatives. 

Another key weakness I see in this paper is that although the improvement is significant -- the results are still not accurate enough to use in critical settings. This is very briefly mentioned in the ethics statments, but not disucssed anywhere. Who do you see adopting your method? For what use cases? Is it possiable to add a human in the loop to improve the accuracy to >95%? (e.g., https://dl.acm.org/doi/full/10.1145/3652591)

### Questions
- Who do you see adopting your method? For what use cases? 

- Are you able to compare your method to more competitive baselines?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces CycleIE, an iterative framework for document information extraction (IE) that combines Reactive Reasoning (ReAct) with Monte Carlo Tree Search (MCTS). Unlike one-pass IE methods that directly extract structured data from documents, CycleIE performs iterative verification and refinement, leveraging multi-agent collaboration (Retriever, Extractor, Verifier, Refiner, Reasoner) to improve data completeness and consistency. Experiments on the Loong benchmark demonstrate notable performance gains (10–37% improvement in LLM and EM scores) over strong baselines such as StructRAG, GraphRAG, and RQ-RAG, especially for long documents up to 250k tokens.

### Strengths
Clear motivation and solid formulation. The paper convincingly argues that one-pass IE fails to perform self-verification and introduces a principled iterative framework with ReAct + MCTS integration.

Technically novel combination. The orchestration of six agents with verifier-driven feedback and MCTS-based action optimization is original and well-justified.

Comprehensive experiments. Strong empirical results on Loong across multiple tasks (spotlight, comparison, clustering, reasoning) show CycleIE’s robustness and scalability to long documents.

Well-analyzed ablations. The ablation on “w/o verify” and “w/o extract” modules clearly isolates the contribution of verification and modular synergy.

Readable and systematic presentation. The figures and tables are clear, and the technical flow (agents → ReAct → MCTS → experiments) is logically organized.

### Weaknesses
Lack of theoretical or statistical significance analysis. The paper reports gains but does not include variance or significance tests across runs, which weakens the empirical rigor.

Limited baselines in iterative IE space. The comparison set focuses on one-pass RAG methods; missing direct comparisons to other iterative refinement or multi-agent reasoning frameworks (e.g., Self-Refine, Graph-of-Thoughts, AutoGen variants).

Computational cost and scalability. No quantitative discussion of iteration overhead, runtime, or resource efficiency; MCTS may scale poorly for very long sequences.

Dependence on large base model. All experiments use Qwen2-72B-Instruct; the method’s generalizability to smaller or open-weight models remains unverified.

Clarity on reward design. The reward terms in Eq. (3) are conceptually sound but not empirically grounded—hyperparameters (e.g., α, μ, η) are not explained or validated.

### Questions
How sensitive is CycleIE’s performance to the choice of α and the reward function components in MCTS?

Could the authors report runtime overhead or number of iterations per query to evaluate cost vs. accuracy trade-offs?

How would CycleIE perform if smaller models (e.g., Qwen1.5-7B) were used for each agent? Would iterative refinement still yield consistent improvements?

Since the paper claims generality beyond financial QA, has CycleIE been tested on non-numerical document types (e.g., legal or biomedical texts)?

How are verification thresholds (e.g., completeness ≥ 3) tuned or decided? Are they fixed across datasets?

### Soundness
3

### Presentation
3

### Contribution
3
