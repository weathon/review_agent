# Plan-Answer-Refine-on-Graph: Structured Planning and Self-Refinement for Large Language Model Reasoning on Knowledge Graphs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Incorporating knowledge graphs (KGs) into large language model (LLM) reasoning has shown promise in alleviating hallucinations and factual errors. Although existing paradigms of KG-augmented LLMs have achieved encouraging results, they still exhibit notable limitations when handling multi-hop reasoning and complex logical queries: (1) search space truncation bias: current methods generate linear entity-relation reasoning paths, which can prune correct candidates prematurely during iterative exploration; and (2) entity error amplification: existing methods typically follow the retrieve-and-answer paradigm which causes LLMs to over-rely on retrieved evidence, exacerbating the impact of incorrect entities during reasoning. To alleviate the existing challenges, we propose Plan-Answer-Refine-on-Graph (PARoG), a novel framework for LLM reasoning on knowledge graphs. First, PARoG leverages SPARQL queries from KG data as references, decomposing them into structured step-by-step plans. We further train LLMs to construct such structured plans, which improves the logical consistency of reasoning, ensures uniform step granularity, and facilitates effective execution on the graph. Second, during reasoning over KGs, PARoG adopts a plan-answer-refine paradigm: the model first attempts to answer each sub-query independently, and then refines its prediction by integrating evidence retrieved from the KG. This process mitigates knowledge conflicts between LLM and KG, substantially reducing hallucinations. Experimental results on multiple KG reasoning benchmarks demonstrate that PARoG significantly outperforms state-of-the-art approaches, achieving especially superior accuracy on multi-hop and logically complex queries.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces PARoG (Plan-Answer-Refine-on-Graph), a novel framework designed to enhance large language model (LLM) reasoning over knowledge graphs (KGs). The method addresses two main issues in prior KG-augmented LLM approaches: (1) Search space truncation bias, caused by linear reasoning paths that prune valid entities prematurely, and (2) Error amplification, where incorrect retrieved entities mislead downstream reasoning.

To mitigate these, PARoG combines SPARQL-guided structured planning and a plan–answer–refine paradigm: The planning stage uses SPARQL queries to supervise LLMs in decomposing questions into structured, compositional sub-queries (handling conjunctions, compositions, superlatives, and comparatives). The reasoning stage first produces tentative answers using LLM parametric knowledge, then refines them using KG evidence iteratively.

Experiments on WebQSP, CWQ, and GrailQA show consistent state-of-the-art performance — e.g., +8–10% improvements over strong baselines such as PoG and ToG — while using smaller LLMs (Llama-3.1-8B). Ablation studies confirm that both SPARQL-guided planning and self-refinement contribute significantly to the gains.

### Strengths
1. Clear identification of key bottlenecks — The paper convincingly motivates the two main issues (search space truncation and error amplification) with illustrative examples (Figure 1).
2. Strong experimental results — The framework consistently outperforms prior SOTA baselines across three KGQA benchmarks, especially on complex or zero-shot queries (Table 2 and Figure 4).
3. Efficiency — Despite higher accuracy, PARoG reduces token cost and LLM calls compared to ToG and PoG (Table 5), showing practical scalability.
4. Good qualitative analyses — The case studies (Appendix E) effectively show how PARoG resolves reasoning drift and leverages LLM priors to fill KG gaps.

### Weaknesses
1. Dependence on high-quality SPARQL annotations – The approach assumes access to accurate SPARQL-grounded data. Generating and maintaining such data (using GPT-4o as a teacher) might limit reproducibility and scalability to low-resource settings.
2. Limited analysis of generalization to unseen KGs or domains – Experiments are confined to Freebase-based benchmarks; performance on other KGs (e.g., Wikidata, DBpedia) is not explored.
3. Clarity and presentation – While figures are helpful, the overall presentation could be streamlined; some sections (e.g., 3.2) are dense with notational details that obscure the main idea.
4. Limited error analysis for failure cases – The error breakdown (Appendix B) is brief and mostly quantitative; a qualitative discussion of remaining failure patterns (e.g., conflicting KG facts) would be useful.
5. Missed reference – More relevant recent works, such as [1][2], should be included and discussed for a better review.

[1] Shen, Xiangqing, Fanfan Wang, and Rui Xia. "Reason-Align-Respond: Aligning LLM Reasoning with Knowledge Graphs for KGQA." arXiv preprint arXiv:2505.20971 (2025).

[2] Zhu, Jiajun, Ye Liu, Meikai Bao, Kai Zhang, Yanghai Zhang, and Qi Liu. "Self-Reflective Planning with Knowledge Graphs: Enhancing LLM Reasoning Reliability for Question Answering." arXiv preprint arXiv:2505.19410 (2025).

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

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
This paper proposes PARoG (Plan-Answer-Refine-on-Graph), a framework designed to tackle two key challenges in the KGQA domain: error amplification and search space truncation bias. The approach relies on task decomposition to break complex questions into sub-objectives and leverages additional information sources to guide reasoning. Specifically, the authors use GPT-4 to annotate decompositions with SPARQL queries, and then fine-tune a dedicated model (the PRoG Planner) to perform this task automatically. For each sub-task, an LLM first generates an initial answer based on its parametric knowledge, which is then refined using evidence retrieved from the knowledge graph. Experiments on three widely used benchmarks—WebQSP, GrailQA, and CWQ—demonstrate that PARoG consistently outperforms strong baselines, achieving both higher accuracy and improved efficiency.

### Strengths
- The paper clearly identifies two key challenges in KGQA—error amplification and search space truncation bias—and proposes a framework that directly addresses both issues.

- The experimental evaluation is comprehensive: the method is tested on three widely used datasets. And results show that the proposed approach outperforms all baselines, and additional efficiency analysis highlights its ability to improve both accuracy and resource usage.

- The experimental findings are particularly interesting; for example, Table 4 shows that the SPARQL-supervised Llama-3.1-8B planner outperforms directly using GPT-3.5 or DeepSeek-R1 as the planner itself, suggesting that structured evaluation and execution can surpass direct generation—a concrete illustration of the idea that evaluation is easier than generation.

### Weaknesses
1. The approach heavily relies on annotation-based task decomposition. While decomposition alleviates the problem of search space explosion, it does not fundamentally resolve it—if a sub-task still corresponds to a large search space, pruning remains unavoidable. Thus, the method mitigates but does not eliminate the need for pruning.

2. During the exploration stage, the framework still depends on LLMs to score and select which paths to follow, effectively using the LLM itself as the policy. However, the paper does not propose any specific optimization or learning strategy for this decision-making process, leaving open the possibility that important candidate paths might still be pruned prematurely.

3. On the evaluated benchmarks, LLMs may possess sufficient parametric knowledge to provide useful initial answers, serving as an additional information source. However, in other datasets or domains where the LLM lacks such prior knowledge, initial answers are more likely to be incorrect, and the self-refinement strategy could then introduce additional noise rather than reducing errors.

4. The writing lacks clarity. Many of the key ideas are only implicit, requiring readers to infer the underlying insights, while the strengths of the method and the reasons behind its strong empirical performance are not clearly articulated. In addition, Figure 1-I(b) appears to contain an inconsistency: the decomposition process described in the text does not align with the example shown. Furthermore, important experimental details are not clearly presented, such as which specific models were used in each stage of the pipeline.

### Questions
See Weakness above. Additional questions:

1. Could the authors provide statistics on the proportion of initial answers that are correct or at least partially contain the correct entity in the evaluated datasets?

2. During the exploration stage, would it be possible to train a dedicated model to assist with relation/entity pruning, rather than relying solely on the LLM’s scoring? Would this lead to more accurate path selection?

3. Beyond the planning stage, what specific models are used for initial answering, exploring and refinement? Are these also performed by the fine-tuned LLaMA-3.1, or by other models?

4. In the ablation study (Table 4), the paper mentions replacing the planning module with different LLMs. Were the subsequent pruning stages (relation/entity selection) also replaced accordingly, or were they kept fixed? The experimental setup is not entirely clear.

5. Regarding the error analysis in Appendix B, could the authors provide more details about the types of errors observed and how they impact the overall results?

6. A very recent paper (Plan-on-Graph: Self-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs, arXiv 2410.23875) also addresses similar challenges in KGQA by combining task decomposition with adaptive exploration and self-correction. How does your planner-based decomposition fundamentally differ from their task decomposition mechanism, and what unique advantages does it provide?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a novel framework for LLM reasoning on knowledge graphs, termed plan-answer-refine-on-graph (PARoG). To be specific, the proposed method leverages SPARQL queries from KG data as references, decomposing them into structured step-by-step plans. Then, PARoG trains LLMs to construct such structured plans for retrieval on KGs. In addition, a plan-answer-refine paradigm is adopted by first answering each sub-query and then refining the predictions. Extensive experiments demonstrate the effectiveness of the proposed method.

### Strengths
1.	This paper is well-organized and easy to follow.
2.	This paper provides the source code to ensure the reproducibility of the proposed method.
3.	Extensive experiments show the effectiveness of the proposed method.

### Weaknesses
1.	The figures could be further refined to enhance readability. In particular, the font size in Figures 1 and Figure 2 may be quite small.
2.	The paper may lack some baseline methods for comparison, such as GNN-RAG [1], SubgraphRAG [2].
3.	The core idea may be similar to PoG, as both methods adopt a comparable pipeline for answer generation (i.e., plan–answer–refine). What is the key difference between the proposed method and PoG?
4.	As stated in the exploration section, the KG exploration process in the proposed method appears similar to existing approaches. How does the proposed method address the associated challenges?

[1] Mavromatis, Costas, and George Karypis. "Gnn-rag: Graph neural retrieval for large language model reasoning." arXiv preprint arXiv:2405.20139 (2024).

[2] Li, Mufei, Siqi Miao, and Pan Li. "Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation." The Thirteenth International Conference on Learning Representations.

### Questions
Please see in Section Weaknesses

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
The paper proposes **Plan-Answer-Refine-on-Graph (PARoG)**, a KG-augmented LLM framework designed to address two limitations of prior KGQA agents: **(i)** *search-space truncation bias* from linear, top-k-pruned walks and **(ii)** *error amplification* from over-reliance on retrieved but partially relevant entities. PARoG has two pillars. First, **SPARQL-guided structured planning**: the authors decompose SPARQL queries into sub-objectives (conjunction, composition, comparatives, superlatives) and train a relatively small planner (Llama-3.1-8B) via SFT to emit stepwise plans with uniform granularity. Second, a **Plan→Answer→Refine** loop: for each sub-objective, the model first proposes a tentative answer using parametric knowledge, then executes KG exploration and **self-refines** the answer against retrieved triples, explicitly checking alignment and stopping when sufficient. Experiments on **WebQSP, CWQ, and GrailQA** report SOTA Hits@1 with fewer LLM calls and tokens than ToG/PoG; ablations show (a) self-refinement materially helps and (b) the 8B SPARQL-supervised planner outperforms much larger planning LLMs (GPT-3.5, DeepSeek-R1).

### Strengths
* **Addresses core failure modes with targeted design.** SPARQL-guided planning avoids linear-path pruning; Answer→Refine reduces retrieval-induced error amplification. 
* **Solid, multi-benchmark gains with efficiency.** PARoG improves Hits@1 over ToG/PoG/KG-Agent on WebQSP, CWQ, and GrailQA, with fewer calls and tokens.  
* **Decision-useful ablations.** Self-refinement helps markedly (esp. with GPT-3.5); the SPARQL-trained 8B planner outperforming larger LLMs is practically important. 
* **Clear problem decomposition.** The taxonomy of sub-query types (conjunction/composition/comparatives/superlatives) aligns well with KGQA structure.

### Weaknesses
1. **Statistical rigor.** Main results lack **CIs/multi-seed variance**; given RL-like exploration and LLM stochasticity, uncertainty reporting is important. 
2. **Planner data generation & potential biases.** The **SPARQL→plan** dataset (74,802 examples via GPT-4o) is central, but quality controls, inter-annotator checks, and error taxonomies are not reported; robustness to noisy/ambiguous SPARQL is unclear. 
3. **Faithfulness audits.** The refinement step relies on LLM judgments of alignment; the paper does not quantify hallucinated/irrelevant triples that slip through nor enforce execution-based correctness constraints during refinement. 
4. **Generality beyond Freebase.** No experiments on **Wikidata/DBpedia** or text-augmented regimes; claims of broad applicability would be stronger with heterogeneous KGs or KG incompleteness stress tests. 
5. **Missing engineering details.** Exact prompts, stopping criteria, and filtering thresholds for relation/entity selection are described at a high level; reproducibility would benefit from fuller protocol disclosure.

### Questions
1. **Statistics & stability.** Please report **means ± 95% CI over ≥3 seeds** for all main tables and ablations; include sensitivity to planner size and decoding parameters. 
2. **SPARQL→plan pipeline.** How do you validate the GPT-4o decompositions for **semantic equivalence** to the original SPARQL, and what is the estimated error rate? Any manual spot-checks? 
3. **Execution-faithfulness checks.** During refinement, can you enforce **KG-execution constraints** (e.g., verifying that asserted triples exist; type constraints) and report the rate at which refinement overturns an initially wrong tentative answer? 
4. **Generality.** Do results transfer to **Wikidata** (different schema and aliasing) and to **KG+text** settings (e.g., Freebase+Wikipedia)? If not, what components would need adaptation?
5. **Efficiency trade-offs.** Table 5 shows fewer calls and tokens; could you add **cost-accuracy curves** (vary beam/iterations) and detail caching/parallelization assumptions used for accounting? 
6. **Error analysis.** Appendix B indicates high failure due to missing answer entities in retrieval for prior methods; can you provide a comparable breakdown for PARoG and the fraction resolved by the refine stage?

### Soundness
3

### Presentation
3

### Contribution
3
