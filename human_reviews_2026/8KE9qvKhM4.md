# SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Real-world Table–Text question answering (QA) tasks require models that can reason across long text and source tables, traversing multiple hops and executing complex operations such as aggregation. Yet existing benchmarks are small, manually curated—and therefore error-prone—and contain shallow questions that seldom demand more than two hops or invoke aggregations, grouping, or other advanced analytical operations expressible in natural-language queries. We present SPARTA, an end-to-end construction framework that automatically generates large-scale Table–Text QA benchmarks with lightweight human validation, requiring only one quarter of the annotation time of HybridQA. The framework first constructs a reference fact database by enriching each source table with grounding tables whose tuples are atomic facts automatically extracted from the accompanying unstructured passages, then synthesizes nested queries whose number of nested predicates matches the desired hop count. To ensure that every SQL statement is executable and that its verbalization yields a fluent, human-sounding question, we propose two novel techniques: provenance-based refinement, which rewrites any syntactically valid query that returns a non-empty result, and realistic-structure enforcement, which confines generation to post-order traversals of the query graph. The resulting pipeline produces thousands of high-fidelity question–answer pairs covering aggregations, grouping, and deep multi-hop reasoning across text and tables. On SPARTA, state-of-the-art models that reach over 70 F1 on HybridQA or over 50 F1 on OTT-QA drop by more than 30 F1 points, exposing fundamental weaknesses in current cross-modal reasoning. We will release the benchmark, construction code, and baseline results to spur progress toward robust, realistic Table–Text QA models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
SPARTA introduces a large-scale, principled benchmark for Table-Text Question Answering (QA) by addressing critical limitations—shallow reasoning, small data scale, and annotation noise—inherent in prior manual datasets like HybridQA. The core methodology involves constructing a unified Reference Fact Database by incorporating text-derived atomic facts into structured Grounding Tables, enabling an LLM-guided pipeline to synthesize highly complex, executable SQL queries that model multi-hop, tree-structured reasoning across modalities. Utilizing novel techniques like provenance-based refinement and post-order structural enforcement, SPARTA efficiently generates thousands of high-fidelity question-answer pairs covering advanced analytical operations such as grouping and aggregation, leading to a profound performance collapse (over 30 F1 points) in state-of-the-art QA models, thereby establishing a new, rigorous standard for cross-modal reasoning evaluation.

### Strengths
Strength 1: High-Fidelity Scaling and Data Rigor SPARTA dramatically increases the realism of Table-Text QA by operating over large-scale databases featuring thousands of rows (e.g., 3,280.5 mean cardinality), moving well beyond the small, toy-scale web tables typically used in previous benchmarks. This substantial increase in data size rigorously evaluates model performance in environments demanding true large-database reasoning and efficient indexing.   


Strength 2: Comprehensive Coverage of Tree-Structured Multi-Hop and Analytical Reasoning The framework systematically generates questions covering advanced analytical operations such as aggregation and grouping, which were largely absent from prior benchmarks. Furthermore, it features deep, tree-structured multi-hop reasoning, which goes beyond the simplistic linear chains of inference found in existing QA resources.   


Strength 3: Cost-Effective Rigor via Provenance The generation pipeline achieves high data quality and efficiency by employing Provenance-based Refinement, guaranteeing every synthesized SQL query is executable and returns a non-empty result. This mechanism ensures semantic validity while drastically reducing the cost of generation, requiring only one quarter of the annotation time compared to manual benchmarks like HybridQA.

### Weaknesses
Weakness 1: Undocumented Quality of Atomic Fact Extraction The crucial first step of automatically extracting atomic facts from unstructured text into grounding tables lacks reported validation or error analysis. If errors in this unquantified automated extraction process propagate, they could undermine the benchmark's fundamental data grounding and rigor.   


Weakness 2: Acute Cross-Modal Fragility in Program-Based Models The inclusion of unstructured text causes a significant F1 score drop (e.g., 63.9% for HProPro), indicating fundamental brittleness in program-based reasoning when integrating facts from relational tables and textual grounding tables. This massive performance collapse suggests inadequate logical bridging and factual integration capabilities, especially for complex analytical queries that span the heterogeneous data store.   


Weakness 3: Vulnerability to Correlated Joins and Aggregation Models demonstrate pronounced vulnerability to the synergistic complexity introduced by combining correlated joins and aggregation (Type-JA queries), yielding the lowest reported F1 scores. This quantitative failure highlights that state-of-the-art models struggle with highly constrained, context-sensitive analytical calculations typical of advanced database logic.   


Weakness 4: Scaling Limitations Across Query Structural Dimensions Empirical results show model performance degrades consistently and sharply as query complexity increases along either the depth (chain length) or breadth (branching structure) dimensions of the query graph. This suggests current methods fail to generalize tree-structured reasoning, exhibiting fundamental computational limitations in managing multiple, concurrent relational paths.   


Weakness 5: Costly Dependence on LLM Corrective Feedback Despite efficiency efforts, the nested query generation pipeline is inherently reliant on numerous expensive error-correction loops, requiring 53.6% more LLM calls than the theoretical ideal. This high overhead dedicated solely to provenance-based repair reveals the intrinsic unreliability of LLMs in autonomously generating semantically sound SQL without database execution feedback.   


Weakness 6: Limited Scope and Insufficient Validation Granularity The benchmark is confined strictly to the Table-Text modality, neglecting other real-world data sources like images or video, thus limiting its applicability in the evolving multimodal QA landscape. Furthermore, the human validation is explicitly "lightweight," avoiding verification of the complex multi-hop reasoning path itself, which risks overlooking subtle semantic misalignment between the generated SQL and the final natural language question.

### Questions
same as weakness

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
3

### Summary
This paper introduces SPARTA, a large-scale Table–Text Question Answering (QA) benchmarks. Unlike existing small and shallow datasets, SPARTA generates complex multi-hop and aggregation-based QA pairs by synthesizing executable SQL queries and corresponding natural-language questions. Through provenance-based refinement and realistic-structure enforcement, it produces thousands of high-quality question–answer pairs with only a quarter of HybridQA’s annotation cost, revealing significant performance drops in state-of-the-art models and highlighting current limitations in cross-modal reasoning.

### Strengths
1. The paper is clear and easy to follow.
2. The analysis of existing benmarks is comprehensive.
3. Several experiments along with analysis on this benchmark.

### Weaknesses
1. **Excessive Dependence on High-Capacity LLMs for Pipeline Efficiency**.
The efficiency of the Provenance-based Refinement loop relies heavily on the advanced reasoning capability of a large LLM (Llama-3.1-70B-Instruct) to accurately diagnose and correct erroneous SQL predicates. However, the paper does not analyze how the framework’s quality and cost-effectiveness (central to its “Scalable” claim) would be affected when using smaller or less capable LLMs. This raises concerns about the robustness, reproducibility, and general accessibility of the proposed generation pipeline.

2. **Lack of Fine-Grained Error Attribution in End-to-End Settings.**
The error analysis highlights major failure categories such as “Relevant data missing” and “Erroneous data analysis”. However, in the end-to-end setting, it remains unclear whether these failures stem from inherent weaknesses in the reasoning and program execution components or from compounded upstream retrieval errors. A more fine-grained breakdown of error contributions across the Retrieval and Reasoning stages is essential to accurately diagnose the true limitations revealed by SPARTA.

3. **Limited Rigor in Query Naturalness Evaluation.**
The evaluation of question naturalness relies on verbalizing SQL queries using LLMs (AST-ICL and ChatGPT-4o). This top-down SQL → NL generation process may not faithfully reflect the organic intent or linguistic variability of real user questions. As a result, the claimed “realism” of the complex, tree-structured questions could be biased or overstated.

4. **Oversimplification of Reference Fact Database Construction.** The foundational step of the pipeline involves merging source tables with Grounding Tables, which are created by automatically extracting atomic facts from unstructured text passages. However, the paper primarily relies on a pre-processed corpus (ROTOWIRE) within the NBA domain, and provides insufficient technical detail on the general and robust mechanisms for extracting and normalizing atomic facts from arbitrary text into structured Grounding Tables. This omission raises concerns about the reliability and domain-agnostic portability of the proposed pipeline.

5. This dataset contains three subsets only, which could be expanded in to larger range of QAs.


Overall, I believe there is still considerable room for improvement in this paper, and the weaknesses outweigh the strengths. Therefore, I assign a final rating of borderline reject.

### Questions
Please see weaknesses.

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
3

### Summary
This paper proposed a framework to scale up synthetic data generation for multi-hop question answering over table+text contexts.
It creates new tables by merging existing tables with atomic facts extracted from texts, then use a subset to verbalize as the text context.
Questions are generated using LLMs to build nested SQL queries bottom-up, then verbalize the queries that will yield result after execution. It also introduced an automated provenance-based approach to fix SQL query.

Contributions
1. The approach seems reasonable for constructing multi-hop type questions
2. Benchmark is challenging and shows there's still significant headroom for improvements.

### Strengths
1. Paper is well written and clear. Results demonstrate effectiveness of the dataset construction approach
2. Automated fixing and provenance-based fixing is a pretty creative idea
3. Substantial analysis of model failures on this dataset is also presented

### Weaknesses
1. seems to be missing the constructed dataset statistics? e.g. how many reference tables are constructed per source dataset?
2. scalability of this approach seems to be bottlenecked by number of source tables?

### Questions
Mostly minor suggestions on presentations, since this paper has a lot of details and analysis

1. What is the why-not based provenance approach (referenced paper only has why-provenance)? I can kind of infer from Figure 3 but maybe have some textual description would help?
2. How is provenance-based approach used to help fix the error => seems like the provenance analysis is added to LLM prompt to ask LLM to fix the query/relax the offending constraints, but it is unclear to me how this is used & what the inferior alternatives are doing when I first read through it -- especially in section 3.3.2 -- and Figure 3 only covers Provenance-based approach, so at first it was confusing how the alternatives are doing the fixes (plus figure 3 is small). Maybe adding some textual descriptions to clarify could help

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
3

### Summary
This paper proposes SPARTA, a Text-Table Q&A dataset which is primarily generated by LLM prompts, with light human evaluation. This allows the authors to generate a large corpus of Q&A problems, over varying aspects of the problem instance. The authors provide extensive and high-quality evaluation of the dataset, including qualitative and ablation results.

### Strengths
Overall, this paper is very well written and I see great value in the proposed dataset. 

1. The scope and scale of the dataset is seemingly quite novel. There is great care to the tested aspects of the dataset, and the overall size is sufficiently large for high-quality model comparison. 

2. Each aspect of the dataset design is well motivated, and easy to interpret. This paper is interpretable for the general AI research reader. 

3. The dataset evaluation is sufficient and provides several insightful, qualitative results.  e.g. Fig 5, Table 6. 

4. The authors give a very extensive appendix with examples.

### Weaknesses
1. A primary weakness is the lack human-written and human-curated Q&A instances. While the synthetic generation methodology is impressive and well-evaluated, the authors could provide small curated data subsets for testing certain aspects of reasoning. For example, reasoning over ranges or negations, data inconsistencies (e.g. where text and tables have differing values for a specific fact), etc. 

2. Similarly, While the scope of SPARTA is impressive and there is variation over structural, breadth, and clause diversity, more qualitative evaluation could be provided related to breadth in weakness (1). It is difficult to understand *what* model aspects are being tested (and not tested) over the entirety of the dataset. 

3. The dataset doesn't include human-verified perturbation groups for robustness analysis. Since machine-guided perturbation models often change the result of complex queries, it would be valuable to validate over a subset of human-verified perturbations (e.g. query rephrasing that retains the same output). 

4. Table 6 is somewhat concerning. Given poor text performance, is this problem primarily Tabular Q&A? Does SPARTA have a view where users can consistently use it for Tabular Q&A evaluation?(e.g. no comparability errors from users).

### Questions
1. Could you more succinctly summarize the aspects which are tested by this dataset, providing toy examples? (lengthy) examples are deep in the appendix. Can this be promoted to the main text in a better way than Table 1? (Perhaps the text around Table 1 can be more contrastive), or a taxonomic figure can be added. 

2. Are there aspects which were omitted from the dataset at design-time?
a. robustness as described above
b. chain of thought reasoning
c. user evaluation to evaluate naturalness and scope of Q&A 
d. grounded attribution and/or explanation quality
e. potential bias in evaluation, e.g. evaluation over generated sensitive attribute subgroups or subgroup-perturbations (curation scope r.e. Weakness 3 above)

Could you summarize the challenges or considerations in scoping the evaluation as you did?

### Soundness
4

### Presentation
4

### Contribution
4
