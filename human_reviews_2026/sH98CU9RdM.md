# When Knowledge Hurts: Enriching Domain Knowledge for Causal Scientific Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Science has long sought to uncover principles of discovery, yet fields like materials science remain slow and labor-intensive. While Large Language Models (LLMs) can accelerate progress by integrating domain knowledge, we reveal a critical failure mode: \textbf{\textit{contextual tunneling}}, where naive knowledge integration causes LLMs to over-anchor on narrow retrieval paths while suppressing broader parametric reasoning. Through the evaluation in materials discovery, we demonstrate that naive knowledge graph augmentation degrades performance by 15–35\% on key reasoning tasks compared to direct prompting. 
To address this challenge, we introduce \texttt{ARIA} (Autonomous Reasoning Intelligence for Atomics), a causal-aware framework featuring: (i) hierarchical reasoning that provides graceful degradation to knowledge graph sparsity, (ii) enhanced analogical transfer for robust reasoning, (iii) knowledge graph enrichment through online searching. Extensive experiments show that while naive KG integration consistently underperforms baseline LLMs, \texttt{ARIA} not only recovers this loss but also provides interpretable causal explanations by tracing reasoning through the knowledge graph, enabling scientists to verify and trust its outputs. Our work demonstrates that external knowledge can inadvertently constrain reasoning and establishes a principled framework for robust KG–LLM integration in scientific discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
### **1. Research Problem**

* The paper addresses the paradox that external knowledge integration can harm LLM reasoning in scientific domains.
* It identifies a phenomenon called “contextual tunneling”, where excessive or poorly filtered domain knowledge narrows the model’s reasoning scope, reducing causal inference and generalization.


### **2. Motivation**

* Existing retrieval-augmented systems lack causal awareness and reliability control over injected knowledge.
* In scientific reasoning, this leads to biased or misleading conclusions.
* The authors aim to design a framework that **selectively and hierarchically integrates knowledge**, preserving both accuracy and interpretability.


### **3. Proposed Method**

* The study introduces ARIA (Autonomous Reasoning Intelligence for Atomics), a three-tier causal reasoning framework.

  1. **Direct Causal Reasoning:** Uses explicit causal paths from a Causal Knowledge Graph (CKG).
  2. **Analogy-Based Transfer:** Infers from analogous entities via an enhanced similarity metric combining semantic, factual, and numeric consistency.
  3. **Parametric Fallback:** Reverts to internal LLM knowledge when external data are unreliable.
* The CKG is automatically built from scientific texts, encoding cause–effect–mechanism tuples for interpretable reasoning.

### **4. Experimental Results**

* Tested on materials discovery tasks (forward prediction and inverse design).
* Naïve and online knowledge augmentation reduced performance by 20–35%, confirming the “knowledge hurts” effect.
* ARIA restored and improved performance by +15–36%, enhancing reasoning quality (+44%) and interpretability (+37%).

### Strengths
### **1. Importance of the Research Problem**

This paper tries to addresses a critical and underexplored issue: how external knowledge can harm rather than help LLM reasoning in scientific domains. The problem is highly significant for building trustworthy AI systems in science and engineering.


### **2. Strengths of the Proposed Method**
* The ARIA framework is a novel causal-aware design that hierarchically integrates knowledge, balancing external facts and model reasoning.
* Its three-tier architecture (causal reasoning, analogy transfer, fallback) is clear, interpretable, and robust.


### **3. Strengths of the Experimental Results**

* Experiments are comprehensive and convincing, showing large and consistent gains (+15–36%) over naive methods and recovers performance lost from over-augmentation.
* Results are well-analyzed and highlight both accuracy and interpretability improvements, confirming practical significance.

### Weaknesses
* The notion of “contextual tunneling,” while compelling, is not formally defined or quantified—its diagnosis relies on qualitative intuition.
* No discussion of alternative explanations (e.g., prompt design flaws, data noise) for the observed performance drop with naive KG use.
* The study focuses narrowly on materials science; it’s unclear how generalizable the phenomenon or solution is to broader scientific or real-world reasoning tasks.


* ARIA’s tiered architecture, while well-motivated, involves manual design choices (e.g., thresholds, cascade conditions) that may limit scalability.
* The Causal Knowledge Graph construction pipeline is somewhat opaque, especially in terms of factual reliability and noise filtering.



* Evaluation is mostly task-specific; there’s no ablation comparing ARIA with partial variants (e.g., Tier 1 only, no fallback).
* Human evaluation is missing, especially to assess the quality of causal chains or explanations generated.
* The paper lacks runtime/performance analysis— it’s unclear how computationally efficient ARIA is compared to standard RAG or other KG-augmented methods.

### Questions
1. Regarding the notion of “contextual tunneling”:

   * Can you offer a more insightful or quantitative definition, beyond the current qualitative description?
   * How do you rule out alternative explanations for the performance drop, such as prompt design flaws or retrieval noise?

2. About generalization:

   * The experiments are focused on materials science. Do you expect the same issues and benefits of ARIA to hold in other domains, such as biomedicine or commonsense reasoning?
   * Have you tested, or do you plan to test, ARIA in other types of causal reasoning tasks?

3. About the method design:

   * Several components of ARIA rely on manual thresholds or rules. How sensitive is the system to these choices, and could they be learned or adapted dynamically?
   * The causal knowledge graph construction process is not fully explained—how do you ensure the reliability and accuracy of extracted causal tuples?

4. About the evaluation:

   * Have you run ablation studies to isolate the contribution of each tier (e.g., using only Tier 1 or Tier 2)?
   * Was any human evaluation conducted to assess the interpretability or usefulness of ARIA’s causal chains?

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
This paper introduces ARIA (Autonomous Reasoning Intelligence for Atomics), a causal-aware framework designed to mitigate “contextual tunneling” — a failure mode where naive knowledge graph (KG) integration constrains an LLM’s reasoning. ARIA employs a three-tier hierarchical reasoning cascade (direct causal reasoning, analogical transfer, and parametric fallback) and dynamic knowledge graph enrichment through online retrieval. Experiments in materials science show that naive KG integration can degrade performance by 20–35%, while ARIA recovers this loss and improves interpretability and reasoning quality.

### Strengths
1.	The paper’s figures and the overall writing are generally clear and easy to follow. 
2.	The paper identifies a relevant and underexplored issue (“contextual tunneling”) in knowledge-augmented reasoning.
3.	The hierarchical design of ARIA is clearly described and intuitive.

### Weaknesses
1.	The paper acknowledges that ARIA’s performance depends heavily on the quality of the constructed causal KG. However, it does not quantify how KG completeness or noise affects reasoning outcomes. It would be useful to see an ablation or sensitivity analysis showing how ARIA behaves with imperfect graphs — e.g., noisy edges or missing causal paths.
2.	In Figure 5 and Table 1, ARIA improves over “Naive KG+LLM” but often remains close to or below the Baseline LLM that uses no external knowledge. This raises the question: if ARIA cannot consistently outperform the baseline, is causal KG integration still beneficial? The paper should clarify under which conditions the added structure becomes an advantage.
3.	In several sub-figures (e.g., 5b, 5d), ARIA only marginally surpasses the Online KG baseline. The paper should discuss whether this marginal gain is statistically significant and what aspects of hierarchical reasoning contribute most to the improvement.
4.	The “LLM-as-judge” evaluation is convenient but potentially biased. The authors should discuss this limitation and, if possible, complement it with results on well-established benchmarks or human expert evaluations.
5.	Some figures (e.g., Figure 5e) have legends that overlap with data, which reduces readability. The authors should improve figure layout and labeling for clarity.

### Questions
Please refer to weaknesses.

### Soundness
2

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
4

### Summary
This paper introduces ARIA (Autonomous Reasoning Intelligence for Atomics), a causal reasoning framework that addresses a key failure mode in RAG systems for scientific discovery — termed “Contextual Tunneling.” Contextual tunneling occurs when naive knowledge integration from external sources causes a LLM to over-anchor on narrow, irrelevant retrieved contexts, reducing its broader parametric reasoning capability. ARIA mitigates this problem through a three-tiered reasoning cascade: (1) Direct Causal Path Reasoning: Uses explicit causal paths from a structured Causal Knowledge Graph (CKG) for grounded, high-fidelity reasoning. (2) Analogy-Based Transfer: Retrieves and adapts analogous causal paths using an enhanced similarity metric that integrates semantic, factual, and numerical compatibility. (3) Parametric Fallback: Reverts to the LLM’s internal knowledge when no reliable causal or analogical evidence exists. A comprehensive evaluation on materials science discovery shows that naive KG integration degrades performance by 20–35%, while ARIA recovers and surpasses baseline LLMs, improving reasoning quality and interpretability by up to +44%. The authors provide detailed ablations, visualizations, and case studies demonstrating ARIA’s “rescue effect.”

### Strengths
1. The paper identifies contextual tunneling as a fundamental and underexplored failure mode in LLM–knowledge integration. This reframing moves beyond the “more knowledge is better” assumption that dominates RAG literature.

2. The hierarchical reasoning cascade is elegant and well-motivated. It gracefully handles uncertainty and graph sparsity by balancing symbolic causality, analogical inference, and parametric fallback.

3. Experiments are thorough and domain-grounded, using a curated dataset of 149 synthesis–property relations across 85 materials systems. Results convincingly show ARIA’s superiority, especially in out-of-domain generalization.

4. ARIA not only predicts accurately but produces verifiable causal paths and analogical traces, aligning well with the interpretability goals of scientific AI.

### Weaknesses
1. The observation that blindly integrating KG information into context can harm the performance of LLMs in scientific reasoning is claimed in recent works [1], the authors of ARIA further formalize it with the term "contextual tunneling", which should be included in discussion. 

2. All experiments are in materials science, which is a compelling but narrow domain. It remains unclear how ARIA generalizes to other sciences (e.g., chemistry, biology).

3. The scoring framework employs Gemini as an automated “LLM judge.” While this is practical, it risks circularity — relying on a similar model to assess reasoning quality.

4, Constructing and maintaining the Causal KG via dynamic web search and causal extraction could be expensive and domain-specific. The paper doesn’t benchmark runtime or scalability.


[1] GIVE: Structured Reasoning of Large Language Models with Knowledge Graph Inspired Veracity Extrapolation

### Questions
1. How scalable is ARIA’s causal knowledge graph construction for domains with millions of entities (e.g., PubChem, Wikidata)?

2. What are the computational overheads of online enrichment and tier selection compared to standard RAG?

3. How robust is the enhanced similarity metric to noisy or conflicting analogies — can it detect and downweight unreliable causal paths?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
** Summary

The work studies LLMs’ generation quality based on the knowledge graph based RAG technique. It specifies the issue that injecting naive or irrelevant knowledge to LLMs may negatively influence the generation quality, termed as contextual tunneling. To tackle this problem, the work extensively utilises LLMs to generate reliable answers through RAG in a three-step pipeline. The pipeline first constructs a causal graph. Then, if the answer can be well reasoned by the causal graph, the LLMs reason based on the causal graph, otherwise, it searches causal paths that can likely connect to the query-answer pair based on semantic and domain knowledge similarity. If both of the steps failed, the pipeline directly query the LLMs to generate an answer. The work then uses the pipeline to solve a two-direction material problem, and evaluates its performance on a synthetic dataset.

** Recommendation 

I would like to recommend a weak rejection to this work for its mild novelty, possible unsoundness in the causality part, and presentation. Generally, I think this work is interesting as an extensive application of LLMs, however, its contribution and soundness are limited. For instance, works that use knowledge graph in RAG process exist.

### Strengths
1. The work uses causal graph in the RAG process to enhance the relevance of retrieved information. In the pipeline, the authors take care of aligning the process with domain knowledge.
2. The experiment results somewhat support their claimed issue of RAG.
3. The paper presents an interesting application to the material domain, and evaluates the method on domain expert annotated datasets.

### Weaknesses
1. Their method of constructing causal graph needs further clarification, and I have concern regarding their method of modifying the causal graph. The paper describes its first step is to construct a causal graph as a DAG, however, it is not clear how they ensure the graph to be acyclic. The method attempts to modify the causal graph based on semantic and domain knowledge similarity, instead of statistical correlation reasoning. This is quite different from the standard in directions such as causal discovery. So, I would recommend to change the term to knowledge graph instead of using causal graph.
2. I would expect more details of the technical description, e.g., how they compute FC and NC.
3. The experimental results show that it seems the best performance is the basic LLM. Though it addresses the issue of RAG, however, the results undermine the motivation of using RAG.
4. The work uses LLM to evaluate the performances. This may be not reliable.

### Questions
See the weakness section.

### Soundness
2

### Presentation
2

### Contribution
2
