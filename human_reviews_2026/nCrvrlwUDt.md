# Bridging Global Intent with Local Details: A Hierarchical Representation Approach for Semantic Validation in Text-to-SQL

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Text-to-SQL translates natural language questions into SQL statements grounded in a target database schema. Ensuring the reliability and executability of such systems requires validating generated SQL—but most existing approaches focus only on syntactic correctness, with few addressing semantic validation (detecting misalignments between questions and SQL). As a consequence, how to achieve effective semantic validation still faces two key challenges: capturing both global user intent and SQL structural details, and constructing high-quality fine-grained sub-SQL annotations.
To tackle these, we introduce HeroSQL, a hierarchical SQL representation approach that integrates global intent (via Logical Plans, LPs) and local details (via Abstract Syntax Trees, ASTs). 
To establish better information propagation, we further employ a Nested Message Passing Neural Network (NMPNN) to capture inherent relational information in SQL and aggregate schema-guided semantics across LPs and ASTs. 
Additionally, to generate high-quality negative samples, we propose an AST-driven sub-SQL augmentation strategy, supporting robust optimization of fine-grained semantic inconsistencies.
Extensive experiments conducted on Text-to-SQL validation benchmarks (in-domain and out-of-domain settings) demonstrate that our approach outperforms existing state-of-the-art (SOTA) methods, achieving an average 9.40\% improvement of AUPRC and 12.35\% of AUROC in identifying semantic inconsistencies.
It excels at detecting fine-grained semantic errors, provides large language models with more granular feedback, and ultimately enhances the reliability and interpretability of data querying platforms. Our code is anonymously available at https://anonymous.4open.science/r/HeroSQL.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the problem of semantic validation in Text-to-SQL systems, determining whether a generated SQL query correctly captures the meaning of a natural language question. The authors propose HEROSQL, a hierarchical representation approach that integrates Logical Plans (LP) for global semantic intent and Abstract Syntax Trees (AST) for local structural details. These representations are processed via a Nested Message Passing Neural Network (NMPNN) to capture multi-level semantic relations. To overcome data scarcity, the paper introduces an AST-based sub-SQL augmentation strategy that generates challenging negative samples, supplemented by LLM-based data augmentation. Experiments on BIRD, Spider, and EHRSQL datasets show improvements in AUPRC and AUROC over several baselines (Prompting, CoT, ConfScore, COVE, TED).

### Strengths
- *S1- Novel hierarchical representation*: Integrating LP and AST for semantic validation is a promising and technically meaningful idea. It captures both global intent and local structural details.

- *S2- Empirical gains*: The method achieves consistent improvements over existing baselines across multiple datasets, suggesting its effectiveness in capturing fine-grained semantic errors.

- *S3- Augmentation strategy*: The AST-driven negative data generation is a creative way to address data scarcity for supervised semantic validation.

### Weaknesses
- *W1- Clarity and organization*:
  The paper is difficult to follow. While individual components (LP, AST, NMPNN) are described in depth, their integration and flow within the full pipeline are not clearly presented. Figure 1 is dense and does not sufficiently illustrate how the components interact. A concise system overview early in the paper would help orient readers.
  
- *W2- Baseline clarity and fairness*:
  The baselines (Prompt, CoT, ConfScore, COVE, TED) are only briefly mentioned, and their adaptation to the semantic validation setting is unclear. For instance, Chain-of-Thought (Wei et al., 2022) and CoVe (Dhuliawala et al., 2024) are general approaches; the paper should explain how they are adapted for text-to-SQL validation. Moreover, HEROSQL is trained in a fully supervised manner using labeled (question, SQL) pairs, while most baselines appear unsupervised or heuristic. This difference should be acknowledged and controlled for (e.g., through fine-tuned supervised baselines).
  
- *W3- Assumptions not fully justified*:
  The claim that hierarchical intermediate representations yield more accurate semantics is plausible but not empirically validated. The assumption that LLM- and AST-based augmentations generalize well remains underexplored. The out-of-domain test on EHRSQL is too limited (only two databases) to conclusively demonstrate generalization.
  
- *W4- Writing and presentation quality*:
  The prose is verbose, with long sentences and repeated statements. Important contributions and intuitions are buried in technical detail. The introduction and methodology sections would benefit from clear motivation, schematic diagrams, and more intuitive examples of semantic errors being detected.

### Questions
1. How are the unsupervised baselines adapted to the supervised validation setting? Are they fine-tuned on the same augmented data?
  
2. Can you provide qualitative examples of semantic inconsistencies that HEROSQL detects but baseline methods miss? I can see some examples in the appendix but I cannot tell how the baselines perform on them.
  
3. Have you analyzed how performance scales with the number or diversity of negative samples generated by the AST-based augmentation?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
HEROSQL addresses the challenge of semantic validation in Text-to-SQL, detecting when a syntactically valid query misaligns with user intent. This work introduces a hierarchical representation that fuses global intent (via logical plans) with local structure (via abstract syntax trees) and propagates information through a Nested Message Passing Neural Network. To overcome the scarcity of fine-grained labels, it employs an AST-driven sub-SQL augmentation strategy to generate diverse negative examples. In in-domain and out-of-domain benchmarks, it achieves an average +12.18 AUPRC and +13.41 AUROC improvement over prior methods.

### Strengths
1. This paper offers a clear and rigorous definition of semantic correctness in Text-to-SQL, grounded in the alignment between unstructured user intent and structured SQL components (at the AST level). While many prior works use the term loosely, this formalization is precise and actionable, and sets a strong foundation for future research in this area.

2. The technical design is solid and well-motivated. By combining AST-level and logical-plan encodings through a Nested Message Passing Neural Network, the model captures both local structure and global query intent, leading to rich, compositional SQL representations for validation.

3. HEROSQL consistently outperforms baselines across in-domain (BIRD, Spider) and out-of-domain (EHRSQL) datasets, using multiple small LLM backbones, and achieves substantial AUPRC and AUROC gains, demonstrating both effectiveness and generalizability.

### Weaknesses
1. AST perturbations are limited to the hand-crafted rule set (e.g. swapping operators, dropping predicates), which may not mimic the full spectrum of mistakes that LLMs actually make. LLM-generated negatives help diversify errors, but they’re still filtered purely by execution mismatch, so edge-case logic bugs that happen to produce the same result go unnoticed.

2. LLM-based baselines only include small LLMs. Could you justify this setup? What would be the performance if you use a commercial model like claude 4, gpt-4o, gpt-o4-mini? I suspect these models would be much stronger in determining SQL semantic correctness. 

3. Improvements in AUROC/AUPRC show a better validator. However, how do the false positives and negatives in this validator affect the downstream application (e.g., text2sql)? For example, FP may lead to overcorrection and regress an originally correct SQL. In practice, if a user wants to use this validator in their text2sql pipeline, what threshold should they set? It would be good if the paper could show that this validator can really improve text2sql e2e accuracy through this error detection and correction step. Additionally, adding a discussion on how end users should use this validator (e.g., how to set the threshold) would make the paper stronger.

### Questions
1. What is the latency of the validator for determining semantic correctness?
2. Why only compare with small models? Could you compare with commercial LLMs such as claude 4? 
3. AUPRC is much lower on Spider. Could you explain the reason?
4. In your results, is the positive class defined as correct or incorrect SQL? What is the exact class distribution in your training and evaluation datasets? Since class imbalance can significantly influence metrics like AUPRC and AUROC, clarifying this distribution is important for properly interpreting the results.
5. Could you show that your validator can improve the end-to-end performance of a text2sql pipeline? For example, use a very simple prompt to ask claude 4 do text2sql on BIRD dev, use your validator to detect semantic errors, and ask the LLM to correct erroneous queries. 
6. Is it possible to use your method on benchmarks with more complicated SQL queries (e.g., spider 2.0)? It's ok that the answer is no. but would like to understand the potential challenges.

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
5

### Summary
The paper introduces HEROSQL, a hierarchical SQL representation approach for semantic validation in Text-to-SQL. The key idea is to bridge global user intent and local SQL structure by using logical plans and AST. The authors propose a nested message passing neural network to aggregate information from AST, LP to full query. An AST-driven sub-SQL augmentation strategy is introduced to generate negative examples. The experimental evaluation shows some effectiveness of the proposed method.

### Strengths
S1. Fine-grained negative augmentation enabling sub-SQL feedback is practical and useful.

S2. The KV-cache and schema compression make the proposed method more practical.

S3. The proposed method with small LLMs shows reasonable performance compared to baseline methods.

### Weaknesses
W1. My primary concern - using LPs to represent global intent is not convincing as LP is also based on the generated SQL, which may not reflect the semantic meaning in the original NL question.

W2. The experimental evaluation is not solid. The authors only provide limited qualitative analysis. The chosen baselines are weak or general purpose error detection, not specific to Text-to-SQL. Hence the results are not convincing to show the actual effectiveness of the proposed method in real-world settings.

W3. The training cost and scalability discussion are limited. Specifically, NMPNN + AST parsing + LP may add much complexity compared to lightweight baselines using LLMs. Also the context length of small models is not scalable to enterprise settings where tables often have hundreds of columns and millions of rows, and complex SQLs' length can exceed 6,000 lines/tokens, or a million characters, as they involve numerous Common Table Expressions (CTEs), multiple joins, unions, and intricate conditional logic.

### Questions
Q1. How robust is HEROSQL to noisy schema names or databases with auto-generated identifiers? Datasets such as AmbiSQL could be useful to evaluate the robustness/effectiveness of the proposed method.

Q2. Can the method handle nested SQL, CTEs, or window functions? Spider/Bird are relatively simple. It is not clear whether the proposed method can scale to more complex schema and SQLs. Spider 2 might be a good candidate for such evaluation.

Q3. Does AST perturbation ever produce false negatives? How do you mitigate over-fitting to synthetic mistakes?

Q4. How would the model perform if integrated into self-refinement/self-correction loops? How does HEROSQL compare to more careful designed Text-to-SQL systems such as DIN-SQL, or simple LLM-as-a-judge methods?

### Soundness
3

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
4

### Summary
The paper proposes HEROSQL, a hierarchical representation and validation framework for Text-to-SQL that bridges global intent and local structural details. Global intent is captured via query Logical Plans (LPs), while local syntactic structure is captured via Abstract Syntax Trees (ASTs). A Nested Message Passing Neural Network (NMPNN) first aggregates within each sub-SQL’s AST and then across the LP graph to encode the full query. A context-guided embedding step conditions attribute text on database schema and predicted SQL. An AST-driven sub-SQL augmentation strategy generates semantically incorrect yet syntactically valid negatives by rule-based perturbations, enabling finer-grained supervision. Experiments on BIRD, Spider and EHRSQL show consistent gains over text/graph baselines in AUPRC/AUROC, with ablations validating each component. Case studies demonstrate fine-grained error localization and actionable feedback to LLMs.

### Strengths
S1. Novel combination of LP-level global semantics and AST-level local structure for semantic validation, with a principled NMPNN over the hierarchy.
S2. AST-driven negative augmentation specifically for validation (vs. generation) is useful.
S3. Strong empirical results across diverse datasets and a useful demonstration of fine-grained feedback to LLMs.

### Weaknesses
W1. Limited justification for decoder-only embeddings vs. strong encoder baselines. Encoder-only models (e.g., E5, GTE, BGE, Contriever) are strong baselines for text embedding; decoder-only choices increase compute and may not help validation. 
W2.Reproducibility gaps (placeholders; limited detail on LP extraction). Placeholders (runs, stdevs) and missing versions reduce confidence and hinder replication.
W3. Practical latency/throughput for interactive validation not reported. Validation is often on-the-fly; build and inference time budgets matter.
W4. Error taxonomy is not fully realized (detection/localization only); classification head is future work.

### Questions
1. Add SQLens to related work and position clearly. SQLens integrates DB-execution and LLM signals for clause-level error detection and performs correction end-to-end.

2. How sensitive are results to the LP generator (Calcite vs. ORCA) and optimization settings? Any LP extraction failures and how are they handled?

3. How do decoder-only embeddings compare to encoder-only baselines for your context-guided embedding?

4. What is the end-to-end latency per query, and is it suitable for interactive validation?

### Soundness
3

### Presentation
3

### Contribution
3
