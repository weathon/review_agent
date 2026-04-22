# Faithful Rule Learning for Tabular Data Cell Completion

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 4, 6, 4

## Abstract
Tabular data cell completion aims to infer the correct constants that could fill a missing cell in a table row. While machine learning (ML) models have proven to be effective for this task, the limited interpretability restricts their applicability in trust-critical domains. In this paper, we develop two interpretable ML models to predict whether a candidate constant should fill the empty cell of an incomplete row by learning Datalog rules describing chain-like patterns of relations. Both models are *fully interpretable with formal guarantees*: we provide algorithms that take a model instance and extract an *equivalent* set of rules, in the sense that both the model and the rules produce the same output for any input table over a fixed relation schema. Furthermore, our models utilize different aggregation strategies to offer distinct trade-offs regarding expressive power and ease of rule extraction. Evaluations reveal that our models achieve state-of-the-art performance on tabular data cell completion with superior interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces two novel, interpretable machine learning models, MC (Multi-Chain) and MC-max, for the task of tabular data cell completion, which aims to infer missing values in table rows. Addressing the limited interpretability of existing ML models , this work proposes models that predict missing constants by learning Datalog rules describing chain-like relational patterns. The two models utilize different aggregation strategies—sum aggregation (MC) and max aggregation (MC-max)—offering distinct trade-offs between expressive power and the complexity of rule extraction. The primary contribution is that both models are fully interpretable and include faithful rule extraction algorithms. These algorithms provide formal guarantees that the extracted set of Datalog rules is equivalent to the model, ensuring that the rules produce the exact same output as the model for any input. Evaluations on standard benchmarks show that the models achieve state-of-the-art performance while offering superior interpretability.

### Strengths
1. Strong Formal Guarantees: The paper's most salient strength is its formal commitment to faithfulness. The authors do not merely claim interpretability; they formally prove via Theorems 4.4 and 5.2 that the extracted Datalog rule sets produce the exact same output as the models themselves (MC and MC-max) on any given input.
2. Clear Model Design and Trade-offs: The paper designs two models with clear semantics. The MC model uses sum aggregation, corresponding logically to counting the number of matching paths. The MC-max model uses max aggregation, corresponding to detecting the existence of the strongest path.
3. Excellent Empirical Performance: The experimental section is convincing. The MC and MC-max models achieve SOTA results on multiple key metrics across several tabular and binary relation datasets. They not only surpass specialized GNN baselines but also significantly outperform strong LLM baselines. This demonstrates that the models achieve interpretability without sacrificing predictive performance.

### Weaknesses
1. Scalability of Global Faithfulness for MC Model: Despite the formal guarantee (Theorem 4.4), the global rule extraction algorithm for the MC model (Algorithm 1) has a time complexity of $\mathcal{O}(C^{\delta^{L}\nu^{2L}})$. This complexity is doubly exponential in both the path length $L$ and the maximum table arity $\nu$. This implies that for database schemas of moderate complexity (e.g., slightly more tables or longer dependency chains), extracting the complete, equivalent Datalog program is computationally infeasible. Although database-specific extraction (Algorithm 3) is feasible, this limitation diminishes the practical significance of the MC model's "global faithfulness".
2. Rigid Dependence of Model Architecture on Database Schema: The dimension $K$ of the model's parameter tensor $b^{h,t}$ depends on all predicates (tables) and their arities (column counts) in the database. This means the model architecture is hard-bound to a specific database schema. If a new table (new predicate) is added to the database, or if the arity of an existing table changes, the entire model must be redesigned and retrained. This limits the model's applicability in dynamic database environments.

### Questions
Handling of Multiple Missing Values (Appendix F): The method in Appendix F for handling multiple missing values involves creating a separate auxiliary fact for each missing value and filling other missing positions with a special constant $c_0$. This approach seems to implicitly assume that the missing values are mutually independent. If the missing values are highly correlated (e.g., "City" and "Country" are missing simultaneously), could this independent prediction method lead to inconsistent completion results (e.g., completing as "Paris" and "Germany")?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces two models (MC and MC-max) for tabular data cell completion, with formal guarantees of faithfulness between model predictions and extracted Datalog rules. The paper focus on the challenge of completing missing cells in tables while providing transparent, human-readable explanations. The paper includes evaluations on several benchmarks

### Strengths
1) The proposed MC and MC-max models introduce reasonable approaches to capturing relational patterns in tabular data. 

2) The paper evaluates on multiple standard benchmarks.

3) The paper includes detailed mathematical formulations, algorithm descriptions, and proofs

### Weaknesses
1) The paper suffers from significant readability issues. The mathematical notation is excessively dense, with nested subscripts and complex symbolic expressions that make the core ideas difficult to grasp. Key concepts like path schemas, multi-chain conjunctions, and the faithful extraction process are explained using notation-heavy formulations that obscure the intuitive understanding.

2) While the theoretical proofs are substantial, the paper provides minimal discussion of practical application scenarios. There's insufficient explanation of how these methods would be deployed in real-world data completion tasks, what types of tables benefit most from this approach, or practical limitations.

3)  The paper overlooks important recent developments in table understanding, particularly methods like Chain-of-Table, and other LLM-based table reasoning approaches that have shown strong performance on similar tasks. This omission limits the contextual positioning of the contributions.

### Questions
How about the efficiency Issues in rule extraction? Would the time cost be very high?

More intuitive explanations and visual examples would help to follow the paper.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces interpretable machine learning models for predicting missing cells in tabular data. The authors propose two rule-based frameworks: Multi-Chain (MC) and MC-max, that infer missing values by aggregating paths among table entries, learning Datalog rules that formally mirror model behavior. Both models are fully interpretable and come with theoretical guarantees of faithfulness, ensuring that each prediction can be exactly derived from the extracted rules. MC uses sum aggregation for expressive power, while MC-max employs max aggregation for efficiency and simpler rule extraction. Extensive experiments on multiple benchmark datasets demonstrate that these models achieve state-of-the-art performance while offering transparent, human-readable explanations of their predictions.

### Strengths
1. A theoretically grounded and practically promising approach is proposed to interpretable rule learning for tabular data, which is a valuable contribution.
2. Experiments on multiple benchmarks are propsed to demonstrate the advantages of the prposed method.

### Weaknesses
1. This paper is written in a way that is somewhat difficult to follow, which makes it particularly unfriendly for readers who are not specialists in this area.
2. Comparison with recent LLM-based reasoning models could be expanded beyond few-shot prompting (e.g., reasoning-tuned LLMs or neuro-symbolic hybrids).
3. The paper provides rule examples but no quantitative results; including such results could enhance the paper’s readability and help readers better understand the approach.
4. I am not sure whether this is a major issue, but the font used in the paper appears to differ from that of the official ICLR template.

### Questions
Please see the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper develops two interpretable models (MC and MC-max) for tabular data cell completion that extract Datalog rules formally equivalent to the trained model's behavior. The MC model uses sum aggregation to count weighted paths between constants, while MC-max uses max aggregation, trading expressivity for extraction complexity. Theorems 4.4 and 5.2 prove that extracted rules produce identical predictions to the model on any input, guaranteeing explanation fidelity. Experiments on tabular (WP-IND, JF-IND, MFB-IND, FB-AUTO) and binary datasets show competitive empirical performance against HyperGCN, GMPNN, and LLM baselines, with rule extraction completing in 3-5 minutes. The main contribution is extending faithful rule learning from binary relations (knowledge graphs) to multi-arity relations (tables) with provable extraction algorithms.

### Strengths
-- This work appears to be the first faithful rule learning approach for multi-arity relations with provable extraction algorithms (Theorems 4.4, 5.2). Unlike approximate methods (Neural-LP, DRUM, SHAP), rules exactly replicate model behavior.

--  Max aggregation reduces extraction complexity with formal characterization of expressivity loss (Proposition E.1), addressing computational constraints.

-- Lemma 3.4 provides clear semantic interpretation as weighted path sums; multichain rules elegantly encode cardinality via inequalities; proofs are detailed with proper Datalog grounding.

-- Precise definitions distinguish cell completion from imputation and hypergraph prediction; comprehensive related work properly situates contribution within faithful rule learning and neuro-symbolic AI.

-- Rule extraction appears efficient on evaluation datasets given the reported extraction times; statistical significance testing validates improvements

### Weaknesses
-- Chain-like patterns cannot capture star patterns (simultaneous multi-attribute constraints), cycles, or branching logic—critical for real tables. Paper provides no analysis of what fraction of ground-truth relationships require non-chain patterns or comparison to unrestricted rule learners. Unclear when method is applicable vs. fundamentally limited.

-- The abstract claims "superior interpretability" but provides zero evidence beyond runtime and example rules. No rule complexity metrics, human studies, ground-truth comparison on synthetic data, or analysis of actionable rules. Core claim is unsubstantiated.

-- L=2, likely providing an explanation of extraction efficiency, and limiting practical effectiveness.  

-- Neural-LP and DRUM are directly related but not compared. Both could potentially be adapted to multi-arity. Missing this baseline makes it impossible to evaluate whether formal guarantees provide practical advantages over approximate explanations.

### Questions
-- Can you provide examples of when L=2 rules provide impactful value to an application?  Can you provide L=3 or 4 examples (and analysis)?  What is the path length distribution in your dataset?  What are the practical limits for L?

-- Your LLM baselines show significant precision/recall imbalance suggesting implementation issues:
(a) Can you provide the CoT prompts used?
(b) What were TabLLM validation curves—did training succeed?
(c) Was temperature/top-p tuning attempted for calibration?
(d) Would more than 3-shot improve CoT?

-- Short chain-like patterns seem fundamentally limiting:  What fraction of relationships in your datasets require non-chain patterns (star, cycles, branching)? Can you provide synthetic experiments where ground-truth rules need complex structures?

### Soundness
3

### Presentation
3

### Contribution
2
