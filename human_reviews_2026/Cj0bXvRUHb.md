# Alpha Discovery via Grammar-Guided Learning and Search

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 6

## Abstract
Automatically discovering formulaic alpha factors is a central problem in quantitative finance. Existing methods often ignore syntactic and semantic constraints, relying on exhaustive search over a unbounded and unstructured space that limits performance and interpretability. 
We present AlphaCFG, the first framework for defining and discovering alpha factors that are syntactically valid, financially interpretable, and computationally efficient. First, AlphaCFG defines an alpha-oriented Context-Free Grammar to construct a tree-structured, size-controlled search space of human-interpretable alpha expressions, enabling grammar-tailored search and learning.  We then formulate the search of high-performance alphas in this space as a very large, tree-structured linguistic Markov Decision Process (TSL-MDP), where each state is an alpha expression with its information coefficient as reward.  To efficiently navigate the TSL-MDP, we develop syntax-similarity-based representation learning method to estimate alpha expression performance (value network) and grammar production rule probabilities (policy network), and integrate it into a grammar-aware Monte Carlo Tree Search. Experiments on China and US stock markets' datasets show that AlphaCFG outperforms state-of-the-art baselines in both search efficiency and trading profitability. AlphaCFG also provides an easy-to-use approach for refining and improving existing formulaic alpha factors.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes AlphaCFG, a grammar-guided framework for automated formulaic alpha discovery in quantitative finance. By introducing a context-free grammar with financial semantics, the authors define a structured and interpretable search space for alpha expressions and formulate the discovery task as a tree-structured linguistic Markov decision process. A grammar-aware Monte Carlo Tree Search with Tree-LSTM-based representation learning jointly learns the policy and value functions to efficiently explore this space. Experiments on CSI300 and S&P500 datasets show that AlphaCFG achieves higher information coefficients than existing reinforcement learning and machine learning baselines.

### Strengths
1. The paper introduces Context-Free Grammar (CFG) into the automatic discovery of alpha factors for the first time, combining linguistic grammar generation principles with reinforcement learning and Monte Carlo Tree Search (MCTS) to propose a unified framework with both theoretical innovation and practical significance.
2. By defining syntactic and semantic constraints through CFG, the generated alpha factors exhibit clear financial meaning and structural readability, significantly enhancing the interpretability and controllability of the model.
3. The effectiveness of the proposed algorithm is validated on both the Chinese and U.S. stock markets (CSI300 and S&P500), where it consistently outperforms comparative methods in IC, ICIR, and Sharpe Ratio, achieving higher cumulative returns.
4. The overall design is methodologically coherent, with a clear theoretical and implementation path spanning grammar definition, semantic constraints, length control, and reinforcement learning–based search, demonstrating strong research rigor and internal consistency.
5. The supplementary materials provide a well-organized code structure, which facilitates reproducibility of the results.

### Weaknesses
1. The problem formulation of alpha discovery could be further improved, as the current description may not provide sufficient clarity for readers who are less familiar with this domain. A more detailed and accessible problem definition would enhance the paper’s readability and impact.
2. The experiments involve numerous hyperparameters, whose tuning is inherently complex and requires extensive search to achieve optimal settings.
3. Both the grammar and reward designs are fixed and manually specified, which limits AlphaCFG’s adaptability to different market conditions and introduces additional human intervention. These components represent extra “dirty work” that undermines the framework’s claimed level of automation.
4. The comparison mainly includes AlphaGen and AlphaQCM as recent baselines, while the others are relatively outdated. The lack of comparison with more recent approaches may underestimate current state-of-the-art performance. In addition, the reference for *gplearn* is missing.

### Questions
1. The combination of Grammar-aware MCTS and Tree-LSTM appears computationally expensive when applied to large-scale market features, but the paper does not provide sufficient discussion of the training cost or scalability. Are there any intuitive quantitative comparisons between AlphaCFG and other baselines in terms of efficiency or computational resources?
2. It is not clear whether the defined α-CFG-Sem-k grammar rules can comprehensively represent the common types of alpha constructions in the financial domain, and there may be potential omissions or biases toward certain categories of factors.
3. The reported results show zero variance for XGBoost, which seems implausible and raises concerns about the experimental setup or reporting. Was this due to a single run or to improper averaging.
4. Figure 7 lacks any explanation of experimental reproducibility since the paper does not report the variance or fluctuation of results. Was each experiment executed only once to obtain the presented outcomes.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces AlphaCFG, a grammar-constrained framework for discovering alpha factors in quantitative finance. The method defines a domain-aware context-free grammar (α-CFG-Sem-k) to generate syntactically and semantically valid formulaic factors within a bounded search space. Alpha generation is formulated as a Tree-Structured Linguistic MDP (TSL-MDP), where leaf rewards correspond to the factor’s Information Coefficient (IC). The MDP is solved using a grammar-aware Monte Carlo Tree Search (MCTS) guided by Tree-LSTM–based policy and value heads that evaluate partial expressions. The authors further include a similarity regularizer based on the largest common subtree to promote diversity among discovered alphas. Experiments on CSI300 and S&P500 show that AlphaCFG achieves higher IC and backtest profitability than several baselines (AlphaGen, AlphaQCM, GPlearn, XGBoost), and it can refine existing factors such as GTJA191 and Alpha101.

### Strengths
- The use of a context-free grammar (CFG) grounded in financial semantics effectively constrains the search space, ensuring valid and interpretable formulas while reducing redundancy compared to unrestricted symbolic approaches. Modeling formula generation as a Tree-Structured Linguistic MDP (TSL-MDP) offers a clean theoretical lens for combining RL with symbolic expression search, justifying the use of MCTS and policy/value learning.
- Using Tree-LSTM–based policy and value estimators enables partial expression evaluation and improves search efficiency. The subtree-based similarity regularization is also a thoughtful mechanism to enhance formula diversity.
- The experiments benchmark AlphaCFG against a broad set of symbolic, ML, and RL-based methods across two major indices, with multiple metrics reported (IC, Sharpe, MaxDD). The demonstration on refining existing alphas adds practical relevance.

### Weaknesses
- Semantic equivalence pruning lacks rigor and scalability analysis. The paper uses subtree-based similarity for pruning equivalent or redundant expressions but does not analyze its computational complexity or describe practical optimizations (e.g., hashing or canonicalization). Given the large candidate pool, this omission is a serious limitation.
- Restrictive grammar and operator set. The experiments use only six primitive features and a small operator vocabulary, which raises concerns about overfitting and limited expressivity. The paper does not demonstrate scalability when introducing richer operators or additional financial features.
- Reward design misaligned with trading objectives. Optimizing solely for IC ignores turnover and transaction costs. Although backtests show strong returns, these metrics are not optimized directly, leaving doubts about robustness under realistic frictions.
- Weak ablations and architectural justification. The choice of Tree-LSTM is not well motivated relative to other plausible encoders (e.g., GNNs or transformers). Ablations on encoder type, embedding size, and similarity penalty strength are missing, weakening the causal claims about representation quality.

### Questions
- How is the semantic similarity function computed efficiently at scale? What is its time complexity, and are approximate or hashing-based methods used?
- What are the compute resources (GPUs, CPUs, wallclock time) and total number of IC evaluations used for AlphaCFG versus AlphaGen/AlphaQCM?
- How sensitive are results to the choice of operators and constants? Would adding common financial primitives (e.g., liquidity or volatility) change outcomes?
- Were results averaged across multiple seeds, and what is the variance? Please report statistical tests versus the strongest baselines.
- In the factor-refinement experiments, how much modification occurs to the original factor, and how do you ensure these changes are not trivial rewrites that merely inflate IC?


I would consider raising my score if the authors can adequately address these questions.

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
This paper introduces AlphaCFG, a framework for automated discovery of interpretable, high-performing formulaic alpha factors using a context-free grammar for syntactic/semantic constraints, coupled with a reinforcement learning-guided MCTS. AlphaCFG explicitly formalizes the syntax and semantics of valid financial expressions, bounds search complexity via grammar and expression length, and employs Tree-LSTM–based neural networks as value and policy estimators within MCTS. The approach is empirically evaluated on China (CSI 300) and US (S&P 500) equities, further showcases the framework’s ability to refine classic domain factors.

### Strengths
1.	Principled, Structured Search via Grammar: By leveraging domain-specific rules—such as operator arity, finance logic, and context-dependent operand constraints, the paper addresses a central limitation in prior factor mining efforts: managing the combinatorial explosion and redundancy of candidate spaces.
2.	Integration of Reinforcement Learning and Neural MCTS: The formulation of alpha search as a Tree-Structured Linguistic MDP and the embedding of neural MCTS with Tree-LSTM–based networks offer a methodologically coherent and scalable solution for symbolic expression discovery.
3.	Empirical Rigor Across Multiple Markets and Metrics: The paper provides comprehensive head-to-head comparisons with existing baselines. The results, across correlation metrics (IC, RankIC) and trading metrics, consistently favor the proposed approach, and the ablation studies convincingly validate the role of syntax, semantics, and length controls.
4.	Interpretability and Factor Refinement: The capacity to refine existing domain factors shows the practical relevance and transferability of the framework, highlighting the potential for human-in-the-loop discovery and sector adoption.

### Weaknesses
1.	The paper lacks a quantitative report (e.g., a 1% syntax error rate) to conclusively prove that all generated expressions are syntactically valid
2.	The paper extends the grammar to α-CFG-Sem (Definition 3) by incorporating domain-specific semantic constraints. The designed constraints are heuristically sound and based on reasonable financial logic. However, the connection between these specific constraints and the formal property of "interpretability" is not rigorously proven. 
3.	Although Sharpe and Max Drawdown are computed, there is no investigation into the impact of these real-world trading constraints or their explicit inclusion within the search objective/loss. 
4.	Scope and Generalization: The framework’s evaluation is limited to two well-established benchmarks (CSI 300, S&P 500). There is no evidence or discussion concerning the transferability to other financial datasets, time periods, asset classes, or tasks outside formulaic alpha mining—whereas some prior works position their language discovery techniques as generally applicable beyond their primary domain.
5.	Figure 6 demonstrates faster convergence for CFG methods compared to RPN, suggesting improved sample efficiency. However, the paper lacks direct evidence of computational efficiencyNo comparison of wall-clock time or CPU/GPU hours against baselines, No analysis of memory usage. No theoretical complexity analysis relating to the size of the grammar-defined search space.

### Questions
1.	Could the authors provide examples of trading strategies (alphas) with clear financial intuition, along with an analysis of their performance during specific market regimes?
2.	Could the authors clarify (with ablation or added experiments) the practical impact of the maximum length and semantic constraints? Are some informative or profitable alphas systematically excluded by the grammar?
3.	How robust is the AlphaCFG approach to changes in the underlying data regime (e.g., bull vs. bear periods, out-of-distribution price actions)? Can the authors provide empirical results/data splits that systematically probe regime sensitivity?

### Soundness
2

### Presentation
2

### Contribution
3
