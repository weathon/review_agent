# Logic-Logit: A Logic-Based Approach to Choice Modeling

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 5, 5, 6, 6

## Abstract
In this study, we propose a novel rule-based interpretable choice model, {\bf Logic-Logit}, designed to effectively learn and explain human choices. Choice models have been widely applied across various domains—such as commercial demand forecasting, recommendation systems, and consumer behavior analysis—typically categorized as parametric, nonparametric, or deep network-based. While recent innovations have favored neural network approaches for their computational power, these flexible models often involve large parameter sets and lack interpretability, limiting their effectiveness in contexts where transparency is essential.

Previous empirical evidence shows that individuals usually use {\it heuristic decision rules} to form their consideration sets, from which they then choose. These rules are often represented as {\it disjunctions of conjunctions} (i.e., OR-of-ANDs). These rules-driven, {\it consider-then-choose} decision processes enable people to quickly screen numerous alternatives while reducing cognitive and search costs. Motivated by this insight, our approach leverages logic rules to elucidate human choices, providing a fresh perspective on preference modeling. We introduce a unique combination of column generation techniques and the Frank-Wolfe algorithm to facilitate efficient rule extraction for preference modeling—a process recognized as NP-hard. Our empirical evaluation, conducted on both synthetic datasets and real-world data from commercial and healthcare domains, demonstrates that Logic-Logit significantly outperforms baseline models in terms of interpretability and accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces the Logic-Logit, an interpretable, rule-based choice model, along with a learning pipeline built on column generation techniques and the Frank-Wolfe algorithm. The authors also provide numerical results demonstrating the model's interpretability and predictive accuracy.

### Strengths
The paper is generally well-written and presents a new explainable choice model with a learning framework. Numerical results indicate strong interpretability and predictive performance, tested on both synthetic and real-world datasets.

### Weaknesses
[a]There is no discussion regarding the model's capacity. It is unclear whether this model falls under the mixed-logit model or if it is equivalent to a mixed-logit model. If so, why not use the approach in [1] to learn a mixed-logit, given its stronger analytical properties, such as provable convergence guarantees? (Also, the mixed-logit's identified customer segments and the parameters in each segment could also be viewed as the interpretation of the choice?)

[b]I'm confused about if the learning requires prior knowledge of $M$ and $\{p_m\}$. For instance, Line 177 mentions, “These mappings, from real-valued features to predicates, are predefined and fixed,” and the learning problem (3) does not appear to learn $\{p_m\}$. Yet, in the experiments, the algorithm identifies these rules. This raises questions about the rule search strategies outlined in lines 308-323, and a pseudo-code would help clarify this process.

[c]The literature review lacks a discussion of consider-then-choose models. Since the proposed model follows this approach, it would be beneficial to compare it with other models in this category, such as [2] and [3].

Minor comments:

[a]Given that interpretability is a key advantage of the proposed model, it would be helpful to compare it with other interpretable choice models, such as [4].

[b]Including the variance/std in the experimental results would provide additional insight.

[c]Some potential typos: line 192 should be “item $s$ ” instead of “item $j$”, and also “offer set $S$” instead of “offer set $S_t$”.


References

[1]Hu, Yiqun, David Simchi-Levi, and Zhenzhen Yan. "Learning mixed multinomial logits with provable guarantees." Advances in Neural Information Processing Systems 35 (2022): 9447-9459.

[2]Liu, Qing, and Neeraj Arora. "Efficient choice designs for a consider-then-choose model." Marketing Science 30.2 (2011): 321-338.

[3]Akchen, Yi-Chun, and Dmitry Mitrofanov. "Consider or Choose? The Role and Power of Consideration Sets." arXiv e-prints (2023): arXiv-2302.

[4]Tomlinson, Kiran, and Austin R. Benson. "Learning interpretable feature context effects in discrete choice." Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining. 2021.

### Questions
See weaknesses [a] [b].

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper addresses the development of choice models by constructing a rule-based approach using logical statements that dissect conjunctions. The proposed method employs a dual-optimization process: an outer optimization for determining preference weights, and an inner optimization to identify new rules. The model iteratively refines the rule set by incorporating each newly discovered rule. Assuming convex optimization, the authors apply the Frank-Wolfe algorithm to update preference weights across all rule types. Experiments on a synthetic dataset and the Expedia Hotel Dataset demonstrate that the proposed method consistently outperforms all baseline models.

### Strengths
- The authors tackle a crucial question in designing interpretable and explainable models.
- They present an interpretable algorithm tailored for choice modeling.
- The method is intuitive and demonstrates strong performance.
- The proposed approach outperforms baseline methods across 2 datasets.

### Weaknesses
- The method does not appear scalable to a large number of features (predicates). What is the computational complexity of this approach?
  
- The experimental details, including hyperparameters (e.g., exact convergence condition used) for the proposed algorithm, are missing and were not found in the Appendix. Significant methodological details are lacking.

- Although logic rules enhance interpretability, they may not be applicable for designing all types of features.

### Questions
- Item $j$ and $S_t$ are introduced but not used in Equation (1), which creates some confusion.
- If $p_m$ returns 1 when a condition is met and $x_s$ is a single item, does that mean the resulting conjunctions reduce to just $x_s$?
- The implementation details of BFS and DFS in the algorithm are unclear, as no specifics or pseudocode are provided. Pseudocode of the algorithm would be helpful with all conditions.
- The notation $\mathcal{X}$ is used but not defined.
- Some figures (1,2) are included but not referenced within the text.
- Could you clarify the rationale behind the rule distance metric? How would the distance between two unrelated features be interpreted?
- Why do the neural network (NN) models lack product features?
- Why aren’t there results for NN-based models on the synthetic dataset?
- The paper does not provide details on how the baseline methods were trained. What are their hyperparameters and how were they trained?
- Other discrete choice models, such as graph-based approaches [1], mixed logit models, and network formation models [2], are not included for comparison, which might help comparing tradeoffs between methods.

[1] Tomlinson, Kiran, and Austin R. Benson. "Graph-based methods for discrete choice." Network Science 12.1 (2024): 21-40.

[2] Gupta, Harsh, and Mason A. Porter. "Mixed logit models and network formation." Journal of Complex Networks 10.6 (2022): cnac045.

### Soundness
3

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
This paper introduces an approach to modeling human choice using OR-of-ANDs logic rules. The authors illustrate how any preference can be transformed into a Boolean logic formula, with rules mapped to Boolean space and interconnected through AND and OR operators. This formula is then incorporated into a mixed logit choice model, enabling preference learning. However, this approach results in an infinite-dimensional optimization problem, a major computational challenge. To address this, the authors apply the functional conditional gradient method (Frank-Wolfe) to reduce the optimization’s complexity. Additionally, due to the exponential size of the search space ($2^M - 1$, where $M$ is the number of rules), they use a column generation technique that incrementally expands the search space by adding new rules in each step. Empirical experiments on both synthetic and real-world datasets (from commercial and healthcare domains) highlight the effectiveness and versatility of this approach.

### Strengths
1. The paper is well-organized and motivated. Its focus on interpretable human preferences is valuable for understanding human decision-making, particularly in high-stakes areas like healthcare and autonomous driving, where trust and transparency are essential.
2. The approach to modeling human choice could also inspire advancements in fields like automated reasoning (the task authors have verified), where a clear understanding of decision-making processes is crucial.

### Weaknesses
1. While innovative, the current solution appears complex, particularly due to the combined use of the functional conditional gradient method and column generation. This complexity may limit its applicability or make implementation challenging for practitioners. A more streamlined or efficient approach could enhance the method’s usability across a wider range of real-world applications.
2. The search space of combinatorial rules is increased exponentially number of rules $M$, which needs to be further reduced to accelerate the learning process.

### Questions
1. Could you estimate the computational cost of the proposed methods, particularly for real-world datasets? 
2. Are there specific conditions or data characteristics under which this model is computationally more efficient or challenging?

### Soundness
3

### Presentation
3

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
This paper presents Logic-Logit, a rule-based interpretable choice model that utilizes logical rules to predict human choices in contexts like healthcare and commercial domains. The authors aim to address limitations in interpretability associated with existing neural network-based models by proposing a model that represents choices through OR-of-ANDs logic rules. These rules enable compact and interpretable representation of human decision-making. The paper introduces an optimization framework using the Frank-Wolfe algorithm combined with column generation to efficiently extract rules, showcasing empirical success in interpretability and accuracy across synthetic, commercial (Expedia Hotel), and healthcare (MIMIC-IV) datasets.

### Strengths
- The model’s combination of interpretable rule-based choice modeling with optimization algorithms is innovative. The approach’s focus on interpretable, structured rule extraction addresses a significant gap in choice modeling literature, especially relevant for high-stakes domains.
- The experimental setup is comprehensive, covering synthetic and real-world datasets. Benchmarks with traditional models and neural networks underscore the model’s effectiveness in balancing accuracy and interpretability. The rule extraction and optimization process is detailed and thoughtfully developed, providing clarity on algorithmic decisions.
- The paper is generally well-organized. The presentation of the OR-of-ANDs rule structure, Frank-Wolfe algorithm, and column generation steps is clear, supporting reproducibility. The inclusion of rule explanations on real-world datasets aids in understanding practical implications.

### Weaknesses
- While the column generation and rule pruning strategies manage computational demands, further discussion on the model’s scalability with significantly larger datasets would enhance the paper. For instance, scalability tests on larger commercial datasets would demonstrate practical feasibility in data-intensive domains.
- The approach involves selecting parameters like the number of rules, rule lengths, and pruning thresholds. More empirical insights into the sensitivity of these parameters on model performance, especially in healthcare contexts, could strengthen robustness claims.
- While the model shows good accuracy on average, there is less discussion about edge cases, where rule-based logic might oversimplify complex decision boundaries. Addressing potential limitations in handling such cases would provide a more balanced view of the model’s capabilities.

### Questions
- Can the authors provide more insight into how the model scales with an increase in the number of features or the size of the dataset? Would any adjustments be necessary in the Frank-Wolfe and column generation steps?
- How does the model handle cases where decision criteria overlap significantly between customer types? For instance, if preferences are highly correlated between types, does the model tend to overfit to certain rules or ignore relevant variation?
- Would incorporating neural components in conjunction with logic-based rules (e.g., neural embeddings for complex feature spaces) enhance performance without compromising interpretability? This hybrid approach could be of interest if straightforward rule-based methods struggle with nuanced distinctions in large feature sets.
- How can this method be applied for RLHF, can authors provide experimental results to demonstrate this? i.e., Finetuing an LLM for a Diffusion models.

### Soundness
3

### Presentation
3

### Contribution
3
