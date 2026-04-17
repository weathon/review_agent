# Beyond the Heatmap: A Rigorous Evaluation of Component Impact in MCTS-Based TSP Solvers

- Decision: Accept (Poster)
- Scores: 2, 6, 2, 10

## Abstract
The ``Heatmap + Monte Carlo Tree Search (MCTS)'' paradigm has recently emerged as a prominent framework for solving the Travelling Salesman Problem (TSP). While considerable effort has been devoted to enhancing heatmap sophistication through advanced learning models, this paper rigorously examines whether this emphasis is justified, critically assessing the relative impact of heatmap complexity versus MCTS configuration. Our extensive empirical analysis across diverse TSP scales, distributions, and benchmarks reveals two pivotal insights: \textbf{1}) The configuration of MCTS strategies significantly influences solution quality, underscoring the importance of meticulous tuning to achieve optimal results and enabling valid comparisons among different heatmap methodologies. \textbf{2}) A rudimentary, parameter-free heatmap based on the intrinsic $k$-nearest neighbor structure of TSP instances, when coupled with an optimally tuned MCTS, can match or surpass the performance of more sophisticated, learned heatmaps, demonstrating robust generalizability on problem scale and distribution shift. To facilitate rigorous and fair evaluations in future research, we introduce a streamlined pipeline for standardized MCTS hyperparameter tuning. Collectively, these findings challenge the prevalent assumption that heatmap complexity is the primary determinant of performance, advocating instead for a balanced integration and comprehensive evaluation of both learning and search components within this paradigm.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper critically evaluates the "Heatmap + Monte Carlo Tree Search" paradigm for solving TSP, challenging the prevailing focus on increasingly complex learned heatmaps. Also, the authors propose a learning-free heatmap baseline (GT-Prior) with shown efficiency and synergy accompanying MCTS.

### Strengths
1. The tuning in search of optimal MCTS configurations and impact study on the respective parameters is valuable.
2. The proposed parameter-free GT-prior is sensible and computationally efficient.
3. The experiments are relatively well-structured.

### Weaknesses
Please correct me if I made mistakes or ignored important statements already addressed in the initial submission.
1. Foremost, there might be a fundamental misposition that the authors put forward in this work regarding the research line of developing more complex heatmap models: heatmaps do not merely serve for the specific MCTS serching, instead, most recent heatmap-related methods inherently embody advancements in backbone design, training schemes, or data representation, etc., aligning major focuses in the broader ML community. So the criteria for assessing a heatmap should probably not be whether it helps MCTS perform better. Rather, increasing consensus has been inclined to testing neural TSP solvers in a "heatmap + greedy" paradigm to evaluate the raw efficacy of neural parts without the results being disguised by post-inference tricks like MCTS, which I personally also deem more reasonable. 
2. The contribution is a bit limited. Though I appreciate the systematic "tuning" of MCTS settings, the grid-search-based evaluations seem more of an engineering practice than some technical innovation. Second, the proposed GT-Prior, though interesting and computationally efficient, is also learning-free and straightforward. So, from a holistic view, the performance reported basically stems from the established MCTS algorithm, leaving the incremental efforts by the authors (conducting parameter search) somewhat simple and limited under the threshold of a top-tier conference.
3. The performance is not sufficiently impressive. The proposed method fails to outperform DIFUSCO on 2 out of 3 benchmarks, while recent literature has proposed much stronger heatmap models than DIFUSCO.
4. Minor issues. The language needs further consideration. "Figure 2 compellingly illustrates...", "directly answering Q1 by unequivocally demonstrating...", and many similar expressions, seem to indicate a slight abuse of adverbs throughout the paper.

### Questions
1. How do you define the "complexity" or "sophistication" of a heatmap? Is it defined by the parameter quantity of neural models that produce the heatmap, or by any mathematical or statistical metrics computed upon individual heatmaps? The authors criticize complex or sophisticated heatmaps but the definition seems obscure. E.g., in Sec 5 the authors say "the prevailing view that increasingly sophisticated heatmap models are the primary drivers of performance in the "Heatmap + MCTS" TSP paradigm." Similar statements do not seem grounded enough.
2. Could you provide comparative results free of intricate search algorithms like MCTS and using a greedy decoder instead, to compare different heatmap baselines including the proposed GT-Prior?
3. What are the results on smaller-sized instances (e.g., 50/100/200)? Do the main conclusions still hold? What about the MCTS parameters and the GT-Prior's performance?
4. Could you report comparisons using more recent heatmap methods, like the successors of DIFUSCO, e.g. Fast-t2t?
5. What is the principle for choosing the specific search space for MCTS configuration instead of a wider or finer-grained range of parameters? The authors stated the settings are "optimally tuned", then how is such optimality guaranteed?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper revisits the widely adopted "Heatmap + Monte Carlo Tree Search (MCTS)" paradigm for large-scale Traveling Salesman Problem (TSP) solvers. It critically analyzes the respective roles of the heatmap and the MCTS search procedure. Contrary to prior trends that emphasize increasingly complex heatmap design, the paper offers a systematic, empirical evaluation. This evaluation demonstrates that the configuration of MCTS—often taken as a fixed backbone—can have as much, if not greater, impact on solution quality as the heatmap itself. The authors propose a parameter-free GT-Prior heatmap based on the empirically observed k-nearest neighbor edge structure of optimal TSP tours. By tuning MCTS hyperparameters via a robust pipeline for each heatmap, they show that this simple baseline matches or outperforms state-of-the-art learning-based and distance-based heatmaps. This holds across scale, distributional shift, and standard benchmarks, challenging current assumptions in the field. The work argues for more balanced and transparent component evaluation in future research. It provides tools and ablation studies to support reproducibility.

### Strengths
- The paper systematically assesses the "Heatmap + MCTS" paradigm for large-scale TSP, isolating and quantifying each component's impact.
- The work challenges a key assumption: that more complex heatmap models always improve TSP solver performance. With a well-tuned baseline, it shows that optimizing MCTS often matters more than increasing heatmap sophistication.
- A parameter-free k-nearest neighbor heatmap (GT-Prior) matches or outperforms complex learned heatmaps when paired with optimized MCTS. It generalizes well to new distributions and larger instances.

### Weaknesses
- Scope: The analysis, experiments, and proposed GT-Prior heatmap are specialized to the Euclidean TSP. It remains unclear whether the insights transfer to other TSP variants (non-Euclidean, with constraints) or different combinatorial optimization problems (e.g., VRP, graph matching).
- Dependency on optimal solutions: GT-Prior construction relies on empirical distributions extracted from near-optimal solutions. In scenarios where such solutions are expensive or unavailable—a typical motivation for using learning-based solvers—how practical is GT-Prior?
- MCTS parameter tuning: While a one-time cost, tuning can be significant for large search spaces or new problem distributions. The paper suggests SMAC3 and other efficient approaches, but more discussion of practical deployment costs would be helpful.
- Incomplete time metrics: Table 1 includes heatmap and MCTS time but omits training time for learning-based models, data preparation, and other one-off costs. This makes it difficult to fully compare runtime and resource requirements across methods.

### Questions
1. Practicality of GT-Prior: For deployment on genuinely new, real-world TSPs where high-quality solutions are not available, how would one construct GT-Prior (since you need optimal/near-optimal tours to compute the empirical k-nearest distribution)? Did you try synthesizing priors from random/greedy solutions as a further baseline?
2. Tuning cost: Can you provide the absolute time and computational resources required for your grid/SMAC3 MCTS hyperparameter search (including how many instances, search depth, etc.), especially for the largest TSP-10k and TSPLIB cases?
3. Have you considered (or could you comment on the prospects for) transfer of either your analysis framework or GT-Prior construction to other vehicle routing or graph-based optimization problems?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper focuses on the classic "heatmap + MCTS" pipeline for solving the Traveling Salesman Problem (TSP). The authors examine the extent to which different MCTS parameter settings affect the solution quality and further perform tuning accordingly. Additionally, they propose an approach named ``GT-PRIOR`` to generate the initial heatmap based on K-nearest neighbors.

### Strengths
1. This paper is written in an accessible manner, offering a clear explanation of the ``MCTS`` method implemented in ``Att-GCN``, and provides a thorough analysis of how each MCTS parameter influences the solution.

2. This paper conducts sufficient generalization tests across different distributions and scales, including experiments on ``TSPLIB``.

### Weaknesses
1. The authors state that ``The underlying assumption is often that heatmap sophistication directly translates to superior solution quality``, yet they provide no experiments to substantiate this claim. They lack analytical experiments to compare the heatmaps produced by different methods, such using greedy strategy.

2. It can be inferred from ``Table 1`` and the sentence ``The Time_Limit for MCTS was set to 0.1 for TSP-500 and TSP-1000, and 0.01 for TSP-10000`` that the authors run MCTS in parallel. However, they never state this explicitly in the table (64 threads for TSP-500/1000 and 2(maybe?) threads for TSP-10000). Moreover, the baseline solvers ``LKH`` and ``Concorde`` are executed in single-thread mode, so the comparison is unfair and likely to mislead readers about the actual efficiency of MCTS. Additionally, previous study [1] suggests that the ``LKH`` and ``Concorde`` figures reported in the table may be outdated or stem from sub-optimal configurations; it is therefore advisable to adopt the updated baseline results.

3. This paper concentrates **solely on the TSP** with ``heatmap+MCTS`` pipeline tailored to it. This approach is hard to extend to richer problems such as the CVRP.

4. In general, as problem size increases, the quality of heatmaps learned by ML methods deteriorates. Previous studiy [2] has shown that heatmaps achieve strong performance on small-scale TSP instances, while this paper does not include experiments on TSP50 or TSP100. 

5. Both ``MCTS`` and ``LKH`` are K-opt algorithms; the former adds a heatmap guidance. In my view, once the instance size exceeds 500, LKH dominates MCTS in both speed and solution quality by a large margin. Hence the authors’ hope that ``heatmap + MCTS`` will ``develop TSP solvers that are not only high-performing but also more robust, efficient, and genuinely impactful`` is questionable.


[1] *COExpander: Adaptive Solution Expansion for Combinatorial Optimization, ICML 2025*

[2] *Unify ML4TSP: Drawing Methodological Principles for TSP and Beyond from Streamlined Design Space of Learning and Search, ICLR 2025*

### Questions
1. See ``Weakness``

2. As the problem size grows, the time spent on the ``Two-Opt`` step inside ``MCTS`` increases sharply. I would like to know what is the performance difference between running the full MCTS and running plain 2-opt alone on TSP-1000 and TSP-10000. This result might reveal whether MCTS actually makes any meaningful difference at these scales.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper presents an investigation of the effect of tuning MCTS hyperparameters
and "simple" heatmaps on the results of TSP solving with heatmaps and MCTS. The
authors describe the shortcomings of the current literature, their experimental
setup including a new method to develop heatmaps, and the results they obtained.

### Strengths
This is a very nice paper that investigates an angle mostly ignored by the
literature. The results nicely support the conclusions the authors come to, and
suggest that research efforts should be focused in a different direction for
more impact. The paper complements the existing literature very nicely.

The proposed GT-Prior is, to the best of my knowledge, novel and seems to work
very well in practice. It would be interesting to investigate to what extent it
differs from heatmaps learned in other ways; in particular whether learned
heatmaps "converge" towards the GT-Prior heatmap. It would be great if the
authors could comment on this.

The time_limit hyperparameter for MCTS should be explained in the main paper,
not just in the appendix. It is mentioned as being set to specific values on
page 6 without having been introduced before, which is confusing (especially as
the values are counter-intuitive).

### Weaknesses
None major.

### Questions
How does the GT-Prior heatmap compare to learned heatmaps?

### Soundness
4

### Presentation
3

### Contribution
4
