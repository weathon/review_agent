# Rethinking Code Similarity for Automated Algorithm Design with LLMs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
The rise of Large Language Model-based Automated Algorithm Design (LLM-AAD) has transformed algorithm development by autonomously generating code implementations of expert-level algorithms. Unlike traditional expert-driven algorithm development, in the LLM-AAD paradigm, the algorithm's ideas are often implicitly embedded in the generated code. Therefore, assessing algorithmic similarity directly from code, distinguishing genuine algorithmic innovation from mere syntactic variation, becomes essential. While code similarity metrics exist, they fail to capture algorithmic similarity, as they focus on surface-level syntax or output equivalence rather than problem-solving behavior.

We propose BehaveSim, a novel method to measure algorithmic similarity through the lens of problem-solving trajectories (PSTrajs)—sequences of intermediate solutions produced during execution. By quantifying the alignment between PSTrajs using dynamic time warping (DTW), BehaveSim distinguishes algorithms with divergent logic despite syntactic or output-level similarities. We demonstrate its utility in two key applications: (i) Enhancing LLM-AAD: Integrating BehaveSim into existing LLM-AAD frameworks (e.g., FunSearch, EoH) promotes behavioral diversity, significantly improving performance on three AAD tasks. (ii) Algorithm analysis: BehaveSim clusters generated algorithms by behavior, enabling systematic analysis of problem-solving strategies—a crucial tool for the growing ecosystem of AI-generated algorithms. Data and code of this work are open-sourced at https://github.com/RayZhhh/behavesim.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper notes existing code similarity metrics (e.g., CodeBLEU) fail to reflect algorithm similarity in LLM-AAD. It proposes BehaveSim, which measures similarity via algorithms’ problem-solving trajectories. Using DTW to align trajectories and normalize distances, BehaveSim outperforms traditional metrics in validation. It enhances LLM-AAD performance and aids algorithm analysis, with limitations on non-iterative algorithms.

### Strengths
The perspective of the paper is quite novel, as it calculates similarity based on the problem-solving trajectories of algorithms. The paper argues that traditional code similarity metrics fail to reflect true algorithm similarity, so it proposes BehaveSim, which quantifies behavioral similarity by analyzing sequences of intermediate solutions (trajectories) generated during algorithm execution.

### Weaknesses
The proposed behavioral similarity metric, is only designed for iterative algorithms (e.g., sorting, optimization algorithms) and cannot be directly applied to other types of algorithms like machine learning models.

### Questions
1. Regarding the differences in trajectory lengths among different iterative algorithms, what methods are adopted in the paper to avoid biases in BehaveSim's similarity calculation? 

2. How is the hyperparameter p_s1 (probability of inter-island selection) tuned in the paper, and what empirical evidence supports that p_s1=0.5 is the optimal value for balancing exploration and exploitation?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces BehaveSim, a novel similarity metric designed to measure algorithm similarity from a behavioral perspective, rather than code-level or output-level similarity. The authors argue that existing metrics (token-, AST-, embedding-, or execution-based) fail to capture the problem-solving behavior of algorithms—especially in the context of LLM-based Automated Algorithm Design (LLM-AAD), where generated code can differ syntactically yet implement equivalent ideas.
BehaveSim represents each algorithm as a problem-solving trajectory, i.e., a sequence of intermediate solutions generated during execution. The similarity between two algorithms is then defined as the resemblance between their trajectories, computed via Dynamic Time Warping (DTW). Experiments show that BehaveSim better differentiates between algorithms that have similar code but distinct behavior (e.g., BFS vs DFS, insertion sort vs bubble sort), and vice versa.
The paper further demonstrates two applications: (1) improving behavioral diversity in LLM-AAD search (enhancing FunSearch performance on ASP and TSP tasks), and (2) clustering algorithm behaviors for interpretability and discovery.

### Strengths
Novel Perspective:
The paper identifies a clear conceptual gap between code similarity and algorithmic behavior similarity, proposing an elegant behavioral abstraction based on execution trajectories. This reframing is insightful and well-motivated in the context of LLM-generated algorithms.

Concrete Implementation (BehaveSim):
The definition of behavioral trajectories and the use of DTW distance provide a simple yet powerful operationalization of behavioral similarity. The methodology is well formalized, reproducible, and extensible.

Comprehensive Benchmark:
The authors curate a systematic dataset with four categories (Type-1 to Type-4) decoupling code-, behavior-, and result-level similarities, offering a rigorous evaluation against existing metrics such as CodeBLEU, CodeBERTScore, and execution-based scores.

Strong Empirical Results:
BehaveSim achieves intuitive and consistent performance across all dataset types (e.g., correctly scoring 1.0 for Type-3 pairs with equivalent behaviors). Integration with FunSearch also improves both convergence and final performance on ASP and TSP benchmarks, validating practical relevance.

Interpretability and Analysis:
The algorithm clustering experiment (Fig. 5) compellingly illustrates how BehaveSim distinguishes semantically similar but syntactically different implementations, supporting new interpretability avenues in algorithm discovery.

### Weaknesses
Scope Limitation:
BehaveSim applies only to iterative algorithms producing discrete trajectories. Many LLM-generated algorithms, including stochastic, differentiable, or recursive paradigms, are excluded. This significantly restricts generality.

Metric Design Choices:
The use of DTW on normalized edit or Euclidean distances is heuristic; there’s limited justification for why DTW best captures “behavioral similarity.” Ablation on alternative measures (ERP, cosine, etc.) is included but not theoretically grounded.

Dependence on Trajectory Definition:
Defining what constitutes a “partial solution” may require manual instrumentation of each algorithm, limiting scalability and automation for arbitrary code. This makes BehaveSim less plug-and-play for general LLM evaluation pipelines.

Comparative Baseline Gaps:
Although the work compares to standard code metrics, it lacks comparison to semantic or dynamic program analysis methods (e.g., symbolic execution traces, graph-based semantic embeddings). These could offer a fairer baseline.

Moderate Empirical Gains:
While improvements in AAD tasks (Table 3) are consistent, they are modest (~10–15% relative gap reduction) and limited to small benchmark scales. Broader tests on complex algorithmic synthesis domains would strengthen impact.

Overclaiming “Novelty” for AAD:
Integrating diversity measures into LLM-AAD (FunSearch + BehaveSim) is conceptually incremental to existing multi-island or MAP-Elite-based diversity frameworks. The true novelty lies more in behavioral similarity than in AAD improvement.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces BehaveSim, a novel similarity metric for measuring algorithm similarity from a behavioral perspective rather than code structure.The paper demonstrates that existing code similarity metrics (token-based, AST-based, embedding-based, execution-based) fail to capture algorithmic behavioral differences. To solve this problem, the paper proposes measuring similarity via problem-solving trajectories - sequences of intermediate solutions generated during algorithm execution, compared using Dynamic Time Warping (DTW).

### Strengths
The curated dataset with 4 algorithm pair types (varying code/behavior/result similarity combinations) provides rigorous validation. The results clearly demonstrate that BehaveSim achieves 1.0 similarity on Type-3 pairs (same behavior, different code) while existing code metrics fail, and correctly identifies behavioral differences where code metrics show high similarity.

### Weaknesses
1. The evaluation methodology does not use any AI models or AI-related methods. BehaveSim is essentially a general algorithm comparison technique based on execution traces and DTW, which appears equally applicable to comparing human-written code. The source of code (LLM-generated versus human-written) seems irrelevant to the core methodology, raising questions about whether this is fundamentally a software engineering contribution rather than an AI/ML contribution suitable for ICLR.
2. The method only applies to iterative algorithms such as sorting, search, and optimization, excluding ML algorithms and non-iterative approaches. This significantly limits the practical applicability and generalizability of the approach.
3. The similarity dataset contains only two AAD tasks, and all code exampels are hand-crafted, well-known algorithms. The experimental scope is very limited. Besides, the paper does not demonstrate whether BehaveSim can distinguish novel, unseen LLM-generated algorithms, which is critical for the claimed contribution to "algorithm discovery." This represents a significant gap between the evaluation (known algorithms) and the application (discovering novel algorithms).

### Questions
See weakness.

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
This paper proposes BehaveSim, a new metric for measuring algorithm similarity from a behavioral perspective rather than focusing on code structure or final outputs. The key idea is to represent an algorithm by its problem-solving trajectory, which records intermediate solutions generated step by step. The similarity between two algorithms is then defined as the resemblance between their trajectories, measured through DTW. The authors demonstrate that BehaveSim can distinguish algorithms that are behaviorally different but structurally similar, and that integrating this measure into LLM-AAD improves search performance by promoting behavioral diversity.

### Strengths
[1] The paper addresses an important and overlooked gap by redefining algorithm similarity from the behavioral viewpoint. This is a clear and original contribution.

[2] The distinction among code-level, behavior-level, and result-level similarity is well presented, and the taxonomy of four types of algorithm pairs is intuitive and pedagogically useful.

[3] The idea of representing algorithm behavior through trajectories and computing DTW-based similarity is theoretically grounded and applicable to both continuous and discrete problems.

### Weaknesses
[1] BehaveSim is currently designed for iterative algorithms only. It cannot yet handle recursive, dynamic programming, or machine-learning-based algorithms. This limits its generality.

[2] Several heuristic parameters, such as trajectory truncation, normalization constants, and distance scaling, are not systematically analyzed. Their influence on stability and reproducibility is unclear.

[3] The benchmark for similarity evaluation mainly includes synthetic or classical algorithm examples. Broader testing on more diverse algorithmic domains would strengthen general claims.

[4] The integration details of BehaveSim into the multi-island FunSearch framework are brief in the main text, making reproduction difficult without reading the appendix.

[5] Some recent semantic or execution-trace-based code similarity methods are discussed only briefly without quantitative comparison.

### Questions
[1] Could the authors elaborate on how BehaveSim would handle non-iterative algorithms or those with stochastic internal states (e.g., randomized search or Monte Carlo methods)? 

[2] Would the DTW-based comparison still be meaningful in these contexts, and if not, how might the behavioral similarity concept be adapted?

### Soundness
3

### Presentation
2

### Contribution
3
