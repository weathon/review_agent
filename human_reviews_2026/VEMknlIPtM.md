# TPD-AHD: Textual Preference Differentiation for LLM-Based Automatic Heuristic Design

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
The design of effective heuristics for complex combinatorial optimization problems has traditionally relied on extensive domain expertise and manual effort. While Large Language Model-based Automated Heuristic Design (LLM-AHD) offers a promising path toward autonomous heuristic generation, existing methods often suffer from undirected search processes and poor interpretability, resulting in a black-box optimization paradigm. To address these limitations, we introduce Textual Preference Differentiation for Automatic Heuristic Design (TPD-AHD), a novel framework that integrates preference optimization with textual feedback to guide LLM-driven heuristic evolution. TPD-AHD employs a best-anchored strategy to pair heuristic candidates and generates a natural language textual loss. This loss is then translated into a textual gradient, which provides explicit, interpretable instructions for iterative heuristic refinement. This approach not only enhances the transparency of the optimization trajectory but also ensures a directed search toward high-performance regions. Extensive experiments on a suite of NP-hard combinatorial optimization problems demonstrate that TPD-AHD consistently outperforms both manually designed heuristics and existing LLM-AHD methods. Furthermore, it exhibits strong generalization capabilities across diverse domains and provides clear insights into the heuristic improvement process. TPD-AHD establishes a new paradigm for interpretable, efficient, and scalable automatic heuristic design.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This manuscript proposes TPD-AHD, a method to automatically design heuristics.

### Strengths
The proposed TPD-AHD seems to perform better than other LLM-based heuristic-generation methods, but the reasons for this improved performance remain unclear.

### Weaknesses
**Limited innovation:** As the author said, Textual Loss is obtained by "ask the LLM to compare $h_w$ and $h_l$, and explain why $h_w$ is preferred". However, this concept is not novel. Prior studies [1-3] have already proposed a similar approach termed "reflection".

**Ambiguous motivation:**  In light of the first weakness, I remain skeptical of the authors’ claim that TPD-AHD offers better interpretability than other LLM-based heuristic-generation methods. Because TPD-AHD remains a framework that relies exclusively on outputs from large language models.

**Poor reproducibility:**  This manuscript does not provide source code, which undermines the reproducibility of the results.

**Limited experiments:**  The evaluation is restricted to only two LLMs, which limits confidence in the generality of the results.

[1] Reevo: Large language models as hyper-heuristics with reflective evolution, NeurIPS'24.

[2] HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs, AAAI'25.

[3] Efficient heuristics generation for solving combinatorial optimization problems using large language models, KDD'25.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
It introduces a framework, named TPD-AHD, for automatic heuristic design by integrating textual preference differentiation with LLMs. It uses a best-anchored pairing strategy to compare heuristics, generating a textual loss that is converted into a textual gradient to guide iterative heuristic refinement, outperforming existing LLM-AHD methods on many NP-hard problems and practical tasks.

### Strengths
The best-anchored pairing mechanism focuses learning on high-performing candidates, reducing noise and improving convergence.

Outperforms state-of-the-art methods across diverse NP-hard problems and practical tasks.

### Weaknesses
The framework's reliance on prompt engineering for "textual backpropagation" and a correct gradient. It may be misleading and trapped in a local optimum.

The number of iterations for each run is insufficient, and further clarification is required.

### Questions
Given that textual gradients are natural language, how was their consistency and quality objectively assessed to ensure they provide accurate and non-degenerate guidance, beyond just the final heuristic performance?

Are you using one of the gradient information or summing up the gradients from all different N-1 pairs? The gradients might point to different revisions. How to handle these cases? 

How does TPD-AHD perform when the initial prompt or candidate pool is of very low quality? Does the textual gradient mechanism robustly recover, or does it require a "warm start"? Misleading gradient information could arise if initial heuristics are poor.

The study uses only 200 heuristic evaluations per run, whereas common practice in much of the existing literature involves 1,000 evaluations. Is this choice due to the gradient information enabling faster convergence, potentially at the risk of local optima? Would increasing the number of evaluations prevent it from outperforming existing baselines?

Could you detail the computational cost (in terms of total LLM tokens and wall-clock time) for each component of TPD-AHD?

In Figure 2, the starting points for different methods vary. Why is this the case? It would be beneficial to compare convergence by using the same initial heuristics across all methods. The proposed method has surprisingly good performance with very low variance. What is the reason? 

The ablation study demonstrates performance drops when components are removed on TSP. Did any ablated variants ever outperform the full model in specific scenarios, potentially indicating over-complexity in the full TPD-AHD approach for those cases?

I will raise the score if concerns are appropriately addressed.

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
4

### Summary
The paper introduces TPD-AHD, a novel framework for LLM-based automatic heuristic design that leverages “textual preference differentiation”: LLMs generate and compare pairs of heuristics, outputting interpretable textual feedback and improvement instructions (“textual gradients”). This aims to address the current inefficiencies and opacity of LLM-based algorithm design by making the optimization process more transparent, iterative, and directed. Empirical results on a suite of classic NP-hard combinatorial optimization problems and diverse real-world tasks show that TPD-AHD consistently outperforms prior LLM-AHD methods and provides better interpretability, with robust ablation and parameter sensitivity studies to support the claims.

### Strengths
1. This paper is well-written and the adopted methods are quite novel in this domain..

2. This paper adopts a wide collection of classic combinatorial benchmarks (TSP, CVRP, JSSP, MKP, etc.) and practical tasks (control and scientific discovery), and all recent state-of-the-art LLM-AHD methods are compared (Funsearch, ReEvo, EoH, MCTS-AHD). Results are persuable.

### Weaknesses
1.  The contribution is a little bit limited, mostly applying TEXTGRAD to LLM-based AHD.

2. The total process will make more LLM calls, which can be more expensive.

### Questions
1. Could you please make a comparison of the LLM cost of algorithm searches with baselines? I am very willing to improve the rating of this paper if you add this part.

2. I have a little doubt about the motivation. ReEvo provides a reflection step, and MCTS-AHD provides a tree structure. Why do you believe the interpreterbility can be improved when you only have both reflection and trajectory? Quote "existing methods often suffer from undirected search processes and poor interpretability, resulting in a black-box optimization paradigm"

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a framework called ``TPD-AHD``, which incorporates a ``textual preference differentiation`` mechanism into the field of ``LLM-AHD``, and conducts experiments on various combinatorial optimization problems.

### Strengths
This paper is clearly written, the figures are well-crafted, and the comparison methods are comprehensive.

### Weaknesses
1. Despite the introduction of the textual gradient, it remains entirely dependent on the LLM, offers no rigorous convergence guarantees, and lacks stability.

2. As shown by the ``select_next_node`` in the appendix, the heuristic still boils down to a localized, fixed-rule-based random-greedy selection loop; the overall algorithm performs no graph-structure learning nor makes use of any global information.

3. Taking the ``TSP`` as an example, the results in Table 1 are even worse than the ``random-insertion`` algorithm proposed by GLOP [1], another step-by-step construction framework.

4. Regarding the choice of baselines and datasets. For the ``CVRP`` task, it is recommended to use HGS as a baseline. Both ``LKH`` and ``HGS`` are not exact algorithms in the strict sense; referring to their results as ``optimal`` is therefore imprecise. In terms of datasets, the paper lacks real-world benchmarks such as ``TSPLIB`` and ``VRPLIB``.

5. The method’s potential is likely bounded by both the initial prompt and the underlying constructive template; the authors need provide deeper studies on these limitations.

[1] *GLOP: Learning Global Partition and Local Construction for Solving Large-scale Routing Problems in Real-time, AAAI 2024*

### Questions
See ``Weakness``

### Soundness
1

### Presentation
3

### Contribution
1
