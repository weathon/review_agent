# EVA-MILP: Towards Standardized Evaluation of MILP Instance Generation

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Mixed-Integer Linear Programming (MILP) is fundamental to solving complex decision-making problems. The proliferation of MILP instance generation methods, driven by machine learning's demand for diverse optimization datasets and the limitations of static benchmarks, has significantly outpaced standardized evaluation techniques. Consequently, assessing the fidelity and utility of synthetic MILP instances remains a critical, multifaceted challenge. This paper introduces a comprehensive benchmark framework designed for the systematic and objective evaluation of MILP instance generation methods. Our framework provides a unified and extensible methodology, assessing instance quality across crucial dimensions: mathematical validity, structural similarity, computational hardness, and utility in downstream machine learning tasks. A key innovation is its in-depth analysis of solver-internal features --particularly by comparing distributions of key solver outputs including root node gap, heuristic success rates, and cut plane usage -- leveraging the solver's dynamic solution behavior as an `expert assessment' to reveal nuanced computational resemblances. By offering a structured approach with clearly defined solver-independent and solver-dependent metrics, our benchmark aims to facilitate robust comparisons among diverse generation techniques, spur the development of higher-quality instance generators, and ultimately enhance the reliability of research reliant on synthetic MILP data. The framework's effectiveness in systematically comparing the fidelity of instance sets is demonstrated using contemporary generative models. The code is available in \url{https://github.com/iclr2026evamilp/EVA-MILP}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents EVA-MILP, a comprehensive evaluation framework for MILP instance generation. The framework establishes two complementary perspectives for assessment: solver-dependent and solver-independent features. Furthermore, the authors conduct extensive experiments to demonstrate the effectiveness and robustness of EVA-MILP.

### Strengths
1. This paper is well written and easy to understand.  

2. I find it interesting that the authors formalize the evaluation from both solver-independent and solver-dependent perspectives. The hardness of a MILP instance is influenced not only by the intrinsic characteristics of the instance itself but also by the behavioral features of the solver. So, establishing a two-dimensional evaluation mechanism is both reasonable and necessary.

3. The authors further strengthen their contribution by conducting extensive experiments on various MILP instance generation methods, which effectively demonstrate the comprehensiveness and methodological soundness of EVA-MILP.

### Weaknesses
1. EVA-MILP primarily relies on synthetic datasets, lacking real-world MILP instances such as those from MIPLIB. MILP instances often exhibit locally optimal behavior, where various methods quickly converge to local optima and then stagnate, making it unclear how such cases should be evaluated.

2. I find that the experiments in this paper make extensive use of high-thread Gurobi configurations, which may reduce the distinguishability of results. The parallel heuristics and scheduling mechanisms in such settings could diminish performance differences across instances, thereby lowering the evaluation sensitivity of the framework.

### Questions
1. Could you please include an analysis of the evaluation on MILP instances from MIPLIB?

2. Could you set the Gurobi thread number to 1 for the analysis?

3. Regarding the GNN downstream tasks, there are several other mainstream frameworks, such as HEM and Predict-and-Search. Could you also analyze the evaluation of EVA-MILP within these frameworks?

4. I am a bit confused about the solver-dependent features. Although EVA-MILP successfully establishes stable statistical characteristics of solver-internal behaviors, its analysis remains limited to a macroscopic distribution level. In real-world or high-dimensional synthetic MILP scenarios, distributional similarity alone may not reveal the dynamic differences in solver behavior during critical phases. For example, the gap improvement ratio after each cut addition.

### Soundness
3

### Presentation
3

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
The paper introduces EVA-MILP, a unified framework for evaluating MILP instance generation methods. It defines both solver-independent and solver-dependent metrics, incorporating internal solver signals (root node gap, heuristic success count, cut plane usage) to characterize computational behavior, and downstream tasks such as hyperparameter tuning and initial-basis prediction to measure practical utility. The authors benchmark several existing generators and discuss interesting findings such as the mismatch between structural and behavioral similarity.

### Strengths
1. This work identifies an important problem: the lack of a standardized, reproducible evaluation protocol for MILP instance generation. It is beneficial to the community.

1. In introduces solver-internal statistics to measure behavioral similarity, which is novel and clearly useful beyond coarse metrics like runtime. It also Includes downstream tasks to measure the real-world utility rather than synthetic difficulty alone.

1. Experiments cover several generators with detailed analyses, and the discussion section provides genuinely interesting insights, especially the finding 3.

### Weaknesses
1. Each generator is only tested on selected datasets with rare explanations. This limits the comparability across different methods. The authors may want to test these methods on all datasets, or detail the reasons why some methods cannot run on the specific datasets.

1. The chosen datasets are mostly simple synthetic ones. Harder or real-world MILPs would make the conclusions stronger. Moreover, I recommand the authors to include some real challenging datasets with only a few instances to demonstrate the effectiveness of generation techniques under data sparsity.

1. Some conclusions (e.g., simple outcome metrics not reflecting true difficulty or structures) are supported only qualitatively. It would be better to provide some correlation analysis or case studies.

1. The choice of metrics is intuitive but not fully justified, and the authors may want to provide further explanations. See questions.

### Questions
1. Why is initial basis prediction chosen as the downstream task? Would other tasks like initial solution prediction or learning2branch also be appropriate?

1. For the internal metrics, why focus on these three ones (root gap, heuristic count, cut usage)? How comprehensive are they in representing solver dynamics?

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
5

### Summary
The paper introduces EVA-MILP, a benchmark framework to evaluate MILP instance generation methods. It combines solver-independent metrics and solver-dependent metrics. Experiments cover Ecole-style datasets (SC/CA/CFL/IS) and ML4CO, and compare several generators (G2MILP, ACM-MILP, DIG-MILP).

### Strengths
Important problem: the community needs more principled evaluation for synthetic MILP instances.

### Weaknesses
- Trivial “copying” generators can score highly by reproducing the original data distribution. The framework lacks novelty/diversity/anti-duplication checks, so high scores may not reflect actual usefulness.

- Heavy solver dependence. Are Gurobi-based evaluations necessarily correct or representative? Branching-node counts and “internal features” can differ across solvers (e.g., Gurobi vs. SCIP). If they are not consistent across solvers, why should “similar branching nodes under Gurobi” be taken as evidence of instance similarity?

### Questions
Same as weaknesses: overall, this feels like a one-sided evaluation framework with limited practical value—rewarding copying without novelty/diversity checks, and relying on Gurobi-specific behavior without cross-solver justification.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces EVA-MILP, a benchmarking framework designed to evaluate existing MILP instance generation methods. EVA-MILP assesses generated MILP instances from two perspectives—Fidelity and Utility—and categorizes evaluation metrics into solver-internal and solver-external features. Furthermore, the paper proposes using solver-internal features as a novel approach to evaluate the quality of generated MILP instances.

### Strengths
1. Establishing a systematic and standardized evaluation framework for MILP instance generation is crucial for advancing research in combinatorial optimization data synthesis.
2. EVA-MILP provides a comprehensive organization of existing evaluation metrics and proposes a clear and structured categorization.
3. The paper offers valuable insights into the design of evaluation metrics, which may inform future work in MILP instance generation and benchmarking.

### Weaknesses
1. While the paper reviews and classifies many evaluation metrics, most of them are derived from previous studies. The only original contribution appears to be the introduction of solver-internal features.
2. The experiments are conducted solely on the CA, IS, and SC datasets from ACM-MILP and DIG-MILP. The framework’s performance on more challenging MILP problems—such as TSP, Graph Coloring, VRP—or on real-world benchmarks like MIPLIB remains unexplored, which is crucial for assessing generalizability.
3. The authors argue that generative models are inherently specialized for specific problem types, and therefore only evaluate each model on a limited set of problems (as shown in Tables 1 and 2). This experimental design prevents an objective comparison among different generation methods.
4. The definition of the challenges in MILP instance generation (lines 088–097) is incomplete. For instance, if a generator simply reproduces training data, it provides no real utility—undermining all similarity-based evaluation metrics between generated and reference instances.
5. I agree with the idea of using downstream tasks to evaluate generated instances, but such tasks must have real practical value (e.g., accelerating ML-based solvers). The current choice of hyper-parameter tuning lacks meaningful application: if one intends to tune solver parameters for a problem set, it is more direct to tune them on the original data rather than on newly generated ones.
6. The authors should consider incorporating a wider variety of practically relevant downstream tasks to more comprehensively demonstrate the framework’s effectiveness.

### Questions
1. Building upon W2, can the authors extend the benchmark to include more diverse problem types and provide results showing which generation methods are suitable for each?
2. The current experiments involve only small-scale instances solvable within one second. Can the authors evaluate how these methods perform on larger MILP instances?
3. In Section 4, the paper evaluates instance similarity by solving for 120 seconds and comparing solver-internal features. How sensitive is the stability of these metrics (measured by W-1 distance) to the choice of time limit (e.g., increasing or decreasing 120s)?
4. Were the experiments in Section 4 repeated to rule out randomness introduced by data partitioning?

### Soundness
2

### Presentation
2

### Contribution
2
