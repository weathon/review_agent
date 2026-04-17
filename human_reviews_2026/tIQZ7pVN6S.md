# Generalizable Heuristic Generation Through LLMs with Meta-Optimization

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 8

## Abstract
Heuristic design with large language models (LLMs) has emerged as a promising approach for tackling combinatorial optimization problems (COPs). However, existing approaches often rely on manually predefined evolutionary computation (EC) heuristic-optimizers and single-task training schemes, which may constrain the exploration of diverse heuristic algorithms and hinder the generalization of the resulting heuristics. To address these issues, we propose Meta-Optimization of Heuristics (MoH), a novel framework that operates at the optimizer level, discovering effective heuristic-optimizers through the principle of meta-learning. Specifically, MoH leverages LLMs to iteratively refine a meta-optimizer that autonomously constructs diverse heuristic-optimizers through (self-)invocation, thereby eliminating the reliance on a predefined EC heuristic-optimizer. These constructed heuristic-optimizers subsequently evolve heuristics for downstream tasks, enabling broader heuristic exploration. Moreover, MoH employs a multi-task training scheme to promote its generalization capability. Experiments on classic COPs demonstrate that MoH constructs an effective and interpretable meta-optimizer, achieving state-of-the-art performance across various downstream tasks, particularly in cross-size settings. Our code is available at: \url{https://github.com/yiding-s/MoH}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Meta-Optimization of Heuristics (MoH), a new framework that uses large language models (LLMs) to automatically discover effective heuristic optimizers for combinatorial optimization problems (COPs). Unlike prior work that relies on fixed evolutionary computation (EC) strategies and single-task training, MoH operates at the optimizer level. It employs an outer loop for meta-optimization (optimizer design) and an inner loop for heuristic design. The framework uses multi-task training to improve generalization and supports diverse optimizer discovery through (self-)invocation mechanisms within LLMs. Extensive experiments—including TSP, online/offline Bin Packing, and CVRP—demonstrate that MoH outperforms traditional, neural, and recent LLM-based heuristic generation methods in both performance and scalability.

### Strengths
- The shift from fixed heuristic-optimizers to directly searching for optimizers automates the design of entire optimization frameworks, not just heuristics. This represents a conceptual advance over state-of-the-art LLM-based automatic algorithm design.
- MoH is benchmarked across multiple tasks and demonstrates state-of-the-art or competitive optimality gaps, especially when generalizing to larger instance sizes.
- Cost analysis—covering LLM requests, tokens, and wall time—helps contextualize practical applicability.

### Weaknesses
- Despite noteworthy generalization to larger problem instances and other COPs, the empirical evaluation focuses on classical, well-studied benchmarks (TSP, BPP, CVRP). The approach hasn't been applied to real-world, domain-constrained, or industrial COPs, which may limit its immediate practical impact. It would be interesting to see how it performs on real-world optimization problems that LLMs are unfamiliar with. Author comments on this aspect would be valuable.
- The distinction between meta-optimizer and heuristic-optimizer is confusing, especially since they turn out to be the same in the end.
- The comparaison to other baselines seems not fair, see questions for details. I would like to hear rebuttal from the authors before making the final decision.

### Questions
- During inference, you run MoH for 10 iterations and select the best-performing heuristic based on the final results. Is this strategy also applied to other baselines?
- In Table 8, you distinguish between MoH-Train and MoH-Inference, but Table 9 only mentions MoH. Could you clarify this difference?
- For Tables 8 and 9, testing on large instances increases MoH inference time but not the other methods, correct? The same concern applies to token consumption. I suggest clarifying the entire procedure.

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
3

### Summary
The paper presents Meta-Optimization of Heuristics (MoH), a framework that uses large language models (LLMs) to generate effective, interpretable heuristics for combinatorial optimization problems (COPs). Unlike traditional methods, which rely on predefined evolutionary computation (EC) heuristics or single-task training, MoH uses meta-learning to automate the design of meta-optimizers, enabling broader heuristic exploration and better generalization to new problems. The authors demonstrate MoH’s superiority over existing LLM-based methods on classic COPs like the Traveling Salesman Problem (TSP) and Bin Packing Problem (BPP).

### Strengths
1. MoH introduces the idea of meta-optimization within LLM-based heuristics for combinatorial optimization, addressing key limitations of existing methods like the lack of diversity in heuristic exploration and challenges in generalization.
2. Extensive experiments demonstrate that MoH outperforms both traditional and LLM-based heuristic methods across various settings, showing its ability to tackle problems like TSP and BPP effectively.

### Weaknesses
1. While the authors claim that MoH does not incur significant computational overhead, the introduction of a meta-optimization layer adds complexity, which may increase the time and resources required, especially for large problems.
2. Though MoH performs well on classical COPs, its scalability to more complex or non-classical optimization problems (e.g., real-world applications) has not been thoroughly tested.
3. While multi-task learning is a strength, it could also lead to overfitting on the training tasks if not managed properly. The paper doesn't provide a clear strategy for mitigating such risks.
4. While MoH increases the exploration space, there is no detailed analysis of how efficiently it can explore very large or complex search spaces in comparison to simpler heuristics or other optimization techniques.

### Questions
1. Please list up and carefully describe any questions and suggestions for the authors. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. This is important for a productive rebuttal and discussion phase with the authors.
2. While MoH performs well on classical COPs, its scalability to more complex, real-world optimization problems (e.g., dynamic environments, non-classical COPs) has not been thoroughly tested. Can you provide any insights into how MoH might adapt to these problems? Have you considered testing MoH on real-world benchmarks or dynamic problem settings?
3. Multi-task learning is a strength of MoH, but it could also lead to overfitting, especially when tasks are not sufficiently diverse or are too similar. The paper does not clarify how overfitting is mitigated during training. Could you elaborate on the strategies used to ensure that the framework generalizes well across tasks? Did you apply any regularization techniques, cross-validation, or other safeguards to address this risk?
4. While MoH expands the heuristic search space, how does it perform when compared to simpler heuristic methods or other optimization techniques, especially in terms of efficiency? Given the large search space, how does MoH ensure it doesn’t waste resources on ineffective explorations? Could you provide a more detailed analysis of MoH’s efficiency in exploring vast search spaces, especially for large and complex problems?

### Soundness
2

### Presentation
2

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
This paper proposes Meta-Optimization of Heuristics (MoH), a novel framework that uses Large Language Models (LLMs) to generate generalizable heuristics for Combinatorial Optimization Problems (COPs). Unlike prior methods that optimize heuristics directly, MoH operates at the optimizer level. It aims to discover a highly effective "heuristic-optimizer" by meta-learning. This process involves optimizing a meta-prompt that guides an LLM to sample and refine heuristics. The meta-optimizer is trained to maximize a utility function across a diverse set of tasks (e.g., TSP instances of varying sizes), with the goal of producing heuristics that generalize well. The authors evaluate MoH on the Traveling Salesperson Problem (TSP), demonstrating improved performance over several baseline methods.

### Strengths
- The core idea of optimizing the optimizer (via a meta-prompt) rather than just the heuristics themselves is a novel and interesting approach to leveraging LLMs in the optimization domain.
- The paper correctly identifies generalizability as a key weakness in existing heuristic generation methods and explicitly designs its utility function to reward performance across different task distributions (i.e., problem sizes).

### Weaknesses
- While the method is described with complex terminology, its core mechanism appears to be a sophisticated form of meta-prompt optimization. The "meta-optimizer" is, in essence, a highly-tuned prompt that guides the LLM to sample effective heuristics. This idea, while implemented well, feels intuitive and perhaps more incremental than a fundamental breakthrough, which may limit the paper's conceptual contribution.
- The paper's "generalizability" claim is weak and potentially misleading. Firstly, the framework does not generalize across problem domains; it requires training a new, specialized "meta-optimizer" for each problem class (TSP, BPP, CVRP). Secondly, even within a single problem, the generalization is limited to varying instance sizes from the same data distribution. There is no evidence that the optimizer generalizes to new instances drawn from a different distribution (e.g., from uniformly distributed TSPs to clustered TSPs).
- The experimental comparison to baselines appears to be unfair. MoH's computational cost includes both a training phase (1,000 heuristic evaluations) and a separate, additional "inference stage" to generate the final heuristics. In contrast, baseline methods like EoH are presented as more "online" and may not have this distinct (and costly) inference phase. For a fair comparison, the baselines should be allocated a total computational budget equal to the sum of MoH's training and inference costs. This is particularly concerning given that the performance gains reported in Table 1 are incremental.

### Questions
- If the experiment is run multiple times, will it produce a "meta-optimizer" with similar performance, or are the results highly variant? This is a critical point for assessing the method's reliability.
- Why was the utility function weighted by the size of the task? Figure 1 seems to suggest that performance suffers when emphasizing larger instances. What is the performance of the MoH framework when using a uniform weight for all task sizes? What is the performance if the baselines use the weighted utility?
- What is the performance of the final heuristics obtained at the end of the training phase? This would help clarify the exact performance gain and cost attributed to the separate inference stage.
- Could you clarify the practical difference between the heuristic generation strategy used in the MoH inference stage and the strategy used by the baseline EoH?
- What specific heuristic was tested on the TSPLIB benchmark? Was it the single best heuristic from Table 1? If so, from which problem size distribution was this heuristic generated?
- Why are the results for the ReEvo baseline missing in the TSP-GLS case in Table 1?
- The paper requires careful proofreading to correct several typos (e.g., "generats" in line 74, "hsmaller" in line 928).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes MoH (Meta-Optimization of Heuristics), a two-level LLM-driven framework that searches not just for task heuristics, but for the heuristic-optimizers that generate them. An outer loop uses a meta-optimizer to invoke an LLM and produce candidate heuristic-optimizers, while an inner loop applies each candidate to evolve concrete heuristics for downstream COP tasks. Selection uses utility on validation tasks, and training is multi-task to encourage cross-size generalization. Experiments on TSP and online BPP report state-of-the-art gaps, especially on larger, unseen sizes. Further ablations suggest benefits from maintaining populations and using natural-language idea descriptions.

### Strengths
Ablations show benefits from the proposed idea and examine different LLM backends and population sizes.

The paper provides concrete examples/analysis indicating discovered strategies can resemble or hybridize classic metaheuristics.

The paper shows strong empirical results and cross-size generalization on TSP and Online BPP. The proposed MoH often outperforms baselines.

Multi-task training and controlled evaluation budgets are thoughtfully designed to encourage generalization.

### Weaknesses
Improvements over strong baselines can be modest in some settings. Further discussion should be included.

According to experimental setups, main tables emphasize best-of-three runs, which can overstate gains versus mean/variance reporting.

There might exist sensitivity to LLM choice and prompts. The robustness under model drift is uncertain.

### Questions
Which two main loops make up the MoH framework?

Does the paper claim cross-size generalization on TSP?

Does the method rely on a population of candidates during search?

What is the difference between heuristic-optimizers and meta-optimizers?

### Soundness
4

### Presentation
3

### Contribution
4
