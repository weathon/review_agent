# Graph Coloring for Multi-Task Learning

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
When different objectives conflict with each other in multi-task learning, gradients begin to interfere and slow convergence, thereby potentially reducing the final model's performance. To address this, we introduce SON-GOKU, a scheduler that computes gradient interference, constructs an interference graph, and then applies greedy graph-coloring to partition tasks into groups that align well with each other. At each training step, only one group (color class) of tasks are activated, and the grouping partition is constantly recomputed as task relationships evolve throughout training. By ensuring that each mini-batch contains only tasks that pull the model in the same direction, our method improves the effectiveness of any underlying multi-task learning optimizer without additional tuning. Since tasks within these groups will update in compatible directions, multi-task learning will improve model performance rather than impede it. Empirical results on six different datasets show that this interference-aware graph-coloring approach consistently outperforms baselines and state-of-the-art multi-task optimizers. We provide extensive theory showing why grouping and sequential updates improve multi-task learning, with guarantees on descent, convergence, and accurately identifying what tasks conflict or align.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents an interference-aware scheduler named SON-GOKU, which is designed to improve the efficiency and effectiveness of multi-task learning by dynamically grouping tasks based on their gradient conflicts. The proposed method addresses the issue of negative transfer, where conflicting tasks slow down or degrade overall performance. The core idea revolves around constructing a conflict graph from task gradient interference and using greedy graph coloring to partition tasks into non-conflicting groups. SON-GOKU periodically recomputes these groups and activates only one group per training step, aiming to ensure that each mini-batch contains tasks pulling the model in compatible directions. Experiments demonstrate the effectiveness and high scalability of this method.

### Strengths
1. The paper is well-motivated and quite novel in solving gradient conflicts. The introduction explains how greedily coloring the graph addresses the existing three issues of grouping methods.
2. The authors provide rigorous theoretical guarantees on grouping, convergence, and conflict graph recovery.
3. The experiments demonstrate that SON-GOKU outperforms baselines (e.g., PCGrad, AdaTask, FAMO) and complements existing optimizers like PCGrad and AdaTask.
4. Using graph relationships to represent gradient conflicts between tasks is quite interesting and provides a new direction for solving multi-task problems.

### Weaknesses
1. This paper uses the interference coefficient to measure the relationships between tasks and then constructs a graph. However, the relationships between tasks can also be represented by a causal matrix, which can also be constructed into a graph. Therefore, what are the advantages or differences between the graph construction method in this paper and the causal graph construction?
2. Appendix D.5 discusses multiple approaches for reducing time complexity—random projections, deterministic covariance sketching, edge sampling, incremental Gram updates, and warm-start initializations. While each method is described, its individual impact on the final performance is not clearly distinguished. There could be more experimental results to support the theoretical analysis.
3. This paper lacks some parameter experiments, such as how the tau parameter changes throughout the training process, and how the task graph evolves in specific experiments
4. Although this article provides a detailed description of the method steps, making it highly reproducible, the author does not appear to have provided the code.

### Questions
Please see weakness.

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
4

### Summary
This paper introduces a scheduler for multitask learning that periodically measures pairwise gradient interference (via cosine similarity of EMA-smoothed gradients), builds a sparse conflict graph, and then applies greedy graph coloring to partition tasks into compatible groups. It first measures pairwise gradient interference using EMA-smoothed cosine similarities, builds a sparse conflict graph linking conflicting tasks, and then applies greedy graph coloring to partition them into non-conflicting groups. During training, SON-GOKU activates one color group (i.e., compatible tasks) per step and cycles through all groups, ensuring updates proceed in consistent gradient directions. The graph is periodically recolored to adapt to evolving task relations, and the approach integrates seamlessly with existing optimizers like PCGrad or AdaTask.

For the theoretical results, SON-GOKU provides guarantees. If there are no edges among its tasks in the conflict graph, the combined step is provably a descent direction with a quantitative lower bound; the overall procedure retains the standard non-convex SGD convergence rate up to a small factor depending on within-group conflict. Moreover, over a refresh window, doing sequential group updates is not worse and can be strictly better than a single mixed update when cross-group gradients oppose each other. 

The authors evaluate SON-GOKU on six benchmarks spanning vision, multimodal, and time-series tasks, using standard backbones (e.g., ResNet-18; CNN/BiLSTM) and deliberately adding positive/negative auxiliary tasks to induce interference. They compare against competitive baselines (including Uniform, GradNorm, AdaTask, MGDA, Nash-MTL, FairGrad, PCGrad, CAGrad, FAMO), and also test hybrid variants where the scheduler is paired with AdaTask, a GradNorm warm start, or PCGrad. Ablations probe two key design choices: freezing a one-shot coloring (to test the need for dynamic recoloring) and estimating conflicts from single minibatches to assess the value of history-averaged interference. Across ten metrics on the six datasets, SON-GOKU consistently matches or outperforms baselines.

### Strengths
- By using a sparse, interference-aware graph that is periodically recolored as training evolves, the proposed method effectively minimizes gradient conflicts without adding significant computational overhead. 

- This work provides theoretical guarantees that sequentially updating low-conflict groups yields at least as much expected descent as mixed-task updates and converges at the same rate as non-convex SGD, with only a small correction for within-group conflict. 

- The method efficiently scales to many tasks by maintaining a sparse conflict graph estimated from lightweight minibatch statistics, avoiding dense and noisy affinity matrices. It also supports dynamic recoloring to adapt to evolving task relations and can be applied on top of existing multi-task optimizers like PCGrad, GradNorm, or AdaTask.

### Weaknesses
- The theoretical results indicate that the proposed algorithm's convergence is not worse than standard SGD. Are there any empirical/theoretical observations to explain why using the gradients within a non-conflict group can lead to improved generalization performance?

- This work uses the averaged gradients to measure task conflicts during training. How would this work with other task affinity measures, such as the lookahead loss in TAG (NeurIPS 2021)?

### Questions
- It would be better to report the standard deviations of the experimental results across multiple random seeds. 
- It would be better to illustrate the convergence of the method in plot figures.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents SON-GOKU, a scheduling framework that estimates gradient interference to build an interference graph, then partitions tasks into mutually compatible groups. By activating only well-aligned groups and updating the grouping over time, SON-GOKU boosts the performance of diverse multi-task baselines and comes with theoretical guarantees.

### Strengths
1. The paper is clearly written and easy to follow.
2. Building a conflict graph and using it for multi-task optimization is a novel idea.
3. It provides theoretical analysis with convergence guarantees, which are commonly expected in this field.

### Weaknesses
1. It is unclear whether the approach scales to larger backbones. In my experience, task interference tends to diminish as model capacity increases, which could limit the gains.
2. The paper reports algorithmic time complexity, but it does not sufficiently compare memory usage against other methods.

### Questions
How should we adapt the refresh rate? Higher refresh rates track task relations quickly but can destabilize updates, while lower rates reduce noise and overhead at the cost of slower adaptation or maybe lower performance.

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
This paper tackles the gradient-conflict problem in multi-task learning that slows convergence and degrades final accuracy.  It proposes SON-GOKU, a scheduler that quantifies pairwise gradient interference, builds an interference graph, and uses greedy graph coloring to split tasks into harmony groups that are updated one at a time.  Extensive experiments on six datasets show consistent gains over strong baselines and state-of-the-art optimizers, while novel theory proves descent guarantees and convergence under the proposed grouping strategy.

### Strengths
1. This paper pinpoints the phenomenon of gradient interference that slows convergence in multi-task learning, and introduces an effective grouping strategy to counter it.
2. Extensive experiments across diverse benchmarks convincingly validate the proposed method, showing consistent and non-trivial improvements over strong baselines.
3. The work delivers a crisp message: task grouping is vital for mitigating gradient conflict.

### Weaknesses
1. The paper mentions runtime overhead but provides no measured training time or inference latency.
2. The ablation studies only include one-shot and single-step conflict estimation, while the key component (e.g., graph building) is lacking. 
3. Ablation results are split across Sections 6.4 and 7.2 without a clear organizing principle, which fragments the narrative and complicates comparison.

### Questions
Please check the weakness

### Soundness
3

### Presentation
2

### Contribution
2
