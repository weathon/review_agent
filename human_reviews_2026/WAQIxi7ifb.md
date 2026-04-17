# Learning Admissible Heuristics for A*:  Theory and Practice

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6, 6, 8, 6

## Abstract
Heuristic functions are central to the performance of search algorithms such as A*, where \emph{admissibility}—the property of never overestimating the true shortest-path cost—guarantees solution optimality. Recent deep learning approaches often disregard full admissibility and provide limited guarantees on generalization beyond the training data. We address both of these limitations. First, we pose heuristic learning as a constrained optimization problem and introduce \emph{Cross-Entropy Admissibility (CEA)}, a loss function that enforces admissibility during training. When evaluated on the Rubik’s Cube domain, our method yields heuristics with near-perfect admissibility and significantly stronger guidance than compressed pattern database (PDB) heuristics. On the theoretical side, we derive a new upper bound on the expected suboptimality of A*. By leveraging PDB abstractions and the structural properties of graphs such as the Rubik’s Cube, we tighten the bound on the number of training samples needed for A* to generalize to unseen states. Replacing a general hypothesis class with a ReLU neural network gives bounds that depend primarily on the network’s width and depth, rather than on graph size. Using the same network, we also provide the first generalization guarantees for \emph{goal-dependent} heuristics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper develops theory and practice for learning admissible A* heuristics. It tightens sample-complexity bounds under equal-cost edges and gives a sharper suboptimality guarantee that bounds A*’s error by the maximum inadmissibility on the optimal path, with extensions to neural-network and goal-dependent settings.
Practically, it introduces the Cross-Entropy Admissibility (CEA) loss, which shifts probability mass onto admissible classes and sharpens around the true class.
On Rubik’s-Cube PDBs, CEA-trained models achieve near-admissibility and match or exceed the average heuristic of same-size compressed PDBs

### Strengths
* The theory derives tighter generalization bounds that incorporate neural network width/depth and handle goal-dependent heuristics.
* Heuristic learning is cleanly framed as a constrained optimization problem.
* The suggested CEA loss seems to effectively encourage admissibility in the provided experimental results.
* Overall, the paper is well written.

### Weaknesses
* The theoretical results mostly assume equal-cost edges (Assumption 4). While this is common for toy problems like Rubik’s Cube it limits immediate transfer to domains with varied costs.
* While the CEA loss suggested in Equation 5 seems conceptually plausible, it lacks a clear motivation why this specific function was chosen. The goal of shifting some probability mass towards admissibility could be expressed through other loss functions as well, for example by minimizing the kl divergence between the predictions and a suitably defined target distribution. Were other loss functions considered? If so, why was this specific one chosen?
* Some details of the experimental results are vague. No details on how the data is split into train/validation/test splits is provided.
* As the theory in section 3 derives the sample complexity for learning heuristics, an ablation study that measures the heuristics performance as a function of the training set size would strengthen the connection between the works theoretical and empirical sections.
* The CEA loss introduces β and η as additional hyperparameters that are adjusted in stages during training, which adds complexity to the training process.

While the theoretical results and novel CEA loss are contributions, I do think these weaknesses currently outweigh the strengths of the paper.

### Questions
* How was the data split for the empirical study? How many samples are used for training?
* Can the CEA loss be extended to settings with non-uniform edge weights?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper extends existing theoretical analyses on the sample complexity of
learning admissible heuristics, then proceeds to define a new loss function
to achieve such admissibility.
The contributions in this work are

-   Thm2: Extension of Sakaue & Oki 2022's result on Pdim to PDB's abstracted states
-   Thm3: a tighter suboptimality bound when A\* with is allowed to reopen nodes &#x2013; extension of Valenzano '14,
-   Thm4: Eq. (3) which has O(sqrt(n)) instead of O(n) in Eq (1) = Prop 1 + Sakaue & Oki 2022.
-   Thm5: Extension of Sakaue & Oki 2022's result on Pdim to NNs, with/without PDBs
-   Thm6: Extension of Thm5 to NNs with specific sizes
-   Thm7: Extension of Thm6 to different goals
-   proposing CEA loss.

I have a mixed impression about this work. While the paper is tackling an important problem,
the paper appears to be written by two authors without a necessary organic integration,
which is a common phenomenon &#x2014; one symbolic author and one machine learning author failing to understand each other, stuck in their own worldview.
This might be causing inconsistent math notations (e.g. log vs. lg, $\ell$ vs. $D$),
overlapping symbols ($W$ as cost upper bound vs parameter size),
and an apparent disconnect between the theoretical result and the empirical result described below.

The paper did excellent work coverint the backgrond on the sample complexity analysis.
However it is light on proofs, which are largely deferred to the appendix,
and has several minor notation issues which must be fixed.

I also have a bigger concern about the practical part of this paper: CEA, and the empirical evaluation.

### Strengths
The paper did excellent work covering the backgrond on the sample complexity analysis.

### Weaknesses
## Issue 1: CEA

As discussed in [Nunez-Molina 2024] cited in this work,
the choice of the loss function must directly correspond to the statistical representation of the output.
Categorical target random variable yields the cross-entropy loss naturally as a negative log likelihood.
The proposed CEA loss does not have a corresponding distribution.
If the author wishes to impose some mathematical property on the prediction target,
the right way to design a machine learning model is to idenfity the correct representation of the target as a random variable,
then derive the loss function naturally from its pdf.

One such alternative as a baseline for comparison against CEA I suggest is **ordinal logistic regression** for ordinal variables.
Ordinal regression [1] is a classical method for predicting ordinal random variables,
which naturally address the lack of orders in the categorical prediction (classification).
If the model believes that the output is larger than k,
then the model architecture must hard-code the invariance that the output is also larger than k-1, k-2, and so on.

[1] McCullagh, Peter (1980). "Regression Models for Ordinal Data"

One of its simplest implementation uses the cumulative binary logistic loss,
which outputs a single scalar with a NN, subtract monotinically increasing threshold values,
apply a sigmoid, then train the output with binary cross entropy.

parameters:

-   A scalar output neural network $f$ (the last layer does not need the bias)
-   A bias vector $\mathbf{b} \in R^k$ for prediction over $\{0, 1, ... k\}$

how:

-   input $x$
-   target value $h^* \in ｛0, 1, ... k｝$
-   binary prediction target $\mathbf{y} \in ｛0,1｝^k$:
    -   $y_h = 1$ if $h > h^*$ ;
    -   $y_h = 0.5$ if $h=h^*$ ; 
    -  0 otherwise.
    -  For example, it looks like $[0,0,0,0,0.5,1,1,1]$ when $h^*=4$ and $k=7$.
-   scalar logit $l = f(x)$
-   thresholds $\mathbf{\theta}$
    -   $\theta_0 = b_0$
    -   $\theta_h = \theta_{h-1} + softplus(b_h)$
        -   This ensures $\theta_h > \theta_{h-1}$
-   output: vector $\mathbf{p}\in [0,1]^k$, where
    -   $p_h = \sigma( l - \theta_h )$
    -   $p_h$ represents $p(h^* < h | x)$ , and this framework guarantees $p(h^* < h | x) < p(h^* < h+1 | x)$ .
-   loss: $\sum_h CE( y_h, p_h )$

Notice that the final loss naturally penalizes the misprediction of $p(h^* < h | x)$ the further $h$ is away from $h^*$.

For example, when $h^*=4$ and $p_6$ is smaller than 0.5, the framework ensures that $p_4, p_5$ are even smaller, resulting in three times or more penalty.
This has a similar characteristics as the first term of the CEA, while maintaining the natural statistical interpretation.

You may however return a rebuttal: Then how do we interpret the tunable &beta; in CEA, and the fact that it only applies to the overestimation?

My answer: As cumulative binary logistic loss is a sum of losses for a series of binary predictions,
we can achieve the same effect by multiplying a set of coefficient to each loss.
This is a common practice for addressing a class imbalance (although our purpose is different)
and it is mathematically equivalent to oversampling/undersampling a subset of the dataset.
(Well, there may indeed be a class imbalance.
see the next section blow.)
Concretely, use the loss as follows:

-   loss: $\sum_h w_h CE( y_h, p_h )$

You are free to choose any $w_h$, e.g. $w_h = g(h, h^*)$ for some formula g that increases as $h$ overestimates.
So, there is no ad-hoc tricks; It simply artificially increases the penalty to the overestimation.


## Issue 2: PDB learning

I found that the method section and the evaluation section is quite ambiguous about the training data.
I struggled to discover what is the input representation to the NN, what is the taget value, and how many data points are used.
Please add a separate section titled "data collection" which details what goes in the pipeline.
I initially had 4x2 = 8 possible interpretations:

-   Input x to the NN:
    -   Raw state representation, factored
    -   Raw state representation, in a form of state id
    -   PDB abstract state representation, factored (\*)
    -   PDB abstract state representation, abstract state id
-   Target data:
    -   optimal solution cost $h^*(x)$ obtained by search from x using admissible PDB
    -   admissible heuristic $h(x)$ from the PDB &#x2014; which is an optimal path cost in the abstract state space
-   The number of data points

The detail of the input representation is only given in the appendix.
While the full detail is not necessary, the main paper must at least mention
that it is the factored abstract state representation of PDB.

What's more important is the target data.
I know you used $h^*$ in Eq.5, but I don't have a good confidence because
there is no mention of how it was collected.

I could not find any information about the number of data points.
Did you use the entire abstract states in the PDB without subsampling?
If so, the dataset is hugely imbalanced, as you pointed out in the paper.
The standard practice is oversampling, undersampling, and bagging. See [1].
These standard measures completely normalizes the number of samples per class.

The lack of mention to how the author addressed the class imbalance imply a danger that
the perceived improvement from the CEA loss could be just an artifact of **accidentally** somewhat addressing
the class imbalance in an ad-hoc way.
(As said above, multiplying $(k/h^*)^\beta$ to the loss is mathematically equivalent to oversampling.)

The authors also mentioned delta-PDB, which would shift h's distribution and also somewhat reduces the class imbalance,
but the Table 3 shows that the entire dataset still contains a massive class imbalance.

[1] Wallace, Byron C., et al. "Class imbalance, redux." 2011 IEEE 11th international conference on data mining. Ieee, 2011.

## Issue 3: Disconnect between theoretical results vs experiments

The theoretical and the empirical results in this paper are, while individually interesting, quite disconnected.


### theorem 4

First, the paper does not empirically verify Theorem 4.
I expected an experiment that estimates the LHS of eq.3 using a validation set as a surrogate or the entire (abstract) state space,
subtract the first term of Eq.3, then plot the remaining value to confirm that it grows $O(\sqrt(n))$.

Theroem 4 does not require factored state representation, Rubiks cube, PDBs or the CEA loss function.
Instead, theorem 4 is a function of the graph size n and the sample size N.

In other words, while section 3.1 is by itself an interesting contribution,
its empirical confirmation must be done separately using a state space whose size is configuratble,
e.g., not in Rubik cube, but in artificial random graphs such as those studied by Cohen & Beck [AAAI17, AAAI18, IJCAI18]
whose $n$ can be configured.

-   Problem Difficulty and the Phase Transition in Heuristic Search, AAAI17
-   Local Minima, Heavy Tails, and Search Effort for GBFS, AAAI18
-   Fat- and Heavy-Tailed Behavior in Satisficing Planning, IJCAI18


### theorem 7

Another main theoretical results in this paper is the sample complexity in theorem 7.
I expected that the empirical analysis verifies this by one of these experiments:

-   Experiment 1: Given a fixed network, compute this upper bound Pdim(U) with
    Thm7. Then gradually reduce the dataset size by subsampling. See what happens
    if the dataset size goes below Pdim(U)
-   Experiment 2: Given a fixed dataset, reduce the number of weights in NN to reduce Pdim(U).
    See what happens if Pdim(U) goes below the dataset size.


## Issue 4: Runtime analysis of using the NN

While the paper claims that it tries to find a smaller NN (Eq.4),
it does not do anything about it actually.
The evaluation focuses on accuracy and admissibility,
and never compares the actual runtime or the number of node evaluations
between the sota solver against A\* guided by the heuristics learned by this method.

In other words, as is often the case with many machine learning papers,
I suspect that this paper obtained a quite accurate heuristic that is too slow
to compute to use in practice.


## Minor comments and questions

-   line 80 : b should be a neighbor of a?
-   l.77-91: floor and indicator functions are not used in the paper?
-   l.165-173: better to have a consistency in the choice of letters. H -> U?
-   l.173: h suddenly becomes a real vector. I know this is an array from a state to R, but this is inconsistent to the previous notation h(s) of state s.
-   l.213: $|G_v| <= n^2$ I suppose.
-   l.215: same regarding h.
-   l.215: add "derived from S & Oki 2022".
-   l.216: same reagarding h.
-   l.234: "general class of performance measure", provide examples, e.g., runtime, memory, node evals, etc.
-   l.252: Results in Valenzano 14 holds regardless of admissibility, which is surprising.
    Although it is not wrong not to mention it (because nothing more is not assumed than stated),
    due to the strong association between A\* and admissibility,
    I strongly suggest the authors to explicitly state that the entire section 3.1 does not depend on the admissibility of h.
-   l.268: $\hat{H}$ undefined.
-   l.270: lg n vs log n.
-   l.302: sum of $w_1 \ldots w_n$ is denoted by U. U -> W?
-   l.305: Indeed, here the number of parameters is W.
-   l.317: the number of output classes is $\ell$. It was $H$ before. or $\hat{H}$. wtf?
-   l.326: this time the number of output classes is $D$. wtf?
-   l.333: Rep(I) is not used at all elsewhere in the paper. maybe appendix. please remove it or move it to the appendix, if used there.
-   l.343: $\ell$ and $D$ used at the same time. Also, U and W are used at the same time. $\ell'$ undefined.
-   l.364: $|h|$ is never explicitly optimized. I doubt Eq.4 is necessary.

### Questions
Q1. Confirm the input format is PDB abstract state representation, factored. (yes/no)

Q2. Confirm the target value is really h\* and is obtained by solving x optimally. (yes/no)

Q3. Did you use the entire abstract states in the PDB without subsampling? (yes/no)

Q4: The author claims that Eq.(3) is tighter than Eq.(1) with no explanation.
I assumed O(sqrt(n)) vs O(n) is what they implied. Am I correct? (yes/no)

Also, please add a sentence that explains conceptually why this reduction is achieved.
(E.g., in sorting literature, radix sort can run faster than O(n log n) &#x2014; why?
it is because it can use more information than comparison sort.)
I assume that this is in line with something like: it uses the max in Thm3,
therefore only one element along the path affects the error,
while the first error term in Eq.(1) is essentially a double summation (sum over paths, sub over states on the path)

### Soundness
2

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
2

### Summary
This paper aims to learn admissible heuristics for A*, combining theoretical generalization analysis with practical training improvements. It incorporates a new loss function, called Cross-Entropy Admissibility (CEA), that enforces admissibility during training and analyzes the sample complexity of learning admissible heuristics. Experiments on the Rubik’s Cube show that CEA produces nearly admissible heuristics with very low inadmissibility rates and stronger guidance against compressed pattern databases (PDBs).

### Strengths
1. This paper introduces CEA as a principled loss to enforce heuristic admissibility is both theoretically sound and practically effective.
2. It provides theoretical analysis and derives tighter generalization bounds, incorporating neural networks as the heuristic with  generalization bounds analyzed primally on network size.
3. Experimental results demonstrates strong results on Rubik’s Cube domains, showing measurable gains over traditional PDBs.

### Weaknesses
1. The experiments results are limited to a single domain(the Rubik's Cube). The claims of the paper would be significantly stronger if supported by results on other classic planning domains.
2. The theoretical analysis relies on Assumption 4, while this holds for the Rubik's Cube but is not universal. How do the bounds change for graphs with variable edge costs? 
3. The proposed method is evaluated on several 3 × 3 Rubik’s Cube pattern databases, where  the performance and data requirements for larger graphs remain unclear.

### Questions
Please refer to the weakness.

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
This paper addresses the challenge of learning admissible heuristics for A* search. Many current deep learning approaches often do not consider admissibility but focus on predictive strength, losing theoretical guarantees on optimality. The authors propose new sample complexity bounds for learning such heuristics, that exploit graph structural properties and PDB abstraction. They also derive bounds for neural network that depend on the networks size rather than the problem size. From a practical point of view, the paper frames the heuristic learning as a constrained optimization problem and further reduce it to a classification task that uses a new loss function called cross-entropy admissibility. Empirical results show the method has low overestimation rate without sacrificing informativeness.

### Strengths
1. The formalization of heuristic learning as a constrained optimization and reduce it to a classification task guided by a new loss function is novel.
2. The tighten sample complexity bounds seem to be a good contribution.
3. The paper addresses a gap between ML-based heuristics and rigorous admissibility requirements for optimal search.
4. The paper is well-presented.

### Weaknesses
I wasn't able to check the theory thoroughly, the weaknesses listed below are mainly from empirical study.

1. Experiments are on Rubik’s Cube domain, it is a good benchmark but requires at least of couple more for the empirical results to be fully convincing. The practical value of CEA would be clearer if demonstrated in another admissible-heuristic domain (e.g., sliding-tile puzzle).
2. The paper didn’t describe how the hyper parameters \beta and \eta are tuned, if the author could provide a principled process, it would help reproducibility. 
3. There is no runtime measurements in the experiments.

### Questions
1. Have you quantified A* search impact in terms of running time on your test problems?
2. Have you empirically evaluated your methods on other domains? How does it perform?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper tackles a core challenge at the intersection of machine learning and heuristic search: **how to learn powerful heuristic functions that preserve the optimality guarantees of algorithms like A***.

The authors make contributions on both theoretical and practical fronts:

*   **New Loss Function:** They propose a Cross-Entropy Admissibility (CEA) loss, a new training objective designed to enforce the admissibility constraint directly.

*   **Strong Empirical Performance:** Evaluated on the 3*3 Rubik's Cube, the approach demonstrates remarkable success, learning heuristics that are nearly perfectly admissible while providing stronger search guidance than traditional baselines.

*   **Theoretical Analysis:** The work provides a sample complexity analysis for learning heuristics. By leveraging problem structure and abstractions like Pattern Databases, the authors derive tighter generalization bounds for A*. They also establish a theoretical guarantee for learning goal-dependent heuristics.

In summary, this work introduces a framework for learning reliable heuristics, effectively bridging data-driven learning and classical search algorithms.

### Strengths
- **Strong Empirical Validation:** The work demonstrates compelling empirical results. On the challenging benchmark of the 3x3 Rubik's Cube, it achieves near-perfect admissibility while learning a heuristic stronger than classically compressed alternatives.
- **Noteworthy Theoretical Scope:** The paper provides a significant theoretical analysis of sample complexity. Its most original contribution is extending this framework to establish the first generalization guarantees for *goal-dependent* heuristics, moving beyond standard fixed-goal analyses.

### Weaknesses
- **Limited Empirical Scope:** The empirical evaluation is exclusively conducted on the 3*3 Rubik's Cube. The performance and effectiveness of the proposed CEA loss on other combinatorial puzzles or planning domains remain unknown.
- **Limitations of Theoretical Claim**: The claim that sample complexity is governed by network parameters and independent of graph size may not translate to practical advantages, as it relies on strong assumptions that may fail in real-world problems.
- **Unvalidated Theoretical Extension:** While establishing the first theoretical guarantees for goal-dependent heuristics is a notable contribution, this significant claim is unsupported by empirical evidence. The experiments only address a fixed-goal setting.

### Questions
- **Comparison with Standard Cross-Entropy Loss:** The paper clearly shows that the CEA loss achieves a much lower overestimation rate than the standard CE loss. However, beyond this crucial metric, did you compare the **resulting A\* search performance** (e.g., number of node expansions, wall-clock time) when using heuristics trained with these different losses? 
- **Connection Between Theory and Experiment**: Could you share more details on how the theoretical insights guided the experimental design? For instance, did the theory influence the choice of neural network capacity or the amount of training data needed?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a method, called cross-entropy admissibility (CEA), to learn heuristic functions which, in the context of pathfinding on a weighted directed graph, map states to an estimate of the cost-to-go to a closest goal state. CEA focuses on learning heuristic functions that are admissible, meaning they never overestimate the cost of a shortest path. CEA formulates heuristic learning as a classification task, where each possible heuristic value is treated like a class, and more emphases is put on penalizing overestimation. The paper focuses on the Rubik’s cube, where all transition costs are one and, therefore, all possible true cost-to-go values are integers. Results with neural networks on learning heuristic values from PDBs show that the proposed method learns heuristic values that have significantly fewer underestimations than regular cross entropy (CE) while having a slightly less or similar average heuristic value.

### Strengths
As machine learning continues to be used to learn heuristic functions, admissibility guarantees become more important. The paper contains important theoretical analyses and proposed practical approaches for learning heuristic functions that are largely admissible.

When compared to compressed PDBs, both the CEA and CE have higher average heuristic values while maintaining similar compression rates. However, CE, has overestimation rates on the order of 1o^-3 to 10^-2. On the other hand, CEA has compression rates on the order of 10^-7 to 10^-5. The average heuristic value of CEA when compared to CE ranges from slightly lower to having the same value.

### Weaknesses
The paper discusses the benefits of admissibility, but it is not clear at what the cost of prioritizing admissibility on problem solving ability. If a neural network uses more of its representational capacity to prevent overfitting, then it may stand to reason that it may perform relatively poorly on higher cost-to-go values. This could make finding paths more time consuming in practice due to the lower overall heuristic value. Table 2 indicates this could be the case due to CEA having a slightly lower average heuristic value compared to CE.

### Questions
Does the theory presented in the paper say anything about drawbacks from prioritizing admissibility?

For Table 2, has using mean-squared error been tried?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 7

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a new loss function, Cross-Entropy Admissibility (CEA), for learning A* heuristics that are both strong and nearly admissible. The authors show that CEA explicitly favors underestimation while still concentrating probability on the true heuristic value, yielding more effective heuristics than compressed pattern databases in domains such as the Rubik’s Cube. 

On the theoretical side, the paper provides generalization guarantees for learned (including goal-dependent) heuristics in A*, deriving improved sample complexity bounds by analyzing inadmissibility along optimal paths. Overall, the work makes both practical and theoretical contributions to learning heuristics for search, bridging a gap between empirical performance and provable guarantees.

### Strengths
Novel loss function tailored to A*: The proposed CEA loss directly encodes admissibility preferences, addressing a key limitation of prior heuristic-learning approaches that treat over- and under-estimation symmetrically.

Strong empirical performance: Experiments (e.g., on Rubik’s Cube) demonstrate that the learned heuristics are nearly admissible while significantly outperforming compressed PDB heuristics of equal memory footprint, showing both practical effectiveness and scalability.

Theoretical contribution with improved sample complexity bounds.

### Weaknesses
Fairness of experimental comparison could be improved: While the comparison with compressed PDBs under equal memory budget is meaningful, it overlooks other relevant dimensions such as training cost, inference time, and tuning effort. In particular, CEA models were tuned, whereas the CE baseline shares hyperparameters without comparable tuning, which may bias results in favor of CEA.

CEA introduces additional tuning complexity

Limited domain diversity in evaluation: Experiments focus primarily on the Rubik’s Cube domain. It remains unclear how well the method generalizes to other search domains with different structure, branching factors, or cost models.

### Questions
The authors mention that the CE and CEA models use the same parameters. Could the authors clarify how these parameters were chosen? For a fair comparison, it may be helpful to tune both loss functions under the same hyperparameter-search budget. It would also be valuable to report the tuning ranges, the best-found values.

While comparing the NN heuristic with a compressed PDB under an equal memory budget is a meaningful and reasonable fairness criterion, may I ask whether memory alone is sufficient to capture the trade-offs between these approaches? Would it be possible to also comment on the differences in training cost and inference time, as these aspects are often relevant for practical deployment?

### Soundness
3

### Presentation
3

### Contribution
3
