# Egg-Sr: Embedding Symbolic Equivalence Into Symbolic Regression Via Equality Graph

Nan Jiang University of Texas - El Paso njiang@utep.edu Ziyi Wang, Yexiang Xue Purdue University
{wang4538,yexiang}@purdue.edu

## Abstract

Symbolic regression seeks to uncover physical laws from experimental data by searching for closed-form expressions, which is an important task in AI-driven scientific discovery. Yet the exponential growth of the search space of expression renders the task computationally challenging. A promising yet underexplored direction for reducing the search space and accelerating training lies in symbolic equivalence: many expressions, although syntactically different, define the same function - for example, log(x 21x 32), log(x 21) + log(x 32), and 2 log(x1) + 3 log(x2).

Existing algorithms treat such variants as distinct outputs, leading to redundant exploration and slow learning. We introduce EGG-SR, a unified framework that integrates symbolic equivalence into a class of modern symbolic regression methods, including Monte Carlo Tree Search (MCTS), Deep Reinforcement Learning (DRL), and Large Language Models (LLMs). EGG-SR compactly represents equivalent expressions through the proposed EGG module (via equality graphs), accelerating learning by: (1) pruning redundant subtree exploration in EGG- MCTS, (2) aggregating rewards across equivalent generated sequences in EGG- DRL, and (3) enriching feedback prompts in EGG-LLM. Theoretically, we show the benefit of embedding EGG into learning: it tightens the regret bound of MCTS and reduces the variance of the DRL gradient estimator. Empirically, EGG-SR consistently enhances a class of symbolic regression models across several benchmarks, discovering more accurate expressions within the same time limit.

Project page is at: https://nan-jiang-group.github.io/egg-sr.

## 1 Introduction

Symbolic regression aims to automatically discover physical knowledge from experimental data and has been widely used in scientific domains (Schmidt & Lipson, 2009; Udrescu & Tegmark, 2020; Cory-Wright et al., 2024; Yu & Wang, 2024; LaFollette et al., 2025). Many contemporary methods for symbolic regression formulate the search for optimal expressions as a sequential decision-making process. In literature, existing methods learn to predict the sequence of grammar rules (Sun et al., 2023), the traversal sequence of expression trees (Petersen et al., 2021; Kamienny et al., 2022), or executable strings that follow Python syntax (Shojaee et al., 2025; Zhang et al., 2025). This task remains computationally challenging due to its NP-hard nature (Virgolin & Pissis, 2022), that is, the search space of candidate expressions grows exponentially with the data dimension. A promising yet underexplored direction for reducing the search space and accelerating discovery is the integration of *symbolic equivalence* into learning algorithms. For example, these expressions log(x 21x 32), log(x 21) + log(x 32), and 2 log(x1) + 3 log(x2) all represent the same math function and are therefore *symbolically equivalent*. Ideally, a well-trained model would recognize such equivalence and assign identical goodness-of-fit, rewards, or losses to the corresponding predicted expressions (Allamanis et al., 2017), since these expressions produce identical functional outputs and attain the same prediction error on the dataset. In the literature, existing SR algorithms treat these expressions as distinct outputs, leading to redundant exploration of the search space and slow training. The main challenge of this direction is: how to represent symbolically-equivalent expressions and embed them into modern learning frameworks in a *unified and scalable* manner?

1 Since the number of equivalent variants grows exponentially with expression length, explicitly maintaining the set of equivalent expressions is not time and memory scalable. To mitigate this scalability challenge, a line of recent works introduced the Equality graph (e-graph), a data structure that compactly encodes the set of equivalent variants by storing shared sub-expressions only once (Nandi et al., 2021; Willsey et al., 2021; Kurashige et al., 2024). e-graphs have been applied to tasks, including program optimization (Barbulescu et al., 2024), dataset generation of equivalent expressions (Zheng et al., 2025). In genetic programming-based symbolic regression, e-graphs have been used for duplicate detection (de Franc¸a & Kronberger, 2025), expression simplification (de Franc¸a & Kronberger, 2023), and expression template pattern matching (de Franc¸a & Kronberger, 2025). Despite the empirical successes, we find that a unified framework that enables diverse symbolic regression algorithms to interact with e-graphs to accelerate learning has room for improvement. We present a unified framework, EGG-SR, that integrates symbolic equivalence into a class of learning algorithms via e-graphs. Our framework encompasses EGG-MCTS, EGG-DRL, and EGG-LLM. The core idea is to leverage EGG to efficiently sample equivalent variants of expressions predicted by SR algorithms and compute a new equivalence-aware learning objective. Specifically, (1) EGG- MCTS prunes redundant exploration over equivalent subtrees; (2) EGG-DRL aggregates rewards over equivalent expressions, stabilizing training; (3) EGG-LLM enriches the feedback prompt with multiple equivalent expressions to better guide next round predictions. Under mild theoretical assumptions, we show the benefit of embedding symbolic equivalence into learning: (1) EGG-MCTS
offers a tighter regret bound than standard MCTS (Sun et al., 2023), and (2) the gradient estimator of EGG-DRL exhibits a lower variance than that of standard DRL (Petersen et al., 2021). In experiments, we evaluate EGG-SR with several representative symbolic regression baselines across several challenging benchmarks. We demonstrate its advantages over existing approaches using EGG than without. EGG consistently improves performance across diverse frameworks, discovering more accurate expressions than baseline within a fixed time budget.

## 2 Preliminaries

Symbolic Expression. Let x = (x1*, . . . , x*n) denote input variables and c = (c1*, . . . , c*m) be coefficients. A symbolic expression ϕ connects these variables and coefficients using mathematical operators such as addition, multiplication, and logarithm. For example, ϕ = 3 log x1 + 2 log x2 is a symbolic expression composed of variables {x1, x2}, operators {+, log}, and coefficients {c1 = 3, c2 = 2}. In literature, symbolic expressions have been represented as binary trees (de Franc¸a
& Kronberger, 2025), pre-order traversal sequences of the binary tree (Petersen et al., 2021), topological traversal sequences of the expression graph (Kahlmeyer et al., 2024; Xiang et al., 2025), or sequences of production rules defined by a context-free grammar (Sun et al., 2023). To handle all symbolic objects in this work, a context-free grammar is adopted to represent symbolic expressions (Brence et al., 2021). The grammar is defined by a tuple ⟨V, Σ*, R, S*⟩ where (1) V is a set of *non-terminal* symbols representing arbitrary sub-expressions, i.e., V = {A}; (2) Σ is a set of *terminal* symbols, including input variables and coefficients, i.e., {x1, . . . , xn*} ∪ {*c}; (3) R is a set of production rules representing mathematical operations. For example, A → A × A denotes multiplication, and the semantics is to replace the left-hand side with the right-hand side; (4) S is the start symbol, typically set to S = A. Given a sequence of production rules, an expression is constructed by sequentially applying each rule to the *leftmost* nonterminal, starting from S. If the resulting string contains no nonterminals, it corresponds to a valid expression.

Figure 6 (in appendix) shows the expression construction with sequence (A → A×A, A → c, A → log(x1)). The first rule A → A × A expands the start symbol ϕ = A to ϕ = A × A. Applying the next rule A → c yields ϕ = c1×A. An index is assigned to the coefficient symbol c, to differentiate multiple coefficients. Finally, applying A → log(x1) to ϕ = c1 × A yields ϕ = c1 log(x1).

Symbolic Equivalence under a Rewrite System. Rewrite rules are widely used to simplify, rearrange, and reformulate expressions in tasks such as code optimization and automated theorem proving (Huet & Oppen, 1980; Nandi et al., 2021). A rewrite system provides a principled procedure for transforming expressions by replacing sub-expressions according to predefined patterns.

Formally, a rewrite rule riis written as "LHS ⇝ RHS", where the left-hand side (LHS) specifies a pattern to be *matched*, and the right-hand side (RHS) specifies the *substitution* applied upon a match.

Given a set of rewrite rules R = {r1, r2, r3*, . . .*}, the *symbolic equivalence* relation ≡R induced by R is defined as follows: for two symbolic expressions ϕ1 and ϕ2, ϕ1 ≡R ϕ2 if and only if ϕ1 ⇒∗ ϕ2 or ϕ2 ⇒∗ ϕ1, (1)
where ϕ1 ⇒∗ ϕ2 means that ϕ1 can be transformed into ϕ2 by applying a finite sequence of rewriting using R. In other words, two expressions are equivalent under R if one can be transformed into the other via repeated rewriting.

For example, consider the rewrite rule log(a × b) ⇝ log(a) + log(b) (denoted as r1), where a and b are placeholders for arbitrary sub-expressions. Since ϕ1 = log(x 3 1x 2 2) can be transformed into ϕ2 = log(x 3 1) + log(x 2 2) by applying r1 once, ϕ1 and ϕ2 are *symbolically equivalent* under r1.

In this work, the known mathematical identities listed in Table 3 (in the appendix) are encoded as rewrite rules in R. Section 3.1 applies these rewrite rules over the e-graph to generate a batch of symbolically equivalent expressions via matching (Figure 1b) and substitution (Figure 1c) operations. Implementation details of rewrite rules are provided in Appendix B.2.

Symbolic Regression (SR) posits that experimental data are generated by an underlying closedform expression, an assumption that is widely adopted across the sciences (Ma et al., 2022). Given a dataset D = {(xi, yi)}
N
i=1, the goal is to find an optimal expression ϕ
∗that minimizes the loss:

$$\phi^{*}\leftarrow\arg\operatorname*{min}_{\phi}\ {\frac{1}{N}}\sum_{i=1}^{N}\ell(\phi(\mathbf{x}_{i},\mathbf{c}),y_{i}),$$

where function ℓ measures the discrepancy between the prediction ϕ(xi, c) and the ground truth yi.

The coefficients c in ϕ are typically optimized on the training data D using numerical optimizers such as BFGS (Fletcher, 2000). This problem is NP-hard (Virgolin & Pissis, 2022), posing a major challenge for SR algorithms. Recent efforts to mitigate this challenge are reviewed in Section 4.

## 3 Methodology 3.1 Egg: Equality Graph For Grammar-Based Symbolic Expression

Enumerating all equivalent variants of a symbolic expression is combinatorially expensive. Storing these variants explicitly is time-consuming and memory-inefficient. To mitigate this scalability challenge, we adopt the recently proposed Equality graph (i.e, E-graph) data structure (Willsey et al., 2021; Waldmann et al., 2022), which compactly represents the set of equivalent expressions by sharing common subexpressions. We extend E-graphs to support grammar-based symbolic expressions, noted as EGG, facilitating unified integration with symbolic regression algorithms.

Definition. An *e-graph* consists of a collection of equivalence classes, called *e-classes*. Each e-class contains a set of *e-nodes* representing symbolically equivalent sub-expressions (Willsey et al., 2021).

Each e-node encodes a mathematical operation and references a list of child e-classes corresponding to its operands. Edges always point from an e-node to e-classes. Figure 1(d) shows an example e-graph. The color-highlighted part is an e-class (in dashed box)
containing two e-nodes (in solid boxes): A → log(A) and A → A + A. The two e-nodes represent logarithmic operation and addition, respectively. The e-node A → log(A) has a single outgoing edge to its child e-class A → A × A, because log(·) operator is unary.

Construction. In this work, an e-graph is initialized with a sequence of production rules, representing an input expression. Each rewrite rule (as defined in section 2) is applied by *matching* its left-hand side (LHS) pattern against the current e-graph, which involves traversing all e-classes to identify subexpressions that match LHS. For every successful match, new e-classes and e-nodes corresponding to the right-hand side (RHS) of the rule are created. A *merge* operation is applied to incorporate the new e-class with the matched e-class, thereby preserving the structure of known equivalences. This process, known as *equality saturation*, iteratively applies pattern matching and merging until either no further rules can be applied or a maximum number of iterations is reached.

Example 3.1. Figure 1 shows an example e-graph construction with the rewrite rule log(a × b) ⇝
log(a) + log(b), where a and b are placeholders for arbitrary sub-expressions. The e-graph is initialized with an expression ϕ = log(x 31x 22) in Figure 1(a). The LHS is *matched* against the colorhighlighted e-classes in Figure 1(b) with a = x 31and b = x 22. The *substitution* step constructs

![3_image_0.png](3_image_0.png)

e-classes and e-nodes that represent RHS, which are color-highlighted in Figure 1(c). The newly created e-class is *merged* with the matched e-class in Figure 1(d). The resulting e-graph represents two equivalent expressions: log(x 3 1x 2 2) and log(x 3 1) + log(x 2 2). The e-graph in Figure 1(d) saves memory by storing two sub-expressions x 3 1and x 2 2 only once. Additional EGG visualizations on more complex expressions are provided in Appendix D.1. Extraction. After the e-graph is saturated, an extraction step is performed to obtain K representative expressions (Goharshady et al., 2024). Because an e-graph encodes up to an exponential number of equivalent expressions, exhaustive enumeration is computationally infeasible. We therefore adopt two practical strategies: (1) *cost-based extraction*, which selects several simplified expressions by minimizing a user-defined cost function over operators and variables (de Franc¸a & Kronberger, 2025), and (2) *random-walk sampling*, which generates a batch of expressions by stochastically traversing the e-graph. A detailed explanation of extraction is provided in Appendix B.3.2. Interaction with SR algorithms. Prior research has leveraged e-graphs, based on the principle of Occam's razor, to obtain the most simplified and least-cost equivalent form (de Franca & Kronberger, 2025). In this study, however, equivalence-aware learning (detailed in Section 3.2) is encouraged by explicitly exposing SR algorithms to an extra subset of equivalent variants. The EGG module is primarily used to generate a subset of equivalent variants—i.e., expressions that represent the same mathematical function under a set of math identities. Given a sequence of production rules predicted by an SR algorithm, EGG constructs the e-graph and then performs extraction (via random-walk sampling) to return a batch of equivalent sequences. EGG is implemented for grammar-based symbolic expressions in pure Python, following the original e-graph paper (Willsey et al., 2021). Implementation of EGG is provided in Appendix B.

## 3.2 Embedding Symbolic Equivalence Into Symbolic Regression Via Egg

Embedding EGG **into Monte Carlo Tree Search.** MCTS (Sun et al., 2023; Ruan et al., 2025) maintains a search tree to explore an optimal sequence of decisions, here corresponding to a sequence of production rules. Each edge is labeled with a production rule, and each node is labeled by the sequence of edge labels from the root. By the grammar definition, this node label corresponds to a partially completed (or complete) expression. During learning, MCTS iterates the following four steps (Brence et al., 2021): (1) *Selection*. Starting from the root node, successively select the edge of node s (noted as a) with the highest Upper Confidence Bound for Trees (UCT) (Kocsis & Szepesvari, 2006): ´
UCT(*s, a*) = reward(*s, a*) + αplog(visits(s))/visits(*s, a*) (2)
where reward(*s, a*) is the average reward obtained by selecting edge a at node s, visits(s) is the number of visits to node s, and visits(*s, a*) is the number of times rule a has been selected at node s. The constant α (often set to 
√2 in theory) balances between exploration and exploitation. (2)

![4_image_0.png](4_image_0.png)

Expansion. When reaching a leaf node s, expand the search tree by generating its child nodes using all production rules. (3) *Simulation*. Perform several rollouts for each child to evaluate the average reward of node s. In each rollout, generate a valid expression ϕ by randomly applying production rules until completion. Then, estimate the optimal coefficients in ϕ and evaluate its reward. A common reward function is 1/(1 + NMSE(ϕ)). (4) *Backpropagation*. Update the reward estimates and visit counts for node s and all its parents up to the root node. After the final iteration, MCTS returns the expression with the highest reward encountered during training as its prediction. EGG*-based Backpropagation.*Our backpropagation strategy is motivated by transposition tables (Childs et al., 2008; Leurent & Maillard, 2020), which use a table to cache identical nodes (e.g., via hashing) in a search tree and share their statistics during training. This mechanism propagates information as if the search had visited all identical nodes. Such tables are effective in domains such as Go and Hearthstone, where nodes that are identical can be easily determined. In symbolic regression, however, two nodes may be identical only up to symbolic equivalence induced by rewrite rules, so a hashing-based transposition table is not directly applicable. To address this, EGG is used to identify equivalent paths and nodes. Concretely, the path—representing a partially completed expression—is first converted into an initial e-graph. The e-graph is then saturated by repeatedly applying the set of rewrite rules. From this saturated e-graph, we sample several distinct equivalent sequences and check if the tree contains corresponding paths. If so, we apply backpropagation to all of them. In this way, we avoid redundant exploration by sharing the rewards and visit counts of equivalent paths and nodes.

This modification in EGG-MCTS changes the interpretation of equation (2). visits(s) no longer counts how many times the specific tree node s appears on simulated paths; instead, it counts visits to any representative within the associated equivalence class. Conceptually, this mirrors the transposition table that shares statistics across identical tree nodes. In Theorem 3.1, we show that EGG-MCTS accelerates learning by reducing the search tree's effective branching factor relative to standard MCTS. It prevents redundant exploration of equivalent subtrees and concentrates sampling on genuinely distinct (and potentially near-optimal) paths. Example 3.2. Figure 2 shows an example execution pipeline of EGG-MCTS. Specifically, Figure 2(d) highlights two distinct root-to-node paths in the search tree:
Path 1 : (A → log(A), A → A × *A, A* → x1), Node s1: log(x1 × A). Path 2 : (A → A + *A, A* → log(A), A → x1, A → log(A)), Node s2: log(x1) + log(A).

Here, each path is a sequence of edge labels from the root to the leaf node si. Based on the grammar definition in section 2, node s1 corresponds to the partially completed expression log(x1 × A),
while node s2 represents log(x1) + log(A). The two nodes are equivalent under the rewrite rule

$${\mathrm{Node~}}s_{1}\colon\,\log(x_{1}\times A).$$ $${\mathrm{Node~}}s_{2}\colon\,\log(x_{1})+\log(A).$$
$$\begin{array}{l}{{1,A\to x_{1}),}}\\ {{1,A\to x_{1},A\to\log(A)),}}\end{array}$$

log(ab) ⇝ log a + log b. Consequently, their rewards, estimating the averaged goodness-of-fit of expressions on the training data, should be approximately equal:

$$\mathtt{reward}(s_{1},a)\approx\mathtt{reward}(s_{2},a),$$

reward(s1, a) ≈ reward(s2, a), ∀a ∈ the set of production rules Standard MCTS would explore the subtrees rooted at s1 and s2 independently, because it is unaware of their equivalence. This results in redundant computation and slows down learning. With EGG- MCTS, the visit counts and reward estimates of both paths are updated simultaneously, eliminating the need for extra iterations on the orange leaf in Figure 2(d). See Example B.1 for the case of other rewrite rules, e.g., sin2(a) + cos2(a) ⇝ 1 and a/a ⇝ 1.

Embedding EGG **into Deep Reinforcement Learning.** DRL typically employs a neural sequential decoder to predict an expression by sampling a sequence of production rules from the model distribution. The reward assigns higher values to expressions that better fit the training data (Petersen et al., 2021; Landajuela et al., 2022; Jiang et al., 2024). The pipeline of DRL and EGG-DRL is presented in Figure 8. At every step, the sequential decoder samples the next production rule from a distribution over all available rules, conditioned on the previously generated sequence. The decoder thus induces a distribution pθ(τ ) over the rule sequence τ . The reward function is typically defined as reward(τ ) = 1/(1 + NMSE(ϕ)), where ϕ is the expression constructed by τ following grammar definition (in section 2). The learning objective is to maximize the expected reward of generated expressions on the training data: Eτ∼pθ[reward(τ )], whose gradient is Eτ∼pθ[reward(τ )∇θ log pθ(τ )]. In practice, the gradient is approximated via Monte Carlo. Sampling N sequences {τ1*, . . . , τ*N } from the decoder, the policy gradient estimator computes:

$$g(\theta)\approx{\frac{1}{N}}\sum_{i=1}^{N}(\texttt{reward}(\tau_{i})-b)\nabla_{\theta}\log p_{\theta}(\tau_{i}),$$

where b is a baseline used to reduce variance (Weaver & Tao, 2001). Recent work (Petersen et al., 2021) further proposes using a top-quantile subset of samples, rather than the sample mean. EGG*-based Policy Gradient Estimator.* For each sampled sequence τi, we construct an e-graph that compactly encodes all of its equivalent expressions. From this e-graph, we sample K −1 equivalent sequences {τ
(2)
i*, . . . , τ*
(K)
i}. We then revise the policy-gradient estimator as

$$g_{\texttt{egg}}(\theta)\approx\frac{1}{N}\sum_{i=1}^{N}\left(\texttt{reward}(\tau_{i})-b^{\prime}\right)\nabla_{\theta}\log\left[\sum_{k=1}^{K}p_{\theta}(\tau_{i}^{(k)})\right],\tag{4}$$
$$(3)$$

where τ
(1)
iis the original sequence τi and b
′is the corresponding baseline, and PK
k=1 pθ(τ
(k)
i)
aggregates the probabilities of all equivalent sequences that share the same reward. In Theorem 3.2, we show that EGG improves DRL training by yielding a lower-variance gradient estimator than standard DRL (Petersen et al., 2021). Embed EGG **into Large-Language Model.** LLM is applied to search for optimal symbolic expressions with prompt tuning (Merler et al., 2024; Shojaee et al., 2025). The procedure consists of three key steps: (1) *Hypothesis Generation*: The LLM generates multiple candidate expressions based on a prompt describing the problem background and the definitions of each variable. (2) Data-Driven Evaluation: Each candidate expression is evaluated based on its fitness on the training dataset. (3) Experience Management: In subsequent iterations, the LLM receives feedback in the form of previously predicted expressions and their corresponding fitness scores, allowing it to refine future generations. High-fitness expressions are retained and updated over multiple rounds of iteration.

EGG*-based Feedback Prompt.* Since LLMs typically generate Python functions rather than symbolic expressions directly, we introduce a wrapper that parses the generated Python code into symbolic expressions. These expressions are then transformed into e-graphs using a set of rewrite rules. From each e-graph, we extract semantically equivalent expressions and summarize them into a similar feedback message, which is incorporated into the prompt for the next round. This augmentation enables the LLM to observe a richer set of functionally equivalent expressions, potentially improving the quality and accuracy of predictions in future iterations.

## 3.3 Connection To Existing Methods

Prior work has explored alternative expression representations based on layer-wise symbolic networks (SymNet) (Sahoo et al., 2018; Li et al., 2024), which are not directly compatible with our grammar-based formulation. Recent studies on SymNet further show that many learned coefficients can be aggregated or merged (Wu et al., 2024). Extending this notion of coefficient equivalence to sub-expression equivalence within SymNet remains an interesting open problem. For DRL-based approaches, several extensions of the original method (Petersen et al., 2021) have been proposed, including (Mundhenk et al., 2021; Landajuela et al., 2022). An important open question is whether symbolic-equivalence can be integrated into these extensions in a compatible and effective manner, and whether doing so can further improve overall performance.

Finally, Kamienny et al. (2022); Shojaee et al. (2023) encode data directly with a Transformer and predict an expression traversal sequence end-to-end under a cross-entropy objective. A natural way to incorporate EGG is to use it during training to generate multiple equivalent, correct target sequences. How to best leverage EGG at inference time, however, remains an open problem.

## 3.4 Theoretical Justification On Egg-Sr Accelerating Learning

Theorem 3.1 shows that EGG-MCTS achieves an asymptotically tighter regret bound than standard MCTS, as the effective branching factor satisfies κ∞ ≤ κ. Intuitively, by identifying symbolically equivalent nodes and sharing their search statistics, EGG prevents redundant exploration of equivalent subtrees and concentrates sampling on genuinely distinct (and potentially near-optimal) paths. After many iterations, EGG-MCTS concentrates more quickly on the near-optimal region of the search space, which is captured by a smaller effective branching factor. Also, Theorem 3.2 shows that embedding EGG into DRL produces an unbiased gradient estimator while strictly reducing gradient variance, ensuring more stable and efficient policy updates. Theorem 3.1. Consider embedding EGG into the MCTS framework. Given Definitions 1 and 3 (in appendix), let n denote the total number of learning iterations, γ ∈ (0, 1) the discount factor of the corresponding Markov decision process, κ be the near-optimal branching factor of standard MCTS, and κ∞ the corresponding branching factor of EGG-MCTS. Then the regret bounds satisfy

$\mathbf{regret}(n)=\widetilde{\mathcal{O}}\left(n^{-\frac{\log(1/\gamma)}{\log n}}\right),\qquad\mathbf{regret}_{\mathbf{reg}}(n)=\widetilde{\mathcal{O}}\left(n^{-\frac{\log(1/\gamma)}{\log n}}\right),\qquad\text{with}\kappa_{\infty}\leq\kappa.$
Proof Sketch. Leurent & Maillard (2020) analyze MCTS on a graph obtained by merging identical tree nodes and sharing their statistics. Their analysis *unrolls* the graph into a tree that contains all graph-traversable paths. The search tree in EGG-MCTS behaves identically to the unrolled tree. Our final results follow their regret analysis on the unrolled tree. A detailed proof is in Appendix A.2.

Theorem 3.2. Consider embedding EGG into the DRL framework. Let τ ∼ pθ denote trajectories sampled from the distribution pθ, and consider the two estimators defined in Equations (3) and (4).

(1) Unbiasedness. The expectation of the standard estimator g(θ) equals that of the EGG-based estimator gegg(θ): Eτ∼pθ
-g(θ)= Eτ∼pθ
-gegg(θ). *(2) Variance Reduction.* The variance of the proposed estimator is smaller than that of the standard estimator:
Varτ∼pθ
-gegg(θ)≤ Varτ∼pθ
-g(θ).

Proof Sketch. For (1), unbiasedness can be obtained by expanding the definitions of g(θ) and gegg(θ). For (2), the key observation is that EGG groups together equivalent trajectories that share the same reward. Averaging over sequences with identical rewards reduces within-group variability, which yields a smaller variance. A full proof is provided in Appendix A.3.

## 4 Related Works

Knowledge-Guided Scientific Discovery. Recent efforts have explored incorporating physical and domain-specific knowledge to accelerate symbolic discovery. AI-Feynman (Udrescu & Tegmark, 2020; Udrescu et al., 2020; Keren et al., 2023; Cornelio et al., 2023) constrained the search space to expressions that exhibit compositionality, additivity, and generalized symmetry. Similarly, Tenachi et al. (2023) encoded physical unit constraints into learning to eliminate physically impossible solutions. Other works further constrained the search space by integrating user-specified hypotheses and prior knowledge, offering a guided approach to symbolic regression (Bendinelli et al., 2023; Kamienny, 2023; Shojaee et al., 2025; Taskin et al., 2026; Zhang et al., 2025). Our EGG-SR presents a new idea in knowledge-guided learning that is orthogonal to existing approaches. Symbolic Equivalence is a central concept in program synthesis and mathematical reasoning (Willsey et al., 2021). In SQL query optimization, it rewrites queries into time-efficient forms (Barbulescu et al., 2024). In hardware synthesis, it supports cost-aware rewrites such as optimized matrix multiplication (Ustun et al., 2022). In formal methods, it accelerates automated theorem proving through normalization and equivalence checking (Kurashige et al., 2024). In mathematical reasoning, it is used to generate paraphrases of math expressions (Zheng et al., 2025). In symbolic regression, de Franc¸a & Kronberger (2023) leverages e-graphs to mitigate overparameterization in candidate expressions. de Franc¸a & Kronberger (2025); de Franca & Kronberger (2025) further incorporates e-graphs into genetic programming (GP) to detect and eliminate redundant individuals, encouraging GP to explore novel expressions. A recent follow-up work (de Franc¸a & Kronberger, 2025) provides a richer API for interacting with GP. This study advances existing work by offering a unified interface for encoding known mathematical equalities as e-graphs, enabling equivalence-aware learning across several modern symbolic regression algorithms, together with theoretical guarantees. Equivalence-aware Learning. Equivalence and symmetry have long been recognized as crucial for improving efficiency in search and learning. In MCTS, transposition tables exploit equivalence by merging nodes that represent the same underlying state, avoiding redundant rollouts and accelerating convergence (Childs et al., 2008). More recent extensions explicitly leverage symmetries to prune symmetric branches of the search space (Saffidine et al., 2012; Leurent & Maillard, 2020). In reinforcement learning, symmetries in the state–action space have been used to accelerate convergence (Grimm et al., 2020). LLMs also benefit from equivalence-awareness, particularly in code generation (Sharma & David, 2025).

## 5 Experiments

We show that (1) EGG enhances existing learning algorithms in discovering expressions with smaller Normalized MSEs. (2) In case studies, EGG consistently exhibits both time and space efficiency. The detailed experimental setups and datasets used in each comparison, are provided in Appendix C.

## 5.1 Overall Benchmarks

Impact of EGG **on MCTS.** We conduct two analyses to evaluate the impact of integrating EGG into standard MCTS: (1) the median normalized MSE of the TopK (K = 10) expressions identified at the end of training, and (2) The growth of the search tree, measured by the number of explored nodes over learning iterations. Table 1 shows that EGG-MCTS consistently discovers expressions with lower normalized quantile scores compared to standard MCTS. The dataset is selected from Jiang & Xue (2023) as the expressions contain sin, cos operators, which contain many symbolic-equivalence variants.

Figure 3(Left) illustrates that EGG-
MCTS maintains a broader and deeper search tree, indicating exploration of a larger and more diverse search space. Across various datasets, augmenting MCTS with

![7_image_0.png](7_image_0.png)

| Noiseless Setting   | Noisy Setting   |           |           |           |           |           |           |       |
|---------------------|-----------------|-----------|-----------|-----------|-----------|-----------|-----------|-------|
| (2, 1, 1)           | (3, 2, 2)       | (4, 4, 6) | (5, 5, 5) | (2, 1, 1) | (3, 2, 2) | (4, 4, 6) | (5, 5, 5) |       |
| EGG-MTCS            | <1E-6           | <1E-6     | 0.006     | 0.009     | 0.005     | 0.012     | 0.091     | 0.121 |
| MTCS                | 0.006           | 0.033     | 0.144     | 0.147     | 0.015     | 0.007     | 0.138     | 0.150 |
| EGG-DRL             | 0.020           | 0.161     | 2.381     | 2.168     | 0.07      | 0.35      | 5.09      | 5.67  |
| DRL                 | 0.030           | 0.277     | 2.990     | 2.903     | 0.09      | 0.44      | 2.46      | 14.44 |

Table 1: On Trigonometric datasets, median NMSE values of the best-predicted expressions found by all the algorithms. The 3-tuples at the top (·, ·, ·) indicate the number of free variables, singular terms, and cross terms in the ground-truth expressions generating the dataset. The set of operators is {sin, cos, +, −, ×}. The best result in each column is underlined.

| measured by the NMSE metric. The best result in each column is underlined. Oscillation I Oscillation II Bacterial growth   | Stress-Strain   |        |        |         |        |        |        |        |
|----------------------------------------------------------------------------------------------------------------------------|-----------------|--------|--------|---------|--------|--------|--------|--------|
| IID↓                                                                                                                       | OOD↓            | IID↓   | OOD↓   | IID↓    | OOD↓   | IID↓   | OOD↓   |        |
| EGG-LLM (GPT3.5)                                                                                                           | <1E-6           | 0.0004 | <1E-6  | <1E-6   | 0.0121 | 0.0198 | 0.0202 | 0.0419 |
| LLM-SR (GPT-3.5)                                                                                                           | <1E-6           | 0.0005 | <1E-6  | 3.81E-5 | 0.0214 | 0.0264 | 0.0210 | 0.0516 |
| EGG-LLM (Mistral)                                                                                                          | <1E-6           | 0.0002 | 0.0021 | 0.0114  | 0.0101 | 0.0107 | 0.0133 | 0.0754 |
| LLM-SR (Mistral)                                                                                                           | <1E-6           | 0.0002 | 0.0030 | 0.0291  | 0.0026 | 0.0037 | 0.0162 | 0.0946 |

EGG improves symbolic expression accuracy. This improvement is primarily due to the effectiveness of our rewrite rules, which cover a rich set of trigonometric identities and enable efficient exploration of symbolic variants in trigonometric expression spaces. Impact of EGG **on DRL.** Table 1 reports the median NMSE values of the best-predicted expressions discovered by EGG-DRL and standard DRL, under identical experiment settings. Expressions returned by EGG-DRL achieve a smaller NMSE value on noiseless and noisy settings. It shows that embedding EGG into DRL helps to discover expressions with better NMSE. In Figure 3 (Right), we plot the estimated objective, defined as R(τi) log pθ(τi) where each trajectory τiis sampled from the sequential decoder with probability pθ(τi) (see Equation 3). We plot the empirical mean and standard deviation of this objective over training iterations. The observed reduction in variance is primarily due to the symbolic variants generated via the e-graph, which enable averaging over multiple equivalent expressions and thus yield more stable gradients. Impact of EGG **on LLM.** Following the dataset and experimental setup from the original paper (Shojaee et al., 2025), we summarize the results in Table 2. The result of LLM-SR directly uses the reported result in Shojaee et al. (2025). The results show that integrating EGG enables the LLM to discover higher-quality expressions under the same experimental conditions, as with richer feedback prompts that incorporate equivalent expressions generated by EGG.

## 5.2 Case Analysis

Space Efficiency of EGG. We evaluate the space efficiency of the e-graph representation in comparison to a traditional array-based approach. We benchmark the memory consumption of storing all equivalent variants of input expressions under two settings: (1) ϕ = log(x1 ×*. . .*×xn), using the logarithmic identity log(ab) ⇝ log a+log b, and (2) ϕ = sin(x1+*. . .*+xn), using the trigonometric identity sin(a + b) ⇝ sin(a) cos(b) + sin(b) cos(a). Both settings yield 2 n−1equivalent variants.

The array-based method explicitly stores each expression variant as a unique sequence, leading to exponential memory growth. In contrast, the e-graph compactly encodes multiple equivalent expressions by sharing common sub-expressions. Figure 4 reports memory consumption as a function of the number of variables n. The results show that e-graphs use substantially less memory than the array-based representation. We also provide additional visualizations of the constructed e-graphs for n = 2, 3, 4: case (1) in Appendix Figure 14 and case (2) in Appendix Figure 15. It visualizes two representative e-graphs, illustrating how shared sub-expressions are stored once and reused across many variants, which underlies the space efficiency of EGG.

![9_image_1.png](9_image_1.png)

![9_image_0.png](9_image_0.png)

Time Efficiency of EGG **with DRL.** As shown in Figure 5, we benchmark the runtime of the four main computations in EGG-DRL on the selected "sincos(3,2,2)" dataset: (1) sampling sequences of rules from the sequential decoder, (2) fitting coefficients in symbolic expressions to the training data, (3) generating equivalent expressions via EGG, and (4) computing the loss, gradients, and updating the neural network parameters. We consider two settings for the sequential decoder: a 3-layer LSTM with hidden dimension 128, and a decoder-only Transformer with 6 attention heads and hidden dimension 128. The EGG module contributes minimal computational overhead relative to more expensive steps such as coefficient fitting and neural network parameter updates. The runtime of EGG depends on the size of the rewrite-rule set. As more rules are included, the e-graphs maintain increasingly large sets of equivalent expressions. This highlights the practicality of incorporating EGG into DRL-based symbolic regression frameworks.

Additional Visualizations of E-graph Construction. To further demonstrate the effectiveness of the proposed EGG module, we present additional visualizations of e-graph construction generated with our API on 7 selected complex expressions from the Feynman dataset (Udrescu & Tegmark, 2020). Each visualization highlights a different set of rewrite rules (see Appendix D.2). These examples further illustrate that EGG can simplify and transform complex scientific expressions in practical settings.

## 6 Conclusion

In this paper, we introduced EGG-SR, a unified framework that integrates symbolic equivalence into symbolic regression through equality graphs (e-graphs) to accelerate the discovery of optimal expressions. Our theoretical analysis establishes the advantages of EGG-MCTS over standard MCTS and EGG-DRL over conventional DRL algorithms. Extensive experiments further demonstrate that EGG consistently enhances the ability of existing methods to uncover high-quality governing equations from experimental data. Currently, many scientific publications use GP-based symbolic regression due to its ease of use. In future work, we plan to extend our more sophisticated solver, EGG-SR, to scientifically grounded problem settings, improving the community's computational toolkit. Ethics Statement. All authors have read and commit to adhering to the ICLR Code of Ethics. This work uses only publicly available datasets and open-source models, and does not involve human subjects or human subjects data. Reproducibility Statement. Appendix B describes the proposed EGG module, and the Appendix A.3 and A.2 include detailed proofs of theoretical justification. Appendix C gives the experimental setting. Appendix D collects extra experimental results. Acknowledgements. We thank the reviewers for their constructive feedback, as well as Fabricio Olivetti de Franc¸a for his public comments. This research was supported by TACC (CCR25054) and the U.S. Department of Energy, Office of Fusion Energy Sciences (DE-SC0024583).

## References

Miltiadis Allamanis, Pankajan Chanthirasegaran, Pushmeet Kohli, and Charles Sutton. Learning continuous semantic representations of symbolic expressions. In *ICML*, volume 70, pp. 80–88. PMLR, 2017.

George-Octavian Barbulescu, Taiyi Wang, Zak Singh, and Eiko Yoneki. Learned graph rewriting with equality saturation: A new paradigm in relational query rewrite and beyond. *CoRR*, abs/2407.12794, 2024.

Zachary Bastiani, Robert M Kirby, Jacob Hochhalter, and Shandian Zhe. Diffusion-based symbolic regression. *arXiv preprint arXiv:2505.24776*, 2025.

Tommaso Bendinelli, Luca Biggio, and Pierre-Alexandre Kamienny. Controllable neural symbolic regression. In *ICML*, volume 202, pp. 2063–2077. PMLR, 2023.

Jure Brence, Ljupco Todorovski, and Saso Dzeroski. Probabilistic grammars for equation discovery.

Knowl. Based Syst., 224:107077, 2021.

George Casella and Christian P. Robert. Rao-blackwellisation of sampling schemes. *Biometrika*, 83
(1):81–94, 1996.

William G. La Cava, Patryk Orzechowski, Bogdan Burlacu, Fabr´ıcio Olivetti de Franc¸a, Marco Virgolin, Ying Jin, Michael Kommenda, and Jason H. Moore. Contemporary symbolic regression methods and their relative performance. In *NeurIPS Datasets and Benchmarks*, volume 1, 2021.

Benjamin E. Childs, James H. Brodeur, and Levente Kocsis. Transpositions and move groups in monte carlo tree search. In CIG, pp. 389–395. IEEE, 2008.

Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. *arXiv preprint arXiv:1412.3555*, 2014.

Cristina Cornelio, Sanjeeb Dash, Vernon Austel, Tyler R Josephson, Joao Goncalves, Kenneth L
Clarkson, Nimrod Megiddo, Bachir El Khadir, and Lior Horesh. Combining data and theory for derivable scientific discovery with ai-descartes. *Nature Communications*, 14(1):1777, 2023.

Ryan Cory-Wright, Cristina Cornelio, Sanjeeb Dash, Bachir El Khadir, and Lior Horesh. Evolving scientific discovery by unifying data and background knowledge with ai hilbert. Nature Communications, 15(1):5922, 2024.

Fabr´ıcio Olivetti de Franc¸a and Gabriel Kronberger. Reducing overparameterization of symbolic regression models with equality saturation. In *GECCO*, pp. 1064–1072. ACM, 2023.

Fabr´ıcio Olivetti de Franc¸a and Gabriel Kronberger. Improving genetic programming for symbolic regression with equality graphs. In *GECCO*, pp. 989–998. ACM, 2025.

Fabricio Olivetti de Franca and Gabriel Kronberger. Equality graph assisted symbolic regression.

arXiv preprint arXiv:2511.01009, 2025.

Fabricio Olivetti de Franc¸a and Gabriel Kronberger. reggression: an interactive and agnostic tool for the exploration of symbolic regression models. In *GECCO*, pp. 4–12. Association for Computing Machinery, 2025.

Wei Deng, Qi Feng, Georgios Karagiannis, Guang Lin, and Faming Liang. Accelerating convergence of replica exchange stochastic gradient MCMC via variance reduction. In *ICLR*. OpenReview.net, 2021.

Roger Fletcher. *Practical methods of optimization*. John Wiley & Sons, 2000. Steven Ganzert, Josef Guttmann, Daniel Steinmann, and Stefan Kramer. Equation discovery for model identification in respiratory mechanics of the mechanically ventilated human lung. In Discovery Science, volume 6332, pp. 296–310. Springer, 2010.

Amir Kafshdar Goharshady, Chun Kit Lam, and Lionel Parreaux. Fast and optimal extraction for sparse equality graphs. *Proc. ACM Program. Lang.*, 8(OOPSLA2):2551–2577, 2024.

Klaus Greff, Rupesh K Srivastava, Jan Koutn´ık, Bas R Steunebrink, and Jurgen Schmidhuber. Lstm: ¨
A search space odyssey. *IEEE transactions on neural networks and learning systems*, 28(10): 2222–2232, 2016.

Christopher Grimm, Andre Barreto, Satinder Singh, and David Silver. The value equivalence prin- ´
ciple for model-based reinforcement learning. In *NeurIPS*, volume 33, pp. 5541–5552, 2020.

Gerard Huet and Derek C Oppen. Equations and rewrite rules: A survey. ´ *Formal Language Theory*,
pp. 349–405, 1980.

Nan Jiang and Yexiang Xue. Symbolic regression via control variable genetic programming. In ECML/PKDD, pp. 178–195. Springer, 2023.

Nan Jiang, Md. Nasim, and Yexiang Xue. Vertical symbolic regression via deep policy gradient. In IJCAI, pp. 5891–5899. ijcai.org, 2024.

Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In *NeurIPS*, pp. 315–323, 2013.

Paul Kahlmeyer, Joachim Giesen, Michael Habeck, and Henrik Voigt. Scaling up unbiased searchbased symbolic regression. In *IJCAI*, pp. 4264–4272. ijcai.org, 2024.

Pierre-Alexandre Kamienny. Efficient adaptation of reinforcement learning agents: from model-free exploration to symbolic world models. Theses, Sorbonne Universite, October 2023. ´
Pierre-Alexandre Kamienny, Stephane d'Ascoli, Guillaume Lample, and Franc¸ois Charton. End-to- ´
end symbolic regression with transformers. In *NeurIPS*, volume 35, pp. 10269–10281, 2022.

Pierre-Alexandre Kamienny, Guillaume Lample, Sylvain Lamprier, and Marco Virgolin. Deep generative symbolic regression with monte-carlo-tree-search. In *ICML*, volume 202. PMLR, 2023. Liron Simon Keren, Alex Liberzon, and Teddy Lazebnik. A computational framework for physicsinformed symbolic regression with straightforward integration of domain knowledge. *Scientific* Reports, 13(1):1249, 2023.

Levente Kocsis and Csaba Szepesvari. Bandit based monte-carlo planning. In ´ *ECML*, volume 4212 of *Lecture Notes in Computer Science*, pp. 282–293. Springer, 2006.

Cole Kurashige, Ruyi Ji, Aditya Giridharan, Mark Barbone, Daniel Noor, Shachar Itzhaky, Ranjit Jhala, and Nadia Polikarpova. Cclemma: E-graph guided lemma discovery for inductive equational proofs. *Proc. ACM Program. Lang.*, 8(ICFP):818–844, 2024.

Kyle J. LaFollette, Janni Yuval, Roey Schurr, David Melnikoff, and Amit Goldenberg. Data-driven equation discovery reveals nonlinear reinforcement learning in humans. *Proc. Natl. Acad. Sci.*,
122(31), 2025.

Mikel Landajuela, Chak Shing Lee, Jiachen Yang, Ruben Glatt, Claudio P. Santiago, Ignacio Ar- ´
avena, Terrell Nathan Mundhenk, Garrett Mulcahy, and Brenden K. Petersen. A unified framework for deep symbolic regression. In *NeurIPS*, volume 35, pp. 33985–33998, 2022.

Edouard Leurent and Odalric-Ambrym Maillard. Monte-carlo graph search: the value of merging similar states. In *ACML*, volume 129, pp. 577–592. PMLR, 2020.

Wenqiang Li, Weijun Li, Lina Yu, Min Wu, Linjun Sun, Jingyi Liu, Yanjie Li, Shu Wei, Yusong Deng, and Meilan Hao. A neural-guided dynamic symbolic network for exploring mathematical expressions from data. In *ICML*, volume 235, pp. 28222–28242. PMLR, 2024.

He Ma, Arunachalam Narayanaswamy, Patrick Riley, and Li Li. Evolving symbolic density functionals. *Science Advances*, 8(36), 2022. Matteo Merler, Katsiaryna Haitsiukevich, Nicola Dainese, and Pekka Marttinen. In-context symbolic regression: Leveraging large language models for function discovery. In *ACL (Student* Research Workshop), pp. 589–606. Association for Computational Linguistics, 2024.