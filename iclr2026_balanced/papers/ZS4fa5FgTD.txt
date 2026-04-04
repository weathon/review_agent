# LEARNING FOR DYNAMIC COMBINATORIAL OPTIMIZA## TION WITHOUT TRAINING DATA


**Anonymous authors**
Paper under double-blind review


ABSTRACT


We introduce DyCO-GNN, a novel unsupervised learning framework for Dynamic
Combinatorial Optimization that requires no training data beyond the problem
instance itself. DyCO-GNN leverages structural similarities across time-evolving
graph snapshots to accelerate optimization while maintaining solution quality. We
evaluate DyCO-GNN on dynamic maximum cut, maximum independent set, and
the traveling salesman problem across diverse datasets of varying sizes, demonstrating its superior performance under tight and moderate time budgets. DyCO-GNN
consistently outperforms the baseline methods, achieving high-quality solutions
up to 3–60x faster, highlighting its practical effectiveness in rapidly evolving
resource-constrained settings.


1 INTRODUCTION


Combinatorial optimization (CO) lies at the heart of many critical scientific and industrial problems (Papadimitriou & Steiglitz, 1982). Since most CO problems are NP-hard, solving large-scale
instances is computationally prohibitive. Traditionally, solving such problems has relied on exact
solvers, heuristics, and metaheuristics, whose design requires problem-specific insights. Recent
advances in machine learning, particularly graph neural networks (GNNs), have opened new avenues
for learning heuristics in a data-driven manner (Joshi et al., 2019; 2022; Gasse et al., 2019; Hudson
et al., 2022; Karalias & Loukas, 2020; Bello et al., 2016; Khalil et al., 2017). GNNs have emerged as
a powerful framework for learning over relational and structured data (Defferrard et al., 2016; Kipf
& Welling, 2017; Hamilton et al., 2017; Veliˇckovi´c et al., 2018), with an inductive bias particularly
suited for representing the underlying graph structures inherent in many CO problems.


Most existing approaches applying machine learning to CO are learning-heavy, requiring training
on a large set of problem instances to learn _heuristics that generalize_ to test instances. Training can
take hours or even days on multiple GPUs (Wang & Li, 2023; Li et al., 2024). Schuetz et al. (2022)
started another line of research that aims to develop an unsupervised learning (UL) method that learns
_instance-specific heuristics_ . Their method, PI-GNN, directly applies a GNN to the problem instance
of interest and optimizes a CO objective. Under this design, no explicit “training” is involved; the
runtime is the time it takes for the model to converge on each problem instance. Prior work has shown
that problems such as maximum cut (MaxCut) and maximum independent set (MIS) on graphs with
thousands of nodes can be solved by PI-GNN-based methods in minutes (Heydaribeni et al., 2024;
Ichikawa, 2024).


While significant progress has been made in learning for static CO, many real-world problems are
inherently dynamic, involving inputs or constraints that evolve over time (Yang et al., 2012; Zhang
et al., 2021). In such settings, decisions must be updated continually, and practical algorithms must
be efficient. A simple approach is to treat each snapshot of the problem instance as a static problem
and solve it from scratch. However, significant overlap often exists in the node and edge sets across
snapshots. Leveraging information from previous snapshots can improve both runtime and solution
quality. For example, in MaxCut, the previous solution might serve as a good starting point after edge
additions. Our proposed approach leverages such overlaps.


Specifically, we focus on learning for dynamic combinatorial optimization (DCO) and aim to advance
the PI-GNN line of research. We present DyCO-GNN for DCO. To the best of our knowledge, our
work is the first to apply machine learning to DCO problems. Our main contributions are as follows.


1


- We propose DyCO-GNN, a general UL-based optimization method for DCO that requires
_no training data_ except the problem instance of interest.

    - We empirically validate the applicability of DyCO-GNN to various CO problems under
dynamic settings, including MaxCut, MIS, and the traveling salesman problem (TSP).

    - We demonstrate that DyCO-GNN achieves superior solution quality when the time budget
is limited and outperforms the static counterpart with up to **3–60x** speedup.


2 RELATED WORK


Machine learning for static CO has predominantly relied on supervised learning approaches (Joshi
et al., 2019; 2022; Vinyals et al., 2015; Gasse et al., 2019; Sun & Yang, 2023; Hudson et al., 2022; Li
et al., 2023; 2024). These methods train models to predict high-quality solutions given a large dataset
of problem instances paired with optimal or near-optimal labels. However, generating such labels is
computationally expensive, especially for large-scale instances, making these methods less practical
for many real-world applications.


To address this limitation, UL and reinforcement learning (RL) approaches have emerged as promising
alternatives (Bello et al., 2016; Khalil et al., 2017; Kool et al., 2019; Karalias & Loukas, 2020; Qiu
et al., 2022; Toenshoff et al., 2021; Tönshoff et al., 2023; Wang & Li, 2023; Sanokowski et al., 2023).
These methods aim to learn optimization strategies directly from instance structures or reward signals
without requiring labeled data. While they alleviate the need for ground-truth solutions, most UL and
RL-based techniques still involve extensive offline training over large datasets to develop _heuristics_
_that_ _generalize_ . Training such models can be time-consuming, often requiring hours or days on
high-performance hardware (Wang & Li, 2023; Li et al., 2024).


A different paradigm was introduced by Schuetz et al. (2022), who proposed PI-GNN—an unsupervised framework that learns _instance-specific heuristics_ by directly optimizing the CO objective on a
single instance. This avoids any offline training and allows the method to be tailored to each problem
instance during inference. PI-GNN combines a learnable embedding layer with a GNN and has been
shown to achieve competitive performance across tasks such as MaxCut and MIS. Subsequent works
have improved its solution quality and extended its design to incorporate higher-order relational
reasoning (Heydaribeni et al., 2024; Ichikawa, 2024), achieving strong results even on large-scale
graph instances.


Despite advances in instance-specific optimization, existing efforts have focused exclusively on static
CO. The challenge of adapting learned heuristics efficiently to dynamic settings remains, to our
knowledge, unexplored. Our work builds upon the PI-GNN line of research to propose new methods
for DCO, where we aim to reuse and adapt learned solutions across temporally evolving problem
instances without requiring optimization from scratch. We review the optimization theory literature
on DCO, which is complementary to our learning-based approach, in Appendix A.


3 PRELIMINARIES


3.1 STATIC CO: PI-GNN AND QUADRATIC BINARY UNCONSTRAINED OPTIMIZATION (QUBO)


Consider a graph _Gp_ = ( _Vp, Ep, wp_ ) with node set _Vp_ = _{_ 1 _,_ 2 _, . . ., np}_, edge set _Ep_, and edge
weights _wp_ representing or inherent in a CO problem instance _p_ . PI-GNN is a general UL framework
for static CO problems based on QUBO (Lucas, 2014; Glover et al., 2018; Djidjev et al., 2018). A
QUBO problem is defined by:


min _ℓ_ ( _x_ ; _Qp_ ) = _x_ _[T]_ _Qpx_ (1)
_x_

where _x ∈{_ 0 _,_ 1 _}_ _[N]_ is a binary vector with _N_ components, and _Qp_ _∈_ R _[N]_ _[×][N]_ is a symmetric matrix
encoding the cost coefficients, obtained based on the graph _Gp_ . There are no explicit constraints, with
all constraints incorporated implicitly via the structure of the Q matrix. Many CO problems, such as
MaxCut, MIS, and TSP, can be reformulated as QUBO instances, making it a universal encoding
framework for a wide class of problems. For a given static CO problem instance, PI-GNN learns to
find a solution via _A_ ( _Gp_ ; _θ_ ), where _θ_ is the set of learnable parameters. Since the input graph has no
node features, PI-GNN randomly initializes learnable embeddings for the nodes and feeds them to a


2


GNN. It then optimizes a differentiable QUBO objective in the form of Equation 1 by outputting a
relaxed solution (i.e., _x ∈_ [0 _,_ 1] _[n]_ ) followed by a final rounding step to obtain valid binary solutions.


**MaxCut.** Without loss of generality, we assume the edges have unit weights. The goal of MaxCut is
to find a partition of the nodes into two disjoint subsets that maximizes the number of edges between
them. We can model MaxCut solutions using _x ∈{_ 0 _,_ 1 _}_ _[n]_ where _xi_ = 0 if node _i_ is in one set, and
_xi_ = 1 if node _i_ is in the other set. The objective can be formulated as [�] ( _i,j_ ) _∈E_ [(2] _[x][i][x][j]_ _[−]_ _[x][i][ −]_ _[x][j]_ [)][,]

which is an instance of QUBO (Glover et al., 2018).


**MIS.** MIS is the largest subset of non-adjacent nodes. Similar to MaxCut, we let _xi_ = 1 if
node _i_ is in the independent set and _xi_ = 0 otherwise. The objective can be formulated as

_−_ [�] _i∈V_ _[x][i]_ [+] _[M]_ [ �] ( _i,j_ ) _∈E_ _[x][i][x][j]_ [,] [where] _[M]_ _[∈]_ [R] _[>]_ [0] [is] [a] [penalty] [term] [enforcing] [the] [independent]

set constraint (Djidjev et al., 2018).


**TSP.** We consider symmetric TSP instances modeled as undirected complete graphs. TSP aims to
find the route that visits each node exactly once and returns to the origin node with the lowest total
distance traveled. In contrast to MaxCut and MIS, we use a binary matrix _X_ _∈{_ 0 _,_ 1 _}_ _[n][×][n]_ to represent
a route where _Xij_ = 1 denotes visiting node _i_ at step _j_ . Following Lucas (2014) and The MathWorks
Inc. (2024), we adopt the following objective: [�] ( _i,j_ ) _∈E_ _[w][ij]_ - _nv_ =1 _[X][iv][X][j]_ [(] _[v]_ [+1)] [+] _[ M]_ [ �] _[n]_ _i_ =1 [(1] _[ −]_

- _nj_ =1 _[X][ij]_ [)][2] [+] _[ M]_ [ �] _[n]_ _j_ =1 [(1] _[ −]_ [�] _[n]_ _i_ =1 _[X][ij]_ [)][2][,] [where] _[w][ij]_ [is the distance between node] _[i]_ [ and node] _[j]_ [,]
_Xj_ ( _n_ +1) = _Xj_ 1, accounting for the requirement of returning to the origin node, and _M_ _∈_ R _>_ 0 is
a penalty term to ensure _X_ represents a valid route. During the QUBO objective computation, we
simply flatten _X_ to an _n_ [2] -dimensional vector.


3.2 DCO ON GRAPHS


We focus on DCO problems on discrete-time dynamic graphs (DTDGs). A DTDG representing a
DCO instance is a sequence [ _G_ [1] _p_ _[, G]_ [2] _p_ _[, . . ., G][T]_ _p_ []][ of graph snapshots where each] _[ G]_ _p_ _[t]_ [=] [(] _[V]_ _p_ _[t][, E]_ _p_ _[t]_ _[, w]_ _p_ _[t]_ [)]
has a node set _Vp_ _[t]_ [=] _[ {]_ [1] _[,]_ [ 2] _[, . . ., n]_ _p_ _[t]_ _[}]_ [, an edge set] _[ E]_ _p_ _[t]_ [, and edge weights] _[ w]_ _p_ _[t]_ [.] [For dynamic MaxCut and]
MIS, we consider the scenario where the node set _Vp_ _[t]_ [and edge set] _[ E]_ _p_ _[t]_ [change over time, and] _[ w]_ _p_ _[t]_ [=] **[ 1]**
for all _t ∈{_ 1 _,_ 2 _, . . ., T_ _}_ . For dynamic TSP, we consider the case where one of the nodes moves along
a trajectory. Effectively, _Vp_ [1] [=] _[ V]_ _p_ [2] [=] _[ . . .]_ [ =] _[ V]_ _p_ _[T]_ [and] _[ E]_ _p_ [1] [=] _[ E]_ _p_ [2] [=] _[ . . .]_ [ =] _[ E]_ _p_ _[T]_ [, but] _[ w]_ _p_ _[t]_ [varies for each] _[ t]_ [.]

Let Ω _[t]_ _p_ [denote the set of discrete feasible solutions for snapshot] _[ G]_ _p_ _[t]_ [.] [For MaxCut and MIS,][ Ω] _[t]_ _p_ [is a]

subset of _{_ 0 _,_ 1 _}_ _[n]_ _p_ _[t]_ . For TSP, Ω _[t]_ _p_ [is the set of all possible routes that visit each node exactly once and]
return to the origin node. The objective is to find the optimal solution for each snapshot of a given
DCO problem instance: _x_ _[t][∗]_ = argmin _x∈_ Ω _tp ℓ_ ( _x_ ; _Q_ _[t]_ _p_ [)][, where] _[ ℓ]_ [(] _[·]_ [;] _[ Q][t]_ _p_ [)][ denotes the cost function as]
defined earlier for each snapshot. For evaluation, we compute the mean approximation ratio (ApR)
across all snapshots: Mean ApR = _T_ [1] - _Tt_ =1 _[ℓ]_ [(] _[x][t]_ [;] _[ Q]_ _p_ _[t]_ [)] _[/ℓ]_ [(] _[x][t][∗]_ [;] _[ Q][t]_ _p_ [)][.]


4 METHOD


4.1 WARM-STARTING PI-GNN
Given a sequence of graph snapshots [ _G_ [1] _p_ _[, G]_ [2] _p_ _[, . . ., G][T]_ _p_ []][, PI-GNN independently initiates an optimiza-]
tion process for each snapshot. One potential issue is that the time interval between two consecutive
snapshots may be shorter than the convergence time of PI-GNN. In such cases, faster convergence is
required. Considering the structural similarity across snapshots, a straightforward baseline method
is to warm start the optimization using the parameters from the previous snapshot. Effectively, the
“solution” to the previous snapshot serves as the initialization for the current one, allowing the model
to “fine-tune” the solution instead of optimizing everything from scratch. We detail the procedure of
warm-starting PI-GNN under different time budgets in Algorithm 1.


**Limitations** **of** **naive** **warm** **start.** Upon empirical evaluation, naively warm-starting PI-GNN
exhibits several limitations. Figure 1 compares the performance of warm-started and static PI-GNN
on dynamic MaxCut and MIS instances. Most notably, we observe that while warm start can
potentially outperform static PI-GNN under a stringent time constraint, its advantages diminish
quickly as the time budget increases. Specifically, when the time constraint is relaxed, warm-started
models tend to produce lower-quality solutions than their static counterparts. Although warm start
does accelerate convergence, the results it converges to are generally suboptimal.


3


_i∈V_ _[x][i]_ [+] _[M]_ [ �]


**TSP.** We consider symmetric TSP instances modeled as undirected complete graphs. TSP aims to
find the route that visits each node exactly once and returns to the origin node with the lowest total
distance traveled. In contrast to MaxCut and MIS, we use a binary matrix _X_ _∈{_ 0 _,_ 1 _}_ _[n][×][n]_ to represent
a route where _Xij_ = 1 denotes visiting node _i_ at step _j_ . Following Lucas (2014) and The MathWorks
Inc. (2024), we adopt the following objective: [�] ( _i,j_ ) _∈E_ _[w][ij]_ - _nv_ =1 _[X][iv][X][j]_ [(] _[v]_ [+1)] [+] _[ M]_ [ �] _[n]_ _i_ =1 [(1] _[ −]_


**Algorithm 2** DyCO-GNN
**Require:**

Problem instance: [ _G_ [1] _p_ _[, G]_ [2] _p_ _[, . . ., G][T]_ _p_ []]
Max epochs for convergence: epochmax
Epochs for warm start optimization: epochws
SP parameters: _λ_ shrink and _λ_ perturb
1: Randomly initialize _θ_ [1]

2: Construct [ _Q_ [1] _p_ _[, Q]_ [2] _p_ _[, . . ., Q][T]_ _p_ []]
for [ _G_ [1] _p_ _[, G]_ [2] _p_ _[, . . ., G][T]_ _p_ []]
3: **for** _i_ = 1 to epochmax **do**
4: Predict _x_ [1] via _A_ ( _G_ [1] _p_ [;] _[ θ]_ [1][)]
5: Update _θ_ [1] using _∇ℓ_ ( _x_ [1] ; _Q_ [1] _p_ [)]
6: **end for**
7: **for** _t_ = 2 to _T_ **do**
8: _θ_ _[t]_ _←_ _λ_ shrink _θ_ _[t][−]_ [1] + _λ_ perturb _ϵ_ _[t]_

9: **for** _i_ = 1 to epochws **do**
10: Predict _x_ _[t]_ via _A_ ( _G_ _[t]_ _p_ [;] _[ θ][t]_ [)]
11: Update _θ_ _[t]_ using _∇ℓ_ ( _x_ _[t]_ ; _Q_ _[t]_ _p_ [)]
12: **end for**
13: **end for**


**Algorithm 1** Warm-starting PI-GNN
**Require:**

Problem instance: [ _G_ [1] _p_ _[, G]_ _p_ [2] _[, . . ., G]_ _p_ _[T]_ []]
Max epochs for convergence: epochmax
Epochs for warm start optimization: epochws


1: Randomly initialize _θ_ [1]

2: Construct [ _Q_ [1] _p_ _[, Q]_ [2] _p_ _[, . . ., Q][T]_ _p_ []]
for [ _G_ [1] _p_ _[, G]_ [2] _p_ _[, . . ., G][T]_ _p_ []]
3: **for** _i_ = 1 to epochmax **do**
4: Predict _x_ [1] via _A_ ( _G_ [1] _p_ [;] _[ θ]_ [1][)]
5: Update _θ_ [1] using _∇ℓ_ ( _x_ [1] ; _Q_ [1] _p_ [)]
6: **end for**
7: **for** _t_ = 2 to _T_ **do**
8: _θ_ _[t]_ _←_ _θ_ _[t][−]_ [1]

9: **for** _i_ = 1 to epochws **do**
10: Predict _x_ _[t]_ via _A_ ( _G_ _[t]_ _p_ [;] _[ θ][t]_ [)]
11: Update _θ_ _[t]_ using _∇ℓ_ ( _x_ _[t]_ ; _Q_ _[t]_ _p_ [)]
12: **end for**
13: **end for**


Figure 1: Performance of static PI-GNN and warm-started PI-GNN on dynamic MaxCut and MIS
instances. Detailed setup is explained in Section 5.


These phenomena are closely tied to the role of initialization in deep learning (Sutskever et al.,
2013; Glorot & Bengio, 2010; He et al., 2015) and can be attributed to the highly nonconvex nature
of the optimization objective with respect to the model parameters, which gives rise to numerous
local optima. When the first snapshot is optimized to convergence, the model undergoes extensive
optimization, resulting in increased confidence in its predictions. Consequently, when the model is
warm-started, the gradients become small due to this overconfidence, making it more difficult for the
model to escape local optima. This hampers its ability to discover higher-quality solutions despite
faster initial progress.


4.2 DYCO-GNN: FAST CONVERGENCE AND ROBUST SOLUTION QUALITY


To address the shortcomings of naively warm-starting PI-GNN, we propose DyCO-GNN, a simple
yet effective method that facilitates both fast convergence and robust solution quality across graph
snapshots. Unlike standard warm start, which reuses the exact model parameters from the previous
snapshot, DyCO-GNN integrates a strategic initialization method, shrink and perturb (SP), originally
proposed in (Ash & Adams, 2020) in a different context for supervised learning tasks. The goal is to
retain the benefits of accelerated convergence while mitigating the tendency of the model to become
trapped in suboptimal local minima due to overconfident predictions caused by warm start.


SP was originally designed to close the generalization gap caused by naively warm-starting neural
network training. It shrinks the model parameters and then adds perturbation noise. Shrinking the
parameters effectively decreases the model’s confidence while preserving the learned hypothesis.
Adding noise empirically improves training time and test performance. It is important to note
that there is no notion of generalization in our method as the model parameters are independently
optimized for each individual problem instance snapshot. Despite the fact that SP tackles a different
problem, we successfully adopt it to resolve the issues of warm-started PI-GNN.


4


MaxCut Infectious

Time (s)
0.06 1.11 2.21 3.32


0.95


0.90


0.85


Epochs


MaxCut UC Social

Time (s)
0.06 1.18 2.36 3.53


MIS Infectious

Time (s)
0.06 1.12 2.22 3.33


0.8


0.7


0.6


MIS UC Social

Time (s)
0.06 1.19 2.37 3.56


Epochs


0.995

0.990

0.985

0.980

0.975


Epochs


Epochs


0.5


0.9

0.8

0.7

0.6

0.5


When a new snapshot _G_ _[t]_ _p_ [arrives, DyCO-GNN applies SP to the warm-started parameters before]
starting the optimization process. The newly initialized parameters are defined as


_θ_ _[t]_ _←_ _λ_ shrink _θ_ _[t][−]_ [1] + _λ_ perturb _ϵ_ _[t]_ _,_


where 0 _< λ_ shrink _<_ 1, 0 _< λ_ perturb _<_ 1, and _ϵ_ _[t]_ _∼N_ (0 _, σ_ [2] ). With the reintroduced gradient diversity
and disrupted premature overconfidence, DyCO-GNN effectively escapes the local minima in the loss
landscape. This gentle destabilization promotes exploration of alternative descent paths, functioning
as a principled soft reset mechanism that facilitates more effective and robust optimization.


With this simple design, DyCO-GNN achieves a balanced trade-off: it preserves the efficiency of
warm start under tight time constraints while enabling the model to reach higher-quality solutions
as more computational budget becomes available. The pseudocode of DyCO-GNN is provided in
Algorithm 2. As we show in Section 5, DyCO-GNN consistently outperforms warm-started PI-GNN
across DCO problems such as dynamic MaxCut, MIS, and TSP, particularly under moderate to
generous time budgets where naive warm start fails to find high-quality solutions effectively. We will
also show that DyCO-GNN finds better solutions than the converged static PI-GNN much faster.


4.3 THEORETICAL SUPPORT


We analytically support the advantage of SP over warm start. The theorem is based on the Goemans
& Williamson (1995) (GW) algorithm, the best-known optimization algorithm for solving MaxCut.
The GW algorithm is structured similar to DyCO-GNN, but with a more computationally intensive
QUBO relaxation (an SDP) and rounding step (randomized projections). We adopt it to show that the
advantage of SP persists even if we advance the relaxation and rounding steps.

**Theorem 1.** _Fix the solution X_ 0 _of the GW SDP step, and define Xλ_ := _ProjX_ ( _X_ 0 + _λZ_ ) _, where_
_λ ∈_ R _≥_ 0 _, Z_ _is a symmetric random matrix sampled from the Gaussian Orthogonal Ensemble, and_
_ProjX_ ( _·_ ) _is projection onto set X_ = _{X| X_ _⪰_ 0 _,_ _diag_ ( _X_ ) = 1 _}._ _Denote the GW rounding step by_
_R_ : _{X_ _,_ Ω _} →{_ 0 _,_ 1 _, . . ., c_ _[∗]_ _}, where_ Ω _is the random seed set of the cut plane and c_ _[∗]_ _is the maximum_
_achievable cut size._ _Let Copt_ = _{X_ _∈X_ : PΩ( _R_ ( _X, ω_ ) = _c_ _[∗]_ ) _>_ 0 _}._ _Assume that the set Copt_ _has_
_positive Lebesgue measure in X_ _, and that X_ 0 _∈C/_ _opt._ _Then, there exists a λ >_ 0 _such that_

PΩ _,Z_ ( _R_ ( _Xλ, ω_ ) = _c_ _[∗]_ ) _>_ PΩ( _R_ ( _X_ 0 _, ω_ ) = _c_ _[∗]_ ) = 0 _._


In words, the theorem shows that (pre-rounding) perturbations can strictly increase the probability of
finding the optimal cut. The proof is shown in Appendix E, along with a detailed description of the
GW algorithm, and a corollary extending the result to perturbations of the GW SDP initialization.


5 EXPERIMENTS


We empirically evaluate DyCO-GNN on dynamic instances of MaxCut, MIS, and TSP of varying
problem sizes. Baselines include static PI-GNN and warm-started PI-GNN. We exclude methods
that rely on extensive offline training across instance distributions. This choice aligns with the core
design philosophy of our proposed approach, which, like PI-GNN, emphasizes _instance-specific_
_adaptability_ over generalization. While generalizable methods trained on large datasets may offer
strong performance on similar distributions, prior research has shown that methods from the PIGNN family often achieve comparable performance across static CO problems like MaxCut and
MIS (Schuetz et al., 2022; Wang & Li, 2023; Ichikawa, 2024). Moreover, the generalization gap of
the methods with generalizability increases significantly when distribution shift is present (Karalias
& Loukas, 2020; Wang & Li, 2023). By focusing our comparison on methods within the PI-GNN
family, we provide a clearer assessment of the algorithmic innovations without conflating results with
the (dis)advantages of dataset-level generalization. We also compare with the results of non-neural
baselines in Appendix D.3.


5.1 DATASETS


We utilized established graph datasets that contain temporal information to ensure that the dynamic
evolution of the graphs accurately reflects realistic, time-dependent structural changes. This approach
allows us to capture authentic patterns of change within the graph over time, thereby enhancing the


5


Table 1: Mean ApR on dynamic MaxCut. Values closer to 1 are better ( _↑_ ). All methods use GCNConv.
“emb”, “GNN”, and “full” refer to applying SP to the embedding layer, GNN layers, and all layers,
respectively. The best result for each time budget is in bold. The first time each method surpasses the
converged solution of static PI-GNN is highlighted.


Time budget (epochs/seconds)


Method 50/0.06 100/0.12 200/0.23 300/0.34 500/0.56 1000/1.11 2000/2.21 3000/3.32


validity and applicability of our experimental results. We summarize the statistics of the original and
preprocessed datasets in Table 4 in Appendix B.


For MaxCut and MIS, we employed Infectious (Isella et al., 2011), a human contact network; UC
Social (Opsahl & Panzarasa, 2009), a communication network; and DBLP (Ley, 2002), a citation
network. All datasets are publicly available in the Koblenz Network Collection (Kunegis, 2013).
Each dataset is a graph with timestamps attached to all edges. For preprocessing, we began by sorting
all edge events in chronological order, followed by the removal of self-loops and duplicate edges.
Finally, we made all graphs undirected. To construct DTDG snapshots for MaxCut, we converted the
graphs into 10 snapshots by linearly increasing the number of edges to include in each of them (i.e.,
the dynamic graph grows over time). For Infectious and UC Social, we added _∼_ 10% of total edges of
the final graph every step; for DBLP, we added _∼_ 2% each step, considering its total number of edges
is much larger. Edge additions will lead to MIS constraint violations. In that case, a postprocessing
step that reuses previous solutions and removes violations would be sufficient to solve the DCO
problem. Therefore, to construct DTDG snapshots for MIS, we reversed the snapshot order and turn
edge additions into edge deletions.


For TSP, we first took static TSP benchmark problems from TSPLIB (Reinelt, 1991). Specifically,
we considered burma14, ulysses22, and st70. We added an extra node and let it move along a straight
line within the region defined by the other existing nodes. We recorded the coordinates of 5 equally
spaced locations along the trajectory (i.e., 5 snapshots in total).


5.2 IMPLEMENTATION DETAILS


DyCO-GNN consists of a node embedding layer and two graph convolution layers. Unlike previous
works (Schuetz et al., 2022; Heydaribeni et al., 2024; Ichikawa, 2024) that used different embedding
and intermediate hidden dimensions for each problem instance, we used 512 for the embedding
dimension and 256 for the hidden dimension in all our experiments. We experimented with the
graph convolution operator (GCNConv) from Kipf & Welling (2017) and the GraphSAGE operator
(SAGEConv) from Hamilton et al. (2017). We found that DyCO-GNN with GCNConv failed to find
good enough solutions to TSP, so we only report results of the SAGEConv models on dynamic TSP.
We set _λ_ shrink = 0 _._ 4 and _λ_ perturb = 0 _._ 1 without further tuning. All MaxCut and MIS experiments were
repeated five times; all TSP experiments were repeated ten times. We obtained the ground truth for
each snapshot by formulating it as a QUBO problem and solving it using the Gurobi solver (Gurobi
Optimization, LLC, 2024) with a time limit of 60 seconds. Additional implementation details are
provided in Appendix C.


6


Infectious


UC Social


DBLP


PI-GNN (static) 0.82906 0.92646 0.97203 0.98509 0.98801 0.98871 **0.98887** **0.98888**
PI-GNN (warm) **0.93460** 0.95681 0.96874 0.97170 0.97186 0.96980 0.97196 0.96959
DyCO-GNN (emb) 0.88562 0.96018 0.97817 0.98458 0.97695 0.97874 0.98117 0.96473
DyCO-GNN (GNN) 0.93187 0.96485 0.97962 0.98366 0.98478 0.98456 0.98460 0.98409
DyCO-GNN (full) 0.86286 **0.96830** **0.98384** **0.98793** **0.98947** **0.98878** 0.98869 0.98836


Time budget (epochs/seconds)


Method 50/0.06 100/0.12 200/0.24 300/0.36 500/0.59 1000/1.18 2000/2.36 3000/3.53


PI-GNN (static) 0.97557 0.99485 0.99764 0.99798 0.99809 0.99817 0.99820 0.99823
PI-GNN (warm) 0.99363 0.99455 0.99469 0.99492 0.99501 0.99519 0.99515 0.99519
DyCO-GNN (emb) **0.99634** 0.99709 0.99734 0.99720 0.99711 0.99710 0.99711 0.99705
DyCO-GNN (GNN) 0.99352 0.99714 0.99779 0.99781 0.99786 0.99782 0.99782 0.99779
DyCO-GNN (full) 0.99404 **0.99724** **0.99825** **0.99841** **0.99843** **0.99848** **0.99846** **0.99848**


Time budget (epochs/seconds)


Method 50/0.14 100/0.29 200/0.57 300/0.85 500/1.41 1000/2.82 2000/5.63 3000/8.44


PI-GNN (static) 0.80083 0.97292 0.98905 0.98973 0.99017 0.99054 0.99081 0.99095
PI-GNN (warm) 0.98804 0.98890 0.98957 0.98969 0.98976 0.99001 0.99009 0.99039
DyCO-GNN (emb) 0.98858 0.98995 0.99028 0.99039 0.99044 0.99047 0.99061 0.99041
DyCO-GNN (GNN) 0.98950 0.99110 0.99145 0.99155 0.99164 0.99176 0.99180 0.99183
DyCO-GNN (full) **0.99319** **0.99515** **0.99524** **0.99515** **0.99509** **0.99495** **0.99484** **0.99477**


Table 2: Mean ApR on dynamic MIS. Values closer to 1 are better ( _↑_ ). All methods use GCNConv.
“emb”, “GNN”, and “full” refer to applying SP to the embedding layer, GNN layers, and all layers,
respectively. The best result for each time budget is in bold. The first time each method surpasses the
converged solution of static PI-GNN is highlighted.


Time budget (epochs/seconds)


Method 50/0.06 100/0.12 200/0.23 300/0.34 500/0.56 1000/1.12 2000/2.22 3000/3.33


5.3 EMPIRICAL RESULTS ON DCO PROBLEMS


Tables 1, 2, and 3 report the mean ApR of all methods on dynamic MaxCut, MIS, and TSP, respectively.
We also evaluate variants of DyCO-GNN that apply SP to different layers of DyCO-GNN. For MaxCut
and MIS, results are reported using the final model checkpoint. For TSP, both the final and the bestperforming checkpoints are considered. The wall-clock time of the best-performing checkpoint
includes the total decoding time over all checkpoints available. As noted in Section 4, the advantage
of warm-started PI-GNN compared to static PI-GNN diminishes quickly when we relax the runtime
constraint. DyCO-GNN closes this performance gap and consistently outperforms static PI-GNN
and warm-started PI-GNN across all problems and datasets. Even on a strict budget (first column
of each table), DyCO-GNN surpasses warm-started PI-GNN in most cases. Crucially, DyCO-GNN
consistently finds better solutions than fully converged static PI-GNN, and often within just **1.67%**
to **33.33%** of the total runtime. This improvement possibly stems from information carryover
and continual learning on a similar graph structure, enabled by the utilization of the previously
learned hypothesis. Although the three versions of DyCO-GNN give strong results, no single setting
dominates all tasks and datasets. We will discuss the implications in Section 6. We visualize snapshotlevel ApRs under different time budgets in Figures 2, 3, and 4. Notably, DyCO-GNN maintains its
edge across almost all evaluated snapshots, confirming its robustness. Additional experimental results
are provided in Appendix D.


**Sensitivity of different methods to varying degrees of change.** Tables 1 and 2 show that the quality
gap of converged solutions between static PI-GNN and DyCO-GNN (resp. warm-started PI-GNN)
is larger (resp. smaller) for DBLP. This is because we constructed the DTDG for DBLP with a
smaller relative change in the edge set ( _∼_ 2% of total edges), which leads to greater structural overlaps.
Consequently, warm-started PI-GNN and DyCO-GNN would be stronger in such a scenario. We
further conducted a sensitivity analysis by varying the degrees of change in the DTDGs. Results are
illustrated in Figure 5. When ∆edges is getting smaller, the gap between static and warm-started
PI-GNN is narrowed, and the advantage of DyCO-GNN compared to static PI-GNN is more salient,
which is consistent with our hypothesis.


**Sensitivity** **analysis** **on** **SP** **parameters.** To demonstrate the generality and robustness of our
method without relying on dataset-specific tuning, we deliberately chose a single, reasonable set
of SP parameter values (i.e., _λ_ shrink and _λ_ perturb) and applied it uniformly, highlighting that strong
performance can be achieved without sensitive hyperparameter tuning. For completeness, we include
a sensitivity analysis on the SP parameters in Appendix D.4.


7


Infectious


UC Social


DBLP


PI-GNN (static) 0.67757 0.70395 0.72587 0.73944 0.78275 0.86258 0.87577 0.87651
PI-GNN (warm) 0.50072 0.50217 0.50255 0.50285 0.50845 0.50620 0.50381 0.50309
DyCO-GNN (emb) **0.78941** **0.79577** 0.77912 0.76798 0.74713 0.72983 0.72221 0.70964
DyCO-GNN (GNN) 0.69457 0.74386 **0.79747** **0.78105** 0.75516 0.73314 0.71517 0.71508
DyCO-GNN (full) 0.69461 0.72760 0.75278 0.77278 **0.84044** **0.88254** **0.88397** **0.88515**


Time budget (epochs/seconds)


Method 50/0.06 100/0.12 200/0.24 300/0.36 500/0.60 1000/1.19 2000/2.37 3000/3.56


PI-GNN (static) 0.50103 0.60923 0.74196 0.82713 0.89385 0.91340 0.91594 0.91655
PI-GNN (warm) 0.73879 0.74222 0.74561 0.74828 0.74731 0.74740 0.75011 0.75016
DyCO-GNN (emb) **0.87396** **0.86808** **0.85321** 0.84667 0.83863 0.82997 0.82259 0.81732
DyCO-GNN (GNN) 0.68646 0.78990 0.82860 0.82898 0.82632 0.82217 0.81881 0.81739
DyCO-GNN (full) 0.54819 0.66512 0.80108 **0.87445** **0.91600** **0.92037** **0.92069** **0.91984**


Time budget (epochs/seconds)


Method 50/0.14 100/0.28 200/0.57 300/0.85 500/1.40 1000/2.80 2000/5.60 3000/8.41


PI-GNN (static) 0.18364 0.48119 0.89690 0.93320 0.94304 0.94636 0.94766 0.94813
PI-GNN (warm) 0.93136 0.93313 0.93459 0.93516 0.93595 0.93672 0.93771 0.93925
DyCO-GNN (emb) **0.95517** **0.95599** **0.95529** 0.95491 0.95478 0.95398 0.95446 0.95448
DyCO-GNN (GNN) 0.73970 0.93442 0.94082 0.94114 0.94119 0.94135 0.94169 0.94183
DyCO-GNN (full) 0.26570 0.65637 0.95175 **0.96550** **0.96895** **0.96972** **0.97016** **0.97042**


Table 3: Mean ApR on dynamic TSP. Values closer to 1 are better ( _↓_ ). The best-performing checkpoint
was taken. “emb”, “GNN”, and “full” refer to applying SP to the embedding layer, GNN layers, and
all layers, respectively. The best result for each time budget is in bold. The first time each method
surpasses the converged solution of static PI-GNN is highlighted.


Time budget (epochs/seconds)


Method 500/0.31 1000/0.62 2000/1.24 3000/1.86 5000/3.10 10000/6.21      

|Time: 0.06s|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|
|||4<br>T|S<br>im|nap<br>e:|sh<br> 0.|7<br><br>ot<br> 59|s|1|0|
|||||||||||


|Time: 0.12s MaxCut UC Social|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|
|||T|S<br>im|nap<br>e:|sh<br> 1.|7<br><br>ot<br> 18|8<br><br> s|9 1|0|
|||||||||||


|Time: 0.24s|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|
|||T|S<br>im|nap<br>e:|6<br><br>sh<br> 3.|7<br><br>ot<br> 53|8<br><br> s|9 1|0|
|||<br>|<br>|<br>||||||
||S<br>W<br>P<br>D|tatic<br>arm-<br>I-GN<br>yCO-|PI-G<br>start<br>N<br>GNN|NN<br>ed|NN<br>ed|NN<br>ed|NN<br>ed|NN<br>ed|NN<br>ed|


Figure 2: Snapshot-level ApRs on dynamic MaxCut instance (UC Social). All methods use GCNConv.


6 DISCUSSION


**Conclusion.** We introduced DyCO-GNN, the first learning-based framework designed to solve DCO
problems. Through extensive experiments on dynamic MaxCut, MIS, and TSP, we demonstrated
that DyCO-GNN consistently outperforms both static and warm-started PI-GNN baselines under
varying runtime constraints and across diverse problem settings. Our analysis reveals that DyCOGNN achieves superior solution quality even under strict time budgets, always surpassing the bestperforming checkpoints of the baseline methods in a fraction of their runtime. These improvements are
significant in rapidly changing environments, where timely decision-making is critical. Furthermore,
DyCO-GNN’s ability to efficiently adapt to evolving problem instance snapshots without optimization
from scratch or external supervision underscores its potential for real-world deployment in dynamic,
resource-constrained settings.


8


burma14


ulysses22


st70


1.00


0.99


1.00


0.99


PI-GNN (static) 1.31234 1.13584 1.06970 1.05998 1.04756 1.03358 PI-GNN (warm) 1.28388 1.28313 1.25798 1.23433 1.21120 1.13860 DyCO-GNN (emb) 1.15106 1.13302 1.09256 1.07814 1.06478 1.04782 DyCO-GNN (GNN) **1.10365** **1.06370** 1.05406 1.05668 1.03737 **1.01811** DyCO-GNN (full) 1.15884 1.10839 **1.04531** **1.03134** **1.03160** 1.04032 

Time budget (epochs/seconds)


Method 500/0.32 1000/0.64 2000/1.27 3000/1.90 5000/3.18 10000/6.37 

PI-GNN (static) 1.76783 1.48249 1.26151 1.17969 **1.11249** 1.08867 PI-GNN (warm) **1.18266** 1.17228 1.18642 1.18381 1.17534 1.19563 DyCO-GNN (emb) 1.20391 **1.16486** 1.16819 1.15758 1.14706 1.13088 DyCO-GNN (GNN) 1.25758 1.19970 **1.14971** **1.12991** 1.12469 1.13545 DyCO-GNN (full) 1.30026 1.27399 1.22204 1.16843 1.12538 **1.08516** 

Time budget (epochs/seconds)


Method 500/0.43 1000/0.85 2000/1.70 3000/2.54 5000/4.23 10000/8.46 20000/16.95


PI-GNN (static) 2.01621 1.98448 1.92607 1.82142 1.59433 1.43824 1.36945
PI-GNN (warm) 1.47258 1.46055 1.42443 1.42589 1.38696 1.36187 1.34213
DyCO-GNN (emb) **1.45563** **1.42486** 1.38436 1.36485 1.35266 1.30753 1.29239
DyCO-GNN (GNN) 1.53036 1.44924 **1.36227** **1.33620** 1.32063 1.27857 **1.23655**
DyCO-GNN (full) 1.56795 1.54532 1.38231 1.34845 **1.30566** **1.27127** 1.24412


2 3 4 5 6 7 8 9 10
Snapshot


1.00


0.99


1.00


0.99


2 3 4 5 6 7 8 9 10
Snapshot


2 3 4 5 6 7 8 9 10
Snapshot


1.00


0.99


1.00


0.99


1.0


0.5


|Static PI-GNN<br>Warm-started<br>PI-GNN<br>DyCO-GNN|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||Static P<br>Warm-s<br>PI-GNN<br>DyCO-G|I-GNN<br>tarted<br>NN|I-GNN<br>tarted<br>NN|I-GNN<br>tarted<br>NN|I-GNN<br>tarted<br>NN|


Figure 5: Sensitivity analysis using UC Social.


**Limitations and future work.** We have shown that applying SP to different layers of DyCO-GNN
yields varying performance. A promising direction for future work is to make the SP step adaptive.
Currently, the embedding layer and the GNN layers are jointly optimized without explicit decoupling.
By introducing appropriate regularization strategies or auxiliary loss functions, we can promote
distinct functional roles across layers; for instance, encouraging the embedding layer to capture
representations specific to both the problem and graph structure and the GNN layers to learn to
aggregate such representations. Additionally, if we evaluate the likelihood of changes in node
assignments under dynamic conditions based on certain heuristics, it becomes possible to apply SP in
a node-wise, adaptive manner.


9


MIS UC Social


1.0


0.5


1.0


0.5


0.0


1.0


0.5


0.0


Snapshot


Snapshot


0.0


0.0


Snapshot


Snapshot


0.0


1.0


0.5


1.0


Snapshot


Snapshot


0.5


0.0


Figure 3: Snapshot-level ApRs on dynamic MIS instance (UC Social). All methods use GCNConv.


TSP burma14 Best


1.000


0.995


0.990


REFERENCES


Rico Angell and Andrew Mccallum. Fast, scalable, warm-start semidefinite programming with
spectral bundling and sketching. In _International Conference on Machine Learning_, pp. 1579–
1615. PMLR, 2024.


Jordan Ash and Ryan P Adams. On warm-starting neural network training. _Advances_ _in_ _neural_
_information processing systems_, 33:3884–3894, 2020.


Sepehr Assadi, Krzysztof Onak, Baruch Schieber, and Shay Solomon. Fully dynamic maximal
independent set with sublinear update time. In _Proceedings of the 50th Annual ACM SIGACT_
_Symposium on theory of computing_, pp. 815–826, 2018.


Giorgio Ausiello, Bruno Escoffier, Jérôme Monnot, and Vangelis Paschos. Reoptimization of
minimum and maximum traveling salesman’s tours. _Journal of Discrete Algorithms_, 7(4):453–463,
2009.


Heinz H Bauschke and Patrick L Combettes. _Convex_ _analysis_ _and_ _monotone_ _operator_ _theory_ _in_
_Hilbert spaces_ . Springer, 2017.


Irwan Bello, Hieu Pham, Quoc V Le, Mohammad Norouzi, and Samy Bengio. Neural combinatorial
optimization with reinforcement learning. _arXiv preprint arXiv:1611.09940_, 2016.


Ravi Boppana and Magnús Halldórsson. Approximating maximum independent sets by excluding
subgraphs. _BIT Numerical Mathematics_, 32:13–25, 01 2006. doi: 10.1007/3-540-52846-6_74.


Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on
graphs with fast localized spectral filtering. _Advances in neural information processing systems_,
29, 2016.


Hristo N Djidjev, Guillaume Chapuis, Georg Hahn, and Guillaume Rizk. Efficient combinatorial
optimization using quantum annealing. _arXiv preprint arXiv:1801.08653_, 2018.


Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. _arXiv_
_preprint arXiv:1903.02428_, 2019.


Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin, and Andrea Lodi. Exact combinatorial optimization with graph convolutional neural networks. _Advances in neural information_
_processing systems_, 32, 2019.


Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural
networks. In _Proceedings of the thirteenth international conference on artificial intelligence and_
_statistics_, pp. 249–256. JMLR Workshop and Conference Proceedings, 2010.


Fred Glover, Gary Kochenberger, and Yu Du. A tutorial on formulating and using qubo models.
_arXiv preprint arXiv:1811.11538_, 2018.


Michel X Goemans and David P Williamson. Improved approximation algorithms for maximum cut
and satisfiability problems using semidefinite programming. _Journal of the ACM (JACM)_, 42(6):
1115–1145, 1995.


Gurobi Optimization, LLC. Gurobi Optimizer Reference Manual, 2024. URL [https://www.](https://www.gurobi.com)
[gurobi.com.](https://www.gurobi.com)


Aric A. Hagberg, Daniel A. Schult, and Pieter J. Swart. Exploring network structure, dynamics,
and function using networkx. In Gaël Varoquaux, Travis Vaught, and Jarrod Millman (eds.),
_Proceedings of the 7th Python in Science Conference_, pp. 11 – 15, Pasadena, CA USA, 2008.


Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs.
_Advances in neural information processing systems_, 30, 2017.


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing
human-level performance on imagenet classification. In _Proceedings of the IEEE international_
_conference on computer vision_, pp. 1026–1034, 2015.


10


Nasimeh Heydaribeni, Xinrui Zhan, Ruisi Zhang, Tina Eliassi-Rad, and Farinaz Koushanfar. Distributed constrained combinatorial optimization leveraging hypergraph neural networks. _Nature_
_Machine Intelligence_, pp. 1–9, 2024.


Benjamin Hudson, Qingbiao Li, Matthew Malencia, and Amanda Prorok. Graph neural network
guided local search for the traveling salesperson problem. In _International Conference on Learning_
_Representations_, 2022.


Yuma Ichikawa. Controlling continuous relaxation for combinatorial optimization. In _The Thirty-_
_eighth Annual Conference on Neural Information Processing Systems_, 2024.


Lorenzo Isella, Juliette Stehlé, Alain Barrat, Ciro Cattuto, Jean-François Pinton, and Wouter Van den
Broeck. What’s in a crowd? analysis of face-to-face behavioral networks. _Journal of theoretical_
_biology_, 271(1):166–180, 2011.


Elizabeth John and E Alper Yıldırım. Implementation of warm-start strategies in interior-point
methods for linear programming in fixed dimension. _Computational Optimization and Applications_,
41(2):151–183, 2008.


Chaitanya K Joshi, Thomas Laurent, and Xavier Bresson. An efficient graph convolutional network
technique for the travelling salesman problem. _arXiv preprint arXiv:1906.01227_, 2019.


Chaitanya K Joshi, Quentin Cappart, Louis-Martin Rousseau, and Thomas Laurent. Learning the
travelling salesperson problem requires rethinking generalization. _Constraints_, 27(1):70–98, 2022.


Nikolaos Karalias and Andreas Loukas. Erdos goes neural: an unsupervised learning framework for
combinatorial optimization on graphs. _Advances in Neural Information Processing Systems_, 33:
6659–6672, 2020.


Elias Khalil, Hanjun Dai, Yuyu Zhang, Bistra Dilkina, and Le Song. Learning combinatorial
optimization algorithms over graphs. _Advances_ _in_ _neural_ _information_ _processing_ _systems_, 30,
2017.


Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _International_
_Conference on Learning Representations_, 2015.


Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks.
In _International Conference on Learning Representations_, 2017.


Wouter Kool, Herke van Hoof, and Max Welling. Attention, learn to solve routing problems! In
_International Conference on Learning Representations_, 2019.


Jérôme Kunegis. Konect: the koblenz network collection. In _Proceedings of the 22nd international_
_conference on world wide web_, pp. 1343–1350, 2013.


Michael Ley. The dblp computer science bibliography: Evolution, research issues, perspectives. In
_International symposium on string processing and information retrieval_, pp. 1–10. Springer, 2002.


Yang Li, Jinpei Guo, Runzhong Wang, and Junchi Yan. T2t: From distribution learning in training
to gradient search in testing for combinatorial optimization. _Advances_ _in_ _Neural_ _Information_
_Processing Systems_, 36:50020–50040, 2023.


Yang Li, Jinpei Guo, Runzhong Wang, Hongyuan Zha, and Junchi Yan. Fast t2t: Optimization
consistency speeds up diffusion-based training-to-testing solving for combinatorial optimization.
_Advances in Neural Information Processing Systems_, 37:30179–30206, 2024.


Andrew Lucas. Ising formulations of many np problems. _Frontiers in physics_, 2:5, 2014.


Krzysztof Onak and Ronitt Rubinfeld. Maintaining a large matching and a small vertex cover. In
_Proceedings of the forty-second ACM symposium on Theory of computing_, pp. 457–464, 2010.


Tore Opsahl and Pietro Panzarasa. Clustering in weighted networks. _Social networks_, 31(2):155–163,
2009.


11


Christos H. Papadimitriou and Kenneth Steiglitz. _Combinatorial_ _optimization:_ _algorithms_ _and_
_complexity_ . Prentice-Hall, Inc., USA, 1982. ISBN 0131524623.


Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward
Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang,
Junjie Bai, and Soumith Chintala. Pytorch: an imperative style, high-performance deep learning
library. In _Proceedings of the 33rd International Conference on Neural Information Processing_
_Systems_, 2019.


Ruizhong Qiu, Zhiqing Sun, and Yiming Yang. Dimes: A differentiable meta solver for combinatorial
optimization problems. _Advances in Neural Information Processing Systems_, 35:25531–25546,
2022.


Gerhard Reinelt. Tsplib - a traveling salesman problem library. _INFORMS J. Comput._, 3:376–384,
1991. [URL https://api.semanticscholar.org/CorpusID:207225504.](https://api.semanticscholar.org/CorpusID:207225504)


Sebastian Sanokowski, Wilhelm Berghammer, Sepp Hochreiter, and Sebastian Lehner. Variational
annealing on graphs for combinatorial optimization. _Advances in Neural Information Processing_
_Systems_, 36:63907–63930, 2023.


Martin JA Schuetz, J Kyle Brubaker, and Helmut G Katzgraber. Combinatorial optimization with
physics-inspired graph neural networks. _Nature Machine Intelligence_, 4(4):367–377, 2022.


Alexander Shapiro. Sensitivity analysis of nonlinear programs and differentiability properties of
metric projections. _SIAM Journal on Control and Optimization_, 26(3):628–645, 1988.


Zhiqing Sun and Yiming Yang. Difusco: Graph-based diffusion solvers for combinatorial optimization.
_Advances in neural information processing systems_, 36:3706–3731, 2023.


Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization
and momentum in deep learning. In _International conference on machine learning_, pp. 1139–1147.
PMLR, 2013.


The MathWorks Inc. Traveling salesperson problem with qubo, 2024. URL [https://www.](https://www.mathworks.com/help/matlab/math/quantum-tsp.html)
[mathworks.com/help/matlab/math/quantum-tsp.html.](https://www.mathworks.com/help/matlab/math/quantum-tsp.html)


Mikkel Thorup. Fully-dynamic min-cut. _Combinatorica_, 27(1):91–127, 2007.


Jan Toenshoff, Martin Ritzert, Hinrikus Wolf, and Martin Grohe. Graph neural networks for maximum
constraint satisfaction. _Frontiers in artificial intelligence_, 3:580607, 2021.


Jan Tönshoff, Berke Kisin, Jakob Lindner, and Martin Grohe. One model, any csp: graph neural
networks as fast global search heuristics for constraint satisfaction. In _Proceedings of the Thirty-_
_Second International Joint Conference on Artificial Intelligence_, IJCAI ’23, 2023.


Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua
Bengio. Graph attention networks. In _International Conference on Learning Representations_,
2018.


Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer networks. _Advances in neural information_
_processing systems_, 28, 2015.


Haoyu Peter Wang and Pan Li. Unsupervised learning for combinatorial optimization needs meta
learning. In _The Eleventh International Conference on Learning Representations_, 2023.


Omer Wasim and Valerie King. Fully dynamic sequential and distributed algorithms for max-cut. In
_40th IARCS Annual Conference on Foundations of Software Technology and Theoretical Computer_
_Science (FSTTCS 2020)_, pp. 33–1. Schloss Dagstuhl–Leibniz-Zentrum für Informatik, 2020.


Shengxiang Yang, Yong Jiang, and Trung Thanh Nguyen. Metaheuristics for dynamic combinatorial
optimization problems. _IMA Journal of Management Mathematics_, 24(4):451–480, 10 2012. ISSN
1471-6798. doi: 10.1093/imaman/dps021.


12


E Alper Yildirim and Stephen J Wright. Warm-start strategies in interior-point methods for linear
programming. _SIAM Journal on Optimization_, 12(3):782–810, 2002.


Zizhen Zhang, Hong Liu, MengChu Zhou, and Jiahai Wang. Solving dynamic traveling salesman
problems with deep reinforcement learning. _IEEE Transactions on Neural Networks and Learning_
_Systems_, 34(4):2119–2132, 2021.


13


A ADDITIONAL RELATED WORK


DCO problems have been addressed from a theoretical perspective (Onak & Rubinfeld, 2010; Thorup,
2007), including for the dynamic MaxCut (Wasim & King, 2020), MIS (Assadi et al., 2018), and
TSP (Ausiello et al., 2009) problems, as well as from the perspective of designing heuristics (Yang
et al., 2012). Our work brings a learning-based approach to this line of work.


The idea of warm-starting semidefinite programs in general, and the starting solutions of interior-point
methods (which can be used to solved SDPs, among other problem classes) has been explored from
a theoretical perspective (Yildirim & Wright, 2002; John & Yıldırım, 2008; Angell & Mccallum,
2024). Our theoretical findings complement this literature by exploring the impacts of warm-starting
SDP solutions for the MaxCut problem. Our learning-based method DyCO-GNN also brings similar
ideas into the realm of learning-based DCO methods.


B DATASET STATISTICS


Table 4: Statistics of all datasets. Original edges can be directed and have duplicates.


Original Preprocessed


Dataset Nodes Edges Nodes Edges ∆ edges


Infectious 410 17,298 410 2765 _∼_ 276 (10%)
UC Social 1899 59,835 1899 13,838 _∼_ 1384 (10%)
DBLP 12,590 49,759 12,590 49,636 _∼_ 993 (2%)
burma14 14 91 15 105           ulysses22 22 231 23 253            st70 70 2,415 71 2,485             

C ADDITIONAL IMPLEMENTATION DETAILS


We used the Adam optimizer (Kingma & Ba, 2015); the learning rate was set to 0.001 for all
MaxCut and MIS experiments, 0.0002 for burma14 and st70, and 0.0005 for ulysses22. As described
in Algorithms 1 and 2, we optimize over the first snapshot for epochmax epochs. Thus, we skip
the first snapshot during evaluation since the results will be identical across all methods. We set
epochmax = 3000 for all MaxCut and MIS experiments, epochmax = 10000 for burma14 and
ulysses22, and epochmax = 20000 for st70.

The penalty term _M_ was set to 2 for MIS and 2 _×_ max( _i,j_ ) _∈E wij_ (i.e., 2 _×_ the max distance between
any two nodes) for TSP. For MIS, we postprocessed the output of our model by greedily removing
violations. For TSP, we decoded the route step by step. More specifically, for burma14 and ulysses22,
we discarded nodes that have already been visited and selected the most likely node to visit based on
our model output; for st70, we found that this greedy decoding failed to find good enough routes and
instead applied beam search that expands the top 5 possible valid nodes at each step.


All models were implemented using PyTorch (Paszke et al., 2019) and PyTorch Geometric (Fey &
Lenssen, 2019). Experiments were conducted on a machine with a single NVIDIA GeForce RTX
4090 GPU, a 32-core Intel Core i9-14900K CPU, and 64 GB of RAM running Ubuntu 24.04.


14


D ADDITIONAL EXPERIMENTAL RESULTS


D.1 DYCO-GNN WITH SAGECONV FOR DYNAMIC MAXCUT AND MIS


Table 5: Mean ApR on dynamic MaxCut. Values closer to 1 are better ( _↑_ ). All methods use
SAGEConv. The best result for each time budget is in bold. The first time each method surpasses the
converged solution of static PI-GNN is highlighted.


Time budget (epochs/seconds)


Method 50/0.06 100/0.12 200/0.23 300/0.34 500/0.56 1000/1.11 2000/2.21 3000/3.32


PI-GNN (static) 0.97028 0.97353 0.97487 0.97540 0.97570 0.97591 0.97626 0.97636
Infectious PI-GNN (warm) 0.95685 0.96117 0.96356 0.96452 0.96396 0.96625 0.96841 0.96619
DyCO-GNN (full) **0.97771** **0.97854** **0.97880** **0.97953** **0.97952** **0.97959** **0.98051** **0.98009**


Time budget (epochs/seconds)


Method 50/0.06 100/0.12 200/0.24 300/0.36 500/0.59 1000/1.18 2000/2.36 3000/3.53


PI-GNN (static) 0.98875 **0.99601** **0.99714** **0.99727** **0.99766** **0.99768** 0.99773 0.99782
UC Social PI-GNN (warm) 0.99292 0.99395 0.99440 0.99456 0.99451 0.99447 0.99437 0.99456
DyCO-GNN (full) **0.99515** 0.99584 0.99595 0.99594 0.99593 0.99589 **0.99856** **0.99852**


Time budget (epochs/seconds)


Method 50/0.14 100/0.29 200/0.57 300/0.85 500/1.41 1000/2.82 2000/5.63 3000/8.44


PI-GNN (static) 0.89694 0.92001 0.93263 0.93688 0.94120 0.94487 0.94610 0.94659
DBLP PI-GNN (warm) 0.94865 0.95037 0.95249 0.95282 0.95336 0.95477 0.95337 0.95444
DyCO-GNN (full) **0.95877** **0.95942** **0.95981** **0.95971** **0.95965** **0.95944** **0.95934** **0.95907**


Table 6: Mean ApR on dynamic MIS. Values closer to 1 are better ( _↑_ ). All methods use SAGEConv.
The best result for each time budget is in bold. The first time each method surpasses the converged
solution of static PI-GNN is highlighted.


Time budget (epochs/seconds)


Method 50/0.06 100/0.12 200/0.23 300/0.34 500/0.56 1000/1.12 2000/2.22 3000/3.33


PI-GNN (static) 0.79468 0.91679 0.95948 0.96362 0.96410 0.96476 0.96518 0.96518
Infectious PI-GNN (warm) **0.96725** **0.96713** **0.96727** **0.96708** **0.96701** **0.96717** 0.96715 0.96687
DyCO-GNN (full) 0.81767 0.96411 0.96540 0.96555 0.96592 0.96707 **0.96746** **0.96746**


Time budget (epochs/seconds)


Method 50/0.06 100/0.12 200/0.24 300/0.36 500/0.60 1000/1.19 2000/2.37 3000/3.56


PI-GNN (static) 0.71087 0.91961 0.96766 0.97126 0.97284 0.97398 0.97478 0.97516
UC Social PI-GNN (warm) **0.97607** 0.97654 0.97665 0.97675 0.97677 0.97670 0.97673 0.97681
DyCO-GNN (full) 0.91199 **0.97913** **0.97838** **0.97760** **0.97761** **0.97708** **0.97698** **0.97708**


Time budget (epochs/seconds)


Method 50/0.14 100/0.28 200/0.57 300/0.85 500/1.40 1000/2.80 2000/5.60 3000/8.41


PI-GNN (static) 0.60354 0.95227 0.98754 0.98950 0.99025 0.99081 0.99113 0.99130
DBLP PI-GNN (warm) 0.99357 0.99363 0.99364 0.99364 0.99367 0.99367 0.99367 0.99366
DyCO-GNN (full) **0.99551** **0.99494** **0.99465** **0.99437** **0.99426** **0.99399** **0.99386** **0.99399**


15


D.2 RESULTS OF THE LAST CHECKPOINTS OF DYCO-GNN ON DYNAMIC TSP


Table 7: Mean ApR on dynamic TSP. Values closer to 1 are better ( _↓_ ). The last checkpoint was taken.
“emb”, “GNN”, and “full” refer to applying SP to the embedding layer, GNN layers, and all layers,
respectively. The best result for each time budget is in bold. The first time each method surpasses the
converged solution of static PI-GNN is highlighted.


Time budget (epochs/seconds)


Method 500/0.31 1000/0.62 2000/1.24 3000/1.86 5000/3.10 10000/6.20      


D.3 RESULTS OF NON-NEURAL BASELINES


Here we include the results of non-neural approaches built into NetworkX (Hagberg et al., 2008).
Each entry of Tables 8, 9, and 10 is of the form ApR (wall clock time). For dynamic MIS, we consider
the method in Boppana & Halldórsson (2006); the implementation in NetworkX only works for
graphs with fewer than several hundred nodes without a recursion error, so we only report the result
on Infectious.


Our method works the best except for dynamic MIS on Infectious and dynamic TSP on st70. We note
that the method in Boppana & Halldórsson (2006) is specifically designed for solving MIS, and the
implementation in NetworkX cannot handle graphs with more than several hundred nodes, whereas
our method and the baselines have broader applicability and can easily handle graphs with more than
10,000 nodes. It has also been shown in Schuetz et al. (2022) that PI-GNN performs on par or better
than Boppana & Halldórsson (2006) on random d-regular graphs. For TSP on st70, the result of the
greedy approach is better than that of Gurobi, which could be an edge case. Nevertheless, our method
improves upon the PI-GNN baselines by a large margin.


Table 8: Comparison with non-neural baseline on dynamic MaxCut. Values closer to 1 are better ( _↑_ ).


Dataset Cut-based greedy PI-GNN (static) PI-GNN (warm) DyCO-GNN


Infectious 0.96921 (5.6s) 0.98888 (3.32s) 0.97186 (0.56s) 0.98947 (0.56s)
UC Social 0.98083 (933.16s) 0.99823 (3.53s) 0.99519 (1.18s) 0.99825 (0.24s)
DBLP Fail to find a solution in 10 hours 0.99095 (8.44s) 0.99039 (8.44s) 0.99319 (0.14s)


Table 9: Comparison with non-neural baseline on dynamic MIS. Values closer to 1 are better ( _↑_ ).


Dataset Boppana & Halldórsson (2006) PI-GNN (static) PI-GNN (warm) DyCO-GNN


Infectious 0.93825(6.88s) 0.87651 (3.33s) 0.50845 (0.56s) 0.88254 (1.12s)


16


burma14


ulysses22


st70


PI-GNN (static) 1.35521 1.14307 1.07890 1.06514 1.05002 1.04025 PI-GNN (warm) 1.30116 1.29941 1.27008 1.28396 1.24094 1.16453 DyCO-GNN (emb) 1.16825 1.15260 1.11613 1.08597 1.09302 1.08198 DyCO-GNN (GNN) **1.13749** **1.08812** 1.06424 1.07396 **1.05675** **1.03002** DyCO-GNN (full) 1.19695 1.15270 **1.05980** **1.04272** 1.06671 1.08105 

Time budget (epochs/seconds)


Method 500/0.32 1000/0.64 2000/1.27 3000/1.90 5000/3.17 10000/6.36 

PI-GNN (static) 1.77516 1.54757 1.29137 1.21368 **1.13068** **1.10295** PI-GNN (warm) **1.18274** 1.17251 1.18645 1.18424 1.18650 1.22231 DyCO-GNN (emb) 1.21109 **1.16790** **1.17498** 1.17131 1.15733 1.13694 DyCO-GNN (GNN) 1.32308 1.27132 1.18273 **1.14293** 1.15254 1.13845 DyCO-GNN (full) 1.36858 1.33861 1.28760 1.22112 1.16052 1.12970 

Time budget (epochs/seconds)


Method 500/0.38 1000/0.76 2000/1.51 3000/2.27 5000/3.78 10000/7.57 20000/15.16


PI-GNN (static) 2.10292 2.11124 2.03993 1.94988 1.70642 1.55780 1.46919
PI-GNN (warm) **1.48778** 1.49261 1.47614 1.46951 1.44118 1.43268 1.43030
DyCO-GNN (emb) 1.49505 **1.49011** **1.44635** 1.45302 1.41491 1.38462 1.35803
DyCO-GNN (GNN) 1.65917 1.58705 1.46435 **1.43776** 1.44108 **1.35839** **1.30869**
DyCO-GNN (full) 1.67305 1.68751 1.50680 1.46486 **1.39795** 1.37431 1.33007


Table 10: Comparison with non-neural baseline on dynamic TSP. Values closer to 1 are better ( _↓_ ).


Dataset Cost-based greedy PI-GNN (static) PI-GNN (warm) DyCO-GNN


burma14 1.19812 (<0.01s) 1.03358 (6.21s) 1.13860 (6.21s) 1.01811 (6.21s)
ulysses22 1.24475 (<0.01s) 1.08867 (6.37s) 1.17228 (0.64s) 1.08516 (6.37s)
st70 0.94963 (<0.01s) 1.36945 (16.95s) 1.34213 (16.95s) 1.23655 (16.95s)


D.4 SENSITIVITY ANALYSIS ON SP PARAMETERS.


From the results reported in Tables 11 and 12, 0 _._ 2 _≤_ _λ_ shrink _≤_ 0 _._ 4 and _λ_ perturb = 0 _._ 1 are good default
choices. When _λ_ shrink approaches 1, the ApRs will potentially be the highest early on, as this setting
is close to a naive warm start; the ApRs may drop in this case when the number of epochs increases
because the parameters closer to convergence on the previous snapshot without proper SP are not
suitable for adaptation. We note that one should not confuse the performance with respect to epochs
with the case in regular deep learning training: we are showing the number of epochs each snapshot
is optimized for before adapting the parameters for the next snapshot, and the ApRs are computed
across all snapshots. In general, a larger _λ_ shrink gives better performance when the time budget is
extremely tight, while a smaller _λ_ shrink gives better performance when the time budget constraint is
slightly relaxed. _λ_ perturb has a smaller effect than _λ_ shrink. _λ_ perturb is usually too aggressive; values like
0.1 and 0.01 typically give the best results.


Table 11: Mean ApR on dynamic MaxCut using UC Social achieved by DyCO-GNN with different
SP parameters. Values closer to 1 are better ( _↑_ ).


Time budget (epochs/seconds)


_λ_ shrink _λ_ perturb 50/0.06 100/0.12 200/0.23 300/0.34 500/0.56 1000/1.12 2000/2.22 3000/3.33


0.2 0.1 0.99142 0.99747 **0.99850** **0.99862** **0.99865** **0.99872** **0.99873** **0.99875**
0.4 0.1 0.99403 0.99724 0.99825 0.99841 0.99844 0.99848 0.99846 0.99848
0.6 0.1 0.99544 **0.99805** 0.99832 0.99830 0.99829 0.99807 0.99793 0.99782
0.8 0.1 **0.99702** 0.99738 0.99760 0.99744 0.99748 0.99742 0.99740 0.99740


0.4 1.0 0.98437 0.99459 0.99731 0.99777 0.99793 0.99793 0.99789 0.99785
0.4 0.1 0.99403 0.99724 0.99825 **0.99841** **0.99844** **0.99848** 0.99846 **0.99848**
0.4 0.01 0.99408 **0.99732** **0.99826** 0.99836 0.99840 0.99847 **0.99848** 0.99846
0.4 0.001 **0.99416** **0.99732** **0.99826** 0.99835 0.99839 0.99845 **0.99848** 0.99845


Table 12: Mean ApR on dynamic MIS using UC Social achieved by DyCO-GNN with different SP
parameters. Values closer to 1 are better ( _↑_ ).


Time budget (epochs/seconds)


_λ_ shrink _λ_ perturb 50/0.06 100/0.12 200/0.23 300/0.34 500/0.56 1000/1.12 2000/2.22 3000/3.33


0.2 0.1 0.48314 0.53766 0.68657 0.80957 0.88256 0.90916 0.91371 0.91607
0.4 0.1 0.54819 0.66505 0.80108 0.87436 **0.91599** **0.92047** **0.92046** **0.91987**
0.6 0.1 0.70601 **0.83711** **0.90015** **0.89537** 0.88777 0.87972 0.87283 0.86969
0.8 0.1 **0.84145** 0.82497 0.80988 0.80543 0.80072 0.79774 0.79606 0.79350


0.4 1.0 **0.59099** **0.69377** **0.81221** 0.86738 0.90229 0.90993 0.90641 0.90533
0.4 0.1 0.54819 0.66505 0.80108 **0.87436** **0.91599** **0.92047** **0.92046** 0.91987
0.4 0.01 0.54720 0.66718 0.80084 0.87241 0.91504 0.92001 0.91994 0.91921
0.4 0.001 0.54733 0.66713 0.80096 0.87235 0.91491 0.92020 **0.92046** **0.91994**


17


D.5 SNAPSHOT-LEVEL APRS BAR PLOTS


|Time:|0.06s|
|---|---|
||7<br>8<br>9 10<br>shot<br> <br><br>7<br>8<br>9 10<br>shot<br> 0.56s<br>  l ApRs on d<br>: 0.14s|
|2<br>3<br>4<br>5<br>6<br>Snap<br>Time:|2<br>3<br>4<br>5<br>6<br>Snap<br>Time:|
|||
|2<br>3<br>4<br>5<br>6<br>Snap<br>  napshot-leve<br>2<br>3<br>4<br>5<br>Sna<br><br><br><br>Time<br>2<br>3<br>4<br>5<br>Sna<br><br><br><br>Time<br>napshot-lev<br>Time:|2<br>3<br>4<br>5<br>6<br>Snap<br>  napshot-leve<br>2<br>3<br>4<br>5<br>Sna<br><br><br><br>Time<br>2<br>3<br>4<br>5<br>Sna<br><br><br><br>Time<br>napshot-lev<br>Time:|
|2<br>3<br>4<br>5<br>6<br>Snap<br>  napshot-leve<br>2<br>3<br>4<br>5<br>Sna<br><br><br><br>Time<br>2<br>3<br>4<br>5<br>Sna<br><br><br><br>Time<br>napshot-lev<br>Time:|<br>|
|2<br>3<br>4<br>5<br>6<br>Snap<br>  napshot-leve<br>2<br>3<br>4<br>5<br>Sna<br><br><br><br>Time<br>2<br>3<br>4<br>5<br>Sna<br><br><br><br>Time<br>napshot-lev<br>Time:|6<br>7<br>8<br>9 10<br>pshot<br>: 1.41s|
|2<br>3<br>4<br>5<br>6<br>Snap<br>  napshot-leve<br>2<br>3<br>4<br>5<br>Sna<br><br><br><br>Time<br>2<br>3<br>4<br>5<br>Sna<br><br><br><br>Time<br>napshot-lev<br>Time:||
|2<br>3<br>4<br>5<br>6<br>Snap<br>  napshot-leve<br>2<br>3<br>4<br>5<br>Sna<br><br><br><br>Time<br>2<br>3<br>4<br>5<br>Sna<br><br><br><br>Time<br>napshot-lev<br>Time:|6<br>7<br>8<br>9 10<br>pshot<br>el ApRs on<br> 0.06s|
||<br>|
|2<br>3<br>4<br>5<br><br>Sna<br>Time:|6<br>7<br>8<br>9 10<br>pshot<br> 0.56s|
||<br>|


|0 Tim|me: 0.23s|Col3|Col4|
|---|---|---|---|
|2<br>3<br>4<br>S<br>0<br>5<br><br> <br>2<br>3<br>4<br>S<br>0<br>5<br>0<br>Tim<br>Static PI-<br>Warm-sta<br>PI-GNN<br>DyCO-GN<br>        s). All me<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>Static PI-G<br>Warm-start<br>PI-GNN<br>DyCO-GNN<br>      . All meth<br>Tim|<br>|<br>||
|2<br>3<br>4<br>S<br>0<br>5<br><br> <br>2<br>3<br>4<br>S<br>0<br>5<br>0<br>Tim<br>Static PI-<br>Warm-sta<br>PI-GNN<br>DyCO-GN<br>        s). All me<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>Static PI-G<br>Warm-start<br>PI-GNN<br>DyCO-GNN<br>      . All meth<br>Tim|5<br>6<br>7<br>8<br><br>napshot<br>e: 3.32s|9 1|0|
|2<br>3<br>4<br>S<br>0<br>5<br><br> <br>2<br>3<br>4<br>S<br>0<br>5<br>0<br>Tim<br>Static PI-<br>Warm-sta<br>PI-GNN<br>DyCO-GN<br>        s). All me<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>Static PI-G<br>Warm-start<br>PI-GNN<br>DyCO-GNN<br>      . All meth<br>Tim|<br> GNN<br>rted<br>N|||
|2<br>3<br>4<br>S<br>0<br>5<br><br> <br>2<br>3<br>4<br>S<br>0<br>5<br>0<br>Tim<br>Static PI-<br>Warm-sta<br>PI-GNN<br>DyCO-GN<br>        s). All me<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>Static PI-G<br>Warm-start<br>PI-GNN<br>DyCO-GNN<br>      . All meth<br>Tim|5<br>6<br>7<br>8<br><br>napshot<br> thods use<br>e: 0.57s|9 1<br>   GC|0<br>   N|
|2<br>3<br>4<br>S<br>0<br>5<br><br> <br>2<br>3<br>4<br>S<br>0<br>5<br>0<br>Tim<br>Static PI-<br>Warm-sta<br>PI-GNN<br>DyCO-GN<br>        s). All me<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>Static PI-G<br>Warm-start<br>PI-GNN<br>DyCO-GNN<br>      . All meth<br>Tim||||
|2<br>3<br>4<br>S<br>0<br>5<br><br> <br>2<br>3<br>4<br>S<br>0<br>5<br>0<br>Tim<br>Static PI-<br>Warm-sta<br>PI-GNN<br>DyCO-GN<br>        s). All me<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>Static PI-G<br>Warm-start<br>PI-GNN<br>DyCO-GNN<br>      . All meth<br>Tim|5<br>6<br>7<br>8<br><br>napshot<br>e: 8.44s|9 1|0|
|2<br>3<br>4<br>S<br>0<br>5<br><br> <br>2<br>3<br>4<br>S<br>0<br>5<br>0<br>Tim<br>Static PI-<br>Warm-sta<br>PI-GNN<br>DyCO-GN<br>        s). All me<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>Static PI-G<br>Warm-start<br>PI-GNN<br>DyCO-GNN<br>      . All meth<br>Tim|<br> NN<br>ed|||
|2<br>3<br>4<br>S<br>0<br>5<br><br> <br>2<br>3<br>4<br>S<br>0<br>5<br>0<br>Tim<br>Static PI-<br>Warm-sta<br>PI-GNN<br>DyCO-GN<br>        s). All me<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>2<br>3<br>4<br><br>S<br><br><br><br>Tim<br>Static PI-G<br>Warm-start<br>PI-GNN<br>DyCO-GNN<br>      . All meth<br>Tim|5<br>6<br>7<br>8<br><br>napshot<br>        ods use G<br>e: 0.23s|9 1<br>          C|0<br>          N|
|||||
|2<br>3<br>4<br>5<br>S<br>Tim|6<br>7<br>8<br>9<br>napshot<br>e: 3.33s|1|0|
||<br>Static<br>Warm-<br>PI-GNN<br>DyCO-|||
||<br>Static<br>Warm-<br>PI-GNN<br>DyCO-|PI-GN<br>starte<br><br>GNN|N<br>d|
||<br>Static<br>Warm-<br>PI-GNN<br>DyCO-|||


Figure 8: Snapshot-level ApRs on dynamic MIS instance (Infectious). All methods use GCNConv.


18


MaxCut Infectious


1.00


0.95


0.90


1.00


0.95


0.90


1.00


0.75


0.50


1.00


0.75


0.50


1.00


0.75


0.50


1.00


1.00


0.75


0.50


1.00


0.75


0.50


Snapshot


0.75


0.50


Snapshot


1.0


0.5


**1000**

**1001**

**1002**


**1003**

**1004**

**1005**

**1006**

**1007**

**1008**


**1009**

**1010**

**1011**

**1012**

**1013**


**1014**

**1015**

**1016**

**1017**

**1018**

**1019**


**1020**

**1021**

**1022**

**1023**

**1024**

**1025**


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||St<br>W<br>PI<br>D|atic PI<br>arm-s<br>-GNN<br>yCO-G|-GNN<br>tarted<br>NN|-GNN<br>tarted<br>NN|-GNN<br>tarted<br>NN|-GNN<br>tarted<br>NN|-GNN<br>tarted<br>NN|-GNN<br>tarted<br>NN|


taken.


19


MIS DBLP


1.0


0.5


1.0


0.5


0.0


1.0


0.5


0.0


Snapshot


Snapshot


0.0


0.0


Snapshot


Snapshot


0.0


1.0


0.5


1.0


2 3 4 5 6 7 8 9 10
Snapshot


Snapshot


0.5


0.0


Figure 9: Snapshot-level ApRs on dynamic MIS instance (DBLP). All methods use GCNConv.


1.6


1.3


TSP burma14 Last


1.6


1.3


1.6


1.3


1.0


1.6


1.3


1.0


1.0


1.0


1.0


1.6


1.3


1.6


1.3


1.0


2.0


1.5


2.0


1.5


2.0


1.5


1.0


2.0


1.5


1.0


1.0


1.0


1.0


2.0


1.5


2.0


1.5


1.0


**1026**

**1027**


**1028**

**1029**

**1030**

**1031**

**1032**

**1033**


**1034**

**1035**

**1036**

**1037**

**1038**

**1039**


**1040**

**1041**

**1042**

**1043**

**1044**


**1045**

**1046**

**1047**

**1048**

**1049**

**1050**


**1051**

**1052**

**1053**

**1054**

**1055**

**1056**


**1057**

**1058**

**1059**

**1060**

**1061**

**1062**


**1063**

**1064**

**1065**

**1066**

**1067**


**1068**

**1069**

**1070**

**1071**

**1072**

**1073**


**1074**

**1075**

**1076**

**1077**

**1078**

**1079**


TSP ulysses22 Best


20


2.0


1.5


2.0


1.5


2.0


1.5


1.0


2.0


1.5


1.0


1.0


1.0


1.0


2.0


1.5


2.0


1.5


1.0


taken.


2.0


1.5


1.0


2.0


1.5


1.0


2.0


1.5


1.0


2.0


1.5


1.0


TSP st70 Last


2.0


1.5


1.0


2.0


1.5


1.0


2.0


1.5


1.0


2.0


1.5


1.0


2.0


1.5


1.0


2.0


1.5


1.0


TSP st70 Best


2.0


1.5


1.0


2.0


1.5


1.0


**1080**

**1081**


**1082**

**1083**

**1084**

**1085**

**1086**

**1087**


**1088**

**1089**

**1090**

**1091**

**1092**

**1093**


**1094**

**1095**

**1096**

**1097**

**1098**


**1099**

**1100**

**1101**

**1102**

**1103**

**1104**


**1105**

**1106**

**1107**

**1108**

**1109**

**1110**


**1111**

**1112**

**1113**

**1114**

**1115**

**1116**


**1117**

**1118**

**1119**

**1120**

**1121**


**1122**

**1123**

**1124**

**1125**

**1126**

**1127**


**1128**

**1129**

**1130**

**1131**

**1132**

**1133**


E PROOF AND EXTENSION OF THEOREM 1


E.1 THE GOEMANS-WILLIAMSON (GW) ALGORITHM


Let _G_ = ( _V, E_ ) be an unweighted graph with _|V |_ = _n_ nodes, and let _L_ _∈_ R _[n][×][n]_ be its Laplacian
matrix. Let _c_ _[∗]_ denote the maximum achievable cut size on this graph. Our goal is to attain this
maximum cut size.


Consider the following QUBO reformulation of the MaxCut problem on this graph.


Goemans & Williamson (1995) showed that the best approximation ratio (of at least 0.87856 times the
optimal value) for the MaxCut problem can be attained by proceeding as follows. We first reformulate
the above QUBO problem as the Semidefinite Program (SDP):


1
max s.t. _X_ _⪰_ 0 _,_ diag( _X_ ) = 1 _,_ (3)
_X_ 4 [Tr][(] _[LX]_ [)] _[,]_

where Tr( _·_ ) denotes the trace of a matrix. Let _X_ SDP _[∗]_ [denote the optimal solution obtained from solving]
the SDP. This solution is within the feasible set _X_ = _{X| X_ _⪰_ 0 _,_ diag( _X_ ) = 1 _}_ ; this is the set of
all positive definite matrices with ones on the diagonal. Once the SDP is solved, the GW algorithm
proceeds by performing a “randomized rounding” on an eigen-decomposition of the solution _X_ SDP _[∗]_ [to]
produce a binary vector representing a cut.

In more detail, consider the eigen-decomposition _X_ SDP _[∗]_ [=] _[ A]_ [Λ] _[A][T]_ [where] _[ A][ ∈]_ [R] _[n][×][n]_ [is an orthogonal]
matrix and Λ is a diagonal matrix, with the eigenvalues of _X_ SDP _[∗]_ [on its diagonal.] [Define the matrix]
1
_Y_ = _A_ Λ 2 ; this can be viewed as an “embedding” of the solution produced by the SDP. In it, each
row _Yi_ corresponds to node _i_, which needs to be mapped to +1 or _−_ 1 (this will be the label of the
node, which in turn will determines the two sets produced by the cut). To attain these node labels,
in the rounding cut, the GW rounding algorithm samples a random vector _r_ _∼N_ (0 _, In_ ) (where _In_
denotes the _n × n_ identity matrix), and defines the cut to be _xi_ = sign( _Yi_ _[T]_ _[r]_ [)][.]


E.2 PROOF OF THEOREM 1


_Proof._ First, note that the noise _Z_ has full support on the space S _[n]_ consisting of all _n × n_ symmetric
matrices, by definition. Let its distribution be denoted by _ν_ .


Next, we show that the projection Proj _X_ is continuous. This is because the space S _[n]_ is a finitedimensional real Hilbert space under the Frobenius inner product _< A, B_ _>_ = Tr( _AB_ ). The feasible
set _X_ _⊂_ S _[n]_ is the intersection of the cone of positive semidefinite matrices and an affine subspace
defined by linear constraints diag( _X_ ) = 1. Both the positive semidefinite cone and the affine subspace
are closed and convex, and therefore _X_ is a closed, convex subset of a Hilbert space. By the Hilbert
Projection Theorem (Bauschke & Combettes, 2017, Theorem 3.12) the Euclidean projection onto
any closed convex subset is continuous. Therefore, the projection function Proj _X_ is continuous.

Now define the function _f_ : S _[n]_ _→X_ as _f_ ( _Z_ ) := Proj _X_ ( _X_ 0 + _λZ_ ). This function determines the
distribution of _Xλ_ through the distribution of _Z_ . Formally, let _µλ_ = _f_ # _ν_ be the pushforward measure
on _X_, i.e., the distribution of _Xλ_ .


We now apply the following standard fact about pushforward measures: Let _ν_ be a probability
measure on a space _Y_, let _f_ : _Y_ _→_ _X_ be a measurable function, and let _µ_ = _f_ # _ν_ be the pushforward
measure on _X_ . Then for any measurable subset _A ⊆_ _X_, we have _µ_ ( _A_ ) = _ν_ ( _f_ _[−]_ [1] ( _A_ )). In particular,
if _ν_ ( _f_ _[−]_ [1] ( _A_ )) _>_ 0, then _µ_ ( _A_ ) _>_ 0.


Back to our problem setting, note that by assumption, _C_ opt _⊂X_ has positive Lebesgue measure. As
the measure _ν_ of _Z_ has full support and is absolutely continuous with respect to the Lebesgue measure
on S _[n]_, and since the projection function Proj _X_ is continuous, the pre-image Proj _[−]_ _X_ [1][(] _[C]_ [opt][)] _[ ⊂]_ [S] _[n]_ [ has]
positive _ν_ -measure for some _λ >_ 0. Therefore,


P( _R_ ( _Xλ_ ) = _c_ _[∗]_ ) = _µλ_ ( _C_ opt) _>_ 0 _._


21


min 

( _i,j_ ) _∈E_


1 _−_ _xixj_

_,_ s.t. _xi_ _∈{−_ 1 _,_ 1 _}._ (2)
2


**1134**

**1135**


**1136**

**1137**

**1138**

**1139**

**1140**

**1141**


**1142**

**1143**

**1144**

**1145**

**1146**

**1147**


**1148**

**1149**

**1150**

**1151**

**1152**


**1153**

**1154**

**1155**

**1156**

**1157**

**1158**


**1159**

**1160**

**1161**

**1162**

**1163**

**1164**


**1165**

**1166**

**1167**

**1168**

**1169**

**1170**


**1171**

**1172**

**1173**

**1174**

**1175**


**1176**

**1177**

**1178**

**1179**

**1180**

**1181**


**1182**

**1183**

**1184**

**1185**

**1186**

**1187**


Finally, note that by the assumption _X_ 0 _∈C/_ _opt_, we have P( _R_ ( _X_ 0) = _c_ _[∗]_ ) = 0. This completes the
proof.


E.3 EXTENSION


Theorem 1 shows that the probability of finding the optimal cut strictly improves if we introduce
noise pre-rounding; i.e., on a given, fixed solution of the SDP. The next corollary shows that the same
result holds if we introduce noise in the initial solution of the SDP.


We use the same notation as in Theorem 1. That is, we let the feasible set of the SDP be denoted by
_X_ = _{X| X_ _⪰_ 0 _,_ diag( _X_ ) = 1 _}_, and Proj _X_ ( _·_ ) is projection onto this feasible set _X_ . Denote the GW
rounding step by _R_ : _{X_ _,_ Ω _} →{_ 0 _,_ 1 _, . . ., c_ _[∗]_ _}_, where Ω is the random seed set of the cut plane and
_c_ _[∗]_ is the maximum achievable cut size. We again let _C_ opt = _{X_ _∈X_ : PΩ( _R_ ( _X, ω_ ) = _c_ _[∗]_ ) _>_ 0 _}_ ;
this is the set of SDP solutions that have positive probability of yielding the optimal cut after the GW
randomized rounding step.
**Corollary 1.** _Let X_ 0 _∈X_ _be the initial solution of the GW SDP. Define Xλ_ := _ProjX_ [ _X_ 0 + _λZ_ ] _,_
_where_ _λ_ _∈_ R _≥_ 0 _,_ _and_ _Z_ _is_ _a_ _symmetric_ _random_ _matrix_ _sampled_ _from_ _the_ _Gaussian_ _Orthogonal_
_Ensemble._ _Let_ Π _SDP_ ( _X_ ) _denote the SDP solver starting from a feasible solution X, and assume_
_that_ _it_ _is_ _locally_ _continuous._ _Assume_ _that_ _the_ _set_ _Copt_ _has_ _positive_ _Lebesgue_ _measure_ _in_ _X_ _,_ _that_
Π( _X_ 0) _∈C/_ _opt._ _Then, there exists a λ >_ 0 _such that_

PΩ _,Z_ ( _R_ (Π( _Xλ_ ) _, ω_ ) = _c_ _[∗]_ ) _>_ PΩ( _R_ (Π( _X_ 0) _, ω_ ) = _c_ _[∗]_ ) = 0 _._


The proof is straightforward: given the assumption that the SDP solver Π( _·_ ) is locally continuous,
that means that small perturbations of the initial SDP solution _X_ 0 map to perturbations of the solution
Π( _X_ 0). Consequently, the results of Theorem 1 are applicable. We note that SDP solvers for solving
MaxCut, such as interior-point methods, are locally continuous under mild assumptions; e.g. Shapiro
(1988).


F USE OF LARGE LANGUAGE MODELS


Large language models were used for editing purposes only.


22