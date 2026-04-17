000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Learning causal structures from observational data remains a fundamental yet computationally intensive task, particularly in high-dimensional settings where existing methods face challenges such as the super-exponential growth of the search space and increasing computational demands. To address this, we introduce VISTA (Voting-based Integration of Subgraph Topologies for Acyclicity), a modular framework that decomposes the global causal structure learning problem into local subgraphs based on Markov Blankets. The global integration is achieved through a weighted voting mechanism that penalizes low-support edges via exponential decay, filters unreliable ones with an adaptive threshold, and ensures acyclicity using a Feedback Arc Set (FAS) algorithm. The framework is modelagnostic, imposing no assumptions on the inductive biases of base learners, is compatible with arbitrary data settings without requiring specific structural forms, and fully supports parallelization. We also theoretically establish finite-sample error bounds for VISTA, and prove its asymptotic consistency under mild conditions. Extensive experiments on both synthetic and real datasets consistently demonstrate the effectiveness of VISTA, yielding notable improvements in both accuracy and efficiency over a wide range of base learners.

## 1 Introduction

Understanding causal relationships from observational data Pearl (2009) is critical across numerous fields such as biology Petersen et al. (2024), economics Hunermund & Bareinboim (2023), and ¨ healthcare Sanchez et al. (2022b). Identifying causal structures enables reliable interventions and scientific insights. A common modeling framework represents the system as a causal graph—a Directed Acyclic Graph (DAG) where nodes are variables and directed edges denote causal links Spirtes et al. (2000). While identifiability of the true DAG generally requires additional structural assumptions, our VISTA framework inherits whatever identifiability guarantees each base learner provides. In practice, large-scale observational datasets further complicate structure recovery, as most existing algorithms struggle to scale efficiently. Constraint-based pipelines Spirtes et al. (2000); Meek (2013) must search over large conditioning sets while the number of CI tests grows combinatorially with the size of graph, and finite-sample CI tests become unreliable in high dimensions, so early mistakes can easily propagate to later steps. Score-based learners Chickering (2002); Loh & Buhlmann (2014) ¨ optimize over a super-exponential DAG space; practical solvers still require heavy global searches or acyclicity constraints with repeated dense updates, driving time and memory up sharply. These disadvantages make them difficult to perform well in large-scale datasets. Given the challenges of learning large-scale causal structures, divide-and-conquer strategies have emerged as a natural solution. By decomposing the global graph into smaller, tractable subgraphs, these methods significantly reduce computational complexity, particularly in sparse settings, and facilitate parallel or distributed computation. In addition, aggregating local structures often enhances robustness relative to learning the full graph in a single pass. Early approaches expand neighborhoods from a random node Gao et al. (2017) or apply hierarchical clustering Gu & Zhou (2020). More recent work often partition the variable set into local neighborhoods, such as Markov Blankets, before aggregating them Dong et al. (2024); Mokhtarian et al. (2021); Tsamardinos et al. (2003); Wu et al. (2023; 2022). However, the majority of these "conquer" steps rely on fixed heuristics for merging, such as voting thresholds, edge overlap rules, or manual conflict resolution. While simple, such rule-based schemes lack adaptability to noise and offer limited theoretical guarantees for global

# Efficient Causal Structure Learning Via Mod- Ular Subgraph Integration

Anonymous authors Paper under double-blind review

## Abstract

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 consistency. DCILP Dong et al. (2024) formulates the merging process as an Integer Linear Program (ILP) and introduces solver-based reconciliation. Although this approach benefits from advances in ILP solvers and distributed optimization, it remains NP-hard and often incurs substantial solver overhead. In practice, even moderate-sized subproblems can lead to high memory usage and long runtimes. Alternatively, recent methods like Shah et al. (2024) retain heuristic-based fusion steps, which are efficient but similarly sensitive to noise and lack theoretical support. In this paper, we propose VISTA (Voting-based Integration of Subgraph Topologies for Acyclicity), a novel modular framework for large-scale causal discovery. The method proceeds in three main stages. First, for each variable we identify its Markov Blanket, thereby reducing the global problem into tractable local neighborhoods. A base learner is then applied to each neighborhood using the data restricted to that subset of variables, producing local subgraphs. Second, these local subgraphs are aggregated through an adaptive voting mechanism that down-weights low-support edges, suppressing statistical noise and inconsistencies. Finally, the aggregated graph is postprocessed with an efficient approximation algorithm that enforces acyclicity while preserving as many high-confidence orientations as possible. We also establish a theoretical result showing that the overall error rate of the procedure is bounded above by that of the subgraph-level aggregation, ensuring soundness of the divide-and-conquer strategy.

Crucially, VISTA is strictly model-agnostic and highly efficient. It makes no assumptions about the internal design or inductive biases of the base learners, places no restrictions on the choice of Markov Blanket identification algorithm, and imposes no conditions on the underlying data distribution beyond standard faithfulness assumptions. It operates purely on the edge-level outputs of local subgraphs and requires only a one-time O(|V | 2) aggregation without any additional solver or training overhead. This lightweight design makes VISTA framework readily compatible with any causal discovery method while enabling broad applicability across baselines and full parallelism in the divide phase. Our key contributions include:
- We propose VISTA, a model-agnostic and modular framework that decomposes global DAG learning into node-centered Markov Blanket subgraphs. It is fully plug-and-play with respect to MB identification and local learners, requiring no identifiability or distributional assumptions on the chosen base learners.

- Our aggregation is lightweight, efficient, and edge-level, performing a one-pass weighted voting instead of relying on expensive global searches or solver-based optimization. We derive finite-sample error bounds and an asymptotic consistency guarantee for this aggregation, which explicitly calibrates errors from imperfect base learners.

- Extensive experiments across diverse graphs and a wide range of base learners demonstrate that VISTA remedies the typical performance drop of base learners, consistently improving robustness and scalability over standalone baselines.

## 2 Preliminaries

Setup and notation. Let V = {V1*, . . . , V*n} be random variables generated by a structural causal model with mutually independent noises ϵi:
Vi = fi(Pa(Vi), ϵi), ϵi ⊥⊥ Pa(Vi).

This induces a directed acyclic graph (DAG) G = (V , E) where Vi →Vj ∈ E iff Vi appears in fj ,
and the observational distribution factorizes as P(V ) = Qn i=1 P(Vi| Pa(Vi)).

Markov Blanket locality. Assuming causal sufficiency for exposition, the *Markov Blanket* MB(V ) of a node V is the minimal set that renders V independent of all others given MB(V ); it consists of parents, children, and *spouses* (other parents of the children). Equivalently, MB(V ) d-separates V from V \ ({V } ∪ MB(V )). This locality motivates our divide–conquer design: by learning MB(V ), causal discovery can be restricted to the induced subgraph G[{V } ∪ MB(V )], substantially reducing search complexity while preserving relevant adjacencies for V . Existing Modular Causal Discovery Paradigms. For large-scale causal discovery, several local-toglobal or fusion-style schemes decompose a graph and then merge the pieces: a top-down CI-driven partition with set-based stitching Xie & Geng (2008), global fusion over multiple full Bayesian networks Puerta et al. (2021), a separation–reunion pipeline that repeatedly searches the structure Liu et al. (2017), a PC-style progressive skeleton requiring iterative bootstraps Guo et al. (2024), and DCILP, which formulates reconciliation as an ILP Dong et al. (2024). However, these methods are typically algorithm-specific rather than modular frameworks; they either assume correct inputs at merging time, depend on heavy global search or solver-based optimization, or perform essentially uncalibrated frequency-based stitching. There also exists a SADA-based or extended model Cai et al. (2013; 2018); Rahman et al. (2021), but it is limited to LiNGAM and lacks a calibration process during merging. By contrast, our framework provides a lightweight, calibrated weighted-voting aggregation that down-weights low-support directions and remains compatible with arbitrary base learners. A more detailed related work discussion appears in Appendix B.

## 3 Methodology

We introduce VISTA (Voting-based Integration of Subgraph Topologies for Acyclicity), a novel modular framework for large-scale DAG learning that is both model-agnostic and efficient. Instead of searching the full graph, VISTA focuses on edge-level evidence: for each node V , we form the subgraph induced by {V } ∪ MB(V ) and run any off-the-shelf local learner, regardless of its parametric form, identifiability assumptions, or internal design. The resulting local predictions are reconciled by a lightweight weighted voting on each ordered pair (*X, Y* ), which calibrates errors from imperfect base learners, and acyclicity is then enforced by a Feedback Arc Set heuristic Eades et al. (1993). This modular design makes VISTA fully plug-and-play: MB identification and local learning can be tailored to the data regime, while aggregation and acyclicity remain fixed, scalable, and consistent. Proposition 3.1 (Coverage of a DAG by Markov-Blanket Subgraphs). Let G = (V , E) *be a DAG.*
For each V ∈ V *, define*
$${\mathcal G}^{\prime}=\bigcup_{V_{i}\in V}{\mathcal G}\left[\{V_{i}\}\cup\mathrm{MB}(V_{i})\right].$$
G [{Vi} ∪ MB(Vi)] . (1)
108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Figure 1: F1 score comparison as the number of nodes increases.

Then every edge of G *is present in* G
′*, i.e.,* E ⊆ E(G

$$E({\mathcal{G}}^{\prime}).$$

Proof. Take any edge (*X, Y* ) ∈ E. If X → Y , then Y is a child of X and X is a parent of Y ,
hence Y ∈ MB(X) and X ∈ MB(Y ). Therefore (*X, Y* ) appears in G[{X} ∪ MB(X)] and in G[{Y } ∪ MB(Y )], and thus in the union G
′.

This coverage property is the foundation of VISTA: once MBs and their local subgraphs are correctly identified, no true edge is lost in the decomposition. Importantly, our framework remains *agnostic* to the specific MB estimator or local learner, that any method suitable for the data distribution can be plugged in. All subsequent aggregation and acyclicity enforcement operate purely at the edge level and rely only on this coverage guarantee. Besides, as shown in Figure 1, the accuracy of MB identification remains relatively stable as the number of nodes increases, whereas the performance of base learners degrades more sharply. This empirical observation is consistent with our theoretical analysis in Section 3.2, where we prove that the proposed merging scheme converges to the correct edge orientations. Furthermore, across different graph sizes, the VISTA-
enhanced versions consistently outperform their corresponding baselines, demonstrating the robustness of our framework. Moreover, since our framework is agnostic to the choice of MB identification methods, we also provide a flexible interface in our implementation that allows practitioners to plug in any suitable MB estimator depending on the specific

$$(1)$$

![2_image_0.png](2_image_0.png)

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 This NV rule serves to demonstrate an important property of our divide-and-conquer framework. By Theorem 3.1, every ground-truth causal edge must appear in the union of MB subgraphs. Therefore, even this unweighted scheme, which simply aggregates raw directional votes, already ensures that all true edges are included in the candidate pool. In other words, NV validates that our subgraph decomposition does not lose any causal edges, providing an essential guarantee for the global reconstruction stage. However, while NV does not distinguish between strong and weak statistical support. Edges appearing rarely across subgraphs receive the same confidence as frequently supported ones, and directional conflicts cannot be resolved in a principled manner. These issues motivate the introduction of our weighted voting formulation, which incorporates frequency-based confidence to produce more reliable global orientation decisions.

Weighted Voting (WV) For each pair of nodes X and Y , let A and B denote the number of times X → Y and Y → X appear across all subgraphs, respectively, and let m = A + B be the total occurrence. We define the confidence-adjusted score as:

$$s(X\to Y)=\left(1-e^{-\lambda m}\right){\frac{A}{m}},$$
, (2)
where λ > 0 is a tunable weighting parameter. An edge X → Y is retained if s(X → Y ) ≥ t, where t ∈ (0, 1) is a global decision threshold.

Here, the weighting term 1 − e
−λmserves as a soft confidence modulator that adapts to the reliability of directional evidence. It plays a role analogous to smoothing priors in Bayesian estimation, where rare events are regularized toward lower confidence. The details in illustrated in Appendix D.1. The inclusion threshold t determines the minimum score required to retain an edge. Compared to naive voting, which treats all local decisions equally, the weighted scheme jointly calibrates confidence and sparsity. Specifically, the parameter λ penalizes edges with weak support, while the threshold t determines the final inclusion criterion. Together, the two parameters govern the precision–recall trade-off, since a larger λ tends to preserve edges with limited but consistent evidence and thus improves recall, while a higher t enforces stricter acceptance and thereby improves precision. This mechanism is particularly beneficial in sparse graphs, where many candidate edges receive only minimal support; the exponential weighting amplifies even small differences in frequency, effectively suppressing unreliable edges. As a result, the aggregation remains robust without relying on strong parametric assumptions, and it provides a tunable handle for balancing false discoveries and missed edges. Beyond the divide-and-conquer efficiency of VISTA, the weighted voting strategy itself enhances the performance of base learners,

$$(2)$$

def VISTA(nodes, base_learner, ...,
MB_solver, lam, t):
local_graphs = [] for v in nodes:
MB_v = MB_solver(v) G_v = base_learner(MB_v ∪ v) local_graphs.append(G_v)
G_merged = WV(local_graphs, lam, t) G_final = post_prune(G_merged) return G_final Figure 2: Pseudocode of VISTA framework data distribution. Notably, we assume that each base learner outputs directed edges on local subgraphs throughout this work. If an undirected adjacency X − Y is returned, it is treated as providing no directional vote in the aggregation.

## 3.1 Vista: Voting-Based Integration Of Subgraph Topologies For Acyclicity

Naive Voting (NV) To merge estimated subgraphs into a globally causal graph, we first consider a naive voting strategy. For each pair of nodes X and Y , let A denote the number of times the directed edge X → Y appears across all subgraphs, and B denote the number of times Y → X appears. The directional support ratio for each orientation is computed as:

$$r_{X\to Y}={\frac{A}{A+B}},\quad r_{Y\to X}={\frac{B}{A+B}}.$$
.
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 yielding substantial gains in recall while tightening theoretical error bounds. A detailed analysis of these effects is provided in Section 3.2 and Appendices D - E. Acyclicity guarantee While the weighted voting improves robustness, the resulting merged graph may still contain cycles. To ensure that the final output is a valid DAG, it is necessary to explicitly break loops introduced during the merging process. So we explicitly enforce acyclicity by solving a Feedback Arc Set (FAS) problem Simpson et al. (2016). As FAS is NP-hard, we adopt a fast GreedyFAS heuristic Eades et al. (1993) adapted to weighted edges; the implementation is detailed in Algorithm 2 in Appendix C. Notably, an important implementation detail involves the ordering between GreedyFAS and thresholdbased filtering. In VISTA, cycles are first removed using GreedyFAS, after which edges with weights below a global threshold t are filtered out. This ordering avoids forcing the cycle removal step to act on already sparse graphs, where eliminating a cycle may require discarding high-confidence edges. In contrast, applying filtering before GreedyFAS can lead to unnecessary precision loss, as the remaining cycles must be resolved by removing stronger edges that would otherwise have been preserved. Besides, taking a subset of nodes from a causal graph introduces unobserved confounding, which will lead to additional edges in the subgraph; the post-processing step here can remove part of these redundant edges. In general, our VISTA offers several key advantages that make it particularly suited for large-scale causal discovery. It operates purely on aggregated edge counts and requires only matrix-level operations, with no reliance on optimization solvers or iterative training. Importantly, it is modelagnostic, i.e., the aggregation is independent of the internal structure of base learners and can be applied to any method that outputs directed subgraphs. This modularity allows seamless integration with a broad class of causal discovery algorithms and supports parallel execution in the divide stage. The complete procedure is implemented as a simple and modular pipeline, summarized in Figure 3.

![4_image_0.png](4_image_0.png)

Figure 3: Overview of VISTA, a modular framework for causal discovery: (1) dividing via Markov Blankets identification, (2) parallel subgraph structure identification using a base learner, and (3) global aggregation through weighted voting. The framework then applies cycle resolution (Greedy- FAS) and weight-based filtering to produce the final DAG.

Theoretical guarantees for Weighted Voting To ensure the reliability of our edge orientation decisions based on the weighted voting mechanism described above, we provide theoretical guarantees derived from concentration inequalities. The core idea is to determine the minimum number of votes
(subgraphs) m required to achieve a desired level of confidence 1 − ϵ in our decision.

Theorem 3.2 (Sufficient Condition for Weighted Voting Accuracy). Let A ∼ Binomial(*m, p*) represent the number of successful votes in m independent subgraphs for the edge direction X → Y , where each subgraph supports this direction independently with probability p ∈ (0, 1)*, a decision* threshold t ∈ (0, 1) *and the weight function* w(m) = 1 − e
−λm, λ > 0. Assume the effective

it follows that $P\left(s(A)\geq t\right)\geq1-\epsilon$. 
$${\frac{m p}{2}}\left(1-{\frac{t}{p(1-e^{-\lambda m})}}\right)^{2}\geq\log{\frac{1}{\epsilon}},$$
$$({\mathfrak{I}}{\mathfrak{I}})$$
, (3)
threshold for accepting the edge direction X → Y is r(m) = t 1−e−λm < p*, i.e., the true support rate* p *is above the effective threshold. Then, if* 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 This theorem guarantees that if m is large enough to satisfy the given inequality, the weighted voting procedure will correctly identify the edge direction with high probability. The condition highlights that the required m depends on the squared relative difference between the true probability p and the effective threshold r(m). Note that r(m) itself depends on m and λ. As m increases or λ increases, 1 − e
−λm approaches 1, and r(m) approaches t. The inequality requires larger m and becomes more difficult to satisfy when p is close to r(m) or when higher confidence is desired. This trade-off illustrates the role of λ in controlling the conservativeness of the decision rule, which we will analyze further in later sections. In practice, the true value of p is unknown, but we can empirically validate the trend predicted by this condition using observed vote frequencies and measured recovery accuracy across different values of λ and t. Notably, Theorem 3.2 is stated under an idealized assumption that the votes from different local subgraphs are independent. In practice, subgraphs learned from the same dataset can induce correlations among votes, so the bound should be interpreted as a qualitative guide, and we expect the same monotone trend to hold more effectively independent votes still reduce error and the gap between p and the effective threshold continues to govern sample complexity. Extending the theory to low-correlation weakly dependent votes will be an interesting future direction. Corollary 3.3 (Lower bound on node in subgraphs). Let λ > 0, t ∈ (0, 1), and ϵ ∈ (0, 1) be fixed.

For a candidate edge (X, Y ), denote by m the number of local subgraphs whose Markov Blankets contain both endpoints. Under the setting of Theorem 3.2, the sufficient condition (3) can be converted into an explicit bound

$$m\geq{\frac{2\log(1/\epsilon)}{p\left((1-t/p)^{2}-2(t/p)(1-t/p)e^{-\lambda}\right)}}.$$
$$(4)$$

Generally, a lower error rate ϵ leads to a larger log(1/ϵ) term, which increases the required size of m. When p is much greater than t, it results in a small required m. This aligns with intuition: if the true voting rate p is far from the threshold t, the distinction is easier, and fewer votes are needed for reliable decisions. Similarly, when the gap p − t is small, it will result in a significantly larger required m. A large lower bound on m primarily indicates that the current setting yields a very small gap between p and t, which, in turn, implies that the decision task has intrinsically high sample complexity.

## 3.2 Error Bound Analysis

We analyze the edge-level errors of the weighted voting rule to understand how the weighting parameter λ and the threshold t affect false positives and false negatives. We first characterize a sufficient condition that converts t into a probability threshold and yields a feasible range for λ, and then show that under this regime, weighted voting achieves asymptotic consistency as the graph size grows.

Theorem 3.4 (Practical choice of λ). Fix a vote count m ≥ 1, a decision threshold t ∈ (0, 1), and a target error level ϵ ∈ (0, 1). If λ *satisfies*

−
$$-\frac{1}{m}\ln(1-t)\;<\;\lambda\;\leq\;-\frac{1}{m}\ln\epsilon,$$
ln ϵ, (5)
Theorem 3.4 establishes a feasible interval for λ that guarantees uniform control of edge-level errors. While the confidence weight 1−e
−λm down-weights low-support orientations at a fixed t, the smaller λ values impose stricter thresholds rλ(m) to suppress low-support edges, while larger values retain weaker true edges and improve recall. The proof of the theorem, as well as detailed discussions, is in Appendix E.1. In practice, we adopt the relatively large admissible λ in (5), which lowers the

$$(S)$$

then the weighted-vote rule achieves the prescribed error control under the union bound. effective threshold and reduces false negatives at the cost of more false positives. This choice is well suited to sparse graphs since false positives typically dominate. The empirical behavior of varying λ is further examined in Section 4.1. Notably, as λ → 0, the rule reduces to naive voting with a fixed threshold t. Building on the finite-sample guarantees above, we next analyze the asymptotic behavior of the weighted voting rule as the number of variables grows. Similarly to p, let q ∈ (0, 1) denote the probability that a false edge is erroneously included. In practice, both p and q can be empirically estimated.

Theorem 3.5 (Asymptotic Consistency). Fix a threshold t ∈ (0, 1) and let δp = p − t and δq = t − q denote the positive margins between t and the inclusion probabilities p, q of true and false edges respectively. Assume δp, δq > 0 and that λ satisfies the conditions in Theorem 3.4. If the number of local subgraphs per candidate edge is m = C log n with C > 2 min{δ 2p
,δ2q }
, then we have

$$=o(1)$$
), $\qquad as\;\;n\to\infty$. 

## Pr(Global Error) = O(1), As N → ∞. (6)

Since most base solvers are reliable and can correctly identify a substantial fraction of true edges, our assumptions are quite mild and practically easy to satisfy. Theorem 3.5 establishes that weighted voting is asymptotically consistent: as the number of subgraph samples increases, the probability of edge-level misclassification vanishes. Notably, the required number of independent subgraphs per edge grows only logarithmically with the graph size, i.e., O(log n), making the approach efficient. From a computational perspective, the global merging procedure involves only one pass of edge counting and scoring, with an overall complexity O(n 2) regardless of the base learner. These guarantees jointly ensure that the method remains scalable and reliable for large-scale structure discovery. The proof of the theorem is provided in Appendix E.3.

## 4 Experiments 4.1 Synthetic Data

We empirically evaluate the performance of the proposed VISTA framework on a range of graph structures and sizes, as well as diverse base learners. To demonstrate the improvement and effectiveness of VISTA, we report representative results that highlight the structural recovery performance of VISTA, its runtime benefits from our modular strategy, and the precision–recall trade-offs induced by different values of λ. All experiments are conducted on a machine equipped with 13th Gen Intel(R) Core(TM) i9-13900HX CPU (24 cores) and NVIDIA A30 GPU (24GB).

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Baselines We benchmark VISTA against recent typical state-of-the-art causal discovery algorithms, including CAM Buhlmann & Peters (2016), NOTEARS Zheng et al. (2018), DAG-GNN Yu et al. ¨ (2019), and GOLEM Ng et al. (2020) for the linear setting, which we modeled as linear Structural Equation Model (SEM) with Gaussian noise, as well as SCORE Rolland et al. (2022) and GraN-DAG
Lachapelle et al. (2020) for the nonlinear setting, defined as quadratic SEM. Each baseline is evaluated both in isolation and when integrated with our modular framework VISTA. Additionally, in Appendix F.2, we provide a comparison between VISTA and DCILP Dong et al. (2024), a recent distributed framework for causal structure learning, where we also implemented the MB solver used in that work. We evaluate the accuracy of our VISTA framework under the Naive Voting (NV) and the Weighted Voting (WV) aggregation schemes. Each base learner is tested standalone and with both VISTA variants. We evaluate the proposed method on synthetic datasets generated from Erdos–R ˝ enyi ´ (ER) and scale-free (SF) graphs, with average out-degree h ∈ {3, 5} and number of nodes n ∈ {30, 50, 100, 300}. Performance is assessed using False Discovery Rate (FDR), True Positive Rate (TPR), Structural Hamming Distance (SHD), and F1 score, as well as runtime metrics. Experiments are conducted under multiple simulation settings, and we report the average performance, with the ± values indicating the corresponding standard deviations. Results Table 1 shows two complementary roles of our aggregation. The NV variant already lifts recall by pooling evidence from overlapping neighborhoods, recovering more true edges. Building on this, WV acts as a principled edge-level filter. By down-weighting orientations with small or inconsistent support and applying a single global threshold, it removes noisy connections and yields substantially cleaner structures. Quantitatively, WV reduces FDR by 50 ∼ 80% relative to the original baselines and by 40 ∼ 70% compared to NV, while generally keeping TPR no less than 0.70.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

| Table 1: Results with linear and nonlinear synthetic datasets (n = 100, h = 5). ER5 SF5   |             |                 |                  |             |             |                 |                  |             |
|-------------------------------------------------------------------------------------------|-------------|-----------------|------------------|-------------|-------------|-----------------|------------------|-------------|
| Method                                                                                    | FDR↓        | TPR↑            | SHD↓             | F1↑         | FDR↓        | TPR↑            | SHD↓             | F1↑         |
| NOTEARS                                                                                   | 0.21 ± 0.21 | 0.74 ± 0.26     | 208.80 ± 199.71  | 0.76 ± 0.24 | 0.37 ± 0.15 | 0.60 ± 0.14     | 352.60 ± 125.39  | 0.61 ± 0.14 |
| +VISTA-NV                                                                                 | 0.87 ± 0.01 | 0.97 ± 0.01     | 3171.80 ± 174.02 | 0.23 ± 0.01 | 0.84 ± 0.01 | 0.97 ± 0.01     | 2443.60 ± 143.74 | 0.27 ± 0.01 |
| +VISTA-WV 0.08 ± 0.03                                                                     | 0.68 ± 0.01 | 182.40 ± 16.03  | 0.79 ± 0.02      | 0.18 ± 0.07 | 0.68 ± 0.03 | 233.00 ± 34.76  | 0.74 ± 0.03      |             |
| GOLEM                                                                                     | 0.61 ± 0.16 | 0.35 ± 0.17     | 567.00 ± 129.77  | 0.35 ± 0.15 | 0.70 ± 0.15 | 0.29 ± 0.19     | 610.10 ± 118.00  | 0.29 ± 0.17 |
| +VISTA-NV                                                                                 | 0.87 ± 0.01 | 0.91 ± 0.04     | 2891.00 ± 224.42 | 0.23 ± 0.01 | 0.86 ± 0.01 | 0.90 ± 0.02     | 2589.00 ± 270.09 | 0.25 ± 0.02 |
| +VISTA-WV 0.23 ± 0.12                                                                     | 0.50 ± 0.13 | 306.70 ± 87.75  | 0.60 ± 0.14      | 0.33 ± 0.15 | 0.40 ± 0.12 | 371.10 ± 88.21  | 0.50 ± 0.13      |             |
| DAG-GNN                                                                                   | 0.66 ± 0.15 | 0.42 ± 0.23     | 739.20 ± 323.34  | 0.35 ± 0.17 | 0.64 ± 0.15 | 0.47 ± 0.22     | 731.40 ± 303.38  | 0.38 ± 0.17 |
| +VISTA-NV                                                                                 | 0.87 ± 0.01 | 0.95 ± 0.01     | 3065.00 ± 136.49 | 0.23 ± 0.01 | 0.85 ± 0.01 | 0.95 ± 0.00     | 2480.00 ± 203.65 | 0.27 ± 0.01 |
| +VISTA-WV 0.36 ± 0.03                                                                     | 0.56 ± 0.05 | 377.00 ± 26.06  | 0.59 ± 0.02      | 0.35 ± 0.10 | 0.49 ± 0.08 | 363.00 ± 41.10  | 0.56 ± 0.09      |             |
| GraN-DAG                                                                                  | 0.92 ± 0.04 | 0.05 ± 0.03     | 715.00 ± 70.14   | 0.06 ± 0.04 | 0.94 ± 0.02 | 0.05 ± 0.03     | 1088.60 ± 31.49  | 0.05 ± 0.02 |
| +VISTA-NV                                                                                 | 0.86 ± 0.04 | 0.18 ± 0.06     | 656.60 ± 83.30   | 0.16 ± 0.03 | 0.89 ± 0.02 | 0.20 ± 0.04     | 947.20 ± 53.33   | 0.14 ± 0.02 |
| +VISTA-WV 0.43 ± 0.06                                                                     | 0.10 ± 0.02 | 503.40 ± 46.68  | 0.17 ± 0.03      | 0.54 ± 0.05 | 0.11 ± 0.02 | 545.80 ± 65.54  | 0.18 ± 0.03      |             |
| SCORE                                                                                     | 0.92 ± 0.10 | 0.58 ± 0.03     | 4039.60 ± 123.3  | 0.14 ± 0.15 | 0.91 ± 0.03 | 0.62 ± 0.05     | 3166.40 ± 258.7  | 0.16 ± 0.05 |
| +VISTA-NV                                                                                 | 0.95 ± 0.08 | 0.76 ± 0.02     | 3464.20 ± 215.6  | 0.09 ± 0.14 | 0.95 ± 0.04 | 0.76 ± 0.05     | 2978.00 ± 367.3  | 0.08 ± 0.07 |
| +VISTA-WV 0.80 ± 0.06                                                                     | 0.65 ± 0.07 | 838.00 ± 364.78 | 0.31 ± 0.09      | 0.81 ± 0.05 | 0.63 ± 0.04 | 892.60 ± 345.58 | 0.29 ± 0.06      |             |

The trend holds for both differentiable and combinatorial base learners, indicating that the gains stem from the aggregation rule rather than any particular estimator. Crucially, λ appears only in the final aggregation, so sweeping it is retraining-free: we reuse cached votes, recompute rλ(m), and rerun the DAG projection to obtain the full curves. To avoid perdataset hyperparameter tuning and cherry-picking, all VISTA results in the main tables use a single, fixed operating point: λ = 0.5 and t = 0.7. This choice lies within (5) and serves as a stable compromise between precision and recall across settings. We report the full precision–recall curves for transparency, but no post-hoc selection is performed for the tabulated results. The observed improvement in WV cases against NV aligns with Theorem 3.4. Edges with limited empirical support are selectively pruned while strongly supported ones are preserved, which is exactly the filtering behavior reflected in Table 1. This validates our weighted voting scheme as an effective, model-agnostic mechanism for stabilizing global structures. To further substantiate this model-agnostic property, we next examine the impact of data standardization as it is known to influence baseline performance Reisach et al. (2021). The results show that, regardless of fluctuations in the performance of individual base learners, the improvements brought by VISTA remain consistent. This stability further supports our claim that VISTA does not rely on any inductive bias of the base learner or data distribution. Rather, the edge-level aggregation mechanism provides robustness across settings. These findings further highlight the model-agnostic nature of our framework. Additional experiments under alternative parameter settings are provided in Appendix F.4.

| Table 2: Results with normalized linear and nonlinear synthetic datasets (n = 50, h = 5). ER5 SF5   |             |                |                 |             |             |                |                |             |
|-----------------------------------------------------------------------------------------------------|-------------|----------------|-----------------|-------------|-------------|----------------|----------------|-------------|
| Method                                                                                              | FDR↓        | TPR↑           | SHD↓            | F1↑         | FDR↓        | TPR↑           | SHD↓           | F1↑         |
| NOTEARS                                                                                             | 0.04 ± 0.02 | 0.39 ± 0.01    | 140.00 ± 4.90   | 0.56 ± 0.01 | 0.02 ± 0.02 | 0.38 ± 0.04    | 138.50 ± 9.87  | 0.55 ± 0.05 |
| +VISTA-NV                                                                                           | 0.27 ± 0.05 | 0.61 ± 0.03    | 135.20 ± 6.16   | 0.66 ± 0.02 | 0.35 ± 0.04 | 0.62 ± 0.04    | 132.80 ± 18.82 | 0.63 ± 0.03 |
| +VISTA-WV 0.19 ± 0.05                                                                               | 0.58 ± 0.03 | 122.90 ± 7.54  | 0.68 ± 0.02     | 0.08 ± 0.04 | 0.54 ± 0.06 | 109.10 ± 19.91 | 0.68 ± 0.05    |             |
| GOLEM                                                                                               | 0.40 ± 0.03 | 0.22 ± 0.04    | 182.00 ± 15.51  | 0.32 ± 0.05 | 0.44 ± 0.07 | 0.20 ± 0.04    | 183.60 ± 6.55  | 0.29 ± 0.05 |
| +VISTA-NV                                                                                           | 0.31 ± 0.03 | 0.75 ± 0.03    | 129.50 ± 4.97   | 0.72 ± 0.02 | 0.29 ± 0.05 | 0.70 ± 0.05    | 122.80 ± 19.87 | 0.70 ± 0.04 |
| +VISTA-WV 0.06 ± 0.03                                                                               | 0.62 ± 0.04 | 95.30 ± 9.88   | 0.75 ± 0.02     | 0.10 ± 0.04 | 0.60 ± 0.06 | 100.20 ± 15.69 | 0.72 ± 0.05    |             |
| DAG-GNN                                                                                             | 0.16 ± 0.03 | 0.41 ± 0.05    | 160.80 ± 53.55  | 0.55 ± 0.05 | 0.19 ± 0.05 | 0.48 ± 0.04    | 183.60 ± 45.37 | 0.60 ± 0.03 |
| +VISTA-NV                                                                                           | 0.85 ± 0.09 | 0.74 ± 0.14    | 609.80 ± 72.70  | 0.25 ± 0.12 | 0.79 ± 0.04 | 0.72 ± 0.09    | 538.40 ± 25.55 | 0.33 ± 0.05 |
| +VISTA-WV 0.14 ± 0.05                                                                               | 0.50 ± 0.09 | 93.50 ± 29.12  | 0.63 ± 0.07     | 0.13 ± 0.08 | 0.56 ± 0.06 | 87.80 ± 16.56  | 0.68 ± 0.05    |             |
| GraN-DAG                                                                                            | 0.82 ± 0.01 | 0.06 ± 0.01    | 275.00 ± 18.50  | 0.09 ± 0.01 | 0.92 ± 0.02 | 0.02 ± 0.02    | 269.80 ± 45.50 | 0.03 ± 0.02 |
| +VISTA-NV                                                                                           | 0.66 ± 0.15 | 0.26 ± 0.06    | 219.20 ± 46.41  | 0.29 ± 0.07 | 0.68 ± 0.05 | 0.17 ± 0.04    | 223.00 ± 26.25 | 0.22 ± 0.04 |
| +VISTA-WV 0.15 ± 0.06                                                                               | 0.18 ± 0.05 | 199.20 ± 13.64 | 0.32 ± 0.07     | 0.33 ± 0.03 | 0.13 ± 0.03 | 205.40 ± 59.15 | 0.23 ± 0.04    |             |
| SCORE                                                                                               | 0.71 ± 0.05 | 0.50 ± 0.05    | 386.80 ± 67.99  | 0.37 ± 0.04 | 0.65 ± 0.13 | 0.52 ± 0.15    | 340.40 ± 81.08 | 0.38 ± 0.05 |
| +VISTA-NV                                                                                           | 0.79 ± 0.03 | 0.60 ± 0.14    | 489.70 ± 123.82 | 0.31 ± 0.04 | 0.77 ± 0.03 | 0.56 ± 0.05    | 471.10 ± 16.68 | 0.33 ± 0.03 |
| +VISTA-WV 0.64 ± 0.09                                                                               | 0.42 ± 0.11 | 305.80 ± 49.93 | 0.39 ± 0.07     | 0.57 ± 0.04 | 0.36 ± 0.06 | 244.20 ± 53.35 | 0.39 ± 0.04    |             |

Time efficiency To assess the scalability of our framework, we report the total computation time for different base learners in Table 3. All results are presented as mean ± standard deviation over repeated runs. Across all tested graph sizes, integrating VISTA consistently yields substantial runtime 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 reductions compared to the original methods. These improvements are not due to algorithmspecific acceleration but result directly from our divide-and-conquer design: since each local subgraph is processed independently, the learning procedure naturally supports parallel execution. This decomposition effectively reduces the per-task computational load and alleviates memory bottlenecks, enabling scalable causal discovery even with large node counts. Further results for other settings are included in Appendix F.3.

Table 3: Comparison of total computing time (s) under ER3 setting.

Method n = 50 n = 100 n = 300 NOTEARS 494.40 ± 98.24 1473.69 ± 395.59 12515.63 ± 1599.06
+VISTA 189.15 ± 65.37 339.90 ± 158.75 2136.72 ± 708.15 GOLEM 72.65 ± 15.41 108.82 ± 70.56 261.84 ± 30.44
+VISTA 21.93 ± 0.81 26.16 ± 2.68 43.40 ± 3.21 DAG-GNN 628.63 ± 55.29 2192.97 ± 323.59 17713.84 ± 2861.06
+VISTA 201.31 ± 43.36 371.25 ± 199.91 1960.43 ± 794.02 GraN-DAG 730.42 ± 89.95 3035.76 ± 481.85 25205.64 ± 2098.85
+VISTA 238.53 ± 51.36 472.30 ± 172.77 2336.32 ± 1028.04 SCORE 426.63 ± 61.15 10040.65 ± 209.31 ——–
+VISTA 105.64 ± 39.65 198.82 ± 34.12 225.16 ± 11.45 Sensitivity study of λ We sweep λ and plot precision/recall in Figure 4. By the conclusion of Theorem 3.4 and Appendix E.1, larger λ shifts the method toward higher recall and lower precision by relaxing the penalty on low-support edges. Within the theoretical range, this precision–recall trade-off is smooth and yields informative voting thresholds rλ(m). The figure also substantiate this point, Small λ strongly discounts low-support edges, yielding high precision and low recall. Similarly, as λ increases, recall rises while precision falls. Beyond the upper end of (5) we have
(1 − e
−λm) ≈ 1 and thus s(X → Y ) ≈ A/m, so the curves plateau and further increases of λ have negligible effect. Therefore, to balance precision and recall in practice, a moderate value of the hyperparameter could be fixed within the theoretical range, which serves as a stable operating point.

![8_image_0.png](8_image_0.png)

![8_image_1.png](8_image_1.png)

## 4.2 Real Data

Table 4: Results on the Sachs protein-signaling network.

Method FDR↓ TPR↑ SHD↓ SID↓ GOLEM 0.80 0.26 16 50

+VISTA 0.57 0.18 16 48

SCORE 0.81 0.18 18 57

+VISTA 0.60 0.12 15 53

DAG-GNN 0.50 0.12 15 54

+VISTA 0.25 0.18 14 52

GraN-DAG 0.82 0.53 16 48

+VISTA 0.00 0.29 12 45

We further evaluate all methods on the wellknown Sachs protein signaling network based on expression levels of proteins and phospholipids Sachs et al. (2005). This benchmark is widely used in causal discovery research, and the ground-truth graph with 11 nodes and 17 directed edges is consistently accepted by the community.

Here we trained normalized data with 853 samples and reported the results in Table 4.

Incorporating VISTA consistently reduces false discoveries and improves structural accuracy, measured by SHD and SID Peters & Buhlmann (2015) across different baselines. This highlights that VISTA is a plug-and-play module ¨ that can reliably enhance the performance of arbitrary causal discovery algorithms.

9

## 5 Conclusion

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

## Reproducibility Statement

We provide the code in the supplementary material, together with a README file that allows experimental results to be reproduced.

THE USE OF LLM
We used LLM to polish the writing and correct grammar in some paragraphs, but it did not contribute to ideas or conceptual content.

## References

Steen A Andersson, David Madigan, and Michael D Perlman. A characterization of markov equivalence classes for acyclic digraphs. *The Annals of Statistics*, 25(2):505–541, 1997.

Taiyu Ban, Lyuzhou Chen, Xiangyu Wang, Xin Wang, Derui Lyu, and Huanhuan Chen. Differentiable structure learning with partial orders. In *The Thirty-eighth Annual Conference on Neural* Information Processing Systems, 2024.

Kevin Bello, Bryon Aragam, and Pradeep Ravikumar. Dagma: Learning dags via m-matrices and a log-determinant acyclicity characterization. *Advances in Neural Information Processing Systems*,
35:8226–8239, 2022.

Peter Buhlmann and Jonas Peters. CAM: Causal additive models, high-dimensional order search and ¨
penalized regression. *The Annals of Statistics*, 44(1):243–274, 2016.

Ruichu Cai, Zhenjie Zhang, and Zhifeng Hao. Sada: A general framework to support robust causation discovery. In Sanjoy Dasgupta and David McAllester (eds.), Proceedings of the 30th International Conference on Machine Learning, volume 28 of *Proceedings of Machine Learning Research*, pp. 208–216, Atlanta, Georgia, USA, 17–19 Jun 2013. PMLR.

Ruichu Cai, Zhenjie Zhang, Zhifeng Hao, and Marianne Winslett. Sophisticated merging over random partitions: A scalable and robust causal discovery approach. IEEE Transactions on Neural Networks and Learning Systems, 29(8):3623–3635, 2018. doi: 10.1109/TNNLS.2017.2734804.

David Maxwell Chickering. Optimal structure identification with greedy search. Journal of machine learning research, 3(Nov):507–554, 2002.

In this paper, we introduced VISTA, a scalable and model-agnostic framework for causal discovery that decomposes global structure learning into Markov Blanket neighborhoods, aggregates them via a weighted voting scheme, and enforces acyclicity through FAS post-processing. The design is fully parallelizable, and the aggregation step operates only at the edge level, enabling efficient exploration of operating points regardless of the base learner. Theoretically, we establish finite-sample error guarantees and asymptotic consistency under mild conditions. Empirically, across diverse graph families and base learners, VISTA improves accuracy and runtime efficiency, typically increasing precision without sacrificing recall. Despite the favorable performance of VISTA, the framework has several limitations. First, when aggregating local graphs, latent confounding introduced by restricting the learner to subsets may produce high-confidence redundant edges. In some cases these edges do not necessarily participate in cycles and our current framework can only mitigate them through the combination of GreedyFAS and threshold-based filtering. Moreover, although the FAS projection guarantees acyclicity, it may also prune edges that are weakly supported yet correct, which can negatively affect downstream tasks that are sensitive to edge directions. Future work includes incorporating interventional data to improve orientation accuracy and extending the VISTA framework to online settings for large-scale applications.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 David Maxwell Chickering, David Heckerman, and Christopher Meek. Large-sample learning of bayesian networks is np-hard. *Journal of Machine Learning Research*, 5:1287–1330, 2004.

Davin Choo, Kirankumar Shiragur, and Arnab Bhattacharyya. Verification and search algorithms for causal dags. *Advances in Neural Information Processing Systems*, 35:12787–12799, 2022.

Diego Colombo, Marloes H Maathuis, Markus Kalisch, and Thomas S Richardson. Learning highdimensional directed acyclic graphs with latent and selection variables. *The Annals of Statistics*, pp. 294–321, 2012.

Shuyu Dong, Michele Sebag, Kento Uemura, Akito Fujii, Shuang Chang, Yusuke Koyanagi, and Koji `
Maruhashi. DCDILP: a distributed learning method for large-scale causal structure learning. arXiv preprint arXiv:2406.10481, 2024.

Peter Eades, Xuemin Lin, and William F Smyth. A fast and effective heuristic for the feedback arc set problem. *Information Processing Letters*, 47(6):319–323, 1993. doi: 10.1016/0020-0190(93) 90079-O.

Zhuangyan Fang, Shengyu Zhu, Jiji Zhang, Yue Liu, Zhitang Chen, and Yangbo He. On low-rank directed acyclic graphs and causal structure learning. IEEE Transactions on Neural Networks and Learning Systems, 35(4):4924–4937, 2023.

Tian Gao, Kshitij Fadnis, and Murray Campbell. Local-to-global bayesian network structure learning.

In *International Conference on Machine Learning*, pp. 1193–1202. PMLR, 2017.

Jiaying Gu and Qing Zhou. Learning big Gaussian Bayesian networks: Partition, estimation and fusion. *Journal of machine learning research*, 21(158):1–31, 2020.

Xianjie Guo, Kui Yu, Lin Liu, Jiuyong Li, Jiye Liang, Fuyuan Cao, and Xindong Wu. Progressive skeleton learning for effective local-to-global causal structure learning. IEEE Transactions on Knowledge and Data Engineering, 36(12):9065–9079, 2024.

Alain Hauser and Peter Buhlmann. Characterization and greedy learning of interventional markov ¨
equivalence classes of directed acyclic graphs. *The Journal of Machine Learning Research*, 13(1): 2409–2464, 2012.

Yang-Bo He and Zhi Geng. Active learning of causal networks with intervention experiments and optimal designs. *Journal of Machine Learning Research*, 9(Nov):2523–2547, 2008.

Biwei Huang, Charles Jia Han Low, Feng Xie, Clark Glymour, and Kun Zhang. Latent hierarchical causal structure discovery with rank constraints. Advances in neural information processing systems, 35:5549–5561, 2022.

Paul Hunermund and Elias Bareinboim. Causal inference and data fusion in econometrics. ¨ The Econometrics Journal, pp. utad008, 2023.

Maximilian Kaiser, Stefan Bauer, and Bernhard Scholkopf. Bootstrap aggregation and confidence ¨
measures for time-series causal discovery. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024.

Sebastien Lachapelle, Philippe Brouillard, Tristan Deleu, and Simon Lacoste-Julien. Gradient-based ´
neural dag learning. In *International Conference on Learning Representations*, 2020.

Thuc Duy Le, Tao Hoang, Jiuyong Li, Lin Liu, Huawen Liu, and Shu Hu. A fast PC algorithm for high dimensional causal discovery with multi-core pcs. IEEE/ACM transactions on computational biology and bioinformatics, 16(5):1483–1495, 2016.

Phillip Lippe, Taco Cohen, and Efstratios Gavves. Efficient neural causal discovery without acyclicity constraints. *arXiv preprint arXiv:2107.10483*, 2021.

Hui Liu, Shuigeng Zhou, Wai Lam, and Jihong Guan. A new hybrid method for learning bayesian networks: Separation and reunion. *Knowledge-Based Systems*, 121:185–197, 2017. ISSN 09507051.

Po-Ling Loh and Peter Buhlmann. High-dimensional learning of linear causal networks via inverse ¨
covariance estimation. *Journal of Machine Learning Research*, 15(140):3065–3105, 2014.

Sindy Lowe, David Madras, Richard Zemel, and Max Welling. Amortized causal discovery: Learning ¨
to infer causal graphs from time-series data. In *Conference on Causal Learning and Reasoning*,
pp. 509–525. PMLR, 2022.

Christopher Meek. Causal inference and causal explanation with background knowledge. arXiv preprint arXiv:1302.4972, 2013.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Ehsan Mokhtarian, Sina Akbari, AmirEmad Ghassami, and Negar Kiyavash. A recursive markov boundary-based approach to causal structure learning. In *The KDD'21 Workshop on Causal* Discovery, pp. 26–54. PMLR, 2021.

Francesco Montagna, Nicoletta Noceti, Lorenzo Rosasco, Kun Zhang, and Francesco Locatello.

Causal discovery with score matching on additive models with arbitrary noise. In *Conference on* Causal Learning and Reasoning, pp. 726–751. PMLR, 2023a.

Francesco Montagna, Nicoletta Noceti, Lorenzo Rosasco, Kun Zhang, and Francesco Locatello.

Scalable causal discovery with score matching. In *Conference on Causal Learning and Reasoning*, pp. 752–771. PMLR, 2023b.

Ivan Ng, Xun Zheng, and Bryon Aragam. Learning sparse causal models is not np-hard. In Advances in Neural Information Processing Systems, volume 33, pp. 16888–16900, 2020.

Judea Pearl. *Causality*. Cambridge university press, 2009. Jonas Peters and Peter Buhlmann. Structural intervention distance for evaluating causal graphs. ¨
Neural Computation, 27(3):771–799, 2015. doi: 10.1162/NECO a 00708.

Jonas Peters, Joris M Mooij, Dominik Janzing, and Bernhard Scholkopf. Causal discovery with ¨
continuous additive noise models. *The Journal of Machine Learning Research*, 15(1):2009–2053, 2014.

Anne Helby Petersen, Claus Thorn Ekstrøm, Peter Spirtes, and Merete Osler. Causal discovery and epidemiology: A potential for synergy. *American Journal of Epidemiology*, pp. kwae101, 2024.

Jose M. Puerta, Juan A. Aledo, Jos ´ e A. G ´ amez, and Jorge D. Laborda. Efficient and accurate structural ´
fusion of bayesian networks. *Information Fusion*, 66:155–169, 2021. ISSN 1566-2535.

Md Musfiqur Rahman, Ayman Rasheed, Md Mosaddek Khan, Mohammad Ali Javidian, Pooyan Jamshidi, and Md Mamun-Or-Rashid. Accelerating recursive partition-based causal structure learning. *arXiv preprint arXiv:2102.11545*, 2021.

Joseph Ramsey, Madelyn Glymour, Ruben Sanchez-Romero, and Clark Glymour. A million variables and more: the fast greedy equivalence search algorithm for learning high-dimensional graphical causal models, with an application to functional magnetic resonance images. *International Journal* of Data Science and Analytics, 3(2):121–129, 2017. doi: 10.1007/s41060-016-0032-z.

Alexander Reisach, Christof Seiler, and Sebastian Weichwald. Beware of the simulated dag! causal discovery benchmarks may be easy to game. *Advances in Neural Information Processing Systems*, 34:27772–27784, 2021.

Paul Rolland, Volkan Cevher, Matthaus Kleindessner, Chris Russell, Dominik Janzing, Bernhard ¨
Scholkopf, and Francesco Locatello. Score matching enables causal discovery of nonlinear additive ¨ noise models. In *International Conference on Machine Learning*, pp. 18741–18753. PMLR, 2022.

Karen Sachs, Omar Perez, Dana Pe'er, Douglas A. Lauffenburger, and Garry P. Nolan. Causal protein-signaling networks derived from multiparameter single-cell data. *Science*, 308(5721): 523–529, 2005. doi: 10.1126/science.1105809.

Pedro Sanchez, Xiao Liu, Alison Q O'Neil, and Sotirios A Tsaftaris. Diffusion models for causal discovery via topological ordering. *arXiv preprint arXiv:2210.06201*, 2022a.