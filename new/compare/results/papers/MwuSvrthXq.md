000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Reinforcement Learning For Heteroge- Neous Dag Scheduling With Weighted Cross- Attention

Anonymous authors Paper under double-blind review

## Abstract

Efficient scheduling of directed acyclic graphs (DAGs) in heterogeneous environments is challenging due to diverse resource capacities and intricate dependencies. In practice, the need for adaptability across environments with varying resource pools, task types, and other settings, alongside rapid schedule generation, complicates these challenges. We propose WeCAN, an end-to-end reinforcement learning framework for heterogeneous DAG scheduling featuring task-resource compatibility. WeCAN rapidly generates schedules through single-pass network inference. Leveraging the weighted cross-attention layer, WeCAN utilizes all available environment information while preserving adaptability across diverse heterogeneous environments. Moreover, we analyze the optimality gap inherent in list-scheduling-based methods, revealing their inability to guarantee optimal solutions and their reduced performance in certain cases. Under the single-pass setting, we develop a method to enable skip actions, addressing this gap without sacrificing computational efficiency. Our approach delivers robust performance and adaptability, outperforming state-of-the-art methods across diverse datasets.

## 1 Introduction

Task scheduling problems are critical for optimizing computational performance in domains such as data centers (Mao et al., 2019), distributed systems, and large-scale cloud platforms (Lin et al., 2024). These problems are often modeled using directed acyclic graphs (DAGs), where nodes represent tasks and edges specify dependencies. The objective is typically to minimize the makespan (total completion time) subject to task dependencies and resource constraints. It is well known that DAG scheduling is an NP-hard problem even in homogeneous settings (Hartmanis, 1982). In heterogeneous environments, tasks need to be assigned to suitable pools (computational resources) possessing differing capacities and specific task-pool compatibility coefficients. This heterogeneity adds significant complexity to scheduling. Traditional approaches often employ heuristics, such as list scheduling (Graham, 1969) which assigns tasks to pools iteratively based on priority scores. These scores are often calculated using computationally inexpensive metrics, such as the critical-path length or the number of remaining operations (Haupt, 1989). Variations include Tetris (Grandl et al., 2014) using dynamic scores that change during scheduling, or HEFT (Topcuoglu et al., 2002) based on inserting tasks into an existing timeline. However, the design of such heuristics often relies heavily on human expertise and struggles to fully utilize all available problem information effectively. The application of machine learning (ML) to combinatorial optimization (CO) problems, originating with Hopfield & Tank (Hopfield & Tank, 1985), has gained significant attention in recent years. As most CO problems establish a one-to-one correspondence between solutions and action sequences, most ML approaches learn policies to construct solutions sequentially (Kool et al., 2019; Kwon et al.,
2020; Liu et al., 2024; Zhou et al., 2024). Alternative approaches include learning policies to refine existing solutions (Wu et al., 2021; Ma et al., 2021; 2024) or simplify problem instances (Li et al., 2021; Hou et al., 2023; Ye et al., 2024). Researchers have also proposed reinforcement learning (RL) based methods for scheduling problems (Mao et al., 2016; 2019; Park et al., 2021; Paliwal et al., 2020; Sun et al., 2021; Gagrani et al., 2022; Sun et al., 2024; Li et al., 2024).

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

## 2.1 Heterogeneous Scheduling Problem

Neural DAG Schedulers. For DAG scheduling problems, Zhang et al. (2020) develop an approach to generate solutions by using a graph neural network (GNN) to encode state and selecting tasks sequentially. Zhou et al. (2020) develop a method that learns a policy to iteratively refine an existing solution. Wang et al. (2021) introduce a bi-level optimization approach applying RL to add auxiliary edges and proposing heuristics on the modified graph. Typically, these methods require multi-round neural network processing for generating a solution, limiting speed for time-sensitive applications. Jeon et al. (2023) introduce an approach using the Gumbel-TopK trick to generate priorities which are subsequently fed to the list scheduling algorithm for a feasible schedule. While their single-pass network inference enables rapid solution generation, their architecture does not consider compatibility coefficients or pool allocation, remaining challenges in highly heterogeneous settings. Furthermore, the reliance on generation maps like list scheduling in many existing approaches introduces an inherent optimality gap, limiting their ability to consistently find optimal solutions. The skip action in Mao et al. (2016) mitigates this gap, but their design relies on the time-consuming multi-round network processing and is difficult to adapt to the single-pass setting.

Neural DAG Schedulers for Heterogeneous Environments. Researchers have also proposed reinforcement learning (RL) based methods for scheduling in heterogeneous environments (Wu et al.,
2018; Ni et al., 2020; Grinsztajn et al., 2021; Zhou et al., 2022; Zhadan et al., 2023; Wang et al.,
2025). For example, Zhou et al. (2022) propose an approach that selects tasks sequentially by network and assigns them to pools using heuristics. Their method represents compatibility coefficients by averaging them across pools, potentially losing fine-grained information. Similar strategies for embedding compatibility coefficients are proposed in Zhadan et al. (2023); Wang et al. (2025). Other works (Wu et al., 2018; Grinsztajn et al., 2021; Jeon et al., 2023) handle compatibility coefficients using representations like one-hot embedding of task types or fixed-dimensional vectors. However, these representations often depend on a fixed number of task types or pools, limiting their adaptability and flexibility to varying environment structures. Therefore, it remains a significant challenge to fully embed detailed environment information, including complex compatibility constraints, while maintaining adaptability across diverse and dynamically sized heterogeneous configurations. To address these challenges, we propose an end-to-end reinforcement learning framework for heterogeneous DAG scheduling, particularly addressing problems featuring compatibility coefficients. Our main contributions are as follows. (1) We design an end-to-end reinforcement learning framework to rapidly solve heterogeneous DAG scheduling problems with diverse task-pool compatibilities. Our approach generates the solution with single-pass network inference, resulting in a processing time close to the heuristics; (2) We design a weighted cross-attention network (WeCAN), consisting of weighted cross-attention layers to capture environment information and a longest directed distance graph neural network to adapt the varying tasks dependence. The weighted cross-attention network fully utilizes available environment information and compatibility coefficients for embedding, while preserving adaptability and scalability across varying heterogeneous environment sizes, such as the number of pools and task types. This facilitates accurate evaluation of task features within the specific environment; (3) We perform an analysis of the solution spaces and the generation maps, and propose a criterion for checking their ability to achieve optimality. This analysis highlights the optimality gap inherent in widely used methods such as list scheduling, which can lead to a significant decrease in performance under some characteristic cases. We develop a method to enable the skip action in the single-pass setting, and demonstrate how this method theoretically closes this gap while retaining the computational efficiency, thereby underscoring the importance of skip. We also reveal which cases benefit the most from the skip action; (4) Empirical evaluations on the Computation Graphs and real-world TPC-H benchmarks validate our approach, demonstrating improved performance and significant gains in computational efficiency compared to state-of-the-art methods.

## 2 Preliminaries

The task scheduling problem seeks a schedule that minimizes the objective function among all feasible schedules subject to given constraints. We focus on heterogeneous scheduling problems characterized by compatibility coefficients. A problem instance P = (V, E, C, t, ρ, λ, Kacc) can be modeled by a DAG G = (V, E) together with a resource pool set C = {c(1)*, ..., c*(nc))}. The node set V and the edge set E represent tasks and their precedence relations, respectively. Each node (task) v ∈ V has a processing time t(v) and a resource demand vector ρ(v). Each pool in C has a resource capacity vector λ(c). The compatibility coefficients Kacc(*v, c*) ≥ 0 reflect the variations in task execution time across resource pools (Topcuoglu et al., 2002), with the actual execution time of task v on pool c proportional to the inverse of this coefficient. A schedule is represented by a function x : V → R × C, which maps each task v ∈ V to a pair (s(v), c(v)), representing its start time and assigned resource pool, respectively. In this paper, the objective function is the makespan f, defined as the latest task completion time. The problem can thus be formulated as:

$\min_{x=(s,c)}\quad f(x):=\max_{v\in V}[s(v)+t(v)/K_{acc}(v,c(v))],$  s.t. $s(v)+t(v)/K_{acc}(v,c(v))\leq s(w),\,\forall(v,w)\in E,$  $\sum_{v\in F(t,c)}\rho(v)\leq\lambda(c),\,\forall t\geq0,\,c\in C,$
108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

$$\begin{array}{l}{{K_{a c c}(v,c(v))>0,\,\forall v\in V,}}\\ {{s(v)\geq0,\,\forall v\in V.}}\end{array}$$

Here, F(*t, v*) is the set of tasks on c at time t. The last constraint ensures the start time nonnegative, while the first three types of constraints are interpreted as follows: (1) dependency constraints, requiring each task to start after all its predecessors complete; (2) resource constraints, ensuring at any time t, the total resource demand of tasks running on pool c does not exceed its capacity λ(c); and (3)
compatibility constraints, requiring that a task v can be assigned to a pool c only if Kacc(*v, c*) > 0. In the homogeneous setting where K ≡ 1 and |C| = 1, the third constraint is naturally satisfied.

In contrast, heterogeneous environments with compatibility constraints pose additional challenges, requiring tasks to run on compatible resource pools and schedulers to adapt to diverse resource characteristics. These scheduling problems can be formulated as mixed integer linear programming (MILP)
problems, as described in Appendix A. This formulation establishes a one-to-one correspondence between feasible schedules and MILP feasible solutions, enabling a unified framework for analyzing scheduling scenarios. Compared to the homogeneous case, these settings introduce additional constraints and significantly increase MILP size, posing challenges for effective scheduling.

## 2.2 Learn To Schedule With Mask And Map

Most neural scheduling methods construct feasible schedules by sequentially assigning tasks to pools, rather than directly generating them as function x : V → R × C. For a scheduling problem X, the scheduling process is modeled as a Markov decision process (MDP), with the state represented by st. At each decision step, the scheduler, using a neural network to implement pθ(πt|st, π<t), selects an action πt = (vt, ct) to execute task vt on pool ct with probability pθ(πt|st, π<t). The probability of a schedule is the product of action probabilities across all T(π) decision steps:

$$p_{\theta}(\pi)=\prod_{t=1}^{T(\pi)}p_{\theta}(\pi_{t}|s_{t},\pi_{<t}).$$

## 3 Learn To Schedule For Heterogeneous Environment

To effectively schedule tasks across diverse heterogeneous resource environments, we propose the weighted cross-attention network. The overall architecture of our framework is illustrated in Figure Generating a feasible schedule from the action sequence (orders) requires a generation map S. Given a problem distribution, the target of learning to schedule is to find a policy pθ that minimizes the **loss**
function L(θ) = EXEπ∼pθ(·|X)f(S(π)). The most widely used map in both heuristic and neural schedulers is the list scheduling map S*list*. In list scheduling, the current time tcur is initialized to 0, and the scheduler updates masks to prevent selections that violate the dependency or resource constraints. The scheduler repeatedly selects an available action (*v, c*) by action sequence or other rules, setting c(v) = c and s(v) = tcur, until all actions are masked. When all actions are masked, the scheduler advances tcur to the next task completion time, updates the masks, and continues to select actions. During this map, masks for resource constraints are constructed from the remaining resources, while masks for dependency constraints are maintained by a dependency table. In the MDP model, the masks set the probabilities of infeasible actions to zero, ensuring feasibility.

1, which consists of two parts: once network processing and a generation map for creating feasible schedules. The weighted cross-attention (WeCA) layer in the network processes environmental information, enabling adaptation to various pool features, while maintaining adaptability regarding the number of task types and resource pools. By introducing skip action with once network processing, the generation map forms a surjection capable of representing the optimal solution without multiround computation, resulting in improved performance in cases involving heavy tasks characterized by long duration and high resource demand. Below, we describe each of the components in detail.

## 3.1 Weighted Cross-Attention Based Graph Encoder

Weighted Cross-Attention: Each task node v is represented by attribute vector (t(v), ρ(v)) and resource pool c by capacity vector ρ(c). Initially, an encoder generates task embeddings fv and pool embeddings f c c using a multilayer perceptron (MLP). These initial embeddings capture only isolated characteristics of tasks and pools. Consequently, tasks with identical attribute vectors receive the same initial embedding, yet their suitability varies across pools due to differing compatibility coefficients Kacc(·, ·). To capture these contextual characteristics within the heterogeneous environment and ensure that the architecture remains adaptable with respect to the number of task types and pools, we apply the WeCA layer. Based on the Transformer architecture (Vaswani et al., 2017) and skipconnection (He et al., 2016), our WeCA layer executes message passing between tasks and pools:

$q_{v}=W^{Q}f_{v},\quad K^{c}=W^{K}[f^{c}_{c(1)},...,f^{c}_{c(n_{c})}],\quad V^{c}=W^{V}[f^{c}_{c(1)},...,f^{c}_{c(n_{c})}],$  $g_{v}=f_{v}+\frac{\text{softmax}(q^{T}_{v}K^{c})}{\sqrt{d}}\text{diag}\{K_{acc}(v,c(1)),...,K_{acc}(v,c(n_{c}))\}V^{c},$
162

![3_image_0.png](3_image_0.png) 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 where {c(1)*, ..., c*(nc)} is the pool set, Kacc is the compatibility coefficient and d is the dimension of qv. This design integrates the compatibility coefficient as the attention bias without imposing fixed dimensionality constraints on the network architecture. Crucially, task v cannot run on pool c if and only if Kacc(*v, c*) = 0. Thus, incompatible pool assignments are inherently masked in attention calculation. The WeCA layer evaluates potential pools for each task and aggregates their features, weighted by suitability. A larger Kacc(*v, c*) indicates that task v runs faster on pool c, indicating a better compatibility. Therefore, for each task, the WeCA layer effectively gathers and integrates information primarily from its most compatible pools. Unlike approaches relying on fixed-size embeddings, this mechanism maintains adaptability, allowing the framework to evaluate contextdependent resource capacity for each task more realistically within the heterogeneous environment. Notably, the acceleration coefficient is multiplied outside the softmax normalization function rather than the inside version of logarithmic form, which is described in Appendix G. This placement aids in embedding information about a task's overall compatibility. Consider a scenario of two pools with identical capacity and two tasks v1, v2 with the same attribute. Suppose v1 is compatible with only one pool, while v2 compatible with both pools. Despite the identical attribute, these tasks differ significantly in their environmental compatibility. If the inside placement were taken, the normalization effect could lead to the same embeddings for both tasks, failing to distinguish their different compatibility profiles. Conversely, the outside placement better reflects a task's overall compatibility within the environment, resulting in a more accurate and distinguishable embedding. Longest Directed Distance based Graph Neural Network (LDDGNN): To embed the dependency, a graph neural network (GNN) is employed. Standard GNN architectures often struggle with longrange dependencies and directed acyclic graphs embedding. Inspired by Graphormer (Ying et al., 2021) and Topoformer (Gagrani et al., 2022), we design the network through attention masks and biases, but based on longest directed distance (LDD) de which is defined as the signed length of the longest directed path. Our LDDGNN (detailed in Appendix G) update the node embeddings through L attention layers, each comprising a multi-head attention (MHA) sub-layer and a node-wise MLP:

$$\begin{array}{l}{{q_{v}^{l,j}=W_{l,j}^{Q}h_{v}^{l-1},\mathbf{k}_{v}^{l,j}=W_{l,j}^{K}\mathbf{h}_{v}^{l-1},\mathbf{v}_{v}^{l,j}=W_{l,j}^{V}\mathbf{h}_{v}^{l-1},}}\\ {{\hat{\mathbf{h}}_{v}^{l}=\mathbf{h}_{v}^{l-1}+\mathrm{concat}_{j}(\sum_{w\in V}[(\mathbf{q}_{v}^{l,j})^{T}\mathbf{k}_{w}^{l,j}\cdot b_{d e}(v,w)\cdot M_{v,w}^{j}]\mathbf{v}_{w}^{l,j}),}}\end{array}$$

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

## 3.3 Training

$$\hbar_{v}^{l}=\hat{\mathbf{h}}_{v}^{l}+\mathrm{MLP}_{(l)}(\hat{\mathbf{h}}_{v}^{l}).$$

Here, Mj v,w is the attention mask of head j for the pair (*v, w*) which is based on LDD, and bde(v,w)
is a learnable bias embedding based on the LDD de(*v, w*). This LDD-aware attention mechanism helps capture both directed and undirected dependency structure within the task graph.

## 3.2 Decoder

The schedule is constructed by the action sequence. The decoder should output probabilities (or scores) for all actions. For improving scalability, we employ a non-auto-regressive decoder (comparison with auto-regressive one in Appendix B), where the action probability pθ(πt|st, π<t) depends only on the initial state s1 and reduces to pθ(πt|s1). Similarly to the encoder, the decoder utilizes WeCA layers (details in Appendix G) to update the final task embeddings hv and pool embedding h
c
c.
Finally, the decoder calculates the score of each action (*v, c*) by weighted inner product:
$$\hat{\mathbf{q}}_{v}=\hat{\mathbf{W}}^{Q}\mathbf{h}_{v},\quad\hat{\mathbf{k}}_{c}=\hat{\mathbf{W}}^{K}\mathbf{h}_{c}^{c},$$
u(v,c) = qˆ
v kˆc + log(Kacc(*v, c*)).
At each step, the generation map identifies invalid actions and applies masks to set their score u(v,c)
to −∞, then calculates the probability for each action π = (*v, c*) by a normalization:
. (1)
$$p_{\theta}(\pi)={\frac{\exp u_{\pi}}{\sum_{\pi}\exp u_{\pi}}}.$$
$\hat{q}^T\hat{k}_{\mu}+\hat{\mu}$
$x(\,K,\quad(u)\,)$

Moreover, to mitigate potential optimality gap, we should introduce the skip action, defined as advancing time tcur to the next task completion time. However, a fixed skip score will lead to endless idling, while generating a dynamic score by network in each step eliminates the single-pass efficiency.

For addressing this problem, we use the network to produce skip coefficients ua ≥ 0, ub ≥ 0 and uc which are derived from an MLP with the average of task embeddings hv and the average of pool embeddings h c cas input, and then calculate skip score as uπ*skip* = ua(1 −
k 2n
)
ub + uc. Here, k is the number of actions taken and n is the number of tasks. This approach fixes the optimality gap and prevents the skip action from overly prioritized, while remaining the single-pass efficiency.

Here we show the full process in Algorithm 1. The following theorem guarantees that the scheduler produces a feasible schedule in finite steps. Moreover, it can produce schedules in any feasible order and then includes the optimal solution. We provide the details of the proof in Appendix A. Theorem 1. i) Algorithm 1 generates a feasible solution within 2n steps where n is the number of nodes. ii) Algorithm 1 assigns positive probabilities to at least one optimal solution and all feasible orders. iii) Without the skip action, statement ii) does not hold for some problem X. iv) For each problem X*, there exist scores* {u(v,c)}v∈V,c∈C and ua, ub, uc *enabling an optimal solution by* greedily selecting the action with the highest pθ(π) *in Algorithm* 1.

Given scheduling problems X from a distribution, the goal is to minimize the expectation of makespan ˆf(π) = f(S(π)) with our policy p(π|X). We use REINFORCE (Williams, 1992) to update the parameters with learning rate α:

$$\nabla_{\theta}L(\theta)=\nabla_{\theta}\mathbb{E}_{X}\mathbb{E}_{p_{\theta}(\pi|X)}\left[(\hat{f}(\pi)-b(X))\nabla_{\theta}\log p_{\theta}(\pi|X)\right],$$ $$\theta\leftarrow\theta=\alpha\nabla_{\theta}L(\theta).$$
θ ← θ − α∇θL(θ).
$$(1)$$

Algorithm 1 Solution generation through the neural scheduler Input: DAG scheduling problem (G, V, C, ρ, t, λ, Kacc) with n tasks and nc pools. Initialize the dependency table and current time tcur = 0.

Perform WeCAN to get the scores uπ for each action π = (*v, c*) and coefficients ua, ub, uc. while unfinished node remains do Mask action (*v, c*) for all tasks v which have been started or have unfinished dependency. Mask action (*v, c*) by the resource requirement and compatible requirement.

Calculate skip score uπ*skip* = ua(1 −
k 2n
)
ub + uc, where k is the number of steps taken.

Mask the skip action if no running tasks on all pools. Calculate the probability by (1) and select (sample) an action π by the probability. if the selected action π = (*v, c*) is not the skip action **then**
Start the task v on pool c. Setting s(v) = tcur and c(v) = c.

else Find the next task completion time tnext > tcur and set tcur = t*next*.

end if Update the dependency table and current resource.

end while Here, b(X) is a baseline for reducing the variance and is taken as average rewards.

## 4 Schedule With Skip To Fix Gaps

We design a method to introduce the skip action, which is defined as advancing time tcur to the next task completion time, in our single-pass network framework. In this section, we demonstrate and analyze how the skip action enhances scheduling performance in characteristic cases by closing the optimality gap. We also provide a criterion for determining whether a generation map can generate optimal schedules and investigating which cases skip benefits most.

## 4.1 Reduced Space

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 4.2 Finding Optimum With Surjection

The scheduling problem can be formulated as an MILP problem. In Section 2, the concept of feasible schedule is introduced. For brevity, we denote the space of all such schedules as the original space A. In the MDP formulation, action sequences do not directly generate a feasible schedule; instead, they correspond to schedule orders in a discrete space B and the scores generated by the neural network represent point (in greedy way) or distributions (in sampling way) on B, as detailed in Appendix A. We refer to the discrete space B as the reduced space, whose point can be characterized by all task orders and pool allocations. Naturally, there exists a map T : A → B that associates each feasible schedule with its schedule order in B. A feasible schedule order is the legal order lying in the feasible reduced space Bf = T(A) ⊂ B. Obtaining a feasible schedule from a schedule order in Bf requires a generation map to A. Most heuristic and neural schedulers employ the list scheduling S*list* : B → A described in Section 2, which is effective for generating sub-optimal schedules. However, we prove in Appendix A that, in certain cases, list scheduling cannot yield an optimal schedule. We further find this theoretical gap leads to a significant performance decrease in some cases, as it prioritizes tasks meeting current resource availability. This prioritization creates a preference for delaying resource-intensive tasks and significantly impacts cases with heavy tasks featured by extreme resource demands and running times. Our experimental results in Appendix C further show that as the rate of heavy task increases, the gap also increases.

Our theoretical analysis and illustrative examples show that the inherent optimality gap of S*list* arises because T S*list* is neither the identity nor surjective. As a result, S*list* maps multiple points in B to the same point in A, shrinking its image and excluding the optimal solution. To construct a map S : Bf → A whose image includes the optimal solution, a projection between A and Bf associating each subspace of A with a point in Bf helps. Such maps must satisfy the following requirements.

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

## 5 Numerical Experiments 5.1 Dataset And Environment Settings

Datasets. i) *TPC-H dataset:* a dataset that comprises real-world DAG tasks derived from industrial queries. We use the version sorted by Wang et al. (2021), and add additional random memory constraints and task types (each with a group of compatibility coefficients). The problems in TPC- H-30, TPC-H-50, and TPC-H-100 contain 275, 459, and 918 tasks on average. ii) Computation Graphs dataset: a synthetic dataset generated using approaches from Jeon et al. (2023), comprising computation graphs for neural networks arising in ML compilers, including layer graphs, Erdos- ˝ Rényi graphs, and stochastic block model graphs. Each problem contains 500 tasks. Problems in both datasets are scheduled on three heterogeneous resource pools. The heterogeneous significantly multiple the problem size. Details of the two datasets and their problem size are shown in Appendix D. Baselines. We compare our method with the following baselines: list scheduling algorithms, including critical path (CP), shortest first task (SFT), and most operations remaining (MOPNR); Tetris (Grandl et al., 2014), a dynamic list scheduling heuristic for multi-resource pool scheduling; HEFT (Topcuoglu et al., 2002), a non-list heterogeneous scheduling algorithm; **Two RL baselines**: PPO-BiHyb (Wang et al., 2021), a bi-level neural scheduler with beam search; and One-Shot (Jeon et al., 2023), a one-shot neural scheduler generating schedules sequentially based on list scheduling. For the 4 list scheduling algorithms, we apply three pool-selection rules and select the one with the best makespan.

We evaluate our method using two modes: greedy, which selects actions with the highest probability pθ and sampling (S(n)), which generates n samples based on pθ within our accelerated environment and selects the schedule with the minimum makespan. For each experiment, we report the makespan, running time or relative improvement over the best heuristic baseline, with further experimental details provided in Appendices D, E, and H.

## 5.2 Experimental Results

On the TPC-H dataset, WeCAN demonstrates up to 18.1% makespan improvement over the best heuristic and 7.7% over the best neural baseline, with superior performance in instances with Assumption 1. (1) T S = I. This ensures that T and S provide an embedding of feasible reduced space Bf into A, while ST serves as a projection.

(2) f(v) ≥ f(ST(v)), ∀v. The map ST(v) minimizes the objective function f *within subspace* T
−1(T(v)).

These properties ensure that S embeds the feasible reduced space as a projection space for A, and the following theorem shows that such a map S can be used to find the optimal solution.

Theorem 2. Let S : B → A be a generation map satisfying Assumption 1. For any optimal solution x, there exists an optimal solution y ∈ Image(S) and T(y) = T(x). We provide the proof and construct a map Sn satisfying Assumption 1 in Appendix A. The map Sn extends list scheduling by relaxing certain constraints to allow waiting. However, although Sn includes the optimal solution, it also produces many inferior solutions scattered across the reduced space. This is because the mapping Sn allows arbitrary idle time, resulting in a large variance of makespan, hindering the training of scheduling policies. Therefore, instead of using Sn directly, we enlarge Bf to include skip actions, lift S*list* to a map S on the enlarged space, and modify T accordingly; the resulting (Bf *, T, S*) meet Assumption 1. Moreover, our design clusters most poor solutions in the high-ua, high-uc region, because excessive skips typically arise from large values of ua and uc, rather than scattering them across the space; this concentration makes such regions easier to handle during training and reduces variance. Theorem 1 demonstrates that our design in the single-pass setting ensures that T S is a surjection, enabling the generation of the optimal schedule while retaining high inference speed. Our experiments results in Appendix C further validate the effectiveness of this design, revealing that the skip benefits more when the percentage of heavy tasks increases. Therefore, our design enhances the ability to find the optimal schedules, leading to enhanced performance in heavy task cases without sacrificing computational efficiency or significantly increasing the variance in training outcomes by clustering poor solutions.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

| Table 1: Experimental results on TPC-H datasets with standard deviation among random seed. TPC-H-30, 3 pools TPC-H-50, 3 pools TPC-H-100, 3 pools MakeSpan Time MakeSpan Time MakeSpan Time SFT 27404 0.23 49172 0.78 84986 3.08 MOPNR 25052 0.30 43545 0.99 77362 3.34 CP 23869 0.29 41597 0.90 74364 3.35 HEFT 23177 0.18 39315 0.54 70137 1.86 Tetris 23170 0.21 38654 0.62 71296 2.13 PPO-BiHyb 21941 20.48 36333 55.74 67695 179.19 One-Shot-S(256) 20399 ± 181 2.26 35561 ± 108 4.16 66173 ± 180 9.85 WeCAN-Greedy 19578 0.15 33428 0.50 62587 1.72 WeCAN-S(64) 19053 ± 28 1.54 32912 ± 40 2.86 61662 ± 118 5.26 WeCAN-S(256) 18964 ± 10 2.43 32814 ± 47 4.39 61373 ± 28 10.43   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Table 2: Experimental results on Computation Graphs datasets with 500 tasks.

Erdos-Rényi Layer Graphs Stochastic Block ˝

MakeSpan Time MakeSpan Time MakeSpan Time

SFT 13317 0.81 16158 0.34 14408 0.53 MOPNR 12771 1.07 14714 0.38 13148 0.68 Tetris 13084 0.52 14271 0.44 13666 0.64 CP 12457 1.08 14797 0.40 13388 0.74 HEFT 11098 0.55 12428 0.75 11260 0.57 PPO-BiHyb 10795 65.51 11883 73.7 10885 73.7

One-Shot-S(256) 11071 ± 76 4.45 12277 ± 49 3.83 11377 ± 40 4.00

WeCAN-Greedy 10270 0.57 11173 0.26 10539 0.41 WeCAN-S(64) 10115 ± 10 3.21 10862 ± 29 3.07 10074 ± 18 3.06 WeCAN-S(256) **10083** ± 13 4.94 **10752** ± 27 4.30 **10019** ± 12 4.58

300∼500 nodes and robust results for 1,000 nodes (see Table 1). Additionally, our results show that WeCAN excels at learning robust scheduling policies in complex heterogeneous environments, as evidenced by its performance across diverse TPC-H instances. Owing to single-pass neural processing, WeCAN-greedy achieves lower running time than PPO-BiHyb and comparable running time to One- Shot-greedy and heuristic baselines, while delivering superior makespan. This similarity in running time arises because, in heterogeneous environments, the generation map's runtime dominates for both WeCAN and One-Shot, approaching the minimum time required to generate a schedule. Furthermore, we conduct experiments in varying resource environments to evaluate generalization from a fixed training environment. Figure 2 shows that our WeCAN shows robust performance under varying environment fluctuations including pool number, pool type (feature), task number, and task type. This result validates our WeCAN effectively utilizes the environment feature while remaining robust and scalable for environment sizes. We also provide results on large-scale problems and more environment fluctuations in Appendix F. In the Computation Graphs dataset, WeCAN demonstrates up to 13.4% makespan improvement over the best heuristic and 9.5% percent over the best neural baseline, with superior performance across different types of graphs (see Table 2). Given the prevalence of heterogeneous resource environments in ML compilers, our results demonstrate WeCAN's applicability for efficient scheduling in neural network compilation.

## 5.3 Ablation Study

We conduct ablation experiments to evaluate the contributions of WeCAN's weighted cross-attention layers and longest directed distance graph neural network components. For WeCA layers, we test the inside version, WeCA layers only in decoder, its inside version, and WeCA layers skipped except in

![8_image_0.png](8_image_0.png)

Figure 2: Evaluations of models on TPC-H-30 with different environment fluctuations under fixed training conditions. The percent of improvement over best heuristics are labeled.

TPC-H-30 TPC-H-50

MakeSpan Improvement MakeSpan Improvement

Tetris 23170 0.0% 38654 0.0%

WeCA + LDDGNN 19908 ± 48 14.0 ± 0.2% 34260 ± 52 11.4 ± **0.1 %** WeCA-inside + LDDGNN 20729 ± 55 10.5 ± 0.2 % 34980 ± 30 9.5 ± 0.1% WeCA-decoder+ LDDGNN 20234 ± 41 12.7 ± 0.2 % 34815 ± 156 9.9 ± 0.4%

WeCA-decoder-inside+ LDDGNN 21981 ± 195 5.1 ± 0.8 % 36984 ± 72 4.3 ± 0.2% WeCA-final-only + LDDGNN 23066 ± 97 0.5 ± 0.3 % 40308 ± 358 -4.2 ± 0.9% WeCA + GAT(forward) 20747 ± 21 10.5 ± 0.1 % 35224 ± 14 8.9 ± 0.1% WeCA + GAT(bi-direction) 20873 ± 7 9.9 ± 0.0 % 35177 ± 20 9.0 ± 0.1%

![8_image_1.png](8_image_1.png)

the final decoder layer. For LDDGNN, we test a standard GAT (GAT forward) and a bidirectional GAT (GAT bidrection). All network variants share the same layer count and hidden dimensions, with fewer WeCA layers offset by additional LDDGNN layers. We train and test each model on TPC-H-30 and TPC-H-50. For each of 10 test problems, we generate 256 samples, computing mean makespan and relative improvement to the best heuristic (Tetris). Table 3 shows that replacing either component increases makespan and decreases relative improvement. The modified WeCA layers yields higher makespan, with skipping WeCA layers causing significantly worse performance, highlighting WeCA layers' importance in capturing task-resource pool compatibility. Both GAT variants yield higher makespan than LDDGNN, demonstrating the superiority of LDDGNN. We conduct experiments to evaluate the skip-action mechanism on TPC-H with "heavy tasks" characterized by high resource demand and long processing time. We modify the TPC-H-30 and TPC-H-50 datasets by randomly replacing 1% tasks with "heavy tasks". Figure 3 shows that HEFT (not list scheduling) achieves smaller makespan than the best list scheduling approach (CP). WeCAN with the skip action achieves lower makespan than its non-skipping variant and all other approaches. This result validates the optimality gap of list scheduling. It also reveals that the skip-action mechanism mitigates the optimality gap and increases the performance in cases with "heavy tasks".

## 6 Conclusion

This paper presents WeCAN, an end-to-end reinforcement learning framework for heterogeneous DAG scheduling with task-pool compatibility. Weighted cross-attention layers enable WeCAN to fully utilize environment information while adapting to diverse problem sizes. Introducing skip-action in the single-pass setting closes optimality gap of list scheduling and improves performance in heavy task cases. Our theoretical analysis further highlights its importance, especially when heavy-task proportions are large. Evaluations on TPC-H and Computation Graphs datasets demonstrate WeCAN's effectiveness for heterogeneous DAG scheduling with task-resource compatibility. Extending our WeCAN to address more complicated settings will be an interesting future research direction.

Table 3: Ablation study results on TPC-H datasets with different architectures.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

## References

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Mukul Gagrani, Corrado Rainone, Yang Yang, Harris Teague, Wonseok Jeon, Roberto Bondesan, Herke van Hoof, Christopher Lott, Weiliang Zeng, and Piero Zappi. Neural topological ordering for computation graphs. In *Advances in Neural Information Processing Systems*, volume 35, pp.

17327–17339. Curran Associates, Inc., 2022.

R. L. Graham. Bounds on multiprocessing timing anomalies. *SIAM Journal on Applied Mathematics*,
17(2):416–429, 1969. doi: 10.1137/0117039.

Robert Grandl, Ganesh Ananthanarayanan, Srikanth Kandula, Sriram Rao, and Aditya Akella.

Multi-resource packing for cluster schedulers. In *Proceedings of the 2014 ACM Conference on* SIGCOMM, SIGCOMM '14, pp. 455–466, New York, NY, USA, 2014. Association for Computing Machinery. ISBN 9781450328364. doi: 10.1145/2619239.2626334.

Nathan Grinsztajn, Olivier Beaumont, Emmanuel Jeannot, and Philippe Preux. READYS:
A reinforcement learning based strategy for heterogeneous dynamic scheduling. In 2021 IEEE International Conference on Cluster Computing (CLUSTER), pp. 70–81, 2021. doi:
10.1109/Cluster48925.2021.00031.

Juris Hartmanis. Computers and intractability: A guide to the theory of NP-Completeness (Michael R. Garey and David S. Johnson). *SIAM Review*, 24(1):90–91, 1982. doi: 10.1137/1024022.

R. Haupt. A survey of priority rule-based scheduling. *Operations-Research-Spektrum*, 11(1):3–16, Mar 1989. ISSN 1436-6304. doi: 10.1007/BF01721162.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp.

770–778, 2016. doi: 10.1109/CVPR.2016.90.

J. J. Hopfield and D. W. Tank. "Neural" computation of decisions in optimization problems. Biol.

Cybern., 52(3):141–152, July 1985. ISSN 0340-1200.

Qingchun Hou, Jingwei Yang, Yiqiang Su, Xiaoqing Wang, and Yuming Deng. Generalize learned heuristics to solve large-scale vehicle routing problems in real-time. In The Eleventh International Conference on Learning Representations, 2023.

Wonseok Jeon, Mukul Gagrani, Burak Bartan, Weiliang Will Zeng, Harris Teague, Piero Zappi, and Christopher Lott. Neural DAG scheduling via one-shot priority sampling. In The Eleventh International Conference on Learning Representations, 2023.

Wouter Kool, Herke van Hoof, and Max Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019.

Yeong-Dae Kwon, Jinho Choo, Byoungjip Kim, Iljoo Yoon, Youngjune Gwon, and Seungjai Min.

POMO: Policy optimization with multiple optima for reinforcement learning. *Advances in Neural* Information Processing Systems, 33:21188–21198, 2020.

Sirui Li, Zhongxia Yan, and Cathy Wu. Learning to delegate for large-scale vehicle routing. Advances in Neural Information Processing Systems, 34:26198–26211, 2021.

Tiangang Li, Shi Ying, Yishi Zhao, and Jianga Shang. Batch jobs load balancing scheduling in cloud computing using distributional reinforcement learning. IEEE Transactions on Parallel and Distributed Systems, 35(1):169–185, 2024. doi: 10.1109/TPDS.2023.3334519.

Liduo Lin, Li Pan, and Shijun Liu. SpotDAG: An RL-based algorithm for DAG workflow scheduling in heterogeneous cloud environments. *IEEE Transactions on Services Computing*, 17(5):2904– 2917, 2024. doi: 10.1109/TSC.2024.3422828.

Fei Liu, Xi Lin, Zhenkun Wang, Qingfu Zhang, Tong Xialiang, and Mingxuan Yuan. Multi-task learning for routing problem with cross-problem zero-shot generalization. In *Proceedings of the* 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1898–1908, 2024.

Yining Ma, Jingwen Li, Zhiguang Cao, Wen Song, Le Zhang, Zhenghua Chen, and Jing Tang.

Learning to iteratively solve routing problems with dual-aspect collaborative transformer. Advances in Neural Information Processing Systems, 34:11096–11107, 2021.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Yining Ma, Zhiguang Cao, and Yeow Meng Chee. Learning to search feasible and infeasible regions of routing problems with flexible neural k-opt. *Advances in Neural Information Processing Systems*, 36, 2024.

Hongzi Mao, Mohammad Alizadeh, Ishai Menache, and Srikanth Kandula. Resource management with deep reinforcement learning. In Proceedings of the 15th ACM Workshop on Hot Topics in Networks, HotNets '16, pp. 50–56, New York, NY, USA, 2016. Association for Computing Machinery. ISBN 9781450346610. doi: 10.1145/3005745.3005750.

Hongzi Mao, Malte Schwarzkopf, Shaileshh Bojja Venkatakrishnan, Zili Meng, and Mohammad Alizadeh. Learning scheduling algorithms for data processing clusters. In *Proceedings of the ACM*
Special Interest Group on Data Communication, SIGCOMM '19, pp. 270–288, New York, NY,
USA, 2019. Association for Computing Machinery. ISBN 9781450359566.

Xiang Ni, Jing Li, Mo Yu, Wang Zhou, and Kun-Lung Wu. Generalizable resource allocation in stream processing via deep reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 857–864, Apr. 2020. doi: 10.1609/aaai.v34i01.5431.

Aditya Paliwal, Felix Gimeno, Vinod Nair, Yujia Li, Miles Lubin, Pushmeet Kohli, and Oriol Vinyals. Reinforced genetic algorithm learning for optimizing computation graphs. In *International* Conference on Learning Representations, 2020.

Junyoung Park, Jaehyeong Chun, Sang Hun Kim, Youngkook Kim, and Jinkyoo Park and. Learning to schedule job-shop problems: representation and policy learning using graph neural network and reinforcement learning. *International Journal of Production Research*, 59(11):3360–3377, 2021.

Binqi Sun, Mirco Theile, Ziyuan Qin, Daniele Bernardini, Debayan Roy, Andrea Bastoni, and Marco Caccamo. Edge generation scheduling for DAG tasks using deep reinforcement learning. IEEE
Transactions on Computers, 73(4):1034–1047, 2024. doi: 10.1109/TC.2024.3350243.

Penghao Sun, Zehua Guo, Junchao Wang, Junfei Li, Julong Lan, and Yuxiang Hu. DeepWeave:
accelerating job completion time with deep reinforcement learning-based coflow scheduling. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI'20, 2021. ISBN 9780999241165.

H. Topcuoglu, S. Hariri, and Min-You Wu. Performance-effective and low-complexity task scheduling for heterogeneous computing. *IEEE Transactions on Parallel and Distributed Systems*, 13(3):
260–274, 2002. doi: 10.1109/71.993206.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

Runzhong Wang, Zhigang Hua, Gan Liu, Jiayi Zhang, Junchi Yan, Feng Qi, Shuang Yang, Jun Zhou, and Xiaokang Yang. A bi-level framework for learning to solve combinatorial optimization on graphs. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), *Advances in Neural Information Processing Systems*, volume 34, pp. 21453–21466. Curran Associates, Inc., 2021.

Zhi Wang, Wenhan Zhan, Hancong Duan, Geyong Min, and Hualong Huang. Deep reinforcement learning-based continuous workflows scheduling in heterogeneous environments. IEEE Internet of Things Journal, pp. 1–1, 2025. doi: 10.1109/JIOT.2024.3524506.

Ronald J. Williams. Simple statistical gradient-Following algorithms for connectionist reinforcement learning. *Mach. Learn.*, 8(3–4):229–256, May 1992. ISSN 0885-6125. doi: 10.1007/BF00992696.

Qing Wu, Zhiwei Wu, Yuehui Zhuang, and Yuxia Cheng. Adaptive DAG tasks scheduling with deep reinforcement learning. In Jaideep Vaidya and Jin Li (eds.), Algorithms and Architectures for Parallel Processing, pp. 477–490, Cham, 2018. Springer International Publishing. ISBN 978-3-030-05054-2.

Yaoxin Wu, Wen Song, Zhiguang Cao, Jie Zhang, and Andrew Lim. Learning improvement heuristics for solving routing problems. *IEEE transactions on neural networks and learning systems*, 33(9): 5057–5069, 2021.

Haoran Ye, Jiarui Wang, Helan Liang, Zhiguang Cao, Yong Li, and Fanzhang Li. GLOP: Learning global partition and local construction for solving large-scale routing problems in real-time. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 20284–20292, 2024.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu. Do transformers really perform badly for graph representation? In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 28877–28888. Curran Associates, Inc., 2021.

Anastasia Zhadan, Alexander Allahverdyan, Ivan Kondratov, Vikenty Mikheev, Ovanes Petrosian, Aleksei Romanovskii, and Vitaliy Kharin. Multi-agent reinforcement learning-based adaptive heterogeneous DAG scheduling. *ACM Trans. Intell. Syst. Technol.*, 14(5), October 2023. ISSN
2157-6904. doi: 10.1145/3610300.

Yanqi Zhou, Sudip Roy, Amirali Abdolrashidi, Daniel Wong, Peter Ma, Qiumin Xu, Hanxiao Liu, Phitchaya Phothilimtha, Shen Wang, Anna Goldie, Azalia Mirhoseini, and James Laudon. Transferable graph optimizers for ML compilers. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp.

13844–13855. Curran Associates, Inc., 2020.

Yunfan Zhou, Xijun Li, Jinhong Luo, Mingxuan Yuan, Jia Zeng, and Jianguo Yao. Learning to optimize DAG scheduling in heterogeneous environment. In *2022 23rd IEEE International* Conference on Mobile Data Management (MDM), pp. 137–146, 2022.

Cong Zhang, Wen Song, Zhiguang Cao, Jie Zhang, Puay Siew Tan, and Xu Chi. Learning to dispatch for job shop scheduling via deep reinforcement learning. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 1621–1632. Curran Associates, Inc., 2020.

Jianan Zhou, Zhiguang Cao, Yaoxin Wu, Wen Song, Yining Ma, Jie Zhang, and Chi Xu. MVMoE:
Multi-task vehicle routing solver with mixture-of-experts. In International Conference on Machine Learning, 2024.