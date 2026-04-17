000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Learning The Minimum Action Distance

Anonymous authors Paper under double-blind review

## Abstract

This paper presents a state representation framework for Markov decision processes (MDPs) that can be learned solely from state trajectories, requiring neither reward signals nor the actions executed by the agent. We propose learning the *minimum* action distance (MAD), defined as the minimum number of actions required to transition between states, as a fundamental metric that captures the underlying structure of an environment. MAD naturally enables critical downstream tasks such as goal-conditioned reinforcement learning and reward shaping by providing a dense, geometrically meaningful measure of progress. Our self-supervised learning approach constructs an embedding space where the distances between embedded state pairs correspond to their MAD, accommodating both symmetric and asymmetric approximations. We evaluate the framework on a comprehensive suite of environments with known MAD values, encompassing both deterministic and stochastic dynamics, as well as discrete and continuous state spaces, and environments with noisy observations. Empirical results demonstrate that the proposed approach not only efficiently learns accurate MAD representations across these diverse settings but also significantly outperforms existing state representation methods in terms of representation quality.

## 1 Introduction

In reinforcement learning (Sutton & Barto, 1998), an agent aims to learn useful behaviors through continuing interaction with its environment. Specifically, by observing the outcomes of its actions, a reinforcement learning agent learns over time how to select actions in order to maximize the expected cumulative reward it receives from its environment. An important need in applications of reinforcement learning is the ability to generalize, not only to previously unseen states, but also to variations of its environment that the agent has not previously interacted with. In many applications of reinforcement learning, it is useful to define a metric that measures the similarity of two states in the environment. Such a metric can be used, e.g., to define equivalence classes of states in order to accelerate learning, to decompose the problem into a hierarchy of smaller subproblems that are easier to solve, or to perform transfer learning in case the environment changes according to some parameters but retains part of the structure of the original environment. Such a metric can also be used as a heuristic in goal-conditioned reinforcement learning, in which the agent has to achieve different goals in the same environment. The Minimum Action Distance (MAD) has proved useful as a similarity metric, with impressive applications in various areas of reinforcement learning, including policy learning (Wang et al., 2023b; Park et al., 2023), reward shaping (Steccanella & Jonsson, 2022), and option discovery (Park et al., 2024a;b). While prior work has demonstrated the advantages of using MAD, how best to approximate it remains an open problem. Existing methods have not been systematically evaluated on their ability to approximate the MAD function itself, and many rely on symmetric approximations, even though the true MAD is inherently asymmetric.

We make three main contributions towards fast, accurate approximation of the MAD. First, we propose two novel algorithms for learning MAD using only state trajectories collected by an agent interacting with its environment. Unlike previous work, the proposed algorithms naturally support both symmetric and asymmetric distances, and incorporate both short- and long-term information about how distant two states are from one another. Secondly, we define a novel quasimetric distance function that is computationally efficient and that, in spite of its simplicity, outperforms more 1

![1_image_0.png](1_image_0.png)

![1_image_2.png](1_image_2.png)

![1_image_3.png](1_image_3.png)

054

![1_image_1.png](1_image_1.png) 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Figure 1: Schematic overview of MAD representation learning. From left to right: (1) the hidden environment graph, (2) trajectories collected by an unknown policy, (3) the embedding function ϕ : S → R
2and (4) the resulting MAD embedding space in R
2.

elaborate quasimetrics in the existing literature. Finally, we introduce a diverse suite of environments
- including those with discrete and continuous state spaces, stochastic and deterministic dynamics, and directed and undirected transitions - in which the ground-truth MAD is known, enabling a systematic and controlled evaluation of different MAD approximation methods. Figure 1 illustrates the steps of MAD representation learning: an agent collects state trajectories from an unknown environment, which are used to learn a state embedding that implicitly defines a distance function between states.

## 2 Related Work

In applications such as goal-conditioned reinforcement learning (Ghosh et al., 2020) and stochastic shortest-path problems (Tarbouriech et al., 2021), the temporal distance is measured as the expected number of steps required to reach one state from another state under some policy. In contrast, the MAD
is a lower bound on the number of steps based solely on the support of the transition function. This distinction makes the MAD efficient to compute and robust to changes in the transition probabilities as long as the support over next states remains the same, making it suitable for representation learning and transfer learning. Prior work has explored the connection between the MAD and optimal goal-conditioned value functions (Kaelbling, 1993). Park et al. (2023) highlight this connection and propose a hierarchical approach that improves distance estimates over long horizons, and Park et al. (2024a) embed states into a learned latent space where the distance between embedded states directly reflects an onpolicy measure of the temporal distance (Hartikainen et al., 2020). Park et al. (2024b) and Ma et al. (2022) extend this idea to the offline setting, learning embeddings from arbitrary experience such that Euclidean distances between state embeddings approximate the MAD. As an alternative to approximating the MAD using goal-conditioned value functions, Steccanella & Jonsson (2022) formulate learning a state embedding in which distances approximate the MAD as a constrained optimization problem, where bounds on the distance between embedded states are derived from state trajectory data. Although their formulations differ, these approaches ultimately seek to learn the same underlying quantity: the minimum number of actions required to move between two states. These existing approaches share a common limitation: they rely on symmetric distance metrics such as the Euclidean distance between state embeddings to approximate the MAD. As such, they cannot capture the asymmetry of the true MAD in environments with irreversible dynamics. In contrast, the approach we develop here supports the use of asymmetric distance metrics (or, *quasimetrics*), which can better capture the directional structure in many environments.

Some prior work has already explored the use of quasimetrics in reinforcement learning. Wang et al. (2023b) learn an asymmetric distance function that approximates the MAD by preserving local structure while maintaining global distances. Their method differs from the one we propose in two ways. First, their method does not leverage the existing distance along a trajectory as supervision for the learning process. Secondly, they use the Interval Quasimetric Embedding (IQE) (Wang & Isola, 2022) to learn the distance function. Dadashi et al. (2021) and Agarwal et al. (2021) learn embeddings and define a pseudometric between states as the Euclidean distance between their 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

## 3 Background 4 The Minimum Action Distance

Given an MDP M = ⟨S, A, R,P, D, γ⟩ and a state pair (s, s′) ∈ S2, the Minimum Action Distance, dMAD(*s, s*′), is defined as the minimum number of decision steps needed to transition from s to s
′. In deterministic MDPs, the MAD is always realizable using an appropriate policy; in stochastic MDPs, the MAD is a lower bound on the actual number of decision steps of any policy. Let R ⊆ S2 be a relation such that (*s, s*′) ∈ R if and only if there exists an action a ∈ A that satisfies P(s
′|*s, a*) > 0.

That is, R contains all state pairs (*s, s*′) such that s
′is reachable in one step from s. We can formulate embeddings. Unlike our work, they use loss functions inspired by bisimulation to learn both state and state-action embeddings. Successor features (Dayan, 1993; Barreto et al., 2017), and time-contrastive representations (Eysenbach et al., 2022) have also been used to define notions of temporal distance. Myers et al. (2024) introduces time-contrastive successor features, defining a distance metric based on the difference between discounted future occupancies of state features learned via time-contrastive learning. While their metric satisfies the triangle inequality and naturally handles both stochasticity and asymmetry, the resulting distances reflect expected discounted state visitations under a specific behavior policy and lack an intuitive interpretation. In contrast, approaches that approximate the MAD are naturally interpretable as a lower bound on the number of actions needed to transition between two states. Laplacian-based representation learning methods (Wu et al., 2019; Machado, 2019; Wang et al., 2021; 2023a) learn embeddings from the spectral structure of random walks over the transition graph, producing representations that reflect global connectivity in the state space. However, these methods are typically defined on a symmetrized transition operator or undirected Laplacian, and the induced geometry measures diffusion-based similarity rather than directed reachability. As a result, distances in these embeddings are fundamentally symmetric and do not correspond to the minimum number of actions required to move between two states, making them poorly suited to environments with irreversible or asymmetric dynamics.

In this section, we introduce the notation and concepts used throughout the paper. Given a finite set X ,
we use ∆(X ) = {p ∈ R
X |Pxpx = 1, px ≥ 0 (∀x)} to denote the probability simplex (i.e. the set of all probability distributions over X ). A rectified linear unit (ReLU) is a function relu : R
d → R
d≥0 defined on any vector x ∈ R
das relu(x) = [max(0, xi)]d i=1.

Markov Decision Processes (MDPs). An MDP (Bellman, 1957) is a tuple M = ⟨S, A, R,P, D, γ⟩,
where S is the state space, A is the action space, R : *S × A →* R is the reward function, P :
S ×A → ∆(S) is the transition kernel, D ∈ ∆(S) is the initial state distribution, and γ ∈ [0, 1] is the discount factor. At each time t, the learning agent observes a state st ∈ S, selects an action at ∈ A, receives a reward rt = R(st, at) and transitions to a new state st+1 ∼ P(st, at). The learning agent selects actions using a policy π : S → ∆(A), a mapping from states to probability distributions over actions. In our work, the state space S can be either discrete or continuous. Reinforcement learning (RL). RL (Sutton & Barto, 2018) is a family of algorithms whose purpose is to learn a policy π that maximizes some measure of expected future reward. In this paper, however, we consider the problem of representation learning, and hence we are not directly concerned with the problem of learning a policy. Concretely, we wish to learn a distance function between pairs of states that can later be used by an RL agent to learn more efficiently. In this setting, we assume that the learning agent uses a behavior policy πb to collect trajectories. Since we are interested in learning a distance function over state pairs, actions are relevant only for determining possible transitions between states, and rewards are not relevant at all. Hence for our purposes a trajectory τ = (s0, s1*, . . . , s*n) is simply a sequence of states.

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 the problem of computing dMAD as a constrained optimization problem:

$$d_{\mathrm{MAD}}=\arg\operatorname*{max}_{d}\sum_{(s,s^{\prime})\in S^{2}}d(s,s^{\prime}),$$
$$(1)$$
d(*s, s*′), (1)
s.t. $d(s,s)=0\quad\forall s\in\mathcal{S}$,  $$d(s,s^{\prime})\leq1\quad\forall(s,s^{\prime})\in R,$$ $$d(s,s^{\prime})\leq d(s,s^{\prime\prime})+d(s^{\prime\prime},s^{\prime})\quad\forall(s,s^{\prime},s^{\prime\prime})\in\mathcal{S}^{3}.$$

It is straightforward to show that dMAD is the unique solution to equation 1 (see Appendix A).

Concretely, dMAD satisfies the second constraint with equality, i.e. d(*s, s*′) = 1 for all (*s, s*′) ∈ R. If the state space S is finite, the constrained optimization problem is precisely the linear programming formulation of the all-pairs shortest path problem for the graph (S, R) with edge costs 1. This graph is itself a determinization of the MDP M (Yoon et al., 2007). In this case we can compute dMAD
exactly using the well-known Floyd-Warshall algorithm (Floyd, 1962; Warshall, 1962). If the state space S is continuous, R is still well-defined, and hence there still exists a solution which satisfies d(*s, s*′) = 1 for all (*s, s*′) ∈ R even though the states can no longer be enumerated.

An alternative to the MAD is computing the stochastic shortest path (SSP; Tarbouriech et al., 2021) between each pair of states. In deterministic MDPs, the MAD and SSP are equivalent. In stochastic MDPs, the SSP can provide more realistic distance estimates than the MAD when some transitions have very low probabilities. However, computing the all-pairs SSP requires solving a linear program over transition probabilities, which is computationally demanding. In contrast, the MAD can be computed efficiently and remains a useful approximation in many domains (e.g. in navigation problems and when using sticky actions). Moreover, unlike the SSP, the MAD depends only on the support of the transition kernel and is otherwise robust to changes in transition probabilities, which is particularly useful for transfer learning. Even when the state space S is finite, we may not have explicit knowledge of the relation R. In addition, the time complexity of the Floyd-Warshall algorithm is O(|S|3), and the number of states may be too large to run the algorithm in practice. If the state space S is continuous, then we cannot even explicitly form a graph (S, R). Hence we are interested in estimating dMAD in the setting for which we can access trajectories only through sampling. For this purpose, let us assume that the learning agent uses a behavior policy πb to collect a dataset of trajectories D = {τ1*, . . . , τ*k}. Define SD ⊆ S as the subset of states that appear on any trajectory in D. Given a trajectory τ = {s0*, ..., s*n} and any two states si and sj on the trajectory such that 0 ≤ *i < j* ≤ n, it is easy to see that j − i is an upper bound on dMAD(si, sj ), since sj is reachable in j − i steps from si on the trajectory τ . By an abuse of notation, we often write (si, sj ) ∈ τ to refer to a state pair on the trajectory τ with indices i and j such that *i < j*, and we write (si, sj ) ∼ τ in order to sample two such states from τ .

Steccanella & Jonsson (2022) learn a parameterized state embedding ϕθ : S → R
dand define a distance function dθ(*s, s*′) = d(ϕθ(s), ϕθ(s
′)), where d is any distance metric in Cartesian space.

The parameter vector θ of the state embedding is learned by minimizing the loss function

$$\mathcal{L}=\mathbb{E}_{\tau\sim\mathcal{D},(s_{i},s_{j})\sim\tau}\left[(d_{\theta}(s_{i},s_{j})-(j-i))^{2}+w_{c}\cdot\text{relu}(d_{\theta}(s_{i},s_{j})-(j-i))^{2}\right],\tag{2}$$

where wc > 0 is a regularization factor that multiplies a penalty term which substitutes the upper bound constraints dθ(si, sj ) ≤ j − i. If the distance metric d satisfies the triangle inequality (e.g. any norm d = *|| · ||*p) then the constraints dθ(s, s) = 0 and the triangle inequality automatically hold. Enforcing the constraint dθ(si, sj ) ≤ j − i for each state pair (si, sj ) on trajectories, rather than only consecutive pairs, helps learn better distance estimates, at the cost of a larger number of constraints.

## 5 Asymmetric Distance Metrics

A limitation of previous work is that the chosen distance metric d is symmetric, while the MAD dMAD
may not be symmetric. In this section, we review several asymmetric distance metrics. Concretely, a quasimetric is a function dq : R
d × R
d → R+ that satisfies the following three conditions:
- Q1 (Identity): dq(*x, x*) = 0.

- Q2 (Non-negativity): dq(x, y) ≥ 0. - Q3 (Triangle inequality): dq(*x, z*) ≤ dq(x, y) + dq(*y, z*).

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 A quasimetric does not require symmetry, i.e., dq(x, y) = dq(*y, x*) does not hold in general. We define a simple quasimetric dsimple using rectified linear units:

$$\leq d_{q}(x,y)+d_{q}(y,z).$$
$$d_{\rm simple}(x,y)=\alpha\max({\rm relu}(x-y))+(1-\alpha)\frac{1}{d}\sum_{i}^{d}{\rm relu}(x_{i}-y_{i}).\tag{3}$$

or, alternatively, using a maxmean reduction:

$$d_{\mathrm{IQE-mm}}(X,Y)=\alpha\,\operatorname*{max}_{1\leq i\leq k}L_{i}(X,Y)+(1-\alpha)\,\frac{1}{k}\sum_{i=1}^{k}L_{i}(X,Y),$$

where α ∈ [0, 1] balances the influence of the maximum and the average. This construction yields a quasimetric that inherently respects the triangle inequality while accounting for directional differences between the matrices X and Y .

Given any of the above quasimetrics dq (i.e., dsimple, dWN or dIQE), we can now define an asymmetric distance function dθ(*s, s*′) = dq(ϕθ(s), ϕθ(s
′)). In the case of dIQE, the state embedding ϕ : S → R
d produces an output that is reshaped into a k × m matrix structure to parameterize the intervals. The choice of quasimetric directly shapes the trade-offs in computational cost and optimization dynamics. In Appendix E, we present an ablation study examining how this choice affects our algorithms. This metric is a weighted average of the maximum and average positive difference between the vectors x and y along any dimension, where α ∈ [0, 1] is a weight. In Appendix B, we show that dsimple satisfies the triangle inequality and latent positive homogeneity (Wang & Isola, 2022). The Wide Norm quasimetric (Pitis et al., 2020), dWN, applies a learned transformation to an asymmetric representation of the difference between two states. The Wide Norm is defined as

$$d_{\mathrm{WN}}(x,y)=||W(\operatorname{relu}(x-y)::\operatorname{relu}(y-x))||_{2},$$

where "::" denotes concatenation and W ∈ R
k×2dis a learned weight matrix. This ensures that dWN(*x, y*) is non-negative and satisfies the triangle inequality, while concatenation is asymmetric.

The Interval Quasimetric Embedding (IQE) (Wang & Isola, 2022) leverages the Lebesgue measure of interval unions to capture asymmetric distances. IQE interprets the latent embeddings as matrices X, Y ∈ R
k×m (typically obtained by reshaping a flat output vector of dimension d = k · m). Let xij denote the element in row i and column j of matrix X. For each row i, we construct an interval by taking the union over the intervals defined by matrices X and Y :

$$I_{i}(X,Y)=\bigcup_{j=1}^{m}\left[x_{i j},\,\operatorname*{max}\{x_{i j},\,y_{i j}\}\right].$$

The length of this interval, denoted by Li(*X, Y* ), is computed as its Lebesgue measure. The IQE
distance is obtained by aggregating these row-wise lengths. For example, one may define

$$d_{\mathrm{IQE}}(X,Y)=\sum_{i=1}^{k}L_{i}(X,Y),$$

## 6 Learning Asymmetric Mad Estimates

Here, we propose two novel variants of the MAD learning approach. Each trains a state encoding ϕθ that maps states to an embedding space and uses a quasimetric dq to compute distances dθ(*s, s*′) =
dq(ϕθ(s), ϕθ(s
′)) between pairs of states (*s, s*′). Both variants support any quasimetric formulation such as dsimple, dWN and dIQE, and can incorporate additional features such as gradient clipping. A
full derivation of these learning objectives is provided in Appendix C.

## 6.1 Maddist: Direct Distance Learning

Then, the constraint loss is defined as:

$${\mathcal{L}}_{c}=\mathbb{E}_{(s_{i},s_{j})\sim{\mathcal{D}}_{\leq H_{c}}}\left[\left(\operatorname{relu}\left(d_{\theta}(s_{i},s_{j})-(j-i)\right)\right)^{2}\right].$$
$$\left(7\right)$$
-(relu (dθ(si, sj ) − (j − i)))2. (7)
The first algorithm, which we call *MadDist*, learns state distances using an approach similar to prior work (Steccanella & Jonsson, 2022), but differs in the use of a quasimetric distance function and a scale-invariant loss. Concretely, MadDist minimizes the following composite loss function:

$${\mathcal{L}}={\mathcal{L}}_{o}+w_{r}{\mathcal{L}}_{r}+w_{c}{\mathcal{L}}_{c}.$$
$$(4)$$
$$(5)$$
L = Lo + wrLr + wcLc. (4)
The main objective, Lo, is a scaled version of the square difference in equation 2:

$${\mathcal{L}}_{o}=\mathbb{E}_{\tau\sim{\mathcal{D}},(s_{i},s_{j})\sim\tau}\left[\left({\frac{d_{\theta}(s_{i},s_{j})}{j-i}}-1\right)^{2}\right].$$

Crucially, scaling makes the loss invariant to the magnitude of the estimation error, which typically increases as a function of j − i. In other words, states that are further apart on a trajectory do not necessarily dominate the loss simply because the magnitude of the estimation error is larger.

The second loss term, Lr, which is weighted by a factor wr > 0, is a contrastive loss that encourages separation between state pairs randomly sampled from all trajectories:
where dmax is a hyperparameter. Finally, the loss term Lc, which is weighted by a factor wc > 0, enforces the upper bound constraints. Specifically, let D≤Hc denote the set of state pairs sampled from trajectories in D such that the index difference satisfies 1 ≤ j − i ≤ Hc (where Hc is a hyperparameter), i.e.

$${\mathcal{D}}_{\leq H_{c}}=\left\{\left(s_{i},s_{j}\right)|\,\tau\in{\mathcal{D}},\;s_{i},s_{j}\in\tau,\;1\leq j-i\leq H_{c}\right\}.$$
$$\mathcal{L}_{o}^{\prime}=\mathbb{E}_{\tau\sim\mathcal{D},(s_{i},s_{j})\sim\tau}\left[\left(\frac{d_{\theta}(s_{i},s_{j})}{\min(j-i,1+d_{\theta^{\prime}}(s_{i+1},s_{j}))}-1\right)^{2}\right].\tag{8}$$

Hence if the current distance estimate dθ
′ (si+1, sj ) computed using the target embedding ϕθ
′ is smaller than j − (i + 1), the objective is to make dθ(si, sj ) equal to 1 + dθ
′ (si+1, sj ).

We also modify the second loss term L
′rto include bootstrapped distances:

$$(10)$$

## 6.2 Tdmaddist: Temporal Difference Learning

The second algorithm, which we call *TDMadDist*, incorporates temporal difference learning principles by maintaining a separate target embedding ϕθ
′ and learning via bootstrapped targets. Specifically, TDMadDist learns by minimizing the loss function L
′ = L
′
o + wrL
′
r + wcLc, where Lc is the loss term from equation 7 that enforces the upper bound constraints.

The main objective L
′o of TDMadDist is modified to include bootstrapped distances:

1+dθ′ (si+1,sr)−1
2.(9) Given a state si sampled from a trajectory of D and a random state sr ∈ SD,
the objective is to make dθ(si, sr) equal to 1 + dθ
′ (si+1, sr).
The target network parameters θ
′are updated in each time step via an exponential moving average
with hyperparameter β ∈ (0, 1):
$$\theta^{\prime}\leftarrow(1-\beta)\theta^{\prime}+\beta\theta.$$
′ + βθ. (10)
$\mathcal{L}_{r}^{\prime}=\mathbb{E}_{r\sim\mathcal{D},(s_{i},s_{j})\sim r,s_{r}\sim\mathcal{S}_{\mathcal{D}}}\left[(d_{\theta}(s_{i},s_{i+1})\right]$
$${\cal L}_{r}=\mathbb{E}_{(s,s^{\prime})\sim{\cal S}_{\cal D}}\left[\left(\mathrm{relu}\left(1-\frac{d_{\theta}(s,s^{\prime})}{d_{\mathrm{max}}}\right)\right)^{2}\right]\tag{6}$$
270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 7 Experiments

We evaluate our proposed MAD learning algorithms on a diverse set of environments with varying characteristics, including deterministic and stochastic dynamics, discrete and continuous state spaces, and environments with noisy observations. Our analysis is directed by the following questions:
- How accurately do our learned embeddings capture the true minimum action distances? - How does the performance of our method compare to existing quasimetric learning approaches? - How robust is our approach to environmental stochasticity and observation noise?

Evaluation Metrics. We evaluate the quality of our learned representations using three metrics:
324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

![6_image_0.png](6_image_0.png)

- **Spearman Correlation (**ρ): Measures the preservation of ranking relationships between state pairs. A high Spearman correlation indicates that if state siis farther from state sj than from state sk in the true environment, our learned metric also predicts this same ordering. Perfect preservation of distance rankings gives ρ = 1.

- **Pearson Correlation (**r): Measures the linear relationship between predicted and true distances.

A high Pearson correlation indicates that our learned distances scale proportionally with true distances (i.e. when true distances increase, our predictions increase linearly as well). Perfect linear correlation gives r = 1.

- **Ratio Coefficient of Variation (CV)**: Measures the consistency of our distance scaling across different state pairs. A low CV indicates that our predicted distances maintain a consistent ratio to true distances throughout the state space. For example, if we consistently predict distances that are approximately 1.5 times the true distance, CV will be low. High variation in this ratio across different state pairs results in high CV. More formally, given a set of ground truth distances d1, d2*, ..., d*n and their corresponding predicted distances ˆd1, ˆd2, ...,
ˆdn where di > 0, we compute the ratios ri = ˆdi/di. The Ratio CV is given by

$$C V={\frac{\sigma_{r}}{\mu_{r}}}={\frac{\sqrt{{\frac{1}{n}}\sum_{i=1}^{n}(r_{i}-\mu_{r})^{2}}}{{\frac{1}{n}}\sum_{i=1}^{n}r_{i}}},$$
$$(11)^{\frac{1}{2}}$$
, (11)
Baselines. We compare our methods against QRL (Wang et al., 2023b), a recent quasimetric reinforcement learning approach that learns state representations using the Interval Quasimetric Embedding (IQE) formulation. QRL employs a Lagrangian optimization scheme where the objective maximizes the distance between states while maintaining locality constraints. We also compare against the approach by Park et al. (2024b), an offline reinforcement learning method that embeds states into a learned Hilbert space. In this space, the distance between embedded states approximates the MAD, leading to a symmetric distance metric that cannot capture the natural asymmetry of the true MAD. We include this comparison to demonstrate the benefits of methods that explicitly model the quasimetric nature of the MAD over those that do not.

Environments. To evaluate the proposed methods, we designed a suite of environments where the true MAD is known, enabling a precise quantitative assessment of our learned representations. This perfect knowledge of the ground truth distances allows us to rigorously evaluate how well different algorithms recover the underlying structure of the environment. A subset of the environments are illustrated in Figure 2, with full details provided in Appendix G. Our test environments span a comprehensive range of MDP characteristics:
378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431
- **NoisyGridWorld**: A continuous grid world environment with stochastic transitions. The agent can move in four cardinal directions, but the action may fail with a small probability, causing the agent to remain in the same state. The initial state is random and the goal is to reach a target state. The MAD is known and can be computed as the Manhattan distance between states. Moreover we included random noise in the observations by extending the state (*x, y*) with a random vector of size two resulting in a 4-dimensional state space, where the first two dimensions are the original coordinates and the last two dimensions correspond to noise.

- **KeyDoorGridWorld**: A discrete grid world environment where the agent must find a key to unlock a door. The agent can move in four cardinal directions and the state (*x, y, k*) is represented by the agent's position (*x, y*) and whether or not it has the key (k). The MAD is known and can be computed as the Manhattan distance between states where the distance between a state without the key and a state with the key is the sum of the distances to the key. The key can only be picked up and never dropped creating a strong asymmetry in the distance function.

- **CliffWalking**: The original CliffWalking environment as described by Sutton & Barto (1998).

The agent starts at the leftmost state and must reach the rightmost state while avoiding falling off the cliff. If the agent falls it returns to the starting state but the episode is not reset. This creates a strong asymmetry in the distance function, as the agent can take the shortcut by falling off the cliff to move between states.

- **PointMaze**: A continuous maze environment where the agent must navigate through a series of walls to reach a goal (Fu et al., 2020). The task in the environment is for a 2-DoF ball that is force-actuated in the Cartesian directions x and y, to reach a target goal in a closed maze. The underlying maze is a 2D grid with walls and obstacles, that we use in our experiments to approximate the ground truth MAD, by computing the all pairs shortest path using the Floyd- Warshall algorithm over the maze graph. We consider two variants of this environment: **UMaze** and **MediumMaze**.

- **OGBench PointMaze**: A suite of physics-based maze environments that extend the standard PointMaze to much larger and more challenging layouts (Park et al., 2024c). These environments are designed to test long-horizon reasoning and provide two types of datasets: *navigate*, collected by a noisy expert policy navigating to random goals, and *stitch*, consisting of short goal-reaching trajectories that must be combined to solve tasks.

Empirical Setup. We compared our two algorithms MadDist and TDMadDist against the QRL and Hilbert baselines. Each method was trained for 50,000 gradient steps on an offline dataset gathered by a random policy. For the CliffWalking, NoisyGridWorld, and KeyDoorGridWorld environments, we used 100 trajectories; for the PointMaze environments, we increased this to 1000 trajectories. All reported results are means over five independent runs (random seeds) to ensure statistical robustness. For full implementation details of our evaluation setup, see Appendix D. Figure 3 shows the Pearson correlation and coefficient of variation (CV) ratio for KeyDoorGridworld, CliffWalking, and the OGBench Giant Maze environments. The full results produced in all environments, including the Spearman correlations (which we found closely matched the Pearson correlations) can be found in Appendix F. Appendix E contains additional ablation studies, and demonstrates that MadDist and TDMadDist are robust to the size of the latent dimension and the choice of quasimetric, and that their performance degrades gracefully with dataset size. Table 1 reports additional results on a downstream planning task, where the learned distance embeddings are used to guide the agent toward specific goals. A detailed description of the planning setup is provided in Appendix H. Discussion. From the results in Figure 3, we can see that our proposed method MadDist outperforms the QRL and Hilbert baselines in all environments, being able to learn a more accurate approximation of the MAD. This is likely due to the fact that QRL only uses the locality constraints to learn the embeddings, while our method leverages the path distances between arbitrary states in a trajectory to form a more globally coherent representation. Both MadDist and TDMadDist significantly outperform the Hilbert baseline, particularly in highly asymmetric environments like CliffWalking and KeyDoorGridWorld. While TDMadDist underperforms the MadDist and QRL algorithm, its strong performance relative to Hilbert highlights the advantages of our quasimetric approach even when paired with a TD-based objective. Crucially, the high accuracy of the learned distance metric directly translates to superior performance in the downstream task of goal-oriented planning, as 432

![8_image_0.png](8_image_0.png) 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

Environments QRL TDMadDist Hilbert MadDist PM Giant Navigate 0.87 ± 0.21 0.99 ± **0.05** 0.16 ± 0.17 0.93 ± 0.17 PM Giant Stitch 0.95 ± 0.12 0.74 ± 0.26 0.05 ± 0.14 0.99 ± **0.07** PM Large Navigate 0.97 ± 0.09 0.70 ± 0.30 0.22 ± 0.20 1.00 ± **0.00** PM Large Stitch 0.90 ± 0.17 0.73 ± 0.24 0.17 ± 0.20 1.00 ± **0.00** PM Medium Navigate 0.86 ± 0.21 0.92 ± 0.16 0.55 ± 0.27 1.00 ± **0.00** PM Medium Stitch 0.81 ± 0.20 0.74 ± 0.24 0.67 ± 0.28 1.00 ± **0.00**

detailed in Table 1. MadDist achieves near-perfect or perfect success rates across all PointMaze environments, decisively outperforming all baselines. Its performance is particularly noteworthy in the Stitch environments, which require the model to compose information from disconnected trajectories, and the large-scale Giant environments, which test the ability to handle long-horizon tasks. This demonstrates that MadDist not only produces a quantitatively accurate distance function but also an effective and practical representation for planning.

## 8 Conclusion

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 In this paper, we present two novel algorithms for learning the Minimum Action Distance (MAD) from state trajectories. We also propose a novel quasimetric for learning asymmetric distance estimates, and introduce a set of benchmark domains that model several aspects that make distance learning difficult. In a controlled set of experiments we illustrate that the novel algorithms and proposed quasimetric outperform state-of-the-art algorithms for learning the MAD. While this work has concentrated on accurately approximating the MAD as a fundamental stepping stone, it opens several promising avenues for future research. One of them is the use of MAD estimates in transfer learning and non-stationary environments, where transition dynamics evolve over time yet maintain a consistent support. On the same line, MAD can be integrated as a heuristic in search algorithms, particularly in stochastic domains, to identify the properties that make it a robust and informative guidance signal under uncertainty. Having established reliable MAD approximation, it can now be incorporated into downstream tasks, including goal-conditioned planning and reinforcement learning, to quantify the empirical benefits it brings to complex decision-making problems. Finally, while MAD can serve as a useful heuristic even in stochastic environments, future work will explore whether it is possible to recover the Shortest Path Distance (SPD) or identify alternative quasimetrics that more closely align with it.

## References

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Rishabh Agarwal, Marlos C. Machado, Pablo Samuel Castro, and Marc G Bellemare. Contrastive Behavioral Similarity Embeddings for Generalization in Reinforcement Learning. In *Proceedings* of the 9th International Conference on Learning Representations, 2021.

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, and Wojciech Zaremba. Hindsight Experience Replay. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

Andre Barreto, Will Dabney, Remi Munos, Jonathan J Hunt, Tom Schaul, Hado P van Hasselt, and David Silver. Successor Features for Transfer in Reinforcement Learning. In Advances in Neural Information Processing Systems, volume 30, pp. 4055–4065. Curran Associates, Inc., 2017.

Richard Bellman. A Markovian Decision Process. *Journal of Mathematics and Mechanics*, 6(5):
679–684, 1957.

Robert Dadashi, Shideh Rezaeifar, Nino Vieillard, Leonard Hussenot, Olivier Pietquin, and Matthieu ´
Geist. Offline Reinforcement Learning with Pseudometric Learning. In *Proceedings of the 38th* International Conference on Machine Learning, pp. 2307–2318. PMLR, 2021.

Peter Dayan. Improving Generalization for Temporal Difference Learning: The Successor Representation. *Neural Computation*, 5(4):613–624, 1993.

Benjamin Eysenbach, Tianjun Zhang, Sergey Levine, and Russ R Salakhutdinov. Contrastive learning as goal-conditioned reinforcement learning. *Advances in Neural Information Processing Systems*, 35:35603–35620, 2022.

Robert W. Floyd. Algorithm 97: Shortest Path. *Communications of the ACM*, 5(6):345, June 1962.

ISSN 0001-0782.

Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4RL: Datasets for Deep Data-Driven Reinforcement Learning. *arXiv preprint arXiv:2004.07219*, 2020.

Dibya Ghosh, Abhishek Gupta, Ashwin Reddy, Justin Fu, Coline Manon Devin, Benjamin Eysenbach, and Sergey Levine. Learning to Reach Goals via Iterated Supervised Learning. In *Proceedings of* the 8th International Conference on Learning Representations, 2020.

Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Deep sparse rectifier neural networks. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, JMLR Workshop and Conference Proceedings, pp. 315–323, 2011.

Kristian Hartikainen, Xinyang Geng, Tuomas Haarnoja, and Sergey Levine. Dynamical Distance Learning for Semi-Supervised and Unsupervised Skill Discovery. In *Proceedings of the 8th* International Conference on Learning Representations, 2020.

Dan Hendrycks and Kevin Gimpel. Gaussian Error Linear Units (GELUs). arXiv preprint arXiv:1606.08415, 2016.

Zhengyao Jiang, Tianjun Zhang, Michael Janner, Yueying Li, Tim Rocktaschel, Edward Grefen- ¨
stette, and Yuandong Tian. Efficient planning in a compact latent action space. arXiv preprint arXiv:2208.10291, 2022.

Leslie Pack Kaelbling. Learning to Achieve Goals. International Joint Conference on Artificial Intelligence, 2:1094–1098, August 1993.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations. PMLR, 2015.

Gunter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter. Self-normalizing ¨
neural networks. *Advances in neural information processing systems*, 30, 2017.

Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline Reinforcement Learning with Implicit Q-Learning. In *Proceedings of the 10th International Conference on Learning Representations*, 2022.

Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training. In *Proceedings of the 11th International Conference on Learning Representations*, 2022.

Marlos C. Machado. Efficient Exploration in Reinforcement Learning through Time-Based Representations. PhD thesis, University of Alberta, 2019.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. *nature*, 518(7540):529–533, 2015.

Vivek Myers, Chongyi Zheng, Anca Dragan, Sergey Levine, and Benjamin Eysenbach. Learning Temporal Distances: Contrastive Successor Features Can Provide a Metric Structure for Decision- Making. In *Proceedings of the 41st International Conference on Machine Learning*, pp. 37076– 37096. PMLR, 2024.

Whitney K Newey and James L Powell. Asymmetric Least Squares Estimation and Testing. Econometrica: Journal of the Econometric Society, pp. 819–847, 1987.

Seohong Park, Dibya Ghosh, Benjamin Eysenbach, and Sergey Levine. HIQL: Offline Goal-
Conditioned RL with Latent States as Actions. In Advances in Neural Information Processing Systems, volume 36, pp. 34866–34891. Curran Associates, Inc., 2023.

Seohong Park, Oleh Rybkin, and Sergey Levine. METRA: Scalable Unsupervised RL with Metric-
Aware Abstraction. In Proceedings of the 12th International Conference on Learning Representations, 2024a.

Seohong Park, Tobias Kreiman, and Sergey Levine. Foundation Policies with Hilbert Representations.

In *Proceedings of the 41st International Conference on Machine Learning*, pp. 39737–39761. PMLR, 2024b.

Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. Ogbench: Benchmarking offline goal-conditioned rl. *arXiv preprint arXiv:2410.20092*, 2024c.

Silviu Pitis, Harris Chan, Kiarash Jamali, and Jimmy Ba. An Inductive Bias for Distances: Neural Nets that Respect the Triangle Inequality. In Proceedings of the 8th International Conference on Learning Representations. Curran Associates, Inc., 2020.

Lorenzo Steccanella and Anders Jonsson. State Representation Learning for Goal-Conditioned Reinforcement Learning. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 84–99. Springer, 2022.

Richard S. Sutton and Andrew G. Barto. *Reinforcement learning: An introduction*. MIT Press, Cambridge, 1998.

Richard S. Sutton and Andrew G. Barto. *Reinforcement Learning: An Introduction*. MIT Press, Cambridge, MA, 2018.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Jean Tarbouriech, Runlong Zhou, Simon S. Du, Matteo Pirotta, Michal Valko, and Alessandro Lazaric.

Stochastic Shortest Path: Minimax, Parameter-Free and Towards Horizon-Free Regret. In *Advances* in Neural Information Processing Systems, volume 34, pp. 6843–6855. Curran Associates, Inc.,
2021.

Hado Van Hasselt, Arthur Guez, and David Silver. Deep Reinforcement Learning with Double Q-Learning. In *Proceedings of the AAAI conference on artificial intelligence*, volume 30, 2016.

Kaixin Wang, Kuangqi Zhou, Qixin Zhang, Jie Shao, Bryan Hooi, and Jiashi Feng. Towards better laplacian representation in reinforcement learning with generalized graph drawing. In *Proceedings* of the 38th International Conference on Machine Learning, volume 139, pp. 11003–11012. PMLR, 2021.

Kaixin Wang, Kuangqi Zhou, Jiashi Feng, Bryan Hooi, and Xinchao Wang. Reachability-aware Laplacian representation in reinforcement learning. In Proceedings of the 40th International Conference on Machine Learning, volume 202, pp. 36670–36693. PMLR, 2023a.