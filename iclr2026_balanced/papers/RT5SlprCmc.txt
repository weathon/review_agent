# LEARNING THE MINIMUM ACTION DISTANCE


**Anonymous authors**
Paper under double-blind review


ABSTRACT


This paper presents a state representation framework for Markov decision processes
(MDPs) that can be learned solely from state trajectories, requiring neither reward
signals nor the actions executed by the agent. We propose learning the _minimum_
_action distance_ (MAD), defined as the minimum number of actions required to
transition between states, as a fundamental metric that captures the underlying
structure of an environment. MAD naturally enables critical downstream tasks
such as goal-conditioned reinforcement learning and reward shaping by providing
a dense, geometrically meaningful measure of progress. Our self-supervised
learning approach constructs an embedding space where the distances between
embedded state pairs correspond to their MAD, accommodating both symmetric
and asymmetric approximations. We evaluate the framework on a comprehensive
suite of environments with known MAD values, encompassing both deterministic
and stochastic dynamics, as well as discrete and continuous state spaces, and
environments with noisy observations. Empirical results demonstrate that the
proposed approach not only efficiently learns accurate MAD representations across
these diverse settings but also significantly outperforms existing state representation
methods in terms of representation quality.


1 INTRODUCTION


In reinforcement learning (Sutton & Barto, 1998), an agent aims to learn useful behaviors through
continuing interaction with its environment. Specifically, by observing the outcomes of its actions,
a reinforcement learning agent learns over time how to select actions in order to maximize the
expected cumulative reward it receives from its environment. An important need in applications of
reinforcement learning is the ability to generalize, not only to previously unseen states, but also to
variations of its environment that the agent has not previously interacted with.


In many applications of reinforcement learning, it is useful to define a metric that measures the
similarity of two states in the environment. Such a metric can be used, e.g., to define equivalence
classes of states in order to accelerate learning, to decompose the problem into a hierarchy of smaller
subproblems that are easier to solve, or to perform transfer learning in case the environment changes
according to some parameters but retains part of the structure of the original environment. Such a
metric can also be used as a heuristic in goal-conditioned reinforcement learning, in which the agent
has to achieve different goals in the same environment.


The Minimum Action Distance (MAD) has proved useful as a similarity metric, with impressive
applications in various areas of reinforcement learning, including policy learning (Wang et al., 2023b;
Park et al., 2023), reward shaping (Steccanella & Jonsson, 2022), and option discovery (Park et al.,
2024a;b). While prior work has demonstrated the advantages of using MAD, how best to approximate
it remains an open problem. Existing methods have not been systematically evaluated on their ability
to approximate the MAD function itself, and many rely on symmetric approximations, even though
the true MAD is inherently asymmetric.


We make three main contributions towards fast, accurate approximation of the MAD. First, we
propose two novel algorithms for learning MAD using only state trajectories collected by an agent
interacting with its environment. Unlike previous work, the proposed algorithms naturally support
both symmetric and asymmetric distances, and incorporate both short- and long-term information
about how distant two states are from one another. Secondly, we define a novel quasimetric distance
function that is computationally efficient and that, in spite of its simplicity, outperforms more


1


elaborate quasimetrics in the existing literature. Finally, we introduce a diverse suite of environments

- including those with discrete and continuous state spaces, stochastic and deterministic dynamics,
and directed and undirected transitions — in which the ground-truth MAD is known, enabling a
systematic and controlled evaluation of different MAD approximation methods.


Figure 1 illustrates the steps of MAD representation learning: an agent collects state trajectories from
an unknown environment, which are used to learn a state embedding that implicitly defines a distance
function between states.


2 RELATED WORK


In applications such as goal-conditioned reinforcement learning (Ghosh et al., 2020) and stochastic
shortest-path problems (Tarbouriech et al., 2021), the temporal distance is measured as the expected
number of steps required to reach one state from another state under some policy. In contrast, the MAD
is a lower bound on the number of steps based solely on the support of the transition function. This
distinction makes the MAD efficient to compute and robust to changes in the transition probabilities
as long as the support over next states remains the same, making it suitable for representation learning
and transfer learning.


Prior work has explored the connection between the MAD and optimal goal-conditioned value
functions (Kaelbling, 1993). Park et al. (2023) highlight this connection and propose a hierarchical
approach that improves distance estimates over long horizons, and Park et al. (2024a) embed states
into a learned latent space where the distance between embedded states directly reflects an onpolicy measure of the temporal distance (Hartikainen et al., 2020). Park et al. (2024b) and Ma
et al. (2022) extend this idea to the offline setting, learning embeddings from arbitrary experience
such that Euclidean distances between state embeddings approximate the MAD. As an alternative
to approximating the MAD using goal-conditioned value functions, Steccanella & Jonsson (2022)
formulate learning a state embedding in which distances approximate the MAD as a constrained
optimization problem, where bounds on the distance between embedded states are derived from state
trajectory data. Although their formulations differ, these approaches ultimately seek to learn the same
underlying quantity: the minimum number of actions required to move between two states.


These existing approaches share a common limitation: they rely on symmetric distance metrics such
as the Euclidean distance between state embeddings to approximate the MAD. As such, they cannot
capture the asymmetry of the true MAD in environments with irreversible dynamics. In contrast, the
approach we develop here supports the use of asymmetric distance metrics (or, _quasimetrics_ ), which
can better capture the directional structure in many environments.


Some prior work has already explored the use of quasimetrics in reinforcement learning. Wang
et al. (2023b) learn an asymmetric distance function that approximates the MAD by preserving local
structure while maintaining global distances. Their method differs from the one we propose in two
ways. First, their method does not leverage the existing distance along a trajectory as supervision
for the learning process. Secondly, they use the Interval Quasimetric Embedding (IQE) (Wang &
Isola, 2022) to learn the distance function. Dadashi et al. (2021) and Agarwal et al. (2021) learn
embeddings and define a pseudometric between states as the Euclidean distance between their


2


embeddings. Unlike our work, they use loss functions inspired by bisimulation to learn both state and
state-action embeddings.


Successor features (Dayan, 1993; Barreto et al., 2017), and time-contrastive representations (Eysenbach et al., 2022) have also been used to define notions of temporal distance. Myers et al. (2024)
introduces time-contrastive successor features, defining a distance metric based on the difference
between discounted future occupancies of state features learned via time-contrastive learning. While
their metric satisfies the triangle inequality and naturally handles both stochasticity and asymmetry,
the resulting distances reflect expected discounted state visitations under a specific behavior policy
and lack an intuitive interpretation. In contrast, approaches that approximate the MAD are naturally
interpretable as a lower bound on the number of actions needed to transition between two states.


Laplacian-based representation learning methods (Wu et al., 2019; Machado, 2019; Wang et al.,
2021; 2023a) learn embeddings from the spectral structure of random walks over the transition graph,
producing representations that reflect global connectivity in the state space. However, these methods
are typically defined on a symmetrized transition operator or undirected Laplacian, and the induced
geometry measures diffusion-based similarity rather than directed reachability. As a result, distances
in these embeddings are fundamentally symmetric and do not correspond to the minimum number
of actions required to move between two states, making them poorly suited to environments with
irreversible or asymmetric dynamics.


3 BACKGROUND


In this section, we introduce the notation and concepts used throughout the paper. Given a finite set _X_,
we use ∆( _X_ ) = _{p ∈_ R _[X]_ _|_ [�] _x_ _[p][x]_ [= 1] _[, p][x]_ _[≥]_ [0 (] _[∀][x]_ [)] _[}]_ [ to denote the probability simplex (i.e. the set]
of all probability distributions over _X_ ). A rectified linear unit (ReLU) is a function relu : R _[d]_ _→_ R _[d]_ _≥_ 0
defined on any vector _x ∈_ R _[d]_ as relu( _x_ ) = [max(0 _, xi_ )] _[d]_ _i_ =1 [.]


**Markov Decision Processes (MDPs).** An MDP (Bellman, 1957) is a tuple _M_ = _⟨S, A, R, P, D, γ⟩_,
where _S_ is the state space, _A_ is the action space, _R_ : _S_ _× A_ _→_ R is the reward function, _P_ :
_S ×A →_ ∆( _S_ ) is the transition kernel, _D_ _∈_ ∆( _S_ ) is the initial state distribution, and _γ_ _∈_ [0 _,_ 1] is the
discount factor. At each time _t_, the learning agent observes a state _st_ _∈S_, selects an action _at_ _∈A_,
receives a reward _rt_ = _R_ ( _st, at_ ) and transitions to a new state _st_ +1 _∼P_ ( _st, at_ ). The learning agent
selects actions using a policy _π_ : _S_ _→_ ∆( _A_ ), a mapping from states to probability distributions over
actions. In our work, the state space _S_ can be either discrete or continuous.


**Reinforcement learning (RL).** RL (Sutton & Barto, 2018) is a family of algorithms whose purpose
is to learn a policy _π_ that maximizes some measure of expected future reward. In this paper, however,
we consider the problem of representation learning, and hence we are not directly concerned with
the problem of learning a policy. Concretely, we wish to learn a distance function between pairs
of states that can later be used by an RL agent to learn more efficiently. In this setting, we assume
that the learning agent uses a behavior policy _π_ b to collect trajectories. Since we are interested
in learning a distance function over state pairs, actions are relevant only for determining possible
transitions between states, and rewards are not relevant at all. Hence for our purposes a trajectory
_τ_ = ( _s_ 0 _, s_ 1 _, . . ., sn_ ) is simply a sequence of states.


4 THE MINIMUM ACTION DISTANCE


Given an MDP _M_ = _⟨S, A, R, P, D, γ⟩_ and a state pair ( _s, s_ _[′]_ ) _∈S_ [2], the Minimum Action Distance,
_d_ MAD( _s, s_ _[′]_ ), is defined as the minimum number of decision steps needed to transition from _s_ to _s_ _[′]_ . In
deterministic MDPs, the MAD is always realizable using an appropriate policy; in stochastic MDPs,
the MAD is a lower bound on the actual number of decision steps of any policy. Let _R ⊆S_ [2] be a
relation such that ( _s, s_ _[′]_ ) _∈_ _R_ if and only if there exists an action _a ∈A_ that satisfies _P_ ( _s_ _[′]_ _|s, a_ ) _>_ 0.
That is, _R_ contains all state pairs ( _s, s_ _[′]_ ) such that _s_ _[′]_ is reachable in one step from _s_ . We can formulate


3


the problem of computing _d_ MAD as a constrained optimization problem:


s.t. _d_ ( _s, s_ ) = 0 _∀s ∈S,_

_d_ ( _s, s_ _[′]_ ) _≤_ 1 _∀_ ( _s, s_ _[′]_ ) _∈_ _R,_

_d_ ( _s, s_ _[′]_ ) _≤_ _d_ ( _s, s_ _[′′]_ ) + _d_ ( _s_ _[′′]_ _, s_ _[′]_ ) _∀_ ( _s, s_ _[′]_ _, s_ _[′′]_ ) _∈S_ [3] _._


It is straightforward to show that _d_ MAD is the unique solution to equation 1 (see Appendix A).
Concretely, _d_ MAD satisfies the second constraint with equality, i.e. _d_ ( _s, s_ _[′]_ ) = 1 for all ( _s, s_ _[′]_ ) _∈_ _R_ . If
the state space _S_ is finite, the constrained optimization problem is precisely the linear programming
formulation of the all-pairs shortest path problem for the graph ( _S, R_ ) with edge costs 1. This graph
is itself a determinization of the MDP _M_ (Yoon et al., 2007). In this case we can compute _d_ MAD
exactly using the well-known Floyd-Warshall algorithm (Floyd, 1962; Warshall, 1962). If the state
space _S_ is continuous, _R_ is still well-defined, and hence there still exists a solution which satisfies
_d_ ( _s, s_ _[′]_ ) = 1 for all ( _s, s_ _[′]_ ) _∈_ _R_ even though the states can no longer be enumerated.


An alternative to the MAD is computing the stochastic shortest path (SSP; Tarbouriech et al., 2021)
between each pair of states. In deterministic MDPs, the MAD and SSP are equivalent. In stochastic
MDPs, the SSP can provide more realistic distance estimates than the MAD when some transitions
have very low probabilities. However, computing the all-pairs SSP requires solving a linear program
over transition probabilities, which is computationally demanding. In contrast, the MAD can be
computed efficiently and remains a useful approximation in many domains (e.g. in navigation
problems and when using sticky actions). Moreover, unlike the SSP, the MAD depends only on the
support of the transition kernel and is otherwise robust to changes in transition probabilities, which is
particularly useful for transfer learning.


Even when the state space _S_ is finite, we may not have explicit knowledge of the relation _R_ . In
addition, the time complexity of the Floyd-Warshall algorithm is _O_ ( _|S|_ [3] ), and the number of states
may be too large to run the algorithm in practice. If the state space _S_ is continuous, then we cannot
even explicitly form a graph ( _S, R_ ). Hence we are interested in estimating _d_ MAD in the setting for
which we can access trajectories only through sampling. For this purpose, let us assume that the
learning agent uses a behavior policy _π_ b to collect a dataset of trajectories _D_ = _{τ_ 1 _, . . ., τk}_ . Define
_SD_ _⊆S_ as the subset of states that appear on any trajectory in _D_ . Given a trajectory _τ_ = _{s_ 0 _, ..., sn}_
and any two states _si_ and _sj_ on the trajectory such that 0 _≤_ _i_ _<_ _j_ _≤_ _n_, it is easy to see that _j −_ _i_
is an upper bound on _d_ MAD( _si, sj_ ), since _sj_ is reachable in _j −_ _i_ steps from _si_ on the trajectory _τ_ .
By an abuse of notation, we often write ( _si, sj_ ) _∈_ _τ_ to refer to a state pair on the trajectory _τ_ with
indices _i_ and _j_ such that _i < j_, and we write ( _si, sj_ ) _∼_ _τ_ in order to sample two such states from _τ_ .


Steccanella & Jonsson (2022) learn a parameterized state embedding _ϕθ_ : _S_ _→_ R _[d]_ and define a
distance function _dθ_ ( _s, s_ _[′]_ ) = _d_ ( _ϕθ_ ( _s_ ) _, ϕθ_ ( _s_ _[′]_ )), where _d_ is any distance metric in Cartesian space.
The parameter vector _θ_ of the state embedding is learned by minimizing the loss function


_L_ = E _τ_ _∼D,_ ( _si,sj_ ) _∼τ_ �( _dθ_ ( _si, sj_ ) _−_ ( _j −_ _i_ )) [2] + _wc ·_ relu( _dθ_ ( _si, sj_ ) _−_ ( _j −_ _i_ )) [2][�] _,_ (2)


where _wc_ _>_ 0 is a regularization factor that multiplies a penalty term which substitutes the upper
bound constraints _dθ_ ( _si, sj_ ) _≤_ _j −_ _i_ . If the distance metric _d_ satisfies the triangle inequality (e.g. any
norm _d_ = _|| · ||p_ ) then the constraints _dθ_ ( _s, s_ ) = 0 and the triangle inequality automatically hold.
Enforcing the constraint _dθ_ ( _si, sj_ ) _≤_ _j −_ _i_ for each state pair ( _si, sj_ ) on trajectories, rather than only
consecutive pairs, helps learn better distance estimates, at the cost of a larger number of constraints.


5 ASYMMETRIC DISTANCE METRICS


A limitation of previous work is that the chosen distance metric _d_ is symmetric, while the MAD _d_ MAD
may not be symmetric. In this section, we review several asymmetric distance metrics. Concretely, a
quasimetric is a function _dq_ : R _[d]_ _×_ R _[d]_ _→_ R+ that satisfies the following three conditions:


 - **Q1** (Identity): _dq_ ( _x, x_ ) = 0.

 - **Q2** (Non-negativity): _dq_ ( _x, y_ ) _≥_ 0.


4


_d_ MAD = arg max
_d_


 - _d_ ( _s, s_ _[′]_ ) _,_ (1)


( _s,s_ _[′]_ ) _∈S_ [2]


- **Q3** (Triangle inequality): _dq_ ( _x, z_ ) _≤_ _dq_ ( _x, y_ ) + _dq_ ( _y, z_ ).


A quasimetric does not require symmetry, i.e., _dq_ ( _x, y_ ) = _dq_ ( _y, x_ ) does not hold in general.


We define a simple quasimetric _d_ simple using rectified linear units:


where _α ∈_ [0 _,_ 1] balances the influence of the maximum and the average. This construction yields a
quasimetric that inherently respects the triangle inequality while accounting for directional differences
between the matrices _X_ and _Y_ .


Given any of the above quasimetrics _dq_ (i.e., _d_ simple, _d_ WN or _d_ IQE), we can now define an asymmetric
distance function _dθ_ ( _s, s_ _[′]_ ) = _dq_ ( _ϕθ_ ( _s_ ) _, ϕθ_ ( _s_ _[′]_ )). In the case of _d_ IQE, the state embedding _ϕ_ : _S_ _→_ R _[d]_
produces an output that is reshaped into a _k × m_ matrix structure to parameterize the intervals. The
choice of quasimetric directly shapes the trade-offs in computational cost and optimization dynamics.
In Appendix E, we present an ablation study examining how this choice affects our algorithms.


6 LEARNING ASYMMETRIC MAD ESTIMATES


Here, we propose two novel variants of the MAD learning approach. Each trains a state encoding _ϕθ_
that maps states to an embedding space and uses a quasimetric _dq_ to compute distances _dθ_ ( _s, s_ _[′]_ ) =
_dq_ ( _ϕθ_ ( _s_ ) _, ϕθ_ ( _s_ _[′]_ )) between pairs of states ( _s, s_ _[′]_ ). Both variants support any quasimetric formulation
such as _d_ simple, _d_ WN and _d_ IQE, and can incorporate additional features such as gradient clipping. A
full derivation of these learning objectives is provided in Appendix C.


5


_d_ simple( _x, y_ ) = _α_ max(relu( _x −_ _y_ )) + (1 _−_ _α_ ) [1]

_d_


_d_

- relu( _xi −_ _yi_ ) _._ (3)


_i_


This metric is a weighted average of the maximum and average positive difference between the
vectors _x_ and _y_ along any dimension, where _α_ _∈_ [0 _,_ 1] is a weight. In Appendix B, we show that
_d_ simple satisfies the triangle inequality and latent positive homogeneity (Wang & Isola, 2022).


The Wide Norm quasimetric (Pitis et al., 2020), _d_ WN, applies a learned transformation to an asymmetric representation of the difference between two states. The Wide Norm is defined as


_d_ WN( _x, y_ ) = _||W_ (relu( _x −_ _y_ ) :: relu( _y −_ _x_ )) _||_ 2 _,_


where “::” denotes concatenation and _W_ _∈_ R _[k][×]_ [2] _[d]_ is a learned weight matrix. This ensures that
_d_ WN( _x, y_ ) is non-negative and satisfies the triangle inequality, while concatenation is asymmetric.


The Interval Quasimetric Embedding (IQE) (Wang & Isola, 2022) leverages the Lebesgue measure of
interval unions to capture asymmetric distances. IQE interprets the latent embeddings as matrices
_X, Y_ _∈_ R _[k][×][m]_ (typically obtained by reshaping a flat output vector of dimension _d_ = _k · m_ ). Let _xij_
denote the element in row _i_ and column _j_ of matrix _X_ . For each row _i_, we construct an interval by
taking the union over the intervals defined by matrices _X_ and _Y_ :


_Ii_ ( _X, Y_ ) =


_m_

- [ _xij,_ max _{xij,_ _yij}_ ] _._


_j_ =1


The length of this interval, denoted by _Li_ ( _X, Y_ ), is computed as its Lebesgue measure. The IQE
distance is obtained by aggregating these row-wise lengths. For example, one may define


_d_ IQE( _X, Y_ ) =


or, alternatively, using a maxmean reduction:


_k_

- _Li_ ( _X, Y_ ) _,_


_i_ =1


_d_ IQE-mm( _X, Y_ ) = _α_ max [1]
1 _≤i≤k_ _[L][i]_ [(] _[X, Y]_ [ ) + (1] _[ −]_ _[α]_ [)] _k_


_k_

- _Li_ ( _X, Y_ ) _,_


_i_ =1


6.1 MADDIST: DIRECT DISTANCE LEARNING


The first algorithm, which we call _MadDist_, learns state distances using an approach similar to prior
work (Steccanella & Jonsson, 2022), but differs in the use of a quasimetric distance function and a
scale-invariant loss. Concretely, MadDist minimizes the following composite loss function:


_L_ = _Lo_ + _wrLr_ + _wcLc._ (4)


The main objective, _Lo_, is a scaled version of the square difference in equation 2:


Hence if the current distance estimate _dθ′_ ( _si_ +1 _, sj_ ) computed using the target embedding _ϕθ′_ is
smaller than _j −_ ( _i_ + 1), the objective is to make _dθ_ ( _si, sj_ ) equal to 1 + _dθ′_ ( _si_ +1 _, sj_ ).

We also modify the second loss term _L_ _[′]_ _r_ [to include bootstrapped distances:]

_L_ _[′]_ _r_ [=][ E] _τ_ _∼D,_ ( _si,sj_ ) _∼τ,sr∼SD_ [[(] _[d][θ]_ [(] _[s][i][, s][i]_ [+1]


1+ _dθ′_ ( _si_ +1 _,sr_ ) _−_ 1 [2] _._ (9) [Given a state] _[ s][i]_ [sampled from a trajectory of] _[ D]_ [ and a random state] _[ s][r]_ _[∈S][D]_ [,]

the objective is to make _dθ_ ( _si, sr_ ) equal to 1 + _dθ′_ ( _si_ +1 _, sr_ ).

The target network parameters _θ_ _[′]_ are updated in each time step via an exponential moving average
with hyperparameter _β_ _∈_ (0 _,_ 1):
_θ_ _[′]_ _←_ (1 _−_ _β_ ) _θ_ _[′]_ + _βθ._ (10)


6


_Lo_ = E _τ_ _∼D,_ ( _si,sj_ ) _∼τ_


�� _dθ_ ( _si, sj_ ) _−_ 1�2 [�]
_j −_ _i_


_._ (5)


Crucially, scaling makes the loss invariant to the magnitude of the estimation error, which typically
increases as a function of _j −_ _i_ . In other words, states that are further apart on a trajectory do not
necessarily dominate the loss simply because the magnitude of the estimation error is larger.


The second loss term, _Lr_, which is weighted by a factor _wr_ _>_ 0, is a contrastive loss that encourages
separation between state pairs randomly sampled from all trajectories:


��2 [�]


_Lr_ = E( _s,s′_ ) _∼SD_


�� relu 1 _−_ _[d][θ]_ [(] _[s, s][′]_ [)]

_d_ max


(6)


where _d_ max is a hyperparameter. Finally, the loss term _Lc_, which is weighted by a factor _wc_ _>_ 0,
enforces the upper bound constraints. Specifically, let _D≤Hc_ denote the set of state pairs sampled
from trajectories in _D_ such that the index difference satisfies 1 _≤_ _j_ _−_ _i_ _≤_ _Hc_ (where _Hc_ is a
hyperparameter), i.e.


_D≤Hc_ = _{_ ( _si, sj_ ) _| τ_ _∈D,_ _si, sj_ _∈_ _τ,_ 1 _≤_ _j −_ _i ≤_ _Hc} ._


Then, the constraint loss is defined as:


_Lc_ = E( _si,sj_ ) _∼D≤Hc_ �(relu ( _dθ_ ( _si, sj_ ) _−_ ( _j −_ _i_ ))) [2][�] _._ (7)


6.2 TDMADDIST: TEMPORAL DIFFERENCE LEARNING


The second algorithm, which we call _TDMadDist_, incorporates temporal difference learning principles
by maintaining a separate target embedding _ϕθ′_ and learning via bootstrapped targets. Specifically,
TDMadDist learns by minimizing the loss function _L_ _[′]_ = _L_ _[′]_ _o_ [+] _[ w][r][L]_ _r_ _[′]_ [+] _[ w][c][L][c]_ [, where] _[ L][c]_ [is the loss]
term from equation 7 that enforces the upper bound constraints.

The main objective _L_ _[′]_ _o_ [of TDMadDist is modified to include bootstrapped distances:]


_L_ _[′]_ _o_ [=][ E] _τ_ _∼D,_ ( _si,sj_ ) _∼τ_


�� _dθ_ ( _si, sj_ ) �2 [�]
min( _j −_ _i,_ 1 + _dθ′_ ( _si_ +1 _, sj_ )) _[−]_ [1]


_._ (8)


CliffWalking MediumMaze OGBenchGiantMaze


Figure 2: A subset of the environments used in our analysis.


7 EXPERIMENTS


We evaluate our proposed MAD learning algorithms on a diverse set of environments with varying
characteristics, including deterministic and stochastic dynamics, discrete and continuous state spaces,
and environments with noisy observations. Our analysis is directed by the following questions:


 - How accurately do our learned embeddings capture the true minimum action distances?

 - How does the performance of our method compare to existing quasimetric learning approaches?

 - How robust is our approach to environmental stochasticity and observation noise?


**Evaluation Metrics.** We evaluate the quality of our learned representations using three metrics:


 - **Spearman Correlation (** _ρ_ **)** : Measures the preservation of ranking relationships between state
pairs. A high Spearman correlation indicates that if state _si_ is farther from state _sj_ than from
state _sk_ in the true environment, our learned metric also predicts this same ordering. Perfect
preservation of distance rankings gives _ρ_ = 1.

 - **Pearson Correlation (** _r_ **)** : Measures the linear relationship between predicted and true distances.
A high Pearson correlation indicates that our learned distances scale proportionally with true
distances (i.e. when true distances increase, our predictions increase linearly as well). Perfect
linear correlation gives _r_ = 1.

 - **Ratio Coefficient of Variation (CV)** : Measures the consistency of our distance scaling across
different state pairs. A low CV indicates that our predicted distances maintain a consistent ratio
to true distances throughout the state space. For example, if we consistently predict distances
that are approximately 1.5 times the true distance, CV will be low. High variation in this ratio
across different state pairs results in high CV. More formally, given a set of ground truth distances
_d_ 1 _, d_ 2 _, ..., dn_ and their corresponding predicted distances _d_ [ˆ] 1 _,_ _d_ [ˆ] 2 _, ...,_ _d_ [ˆ] _n_ where _di_ _>_ 0, we compute
the ratios _ri_ = _d_ [ˆ] _i/di_ . The Ratio CV is given by


**Baselines.** We compare our methods against QRL (Wang et al., 2023b), a recent quasimetric
reinforcement learning approach that learns state representations using the Interval Quasimetric
Embedding (IQE) formulation. QRL employs a Lagrangian optimization scheme where the objective
maximizes the distance between states while maintaining locality constraints.


We also compare against the approach by Park et al. (2024b), an offline reinforcement learning
method that embeds states into a learned Hilbert space. In this space, the distance between embedded
states approximates the MAD, leading to a symmetric distance metric that cannot capture the natural
asymmetry of the true MAD. We include this comparison to demonstrate the benefits of methods that
explicitly model the quasimetric nature of the MAD over those that do not.


**Environments.** To evaluate the proposed methods, we designed a suite of environments where the
true MAD is known, enabling a precise quantitative assessment of our learned representations. This
perfect knowledge of the ground truth distances allows us to rigorously evaluate how well different
algorithms recover the underlying structure of the environment. A subset of the environments are
illustrated in Figure 2, with full details provided in Appendix G.


Our test environments span a comprehensive range of MDP characteristics:


7


_CV_ = _[σ][r]_ =

_µr_


- 1 - _n_
_n_ _i_ =1 [(] _[r][i][ −]_ _[µ][r]_ [)][2]

1    - _n_ _,_ (11)
_n_ _i_ =1 _[r][i]_


- **NoisyGridWorld** : A continuous grid world environment with stochastic transitions. The agent
can move in four cardinal directions, but the action may fail with a small probability, causing the
agent to remain in the same state. The initial state is random and the goal is to reach a target state.
The MAD is known and can be computed as the Manhattan distance between states. Moreover we
included random noise in the observations by extending the state ( _x, y_ ) with a random vector of
size two resulting in a 4-dimensional state space, where the first two dimensions are the original
coordinates and the last two dimensions correspond to noise.


 - **KeyDoorGridWorld** : A discrete grid world environment where the agent must find a key to
unlock a door. The agent can move in four cardinal directions and the state ( _x, y, k_ ) is represented
by the agent’s position ( _x, y_ ) and whether or not it has the key ( _k_ ). The MAD is known and can
be computed as the Manhattan distance between states where the distance between a state without
the key and a state with the key is the sum of the distances to the key. The key can only be picked
up and never dropped creating a strong asymmetry in the distance function.


 - **CliffWalking** : The original CliffWalking environment as described by Sutton & Barto (1998).
The agent starts at the leftmost state and must reach the rightmost state while avoiding falling off
the cliff. If the agent falls it returns to the starting state but the episode is not reset. This creates a
strong asymmetry in the distance function, as the agent can take the shortcut by falling off the
cliff to move between states.


 - **PointMaze** : A continuous maze environment where the agent must navigate through a series
of walls to reach a goal (Fu et al., 2020). The task in the environment is for a 2-DoF ball that
is force-actuated in the Cartesian directions x and y, to reach a target goal in a closed maze.
The underlying maze is a 2D grid with walls and obstacles, that we use in our experiments to
approximate the ground truth MAD, by computing the all pairs shortest path using the FloydWarshall algorithm over the maze graph. We consider two variants of this environment: **UMaze**
and **MediumMaze** .


 - **OGBench PointMaze** : A suite of physics-based maze environments that extend the standard
PointMaze to much larger and more challenging layouts (Park et al., 2024c). These environments
are designed to test long-horizon reasoning and provide two types of datasets: _navigate_, collected
by a noisy expert policy navigating to random goals, and _stitch_, consisting of short goal-reaching
trajectories that must be combined to solve tasks.


**Empirical Setup.** We compared our two algorithms MadDist and TDMadDist against the QRL and
Hilbert baselines. Each method was trained for 50 _,_ 000 gradient steps on an offline dataset gathered
by a random policy. For the CliffWalking, NoisyGridWorld, and KeyDoorGridWorld environments,
we used 100 trajectories; for the PointMaze environments, we increased this to 1000 trajectories. All
reported results are means over five independent runs (random seeds) to ensure statistical robustness.
For full implementation details of our evaluation setup, see Appendix D.


Figure 3 shows the Pearson correlation and coefficient of variation (CV) ratio for KeyDoorGridworld, CliffWalking, and the OGBench Giant Maze environments. The full results produced in all
environments, including the Spearman correlations (which we found closely matched the Pearson
correlations) can be found in Appendix F. Appendix E contains additional ablation studies, and
demonstrates that MadDist and TDMadDist are robust to the size of the latent dimension and the
choice of quasimetric, and that their performance degrades gracefully with dataset size.


Table 1 reports additional results on a downstream planning task, where the learned distance embeddings are used to guide the agent toward specific goals. A detailed description of the planning setup
is provided in Appendix H.


**Discussion.** From the results in Figure 3, we can see that our proposed method MadDist outperforms
the QRL and Hilbert baselines in all environments, being able to learn a more accurate approximation
of the MAD. This is likely due to the fact that QRL only uses the locality constraints to learn the
embeddings, while our method leverages the path distances between arbitrary states in a trajectory
to form a more globally coherent representation. Both MadDist and TDMadDist significantly
outperform the Hilbert baseline, particularly in highly asymmetric environments like CliffWalking
and KeyDoorGridWorld. While TDMadDist underperforms the MadDist and QRL algorithm, its
strong performance relative to Hilbert highlights the advantages of our quasimetric approach even
when paired with a TD-based objective. Crucially, the high accuracy of the learned distance metric
directly translates to superior performance in the downstream task of goal-oriented planning, as


8


Figure 3: Pearson correlation coefficients and coefficient of variation (CV) ratios across a selection
of test environments. Shaded regions minimum and maximum values across three random seeds.


Environments QRL TDMadDist Hilbert MadDist


PM Giant Navigate 0.87 _±_ 0.21 **0.99** _±_ **0.05** 0.16 _±_ 0.17 0.93 _±_ 0.17
PM Giant Stitch 0.95 _±_ 0.12 0.74 _±_ 0.26 0.05 _±_ 0.14 **0.99** _±_ **0.07**
PM Large Navigate 0.97 _±_ 0.09 0.70 _±_ 0.30 0.22 _±_ 0.20 **1.00** _±_ **0.00**
PM Large Stitch 0.90 _±_ 0.17 0.73 _±_ 0.24 0.17 _±_ 0.20 **1.00** _±_ **0.00**
PM Medium Navigate 0.86 _±_ 0.21 0.92 _±_ 0.16 0.55 _±_ 0.27 **1.00** _±_ **0.00**
PM Medium Stitch 0.81 _±_ 0.20 0.74 _±_ 0.24 0.67 _±_ 0.28 **1.00** _±_ **0.00**


Table 1: Success rates ( _±_ standard deviation) across different OGBench PointMaze environments.
Best results per environment are shown in bold.


detailed in Table 1. MadDist achieves near-perfect or perfect success rates across all PointMaze
environments, decisively outperforming all baselines. Its performance is particularly noteworthy
in the Stitch environments, which require the model to compose information from disconnected
trajectories, and the large-scale Giant environments, which test the ability to handle long-horizon
tasks. This demonstrates that MadDist not only produces a quantitatively accurate distance function
but also an effective and practical representation for planning.


9


8 CONCLUSION


In this paper, we present two novel algorithms for learning the Minimum Action Distance (MAD)
from state trajectories. We also propose a novel quasimetric for learning asymmetric distance
estimates, and introduce a set of benchmark domains that model several aspects that make distance
learning difficult. In a controlled set of experiments we illustrate that the novel algorithms and
proposed quasimetric outperform state-of-the-art algorithms for learning the MAD.


While this work has concentrated on accurately approximating the MAD as a fundamental stepping
stone, it opens several promising avenues for future research. One of them is the use of MAD estimates
in transfer learning and non-stationary environments, where transition dynamics evolve over time
yet maintain a consistent support. On the same line, MAD can be integrated as a heuristic in search
algorithms, particularly in stochastic domains, to identify the properties that make it a robust and
informative guidance signal under uncertainty. Having established reliable MAD approximation, it can
now be incorporated into downstream tasks, including goal-conditioned planning and reinforcement
learning, to quantify the empirical benefits it brings to complex decision-making problems.


Finally, while MAD can serve as a useful heuristic even in stochastic environments, future work
will explore whether it is possible to recover the Shortest Path Distance (SPD) or identify alternative
quasimetrics that more closely align with it.


10


REFERENCES


Rishabh Agarwal, Marlos C. Machado, Pablo Samuel Castro, and Marc G Bellemare. Contrastive
Behavioral Similarity Embeddings for Generalization in Reinforcement Learning. In _Proceedings_
_of the 9th International Conference on Learning Representations_, 2021.


Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob
McGrew, Josh Tobin, Pieter Abbeel, and Wojciech Zaremba. Hindsight Experience Replay. In
_Advances in Neural Information Processing Systems_, volume 30. Curran Associates, Inc., 2017.


Andre Barreto, Will Dabney, Remi Munos, Jonathan J Hunt, Tom Schaul, Hado P van Hasselt, and
David Silver. Successor Features for Transfer in Reinforcement Learning. In _Advances in Neural_
_Information Processing Systems_, volume 30, pp. 4055–4065. Curran Associates, Inc., 2017.


Richard Bellman. A Markovian Decision Process. _Journal of Mathematics and Mechanics_, 6(5):
679–684, 1957.


Robert Dadashi, Shideh Rezaeifar, Nino Vieillard, Leonard Hussenot, Olivier Pietquin, and Matthieu´
Geist. Offline Reinforcement Learning with Pseudometric Learning. In _Proceedings of the 38th_
_International Conference on Machine Learning_, pp. 2307–2318. PMLR, 2021.


Peter Dayan. Improving Generalization for Temporal Difference Learning: The Successor Representation. _Neural Computation_, 5(4):613–624, 1993.


Benjamin Eysenbach, Tianjun Zhang, Sergey Levine, and Russ R Salakhutdinov. Contrastive learning
as goal-conditioned reinforcement learning. _Advances in Neural Information Processing Systems_,
35:35603–35620, 2022.


Robert W. Floyd. Algorithm 97: Shortest Path. _Communications of the ACM_, 5(6):345, June 1962.
ISSN 0001-0782.


Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4RL: Datasets for Deep
Data-Driven Reinforcement Learning. _arXiv preprint arXiv:2004.07219_, 2020.


Dibya Ghosh, Abhishek Gupta, Ashwin Reddy, Justin Fu, Coline Manon Devin, Benjamin Eysenbach,
and Sergey Levine. Learning to Reach Goals via Iterated Supervised Learning. In _Proceedings of_
_the 8th International Conference on Learning Representations_, 2020.


Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Deep sparse rectifier neural networks. In
_Proceedings_ _of_ _the_ _fourteenth_ _international_ _conference_ _on_ _artificial_ _intelligence_ _and_ _statistics_,
JMLR Workshop and Conference Proceedings, pp. 315–323, 2011.


Kristian Hartikainen, Xinyang Geng, Tuomas Haarnoja, and Sergey Levine. Dynamical Distance
Learning for Semi-Supervised and Unsupervised Skill Discovery. In _Proceedings_ _of_ _the_ _8th_
_International Conference on Learning Representations_, 2020.


Dan Hendrycks and Kevin Gimpel. Gaussian Error Linear Units (GELUs). _arXiv_ _preprint_
_arXiv:1606.08415_, 2016.


Zhengyao Jiang, Tianjun Zhang, Michael Janner, Yueying Li, Tim Rocktaschel,¨ Edward Grefenstette, and Yuandong Tian. Efficient planning in a compact latent action space. _arXiv preprint_
_arXiv:2208.10291_, 2022.


Leslie Pack Kaelbling. Learning to Achieve Goals. _International_ _Joint_ _Conference_ _on_ _Artificial_
_Intelligence_, 2:1094–1098, August 1993.


Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _Proceedings of_
_the 3rd International Conference on Learning Representations_ . PMLR, 2015.


Gunter¨ Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter. Self-normalizing
neural networks. _Advances in neural information processing systems_, 30, 2017.


Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline Reinforcement Learning with Implicit
Q-Learning. In _Proceedings of the 10th International Conference on Learning Representations_,
2022.


11


Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy
Zhang. VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training.
In _Proceedings of the 11th International Conference on Learning Representations_, 2022.


Marlos C. Machado. _Efficient Exploration in Reinforcement Learning through Time-Based Represen-_
_tations_ . PhD thesis, University of Alberta, 2019.


Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare,
Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control
through deep reinforcement learning. _nature_, 518(7540):529–533, 2015.


Vivek Myers, Chongyi Zheng, Anca Dragan, Sergey Levine, and Benjamin Eysenbach. Learning
Temporal Distances: Contrastive Successor Features Can Provide a Metric Structure for DecisionMaking. In _Proceedings of the 41st International Conference on Machine Learning_, pp. 37076–
37096. PMLR, 2024.


Whitney K Newey and James L Powell. Asymmetric Least Squares Estimation and Testing. _Econo-_
_metrica:_ _Journal of the Econometric Society_, pp. 819–847, 1987.


Seohong Park, Dibya Ghosh, Benjamin Eysenbach, and Sergey Levine. HIQL: Offline GoalConditioned RL with Latent States as Actions. In _Advances in Neural Information Processing_
_Systems_, volume 36, pp. 34866–34891. Curran Associates, Inc., 2023.


Seohong Park, Oleh Rybkin, and Sergey Levine. METRA: Scalable Unsupervised RL with MetricAware Abstraction. In _Proceedings of the 12th International Conference on Learning Representa-_
_tions_, 2024a.


Seohong Park, Tobias Kreiman, and Sergey Levine. Foundation Policies with Hilbert Representations.
In _Proceedings_ _of_ _the_ _41st_ _International_ _Conference_ _on_ _Machine_ _Learning_, pp. 39737–39761.
PMLR, 2024b.


Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. Ogbench: Benchmarking
offline goal-conditioned rl. _arXiv preprint arXiv:2410.20092_, 2024c.


Silviu Pitis, Harris Chan, Kiarash Jamali, and Jimmy Ba. An Inductive Bias for Distances: Neural
Nets that Respect the Triangle Inequality. In _Proceedings of the 8th International Conference on_
_Learning Representations_ . Curran Associates, Inc., 2020.


Lorenzo Steccanella and Anders Jonsson. State Representation Learning for Goal-Conditioned
Reinforcement Learning. In _Joint European Conference on Machine Learning and Knowledge_
_Discovery in Databases_, pp. 84–99. Springer, 2022.


Richard S. Sutton and Andrew G. Barto. _Reinforcement_ _learning:_ _An_ _introduction_ . MIT Press,
Cambridge, 1998.


Richard S. Sutton and Andrew G. Barto. _Reinforcement Learning:_ _An Introduction_ . MIT Press,
Cambridge, MA, 2018.


Jean Tarbouriech, Runlong Zhou, Simon S. Du, Matteo Pirotta, Michal Valko, and Alessandro Lazaric.
Stochastic Shortest Path: Minimax, Parameter-Free and Towards Horizon-Free Regret. In _Advances_
_in Neural Information Processing Systems_, volume 34, pp. 6843–6855. Curran Associates, Inc.,
2021.


Hado Van Hasselt, Arthur Guez, and David Silver. Deep Reinforcement Learning with Double
Q-Learning. In _Proceedings of the AAAI conference on artificial intelligence_, volume 30, 2016.


Kaixin Wang, Kuangqi Zhou, Qixin Zhang, Jie Shao, Bryan Hooi, and Jiashi Feng. Towards better
laplacian representation in reinforcement learning with generalized graph drawing. In _Proceedings_
_of the 38th International Conference on Machine Learning_, volume 139, pp. 11003–11012. PMLR,
2021.


Kaixin Wang, Kuangqi Zhou, Jiashi Feng, Bryan Hooi, and Xinchao Wang. Reachability-aware
Laplacian representation in reinforcement learning. In _Proceedings_ _of_ _the_ _40th_ _International_
_Conference on Machine Learning_, volume 202, pp. 36670–36693. PMLR, 2023a.


12


Tongzhou Wang and Phillip Isola. Improved Representation of Asymmetrical Distances with
Interval Quasimetric Embeddings. In _NeurIPS Workshop on Symmetry and Geometry in Neural_
_Representations_ . Curran Associates, Inc., 2022.


Tongzhou Wang, Antonio Torralba, Phillip Isola, and Amy Zhang. Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning. In _Proceedings of the 40th International Conference on_
_Machine Learning_, pp. 36411–36430. PMLR, 2023b.


Stephen Warshall. A Theorem on Boolean Matrices. _Journal of the ACM_, 9(1):11–12, January 1962.
ISSN 0004-5411.


Yifan Wu, George Tucker, and Ofir Nachum. The Laplacian in RL: Learning Representations
with Efficient Approximations. In _Proceedings of the 7th International Conference on Learning_
_Representations_, 2019.


Sungwook Yoon, Alan Fern, and Robert Givan. FF-Replan: A baseline for probabilistic planning. In
_Proceedings of the 5th International Conference on Automated Planning and Scheduling (ICAPS)_,
pp. 352–359, 2007.


A PROOF OF UNIQUENESS FOR THE MAD OPTIMIZATION PROBLEM


We begin by formally defining the Minimum Action Distance (MAD) in terms of policies and first
passage times within a Markov Decision Process (MDP).
**Definition 1** (Minimum Action Distance) **.** _Let T_ ( _sj_ _| π, si_ ) _be the random variable for the first time_
_step at which state sj_ _is reached when starting in state si_ _and following policy π._ _The support of this_
_random variable, denoted supp_ ( _T_ ( _sj_ _| π, si_ )) _, is the set of all possible first passage times that occur_
_with non-zero probability._ _The Minimum Action Distance dMAD_ : _S × S_ _→_ N _∪{∞} is defined as:_

_dMAD_ ( _si, sj_ ) := min _[|][ π, s][i]_ [))]] _[ .]_
_π_ [min [] _[supp]_ [(] _[T]_ [(] _[s][j]_


This definition finds the length of the shortest possible trajectory from _si_ to _sj_ . The inner minimum,
min[supp( _·_ )], identifies the shortest-in-time realization possible under a fixed policy _π_ . The outer
minimum, min _π_, then finds the policy that makes this shortest possible realization as short as
possible. Note that if the process starts in the target state _s_ 0 = _sj_, the first passage time is zero, i.e.,
0 _∈_ supp( _T_ ( _sj_ _| π, sj_ )).


**Equivalence to Graph Shortest Path** Let _G_ = ( _S, R_ ) be the state-transition graph where an edge
( _s, s_ _[′]_ ) _∈_ _R_ exists if and only if there is an action _a_ with _P_ ( _s_ _[′]_ _|s, a_ ) _>_ 0. A path of length _k_ from _si_ to
_sj_ in _G_ corresponds to a sequence of actions that can transition between these states with non-zero
probability. We can always construct a policy _π_ that executes this specific sequence. Therefore,
minimizing over all policies is equivalent to finding the length of the shortest path between nodes _si_
and _sj_ in the graph _G_ . This equivalence allows us to leverage the properties of shortest path distances
in the proof below.

**Theorem 1.** _The Minimum Action Distance, dMAD, as defined above, is the unique solution to the_
_constrained optimization problem:_


_subject to_ _d_ ( _s, s_ ) = 0 _∀s ∈S_ (C1)

_d_ ( _s, s_ _[′]_ ) _≤_ 1 _∀_ ( _s, s_ _[′]_ ) _∈_ _R_ (C2)

_d_ ( _s, s_ _[′]_ ) _≤_ _d_ ( _s, s_ _[′′]_ ) + _d_ ( _s_ _[′′]_ _, s_ _[′]_ ) _∀_ ( _s, s_ _[′]_ _, s_ _[′′]_ ) _∈S_ [3] (C3)


_Proof._ The proof is structured in two parts. First, we show that _d_ MAD is a feasible solution. Second,
we show that for any other feasible solution _d_, we must have _d_ ( _s, s_ _[′]_ ) _≤_ _d_ MAD( _s, s_ _[′]_ ), establishing
both optimality and uniqueness.


**Part 1:** **Feasibility of** _d_ **MAD**


Using the shortest path interpretation of _d_ MAD, we verify that it satisfies each constraint.


13


_maximize_
_d_


 - _d_ ( _s, s_ _[′]_ )


( _s,s_ _[′]_ ) _∈S_ [2]


- **Constraint (C1) - Identity:** The shortest path from any state _s_ to itself is the empty path of
length 0. Thus, _d_ MAD( _s, s_ ) = 0.


    - **Constraint (C2) - One-Step Reachability:** If ( _s, s_ _[′]_ ) _∈_ _R_, there exists a direct edge from _s_
to _s_ _[′]_ in _G_ . This corresponds to a path of length 1. The shortest path, _d_ MAD( _s, s_ _[′]_ ), cannot be
longer than this path, so _d_ MAD( _s, s_ _[′]_ ) _≤_ 1.


    - **Constraint (C3) - Triangle Inequality:** This is a fundamental property of shortest paths.
The shortest path from _s_ to _s_ _[′]_ is, by definition, no longer than the path formed by concatenating the shortest path from _s_ to an intermediate state _s_ _[′′]_ and the shortest path from _s_ _[′′]_ to
_s_ _[′]_ . This directly gives the inequality _d_ MAD( _s, s_ _[′]_ ) _≤_ _d_ MAD( _s, s_ _[′′]_ ) + _d_ MAD( _s_ _[′′]_ _, s_ _[′]_ ).


As _d_ MAD satisfies all constraints, it is a feasible solution.


**Part 2:** **Optimality and Uniqueness of** _d_ **MAD**


Let _d_ be an arbitrary feasible solution satisfying (C1), (C2), and (C3). We show by induction on the
shortest path length _k_ = _d_ MAD( _s, s_ _[′]_ ) that _d_ ( _s, s_ _[′]_ ) _≤_ _d_ MAD( _s, s_ _[′]_ ).


    - **Base** **Case** **(** _k_ = 0 **):** If _d_ MAD( _s, s_ _[′]_ ) = 0, then _s_ = _s_ _[′]_ . By constraint (C1), any feasible
solution _d_ must satisfy _d_ ( _s, s_ ) = 0. Thus, _d_ ( _s, s_ _[′]_ ) = 0 = _d_ MAD( _s, s_ _[′]_ ).


    - **Inductive** **Hypothesis:** Assume for some integer _k_ _≥_ 0 that for all pairs ( _s, s_ _[′]_ ) with
_d_ MAD( _s, s_ _[′]_ ) _≤_ _k_, the inequality _d_ ( _s, s_ _[′]_ ) _≤_ _d_ MAD( _s, s_ _[′]_ ) holds.


    - **Inductive** **Step:** Consider a pair ( _s, s_ _[′]_ ) with _d_ MAD( _s, s_ _[′]_ ) = _k_ + 1. By the shortest path
definition, there must exist a predecessor state _s_ _[′′]_ on a shortest path from _s_ to _s_ _[′]_ such that
( _s_ _[′′]_ _, s_ _[′]_ ) _∈_ _R_ and _d_ MAD( _s, s_ _[′′]_ ) = _k_ .


Applying the constraints on _d_ :
_d_ ( _s, s_ _[′]_ ) _≤_ _d_ ( _s, s_ _[′′]_ ) + _d_ ( _s_ _[′′]_ _, s_ _[′]_ ) by (C3), the triangle inequality

_≤_ _d_ MAD( _s, s_ _[′′]_ ) + _d_ ( _s_ _[′′]_ _, s_ _[′]_ ) by Inductive Hypothesis, since _d_ MAD( _s, s_ _[′′]_ ) = _k_

_≤_ _k_ + 1 by (C2), since ( _s_ _[′′]_ _, s_ _[′]_ ) _∈_ _R_
Since _d_ MAD( _s, s_ _[′]_ ) = _k_ + 1, we have shown that _d_ ( _s, s_ _[′]_ ) _≤_ _d_ MAD( _s, s_ _[′]_ ).


By induction, we have established that for any feasible solution _d_, the inequality _d_ ( _s, s_ _[′]_ ) _≤_
_d_ MAD( _s, s_ _[′]_ ) holds for all pairs ( _s, s_ _[′]_ ) _∈S_ [2] .


    - **Optimality:** The objective is to maximize the sum [�] ( _s,s_ _[′]_ ) _∈S_ [2] _[ d]_ [(] _[s, s][′]_ [)][.] [Since] [we] [have]

shown that every term _d_ ( _s, s_ _[′]_ ) is less then or equal to the corresponding term _d_ MAD( _s, s_ _[′]_ ),
the total sum for any feasible solution _d_ cannot exceed the sum for _d_ MAD:


**–** From the induction proof we know that _d_ _[∗]_ ( _s, s_ _[′]_ ) _≤_ _d_ MAD( _s, s_ _[′]_ ) for every single pair
( _s, s_ _[′]_ ).


Therefore _d_ _[∗]_ ( _s, s_ _[′]_ ) = _d_ MAD( _s, s_ _[′]_ ) _∀_ ( _s, s_ _[′]_ ) _∈S_ [2] .


14


 - _d_ ( _s, s_ _[′]_ ) _≤_ 

( _s,s_ _[′]_ ) _∈S_ [2] ( _s,s_ _[′]_ )


 - _d_ MAD( _s, s_ _[′]_ )


( _s,s_ _[′]_ ) _∈S_ [2]


Since _d_ MAD is itself a feasible solution, it achieves the maximum possible value, proving it
is an optimal solution.


- **Uniqueness:** Let’s assume _d_ _[∗]_ is another solution that is also optimal.


**–** For _d_ _[∗]_ to be optimal, its total sum must equal the maximum possible sum:

    - _[∗]_ _[′]_    - _[′]_


 - _d_ _[∗]_ ( _s, s_ _[′]_ ) = 

( _s,s_ _[′]_ ) _∈S_ [2] ( _s,s_ _[′]_ )


 - _d_ MAD( _s, s_ _[′]_ )


( _s,s_ _[′]_ ) _∈S_ [2]


B QUASIMETRIC CONSTRUCTIONS VIA RELU REDUCTION


Let _x, y_ _∈_ R _[d]_ . We begin by defining a ReLU-based coordinate reduction, then derive scalar
quasimetrics through several aggregation operators, and finally state general results for convex
combinations.


B.1 COORDINATEWISE RELU REDUCTION


**Definition 2** (ReLU Reduction) **.** _Define the map r_ : R _[d]_ _×_ R _[d]_ _→_ R _[d]_ _by_


_r_ ( _x, y_ ) = relu( _x −_ _y_ ) _,_ _ri_ ( _x, y_ ) = max� _xi −_ _yi,_ 0� _,_ _i_ = 1 _, . . ., d._


**Proposition 1.** _For all x, y, z_ _∈_ R _[d]_ _and λ >_ 0 _, each coordinate ri_ _satisfies:_


_(a)_ Nonnegativity and identity: _ri_ ( _x, y_ ) _≥_ 0 _and ri_ ( _x, x_ ) = 0 _._


_(b)_ Asymmetry: _ri_ ( _x, y_ ) _̸_ = _ri_ ( _y, x_ ) _unless xi_ = _yi._


_(c)_ Triangle inequality: _ri_ ( _x, y_ ) _≤_ _ri_ ( _x, z_ ) + _ri_ ( _z, y_ ) _._


_(d)_ Positive homogeneity: _ri_ ( _λx, λy_ ) = _λ ri_ ( _x, y_ ) _._


_Proof._ (a) and (b) follow directly from the definition of the max operation.


(c) Observe that


_ri_ ( _x, y_ ) = max( _xi −_ _yi,_ 0) = max�( _xi −_ _zi_ ) + ( _zi −_ _yi_ ) _,_ 0�

_≤_ max( _xi −_ _zi,_ 0) + max( _zi −_ _yi,_ 0) = _ri_ ( _x, z_ ) + _ri_ ( _z, y_ ) _._


(d) Linearity of scalar multiplication inside the max gives


_ri_ ( _λx, λy_ ) = max( _λxi −_ _λyi,_ 0) = _λ_ max( _xi −_ _yi,_ 0) = _λri_ ( _x, y_ ) _._


This concludes the proof.


B.2 SCALAR QUASIMETRICS VIA AGGREGATION


We now obtain real-valued quasimetrics by aggregating the vector _r_ ( _x, y_ ).


**Definition 3** (Max Reduction) **.**


_d_ max( _x, y_ ) = max
1 _≤i≤d_ _[r][i]_ [(] _[x, y]_ [)] _[.]_


**Definition 4** (Sum and Mean Reductions) **.**


**Proposition 2.** _Each of d_ max _, d_ sum _, and d_ mean _satisfies for all x, y, z_ _∈_ R _[d]_ _and λ >_ 0 _:_


_(a)_ Triangle inequality: _d_ ( _x, y_ ) _≤_ _d_ ( _x, z_ ) + _d_ ( _z, y_ ) _._


_(b)_ Positive homogeneity: _d_ ( _λx, λy_ ) = _λ d_ ( _x, y_ ) _._


_Proof._ (a) follows by combining coordinate-wise triangle bounds with either:


    - _d_ max : max _i_ [ _ai_ + _bi_ ] _≤_ max _i ai_ + max _i bi_,


    - _d_ sum and _d_ mean: term-wise summation.


(b) is immediate from the linearity of scalar multiplication and properties of max/sum.


15


_d_ sum( _x, y_ ) =


_d_


- _ri_ ( _x, y_ ) _,_ _d_ mean( _x, y_ ) = _d_ [1]

_i_ =1


_d_


_d_

- _ri_ ( _x, y_ ) _._


_i_ =1


B.3 CONVEX COMBINATIONS OF QUASIMETRICS


More generally, let _d_ 1 _, . . ., dn_ be any quasimetrics on R _[d]_ each obeying the triangle inequality and
positive homogeneity. For weights _α_ 1 _, . . ., αn_ _≥_ 0 with [�] _k_ _[α][k]_ [= 1][, define]


subject to _d_ ( _s, s_ ) = 0 _∀s ∈S_ (Constraint 1: Identity)

_d_ ( _s, s_ _[′]_ ) _≤_ 1 _∀_ ( _s, s_ _[′]_ ) _∈_ _R_ (Constraint 2: One-Step)

_d_ ( _s, s_ _[′]_ ) _≤_ _d_ ( _s, s_ _[′′]_ ) + _d_ ( _s_ _[′′]_ _, s_ _[′]_ ) _∀_ ( _s, s_ _[′]_ _, s_ _[′′]_ ) _∈S_ [3] (Constraint 3: Triangle Inequality)


Although the formulation uses a global maximization, the solution corresponds exactly to the
*minimum* number of actions required to transition between states. The constraints enforce the
correct dynamical structure:


    - The one-step constraint enforces that any directly connected states must lie within distance
1, effectively _pulling_ them close.

    - The triangle inequality propagates this local structure globally, ensuring consistency across
multi-step paths.

    - The maximization objective _pushes_ all state pairs as far apart as the constraints allow,
yielding distances that exactly match the smallest number of steps connecting them.


This formulation is computationally intractable for large or continuous state spaces, primarily due to
the triangle inequality (Constraint 3), which must hold for all triplets of states.


C.2 SIMPLIFICATION VIA QUASIMETRIC EMBEDDINGS


To make this problem tractable, we enforce the triangle inequality _by construction_ rather than as an
explicit constraint. We achieve this by learning a state embedding function _ϕ_ : _S_ _→_ R _[k]_ and defining
the distance between any two states _s, s_ _[′]_ using a **quasimetric** function _dq_ on their embeddings:
_dϕ_ ( _s, s_ _[′]_ ) := _dq_ ( _ϕ_ ( _s_ ) _, ϕ_ ( _s_ _[′]_ ))
A quasimetric function _dq_ ( _x, y_ ) satisfies the following properties by definition:


16


_d_ conv( _x, y_ ) =


_n_

- _αk dk_ ( _x, y_ ) _._


_k_ =1


**Proposition 3.** _d_ conv _is a quasimetric satisfying:_


_(a)_ Triangle inequality: _d_ conv( _x, y_ ) _≤_ _d_ conv( _x, z_ ) + _d_ conv( _z, y_ ) _._


_(b)_ Positive homogeneity: _d_ conv( _λx, λy_ ) = _λ d_ conv( _x, y_ ) _._


_Proof._ Linearity of the weighted sum together with the corresponding property for each _dk_ yields
(a)–(b).


C DERIVATION OF LEARNING OBJECTIVES FOR MINIMUM ACTION DISTANCE


This appendix details the derivation of the _MadDist_ and _TDMadDist_ loss functions. The derivation
begins with the foundational, but computationally intractable, constrained optimization problem for
the Minimum Action Distance (MAD) and systematically transforms it into a pair of scalable learning
objectives.


C.1 CONSTRAINED OPTIMIZATION PROBLEM FOR MAD


The Minimum Action Distance, _d_ MAD, is the solution to the following constrained optimization
problem. This formulation seeks a distance function that maximizes the sum of all pairwise distances
while remaining consistent with the environment’s one-step transition dynamics.


maximize
_d_


 - _d_ ( _s, s_ _[′]_ ) (Objective 1)


( _s,s_ _[′]_ ) _∈S_ [2]


1. **Identity:** _dq_ ( _x, x_ ) = 0


2. **Non-negativity:** _dq_ ( _x, y_ ) _≥_ 0


3. **Triangle Inequality:** _dq_ ( _x, z_ ) _≤_ _dq_ ( _x, y_ ) + _dq_ ( _y, z_ )


By defining _dϕ_ as a quasimetric over the embedding space, the identity (Constraint 1) and triangle
inequality (Constraint 3) properties are satisfied for any choice of embedding function _ϕ_ . This
simplification is crucial, as it removes the most computationally expensive constraint and leaves us
with a more manageable learning problem:


where _SD_ is the set of all states appearing in the dataset _D_ . Minimizing _Lr_ incentivizes _dϕ_ ( _s, s_ _[′]_ ) for
random pairs to approach a large value _d_ max, again serving the **maximize** objective.


**Term** **3:** **The** **Constraint** **Loss** **(** _Lc_ **).** While _Lo_ encourages matching the upper bound, it does
not strictly enforce the inequality. We add an explicit penalty term that penalizes violations of the
trajectory upper bound.


_Lc_ = E( _si,sj_ ) _∼D<Hc_ �relu( _dϕ_ ( _si, sj_ ) _−_ ( _j −_ _i_ )) [2][�]


This term enforces the constraint _dϕ_ ( _si, sj_ ) _≤_ _j_ _−_ _i_, which is a generalization of the one-step
constraint ( _dϕ_ ( _s, s_ _[′]_ ) _≤_ 1). The learning process finds an equilibrium where the objective terms
( _Lo, Lr_ ) encourage larger distances, while this constraint term ( _Lc_ ) and the implicit triangle inequality
provide regularization.


17


maximize
_ϕ_


 - _dq_ ( _ϕ_ ( _s_ ) _, ϕ_ ( _s_ _[′]_ ))


( _s,s_ _[′]_ ) _∈S_ [2]


subject to _dq_ ( _ϕ_ ( _s_ ) _, ϕ_ ( _s_ _[′]_ )) _≤_ 1 _∀_ ( _s, s_ _[′]_ ) _∈_ _R_ (Constraint 2: One-Step)


C.3 THE _MadDist_ LOSS FUNCTION


We now translate this simplified problem into a loss function suitable for minimization via gradient
descent. Given a dataset of state trajectories _D_ = _{_ ( _s_ 0 _, s_ 1 _, . . ., sn_ ) _, . . . }_, the path length _j −_ _i_ for
any pair of states ( _si, sj_ ) on a trajectory with _i < j_ provides a valid upper bound on the true MAD,
i.e., _d_ MAD( _si, sj_ ) _≤_ _j −_ _i_ .


The _MadDist_ loss, _L_ = _Lo_ + _wrLr_ + _wcLc_, is composed of three terms, each corresponding to a
component of the optimization problem.


**Term 1:** **The Objective Loss (** _Lo_ **).** The original goal is to maximize all pairwise distances. As
a practical proxy, we formulate a loss term that is minimized when the learned distance _dϕ_ ( _si, sj_ )
matches its trajectory-based upper bound, _j −_ _i_ . This encourages the learned distances to increase,
directly addressing the maximization objective by using information from the dataset. We use a
scale-invariant squared error to prevent long-horizon pairs from dominating the loss.


_Lo_ = E( _si,sj_ ) _∼D_


�� �2 [�]
_dϕ_ ( _si, sj_ ) _−_ 1
_j −_ _i_


Minimizing _Lo_ encourages _dϕ_ ( _si, sj_ ) _→_ _j −_ _i_, serving as a proxy for the **maximize** objective.


**Term 2:** **The Contrastive Loss (** _Lr_ **).** To further support the global maximization objective, we
introduce a contrastive term. We sample random pairs of states ( _s, s_ _[′]_ ) from the dataset and penalize
them for having a small distance. This encourages all states to be far apart, which aligns with the
goal of maximizing the sum of all distances, especially for pairs not on the same trajectory.


 relu 1 _−_ _[d][ϕ]_ [(] _[s, s][′]_ [)]

_d_ max


�2 [�]


_Lr_ = E( _s,s′_ ) _∼SD_


C.4 TEMPORAL DIFFERENCE BOOTSTRAPPING ( _TDMadDist_ )


_TDMadDist_ integrates principles from Temporal Difference (TD) learning. Instead of relying solely
on the data-driven target _j_ _−_ _i_, it uses the model’s own predictions to form a potentially tighter,
more informed target. From the Bellman equation for shortest paths, we have _d_ MAD( _si, sj_ ) =
1 + _d_ MAD( _si_ +1 _, sj_ ). We can therefore use the bootstrapped value 1 + _dϕ′_ ( _si_ +1 _, sj_ ) using a stable
target network _ϕ_ _[′]_ as the new target for our objective.


The objective terms are modified as follows:


**The TD Main Objective (** _L_ _[′]_ _o_ **[).]** The target for _dϕ_ ( _si, sj_ ) becomes the minimum of the trajectory
upper bound and the bootstrapped target.


In this section, we describe the implementation details of each algorithm included in our evaluation.


D.1 COMPUTER RESOURCES


We run all experiments on a single NVIDIA RTX 4070 GPU with 8GB of VRAM and an Intel
i7-4700-HX with 32GB of RAM. We will provide the code for all experiments upon acceptance of
the paper.


D.2 MADDIST


To train the MadDist distance models, we used the Adam optimizer with a learning rate of 1 _×_ 10 _[−]_ [4],
a batch size of 256 for the objective ( _Lo_, _Lr_ ), and a separate batch of size 1024 for the constraint loss
( _Lc_ ). For our main experiment, we used the novel simple quasimetric function and a latent dimension
size of 512. We include an ablation over different quasimetric functions and latent dimension sizes in
Appendix E.


The full set of hyperparameter values used to train the MadDist models can be found in Table 2.


D.3 TDMADDIST


To train the TDMadDist distance models, we used the the Adam optimizer with a learning rate of
1 _×_ 10 _[−]_ [4], a batch size of 256 for the objective ( _Lo_, _Lr_ ), and a separate batch of size 1024 for the
constraint loss ( _Lc_ ). For our main experiment, we used the novel simple quasimetric function and a
latent dimension size of 512. We include an ablation over different quasimetric functions and latent
dimension sizes in Appendix E.


For TDMadDist, we remove the hyperparameter _d_ max from the MadDist algorithm, because it is
not included in TDMadDist’s objective ( _Lr_ ). The temporal-difference update used when training
the TDMadDist distance models involves the use of a target network, _dθ′_, which is updated using a
Polyak averaging factor _τ_ = 0 _._ 005.


The full set of hyperparameter values used to train the TDMadDist models can be found in Table 3.


18


_L_ _[′]_ _o_ [=][ E] ( _si,sj_ ) _∼D_


�� _dϕ_ ( _si, sj_ ) �2 [�]
min( _j −_ _i,_ 1 + _dϕ′_ ( _si_ +1 _, sj_ )) _[−]_ [1]


Minimizing this loss still serves the **maximize** objective, but now encourages distances toward a
dynamically updated target.


**The TD Contrastive Objective (** _L_ _[′]_ _r_ **[).]** The contrastive term is modified to be consistent with the
one-step Bellman logic, using a bootstrapped target.


_L_ _[′]_ _r_ [=][ E] ( _si,sr_ ) _∼D_


The constraint loss _Lc_ remains unchanged.


D IMPLEMENTATION DETAILS


�� �2 [�]
_dϕ_ ( _si, sr_ )
1 + _dϕ′_ ( _si_ +1 _, sr_ ) _[−]_ [1]


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


Table 2: Hyperparameters used to train the MadDist algorithm.


**Hyperparameter** **Value**
Quasimetric Function _dsimple_
Optimizer Adam Kingma & Ba (2015)
Learning Rate 1 _×_ 10 _[−]_ [4]
Batch Size ( _Lo_, _Lr_ ) 256
Batch Size ( _Lc_ ) 1024
Activation Function (Hidden Layers) SELU Klambauer et al. (2017)
Neural Network (512, 512, 256, 128)
_wr_ 1, 10
_wc_ 0.1
_dmax_ 100, 500
_Hc_ 6


Table 3: Hyperparameters used to train the TDMadDist algorithm.


**Hyperparameter** **Value**
Quasimetric Function _dsimple_
Optimizer Adam Kingma & Ba (2015)
Learning Rate 1 _×_ 10 _[−]_ [4]
Batch Size ( _Lo_, _Lr_ ) 256
Batch Size ( _Lc_ ) 1024
Activation Function (Hidden Layers) SELU Klambauer et al. (2017)
Neural Network (512, 512, 256, 128)
_wr_ 1
_wc_ 0.1
_Hc_ 6
_τ_ 0.005


D.4 QRL


We trained QRL distance models following the approach of Wang et al. (2023b). We used the
Lagrangian formulation


                 -                 min max _θ_ ( _s, s_ _[′]_ ))] + _λ_ E( _s,s′_ ) _∼p_ transition[relu( _d_ [IQE] _θ_ ( _s, s_ _[′]_ ) + 1) [2] ] _,_ (12)
_θ_ _λ≥_ 0 _[−]_ [E] _[s,s][′][∼][S][D]_ [[] _[ϕ]_ [(] _[d]_ [IQE]


where _ϕ_ ( _x_ ) ≜ _−_ softplus( _α −_ _x, β_ ) and _d_ [IQE] _θ_ ( _s, s_ _[′]_ ) is the IQE distance between states _s_ and _s_ _[′]_ .
Following Wang et al. (2023b), we set ( _α, β_ ) = (15 _,_ 0 _._ 1) for short-horizon environments and
( _α, β_ ) = (500 _,_ 0 _._ 01) for long-horizon environments. The first term in the objective maximizes the
expected distance between states sampled from the dataset, while the second term penalizes distances
between state–next-state pairs ( _s, s_ _[′]_ ) observed in the data.


Through our experiments, we observed that setting the softplus offset to 15 and the steepness to
0 _._ 1, as suggested for short-horizon environments by Wang et al. (2023b), led to better performance
overall.


For the neural network architecture, we used a multi-layer perceptron with an overall layer structure
of _x_ - 512 - 512 - 128 (where _x_ is the input observation dimension). Its two hidden layers (each of size
512) use ReLU activations, as described for state-based observations environments (i.e., environments
with real vector observations, as opposed to images or other high-dimensional inputs) in the original
paper. For the distance function, the resulting 128-dimensional MLP output is fed into a separate
128-512-2048 projector, followed by an IQE-maxmean head with 64 components each of size 32.


The full set of hyperparameter values used to train the QRL distance models can be found in Table 4.


19


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


Table 4: Hyperparameters used to train the QRL model.


**Hyperparameter** **Value**
Neural Network State embedding _x_        - 512 - 512 - 128
Neural Network IQE Projector 128-512-2048
Activation Function (Hidden Layers) ReLU Glorot et al. (2011)
Optimizer Adam Kingma & Ba (2015)
_λ_ Learning Rate 0 _._ 01
Learning Rate Model 1 _×_ 10 _[−]_ [4]
Batch Size 256
Quasimetric function IQE
IQE n components 64
IQE Reduction maxmean


D.5 HILBERT REPRESENTATION


A Hilbert representation model is a function _ϕ_ : _S_ _→_ R _[d]_ that embeds a state _s_ _∈S_ into a _d_ dimensional space, such that the Euclidean distance between embedded states approximates the
number of actions required to transition between them under the optimal policy.


We trained Hilbert representation models following the approach of Park et al. (2024b), using action-free Implicit Q-Learning (IQL) (Park et al., 2023) and Hindsight Experience Replay (HER) (Andrychowicz et al., 2017).


We used a dataset of state–next-state pairs ( _s, s_ _[′]_ ), which we relabeled using HER to produce state–
next-state–goal tuples ( _s, s_ _[′]_ _, g_ ). Goals were sampled from a geometric distribution Geom( _γ_ ) over
future states in the same trajectory with probability 0 _._ 625, and uniformly from the entire dataset with
probability 0 _._ 375.


We trained the Hilbert representation model _ϕ_ to minimize the temporal-difference loss


E[ _lτ_ ( _−_ **1** ( _s ̸_ = _g_ ) _−_ _γ||ϕ_ ( _s_ _[′]_ ) _−_ _ϕ_ ( _g_ ) _||_ + _||ϕ_ ( _s_ ) _−_ _ϕ_ ( _g_ ) _||_ )] _,_ (13)


where _lτ_ denotes the expectile loss (Newey & Powell, 1987), an asymmetric loss function that
approximates the max operator in the Bellman backup (Kostrikov et al., 2022). This objective
naturally supports the use of target networks (Mnih et al., 2015) and double estimators (Van Hasselt
et al., 2016) to improve learning stability. We included both in our implementation, following the
original setup used by Park et al. (2024b).


The full set of hyperparameter values used to train the Hilbert models can be found in Table 5.


Table 5: Hyperparameters used to train the Hilbert representation models.


**Hyperparameter** **Value**
Latent Dimension 32
Expectile 0 _._ 9
Discount Factor 0 _._ 99
Learning Rate 0 _._ 0003
Target Network Smoothing Factor 0 _._ 005
Multi-Layer Perceptron Dimensions (512, 512) Fully-Connected Layers
Activation Function (Hidden Layers) GELU (Hendrycks & Gimpel, 2016)
Layer Normalization (Hidden Layers) True
Activation Function (Final Layer) Identity
Layer Normalization (Final Layer) False
Optimizer Adam (Kingma & Ba, 2015)
Batch Size 1024


20


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


E ABLATION STUDY


In this section, we present additional ablation studies to analyze the performance of our proposed
methods. We evaluate the impact of different hyperparameters and design choices on the performance
of the learned embeddings.


We conduct experiments in the CliffWalking environment, which is a highly asymmetric environment
with a known ground truth MAD. For each experiment we train the _MadDist_ algorithm using the
same hyperparameters from the main experiments, varying only the hyperparameter of interest while
keeping all others fixed. We then evaluate the learned embeddings using Spearman correlation,
Pearson correlation, and Ratio CV metrics.


E.1 EFFECT OF LATENT DIMENSION ON MAD ACCURACY


Figure 4: Impact of latent size on Spearman correlation, Pearson correlation and Ratio CV of the
MadDist and TDMadDist algorithms, evaluated in the CliffWalking environment. Shaded regions
show the range of values across five random seeds, with upper and lower boundaries representing
maximum and minimum values.


Figure 4 shows the impact of the latent dimension size on the performance of our proposed methods.
We can see that increasing the latent dimension size improves the performance of our methods.
We note that the performance starts to saturate after a latent dimension size of 10, but larger latent
dimension sizes still slightly improve the performance and do not harm the performance. This is
likely due to the fact that larger latent dimension sizes allow for more expressive representations,
which can help to better capture the underlying structure of the environment.


E.2 EFFECT OF QUASIMETRIC CHOICE ON MAD ACCURACY


outperforming both the Wide Norm (MadDistance-WideNorm) and IQE (MadDistance-IQE) variants.
While Wide Norm and IQE perform similarly to each other, they consistently underperform the
simple quasimetric across all three evaluation metrics.


Figure 6 presents the same ablation over quasimetric functions, now applied to learning the TDMadDist model. The results mirror the previous setting: the simple quasimetric (TDMadDist-Simple)
again achieves the strongest performance, while the Wide Norm (TDMadDist-WideNorm) and IQE
(TDMadDist-IQE) variants lag slightly behind and show comparable results to each other.


21


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


In this experiment, we used a latent dimension size of 256. For the Wide Norm quasimetric, we
configure the model with 32 components, each having an output component size of 32. For the IQE
quasimetric, we set each component to have a dimensionality of 16. For both quasimetric functions
we use maxmean reduction (Pitis et al., 2020).


E.3 EFFECT OF DATASET SIZE ON MAD ACCURACY


Figure 7: Impact of dataset size on Spearman correlation, Pearson correlation and Ratio CV of the
MadDist and TDMadDist algorithms, evaluated in the CliffWalking environment. Shaded regions
show the range of values across five random seeds, with upper and lower boundaries representing
maximum and minimum values.


Figure 7 illustrates how dataset size affects the performance of our proposed methods. As the number
of trajectories increases, the dataset provides broader coverage of all the possible transitions in the
environment, leading to a more accurate approximation of the MAD.


E.4 NEURAL NETWORK SIZE CHOICE FOR QRL AND HILBERT


In this section, we present ablation studies examining how the size of the neural network affects
performance for both QRL and Hilbert. For QRL, we evaluate three architectures, each consisting of
an embedding network followed by a projection network used with the IQE quasimetric as described
in Wang et al. (2023b):


    - QRL ~~n~~ n ~~1~~ : (512, 512, 128) embedding + (128, 512, 2048) projection.

    - QRL ~~n~~ n ~~2~~ : (512, 512, 256, 128) embedding + (128, 512, 2048) projection.

    - QRL ~~n~~ n ~~3~~ : (1024, 1024, 256) embedding + (1024, 1024, 1024, 2048) projection.


QRL ~~n~~ n ~~1~~ corresponds to the architecture used for state-based observations in Wang et al. (2023b),
while QRL ~~n~~ n ~~2~~ shares the same embedding network as MAD and TDMAD. QRL ~~n~~ n ~~3~~ represents
the larger architecture considered in Wang et al. (2023b). As shown in Figure 8, performance
differences across these architectures are minor, with QRL ~~n~~ n ~~1~~ achieving the best results.


For the Hilbert algorithm, we compare two fully connected architectures:


    - HILBERT ~~n~~ n ~~1~~ : (512, 512), as used in the original paper Park et al. (2024b).

    - HILBERT ~~n~~ n ~~2~~ : (512, 512, 256, 128), matching the architecture used for MAD and TDMAD.


22


**1188**

**1189**


**1190**

**1191**

**1192**

**1193**

**1194**

**1195**


**1196**

**1197**

**1198**

**1199**

**1200**

**1201**


**1202**

**1203**

**1204**

**1205**

**1206**


**1207**

**1208**

**1209**

**1210**

**1211**

**1212**


**1213**

**1214**

**1215**

**1216**

**1217**

**1218**


**1219**

**1220**

**1221**

**1222**

**1223**

**1224**


**1225**

**1226**

**1227**

**1228**

**1229**


**1230**

**1231**

**1232**

**1233**

**1234**

**1235**


**1236**

**1237**

**1238**

**1239**

**1240**

**1241**


|gure 8: Impact o<br>the QRL algor<br>ee random seed<br>shown in Figur|Col2|Col3|
|---|---|---|
|gure 8: Impact o<br> the QRL algor<br>ee random seed<br> shown in Figur|||
|gure 8: Impact o<br> the QRL algor<br>ee random seed<br> shown in Figur|||
|gure 8: Impact o<br> the QRL algor<br>ee random seed<br> shown in Figur||f neural netw<br>  ithm in the C<br> s, with upper<br>  e 9, HILBER|
|gure 8: Impact o<br> the QRL algor<br>ee random seed<br> shown in Figur|||


|size on Spearm<br>Walking enviro<br>lower bounds c<br>n 1 performs at|Col2|
|---|---|
|size on Spearm<br>   Walking enviro<br>   lower bounds c<br>n ~~1~~ performs at<br>||
|size on Spearm<br>   Walking enviro<br>   lower bounds c<br>n ~~1~~ performs at<br>||
|size on Spearm<br>   Walking enviro<br>   lower bounds c<br>n ~~1~~ performs at<br>||
|size on Spearm<br>   Walking enviro<br>   lower bounds c<br>n ~~1~~ performs at<br>|an correlation<br>    nment. Shade<br>    orresponding<br>  best in our ev|
|size on Spearm<br>   Walking enviro<br>   lower bounds c<br>n ~~1~~ performs at<br>||


|maximum and m<br>uation.|inimum valu|
|---|---|
|<br>     maximum and m<br>   uation.||


for the Hilbert algorithm in the CliffWalking environment. Shaded regions indicate variation across
three random seeds.


F COMPLETE LIST OF RESULTS


In this section, we report the complete list of results, including the Spearman and Pearson Correlation
metrics together with the Ratio Coefficient of Variation. The results appear in Figure 12 and in
Figure 12.


F.1 QUALITATIVE EVALUATION


In this section, we present a qualitative evaluation of the MadDist algorithm within the MediumMaze
environment. We visualize the learned geometry of the state space to demonstrate how the metric
captures the underlying structure of the maze.


G ENVIRONMENTS


Our test environments were specifically chosen to span a comprehensive range of reward-free MDP
characteristics and challenges, ensuring a thorough evaluation. Key design considerations for this
suite include:


 - _Noisy Observations:_ To assess robustness to imperfect state information, which can challenge
algorithms relying on precise state identification.


 - _Stochastic Dynamics:_ To evaluate if our algorithm can retrieve the MAD even when transitions
are not deterministic. This reflects real-world scenarios where environments have inherent
randomness or agent actions have uncertain outcomes.


 - _Asymmetric:_ To test the capability of our algorithm to learn true quasimetric distances that capture
directional dependencies (e.g., one-way paths, key-door mechanisms).


 - State Spaces:


**–** _Continuous State Spaces:_ To demonstrate applicability to problems with real-valued state
representations where function approximation is essential.


23


**1242**

**1243**


**1244**

**1245**

**1246**

**1247**

**1248**

**1249**


**1250**

**1251**

**1252**

**1253**

**1254**

**1255**


**1256**

**1257**

**1258**

**1259**

**1260**


**1261**

**1262**

**1263**

**1264**

**1265**

**1266**


**1267**

**1268**

**1269**

**1270**

**1271**

**1272**


**1273**

**1274**

**1275**

**1276**

**1277**

**1278**


**1279**

**1280**

**1281**

**1282**

**1283**


**1284**

**1285**

**1286**

**1287**

**1288**

**1289**


**1290**

**1291**

**1292**

**1293**

**1294**

**1295**


Figure 10: Visualization of the learned MAD landscape using MadDist on the MediumMaze environment. The heatmap represents the predicted distance from a fixed goal state to every other point in
the maze


**–** _Discrete State Spaces:_ To provide foundational testbeds with clearly defined structures and
allow for exact MAD computation.

 - Action Spaces:

**–** _Continuous_ _Action_ _Spaces:_ To evaluate performance in environments where actions are
defined by real-valued parameters, common in robotics and physical control tasks.

**–** _Discrete Action Spaces:_ To ensure applicability to environments with a finite set of distinct
actions.

 - _Complex Dynamics:_ Incorporating environments like PointMaze, which feature non-trivial physics
(velocity, acceleration).

 - _Hard Exploration:_ Utilizing environments with complex structures (e.g., intricate mazes) that
pose significant exploration challenges for naive data collection policies (like the random policy
we used in our experiments).


NOISYGRIDWORLD


_Noisy Observations, Stochastic Dynamics, Continuous State Space, Discrete Action Space_


 - **State space:** The agent receives a 4-dimensional observation vector ( _x, y, n_ 1 _, n_ 2)at each step. In
this observation, ( _x, y_ ) are discrete coordinates in a 13 _×_ 13 grid, and ( _n_ 1 _, n_ 2) _∼N_ (0 _, σ_ [2] _I_ ) are
i.i.d. Gaussian noise components. The true underlying latent state, which is not directly observed
by the agent in its entirety without noise, is the coordinate pair ( _x, y_ ). The presence of the noise
components ( _n_ 1 _, n_ 2) in the observation makes the sequence of observations non-Markovian with
respect to this true latent state.

 - **Action space:** Four stochastic actions are available in all states: UP, DOWN, LEFT, and RIGHT.

 - **Transition dynamics:** With probability 0.5, the intended action is executed; with probability 0.5,
a random action is applied. Transitions are clipped at grid boundaries.

 - **Initial state distribution (** _µ_ 0 **):** The agent’s initial true latent state ( _x_ 0 _, y_ 0) is a random real-valued
position sampled uniformly from the grid. The full initial observation is ( _x_ 0 _, y_ 0 _, n_ 1 _,_ 0 _, n_ 2 _,_ 0), where
the initial noise components ( _n_ 1 _,_ 0 _, n_ 2 _,_ 0) are also sampled i.i.d. from _N_ (0 _, σ_ [2] _I_ ). The real-valued
nature of both the initial position and the noise components makes the observed state space
continuous.


24


**1296**

**1297**


**1298**

**1299**

**1300**

**1301**

**1302**

**1303**


**1304**

**1305**

**1306**

**1307**

**1308**

**1309**


**1310**

**1311**

**1312**

**1313**

**1314**


**1315**

**1316**

**1317**

**1318**

**1319**

**1320**


**1321**

**1322**

**1323**

**1324**

**1325**

**1326**


**1327**

**1328**

**1329**

**1330**

**1331**

**1332**


**1333**

**1334**

**1335**

**1336**

**1337**


**1338**

**1339**

**1340**

**1341**

**1342**

**1343**


**1344**

**1345**

**1346**

**1347**

**1348**

**1349**


Figure 11: Pearson and Spearman correlation coefficients and coefficient of variation (CV) ratios
across test environments. Shaded regions minimum and maximum values across three random seeds.


25


**1350**

**1351**


**1352**

**1353**

**1354**

**1355**

**1356**

**1357**


**1358**

**1359**

**1360**

**1361**

**1362**

**1363**


**1364**

**1365**

**1366**

**1367**

**1368**


**1369**

**1370**

**1371**

**1372**

**1373**

**1374**


**1375**

**1376**

**1377**

**1378**

**1379**

**1380**


**1381**

**1382**

**1383**

**1384**

**1385**

**1386**


**1387**

**1388**

**1389**

**1390**

**1391**


**1392**

**1393**

**1394**

**1395**

**1396**

**1397**


**1398**

**1399**

**1400**

**1401**

**1402**

**1403**


Figure 12: Pearson and Spearman correlation coefficients and coefficient of variation (CV) ratios
across OGBench test environments. Shaded regions minimum and maximum values values across
three random seeds.


26


**1404**

**1405**


**1406**

**1407**

**1408**

**1409**

**1410**

**1411**


**1412**

**1413**

**1414**

**1415**

**1416**

**1417**


**1418**

**1419**

**1420**

**1421**

**1422**


**1423**

**1424**

**1425**

**1426**

**1427**

**1428**


**1429**

**1430**

**1431**

**1432**

**1433**

**1434**


**1435**

**1436**

**1437**

**1438**

**1439**

**1440**


**1441**

**1442**

**1443**

**1444**

**1445**


**1446**

**1447**

**1448**

**1449**

**1450**

**1451**


**1452**

**1453**

**1454**

**1455**

**1456**

**1457**


- **Ground-truth MAD:** Since the latent state is deterministic apart from noise, the MAD between
two states ( _x_ 1 _, y_ 1) and ( _x_ 2 _, y_ 2) is the Manhattan distance _|x_ 1 _−x_ 2 _|_ + _|y_ 1 _−y_ 2 _|_ . Noise components
are ignored.


KEYDOORGRIDWORLD


_Asymmetric, Deterministic Dynamics, Discrete State Space, Discrete Action Space_


 - **State space:** States are triples ( _x, y, k_ ), where ( _x, y_ ) is the agent’s position in a 13 _×_ 13 grid, and
_k_ _∈{_ 0 _,_ 1 _}_ indicates whether the key has been collected.


 - **Action** **space:** Four deterministic actions are available in all states: UP, DOWN, LEFT, and
RIGHT.


 - **Transition dynamics:** Transitions are deterministic. The agent picks up the key by visiting the
key’s cell; the key cannot be dropped once collected. The door can only be passed if the key has
been collected.


 - **Initial state distribution (** _µ_ 0 **):** The agent starts at position (1 _,_ 1).


 - **Ground-truth MAD:** Defined as the minimum number of steps to reach the target state, accounting for key dependencies. For example, if the agent lacks the key and the goal requires it, the path
must include visiting the key first.


CLIFFWALKING


_Asymmetric, Deterministic Dynamics, Discrete State Space, Discrete Action Space_


 - **State space:** The environment is a 4 _×_ 12 grid. Each state corresponds to a discrete cell ( _x, y_ ).


 - **Action space:** Four deterministic actions are available in all states: UP, DOWN, LEFT, or RIGHT.


 - **Transition dynamics:** Transitions are deterministic unless the agent steps into a cliff cell, in
which case it is returned to the start. The episode is not reset.


 - **Initial state distribution (** _µ_ 0 **):** The agent starts at position (1 _,_ 1).


 - **Ground-truth MAD:** The MAD is the minimal number of steps required to reach the target state,
allowing for cliff transitions. Since falling into the cliff resets the agent’s position, it can create
shortcuts and lead to strong asymmetries in the distance metric.


POINTMAZE


_Continuous State Space, Complex Dynamics, Hard exploration, Continuous Action Space_


 - **State space:** The agent observes a 4-dimensional vector ( _x, y,_ _x,_ ˙ _y_ ˙), where ( _x, y_ ) is the position of
a green ball in a 2D maze and ( ˙ _x,_ _y_ ˙) are its linear velocities in the _x_ and _y_ directions, respectively.


 - **Action space:** Continuous control inputs ( _ax, ay_ ) corresponding to applied forces in the _x_ and _y_
directions. The applied force is limited to the range [ _−_ 1 _,_ 1] N in each direction.


 - **Transition** **dynamics:** The system follows simple force-based dynamics within the MuJoCo
physics engine. The applied forces affect the agent’s velocity, which in turn updates its position.
The ball’s velocity is limited to the range [ _−_ 5 _,_ 5] m _/_ s in each direction. Collisions with the maze’s
walls are inelastic: any attempted movement through a wall is blocked.


 - **Initial state distribution (** _µ_ 0 **):** The agent starts at a random real-valued position ( _x, y_ ) sampled
uniformly from valid maze locations. The initial velocities ( ˙ _x_ 0 _,_ _y_ ˙0) are set to (0 _,_ 0).


 - **Ground-truth MAD:** The maze is discretized into a uniform grid. Using the Floyd-Warshall
algorithm on the resulting connectivity graph, we compute shortest path distances between all
reachable pairs of positions.


27


**1458**

**1459**


**1460**

**1461**

**1462**

**1463**

**1464**

**1465**


**1466**

**1467**

**1468**

**1469**

**1470**

**1471**


**1472**

**1473**

**1474**

**1475**

**1476**


**1477**

**1478**

**1479**

**1480**

**1481**

**1482**


**1483**

**1484**

**1485**

**1486**

**1487**

**1488**


**1489**

**1490**

**1491**

**1492**

**1493**

**1494**


**1495**

**1496**

**1497**

**1498**

**1499**


**1500**

**1501**

**1502**

**1503**

**1504**

**1505**


**1506**

**1507**

**1508**

**1509**

**1510**

**1511**


OGBENCH POINTMAZE


_Continuous State Space, Complex Dynamics, Hard Exploration, Continuous Action Space_


This benchmark extends the PointMaze environment to significantly larger and more challenging
mazes, designed to test long-horizon reasoning and exploration capabilities. The controlled agent is
the same 2D ball as in PointMaze, but the scale and complexity of the mazes increase substantially.


 - **Medium:** Matches the original medium maze from D4RL.

 - **Large:** Matches the original large maze from D4RL.

 - **Giant:** Twice the size of Large, with a layout adapted from the antmaze-ultra maze of
Jiang et al. (2022). It contains longer paths, requiring up to 1000 environment steps, making it
especially demanding for long-horizon planning.


Two datasets are provided for each maze:


 - **Navigate:** Collected using a noisy expert policy that repeatedly navigates to randomly sampled
goals throughout the maze.

 - **Stitch:** Consists of short, goal-reaching trajectories of at most 4 cell units in length. Solving tasks
requires stitching together multiple short demonstrations (up to 8), testing the agent’s ability to
compose behaviors across long horizons.


H PLANNING EXPERIMENTS


To assess the practical utility of the learned MAD embeddings, we evaluated the performance
of our algorithms and baselines on a downstream goal-reaching task in the OGBench PointMaze
environments.


PLANNING ALGORITHM


We employed a simple planning algorithm based on random shooting, a form of model-predictive
control (MPC), which allows for a direct evaluation of the distance metric as a planning heuristic.
This approach isolates the effectiveness of the learned metric from confounding factors that would be
introduced by more complex planners.


The planning process at each time step _t_, given a current state _st_ and a goal state _g_, is as follows:


1. Generate _K_ = 100 candidate action sequences, each of length _H_, by sampling actions uniformly
at random at each step in the sequence.

2. For each of the _K_ action sequences, use the true environment simulator to roll out the corresponding state trajectory _{st_ +1 _, . . ., st_ + _H_ _}_ .

3. Score each trajectory by finding the state within it that minimizes the learned distance to the goal.
The score for a trajectory is given by min0 _<i≤H dθ_ ( _st_ + _i, g_ ), where _dθ_ is the learned distance.

4. Identify the action sequence that achieved the minimum score (i.e., the one that brought the agent
closest to the goal).

5. Execute the first action from this best-scoring sequence to transition to the next state, _st_ +1.


This entire process is repeated at each step in a receding-horizon fashion until the agent reaches the
goal or a maximum episode length is exceeded.


Our choice of this simple planning framework is deliberate. By relying on the true simulator and
random action sampling, the success of the planner depends directly on the metric’s ability to provide
a meaningful and accurate signal for progress toward the goal. This avoids confounding the evaluation
with inaccuracies that might arise from a learned dynamics model or the complexities of a separate
policy optimization algorithm.


It is important to note the limitations of this planner: since actions are sampled randomly, the resulting
trajectories are sub-optimal and tend to explore only a local region around the agent’s current state.


28


**1512**

**1513**


**1514**

**1515**

**1516**

**1517**

**1518**

**1519**


**1520**

**1521**

**1522**

**1523**

**1524**

**1525**


**1526**

**1527**

**1528**

**1529**

**1530**


**1531**

**1532**

**1533**

**1534**

**1535**

**1536**


**1537**

**1538**

**1539**

**1540**

**1541**

**1542**


**1543**

**1544**

**1545**

**1546**

**1547**

**1548**


**1549**

**1550**

**1551**

**1552**

**1553**


**1554**

**1555**

**1556**

**1557**

**1558**

**1559**


**1560**

**1561**

**1562**

**1563**

**1564**

**1565**


Therefore, success in these long-horizon tasks heavily relies on the learned metric providing a
consistent and reliable global signal toward the goal, guiding the planner effectively even with its
limited local search.


EVALUATION PROTOCOL


Each task in OGBench accompanies five pre-defined state-goal pairs for evaluation. To ensure
statistical robustness, we evaluate over 3 independent random seeds. For each seed and each of
the five state-goal pairs, we run 50 evaluation episodes, each with slightly randomized initial and
goal states. Performance, as reported in Table 1, is measured by the average success rate across all
episodes. An episode is considered successful if the agent reaches a state within a small Euclidean
distance of the goal coordinates.


29