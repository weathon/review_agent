# CASE-GUIDED SEQUENTIAL ASSAY PLANNING IN DRUG DISCOVERY


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Optimally sequencing experimental assays in drug discovery is a high-stakes planning problem under severe uncertainty and resource constraints. A primary obstacle for standard reinforcement learning (RL) is the absence of an explicit environment simulator or transition data ( _s, a, s_ _[′]_ ); planning must rely solely on a static
database of historical outcomes. We introduce the Implicit Bayesian Markov
Decision Process (IBMDP), a model-based RL framework designed for such
simulator-free settings. IBMDP constructs a case-guided implicit model of transition dynamics by forming a nonparametric belief distribution using similar historical outcomes. This mechanism enables Bayesian belief updating as evidence
accumulates and employs ensemble MCTS planning to generate stable policies
that balance information gain toward desired outcomes with resource efficiency.
We validate IBMDP through comprehensive experiments. On a real-world central
nervous system (CNS) drug discovery task, IBMDP reduced resource consumption by up to 92% compared to established heuristics while maintaining decision
confidence. To rigorously assess decision quality, we also benchmarked IBMDP
in a synthetic environment with a computable optimal policy. Our framework
achieves significantly higher alignment with this optimal policy than a deterministic value iteration alternative that uses the same similarity-based model, demonstrating the superiority of our ensemble planner. IBMDP offers a practical solution
for sequential experimental design in data-rich but simulator-poor domains.


1 INTRODUCTION


To discover new drugs, scientists make sequential decisions to conduct multiple assays, often constrained by limited time, budget, and materials. The process typically begins with sparse evidence
from historical assay outcomes on past compounds. Executing an assay for a new drug candidate
compound yields an observation of the assay consuming monetary and time resources, each probing a distinct facet of developability of the compound (e.g., potency, ADME, safety). For example,
an _in_ _vitro_ assay may be cheap and fast but only weakly informative downstream, whereas an _in_
_vivo_ assay is slower and more expensive yet more decisive for Go/No-Go decisions. Under tight
budget and schedule constraints, the central question is whether to run another assay or stop now.
Ideally, each chosen assay reduces posterior uncertainty while increasing the likelihood that the
compound satisfies predefined developability criteria. This is a planning problem under uncertainty,
further complicated by the absence of transition tuples ( _s, a, s_ _[′]_ )—only historical assay outcomes
from past compounds are available. In practice, rule-based playbooks and expert heuristics are often
risk-averse or myopic, leading to inefficient use of constrained resources and suboptimal portfolio
outcomes.


To address these challenges, we propose the Implicit Bayesian Markov Decision Process (IBMDP),
a reinforcement learning (RL) framework for case-guided sequential assay planning that uses assay
outcomes of historical compounds to construct an implicit probabilistic model of information gain
acquired from assays. At each step, IBMDP forms a categorical distribution over historical compound records using a variance-normalized distance kernel and samples plausible assay outcomes
consistent with the current partial evidence, thereby updating the candidate’s observed state. This
implicit, nonparametric transition model emphasizes contexts most relevant to the candidate without
requiring an explicit mechanistic simulator. Planning is performed with Monte Carlo Tree Search
with Double Progressive Widening (MCTS-DPW), and we run an ensemble of MCTS planners to
reduce variance from both stochastic sampling and tree search; majority voting across runs yields


1


Figure 1: Sequential decisions in drug discovery through a data-driven, analog-guided simulator for
planning, which maintains a Bayesian belief over the most relevant historical compound analogs.


a Maximum-Likelihood Action-Sets Path (MLASP) that is stable across uncertainty levels. When
simulating possible courses of actions during search, IBMDP takes into account the resulting reduction in uncertainty towards desirable states (e.g., high drug likeliness _in vivo_ ) and recommends the
next assay only when the uncertainty reduction reaches a sufficient magnitude towards the desirable
states. From the Partially Observable MDP (POMDP) perspective, while standard methods maintain explicit probability distributions over hidden states and update them via Bayes’ rule, IBMDP
makes decisions by sampling from past experiences weighted by similarity to the current observed
state (Appendix A). While IBMDP trades formal convergence guarantees for practical applicability
in simulator-free settings, it provides empirically robust policies through ensemble MCTS planning
(Appendix A.6).


**Contributions.** **(i) RL planning with evidence-adaptive dynamics:** Unlike traditional RL with
fixed transition functions, IBMDP’s implicit dynamics evolve as observations accumulate—the
similarity-based belief continuously adapts, creating non-stationary but principled state transitions
from static historical data. **(ii) Similarity-weighted Bayesian belief mechanism:** We transform historical outcomes into an adaptive generative model where transition probabilities dynamically shift
based on accumulated evidence, enabling planning without explicit dynamics or ( _s, a, s_ _[′]_ ) trajectories. **(iii)** **Robust** **ensemble** **MCTS** **despite** **non-stationary** **dynamics:** Our ensemble approach
with majority voting (MLASP) produces stable policies even with evolving transition models, optimally balancing information gain with resource efficiency.


2 PRELIMINARIES


**Compounds, Assays, and Historical Data** Let _A_ = _{a_ 1 _, . . ., aM_ _}_ be the set of available assays
and _X_ = _{x_ 1 _, . . ., xN_ _}_ be the set of historical compounds, each with a fixed molecular representation. For a historical compound _xi_ and an assay _aj_, the observed outcome is denoted by _yi,j_ . The
complete historical dataset is represented as a set of tuples:
_D_ = _{_ ( _xi,_ **y** _i_ ) _}_ _[N]_ _i_ =1 _[,]_
where **y** _i_ = ( _yi,_ 1 _, . . ., yi,M_ ) is the vector of all assay outcomes for compound _xi_ . The new drug
candidate compound for which we are planning is denoted _x⋆_ _≡_ _xN_ +1. We also have access to perassay predictor models, such as Quantitative Structure-Activity Relationship (QSAR) models, which
are functions _fj_ : _x_ _�→_ _y_ ˆ _j_ = _fj_ ( _x_ ) that can be queried for the candidate _x⋆_ during the planning
phase (Chen et al., 2024). For convenience, Appendix G collects all symbols used throughout the
paper in the _Global Notation Reference_ .


**Target Property** Let _g_ be the primary scalar target of interest, such as a definitive _in vivo_ endpoint
that determines a compound’s success. The historical values for this target form the set _G_ = _{gi}_ _[N]_ _i_ =1 [.]
In many applications, the target property may correspond to one of the available assays. That is, for
some specific assay index _j_ _∈{_ 1 _, . . ., M_ _}_, we have _gi_ _≡_ _yi,j_ for all compounds. _Crucially_, to
prevent data leakage, the set of target values _G_ is never used in the computation of similarity or
distance metrics during planning. We define _Ig_ = _{i_ : _gi_ is available _}_ as the set of indices for
historical compounds where the target value has been measured.


**State** **and** **Actions** We formulate the assay planning problem for the candidate _x⋆_ as a finitehorizon MDP with a discount factor _γ_ _∈_ [0 _,_ 1) and a maximum horizon _T_ . At any decision step _t_,
we maintain an index set of assays that have already been performed, _Mt_ _⊆{_ 1 _, . . ., M_ _}_, and the set
of unmeasured assays, _Ut_ := _{_ 1 _, . . ., M_ _} \ Mt_ . The process starts with an empty set of measured
assays, _M_ 0 = _∅_ .


2


The **state** at step _t_ summarizes all accumulated knowledge about the candidate compound: _st_ =

- _x⋆,_ _{y⋆,j}j∈Mt_ - _._ The **action set** at step _t_, _At_, consists of choosing a batch of up to _m_ currently unmeasured assays to perform, or deciding to stop the experiment. Formally: _At_ = _P≤m_ ( _Ut_ ) _∪{_ eox _}._
Here, _P≤m_ ( _Ut_ ) is the set of all subsets of _Ut_ with size at most _m_, and ‘eox‘ (end-of-experiment) is
the terminal action. The parameter _m ≤_ _M_ is a user-specified throughput limit that caps how many
assays can be run in parallel at a single step. Executing an action _At_ _⊆_ _Ut_ reveals the outcomes
_{y⋆,j}j∈At_ and updates the measured and unmeasured sets for the next step, _t_ + 1.


**Reward Function** Each action incurs a cost based on the resources it consumes (e.g., time, materials, monetary expense). Let _cj_ _∈_ R _[q]_ _≥_ 0 [be the cost vector for an individual assay] _[ a][j]_ [.] [The cost for a]
batch action _At_ is the sum of the costs of the individual assays within it, i.e., _c_ ( _st, At_ ) = [�] _aj_ _∈At_ _[c][j]_ [.]

Let _**ρ**_ _∈_ R _[q]_ _≥_ 0 [be a user-defined vector of weights that specifies the trade-offs between different re-]
sources. The scalar step reward, _R_ ( _st, At_ ), is defined as:

          - _−_ _**ρ**_ T _c_ ( _st, At_ ) _,_ if _At_ _∈At \ {_ eox _},_
_R_ ( _st, At_ ) = 0 _,_ if _At_ = eox _._


**Uncertainty** **and** **Goal-Likelihood** **Functionals** To ensure resources are directed toward viable
drug candidates, we define two key state-dependent scalar functions based on similarity weights
_wi_ ( _st_ ) over historical records (to be formally defined in a later section). First, we renormalize the
weights to consider only the historical compounds for which the target value _g_ is available:

_wi_ ( _st_ )
_w_ ˜ _i_ ( _st_ ) =         - for _i ∈_ _Ig._
_ℓ∈Ig_ _[w][ℓ]_ [(] _[s][t]_ [)]

Note that when _|Ig|_ _≪_ _N_, this renormalization may lead to variance underestimation as it restricts
the effective sample size. This limitation is discussed in the experimental analysis. Using these
normalized weights, we define:


          - _H_ ( _sT_ ) _≤_ _ϵ,_ _ϵ ∈_ [0 _,_ 1] _,_
subject to
_L_ ( _st_ ) _≥_ _τ,_ _∀t_ = 0 _, . . ., T_ _−_ 1 _,_

where _ϵ_ _>_ 0 is the maximum tolerable uncertainty at the terminal state _sT_, and _τ_ _∈_ (0 _,_ 1) is the
minimum acceptable goal-likelihood at every intermediate step. The feasibility constraint _L_ ( _st_ ) _≥_ _τ_
ensures that the planning process remains on a trajectory toward a successful outcome, while the
terminal constraint _H_ ( _sT_ ) _≤_ _ϵ_ guarantees that a decision is made with sufficient confidence.


3 IMPLICIT MODEL OF ENVIRONMENT DYNAMICS


The key challenge is updating the state transition _st_ _−−→At_ _st_ +1—i.e., how the state evolves after
executing a batch of assays—when no explicit simulator is available and only historical data _D_ can


3


1. **State-Uncertainty (** _H_ ( _st_ ) **):** The weighted variance of the target property _g_ over the relevant historical data, which serves as a measure of uncertainty about the candidate’s potential
outcome.
_H_ ( _st_ ) =    - _w_ ˜ _i_ ( _st_ )    - _gi −_ _g_ ¯( _st_ )�2 _,_ where _g_ ¯( _st_ ) =    - _w_ ˜ _i_ ( _st_ ) _gi._ (1)


- _w_ ˜ _i_ ( _st_ ) - _gi −_ _g_ ¯( _st_ )�2 _,_ where _g_ ¯( _st_ ) = 
_i∈Ig_ _i∈Ig_


_w_ ˜ _i_ ( _st_ ) _gi._ (1)

_i∈Ig_


2. **Goal-Likelihood** **(** _L_ ( _st_ ) **):** The weighted probability that the candidate’s target property
falls within a predefined desirable range [ _g_ min _, g_ max].

_L_ ( _st_ ) =          - _w_ ˜ _i_ ( _st_ ) **1** [ _gi_ _∈_ [ _g_ min _, g_ max] ] _._ (2)


_i∈Ig_


Here, **1** [ _·_ ] denotes the indicator function, which returns 1 when its argument is true and 0
otherwise.


**Constrained** **Objective** The optimal policy _π_ _[∗]_ is one that maximizes the total expected reward,
subject to constraints on terminal uncertainty and stepwise feasibility. Specifically, we aim to solve:


(3)


_π_ _[∗]_ _∈_ arg max E _π_
_π_


- _T_ 
- _γ_ _[t]_ _R_ - _st, π_ ( _st_ )�


_t_ =0


be leveraged to infer dynamics. We address this by constructing an implicit, generative model of the
environment’s dynamics. This model uses a similarity metric to dynamically re-weight historical
compound profiles, forming a belief over plausible outcomes for the candidate compound _x⋆_ . This
avoids explicit parameterization of transition probabilities and implicitly propagates uncertainty by
sampling from historical compound analogs most relevant to the current state of _x⋆_ .


**Similarity Weight Computation** The transition model is centered on a similarity weight, _wi_ ( _st_ ),
assigned to each historical compound record _Di_ = ( _xi,_ **y** _i_ ) _∈D_ . These weights quantify the
relevance of each historical case to the current state, _st_ . The weights are computed using a variancenormalized exponential kernel:
_wi_ ( _st_ ) = exp ( _−λw · d_ ( _st, Di_ )) _,_ (4)
where _d_ ( _st, Di_ ) is a distance metric. The distance is computed over the set of all features known for
the candidate _x⋆_ at step _t_, which we denote as the feature set _Kt_ . This set includes all initial QSAR
predictions and the outcomes of all measured assays in _Mt_ . The distance compares these known
values for the candidate to the corresponding values for the historical compound _xi_ :


where _⊕_ denotes the state update operation that augments the current state by adding new assay
outcomes to _Mt_ and updating the observed values _{y⋆,j}_, and **1** [ _·_ ] is the indicator function. This
sampling-based approach approximates the true transition dynamics when the historical dataset _D_
is sufficiently representative of the underlying system. The quality of the approximation depends on
the coverage of _D_, the appropriateness of the similarity metric, and the dataset size _N_ = _|D|_ .


**Bayesian** **Weight** **Update** After transitioning to the new state _st_ +1, the similarity weights are
re-evaluated to incorporate the new evidence. This recalculation is a direct and principled implementation of a Bayesian belief update. As we formally derive in Appendix A, our framework is


4


- _λk ·_ [(] _[ϕ][k]_ [(] _[s][t]_ [)] _[ −]_ _[ϕ][k]_ [(] _[D][i]_ [))][2]

_k∈Kt_ _σk_ [2]


_d_ ( _st, Di_ ) = 


_,_ (5)
_σk_ [2]


Here, _ϕk_ ( _·_ ) is an extractor function that returns the value of the _k_ -th feature from a given state
or historical record. For the candidate, _ϕk_ ( _st_ ) is either a QSAR prediction or a measured assay
outcome _{y⋆,j}j∈Mt_ . For the historical compound, _ϕk_ ( _Di_ ) is the corresponding recorded value.
The term _σk_ [2] [is] [the] [empirical] [variance] [of] [feature] _[k]_ [across] [the] [historical] [dataset] _[D]_ [,] [computed] [as]
_σk_ [2] [=] _N_ 1 - _Ni_ =1 [(] _[ϕ][k]_ [(] _[D][i]_ [)] _[−]_ _[ϕ]_ [¯] _[k]_ [)][2] [where] _[ϕ]_ [¯] _[k]_ [=] _N_ 1 - _Ni_ =1 _[ϕ][k]_ [(] _[D][i]_ [)][.] [The] [parameter] _[λ][k]_ [is] [a] [feature-]
specific weight, and _λw_ is a global temperature parameter. The variance normalization ensures a
dimensionless comparison across features with different scales.


**Similarity-Based State Transition** The transition from state _st_ to _st_ +1 after executing an action
(a batch of assays) _At_ _⊆_ _Ut_ is simulated through a weighted sampling process. First, a historical
case is sampled from _D_ with probability proportional to its similarity weight:


     - _w_ 1( _st_ )
_I_ _∼_ Categorical _, . . .,_ _[w][N]_ [(] _[s][t]_ [)]
_Z_ _Z_


_,_ where _Z_ =


_N_

- _wi_ ( _st_ ) _._


_i_ =1


Let the selected historical case be _DI_ = ( _xI_ _,_ **y** _I_ ). The outcomes for the assays in the action batch
_At_ are then ”revealed” by taking the corresponding values from this sampled case:
_{y⋆,j_ := _yI,j}j∈At._
The new state _st_ +1 is formed by augmenting the previous state with these newly generated outcomes.
Formally, _Mt_ +1 = _Mt ∪_ _At_, and the new state is:
_st_ +1 =                   - _x⋆,_ _{y⋆,j}j∈Mt_ +1� _._
This generative process ensures that the simulated outcomes for the new assays are consistent with a
plausible, historically observed compound profile, thereby preserving correlations between assays.


**Implicit** **Transition** **Modeling** **via** **Sampling** The sampling mechanism described above defines
an implicit transition probability distribution _P_ ( _st_ +1 _|st, At_ ). This distribution is a mixture model
where each component corresponds to one of the historical cases in _D_ . The probability of transitioning to a specific next state _st_ +1 is the total weight of all historical cases that would produce that
state:


_P_ ( _st_ +1 _|st, At_ ) =


_N_


_i_ =1


_wiZ_ ( _st_ ) _·_ **1** [ _st_ +1 = _st ⊕{_ ( _aj, yi,j_ ) _}j∈At_ ] _,_ (6)


equivalent to a POMDP where the hidden state is a latent index over the historical cases in _D_ . The
similarity weights _wi_ ( _st_ ) represent the posterior belief over these latent ”prototypes,” and the recalculation after observing new assay outcomes is equivalent to applying Bayes’ rule. The new weights
_{wi_ ( _st_ +1) _}_ _[N]_ _i_ =1 [are] [computed] [using] [the] [same] [distance] [function] [as] [before,] [but] [now] [applied] [to] [the]
augmented state _st_ +1:
_wi_ ( _st_ +1) = exp ( _−λw · d_ ( _st_ +1 _, Di_ )) _._ (7)
This update mechanism shifts the model’s belief toward historical cases that are most consistent with
the expanded set of evidence for the candidate _x⋆_, allowing the planner to refine its predictions and
subsequent decisions as more data is gathered.


4 IMPLICIT BAYESIAN MARKOV DECISION PROCESS (IBMDP)


The Implicit Bayesian Markov Decision Process (IBMDP) is a planning framework designed to
solve the constrained optimization problem defined in Equation equation 3. It integrates the implicit, case-based transition model with a powerful planning algorithm to find reward-maximizing
sequences of assays. The core of the framework is a Monte Carlo Tree Search (MCTS) planner that
navigates the decision space by simulating potential experimental paths using the generative model
derived from historical data. To ensure the robustness of its recommendations, IBMDP employs an
ensemble method, aggregating the results of multiple independent planning runs.


The overall workflow proceeds as follows: Historical Data _D_ informs a Similarity Module, which
computes weights _wi_ ( _st_ ) for the current state _st_ . These weights drive the implicit transition model
used by an MCTS-DPW planner. The planner generates a policy, and this process is repeated
across an ensemble of runs. Finally, the policies are aggregated to construct a Maximum-Likelihood
Action-Sets Path (MLASP), which constitutes the final recommended experimental plan. The detailed procedure is outlined in Algorithm 1.


**Algorithm 1** Ensemble IBMDP Algorithm
**Require:** Initial state _s_ 0 = ( _x⋆, M_ 0 = _∅_ ), historical data _D_, reward function _R_ ( _s, A_ ), functionals _H_ ( _s_ ) _, L_ ( _s_ ),
thresholds _ϵ, τ_, horizon _T_, iterations _n_ itr, ensemble size _Ne_ .
**Ensure:** A Maximum-Likelihood Action-Sets Path (MLASP).

1: Initialize policy set Π _←∅_ .
2: **for** _j_ = 1 to _Ne_ **do** _▷_ Ensemble loop
3: Initialize MCTS tree _T_ with root node _s_ 0.
4: **for** _i_ = 1 to _n_ itr **do** _▷_ MCTS iterations
5: **Selection:** Traverse _T_ from _s_ 0 using a tree policy (e.g., UCB1) to select a leaf node _sleaf_ .
6: **Expansion:** If _sleaf_ is not a terminal state ( _H_ ( _sleaf_ ) _> ϵ_ ), choose an untried action _A ∈At_ ( _sleaf_ )
and create a new child node _snew_ .
7: **Simulation (Rollout):** From _snew_, simulate a trajectory of states and actions using a reward-aware
heuristic policy until a terminal state or horizon _T_ is reached.
8: During rollout, for a transition ( _s, A_ ) _→_ _s_ _[′]_, the next state _s_ _[′]_ is generated by the implicit model:
sample _I_ _∼_ Cat( _{wk_ ( _s_ ) _/Z}_ _[N]_ _k_ =1 [)][ where] _[ Z]_ [=][ �] _[N]_ _k_ =1 _[w][k]_ [(] _[s]_ [)][ and set] _[ s][′]_ [=] _[ s][ ⊕{]_ [(] _[a][k][, y][I,k]_ [)] _[}][k][∈][A]_ [.]
9: The total return _Q_ is the cumulative reward, with a large negative reward (e.g., _−_ 10 [6] ) if _L_ ( _s_ ) _< τ_
for any state in the trajectory.
10: **Backpropagation:** Update the visit counts and value estimates for all nodes on the path from _snew_
back to the root using the return _Q_ .
11: **end for**
12: Extract the optimal policy _πj_ _[∗]_ [from] [the] [final] [tree] _[T]_ [by] [selecting] [the] [action] [with] [the] [highest] [value] [at]
each node.
13: Π _←_ Π _∪{πj_ _[∗][}]_ [.]
14: **end for**
15: Construct MLASP by aggregating policies in Π via majority voting at each decision step.
16: **return** MLASP


IBMDP begins with the initial state _s_ 0, which contains the candidate compound _x⋆_ and any preexisting QSAR predictions, with an empty set of measured assays ( _M_ 0 = _∅_ ). The planner, MCTS
with Double Progressive Widening (MCTS-DPW), is particularly well-suited for this problem due
to its ability to handle large, combinatorial action spaces—in this case, the power set of unmeasured
assays, _P≤m_ ( _Ut_ ).


During each simulation step within the MCTS algorithm, the planner must evaluate the consequence
of taking an action _At_ . It does this by invoking the implicit transition model from the previous
section. A historical case _DI_ is sampled based on the current similarity weights _wi_ ( _st_ ), and the


5


outcomes for the assays in _At_ are drawn from this case. This yields a simulated next state, _st_ +1.
The planner then recalculates the similarity weights for this new state, _wi_ ( _st_ +1), and evaluates the
state-uncertainty _H_ ( _st_ +1) and goal-likelihood _L_ ( _st_ +1). A state is considered terminal if the uncertainty _H_ ( _s_ ) falls below the threshold _ϵ_, and the planner receives a negative reward if the feasibility
constraint _L_ ( _s_ ) _< τ_ is violated at any step. The immediate step reward, _R_ ( _st, At_ ), is also recorded.
This process allows the MCTS to build a search tree that accurately reflects the trade-off between
reward (resource efficiency) and the expected information gain towards desired states, all guided by
the historical data.


To mitigate stochasticity, we run the planning process multiple times to form an ensemble. The
final recommendation (MLASP) is constructed by majority vote over the actions recommended by
the ensemble policies at each stage. This ensures the plan is robust and not an artifact of a single
simulation run.

5 EXPERIMENTS


We validate the performance of IBMDP through a two-part evaluation. First, we apply it to a realworld sequential assay planning task in central nervous system (CNS) drug discovery to demonstrate
its practical utility and potential for resource savings. For reproducibility, we performed the same
experiment on a public dataset on selecting _in_ _vivo_ pharmacokinetics assays between rat and dog
to determine _in vivo_ clearance in human (Appendix E). Second, we set up a synthetic environment
with a known optimal policy to rigorously assess the quality of its decision-making process.


5.1 BRAIN PENETRATION ASSAYS: A REAL-WORLD CASE STUDY


**Problem Setting and Data.** We evaluate IBMDP on a sequential assay-planning task for central
nervous system (CNS) drug discovery, where the objective is to efficiently determine a compound’s
brain penetration potential. This property is critically dependent on the compound’s ability to cross
the blood-brain barrier (BBB). The decision involves selecting from cheap, fast, but less informative
_in_ _vitro_ transporter assays (P-glycoprotein, PgP; Breast Cancer Resistance Protein, BCRP) and a
definitive but slow and expensive _in vivo_ assay that measures the unbound brain-to-plasma partition
coefficient ( _k_ puu).


Our historical dataset, _D_, comprises _N_ = 220 compounds with complete measurements for all
relevant assays (100 nM PgP, 1 µM PgP, 100 nM BCRP) and the target property, _k_ puu. All compounds
also have associated QSAR predictions, which provide initial estimates for the assay outcomes (e.g.,
for PgP and BCRP activity) and other relevant properties such as Mean Residence Time (MRT). The
operational costs are defined as $400 and a 7-day turnaround for each _in vitro_ assay, and $4,000 and
a 21-day turnaround for the _in vivo k_ puu assay. Actions are constrained to a maximum of 3 parallel
assays per step. A compound is considered to have high potential if its _k_ puu _>_ 0 _._ 5.


**Experimental Setup.** The planning objective is to balance the reduction of state uncertainty _H_ ( _s_ )
on the target _k_ puu, the increase in goal likelihood _L_ ( _s_ ) that _k_ puu is in the desirable range, and the
maximization of reward (efficient use of resources). An experimental sequence terminates when
the planner reaches a state of sufficient confidence, defined by the joint criteria _H_ ( _sT_ ) _≤_ _ϵ_ and
_L_ ( _st_ ) _≥_ _τ_ for all intermediate steps. Outcomes are compared against a conventional, rule-based
decision strategy.


**Rule-Based** **Baseline.** In practice, decisions often follow simple heuristics based on QSAR predictions. For this task, the baseline heuristic is:


    - A compound is deemed **promising** (likely 0 _._ 5 _≤_ _k_ puu _≤_ 1) if QSAR1uM ~~P~~ gP _<_ 2 AND
QSAR100nM BCRP _<_ 2.


    - A compound is deemed **non-promising** (likely _k_ puu _≤_ 0 _._ 5) if QSAR1uM ~~P~~ gP _>_ 4 OR
QSAR100nM BCRP _>_ 4.


We evaluated IBMDP across four representative scenarios from three categories designed to test
its performance against this heuristic: (i) **Baseline** **confirmation** (clear QSAR signals, Scenario
3), (ii) **Heuristic** **challenge** (conflicting or borderline QSAR signals, Scenarios 2 and 4), and (iii)
**Opportunity discovery** (QSARs suggest a non-promising compound that is, in fact, good, Scenario
1).


6


Figure 2: Monetary-prioritized results from IBMDP for four representative compounds. Each plot
shows the Pareto front of achievable resource consumption versus terminal state uncertainty, with
the Maximum-Likelihood Action-Sets Path (MLASP) highlighted. This illustrates how IBMDP
provides a trade-off curve, allowing decision-makers to select a plan based on their risk and budget
tolerance.


**Results.** As shown in Table 1, IBMDP consistently identifies more resource-efficient experimental
plans than the traditional approach, which often defaults to running a full panel of assays consuming $5,200. The table rows are ordered to correspond directly to the scenarios shown in Figure 2.
In the opportunity discovery scenario (row 1/Scenario 1), the heuristic would have incorrectly discarded a valuable compound, whereas IBMDP recommends an efficient $800 plan to reveal its true
potential. For the next compound (row 2/Scenario 2), IBMDP finds a minimal $400 plan to resolve uncertainty. In the baseline confirmation scenario (row 3/Scenario 3), IBMDP recommends
just $400-$800 to confirm the promising profile. Finally, for the challenging case with conflicting
QSARs (row 4/Scenario 4), IBMDP efficiently resolves uncertainty for $400-$800. Across these
representative cases, IBMDP achieves the same or higher level of decision confidence with up to
92% reduction in resource consumption.


5.2 SIMULATION WITH SYNTHETIC DATA


**Benchmark Setup.** To rigorously assess the policy quality of IBMDP in a controlled setting, we
benchmarked it using a synthetic dataset where a theoretically optimal policy is computable (full details in Appendix D). We established this optimal policy using Value Iteration with the true, analytic
uncertainty dynamics (VI-Theo). We then compared IBMDP against both this VI-Theo baseline
and a deterministic variant using the same similarity-based estimation as IBMDP, but planned with
Value Iteration (VI-Sim).


**Results.** The results, summarized in Table 2, demonstrate the effectiveness of IBMDP’s stochastic, ensemble-based planning. Over 100 independent trials, IBMDP’s primary recommendation (Top
1) aligned with the optimal VI-Theo policy in 47% of cases. In contrast, the deterministic VI-Sim


7


Table 1: Resource expense comparison between the traditional heuristic approach and IBMDP for
representative compounds. Rows are ordered to match scenarios 1-4 in Figure 2. The traditional
approach expense of $5200 reflects running the full assay panel ($4000 for _k_ puu plus 3 _×_ $400 for
_in vitro_ assays), which IBMDP consistently avoids.

QSAR Predictor Assays Expense ( _×_ $100)


1uM 100nM mrt kpuu 100nM 1uM 100nM Trad. IBMDP
PgP BCRP PgP PgP BCRP


5.0 9.6 1.0 0.53 15.9 12.9 8.2 52 8


0.9 8.5 2.6 0.53 2.2 1.1 14.2 52 4


1.7 1.3 1.8 0.54 1.1 0.8 1.3 52 4 - 8


21.4 0.7 1.2 0.64 17.4 19.7 0.8 52 4 - 8


approach achieved only 36% alignment. The advantage of the ensemble approach is further highlighted by the fact that the optimal action was contained within IBMDP’s top two recommendations
66% of the time, providing robust and effective coverage of the high-value policy space.


Table 2: Policy Alignment with Theoretical Baseline

Method Matches Match Rate (%)


IBMDP Top 1 47 47.0
IBMDP Top 2 66 66.0
VI Similarity 36 36.0


This superior performance stems from a fundamental difference in policy generation. While VIbased methods converge to a single, deterministic policy, IBMDP’s ensemble of MCTS agents explores the policy space more broadly. This allows it to identify multiple, often near-equivalent,
high-value actions, which is particularly advantageous in assay selection where different feature
combinations can yield similar information gains. The results confirm that our ensemble-based
planner provides more robust and reliable recommendations than a deterministic alternative by effectively navigating the uncertainty inherent in the policy space itself.


6 CONCLUSIONS


To achieve case-guided planning, we presented IBMDP, a reinforcement learning framework that
turns historical cases into a generative model for sequential assay selection. By weighting historical
cases based on similarity, the algorithm enables robust, multi-step planning with Monte Carlo Tree
Search without requiring an explicit transition function. The application to a real-world drug discovery problem demonstrated it uncovers ground truth of a compound with fewer, cheaper assays.
This work establishes a powerful methodology for leveraging past experience to guide future experiments, with broad applicability in fields beyond drug discovery where historical data is abundant
but mechanistic models are scarce.


7 RELATED WORK


**MDPs** **and** **Model-Based** **RL.** MDPs formalize sequential decision-making (Puterman, 2014);
model-based RL learns dynamics for planning (Sutton & Barto, 2018; Kaiser et al., 2019; Moerland
et al., 2023). Kernel-based RL leverages similarity primarily for value approximation or smoothing
learned transitions (Ormoneit & Sen, 2002; Kveton & Theocharous, 2012; Xu et al., 2007). IBMDP
uses similarity to build a _generative_, nonparametric transition _without_ ( _s, a, s_ _[′]_ ) tuples—sampling
assay outcomes from historical records rather than learning explicit kernels over next-states.


**Bayesian** **RL** **and** **Bayesian** **Optimization.** BRL maintains posteriors over model parameters
or values and samples explicit MDPs (e.g., PSRL) (Ghavamzadeh et al., 2015; Osband et al.,
2013; Agrawal & Jia, 2017). BO targets one-shot improvement of objective functions (Griffiths &
Hern´andez-Lobato, 2020; G´omez-Bombarelli et al., 2018). IBMDP avoids explicit parameter posteriors and performs _Bayesian case-based generation_ via similarity-weighted reweighting of records,
enabling _multi-step_ planning with reward optimization and feasibility constraints.


8


**Bayesian** **Experimental** **Design** **and** **Implicit** **Models.** Canonical single-step BO/BED methods
are myopic and assume an explicit likelihood or simulator(Chaloner & Verdinelli, 1995; Rainforth
et al., 2024); implicit-BED handles intractable likelihoods with info-theoretic surrogates or policy
learning (Kleinegesse & Gutmann, 2020; 2021; Ivanova et al., 2021). IBMDP embeds an implicit
model inside an _RL_ _planner_ (MCTS-DPW), balancing reward, time, and feasibility—not solely
information gain.


**Constrained** **MDPs** **and** **POMDPs.** CMDPs typically constrain cumulative costs (Achiam et al.,
2017); our constraints target _state_ properties (terminal uncertainty, per-step likelihood) enforced
during planning. The setting is akin to POMDPs (Kaelbling et al., 1998); our similarity-weighted
posterior over records acts as an _implicit belief_ . While multi-step RL/POMDP solvers require simulators or ( _s, a, s_ _[′]_ ) tuples, IBMDP uses a _similarity-weighted, implicit generative model_ built directly
from historical assay profiles, preserving cross-assay dependence without learning explicit dynamics. A direct benchmark is therefore not strictly comparable without substantial adaptation: (i)
redefining utilities over _assays_ rather than inputs, (ii) adding _resource-aware_ batching and a principled _stopping_ rule aligned with our constraints on _H_ ( _s_ ) and _L_ ( _s_ ), and (iii) supplying a _posterior_
_predictive_ consistent with the no-simulator setting (Appendix A).


**Ensembles** **in** **RL.** Ensembles improve robustness and uncertainty estimates (Dietterich, 2000;
Zhou, 2012; Wiering & Van Hasselt, 2008; Osband et al., 2016; Lakshminarayanan et al., 2017).
IBMDP use ensembling pragmatically to stabilize stochastic planning.


**Application** **Context.** Prior RL in biomedicine focuses on trials or molecule generation/synthesis
(Bennett & Hauser, 2013; Eghbali-Zarch et al., 2019; Abbas et al., 2007; Fard et al., 2018; Wang
et al., 2021; Bengio et al., 2021; You et al., 2018; Zhou et al., 2019; Segler et al., 2018). Assay selection in early discovery remains underexplored. IBMDP supplies a practical planner that converts
historical assay records into a coherent, generative transition model with operational constraints,
addressing the ”no ( _s, a, s_ _[′]_ )” regime typical of discovery.


**On** **Fair** **Comparison** **with** **Related** **Methods.** While the above methods appear relevant, direct
benchmarking would be fundamentally unfair—each operates under different mathematical assumptions and problem formulations. Model-based RL requires ( _s, a, s_ _[′]_ ) data or simulators; Bayesian RL
samples from parameter posteriors; BO performs single-step optimization; POMDPs need explicit
transition models. IBMDP uniquely addresses the setting where only static historical outcomes
exist, making these comparisons ”apples to oranges.” See Appendix C for detailed analysis.


8 LIMITATIONS


**Historical data coverage.** Effectiveness hinges on the quality/representativeness of _D_ ; gaps or bias
can yield suboptimal choices. Unlike model-free RL with exploration, similarity-based sampling
cannot discover strategies absent from _D_ —though in discovery, stable physico-chemical regularities
partly mitigate this risk.


**Similarity metric assumptions.** The exponential kernel over (normalized) Euclidean distances assumes these distances reflect assay behavior. Nonlinear/threshold biology may violate this; domaintailored metrics may be required to capture structure–activity relations.


**Scalability.** The worst-case total complexity is _O_ ( _Ne · n_ itr _·_ min( _b_ _[H]_ _, n_ itr) _· |D| · d_ ), where _b_ is the
effective branching factor and _H_ is the maximum tree depth. The per-iteration cost is dominated by
the similarity weight calculation ( _O_ ( _|D| · d_ )). Large datasets _|D|_ or high feature dimensions _d_ can
strain memory and compute, potentially requiring distributed infrastructure or data subsampling for
enterprise-scale use.


**Hyperparameter** **sensitivity.** Performance depends on tuning _{λw, λk, Ne, c, ϵ, τ_ _}_ ; robustness
across programs may require expert priors or nontrivial validation budgets.


9


**Ethics Statement**


In accordance with the ICLR Code of Ethics, this work is intended to contribute positively
to society by addressing a key challenge in pharmaceutical research, the resource waste due to
inefficient decisions and the use of preclinical animals in drug discovery.


The primary goal of the proposed framework, the Implicit Bayesian Markov Decision Process (IBMDP), is to enhance human well-being by making the drug discovery process more efficient. By
optimizing the sequence of experimental assays, this research aims to reduce the significant monetary and time costs associated with developing new medicines. An ethical benefit of this approach is
the potential to minimize harm by reducing the number of costly and lengthy _in vivo_ animal assays,
prioritizing such scarce resource for only the most promising compounds.


We are committed to upholding high standards of scientific excellence and transparency. The IBMDP framework was rigorously evaluated on both a real-world central nervous system (CNS) drug
discovery task and a synthetic environment where the optimal policy was computable, ensuring a
thorough assessment of its performance. We have been transparent about the method’s limitations,
particularly its dependence on the quality and representativeness of the historical data used for planning. The main ethical consideration is that a biased or incomplete historical dataset could lead to
suboptimal decisions, potentially resulting in missed opportunities or wasted resources.


The research utilizes preclinical data on chemical compounds and their assay outcomes. It does not
involve data from human subjects, thereby minimizing concerns related to personal privacy. We
believe this work represents a responsible application of machine learning to a critical scientific
domain.


10


**Reproducibility Statement**


To ensure the reproducibility of our findings, we have provided detailed descriptions of our
methodology and experimental setup. The IBMDP framework is outlined in Section 4, with a
concrete implementation provided in Algorithm 1. The theoretical underpinnings of our similaritybased model, including its formal correspondence to a POMDP, are detailed in Appendix A. For our
theoretical claims, a complete derivation and consistency proof for the similarity-based estimator in
the synthetic setting is available in Appendix D.


All experimental setups are described in Section 5. Full details on hyperparameter selection can
be found in Appendix B. The process for generating the synthetic dataset is specified in Appendix
D.1, and the public dataset used for the clearance optimization benchmark is cited and described
in Appendix E. The source code, data, and scripts to reproduce results have been uploaded to the
Supplementary Material, which will be visible to reviewers and the public throughout and after the
review period.


11


REFERENCES


Ismail Abbas, Joan Rovira, and Josep Casanovas. Clinical trial optimization: Monte carlo simulation
markov model for planning clinical trials recruitment. _Contemporary Clinical Trials_, 28(3):220–
231, 2007.


Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In
_International conference on machine learning_, pp. 22–31. PMLR, 2017.


Shipra Agrawal and Randy Jia. Posterior sampling for reinforcement learning: worst-case regret
bounds. _Advances in Neural Information Processing Systems_, 30, 2017.


Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, and Yoshua Bengio. Flow
network based generative models for non-iterative diverse candidate generation. _Advances_ _in_
_Neural Information Processing Systems_, 34:27381–27394, 2021.


Casey C Bennett and Kris Hauser. Artificial intelligence framework for simulating clinical decisionmaking: A markov decision process approach. _Artificial intelligence in medicine_, 57:9–19, 2013.


Kathryn Chaloner and Isabella Verdinelli. Bayesian experimental design: A review. _Statistical_
_science_, pp. 273–304, 1995.


Jacky Chen, Yunsie Chung, Jonathan Tynan, Chen Cheng, Song Yang, and Alan Cheng. Performance insights for small molecule drug discovery models: data scaling, multitasking, and generalization, 2024. Preprint.


Thomas G Dietterich. Ensemble methods in machine learning. In _International workshop on multi-_
_ple classifier systems_, pp. 1–15. Springer, 2000.


Maryam Eghbali-Zarch, Reza Tavakkoli-Moghaddam, Fatemeh Esfahanian, Amir Azaron, and Mohammad Mehdi Sepehri. A markov decision process for modeling adverse drug reactions in
medication treatment of type 2 diabetes. _Proceedings of the Institution of Mechanical Engineers,_
_Part H: Journal of Engineering in Medicine_, 233(8):793–811, 2019.


Mahdi M Fard, Sandor Szalma, Shashikant Vattikuti, and Gyan Bhanot. A bayesian markov decision
process framework for optimal decision making in clinical trials. _IEEE Journal of Biomedical and_
_Health Informatics_, 22(6):2061–2068, 2018.


Mohammad Ghavamzadeh, Shie Mannor, Joelle Pineau, and Aviv Tamar. Bayesian reinforcement
learning: A survey. _Foundations and Trends® in Machine Learning_, 8(5-6):359–483, 2015.


Rafael G´omez-Bombarelli, Jennifer N Wei, David Duvenaud, Jos´e Miguel Hern´andez-Lobato,
Benjam´ın S´anchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D Hirzel,
Ryan P Adams, and Al´an Aspuru-Guzik. Automatic chemical design using a data-driven continuous representation of molecules. _ACS central science_, 4(2):268–276, 2018.


Ryan-Rhys Griffiths and Jos´e Miguel Hern´andez-Lobato. Constrained bayesian optimization for
automatic chemical design using variational autoencoders. _Chemical_ _science_, 11(2):577–586,
2020.


Desislava R Ivanova, Adam Foster, Simon Kleinegesse, Michael U Gutmann, and Tom Rainforth.
Implicit deep adaptive design: Policy-based experimental design without likelihoods. In _Advances_
_in Neural Information Processing Systems_, volume 34, pp. 25785–25798, 2021.


Leslie Pack Kaelbling, Michael L Littman, and Anthony R Cassandra. Planning and acting in
partially observable stochastic domains. _Artificial intelligence_, 101(1-2):99–134, 1998.


Lukasz Kaiser, Mohammad Babaeizadeh, Piotr Milos, Blazej Osinski, Roy H Campbell, Krzysztof
Czechowski, Dumitru Erhan, Chelsea Finn, Patryk Kozakowski, Sergey Levine, et al. Modelbased reinforcement learning for atari. _arXiv preprint arXiv:1903.00374_, 2019.


Steven Kleinegesse and Michael U Gutmann. Bayesian experimental design for implicit models
by mutual information neural estimation. In _International_ _conference_ _on_ _machine_ _learning_, pp.
5316–5326. PMLR, 2020.


Steven Kleinegesse and Michael U Gutmann. Gradient-based bayesian experimental design for
implicit models using mutual information lower bounds. _arXiv preprint arXiv:2105.04379_, 2021.


12


Branislav Kveton and Georgios Theocharous. Kernel-based reinforcement learning on representative states. In _Proceedings_ _of_ _the_ _AAAI_ _Conference_ _on_ _Artificial_ _Intelligence_, number 1, pp.
977–983, 2012.


Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive
uncertainty estimation using deep ensembles. _Advances in neural information processing systems_,
30, 2017.


Thomas M Moerland, Joost Broekens, Aske Plaat, and Catholijn M Jonker. Model-based reinforcement learning: A survey. _Foundations and Trends® in Machine Learning_, 16(1):1–118, 2023.


Dirk Ormoneit and Saunak Sen. [´] Kernel-based reinforcement learning. _Machine learning_, 49:161–
178, 2002.


Ian Osband, Daniel Russo, and Benjamin Van Roy. More effective reinforcement learning via posterior sampling. In _Advances in neural information processing systems_, volume 26, 2013.


Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin Van Roy. Deep exploration via
bootstrapped dqn. _Advances in neural information processing systems_, 29, 2016.


Martin L Puterman. _Markov_ _decision_ _processes:_ _discrete_ _stochastic_ _dynamic_ _programming_ . John
Wiley & Sons, 2014.


Tom Rainforth, Adam Foster, Desi R Ivanova, and Freddie Bickford Smith. Modern bayesian experimental design. _Statistical Science_, 39:100–114, 2024.


Marwin HS Segler, Mike Preuss, and Mark P Waller. Planning chemical syntheses with deep neural
networks and symbolic ai. _Nature_, 555(7698):604–610, 2018.


Richard S Sutton and Andrew G Barto. _Reinforcement learning:_ _An introduction_ . MIT press, 2018.


Jike Wang, Chang-Yu Hsieh, Mingyang Wang, Xiaorui Wang, Zhenxing Wu, Dejun Jiang, Benben
Liao, Xujun Zhang, Bo Yang, Qiaojun He, et al. Multi-constraint molecular generation based
on conditional transformer, knowledge distillation and reinforcement learning. _Nature Machine_
_Intelligence_, 3(10):914–922, 2021.


Marco A Wiering and Hado Van Hasselt. Ensemble algorithms in reinforcement learning. _IEEE_
_Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics)_, 38(4):930–936, 2008.


Xin Xu, Dewen Hu, and Xicheng Lu. Kernel-based least squares policy iteration for reinforcement
learning. _IEEE transactions on neural networks_, 18(4):973–992, 2007.


Jiaxuan You, Bowen Liu, Zhitao Ying, Vijay Pande, and Jure Leskovec. Graph convolutional policy
network for goal-directed molecular graph generation. _Advances in neural information processing_
_systems_, 31, 2018.


Zhenpeng Zhou, Steven Kearnes, Li Li, Richard N Zare, and Patrick Riley. Optimization of
molecules via deep reinforcement learning. _Scientific reports_, 9(1):10752, 2019.


Zhi-Hua Zhou. _Ensemble methods:_ _foundations and algorithms_ . CRC press, 2012.


13


# Appendix


A THEORETICAL FRAMEWORK: IB-MDP AS A POMDP


This appendix provides a formal conceptual grounding for the IB-MDP framework. We demonstrate
that our similarity-weighted, case-based approach is not an ad-hoc heuristic, but rather a computationally tractable implementation of Bayesian belief updating within a Partially Observable Markov
Decision Process (POMDP) tailored for information-gathering problems.


A.1 POMDP PRELIMINARIES


A POMDP is formally defined by the tuple ( _S, A,_ Ω _, P, O, R, γ_ ), where _S_ is a set of hidden states,
_A_ is the set of actions, and Ω is the set of observations. Since the agent cannot observe the true
state _s ∈S_, it maintains a belief state, _bt_ ( _s_ ), which is a probability distribution over _S_ . After taking
action _At_ and receiving observation _ωt_, the belief is updated via the Bayes filter:

_bt_ +1( _s_ _[′]_ ) _∝_ _O_ ( _ωt_ _| s_ _[′]_ _, At_ )              - _P_ ( _s_ _[′]_ _| s, At_ ) _bt_ ( _s_ ) _._ (8)


_s∈S_


**The** **Information-Gathering** **Case.** The sequential assay planning task is an instance of an
_information-gathering_ problem. The underlying intrinsic properties of the candidate compound _x⋆_
are fixed; performing an assay reveals information about these properties but does not change them.
This corresponds to a static latent state, where the transition probability is an identity function:
_P_ ( _s_ _[′]_ _| s, At_ ) = **1** [ _s_ _[′]_ = _s_ ]. In this common special case, the belief update from Equation equation 8
simplifies to the multiplicative Bayes’ rule:
_bt_ +1( _s_ ) _∝_ _O_ ( _ωt_ _| s, At_ ) _bt_ ( _s_ ) _._ (9)


A.2 THE IB-MDP LATENT INDEX MODEL AND ITS POMDP CORRESPONDENCE


To map our framework to a POMDP, we introduce a discrete latent variable _Z_ _∈{_ 1 _, . . ., N_ _}_, where
each value _i_ corresponds to one of the historical records _Di_ _∈D_ . We treat _Z_ as the hidden state,
representing the ”true prototype” of our candidate compound _x⋆_ from among the known historical
cases. The core idea is that by maintaining a belief over _Z_, we are implicitly maintaining a belief
about the complete, unobserved profile of _x⋆_ . The explicit correspondence is detailed in Table 3.


A.3 EQUIVALENCE OF THE SIMILARITY UPDATE AND BAYESIAN FILTERING


With the mapping established, we now demonstrate that the similarity weight update mechanism in
IB-MDP is a direct implementation of the Bayesian belief update from Equation equation 9.


Let the prior belief over the latent index before step _t_ be the weights _wi_ ( _st_ ) _≡_ _P_ ( _Z_ = _i_ _|_ _st_ ).
Executing the assay batch _At_ yields the observation _ωt_ _≡{y⋆,j}j∈At_ . By substituting the IB-MDP
analogs into Equation equation 9, we derive the IB-MDP belief update rule for the weights:

_p_ ( _ωt_ _| Z_ = _i, At_ ) _wi_ ( _st_ )
_wi_ ( _st_ +1) =            - _Nℓ_ =1 _[p]_ [(] _[ω][t]_ _[|][ Z]_ [=] _[ ℓ, A][t]_ [)] _[ w][ℓ]_ [(] _[s][t]_ [)] _._ (10)

This confirms that the evolution of weights in IB-MDP is a principled Bayesian recursion.


**Connecting the Likelihood to the Similarity Kernel.** The final step is to show that our specific
implementation of similarity weights corresponds to a valid probabilistic likelihood model. If we
model the likelihood of observing an assay outcome _ya_ for the candidate with a Gaussian kernel
centered on the historical value _yi,a_ :


and assume conditional independence of assays in a batch given the prototype _Z_ (a modeling assumption that enables tractable inference; while biochemical assays may exhibit correlations even
given compound properties, our empirical results demonstrate robustness to violations of this assumption through the ensemble averaging mechanism), the joint likelihood for the observation _ωt_
is the product of individual likelihoods. Applying the Bayesian update recursively from a uniform


14


_[a]_ ( _ya −_ _yi,a_ ) [2]

2 _σ_ [2]


       _p_ ( _ya_ _| Z_ = _i, a_ ) _∝_ exp _−_ _[λ][a]_


_σa_ [2]


_,_


prior over all observed assays _{ya}a∈Mt_ yields a posterior over _Z_ that has the exact form of our
similarity weights:


where _δx_ ( _y_ ) denotes the Dirac delta measure that equals 1 if _y_ = _x_ and 0 otherwise. This confirms
that our sampling mechanism—drawing a historical case _Di_ according to the weights _wi_ ( _st_ ) and using its outcomes—is a principled way to sample from the posterior predictive distribution, allowing
the planner to explore plausible future scenarios consistent with all evidence gathered so far.


A.5 IMPLICATIONS AND SUMMARY


Framing IB-MDP within the POMDP context provides strong conceptual grounding and yields several key insights, summarized in Table 3.


    - **Justification** **for** **Dynamics** : The changing similarity weights observed _within_ an MCTS
simulation are not arbitrary non-stationarity. They represent the agent’s evolving belief
state. As information is gathered (the state _st_ is augmented), the model used for subsequent
predictions naturally and correctly changes, reflecting a refined belief.


    - **Suitability** **of** **MCTS** : MCTS is well-suited for this task because it is a simulation-based
planner designed to handle complex state spaces. It effectively explores the consequences
of actions on the future belief state and its associated rewards without needing an explicit
representation of the belief space itself.


    - **Principled** **Approximation** : IB-MDP provides a practical, data-driven approximation to
solving a formal POMDP. Its effectiveness relies on two key assumptions: the quality and
coverage of the historical data _D_, and the appropriateness of the chosen kernel (e.g., Gaussian) for the observation likelihood model. While the Gaussian kernel provides computational tractability and aligns with common assumptions about measurement noise in biochemical assays, we acknowledge that alternative kernels (e.g., Laplacian, Student-t) may
better capture heavy-tailed distributions or outliers. The robustness of our approach to
kernel choice remains an important area for future empirical validation.


    - **Computational** **Efficiency** : By representing the belief state implicitly through weights
over the case-base _D_, IB-MDP avoids the intractable calculations of maintaining and updating an explicit probability distribution over a potentially vast hidden state space.


In conclusion, interpreting IB-MDP as an approximate POMDP framework clarifies that the recalculation of similarity weights is a direct implementation of Bayesian belief updating. This justifies our
methodology and the use of MCTS for principled planning under uncertainty when only historical
data is available.


15


_wi_ ( _st_ ) _∝_ - _p_ ( _ya_ _| Z_ = _i, a_ ) = exp


_a∈Mt_


_−_ [1]


_≡_ exp� _−_ _λw d_ ( _st, Di_ )� _,_


2


- ( _ya −_ _yi,a_ ) [2]

_λa_
_a∈Mt_ _σa_ [2]


_σa_ [2]


where we can identify the global temperature parameter _λw_ = _β/_ 2 where _β_ is the inverse temperature of the tempered posterior. With _β_ = 1 (standard posterior), we have _λw_ = 1 _/_ 2. Therefore,
our similarity function is not an arbitrary heuristic but corresponds to a tempered Bayesian posterior
over the latent historical prototypes.


A.4 THE POSTERIOR PREDICTIVE TRANSITION MODEL


The belief state (the weight vector _w_ ( _st_ )) is used for planning. The transition model used to simulate
future trajectories within the MCTS planner is derived by marginalizing over the uncertainty in the
latent variable _Z_ . The probability of transitioning to a next state _st_ +1 is the posterior predictive
distribution over outcomes, conditioned on the current belief:


_P_ ( _st_ +1 _| st, At_ ) =


=


_N_

- _P_ ( _st_ +1 _| Z_ = _i, st, At_ ) _P_ ( _Z_ = _i | st_ ) (11)


_i_ =1


_N_

- _wi_ ( _st_ ) _δ st⊕{_ ( _aj_ _,yi,j_ ) _}j∈At_ ( _st_ +1) _,_ (12)

_i_ =1


Table 3: Summary of the Conceptual Mapping between POMDP and IB-MDP.
**POMDP Component** **IB-MDP Conceptual Equivalent** **Notes**


Hidden State ( _s_ _∈S_ ) Latent Index _Z_ = _i_ over historical cases _Di_ _∈D_ The ”true” but unknown profile of
the candidate.
Belief State ( _bt_ ( _s_ )) Similarity weights _wi_ ( _st_ ) _≡_ _P_ ( _Z_ = _i_ _|_ _st_ ) A probability distribution over possible prototypes.
Action ( _At_ _∈At_ ) Batch of assays to perform, _At_ _⊆_ _Ut_ Direct equivalence.
Observation ( _ωt_ ) Set of assay outcomes _{y⋆,j_ _}j∈At_ The new evidence gathered.
Observation Model ( _O_ ( _ωt|s_ _[′]_ _, At_ )) Likelihood _p_ ( _ωt_ _|_ _Z_ = _i, At_ ) Implemented via a similarity kernel.
Belief Update Recalculation of weights _wi_ ( _st_ +1) A direct, principled Bayesian update.


A.6 CONVERGENCE GUARANTEES AND THEORETICAL CONSIDERATIONS


The convergence properties of IBMDP differ fundamentally from traditional Bayesian reinforcement learning due to its unique reliance on historical data rather than environment interaction. This
subsection examines what convergence guarantees can and cannot be provided.


A.6.1 TRADITIONAL BAYESIAN RL GUARANTEES


Methods such as Posterior Sampling for Reinforcement Learning (PSRL) provide formal regret
_√_
bounds of _O_ [˜] ( _SAT_ ) for finite MDPs, where _S_ denotes states, _A_ denotes actions, and _T_ denotes

the horizon. These approaches guarantee PAC-style convergence to _ϵ_ -optimal policies with high
probability, leveraging the principle that posteriors concentrate on the true MDP as data accumulates.


A.6.2 IBMDP CONVERGENCE PROPERTIES


IBMDP’s convergence behavior is more nuanced due to its implicit model construction:


**Achievable** **Guarantees.** Standard MCTS with UCB1 provides asymptotic convergence to optimal policies as iterations approach infinity, assuming a fixed MDP model. For IBMDP specifically:


    - MCTS-DPW convergence: The Double Progressive Widening variant used in IBMDP
maintains convergence properties for large combinatorial action spaces


    - Linear case consistency: For synthetic data with linear relationships and independent features, we prove (Section D) that the similarity-based variance estimator converges in probability to the true conditional variance as _N_ _→∞_


    - Empirical robustness: The ensemble approach with majority voting provides stable recommendations, achieving 47% optimal policy alignment compared to 36% for deterministic
methods


**Fundamental Limitations.** Unlike traditional Bayesian RL, IBMDP cannot provide:


    - Formal regret bounds: The implicit model introduces approximation error bounded by historical data coverage rather than converging to true dynamics


    - PAC guarantees: Cannot ensure _ϵ_ -optimality with high probability due to dependence on
data representativeness


    - True dynamics recovery: The similarity-based model approximates but does not learn the
true transition function _P_ ( _s_ _[′]_ _|s, a_ )


The approximation quality depends on three key factors: (i) the coverage and representativeness of
historical data _D_, (ii) the appropriateness of the similarity metric for the domain, and (iii) the size
of the historical dataset _|D|_ . While formal convergence rates cannot be established without access
to the true dynamics, empirical validation demonstrates robust performance when these factors are
satisfied.


16


**Convergence Trade-offs.** IBMDP trades formal convergence guarantees for practical applicability. Under assumptions of sufficient data coverage, appropriate similarity metrics, and regularity
conditions, as MCTS iterations _n_ itr _→∞_ and ensemble size _Ne_ _→∞_ :


_P_
_||π_ IBMDP _−_ _π_ empirical _[∗]_ _[||][∞]_ _−→_ 0 (13)
where _π_ empirical _[∗]_ [is] [the] [optimal] [policy] [for] [the] [empirical] [MDP] [induced] [by] _[D]_ [.] [However,] [the] [gap] [be-]
tween this empirical optimal and the true optimal policy depends on data quality and coverage—a
fundamental limitation when operating without simulators.


This trade-off is not a weakness but a necessary adaptation: IBMDP provides a principled solution where traditional methods with stronger guarantees cannot operate at all due to the absence of
environment interaction capabilities.


B IMPLEMENTATION AND ALGORITHMIC DETAILS


B.1 HYPERPARAMETER SELECTION METHODOLOGY


The performance of the IB-MDP framework depends on a set of key hyperparameters that govern the
similarity model, the MCTS planner, and the problem’s objective constraints. The values used in our
experiments were determined through a combination of established literature guidelines, empirical
testing on our specific dataset, and domain-specific considerations to balance decision quality with
computational feasibility.


**Similarity Model Parameters.** These parameters define the core of the implicit, generative transition model.


    - **Similarity** **Bandwidth** **(** _λw_ **):** This parameter controls the ”smoothness” of the similarity
function. From the theoretical derivation in Section A, _λw_ = _β/_ 2 where _β_ is the inverse
temperature. For the standard posterior ( _β_ = 1), we have _λw_ = 0 _._ 5. However, in practice,
we tested values in the range [0 _._ 5 _,_ 2 _._ 0] and found that _λw_ = 1 _._ 0 provided better empirical
performance for our dataset ( _|D|_ = 220 _, d_ = 6). This corresponds to a tempered posterior
with _β_ = 2 _._ 0, which was large enough to ensure locality (giving higher weight to truly
similar compounds) but small enough to draw support from a sufficient number of historical
examples to make robust predictions.


    - **Feature Weights (** _λk_ = 1 _._ 0 **for all** _k_ **):** These weights allow for emphasizing more or less
informative features in the distance calculation. As we lacked detailed prior information on
the relative reliability of the QSAR predictions and assay measurements, we set all weights
to be equal to avoid introducing subjective bias. This treats all known features as equally
important for determining similarity.


**Ensemble and Planner Parameters.** These parameters control the MCTS search algorithm and
the robustness of its final policy.


    - **Ensemble** **Size** **(** _Ne_ = 50 **):** To mitigate variance from the stochastic nature of both the
transition model and the MCTS planner, we use an ensemble of independent runs. We
tested sizes from 20 to 100 and found that _Ne_ = 50 provided a stable policy recommendation (i.e., a consistent MLASP) without incurring excessive computational cost. Figure 3
illustrates how the ensemble’s majority vote leads to a robust action choice.


    - **MCTS Iterations (** _n_ **itr** = 20 _,_ 000 **):** This determines the search budget for each MCTS run.
Our analysis showed that policy recommendations stabilized around 20,000 iterations for
our problem’s complexity, with diminishing returns for higher values.


    - **Exploration Constant (** _c_ = 5 _._ 0 **):** Following standard practice for MCTS, this value balances exploration of new actions with exploitation of known high-value actions within the
search tree. A value of 5.0 provided effective exploration in our experiments.


17


Figure 3: Example histogram of actions proposed across an ensemble of _Ne_ = 50 runs. For a given
state with uncertainty _H_ ( _s_ ) = 0 _._ 2 and a likelihood constraint of _τ_ = 0 _._ 9, the action with the highest
frequency is selected for the MLASP. This demonstrates how the ensemble method produces robust
and stable recommendations via majority voting.


**Problem Constraint Parameters.** These parameters define the termination conditions and feasibility constraints of the planning problem itself.


    - **Terminal Uncertainty Threshold (** _ϵ_ = 0 _._ 10 **):** We stop when _H_ ( _sT_ ) drops below 0 _._ 10. In
our runs the initial uncertainty is between 0 _._ 2 and 0 _._ 6, so this threshold guarantees at least
a two- to six-fold reduction before declaring the policy sufficiently confident.


    - **Goal-Likelihood Threshold (** _τ_ _∈{_ 0 _._ 6 _,_ 0 _._ 9 _}_ **):** The threshold on the goal likelihood, _L_ ( _st_ ),
enforces that the planner only pursues trajectories that remain sufficiently likely to succeed.
We tested two values to explore the trade-off between cost and confidence. A lower value
( _τ_ = 0 _._ 6) permits more exploratory, potentially cheaper plans, while a higher value ( _τ_ =
0 _._ 9) enforces a more conservative and confident, but potentially more expensive, policy.
Figure 4 explicitly illustrates how a higher _τ_ leads to a different and more costly MLASP
to satisfy the stricter confidence requirement.


B.2 COMPUTATIONAL COMPLEXITY ANALYSIS


The computational complexity of the IB-MDP algorithm is a critical factor for its practical application. The worst-case total complexity is given by:
_O_ ( _Ne · n_ itr _·_ min( _b_ _[H]_ _, n_ itr) _· |D| · d_ ) _,_
where _Ne_ is the ensemble size, _n_ itr is the number of MCTS iterations, _b_ is the effective branching
factor (average number of actions explored per node), _H_ is the maximum tree depth (bounded by the
horizon _T_ ), _|D|_ is the number of historical cases, and _d_ is the dimensionality of the feature space.
The term min( _b_ _[H]_ _, n_ itr) represents the maximum number of nodes that can be expanded, bounded
either by the tree structure or the iteration budget. In practice, with progressive widening and UCT
selection, the effective number of expansions is often much smaller than this worst-case bound.


The dominant factor within a single MCTS simulation step is the calculation of the similarity
weights, which requires computing the distance from the current state to every historical case in
_D_ . This operation has a complexity of _O_ ( _|D| · d_ ) and is performed at each node expansion in the
search tree.


**Comparison to Alternatives.** This complexity, while significant, compares favorably to alternative approaches for principled planning under uncertainty. Exact POMDP solvers are computationally intractable for problems of this scale, as their complexity is exponential in the size of the belief
space. Traditional value iteration would require discretizing the state space, which becomes in

18


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


Figure 4: Comparison of MLASP paths for the same compound under two different goal-likelihood
thresholds: _τ_ = 0 _._ 6 (blue) and _τ_ = 0 _._ 9 (red). The stricter constraint ( _τ_ = 0 _._ 9) forces the planner
to recommend a more expensive sequence of assays to achieve higher confidence, illustrating the
direct trade-off between cost and decision confidence controlled by this parameter.


feasible with a growing number of continuous-valued assays. IB-MDP’s sampling-based approach
effectively navigates this high-dimensional space without requiring explicit enumeration.


**Practical** **Performance** **and** **Scalability.** In our experimental setup ( _Ne_ = 50, _n_ itr = 20 _,_ 000,
_|D|_ = 220, _d_ = 6), the total time to generate a policy for a single compound was approximately
one hour on an Apple M1 Pro chip with 16GB of memory. The algorithm’s complexity scales
linearly with the size of the historical dataset ( _|D|_ ), the feature dimension ( _d_ ), and the number of
ensemble runs ( _Ne_ ). This predictable scaling suggests that the method remains computationally
feasible for the larger datasets and higher-dimensional problems typically encountered in real-world
drug discovery campaigns, especially with access to parallel computing resources.


C FRAMEWORK DIFFERENTIATION AND THE UNFAIRNESS OF DIRECT
COMPARISON


C.1 KEY DIFFERENTIATING FEATURES


Table 4 summarizes the fundamental distinctions between IBMDP and traditional reinforcement
learning frameworks. These differences stem from IBMDP’s unique design for sequential experimental planning in simulator-free, data-rich environments—a problem class that existing methods
cannot address without fundamental restructuring.


C.2 THE FUNDAMENTAL INNOVATION


IBMDP operates in an entirely different problem setting from traditional reinforcement learning.
While conventional frameworks assume access to either environment simulators or transition data,
IBMDP functions with only a static database of historical experimental outcomes. This constraint,
common in drug discovery where mechanistic models are unavailable and experiments are irreversible, necessitates a fundamentally different approach.


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


~~Table 4:~~ ~~Fundamental distinctions between IBMDP and traditional RL frameworks~~
**Aspect** **Traditional Frameworks** **IBMDP**

**Data Requirements** ( _s, a, s_ _[′]_ ) tuples or simulator Static historical outcomes only
**Transition Model** Learned explicit _P_ ( _s_ _[′]_ _|s, a_ ) Implicit via similarity sampling
**Belief Representation** Explicit probability distributions Similarity weights _wi_ ( _st_ )
**Planning Method** Single policy or parametric Ensemble MCTS with majority voting
**Constraints** Cumulative: [�] _t_ _[c][t]_ _[≤]_ _[C]_ State-based: _H_ ( _sT_ ) _≤_ _ϵ_, _L_ ( _st_ ) _≥_ _τ_

**Action Space** Parameter optimization Combinatorial assay selection
**Action Effect** Changes underlying state Reveals fixed properties
**Correlation Handling** Requires explicit modeling Preserves empirically via sampling
**Objective** Single reward maximization Multi-objective optimization


The core mechanism constructs an implicit generative model through similarity-weighted sampling:
_wi_ ( _st_ ) = exp ( _−λw · d_ ( _st, Di_ )) (14)


This mechanism generates plausible transitions by sampling from historical cases most similar to the
current experimental state, thereby preserving the natural correlations between assays observed in
real compounds—correlations that would be difficult or impossible to model explicitly given current
scientific understanding.


C.3 COMPARISON WITH EXISTING FRAMEWORK CATEGORIES


C.3.1 DISTINCTION FROM MDPS AND MODEL-BASED RL


Model-based reinforcement learning fundamentally relies on learning transition dynamics from
( _s, a, s_ _[′]_ ) tuples, typically through parametric models that approximate _P_ ( _s_ _[′]_ _|s, a_ ). Even kernelbased RL methods, which employ similarity metrics, use them primarily for value function approximation rather than transition generation.


IBMDP diverges by using similarity not as a smoothing mechanism but as the foundation for a
complete generative process. Without access to any transition data, it samples entire assay outcome
profiles from historical compounds, weighted by their relevance to the current state. This nonparametric approach sidesteps the need for explicit dynamics modeling while naturally preserving
cross-assay dependencies present in the historical data.


C.3.2 DISTINCTION FROM BAYESIAN METHODS


Bayesian reinforcement learning and Bayesian optimization maintain explicit posterior distributions—over model parameters in BRL (exemplified by PSRL) or over objective functions in BO.
These methods either sample complete MDPs from parameter posteriors or perform myopic singlestep optimization.


IBMDP performs what we term Bayesian case-based generation: the similarity weights serve as an
implicit posterior over historical compound prototypes, updated through reweighting as evidence
accumulates. Unlike BO’s single-step focus, IBMDP enables multi-horizon planning that simultaneously considers experimental costs, time constraints, and the probability of achieving desired
outcomes—a multi-objective optimization fundamentally different from traditional Bayesian approaches.


C.3.3 DISTINCTION FROM EXPERIMENTAL DESIGN


Classical Bayesian experimental design assumes availability of a likelihood function or simulator,
optimizing for immediate information gain. Even implicit-BED methods for intractable likelihoods
rely on information-theoretic surrogates that assume some form of generative model.


IBMDP embeds its implicit model directly within a reinforcement learning planner (MCTS-DPW),
optimizing entire experimental sequences rather than individual experiments. The framework’s
state-uncertainty functional _H_ ( _st_ ) and goal-likelihood functional _L_ ( _st_ ) provide interpretable,


20


_P_ ( _st_ +1 _|st, At_ ) =


_N_


_i_ =1


_wi_ ( _st_ )

_Z_ _·_ **1** [ _st_ +1 = _st ⊕{_ ( _aj, yi,j_ ) _}j∈At_ ] (15)


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


domain-specific measures that directly relate to experimental objectives, unlike abstract informationtheoretic quantities.


C.3.4 DISTINCTION FROM POMDPS


Standard POMDP formulations maintain explicit belief distributions over hidden states, requiring
specification of both transition models _P_ ( _s_ _[′]_ _|s, a_ ) and observation models _O_ ( _o|s, a_ ). The belief
update follows the Bayes filter, necessitating these explicit models.


IBMDP’s similarity-weighted posterior serves as an implicit belief representation, eliminating the
need for high-dimensional belief state maintenance. The framework’s constraints—terminal uncertainty _H_ ( _sT_ ) _≤_ _ϵ_ and per-step feasibility _L_ ( _st_ ) _≥_ _τ_ —target state properties rather than cumulative
quantities, directly encoding experimental requirements for decision confidence and trajectory viability.


C.4 WHY DIRECT BENCHMARKING IS FUNDAMENTALLY UNFAIR


The fundamental incompatibility between IBMDP and traditional frameworks makes direct benchmarking not merely challenging but inherently unfair—comparing methods designed for entirely
different problem settings and data availability.


C.4.1 INCOMPATIBLE PREREQUISITES


Traditional RL methods universally require either an environment simulator for generating transitions on demand or a collection of ( _s, a, s_ _[′]_ ) tuples for learning dynamics. IBMDP operates precisely
where these prerequisites are absent: only static historical compound profiles exist, with no mechanism to query counterfactual outcomes. Creating a simulator would require mechanistic understanding of biochemical interactions that current science lacks, while collecting transition data through
exhaustive experimentation defeats the very purpose of efficient planning.


C.4.2 FUNDAMENTAL STRUCTURAL DIFFERENCES


The action spaces are categorically different. Traditional methods optimize over continuous or discrete parameter spaces where actions affect state transitions. IBMDP selects from combinatorial
sets of experimental assays— _P≤m_ ( _Ut_ ) _∪{_ eox _}_ —where actions reveal information about unchanging molecular properties. This distinction between control and information gathering necessitates
entirely different planning paradigms.


Furthermore, the constraint structures are incompatible. Traditional constrained MDPs limit cumulative costs across trajectories, while IBMDP enforces instantaneous feasibility constraints and
terminal uncertainty bounds that directly encode experimental requirements.


C.4.3 REQUIRED TRANSFORMATIONS


Adapting traditional methods to this setting would require:


1. Completely redefining the action space from parameter optimization to combinatorial assay
selection with batching constraints


2. Implementing reward-aware stopping rules aligned with uncertainty and feasibility functionals rather than simple cumulative objectives


3. Creating posterior predictive distributions without access to simulators or transition data


4. Restructuring from single-objective to multi-objective optimization with state-based constraints


Such extensive modifications would fundamentally alter the nature of these methods, creating essentially new algorithms rather than variants of existing ones. Any resulting comparison would be
between IBMDP and these newly created methods, not the original frameworks.


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


C.5 THEORETICAL FOUNDATION


Despite operating in this unique setting, IBMDP maintains rigorous theoretical grounding. Section A establishes that the framework is mathematically equivalent to a POMDP where the hidden state represents a latent index over historical cases. The similarity weights constitute a valid
Bayesian posterior, with weight updates implementing exact Bayesian belief updates. This equivalence:

_p_ ( _ωt|Z_ = _i, At_ ) _· wi_ ( _st_ )
_wi_ ( _st_ +1) =            - _Nℓ_ =1 _[p]_ [(] _[ω][t][|][Z]_ [=] _[ ℓ, A][t]_ [)] _[ ·][ w][ℓ]_ [(] _[s][t]_ [)] (16)

provides principled justification for the empirical success observed in our experiments, where IBMDP achieved up to 92% reduction in resource consumption while maintaining decision quality.


C.6 IMPLICATIONS


IBMDP addresses a problem class—sequential experimental planning without simulators—that existing reinforcement learning frameworks were not designed to handle. The inherent unfairness of
direct benchmarking reflects not a limitation but the framework’s fundamental novelty operating in
a unique problem setting. By leveraging historical data through similarity-weighted sampling and
ensemble planning, IBMDP provides the first practical solution for case-guided sequential decisionmaking in drug discovery and similar experimental sciences where traditional RL assumptions fail
to hold.


D BENCHMARK WITH SYNTHETIC DATA


**Overview** **and** **Motivation.** This appendix presents a rigorous benchmark study comparing IBMDP against theoretically optimal and deterministic baselines using synthetic data. The synthetic
environment provides a unique advantage: we can compute the true optimal policy exactly, enabling
principled validation of our similarity-based planning approach. This controlled setting allows us to
isolate and evaluate the effectiveness of IBMDP’s core innovations—the similarity-weighted belief
mechanism and ensemble planning—against ground truth.


**Aim.** We provide a controlled benchmark to compare three planners for sequential assay selection: (i) a _theoretical_ Value Iteration baseline with _exact_ uncertainty dynamics ( **VI-Theo** ); (ii) a
_deterministic_ Value Iteration with _similarity-based_ uncertainty ( **VI-Sim** ); and (iii) the _stochastic_
IBMDP planner using _similarity-weighted_ _posterior_ _predictive_ transitions inside an ensemble of
MCTS-DPW trees ( **IBMDP** ). A synthetic data generator with known structure enables exact computation of the VI-Theo policy and a principled testbed for the two similarity-based planners.


**Notation** **consistency** **with** **the** **main** **text.** We maintain consistency with the notation from the
main exposition (state/action tuples, similarity weights, _H_ ( _s_ ), _L_ ( _s_ ), etc.; see Equations equation 1–
equation 3 and Table 8). Throughout this appendix, for notational convenience when the focus is
on the set structure rather than individual measurements, we may write the state as _s_ = ( _x⋆, M_ )
where _x⋆_ is the fixed candidate compound and _M_ _⊆{_ 1 _, . . ., M_ _}_ represents the set of measured
assays. This is equivalent to the main text notation _st_ = ( _x⋆, {y⋆,j}j∈Mt_ ) where _M_ = _Mt_ indexes
the measured assays and their values. When the candidate is fixed and clear from context, we may
write the state simply as _M_ .


D.1 SYNTHETIC DATA: GENERAL MODEL AND INSTANTIATION


**Purpose.** We construct a synthetic data environment where the true conditional variance can be
computed analytically, providing ground truth for evaluating our similarity-based estimators. The
linear structure with independent features represents a simplified but informative test case where
theoretical optimality is tractable.


D.1.1 GENERAL DATA-GENERATING PROCESS (GENERIC FORMULATION).


Fix integers _N_ (number of historical cases) and _M_ (number of assays/features). For each historical
case _i ∈{_ 1 _, . . ., N_ _}_ we draw a feature vector

**y** _i_ = ( _yi,_ 1 _, . . ., yi,M_ ) _[⊤]_ _∈_ R _[M]_ _._


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


For each assay _a_ _∈{_ 1 _, . . ., M_ _}_ specify distributional parameters ( _µa, σa, aa, ba_ ) and draw independently
_yi,a_ _∼T N_ ( _µa, σa_ ; _aa, ba_ ) _,_
the (univariate) truncated normal on [ _aa, ba_ ] with location _µa_ and scale _σa_ . [1] Let _β_ =
( _β_ 1 _, . . ., βM_ ) _[⊤]_ _∈_ R _[M]_ and draw independent measurement noise
_ϵi_ _∼T N_ ( _µϵ, σϵ_ ; _aϵ, bϵ_ ) _._
The scalar target is


The historical dataset is _D_ = _{_ ( _xi,_ **y** _i_ ) _}_ ; targets _G_ = _{gi}_ are stored separately (and _never_ used in
any distance/weight computation).


**Closed-form** **variance** **under** **independence.** Let Σ = diag( _σ_ 1 [2] _[, . . ., σ]_ _M_ [2] [)] [denote] [the] [per-assay]
variance parameters (treated as the empirical variances of the generated _yi,a_ ’s). Then

Var( _g_ ) = _β_ _[⊤]_ Σ _β_ + _σϵ_ [2] _[.]_
To derive the conditional variance, we partition assays into measured _M_ and unmeasured _U_ and
write _β_ = ( _βM, βU_ ), Σ = diag(Σ _M,_ Σ _U_ ). Under independence, we have the following derivation:

Var( _g_ _|_ **y** _M_ ) = Var� _βM_ _[⊤]_ **[y]** _[M]_ [+] _[ β]_ _U_ _[⊤]_ **[y]** _[U]_ [+] _[ ϵ][ |]_ **[ y]** _[M]_     - (17)

= Var� _βU_ _[⊤]_ **[y]** _[U]_ [+] _[ ϵ][ |]_ **[ y]** _[M]_          - (since _βM_ _[⊤]_ **[y]** _[M]_ [is fixed given] **[ y]** _[M]_ [)] (18)

= Var� _βU_ _[⊤]_ **[y]** _[U]_          - + Var( _ϵ_ ) (by independence of **y** _U_, _ϵ_ from **y** _M_ ) (19)

= _βU_ _[⊤]_ [Var(] **[y]** _[U]_ [)] _[β][U]_ [+] _[ σ]_ _ϵ_ [2] (20)

= _βU_ _[⊤]_ [Σ] _[U]_ _[β][U]_ [+] _[ σ]_ _ϵ_ [2] _[.]_ (21)
This identity is central to the VI-Theo derivation below.


D.1.2 INSTANTIATION USED IN THE BENCHMARK.


We set _M_ = 6 and _N_ = 200. For _a_ = 1 _, . . .,_ 6,

_µa_ = 50 _·_ _[a]_ _σa_ = 0 _._ 3 _µa,_ ( _aa, ba_ ) = (0 _,_ 2 _µa_ ) _,_

6 _[,]_

_β_ = (0 _._ 3 _,_ 0 _._ 25 _,_ 0 _._ 2 _,_ 0 _._ 15 _,_ 0 _._ 07 _,_ 0 _._ 03) _,_ _ϵ ∼T N_ (0 _,_ 5; _−_ 10 _,_ 10) _._
All feature draws are independent across _a_ and _i_ ; noise is independent of features.


**Sampling** **recipe** **(for** **reproducibility).** For each trial: (1) fix _M, N, {µa, σa, aa, ba}_ and _β_ ; (2)
draw _{_ **y** _i}_ _[N]_ _i_ =1 [componentwise; (3) draw] _[ {][ϵ][i][}]_ _i_ _[N]_ =1 [; (4) set] _[ g][i]_ [=] _[β][⊤]_ **[y]** _[i]_ [ +] _[ ϵ][i]_ [; (5) store] _[ D]_ [=] _[{]_ [(] _[x][i][,]_ **[ y]** _[i]_ [)] _[}]_
and _G_ = _{gi}_ with availability set _Ig_ = _{i_ : _gi_ used in evaluation _}_ .


D.2 THEORETICAL BASELINE (VI-THEO): FULL DERIVATION


**Overview.** VI-Theo represents the theoretically optimal policy under perfect information about the
data-generating process. It uses the exact conditional variance formula derived above to compute
optimal uncertainty reduction at each step. This baseline is only computable in synthetic settings
where the true model parameters are known.


**State, action, and dynamics.** The state is the set _M ⊆{_ 1 _, . . .,_ 6 _}_ of measured assays (consistent
with our notation convention). The action space is the power set of unmeasured assays:
_A_ ( _M_ ) = _P_          - _{_ 1 _, . . .,_ 6 _} \ M_          - _._

Transitions are deterministic: executing _a ∈_ _A_ ( _M_ ) yields _M ←M ∪_ _a_ .


1In our experiments we treat ( _µa, σa_ ) as the empirical mean/scale of the generated samples; the truncation
mildly perturbs the theoretical moments.


23


_gi_ =


_M_

- _βa yi,a_ + _ϵi_ = _β_ _[⊤]_ **y** _i_ + _ϵi._


_a_ =1


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


**Exact** **conditional** **variance** **and** **reduction.** Write _g_ = _βM_ _[⊤]_ **[y]** _[M]_ [+] _[ β]_ _U_ _[⊤]_ **[y]** _[U]_ [+] _[ ϵ]_ [.] [Conditioning] [on]
the realized measurements **y** _M_,
Var( _g_ _|_ **y** _M_ ) = Var� _βU_ _[⊤]_ **[y]** _[U]_ [+] _[ ϵ]_          - = _βU_ _[⊤]_ [Σ] _[U]_ _[β][U]_ [+] _[ σ]_ _ϵ_ [2] _[,]_
since **y** _U_ is independent of **y** _M_ and _ϵ_ . Hence, the _exact_ uncertainty reduction achieved by measuring
a batch _a ⊆U_ is computed as follows:
∆ _σa_ [2] [= Var(] _[g]_ _[|]_ **[ y]** _[M]_ [)] _[ −]_ [Var(] _[g]_ _[|]_ **[ y]** _[M∪][a]_ [)] (22)

=        - _βU_ _[⊤]_ [Σ] _[U]_ _[β][U]_ [+] _[ σ]_ _ϵ_ [2]        - _−_        - _βU\_ _[⊤]_ _a_ [Σ] _[U\][a][β][U\][a]_ [+] _[ σ]_ _ϵ_ [2]        - (23)

= _βU_ _[⊤]_ [Σ] _[U]_ _[β][U]_ _[−]_ _[β]_ _U\_ _[⊤]_ _a_ [Σ] _[U\][a][β][U\][a]_ (24)

=        - _βk_ [2] _[σ]_ _k_ [2] _[.]_ (25)

_k∈a_

The last equality follows because Σ is diagonal and the contribution of each measured assay _k_ is
exactly _βk_ [2] _[σ]_ _k_ [2][.]


**Costs and reward.** Let per-assay costs be
_c_ 1 = 1 _._ 0 _,_ _c_ 2 = 1 _._ 2 _,_ _c_ 3 = 1 _._ 5 _,_ _c_ 4 = 1 _._ 8 _,_ _c_ 5 = 2 _._ 0 _,_ _c_ 6 = 2 _._ 2 _,_
and (optionally) a terminal target-measurement cost _c_ target = 10 _._ 0. The batch cost is _ca_ = [�] _k∈a_ _[c][k]_
and the immediate reward is _uncertainty reduction per unit cost_ :

_R_ ( _M, a_ ) = [∆] _[σ]_ _a_ [2] _._
_ca_


**Bellman recursion.** With discount _γ_ = 0 _._ 95,


Here _K_ ( _s_ ) are the known features (initial QSARs and any measured assays); _ϕk_ ( _·_ ) extracts feature
_k_ ; _λk_ are feature weights (default = 1), _λw_ _>_ 0 is the bandwidth, and _σk_ [2] [are the empirical variances]
over _D_ . _The targets {gi} are never used in d_ ( _·, ·_ ) _._


24


_V_ ( _M_ ) = max
_a∈A_ ( _M_ )


- _R_ ( _M, a_ ) + _γ V_ ( _M ∪_ _a_ ) _,_


initialized at _V_ ( _{_ 1 _, . . .,_ 6 _}_ ) = 0 (no uncertainty left, no action left). We iterate to a tolerance of
10 _[−]_ [6] or 1000 iterations to obtain the optimal policy _π_ Theo( _M_ ).


**Remarks on optimality.** Because transitions are deterministic and rewards are additive with discount, the recursion gives the exact optimal policy under the synthetic uncertainty model. This
policy serves as the _ground-truth baseline_ against which we compare similarity-based planners.


D.3 DETERMINISTIC SIMILARITY-BASED VI (VI-SIM): FULL DERIVATION


**Overview.** VI-Sim represents a deterministic planner that uses the same similarity-based variance
estimation as IBMDP but applies Value Iteration instead of stochastic tree search. This baseline
isolates the contribution of IBMDP’s ensemble planning approach by using the same implicit model
but with deterministic optimization.


**Similarity weights and distance.** At state _s_ = ( _x⋆, M_ ) (maintaining our consistent notation; here
we use _s_ to denote a generic state rather than _st_ for a specific time step), define:

exp _{−λw d_ ( _s, Di_ ) _}_
_wi_ ( _s_ ) =             - _N_ _,_ (26)
_j_ =1 [exp] _[{−][λ][w][ d]_ [(] _[s, D][j]_ [)] _[}]_


_d_ ( _s, Di_ ) = - _λk_


_k∈K_ ( _s_ )


- _ϕk_ ( _s_ ) _−_ _ϕk_ ( _Di_ )�2

_._ (27)
_σk_ [2]


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


**Weighted** _g_ **-mean** **and** **variance** **(renormalized** **over** _Ig_ **).** Define the renormalized weights and
weighted mean:

_wi_ ( _s_ )
_w_ ˜ _i_ ( _s_ ) =          - for _i ∈_ _Ig,_ (28)

_ℓ∈Ig_ _[w][ℓ]_ [(] _[s]_ [)]

_g_ ¯( _s_ ) =               - _w_ ˜ _i_ ( _s_ ) _gi._ (29)


_i∈Ig_


The estimated conditional variance at state _s_ is:
_σ_ �cond [2] [(] _[M]_ [) =]            - _w_ ˜ _i_ ( _s_ )            - _gi −_ _g_ ¯( _s_ )�2 _._ (30)

_i∈Ig_

After executing batch _a_, we update the observed state to _s_ _[′]_ = ( _x⋆, M∪a_ ), recompute weights based
on the expanded feature set, and obtain � _σ_ cond [2] [(] _[M ∪]_ _[a]_ [)][.]


**Estimated reduction and reward.** Using the similarity-weighted variance estimate, we define the
variance reduction and reward:
∆ _σ_         - _a_ [2] [=] _[σ]_ [�] cond [2] [(] _[M]_ [)] _[ −]_ _[σ]_ [�] cond [2] [(] _[M ∪]_ _[a]_ [)] _[,]_ (31)

_R_ ( _M, a_ ) = [∆] _[σ]_ [�] _a_ [2] _._ (32)
_ca_

We then apply this _R_ to the same deterministic VI recursion as in Section D.2 to obtain the policy
_π_ Sim.


D.4 IBMDP: IMPLICIT POSTERIOR-PREDICTIVE MODEL AND PLANNING DETAILS


**Overview.** IBMDP extends the similarity-based approach with two key innovations: (1) a stochastic posterior-predictive model that samples from historical cases weighted by similarity, and (2) ensemble MCTS planning that explores multiple policy trajectories. This combination enables robust
planning despite the implicit model’s inherent uncertainty.


**Latent-index view and likelihood.** Introduce a discrete latent index _Z_ _∈{_ 1 _, . . ., N_ _}_ over historical cases _Di_ . Given _Z_ = _i_ and selecting assay _a_, a Gaussian discrepancy model leads to:


With a uniform prior over _Z_ and temperature _λw_, the similarity weight equals a tempered posterior

exp _{−λw d_ ( _st, Di_ ) _}_
_wi_ ( _st_ ) =             - _N_ _._
_j_ =1 [exp] _[{−][λ][w][ d]_ [(] _[s][t][, D][j]_ [)] _[}]_

If we then measure assay _at_ and observe _yat_, the distance updates additively:

_d_ ( _st_ +1 _, Di_ ) = _d_ ( _st, Di_ ) + _[λ]_ _σa_ _[a]_ [2] _t_ _[t]_ ( _yat_ _−_ _yi,at_ ) [2] _._ (35)

This yields the multiplicative weight update:

_wi_ ( _st_ +1) _∝_ _wi_ ( _st_ ) _·_ exp� _−λw_ _λσaa_ [2] _tt_ ( _yat_ _−_ _yi,at_ ) [2]       - (36)

_∝_ _wi_ ( _st_ ) _·_        - _p_ ( _yat_ _| Z_ = _i, at_ )�2 _λw_ _,_ where _λw_ = _β/_ 2 _,_ (37)


25


       _p_ ( _ya_ _| Z_ = _i, a_ ) _∝_ exp _−_ _[λ][a]_ [(] _[y][a][ −]_ _[y][i,a]_ [)][2]

2 _σa_ [2]


_,_ (33)


with per-assay weight _λa_ _>_ 0. For a batch _At_ and assuming conditional independence across assays
given _Z_ (see discussion in Section A regarding this assumption):


   
- exp _−_ _[λ][a]_ [(] _[y][a][ −]_ _[y][i,a]_ [)][2]

_a∈At_ 2 _σa_ [2]


_p_ ( **y** _At_ _| Z_ = _i, At_ ) _∝_ 


2 _σa_ [2]


_._ (34)


**Weights as (tempered) posteriors and incremental recursion.** Let _Ot_ denote the set of observed
assays up to time _t_ . Define the variance-normalized distance


_d_ ( _st, Di_ ) = 

( _a,ya_ ) _∈Ot_


_λa_
( _ya −_ _yi,a_ ) [2] _._
_σa_ [2]


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


followed by normalization. Thus the reweighting is a (tempered) Bayesian belief update.


**Posterior** **predictive** **(implicit** **transition).** Marginalizing over _Z_ gives the posterior predictive
over next information states:


which is implemented operationally by sampling a single historical case _i_ _∼_ Cat( _{wk_ ( _st_ ) _}_ ) and
_copying_ the batch outcomes _{yi,a}a∈At_ into the candidate—thereby preserving cross-assay correlation within the sampled historical profile.


**Planning with MCTS-DPW and ensembling.** Within each MCTS rollout, we generate stochastic
next states by the posterior predictive above, accrue step cost _R_ ( _st, At_ ), and apply penalties when
feasibility is violated (e.g., _L_ ( _s_ ) _<_ _τ_ at any step) until a terminal state (e.g., _H_ ( _s_ ) _≤_ _ϵ_ ) or horizon
_T_ . To reduce variance from stochastic sampling and tree search, we run _Ne_ independent trees and
aggregate recommendations by majority vote, reporting both the _Top-1_ action and the _Top-2_ action
set at each decision step, forming an MLASP.


D.5 ILLUSTRATIVE EXAMPLE: EVOLUTION OF SIMILARITY WEIGHTS


**Purpose.** This toy example demonstrates how similarity weights evolve as evidence accumulates,
providing intuition for the adaptive nature of IBMDP’s implicit dynamics.


**Setup.** Three historical records with one feature each at values _{_ 0 _,_ 1 _,_ 2 _}_ ; let _σ_ [2] = 1, _λw_ = _λ_ 1 = 1.
The initial candidate state is _s_ [(0)] with value 1 _._ 1.


**Step 0 (initial).** Raw weights:

_w_ 1 = _e_ _[−]_ [(1] _[.]_ [1] _[−]_ [0)][2] = _e_ _[−]_ [1] _[.]_ [21] = 0 _._ 297 _,_ (39)

_w_ 2 = _e_ _[−]_ [(1] _[.]_ [1] _[−]_ [1)][2] = _e_ _[−]_ [0] _[.]_ [01] = 0 _._ 990 _,_ (40)

_w_ 3 = _e_ _[−]_ [(1] _[.]_ [1] _[−]_ [2)][2] = _e_ _[−]_ [0] _[.]_ [81] = 0 _._ 445 _._ (41)
After normalization _Z_ = 0 _._ 297 + 0 _._ 990 + 0 _._ 445 = 1 _._ 732, we obtain:
_w_ ˜ = (0 _._ 171 _,_ 0 _._ 571 _,_ 0 _._ 257) _._ (42)


**Step 1 (after action moves state to 1.6).** Raw weights:

_w_ 1 = _e_ _[−]_ [(1] _[.]_ [6] _[−]_ [0)][2] = _e_ _[−]_ [2] _[.]_ [56] = 0 _._ 077 _,_ (43)

_w_ 2 = _e_ _[−]_ [(1] _[.]_ [6] _[−]_ [1)][2] = _e_ _[−]_ [0] _[.]_ [36] = 0 _._ 697 _,_ (44)

_w_ 3 = _e_ _[−]_ [(1] _[.]_ [6] _[−]_ [2)][2] = _e_ _[−]_ [0] _[.]_ [16] = 0 _._ 852 _._ (45)
Normalizing with _Z_ _[′]_ = 1 _._ 626 gives:
_w_ ˜ = (0 _._ 047 _,_ 0 _._ 429 _,_ 0 _._ 524) _._ (46)
The posterior shifts toward the historical record at 2 as evidence moves rightward.


D.6 THEORETICAL ANALYSIS: CONSISTENCY OF VI-SIM


**Theorem** **(Consistency** **of** **Similarity-Based** **Estimation).** Under the synthetic linear model with
independent features, the similarity-based variance estimator _σ_ �cond [2] [(] _[M]_ [)][ used by VI-Sim converges]

_P_
in probability to the exact conditional variance _σ_ cond [2] [(] _[M]_ [)] [used] [by] [VI-Theo,] [i.e.,] _[σ]_ [�] cond [2] [(] _[M]_ [)] _−→_
_σ_ cond [2] [(] _[M]_ [)][ as] _[ N]_ _[→∞]_ [.]


**Proof.**


**Assumptions.** (A1) Features _{yi,a}_ are independent across _a_ and i.i.d. across _i_, each with finite
variance _σa_ [2][; (A2) noise] _[ ϵ][i]_ [is independent of features with finite variance] _[ σ]_ _ϵ_ [2][; (A3) the weight func-]
tion _wi_ ( _s_ ) depends only on _measured_ assays _M_ of each record and on the candidate’s observed


26


_P_ ( _st_ +1 _| st, At_ ) =


_N_

- _wi_ ( _st_ ) _· δ_ - _st_ +1 = _st ⊕{_ ( _a, yi,a_ ) _}a∈At_ - _,_ (38)


_i_ =1


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


values at those assays; (A4) the renormalized weights _w_ ˜ _i_ ( _s_ ) over _Ig_ form a probability vector; (A5)
_Ig_ grows with _N_ so that _|Ig| →∞_ .


**Step 1:** **Setup and notation.** Fix a state _s_ = ( _x⋆, M_ ) with measured set _M_ and unmeasured set
_U_ = _{_ 1 _, . . ., M_ _} \ M_ . Define the target for record _i_ :

_gi_ = _βM_ _[⊤]_ **[y]** _[i,][M]_ [+] _[ β]_ _U_ _[⊤]_ **[y]** _[i,][U]_ [+] _[ ϵ][i][.]_
The exact conditional variance (under the model) is:
_σ_ cond [2] [(] _[M]_ [) =] _[ β]_ _U_ _[⊤]_ [Σ] _[U]_ _[β][U]_ [+] _[ σ]_ _ϵ_ [2] _[.]_
The estimator used by VI-Sim at _s_ is:


**Step** **4:** **Simplification** **of** **the** **second** **term.** The second sum simplifies because _{zi}_ are i.i.d.,
mean zero and independent of the weights:
E�( _zi −_ _z_ ¯ _w_ ) [2][ ��] _{_ **y** _i,M}_     - = E[ _zi_ [2][] +][ E][[¯] _[z]_ _w_ [2] []] _[ −]_ [2][E][[] _[z][i][z]_ [¯] _[w]_ []] (expanding the square) (53)

= Var( _zi_ ) + Var(¯ _zw_ ) _−_ 2 Cov( _zi,_ ¯ _zw_ ) (54)

= Var( _zi_ ) + Var( _zi_ )               - _w_ ˜ _j_ ( _s_ ) [2] _−_ 2 ˜ _wi_ ( _s_ )Var( _zi_ ) (55)


_j∈Ig_

= Var( _zi_ ) (1 +               - _w_ ˜ _j_ ( _s_ ) [2] _−_ 2 ˜ _wi_ ( _s_ )) _,_ (56)


_j∈Ig_


27


- _w_ ˜ _i_ ( _s_ ) - _gi −_ 
_i∈Ig_ _j∈Ig_


_σ_ �cond [2] [(] _[M]_ [) =] 


- _w_ ˜ _j_ ( _s_ ) _gj_ �2 _._


_j∈Ig_


**Step** **2:** **Decomposition** **via** **independence.** We decompose each target as _gi_ =
_βM_ _[⊤]_ **[y]** _[i,][M]_ + _zi_, where

 - �� - ����
measured term unmeasured term

_zi_ := _βU_ _[⊤]_ **[y]** _[i,][U]_ [+] _[ ϵ][i][.]_
By assumptions (A1)–(A2) on independence, _zi_ is independent of **y** _i,M_ and thus independent of any
measurable function of **y** _i,M_, including _wi_ ( _s_ ) and _w_ ˜ _i_ ( _s_ ). This yields:

E( _zi |_ **y** _i,M_ ) = E( _zi_ ) = E( _βU_ _[⊤]_ **[y]** _[i,][U]_ [) +][ E][(] _[ϵ][i]_ [) = 0] _[,]_ (47)

Var( _zi |_ **y** _i,M_ ) = Var( _zi_ ) = Var( _βU_ _[⊤]_ **[y]** _[i,][U]_ [) + Var(] _[ϵ][i]_ [)] (48)

= _βU_ _[⊤]_ [Σ] _[U]_ _[β][U]_ [+] _[ σ]_ _ϵ_ [2] _[.]_ (49)


+ _zi_
����
unmeasured term


, where


**Step** **3:** **Analysis** **of** **the** **weighted** **variance** **estimator.** Define the weighted means: _g_ ¯ _w_ :=

- _i∈Ig_ _[w]_ [˜] _[i]_ [(] _[s]_ [)] _[g][i]_ [,] _[y]_ [¯] _[M][,w]_ [:=] [�] _i∈Ig_ _[w]_ [˜] _[i]_ [(] _[s]_ [)] _[ β]_ _M_ _[⊤]_ **[y]** _[i,][M]_ [,] [and] _[z]_ [¯] _[w]_ [:=] [�] _i∈Ig_ _[w]_ [˜] _[i]_ [(] _[s]_ [)] _[z][i]_ [.] [Then] [the] [variance]

estimator becomes:


_i∈Ig_ _[w]_ [˜] _[i]_ [(] _[s]_ [)] _[ β]_ _M_ _[⊤]_ **[y]** _[i,][M]_ [,] [and] _[z]_ [¯] _[w]_ [:=] [�]


_i∈Ig_ _[w]_ [˜] _[i]_ [(] _[s]_ [)] _[g][i]_ [,] _[y]_ [¯] _[M][,w]_ [:=] [�]


_σ_ �cond [2] [(] _[M]_ [) =] - _w_ ˜ _i_ ( _s_ ) - _gi −_ _g_ ¯ _w_ �2 (50)

_i∈Ig_


- _w_ ˜ _i_ ( _s_ ) - _gi −_ _g_ ¯ _w_ �2 (50)

_i∈Ig_

- _w_ ˜ _i_ ( _s_ ) �( _βM_ _[⊤]_ **[y]** _[i,][M]_ [+] _[ z][i]_ [)] _[ −]_ [(¯] _[y][M][,w]_ [+ ¯] _[z][w]_ [)] �2 (51)

_i∈Ig_

- _w_ ˜ _i_ ( _s_ ) - _βM_ _[⊤]_ **[y]** _[i,][M]_ _[−]_ _[y]_ [¯] _[M][,w]_ [+] _[ z][i]_ _[−]_ _[z]_ [¯] _[w]_ �2 _._ (52)

_i∈Ig_


= - _w_ ˜ _i_ ( _s_ ) �( _βM_ _[⊤]_ **[y]** _[i,][M]_ [+] _[ z][i]_ [)] _[ −]_ [(¯] _[y][M][,w]_ [+ ¯] _[z][w]_ [)] �2 (51)

_i∈Ig_

= - _w_ ˜ _i_ ( _s_ ) - _βM_ _[⊤]_ **[y]** _[i,][M]_ _[−]_ _[y]_ [¯] _[M][,w]_ [+] _[ z][i]_ _[−]_ _[z]_ [¯] _[w]_ �2 _._ (52)

_i∈Ig_


Taking expectation conditional on the _entire_ measured panel _{_ **y** _i,M}i∈Ig_ (which determines
_{w_ ˜ _i_ ( _s_ ) _}i∈Ig_ ), and using E( _zi|_ **y** _i,M_ ) = 0 with independence across _i_ :


- _w_ ˜ _i_ ( _s_ ) - _βM_ _[⊤]_ **[y]** _[i,][M]_ _[−]_ _[y]_ [¯] _[M][,w]_ �2 + 
_i∈Ig_ _i∈Ig_


E� _σ_ �cond [2] [(] _[M]_ [)] �� _{_ **y** _i,M}_ - = 


- _w_ ˜ _i_ ( _s_ ) E�( _zi −_ _z_ ¯ _w_ ) [2][ ��] _{_ **y** _i,M}_ - _._

_i∈Ig_


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


Specifically, if max _i_ _w_ ˜ _i_ ( _s_ ) _−→_ 0, then [�] _i_ _[w]_ [˜] _[i]_ [(] _[s]_ [)] _[≤]_ [max] _[i]_ _[w]_ [˜] _[i]_ [(] _[s]_ [)][ �] _i_ _[w]_ [˜] _[i]_ [(] _[s]_ [)] _[ →]_ [0][.] [This condition requires]

that the similarity kernel bandwidth is chosen such that as _N_ _→∞_, no single historical case dominates the
weights.


where we used Var(¯ _zw_ ) = Var( _zi_ ) [�] _j∈Ig_ _[w]_ [˜] _[j]_ [(] _[s]_ [)][2] [(by] [independence)] [and] [Cov(] _[z][i][,]_ [ ¯] _[z][w]_ [)] =

_w_ ˜ _i_ ( _s_ ) Var( _zi_ ). Therefore

- [2] - - [2]


where we used Var(¯ _zw_ ) = Var( _zi_ )

[�]


- _w_ ˜ _i_ ( _s_ ) E�( _zi −_ _z_ ¯ _w_ ) [2][ ��] _{_ **y** _i,M}_ - = Var( _zi_ ) 
_i∈Ig_ _i∈Ig_


- _w_ ˜ _i_ ( _s_ ) (1 + 

_i∈Ig_ _j∈Ig_


- _w_ ˜ _j_ ( _s_ ) [2] _−_ 2 ˜ _wi_ ( _s_ )) (57)


_j∈Ig_


��
= Var( _zi_ )


- _w_ ˜ _i_ ( _s_ ) 

_i∈Ig_ _j∈Ig_


_w_ ˜ _i_ ( _s_ ) +  
_i∈Ig_ _i∈Ig_


- _w_ ˜ _j_ ( _s_ ) [2] _−_ 2 

_j∈Ig_ _i∈Ig_


- _w_ ˜ _i_ ( _s_ ) [2][�]


_i∈Ig_


(58)


- _w_ ˜ _j_ ( _s_ ) [2] _−_ 2 

_j∈Ig_ _i∈Ig_


= Var( _zi_ ) �1 + 


- _w_ ˜ _i_ ( _s_ ) [2][�] (59)


_i∈Ig_


= Var( _zi_ ) �1 _−_ - _w_ ˜ _i_ ( _s_ ) [2][�] _._ (60)


_i∈Ig_


Note: In the third line, we used [�]


_i∈Ig_ _[w]_ [˜] _[i]_ [(] _[s]_ [) = 1][ and][ �]


_i∈Ig_ _[w]_ [˜] _[i]_ [(] _[s]_ [)][ �]


_j∈Ig_ _[w]_ [˜] _[j]_ [(] _[s]_ [)][2] [=][ �]


_j∈Ig_ _[w]_ [˜] _[j]_ [(] _[s]_ [)][2]


since the weights sum to 1. Thus

E� _σ_ �cond [2] [(] _[M]_ [)] �� _{_ **y** _i,M}_   - =   - _w_ ˜ _i_ ( _s_ )   - _βM_ _[⊤]_ **[y]** _[i,][M]_ _[−]_ _[y]_ [¯] _[M][,w]_ �2

_i∈Ig_


              - ��               weighted variance of measured part


+Var( _zi_ ) �1 _−_ - _w_ ˜ _i_ ( _s_ ) [2][�] _._


_i∈Ig_


**Asymptotics and conclusion.** By (A5) and boundedness of the weights (since [�]


**Asymptotics and conclusion.** By (A5) and boundedness of the weights (since [�] _i_ _[w]_ [˜] _[i]_ [(] _[s]_ [) = 1][ and]

0 _≤_ _w_ ˜ _i_ ( _s_ ) _≤_ 1), we have [�] _i∈Ig_ _[w]_ [˜] _[i]_ [(] _[s]_ [)][2] _[→]_ [0] [in] [probability] [when] _[|][I][g][|]_ _[→∞]_ [and] [the] [weights] [are]


0 _≤_ _w_ ˜ _i_ ( _s_ ) _≤_ 1), we have [�] _i∈Ig_ _[w]_ [˜] _[i]_ [(] _[s]_ [)][2] _[→]_ [0] [in] [probability] [when] _[|][I][g][|]_ _[→∞]_ [and] [the] [weights] [are]

not degenerate. [2] Also, by a (weighted) law of large numbers for triangular arrays with random but
_measured-part_ -measurable weights and finite second moments, the first term converges in probability to the _true_ conditional variance of the measured contribution _given the measured panel_ . However,
the exact VI-Theo conditional variance _does not depend_ on the measured panel (independence across
assays), hence

    -     - _[⊤]_ �2 _p_


- _w_ ˜ _i_ ( _s_ ) - _βM_ _[⊤]_ **[y]** _[i,][M]_ _[−]_ _[y]_ [¯] _[M][,w]_ �2 _−→p_ 0 _._

_i∈Ig_


Combining, we get

_σ_ �cond [2] [(] _[M]_ [)] _−→P_ Var( _zi_ ) (1 _−_ 0) = _βU⊤_ [Σ] _[U]_ _[β][U]_ [+] _[ σ]_ _ϵ_ [2] [=] _[σ]_ cond [2] [(] _[M]_ [)] _[.]_
Hence, under the synthetic linear/independent model, VI-Sim’s variance estimator is consistent for
the exact VI-Theo conditional variance.


**Implications.** Because _σ_ cond [2] [(] _[M]_ [)][ is constant in] **[ y]** _[M]_ [ under independence, any reasonable similarity]
weighting over measured assays yields the same limiting conditional variance. In more general
(correlated or non-linear) settings, the estimator targets Var( _g_ _|_ **y** _M_ = candidate) provided the
kernel and bandwidth obey standard nonparametric conditions; IBMDP’s stochastic ensembling
further mitigates finite-sample bias/variance.


D.7 EXPERIMENTAL PROTOCOL AND METRICS


**Overview.** We conduct a systematic comparison of the three planning methods across 100 independent trials, focusing on their alignment with the theoretically optimal policy.


For each of 100 independent trials:


(i) Generate a fresh synthetic dataset as in Section D.1.


(ii) Compute VI-Theo’s optimal first action at the initial state.


(iii) Compute VI-Sim’s recommended first action.


_i_ _[w]_ [˜] _[i]_ [(] _[s]_ [)][2] _[≤]_ [max] _[i]_ _[w]_ [˜] _[i]_ [(] _[s]_ [)][ �]


2Specifically, if max _i_ _w_ ˜ _i_ ( _s_ ) _−→P_ 0, then

[�]


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


(iv) Run IBMDP with an ensemble of MCTS-DPW planners; record (a) the _Top-1_ action (most frequent across the ensemble), and (b) the _Top-2_ action set (two most frequent).


We report three alignment metrics per trial:


  - **T1 Match:** indicator that IBMDP’s Top-1 equals VI-Theo’s action.


  - **T2 Match:** indicator that the VI-Theo action appears in IBMDP’s Top-2 set.


  - **Sim Match:** indicator that VI-Sim equals VI-Theo.


D.8 EXPERIMENTAL RESULTS AND ANALYSIS


**Summary.** Table 5 presents the main results. Over 100 trials, IBMDP’s Top-1 matches the VITheo optimum in 47 cases; IBMDP’s Top-2 covers the optimum in 66 cases; VI-Sim matches the
optimum in 36 cases. These results demonstrate that the stochastic, ensemble planner recovers a
larger fraction of near-equivalent high-value actions than a deterministic similarity planner, validating the value of IBMDP’s ensemble approach.


Table 5: Policy alignment with the theoretical baseline over 100 trials.

Method Matches Match Rate (%)


IBMDP Top 1 47 47.0
IBMDP Top 2 66 66.0
VI-Sim 36 36.0


**Statistical Consistency Across Independent Trials.** To validate the statistical reproducibility of
our method, we present the complete trial-by-trial alignment results below. This detailed analysis
demonstrates that IBMDP’s policy recommendations consistently align with the theoretical optimum across diverse problem instances, providing empirical evidence of the method’s robustness
and reliability (feature indices refer to assays _a ∈{_ 1 _, . . .,_ 6 _}_ ).


Table 6: Trial-wise comparison of VI-Theo vs. IBMDP and VI-Sim.


Iter VI-Theo IBMDP Top-1 Features T1 Match IBMDP Top-2 Features T2 Match VI-Sim Sim Match


1 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
2 5 _{_ 3, 4 _}_ 0 _{_ 3, 5, 6 _}_ 1 3 0
3 3 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 1 5 0
4 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
5 3 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 1 3 1
6 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 3 0
7 3 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 1 3 1
8 5 _{_ 3, 4 _}_ 0 _{_ 3, 5, 6 _}_ 1 3 0
9 3 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 1 3 1
10 4 _{_ 3, 4 _}_ 1 _{_ 3, 5, 6 _}_ 0 6 0
11 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 4 1
12 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 4 1
13 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 6 0
14 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 6 0
15 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
16 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
17 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 4 1
18 4 _{_ 3, 4 _}_ 1 _{_ 3, 5, 6 _}_ 0 3 0
19 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
20 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
21 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
22 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 4 1
23 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
24 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 6 0
25 5 _{_ 3, 4 _}_ 0 _{_ 3, 6 _}_ 0 5 1
26 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0


_Continued on next page_


29


**1566**

**1567**

**1568**

**1569**


**1570**

**1571**

**1572**

**1573**

**1574**

**1575**


**1576**

**1577**

**1578**

**1579**

**1580**


**1581**

**1582**

**1583**

**1584**

**1585**

**1586**


**1587**

**1588**

**1589**

**1590**

**1591**

**1592**


**1593**

**1594**

**1595**

**1596**

**1597**

**1598**


**1599**

**1600**

**1601**

**1602**

**1603**


**1604**

**1605**

**1606**

**1607**

**1608**

**1609**


**1610**

**1611**

**1612**

**1613**

**1614**

**1615**


**1616**

**1617**

**1618**

**1619**


Table 6: _(continued)_


Iter VI-Theo IBMDP Top-1 Features T1 Match IBMDP Top-2 Features T2 Match VI-Sim Sim Match


27 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
28 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 3 0
29 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 3 0
30 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
31 5 _{_ 3, 4 _}_ 0 _{_ 3, 5, 6 _}_ 1 3 0
32 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 6 0
33 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 5 1
34 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 4 1
35 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 4 1
36 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
37 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
38 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
39 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
40 3 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 1 3 1
41 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 4 1
42 4 _{_ 3, 4, 5 _}_ 1 _{_ 3, 4, 5 _}_ 1 6 0
43 3 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 1 3 1
44 5 _{_ 3, 4 _}_ 0 _{_ 3, 5, 6 _}_ 1 3 0
45 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
46 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
47 3 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 1 3 1
48 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 5 1
49 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 3 0
50 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
51 4 _{_ 3, 4 _}_ 1 _{_ 3, 5, 6 _}_ 0 4 1
52 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
53 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 5 1
54 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
55 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 3 0
56 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
57 4 _{_ 3, 4 _}_ 1 _{_ 3, 4, 5 _}_ 1 3 0
58 3 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 1 3 1
59 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 6 0
60 3 _{_ 3, 4 _}_ 1 _{_ 3, 4, 5 _}_ 1 3 1
61 3 _{_ 3, 4 _}_ 1 _{_ 3, 5, 6 _}_ 1 3 1
62 6 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 0 3 0
63 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 5 1
64 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 3 0
65 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
66 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
67 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
68 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 6 0
69 6 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 0 6 1
70 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 4 1
71 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
72 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
73 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 6 0
74 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
75 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 5 1
76 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 5 1
77 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
78 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 4 1
79 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
80 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 5 1
81 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
82 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
83 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 4 1
84 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
85 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
86 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
87 4 _{_ 3, 4 _}_ 1 _{_ 3, 5, 6 _}_ 0 3 0
88 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0


_Continued on next page_


30


**1620**

**1621**


**1622**

**1623**

**1624**

**1625**

**1626**

**1627**


**1628**

**1629**

**1630**

**1631**

**1632**

**1633**


**1634**

**1635**

**1636**

**1637**

**1638**


**1639**

**1640**

**1641**

**1642**

**1643**

**1644**


**1645**

**1646**

**1647**

**1648**

**1649**

**1650**


**1651**

**1652**

**1653**

**1654**

**1655**

**1656**


**1657**

**1658**

**1659**

**1660**

**1661**


**1662**

**1663**

**1664**

**1665**

**1666**

**1667**


**1668**

**1669**

**1670**

**1671**

**1672**

**1673**


Table 6: _(continued)_


Iter VI-Theo IBMDP Top-1 Features T1 Match IBMDP Top-2 Features T2 Match VI-Sim Sim Match


89 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
90 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
91 3 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 1 3 1
92 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 4 1
93 3 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 1 3 1
94 3 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 1 3 1
95 4 _{_ 3, 4 _}_ 1 _{_ 3, 5, 6 _}_ 0 3 0
96 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 5 1
97 5 _{_ 3, 4 _}_ 0 _{_ 3, 5 _}_ 1 3 0
98 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 4 1
99 3 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 1 3 1
100 4 _{_ 3, 4 _}_ 1 _{_ 3, 5 _}_ 0 6 0


**Interpretation.** VI-Theo and VI-Sim return a single deterministic action per state. IBMDP explores the posterior-predictive policy space via stochastic rollouts and, by ensembling, surfaces
_multiple_ near-equivalent high-value choices. The superior Top-2 coverage (66% vs. VI-Sim’s 36%
matching rate) reflects better policy-space exploration and robustness to finite-sample effects.


E BENCHMARK WITH PUBLIC DATASET


E.1 HIGH-COST DIFFERENTIAL CLEARANCE OPTIMIZATION


We reuse a publicly available pharmacokinetics dataset (rat, dog, human clearance plus QSAR predictors) to stress-test IBMDP under large assay cost differentials. The dataset is described in ( **?** ) and
is available to download. The planner may propose at most two assays per decision step, and the
expensive human clearance assay is treated just like the rat and dog assays (i.e., it can be scheduled
in any batch). The operational objective is to finish with human clearance exceeding 1.0 mL/min/kg
while spending as little as possible. Species-specific costs are listed in Table 7.


**Assay** **Cost ($)** **Relative Cost**
Rat clearance 400 1.0×
Dog clearance 800 2.0×
Human clearance 4,000 10.0×


Table 7: Assay cost structure for high-cost differential experiment

Unlike traditional gated progression (e.g., “rat before dog before human”), every episode starts with
the unmeasured state _s_ 0 = _{_ CL [pred] rat _[,]_ [ CL][pred] dog _[,]_ [ CL] human [pred] _[}]_ [ so the solver can pick any eligible batch.] [The]
IBMDP ensemble (30 runs, _c_ = 5 _._ 0, 5,000 iterations per run, _τ_ _∈{_ 0 _._ 6 _,_ 0 _._ 9 _}_ ) produces a MaximumLikelihood Action-Set Path (MLASP) by majority vote over recommended assay batches. The voting tally reveals three regimes: (i) high-uncertainty states favour rat/dog assays before committing
to human tests; (ii) low-uncertainty states jump directly to human clearance; and (iii) intermediate
states switch behaviour depending on the belief threshold _τ_ . Figure 5 visualizes the resulting Pareto
front and highlights how the MLASP navigates the trade-off between total spend and terminal uncertainty.


E.2 INTERPRETING THE PARETO FRONTIER


Figure 5 aggregates planning outcomes for tolerances _τ_ _∈{_ 0 _._ 0 _,_ 0 _._ 1 _, . . .,_ 1 _._ 0 _}_ under the two-assaysper-step constraint. For each tolerance we execute a 30-run ensemble and record the first assay batch
proposed by every run. Each marker therefore represents the rule ”if _H_ ( _sT_ ) _≤_ _τ_ is required, begin
with batch _A_ 0”; the horizontal axis reports the corresponding assay spend (rat + dog + human) and
the vertical axis equals the targeted uncertainty _τ_ .The blue locus links the Pareto-efficient points, exposing the spend-versus-uncertainty trade-off that emerges when _τ_ is tightened. The starred marker
denotes the Maximum-Likelihood Action-Set Path (MLASP)—the batch occurring most frequently
across ensemble members for the displayed tolerance. Progressing from higher to lower _τ_ shows
that lenient tolerances favour inexpensive rat/dog assays, whereas stringent requirements such as
_H_ ( _sT_ ) _≤_ 0 _._ 10 eventually demand the human clearance assay despite its 10 _×_ cost in Table 7. After


31


**1674**

**1675**


**1676**

**1677**

**1678**

**1679**

**1680**

**1681**


**1682**

**1683**

**1684**

**1685**

**1686**

**1687**


**1688**

**1689**

**1690**

**1691**

**1692**


**1693**

**1694**

**1695**

**1696**

**1697**

**1698**


**1699**

**1700**

**1701**

**1702**

**1703**

**1704**


**1705**

**1706**

**1707**

**1708**

**1709**

**1710**


**1711**

**1712**

**1713**

**1714**

**1715**


**1716**

**1717**

**1718**

**1719**

**1720**

**1721**


**1722**

**1723**

**1724**

**1725**

**1726**

**1727**


Figure 5: ADME clearance optimization results comparing IBMDP performance under two belief
thresholds ( _τ_ = 0 _._ 6 and _τ_ = 0 _._ 9) for a representative compound from the public CNS clearance
benchmark. The plot demonstrates the Pareto-optimal trade-offs between total assay spend (horizontal axis) and terminal state uncertainty _H_ ( _sT_ ) (vertical axis) achieved by the IBMDP ensemble
across 30 runs. The two distinct curves for _τ_ = 0 _._ 6 (more lenient) and _τ_ = 0 _._ 9 (more stringent)
illustrate how tighter belief thresholds drive higher assay expenditure to achieve lower uncertainty.
Notably, the two tau configurations exhibit strong alignment in their Pareto frontiers, confirming that
IBMDP produces consistent and robust planning strategies across different confidence requirements.
The Maximum-Likelihood Action-Set Paths (MLASPs) for each threshold are marked, showing how
the ensemble consensus adapts to balance the high cost of human clearance assays ($4,000) against
the need to reduce decision uncertainty below the specified threshold.


the first batch is executed the IBMDP policy updates the belief state and recomputes the next action,
so the figure captures the initial decision while the full policy remains adaptive.


F USE OF LLM


We used a large language model (LLM) solely as a general-purpose writing aid for light copyediting
and polishing. Specifically, the LLM was used to improve grammar, clarity, and flow of sentences
written by the authors, and to suggest minor phrasing alternatives. The LLM did not contribute
to research ideation, methodology, experimental design, data analysis, interpretation of results, or
substantive content generation. All technical claims, analyses, references, and conclusions were
conceived, written, and verified by the authors. The authors take full responsibility for all content in
this paper, including any text that was edited with the assistance of an LLM. No LLM is listed as an
author, and no text was accepted without author review and verification.


G GLOBAL NOTATION REFERENCE


This appendix provides a comprehensive reference for all mathematical notation used throughout
the manuscript. The table below organizes symbols by category for easy reference.


32


**1728**

**1729**


**1730**

**1731**

**1732**

**1733**

**1734**

**1735**


**1736**

**1737**

**1738**

**1739**

**1740**

**1741**


**1742**

**1743**

**1744**

**1745**

**1746**


**1747**

**1748**

**1749**

**1750**

**1751**

**1752**


**1753**

**1754**

**1755**

**1756**

**1757**

**1758**


**1759**

**1760**

**1761**

**1762**

**1763**

**1764**


**1765**

**1766**

**1767**

**1768**

**1769**


**1770**

**1771**

**1772**

**1773**

**1774**

**1775**


**1776**

**1777**

**1778**

**1779**

**1780**

**1781**


Table 8: Global Notation Reference summarizing the symbols used across the manuscript.
**Symbol** **Meaning**


**Sets & Indices**


_N, M_ Number of historical compounds and total available assays,
respectively.
_X_ = _{xi}_ _[N]_ _i_ =1 Set of _N_ historical compounds with fixed representations.
_A_ = _{a_ 1 _, . . ., aM_ _}_ Set of _M_ available assays.
_D_ = _{_ ( _xi,_ **y** _i_ ) _}_ _[N]_ _i_ =1 Historical dataset of compounds and their assay outcome vectors.
_i, j, k, t_ Indices for historical case, assay, feature, and decision step.


**Candidate Compound & State**


_x⋆_ The candidate compound for which a plan is being made.
_st_ = ( _x⋆, {y⋆,j}j∈Mt_ ) State at step _t_, comprising the candidate and all outcomes measured
so far.
_Mt_ _⊆A_ The set of assays that have been **m** easured for _x⋆_ up to step _t_ .
_Ut_ = _A \ Mt_ The set of **u** nmeasured assays for _x⋆_ at step _t_ .


**Actions, Costs & Policy**


_At_ = _P≤m_ ( _Ut_ ) _∪{_ eox _}_ Action set at _st_ : batches of up to _m_ unmeasured assays, plus the
stop action.
_m_ Maximum number of assays that can be run in parallel per step.
_At_ _∈At_ The action (a batch of assays) chosen at step _t_ .
_c_ ( _st, At_ ) _∈_ R _[q]_ _≥_ 0 Vector of _q_ resource costs for taking action _At_ .
_**ρ**_ _∈_ R _[q]_ _≥_ 0 User-defined weights for trading off different cost types.
_R_ ( _st, At_ ) Scalar step cost: _**ρ**_ [T] _c_ ( _st, At_ ). _R_ ( _st,_ eox) = 0.
_π, π_ _[⋆]_ A policy mapping states to actions, and the optimal policy.


**Similarity Model & Target Functionals**


_g_ The primary scalar target property of interest (e.g., an _in vivo_
endpoint).
_G_ = _{gi}, Ig_ Set of historical target values and the index set where they are
available.
_d_ ( _st, Di_ ) Variance-normalized distance between the current state and
historical case _i_ .
_wi_ ( _st_ ) Similarity weight of historical case _i_ given the current state _st_ .
_w_ ˜ _i_ ( _st_ ) Similarity weight _wi_ ( _st_ ) re-normalized over the set _Ig_ .
_H_ ( _st_ ) State uncertainty: the weighted variance of the target _g_ based on
_w_ ˜ _i_ ( _st_ ).
_L_ ( _st_ ) Goal likelihood: the weighted probability that _g_ is in a desirable
range.
**1** [ _·_ ] Indicator function: returns 1 when the condition inside the brackets
holds, and 0 otherwise.


**Hyperparameters & Constraints**


_λw, λk_ Hyperparameters: global similarity bandwidth and per-feature
weights.
_ϵ, τ_ Thresholds for the constrained objective: max terminal uncertainty
and min goal likelihood.
_γ, T_ Discount factor and maximum horizon for the MDP.
_Ne, n_ itr Planning parameters: ensemble size and MCTS iterations per run.


**Algorithm Components**


MCTS-DPW Monte Carlo Tree Search with Double Progressive Widening.
MLASP Maximum-Likelihood Action-Sets Path: final plan from ensemble
majority voting.
eox End of experiment action (stop action).


33