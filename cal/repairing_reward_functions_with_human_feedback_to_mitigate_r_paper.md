# REPAIRING REWARD FUNCTIONS WITH HUMAN FEED## BACK TO MITIGATE REWARD HACKING


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Human-designed reward functions for reinforcement learning (RL) agents are
frequently misaligned with the humans’ true, unobservable objectives, and thus
act only as proxies. Optimizing for a misspecified proxy reward function often
induces reward hacking, resulting in a policy misaligned with the human’s true
objectives. An alternative is to perform RL from human feedback, which involves
learning a reward function from scratch by collecting human preferences over pairs
of trajectories. However, building such datasets is costly. To address the limitations
of both approaches, we propose Preference-Based Reward Repair (PBRR): an
automated iterative framework that repairs a human-specified proxy reward function
by learning an additive, transition-dependent correction term from preferences. A
manually specified reward function can yield policies that are highly suboptimal
under the ground-truth objective, yet corrections on only a few transitions may
suffice to recover optimal performance. To identify and correct for those transitions,
PBRR uses a targeted exploration strategy and a new preference-learning objective.
We prove in tabular domains PBRR has a cumulative regret that matches, up to
constants, that of prior preference-based RL methods. In addition, on a suite of
reward-hacking benchmarks, PBRR consistently outperforms baselines that learn a
reward function from scratch from preferences or modify the proxy reward function
using other approaches, requiring substantially fewer preferences to learn high
performing policies.


1 INTRODUCTION


The reward hypothesis states that “all of what we mean by goals and purposes can be well thought of
as maximization of the expected value of the cumulative sum of reward” (Sutton & Barto, 2018). This
idea underpins much of reinforcement learning (RL): if we can specify the right reward function, then
optimizing for it should yield the desired behavior. However, manually designing a reward function
that fully captures a human designer’s true objectives is rarely possible (Amodei et al., 2016).


One approach is to instead rely on _proxy_ reward functions—simpler specifications that reflect the
intended but unobservable ground-truth objective. Unfortunately even well-considered proxies, when
optimized for by RL, often fail to produce policies that achieve the desired behavior; a failure mode
informally known as _reward hacking_ (Krakovna et al., 2020; Pan et al., 2022). For example, in an
autonomous driving task, maximizing mean velocity—a proxy for traffic flow—-could lead vehicles
to block highway on-ramps. The common recourse is an iterative, trial-and-error design process.
A designer specifies a proxy reward function, trains an agent, observes the resulting behavior, and
then manually edits the reward function to remove unwanted incentives (Booth et al., 2023; Knox
et al., 2023). We conjecture that while this process can eventually produce usable reward functions
and aligned policies, it is slow, ad hoc, and depends on RL expertise that many domain experts do
not have. Automating this process would make RL more practical, for example in domains such as
pandemic lockdown policy design (Kompella et al., 2020), autonomous driving (Wu et al., 2021;
Dosovitskiy et al., 2017), clinical decision making (Man et al., 2014; Petersen et al., 2019; Eastman
et al., 2021), energy management (Henry & Ernst, 2021; Orfanoudakis et al., 2024), or tax policy
optimization (Mi et al., 2023).


Another path to alignment is to remove the need for any explicit human reward design through
learning a reward function from human preferences over trajectories, i.e., reinforcement learning
from human feedback (RLHF). However, standard RLHF approaches typically require large datasets


1


of human preferences, which are often prohibitively costly to collect (Casper et al., 2023). Some
work (Novoseller et al., 2020; Pacchiano et al., 2023) has proposed methods for RLHF with cumulative
regret guarantees, using strategic exploration under uncertainty to require fewer preferences, but
rely on restrictive assumptions such as discrete state-action spaces and specific human preference
generation processes. Scaling up uncertainty-based approaches for RLHF to more complex domains
is non-trivial and empirical success has been mixed (Ji et al., 2024; Das et al., 2024; Dwaracherla
et al., 2024; Mehta et al., 2023).


Human specified reward functions
are often misaligned, requiring an informal and manual reward correction
process, while RLHF approaches are
potentially data intensive. To address these limitations, we introduce
**Preference-Based Reward Repair**
**(PBRR)**, an iterative framework for
efficiently and automatically repairing a human-specified proxy reward
function using preferences. For many
tasks, a human can readily specify a Figure 1: Illustration of Preference-Based Reward Repair (PBRR).
proxy reward function that reflects A human specifies a proxy reward function ˆ _r_, which is optimized for
the unobservable ground-truth objec- with reinforcement learning (RL) to produce a policy _πr_ _[∗]_ ˆ [. Preferences]
tive they have in mind, but lacks ro- between trajectories from _πr_ _[∗]_ ˆ [and] [a] [safe] [reference] [policy] _[π]_ [ref] [are]
bustness to the ways in which an RL elicited to identify instances of unaligned behavior. The preferences
agent might exploit it. However, a are used to update ˆ _r_ and the process repeats, iteratively aligning the

proxy reward function with the human’s unobservable ground-truth

limited number of targeted adjust
reward function _r_ .

ments to the proxy reward function
may be enough to restore near-optimal behavior. PBRR makes these adjustments as an automated
process, in contrast to the manual iterative reward correction process that humans often perform. To
automate reward function repair, PBRR leverages two core components: (i) a targeted exploration
strategy, which elicits preferences between trajectories generated by the policy trained with the
proxy reward function and those from a supplied reference policy, and (ii) a new preference-learning
objective to update the proxy reward function only over transitions incorrectly assigned high reward.
Figure 1 provides an overview.


On sequential decision process benchmark environments that highlight the challenges of reward
hacking (Pan et al., 2022), PBRR significantly outperforms approaches that learn a reward function
from scratch from preferences–i.e., standard RLHF—or attempt to repair the proxy reward function
using alternative strategies. We also prove that when operating in the tabular settings of past
theoretical work, a variant of PBRR matches the cumulative regret bounds of a prior strategic RLHF
method (Pacchiano et al., 2023) up to constant terms. Our contributions are three-fold:


- We introduce Preference-Based Reward Repair (PBRR), a method for efficiently repairing a
human-specified proxy reward function using a new exploration method and learning objective.

- We prove a variant of PBRR matches, up to constants, the sublinear cumulative regret bounds of
Pacchiano et al. (2023) in the same regime.

- We show that PBRR effectively repairs a proxy reward function even when that initial proxy reward
induces a substantially suboptimal policy, and consistently outperforms all baselines on a suite of
reward-hacking benchmarks.


2 BACKGROUND AND SETTING


Consider an MDP _M_ ≜ ( _S, A,_ Ω _, γ, p_ 0 _, r_ ) with state space _S_, action space _A_, transition dynamics
Ω: _S × A →_ ∆( _S_ ), discount factor _γ_ _∈_ [0 _,_ 1], and initial state distribution _p_ 0. The horizon is _H_ and
the ground-truth reward function _r_ : _S × A × S_ _→_ R. _M \ r_ ≜ ( _S, A,_ Ω _, γ, p_ 0 _,_ _) is an environment
without a specified reward function. Let _r_ be the (unobservable) ground-truth reward function, ˆ _r_ a
candidate approximation (e.g., learned or specified proxy), and ˜ _r_ an arbitrary reward function.


A policy _π_ : _S ×_ _A →_ [0 _,_ 1] maps states to action distributions. Its expected discounted return under ˜ _r_
from start distribution _p_ 0 is _Jr_ ˜( _π_ ). An _optimal policy_ for ˜ _r_ is any _πr_ _[∗]_ ˜ _[∈]_ [arg max] _[π][ J][r]_ [˜][(] _[π]_ [)][.] [We say][ ˆ] _[r]_ [ is]
misspecified in environment _M_ with ground-truth reward _r_ if _Jr_ ( _πr_ _[∗]_ ˆ [)] _[ < J][r]_ [(] _[π]_ _r_ _[∗]_ [)][.] [Unless noted, policy]


2


performance refers to expected discounted return under _r_ . Let _τ_ denote a trajectory starting at state
_s_ _[τ]_ 0 _[∼]_ _[p]_ [0][:] _[τ]_ [=] [(] _[s][τ]_ 0 _[, a][τ]_ 0 _[, s][τ]_ 1 _[, a][τ]_ 1 _[, ..., s][τ]_ _H_ [)][.] [Let a trajectory’s return be] _[r]_ [˜][(] _[τ]_ [)] [=] [�] _t_ _[H]_ =0 _[γ][t][r]_ [˜][(] _[s]_ _t_ _[τ]_ _[, a]_ _t_ _[τ]_ _[, s]_ _t_ _[τ]_ +1 [)]
and _Tπ_ denote the support of the trajectory distribution induced by _π_ from _p_ 0.


We learn a reward ˆ _r_ from trajectory pair preferences. Let
_D_ = _{_ ( _τ_ 1 _, τ_ 2 _, µ_ ) _}_ _[N]_ _k_ =1 _[,]_ _µ ∈{_ 0 _,_ 1 _,_ [1] 2 _[}]_ [ with][ 0 :] _[ τ]_ [1] _[≻]_ _[τ]_ [2] _[,]_ [1 :] _[ τ]_ [2] _[≻]_ _[τ]_ [1] _[,]_ 12 [:] _[ τ]_ [1] _[∼]_ _[τ]_ [2] _[.]_


As is standard in RLHF (Christiano et al., 2017), we assume a Bradley–Terry preference model:

_P_ ( _τ_ 1 _≻_ _τ_ 2 _|_ _r_ ˜) = _σ_       - _r_ ˜( _τ_ 1) _−_ _r_ ˜( _τ_ 2)� _,_ _σ_ ( _x_ ) = 1 _/_ (1 + _e_ _[−][x]_ )


Although ubiquitous, this model of noisy rationality may not account for all the ways in which
humans fail to act optimally; see Zhi-Xuan et al. (2025) for further discussion.


Unless otherwise stated, we fit ˆ _r_ by minimizing the cross-entropy loss:

_L_ pref(ˆ _r_ ; _Dt_ ) = _−_ �(1 _−_ _µ_ ) log _P_ ( _τ_ 1 _≻_ _τ_ 2 _|r_ ˆ) + _µ_ log _P_ ( _τ_ 1 _≺_ _τ_ 2 _|r_ ˆ) _._ (1)

( _τ_ 1 _,τ_ 2 _,µ_ ) _∈D_

We assume a preference _µ_ is elicited over a pair of trajectories, rather than shorter trajectory segments,
to mitigate issues relating to misspecified human preference models (see Appendix. A).


3 RELATED WORK


Alignment has received extensive attention, particularly in the context of large language models. Here
we focus instead on sequential decision processes.


Prior work has explored how to align agent behavior in MDPs despite a human’s misspecified reward
function under two broad classes of restrictive assumptions. First, some methods assume particular
structural properties of the underlying MDP, such as complete knowledge of the human’s MDP
(Mechergui & Sreedharan, 2024) or that the provided reward function already induces near-optimal
behavior (Hadfield-Menell et al., 2017), requiring only additional calibration (Fu et al., 2025). These
assumptions do not hold in the environments we study. Second, other methods assume access
more demanding human feedback, such as corrective actions (Jiang et al., 2024; Peng et al., 2023),
continuous-valued human ratings (Zhang et al., 2024), or feature-attribution–based explanations
(Mahmud et al., 2023). Relying on corrective actions would require the human reward designer to
provide demonstrations—e.g., controlling a fleet of autonomous vehicles on a highway or determining
appropriate pandemic lockdown policies—which demands substantial expertise. Continuous-valued
feedback and explanation-based supervision similarly impose a high cognitive burden, limiting who
can design aligned reward functions. Our approach, by contrast, requires the human to provide
comparative judgments.


Other work infers a posterior over plausible reward functions from human data to mitigate errors in
the reward functions (Eisenstein et al., 2023; Mahmud et al., 2023; Coste et al., 2023), and disjoint
work shows how such a prior (either learned or provided directly by a stakeholder) can be leveraged
for efficient exploration (Novoseller et al., 2020). However it can be challenging for stakeholders to
provide Bayesian priors, and learning them may be brittle. We instead only require a stakeholder to
provide a single proxy reward function.


In RLHF for large language models, given access to a sufficiently performant reference policy and an
estimated but flawed reward function, penalizing KL-divergence between action distributions during
training can induce a high-performing policy (see, e.g., Ziegler et al. (2019); Ouyang et al. (2022);
Bai et al. (2022); Glaese et al. (2022); OpenAI (2022); Touvron et al. (2023)). In MDPs, Laidlaw
et al. (2025) highlight that using different divergences measures can improve policy performance. In
contrast, our work focuses on settings where no such high-performing reference policy is available.


Another line of work focuses on efficiently learning a reward function from preferences. Theoretical
results for RL in discrete state and action spaces are promising (Novoseller et al., 2020; Pacchiano
et al., 2023) but rely on quantifying priors or precise measures of uncertainty over the reward
function, which is unclear how to replicate in more complex settings. In large language model
settings, preference-based algorithms leveraging coarse approximations of uncertainty or optimism
have yielded modest empirical benefit (Mehta et al., 2023; Xie et al., 2024; Das et al., 2024).


Concurrent to our work, Cao et al. (2025) also assume access to a proxy reward function and learn an
additive correction term from preferences, demonstrating benefits on robotic manipulation tasks. Our


3


work differs in several important ways. We study settings where a misspecified proxy reward function
induces highly suboptimal behavior. We then propose an exploration strategy that effectively corrects
the proxy reward function by leveraging a suboptimal reference policy, as in other RLHF methods.
Further, we introduce a new learning objective for repairing a proxy reward function. Together, these
components substantially improve performance compared to applying the standard RLHF procedure
to update an inputted proxy reward function, which corresponds directly to the baseline of Cao
et al. (2025), in the reward-hacking benchmark introduced by Pan et al. (2022). We also provide a
theoretical analysis of our approach, whereas Cao et al. (2025) focus solely on empirical results.


4 METHODOLOGY


We now present our Preference-Based Reward Repair (PBRR) algorithm. We assume a human
stakeholder initially provides a reward function _r_ ˆproxy( _s, a, s_ _[′]_ ). PBRR then iteratively aligns this
human-specified proxy reward function to their ground-truth objective by eliciting preferences.


Without loss of generality, the ground-truth reward function _r_ ( _s, a, s_ _[′]_ ) can be written as the proxy
reward function ˆ _r_ proxy( _s, a, s_ _[′]_ ) plus a correction _g_ ( _s, a, s_ _[′]_ ). Thus, repairing ˆ _r_ proxy amounts to learning
a transition-dependent correction _g_ . At iteration _t_, we elicit a preference batch _Dt_ and update _gt_ +1,
yielding the modified proxy reward function:


_r_ ˆ _t_ +1( _s, a, s_ _[′]_ ) ≜ _r_ ˆproxy( _s, a, s_ _[′]_ ) + _gt_ ( _s, a, s_ _[′]_ ) (2)


_r_ ˆ _t_ always denotes a modified proxy reward function, while ˆ _r_ proxy denotes the original proxy reward
function. _gt_ ( _s, a, s_ _[′]_ ) is parameterized as a neural network.


This specification offers three benefits. First, the stakeholder need only provide a point-estimate
reward function—which will then be corrected—-rather than a full Bayesian prior over reward
functions and uncertainties for Bayesian methods. Second, data-efficiency may increase when the
complexity of the additive correction term lies in a lower dimensional space than the full reward
function. Third, as we will now show, it enables the design of a loss function that explicitly leverages
expected properties of the proxy reward function.


In particular, we expect that humans typically provide reward functions that are aligned or overly
optimistic. [1] Many cases of reward hacking arise because humans misestimate the cumulative effect of
small or multi-objective rewards which can dominate long-term outcomes (e.g., an agent in a racing
task learns to loop endlessly through checkpoints to accumulate points, not to finish the race), or
because humans mispredict which actions will maximize expected return (e.g., an RL agent exploits
a bug in its environment). These types of over-optimistic reward functions cover most examples
of reward-hacking from Krakovna et al. (2020). A standard way to learn _gt_ ( _s, a, s_ _[′]_ ) would be to
minimize cross-entropy (Eq. 1), but this ignores the assumed optimism of the input proxy reward
function, potentially increasing corrections for already aligned or optimistic transition rewards and
leading to inaccuracies or instability.


**Repairing a proxy reward function** Based on this intuition, we design a loss function for learning
the correction term _g_ that regularizes towards only correcting transitions that are incorrectly assigned
high reward, conflicting with observed preferences. We first partition the preference dataset into
_Dt_ [+][, the set that contains all samples where the proxy reward function’s induced ranking matches the]
elicited preference _µ_, and _Dt_ _[−]_ [, the set that contains all other samples][ (] _[τ]_ [1] _[, τ]_ [2] _[, µ]_ [)][:]

_Dt_ [+] [=] _[ {]_ [(] _[τ]_ [1] _[, τ]_ [2] _[, µ]_ [)] _[ |]_ [ sign (ˆ] _[r]_ [proxy][(] _[τ]_ [2][)] _[ −]_ _[r]_ [ˆ][proxy][(] _[τ]_ [1][)) = sign(] _[µ][ −]_ [0] _[.]_ [5)] _[}]_ [ and] _[ D]_ _t_ _[−]_ [=] _[ D][t]_ _[\ D]_ _t_ [+]

We then learn the corrective term _g_ by minimizing the following three-term loss:


              - ��              _L_ _[−]_


1 _r_ ˆ is overly optimistic if, _∀_ ( _s, a, s′_ ), ˆ _r_ ( _s, a, s′_ ) _≥_ _r_ ( _s, a, s′_ )


4


_L_ ( _g_ ; ˆ _r_ proxy _, Dt_ ) ≜ _L_ pref(ˆ _r_ proxy + _g_ ; _Dt_ ) + _λ_ 1 _|D_ 1 _t_ [+] _[|]_ 
( _τ_ 1 _,τ_ 2) _∈Dt_ [+]


- _g_ ( _τ_ 1) [2] + _g_ ( _τ_ 2) [2][�]


+ _λ_ 2 _|D_ 1 _t_ _[−][|]_ 
( _τ_ 1 _,τ_ 2) _∈Dt_ _[−]_


  - ��   _L_ [+]

- 1 _{τ_ 1 _≻_ _τ_ 2 _}g_ ( _τ_ 1) [2] + 1 _{τ_ 2 _≻_ _τ_ 1 _}g_ ( _τ_ 2) [2][�]


(3)


The first term, _L_ pref, is a standard preference loss that encourages the modified reward function
_r_ ˆproxy + _g_ to satisfy the preferences in _Dt_ . The second term _L_ [+] regularizes the correction term _g_
towards zero on trajectory pairs where the proxy reward function agrees with the human preference.
This assumes that when the proxy reward function correctly ranks a pair of trajectories, its assigned
reward for the transitions in those trajectories is consistent with the ground-truth reward function; _L_ [+]
prevents unnecessary adjustments that could degrade an otherwise correct reward signal. Finally, the
third term _L_ _[−]_ focuses on trajectory pairs that were misclassified by the modified reward function.
Adjusting the correction term _g_ to correctly classify such trajectories is generally underspecified–
one could (a) increase the reward for the preferred trajectory, or (b) decrease the reward for the not
preferred trajectory. Consistent with our assumptions on the proxy reward function, _L_ _[−]_ prioritizes
option (b), regularizing the correction term _g_ to zero on transitions in the preferred trajectories, which
will prioritize a negative correction for undesirable behaviors.


To ensure our approach still learns the ground-truth reward function when our assumptions about the
specified proxy reward function fail to hold, we decay _λ_ 1 and _λ_ 2 over iterations (see Appendix E.6).
Although Eq. 3 leverages the assumption that the proxy reward function is optimistic, our algorithm
does not require this assumption, nor does our theoretical analysis in Section 5.


**Algorithm 1** Preference-Based Reward Repair (PBRR)


1: **Input:** Initial proxy reward function ˆ _r_, reference policy _π_ ref, number of iterations _N_
2: Initialize _gt_ ( _s, a, s_ _[′]_ ) _←_ 0 for all ( _s, a, s_ _[′]_ )
3: **for** _t_ = 1 to _T_ **do**
4: Compute _πr_ _[∗]_ ˆ _t_ [given the proxy reward function][ ˆ] _[r][t]_ [=] _[r]_ [ˆ][proxy][ +] _[ g][t]_ [2]

5: _π_ 1 = _πr_ _[∗]_ ˆ _t_ [,] _[ π]_ [2] [=] _[ π]_ [ref]
6: **if** _C_ 1 _>_ 0 **then**
7: Compute Π _t_, non-dominated policy set
8: **if** _πr_ _[∗]_ ˆ _t_ _[∈][/]_ [Π] _[t]_ [ or] _[ π]_ [ref] _[∈][/]_ [Π] _[t]_ [ or] _[ C]_ [1] _[f]_ [(] _[π]_ _r_ _[∗]_ ˆ _t_ _[, π]_ [ref] [)] _[ ≤]_ [max] _[π]_ 1 _[,π]_ 2 _[∈]_ [Π] _t_ _[f]_ [(] _[π]_ [1] _[, π]_ [2][)] **[ then]**
9: _π_ 1 _, π_ 2 = arg max _π_ 1 _,π_ 2 _∈_ Π _t f_ ( _π_ 1 _, π_ 2)
10: **end if**
11: **end if**
12: Collect trajectories _Tπ_ 1 and _Tπ_ 2 and sample trajectory pairs ( _τ_ 1 _, τ_ 2) with _τ_ 1 _∈Tπ_ 1 _, τ_ 2 _∈Tπ_ 2
13: Elicit preferences _µ_ over each pair ( _τ_ 1 _, τ_ 2) and add labeled pairs ( _τ_ 1 _, τ_ 2 _, µ_ ) to _Dt_
14: Update ˆ _ri_ +1 by learning additive correction _gi_ +1 using Equation 3 with _Dt_
15: **end for**
16: **Output:** Final modified reward function ˆ _rT_


**Constructing a preference dataset** We next describe how data are gathered to repair the proxy
reward function. We assume access to a reference policy, e.g., constructed from heuristics or a
previously used policy. Our hypothesis is that such a policy could provide a valuable contrast to
when the proxy reward function’s induced policy still leads to behavior considered undesirable by the
stakeholder.


Accordingly, PBRR prioritizes eliciting preferences between trajectories sampled from the policy that
optimizes for the corrected proxy reward function and the reference policy, matching the exploration
strategy of Xie et al. (2024). However, in some settings, this exploration strategy may be insufficient
to correct the proxy reward function to induce an optimal policy. Therefore, we follow Pacchiano
et al. (2023) and define an undominated policy set: the set of policies that remain potentially optimal
given the observed data. From this set, we identify the pair of policies with the largest divergence
in expected feature values under a weighted covariance norm. If the reference policy and policy
optimizing for the corrected proxy reward function have a divergence within a constant of this
maximal value, we use the reference policy and corrected proxy reward function’s induced policy
for exploration. If not, the algorithm can use the maximum divergence policy pair. This strategy
enables a principled fallback for additional optimistic exploration when needed, and is analogous to
prior work in contextual bandits that defaults to explicit optimistic exploration only when necessary
(Bastani et al., 2021). Our PBRR algorithm is detailed in Algorithm 1. The method for constructing
the preference dataset, i.e., the exploration strategy, is specified in Lines 5-12.


2In practice, we train a policy using ˆ _rt_ but are not guaranteed to find an optimal policy _πr∗_ ˆ _t_ [.]


5


5 REGRET ANALYSIS


We now show that if the ground-truth return for a trajectory can be expressed as a linear function
_√_
of the trajectory embedding, _r_ ( _τ_ ) = _⟨ϕ_ ( _τ_ ) _, w_ _[∗]_ _⟩_, then PBRR achieves _T_ cumulative regret. We

draw upon recent theoretical results from (Pacchiano et al., 2023). Our key observation is that if the
reference policy _π_ ref and optimal policy for the repaired reward function _πr_ ˆ _t_ lie in the set of possibly
optimal policies under the ground-truth reward function given the observed data, and the features
induced by their policies maximize the uncertainty with respect to paired feature covariance matrix
up to a constant of the maximizing policy pair, then sampling trajectories from the support of _π_ ref
and _πr_ ˆ _t_ will yield bounded cumulative regret within a constant factor of selecting the maximizing
uncertainty pair. If these conditions do not hold, our algorithm will instead reduce [3] to selecting the
maximizing uncertainty pair. Proofs and additional details are provided in Appendix J.
**Assumption 5.1.** _The trajectory return is linear in a feature trajectory embedding, r_ ( _τ_ ) = _⟨ϕ_ ( _τ_ ) _, w⟩._

**Assumption 5.2.** _Preferences over trajectories are sampled from the Bradley-Terry preference model_
_defined over the difference in return between trajectories._
**Assumption 5.3.** _We assume that ∥_ **w** _[∗]_ _∥≤_ _W_ _for some known W_ _>_ 0 _._
**Assumption 5.4.** _For all trajectories τ_ _we assume that ∥ϕ_ ( _τ_ ) _∥≤_ _B for some known B_ _>_ 0 _._


In Theorem 5.1 we show that when the dynamics model is known, PBRR inherits the same cumulative
regret bounds as Pacchiano et al. (2023) up to constants. We provide full details in Appendix J.


We first define the undominated set of policies as the set of policies that remain potentially optimal
under the current uncertainty in the linear reward model parameters:

  -   Π _t_ ≜ _πi|_ ( _ϕ_ ( _πi_ ) _−_ _ϕ_ ( _π_ )) _[T]_ _wt_ + _γt_ ( _δ_ ) _||ϕ_ ( _π_ _[i]_ ) _−_ _ϕ_ ( _π_ ) _||Vt−_ 1 _≥_ 0 _∀_ _π_

We also define the regularized feature covariance matrix with respect to the difference in trajectory feature values: _Vt_ ≜ [�] _[t]_ _l_ =1 _[−]_ [1][(] _[ϕ]_ [(] _[τ]_ [ 1] _l_ [)] _[ −]_ _[ϕ]_ [(] _[τ]_ [ 2] _l_ [))(] _[ϕ]_ [(] _[τ]_ [ 1] _l_ [)] _[ −]_ _[ϕ]_ [(] _[τ]_ [ 2] _l_ [))] _[T]_ [+] _[ κλI][d]_ [.] [Next, we consider the difference]
in expected feature embeddings of two policies under the known dynamics model, measured in the inverse covariance norm: _fmk_ ( _π_ 1 _, π_ 2) = _||ϕ_ ( _π_ 1) _−_ _ϕ_ ( _π_ 2) _||Vt−_ 1 . And finally we define a measure of the

non-linearity of the sigmoid function over the parameters space: _κ_ ≜ sup **x** _∈BB_ ( _d_ ) _,_ **w** _∈BS_ ( _d_ ) _σ_ _[′]_ ( **w** 1 _[⊤]_ **x** )
where _σ_ _[′]_ denotes the derivative and _BB_ ( _d_ ) defines the l2-norm ball of radius _B_ in dimension _d_ . We
then have:
**Theorem 5.1.** _Let δ_ _≤_ [1] _e_ _[and][ λ]_ _[≥]_ _[B/κ][.]_ _[Then under Assumptions 5.1,5.3, 5.2, and 5.4, and that]_

_the dynamics model is known, for f_ = _fmk_ _and_ Π _t_ = Π _t,mk, with probability at least_ 1 _−_ _δ, the_
_expected regret of Algorithm 1 is bounded by_


_for all T_ _simultaneously with probability at least_ 1 _−_ 15 _δ, where_ _O_ [˜] _hides logarithmic factors in δ, |S|_
_and |A|._


3It is possible to define a variant of our algorithm to handle the various subcases of if _π_ ref is not in the
non-dominated policy class Π _t_, if _πr_ ˆ _t_ is not in the non-dominated policy class Π _t_, or to consider if either
either _πr_ _[∗]_ ˆ _t_ [or] _[ π]_ [ref] [can be used as one of the uncertainty-maximizing policy pair; these cases do not impact our]
theoretical results and so for simplicity we keep the algorithm as is.


6


     - _√_
_Regrett_ _≤_ _O_ [˜] _C_ 1( _κ_


_√_
_λW_ +


_√_
_d_ + _BW_


_√_
_d_ )


 _Td_ (4)


_where_ _O_ [˜] _hides logarithmic factors in T,_ [1]


[1] [1]

_κ_ _[,]_ _λ_


[1] [1]

_δ_ _[, B,]_ _κ_


[1] [1]

_λ_ _[,]_ _d_


_d_ _[.]_


We explicitly retain the _C_ 1 factor to show that our regret guarantees are slightly looser than the results
of Pacchiano et al. (2023), owing to our alternative trajectory selection strategy.


In Theorem 5.2 we show that when the dynamics model is unknown, PBRR also inherits the same
cumulative regret bounds as Pacchiano et al. (2023) up to constants. We defer further details to
Appendix J, which redefines the undominated policy set to account for uncertainty in the unkown
dynamics model P [ˆ] _t_, and defines _fu_ to compute the expected trajectory feature difference using P [ˆ] _t_ .
**Theorem 5.2.** _Under Assumptions 5.1,5.3,5.2, and 5.4, for f_ = _fu_ _and_ Π _t_ = Π _t,u, the regret of_
_Algorithm 1 is bounded by_


   -    - _√_
_RT_ _≤_ _O_ [˜] _C_ 1 _κd_


_T_ + _H_ [3] _[/]_ [2][�]


_|S||A|dTH_ + _H|S|_ ~~�~~ _|A|dTH_ �� _,_ (5)


Theorems 5.1 and 5.2 suggest that it may often be possible to select the reference policy and policy
that optimizes for the current proxy reward function while matching—up to constants—the regret
bounds of prior work.


6 EXPERIMENTS


We now empirically evaluate PBRR. Our settings largely involve high-dimensional state spaces
where the ground-truth reward function is not linear. [4] Defining undominated policy sets for complex,
non-linear reward functions learned from preferences is intractable without further restrictions on
the policy and reward class, as is finding the best uncertainty-maximizing policy pairs. Therefore, in
our empirical results we set _C_ 1 = 0, which implies, from Line 6 of Algorithm 1, that for exploration
PBRR always uses the reference policy and the policy that optimizes for the corrected proxy reward
function. We find that repairing a proxy reward function with PBRR and _C_ 1 = 0 induces substantially
better performance with fewer preferences than either learning a reward function from scratch via
RLHF or repairing a proxy reward function using alternative methods.


6.1 ENVIRONMENTS


We evaluate PBRR across the reward hacking benchmark environments used in Pan et al. (2022), summarized in Table 1. More details are in Appendix B. These environments include high-dimensional
continuous state and action spaces. For comparison, the state spaces for the Glucose Monitoring,
Traffic Control, and Pandemic Mitigation environments are larger than that of the popular Meta-World
robotics environments Yu et al. (2020). The action space for Traffic-Control is also larger. All preference labels are sampled from the Boltzmann distribution under the environment’s ground-truth reward
function over full trajectories. For all environments, the proxy reward function induces substantially
sub-optimal performance under the ground-truth reward function.


**Name** **Objective** **State** **Action** **H** **Ref.** **policy** _r_ ˆ **proxy & Summary of** **Simulator**
_πr_ _[∗]_ ˆ **proxy**


Table 1: Reward-hacking environments, used by Pan et al. (2022). H = horizon. BC = behavior cloning. Cont. =
Continuous. Disc. = Discrete.


6.2 BASELINES


We compare PBRR against state-of-the-art and natural ablation baselines. More details are in
Appendix D. Appendix G.3 presents additional experiments where we update the reference policy
across iterations, when relevant.


- **State-Constrained-PPO** A natural baseline is to optimize for the proxy reward function using
the reference policy as a constraint in PPO, with no additional data collection. Laidlaw et al.
(2024) showed that using a state-based divergence measure often yields better performance, so we
optimize over the set of best-performing divergences amongst their proposals.


4At least, not in a feature space that is likely to be known to the decision maker in advance.


7


**Pandemic**
**Mitigation**
(Kompella
et al., 2020)


**Glucose**
**Monitoring**
(Man et al.,
2014; Fox
et al., 2020)


**Traffic**
**Control**
(Wu et al.,
2021)


**AI Safety**
**Gridworld**
(Leike et al.,
2017)


Design COVID-19
pandemic lockdown
regulations to
balance economic
and health outcomes


Administer insulin
to patient with Type
II diabetes to
prioritize patient
health outcomes


Control autonomous
vehicle (AV) fleet on
highway to
maximize traffic
flow


Cont., Disc. 192 BC on real-world
312-dim _{−_ 1 _,_ 0 _,_ 1 _}_ mitigation
strategies from
Kompella et al.
(2020)


Cont., 1-d Cont. 5760 BC on _few_ expert
96-dim in [0 _,_ 1] demos (Willis,
1999)


Modified SEIR
simulator
(Kompella et al.,
2020)


FDA-approved
simulator (Man
et al., 2014; Fox
et al., 2020)


FLOW highway
simulator (Wu
et al., 2021)


Toy environment
from Leike et al.
(2017)


Cont., 10-d
50-dim Cont. in

[0 _,_ 1]


300 BC on _few_ human
driving demos
(Treiber et al.,
2000)


Omits political cost of
lockdowns _⇒_ policy
keeps high lockdown
restrictions even at low
infection rates


Prioritizes reducing
financial cost of
treatment _⇒_ policy
sacrifices patient health
for low costs


Maximizes mean
velocity _⇒_ AVs block
on-ramp so other
vehicles can’t merge
onto highway


Tomatoes look watered
when they are not if the
agent visits the sprinkler
state _⇒_ policy never
waters all tomatoes


Water all tomatoes Disc., Disc. 100 Policy trained to
in a grid-world 36-dim, _{_ 0 _,_ 1 _,_ 2 _,_ 3 _}_ only water a
subset of tomatoes


Figure 2: Mean return under the ground-truth reward function achieved by PBRR compared to the baselines
from Section 6.2, averaged over 3 random seeds. The mean return for each seed is scaled so that, after scaling,
the reference policy has a mean return of 0 and _r_ -PPO has a mean return of 1. Values are clipped to [ _−_ 1 _,_ 1],
with values below _−_ 1 indicating very poor performance. Shaded regions indicate the standard error. Unscaled,
unclipped returns and further plotting details are provided in Appendix G.7 and Appendix G.1.★ and ★ mark
PBRR’s performance after two proxy reward function updates and the final update respectively.


- **Online-State-Constraint** This method also uses the best divergence-regularized objectives of
Laidlaw et al. (2024), but now the reward function is learned from scratch. Preferences are elicited
between trajectories sampled from the learned policy and the reference policy.

- **Online-RLHF** (Christiano et al. (2017)) Their method learns a reward model from scratch and
maintains a reward ensemble. Trajectory pairs with the highest predictive uncertainty are selected
for preference elicitation. To construct a stronger baseline, the reference policy is used for initial
exploration data rather than the initial exploration method of follow-up work (Lee et al., 2021a).

- **Residual Reward Modeling (RRM)** (Cao et al. (2025)) This method learns the correction term
_g_ for Eq. 2 using the standard cross-entropy loss. It elicits preferences over trajectory pairs with
the highest predictive uncertainty sampled from the policy optimized for the current proxy reward
function. This baseline is a modification of Online-RLHF that leverages the proxy reward function.

- **RRM** **+** **State-Constraint** Here we adapt the Online-State-Constraint baseline to learn the
correction term _g_ for Eq. 2 to repair a proxy reward function.

- _r_ **-PPO** : PPO using the ground truth reward model. Serves as an oracle upper bound.


6.3 MAIN RESULTS: REPAIRING _r_ ˆPROXY WITH PREFERENCES


Results are shown in Figure 2, with additional experiments in Appendix G. Using PBRR to repair
the proxy reward function with preferences is more data efficient than learning a reward function _ab_
_initio_ without the human-specified prior, and alternative approaches for repairing the proxy reward
function. Moreover, PBRR outperforms State-Constrained-PPO, indicating that comparisons to the
reference policy enable efficient learning of _g_, even when the reference policy itself is not performant
enough to successfully employ any divergence-based method we consider.


★ **Better** **Jump** **Start** **Performance** After the first two updates of the proxy reward function,
PBRR attains significantly higher performance than all baselines in all environments except for
Traffic Control, where PBRR matches the performance of Online-RLHF.


★ **Strong** **Final** **Performance** Within the preference budgets considered, PBRR matches or
outperforms all baselines in every environment. The performance of RRM indicates that the proxy


8


reward function alone does not provide enough exploration guidance to learn a correction term even
after eliciting a large dataset of preferences; see Appendix H.2 for a qualitative analysis. PBRR
overcomes this limitation by leveraging the reference policy to direct exploration.


**Stability** PBRR is substantially more stable in the AI safety gridworld, Traffic, and Pandemic
environments. While other methods can eventually match its performance with enough preferences,
their instability makes them impractical without ground-truth evaluation. Unlike PBRR, they show
oscillatory or degrading performance as more preferences are collected, since small changes in the
reward function can cause large changes in policy performance. Appendix H.1 analyzes this effect,
explaining why Online-RLHF is unstable in the AI Safety Gridworld.


**Outperforming the reference policy** _π_ **ref** Our theoretical analysis of PBRR with _C_ 1 = 0 (Appendix
K) only guarantees that it asymptotically performs no worse than the reference policy. See Appendix
I.1 for illustrative examples. Nevertheless, PBRR empirically induced policies that significantly
outperform the reference policy in all environments. The reference policy used by PBRR need not be
performant; it must only provide a useful comparison to the proxy reward function’s induced policy.
In Appendix G.8, we empirically show that a randomly initialized reference policy suffices, with
PBRR continuing to match or outperform all baselines when using a randomly initialized reference
policy.


**Optimism Assumption** PBRR leverages the assumption that the proxy reward function is optimistic.
However, in the Glucose Monitoring environment, this assumption does not hold; policies that
maximize patient health outcomes, i.e., are optimal under the ground-truth reward function, are
penalized under the proxy reward function due to their financial cost. Nonetheless, PBRR still
outperforms all baselines. Appendix K.7 reports similar findings in the AI Safety Gridworld when
repairing a proxy reward function that is not optimistic.


6.4 ABLATION STUDY: PREFERENCE LEARNING OBJECTIVE VS. EXPLORATION STRATEGY


We also sought to isolate the contributions of our preference-learning objective (Eq. 3) and exploration
strategy (Section 4). Figure 3 shows that repairing the proxy reward function with PBRR but using
the standard loss in Eq. 1 instead of Eq. 3 yields substantially less stable performance and a lower
mean-return under the ground-truth reward function across all environments. Without _L_ [+] and _L_ _[−]_
from Eq. 3, the updated proxy reward function incorrectly assigns a higher reward to the suboptimal
actions taken by the reference policy. Appendix H.3 provides a qualitative analysis in the AI safety
gridworld, and Appendix G.4 reports further ablations of each regularization term. Conversely,
using PBRR’s objective (Eq. 3) within the RRM or RRM+State-Constraint baselines fails to match
PBRR’s data efficiency in all environments except Traffic Control. This shows that both PBRR’s
preference-learning objective and exploration strategy are important to efficiently repair the proxy
reward function with preferences.


7 CONCLUSION


We introduce Preference-Based Reward Repair (PBRR), a framework to repair a human-specified
proxy reward function by learning a transition-level correction from preferences. Across a diverse set
of benchmarks, spanning autonomous vehicle traffic control, pandemic lockdown regulation design,
and insulin regulation for diabetes patients, PBRR achieves higher performance with greater stability
and fewer preferences than methods that either learn a reward function from scratch or modify the
proxy reward function using alternative strategies. Our ablations show that both components of
PBRR are necessary: the exploration strategy that leverages a supplied reference policy to identify
transitions over which the proxy reward function is incorrect, and the preference-learning objective
that encourages only correcting the proxy reward function on transitions where it incorrectly assigns
high reward. Our theoretical analysis further shows that a variant of PBRR attains a sub-linear regret
bound comparable with prior work. Additional techniques for improving RLHF data efficiency—such
as data augmentation (Park et al., 2022) and intrinsic exploration rewards (Liang et al., 2022)—remain
untested in our domains and are orthogonal to our contributions. Future work could explore combining
these methods with PBRR to see if further gains in data efficiency are possible. Overall, our results
suggest that repairing a human-specified proxy reward function with PBRR, rather than learning a
reward function from scratch, is a data-efficient path to alignment in complex, sequential decision
making tasks.


9


Figure 3: Mean return under the ground-truth reward function achieved by PBRR, compared to (i) PBRR using
the standard preference-learning objective (Eq.1) instead of our proposed objective (Eq.3), and (ii) other methods
that repair the proxy reward function equipped with our proposed objective. See Figure 2 for plotting details.


**Reproducibility statement** The code to reproduce all experiments is attached with this submission
and will be released upon publication.


REFERENCES


Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané.
Concrete problems in ai safety. _arXiv preprint arXiv:1606.06565_, 2016.


Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain,
Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with
reinforcement learning from human feedback. _arXiv preprint arXiv:2204.05862_, 2022.


Peter L Bartlett, Michael I Jordan, and Jon D McAuliffe. Convexity, classification, and risk bounds.
_Journal of the American Statistical Association_, 101(473):138–156, 2006.


Hamsa Bastani, Mohsen Bayati, and Khashayar Khosravi. Mostly exploration-free algorithms for
contextual bandits. _Management Science_, 67(3):1329–1349, 2021.


Serena Booth, W Bradley Knox, Julie Shah, Scott Niekum, Peter Stone, and Alessandro Allievi. The
perils of trial-and-error reward design: misdesign through overfitting and invalid task specifications.
In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 37, pp. 5920–5929, 2023.


Chenyang Cao, Miguel Rogel-García, Mohamed Nabail, Xueqian Wang, and Nicholas Rhinehart. Residual reward models for preference-based reinforcement learning. _arXiv_ _preprint_
_arXiv:2507.00611_, 2025.


Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier
Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, et al. Open problems
and fundamental limitations of reinforcement learning from human feedback. _arXiv_ _preprint_
_arXiv:2307.15217_, 2023.


Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep
reinforcement learning from human preferences. _Advances_ _in_ _neural_ _information_ _processing_
_systems_, 30, 2017.


10


Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. Reward model ensembles help
mitigate overoptimization. _arXiv preprint arXiv:2310.02743_, 2023.


Nirjhar Das, Souradip Chakraborty, Aldo Pacchiano, and Sayak Ray Chowdhury. Provably sample
efficient rlhf via active preference optimization. _arXiv preprint arXiv:2402.10500_, 2024.


Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen
Sahoo, Caiming Xiong, and Tong Zhang. Rlhf workflow: From reward modeling to online rlhf.
_arXiv preprint arXiv:2405.07863_, 2024.


Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An
open urban driving simulator. In _Conference on robot learning_, pp. 1–16. PMLR, 2017.


Vikranth Dwaracherla, Seyed Mohammad Asghari, Botao Hao, and Benjamin Van Roy. Efficient
exploration for llms. _arXiv preprint arXiv:2402.00396_, 2024.


Brydon Eastman, Michelle Przedborski, and Mohammad Kohandel. Reinforcement learning derived
chemotherapeutic schedules for robust patient-specific therapy. _Scientific reports_, 11(1):17882,
2021.


Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D’Amour, DJ Dvijotham,
Adam Fisch, Katherine Heller, Stephen Pfohl, Deepak Ramachandran, et al. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. _arXiv_ _preprint_
_arXiv:2312.09244_, 2023.


Ian Fox, Joyce Lee, Rodica Pop-Busui, and Jenna Wiens. Deep reinforcement learning for closed-loop
blood glucose control. In _Machine Learning for Healthcare Conference_, pp. 508–536. PMLR,
2020.


Jiayi Fu, Xuandong Zhao, Chengyuan Yao, Heng Wang, Qi Han, and Yanghua Xiao. Reward shaping
to mitigate reward hacking in rlhf. _arXiv preprint arXiv:2502.18770_, 2025.


Amelia Glaese, Nat McAleese, Maja Tr˛ebacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth
Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, et al. Improving alignment of dialogue
agents via targeted human judgements. _arXiv preprint arXiv:2209.14375_, 2022.


Dylan Hadfield-Menell, Smitha Milli, Pieter Abbeel, Stuart J Russell, and Anca Dragan. Inverse
reward design. _Advances in neural information processing systems_, 30, 2017.


Robin Henry and Damien Ernst. Gym-anm: Reinforcement learning environments for active network
management tasks in electricity distribution systems. _Energy and AI_, 5:100092, 2021.


Kaixuan Ji, Jiafan He, and Quanquan Gu. Reinforcement learning from human feedback with active
queries. _arXiv preprint arXiv:2402.09401_, 2024.


Zhaohui Jiang, Xuening Feng, Paul Weng, Yifei Zhu, Yan Song, Tianze Zhou, Yujing Hu, Tangjie Lv,
and Changjie Fan. Reinforcement learning from imperfect corrective actions and proxy rewards.
_arXiv preprint arXiv:2410.05782_, 2024.


W Bradley Knox, Stephane Hatgis-Kessell, Serena Booth, Scott Niekum, Peter Stone, and Alessandro Allievi. Models of human preference for learning reward functions. _arXiv_ _preprint_
_arXiv:2206.02231_, 2022.


W Bradley Knox, Alessandro Allievi, Holger Banzhaf, Felix Schmitt, and Peter Stone. Reward (mis)
design for autonomous driving. _Artificial Intelligence_, 316:103829, 2023.


Varun Kompella, Roberto Capobianco, Stacy Jong, Jonathan Browne, Spencer Fox, Lauren Meyers,
Peter Wurman, and Peter Stone. Reinforcement learning for optimization of covid-19 mitigation
policies. _arXiv preprint arXiv:2010.10560_, 2020.


Victoria Krakovna, Jonathan Uesato, Vladimir Mikulik, Matthew Rahtz, Tom Everitt, Ramana
Kumar, Zac Kenton, Jan Leike, and Shane Legg. Specification gaming: the flip side of AI ingenuity. Google DeepMind Blog, April 2020. [URL https://deepmind.google/discover/](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)
[blog/specification-gaming-the-flip-side-of-ai-ingenuity/.](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) Published
April 21, 2020. Accessed August 5, 2025.


11


Cassidy Laidlaw, Shivam Singhal, and Anca Dragan. Correlated proxies: A new definition and
improved mitigation for reward hacking. _arXiv preprint arXiv:2403.03185_, 2024.


Cassidy Laidlaw, Shivam Singhal, and Anca Dragan. Correlated proxies: A new definition and
improved mitigation for reward hacking, 2025. URL [https://arxiv.org/abs/2403.](https://arxiv.org/abs/2403.03185)
[03185.](https://arxiv.org/abs/2403.03185)


Kimin Lee, Laura Smith, and Pieter Abbeel. Pebble: Feedback-efficient interactive reinforcement
learning via relabeling experience and unsupervised pre-training. _arXiv preprint arXiv:2106.05091_,
2021a.


Kimin Lee, Laura Smith, Anca Dragan, and Pieter Abbeel. B-pref: Benchmarking preference-based
reinforcement learning. _arXiv preprint arXiv:2111.03026_, 2021b.


Jan Leike, Miljan Martic, Victoria Krakovna, Pedro A Ortega, Tom Everitt, Andrew Lefrancq, Laurent
Orseau, and Shane Legg. Ai safety gridworlds. _arXiv preprint arXiv:1711.09883_, 2017.


Xinran Liang, Katherine Shu, Kimin Lee, and Pieter Abbeel. Reward uncertainty for exploration in
preference-based reinforcement learning. _arXiv preprint arXiv:2205.12401_, 2022.


Saaduddin Mahmud, Sandhya Saisubramanian, and Shlomo Zilberstein. Explanation-guided reward
alignment. In _IJCAI_, pp. 473–482, 2023.


Chiara Dalla Man, Francesco Micheletto, Dayu Lv, Marc Breton, Boris Kovatchev, and Claudio
Cobelli. The UVA/PADOVA type 1 diabetes simulator. _Journal of Diabetes Science and Technology_,
8(1):26–34, January 2014. ISSN 1932-2968. doi: 10.1177/1932296813514502. [URL https:](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4454102/)
[//www.ncbi.nlm.nih.gov/pmc/articles/PMC4454102/.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4454102/)


Henrik Marklund, Alex Infanger, and Benjamin Van Roy. Misalignment from treating means as ends.
_arXiv preprint arXiv:2507.10995_, 2025.


Lev McKinney, Yawen Duan, David Krueger, and Adam Gleave. On the fragility of learned reward
functions. _arXiv preprint arXiv:2301.03652_, 2023.


Malek Mechergui and Sarath Sreedharan. Expectation alignment: Handling reward misspecification
in the presence of expectation mismatch. _Advances in Neural Information Processing Systems_, 37:
62458–62479, 2024.


Viraj Mehta, Vikramjeet Das, Ojash Neopane, Yijia Dai, Ilija Bogunovic, Jeff Schneider, and Willie
Neiswanger. Sample efficient reinforcement learning from human feedback via active exploration.
_arXiv preprint arXiv:2312.00267_, 2023.


Katherine Metcalf, Miguel Sarabia, Natalie Mackraz, and Barry-John Theobald. Sampleefficient preference-based reinforcement learning with dynamics aware rewards. _arXiv preprint_
_arXiv:2402.17975_, 2024.


Qirui Mi, Siyu Xia, Yan Song, Haifeng Zhang, Shenghao Zhu, and Jun Wang. Taxai: A dynamic economic simulator and benchmark for multi-agent reinforcement learning. _arXiv preprint_
_arXiv:2309.16307_, 2023.


Ellen Novoseller, Yibing Wei, Yanan Sui, Yisong Yue, and Joel Burdick. Dueling posterior sampling
for preference-based reinforcement learning. In _Conference on Uncertainty in Artificial Intelligence_,
pp. 1029–1038. PMLR, 2020.


OpenAI. Chatgpt: Optimizing language models for dialogue. [OpenAI Blog https://openai.](https://openai.com/blog/chatgpt/)
[com/blog/chatgpt/, 2022.](https://openai.com/blog/chatgpt/) Accessed: 2022-12-20.


Stavros Orfanoudakis, Cesar Diaz-Londono, Yunus Emre Yılmaz, Peter Palensky, and Pedro P
Vergara. Ev2gym: A flexible v2g simulator for ev smart charging research and benchmarking.
_IEEE Transactions on Intelligent Transportation Systems_, 2024.


Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow
instructions with human feedback. _Advances in neural information processing systems_, 35:27730–
27744, 2022.


12


Aldo Pacchiano, Aadirupa Saha, and Jonathan Lee. Dueling rl: Reinforcement learning with trajectory
preferences. In _International conference on artificial intelligence and statistics_, pp. 6263–6289.
PMLR, 2023.


Alexander Pan, Kush Bhatia, and Jacob Steinhardt. The effects of reward misspecification: Mapping
and mitigating misaligned models. _arXiv preprint arXiv:2201.03544_, 2022.


Jongjin Park, Younggyo Seo, Jinwoo Shin, Honglak Lee, Pieter Abbeel, and Kimin Lee. Surf:
Semi-supervised reward learning with data augmentation for feedback-efficient preference-based
reinforcement learning. _arXiv preprint arXiv:2203.10050_, 2022.


Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style,
high-performance deep learning library. _Advances in neural information processing systems_, 32,
2019.


Zhenghao Mark Peng, Wenjie Mo, Chenda Duan, Quanyi Li, and Bolei Zhou. Learning from active
human involvement through proxy value propagation. _Advances in neural information processing_
_systems_, 36:77969–77992, 2023.


Brenden K Petersen, Jiachen Yang, Will S Grathwohl, Chase Cockrell, Claudio Santiago, Gary An,
and Daniel M Faissol. Deep reinforcement learning and simulation as a path toward precision
medicine. _Journal of Computational Biology_, 26(6):597–604, 2019.


John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy
optimization algorithms. _arXiv preprint arXiv:1707.06347_, 2017.


Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and
characterizing reward hacking. In _Proceedings of the 36th International Conference on Neural_
_Information Processing Systems_, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc.
ISBN 9781713871088.


Richard S. Sutton and Andrew G. Barto. _Reinforcement Learning:_ _An Introduction_ . MIT Press,
Cambridge, MA, 2 edition, 2018. ISBN 9780262039246. [URL http://incompleteideas.](http://incompleteideas.net/book/the-book-2nd.html)
[net/book/the-book-2nd.html. Contains the original statement of the Reward Hypothesis:](http://incompleteideas.net/book/the-book-2nd.html)
that all of what we mean by goals and purposes can be well thought of as the maximization of the
expected value of the cumulative sum of a received scalar signal (reward).


Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation
and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_, 2023.


Martin Treiber, Ansgar Hennecke, and Dirk Helbing. Congested traffic states in empirical observations
and microscopic simulations. _Physical review E_, 62(2):1805, 2000.


Mark J Willis. Proportional-integral-derivative control. _Dept. of Chemical and Process Engineering_
_University of Newcastle_, 6, 1999.


Cathy Wu, Abdul Rahman Kreidieh, Kanaad Parvate, Eugene Vinitsky, and Alexandre M Bayen.
Flow: A modular learning framework for mixed autonomy traffic. _IEEE Transactions on Robotics_,
38(2):1270–1286, 2021.


Tengyang Xie, Dylan J Foster, Akshay Krishnamurthy, Corby Rosset, Ahmed Awadallah, and
Alexander Rakhlin. Exploratory preference optimization: Harnessing implicit q*-approximation
for sample-efficient rlhf. _arXiv preprint arXiv:2405.21046_, 2024.


Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey
Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning.
In _Conference on robot learning_, pp. 1094–1100. PMLR, 2020.


Lingyu Zhang, Zhengran Ji, Nicholas Waytowich, and Boyuan Chen. Guide: Real-time human-shaped
agents. _Advances in Neural Information Processing Systems_, 37:138959–138980, 2024.


13


Tong Zhang. Statistical behavior and consistency of classification methods based on convex risk
minimization. _The Annals of Statistics_, 32(1):56–85, 2004.


Tan Zhi-Xuan, Micah Carroll, Matija Franklin, and Hal Ashton. Beyond preferences in ai alignment:
T. zhi-xuan et al. _Philosophical Studies_, 182(7):1813–1863, 2025.


Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul
Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. _arXiv_
_preprint arXiv:1909.08593_, 2019.


A WHY ELICIT PREFERENCES OVER TRAJECTORIES INSTEAD OF TRAJECTORY
SEGMENTS?


We focus on eliciting preferences over pairs of trajectories, rather than over pairs of trajectory
segments. In practice, eliciting preferences over full trajectories may introduce credit assignment
issues that are resolved when eliciting preferences over shorter trajectory segments. The latter is
more common than the former (e.g., Christiano et al. (2017)), and we hypothesize that this credit
assignment issue results in the noise we observe in our results in Section 6.


Despite the limitation of eliciting preferences over full trajectories rather than shorter segments, this
remains a more principled approach for simulating human feedback. In particular, Knox et al. (2022)
studied different models of human preferences, finding that the difference in regret between trajectory
segments is more predictive of the human preference label than the difference in the sum of rewards.
To simulate human preference labels—as we do in this work to avoid collecting preferences from real
humans—we therefore should label preferences in accordance with regret under the ground-truth
reward function. Unfortunately, for the real-world tasks we consider, computing the regret with
respect to the ground-truth reward function is exceedingly difficult and computationally expensive
due to the continuous state and action spaces. Consequently, we label preferences over trajectory
pairs by the difference in the sum of rewards. These trajectories begin and end in the same state.
Therefore, preferences follow the change-in-expected-return model, also proposed by Knox et al.
(2022).


Labeling preferences over shorter trajectory segments determined by the difference in sum of rewards—
which Knox et al. (2022) call the partial return preference model—- results in preference labels that
empirically do not match human judgments. We follow the recommendation of Knox et al. (2022)
and avoid using this partial return preference model to simulate preference labels.


The tradeoff however is that credit assignment for reward learning becomes more challenging due to
our reliance on eliciting preferences over full trajectories. We argue that simulating higher-fidelity
preference labels is more principled, as it better reflects how our methods would perform when no
ground-truth reward function is available and real human preferences must be elicited. In other words,
trajectory-level preferences allow us to focus on evaluating whether a method can repair a misspecified
proxy reward function, rather than conflating this with its capacity to learn a reward function from a
misspecified model of human preferences. See Knox et al. (2022) for further discussion on different
models of human preferences and their pitfalls, noting that the preference models they consider are
equivalent when eliciting preferences over full trajectories that begin and end in the same state in
MDPs with deterministic transition dynamics.


B ENVIRONMENT DETAILS


We use the same environments and configurations as Laidlaw et al. (2024) and Pan et al. (2022),
except for the AI safety gridworld. To make the task from the AI safety gridworld domain harder, we
reconfigure the placement of the tomatoes in the grid-world. Full implementation details, including
this grid-world configuration, are available in our codebase. We provide a brief description of each
environment, as well as its accompanying proxy reward function and reference policy below.


**Pandemic Mitigation** The agent controls the level of lockdown restrictions imposed on a population
by observing COVID-19 test outcomes, as simulated by a modified SEIR model (Kompella et al.,
2020). The proxy reward function captures epidemiological and economic outcomes but omits the


14


political cost associated with aggressive lockdown regulations. Optimizing for the proxy reward
function induces a policy that maintains a high lockdown level even when infection rates are low. The
supplied reference policy is trained via behavioral cloning on a combination of government strategies
used during the pandemic.


Observations are vectors of 312 continuous values; an action is a single discrete value in _{−_ 1 _,_ 0 _,_ 1 _}_ ;
the horizon is 192 time steps.


**Glucose Monitoring** The agent controls the insulin administered to a simulated patient with Type
I diabetes to maintain healthy glucose levels in an FDA approved simulator (Man et al., 2014; Fox
et al., 2020). The proxy reward function prioritizes reducing the financial cost of insulin and hospital
visits, while the ground-truth reward function is a standard measure of health risk. Optimizing for the
proxy reward function induces a policy that minimizes financial costs but not the patient’s overall
health risk. The supplied reference policy is trained via behavioral cloning on _only_ _a_ _handful_ of
demonstrations executed by a PID controller tuned by Willis (1999), illustrating a case where only a
handful of expert demonstrations are available.


Observations are vectors of 96 continuous values; an action is a single continuous value in [0 _,_ 1]; the
horizon is 5760 time steps.


**Traffic Control** The agent controls a fleet of autonomous vehicles on an on-ramp attempting to
merge into traffic on a highway with simulated human drivers (Wu et al., 2021). The proxy reward
function prioritizes maximizing the mean velocity of all vehicles. When optimized for, it induces
a policy where the autonomous vehicles block the on-ramp, allowing highway traffic to maintain
maximum speed while preventing other vehicles from entering. The ground-truth reward function
instead prioritizes the mean commute time of all vehicles. The supplied reference policy is trained
via behavioral cloning on _only a handful_ of demonstrations executed by an Intelligent Driver Model
(Treiber et al., 2000) aimed to mimic human driving.


Observations are vectors of 50 continuous values; actions are vectors of 10 continuous values in

[0 _,_ 1]; the horizon is 300 time steps.


**AI Safety Gridworld** In the only toy-task we consider, the agent moves around a grid-world with
the objective of watering all tomatoes by visiting tomato-containing states. There exists a sprinkler
state that makes the tomatoes look watered, i.e., the agent attains positive reward under the proxy
reward function, when the tomatoes are not actually watered, i.e., the agent attains no reward under
the ground-truth reward function. This environment was introduced by Leike et al. (2017). The
supplied reference policy is trained using the ground-truth reward function in a grid-world layout that
contains only a small subset of the tomatoes present in the task we use for evaluation.


Observations are vectors of 36 discrete values; an action is a single discrete value in _{_ 0 _,_ 1 _,_ 2 _,_ 3 _}_ ; the
horizon is 100 time steps.


C OPTIMIZING FOR _r_ ˆ TO AVOID REWARD HACKING


Some prior work (e.g., (Ziegler et al., 2019; Ouyang et al., 2022; Bai et al., 2022; Glaese et al., 2022;
OpenAI, 2022; Touvron et al., 2023)) assumes access to a reference policy _π_ ref, and then optimizes
for the following objective:
maximize _Jr_ ˆ( _π_ ) _−_ _βF_ ( _π, π_ ref ) (6)


where _F_ is some measure of divergence between _π_ and _π_ ref . For example, _F_ can be defined as the
expected KL divergence between the action distributions of _π_ and _π_ ref :


Laidlaw et al. (2024) proposes other divergence measures that dictate how _π_ can deviate from
_π_ ref, which we consider in our empirical results. Eq. 6 applies a divergence penalty between the
learned policy _π_ and _π_ ref to balance between optimizing for _r_ ˆ without straying too far from a
known behavior distribution, i.e., _π_ ref . While this approach can attain better performance under the
ground-truth reward function than only optimizing for the misspecified proxy reward function, it


15


- _∞_


_F_ ( _π, π_ ref ) = (1 _−_ _γ_ )E _π_


_∞_ 
- _γ_ _[t]_ _D_ KL� _π_ ( _· | st_ ) �� _π_ ref ( _· | st_ )�


_t_ =0


requires a sufficiently performant reference policy or proxy reward function ˆ _r_ to attain near-optimal
performance under the ground-truth reward function. Otherwise, constraining the learned policy to be
close to the reference policy as defined by _F_ will fundamentally limit the performance of the resulting
policy. On the other hand, reducing the divergence penalty by lowering _β_ can weaken the constraint
on _π_, allowing it to exploit misspecifications in ˆ _r_ and potentially learn a policy that is substantially
sub-optimal with respect to the ground-truth reward function. For this reason, Laidlaw et al. (2024)
use a reasonably performant reference policy in their experiments. In contrast, we operate in a setting
where no such policy exists; under these conditions, optimizing Eq. 6 with any of the divergence
measures considered by Laidlaw et al. (2024) and any choice of _β_ is unlikely to yield near-optimal
performance with respect to _r_ . Therefore, in our setting, ˆ _r_ itself must be updated.


D BASELINES FOR LEARNING AND REPAIRING REWARD FUNCTIONS


Each baseline described below either learns a reward function _ab_ _initio_ or assumes access to a
manually specified proxy reward function and learns an additive correction term _g_ ( _s, a, s_ _[′]_ ) as used in
Equation 2. In both cases, the additive correction term and the reward function are parametrized as
a neural network or an ensemble of neural networks, depending on the baseline. Here we describe
how each baseline constructs a batch of trajectory pairs to elicit preferences over. Upon eliciting
preferences, the resulting ( _τ_ 1 _, τ_ 2 _, µ_ ) samples are added to dataset _Dt_ and—unless otherwise stated—
the reward function or correction term parameters are updated using the standard preference loss in
Eq. 1 given _Dt_ .


Our approach and all baselines use Proximal Policy Optimization (PPO) (Schulman et al., 2017) to
train a policy with the current estimate of the reward function ˆ _r_ . Like prior RLHF approaches (e.g.,
Christiano et al. (2017); Lee et al. (2021a;b)), we decouple reward learning and policy learning: PPO
samples environment rollouts independently of those used to update the reward function. As a result,
differences between methods lie primarily in their reward modeling and data collection strategies,
rather than in their policy optimization.


D.1 ONLINE-RLHF BASELINE


This baseline adapts the methodology of Christiano et al. (2017) and Lee et al. (2021a) to our problem
setting. At each iteration, an ensemble of randomly initialized reward models is updated using all
collected preferences over trajectories. A new batch of preferences for elicitation is constructed as
follows:


At each iteration we sample up to 200 trajectories from the current policy, which are then added to a
candidate batch. For a more fair comparison with our approach that assumes access to a safe reference
policy, the candidate batch also contains trajectories sampled from the supplied reference policy.
We note that the candidate batch initially contains trajectories sampled from the reference policy—
-in addition to a policy trained with a randomly initialized reward model—rather than trajectories
sampled from a policy pre-trained with the state-entropy objective proposed by Lee et al. (2021a),
which is a follow-up work to Christiano et al. (2017); this baseline harnesses the safe reference policy
for exploration.


All possible pairs of trajectories are constructed from the candidate batch, and the top _k_ pairs with the
highest variance over the predicted preference probabilities—computed via preference model _P_ ( _.|r_ ˜)
with respect to the reward model ensemble—are labeled with preferences. Note that this baseline
may elicit preferences between trajectories sampled from the reference policy and the current policy
if those pairs have the highest variance in predicted preference probability.


D.2 RESIDUAL REWARD MODEL (RRM) BASELINE


This baselines mirrors the method of Cao et al. (2025), and matches the Online-RLHF baseline except
a correction term _g_ for Eq. 2 is learned instead of learning a reward function _ab initio_ . A tanh
function is also applied to the output of the learned correction term _g_ to match the implementation of
Cao et al. (2025). The primary difference between this baseline implementation and Cao et al. (2025)
is that they adopt Soft Actor-Critic (SAC) while we use Proximal Policy Optimization (PPO) for a
stringent comparison with our other baselines.


16


D.3 STATE-CONSTRAINT REWARD LEARNING BASELINES


Here we describe both the Online-State-Constraint and RRM + State-Constraint baselines. These
baselines adapt recent insights from alignment research in large language model (LLM) settings,
which suggest that constraining the learned policy to remain close to a reference policy can improve
performance when learning from preferences. We build on the methodology proposed by Dong et al.
(2024), modifying it to suit our setting. In particular, unlike Dong et al. (2024), our policies are
not parameterized as LLMs and therefore we cannot exploit stochastic decoding to generate diverse
outputs for preference elicitation. To get around this, we sample trajectories from two different
policies instead; we sample one trajectory from the learned policy and one from the reference policy.


At each iteration, after learning a reward function from the constructed dataset of preferences, we
learn a policy from that reward function using the divergence-regularized objective in Eq. 6. We
follow the insights of Laidlaw et al. (2024) and select the divergence measure to be the one that
induces the best performance as outlined in Appendix E.4.


We consider two baselines that follow this approach; one that learns a reward function _ab_ _initio_
and one that repairs an existing reward function. For each baseline we additionally implement two
versions: one that keeps _π_ ref fixed—as presented in Section 6, and one that updates _π_ ref to be the
policy learned in the previous iteration—as presented in Appendix E.3.

**Online-State-Constraint** At each iteration _t_, we roll-out the reference policy _π_ ref _[t]_ [and the current]
learned policy _πconst_ _[t]_ [, where] _[ π]_ _const_ _[t]_ [is found by optimizing for the objective in Eq.] [6 with] _[r]_ [ˆ] _[t]_ [and]
_π_ ref _[t]_ [.] [The specific divergence measure used for this o] _√_ [bje][ctive is chosen as the measure that induces]
the best performance as described in Section E.4. _k_ trajectories are then sampled from _π_ ref _[t]_ [and]

_πconst_ _[t]_ [respectively, and] _[ k]_ [ exhaustive][ (] _[τ]_ [1] _[, τ]_ [2][)][ pairs are constructed where] _[ τ]_ [1] _[∼]_ _[π]_ ref _[t]_ [and] _[ τ]_ [1] _[∼]_ _[π]_ _const_ _[t]_ [.]
Preferences are elicited over all _k_ pairs and added to the dataset _D_ . Depending on the baseline variant,
the reference policy at the next iteration is moved such that _π_ ref _[t]_ [+1] = _π_ ref _[t]_ [or] _[ π]_ ref _[t]_ [+1] = _πconst_ _[t]_ [.]


**RRM + State-Constraint** This baseline follows the same procedure as described above, with the
exception that a correction term _g_ is learned and applied to the initial proxy reward function, rather
than learning a reward function _ab initio_ .


E IMPLEMENTATION DETAILS


All experiments were implemented with Python 3.9 and PyTorch 2.7 (Paszke et al., 2019), largely
building off of the codebase provided by Laidlaw et al. (2024).


E.1 POLICY AND REWARD MODEL ARCHITECTURES


We use the same policy network architectures as Laidlaw et al. (2024). For the pandemic, traffic,
and AI safety gridworld environments, we use feedforward policy networks with the following
architectures: two hidden layers of 128 units for the pandemic environment, four hidden layers of
512 units for the traffic environment, and four hidden layers of 512 units for the AI safety gridworld
environment. In the glucose environment, the policy is implemented as a three-layer LSTM, each
layer containing 64 units.


When learning a reward function _ab initio_, we parameterize the reward function as a feedforward
neural network with 5 hidden layers of 256 units for the pandemic environment and 5 hidden layers
of 512 units for the glucose, traffic, and AI safety gridworld environments. For the Online-RLHF
baseline, we maintain an ensemble of 5 reward functions. When learning a corrective term _g_ for a
specified proxy reward function, we parameterize _g_ as a feedforward neural network with 5 hidden
layers of 512 units for the pandemic, glucose, and AI safety gridworld environments, and 3 hidden
layers of 256 units for the traffic environment. For the RRM baseline, we maintain an ensemble of 3
reward functions. Table 2 shows the additional hyperparameters used for training the reward function
and correction term. The Adam optimizer was used to learn the reward function or correction term
parameters with default Pytorch hyperparameter values.


All hyperparameters listed above were found via the following process: for each environment, we
constructed a dataset of exhaustive preferences between trajectories sampled from a policy trained
with the ground-truth reward function and the proxy reward function respectively, and trajectories


17


Table 2: Hyperparameters used for learning a reward function _ab initio_ and learning a correction term to repair a
proxy reward function across all environments.


**Environment** **Repair Proxy Reward Function?** **Learning Rate** **Weight Decay** **Epochs**


Pandemic True 0.001 False 200
Pandemic False 0.0001 False 200
Glucose True 0.0001 False 200
Glucose False 0.0001 True 200
Traffic True 0.0001 True 50
Traffic False 0.001 False 50
AI safety gridworld True 0.0001 True 200
AI safety gridworld False 0.0001 True 200


sampled from the reference policy. Separately for learning a reward function from scratch and a
corrective term for Eq. 2, we performed a grid-search over the following hyperparameter candidates
when learning from the constructed dataset of preferences:


    - learning rate: [0.001, 0.001]

    - weight decay: [True, False]

    - number of epochs: [50, 100, 200]

    - number of hidden layers: [3, 5]

    - number of units per hidden layer: [256, 512]


We chose the hyperparameter values that invoked the lowest loss on a held out test-set. Our objective
was to identify the hyperparameters that most effectively enabled the learned reward function to
distinguish between trajectories of varying optimality.


E.2 POLICY AND REWARD MODEL INITIALIZATION


For the glucose, traffic, and pandemic environments, at each iteration, i.e., before any policy is
learned, we always initialize the policy weights to be the reference policy weights. Following the
methodology of Laidlaw et al. (2024), we randomly initialize the policy weights for the AI safety
gridworld environment because otherwise we do not observe reward hacking when optimizing for the
initial proxy reward function.


For all environments and experiments, at each iteration we re-initialize the reward function or
correction term’s weights. We then re-train the reward function or correction term on all collected
samples ( _τ_ 1 _, τ_ 2 _, µ_ ) _∈D_ = [�] _t_ _[D][t]_ [, not just the most recently collected batch of preferences] _[ D][t]_ [.]


E.3 CONTROLLING DIVERGENCE FROM _π_ REF


For the State-Constrained-PPO, Online-State-Constraint, and RRM + State-Constraint baselines,
we optimize for the objective in Eq. 6 when constructing a policy. To attain the best performance
possible given only the proxy reward function and a reference policy, we pick the divergence measure
that induces the highest performance under the ground-truth reward function. This methodology,
via privileged access to the ground-truth reward function, ensures that the State-Constrained-PPO
baseline induces the best performance achievable without updating the proxy reward function. Note
that only privileged access to the ground-truth reward function is used to select the divergence
measure parameters for the State-Constrained-PPO, Online RLHF + State-Constraint, and RRM +
State-Constraint baselines; our approach never uses privileged information.


For the Pandemic environment, we use the same reference policy as Laidlaw et al. (2024), and therefore choose the divergence measure _F_ and constant _β_ from Eq. 6 that induces the best performance
in their paper: _F_ is the KL-divergence between state occupancy measures and _β_ = 0 _._ 06. For all
other environments, we perform a joint search over the choice of occupancy measure—either state
or state-action occupancy—and the divergence measure and _β_ coefficient specified in Table 3. For
each combination, we evaluate performance using the ground-truth reward function and select the


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


setting that yields the highest mean return across 3 random seeds. For the Traffic environment, _F_
is the KL-divergence between state-action occupancy measures and _β_ = 0 _._ 0005; for the Glucose
environment, _F_ is the KL-divergence between state-action occupancy measures and _β_ = 0 _._ 06; for the
AI safety gridworld environment, _F_ is the KL-divergence between state-action occupancy measures
and _β_ = 0 _._ 08. Note that our search did not include using action-occupancy measures due to their
consistently inferior performance in Laidlaw et al. (2024).


Table 3: Coefficient values we searched over for each divergence measure, environment, and occupancy measure.


**Environment** **Divergence** **Coefficient Values**


1 2 3 4 5 6 7


AI safety gridworld ~~�~~ _χ_ [2] 0.0005 0.001 0.0025 0.005 0.01 0.025 0.05
AI safety gridworld KL 0.8 0.4 0.16 0.08 0.04 0.016 0.008
Traffic ~~�~~ _χ_ [2] 2e-6 4e-6 1e-5 2e-5 4e-5 1e-4 2e-4
Traffic KL 0.005 0.0025 0.001 0.0005 0.00025 0.0001 0.00005
Glucose ~~�~~ _χ_ [2] 0.0005 0.001 0.0025 0.005 0.01 0.025 0.05
Glucose KL 0.0015 0.003 0.006 0.015 0.03 0.06 0.15


E.4 HYPERPARAMETERS FOR PPO


We used the same hyperparameters for PPO as Laidlaw et al. (2024), with two exceptions. To reduce
the running time of our approach and all baselines that learn a reward function, which each require
training at least one new policy per reward-learning iteration, we reduce the number of PPO training
iterations and the PPO batch size. We ensure that with the reduced training batch size and PPO
training iterations, optimizing for the proxy reward function still consistently induces reward hacking
behavior, and optimizing for the ground-truth reward function still consistently induces the highest
achievable expected return or is less than a standard deviation away. The reduced number of iterations
and training batch sizes for each environment are shown in Table 4.


Table 4: PPO training hyperparameters for each environment that differ from the ones originally used by Laidlaw
et al. (2024).


**Environment** **PPO Batch Size** **Training Iterations**


Pandemic 3,840 100
Glucose 10,000 150
Traffic 80,000 100
AI safety gridworld 1,000 100


E.5 LEARNING FROM PREFERENCES EXPERIMENTAL DETAILS


**How many preferences are elicited per iteration?** For each environment, we elicit _k_ [2] preferences
per iteration. If a method rolls out two policies _π_ 1 and _π_ 2 per iteration, e.g., PBRR samples
trajectories from _π_ ref and _πr_ _[∗]_ ˆ _t_ [, then] _[ k]_ [ trajectories are sampled from each policy and preferences are]
elicited between all _k_ [2] pairs ( _τ_ 1 _, τ_ 2) where _τ_ 1 _∼_ _π_ 1 and _τ_ 2 _∼_ _π_ 2. All baselines that roll out a single
policy per iteration select _k_ [2] trajectory pairs from a candidate batch via ensemble-based uncertainty
estimates (see Appendix D for details on how this candidate batch is constructed for each relevant
method). _k_ = 19 for the AI safety gridworld and traffic environments, _k_ = 39 for the glucose
environment, and _k_ = 79 for the pandemic environment. These values were chosen to balance
between distinguishing the sample efficiency of different methods and limiting the computational
cost of training a new policy after every reward function update.


**How are trajectories for preference elicitation constructed?** For all environments except glucose,
we elicit preferences over full trajectories of length _H_ : _H_ = 100 for AI safety gridworld, _H_ = 192
for pandemic, and _H_ = 300 for traffic. In the glucose environment, episodes span _H_ = 5760
timesteps. If a simulated patient dies, the episode enters an absorbing state where all observations,


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


actions, and rewards are zero until the horizon is reached. To avoid out-of-memory issues when
learning from preferences, we split each glucose trajectory into three equal segments (each of length
1920), discarding any segments that consist entirely of absorbing-state transitions. Preferences are
then elicited over the remaining segments.


**How are preferences labeled?** We label all trajectory pairs with preference labels sampled from
the Boltzman distribution given the ground-truth reward function. We use _γ_ = 0 _._ 99 when labeling
preferences using _r_, and when learning _r_ ˆ from preferences. This is the discount factor used by
Laidlaw et al. (2024) when learning from _r_ for the same benchmark environments.


E.6 PBRR _λ_ 1 AND _λ_ 2


In Section 4 we outline how the proxy reward function is updated with PBRR via the preferencelearning objective in Eq. 3. In practice, we set _λ_ 1 = _λ_ 2 = 10, and then divide both terms by the
number of trajectory pairs collected where the proxy reward function’s induced ranking agrees with
the human preference label, _|D_ [+] _|_ . In effect, this decays _λ_ 1 and _λ_ 2 as the proxy reward function is
repaired. Specifically, at iteration _i_, _λ_ _[i]_ 1 [=] _[ λ]_ 2 _[i]_ [=] _|D_ 10 [+] _|_ [.]


F ONLINE-RLHF ABLATIONS


A line of prior work (Lee et al., 2021a; Liang et al., 2022; Metcalf et al., 2024; Park et al., 2022)
explored how to improve the data efficiency of RLHF methods across a suite of robotics control tasks.
The techniques introduced in these works are complementary to our proposed approach, PBRR, and
could in principle be combined with it. We leave such an exploration for future work. In this section,
we examine whether the data-efficiency strategies proposed by Lee et al. (2021a); Liang et al. (2022);
Metcalf et al. (2024) substantially enhance the initial performance of the Online-RLHF baseline. We
find that PBRR continues to outperform the initial performance achieved by Online-RLHF, even
when these strategies are applied. We compare to the following methods:


**Online-RLHF** The baseline is used for the main results in Section 6.


**Online-RLHF + Lee et al. (2021a)** Identical to the Online-RLHF baseline, except that, following
the unsupervised pre-training procedure introduced by Lee et al. (2021a), we first train an exploration
policy using their entropy-maximization objective. Trajectories collected from this exploration policy
are then added to the candidate batch used to construct trajectory pairs for preference elicitation. As
in Online-RLHF, the candidate batch also includes trajectories sampled from the reference policy.


**Online-RLHF + Liang et al. (2022)** Identical to the Online-RLHF + Lee et al. (2021a) baseline,
except that, following the approach of Liang et al. (2022), the learned reward function is augmented
with an exploration bonus term that encourages the policy to visit states where the reward estimate is
uncertain. This exploration bonus is gradually decayed to 0 once half the total number of preferences
used in Figure 2 have been collected.


As shown in Figure 4, PBRR consistently outperforms the initial performance achieved by the OnlineRLHF baselines, even when augmented with these data-efficiency techniques, for all environments
except Traffic Control. In Traffic Control, PBRR matches the performance of Online-RLHF + Liang
et al. (2022) after eliciting a single batch of preferences. We hypothesize that exploration procedures
designed to maximize state coverage (e.g., Lee et al. (2021a)) or reward uncertainty (e.g., Liang et al.
(2022)) may be less effective in non-ergodic MDPs where most states correspond to undesirable
outcomes. For instance, in the Glucose Monitoring environment, many treatment policies yield
poor health outcomes for the patient, and exploring such regions of the state space—even if those
regions induce the most uncertainty in predicted reward—may provide little information about what
constitutes a good treatment policy.


We additionally compare PBRR against the following baseline in the AI safety gridworld:


**Online-RLHF + Metcalf et al. (2024)** Identical to the Online-RLHF + Lee et al. (2021a) baseline,
except that, following the approach of Metcalf et al. (2024), a self-supervised dynamics representation
is jointly learned with the reward model parameters using their proposed training procedure.


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


Figure 4: Mean return under the ground-truth reward function achieved by the Online-RLHF baseline and its
variants that utilize proposed data-efficiency techniques from Lee et al. (2021a) and Liang et al. (2022). See
Figure 2 for plotting details.


The results in Figure 5 show that Online-RLHF + Metcalf et al. (2024) underperforms PBRR and
Online-RLHF; we suspect that the learned dynamics aware representation is not sufficient to learn a
reward function over.


Figure 5: Mean return under the ground-truth reward function achieved by the Online-RLHF baseline and the
Online-RLHF + Metcalf et al. (2024). See Figure 2 for plotting details.


G PLOTTING DETAILS AND ADITIONAL RESULTS


G.1 PLOTTING DETAILS


For Figures 2 and 3, for each seed we roll out a policy for the number of episodes specified in
Appendix Table 5. We compute the mean return over all episodes with respect to the ground-truth
reward function. We then scale the mean return for policy _π_ as follows:


_J_ ˆ _r_ -scaled( _π_ ) = _J_ ˆ _Jr_ ˆ(( _ππr_ _[∗]_ ) ) _−−J_ [ˆ] _Jr_ [ˆ] _r_ ( _π_ ( _π_ refref)) [:]


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


where _J_ [ˆ] _r_ is the empirical mean return under the ground-truth reward function _r_ . We clip _J_ [ˆ] _r_ -scaled( _π_ )
to be in range [ _−_ 1 _,_ 1]. We perform this clipping for visual clarity, noting that any policy with a scaled
return less than 0 does worse than the supplied reference policy, and any policy with a scaled return
less than _−_ 1 is considerably sub-optimal. In Figures 2 and 3, we plot the resulting mean scaled return
over 3 seeds.


**Environment** **Number of Episodes**
Traffic 260
Pandemic 20
Glucose 20
AI safety gridworld 10


Table 5: Number of episodes used to compute the mean return.


G.2 RETRAINING A POLICY WITH AN UPDATED PROXY REWARD FUNCTION


McKinney et al. (2023) show that reward functions learned online from human feedback can fail to
re-train new policies initialized from scratch with a different random seed. To test whether PBRR’s
updated proxy reward function exhibits similar fragility, we reinitialize new policies with different
seeds and train the policies using the repaired reward functions learned in Section 6. We also extend
PPO training beyond the number of PPO steps used when learning the additive correction term with
PBRR, probing whether the updated proxy reward function could induce undesirable behavior if
optimized for longer. Figure 6 presents these results: in the Traffic Control, Glucose Monitoring,
and Pandemic environments, retraining with the repaired proxy reward function still yield policies
that induce near-optimal performance. In the Pandemic Mitigation environment, training PPO for
more steps even yields an improvement. In the Glucose Monitoring environment, however, the
performance of the policy induced by the repaired proxy reward function deteriorates when trained
with substantially more RL steps than were used during reward learning. These findings broadly
suggest that PBRR’s repaired proxy reward function is robust to policy reinitialization and to longer
RL optimization, but the number of RL steps used while updating the proxy reward function can play
a critical role in determining whether the learned signal remains valid under extended training. In
other words, when re-training a new policy with a repaired proxy reward function, it is likely best to
train that policy with the same number of RL steps as used when repairing the proxy reward function.


Figure 6: Mean return under the ground-truth reward function achieved by re-training a newly initialized policy
with the repaired proxy reward function learned by PBRR. The vertical line marks the number of steps used
to train the policies when updating the proxy reward function in Algorithm 1; performance beyond this point
illustrates the robustness of that updated proxy reward function to extended optimization. Results are averaged
over trajectories sampled from policies trained with the updated proxy reward function across 3 random seeds.


G.3 COMPARISONS AGAINST BASELINES THAT UPDATE _π_ REF


Figure 2 compares PBRR against the Online-State-Constraint and RRM + State-Constraint baselines
respectively, which each constrain the learned policy to _π_ ref. Section D details these baseline
implementations. Here we compare PBRR against those baselines, except we update _π_ ref every
iteration to be the policy constructed at the previous iteration as outlined in Section D. We refer to
these additional baselines as Online-RLHF + Moving-State-Constraint and RRM + Moving-StateConstraint.


Our goal with these additional baselines is to attempt to overcome a fundamental limitation of the
Online-State-Constraint and RRM + State-Constraint baselines, namely that they may fundamentally


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


limit the performance of the induced policy by penalizing divergences from a fixed _π_ ref. Figure 7
below shows that updating _π_ ref as described in Section D does not lead to a substantial improvement
in performance; PBRR still remains the most performant approach in all environments.


Figure 7: Mean return under the ground-truth reward function achieved by the Online-State-Constraint and RRM
+ State-Constraint baselines, and variants of those baselines that update _π_ ref used for the state-constraint at each
iteration as described in Section D. See Figure 2 for plotting details.


G.4 ABLATING PBRR’S REGULARIZATION TERMS


PBRR’s preference learning objective in Eq. 3 consists of two regularization terms, _L_ _[−]_ and _L_ [+] . In
Figure 3 we empirically show that, without these regularization terms, PBRR performs relatively
poorly in both stability and achieved mean return. Here we investigate the individual impact of
_L_ _[−]_ and _L_ [+] respectively. Figure 8 illustrates that PBRR with both regularization terms matches
or outperforms all alternatives across environments. The relative effect of using only _L_ _[−]_ or _L_ [+] is
environment dependent. PBRR’s performance is only degraded by removing a regularization term in
the Pandemic Control and Glucose Monitoring environment: removing _L_ [+] causes the larger drop in
Pandemic, while removing _L_ _[−]_ has the greater effect in Glucose.


G.5 PBRR WITH INTRA-POLICY PREFERENCES


PBRR, as implemented for the results in Section 6, elicits preferences over trajectory pairs where
one trajectory is sampled from _πr_ ˆ _t_ and the other from _π_ ref. Here we investigate adding intra-policy
preferences, i.e., preferences over trajectory pairs where both trajectories are sampled from _πr_ ˆ _t_ .
Figure 9 plots the results when half of the elicited batch is intra-policy preferences. We find that
including these preferences degrades data-efficiency and stability. In particular, the performance in
the early iterations of the PBRR + intra-policy preferences approach is worse than PBRR in the AI
Safety Gridworld and Pandemic environment, and performance decreases in later iterations in the
Glucose environment. For the Pandemic environment, PBRR + intra-policy preferences eventually
outperforms PBRR, suggesting a potential benefit to including intra-policy preferences. Future work
should explore how to leverage this additional data to improve PBRR’s performance without reducing
data-efficiency or learning stability in environments where PBRR already performs well.


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


Figure 8: Mean return under the ground-truth reward function achieved by PBRR, compared to PBRR using the
standard preference-learning objective _L_ pref (Eq.1) with (i) only the _L_ _[−]_ regularization term, and (ii) only the
_L_ [+] regularization term. Note that PBRR uses the preference learning objective in Eq. 3 with all three terms:
_L_ pref + _λ_ 1 _L_ [+] + _λ_ 2 _L_ _[−]_ . See Figure 2 for plotting details.


Figure 9: Mean return under the ground-truth reward function achieved by PBRR, compared to PBRR + intrapolicy preferences. See Figure 2 for plotting details.


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


G.6 PBRR WITH A PESSIMISTIC PROXY REWARD FUNCTION


In Section 4 we discuss a key assumption behind PBRR’s preference learning objective: the humanspecified proxy reward function is either aligned or optimistic. Here we investigate PBRR’s performance in the AI safety gridworld when repairing a pessimistic proxy reward function. In particular,
we construct a new proxy reward function that is the same as the proxy reward function used for the AI
safety gridworld in Section 6, except that it assigns a negative reward for visiting [3] 9 [tomato-containing]

states—which the ground-truth reward function assigns a positive reward for visiting. As such, this
proxy reward function breaks our optimism assumption. We plot the results in Figure 10. PBRR
eventually attains near-optimal performance but requires more preferences than when repairing an
optimistic proxy reward function, as assumed in Figure 2. We note that even when the optimism
assumption does not hold, PBRR still outperforms other methods for repairing the proxy reward
function.


Figure 10: Mean return under the ground-truth reward function achieved by PBRR compared to the baselines
from Section 6.2. The proxy reward function here does not follow the optimism assumption leveraged by PBRR.
See Figure 2 for plotting details.


G.7 UNSCALED MAIN RESULTS


In Figures 2 and 3 we scale and clip the plotted mean return, as detailed in Appendix G.1, for
clearer visual comparison. In Figure 11 we plot the unclipped, unscaled mean-return for PBRR
and the baselines that learn a reward function _ab initio_ —Online-RLHF and Online-State-Constraint.
In Figure 12 we plot those results for PBRR and the baselines that learn to repair a proxy reward
function—RRM and RRM + State-Constraint. In Figure 13 we plot those results for the ablations
from Figure 3.


G.8 PBRR WITH A RANDOMLY INITIALIZED REFERENCE POLICY


PBRR requires a reference policy for preference elicitation. For all prior experiments, unless
otherwise stated, we use a realistic reference policy that a human could provide such as one trained
with a handful of human demonstrations or from imperfect hand-written rules. Here, we investigate
PBRR’s performance when using a randomly initialize reference policy instead. Figure 14 shows that
performance is largely unchanged; the quality of the reference policy used by PBRR is not judged by
its performance, but rather its coverage of the state-action space relative to the policy induced by the
proxy reward function. These results imply that for many tasks, a sufficient reference policy should
always be available by simply using a randomly initialized policy.


G.9 EVALUATING PBRR OVER ADDITIONAL RANDOM SEEDS


We aim to establish 95%-confidence intervals for the main results in Figure 2. As a step towards
this, we evaluate PBRR across 10 seeds in the Pandemic Mitigation environment. We compare
PBRR against the two best performing baselines—Online-RLHF and RRM. Due to computational
constraints, we restrict this analysis to the first two updates of the proxy reward function. As shown in
Table 6, after the first update PBRR produces a reward function that yields statistically significantly


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


Figure 11: Complementing Figure 2: Mean return under the ground-truth reward function achieved by PBRR
compared to baselines that learn a reward function _ab initio_ from preferences, averaged over trajectories sampled
from policies trained with the learned reward function across 3 random seeds. Shaded regions indicate the
standard error. No scaling or clipping is applied to the plotted mean return values.


Figure 12: Complementing Figure 2: Mean return under the ground-truth reward function achieved by PBRR
compared to other approaches that repair the proxy reward function with preferences, averaged over trajectories
sampled from policies trained with the updated proxy reward function across 3 random seeds. No scaling or
clipping is applied to the plotted mean return values.


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


Figure 13: Complementing Figure 3: Mean return under the ground-truth reward function achieved by PBRR,
compared to (i) PBRR using the standard preference-learning objective (Eq.1) instead of our proposed objective
(Eq.3), and (ii) other methods that repair the proxy reward function equipped with our proposed objective. The
mean return is averaged over trajectories sampled from policies trained with the updated proxy reward function
across 3 random seeds. No scaling or clipping is applied to the plotted mean return values.


**Method** **Initial Mean Return** **After Update 1** **After Update 2**


Online-RLHF _−_ 103 _._ 09 _±_ 77 _._ 24 _−_ 141 _._ 44 _±_ 86 _._ 12 _−_ 65 _._ 27 _±_ 61 _._ 81
RRM _−_ 39 _._ 87 _±_ 12 _._ 82 _−_ 9 _._ 39 _±_ 2 _._ 80 _−_ 15 _._ 37 _±_ 6 _._ 75
PBRR _−_ 26 _._ 81 _±_ 4 _._ 96 _−_ 7 _._ 91 _±_ 1 _._ 17 _−_ 7 _._ 21 _±_ 0 _._ 84


Table 6: Pandemic Monitoring: mean return over 10 seeds ( _±_ 95% CI) before and after proxy reward function
updates.


better performance than Online-RLHF, and after the second update it achieves statistically significantly
better performance than RRM.


H AI SAFETY GRIDWORLD QUALITATIVE ANALYSIS


Figure 16 shows the board layout we use for the AI safety gridworld environment. Here we qualitatively analyze why some approaches fail to learn a performant policy in this simple environment.
While the other environments are less interpretable than AI safety gridworld, we suspect the methods
we consider in this section may share poor performance in those environment for similar reasons.


H.1 WHY DOES ONLINE-RLHF PERFORM POORLY?


The results in Figure 2 show that the Online-RLHF baseline exhibits oscillatory performance; its
performance with respect to the ground-truth reward function decreases upon acquiring new preferences, only to increase again in subsequent iterations. This instability reflects the difficulty of
reward learning in these reward-hacking benchmark environments. For instance, after acquiring
1,800 preferences in AI Safety Gridworld, the Online-RLHF baseline observes many trajectory
pairs in which visiting more tomato-containing states is always preferred to visiting fewer. As a
result, it learns a reward function that assigns positive reward whenever the agent enters a tomato

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


Figure 14: Mean return under the ground-truth reward function achieved by PBRR with the default reference
policy used for all other experiments (PBRR ( _π_ ref _−_ default)), compared to (i) PBRR using a randomly initialized
reference policy (PBRR ( _π_ ref _−_ random-init)), and (ii) the Online-RLHF method. The plot for the Pandemic
Mitigation environment is truncated to make it easier to differentiate performance between various methods; the


Figure 15: Mean return under the ground-truth reward function achieved by PBRR with the default reference
policy used for all other experiments (PBRR ( _π_ ref _−_ default)), compared to (i) PBRR using a randomly initialized
reference policy (PBRR ( _π_ ref _−_ random-init)), and (ii) the Online-RLHF method. For easier visual comparison,
we truncate the plot for the Pandemic Mitigation environment in Figure 14.


containing state. In contrast, the ground-truth reward function assigns positive reward only when
a tomato is watered—i.e., upon its first visit. Consequently, the learned reward function induces
a looping policy that remains in a single tomato-containing state, whereas the ground-truth reward function induces a policy that visits all tomato-containing states. This failure mode illustrates how RLHF methods can conflate instrumental goals (e.g., visiting a tomato-containing state)
with terminal goals (e.g., visiting all tomato-containing states) even after learning from a large
dataset of preferences; see Marklund et al. (2025) for further characterization of this misalignment
type. We suspect that the Online-RLHF baseline learns similarly misaligned reward functions
in the other environments, while empirically PBRR does not exhibit this type of misalignment.


H.2 WHY DOES RRM PERFORM POORLY?


The results in Figure 2 show that the RRM performs poorly in this
environment after collecting a substantial number of preferences.
This result appears particularly surprising given that Cao et al.


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


(2025) follow the same methodology and achieve strong results
on a suite of robotics tasks. Upon observing the policy that RRM
learns after the first iteration, _πr_ _[∗]_ ˆ0 [, we note that it only aims to visit]
the sprinkler state to attain a high reward under the proxy reward
function. Trajectories are then sampled from _πr_ _[∗]_ ˆ0 [and] [used] [to]
update the proxy reward function, inducing policy _πr_ _[∗]_ ˆ1 [at the next]
iteration. Trajectories sampled from _πr_ _[∗]_ ˆ0 [that visit more tomatoes—]
specifically the ones in coordinates (5 _,_ 4) _,_ (5 _,_ 5) _,_ (5 _,_ 6) that are
on the way to the sprinkler—are always preferred to trajectories
that visit less tomatoes. Therefore, the proxy reward function
is updated to assign a higher reward for visiting those states.
Consequently, the policy derived from the updated proxy reward
function does not explore the other states—such as the tomatocontaining states in the top half of the grid-world—-because
the states including and around (5 _,_ 4) _,_ (5 _,_ 5) _,_ (5 _,_ 6) are predicted
to have higher reward. This process repeats, where the proxy
reward function is only ever updated for a particular region of
the grid-world—incorrectly assigned high reward—and therefore
performance does not improve. More broadly, exploiting the
proxy reward function does not necessarily induce an effective
exploration policy—an observation also noted by Xie et al. (2024). We suspect Cao et al. (2025)
achieve strong performance in their robotics tasks as a result of learning from a proxy reward function
that induces a policy that makes meaningful progress toward the true objective. This is notably not
the case in the settings we consider, although the proxy reward function can still be updated with
relatively few preferences to induce near-optimal performance via PBRR.


H.3 WHY DOES PBRR WITH THE OBJECTIVE IN EQ. 1 PERFORM POORLY?


The results in Figure 3 show that, in this environment, PBRR with the preference-learning objective
in Eq. 1 substantially under-performs PBRR’s default implementation, which uses the preferencelearning objective in Eq. 3. To understand why, we note that _π_ ref by construction only visits the
tomato-containing states at coordinates (1 _,_ 1) _,_ (3 _,_ 1) _,_ (3 _,_ 2). Throughout training, PBRR observes
preferences over trajectories sampled from _π_ ref and _πr_ _[∗]_ ˆ _t_ [.] [It is usually the case—inevitably during]
the early iterations—that trajectories from _π_ ref are preferred over trajectories from _πr_ _[∗]_ ˆ _t_ [.] [Without the]
regularization terms of Eq. 3, minimizing the cross-entropy loss from Eq. 1 over the collected dataset
of preferences then results in high predicted reward being assigned to the transitions encountered by
_π_ ref. As a result, _πr_ _[∗]_ ˆ _t_ [begins to visit those transitions at subsequent iterations, but does not explore]
other parts of the state space such as the tomato-containing states on the right hand side of the
grid-world. In effect, minimizing the standard preference loss induces unbounded updates to the
proxy reward function, resulting in a policy that over-values the actions taken by _π_ ref and does not
explore other, potentially optimal actions. The regularization terms we add for Eq. 3 encourage the
updated proxy reward function to remain sufficiently optimistic by encouraging only decrements
to the predicted reward when the proxy reward function induces a ranking that doesn’t match the
elicited preference.


I ILLUSTRATIVE SCENARIOS: MINIMAL MDPS


In this section we present simple MDPs that highlight PBRR’s strengths and weakness respectively.


We first show that there exists MDPs where PBRR can learn an optimal policy much faster than
random sampling:


**Theorem I.1.** _There exists an MDP in which PBRR recovers an optimal policy after O_ (1) _preference_
_query, whereas a uniform-exploration baseline requires O_ ( _|S|_ ) _preferences in expectation._


_Proof._ **MDP and rewards.** Consider a horizon- _H_ = 1 MDP with start state _s_ 0. From _s_ 0 there are _n_
actions _a_ 1 _, . . ., an_ leading deterministically to terminal states _s_ 1 _, . . ., sn_, respectively. Rewards are
tabular over next-states: taking _ai_ yields immediate reward _r_ ( _si_ ).


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


(a) MDP 1 (b) MDP 2


Figure 17: Example MDPs highlighting the strengths and weaknesses of PBRR, as outlined in Appendix I.1. For
both MDPs, assume deterministic transition dynamics and 4 available actions from state _s_ 0. _π_ ref marks the state
that is visited by the supplied reference policy. The proxy reward function’s outputted rewards are in red, and the
ground-truth reward function’s outputted rewards are in green. Rewards are only defined over states and the time
horizon _H_ = 1.


Let the initial _proxy_ reward function satisfy


_r_ ˆ0( _s_ 1) _>_ _r_ ˆ0( _sn_ ) _>_ _r_ ˆ0( _si_ ) for all _i_ _∈{/_ 1 _, n}._


Let the _ground-truth_ reward function satisfy


_r_ ( _sn_ ) _>_ _r_ ( _s_ 2) _>_ _r_ ( _si_ ) for all _i_ _∈{/_ 2 _, n},_


so _an_ is uniquely optimal. Assume a reference policy _π_ ref is supplied that always chooses action _a_ 2.
Note that the reference action _a_ 2 is strictly better than _a_ 1 under _r_ . Assume noiseless preferences
labeled by _r_ over single-step transitions: ( _s_ 0 _, ai, si_ ) _≻_ ( _s_ 0 _, ak, sk_ ) iff _r_ ( _si_ ) _> r_ ( _sk_ ).

**PBRR needs one preference.** First, _πr_ _[∗]_ ˆ0 [is constructed from the initialy proxy reward function][ ˆ] _[r]_ [0][.]
_πr_ _[∗]_ ˆ0 [chooses] _[ a]_ [1] [(since][ ˆ] _[r]_ [0][(] _[s]_ [1][)][ is maximal), while the supplied reference] _[ π]_ [ref] [deterministically chooses]
_a_ 2. PBRR constructs the pair
( _s_ 0 _, a_ 1 _, s_ 1) vs. ( _s_ 0 _, a_ 2 _, s_ 2) _,_
which is labeled ( _s_ 0 _, a_ 2 _, s_ 2) _≻_ ( _s_ 0 _, a_ 1 _, s_ 1) because _r_ ( _s_ 2) _>_ _r_ ( _s_ 1) by construction. In accordance
with Eq. 3, the proxy reward function is updated by decrementing ˆ _r_ ( _s_ 1) below ˆ _r_ ( _s_ 2) while leaving
all uncompared transitions unchanged. This update produces ˆ _r_ 1 with


_r_ ˆ1( _s_ 2) _>_ _r_ ˆ1( _s_ 1) and _r_ ˆ1( _si_ ) = _r_ ˆ0( _si_ ) for all _i_ _∈{/_ 1 _,_ 2 _}._


Since initially ˆ _r_ 0( _sn_ ) _>_ _r_ ˆ0( _si_ ) for all _i_ _∈{/_ 1 _, n}_, and the update leaves ˆ _r_ ( _sn_ ) and ˆ _r_ ( _s_ 2) unchanged,
we still have
_r_ ˆ1( _sn_ ) _>_ _r_ ˆ1( _si_ ) for all _i ̸_ = _n._
Hence _πr_ _[∗]_ ˆ1 [selects] _[ a][n]_ [, which is optimal under] _[ r]_ [.] [Therefore PBRR reaches an] _[ r]_ [-optimal policy after a]
single preference, i.e., _O_ (1).


**Uniform exploration needs** _O_ ( _|S|_ ) **preferences.** Consider a baseline that, at each iteration, selects
two actions _ai_ and _ak_ uniformly from _{a_ 1 _, . . ., an}_ and elicits a preference comparing ( _s_ 0 _, ai, si_ )
against ( _s_ 0 _, ak, sk_ ). Assuming the same update rule as PBRR above, the only comparison that can
demote the action chosen by the policy induced by the proxy reward function _a_ 1 below _ai_ for _i ̸_ = 1
(and thereby expose _an_ as the new argmax, given the update) is the pair involving _i_ = 1. Each
iteration hits _i_ = 1 with probability 2 _/n_, so the waiting time to the first such informative comparison
is geometric with mean _n/_ 2. Thus the baseline requires E[#preferences] = _n/_ 2 = _O_ ( _n_ ) = _O_ ( _|S|_ )
(since _|S|_ = _n_ + 1) before it can recover the optimal action _an_ .


MDP 1 shows a specific illustration of one such MDP that satisfied Theorem I.1. We now step through
the procedure executed by PBRR following Algorithm 1 in MDP 1. First the proxy reward function
initially induces a policy that visits _s_ 1. Upon observing a preference between this trajectory and the
trajectory sampled from _π_ ref— _s_ 3 _≻_ _s_ 1— the proxy reward function is updated so that the proxy
reward for _s_ 1 is lower than the proxy reward for _s_ 3. Therefore, the updated proxy reward function at
the next iteration induces a policy that visits _s_ 4, which is optimal with respect to the ground-truth


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


reward function. MDP 1 illustrates a scenario where the proxy reward function induces a substantially
sub-optimal policy, but updating the proxy reward function at only a single state can repair it so as to
induce an optimal policy under the ground-truth reward function. PBRR succeeds in such scenarios.


We next present a scenario, shown in MDP 2 (Figure 17, right) where PBRR with _C_ = 0 (no explicit
additional exploration) can fail to learn the optimal policy. Here after the same initial step as in MDP,
the proxy reward function is updated so that the induced policy visits _s_ 4. The preference collected at
this iteration will be _s_ 4 _≻_ _s_ 3. Because the objective in Eq. 3 discourages updating the proxy reward
function when it induces a ranking that matches the elicited preference, the proxy reward function is
not updated further. But the proxy reward function does not induce an optimal policy with respect
to the ground-truth reward function. MDP 2 highlights an example where PBRR with _C_ = 0 (no
explicit additional exploration) will induce a policy that outperforms the reference policy but would
not induce optimal performance.


I.1 THE BENEFITS OF PBRR’S LEARNING OBJECTIVE


Here we present another minimal MDP to illustrate the purpose of the regularization terms we propose
in Eq. 3.


**The benefit of** _L_ [+] To illustrate why our objective includes the _L_ [+] regularization term, take the
MDP in Figure 18. Assume that we observe the preference _s_ 3 _≻_ _s_ 4, and then update the proxy reward
function with the standard cross entropy loss in Eq. 1 rather than our proposed objective. Minimizing
this loss will push ˆ _r_ ( _s_ 3) _→∞_ and ˆ _r_ ( _s_ 4) _→−∞_ as the loss will continue to decrease as ˆ _r_ ( _s_ 3) _−r_ ˆ( _s_ 4)
increases. This may result in an update to the proxy reward function where _r_ ˆ( _s_ 3) _>_ _r_ ˆ( _s_ 1), which
would result in a policy that goes to state _s_ 3 instead of _s_ 1. Therefore, even though the proxy reward
function correctly ranked states ( _s_ 3 _, s_ 4) and produced an optimal policy under _r_ before being updated,
minimizing the loss in Eq. 1 given the preference _s_ 3 _≻_ _s_ 4 can update the proxy reward function such
that it no longer induces an optimal policy under _r_ . To avoid this undesirable scenario, we add the
regularization term _L_ [+] to discourage updates to the proxy reward function when it induces a correct
ranking over trajectories in a pair.


**The benefit of** _L_ _[−]_ Assume that we observe the preference _s_ 2 _≻_ _s_ 3, and then update the proxy
reward function with the standard cross entropy loss in Eq. 1 rather than our proposed objective.
Minimizing this loss will push ˆ _r_ ( _s_ 2) _→∞_ and ˆ _r_ ( _s_ 3) _→−∞_ as the loss will continue to decrease
as _r_ ˆ( _s_ 2) _−_ _r_ ˆ( _s_ 3) increases. This may result in an update to the proxy reward function where
_r_ ˆ( _s_ 2) _>_ _r_ ˆ( _s_ 1), which would result in a policy that goes to state _s_ 2 instead of _s_ 1. Therefore, even
though the proxy reward function is updated to produce a correct ranking over the pair ( _s_ 2 _, s_ 3), it no
longer induces an optimal policy under _r_ . To avoid this scenario, we add the regularization term _L_ _[−]_
to encourage only decrementing the proxy reward function’s output (e.g., decreasing ˆ _r_ ( _s_ 3)) rather
than also increasing its output (e.g., increasing ˆ _r_ ( _s_ 2)).


(a) MDP 3


Figure 18: Example MDPs highlighting the benefits of the regluarization terms proposed in Eq. 3, as outlined
in Appendix I.1. Assume deterministic transition dynamics and 4 available actions from state _s_ 0. The proxy
reward function’s outputted rewards are in red, and the ground-truth reward function’s outputted rewards are in
green. Rewards are only defined over states and the time horizon _H_ = 1.


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


J REGRET BOUND PROOFS


We now define some additional notation, following Pacchiano et al. (2023). Note when the dynamics
model is known, one can compute the expected features _ϕ_ under a policy _πi_, which we denote as
_ϕ_ ( _πi_ )


_Proof._ Our proof closely follows the proof of Theorem 1 in Pacchiano et al. (2023). While their proof
was for a different algorithm, we note that their Lemma 7, Corollary 1, Lemma 2 and Lemma 8 all
continue to hold in our setting, as they do not depend on the specific policies chosen for exploration.


The first part of the proof of our theorem exactly follows the proof of Theorem 1Pacchiano et al.
(2023), where conditioned on the event _Eδ_ holding, they bound the regret due to executing the two


32


_Vt_ =


_gt_ ( _w_ ) =


_t−_ 1
�( _ϕ_ ( _τl_ [1][)] _[ −]_ _[ϕ]_ [(] _[τ]_ [ 2] _l_ [))(] _[ϕ]_ [(] _[τ]_ [ 1] _l_ [)] _[ −]_ _[ϕ]_ [(] _[τ]_ [ 2] _l_ [))] _[T]_ [+] _[ κλI][d]_ (7)

_l_ =1


_t−_ 1

- _σ_ ( _< ϕ_ ( _τl_ [1][)] _[ −]_ _[ϕ]_ [(] _[τ]_ [ 2] _l_ [)] _[, w]_ _[>]_ [)(] _[ϕ]_ [(] _[τ]_ [ 1] _l_ [)] _[ −]_ _[ϕ]_ [(] _[τ]_ [ 2] _l_ [)) +] _[ λW]_ (8)

_l_ =1


_wt_ _[L]_ = arg _ws.t._ min _||w||≤W_ _[||][g][t]_ [(] _[w]_ [)] _[ −]_ _[g][t]_ [( ˆ] _[w]_ _t_ _[MLE]_ ) _||Vt−_ 1 (9)

_fmk_ ( _π_ 1 _, π_ 2) = _||ϕ_ ( _π_ 1) _−_ _ϕ_ ( _π_ 2) _||Vt−_ 1 (10)


_αd,T_ ( _δ_ ) = 20 _BW_


  - _T_ (1 + 2 _T_ )
_d_ log
_δ_


(11)


     log(1 _/δ_ ) + 2 _d_ log 1 + _[tB]_

_κλd_


_√_
_βt_ ( _δ_ ) =


(12)


_λW_ +


_Ct_ ( _δ_ ) = _w_ _s.t. ||w −_ _wt_ _[L][||][V]_ _t_ _[≤]_ [2] _[κβ][t]_ [(] _[δ]_ [)] (13)
_γt_ ( _δ_ ) = 2 _κβt_ ( _δ_ ) + _αd,T_ ( _δ_ ) (14)


        -         Π _t_ = _πi|_ ( _ϕ_ ( _πi_ ) _−_ _ϕ_ ( _π_ )) _[T]_ _wt_ + _γt_ ( _δ_ ) _||ϕ_ ( _π_ _[i]_ ) _−_ _ϕ_ ( _π_ ) _||Vt−_ 1 _≥_ 0 _∀_ _π_ (15)

(16)


When cross entropy loss is used to fit preference data and the reward model is linear, the resulting
loss can be expressed as


2 [+(1] _[−]_ _[o][l]_ [) log(] _[σ]_ [(1] _[−⟨][θ]_ [(] _[τ]_ [ 1] _l_ [)] _[−]_ _[θ]_ [(] _[τ]_ [ 2] _l_ [))] _[, w][⟩]_ [))] _[,]_
2 _[||][w][||]_ [2]


_L_ _[λ]_ _t_ [(] _[w]_ [) =]


_t_


�( _ol_ log( _σ_ ( _⟨θ_ ( _τl_ [1][)] _[−]_ _[θ]_ [(] _[τ]_ [ 2] _l_ [))] _[, w][⟩]_ [))] _[−]_ _[λ]_

2

_l_ =1


(17)
where _ol_ = 1 or 0 depending on which trajectory is preferred.


As the maximum likelihood estimator _wMLE_ may not satisfy the required boundness assumption,
prior work Pacchiano et al. (2023) defined a projected version _wt_ _[L]_ [of the weight vector] _[ w]_ [.]

We also define the event that the true _w_ _[∗]_ iles in the specified confidence interval _Ct_ ( _δ_ ) on all time
steps as
_Eδ_ = _{∀t ≥_ 1 _,_ **w** _⋆_ _∈Ct_ ( _δ_ ) _}._ (18)


While in the main text we show the big-O version to avoid the additional notation complexity required
to define the terms, we now state a more precise version of Theorem 5.1:


**Theorem** **J.1.** _Let_ _δ_ _≤_ 1 _/_ _and_ _λ_ _≥_ _B/κ._ _Then_ _under_ _Assumptions_ _1_ _and_ _2,_ _for_ _f_ = _fmk_ _and_
Π _t_ = Π _t,mk, with probability at least_ 1 _−_ _δ, the expected regret of Algorithm 1 is bounded by_


        
_Regrett_ _≤_ _C_ 1(4 _κβt_ ( _δ_ ) + 2 _αd,T_ ( _δ_ ))


   2 _Td_ log 1 + _[TB]_

_κd_


(19)


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


exploration policies _π_ 1 and _π_ 2:

2 _rt_ = ( _ϕ_ ( _π_ _[∗]_ ) _−_ _ϕ_ ( _πt_ [1][))] _[⊤]_ **[w]** _[∗]_ [+ (] _[ϕ]_ [(] _[π][∗]_ [)] _[ −]_ _[ϕ]_ [(] _[π]_ _t_ [2][))] _[⊤]_ **[w]** _[∗]_

= ( _ϕ_ ( _π_ _[∗]_ ) _−_ _ϕ_ ( _πt_ [1][))] _[⊤]_ **[w]** _t_ _[L]_ [+ (] _[ϕ]_ [(] _[π][∗]_ [)] _[ −]_ _[ϕ]_ [(] _[π]_ _t_ [1][))] _[⊤]_ [(] **[w]** _[∗]_ _[−]_ **[w]** _t_ _[L]_ [) + (] _[ϕ]_ [(] _[π][∗]_ [)] _[ −]_ _[ϕ]_ [(] _[π]_ _t_ [2][))] _[⊤]_ [(] **[w]** _[∗]_ _[−]_ **[w]** _t_ _[L]_ [) + (] _[ϕ]_ [(] _[π][∗]_ [)] _[ −]_ _[ϕ]_ [(] _[π]_ _t_ [2][))] _[⊤]_ **[w]** _t_ _[L]_
_≤_ ( _ϕ_ ( _π_ _[∗]_ ) _−_ _ϕ_ ( _πt_ [1][))] _[⊤]_ **[w]** _t_ _[L]_ [+ (] _[ϕ]_ [(] _[π][∗]_ [)] _[ −]_ _[ϕ]_ [(] _[π]_ _t_ [2][))] _[⊤]_ **[w]** _t_ _[L]_
+ _∥_ **w** _[∗]_ _−_ **w** _t_ _[L][∥][V]_ _t_ _[· ∥][ϕ]_ [(] _[π][∗]_ [)] _[ −]_ _[ϕ]_ [(] _[π]_ _t_ [1][)] _[∥]_ _Vt_ _[−]_ [1] + _∥_ **w** _[∗]_ _−_ **w** _t_ _[L][∥][V]_ _t_ _[· ∥][ϕ]_ [(] _[π][∗]_ [)] _[ −]_ _[ϕ]_ [(] _[π]_ _t_ [2][)] _[∥]_ _Vt_ _[−]_ [1] (20)


They then note that in this sum, the last two terms can be bounded by their Corollary 1:

_∥_ **w** _[∗]_ _−_ **w** _t_ _[L][∥][V]_ _t_ _[· ∥][ϕ]_ [(] _[π][∗]_ [)] _[ −]_ _[ϕ]_ [(] _[π]_ _t_ [1][)] _[∥]_ _Vt_ _[−]_ [1] + _∥_ **w** _[∗]_ _−_ **w** _t_ _[L][∥][V]_ _t_ _[· ∥][ϕ]_ [(] _[π][∗]_ [)] _[ −]_ _[ϕ]_ [(] _[π]_ _t_ [2][)] _[∥]_ _Vt_ _[−]_ [1] (21)


using Cauchy-Schwarz for the second inequality and Lemma 8 Pacchiano et al. (2023) for the final
inequality.


We now provide an analogous proof for the case when the dynamics model is not known. We need
some additional notation. Let _Nt_ ( _s, a_ ) represent the number of times the trajectories have included
( _s, a_ ) tuples. The proof again follows prior work Pacchiano et al. (2023). They define an alternate
covariance matrix that leverages the empirical covariance matrix, and an alternate bonus term and
confidence sets needed to account for the uncertainty since the dynamics model is estimated from
finite samples.


33


         -         _≤_ (2 _κβt_ ( _δ_ ) + _αT,d_ ( _δ_ )) _·_ _∥ϕ_ ( _π_ _[∗]_ ) _−_ _ϕ_ ( _πt_ [1][)] _[∥]_ _Vt_ _[−]_ [1] + _∥ϕ_ ( _π_ _[∗]_ ) _−_ _ϕ_ ( _πt_ [2][)] _[∥]_ _Vt_ _[−]_ [1] (22)


Up to this point, the proof is identical. We now seek to bound the first two terms in Equation 20. First
note that


                       ( _ϕ_ ( _π_ _[∗]_ ) _−ϕ_ ( _πt_ [1][))] _[⊤]_ **[w]** _t_ _[L]_ [+(] _[ϕ]_ [(] _[π][∗]_ [)] _[−][ϕ]_ [(] _[π]_ _t_ [2][))] _[⊤]_ **[w]** _t_ _[L]_ _[≤]_ [(2] _[κβ][t]_ [(] _[δ]_ [)+] _[α][T,d]_ [(] _[δ]_ [))] _||ϕ_ ( _π_ _[∗]_ ) _−_ _ϕ_ ( _πt_ [1][)] _[||]_ _V_ [ ¯] _t_ _[−]_ [1] + _||ϕ_ ( _π_ _[∗]_ ) _−_ _ϕ_ ( _πt_ [2][)] _[||]_ _V_ [ ¯] _t_ _[−]_ [1]

(23)
since Line 9 in Algorithm 1 ensures that the selected _π_ 1 and _π_ 2 always lie in Π _t_, and therefore holds
from the definition of Π _t_ .

In addition, Line 9 in Algorithm 1 ensures if _πr_ _[∗]_ ˆ _t_ [=] _[π]_ [1] [and] _[ π]_ [ref] [=] _[π]_ [2] [are used as the exploration]
policies, then they must satisfy:


max _r_ ˆ _t_ _[, π]_ [ref] [)] _[.]_ (24)
_πi,πj_ _∈_ Π _t_ _[f]_ [(] _[π][i][, π][j]_ [)] _[ ≤]_ _[C]_ [1] _[f]_ [(] _[π][∗]_


Therefore

_∥ϕ_ ( _π_ _[⋆]_ ) _−_ _ϕ_ ( _π_ ref) _∥_ _[−]_ _V_ ¯ _t_ [1] [+] _[ ∥][ϕ]_ [(] _[π][⋆]_ [)] _[ −]_ _[ϕ]_ [(] _[π]_ proxy [)] _[∥][−]_ _V_ ¯ _t_ [1] _≤_ 2 _C_ 1 _· ∥ϕ_ ( _π_ 1) _−_ _ϕ_ ( _π_ 2) _∥V_ _[−]_ ¯ _t_ [1] _[.]_ (25)


The remaining part of the proof follows Theorem 1 Pacchiano et al. (2023). Specifically substituting
Equations 22, 23 and 25 into Equation 20, and using that _π_ _[∗]_ _∈_ Π _t_ under the assumed event and
Lemma 2, yields


2 _rt_ _≤_ 2�2 _κβt_ ( _δ_ ) + _αT,d_ ( _δ_ )� _·_ - _∥ϕ_ ( _π_ _[∗]_ ) _−_ _ϕ_ ( _πt_ [1][)] _[∥]_ _Vt_ _[−]_ [1] + _∥ϕ_ ( _π_ _[∗]_ ) _−_ _ϕ_ ( _πt_ [2][)] _[∥]_ _Vt_ _[−]_ [1] - (26)


_≤_ 4 _C_ 1�2 _κβt_ ( _δ_ ) + _αT,d_ ( _δ_ )� _· ∥ϕ_ ( _πt_ [1][)] _[ −]_ _[ϕ]_ [(] _[π]_ _t_ [2][)] _[∥]_ _Vt_ _[−]_ [1] (27)


The remaining few steps in the proof of Theorem 1 then bound


_RegretT_ =      - _rt_ (28)


_t_


_≤_ - 4 _C_ 1�2 _κβt_ ( _δ_ ) + _αT,d_ ( _δ_ )� _·_


_t_


_≤_ 


_T_

- _∥ϕ_ ( _πt_ [1][)] _[ −]_ _[ϕ]_ [(] _[π]_ _t_ [2][)] _[∥]_ _Vt_ _[−]_ [1] (29)

_t_ =1


_T_

- _∥|ϕ_ ( _πt_ [1][)] _[ −]_ _[ϕ]_ [(] _[π]_ _t_ [2][)] _[∥|]_ [2] _Vt_ _[−]_ [1] (30)

_t_ =1


_≤_ 4 _C_ 1�2 _κβt_ ( _δ_ ) + _αT,d_ ( _δ_ )� _·_


_≤_ 4 _C_ 1�2 _κβt_ ( _δ_ ) + _αT,d_ ( _δ_ )� _·_


~~�~~


- _T_


2 _Td_ log(1 + _[TB]_ (31)

_d_ [)] _[,]_


**1782**


**1783**

**1784**

**1785**

**1786**

**1787**


**1788**

**1789**

**1790**

**1791**

**1792**

**1793**


**1794**

**1795**

**1796**

**1797**

**1798**

**1799**


**1800**

**1801**

**1802**

**1803**

**1804**

**1805**


**1806**

**1807**

**1808**

**1809**

**1810**


**1811**

**1812**

**1813**

**1814**

**1815**

**1816**


**1817**

**1818**

**1819**

**1820**

**1821**

**1822**


**1823**

**1824**

**1825**

**1826**

**1827**

**1828**


**1829**

**1830**

**1831**

**1832**

**1833**


**1834**

**1835**


(41)
The remainder of the proof of follows the rest of the proof of Lemma 15 Pacchiano et al. (2023).


K REWARD HACKING ANALYSIS PROOFS


Here we provide a theoretical analysis motivating PBRR, relying on different assumptions than the
regret analysis in Section 5.


We show that PBRR’s repaired reward function is guaranteed to induce an optimal policy that matches
or exceeds the performance of the reference policy (Thm. K.1) and resolve two specific instantiations
of reward hacking (Cors. K.2, K.3) in the infinite data setting as the number of iterations goes to
infinity.
**Assumption K.1.** _The regularization weights vanish λ_ 1 _, λ_ 2 _→_ 0 _as t →∞_ _in Algorithm 1._

**Assumption K.2.** _Preferences over trajectories are noiseless and determined by the difference in_
_regret between trajectories (Eq._ _42)._


34


**V** ˜ _t_ = _κλId_ +


_t−_ 1


_ℓ_ =1


- _ϕ_ Pˆ _ℓ_ ( _πℓ_ 1 [)] _[ −]_ _[ϕ]_ Pˆ _ℓ_ ( _πℓ_ 2 [)] �� _ϕ_ Pˆ _ℓ_ ( _πℓ_ 1 [)] _[ −]_ _[ϕ]_ Pˆ _ℓ_ ( _πℓ_ 2 [)] - _⊤_ (32)


_U_
_Nt_ ( _s, a_ )


_,_ (33)


_ξs,a_ [(] _[t]_ [)] [(] _[η, δ]_ [) = min]


2 _η,_ 4 _η_


where


The bonus function is:


_U_ = _H_ log - _|S||A|_ - + log� 6 log( _Nt_ ( _s, a_ ))
_δ_


_._ (34)


  


(35)


_B_ ˆ _t_ ( _π, η, δ_ ) = E _s_ 1 _∼ρ, τ_ _∼_ Pˆ _πt_ [(] _[·|][s]_ [1][)]


and the undominated policy set is:


- _H−_ 1

 


- _ξs_ [(] _[t]_ _h_ [)] _,ah_ [(] _[η, δ]_ [)]

_h_ =1


Π _t,µ_ =


and we define _f_ as:


_π_ _[i]_ - _ϕ_ [P][ˆ] _[t]_ ( _π_ _[i]_ ) _−_ _ϕ_ [P][ˆ] _[t]_ ( _π_ )� _⊤_ **w** _tL_ [+] _[ γ][t]_ �� _ϕ_ Pˆ _t_ ( _πi_ ) _−_ _ϕ_ Pˆ _t_ ( _π_ )�� **V** ˜ _t−_ 1 (36)
�����


+ _B_ [�] _t_ - _π_ _[i]_ _,_ 2 _SB,_ 2 _|A||S|δ_ - + _B_ [�] _t_ - _π,_ 2 _SB,_ 2 _|A||S|δ_ - _≥_ 0 _,_ _∀π_


_._ (37)


_fu_ ( _πt_ [1] _[, π]_ _t_ [2][) =] _[ γ][t]_ �� _ϕ_ P _t_ ˆ _t_ [(] _[π]_ [1][)] _[ −]_ _[ϕ]_ P _t_ ˆ _t_ [(] _[π]_ [2][)] �� _∥_ **V** ˜ _t−_ 1 + 2 � _Bt_ ( _π_ 1 _,_ 2 _WB, δ_ ) + 2 � _Bt_ ( _π_ 2 _,_ 2 _WB, δ_ ) _._ (38)


We now restate our Theorem 5.2:

**Theorem J.2.** _(Theorem 5.2) Under Assumptions 5.1,5.3 and 5.4, for f_ = _fu_ _and_ Π _t_ = Π _t,u, the_
_regret of Algorithm 1 is bounded by_


   -    - _√_
_RT_ _≤_ _O_ [˜] _C_ 1 _κd_


_T_ + _H_ [3] _[/]_ [2][�]


_|S||A|dTH_ + _H|S|_ ~~�~~ _|A|dTH_ �� _,_ (39)


_Proof._ (sketch) The proof follows the proof of Lemma 15 Pacchiano et al. (2023) with the analogous
modification as the one we made in our proof of Theorem 1. Specifically, note that
��� _ϕ_ Pˆ _t_ ( _π∗_ ) _−_ _ϕ_ Pˆ _t_ ( _πt_ 1 [)] ��� _V_ ˜ _t−_ 1 + ��� _ϕ_ Pˆ _t_ ( _π∗_ ) _−_ _ϕ_ Pˆ _t_ ( _πt_ 2 [)] ��� _V_ ˜ _t−_ 1 _≤_ _πi,π_ max _j_ _∈_ Π _t,u_ ��� _ϕ_ Pˆ _t_ ( _πi_ ) _−_ _ϕ_ Pˆ _t_ ( _πj_ )��� _V_ ˜ _t−_ 1

(40)
and from Line 9, the definition of _fu_ ensures:


���� _ϕ_ Pˆ _t_ ( _π∗_ ) _−_ _ϕ_ Pˆ _t_ ( _πt_ 1 [)] ��� _V_ ˜ _t−_ 1 + ��� _ϕ_ Pˆ _t_ ( _π∗_ ) _−_ _ϕ_ Pˆ _t_ ( _πt_ 2 [)] ��� _V_ ˜ _t−_ 1


max
_πi,πj_ _∈_ Π _t,u_


��� _ϕ_ Pˆ _t_ ( _πi_ ) _−_ _ϕ_ Pˆ _t_ ( _πj_ )��� _V_ ˜ _t−_ 1 _≤_ _C_ 1


**1836**


**1837**

**1838**

**1839**

**1840**

**1841**


**1842**

**1843**

**1844**

**1845**

**1846**

**1847**


**1848**

**1849**

**1850**

**1851**

**1852**

**1853**


**1854**

**1855**

**1856**

**1857**

**1858**

**1859**


**1860**

**1861**

**1862**

**1863**

**1864**


**1865**

**1866**

**1867**

**1868**

**1869**

**1870**


**1871**

**1872**

**1873**

**1874**

**1875**

**1876**


**1877**

**1878**

**1879**

**1880**

**1881**

**1882**


**1883**

**1884**

**1885**

**1886**

**1887**


**1888**

**1889**


We adopt Assumption K.2, following Knox et al. (2022), who show that regret better reflects human
preferences—see Appendix K.1 for details.
**Assumption K.3.** _C_ 1 = 0 _in Algorithm 1._
**Assumption K.4.** _Each trajectory set contains the (potentially infinite) support of the corresponding_
_policy, i.e.,_ support( _π_ ) _⊆Tπ._
**Assumption K.5.** _All trajectories begin in the same start state s_ 0 _._
**Theorem K.1.** _Suppose assumptions K.2 through K.5 hold, then the optimal policy for the repaired_
_reward function πr_ _[∗]_ ˆ _[matches or outperforms the reference policy in the limit as][ t][ →∞][:]_
_Jr_ ( _πr_ _[∗]_ ˆ [)] _[ ≥]_ _[J][r]_ [(] _[π][ref]_ [)] _[.]_


Theorem K.1 implies that reward hacking is resolved as _t →∞_, following specific instantiations of
the two different definitions of reward hacking from Skalse et al. (2022) and Laidlaw et al. (2025):
**Corollary K.2.** _(No Skalse et al. (2022) Hacking) The reward function_ ˆ _r and the ground-truth reward_
_function r are unhackable relative to the optimal policy set for_ ˆ _r and the reference policy πref._
**Corollary K.3.** _(No Laidlaw et al. (2025) Hacking) The reward function_ ˆ _r is unhackable with respect_
_to πref._


K.1 ASSUMPTIONS


For this theoretical analysis we assume Algorithm 1 is executed with _C_ 1 = 0, like we execute in
practice.


We define regret under _r_ ˜ for a trajectory _τ_ = ( _s_ 0 _, a_ 0 _, . . ., sH_ ) as regret( _τ_ _|_ _r_ ˜) ≜

- _|tτ_ =0 _|−_ 1 _γ_ _[t]_ [ _Vr_ ˜ _[∗]_ [(] _[s][t]_ [)] _[ −]_ _[Q]_ _r_ _[∗]_ ˜ [(] _[s][t][, a][t]_ [)]] [=] [�] _t_ _[|][τ]_ =0 _[|−]_ [1][(] _[−][γ][t][A]_ _r_ _[∗]_ ˜ [(] _[s][t][, a][t]_ [))][ where] _[ A][∗]_ _r_ ˜ [(] _[s, a]_ [)][ ≜] _[Q][∗]_ _r_ ˜ [(] _[s, a]_ [)] _[ −]_ _[V]_ _r_ ˜ _[∗]_ [(] _[s]_ [)]
and _Vr_ ˜ _[∗][, Q]_ _r_ _[∗]_ ˜ [are the optimal value and action-value functions under][ ˜] _[r]_ [.] [We then assume a preference]
between _τ_ 1 and _τ_ 2 is determined by regret:


We also assume _L_ pref uses regret-based preferences.


Knox et al. (2022) shows that human preferences are better described by the difference in regret
between trajectory segments, as opposed to the difference in the sum of rewards. In practice, we
simulate human preference labels using the difference in sum of rewards between trajectories because
regret is intractable to compute in our empirical environments, as discussed in Appendix A.


Our assumption about how preferences are labeled (Eq. 42) for this analysis differs in two key ways
from Knox et al. (2022): (i) we assume preferences are over trajectories, not trajectory segments
(ii) we assume preferences are noiselessly determined by regret, not sampled from a Boltzmann
distribution. Our analysis holds without assumption (i) as long as preferences are elicited over an
exhaustive set of trajectory segments. We argue that (ii) is a reasonable assumption when executing
PBRR, where preferences are only collected between trajectories sampled from _πr_ ˆ _t_ —which is initially
substantially sub-optimal due to a misspecified proxy reward function ˆ _rt_, and _π_ ref—which is assumed
to be a safe policy. There is clear distinction between the trajectories from these policies—both
qualitatively and in terms of regret under _r_, so we assume no preference noise via Eq. 42.


We also assume that all trajectories begin from the same start state _s_ 0. This assumption is without loss
of generality: for any MDP with initial state distribution _p_ 0, we can construct an equivalent MDP by
introducing a new start state _s_ _[′]_ 0 [that transitions in a single step according to] _[ p]_ [0][.] [In this construction,]
all trajectories then originate from _s_ _[′]_ 0 [, satisfying the assumption of a single start state.]


35


_µ_ ≜






0 if regret( _τ_ 1 _|r_ ) _<_ regret( _τ_ 2 _|r_ ) _,_
1 if regret( _τ_ 1 _|r_ ) _>_ regret( _τ_ 2 _|r_ ) _,_ (42)
12 if regret( _τ_ 1 _|r_ ) = regret( _τ_ 2 _|r_ ) _,_





where regret is the sum of negative optimal advantage:


regret( _τ_ _|_ _r_ ˜) ≜


_|τ_ _|−_ 1

- _−A_ _[∗]_ _r_ ˜ [(] _[s]_ _t_ _[τ]_ _[, a]_ _t_ _[τ]_ [)] _[.]_

_t_ =0


**1890**

**1891**


**1892**

**1893**

**1894**

**1895**

**1896**

**1897**


**1898**

**1899**

**1900**

**1901**

**1902**

**1903**


**1904**

**1905**

**1906**

**1907**

**1908**


**1909**

**1910**

**1911**

**1912**

**1913**

**1914**


**1915**

**1916**

**1917**

**1918**

**1919**

**1920**


**1921**

**1922**

**1923**

**1924**

**1925**

**1926**


**1927**

**1928**

**1929**

**1930**

**1931**


**1932**

**1933**

**1934**

**1935**

**1936**

**1937**


**1938**

**1939**

**1940**

**1941**

**1942**

**1943**


K.2 PROOF OF THEOREM K.1


**Lemma K.4.** _Suppose Assumptions K.1 and K.2 hold._ _Then the repaired reward function_ ˆ _r and the_
_ground-truth reward function r induce identical regret-based orderings on all inter-policy trajectory_
_pairs_ ( _τ_ 1 _, τ_ 2) _∈Tπr_ _[∗]_ ˆ _[× T][π]_ [ref] _[:]_

sign [regret( _τ_ 1 _|_ _r_ ˆ) _−_ regret( _τ_ 2 _|_ _r_ ˆ)] = sign [regret( _τ_ 1 _| r_ ) _−_ regret( _τ_ 2 _| r_ )] _._


_Proof._ Let ( _τ_ 1 _, τ_ 2) _∈_ ( _τ_ 1 _, τ_ 2) _∈Tπr_ _[∗]_ ˆ _[× T][π]_ [ref] [denote an arbitrary inter-policy trajectory pair.] [Define]
the Bayes optimal decision rule for distinguishing preferences as


Now, under Assumption K.1, the PBRR loss (Eq. 3) reduces in the limit to the standard preference
cross-entropy loss (Eq. 1):


lim
_t→∞_ _[L]_ [(] _[g]_ [; ˆ] _[r]_ [proxy] _[,][ D]_ [) =] _[ L]_ [pref][(ˆ] _[r]_ [proxy][ +] _[ g]_ [;] _[ D]_ [)] _[.]_


Since cross-entropy is a convex margin loss known to be Bayes consistent (Zhang, 2004; Bartlett
et al., 2006), any minimizer ˆ _r_ of _L_ pref must induce a predictor _fL_ _[∗]_ pref [whose sign agrees with the Bayes]
optimal rule:
sign               - _fL_ _[∗]_ pref [(] _[τ]_ [1] _[, τ]_ [2][)]               - = _f_ 0 _[∗]_ _/_ 1 [(] _[τ]_ [1] _[, τ]_ [2][;] _[ r]_ [)] _[.]_


Substituting the regret-based form of _f_ 0 _[∗]_ _/_ 1 [, it follows that the ordering induced by][ ˆ] _[r]_ [ must coincide]
with that induced by _r_ . Equivalently,


sign [regret( _τ_ 1 _|_ _r_ ˆ) _−_ regret( _τ_ 2 _|_ _r_ ˆ)] = sign [regret( _τ_ 1 _| r_ ) _−_ regret( _τ_ 2 _| r_ )] _._


**Restatement** **of** **Theorem** **K.1.** Suppose Assumptions K.1 through K.5 hold, then the optimal
policy for the repaired reward function _πr_ _[∗]_ ˆ [matches or outperforms the reference policy in the limit as]
_t →∞_ :

_Jr_ ( _πr_ _[∗]_ ˆ [)] _[ ≥]_ _[J][r]_ [(] _[π]_ [ref][)] _[.]_


_Proof._ Pick any _τ_ 1 _∈Tπr_ _[∗]_ ˆ [and] _[ τ]_ [2] _[∈T]_ [ref][.] [By the optimality of] _[ π]_ _r_ _[∗]_ ˆ [under][ ˆ] _[r]_ [:]

regret( _τ_ 1 _|_ _r_ ˆ) = 0 _≤_ regret( _τ_ 2 _|_ _r_ ˆ) _._


By Lemma K.4, this inequality is preserved under _r_, so


regret( _τ_ 1 _| r_ ) _≤_ regret( _τ_ 2 _| r_ ) _._


Note that this implies
max _[|][ r]_ [)] _[≤]_ [min] _[|][ r]_ [)] _[.]_
_τ_ 1 [regret(] _[τ]_ [1] _τ_ 2 [regret(] _[τ]_ [2]

On the left hand side, the maximum is an upper bound to the expectation over _τ_ 1 _∈Tπr_ _[∗]_ ˆ [, and the]
minimum is a lower bound to the expectation over _τ_ 2 _∈Tπ_ ref :

E _τ_ 1 _∈Tπr∗_ ˆ�regret( _τ_ 1 _| r_ )� _≤_ max _τ_ 1 [regret(] _[τ]_ [1] _[|][ r]_ [)] _[≤]_ [min] _τ_ 2 [regret(] _[τ]_ [2] _[|][ r]_ [)] _[ ≤]_ [E] _[τ]_ [2] _[∈T][π]_ [ref] �regret( _τ_ 2 _| r_ )� _._


Therefore, by Assumption K.4,

E _τ_ 1 _∼πr_ _[∗]_ ˆ�regret( _τ_ 1 _| r_ )� _≤_ E _τ_ 2 _∼π_ ref�regret( _τ_ 2 _| r_ )� _._


36


_f_ 0 _[∗]_ _/_ 1 [(] _[τ]_ [1] _[, τ]_ [2][;] _[ r]_ [)][ ≜]






1 if _p_ (1 _| τ_ 1 _, τ_ 2 _, r_ ) _> p_ ( _−_ 1 _| τ_ 1 _, τ_ 2 _, r_ ) _,_
0 if _p_ (1 _| τ_ 1 _, τ_ 2 _, r_ ) = _p_ ( _−_ 1 _| τ_ 1 _, τ_ 2 _, r_ ) _,_

_−_ 1 if _p_ (1 _| τ_ 1 _, τ_ 2 _, r_ ) _< p_ ( _−_ 1 _| τ_ 1 _, τ_ 2 _, r_ ) _,_





where _p_ (1 _| τ_ 1 _, τ_ 2 _, r_ ) denotes the probability that _τ_ 1 is preferred to _τ_ 2 under the ground-truth reward
_r_ . Under Assumption K.2, preferences are noiselessly determined by regret, i.e.






_p_ (1 _| τ_ 1 _, τ_ 2 _, r_ ) =


1 if regret( _τ_ 1 _| r_ ) _<_ regret( _τ_ 2 _| r_ ) _,_
12 if regret( _τ_ 1 _| r_ ) = regret( _τ_ 2 _| r_ ) _,_
0 if regret( _τ_ 1 _| r_ ) _>_ regret( _τ_ 2 _| r_ ) _._





**1944**

**1945**


**1946**

**1947**

**1948**

**1949**

**1950**

**1951**


**1952**

**1953**

**1954**

**1955**

**1956**

**1957**


**1958**

**1959**

**1960**

**1961**

**1962**


**1963**

**1964**

**1965**

**1966**

**1967**

**1968**


**1969**

**1970**

**1971**

**1972**

**1973**

**1974**


**1975**

**1976**

**1977**

**1978**

**1979**

**1980**


**1981**

**1982**

**1983**

**1984**

**1985**


**1986**

**1987**

**1988**

**1989**

**1990**

**1991**


**1992**

**1993**

**1994**

**1995**

**1996**

**1997**


Under the single-start-state assumption (Assumption K.5),

E _τ_ _∼π_ �regret( _τ_ _| r_ )� = _Vr_ _[∗]_ [(] _[s]_ [0][)] _[ −]_ _[V]_ _r_ _[π]_ [(] _[s]_ [0][)] _[.]_


Substituting this identity and rearranging gives

_Vr_ _[∗]_ [(] _[s]_ [0][)] _[ −]_ _[V]_ _r_ _[π]_ _r_ _[∗]_ ˆ ( _s_ 0) _≤_ _Vr_ _[∗]_ [(] _[s]_ [0][)] _[ −]_ _[V]_ _r_ _[π]_ [ref] ( _s_ 0) _._


Rearranging terms again gives

_r_ ˆ
_Vr_ _[π][∗]_ ( _s_ 0) _≥_ _Vr_ _[π]_ [ref] ( _s_ 0)

which is equivalent to
_Jr_ ( _πr_ _[∗]_ ˆ [)] _[≥]_ _[J][r]_ [(] _[π]_ [ref] [)] _[.]_


K.3 THEOREM K.1, COROLLARY K.2


Skalse et al. (2022) defines reward hacking as:

**Definition K.5.** _(Skalse et al., 2022) A pair of reward functions_ (˜ _r_ 1 _,_ ˜ _r_ 2) _is hackable relative to a_
_policy set_ Π _and an environment_ ( _S, A, T, γ, p_ 0 _, __ ) _if there exists π, π_ _[′]_ _∈_ Π _such that_

_Jr_ ˜1( _π_ ) _> Jr_ ˜1( _π_ _[′]_ ) _and Jr_ ˜2( _π_ ) _< Jr_ ˜2( _π_ _[′]_ )


This canonical definition of reward hacking intuitively states that, if (˜ _r_ 1 _,_ ˜ _r_ 2) is not hackable given a
set of policies Π, then there does not exist any pair of policies in Π such that one policy has a strictly
higher expected return under ˜ _r_ 1 while the other has a strictly higher expected return under ˜ _r_ 2. Framed
differently, any increase in expected return under ˜ _r_ 1 from switching policies in Π never decreases the
expected return under ˜ _r_ 2. This definition is particularly strict, as noted by Skalse et al. (2022).

The proof for Corollary K.2 immediately follows from Theorem K.1 that if _K_ (ˆ _r, r_ ; _Tπr_ _[∗]_ ˆ _[∪T][π]_ [ref] [) = 1]
and we define the set of policies as

Π = _{π_ ref _}_ _∪_ Π _[∗]_ _r_ ˆ _[,]_

where Π _[∗]_ _r_ ˆ [denotes the set of all policies optimal for][ ˆ] _[r]_ [, then the pair][ (ˆ] _[r, r]_ [)][ is not hackable relative to][ Π]
under Definition K.5.


K.4 THEOREM K.1, COROLLARY K.3


Laidlaw et al. (2024) define reward hacking as:
**Definition K.6.** _(Laidlaw et al., 2024) Suppose_ ˜ _r is a ρ-correlated proxy_ [5] _with respect to πref._ _The_
_proxy reward function is hackable with respect to the ground-truth reward function and πref if,_

_Jr_ ( _πr_ _[∗]_ ˜ [)] _[ < J][r]_ [(] _[π][ref]_ [)] _[.]_


This definition of reward hacking only considers reward functions that are reasonable optimization
targets, i.e., by requiring the inputted reward function to be _ρ_ -correlated with the ground-truth reward
function, and sets a threshold for reward hacking via the expected return of the reference policy under
the ground-truth reward function. This definition is less strict than that of Skalse et al. (2022); see
Laidlaw et al. (2024) for an in-depth comparison,


The proof for Corollary K.3 immediately follows from Theorem K.1 that if _K_ (ˆ _r, r_ ; _Tπr_ _[∗]_ ˆ _[∪T][π]_ [ref] [) = 1][,]
then ˆ _r_ is not hackable with respect to _r_ and _π_ ref under Definition K.6.


5For the full definition of _ρ_ -correlation, we refer readers to the original paper from Laidlaw et al. (2024).


37