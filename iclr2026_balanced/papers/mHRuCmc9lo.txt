## ROBUST DECISION MAKING WITH PARTIALLY CALI### BRATED FORECASTS


**Shayan Kiyani, Hamed Hassani, George Pappas & Aaron Roth**
University of Pennsylvania
_{_ shayank, hassani, pappasg, aaroth _}_ @seas.upenn.edu


ABSTRACT


Calibration has emerged as a foundational goal in “trustworthy machine learning”,
in part because of its strong decision theoretic semantics. Independent of the underlying distribution, and independent of the decision maker’s utility function, calibration promises that amongst all policies mapping predictions to actions, the uniformly best policy is the one that “trusts the predictions” and acts as if they were
correct. But this is true only of _fully_ _calibrated_ forecasts, which are tractable to
guarantee only for very low dimensional prediction problems. For higher dimensional prediction problems (e.g. when outcomes are multiclass), weaker forms
of calibration have been studied that lack these decision theoretic properties. In
this paper we study how a conservative decision maker should map predictions
endowed with these weaker (“partial”) calibration guarantees to actions, in a way
that is robust in a minimax sense: i.e. to maximize their expected utility in the
worst case over distributions consistent with the calibration guarantees. We characterize their minimax optimal decision rule via a duality argument, and show
that surprisingly, “trusting the predictions and acting accordingly” is recovered in
this minimax sense by _decision_ _calibration_ (and any strictly stronger notion of
calibration), a substantially weaker and more tractable condition than full calibration. For calibration guarantees that fall short of decision calibration, the minimax
optimal decision rule is still efficiently computable, and we provide an empirical
evaluation of a natural one that applies to any regression model solved to optimize
squared error.


1 INTRODUCTION


Machine learning systems are increasingly deployed in high-stakes decision making domains such
as healthcare, finance, and law. The predictive power of these models can be extraordinary, but
scoring well on predictive error metrics does not directly guarantee that decisions downstream of
those predictions will be correct. For predictions to be operationally useful, a decision-maker must
be able to treat them as reliable inputs into a downstream decision making policy. This raises two
fundamental questions:


**On the Model Side:** _What does it mean for machine learning predictions to be trustworthy in_
_decision-making contexts?_


**On the Decision Making Side:** _Given predictions that satisfy a particular type of_
_“trustworthiness”, how should the decision maker adapt its actions to the promised guarantees?_


**On the Model Side:** A natural answer is that trustworthy predictions should directly support good
decisions as they are. In other words, the decision-maker should be able to reliably best respond
to the forecaster’s predictions as if they were correct. Formally, let ( _X, Y_ ) be a pair of random
variables drawn from a joint distribution _D_, where _X_ _∈X_ represents the observed features and
_Y_ _∈_ [0 _,_ 1] _[d]_ is the outcome of interest. Let _A_ denote the action set, and suppose the decision-maker
follows a policy _a_ ( _·_ ) : [0 _,_ 1] _[d]_ _→A_ mapping predictions to actions. Given a predictor _f_, the decision
maker’s performance when using a policy _a_ is measured by its expected utility on the underlying
distribution:
E( _X,Y_ ) [ _u_ ( _a_ ( _f_ ( _X_ )) _, Y_ )] _,_
_∼D_


1


where _u_ ( _a, y_ ) _∈_ R is a utility function. Given a forecaster _f_ : _X_ _→_ [0 _,_ 1] _[d]_, the _plug-in best response_
to a forecast is defined as
_a_ BR( _f_ ( _x_ )) = arg max (1)
_a_ _[u]_ [(] _[a, f]_ [(] _[x]_ [))] _[.]_
_∈A_

Thus, a forecaster _f_ is trustworthy if the decision-maker’s best-response policy _a_ BR( _f_ ( _x_ )) achieves
higher utility than any other policy. When is this the case?


The classical answer lies in the notion of _calibration_ . Intuitively, a forecaster is calibrated if, whenever it predicts a vector _f_ ( _x_ ) = _v_ _∈_ [0 _,_ 1] _[d]_, the empirical outcomes are consistent with that prediction. More formally, a forecaster _f_ is said to be _fully calibrated_ if for every _v_ _∈_ [0 _,_ 1] _[d]_,

E[ _Y_ _| f_ ( _X_ ) = _v_ ] = _v._

It is well known that best responding to calibrated forecasts is the optimal decision policy among all
policies that map forecasts to actions (Foster & Vohra, 1997; Kleinberg et al., 2023; Noarov et al.,
2023; Roth, 2022).


However, achieving full calibration is extremely difficult, both in theory—the sample complexity
of calibrating an existing forecaster without harming its accuracy grows exponentially with the outcome dimension _d_ (Gopalan et al., 2024a)—and in practice, where empirical evidence shows systematic deviations from calibration, ranging from neural networks to large language models (Guo
et al., 2017; Kull et al., 2019; Gupta & Ramdas, 2022; Plaut et al., 2024). Thus, despite the appealing link between calibration and trustworthy ML-powered decision-making, this connection quickly
breaks down in real-world applications.


**On the Decision Making Side:** Decision making from predictions admits two canonical extremes.
At one end, the decision maker _aggressively_ _best_ _responds_ to the forecasts, acting as if they were
fully correct. At the other end, the decision maker _conservatively_ _plays_ _a_ _minimax-safety_ _strat-_
_egy_, arg max _a_ min _y_ _u_ ( _a, y_ ), treating the forecasts as if they carried no information about the
_∈A_ _∈Y_
instance.


Departing from these extremes, we treat a model _f_ and its forecast _f_ ( _x_ ) as information that constrains what the true, instance-conditional outcome distribution could be. In other words, after
observing _f_ ( _x_ ), the decision maker considers the set of _candidate realities_ —outcome distributions
consistent with the forecast and the available calibration guarantees. Intuitively, the “volume” of
this set is governed by the strength of calibration: under full calibration, the set collapses to the
forecast itself (the prediction can be treated as reality, at least in expectation), whereas as calibration
weakens, the set enlarges. A principled decision rule should therefore _tune its conservatism to what_
_the reality could be_, consistent with the provided guarantees. This idea, together with the fragility
of full calibration in practice, leads to the central question of this paper: _can_ _we_ _derive_ _optimal_
_decision-making policies under weaker and more practical conditions than full calibration?_


We answer this question affirmatively. We introduce a framework based on _conservative_ decision
making that nevertheless fully exploits _partially_ calibrated forecasts. This viewpoint echoes ideas in
robust optimization and control, but it has not been systematically developed for post hoc decision
making with partially calibrated machine-learning forecasts.


1.1 OUR RESULTS


We consider a parameterized family of weighted calibration guarantees that have recently become a
popular object of study (H´ebert-Johnson et al., 2018; Gopalan et al., 2022). Informally speaking, this
family of guarantees constrains the residuals of a predictor _f_ to be uncorrelated with a collection
of “test functions” _h_ _∈H_ mapping the range of _f_ to the reals. When _H_ consists of all such
test functions, we recover full calibration, but many popular variants of calibration (e.g. top label
calibration, decision calibration, etc) can be expressed as instances of _H_ -calibration under much
smaller/more tractable sets _H_ . Our contributions are as follows:

1. In Section 2 we formalize the following question: given a set of test functions _H_ and a
predictor _f_ ( _x_ ) that is promised to satisfy _H_ -calibration, what decision rule _a_ : [0 _,_ 1] _[d]_ _→A_,
mapping predictions to actions, will maximize a decision maker’s expected utility in the
worst case over all joint distributions over _X_ _× Y_ that are consistent with the promise that
_f_ is _H_ -calibrated?


2


2. In Section 3 we answer this question by giving a closed-form for the decision maker’s optimal decision rule, in terms of the dual variables of a convex program that can be efficiently
computed for any finite _H_ .


3. In Section 4 we instantiate this decision rule for various calibration guarantees of interest.
Of particular note, we find that when _H_ corresponds to the tractable notion of _decision_
_calibration_ (Zhao et al., 2021; Noarov et al., 2023), then the optimal decision rule is the
best response decision rule _a_ BR, just as it is for (the intractable notion of) full calibration.
In fact, it suffices that _H contains_ the decision calibration constraints — any larger set _also_
makes best response the optimal decision rule. Thus what could have been a very large
hierarchy of minimax optimal decision rules “collapses” to best response at the level of
decision calibration. An upshot of this is that a predictor can be simultaneously decision
calibrated for many downstream decision makers, and for each of them, best response will
be their optimal decision policy in this minimax sense. We also derive the minimax optimal
decision rule for a simple “self-orthogonality” calibration condition that will hold for any
regression model with a linear final layer trained to optimize squared loss, and hence will
be commonly satisfied without any algorithmic intervention.


4. In Section 5 we train a two-layer MLP to minimize squared loss on two regression datasets,
and evaluate both the best-response decision rule and the robust decision rule that results
from the self-orthogonality condition of squared error regression. We find that, as predicted
by our theory, the robust decision rule outperforms the best-response decision rule under
calibration-preserving distribution shift, and that the cost of this robustness is mild even
under ideal conditions.


1.2 RELATED WORK


Rothblum & Yona (2023) consider a setting in which both the outcome and decision maker’s action
set are binary, and study how a decision maker should act to minimize their worst case regret over
distributions such that the predictor has maximum calibration error bounded by _α_ : informally that
_|_ E[ _Y |f_ ( _x_ ) = _v_ ] _−_ _v|_ _≤_ _α_ for all _v_ . The models _f_ they study are (approximately) fully calibrated,
which is a reasonable assumption in their setting, since they limit their study to 1-dimensional outcomes. In contrast, our interest is not (just) in quantitative measures of full calibration error, but
rather qualitatively weaker calibration guarantees, as even approximate full calibration becomes intractable in high dimensions.


A line of recent work (Zhao et al., 2021; Kleinberg et al., 2023; Noarov et al., 2023; Roth & Shi,
2024; Hu & Wu, 2024; Okoroafor et al., 2025) has studied the guarantees that can be given to
downstream decision makers who best respond to predictions that have weaker guarantees than full
calibration (and which in the cases of Zhao et al. (2021); Noarov et al. (2023); Roth & Shi (2024) can
be tractably guaranteed in higher dimensional outcome settings). These guarantees take the form of
(external and swap) _regret_ bounds, which are qualitatively weaker than the kind of “trustworthiness”
promised by full calibration. Informally, regret bounds promise that the decision maker could not
have done better by consistently playing a fixed action (or a fixed function remapping their actions to
other actions), not that they could not have done better by using a different policy from predictions
to actions. We show that even in high dimensions, the tractable “decision calibration” condition
given by Zhao et al. (2021) recovers the same “trustworthiness” semantics of full calibration when
viewed through our minimax decision making lens.


Analyzing minimax optimal decision policies is a common way of analyzing _robust_ or _risk-_
_averse_ decision making guarantees, with deep roots in economics (Gilboa & Schmeidler, 1989;
Hansen & Sargent, 2001; Manski, 2000; 2004; Manski & Tetenov, 2007; Manski, 2011), statistics
(Wald, 1950), and robust optimization (Ben-Tal & Nemirovski, 2002; Kuhn et al., 2019; Duchi &
Namkoong, 2021). For example, Carroll (2015) adopts this lens this in the context of contract theory
and Kiyani et al. (2025) and Andrews & Chen (2025) do so in the context of conformal prediction.
To the best of our knowledge, we are the first to apply this “robust” minimax lens to the problem of
partially calibrated high dimensional forecasts.


3


forecasters guaranteed to satisfy _H_ -calibration. This family of calibration guarantees has been studied extensively in the recent literature on multicalibration and its extensions (H´ebert-Johnson et al.,
2018; Dwork et al., 2021; Gopalan et al., 2022; Deng et al., 2023) — in particular, _H_ -calibration is
a special case of what Gopalan et al. (2022) call weighted multicalibration.

_H_ **-Calibration.** Let _H_ be a set of functions _h_ : [0 _,_ 1] _[d]_ _→_ R. A forecaster _f_ is said to be _H_ **-**
**calibrated** if for every _h ∈H_, - 


 -  E _h_ ( _f_ ( _X_ )) _·_ ( _Y_ _−_ _f_ ( _X_ )) = 0 _._ (2)


Equivalently, writing _q_ ( _v_ ) := E[ _Y_ _|_ _f_ ( _X_ ) = _v_ ] for the true conditional expectation, _H_ -calibration
requires


 -  E _h_ ( _f_ ( _X_ )) _·_ ( _q_ ( _f_ ( _X_ )) _−_ _f_ ( _X_ )) = 0 _,_ _∀h ∈H._ (3)


This definition captures a spectrum of guarantees. When _H_ contains all bounded measurable functions, _H_ -calibration reduces to full calibration — i.e. it requires that _f_ ( _v_ ) = _q_ ( _v_ ) := E[ _Y_ _| f_ ( _X_ ) =
_v_ ] almost surely. For smaller classes _H_, the requirement is weaker and can be seen as a relaxation of
calibration, enforcing consistency only with respect to a restricted set of tests. In the main body of
the paper we focus on the _H_ -calibration defined above, but in Appendix B we also discuss scenarios
in which only approximate _H_ -calibration is available.


**Robust Decision Making.** - Fix an _H_ -calibrated forecaster _f_ . Define the set


  - [��]   -   -   _Q_ = _q_ : [0 _,_ 1] _[d]_ _→_ [0 _,_ 1] _[d]_ E _h_ ( _f_ ( _X_ )) _·_ ( _q_ ( _f_ ( _X_ )) _−_ _f_ ( _X_ )) = 0 _,_ _∀h ∈H_ _._ (4)


In words, _Q_ consists of all candidate conditional expectations consistent with _f_ satisfying _H_ calibration. Because the perfect predictor _f_ ( _X_ ) = E[ _Y |X_ ] satisfies _H_ -calibration for every _H_,
the identity map _q_ ( _v_ ) = _v_ is always in _Q_ —but in general the set may contain many maps. From
the perspective of the decision-maker who knows _f_ and the promised calibration guarantee _H_, but
does not know the underlying distribution, given a forecast _f_ ( _x_ ), the true expectation E[ _Y_ _| f_ ( _x_ )] is
uncertain but must lie within _Q_ . As _H_ grows richer, _Q_ shrinks, eventually reducing to _{q_ ( _v_ ) = _v}_
in the case of full calibration.


Faced with this uncertainty, a natural strategy is to adopt a robust policy that guards against the
worst-case admissible reality. Formally, the robust decision rule is

                           -                           _a_ robust( ) = arg max min _u_ ( _a_ ( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ ))) _._ (5)

_·_ _a_ ( _·_ ):[0 _,_ 1] _[d]_ _→A_ _q∈Q_ [E]

That is, the decision-maker chooses an action policy that maximizes utility under the worst-case
conditional expectation consistent with calibration guarantees.


**Interpolating** **Property.** The robust policy in Equation 5 interpolates between two classical extremes (Figure 1). If contains all functions, then = _q_ ( _v_ ) = _v_ and _a_ robust reduces to the
_H_ _Q_ _{_ _}_
best response _a_ BR( ) (Equation equation 1). If is empty, then contains all functions and the
policy collapses to the constant minimax strategy _·_ _H a_ Minimax( _x_ ) = arg max _Q_ _a_ min _y_ [0 _,_ 1] _d u_ ( _a, y_ ) _._
_∈A_ _∈_
Thus, Equation 5 provides a principled bridge between best-responding to calibrated forecasts and
adopting fully conservative policies, with the level of conservatism controlled by the richness of _H_ .


4


The central theme of the remainder of this paper is to investigate the interaction between different
levels of _H_ -calibration and the resulting optimal robust policies. Our focus is not on developing
methods for achieving _H_ -calibration itself (for which we refer the reader to a rich line of recent work
showing how to accomplish this in both the batch and online adversarial setting H´ebert-Johnson
et al., 2018; Gopalan et al., 2022; Deng et al., 2023; Noarov et al., 2023; Globus-Harris et al., 2023),
but rather on understanding the decision-making consequences once such guarantees are in place. In
the next section, we begin by analyzing the general problem of deriving optimal robust decision rules
for arbitrary classes _H_ . We then specialize to the important case of decision calibration, showing that
this weaker and more practical notion identifies large classes of partially calibrated forecasters for
which best responding remains optimal. Beyond its theoretical appeal, this result is also practically
useful: when a decision-maker can influence the design or post-processing of the forecaster, they
can request a decision-calibrated forecaster, to which they can then simply, reliably, and optimally
best respond.

**Assumption 2.1.** The utility _u_ ( _a, v_ ) is linear in its second argument _v_ _∈_ [0 _,_ 1] _[d]_ for each _a ∈A_ .

This assumption naturally holds in multi-class settings where _v_ is a probability vector over _d_ outcomes and the decision maker has arbitrary utilities _U_ ( _a, k_ ) for each action–outcome pair. In this
case, _u_ ( _a, v_ ) = E[ _U_ ( _a, Y_ )] = [�] _[d]_ _k_ =1 _[v][k][ U]_ [(] _[a, k]_ [)] _[,]_ [ which is linear in] _[ v]_ [.] [Such risk-neutral expected-]
utility models underlie much of the calibration and decision-making literature (e.g., (Foster & Vohra,
1997; Kleinberg et al., 2023; Roth & Shi, 2024)). Utilities that are nonlinear in _v_, for example,
risk-averse utilities depending on outcome variance, fall outside our framework and represent an
important direction for future work.


3 OPTIMAL DECISION POLICIES FOR FINITE DIMENSIONAL
_H_ -CALIBRATION


In this section, we characterize the optimal robust decision making policies, i.e., solutions to Equation 5. Throughout this section, we assume the function class _H_ is a finite dimensional space, i.e.
it can be described as span of finitely many functions. Formally, let = span _h_ 1 _, . . ., hk_ be the
_H_ _{_ _}_
linear class generated by measurable _hi_ : [0 _,_ 1] _[d]_ _→_ R. Then the _H_ -calibration condition equation 3
is equivalent to the _k_ linear moment equalities

           -           E _hi_ ( _f_ ( _X_ )) _·_ ( _q_ ( _f_ ( _X_ )) _−_ _f_ ( _X_ ) ) = 0 _,_ _i_ = 1 _, . . ., k,_

so that the ambiguity set in equation 4 may be written as

   -    -    -    
[��]
_Q_ = _q_ : [0 _,_ 1] _[d]_ _→_ [0 _,_ 1] _[d]_  - E _hi_ ( _f_ ( _X_ )) _·_ ( _q_ ( _f_ ( _X_ )) _−_ _f_ ( _X_ ) ) = 0 for _i_ = 1 _, . . ., k_ _._


Intuitively, each equality enforces that, conditional on the forecast, the forecast error has zero correlation with the corresponding test _hi_ ; taken together, these constraints exhaust the information
provided by _H_ -calibration criteria and hence precisely describe the admissible reality faced by the
robust decision-maker in equation 5.

**Theorem** **3.1** (Characterization of the Optimal Robust Policy) **.** _Suppose_ = span _h_ 1 _, . . ., hk_
_H_ _{_ _}_
_with each hi_ : [0 _,_ 1] _[d]_ _→_ R _, and let Q be defined as above._ _Then the minimax problem in Equation_
_5 admits a saddle point_ ( _a_ robust _, q_ _[⋆]_ ) _with the following structure:_

_Therev_ = _f_ ( _existx_ ) _the worst-case mapmultipliers_ _λ_ _[⋆]_ = _q_ ( _λ_ _[⋆][⋆]_ 1( _v_ _[, . . ., λ]_ ) _solves_ _[⋆]_ _k_ [)] _[with]_ _[each]_ _[λ][⋆]_ _i_ _[∈]_ [R] _[d]_ _[such]_ _[that]_ _[for]_ _[almost]_ _[every]_ _[forecast]_


- _k_ 
_hi_ ( _v_ ) _λ_ _[⋆]_ _i_ _,_ _where_ val( _p_ ) = max
_a_ _[u]_ [(] _[a, p]_ [)] _[.]_
_i_ =1 _∈A_


_q_ _[⋆]_ ( _v_ ) arg min
_∈_ _p_ [0 _,_ 1] _[d]_
_∈_


val( _p_ ) + _p ·_


_Given q_ _[⋆]_ _, the optimal robust action at v is the best response to q_ _[⋆]_ ( _v_ ) _:_

                         -                          _a_ robust( _v_ ) arg max _a, q_ _[⋆]_ ( _v_ ) _._
_∈_ _a_ _[u]_
_∈A_

**Interpretation.** Theorem 3.1 characterizes both the worst-case distribution consistent with _H_ calibration and the corresponding optimal response. For any realized forecast _ν_ = _f_ ( _x_ ), the theorem


5


yields a simple two-step procedure: compute the adversarial belief


_q_ _[⋆]_ ( _ν_ ) arg min _s_ _[⋆]_ ( _ν_ ) =
_∈_ _p_ [0 _,_ 1] _[d][{]_ [val(] _[p]_ [) +] _[ p][ ·][ s][⋆]_ [(] _[ν]_ [)] _[}][,]_
_∈_


- _k_

_hi_ ( _ν_ ) _λ_ _[⋆]_ _i_ _[,]_
_i_ =1


and then take the best response _a_ robust( _ν_ ) arg max _a_ _u_ ( _a, q_ _[⋆]_ ( _ν_ )). Thus, the optimal policy
_∈_ _∈A_
is always a best response, not to the raw forecast _f_ ( _x_ ), but to the adversarially tilted distribution
_q_ _[⋆]_ ( _ν_ ) allowed by the calibration constraints. Additionally, a useful consequence is _pointwise com-_
_putability_ : evaluating _a_ robust at a given _ν_ reduces to two low-dimensional optimizations, without
constructing the full mapping _x_ _a_ robust( _x_ ).
_�→_

From an optimization perspective, the multipliers _λ_ _[⋆]_ solve a finite-dimensional concave maximization problem (see the proof of Theorem 3.1), and _q_ _[⋆]_ ( _ν_ ) is obtained by a pointwise convex minimization over _p_ _∈_ [0 _,_ 1] _[d]_ . Both stages can be carried out by standard, fast methods
with provable guarantees (e.g., projected subgradient ascent for the dual, or a simple primal–dual
scheme), after which one evaluates _q_ _[⋆]_ ( _ν_ ) via the pointwise minimization and takes the best response
_a_ robust( _ν_ ) = arg max _a u_ ( _a, q_ _[⋆]_ ( _ν_ )).


In the next section, we analyze the behavior of the resulting decision rules by specializing to concrete
_H_ -classes. One might expect that Theorem 3.1 induces a vast hierarchy of policies whose form
depends sensitively on _H_ . _Perhaps surprisingly, this is not the case._ In particular, we show a sharp
transition: for each decision maker, there exists a specific test class, precisely the one associated with
_decision calibration_, such that as soon as _H_ contains this class, the adversarial tilt collapses ( _q_ _[⋆]_ ( _ν_ ) =
_ν_ for a.e. _ν_ ) and the optimal robust rule reduces to the plug-in best response to the forecaster.


4 ROBUST POLICIES UNDER DECISION CALIBRATION AND BEYOND


In this section, we specialize the general characterization derived in Theorem 3.1 to concrete test
classes _H_ . Our core result concerns _decision_ _calibration_ : a practically tractable guarantee under
which the minimax-optimal robust policy collapses to the plug-in (best-response) rule. This identifies a simple path to decision-theoretic trustworthiness that does not require full calibration.


4.1 DECISION CALIBRATION AND PLUG-IN BEST RESPONSE OPTIMALITY


Here we define the variant of decision calibration given by Noarov et al. (2023), a slight strengthening of the definition originally given by Zhao et al. (2021). Fix a single decision problem with
action set _A_ and utility function _u_ ( _a, v_ ). For each action _a ∈A_, let

          -          _Ra_ = _v_ [0 _,_ 1] _[d]_ : _u_ ( _a, v_ ) _u_ ( _a_ _[′]_ _, v_ ) for all _a_ _[′]_
_∈_ _≥_ _∈A_

be the (closed, convex) decision region on which _a_ is a plug-in best response. The _decision-_
_calibration class_ is dec = **1** _Ra_ : _a_ _._ Here, we denote **1** _A_ ( _x_ ) := **1** _x_ _A_ . A forecaster
_H_ _{_ _∈A }_ _{_ _∈_ _}_
_f_ is _decision calibrated_ if it is dec-calibrated, i.e.,

             - _H_             - �� ��
E **1** _Ra_ _f_ ( _X_ ) _Y_ _−_ _f_ ( _X_ ) = 0 for all _a ∈A._


Compared to full calibration, decision calibration is far more statistically tractable, since its test class
has size dec =, a potentially small and fixed number of actions, rather than the large families
_|H_ _|_ _|A|_
required for full calibration.
**Theorem** **4.1** (Decision calibration plug-in best response optimality) **.** _If_ _f_ _is_ dec _-calibrated,_
_⇒_ _H_
_then the minimax-optimal robust rule in equation 5 coincides with the plug-in best response:_


_a_ robust( _v_ ) arg max _for almost every v_ = _f_ ( _x_ ) _._
_∈_ _a_ _[u]_ [(] _[a, v]_ [)]
_∈A_

_Equivalently,_ _under_ _decision_ _calibration,_ _best_ _responding_ _to_ _the_ _forecaster_ _is_ _minimax_ _optimal_
_among all forecast-based policies._


Put differently, upon observing a forecast _v_ = _f_ ( _x_ ), the decision-maker need only best respond to
_v_ ; no adversarial “tilt” survives the decision-calibration constraints. Conceptually, this upgrades the
previously known guarantees of decision calibration—that it implies no swap regret (Noarov et al.,


6


Figure 2: Schematic of the Sharp Transition


2023)—to _minimax_ _optimality_ . Swap regret guarantees do not preclude the existence of a policy
_a_ : [0 _,_ 1] _[d]_ _→A_ that dominates the plugin best response policy _a_ BR — only that no improved policy
has the form _a_ ( _v_ ) = _ϕ_ ( _a_ BR( _v_ )) for some mapping _ϕ_ :, using “actions as a bottleneck”. In
_A_ _→A_
contrast, Theorem 4.1 directly establishes that no other policy _a_ : [0 _,_ 1] _[d]_ _→A_ can improve on the
plugin policy _a_ BR in our minimax sense.


The preceding result assumes that the information conveyed by the forecaster to the decision-maker
is exhausted by the decision-calibration tests _{_ **1** _Ra_ _}a∈A_ . In practice, a forecaster might satisfy
additional calibration equalities,

                 -                 E _h_ ( _f_ ( _X_ )) _· {Y_ _−_ _f_ ( _X_ ) _}_ = 0 _,_

for functions _h_ beyond the indicators **1** _Ra_ . The next theorem shows that the plug-in optimality
conclusion is stable under such enrichments. This is intuitive: if a forecaster is trustworthy, then
making it more calibrated (i.e., adding information) should not diminish that trustworthiness.
**Theorem** **4.2.** _Let_ _be_ _any_ _test_ _class_ _that_ _contains_ _the_ _decision-calibration_ _indicators,_ dec =
_H_ _H_
**1** _Ra_ : _a_ _._ _If f_ _is perfectly_ _-calibrated, then the minimax-optimal robust rule in equation 5_
_{_ _∈A}_ _H_
_coincides (a.e.)_ _with the plug-in best response:_


_a_ robust( _v_ ) arg max _for a.e. v_ = _f_ ( _x_ ) _._
_∈_ _a_ _[u]_ [(] _[a, v]_ [)]
_∈A_


As we make precise in the proof of Theorem 4.2, the “collapse” occurs because the decisioncalibration constraints ensure that the expected utility of the plug-in best-response policy _aBR_ is
_invariant_ to the adversary’s choice of _q_ . For any _q_ satisfying the dec constraints,
_∈Q_ _H_

E[ _u_ ( _aBR_ ( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ )))] = E[ _u_ ( _aBR_ ( _f_ ( _X_ )) _, f_ ( _X_ ))] _._


Thus, the adversary cannot reduce the utility of _aBR_ ; its worst-case utility equals its nominal utility.
Since _aBR_ is the optimal policy under the nominal distribution, and its performance cannot degrade
under any admissible _q_, it must also be the minimax-optimal policy.


**Sharp** **transition.** One might initially expect a _gradual_ shift from fully conservative to plug-in
best response as _H_ is enriched. Theorems 4.1–4.2 show a sharper phenomenon (Figure 2): once _H_
contains the _|A|_ decision tests _{_ **1** _Ra_ _}a∈A_, the adversarial tilt disappears ( _q_ _[⋆]_ ( _ν_ ) = _ν_ a.e.) and the
robust rule _collapses_ to the plug-in best response equation 1. Enlarging _H_ further does not change
the minimax-optimal policy.


As a byproduct, this leads to another practical advantage of decision calibration: a single forecaster
can be made simultaneously reliable for a _collection_ of downstream decision problems. Intuitively,
if the forecast passes the decision calibration tests of each problem, then none of the decision makers
needs additional robustness, the plug-in best-response is minimax-optimal for all of them.
**Corollary** **4.3** (Simultaneous plug-in optimality across multiple decisions) **.** _Let_ _u_ 1 _, . . ., um_ _be_ _m_
_decision_ _problems,_ _with_ _respective_ _action_ _sets_ _j_ _and_ _linear_ _utilities_ _uj_ ( _a, v_ ) _in_ _v_ [0 _,_ 1] _[d]_ _._ _For_
_A_ _∈_
_each j_ _and a_ _j, let_
_∈A_

_Ra,j_ = _v_ [0 _,_ 1] _[d]_ : _uj_ ( _a, v_ ) _uj_ ( _a_ _[′]_ _, v_ ) _for all a_ _[′]_ _j_
_{_ _∈_ _≥_ _∈A_ _}_


7


_be the plug-in decision region of action a in problem j, and define the combined test class_


_H_ dec [all] [=]


- _m_


_j_ =1


- **1** _Ra,j_ : _a_ _j_ _._
_∈A_


_If f_ _is_ _-calibrated for some_ _satisfying_ dec
_optimal robust policy for problem H_ _H_ _j_ _coincides (a.e.) H_ [all] _[⊆H]_ _with the plug-in best response:_ _[, then for every][ j]_ _[∈{]_ [1] _[, . . ., m][}][ the minimax-]_


_a_ robust _,j_ ( _v_ ) arg max _for a.e. v_ = _f_ ( _x_ ) _._
_∈_ _a_ _j_ _[u][j]_ [(] _[a, v]_ [)]
_∈A_


_Proof._ For each problem _j_, the included indicators _{_ **1** _Ra,j_ _}a∈Aj_ ensure that _H_ contains the decisioncalibration tests of problem _j_ . Theorem 4.2 then applies verbatim to each _j_, yielding plug-in optimality problem by problem.


4.2 BEYOND DECISION CALIBRATION: GENERIC _H_ -CLASSES FROM TRAINING PIPELINES

Thus far we have focused on _decision calibration_, which, when attainable, collapses _a_ robust to the
plug-in best response. In practice, two regimes arise. (i) If one can influence the forecaster’s training
pipeline, decision calibration is the natural target: it is practical, and our results guarantee plug-in
minimax optimality. (ii) If one _cannot_ control training, the forecaster might not be decision calibrated for the downstream task. Identifying its partial-calibration profile may be difficult, yet certain
moment conditions arise _structurally_ from standard training procedures. We give two examples of
how to leverage such “free” structure to specify usable _H_ ’s and derive the associated robust policies.

**Self-orthogonality from squared-loss training.** A ubiquitous example is _self-orthogonality_ (a form
of self-calibration) that follows from first-order optimality when a model with a linear last layer is
trained to minimize mean squared error. This includes the universally adopted cases of regression
with either a linear model or a neural network with a linear head, trained by mean squared error. This
and similar guarantees for other loss functions have previously been investigated as consequences
of _low degree multicalibration_ (Gopalan et al., 2022).
**Proposition 4.4** (Self-orthogonality under squared loss) **.** _Let X_ _�→_ _zϕ_ ( _X_ ) _∈_ R _[k]_ _be a representation_
_and_ _fθ_ ( _X_ ) = _Wzϕ_ ( _X_ ) _∈_ R _[d]_ _a_ _linear_ _last_ _layer._ _Suppose_ _θ_ = ( _ϕ, W_ ) _is_ _trained_ _to_ _a_ _first-order_
_stationary point of the expected squared loss_


_L_ ( _θ_ ) = 21 [E] ��� _fθ_ ( _X_ ) _−_ _Y_ ��22


_._


_Then the following calibration moments hold:_

        -        E _zϕ_ ( _X_ ) ( _Y_ _−_ _fθ_ ( _X_ )) _[⊤]_ [�] = 0 _and_ E _fθ_ ( _X_ ) ( _Y_ _−_ _fθ_ ( _X_ )) _[⊤]_ [�] = 0 _._

_In particular, fθ_ _is H-calibrated for the test class H_ = _{hj_ ( _v_ ) = _e_ _[⊤]_ _j_ _[v]_ [:] _[j]_ [= 1] _[, . . ., d][}][ (and for any]_
_linear combination thereof)._


**Implications.** Proposition 4.4 provides a generic, pipeline-induced _H_ -calibration guarantee whenever a linear head is trained to stationarity under squared loss. Specializing Theorem 3.1 to this
setting yields a simple dual. For _d_ = 1 (e.g., one-dimensional regression) with _H_ = _{h_ ( _v_ ) = _v}_,
the multiplier is a scalar _λ_, and for each forecast _ν_ = _f_ ( _x_ ) the worst-case distribution is

_q_ _[⋆]_ ( _ν_ ) arg min val( _p_ ) = max
_∈_ _p_ [0 _,_ 1] _[{]_ [val(] _[p]_ [) +] _[ λ ν p][}][,]_ _a_ _[u]_ [(] _[a, p]_ [)] _[.]_
_∈_ _∈A_

The robust action is then: _a_ robust( _ν_ ) arg max _a_ _u_ ( _a, q_ _[⋆]_ ( _ν_ )) _._ When _u_ ( _a, p_ ) is linear in _p_ and
_∈_ _∈A_ _A_
is finite, val is convex piecewise linear, so the inner minimization reduces to checking finitely many
candidate points (endpoints and pairwise breakpoints). The dual objective

                -                _G_ ( _λ_ ) = E min _λ_ E[ _f_ ( _X_ ) [2] ]
_p_ [0 _,_ 1] _[{]_ [val(] _[p]_ [) +] _[ λf]_ [(] _[X]_ [)] _[p][}]_ _−_
_∈_

is concave in _λ_ and can be maximized via standard one-dimensional methods (e.g., bisection on a
monotone subgradient). In higher dimensions ( _d_ _>_ 1), the correction term _λνp_ becomes Λ _νp_ for a
matrix of multipliers Λ, and the pointwise problem remains a small convex program over _p ∈_ [0 _,_ 1] _[d]_ ;
for finite _A_ and linear utilities, it is again efficiently solvable.


8


**Zero-bias** **and** **bin-wise** **calibration.** A widely available source of partial calibration comes from
_post-hoc_ _recalibration_ that many practitioners already apply (mean correction, histogram binning,
isotonic-style step fits on a held-out split). These procedures enforce generic (not task-specific)
moment constraints that are directly usable in our framework. We focus on _bin-wise_ calibration:
take a partition of the forecast range into bins� _{B_ 1 _, . . ., B_ - _J_ _}_ and enforce, for each bin,

E **1** _f_ ( _X_ ) _Bj_ ( _Y_ _f_ ( _X_ )) = 0 _,_ _j_ = 1 _, . . ., J._
_{_ _∈_ _}_ _−_

This corresponds to the test class bin = **1** _Bj_ : _j_ = 1 _, . . ., J_ _,_ and reduces to zero-bias when
_H_ _{_ _}_
_J_ =1 with _B_ 1 = [0 _,_ 1] _[d]_ .
**Proposition 4.5** (Robust policy under bin-wise calibration) **.** _Let the utility be linear in the outcome_
_and the action set_ _be finite._ _If f_ _is_ bin _-calibrated, then with_
_A_ _H_
_mj_ := E[ _f_ ( _X_ ) _| f_ ( _X_ ) _∈_ _Bj_ ] = E[ _Y_ _| f_ ( _X_ ) _∈_ _Bj_ ] _,_
_the worst-case belief is piecewise constant_
_q_ _[⋆]_ ( _v_ ) = _mj_ _for v_ _Bj_ _(a.e.),_
_∈_
_and the robust action best-responds to the bin mean:_

               -                _a_ robust( _v_ ) arg max _u_ ( _a, mj_ ) _for v_ _Bj_ _(a.e.)._
_∈_ _a_ _∈_
_∈A_

**Implications.** Bin-wise calibration bin can be obtained cheaply via standard post-hoc methods
_H_
(histogram binning or isotonic regression), and Proposition 4.5 yields an especially simple, closedform characterization of the robust policy. Computing _a_ robust reduces to: (i) estimating _mj_ on
a calibration split, and (ii) at test time, mapping _v_ to its bin _Bj_ and best-responding to _mj_ . No
additional optimization is needed to compute actions. As a special case, when _J_ = 1 we recover the
global-mean constraint E[ _Y_ _−_ _f_ ( _X_ )] = 0. Then _q_ _[⋆]_ is constant, _q_ _[⋆]_ ( _v_ ) _≡_ _m_ ¯, with _m_ ¯ = E[ _f_ ( _X_ )] =
E[ _Y_ ], and the robust rule ignores _v_ and plays arg max _a_ _u_ ( _a,_ _m_ ¯ ). As the partition is refined,
_∈A_
the robust rule moves from a single global plug-in best response at _m_ ¯ to a piecewise plug-in best
response at _mj_, yielding a richer, finer-grained decision policy.


5 EXPERIMENTS


In this section, we evaluate the validity and practical consequences of our framework by implementing our methods on two real-world datasets. We compare the _plug-in_ _best_ _response_ ( _a_ BR) against
the _robust policy_ ( _a_ robust), which enjoys minimax optimality guarantees under -calibration.
_H_

We focus on two classes of metrics. _Nominal_ _performance_ measures average utility when the test
data are i.i.d. from the same distribution as the training and calibration splits; this reflects an optimistic regime that often degrades in practice. _Adversarial performance_ probes the other extreme by
altering the test-time outcome distribution in two ways: (i) a worst case tailored to the plug-in policy, and (ii) a worst case induced by the robust dual, tailored to the robust policy. In both cases, the
adversarial distributions respect the _H_ -calibration constraints and are therefore indistinguishable,
from the decision-maker’s perspective, from i.i.d. test draws given an _H_ -calibrated forecaster.

Our theory predicts two patterns. First, by minimax optimality, the robust policy should dominate
the plug-in rule when each is evaluated against its _own_ worst-case distribution (and typically also
under the adversary tuned to hurt the plug-in). Second, because ( _a_ robust _, q_ _[⋆]_ ) forms a saddle point
of equation 5, when both policies are evaluated under the robust-tuned adversary, the robust policy
should not underperform the plug-in rule. Under nominal i.i.d. evaluation, the plug-in rule may
achieve higher utility, reflecting the lack of need for conservatism in that regime.


5.1 CASE STUDIES: BIKE SHARING AND CALIFORNIA HOUSING


We evaluate our framework on two regression datasets with distinct decision-making interpretations.


**Bike Sharing (UCI).** The UCI _Bike Sharing_ (daily) dataset Fanaee-T & Gama (2014) records daily
rider counts alongside calendar and weather covariates (season, month, weekday, holiday, working
day, weather state, temperature, humidity, wind). The outcome _Y_ _∈_ [0 _,_ 1] is the rescaled total rider
count, and the decision-maker chooses a staffing/capacity multiplier from _A_ = _{_ 0 _._ 8 _,_ 1 _._ 0 _,_ 1 _._ 2 _},_
interpretable as conservative, nominal, and aggressive provisioning.


9


Table 1: Mean utility on the test set under natural i.i.d. evaluation and two adversarial evaluations.
Adversaries respect _H_ -calibration ( _H_ = _{h_ ( _v_ ) = _v}_ ).


i.i.d. Worst-case for robust Worst-case for plug-in
Dataset

Plug-in Robust Plug-in Robust Plug-in Robust


Bike Sharing (UCI) 0.474 0.463 0.402 0.410 0.393 0.412


California Housing 0.216 0.207 0.160 0.164 0.155 0.166


**California Housing.** The _California Housing_ dataset Pace & Barry (1997) records median house
values (rescaled to [0 _,_ 1]) with demographic and geographic covariates (median income, housing
age, population, latitude/longitude, etc.). Here the decision-maker chooses an investment multiplier
from _A_ = _{_ 0 _._ 6 _,_ 0 _._ 75 _,_ 0 _._ 90 _},_ interpretable as conservative, nominal, and aggressive investment.

**Utility specification.** In both settings we adopt the utility function _u_ ( _a, y_ ) = _α a y −_ _C_ ( _a_ ) _,_ which
is linear in _y_ . The benefit term _α a y_ captures service or return proportional to realized outcome _y_,
scaled by _α >_ 0. The cost term _C_ ( _a_ ) grows in _a_, penalizing aggressive choices via over-provisioning
costs or investment risk. This form tunes the under/over-trade-off without departing from linearity.
For Bike Sharing we use ( _α, C_ ( _·_ )) = (0 _._ 9 _, {_ 0 _._ 02 _,_ 0 _._ 05 _,_ 0 _._ 1 _}_ ), while for California Housing we use
( _α, C_ ( _·_ )) = (0 _._ 9 _, {_ 0 _._ 02 _,_ 0 _._ 05 _,_ 0 _._ 20 _}_ ). The qualitative conclusions of this Section remain the same
under other reasonable parameter choices.


**Forecasting** **model.** In both datasets, the forecaster _f_ is a two-layer MLP regressor trained to
optimize mean squared error. By the self-orthogonality property of linear heads under squared loss
(Proposition 4.4), the learned forecaster approximately satisfies _H_ -calibration with _H_ = _{h_ ( _v_ ) =
_v_, which is the calibration constraint used to derive the robust policy _a_ robust. All experiments use
_}_
an i.i.d. train/calibration/test split (60/20/20). We use the calibration data to substitute any population
level expectation that is needed to be computed to derive _a_ robust.


**Results.** Table 1 reports the mean utilities. The results match theory: under adversaries tailored
to the robust policy, the robust rule achieves at least the plug-in performance; under adversaries
tuned to harm the plug-in rule, the robust policy secures noticeably higher utility, reflecting its
minimax protection. Moreover, the robust policy outperforms the plug-in best response when each
is evaluated against its own worst-case distribution.


6 CONCLUSION AND LIMITATIONS


We developed a decision-theoretic framework for acting on partially calibrated forecasts via a
minimax-optimal robust policy over _H_ -calibrated forecasters. We then identified a sharp transition
in the behavior of these policies: for any decision problem with _m_ actions, there exist _m_ decision
tests (the decision-calibration class) such that, once they are included in _H_, the robust policy _col-_
_lapses_ to the plug-in best response. This spotlights decision calibration as a natural requirement
whenever the decision-maker can influence the training pipeline. Moreover, even when decision
calibration is unavailable, we showed that generic properties induced by standard training and post
hoc procedures (e.g., self-orthogonality under squared loss and bin-wise calibration) yield usable
test classes _H_ and tractable robust policies within our framework.

Our model assumed that downstream decision makers were risk neutral — i.e., their utility functions
_u_ ( _a, v_ ) are linear in _v_ and _A_ is finite; these are standard assumptions in the calibration literature, but
broadening them would be interesting. We note that certain classes of non-linear utility functions can
be linearized over an appropriate basis (Gopalan et al., 2024b; Lu et al., 2025), which would allow
our results to apply — though these bases are not always low dimensional enough to be practical.


REFERENCES


Isaiah Andrews and Jiafeng Chen. Certified decisions. _arXiv preprint arXiv:2502.17830_, 2025.


10


Aharon Ben-Tal and Arkadi Nemirovski. Robust optimization–methodology and applications.
_Mathematical programming_, 92(3):453–480, 2002.


Gabriel Carroll. Robustness and linear contracts. _American_ _Economic_ _Review_, 105(2):536–563,
2015.


Zhun Deng, Cynthia Dwork, and Linjun Zhang. Happymap: A generalized multicalibration method.
In _14th Innovations in Theoretical Computer Science Conference (ITCS 2023)_, pp. 41–1. Schloss
Dagstuhl–Leibniz-Zentrum f¨ur Informatik, 2023.


John C Duchi and Hongseok Namkoong. Learning models with uniform performance via distributionally robust optimization. _The Annals of Statistics_, 49(3):1378–1406, 2021.


Cynthia Dwork, Michael P Kim, Omer Reingold, Guy N Rothblum, and Gal Yona. Outcome indistinguishability. In _Proceedings_ _of_ _the_ _53rd_ _Annual_ _ACM_ _SIGACT_ _Symposium_ _on_ _Theory_ _of_
_Computing_, pp. 1095–1108, 2021.


Hadi Fanaee-T and Joao Gama. Event labeling combining ensemble detectors and background
knowledge. _Progress in Artificial Intelligence_, 2(2):113–127, 2014.


Dean P Foster and Rakesh V Vohra. Calibrated learning and correlated equilibrium. _Games_ _and_
_Economic Behavior_, 21(1-2):40–55, 1997.


Itzhak Gilboa and David Schmeidler. Maxmin expected utility with non-unique prior. _Journal_ _of_
_mathematical economics_, 18(2):141–153, 1989.


Ira Globus-Harris, Declan Harrison, Michael Kearns, Aaron Roth, and Jessica Sorrell. Multicalibration as boosting for regression. In _International_ _Conference_ _on_ _Machine_ _Learning_, pp. 11459–
11492. PMLR, 2023.


Parikshit Gopalan, Michael P Kim, Mihir A Singhal, and Shengjia Zhao. Low-degree multicalibration. In _Conference on Learning Theory_, pp. 3193–3234. PMLR, 2022.


Parikshit Gopalan, Lunjia Hu, and Guy N Rothblum. On computationally efficient multi-class calibration. In _The Thirty Seventh Annual Conference on Learning Theory_, pp. 1983–2026. PMLR,
2024a.


Parikshit Gopalan, Princewill Okoroafor, Prasad Raghavendra, Abhishek Sherry, and Mihir Singhal. Omnipredictors for regression and the approximate rank of convex functions. In _The Thirty_
_Seventh Annual Conference on Learning Theory_, pp. 2027–2070. PMLR, 2024b.


Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural
networks. In _International conference on machine learning_, pp. 1321–1330. PMLR, 2017.


Chirag Gupta and Aaditya Ramdas. Top-label calibration and multiclass-to-binary reductions. In
_International Conference on Learning Representations_ . OpenReview, 2022.


Lars Peter Hansen and Thomas J Sargent. Robust control and model uncertainty. _American_ _Eco-_
_nomic Review_, 91(2):60–66, 2001.


Ursula H´ebert-Johnson, Michael Kim, Omer Reingold, and Guy Rothblum. Multicalibration: Calibration for the (computationally-identifiable) masses. In _International_ _Conference_ _on_ _Machine_
_Learning_, pp. 1939–1948. PMLR, 2018.


Lunjia Hu and Yifan Wu. Predict to minimize swap regret for all payoff-bounded tasks. In _65th_
_IEEE Annual Symposium on Foundations of Computer Science,_ _FOCS 2024,_ _Chicago, IL, USA,_
_October 27-30, 2024_, pp. 244–263. IEEE, 2024.


Shayan Kiyani, George Pappas, Aaron Roth, and Hamed Hassani. Decision theoretic foundations
for conformal prediction: Optimal uncertainty quantification for risk-averse agents, 2025. URL
[https://arxiv.org/abs/2502.02561.](https://arxiv.org/abs/2502.02561)


11


Bobby Kleinberg, Renato Paes Leme, Jon Schneider, and Yifeng Teng. U-calibration: Forecasting
for an unknown agent. In Gergely Neu and Lorenzo Rosasco (eds.), _The_ _Thirty_ _Sixth_ _Annual_
_Conference on Learning Theory, COLT 2023, 12-15 July 2023, Bangalore, India_, volume 195 of
_Proceedings of Machine Learning Research_, pp. 5143–5145. PMLR, 2023.


Daniel Kuhn, Peyman Mohajerin Esfahani, Viet Anh Nguyen, and Soroosh Shafieezadeh-Abadeh.
Wasserstein distributionally robust optimization: Theory and applications in machine learning. In
_Operations research & management science in the age of analytics_, pp. 130–166. Informs, 2019.


Meelis Kull, Miquel Perello Nieto, Markus K¨angsepp, Telmo Silva Filho, Hao Song, and Peter
Flach. Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with
dirichlet calibration. _Advances in neural information processing systems_, 32, 2019.


Jiuyao Lu, Aaron Roth, and Mirah Shi. Sample efficient omniprediction and downstream swap regret
for non-linear losses. In Nika Haghtalab and Ankur Moitra (eds.), _The_ _Thirty_ _Eighth_ _Annual_
_Conference_ _on_ _Learning_ _Theory,_ _30-4_ _July_ _2025,_ _Lyon,_ _France_, volume 291 of _Proceedings_ _of_
_Machine Learning Research_, pp. 3829–3878. PMLR, 2025. [URL https://proceedings.](https://proceedings.mlr.press/v291/lu25b.html)
[mlr.press/v291/lu25b.html.](https://proceedings.mlr.press/v291/lu25b.html)


Charles F Manski. Identification problems and decisions under ambiguity. _Journal of Econometrics_,
95(2):415–442, 2000.


Charles F Manski. Statistical treatment rules for heterogeneous populations. _Econometrica_, 72(4):
1221–1246, 2004.


Charles F Manski. Choosing treatment policies under ambiguity. _Annual Review of Economics_, 3:
25–49, 2011.


Charles F Manski and Aleksey Tetenov. Admissible treatment rules for a risk-averse planner. _Econo-_
_metrica_, 75(3):715–752, 2007.


Georgy Noarov, Ramya Ramalingam, Aaron Roth, and Stephan Xie. High-dimensional prediction
for sequential decision making. _arXiv preprint arXiv:2310.17651_, 2023.


Princewill Okoroafor, Robert Kleinberg, and Michael P Kim. Near-optimal algorithms for omniprediction. _arXiv preprint arXiv:2501.17205_, 2025.


R Kelley Pace and Ronald Barry. Sparse spatial autoregressions. _Statistics & Probability Letters_, 33
(3):291–297, 1997.


Benjamin Plaut, Nguyen X Khanh, and Tu Trinh. Probabilities of chat llms are miscalibrated but
still predict correctness on multiple-choice q&a. _arXiv preprint arXiv:2402.13213_, 2024.


Aaron Roth. Uncertain: Modern topics in uncertainty estimation. _Lecture Notes_, 11:30–31, 2022.


Aaron Roth and Mirah Shi. Forecasting for swap regret for all downstream agents. In Dirk Bergemann, Robert Kleinberg, and Daniela Sab´an (eds.), _Proceedings of the 25th ACM Conference on_
_Economics_ _and_ _Computation,_ _EC_ _2024,_ _New_ _Haven,_ _CT,_ _USA,_ _July_ _8-11,_ _2024_, pp. 466–488.
ACM, 2024.


Guy N Rothblum and Gal Yona. Decision-making under miscalibration. In _14th Innovations in The-_
_oretical_ _Computer_ _Science_ _Conference,_ _ITCS_ _2023_, pp. 92. Schloss Dagstuhl-Leibniz-Zentrum
fur Informatik GmbH, Dagstuhl Publishing, 2023.


Abraham Wald. Statistical decision functions. In _Breakthroughs_ _in_ _Statistics:_ _Foundations_ _and_
_Basic Theory_, pp. 342–357. Springer, 1950.


Shengjia Zhao, Michael Kim, Roshni Sahoo, Tengyu Ma, and Stefano Ermon. Calibrating predictions to decisions: A novel approach to multi-class calibration. _Advances in Neural Information_
_Processing Systems_, 34:22313–22324, 2021.


12


# **Appendix**


A MISSING PROOFS FROM THE MAIN BODY


**Proof of Theorem 3.1**


_Proof._ We begin from the robust formulation


        -         - ��
max _u_ _a_ ( _f_ ( _X_ )) _,_ _q_ ( _f_ ( _X_ )) _,_ (6)
_a_ ( _·_ ): _X→A_ [min] _q∈Q_ [E]


where _A_ _⊂_ R _[m]_ is compact, _u_ ( _·, ·_ ) is linear in its second component, _Q_ is the nonempty, convex,
and compact set of measurable maps _q_ : [0 _,_ 1] _[d]_ _→_ [0 _,_ 1] _[d]_ satisfying the linear moment equalities
in equation 4, and _a_ ( _·_ ) ranges over measurable policies with values in _A_ . The mapping ( _a, q_ ) _�→_
E[ _u_ ( _a_ ( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ )))] is convex in _q_ (since _u_ ( _a, ·_ ) is linear, hence convex, in _y_ and expectation
preserves convexity), concave in _a_ (as a pointwise maximum over linear functionals in _a_ on the
compact set _A_ ). Hence, by Sion’s minimax theorem,� - - 


      -       -       -       max _u_ ( _a_ ( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ ))) = min _u_ ( _a_ ( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ ))) _._
_a_ ( _·_ ) [min] _q∈Q_ [E] _q∈Q_ [max] _a_ ( _·_ ) [E]


Fix any _q_ _∈Q_ . The inner maximization over policies separates pointwise in _v_ = _f_ ( _x_ ), yielding the
value function

                      -                       - ��                       -                       - ��
val( _p_ ) ≜ max and max _u_ _a_ ( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ )) = E val _q_ ( _f_ ( _X_ )) _._
_a∈A_ _[u]_ [(] _[a, p]_ [)] _a_ ( _·_ ) [E]


Therefore the robust value equals the convex adversarial problem

                      -                      - ��
min val _q_ ( _f_ ( _X_ )) _,_ (7)
_q∈Q_ [E]

which will be analyzed via Lagrangian duality below.

Introduce vector Lagrange multipliers _λi_ _∈_ R _[d]_ for the _d_ -dimensional equalities in equation 4, and
let _λ_ = ( _λ_ 1 _, . . ., λk_ ). Define


_s_ ( _v_ ) ≜


The Lagrangian of equation 7 is


- _k_

_hi_ ( _v_ ) _λi_ _∈_ R _[d]_ _,_ _v_ _∈_ [0 _,_ 1] _[d]_ _._
_i_ =1


       -       - ��
_L_ ( _q, λ_ ) = E val _q_ ( _f_ ( _X_ )) +


- _k_ - - �� ��

_i_ =1 _λi ·_ E _hi_ _f_ ( _X_ ) _q_ ( _f_ ( _X_ )) _−_ _f_ ( _X_ ) _._


By linearity of expectation,

           -            -            -            -            -            -            - [�]
_L_ ( _q, λ_ ) = E val _q_ ( _f_ ( _X_ )) + _q_ ( _f_ ( _X_ )) _· s_ _f_ ( _X_ ) _−_ _f_ ( _X_ ) _· s_ _f_ ( _X_ ) _._

The dual function is obtained by minimizing _L_ ( _q, λ_ ) over measurable _q_ : [0 _,_ 1] _[d]_ _→_ [0 _,_ 1] _[d]_ . Since the
integrand depends on _q_ only through _q_ ( _f_ ( _X_ )), the infimum can be taken _pointwise_ in the forecast
value _v_ = _f_ ( _X_ ):

              -               -               - �� [�]               -               - ��
_G_ ( _λ_ ) = inf [=] [E] inf val( _p_ ) + _p_ _s_ _f_ ( _X_ ) E _f_ ( _X_ ) _s_ _f_ ( _X_ ) _._
_q_ _[L]_ [(] _[q, λ]_ [)] _p_ [0 _,_ 1] _[d]_ _·_ _−_ _·_
_∈_

The primal problem equation 7 is convex (convex objective, affine constraints) and feasible (e.g.,
_q_ ( _v_ ) = _v_ ), thereby strong duality holds. Hence,

                 -                  - ��
min val _q_ ( _f_ ( _X_ )) = max
_q∈Q_ [E] _λ∈_ (R _[d]_ ) _[k][ G]_ [(] _[λ]_ [)] _[,]_

and there exists a maximizing multiplier _λ_ _[⋆]_ . Define


_s_ _[⋆]_ ( _v_ ) ≜


- _k_

_hi_ ( _v_ ) _λ_ _[⋆]_ _i_
_i_ =1 _[∈]_ [R] _[d][.]_


13


By the definition of _G_ ( _λ_ ) and strong duality, any primal optimizer _q_ _[⋆]_ _∈Q_ must minimize the
Lagrangian at _λ_ _[⋆]_ . Since the dependence on _q_ is only through _q_ ( _f_ ( _X_ )), this yields the pointwise
characterization, for _v_ = _f_ ( _x_ ) almost surely,


_q_ _[⋆]_ ( _v_ ) arg min
_∈_ _p_ [0 _,_ 1] _[d]_
_∈_


- val( _p_ ) + _p · s_ _[⋆]_ ( _v_ ) _._


With _q_ _[⋆]_ fixed, define the policy


           -            _a_ robust( _v_ ) arg max _a, q_ _[⋆]_ ( _v_ ) _._
_∈_ _a_ _[u]_
_∈A_


Then, by the definition of val and the construction of _q_ _[⋆]_,

        -        - ��        -        - ��        -        - ��
max _u_ _a_ ( _f_ ( _X_ )) _, q_ _[⋆]_ ( _f_ ( _X_ )) = E val _q_ _[⋆]_ ( _f_ ( _X_ )) = min val _q_ ( _f_ ( _X_ )) _,_
_a_ ( _·_ ) [E] _q∈Q_ [E]

which shows that ( _a_ robust _, q_ _[⋆]_ ) is a saddle point of equation 6. In particular, _a_ robust is optimal for
the outer maximization, and _q_ _[⋆]_ is worst–case optimal for the inner minimization, with _q_ _[⋆]_ characterized pointwise by the minimization problem above and determined by the dual multiplier _λ_ _[⋆]_ . This
matches the statement of Theorem 3.1 and completes the proof.


**Proof of Theorem 4.1** :


_Proof._ We use the reduction

              -               -               -               max _u_ ( _a_ ( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ ))) = min val( _q_ ( _f_ ( _X_ ))) _,_
_a_ ( _·_ ) [min] _q∈Q_ [E] _q∈Q_ [E]


established in the proof of Theorem 3.1. Fix the decision regions


_Ra_ = _v_ [0 _,_ 1] _[d]_ : _u_ ( _a, v_ ) _u_ ( _a_ _[′]_ _, v_ ) _a_ _[′]_ _,_
_{_ _∈_ _≥_ _∀_ _∈A }_

each convex. Under dec = **1** _Ra_ : _a_, admissible _q_ satisfy
_H_ _{_ _∈A}_

              -              E **1** _Ra_ ( _f_ ( _X_ )) _{q_ ( _f_ ( _X_ )) _−_ _f_ ( _X_ ) _}_ = 0 _∀a,_


equivalently (whenever P( _f_ ( _X_ ) _∈_ _Ra_ ) _>_ 0),


E[ _q_ ( _f_ ( _X_ )) _| f_ ( _X_ ) _∈_ _Ra_ ] = E[ _f_ ( _X_ ) _| f_ ( _X_ ) _∈_ _Ra_ ] =: _µa_ _∈_ _Ra._


By Jensen’s inequality (convexity of val), for any _q_ _∈Q_ and any _a_,


E�val� _q_ ( _f_ ( _X_ ))��� _f_ ( _X_ ) _∈_ _Ra_ - _≥_ val( _µa_ ) _._


Define the piecewise-constant _q_ ¯( _v_ ) = [�]


Define the piecewise-constant _q_ ¯( _v_ ) = _a_ _[µ][a]_ **[ 1]** _[R][a]_ [(] _[v]_ [)][.] [Then] _[q]_ [¯] _[∈Q]_ [ and,] [conditionally on] _[ f]_ [(] _[X]_ [)] _[∈]_

_Ra_, we have _q_ ¯( _f_ ( _X_ )) = _µa_ a.s., hence the bound is attained:


 -  - ��  -  -  - ��
E val _q_ ¯( _f_ ( _X_ )) = P( _f_ ( _X_ ) _∈_ _Ra_ ) val( _µa_ ) _≤_ E val _q_ ( _f_ ( _X_ )) _∀q_ _∈Q._

_a_


Thus a worst-case belief is _q_ _[⋆]_ = _q_ ¯, region-wise constant with _q_ _[⋆]_ ( _v_ ) = _µa_ on _Ra_ .

Finally, since _µa_ _Ra_, by definition of _Ra_ we have _u_ ( _a, µa_ ) _u_ ( _a_ _[′]_ _, µa_ ) for all _a_ _[′]_, so _a_ is a best
_∈_ _≥_
response to _µa_ . Therefore the robust action at _v_ _∈_ _Ra_ is

_a_ robust( _v_ ) arg max
_∈_ _a_ _[′]_ _[u]_ [(] _[a][′][, q][⋆]_ [(] _[v]_ [)) = arg max] _a_ _[′]_ _[u]_ [(] _[a][′][, µ][a]_ [)] _[ ∋]_ _[a,]_


which coincides (a.e.) with the plug-in best response to _v_ . This proves Theorem 4.1.


**Proof of Theorem 4.2:**


Recall val( _p_ ) = max _a_ _u_ ( _a, p_ ) and the reduction
_∈A_

              -               - ��               -               - ��
max _u_ _a_ ( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ )) = min val _q_ ( _f_ ( _X_ )) _,_
_a_ ( _·_ ) _q_ [min] _∈QH_ [E] _q∈QH_ [E]


14


established earlier in the proof of Theorem 3.1. Moreover, the identity map _q_ id( _v_ ) = _v_ always lies
in _QH_ (the perfect forecaster is consistent with every _H_ -calibration constraint), so for any policy
_a_ ( ),

             -              - ��              -              - ��

_·_
min _u_ _a_ ( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ )) E _u_ _a_ ( _f_ ( _X_ )) _, f_ ( _X_ ) _._ (8)
_q∈QH_ [E] _≤_

Let _a_ BR( _v_ ) arg max _a_ _u_ ( _a, v_ ) be a plug-in best response. [1] We show that, assuming contains
_∈_ _∈A_ _H_
the decision-calibration tests _{_ **1** _Ra_ _}a∈A_,


 -  - ��  -  - ��
E _u_ _a_ BR( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ )) = E _u_ _a_ BR( _f_ ( _X_ )) _, f_ ( _X_ ) _q_ _._ (9)
_∀_ _∈QH_


Write _µa_ := E[ _f_ ( _X_ ) _|_ _f_ ( _X_ ) _∈_ _Ra_ ] whenever P( _f_ ( _X_ ) _∈_ _Ra_ ) _>_ 0 (if P( _f_ ( _X_ ) _∈_ _Ra_ ) = 0, any
choice of _µa_ is harmless since the corresponding terms vanish). Then


 -  - ��  E _u_ _a_ BR( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ )) =


- - - - 
E _u_ _a, q_ ( _f_ ( _X_ )) **1** _f_ ( _X_ ) _Ra_
_{_ _∈_ _}_
_a∈A_


( _i_ ) =


P( _f_ ( _X_ ) _∈_ _Ra_ ) _u_ ( _a,_ E[ _q_ ( _f_ ( _X_ )) _| f_ ( _X_ ) _∈_ _Ra_ ])
_a∈A_


( _ii_ ) =


P( _f_ ( _X_ ) _∈_ _Ra_ ) _u_ ( _a,_ E[ _f_ ( _X_ ) _| f_ ( _X_ ) _∈_ _Ra_ ])
_a∈A_


 =


P( _f_ ( _X_ ) _∈_ _Ra_ ) _u_ ( _a, µa_ )
_a∈A_


( _iii_ ) =


- - - - 
E _u_ _a, f_ ( _X_ ) **1** _f_ ( _X_ ) _Ra_
_{_ _∈_ _}_

 - _a∈A_ - ��


  -   - ��
= E _u_ _a_ BR( _f_ ( _X_ )) _, f_ ( _X_ ) _._


Here: ( _i_ ) uses that _u_ ( _a, ·_ ) is linear in its second argument, so


E[ _u_ ( _a, q_ ( _f_ ( _X_ ))) _| f_ ( _X_ ) _∈_ _Ra_ ] = _u_ ( _a,_ E[ _q_ ( _f_ ( _X_ )) _| f_ ( _X_ ) _∈_ _Ra_ ]) _,_


( _ii_ ) uses the decision-calibration equalities E[ **1** _Ra_ ( _f_ ( _X_ )) _{q_ ( _f_ ( _X_ )) _−_ _f_ ( _X_ ) _}_ ] = 0, equivalently
E[ _q_ ( _f_ ( _X_ )) _f_ ( _X_ ) _Ra_ ] = E[ _f_ (� _X_ ) _f_ ( _X_ ) _Ra_ ] = _µa_ whenever� P( _f_ ( _X_ ) _Ra_ ) _>_ 0; and ( _iii_ )
_|_ _∈_ _|_ _∈_ _∈_
again uses linearity: _u_ ( _a, µa_ ) = _u_ _a,_ E[ _f_ ( _X_ ) _| f_ ( _X_ ) _∈_ _Ra_ ] = E[ _u_ ( _a, f_ ( _X_ )) _| f_ ( _X_ ) _∈_ _Ra_ ].


Combining equation 8, the optimality of best response on the _perceived_ outcomes,

      -      - ��      -      - ��
E _u_ _a_ ( _f_ ( _X_ )) _, f_ ( _X_ ) _≤_ E _u_ _a_ BR( _f_ ( _X_ )) _, f_ ( _X_ ) for all policies _a_ ( _·_ ) _,_


and the invariance equation 9, we obtain the minimax dominance

    -     - ��     -     - ��     -     - ��
min _u_ _a_ BR( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ )) = E _u_ _a_ BR( _f_ ( _X_ )) _, f_ ( _X_ ) min _u_ _a_ ( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ )) _,_
_q∈QH_ [E] _≥_ _q∈QH_ [E]

for every forecast-based policy _a_ ( _·_ ). Hence the plug-in best response is minimax optimal under any
_H_ that contains the decision-calibration tests, as claimed.

**Proof of Proposition 4.4** :


_Proof._ Assume E _∥zϕ_ ( _X_ ) _∥_ 2 [2] _[<]_ _[∞]_ [and][ E] _[∥][Y][ ∥]_ 2 [2] _[<]_ _[∞]_ [so that all derivatives and expectations below]
are well-defined and we may interchange expectation and differentiation by dominated convergence.
Write _z_ := _zϕ_ ( _X_ ) _∈_ R _[k]_ and _f_ := _fθ_ ( _X_ ) = _Wz_ _∈_ R _[d]_ . The squared-loss risk is

                -                -                -                _L_ ( _θ_ ) = 12 [E] _∥f_ _−_ _Y ∥_ 2 [2] = 12 [E] ( _Wz −_ _Y_ ) _[⊤]_ ( _Wz −_ _Y_ ) _._

For the linear head _W_ _∈_ R _[d][×]_ - _[k]_, the gradient with respect to� _W_ satisfies the standard identity

_∇W_ 12 _[∥][Wz][ −]_ _[Y][ ∥]_ 2 [2] = ( _Wz −_ _Y_ ) _z_ _[⊤]_ _∈_ R _[d][×][k]_ _._

Taking expectation and interchanging _∇_ with E yields�

_∇W L_ ( _θ_ ) = E ( _f_ _−_ _Y_ ) _z_ _[⊤]_ [�] _._

1Fix any deterministic tie-breaking so that _a_ BR and the regions _Ra_ = _{v_ : _a_ BR( _v_ ) = _a}_ are measurable.


15


At a first-order stationary point (in particular, when the gradient with respect to _W_ vanishes) we
have E ( _f_ _Y_ ) _z_ _[⊤]_ [�] = 0 _d_ _k._

_−_ _×_
Transposing gives

         -         E _z_ ( _f_ _Y_ ) _[⊤]_ [�] = 0 _k_ _d_ E _z_ ( _Y_ _f_ ) _[⊤]_ [�] = 0 _k_ _d,_

_−_ _×_ _⇐⇒_ _−_ _×_
which is the first claimed moment identity.


For the second identity, observe that _f_ = _Wz_, hence

    -     -     E _f_ ( _Y_ _f_ ) _[⊤]_ [�] = E _Wz_ ( _Y_ _f_ ) _[⊤]_ [�] = _W_ E _z_ ( _Y_ _f_ ) _[⊤]_ [�] = _W_ 0 _k_ _d_ = 0 _d_ _d._

_−_ _−_ _−_ _×_ _×_

Therefore both E[ _zϕ_ ( _X_ ) ( _Y_ _−_ _fθ_ ( _X_ )) _[⊤]_ ] = 0 and E[ _fθ_ ( _X_ ) ( _Y_ _−_ _fθ_ ( _X_ )) _[⊤]_ ] = 0 hold. In particular,
for each coordinate _j_ = 1 _, . . ., d_, E[ _e_ _[⊤]_ _j_ _[f][θ]_ [(] _[X]_ [) (] _[Y][ −][f][θ]_ [(] _[X]_ [))] _[⊤]_ [] = 0][ and][ E][[] _[ z][ϕ]_ [(] _[X]_ [)] _[ e]_ _j_ _[⊤]_ [(] _[Y][ −][f][θ]_ [(] _[X]_ [))] =]
0 _,_ so _fθ_ is _H_ -calibrated for _H_ = _{hj_ ( _v_ ) = _e_ _[⊤]_ _j_ _[v]_ [:] _[j]_ [=] [1] _[, . . ., d][}]_ [and] [for] [any] [linear] [combinati][on]
thereof. This proves the proposition.


**Proof of Proposition 4.5:**


_Proof._ By the reduction established earlier (see the proof of Theorem 3.1), the robust problem

                     -                      max _u_ ( _a_ ( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ )))
_a_ ( _·_ ) [min] _q∈Q_ [E]


with linear utilities and finite is equivalent to the convex program
_A_            -            - ��
min val _q_ ( _f_ ( _X_ )) _,_ val( _p_ ) := max
_q∈Q_ [E] _a∈A_ _[u]_ [(] _[a, p]_ [)] _[,]_


subject to the _H_ bin-calibration constraints� E **1** _f_ ( _X_ ) _Bj_ ( _q_ ( _f_ ( _X_ )) _f_ ( _X_ )) = 0 _,_ _j_ = 1 _, . . ., J._
_{_ _∈_ _}_ _−_


Write _Ej_ := _{f_ ( _X_ ) _∈_ _Bj}_ and assume P( _Ej_ ) _>_ 0 (bins with zero probability are immaterial).
Then the constraints are equivalent to
E[ _q_ ( _f_ ( _X_ )) _| Ej_ ] = E[ _f_ ( _X_ ) _| Ej_ ] =: _mj,_ _j_ = 1 _, . . ., J._
Because _u_ ( _a, ·_ ) is linear in the outcome, val is the pointwise maximum of linear maps and hence
convex. Decomposing by bins and applying Jensen’s inequality gives, for any feasible _q_,


            -            - ��
E val _q_ ( _f_ ( _X_ )) =


_≥_


=


Define the piecewise-constant candidate


- _J_ P( _Ej_ ) E�val� _q_ ( _f_ ( _X_ ))��� _Ej_ 
_j_ =1


- _J_

P( _Ej_ ) val(E[ _q_ ( _f_ ( _X_ )) _| Ej_ ])
_j_ =1


- _J_

P( _Ej_ ) val( _mj_ ) _._
_j_ =1


_q_ ¯( _v_ ) :=


- _J_

_mj_ **1** _Bj_ ( _v_ ) _._
_j_ =1


Then _q_ ¯ is feasible, since for each _j_,

         -          -          -          E **1** _Ej_ (¯ _q_ ( _f_ ( _X_ )) _−_ _f_ ( _X_ )) = P( _Ej_ ) _mj_ _−_ E[ _f_ ( _X_ ) _| Ej_ ] = 0 _,_
and it attains the Jensen lower bound because _q_ ¯( _f_ ( _X_ )) = _mj_ almost surely on _Ej_ :


E�val� _q_ ¯( _f_ ( _X_ ))��� _Ej_               - = val( _mj_ ) _._
Therefore _q_ ¯ is an optimizer, and any minimizer _q_ _[⋆]_ can be chosen (a.e.) piecewise constant with
_q_ _[⋆]_ ( _v_ ) = _mj_ for _v_ _Bj._
_∈_


Finally, fixing such a _q_ _[⋆]_, the robust action at forecast� _v_ - _∈_ _Bj_ solves


           -           -           -           _a_ robust( _v_ ) arg max _a, q_ _[⋆]_ ( _v_ ) = arg max _a, mj_ _,_
_∈_ _a_ _[u]_ _a_ _[u]_
_∈A_ _∈A_


which depends only on the bin index, i.e., it is the best response to the bin mean. This proves the
claim.


16


B APPROXIMATE _H_ -CALIBRATION: STABILITY UNDER _ε_ -SLACK


This appendix extends the main results to the practically relevant regime in which _H_ -calibration
holds only approximately. Concretely, we relax each linear calibration equality in equation 3 to an
_ℓ_ 2–ball of radius _ε_ . Throughout, we retain the standing assumptions of the main text: utilities are
linear in the outcome, so there exist _ra_ R _[d]_ _,_ _ca_ R _a_ with
_{_ _∈_ _∈_ _}_ _∈A_

_u_ ( _a, p_ ) = _ra_ _p_ + _ca_ = val( _p_ ) := max [is convex and] _[ L]_ [-Lipschitz w.r.t.] _[ ∥· ∥]_ [2][,]

_·_ _⇒_ _a_ _[u]_ [(] _[a, p]_ [)]
_∈A_

_f_ where: _X_ _L→_ := max[0 _,_ 1] _[d]_ denotes the given forecaster. _a∈A ∥ra∥_ 2. We write expectations over ( _X, Y_ ) distributed as in the main body, and


**Approximate** **calibration** **constraints.** Let = span _h_ 1 _, . . ., hk_ with measurable _hi_ :
_H_ _{_ _}_

[0 _,_ 1] _[d]_ _→_ R bounded by _|hi_ ( _v_ ) _|_ _≤_ 1. For a candidate conditional expectation _q_ : [0 _,_ 1] _[d]_ _→_ [0 _,_ 1] _[d]_,
define the (vector) calibration moments


      -       - ��
_mi_ ( _q_ ) := E _hi_ ( _f_ ( _X_ )) _q_ ( _f_ ( _X_ )) _−_ _f_ ( _X_ ) _∈_ R _[d]_ _,_ _i_ = 1 _, . . ., k._


We say _q_ is _ε–approximately_ _-calibrated_ if _mi_ ( _q_ ) 2 _ε_ for all _i_ . The corresponding ambiguity
_H_ _∥_ _∥_ _≤_
set and robust value are


   -   -   -   - ��
_ε_ := _q_ : [0 _,_ 1] _[d]_ [0 _,_ 1] _[d]_ : _mi_ ( _q_ ) 2 _ε,_ _i_ = 1 _, . . ., k_ _,_ _Vε_ := min val _q_ ( _f_ ( _X_ )) _._
_Q_ _→_ _∥_ _∥_ _≤_ _q∈Qε_ [E]


For reference, the exact-calibration value is _V_ 0 = min _q∈Q_ E[val( _q_ ( _f_ ( _X_ )))], where _Q_ is the
equality-based set from equation 4.


**Roadmap.** We first show a _dual penalty_ bound: moving from exact to _ε_ –approximate constraints
subtracts an explicit _ℓ_ 2–norm penalty from the exact dual objective, yielding two-sided value bounds
and a linear-in- _ε_ degradation guarantee. We then quantify the robustness of _decision_ _calibration_ :
even under _ε_ –slack, the plug-in best response is _O_ ( _mLε_ )–minimax optimal (with _m_ := _|A|_ ). Finally,
for _bin-wise_ (histogram) calibration with _ε_ –slack, we obtain piecewise-constant worst-case beliefs
and tight value bounds, recovering the exact structural picture up to _O_ ( _JLε_ ) terms when there are
_J_ bins.


**Policy characterization under** _ε_ **–slack.** The primal inner problem remains convex and pointwise
in _v_ = _f_ ( _x_ ), while the dual acquires the norm penalty from Theorem B.1. Consequently, the
optimal robust policy admits the same form as in the exact case, with the unique change that the
dual multiplier solves a penalized maximization.
**Theorem B.1** ( _ε_ –robust policy via penalized dual) **.** _Let_ = span _h_ 1 _, . . ., hk_ _and define G_ ( _λ_ ) _as_
_H_ _{_ _}_
_in the main text._ _Let_


- _k_ 
_λi_ 2 _,_ _sλ⋆ε_ ( _v_ ) :=
_∥_ _∥_
_i_ =1


- _k_

_hi_ ( _v_ ) _λ_ _[⋆]_ _ε,i_ _[.]_
_i_ =1


_λ_ _[⋆]_ _ε_ _λ_ [max] (R _[d]_ ) _[k]_

_[∈]_ [arg] _∈_


_G_ ( _λ_ ) _−_ _ε_


_Then_ _there_ _exists_ _a_ _worst-case_ _belief_ _qε_ _[⋆]_ [:] [[0] _[,]_ [ 1]] _[d]_ _[→]_ [[0] _[,]_ [ 1]] _[d]_ _[such]_ _[that]_ _[for]_ _[almost]_ _[every]_ _[forecast]_
_v_ = _f_ ( _x_ ) _,_


_qε_ _[⋆]_ [(] _[v]_ [)] [min]

_[∈]_ [arg] _p_ [0 _,_ 1] _[d]_
_∈_


- val( _p_ ) + _p_ _sλ⋆ε_ ( _v_ ) _,_ val( _p_ ) = max

_·_ _a_ _[u]_ [(] _[a, p]_ [)] _[.]_
_∈A_


_The ε–robust action is the best response to qε_ _[⋆]_ [(] _[v]_ [)] _[:]_

                        -                        _a_ _[⋆]_ _ε_ [(] _[v]_ [)] _a,_ _qε_ _[⋆]_ [(] _[v]_ [)] _._

_[∈]_ [arg max] _a_ _[u]_
_∈A_


_Proof of Theorem B.1._ Recall the robust formulation under linear utilities and forecast–based policies reduces to the adversarial convex program

               -               - ��
min val _q_ ( _f_ ( _X_ )) _,_ val( _p_ ) := max
_q∈Qε_ [E] _a∈A_ _[u]_ [(] _[a, p]_ [)] _[,]_


with the _ε ε_ =–approximate� _q_ : [0 _,_ 1] _[d]_ _H_ –calibration set[0 _,_ 1] _[d]_ : ��E� _hi_ ( _f_ ( _X_ )) _q_ ( _f_ ( _X_ )) _f_ ( _X_ ) ���2 _[i]_ [ = 1] _[, . . ., k]_ - _._

_Q_ _→_ _{_ _−_ _}_ _[≤]_ _[ε,]_


17


Introduce slack vectors� _si_ _∈_ R _[d]_ (one per test) so that each constraint is rewritten as the� _equality_


 - E _hi_ ( _f_ ( _X_ )) _{q_ ( _f_ ( _X_ )) _−_ _f_ ( _X_ ) _}_ = _si_ with _∥si∥_ 2 _≤_ _ε_ ( _i_ = 1 _, . . ., k_ ) _._


Let _λi_ _∈_ R _[d]_ be the Lagrange multipliers for these equalities and set _sλ_ ( _v_ ) := [�] _[k]_ _i_ =1 _[h][i]_ [(] _[v]_ [)] _[λ][i]_ [.] [The]
Lagrangian reads


      -       -       - [�]
_L_ ( _q, s_ ; _λ_ ) = E val _q_ ( _f_ ( _X_ )) +


- _k_ - - - 
_λi_ _·_ E _hi_ ( _f_ ( _X_ )) _{q_ ( _f_ ( _X_ )) _−_ _f_ ( _X_ ) _}_ _−_ _si_ _._
_i_ =1


Minimizing _L_ over the slacks _si_ subject to _∥si∥_ 2 _≤_ _ε_ contributes the support function of the _ℓ_ 2–ball,

inf sup ( _λi_ _si_ ) = _ε_ _λi_ 2 _._
_∥si∥_ 2 _≤ε_ [(] _[−][λ][i]_ _[·][ s][i]_ [) =] _[ −]_ _∥si∥_ 2 _≤ε_ _·_ _−_ _∥_ _∥_

Minimizing the remaining part over _q_ depends on _q_ only through _q_ ( _f_ ( _X_ )) and yields, pointwise in
_v_ = _f_ ( _X_ ),


         -          -          - [�]          -          inf inf val( _p_ ) + _p_ _sλ_ ( _f_ ( _X_ )) E _f_ ( _X_ ) _sλ_ ( _f_ ( _X_ )) _ε_
_q_ _[L]_ [(] _[q, s]_ [;] _[ λ]_ [) =][ E] _p_ [0 _,_ 1] _[d]_ _·_ _−_ _·_ _−_
_∈_


Therefore the dual function is


- _k_

_λi_ 2 _._
_∥_ _∥_
_i_ =1


      -      -      -      _Gε_ ( _λ_ ) = E min E _f_ ( _X_ ) _sλ_ ( _f_ ( _X_ ))
_p_ [0 _,_ 1] _[d][{]_ [val(] _[p]_ [) +] _[ p][·][ s][λ]_ [(] _[f]_ [(] _[X]_ [))] _[}]_ _−_ _·_

~~�~~ _∈_ ~~�~~      - ~~�~~
=: _G_ ( _λ_ )


_−_ _ε_


- _k_

_λi_ 2 _,_
_∥_ _∥_
_i_ =1


i.e., the exact-calibration dual _G_ ( _λ_ ) penalized by _ε_ [�]


_i_

_[∥][λ][i][∥]_ [2][.]


The primal problem is convex (convex objective, affine moment constraints) and feasible (e.g.,
_q_ ( _v_ ) _≡_ _v_ makes all moments 0, which is strictly feasible when _ε_ _>_ 0), so Slater’s condition holds;
hence strong duality holds and a maximizer _λ_ _[⋆]_ of _Gε_ exists:


- _k_ 
_λi_ 2 _._
_∥_ _∥_
_i_ =1


    -    - ��
min val _q_ ( _f_ ( _X_ )) = max
_q∈Qε_ [E] _λ∈_ (R _[d]_ ) _[k]_


_G_ ( _λ_ ) _−_ _ε_


Moreover, comparing with the exact case (which corresponds to _ε_ = 0) gives the two-sided value
bound - - max _G_ ( _λ_ ) _ε_ _λi_ 2 _Vε_ _V_ 0 := max _G_ ( _λ_ ) _,_
_λ_ _−_ _∥_ _∥_ _≤_ _≤_ _λ_


   _λi_ 2 _Vε_ _V_ 0 := max _G_ ( _λ_ ) _,_
_∥_ _∥_ _≤_ _≤_ _λ_
_i_


- _G_ ( _λ_ ) _−_ _ε_


       and 0 _≤_ _V_ 0 _−_ _Vε_ _≤_ _ε_ min _λ∈_ arg max _G_


_i_

_[∥][λ][i][∥]_ [2][.]


_L_ By strong duality, any primal minimizer( _qε_ _[⋆][, λ][⋆]_ [)] _[≤]_ _[L]_ [(] _[q, λ][⋆]_ [)] _[.]_ [The] [first] [inequality] _qε_ _[⋆]_ _[∈Q]_ [implies] _[ε]_ [together with][that] _[q]_ _ε_ _[⋆]_ [minimizes] _[ λ][⋆]_ [forms a saddle point:][the] [Lagrangian] [at] _[L][λ]_ [(] _[⋆][q]_ [,] _ε_ _[⋆][, λ]_ [which][)] _[ ≤]_
(by the pointwise structure above) yields, for almost every forecast _v_ = _f_ ( _x_ ),


- _k_

_hi_ ( _v_ ) _λ_ _[⋆]_ _i_ _[.]_
_i_ =1


_qε_ _[⋆]_ [(] _[v]_ [)] _[ ∈]_ [arg] [min]
_p∈_ [0 _,_ 1] _[d]_


- val( _p_ ) + _p_ _sλ⋆_ ( _v_ ) _,_ _sλ⋆_ ( _v_ ) =

_·_


With _qε_ _[⋆]_ [fixed, the optimal robust action at] _[ v]_ [ solves]

                         -                          _a_ robust _,ε_ ( _v_ ) arg max _a, qε_ _[⋆]_ [(] _[v]_ [)] _,_
_∈_ _a_ _[u]_
_∈A_

i.e., it is the best response to the worst-case belief _qε_ _[⋆]_ [(] _[v]_ [)][.] [This] [is] [the] [same] [best-response] [structure]
as in the exact case, now using the penalized dual optimizer _λ_ _[⋆]_ (cf. the exact characterization in the
main text).


Altogether, we have (i) the dual penalty representation with value bounds, (ii) existence of a dual
maximizer _λ_ _[⋆]_, (iii) the pointwise form of the worst-case belief _qε_ _[⋆]_ [,] [and] [(iv)] [the] [robust] [policy] [as] [a]
pointwise best response to _qε_ _[⋆]_ [, completing the proof.]


18


**Computation.** tive _G_ ( _λ_ ) _ε_ [�] _i_ Algorithmically, the recipe mirrors the exact case: (i) maximize the concave objec- [(e.g., projected/subgradient or bisection in 1D; small-scale mirror descent]

otherwise); _−_ (ii) for _[∥][λ]_ each _[i][∥]_ [2] forecast _v_, compute _qε_ _[⋆]_ [(] _[v]_ [)] [by] [solving] [the] [convex] [problem] [in] _[p]_ [;] [(iii)] [play]
_a_ _[⋆]_ _ε_ [(] _[v]_ [)][ as the best response to] _[ q]_ _ε_ _[⋆]_ [(] _[v]_ [)][.] [For finite] _[ A]_ [ and utilities linear in] _[ p]_ [, step (ii) reduces to checking]
a small finite set of candidates (endpoints and pairwise breakpoints of val), exactly as in the main
text.


**Decision** **tests** **contained** **in** **under** _ε_ **–slack:** **near-optimality** **of** **plug-in.** Let _Ra_ := _v_ :
_H_ _{_
_u_ ( _a, v_ ) _≥_ _u_ ( _a_ _[′]_ _, v_ ) _∀a_ _[′]_ _∈A}_ be the plug-in region for action _a_, and write _Pa_ := P( _f_ ( _X_ ) _∈_ _Ra_ )
(regions with _Pa_ = 0 are ignorable). Assume is a test class that _contains the decision indicators_
_H_
_{_ **1** _Ra_ : _a ∈A}_, with each test bounded by _∥_ **1** _Ra_ _∥∞_ _≤_ 1. We impose _ε_ –approximate _H_ –calibration
in the componentwise sense of Section B, so in particular
��E� **1** _Ra_     - _f_ ( _X_ )�� _q_ ( _f_ ( _X_ )) _f_ ( _X_ )����2 for all _a_ and all _q_ _ε._
_−_ _[≤]_ _[ε,]_ _∈A_ _∈Q_

**Theorem B.2** (Plug-in is _O_ ( _mLε_ )–minimax optimal when decision tests lie in _H_ ) **.** _Let m_ := _|A|_
_εand–approximatelyL_ := max _Ha∈A–calibrated, then the plug-in rule ∥ra∥_ 2 _as_ _above._ _If_ _H_ _contains a_ BR _the_ ( _v_ ) _decision∈_ arg max _indicatorsa u_ ( _a, v_ ) _satisfies, for any{_ **1** _Ra_ _}_ _and_ _f_ _is_
_forecast-based policy a_ ( _·_ ) _,_

        -        -        -        min _u_ ( _a_ BR( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ ))) min _u_ ( _a_ ( _f_ ( _X_ )) _, q_ ( _f_ ( _X_ ))) _m L ε._
_q∈Qε_ [E] _≥_ _q∈Qε_ [E] _−_


_Proof._ Fix any _q_ _ε_ . Decompose by plug-in regions:
_∈Q_


 -  -  E _u_ ( _a_ BR( _f_ ) _, q_ ( _f_ )) =


- - 
_Pa_ E _u_ ( _a, q_ ( _f_ )) _| f_ _∈_ _Ra_ _._
_a∈A_


Since _u_ ( _a, ·_ ) is linear,


          -          E[ _u_ ( _a, q_ ( _f_ )) _| f_ _∈_ _Ra_ ] = _u_ _a,_ E[ _q_ ( _f_ ) _| f_ _∈_ _Ra_ ] _._


Let _µa_ := E[ _f_ _| f_ _∈_ _Ra_ ]. By _L_ –Lipschitzness of _u_ ( _a, ·_ ) and the _ε_ –slack on the indicator test,


��� _u_ - _a,_ E[ _q_ ( _f_ ) _Ra_ ]� _u_ - _a, µa_ - [�] �� _L_ ���E[ _q_ ( _f_ ) _f_ _Ra_ ]��� _[L]_
_|_ _−_ _≤_ _−_ _|_ 2 [=]


��E[ **1** _Ra_ ( _q_ ( _f_ ) _f_ )]��2
_−_ _L_ _[ε]_
_Pa_ _≤_ _P_


_._
_Pa_


Therefore,


    E[ _u_ ( _a_ BR( _f_ ) _, q_ ( _f_ ))] _≥_


   _Pa u_ ( _a, µa_ ) _−_
_a_ _a_


_Pa_ _L_ _[ε]_
_a_ _·_ _P_


= E[ _u_ ( _a_ BR( _f_ ) _, f_ )] _m L ε._
_Pa_ _−_


Let _q_ ˆ _∈_ arg min _q∈Qε_ E[ _u_ ( _a_ BR( _f_ ) _, q_ ( _f_ ))]. Then

min
_q∈Qε_ [E][[] _[u]_ [(] _[a]_ [BR][(] _[f]_ [)] _[, q]_ [(] _[f]_ [))] =][ E][[] _[u]_ [(] _[a]_ [BR][(] _[f]_ [)] _[,]_ [ ˆ] _[q]_ [(] _[f]_ [))]] _[≥]_ [E][[] _[u]_ [(] _[a]_ [BR][(] _[f]_ [)] _[, f]_ [)]] _[ −]_ _[mLε.]_

For any forecast-based policy _a_ ( _·_ ), optimality of the plug-in action on _f_ implies E[ _u_ ( _a_ BR( _f_ ) _, f_ )] _≥_
E[ _u_ ( _a_ ( _f_ ) _, f_ )] _._ Moreover, since _q_ id( _v_ ) _≡_ _v_ is feasible for _Qε_, we have min _q∈Qε_ E[ _u_ ( _a_ ( _f_ ) _, q_ ( _f_ ))] _≤_
E[ _u_ ( _a_ ( _f_ ) _, f_ )] _._ Combining the last three displays yields the claimed inequality.


**Remark.** The proof uses only the _ε_ –slack constraints for the decision indicators **1** _Ra_ ; any
_{_ _}_ _H_
that contains these tests (with per-test slack bounded by _ε_ ) suffices. Thus Theorem B.2 generalizes
both Theorem 4.1 and Theorem 4.2.


**Bin-wise calibration under** _ε_ **–slack:** **value stability and structure.** Let _{Bj}j_ _[J]_ =1 [be a measur-]
able partition of [0 _,_ 1] _[d]_ . Assume _ε–bin-wise calibration_ :
��E� **1** _{f_ ( _X_ ) _∈Bj_ _} {q_ ( _f_ ( _X_ )) _−_ _f_ ( _X_ ) _}_ ���2 _[≤]_ _[ε,]_ _j_ = 1 _, . . ., J._

Write _Ej_ := _{f_ ( _X_ ) _∈_ _Bj}_, _Pj_ := P( _Ej_ ), and _mj_ := E[ _f_ ( _X_ ) _|_ _Ej_ ] (bins with _Pj_ = 0 are
ignorable).


19


**Proposition B.3** (Value stability and piecewise-constant worst-case beliefs) **.** _Under ε–bin-wise cal-_
_ibration,_


- _J_ - 
_Pj_ val( _mj_ ) _J L ε_ min val( _q_ ( _f_ ( _X_ )))
_j_ =1 _−_ _≤_ _q∈Qε_ [E] _≤_


- _J_

_Pj_ val( _mj_ ) _._
_j_ =1


_Moreover, there exists a worst-case (or arbitrarily near-worst-case) belief that is piecewise constant:_
_for each j_ _one can take_

_qε_ _[⋆]_ [(] _[v]_ [) =] _[ p][⋆]_ _j_ min _v_ _Bj_ _(a.e.),_

_[∈]_ [arg] _∥p−mj_ _∥_ 2 _≤ε/Pj_ [val(] _[p]_ [)] _[,]_ _∈_

_and the robust action on Bj_ _best-responds to p_ _[⋆]_ _j_ _[.]_


_Proof._ For any feasible _q_,


- _J_ - 
_Pj_ val E[ _q_ ( _f_ ) _| Ej_ ]
_j_ =1


 -  E val( _q_ ( _f_ )) =


- _J_ - 
_Pj_ E val( _q_ ( _f_ )) _| Ej_ _≥_
_j_ =1


by Jensen since val is convex. The slack constraint implies


��E[ _q_ ( _f_ ) _−_ _f_ _| Ej_ ]��2 [=]


so, using _L_ –Lipschitzness of val,


��E[ **1** _Ej_ ( _q_ ( _f_ ) _f_ )]��2 _ε_
_−_ _,_
_Pj_ _≤_ _Pj_


  -  val E[ _q_ ( _f_ ) _Ej_ ] val( _mj_ ) _L_ _[ε]_
_|_ _≥_ _−_ _P_


Summing over _j_ yields the lower bound E[val( _q_ ( _f_ ))] _≥_ [�]


_._
_Pj_


Summing over _j_ yields the lower bound E[val( _q_ ( _f_ ))] _≥_ _j_ _[P][j]_ [ val(] _[m][j]_ [)] _[ −]_ _[J L ε.]_ [ The upper bound]

holds because _Q ⊆Qε_ and equality is achieved at _ε_ = 0 by the exact bin-wise result.


For structure, fix any feasible _q_ . Replacing _q_ by its conditional mean on each bin,


_q_ ˜( _v_ ) :=


- _J_

E[ _q_ ( _f_ ) _| Ej_ ] **1** _Bj_ ( _v_ ) _,_
_j_ =1


does not increase the objective (by Jensen within each bin) and preserves feasibility (the bin-wise
moments are unchanged). Hence the minimization reduces to choosing, for each bin, a point _pj_
_∈_

[0 _,_ 1] _[d]_ subject to _∥pj −mj∥_ 2 _≤_ _ε/Pj_ to minimize [�] _j_ _[P][j]_ [ val(] _[p][j]_ [)][, which yields the stated piecewise-]

constant form withon each bin is immediate from the definition of _p_ _[⋆]_ _j_ _[∈]_ [arg min] _[∥][p][−][m]_ _j_ _[∥≤][ε/P]_ _j_ val [val(] . _[p]_ [)][.] [The best-response form of the robust acti][on]


20