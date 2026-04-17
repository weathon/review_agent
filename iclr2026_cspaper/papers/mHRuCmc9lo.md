# Robust Decision Making With Partially Cali- Brated Forecasts

Shayan Kiyani , Hamed Hassani, George Pappas & Aaron Roth University of Pennsylvania
{shayank, hassani, pappasg, aaroth}@seas.upenn.edu

## Abstract

Calibration has emerged as a foundational goal in "trustworthy machine learning", in part because of its strong decision theoretic semantics. Independent of the underlying distribution, and independent of the decision maker's utility function, calibration promises that amongst all policies mapping predictions to actions, the uniformly best policy is the one that "trusts the predictions" and acts as if they were correct. But this is true only of *fully calibrated* forecasts, which are tractable to guarantee only for very low dimensional prediction problems. For higher dimensional prediction problems (e.g. when outcomes are multiclass), weaker forms of calibration have been studied that lack these decision theoretic properties. In this paper we study how a conservative decision maker should map predictions endowed with these weaker ("partial") calibration guarantees to actions, in a way that is robust in a minimax sense: i.e. to maximize their expected utility in the worst case over distributions consistent with the calibration guarantees. We characterize their minimax optimal decision rule via a duality argument, and show that surprisingly, "trusting the predictions and acting accordingly" is recovered in this minimax sense by *decision calibration* (and any strictly stronger notion of calibration), a substantially weaker and more tractable condition than full calibration. For calibration guarantees that fall short of decision calibration, the minimax optimal decision rule is still efficiently computable, and we provide an empirical evaluation of a natural one that applies to any regression model solved to optimize squared error.

## 1 Introduction

Machine learning systems are increasingly deployed in high-stakes decision making domains such as healthcare, finance, and law. The predictive power of these models can be extraordinary, but scoring well on predictive error metrics does not directly guarantee that decisions downstream of those predictions will be correct. For predictions to be operationally useful, a decision-maker must be able to treat them as reliable inputs into a downstream decision making policy. This raises two fundamental questions:
On the Model Side: What does it mean for machine learning predictions to be trustworthy in decision-making contexts?

On the Decision Making Side: *Given predictions that satisfy a particular type of*
"trustworthiness", how should the decision maker adapt its actions to the promised guarantees?

On the Model Side: A natural answer is that trustworthy predictions should directly support good decisions as they are. In other words, the decision-maker should be able to reliably best respond to the forecaster's predictions as if they were correct. Formally, let (*X, Y* ) be a pair of random variables drawn from a joint distribution D, where X ∈ X represents the observed features and Y ∈ [0, 1]dis the outcome of interest. Let A denote the action set, and suppose the decision-maker follows a policy a(·) : [0, 1]d → A mapping predictions to actions. Given a predictor f, the decision maker's performance when using a policy a is measured by its expected utility on the underlying distribution:
E(X,Y )∼D[u(a(f(X)), Y )],
1 where u(*a, y*) ∈ R is a utility function. Given a forecaster f : X → [0, 1]d, the *plug-in best response* to a forecast is defined as aBR(f(x)) = arg max a∈A
u(*a, f*(x)). (1)

Thus, a forecaster f is trustworthy if the decision-maker's best-response policy aBR(f(x)) achieves
higher utility than any other policy. When is this the case? The classical answer lies in the notion of *calibration*. Intuitively, a forecaster is calibrated if, whenever it predicts a vector f(x) = v ∈ [0, 1]d, the empirical outcomes are consistent with that prediction. More formally, a forecaster f is said to be *fully calibrated* if for every v ∈ [0, 1]d,
$$\mathbb{E}[Y\mid f(X)=v]=v.$$

## It Is Well Known That Best Responding To Calibrated Forecasts Is The Optimal Decision Policy Among All
Policies That Map Forecasts To Actions (Foster & Vohra, 1997; Kleinberg Et Al., 2023; Noarov Et Al.,
2023; Roth, 2022). However, Achieving Full Calibration Is Extremely Difficult, Both In Theory—The Sample Complexity Of Calibrating An Existing Forecaster Without Harming Its Accuracy Grows Exponentially With The Outcome Dimension D (Gopalan Et Al., 2024A)—And In Practice, Where Empirical Evidence Shows Systematic Deviations From Calibration, Ranging From Neural Networks To Large Language Models (Guo Et Al., 2017; Kull Et Al., 2019; Gupta & Ramdas, 2022; Plaut Et Al., 2024). Thus, Despite The Appealing Link Between Calibration And Trustworthy Ml-Powered Decision-Making, This Connection Quickly Breaks Down In Real-World Applications. On The Decision Making Side: Decision Making From Predictions Admits Two Canonical Extremes. At One End, The Decision Maker *Aggressively Best Responds* To The Forecasts, Acting As If They Were Fully Correct. At The Other End, The Decision Maker Conservatively Plays A Minimax-Safety Strategy, Arg Maxa∈A Miny∈Y U(*A, Y*), Treating The Forecasts As If They Carried No Information About The
Instance. Departing From These Extremes, We Treat A Model F And Its Forecast F(X) As Information That Constrains What The True, Instance-Conditional Outcome Distribution Could Be. In Other Words, After Observing F(X), The Decision Maker Considers The Set Of *Candidate Realities*—Outcome Distributions Consistent With The Forecast And The Available Calibration Guarantees. Intuitively, The "Volume" Of This Set Is Governed By The Strength Of Calibration: Under Full Calibration, The Set Collapses To The Forecast Itself (The Prediction Can Be Treated As Reality, At Least In Expectation), Whereas As Calibration
Weakens, The Set Enlarges. A Principled Decision Rule Should Therefore Tune Its Conservatism To What The Reality Could Be, Consistent With The Provided Guarantees. This Idea, Together With The Fragility Of Full Calibration In Practice, Leads To The Central Question Of This Paper: Can We Derive Optimal
Decision-Making Policies Under Weaker And More Practical Conditions Than Full Calibration? We Answer This Question Affirmatively. We Introduce A Framework Based On *Conservative* Decision Making That Nevertheless Fully Exploits *Partially* Calibrated Forecasts. This Viewpoint Echoes Ideas In Robust Optimization And Control, But It Has Not Been Systematically Developed For Post Hoc Decision Making With Partially Calibrated Machine-Learning Forecasts. 1.1 Our Results

We consider a parameterized family of weighted calibration guarantees that have recently become a popular object of study (Hebert-Johnson et al., 2018; Gopalan et al., 2022). Informally speaking, this ´
family of guarantees constrains the residuals of a predictor f to be uncorrelated with a collection of "test functions" h ∈ H mapping the range of f to the reals. When H consists of all such test functions, we recover full calibration, but many popular variants of calibration (e.g. top label calibration, decision calibration, etc) can be expressed as instances of H-calibration under much smaller/more tractable sets H. Our contributions are as follows:
1. In Section 2 we formalize the following question: given a set of test functions H and a predictor f(x) that is promised to satisfy H-calibration, what decision rule a : [0, 1]d → A,
mapping predictions to actions, will maximize a decision maker's expected utility in the worst case over all joint distributions over X × Y that are consistent with the promise that f is H-calibrated?

2. In Section 3 we answer this question by giving a closed-form for the decision maker's optimal decision rule, in terms of the dual variables of a convex program that can be efficiently computed for any finite H.

3. In Section 4 we instantiate this decision rule for various calibration guarantees of interest.

Of particular note, we find that when H corresponds to the tractable notion of *decision* calibration (Zhao et al., 2021; Noarov et al., 2023), then the optimal decision rule is the best response decision rule aBR, just as it is for (the intractable notion of) full calibration.

In fact, it suffices that H *contains* the decision calibration constraints - any larger set *also* makes best response the optimal decision rule. Thus what could have been a very large hierarchy of minimax optimal decision rules "collapses" to best response at the level of decision calibration. An upshot of this is that a predictor can be simultaneously decision calibrated for many downstream decision makers, and for each of them, best response will be their optimal decision policy in this minimax sense. We also derive the minimax optimal decision rule for a simple "self-orthogonality" calibration condition that will hold for any regression model with a linear final layer trained to optimize squared loss, and hence will be commonly satisfied without any algorithmic intervention.

4. In Section 5 we train a two-layer MLP to minimize squared loss on two regression datasets, and evaluate both the best-response decision rule and the robust decision rule that results from the self-orthogonality condition of squared error regression. We find that, as predicted by our theory, the robust decision rule outperforms the best-response decision rule under calibration-preserving distribution shift, and that the cost of this robustness is mild even under ideal conditions.

## 1.2 Related Work

Rothblum & Yona (2023) consider a setting in which both the outcome and decision maker's action set are binary, and study how a decision maker should act to minimize their worst case regret over distributions such that the predictor has maximum calibration error bounded by α: informally that |E[Y |f(x) = v] − v| ≤ α for all v. The models f they study are (approximately) fully calibrated, which is a reasonable assumption in their setting, since they limit their study to 1-dimensional outcomes. In contrast, our interest is not (just) in quantitative measures of full calibration error, but rather qualitatively weaker calibration guarantees, as even approximate full calibration becomes intractable in high dimensions. A line of recent work (Zhao et al., 2021; Kleinberg et al., 2023; Noarov et al., 2023; Roth & Shi, 2024; Hu & Wu, 2024; Okoroafor et al., 2025) has studied the guarantees that can be given to downstream decision makers who best respond to predictions that have weaker guarantees than full calibration (and which in the cases of Zhao et al. (2021); Noarov et al. (2023); Roth & Shi (2024) can be tractably guaranteed in higher dimensional outcome settings). These guarantees take the form of (external and swap) *regret* bounds, which are qualitatively weaker than the kind of "trustworthiness" promised by full calibration. Informally, regret bounds promise that the decision maker could not have done better by consistently playing a fixed action (or a fixed function remapping their actions to other actions), not that they could not have done better by using a different policy from predictions to actions. We show that even in high dimensions, the tractable "decision calibration" condition given by Zhao et al. (2021) recovers the same "trustworthiness" semantics of full calibration when viewed through our minimax decision making lens.

Analyzing minimax optimal decision policies is a common way of analyzing robust or riskaverse decision making guarantees, with deep roots in economics (Gilboa & Schmeidler, 1989; Hansen & Sargent, 2001; Manski, 2000; 2004; Manski & Tetenov, 2007; Manski, 2011), statistics
(Wald, 1950), and robust optimization (Ben-Tal & Nemirovski, 2002; Kuhn et al., 2019; Duchi & Namkoong, 2021). For example, Carroll (2015) adopts this lens this in the context of contract theory and Kiyani et al. (2025) and Andrews & Chen (2025) do so in the context of conformal prediction. To the best of our knowledge, we are the first to apply this "robust" minimax lens to the problem of partially calibrated high dimensional forecasts.

![3_image_0.png](3_image_0.png)

$$(2)$$
$$({\mathfrak{I}})$$

## 2 Robust Decision Making And H-Calibration

In this Section, we define H-calibration as a flexible relaxation of full calibration and then introduce a framework to derive minimax optimal decision making policies that are designed to act on forecasters guaranteed to satisfy H-calibration. This family of calibration guarantees has been studied extensively in the recent literature on multicalibration and its extensions (Hebert-Johnson et al., ´
2018; Dwork et al., 2021; Gopalan et al., 2022; Deng et al., 2023) - in particular, H-calibration is a special case of what Gopalan et al. (2022) call weighted multicalibration.

$$\mathbb{E}{\big[}\,h(f(X))\cdot(Y-f(X))\,{\big]}=0.$$

H**-Calibration.** Let H be a set of functions h : [0, 1]d → R. A forecaster f is said to be H-
calibrated if for every h ∈ H,
E-h(f(X)) · (Y − f(X)) = 0. (2)
Equivalently, writing q(v) := E[Y | f(X) = v] for the true conditional expectation, H-calibration requires

* [16] M. C.  
$$\mathbb{E}\big[\,h(f(X))\cdot(q(f(X))-f(X))\,\big]=0,\quad\forall h\in{\mathcal{H}}.$$
This definition captures a spectrum of guarantees. When H contains all bounded measurable functions, H-calibration reduces to full calibration - i.e. it requires that f(v) = q(v) := E[Y | f(X) = v] almost surely. For smaller classes H, the requirement is weaker and can be seen as a relaxation of calibration, enforcing consistency only with respect to a restricted set of tests. In the main body of the paper we focus on the H-calibration defined above, but in Appendix B we also discuss scenarios in which only approximate H-calibration is available. Robust Decision Making. Fix an H-calibrated forecaster f. Define the set

$${\mathcal{Q}}\;=\;\Big\{q:[0,1]^{d}\to[0,1]^{d}\;\;\big|\;\;\mathbb{E}\big[\,h(f(X))\cdot(q(f(X))-f(X))\,\big]=0,\;\;\forall h\in{\mathcal{H}}\Big\}.$$

In words, Q consists of all candidate conditional expectations consistent with f satisfying H- calibration. Because the perfect predictor f(X) = E[Y |X] satisfies H-calibration for every H, the identity map q(v) = v is always in Q—but in general the set may contain many maps. From the perspective of the decision-maker who knows f and the promised calibration guarantee H, but does not know the underlying distribution, given a forecast f(x), the true expectation E[Y | f(x)] is uncertain but must lie within Q. As H grows richer, Q shrinks, eventually reducing to {q(v) = v}
in the case of full calibration. Faced with this uncertainty, a natural strategy is to adopt a robust policy that guards against the worst-case admissible reality. Formally, the robust decision rule is
$$a_{\mathrm{robust}}(\cdot)\;=\;\operatorname*{arg\operatorname*{max}}_{a(\cdot):[0,1]^{d}\to{\mathcal{A}}}\;\operatorname*{min}_{q\in{\mathcal{Q}}}\;{\mathbb{E}}\big[u(a(f(X)),q(f(X)))\big].$$
E-u(a(f(X)), q(f(X))). (5)
That is, the decision-maker chooses an action policy that maximizes utility under the worst-case conditional expectation consistent with calibration guarantees. Interpolating Property. The robust policy in Equation 5 interpolates between two classical extremes (Figure 1). If H contains all functions, then Q = {q(v) = v} and arobust reduces to the best response aBR(·) (Equation equation 1). If H is empty, then Q contains all functions and the policy collapses to the constant minimax strategy aMinimax(x) = arg maxa∈A miny∈[0,1]d u(*a, y*).

Thus, Equation 5 provides a principled bridge between best-responding to calibrated forecasts and adopting fully conservative policies, with the level of conservatism controlled by the richness of H.

$$(4)$$
$$({\boldsymbol{5}})$$

The central theme of the remainder of this paper is to investigate the interaction between different levels of H-calibration and the resulting optimal robust policies. Our focus is not on developing methods for achieving H-calibration itself (for which we refer the reader to a rich line of recent work showing how to accomplish this in both the batch and online adversarial setting Hebert-Johnson ´ et al., 2018; Gopalan et al., 2022; Deng et al., 2023; Noarov et al., 2023; Globus-Harris et al., 2023), but rather on understanding the decision-making consequences once such guarantees are in place. In the next section, we begin by analyzing the general problem of deriving optimal robust decision rules for arbitrary classes H. We then specialize to the important case of decision calibration, showing that this weaker and more practical notion identifies large classes of partially calibrated forecasters for which best responding remains optimal. Beyond its theoretical appeal, this result is also practically useful: when a decision-maker can influence the design or post-processing of the forecaster, they can request a decision-calibrated forecaster, to which they can then simply, reliably, and optimally best respond.

Assumption 2.1. The utility u(*a, v*) is linear in its second argument v ∈ [0, 1]dfor each a ∈ A.

This assumption naturally holds in multi-class settings where v is a probability vector over d outcomes and the decision maker has arbitrary utilities U(*a, k*) for each action–outcome pair. In this case, u(*a, v*) = E[U(*a, Y* )] = Pdk=1 vk U(*a, k*), which is linear in v. Such risk-neutral expectedutility models underlie much of the calibration and decision-making literature (e.g., (Foster & Vohra, 1997; Kleinberg et al., 2023; Roth & Shi, 2024)). Utilities that are nonlinear in v, for example, risk-averse utilities depending on outcome variance, fall outside our framework and represent an important direction for future work.

## 3 Optimal Decision Policies For Finite Dimensional H-Calibration

In this section, we characterize the optimal robust decision making policies, i.e., solutions to Equation 5. Throughout this section, we assume the function class H is a finite dimensional space, i.e. it can be described as span of finitely many functions. Formally, let H = span{h1*, . . . , h*k} be the linear class generated by measurable hi: [0, 1]d → R. Then the H-calibration condition equation 3 is equivalent to the k linear moment equalities

$$\mathbb{E}\big{[}h_{i}(f(X))\cdot(\,q(f(X))-f(X)\,)\big{]}=0,\qquad i=1,\ldots,k,$$

so that the ambiguity set in equation 4 may be written as

$\mathcal{Q}=\Big{\{}\,q:[0,1]^{d}\to[0,1]^{d}\,\Big{|}\,\,\mathbb{E}\big{[}h_{i}(f(X))\cdot(\,q(f(X))-f(X)\,)\big{]}=0\,\,\,\,\text{for}\,\,i=1,\ldots,k\Big{\}}$
Intuitively, each equality enforces that, conditional on the forecast, the forecast error has zero correlation with the corresponding test hi; taken together, these constraints exhaust the information provided by H-calibration criteria and hence precisely describe the admissible reality faced by the robust decision-maker in equation 5.

Theorem 3.1 (Characterization of the Optimal Robust Policy). *Suppose* H = span{h1*, . . . , h*k}
with each hi: [0, 1]d → R, and let Q be defined as above. Then the minimax problem in Equation 5 admits a saddle point (arobust, q⋆) with the following structure:
There exist multipliers λ
⋆ = (λ
⋆1, . . . , λ⋆k) *with each* λ
⋆
i ∈ R
d*such that for almost every forecast* v = f(x) *the worst-case map* q
⋆(v) *solves*

$q^{*}(v)\in\arg\min_{p\in[0,1]^{d}}\Big{\{}\operatorname{val}(p)+p\cdot\sum_{i=1}^{k}h_{i}(v)\lambda_{i}^{*}\Big{\}},\quad$_where $\operatorname{val}(p)=\max_{a\in\mathcal{A}}u(a,p)$._
$\left(\text{\hspace{0.17em}}0,1\right)$
Given q
⋆, the optimal robust action at v *is the best response to* q
⋆(v):

arobust(v) ∈ arg max
$$-\mathbf{\nabla}a{\in}A$$
u*a, q*⋆(v).

Interpretation. Theorem 3.1 characterizes both the worst-case distribution consistent with H-
calibration and the corresponding optimal response. For any realized forecast ν = f(x), the theorem

$q^{\cdot}(v)$]. 
yields a simple two-step procedure: compute the adversarial belief

$$q^{\star}(\nu)\in\arg\operatorname*{min}_{p\in[0,1]^{d}}\{\operatorname{val}(p)+p\cdot s^{\star}(\nu)\},\qquad s^{\star}(\nu)=\sum_{i=1}^{k}h_{i}(\nu)\lambda_{i}^{\star},$$

and then take the best response arobust(ν) ∈ arg maxa∈A u(*a, q*⋆(ν)). Thus, the optimal policy is always a best response, not to the raw forecast f(x), but to the adversarially tilted distribution q
⋆(ν) allowed by the calibration constraints. Additionally, a useful consequence is pointwise computability: evaluating arobust at a given ν reduces to two low-dimensional optimizations, without constructing the full mapping x 7→ arobust(x).

From an optimization perspective, the multipliers λ
⋆solve a finite-dimensional concave maximization problem (see the proof of Theorem 3.1), and q
⋆(ν) is obtained by a pointwise convex minimization over p ∈ [0, 1]d. Both stages can be carried out by standard, fast methods with provable guarantees (e.g., projected subgradient ascent for the dual, or a simple primal–dual scheme), after which one evaluates q
⋆(ν) via the pointwise minimization and takes the best response arobust(ν) = arg maxa u(*a, q*⋆(ν)).

In the next section, we analyze the behavior of the resulting decision rules by specializing to concrete H-classes. One might expect that Theorem 3.1 induces a vast hierarchy of policies whose form depends sensitively on H. *Perhaps surprisingly, this is not the case.* In particular, we show a sharp transition: for each decision maker, there exists a specific test class, precisely the one associated with decision calibration, such that as soon as H contains this class, the adversarial tilt collapses (q
⋆(ν) =
ν for a.e. ν) and the optimal robust rule reduces to the plug-in best response to the forecaster.

## 4 Robust Policies Under Decision Calibration And Beyond

In this section, we specialize the general characterization derived in Theorem 3.1 to concrete test classes H. Our core result concerns *decision calibration*: a practically tractable guarantee under which the minimax-optimal robust policy collapses to the plug-in (best-response) rule. This identifies a simple path to decision-theoretic trustworthiness that does not require full calibration.

## 4.1 Decision Calibration And Plug-In Best Response Optimality

Here we define the variant of decision calibration given by Noarov et al. (2023), a slight strengthening of the definition originally given by Zhao et al. (2021). Fix a single decision problem with action set A and utility function u(*a, v*). For each action a ∈ A, let Ra =v ∈ [0, 1]d: u(a, v) ≥ u(a′, v) for all a′ ∈ A 	
be the (closed, convex) decision region on which a is a plug-in best response. The decisioncalibration class is Hdec = { 1Ra: a *∈ A }*. Here, we denote 1A(x) := 1{x ∈ A}. A forecaster f is *decision calibrated* if it is Hdec-calibrated, i.e.,

$$\mathbb{E}\big[{\bf1}_{R_{a}}\big(f(X)\big)\;\big(Y-f(X)\big)\big]\;=\;0\quad\mathrm{for~all}\;a\in{\mathcal{A}}.$$

Compared to full calibration, decision calibration is far more statistically tractable, since its test class has size |Hdec| = |A|, a potentially small and fixed number of actions, rather than the large families required for full calibration.

Theorem 4.1 (Decision calibration ⇒ plug-in best response optimality). If f is Hdec-calibrated, then the minimax-optimal robust rule in equation 5 coincides with the plug-in best response:

arobust(v) ∈ arg max
a∈A
$$f o r\,a l m o s t\,e v e r y\,v=f(x).$$
Equivalently, under decision calibration, best responding to the forecaster is minimax optimal among all forecast-based policies. Put differently, upon observing a forecast v = f(x), the decision-maker need only best respond to v; no adversarial "tilt" survives the decision-calibration constraints. Conceptually, this upgrades the previously known guarantees of decision calibration—that it implies no swap regret (Noarov et al.,

$\underline{I}$. 

![6_image_0.png](6_image_0.png)

![6_image_1.png](6_image_1.png)

2023)—to *minimax optimality*. Swap regret guarantees do not preclude the existence of a policy a : [0, 1]d → A that dominates the plugin best response policy aBR - only that no improved policy has the form a(v) = ϕ(aBR(v)) for some mapping ϕ : *A → A*, using "actions as a bottleneck". In contrast, Theorem 4.1 directly establishes that no other policy a : [0, 1]d → A can improve on the plugin policy aBR in our minimax sense.

The preceding result assumes that the information conveyed by the forecaster to the decision-maker is exhausted by the decision-calibration tests {1Ra}a∈A. In practice, a forecaster might satisfy additional calibration equalities, E-h(f(X)) · {Y − f(X)}= 0, for functions h beyond the indicators 1Ra. The next theorem shows that the plug-in optimality conclusion is stable under such enrichments. This is intuitive: if a forecaster is trustworthy, then making it more calibrated (i.e., adding information) should not diminish that trustworthiness.

Theorem 4.2. Let H *be any test class that contains the decision-calibration indicators,* Hdec = {1Ra: a ∈ A}. If f is perfectly H*-calibrated, then the minimax-optimal robust rule in equation 5* coincides (a.e.) with the plug-in best response:
arobust(v) ∈ arg max a∈A
u(a, v) *for a.e.* v = f(x).

As we make precise in the proof of Theorem 4.2, the "collapse" occurs because the decisioncalibration constraints ensure that the expected utility of the plug-in best-response policy aBR is invariant to the adversary's choice of q ∈ Q. For any q satisfying the Hdec constraints, E[u(aBR(f(X)), q(f(X)))] = E[u(aBR(f(X)), f(X))] .

Thus, the adversary cannot reduce the utility of aBR; its worst-case utility equals its nominal utility. Since aBR is the optimal policy under the nominal distribution, and its performance cannot degrade under any admissible q, it must also be the minimax-optimal policy. Sharp transition. One might initially expect a *gradual* shift from fully conservative to plug-in best response as H is enriched. Theorems 4.1–4.2 show a sharper phenomenon (Figure 2): once H contains the |A| decision tests {1Ra }a∈A, the adversarial tilt disappears (q
⋆(ν) = ν a.e.) and the robust rule *collapses* to the plug-in best response equation 1. Enlarging H further does not change the minimax-optimal policy.

Decision calibration is a tractable, task-specific threshold at which robust decision making and plug-in best-response coincide, providing a crisp target for forecaster design and a clear requirement for downstream decision makers.

As a byproduct, this leads to another practical advantage of decision calibration: a single forecaster can be made simultaneously reliable for a *collection* of downstream decision problems. Intuitively, if the forecast passes the decision calibration tests of each problem, then none of the decision makers needs additional robustness, the plug-in best-response is minimax-optimal for all of them.

Corollary 4.3 (Simultaneous plug-in optimality across multiple decisions). Let u1, . . . , um be m decision problems, with respective action sets Aj *and linear utilities* uj (a, v) in v ∈ [0, 1]d*. For* each j and a ∈ Aj *, let* Ra,j = { v ∈ [0, 1]d: uj (a, v) ≥ uj (a′, v) for all a′ ∈ Aj }

$=\;\{\,v\in[0,1]^d:\;u_j(a,v)\geq0\}$  . 
be the plug-in decision region of action a in problem j*, and define the combined test class*

$${\mathcal{H}}_{\mathrm{dec}}^{\mathrm{all}}\;=\;\bigcup_{j=1}^{m}\big\{\,{\mathbf{1}}_{R_{a,j}}\,:\;a\in{\mathcal{A}}_{j}\,\big\}.$$

If f is H-calibrated for some H *satisfying* Hall dec ⊆ H, then for every j ∈ {1, . . . , m} the minimaxoptimal robust policy for problem j *coincides (a.e.) with the plug-in best response:*

$$a_{\mathrm{robust},j}(v)\ \in\ \arg\operatorname*{max}_{a\in{\mathcal{A}}_{j}}u_{j}(a,v)\qquad{\mathrm{~for~}}a.e.\ v=f(x).$$

Proof. For each problem j, the included indicators {1Ra,j }a∈Ajensure that H contains the decisioncalibration tests of problem j. Theorem 4.2 then applies verbatim to each j, yielding plug-in optimality problem by problem.

4.2 BEYOND DECISION CALIBRATION: GENERIC H-CLASSES FROM TRAINING PIPELINES
Thus far we have focused on *decision calibration*, which, when attainable, collapses arobust to the plug-in best response. In practice, two regimes arise. (i) If one can influence the forecaster's training pipeline, decision calibration is the natural target: it is practical, and our results guarantee plug-in minimax optimality. (ii) If one *cannot* control training, the forecaster might not be decision calibrated for the downstream task. Identifying its partial-calibration profile may be difficult, yet certain moment conditions arise *structurally* from standard training procedures. We give two examples of how to leverage such "free" structure to specify usable H's and derive the associated robust policies.

Self-orthogonality from squared-loss training. A ubiquitous example is *self-orthogonality* (a form of self-calibration) that follows from first-order optimality when a model with a linear last layer is trained to minimize mean squared error. This includes the universally adopted cases of regression with either a linear model or a neural network with a linear head, trained by mean squared error. This and similar guarantees for other loss functions have previously been investigated as consequences of *low degree multicalibration* (Gopalan et al., 2022).

Proposition 4.4 (Self-orthogonality under squared loss). Let X 7→ zϕ(X) ∈ R
k be a representation and fθ(X) = W zϕ(X) ∈ R
d a linear last layer. Suppose θ = (ϕ, W) is trained to a first-order stationary point of the expected squared loss

$${\mathcal{L}}(\theta)\;=\;\frac{1}{2}\,\mathbb{E}\Big[\big|\big|f_{\theta}(X)-Y\big|\big|_{2}^{2}\Big]\;.$$

Then the following calibration moments hold:

$$\mathbb{E}[z_{\phi}(X)\left(Y-f_{\theta}(X)\right)^{\top}]=0\quad\text{and}\quad\mathbb{E}[f_{\theta}(X)\left(Y-f_{\theta}(X)\right)^{\top}]=0.$$

In particular, fθ is H*-calibrated for the test class* H = {hj (v) = e⊤
jv : j = 1, . . . , d} *(and for any* linear combination thereof).

Implications. Proposition 4.4 provides a generic, pipeline-induced H-calibration guarantee whenever a linear head is trained to stationarity under squared loss. Specializing Theorem 3.1 to this setting yields a simple dual. For d = 1 (e.g., one-dimensional regression) with H = {h(v) = v},
the multiplier is a scalar λ, and for each forecast ν = f(x) the worst-case distribution is

$$q^{*}(\nu)\in\arg\operatorname*{min}_{p\in[0,1]}\{\operatorname{val}(p)+\lambda\,\nu\,p\},\qquad\operatorname{val}(p)=\operatorname*{max}_{a\in{\mathcal{A}}}u(a,p).$$

The robust action is then: arobust(ν) ∈ arg maxa∈A u(*a, q*⋆(ν)). When u(*a, p*) is linear in p and A
is finite, val is convex piecewise linear, so the inner minimization reduces to checking finitely many candidate points (endpoints and pairwise breakpoints). The dual objective

$$G(\lambda)=\mathbb{E}{\big[}\operatorname*{min}_{p\in[0,1]}\left\{\operatorname{val}(p)+\lambda f(X)p\right\}{\big]}-\lambda\,\mathbb{E}[f(X)^{2}]$$

is concave in λ and can be maximized via standard one-dimensional methods (e.g., bisection on a monotone subgradient). In higher dimensions (d > 1), the correction term λνp becomes Λνp for a matrix of multipliers Λ, and the pointwise problem remains a small convex program over p ∈ [0, 1]d; for finite A and linear utilities, it is again efficiently solvable.

Zero-bias and bin-wise calibration. A widely available source of partial calibration comes from post-hoc recalibration that many practitioners already apply (mean correction, histogram binning, isotonic-style step fits on a held-out split). These procedures enforce generic (not task-specific) moment constraints that are directly usable in our framework. We focus on *bin-wise* calibration:
take a partition of the forecast range into bins {B1*, . . . , B*J } and enforce, for each bin, E
h1{f(X)∈Bj } (Y − f(X))i= 0, j = 1*, . . . , J.*
This corresponds to the test class Hbin = {1Bj: j = 1*, . . . , J*}, and reduces to zero-bias when J=1 with B1 = [0, 1]d.

Proposition 4.5 (Robust policy under bin-wise calibration). *Let the utility be linear in the outcome* and the action set A *be finite. If* f is Hbin*-calibrated, then with* mj := E[f(X)| f(X) ∈ Bj ] = E[Y | f(X) ∈ Bj ] ,
the worst-case belief is piecewise constant q
⋆(v) = mj for v ∈ Bj *(a.e.)*,
and the robust action best-responds to the bin mean:
arobust(v) ∈ arg max a∈A
u(a, mj )	for v ∈ Bj *(a.e.)*.

Implications. Bin-wise calibration Hbin can be obtained cheaply via standard post-hoc methods
(histogram binning or isotonic regression), and Proposition 4.5 yields an especially simple, closedform characterization of the robust policy. Computing arobust reduces to: (i) estimating mj on a calibration split, and (ii) at test time, mapping v to its bin Bj and best-responding to mj . No additional optimization is needed to compute actions. As a special case, when J = 1 we recover the global-mean constraint E[Y − f(X)] = 0. Then q
⋆is constant, q
⋆(v) ≡ m¯ , with m¯ = E[f(X)] =
E[Y ], and the robust rule ignores v and plays arg maxa∈A u(a, m¯ ). As the partition is refined, the robust rule moves from a single global plug-in best response at m¯ to a piecewise plug-in best response at mj , yielding a richer, finer-grained decision policy.

## 5 Experiments

In this section, we evaluate the validity and practical consequences of our framework by implementing our methods on two real-world datasets. We compare the *plug-in best response* (aBR) against the *robust policy* (arobust), which enjoys minimax optimality guarantees under H-calibration.

We focus on two classes of metrics. *Nominal performance* measures average utility when the test data are i.i.d. from the same distribution as the training and calibration splits; this reflects an optimistic regime that often degrades in practice. *Adversarial performance* probes the other extreme by altering the test-time outcome distribution in two ways: (i) a worst case tailored to the plug-in policy, and (ii) a worst case induced by the robust dual, tailored to the robust policy. In both cases, the adversarial distributions respect the H-calibration constraints and are therefore indistinguishable, from the decision-maker's perspective, from i.i.d. test draws given an H-calibrated forecaster.

Our theory predicts two patterns. First, by minimax optimality, the robust policy should dominate the plug-in rule when each is evaluated against its own worst-case distribution (and typically also under the adversary tuned to hurt the plug-in). Second, because (arobust, q⋆) forms a saddle point of equation 5, when both policies are evaluated under the robust-tuned adversary, the robust policy should not underperform the plug-in rule. Under nominal i.i.d. evaluation, the plug-in rule may achieve higher utility, reflecting the lack of need for conservatism in that regime.

## 5.1 Case Studies: Bike Sharing And California Housing

We evaluate our framework on two regression datasets with distinct decision-making interpretations.

Bike Sharing (UCI). The UCI *Bike Sharing* (daily) dataset Fanaee-T & Gama (2014) records daily rider counts alongside calendar and weather covariates (season, month, weekday, holiday, working day, weather state, temperature, humidity, wind). The outcome Y ∈ [0, 1] is the rescaled total rider count, and the decision-maker chooses a staffing/capacity multiplier from A = {0.8, 1.0, 1.2},
interpretable as conservative, nominal, and aggressive provisioning.

| Dataset            | i.i.d.   | Worst-case for robust   | Worst-case for plug-in   |         |        |       |
|--------------------|----------|-------------------------|--------------------------|---------|--------|-------|
| Plug-in            | Robust   | Plug-in                 | Robust                   | Plug-in | Robust |       |
| Bike Sharing (UCI) | 0.474    | 0.463                   | 0.402                    | 0.410   | 0.393  | 0.412 |
| California Housing | 0.216    | 0.207                   | 0.160                    | 0.164   | 0.155  | 0.166 |

California Housing. The *California Housing* dataset Pace & Barry (1997) records median house values (rescaled to [0, 1]) with demographic and geographic covariates (median income, housing age, population, latitude/longitude, etc.). Here the decision-maker chooses an investment multiplier from A = {0.6, 0.75, 0.90}, interpretable as conservative, nominal, and aggressive investment. Utility specification. In both settings we adopt the utility function u(*a, y*) = *α a y* − C(a), which is linear in y. The benefit term *α a y* captures service or return proportional to realized outcome y, scaled by α > 0. The cost term C(a) grows in a, penalizing aggressive choices via over-provisioning costs or investment risk. This form tunes the under/over-trade-off without departing from linearity.

For Bike Sharing we use (*α, C*(·)) = (0.9, {0.02, 0.05, 0.1}), while for California Housing we use (α, C(·)) = (0.9, {0.02, 0.05, 0.20}). The qualitative conclusions of this Section remain the same under other reasonable parameter choices. Forecasting model. In both datasets, the forecaster f is a two-layer MLP regressor trained to optimize mean squared error. By the self-orthogonality property of linear heads under squared loss
(Proposition 4.4), the learned forecaster approximately satisfies H-calibration with H = {h(v) = v}, which is the calibration constraint used to derive the robust policy arobust. All experiments use an i.i.d. train/calibration/test split (60/20/20). We use the calibration data to substitute any population level expectation that is needed to be computed to derive arobust.

Results. Table 1 reports the mean utilities. The results match theory: under adversaries tailored to the robust policy, the robust rule achieves at least the plug-in performance; under adversaries tuned to harm the plug-in rule, the robust policy secures noticeably higher utility, reflecting its minimax protection. Moreover, the robust policy outperforms the plug-in best response when each is evaluated against its own worst-case distribution.

## 6 Conclusion And Limitations

We developed a decision-theoretic framework for acting on partially calibrated forecasts via a minimax-optimal robust policy over H-calibrated forecasters. We then identified a sharp transition in the behavior of these policies: for any decision problem with m actions, there exist m decision tests (the decision-calibration class) such that, once they are included in H, the robust policy collapses to the plug-in best response. This spotlights decision calibration as a natural requirement whenever the decision-maker can influence the training pipeline. Moreover, even when decision calibration is unavailable, we showed that generic properties induced by standard training and post hoc procedures (e.g., self-orthogonality under squared loss and bin-wise calibration) yield usable test classes H and tractable robust policies within our framework.

Our model assumed that downstream decision makers were risk neutral - i.e., their utility functions u(*a, v*) are linear in v and A is finite; these are standard assumptions in the calibration literature, but broadening them would be interesting. We note that certain classes of non-linear utility functions can be linearized over an appropriate basis (Gopalan et al., 2024b; Lu et al., 2025), which would allow our results to apply - though these bases are not always low dimensional enough to be practical.

## References

Isaiah Andrews and Jiafeng Chen. Certified decisions. *arXiv preprint arXiv:2502.17830*, 2025. Aharon Ben-Tal and Arkadi Nemirovski. Robust optimization–methodology and applications.

Mathematical programming, 92(3):453–480, 2002.

Gabriel Carroll. Robustness and linear contracts. *American Economic Review*, 105(2):536–563, 2015.

Zhun Deng, Cynthia Dwork, and Linjun Zhang. Happymap: A generalized multicalibration method.

In *14th Innovations in Theoretical Computer Science Conference (ITCS 2023)*, pp. 41–1. Schloss Dagstuhl–Leibniz-Zentrum fur Informatik, 2023. ¨
John C Duchi and Hongseok Namkoong. Learning models with uniform performance via distributionally robust optimization. *The Annals of Statistics*, 49(3):1378–1406, 2021.

Cynthia Dwork, Michael P Kim, Omer Reingold, Guy N Rothblum, and Gal Yona. Outcome indistinguishability. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing, pp. 1095–1108, 2021.

Hadi Fanaee-T and Joao Gama. Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence*, 2(2):113–127, 2014.

Dean P Foster and Rakesh V Vohra. Calibrated learning and correlated equilibrium. Games and Economic Behavior, 21(1-2):40–55, 1997.

Itzhak Gilboa and David Schmeidler. Maxmin expected utility with non-unique prior. Journal of mathematical economics, 18(2):141–153, 1989.

Ira Globus-Harris, Declan Harrison, Michael Kearns, Aaron Roth, and Jessica Sorrell. Multicalibration as boosting for regression. In *International Conference on Machine Learning*, pp. 11459– 11492. PMLR, 2023.

Parikshit Gopalan, Michael P Kim, Mihir A Singhal, and Shengjia Zhao. Low-degree multicalibration. In *Conference on Learning Theory*, pp. 3193–3234. PMLR, 2022.

Parikshit Gopalan, Lunjia Hu, and Guy N Rothblum. On computationally efficient multi-class calibration. In *The Thirty Seventh Annual Conference on Learning Theory*, pp. 1983–2026. PMLR, 2024a.

Parikshit Gopalan, Princewill Okoroafor, Prasad Raghavendra, Abhishek Sherry, and Mihir Singhal. Omnipredictors for regression and the approximate rank of convex functions. In The Thirty Seventh Annual Conference on Learning Theory, pp. 2027–2070. PMLR, 2024b.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In *International conference on machine learning*, pp. 1321–1330. PMLR, 2017.

Chirag Gupta and Aaditya Ramdas. Top-label calibration and multiclass-to-binary reductions. In International Conference on Learning Representations. OpenReview, 2022.

Lars Peter Hansen and Thomas J Sargent. Robust control and model uncertainty. American Economic Review, 91(2):60–66, 2001.

Ursula Hebert-Johnson, Michael Kim, Omer Reingold, and Guy Rothblum. Multicalibration: Cal- ´
ibration for the (computationally-identifiable) masses. In International Conference on Machine Learning, pp. 1939–1948. PMLR, 2018.

Lunjia Hu and Yifan Wu. Predict to minimize swap regret for all payoff-bounded tasks. In 65th IEEE Annual Symposium on Foundations of Computer Science, FOCS 2024, Chicago, IL, USA,
October 27-30, 2024, pp. 244–263. IEEE, 2024.

Shayan Kiyani, George Pappas, Aaron Roth, and Hamed Hassani. Decision theoretic foundations for conformal prediction: Optimal uncertainty quantification for risk-averse agents, 2025. URL https://arxiv.org/abs/2502.02561.

Bobby Kleinberg, Renato Paes Leme, Jon Schneider, and Yifeng Teng. U-calibration: Forecasting for an unknown agent. In Gergely Neu and Lorenzo Rosasco (eds.), The Thirty Sixth Annual Conference on Learning Theory, COLT 2023, 12-15 July 2023, Bangalore, India, volume 195 of Proceedings of Machine Learning Research, pp. 5143–5145. PMLR, 2023.

Daniel Kuhn, Peyman Mohajerin Esfahani, Viet Anh Nguyen, and Soroosh Shafieezadeh-Abadeh.

Wasserstein distributionally robust optimization: Theory and applications in machine learning. In Operations research & management science in the age of analytics, pp. 130–166. Informs, 2019.

Meelis Kull, Miquel Perello Nieto, Markus Kangsepp, Telmo Silva Filho, Hao Song, and Peter ¨
Flach. Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration. *Advances in neural information processing systems*, 32, 2019.

Jiuyao Lu, Aaron Roth, and Mirah Shi. Sample efficient omniprediction and downstream swap regret for non-linear losses. In Nika Haghtalab and Ankur Moitra (eds.), The Thirty Eighth Annual Conference on Learning Theory, 30-4 July 2025, Lyon, France, volume 291 of *Proceedings of* Machine Learning Research, pp. 3829–3878. PMLR, 2025. URL https://proceedings. mlr.press/v291/lu25b.html.

Charles F Manski. Identification problems and decisions under ambiguity. *Journal of Econometrics*,
95(2):415–442, 2000.

Charles F Manski. Statistical treatment rules for heterogeneous populations. *Econometrica*, 72(4):
1221–1246, 2004.

Charles F Manski. Choosing treatment policies under ambiguity. *Annual Review of Economics*, 3:
25–49, 2011.

Charles F Manski and Aleksey Tetenov. Admissible treatment rules for a risk-averse planner. Econometrica, 75(3):715–752, 2007.

Georgy Noarov, Ramya Ramalingam, Aaron Roth, and Stephan Xie. High-dimensional prediction for sequential decision making. *arXiv preprint arXiv:2310.17651*, 2023.

Princewill Okoroafor, Robert Kleinberg, and Michael P Kim. Near-optimal algorithms for omniprediction. *arXiv preprint arXiv:2501.17205*, 2025.

R Kelley Pace and Ronald Barry. Sparse spatial autoregressions. *Statistics & Probability Letters*, 33
(3):291–297, 1997.

Benjamin Plaut, Nguyen X Khanh, and Tu Trinh. Probabilities of chat llms are miscalibrated but still predict correctness on multiple-choice q&a. *arXiv preprint arXiv:2402.13213*, 2024.

Aaron Roth. Uncertain: Modern topics in uncertainty estimation. *Lecture Notes*, 11:30–31, 2022.

Aaron Roth and Mirah Shi. Forecasting for swap regret for all downstream agents. In Dirk Bergemann, Robert Kleinberg, and Daniela Saban (eds.), ´ Proceedings of the 25th ACM Conference on Economics and Computation, EC 2024, New Haven, CT, USA, July 8-11, 2024, pp. 466–488. ACM, 2024.

Guy N Rothblum and Gal Yona. Decision-making under miscalibration. In 14th Innovations in Theoretical Computer Science Conference, ITCS 2023, pp. 92. Schloss Dagstuhl-Leibniz-Zentrum fur Informatik GmbH, Dagstuhl Publishing, 2023.

Abraham Wald. Statistical decision functions. In Breakthroughs in Statistics: Foundations and Basic Theory, pp. 342–357. Springer, 1950.

Shengjia Zhao, Michael Kim, Roshni Sahoo, Tengyu Ma, and Stefano Ermon. Calibrating predictions to decisions: A novel approach to multi-class calibration. Advances in Neural Information Processing Systems, 34:22313–22324, 2021.