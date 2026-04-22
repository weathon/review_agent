# On Measuring Influence in Avoiding Undesired Future

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 6, 2, 4

## Abstract
When a predictive model anticipates an undesired future event, a question arises: What can we do to avoid it? Resolving this forward-looking challenge requires determining the variables that positively influence the future, moving beyond the statistical *association* typically exploited for prediction. In this paper, we introduce a novel measure for evaluating the *influence* of actionable variables in successfully avoiding the undesired future. We quantify influence as the degree to which the success probability can be increased by altering variables under the principle of maximum expected utility. Our analysis demonstrates a counterintuitive insight: while related to *causality*, influential variables may not necessarily be those with strong intrinsic causal effects on the target event. In fact, it can be highly beneficial to alter a weak causal factor, or even a variable that is not an intrinsic factor at all. We provide a practical implementation for estimating the proposed measure and validate its utility through experiments on synthetic and real-world tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces the notion of influence power, a measure of the potential
importance of a variable for preventing a system from leading to an undesirable
outcome. This notion is related to, but is shown to differ, to that of average
causal effects. The paper also introduces a UCT-based Monte-Carlo Tree Search
estimator (from noiseless observational data). Theoretical support is provided
for the consistency of the estimator in the unconfounded case, as well as
empirical support on three toy tasks with noiseless data.

### Strengths
**Originality**: to the best of my knowledge, the contribution is novel. It
is definitely related to work in utility maximization and possibly algorithmic
recourse, but it is also different enough -- as clarified by the authors.

[**Q**] Could you please clarify the points of difference with works in causal
algorithmic recourse? E.g., the works by Karimi and colleagues.  I'm sure this
would help the readers.

**Quality**:

**Clarity**: The text is well written and easy to follow.  All key definitions
and arguments are accompanied by illustrative examples, which help a lot.

**Significance**: I am not the most well versed in algorithmic decision making,
but I think this paper provides a useful contribution mixing elements from causality and decision making, and it could open the door to further research.

I'm curious to know what the other reviewers think.

### Weaknesses
Focus on constructive and actionable insights on how the work could improve
towards its stated goals.

**Clarity**: No major complaints on my end, but I did notice a handful of small
linguistic idiosyncrasies, such as:

- Title: I'm not a native English speaker, but the sentence "avoiding
  undersired future" sounds off to me. What about "Avoiding undesirable future
  events"? This is how it's written in the abstract and it works just fine.

- "alterable variable" also sounds off to me -- what about "actionable variable"?

- Section 2, Notation: I don't think the notation {\cal M}_{V_i} used
  elsewhere in the text, it can probably be removed. Especially because it
  looks incorrect: I suspect it should be {\cal M}_{v_i} instead (lower case,
  constant).

- Section 2, Problem Definition: no need to repeat that \Delta_{V_i} is the
  alterable domain. This was already introduced previously.

- Section 3.2: "an alterable variable are worth" -> is worth.

- Section 3.3: "Also, a variable such as..." - why "also"? I'd drop it.

- Above Example 4: "it is not a causal" -> "not being a causal".

A minor issue is it's not immediately obvious to see the difference between
the two main terms in the main equation (at the end of p 3): one contains an
intervention, the other an observation, but they are marked as "a" and "o",
which are quite difficult to recognize as different. Perhaps use color to
help the reader? The equation is much easier to understand once one spots this
otherwise tiny difference ;-)

**Significance**: [**Q**] It'd be good to report the runtime difference between the
three algorithms (the baselines and the MCTS estimator). The results indicate
the estimator outperforms the baselines, which is good, but without
understanding the its computational cost it's difficult to gauge whether the
benefits are worth it. This is my only major complaing with the work.

[**Q**] It'd also be good to clarify from the get go - as soon as the introduction -
that Proposition 1 works only for unconfounded models ("independent background
noises"), which can be a strong assumption in practice, for clarity.

### Questions
I'd appreciate if the authors could comment on the points I've marked as [**Q**] above.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present a novel approach to the AUF (avoiding undesirable future) problem
which improves on SOTA (one-alteration and all-alteration). Their approach can also capture negative
influence, ie changes which are strictly bad with respect to a given target. They use a Monte Carlo
algorithm to make estimating their measure more efficient (than brute force). They evaluate their
method on three standard(?) problems and show that their measure allows them to compute
interventions which increase the chances of the model reaching the desired target region.

### Strengths
I quite like this paper, I'd give it a weak accept. It's far from perfect, the novelty (while
there) is not earth-shattering and most of the worked
examples are little more than toys, but I enjoyed reading it.

### Weaknesses
I quite like this paper, I'd give it a weak accept. It's far from perfect, the novelty (while
there) is not earth-shattering and most of the worked
examples are little more than toys, but I enjoyed reading it. See the questions below.

### Questions
(numbers are line numbers)

089 how much is nearly negligible?
103 it is assumed that all variables are alterable? How strong is this assumption?
- it feels like a very strong assumption to me, as any variable on the causal path (or off of it)
  can be directly manipulated. This certainly does not reflect real world limitations.

104 "desired region" feels a little vague. S is a subset of the possible values of Y?
107 "as much as possible" is also rather vague. Is there an implicit threshold or ratio here?

I don't think that eq 2 implies that all variables *must* be altered, only that they can be altered.
After all, the equation simply states that we must go over all variables and set their values, but
it does not state that the values must be **different** from observed. So the criticism on l.139 does not hold.

133 do not use contractions

155 I don't like the way this is presented. k = d comes out of nowhere, and the MEP and AUF
equations appear to be identical at this point. It's only later that it becomes (slightly) more
obvious that this is deliberate.

158 eq 3 is not well formatted. =o, unlike =a, has not been previously defined. Presumably it means
setting V_k to its actual value in the context. But wait, we get the definition on l.164. This is
backwards and very hard to read. Why not use <- for alteration, and = for observation?

173 Def 1: clearly related to average causal effect. Is it a generalisation?

211 causality need not be transitive. See Halpern for a detailed discussion.
It is also not that surprising that a alterable variable be non-influential. After all, the variable
may cause an exact value of Y in S, but not have sufficient influence to move out of S.

200 notation for \tau is really buried in the text here.

226 this is a slightly forced example, as X and Z are essentially independent, they are both on the
causal path, but X is not a cause. However, it's good to see that IP can detect this.

245 here we have the problem of talking about negligible ACE, as 0.08 is surely very low. Is this
negligible (see comment above)?

262 again, not a surprising result. And the influence (271) is small.

274 small note, but it should read "considering altering", not "considering to alter"

278 "despite it not being a causal ancestor"

Example 4. I do not understand your explanation. It is not intuitive (to me) and seems quite
an important point. This needs clarification.

Eq (9) is the \delta embedded in the equation standard notation for something?
What is A hat? The MC approximation of A at any point?

420 "repeat experiment _with_ ten times"

421 I looked at Appendix A. I'm slightly disappointed by how small/simple the examples are. I wonder
how well the MC sampling approach would work on bigger/more difficult models. Moreover, they are
largely identical to your examples. I think you could just merge this together, name the examples
appropriately, highlight differences if they exist, and remove the appendix. It would make it much
easier to read.

438 "demosntrating"

Figure 2 is very small and quite difficult to read. Please replot or make larger for camera ready.

Table 1 does not show how much work is required for your MC method. Your results are already better
at T=10. I'd like an idea of how much comparable work is performed by Max-One and Max-All.
I'm assuming (because it's not in the comment at least) that these are mean values. While I'm not in
favour of unnecessary statistical analysis, some might be appropriate here. Perhaps a box plot might
be more revealing here, as the standard deviations are quite large for all the tools, and this is
someone obscured by presenting the data as a table.

The results seems to follow a logaritmic curve... why is this? Is this a limitation of the the MC
method? After all, it should be possible, if all variables in V are alterable, to always guarantee Y
in S? Or is this a result of the fact that you cannot alter variables before some point d?

Conclusion section. "intriguing possibility..." but I do not understand why this happens, and your
intuition above is not (to me) intuitive. I could perhaps see the sharing of information via mutual
information at play, but this is non-directional. Is this not instead revealing a limitation in
(Pearl's conception of) SCMs based on probabilities? It's probably always possible to construct
these pathological examples (because they are divorced from real data producing systems) but do they
really tell us anything interesting, other than to look at for these potential errors?

_Questions_:
  * distance of alterations from outcome Y. Are they usually proximate or far away?
  * are interventions consecutive or disjoint in general?
  * are there any patterns/implications in distance from Y?

  * frequency of achieving S in your simulations?
      + S is always limited to one outcome I think, so the entire thing is boolean. Will this scale
    to non-boolean settings.
  * size of alterations vs frequency of S?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces *Influence Power*, a metric aimed at quantifying how altering a variable affects the probability that an outcome variable falls within a predefined desired region ($P(Y\in S)$).

The authors build on a Bellman-style recursive formulation and define Influence Power as the difference between the optimal success probability under intervention and the expected success under natural observation. They then propose a Monte-Carlo tree search (MCTS) approach to approximate influence power using observational data.

Experiments on small synthetic SCMs suggest that the proposed MCTS approach outperforms trivial baselines in increasing $P(Y\in S)$.

### Strengths
- Influence power is a novel metric that combines ideas of potential intervention effect and value-of-information analysis.
- The recursive Bellman-inspired formulation is a creative way to connect causal reasoning with sequential decision making. If done right, this could be impactful.
- Examples 1-4 clearly illustrate the core intuitions (but their extended form could be moved to the Appendix).

### Weaknesses
- Conceptual confusion: counterfactual vs. interventional reasoning

The AUF problem is framed as counterfactual ("given that the world will look like X, what if we changed Z?"), but the paper only computes interventional and observational probabilities. This gap violates the Causal Hierarchy Theorem [1]: interventional data alone cannot answer counterfactual queries. For example, in lines 11-12: "When a model predicts an undesired outcome, it is often crucial to determine what we can change to avoid it.". Posed this way, it is theoretically impossible to solve the AUF problem (at least exactly). The formulation in the paper, though, is closer to a sequential decision process (or a planning problem) than to counterfactual inference. The authors likely need to just write their problem more formally to make this clear. See more points below around assumptions and the concept of time.

[1] Bareinboim et al. "On Pearl's Hierarchy and the Foundations of Causal Inference", 2020


- Ambiguous use of time and ordering

The paper implicitly assumes a temporal sequence of variables $(V_1,\dots,V_d,Y)$ but **SCMs are static**.
Phrases such as "subsequent variables" or "before Y is finalized" suggest temporal evolution, yet this is never formalized (no explicit time, transition, or policy definition).
The authors effectively simulate time by imposing a topological order, conflating causal order with temporal decision order (see below for discussion of ACE suitability to this setting). To resolve this ambiguity, the authors need to formalize their problem within a framework that supports time (Example unclear question: Are all variables observed before Y is realized? I would guess yes, but it's very unclear). 


- Unclear problem definition and assumptions

Key assumptions are not clearly stated:

1. Causal sufficiency (no hidden confounders) seems assumed but contradicted by examples involving an unobserved $U$. For instance, in example 4 the agent must choose $X$ without access to $U$, even though $U$ influences both $X$ and $Y$. From the decision-maker's perspective, $U$ acts as a hidden confounder, creating an inconsistency with the assumption of causal sufficiency (the time dimension is what makes this so complicated).
2. Is the causal order taken as known? (I assume yes after reading Proposition 1)
3. Positivity/overlap conditions for feasible alterations are unstated.
4. The predictor $h(x)$ is mentioned in preliminaries but never used formally.

The absence of assumptions makes the formal objective of the AUF problem ambiguous.


- Misinterpretation of causal relationships

Several claims conflict with standard causal semantics. Some examples:

1. *Example 2:* the paper concludes that altering $X$ is "counterproductive" despite a positive ACE. In standard SCMs, this is impossible. The claim arises only because intervening on $X$ destroys information useful for later decisions (N.B. SCMs don't support time and order of decisions).
2. *Example 4:* a non-ancestor variable $W$ is said to have positive influence on $Y$. In causal terms, $do(W)$ cannot affect $Y$ at all. The "influence" is because of the effect of $W$ on how informative later observations are, not on the data-generating process itself.

Hence, "influence" in this paper measures more like *policy utility* rather than *causal effect* (which is also why comparing to ACE is not very suitable, see next point).


- Over-reliance on ACE comparisons and why ACE is unsuitable for the current formulation.

The paper repeatedly contrasts Influence Power with ACE.
ACE is defined for **static SCMs**.
Comparing it to a metric that implicitly models information propagation over time is conceptually inconsistent. If the SCM were explicitly **unrolled over time** (with one variable per time step) and ACE were computed in that temporal formulation (e.g., changing $W_0$ and measuring its effect on $Y_t$), the comparison would then be meaningful, and $W_0$ would indeed be a causal ancestor of $Y_t$, even though it's not a causal ancestor of $Y_0$.

Additionally, ACE measures the *expected* change in $Y$ when a binary variable flips from 0 to 1. It is not informative when the goal is to increase the probability that $Y$ falls within an arbitrary region (even in a counterfactual setting, such reasoning is on an individual level, while ACE is population level) or when variables are non-binary. This comparison adds little insight and occasionally misleads (e.g., claiming that a variable with negligible ACE may still be "highly influential").

- Toy experiments and limited validation

1. The experiments involve only tiny binary SCMs, no non-linear or continuous settings are tested.
2. The baselines ("max-one", "max-all") are trivial (and arguably not suited for this problem without SCM unrolling), so improvement isn't that unsurprising. There should also be a comparison to a baseline that only observes.
3. No runtime analysis or sensitivity tests ($\alpha$ parameter, wrong topology, data size). For example, MCTS becomes expensive as the number of alterable variables grows, since the tree expands exponentially.
4. The claim that "a rough approximation suffices" lacks evidence.

Furthermore, the experiments evaluate success in expectation over the entire data-generating process rather than at the level of individual contexts.
In each task, the authors compute ($P(Y\in S)$) under different alteration strategies averaged across all exogenous realizations. This measures population-level effects but not the per-instance decision problem implied by the AUF formulation ("given this particular observation, what should be altered to avoid the undesired future?"). Consequently, the experiments do not validate whether influence power improves decision-making for specific predicted outcomes. They only show population-level improvements. This is misaligned with what AUF is framed as, a context-specific (conditional) decision problem.

- The paper includes no reproducibility statement or supplementary material. Although not strictly required, this limits my ability to verify the reported experimental results.

### Questions
- Are you solving a *counterfactual* question ("what if we changed X in this observed world?") or a *decision-theoretic* question ("which variable should we alter next to maximize expected success")?

- Are all variables observed before Y is realized?

- Clarify the formal problem and specify all assumptions clearly.

- Improve the experimental design. 

If you choose to use ACE, unroll the SCM over time to make it fair. Consider also adding the following:
1. Add tasks with non-binary variables, nonlinear or multiplicative mechanisms, or denser graphs.
2. Add a baseline that only observes.
3. Include sensitivity analysis for $\alpha$, order errors, and sample size.
4. Report runtimes and sensitivity tests

- How stable is influence estimation across multiple Monte Carlo seeds or data resamples?

- Do you think that, given the exponential growth of the search tree in MCTS, the method is inherently limited to a small number of alterable variables (which is likely partly why the experiments use small toy models)? Perhaps you have future work in mind on sampling or pruning strategies to improve scalability?

- Notes on improving notation (minor):
1. The notation $(V_i^a=v_i) / (V_i^o=v_i)$ is non-standard. The paper introduces both this notation and the standard *do*-notation (lines 87-88). Ideally, introduce only one and use it consistently.
2. Keep observed context (x) explicit in all conditional expressions.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a framework for the “avoiding undesired future” (AUF) problem, which aims to identify which variables should be changed to prevent a predicted negative outcome. The authors introduce a metric, called influence power, to measure how useful each variable is to achieving this goal. They claim that influence power differs from traditional causal measures such as the average causal effect,  since variables with strong causal effects may have little or even a negative influence. In contrast, weakly causal or non-causal variables may still have a positive impact. The paper also introduces a Monte Carlo Tree Search (MCTS) method to estimate influence power using observational data.

### Strengths
The paper tackles a relevant and important conceptual problem: moving from passive prediction to proactive intervention to avoid undesired outcomes. The motivation is to develop a principled approach to determine which variables to alter. I liked the connection between causal reasoning and utility-based decision theory, specifically through the principle of maximum expected utility and a Bellman-style recursive definition.

### Weaknesses
* The notion of "influence power" should be compared and contrasted with similar intervention-based approaches: for example, actual causes [1] (smallest set of variables that can be altered to change an outcome), counterfactual explanations [2] (set of interventions that optimise a counterfactual outcome), or the paper [3] on agent incentives (which also uses utility to evaluate interventions, and discusses a related notion of value of control).

* The experimental section is limited to three toy models (trader, farmer, and doctor). The baseline comparisons focus on simple strategies (altering the highest probability variables or altering all variables); it'd be good to compare against a selection of the above-recommended approaches.


[1] Halpern, Joseph Y. "A modification of the Halpern-Pearl definition of causality." arXiv preprint arXiv:1505.00162 (2015).

[2] Tsirtsis, Stratis, Abir De, and Manuel Rodriguez. "Counterfactual explanations in sequential decision making under uncertainty." Advances in Neural Information Processing Systems 34 (2021): 30127-30139. 

[3] Everitt, Tom, Ryan Carey, Eric D. Langlois, Pedro A. Ortega, and Shane Legg. "Agent incentives: A causal perspective." In Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 13, pp. 11487-11495. 2021.

### Questions
* Can the authors clarify how their "influence power" notion and overall framework relate to the above-suggested approaches?

* Given the computational intensity and large number of Monte Carlo simulations required to achieve meaningful results, can the authors clarify how their method scales in practice?

### Soundness
3

### Presentation
3

### Contribution
2
