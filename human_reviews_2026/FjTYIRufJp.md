# A Unifying View of Coverage in Linear Off-policy Evaluation

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Off-policy evaluation (OPE) is a fundamental task in reinforcement learning (RL). In the classic setting of \emph{linear OPE}, finite-sample guarantees often take the form
$$
\textrm{Prediction error} \le \textrm{poly}(C^\pi, d, 1/n, log(1/\delta)),
$$
where $d$ is the dimension of the features, and $C^\pi$ is a **_feature coverage parameter_** that characterizes the degree to which the visited features lie in the span of the data distribution. While such guarantees are well-understood for several popular algorithms under stronger assumptions (e.g. Bellman completeness), the understanding is lacking and fragmented in the minimal setting where the target value function is linearly realizable in the features. Despite recent interest in tight characterizations of the statistical rate in this setting, the right notion of coverage remains unclear, and candidate definitions from prior analyses have undesirable properties and are starkly disconnected from more standard definitions in the literature.  

We provide a novel finite-sample analysis of a canonical algorithm for this setting, LSTDQ. Inspired by an instrumental-variable view, we develop error bounds that depend on a novel coverage parameter, the **feature-dynamics coverage**, which can be interpreted as linear coverage in an induced dynamical system for feature evolution. With further assumptions---such as Bellman-completeness---our definition successfully recovers the coverage parameters specialized to those settings, finally yielding a unified understanding for coverage in linear OPE.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work makes an important contribution, studying offline linear policy evaluation via the linear temporal differences method without assuming Bellman completeness.

Moreover, the current submission shows that the analysis of LSTDQ offers a unification in the linear setting of several previous methods initially proposed for the general function approximation setting.

### Strengths
1)The bound in Corollary 1 scales with the population version of the coverage parameter, not only on the stochastic version.

2) The results are sound and interesting.

3) I really appreciated Section 5.3 and Proposition 2, which makes the comparison with the existing literature very clear.

4) I think that also the unification with MWL is an important contribution.

### Weaknesses
1) I think that the invertibility assumption on $\Sigma$ and $A$ is quite strong. Could you avoid this assumption adding a small scalar times the identity in the definition of $A$ and $\Sigma$?

2) Same comment holds for the empirical estimates $\hat{\Sigma}$ and $\hat{A}$

3) Corollary 1 seems to guarantee good approximation of the performance of the policy $\pi$ starting from states sampled from the initial distribution. 
Can the Q value at states action pairs which are not visited under the initial distribution be estimated reliably with your method ?

4) In proposition 3, in the definition of $C^\pi_\phi$ the indices of $k,a$ do not appear inside the expectation, please fix this typo.

### Questions
1) See weaknesses.

2) How would a linear Monte carlo approach perform in this setting ?  Intuitively a Monte carlo method seems to me more suitable than temporal difference methods when the Bellman completeness assumption is not imposed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a finite-sample analysis of linear off-policy evaluation using LSTDQ under the assumption of linear realizability, framed through an instrumental-variable view of the TD equation. Its main technical contribution is the introduction of a scale-invariant "feature-dynamics coverage" term, defined in the feature-compressed MDP induced by the given representation, which leads to a value-error bound with the standard $\sqrt{d/n}$ dependence. A notable aspect is that this coverage quantity is shown to subsume several existing notions: it recovers the usual $\chi^2$ / density-ratio concentrability in the tabular case, the aggregated concentrability used in abstraction-based work, and the classical linear-coverage condition when Bellman-completeness is additionally assumed. In this way, the paper provides a unifying perspective on when LSTDQ can be expected to succeed off-policy, without proposing a new algorithm but clarifying the role and interpretation of coverage in this widely used setting.

### Strengths
- The paper introduces the feature-dynamics coverage parameter $C_\phi^\pi$, providing a unified perspective on coverage in linear off-policy evaluation (OPE). Derived from an IV view of the LSTDQ algorithm, $C_\phi^\pi$ quantifies how well features induced by the behavior policy capture the subspace relevant to the target policy. It interprets coverage as occurring within a feature-compressed MDP, linking the environment’s dynamics with the feature representation and offering a scale-invariant, interpretable notion of coverage.
- The analysis establishes a finite-sample bound for LSTDQ under the realizability assumption, avoiding the stronger Bellman-completeness requirement. The off-policy value estimation error 

$$|J_{\hat{Q}_{\text{lstd}}}(\pi) - J(\pi)|$$

scales as $O(\sqrt{C^\pi_{\phi}   d \log(1/\delta) / n})$, matching the optimal linear regression rate for $\gamma = 0$. Expressing the bound in terms of $C_\phi^\pi$ highlights how coverage, feature dimension, and sample size interact. The IV-based formulation offers a clear statistical interpretation of LSTDQ as a solution to an errors-in-variables problem, yielding transparent and tight guarantees.
- The proposed framework unifies several prior notions of coverage in reinforcement learning. In the tabular case, $C_\phi^\pi$ reduces to the $\chi^2$-divergence between the target and behavior occupancies; under state abstraction, it coincides with aggregated concentrability; and with Bellman-completeness, it becomes the standard linear coverage. This unified formulation connects previously distinct definitions and provides a coherent understanding of coverage across tabular, abstracted, and linear approximation settings.

### Weaknesses
1. The paper focuses on the linear function approximation setting, assuming $Q_\pi(s,a) = \phi(s,a)^\top \theta^\star$. This assumption enables a clean finite-sample analysis of the LSTDQ estimator and the introduction of the coverage parameter in Equation (13). However, the framework relies on the invertibility of $\Sigma$ and $A$ and applies only to the linear regime. Recent work in off-policy evaluation has advanced toward general function approximation via eluder dimension, where representation error and nonlinearity become crucial and closer to real world settings.
2. Theorem 1 presents a finite-sample error bound of the form 

$$|J_{Q^{\text{lstd}}}(\pi) - J(\pi)| \lesssim \frac{V_{\max}}{1 - \gamma}   \sqrt{\hat C^{\pi}_{\phi}   \frac{d \log(1/\delta)}{n}}$$ 

While this expression is elegant, $\frac{V_{\max}}{1-\gamma}$ and the coverage parameter $\hat{C^{\pi}_{\phi}}$ 

can be extremely large, and the feature dimension $d$ may also be high. In practice, for high-dimensional representations or discount factors $\gamma$ close to $1$ (i.e., long-horizon problems), the required sample size could become prohibitively large. Furthermore, Corollary 1 introduces a burn-in threshold $n_0$, which depends on spectral quantities such as $1/\sigma_{\min}(A)$. This implies that the theoretical guarantee may not hold -- or may severely degrade -- when $n < n_0$.

### Questions
Most of my concerns are already detailed in the cons section. I would be happy to raise the score if the authors can address them convincingly. One remaining question: The definition of the feature–dynamics coverage parameter $C_\phi^\pi$ is mathematically elegant but somewhat opaque in intuition. It would be helpful if the authors could provide more explanation or intuition for what this quantity measures—e.g., how it relates to state–action visitation overlap or spectral properties of the induced feature dynamics—and how to interpret large or small values of $C_\phi^\pi$ in practice. Can the authors give an concrete, real-world example for this?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel finite-sample analysis of LSTDQ, based on a new coverage parameter $C_{\phi}^{\pi}$ (known as ``feature-dynamics coverage'') derived from an instrument variable (IV) point of view. It also provides sufficient discussion on how the new coverage parameter relate to the existing parameters by showing a good collection of equivalence results.

### Strengths
1. The proofs are checked to be mathematically sound.
2. The perspective of analysis looks new to me.
3. Section 5 is appreciated since it delivers very clear messages on how to make sense of the newly defined parameter, as well as providing a good collection of equivalence results with existing parameters.

### Weaknesses
1. The so-called ``IV perspective'' that inspires the new results confuses me a bit.
    * As far as I'm concerned, in a linear model $Y = X^{\top} \theta + \epsilon$, IV is only necessary when $X$ and $\epsilon$ are not independent. Speaking of intuitions, I don't see why it should be the case here.
    * It is also a little confusing to refer to Eq. (7) as the linear regression problem, since linear regression shouldn't come with the $\mathbb{E}$, but rather, with observable individual data points.
    * The IV perspective totally disappears in the proofs in the appendix. It appears to me that the analysis, at most, borrows tools from the IV literature rather than providing new interpretations of the LSTDQ estimator.
2. Empirical/numerical results are missing to support the quality of the analysis.
     * Is it possible to provide some evidence for the advantage of the new coverage parameter? Specifically, is it helpful to include an example where $C_{\phi}^{\pi}$ is much tighter than the existing estimators?

### Questions
See weakness above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper revisits linear off-policy evaluation under mere realizability by
casting LSTDQ in an instrumental-variable (IV) form and introducing the
feature-dynamics coverage $C^\pi_\phi$ that tightens return-estimation guarantees. In doing so, the analysis addresses three shortcomings of
prior coverage parameters: (i) Scale Invariance: stability to feature
rescaling, unlike $1/\sigma_{\min}(A)$); (ii) Off-Policy Characterization:
the dependence on off-policy error propagation rather than only on strict
on-policy boundedness; and (iii) Unification:
$C^\pi_\phi$ aligns with prior coverage concepts, including Bellman completeness, importance weighting and abstract MDP analysis.

### Strengths
- Use $Z=\phi(s,a)$ as an instrumental variables to solve the "error in variables", which is induced by
    $X=\phi(s,a)-\gamma\,\phi(s',a')$, yielding a
    finite-sample value bound.

- The proposed feature-dynamics coverage resolves key deficiencies of prior metrics, by ensuring scale-invariance and meaningful characterization under general off-policy distributions.

- The new definition of coverage via Proposition 1 is elegant, interpretable, and enables unification of various existing notions.

### Weaknesses
1. The motivation for key constructions appears late, making the early sections harder to follow.
2. The paper could better distinguish the roles of Theorem 1 and Proposition 1 to clarify the main message.

### Questions
1. The construction of variables Z and X in the IV-style formulation is introduced without context or intuition upfront, making the initial analysis
hard to follow. Only later is this formulation justified through IV theory.
2. Theorem 3, which avoids the drawbacks of 1/σmin(A), is mentioned in
the appendix despite directly addressing the key concerns raised in the
introduction.
3. The three issues raised at the beginning of the paper are all addressed
through the coverage form in Proposition 1. In contrast, Theorem 1 seems
to be more of an intermediate result, and the main storyline becomes
unclear.

### Soundness
3

### Presentation
2

### Contribution
3
