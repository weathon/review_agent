## Human Reviewer 1

### Summary
This paper presents a nonparametric causal identification framework for functional longitudinal data, exemplified by the MIMIC-IV dataset, which includes complexities such as events like death. By leveraging stochastic process theory and measure theory, the framework generalizes g-computation, inverse probability weighting, and doubly robust formulas, effectively handling time-varying outcomes with mortality and censoring. Monte Carlo simulations verifies it.

### Strengths
This paper addresses the problem of causality identification in complex scenarios inherent in functional longitudinal data, a highly general and advanced form of data. If it can overcome the criticisms and shortcomings pointed out by other reviewers and get published, it could serve as a milestone, taking one step forward in functional data analysis for causal inference.

### Weaknesses
This paper is more abstract than necessary, making it harder to read than other papers. Even after spending more time on it than on other review papers, the preliminaries, equations, and theorems presented in the paper are not easily understood at once. As a result, I had to refer to the cited papers, and I found two papers that share many aspects with this one. One is "Causality for Functional Longitudinal Data," which is cited by (Ying, 2024) in this paper, and the other is "Causality for Complex Continuous-Time Functional Longitudinal Studies with Dynamic Treatment Regimes," which is not cited in this paper. The title of this paper is "Causal Identification for Complex Functional Longitudinal Studies," suggesting that the most significant update in this paper is the "complex" aspect in the title. The paper "Causality for Complex Continuous-Time Functional Longitudinal Studies with Dynamic Treatment Regimes" seems to have updated the dynamic treatment regimes element one step further.

Specifically, in the preparation section, the time-to-event endpoint $T$, $C$, $X$, and $\Delta$ are newly defined, assuming a more complex situation where the study may be forcibly terminated due to an event like death before the study is completed. This updates Theorems 1, 2, and 3 from the previous paper. I would like to ask about the academic and practical significance of solving the more complex problem introduced by those new variables. The most important contribution of this paper seems to be the introduction of the non-parametric property through Theorem 4. Why is the non-parametric property important in a functional data framework? The paper states that it makes the model more flexible and adaptable to various data, but doesn't the continuous functional data, which is more extensive, lead to increased computational and implementation complexity, a common drawback of non-parametric models? Doesn't this also create issues with the interpretability required for healthcare data analysis?

One of the most important equations in this paper seems to be Equation (1). The rest of the paper is dedicated to finding another representation of Equation (1). However, it is not easy for readers to immediately understand what Equation (1) means and why we should be interested. Additionally, it is not straightforward to grasp what $\mathbb{G}$ represents. By referring to the paper by Ying (2024), I could somewhat understand $\mathbb{G}$ through the following example:
*When the causal outcome under a specific regime $\bar{a}$ is of interest, for instance, all patients were under treatment, the point mass (delta) measure $\mathbb{G} = \mathbb{1}(\bar{A} = \bar{a})$ can be considered.*
Including this example in this paper would help in understanding. Furthermore, providing a concrete example of what $\nu$ represents would help comprehend Equation (1). Can Equation (1) be understood as a general expression representing the average treatment effect, the averaged treatment outcome, or a transformed form of these?

If the non-parametric property is a major contribution of the paper, it should be demonstrated through more concrete experimental examples, such as using the MIMIC-IV data mentioned in the introduction. The experimental section currently numerically verifies Theorem 1, which has already been proven in the Appendix, but a demonstration of Theorem 4 seems more necessary. However, under Theorem 4, it is only mentioned that "we have not achieved the full nonparametric paradigm."

Additionally, there is a need to clearly and definitively define loosely defined “functional” data. The abstract of this paper describes it as "characterized by continuous-time measurements," while another cited paper describes it as "characterized by continuous-time processes and high-dimensional measurements." I believe "continuous" alone is not sufficient to be called functional. What is the rationale for developing a framework that assumes functional continuity in the model even though real-world healthcare data does not have mathematically rigorous time continuity and does not observe over an infinite time? (The previous paper assumed up to time $\tau$, but this paper assumes up to $\infty$.) What is the justification for this assumption?

### Questions
Please provide additional explanations for the questions raised in the Weakness section.

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
3

### Confidence
2

---

## Human Reviewer 2

### Summary
The paper is challenging to follow, so I may have misunderstood some parts. My understanding is that the "functional longitudinal data" investigated here are conventional functional data, as described by Wang et al. (2016), which can be measured intensively, sparsely, or irregularly. However, this paper focuses solely on the ideal (hypothetical) setting where continuous-time measurements are available for each experimental subject, resulting in infinite-dimensional data. If this interpretation is correct, the goal of this paper is to explore causal identification for infinite-dimensional functional (time-varying) outcomes that are subject to mortality and censoring by generalizing the classical g-computation, inverse probability weighting, and doubly robust formulas.

Reference: 
Wang, Chiou and Müller (2016). Functional data analysis. Annual Review of Statistics ands its application.

### Strengths
The approach is nonparametric and it accommodates functional treatment processes A(t) and functional confounders L(t), as well as functional response Y(t).

### Weaknesses
The paper is hard to follow and the connection of the event-time T to the outcome Y(t) is unclear.

### Questions
Could you elaborate on the situation when A(t) is a function? 

Why should Y(t) be a subset of L(t), and what does it mean?

### Soundness
3

### Presentation
1

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper proposes a causal identification framework that bridges classical causal inference framework, continuous-time longitudinal analysis and functional data analysis. In this framework, the parameter of interest is the marginal mean of counterfactual outcomes under a measure that allows randomly assigned treatments, with absence of censoring. Leveraging the tools in stochastic process, the authors then demonstrate the identification results for three classical estimation strategies in causal inference: g-computation, IPW and doubly robust estimation. The authors further claims that the identification framework also has non-parametric property.

### Strengths
This paper establishes a new causal identification framework for continuous-time longitudinal studies with functional data, and provides clear and concise theoretical demonstration. I believe that this framework will be of interest to causal inference and machine learning communities.

### Weaknesses
1. The numerical experiment might be an over-simplification of the survival analysis scenario since neither mortality nor censoring are taken into consideration. 
2. What is the causal structure that the framework is focusing on? Specifically, why set $Y(t)$ (outcome of interest) to be a subset of $L(t)$ (measured confounders)? I might misunderstood but are we assuming that previous outcome will impact the current treatment assignment (since confounders, from my understanding, will impact treatment assignment)?
3. I guess it would be helpful to attract readers in a wider community if more intuitive explanation could be added after stating definitions/propositions.

### Questions
1. Why is the interventional distribution ((7)-(10)) formulated in this way? Specifically, I’m curious about where the term $\{1 -   \mathbb{1}(x \leq t_{j+1}, \delta=0) }$ comes from.
2. Can this framework be extended to dependent censoring?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
In this paper, authors consider causal inference on time-varying data (functional longitudinal data). They generalize the classical g-computation, inverse probability weighting and doubly robust formulas to the time-varying setting subject to censoring and mortality. The g-computation formula is simulated using Monte-Carlo on a toy dataset, achieving promising results.

### Strengths
Treatment effect on functional longitudinal data seems to be an understudied subject. This research nicely fills the gap of existing works. 

The resulting G-computation can be quite straightforwardly approximated using observations under simulation settings.

### Weaknesses
The way this paper is written obscures its main ideas (at least to a general, non-expert reader). There are many terms and phrases used without clear explanation (e.g., "g-computation", "counterfactual time-to-event endpoint"). This restricts the range of potential readers of this paper.  

**The experiments are limited to only simulation data and only validate the G-computation formula**. 

The literature review of this paper (section 2) does not seem to provide much information of existing works as without proper explanation, readers may be unclear what "temporal aspect", "point exposure" and "end-of-study outcome" means. I recommend removing Figure 1 and expanding on each of the subsections, providing more details of existing works. 

The preparation in Section 3.1 is quite long. Without concrete examples, it is hard for readers to understand what they actually mean. I suggest skip some unnecessary notations, and explain them as the paper progresses. 
    - Some symbols are better explained with examples. For instance, authors could give an example of nu and G, around equation (1).

### Questions
Line 141, authors mentioned "note this is not a density function". Then please specify what this is. 

Line 325, why is it sufficient to evaluate the approximation of the G-computation formula? I don't think on population level, the values of three formulas are numerically equal. Even if they are equal, they may have quite different finite-sample behaviors.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
2