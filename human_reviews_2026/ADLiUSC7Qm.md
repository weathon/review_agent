# Composition of Pretrained Diffusion Models: A Logic-Based Calculus

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Composing pretrained diffusion models provides a cost-effective mechanism to encode constraints and unlock complex generative capabilities. Prior work relies on crafting compositional operators that seek to extend set-theoretic notions such as union and intersection to diffusion models, e.g., using a product or mixture of the underlying energy functions. We expose the inadequacy and inconsistency of combining these operators in terms of limited mode coverage, biased sampling, instability under negation queries, and failure to satisfy basic compositional laws such as idempotency and distributivity. We introduce a principled calculus grounded in fuzzy logic that resolves these issues. 
Specifically, we define a general class of conjunction, disjunction, and negation operators that generalize the classical mixtures, illustrating how they circumvent various pathologies and enable precise combinatorial reasoning with score models. Beyond existing methods, the proposed *Dombi* operators yield complex generative outcomes, such as the Exclusive-OR (XOR) of individual scores. We establish rigorous theoretical guarantees on the stability and temperature scaling of Dombi compositions, and derive Feynman-Kac correctors to mitigate the sampling bias in score composition. Empirical results on image generation with stable diffusion and multi-objective molecular generation substantiate the conceptual, theoretical, and methodological benefits. Overall, this work lays the foundation for systematic design, analysis, and deployment of diffusion ensembles.
Code is available at [https://github.com/Aalto-QuML/logic-diffusion-composition](https://github.com/Aalto-QuML/logic-diffusion-composition)

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Dombi operators as a principled way of implementing conjunction, disjunction, and negation operations for composing diffusion models. The Dombi operators seek to correct flaws in the standard implementations of the composition operations including bias, instability, and failure to obey compositional laws. The proposal is tested on image and molecular generation problems.

### Strengths
I appreciate the rigorous derivation of "correct" operators for conjunction, disjunction, and negation. The perspective is helpful for clarifying and unifying a somewhat-messy landscape in which different works use different implementations of compositional operators loosely tied to logical operators -- but which may not represent those logical operators very well, and also may not combine together (e.g. conjunction with negation) in logically consistent ways.

### Weaknesses
The analysis of the flaws in PoE/MoE methods, as well as the comparison of the Dombi operators to "standard implementations", make particular choices of implementation that are in some sense the "naive" ones rather than what I would consider the "standard" ones: for example, implementing conjunction as $p^1(x) p^2(x)$ and negation as $p^1(x) / p^2(x)^\gamma$. Works (referenced in this paper but not discussed in great detail) such as Du23 and Liu22 implement conjunction not as a simple product but actually $p(x) \prod_i p(x|c_i) / p(x)$ (for two distributions, $p(x|c_1)p(x|c_2) / p(x)$), and negation not as $p(x|c_1)/p(x|c_2)$ but $p(x) p(x|c_1) / p(x|c_2)$, where $p(x)$ is the unconditional. Also, CFG is essentially a negation $p(x|c)^\gamma p(x)^{1 - \gamma} = p(x) (p(x|c)/p(x))^\gamma$. Choices like these work better in practice, and seem to be closer to the Dombi operators derived in Def 4.1. (For example the denominator in (8) is somewhat related to the unconditional). I also suspect that these implementations might not suffer from some of the failure cases (instability, failure to compose, etc.) I feel that these Du/Liu variant would be a valuable (fairer) baseline for comparison. (I still think there is theoretical value in the derivations of the Dombi forms even if they turn out to be similar to the Du/Liu variants, because to my knowledge the forms described in these earlier works were essentially discovered to work well empirically without much theoretical justification.)

### Questions
Can you comment on connections to the Du/Liu variants of compositional operators as discussed in Weaknesses? Did you try any of these variants for comparison, or test whether they suffer from the identified failure cases?

Discussion around Definition 4.1: I wonder if there is any connection with parametrization-independence here. Bradley&Nakkiran25 (Appendix I) (https://arxiv.org/pdf/2502.04549) (and possibly other references) suggest that parametrization-independence may be a beneficial property for compositional operators (a property possessed by the CFG operator and the Du/Liu styles of product composition); they also show that all 1-homogeneous operators are independent of parametrization. The Dombi operators you obtain appear to be 1-homogeneous as well. Can you comment on potential connections with parametrization-independence and/or other reasons that 1-homogeneity may be desirable for these operators?

Minor comments:
L123 “important to note”?
L142: CFG is usually defined as $p^1(x)^\gamma p^2(x)^(1 - gamma) = p^2(x) (p^1(x)/p^2(x))^\gamma$ — not sure what is meant here?
L215 I think this is the first time $\phi_c$ is mentioned; if so, not very well motivated/introduced 
L244 Without reading Appendix A, it’s not clear what $g$ is

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper examines techniques to compose diffusion models with set theoretic operations such as unions and negations. They find inconsistencies in existing techniques, in particular for negation and complex combinations of operations. They then address these problems with Dombi operators which they derive rigorously from fuzzy logic. They show that their operators are more well behaved and allows for consistent composition and algebraic operations and show more stable sampling of their method on three datasets resulting in higher sample quality.

### Strengths
- Composing diffusion models is extremely useful, in particular negation has many practical use cases.
- The paper clearly demonstrates the shortcomings of existing approaches to motivate the need for their method. 
- The operators are rigorously derived from fuzzy logic, putting their method on firm mathematical grounds
- The experimental results convincingly demonstrate the effectiveness of their method

### Weaknesses
- The presentation of the paper could be slightly improved. For example, the different colors in figure 2 are too similar, making it hard to read. - - It is also not clear which of the dark curves represents which probability distribution
- Some variables seem to be undefined or fall out of nowhere, like the form of g(x) and s_c in definition 4.1 
- The used lambda hyperparameter seems to very significantly between different experiments, raising the question how carefully this has to be tuned

### Questions
- How sensitive is the lambda hyperparameter?
- Did you try larger logical formulas? How robust is this?
- How expensive is the sampling; does the integration become more expensive in particular with the mentioned oscillations, and does the construction of the composition incur any computational overhead?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates compositional generation in diffusion models through the lens of fuzzy logic. By building on theoretical foundations in score-based generative models, existing compositional operators, and fuzzy set theory, the authors proposes Dombi operator that addresses limitations of prior approaches and demonstrates improved stability. Experimental results on small diffusion models, Stable Diffusion 1.4, and protein synthesis further support the effectiveness of the method.

### Strengths
- The paper is grounded in solid theoretical analysis, clearly identifying weaknesses in existing compositional operators for diffusion models.
- The proposed Dombi operator demonstrates strong compositional performance (e.g., Figure 3) and provides a flexible framework for combining multiple operators.

### Weaknesses
- The presentation of Sections 2-5 would benefit from improved clarity: several terms are introduced without sufficient explanation, and there are noticeable typos that hinder reading.
- Although the theoretical justification is strong, the empirical evaluation is relatively limited. Additional qualitative visualizations and ablation studies would help further substantiate the claims, particularly in the context of real-world diffusion models.
- Some empirical results for SD1.4 are referenced as being in the supplementary material (in Appendix C.2), but no supplementary was provided.

### Questions
- How does the Dombi operator affect inference-time efficiency when applied to large-scale diffusion models (e.g., Stable Diffusion)? Is the overhead negligible or significant?
    - How does the method extend to models with stronger inherent compositional abilities (e.g., SDXL, SD3)? Do the benefits persist or diminish?
- Were the images in Figure 3 generated with FKC correction? If so, how large is the performance gap between Dompi with and without FKC correction?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Combining or negating (conditional) diffusion models is an important problem for controllable generation, whether in text-to-image generation or designing molecules which (do not) have certain properties.   The paper highlights pitfalls with several existing heuristics for combining models via the product or negating models with quotients, and seeks to correct these by turning to fuzzy logic operations.    The proposed Dombi Operators satisfy DeMorgan's Laws and encompass several existing heuristics for combination or negation.     Finally, the authors derive a practical scheme for sampling from the combination of diffusion, combining techniques from several recent works.

The authors demonstrate gains from the proposed compositions on text-to-image generation and dual target drug design, along with several insightful examples to elucidate properties in Figure 2.

### Strengths
The paper is logically structured and well-executed in (i) demonstrating the pitfalls of existing heuristic for model combination, (ii) deriving Dombi Operators which satisfy DeMorgan's Laws as we would expect from logical operations (iii) deriving an SMC resampling scheme combining two recent methods (Ito Density Estimators and Feynman Kac Correctors), and (iv) demonstrating its efficacy.


Sec. 3.2 makes a *very* nice point about implicit temperature scaling of score addition!

I am very positive on the paper apart from a technical concern about the negation derivations (described below).   If valid, this should be mathematically salvageable but also may change experiment results.    Thus, I am giving a low initial score, but I am looking forward to discussion with the authors.

### Weaknesses
**Concerns re: Negation** 
I have a concern regarding negation in Definition 4.1, especially moving from constant $c \perp x$ to $c(x)$.  
- First, even for $c\perp x$, I get $\phi_c(x) = \frac{x}{x+c^2} \implies \phi_c^{-1}(y) = c^2 \frac{y}{1-y}$. 
     - Now, $\phi_c^{-1}(1-\phi_c(p(z))) = \phi_c^{-1}( \frac{c^2}{p(z)+c^2}) = c^2\left( \frac{c^2}{p(z)+c^2} \right)\left( \frac{p(z)+c^2}{p(z)} \right) =\frac{ c^4 }{ p(z)}$
     - This does not match the stated $\neg_c p(z) = \frac{c^2}{p(z)}$ in Eq. 7

- What would it mean to take $\phi_{c(x)}(p(x))$?   This is using a different function $\phi_{\mathfrak{c}}(t)$ depending on the input $t \in [0, \infty]$ (since for given $x$, we are interested in $\mathfrak{c}=c(x)$ and $t = p(x)$ )
     - It is also unclear how to invert $\phi$ now
     - Not sure where the `order isomorphism' condition is used in Def A1, but again, now $\phi$ isn't a fixed function.

*Nevertheless, the conjunction and disjunction distribution operations in Eq 8-9 are correct for constant $c$.   Thus, only the negation operation is problematic*.   If the main goal for this definition of negation was to derive the contrast operator of Garipov et. al 2023, there should be other ways (see Observations below).

**Clarity**  
The clarity of the paper is further lacking in several places.

*Definitions* (mostly fixed by including Def. A1 in main text)
- L215:  The role of $\phi_c$ has not been introduced.
    - $N(x) = 1-x$ corresponds to $\phi(x) = x$, and it probably does make sense to start with this $N(x)$.  You can then mention more general $N_\phi(x) = \phi^{-1}[ 1- \phi(x)]$ and reference App. Def A1  (*not referenced anywhere in main text!*)
    - This would also emphasize the role of $\phi$ *alone* to define negation, which then feeds through to conjunction and disjunction via DeMorgan's laws for a given $f$
- App Def. A1:  should have citation or proof that these satisfy DeMorgan's Laws (I confirmed that both hold for the given definition)
- L242:  $g(x)$ is not defined.  


*Sec 5:  Bounds on Precision and Stability*
- Lines 274-275 and L276-277 should be reworded.   Only after reading the title of Corollary 5.1 "Idempotency and Distributivity Bias" did I understand what we we talking about.


*Algorithm*
- In Sec 6 (and possible Alg 1), it would be useful to re-emphasize the use of the Ito density estimator (Eq. 3) to track weights in the composite scores.   

*Experiments*
- What $\lambda$ are used for operations in Figure 3?
- Please check the logic in Lines 396-398 and consider writing the result of each logical clause for legibility.  I get that $p_{\text{xor}} = \{ \textcolor{blue}{2,3}, \textcolor{orange}{0,1} \}$ so that $p_{\text{xor}} \wedge \neg p_3 = \{ \textcolor{blue}{3}, \textcolor{orange}{1} \}$ since $\{ \textcolor{blue}{2}, \textcolor{orange}{0} \} \in p_3$

### Questions
I am confused by the equation in Line 204, where the first and third logical statements are equivalent but the second is not.   I presume the authors wanted to obtain different "probability-distribution logic" for equivalent logical statements.



*Minor comments:*
- Prop. 6.1 should reference the Feynman-Kac / weighted PDE in Eq. 6.  Otherwise it is not clear where $g_t^{1,2}$ are coming from.
- why introduce the notion of energy?  I thought it might be cleaner to just use densities.   
- In Alg. 1 line 4, the authors point out the transformation $\lambda \leftrightarrow -\lambda$ between conjunction and disjunction for fixed $\lambda$.   This is a useful observation worth emphasizing!  
    - I suppose it's obvious from the score expressions in Lines 8-9, but less obvious from the distributions

**Observations:**

A distinct justification for the Dombi conjunction and disjunction in Eq 8-9 can be given using $\lambda$ (a.k.a. '$\alpha$' or power-) mixtures or quasi-arithmetic means of densities ([1], used e.g. for annealing paths in [2]).      For a similar generator $h(p(x)) = \frac{1}{\lambda}p(x)^{\lambda} - \frac{1}{\lambda}$, define
$$
\mu^{(\lambda)}_w(x; p,q) = h^{-1}[(1-w) h(p(x)) + w h(q(x))] / Z \\
= [ (1-w) p(x)^{\lambda} + w ~q(x)^{\lambda}]^{\frac{1}{\lambda}} / Z
$$
Now, for $w=1/2$, this term factors out and scales the mean by $(\frac{1}{2})^{\frac{1}{\lambda}}$ (similar to the factors in Corollary 5.1).   This $\lambda$ representation is closely related to the $\alpha$-divergence [1, Sec 4-5] (c.f. $\chi^2$-divergence in Eq. 10, see below)
This may be frivolous, or may be useful for the following reasons:

- $\lim_{\lambda \rightarrow 0} h(p(x)) = \log p(x)$, so $\mu^{(\lambda=0)}_w = p(x)^{1-w} q(x)^w / Z$, matching geometric averages or classifer-free guidance
    - for $w=1/2$, we get an `idempotent version' of the product ($\wedge$), i.e. $\mu^{(\lambda=0)}_{\frac{1}{2}} = p(x)^{\frac{1}{2}} q(x)^{\frac{1}{2}}/Z$
- Continuing with $w=\frac{1}{2}$ and emphasizing the $\lambda \rightarrow -\lambda$ relationship as in authors' Alg 1 Line 4, the product now fits nicely into the rest of the special cases described by the authors:
    - max:  $\lambda \rightarrow \infty$
    - mixture (MoE disjunction)  $\lambda = 1$
    - Dombi disjunction:  $\lambda > 0$
    - product (PoE conjunction) $\lambda \rightarrow 0$
    - Dombi conjunction:  $\lambda < 0$
    - harmonic mean:  $\lambda = -1$
    - min:  $\lambda \rightarrow -\infty$
- Perhaps it is not obvious to introduce $w \neq \frac{1}{2}$ from the fuzzy logic perspective (?), but quasi-arithmetic mixtures could expand the search space for effective model combinations.    
- Logic is probably suitable for distributivity and chaining operations though.


*Garipov et. al 2023 Contrast?*

Given my concerns about negation as presented (and for fun), consider a different way to derive Garipov et. al 2023's contrast operator.   We want $p(x)$ 'and' $p(x)^2/q(x)$ (high p, low q, needs to be integrable)

$$
p(x) \wedge_{\lambda} \frac{p(x)^2}{q(x)}  = \frac{p(x)^3 / q(x)}{[ p(x)^{\lambda} + p(x)^{\lambda} \frac{p(x)^{\lambda}}{q(x)^{\lambda}}]^{\frac{1}{\lambda}}} = \frac{p(x)^3 / q(x)}{[  \frac{p(x)^{\lambda}}{q(x)^{\lambda}}\left( q(x)^{\lambda} + p(x)^{\lambda} \right)]^{\frac{1}{\lambda}}} =  \frac{p(x)^2}{\left( q(x)^{\lambda} + p(x)^{\lambda} \right)^{\frac{1}{\lambda}}}
$$
which matches for $\lambda = 1$.





[1] Amari 2007, Integration of Stochastic Models by Minimizing α-Divergence

[2] Masrani et. al 2021, q-paths: Generalizing Geometric Average Paths using Power Means

### Soundness
2

### Presentation
3

### Contribution
4
