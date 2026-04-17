# Edge of Stochastic Stability: Revisiting the Edge of Stability for SGD

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Recent findings by Cohen et al. demonstrate that when training neural networks with full-batch gradient descent with step size $\eta$, the largest eigenvalue~$\lambda_{\max}$ of the full-batch Hessian consistently stabilizes around $2/\eta$. 
    These results have significant implications for convergence and generalization. 
    This, however, is not the case of mini-batch stochastic gradient descent (SGD), limiting the broader applicability of its consequences.
    We show that SGD trains in a different regime we term Edge of Stochastic Stability (EoSS).
    In this regime, what stabilizes at $2/\eta$ is Batch Sharpness: the expected directional curvature of mini-batch Hessians along their corresponding stochastic gradients.
    As a consequence, $\lambda_{\max}$---which is generally smaller than Batch Sharpness---is suppressed, aligning with the long-standing empirical observation that smaller batches and larger step sizes favor flatter minima.
    We further discuss implications for mathematical modeling of SGD trajectories.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In the past, it has been shown that gradient descent (GD) exhibits a phenomenon known as the edge of stability. This refers to the empirical observations that the iterates of GD typically experience a progressive sharpening effect, where the sharpness (maximal eigenvalue of the Hessian) increases until it hits 2/step size, which is the stability threshold of GD, and then it hovers just above it with some oscillations until the end. This paper aims to generalize the edge of stability phenomenon to the stochastic setting (SGD), where it finds a new notion of sharpness that exhibits similar properties to the deterministic setting. They verify this with experiments.

### Strengths
The main strength of this paper lies in the following insight. Prior work [R1] has established that stable training for SGD with a constant step size $ \\eta $ occurs when the Gradient Noise Interaction (GNI) satisfies $ \\text{GNI} = \\frac{2}{\\eta} $. This paper introduces a new observation, arguing that GNI may not be the appropriate measure of stability. Specifically, whenever $ \\theta\_t $ converges in probability (*i.e.*, converges to a stationary distribution), for sufficiently large $ t $ we have
$ \\mathbb{E}[L(\\theta\_t)] \\approx \\mathbb{E}[L(\\theta\_{t+1})] = \\mathbb{E}[\\mathbb{E}[L(\\theta\_{t+1}) \\mid \\theta\_t]]. $
In other words, the stability condition implied by the GNI definition holds in expectation, so that, on average, $ \\text{GNI} \\approx \\frac{2}{\\eta} $. However, this relationship is independent of curvature, which is central to the deterministic EoS.

Additionally, I think that classifying the origin of instabilities (gradient noise or curvature) in SGD is interesting.

**References:**\
[R1] - Sungyoon Lee and Cheongjae Jang. A new characterization of the edge of stability based on a sharpness measure aware of batch gradient distribution.

### Weaknesses
**Main issues:**
* The paper claims to generalize the edge-of-stability (EoS) phenomenon observed in gradient descent (GD), *i.e.* $ \\lambda\_{\\max} \approx \\frac{2}{\\eta}  $, to the stochastic setting of SGD. However, this claim is incorrect. A straightforward way to see this is by considering the full-batch case: under this setting, the proposed measure fails to recover the deterministic EoS condition. This discrepancy stems from the fact that batch sharpness is derived from the descent lemma on a random batch loss, which characterizes a single gradient step. In contrast, the deterministic EoS condition in GD emerges from multi-step dynamics, where the iterates align with the sharpest curvature direction near the minimum. Such long-term dynamical behavior cannot be captured by the descent lemma alone. Therefore, the purported generalization of EoS to the stochastic case is not merely an overstatement—it is a false claim.

* The paper does not provide a clear explanation or justification for why the proposed batch sharpness measure is of practical or theoretical interest. 
The only result presented in the main paper regarding this measure is Theorem 1. According to Theorem 1 if the batch sharpness happens to be too large, then consecutive steps of SGD using the same batch lead, on average, to a rapid increase in the gradient of the loss over that batch. However, in standard SGD, the probability of sampling the same batch in consecutive iterations is effectively zero. This raises the question: why should this result matter in practice? The natural assumption is that the batches across iterations are independent. Furthermore, the theorem does not specify the exact condition—it is stated only up to an unspecified absolute constant, which could potentially be large. This makes the claim that batch sharpness $ \\approx \\frac{2}{\\eta} $ rather peculiar. Finally, it is hard for me to see how the batch sharpness can govern the stability of SGD.

* This paper considers a quadratic approximation of the loss (see Def. 1), namely, linearized dynamics of SGD. This **precise** setting was already studied in the past, and the exact stability condition in a closed-form expression was given in [R2]. Relating to the first point here, in contrast to this paper, [R2] gives a multi-step analysis. Importantly, the stability threshold in [R2] does recover the stability condition in the deterministic case, *i.e.*, it generalizes the stability threshold of GD to SGD (while the approach here fails). Moreover, many calculations in the appendix overlap with [R2]. For example, the evolution of the covariance matrix in App. C and D. Therefore, it seems that the novelty of the current manuscript is limited, and the presented results are weaker than those of published literature. Moreover, the use of the descent lemma to characterize stability in SGD was done by prior work [R1], and the proposed measure of batch sharpness is very similar to the terms in [R1]. This fact further diminishes the novelty of this paper.

**Presentation and writing issues:**
The paper lacks clarity and rigor. Here are a few examples that illustrate these issues:
* In Definition 3, we have the following expression 
$$ \\text{Batch Sharpness}:= \\mathbb{E} \\left[ \\frac{( \\nabla L\_B)^T \\nabla^2 L\_B \nabla L\_B  }{\\| \nabla L_B \\|  } \\right] $$
However, in the appendix line 1867, also in equation (49), and in the definitions and the derivations in App. F where $ \\text{Batch Sharpness}:= \\frac{\\mathcal{A}}{\\mathcal{C}} $ we have
$$ \\text{Batch Sharpness}:= \\frac{\\mathbb{E} [ ( \\nabla L\_B)^T \\nabla^2 L\_B \nabla L\_B ] }{\\mathbb{E} [ \\| \nabla L_B \\| ] }$$
These two expressions are different. I tend to believe that the latter is the correct expression. 
* Although the stability measures (GNI and batch sharpness) have clear mathematical formulas, no mathematical definitions are given to Type 1 and Type 2 oscillations.
* Definition 1 is unclear (maybe a fault?)


**Reference**:\
[R2] - Rotem Mulayoff, Tomer Michaeli. Exact Mean Square Linear Stability Analysis for SGD

### Questions
1)  [R1] used a notion of stability defined as $\\mathbb{E} [L(\\theta\_{t+1}) | \\theta\_t ] \leq L(\\theta\_t) $. This translates to a stability condition of (if and only if)
$$ \\text{GNI}: = \\frac{\\mathbb{E} [ ( \\nabla L_B)^T \\nabla^2 L \nabla L_B ] }{\mathbb{E} [ \\| \\nabla L \\| ]  } \leq \frac{2}{\eta},$$
where the edge of stability is obtained in equality. In the paper, you showed that this notion of stability (condition or definition, they are equivalent) is misleading and fails to capture the right characteristics of EoS.\
On the other hand, your underlying definition of stability is $ \\mathbb{E} [L\_B(\\theta\_{t+1}) | \\theta\_t ] \\leq \\mathbb{E} [L_B(\\theta\_t) | \\theta\_t ] $, and the corresponding stability condition is (if and only if)
$$ \\text{Batch Sharpness}:= \\frac{\\mathbb{E} [ ( \\nabla L\_B)^T \\nabla^2 L\_B \nabla L\_B ] }{\\mathbb{E} [ \\| \nabla L_B \\| ] } \\leq \\frac{2}{\\eta}.$$
My question is the following. Why does your underlying definition of stability make sense? Please base your answer on the definition.

2) Why do you claim that batch sharpness governs the stability of SGD?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the regime that is coined the edge of stochastic stability for the mini-batch SGD.  In this regime, what stabilizes at $2/\eta$ is batch sharpness, defined as the expected directional curvature of mini-batch Hessians along their corresponding stochastic gradients. This leads to the observation that is in line with the well-known fact that smaller batches and larger stepsizes favor flatter minima.

### Strengths
The paper contains many fine details and extensive discussions on many different aspects, as well as the literature background. The research problem that is proposed in the paper is an interesting and important topic worth investigating.

### Weaknesses
(1) The presentation of the paper needs to be improved. I find it not that easy to follow. The Appendix is super long, and contains many random topics that do not seem to capture the essence of the major contributions of the paper. 

(2) When I read the proofs in the Appendix, there are too many places you used $\approx$ which should be made more rigorous by using Big O notation or other notations that can be made rigorous or at least you should make the meaning of $\approx$ more transparent and explicit.

### Questions
(1) In terms of literature review, since previous works on SGD stability is most relevant. I actually did not see you have more discussions until I see Appendix B. Perhaps you should mention in the main paper that more details will be provided in Appendix B.

(2) The statement of Lemma 1 is not very rigorous. You should specify the meaning of $\approx$ or simply avoid using $\approx$.

(3) I would suggest you to add footnote 7 at the end of the sentence as the ratio, instead of inside equation (3).

(4) On page 22, you discussed by Taylor expansion $\nabla L(x)\approx\mathcal{H}(x-x^{\ast})$, but if you use an approximation here, it is not clear to me why you get an equality in $\mathbb{E}_{k}[\Vert\nabla L(x_{k})\Vert^{2}]=\text{Tr}(\mathcal{H}\Sigma_{x}\mathcal{H})$.

(5) On page 25, you should make the meaning of $\approx$ in $\Sigma_{x}\approx\frac{\eta}{b}\mathcal{K}^{\dagger}(\Sigma_{g})$
more transparent or simply avoid using $\approx$ notation.

(6) In the statement of Theorem 1, a absolute should be an absolute.

(7) On page 27, in the last equation, you had an extra $)$ which is a typo.

(8) On page 33, Remind that should be Recall that.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors discover a counterpart of the edge-of-stability (EoS) phenomenon by Cohen et al. 2021 in the case of training under SGD, which they call the edge of stochastic stability (EoSS). The authors' augment the curvature-based oscillations of EoS with (gradient) noise-driven oscillations. In this context they name these type-1 (noise-driven) oscillations and type-2 (curvature-driven) oscillations. According to the authors, only the latter is what drives EoS-like behavior under SGD: batch sharpness (BS) stabilizes at $\approx2/\eta$, and sudden increases in BS lead to catapult-like behavior that puts back the iterates in a different basin with BS $\approx2/\eta$. Decreasing the batch size (through increasing BS) or increasing the learning rate (through decreasing $2/\eta$) lead to similar catapult effects, and make the iterates settle in a sharper basin.

### Strengths
- The paper is very well motivated. Its contributions are solid and important. The breakdown of the oscillation in the SGD case is simple, elegant, and very useful. The comparisons with previous work seem sufficient.
- In addition to their main findings, the authors' conclusions re. SGD vs. noisy gradient descent is very useful, and speaks to the potential of their findings to progress the field.

### Weaknesses
- Although the paper is about extending EoS to SGD, and involves a lot of comparisons between the two, the authors do not have an authoritative explanation why SGD does not follow EoS. That is, their results show why SGD follows EoSS, but it does not show why/when following EoSS corresponds to not following EoS - in the form of a more specific relationship between batch sharpness and $\lambda_{\max}$. In this sense the paper parallels Cohen et al. 2021 but does not complement it. Experiments at Section 4.1 and Appendix J are attempts at this, but I believe a more explicit discussion of this in the main paper is warranted.
- The paper is sometimes very difficult to read. This is due to a range of issues from conceptual conflicts to more mundane errors. I will provide a main example here and defer the rest to the following section. In Section 4, part 3. Catapults, the authors highlight the batch stochasticity-based spikes in batch sharpness to account for the "catapults". However, their original understanding of catapults seem to include the sharpness-recovering behavior in both EoS and EoSS.

### Questions
- There seems to be a problem with the style file, and the fonts are not compiled as intended
- Alphabetical ordering of citations needed
- Footnotes (3, 4, 7) are made following right after equations, disrupting the reading of the formulas.
- L014: Cohen et al. (2021) consistently states that sharpness stabilizes *above* this value. 
- L015: Cohen et al. (2021) explicitly ignores generalization concerns.
- L039: Items (2) and (3) are unclear.
- Boxed summary:
	- L046: I think at least a brief mention of the notation should be made before the batch sharpness equation is presented.
	- L046: Similar to? The boxed summary should be mostly self-contained
- L052: "implicitly functions as sharpness", "stability for SGD is stability on the mini-batch landscape" sound confusing. I would encourage the authors to slightly expand on these, as their vagueness defeats the purpose of this highlight.
- L063: $\eta>0$?
- L077: What is $\tilde{L}$? Please define.
- L093: First couple of sentences make me think that another paragraph with the title e.g. "Learning rate and gradient-based optimization" can be used.  As Jastrzebski et al. 2020 is about SGD, for example.
- L115: How is this paragraph title different than L091?
- L121: I feel this lengthy quote from Cohen et al. is unnecessary.
- L130: "the answer" to?
- L186: What does "step size does not vanish" mean? Please explain within the text.
- L191: "leaves all the compacts in which..." -> "exits all compact subset of the region in which..." 
- L225: Why not $\mathcal{H}(L)$ to be consistent with $\mathcal{H}(L_B)$?
- L229: "descent lemmas" are unintroduced.
- L266: I do not understand this footnote.
- L270: "Importantly, a reason why..." I do not understand this sentence.
- L302: What does step sharpness mean?
- L313: Referring to a future figure for notation seems suboptimal
- L324: Please increase the axis and legend fonts for figures dramatically
- Minimal typos/errors:
	- L039: Seemingly mistaken use of $L$ instead of $\lambda_{\mathrm{max}}$.
	- L044: "as $\lambda_{\mathrm{max}}$" -> "such as $\lambda_{\mathrm{max}}$"
	- L176: "constraints" -> "constrains"
	- L181: The word "oscillations" neglected from the title?
	- L215: "Dyanmics"
	- L285: "traind"

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
This paper investigates the instability regime of stochastic gradient descent (SGD) in neural network optimization, which has not been sufficiently explained by the existing edge of stability (EoS) framework for gradient descent (GD). The authors distinguish two types of oscillatory behavior during SGD instability: noise-driven and curvature-driven oscillations. The latter, of particular interest, is characterized through batch sharpness, defined as the expectation of directional sharpness of the mini-batch loss along its gradient direction. Experiments demonstrate that, within the edge of stochastic stability (EoSS) regime, batch sharpness consistently hovers around 2/η and provides a potentially better explanation of the catapult effect than gradient-noise interaction (GNI).

### Strengths
- The paper addresses a fundamental and underexplored aspect of deep learning optimization—the instability regime of SGD—and makes an original attempt to distinguish between two forms of oscillation. The proposed concepts of curvature-driven oscillation and batch sharpness are interesting and potentially valuable for understanding SGD dynamics.
- Extensive experiments on CIFAR-10 provide empirical evidence that batch sharpness remains close to 2/η, supporting the proposed hypothesis.

### Weaknesses
- The Introduction and Background sections are disproportionately long (occupying over one-third of the main text), while the intuitive motivation or justification for the definition of batch sharpness, as well as its theoretical connection to the catapult effect, are insufficiently developed. Reducing the background in favor of more focused explanations on these points would improve the paper’s overall clarity and intuition.
- The current definition of the catapult effect is somewhat ambiguous. It is difficult to interpret what exactly constitutes the catapult phenomenon or its observable behavior from the experimental figures. A clearer and more formal definition, along with an analysis of its relationship to batch sharpness and GNI, is needed.

### Questions
- The paper provides a theoretical justification (Theorem 1) showing that when batch sharpness exceeds 2/η, the expected batch gradient squared norm can locally increase exponentially. However, the intuitive motivation for defining batch sharpness as the average of directional sharpness values is not clearly articulated. Why should the average—rather than other statistics such as the maximum or the proportion exceeding 2/η—be considered the most meaningful indicator of instability? Some discussion on this intuition would help readers better understand the rationale behind the definition.
- In Figure 4, batch sharpness drops sharply at several points, yet no explanation is provided. What mechanism causes these abrupt decreases?

### Soundness
2

### Presentation
2

### Contribution
3
