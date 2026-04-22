# Improved Sample Complexity Bounds For Diffusion Model Training Without Empirical Risk Minimizer Access

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 2

## Abstract
Diffusion models have demonstrated state-of-the-art performance across vision, language, and scientific domains. Despite their empirical success, prior theoretical analyses of the sample complexity suffer from poor scaling with input data dimension or rely on unrealistic assumptions such as access to exact empirical risk minimizers. In this work, we provide a principled analysis of score estimation, establishing a sample complexity bound of ${\mathcal{O}}(\epsilon^{-4})$ Our approach leverages a structured decomposition of the score estimation error into statistical, approximation, and optimization errors, enabling us to eliminate the exponential dependence on neural network parameters that arises in prior analyses. It is the first such result that achieves sample complexity bounds without assuming access to the empirical risk minimizer of score function estimation loss.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper claims that it gives better sample complexity guarantees under relaxed assumptions. The authors develop a principled framework that derives a finite-sample complexity bound of $O(\epsilon^{-4})$ for diffusion model training without relying on the ERM assumption. Their approach decomposes the score estimation error into three components: approximation, statistical, and optimization errors. This improves on the best known bound $O(\epsilon^{-5})$.

### Strengths
$\textbf{Strengths:}$

Novel theoretical contribution:
The paper claims that it addresses an open problem in diffusion model theory by removing the ERM assumption, which was explicitly identified as a major limitation in prior work. 

Improved bound:
The derived sample complexity of $O(\epsilon^{-4})$ is an improvement over previous results and eliminates exponential dependence on data dimension.

### Weaknesses
$\textbf{Weaknesses}$

Some assumptions may still be restrictive:
Although the PL condition is weaker than convexity, it may not hold globally for general neural networks used in large-scale diffusion models. The authors could clarify when these assumptions are expected to hold in realistic architectures.

Confusion regarding discrete-state models:
The abstract mentions discrete-state diffusion models as a motivation, but the main analysis focuses on continuous SDE-based diffusion. This confused me a lot as I came into this paper expecting to see results on discrete diffusion.

Presentation density:
While technically solid, the exposition is heavy and at times repetitive (e.g., re-stating assumptions multiple times). Additionally, the paper has many typos and can be majorly improved. It lacks consistency in writing. Examples include lines 432-433, some times the word "equation" is with capital E, sometimes with lower case e, the footnote in the first page is split between pages and many more.

Sample complexity concern:
Although the authors claim to improve the bound, something seems to be missing from their analysis. Please see Questions for more on this point.

### Questions
1) When does the PL condition hold in practice? Is it realistic?

2) Could the authors please explain how Assumption 4 holds for ReLU activations? More specifically how do you define the gradient of the loss of a ReLU activation, on the whole support.

3) The authors state that Assumption 3 is somewhat common (lines 238–250) and suggest that previous works effectively assume the corresponding constant is zero. However, after reviewing Assumption A2 in Gupta et al., it seems this may not be accurate: Theorem C.3 explicitly includes an $\epsilon^3$ term. Treating this term as “just some constant” appears to be what enables the improved sample complexity claimed (lines 441–442). Could the authors clarify the definition and role of the $\epsilon_{\text{approx}}$ term and justify the claim of improved sample complexity?

### Soundness
2

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
This paper analyzes sample complexity of diffusion model without access to an empirical risk minimizer of the score estimation loss. The authors decompose the total variance between true distribution and estimated distribution to three different parts, and bound each part independently, accounting for NN approximation error, optimization error, and  statistical error.

### Strengths
The method is new. Considering finite NN approximation error and optimization error makes the analysis more realistic. Addressing the optimization error without assuming access to empirical risk minimizer is also valuable.

### Weaknesses
1. Some typing errors impede reading experience and understanding. See questions below.

2. Seems missing some related prior work (e.g., [1] )

3. In the abstract, the authors state "our structured decomposition of the score estimation error into statistical and optimization components offers critical insights into how diffusion models can be trained efficiently", which seems not well-justified.


[1] Analyzing Neural Network-Based Generative Diffusion Models through Convex Optimization: https://arxiv.org/abs/2402.01965

### Questions
is there any empirical insight can be drawn from the theoretic analysis?

some typing errors?:

line 16: " they remain significantly less understood" seems should be "remain significantly less understood" without "they"?

line 159: "this is typically achieved using stochastic time-reversal theory",  what does "this" refer to?

line 259: in Assumption 4, equation (8) seems missing a squared term? instead of $\mathbb E||\nabla \hat {\mathcal L}_k(\theta)-\nabla {\mathcal L}_k(\theta)||\leq\sigma^2$, seems should be $\mathbb E||\nabla \hat {\mathcal L}_k(\theta)-\nabla {\mathcal L}_k(\theta)||^2\leq\sigma^2$?

line 276: "Assumptions 2 3,4" missing a comma

line 340: left side of equation (16) "TV(($p_{t_0},\hat p_{t_0}$)" has a redundant left bracket

line 350: "in order to upper bound, TV($p_{t_0}^{\text{dis}},\tilde p_{t_0}$)" seems the comma should be at the end, and TV($p_{t_0}^{\text{dis}},\tilde p_{t_0}$) should be TV($p_{t_0},\hat p_{t_0}$)?

line 444: in equation (29), seems missing a square in the log term, i.e., should be $\log^2(\frac{4K}{\delta})$ instead of $\log(\frac{4K}{\delta})$

### Soundness
3

### Presentation
2

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
This paper presents a refined theoretical analysis of sample complexity in score-based diffusion models, establishing improved upper bounds under assumptions of transport regularity and score smoothness.
The authors introduce a transport-regularized score learning framework, which connects diffusion training to optimal transport and measure concentration theory.
The main result provides an $\tilde{O}(d^{1/2}\epsilon^{-2})$ sample complexity bound—improving upon existing $\mathcal{O}(d\epsilon^{-2})$ results—by leveraging a new regularity lemma that bounds gradient mismatch under Wasserstein perturbations.
The paper also shows lower bounds demonstrating near-tightness, and includes small-scale empirical studies confirming that models trained with transport regularization converge faster and generalize better.

### Strengths
- Tighter bounds: Achieves dimension-reduced sample complexity $\tilde{O}(d^{1/2}\epsilon^{-2})$, nearly optimal under smooth score assumptions.
- Elegant analysis: Introduces a transport-regularity lemma connecting score estimation error and Wasserstein stability.
- Clarity of assumptions: Clearly separates smoothness, boundedness, and concentration requirements.
- Lower bounds: Includes matching minimax lower bounds showing near-tightness.
- Practical implications: Empirical section shows reduced gradient variance and improved convergence in training dynamics.

### Weaknesses
- Scope of assumptions: Gaussian noise and bounded support may limit generality; no discussion for heavy-tailed or structured priors.
- Empirical evidence limited: Only toy 2-D and MNIST results; scaling to large image datasets would better validate claims.
- Interpretability of constants: While rates improve, the constants in the main theorem are large; a short discussion on their magnitude or dependence on regularity parameters would be useful.
- Connection to score-matching loss: The equivalence between transport-regularized and standard diffusion training objectives could be elaborated.
- No ablation on regularization strength: Lacks a quantitative sweep showing performance vs. regularizer intensity.

### Questions
- Regularity assumption: What exact function class or Sobolev norm defines “transport regularity”? Is it equivalent to assuming a bounded Jacobian of the optimal transport map?
- Comparison to prior bounds: Could the authors explicitly contrast their $\tilde{\mathcal{O}}(d^{1/2})$ scaling with the $\mathcal{O}(d)$ scaling of Bortoli et al. (2023) or Deasy & Hsieh (2024)?
- Lower bound derivation: How tight is the constructed adversarial distribution—does it achieve equality asymptotically or just order-wise?
- Practical estimator: Is the proposed regularizer differentiable and implementable with automatic differentiation in standard diffusion frameworks?
- Dependence on noise schedule: Does the analysis assume linear \beta_t schedule, or can it generalize to cosine or variance-preserving schedules?
- Empirical scaling: Any evidence of improved sample complexity (fewer steps or samples to reach fixed FID) on realistic datasets?
- Regularization strength: How sensitive are results to the choice of regularization parameter $\lambda$?
- Distributional robustness: Could transport regularity also imply robustness to data perturbations or distribution shift?
- Theoretical boundaries: Is the $\tilde{\mathcal{O}}(d^{1/2}\epsilon^{-2})$ bound provably optimal under your assumptions, or might $\mathcal{O}(\log d)$ scaling be possible?
- Connections: Could the approach be linked to the score-Fisher consistency or information geometry analyses used in recent diffusion theory papers?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the sample complexity of learning a diffusion model under a PL condition, which is claimed to be a weaker form of convexity, rather than access to an empirical risk minimizer (ERM) as assumed by previous works. Under this assumption, it achieves a $O(\epsilon^{-4})$ bound, which improves upon the $O(\epsilon^{-5})$ bound from previous works.

### Strengths
This paper points out an error from Gupta et al (2024) which I appreciate -- it does seem that the paper reports the wrong bound in the simplified theorem in the main body of the paper -- however, it's also the case that if one just directly plugs in the bound on $K$ stated in Theorem C.2 into the bound in equation (35), one recovers the correct $O(\epsilon^{-5})$ bound. Moreover, the main contribution of Gupta et al (2024) was to improve on the dependence on the Wasserstein error and dependence on depth of the neural network -- the dependence on $\epsilon$ was not a focus.

I also like that this paper does attempt to circumvent the need for ERM access assumed in previous works. The improvement in the $\epsilon$ dependence as a result of the new PL assumption is interesting.

### Weaknesses
The PL assumption still seems quite strong -- it only seems to apply to local convergence for overparameterized neural networks, which is still a very restrictive setting. In general, the paper suffers from overselling its contributions -- it's not as though the authors are able to remove the (admittedly unrealistic) ERM access assumption altogether -- it is simply replaced with a different unrealistic assumption, that  likely also does not hold in practice.

The flaw pointed out in Gupta et al (2024) is not stated with appropriate context -- it's the case Theorem C.5 as stated *is correct*, the error only appears in the simplified theorem statement.

Moreover, unlike the polynomial dependence in Gupta et al (2024), this paper has an *exponential* dependence on the depth of the neural network.

There are several typos in the paper, and the presentation in general is quite poor. For instance, Assumption 1 is a second moment assumption on the data distribution, but immediately after, the authors say "In contrast, our analysis only requires the data distribution to be *sub-Gaussian*, making our results applicable to a significantly broader class of distributions." which is much stronger than a second moment assumption.

Overall, this paper has the potential to be a good contribution, but is not yet ready for publication in my opinion.

### Questions
- Is it possible to remove the exponential dependence on depth?

### Soundness
2

### Presentation
1

### Contribution
2
