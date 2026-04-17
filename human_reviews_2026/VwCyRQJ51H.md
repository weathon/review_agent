# On the Bayes Inconsistency of Disagreement Discrepancy Surrogates

- Decision: Accept (Poster)
- Scores: 8, 4, 6

## Abstract
Deep neural networks often fail when deployed in real-world contexts due to distribution shift, a critical barrier to building safe and reliable systems. An emerging approach to address this problem relies on _disagreement discrepancy_—a measure of how the disagreement between two models changes under a shifting distribution. The process of maximizing this measure has seen applications in bounding error under shifts, testing for harmful shifts, and training more robust models. However, this optimization involves the non-differentiable zero-one loss, necessitating the use of practical surrogate losses. We prove that existing surrogates for disagreement discrepancy are not Bayes consistent, revealing a fundamental flaw: maximizing these surrogates can fail to maximize the true disagreement discrepancy. To address this, we introduce new theoretical results providing both upper and lower bounds on the optimality gap for such surrogates. Guided by this theory, we propose a novel disagreement loss that, when paired with cross-entropy, yields a provably consistent surrogate for disagreement discrepancy. Empirical evaluations across diverse benchmarks demonstrate that our method provides more accurate and robust estimates of disagreement discrepancy than existing approaches, particularly under challenging adversarial conditions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper considers the problem of measuring maximal discrepancy in covariate shifts in the context of bounding generalization errors, noting that the theory is based on 0-1 losses, but surrogate losses used in practice, while convex, are not consistent. That is, the optimizer of the surrogate loss does not converge to the solution of 0-1 loss due to the choice of the disagreement loss. A consistent disagreement loss is proven and shown to perform better in estimating the actual discrepancy.

### Strengths
The paper is clear and address a perennial problem of understanding how covariate shift can affect classifiers. Overall, the paper is well written.

### Weaknesses
There are a few points that could be made more clear (questions below). 

Additionally, is it possible to use the consistent form as a bound on error to create distributional robustness? I understand the adversarial attack is a step in this direction, but commenting on how this connects to making better unsupervised domain adaption algorithms would be interesting.

### Questions
What is the purpose of minimum in (1) and if it is needed what is it taken with respect to what?

Line 95 The limitations of theory is noted but could be made clearer on why 'practical model classes' are note covered by $\mathcal{H}$-consistency. "While H-consistency provides valuable insights, its limited applicability to practical model classes" It is not until line 493 that deep neural networks are mentioned. 

Line 111 Could be more clear by defining what exactly "this framework" is. I understand it to be the description at line 104–105.

Line 132 The description of a probability "intersected with $\mathcal{X}'$ is strange and imprecise. Why not say it is the conditional distribution for points in the subset $\mathcal{X}'$. 

Line 399 The critic was using the frozen weights. In cases where the reference model like ERM is not aware of the target data, couldn't a better critic be built on top of the backbone optimized in a different unsupervised domain adaption method? I.e. couldn't $f$ be used as a critic for $h$ even though $f$ uses $h'$?

Line 1616 the $\ell_\infty$ should be on both target training and target testing as both are optimized in the two steps of the optimization. 

Line 1621 Couldn't the adjoint/ implicit function method  be used here?

### Soundness
4

### Presentation
4

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
The paper studies the use of surrogate losses to maximize disagreement discrepancy - a quantity used for error bounds under covariate shift, shift detection, and robust training. The authors claim to prove that commonly used disagreement surrogates are not Bayes consistent for multi-class problems and develop upper/lower bounds on the optimality gap. Guided by this analysis, they propose a new disagreement loss, $ ℓ^{ours}_{​dis}(x,y,s)=−log(1−σ(s)_y​) $
which, when paired with cross-entropy agreement loss, yields a surrogate provably Bayes-consistent. Empirical results on a variety of vision benchmarks (WILDS, BREEDS, DomainNet) and adversarially perturbed target data show improved estimation of the disagreement discrepancy and better calibration of the associated error bounds versus prior surrogates.

### Strengths
The paper tackles a fundamental open question in the theory of domain adaptation and covariate shift: whether the surrogate losses used for estimating disagreement discrepancy are theoretically sound.
Few prior works examined the consistency properties of such surrogates; most focused on empirical performance.
The authors provide what appears to be the first demonstration that the popular disagreement surrogates of Rosenfeld & Garg (2023) and Ginsberg et al. (2023) are Bayes-inconsistent in the multi-class (K > 2) case.
This negative result is conceptually important because it challenges implicit assumptions underlying a growing body of empirical research.
The proposed alternative surrogate loss, proven (under stated assumptions) to be Bayes-consistent, gives the work a constructive dimension - turning a theoretical diagnosis into a practical remedy.
While the proofs appear well-structured and follow logically from standard surrogate-consistency frameworks, a full validation would require a detailed check of each derivation and of how the assumptions are applied. The arguments look rigorous and well-motivated, but independent verification is still needed to certify formal correctness.

### Weaknesses
The main theoretical guarantee - Bayes consistency - is asymptotic and does not specify convergence rates or finite-sample behavior.
This makes it unclear how the surrogate behaves with limited data or under model misspecification. Without sample-complexity or generalization bounds, practitioners lack quantitative guidance on when the theoretical improvement translates into measurable gains.

The analysis assumes covariate shift only, i.e., $P(Y \textbar X) $ remains constant across domains. In real-world distribution shifts, label shift, concept drift, or conditional shift frequently occur - cases where the proposed theory may not apply or may even break down. No empirical evidence is provided for robustness beyond this assumption.

The paper largely dismisses H-consistency (consistency under restricted hypothesis classes) as "limited in practice" without detailed justification or experiments comparing H-consistent and Bayes-consistent outcomes. Since DNNs are not universal function approximators in practice, H-consistency might be more relevant than Bayes consistency.

The results depend on the structure of discrete label sets and the softmax parameterization. It is unclear whether the framework extends to regression, structured prediction, or representation-learning contexts where disagreement discrepancy is also used.

While the experimental results are broad in scope, they are not statistically rigorous. Most results are reported as single runs, without standard deviations, making it difficult to assess robustness or reproducibility. The paper would benefit from multi-seed evaluations and statistical significance tests.

Hyperparameter sensitivity is not explored: the method introduces parameter $\alpha$ whose effect on stability and performance remains unclear.

There is no discussion of computational overhead (e.g., runtime, convergence rate) introduced by the new surrogate.
The theoretical contribution is strong, but the empirical evidence supporting its practical relevance remains somewhat limited in depth and statistical reliability.

The paper compares primarily with Rosenfeld & Garg (2023) and Ginsberg et al. (2023), which are relevant but now slightly dated.
More recent 2024–2025 methods tackle disagreement-based discrepancy estimation and should be included for completeness.
Without these baselines, it is difficult to assess whether the proposed surrogate remains competitive with the current state of the art in practical terms.

The lack of up-to-date comparisons weakens the empirical claim of superiority, even though the theoretical advance remains valid.
Overall, the paper’s main weaknesses lie not in its conceptual or theoretical core, which is solid, but in its empirical completeness and clarity of assumptions. Statistical validation is thin, comparisons are slightly outdated, and the theoretical applicability to finite neural critics could be more clearly discussed. These issues do not undermine the core contribution but do limit the paper’s perceived maturity and readiness for publication at the very top tier.

### Questions
1) How general is the proposed framework for difference-of-risk objectives? Could it apply to other discrepancy-based metrics (e.g., fairness gaps, calibration errors), or is it specific to disagreement discrepancy?
2) Theorems 4 and 6 provide lower and upper bounds on optimality gaps. Are these bounds tight in practice, or are they mostly asymptotic constructs with little finite-sample relevance?
3) How sensitive are your results to the choice of the critic’s hypothesis class H? Have you considered the notion of H-consistency or provided any analysis for finite-capacity critics?
4) Does the new surrogate require calibration of $\alpha$ to maintain consistency, or is it theoretically invariant to this hyperparameter?
5) What guided the decision to use a frozen feature extractor and a linear critic head? Would your results change with a trainable or deeper critic? 
6) Can you provide mean ± std results over multiple random seeds for key tables and figures?
7) Have you compared against any of the newer disagreement-based methods?
8) Would it be possible to include a visual example or schematic showing how your surrogate modifies the disagreement landscape compared to Rosenfeld & Garg’s surrogate?
9) Do you expect your theoretical framework to extend naturally to other discrepancy measures?
10) Could your approach provide practical benefits in domains beyond covariate shift?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies disagreement discrepancy, an approach that has been recently popular for distribution shifts problem related to error estimation, testing, and robustness. Earlier works propose to maximise the disagreement discrepancy by employing some surrogate loss function, however the main finding of this paper is that the employed surrogate may not be Bayes consistent, which is necessary. The paper then  proposes a Bayes consistent surrogate for the problem.

### Strengths
1. I think paper makes clear theoretical contribution, which is also important. I've seen the works on disagreement discrepancy before as applied for testing, and have found the application to be hacky. The current contribution studies the soundness of it, and indeed, the finding that prior approaches are not sound is original and of significance. 
2. The paper is also thorough (I like the appendix, although I didn't verify everything too closely), but the decomposition of the risk using density ratios, and the resulting Bayes inconsistency analysis seems correct to me. In that sense, the proposed surrogate is also sound (wonder why do the authors think, that was not already employed in prior works as a disagreement loss, as it seems super intuitive).

### Weaknesses
1. Overall, I do like the paper from the theoretical contribution perspective. However, I'm not sure I followed the experimental section that well. Could authors clarify what is max disagreement discrepancy (in Figure 1) and how it is defined and estimated? I also feel writing wise, the paper assumes too much background on prior work. 
2. Furthermore, I think while the quality of surrogate is the main focus of the paper, the paper does not connect to downstream applications. I'd assume in testing works, the proposed surrogate could help aid better testing power.
3. Additionally, I feel like the new disagreement loss can be unstable as $\sigma(s)_{y}$ approaches 1, which it can as neural networks can get overconfident. Has that been considered before? I assume that could or could not happen, and in fact, maybe the proposed disagreement could even lead to better calibrated probability estimates? I'd be curious to learn some perspectives on this. 

Summary: I think the paper makes clean theoretical contribution, that is useful. But from the practical side, I can have a better assessment if the above questions are addressed in the rebuttal.

### Questions
In Equation 1, what is the $\min$ over?

### Soundness
3

### Presentation
2

### Contribution
2
