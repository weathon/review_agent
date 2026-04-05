=== CALIBRATION EXAMPLE 48 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately signals the topic, but it is incomplete/garbled (“DECOMPO SITION” in the parser output). Ignoring the artifact, the scope is clear: an adversarial attack for tensor ring decomposition.
- The abstract states the problem, the proposed AdaTR and FAG-AdaTR methods, and the broad experimental domains. That said, the key results are still very high-level; it does not quantify the magnitude of gains over ATTR or runtime savings, which matters for judging significance.
- A notable unsupported claim is “develop numerically convergence guarantees” and “Numerical experiments ... validate the attack effectiveness” without any quantitative summary. The abstract implies a general robustness result for TR decomposition, but the paper’s evidence is limited to particular datasets, ranks, and attack settings.

### Introduction & Motivation
- The motivation is relevant and well positioned for ICLR standards: robustness/vulnerability of factorization methods is a meaningful, somewhat underexplored problem.
- The gap is stated reasonably: prior adversarial work on NMF/subspace methods does not directly address tensor ring decomposition, and the ATTR-style symmetric objective can misalign with attack goals.
- The main concern is that the introduction somewhat overstates novelty and generality. Claims like “the first evidence that tensor decomposition models are susceptible to adversarial attacks” are too broad, because the paper only demonstrates attacks on selected ALS-based tensor models under specific experimental setups. It is more accurate to say this is the first systematic adversarial attack study of TR decomposition.
- The introduction also asserts that “all ALS-based tensor decompositions... are entirely driven by the observed input tensor” and therefore vulnerable, but this is more an intuitive claim than a demonstrated theorem. The paper should be careful not to overgeneralize from the TR-ALS setting.

### Method / Approach
- The method is described in enough detail to follow the high-level optimization idea: AdaTR is a bilevel max-min attack where the perturbation maximizes reconstruction error while TR factors are re-fit by ALS; FAG-AdaTR approximates the gradient to reduce backprop overhead.
- However, there are important reproducibility and correctness concerns in the derivation:
  - Eq. (10) introduces the bilevel formulation, but the notation is inconsistent and sometimes ambiguous about whether the inner problem is over factors for a fixed perturbed tensor or about iterative dependence across ALS steps.
  - The transition from the exact gradient-based AdaTR to FAG-AdaTR relies on strong independence assumptions (“we assume G is independent of E,” and later that several terms are independent of E). This is a major approximation, not a minor simplification, and the paper does not characterize when it is valid or how much attack quality is lost.
  - The gradient expression in Eq. (16) is presented as “closed-form,” but given the approximations made in Eq. (13)–(15), it is not clear that this is the gradient of the original objective. The paper should distinguish clearly between the true objective, the surrogate objective, and the approximate gradient used.
- The theoretical claims need more scrutiny:
  - Theorem 1’s condition appears loose and somewhat ad hoc. It claims ATTR can improve reconstruction if \(\epsilon < \delta - \|R_2\|_F\), but the proof seems to manipulate inequalities rather than establish a robust statement about the actual optimization trajectory.
  - Theorem 2 and Theorem 3 claim monotonic ascent, Cauchy convergence, and KKT convergence. These are standard projected-gradient-type results, but they depend on differentiability/Lipschitz assumptions that are nontrivial for ALS-generated mappings \(E \mapsto G^{(T)}(E)\). Those assumptions are stated but not justified. For ALS, differentiability can fail at rank-deficient or non-unique update points.
  - Theorem 4 gives an approximate stationarity bound, but the constant and the dependence on \(\sqrt{\epsilon}\epsilon_g\) are not especially informative for practice.
- Edge cases/failure modes are under-discussed:
  - What happens when the perturbation budget is extremely small, where ATTR may improve performance?
  - What if ALS does not converge within \(T_{\text{in}}\)?
  - How sensitive is the attack to rank mis-specification or initialization?
- The extension claim that the framework “can be directly applied to CP, Tucker, TT” is plausible in spirit, but not demonstrated at the methodological level beyond a brief appendix experiment.

### Experiments & Results
- The experiments generally test the core claim that the attack increases reconstruction error under TR decomposition and transfers to completion/recommendation settings. This is aligned with the paper’s goals.
- That said, several issues prevent the results from fully meeting ICLR’s standard for strong empirical support:
  - **Baselines:** ATTR is the obvious baseline and is included. Gaussian noise is also reasonable. But for the claimed “attack on tensor decomposition” setting, stronger comparisons would help, such as direct gradient attacks without the surrogate approximation, or ablations isolating the effect of the asymmetric objective versus the gradient approximation.
  - **Ablations missing:** The most important missing ablation is a direct comparison between AdaTR and a version that uses the same objective but without the approximate gradient simplification, to quantify the cost/benefit of FAG-AdaTR. Another needed ablation is attack effectiveness versus number of outer/inner ALS iterations, since the method’s effectiveness depends on these settings.
  - **Statistical reporting:** Some tables report mean ± variance, but many results do not report the number of runs, confidence intervals, or significance testing. This weakens claims like “superior” or “approximately 2 times gain.”
  - **Metric consistency:** In the recommendation experiments, Table 2 reports RMSE, Precision@10, Recall@10, but the “best” direction annotations are confusing in places, and the numbers are surprisingly close across some conditions. The paper should more carefully interpret whether the attack meaningfully degrades recommendation utility beyond minor metric shifts.
  - **Transferability claims:** The paper says attacks generated on TR-ALS transfer to all tested TR-based defenses. This is an interesting claim, but it is based on a fairly narrow set of defenses and tasks; it should not be framed as broad transferability without caveats.
  - **Hyperparameter sensitivity:** The paper shows some dependence on \(\epsilon\) and ranks, but there is not enough systematic analysis of sensitivity to learning rate, number of outer iterations, or initialization.
- Some result presentations suggest strong visual degradation, which is useful, but ICLR-level evidence should also include careful quantitative comparisons on multiple seeds and possibly stronger statistical characterization. The reported standard deviations vary, but not uniformly across all experiments.
- The extension to CP/TT/Tucker in Appendix H.4 is useful, but the experiment is thin: only reconstruction error is reported, on resized images, and the setup is not enough to support a broad claim of generality across tensor decompositions.

### Writing & Clarity
- The paper’s main ideas are understandable, but several sections are conceptually difficult because the notation is overloaded and sometimes inconsistent.
- The method section would benefit from clearer separation among: the original TR-ALS objective, the ATTR baseline, the exact AdaTR objective, and the approximate FAG-AdaTR surrogate. Right now these are easy to conflate.
- The proofs are difficult to follow, not only because of parser artifacts but also because some steps are too terse for the nontrivial assumptions being made.
- Figures and tables generally support the narrative, but the paper would benefit from a more careful explanation of what each metric means in each task. In particular, the connection between reconstruction error on images/videos and recommendation utility is asserted but not always clearly justified.

### Limitations & Broader Impact
- The paper includes an ethics statement and mentions dual-use concerns, which is good.
- However, the limitations are underdeveloped. Important limitations that should be acknowledged:
  - The attack is white-box and relies on differentiating through ALS iterations or an approximation thereof.
  - The guarantees depend on smoothness/differentiability assumptions that may not hold in practice for all ALS runs.
  - The empirical evaluation is limited to a small set of datasets and tensor formats.
  - The threat model is narrow: bounded additive perturbations to the observed tensor. Real-world adversaries may have different constraints.
- Broader impact: this work is clearly dual-use. On one hand, it advances understanding of robustness of tensor methods; on the other, it can be used to degrade downstream systems such as recommendation and compression pipelines. The paper mentions responsible deployment, but it would be stronger if it discussed mitigation strategies or defensive implications more concretely.
- One missed broader point is that demonstrating attacks on tensor decompositions may motivate robust tensor methods; the paper touches on this only indirectly through ATTR.

### Overall Assessment
This is a timely and technically interesting paper that identifies a plausible vulnerability in tensor ring decomposition and proposes a reasonably articulated attack framework with a lighter-weight approximation. The empirical results suggest the attack can substantially worsen reconstruction and downstream metrics in the studied settings. However, for ICLR’s bar, the main weaknesses are the heavy reliance on approximations in the method, the incomplete justification of theoretical assumptions, and the limited empirical breadth/ablation depth needed to substantiate the strongest claims. I think the contribution is promising and likely publishable in a stronger form, but in its current state it does not yet fully convince me that the method is both theoretically solid and empirically comprehensive enough for acceptance at ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper studies adversarial perturbations against tensor ring (TR) decomposition and proposes AdaTR, a bilevel max-min attack that seeks perturbations maximizing reconstruction error under a Frobenius-norm budget. To reduce computational overhead, it also introduces FAG-AdaTR, an approximate-gradient variant with convergence claims, and evaluates both methods on tensor decomposition, completion, and recommendation tasks using images, videos, and recommender datasets.

### Strengths
1. **Timely and underexplored problem setting.** The paper targets the robustness of tensor ring decomposition to adversarial perturbations, which is a relatively underexamined topic compared with adversarial attacks on classifiers or matrix factorization. This is a meaningful direction for ICLR, especially given the growing use of low-rank tensor methods in vision and recommendation.
2. **Clear attack formulation tailored to decomposition.** The authors identify a limitation of the ATTR-style symmetric objective and instead define an asymmetric bilevel objective that directly maximizes reconstruction error of the TR factors. This is a conceptually reasonable adaptation of adversarial optimization to a nonparametric decomposition setting.
3. **Practical acceleration idea.** FAG-AdaTR attempts to reduce memory/computation costs by replacing full backpropagation through TR-ALS with a closed-form approximate gradient. The paper also reports runtime/memory comparisons suggesting a substantial efficiency gain.
4. **Broad empirical scope.** The experiments span color images, videos, tensor completion, and recommendation, and compare against Gaussian noise and ATTR across several defenses. The appendix additionally includes results for Tucker, CP, and TT, which strengthens the claim that the framework is broadly applicable.
5. **Some theoretical analysis is provided.** The paper includes convergence-style results for AdaTR and FAG-AdaTR and discusses why the ATTR formulation may sometimes improve performance under small perturbation budgets.

### Weaknesses
1. **Novelty is moderate rather than strong for ICLR standards.** The core idea is largely an adaptation of existing adversarial matrix factorization work (ATNMF/ATTR) to TR decomposition, plus an approximate-gradient variant. While useful, the main conceptual leap appears incremental, and the paper does not clearly establish a deeper new principle beyond “replace matrix factorization with TR.”
2. **Theoretical claims rely on strong assumptions and are not fully convincing.** The convergence proofs assume differentiability, bounded Jacobians, bounded iterates, smoothness, and Lipschitz properties that are not obviously guaranteed for TR-ALS with pseudoinverses and alternating updates. The approximate-gradient theorem is also only as good as the mismatch bound, which is assumed rather than derived.
3. **Attack evaluation is limited in threat-model rigor.** The paper primarily uses white-box optimization against TR-ALS with transfer to other TR-based methods. However, it is not fully clear how standardized the threat model is, whether perturbations are perceptually constrained for image/video settings, and how the attack compares to stronger generic optimization-based baselines beyond ATTR and Gaussian noise.
4. **Experimental design and reporting have gaps.** The paper reports many metrics, but key details are sometimes underspecified: attack hyperparameters, stopping criteria, number of restarts, and sensitivity to random seeds are not sufficiently analyzed in the main text. The results often emphasize “best/worst” tables without enough statistical analysis or ablation studies.
5. **Clarity of presentation is uneven.** The mathematical exposition is hard to follow in places, especially around the bilevel objective, the role of the intrinsic mapping from perturbation to factors, and the derivation of the approximate gradient. Even accounting for parser artifacts, the paper itself would benefit from tighter notation and a more careful explanation of the algorithmic steps.
6. **The generality claim is not fully substantiated.** The paper says the framework can be applied to CP/Tucker/TT, but the main text focuses on TR and the extension results are placed in the appendix. There is not enough analysis to show that the method works equally well across qualitatively different tensor models, or that the gradient approximations behave similarly there.
7. **Potential inconsistency in optimization interpretation.** The attack objective is presented as maximizing reconstruction error, but some experimental sections and baselines appear framed as “defense” or “adversarial training” in ways that could confuse the distinction between attacker and defender, especially regarding ATTR and the “collapse” phenomenon.

### Novelty & Significance
**Novelty:** Moderate. The paper’s main contribution is a domain-specific adaptation of adversarial optimization to TR decomposition, along with an efficiency-oriented approximation. This is useful, but it feels like a relatively direct extension of existing adversarial factorization ideas rather than a major methodological breakthrough.

**Significance:** Moderate. If the claims hold, the work highlights a real vulnerability in a class of tensor methods used in imaging and recommendation, which is relevant to robustness research. However, for ICLR’s acceptance bar, the significance is somewhat limited by the narrowness of the attack setting and the absence of stronger evidence that the approach changes how the community should think about robustness in tensor methods.

**Clarity:** Below average to moderate. The high-level story is understandable, but the technical presentation is dense and occasionally hard to parse. The paper would benefit from a cleaner exposition of the optimization problem and a more explicit step-by-step description of why the approximate gradient is valid.

**Reproducibility:** Moderate. The paper includes algorithmic pseudocode, datasets, and hardware information, which is helpful. But reproducibility is weakened by missing details on hyperparameter selection, attack restarts, seed variance, and the exact implementation of the TR-ALS inner loop and gradient approximation.

**Overall ICLR fit:** Borderline. The paper is relevant and technically grounded, but ICLR typically expects either a clearly novel learning method, a strong conceptual insight, or a broadly impactful empirical result. Here, the contribution is interesting but somewhat incremental, and the theoretical/experimental evidence does not fully elevate it to a clearly above-bar acceptance case.

### Suggestions for Improvement
1. **Strengthen the novelty argument.** Explicitly distinguish AdaTR from prior adversarial factorization work by isolating the genuinely new ingredient: why the asymmetric objective is fundamentally better for tensor decomposition, not just a TR instantiation of ATNMF.
2. **Provide stronger baselines.** Compare against more general optimization-based attacks, including multi-start projected gradient methods and any natural adaptation of black-box or gradient-free attacks for tensor decomposition.
3. **Add ablations.** Quantify the impact of each design choice: asymmetric objective vs. symmetric ATTR-style objective, full backprop vs. approximate gradient, number of inner ALS steps, and dependence on rank and budget.
4. **Clarify the threat model.** State exactly what is white-box, what is transferred, what constraints are used for image/video settings, and whether perturbations are perceptually bounded or only norm-bounded.
5. **Improve theoretical rigor or narrow the claims.** Either justify the assumptions more carefully or present the results as conditional convergence statements with clearer scope. In particular, explain whether the assumptions are realistic for practical TR-ALS.
6. **Report robustness and variance more thoroughly.** Include confidence intervals, multiple random seeds, and sensitivity analyses for key hyperparameters and budgets. This would help establish that the attack is consistently effective.
7. **Tighten the presentation.** Rewrite the optimization section with cleaner notation, a concise derivation flow, and a schematic showing the dependency between perturbation updates and ALS factor updates.
8. **Demonstrate broader applicability more convincingly.** If generality to CP/Tucker/TT is an important claim, move those results into the main paper or add a focused comparison showing that the same attack logic truly transfers across decomposition families.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison to standard first-order adversarial optimization baselines (e.g., PGD-style attack, projected gradient ascent on the input tensor, and random search under the same budget). Without this, it is unclear whether AdaTR is actually better than generic black-box/white-box input attacks or just specialized to TR-ALS.  
2. Evaluate against stronger and more relevant decomposition baselines on the same attack objective: CP-ALS, Tucker-ALS, TT-ALS, and ideally robust variants on all three tasks, not just a small appendix extension. ICLR will expect evidence that the method is not TR-specific and that the claimed “general ALS-based tensor model” framing is supported.  
3. Add an ablation isolating the contribution of the bilevel/asymmetric formulation versus the approximate gradient shortcut in FAG-AdaTR. Right now it is not clear whether the gains come from the objective, the optimization scheme, or simply more attack iterations/compute.  
4. Include attack-vs-budget curves with multiple seeds and statistical tests for all main tasks, not just a few tables and single-image plots. The paper claims vulnerability and consistent transferability, but the current evidence is too narrow to establish robustness of the effect.  
5. Compare against stronger defenses or retraining-based robustification under the same compute/data regime, including whether adversarial training on TR or input smoothing materially reduces attack success. Without defense-side evidence, the claim that decomposition is “truly vulnerable” is incomplete.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify the discrepancy between the surrogate objective used for optimization and the true reconstruction error after ALS convergence. The paper’s main method depends on this bilevel approximation, so readers need to know when the surrogate is faithful and when it fails.  
2. Analyze sensitivity to TR rank, initialization, and stopping criteria for both attack and defense. ICLR reviewers will likely question whether the reported gains are stable or just artifacts of a favorable rank/initialization setting.  
3. Provide a rigorous empirical validation of the “FAG-AdaTR approximate gradient” assumption by measuring gradient error, cosine similarity with true gradients, and how that error affects attack success. As written, the approximation is only justified by outcomes, not by evidence that the gradient is accurate enough.  
4. Examine transferability more carefully: attacks generated on TR-ALS are claimed to transfer across defenses, but the paper does not explain when transfer works or fails. A breakdown by defense type and perturbation budget is needed to trust the generality claim.  
5. Clarify the actual threat model and feasibility of the perturbation budgets in the visual domains. The paper uses budgets that may produce visible corruption, so it needs a stronger analysis of perceptual stealth and practical attack constraints.

### Visualizations & Case Studies
1. Show side-by-side reconstructions for the same input under clean, Gaussian noise, ATTR, AdaTR, and FAG-AdaTR, along with residual/error heatmaps. This would expose whether the attack destroys global structure or merely adds obvious pixel artifacts.  
2. Plot optimization trajectories of attack loss, true reconstruction error, and perturbation norm over iterations. This would reveal whether AdaTR is genuinely maximizing the intended objective or just exploiting transient ALS instability.  
3. Include a case study where ATTR appears to improve performance at small budgets, then show how AdaTR avoids that collapse on the same examples. This would directly test the paper’s central claim about the flaw in symmetric min-max training.  
4. Add visualizations of gradient directions or perturbation spectra/frequency content. That would reveal whether the attack is using meaningful structure or just high-frequency noise that may not survive realistic preprocessing.  
5. For recommendation, show per-user/per-item ranking changes before and after attack, not just aggregate RMSE/Precision/Recall. This would make it clear whether the attack meaningfully changes top-k behavior or only shifts global error metrics.

### Obvious Next Steps
1. Move beyond showing that TR is vulnerable and demonstrate a defense or mitigation strategy tailored to the attack. For an ICLR paper, identifying a weakness is not enough unless the paper also clarifies how the vulnerability can be used or countered.  
2. Generalize the method into a unified attack framework with a cleaner theoretical treatment for CP/Tucker/TT rather than a TR-centric derivation plus appendix extensions. The current contribution is too narrow relative to the broad claim of attacking tensor decompositions.  
3. Add a principled comparison against adversarial training or robust tensor recovery methods under matched compute, since the paper’s critique of ATTR is central. Without this, the paper does not convincingly establish that asymmetric attacks are necessary rather than just convenient.  
4. Include reproducibility-critical details for all experiments: exact ranks, iteration counts, stopping tolerances, initialization distributions, and hyperparameter search ranges. ICLR standards are strict here, and missing details directly weaken confidence in the reported results.

# Final Consolidated Review
## Summary
This paper studies adversarial perturbations against tensor ring (TR) decomposition and proposes AdaTR, a bilevel max-min attack that explicitly maximizes reconstruction error under a Frobenius-norm budget. To reduce the heavy cost of backpropagating through ALS updates, it also introduces FAG-AdaTR, an approximate-gradient variant, and evaluates both methods on tensor decomposition, completion, and recommendation tasks for images, videos, and recommender data.

## Strengths
- The paper targets a genuinely underexplored robustness question for tensor decompositions, rather than another standard classifier attack. The problem is timely and relevant because TR/ALS-style factorization is used in vision and recommendation settings.
- The attack formulation is conceptually aligned with the actual failure mode of decomposition: AdaTR directly optimizes reconstruction error rather than using the misaligned symmetric objective inherited from ATTR/ATNMF. The paper also provides a concrete efficiency-oriented variant, FAG-AdaTR, and reports substantially lower runtime and memory than full backpropagation through ALS.
- The empirical scope is reasonably broad for a first paper in this space: the attack is tested on images, videos, tensor completion, and recommendation, with additional appendix experiments on CP/Tucker/TT and runtime comparisons.

## Weaknesses
- The main method is built on strong and only partially justified approximations. In FAG-AdaTR, the paper explicitly assumes the factor updates are independent of the perturbation to obtain a tractable gradient, which means the “fast” method is not the true gradient of the original bilevel objective. The paper does not quantify how far this surrogate deviates from the true attack objective, so it is hard to tell when the approximation is reliable.
- The theory is conditional and somewhat fragile for the stated setting. The convergence claims rely on differentiability, bounded Jacobians, and Lipschitz assumptions for the ALS-induced mapping \(E \mapsto G^{(T)}(E)\), which are nontrivial for alternating least squares with pseudoinverse updates and possible rank-deficiency/degeneracy. The results read more like standard projected-gradient folklore than a strong guarantee specific to TR.
- The empirical validation is not as rigorous as the claims suggest. The paper mostly compares against ATTR and Gaussian noise, but does not include stronger first-order baselines, multi-start attacks, or ablations that separate the benefit of the asymmetric objective from the approximate-gradient shortcut. This makes it hard to know whether the gains come from the new formulation or simply from a well-tuned projected optimization loop.
- The claim of broad transferability/generality is overstated relative to the evidence. The paper mainly attacks TR-ALS and then shows transfer to a set of TR-based defenses; the appendix extensions to CP/Tucker/TT are useful but too thin to support a strong general “tensor decomposition attack framework” claim.

## Nice-to-Haves
- A direct ablation comparing AdaTR to the same objective optimized with the exact/most faithful gradient available, to isolate how much performance is lost in FAG-AdaTR.
- More systematic sensitivity analysis over outer/inner iterations, initialization, rank choice, and learning rate.
- Attack-vs-budget curves with multiple seeds and clearer statistical reporting across all main tasks.
- A more explicit characterization of perceptual stealth for image/video perturbations, especially for the larger budgets used in some experiments.

## Novel Insights
The most interesting insight is that the previously natural ATR/ATNMF-style symmetric min-max formulation is misaligned with the decomposition attack goal: it can optimize a perturbation update that does not actually worsen the final reconstruction and may even improve it at small budgets. AdaTR’s asymmetric bilevel view is therefore the right conceptual fix, and the paper’s own theorem and experiments both support the point that “attacking the optimization dynamics” is not the same as “attacking the final reconstruction.” That said, the practical value of this insight is partially diluted by the heavy reliance on surrogate gradients and by the fact that the strongest evidence is still concentrated on TR-ALS and a small set of downstream benchmarks.

## Potentially Missed Related Work
- **Adversarially-trained nonnegative matrix factorization (Cai et al., 2021)** — directly relevant as the closest preceding adversarial factorization formulation that this paper extends and criticizes.
- **Adversarial nonnegative matrix factorization (Luo et al., 2020)** — important prior art on adversarial optimization for factorization models.
- **On the adversarial robustness of PCA / subspace learning (Li et al., 2020; Li et al., 2021; Pimentel-Alarcón et al., 2017)** — relevant background for perturbation-driven factorization vulnerability, though the paper already cites them.
- **Robust tensor completion / robust TR methods** such as HQTRC and TRPCA-TNN — relevant because a stronger defense-side comparison would help contextualize attack strength.

## Suggestions
- Add one ablation that measures the fidelity of FAG-AdaTR’s approximate gradient against the exact backpropagated gradient on small instances, and report how that fidelity correlates with attack success.
- Include a stronger baseline suite: projected gradient ascent on the input tensor, multi-start variants, and a direct version of the same objective without the approximation shortcut.
- Tighten the claims about generality and transfer: state explicitly that the main validated setting is TR-ALS, and present CP/Tucker/TT as preliminary appendix evidence rather than established broad coverage.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 4.0, 4.0]
Average score: 4.5
Binary outcome: Reject
