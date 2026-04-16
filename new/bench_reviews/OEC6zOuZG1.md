## Summary
This paper studies random feature models under anisotropic Gaussian inputs with a rank-1 spiked covariance structure, moving beyond the standard isotropic setting where RFMs are known to be asymptotically equivalent to noisy linear models. Its main technical contribution is a universality theorem under spiked data and a characterization showing that, depending on a correlation-dependent quantity \(\eta\), the effective surrogate for the RFM transitions from a noisy linear model to a noisy polynomial model.

## Strengths
- **Meaningful extension of prior RFM universality analysis to anisotropic data.** The paper clearly targets a real gap in the literature: prior equivalence results were mostly under isotropic inputs, while this work studies the spiked covariance model in the proportional asymptotic regime.
- **Theorem 2 is the paper’s strongest contribution.** The result that RFMs can be replaced by noisy polynomial surrogates of degree controlled by \(\eta\) is conceptually useful and does go beyond the standard noisy-linear equivalence story.
- **The linear regime is recovered cleanly as a special case.** Corollary 3 gives a simple condition under which the noisy linear model remains equivalent, which helps organize the overall picture.
- **Remark 4 adds genuine nuance.** The paper does not merely say “more correlation means more nonlinearity”; it also shows that the effective degree depends on the Hermite coefficients of both the activation \(\sigma\) and target function \(\sigma_*\).
- **The synthetic experiments are reasonably aligned with the surrogate-model narrative.** Figures 1–2 are useful in illustrating when the noisy linear approximation tracks the RFM and when a higher-order surrogate is needed.
- **The paper is generally clear at a high level.** The question it asks is easy to follow, and the organization from universality \(\rightarrow\) polynomial equivalence \(\rightarrow\) linear regime is coherent.

## Weaknesses

###: Fatal
- **The headline claim “Random Features Outperform Linear Models” is stronger than what the theory actually proves.**  
  The central theoretical results in Section 4 are equivalence statements: Theorem 1 compares RFMs under two activations when certain moments match, and Theorem 2/Corollary 3 characterize when the RFM is equivalent to a noisy polynomial or noisy linear surrogate. These results do **not** prove that the RFM achieves lower risk than the best linear predictor on the original input space. The paper repeatedly claims more than this, e.g. the abstract says “a high correlation between inputs and labels is a critical factor enabling the RFM to outperform linear models,” and Section 5 is titled “NONLINEARITY IN RFM BENEFITS HIGH INPUT-LABEL CORRELATION.” What is convincingly established is that strong correlation can move the effective surrogate beyond the noisy-linear regime; superiority over linear models in the broader sense is not theoretically established. This claim-level mismatch affects the framing of the entire paper.

### Major:
- **The empirical “linear” comparisons do not fully substantiate the title claim.**  
  Equation (5) is a noisy linear surrogate arising from the analysis, not the generic best linear predictor \(x \mapsto w^\top x\). In Section 5, the “optimal linear activation” in Eq. (21) is still inside the RFM architecture, i.e. a linear activation applied after random projection, not ridge regression or another strong linear baseline on raw inputs. Since the paper’s claim is about “linear models” broadly, the lack of comparison to a true input-space linear predictor materially weakens the empirical support for the headline conclusion.

- **Section 5’s activation-function comparison is confounded by unequal optimization.**  
  The paper explicitly says the coefficients in the linear activation (21) and polynomial activation (22) are “determined numerically to minimize the generalization error.” ReLU and Softplus are instead fixed activations. This makes Figure 3 difficult to interpret as evidence that input-label correlation intrinsically favors nonlinear RFMs over linear ones: part of the observed advantage may simply come from directly optimizing a richer parametric activation family against generalization error. This is a substantive fairness issue within the paper’s own comparisons.

- **The CIFAR-10 validation is only weakly connected to the theory and does not isolate the claimed mechanism.**  
  The theory analyzes a Gaussian rank-1 spiked covariance model with labels generated as in Eq. (6), while Figure 4 varies “input-label correlation” by label flipping and, in one condition, adds Gaussian noise to inputs. These manipulations alter multiple aspects of the problem simultaneously: label flipping changes noise rate/Bayes difficulty, and added Gaussian noise changes covariance and SNR. So while the trends are suggestive, they do not cleanly validate the paper’s claimed causal explanation that strong input-label correlation specifically is the driver.

- **The narrative sometimes oversimplifies \(\eta\) as merely “input-label correlation.”**  
  The theory’s operative quantity in Theorem 2 is  
  \[
  \eta := \max_{1\le i\le k} \frac{|(\xi+\theta\alpha\gamma)^T t_i|}{\sqrt{1+\theta\alpha^2}},
  \]
  which depends on projections onto the random feature rows, not only on \(\alpha\). The paper does discuss the connection via Eq. (19), but its prose often collapses this into the simpler statement that “high input-label correlation” is the determining factor. That is directionally reasonable, but somewhat looser than the actual theorem.

- **The theoretical assumptions limit the scope of the claims, especially for practice-facing conclusions.**  
  The analysis assumes a rank-1 spiked Gaussian model, a particular Gaussian random-feature matrix scaling, and in theory an odd activation satisfying (A.6). These assumptions are standard enough for a learning-theory paper, so they are not fatal. But given the paper’s strong practical framing, the gap from these assumptions to real data remains substantial and should be acknowledged more carefully.

### Minor
- **The odd-activation assumption excludes ReLU and Softplus, yet both are heavily used in experiments.**  
  The paper explicitly notes: “while ReLU (9) does not conform to the odd function assumption stipulated in (A.6), empirical evidence suggests that our findings remain valid even when using ReLU.” This acknowledgment is helpful, so the issue is not a misunderstanding. Still, since some of the most visible empirical conclusions rely on activations outside the theorem’s assumptions, the theory-experiment bridge is incomplete.

- **The \(\beta < 1/2\) restriction is underexplained.**  
  The paper states that proofs require \(\beta < 1/2\), and Figure 3c even discusses behavior beyond that range empirically. More intuition about whether this boundary is believed to be fundamental or just technical would improve the paper.

- **The novelty is more in the extension and synthesis than in fundamentally new methodology.**  
  The proof sketch for Theorem 1 explicitly follows Hu & Lu (2023) via Lindeberg’s method. The extension to spiked covariance is still nontrivial and worthwhile, but the paper could better articulate the new technical difficulties introduced by anisotropy.

### Trivial
- None.

## Nice-to-Haves
- Add a comparison to a true linear baseline on raw inputs, such as ridge regression, to support the paper’s title and framing.
- Clarify more sharply that the main proven result is a shift from noisy-linear to noisy-polynomial equivalence, and present “outperforming linear models” as an empirical observation unless a direct theorem is added.
- Provide more discussion of how restrictive \(\eta = O(n^{-1/l})\) is in typical regimes and whether it can be estimated from data.
- Expand on whether the \(\beta<1/2\) boundary is technical or fundamental.
- Either extend the theory beyond odd activations or better delimit the scope of the claims when using ReLU/Softplus in experiments.
- If space permits, add experiments with multi-spike covariance or at least discuss how the results might extend.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper should compare to more feature-learning methods / broader related work.”**  
  Removed because this drifts into missing-related-work or scope-creep territory. The paper is about RFMs without feature learning; it need not also solve or benchmark feature learning comprehensively.

- **Pure formatting/parser complaints.**  
  Removed because the user explicitly noted PDF extraction artifacts, and minor style issues are not substantive review points.

- **Claims that the paper is invalid because assumptions are stylized or unrealistic by themselves.**  
  Weakened/removed as a standalone criticism. Stylized assumptions are normal in learning theory; the valid issue is instead whether the paper overstates practical conclusions given those assumptions.

## Novel Insights
The most important synthesis is that the paper is stronger as a contribution on **which surrogate class is asymptotically appropriate for RFMs under anisotropy** than as a paper proving broad superiority over linear methods. In particular, Remark 4 suggests the real story is not simply “correlation helps nonlinearity,” but that the effective surrogate degree is governed jointly by geometry (\(\eta\)) and the Hermite structure of both the activation and target. This is a more precise and interesting message than the title’s broad outperforming-linear framing.

## Suggestions
- Reframe the paper around the strongest supported claim: under spiked covariance data, the effective asymptotic surrogate for RFMs can move from noisy linear to noisy polynomial as \(\eta\) increases.
- Add a theorem or proposition comparing against a genuine input-space linear predictor if the title/conclusion are to remain unchanged.
- Include a true linear baseline on raw inputs in synthetic and CIFAR experiments.
- Avoid optimizing some activation families for generalization error while leaving others fixed, or else clearly label Figure 3 as an oracle/upper-bound style comparison.
- Tighten the discussion around \(\eta\), making clear that \(\alpha\) is only one contributor to the operative condition.
- Expand the discussion of the odd-activation gap and the \(\beta<1/2\) restriction.

## Score and Decision
**Assessment by axis:**  
- **Originality:** moderate. The anisotropic/spiked extension and noisy-polynomial equivalence are real contributions, though the proof strategy is largely inherited.  
- **Importance:** moderate-to-good. The question is important for understanding when random features escape the noisy-linear picture.  
- **Support for claims:** mixed. The surrogate-equivalence claims are reasonably supported; the stronger “outperform linear models” framing is not.  
- **Experimental soundness:** moderate. Synthetic experiments are aligned with the theory, but the key empirical comparisons and CIFAR validation are not fully convincing for the paper’s broad claims.  
- **Clarity:** fairly good overall, though some narrative statements overstate what is actually proved.  
- **Community value:** moderate. The paper should interest the random-features/high-dimensional theory community, but the current framing overreaches.

**Calibration:** I compared this paper against:
- **`/home/wg25r/review_agent/human_reviews/zxqdVo9FjY.md`** (Reject; scores 6,5,5,3,5): another spiked-covariance theory paper with meaningful but somewhat narrow technical contributions and questions about practical relevance. The current paper is in a similar quality band, though somewhat better framed technically than that paper and somewhat weaker in claim calibration.
- **`/home/wg25r/review_agent/human_reviews/MY8SBpUece.md`** (Reject; scores 5,6,6,5): a theory paper on nonlinear feature learning with real technical content but notable assumptions/practical gaps. The current paper feels comparable in technical interest and limitations.
- **`/home/wg25r/review_agent/human_reviews/Of2nEDc4s7.md`** (Accept poster; scores 6,6,8): an anisotropic-data theory paper that appears to earn acceptance by having a sharper, better-supported main claim and stronger overall positioning. The current paper falls below this anchor because its headline claim is materially overstated relative to what is proved.
- **`/home/wg25r/review_agent/human_reviews/UZ893n8FXr.md`** (Accept poster; scores 8,8,6,6,6): a stronger positive anchor with clearer novelty and a better-aligned claim/contribution package. The current submission is clearly below that level.

Given these anchors, this paper lands **below the acceptance threshold**: it has a solid technical core, but the main framing and evidence do not adequately support the strongest claims.

**Score: 4.5 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>