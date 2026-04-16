## Summary
This paper extends decision trees and random forests to constant-curvature and mixed-curvature product manifolds via an angular split formulation in 2D projections. The core idea is interesting and likely useful: it unifies Euclidean, hyperbolic, and hyperspherical cases, and the empirical study is broad, but the paper overstates how decisively the experiments support broad downstream superiority, especially on the very mixed-curvature settings that motivate the work.

## Strengths
- **Clear and meaningful problem setting.** The paper targets a genuine gap: downstream supervised learning tools for product-manifold embeddings are much less developed than the embedding methods themselves. This is well motivated in the introduction and reinforced by the breadth of applications considered.
- **Technically neat unification.** Section 3 presents a clean angular reformulation that ties together Euclidean, hyperbolic, and spherical splits through 2D projections. The extension to hyperspherical DTs/RFs and then to product manifolds is a real methodological contribution.
- **Good empirical breadth.** The evaluation spans 57 benchmarks across synthetic data, graph embeddings, VAE latent spaces, and native non-Euclidean data, including classification, regression, and link prediction. Even if not all of these are equally convincing, the paper clearly attempts to stress the method across diverse regimes rather than a single cherry-picked setting.
- **Strong single-manifold results.** The reported results on single-curvature tasks are genuinely impressive: Table 1 claims top-1 on 21/22 single-manifold benchmarks, and Figures 3–4 show consistent gains over ambient/tangent tree baselines and geodesic \(k\)-NN in those settings.
- **Useful qualitative illustration.** Figure 5 makes the intended inductive bias concrete by showing smoother geometry-respecting decision regions than Euclidean/tangent RFs on spherical data.
- **Reasonably candid limitations section.** The paper explicitly acknowledges dependence on embedding quality/signature selection and the lack of a privileged basis in non-Euclidean embeddings. That honesty helps calibrate the contribution appropriately.

## Weaknesses

###: Fatal
None.

### Major:
- **The empirical framing is overstated relative to the actual results, especially for product-manifold tasks.**  
  The abstract and summary statistics emphasize broad winning rates (“ranked first on 21 of 22 single-manifold benchmarks and 18 of 35 product manifold benchmarks” and “top 2 on 53 of 57”). But the detailed tables are more mixed. In Table 2, product-manifold classification is only top-1 on 11/24 tasks, and on the 8 synthetic multi-\(K\) classification tasks, \(k\)-NN wins every row. In Table 3, there are also clear failures, e.g. **Temperature** and **Traffic**, where the Ambient baseline is materially better than Product. So the evidence supports “often competitive and sometimes best,” not a strong general claim of superiority for downstream learning on product manifolds.
- **Baseline strength in the main paper is limited for supporting a broad downstream-learning claim.**  
  The core comparisons are to ambient-space DT/RF, tangent-space DT/RF, geodesic \(k\)-NN, and a perceptron that is explicitly omitted from Figure 3 because it “never achieved competitive results.” For a paper positioning the method as a broadly useful tool for classification/regression on manifold embeddings, this is a somewhat narrow baseline set. The paper does mention MLP/GNN comparisons only in the appendix, and the main-text evidence is therefore strongest only for “better than straightforward tree baselines plus \(k\)-NN,” not for broader superiority among downstream methods.
- **The paper’s count-based summary statistics somewhat overstate the strength of the empirical case.**  
  “Top-2 on 53/57” is not especially compelling when the comparison set is small and many margins are tiny or ties. Table 2 and Table 3 show that effect sizes vary substantially and sometimes reverse sharply. The count-based presentation compresses away where the method systematically loses, particularly in mixed-curvature classification. This is a presentation issue rather than a methodological flaw, but it materially affects how convincing the paper appears.
- **Some central geometric claims are not adequately substantiated in the visible main text.**  
  The paper repeatedly claims angular splits are “geodesically convex, maximum-margin, and composable,” and Contribution 1 elevates these properties as a key reason the method is principled. In the provided text, geodesic-convexity is plausible from the homogeneous-hyperplane construction, but “maximum-margin” is asserted rather than demonstrated here, and Euclidean equivalence is deferred to Appendix C. Since these are central novelty claims, the main paper would be stronger if at least the key argument for maximum-margin behavior were visible in the body.

### Minor
- **The hyperspherical construction introduces an arbitrary basis choice that is not analyzed.**  
  Section 3.3 explicitly says: “We adopt the convention of fixing the first dimension of the embedding space as \(x_0\), which intuitively corresponds to fixing a ‘north pole’.” This is not just notation; it is a real inductive-bias choice. The paper later acknowledges in Limitations that “the lack of a privileged basis ... makes the inductive bias of decision trees less well-motivated.” That concern is valid. Some sensitivity analysis to rotations/basis choice would help establish robustness.
- **The experimental protocol is somewhat weaker than ideal for benchmark-style claims.**  
  Section 4.1 states that the paper uses “an identical 80:20 train-test split” and then reports confidence intervals/significance. While this is not invalid, a single split is less convincing than repeated resampling/cross-validation when making broad benchmark claims across many datasets.
- **Hyperparameter fairness is not very transparent in the main text.**  
  Section 4.3 says the authors “set hyperparameters identically to Scikit-Learn DTs and RFs,” and for their own model “consider all \(\binom{D}{2}\) projections.” But the main text does not clearly explain whether parameters such as tree depth, forest size, or \(k\) for \(k\)-NN were comparably tuned across methods. This leaves some uncertainty about whether the reported differences reflect geometry awareness or tuning choices.
- **The merged reporting of DT and RF baselines obscures model-class comparisons.**  
  Tables 2–3 define “Ambient” and “Product” as the better of DT or RF means. That is convenient for summarization, but it blurs whether Product DT beats Ambient DT or Product RF beats Ambient RF. Since the paper says significance is computed between same-type models, this merged presentation is somewhat misaligned with the inferential story.
- **The product-manifold motivation is strongest where the method is least dominant.**  
  This is partly covered above, but it is worth stressing as a scientific point: the paper is most convincing on single-manifold tasks, whereas the main conceptual motivation is mixed-curvature product manifolds. That does not negate the contribution, but it lowers the practical significance of the headline framing.

### Trivial
- **Training-time complexity should be discussed more prominently in the main text.**  
  Section 3.4 notes one can search over all \(\binom{D}{2}\) projections while retaining \(O(1)\) decision complexity, which is true per decision once a split is chosen, but the distinction between inference-time and split-search complexity could be stated more clearly to avoid overreading.

## Nice-to-Haves
- Add a failure-mode analysis for why \(k\)-NN wins all 8 synthetic multi-\(K\) classification tasks in Table 2.
- Include a sensitivity study to misspecified signatures or rotated embeddings, especially for the spherical case with a fixed “north pole.”
- Show main-text comparisons to the appendix neural baselines if those results are competitive enough to help calibrate practical value.
- Report per-model-class tables (DT vs DT, RF vs RF) in the main paper or a compact appendix pointer, rather than only merged maxima.
- Provide a product-manifold decision-boundary visualization analogous to Figure 5, not only a single-manifold (\(S^2\)) example.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The comparison is unfair because Product considers all \(\binom{D}{2}\) projections while ambient/tangent trees do not.”**  
  Removed because this asymmetry favors the baseline family less than the authors’ method only if one assumes matched candidate-budget fairness is required; the paper explicitly motivates angular features as the method itself, so this is better framed as a nice-to-have ablation, not a core weakness.
- **“The paper should compare to specific missing prior works / advanced tree packages.”**  
  Softened/removed as a direct criticism because we should not speculate about missing related works or impose a very broad baseline wish list as a hard requirement. The valid retained point is simply that the main-text baseline set is limited for the strength of the claims made.
- **“The method cannot be trusted because the appendix is omitted here.”**  
  Removed. It is fair to say some claims are not substantiated in the visible main text; it is not fair to imply the appendix does not address them.
- **Pure formatting or parser-artifact issues.**  
  Removed per instructions.

## Novel Insights
The paper’s most interesting tension is that it appears strongest as a **single-manifold geometry-aware tree method** and only secondarily as a convincing **mixed-curvature product-manifold** breakthrough. This does not diminish the technical novelty of the angular split formulation, but it does suggest the true contribution is a clean geometric generalization of tree learning to constant-curvature spaces, with product-manifold success being promising but still uneven. In other words, the method may already be worthy as a toolbox contribution, but the paper weakens itself by selling it as more universally dominant on mixed-curvature downstream tasks than the tables actually support.

## Suggestions
- Recalibrate the headline claims to match the detailed results, especially for mixed-curvature classification.
- Put the strongest supporting evidence for “maximum-margin” and Euclidean-equivalence claims into the main text, not only the appendix.
- Add a targeted analysis of the synthetic multi-\(K\) failure mode where \(k\)-NN consistently wins.
- Report same-type comparisons explicitly (Product DT vs Ambient DT vs Tangent DT, and RF likewise) instead of only merged maxima in Tables 2–3.
- Add sensitivity experiments for basis choice/rotations in spherical embeddings and for signature misspecification.
- If appendix MLP/GNN comparisons are favorable, move a compact version into the main paper; if not, temper the broad “powerful new tools” framing accordingly.

## Score and Decision
**Assessment by axis:**  
- **Originality:** Good. Extending angular split-based tree learning to hyperspherical and product manifolds is a meaningful contribution.  
- **Importance of research question:** Good. Product-manifold downstream learning is a real and relevant problem.  
- **Claims support:** Mixed. Stronger for single-manifold results, weaker for broad product-manifold superiority.  
- **Experimental soundness:** Moderate. Broad and thoughtful, but baseline strength, reporting choices, and the single-split protocol limit conclusiveness.  
- **Clarity:** Generally good; the method is conveyed clearly, though some central guarantees are under-argued in the visible main text.  
- **Value to the community:** Moderate to high. Even with caveats, this is a useful geometric extension of DT/RF methodology.

**Calibration against human-reviewed anchors:**  
- I compared this paper most directly to **TTonmgTT9X (Fast Hyperboloid Decision Tree Algorithms; scores 5, 6, 8, 6, 8; accepted poster)**. The current paper is broader and more ambitious, with a more diverse evaluation and a real extension beyond hyperbolic space, which argues for a score in a similar accept-leaning band. However, it is also empirically less clean relative to its central mixed-curvature claim than a strong focused extension paper would ideally be.  
- I also compared to **2MLvV7fvAz (Spectro-Riemannian GNNs; scores 6, 5, 6, 6; accepted poster)**, which had a similarly moderate-but-real contribution with some limitations yet enough substance for acceptance. This paper feels comparable in overall contribution quality: technically interesting, useful to the community, but not airtight.  
- For lower-end anchors, **AN5uo4ByWH (Mixed-Curvature Transformers; scores 1, 5, 5; rejected/withdrawn)** and **HAMBmtKLc8 (SPD GNN; scores 5, 3, 8, 3; rejected)** had more fundamental support/positioning problems. The current submission is clearly stronger than those: it has a concrete algorithmic contribution, substantial experiments, and no fatal flaw.  
- Relative to these anchors, this paper lands in the **weak accept / borderline accept** range rather than the reject range.

**Final score: 6.5 / 10**  
**Decision: Accept**

MY FINAL SCORE: <pineapple>6.5</pineapple>  
MY FINAL DECISION: <orange>Accept</orange>