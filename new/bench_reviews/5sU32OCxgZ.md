## Summary

The paper proposes TTVD (Test-Time adjustment by Voronoi Diagram guidance), a neighbor-based test-time adaptation framework that recasts adaptation through the lens of computational geometry. It defines a basic Voronoi Diagram (VD) loss, then extends it to Cluster-induced Voronoi Diagrams (CIVD) using multi-prototype clusters and to Cluster-induced Power Diagrams (CIPD) for noisy-sample filtering. Under the standardized TTAB evaluation protocol, TTVD achieves competitive or state-of-the-art classification error and substantially lower Expected Calibration Error on CIFAR-10-C, CIFAR-100-C, ImageNet-C, and ImageNet-R.

## Strengths

- **Principled geometric reformulation**: The paper formally connects neighbor-based TTA to Voronoi Diagrams (Definition 3.1) and introduces Cluster-induced Voronoi Diagrams (Definition 3.2, Eq. 4) and Cluster-induced Power Diagrams (Definition 3.4, Eq. 6) to the online adaptation setting. This provides an interpretable, analytical framework not previously articulated for TTA.
- **Strong empirical calibration improvements**: TTVD dramatically reduces ECE on ImageNet-C (from ~38% for Tent/SAR down to 21.0%) and achieves consistently lower classification error across four benchmarks (Table 1), under a peer-reviewed, standardized evaluation protocol.
- **Internally consistent ablation**: Table 2 demonstrates monotonic improvement from VD (28.4%) to CIVD (22.7%) to CIPD (20.5%) on CIFAR-10-C, indicating that the design choices are coherent and additive.
- **Robustness to practical constraints**: Table 4 shows TTVD is insensitive to class-mean estimation precision, achieving nearly identical error with 1%, 5%, or 10% of training data (59.9%, 59.8%, 59.8%), mitigating a deployment concern.

## Weaknesses

### Fatal
None.

### Major
- **Ambiguity in adaptation protocol threatens fair comparison**: The paper does not explicitly state which parameters TTVD updates during adaptation. Algorithm 1 updates $\sigma_{t+1}$ via gradient descent on $\mathcal{L}_{VD}$, but the text only notes that TTA methods "commonly" update channel-wise affine normalization parameters while leaving the rest fixed. TTAB-standardized comparisons typically enforce BN-only updates for fairness, yet the paper never confirms this is what TTVD does. If TTVD adapts more parameters (e.g., the full feature extractor backbone) or modifies the classifier head in a way Tent/SAR do not, the comparisons in Table 1 become uncontrolled. The authors must explicitly clarify the adaptation scope in the main text.
- **“Self-supervision unification” claim is substantively misleading**: CIVD does not optimize any self-supervision objective (e.g., rotation-angle prediction) at test time. Section 3.2 describes using rotation-augmented training images to compute multiple prototypes per class via “self-supervised label augmentation” (Lee et al., 2020), but this is an offline prototype-computation step, not online test-time self-supervision. Yet the paper claims CIVD “integrates the joint contribution of self-supervision and entropy-based methods” and that “the joint label $\tilde{y}_k^{(\alpha)}$ avoids the negative transfer since the objective is now unified.” Because there is no joint optimization of multiple objectives to begin with—only entropy minimization over prototype-based predictions—the claim that CIVD sidesteps negative transfer through “unification” is unsupported and conceptually confused.

### Minor
- **Power Diagram filtering is algorithmically vague in the main text**: Section 3.3 describes filtering via “subtracting the PD from the VD” without providing an explicit decision rule, loss term, or clear statement of how PD weights $v_k$ are obtained at test time. Lemma 3.1 establishes that a linear classifier induces a PD with specific site/weight parameters, but the paper never explicitly confirms in the main text that it uses the pre-trained linear classifier weights for this purpose. Algorithm 3 is deferred to Appendix H, leaving the main-text reader with only a geometric metaphor.
- **Missing controlled ablations for mechanistic attribution**: Table 2 ablates VD → CIVD → CIPD but does not include controls such as Tent with rotation-augmented prototype computation, or standard sample-filtering baselines (e.g., entropy-based filtering as in SAR) while holding the adaptation backbone constant. Without these, the evidence cannot fully disentangle whether the gains stem from the geometric constructs themselves or from the underlying augmentation and filtering heuristics they encode.

### Trivial
- **Figure 4 reports cumulative error**: The caption states “Error (%) calculated over all retrospective test samples.” This is a non-standard cumulative metric; its interpretation is unclear without explicit definition.

## Nice-to-Haves
- Replace or supplement the 3-class MNIST-C toy visualization (Figure 1) with feature-space visualizations (e.g., t-SNE or PCA) on ImageNet/CIFAR showing how partitions evolve during adaptation.
- Explicit analysis distinguishing noisy samples from merely hard boundary samples: show that samples filtered by the PD/VD disagreement region are indeed more harmful to adaptation than low-confidence but informative samples.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **"Unfair comparison / fundamentally mismatched" in the harshest sense**: Removed because the framework is explicitly scoped as a *neighbor-based* TTA method; prototype-based inference is endemic to this family (cf. T3A, TAST, AdaNPC in the baseline pool), so comparing against linear-head methods is not inherently invalid—it is comparing different methodological families. The valid concern is *ambiguity* about adaptation scope, which is retained above, not the strong claim that the comparison is nullified.
- **"Geometric framework is largely a redescription without algorithmic substance"**: Removed as overstated. Definitions 3.2–3.4 and Equations 4–6 are mathematically precise extensions; whether they translate to novel algorithmic behavior beyond standard augmentation/filtering is an open empirical question, not a vacuity.
- **Missing EATA baseline**: The paper already compares against eight strong methods under a standardized protocol; lack of one additional recent baseline is not a core flaw.
- **Oracle model-selection footnote complaint**: The paper explicitly discloses and contextualizes the oracle values (footnote 1); this is transparency, not concealment.
- **Memory cost / storing class means**: A trivial concern given that class means are negligible compared to model parameters, and Table 4 directly addresses robustness to estimation precision.
- **MNIST-C toy example as insufficient motivation**: It is illustrative only; the actual large-scale evidence is in Tables 1–4.

## Novel Insights
The most genuinely novel observation is that the paper bridges two largely disconnected bodies of work—computational geometry (CIVD, Power Diagrams) and online test-time adaptation—by showing that the structure underlying prototype-based TTA can be formalized as space partitioning. If the mechanistic claims are tightened, this geometric lens could inspire principled extensions beyond the specific adaptations tested here.

## Suggestions
- **Clarify adaptation target in the main text**: Add a sentence to Algorithm 1 or Section 4.1 explicitly stating whether only BN affine parameters, a subset of layers, or the full feature extractor $\sigma$ is updated.
- **Rename or rigorously justify “self-supervision” terminology**: If the intent is prototype enrichment via training-time augmentation, remove references to “self-supervision” and “negative transfer” from the CIVD description, or provide empirical evidence that CIVD mitigates negative transfer compared to a true joint-objective baseline.
- **Add a mechanistic control experiment**: Compare CIPD against an ablation that uses the same prototype-based adaptation but replaces PD-based filtering with simple entropy thresholding, to isolate the contribution of the geometric filtering criterion.

<context>
- **Original reviewer signal**: Harsh Critic believed the adaptation protocol is ambiguous to the point of unfair comparison, the geometric framework is superficial redescription, and key mechanistic claims lack isolation experiments. Strength Finder viewed the geometric reformulation as principled and novel, with strong additive ablations and benchmark results under standardized TTAB settings.
- **What was dropped and why**: The harshest "unfair comparison" charge was softened because the paper's scope is explicitly neighbor-based TTA, and prototype-based inference is inherent to that family (cf. T3A/AdaNPC baselines). The "redescription without substance" claim was weakened: the mathematical definitions are precise; the open issue is whether geometry yields empirically novel behavior beyond augmentation/filtering, not whether it is vacuous. EATA omission, memory nits, oracle-footnote complaint, and MNIST toy-motivation critiques were dropped as minor or parser/reviewer-side issues.
- **Cross-checks performed**: Algorithm 1 confirms $\sigma$ is updated and inference uses distance-to-prototype softmax (Eq. 3). Section 3.1/3.2 confirms "self-supervision" in CIVD refers to offline rotation-augmented prototype computation, not a test-time self-supervision loss. Lemma 3.1 links linear classifiers to PD weights, but the main text never explicitly states this linkage is used for $v_k$. Table 1 margins are modest for error (0.7–1.6%) but very large for ECE (up to 17.4 pp). Footnote 1 explicitly flags oracle results as potentially unrealistic.
- **Severity read**: The survived weaknesses are methodologically significant but not fatal. The adaptation-scope ambiguity is the load-bearing concern; if the paper clarifies TTVD uses BN-only updates (as TTAB standard suggests), this becomes minor. The "self-supervision unification" overclaim is a presentation/substantive issue that could be fixed with more careful wording and ablations. No single verified weakness invalidates the core empirical results.
- **Anything else load-bearing**: The paper operates within the neighbor-based TTA family and should be evaluated as such; it does not claim to improve Tent by changing only the loss type. The appendix (stripped from this parsed version) likely contains Algorithm 3 and implementation specifics that would resolve some of the main-text ambiguity. The ECE improvements are a major strength that partially offsets modest error-margin improvements.
</context>