Now let me run calibration searches in parallel.Now I have enough information to write the final review. Let me synthesize everything.

---

## Summary

CrystalFramer introduces "dynamic frames" for SE(3)-invariant crystal property prediction, extending the Crystalformer transformer baseline by constructing per-atom, per-layer local coordinate frames driven by learned self-attention weights rather than static structural features. Two frame variants are proposed—weighted PCA frames and max frames—and combined with three angular GBF features to enrich the position embedding beyond distances. Experiments across JARVIS, Materials Project, and OQMD datasets show that max frames consistently outperform the baseline and competing methods on most tasks with minimal parameter overhead (~+100K parameters).

---

## Strengths

- **Novel conceptual contribution, validated by ablation**: The paper demonstrates that conventional PCA frames and lattice frames applied to Crystalformer actually *degrade* performance (JARVIS formation energy: PCA 0.0325 vs. baseline 0.0306, Table 1), while the proposed dynamic max frames substantially improve it (0.0263). Critically, the ablation against static local frames (Tables 1–2) isolates the dynamic contribution from simply adding angular features, and dynamic max frames outperform static local frames on 7 of 9 reported tasks, providing direct empirical support for the core hypothesis.

- **Consistent improvements across scale**: CrystalFramer improves over the Crystalformer baseline not only on JARVIS (55K) and MP (69K) but also on the 817K-material OQMD dataset (formation energy: 0.02115 → 0.01871, Table 3), demonstrating that the approach scales beyond small benchmarks.

- **Parameter efficiency**: Only ~100K additional parameters over Crystalformer (952K vs. 853K), while outperforming PotNet (1.8M), Matformer (2.9M), and iComFormer (5.0M) on most tasks—making the contribution clearly attributable to the frame design rather than model capacity.

- **Interpretable frame visualizations**: Figure 3 shows that dynamic frames in different layers capture qualitatively distinct coordination environments (octahedral Mg–F vs. tetrahedral Sn–Mg), providing physics-meaningful evidence that the frames adapt to local chemistry rather than producing arbitrary orientations.

- **Principled max frame design**: The max frame construction (Section 3.1) elegantly avoids the eigenvalue degeneration problem that afflicts PCA frames for symmetric crystals (~10% two-degree, ~1% full degeneration), and is also inherently invariant to unit-cell variations since it uses the full infinite structure $\tilde{P}$.

---

## Weaknesses

### Fatal
None.

### Major

- **Training epoch mismatch confounds the headline comparison.** Section 5 explicitly states "we have increased the number of training epochs to account for the increased complexity of our edge feature design." CrystalFramer is trained for 2000 epochs on JARVIS while Crystalformer numbers are cited from prior work (trained for fewer epochs). The paper argues CrystalFramer "takes longer to converge, but reduces validation losses more rapidly," but it never re-evaluates Crystalformer at an equal epoch budget. Since all claims in Tables 1–3 are anchored against cited Crystalformer scores, the fraction of the improvement attributable to simply training longer versus the dynamic frame design is unknown. This is the most significant methodological gap. The gains (e.g., JARVIS formation energy 0.0306 → 0.0263) may hold up entirely, but the paper as submitted cannot demonstrate this.

### Minor

- **Weighted PCA frames often degrade performance, undermining the generality of the "dynamic frames" concept.** Weighted PCA frames perform *worse* than the Crystalformer baseline on MP formation energy (0.0197 vs. 0.0186) and bandgap (0.214 vs. 0.198, Table 2). The paper attributes this to eigenvalue degeneration and discusses it in Appendix F, but the framing in the abstract and introduction implies both dynamic frame variants are improvements. Since max frames are the clearly successful variant, the paper should more explicitly present weighted PCA frames as a partially negative result and center the contribution on max frames alone. The current presentation slightly inflates the impression of the concept's universality.

- **Gradient blocking severs the optimization loop for dynamic frame construction, with no quantitative ablation of alternatives.** Footnote 2 acknowledges that gradients from frames to attention weights are blocked (due to PCA instability and argmax non-differentiability), and that alternatives (straight-through estimators, temperature annealing) were tried but gave inferior results. However, no quantitative comparison of these alternatives is provided. The reader cannot assess whether joint optimization was adequately explored or whether the "dynamic" property emerges from a principled design or simply reuses attention weights trained for a different purpose. This weakens the theoretical claim that frames are "optimized to focus on actively interacting atoms."

- **Test-time stochasticity of max frames is uncharacterized.** Max frames resolve weight ties via small perturbation noise, meaning different forward passes on the same structure can produce different frame orientations and therefore different predictions. No characterization of prediction variance across multiple runs is provided. For a method advocated for production use, this instability (even if minor in practice) deserves quantification.

### Trivial

- The OQMD scalability table (Table 3) includes only CrystalFramer vs. Crystalformer, without any other competing methods. This is understandable given compute requirements, but it limits the table's claim as a demonstration of scalability to "CrystalFramer scales as well as its base architecture," which is weaker than a broader scalability argument.

---

## Nice-to-Haves

- Run Crystalformer for the same number of epochs as CrystalFramer to directly rule out the training-length confound. Even a single re-run on one dataset (e.g., JARVIS formation energy) would substantially strengthen the comparison.
- Add a "random frame + angular features" baseline—a randomly oriented but consistent-within-sample frame with the same Eq. 7 angular features—to isolate whether frame *structure* (attention-guided vs. random) matters, or whether directional features under any frame are sufficient.
- Quantify test-time variance of max frames by reporting mean ± std MAE over multiple runs with different random seeds.
- Report quantitative ablation of the straight-through / temperature-annealing gradient alternatives (even as a "we tried these and they gave X% worse performance" table), directly supporting the gradient-blocking design choice.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Dynamic attribution confounded with local+directional" (Issue 2, Harsh Critic)**: The critic claims "roughly two-thirds" of the improvement comes from static angular features. Checking the actual numbers: JARVIS formation energy gains are nearly split equally (static: 0.0021, dynamic: 0.0022 improvement). The exact fraction varies by task but is not "two-thirds" on average. The general observation that static local frames also provide substantial improvement is correct and worth noting (kept as a minor framing point in the abstract/intro criticism), but the severity assigned to this criticism is overstated and partly factually wrong.

- **"Comparison may be unfair because CrystalFramer didn't tune hyperparameters"**: The paper presents this as an advantage for CrystalFramer (it's more robust). This is not a weakness; if anything it demonstrates the method's ease of use.

- **"CrystalFramer demonstrated on only one base architecture"**: The paper explicitly explains why other architectures (using channel-wise sigmoid attention) are incompatible with per-head frame construction, and acknowledges this limitation in Section 6. Criticizing this as a weakness is scope creep given the clear architectural reason.

- **"Max frames don't consistently beat iComFormer"**: The abstract says "various tasks," which is accurate (7/9 tasks). Not claiming universal state-of-the-art is not a weakness.

---

## Novel Insights

The most genuinely novel observation synthesized from the reviews is the distinction between *structure-aligned* and *interaction-aligned* frames for periodic systems. The paper's ablation reveals that the failure of PCA and lattice frames—architectures that succeed for molecules—stems not from poor frame quality per se but from the mismatch between globally determined frames and locally acting interactions in each message-passing layer. This suggests a broader design principle: for any message-passing network with localized, weighted interactions, the geometric encoding should be conditioned on the local weighting pattern rather than on the global structure. The result that even static local frames (with fixed distance-decay weights rather than learned attention weights) recover much of the improvement over global frames reinforces that *locality* is the primary driver, with *dynamism* providing a consistent additional benefit on top.

---

## Suggestions

1. Re-run Crystalformer baseline at 2000 epochs (or your full epoch budget) on at least one dataset and report the result, even in an appendix. This single addition would substantially address the largest methodological concern.
2. Reframe the introduction and abstract to accurately reflect that max frames are the primary contribution and weighted PCA frames provide more limited improvements, rather than presenting both as equally successful demonstrations of the concept.
3. Add test-time variance measurements for max frames (e.g., std MAE over 5 seeds) in Table 4 to characterize the stochastic nature of the method.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| Crystalformer | fxQiecl9HB.md | **7.25** (Accept/poster) | Direct baseline paper; introduced infinitely-connected attention for crystals — a similarly incremental but well-validated contribution |
| Space Group Crystal Gen. | jkvZ7v4OmP.md | **7.33** (Accept/poster) | Different task (generation), but similarly strong experimental scope for crystal domain |
| SLEM (quantum operators) | kpq3IIjUD3.md | **7.33** (Accept/poster) | Strong equivariant method for materials; provides a high-scoring topical anchor |
| HDGNN | BBD6KXIGJL.md | **7.33** (Accept/poster) | Equivariant GNN with solid ablation and results |
| PDDFormer | ewjN1MAnJi.md | **5.00** (Withdrawn) | Crystal property prediction with less compelling results and narrower scope |
| CrysToGraph | zUDbPgskDS.md | **3.25** (Reject) | Crystal transformer with weaker methodology and less thorough evaluation |
| ZHTYtXijEn | ZHTYtXijEn.md | **2.33** (Reject) | Low-quality paper with weak experimental validation |

**Reasoning:** CrystalFramer is clearly above the medium-low range (PDDFormer, 5.0; CrysToGraph, 3.25). It has a genuine novel concept, careful ablation, multi-dataset validation, and competitive results. The closest topical anchor is Crystalformer itself (7.25), which CrystalFramer directly extends. CrystalFramer's contribution is arguably more incremental than Crystalformer's (which introduced an entirely new architecture), but its experimental scope is broader (three datasets vs. two) and the ablation is more rigorous. The main weakness—the training epoch mismatch—is real and unresolved, creating uncertainty about the exact magnitude of gains, but the internal ablation (static local vs. dynamic max, same epoch budget for all ablation variants) still supports the core claim. Weighting against the 7.25+ cluster due to the methodological gap and incremental nature, and above the 5.0 mark given substantively stronger contributions, I place this paper at **6.5**.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>