## Summary
This paper proposes a DWT-based probe for studying whether ViT representations can be reconstructed from representations of wavelet-decomposed image primitives. Concretely, it decomposes images into DWT sub-bands, feeds each reconstructed sub-band image through a frozen ViT, and learns a simple linear composition function over the resulting last-layer representations. The main empirical finding is that for one-level DWT decomposition, such a learned composition nearly recovers the original ViT’s classification behavior, whereas naive summation fails badly; this effect weakens substantially for two-level decomposition.

## Strengths
- **The use of DWT to define input-dependent image primitives is a concrete and original probe design.** The paper identifies a real challenge in translating compositionality analyses from language to images—there is no obvious discrete dictionary of “parts” in pixel space—and uses DWT’s invertible sub-band decomposition as a principled workaround (Sec. 2.3, 3.3). That is a specific and nontrivial contribution.
- **The strongest empirical result is genuinely interesting:** for one-level DWT decomposition, a very small learned linear composition almost recovers the original classifier’s performance, while direct summation catastrophically fails. In Table 1, e.g. ViT-B/Haar goes from 0.13 (“Summed”) to 0.775 (“Learned”) against 0.792 original; ViT-L/Haar reaches 0.797 against 0.809 original. This is a real signal worth reporting.
- **The paper includes useful sanity checks beyond headline accuracy.** In particular, Table 5 shows the low-pass coefficient alone is insufficient (0.494 vs. 0.771 learned convex on ViT-B Haar level-1), supporting the claim that detail bands contribute nontrivially rather than the result being entirely explained by the LL band.
- **The result is at least somewhat consistent across the tested settings.** The same level-1 pattern appears across ViT-B and ViT-L and across Haar/db4, while level-2 consistently degrades, suggesting the phenomenon is not a one-off artifact of a single model/basis pair.

## Weaknesses
###: Fatal
- **The paper’s central claim is overstated relative to what is actually evaluated.** The paper defines compositionality in representation-space terms (Sec. 2.2–2.4), but the main objective in Eq. (5) optimizes and evaluates *post-classifier outputs* \(E_c(\cdot)\), not direct similarity of the encoder representations themselves. As written, the evidence supports a weaker claim—*a linear combination of primitive last-layer features can recover much of the original classification behavior*—not that the ViT encoder representations “satisfy compositionality in the representation space.” This mismatch undermines the headline claim in the abstract, contributions, and conclusion.

### Major:
- **There is a direct claim/evidence mismatch on patch representations.** The abstract and introduction claim that “ViT patch representations at the last encoder layer are compositional,” but Sec. 4 explicitly states: “we extract the encoder layer representations \(E_l(\tilde{I})\), **specifically the cls token** of the representation, for each wavelet coefficient.” No patch-token composition experiment is reported. This is not just imprecise phrasing; it is unsupported by the paper’s own method.
- **The current evidence is not diagnostic of DWT-specific compositionality because there are no alternative or null decomposition baselines.** Since the learned \(g^*\) is just a tiny linear combination over a handful of primitive CLS embeddings, near-original level-1 performance could reflect generic recoverability from multiple transformed views rather than something special about DWT-induced compositional structure. A comparison against at least one alternative basis (e.g., Fourier/PCA/random orthogonal decomposition) is needed to support the stronger interpretation.
- **The theoretical framing and empirical operationalization are not well reconciled.** The paper starts from a homomorphism-based notion of compositionality, shows direct additive reconstruction fails, then replaces it with a learned linear composition trained through the classifier head. That move is reasonable as a probe, but it materially changes the construct being measured. The brief justification that direct representation-space metrics are unreliable due to “curse of dimensionality” is too weak to dismiss direct latent comparisons entirely, especially when CKA is already used earlier in the paper.
- **The scope of the positive conclusion is narrow, but the framing is broad.** The main positive result holds for one-level DWT decomposition at the last layer and using CLS-token composition. Two-level decomposition drops sharply (e.g., ViT-B from ~0.77 to ~0.51 in Table 1), and the paper does not explain whether this reflects a failure of linear composition, a property of deeper DWT trees, or a limitation of the probe. The title/abstract read more broadly than the actual support warrants.

### Minor
- **The layerwise story is underdeveloped despite suggestive preliminary results.** Figure 2 shows a marked CKA peak in earlier layers and then a collapse to ~0.4 after layer 5, yet the main experiments are restricted to the last layer. Since the paper claims to offer a framework for ViT encoder compositionality, not exploiting the layer dimension leaves an obvious gap.
- **The analysis of learned weights is interesting but incomplete.** Tables 3–4 show strong low-pass dominance and high variability across constraints/settings, and the paper itself notes “there is no discernible pattern among the parameters.” That non-uniqueness may be telling, but it is not probed further, so it is hard to know whether the learned composition reveals stable structure or just one of many correlated solutions.
- **“Relative accuracy” in Table 2 is only a weak auxiliary metric.** Because the target is agreement with the original model’s prediction rather than ground truth, and training already optimizes toward the original classifier output, this metric is favorable to the probe and should not be treated as strong evidence of representational equivalence.

### Trivial
- **Some peripheral analyses are only loosely connected to the main claim.** For example, reconstructing images in pixel space using latent-space learned weights (Sec. 4.3) is interesting descriptively, but its connection to the formal compositionality question is not especially tight.

## Nice-to-Haves
- Add direct representation-level evaluations at the last layer (e.g., cosine similarity, MSE, CKA/RSA on CLS and patch tokens) to align the measurement with the stated claim.
- Evaluate at intermediate encoder layers, especially given the non-monotonic CKA pattern in Figure 2.
- Compare DWT against at least one alternative decomposition and one null baseline.
- Test non-linear composition functions to determine whether the level-2 failure is due to the linear restriction or a genuine limit of the representation structure.
- Expand the low-pass/detail ablation beyond the single setting in Table 5.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Requests for missing related work.** Some reviewer comments asked for additional prior work comparisons. Per instruction, I am not using unverified missing-citation critiques.
- **Formatting/style issues and typos.** These are not decision-relevant.
- **Generic reproducibility complaints.** The paper provides core setup details and code availability; absence of exhaustive implementation minutiae is not a substantive weakness here.
- **Demand for confidence intervals/seeds as a core flaw.** While variance reporting would improve the paper, this is not standard enough in this setting to be a decisive criticism; it is better viewed as a nice-to-have.
- **Critiques about unreleased or unverifiable cited artifacts.** Ignored by instruction.
- **The Human Finder point about missing generalization tests as a core weakness.** This paper is framed as a representation analysis/probing paper, not a paper claiming compositional generalization improvements. It is fair to mention practical implications are underexplored, but not to require OOD generalization experiments as a central flaw.

## Novel Insights
The most interesting synthesis across the evidence is that the paper has uncovered a real but narrower phenomenon than it claims: ViTs appear to preserve enough class-relevant information across one-level wavelet sub-band views that a tiny linear recombination of sub-band CLS embeddings can reconstruct the original decision function surprisingly well, even though naive addition fails and deeper wavelet decompositions degrade sharply. That pattern suggests the result may be more about the stability and redundancy structure of last-layer decision features under coarse frequency-space factorization than about compositionality in the formal homomorphism sense used in the paper. In other words, the empirical finding is likely publishable in spirit, but the interpretation attached to it is currently too strong.

## Suggestions
- Narrow the main claim throughout the paper to match the actual evidence: emphasize recovery of last-layer **classification behavior from DWT primitive CLS representations**, not representational compositionality in general.
- Remove or substantiate all claims about **patch representations**; as written, they are unsupported.
- Add at least one **direct latent-space** evaluation for the learned composition, ideally on both CLS and patch tokens.
- Add one **alternative decomposition baseline** and one **null baseline** to test whether DWT is actually special.
- Exploit the existing framework to run the same analysis **across layers**, since Figure 2 already suggests that the story may be layer-dependent.
- Clarify the exact parameterization of \(g^*\) early and explicitly, since the simplicity of the probe is central to interpreting the result.

## Score and Decision
**Novelty:** Moderate. The DWT-based probing angle is original and the level-1 recovery result is interesting.  
**Technical soundness:** Mixed. The experiments themselves are mostly coherent, but the central interpretation does not follow from the operationalization.  
**Empirical support:** Moderate for the narrow finding, weak for the stronger compositionality claim.  
**Significance:** Limited-to-moderate in current form; the result is intriguing, but its scope is narrow and the paper overstates what it establishes.  
**Clarity:** Reasonably clear overall, but the paper blurs the distinction between representational compositionality and classifier-output recoverability, and overclaims patch-level conclusions.

Relative to the calibration examples, this looks stronger than clearly flawed low-score rejects, because there is a concrete and nontrivial empirical signal here. But it falls short of accept because the main claim is materially overstated and the evidence does not measure the construct the paper says it measures. That puts it in the mid-reject range rather than near-borderline accept.

MY FINAL SCORE: <pineapple>4.8</pineapple>
MY FINAL DECISION: <orange>Reject</orange>