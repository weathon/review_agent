Now I have all the information needed. Let me compile my final review.

## Summary

The paper introduces Language-Assisted Feature Transformation (LAFT), a training-free method that projects CLIP image features onto "concept subspaces" derived from pairwise differences of text embeddings, enabling users to control which attributes anomaly detection is sensitive to. LAFT offers two modes: *guide* (project onto the concept subspace to retain relevant attributes) and *ignore* (orthogonal projection to remove nuisance attributes). The method is combined with kNN for semantic anomaly detection (LAFT AD) and with WinCLIP for industrial anomaly detection (WinCLIP+LAFT).

## Strengths

- **Elegant and training-free core idea:** Projecting CLIP image features onto subspaces defined by text embedding differences (Eq. 3–5) is simple, intuitive, and requires no additional training — a genuine practical advantage over methods like InCTRL and APRIL-GAN that require adapter pre-training. This is clearly demonstrated across multiple datasets (Tables 1–3).

- **Strong guide-mode results on semantic AD:** LAFT AD achieves 95.6 AUROC on Waterbirds (vs. 82.3 for the subset kNN baseline and 92.2 for the best language-guided competitor ZOE), 98.5 on Colored MNIST, and 95.0/98.1 on CelebA (Table 1 and 2). These are convincing improvements showing the guide mode adds real value beyond merely providing more training data.

- **Ablation on prompt quality (Table 4):** Demonstrates robustness to incomplete concept value specifications — performance with only "seen" normal values (96.2 on Colored MNIST number) remains close to the full specification (98.5), which is an important practical property.

- **Clean formal framing:** Eqs. 1–2 define invariance and informativeness properties that provide a principled foundation for the method, even if empirical measurement of these properties is incomplete.

## Weaknesses

### Fatal
None.

### Major

- **The "ignore" mode — presented as an equal, symmetrical contribution — substantially fails on the primary real-world dataset:** On Waterbirds (Table 1), ignore (background) achieves 84.8 AUROC vs. 95.6 for guide (bird), and only marginally exceeds kNN with all normals (83.0). On Colored MNIST, ignore works well (97.4), but this is a toy dataset where attributes are independent. The paper presents guide and ignore as two equally principled instantiations of the same idea (Section 3, Eqs. 5–6), yet the evidence supports only guide. The paper's sole comment on this is: "ignoring one attribute (background) does not directly improve performance on the other attribute (bird)" (Section 5.1). This is exactly the scenario ignore mode is designed for. No analysis is provided to explain whether the concept subspace is impure (contains non-background information) or whether projecting out the background also removes bird-relevant information. This asymmetry significantly limits the claimed generality of LAFT.

- **Missing validation of core theoretical properties:** Section 3 explicitly promises empirical evaluation of invariance (Eq. 1) and informativeness (Eq. 2): "Empirical evaluations of these measures for various datasets are provided in the Experiments." Yet the experiments only report anomaly detection AUROC/AUPRC — they never directly measure whether transformed features become invariant to irrelevant attributes or retain information about relevant ones. This is a gap between the theoretical framing and experimental validation, and would have been straightforward to assess (e.g., predicting background from guide-transformed features to show invariance).

### Minor

- **Marginal improvements on industrial AD and conceptual mismatch with IAD scope:** WinCLIP+LAFT improves over WinCLIP+ by 0.5–1.5 AUROC points on MVTec AD/VisA (Table 3). Additionally, the IAD prompts (e.g., "a photo of a malformed bottle") define concept subspaces based on normal-vs-abnormal distinctions, which is a qualitatively different mechanism than the attribute-level concept guidance used in semantic AD. The paper presents LAFT as a unified method, but the semantic and industrial applications rely on substantively different uses of text embeddings.

- **No ablation on PCA dimension d despite its wide range:** The dimension d varies from 8–384 across settings (Section 4.2), a 48× range. The paper refers readers to an "Ablation Study" for the impact of d (line 164), but the actual ablation (Section 5.3) only studies prompt quality. Whether d requires per-task tuning — which would undermine the "training-free" convenience — is not analyzed.

### Trivial
None.

## Nice-to-Haves

- Analysis of why the ignore mode fails on Waterbirds (e.g., visualizing transformed features, measuring invariance to the background attribute from guide-transformed features), which would clarify the method's scope and strengthen future work.

- Separate reporting of invariance and informativeness metrics as promised in Section 3, which would validate the method's mechanism beyond end-task performance.

- Comparison against data augmentation baselines (e.g., Deep SAD with color augmentation on Colored MNIST) to contextualize the value of language guidance vs. traditional approaches.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"Ignore mode doesn't even outperform kNN baseline"** — This is factually incorrect: on Waterbirds, ignore achieves 84.8 vs. 83.0 for kNN with all normals, so it does outperform, albeit marginally. The broader point about ignore mode's weakness on Waterbirds is valid and retained above.

- **"LAFT-C requires knowing anomaly types in advance, undermining the 'no additional training data' framing"** — LAFT-C uses category-specific prompts with anomaly class names, but this is standard practice in zero/few-shot CLIP-based detection (WinCLIP uses similar prompts). The "no additional training data" claim refers to not needing labeled images, and the prompts are text-based knowledge, not training data. This criticism conflates two different things.

- **"IAD results don't validate the core claim about attribute-level language guidance"** — While the mechanism differs from semantic AD, exploring a different use of the same method on IAD datasets is a valid extension. The concern about conceptual mismatch is retained as a minor point above but should not be treated as invalidating.

- **"Colored MNIST results are near-trivial because digit and color are independent"** — This is a valid experimental choice for demonstrating the concept, and the paper explicitly introduces it as a "simplest way" demonstration. The Waterbirds and CelebA results provide the real-world validation.

- **"Requesting comparison with image-only methods on semantic AD"** — The paper provides a principled justification for this exclusion (Section 5): image-only methods without guidance detect all deviations from training data, leading to high FPR, and they simulate augmented image-only methods via kNN with all normal images. This is a fair design choice.

## Novel Insights

The ignore mode's failure on Waterbirds versus its success on Colored MNIST reveals a key insight that the paper does not fully develop: when attributes are entangled in the feature space (as backgrounds and bird types are in real-world images), orthogonal projection cannot cleanly remove one attribute without also degrading information about the other. The guide mode succeeds precisely because it only needs to *retain* the relevant subspace, not *purify* it. This asymmetry suggests that concept subspaces in CLIP's embedding space are not orthogonal across attributes — a finding with implications beyond this paper for any method attempting linear disentanglement in VLM feature spaces.

## Suggestions

- Reframe the paper to clearly position the guide mode as the primary contribution, with the ignore mode as an attempted extension that currently has limited applicability. This would improve honesty and still preserve the core contribution.
- Add an explicit invariance/informativeness measurement (as already promised in Section 3) by evaluating attribute prediction accuracy on transformed features — this would validate the mechanism beyond end-task metrics.
- Include an ablation on d (PCA components) in the main paper, since the values span a 48× range and this is critical for practical deployment.

## Calibration

**Anchors compared against:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| 5Ca9sSzuDp (CLIP decomposition, training-free) | 8.0 | More thorough analysis, deeper insights; LAFT is narrower in scope and has weaker mechanistic validation |
| buC4E91xZE (AnomalyCLIP) | 6.17 | Similar topic (CLIP for AD); LAFT is training-free (advantage) but has weaker empirical validation and overclaims ignore mode |
| cJs4oE4m9Q (DOHA, orthogonal projection for AD) | 8.0 | Similar projection-based idea but much stronger validation; LAFT's contribution is less polished |
| LCpayOuqBx (DOS for OOD) | 5.0 | Similar CLIP training-free approach but rejected for limited validation; LAFT has more results but similar overclaiming issues |
| zE4mL85zgg (CLAD) | 2.2 | LAFT is clearly stronger — genuine novelty, real results, and meaningful baselines |

LAFT's guide mode is a genuine and well-demonstrated contribution (simple, training-free, effective). Its weaknesses — the failing ignore mode, missing property validation, and marginal IAD improvements — place it below papers like 5Ca9sSzuDp (8.0) and AnomalyCLIP (6.17) but well above weak submissions like CLAD (2.2). The method itself is sound for the guide case; the main issue is overclaimed generality. I score it slightly below AnomalyCLIP due to the more significant weaknesses.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>