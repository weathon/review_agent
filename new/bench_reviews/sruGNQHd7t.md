Now let me do calibration searches to anchor my scoring.Now I have all the information I need to write the final review. Let me synthesize everything.

---

## Summary

The paper introduces a user-side privacy-preserving framework for cloud-based deep learning inference through "domain shifting." Users encode their inputs before querying a remote DNN, then decode the obfuscated outputs to recover the original classification. Two approaches are proposed: model-specific out-of-place shifting (whitebox access, using a trained encoder/decoder pair with oracle model architecture) and model-agnostic in-place shifting (blackbox access, using GAN or GAN+DDPM to map inputs to visually different but legitimate domain images). Experiments span MNIST, Fashion-MNIST, CIFAR-10, Tiny-ImageNet, and ImageNet-20 across multiple oracle architectures.

---

## Strengths

- **User-side-only privacy framework**: The paper correctly identifies a practical gap — existing HE/MPC schemes require server-side modifications, while this approach requires none. Section 3 formalizes this distinction well, and it represents a genuinely underexplored angle for inference privacy.

- **Model-agnostic generalization demonstrated across oracle architectures**: Table 4 shows the same GAN encoder working across MLP and ViT oracles on MNIST (97.28% vs. 97.49% fidelity) and across ViT_H_14 and Swin_V2_B on ImageNet, validating the model-agnostic claim.

- **High fidelity with near-zero SSIM² for model-specific approach**: Table 2 reports SSIM² as low as 1.94×10⁻⁸ (MNIST/ViT, row ③) alongside 98.21% fidelity, and even super-oracle fidelity on Tiny-ImageNet (e.g., 87.36% vs. 78.81% oracle for Swin, row ⑥). This demonstrates technical feasibility.

- **Logical necessity argument for in-place model-agnostic design**: Section 4.3 provides a principled impossibility argument explaining why out-of-place shifting cannot be model-agnostic (different oracles give inconsistent predictions on non-domain inputs), motivating the in-place design. This is a sound theoretical grounding.

- **Significant inference speed advantage**: Section 5.3 reports 0.5ms (MNIST, model-specific) and 1.2ms (CIFAR-10) vs. 481ms–3.58s for HE/MPC; the GAN-only model-agnostic is similarly fast at ~0.35ms overhead.

---

## Weaknesses

### Fatal
*None — the paper's core ideas are sound and the experiments confirm that the encoder-decoder pipelines work. However, see Major weaknesses below, which together substantially undermine the privacy claims.*

### Major

- **SSIM is not a privacy metric — the headline privacy claim is unverified.** The paper's sole quantitative privacy evidence is SSIM² between original and encoded images (Table 2) and an SSIM-based uniformity measure for the model-agnostic setting (Table 4). SSIM measures visual distortion, not information leakage. A low SSIM between x and EN(x) means the encoded image looks different, but says nothing about whether an adversary can reconstruct the original image, infer its class via statistical analysis of multiple queries, or learn the encoder mapping. No reconstruction attack, membership inference test, or inversion attack of any kind is presented. For a paper submitted to a privacy track, the absence of adversarial evaluation is a critical gap: "privacy is preserved" based on SSIM alone is not a defensible claim. The most important missing experiment is: given a set of (EN(x), y^ob) pairs, can an adversary trained on public data recover x or its class label?

- **The formal training objective in Equation (1) contains a type error that propagates through the method description.** The surrounding text (Section 4.2) states explicitly that obfuscation is measured "through SSIM between real input x and obfuscated input x^ob = EN(x)." However, Equation (1) writes $L_{ob} = \mathbb{E}[SSIM^2(f(x), EN(x))]$ where f(x) is the oracle model's classification output (a scalar class label or logit vector). Equation (2) then defines $\mu_{f(x)}$ as the "mean intensity values of the original input f(x)" — which is dimensionally inconsistent since f(x) is not an image. The paper almost certainly implemented SSIM(x, EN(x)) correctly in practice, but the formal statement of the training objective — the core technical contribution — is wrong as written. For a method paper where the training loss is central, this error in the formalization is not a trivial typo.

- **No formal privacy definition or adversary model.** Section 3 specifies the adversary as "honest-but-curious" but never specifies the adversary's goal (reconstruct x? infer class of x? link multiple queries to the same user?) or computational capabilities. Without a formal privacy objective, it is impossible to evaluate whether the method succeeds. The claim "no class information about x can be inferred from x^ob alone" (Section 4.3) is stated but never tested even empirically, let alone proved.

### Minor

- **Whitebox threat model requires clarification.** The paper states users have "whitebox access" including "access to the internal model parameters, including detailed logit scores... and can calculate gradients." Section 5.1 then describes building the encoder using "a few multi-head attention layers or convolution layers of the oracle model." If this requires full oracle model weights locally, the threat model is nearly self-defeating: a user who can run oracle model layers locally could also just run the full model locally, making cloud querying unnecessary. The paper should clarify whether only the architecture (not the trained weights) is used for the encoder structure, or whether the whitebox encoder uses the oracle's actual trained layers.

- **Significant accuracy drops on complex datasets are underreported.** Table 4 shows ~8.7pp drop on CIFAR-10 (88.91% → ~80.2%) and ~13-14pp drop on ImageNet-20 (88.55%/84.12% → 75.10%/70.40%) for the model-agnostic GAN+DDPM approach. These are described as "some accuracy drops" without analysis of root causes or whether they are fundamental to the GAN+DDPM translation quality. A 13pp accuracy penalty would likely be unacceptable in practice and the paper does not address this.

- **Cross-architecture fidelity degradation is unexplained.** Table 2 rows ⑦–⑧ (cross-architecture: ConvNeXt encoder with Swin oracle and vice versa) show fidelity drops to 77.22% and 80.09%, compared to 86.86% and 87.36% for matched architectures. The paper notes these results but provides no explanation. This suggests the model-specific encoder encodes architecture-specific assumptions that don't generalize even to closely related architectures — an important practical limitation.

- **DDPM overhead of ~4 seconds per image is not practically viable for high-throughput services.** The paper acknowledges the 4.12s overhead but frames it positively against HE/MPC baselines. However, 4 seconds per query is 400–4000× slower than the oracle itself for complex datasets, which would be prohibitive in any real-time or production cloud service. The comparison to HE/MPC (which also provides cryptographic guarantees that this work does not) is misleading without acknowledging the security-guarantee asymmetry.

### Trivial

- The obfuscation loss in Eq. (1) should use x (original input) rather than f(x) (oracle output) — this equation-level notation error should be corrected regardless of whether the implementation was correct.

---

## Nice-to-Haves

- **Fidelity-privacy tradeoff curve**: Varying α across at least one non-trivial dataset and plotting fidelity vs. SSIM² would characterize the operating range of the method far better than a single operating point at α = 0.01.
- **Statistical de-anonymization experiment**: For the model-agnostic scheme, test whether an adversary seeing k encoded queries can recover i (the permutation offset) for different values of k. Even if i is refreshed per query, this would establish an empirical bound on how many queries are safe.
- **Content leakage analysis**: For in-place shifting, analyze whether encoded images (e.g., T-shirt → coat) preserve body-shape, texture, or fine-grained visual features from the original. If significant visual content persists, the privacy protection is weaker than class-label shuffling implies.
- **Formal privacy framing**: Even an informal information-theoretic argument (e.g., mutual information between x and EN(x) is bounded by some quantity) would substantially strengthen the paper's positioning.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic Issue 2 — "3.3-bit key entropy makes the scheme trivially breakable"]**: The paper states "Our encoder first generates a random number i from the set 0, 1, ..., M-1 with equal probability" — this implies per-query fresh randomization. With fresh i per query, the adversary sees independent (EN_i(x), y^ob) pairs where each uses a different unknown i, so i cannot be recovered by correlating multiple queries (each has its own independent i). The critic's attack scenario ("same i is reused") is not what the paper describes. The real content-leakage concern (that visual features of x persist in EN(x)) is kept as a Nice-to-Have above, but the specific entropy-counting argument is unfair to the scheme as described. — **Removed.**

- **[Harsh Critic Issue 4 — Whitebox setting "self-defeating"]**: This is kept but weakened above (moved to Minor). The critic's framing is too strong — the whitebox setting can be interpreted as using public architectural knowledge, not necessarily full local model weights. — **Partially retained as a Minor clarification request.**

- **[Strength Finder — "Comprehensive evaluation across 5 datasets and 8+ oracle architectures as a standalone strength"]**: This is accurate and mentioned in context throughout, but it is a supporting detail, not a novel insight. Absorbed into the Major weakness about accuracy drops and cross-architecture degradation. — **Absorbed elsewhere.**

---

## Novel Insights

The most genuinely novel aspect of the paper is the logical argument in Section 4.3 that model-agnostic out-of-place shifting is *impossible*: since different oracle models disagree on non-domain inputs, no universal decoder can exist for out-of-place encoded inputs. This impossibility motivates in-place shifting and explains why the two-approach framework (model-specific/model-agnostic) is a necessary dichotomy rather than an arbitrary choice. This reasoning is clean and original. However, the empirical privacy evaluation falls short of the theoretical ambition — the framework would benefit substantially from adversarial evaluation aligned with this formal reasoning.

---

## Suggestions

1. **Add at minimum one adversarial experiment**: Train a simple inversion network on (EN(x), x) pairs from a public dataset, then test it on held-out encoded queries. If the inversion network fails to recover x or its class label, this is meaningful evidence of privacy. If it succeeds, the paper's claim is falsified.
2. **Fix Equation (1)**: Replace f(x) with x throughout the SSIM loss formulation and Eq. (2).
3. **Clarify key management**: Explicitly state whether i is drawn fresh per query (which makes the scheme secure in the output space) or reused across queries (which degrades security dramatically).
4. **Clarify whitebox access**: State whether the model-specific encoder uses the oracle's trained weights or only its architecture. If the former, discuss the practical availability assumption.
5. **Reframe the overhead comparison**: When comparing latency with HE/MPC, note that those methods provide cryptographic security guarantees that this work does not, so the comparison is at different security levels.

---

## Score and Decision

**Calibration anchors consulted:**

| Path | Avg Human Score | Comparison to this paper |
|---|---|---|
| `/human_reviews/SX2Z5tgiUu.md` (PrivateChat) | 4.00 (Reject) | Most similar: user-side encryption/obfuscation for cloud AI queries, rejected for weak threat model and SSIM-based privacy evaluation instead of cryptographic guarantees |
| `/human_reviews/lPJUQsSIxm.md` (DCT-CryptoNets) | 6.33 (Accept) | HE-based inference with latency improvements — provides formal cryptographic guarantees; much stronger privacy evaluation foundation |
| `/human_reviews/mUMvr33FTu.md` (CipherPrune) | 6.25 (Accept) | MPC-based private Transformer inference — formal security + solid engineering contributions |
| `/human_reviews/HMe5CJv9dQ.md` (DP similarity) | 7.50 (Accept) | Strong theoretical DP contributions with rigorous proofs — clearly above this paper |
| `/human_reviews/cPmLjxedbD.md` (low-scoring) | 1.00 (Reject) | No experiments, no methodology — clearly below this paper |
| `/human_reviews/OXIIFZqiiN.md` (low-scoring) | 1.50 (Reject) | LLM-generated nonsense — clearly below this paper |
| `/human_reviews/JKpk2p4O99.md` (unlearnable examples) | 5.25 (Borderline) | Input-perturbation based privacy scheme at similar scope level |

**Rationale:** This paper is most analogous to PrivateChat (score 4.0): both propose user-side obfuscation for cloud AI queries, both rely on heuristic/non-cryptographic privacy metrics (SSIM, embedding similarity) rather than formal adversarial evaluation, and both have weak threat models. This paper is somewhat broader empirically (5 datasets, 8+ oracles vs. PrivateChat's narrower scope), and the formal impossibility argument in Section 4.3 is a genuine theoretical contribution that PrivateChat lacks. However, the privacy evaluation gap is equally severe, the Eq. (1) notation error is unique to this paper, and the accuracy drops on complex datasets are significant and underanalyzed. The score-6 papers in this domain (DCT-CryptoNets, CipherPrune) provide cryptographic guarantees and rigorous privacy analysis — this paper falls substantially short of that bar. Placing between PrivateChat (4.0) and the borderline input-perturbation paper (5.25), with the severity of the privacy evaluation gap pulling toward the lower end: **4.0**.

**Axis evaluation:**
- *Originality*: Moderate — the encoder-decoder framing for user-side privacy is a genuine and clean idea, but not deeply novel
- *Importance of research question*: High — user-side privacy for inference is practically important
- *Claim support*: Weak — the central privacy claim lacks adversarial evaluation; the core training objective has a notation error
- *Soundness of experiments*: Moderate — fidelity experiments are sound but privacy "experiments" use SSIM rather than adversarial tests
- *Clarity of writing*: Fair — the general structure is clear but the formal notation in Section 4.2 is internally inconsistent
- *Value to research community*: Low-to-moderate — the framing and GAN+DDPM architecture are interesting contributions, but without proper privacy evaluation the paper cannot be used as a reliable reference

**Decision: Reject.** The paper addresses an important problem and demonstrates that the encoder-decoder approach can preserve utility, but it does not establish that the approach preserves privacy in any meaningful adversarial sense. The lack of adversarial evaluation and the use of SSIM as a privacy metric are fundamental gaps for a paper whose headline contribution is privacy preservation. These are not gaps resolvable in a rebuttal — they require additional experiments.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>