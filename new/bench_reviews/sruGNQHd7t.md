---

## Summary
This paper proposes a user-side privacy-preserving scheme for querying cloud-based deep learning classifiers via "domain shifting." Users encode inputs before sending them to an oracle model and decode the outputs locally. Two variants are introduced: (1) *model-specific out-of-place shifting* (whitebox access, pushes inputs outside the natural domain) and (2) *model-agnostic in-place shifting* (blackbox access, uses GAN/DDPM to translate inputs into different classes via a permutation cipher). The conceptual split between these two modes is the paper's main intellectual contribution.

---

## Claims and Support

**Claim 1: The pipeline preserves oracle utility with "minimal impact on classification performance."**
Partially supported. On MNIST and Fashion-MNIST, pipeline accuracy is within 1–2% of oracle. On CIFAR-10 (model-agnostic), accuracy drops from ~89% to ~80% (−9 pp). On ImageNet-20 (model-agnostic), drops are 88.55%→75.10% and 84.12%→70.40% (~13–14 pp). These are not "minimal." The claim is valid for simple datasets but overclaims for harder ones. Additionally, "fidelity" as defined in Section 5.1 is oracle–pipeline *agreement*, yet rows ④–⑥ of Table 2 report fidelity *exceeding* oracle accuracy (e.g., 90.56% vs oracle 88.91%, 87.36% vs 78.81%), which is unexplained under the stated definition and makes the utility evidence difficult to interpret.

**Claim 2: The method is privacy-preserving.**
Unsupported. The paper uses SSIM² between original and encoded images as its sole privacy proxy. However, the SSIM formula itself contains a notation error: Equations (1) and (2) write `SSIM[f(x), EN(x)]`, but `f(x)` is defined in Section 3 as a class label, not an image. The description in Eq. (2) then contradicts Section 3 by treating `f(x)` as having "mean intensity values" — strongly suggesting the intended formula is `SSIM[x, EN(x)]`. Beyond the notation issue, low SSIM does not imply semantic non-recoverability; no inversion, reconstruction, attribute-inference, or membership-inference attack is evaluated. This is a critical gap for a paper whose central claim is privacy.

**Claim 3: Model-specific fidelity is high.**
Partially supported for MNIST and CIFAR-10, but the fidelity-exceeding-oracle-accuracy anomaly in Table 2 (rows ④–⑥) is not addressed or explained and undermines confidence in the metric definition.

**Claim 4: Model-agnostic encoding generalizes across oracle models.**
Weakly supported. Only two oracle models per dataset are tested; performance on CIFAR-10 and ImageNet degrades substantially. The Section 5.2 statement "we calculated the SSIM between the true class of the input images and the oracle model's classification of the encoded images" is nonsensical as written — SSIM is an image similarity metric and cannot be applied to class labels. This appears to be an additional description error.

**Claim 5: The approach is faster than HE/MPC.**
Misleading as framed. For the GAN-only variant (MNIST/Fashion-MNIST), overhead is indeed small (~0.3–0.4ms). For GAN+DDPM (CIFAR-10/ImageNet), overhead is ~4.1 seconds per query. The cited HE/MPC comparisons use old results from 2017–2018; more importantly, the approaches provide fundamentally different security guarantees, so the latency comparison is not like-for-like.

---

## Strengths

- **Principled decomposition of access models:** The paper correctly identifies that model-agnostic encoders *must* use in-place shifting to ensure class-label consistency across different oracle models, and provides a clear motivating example (Figure 1) showing cross-model inconsistency for out-of-domain inputs. This conceptual contribution is non-trivial and practically useful.

- **User-side privacy without provider cooperation:** Unlike HE/MPC, the proposed scheme requires no modification to the service provider's infrastructure. This is a genuinely different and practical deployment point, especially for organizations constrained from modifying third-party services.

- **GAN+DDPM combination for image-to-image class translation:** The two-stage pipeline (GAN produces an embedding, DDPM conditions on that embedding to generate a class-shifted image) is a well-motivated engineering design that addresses quality limitations of pure GAN approaches on complex datasets.

- **Cross-architecture generalization for model-specific case:** Rows ⑦ and ⑧ of Table 2 show that an encoder trained for one backbone (ConvNeXt or Swin) maintains reasonable fidelity (77–80%) when evaluated on a *different* oracle backbone — a modestly encouraging result not highlighted enough.

---

## Weaknesses

### Fatal
*(None that would classify this as "not a paper" — a real contribution exists — but the combination of the following major issues is cumulatively severe.)*

### Major

- **Privacy is not evaluated under the stated threat model.** The paper's central claim is privacy preservation, but the only privacy proxy is SSIM (and, for model-agnostic, class-distribution uniformity in Figure 3). Neither demonstrates resistance to realistic attacks. Under the honest-but-curious threat model of Section 3, the provider sees many encoded queries and can train an inversion model, surrogate decoder, or do nearest-neighbor lookups in a public dataset. None of this is tested. Class-distribution uniformity only shows that coarse class labels are concealed; it says nothing about instance-level appearance, fine-grained attributes, or invertible latent structure. For a paper whose headline contribution is "privacy-preserving," this is a decisive gap.

- **Notation error in the core privacy loss function.** Equations (1) and (2) define `L_ob = E[SSIM²[f(x), EN(x)]]`, where the surrounding text describes `f(x)` as having "mean intensity values" and other image statistics. But Section 3 defines `f(x)` as a scalar class label. The intended formula is `SSIM[x, EN(x)]`. Because Eq. (1) defines the training objective for the encoder, this error in a load-bearing equation undermines confidence in the described method matching the experiments.

- **Fidelity exceeding oracle accuracy is unexplained.** Table 2, rows ④–⑥ show pipeline "fidelity" (defined as pipeline–oracle agreement) of 90.56%, 86.86%, and 87.36% against oracle accuracies of 88.91%, 81.18%, and 78.81%, respectively. Under the stated metric (agreement between pipeline output and oracle output), fidelity is bounded by oracle accuracy only if the two are correlated on the same inputs — but fidelity *exceeding* oracle accuracy by up to 8.5 pp suggests either the metric is actually reporting ground-truth accuracy (contradicting the definition), or there is an evaluation artifact. The paper does not address this, and the discrepancy makes the core empirical results for utility uninterpretable.

- **Model-agnostic SSIM reporting in Section 5.2 is inconsistent and incorrect.** The text says "we calculated the SSIM between the true class of the input images and the oracle model's classification of the encoded images." SSIM requires image inputs, not class labels. This appears to be a description error, but it affects the reader's ability to understand what was actually measured and reported in Table 4. Furthermore, SSIM² values in Table 4 (0.003–0.077) are orders of magnitude higher than in Table 2 (10⁻⁸–10⁻⁵), yet the paper does not discuss this large discrepancy in obfuscation quality between the two approaches.

### Minor

- **GAN+DDPM latency is impractical for real-time use.** At ~4.1 seconds per query vs. oracle inference of ~10ms, the latency overhead is ~400×. The claim that this is "acceptable" because it beats HE methods from 2017–2018 is weak; modern lightweight MPC/HE implementations and the commercial setting (where ML inference is cheap) change the comparison. The paper should be honest that GAN+DDPM encoding is suitable only for offline or batch applications.

- **Whitebox access requirement contradicts the central motivation.** The paper motivates protecting privacy from cloud service providers who "have no control" over users' data, yet model-specific transform training requires whitebox access to the oracle (parameters, gradients). Commercial cloud providers (AWS, GCP, etc.) do not expose model internals. The paper should more clearly distinguish the whitebox use case (e.g., in-house deployment) from the blackbox use case (commercial cloud) rather than treating both as equally valid instances of the same motivation.

- **Model-agnostic generalization is tested on only two oracle models per dataset.** For a method advertised as "model-agnostic," this is thin evidence. Both tested models per dataset are from similar families (ViT and Swin), often trained on the same benchmark. Diversity in oracle architecture, training recipe, and distribution shift is needed to substantiate the broader claim.

- **Performance on complex datasets undermines practical relevance.** The ~13–14 pp accuracy drop on ImageNet-20 for the model-agnostic approach is large enough that the method provides limited practical benefit for the exact settings (powerful cloud models, sensitive data) where privacy matters most.

### Trivial

- The paper reports only batch-size-1 inference times. Batch-size-1 latency is not representative of throughput in realistic service deployments.

---

## Nice-to-Haves

- Conduct at least one inversion/reconstruction attack experiment (e.g., train a simple convolutional inverter on encoded→original pairs) to empirically lower-bound the difficulty of recovering originals from encoded inputs.
- Replace or supplement SSIM with an adversarially meaningful privacy metric (e.g., accuracy of a trained attribute classifier on encoded inputs, or mutual information estimates) aligned to the stated threat model.
- Report both pipeline-oracle agreement *and* pipeline ground-truth accuracy separately so that Table 2 utility results are unambiguous.
- Evaluate the model-agnostic method against a broader, heterogeneous set of oracle architectures (e.g., ResNets, EfficientNets, ConvNeXt) to substantiate the "model-agnostic" claim.
- Analyze multi-query security: since the permutation index `i` is the only secret in the model-agnostic scheme, an adversary seeing many queries may infer `i` from accumulated class-frequency statistics.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic – "White-box information does not provide advantage claim is too strong."** The paper states this in the context that model internals vary across oracles, so model-agnostic encoders shouldn't rely on them. This is presented as architectural motivation, not a formally proved theorem. It's fair to note this is intuition rather than proof, but the harsh critic elevated it to a fatal flaw — the paper is transparent that this is motivational reasoning.

- **Harsh Critic – lack of variance/multi-seed reporting.** While statistically rigorous reporting would be ideal, single-run evaluation is standard practice for large benchmark evaluations in this subfield. This is MOVED as a nice-to-have per soft rules.

- **Spark – "GAN+DDPM latency comparison uses outdated HE baselines."** Valid point but the paper uses the cited latency numbers as they appear in referenced papers — criticizing the currency of those external numbers is outside the authors' control and partly speculative. The broader concern (comparison is unfair because of different security guarantees) is already captured in the main weaknesses.

- **Human Finder – "No comparison with perturbation-based privacy methods."** No external sources confirmed as to what perturbation-based methods exist and are directly comparable; removed per instructions on missing related works.

- **Neutral Reviewer Strength – "Strong empirical results on simple datasets."** Retained in compressed form above. The generic framing "extensive experiments" is removed.

- **Neutral Reviewer Strength – "Clear presentation."** Removed as a generic formatting praise; not specific to this paper's contributions.

---

## Novel Insights

The paper's most novel conceptual contribution — the formal necessity argument that *model-agnostic* encoders must use in-place shifting because out-of-domain inputs produce inconsistent cross-model predictions — is genuinely useful. This decomposition (in-place/model-agnostic vs. out-of-place/model-specific) provides a structured lens for the design space of user-side input obfuscation that goes beyond prior work treating obfuscation as a single undifferentiated category. The GAN+DDPM composition for controllable image-class translation as a privacy primitive is also a creative application, though its practical limitations (4-second latency, significant accuracy degradation on complex data) presently outpace the theoretical appeal.

---

## Suggestions

1. **Fix Equations (1) and (2).** Replace `f(x)` with `x` throughout the SSIM formula to correctly define the obfuscation loss as measuring dissimilarity between original and encoded images.
2. **Fix Section 5.2 SSIM description.** Clarify exactly what image pairs are used to compute SSIM² in the model-agnostic setting, and explain the orders-of-magnitude difference vs. Table 2.
3. **Explain or redefine the fidelity metric in Table 2.** If fidelity is ground-truth accuracy of the pipeline, rename it. If it is oracle–pipeline agreement, explain why it exceeds oracle accuracy in rows ④–⑥.
4. **Add at least one adversarial privacy evaluation.** Even a simple trained convolutional inverter (trained on encoded→original pairs using a held-out split of the public training data) would lend concrete empirical grounding to privacy claims.
5. **Add a section on multi-query attacks.** Analyze how many queries are needed to statistically infer permutation index `i` from observed output class frequencies.
6. **Reframe the HE/MPC comparison.** Explicitly state that the comparison is between different security guarantee levels and that the latency advantage comes at the cost of weaker (heuristic, not cryptographic) privacy.

---

## Evaluation on Key Axes

- **Novelty:** Moderate. The in-place/out-of-place conceptual split is original; the components (GAN-based image translation, DDPM, permutation cipher) are established techniques.
- **Technical soundness:** Weak. Core equations contain a notation error, the privacy metric is not a privacy metric, and metric definitions are inconsistently applied.
- **Empirical support:** Weak to moderate. Results are reasonable on simple datasets, but the main privacy evaluation is missing and utility degrades substantially on harder tasks.
- **Significance:** Limited in current form. The approach has practical appeal but cannot be accepted as a privacy-preserving method without adversarial evaluation.
- **Clarity:** Mixed. The conceptual exposition is clear; the mathematical notation (especially around the SSIM loss) and the metric definitions are confused enough to impede reproducibility and interpretability of results.

## Score and Decision

The paper presents a genuinely useful conceptual framework and a practically motivated problem, but fails at its core responsibility: establishing that the proposed scheme actually preserves privacy. The SSIM-based privacy evaluation is not only insufficient but contains a notation error in its definition. The anomalous "fidelity > oracle accuracy" results in Table 2 and the inconsistent SSIM description in Section 5.2 further erode confidence in the empirical claims. The performance degradation on harder datasets (13–14 pp on ImageNet-20) and the impractical GAN+DDPM latency (~400× overhead) additionally limit practical significance. These are not minor revision issues — the privacy evaluation needs to be redesigned from scratch with adversarial evaluation, and core metric definitions need to be corrected. The paper does not meet ICLR's bar for technical soundness in its current form.

**Score: 3.5 / 10 — Reject**

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>