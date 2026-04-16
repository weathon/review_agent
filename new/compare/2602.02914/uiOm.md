
## Weaknesses

### Major:

- **Overgeneralization beyond the oracle threat model.** The paper's headline framing and conclusion claim that "current frequency-domain PPFR systems do not provide the level of identity privacy their pixel-level evaluation results imply" (paraphrased from abstract/conclusion), but the experiments support this primarily under the specific insider/oracle-access threat model. While the constrained-adversary experiments (Section 6) extend this somewhat, they are under-specified. The paper's own future directions section acknowledges that cryptographic/key-based hardening could block the attack, which implicitly concedes that the vulnerability is not inherent to all PPFR but rather to unkeyed, oracle-accessible transformations. The paper should more carefully scope its claims to the tested threat model rather than presenting a universal indictment of PPFR evaluation.

- **Regeneration results are partially entangled with the Arc2Face generator.** The student model is trained to map protected templates into ArcFace-family embeddings, and then Arc2Face (which is specifically designed to invert ArcFace embeddings) regenerates faces. While cross-verification with commercial APIs partially addresses this concern, it does not fully disentangle whether the high regeneration success reflects how well identity is preserved in the template or how effectively Arc2Face can synthesize identity-consistent faces from any embedding. The linkage results (which do not depend on any generator) provide the stronger and cleaner evidence. The paper would be strengthened by re-running prior reconstruction attacks (U-Net, StyleGAN from FracFace/MinusFace) under the same identity-centric (Face++) metric to show that they genuinely fail, rather than relying solely on Figure 3's visual evidence.

- **Insufficient quantitative comparison with prior reconstruction attacks under the proposed identity-centric metric.** The paper's thesis is that pixel-level attacks fail while identity-centric attacks succeed, but it only provides visual evidence (Figure 3) that prior attacks fail, without quantitative identity-level evaluation of those baselines. Directly re-evaluating the U-Net/StyleGAN attacks from FracFace and MinusFace under Face++ verification would make the paradigm comparison definitive rather than illustrative.

### Minor:

- **The constrained-attack experiment (Section 6) is under-specified.** The protocol for selecting the 30 paired samples, hyperparameter selection for the high-pass filter proxy, and threshold tuning specifics are not detailed enough for the claims to carry the same weight as the main experiments. This section reads as a promising extension rather than a rigorous result.

- **Near-ceiling results without variance reporting.** The regeneration success rates are consistently >96% across methods and datasets, making statistical significance less of a concern, but the linkage results (Table 4) show more variation and would benefit from standard deviations across runs or different train/test splits.

- **Minimal-resource experiment lacks clarity on identity separation.** Section 5.1 mentions training with 256 images but does not specify how many identities are represented or confirm that evaluation identities are fully disjoint from training. If identities overlap, the results could partly reflect memorization.

- **The TIP-IM and CanFG evaluations (Table 8) are pilot-level.** These are labeled as "pilot evaluations" by the authors, but the claim that TIP-IM shows a "structural limitation of adversarial de-identification as a privacy mechanism" goes beyond what a single-method pilot evaluation can support.

### Trivial:

- The hashing analogy for linkage attacks (Section 4.1) is rhetorically useful but technically imprecise — face identity spaces are open-set and continuous, unlike finite hash input domains.
