## Summary
This paper proposes a privacy-preserving inference framework using input domain shifting, where users encode their data before sending it to cloud-based deep learning services. The authors present two approaches: a model-specific method requiring whitebox access (out-of-place shifting) and a model-agnostic method requiring only blackbox access (in-place shifting using GANs/DDPMs). Evaluation spans multiple datasets (MNIST to ImageNet) and architectures.

## Strengths
- **Dual-protocol framework with clear threat model differentiation**: The paper clearly distinguishes between whitebox and blackbox access scenarios (Section 3, Section 4), offering specific mechanisms for each that do not require server-side modifications. This contrasts with cryptography-based schemes described in Section 2 that require provider cooperation.
- **Comprehensive empirical evaluation across datasets and architectures**: Table 2 and Table 4 present results across MNIST, CIFAR-10, Tiny-ImageNet, and ImageNet with multiple oracle model architectures (MLP, CNN, Vision Transformer, Swin Transformer), demonstrating the method's behavior across complexity levels.
- **Significant latency improvements for whitebox scenario**: Section 5.3 reports inference times of 0.4-13ms for the model-specific approach, substantially lower than the 9.74-472 seconds reported for HE/MPC baselines (Liu et al., 2017; Juvekar et al., 2018) when whitebox access is available.

## Weaknesses

### Fatal
// None - the paper makes real contributions, though with significant limitations

### Major

- **SSIM as the primary privacy metric without reconstruction attack evaluation**: The paper relies on SSIM between original and obfuscated images (Table 2, Table 4) as the primary evidence of privacy protection. However, SSIM measures perceptual similarity, not information leakage. An image can have SSIM≈0 with the original (e.g., a negative image, encrypted data, or scrambled permutation) yet remain trivially invertible. The paper provides no evaluation against reconstruction attacks (training an inverter R(x^ob)→x) or information-theoretic metrics. This is a fundamental methodological gap for a privacy paper—comparable to papers that scored 3-4 in calibration (e.g., NmWf0gLufZ: 3.50, lddpNkrgXV: 4.00) where reviewers criticized weak or missing reconstruction attack evaluation. Without adversarial evaluation, the claim that "original inputs remain private" (Abstract) is unsupported.

- **Threat model mismatch between motivation and high-performance method**: The Introduction motivates the work via "cloud-based deep learning services" where users upload data to third-party providers—typically offering only blackbox API access. However, the best-performing method (Model-specific Transform Training, Section 4.2) explicitly requires **whitebox access** (parameters, gradients) to train the encoder. The applicable blackbox method (Model-agnostic) suffers from severe utility degradation (~10-15% accuracy drop on CIFAR-10/ImageNet, Table 4) and high latency (see next point). The paper does not adequately reconcile this contradiction between the stated problem setting and the method's requirements.

- **Prohibitive latency for the applicable blackbox scenario**: For the Model-agnostic approach—the only one viable for the stated cloud API scenario—the inference overhead is approximately **4.1 seconds per image** (Table 4, Section 5.3). The Introduction argues that Homomorphic Encryption is impractical due to latency, yet this method introduces latency comparable to or exceeding some optimized HE/MPC schemes. The paper frames 4.1s as a success by comparing to 5s (Nie et al., 2024), but ignores that this is orders of magnitude slower than the Model-specific method (0.4-13ms) and impractical for interactive or high-throughput services. This undermines the core practicality claim.

### Minor

- **Notation errors in mathematical formulation**: Equation (1) (line 106) contains `SSIM^2[f(x), EN(x)]` where f(x) is the model output (logits/labels) while EN(x) is an image—SSIM is defined between images, not between labels and images. The text states the metric should be between "real input x and obfuscated input x^ob", implying the equation should read `SSIM^2[x, EN(x)]`. Additionally, Section 5.2 (line 254) states "we calculated the SSIM between the true class of the input images and the oracle model's classification of the encoded images"—SSIM cannot be calculated between class labels. These errors suggest insufficient rigor in the mathematical formulation.

- **Overstated accuracy preservation claims**: The Abstract claims "minimal impact on classification performance," but Table 4 shows accuracy drops from 88.55% to 75.10% (ImageNet) and 88.91% to 80.30% (CIFAR-10) for the model-agnostic approach. A ~10-15% absolute drop is not "minimal" and contradicts the abstract's framing.

### Trivial
// None beyond the notation errors already captured above

## Nice-to-Haves
- **Reconstruction attack evaluation**: Adding experiments where an adversary trains an inverter network to recover x from x^ob would substantially strengthen the privacy claims, even if the results show the method is vulnerable (honest reporting would be valuable).
- **Encoder security analysis**: Discussing key management, encoder rotation, or the risk of encoder extraction/inversion would address practical deployment concerns for the model-specific approach where privacy relies on encoder secrecy.
- **Failure case visualization**: Showing examples where the Model-agnostic encoder fails to preserve utility (the ~15% accuracy drop cases) would help readers understand the method's limitations.

## Removed Points
These points are flagged to be removed, treat them with caution:
- "Security through obscurity" criticism about encoder secrecy: While valid as a concern, the paper does frame the encoder as user-side and secret. This is more of a design choice than a flaw—the paper could discuss it more, but it's not a fundamental error. Moved to nice-to-have.
- Criticism about Figure 1 not proving privacy: Figure 1 is illustrative of the out-of-place shifting concept, not a privacy proof. This is a minor presentation issue, not a substantive weakness.
- Generic requests for membership/attribute inference tests: These would strengthen the paper but are not standard for this type of input obfuscation work. Moved to nice-to-have.
- Requests for information-theoretic metrics (mutual information): While valuable, these are not standard in empirical privacy papers. The reconstruction attack gap is more critical.

## Novel Insights
The calibration reveals a consistent pattern: privacy papers relying on empirical obfuscation metrics (SSIM, perceptual similarity) without adversarial reconstruction attack evaluation typically score 3-4, regardless of empirical breadth. Papers scoring 6+ either provide formal privacy guarantees or include comprehensive attack evaluations (e.g., 1GMw3IwEHW, BnEG8pn3pK). This paper's empirical breadth is comparable to high-scoring papers, but the privacy evaluation gap places it firmly in the 3-4 range. The threat model mismatch (whitebox method for blackbox-motivated problem) is a structural issue that compounds the evaluation weakness.

## Suggestions
1. **Add reconstruction attack evaluation**: Train inverter networks to attempt recovery of x from x^ob for both methods. Report success rates and visualize recovered images. This is essential for any privacy paper making obfuscation claims.
2. **Reframe contributions honestly**: The model-specific (whitebox) method shows genuine latency improvements over HE/MPC but requires unrealistic access for cloud APIs. The model-agnostic (blackbox) method is applicable but has significant accuracy/latency trade-offs. Acknowledge these limitations explicitly rather than claiming "minimal impact."
3. **Fix notation errors**: Correct Equation (1) to reference x and EN(x) instead of f(x) and EN(x). Revise the Section 5.2 text to accurately describe what SSIM was calculated between.
4. **Discuss practical deployment**: Address encoder management (how users obtain/share encoders), the risk of encoder leakage, and whether the 4.1s latency is acceptable for target applications.

## Score and Decision

**Calibration anchors consulted:**
- **Low-scoring (≤4)**: NmWf0gLufZ (3.50) - limited reconstruction evaluation; lddpNkrgXV (4.00) - critiques weak privacy evaluation standards; mXlAdNtLN5 (4.00) - empirical privacy without formal guarantees; iVe8A0yUxu (3.00) - weak evaluation completeness.
- **Medium-scoring (~5)**: 4XMPZGOQ5d (5.33) - novel framing but evaluation gaps; mTsWEVhcZM (5.00) - empirical evaluation only.
- **High-scoring (≥6)**: 1GMw3IwEHW (6.00) - strong attack/defense evaluation; BnEG8pn3pK (6.00) - strong empirical evidence with causal analysis.

**Reasoning**: This paper has empirical breadth comparable to high-scoring papers but lacks the adversarial privacy evaluation that distinguishes 6+ papers from 3-4 papers. The SSIM-only privacy metric without reconstruction attack evaluation is the same weakness that caused papers like NmWf0gLufZ (3.50) and lddpNkrgXV (4.00) to be rejected. The threat model mismatch and 4.1s latency for the applicable scenario are additional structural issues. The paper makes real contributions (dual-protocol framework, comprehensive empirical results, whitebox latency improvements) but the core privacy claim is unsupported. This places it at the boundary between 3 and 4—leaning toward 4 due to the genuine empirical contributions, but the privacy evaluation gap is severe for a privacy paper.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>