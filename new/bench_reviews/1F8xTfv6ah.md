Now I have sufficient information to write my final review. Let me synthesize everything.

**Calibration Summary:**

- **High (7+):** NegLabel (7.5) — novel OOD detection with VLMs; SCALE (6.25) — simple but effective post-hoc method; Neural Collapse OOD (6.75) — novel feature separation approach
- **Medium (4-6):** DC-DPM (4.25) — partitioning strategy overclaimed; HAct (3.67) — activation histogram OOD, marginal novelty
- **Low (<3):** various fundamentally flawed papers

**My paper comparison:**
- Like HAct (3.67), this is an activation-based OOD detection method with strong claimed results but concerns about novelty
- Unlike HAct, this paper has better baselines (OpenOOD protocol) and more benchmarks
- Like DC-DPM (4.25), the core contribution is misattributed — partitioning does the heavy lifting
- Unlike NegLabel (7.5), the novelty is incremental (class-conditional modeling is well-known; KAN adds ~9 points over histograms)
- Has a mathematical error in Eq. 5, though it doesn't affect the implementation
- Strong training data robustness, unlike most OOD methods
- Near-random performance on the Age benchmark (50.5 AUROC)

I'll place this at around **4.5** — the strong empirical results are real, but the core narrative is undermined by the misattribution of the dominant mechanism and the mathematical error, placing it below the acceptance threshold.</think>## Summary

This paper proposes an OOD detection method based on Kolmogorov-Arnold Networks (KANs). The core idea is to exploit KANs' local neuroplasticity: since training only modifies spline coefficients near InD data, OOD samples should activate regions closer to the untrained state. The method compares trained vs. untrained KAN activations, aggregating differences into an InD score. To address KANs' inability to model joint feature distributions (they process features independently), the paper proposes partitioning the InD data and training separate KANs per partition. The method is evaluated on 7 benchmarks (4 image, 3 tabular medical), achieving SOTA overall average AUROC across all benchmarks, with notable robustness to training set size.

## Strengths

- **Comprehensive benchmarking across domains**: The method is evaluated on 7 benchmarks from two domains (image: CIFAR-10, CIFAR-100, ImageNet-200 FS, ImageNet-1K FS; tabular: Ethnicity, Age, Synthetic), following established OpenOOD protocols. The breadth of evaluation strengthens confidence in the results. (Tables 1–5)

- **Strong empirical results**: KAN achieves the highest overall average AUROC on all 7 benchmarks, outperforming strong baselines including NAC, VIM, and KNN. On CIFAR-10, it reaches 94.12% overall AUROC (Table 1); on ImageNet-1K FS, 78.52% (Table 2). These are credible improvements over prior SOTA.

- **Remarkable robustness to training data size**: Table 6 shows KAN maintains 93.21% AUROC on CIFAR-10 with only 0.1% of training data, while KNN collapses to 8.15%. This is a genuine and practically significant advantage, especially for real-world scenarios with limited labeled data.

- **Clear and intuitive visualization**: Figure 4 effectively demonstrates that InD samples produce larger activation differences than OOD samples, and the L-shaped toy example (Figure 3) clearly illustrates the marginal distribution limitation and how partitioning addresses it.

## Weaknesses

### Fatal
None

### Major

- **Misattribution of the dominant mechanism**: The paper consistently frames local neuroplasticity as the key innovation (abstract: "leverages the unique local neuroplasticity property of KANs"; Sec. 2.2: "leverage the local plasticity of KANs for OOD detection"). However, the ablation in Table 7 reveals that without partitioning (P=1), performance is 46.08±15.58 AUROC on CIFAR-10 — essentially chance level. With P=10 (roughly per-class partitioning), performance jumps to 94.12%. This ~48-point gain comes from partitioning, which is class-conditional density modeling — a well-established technique (Mahalanobis distance-based methods model per-class Gaussians). The specific contribution of KAN's spline-based plasticity over a histogram baseline is only ~9 points (85.29 vs. 94.12). While the KAN component adds meaningful value, the paper's narrative and framing do not honestly reflect the relative contributions. This matters because it misrepresents where the novelty lies and overclaims the role of local neuroplasticity.

- **Incorrect mathematical derivation in Eq. 5**: From Eqs. 1 and 3, $\Delta_{p,q}(x_p) = |\sum_i (c_{p,q,i}^{\text{trained}} - c_{p,q,i}^{\text{untrained}}) \cdot B_i(x_p)|$. Eq. 5 states $\Delta_{p,q}(x_p) = \sum_i |c_{p,q,i}^{\text{trained}} - c_{p,q,i}^{\text{untrained}}| \cdot B_i(x_p)$. By the triangle inequality, these are not generally equal: $|\sum a_i B_i| \leq \sum |a_i| B_i$ when $B_i \geq 0$. The subsequent interpretation — that $|c^{\text{trained}} - c^{\text{untrained}}|$ "defines the locations where InD information is stored" and $B_i(x_p)$ "specifies the regions activated by the sample" — relies on this incorrect factorization. While the method itself uses Eq. 3 (not Eq. 5) for computation, the theoretical justification for *why* the method works is built on this error. The correct reading (Eq. 3) does not admit the clean decomposition the paper claims.

- **Near-random performance on the Age benchmark is unaddressed**: Table 4 shows KAN achieves 50.5±0.5 AUROC on the Age benchmark — statistically indistinguishable from random guessing. The paper highlights performance across all benchmarks but never discusses this failure case. For a method claiming SOTA across all 7 benchmarks, a result no better than a coin flip on one benchmark warrants explanation.

### Minor

- **The "post-hoc" characterization is misleading without qualification (Sec. 3.1)**: The paper states "our detector operates in a post-hoc manner." While the backbone is pre-trained, the method still requires training P separate KAN models (P=10 recommended) plus maintaining an untrained reference KAN. This is more computationally demanding than true post-hoc methods like MSP, ODIN, or VIM that require no additional training. Calling the method "post-hoc" without acknowledging this asymmetry in computational overhead is imprecise.

- **The P=1 failure mode is inadequately discussed**: The ablation shows P=1 yields 46.08±15.58 AUroc. The text briefly mentions partitioning addresses the marginal distribution limitation, but the catastrophic P=1 result and its extremely high variance deserve deeper analysis — e.g., why does the variance drop so dramatically with partitioning?

## Nice-to-Haves

- Running the histogram baseline *with* the same P=10 partitioning to isolate KAN's spline contribution from partitioning alone would clarify the specific value added by KANs over simple class-conditional binning.
- Reporting computational cost (training time, inference time, parameter counts for P=10 KAN detectors) would help practitioners assess practical tradeoffs.
- Per-class analysis of the partition-based KANs could illuminate when and why the method succeeds or fails.

## Removed Points

- **Formatting/style nitpicks**: Removed — these are parser artifacts.
- **Demand for missing appendix proofs or references**: Removed — the parser strips appendices; they exist in the original submission.
- **Claim that the method cannot be a "post-hoc" detector at all**: Softened to a minor point — the backbone is indeed pre-trained, and "post-hoc" is used in the field to include methods that add lightweight training (e.g., NAC also requires processing), though the computational overhead should be acknowledged.
- **Demand for experiments on additional architectures (ViT, Swin)**: Removed as scope creep — the paper uses established OpenOOD benchmark protocols.
- **Demand that the paper discuss the Age failure case in more detail**: Kept as a major weakness (near-random performance must be acknowledged) but the demand for "deeper analysis of why" moved to Nice-to-Have.
- **Strength claim about "mathematically grounded mechanism from Eq. 5"**: Removed from strengths since Eq. 5 is mathematically incorrect.

## Novel Insights

The most important insight from the review analysis is that the paper's own ablation inadvertently reveals that KAN's local neuroplasticity is a secondary rather than primary mechanism for OOD detection. The dominant performance gain comes from class-conditional partitioning, which is a well-known technique in OOD detection (e.g., Mahalanobis distance methods). The specific contribution of KAN's spline-based representation over a simpler histogram is moderate (~9 AUROC points), which is still meaningful but far from the narrative's emphasis. This gap between the paper's framing and its evidence is the central issue.

## Suggestions

- **Reframe the contribution honestly**: Position the method as "class-conditional density estimation enhanced by KAN's spline-based local plasticity," acknowledging that partitioning addresses the joint distribution limitation and is the dominant factor, while KANs provide the continuous registration that improves upon discrete histograms.
- **Correct Eq. 5**: Either fix the derivation by using Eq. 3 directly, or add the necessary conditions (e.g., when coefficient differences are non-negative for all i) under which the equality would hold, and discuss whether those conditions hold in practice.
- **Discuss the Age benchmark failure**: A 50.5 AUROC result warrants acknowledgment and analysis. If the method fundamentally struggles with certain data types, this should be transparently reported.
- **Add the histogram+partitioning baseline**: This single ablation would cleanly isolate KAN's contribution and strengthen (not weaken) the paper if the ~9 points are indeed meaningful.

## Calibration

- **NegLabel (avg 7.5)** — Novel OOD detection with VLMs using negative labels. Strong novelty, strong results. This paper is below NegLabel due to misattribution concerns and the mathematical error.
- **SCALE (avg 6.25)** — Simple but effective post-hoc OOD method, considered incremental. This paper has more novelty but more serious methodological issues.
- **HAct (avg ~3.67)** — Activation histogram OOD detection with overclaimed results and limited novelty. This paper has better benchmarks and a more developed method, but faces similar concerns about the core mechanism vs. framing.
- **DC-DPM (avg 4.25)** — Partitioning strategy for diffusion models where partitioning is the real contribution. Directly analogous: this paper also has partitioning as the dominant mechanism with overclaimed novelty.
- **Papers with ablation showing simpler mechanisms dominate (avg 4.0-4.5)** — e.g., ReAct-style prompting (avg 4.0), where ablation reveals the claimed mechanism isn't the real driver.

The paper's strong empirical results and training data robustness bring it above clearly flawed papers. However, the misattribution of the dominant mechanism, the mathematical error, and the unaddressed failure case place it below the acceptance threshold for a top venue. Compared to HAct (3.67) and DC-DPM (4.25), this paper has stronger experiments but similarly overclaimed novelty. Compared to SCALE (6.25), this paper has a more serious methodological concern.

## Score and Decision

**Score: 4.5** — The paper presents a method with genuinely strong empirical results and an interesting application of KANs, but the core narrative is undermined by the evidence: partitioning (not local neuroplasticity) is the dominant performance driver, the theoretical justification contains a mathematical error, and the near-random performance on the Age benchmark is unaddressed. A reformed paper that honestly attributes the contribution hierarchy and fixes the math error could be significantly stronger.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>