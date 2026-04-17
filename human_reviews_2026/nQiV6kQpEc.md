# Attacking and Securing Masking Scheme for TEE-Based Model Protection

- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 4, 0

## Abstract
Deep learning (DL) models are being increasingly adopted across a wide range of applications.
Many inference models are deployed on edge devices to enable efficient and low-latency computation.
However, such deployment exposes security risks, including the potential leakage of model parameters.
To address these security risks, several researchers have proposed protection schemes for deployed models based on Trusted Execution Environments (TEEs).

In this paper, we analyze a common weakness of existing TEE-based protection schemes, namely the insecurity of the masking mechanism.
Existing masking schemes not only provide limited security guarantees but also incur high computational and storage complexity.
Motivated by these inherent weaknesses, we develop a targeted differential attack that can accurately recover the parameters of linear layers in ReLU-based neural networks.
Furthermore, we propose an improved masking scheme that achieves higher security and efficiency by generating substantially more mask combinations under the same computational cost, thereby considerably strengthening TEE-based model protection.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper significantly alters the margins, violating the official ICLR 2026 formatting guidelines.

### Strengths
N/A

### Weaknesses
This paper significantly alters the margins, violating the official ICLR 2026 formatting guidelines.

### Questions
N/A

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the weakness of existing TEE-based model masking schemes, presents a differential attack that can recover model parameters with high accuracy, and proposes a new combined masking scheme that generates more secure mask variants with the same cost.

### Strengths
1) Attack design is concrete and reproducible, showing strong analytical reasoning using collision triplets.
2) Well-written manuscript
3) Empirical validation (98% recovery) is convincing and well presented.

### Weaknesses
1) The threat model is too simplified as it assumes a fully trusted TEE and ignores side-channel or partial leakage.
2) The attack success is limited to shallow networks and deeper architectures only work after retraining.
3) Comparisons to prior TEE defenses (e.g., NNSplitter) are minimal
4) No runtime overhead analysis.

### Questions
1) The paper provides a valuable contribution by exposing a clear vulnerability in precomputed masking used by TEE-protected inference systems.
2) However, the scope of evaluation is narrow, by focusing only on one small CNN. There’s no test on real frameworks like SGX/TrustZone nor a performance comparison for the new masking scheme. 
3) I also suggest adding overhead analysis of the proposed method too, and demonstrate its effectiveness in presence of recent TEE defenses.
4) The paper is well written and logically structured. Minor typos (“adersary”, “unkonwn”) should be fixed.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper seriously violates the ICLR formatting rules by using reduced margins, which allows the authors to include more content than permitted. This creates an unfair advantage compared to other submissions. I recommend a desk rejection.

### Strengths
NA

### Weaknesses
NA

### Questions
NA

### Soundness
1

### Presentation
1

### Contribution
1
