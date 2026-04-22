# DepthSense+DP: Adaptive Learning for Robust and Differential Private Silent Speech Recognition

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
DepthSense+DP is a privacy-preserving framework for silent speech recognition from dynamic 3D depth point clouds. It integrates calibrated input perturbation, feature-level differential privacy, and geometry-preserving alignment within a lightweight P4DConv front end and Conformer encoder to ensure robust cross-user and cross-device generalization under formal DP guarantees. A dual-stage DP pipeline injects noise at point and feature levels while maintaining articulatory geometry, aided by an adaptive DAD gate for improved privacy–utility trade-off. The co-designed architecture enables efficient on-device inference. Experiments on a large multi-location corpus show near-baseline accuracy with significant reductions in membership, inversion, and attribute-inference risks, supported by full DP accounting and attack evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes DepthSense+DP, an adaptive and privacy-preserving framework for silent speech recognition (SSR) using 3D depth point clouds. It integrates differential privacy noise injection, DP-aware T-Net alignment, 4D spatio-temporal convolution, and a Conformer encoder to achieve robust, cross-device, and user-independent recognition. Experiments on multiple device setups show notable accuracy gains (WER 17%, CER 11%) and strong resistance to membership and inversion attacks with minimal performance loss. Overall, the paper provides a technically solid and comprehensive redesign for privacy-aware SSR with clear novelty and strong empirical results.

### Strengths
+. The paper clearly identifies that silent speech recognition (SSR) can still leak users’ physiological information (e.g., lip geometry and facial motion trajectories), highlighting the real-world necessity of privacy protection even in non-audio modalities.

+. Introducing differential privacy into the SSR domain is novel and represents a meaningful extension of privacy-preserving learning to a new modality.

+. The system design is comprehensive, covering data acquisition, point-cloud preprocessing, feature extraction, and decoding, demonstrating strong system integration and engineering maturity.

### Weaknesses
-. The paper applies differential privacy primarily at the feature level, but it does not clearly justify why noise injection is limited to the point-feature stage rather than being extended to deeper encoding or output layers. It also does not discuss why alternative privacy-preserving training methods such as DP-SGD were not adopted for end-to-end privacy guarantees.

-. The privacy–utility trade-off analysis remains largely empirical and lacks theoretical grounding. The discussion is somewhat fragmented, and the conclusions would benefit from a more principled quantitative or analytical interpretation.

-. The paper provides insufficient discussion of related work on differential privacy, particularly studies combining DP with 3D point cloud processing or geometric data. A more comprehensive review would help contextualize the novelty and clarify how this work differs from prior DP applications in spatial or multimodal domains.

-. In the methodology section, Figure 2 illustrates the overall technical pipeline, but none of the modules in the diagram explicitly represent or integrate the differential privacy mechanism. This makes the DP component appear somewhat detached from the main framework and may cause readers to question how DP is systematically embedded into the model design.

-. Although the paper’s title and claims emphasize differential privacy, the experiments and results focus primarily on the baseline SSR system’s performance, with limited empirical validation or quantitative evaluation of DP effectiveness. This gives the impression that the work centers more on the system design than on privacy mechanisms.

### Questions
Q1: What level of privacy protection does the proposed scheme achieve, and how is it theoretically analyzed?

Q2: Why is depth data converted into point clouds in silent speech recognition, and what are the advantages?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper defines four critical constraints for cross-device SSR and introduces DepthSense+DP, the first solution to jointly achieve real-time performance, robustness, and DP-based privacy for 3D depth point cloud-driven SSR.

### Strengths
- Addresses the unique challenge of DP for 3D point clouds by proposing controllable noise injection that anonymizes biometric data without degrading articulatory geometry
- Demonstrates significant reductions in CER and WER across diverse devices, users, and environments, establishing DepthSense+DP as a universal foundation for next-generation SSR

### Weaknesses
- The dataset relies heavily on English utterances and includes only 20 native English speakers. Performance degrades for users with strong accents (e.g., Participant P9), limiting generalization to non-English languages or global user groups.
- While the study evaluates membership inference and model inversion attacks, it does not address emerging threats, leaving potential privacy gaps untested.
- Synthetic depth point clouds are generated via simple motion scaling and noise injection—these do not capture complex real-world variations, restricting the model’s robustness to diverse user conditions.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes DepthSense+DP, a privacy-aware silent speech recognition (SSR) framework using depth sensing and differential privacy. The system transforms depth images into 3D point clouds, learns articulatory features via adaptive spatio-temporal sampling and Conformer-based encoding, and injects calibrated Gaussian noise to satisfy differential privacy (with optimal $\epsilon = 1.5$ for the best privacy-utility trade-off, and $\delta$ fixed at $10^{-5}$ for all experiments). It claims to achieve real-time, cross-device, and cross-user generalization, outperforming baseline systems proposed in last 5 years, including RGB and mmWave SSR, on WER and CER while resisting membership-inference and inversion attacks.

Key contributions include:
1. Application of differential privacy to dynamic 3D point clouds.
2. DP-aware adaptations of T-Net and Conformer modules.
3. A new multi-sensor SSR dataset with device-placement diversity.
4. Analysis of privacy-utility trade-offs and empirical robustness.

### Strengths
- Solid engineering effort combining robustness, privacy, and real-time constraints
- Careful empirical design: cross-device, cross-user, ablation, and privacy-utility trade-off analysis
- Quantitative evidence of low privacy overhead ($\Delta$WER $\approx 1$%)
- Demonstrated feasibility of DP noise in geometric and feature space for SSR
- Valuable dataset and reproducible methodological detail

### Weaknesses
- Although this work introduces a new system, it over-claims novelty. It adapts existing 3D and DP ideas rather than introducing a new learning principle
- Lacks formal privacy proofs, composition reasoning, and noise calibration analysis across stages
- Tested only on English scripted phrases; unclear how it generalizes to spontaneous, multilingual, or emotional speech
- No error bars, confidence intervals, or hypothesis testing; results could be dataset-specific
- The architecture (T-Net + P4DConv + Conformer + Bi-GRU) may be over-engineered relative to performance gains
- Discussion of fairness implications is minimal, which is critical given the biometric domain

### Questions
1. How is the overall privacy budget ($\epsilon$, $\delta$) computed when applying DP noise at both the point-cloud and feature levels ?
2.  Could you provide statistical confidence intervals or standard deviations for WER/CER to assess robustness ?
3. What is the (expected) computational latency on actual wearable hardware, not GPUs ?
4. How does the system handle multilingual data or unseen phonetic inventories ?
5. Could adaptive or learned DP noise (e.g., per-user calibration) improve the privacy–utility balance ?
6. How does this approach compare to non-DP anonymization (e.g., adversarial suppression) in terms of privacy leakage ?

### Soundness
3

### Presentation
3

### Contribution
3
