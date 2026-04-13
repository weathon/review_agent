=== CALIBRATION EXAMPLE 21 ===

# Final Consolidated Review
## Summary
SecureGS proposes a 3D Gaussian Splatting steganography framework that addresses a specific security flaw in prior work (GS-Hider): the geometric structure of hidden content is exposed in the public point cloud. The method leverages Scaffold-GS's anchor-point architecture with hybrid decoupled encryption (storing hidden Gaussian attributes via private MLPs) and region-aware density optimization (RDO) to conceal hidden object geometry while maintaining rendering fidelity.

## Strengths
- **Identifies and addresses a concrete security vulnerability:** The paper convincingly demonstrates that GS-Hider reveals hidden object geometry in the visualized point cloud (Fig. 1b, Fig. 7). The proposed RDO strategy adaptively promotes anchor growth in hidden regions so original-scene anchors obscure the hidden structure—a technically creative solution to a real problem.
- **Strong empirical efficiency gains:** SecureGS reduces storage by ~200MB versus GS-Hider (Table 1: 267.39MB vs 468.63MB) while achieving ~3x faster rendering (131.71 FPS vs 48.28 FPS). PSNR for both original and hidden scenes improves across all tested scenes.
- **Versatile steganographic payload support:** The framework extends beyond 3D object hiding to bit hiding (100% accuracy, Table 3) and single-image hiding (Table 5), demonstrating breadth of application.

## Weaknesses
- **Security claims are overstated relative to what is actually provided:** The framework's "security" rests entirely on keeping private MLP weights $\{\mathcal{F}_o^\dagger, \mathcal{F}_c^\dagger, \mathcal{F}_\alpha^\dagger, \mathcal{F}_q^\dagger, \mathcal{F}_s^\dagger\}$ secret. This is security through obscurity, not cryptographic protection. No analysis is provided on the difficulty of weight extraction, no key-based authentication mechanism exists, and no cryptographic binding anchors the public features to the private MLPs. For a paper titled "SecureGS" with "security" as a central contribution, this gap is substantial. A clear threat model defining adversary capabilities is absent.
- **Security evaluation is purely visual:** Section 4.3 relies entirely on Fig. 7's qualitative visualization. No quantitative steganalysis, statistical tests on point density distributions, or adversarial recovery experiments are conducted. Visual inspection is necessary but insufficient for security claims.
- **RDO storage overhead is substantial but downplayed:** Table 4 shows the full model (290.54MB) is 72% larger than without RDO (168.75MB). The paper describes this as "small impact on storage efficiency" (Section 3.4), which is misleading—nearly doubling storage is significant.
- **RDO security depends on scene density and lacks failure case analysis:** If a hidden object is placed in a naturally sparse region (open sky, flat wall), original anchors will not grow there even with a reduced threshold, since rendering gradients are also low. The paper only shows favorable configurations (Fig. 7).
- **View-dependent hidden positions lack justification:** Eq. 4 makes hidden Gaussian positions view-dependent via $\vec{d}_{vc}$ and $\delta_{vc}$ inputs to the offset predictor. Standard Gaussian positions are view-independent (only color/SH coefficients vary with view). No justification is provided for this unconventional design choice, which could introduce multi-view inconsistencies.
- **Bit hiding comparison conflates base representation with steganographic contribution:** Table 3 compares SecureGS (Scaffold-GS based) against CopyRNeRF and NeRFProtector (NeRF-based). The 100% bit accuracy and higher PSNR largely reflect Scaffold-GS's superior reconstruction over NeRF, not the steganographic method's superiority. The comparison should acknowledge this conflation.
- **Undefined notation in Section 3.5:** The MLP list includes $\mathcal{F}_s^3$ twice and $\mathcal{F}_\phi, \mathcal{F}_\phi^3$ which were never defined, suggesting an uncleaned copy from Scaffold-GS notation. This obscures the training procedure.
- **Factual inconsistencies:** (1) Section 4.2 references "Tab. 5" for 3D object hiding but the relevant table is Table 1. (2) Section 4.4 states "Even at a larger pruning rate of 25%" while Table 2 only shows up to 20%. (3) The GPU "RTX 4090Ti" does not exist (NVIDIA offers RTX 4090 and 4090D). (4) Throughout figures and tables, the prior method is called "GS-Header" instead of its correct name "GS-Hider."
- **Missing specifications:** DBSCAN hyperparameters (epsilon, min_samples) used in RDO are not reported, making reproducibility incomplete. Training time is not compared despite additional MLPs, asynchronous gradient accumulation, and DBSCAN calls adding overhead.

## Nice-to-Haves
- Quantitative steganalysis (e.g., trained classifier, statistical density tests) to complement visual security claims
- Payload capacity curve showing fidelity/security degradation as hidden content size increases
- Robustness evaluation against standard 3D operations (quantization, compression with Draco/Gzip) beyond random pruning
- Extraction latency measurement for authorized decoding

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Shallow engagement with cryptographic literature:** While valid, this extends beyond the paper's stated scope of 3DGS steganography. The paper need not engage cryptographic literature deeply if it reframes claims appropriately.
- **Dependency on Scaffold-GS limits applicability:** This is a design choice, not a flaw. The method explicitly builds on Scaffold-GS architecture.
- **Code availability concern:** Common for anonymous submissions; reproducibility details in Section 4.1 are sufficient for now.

## Novel Insights
The insight that explicit 3D representations (3DGS point clouds) pose unique steganographic security risks beyond those of implicit representations (NeRF)—specifically, geometric structure leakage in the public file itself—is genuinely valuable. The RDO strategy's use of gradient-based detection of hidden regions to drive adaptive anchor densification is clever: it turns the training dynamics into a security mechanism rather than treating security as purely post-hoc. However, the tension between RDO's storage overhead and its security benefit deserves more explicit acknowledgment, as does the dependency on scene density for security—sparse scenes may be fundamentally harder to secure with this approach.

## Suggestions
- Reframe security claims to accurately reflect what is provided: obscurity-based protection via private MLP weights, not cryptographic security. Explicitly state the assumed threat model (adversary has public anchor file and public MLPs but not private MLPs).
- Add quantitative security evaluation: at minimum, compare point density statistics between SecureGS and clean Scaffold-GS models, or train a simple classifier to detect hidden regions.
- Justify or ablate the view-dependent position prediction in Eq. 4.
- Correct the factual errors (table cross-references, pruning rate inconsistency, GPU name, method naming).
- Report DBSCAN parameters and training time.
- Discuss failure cases: what happens when hidden objects occupy sparse scene regions, or when hidden content is large relative to the container scene?

# Actual Human Scores
Individual reviewer scores: [5.0, 6.0, 8.0, 8.0, 3.0]
Average score: 6.0
Binary outcome: Accept
