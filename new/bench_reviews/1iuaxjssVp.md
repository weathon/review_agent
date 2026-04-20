## Summary
InvMSAFold introduces a two-stage framework for protein inverse folding that generates Potts model parameters from protein structures in a single neural network pass, then uses these parameters for fast, diverse sequence sampling on CPU. The key methodological insight is training on MSAs rather than single native sequences to capture the one-to-many nature of structure-sequence relationships, combined with a low-rank coupling factorization that enables $\mathcal{O}(L)$ pseudo-likelihood computation. The paper demonstrates substantially better covariance reconstruction of natural MSAs and orders-of-magnitude faster sampling compared to ESM-IF1, with results extending to broader distributions in predicted protein properties.

## Strengths

- **Mathematically rigorous tractable inference**: The derivation in Section 2.2.1 (Eqs 1-7) proving that low-rank Potts couplings enable $\mathcal{O}(L)$ memory and computation for pseudo-likelihood is a clear, reusable technical contribution. The explicit reformulations (e.g., Eq 4, Eq 6) avoiding materialization of the full $L^2 q^2$ coupling tensor directly address the quadratic bottleneck of traditional DCA models.

- **Strong covariance reconstruction of natural MSAs**: Figure 5 and the accompanying table show InvMSAFold-AR achieving median Pearson correlations of 0.53 (inter-cluster) and 0.53 (intra-cluster) vs. 0.31 for ESM-IF1, demonstrating superior capture of evolutionary patterns. This is a well-established metric in the DCA/statistical physics community for measuring how well a generative model captures the sequence landscape.

- **Architecturally principled speed advantage**: The two-stage design (one neural network forward pass → lightweight Potts parameters → fast CPU sampling) is not just an implementation detail but a fundamental architectural difference. Figure 4 shows that for many samples, InvMSAFold amortizes the single forward pass cost, enabling virtually constant sampling time via CPU batching, while ESM-IF1 scales linearly with one forward pass per sample.

- **MSA training for diversity**: Training on multiple sequence alignment subsamples (Sec 2.4) rather than single native sequences is a principled design choice that aligns with the biological reality of structural tolerance. The hardness ordering (inter-cluster > intra-cluster > MSA, Sec 4.1) confirms generalization across homology levels.

## Weaknesses

### Fatal
None.

### Major
- **Structural validation lacks pLDDT confidence filtering**: Section 4.5 evaluates structural fidelity using AlphaFold2-predicted RMSD vs. Hamming distance (Fig 8), but does not report pLDDT scores or apply confidence-based filtering. In protein design, AlphaFold2 is known to produce artificially low RMSD alignments for sequences with detectable homology to its training set, even when those sequences are physically unstable. Without confidence metrics, the RMSD results cannot fully distinguish true *de novo* foldability from low-confidence alignments. This weakens the claim that generated sequences "maintain structural fidelity at high normalized Hamming distances."

- **Hardware-confounded speed comparison with omitted generation overhead**: Section 4.2 compares InvMSAFold-AR running on a single CPU core (i9-13905H) against ESM-IF1 on a consumer laptop GPU (RTX 4060, 8GB). While the architectural advantage (one forward pass vs. many) is real, the comparison conflates CPU vs. GPU performance and omits the initial neural network forward pass time required to generate Potts parameters. In virtual screening scenarios requiring only hundreds or thousands of samples per backbone (rather than millions), the fixed NN forward pass cost could significantly reduce the practical speedup. A hardware-matched comparison (both on GPU) and an amortized timing table showing end-to-end latency including the generation step would establish the computational advantage more convincingly.

### Minor
- **2D PCA KL-divergence as a diversity metric is limited**: Table 1 and Figure 6 report KL divergence between distributions projected onto the first two PCA components of one-hot encoded sequences. While this provides intuitive visualization, projecting ~$20L$-dimensional categorical space onto two continuous components and estimating densities via Gaussian KDE (bandwidth 1.0) is a coarse approximation that cannot fully capture high-dimensional sequence space structure or mode coverage. The covariance reconstruction results (Figure 5, Pearson correlations) partially compensate for this, but the over-reliance on 2D PCA for the diversity claims (especially "cover multiple distinct modes") is not fully substantiated.

- **Property prediction experiments are somewhat circular**: Section 4.6 demonstrates that InvMSAFold generates sequences with wider distributions in predicted thermostability (Thermoprot) and solubility (Protein-Sol) compared to ESM-IF1 (Figure 9). However, as the critic notes, this is partially tautological: by construction, sampling a wider sequence distribution yields wider outputs from sequence-based predictors. The paper does not demonstrate that any of the generated sequences are actually more stable, soluble, or functionally improved—only that the predicted distributions are broader. This does not invalidate the diversity claim but weakens the assertion that this "translates into greater variability in biochemical properties" relevant for virtual screening.

### Trivial
- **ProteinMPNN relegated to appendix for key comparisons**: While ProteinMPNN is mentioned as the field standard for inverse folding, its performance metrics (covariance reconstruction, sequence diversity) are deferred to the appendix (A.1, B.2) rather than presented alongside ESM-IF1 in the main narrative. Given its prominence in the inverse folding community, ProteinMPNN should have comparable visibility to ESM-IF1 in the main results.

## Nice-to-Haves
- **Ablation on low-rank dimension $K$**: Analyzing the sensitivity of covariance fidelity and structural preservation to the rank $K$ would validate that the $\mathcal{O}(L)$ compression does not discard critical long-range evolutionary constraints, and provide practical guidance for choosing $K$.
- **Guided sampling demonstration**: Implementing a loop that explicitly maximizes property predictors (e.g., thermophilicity) under the Potts energy would demonstrate the claimed virtual screening workflow rather than passive distribution analysis.
- **Solvent accessibility or secondary structure conditioning**: Incorporating additional structural conditioning into the parameter decoder would substantiate the method's utility in tailored protein design beyond the current fixed-backbone formulation.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic: "AlphaFold2 RMSD without pLDDT is scientifically meaningless / structurally invalidates central claim"**: While the lack of pLDDT is a valid concern (moved to Major weakness), calling it "scientifically meaningless" and saying it "structurally invalidates" the central claim is an overstatement. The paper compares relative performance between methods using the same evaluation protocol; even without pLDDT, the trend showing InvMSAFold maintaining lower RMSD at higher Hamming distances vs. ESM-IF1 is informative. The paper also follows established practice in the literature (citing Li et al., 2023; Dauparas et al., 2022 for the same protocol).

- **Harsh Critic: "KL divergence on 2D PCA projections is statistically invalid / fundamentally cannot support headline contribution"**: While the 2D PCA metric is limited (moved to Minor), the paper's headline contribution is supported primarily by the covariance reconstruction results (Figure 5, Pearson correlation), which is a well-established metric. Characterizing this as making the metric "fundamentally incapable" of supporting the paper's claims ignores the complementary evidence.

- **Harsh Critic: "Speed comparison is biased and cannot establish claimed computational advantage"**: The architectural difference (one forward pass vs. many) is a genuine speedup mechanism, not a measurement artifact. The direction of the speedup is clear even if the hardware comparison is not ideal (moved to Major for the confounding issue, not for invalidating the claim entirely).

- **Harsh Critic: "MSA cross-contamination between train and test splits artificially inflates covariance recovery"**: The paper uses CATH cluster-based splits (10% of clusters to inter-cluster test, excluded from training). While explicit cross-contamination rates would strengthen the paper, the cluster-based split methodology is standard and should largely address this concern. No evidence is presented that the specific MSA construction procedure causes leakage from test to training clusters.

- **Harsh Critic: "Protein property prediction is circular"**: Moved to Minor. The paper does acknowledge that it is evaluating predicted property distributions, not experimentally validated improvements. While the claim is somewhat overstated, the result is consistent with the diversity advantage demonstrated elsewhere.

## Novel Insights
The paper's contribution lies in bridging the statistical physics of Potts models (direct coupling analysis, low-rank factorizations) with modern deep learning-based inverse folding. The key insight—that a neural network can be trained to output the parameters of a lightweight generative model (rather than directly generating sequences), enabling amortized generation with diverse sampling strategies (MCMC, autoregressive, constrained decoding)—is elegant and potentially generalizable to other structure-conditioned sequence generation problems. The MSA-training approach for inverse folding, while conceptually simple, effectively transforms the one-to-one mapping problem into the biologically realistic one-to-many problem without requiring architectural changes to the underlying encoder.

## Suggestions
1. **Add pLDDT-filtered analysis**: Report the distribution of AlphaFold2 pLDDT scores for generated sequences at various Hamming distances, and show that the observed RMSD trends hold when filtering for pLDDT > 70 or 80.
2. **Provide amortized timing table**: Include a table showing end-to-end latency (NN forward pass + N samples) for varying N (10, 100, 1000, 10000) on matched hardware to quantify the crossover point where InvMSAFold's advantage becomes significant.
3. **Elevate ProteinMPNN comparisons**: Include ProteinMPNN results for covariance reconstruction (Figure 5) and sequence diversity (Table 1) in the main text, given its status as the field standard for inverse folding.

## Score and Decision
Compared against calibration anchors:
- **Above rejected papers** like Ipe4fMCBXk (scores 3,3,5,6,8,8) which had missing critical baselines (no designability comparison with FrameFlow/FrameDiff) and mathematical notation errors. InvMSAFold provides a complete evaluation pipeline with no such fundamental gaps.
- **Similar to borderline accepted papers** like BxcEqwl9es (scores 6,5,5,5) and DP4NkPZOpD (scores 3,3,6,8), which have solid ideas but evaluation concerns. This paper is stronger than these in mathematical rigor and covariance results.
- **Below accepted spotlight papers** like kJFIH23hXb (scores 8,8,8,8 for FoldFlow) and lvw3UgeVxS (scores 6,8,8,8 for gRNAde), which include wet-lab validation or comprehensive multi-benchmark evaluations. InvMSAFold lacks experimental validation of generated sequences.

The paper makes a genuine methodological contribution with strong theoretical grounding and empirical evidence for its core claims. The weaknesses (pLDDT reporting, hardware-matched comparison, 2D PCA metric) are real but addressable and do not invalidate the central contribution.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>