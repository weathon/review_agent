=== CALIBRATION EXAMPLE 54 ===

# Harsh Critic Review
I now have sufficient material from the paper to write a thorough review. Let me compile my analysis.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "We Can Hide More Bits: The Unused Watermarking Capacity in Theory and in Practice" is direct, descriptive, and well-matched to the paper's content. The abstract cleanly summarizes three distinct contributions: (i) geometric capacity bounds under PSNR and robustness constraints, (ii) empirical evidence that state-of-the-art models fall far short even in trivially simplified setups, and (iii) the Chunky Seal model as existence proof that higher-capacity is achievable. However, the abstract does not mention that the robustness bounds (Bounds 10–12) are **heuristic** rather than formal upper bounds — a distinction the main text handles with some care but that is elided here. Calling something a "bound" without this qualifier in the abstract may mislead readers about the paper's theoretical guarantees.

---

### Introduction & Motivation

The motivation is strong: empirical progress has plateaued at ~100–200 bits, yet whether we are genuinely near theoretical limits is an open question with real practical stakes for content provenance. The five competing hypotheses (A through E) are a particularly useful framing device that structures the paper clearly. Contribution (iii) — Chunky Seal — is somewhat understated in the introduction relative to its section length and significance; framing it only as a "scale-up" of Video Seal may make it seem like an engineering afterthought rather than a principled demonstration.

One concern: the introduction claims state-of-the-art progress has "stagnated," citing capacities of 100–200 bits at PSNR >40 dB. This is accurate for most methods, but the very recently appearing WAM and LISO-JPEG points on Figure 1 complicate the narrative slightly. The authors should more carefully characterize those outliers in the introduction, not just in Appendix A.

---

### Section 2: Bounds on Watermarking Capacity

**Strengths.** The geometric framing — modeling the watermarking capacity problem as counting integer lattice points inside an ℓ₂-ball intersected with the image hypercube — is elegant and largely novel relative to the classical information-theoretic approach. The progression from Bound 1 (trivial) through Bounds 2–9 (increasingly refined PSNR-only) to Bounds 10–13 (with robustness) is methodically laid out. The use of Mitchell's (1966) hypersphere lattice-counting algorithm for small-radius exact counting is an appropriate and non-trivial technical tool.

**Concern 1: PSNR as a proxy for imperceptibility.** The paper's central theoretical contribution hinges on equating perceptual imperceptibility with a PSNR (equivalently ℓ₂-ball) constraint. The authors briefly acknowledge that perceptual metrics like LPIPS differ from PSNR, but do not study how much the bounds change under such metrics. For images with highly non-uniform content (edges, textures), the ℓ₂-ball is a poor proxy for the perceptually admissible region. This could mean the bounds are dramatically looser than they appear. Given that the paper's central message is that the gap between theory and practice is enormous, it would substantially strengthen the work to show that the gap persists even under perceptually-motivated constraints (e.g., LPIPS ≤ δ). The dismissal of hypothesis **B** (perceptual constraints close the gap) in Section 3.1 relies solely on showing Video Seal fails on gray images under PSNR — but this tests hypothesis B indirectly at best.

**Concern 2: Heuristic robustness bounds (Bounds 10–12).** The paper explicitly states in Section 2.5 that Bounds 10–12 are *heuristic*: "We can show cases where these heuristic bounds under-approximate and cases where they over-approximate the true capacity." This is an unusual situation — bounds that can go in either direction are not bounds in the formal sense. The conservative Bound 13 is then introduced as an actual lower bound but acknowledged as "extremely conservative and unrealistic." The practical take-away (Figure 4) rests mainly on the heuristic Bounds 10–12. While the authors are transparent about this, the gap between the heuristic bounds (0.5 bpp under aggressive crop) and the conservative bound (Table 2: 0.015 bpp for 50% Crop&Rescale, or ~3,000 bits for 256×256) is extremely large — roughly 30×. This range is too wide to draw the strong conclusion that "robustness to geometric transformations... cannot fully explain the low watermarking capacity." If the true capacity under aggressive cropping is closer to the conservative bound (Table 2), and JPEG/augmentation constraints compound with it, the gap between theory and 0.001 bpp practice becomes much more modest. The paper needs a tighter bound or a more qualified claim.

**Concern 3: Independence of channels.** The bounds treat the cwh-dimensional pixel space as isotropic. In practice, RGB channels are correlated (sometimes one only has Y-channel watermarking, as in the base Video Seal), JPEG compresses in YCbCr, and perceptual metrics treat channels differently. The paper notes the switch to all-three-channel watermarking in Chunky Seal, but the bounds themselves assume a channel-symmetric treatment. The implications of channel correlations for the bounds are not analyzed.

**Concern 4: Data distribution argument (Section 2.6).** The argument that the data distribution reduces capacity by only ~10,240 bits (based on VQGAN codebook size) is appealing but circular: if the VQGAN codebook provides an upper bound on distinct perceptual images, it should more properly be compared against the *perceptual* ball rather than the *ℓ₂* ball. Using this argument to dismiss hypothesis **C** while simultaneously relying on PSNR as a proxy for perception is inconsistent.

---

### Section 3: Empirical Performance Gap

**Strengths.** The experimental design of Section 3 is admirably simple and controlled: isolating Video Seal on a single fixed gray image with no augmentations directly targets whether architectural constraints (not task constraints) are responsible for the gap. The finding that a 256×256 model achieves similar results to a 32×32 model is a compelling demonstration of architecture failure to exploit resolution. The handcrafted embedder (Equation 2), which essentially implements quantized scalar coding in each pixel and achieves 456,509 bits at 42 dB on a gray image, elegantly shows the bounds are achievable.

**Concern 1: Single gray image as a proxy for general images.** The controlled experiment rules out hypotheses A, B, C for the *gray-image, no-robustness* case — but the paper then extrapolates this finding to argue that real-world models are broadly under-performing (conclusion E). This extrapolation is only valid if the gray-image case is representative. Natural images have structure that may genuinely reduce achievable capacity in ways that a gray image does not. This limitation is mentioned in passing at Section 5, but deserves more prominent discussion in Section 3, especially given how central the argument is.

**Concern 2: Training protocol and convergence.** The paper sweeps over three learning rates and three λᵢ values, selecting the best run. For 1024 bits in Video Seal (256×256px), the best result is 89.63% bit accuracy at 40.10 dB — clearly a failure. But did all 9 runs fail, or only some? Are the runs shown in Figure 5 best-of-9 cherry picks? The reported sweep results feel under-characterized. For a paper making strong architectural failure claims, fuller learning curves and a statistical summary (e.g., mean and variance over runs at each hyperparameter setting) would be more convincing.

**Concern 3: Linear model success does not directly imply the failure is "architectural."** The linear encoder/decoder achieving 2048 bits in 50 epochs on a *single fixed gray image* is impressive but the task is trivially solvable by any function approximator given the complete absence of variation across training samples. Since the gray image is fixed and messages are the only variable, a linear decoder (from image residual to message) can succeed by design. This is less a proof of Video Seal's architectural limitation and more a demonstration that the task, as simplified, becomes a pure memorization/recall problem that a linear model handles well. The conclusion "all one needs is the right architecture" may be too strong; the right architecture for fixed-gray-image-no-augmentation may look very different from the right architecture for real-world watermarking.

**Concern 4: Tiling.** The tiling strategy of 32×32 patches to yield 32,768 bits is valid but not a practical watermarking scheme — it requires the decoder to know the exact patch boundaries, making it trivially vulnerable to geometric attacks. This should be made explicit; the authors frame it correctly as being in the no-robustness setting but the practical relevance deserves a more prominent caveat.

---

### Section 4: Chunky Seal

**Strengths.** Chunky Seal is a meaningful empirical contribution: a 4× increase in capacity (256→1024 bits) at comparable PSNR, SSIM, MS-SSIM, and overall bit accuracy. Results are reported on both SA-1B and COCO (Appendix J.1), providing some breadth. The explicit table comparing Chunky Seal and Video Seal across nine augmentation types is informative.

**Concern 1: Model size and practical utility.** The embedder is 90× and the extractor is 23× larger than Video Seal. This is noted without any latency benchmarks. For a paper aimed at the ICLR community, omitting inference time is a notable gap. A model that takes 10× longer to embed and cannot run on consumer hardware has limited practical significance, regardless of its capacity.

**Concern 2: LPIPS degradation.** Table 3 shows LPIPS for Chunky Seal is 0.0085 vs. 0.0019 for Video Seal — a 4.5× worse perceptual similarity score. This is non-trivial: LPIPS correlates well with human perception and is routinely used as a quality metric in watermarking papers. The paper dismisses this difference with the phrase "only slightly higher LPIPS" (Section 4), which is inconsistent with a 4.5× difference. Figure 1 plots methods against a PSNR constraint; if one were to use LPIPS as the quality axis, Chunky Seal's position would likely look less favorable. This deserves honest treatment, especially given that Section 2 dismisses hypothesis B (perceptual constraints cannot explain the gap) without formally analyzing LPIPS-based bounds.

**Concern 3: No hyperparameter tuning.** The authors claim Chunky Seal was "achieved without hyperparameter tuning" whereas Video Seal was "extensively optimized." This is used to argue that further gains are possible, but it also means the current comparison may be unfair in the reverse direction: an optimized Chunky Seal might perform significantly better (stronger claim for the paper) or the gap might narrow (weaker claim). This asymmetry should be addressed.

**Concern 4: Bit accuracy at 1024 bits vs. 256 bits.** The paper compares bit accuracy, but with 1024 bits per image, any single-image attack probability scales differently: a bit accuracy of 99.15% over 1024 bits yields an expected 8.7 incorrectly decoded bits per image, meaning the decoded message is almost never exactly correct. Error correction codes are not discussed. At 256 bits and 99.31%, approximately 1.8 bits are in error. For use cases like C2PA manifest embedding, error-free decoding matters. The paper should discuss error correction overhead and its effect on effective capacity.

**Concern 5: Missing comparison against LISO-JPEG.** LISO-JPEG achieves ~1 bpp in Figure 1 (albeit at low PSNR and with weak robustness). Although the authors discuss in Appendix A that LISO is impractical due to quality and robustness limitations, it is conspicuously absent from Table 3's comparison. A fairer comparison would include methods positioned closest to Chunky Seal on the capacity-PSNR tradeoff curve.

---

### Section 5: Discussion and Conclusions

The discussion is appropriately honest: it acknowledges Chunky Seal's impracticality at scale and flags that the architecture-limitation diagnosis may not extend directly to video or more complex robustness settings. The proposed "sanity checks" for future methods (capacity linear in image size, decreasing in PSNR, etc.) are a valuable community contribution. 

One gap: the paper motivates higher capacity primarily through C2PA manifest embedding, but does not discuss the *security* implications of higher-capacity watermarks. Higher capacity watermarks are also easier to read/detect by an adversary who wants to strip them, potentially making provenance systems more vulnerable to targeted attacks. This dimension of the problem is unaddressed.

---

### Limitations & Broader Impact

The paper acknowledges computational intractability at large resolutions for the exact bounds, the heuristic nature of robustness bounds, and Chunky Seal's size. What is missing:

1. **Security considerations:** As noted above, the paper does not discuss adversarial watermark removal or forgery under higher-capacity schemes.
2. **Generalization beyond PSNR:** Perceptual quality metrics are only mentioned as an acknowledged limitation without any analysis of their impact on the bounds.
3. **Scope of the empirical claims:** The Video Seal architectural failure is demonstrated on a degenerate setup (single fixed gray image). Generalizing this finding to claim that all current architectures broadly underperform requires further evidence.

---

### Overall Assessment

This paper makes a genuinely interesting contribution by establishing geometric capacity bounds for image watermarking and demonstrating — both theoretically and empirically — that current deep learning-based methods are far from these limits. The three-pronged structure (theory → empirical gap analysis → proof-of-concept improvement) is well-organized and the controlled gray-image experiments are elegant. However, several weaknesses limit its current form. Most critically: the robustness bounds (Bounds 10–12) are heuristic and non-monotone (they can over- or under-approximate), which substantially weakens the conclusion that robustness constraints cannot explain the gap — the conservative Bound 13 leaves a 30× uncertainty window. Additionally, PSNR-based bounds may dramatically over-estimate practical capacity for perceptual constraints, undermining the dismissal of hypothesis B. On the empirical side, the Chunky Seal model's 4.5× LPIPS degradation and its massive size are downplayed, and the linear model success in the degenerate single-gray-image setting arguably does not establish architectural failure in the general sense claimed. Nevertheless, the core message — that watermarking capacity has been severely underexplored — is well-supported by the combination of the geometric analysis, the tiling experiments, and the handcrafted encoder result, and the paper provides a valuable theoretical framework and a concrete challenge to the community. With revisions that tighten the robustness bounds argument, more honestly characterize LPIPS quality tradeoffs, and temper some of the stronger extrapolations, this work would be a solid ICLR contribution.

# Neutral Reviewer
## Balanced Review

### Summary
This paper establishes new geometric capacity bounds for image watermarking under PSNR and linear robustness constraints, arguing that current neural models operate far below these theoretical limits. Through empirical experiments, the authors demonstrate that existing architectures like Video Seal struggle to scale capacity, while a scaled-up model, "Chunky Seal," achieves 1024-bit capacity with comparable robustness and quality. The work highlights a significant gap between theoretical feasibility and current empirical performance, suggesting that architectural innovation rather than fundamental limits is the bottleneck.

### Strengths
1.  **Novel Theoretical Framework:** The paper moves beyond traditional Gaussian information-theoretic bounds to derive capacity limits based on the geometry of discrete image grids (lattice point counting within ℓ₂-balls). This offers a fresh perspective on watermarking capacity that is directly tied to pixel resolution and quality constraints (Section 2).
2.  **Clear Empirical Demonstration of Architectural Bottlenecks:** The comparison between Video Seal, simple linear models on gray images, and the scaled "Chunky Seal" provides strong evidence that current performance is limited by architecture rather than data or difficulty. The finding that a small linear model outperforms Video Seal on gray images with 1024+ bits is particularly compelling (Section 3.2, Figure 5).
3.  **Practical Proof-of-Concept:** The "Chunky Seal" model serves as a concrete validation that higher capacities (1024 bits vs. standard ~256 bits) are achievable without sacrificing image quality (PSNR ~45dB) or robustness to common augmentations (Section 4, Table 3). This pushes the empirical Pareto frontier for watermarking.
4.  **Rigor in Bound Derivation:** The authors provide specific algorithms (e.g., Algorithm 2 for lattice point counting) and handle multiple regimes (low/medium/high PSNR, arbitrary cover images), ensuring the theoretical claims are mathematically grounded rather than heuristic estimates.

### Weaknesses
1.  **Learnability vs. Geometric Capacity:** While the paper establishes upper bounds on the number of valid images, it does not rigorously address the *learnability* of these bounds by neural networks. Counting points in a ball does not guarantee a neural encoder/decoder can partition this space to recover bits robustly, especially for natural images (Section 2.6 dismisses data distribution effects too quickly).
2.  **Over-reliance on Simplified Baselines for Theory Validation:** The strongest empirical evidence for higher capacity comes from linear models on solid gray images (Table 1). While valid for the theoretical setup, these baselines do not generalize to the natural image manifold, potentially overstating the immediate applicability of the capacity bounds to real-world scenarios.
3.  **Efficiency Trade-offs in Chunky Seal:** The proposed solution for higher capacity (scaling model size by ~90×) comes at a significant computational cost (embedding time 0.27s vs 0.06s in Table 3). The paper acknowledges this but treats model scaling more as an architectural proof-of-concept than a deployable solution, which weakens the argument for "innovation" in training strategies specifically.
4.  **Conservative Robustness Bounds:** The robustness bounds (Section 2.5, Table 2) rely on heuristics for linear transformations and conservative lower bounds that admit being "extremely conservative." This introduces uncertainty into the precise gap between theory and practice, as the heuristic scaling factors for singular values may not perfectly capture real-world distortion effects (e.g., quantization artifacts).

### Novelty & Significance
*   **Novelty:** Moderate to High. The geometric approach to capacity bounds is distinct from the standard literature on watermarking (which relies heavily on Cover's "dirty paper" coding). The empirical demonstration that scaling existing watermarking architectures can yield large capacity jumps is also a novel finding in this specific domain.
*   **Significance:** High. The field often assumes robust image watermarking has hit a ceiling (e.g., ~200 bits). This paper challenges that narrative with theoretical evidence and practical models, potentially unlocking new applications for content provenance that require larger payloads (e.g., embedding full manifests).
*   **Clarity:** Good. The paper is structured logically, moving from theory to simplified experiments to full-scale models. Despite OCR artifacts in the provided text, the mathematical derivations and experimental logic appear coherent in the source material.
*   **Reproducibility:** Good. The authors commit to releasing code and checkpoints for Chunky Seal. The algorithms for lattice counting are described with pseudo-code, allowing the bounds to be replicated.

### Suggestions for Improvement
1.  **Deepen the "Learnability" Discussion:** The paper should expand on why neural networks fail to approach the geometric bounds better than linear models on gray images. Does the inductive bias of U-Nets or Transformers inherently restrict the effective search space for watermark bits? Addressing the relationship between model capacity (parameters) and *informational* capacity would strengthen the "architecture limitation" claim.
2.  **Natural Image Generalization of Bounds:** Include an evaluation of the theoretical bounds specifically on a dataset of natural images (rather than just the single gray image assumption) to quantify the entropy of natural image manifolds more accurately. Section 2.6's estimate based on VQ-VAE should be contextualized with why it doesn't reduce capacity by more than ~1 bpp.
3.  **Clarify Efficiency Implications:** Since "Chunky Seal" is significantly larger, provide a discussion on whether future architectures (e.g., attention-based, sparse) could achieve this capacity more efficiently. ICLR reviewers expect insights beyond "make it bigger."
4.  **Robustness Heuristic Validation:** Provide more empirical data validating the heuristic scaling factors for robustness (Equation 6). Comparing the theoretical singular value reduction against measured bit accuracy degradation on robust augmentations would better quantify the gap between the heuristic bounds and reality.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on Chunky Seal's architectural changes** — The paper claims scaling enables 4× capacity, but doesn't isolate whether improvement comes from embedding dimension, U-Net channels, ConvNeXt depth, or multi-channel embedding. Without this, the claim that "architectural innovation" is needed (vs. just scaling) is unsupported.

2. **Efficiency vs. capacity tradeoff analysis** — Chunky Seal is 90× larger (embedder) and 23× larger (extractor) than Video Seal, but no analysis shows whether this scaling is practical or exhibits diminishing returns. ICLR reviewers expect computational cost to be evaluated for claimed improvements.

3. **Baseline scaling comparison** — The paper shows Video Seal fails at 1024 bits but doesn't test whether other SOTA methods (HiDDeN, MBRS, TrustMark) could also scale to higher capacities. Without this, the claim that "current architectures fall well short" may just reflect Video Seal's specific limitations.

4. **Adaptive attack evaluation** — Robustness is tested only on fixed transformations (crop, rotate, JPEG). No evaluation against adaptive attackers who know the watermarking scheme and specifically target bit extraction. This is critical for security claims.

5. **Real-world pipeline testing** — SA-1B and COCO results don't reflect actual deployment (e.g., social media compression pipelines like Instagram/Twitter). Performance drop on real platforms would significantly undermine practical claims.

### Deeper Analysis Needed (top 3-5 only)
1. **Why linear models outperform Video Seal** — The paper shows a linear embedder/extractor achieves 2048 bits while Video Seal fails at 1024 bits on gray images, but provides no analysis of *why* the neural architecture fails. This is central to the claim of "structural limitations."

2. **Perceptual quality discrepancy** — Chunky Seal has worse LPIPS than Video Seal (0.0085 vs 0.0019 in Table 3), yet the paper claims "nearly identical image quality." This contradiction needs resolution—PSNR alone is insufficient for perceptual claims.

3. **Bound tightness validation** — Theoretical bounds predict ~600,000 bits at 40 dB PSNR for 256×256 images, but even the handcrafted method achieves only 456,509 bits. No analysis explains the remaining gap or whether bounds are actually achievable.

4. **Blind decoding constraints** — The bounds assume the decoder knows the cover image position in pixel space, but practical blind watermarking doesn't have this. The paper acknowledges this in Section 2.6 but doesn't quantify its impact on achievable capacity.

5. **Diminishing returns analysis** — No discussion of whether capacity scales linearly with model size or plateaus. The claim that "substantially higher capacities are within reach" needs evidence beyond a single 4× scaling point.

### Visualizations & Case Studies
1. **Watermark pattern visualization** — Show the actual embedded signals from Video Seal vs. Chunky Seal vs. linear model. This would reveal whether higher capacity uses image structure differently or just increases signal magnitude.

2. **Failure case breakdown by transformation type** — Table 4/5 show bit accuracy drops to ~50% for 32% crop and 30°+ rotation. Visualize which specific images/bits fail to understand if failures are systematic or random.

3. **Capacity-PSNR-robustness Pareto frontier** — Plot all methods (including baselines) on a 3D tradeoff curve to show whether Chunky Seal actually advances the frontier or just moves along it.

### Obvious Next Steps
1. **Train a properly tuned baseline at 1024 bits** — Video Seal was "extensively optimized" for 256 bits but Chunky Seal at 1024 bits was trained "without hyperparameter tuning." This confounds capacity claims with optimization effort.

2. **Evaluate on video watermarking** — The paper uses Video Seal (designed for video) but only tests on images. If claims generalize, video results should be included given the base model's intended use case.

3. **Test progressive scaling** — Instead of one jump from 256→1024 bits, show results at 256, 512, 1024, 2048 bits to demonstrate capacity scales predictably with model size.

4. **Include inference latency in main results** — Embedding time (0.27s vs 0.06s) and extraction time differences should be in Table 3, not buried in Appendix J. This affects practical deployability claims.

# Final Consolidated Review
## Summary

This paper establishes geometric capacity bounds for image watermarking by counting integer lattice points within ℓ₂-balls intersected with the image hypercube, demonstrating that current neural watermarking methods operate orders of magnitude below theoretical limits. Through controlled experiments, the authors show that Video Seal fails to scale capacity even on trivially simplified tasks (single gray image, no augmentations), while a linear model and a scaled-up variant ("Chunky Seal") achieve substantially higher capacities, supporting the conclusion that architectural limitations—not fundamental limits—are the primary bottleneck.

## Strengths

- **Novel theoretical framework:** The geometric approach to capacity bounds via lattice point counting within PSNR balls (Bounds 1-13) is more directly tied to actual image constraints than classical information-theoretic approaches based on Gaussian noise models.

- **Compelling controlled experiments:** The gray-image experiments (Section 3) cleanly isolate architectural failure—Video Seal achieves similar results at 256×256px and 32×32px, demonstrating inability to exploit resolution, while a simple linear embedder/extractor succeeds at 2048 bits.

- **Existence proof for bounds:** The handcrafted embedder (Equation 2) achieves 456,509 bits at 42 dB on gray images, demonstrating the theoretical bounds are achievable in principle and not merely mathematical artifacts.

- **Meaningful empirical advance:** Chunky Seal achieves 4× higher capacity (1024 vs 256 bits) at comparable PSNR (~45 dB) and robustness across standard augmentations, pushing the empirical frontier.

## Weaknesses

- **Heuristic robustness bounds undermine strong conclusions:** Bounds 10-12 are explicitly heuristic—capable of over- or under-approximating true capacity—while Bound 13 is acknowledged as "extremely conservative." The gap between heuristic predictions (~0.5 bpp for aggressive crop) and conservative bounds (~0.015 bpp, Table 2) spans ~30×, which substantially weakens the paper's conclusion that robustness constraints "cannot fully explain the low watermarking capacity of current models."

- **LPIPS degradation is downplayed:** Chunky Seal's LPIPS of 0.0085 versus Video Seal's 0.0019 (Table 3) represents a 4.5× worse perceptual similarity score. The paper's characterization of this as "only slightly higher LPIPS" (Section 4) is misleading given the magnitude of the difference. If quality were measured by LPIPS rather than PSNR, Chunky Seal's position would look less favorable.

- **Extrapolation from gray-image experiments to general architectural failure is not fully justified:** The strongest evidence for architectural limitations comes from single-gray-image experiments, but natural images have structure that may genuinely reduce achievable capacity. The paper acknowledges this limitation but the central claim—"models are likely significantly underperforming"—rests partly on this extrapolation.

- **Error correction overhead not discussed:** At 1024 bits with 99.15% bit accuracy, the expected ~8.7 incorrect bits per image means the decoded message is rarely exact. For applications like C2PA manifest embedding where exact recovery matters, effective capacity after error correction would be lower.

## Nice-to-Haves

- **Ablation of Chunky Seal architectural changes:** The paper attributes capacity gains to scaling, but doesn't isolate whether improvement comes from embedding dimension, U-Net channels, ConvNeXt depth, or multi-channel embedding. This would clarify whether "architectural innovation" or mere scaling is the key.

- **Inclusion of LISO-JPEG in empirical comparison:** Figure 1 shows LISO-JPEG achieving ~1 bpp (albeit at lower PSNR), but it's excluded from Table 3. Including it would provide context for where Chunky Seal sits on the capacity-quality tradeoff curve.

- **Analysis of why linear models outperform Video Seal on gray images:** The paper demonstrates this phenomenon but doesn't explain it—theoretical analysis of what architectural properties prevent Video Seal from learning simple identity mappings would strengthen the structural limitation claim.

## Removed Points

These points are flagged for removal but preserved for reference:

- *Security against adaptive attackers is outside the paper's stated scope and contribution.*

- *Requests for natural image bounds are already partially addressed by SA-1B and COCO evaluation in Appendix J; this is not a missing experiment.*

- *Request for training curves and statistical summaries across runs is reasonable but not essential given the clear failure modes demonstrated (Video Seal never achieves 100% bit accuracy at 1024 bits even in best runs).*

- *Efficiency analysis beyond model size is a nice-to-have; the paper already reports embedding time (0.27s vs 0.06s) in Appendix J.*

## Novel Insights

The paper's most significant insight is that watermarking capacity has been constrained by architectural choices rather than fundamental limits—a finding that runs counter to the field's implicit assumption of diminishing returns. The demonstration that a simple linear model can embed 2048 bits on a gray image while Video Seal fails at 1024 bits suggests that current neural architectures have inductive biases poorly suited to the watermarking task, potentially because they were designed for perceptual tasks where learning identity mappings is difficult. The lattice-based bound derivation also provides a concrete, computable alternative to information-theoretic approaches that rely on intractable image distributions.

## Suggestions

- Tighten the robustness bounds analysis: either provide a formal upper bound, or explicitly quantify the uncertainty range and its implications for the main conclusion.

- Report LPIPS honestly as a quality metric and acknowledge the perceptual quality tradeoff rather than dismissing it; consider whether 0.0085 LPIPS meets practical watermarking requirements.

- Add a brief discussion of error correction codes and their impact on effective capacity for applications requiring exact message recovery.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 6.0]
Average score: 5.3
Binary outcome: Reject
