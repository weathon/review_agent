## Summary

The paper introduces Content-Aware Mamba (CAM), a state-space model adapted for learned image compression that addresses two limitations of standard Mamba: its rigid raster-scan order and strict causality. CAM proposes Content-Adaptive Token Permutation (CTP), which reorders tokens by feature-space clustering so semantically similar tokens are processed consecutively, and Global-Prior Prompting (GPP), which injects sample-specific prompts derived from cluster centroids into the SSM's output projection matrix to provide global context. The resulting CMIC model achieves state-of-the-art rate-distortion performance, surpassing VTM-21.0 by up to 21.34% BD-rate on Tecnick.

## Strengths

- **Well-motivated and novel token permutation strategy.** The observation that Mamba's raster scan separates content-correlated but spatially distant tokens is precise and important for compression. The codebook-based K-Means clustering with EMA updates (Sec. 3.3, Algorithm 1) provides a stable, efficient alternative to naive online K-Means, and the visualization of cluster assignments (Fig. 10) convincingly shows semantically coherent groupings (e.g., centroid #10 for edges, #26 for textured warm regions, #33 for smooth blue/green backgrounds).

- **Strong empirical results with clear margins.** CMIC achieves BD-rate savings of 15.91%–21.34% over VTM-21.0 and outperforms Mamba-based baselines MambaVC and MambaIC by 2.36%–10.09% BD-rate (Tab. 1), all while reducing parameters by 56% and memory by 78% versus MambaIC. These are substantial and consistent improvements across three datasets.

- **Compelling ERF visualizations demonstrating content adaptivity.** Figures 7–9 provide unusually strong mechanistic evidence. The per-image ERF visualizations (Fig. 8) show the model's receptive field concentrating on semantically relevant distant regions (e.g., hair, feathers, shoreline), in stark contrast to the isotropic, content-agnostic ERFs of TCM-L and FTIC. The single-layer analysis in Fig. 9 cleanly isolates the contributions of CTP and GPP to breaking spatial and causal constraints.

- **Efficient architecture avoiding multi-directional scan overhead.** By using a single selective scan with content-adaptive ordering rather than four directional scans, CMIC achieves 78% lower GPU memory and 39% lower decoding latency than MambaIC (Tab. 1), making the efficiency advantage concrete rather than theoretical.

## Weaknesses

### Major:

- **Imprecise claims about "mitigating causality."** The paper repeatedly claims GPP "mitigates the strict causality" (Abstract, Sec. 1, Sec. 3.4). However, the state update $\mathbf{h}_i = \bar{\mathbf{A}}\mathbf{h}_{i-1} + \bar{\mathbf{B}}\mathbf{x}_i$ remains strictly causal—future tokens cannot influence the hidden state of current tokens. GPP modifies only the output projection via $\mathbf{O}_i = (\mathbf{C} + \mathbf{P})\mathbf{h}_i + \mathbf{Dx}_i$, which makes the *output* globally conditioned but does not make the recurrent state accumulation non-causal. The ERF visualizations (Fig. 9) support this: non-zero activations beyond the causal boundary appear because the prompt carries global statistics, not because the state sees future tokens. The paper should clearly distinguish "globally-conditioned causal modeling" from "non-causal modeling," as the current framing overstates the mechanism's capability.

### Minor:

- **Lack of explicit discussion on gradient flow through non-differentiable permutation.** The clustering assignments and token permutation are discrete, non-differentiable operations. While the EMA update for centroids is explained (Algorithm 1), the paper does not explicitly state how gradients propagate through the permutation to update the analysis transform weights $\theta_a$. Presumably, gradients from the SSM output flow back through the inverse permutation to the input tokens (a form of straight-through estimation), but this should be stated clearly, as the correctness of the gradient signal affects convergence guarantees. The training stability experiments in Appendix A.8 provide empirical evidence but do not substitute for an explicit gradient-flow explanation.

- **Incremental novelty of Global-Prior Prompting relative to MambaIRv2.** The paper acknowledges (Sec. 3.4, Appendix A.13) that the attentive state-space equation follows MambaIRv2 (Guo et al., 2024a). The main difference—tying the prompt dictionary to clustering centroids rather than using a standalone learnable matrix—is meaningful but incremental. The ablation in Table 9 (standalone dictionary: -15.02% vs. CAM: -15.91% on Kodak) confirms the benefit is real but modest (~0.9% BD-rate). The primary novelty thus rests more heavily on CTP than on GPP.

- **Decoder-side clustering stability under quantization noise is under-analyzed.** Each CAM block independently clusters its input features. During training, the encoder's CAM blocks receive pre-quantization features, while at inference the decoder's CAM blocks receive features derived from quantized latents $\hat{\mathbf{y}}$. Although the codebook centroids are learned across the dataset distribution and the EMA mechanism provides stability, the paper provides no quantitative analysis of how often cluster assignments differ between training and inference at the decoder side, or whether quantization noise causes tokens near cluster boundaries to flip assignments. This is not the same as an encoder-decoder synchronization problem (since each side clusters independently), but the robustness of decoder-side clustering to distribution shift from quantization deserves at least brief analysis.

- **Entropy model ablation reveals limitation not fully explained.** Section 4.5 notes that "adding CAM yields negligible performance gains while increasing latency" for the entropy model. The brief explanation—that the entropy model models distributions after redundancy removal where local relationships suffice—is reasonable but underdeveloped. If global context is crucial for transform networks, why would the conditional probability model of the transformed latents not benefit from the same awareness? A more principled discussion would strengthen the paper.

### Trivial:

- **Section 3.2 appears as a bare heading with no body text** before Section 3.3 begins. If this is present in the final PDF (not a parsing artifact), the overview architecture description should be completed or the section merged with 3.3.

## Nice-to-Haves

- A direct ablation comparing CMIC against a 4-directional Mamba variant built on the same backbone architecture (matched parameters) would more cleanly validate the efficiency advantage of single-scan CAM over multi-directional scanning, beyond the comparison with the architecturally different MambaIC.
- Analysis of failure cases: Table 5 shows high variance in activated cluster counts. Examining images where few clusters are activated (<20%) versus many (>50%) could reveal when CTP provides diminishing returns and when it is most critical.
- Sensitivity analysis of the K-Means iteration count (currently T=5) and EMA decay λ on final RD performance, to further substantiate robustness beyond the K-value ablation in Table 6.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Bitrate overhead of permutation side-information"** (from Spark Finder): Based on a misunderstanding. The permutation is an internal operation within each CAM block—tokens are clustered, permuted, processed by SSM, then inverse-permuted back to the original spatial layout before exiting the block. No permutation indices need to be transmitted between encoder and decoder. Each side independently computes its own clustering on its own features.
- **"Encoder-decoder synchronization risk from quantization mismatch"** (from Harsh Critic): While framed as a "critical" synchronization issue, this also stems from the mistaken assumption that the encoder's permutation must be reproduced at the decoder. Since each CAM block independently clusters its own features and applies its own permutation internally (with inverse permutation restoring spatial layout at the block output), there is no cross-side synchronization requirement. A residual concern about decoder-side clustering stability under quantization is kept above as a minor weakness.
- **"Distributed training overhead of per-block codebooks"** (from Balanced Review): Not standard to report in this venue for this type of work; the paper already notes codebook parameters are only 0.166% of total (Appendix A.9). Moved to nice-to-have territory.
- **"Training budget fairness verification"** (from Spark Finder): Reporting training GPU hours is not standard practice in the LIC literature. The paper provides standard setup details (optimizer, learning rate, dataset) in Section 4.1.
- **"Cross-dataset generalization to medical/satellite/screen content"** (from Spark Finder): Scope creep. The paper targets natural image compression; testing on out-of-domain modalities is beyond its stated scope.
- **"Section 3.2 empty section is a structural gap"** (from Harsh Critic): If present in the PDF, this is a trivial formatting issue; if a parser artifact, irrelevant either way.
- **"Missing comparison with Zhang et al. (2024b) under matched settings"** (from Spark Finder): The paper already provides a detailed comparison in Appendix A.2, noting Zhang et al. achieves -8.75%/-9.64% BD-rate on Kodak/Tecnick versus CMIC's -15.91%/-21.34%, and explains the architectural differences (grid-anchored coarse clustering vs. fine-grained codebook-based clustering).

## Novel Insights

The paper reveals an interesting asymmetry in the value of global context across different components of a learned image compression pipeline. CTP provides large gains (1.6%–2.2% BD-rate) in the transform networks, where discovering and exploiting long-range redundancy is the core task, but the same mechanism offers negligible benefit in the entropy model, which operates on already-decorrelated latents where local dependencies dominate. This suggests that the "globality premium" in LIC is highly task-dependent: it matters for removing redundancy but not for modeling the residual distribution, a distinction that could inform future architecture design choices beyond this specific method.

## Suggestions

- Revise the causality claims throughout the paper (Abstract, Sec. 1, Sec. 3.4) to use precise language such as "globally-conditioned output projection" or "output-level global context injection" rather than "mitigating strict causality" or "non-causal modeling," since the recurrent state update remains strictly causal.
- Add an explicit paragraph in Section 3.3 explaining how gradients propagate through the non-differentiable permutation (i.e., straight-through estimation via the inverse permutation), and optionally include a small quantitative analysis of cluster assignment consistency between training and inference at the decoder side to address the robustness concern.

---
**Quality Assessment:**
- **Novelty:** Moderate-to-strong. CTP is a genuinely novel and well-motivated mechanism. GPP is a meaningful but incremental adaptation of MambaIRv2's prompting.
- **Technical Soundness:** Mostly sound, with a notable gap in explicitly addressing gradient flow through non-differentiable operations and imprecise causality claims.
- **Empirical Support:** Strong. Comprehensive comparisons, consistent improvements across datasets, compelling ablations and visualizations.
- **Significance:** Strong. Establishes a clear new SOTA for Mamba-based LIC with meaningful efficiency advantages.
- **Clarity:** Generally good, weakened slightly by the overclaiming on causality and the imprecise gradient-flow discussion.