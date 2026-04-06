=== CALIBRATION EXAMPLE 37 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the core contributions (trajectory and frequency guidance for rectified flow editing). The abstract clearly states the problem (multi-object editing challenges), proposed solutions (SPO, trajectory optimization in time/frequency domains, attention injection), and claims (outperforms existing methods, high user preference). Claims are supported by the experiments presented later.

### Introduction & Motivation
The introduction effectively motivates the problem: editing complex, multi-object scenes with rectified flow models remains difficult due to semantic entanglement and structural inconsistency. It succinctly critiques inversion-based, attention-based, and mask-based methods, establishing a clear gap. The contributions are explicitly listed and align with the paper's content. However, the flow from problem to solution is slightly disrupted by a premature mention of "FlowEdit" (an existing method) before fully introducing the proposed approach.

### Method / Approach
This is the core section and contains several innovative ideas (SPO, vector orthogonalization, frequency-domain scaling, phase-aware attention injection). However, key details are missing or unclear, which hinders reproducibility and full evaluation:

1.  **Starting Point Optimization (SPO):** The criterion for determining the optimal start point \( T_0 \) is described as when "the low frequency and overall values of CosSim2 are first equal" (Fig. 4 caption). This is ambiguous. The operational definition (how this equality is measured/calculated) and the rationale for using low-frequency MSE for SPO are not sufficiently detailed.
2.  **Trajectory Optimization – Time Domain:** Equation (8) introduces a weight \( \omega \) for the orthogonalized cross-prompt term \( \Delta_{ts}^{orth} \). The value or schedule for \( \omega \) is never defined, making this step incomplete.
3.  **Trajectory Optimization – Frequency Domain:** Equation (11) for the adaptive scaling coefficient \( \lambda_{type}(t_i) \) is poorly formatted and hard to parse. The description "compensates for missing frequency information" is vague. The role and setting of the hyperparameter \( \alpha \) are not explained.
4.  **Attention Remapping:** The token mapping function \( \phi(j) \) is crucial for determining correspondences between source and target prompt tokens (e.g., for object replacement), but its implementation is not described. The choice of which attention layers to inject features into during different phases (layers 5-20, 20-45) is justified via an empirical analysis in the appendix, but the main text lacks a principled explanation linking this to the frequency-aware properties of MM-DiT.
5.  **Mathematical Correctness:** The vector orthogonalization in Eq. (7) appears mathematically sound for removing the component of \( \Delta_{ts} \) that aligns with \( \Delta_{ss} \). The overall editing ODE (Eq. 5) follows standard rectified flow formulation.

### Experiments & Results
The experimental setup is comprehensive: multiple datasets (PIE-Bench, PIE-Bench++, OIR), a wide range of baseline methods (both DM and RF-based), and standard metrics for preservation (PSNR, LPIPS, SSIM) and alignment (CLIP). The results in Tables 1 and 2 show strong performance, often best or second-best, indicating a good balance between edit fidelity and source preservation.

**Major Concerns:**
- **Metric Interpretation:** The paper claims superiority based on metrics like Structure Distance and LPIPS, which heavily penalize any change. For editing tasks that *should* alter the background (e.g., "change time of day"), high preservation scores might indicate insufficient editing, not success. The "Edited" CLIP score helps but isn't a perfect counterbalance. The trade-off should be discussed.
- **Baseline Completeness:** While many baselines are included, a notable omission is **FlowAlign (Kim et al., 2025)**, a concurrent and highly relevant inversion-free method for rectified flow editing cited in Related Work. Its absence weakens the claim of state-of-the-art performance.
- **Statistical Significance & User Study:** The user study results (Table 4) show a dominant preference for the proposed method (>50%). However, details are relegated to the appendix. The main text should state the number of participants, questions, and the precise evaluation protocol (e.g., side-by-side comparison, criteria) to assess validity.
- **Ablation Study:** The ablation (Table 3, Fig. 8) effectively shows each component's contribution. However, it does not ablate the *frequency-domain* scaling separately from the time-domain orthogonalization (both are part of "Trajectory Optimization"). Disentangling their individual effects would be informative.

### Writing & Clarity
The paper is generally well-structured and readable. However, the **Method** section suffers from significant clarity issues, as noted above (undefined variables, ambiguous descriptions). Figures and their captions (e.g., Fig. 3, 4) are referenced but are not fully self-explanatory without careful reading of the text. Some formatting/parsing artifacts (e.g., broken equation (11), misplaced figure references) slightly disrupt the flow but do not fundamentally obscure the content.

### Limitations & Broader Impact
**This section is entirely missing**, which is a critical flaw for an ICLR submission. The authors must add a discussion covering:
- **Technical Limitations:** Failure cases (e.g., edits requiring extreme geometrical or topological changes), dependence on the FLUX model's architecture (MM-DiT), and computational cost.
- **Broader Impact:** Potential misuse for creating deceptive imagery (deepfakes), and any biases inherited from the base generative model.

## Overall Assessment
This paper presents a thoughtful and novel approach to image editing with rectified flow models, introducing several technically interesting ideas (adaptive start point, dual-domain trajectory optimization, phase-aware attention injection). The experimental results are solid and demonstrate improved performance over a strong set of baselines. However, the current manuscript has significant shortcomings: the method description lacks crucial details for reproducibility, the experiments omit a key baseline and lack a thorough discussion of metric trade-offs, and—most critically—it completely neglects the required limitations and broader impact section. Addressing these issues, particularly by providing a complete, clear method specification and a honest discussion of limitations, is essential for this work to meet ICLR's standards. The core ideas are promising, but the presentation must be strengthened.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a dual-domain framework for text-guided image editing using Rectified Flow (RF) models, specifically targeting complex multi-object scenes. The core contributions are an adaptive Starting Point Optimization (SPO) strategy to determine the optimal timestep to begin editing based on image structural complexity, and a Trajectory Optimization (TO) method that refines edits in both time (via semantic-aware vector orthogonalization) and frequency (via dynamic re-weighting) domains. The method also selectively injects structural priors from the source image into specific attention layers of the MM-DiT architecture across different denoising phases.

### Strengths
1. **Well-Motivated and Analyzed Framework**: The paper provides a clear, phase-based analysis (Chaos, Layout, Refinement) of the editing process in noise space, which directly motivates the design of SPO and TO. The observation of frequency shifts and attention layer behaviors (Figs. 3, 4, 10) offers a solid foundation for the proposed technical components.
2. **Comprehensive and Strong Empirical Evaluation**: The method is evaluated on multiple benchmarks (PIE-Bench, PIE-Bench++, OIR) for both single and multi-object editing. Quantitative results (Tables 1, 2) demonstrate superior or competitive performance across a wide range of metrics (structure preservation, background consistency, CLIP similarity), significantly outperforming many recent baselines. The extensive ablation studies (Table 3, Fig. 8) effectively validate the contribution of each component.
3. **Practical Utility and Qualitative Results**: The method enables a diverse set of editing operations (addition, replacement, modification, style change) without relying on masks, as shown in numerous qualitative examples (Figs. 1, 7, 13-16). The results show improved handling of complex scenes and reduced semantic entanglement compared to prior work.

### Weaknesses
1. **Limited Novelty in Core Components**: While the integration is novel, several core ideas have precedents. The phase-based analysis resembles observations in diffusion editing literature (e.g., Yu et al., 2023). Vector orthogonalization for disentanglement is a known technique. Frequency-domain manipulation has been explored in other generative tasks. The paper could more clearly delineate its novel insights versus the adaptation of existing concepts.
2. **Technical Clarity and Reproducibility Concerns**: The description of the frequency-aware scaling (Eq. 11) and the attention remapping (Eq. 12) is dense and potentially ambiguous. Key details (e.g., the exact definition of low/high frequency regions `R_low/R_high`, the mapping function φ(j)) are deferred to the appendix or insufficiently explained, which may hinder reproduction. The heavy reliance on the non-open-source FLUX model also poses a reproducibility barrier for the community.
3. **Incomplete Comparative Analysis**: The evaluation lacks comparison against some very recent and relevant state-of-the-art methods for RF-based editing (e.g., FlowAlign (Kim et al., 2025), which also focuses on trajectory regularization). The user study, while positive, has limited scale (28 users, 14 questions) and does not report statistical significance tests.

### Novelty & Significance
**Novelty:** The primary novelty lies in the *integration* of a phase-aware, dual-domain (time and frequency) optimization strategy within the RF/MM-DiT framework for inversion-free editing. The adaptive SPO strategy and the phase-specific, frequency-informed attention layer selection are distinctive contributions.
**Significance:** The work addresses a pertinent challenge—multi-object editing in powerful but under-explored RF models. The demonstrated improvements in balancing editability and fidelity are meaningful. If the method is made reproducible and the insights about frequency-domain behavior in MM-DiT are generalized, it could influence future editing approaches for transformer-based diffusion/flow models.

### Suggestions for Improvement
1. **Enhance Methodological Clarity**: Provide a more intuitive, step-by-step explanation of the algorithm, perhaps with a pseudo-code block in the main paper. Crucially, elaborate on the implementation details of Eqs. 11 and 12 in the main text (e.g., threshold for frequency split, token mapping logic) to improve reproducibility.
2. **Strengthen the Novelty Narrative and Comparisons**: Conduct a more thorough discussion relating and contrasting the phase analysis and frequency manipulation to prior work in diffusion (not just RF) models. Include comparisons with additional strong contemporaneous baselines like FlowAlign to better situate the contribution.
3. **Deepen the Analysis and Ablation**: Perform an ablation on the `α` parameter in Eq. 11 to show the sensitivity and rationale for its setting. Analyze failure cases more explicitly: under what conditions (e.g., extreme structural changes, very long prompts) does the method still struggle? This would better define the method's boundaries.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **No comparison to key recent SotA multi-object editing methods.** The paper benchmarks against older methods (OIR, PnP) but omits critical recent works on multi-object/multi-concept editing specifically for RF/transformer architectures (e.g., AnimateDiff for multi-concept, Parallel-Edits, RichControl). Without this, the claim of "significantly outperforming" in multi-object editing is unsupported.
2. **Inadequate failure analysis on complex scenes.** The paper shows successful cases but lacks a systematic experiment showing failure modes (e.g., edits involving >3 objects, significant scene layout changes, or attribute binding failures). A dedicated "challenge set" with quantitative failure rates is needed to establish the method's true limits.
3. **Ablation on frequency weighting hyperparameter `α` is missing.** The core `λ_type(t_i)` formula (Eq. 11) introduces a critical hyperparameter `α` that controls the frequency re-weighting. No ablation study shows how performance degrades with different `α` values or justifies the chosen value. This makes the frequency-domain contribution unvalidated.
4. **No user study for multi-object editing.** The reported user preference (54.8%) is for a mix of single and multi-object methods, but the key claim is enabling *multi-object* editing. A separate, focused user study comparing only multi-object editing methods (OIR, Paralleledits, etc.) is essential to prove this claim.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantitative analysis of the "phases" (Chaos, Layout, Refinement).** The paper defines phases based on cosine similarity but provides no quantitative evidence that SPO and layer injection boundaries (`T_0`, `T_turn`) are optimal or general. An analysis showing metric sensitivity to these boundaries (e.g., how PSNR/CLIP changes if `T_0` is shifted) is crucial.
2. **Analysis of frequency-domain behavior is qualitative only.** The claim of "stage-specific spectral characteristics" and adaptive re-weighting is central but is supported only by a single, unclear equation and a qualitative layer visualization (Fig. 10). A quantitative analysis (e.g., plotting average frequency energy in `V_edit` across timesteps, or showing the effect of `λ` on output spectrum) is necessary to trust the mechanism.
3. **No analysis of cross-attention remapping effectiveness.** The token mapping function `φ(j)` for cross-attention injection (Sec. 3.4) is a key component for preventing leakage, but there is no analysis of its success rate (e.g., % of tokens correctly mapped) or its impact when prompts are semantically complex (not simple word swaps).

### Visualizations & Case Studies
1. **Visualization of the orthogonalization and frequency weighting effects.** The text claims orthogonalization reduces artifacts and frequency weighting balances detail. To verify this, side-by-side visualizations of intermediate `V_edit` vectors, their frequency spectra, and the corresponding image updates at key timesteps are needed.
2. **Case studies showing *binding* of attributes to specific objects in multi-object edits.** The paper claims to edit multiple objects without masks. To prove this, show a series of edits on the same source image where different target prompt words are altered, demonstrating that changes are correctly localized (e.g., "change the cat to a dog *and* the sofa to wood" – does the edit affect only the specified objects?).

### Obvious Next Steps
1. **Formalize and justify the layer selection strategy (5-20, 20-45).** The choice of injection layers for layout vs. refinement phases is based on an ad-hoc ablation (Appendix B.1). This needs a principled justification linked to the known frequency properties of MM-DiT layers, perhaps via a systematic sweep and correlation analysis with frequency bands.
2. **Include a baseline that directly uses a predicted mask.** Since the method is mask-free, a critical baseline is to compare against a simple pipeline that uses an off-the-shelf segmentation model (e.g., SAM) to generate a mask for the described objects, followed by a masked-editing method. This would concretely demonstrate the advantage of being mask-free.
3. **Clarify the relationship and novelty over FlowEdit.** The method is inversion-free like FlowEdit. A direct, detailed comparison (both qualitative and quantitative) highlighting the specific improvements attributable to SPO and trajectory optimization over FlowEdit's core mechanism is required, as FlowEdit is the most directly related prior work.

# Final Consolidated Review
## Summary
This paper proposes a dual-domain framework for text-guided image editing with Rectified Flow models, focusing on complex multi-object scenes. It introduces an adaptive Starting Point Optimization (SPO) strategy and a Trajectory Optimization (TO) method that refines edits in both time (via vector orthogonalization) and frequency (via dynamic re-weighting) domains, along with phase-aware attention injection.

## Strengths
- The phase-based analysis (Chaos, Layout, Refinement) of the editing process is well-motivated and directly informs the design of SPO and TO, providing a clear conceptual framework.
- Extensive experiments on multiple benchmarks (PIE-Bench, PIE-Bench++, OIR) demonstrate strong quantitative performance across a range of metrics, balancing edit fidelity and source preservation better than many existing methods.
- The method enables diverse, mask-free editing operations (addition, replacement, modification) on complex multi-object scenes, as shown by compelling qualitative results.

## Weaknesses
- The methodological description lacks clarity in key parts, hindering reproducibility. The criterion for SPO (Fig. 4) is ambiguous, the weight ω in Eq. (8) is undefined, and the frequency scaling mechanism (Eq. 11) is poorly explained without defining critical hyperparameters (e.g., α) or frequency regions.
- The comparison omits a key contemporary baseline, FlowAlign (Kim et al., 2025), which also addresses inversion-free editing with rectified flow. Its absence weakens the claim of state-of-the-art performance.
- The paper lacks a dedicated limitations and broader impact section, which is a standard expectation for ICLR submissions.

## Nice-to-Haves
- A separate ablation study isolating the frequency-domain component of TO would help disentangle its contribution from the time-domain orthogonalization.
- A more detailed description of the user study protocol in the main text (e.g., number of participants, exact criteria) would strengthen the validity of the preference claims.
- Analysis of typical failure cases (e.g., edits requiring extreme geometric changes) would better define the method's boundaries and practical applicability.

## Novel Insights
The paper's primary novel insight lies in the integrated application of time-domain vector orthogonalization and frequency-domain adaptive scaling within the rectified flow framework, coupled with a phase-aware strategy for selecting which attention layers to inject. This dual-domain approach effectively balances edit strength and structural preservation in multi-object editing. However, the core components individually—phase analysis, orthogonalization for disentanglement, and frequency manipulation—have precedents in the broader diffusion/editing literature.

## Suggestions
- Provide a clearer, step-by-step description of the algorithm, including explicit definitions for all variables and hyperparameters (e.g., ω, α, R_low/R_high) either in the main text or a well-referenced appendix.
- Add quantitative and qualitative comparisons with FlowAlign to better situate the contribution.
- Include a limitations and broader impact section addressing computational cost, failure modes, and potential societal implications.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 2.0, 4.0, 2.0]
Average score: 2.8
Binary outcome: Reject
