=== CALIBRATION EXAMPLE 25 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**  
The title clearly signals a new paradigm ("Active Speech Enhancement"), though the phrase "Beyond Passive Denoising Declipping and Dereverberation" is slightly redundant. The abstract succinctly states the core idea: ASE actively shapes speech by both suppressing noise and amplifying speech-relevant frequencies, unlike traditional passive enhancement or ANC. The claims of outperforming baselines and providing code/demo are appropriate, but the abstract does not explicitly highlight the key architectural novelty (Transformer-Mamba hybrid) or the joint loss—these are important for ICLR’s novelty bar.

**Introduction**  
The motivation is well laid out: passive enhancement and ANC are complementary, and unifying them is a logical step. The four contributions are clearly listed. However, the introduction does not crisply articulate the *specific limitations* of existing ANC methods that ASE overcomes. For example, why can’t existing ANC architectures be directly used for speech enhancement? A sharper gap statement would strengthen the case. Also, the term “actively shaping the speech signal” is introduced but not precisely defined—does this mean spectral boosting, time-domain amplification, or something else?

**Related Work**  
The coverage of deep learning-based speech enhancement and ANC is thorough and up-to-date. However, the section is mostly a descriptive list of prior work without a critical synthesis that explains why these lines have remained separate and what makes their integration non-trivial. Crucially, the paper later compares ASE-TM against adapted ANC baselines (DeepANC, ARN, etc.), but the related work does not discuss how such adaptations might be done or why they are challenging. This omission makes the experimental comparisons feel somewhat unmotivated.

**Background**  
The derivation of the ANC framework is standard. The transition to ASE is problematic. Equation (4) defines the enhanced signal as \(e_h(n) = d(n) + a(n)\), whereas ANC uses \(e(n) = d(n) - a(n)\). The paper states that ASE seeks to recover clean speech from a noisy mixture, but if \(a(n)\) is the anti-signal (designed to cancel noise), then adding it to \(d(n)\) would *increase* the noise component, not reduce it. The text says “ASE seeks to recover clean speech from a noisy mixture of distorted speech \(x(n)\)”, but the equations do not clearly show how \(a(n)\) is re-purposed from cancellation to enhancement. This is a critical conceptual ambiguity that must be resolved. The authors likely intend \(a(n)\) to be a *modification signal* that, when added, yields enhanced speech, but this is not mathematically distinguished from the ANC anti-signal. The diagram (Figure 1b) does not clarify this point.

**Method**  
*Architecture*: The ASE-TM model is based on SEmamba, with Mamba2 blocks and an inserted attention block. The description is detailed, but the motivation for the hybrid design (why insert attention in the middle of Mamba blocks?) is not justified theoretically or empirically in this section. The claim that Mamba2 offers “potential improvements in efficiency” is vague; a brief explanation or citation would help.  
*Optimization*: The loss function combines multiple components from prior work (MP-SENet, SEmamba). The modifications (e.g., using L1+L2 for magnitude loss) are noted, but the rationale for these choices is not discussed. The consistency loss is a good addition. However, the weighting hyperparameters \(\gamma_i\) are not provided, which affects reproducibility. More importantly, the loss is applied between the enhanced signal \(e_h(n)\) and the clean target \(c(n)\), but how the network learns to produce an \(a(n)\) that achieves this is not explained—this relates to the ambiguity in the ASE formulation.

**Experiments**  
*Datasets and Task Generation*: Standard datasets are used, and the simulation of distortions is appropriate. However, it is unclear whether a single ASE-TM model is trained for all tasks or separate models are used for each. The text suggests separate training because the target \(c(n)\) is defined differently per task, but this should be stated explicitly.  
*Acoustic Path Simulation*: The simulation of primary/secondary paths and loudspeaker non-linearity is thorough and necessary for the active setting. Varying \(T_{60}\) and \(\lambda^2\) during training is good for robustness.  
*Baseline Methods*: A major weakness is the lack of detail on how the ANC baselines (THF-FxLMS, DeepANC, ARN) are “adapted and retrained or configured to the ASE framework.” Without a clear description of these adaptations, the comparisons are not fair or interpretable. For instance, were these models retrained with the same ASE objective (Eq. 4)? Did they also generate an anti-signal \(a(n)\) that is added to \(d(n)\)? If so, how were their architectures modified to output such a signal? This omission undermines the validity of the results.

**Results and Analysis**  
*Denoising Performance*: ASE-TM outperforms baselines by large margins (e.g., PESQ 2.98 vs. 2.45 for ARN). However, the poor performance of DeepANC (PESQ 1.48) and ARN (relative to ASE-TM) raises questions about whether these models were properly adapted. The authors should analyze why these deep ANC methods fail in the ASE setting.  
*Dereverberation and Declipping*: Similar trends are observed. The baselines sometimes perform worse than the unprocessed distorted speech (e.g., DeepANC PESQ 1.06 vs. reverberant 1.60), which suggests the adaptation may be fundamentally flawed. This requires explanation.  
*Ablation Study*: The ablation (Figure 3a) shows contributions of loss modifications, attention, and Mamba2, but it is only shown for denoising. Ablations on other tasks would strengthen the claims. Also, the “modified loss” is not clearly defined—which components were modified?  
*Runtime Analysis*: The real-time feasibility analysis is well done and shows minimal degradation with future-frame prediction.  
*Power Spectra*: Figure 4 shows increased power in enhanced signals, but a qualitative analysis of spectrograms (as in Fig. 3b for denoising) for dereverberation and declipping would be more informative.

**Limitations & Broader Impact**  
The limitations section is cursory. It only notes that baselines were adapted from ANC, which may explain their poor performance, and suggests a unified multi-task model as future work. Important limitations are not discussed: (1) The reliance on simulated acoustic paths; real-world validation is missing. (2) The computational cost (10 days training on an A6000) and model complexity. (3) The assumption of known or identifiable secondary paths—how would ASE perform with estimation errors? (4) Potential negative societal impacts (e.g., misuse for surreptitious speech manipulation) are not considered.

**Writing & Clarity**  
Overall the paper is well-structured, but the core conceptual ambiguity regarding the ASE formulation (Eq. 4) significantly hampers understanding. The method section would benefit from a clearer step-by-step explanation of how the network output leads to an enhanced signal. Some technical details (e.g., loss weights, baseline adaptation) are missing, affecting reproducibility.

### Overall Assessment
The paper introduces a novel paradigm, Active Speech Enhancement, that merges active noise control with speech enhancement. The idea is promising and could be of interest to the ICLR community. The proposed ASE-TM architecture is a sensible hybrid of Transformer and Mamba components, and the comprehensive evaluation across three tasks is a strength. However, the paper has significant weaknesses: (1) The fundamental formulation of ASE is mathematically unclear and potentially contradictory with the ANC framework. (2) The adaptation of baseline methods is not described, making the comparisons questionable. (3) Key implementation details are omitted, affecting reproducibility. (4) The analysis of why baselines fail is lacking. Addressing these issues—especially providing a rigorous derivation of the ASE objective and detailing baseline adaptations—is essential for the paper to meet ICLR’s standards. With major revisions, the contribution could be solid.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Active Speech Enhancement (ASE), a novel paradigm that unifies active noise control and traditional speech enhancement. ASE actively shapes the speech signal by both suppressing interference and amplifying speech-relevant frequencies. The authors propose ASE-TM, a Transformer-Mamba-based architecture with a multi-component loss, and demonstrate superior performance on denoising, dereverberation, and declipping tasks compared to adapted active noise cancellation baselines.

### Strengths
1. **Novel Conceptual Contribution**: The paper formalizes ASE as a new paradigm that bridges active noise cancellation and speech enhancement, offering a fresh perspective with potential for new research directions.
2. **Strong Empirical Results**: Comprehensive experiments across three tasks (denoising, dereverberation, declipping) show consistent and significant improvements in multiple objective metrics (PESQ, STOI, etc.) over adapted ANC baselines.
3. **Architectural Innovation**: ASE-TM effectively integrates Mamba2 blocks with an attention mechanism in a hybrid design, leveraging both long-range dependencies and efficient sequence modeling.
4. **Robustness Analysis**: The model is evaluated under varying acoustic conditions (reverberation times and loudspeaker nonlinearities), demonstrating reliable performance across settings.
5. **Real-time Feasibility**: The paper includes a runtime analysis with future-frame prediction, showing minimal performance degradation while meeting real-time constraints.
6. **Reproducibility**: Detailed descriptions of datasets, acoustic simulation, hyperparameters, and training are provided, and code/demo are promised in supplementary materials.

### Weaknesses
1. **Inadequate Baseline Comparisons**: The paper only compares against ANC methods (THF-FxLMS, DeepANC, ARN) adapted to ASE tasks. State-of-the-art passive speech enhancement methods (e.g., recent diffusion or GAN-based models) are not included, making it unclear whether ASE advances the field beyond existing enhancement techniques.
2. **Limited Ablation Study**: The ablation (Figure 3a) is only shown for denoising, lacks detailed explanation of components, and does not cover other tasks. The contributions of specific loss modifications and architectural choices are not fully quantified.
3. **No Subjective Evaluation**: Reliance on objective metrics (PESQ, STOI) without subjective listening tests (e.g., MOS) weakens claims about perceptual quality and intelligibility improvements.
4. **Computational Cost**: Training required ~10 days on an NVIDIA RTX A6000 GPU, but model size, inference efficiency, and comparisons of computational complexity with baselines are not discussed.
5. **Clarity Issues**: The methodology is dense and heavily relies on references (e.g., SEmamba). The loss weightings (γ) are not provided, and the target signal definition for different tasks could be clearer.

### Novelty & Significance
- **Novelty**: The ASE paradigm is a novel conceptual fusion of active noise control and speech enhancement. The hybrid Transformer-Mamba architecture with Mamba2 and attention is also innovative.
- **Significance**: If substantiated by comparisons to state-of-the-art passive methods, ASE could significantly impact speech enhancement in challenging acoustic environments. The empirical results are promising but require broader validation.

### Suggestions for Improvement
1. **Compare with State-of-the-Art Passive Methods**: Add comparisons to recent strong speech enhancement models (e.g., diffusion, GANs, or Transformer-based) to demonstrate that ASE genuinely advances beyond passive enhancement.
2. **Expand and Clarify Ablation Studies**: Conduct ablations for all tasks, clearly explain each component (e.g., loss terms, attention, Mamba2), and quantify their individual contributions.
3. **Include Subjective Evaluation**: Perform a subjective listening test (e.g., MOS) to validate perceptual quality improvements claimed in the abstract and experiments.
4. **Analyze Computational Efficiency**: Report model size, inference time, and computational complexity relative to baselines to assess practicality for real-time applications.
5. **Improve Methodological Clarity**: Provide loss weighting details, clarify target signal definitions per task, and make the architecture description more self-contained.
6. **Test on More Realistic/Diverse Conditions**: Evaluate on additional datasets (e.g., real-recorded noisy/reverberant clips) and for declipping, test across a range of clipping thresholds.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Missing comparison to state-of-the-art (SOTA) speech enhancement models.** The paper only compares against ANC baselines (THF-FxLMS, DeepANC, ARN). To validate the "active enhancement" claim, it must be benchmarked against SOTA passive enhancement models (e.g., SEmamba, diffusion models, CMGAN) on the same tasks (denoising, dereverberation, declipping). Without this, the performance claims are not meaningful for the speech enhancement community, and the novelty of ASE is unclear.
2. **No ablation on the "active" formulation.** The core idea is that the model predicts a cancellation signal `a(n)` to be added. A critical ablation is to compare this to a model that directly predicts the enhanced signal `e_h(n)` (i.e., standard enhancement) using the same architecture and loss. If the performance is similar, the "active" framework is just a reparameterization and offers no fundamental advantage.
3. **Insufficient validation on real-world or standardized test sets for dereverberation and declipping.** The tasks use VoiceBank data with synthetically generated distortions. For dereverberation, there is no evaluation on established benchmarks like the REVERB challenge or DNS datasets. For declipping, no comparison to specialized methods (e.g., VoiceFixer, which is cited) on their standard test conditions undermines the claimed efficacy.
4. **No experiment varying the simulated acoustic path parameters during test.** The model is trained with variability in `T60` and `λ²`, but Table 4 only shows a few combinations. A systematic sweep, especially for extreme non-linearities (`λ²=0.01`) or very long reverberation times, is needed to properly assess robustness, which is critical for a claimed "active" system.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of what the model is actually amplifying/enhancing.** The paper claims ASE "amplifies speech-relevant frequencies." A frequency-band or formant-level analysis comparing the spectrum of the output `e_h(n)` to the clean target `c(n)` and the noisy input `d(n)` is essential. Does it selectively boost specific bands, or is it just a global gain? This analysis is needed to substantiate the core "active shaping" claim.
2. **Analysis of the generated cancellation signal `a(n)`.** What does the anti-signal look like in the time-frequency domain for different tasks? Is it primarily containing noise (for cancellation) or also speech components (for enhancement)? Visualizing and analyzing `a(n)` would directly reveal the mechanism of the proposed "active" operation.
3. **Causal vs. non-causal performance gap analysis.** The paper mentions a minimal performance drop in causal mode but only reports aggregate PESQ/STOI. A breakdown by SNR or distortion level is needed, as real-time constraints often hurt performance most in challenging (e.g., low SNR) conditions. The claimed real-time capability is a key advantage but is not rigorously validated.
4. **Failure case analysis.** The paper shows only successful spectrograms. Presenting examples where the method fails (e.g., introducing artifacts, over-suppression, or under-enhancement) and analyzing why would provide a more honest assessment of limitations and guide future work.

### Visualizations & Case Studies
1. **Time-frequency visualizations comparing input, target, ANC baseline output, and ASE-TM output for all three tasks.** The single denoising spectrogram (Fig 3b) is insufficient. Side-by-side spectrograms for dereverberation and declipping, alongside the corresponding *error signals* (output - target), would visually demonstrate superior noise suppression and signal preservation/amplification compared to baselines.
2. **Waveform plots highlighting "active shaping".** Show short (e.g., 100ms) segments of the waveform for `d(n)`, `a(n)`, and `e_h(n)` for a declarative sentence. This would illustrate whether `a(n)` constructively adds to speech components (e.g., making plosives sharper, vowels stronger) or simply adds a phase-inverted noise estimate.
3. **Perceptual metric (e.g., PESQ) distribution across test samples.** A histogram or violin plot showing the distribution of scores for ASE-TM vs. key baselines would reveal consistency and robustness, rather than just reporting averages which can mask many poor results.

### Obvious Next Steps
1. **Direct comparison to SOTA speech enhancers.** This is the most glaring omission. The authors must train and evaluate at least 2-3 recent, strong passive enhancement models (e.g., a diffusion model, a Conformer-based model) on their exact ASE task setup and dataset to contextualize their contribution.
2. **User study or subjective evaluation (MUSHRA).** For a paper claiming improved "perceptual quality," objective metrics (PESQ, etc.) are not enough for ICLR. A small-scale but proper listening test comparing ASE-TM to a top passive enhancer and the noisy baseline is necessary to validate the perceptual claim.
3. **Unified model experiment.** The paper mentions developing a unified model as future work. A natural and expected next step for this submission is to present a single ASE-TM model trained jointly or in a multi-task manner on all three distortions (noise, reverb, clip) and evaluate its generalizability, which would strongly support the paradigm's versatility.
4. **Experiments with real acoustic paths.** The entire framework is simulated. A critical step to demonstrate practical viability is to test the pre-trained model (or fine-tune it) using data recorded in a real room with a physical speaker and microphone, even if only for a subset of tasks. Without this, the applicability to real "active" systems remains purely theoretical.

# Final Consolidated Review
## Summary
This paper introduces Active Speech Enhancement (ASE), a new paradigm that unifies active noise control and speech enhancement by actively shaping the speech signal—both suppressing interference and amplifying speech-relevant frequencies. The authors propose ASE-TM, a Transformer-Mamba-based architecture with a multi-component loss, and demonstrate improved performance on denoising, dereverberation, and declipping tasks compared to adapted active noise cancellation baselines.

## Strengths
- **Novel paradigm**: ASE formally bridges active noise cancellation and speech enhancement, offering a fresh conceptual framework with potential for new research directions, as detailed in Sections 1 and 3.
- **Strong empirical performance**: ASE-TM consistently outperforms adapted ANC baselines across denoising, dereverberation, and declipping tasks, with significant gains in metrics like PESQ, STOI, and NMSE (Tables 1, 2, 3).
- **Architectural innovation**: The hybrid Transformer-Mamba design effectively integrates Mamba2 blocks and an attention mechanism to capture long-range dependencies, with ablation showing its contribution (Section 4.1, Figure 3a).
- **Robustness under varying conditions**: The model maintains high performance across different reverberation times and loudspeaker nonlinearities, demonstrating reliability in simulated acoustic environments (Table 4).
- **Real-time feasibility**: Runtime analysis with future-frame prediction shows minimal performance degradation while meeting real-time constraints, a key requirement for active systems (Section 6.4).

## Weaknesses
- **Missing comparison to state-of-the-art passive speech enhancement methods** — The paper only compares to ANC baselines adapted to ASE tasks, leaving it unclear whether ASE advances beyond existing enhancement techniques like diffusion or Transformer-based models. This gap undermines the significance of the contribution relative to the field.
- **Inadequate description of baseline adaptations** — The baseline methods (THF-FxLMS, DeepANC, ARN) are stated to be "adapted and retrained or configured" without details, making the comparisons potentially unfair and reducing reproducibility (Section 5.4).
- **Limited ablation study** — Ablation analysis is only provided for denoising, and the contributions of specific components (e.g., loss modifications, attention mechanism) are not quantified for dereverberation and declipping, hindering a full understanding of the model's design choices (Figure 3a).
- **No subjective evaluation** — Reliance on objective metrics without listening tests (e.g., MOS) weakens claims about perceptual quality and intelligibility improvements, which are central to the ASE paradigm.
- **Conceptual clarity** — The ASE formulation, particularly the role of the anti-signal \(a(n)\) in enhancing speech via addition (Eq. 4), could be more clearly explained to avoid ambiguity about how active shaping is achieved mathematically.

## Nice-to-Haves
- Subjective listening tests to validate perceptual improvements beyond objective metrics.
- Experiments with real acoustic paths or recorded data to demonstrate practical viability beyond simulation.
- A unified model trained jointly on multiple tasks to assess generalizability and versatility.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add comparisons to recent state-of-the-art passive speech enhancement models (e.g., diffusion-based or Conformer-based methods) on the same tasks to properly contextualize the performance of ASE.
- Provide detailed descriptions of how baseline methods were adapted to the ASE framework, including any architectural changes or training procedures, to ensure fair and interpretable comparisons.
- Extend ablation studies to all tasks (dereverberation, declipping) and quantify the individual contributions of key components like the loss modifications and attention mechanism.

# Actual Human Scores
Individual reviewer scores: [2.0, 0.0, 2.0]
Average score: 1.3
Binary outcome: Reject
