=== CALIBRATION EXAMPLE 38 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title accuracy:** The title accurately reflects the paper’s core contribution: adapting Long-Term Action Anticipation (LTA) to a transcript-only supervision setting.
- **Abstract clarity:** The abstract clearly outlines the problem (reliance on dense frame-level labels), the proposed method (TbLTA using temporal alignment for pseudo-labels, cross-modal attention, and global CTC supervision), and the evaluation setup (3 datasets). 
- **Unsupported claims:** The abstract claims transcript supervision offers a "very robust and less costly alternative." While the cost reduction is intuitively true (transcripts are cheaper than dense labels), it is never quantified in the paper. Furthermore, "robust" is slightly overstated given that supervised methods still outperform TbLTA on several horizons and datasets. The "first weakly-supervised approach for LTA" claim is valid relative to prior work that still required observed-segment labels (Zhang et al., 2021), but should be carefully scoped to "transcript-only" to avoid confusion with existing semi-weak paradigms.

### Introduction & Motivation
- **Motivation & gap:** The motivation is strong and well-grounded in the literature. Dense frame-level annotation costs for long, fine-grained videos are a recognized bottleneck. The gap—LTA lacking a pure weak-supervation paradigm—is clearly identified, and the positioning relative to Zhang et al. (2021) is accurate.
- **Contributions:** The five contributions are clearly listed. However, several architectural components are adapted from prior work rather than proposed from scratch: ATBA for alignment (Xu & Zheng, 2024), encoder design (standard temporal transformers), decoder structure (adapted from FUTR/ANTICIPATOR), and CRF refinement (Maté & Dimiccoli, 2024). The true novelty lies in the *integration* and *re-purposing* of these modules for transcript-only LTA. The introduction should explicitly distinguish between novel architectural contributions and synergistic combinations of existing components to avoid overclaiming.
- **Claim calibration:** The introduction does not overtly undersell or overclaim, but it would benefit from a clearer statement on *why* transcripts are particularly well-suited for LTA beyond "semantic abstraction" (e.g., procedural videos exhibit strong verb/noun co-occurrence and ordering priors that transcripts capture).

### Method / Approach
- **Clarity & reproducibility:** The high-level architecture is coherent. The progressive training schedule is well described, which aids reproducibility. However, several implementation details necessary for reproduction are missing or underspecified (e.g., exact definition of the binary mask $M$ in Eq. 1, neighborhood size for temporal restriction, optimizer/LR specifics for the progressive stages).
- **Key assumptions:** The method assumes the transcript perfectly matches the video's action sequence in order and vocabulary. It does not discuss handling missing steps, spurious actions in the video, or transcript errors—common in real-world instructional videos. This is a strong but somewhat standard assumption for weakly-supervised segmentation/anticipation.
- **Logical gaps & methodological concerns:**
  1. **Future pseudo-label generation:** Section 3.1 states ATBA partitions $Y$ into $Y_{obs}$ and $Y_{future}$ and produces pseudo-labels $\hat{Y}$ for the *full* video. However, ATBA (Xu & Zheng, 2024) is designed for temporal action *segmentation*, where it aligns features to a known transcript. In the LTA setting, the model must generate pseudo-labels for the *unseen* future interval during training. It is unclear how the alignment module resolves temporal ambiguity in $X_{pred}$ without ground-truth boundaries. If pseudo-labels for the future rely on global video context during training, how does this translate to inference when only $X_{obs}$ is available? This training-inference discrepancy needs explicit clarification.
  2. **Circular duration learning:** The affinity-based duration loss (Eq. 7) uses a momentum-based buffer $\hat{d}$ to store class-wise duration estimates derived from the model's own pseudo-labels. This creates a self-reinforcing loop: inaccurate pseudo-labels corrupt the buffer, which then biases future duration predictions. The paper does not discuss how this avoids compounding errors or whether the buffer is initialized/stabilized in any way.
  3. **Cross-attention masking:** The mask $M$ restricts each action embedding to a "temporal neighborhood around its predicted occurrence," but the radius/definition of this neighborhood is unspecified. Is it fixed? Learned? Based on pseudo-label boundaries? This choice significantly impacts how much semantic grounding the model actually leverages versus attending broadly.
- **Edge cases:** The method does not address failure modes where actions occur out-of-order relative to the transcript, or where multiple valid future paths exist. The stochastic decoder attempts to handle this, but the training supervision remains deterministic via pseudo-labels, which may limit the diversity of plausible futures.

### Experiments & Results
- **Claims testing:** The experiments directly evaluate the main claim: can transcript-only supervision yield competitive LTA performance? Yes, the setup aligns with the claim.
- **Baselines & fairness:** Baselines are appropriate and representative (supervised: FUTR, ActFusion; weak: Zhang et al., 2021). Following established protocols ($\alpha/\beta$ splits, MoC/mAP metrics) ensures fair comparison. However, the comparison between weakly-supervised and fully-supervised models is inherently asymmetric. While TbLTA's performance is impressive for its supervision level, framing it as consistently "competitive with, and occasionally superior to" supervised methods is slightly misleading on 50Salads and EGTEA, where supervised models maintain clear advantages.
- **Missing ablations:** 
  - **Pseudo-label quality:** No analysis is provided on how pseudo-label accuracy evolves during training or across observation horizons. This is crucial given the model's heavy reliance on them for LTA supervision.
  - **ATBA vs. naive alignment:** It's unclear if ATBA's specific alignment mechanism is critical, or if simpler CTC-only or DTW-based alignment would yield similar results.
  - **Duration inconsistency:** Table 4 shows that removing the duration loss (`w/o duration`) actually *improves* performance on 50Salads (Obs 30%, 10% horizon: 38.2 vs. 34.5) and performs comparably elsewhere. The paper claims duration prediction benefits models, yet the ablation contradicts this for key settings. This discrepancy requires analysis (e.g., is the momentum buffer destabilizing training on highly variable datasets like 50Salads?).
- **Statistical reporting:** No error bars, confidence intervals, or significance tests are reported. For ICLR, especially when claiming competitive or superior performance against strong baselines, reporting variance across random seeds/splits is expected.
- **Datasets & metrics:** Standard and appropriate. EGTEA evaluation focuses only on verb prediction, which should be explicitly justified given the transcript includes noun/verb compositions.

### Writing & Clarity
- **Confusing sections:** 
  - Section 3.2.3 (Anticipation-Oriented Losses): The CRF formulation equations are mathematically standard, but the text does not clarify how $Y_{LTA}$ is constructed for supervision during training. Are ground-truth transcripts truncated to the future window? Are they aligned to pseudo-labels? This gap obscures how the CRF actually learns transition constraints under weak supervision.
  - The relationship between the encoder processing the *full* video during training and the decoder operating only on $X_{obs}$ at inference is stated but not mechanistically explained. Does the decoder see leaked future information during training via the encoder's full-sequence context? If so, how severe is the train-test mismatch?
- **Figures & Tables:** Despite parser-induced formatting breaks, Figure 1 and the architectural overview conceptually map to the text. Table 1 is clear. Table 2 (EGTEA results) and Tables 3/4 (ablations) are referenced in the text but appear fragmented or missing in the provided extract. I am evaluating based on the text discussion, but the authors should ensure all referenced tables are complete in the final version.

### Limitations & Broader Impact
- **Acknowledged limitations:** The authors correctly identify duration estimation as a major remaining challenge and note that fully supervised methods still dominate in absolute performance.
- **Missed limitations:** 
  - **Transcript fidelity dependency:** The entire pipeline assumes high-quality, accurate transcripts. Transcript noise, vocabulary mismatch, or coarse granularity would severely degrade pseudo-label quality and downstream anticipation.
  - **Non-procedural domains:** The method relies on strong action-order priors implicit in instructional datasets (Breakfast, 50Salads). It may not generalize to domains with high action variability, concurrent actions, or non-linear task structures (e.g., sports, unstructured daily life).
  - **Compute overhead:** The use of ATBA, cross-modal attention, CRF refinement, and progressive multi-stage training increases computational complexity compared to simpler supervised pipelines. No runtime or FLOP analysis is provided.
- **Broader impact / failure modes:** Not discussed, which is acceptable given the applied nature of the work. However, potential failure in safety-critical anticipation (e.g., robotics, autonomous driving) if duration estimates are unreliable or if transcripts contain hallucinations should be noted.

### Overall Assessment
This paper addresses a meaningful and underexplored direction: weakly-supervised LTA using only transcripts. The core idea—leveraging transcript alignment for pseudo-label generation, reinforced by CTC global consistency and cross-modal grounding—is well-motivated and shows promising empirical results, particularly establishing a strong new baseline for transcript-only supervision. However, the paper currently suffers from several methodological ambiguities that impact reproducibility and technical confidence: (1) how future pseudo-labels are reliably generated without boundary annotations in the prediction interval, (2) the potentially self-reinforcing nature of the duration momentum buffer, (3) unclear masking definitions in the cross-attention module, and (4) the lack of variance reporting despite competitive performance claims. The ablation studies also present contradictory evidence regarding the utility of the duration loss that is not analyzed. If the authors can clarify the pseudo-label alignment mechanism for future windows, provide statistical error bars, rigorously analyze the duration loss behavior, and better scope architectural novelty versus component integration, this work could meet ICLR's standards for empirical rigor and novelty. As it stands, it is a solid foundation that requires deeper technical clarification and empirical validation to justify acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
TbLTA introduces the first weakly-supervised framework for dense Long-Term Action Anticipation (LTA), relying exclusively on video transcripts (ordered action lists without temporal boundaries) rather than costly frame-level annotations. The method integrates a temporal alignment module to generate pseudo-labels, a cross-modal attention mechanism for semantic grounding, and a composite loss (CTC, CRF, and affinity-based duration) to jointly train a segmentation head and an anticipation decoder. Experiments across Breakfast, 50Salads, and EGTEA demonstrate that transcript-only supervision can achieve performance competitive with fully supervised methods, establishing a scalable new baseline for long-horizon video forecasting.

### Strengths
1. **Addresses a clear scalability bottleneck:** The shift from dense frame-level labels to transcript-level supervision directly targets the annotation cost that limits real-world LTA deployment. The motivation that transcripts capture high-level narrative structure is well-aligned with the sequential nature of procedural tasks.
2. **Cohesive and well-motivated architecture:** The pipeline logically combines pseudo-label generation (via ATBA), global sequence consistency (CTC loss), and structured decoding (CRF + self-supervised duration prior). Ablation results consistently validate each component's utility, e.g., removing CTC causes ~0.6-0.8 point average drops, confirming its role in stabilizing pseudo-label alignment.
3. **Strong empirical performance and benchmarking:** TbLTA is rigorously evaluated on three standard LTA datasets and frequently matches or exceeds recent fully supervised baselines. Notably, it outperforms all supervised methods at 30% observation on Breakfast, demonstrating that high-level semantic guidance can compensate for the lack of fine-grained temporal labels.

### Weaknesses
1. **Ambiguous inference protocol:** The training pipeline explicitly leverages the full video transcript for alignment, cross-modal attention, and global supervision. However, the text states that at inference, only observed features and class tokens are provided (`[E ∥ X_obs]`), making it unclear how the decoder determines the prediction horizon or handles variable-length futures without transcript context. This gap complicates fair comparison and practical deployment.
2. **Missing comparisons with modern foundation models:** While the related work acknowledges vision-language and LLM-based anticipation methods (e.g., AntGPT, PALM), these are absent from the empirical tables. Given the model's reliance on pretrained text embeddings (DistilBERT), contrasting against recent LLM/VLM baselines is necessary to position the proposed architectural design.
3. **Insufficient reproducibility details:** Key training hyperparameters are omitted, including loss weights (`γ1, γ2, γ3`), optimizer configuration, learning rate schedules, batch sizes, and total compute budget. The progressive training scheme mentions re-initializing optimizer states but lacks precise epoch counts for the final joint optimization stage, falling short of ICLR's reproducibility expectations.
4. **Limited statistical reporting and failure analysis:** Results in Tables 1 and 2 report point estimates without variance, standard deviations, or significance testing across splits. Additionally, while duration estimation is acknowledged as a challenge, there is no quantitative evaluation (e.g., MAE on predicted durations) or systematic analysis of failure modes (e.g., high transcript-video misalignment in 50Salads).

### Novelty & Significance
The paper achieves notable novelty by formally defining and tackling dense LTA under a strictly transcript-only supervision regime, a setting that had remained largely unexplored. The significance is high: it demonstrates that annotation-light, language-informed supervision can yield forecasting quality competitive with fully supervised pipelines, offering a practical pathway for scaling LTA to new domains. Methodologically, the work synthesizes established techniques (CTC, CRF, weak alignment) into a unified framework tailored to this new task. While individual components are not wholly new, their integration and empirical validation represent a solid, impactful contribution to weakly-supervised video understanding.

### Suggestions for Improvement
1. **Clarify the test-time pipeline:** Explicitly explain how the anticipation horizon is determined during inference without the transcript. If parallel queries or an `<EOS>` termination mechanism is used, detail how this adapts to the variable `β` evaluation protocol and discuss potential error accumulation versus ground-truth boundary availability.
2. **Strengthen baselines and statistical rigor:** Add recent LLM/VLM-based LTA methods (e.g., AntGPT from ICLR 2024) to the comparison tables to contextualize the proposed cross-modal design. Report standard deviations across dataset splits and, if possible, perform significance testing (e.g., paired t-tests) to substantiate performance claims.
3. **Fill reproducibility gaps:** Provide exact loss weightings (`γ_i`), optimizer details, learning rate schedules, batch sizes, and hardware/compute requirements in an appendix. Clarify the CRF transition matrix initialization and training dynamics, as the current formulation and equations need tighter exposition.
4. **Add quantitative duration and error analysis:** Report a standard metric for duration prediction (e.g., mean absolute error or IoU with ground-truth boundaries) across datasets. Include a dedicated discussion or figure analyzing cases where transcript-video misalignment degrades anticipation, providing actionable insights for future weakly-supervised LTA research.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Report frame-level alignment accuracy of the ATBA-generated pseudo-labels against held-out ground-truth boundaries; without this, the core claim that transcripts provide a reliable training signal is unverified and the entire weak-supervision pipeline rests on an unproven assumption.
2. Benchmark against a strong transcript-only baseline (e.g., pure CTC encoder paired with a standard autoregressive decoder) to demonstrate that TbLTA's architectural complexity is necessary rather than merely inheriting gains from the borrowed alignment module.
3. Re-run evaluations using modern video-language features (e.g., CLIP, SigLIP, or VideoMAE) instead of frozen I3D, because outdated visual backbones artificially depress the supervised ceiling and invalidate claims of parity with contemporary fully supervised LTA methods.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify error propagation by correlating pseudo-label boundary noise with anticipation accuracy drops across increasing observation horizons; without this, it is impossible to distinguish whether performance degrades due to alignment drift or inherent long-horizon uncertainty.
2. Analyze performance on videos containing reversible, parallel, or highly repeated action steps to prove the method can handle transcript ordering ambiguity, which is a primary limitation of timestamp-free supervision.
3. Ablate the three-stage progressive training schedule by comparing it to direct end-to-end optimization, ensuring the reported gains are not solely artifacts of careful initialization preventing early pseudo-label collapse.

### Visualizations & Case Studies
1. Provide side-by-side alignment heatmaps of pseudo-labels vs ground-truth for low-density and high-transition videos to visually verify the alignment module captures procedural structure instead of collapsing to uniform or trivial segments.
2. Visualize cross-modal attention weights over video frames and transcript terms to confirm the model genuinely grounds visual features on semantic cues rather than attending to high-activation visual modes or spatial priors.
3. Show explicit failure trajectories where early anticipation mistakes compound into catastrophic drift, clarifying whether breakdowns originate from transcript misalignment, duration miscalibration, or inherent task stochasticity.

### Obvious Next Steps
1. Remove or fundamentally replace the self-supervised duration loss that regresses predictions against a class-mean momentum buffer; it cannot capture instance-level variability and currently adds unvalidated complexity while being explicitly acknowledged as ineffective.
2. Clearly separate deterministic and stochastic evaluation protocols into distinct tables with reported variance, as conflating them obscures whether transcript supervision genuinely improves robustness or merely inflates prediction entropy.
3. Explicitly document and ablate the future-masking strategy used during full-video pseudo-label generation to guarantee no temporal leakage contaminates the observed representation learning, which is critical for ICLR reproducibility standards.

# Final Consolidated Review
## Summary
TbLTA introduces the first purely transcript-only supervision framework for dense Long-Term Action Anticipation (LTA), replacing costly frame-level annotations with ordered action lists. By integrating a weak temporal alignment module for pseudo-label generation, CTC consistency losses, cross-modal semantic grounding, and a CRF-refined decoder, the method establishes a new weakly-supervised baseline and reports performance competitive with fully supervised approaches across three standard procedural video benchmarks.

## Strengths
- **Directly addresses a critical scalability bottleneck:** Shifting LTA training from dense boundary labels to high-level transcript supervision is a practical and well-motivated paradigm shift that aligns naturally with the procedural narrative structure of instructional videos.
- **Strong empirical validation on standard benchmarks:** The model consistently outperforms prior weakly-supervised baselines and frequently matches or exceeds fully supervised methods, successfully proving that language-informed ordinal constraints can compensate for the absence of fine-grained temporal labels.

## Weaknesses
- **Unvalidated pseudo-label supervision and self-reinforcing duration objective:** The entire pipeline's viability hinges on the quality of ATBA-generated pseudo-labels, yet the paper provides zero frame-level alignment metrics to verify their reliability. This is compounded by the affinity-based duration loss (Eq. 7), which regresses predictions against a momentum buffer derived from the model's own pseudo-labels, creating a closed feedback loop prone to compounding errors. Critically, the ablation study itself demonstrates that removing this component *improves* performance on key 50Salads splits (e.g., +3.7 MoC at 30% observation, 10% horizon), an adverse effect the authors completely ignore.
- **Train-test context mismatch and unquantified future leakage:** The encoder is explicitly trained over the *complete* video sequence to "acquire a comprehensive representation of the future’s temporal structure," yet inference is strictly limited to the observed fraction `X_obs`. While this follows certain prior protocols, the paper lacks any mechanism (e.g., future masking, explicit boundary estimation analysis, or distribution-shift ablation) to bridge this gap or prove the decoder is learning genuine anticipation rather than over-relying on leaked future context during training.
- **Lack of statistical rigor and reproducibility transparency:** All empirical claims rest on point estimates without reported variance, standard deviations, or significance tests across dataset splits, making it impossible to assess the stability of the weak supervision signal. Furthermore, critical training specifications—including loss weighting coefficients ($\gamma_1, \gamma_2, \gamma_3$), optimizer hyperparameters, learning rate schedules for the progressive stages, and compute budgets—are omitted, rendering independent reproduction infeasible.

## Nice-to-Haves
- Provide an ablation directly comparing the 3-stage progressive training schedule against standard end-to-end optimization to isolate the impact of initialization on pseudo-label collapse.
- Include a runtime/FLOPs analysis or parameter budget comparison against fully supervised pipelines to quantify the true cost-benefit tradeoff of the proposed multi-module architecture.
- Explicitly document how the binary mask $M$ in the cross-modal attention (Eq. 1) defines its temporal neighborhood radius (fixed, learned, or pseudo-boundary dependent) to improve component transparency.

## Novel Insights
The work effectively demonstrates that high-level narrative constraints, when rigorously aligned to visual features and enforced through global sequence losses (CTC), can partially substitute for granular temporal supervision in dense anticipation. By shifting the learning objective from memorizing low-level frame transitions to reasoning over verb/noun co-occurrence and procedural ordering, the model learns to ground predictions in semantic structure rather than brittle boundary cues. This highlights a promising trajectory for weakly-supervised video understanding where language-informed priors guide temporal reasoning, provided the inevitable pseudo-label noise and train/test distribution shifts are carefully managed.

## Potentially Missed Related Work
- Recent vision-language and LLM-based anticipation frameworks (e.g., AntGPT, PALM) — These are cited in the related work but entirely absent from empirical comparisons. Contrasting against them would contextualize whether explicit architectural cross-modality or simple prompting with modern foundation models yields more robust transcript grounding.
- Modern visual backbones (e.g., SigLIP, VideoMAE) — The reliance on frozen I3D features may artificially depress the supervised baseline ceiling; evaluating with contemporary features would better isolate the true contribution of the transcript supervision strategy.

## Suggestions
- Replace or fundamentally redesign the momentum-based duration loss with an explicit, non-circular supervision signal (e.g., relative duration ranking or distribution matching) and report the impact.
- Mandate reporting of frame-level alignment accuracy (IoU or boundary F1) for the ATBA module alongside anticipation metrics, and include variance/standard deviations across all dataset splits.
- Provide complete training hyperparameters, loss weight schedules, and a clear architectural or empirical analysis (e.g., masking study) demonstrating how the model handles the removal of future context at inference without performance degradation.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 4.0]
Average score: 3.0
Binary outcome: Reject
