---

## Summary

Co³Gesture introduces the new task of *concurrent* co-speech gesture generation for two-person conversations, tackling it with two contributions: (1) **GES-Inter**, a pseudo-labeled 3D dataset of 70 hours / 7M validated frames extracted from in-the-wild talk-show videos via PyMAF-X, and (2) a bilateral cooperative diffusion model with a **Temporal Interaction Module (TIM)** and mutual attention mechanism that models asymmetric, interaction-aware gesture dynamics for both speakers simultaneously. The framework outperforms all tested baselines (single-speaker gesture models and text-to-motion models) on FGD, BC, and Diversity metrics evaluated on GES-Inter.

---

## Strengths

- **Genuine task gap and dataset contribution.** As Table 1 documents, no prior dataset provides large-scale mesh-based concurrent co-speech gestures with multi-modal annotations (audio, text, phoneme, face, body). GES-Inter is 4× larger by duration than the only prior concurrent dataset (TWH16.2) and is the only pseudo-mesh one. This is a concrete, verifiable contribution that the community can directly build on.

- **Well-supported bilateral design choice.** The ablation in Table 4 shows a dramatic FGD improvement from single-branch to bilateral (1.669 → 0.769), providing strong empirical justification for the core architectural decision. The removal of mixed audio or audio separation also produces clear degradation, substantiating both the separated-audio conditioning and the interaction-via-mixed-audio design.

- **TIM outperforms MLP fusion with nontrivial margin.** Table 3 shows TIM (FGD 0.769) substantially outperforms a naive MLP replacement (FGD 1.202), providing at least partial evidence that the temporal correlation mechanism does more than generic feature fusion. The ablation granularity (full TIM, w/o TIM, w/ MLP, w/o mutual attention) is informative.

- **Comprehensive comparison set.** The paper evaluates against 7 baselines from two distinct families (single-speaker gesture and text-to-motion interaction), with all retrained or adapted on GES-Inter for fairness, and reports consistent superiority across all three metrics.

---

## Weaknesses

### Fatal
None identified.

### Major

- **No interaction-specific automated metric — the central claim is not directly evaluated.** The paper's core contribution is *coherent concurrent interaction*, yet FGD measures distributional realism, BC measures individual speech-rhythm alignment, and Diversity measures sample spread. None quantifies whether the two generated speakers are coordinated with each other (e.g., reactive gesturing, turn-taking latency, mirroring). The paper acknowledges this gap in Limitations, but for a paper whose title centers on "coherent concurrent" generation, this is a significant evidential hole. The user study's "Interaction Coherency" rating partially compensates but is too thin statistically (see below) to carry the weight.

- **The most natural baseline is absent: two independent single-speaker generators.** The strongest test of whether the bilateral interaction architecture adds value is to run two independent single-speaker models (e.g., EMAGE or DiffSHEG) one per speaker on their separated audio, and compare against Co³Gesture. The paper never does this. Without it, we cannot determine whether the gains in Table 2 stem from the *interaction modeling* or simply from the bilateral/separated-audio architecture per se, which any independent dual-model would replicate.

- **TIM is technically under-specified to the point of impeding reproducibility.** Equation (1) writes a single shared projection matrix **W** for Q (from *f*_{C_a}), K, and V (both from *f*_{x_a}). Standard attention uses three distinct projections; if this is intentional (e.g., a single linear map per modality), it should be stated. More critically, the temporal correlation matrix **M** ∈ ℝ^{N×N} — the central novelty of TIM — is never defined mathematically. The paper says it "represents temporal variants between the current gesture sequences and interactive ones," but does not specify whether it is a dot product, cosine similarity, learned affinity, or another operation. Similarly, the motion encoder Enc whose output is passed through sigmoid to produce σ is never described. Given that TIM is the paper's primary technical contribution, these omissions are substantive.

- **Foot contact loss applied to T-posed lower body yields an implausibly large gain.** The paper explicitly states: "we only model the upper body joints in experiments, we complete the lower body joints as T-pose in forward kinematic function during calculate loss" (Section 4.2). Yet removing this loss degrades FGD from 0.769 to 1.082 — a 40% increase. For a loss computed over fixed, artificially-posed joints that are never generated or evaluated, this gain is very hard to interpret as evidence of "physical reasonableness." The most likely explanation is that the loss incidentally regularizes global body orientation or root trajectory, but the paper provides no analysis. Until this is explained, the loss and its contribution should be treated with skepticism.

- **Evaluation limited to the authors' own dataset.** All quantitative results are on GES-Inter. Since the dataset was designed and curated by the same group, and the model was trained and tuned on it, there is no way to assess whether Co³Gesture generalizes beyond GES-Inter's talk-show domain. TWH16.2 exists as a concurrent gesture benchmark and would serve as a natural external evaluation set.

### Minor

- **Weight-sharing justification is intuitive but unvalidated.** Section 3.3 motivates shared denoiser weights by arguing that "exchanging the input order of the speaker's audio results in an invariance effect." This holds for symmetric interactions (dancing, sports), but is far less obvious for conversational gestures between a host and guest, where social roles, dominance, and listener behavior systematically differ. No ablation tests shared vs. unshared branch weights, and no dataset statistics (e.g., motion energy conditioned on diarized speaking state) are offered to empirically validate the symmetry claim.

- **Dataset quality control is insufficiently documented in the main paper.** For a dataset-centric contribution, the main paper should provide at minimum: filtering criteria, percentage of frames rejected at each processing stage, pose extractor confidence thresholds, and failure-mode examples. The paper says these details are in supplementary, which is insufficient for a dataset contribution.

- **Baseline adaptation transparency.** The paper adapts text-to-motion models (MDM, InterGen, InterX) by adding the same audio encoder used by Co³Gesture. While this grants the baselines richer conditioning (benefiting baselines), it is unclear how well these models were tuned under their new conditioning regime. The paper should clarify how many epochs these baselines were trained and whether their hyperparameters were re-tuned.

- **Audio separation failure modes are unaddressed.** The entire pipeline relies on pyannote-audio for speaker separation. The paper provides no analysis of separation error rates, behavior under overlapping speech or laughter, or downstream impact of incorrect speaker-audio assignments on gesture generation quality.

- **User study is statistically thin.** 15 volunteers rating 2 videos per method, with no significance testing reported. Mean ratings alone are insufficient to support the claims made. The study is directionally useful but cannot carry strong conclusions.

### Tiny

- Section 3.2 formulates the condition as only *C_mix*, but Section 3.3 clarifies the full conditioning is *(C_a, C_b, C_mix)*. The problem formulation should state this upfront.
- Loss weights for *L_vel* and *L_foot* in Equation (4) are not specified (implicit 1.0). Should be stated explicitly for reproducibility.
- Describing 90-frame (6-second) clips as enabling "long sequence gestures" is mild overreach; this is a standard clip length.

---

## Nice-to-Haves

- **Interaction coherence metric**: Cross-correlation of motion features between speakers, reaction-time alignment, or turn-taking synchrony scores would directly evaluate the paper's core claim.
- **External evaluation**: Testing on TWH16.2 (even as a zero-shot transfer) would establish generalization beyond GES-Inter.
- **Shared vs. unshared branch weights ablation**: Directly tests the symmetry assumption motivating weight sharing.
- **Dataset statistics**: Distribution of speaking-state frames (active, listening, overlap), pose confidence histograms, and rejection rates at each processing stage would strengthen dataset credibility.
- **Turn-taking vs. simultaneous speech qualitative analysis**: Showing how the model handles distinct conversational dynamics (one speaker active / one passive, both speaking, transitions) would illuminate whether TIM captures genuine interaction or temporally plausible but interactionally shallow outputs.
- **Inference cost comparison**: Brief FLOPs or latency comparison with single-branch baselines to assess practical overhead.

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **"Equation (1) is possibly incorrect"** (Harsh Critic): Q uses *f*_{C_a} as input while K/V use *f*_{x_a}; the inputs are different even if the projection label is shared. This is likely a notational simplification of distinct projections rather than a factual error. Retained as a minor notation concern only.
- **Lack of statistical significance for FGD/BC** (Harsh Critic): Single-run evaluation of FGD and BC is standard practice in co-speech gesture papers; demanding confidence intervals on these is not the community norm. Removed as a standalone weakness.
- **Sequence length "not especially long"** (Harsh Critic): The paper's contribution is the *task and architecture*, not the generation of arbitrarily long sequences. Criticizing the fixed 6-second clip length is scope creep. Removed.
- **Missing ethical/licensing discussion** (Harsh Critic): While dataset transparency is good practice, the absence of a full legal analysis is not a technical weakness specific to this paper's contributions. Removed as a standalone weakness.
- **Related work "too broad"** (Harsh Critic): This is a style/formatting critique with no substantive bearing on the contributions. Removed.
- **Requirement for theoretical proofs of gating mechanism** (Harsh Critic): This is an empirical systems paper; demanding theoretical motivation for the gating form goes beyond community expectations. Moved to nice-to-have.
- **"FGD cannot be reliable on pseudo-labeled data"** (Harsh Critic): FGD measures distributional alignment with real data (also pseudo-labeled); this is the standard and accepted evaluation protocol for this domain. Removed.
- **Criticism that the comparison with text-to-motion methods is unfair because they receive the same audio encoder as Co³Gesture** (Harsh Critic): Giving baselines Co³Gesture's audio encoder is asymmetric in the *baselines'* favor, not the authors', making the comparisons more conservative. Removed.
- **"Abstract claims 'coherent' and 'vivid' without proving it"** (Harsh Critic): These are adjective choices in a paper abstract, and the human study does evaluate coherency. Removed as a standalone critique.

---

## Novel Insights

The synthesis of the reviews surfaces one genuinely actionable observation beyond the paper's own analysis: the foot contact loss result (Table 5) is both the most surprising ablation finding and the least explained. If the gain is real and robust, it suggests that global body-coherence constraints applied even to synthetic/imputed lower-body joints can act as useful regularizers for upper-body diffusion — a potentially generalizable insight for upper-body-only gesture models. However, this requires careful mechanistic analysis to distinguish principled regularization from a confounding artifact. This is a thread worth pulling explicitly rather than leaving as an unexplained positive result.

---

## Suggestions

1. **Add a "two independent single-speaker generators" baseline** (e.g., EMAGE × 2, each on separated audio). This is the single most important missing experiment — it directly tests whether interaction modeling adds value over simply running two separate models.

2. **Mathematically define the temporal correlation matrix M in TIM.** State the exact operation (e.g., normalized dot product between frame-level embeddings of *f*_{x_a, C_a} and *f*_{x_a, C_mix}), the shape transformations, and what Enc consists of (layer type, depth, output dimension).

3. **Explain or re-examine the foot contact loss behavior.** Either provide an ablation showing *what* the loss constrains (e.g., add visualization of root motion with vs. without it) or reframe its interpretation. The current "physical reasonableness" framing is not credible for imputed T-pose lower body.

4. **Propose at least one automated interaction metric.** For example: cross-correlation of per-frame motion energy between the two speakers (to capture responsiveness), or gesture-beat temporal offset between active-speaker and passive-speaker motions. Even an imperfect metric would ground the central claim in quantitative evidence.

5. **Add an ablation for shared vs. unshared branch weights.** This tests the symmetry assumption directly and would either validate the design or reveal a currently unexplored performance gap.

6. **Report dataset quality statistics in the main paper.** Include at minimum: total raw frames, frames rejected at each filtering step, and (ideally) a small-scale human audit of pose quality, so that downstream users can assess the dataset's reliability.

---

**Overall evaluation:** Co³Gesture addresses a real and underexplored problem, and the dataset contribution alone has clear community value. The empirical gains are consistent and the ablations support the core design choices. However, the paper's central evaluation gap — no automatic metric for the claimed "interaction coherence" — paired with the absence of the most natural baseline (independent per-speaker models), the under-specification of TIM, and the unexplained foot-contact-loss behavior collectively prevent a clean case for acceptance in the current form. The paper is at a level where targeted revisions to the evaluation and technical exposition could make it a strong contribution, but as submitted it falls short of a convincingly substantiated ICLR paper.

| Axis | Assessment |
|---|---|
| Novelty | High — new task definition and dataset; moderate architectural novelty |
| Technical Soundness | Moderate — TIM under-specified; foot loss unexplained; weight-sharing assumption untested |
| Empirical Support | Moderate — strong on own benchmark, absent externally; core claim lacks direct metric |
| Significance | Substantial — dataset and task framing will benefit the community |
| Clarity | Moderate — high-level story is accessible; method section has reproducibility gaps |