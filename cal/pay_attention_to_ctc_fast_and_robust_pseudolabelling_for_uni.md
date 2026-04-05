=== CALIBRATION EXAMPLE 87 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate: the paper is indeed about a CTC-centric change to pseudo-labelling for unified speech recognition. “Pay attention to CTC” is a bit catchy, but it does convey the main idea.
- The abstract clearly states the problem: USR’s autoregressive pseudo-labelling is expensive and brittle under distribution shift. It also states the proposed solution: CTC-driven teacher forcing plus mixed sampling.
- The main concern is that the abstract makes several strong claims that are only partially substantiated in the paper: “halves training time,” “improves robustness,” and “achieves state-of-the-art results” are all plausible from the results, but the paper should be careful about the exact scope. In particular, the state-of-the-art claim depends on comparisons across heterogeneous setups, and the “halves training time” claim is not presented with a fully standardized compute comparison across all baselines.
- The abstract also implies the method is generally better for unified speech recognition, but the paper shows a more nuanced picture: much of the gain appears under OOD / long-utterance conditions and in low-resource semi-supervised settings, while in-distribution improvements are smaller and in some settings rely on more epochs or larger scale.

### Introduction & Motivation
- The problem is well motivated. The introduction correctly identifies a real bottleneck in USR: iterative autoregressive pseudo-labelling is expensive, and decoupled supervision can amplify errors under distribution shift.
- The gap in prior work is reasonably clear: existing joint CTC-attention systems use beam search or AR decoding in a way that is not ideal for repeated self-training, and CTC-only approaches lose sequence modelling quality.
- The contributions are stated clearly and mostly accurately: CTC-driven teacher forcing, mixed sampling, and improved robustness/efficiency. This is a strong and relevant contribution for ICLR’s standards because it is both methodologically motivated and experimentally grounded.
- The introduction is somewhat ambitious in claiming broad applicability and “state-of-the-art” across ASR/VSR/AVSR. The paper does support strong performance, but the strongest claims are confined to specific benchmarks and training regimes. It would benefit from more precise delimitation of where the gains come from: pseudo-labelling efficiency, OOD robustness, or scaling to much larger unlabelled corpora.

### Method / Approach
- The method is clearly described overall, and the main equations are understandable: Eq. (1)–(6) define the teacher/student pseudo-labelling process, CTC-driven teacher forcing, and the two training modes.
- The core idea is novel and interesting: use greedily decoded CTC outputs as decoder inputs to produce attention pseudo-labels without AR decoding. This is a sensible ICLR-level idea because it is a concrete algorithmic change with empirical consequences.
- There is, however, an important conceptual question around the “global coherence is unnecessary” argument in Section 4.1. The paper argues that sequence-level incoherence of the teacher’s CTC-conditioned outputs does not matter because student and teacher are conditioned on the same inputs. This is plausible, but the explanation is not fully rigorous. It would help to more formally clarify why the student can learn useful autoregressive inference behavior from teacher outputs that are not themselves globally coherent. The appendix acknowledges sequence-level inconsistencies (C.4), which makes this a real issue rather than a hypothetical one.
- Another concern is the train-test mismatch introduced by CTC-driven teacher forcing. Mixed sampling is a reasonable remedy, but the justification is empirical rather than principled. Why 0.5? The appendix suggests other schedules do similarly, but the paper still does not give a strong theoretical or diagnostic explanation for when this mismatch is benign versus harmful.
- The proposed loss definitions in Eq. (5) and Eq. (6) are clear in spirit, but the paper should more explicitly explain how the decoder can be trained on both CTC and attention pseudo-labels “in a single forward pass” without confusion about length alignment. The alignment argument is central to the method, and although Section 4.1 states it, the operational details would benefit from a clearer algorithm box.
- Potential failure modes are partly discussed in Appendix C.4, including repeated or malformed outputs from CTC-conditioned prefixes. This is good, but the main paper should more explicitly acknowledge that the method relies on CTC prefixes being sufficiently meaningful; if CTC is poor, the decoder targets can inherit error patterns that are different from standard AR teacher forcing.
- For a theoretical claim, the paper does not make a formal theorem, so completeness is not expected. Still, the key reasoning step—that incoherent teacher sequences do not hurt because teacher and student share the same prefix conditioning—remains more intuitive than proven.

### Experiments & Results
- The experiments are largely aligned with the claims. The paper evaluates exactly the settings it cares about: long utterances, noise, OOD datasets, in-distribution benchmarks, ablations, and training efficiency.
- The baselines are mostly appropriate. Comparing against USR, AV-HuBERT, BRAVEn, and self-supervised multimodal baselines is reasonable. The inclusion of prior semi-supervised and supervised state-of-the-art models in the appendix strengthens the paper.
- That said, some comparisons mix substantially different training setups, data scales, and supervision sources. The appendix tables make this transparent, but the main text sometimes risks implying a more uniform comparison than is warranted. The state-of-the-art claim is strongest when interpreted as “best among comparable semi-/self-supervised unified models under these data regimes,” not as an unconditional benchmark domination.
- The main empirical claim that the method is both faster and more robust is well supported. Figure 1, Figure 3, Table 1, and Table 3 all point in the same direction.
- The ablations are useful and materially informative. Table 4 and Table 10 show that both CTC and attention supervision matter, and that the mixed sampling probability trades off robustness and in-distribution accuracy. This is exactly the kind of ablation ICLR reviewers expect.
- A notable missing ablation is a more direct breakdown of where the training-time savings come from. The paper says training is about 2× faster due to faster steps and fewer epochs, but it would be valuable to separate these two effects quantitatively and to compare wall-clock compute at matched performance.
- Another important omission is statistical reporting for the main results. Some appendix tables report mean ± std over three seeds, but many headline comparisons in the main paper are single numbers. Given the modest differences in some in-distribution results, error bars or confidence intervals would strengthen the claims.
- The OOD evaluations are compelling, but some use automatically transcribed data from Whisper as evaluation labels. That is acceptable as a proxy in this domain, but it introduces label noise. The paper mentions this, yet the extent to which noisy reference transcripts affect the reported WERs could be more explicitly discussed.
- The “samples with >100 frames” and length-bucket analyses are well chosen and directly test the hypothesis about long-sequence brittleness. This is a strong experimental design choice.
- The results support the central claims, though the paper should be more careful not to overstate universal gains: in-distribution performance improvements are sometimes incremental, and the large OOD gains are the more distinctive contribution.

### Writing & Clarity
- Overall the paper is readable and logically organized, with a clear progression from motivation to method to experiments.
- The main sections are mostly clear, but there are a few places where understanding the contribution requires cross-referencing appendix material. In particular, the details of how USR 2.0’s two modes interact with the student losses and filtering are easier to follow in the appendix than in the main method section.
- Figure 2 is conceptually important, but the parser has garbled parts of it in the provided text. Ignoring the OCR artifacts, the intended figure seems to illustrate the workflow well. Still, the paper depends heavily on this figure, so in the final PDF it must be very clear for the method to be easily understood.
- The explanation of why CTC-driven teacher forcing should help is reasonable, but Section 4.1 would benefit from a more compact algorithmic summary. At the moment, the conceptual description is strong but the operational sequence of steps is somewhat spread across text and equations.
- The appendix is detailed and helpful, especially the qualitative failure cases and the discussion of sequence-level inconsistencies. This materially improves clarity.
- One clarity issue that matters scientifically is the distinction between “CTC pseudo-labels” at frame-level versus collapsed token-level pseudo-labels in USR 2.0. The paper explains it, but because this distinction is central to the method and ablations, it should be made even more explicit in the main method section.

### Limitations & Broader Impact
- The paper does include a limitations section, which is good and aligned with ICLR expectations.
- The acknowledged limitations are relevant: the method is still slower than fully supervised fine-tuning approaches, unlabelled data quality remains a bottleneck, and CTC-driven teacher forcing is mainly justified in iterative self-training rather than inference.
- However, one key limitation is underemphasized: the method depends on a CTC branch that is already reasonably competent. If the CTC output is highly unstable, the decoder pseudo-labels can inherit systematic errors. The appendix examples show this can happen, but the main paper could more directly frame this as a core limitation.
- Another limitation is that the method’s gains are tied to speech data where CTC’s monotonic alignment is a natural fit. The broader applicability claims to handwriting or biological sequence tasks are plausible but speculative; they are not demonstrated and may require stronger assumptions than the paper acknowledges.
- The broader impact section is balanced and appropriately notes privacy and surveillance concerns. It would be useful to say a bit more about demographic bias and accent/domain disparities, since the system is explicitly meant to improve robustness across diverse inputs; robustness claims should not obscure potential subgroup harms.

### Overall Assessment
This is a strong and well-motivated ICLR submission with a genuinely interesting algorithmic idea: replacing expensive autoregressive pseudo-labelling with CTC-driven teacher forcing, then recovering some train-test fidelity via mixed sampling. The empirical evidence is persuasive that this improves training efficiency and out-of-distribution robustness in unified speech recognition, and the ablations largely support the mechanism. My main reservation is not about novelty or usefulness, but about the scope and interpretation of the gains: the method’s logic around globally incoherent teacher outputs is convincing but not fully principled, the biggest improvements are in OOD/long-sequence settings rather than uniformly everywhere, and some headline claims would benefit from more rigorous compute and statistical reporting. Even so, the contribution stands as a meaningful and likely impactful advance for semi-supervised unified speech recognition, and it appears to meet ICLR’s bar provided the authors tighten the claims and clarify the mechanism more explicitly.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes USR 2.0, a modification to unified speech recognition (USR) that replaces expensive autoregressive pseudo-label generation with CTC-driven teacher forcing, and adds mixed sampling to reduce train-test mismatch. The main claim is that this makes semi-supervised training faster and more robust, while also improving in-distribution performance across ASR, VSR, and AVSR with a single unified model.

### Strengths
1. **Addresses a real bottleneck in semi-supervised speech recognition.**  
   The paper clearly identifies that USR’s autoregressive pseudo-labeling is expensive because it must run at every training step, and argues that this is a practical scaling limitation. The proposed CTC-driven teacher forcing directly targets this issue and the paper reports nearly 2× faster training.

2. **The method is conceptually simple and well-motivated.**  
   The core idea—use greedy CTC outputs as decoder inputs during pseudo-label generation—is easy to understand and grounded in a plausible observation: CTC is more robust under domain shift, while attention decoding is better in-distribution. This makes the design easier to adopt than more elaborate sequence modeling alternatives.

3. **Extensive empirical evaluation across multiple robustness axes.**  
   The paper evaluates long utterances, noise, and OOD datasets, not just standard benchmark splits. This is a strong fit for ICLR’s preference for methods that are tested under distribution shift and not only on average-case in-distribution metrics.

4. **Strong results on multiple benchmarks and modalities.**  
   The paper reports improvements over USR and several self-supervised baselines on LRS3, LRS2, and WildVSR, with a unified model covering ASR, VSR, and AVSR. The scaling results for the Huge model are particularly compelling if reproduced faithfully.

5. **Ablations support the core design choices.**  
   The paper includes ablations on decoder supervision weights, CTC supervision weights, merge-and-collapse, and AR sampling probability. These help justify why the mixed CTC/AR training strategy is used rather than a simpler single-mode variant.

### Weaknesses
1. **Novelty is moderate rather than high by ICLR standards.**  
   The paper combines known ingredients—CTC, teacher forcing, self-training, and scheduled-sampling-like mixing—in a thoughtful way, but the conceptual leap is incremental. The contribution seems more like an effective engineering refinement of USR than a fundamentally new learning paradigm.

2. **The main technical justification is somewhat heuristic.**  
   The key claim is that global coherence of the decoder targets is unnecessary during pseudo-labeling because the student and teacher share the same CTC-derived conditioning. This is plausible, but the paper does not provide a strong theoretical argument or formal analysis of why this should preserve learning dynamics or when it might fail.

3. **Some claims appear stronger than the evidence directly shown.**  
   The abstract and introduction emphasize state-of-the-art results and robustness gains, but several comparisons are against prior work with different pretraining data, labeling regimes, or model scales. For ICLR reviewers, this can weaken the certainty of “state-of-the-art” claims unless the comparison protocol is very carefully matched.

4. **Reproducibility is helped by details, but still not fully ideal.**  
   The paper provides many hyperparameters and references code availability, which is good. However, the method depends on a fairly complex training pipeline, multiple datasets, filtering thresholds, and mode-sampling procedures, so reproducing exact numbers would still be challenging without very precise implementation details and released checkpoints/logs.

5. **The robustness story is compelling but somewhat domain-specific.**  
   The method is demonstrated on speech recognition, especially audiovisual settings. The broader claim that it generalizes to other sequence-to-sequence domains is speculative; no non-speech experiments are provided to support this extension.

6. **Potential train/test mismatch is acknowledged but not fully resolved.**  
   Mixed sampling is introduced to mitigate exposure bias, yet the training objective still relies on a conditioning scheme that differs from inference. The paper shows this works empirically, but the mismatch remains an important concern, especially for attention-heavy decoding.

### Novelty & Significance
**Novelty:** Moderate. The paper’s main idea is a clever and useful adaptation of CTC-guided training for pseudo-label generation, but it builds directly on established components rather than introducing a substantially new framework.  
**Significance:** Good for the speech recognition community, especially for semi-supervised and multimodal speech models where decoding cost and OOD robustness matter. For ICLR, the work is relevant and potentially impactful, but the acceptance bar is high, and the contribution may be viewed as more specialized and incremental than broadly field-shaping.  
**Clarity:** Generally good. The paper explains the motivation, method, and empirical findings clearly, though the training modes and loss formulations are somewhat intricate and would benefit from even cleaner algorithmic presentation.  
**Reproducibility:** Above average but not ideal. There is substantial experimental detail and an apparent code release, yet the full system is large and complex, with many moving parts that make exact reproduction nontrivial.  
**Significance relative to ICLR standards:** Borderline-to-strong. If the reported gains are robust and comparisons are fair, this is a meaningful practical improvement. However, at ICLR, reviewers may ask whether the method is sufficiently novel beyond a well-engineered optimization of a prior system.

### Suggestions for Improvement
1. **Add a more formal justification for CTC-driven teacher forcing.**  
   A short analysis of why matched teacher-student conditioning should stabilize pseudo-label training would strengthen the paper substantially.

2. **Clarify the exact algorithm with a step-by-step pseudocode box.**  
   The two-mode training procedure, confidence filtering, and loss routing are somewhat complex. A concise algorithm listing would make implementation much easier.

3. **Strengthen fairness of comparisons.**  
   Where possible, include matched baselines trained under the same pretraining data, same compute budget, and same decoding setup. This would make the state-of-the-art claims more convincing to ICLR reviewers.

4. **Report compute and memory more systematically.**  
   Since the main motivation is efficiency, provide a table of training throughput, decoding latency, peak memory, and total GPU-hours across model sizes and modes.

5. **Include more ablations on failure modes.**  
   It would be useful to study when CTC-driven conditioning hurts most, e.g., under very noisy CTC prefixes or in cases where CTC and attention disagree strongly.

6. **Improve the generality argument.**  
   If the paper claims applicability beyond speech, a small demonstration on another sequence-to-sequence task would make that claim much more credible.

7. **Discuss limitations more explicitly in the main paper.**  
   The appendix and limitations section are helpful, but the main text would benefit from a clearer acknowledgment that the method is tailored to speech and still relies on complex multi-stage training.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct training-cost breakdown: teacher pseudo-label generation time, total wall-clock per epoch, GPU memory, and end-to-end cost including AR mode. The claim of “nearly 2× faster training” is central, but without separating PL generation from optimization, it is not clear where the speedup comes from or whether it holds at ICLR-scale workloads.

2. Compare against a stronger pseudo-labeling baseline that also tries to remove AR decoding cost, such as offline/frozen teacher pseudo-labeling or CTC-only self-training variants with the same unified architecture. Without this, the paper does not establish that the gains come from CTC-driven teacher forcing rather than from simply changing the pseudo-label quality or reducing decode frequency.

3. Add a beam-size/compute Pareto comparison against USR and any close CTC-attention/self-training baselines on the same benchmarks. The paper claims better robustness under fast decoding, but ICLR reviewers will want to see whether USR 2.0 is actually better at matched latency/compute, not only at fixed beam settings.

4. Run controlled experiments on unlabeled data quality and domain mismatch: train with in-domain unlabeled data only, OOD unlabeled data only, and mixed unlabeled data. The paper’s main justification is that OOD unlabeled data hurts USR more and that USR 2.0 fixes this; that claim is incomplete without isolating which unlabeled-data regimes drive the improvement.

5. Add a comparison to simpler consistency-style or confidence-filtering fixes to USR, such as stronger filtering, different confidence thresholds, or scheduled AR/CTC pseudo-label selection. Without these, it is not clear that the proposed mechanism is necessary rather than a tunable variant of the existing framework.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze how often CTC-driven teacher forcing produces malformed attention pseudo-labels and whether those errors correlate with final WER. The paper admits sequence-level incoherence, but does not quantify how much it occurs or whether it ever harms training; without that, the core argument that incoherence “does not matter” is not convincing.

2. Provide a modality-wise error analysis for ASR vs VSR vs AVSR under ID and OOD conditions. The paper claims the method especially helps OOD and long utterances, but does not explain why gains differ across modalities or whether the effect is driven mostly by the audio branch, visual branch, or multimodal fusion.

3. Report sensitivity to the mixed-sampling probability and CTC/attention loss weights beyond the few points in the appendix, ideally with variance across seeds. ICLR reviewers will expect evidence that the method is stable and not dependent on a narrow hyperparameter setting, especially since the paper introduces a new training regime.

4. Quantify whether improvements come from better pseudo-label accuracy, better alignment, or just regularization from forcing the decoder to see shorter aligned prefixes. The mechanism is underspecified, and without a causal analysis the method reads as an engineering change rather than a principled advance.

5. Include seed-wise variance or confidence intervals on the main benchmark tables, especially for the reported SOTA results. The paper makes strong performance claims across several datasets, but the evidence shown is mostly point estimates; at ICLR, robustness and reproducibility matter for judging whether gains are meaningful.

### Visualizations & Case Studies
1. Add side-by-side pseudo-label examples showing teacher CTC outputs, CTC-driven attention outputs, AR teacher outputs, and final student predictions. This would reveal whether USR 2.0 truly improves supervision quality or merely shifts errors into a different form.

2. Add a scatter or calibration plot of teacher confidence versus student correctness for CTC-driven and AR modes. That would expose whether the proposed filtering and mixed sampling actually identify reliable pseudo-labels under domain shift.

3. Show error breakdowns by utterance length, noise level, and OOD dataset with failure categories such as truncation, repetition, substitution, and omission. The current qualitative examples are too selective; a structured breakdown would show whether the method systematically fixes the claimed failure modes.

4. Visualize training dynamics for the teacher and student separately: pseudo-label WER over epochs, not just validation WER. This is needed to support the claim that the self-reinforcing error loop is reduced.

5. Add a latency/accuracy plot for pseudo-label generation itself, not just final inference. Since the method’s main contribution is faster self-training, the reader needs to see whether the speedup survives realistic batch sizes and sequence lengths.

### Obvious Next Steps
1. Test the same CTC-driven teacher-forcing idea on non-audiovisual ASR and on streaming/online speech setups. The paper claims broad applicability, but it only validates the idea in unified speech recognition; extending it would show whether the contribution is general or domain-specific.

2. Evaluate whether a stronger sequence-level objective or constrained decoding during pseudo-label generation can further reduce malformed attention targets without restoring full AR cost. The paper identifies a real weakness but stops at mixed sampling instead of pursuing a more principled fix.

3. Add a full ablation on unlabeled-data scale to determine whether USR 2.0 scales better than USR as more unlabeled hours are added. Since scalability is part of the claim, the paper should show the method’s benefit persists rather than saturating early.

4. Compare against a unified model trained without pseudo-labeling but with the same architecture and supervision budget. That baseline is needed to separate gains from the architecture itself versus gains from the new pseudo-labeling strategy.

5. Evaluate on at least one additional challenging speech benchmark outside the LRS family and WildVSR. ICLR reviewers will look for evidence that the method is not overfit to a narrow family of TED-talk/lipreading datasets and that the robustness claims transfer.

# Final Consolidated Review
## Summary
This paper proposes USR 2.0, a modification to unified speech recognition that replaces expensive autoregressive pseudo-label generation with CTC-driven teacher forcing, and uses mixed sampling to partially recover train-test fidelity. The claimed benefit is a single unified model that trains substantially faster and is more robust under long utterances, noise, and dataset shift, while retaining strong in-distribution performance across ASR, VSR, and AVSR.

## Strengths
- The paper targets a real and well-motivated bottleneck in semi-supervised unified speech recognition: autoregressive pseudo-label generation is slow and brittle, especially when repeated at every training step. The proposed CTC-driven teacher forcing is a concrete and plausible fix, and the paper does show substantial reductions in training time.
- The empirical scope is strong and aligned with the claim: the authors test long utterances, noisy inputs, OOD datasets, and in-distribution benchmarks, and the results consistently show that USR 2.0 is more robust than USR, especially under greedy decoding and distribution shift.
- The ablation suite is genuinely useful. The paper isolates the roles of CTC vs attention supervision, the mixed AR/CTC sampling probability, and the CTC merge-and-collapse step, which supports the core design rather than leaving the method as a black box.

## Weaknesses
- The core justification for CTC-driven teacher forcing remains heuristic. The paper argues that global coherence of attention pseudo-labels is unnecessary because teacher and student share the same CTC-derived prefix conditioning, but this is only explained intuitively. The appendix even acknowledges malformed or repeated sequence-level outputs, so this is not a minor corner case; the paper does not convincingly explain why such inconsistencies do not corrupt learning.
- The main efficiency claim is under-measured. The paper repeatedly states “nearly 2× faster training,” but it does not provide a clean breakdown of where the savings come from in the main text: pseudo-label generation time, optimization time, memory, and epoch count are all mixed together. For a paper whose central contribution is efficiency, this is not rigorous enough.
- Several headline performance claims are stronger than the evidence presented. The comparisons span different pretraining data, labeled/unlabeled hours, and sometimes different supervision regimes, so the “state-of-the-art” framing is only safe within a fairly narrow comparison class. The paper sometimes blurs this distinction.
- The method is still dependent on a reasonably competent CTC branch. When CTC prefixes are wrong, the derived attention targets can become malformed or repetitive; the paper shows examples, but does not quantify how often this happens or how sensitive the final model is to such failures. That weakens the robustness story.

## Nice-to-Haves
- A step-by-step algorithm box for the two-mode training procedure would make the method easier to reproduce.
- A more explicit compute/latency table, including pseudo-label generation cost and peak memory across modes, would strengthen the efficiency story.
- A structured analysis of failure cases by utterance length, noise level, and modality would make the robustness claims more convincing.

## Novel Insights
The most interesting aspect of this paper is that it reframes pseudo-label generation as a conditioning problem rather than a sequence-generation problem: instead of insisting that the teacher produce globally coherent autoregressive outputs, the method uses CTC to anchor the decoder into a stable token prefix space where teacher and student can be aligned cheaply. That is a useful insight for iterative self-training, and the paper’s results suggest that under pseudo-labeling, local conditional consistency may matter more than globally polished teacher sequences. The downside is that this idea is still only empirically justified; the paper demonstrates the phenomenon works, but does not fully explain why it should be reliable beyond speech.

## Potentially Missed Related Work
- None identified

## Suggestions
- Add a dedicated efficiency analysis table with per-epoch wall-clock time, pseudo-label generation time, GPU memory, and total GPU-hours for USR vs USR 2.0.
- Quantify malformed/degenerate CTC-driven attention pseudo-labels and correlate them with final WER to test the central “incoherence does not matter” claim.
- Tighten the wording of the state-of-the-art claims so they are explicitly scoped to matched semi-/self-supervised unified settings.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
