=== CALIBRATION EXAMPLE 38 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately captures the central idea: long-term action anticipation using transcript-based supervision.
- The abstract clearly states the motivation, the weak-supervision setting, and the main components of TbLTA. It also names the datasets used.
- The main concern is that several claims are stronger than what the paper later substantiates. In particular, “the first weakly-supervised approach for LTA” and “the first transcript-only supervision baseline for LTA” are plausible, but the paper does not fully distinguish between dense anticipation, transcript-only supervision, and prior weak/semi-supervised dense LTA in a way that establishes novelty unambiguously. Also, the abstract implies broad robustness and cost-effectiveness, but the experiments show mixed results, especially on EGTEA and 50Salads.

### Introduction & Motivation
- The problem is well motivated: dense LTA annotations are expensive, and transcripts are a natural weaker form of supervision for procedural activities.
- The gap in prior work is mostly clear, especially relative to fully supervised dense LTA and the semi/weakly supervised baseline of Zhang et al. (2021). The paper also usefully connects LTA to transcript-based alignment literature from TAS.
- The contribution statements are concrete, but some are overstated. “Competitive long-horizon anticipation” and “dense LTA feasible with transcript supervision alone” are stronger than the evidence supports, because the results are not uniformly competitive across datasets and settings.
- ICLR-standard concern: the introduction positions this as a first-of-its-kind framework, but the paper should be more careful in delimiting the novelty relative to weakly supervised segmentation, transcript-to-frame alignment, and prior anticipation models that use language or procedural structure. The novelty is real, but the exact boundary of “first” needs tighter substantiation.

### Method / Approach
- The overall method is interesting and modular: encoder, weak alignment via ATBA, cross-modal transcript grounding, segmentation head, and anticipation decoder with CRF.
- However, the method description has important reproducibility and clarity gaps. Several components are described at a high level without enough detail to implement faithfully:
  - The temporal alignment module is said to “adopt ATBA” and partition transcripts into observed/future subtranscripts, but the paper does not fully specify how this is integrated into the LTA pipeline beyond a general reference to Xu & Zheng (2024).
  - The cross-attention mechanism in Eq. (1)–(2) is not cleanly presented in the main text. The exact form of the mask, what timestamps it uses, and how gradients flow through pseudo-labels are not fully clear.
  - The decoder is said to operate on “the fused encoder output” and use fixed queries, but the target formulation is ambiguous: is it predicting action tokens, durations, or both in parallel?
- There are also logical tensions in the objectives:
  - The paper uses ATBA pseudo-labels, CTC, a CRF over anticipation tokens, and an affinity-based duration head. It is not fully clear why all of these are necessary simultaneously, or how they interact without redundant constraints.
  - CTC is used for segmentation supervision, but the paper also says ATBA provides dense pseudo-labels for both segmentation and anticipation. The division of labor between ATBA and CTC is not precisely justified.
  - The duration loss is especially weakly grounded: the “ground truth target is approximated by the class-wise prior” from a momentum buffer. This is not really supervised learning in the usual sense, so the paper should more explicitly discuss what this signal can and cannot identify.
- Edge cases/failure modes are under-discussed:
  - Transcript-to-video alignment is likely to fail when repeated actions occur or when transcript order does not map cleanly to visual boundaries.
  - The approach seems particularly vulnerable when transcript granularity differs across videos or when actions are visually ambiguous.
  - The method assumes transcripts are available at training time and are semantically aligned with activity structure, which may not hold in broader LTA settings.
- The theoretical parts are not presented as formal claims, so proof completeness is not applicable. But Eq. (4)–(7) and the text around them would benefit from a more careful derivation of the objective and the decoding procedure.

### Experiments & Results
- The experiments do test the core claim that transcript-only supervision can support dense anticipation, and the dataset choices are appropriate for procedural activity understanding.
- That said, the evaluation is not fully convincing relative to ICLR’s bar because the comparison set and reporting are uneven:
  - The main tables compare against supervised methods, but the weakly-supervised dense LTA baseline (Zhang et al., 2021) is only sparsely reported and not in a fully comparable setting across horizons.
  - The paper says it reports deterministic and stochastic protocols, but the presentation mixes them in a way that makes direct comparison difficult.
  - For EGTEA, the metric switches to mAP over verbs only, which is reasonable for that benchmark, but it is not directly comparable to the MoC results on Breakfast and 50Salads; the paper should be clearer that these are different task formulations.
- There are several missing ablations that materially affect the conclusions:
  - No ablation isolating the effect of the transcript-only supervision itself versus pseudo-label alignment versus cross-modal grounding.
  - No ablation on the use of class tokens E, despite them being a central modeling choice.
  - No ablation on the CRF in the anticipation decoder versus a plain decoder.
  - No ablation on the progressive training schedule, which seems potentially critical because the model is trained in stages and reinitializes optimizer/scheduler.
  - No ablation on the observed/future split proportions or on transcript granularity.
- Error bars/statistical significance are not reported. Given the modest gains in several settings and the variance expected in dense anticipation, this is a notable omission.
- The results are mixed:
  - On Breakfast, the method sometimes matches or exceeds supervised baselines, which is impressive.
  - On 50Salads, gains are more limited and the paper itself acknowledges sensitivity to boundary precision.
  - On EGTEA, supervised baselines remain clearly stronger overall.
- This means the paper’s strongest conclusion should be that transcript supervision is promising and competitive in some procedural settings, not that it broadly surpasses fully supervised LTA.
- A further concern is that the paper’s tables and discussion suggest some cherry-picking in emphasis: the narrative foregrounds the best cases, while the weaker cases are framed more briefly.

### Writing & Clarity
- The paper is generally understandable, and the high-level storyline is coherent.
- The main clarity issue is not grammar but conceptual overload: the method combines ATBA pseudo-labeling, CTC, cross-modal attention, CRF decoding, and a duration prior, and the precise causal role of each component is difficult to disentangle from the current exposition.
- Several equations and definitions in the methodology section are not sufficiently explained in prose. In particular, the relationship among the segmentation head, the anticipation decoder, and the transcript alignment mechanism would benefit from a more explicit pipeline description.
- Figures and tables are mostly informative, but the experimental tables would be easier to interpret if the paper more clearly separated deterministic vs stochastic evaluation and explained which numbers are directly comparable.

### Limitations & Broader Impact
- The paper does acknowledge one important limitation: future duration prediction remains difficult, especially for unseen actions.
- However, it misses several fundamental limitations:
  - Reliance on transcripts at training time is still a meaningful annotation cost, and the paper does not quantify how transcript collection compares to sparse boundary labeling in practice.
  - The approach appears tailored to procedural, ordered activities; it is unclear how well it transfers to open-world video or activities with weak canonical order.
  - Alignment-based methods are sensitive to transcript quality and action vocabulary mismatch, but this is not discussed.
- Broader impact considerations are minimal. Since the work is mainly a modeling paper for procedural video understanding, there are no obvious direct harms, but the paper should still mention potential bias toward datasets with well-structured routines and the possibility of poor performance in more diverse settings.

### Overall Assessment
This is an interesting and timely paper that tackles a real annotation bottleneck in dense long-term action anticipation, and the transcript-only formulation is a meaningful contribution. The experimental results are encouraging on Breakfast and show that weak supervision can be viable in procedural domains. However, at ICLR’s bar, the paper currently falls short of fully convincing because the method is over-complex relative to the explanation, the interaction of losses is not sufficiently justified, key ablations are missing, and the empirical gains are uneven across datasets. I think the core idea stands, but the paper needs stronger methodological clarification and more rigorous ablation/evaluation to support its broad claims.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes TbLTA, a weakly supervised framework for dense long-term action anticipation (LTA) that trains using only ordered action transcripts, without frame-level boundary annotations. The core idea is to combine transcript-video temporal alignment, transcript-guided cross-modal grounding, a segmentation head, and an anticipation decoder with CRF and duration modeling, aiming to make dense anticipation scalable while retaining competitive performance on Breakfast, 50Salads, and EGTEA.

### Strengths
1. **Timely and practically important problem formulation for ICLR.**  
   The paper tackles a meaningful annotation bottleneck: dense LTA normally requires expensive frame-level labels, while transcripts are much cheaper. This is a strong systems-and-learning motivation that fits ICLR’s interest in learning from weak supervision and language-conditioned representation learning.

2. **Clear attempt to move LTA toward weaker supervision than prior work.**  
   The paper explicitly claims to be the first fully transcript-only approach for dense LTA, whereas prior weak/semi-weak work still used localized labels or partial frame supervision. If the experimental setup is correct, this is a concrete reduction in supervision that is potentially significant.

3. **Combines several complementary signals rather than relying on a single weak-loss trick.**  
   TbLTA integrates pseudo-label alignment, CTC supervision, cross-modal transcript grounding, and a structured anticipation decoder with CRF. This is a reasonable design for a hard weak-supervision problem, and the ablation discussion suggests each component contributes.

4. **Evaluated on standard benchmarks with multiple horizons.**  
   The use of Breakfast, 50Salads, and EGTEA is appropriate for LTA/TAS-style evaluation, and the paper reports performance over several observation/prediction splits. This increases the relevance of the results to the established community benchmarks.

5. **Some evidence of robustness of transcript supervision.**  
   The reported results suggest that transcript-only supervision can be competitive with stronger supervision in some settings, especially on Breakfast. Even if not universally best, this is an interesting empirical result that may change assumptions about what supervision is needed for LTA.

### Weaknesses
1. **Novelty is somewhat incremental relative to adjacent weakly supervised segmentation literature.**  
   The method appears to repurpose established components—ATBA-style alignment, CTC, cross-attention, CRF, and duration priors—into the LTA setting. The main novelty is the supervision regime and task transfer rather than a fundamentally new modeling principle, which may limit the conceptual advance expected at ICLR.

2. **The paper’s technical story is not always cleanly disentangled.**  
   TbLTA combines several losses and modules, but the exact causal role of each part is hard to isolate from the narrative. For example, the interaction among ATBA pseudo-labels, CTC, segmentation supervision, and anticipation supervision is complex, and the paper does not fully justify why this specific combination is necessary versus simpler alternatives.

3. **Reproducibility is weakened by incomplete methodological specificity.**  
   Key implementation details appear underspecified from the paper text alone: exact ATBA settings, the construction and update of the temporal mask for cross-attention, the precise CRF parameterization, how duration priors are initialized/updated, and how transcript partitioning into observed/future subtranscripts is handled in practice. For ICLR, where reproducibility expectations are high, this is a notable weakness.

4. **The empirical comparison may not be fully convincing as presented.**  
   The paper claims competitiveness with fully supervised methods, but in the results narrative there are also clear cases where supervised baselines remain better, especially on EGTEA. The claims are therefore somewhat stronger than the evidence supports, and the paper should be more careful in framing the gains as benchmark- and dataset-dependent.

5. **Potential concern about fairness of comparisons under different supervision regimes.**  
   The paper compares transcript-only training to fully supervised methods, but the training protocols are inherently different. Without a very careful accounting of pretraining, feature extraction, and tuning budgets, it is difficult to tell whether gains come from the weak-supervision strategy itself or from other design choices.

6. **The duration modeling component seems weakly grounded.**  
   The duration predictor is trained using class-wise priors estimated from predicted labels, which risks circularity and propagating bias from early-stage segmentation errors. The paper itself notes that duration estimation remains challenging, which suggests this component is not yet robust.

7. **Limited evidence of generalization beyond the benchmark setting.**  
   All experiments are on instructional/procedural datasets with strong action order regularities. That is a sensible testbed, but the method’s reliance on ordered transcripts may make it less clear how broadly it applies to messier real-world videos where transcript order is noisier or ambiguous.

### Novelty & Significance
For ICLR standards, the paper has **moderate novelty and moderate-to-strong practical significance**, but not a breakthrough-level algorithmic novelty. Its main contribution is introducing transcript-only supervision for dense LTA, which is a meaningful reduction in annotation cost and could influence future work on weak supervision for structured video prediction.

That said, the method is largely a composition of known ingredients rather than a fundamentally new learning paradigm. The significance depends heavily on whether the reported performance is robust and whether transcript-only supervision can reliably substitute for dense labels across more settings than the chosen procedural benchmarks. The paper is promising and relevant, but the acceptance case at ICLR would hinge on stronger evidence of generality, cleaner ablations, and sharper articulation of what is truly new.

### Suggestions for Improvement
1. **Provide a more rigorous ablation of each training signal and module.**  
   Separate the effects of ATBA, CTC, cross-modal attention, CRF, and duration loss, including combinations, not just individual removals. This would clarify which part is driving the gains.

2. **Strengthen the comparison protocol and reporting.**  
   Clearly state whether all methods use the same input features, same pretraining, same temporal backbone capacity, and same evaluation protocol. Add statistical variability across splits or multiple runs if possible.

3. **Clarify the exact weak-supervision pipeline in algorithmic form.**  
   A step-by-step pseudocode description of training and inference would help substantially, especially for how pseudo-labels, transcript alignment, and stage-wise optimization interact.

4. **Be more careful in the claims.**  
   Instead of saying the model is competitive with or superior to fully supervised methods broadly, qualify the claim by dataset, horizon, and metric. This would make the paper more credible.

5. **Improve justification for the duration modeling design.**  
   Provide evidence that the momentum-based duration priors are stable, not merely circular. Consider alternative formulations or an analysis of failure cases when duration priors are inaccurate.

6. **Add a stronger analysis of transcript quality and noise sensitivity.**  
   Since transcripts are the only supervision, it would be valuable to test robustness to transcript noise, missing actions, or permutation errors. This would make the weak-supervision claim more convincing.

7. **Include runtime and annotation-cost analysis.**  
   For an ICLR audience, the practical value of weak supervision is clearer if the paper quantifies annotation savings and training/inference overhead relative to fully supervised baselines.

8. **Discuss broader applicability and limitations more explicitly.**  
   The method is well-matched to procedural videos; the paper should more clearly state when transcript-only supervision is likely to work and when it may fail. This would improve scientific honesty and help readers understand the scope of the contribution.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison to the strongest recent dense LTA baselines across the exact same protocols and features, especially ActFusion, ANTICIPATR, and the CRF-based LTA method of Mate & Dimiccoli, because the current tables do not convincingly isolate whether transcript supervision or just newer architecture/losses explain gains. ICLR reviewers will expect a fair, up-to-date benchmark against the best supervised and stochastic methods under matched settings.

2. Add a comparison against transcript-only or weakly-supervised sequence-alignment baselines adapted from TAS/weak supervision, such as CTC-only, ATBA-only, and HMM/DTW-style alignment variants, because the core claim is that transcript supervision is sufficient for dense LTA. Without showing that TbLTA beats simpler alignment-based weak supervision, the method contribution is not credible.

3. Add an ablation that removes each supervision source independently: alignment loss, CTC, cross-modal transcript grounding, global transcript loss, CRF, and duration head. Right now the paper claims multiple mechanisms are necessary, but there is no evidence which one actually drives performance versus redundancy or overfitting.

4. Add experiments under varying transcript quality: noisy transcripts, incomplete transcripts, shuffled action order, and truncated action lists. Since the paper claims transcript-only supervision is practical and scalable, it must show robustness to the kind of imperfect transcripts that would occur in real use.

5. Add a label-efficiency study comparing transcript cost against frame-level annotation cost and performance at different supervision budgets. The main claim is reduced annotation burden, but the paper does not quantify how much supervision is saved relative to performance loss or compare against mixed-supervision baselines.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze alignment quality against ground-truth boundaries, not just downstream accuracy. Without boundary-level metrics such as alignment IoU, edit score, or boundary F1 for the pseudo-labels, the claim that transcript alignment produces meaningful frame supervision is unverified.

2. Analyze failure modes by action type and transition complexity, especially for short actions, repeated actions, and visually ambiguous steps. The paper itself suggests duration estimation is weak, so the reviewer needs to know whether the method works only on clean procedural sequences or actually generalizes to hard cases.

3. Analyze how much each component depends on the observed portion length and anticipation horizon. The results are reported over multiple observation ratios, but there is no study showing whether the method degrades gracefully as horizons grow, which is central to LTA.

4. Analyze the learned duration priors and whether they transfer across classes or simply memorize dataset statistics. The duration head is a major claimed contribution, but without inspecting predicted durations and their calibration, it is unclear whether it learns meaningful temporal structure.

5. Analyze whether the transcript grounding actually improves the video representation or just injects label information that leaks the target sequence. A representation-level study or probing task is needed to show the cross-attention is doing more than shortcutting the prediction pipeline.

### Visualizations & Case Studies
1. Show side-by-side visualizations of transcript-to-video alignment, pseudo-labels, and final predictions for successes and failures. This would directly reveal whether the temporal alignment module is producing sensible boundaries or just smearing labels across the video.

2. Show qualitative timelines for hard cases with repeated actions, long pauses, and abrupt transitions. These cases would expose whether TbLTA truly anticipates future structure or only works when action order is highly stereotyped.

3. Show attention maps from the cross-modal module over time for specific transcript tokens. This would reveal whether the language grounding is actually localized to relevant video segments or diffusely attends everywhere.

4. Show predicted versus true durations per action class, especially for rare classes. This would expose whether the duration head is genuinely learning temporal priors or collapsing to average duration heuristics.

### Obvious Next Steps
1. Add a stronger weak-supervision baseline family and a clean supervised/weakly-supervised tradeoff curve, because the paper’s central contribution is a new supervision regime and that must be positioned against the closest alternatives.

2. Add transcript-noise and transcript-availability experiments, because the practical value of transcript-only LTA depends on whether the method survives imperfect, partially missing, or automatically generated transcripts.

3. Add a principled study of whether transcripts are best used as supervision, as input context, or both. The paper currently mixes these roles, but the claim that transcript supervision alone enables anticipation needs a clearer decomposition of what is actually necessary.

4. Add calibration or uncertainty analysis for stochastic anticipation outputs. Since the paper claims to model multiple plausible futures, it should show whether the predicted diversity is meaningful and whether uncertainty correlates with ambiguity in the video.

5. Add a compute and annotation-cost analysis. ICLR expects the practical claim of “less costly” supervision to be backed by explicit savings in annotation effort and ideally training/inference overhead.

# Final Consolidated Review
## Summary
This paper proposes TbLTA, a weakly supervised framework for dense long-term action anticipation that trains only from ordered action transcripts rather than frame-level boundaries. The method combines transcript-video temporal alignment, transcript-guided cross-modal grounding, a segmentation head with CTC supervision, and a structured anticipation decoder with CRF and duration modeling.

## Strengths
- The problem setting is genuinely important: dense LTA normally depends on costly frame-level annotation, and using transcripts is a meaningful step toward cheaper supervision for procedural video understanding.
- The paper makes a concrete supervision reduction relative to prior weak/semi-weak dense LTA work, and the experiments do establish that transcript-only supervision can work nontrivially on standard benchmarks, especially Breakfast.

## Weaknesses
- The method is a fairly heavy composition of existing ingredients rather than a clearly novel learning principle — ATBA-style alignment, CTC, cross-attention grounding, CRF decoding, and duration priors are stacked together, but the paper does not convincingly isolate what is essential versus decorative. This makes the contribution feel more like engineering around weak supervision than a crisp algorithmic advance.
- The empirical story is uneven and the main claims are overstated. Results are strongest on Breakfast, weaker on 50Salads, and supervised baselines still remain clearly better on EGTEA; that does not support broad claims that transcript-only supervision is generally competitive with or superior to fully supervised LTA.
- Reproducibility and methodological clarity are not strong enough for a paper with this many moving parts. The interaction between pseudo-label alignment, CTC, the segmentation head, the anticipation decoder, and the duration prior is not fully spelled out, and key details such as the exact alignment/masking behavior and the stage-wise training procedure would be hard to reimplement faithfully from the paper alone.
- The duration modeling component is particularly weakly justified. It relies on class-wise priors estimated from predicted labels, which risks circularity and makes the signal look more like a heuristic than a principled supervision source; the paper itself also acknowledges that future duration estimation remains difficult.

## Nice-to-Haves
- A cleaner algorithmic description, ideally with pseudocode, would make the training/inference pipeline much easier to follow.
- A more direct comparison against simpler weak-supervision baselines adapted from transcript alignment literature would help show that TbLTA is doing more than combining known blocks.
- A label-efficiency or annotation-cost analysis would strengthen the practical motivation for transcript-only supervision.

## Novel Insights
The most interesting insight here is not just that transcripts are cheaper than boundaries, but that they may be especially well matched to long-horizon procedural anticipation because they encode the canonical ordering of actions rather than their exact timing. That said, the paper also reveals the limits of this idea: once the benchmark becomes less regular or more temporally ambiguous, the gains shrink, suggesting that transcript supervision is useful mainly when the action script has strong structure and stable ordering. In other words, the method seems to exploit procedural regularity as much as it exploits weak supervision itself.

## Potentially Missed Related Work
- Zhang et al. 2021, weakly-supervised dense action anticipation — directly relevant as the closest prior weak/semi-weak LTA setting.
- Xu & Zheng 2024, weakly supervised action segmentation with action-transition-aware boundary alignment — relevant because the paper’s alignment module is adapted from this line.
- Ng & Fernando 2021, weakly supervised action segmentation with attention — relevant as transcript-level weak supervision for sequence alignment.
- Huang et al. 2016 / ECTC and other CTC-based weak action labeling work — relevant because the method heavily relies on CTC-style alignment.
- Maté & Dimiccoli 2024, temporal context consistency above all — relevant to the CRF-style anticipation refinement used here.

## Suggestions
- Add a full ablation table that separates the contribution of ATBA alignment, CTC, cross-modal grounding, CRF decoding, class tokens, and duration prediction.
- Report boundary/alignment quality directly, not only downstream anticipation accuracy, to validate that pseudo-labels are actually meaningful.
- Include robustness tests under transcript noise, incomplete transcripts, or shuffled action order to support the claim that transcript-only supervision is practical.
- Tighten the claims in the paper: describe the method as promising and effective in procedural settings, rather than implying broad superiority over fully supervised LTA.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 4.0]
Average score: 3.0
Binary outcome: Reject
