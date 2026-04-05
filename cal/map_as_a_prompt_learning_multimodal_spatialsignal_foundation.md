=== CALIBRATION EXAMPLE 54 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Does the title accurately reflect the contribution?**  
  Mostly yes. The title signals the two main ideas the paper claims: multimodal spatial-signal foundation models and cross-scenario wireless localization. However, “Map as a Prompt” is a bit stronger and more specific than what is actually demonstrated unless the prompt mechanism is shown to be central across all settings, not just an added fine-tuning component.

- **Does the abstract clearly state the problem, method, and key results?**  
  Yes at a high level. It identifies the localization generalization problem, proposes cycle-adaptive masking plus map-as-prompt, and claims strong gains.

- **Are any claims in the abstract unsupported by the paper?**  
  The abstract says the model is a “multimodal foundation model” and achieves “state-of-the-art performance across multiple localization tasks” with “strong zero-shot generalization in unseen environments.” The paper does include pre-training and transfer experiments, but the “foundation model” framing is somewhat overstated relative to the evidence: experiments are limited to DeepMIMO/WAIR-D-style ray-tracing settings, mostly one scenario family, and the transfer setup still fine-tunes task heads. The “zero-shot” wording is also not fully aligned with the reported 100-sample fine-tuning in Section 4.5.

### Introduction & Motivation
- **Is the problem well-motivated? Is the gap in prior work clearly identified?**  
  Yes, the general problem is well motivated: CSI-based localization struggles in NLoS, multipath, and new environments. The paper also clearly distinguishes prior supervised, SSL, and LLM-based wireless approaches.

- **Are the contributions clearly stated and accurate?**  
  The three contributions are stated clearly in Section 1.2. The first two are plausible and specific. The third, “Parameter-Efficient Generalization,” is more of an empirical claim than a contribution, and it depends heavily on the restricted experimental setup.

- **Does the introduction over-claim or under-sell?**  
  It over-claims in a few places. The paper positions SigMap as a “foundation model” and suggests broad generalization across wireless tasks, but the evaluation only covers localization on a small number of simulated scenarios. The critique is not that foundation-model direction is inappropriate for ICLR, but that the evidence here does not yet justify the breadth of the framing.

### Method / Approach
- **Is the method clearly described and reproducible?**  
  The overall pipeline is understandable: self-supervised pre-training with cycle-adaptive masking, then map-conditioned prompt tuning. But there are important reproducibility gaps. In particular:
  - The **cycle-adaptive masking** procedure in Section 3.3 and Appendix B.4 is described with several overlapping notions (cross-correlation, periodicity shift, slope parameter, antenna/subcarrier masks), but the exact algorithmic pipeline is not crisp enough to implement unambiguously.
  - The **geographic prompt generation** in Section 3.4 is conceptually clear, but the graph construction is under-specified: how exactly are mesh vertices sampled, what neighborhood radius or triangulation variant is used, and how are buildings vs. base stations represented in a heterogeneous graph?
  - The paper says the backbone is frozen during fine-tuning, but the relation between the frozen transformer, the GNN prompt module, and the task head is not fully specified in a way that would let a reader reproduce the full training dynamics.

- **Are key assumptions stated and justified?**  
  Some are stated, but not well justified. The method assumes:
  - CSI periodicity can be reliably estimated from row-wise cross-correlation;
  - periodic masking prevents shortcut learning;
  - 3D map geometry can be meaningfully compressed into a single soft prompt token.  
  These are plausible, but the paper does not provide enough evidence that each assumption holds robustly across environments or hardware settings.

- **Are there logical gaps in the derivation or reasoning?**  
  Yes, several:
  - In Section 3.3, the leap from periodicity detection to “meaningful global representations” is asserted, but the causal link is not established beyond the ablation table.
  - In Section 3.4, the claim that the map prompt provides “interpretable fusion of environmental constraints” is not demonstrated; there is no interpretability analysis.
  - The multi-BS fusion in Equation (10) is described as attention-based, but it is not clear whether the fusion operates on independent per-BS predictions or on shared latent features, which matters for understanding the model’s behavior.

- **Are there edge cases or failure modes not discussed?**  
  Yes. The method likely depends on:
  - reliable map availability and alignment;
  - similar ray-tracing or sensing conditions between pre-training and deployment;
  - stable periodic structure in CSI, which may change with hardware, antenna layout, bandwidth, or calibration.  
  None of these failure modes are seriously discussed.

- **For theoretical claims: are proofs correct and complete?**  
  There are no formal theoretical claims or proofs, so this is not applicable.

### Experiments & Results
- **Do the experiments actually test the paper's claims?**  
  Partially. The experiments do test localization accuracy, the effect of map prompts, the effect of masking, and some cross-scenario transfer. That said, they do not fully justify the broader “foundation model” claim or the “cross-scenario” generality, since the evaluation is confined to DeepMIMO-derived and WAIR-D scenarios, both still within a ray-tracing/localization niche.

- **Are baselines appropriate and fairly compared?**  
  The chosen baselines are relevant but incomplete. Supervised and SSL baselines like CNN, SWiT, and LWLM are appropriate. However:
  - There is no comparison to stronger recent transformer-based or prompt-based localization models beyond the cited ones.
  - The paper does not clearly state whether all baselines use the same data splits, same pre-training data, and same label budgets.
  - The reported training budgets differ substantially; fairness is hard to assess without more detail.

- **Are there missing ablations that would materially change conclusions?**  
  Yes. The most important missing ablations are:
  1. **Prompt token design ablation**: Why a single geographic prompt token, and how does it compare to multiple prompts or alternative injection locations?
  2. **Map encoder ablation**: Does the gain come from graph structure, from base station coordinates alone, or from detailed mesh vertices?
  3. **Masking hyperparameter sensitivity**: The cycle-adaptive mask width and periodicity estimation are central, but there is no sensitivity analysis.
  4. **Pretraining-data scale ablation**: Since “foundation model” framing depends on pretraining, it would be important to show performance as a function of unlabeled data size.
  5. **Generalization with no fine-tuning**: Section 4.5 still fine-tunes task heads on around 100 samples. A true zero-shot test would materially strengthen the claim.

- **Are error bars / statistical significance reported?**  
  The paper says results are averaged over 5 independent runs, which is good, but it does not report standard deviations or confidence intervals in the main tables. That is a weakness, especially because some differences are modest in Table 3 and Table 4.

- **Do the results support the claims made, or are they cherry-picked?**  
  The results generally support the claim that adding map prompts helps and that the proposed masking improves over the two masking alternatives. However, some claims are stronger than the evidence:
  - The “state-of-the-art” claim is only supported against a limited baseline set.
  - The “strong zero-shot generalization” claim is weakened by the few-shot head fine-tuning.
  - The reported cross-scenario results are on only two target scenarios, one of which is still DeepMIMO and the other ray-traced urban scenes. This is not broad cross-domain evidence.

- **Are datasets and evaluation metrics appropriate?**  
  Yes, MAE/RMSE/CDF@1m are standard localization metrics. DeepMIMO and WAIR-D are reasonable datasets. However, evaluation exclusively on simulated/ray-traced data limits external validity for ICLR-level claims about robust real-world localization.

### Writing & Clarity
- **Are there sections that are confusing or poorly explained?**  
  Yes, mostly in the method:
  - Section 3.3 blends mask-generation intuition with several mathematical forms, but the exact sequence of operations is difficult to follow.
  - Section 3.4 has ambiguity around the graph representation and prompt insertion.
  - The multi-BS fusion mechanism in Section 3.5 is not fully transparent about whether the per-BS heads are independent or shared.
  - Appendix B.4 is useful, but it reads more like an extended heuristic description than a precise algorithmic specification.

- **Are figures and tables clear and informative?**  
  The tables do communicate the main quantitative results well. But several figures are referenced as if they establish a specific causal mechanism without showing the relevant evidence:
  - Figure 3 illustrates masking, but not why the pattern is optimal.
  - Figure 4 shows the prompt pipeline, but not the exact graph construction.
  - Figures 8 and 9 are mentioned for CDFs, but the paper’s strongest claims rely on them without including statistical dispersion or more extensive scenario diversity.

### Limitations & Broader Impact
- **Do the authors acknowledge the key limitations?**  
  Only partially. The conclusion mentions future work on additional wireless tasks and integrating visual modalities, but this does not address the core limitations of the current study.

- **Are there fundamental limitations they missed?**  
  Yes:
  - Dependence on accurate 3D maps and base-station coordinates.
  - Limited validation beyond ray-tracing/benchmark datasets.
  - Potential sensitivity to antenna geometry, frequency band, and hardware calibration.
  - The need for some downstream fine-tuning even in the transfer setting.

- **Are there failure modes or negative societal impacts not discussed?**  
  The paper does not discuss failure modes in difficult or adversarial settings, such as incorrect or outdated maps, mislocalized infrastructure, or degraded performance in unmodeled propagation conditions. Societal impact is likely modest and mostly beneficial, but the privacy/security implications of precise localization systems are not addressed.

### Overall Assessment
SigMap is an interesting and plausible direction: combining self-supervised wireless representation learning with map-conditioned prompt tuning is a sensible idea, and the experiments do suggest gains over a limited set of baselines. However, for ICLR’s bar, the paper currently feels narrower and less rigorously validated than its “foundation model” framing implies. The main concerns are the limited breadth of evaluation, the lack of a true zero-shot test, incomplete ablations for the prompt and masking design, and some under-specified methodological details that affect reproducibility. I think the contribution is promising, but it needs stronger evidence and clearer substantiation before the broad claims are fully convincing.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes SigMap, a wireless localization foundation model that combines self-supervised masked channel modeling with a map-conditioned prompt mechanism for cross-scenario adaptation. The core idea is to learn CSI representations via a cycle-adaptive masking scheme that aims to avoid periodicity shortcuts, then inject 3D map information as lightweight prompts during fine-tuning to improve localization accuracy and transfer to unseen environments.

### Strengths
1. The paper targets an important and timely problem for ICLR-relevant representation learning: learning transferable multimodal foundation models for a structured scientific domain. The combination of self-supervised pretraining and parameter-efficient adaptation is conceptually aligned with current foundation-model directions.
2. The proposed cycle-adaptive masking is domain-aware rather than generic. The authors motivate it with CSI periodicity and describe a concrete mechanism based on cross-correlation-derived shifts, which is more specific than standard random masking.
3. The map-as-prompt idea is a reasonable multimodal design choice. Using a GNN over 3D building geometry and base-station positions to generate soft prompts is a plausible way to inject environmental context without fully fine-tuning the backbone.
4. The experimental section covers several relevant axes: single-BS vs multi-BS localization, map ablations, masking ablations, unseen-scenario generalization, and parameter-efficiency reporting. This breadth is stronger than in many domain papers.
5. The reported gains are substantial over the listed baselines, especially in multi-BS localization and zero-shot/few-shot transfer settings. If correct, these results suggest the method is practically meaningful.

### Weaknesses
1. The paper’s empirical credibility is weakened by several internal inconsistencies and missing methodological details. For example, the text reports different values for the same unseen-scenario setting in the generalization section, and some setup descriptions are not fully aligned across sections. This makes it difficult to assess exact experimental validity.
2. The novelty bar for ICLR is only partially met. Both masked self-supervised pretraining and prompt tuning are established ideas; the paper’s contribution is mainly in domain-specific adaptations. That can be acceptable, but the paper does not sufficiently demonstrate that the cycle-adaptive masking and map prompting are fundamentally new enough beyond clever engineering.
3. The method description is not precise enough for full reproducibility. Key details such as the exact transformer backbone, prompt length, mask ratio, pretraining objective, optimization schedule across stages, and how periodicity is estimated are not fully specified in a rigorous, implementation-ready way.
4. The evaluation appears limited to a narrow simulation-centric setup. Most experiments are on DeepMIMO and WAIR-D-derived scenes, which are useful but still largely synthetic or ray-tracing-based. For ICLR, stronger evidence on real-world deployment robustness would be expected if the paper claims broad foundation-model generalization.
5. The baseline comparison set is somewhat limited for the claims being made. Given the foundation-model framing, it would be important to compare more directly against stronger recent localization or multimodal adaptation methods, and to clarify whether all baselines were tuned fairly under the same data budget.
6. Some claims are stronger than the evidence supports. For instance, the paper repeatedly uses “foundation model” language, but the pretraining/fine-tuning setup is still fairly task-specific and restricted to one data modality plus map prompts. The paper does not clearly show broad task reuse beyond localization.
7. The ablation on map representation is informative, but the causal story remains incomplete. The paper claims geometric/topological benefits, yet does not isolate which components of the map prompt drive improvement: geometry, building adjacency, LoS/NLoS structure, or simply extra scene-specific information.
8. Clarity is uneven. The paper is generally understandable at a high level, but several equations, algorithmic descriptions, and training/evaluation details are presented in a way that would be hard for another researcher to reproduce exactly without the code.

### Novelty & Significance
**Novelty:** Moderate. The overall recipe combines known ingredients—masked self-supervised learning, prompt tuning, and graph-based geometric encoding—but applies them to wireless localization in a thoughtful, domain-specific way. The cycle-adaptive masking is the most novel technical element, though its practical distinctiveness relative to prior masking heuristics is not fully established.

**Significance:** Potentially good for the wireless localization community, especially if the gains hold under rigorous independent replication. For ICLR, the significance is somewhat narrower because the contribution is domain-specific rather than broadly advancing general representation learning or multimodal foundation models.

**Clarity:** Medium. The high-level narrative is clear, but the implementation and experimental details need tightening. Internal numerical inconsistencies also hurt confidence.

**Reproducibility:** Medium-low as written. The paper states that code is available, which helps, but the manuscript itself does not provide enough exact detail to reproduce the results independently without substantial guesswork.

**Overall ICLR assessment:** The paper is interesting and potentially valuable, but it currently reads more like a strong application paper than a fully convincing ICLR-level methodology paper. To clear ICLR’s acceptance bar, it would need a sharper articulation of what is fundamentally new, stronger experimental rigor, and cleaner reporting.

### Suggestions for Improvement
1. Provide a fully specified algorithmic description of cycle-adaptive masking: exact periodicity estimation, mask ratio, window size selection, whether the procedure is deterministic or stochastic, and how it differs from standard shift/strip masking in implementation.
2. Clarify the exact architecture and training recipe: transformer depth/width, prompt length, pretraining objective, loss weighting, learning-rate schedule for each stage, number of pretraining/fine-tuning samples, and whether all baselines were trained under identical budgets.
3. Fix all numerical inconsistencies and ensure that results are reported consistently across text, tables, and appendix. Any mismatch between claimed MAE/RMSE values should be resolved, since this directly affects trust.
4. Strengthen the novelty argument by including a more direct comparison to prior masked modeling and prompt-based adaptation methods, ideally with a controlled analysis showing why cycle-adaptive masking is better than simpler alternatives.
5. Expand the evaluation beyond simulation-heavy datasets if possible. Even a small real-world testbed or transfer experiment would significantly strengthen the ICLR-level significance and credibility.
6. Add more granular ablations to isolate the map prompt mechanism: GNN vs simple MLP on map features, 2D vs 3D geometry, BS positions only vs full mesh, and prompt length/sensitivity.
7. Report statistical variation more thoroughly, including error bars or confidence intervals across more seeds, especially for the generalization experiments.
8. Tighten the framing around “foundation model.” If the model is specialized to localization, say so explicitly; if the goal is broader, demonstrate transfer to additional downstream wireless tasks such as channel estimation, beam prediction, or environmental inference.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add cross-dataset pretrain/fine-tune transfer beyond DeepMIMO/WAIR-D on the same or similar CSI localization setting, because the core “foundation model” claim depends on transferability, not just performance within ray-traced DeepMIMO variants. ICLR reviewers will expect evidence that the pretraining actually helps when the target domain, antenna setup, or frequency band changes materially.

2. Add stronger baselines for parameter-efficient adaptation: prompt tuning, LoRA, adapters, and full fine-tuning on the same frozen backbone. Without this, the claim that “map-as-prompt” is the right adaptation mechanism is not convincing; the improvement may simply come from any small amount of task-specific tuning.

3. Add a baseline that uses map information in a more direct, non-prompt form, such as concatenating map embeddings, graph-based fusion at the prediction head, or a learned occupancy/geometry encoder. The paper claims the prompt formulation is special, but it does not show that it beats simpler map-conditioning strategies.

4. Add ablations that isolate cycle-adaptive masking from the rest of the pretraining pipeline under the same compute budget and masking ratio, including random masking, block masking, periodic masking, and no pretraining. Right now it is unclear whether the gains come from the masking principle or just from additional training regularization.

5. Add scaling/label-efficiency curves across multiple fine-tuning set sizes, not just the ~100-sample setting. ICLR expects evidence that the method behaves as a foundation model; a single few-shot point is not enough to support the generalization and data-efficiency claims.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify how much the model truly relies on the map versus CSI alone under different geometric ambiguity regimes, e.g., LoS, mild NLoS, heavy NLoS, dense urban canyon. The current results show aggregate gains, but not whether map prompts solve the hard cases that motivate the method.

2. Analyze the learned cycle-adaptive masks and the periodicity estimator statistically, including failure cases where detected shifts are wrong or unstable. Without this, the masking method reads as a heuristic with unclear robustness, which weakens the pretraining claim.

3. Report per-scenario and per-BS breakdowns, not only averages, to show whether performance is consistent or driven by a subset of easy scenes/base stations. This matters because cross-scenario localization claims can be inflated by averaging over heterogeneous difficulty.

4. Include sensitivity analysis to map quality and alignment error, since the method assumes accurate 3D building geometry and base-station positions. The paper’s claims are fragile if small map misalignment collapses performance, and ICLR reviewers will look for that.

5. Provide parameter and compute comparisons against baselines, including training cost, inference cost, and adaptation cost. The paper claims parameter efficiency, but without a like-for-like cost analysis the practical contribution is unverified.

### Visualizations & Case Studies
1. Show qualitative localization maps with predicted vs. true positions overlaid on 3D scenes for both success and failure cases. This would reveal whether the model actually resolves NLoS ambiguity or is just benefiting from easy spatial priors.

2. Visualize attention weights from the map prompt and multi-BS fusion on representative scenes. If the prompt is meaningful, it should focus on map regions and base stations that are physically relevant; if not, the mechanism is likely decorative.

3. Visualize masked CSI reconstructions under random, strip, and cycle-adaptive masking to show what features the model learns to recover. This would make it clear whether cycle-adaptive masking prevents shortcut learning or merely makes reconstruction harder.

4. Show error maps or heatmaps over the environment, especially near corners, occlusions, and building edges. That would expose whether the method truly uses geometry or simply smooths predictions in ways that break in ambiguous regions.

### Obvious Next Steps
1. Test the method on real measured CSI, not only ray-traced datasets, because the ICLR-level claim of robustness and generalization is not credible without at least one hardware-based validation.

2. Extend evaluation to unseen antenna configurations, carrier frequencies, and bandwidths, since a foundation model for wireless localization should generalize across physical setups, not just new scenes.

3. Compare against a broader set of modern wireless foundation-model and localization methods under the same protocol, including recent prompt-based or multimodal approaches. The current baseline set is too narrow to justify “state-of-the-art” in an ICLR submission.

4. Add explicit uncertainty estimation or calibrated confidence outputs for localization. In practical wireless systems, uncertainty is as important as point accuracy, and it would strengthen the case that the model is usable beyond benchmark regression.

5. Validate whether the learned representations transfer to related tasks such as path loss prediction, LOS/NLOS classification, or beam selection. That would directly support the “foundation model” framing, which currently rests almost entirely on localization metrics.

# Final Consolidated Review
## Summary
This paper proposes SigMap, a wireless localization model that combines self-supervised masked CSI pretraining with map-conditioned prompt tuning for few-shot adaptation across scenarios. The core idea is to use cycle-adaptive masking to discourage shortcut learning from periodic CSI structure, then inject 3D geographic context through a GNN-generated soft prompt during fine-tuning. The paper reports strong gains on DeepMIMO and WAIR-D-style ray-tracing benchmarks, especially in multi-BS and map-augmented settings.

## Strengths
- The paper tackles a meaningful problem: cross-scenario wireless localization under multipath and NLoS conditions, which is genuinely hard and relevant to representation learning. The combination of SSL pretraining and parameter-efficient adaptation is conceptually well aligned with foundation-model style methods.
- The proposed cycle-adaptive masking is domain-aware rather than generic. The paper gives a concrete cross-correlation-based masking construction and shows an ablation where it outperforms grid and strip masking, which at least supports the claim that masking choice matters for CSI.

## Weaknesses
- The “foundation model” and “zero-shot generalization” framing is overstated relative to the evidence. The evaluation is limited to DeepMIMO and WAIR-D-style ray-traced scenarios, and the “unseen environment” results still fine-tune downstream heads on about 100 samples. This is better described as few-shot transfer on benchmark scenarios, not convincing foundation-model-level generalization.
- The method and experimental setup are under-specified in ways that matter for reproducibility and trust. The cycle-adaptive masking algorithm is described with overlapping notation, the graph construction for map prompts is not fully pinned down, and the exact backbone/training recipe is not presented cleanly enough to implement without guesswork.
- Several results and claims are not fully internally consistent. The manuscript contains inconsistent numerical reporting in the generalization section, and the reported performance claims are stronger than the comparison set supports. This weakens confidence in the exact experimental conclusions.
- The ablation coverage is incomplete for the paper’s main claims. There is no strong isolation of whether gains come from the map prompt itself versus simpler map conditioning, no sensitivity study for masking hyperparameters or periodicity estimation, and no scaling study showing how performance varies with pretraining data or fine-tuning budget. These omissions matter because the paper’s central claims depend on these design choices.
- Baseline coverage is too narrow for the breadth of the framing. The comparisons are relevant but limited, and the paper does not convincingly establish that SigMap is better than other plausible parameter-efficient adaptation or map-fusion strategies under a matched protocol.

## Nice-to-Haves
- A more direct comparison against simpler map-conditioning alternatives, such as concatenating map embeddings or using map features only at the prediction head, would help establish that the prompt formulation is actually useful.
- A broader label-efficiency curve across several fine-tuning budgets would make the data-efficiency story much more convincing than a single ~100-sample setting.
- Reporting standard deviations or confidence intervals in the main tables would improve interpretability, especially where gains are modest.

## Novel Insights
The most interesting aspect of the paper is not the broad “foundation model” framing, but the domain-specific coupling of masked pretraining with map-conditioned adaptation. Cycle-adaptive masking is a plausible way to force the model away from easy periodic shortcuts in CSI, and the map prompt idea is a lightweight way to inject scene geometry without full multimodal fusion. That said, the paper currently reads more like a promising domain adaptation recipe than a broadly validated foundation model; the main insight is promising, but the evidence does not yet support the larger narrative.

## Potentially Missed Related Work
- Prompt tuning for multimodal or structured scientific data — relevant as a closer conceptual comparison for the map-as-prompt design.
- Parameter-efficient adaptation methods such as adapters or LoRA — relevant because the paper’s fine-tuning mechanism is a core claim and should be compared against standard PEFT baselines.
- Recent wireless localization foundation-model / masked modeling work beyond the cited set — relevant because the baseline comparison appears somewhat narrow for the strength of the claims.

## Suggestions
- Add a true no-fine-tuning transfer test, or clearly stop calling the setting zero-shot.
- Report a fully specified masking algorithm and map-graph construction, including all hyperparameters and implementation details.
- Include PEFT baselines and simpler map-fusion baselines under the same protocol.
- Expand the evaluation to at least one real measured CSI testbed or a materially different antenna/frequency configuration.
- Fix all numerical inconsistencies and add variance estimates across seeds.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 6.0]
Average score: 5.3
Binary outcome: Accept
