=== CALIBRATION EXAMPLE 64 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- Does the title accurately reflect the contribution?
  - Yes. “History-Guided Sampling for Plug-and-Play Enhancement of Diffusion Models” accurately signals a sampling-time, training-free method that uses past predictions.
- Does the abstract clearly state the problem, method, and key results?
  - Mostly yes: it identifies poor quality at low NFE / low guidance, proposes a momentum/history-based sampler, and claims broad improvements plus a strong ImageNet FID result.
- Are any claims in the abstract unsupported by the paper?
  - The strongest claim is the “new state-of-the-art FID of 1.61 for unguided ImageNet generation at 256×256 with only 30 steps.” The paper does provide Table 3 for this, but the claim is somewhat fragile because the comparison is against specific pretrained checkpoints and a specific evaluation protocol; ICLR readers will expect very careful contextualization of what exactly is being beaten. Also, “practically no additional computation” is only partially supported: the method adds EMA/history maintenance plus DCT filtering and optional projection, which may be small, but “practically no” should be justified with timing/memory measurements in a more systematic way than one runtime note in Appendix D.

### Introduction & Motivation
- Is the problem well-motivated? Is the gap in prior work clearly identified?
  - Yes. The paper motivates two common failure modes: low NFE sampling and low CFG leading to blur/artifacts. That is relevant and well aligned with ICLR interests in improving diffusion sampling efficiency and quality.
  - The gap is also plausible: many works focus on better solvers or distillation, but fewer address a training-free, plug-and-play refinement of the inference trajectory itself.
- Are the contributions clearly stated and accurate?
  - The contributions are stated, but the framing overreaches a bit. The paper claims a “universal enhancement” and “consistently improves” across settings. That is stronger than the evidence really supports, which is limited to selected diffusion families and benchmarks.
- Does the introduction over-claim or under-sell?
  - It over-claims in saying HiGS “consistently improves” quality across different models and guidance scales. The experiments do show improvements in many reported settings, but not enough to justify a universal statement, especially since the gains are modest in some tables and the method depends on several extra design choices (projection, DCT filtering, schedule) that are not universally beneficial.

### Method / Approach
- Is the method clearly described and reproducible?
  - The high-level idea is understandable: form a history buffer of past predictions, subtract an EMA-style history signal from the current CFG prediction, optionally project and frequency-filter the update, then add it back with a schedule.
  - However, the method description has several clarity/reproducibility issues:
    - Equation (5) for the EMA/history term is not clearly typeset in the extracted text, but more importantly the conceptual definition of \(g(H_k)\) and its implementation in Algorithm 1/3 are not fully aligned. The paper says it may use all past predictions with EMA computed on the fly, but the exact recurrence and whether the buffer stores only EMA or multiple outputs should be unambiguous.
    - The projection step in Equation (7) and Algorithm 2 appears internally inconsistent in the parsed text, and the pseudocode suggests a potentially incorrect implementation. Since parser artifacts may exist, the core issue is that the paper must present this step cleanly and mathematically so readers can verify it.
    - The DCT masking operation is introduced as a key component, but its role is somewhat ad hoc: it is a heuristic fix for color artifacts rather than part of the central derivation.
- Are key assumptions stated and justified?
  - Not sufficiently. The central theoretical motivation is that Euler sampling can be interpreted as gradient descent on a time-varying energy, and HiGS can reduce local truncation error. But the assumptions needed for this to be meaningful in diffusion sampling are strong:
    - The paper assumes smoothness and bounded derivatives for the ODE drift \(u(z,t)\), but does not connect this to actual diffusion samplers and neural denoisers beyond standard ODE intuition.
    - The step-size/result in Appendix B depends on a specific weight choice \(w_k = 2h_k / h_{k-1}\), which does not obviously correspond to the actual heuristic schedule used in experiments. That weakens the claim that the method is theoretically justified by the error analysis.
- Are there logical gaps in the derivation or reasoning?
  - Yes, several:
    - The “Euler sampler as SGD on a time-varying energy” is more of an analogy than a rigorous equivalence, and the paper uses it to motivate a momentum-like update. That is fine as intuition, but the paper then leans on it quite heavily.
    - Appendix B claims local truncation error improves from \(O(h_k^2)\) to \(O(h_k^3)\), hence global error from \(O(h)\) to \(O(h^2)\), but this is not established for the actual HiGS algorithm in the main text. The derivation appears to analyze a simplified update with one-step history and a special weight, not the full algorithm with EMA, projection, DCT filtering, and time scheduling.
    - The claim that HiGS is “history-guided” because it uses past predictions is fine, but the mechanism by which subtracting an EMA of prior outputs improves sample quality is not theoretically grounded beyond analogy to momentum/variance reduction.
- Are there edge cases or failure modes not discussed?
  - Yes. The paper does not sufficiently discuss:
    - Behavior when sampling uses extremely few steps, where the history buffer is very short and the update may be unstable.
    - Whether the method can amplify model bias or hallucinated details, especially given the use of history-dependent corrections.
    - Whether the DCT filtering or projection can hurt semantic fidelity in cases where low-frequency changes are actually required.
    - What happens with samplers or parameterizations where CFG is unavailable or not meaningful.
- For theoretical claims: are proofs correct and complete?
  - The proofs are not fully convincing. Appendix B’s theorem statements are plausible in spirit, but the derivation is tied to a simplified update rule that does not match the actual full method. The claimed improvement from first-order to second-order global accuracy is especially strong for a heuristic inference modification and would need much more careful justification. As written, the proof is more of an intuition sketch than a complete argument for HiGS.

### Experiments & Results
- Do the experiments actually test the paper's claims?
  - Largely yes in the sense that they test sampling quality across multiple diffusion models, CFG scales, and step counts, which aligns with the claimed use case.
  - But they do not fully test the main scientific claim that the method is a generally valid improvement to diffusion sampling. The experiments focus on a limited set of pretrained image models, mostly Stable Diffusion variants plus DiT/SiT on ImageNet.
- Are baselines appropriate and fairly compared?
  - Mostly yes, but there are important caveats:
    - For the ImageNet comparison, Table 3 compares against recent models and reports improved FID with fewer steps. That is useful, but because HiGS is applied to specific pretrained checkpoints, the fairness depends critically on exact evaluation details and whether the baseline samplers are truly the official defaults.
    - In Table 1 and Table 2, the paper compares against “CFG” but the exact baseline configuration is not always transparent enough; some gains may reflect tuning of step counts or schedules.
    - Some tables mix models, datasets, and metrics in a way that makes it harder to understand what is being compared against what.
- Are there missing ablations that would materially change conclusions?
  - Yes, several matter:
    - A clean ablation of each main component: history term only, + schedule, + projection, + DCT filtering, and possibly alternative filter types. The appendix discusses these individually, but the main paper would benefit from a more explicit decomposition showing which components drive the gains.
    - An ablation against simpler temporal smoothing or momentum baselines, e.g., using the previous prediction, a moving average, or a direct multistep solver variant without the extra heuristics.
    - A comparison to established multi-step samplers or predictor-corrector approaches as direct baselines for the “use history at inference” idea.
    - Sensitivity to the history window length \(W\) is not really tested because the paper says it uses all past predictions with EMA anyway, which blurs the distinction.
- Are error bars / statistical significance reported?
  - No. This is a notable weakness for ICLR standards, especially because many of the reported gains are modest on some metrics (e.g., Table 1 on SDXL and SD3, Table 6 on samplers). Without variance, repeated runs, or confidence intervals, it is hard to know how robust the differences are.
- Do the results support the claims made, or are they cherry-picked?
  - The results are supportive overall, but there is some cherry-picking risk:
    - The paper emphasizes the strongest gains and the new SOTA FID in Table 3, while smaller or mixed results are less foregrounded.
    - In some settings, improvements are marginal (e.g., CLIP score in Table 5 is essentially unchanged), yet the narrative emphasizes broad quality improvements.
    - The method appears to rely on tuning several hyperparameters per model; the claim of universal plug-and-play improvement would be more credible with a more standardized tuning protocol or held-out tuning set.
- Are datasets and evaluation metrics appropriate?
  - Generally yes. ImageNet FID/IS/precision/recall is standard for class-conditional generation. HPSv2 and ImageReward are acceptable for text-to-image, though both are proxy metrics and should not be treated as definitive human preference measures.
  - One concern: the paper states HPSv2 is “most aligned with human judgment,” which may be true in some settings, but this should be presented more cautiously. Also, the use of COCO validation pairs for text-to-image evaluation is standard but not necessarily sufficient to support broad claims about prompt adherence across open-ended prompts.

### Writing & Clarity
- Are there sections that are confusing or poorly explained?
  - Yes, mainly in the method and appendix:
    - The connection between the theoretical motivation and the actual algorithm is not clean enough. Readers have to mentally bridge from a simplified gradient-descent analogy to a fairly complex heuristic pipeline.
    - The role of projection and DCT filtering is not integrated into the main conceptual story; they feel like corrective heuristics appended after the core idea.
    - The theoretical appendix is difficult to relate to the actual empirical algorithm.
- Are figures and tables clear and informative?
  - The intended figures/tables are useful, but several are difficult to interpret from the extracted text because they rely on visual layouts. Ignoring parser artifacts, the paper would still benefit from clearer exposition of what each figure is testing and how to read the comparisons.
  - More importantly, the tables could be structured to make the exact baseline settings, step counts, and hyperparameter choices more transparent. This matters because the method is sensitive to several knobs.

### Limitations & Broader Impact
- Do the authors acknowledge the key limitations?
  - Only partially. The discussion admits that HiGS still inherits some biases and limitations of the base model, but this is generic.
- Are there fundamental limitations they missed?
  - Yes:
    - The method appears to require tuning several hyperparameters per model and per regime (\(w_{\text{HiGS}}, \eta, \alpha, t_{\min}, t_{\max}, R_c\)), which may limit true “plug-and-play” robustness.
    - The method is empirically focused on image generation; its claim of general diffusion-model enhancement is not validated on other modalities, which the paper’s broad language sometimes suggests.
    - The method may be less helpful or even harmful when the base model is already highly optimized or when the sampler is itself multi-step and accurate.
- Are there failure modes or negative societal impacts not discussed?
  - The broader impact statement is reasonable but generic. It does mention misuse and synthetic content concerns.
  - More specific limitations around bias amplification, misinformation, and the potential for stronger photorealism to increase deceptive content creation would be worth acknowledging more concretely, especially since the method’s purpose is to improve realism.

### Overall Assessment
HiGS is a plausible and potentially useful training-free sampling enhancement for diffusion models, and the empirical gains across several image-generation settings are real enough to make the paper interesting for ICLR. That said, the paper currently overstates the universality and theoretical grounding of the method relative to the evidence. The core idea is intuitive, but the full algorithm relies on multiple heuristic components (history EMA, schedule, projection, DCT filtering) whose necessity and interactions are not cleanly isolated, and the theoretical appendix does not justify the actual implemented method. The strongest empirical result—the 1.61 unguided ImageNet FID—is notable, but the paper would need stronger ablations, uncertainty reporting, and a more careful framing of generality to meet ICLR’s bar for a robust, broadly convincing contribution.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes HiGS, a training-free plug-and-play modification to diffusion sampling that incorporates a weighted history of past model predictions as an additional guidance signal. The method is presented as a momentum-like correction for Euler-style sampling, and the authors combine it with optional projection and DCT-based filtering to reduce artifacts. Empirically, they report improved perceptual quality and FID/HPS-type metrics across several diffusion models, claiming a new unguided ImageNet FID of 1.61 with SiT-XL + REPA at 30 steps.

### Strengths
1. **Practical, plug-and-play framing with no retraining**
   - The core method is explicitly designed to work with pretrained diffusion models and existing samplers, requiring no extra training or fine-tuning.
   - The paper repeatedly emphasizes negligible overhead and compatibility with multiple architectures and samplers, including Stable Diffusion variants, DiT/SiT, Flux, and distilled models.

2. **Broad empirical coverage**
   - The experiments span text-to-image and class-conditional ImageNet generation, plus compatibility checks with distilled samplers and alternative CFG variants.
   - The paper reports results across multiple metrics: FID, IS, precision/recall, HPSv2, ImageReward, win rate, and CLIP score, which is helpful for assessing both fidelity and human-preference alignment.

3. **Strong reported gains in low-step / low-guidance settings**
   - The main claim is supported by figures and tables showing improvements at reduced NFEs and lower CFG scales, which is exactly where diffusion sampling quality typically degrades.
   - The reported reduction from 250 to 30 steps for unguided ImageNet generation, while improving FID from 1.83 to 1.61, is a potentially notable efficiency result if robustly validated.

4. **Ablation-oriented presentation**
   - The appendix includes ablations on buffer input, DCT filtering, projection, weight scheduling, EMA parameter, DCT threshold, and alternative history functions.
   - This helps justify some design choices beyond the initial method idea.

5. **Clear attempt to connect to optimization intuition**
   - The paper offers an optimization-inspired motivation via momentum/STORM analogies and an error-analysis appendix claiming improved local truncation error.
   - Even if the theory is not fully rigorous, this provides a conceptual lens that is more substantial than a purely heuristic sampler tweak.

### Weaknesses
1. **Novelty is somewhat incremental relative to existing guidance/sampling refinements**
   - The method is essentially a history-averaging correction applied to sampling updates, combined with schedule tuning and optional projection/filtering.
   - The paper itself cites prior work on guidance modification, multistep solvers, APG, autoguidance, and CFG-related improvements, and HiGS feels like a composition of known ideas rather than a clearly distinct new principle.
   - For ICLR, which values conceptual and methodological novelty, the contribution may be seen as a useful heuristic rather than a major algorithmic advance.

2. **The theoretical analysis is not convincing enough for the claims made**
   - The appendix claims improved local truncation error from \(O(h_k^2)\) to \(O(h_k^3)\) and global error from \(O(h)\) to \(O(h^2)\), but this derivation relies on a specific choice of weight \(w_k\) and appears disconnected from the full HiGS procedure with EMA, projection, DCT filtering, and nonlinear scheduling.
   - The proof is stated at a high level and does not convincingly establish that the actual implemented algorithm enjoys the claimed higher-order behavior.
   - For ICLR standards, the analysis is suggestive but not rigorous enough to substantiate the headline theoretical interpretation.

3. **A significant portion of the method is heuristic and under-justified**
   - The final algorithm includes not just history guidance, but also orthogonal projection, DCT high-pass filtering, and a time-dependent schedule.
   - These components are motivated empirically, but the paper does not clearly explain why these particular choices are necessary, nor does it provide a principled unification of them.
   - The method’s simplicity is somewhat reduced by these extra knobs, which makes the “simple, universal enhancement” framing less clean.

4. **Evaluation is strong in breadth but weaker in rigor**
   - The paper reports many metrics and claims consistent improvement, but it does not sufficiently discuss statistical significance, run-to-run variance, or the sensitivity of results to exact hyperparameter choices beyond a few appendix plots.
   - Several reported gains are modest on some benchmarks, and in a few tables the improvements are small enough that robustness matters.
   - It is unclear whether all comparisons are equally fair across models and samplers, especially when different models use different tuned parameters for HiGS.

5. **Dependence on nonstandard quality metrics for text-to-image**
   - The paper relies heavily on HPSv2 and ImageReward, which are useful but imperfect proxies for human preference.
   - The use of these metrics is reasonable, but the paper does not provide user studies or stronger human evaluation to corroborate the claimed visual improvements.

6. **Reproducibility is helped by appendices, but still not fully complete**
   - The paper does provide algorithmic pseudocode and parameter tables, which is good.
   - However, exact implementation details of the DCT masking, projection operator, and integration into each base sampler could be clearer, and the method appears to require several tuned hyperparameters per setting.
   - The code-like snippets in the extracted text suggest possible implementation ambiguity, though this may partly reflect parsing artifacts.

### Novelty & Significance
**Novelty:** Moderate to low-moderate. HiGS is a sensible and potentially effective sampling heuristic, but it does not appear to introduce a fundamentally new class of diffusion model or sampling theory. The main idea—using past predictions as a momentum-like correction—is intuitive and related to prior multistep and guidance-stabilization methods.

**Significance:** Moderate. If the empirical gains hold up, especially the step reduction for ImageNet and the consistent improvements across multiple pretrained models, this could be practically useful for the diffusion community. However, for ICLR’s acceptance bar, the paper would likely need a stronger case that the method is both conceptually novel and broadly reliable beyond a well-tuned heuristic.

**Clarity:** Fair. The overall narrative is understandable, but the method becomes harder to parse once projection, scheduling, and DCT filtering are introduced. The paper would benefit from a cleaner separation between the core idea and optional refinements.

**Reproducibility:** Fair to good. The appendix includes pseudocode and hyperparameter tables, which is a plus. Still, the number of tunable components and the reliance on model-specific settings reduce confidence that the gains are effortless to reproduce.

### Suggestions for Improvement
1. **Strengthen the theoretical grounding**
   - Provide a rigorous derivation showing exactly under what assumptions the history correction improves order, and clarify whether this applies to the implemented EMA/projection/DCT version or only to an idealized simplification.
   - Distinguish clearly between intuition, local error analysis, and provable guarantees.

2. **Isolate the contribution of each component**
   - Add a cleaner ablation that separates: history term alone, history + schedule, history + projection, history + DCT filtering, and the full method.
   - This would reveal whether the core idea is truly sufficient or whether most gains come from auxiliary heuristics.

3. **Improve fairness and robustness analysis**
   - Report mean and standard deviation over multiple seeds for the major benchmarks.
   - Include significance testing or confidence intervals where feasible, especially for metrics with small margins.
   - Clarify whether hyperparameters were tuned per model, per benchmark, or transferred across settings.

4. **Provide stronger human-centric evaluation**
   - Since the paper emphasizes perceptual improvements, add a small human preference study or pairwise human ranking experiment.
   - This would better support the claims than relying mainly on learned preference metrics.

5. **Clarify implementation details**
   - Specify exactly how HiGS is inserted into each sampler family and what is stored in the history buffer.
   - Give a concise, model-agnostic implementation recipe for CFG, unguided, and distilled settings.
   - Clarify default hyperparameter choices and how sensitive performance is to them.

6. **Refine the method narrative**
   - Present HiGS as a core history-based update with optional stabilizers, rather than as a single monolithic method.
   - This would make the paper easier to understand and help readers identify what is essential versus optional.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a comparison against the strongest recent low-step samplers and guidance modifiers on the same backbones, not just a few selected baselines. For ICLR, the claim that HiGS is a “universal enhancement” is not convincing without direct head-to-head results versus methods like UniPC, DPM-Solver++ variants, APG, guidance-interval sampling, autoguidance, and modern distillation baselines under matched step budgets and CFG settings.

2. Add a real ablation that isolates each claimed component of HiGS on the same benchmark: history only, history + schedule, history + projection, history + DCT, and full method. Without this, it is unclear whether the gains come from the history mechanism or from ad hoc post-processing/filtering tricks.

3. Add experiments showing whether HiGS helps beyond the specific pretrained models shown, especially on non-Stable-Diffusion architectures and non-image tasks if the paper claims generality. The method is framed as plug-and-play for diffusion models, but the evidence is mostly image generation with a narrow set of backbones; ICLR reviewers will expect broader verification before accepting a universality claim.

4. Add sampling-quality vs compute curves with wall-clock time and memory for all main settings, not just a single throughput measurement. The paper claims “practically no additional computation,” but the method introduces EMA history, projection, and DCT operations; these need a concrete cost/quality tradeoff comparison to justify the efficiency claim.

5. Add a human preference study or at least a stronger evaluation protocol for text-to-image results beyond HPSv2/ImageReward win rates. Relying mainly on learned reward metrics makes the improvement less trustworthy, especially when the paper claims consistent visual quality gains across prompts and models.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze when HiGS fails or degrades samples, especially at high guidance scales, very low step counts, and different samplers. The paper states the method is robust, but the missing failure analysis makes it hard to trust the claimed stability across “all settings.”

2. Quantify sensitivity to the hyperparameters that define the method’s behavior: EMA coefficient, history length, schedule shape, projection weight, and DCT cutoff. The current discussion is mostly empirical and local to SDXL; ICLR reviewers will want to know whether the method actually generalizes or just requires careful tuning per model.

3. Provide a clearer mechanistic analysis of why the history term should improve diffusion sampling rather than just showing it empirically. The current energy/SGD analogy and truncation-error argument are not enough to establish that the update improves sample quality in the actual diffusion dynamics used by modern samplers.

4. Analyze whether HiGS changes diversity, calibration, or prompt adherence in a measurable way, not just fidelity-style scores. If the method sharpens images by pushing samples toward “more realistic” regions, it may also reduce diversity or alter semantic alignment; that tradeoff is not sufficiently examined.

5. Validate the numerical claims in the theory section with experiments tied to the analysis. The paper asserts improved local/global error order, but no experiment tests convergence behavior versus step size or compares predicted error reduction against actual sample quality trajectories.

### Visualizations & Case Studies
1. Show per-step trajectories of a few generated samples, comparing baseline vs HiGS at intermediate timesteps. This would reveal whether HiGS genuinely corrects structure over time or merely changes the final output appearance.

2. Visualize the history signal, the projection component, and the DCT-filtered update on representative examples. Without these diagnostics, it is impossible to tell whether the method is doing meaningful guidance or just suppressing visually obvious artifacts after the fact.

3. Include failure case galleries where HiGS worsens results, especially for complex prompts, fine-grained textures, and high-saturation scenes. This would expose whether the method is robust or only improves a subset of easy cases.

4. Add qualitative comparisons at matched compute budgets against the strongest baselines, not just against plain CFG. The current figures mostly show “baseline vs ours,” which is insufficient to establish that the method is actually better than existing sampling improvements.

### Obvious Next Steps
1. Benchmark HiGS as a drop-in module for the current best diffusion samplers and distillation pipelines under standardized settings. That is the most direct way to test whether the method is genuinely useful rather than narrowly tuned to the authors’ preferred samplers.

2. Extend the evaluation to additional tasks where diffusion sampling quality matters, such as image editing or inpainting. If the paper claims plug-and-play generality, it should demonstrate that the history mechanism transfers beyond text-to-image and class-conditional synthesis.

3. Replace or supplement reward-model-based evaluation with stronger human-aligned evaluation protocols. For an ICLR submission, claims about perceptual quality need evidence that is less dependent on proxy metrics.

4. Test whether a simpler baseline can explain most of the gains, such as longer EMA smoothing, prediction averaging, or a standard multistep solver with matched compute. If a simpler alternative closes the gap, the method’s contribution is much weaker than claimed.

# Final Consolidated Review
## Summary
This paper proposes HiGS, a training-free, plug-and-play modification to diffusion sampling that uses a weighted history of past model predictions as an additional correction term. The authors argue that this history-based momentum stabilizes sampling and improves image quality, especially at low NFE or low CFG, and they report gains across several image-generation backbones, including a new unguided ImageNet FID result on a specific SiT checkpoint.

## Strengths
- **Training-free and broadly applicable in principle.** The core idea is easy to insert into existing samplers without retraining, and the paper demonstrates compatibility across multiple diffusion backbones and sampler families, including Stable Diffusion variants, DiT/SiT, distilled models, and several solver choices.
- **Empirical gains are real in the regimes the paper targets.** The strongest evidence is in low-step and low-guidance settings, where the qualitative improvements are visible and the quantitative results generally move in the right direction; the reported 1.61 FID on unguided ImageNet with 30 steps is the most notable result.

## Weaknesses
- **The method is overloaded with heuristics, not a clean single contribution.** HiGS is not just “history-guided sampling”; the final system also adds a time-dependent schedule, optional orthogonal projection, and DCT high-pass filtering. These pieces are presented as crucial in practice, but the paper does not cleanly isolate which part actually drives the gains, so the core idea is harder to assess than the narrative suggests.
- **The theory does not justify the actual algorithm.** The appendix gives an error-analysis story for a simplified one-step history update with a special weight choice, but the implemented method uses EMA history, scheduling, projection, and filtering. The claimed jump from first-order to second-order global accuracy is therefore not established for the real HiGS procedure and reads more like intuition than proof.
- **The “universal enhancement” framing is overstated.** The experiments are helpful but still limited to a relatively narrow slice of image-generation backbones and evaluation protocols. The paper’s language suggests broad generality, but the evidence is not enough to support such a strong claim.
- **Robustness and fairness are under-quantified.** The paper reports many single-number improvements, but provides no variance, confidence intervals, or repeated-seed analysis. That is especially problematic because some gains are modest and the method has several tunable hyperparameters that appear to be set per model/regime.

## Nice-to-Haves
- A cleaner main-paper ablation that separates: history term only, + schedule, + projection, + DCT, and full HiGS.
- More explicit reporting of tuning protocol and sensitivity for \(w_{\text{HiGS}}, \alpha, \eta, t_{\min}, t_{\max}, R_c\).
- A small human preference study or stronger user-facing evaluation for the text-to-image claims.

## Novel Insights
The most interesting aspect of the paper is not the final HiGS pipeline itself, but the observation that diffusion sampling can be viewed as a trajectory that benefits from temporal correction using prior predictions, much like momentum in optimization or a multistep solver. That lens helps explain why the method tends to help most when the sampler is under-resolved, where successive predictions are noisy and the current estimate can be nudged toward a more stable direction. However, the paper blurs the line between this core insight and a collection of empirical fixes, so the conceptual contribution is stronger than the methodological one.

## Potentially Missed Related Work
- **Autoguidance / bad-version-of-itself guidance** — relevant because it also uses weaker or earlier predictions to shape guidance, and the paper’s own appendix explicitly positions HiGS near this line of work.
- **Guidance interval / limited-interval CFG** — relevant because the paper uses a time-scheduled guidance window and the comparison would help separate “history helps” from “schedule helps.”
- **Modern multistep samplers and predictor-corrector methods** — relevant because HiGS’s historical correction is closely adjacent to multistep numerical integration ideas.

## Suggestions
- Provide a single, explicit ablation table on the main benchmark that shows how much each HiGS component contributes on its own.
- Add multiple-run statistics or confidence intervals for the main quantitative claims.
- Tighten the theory section: either prove something about the actual implemented algorithm, or clearly label the current analysis as a heuristic motivation only.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
