# Score Replacement with Bounded Deviation for Rare Prompt Generation

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Diffusion models achieve impressive performance in high-fidelity image generation but often struggle with rare concepts that appear infrequently in the training distribution. Prior work attempts to address this issue by prompt switching, where generation begins with a frequent proxy prompt and later transitions to the original rare prompt. However, such designs typically rely on fixed schedules that disregard the model’s internal dynamics, making them brittle across prompts and backbones. In this paper, we re-frame rare prompt generation through the lens of score replacement: the denoising trajectory of a rare prompt can be initially guided by the score of a semantically related frequent prompt, which acts as a proxy. However, as the process unfolds, the proxy score gradually diverges from the true rare prompt score. To control this drift, we introduce a bounded deviation criterion that triggers the switch once the deviation exceeds a threshold. This formulation offers both a principled justification and a practical mechanism for rare prompt generation, enabling adaptive switching that can be widely adopted by different models. Extensive experiments across SDXL, SD3, Flux, and Sana confirm that our method consistently improves rare concept synthesis, outperforming strong baselines in both automated metrics and human evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on image generation under rare concept descriptions. Unlike previous methods that adopt a fixed prompt-switching time, the authors propose an adaptive prompt-switching strategy. Specifically, the method pre-calculates a threshold to control when the prompt should switch. Experimental results demonstrate strong performance and good generalization across various models.

### Strengths
1. This paper first provides a mathematical analysis of semantic drift (or semantic difference) that occurs during prompt switching.

2. Based on the mathematical analysis, this paper further proposes an adaptive prompt-switching rule that generalizes well across various generation models.

3. The experiment results show the great performance and robustness.

### Weaknesses
1. The paper still relies on prompt switching to address the rare concept generation problem, and the novelty of the proposed approach is therefore limited.

2. The bucket threshold needs to be pre-calculated, which is time-consuming.

3. The bucket threshold is calculated from the dataset. Therefore, it is unclear whether the pre-calculated threshold can generalize to rare concepts that are not present in the dataset.

4. It is still complex to calculate the threshold under various settings, as stated in the paper: “Finally, since semantic categories (e.g., shape and texture) often induce distinct thresholds…”. This remains a practical problem in real-world scenarios.

### Questions
Do we need to pre-calculate different thresholds for each model when using the same dataset?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper tackles rare-prompt text-to-image generation by improving the framework, which utilizes prompt switching. Specifically, the authors reframe "prompt switching" as score replacement: early in sampling, they guide the denoiser with a semantically related, frequent proxy prompt whose score closely matches the rare prompt’s score, then switch to the rare prompt once that proxy starts to drift. They measure the score difference over timesteps and introduce RAP, which is equipped with a heuristic rule with per-segment budget that bounds the accumulated deviation from the rare-prompt trajectory. When the budget is exceeded, they switch to the rare prompt. The bucket threshold is estimated from an early stable regime of score differences, and a decay factor allocates larger budgets to prompts closer to the rare target, making the schedule model-agnostic and prompt-aware.

### Strengths
- The paper is well written and systematically analyzes the limitations of existing rare-prompt switching pipelines. The proposed controller addresses those issues cleanly and yields consistent improvements across models.

### Weaknesses
- The method's effectiveness may lean on several choices and hyperparameters that feel more engineered than principled. The bucket threshold is estimated from an empirically detected stable region, which can depend on the backbone generative model. Category- and model-specific thresholds further improve results but also signal configuration fragility: gains partly come from per-benchmark tailoring rather than a single robust rule.

- Minor: While the paper substantially strengthens R2F with a principled, adaptive controller, the contribution is still anchored to the R2F formulation and evaluation protocol. As a result, the impact feels scoped to a niche use case (rare-prompt switching) rather than a broadly applicable T2I strategy.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses an interesting problem, diffusion models often struggle on text-to-image generation when guided by rare prompts.
To tackle this, it proposes RAP (Rare-to-Adapt via score replacement).
Specifically, early denoising uses the score of a semantically related, frequent proxy prompt to stabilize generation, then adaptively switches to the rare prompt once an estimated score-deviation budget is exceeded.
The paper further derives a theoretical bound linking final-sample deviation to cumulative score differences along the trajectory, and instantiates a practical bucketed switching rule whose budget decays with the text similarity between the proxy and rare prompts.
Extensive experiments across multiple diffusion backbones show consistent gains over prompt-switching baselines (e.g., R2F) on rare-prompt benchmarks and in human-preference studies.

### Strengths
1. It reframes prompt switching as a score-aware control problem, introducing an adaptive score-replacement trigger instead of brittle, fixed schedules.
2. A clear theoretical bound ties final-sample deviation to accumulated score differences, giving principled guidance for when to switch prompts.
3. The method is practical and backbone-agnostic, requiring no architectural changes or extra training, and works across SDXL/SD3/Flux/Sana. Empirically, it consistently outperforms R2F on rare-prompt benchmarks and is supported by human preference studies.

Overall, it delivers a significant, deployable improvement for rare-prompt fidelity while keeping implementation complexity low.

### Weaknesses
1. Score-difference tracking and similarity-based budgeting introduce computational overhead. However, the paper does not quantify this cost—e.g., wall-clock time, GPU-hours, and memory. Therefore, comparing these metrics against prompt-switching baselines is necessary.

2. In light of the possible brittleness of early-regime budgeting across models and prompts, please report robustness evidence or adaptation strategies spanning backbone changes, prompt diversity, and noise-schedule shifts.

3. The main paper acknowledges dependence on proxy-prompt quality and the chosen similarity metric, but sensitivity analyses for proxy/encoder choices are missing, and there is no automatic fallback for low-similarity scenarios.

### Questions
1. How are proxy prompts chosen when multiple candidates exist or similarity is low? How sensitive is RAP to the choice of text encoder (CLIP vs. T5) and similarity metric?

2. When does RAP hurt fidelity (e.g., very low similarity, proxy semantically drifts)? Can the method detect such cases on the fly?

3. What is the overhead of score-difference tracking and similarity computation at inference time?

4. Are hyperparameters transferable across SDXL/SD3/Flux/Sana, or re-tuned per model?

5. How interpretable is the switching timeline to practitioners?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles rare-concept text-to-image generation, focusing on the limitations of R2F’s fixed switching strategy between rare and frequent prompts. The authors propose an adaptive proxy-prompt scheduling method based on a score-approximation perspective, arguing that the denoising scores of rare and frequent prompts follow similar trajectories. By monitoring score displacement during diffusion, the method adaptively transitions from the frequent proxy to the rare target. Various experiments are presented to demonstrate the effectiveness of the method.

### Strengths
- **Adaptive scheduling.** The paper replaces R2F’s fixed, heuristic switching rule with an adaptive scheduling using the score of the rare prompt.

- **Comprehensive analyses.** It provides diverse supporting analyses, including score-trajectory visualization, cross-model comparisons, and derivations that validate the method’s design.

- **Theoretical grounding via score approximation.** The method offers a theoretical perspective that connects the adaptive switching behavior to the score approximation, improving interpretability.

### Weaknesses
- **Limited methodological novelty.** While the paper’s attempt to make R2F’s fixed strategy adaptive is meaningful, the method itself is not substantially novel and largely builds on existing switching ideas.

- **Optimality of switching approach.** It remains unclear whether prompt switching is an optimal formulation for rare-concept generation. Since this strategy inherently depends on finding a proxy prompt, it may be less efficient or generalizable than other rare-concept generation methods. (more on Questions)

- **Marginal improvements** Both quantitative results on T2I-Combench and qualitative findings from the user study show only marginal improvements over prior methods, suggesting limited practical benefit despite the adaptive formulation.

- **Increased computational cost.** Because the adaptive schedule requires pre-computing or estimating scores for multiple timesteps, it likely incurs higher computational overhead than R2F’s fixed switching strategy, reducing its practical efficiency.

### Questions
- As noted in the weaknesses, the paper does not show that prompt switching is an optimal or necessary formulation for rare-concept generation. Could you provide additional comparisons with other methods [1, 2], or theoretical reasoning that supports the optimality (or sufficiency) of the switching-based formulation? 
- Derivation around line 647. It appears that line 647 is derived by recursively applying Eqs. (12) and (13). In this case, the superscript of ​
x_t  on the right-hand side of Eq. (13) should become C_(*|R). Could you provide a clear step-by-step derivation to confirm this?
- Please provide additional details about the score-displacement experiments: what prompts were used, how many samples were averaged.
- The authors mention category-specific thresholds, but such categorical thresholding appears to require predefined concept sets, which might limit the method’s general applicability. Is this understanding correct? Additionally, beyond the SANA results, are there experiments on other models showing the impact of these thresholds?

If the authors address these concerns appropriately, I would be inclined to raise my rating.

---
#Reference
[1] Unleashing the diversity of diffusion models through condition-annealed sampling. 
[2] Minority-Focused Text-to-Image Generation via Prompt Optimization

### Soundness
3

### Presentation
2

### Contribution
2
