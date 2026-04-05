=== CALIBRATION EXAMPLE 58 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate in signaling a CLIP-based, prompt-driven, training-free multi-label method, but it slightly overstates novelty with “Unlocking the Power of Co-Occurrence” given that the core idea is a relatively direct use of co-occurrence prompts plus calibration.
- The abstract clearly states the problem, the prompt-based method, and the main empirical claim.
- However, some claims are stronger than the evidence presented in the paper:
  - “unlocks the power of co-occurrence” is more of an interpretation than a demonstrated general principle.
  - “can achieve better performance than the state-of-the-art methods” is supported on the reported benchmarks, but the abstract does not mention that the method depends on a chosen co-occurrence source (ChatGPT or a small amount of data), which is a substantive part of the method and affects reproducibility and generality.
- The abstract also frames the causal explanation somewhat confidently, but the paper’s causal derivation is more heuristic than rigorously established.

### Introduction & Motivation
- The problem is well-motivated: zero-shot multi-label classification with CLIP is indeed underexplored relative to single-label CLIP use, and the gap between CLIP’s training objective and multi-label inference is a real concern.
- The paper identifies two relevant limitations in prior work:
  1. Existing CLIP inference ignores label co-occurrence.
  2. Prior multi-label CLIP methods such as TagCLIP rely on ViT-specific local features and are less universal.
- The contribution statements are understandable and mostly accurate: correlative prompts, dual prompts, and a causal perspective. That said, the introduction somewhat over-credits co-occurrence as the key missing ingredient while downplaying the strong role of localization and patch-level evidence in multi-label recognition. For ICLR-level standards, the paper should better position co-occurrence relative to other mechanisms.
- The introduction’s main question is interesting, but the paper is somewhat one-sided in presenting co-occurrence as the central deficiency of CLIP without sufficiently acknowledging alternative explanations for multi-label failure, such as thresholding, calibration, prompt sensitivity, and weak localization.

### Method / Approach
- The method is understandable at a high level: construct a correlative prompt (CoP) by appending co-occurring labels to the target label prompt, then combine CoP and discriminative prompt (DiP) outputs to form DualPrompt.
- Reproducibility is not fully sufficient:
  - The exact procedure for selecting co-occurring labels is underspecified. The paper says ChatGPT-4o can generate up to \(l\) labels and that a small subset of training data can estimate co-occurrence probabilities, but it does not fully specify ranking, thresholding, tie-breaking, or prompt construction details.
  - The exact inference rule for turning the combined score in Eq. (2) into multi-label decisions is not clearly described. For multi-label tasks, thresholding matters materially.
- There is a logical gap in the causal argument:
  - Eq. (1) and Eq. (2) are presented as causal quantities, but the transition from a TDE-style subtraction to an additive combination in Eq. (2) is not rigorously justified as equivalent in the usual causal-inference sense.
  - The derivation in Appendix A appears to rely on algebraic manipulation and a proportionality argument with a trade-off parameter \(\lambda\), but the exact assumptions under which the final additive form follows are not clearly stated.
- The “object hallucination” interpretation is plausible, but the paper does not convincingly isolate whether false positives come specifically from co-occurrence overfitting rather than from general prompt bias or class prior effects.
- A major methodological concern is that the approach depends on external co-occurrence knowledge. This makes the method less “training-free” in a strong sense, because one still needs either LLM-generated priors or labeled data statistics.
- Another important issue: the paper claims universality across backbones, but experiments are limited to ResNet-101 and ViT-B/16. That is a meaningful but still narrow test of universality.

### Experiments & Results
- The experiments do test the central claim that co-occurrence-aware prompting improves zero-shot multi-label CLIP performance.
- The baseline set is partially appropriate:
  - CLIP is a necessary baseline.
  - TagCLIP is a strong and relevant training-free baseline.
  - DualCoOp and TaI are reasonable training-based comparators.
- However, there are several concerns about fairness and completeness:
  - The paper compares a training-free method using external co-occurrence estimation with training-based methods that use 1%/2% labeled data, but the amount of supervision is not always directly comparable in utility or source. The paper discusses this, but the comparison still needs more careful framing.
  - The paper reports a “1% Data for Coo. Est.” setting, but it is not fully clear whether those examples are used only for statistics or whether any selection bias is involved.
- Missing ablations that would materially change conclusions:
  - No ablation on the number of co-occurring labels \(l\) beyond qualitative discussion.
  - No ablation showing performance as a function of co-occurrence quality/noise.
  - No ablation separating the effect of adding co-occurrence in prompts from the effect of the calibration formula in Eq. (2).
  - No ablation on whether a simpler logit-averaging or score-rescaling baseline could capture most of the gain.
- Statistical significance is not reported; no error bars or multiple seeds are provided, despite the fact that some gains are modest and potentially seed-sensitive.
- The reported gains are consistent and plausible on MS-COCO and VG-256, but the analysis is somewhat cherry-picked toward favorable examples:
  - Figure 2 focuses on top-10 gains and losses.
  - Figure 5 and Figure 7 are illustrative, but not enough to establish robust behavior across the label space.
- The additional Objects365 results in Appendix C are useful, but this dataset is only lightly reported and not integrated into the main experimental narrative.
- Evaluation metrics are standard and appropriate for multi-label classification, but the paper should more clearly explain how thresholds are chosen for F1, since this can affect comparisons substantially.

### Writing & Clarity
- The overall paper structure is understandable, and the main idea is communicated reasonably well.
- That said, several parts are conceptually unclear:
  - The causal explanation in Section 4 and Section 5 is difficult to follow, especially the distinction between TDE, direct effect, mediated effect, and the final additive DualPrompt form.
  - Figure 3 is conceptually helpful, but the text does not fully translate the graph into a precise mathematical statement.
  - Figure 4 explains the dual prompt framework visually, but the operational pipeline is still underspecified.
- Some figures/tables are informative:
  - Figure 2 gives intuitive evidence for both gains and hallucination.
  - Table 1 and Table 2 support the claimed benchmark improvements.
- However, the paper would benefit from clearer presentation of:
  - exact inference procedure,
  - co-occurrence construction,
  - thresholding for multi-label outputs,
  - and the relation between DiP, CoP, and DualPrompt at the score level.
- The paper is generally readable, but the ambiguity around the method’s exact computation is a real clarity issue because it impedes reproducibility and evaluation of the contribution.

### Limitations & Broader Impact
- The paper does acknowledge one major limitation: co-occurrence can be noisy and can cause object hallucination.
- It also implicitly acknowledges that co-occurrence estimation quality matters, since ChatGPT-generated labels can be incorrect and small-data estimation is preferred.
- However, the paper does not sufficiently discuss broader limitations:
  - Dependence on dataset-specific label correlations may harm transferability across domains with weak or changing co-occurrence structure.
  - The method may amplify spurious societal correlations if used in sensitive domains.
  - Prompting with co-occurring labels could systematically bias predictions toward context-heavy stereotypes rather than actual object presence.
- A more serious limitation is that the method assumes reasonably stable and meaningful label co-occurrence statistics. In datasets or domains where co-occurrence is weak, rare, or adversarial, the method may not help and may hurt.
- The paper does not discuss failure modes under distribution shift, long-tail labels, or rare-object scenarios, all of which are especially relevant for CLIP-style open-vocabulary recognition.

### Overall Assessment
This is a thoughtful and relevant paper with a clear idea: use co-occurrence-aware prompts to improve training-free zero-shot multi-label CLIP, then calibrate away the resulting hallucination. The empirical results on MS-COCO, VG-256, and Objects365 are promising and suggest that the method is genuinely useful. However, for ICLR’s standard, the paper is held back by an under-specified method, a somewhat heuristic causal derivation, and insufficient ablation/statistical analysis to fully justify the claimed generality and robustness. The contribution likely stands as a practical prompt-engineering improvement, but the paper does not yet fully establish the deeper methodological claims it makes about causal effect decomposition and universal co-occurrence exploitation.

# Neutral Reviewer
## Balanced Review

### Summary
This paper studies zero-shot multi-label classification with CLIP and argues that the main missing ingredient is explicit label co-occurrence. The proposed DualPrompt method combines a discriminative prompt with a correlative prompt that appends co-occurring labels, then calibrates the resulting scores to mitigate hallucination; the paper claims this yields consistent gains over CLIP, TagCLIP, and other baselines on MS-COCO, VG-256, and Objects365.

### Strengths
1. **Clear and relevant problem framing for ICLR**
   - The paper targets an important gap: CLIP is strong for single-label zero-shot classification but weaker for multi-label settings.
   - The motivation is well aligned with ICLR interests in foundation models, prompt-based adaptation, and zero-shot generalization.

2. **Simple method with intuitive mechanism**
   - DualPrompt is conceptually straightforward: use both a discriminative prompt and a correlative prompt, rather than learning extra modules.
   - This simplicity is appealing for zero-shot settings because it avoids training-heavy adaptation.

3. **Good empirical evidence across multiple datasets**
   - The paper reports results on MS-COCO, VG-256, and Objects365, which helps demonstrate broader applicability.
   - It also evaluates different backbones, including ResNet-101 and ViT-B/16, and compares against both training-free and training-based baselines.

4. **Ablation-style analyses support the central claim**
   - The paper shows that co-occurrence can help on some classes but hurt on others, matching the thesis that naive co-occurrence use has both positive and negative effects.
   - Figure 5 and the retrieval visualizations are useful for illustrating the difference between discriminative, correlative, and dual prompting behavior.

5. **Practical consideration of co-occurrence estimation**
   - The paper does not assume a large labeled auxiliary set is always available; it considers using a small fraction of training data or ChatGPT-generated co-occurring labels.
   - This makes the method more practical than approaches that require substantial fine-tuning.

### Weaknesses
1. **Novelty is moderate rather than strong by ICLR standards**
   - The core idea is an inference-time combination of two prompts plus a score calibration step.
   - Similar themes already appear in prior CLIP prompting and causal-effect-based calibration work, and the paper’s main advance seems to be adapting them to multi-label co-occurrence rather than introducing a fundamentally new learning paradigm.

2. **The causal formulation feels under-justified**
   - The paper presents a causal mediation-style interpretation, but the assumptions behind the causal graph are not convincingly validated.
   - The derivation appears more heuristic than principled, especially the move from a subtraction formulation to an additive one with \(\lambda=1\), which is claimed to work empirically rather than derived rigorously.

3. **Reproducibility is incomplete for an ICLR submission**
   - Key implementation details are missing or underspecified, such as exactly how co-occurring labels are selected from ChatGPT outputs or from the small labeled subset.
   - The text does not fully specify thresholding, prompt construction details, hyperparameters, or how performance is computed from the two prompt scores.

4. **Experimental comparison is not fully convincing**
   - The paper compares against strong baselines, but many baselines use different assumptions and resources, making fairness hard to judge.
   - For example, some methods use training data for adaptation while DualPrompt uses only a tiny subset for co-occurrence estimation; this is useful, but the comparison would benefit from more controlled settings and standardized resource budgets.

5. **The evaluation is somewhat narrow**
   - The main benchmarks are all object-centric multi-label datasets, which is sensible, but they may favor co-occurrence-heavy reasoning.
   - There is limited evidence on datasets where co-occurrence is weaker, noisy, or more long-tailed, so it is unclear how robust the method is outside these settings.

6. **Claims about “object hallucination” are suggestive but not fully established**
   - The paper provides qualitative and aggregate evidence, but not a rigorous analysis of calibration, false positive rates, or hallucination under controlled counterfactual settings.
   - The explanation that the correlative prompt causes hallucination is plausible, but the causal evidence is not definitive.

### Novelty & Significance
**Novelty:** Moderate. The paper’s main contribution is a simple prompt-based strategy for injecting label co-occurrence into CLIP and then calibrating away over-reliance on that co-occurrence. This is a useful synthesis for zero-shot multi-label recognition, but it is not a clearly groundbreaking algorithmic advance by ICLR standards.

**Clarity:** Good overall. The motivation, high-level method, and empirical story are easy to follow. However, the causal derivation and the exact construction of co-occurring prompts are not explained with enough precision to be fully transparent.

**Reproducibility:** Fair to moderate. The paper provides datasets, some backbone choices, and high-level experimental settings, but it lacks enough detail on prompt generation, estimation of co-occurrence, and calibration specifics to ensure straightforward reproduction.

**Significance:** Moderately significant. If the reported gains hold under rigorous replication, this could be a practical improvement for zero-shot multi-label classification with CLIP. That said, the advance appears incremental rather than a major conceptual leap, which makes the work borderline relative to ICLR’s typical acceptance bar.

### Suggestions for Improvement
1. **Provide a much more detailed algorithmic specification**
   - Include exact prompt templates, how many co-occurring labels are used, how labels are ordered, whether labels are weighted, and how the final scores are combined.
   - Give a full pseudocode algorithm for inference.

2. **Strengthen the causal argument**
   - Clarify what assumptions justify the mediation-style interpretation.
   - Add a more rigorous analysis showing when the additive approximation is valid and when it may fail.

3. **Improve reproducibility of co-occurrence estimation**
   - Specify how the small labeled subset is sampled, how co-occurrence probabilities are computed, and how sensitive results are to the subset size.
   - For the ChatGPT route, report the exact prompting protocol and selection criteria.

4. **Add stronger calibration and robustness analysis**
   - Report false positive rate, precision/recall trade-offs, and calibration metrics to better substantiate the hallucination claim.
   - Test robustness across different thresholding rules and different numbers of co-occurring labels.

5. **Include more controlled baselines**
   - Compare against simpler alternatives such as prompt ensembling, label prior correction, and score averaging with co-occurrence priors.
   - This would help isolate whether DualPrompt’s gains come from the co-occurrence idea itself or from a generic ensembling effect.

6. **Broaden evaluation**
   - Add datasets where co-occurrence structure is weaker or noisier, or where labels are less object-centric.
   - This would better establish whether the method generalizes beyond co-occurrence-rich benchmarks.

7. **Clarify the resource comparison**
   - When comparing with training-based methods, explicitly separate gains due to using any labeled data from gains due to the prompt design.
   - A matched-budget comparison would make the results more convincing for ICLR reviewers.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a full comparison against stronger zero-shot multi-label baselines beyond CLIP/TagCLIP, especially recent open-vocabulary tagging methods and prompt-based zero-shot methods on the same datasets. Without this, the claim that DualPrompt is state-of-the-art is not convincing under ICLR standards.

2. Add a clean ablation separating the two core ingredients: correlative prompt alone, discriminative prompt alone, and their combination under the exact same co-occurrence source and backbone. Right now it is unclear whether gains come from co-occurrence prompting, the causal calibration, or simply using more text tokens.

3. Add ablations on the co-occurrence construction itself: number of co-occurring labels, noisy vs accurate co-occurrence, ChatGPT-generated vs dataset-estimated co-occurrence, and the effect of using wrong co-occurrence sets. The paper’s main claim depends on co-occurrence quality, so robustness to noise is essential.

4. Add experiments on additional zero-shot multi-label datasets with different label statistics and long-tail structure, not just COCO/VG/Objects365. The method is claimed to exploit a general property of CLIP, but the current evidence is too concentrated on object-centric datasets with similar co-occurrence patterns.

5. Add comparisons against a stronger “oracle” or supervised upper bound for co-occurrence estimation, e.g. using ground-truth co-occurrence from the test set only for analysis or a fully supervised co-occurrence estimator. Without this, it is hard to judge whether the proposed estimation is actually good enough or whether the method is bottlenecked by imperfect co-occurrence.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify when DualPrompt helps vs hurts by class frequency, object size, and co-occurrence strength. ICLR reviewers will want to know whether the method actually fixes hard multi-label cases or mainly improves easy frequent co-occurring classes.

2. Analyze hallucination directly with false positive/false negative decomposition, calibrated precision/recall, and class-wise error shifts. The paper claims to reduce object hallucination, but the evidence is mostly indirect and does not isolate hallucination from general thresholding or ranking effects.

3. Provide a principled justification for the “addition form” of the causal derivation, including assumptions under which Eq. (2) is equivalent to the TDE formulation. As written, the derivation looks more heuristic than theoretically grounded, which weakens the causal interpretation.

4. Measure sensitivity to prompt wording and prompt template choice. Since the method hinges on natural-language co-occurrence prompts, the paper needs to show that performance is not brittle to specific phrasing or template engineering.

5. Analyze the computational overhead and scaling with label set size. A method that requires per-class co-occurrence prompts and dual forward passes may become costly for larger label spaces, and this practical limitation matters for ICLR-level claims of generality.

### Visualizations & Case Studies
1. Show image-level success/failure examples where DiP fails due to missing co-occurrence, CoP hallucinates due to suspicious co-occurrence, and DualPrompt corrects both. This would directly test the paper’s central narrative instead of relying on aggregate metrics.

2. Visualize per-class precision-recall shifts and confidence distributions before vs after DualPrompt. This would reveal whether the method truly calibrates predictions or just changes ranking behavior in a way that benefits AP.

3. Show nearest-neighbor or retrieval examples for classes with high co-occurrence ambiguity, especially cases where the target label is absent but co-occurring labels are present. Those are the exact failure modes the paper claims to address.

4. Provide a t-SNE or similarity-map style visualization of text embeddings for DiP, CoP, and DualPrompt prompts. This would expose whether DualPrompt actually separates discriminative from correlative semantics or just blends them superficially.

### Obvious Next Steps
1. Formalize the method as a general zero-shot multi-label calibration framework and test whether the dual-prompt idea transfers to other VLMs beyond CLIP. ICLR would expect more than a CLIP-specific trick if the claim is about a general mechanism.

2. Develop an automatic co-occurrence mining method that does not require any labeled subset or hand-crafted ChatGPT outputs. The current dependence on dataset-specific co-occurrence estimation weakens the “training-free” claim.

3. Extend the method to open-vocabulary detection or segmentation, where co-occurrence is even more important and hallucination is easier to measure. That would be the most natural next step if the causal prompt idea is genuinely useful.

# Final Consolidated Review
## Summary
This paper tackles zero-shot multi-label classification with CLIP, arguing that vanilla CLIP underuses label co-occurrence and therefore misses labels or misranks them. The proposed DualPrompt method combines a discriminative prompt with a correlative prompt that injects co-occurring labels, then calibrates the resulting scores to reduce co-occurrence-driven false positives. The idea is simple and the reported gains on MS-COCO, VG-256, and Objects365 are consistent, but the method remains more of a prompt-engineering heuristic than a deeply justified causal solution.

## Strengths
- The paper identifies a real failure mode of CLIP in multi-label settings and gives a plausible mechanism: single-label prompts ignore contextual label relations, while correlative prompts can recover some missing labels. The qualitative evidence in Figures 2, 5, 7, and 8 supports this narrative reasonably well.
- The method is lightweight and training-free at inference time, with a practical implementation that combines two prompt types rather than introducing new trainable modules. The empirical results show consistent improvements over vanilla CLIP and TagCLIP, and also over training-based baselines in the reported settings.

## Weaknesses
- The core method is underspecified in important ways. The paper does not clearly spell out how co-occurring labels are selected, ordered, or weighted in the prompt, nor does it fully define the multi-label decision rule after combining DiP and CoP scores. This hurts reproducibility and makes the reported gains harder to assess.
- The causal/TDE story is more heuristic than rigorous. Eq. (1) to Eq. (2) is presented as if it follows from causal mediation, but the derivation is not well justified and the final additive form looks like an empirically convenient rewrite rather than a principled equivalence. For an ICLR paper, that is a significant weakness because the theory is being used to motivate the entire method.
- The evaluation does not fully isolate what actually drives the gains. There is no clean ablation separating the benefit of co-occurrence prompting from the calibration step, no sensitivity analysis over the number or quality of co-occurring labels, and no robust study of thresholding or seed variance. The paper therefore does not convincingly show that DualPrompt is robust rather than simply well-tuned to these benchmarks.
- The “training-free” claim is softened by the dependence on external co-occurrence estimation, either from ChatGPT or a small labeled subset. That may be acceptable in practice, but it means the method is not purely data-free, and its performance depends on the quality of these priors.

## Nice-to-Haves
- A full algorithm box with exact prompt templates, co-occurrence selection rules, and thresholding details.
- A sensitivity study on the number of co-occurring labels, prompt wording, and noise in the co-occurrence set.
- A clearer resource-matched comparison against simple baselines such as prompt ensembling or score averaging with label priors.

## Novel Insights
The most interesting aspect of the paper is that it treats co-occurrence as a double-edged signal for CLIP: adding co-occurring labels to the prompt can recover weak or inconspicuous objects, but it also creates a strong bias toward contextual hallucination when the target label is absent. DualPrompt’s main contribution is not a new representation learner, but a way to balance these two effects by pairing a contextual prompt with a more discriminative one; this is a sensible insight, though the paper overstates how theoretically clean the resulting causal interpretation really is.

## Potentially Missed Related Work
- Recent prompt-based open-vocabulary tagging and zero-shot multi-label recognition methods beyond TagCLIP and the cited prompt-tuning baselines — relevant because the paper’s central claim is benchmark superiority, so stronger and more directly comparable zero-shot tagging baselines would matter.
- Prior work on prompt calibration / score correction for CLIP-style models — relevant because the paper’s “dual prompt plus calibration” design is close in spirit to that line of work and would help contextualize novelty.

## Suggestions
- Add a precise inference algorithm and ablate each component separately: DiP alone, CoP alone, and DualPrompt, all under the same co-occurrence source and thresholding rule.
- Report robustness to co-occurrence noise, number of co-occurring labels, and prompt phrasing.
- Provide stronger evidence that the calibration step specifically reduces hallucination, e.g. by decomposing false positives/false negatives and showing class-wise precision-recall shifts.
- Clarify exactly how co-occurrence is obtained in the “small data” setting and how much labeled data is needed before the gains saturate.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 6.0, 6.0]
Average score: 5.0
Binary outcome: Accept
