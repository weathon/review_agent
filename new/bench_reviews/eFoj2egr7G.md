Now I have a good calibration set. Let me analyze the key comparison points:

**Calibration anchors:**
- Visual Evidence Prompting (similar method, uses small models for LVLM guidance): Reject, scores 5,6,6,5 (avg ~5.5)
- PATCH (detection-based hallucination mitigation, fined-tuned): Reject, scores 5,8,5,6 (avg ~6.0)
- DeCo (decoding-based, training-free): Accept Poster, scores 6,6,6,6 (avg 6.0)
- Ensemble Decoding (logit manipulation, training-free): Accept Poster, scores 6,6,6,6 (avg 6.0)
- LURE (object hallucination mitigation): Accept Poster, scores 6,8,6,5 (avg ~6.25)
- RITUAL (random transforms, training-free, incremental): Withdrawn/Reject, scores 5,5,5,5 (avg 5.0)
- VDGD (description-grounded decoding, training-free): Accept Poster, scores 8,6,6,6 (avg 6.5)
- Modular Attribution (identify hallucination heads, training-free + fine-tuning): Accept Poster, scores 8,8,6,6 (avg 7.0)
- Latent Space Steering (VTI, training-free): Accept Spotlight, scores 6,8,8 (avg ~7.3)

Now let me synthesize the review, carefully checking the harsh critic's claims against the actual paper content.

**Checking Critical Issue 1 (CFG derivation):**
The harsh critic claims the probabilistic derivation is incorrect. Let me verify:

The paper writes:
$$p(y|x) \propto p_\theta(y)p(x|y)^\gamma$$

Then using Bayes rule: $p(x|y) \propto p_\theta(y|x)/p_\theta(y)$, they get:
$$\hat{p}_\theta(y|x) \propto p_\theta(y|x)^\gamma / p_\theta(y)^{\gamma-1}$$

Then in Section 4.2:
$$\hat{p}_\theta(y_t|b,c,x,y_{<t}) \propto p_\theta(y_t|b,c,x,y_{<t})^\gamma / p_\theta(y_t|b,x,y_{<t})^{\gamma-1}$$

And then:
$$\log \hat{p}_\theta(y_t|b,c,x,y_{<t}) = \gamma \log p_\theta(y_t|b,c,x,y_{<t}) + (1-\gamma)\log p_\theta(y_t|b,x,y_{<t})$$

Let me check: from the formula $\hat{p}_\theta(y_t|\cdot) \propto p_\theta(y_t|b,c,x,y_{<t})^\gamma / p_\theta(y_t|b,x,y_{<t})^{\gamma-1}$, taking log:
$$\log \hat{p}_\theta(y_t|\cdot) = \gamma \log p_\theta(y_t|b,c,x,y_{<t}) - (\gamma-1)\log p_\theta(y_t|b,x,y_{<t}) + \text{const}$$
$$= \gamma \log p_\theta(y_t|b,c,x,y_{<t}) + (1-\gamma)\log p_\theta(y_t|b,x,y_{<t}) + \text{const}$$

The paper writes this as an equality rather than proportionality, which is problematic. The "const" is the log-partition function (normalization over the vocabulary), which varies per time step. When you apply softmax to these mixed logits, you effectively get a different distribution than what the ratio form prescribes.

However, in practice, this is exactly how CFG is implemented in diffusion models and language models - you mix logits/scores and then apply softmax/sampling. The "correct" implementation would also require normalization, but in practice, the logit mixing approach is the standard operationalization of CFG. This is well-established.

So the critic's concern is technically valid at the mathematical level - the paper conflates proportionality expressions with equality and doesn't discuss the normalization issue. But the practical implementation (Algorithm 1) is the standard way CFG is applied. This is a presentation issue, not a fundamental methodological flaw. The method *is* a CFG-style approach; the math presentation is imprecise but doesn't change what the method actually does.

**Checking Critical Issue 2 ("root causes" claim):**
The paper states in the introduction: "these approaches do not address the intrinsic causes of object hallucination in LVLMs" and then claims MARINE "addresses object hallucination at inference time by targeting its two root causes."

The method adds external visual information via textual prompts and blends logits. The critic argues this doesn't address "root causes" since it doesn't modify the visual encoder or alignment layer.

This is partially valid. The method compensates for visual encoder deficiency by providing supplementary object information, and addresses misalignment by feeding that information through the text channel (bypassing alignment issues). But it doesn't *fix* the encoder or alignment. Calling this "addressing root causes" is an overclaim - it's more accurate to say "compensating for" those causes. This is a framing issue, not a fatal flaw, but the language is stronger than warranted.

**Checking Critical Issue 3 (evaluation scope):**
- LURE only compared on one fine-tuning baseline - valid concern
- VCD/OPERA hyperparameter fairness - partially valid but MARINE uses fixed hyperparameters too
- Small GPT-4V evaluation - valid
- Claim of "outperforms fine-tuning methods" based on one baseline - overclaimed

**Checking Critical Issue 4 (failure modes from guidance errors):**
This is a valid concern. No analysis of when detectors fail and how this propagates.

Let me also check the "intersection" concern - Table 7 shows intersection vs. union. The critic is right that only CHAIR is shown, not recall, for this ablation.

Now let me write the consolidated review.</think>Now I have sufficient calibration data. Let me synthesize the final review.

## Summary

MARINE proposes a training-free, API-free framework to mitigate object hallucination in Large Vision-Language Models (LVLMs) during inference. It extracts object-level information from external vision models (DETR, RAM++) as textual guidance prompts, then uses a classifier-free guidance (CFG)-style logit interpolation between guidance-conditioned and unconditioned generation to steer the LVLM away from hallucinated objects. Experiments across 5 LVLMs and multiple benchmarks (CHAIR, POPE, GPT-4V evaluation) demonstrate consistent reductions in object hallucination with low computational overhead (~2× greedy decoding latency).

## Strengths

- **Practical and deployable design**: MARINE requires no training, fine-tuning, or API access, making it immediately applicable to any existing LVLM. The latency analysis (Table 5) shows the lowest computational overhead among baselines (1.98× greedy), a meaningful practical contribution.

- **Comprehensive empirical evaluation across 5 LVLMs**: Testing on LLaVA, LLaVA-v1.5, MiniGPT-v2, mPLUG-Owl2, and InstructBLIP with CHAIR, POPE, and GPT-4V metrics provides broad evidence of method effectiveness, unlike many prior works that evaluate on 2–3 models.

- **Consistent improvements on object hallucination metrics**: MARINE achieves the best average CHAIR_S (8.4 vs. 10.1 for second-best VCD) and POPE accuracy (79.9% vs. 78.8% for Woodpecker), with improvements across all tested architectures.

- **Flexible vision toolbox with informative ablations**: Tables 6–7 demonstrate that combining DETR and RAM++ via intersection outperforms individual models, and the MARINE-Truth oracle provides useful context on the method's ceiling performance.

- **Maintains caption/VQA quality**: Figure 2 shows that standard caption quality metrics (BLEU, ROUGE, CIDEr, SPICE) are preserved or improved, supporting the claim that hallucination reduction does not come at excessive cost to generation quality.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed conceptual contribution regarding "root causes"**: The introduction states MARINE "addresses object hallucination at inference time by targeting its two root causes" (visual encoder deficiency and domain misalignment). However, the method does not modify, improve, or intervene on the visual encoder or alignment layer—it injects textual object descriptions through a prompting mechanism and blends logits. This *compensates* for missing visual information rather than *addressing root causes*. The actual mechanism is a test-time logit interpolation with external detector signals, placing MARINE in the same broad family of inference-time control methods as VCD and OPERA. The conceptual claim overstates the depth of the intervention. The empirical contribution—training-free guidance using detectors reduces object hallucinations—is valid and valuable on its own; it need not be dressed as addressing "intrinsic causes."

- **Imprecise CFG derivation that conflates probability ratios with logit interpolation**: Section 3 and Section 4.2 present a probabilistic derivation claiming that the guided distribution is $\hat{p}_\theta(y_t|\cdot) \propto p_\theta(y_t|b,c,x,y_{<t})^\gamma / p_\theta(y_t|b,x,y_{<t})^{\gamma-1}$, then state that the log is $\gamma \log p_{\text{cond}} + (1-\gamma) \log p_{\text{uncond}}$. This equality ignores the log-partition function required for normalization over the vocabulary at each step. Applying softmax to this linearly-mixed logits vector does not yield samples from the ratio distribution as written. In practice, logit mixing is the standard operationalization of CFG (used in both diffusion models and language models), so Algorithm 1 is correct as an implementation; the issue is that the probabilistic framing presents a heuristic as a principled derivation. The paper should either present MARINE honestly as a logit-blending heuristic inspired by CFG, or provide a rigorous derivation accounting for normalization.

- **Limited analysis of failure modes from detector errors**: MARINE's guidance is only as reliable as DETR and RAM++ detections. When these models miss objects (reducing recall of the guidance), the intersection strategy will systematically suppress those objects from generation. When detectors produce false alarms, those can be amplified through logit blending. The paper provides no error analysis: how often do detectors miss or hallucinate objects? How often does MARINE remove *correct* mentions that the LVLM would have otherwise produced? Table 1 already shows MARINE's recall is lower than LURE's and only modestly above greedy decoding for some models, suggesting the intersection strategy may be over-suppressing correct content. This gap is acknowledged but not analyzed.

- **Overstated claim about outperforming fine-tuning methods**: The claim "outperforms existing fine-tuning-based methods" (Abstract) rests entirely on comparison with LURE, a single method. LURE shows unusual behavior (much higher recall but often worse CHAIR than greedy decoding), and its applicability across LVLMs is unclear (the original LURE paper fine-tunes MiniGPT4; it is not stated whether LURE was re-trained for each architecture tested). The evidence supports "outperforms LURE," not a general statement about fine-tuning-based approaches.

### Minor

- **Small GPT-4V evaluation set**: Table 3 uses only 90 questions (LLaVA-QA90) and 50 image captions, evaluated on only 2 of 5 models. Some improvements have overlapping confidence intervals (e.g., 5.82→5.94 for LLaVA QA90 accuracy). This limits confidence in the qualitative evaluation claims.

- **Missing recall in intersection vs. union ablation**: Table 7 compares intersection and union strategies on CHAIR metrics only. Since intersection explicitly trades recall for precision, not showing recall makes this comparison incomplete.

- **Fixed hyperparameters without per-model justification**: γ=0.7, DETR noise intensity=0.95, and RAM++ detection threshold=0.68 are held constant across all models and tasks. Figure 3 shows different optimal γ ranges per model, suggesting model-specific tuning may yield better results.

- **Scope limited to object hallucination**: The method relies on object-level detections and cannot address attribute, relational, or other non-object hallucination types. This is explicitly acknowledged in Section 6 but still limits the generality of the framework.

### Trivial
None.

## Nice-to-Haves

- **Error analysis of guidance models**: Reporting detection metrics (mAP/recall) of DETR and RAM++ on the evaluation benchmarks would clarify how close MARINE is to its ceiling and how often guidance errors propagate.

- **Per-model γ tuning or adaptive guidance**: An adaptive γ selection strategy (e.g., based on detection confidence) could improve robustness across models and inputs.

- **Integration with other decoding methods**: Since the paper notes MARINE is complementary to VCD/OPERA, an experiment combining them would quantify the additive benefit.

- **Evaluation on broader benchmarks**: Testing on general VLM benchmarks (e.g., MMBench, MM-Vet) would strengthen claims about preserving overall model capability.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claim that external vision models undermine "training-free" framing**: The abstract's "training-free and API-free" refers to not training/fine-tuning the LVLM or calling GPT APIs. Using pre-trained open-source detectors is consistent with this claim. Removed because the paper's framing is accurate about what it means by these terms.

- **Concern about MSCOCO data overlap for detection models**: The reviewer speculates MSCOCO may be in DETR/RAM++ training data, but this is an unverified claim about training data composition, and the paper does test on additional datasets (A-OKVQA, GQA) outside MSCOCO.

- **Demand for recent/larger LVLMs**: The paper tests on 5 diverse architectures, which is more thorough than most comparable work (e.g., DeCo and Ensemble Decoding test on 3–4 models). Requesting newer models is a nice-to-have, not a weakness of the current scope.

- **Demand for statistical significance / error bars**: Single-run evaluation is standard practice in this research area. Requesting bootstrap CI for large-scale benchmarks like POPE and CHAIR is above community norms.

- **Formatting/style nitpicks**: Removed as per instructions.

- **Demand for concurrent CFG-based method integration experiments**: While combining MARINE with VCD/OPERA would be informative, the paper already compares against them separately. Integration is a natural future direction, not a required experiment.

## Novel Insights

The paper demonstrates that a simple intersection of detections from complementary vision models (DETR for accuracy, RAM++ for broad recognition) can provide surprisingly effective guidance for reducing hallucinations—suggesting that the precision-recall trade-off in guidance quality strongly favors precision. This aligns with the broader lesson from CFG: strong conditional signals, even if incomplete, can effectively steer generative models when applied via logit manipulation.

## Suggestions

- **Soften the "root causes" framing**: Describe MARINE as *compensating for* or *mitigating the effects of* visual encoder limitations and domain misalignment, rather than addressing their root causes. This preserves the motivation while being accurate.

- **Fix the CFG derivation**: Either acknowledge that Algorithm 1 implements a heuristic logit-blending scheme inspired by CFG (which is perfectly respectable), or provide a proper derivation that accounts for the softmax normalization step.

- **Add a brief error analysis**: Even a small table showing DETR/RAM++ detection recall/precision on MSCOCO, and how many ground-truth objects survive the intersection, would clarify the method's reliability ceiling.

- **Report recall alongside CHAIR in the intersection-vs-union ablation**: This would give a complete picture of the precision-recall trade-off.

- **Scope down the fine-tuning comparison claim**: Change "outperforms existing fine-tuning-based methods" to "outperforms LURE, a fine-tuning-based method."

## Score and Decision

**Calibration**: Compared against similar training-free hallucination mitigation papers:
- DeCo (Accept Poster, avg 6.0): Decoding-time intervention, similar evaluation scope, incremental mechanism
- Ensemble Decoding (Accept Poster, avg 6.0): Logit manipulation, training-free, limited theoretical justification
- Visual Evidence Prompting (Reject, avg 5.5): Very similar method—small models provide evidence to LVLMs—weaker evaluation
- RITUAL (Reject, avg 5.0): Simple training-free method, limited novelty
- VDGD (Accept Poster, avg 6.5): Training-free, stronger analysis and benchmark contribution

MARINE is empirically stronger than Visual Evidence Prompting and RITUAL (broader evaluation, better results), comparable to DeCo and Ensemble Decoding in practical contribution, but has meaningful overclaims both in the CFG derivation and "root causes" framing. The method is simple but effective; the empirical contribution is solid but the conceptual claims exceed what is supported.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>