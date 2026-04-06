
========================================================================
INDIVIDUAL REVIEWS
========================================================================

────────────────────────────────────────
HARSH CRITIC (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## Section-by-Section Critical Review

**Title & Abstract:** The title clearly indicates the method (VSF) and its key properties (simple, efficient, effective). The abstract concisely states the problem, the proposed solution, and the main results, including quantitative claims (e.g., negative scores of 0.420 and 0.545). These claims are specific and set a clear expectation for the experiments to validate.

**Introduction & Motivation:** The introduction effectively motivates the problem: the need for effective negative guidance in few-step (distilled) generative models where CFG fails. It clearly outlines the limitations of existing approaches (NASA, NAG) and states the contributions (method, dataset, evaluation). The connection to related challenges (negation understanding in VLMs) is appropriate.

**Method / Approach:** The core idea of flipping the sign of negative prompt *values* (V) while keeping keys (K) intact is intuitive and presented as a form of adaptive, token-level guidance. The mathematical derivation connecting the adaptive weight formulation (Eq. 7-8) to the final concatenated form (Eq. 9) is crucial for grounding the method; the claim of equivalence is deferred to an appendix, which must be checked. The extensions for MMDiT architectures (duplication, masking, bias) are necessary but complicate the initial "simple" claim. The description is sufficiently detailed for implementation, though the pseudo-code is in the appendix. A significant concern is **novelty**: the acknowledgement mentions the independent but prior community project "NegPiP" which uses a similar approach. The authors must explicitly discuss and differentiate their work from NegPiP in the main text, not just in the acknowledgements, to clarify the novel contribution.

**Experiments & Results:**
*   **Dataset (NegGenBench):** The use of GPT-4 to generate challenging prompt pairs is reasonable. However, the paper does not provide analysis of the dataset's characteristics (e.g., diversity, difficulty distribution, potential biases). A sample of the prompts should be included in the main paper or appendix to allow assessment.
*   **Metrics & Evaluation:** Relying on an MLLM (LLaMA) as a judge for adherence and quality is a pragmatic choice but introduces potential bias. The authors partially address this with limited human validation (Table 3, 10 prompts) and a fine-tuned Qwen model (Appendix), which is good practice. However, the human validation is very small-scale (n=10 prompts, 2 seeds). The results show a consistent ranking, which supports the automatic evaluation, but more robust human evaluation (e.g., on a larger subset) would strengthen the claims. The quality scores from LLaMA are noted to be "generous"; the interpretation of these scores (e.g., what <90 means) is helpful but subjective.
*   **Baselines and Comparisons:** The comparisons against NASA and NAG (and their strong variants) are appropriate and fair, as these are the direct predecessors for few-step models. The inclusion of a "generate-then-edit" pipeline (Flux Kontext) is insightful. The external baseline table (Table 1) is useful for context but mixes open/closed models and different inference paradigms; the comparison to CFG on a non-few-step model is a valuable point of reference. The trade-off curves (Figure 5) are a strong analysis, effectively showing VSF's superior Pareto frontier. However, the hyperparameter sweep for NAG (287 runs) vs. VSF (66 runs) and NASA (10 runs) seems imbalanced; the justification for the different sweep strategies should be clarified.
*   **Ablation Study:** The ablation (Figure 5 Right) is excellent, systematically testing the contribution of masking, duplication, and bias. The finding that the "Whole Embedding Flip" (WEF) baseline fails is important and helps justify the core design. The explanation for why WEF fails (flipping keys changes semantic matching) is plausible but could be elaborated.

**Writing & Clarity:** The paper is generally well-written and logically structured. Some sections are dense (e.g., Section 3.2 on MMDiT modifications), but the figures (Fig. 3) help. The frequent referencing to the appendix for proofs, pseudo-code, and additional results is necessary but requires the reader to trust those details. The flow from problem to method to experiments is clear.

**Limitations & Broader Impact:** The paper briefly mentions future work in the appendix but lacks a dedicated "Limitations" section in the main text. Key limitations that should be explicitly discussed include: 1) The reliance on MLLM-based evaluation and its potential biases, 2) The complexity added by the masking/duplication scheme for MMDiT, potentially undermining the "simple" claim, 3) Any observed failure modes or types of negative prompts where VSF does not work well. The broader impact discussion is minimal. The method's application for content moderation and bias reduction is noted, but potential negative societal impacts (e.g., making it easier to generate content that avoids safety filters) should be considered.

### Overall Assessment
The paper presents a well-motivated, technically sound method (VSF) that addresses a clear gap in negative guidance for few-step generative models. The core idea is elegant, and the experimental evaluation is thorough in its comparison to relevant baselines and includes insightful ablation studies and trade-off analysis. The most significant concerns are the need for a more explicit discussion of novelty relative to the concurrently discovered NegPiP, a more robust analysis of the proposed dataset, and a more thorough discussion of limitations and evaluation caveats. The contribution is solid and likely meets ICLR's bar for acceptance, provided these concerns are adequately addressed in a revision.

────────────────────────────────────────
NEUTRAL REVIEWER (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## Balanced Review

### Summary
This paper introduces Value Sign Flip (VSF), a method for incorporating negative prompt guidance in few-step (1-8 steps) diffusion and flow-matching models. VSF works by dynamically flipping the sign of attention values associated with negative prompt tokens, coupled with token duplication and masking strategies for MMDiT architectures. The authors also contribute a challenging dataset, NegGenBench, for evaluating negative prompt adherence. Experiments demonstrate that VSF outperforms prior methods (NASA, NAG, and CFG) in avoiding negative concepts while maintaining image quality and positive prompt alignment.

### Strengths
1.  **Clear Practical Contribution**: The method addresses a well-known and practical limitation—ineffective negative guidance in efficient, few-step generative models. The solution is conceptually simple (sign flip in attention) and is shown to integrate with major architectures (MMDiT and cross-attention).
2.  **Comprehensive Experimental Validation**: The paper provides extensive quantitative comparisons against strong baselines (NAG, NASA, CFG) and external models (GPT-4o, Qwen-Image). The proposed NegGenBench dataset appears to be a non-trivial testbed for negation. The inclusion of trade-off curves (Figure 5) effectively demonstrates VSF's superior Pareto frontier.
3.  **Efficiency and Usability**: VSF adds minimal computational overhead compared to dual-forward-pass methods like CFG, maintaining the speed advantage of few-step models. The release of code, a ComfyUI node, and the dataset supports reproducibility and practical adoption.

### Weaknesses
1.  **Lack of Theoretical Grounding or Deeper Analysis**: While the method is motivated intuitively ("noise-canceling headphones"), the paper lacks a rigorous theoretical analysis. For instance, the claim that Equation 9 is mathematically equivalent to the adaptive weighting of Equation 7 is relegated to the appendix without elaboration in the main text. A deeper discussion on why flipping values within attention is more effective than operating on attention outputs (like NASA/NAG) is needed.
2.  **Unclear Generalization and Limitations**: The evaluation is heavily focused on a specific type of "challenging" negation (removing essential object parts). It's unclear how VSF performs on more conventional negative prompts (e.g., "blurry", "ugly", "extra fingers") or for style removal/debiasing, which are mentioned but not quantitatively evaluated. The failure mode of the "Whole Embedding Flip" ablation is noted but not sufficiently analyzed.
3.  **Potential Biases in Evaluation**: The reliance on an MLLM (LLaMA) as the primary judge for adherence and quality scores is a potential weakness. While human validation on a subset aligns with trends, the paper does not thoroughly address the known limitations of MLLMs in evaluating image quality and compositional understanding. The high quality scores (often >0.95) reported in Table 2 seem suspiciously high and may indicate a lack of sensitivity in the metric.

### Novelty & Significance
**Novelty:** The core idea of in-attention value sign flipping for dynamic, token-level negative guidance in few-step models is novel. While related to prior work (NASA, NAG, Schramowski et al.), the specific mechanism and its adaptation for MMDiT architectures via duplication and masking constitute a clear incremental advance.
**Significance:** The work is significant for the growing community using efficient, distilled generative models. Providing an effective negative guidance mechanism for models like SD3.5 Turbo and Flux Schnell solves a real user pain point, enhancing controllability. The release of a challenging negation benchmark is also a valuable community resource. The performance lift over baselines is substantial, making a strong case for the method's utility.

### Suggestions for Improvement
1.  **Strengthen the Analysis Section**: Add a subsection or expanded discussion providing more intuition or a proof sketch for why in-attention value manipulation is more effective than output-space guidance. Analyze the attention maps (Figure 6) more quantitatively to show how the "negative attention" correlates with suppressed regions.
2.  **Broaden the Evaluation Scope**: Include a standard set of negative prompts (e.g., from DiffusionDB) to demonstrate performance on common use cases beyond the curated NegGenBench. Test VSF explicitly on style removal and demographic debiasing tasks to support the claims made in the discussion.
3.  **Improve Evaluation Rigor**: Complement the MLLM scores with established automated metrics (e.g., CLIP score for positive adherence, perhaps using a negation-augmented version) and a more extensive human evaluation. Discuss the limitations of the chosen evaluation MLLM and how the fine-tuned Qwen-2.5-VL model addresses them.
4.  **Clarify the Contribution vs. Concurrent Work**: The note about the independent discovery of NegPiP should be moved from the acknowledgements to the related work or a discussion section. A brief comparison or discussion of the similarities/differences would be intellectually honest and help position VSF.

────────────────────────────────────────
SPARK FINDER (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare against recent dynamic negative guidance methods (e.g., Koulischer et al. 2025) and pixel-adaptive methods (Schramowski et al. 2023).** The paper builds on these works but only compares to NASA and NAG. Without this comparison, the claim of superior adaptation is unsubstantiated.
2. **Validate automatic metrics with a larger-scale human evaluation.** The human study uses only 10 prompts (2 seeds each), which is insufficient to trust the LLaMA-based scores that drive all quantitative claims.
3. **Test on established negation benchmarks (e.g., from Park et al. 2025).** Relying solely on a new, unreleased dataset (NegGenBench) risks overfitting to its specific construction and lacks external validation of generalization.
4. **Include video generation experiments.** The abstract and introduction claim applicability to video, but no video results are shown, undermining the claim of effectiveness for video models.

### Deeper Analysis Needed (top 3-5 only)
1. **Provide a thorough computational efficiency analysis.** The paper claims "small computational overhead" but offers no FLOPs, memory, or precise runtime comparison against baselines (especially CFG's two forward passes).
2. **Explain why flipping value signs works better than existing approaches.** The analogy to noise cancellation is insufficient; a mechanistic analysis (e.g., how gradient signals propagate) is needed to justify the core design.
3. **Ablate the role of the attention bias (beta) and duplication strategy.** The paper ablates masking but not the bias term, which is introduced for quality. The duplication is attributed to an implementation detail, but its functional impact is unclear.
4. **Analyze failure modes and limitations.** The trade-off curves show VSF eventually degrades; understanding what types of negative prompts or concepts remain challenging is critical for assessing practical utility.

### Visualizations & Case Studies
1. **Show more attention map visualizations across different steps, layers, and prompt types.** Figure 6 (appendix) shows one example; systematic visualizations are needed to confirm the claimed adaptive behavior.
2. **Include side-by-side qualitative comparisons for all major baselines on the same prompts.** The paper shows isolated successes; without direct visual comparisons, it's hard to judge relative performance.
3. **Display failure cases where VSF does not suppress the negative concept or severely degrades quality.** This would help delineate the method's boundaries and inform users.

### Obvious Next Steps
1. **Integrate the fine-tuned negation-aware MLLM (Qwen-2.5-VL) into the main evaluation.** Using it only as an external baseline (Appendix) misses the opportunity to validate the primary metrics with a more reliable judge.
2. **Conduct a controlled hyperparameter study for alpha and beta.** The random sweep shows trade-offs but does not isolate each parameter's effect, making tuning recommendations vague.
3. **Test VSF on more few-step models (e.g., SDXL Lightning, SNOOPI) to prove architectural generality.** The experiments are limited mainly to SD3.5-Turbo and Flux Schnell.
4. **Compare against a generate-then-edit pipeline with more advanced editors (beyond Flux Kontext).** The chosen editing baseline may not represent the state-of-the-art, weakening the claim of superiority over pipelines.

────────────────────────────────────────
POTENTIALLY MISSED RELATED WORK (deepseek/deepseek-v3.2:online via OpenRouter)
────────────────────────────────────────
Related work search was skipped.

========================================================================
FINAL CONSOLIDATED REVIEW (deepseek/deepseek-v3.2 via OpenRouter)
========================================================================

## Summary
This paper introduces Value Sign Flip (VSF), a method for incorporating negative prompt guidance in few-step (1-8 step) diffusion and flow-matching models. VSF works by flipping the sign of attention values associated with negative prompt tokens, enabling adaptive, token-level suppression of unwanted content. The authors also contribute a challenging dataset, NegGenBench, and demonstrate that VSF outperforms prior specialized methods (NASA, NAG) and even standard CFG on non-few-step models in negative prompt adherence, with a favorable trade-off in image quality and positive prompt following.

## Strengths
- **Effective and Efficient Solution to a Clear Problem**: VSF addresses the well-known failure of classifier-free guidance in few-step distilled models. The method is conceptually simple (in-attention value sign flip), adds minimal computational overhead, and is shown to integrate with both cross-attention and modern MMDiT architectures (e.g., Stable Diffusion 3.5 Turbo, Flux Schnell).
- **Comprehensive and Conclusive Empirical Evaluation**: The paper provides extensive quantitative comparisons against the relevant few-step baselines (NASA, NAG) and includes insightful external comparisons. The analysis is strengthened by Pareto frontier plots (trade-off curves) that convincingly show VSF's superior performance profile and a thorough ablation study validating the design choices (masking, duplication).

## Weaknesses
- **Evaluation Relies Heavily on a Single MLLM Judge with Limited Human Validation**: The primary metrics (negative/positive/quality scores) are derived from an off-the-shelf MLLM (LLaMA). While the authors include a small human study (10 prompts) and a fine-tuned model in the appendix, the core claims would be more robust with a larger-scale human evaluation or the use of their improved Qwen-2.5-VL judge in the main experiments.
- **Incomplete Comparison to Cited Prior Work on Dynamic Guidance**: The paper builds upon the ideas of Koulischer et al. (2025) and Schramowski et al. (2023), stating that VSF expands Koulischer et al. to token-level modulation. However, it does not quantitatively compare against these dynamic guidance methods, leaving the claimed advancement in adaptation partially unsubstantiated.

## Nice-to-Haves
- A more detailed analysis of failure cases or categories of negative prompts where VSF struggles, which would help users understand its practical boundaries.
- Testing on a broader set of few-step models (e.g., SDXL Lightning, SNOOPI) to further demonstrate architectural generality beyond SD3.5-Turbo and Flux Schnell.
- Inclusion of the fine-tuned negation-aware Qwen-2.5-VL model in the main evaluation table to provide a more reliable automated metric.

## Novel Insights
The core novel insight is the mechanism of applying negative guidance *inside* the attention computation by flipping the sign of the value (V) vectors for negative prompt tokens, while keeping their key (K) vectors intact. This creates a dynamic, token-level steering effect where the strength of suppression is adaptively proportional to how much the image attends to the negative concept, a form of "noise cancellation" within the model's feature space. This is a distinct approach from prior methods that operate on attention outputs (NASA, NAG) or the final noise prediction (CFG, Schramowski et al.).

## Suggestions
- Conduct a larger-scale human evaluation (e.g., on 50+ prompt pairs) to solidify the trust in the automated scores and better calibrate the interpretation of the quality metric.
- Add a quantitative comparison against Koulischer et al. (2025) to directly demonstrate the advantage of token-level adaptation, even if this requires a re-implementation for the few-step setting.

========================================================================
PREDICTED SCORE
========================================================================

Score: 4.6
Decision: N/A
Total Cost: $0.0194
