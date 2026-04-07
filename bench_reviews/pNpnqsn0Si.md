## Summary
This paper introduces Thoughtbubbles, a transformer variant that learns to dynamically allocate parallel computation in latent space by forking or deleting residual streams based on learned cumulative scores. The method is trained solely with a language modeling objective during pretraining, without explicit supervision. It demonstrates consistent improvements in perplexity and zero-shot performance (LAMBADA, HellaSwag) over parameter-matched and computation-matched baselines across model scales from 150M to 772M parameters.

## Strengths
- **Novel and well-motivated architecture:** The forking mechanism for dynamic, input-adaptive allocation of parallel residual streams is a genuinely new approach to enabling inference-time scaling within standard pretraining, moving beyond fixed pause tokens or chain-of-thought.
- **Strong and consistent empirical gains:** The method consistently outperforms both parameter-matched transformers and a simple computation-matched baseline (duplicated filler tokens) on validation perplexity across two datasets and multiple model sizes. Notably, a 319M Thoughtbubbles model achieves lower perplexity than a 772M baseline on OpenWebText. Zero-shot improvements on LAMBADA and HellaSwag further validate the approach.
- **Interpretable, unsupervised adaptation:** Analysis shows the model allocates more computation to tokens with higher predictive entropy (Fig. 5), an emergent property that aligns with intuitive notions of computational difficulty. Additional analysis (Fig. 4, Appendix C) provides evidence that forked tokens meaningfully influence their parent and that forking occurs at interpretable locations.

## Weaknesses
- **Lacks comparison to strong adaptive computation baselines:** The primary computation-matched baseline (duplicating input tokens) is relatively naive. To properly situate the contribution, comparisons to more recent adaptive methods (e.g., pause tokens, Mixture-of-Depths, Universal Transformers) are needed. Without this, the claimed advantage over prior art is not fully substantiated.
- **Limited evaluation on tasks requiring complex reasoning:** While motivated by enabling more difficult, multi-step problems, evaluation is restricted to perplexity and relatively simple zero-shot tasks (LAMBADA, HellaSwag, BLiMP, PIQA). There is no assessment on benchmarks like GSM8k or MATH, which are more direct tests of improved computational capability, though the authors acknowledge this limitation.
- **Incomplete analysis of computational efficiency:** The paper notes wall-clock inefficiency in the limitations but provides no quantitative measurements of inference time, FLOPs, or memory compared to baselines. For a method that introduces adaptive parallel computation, a clearer understanding of its practical trade-offs is important.
- **Training dynamics and gradient flow through hard top-k are underexplored:** The method relies on hard top-k decisions for forking, which creates a non-differentiable bottleneck. The authors mention this can cause gradient issues and limit deeper forking, but the paper does not detail how gradients are propagated (e.g., via straight-through estimation) or fully analyze the consequences. This affects reproducibility and understanding of optimization stability.

## Nice-to-Haves
- **More comprehensive ablation studies:** While an ablation on forking location is provided (Appendix B), studies on the necessity of the score attenuation mechanism, the impact of the learned fork embedding, and the effect of different forking budgets (κ) would help validate design choices.
- **Analysis of what triggers forking beyond entropy:** A deeper investigation into the linguistic or structural features (e.g., syntactic complexity, coreference) that correlate with forking decisions would enrich the interpretability of the adaptive behavior.
- **Statistical significance via multiple runs:** Reporting results across multiple random seeds would strengthen the robustness of the claimed improvements, especially at smaller scales.
- **Case studies illustrating forking behavior:** Concrete examples of text passages where forking succeeds or fails would provide intuitive insight into the method's real-world operation.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Claim about 319M model outperforming 772M baseline is overstated:** The paper's claim is specifically about perplexity (Fig. 3), and it does not assert consistent downstream task superiority. The results are presented accurately.
- **Criticism that the link between importance and forking benefit is not formally justified:** The paper provides empirical support via attention analysis (Fig. 4) and entropy correlation (Fig. 5), which is reasonable for an empirical architecture paper.
- **Request for derivation of the position encoding heuristic:** The partial rotation for RoPE is an engineering solution motivated by the need to pack forks; requiring a first-principles derivation is beyond the scope.
- **Demand for scaling to much larger models/datasets as a core weakness:** The experiments (up to 772M params, 2.5B tokens) are sufficient to demonstrate the method's viability; larger-scale training is a natural future direction.

## Novel Insights
The paper shows that a transformer can learn to dynamically allocate parallel computation in a fully unsupervised manner, with forking decisions emerging to focus on tokens of moderate predictive entropy. This suggests the model develops an implicit notion of which tokens benefit from extra "thinking," aligning with recent literature on the informativeness of high-entropy tokens. The analysis further reveals that forked tokens exert substantial influence on their parent via attention, indicating the created latent streams perform meaningful auxiliary computation rather than being mere noise.

## Suggestions
- Implement comparisons to state-of-the-art adaptive computation baselines (e.g., pause token methods, Mixture-of-Depths) to clearly demonstrate the relative advantages of Thoughtbubbles.
- Provide quantitative measurements of inference-time efficiency (wall-clock, FLOPs) to better characterize the practical trade-offs of the method.
- Include a brief description of how gradients are propagated through the hard top-k operation (e.g., straight-through estimator) in the main text or appendix to improve reproducibility.