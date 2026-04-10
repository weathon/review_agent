=== CALIBRATION EXAMPLE 7 ===

# Final Consolidated Review
## Summary
This paper proposes Principal Spectral Regularization (PSR), a method that selectively regularizes only the dominant spectral directions in SGD momentum. The authors argue that full orthogonalization (as in Muon) is computationally expensive and sometimes suboptimal, and they show that penalizing a small subset of “spiked-head” components enables SGD with momentum to outperform AdamW in LLM pretraining while being more efficient than Muon for large-scale models.

## Strengths
- **Novel spectral perspective and visualization:** The paper provides insightful visualizations of momentum spectra across LLM layers (Fig. 1), identifying a “spiked-head-heavy-tail” structure. This spectral lens is underexplored in optimizer analysis and motivates the core idea of regularizing only principal directions.
- **Extensive multi-scale empirical validation:** Experiments on LLaMA models from 350M to 7B parameters demonstrate consistent improvements over AdamW in validation perplexity (Fig. 3) and downstream task performance (Table 4). The inclusion of long-run training (36B tokens) adds credibility.
- **Rigorous computational complexity analysis:** Theorem 4.1 and the accompanying proof (Appendix E) give a clear upper bound on the FLOP overhead of PSR, showing it is about 2% of Muon’s Newton-Schulz cost. Memory and runtime comparisons (Table 3) further support efficiency claims for large-scale models (7B, 70B).

## Weaknesses
### Major:
- **Limited theoretical justification:** While computational complexity is analyzed, the paper does not provide a theoretical explanation for why regularizing only the top spectral directions should outperform full orthogonalization or Adam. The connection between the Styblinski‑Tang toy example (Section 3.2) and LLM optimization remains intuitive rather than principled.
- **Practical computational overhead undermines efficiency claims for small-to-medium models:** Although PSR has lower theoretical FLOP overhead, wall‑clock measurements (Table 3) show that for models up to 3B parameters, PSR can be slower than Newton-Schulz due to sequential QR/SVD steps in the current PyTorch implementation. This practical caveat is acknowledged but not resolved, weakening the efficiency argument for a common training regime.
- **PSR consistently trails Muon in extended training and downstream performance:** Results (Fig. 4, Table 4) indicate that PSR is worse than Muon in downstream performance and in later training stages (e.g., after 36B tokens). This limits the claimed “surpassing” of Adam to a specific regime and suggests full orthogonalization remains beneficial for final convergence.

### Minor:
- **Styblinski‑Tang motivation is tenuous:** The use of a modified Styblinski‑Tang function with power-law weights and noise as motivational insight is creative, but its relevance to billion-parameter LLM optimization is not substantiated. The leap from this synthetic benchmark to real training dynamics is large and weakly justified.
- **Hyperparameter sensitivity and ablation are limited:** The choice of \(K=2\), \(r=m/32\), and \(\eta=0.95\) is presented as optimal but not rigorously explored across different model scales or architectures. While Table 9 provides some ablations, a more systematic sensitivity analysis would strengthen the method’s practical guidance.

### Trivial:
- **Update RMS rescaling factor is empirically chosen:** The factor 0.18 is derived from observation (Appendix C) but lacks theoretical grounding; however, this is common practice in optimizer design and does not harm the core contribution.

## Nice-to-Haves
- Compare with other modern memory-efficient optimizers like Sophia or Lion to better situate PSR in the landscape.
- Extend evaluation to fine-tuning tasks to demonstrate generality beyond pretraining.
- Implement a fused CUDA kernel for the Lanczos-bidiagonalization step to reduce the practical overhead from sequential operations.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness:** “The central claim that PSR enables momentum to ‘surpass Adam’ is not supported by the evidence.” → **Removed because the paper shows consistent, albeit marginal, improvements over Adam in validation perplexity and downstream averages (Fig. 3, Table 4). The claim is technically supported.**
- **Weakness:** “The computational efficiency claims are misleading and not borne out in practice.” → **Removed because the paper provides both theoretical complexity analysis and empirical wall-clock measurements (Table 3), openly acknowledging that PSR is slower for smaller models due to implementation limitations. This is a nuanced weakness, not a misleading claim.**
- **Weakness:** “The connection between Styblinski‑Tang and LLM optimization is poorly justified.” → **Weakened and moved to Minor weaknesses, as the paper uses it only as motivational insight, not as proof of effectiveness.**

## Suggestions
- **Strengthen the theoretical foundation:** Provide a convergence analysis or geometric interpretation explaining why penalizing only dominant directions works better than full orthogonalization in certain training phases.
- **Clarify the practical regime for PSR:** Explicitly characterize conditions (e.g., early training, limited compute budget, large model sizes) under which PSR is preferred over Muon or Adam, based on the observed trade-offs.
- **Conduct a hyperparameter sensitivity study:** Systematically evaluate the impact of \(K\), \(\eta\), and \(r\) across different model scales and datasets to provide clearer tuning guidance.

**Overall Assessment:** The paper presents a novel and empirically well-supported interpolation between full orthogonalization and no regularization, with a compelling spectral perspective. While it has limitations in theoretical grounding and practical overhead for smaller models, the core contribution—showing that regularizing only principal spectral directions can beat Adam—is valid and could influence optimizer design. The work is technically sound, clearly presented, and of interest to the LLM training community.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0, 4.0, 2.0]
Average score: 3.2
Binary outcome: Reject
