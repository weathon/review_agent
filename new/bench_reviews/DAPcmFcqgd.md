Now I have enough calibration data. Let me synthesize the final review.

## Summary

The paper introduces MoEP (Modular Expert Paths), a sparse decoder-only architecture that combines layer-level top-k routing across parallel Transformer blocks with MoE-style shrink/grow projection layers, aiming to add sparsity while keeping total parameter count fixed relative to a dense baseline. Evaluated on the BabyLM strict-small track (~10M words, 28M parameters), MoEP shows competitive or slightly better performance than the GPT-2 baseline and claims faster early learning, though gains are modest and task-dependent.

## Strengths

- **Well-motivated design goal**: The aim of adding sparsity without increasing total parameters (by having parallel blocks operate at reduced dimensionality, dP < dL) addresses a real concern in MoE design and is a creative architectural idea. The shrink→parallel stack→grow pipeline is conceptually clean and illustrated well in Figure 2.

- **Parameter-matched comparison for the main MoEP variant**: For MoEP (28M params) vs. GPT-2 (28M params), the authors match total parameters, tokenizer, training data, and evaluation protocol, making the comparison fair within the BabyLM framework.

- **Standardized evaluation and reproducibility**: Following the official BabyLM evaluation pipeline, reporting official baselines (GPT-2, GPT-BERT variants), and releasing code/models on HuggingFace and GitHub constitute a genuine reproducibility contribution.

- **Training dynamics analysis**: The checkpoint-by-checkpoint analysis (Appendix A.3) showing MoEP peaks at 30M words vs. later for other models is informative and relatively uncommon in small-scale architecture papers.

## Weaknesses

### Major:

- **Overclaimed performance gains with marginal, non-robust evidence**: The paper states MoEP "outperforms all BabyLM strict-small baseline models," but the margins are thin and inconsistent. Across individual tasks, MoEP trades blows with baselines (losing on BLiMP, EWOK, some fine-tuning tasks). The macro-average improvement depends on whether AoA is included, and MoEP's own AoA scores are missing (marked as "—" in Table 1). No variance, confidence intervals, or multiple seeds are reported—just single runs. At this scale, differences of 1–2 points on noisy benchmarks could easily be within random variation. The claim of "outperforming all baselines" is not justified by this evidence.

- **"Fast and stable training" claim contradicted by own analysis**: Contribution #3 states the paper shows "layer level parallelism enable fast and stable training." But the training dynamics (Appendix A.3) explicitly describe MoEP as showing instability: "sparse modular routing can accelerate early learning but also introduces instability," and post-peak checkpoints show "diminished generalization" and "sharp collapse." GPT-2 is described as showing "steadier learning" with "fewer dramatic changes." The paper's own analysis contradicts the "stable" claim; MoEP is faster but less stable.

- **MoEP-SwiGLU violates the core design principle without acknowledgment**: The paper's central pitch is "sparsity while keeping the total parameter count fixed," yet MoEP-SwiGLU has 38M parameters—36% more than the 28M baseline (Table 2). This variant is presented as a main contribution but is not flagged as violating the fixed-parameter constraint, nor compared against a 38M-parameter dense baseline. This makes the MoEP-SwiGLU comparison uninformative about the compactness claim.

- **No ablations isolating architectural contributions**: The paper proposes combining three ideas (shrink/grow MoE projections, parallel routed layers, dimensionality reduction), but provides no ablation—e.g., parallel blocks without routing, routing without MoE projections, or varying P, E, k, or dP/dL ratios. Without ablations, it is impossible to determine which components matter or whether a simpler variant would suffice. The claim that "improving routing mechanism increased performance within parallel architecture" is made without any controlled experiment (e.g., no comparison to PaPaformer-style parallel paths without top-k routing).

- **Missing essential implementation details**: Several methodologically critical details are omitted: (1) the λ^block and λ^expert balancing loss coefficients, which directly control collapse vs. diversity; (2) whether routed outputs are weighted by routing probabilities or uniformly averaged; (3) the router temperature/noise settings; (4) how dP/dL was derived to match the parameter budget. For a new architecture whose selling point is compact sparsity, the parameter accounting needs to be fully transparent.

### Minor:

- **Incomplete evaluation comparability**: Some models in Table 1 lack AoA scores, making macro averages inconsistent. When AoA is excluded, the margin between MoEP and the authors' own GPT-2 is described as "near comparable" in the text, undermining the claim of clear superiority.

- **Unclear novelty relative to existing work**: The paper acknowledges PaPaformer (parallel paths) and MoLE (layer-level MoE with LoRA). MoEP combines these ideas with standard top-k routing and entropy balancing—each component is known. The paper does not experimentally demonstrate that this particular combination offers advantages over simpler alternatives (e.g., parallel paths alone, or standard FFN-level MoE).

## Nice-to-Haves

- **FLOPs or active-parameter analysis per forward pass**: The paper claims "efficiency" but never reports FLOPs or active parameter counts per token, making it impossible to verify whether MoEP is actually more efficient than a dense baseline of equal total parameters.

- **Routing behavior analysis**: Contribution #3 promises "analysis of expert networks routing behavior," but only global evaluation scores over training are shown. Token-level routing visualizations, expert utilization distributions, and specialization metrics would substantiate this claim.

- **A standard MoE baseline**: Comparison with a conventional FFN-level MoE (e.g., Switch Transformer-style) under matched conditions would clarify whether layer-level routing is actually preferable.

- **Scaling beyond BabyLM**: The authors acknowledge this limitation; even one experiment at 100M+ parameters on a standard corpus would substantially strengthen claims about scalability.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"No statistical robustness / no confidence intervals"**: While true, single-run evaluation is the norm in the BabyLM community and at this model scale, so this is a community-standard practice rather than a core flaw. It remains a minor concern given the small margins.

- **"Writing quality / grammatical errors"**: The review criticisms about grammar and formatting are formatting/style nitpicks per our rules, and are removed.

- **"Vaswani 2023 citation inaccuracy"**: A minor citation error, not a substantive methodological concern.

- **"Missing related works"**: Per rules, we do not flag missing related works since we cannot verify their existence.

- **"The paper should compare to PaPaformer or other parallel architectures at the same scale"**: While a fair suggestion, PaPaformer is not an established baseline in the BabyLM framework, and asking for comparisons with methods outside the evaluation ecosystem is scope creep beyond the paper's stated scope. Moved to nice-to-have.

## Novel Insights

The paper surfaces an interesting architectural idea—compressing representations through MoE shrink/grow blocks into a smaller-dimension parallel stack creates diverse token pathways at matched total parameters—but fails to deliver convincing evidence that this particular combination meaningfully outperforms simpler alternatives. The training dynamics analysis is a genuine contribution that reveals MoEP achieves peak performance earlier but then overfits more aggressively than dense models, suggesting that sparse routing accelerates initial pattern acquisition at the cost of generalization stability. This trade-off between fast early learning and eventual overfitting is an important observation for the MoE community, though the paper initially misframes it as "stability."

## Suggestions

- **Add at least two ablation experiments**: (1) Parallel blocks *without* top-k routing (all blocks activated) to isolate the value of the router, and (2) top-k routing *without* dimensionality reduction (full dL in parallel blocks) to isolate the value of the shrink/grow design. These would directly test which components drive any observed gains.

- **Report FLOPs and active parameters per token** for each model configuration, so efficiency claims can be evaluated.

- **Reframe the "stable training" claim**: The data actually shows faster but *less stable* training for MoEP. An honest framing—"faster early learning convergence but greater risk of overfitting"—would be both more accurate and more useful to readers.

- **Run multiple seeds** and report means ± standard deviations, at minimum on a few key benchmarks, to establish that the observed margins are real.

## Score and Decision

I calibrated against: **NanoMoE** (scores: 3,3,3,3 — incrementally novel MoE with limited baselines and small scale), **ViMoE** (scores: 3,3,3,3 — incremental design study with weak baselines and missing comparisons), **Smoothness Bridges in MoEs** (scores: 1,3,3,5 — MoE stability theory with questionable assumptions and small experiments), **MoDE** (scores: 3,3,3,5 — modular experts with limited baselines and unfair comparisons), **OLMoE** (scores: 8,10,8 — strong open-source MoE with extensive analysis, scaling, and ablations), **From Sparse to Soft MoE** (scores: 6,8,8,8 — novel differentiable MoE with strong results), and **RMoE** (scores: 6,6,8 — novel recurrent router with good ablations but scaling concerns).

This paper sits firmly in the lower quality range of MoE architecture papers. Like NanoMoE and ViMoE, it proposes an architectural variant but provides limited evidence for its advantages: marginal gains without statistical robustness, no ablations, core claims contradicted by own data, and a variant (SwiGLU) that violates the paper's central principle. Unlike RMoE or Soft MoE, there is no thorough ablation or scaling story. The training dynamics analysis is a genuine contribution but insufficient to carry the paper. At best comparable to NanoMoE (scores ~3) or slightly above given the more thorough evaluation protocol.

MY FINAL SCORE: <pineapple>3.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>