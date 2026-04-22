Now I have all the information needed for a comprehensive review. Let me consolidate.

## Summary

The paper identifies "super weights" — as few as one to six individual scalar parameters in LLMs whose removal catastrophically degrades model quality (e.g., Llama-7B perplexity jumps from 7.08 to 763.65). The authors propose a data-free identification method requiring a single forward pass, analyze how super weights operate through induced "super activations" and stopword suppression, and demonstrate that preserving these few outliers makes round-to-nearest quantization competitive with state-of-the-art methods for activation quantization and enables larger block sizes for weight quantization.

## Strengths

- **Striking empirical finding with strong evidence**: Table 1 provides compelling evidence — pruning a single super weight in Llama-7B drops accuracy from 70.11 to 35.14 (perplexity from 7.08 to 763.65), while pruning the next 7,000 largest weights barely changes accuracy (70.11 → 69.22). This 7000:1 importance ratio is genuinely surprising and novel.

- **Data-free identification method with practical utility**: Section 3.1 describes how super weights can be identified via activation spikes in a single forward pass, and Table 2 provides exact coordinates (e.g., `layers[2].mlp.down_proj.weight[3968, 7003]` for Llama-7B) that can be directly applied to HuggingFace models. This is efficient and immediately reproducible.

- **Solid activation quantization result with practical value**: Table 3 shows that holding out and restoring a single super activation per tensor recovers 70–83% of SmoothQuant's improvement over naive W8A8 on Llama models, without requiring calibration data. Table 4 additionally shows the method works on OLMo models where SmoothQuant is incompatible due to non-parametric LayerNorm.

- **Cross-model consistency**: Table 2 demonstrates that super weights are consistently found in early-layer `mlp.down_proj` across Llama (7B–30B), Llama2, Mistral, OLMo, and Phi-3, suggesting this is a general architectural phenomenon rather than an artifact of one model.

- **Honest reporting of partial mediation**: The "Prune SW +SA" row in Table 1 transparently shows only 42% quality recovery, and the paper explicitly states "super activations only partially explain how super weights operate" (line 63, line 153), rather than overclaiming.

## Weaknesses

### Fatal
None.

### Major

- **Structural position vs. learned value conflation**: The paper frames its discovery as a uniquely important *learned parameter* ("super weight"), but never tests whether the catastrophic effect of pruning is due to the specific learned value or simply the architectural position (early-layer `down_proj` feeding into a skip connection). A critical missing experiment is replacing the super weight with a random large value at the same position. The scaling experiment (Figure 6, where increasing the super weight's magnitude improves quality) partially addresses this — it shows that the magnitude matters — but doesn't test whether *any* large signal at that position would produce similar effects. Without this disambiguation, the discovery is better described as "there exists a structural bottleneck in early-layer MLPs" rather than "a single specific parameter is uniquely important." This weakens the depth of the mechanistic understanding that the paper claims to provide.

- **Weight quantization contribution lacks meaningful baselines**: Section 5.2 evaluates weight quantization by comparing only against naive round-to-nearest (RTN) (Figure 7). While the paper notes that AWQ and SqueezeLLM implicitly protect super weights, it never directly compares against these methods on the same metrics. For activation quantization, there is a proper comparison against SmoothQuant (Table 3); for weight quantization, no such comparison exists. The reader cannot judge whether super-weight-preserving RTN at block size 1024×1024 is better than, worse than, or comparable to AWQ/GPTQ at standard configurations. This is a significant evidential gap for a paper that frames quantization as one of two main application contributions.

### Minor

- **Mistral-7B shows limited improvement on activation quantization**: Table 4 shows the method recovers only 14–25% of FP16 improvement on Mistral-7B, and the paper hypothesizes this is due to Mistral's LayerNorm learning weights that "aggressively suppress" the super activation. If the method's effectiveness depends heavily on the model's normalization strategy, this is a notable scope limitation that deserves clearer acknowledgment.

- **The "Prune Non-SW" baseline conflates parameter importance with structural position**: Pruning 7,000 weights distributed across the entire model is not directly comparable to pruning a single weight concentrated at an architectural bottleneck. A more informative comparison would prune other individual large-magnitude weights at different positions, to demonstrate that the super weight's position is not solely responsible. This doesn't invalidate the current comparison (the claim "1 weight > 7000 weights" still holds), but it leaves the uniqueness-of-position question open.

- **Weight quantization clipping parameter requires data for tuning**: The z-score clipping threshold in Equation 2 is tuned using 500 Wikitext-2 examples (line 213), which partially contradicts the "data-free" framing. The identification method itself is genuinely data-free, but the weight quantization application is not. This should be stated more precisely.

### Trivial
None.

## Nice-to-Haves

- Test whether replacing the super weight with a random large value at the same position preserves model quality — this would definitively distinguish structural position from learned value.
- Compare weight quantization against AWQ/GPTQ on identical metrics and configurations.
- Ablate the identification method across different prompts and sequence lengths to verify robustness of the "single forward pass" claim.
- Investigate whether super weights persist across fine-tuning in a systematic way, beyond the brief note that instruction-tuned models share coordinates.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic's claim that "the paper never tests" structural position vs. learned value**: While the specific experiment (random large value at same position) is indeed missing, the paper does provide Figure 6 showing scaling relationships, and the 42% SA recovery result partially addresses mechanism. The concern is real but should be framed as an incomplete mechanistic analysis, not a complete absence.

- **Harsh critic's claim about "misleading baseline"**: The Prune Non-SW comparison directly tests the stated claim — a single super weight is more important than 7,000 other outlier weights combined. Calling this "misleading" is too strong; it's a valid comparison for the specific claim, just incomplete as a mechanistic investigation. Moved to Minor.

- **Harsh critic's claim that "data-free" framing is undermined by z-score tuning**: This overstates the issue. The identification method (Section 3.1) is genuinely data-free. The weight quantization clipping is a small hyperparameter that requires minimal data. This is a precision issue, not a fundamental flaw. Moved to Minor.

- **Harsh critic's concern about stopword suppression being correlational**: The paper presents this as an observed effect, not a strong causal claim. The case study is explicitly labeled as illustrative. This is a minor concern.

- **Strength finder's claim about "competitive activation quantization" on OLMo**: This is already covered in the main text and Tables 3–4. The word "competitive" is used in the abstract, which is fair for the Llama results but should note the Mistral limitation. Keeping as stated.

## Novel Insights

The paper reveals an important structural insight about LLMs: that individual scalar parameters in early-layer `down_proj` matrices create persistent, prompt-invariant activation outliers ("super activations") that propagate through skip connections and disproportionately affect model output distributions. The 42% partial mediation finding actually deepens the picture — it shows super weights operate through multiple channels beyond just the super activation, suggesting more complex mechanistic pathways remain to be understood. The finding that AWQ implicitly protects super weights (scaling the super weight position up by 12×) and SqueezeLLM includes them in its sparse matrix is a meaningful connection between existing methods and this new understanding.

## Suggestions

- Add a direct comparison with at least one established weight quantization method (AWQ or GPTQ) on identical benchmarks and block sizes, so readers can assess the practical significance of the super-weight-aware approach.
- Run the "random large value replacement" experiment at the super weight's position — this single experiment would significantly strengthen (or nuance) the core claim about parameter importance vs. architectural position.
- Clearly separate the "data-free identification method" claim from the "data-free quantization" claim in the framing, since the weight quantization clipping parameter requires calibration data.

## Calibration Summary

| Paper | Path | Avg Score | Relation |
|-------|------|-----------|----------|
| Attention Sink Emergence | `78Nn4QJTEN.md` | 7.33 | Striking empirical discovery about LLM internals (attention sinks); comparable novelty profile, stronger mechanistic depth → our paper is slightly below this |
| Safety Attention Heads | `h0Ak8A5yqw.md` | 7.0 | Parameter importance finding (single head = 16× harmful outputs); similar "small intervention, large effect" profile → our paper is comparable but with weaker mechanistic analysis |
| LLM-Streamline layer pruning | `IC5RJvRoMp.md` | 7.5 | Layer pruning with good empirical results; our paper is below this in practical completeness but above in novelty of finding |
| House of Cards: Massive Weights | `LvuSFvGShf.md` | 5.25 | Directly comparable topic (massive weights in LLMs); our paper is significantly stronger with quantization applications and clearer methodology |
| PrefixQuant | `vw0NurJ7UX.md` | 3.0 | Weak quantization paper with methodological gaps; our paper is well above this |
| SpQR outlier quantization | `Q1u25ahSuy.md` | 6.5 | Outlier-aware quantization; our paper has a more striking empirical finding but less complete quantization evaluation |

This paper sits above the medium-scoring papers (5–6 range like House of Cards, pruning metrics) due to its genuinely surprising finding and practical quantization contribution, but below the high-scoring papers (7+ like Attention Sinks, Safety Heads) due to the incomplete mechanistic analysis and missing weight quantization baselines.

## Score and Decision

The paper makes a genuinely surprising and well-supported empirical discovery — individual scalar parameters whose removal catastrophically degrades LLM quality, far beyond the effect of pruning thousands of other outliers. The activation quantization application (70–83% of SmoothQuant without calibration data) is a strong practical contribution. The main weaknesses are the structural position vs. learned value conflation (which limits mechanistic depth) and the absence of weight quantization baselines (which limits the practical evaluation of the second main claim). These are real but addressable gaps, not fatal flaws.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>