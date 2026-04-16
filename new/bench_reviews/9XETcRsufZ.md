Now I have enough context from the paper and calibration papers. Let me write the final consolidated review.

## Summary

This paper investigates the performance tradeoffs between Mixture-of-Experts (MoE) transformers and standard dense transformers, arguing that increasing the number of experts improves memorization more than reasoning. The authors provide theoretical results showing that single-layer MoEs require a critical hidden dimension to solve graph reasoning problems (regardless of expert count), while MoEs can efficiently memorize data with far fewer active parameters than dense models. These findings are validated through synthetic experiments (shortest path, phone-book memorization) and large-scale pretraining evaluations showing that MoEs match dense models on world-knowledge benchmarks but underperform on commonsense and math reasoning tasks at fixed total parameters.

## Strengths

- **Timely and important research question.** Whether MoE scaling truly offers a "free lunch" is of substantial practical relevance given the widespread adoption of MoE architectures in frontier models. The paper provides a systematic, multi-faceted investigation of this question that goes well beyond the typical scaling law analysis.

- **Theory + synthetic + real empirical pipeline.** The integration of formal results (Sections 3.2–3.3), controlled synthetic experiments (Section 4), and pre-trained model evaluations (Section 5) is ambitious and provides complementary evidence at multiple levels of abstraction. Few papers on MoEs attempt this breadth.

- **Constructive memorization upper bound (Theorem 3.5).** The result that an MoE can memorize n examples with only Õ(√(nm)) active parameters is technically non-trivial and clearly illustrates how expert routing can partition the memorization workload, yielding computational (not just representational) savings.

- **Perplexity-matched comparison (Figure 6).** Comparing models at matched validation perplexity rather than only at matched parameter counts is a genuinely insightful design choice. Figure 6a shows that MoEs achieve better world-knowledge accuracy at the same perplexity, suggesting an architectural bias toward memorization that goes beyond what parameter-count comparisons reveal.

- **Generalization gap analysis (Figure 5).** The finding that MoEs exhibit larger train-test gaps on math tasks is an interesting empirical observation with potential practical implications about MoE overfitting behavior in reasoning domains.

## Weaknesses

### Major

- **Overgeneralization from depth-1 theory to general MoE claims.** The core theoretical separation (Theorem 3.2, Corollary 3.4) applies only to single-layer (depth-1) transformers with top-1 routing. The paper acknowledges that "the proof follows almost identically from the proof in (Sanford et al., 2024)" and relies on the fact that the original proof does not constrain ψ. However, the narrative throughout the paper—from the Abstract ("reasoning capabilities saturate"), to the Introduction ("MoEs offer no benefit" for reasoning), to the Discussion ("increasing the dimension d is inevitable")—generalizes well beyond what the theorem establishes. Multi-layer MoEs with residual connections could potentially distribute reasoning across layers; depth could compensate for per-layer width constraints. The paper provides no theoretical or empirical argument for why this does not happen. This gap between formal result and claimed scope is the paper's most significant weakness.

- **The memorization "separation" is about computational efficiency, not representational capacity.** Theorem 3.6's lower bound is a standard parameter-counting argument (2^{2^n} labelings vs. 2^{cW} representable functions) that applies to any finite-precision parametric model, not specifically to dense transformers. Crucially, Theorem 3.5 shows the MoE requires Õ(n + Km) total parameters—still linear in n—so both dense and MoE models require Θ̃(n) total parameters for worst-case memorization. The genuine advantage is in *active* parameters (Õ(√(nm)) vs. Õ(n)), which is a computational/FLOPs advantage, not a representational one. The paper's language often blurs this distinction, e.g., claiming "MoEs can effectively leverage a small number of active parameters with a large number of experts to memorize the data" (Abstract), which while technically accurate, is presented alongside the framing of a representational separation that the theorems do not support.

- **Pre-training experiments conflate multiple variables.** All models are trained on a fixed 65B tokens regardless of size, which disadvantages larger models (especially the 2.1B-parameter MoE) in terms of training compute per parameter—a factor that may disproportionately affect reasoning tasks. MoEs are only trained up to width 1024 while dense goes to 4098, leaving a gap in the comparison grid. The data mixture includes the downstream training sets (Section 5.1), complicating the interpretation of "memorization vs. reasoning" labels since training overlap differs across tasks. These confounds make it difficult to attribute the observed performance differences purely to architectural bias rather than training regime artifacts.

### Minor

- **Binary "reasoning vs. memorization" categorization oversimplifies task demands.** World-knowledge benchmarks like HotpotQA require multi-hop inference, and commonsense tasks like HellaSwag involve significant recall of patterns from pretraining data. The paper assigns these to opposite categories, but many tasks blend both abilities. While the empirical trends are still meaningful at a high level, the categorical language occasionally overstates the dichotomy.

- **Synthetic experiments have limited controls.** The shortest-path experiments use only 12-layer models with a limited width range (256–1024). No analysis of expert utilization, routing patterns, or load balancing is provided, making it hard to distinguish whether MoE underperformance on reasoning reflects fundamental architectural limitations or optimization failures (e.g., routing collapse on graph-structured inputs).

- **The generalization-gap interpretation is suggestive but not definitive.** Figure 5 shows larger train-test gaps for MoEs on math, interpreted as evidence that "memorization from MoEs may be harming reasoning performance." However, a larger gap could also reflect differences in optimization dynamics, representation learning, or data efficiency, rather than a direct memorization–reasoning tradeoff. No intervention experiment (e.g., controlling for memorization pressure) is conducted to establish causality.

### Trivial

- **Theorem 3.2 inherits its proof from Sanford et al. (2024).** The extension to MoEs is straightforward given that the original proof does not constrain ψ. While this limits the theoretical novelty of the reasoning lower bound, the observation that it applies unchanged to the MoE setting is itself a meaningful formal contribution.

## Nice-to-Haves

- An iso-FLOPs comparison (matching total training compute rather than parameter counts) would be practically more relevant and would help clarify whether MoEs' reasoning deficits reflect fundamental architectural limitations or simply under-training relative to parameter count.

- Analysis of expert routing patterns on reasoning vs. memorization tasks (e.g., routing entropy, expert utilization per task type) would illuminate *why* MoEs fail at reasoning, not just *that* they fail.

- Ablation with the standard FFN intermediate dimension (4d instead of d) would verify that the claimed MoE reasoning deficit is robust to this architectural choice.

- A brief discussion of whether depth-1 lower bounds likely extend to multi-layer models, or what theoretical form such an extension would take, would help bridge the theory-practice gap.

## Removed Points

- *Demand for iso-FLOPs comparison as a fatal flaw*: Iso-FLOPs comparison is a valuable addition but not strictly required—parameter-matched and perplexity-matched comparisons are meaningful ablation points. Moved to Nice-to-Have.

- *Criticism that the FFN dimension = d choice "threatens generalizability"*: While non-standard, the paper is consistent across all models (both dense and MoE use d). This is a controlled comparison, and the relative patterns should still hold. The relative ranking between MoE and dense should not be affected by this choice. Moved to Nice-to-Have as an ablation request.

- *Demand for routing mechanism ablations*: This is a reasonable future direction but the paper uses the standard Mixtral routing, which is the most widely-deployed MoE variant. Not including every routing variant is standard scope for a paper of this kind. Moved to Nice-to-Have.

- *Criticism that 65B tokens is too small*: The paper studies scaling trends across multiple model sizes, not absolute performance. While more tokens could shift absolute numbers, the relative patterns between MoE and dense are the key variable, and these are observed consistently across model sizes. The scale is adequate for the scientific question being asked.

- *Criticism that the theory uses top-1 routing while experiments use top-2*: The top-1 routing is a standard simplification for theory. The empirical results use top-2 as is standard practice. The theoretical point about width constraints under any fixed routing budget is orthogonal to this detail—if anything, top-2 is slightly more generous to MoEs, making the empirical finding more striking, not less.

- *Demand for per-task breakdowns in the main text*: Per-task results are reported in Appendix C. This is standard practice and does not constitute a weakness.

- *Criticizing the inclusion of downstream training sets in the pretraining mixture*: While this complicates interpretation, the paper is transparent about it. Inclusion of evaluation training data in pretraining is common in LLM training methodology and creates a controlled memorization signal that is actually informative for the paper's thesis.

## Novel Insights

The most novel empirical observation is from Figure 6: at matched perplexity, MoEs systematically outperform dense models on world-knowledge benchmarks while matching them on reasoning benchmarks. This suggests that MoE architectures have an implicit bias to allocate model capacity toward memorizing patterns from the training distribution before developing reasoning capabilities—a finding with practical implications for architecture selection in deployment.

## Suggestions

- Narrow the theoretical claims to explicitly state they apply to depth-1 models, and add a discussion (even speculative) of how depth might interact with the observed width requirements for reasoning.

- Reframe the memorization "separation" more carefully as a computational efficiency advantage (active parameters) rather than a representational advantage (total parameters), since both architectures require Θ̃(n) total parameters.

- Add expert utilization statistics (e.g., routing entropy, per-expert token counts) for reasoning vs. memorization tasks to illuminate the mechanism behind MoE reasoning deficits.

## Score and Decision

**Calibration.** I compared against:
- *When Can Transformers Count to n?* (scores 5–6, rejected): Similar pattern of depth-1 theoretical expressiveness results overgeneralized to broader claims, with some empirical validation. This paper has stronger empirical evidence (full pretraining) and addresses a more practically impactful question.
- *On Optimal Memorization Capacity of Transformers* (scores 6–8, accepted poster): Formal memorization capacity results with counting arguments. Similar theoretical style, but this paper has much richer empirical validation and scope.
- *Theory on MoE in Continual Learning* (scores 6–8, accepted spotlight): MoE theory paper with linear model analysis extended to DNNs. Comparable theory-practice gap issues, scored around 7.
- *Smoothness Bridges Sparsity and Stability in MoEs* (scores 1–5, rejected): Weaker MoE theory with questionable assumptions.
- *Reasoning with Latent Thoughts* (scores 6–8, accepted poster): Reasoning vs. memorization dichotomy in transformers, depth vs. width analysis.

This paper makes a genuine and important contribution in identifying a systematic tradeoff in MoE architectures, backed by theory, synthetic experiments, and real pretraining. However, the gap between the depth-1 theoretical result and the broad narrative claims, the partially misleading framing of the memorization separation, and the uncontrolled confounds in the pretraining setup are substantive weaknesses. The paper is clearly above the rejected papers in this space, but the overgeneralization from theory to practice prevents it from reaching the quality of the strongest papers.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>