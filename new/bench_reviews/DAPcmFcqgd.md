Now let me search for calibration papers to score appropriately.Now I have enough information to write the final review. Let me synthesize everything.

---

## Summary
MoEP (Modular Expert Paths) proposes a decoder-only language model architecture that achieves sparsity via layer-level parallel blocks at reduced hidden dimension, combined with MoE-style projection routing at the boundaries, while keeping total parameter count identical to a dense GPT-2 baseline. The model is trained and evaluated under the BabyLM strict-small track (10M words, 28M parameters), outperforming the BabyLM GPT-2 baseline and establishing a cleaner AoA score than GPT-BERT models.

---

## Strengths

- **Parameter-preserving sparsity is validated:** Table 2 explicitly confirms MoEP and GPT-2 both sit at 28M parameters, directly substantiating the paper's core architectural claim that sparsity can be added without expanding the parameter budget—a genuinely unusual design point in the MoE literature.

- **Transparent reporting of AoA dependency:** Section 5.1 explicitly states "MoEP achieved the highest performance… when the AoA task score was included… Even when excluding AoA from the macro average, MoEP still outperformed the BabyLM GPT-2 baseline, which we consider our primary comparison point." The authors do not hide the limitation and correctly identify the GPT-2 analog as the most relevant comparator.

- **Useful taxonomy of MoE placement strategies:** Section 2.2 provides a structured survey of FFN-level, attention-level, and layer-level MoE approaches, concisely situating MoEP in the landscape and highlighting layer-level design as underexplored.

- **Simplicity vs. complexity finding:** Table 1 documents that MoEP with linear projection experts (28M) outperforms MoEP-SwiGLU (38M) despite the latter's greater complexity. While the parameter counts differ (see Weaknesses), the finding that lightweight projections suffice at this scale is a concrete, reproducible observation.

- **Code and model weights released** (Section 4), enabling community reproduction.

---

## Weaknesses

### Fatal
None. The core architectural concept is sound in principle.

### Major

- **Load-balancing loss formulation appears mathematically inverted.** Equation 2 defines $\mathcal{L}_{\text{balance}} = -\sum_i p_i \log p_i$, which is exactly the Shannon entropy $H(p)$—always non-negative, maximized at uniform distribution. Equation 3 adds it positively to the total loss with coefficient $\lambda > 0$. Gradient descent minimizes the total loss, which means it will minimize $\lambda H(p)$, driving routing entropy *downward*—toward collapsed, single-expert routing, the exact failure mode the loss is designed to prevent. Table 3 does not report the $\lambda$ values, so it cannot be confirmed whether the sign was negated in the actual code. If the sign is wrong in the code, expert collapse may be occurring and the routing mechanism cannot be trusted; if the sign is only wrong in the paper, the description does not match the implementation. Either scenario is a serious concern, compounded by the absence of routing analysis (Contribution 3, see below) that could have provided empirical evidence that collapse did not occur.

- **Stated Contribution 3 is not delivered.** The paper claims "We analyze expert networks routing behavior and show that layer level parallelism enable fast and stable training." The appendix (A.3) and Section 5.1 show only aggregate task-accuracy curves over checkpoints—no per-expert utilization, no routing entropy trajectories, no token-type specialization analysis. This contribution does not exist in the submission. Critically, this missing analysis is exactly what would resolve uncertainty about the load-balancing issue above.

- **Primary quantitative claim (MoEP beats authors' GPT-2 rerun) rests on a 0.9-point gap from a single seed.** The margin between MoEP (49.00) and the authors' own GPT-2 (48.10) excluding AoA is 0.9 percentage points. This is a single-run result from seed 42 with no variance reported and no statistical significance test. Individual task scores differ by fractions of a percent; any random-seed variation could plausibly account for the entire gap. This is insufficient evidence to claim architectural superiority over a dense baseline.

### Minor

- **MoEP-SwiGLU has 38M parameters, not 28M** (Table 2 confirmed). The abstract frames the contribution as "keeping the total parameter count fixed," but the SwiGLU variant adds 10M parameters without explanation. The simplicity-vs-complexity comparison (Table 1) is therefore not parameter-controlled.

- **Checkpoint selection conflates model selection and test evaluation.** Section 4 describes saving 12 checkpoints and selecting final weights based on "fast evaluation scores" drawn from the same benchmark pipeline. For MoEP, this selects the 30M-word checkpoint. If the metrics used for selection overlap with those used for final reporting, this procedure can inflate results and is not described precisely enough to determine otherwise.

- **Sample efficiency claim is unsupported.** Section 5.1 claims "MoEP extracted useful patterns earlier during training," but the paper itself states "MoEP and GPT-2 achieved their best accuracy at 30M words"—the same checkpoint. The appendix documents that MoEP overfits and degrades after 30M words, which is deteriorating generalization, not superior sample efficiency.

- **Introduction overclaims relative to Section 5.1.** The introduction states MoEP "outperformed all BabyLM strict-small baseline models, including GPT-BERT" without qualification. Excluding AoA, GPT-BERT (causal) scores 54.10 vs. MoEP's 49.00—a 5-point gap. The nuanced acknowledgment in Section 5.1 does not carry through to the abstract and introduction.

### Trivial
- Table 3 ("Training setup") lists no $\lambda$ values for the balance loss terms despite these being the sole hyperparameters specific to the novel MoEP mechanism.

---

## Nice-to-Haves

- Multi-seed evaluation (3–5 runs) with variance reporting to establish statistical reliability of the 0.9-point margin.
- Explicit $\lambda$ values and sign clarification for the load-balancing terms.
- Expert utilization plots (per-expert activation frequency across training steps) to validate that routing is not collapsing.
- Ablation separating the two mechanisms: (1) MoE projection Shrink/Grow blocks and (2) Parallel Layer top-k routing. These are independent contributions and their individual effect is unknown.
- Controlled tokenizer comparison: both the authors' GPT-2 rerun and MoEP use a custom 16K vocabulary BPE tokenizer, while the BabyLM baseline uses a different one; some of the improvement over the BabyLM GPT-2 baseline may be tokenizer-driven.

---

## Removed Points
*These points were flagged for removal; treat with caution.*

- **Strength Finder claim: "MoEP achieves the highest macro average (49.00 excluding AoA) among all models."** Factually wrong. Excluding AoA, GPT-BERT causal = 54.10, GPT-BERT focus-causal = 53.65, GPT-BERT mixed-causal = 52.40, all above MoEP's 49.00. MoEP ranks fourth when AoA is excluded. Removed because it conflicts with verified Table 1 data.

- **Harsh Critic claim: "The paper never acknowledges [the AoA rank reversal]."** Incorrect. Section 5.1 explicitly acknowledges that the highest macro average holds "when the AoA task score was included" and separately notes MoEP outperforms only the BabyLM GPT-2 baseline when AoA is excluded. Removed as a misread.

- **Harsh Critic claim: "The parameter count math is not spelled out."** Table 2 explicitly reports 28M = 28M. No computation is needed; the table provides the answer directly. Removed as a nitpick without merit.

- **Harsh Critic claim: "Selecting the macro average that artificially favors MoEP is a fundamental framing problem."** Partially unfair: Table 1 reports both macro averages side-by-side (with and without AoA), and Section 5.1 is explicit about the conditions. The framing is optimistic but not deceptive. Reduced to the minor presentation point noted above.

- **Strength Finder claim: "Demonstrates faster sample efficiency via routing."** Removed because the paper explicitly contradicts it: both models peak at 30M words, and MoEP overfits thereafter.

---

## Novel Insights
The most genuinely interesting observation buried in this paper is that at 28M parameters and 10M words of training data, a sparse layer-level architecture with simple linear projection experts can match or slightly exceed a dense baseline in macro-average performance—not through richer experts, but through routing among reduced-dimension parallel blocks. The parallel SwiGLU variant's *lower* performance despite having 38M parameters suggests diminishing returns from increasing expert capacity at this regime, consistent with the broader small-model literature. Neither insight is rigorously established due to missing statistical controls, but together they hint at a parameter-regime-dependent trade-off between expert richness and expert count worth investigating with proper ablations and multi-seed runs.

---

## Suggestions
1. **Fix or clarify the load-balancing loss sign.** Either negate the term in the total loss ($\mathcal{L} - \lambda \mathcal{L}_{\text{balance}}$), show the $\lambda$ values are negative, or provide an alternative derivation. Then add per-expert utilization plots to confirm expert usage is roughly uniform.
2. **Add multi-seed evaluation** with at least 3 seeds for both MoEP and the GPT-2 rerun. With margins of <1 point, this is necessary evidence.
3. **Fulfill Contribution 3**: Replace or supplement the training dynamics appendix with actual routing analysis—expert assignment probabilities per layer, per token type, and entropy of routing distribution over training.
4. **Separate MoEP-SwiGLU into a parameter-matched ablation** or clearly present it as a larger model variant, not a same-budget comparison.

---

## Score and Decision

**Calibration anchors consulted:**

| Path | Avg score | Comparison to paper under review |
|---|---|---|
| `/human_reviews/1Ogw1SHY3p.md` (Monet) | 7.0 | High anchor — strong multi-scale MoE paper with comprehensive interpretability analysis; far stronger than this paper |
| `/human_reviews/T26f9z2rEe.md` (DynMoE) | 7.0 | High anchor — MoE architecture with extensive empirical validation across vision and language; multi-seed, multi-scale; far stronger |
| `/human_reviews/pHOH8FVrTp.md` (SmalltalkLM) | 7.33 | High anchor — asynchronous mixture of LMs with robust perplexity comparisons; clearly stronger |
| `/human_reviews/UU9Icwbhin.md` (RetNet) | 4.75 | Medium anchor — architecture paper with meaningful evaluations but structural weaknesses; this paper is weaker due to mathematical formulation issue and smaller margins |
| `/human_reviews/6ApaDkSMtX.md` (ENTP) | 5.75 | Medium anchor — small-scale architecture study, but with theoretical grounding and clearer ablation; stronger than this paper |
| `/human_reviews/VAqRZIuW8m.md` (MoDE) | 3.5 | Low anchor — modular expert architecture with limited evaluation; comparable to this paper in scope and rigor |
| `/human_reviews/7DY2DFDT0T.md` (EfficientSkip) | 2.5 | Low anchor — sparse LLM paper with weak evaluation; this paper is better because BabyLM provides a more rigorous framework |
| `/human_reviews/vOfDGYGVyj.md` (Sparse Mamba) | 2.5 | Low anchor — sparse architecture paper with serious methodological issues; this paper is better conceived but still weak |

**Positioning:** MoEP sits below the medium anchors (RetNet at 4.75, ENTP at 5.75) because of the unresolved load-balancing formulation issue, a missing key contribution (routing analysis), near-zero statistical evidence for the primary claim, and evaluation restricted to a single small-scale benchmark. It is above the weakest low anchors (2.5) because the architectural idea is coherent, the BabyLM evaluation framework is legitimate, and the authors are transparent about the AoA dependence. The closest analogue is MoDE (3.5)—similarly modest scope, single-benchmark evaluation, and limited ablation.

**Score: 3.5 — Reject**

The architectural concept is interesting and publishable in principle, but the current submission does not provide the evidence needed to support its claims: the load-balancing mechanism may be implemented incorrectly or is at minimum underdocumented, the routing analysis promised as a contribution is absent, and the quantitative edge over the dense baseline is not statistically credible. These are not cosmetic issues resolvable in a rebuttal; they require new experiments.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>