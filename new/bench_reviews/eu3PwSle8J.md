Now let me search for calibration anchors.Now I have enough to write the final review. Let me compile my assessment based on reading the paper directly and the calibration anchors.---

## Summary

This paper proposes Augmented Intermediate Representations (AIR), a defense against indirect prompt injection attacks. The key observation is that prior instruction hierarchy (IH) defenses inject privilege signals only at the input layer, causing them to degrade as they propagate through the model. AIR addresses this by adding per-layer trainable embedding tables (indexed by privilege level) to every decoder block, maintaining IH signals throughout the network. Evaluation spans three model families (Llama-3.2-3B, Qwen-2.5-7B, Llama-3.1-8B), two training paradigms (SFT, DPO), and six attack types, reporting 1.6×–9.2× reduction in gradient-based attack success rate over the best competing method.

---

## Strengths

- **Mechanistically motivated core idea with diagnostic evidence**: Figure 3 directly measures cosine similarity between hidden representations of tokens with different privilege levels across decoder layers. For Delim and ISE, similarity rises toward 0.92–1.0 by layer 25; for AIR, it peaks at ~0.88. This cleanly demonstrates the identified limitation and motivates injecting IH signals at all layers rather than just the input.

- **Comprehensive and internally consistent evaluation matrix**: The experiments cover 3 model sizes × 2 training methods × 3 IH mechanisms × 6 attack types across 2 benchmarks (AlpacaFarm and SEP). Critically, the paper includes combinations not explored in prior work (e.g., ISE+DPO, AIR+SFT), strengthening generalizability claims.

- **Negligible parameter overhead**: For Llama-3.1-8B with K=3 privilege levels, AIR adds just 0.4M parameters (0.005% overhead), making it trivially layerable on any existing training pipeline. This is a genuine practical virtue.

- **Consistent loss-trajectory signal in Figure 7**: The attack loss curves show AIR maintaining a systematic gap above ISE and Delim throughout the optimization trajectory for all model sizes and both training methods—not just at the final step—providing evidence of robustness depth rather than merely a one-point measurement.

- **DPO corroboration and consistent trend across training methods**: Both SFT and DPO variants of AIR outperform their respective baselines on gradient-based attacks, confirming the mechanism's effectiveness is not training-method-specific.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Under-budgeted gradient-based attack evaluation**: Section 5.4 explicitly uses only **50 optimization steps for SFT models** and 200 for DPO models for both GCG and Astra. Standard practice in adversarial robustness for GCG is 500–1,000 steps with restarts. This matters directly to the headline claim: AIR injects fixed per-layer additive biases that shift the loss landscape but are constant w.r.t. token content. Even if GCG correctly differentiates through these biases, it is plausible that AIR's biases shift the loss surface in ways that slow early convergence but not ultimate convergence. At 50 steps, the paper cannot distinguish "AIR is fundamentally more robust" from "AIR's per-layer injections slow early GCG descent." The use of *different budgets* for SFT (50 steps) and DPO (200 steps) further conflates training method with attack budget, inflating the apparent advantage of DPO over SFT. This is the most important unresolved issue given that the paper's primary quantitative claim rests entirely on gradient-based attack results.

- **Inconsistent ASR metric across attack types in Table 1**: Section 5.4 confirms that static attacks measure ASR via **string match** ("determining if the model's response contains the literal phrase `hacked!`"), while gradient-based attacks measure ASR via **logit likelihood** ("the likelihood (from model's logits) of generating the target phrase"). These are fundamentally different quantities—logit probability is a soft score, string match requires actual generation. No threshold is defined to convert logit probability into a binary success. Mixing these in a single Table 1 makes across-column comparisons misleading: a defense that reduces the probability of the target phrase from 0.9 to 0.7 could appear as a major ASR drop while having minimal effect on actual generation-based success rates. Making gradient-based ASR comparable to static attack ASR requires running the optimized prefix through actual decoding and applying string match, as is done for static attacks.

### Minor

- **Figure 3 is simultaneously motivation and post-hoc result**: Section 3 uses Figure 3 to motivate the limitation of input-only IH signals, but the figure plots trained models (including AIR). This means the curve for AIR reflects the effect of training with AIR rather than an independent measurement of the limitation. The motivating observation is valid for Delim and ISE, but presenting the AIR curve in the same figure conflates motivation with result. More critically, the paper treats lower cosine similarity as a proxy for robustness without validating this link causally. A model could achieve equally low cosine similarity through heavier regularization on ISE and not achieve the same robustness gains; the link is correlational.

- **No ablation of privilege level assignment**: Section 5.3 assigns P0 to both system *and* user instruction tokens, and P1 to data segment tokens. For indirect prompt injection—where the core threat is injected content within the data segment—the relevant distinction is between user/system instructions and data tokens, so this design is reasonable. However, the paper does not ablate alternative assignments (e.g., distinguishing system from user, or using finer granularity), leaving it unclear whether this is the optimal or merely sufficient IH design.

### Trivial
*None beyond formatting artifacts from PDF parsing.*

---

## Nice-to-Haves

- **Generation-based ASR for gradient attacks**: Run the optimized adversarial prefix through greedy/temperature decoding and apply string match (identical methodology to static attacks), making Table 1 self-consistent and directly interpretable.

- **ASR-versus-steps curve at extended budget (500–1,000 steps)**: Plotting generation-based ASR as a function of GCG steps (up to 1,000) for all three IH mechanisms would directly resolve the convergence-vs.-robustness ambiguity and make the comparison interpretable even under a longer attack regime.

- **Evaluation on AgentDojo or a realistic agentic benchmark**: The paper's threat model centers on agentic LLMs processing untrusted data. AlpacaFarm is a general instruction-following benchmark; demonstrating that AIR's advantages transfer to realistic agentic injection scenarios (e.g., Debenedetti et al., 2024, which is already cited) would substantially strengthen the practical motivation.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Adaptive attack concern for GCG (Harsh Critic, Critical Issue 3, GCG component)**: The critic argues AIR's per-layer embeddings are not "adapted to" in white-box GCG. This is factually incorrect: GCG computes exact gradients of the loss w.r.t. adversarial tokens through the entire forward pass. Since AIR's per-layer embeddings s_j^k are fixed constants w.r.t. the adversarial prefix tokens, they do not impede gradient flow—GCG naturally and correctly differentiates through them. The concern is not valid for GCG. For Astra's attention-warm-start phase specifically, there is a plausible (minor) mismatch since Astra's attention-loss was designed for non-AIR architectures, but the subsequent GCG stage still computes correct gradients. This is too minor to constitute a standalone weakness.

- **RoPE analogy being mechanistically imprecise (Harsh Critic, Section 4 note)**: The critic notes that RoPE modifies Q/K dot products while AIR adds biases to the residual stream—a correct observation—but the paper explicitly frames this as an analogy for the *principle* of distributing structural signals across layers, not a claim of mechanistic identity. This is a reasonable use of analogy; removing.

- **Generic strength claims from Strength Finder**: "This paper addressed an important problem" (generic), "broader evaluation matrix" without substantive comparison (somewhat generic but verified as real; kept in modified form in Strengths above). Dropped: The strength about DPO outperforming SFT is real but is a corroboration of SecAlign's prior finding, not a novel contribution of this paper—kept as a minor supporting point rather than a primary strength.

---

## Novel Insights

The paper's core insight—that input-only privilege signals degrade monotonically across decoder layers and that per-layer re-injection of these signals can reverse this—is a genuinely underexplored design dimension in the prompt injection defense literature. The direct parallel to the evolution of positional encodings (from input-only sinusoidal to layer-wise RoPE) is instructive as a design principle, even if not mechanistically exact. The empirical finding that DPO training consistently improves adversarial robustness across all three IH injection mechanisms extends SecAlign's observation to previously untested combinations (ISE+DPO, AIR+DPO), suggesting the training-paradigm effect is mechanism-agnostic. The layer-wise cosine similarity diagnostic is a simple but useful tool for characterizing how much IH information survives propagation through the network.

---

## Suggestions

1. **Run GCG and Astra with 500 steps minimum** (ideally 1,000) with random restarts, and report generation-based ASR (string match on decoded output). This is the single most important experiment to perform before resubmission.

2. **Unify the ASR metric**: Use string-match-on-generation for all attacks (static and gradient-based) in Table 1. If logit-based likelihood is retained for any purpose, report it in a separate table or column with a clear explanation of how it differs from generation-based ASR.

3. **Separate Figure 3 into pre-training motivation (Delim, ISE only) and post-training result (all three)** to cleanly distinguish the motivating observation from the empirical result.

4. **Add a brief ablation of privilege assignment** (e.g., 2-level vs. 3-level hierarchy, or distinguishing system from user) to justify the chosen design.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| **SPIN (D-SPIN)** | `PNHGYziAsL.md` | 5.50 | Prompt injection defense with evaluation concerns and limited model coverage; adaptive attack is evaluated (a plus vs. this paper), but weaker evaluation matrix. Roughly comparable quality. |
| **SurF** | `5eqkTIQD9v.md` | 3.00 | Prompt injection defense rejected for impractical threat model and weak evaluation. Much weaker than this paper—the threat model is sound here, and the evaluation is more systematic. |
| **Tensor Trust** | `fsW7wJGLBd.md` | 7.00 | Prompt injection dataset/benchmark paper with strong novel contribution and broader scope. Higher quality than this paper due to cleaner contribution and no evaluation methodology issues. |
| **AutoDAN** | `ZuZujQ9LJV.md` | 6.25 | LLM adversarial attack paper accepted with strong novel contribution and large-scale evaluation. Higher than this paper. |
| **NEMESIS (jailbreak)** | `5kMwiMnUip.md` | 1.40 | LLM jailbreak paper rejected for trivial contribution and poor evidence. Much weaker—no comparison to this paper. |

**Positioning**: The paper is solidly above SurF (avg 3.0): the threat model is sound, the idea is novel and mechanistically grounded, and the evaluation covers multiple models and training paradigms. It is below Tensor Trust (avg 7.0) and AutoDAN (avg 6.25) because its headline quantitative results rely on an evaluation protocol with two specific methodological flaws (under-budgeted steps, inconsistent metric) that directly determine the reliability of the paper's central claim. SPIN (avg 5.5) is the closest match: a prompt injection defense with real evaluation concerns, a genuine but incremental contribution, and a borderline verdict. This paper is comparable to SPIN but slightly below due to the gradient-based evaluation problems being more central to the main claim than SPIN's limitations were. 

**Final score: 5.0 — Reject (revise and resubmit).** The idea is sound, the implementation is lightweight, and the direction of results is encouraging. However, the core quantitative claim (1.6×–9.2× reduction in gradient-based ASR) is based on an evaluation that uses a 10× to 20× smaller attack budget than community standard and an inconsistent metric that mixes logit likelihood with string match in the same table. These are not minor presentation issues—they determine whether the claimed robustness advantage reflects genuine defense strength or evaluation artifact. Acceptance requires re-running gradient-based evaluations with proper methodology.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>