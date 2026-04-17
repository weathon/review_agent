Now I have sufficient understanding of the paper and the calibration landscape. Let me compose the final review.

## Summary

AUTO-RT proposes a reinforcement learning framework for automatic jailbreak strategy exploration in LLMs. It decomposes the attack model into a trainable strategy generator (AM_g) and a strategy rephraser (AM_r), and introduces two key techniques: Dynamic Strategy Pruning (DSP) for early termination of unpromising exploration paths, and Progressive Reward Tracking (PRT) with a First Inverse Rate (FIR) metric to smooth sparse rewards using intermediate "downgrade" models. Experiments across 16 white-box and 2 black-box LLMs show consistent ASR improvements over baselines.

## Strengths

- **Well-motivated problem and design**: The hierarchical decomposition into strategy generation and rephrasing is a sensible approach to enable strategy-level generalization across toxic intents, addressing a genuine limitation of template-based methods. The CMDP formulation for constrained red-teaming is technically sound.

- **Novel reward shaping mechanism**: PRT using downgrade models and FIR for model selection is an interesting and creative solution to the sparse reward problem in safety evaluation. The observation that FIR can identify transition points in model safety capability is empirically backed across 6 models (Figure 4) and provides a principled heuristic for downgrade model selection.

- **Comprehensive and consistent empirical gains**: Evaluation across 18 models spanning 6 families consistently shows meaningful ASR improvements (e.g., Vicuna-7B: 31.95→56.40 vs. RL; Gemma-2-2B: 6.15→48.15). The ablation study (Table 2) shows both DSP and PRT contribute independently and jointly. The diversity metrics (SeD, DeD) also show consistent improvements.

- **Ablation confirms component contributions**: Table 2 demonstrates that both +DSP and +PRT independently improve over vanilla RL, and their combination (full AUTO-RT) further improves results on most models, confirming they address complementary challenges.

## Weaknesses

### Major

- **No ablation validating the core hierarchical decomposition**: The paper's central claim is that "strategy-level exploration" via the AM_g/AM_r decomposition provides qualitatively different benefits over direct query-level optimization. However, there is no experiment comparing against a single, non-hierarchical model of comparable capacity that directly generates attack queries given a toxic intent (no rephrasing stage). Without this control, it is unclear whether the gains come from the hierarchical structure or from other components (DSP, PRT, training setup). This is not a minor missing ablation—it directly undermines the paper's central conceptual claim.

- **Missing comparisons with the most relevant baselines**: The paper cites CRT (Hong et al., 2024) and Diver-CT (Zhao et al., 2024) as closely related RL-based red-teaming methods in the related work section, yet excludes them from experiments. These are the most directly comparable baselines since they also use numerical reward signals and RL for strategy exploration. The paper justifies this by noting "limited prior research on strategic red-teaming," but CRT and Diver-CT directly address strategy-level red-teaming with diversity objectives. Their omission materially weakens the claims of superiority.

- **PRT uses non-potential-based reward shaping without sufficient validation**: The paper acknowledges that the proposed shaping is not potential-based (Ng et al., 1999), meaning it does not preserve optimal policies for the original CMDP. While this is a known trade-off in practice, there is no comparison against naive downgrade model selection (e.g., always using the first checkpoint, midpoint checkpoint, or uniformly sampling). Without showing that FIR-based selection outperforms simpler alternatives, it remains unclear whether the FIR mechanism is necessary or whether any downgrade model would suffice to provide a denser reward signal.

### Moderate

- **Framing-evaluation gap on exploitability and severity**: The introduction explicitly distinguishes exploitability from severity as dual goals, motivating the entire framework around discovering vulnerabilities that are "both high exploitability and high severity." However, the evaluation only measures ASR on a fixed intent set—a single proxy that conflates these concepts. No metric directly measures exploitability (how generic/reusable a strategy is) or severity (how harmful the output is beyond a binary label). This disconnect between the motivating narrative and the empirical evidence weakens the paper's framing.

- **Reward-evaluation circularity with Llama-Guard2**: Llama-Guard2-8B is used both as the reward signal during training (via R(a,y)) and as the evaluation metric for ASR. While this is standard practice in the field, it creates a risk of reward hacking—strategies may be optimized to exploit Llama-Guard2's specific blind spots rather than producing genuinely harmful content. The paper does not include any cross-validation with an independent safety classifier or human evaluation to confirm that the ASR gains reflect real increases in harmful output.

- **Cross-distribution concern for downgrade model construction**: Downgrade models are built using toxic data A derived from AdvBench+Alpaca, while strategies are optimized and evaluated on HarmBench splits. If A induces different failure modes than the HarmBench intents, PRT could guide the RL optimizer toward strategies effective on AdvBench-style vulnerabilities in TM′ rather than strategies that generalize to the true objective. This distributional mismatch is not discussed as a limitation.

### Minor

- **Top-100 strategy selection inflates reported ASR**: ASR_tst selects the 100 best strategies by training-set ASR before evaluating on the test set. In practical deployment, one must commit to strategies without knowing their success rate. The paper should report raw (unselected) performance or clarify that this is a retrospective evaluation protocol.

- **Limited qualitative examples**: No concrete examples of generated strategies are shown in the main text, making it difficult to assess whether AUTO-RT discovers genuinely novel attack patterns or simply more effective variants of known approaches.

- **Black-box results are substantially weaker and underdescribed**: The black-box results (Table 4) show much lower ASR (e.g., 14.88% for LLaMA 3 70B), and the ICL-based downgrade construction is only briefly described. The DeD notation (e.g., "1.17-4.32") is confusing and insufficiently explained.

### Trivial

- The "up to 16.63%" claim in the abstract appears conservative given that Table 1 shows improvements much larger than this (e.g., Vicuna-7B: +24.45 percentage points over RL). The number should be clarified or corrected.

## Nice-to-Haves

- Ablation replacing AM_g+AM_r with a single model to validate the hierarchical decomposition
- Comparison with CRT/Diver-CT on the same evaluation protocol
- Cross-validation using a held-out safety classifier (e.g., Llama-Guard3 or a different judge) to confirm ASR gains
- Concrete examples of discovered strategies to illustrate the "strategy-level" contribution
- Report of computational cost (GPU-hours) relative to baselines
- Sensitivity analysis on downgrade model selection (FIR vs. naive checkpoints)

## Novel Insights

The Progressive Reward Tracking mechanism and FIR metric represent a genuinely novel idea for addressing sparse rewards in safety evaluation. The observation that there exist transition points in model safety capability (detectable via FIR) that can serve as informative intermediate reward signals is interesting and could generalize beyond red-teaming. However, this insight is currently under-validated—in particular, it remains unclear whether FIR captures something fundamental about safety alignment degradation or merely picks a reasonable intermediate checkpoint by proxy.

## Suggestions

- Add an ablation with a flat (non-hierarchical) attack model to directly test whether the strategy–rephrasing decomposition is necessary or whether a single comparably-sized model with PRT+DSP achieves similar results.
- Run AUTO-RT against CRT and Diver-CT on HarmBench using the same evaluation protocol to properly position the method among RL-based red-teamers.
- Validate FIR by comparing against at least two simple alternatives: (a) using the midpoint downgrade checkpoint and (b) selecting the downgrade model with highest ASR on A. If FIR outperforms these, it validates the metric; if not, PRT's benefit is from denser rewards generally rather than from FIR specifically.
- Add per-model breakdowns for the human-template comparison (Table 3) to support the "near-human-level" claim.

## Score and Decision

**Calibration papers compared:**

| Paper | Topic | Scores | Decision |
|-------|-------|--------|----------|
| Active Attacks | RL-based red-teaming, diversity | 6,4,4 (avg 4.7) | Reject |
| RedTopic | Topic-diverse red-teaming, RL+diversity | 4,6,4,8 (avg 5.5) | Reject |
| iART | Imitation-guided red-teaming, RL | 2,6,6 (avg 4.7) | Reject |
| CCR | RL-based jailbreak, dense reward | 4,2,2,6 (avg 3.5) | Reject |
| ADJ | Dialectic jailbreak, game-theoretic | 8,6,4,4 (avg 5.5) | Accept Poster |
| Bypassing Prompt Guards | LLM attack, prompt guards | 2,4,2,2 (avg 2.5) | Reject |
| RewardMap | Sparse reward RL, multi-stage | 2,4,8,6 (avg 5.0) | Accept Poster |

AUTO-RT is stronger than CCR and Bypassing Prompt Guards on methodology and evaluation scope. It is comparable to RedTopic and Active Attacks in the same space: similar RL-based approach to red-teaming, similar weaknesses (missing baselines, single-classifier evaluation). AUTO-RT has more comprehensive experiments and more techniques, but shares the same core issues. It is weaker than ADJ, which has theoretical guarantees and more systematic evaluation, and received mixed-but-accepted reviews. The missing baseline comparison with CRT/Diver-CT and the lack of validation for the hierarchical decomposition are significant gaps that prevent confident assessment of the contribution. The paper is weaker than the accepted ADJ and RewardMap, and comparable to the rejected RedTopic and Active Attacks, possibly slightly stronger due to more comprehensive experiments, but pulled down by the ablation gap and missing baselines.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>