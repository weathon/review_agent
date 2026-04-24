## Summary

This paper argues that discrete diffusion models are a superior paradigm to autoregressive (AR) language models for complex reasoning and planning tasks. It identifies “subgoal imbalance” as a key failure mode of AR models and proposes Multi-Granularity Diffusion Modeling (MGDM), which adds token-level adaptive reweighting to standard discrete diffusion training. Across synthetic and real-world tasks (Countdown, Sudoku, SAT), MGDM achieves dramatically higher accuracy than same-sized AR baselines, including 100% Sudoku accuracy with a 6M-parameter model.

## Strengths

- **Striking empirical gains on algorithmic reasoning benchmarks.** MGDM achieves 91.5% on Countdown-4 and 100% on Sudoku with 85M and 6M parameters, respectively, compared to 45.8% and ~20% for GPT-2 Scratch models of comparable size, and even outperforms LLaMA-13B (51.1% / 32.9%) on these tasks (Table 1, Figure 4). These are large, surprising margins that warrant attention.
- **Novel and practically motivated training extension.** The token-level reweighting term in MGDM (Equation 8) is intuitive, and ablations in Table 3 show it improves Sudoku accuracy from 87.3% to 90.4% under TopK decoding, establishing that granularity-aware weighting provides gains beyond prior diffusion models (VDM, D3PM, RDM).
- **Clean synthetic probe and qualitative error analysis.** The directed-graph planning task (§3.1, Figure 1) isolates planning distance in a controlled way, and the “regretful compromise” error analysis (§4.4, Figure 6b) reveals a qualitatively different failure mode for AR models (48.9% calculation error in the final equation vs. 0.2% in the first), which is insightful.

## Weaknesses

### Fatal
None.

### Major

- **Empirical comparison is confounded by architecture (causal vs. bidirectional attention).** The paper compares causal decoder-only AR models (GPT-2 Scratch, LLaMA) against discrete diffusion models that use bidirectional Transformer encoders. Tasks such as Sudoku and SAT are global constraint-satisfaction problems where full bidirectional context provides a known advantage independent of the diffusion noise schedule or the proposed multi-granularity reweighting. The synthetic task in §3.1 controls for model size but still uses different attention masks for AR and diffusion; the main experiments in §4 do not control for this at all. Without a bidirectional but non-diffusion baseline (e.g., iterative masked prediction) or a causal-diffusion variant, the headline results cannot be cleanly attributed to the diffusion objective or to MGDM.
- **Equation (6) mischaracterizes the diffusion objective.** The paper labels the weighted sum of cross-entropy terms in the diffusion ELBO as $-\log p_{\text{DM}}(\mathbf{x}_n | \mathbf{x}_{\neq n})$ (§3.2, Eq. 6). This is not a valid equality: the variational bound does not factorize into independent per-token conditionals given all other tokens. The term under the brace is a sum over timesteps and positions of denoising losses, not a single masked-prediction probability. By falsely equating diffusion training to a set of independent “multi-view” conditionals, the authors build an analogy to Xu et al. (2013) that lacks formal grounding. The theoretical intuition in §3.2 therefore rests on incorrect notation and should be rebuilt or presented explicitly as a heuristic analogy.

### Minor

- **Overstated “theoretical evidence.”** The introduction claims the paper provides “theoretical and empirical evidence” for subgoal imbalance, but Proposition 1 (§3.1) is an unproved informal statement rather than a theorem, and §3.2 offers heuristic intuition rather than rigorous theory. The claims should be toned down to reflect the empirical nature of the contribution.
- **Test-set construction for Sudoku may introduce distribution shift.** The paper uses “the first 100k” puzzles for training and “the subsequent 1k” for testing from a sorted corpus (§4.2). Without knowing the sorting criterion or shuffling, the test split may not be i.i.d. relative to training, which weakens the claim of 100% accuracy on 1,000 sequential puzzles.
- **Token reweighting lacks stability analysis.** The adaptive weight $v(\mathbf{x}_{t,n}) = \alpha(1-\exp(-u(\cdot)))^\beta$ depends on the loss magnitude, creating a feedback loop where harder tokens receive higher weight (§3.3). The paper provides no analysis of why this does not collapse or diverge, though it works empirically in the reported experiments.

### Trivial
None.

## Nice-to-Haves

- Add a bidirectional but non-diffusion baseline (e.g., BERT-style iterative masked prediction) in the main tasks to disentangle the benefit of bidirectional context from the diffusion objective.
- Add a causal-diffusion variant or explicitly ablate attention masking in the synthetic task to isolate the contribution of the training objective.
- Clarify or correct Equation (6) by presenting the multi-view analogy as informal intuition rather than an equality to a non-existent conditional probability.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Figure 2 caption contradiction.** The claimed contradiction between the text and Figure 2 caption is based on the parser’s auto-generated image alt text (`![...]`), not the authors’ actual caption. The paper’s real caption is the short text following the image and does not contradict the body text. This is a parser artifact, not an author error.
- **“Without using search techniques” is misleading.** Diffusion inference involves iterative denoising, but this is the standard generative procedure for diffusion models, not a “search technique” in the sense of tree search or beam search used in the LLM literature. The claim is reasonable.
- **Teacherless training as a special case of diffusion.** The paper presents this as a conceptualization (“can be conceptualized as”), not a formal mathematical claim. Criticizing it as inaccurate is a nitpick.
- **SAT framed as planning.** The paper treats SAT as a reasoning/constraint-satisfaction benchmark; calling this a “methodological gap” is scope creep.
- **Regretful compromise lacks statistical quantification.** Figure 6b explicitly reports aggregated error ratios (e.g., 48.9%, 0.2%) across the test set; this is statistical quantification.
- **Typos, grammar, and formatting issues.** These are parser artifacts from the PDF extraction and should be ignored.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

- Re-run the main experiments with an explicit architecture ablation: keep the backbone identical (e.g., GPT-2-sized Transformer) and vary only the attention mask (causal vs. full) and training objective (AR vs. diffusion). This would transform the paper’s interesting empirical observation into a rigorous causal claim.
- Rephrase Equation (6) and the surrounding text to avoid the false equality. If the multi-view intuition is retained, clearly label it as a motivating heuristic and define the plotted quantities in Figure 3 as aggregate losses rather than conditional log-probabilities.

## Score and Decision

**Calibration reasoning.** I compared this paper against several anchors from the human-review corpus:

- *High anchor (avg 8.0, Oral)*: Block Diffusion (tyEyYT267x) — a rigorous, well-controlled interpolation between AR and diffusion with strong theory and experiments. Our paper is well below this due to the uncontrolled architecture confound and the Eq. (6) mischaracterization.
- *Medium anchor (avg 6.0, Accept Poster)*: Physics of Language Models (Tn5B6Udq3E) — rich controlled experiments on synthetic reasoning, with concerns about generalizability to natural language. Our paper has stronger real-world results but a more serious confound in the main experiments (causal vs. bidirectional attention), which weakens causal attribution more than a synthetic-to-real gap.
- *Medium-low anchor (avg 5.5, Reject)*: Reparameterized Discrete Diffusion (1pTlvxIfuV) — a discrete-diffusion paper with inconsistent improvements and weak motivation, uniformly scored 5. Our paper has a clearer motivation and more striking results, but shares the weakness of missing key controlled comparisons.
- *Low anchor (avg 4.6, Withdrawn)*: SC-MCTS (F4f1afsm3R) — strong results on a single dataset, limited baselines, and unclear presentation. Our paper is stronger with multiple tasks and cleaner presentation.
- *Reject anchor (avg 7.0, Reject)*: Don’t Trust Your Eyes (OZWHYyfPwY) — strong empirical attacks but theoretical framing criticized as unconvincing. Our paper has a similar empirical-over-theory profile, though the theory here is more central to the motivation.

Relative to these anchors, the paper sits at the borderline: the empirical phenomenon is genuinely surprising and the MGDM extension is sensible, but the confounded comparison in the main experiments and the mathematical sloppiness in §3.2 are substantive methodological flaws that would need to be addressed in a major revision. This positions the paper slightly below the medium acceptance threshold.

**Score: 5.5**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>