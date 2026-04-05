=== CALIBRATION EXAMPLE 62 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate: the paper does introduce a group-preference optimization method for diffusion models. However, “Reinforcing Diffusion Models” is somewhat generic; the actual novelty is more specific: an online preference optimization objective that avoids stochastic-policy rollouts and instead uses group-level Bradley-Terry-style learning.
- The abstract is not visible in the extracted text, so I cannot assess whether it clearly states the problem, method, and key results. This is itself a concern for review completeness, though likely a parsing artifact rather than a paper issue.
- The main result claims in the introduction and conclusion are very strong: “around 20× faster training” and “boosts GenEval from 63% to 97%.” These claims are supported by later tables/figures, but the speedup appears to depend on the specific rollout/training setup and may not generalize. The paper should be careful not to overstate universality.

### Introduction & Motivation
- The motivation is clear and timely: ICLR readers will recognize the gap between RLHF-style alignment methods for LLMs and diffusion models, especially the mismatch between GRPO-style policy-gradient methods and deterministic ODE sampling.
- The paper does a good job identifying why prior diffusion-GRPO methods are expensive: they require SDE rollouts, full trajectory training, and model-agnostic noise. This is a meaningful gap.
- The contributions are stated clearly: DGPO, an online method using group-level preference optimization and ODE rollouts; a timestep clip strategy; and empirical improvements on GenEval, OCR, and PickScore.
- However, the introduction somewhat over-attributes the success of GRPO to “group-level information” alone. That is plausible, but the paper does not yet establish that this is the decisive factor rather than other elements of the RL setup, such as exploration or on-policy adaptation. This is a substantive hypothesis, not yet a demonstrated fact.

### Method / Approach
- The method is explained in a generally understandable sequence: preliminaries on diffusion/RLHF/GRPO/DPO, then group-level Bradley-Terry optimization, then a weighting scheme, then a final diffusion loss.
- The most important methodological issue is in the derivation from Eq. (8) to Eq. (17). The paper claims to derive a direct group-preference objective and eliminate the partition function \(Z(c)\) via the advantage-based weights \(w(x)=|A(x)|\). But the logical connection is not fully convincing:
  - Eq. (11) still contains \(Z(c)\) in a form that is not obviously removed by the weighting scheme.
  - The claim that \(\sum_{x\in G^+} w(x)=\sum_{x\in G^-} w(x)\) because normalized advantages have zero mean is not sufficient to justify cancellation inside the log-sigmoid expression. The paper seems to rely on a heuristic symmetry rather than a rigorous equivalence.
  - The move from Eq. (15) to Eq. (16) via Jensen’s inequality is mathematically plausible in form, but the final objective in Eq. (17) is still somewhat opaque: the transition from log-probability ratios to a difference of diffusion denoising losses needs a cleaner, fully annotated derivation.
- The parameterization of group reward in Eq. (9) is also conceptually underspecified. If \(R_\theta(G|c)\) is a weighted sum of per-sample rewards, why is the weight chosen as absolute normalized advantage, rather than, say, a signed or softmaxed function? The justification is mainly convenience and cancellation, not principled group modeling.
- Edge cases are under-discussed:
  - What happens when all group rewards are equal and the standard deviation in Eq. (12) is near zero?
  - What if a group has no positive or no negative members under Eq. (13)?
  - How sensitive is the method to group size \(G\) and reward noise?
- The timestep clip strategy is plausible, but it is presented as a fix for degraded sample quality from few-step rollouts rather than derived from first principles. The paper should clarify whether this is specific to OCR-like tasks or generally beneficial.
- Reproducibility is moderate, but not yet ideal for ICLR standards: important implementation details like exact reward models, the frequency of online sampling vs parameter updates, EMA schedule rationale, and how the “shared noise among samples within the same complete groups” is implemented should be more explicit.

### Experiments & Results
- The experiments do test the core claims reasonably well:
  - Table 1 evaluates GenEval compositional generation.
  - Table 2 covers OCR accuracy, PickScore, and out-of-domain quality metrics.
  - Figs. 3–5 address training speed and ablations.
- The main empirical claim is supported: DGPO substantially outperforms SD3.5-M and Flow-GRPO on the reported benchmarks, especially GenEval (0.97 vs 0.95 Flow-GRPO and 0.63 base).
- That said, there are several concerns about fairness and completeness:
  - The baselines are limited. The paper compares heavily against Flow-GRPO and a few pretrained/foundation models, but does not systematically compare against other relevant diffusion-alignment methods in the same training regime, especially diffusion-DPO and other recent RL fine-tuning variants, beyond a small ablation.
  - The table suggests comparison against GPT-4o and Janus-Pro on GenEval, but those are not directly comparable training pipelines to SD3.5-M. Their inclusion is interesting but can be misleading if readers infer a like-for-like RL comparison.
  - There are no reported error bars, confidence intervals, or multiple runs. For an ICLR paper making strong claims about improved alignment and speed, this is a notable omission. Metrics like GenEval and PickScore can have nontrivial variance.
  - The paper does not appear to provide sample-efficiency curves with matched compute budgets across all baselines, only selected training-time plots. Since the headline claim is 20×–30× speedup, compute normalization is crucial.
  - The ablation on timestep clipping is too narrow. It would be materially useful to see sensitivity to \(t_{\min}\), group size \(G\), weight design in Eq. (14), and whether the method still works without EMA.
- The “out-of-domain” metrics are a good idea, but they are still all image-quality proxies. They help check for reward hacking, yet they do not fully validate that the method preserves broader generative diversity or prompt fidelity.
- There is some risk of cherry-picking in the presentation: the strongest numbers are emphasized, while failure cases and negative tradeoffs are only lightly discussed.

### Writing & Clarity
- The paper’s overall narrative is clear: diffusion models lack a natural stochastic policy, so DGPO bypasses policy gradients and trains directly on group preferences.
- The method section is harder to follow than it should be, especially around Eqs. (11)–(17). The derivation is mathematically dense and, as presented, does not make the cancellation logic fully transparent.
- Figures and tables, as extracted, are partly garbled by parser/OCR issues, so I do not judge their visual formatting. But conceptually:
  - Table 1 and Table 2 are useful and align with the main claims.
  - Figs. 3–5 seem relevant for speed and ablation, though the textual description of what each curve exactly measures could be more precise.
- The paper would benefit from a clearer statement of the algorithmic difference between DGPO, Flow-GRPO, and Diffusion-DPO right at the start of the method section. Right now the novelty is understandable, but not maximally crisp.

### Limitations & Broader Impact
- The limitations section is too thin for ICLR expectations. It states only that the method focuses on text-to-image synthesis and could extend to text-to-video.
- Missing limitations that matter:
  - Dependence on a reward model during online training, which may itself introduce bias or reward hacking.
  - Sensitivity to reward noise and to the quality/calibration of the reward model.
  - Dependence on a strong base model (SD3.5-M); it is unclear how much of the gain comes from the post-training method versus the backbone.
  - Potential degradation in diversity or robustness when aggressively optimizing group preferences.
- The ethics statement is very limited. While the paper is about image generation, it should at least acknowledge risks of misuse, content generation harms, and the possibility of amplified bias from preference optimization.
- The reward-hacking discussion in Appendix D is helpful, but it does not substitute for a broader limitations analysis.

### Overall Assessment
This is a promising and potentially impactful ICLR submission. The core idea—replacing stochastic-policy-based diffusion GRPO with a direct group preference objective over efficient ODE rollouts—is appealing and empirically strong, and the reported gains on GenEval and training time are substantial. The main concerns are not about the intuition, but about the rigor of the derivation and the strength of the empirical evidence. In particular, the path from the Bradley-Terry group objective to the final DGPO loss is not fully convincing as written, and the experiments would be more persuasive with stronger baselines, error bars, and broader ablations. Still, the contribution appears real and relevant for ICLR; if the derivation can be tightened and the evaluation broadened, this could be a solid acceptance-level paper.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes DGPO, an online reinforcement learning method for diffusion model post-training that aims to combine the group-relative preference learning idea of GRPO with the direct-preference-style simplicity of DPO. The core claim is that diffusion models can be aligned more efficiently by directly optimizing group-level preferences using ODE-based rollouts, avoiding stochastic-policy/SDE requirements and reducing training cost while improving performance on text-to-image benchmarks.

### Strengths
1. **Addresses an important and timely problem for ICLR.** The paper targets reinforcement learning / preference optimization for diffusion models, a fast-moving area with clear practical relevance for generative modeling and alignment. This is a meaningful extension of RLHF-style methods beyond language models.
2. **Clear high-level motivation for the method.** The authors identify a concrete mismatch between GRPO-style policy-gradient methods and diffusion samplers, namely the reliance on stochastic SDE rollouts. The paper makes a compelling argument that this mismatch causes inefficiency and motivates a direct optimization approach.
3. **Potentially strong empirical efficiency gains.** The paper reports large speedups over Flow-GRPO, including around 20× overall training speedup and nearly 30× faster training on GenEval, while claiming superior quality. For ICLR, such improvements in compute-efficiency are highly relevant if the comparisons are sound.
4. **Broad evaluation across multiple tasks and metrics.** The experiments cover compositional image generation, visual text rendering, and human preference alignment, and also include out-of-domain quality metrics. This is better than evaluating on a single benchmark and suggests some attempt to assess both target reward improvement and general quality preservation.
5. **Ablation studies for key design choices.** The paper studies timestep clipping, ODE vs. SDE rollouts, offline vs. online DGPO, and comparison with Diffusion-DPO. These ablations help support the claimed role of the method’s components.
6. **A reproducibility statement is included.** The paper states that code and experimental setup will be released, which is consistent with ICLR’s expectations for empirical reproducibility.

### Weaknesses
1. **Theoretical derivation is not fully convincing or clearly grounded.** The key objective derivation from group preferences to the final diffusion-loss form is quite intricate, but the paper does not make the assumptions explicit enough to verify that Eq. 17 is a principled consequence of the earlier objectives rather than a heuristic approximation. In particular, the transition from group-level BT modeling to a weighted sum over sample-level diffusion scores feels under-justified.
2. **The method seems close to a combination of existing ideas, limiting novelty.** The paper builds directly on GRPO, DPO, and Diffusion-DPO, and the main novelty is the “group” extension plus ODE rollouts. While this is a reasonable synthesis, the conceptual leap appears moderate rather than clearly substantial compared with the strongest ICLR papers, which typically require a more distinctive algorithmic or theoretical insight.
3. **The experimental claims may be stronger than the evidence presented.** The paper reports large improvements over Flow-GRPO, but the tables do not make clear whether all baselines were tuned equally, whether multiple seeds were used, or how variance was measured. For ICLR, strong claims like “state-of-the-art” and “30× faster” need careful statistical and methodological backing.
4. **Fairness of baseline comparison is not fully established.** Flow-GRPO is compared with DGPO, but the paper’s own method benefits from ODE rollouts, group preference weighting, timestep clipping, and possibly different optimization dynamics. It is not fully clear whether the gain comes from the proposed principle or from a bundle of training changes.
5. **Limited scope of evaluation domain.** The experiments are all on text-to-image generation, primarily on SD3.5-M and related metrics. The paper claims broader applicability, but does not test other diffusion settings or modalities beyond a brief future-work mention.
6. **Reproducibility details are incomplete for an ICLR standard.** The paper gives some implementation details, but key information needed for exact replication is missing or underspecified, such as reward model training details, exact seed counts, hyperparameter sweeps, and how the online/offline training data are sampled across iterations.
7. **Some conceptual claims are overstated.** Statements that the method “circumvents the policy-gradient framework entirely” and that GRPO’s success is mainly due to group information are plausible but not strongly established. The paper does not provide enough ablation or analysis to isolate these factors convincingly.
8. **Limited analysis of failure modes and robustness.** There is a brief discussion of reward hacking and a visualization, but little systematic study of robustness, sensitivity to group size, reward model noise, or prompt distribution shift.

### Novelty & Significance
For ICLR standards, the paper is **moderately novel** rather than highly novel. The main contribution is a sensible and potentially useful bridge between group preference learning and diffusion model alignment, and the efficiency results could be significant if robustly validated. However, the method appears to be a fairly incremental synthesis of existing GRPO/DPO/diffusion-alignment ideas rather than a fundamentally new learning paradigm.

In terms of significance, the problem is important and the reported speed/quality gains are potentially impactful for practical diffusion post-training. That said, the paper’s acceptance prospects at ICLR would depend heavily on whether the empirical gains are shown to be robust, statistically reliable, and attributable specifically to DGPO rather than to implementation or sampling advantages.

Clarity is mixed: the motivation and overall algorithm are understandable, but the derivation is dense and hard to follow, making it difficult to assess correctness. Reproducibility is moderate: the paper includes some training details and promises code release, but not enough detail for full confidence. Overall, this is a promising applied methods paper with useful ideas, but it does not yet read as a clear top-tier ICLR breakthrough.

### Suggestions for Improvement
1. **Strengthen the theoretical justification.** Clearly state all assumptions used to derive the DGPO objective, and separate exact derivations from approximations. A cleaner proposition/lemma structure would help readers verify the method.
2. **Add more rigorous ablation studies.** Isolate the effect of each component: group weighting, ODE rollout, timestep clipping, EMA usage, and online vs. offline updates. This would clarify which parts drive the gains.
3. **Report variance and statistical reliability.** Include multiple random seeds, mean/std or confidence intervals, and significance testing for key benchmark results and speed comparisons.
4. **Improve baseline fairness and transparency.** Explain baseline tuning procedures, rollout budgets, and any differences in compute or sampling steps. Ideally, compare against stronger or more carefully matched baselines, including variants that share some but not all of DGPO’s design choices.
5. **Expand evaluation beyond one model family.** Test DGPO on at least one additional diffusion backbone or task to demonstrate generality beyond SD3.5-M and the selected reward settings.
6. **Provide a more detailed reproducibility appendix.** Include exact hyperparameters, optimizer settings, reward model specifics, prompt sets, group construction details, seed values, and compute accounting.
7. **Add analysis of sensitivity and robustness.** Study how performance changes with group size, reward noise, timestep clipping threshold, and reward model quality. This would help establish practical reliability.
8. **Tone down over-strong claims unless better supported.** Rephrase statements about “circumventing” policy gradients or the fundamental cause of GRPO’s success unless backed by stronger evidence or theory.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison to Diffusion-DPO and Flow-GRPO under matched compute, rollout steps, and reward signals on the same SD3.5-M backbone. ICLR reviewers will not accept a claim of “better than prior GRPO/DPO-style methods” without showing that gains persist when the baselines are tuned fairly and evaluated on the same tasks.

2. Add ablations on group size, group partitioning, and weighting \(w(x)=|A(x)|\), including alternatives such as uniform weights, rank-based weights, and no positive/negative split. Without this, the paper’s core claim that “group-level preference information” is the key contribution is not convincing.

3. Add a compute-normalized comparison that separates rollout cost, reward evaluation cost, and backprop cost, ideally with wall-clock and total FLOPs. The current “20–30× faster” claim is too coarse for ICLR standards because speedups can be dominated by rollout length, optimizer settings, or logging frequency.

4. Add baselines that isolate whether the gains come from ODE sampling rather than DGPO itself, such as offline DPO/DGPO using the same ODE samples, and online DGPO with SDE samples. Without this, the paper does not establish that the proposed objective—not just better sampling—drives the improvements.

5. Add evaluation on at least one additional backbone or reward signal beyond SD3.5-M + PickScore/GenEval/OCR. ICLR expects evidence of method robustness; a single-model, single-family result leaves open that the method is overfit to one architecture or one reward setup.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze why the advantage-based weights and zero-mean partition make the objective well-behaved in practice, including sensitivity to reward scale and reward noise. Right now the derivation is formal, but the paper does not justify that the chosen weighting actually stabilizes learning instead of amplifying noisy rewards.

2. Provide a failure-mode analysis showing when DGPO harms image quality, collapses diversity, or overfits to reward hacking. The paper claims strong out-of-domain quality preservation, but ICLR reviewers will want evidence that these gains do not hide systematic degradation in diversity or robustness.

3. Quantify how much each component contributes to training stability: ODE rollout, timestep clipping, EMA vs current-policy generation, and shared noise across a group. Without this decomposition, it is unclear which part of DGPO is the real algorithmic advance and which part is a heuristic patch.

4. Include statistical variance across multiple random seeds for the main results. The reported gains are large enough to be interesting, but for ICLR they still need seed-wise variability to show the method is reliably better rather than occasionally lucky.

5. Analyze whether DGPO actually improves calibration between reward scores and human-aligned quality, not just benchmark metrics. A method that optimizes GenEval/OCR or PickScore can still exploit reward proxies, so the paper needs evidence that the learned model improves the underlying alignment target.

### Visualizations & Case Studies
1. Show per-group reward distributions before and after training, including the positive/negative partition and the induced weights. This would reveal whether DGPO is truly using fine-grained group structure or just turning groups into pseudo-pairs.

2. Add trajectories of training curves for main metrics, out-of-domain metrics, and wall-clock time on the same plot for DGPO vs baselines. The current figures do not let readers judge whether the speedup comes with slower metric improvement early on or later instability.

3. Show side-by-side failure cases for each benchmark category: counting, spatial relations, text rendering, and human-preference prompts. These are the clearest way to see whether DGPO improves semantics or merely produces visually nicer samples that still fail specific constraints.

4. Visualize reward hacking examples across multiple reward models, not just OCR and HPS/PickScore. If the method is genuinely robust, it should not merely transfer reward overoptimization from one proxy to another.

### Obvious Next Steps
1. Run a full fairness study against prior diffusion RL methods under identical sampler, prompt set, and training budget. This should have been in the paper because the central claim is efficiency and superiority over Flow-GRPO-style approaches.

2. Add a principled justification or alternative derivation for why group-level Bradley-Terry optimization is preferable to pairwise DPO-style training in diffusion. The current argument is mostly conceptual; ICLR would expect a clearer theoretical or empirical bridge from pairwise preference learning to the group objective.

3. Extend the method to a more diverse set of generation tasks or modalities, not just text-to-image. The paper itself frames DGPO as broadly applicable, but only one modality is tested, so the generality claim is premature.

4. Include a reproducibility appendix with exact training schedules, reward model versions, prompt sources, and total sampled images. For ICLR, this is not optional when the paper’s main claims hinge on training efficiency and benchmark gains.

# Final Consolidated Review
## Summary
This paper proposes DGPO, an online reinforcement-learning-style post-training method for diffusion models that tries to import the group-relative signal of GRPO while avoiding stochastic-policy rollouts. The core idea is to train directly from group-level preferences over ODE-generated samples, with an advantage-based weighting scheme and a timestep-clipping trick to stabilize few-step rollout training.

## Strengths
- The paper targets a real and important gap: existing GRPO-style diffusion alignment methods are tied to SDE rollouts and full-trajectory optimization, which are indeed expensive and awkward for diffusion samplers. The proposed shift to ODE-based rollouts is a sensible practical direction.
- The empirical results are strong on the reported tasks: DGPO improves GenEval substantially over the SD3.5-M baseline and reports large gains over Flow-GRPO while also claiming much faster training. The paper also evaluates on OCR accuracy, PickScore, and several out-of-domain quality metrics, which is better than a single-metric evaluation.
- The ablations at least partially support the intended design choices, especially the timestep-clipping strategy and the ODE-vs-SDE rollout difference. The visualization of reward hacking is also a useful sanity check.

## Weaknesses
- The main derivation is not convincing enough. The path from group Bradley-Terry preferences to the final DGPO diffusion loss is mathematically dense, and the crucial cancellation of the partition function is not made rigorous. The weighting choice \(w(x)=|A(x)|\) looks more like a heuristic that makes the algebra work than a principled derivation. This matters because the entire method rests on that objective.
- The paper’s strongest claim — that DGPO is better and much faster than prior diffusion GRPO/DPO-style methods — is not yet supported with enough controlled evidence. The baselines are not exhaustively matched on sampler, rollout budget, seeds, or compute accounting, and there are no error bars or variance estimates. For a paper making 20×–30× speedup claims, that is a major omission.
- The novelty is moderate rather than deep. DGPO is a fairly direct synthesis of GRPO, DPO, and diffusion-DPO ideas, plus the use of ODE rollouts and a weighted group objective. That may be useful, but the algorithmic leap is not as substantial as the paper’s rhetoric suggests.
- Evaluation scope is narrow. Everything is centered on SD3.5-M and text-to-image tasks. The paper hints at broader applicability, but does not demonstrate robustness across backbones or modalities. That makes the generality claim premature.

## Nice-to-Haves
- A cleaner theorem/lemma-style derivation that explicitly separates exact identities from approximations would make the method much easier to trust.
- More ablations on group size, weighting choice, and sensitivity to reward noise would help establish that the group construction is actually doing useful work rather than acting as a repackaged pairwise objective.
- A more detailed failure-mode analysis would be helpful, especially on diversity collapse, reward hacking, and whether quality gains persist under stronger optimization.

## Novel Insights
The paper’s most interesting insight is that the apparent “GRPO advantage” for diffusion may come less from policy gradients themselves and more from the availability of group-relative preference information. That is a plausible and practically useful reframing: if true, it explains why a diffusion-native direct objective can outperform a policy-gradient adaptation while using better samplers. The catch is that the current implementation does not fully prove this conceptual story; the method seems to combine several efficiency tricks, so it is still unclear how much of the gain comes from the group-preference formulation versus the switch to ODE rollouts and the specific clipping/weighting heuristics.

## Potentially Missed Related Work
- **Diffusion-DPO (Wallace et al., 2024)** — directly relevant because DGPO is positioned as a group extension of diffusion DPO.
- **Flow-GRPO (Liu et al., 2025)** — the main online RL baseline and closest comparison point.
- **DanceGRPO (Xue et al., 2025)** — another GRPO-style adaptation to visual generation that should be discussed carefully in any comparison of policy-gradient diffusion alignment.
- **Towards self-improvement of diffusion models via group preference optimization (Chen et al., 2025a)** — relevant because it also exploits group information in diffusion preference learning.

## Suggestions
- Provide a much cleaner derivation of Eq. 17, explicitly stating which steps are approximations and why the weighting scheme is valid, rather than relying on algebraic cancellation arguments that are hard to verify.
- Add a fairness study against Flow-GRPO and Diffusion-DPO with matched rollout steps, identical reward signals, identical backbone, and multiple seeds.
- Report mean/std or confidence intervals for all main metrics and training-time comparisons.
- Add ablations for group size, weighting function, EMA usage, and the timestep-clipping threshold.
- If space permits, test DGPO on at least one additional diffusion backbone to support the broader applicability claim.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 8.0, 6.0]
Average score: 6.0
Binary outcome: Accept
