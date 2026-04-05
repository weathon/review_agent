=== CALIBRATION EXAMPLE 77 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is mostly accurate: the paper introduces a weighted policy optimization method, `wd1`, for reasoning in diffusion language models.
- The abstract clearly states the problem, the core idea, and the claimed empirical gains. It also distinguishes `wd1` from `wd1++`, which is helpful.
- The strongest concern is that the abstract makes very broad claims about “outperform[ing] diffusion-based GRPO” and “state-of-the-art math performance” without enough context on the evaluation regime. In particular, the +59% accuracy claim is on Sudoku, where the baseline is quite weak; the abstract could be read as implying more general superiority than the paper actually establishes.
- The “theoretical soundness” claim is presented strongly, but the paper’s theory section is complex and somewhat dependent on idealized assumptions; the abstract should be more careful about the scope of those results.

### Introduction & Motivation
- The motivation is strong and timely for ICLR: RL for diffusion LLMs is an important open problem, and the paper identifies a real practical bottleneck in likelihood-ratio estimation for dLLMs.
- The gap in prior work is reasonably identified: diffusion-based GRPO methods require multiple likelihood approximations and can suffer from high variance and approximation error.
- The contributions are stated clearly, but some claims feel slightly overextended. For example, the paper suggests `wd1` “dispenses with explicit policy ratios” and avoids approximation error; that is true for the training objective, but the method still relies on approximate likelihoods for `d1`-style token likelihood estimation in practice (Section 3.2 and Appendix B.3).
- The introduction does not sufficiently acknowledge that the method is conceptually close to weighted regression / advantage-weighted log-likelihood / negative-sample penalization already known in AR settings. The novelty lies in the dLLM adaptation and the stepwise extension, but that distinction should be sharper.

### Method / Approach
- The method is generally understandable, but several key derivations are difficult to verify from the main text alone.
- In Section 3.1, the transition from reverse-KL regularized optimization to the weighted log-likelihood objective in Eq. (6) is plausible, but the exact assumptions under which the group-normalized weights in Eq. (7) preserve the optimum are not fully discussed. The dependence on group sampling and the replacement of the true advantage by group-relative advantage is a material approximation.
- The core `wd1` loss in Eq. (8)–(9) is intuitive, but the justification for the specific form  
  \[
  -w^+(q,o_i)+w^-(q,o_i)
  \]
  is not as rigorous as the paper’s tone suggests. In particular, it is not clear why the symmetric negative term is the “right” correction rather than one of several possible choices.
- The method still depends on likelihood approximation via `d1`-style estimators for `log πθ(o|q)` (Section 3.2 and Algorithm 1). So while it is ratio-free, it is not approximation-free. This nuance matters because the abstract and intro emphasize avoiding approximation error.
- The sampling scheme using a geometric mixture of old and reference policies is underexplained operationally. Appendix B.3 hints at alternatives, but the main text does not fully clarify when the reference model is required and how costly this sampling step is relative to the gains from removing ratios.
- For `wd1++`, the stepwise extension in Section 3.3 is promising, but the construction of the expanded set \(\{O_i\}\) and the exact weighting of intermediate denoising states need more careful explanation. It is not fully clear how duplicates, varying-quality intermediate states, or different denoising depths are handled.
- The theoretical interpretation in Section 4 is interesting but ambitious. The mapping to energy-guided diffusion and unlearning is elegant, yet some steps appear to rely on strong equivalences between DCE, CSM, DSE, and DCE-based likelihood approximations. These are plausible in spirit, but the paper does not clearly delimit where the equivalence is exact versus approximate.
- Failure modes are partially acknowledged only later. Important edge cases include: all samples in a group receiving the same reward, reward sparsity, and whether large negative weights could destabilize training when rewards are noisy.

### Experiments & Results
- The experiments do test the main claims: reasoning accuracy, training cost, and the benefit of negative-sample reinforcement.
- The main comparison against `d1` is appropriate, since `wd1` is positioned as a replacement for diffusion-based GRPO-style optimization.
- That said, the evaluation is not fully exhaustive relative to the claims. The paper emphasizes computational efficiency and ratio-free optimization, but the empirical cost analysis is limited to selected metrics and a short training window; it does not report wall-clock variability across runs, confidence intervals, or end-to-end sampling/training tradeoffs under different sequence lengths in a systematic way.
- The results on Sudoku and Countdown are striking, but the magnitude of improvement there raises a question: these tasks are relatively small and rewardable, so large gains may not generalize to more complex reasoning settings. The paper partially addresses math tasks, where gains over `d1` are modest for `wd1` and larger for `wd1++`.
- The math results are mixed in a way that deserves more careful presentation. In Table 1, `wd1` matches or only slightly improves on `d1` for GSM8K and MATH500 at 256/512, while `wd1++` drives the stronger gains. This suggests that the core `wd1` objective alone is not uniformly superior on harder reasoning tasks.
- The baseline set is decent but incomplete for the strongest claims. Since the paper also positions itself against recent RL-for-dLLM methods, it is good that Table 3 includes SDPO, TCR, and MDPO. However, the comparison is mostly on final accuracy; there is limited evidence that the comparison is fair across equal compute or equal tuning budgets.
- A major concern is reproducibility/fairness of comparisons when some methods are full fine-tuning and others use LoRA, and when `wd1++` is trained with a different dataset and a different reference setting (\(\beta=0\), \(\lambda=1\)) than the main `wd1` runs.
- Error bars, multiple seeds, and statistical significance are largely absent from the main results. Given that many gains are modest on GSM8K/MATH500 and that some figures show seed sensitivity, this is a material omission for ICLR standards.
- The ablations are relevant: removing negative weights hurts performance (Table 4), and varying the combined weight affects training. But the ablation suite would be more convincing with a cleaner isolation of design choices: group size, reward normalization, choice of \(\psi\), sampling strategy, and the effect of intermediate-step inclusion in `wd1++`.
- Table 2’s cost comparison is useful, but the NFEs accounting is somewhat narrow. The paper states that sampling costs are excluded because both methods share them, but `wd1` changes the sampling regime and uses a geometric mixture setup; this deserves a more careful end-to-end cost comparison.
- The coding benchmark in Appendix C.3 is a nice extra, but it is peripheral and underpowered relative to the main claims. It does not materially strengthen the central contribution.

### Writing & Clarity
- The paper’s overall narrative is coherent, but the theory sections are hard to parse in the current form. This is not a minor stylistic issue: several derivations are difficult to evaluate because the exposition jumps between reverse-KL optimization, weighted regression, energy-guided sampling, and concrete-score matching without a clean road map.
- Section 4 is especially dense. The claims of equivalence between `wd1`, AW-DCSM, AW-DCE, and energy-guided diffusion are interesting, but the presentation makes it hard to separate exact results from heuristic interpretations.
- Figures and tables generally serve their purpose, especially Tables 1–4. However, Figure 1 and some ablation figures are not fully informative without more quantitative context. In particular, the reader would benefit from clearer reporting of variance across runs and clearer labels for the exact training checkpoints shown.
- The experimental narrative sometimes mixes claims about speed, NFEs, rollouts, and accuracy in ways that make it hard to tell which advantage comes from the objective itself and which comes from training-budget differences.

### Limitations & Broader Impact
- The paper does include a limitations section, which is good. It correctly notes issues with identical-reward groups, the restriction to text-based reasoning, and reliance on approximate likelihoods.
- However, the limitations discussion is incomplete relative to the method’s own claims. It should more directly acknowledge:
  - sensitivity to reward design and reward noise,
  - dependence on group sampling and group-relative normalization,
  - instability risk from strongly negative updates,
  - and the fact that the method’s strongest gains are on tasks with verifiable rewards.
- Broader impact is not deeply discussed. Since the method is about RL for reasoning models, the likely societal implications are mostly indirect, but the paper should at least note that improved reasoning capability can be used for both beneficial and potentially harmful automation.
- The “no ethics concern” statement is too terse for a paper that meaningfully changes model behavior via reinforcement learning, though this is a common issue rather than a fatal flaw.

### Overall Assessment
`wd1` is a solid and potentially useful contribution to RL for diffusion language models. Its main idea—recasting policy optimization as weighted log-likelihood with positive and negative sample terms—is sensible, and the empirical improvements on Sudoku and the `wd1++` math results are promising. That said, the paper’s ICLR-strength weakness is that the method’s claimed efficiency/simplicity is partly offset by lingering dependence on approximate likelihoods, the theory is elegant but somewhat harder to verify than the prose suggests, and the empirical evidence is not yet fully convincing on the harder reasoning benchmarks where the core `wd1` gains are modest. I think the contribution stands, but it would benefit from clearer separation of exact versus approximate claims, stronger multi-seed/statistical evaluation, and a more rigorous end-to-end accounting of compute and fairness across baselines.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes **wd1**, a ratio-free reinforcement learning objective for diffusion-based language models (dLLMs), motivated by the difficulty of estimating policy ratios when dLLM likelihoods are intractable. The core idea is to reformulate reverse-KL-regularized policy optimization as a **weighted log-likelihood** loss using advantage-based weights, and to extend it with a stepwise variant (**wd1++**) that exploits intermediate denoising states. Empirically, the authors report large gains over their reproduced baseline d1 on several reasoning benchmarks, plus reduced training cost, and they provide a theoretical interpretation connecting the method to energy-guided diffusion and negative-sample unlearning.

### Strengths
1. **Clear and relevant problem formulation for dLLMs RL.**  
   The paper identifies an important bottleneck in RL fine-tuning of diffusion LMs: policy-ratio computation is expensive and noisy because likelihoods are approximated. This is well aligned with ICLR interests in scalable learning methods for generative models.

2. **A plausible ratio-free objective with a useful conceptual angle.**  
   The main proposal, wd1, is conceptually appealing because it avoids explicit policy ratios and instead trains with weighted log-likelihood. The paper also provides an interpretation as reverse-KL policy optimization, and links the objective to energy-guided diffusion and unlearning, which strengthens the methodological narrative.

3. **Empirical gains on multiple reasoning benchmarks.**  
   The reported results on Sudoku, Countdown, GSM8K, and MATH500 suggest substantial improvements over d1 in some settings, especially on Sudoku and Countdown. The wd1++ extension is also reported to improve math performance further, which indicates that the authors explored a meaningful variant beyond the core method.

4. **Efficiency-oriented evaluation.**  
   Unlike many papers that only report final accuracy, this work also compares training time, FLOPs, and NFEs, which is valuable for ICLR because practical efficiency is often a major criterion for acceptance.

5. **Ablation studies address some design choices.**  
   The paper includes ablations on SFT, negative-sample weighting, the combined weight balance, and the exponential scale parameter ψ. These experiments help justify why the proposed weighted objective is not just a generic reweighting scheme.

### Weaknesses
1. **Novelty is moderate relative to closely related weighted regression / RL formulations.**  
   The central idea—turning policy optimization into a weighted likelihood or weighted regression objective—is closely related to AWR, RAFT-style methods, preference optimization, and several recent “ratio-free” RL objectives. The paper’s novelty is mainly in adapting this template to dLLMs and adding a negative-sample term, but the conceptual leap appears incremental rather than fundamentally new.

2. **The empirical comparison may not be fully convincing against the strongest relevant baselines.**  
   The main comparison is against d1, which is appropriate, but many stronger or concurrent baselines are only partially compared, and some results are reported under different regimes (e.g., SFT vs no SFT, full fine-tuning vs LoRA, different training budgets). For ICLR standards, the evidence for a clear state-of-the-art claim would benefit from more tightly controlled comparisons and statistical reporting.

3. **Potential confounding from training setup differences.**  
   wd1 is highlighted as not requiring SFT, but d1 is compared both with and without SFT in different tables, and some later comparisons use different parameterization settings (e.g., “full” fine-tuning for some methods). This makes it harder to isolate the algorithmic contribution from differences in training pipeline, data, or model update regime.

4. **The theoretical results, while interesting, may be more interpretive than foundational.**  
   The paper’s theory connects wd1 to an energy-guided diffusion perspective and unlearning, but the result appears to rely on rewriting the objective rather than proving a new optimization principle with broad consequences. The monotonic-improvement discussion is also based on standard RL machinery, so the theoretical novelty may be overstated.

5. **Reproducibility is only partial.**  
   The authors provide hyperparameters, datasets, and code, which is good, but the paper still leaves some practical details under-specified: exact reward implementation nuances, how approximation bias is handled in practice, and how sensitive results are to seed, checkpoint selection, and evaluation protocol. Given the large performance swings on some tasks, stronger reproducibility evidence would be expected at ICLR.

6. **Some claims are ambitious relative to the evidence.**  
   Statements such as “state-of-the-art math performance” and broad claims about eliminating the need for SFT may be too strong given the limited benchmark set, possible dependence on evaluation choices, and incomplete coverage of alternative strong methods.

### Novelty & Significance
**Novelty:** Moderate. The paper introduces a sensible adaptation of weighted likelihood optimization to diffusion LMs and adds a negative-sample term plus a stepwise extension. However, the main idea is closely related to existing weighted regression, AWR-like, and ratio-free policy optimization approaches, so the novelty is more in the dLLM-specific instantiation than in a fundamentally new learning principle.

**Clarity:** Moderate. The high-level story is understandable, but the presentation is mathematically dense and sometimes hard to parse. The core algorithmic intuition is present, yet the derivations and notation are heavy enough that readers may struggle to separate the main idea from the proof machinery.

**Reproducibility:** Moderate. The paper provides code, datasets, and hyperparameters, and this is a plus for ICLR. Still, the results would be more reproducible if the authors reported variance across seeds, clearer evaluation details, and more explicit ablation of pipeline differences.

**Significance:** Moderately strong if the results hold up. Improving RL fine-tuning for diffusion LMs is timely and relevant, and an efficient ratio-free approach could matter in practice. That said, the impact depends on whether wd1 consistently outperforms strong baselines under matched conditions, which is not fully established here.

### Suggestions for Improvement
1. **Strengthen the baseline suite and control conditions.**  
   Evaluate wd1 under matched settings against the strongest available dLLM RL methods, ideally with identical data, compute, decoding, and fine-tuning regimes. If some baselines use SFT or full fine-tuning, provide fair ablations that isolate those factors.

2. **Report variance and statistical significance.**  
   Add multiple random seeds, confidence intervals, or standard deviations for the main benchmarks. This is especially important because some gains appear very large, while others are marginal.

3. **Clarify the exact source of improvement.**  
   Separate the effects of: ratio-free optimization, negative-sample unlearning, removal of the reference policy, and stepwise training in wd1++. A cleaner factorized ablation would make the contribution much more convincing.

4. **Tighten the theoretical claims.**  
   Reframe the theory as an interpretation or reparameterization if that is what it is, rather than implying a deep new guarantee. Highlight precisely which parts are novel and which are standard RL/energy-based results.

5. **Improve exposition of the algorithm.**  
   A concise algorithmic summary with all variables explicitly defined in one place would help. In particular, the relationship between wd1, wd1++, the group-relative advantage, and the geometric-mixture sampling should be made easier to follow.

6. **Add a deeper analysis of failure modes.**  
   Since the method can struggle when all rewards in a group are identical, it would be useful to explore how often this occurs in practice, how severe the issue is, and whether adaptive group sizing or reward shaping alleviates it.

7. **Provide stronger reproducibility artifacts.**  
   Release exact evaluation scripts, seed settings, checkpoint selection rules, and full training logs. Also report compute in a standardized way so readers can better assess the efficiency gains.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison to stronger ratio-free or ratio-avoiding RL methods for dLLMs, especially **d2** and **SPG**, on the same tasks and model. Without these baselines, the claim that wd1 is a more principled or superior alternative to existing dLLM RL is not convincing at ICLR’s bar.

2. Report **matched-budget** comparisons against d1 and wd1++ with equalized training compute, rollout count, and number of likelihood evaluations. The current gains could be driven by more steps, different checkpoints, or different total sampling, so the improvement claim is not yet robust.

3. Add an ablation that isolates the contribution of the **negative-sample term** versus the ratio-free objective itself across multiple datasets, not just Sudoku. The paper’s central claim is that explicit penalization of low-advantage samples matters, but the evidence is too narrow to support generality.

4. Evaluate on at least one additional reasoning benchmark family beyond Sudoku/Countdown/GSM8K/MATH, ideally one that stresses different failure modes such as longer-horizon arithmetic or compositional reasoning. ICLR reviewers will expect evidence that the method is not overfit to the specific reward structure and small benchmark set used here.

5. Compare against a **non-diffusion AR baseline using the same reward setup** or an RL method with weighted regression on the same underlying model family. Without a cross-family control, it is unclear whether the gains come from the algorithm or from properties specific to LLaDA and its decoding/reward setup.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify how much wd1 reduces the claimed **likelihood-approximation error** in practice, not just in theory. The paper argues ratio-free optimization avoids amplified error, but it never measures approximation error, ratio variance, or objective mismatch empirically.

2. Analyze the **distribution of weights** and how often negative-sample penalties dominate or vanish during training. The core mechanism depends on exponential weighting; without diagnostics on weight entropy, saturation, and sign balance, it is hard to know whether wd1 is stable or just fortunate on these tasks.

3. Provide a systematic study of **sensitivity to ψ, λ, β, group size G, and number of diffusion steps**. The method introduces several coupled knobs, and the current ablations are too incomplete to establish that the improvement is not fragile to hyperparameter choices.

4. Test whether the “no SFT needed” claim holds under **multiple random seeds** and across more than the shown subset of tasks. The paper presents a strong efficiency narrative, but the evidence for removing SFT is thin and could be seed- or task-dependent.

5. Analyze when wd1 fails: cases where all completions get identical rewards, low-reward collapse, or over-penalization of near-correct samples. The limitation section mentions this, but the paper needs empirical failure analysis because this directly affects whether the method is usable in realistic RL settings.

### Visualizations & Case Studies
1. Add training curves for **reward, accuracy, completion length, and weight statistics** side by side for wd1, wd1++, and d1. This would show whether gains come from genuinely better optimization or just shorter outputs / faster early progress.

2. Show representative **case studies of sampled completions before and after updates**, including positive and negative samples with their weights and rewards. This would reveal whether the algorithm meaningfully suppresses bad reasoning traces or simply downweights them without improving behavior.

3. Visualize the **effective sample usage** across training: fraction of completions receiving substantial positive/negative weight, and how this changes over time. Without this, the claim that wd1 “fully utilizes completions” is not actually demonstrated.

4. Provide a failure-case gallery on tasks where the method underperforms or destabilizes, especially math prompts with partial correctness. ICLR reviewers will want to see whether the method improves reasoning broadly or only on easy-to-score examples.

### Obvious Next Steps
1. Extend the method to a **single unified benchmark suite with compute-matched comparisons** against all recent dLLM RL baselines. That is the minimum needed for a convincing ICLR-level claim of state-of-the-art performance.

2. Remove the remaining dependence on the **d1 likelihood approximation** by either validating with ELBO-based estimates or proposing a lower-bias estimator. The paper’s main promised benefit is ratio-free optimization, but the implementation still inherits a biased approximation that weakens the core claim.

3. Study **generalization across models and domains**, including at least one non-LLaDA dLLM and one non-math task. The current evidence is too model-specific and task-specific to support the broader algorithmic contribution implied by the title and abstract.

4. Formalize the relationship between wd1 and existing weighted-regression / unlearning methods with a sharper empirical distinction. Right now the theoretical framing is broad, but the paper needs experiments that show wd1 is more than a repackaging of known weighted likelihood ideas.

# Final Consolidated Review
## Summary
This paper proposes **wd1**, a ratio-free RL objective for fine-tuning diffusion language models, by rewriting reverse-KL policy optimization as a weighted log-likelihood update over sampled completions. It further introduces **wd1++**, a stepwise variant that reuses intermediate denoising states, and claims both improved reasoning performance and lower training cost on LLaDA-8B across several benchmarks.

## Strengths
- The paper tackles a real bottleneck in RL for diffusion LMs: policy-ratio estimation is indeed awkward because likelihoods are intractable and approximated. The proposed ratio-free formulation is therefore timely and practically relevant.
- The empirical gains are strongest on **verifiable, low-dimensional reasoning tasks** such as Sudoku and Countdown, and the paper also reports meaningful training-efficiency improvements. The negative-sample ablation is especially useful evidence that the extra penalty term is not decorative.

## Weaknesses
- The core method is **less novel than presented**. At a high level, wd1 is a dLLM adaptation of already familiar weighted regression / advantage-weighted likelihood ideas, with an added negative-sample penalty. The paper’s framing makes this sound more fundamental than it really is.
- The method is **not actually approximation-free** in implementation. Although it removes explicit policy ratios, it still relies on **d1-style likelihood approximations** for training. This weakens the main “ratio-free avoids approximation error” narrative, since the critical likelihood estimator is still a biased component.
- The strongest performance gains are **task- and setting-dependent**. On the harder math benchmarks, wd1 alone is only modestly better than d1, and the more convincing gains come from **wd1++** plus a changed training setup. That makes the paper’s broad superiority claims too strong.
- The evaluation is **not tight enough to justify the strongest claims**. There are no multi-seed statistics, confidence intervals, or fully matched compute comparisons across all baselines. In particular, some comparisons mix LoRA vs full fine-tuning, SFT vs no SFT, and different dataset/training regimes, which makes it hard to isolate the algorithmic contribution.
- The paper’s main story about “fully utilizing completions” is only partially supported. The exponential weighting can easily saturate, and the method itself admits a failure mode when all rewards in a batch are identical. This is not a corner case; it is a real limitation for sparse or noisy verifier rewards.

## Nice-to-Haves
- A cleaner factorized ablation separating: ratio-free optimization, negative-sample unlearning, removal of the reference model, and stepwise intermediate-state training.
- More explicit reporting of weight statistics over training, to show whether the negative/positive terms are balanced in practice or just work on the chosen benchmarks.
- A clearer end-to-end compute accounting that includes the sampling side, not just likelihood-evaluation NFEs.

## Novel Insights
The most interesting substantive insight is that the paper’s “ratio-free” gain is not primarily about eliminating all approximation, but about **changing where approximation error enters**: it avoids exponentiating noisy likelihood ratios, which is plausibly the main source of instability in diffusion-RL. The stepwise extension, wd1++, is also more than a minor tweak; it tries to exploit the otherwise discarded intermediate denoising states, and that appears to matter more on harder math tasks than the base wd1 objective. Still, the results suggest wd1 is best viewed as a **practical weighted-regression wrapper around existing dLLM RL machinery**, not a fundamentally new optimization principle.

## Potentially Missed Related Work
- **d2** — directly relevant ratio-free / ratio-avoiding RL for diffusion language models; essential baseline for the paper’s main claim.
- **SPG** — another recent policy-gradient formulation tailored to masked diffusion LMs; also directly relevant to the ratio-free positioning.
- **SEPO** — concrete-score-based policy optimization for discrete diffusion; relevant because it offers a different route to avoiding likelihood-ratio computation.
- **AWR / RAFT / weighted regression RL papers** — relevant because wd1 is structurally close to advantage-weighted or reward-weighted likelihood objectives.
- **Negative-sample / unlearning style language-model fine-tuning papers** — relevant to the interpretation of the negative term as unlearning.

## Suggestions
- Add a **compute-matched, multi-seed comparison** against d1, d2, SPG, and the other recent dLLM RL methods under the same model, same decoding setup, and same budget.
- Report the **variance of the main benchmarks** across seeds and checkpoints, especially for GSM8K/MATH where the gains are small.
- Include an experiment that measures **ratio variance / likelihood-approximation mismatch** directly, to justify the motivation for ratio-free optimization empirically.
- Provide a stronger ablation showing whether wd1 still helps when the **reference policy is kept** and when the **negative-sample term is removed** across multiple tasks, not only Sudoku.
- Clarify in the main text how much of the reported improvement comes from **wd1 itself versus wd1++ and training-budget differences**.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
