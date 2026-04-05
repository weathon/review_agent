=== CALIBRATION EXAMPLE 86 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- Does the title accurately reflect the contribution?
  - Yes, broadly. The paper is about scaling RL compute for LLMs, and the title reflects that. “The Art of Scaling” is a bit rhetorical, but not misleading.
- Does the abstract clearly state the problem, method, and key results?
  - Mostly yes. It identifies the gap: lack of predictive scaling methodology for RL comparable to pre-training. It states the main method: fitting sigmoidal compute-performance curves and studying design choices, then introducing SCALERL.
  - The abstract also communicates the main empirical result: a 100,000 GPU-hour run whose later performance is predicted from earlier data.
- Are any claims in the abstract unsupported by the paper?
  - The strongest claims are “first large-scale systematic study” and “scientific framework for analyzing scaling in RL.” The paper does present a substantial empirical study, but “first” is difficult to substantiate and not rigorously defended against the broader RL literature.
  - The claim that SCALERL is a “best-practice recipe” and “state-of-the-art” is plausible within the paper’s experimental setup, but the comparison set is limited to a subset of recent recipes and mostly one domain. ICLR reviewers would likely want more caution around generality.

### Introduction & Motivation
- Is the problem well-motivated? Is the gap in prior work clearly identified?
  - Yes. The introduction makes a strong case that RL for LLMs has scaled rapidly while methodology for predicting scaling behavior lags behind.
  - The distinction between isolated algorithmic reports and a principled scaling methodology is clearly framed.
- Are the contributions clearly stated and accurate?
  - The contributions are stated clearly: a compute-performance framework, a set of ablations, and SCALERL.
  - However, the framing sometimes conflates two different contributions: understanding scaling behavior and proposing a new recipe. The paper would benefit from separating these more explicitly, because the scientific contribution depends on the validity of the scaling framework, whereas the practical contribution depends on benchmark outcomes.
- Does the introduction over-claim or under-sell?
  - It somewhat over-claims. In particular, calling the framework “principled” and “predictive” is reasonable only if its predictive success is shown robustly across settings, not just on selected stable runs.
  - The statement that this provides predictability “long achieved in pre-training” is aspirational; the evidence here is narrower and less mature than scaling-law work in pre-training.

### Method / Approach
- Is the method clearly described and reproducible?
  - The paper gives a reasonable high-level description of the RL setup, the sigmoid scaling law, and SCALERL’s components.
  - But reproducibility is limited by several places where implementation details matter enormously:
    - The exact objective in Equations (2)–(4) is hard to parse, and the parser garbles some notation; still, the conceptual definitions are there.
    - The method relies on many interacting choices: PipelineRL-8, CISPO, FP32 logits, interruptions, prompt-level aggregation, batch-level normalization, zero-variance filtering, and no-positive-resampling. The paper studies them, but the combined system is complex enough that reproducing the exact gains would require very careful implementation fidelity.
  - The scaling-fit procedure is described with a grid search over \(A\) and \(C_{mid}\), then fitting \(B\), which is helpful. But the selection procedure and evaluation protocol raise concerns about overfitting the curve to the same runs used for method selection.
- Are key assumptions stated and justified?
  - Some are, but several important assumptions are under-justified:
    - The core assumption is that validation pass rate follows a sigmoidal function of compute after an initial burn-in. The paper motivates this empirically, but not theoretically.
    - The paper assumes that fitting only the later portion of training is legitimate, yet this excludes regimes where some methods are unstable or non-monotonic—the exact regimes that may matter most when comparing algorithms.
    - The use of in-distribution held-out validation prompts as the main scaling target is consistent with scaling-law practice, but the paper then draws conclusions about downstream generalization that are not supported by the same fitted target.
- Are there logical gaps in the derivation or reasoning?
  - Yes, several:
    - The paper argues that the asymptote \(A\) captures “ceiling” and \(B\) captures efficiency, but in practice these parameters can be coupled under noisy fits. The paper partially acknowledges this in the appendix, yet the interpretation is treated more strongly than the estimation procedure warrants.
    - The rearrangement in Figure 5 to compare efficiency by fixing averaged \(A\) and re-fitting slopes is not obviously statistically sound; it can obscure uncertainty in \(A\) and propagate model-misspecification into \(B\).
    - The claim that many interventions “primarily modulate compute efficiency without materially shifting asymptote” is plausible, but the evidence appears based on relatively few runs and a fitting framework that may not have enough power to distinguish small \(A\) shifts from estimation noise.
- Are there edge cases or failure modes not discussed?
  - Yes. Important ones include:
    - How the sigmoid fit behaves when training is unstable, discontinuous, or has multiple phases.
    - Whether SCALERL remains predictive on tasks where reward is much sparser or noisier than the math/code tasks studied.
    - Whether the choice of 1,000 held-out prompts is sufficient when training spans multiple epochs over a finite dataset.
    - What happens when compute is scaled by changing other aspects of the system simultaneously, e.g. reward model changes or different prompt distributions.
- For theoretical claims: are proofs correct and complete?
  - There are no major theoretical proofs, so this mainly concerns the interpretation of the sigmoid-power-law relationship. The derivation in Appendix A.4 that the sigmoid approximates a power law at large compute is conceptually fine, but it is not a substitute for a principled justification of why the sigmoid is the right predictive model.

### Experiments & Results
- Do the experiments actually test the paper's claims?
  - Largely yes. The experiments directly probe scaling, ablations, extrapolation, and larger-compute confirmation.
  - The key claim is predictive scaling from small-to-large compute, and the paper does test this with longer runs and extrapolated curves.
- Are baselines appropriate and fairly compared?
  - Mostly, but with caveats:
    - The paper compares against several recent recipes (GRPO/DeepSeek, DAPO/Qwen2.5, Magistral, MiniMax). This is a relevant set.
    - However, many baselines are not necessarily standardized across exactly the same infrastructure, batch settings, or hyperparameter tuning budget. The paper acknowledges some adjustments, but it is still hard to know whether comparisons are fully fair.
    - In Appendix A.17, some baselines are modified because of codebase constraints. That may be reasonable, but it makes the “state-of-the-art” claim less clean.
- Are there missing ablations that would materially change conclusions?
  - Yes, several likely matter:
    - A more systematic analysis of interactions between components, not just leave-one-out. LOO helps, but interactions among loss type, normalization, and batch dynamics could be substantial.
    - An ablation on the choice of fitting window and validation set size more rigorously than the current heuristic discussion.
    - More evidence that the sigmoid fit is superior across multiple random seeds and not just representative runs.
    - A direct comparison with alternative predictive models beyond sigmoid vs power law.
- Are error bars / statistical significance reported?
  - Partially. The paper reports three independent SCALERL runs and suggests an approximate \(\pm 0.02\) margin for \(A\).
  - But the main comparisons in the figures do not seem to include error bars or confidence intervals, and there is no formal statistical significance analysis.
  - Given how close many curves appear, uncertainty quantification is important for ICLR standards.
- Do the results support the claims made, or are they cherry-picked?
  - The results do support the broad conclusion that some recipes scale more predictably than others.
  - Still, there is some risk of cherry-picking in the sense that the most dramatic predictive examples are emphasized, while unstable or poorly fitting runs are mostly relegated to the appendix.
  - The claim that SCALERL “surpasses all existing recipes” should be tempered: it is supported within the paper’s experimental suite, but not universally.
- Are datasets and evaluation metrics appropriate?
  - For the core scaling target, yes: held-out in-distribution validation pass rate is appropriate for studying compute scaling.
  - But the paper also claims downstream transfer improvements on AIME and LiveCodeBench; these are useful, but they are secondary. The paper is strongest on in-distribution pass rate, weaker on downstream generalization.
  - Because the training data is finite and reused over multiple epochs, pass rate on held-out prompts may still be somewhat entangled with memorization and dataset-specific dynamics.

### Writing & Clarity
- Are there sections that are confusing or poorly explained?
  - The methodological section is generally understandable, but the paper is dense and the distinction between “scaling for prediction” and “scaling for method selection” is not always clean.
  - The interpretation of the sigmoid parameters \(A, B, C_{mid}\) is helpful, but the paper sometimes treats them as more directly identifiable than they are.
  - The comparison between compute efficiency and asymptotic performance could be explained more carefully, especially when LOO re-fits fix \(A\).
- Are figures and tables clear and informative?
  - The figures, as described, appear central and informative:
    - Figure 1 demonstrates the core predictive claim.
    - Figure 2 supports cross-method comparison.
    - Figure 5 summarizes LOO results.
    - Table 1 provides parameter values for large-scale runs.
  - The main issue is not clarity of purpose, but that the argument depends heavily on fitted curves. A reader needs uncertainty bands and more explicit statements of fit quality to fully trust the visual conclusions.

### Limitations & Broader Impact
- Do the authors acknowledge the key limitations?
  - Some are acknowledged:
    - Generalization beyond in-distribution validation is not fully characterized.
    - The paper notes that different choices may help downstream performance more than validation curves suggest.
    - It acknowledges that some recipes are unstable beyond certain compute ranges.
- Are there fundamental limitations they missed?
  - Yes, several important ones:
    - The framework is validated mainly on a math-centric RL regime with verifiable rewards. It is not yet shown for broader RLHF settings, agentic tasks, or multi-turn environments.
    - The claimed predictability depends on stable training runs; the framework is less informative precisely when training is unstable, which is often the interesting regime.
    - The choice of a single saturating curve may miss multi-regime dynamics.
    - The paper’s practical recipe may be tightly coupled to the specific infrastructure, kernels, and model family used.
- Are there failure modes or negative societal impacts not discussed?
  - There is little broader-impact discussion.
  - A reasonable concern is that scaling RL for reasoning models may exacerbate compute concentration among large industrial labs, which the introduction implicitly acknowledges but the paper does not discuss as a broader impact.
  - Another issue is that stronger reasoning models can also be used in harmful or deceptive contexts, though that is broader than this paper’s scope.

### Overall Assessment
This is an ambitious, compute-heavy paper with a genuine and timely question: can we make RL scaling for LLMs as predictable as pre-training scaling? The answer presented here is partially convincing. The empirical story is strong that, within the authors’ math/verifiable-reward setting, a sigmoidal compute-performance model can be useful for extrapolation, and that SCALERL is a carefully engineered recipe with good scaling behavior. However, the paper’s strongest claims slightly outrun the evidence: uncertainty is not quantified enough, the fitting-based comparisons can blur parameter identifiability, fairness of all baselines is not fully established, and the generality beyond the studied regime remains limited. For ICLR, this is promising and potentially impactful, but I would want stronger uncertainty analysis, more rigorous baseline normalization, and clearer evidence that the predictive framework is robust beyond a narrow set of stable runs before fully endorsing the broad scaling-law claims.

# Neutral Reviewer
## Balanced Review

### Summary
This paper studies how to scale reinforcement learning compute for LLM reasoning in a predictive way, proposing a sigmoid-based compute-performance framework and an RL recipe called SCALERL. The authors claim that, across a large empirical study, many common RL design choices mainly affect compute efficiency rather than the asymptotic performance ceiling, and that early training dynamics can extrapolate reliably to much larger compute budgets.

### Strengths
1. **Addresses an important and timely problem for ICLR.**  
   The paper targets a core open issue in LLM post-training: how to predictably scale RL compute, which is highly relevant to current reasoning-model practice and to ICLR’s interest in empirical methodology and scalable learning.

2. **Large empirical effort with multiple scales and axes.**  
   The paper reports a substantial experimental campaign, including a 100k GPU-hour run, 16k GPU-hour leave-one-out studies, and broader scaling across model size, batch size, generation length, and task mixture. This is stronger than many RL-for-LLM papers that only present a single recipe or one-off benchmark gains.

3. **Clear attempt to separate asymptotic performance from efficiency.**  
   The central framing around the sigmoid parameters \(A\), \(B\), and \(C_{mid}\) is conceptually useful: it gives a concrete way to discuss whether a change improves the ceiling or just makes progress faster. This is a meaningful methodological contribution if the fitting is reliable.

4. **Practical recipe assembled from multiple design choices.**  
   SCALERL combines several engineering and algorithmic decisions—PipelineRL, CISPO, prompt-level aggregation, batch-level normalization, FP32 logits, zero-variance filtering, and no-positive-resampling—and the paper evaluates these choices systematically rather than only presenting the final recipe.

5. **Some evidence of predictive extrapolation.**  
   The paper provides repeated examples where early fitted curves match later training points, including the large 100k GPU-hour run and several ablations. If valid, this is a valuable result for reducing the need for full-scale sweeps.

### Weaknesses
1. **The novelty is somewhat incremental relative to the current RL-for-LLM literature.**  
   Much of the paper combines or re-ranks existing components from recent systems and technical reports (e.g., GRPO/DAPO/CISPO/PipelineRL/FP32 fixes/curricula) rather than introducing a fundamentally new RL algorithm or theoretical insight. For ICLR, this may read more as a strong engineering synthesis than a clearly novel scientific advance.

2. **The empirical claims depend heavily on one task family and one validation protocol.**  
   Most evidence comes from verifiable math reasoning with a held-out in-distribution validation set. That makes the compute-scaling story less general than the paper’s framing suggests, especially since ICLR typically expects broader evidence or a stronger argument for why the findings transfer across RL settings.

3. **The claim that design choices mostly affect efficiency, not ceiling, is not always fully convincing.**  
   The paper itself reports some interventions that materially change asymptotic reward, such as FP32 at the LM head and different loss types/clipping settings. The distinction between “ceiling” and “efficiency” sometimes appears retrospective and fit-dependent, which weakens the interpretability of the conclusion.

4. **Curve-fitting methodology may be too dependent on heuristic choices.**  
   The sigmoid is fitted after excluding early compute, with grid search over \(A\) and \(C_{mid}\), and the paper notes that a specific cutoff is chosen heuristically. This raises concerns about selection bias, especially because the fitted asymptotes are a major part of the paper’s claims. ICLR reviewers are likely to ask how sensitive the conclusions are to the fitting procedure.

5. **Reproducibility is limited by missing implementation detail and restricted comparability.**  
   Although the paper describes many hyperparameters, the complete recipe still relies on a complex stack of systems choices and recent proprietary/open technical-report components. The paper also mentions releasing code only after acceptance, which limits immediate reproducibility. Given the scale and complexity, independent replication would be difficult.

6. **Some comparisons may not be strictly apples-to-apples.**  
   The paper compares against several recent methods, but those methods may differ in batch size, stability settings, dynamic sampling behavior, or system throughput. The paper acknowledges some such issues, but the extent to which gains come from algorithmic superiority versus implementation/system advantages is not always cleanly isolated.

### Novelty & Significance
**Novelty:** Moderate. The main contribution is a systematic empirical framework for compute scaling in RL for LLMs, plus a curated recipe that performs well. The framing is useful, but the paper is more a synthesis and scaling study than a clearly new learning algorithm.

**Significance:** Potentially high if the predictive scaling result is robust. A reliable way to extrapolate RL performance from smaller runs would be very valuable for the field and aligns well with ICLR’s interest in principled empirical methodology. However, the significance is tempered by the narrow domain focus, the heavy reliance on heuristic curve fitting, and the fact that much of SCALERL is an aggregation of known components.

**Clarity:** Generally good at the high level, with a strong narrative and clear experimental story. That said, the paper is dense, highly specialized, and sometimes mixes claims about algorithmic improvements, system throughput, and scaling-law interpretation in ways that make causal attribution harder.

**Reproducibility:** Moderate to weak. There is substantial reporting of settings and ablations, but the full setup is complex, code is deferred to after acceptance, and the fitted-curve methodology plus compute-heavy experiments make exact replication challenging.

### Suggestions for Improvement
1. **Add stronger sensitivity analyses for the sigmoid fitting framework.**  
   Report how conclusions change with different early-cutoff thresholds, alternative functional forms, different validation-set sizes, and different weighting schemes. This is essential because the fitted asymptote is central to the paper.

2. **Provide direct evidence that the same scaling behavior holds beyond one domain.**  
   The paper already has some math+code and downstream plots; it would be stronger to test a second qualitatively different RL setting, or at least more clearly delineate the scope of claims.

3. **Separate algorithmic gains from systems gains more rigorously.**  
   Since PipelineRL appears to improve throughput as much as sample efficiency, the paper should isolate wall-clock, token, and update-based contributions more cleanly. This would help reviewers judge whether SCALERL is an algorithmic advance or primarily a systems optimization.

4. **Strengthen the baseline fairness discussion.**  
   For each compared method, explicitly report whether batch size, truncation handling, effective batch, and precision settings were matched as closely as possible. ICLR reviewers will care a lot about whether the baselines were strong and fairly tuned.

5. **Make the core contribution more explicit and concise.**  
   The paper would benefit from a sharper statement of what is scientifically new: is it the scaling-law fit, the decomposition into ceiling vs efficiency, the specific SCALERL recipe, or the empirical finding that early runs predict large-scale RL? Right now these are blended together.

6. **Release code and run logs, not just a minimal repository.**  
   Given the scale and complexity, reproducibility would improve substantially with full training configs, exact evaluation scripts, and raw learning curves. This would also help validate the fitted scaling parameters independently.

7. **Discuss failure modes more openly.**  
   The paper mentions instability, truncation, and sensitivity in some settings, but a clearer analysis of when predictive scaling fails would make the contribution more credible and more useful to the community.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add strong, apples-to-apples baselines at the same compute budget and same model/dataset: GRPO, DAPO, PipelineRL, CISPO, GSPO, and recent RL recipes should all be run to the same GPU-hour targets with identical evaluation cadence. Right now the “SCALERL is best” claim is hard to trust because the comparisons mix different batch sizes, stability regimes, and sometimes different effective compute.

2. Add compute-matched ablations that isolate each SCALERL component in the final recipe at the full 100k-GPU-hour scale, not just 16k. The paper claims several changes mainly affect efficiency rather than asymptote, but that conclusion is only partly supported at lower scale where interactions are likely different.

3. Add a truly independent test set for scaling prediction, not just held-out prompts from the same training distribution. The core claim is that early validation curves predict final performance; without showing this on a genuinely unseen distribution, the predictive-scaling claim is too narrow for ICLR.

4. Add repeated runs for the main comparisons, especially the large-scale 100k and MoE experiments, with variance bars or confidence intervals on the extrapolated asymptote. The paper reports only three SCALERL runs for fit variance, but the key claims depend on whether the ranking of recipes is stable under seed variation.

5. Add scaling comparisons against a simpler control such as “same recipe but no clever fit/extrapolation” or “linear/logistic alternatives with held-out validation” across several tasks. Without this, it is unclear whether the sigmoid is genuinely predictive or just a convenient post hoc curve that happens to fit the chosen runs.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify how sensitive the fitted asymptote and efficiency parameters are to the choice of fitting window, threshold, and grid search range for A and C_mid. The paper’s main scientific claim depends on extrapolation, so the method needs a much tighter robustness analysis than one or two examples.

2. Analyze whether the fitted scaling law is causal or merely descriptive by checking if early-curve fits made at different fractions of the budget rank methods consistently. ICLR reviewers will want to know whether the framework can actually be used to choose methods before the end of training, rather than only explaining finished runs.

3. Separate system-level throughput gains from algorithmic sample-efficiency gains more rigorously. The paper repeatedly claims that some methods improve compute efficiency without changing the asymptote, but the current evidence confounds algorithmic effects with PipelineRL scheduling and hardware utilization.

4. Analyze interaction effects between the “best-practice” components. The paper presents mostly one-factor ablations and leave-one-out studies, but the claims about cumulative benefits require showing whether some components only help in the presence of others or whether they are redundant.

5. Provide a clearer failure analysis for unstable methods, including when and why divergence happens. The paper’s scaling story depends on stable curves; without diagnosing instability modes, it is impossible to know whether SCALERL is broadly scalable or just tuned to avoid a narrow failure regime.

### Visualizations & Case Studies
1. Show per-run training trajectories with raw points, fitted curves, and extrapolated points for several seeds and several recipes on the same axes. This would expose whether the sigmoidal fits are genuinely predictive or whether the apparent agreement is driven by cherry-picked stable runs.

2. Add a “method ranking over compute” plot showing which recipe looks best at 1k, 5k, 10k, 50k, and 100k GPU-hours. The paper explicitly argues that small-scale rankings can invert at scale, so this visualization is essential to validate that claim.

3. Show failure-case traces for unstable methods: truncation rate, entropy, reward, KL-like proxies, and effective batch size over time. This would reveal whether instability is caused by one dominant mechanism or multiple interacting ones.

4. Add qualitative case studies of prompts where SCALERL and a strong baseline diverge, with generated reasoning traces and final answers. Without examples, it is unclear whether the method improves genuine reasoning or mostly exploits reward/dataset artifacts.

5. Show calibration plots for early-predicted versus final downstream performance on AIME/MATH/LiveCodeBench across runs. That would make the “predictive scaling” claim concrete and show whether validation-pass-rate forecasting transfers beyond the in-distribution validation set.

### Obvious Next Steps
1. Extend the framework to a broader set of RL tasks and reward types, including sparse/structured rewards and multi-turn agentic settings. The current claims are almost entirely on math/code verifiable rewards, which is too narrow for the paper’s broad scaling thesis.

2. Test whether the same scaling law holds when pretraining size, model architecture, and RL data mixture are varied jointly. The paper hints at this as future work, but it is the most obvious way to determine whether the proposed framework is actually general.

3. Release a minimal reproducibility package with the exact fitting procedure, seeds, and compute accounting before acceptance. The paper’s central contribution is methodological; without reproducible fits and standardized compute accounting, the claims are not actionable for the community.

# Final Consolidated Review
## Summary
This paper studies how RL post-training for LLMs scales with compute and proposes a sigmoidal compute-performance framework to extrapolate performance from early training. It also introduces SCALERL, a bundled recipe combining several recent choices (PipelineRL, CISPO, FP32 logits, interruption-based length control, prompt-level aggregation, batch-level normalization, zero-variance filtering, and no-positive-resampling) and shows strong results on a math/verifiable-reward setting, including a 100k GPU-hour run.

## Strengths
- The paper tackles an important and timely problem: RL for reasoning models is now compute-intensive, but unlike pre-training there is little discipline around predicting how algorithmic changes scale. The large-scale empirical framing is genuinely relevant.
- The study is unusually extensive for this area, with a 100k GPU-hour run, 16k GPU-hour leave-one-out ablations, and additional scaling axes such as batch size, generation length, model size, and math+code mixtures. That breadth gives the paper more weight than a typical “one recipe, one benchmark” RL report.
- The paper’s central decomposition into asymptotic ceiling \(A\) versus compute efficiency \(B\) is useful and conceptually clean. Several ablations do appear to support the main story that some choices mostly change speed of progress, while others can change the ceiling.
- There is real evidence of predictive extrapolation in the stable regimes the paper studies: the early sigmoid fits often align well with later training points, including the large 100k GPU-hour run.

## Weaknesses
- The biggest weakness is that the paper’s strongest claims outpace the evidence. The work is presented as a general framework for “predictable RL scaling,” but almost all results come from a narrow math/verifiable-reward regime with in-distribution held-out validation. That is not enough to justify the broad language about a general science of RL scaling.
- The sigmoid-fitting methodology is central, but it is still heuristic-heavy. The paper excludes early training, grid-searches over \(A\) and \(C_{mid}\), and then interprets fitted parameters as if they were cleanly identifiable. This is fragile: small fit changes can alter the apparent asymptote, and the paper does not quantify uncertainty rigorously enough for such strong conclusions.
- Comparisons across methods are not perfectly apples-to-apples. The paper itself acknowledges that some baselines needed different batch sizes, effective batch handling, or implementation adjustments. Since SCALERL combines algorithmic and systems improvements, it remains unclear how much of the gain comes from a better learning recipe versus better throughput and training stability engineering.
- The paper’s “most design choices only affect efficiency, not ceiling” narrative is somewhat overstated. The authors themselves show that some choices, especially loss type and FP32 logits, can materially change asymptotic reward in some settings. The ceiling/efficiency separation is useful, but the paper treats it more cleanly than the data really supports.
- Reproducibility is still limited. The system is complex, the code is only promised after acceptance, and the exact combination of recent components and infrastructure details would be difficult for an outside group to replicate without substantial effort.

## Nice-to-Haves
- Stronger sensitivity analyses for the sigmoid fitting procedure: different cutoffs, different fitting windows, alternative curve families, and different held-out set sizes.
- More repeated runs and confidence intervals on the large-scale results and on the extrapolated asymptotes.
- A cleaner separation of algorithmic improvements from throughput/system-efficiency gains, especially for PipelineRL.
- A more explicit analysis of when the framework fails, not just when it works.

## Novel Insights
The most interesting insight is not simply that one recipe is better than another, but that some RL post-training choices appear to move methods along two different axes: how fast they improve and how high they ultimately saturate. The paper also suggests that, at least in a stable reasoning-RL regime, early validation trajectories may be sufficient to forecast much larger training runs, which is a meaningful methodological step if it holds up beyond this domain. That said, the work still reads more like a strong empirical synthesis of recent ideas than a broadly validated scaling law for RL in the same sense that pre-training scaling laws became foundational.

## Potentially Missed Related Work
- **Asynchronous RLHF: Faster and more efficient off-policy RL for language models** — directly relevant because PipelineRL builds on asynchronous off-policy RL ideas.
- **DeepSeekMath / DeepSeek-R1** — relevant as a foundational reference point for large-scale reasoning RL and GRPO-style training.
- **DAPO** — already cited and directly relevant; its dynamic sampling and prompt-level aggregation are central comparison points.
- **VAPO** — relevant because it explores another major reasoning-RL recipe and stability-oriented design choices.
- **ProRL** — relevant as a compute-heavy prolonged RL baseline with related scaling motivations.
- **Part I: Tricks or traps? A deep dive into RL for LLM reasoning** — relevant as a diagnostic perspective on what really matters in reasoning RL recipes.

## Suggestions
- Add a robustness section that systematically varies the fitting window and compares sigmoid against simpler alternatives under the same protocol.
- Report seed variance and uncertainty bands for the main scaling curves, especially the 100k GPU-hour results.
- Include a more controlled comparison table where each baseline is matched as closely as possible on batch size, truncation handling, precision, and effective compute.
- State the scope of the claims more narrowly: this is compelling evidence for stable, verifiable-reward reasoning RL, not yet a universal RL scaling law.


# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept
