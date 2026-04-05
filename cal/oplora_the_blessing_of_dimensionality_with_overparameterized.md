=== CALIBRATION EXAMPLE 36 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is mostly accurate: the paper does introduce an overparameterized reparameterization of LoRA with a discarded MLP, and the “blessing of dimensionality” framing matches the optimization story in Section 3.2.
- The abstract clearly states the problem, the proposed method, and the broad empirical claim.
- However, some claims are stronger than the evidence presented later in the paper. In particular, the abstract says OP-LoRA “requires less wall time than custom optimizers,” “has zero extra cost at inference,” and “improves and decreases sensitivity to learning rate.” The inference claim is correct by construction, but the wall-time and general superiority claims are not uniformly supported across all reported settings. For example, Table 4 reports OP-LoRA is slower than vanilla LoRA, and Table 3 shows OP-LoRA is not always best on commonsense tasks compared with LoRA-Pro/ScaledAdamW.

### Introduction & Motivation
- The paper motivates the problem well: LoRA can be hard to optimize, and custom optimizers can be effective but are cumbersome and expensive.
- The gap in prior work is reasonably identified: existing methods either change initialization, optimizer behavior, or adapter structure, but do not offer a simple train-time-only overparameterization that generalizes across LoRA variants.
- The contributions are clearly listed, but one contribution is somewhat overstated: “OP-LoRA navigates loss landscapes better than standard LoRA due to a built-in acceleration mechanism.” This is plausible, but the evidence in the paper is mostly suggestive rather than conclusive, especially since the theoretical analysis is simplified and the empirical support is limited to small-scale diagnostics.
- For ICLR’s bar, the introduction is strong in positioning and practical relevance, but it should be more careful in distinguishing demonstrated results from interpretation.

### Method / Approach
- The core method is simple and clear: use an MLP hypernetwork to generate LoRA matrices during training, then discard the MLP and keep the resulting adapter weights for inference.
- Reproducibility is mostly adequate at the conceptual level, but there are important ambiguities:
  - The exact MLP architecture is not fully specified in the main method section. Section 3.1 says “two-layer MLP,” but the dimensionality, activation, and output parameterization across A/B are only partially clarified.
  - It is not fully specified how the learned input vector \(z\) is initialized and optimized relative to the MLP weights in all experiments.
  - The paper states OP-LoRA can be extended to other adapters by “modifying the size of the prediction head,” but the details of how this interacts with adapter-specific constraints are only briefly discussed.
- Theoretical derivation issues matter here:
  - In Section 3.2, the update derivation is heavily simplified by treating the hidden layer output as a free parameter vector \(h\), which effectively changes the model. That is acceptable as an intuition-building approximation, but the paper presents it as central to the claimed acceleration mechanism.
  - The derivation of the “trainable learning rate” and “adaptive line search” is not rigorous enough to establish the general claim. The update contains both the direct gradient term and the projection term, but the paper does not prove that this necessarily improves convergence in the nonconvex, stochastic, Adam-optimized setting used in experiments.
  - The Hessian analysis in Section 3 and Appendix A is mathematically plausible in spirit, but it is framed for a local SPD approximation near minima. This is a very narrow regime, and the paper does not clearly state how much the conclusions extend to the actual training trajectory.
- Edge cases/failure modes are under-discussed:
  - Since the MLP is discarded, OP-LoRA can only help if the generated adapter weights are actually good at the end of training; the method provides no guarantee that the train-time optimization path will be better for all tasks or all scales.
  - The extra train-time parameters increase memory substantially, which may make the method unusable in constrained settings.
  - The paper does not discuss whether the MLP can overfit optimization noise or whether its benefit depends critically on width, initialization, or task scale.
- Overall, the method is novel and practically elegant, but the theoretical story is stronger as intuition than as proof.

### Experiments & Results
- The experiments broadly test the paper’s claims across image generation, VQA, commonsense reasoning, and a few appendix extensions. This is a good sign for breadth.
- That said, there are several important concerns for ICLR-level evidence:
  - The paper’s main claim is that OP-LoRA improves optimization and generalizes across tasks. The strongest evidence for optimization comes from the Rotated MNIST case study in Section 3.3 and Appendix B.2, which are small-scale toy or controlled settings. These are useful diagnostics, but they do not establish that the same mechanism explains the larger benchmark gains.
  - The reported gains are mixed. Table 1 shows large improvements on image generation, but Table 2 shows only around 1% gains on VQA, and Table 3 shows OP-LoRA is competitive but not uniformly best on commonsense reasoning.
  - Table 4 shows the method increases GPU memory substantially (44 GB to 69 GB) and slightly increases wall time. This is important because the paper emphasizes “easy to implement” and “less wall time than custom optimizers,” but the method is not lightweight in training resources.
- Baselines:
  - The baseline set is fairly strong and includes relevant gradient-alignment methods and non-gradient-alignment methods.
  - However, comparisons are not always fully apples-to-apples. In Table 3, the “LoRA r=466” baseline is intended to match parameter count, which is good, but it is not clear whether the training hyperparameters were tuned equivalently to the OP-LoRA settings.
  - For image generation, it would be helpful to know whether all baselines used the same prompt set, seed count, and exact evaluation protocol. The paper says it follows prior work, but the lack of concrete variance reporting weakens confidence in the magnitude of the CMMD gains.
- Missing ablations that would materially affect conclusions:
  - A direct ablation comparing OP-LoRA to plain LoRA with matched extra train-time parameter count but no hypernetwork structure would be important to isolate whether the benefit comes from overparameterization itself or specifically from the MLP reparameterization.
  - There is no strong ablation on MLP depth/activation/input choice beyond width. Since the theory suggests the reparameterization structure matters, this is a notable omission.
  - The paper does not report sensitivity to initialization of the MLP or learned input vector \(z\), even though optimization behavior is central.
  - It would also be valuable to test whether a simpler reparameterization, such as factorizing A and B through additional linear layers without nonlinearity, achieves similar gains.
- Statistical reporting is weak:
  - Most main tables report single numbers with no error bars or confidence intervals.
  - Appendix B.4 gives standard deviations for one setting, which is helpful, but this is not enough for the headline claims.
  - The paper does not report significance testing, repeated-run variance, or seed sensitivity on the main benchmarks.
- The results generally support the claim that OP-LoRA is a useful optimization-oriented reparameterization, but the extent of the gains appears task-dependent and sometimes modest.

### Writing & Clarity
- The overall narrative is understandable, but several parts of the paper impede full comprehension of the contribution:
  - Section 3.2 is the main theoretical justification, yet it mixes an informal derivation with a high-level interpretation in a way that makes it hard to tell what is proven versus what is heuristic.
  - The relationship between “trainable learning rate,” “adaptive line search,” and the actual AdamW optimization used in experiments is not crisply reconciled.
  - The appendix math is useful, but the main paper still relies on it for the key claims.
- Figures and tables are generally informative:
  - Table 1 and Table 3 clearly summarize the main results.
  - Figure 2 is central to the optimization argument, but without more quantitative detail it is difficult to assess how robust the effect is.
  - Figure 4 is helpful for the width ablation, though the paper would benefit from a more explicit discussion of why the curve is inverted-U for one task and flat for another.
- The paper is readable overall, but the main weakness is conceptual clarity around what exactly OP-LoRA contributes beyond “extra train-time capacity improves optimization.”

### Limitations & Broader Impact
- The limitations section is too light for an ICLR paper with a new optimization method.
- Key limitations that should be acknowledged more directly:
  - Increased training memory: Table 4 shows OP-LoRA uses substantially more GPU memory than LoRA.
  - Extra train-time compute: although inference is unchanged, training is not free, and in some settings it may be nontrivial.
  - Unclear scaling behavior: the paper does not show large-scale evidence that the same benefits persist as model size, adapter count, or task complexity increases.
  - The method’s benefit may depend on the specific architecture of the hypernetwork and the adapter type, despite the claim of easy extensibility.
- Broader impact discussion is minimal. There is no serious discussion of whether making adaptation easier could facilitate misuse in personalization, model impersonation, or unsafe content generation. Given that the paper includes image generation and model personalization applications, this omission is notable, though not disqualifying.

### Overall Assessment
This paper presents a genuinely interesting and potentially useful idea: train-time-only overparameterization of LoRA via a hypernetwork that is discarded at inference. The method is attractive because it is simple, architecture-flexible, and empirically helpful across several tasks. That said, for ICLR the main concern is that the core mechanism is not yet convincingly established. The theoretical argument is insightful but heuristic, the strongest optimization evidence comes from small-scale case studies, and the main benchmark improvements are mixed and sometimes modest. The added training memory is also substantial. I think the contribution is promising and likely publishable only if the authors more carefully delimit the claims, strengthen the ablations, and clarify when and why OP-LoRA reliably helps.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes OP-LoRA, a train-time overparameterized reparameterization of LoRA that uses a small MLP to generate the low-rank adapter weights and then discards the MLP at inference. The central claim is that this added train-time flexibility improves optimization by dynamically modulating effective step size and update directions, while preserving LoRA’s inference/storage footprint; the paper supports this with experiments on image generation, VQA, commonsense reasoning, and several appendices extending to other PEFT methods.

### Strengths
1. **Clear and practically appealing core idea.**  
   The method is simple in concept: replace direct learning of LoRA factors with an MLP that predicts them during training, then remove the MLP at test time. This is attractive for ICLR because it targets a real pain point in PEFT—optimization instability—without increasing inference cost.

2. **Broad empirical evaluation across modalities and architectures.**  
   The paper evaluates on Stable Diffusion XL, VL-BART, LLaMA-7B, LLaVA, and additional appendix studies on VeRA and matrix factorization. This breadth strengthens the claim that the approach is not limited to a single model family.

3. **Consistent gains over standard LoRA and DoRA in many settings.**  
   The reported tables show improvements in several tasks, including substantial CMMD reductions on image generation, modest but consistent accuracy gains on VQA and commonsense reasoning, and improved stability across runs for OP-DoRA. These are concrete, repeated wins rather than a one-off result.

4. **Inference-time efficiency is preserved.**  
   Because the MLP is discarded after training, OP-LoRA retains LoRA’s storage and inference footprint. That is an important engineering advantage over approaches that permanently add capacity.

5. **Attempt to provide an optimization explanation.**  
   The paper does not stop at empirical performance; it also offers a theoretical interpretation in terms of reparameterization, condition numbers, and an acceleration-like effect. Even if imperfect, this is aligned with ICLR’s preference for mechanistic understanding over purely empirical claims.

6. **Extensions to multiple adapter variants.**  
   The paper shows OP-LoRA can be adapted to DoRA, VeRA, and other parameterizations, which supports the claim that it is architecture-agnostic and not a one-off optimizer trick.

### Weaknesses
1. **The theoretical argument feels incomplete and in places overstated.**  
   The paper claims OP-LoRA “does not increase model capacity” and therefore avoids overfitting risk, but the MLP creates a much larger train-time parameterization that can still change optimization and implicit regularization in nontrivial ways. The analysis is suggestive, but the leap from reparameterization to improved conditioning and “adaptive line search” is not fully rigorous, especially for the nonlinear MLP case.

2. **Strong claims about generality may be ahead of the evidence.**  
   The method is evaluated on several tasks, but many results are modest, and in some cases the gains are small. For example, on VQA the improvements are around 0.5–1 point, and on commonsense reasoning the gains vary by dataset. This is useful but not yet enough to justify broad claims that OP-LoRA is universally superior.

3. **Comparisons are not always controlled cleanly enough.**  
   OP-LoRA increases train-time memory substantially, and its width/hyperparameters differ across tasks and sometimes across baselines. While the paper reports wall time and memory, it is not always clear that each baseline is equally well tuned or that performance differences are solely due to the reparameterization rather than extra tuning effort or differing effective optimization budgets.

4. **The optimization story is empirically underdeveloped.**  
   The MNIST case study and gradient analysis are helpful, but they are small-scale and somewhat stylized. The link between the observed behavior and the main-scale improvements remains correlational rather than demonstrably causal.

5. **Potential reproducibility concerns remain.**  
   The paper says code is provided, which is good, but several experimental details that matter for reproducing the strongest claims are only summarized briefly or deferred to appendices. Given the number of tasks and model families, ICLR reviewers would likely want more explicit details on tuning protocol, random seeds, and selection criteria.

6. **Some claims read as promotional rather than carefully qualified.**  
   Statements such as “blessing of dimensionality” and “adaptive line search” are interesting, but the wording sometimes implies a stronger understanding than the evidence strictly supports. ICLR tends to favor precise, limited claims backed by direct evidence.

### Novelty & Significance
The idea of using train-time-only overparameterization for PEFT is reasonably novel in the LoRA literature, especially in the specific form of an MLP that generates adapter weights and is then discarded. It is more novel as an optimization/reparameterization contribution than as a new adapter family, and it could be significant if the gains hold broadly under controlled comparisons.

Against ICLR’s standards, this is a potentially publishable direction because it is simple, broadly applicable, and empirically promising. However, the acceptance bar at ICLR is high: reviewers will likely ask whether the gains are large and consistent enough, whether the theory meaningfully explains the effect, and whether the method truly offers a better trade-off than stronger baselines. As written, the work looks promising but somewhat overclaimed relative to the strength of the evidence.

### Suggestions for Improvement
1. **Tighten the empirical comparisons with stronger controls.**  
   Add matched hyperparameter tuning budgets for all baselines, report validation-selection procedures, and provide sweeps showing that OP-LoRA’s gains are not due to more favorable tuning. Explicitly compare against parameter-matched alternatives where possible.

2. **Strengthen the theory with clearer assumptions and limitations.**  
   Separate rigorous claims from intuition. For example, clearly state which parts apply only to the linearized setting and which are conjectural for the nonlinear MLP. A more careful treatment of the nonlinear case would improve credibility.

3. **Report more complete ablations.**  
   Include sensitivity to MLP depth, width, initialization, and whether the MLP is shared across layers. Also ablate the effect of generating A and B separately versus jointly, and quantify when OP-LoRA stops helping.

4. **Clarify compute trade-offs in a standardized way.**  
   Report training memory, wall-clock, throughput, and number of optimizer steps for all methods in a single table across tasks. This would make the practical cost-benefit of OP-LoRA much easier to judge.

5. **Add more direct evidence for the optimization mechanism.**  
   The MNIST and matrix factorization studies are useful but small. Consider adding Hessian/curvature or gradient-alignment analysis on at least one large-scale task, or an ablation that isolates whether the main benefit comes from dynamic step-size modulation versus simply added train-time capacity.

6. **Be more cautious in framing the claims.**  
   Rephrase broad statements about avoiding overfitting or universally improving optimization into narrower, evidence-backed claims. This would better fit ICLR’s expectation for precise scientific writing.

7. **Improve reproducibility details.**  
   Provide exact seeds, early stopping/selection criteria, all dataset preprocessing steps, and full training scripts or config files. Since the method spans multiple model families, reproducibility will be a key concern.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a matched-parameter baseline that gives standard LoRA the same train-time parameter budget as OP-LoRA, e.g. a wider LoRA or an equivalent extra MLP that is not discarded, because otherwise the gains may just come from more trainable parameters rather than the proposed reparameterization. ICLR reviewers will expect a fair test of “overparameterization helps optimization” versus “more optimization capacity helps.”

2. Run ablations that separate the effect of the MLP reparameterization from simple changes in optimization dynamics: compare OP-LoRA to LoRA with stronger learning-rate schedules, gradient clipping, larger batch size, more warmup, and multiple optimizers across all main tasks. Without this, the claim that OP-LoRA is intrinsically more robust and not just better tuned is not convincing.

3. Compare against stronger and more relevant PEFT baselines on the same tasks, especially recent diffusion and LLM adapters beyond the selected set, and include direct comparisons at matched inference rank/parameter budgets. The main ICLR contribution should survive against current best PEFT methods, not only against a subset of gradient-alignment methods.

4. Add full multi-seed results with confidence intervals or statistical tests on every main benchmark, especially Table 1 and Table 3 where many gains are small. Single-run improvements of 1–2% are not enough for ICLR-level confidence when variance can be comparable to the reported gains.

5. Include ablations over MLP design choices that are core to the method: depth, activation, initialization, whether one shared MLP generates A and B or separate heads, and whether discarding the MLP is essential. Right now it is unclear which part of OP-LoRA is actually responsible for the gains.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify how OP-LoRA changes optimization geometry over training, not just at the end: report Hessian spectrum proxies, gradient norms, and condition-number estimates at multiple checkpoints. The paper’s central claim is about escaping ill-conditioning, but the evidence is currently too sparse to show this mechanism is real and not task-specific.

2. Analyze whether OP-LoRA merely increases effective rank or changes the rank trajectory differently from LoRA. Since the method is sold as “same capacity, better optimization,” you need to show that improvements are not due to hidden capacity expansion or a different final rank distribution.

3. Provide a careful cost analysis that includes peak memory, throughput, and total training cost normalized by task size and number of trainable parameters. The current wall-time comparison is too narrow for ICLR, and the memory overhead of OP-LoRA is substantial enough to affect its practical claim.

4. Test sensitivity across ranks, base models, and task scales in a systematic grid. The paper claims generality across image, VQA, and commonsense tasks, but the evidence is not enough to show that OP-LoRA reliably helps across low-rank and moderate-rank regimes or only in a few favorable settings.

5. Clarify the relationship to existing overparameterization theory with a direct empirical-theoretical link. Right now the “trainable learning rate” and “adaptive line search” story is plausible, but the paper needs analysis showing these terms actually correlate with improved progress during training on real models.

### Visualizations & Case Studies
1. Plot training dynamics for LoRA vs. OP-LoRA across several layers: effective step size, gradient alignment, update norms, and loss over time. This would reveal whether OP-LoRA truly adapts step size as claimed or whether the effect is localized to the toy MNIST case.

2. Add failure-case visualizations for image generation and VQA where OP-LoRA still fails, alongside success cases. Without counterexamples, it is impossible to tell whether the method improves robustness or just shifts the failure mode.

3. Show per-layer adapter evolution for A and B under OP-LoRA, including singular values and how many columns of the generating MLP are actually used. This would expose whether the method is learning a meaningful overparameterized representation or simply converging to an ordinary LoRA solution with extra overhead.

4. Include side-by-side samples at matched seed prompts and multiple random seeds for diffusion results, not just a curated set. The current qualitative figures are not enough to rule out cherry-picking or prompt-specific wins.

### Obvious Next Steps
1. Benchmark OP-LoRA on more standard ICLR PEFT settings such as instruction tuning, QA, and more widely used LLM backbones, with public, reproducible scripts. The current task mix is broad but not yet enough to establish OP-LoRA as a general PEFT advance.

2. Extend the method to a truly new adapter family and show the implementation simplicity claim in practice, not just OP-DoRA/OP-VeRA. ICLR reviewers will want evidence that the method generalizes beyond minor variants of LoRA.

3. Perform a clean compute-efficiency study that searches for the best LoRA baseline under the same compute budget as OP-LoRA. If OP-LoRA needs more training memory and slightly more time, then budget-matched comparisons are necessary to support the contribution.

4. Evaluate whether the discarded MLP could be distilled or compressed further without losing gains. Since the method’s selling point is training-time overparameterization, the obvious next step is to determine the minimum extra machinery needed to preserve the benefit.

5. Release a standardized ablation suite showing when overparameterization helps and when it hurts. This would make the paper’s claim about “the blessing of dimensionality” substantively useful rather than anecdotal.

# Final Consolidated Review
## Summary
This paper proposes OP-LoRA, a train-time-only overparameterization of LoRA in which a small MLP predicts the low-rank adapter weights and is then discarded before inference. The core claim is that this reparameterization makes LoRA optimization easier and less learning-rate sensitive while preserving LoRA’s inference/storage footprint; the paper supports this with a theoretical sketch plus experiments on diffusion, VQA, commonsense reasoning, and a few appendix extensions.

## Strengths
- The method is simple and attractive: generate LoRA factors with a small MLP during training, then remove the MLP at inference, so deployment cost stays the same as standard LoRA. This is a clean PEFT idea and is easy to integrate in code.
- The paper evaluates across multiple model families and tasks, including Stable Diffusion XL, VL-BART, LLaMA-7B, and LLaVA, and also includes appendix extensions to VeRA and matrix factorization. The broad scope strengthens the claim that the idea is not tied to one backbone.
- There are consistent improvements over standard LoRA/DoRA in many settings, with especially large gains on image generation (e.g., substantial CMMD reductions on Naruto and WikiArt) and modest but repeatable gains on VQA and commonsense tasks. The appendix also shows improved stability for OP-DoRA and better performance on a toy optimization case study.

## Weaknesses
- The main optimization story is still more heuristic than established. The paper’s “trainable learning rate” and “adaptive line search” interpretation is derived from a heavily simplified reparameterized setting, and the connection to the actual AdamW-based, nonconvex training used in the experiments is not demonstrated rigorously. This matters because the paper’s central claim is about optimization, not just another way to add train-time capacity.
- The empirical evidence is not strong enough to support the breadth of the claims. Some gains are large in diffusion, but others are small, and OP-LoRA is not uniformly best on commonsense reasoning where recent gradient-alignment methods remain competitive or better on some tasks. This weakens the impression that OP-LoRA is a generally superior PEFT method rather than a task-dependent trick.
- The training cost is nontrivial and should be emphasized more honestly. OP-LoRA increases GPU memory substantially over vanilla LoRA and is slightly slower in wall time, so the method trades inference efficiency for heavier training. That is acceptable, but it limits the practical appeal and undercuts any overly broad “cheap optimization” framing.

## Nice-to-Haves
- A more complete ablation on MLP design would be useful: depth, activation, initialization, whether A and B should be predicted jointly or separately, and whether the learned input vector z matters.
- The paper would benefit from more standardized reporting of variance: multi-seed means and confidence intervals on the main tables, especially where gains are only 1–2 points.
- A clearer cost table that combines peak memory, throughput, total wall time, and adapter rank would make the practical trade-off easier to judge.

## Novel Insights
The genuinely interesting idea here is that train-time overparameterization can be used as an optimization aid for PEFT without changing inference-time complexity, which is a cleaner framing than many optimizer-specific LoRA variants. The strongest mechanistic takeaway is not that the MLP adds capacity in the usual sense, but that it changes the training dynamics in a way that appears to improve robustness to learning-rate choice and can help escape poor curvature, as suggested by the small-scale MNIST and matrix-factorization diagnostics. That said, the evidence currently supports this as a plausible and useful hypothesis more than a settled explanation.

## Potentially Missed Related Work
- None identified

## Suggestions
- Add controlled baselines that match OP-LoRA’s train-time parameter budget and compare against stronger tuning of vanilla LoRA, including more learning-rate schedules and optimizers, to isolate whether the gains come from reparameterization itself.
- Report multi-seed results with error bars on the main benchmarks.
- Provide a clearer optimization analysis on at least one larger-scale task, ideally with curvature or gradient-alignment measurements over training, not just toy settings.
- Tighten the claims: present OP-LoRA as a useful train-time reparameterization that often helps optimization, rather than as a broadly solved explanation for LoRA’s optimization issues.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 4.0, 2.0]
Average score: 3.0
Binary outcome: Reject
