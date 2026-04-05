=== CALIBRATION EXAMPLE 78 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate: the paper is indeed about quantized video generative models and pushes low-bit QAT settings. “Pushing the Limit” is somewhat promotional, but that is not uncommon for ICLR.
- The abstract clearly states the problem, the proposed method, and the main empirical claim.
- However, some claims are stronger than the evidence presented in the main paper as extracted here:
  - “the first to reach full-precision comparable quality under 4-bit settings” is a very strong claim that depends on the exact baselines, evaluation protocol, and model families. The paper does compare against several PTQ/QAT methods, but the “first” claim would need careful qualification.
  - The abstract emphasizes the rank-decay strategy as zero-overhead, but the paper later acknowledges training-time overhead from the extra phases and from storing Φ during training. The zero-overhead claim is only about inference, which should be stated more explicitly.

### Introduction & Motivation
- The motivation is strong and relevant for ICLR: video diffusion models are expensive, and low-bit deployment is an important practical bottleneck.
- The paper does identify a real gap: quantization for image DMs has progressed, but video DMs at 3/4-bit appear much harder, and existing methods degrade substantially.
- The contributions are stated clearly, but one point feels overstated: calling QVGen “the first QAT method for video generation” is plausible given the related work cited, yet the paper should be more precise about whether it is the first for diffusion-based video generation, DiT-based video generation, or all video generative models.
- The introduction does not sufficiently explain why prior image-DM QAT methods fail on video beyond “poor convergence.” That gap is later addressed, but the motivating problem could have been framed more concretely with the temporal complexity of video and the different gradient behavior.
- On ICLR standards, the problem is timely and important, but the novelty should be framed more cautiously unless the comparison set is exhaustive.

### Method / Approach
- The method is clearly structured into two pieces: auxiliary modules Φ to reduce gradient norm, and rank-decay to remove Φ at inference.
- The main high-level idea is understandable and potentially useful, but several technical aspects need clarification for reproducibility and for judging correctness.

- **Theoretical analysis**
  - Theorem 3.1 is used to argue that minimizing gradient norm improves convergence. This is a fairly standard and somewhat indirect result: it basically restates that the average regret bound depends on gradient norms.
  - The theorem appears to be adapted from online convex optimization and then used as a surrogate justification for deep nonconvex QAT. The paper does acknowledge nonconvexity in Appendix C, which is good, but the core argument still relies on assumptions that are not really valid for large video DMs.
  - The logical leap from regret bounds to the practical design of Φ is not fully justified. The theorem supports “smaller gradients are better,” but not specifically that adding low-rank auxiliary linear modules initialized from weight quantization error is the best or necessary way to achieve this.
  - In particular, the claimed causal link between high gradient norm and performance drops in Q-DM is suggestive, but not a proof that gradient norm reduction is the primary cause.

- **Auxiliary module Φ**
  - Eq. (9) defines Φ as an extra linear term with weights initialized from the quantization error **W** − Qb(**W**). This is a sensible initialization, but the paper should better justify why this particular residual structure is sufficient for all linear layers in a DiT.
  - It is also unclear whether Φ is inserted into every linear layer uniformly or only selected layers. The text suggests “enhance each quantized linear layer,” but the exact scope matters for overhead and reproducibility.
  - The paper does not discuss whether Φ changes the optimization landscape in a way that could destabilize some layers, especially attention projections versus MLPs.

- **Rank-decay strategy**
  - The SVD-based decomposition and progressive truncation are reasonable and interesting.
  - But the explanation of Eq. (12)–(14) is not fully transparent:
    - The role of γ as a per-component decay mask is understandable, but the exact indexing and schedule are somewhat hard to follow.
    - It is unclear how often SVD is recomputed, how expensive it is, and whether it is done layerwise or globally.
    - The method repeatedly decomposes and truncates Φ, but the paper does not quantify the cost of repeated SVDs beyond training-time overhead in aggregate.
  - A key question is stability: repeated SVD truncation can introduce discontinuities. The paper reports good results, but does not thoroughly analyze failure cases when the singular spectrum is not clearly separated.
  - Another important issue is whether rank-decay is genuinely needed or whether simple low-rank training from the outset would work similarly. The ablation with “+Φ” vs “+Rank” helps somewhat, but more direct baselines would strengthen the claim.

- **Reproducibility**
  - The paper gives many training details in Appendix D, which is good.
  - Still, the exact implementation of quantization, layer selection, SVD frequency, decay phase schedule, and whether all quantization parameters are updated jointly with Φ are not stated at a level that would allow direct reproduction without the code.

### Experiments & Results
- The experiments do test the core claims: 3-bit and 4-bit quantization on multiple video diffusion models, ablations on auxiliary modules and rank-decay, and comparisons with several baselines.
- The evaluation is meaningful and uses established benchmarks (VBench, VBench-2.0), which is appropriate for ICLR.
- That said, there are several concerns about the strength and completeness of the evidence.

- **Baselines**
  - The baseline set is reasonable in spirit, including PTQ and QAT methods. However, the paper adapts image-DM or CNN-oriented methods to video DMs. That is acceptable, but some of these adaptations may disadvantage baselines.
  - For fair comparison, it would help to know whether each baseline was tuned equally carefully, especially given the sensitivity of ultra-low-bit QAT.
  - The paper compares against SVDQuant with group-wise settings and some layers kept in higher precision. That is informative, but it also means some comparisons are not apples-to-apples. The authors do note this, but the main tables mix direct and adapted settings, which complicates interpretation.

- **Main results**
  - Table 1 is the central result. QVGen clearly outperforms the adapted baselines in the reported metrics, especially at 3-bit and 4-bit. This is convincing evidence that the method helps.
  - However, the claim that 4-bit QVGen is “full-precision comparable” is somewhat dependent on metric selection. In Table 1, some metrics are close, but others still show nontrivial gaps, and performance on Scene Consistency remains challenging.
  - The paper should be more careful in distinguishing “close on average” from “comparable in every dimension.” For ICLR, a stronger claim would need either more benchmarks or clearer statistical reporting.

- **Ablations**
  - The ablations in Tables 3–6 are useful and directly relevant.
  - The auxiliary module Φ clearly helps, and rank-decay preserves most of the gain while removing inference overhead.
  - Still, a few missing ablations would materially strengthen the conclusions:
    - No direct comparison to a plain low-rank adapter baseline without the specific initialization from quantization error.
    - No ablation on the choice of inserting Φ into different subsets of layers.
    - No comparison to gradient-norm regularization or clipping as a primary method beyond Appendix H.3. Since the paper’s theory emphasizes gradient norm, this baseline is important.
    - No ablation on how many SVD refreshes are performed, or whether one-time decomposition would suffice.

- **Statistical reporting**
  - The tables do not show error bars or multiple-seed variance for the main results.
  - There is one random-control ablation in Table O, but the core benchmark results appear to be single-run values.
  - Given the complexity of video generation evaluation, and the relatively large claims, ICLR-level evidence would benefit from confidence intervals or at least seed averages.

- **Evaluation metrics**
  - VBench and VBench-2.0 are appropriate benchmarks for video generation.
  - The choice of 8 dimensions is reasonable, but the paper relies heavily on benchmark scores rather than human evaluation or downstream utility.
  - The paper also reports PSNR/SSIM/LPIPS against BF16 outputs, but these metrics are limited for generative models and should not be overinterpreted.

- **Efficiency claims**
  - The latency and memory analysis is useful, and the paper does make an effort to address the overhead introduced by Φ.
  - Still, some efficiency claims depend on custom kernels and profiling assumptions. The paper notes that fusion could yield more speedup, but the reported runtime gains are not fully end-to-end in the strongest sense.
  - Training cost is substantial, and while the paper reports it, the method is not especially lightweight during training.

### Writing & Clarity
- The overall structure is good, and the paper is ambitious and well-motivated.
- The main clarity issue is technical density in the method section:
  - Eq. (12)–(14) and the rank-decay schedule are conceptually interesting, but the exposition is hard to parse without closely reading the appendix.
  - The transition from the theorem to the design of Φ feels somewhat abrupt.
- Tables and figures generally communicate the intended story, especially Fig. 1, Fig. 3, Fig. 4, and the ablation tables. The qualitative figures support the main claims.
- The paper could be clearer in separating:
  - what is theoretically justified,
  - what is empirically observed,
  - and what is an engineering choice.

### Limitations & Broader Impact
- The limitations section is too weak for an ICLR paper.
  - The stated limitation is essentially “we focus on video generation, but hope to generalize to NLP.” That is not the key limitation.
- Important missing limitations include:
  - The method is still expensive to train, requiring substantial GPU time and memory.
  - The approach depends on access to a full-precision teacher and a sizable training set.
  - It is validated on a small set of large video diffusion families; generalization to other architectures, especially non-DiT or non-diffusion video generators, is not established.
  - The dependence on SVD-based shrinking may make the method less appealing in settings where training-time complexity matters.
- Broader impact is not meaningfully discussed. Since the paper concerns generative video models, it should at least acknowledge potential misuse, synthetic media concerns, and deployment implications. This is especially relevant at ICLR.

### Overall Assessment
QVGen appears to be a substantive and timely contribution: it addresses an important and underexplored problem, proposes a coherent method, and shows strong empirical gains over adapted baselines on several major video diffusion models. The core idea of using auxiliary modules to stabilize ultra-low-bit QAT, then removing them through rank-decay, is plausible and practically relevant. That said, the paper’s strongest claims outpace the evidence in a few places: the theoretical justification is not fully compelling for deep nonconvex video diffusion, the baseline comparisons are sometimes uneven, and the experiments lack multi-seed variance and some crucial ablations. For ICLR, this is promising and likely impactful work, but I would want a more careful statement of novelty and stronger evidence that the gains are robust beyond the specific model families and evaluation setup studied here.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces QVGen, a quantization-aware training framework for video diffusion models that targets ultra-low-bit regimes (3-bit and 4-bit). The main idea is to add auxiliary low-rank compensation modules during training to stabilize optimization by reducing gradient norms, and then progressively remove these modules via a rank-decay schedule so that inference remains overhead-free.

### Strengths
1. **Addresses an important and timely problem for ICLR.**  
   Efficient deployment of large video diffusion models is highly relevant, and the paper tackles a concrete bottleneck: direct low-bit quantization of video DMs performs poorly. The paper makes a credible case that video generation is harder to quantize than image generation, supported by comparisons showing much larger degradation for existing methods in W3A3/W4A4 settings.

2. **Clear empirical gains over strong baselines.**  
   Across multiple model families (CogVideoX-2B/1.5-5B, Wan 1.3B/14B), QVGen substantially outperforms PTQ and QAT baselines, often by large margins on VBench metrics. The paper also reports results on VBench-2.0 for large models and shows near full-precision performance in some 4-bit settings, which is a strong result if the evaluation is sound.

3. **The method is practically motivated and deployment-oriented.**  
   A notable contribution is the rank-decay mechanism that aims to remove the auxiliary modules after training, preserving inference efficiency. This is important for ICLR standards because it goes beyond accuracy-only improvements and considers deployment costs.

4. **Reasonable ablation support for key design choices.**  
   The paper includes ablations on shrinking ratio, initial rank, decay strategy, initialization, and annealing functions. These experiments help support the specific design of the auxiliary-module plus progressive-decay framework.

5. **Attempts to provide theory and additional analysis.**  
   The paper offers a convergence analysis linking regret to gradient norm, and also includes a nonconvex variant in the appendix. While not fully rigorous for deep nets, it shows an effort to explain why the method works rather than relying only on empirical evidence.

### Weaknesses
1. **The theoretical claims are weakly connected to the actual setting.**  
   The main theorem is based on convex regret analysis, and the paper itself acknowledges that this may not hold for deep networks. The nonconvex appendix result is more standard and does not uniquely explain the specific benefits of the proposed auxiliary modules. As written, the theory is more suggestive than explanatory, and the leap from the analysis to the method’s success is not fully justified.

2. **The method appears somewhat ad hoc despite the strong empirical gains.**  
   The introduction of auxiliary modules initialized from quantization error, followed by rank decomposition and scheduled decay, is plausible, but the paper does not fully establish why this particular combination is the best or most principled solution. Several choices, such as the rank schedule, shrinking ratio, and decay ordering, seem empirically tuned.

3. **Limited clarity on reproducibility details for a complex training pipeline.**  
   Although the paper includes many implementation details and an appendix, the overall training recipe is still fairly complex: different models use different GPU counts, different epoch counts, different solvers, different resolutions, and different decay schedules. For ICLR-level reproducibility, it would help to have a more compact, explicit algorithmic specification and a clearer statement of all hyperparameters per model.

4. **Potential concern about baseline fairness and comparability.**  
   The paper adapts image-DM quantization baselines to video DMs, which is necessary, but this can make comparisons less definitive because some baselines were not originally designed for this setting. The paper tries to be fair by matching settings, yet the extent to which each baseline is optimally adapted remains somewhat unclear.

5. **The empirical evaluation, while broad, is focused mainly on one benchmark family.**  
   Most results rely on VBench and VBench-2.0, which are relevant and standard, but the paper would be stronger with more diverse evaluation, such as human preference studies, downstream task evaluation, or stress tests on longer/higher-resolution generations. The qualitative examples are helpful but not enough to fully validate the claims.

6. **Some claimed novelty may be narrower than stated.**  
   The paper claims to be the first QAT method for video generation and the first to reach full-precision-comparable 4-bit quality. The first claim is plausible but should be carefully scoped relative to prior video quantization work; the second is an empirical claim that depends heavily on benchmark choice and chosen models.

### Novelty & Significance
**Novelty:** Moderate to strong. The core novelty is adapting QAT to video diffusion models with a training-time compensation module and a progressive low-rank decay mechanism to remove inference overhead. The specific combination of gradient-norm motivation, low-rank auxiliary compensation, and scheduled elimination is reasonably novel in the video quantization space, though each individual ingredient builds on known ideas from quantization, low-rank adaptation, pruning, and optimization.

**Significance:** High if validated. If the reported gains are robust, this is a meaningful step toward practical deployment of large video generative models at 4-bit precision. That said, ICLR typically expects not only strong benchmark gains but also clear methodological insight and convincing generality; this paper is strongest on empirical utility, somewhat weaker on conceptual depth.

**Clarity:** Fair. The paper is generally understandable, but the mathematical exposition is uneven and the presentation is dense, especially in the theory and decay mechanism. The high-level idea is clear, but some implementation and theoretical details are hard to parse.

**Reproducibility:** Moderate. The paper provides a code link, appendix details, and many hyperparameters, which is positive. However, the complexity of the pipeline and the model-specific training setup reduce reproducibility unless the released code is very complete and faithful.

### Suggestions for Improvement
1. **Strengthen the theoretical justification.**  
   Either provide a tighter analysis connecting the auxiliary modules and rank-decay schedule to optimization stability in the nonconvex setting, or clearly frame the current theory as intuition rather than a formal explanation. A more direct derivation showing why the auxiliary module reduces gradient variance or loss spikes would be more convincing.

2. **Provide a cleaner algorithmic description.**  
   Add a compact pseudocode block with all phases, transitions, and stopping criteria spelled out unambiguously. A per-model hyperparameter table would also help readers reproduce the method without cross-referencing many sections.

3. **Include stronger baseline and fairness discussion.**  
   Clarify exactly how each baseline was adapted to video DMs, what was held fixed, and whether any tuning was done per baseline. If possible, include a broader set of baselines or show that the gains persist under multiple adaptation strategies.

4. **Add more evidence on generality.**  
   Since the method is claimed to be general-purpose, it would be useful to test on additional video architectures, longer horizons, or at least a non-diffusion generative setting. The image-generation appendix results help, but broader cross-domain evidence would strengthen the paper substantially.

5. **Report more direct efficiency outcomes.**  
   The paper discusses latency and memory, but ICLR readers would benefit from end-to-end deployment metrics such as throughput, peak memory, and wall-clock generation time under standard settings, ideally compared against the strongest practical quantization baselines on the same hardware.

6. **Clarify what parts of the method are essential.**  
   Ablations should more explicitly separate the contributions of: initialization of Φ, low-rank factorization, gradient-norm reduction, and the decay schedule. This would help readers understand whether the core gain comes from low-rank compensation itself or from the progressive elimination procedure.

7. **Tone down or qualify strongest claims.**  
   Phrases such as “the first to reach full-precision comparable quality” should be scoped carefully to the evaluated models, metrics, and settings. ICLR reviewers often react negatively to broad claims that are stronger than the evidence supports.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a true end-to-end speed/memory evaluation on the same hardware and inference stack used by the strongest PTQ baselines, including total throughput over a full video generation pipeline, because the paper’s “zero overhead” and deployment claims depend on more than per-layer INT4 GEMM speed. ICLR reviewers will expect the claimed benefits to survive realistic sampling schedules, VAE decoding, attention kernels, and batch-size effects.

2. Add comparisons against stronger and more recent video-specific quantization baselines at the exact same bit settings and with identical treatment of unquantized layers, because current results mix 4/6-bit, per-group exceptions, and partial exclusions that can inflate the apparent gain. Without a stricter apples-to-apples benchmark, the “first to reach BF16-like 4-bit video DM quality” claim is not convincing.

3. Add ablations that isolate the auxiliary module Φ design from rank-decay, including a constant-rank low-rank adapter, a no-SVD low-rank adapter, and a direct penalty on the full module norm. The paper claims gradient-norm reduction and rank-decay are the key innovations, but the current ablations do not rule out that any extra trainable residual branch would give most of the benefit.

4. Add results across multiple training-data sizes and sampling subsets, not just 16K clips from one dataset, because the method’s stability claim is under-supported. ICLR reviewers will likely ask whether the method still works with less data, different caption quality, or a different video corpus, since quantization-aware training can be highly data-sensitive.

5. Add at least one genuine 2-bit or mixed-precision setting if the paper keeps implying a path toward “2-bit quantization,” because the current evidence is only for 3/4/6-bit and does not justify extrapolation. If the paper’s contribution is “pushing the limit,” the limit must be tested where the method becomes genuinely hard.

### Deeper Analysis Needed (top 3-5 only)
1. Provide a stronger causal analysis connecting reduced gradient norm to improved quantized convergence, ideally with controlled experiments where gradient norm is directly manipulated while keeping the rest fixed. Right now the paper shows correlation, but not that gradient norm is the mechanism rather than a symptom of the added capacity from Φ.

2. Quantify how much of the gain comes from the low-rank structure itself versus simply adding extra parameters during training, because the rank-decay story is only convincing if the low-rank path is better than an equally sized dense or LoRA-like alternative. Without this, the SVD/rank-decay explanation is not yet a necessary part of the method.

3. Analyze sensitivity to the choice of layers being quantized and to model scale, especially for attention, normalization-adjacent, and cross-attention projections. The current results suggest some metrics remain fragile; reviewers will want to know whether failures concentrate in a few components or are systemic.

4. Report variance across seeds and prompts for the main comparisons, not just averaged benchmark scores, because video generation metrics can be noisy and the paper makes strong “SOTA” claims. A single-seed or low-variance presentation is insufficient for ICLR-level confidence.

5. Clarify whether the performance gains persist after the auxiliary module is fully removed in all layers and at all stages, with a layerwise trajectory of metric recovery during decay. The paper claims negligible overhead at the end, but it is unclear where the irrecoverable degradation, if any, enters during the decay schedule.

### Visualizations & Case Studies
1. Show layerwise and timewise plots of gradient norms, singular-value spectra, and quantization error before/after Φ and during decay, because the central hypothesis is that Φ stabilizes training and then becomes redundant. These plots would reveal whether the mechanism is real or whether rank-decay is just a heuristic that happened to work on a few models.

2. Add failure-case videos and frame-by-frame comparisons for the hardest metrics, especially Scene Consistency and long-range motion, because the benchmark tables already show these are the weakest points. Case studies should include prompts where QVGen still fails so readers can see the method’s actual boundary.

3. Provide a per-layer decomposition of where Φ is inserted and where decay removes capacity first, ideally with layerwise importance heatmaps. This would reveal whether the method is learning meaningful error compensation or simply shrinking all components uniformly until performance degrades.

4. Add visual comparisons under the same prompt and seed across BF16, QVGen, and the strongest baselines for both 3-bit and 4-bit settings, with synchronized frame strips. The current figures are helpful but not enough to verify temporal consistency, object identity preservation, and prompt adherence across the full clip.

### Obvious Next Steps
1. Extend the method to 2-bit and mixed-precision video quantization with a clear failure analysis, because that is the natural next test for a paper claiming to push the limit of ultra-low-bit video generation.

2. Replace or complement the current auxiliary-module training with a method that avoids extra full-precision storage during QAT, since Φ still increases training complexity and memory even if inference cost is removed. An ICLR-level contribution would show how to get the same gain with less training overhead.

3. Validate on additional video generators and training regimes beyond the four chosen DiT-style models, including newer architectures or non-diffusion video generators if applicable. The current evidence is strong on one model family but not enough to claim generality of the framework.

4. Demonstrate compatibility with aggressive inference optimizations such as quantized attention, kernel fusion, and structured sparsity in a single integrated pipeline. The paper already hints at this, and it is the most direct route to turning the method from a model-quality result into a deployable systems contribution.

# Final Consolidated Review
## Summary
This paper proposes QVGen, a quantization-aware training framework for video diffusion models at ultra-low bitwidths, targeting 3-bit and 4-bit settings. The core idea is to add auxiliary low-rank compensation modules during training to stabilize optimization, then progressively eliminate those modules with an SVD-based rank-decay schedule so inference remains standard quantized inference with no extra module cost.

## Strengths
- The paper tackles a genuinely important and underexplored problem: low-bit quantization for large video diffusion models, where prior image-DM quantization methods break down badly. The reported gains across CogVideoX and Wan models, especially in 4-bit and even 3-bit settings, suggest the problem is real and the method is not just a minor tweak.
- The empirical results are strong and broadly convincing on the chosen benchmarks. QVGen consistently outperforms adapted PTQ/QAT baselines on VBench and shows near-full-precision behavior in several 4-bit cases, while also providing ablations on the auxiliary module, rank-decay schedule, shrinking ratio, initialization, and decay strategy.

## Weaknesses
- The theory is much weaker than the paper’s framing suggests. The main convergence argument is essentially a standard regret bound tied to gradient norm, and the paper openly acknowledges the convex assumption does not hold for the actual deep video models. The analysis is therefore more of a loose intuition than a real justification for why this specific auxiliary-module-and-rank-decay design should work.
- Several of the strongest claims outpace the evidence. Phrases like “the first” and “full-precision comparable quality” are too broad unless tightly scoped to the exact model families, metrics, and adaptation choices used here. The tables show clear improvement, but not uniformly across all dimensions, and some gaps remain, especially on harder consistency metrics.
- The method is still training-heavy and somewhat ad hoc. The paper removes inference overhead, but it does so by introducing extra full-precision modules, repeated SVD steps, and multiple decay phases during training. That may be acceptable for research, but it limits the practical appeal of the method and the paper does not convincingly show that simpler alternatives would not achieve much of the same benefit.

## Nice-to-Haves
- A cleaner ablation that separates the benefit of “extra trainable residual branch” from the benefit of the specific low-rank decomposition and decay schedule would make the mechanism much more credible.
- More direct runtime reporting for the full generation pipeline, not just per-layer or partial kernel profiling, would make the deployment story stronger.
- A clearer per-model hyperparameter table and a compact pseudocode summary of all decay phases would improve reproducibility.

## Novel Insights
The most interesting insight in the paper is not just that low-bit video diffusion is hard, but that the training dynamics seem to be the real bottleneck: the authors connect instability to large gradient norms, then use an auxiliary residual path to absorb quantization error and reduce those gradients. The rank-decay idea is also a clever systems-oriented twist, because it tries to make a training-time crutch disappear by exploiting the observation that the auxiliary module becomes increasingly low-rank over time. That said, the novelty is mainly in the combination and packaging of known ingredients rather than in a deep new theoretical principle.

## Potentially Missed Related Work
- EfficientDM — relevant as the closest QAT baseline for diffusion models and already compared against, but worth explicitly distinguishing from the paper’s gradient-stabilization angle.
- SVDQuant — relevant because the paper reuses or compares to low-rank compensation ideas and even combines with it in the appendix; this is the most important adjacent low-rank quantization work.
- Q-DM / QVD / ViDiT-Q — relevant video-diffusion quantization baselines that frame the practical comparison set and the extent of the claimed improvement.

## Suggestions
- Tighten the claims: explicitly scope “first” and “full-precision comparable” to the evaluated model families, metrics, and bit settings.
- Add a direct baseline where the model uses a fixed low-rank adapter without rank-decay, and another where the same extra capacity is added as a dense residual branch, to show rank-decay and low-rank structure are actually necessary.
- Report end-to-end video generation latency and memory on the same hardware stack used for the strongest baselines, including attention, VAE decoding, and any kernel-fusion effects.
- Include seed-averaged results or error bars for the main benchmark tables, since the claims are strong and video metrics can be noisy.
- Clarify the exact insertion points of Φ and the schedule for SVD refreshes/decay phases so the method is reproducible without code inspection.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0, 8.0, 6.0]
Average score: 6.8
Binary outcome: Accept
