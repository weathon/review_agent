=== CALIBRATION EXAMPLE 53 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is directionally accurate: the paper is about LoRA hyperparameter tuning and a system for making that tuning more efficient. However, “PLORA” and “efficient LoRA hyperparameter tuning” slightly understate that the core contribution is not just tuning, but also packing multiple LoRA jobs and custom kernels.
- The abstract states the problem, method, and headline speedups clearly enough.
- The main concern is that the abstract claims “reduces the makespan of LoRA fine-tuning over a given hyperparameter search space by up to 7.52× and improves training throughput by up to 12.8×,” but does not clarify that these gains depend on specific search spaces, hardware, and packing assumptions. At ICLR, such system-level claims should be framed more carefully to avoid sounding overly general.

### Introduction & Motivation
- The motivation is solid: LoRA hyperparameters matter, and tuning them naively is wasteful. The paper identifies a real gap between LoRA serving work and LoRA training/tuning efficiency.
- The contributions are mostly clear, especially the shift from sequential tuning to “intra-run” packed tuning.
- That said, the introduction somewhat blurs two different contributions: empirical evidence that LoRA hyperparameters matter, and the systems contribution of packing multiple configurations with optimized kernels. It also briefly hints at “provable performance bounds” before the formal development is clear, which risks over-claiming unless the theorem is made precise.
- For ICLR standards, the introduction would benefit from a sharper articulation of why this is more than an engineering optimization: what new algorithmic or methodological insight is learned beyond better GPU utilization?

### Method / Approach
- The method is broadly described, but there are several places where reproducibility and logical clarity are not yet fully convincing.
- The main optimization formulation in Section 5 / Appendix A is difficult to follow and appears to mix scheduling, packing, and parallelism decisions. The paper claims the problem is NP-complete and then proposes DTM, but the exact objective/constraints and how the recursion over GPU budgets interacts with the ILP are not fully transparent from the presentation alone.
- Algorithm 1 and Algorithm 2 are conceptually plausible, but key details are missing:
  - How exactly is throughput estimated by the cost model from the first 10 iterations?
  - How stable is this estimate across tasks/models?
  - What happens if the cost model mispredicts memory or iteration time?
- The “provable near-optimality” claim appears to hinge on Appendix D, but the theorem statement and proof as presented are not cleanly complete in the main text. The bound seems to be about tail effects rather than the full scheduling problem, so the scope of the guarantee should be stated very precisely.
- The packed-kernel design in Appendix C is the most technical part, but the explanation remains somewhat high-level. The four backward cases are identified, yet the paper does not provide enough detail to judge correctness, especially for gradient accumulation across heterogeneous adapters and for boundary cases in tiling/reduction.
- Important edge cases are under-discussed:
  - What happens when adapter ranks differ substantially across packed configs?
  - How does packing interact with variable sequence lengths across tasks?
  - What if memory utilization is high enough that packing provides little benefit?
  - How robust is the approach if the hardware pool is heterogeneous?
- The memory model in Appendix B is useful, but it reads as an approximate analytic model rather than a validated predictor. Since scheduling depends on it, the paper should explain calibration and error bounds.
- Overall, the approach is promising and internally coherent at a high level, but the paper is currently stronger on system design intent than on fully reproducible, rigorously specified methodology.

### Experiments & Results
- The experiments do address the main claims: makespan reduction, job throughput improvement, and final model quality.
- The baseline choices are reasonable but incomplete. Comparing against sequential tuning variants with Min GPU and Max GPU makes sense, but the paper does not compare against more realistic hyperparameter tuning systems that optimize wall-clock time under a fixed budget, nor against other concurrent-training schedulers adapted to this setting.
- A key missing ablation is the effect of the offline packing planner versus simpler heuristics. The paper does include a “Sequential PLORA” ablation, which is good, but it would materially strengthen the claims to compare DTM against greedy packing, bin packing heuristics, or random packing under the same kernel implementation.
- Another missing study is sensitivity to the size of the hyperparameter search space. The main evaluation uses 120 configurations; it is unclear how PLORA scales as the search space grows beyond that, or how the gains vary with fewer/more candidate configurations.
- The results do support the claim that packing improves throughput and can reduce makespan substantially, especially on small batch sizes. However:
  - The paper does not report error bars, variance across runs, or any statistical significance.
  - The model-quality table reports only the best configuration found, which is expected for hyperparameter tuning, but it also makes the method look stronger than it may be in expectation unless the search cost and success rate are reported.
  - The comparisons are somewhat cherry-picked toward settings where packing helps most: batch size 1–4, underutilized GPU settings, and specific model/task combinations.
- The evaluation metrics are appropriate in principle, but makespan and throughput are partly a consequence of the packing policy rather than an end-to-end measure of tuning efficiency under a fixed compute budget. It would help to report “quality vs. wall-clock time” or “best achieved accuracy under a fixed time budget,” which is often what matters for tuning systems.
- One concern for ICLR is generality: the evaluation is restricted to a small set of tasks (MRPC, CoLA, WNLI, GSM8K) and mostly Qwen/Llama models. The claimed generality across “a range of state-of-the-art LLMs” is not fully demonstrated.

### Writing & Clarity
- The paper has a number of clarity issues that impede understanding of the contribution, especially in the method and appendix sections.
- The exposition of the optimization problem is hard to parse. The transition from minimizing makespan to maximizing throughput, and then to the DTM recursion, is not explained cleanly enough for a reader to reconstruct the algorithm.
- Appendix B’s memory model is especially difficult to follow because the definitions of base-model memory, adapter memory, activation memory, and sharding are interleaved. This makes it hard to tell what is actually implemented versus what is an approximate analytic abstraction.
- Figures and tables are mixed with parser artifacts in the provided text, but beyond that the paper seems to rely heavily on them to carry crucial ideas. In particular:
  - Figure 2 and Figure 3 are essential to understanding packed execution and the system architecture.
  - Table 2, Table 3, and Table 5 are central to the empirical claims.
  - The paper should ensure these figures/tables are sufficiently self-contained in the actual PDF, because the prose alone is not always enough.
- The writing is generally intelligible, but the amount of cross-referencing and late introduction of definitions makes it harder than necessary to assess the correctness of the design.

### Limitations & Broader Impact
- The paper acknowledges some limitations implicitly, such as focusing on offline search spaces and noting that other parallelisms could be future work.
- However, it does not clearly state the most important practical limitations:
  - PLORA appears most beneficial when many candidate configurations coexist and when single-config fine-tuning underutilizes the GPU.
  - The system relies on an offline planner and a cost model, which may not be suitable for highly dynamic or interactive workloads.
  - The gains likely diminish as per-job batch size or model size increases and GPU utilization improves.
- A major limitation is that packed tuning may complicate debugging, reproducibility, and fair comparison across hyperparameters, since multiple configurations are co-scheduled and their runtime interactions depend on cost-model fidelity.
- Broader-impact discussion is minimal. While this is not necessarily a societal-impact-heavy paper, the authors should at least mention that the method is an infrastructure optimization, not a new learning algorithm, and that it may shift compute efficiency rather than reduce total compute demand if used to encourage broader hyperparameter sweeps.

### Overall Assessment
PLORA addresses a real and practically important bottleneck in LoRA hyperparameter tuning: single-configuration runs often leave GPU resources underused, and packing multiple LoRA jobs is an intuitively strong idea. The empirical results suggest substantial speedups in the regimes studied, and the kernel work appears to be a meaningful systems contribution. That said, for ICLR’s bar, the paper is currently limited by incomplete methodological clarity, weak validation of the scheduling/cost-model assumptions, and an evaluation that is narrower than the breadth of the claims suggests. The contribution looks promising and likely useful, but the paper needs a clearer, more rigorous account of when the gains hold, how robust the planner is, and how it compares to stronger baselines before its generality can be fully accepted.

# Neutral Reviewer
## Balanced Review

### Summary
This paper studies the problem of efficiently tuning LoRA hyperparameters for LLM fine-tuning, arguing that standard HPO methods ignore a major source of inefficiency: many LoRA trial runs underutilize GPU compute and memory. To address this, the authors propose PLORA, a system that packs multiple LoRA configurations into concurrent fine-tuning jobs, uses an offline planner to schedule them, and introduces custom packed LoRA kernels to improve throughput. Empirically, they report substantial speedups in tuning makespan and kernel throughput, while still finding higher-quality adapters than a default LoRA setup.

### Strengths
1. **Timely and practically relevant problem framing.**  
   The paper addresses an important bottleneck in LLM adaptation: hyperparameter tuning for LoRA is expensive, and the paper’s focus on reducing the end-to-end tuning makespan is aligned with real deployment constraints that ICLR readers care about.

2. **Clear empirical motivation that LoRA hyperparameters matter.**  
   The paper presents evidence from more than 1,000 experiments and tables showing large sensitivity to learning rate, batch size, rank, and alpha, plus task/model-specific optima. This supports the need for tuning rather than relying on defaults.

3. **System-level contribution beyond a simple algorithmic tweak.**  
   PLORA combines scheduling/planning with custom GPU kernels, which makes the work more than a standard HPO heuristic. The integration of offline packing and online execution is a coherent systems design.

4. **Reported gains are substantial on the evaluated setups.**  
   The paper reports up to 7.52× makespan reduction and up to 12.8× throughput improvement, with additional gains across multiple base models (Qwen-2.5 and LLaMA families) and tasks. If accurate, these are meaningful improvements.

5. **Attempts at formalization and analysis.**  
   The paper provides an optimization formulation and an approximation-style analysis for scheduling, which is a positive sign for a systems paper and helps explain the planner’s intent.

### Weaknesses
1. **The core novelty relative to prior systems work is somewhat unclear.**  
   The main idea—packing multiple small jobs to improve GPU utilization—feels closely related to existing work on concurrent serving, batching, and resource-aware scheduling. The paper argues that prior LoRA serving systems assume pre-trained adapters, but it is not fully convincing that “packing configurations during tuning” is a sufficiently distinct conceptual leap for ICLR unless the technical scheduling and kernel contributions are shown to be genuinely new and broadly useful.

2. **The empirical evaluation is narrow relative to the claims.**  
   The experiments are mostly on four tasks and a small set of base models, with a single hyperparameter search space of 120 configurations. This is useful, but ICLR typically expects stronger evidence of generality, especially for claims about a new tuning paradigm. It is not fully shown how robust the approach is across more tasks, larger search spaces, different adapter placement choices, or other fine-tuning regimes.

3. **The comparison baselines are limited.**  
   The main baselines are Min GPU and Max GPU, which are scheduling heuristics rather than strong state-of-the-art hyperparameter tuning systems. The paper compares against default LoRA settings, but it does not clearly compare end-to-end against modern HPO methods under equal time/compute budgets, which is important for an ICLR audience evaluating tuning methods.

4. **The scheduling objective may not map cleanly to the practical tuning goal.**  
   The planner optimizes throughput/makespan over the search space, but the paper does not convincingly connect this to best-found model quality under a fixed budget. In hyperparameter tuning, the critical metric is often “best validation performance under budget,” not just “all configurations finish faster.” The paper partly addresses this, but the decision criterion remains somewhat indirect.

5. **Reproducibility details are incomplete for a systems paper.**  
   While the paper gives model families, hardware, and some implementation notes, several important details are missing or underspecified: exact kernel launch/configuration choices, full planner settings, cost-model calibration procedure, and the complete set of hyperparameter values used per task/model. For an ICLR submission, this reduces reproducibility confidence.

6. **The paper’s presentation suggests possible technical inconsistencies.**  
   Even accounting for parser artifacts, some parts of the optimization derivation and algorithm descriptions are hard to follow and appear internally awkward. The proof/analysis is not presented in a way that makes it easy to assess correctness, and the relation between the ILP, the recursive decomposition, and the claimed approximation bound needs clearer exposition.

7. **Quality improvements are demonstrated, but not fully contextualized.**  
   The paper shows the best found LoRA adapter can outperform default settings, but it does not thoroughly analyze whether these gains come from the scheduling method itself or simply from exploring a larger search space more efficiently. It also does not discuss cost-performance tradeoffs relative to alternative budget allocations.

### Novelty & Significance
**Novelty: Moderate.** The combination of concurrent packing, scheduling, and packed LoRA kernels is a useful systems contribution, but the underlying idea is an extension of established resource-packing and batching principles rather than a fundamentally new learning algorithm. The novelty is stronger on the systems integration side than on the ML-methodology side.

**Significance: Moderately high if the reported speedups hold broadly.** For practitioners tuning many LoRA adapters, reducing makespan by several multiples could be very valuable. However, under ICLR’s standards, the contribution would be stronger if it demonstrated broader generality, stronger baselines, and clearer evidence that the method changes how hyperparameter tuning should be done rather than providing a specialized optimization for one setup.

**Clarity: متوسط.** The paper communicates the high-level idea well, but the formal sections, scheduling derivation, and kernel explanation are hard to parse and would benefit from more intuitive exposition.

**Reproducibility: Moderate to weak.** There is enough information to understand the prototype at a high level, but not enough to confidently reproduce the full planner, cost model calibration, and packed-kernel implementation.

### Suggestions for Improvement
1. **Add stronger HPO baselines under equal resource budgets.**  
   Compare against random search, Bayesian optimization, ASHA/Hyperband-style methods, and possibly an HPO system that is aware of runtime heterogeneity, using the same wall-clock budget and the same search space.

2. **Demonstrate broader generality.**  
   Evaluate on more downstream tasks, more model families, and more diverse search spaces; include scenarios where the best LoRA settings are not small-batch-friendly, to show when packing helps or hurts.

3. **Clarify the novelty relative to concurrent serving and batch scheduling.**  
   Explicitly state what is algorithmically new in PLORA versus reusing known packing/scheduling ideas, and isolate the contribution of the planner from the kernel implementation with more targeted ablations.

4. **Strengthen the quality-oriented analysis.**  
   Report best validation/accuracy versus total wall-clock time curves, not only final best adapter results. This would better show that faster tuning actually translates into better models under realistic budgets.

5. **Improve the reproducibility package.**  
   Provide pseudocode and/or code for the cost model calibration, exact planner behavior, kernel launch parameters, and the complete hyperparameter grids. A public artifact would be especially valuable for a systems-heavy ICLR submission.

6. **Rewrite the optimization and analysis sections for readability.**  
   The formal scheduling derivation should be simplified with a concrete toy example and a clearer explanation of the ILP, the recursive decomposition, and the approximation guarantee.

7. **Report more detailed ablations.**  
   Separate the gains from packing, the gains from custom kernels, and the gains from the planner across different model sizes and batch sizes, and include sensitivity to memory load factor and GPU topology.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add comparisons against real hyperparameter tuning baselines like random search, Bayesian optimization, Hyperband/ASHA, and Optuna under the same wall-clock budget. Without this, the claim that PLORA improves “hyperparameter tuning” is not convincing, because the paper mostly compares execution strategies rather than tuning methods.

2. Add an end-to-end comparison against prior LoRA training systems, especially mLoRA and any parallel fine-tuning baseline that can overlap multiple adapters or jobs. The current baselines are too weak; ICLR reviewers will see the speedup claims as potentially inflated unless you show PLORA beats the closest systems that already target LoRA training efficiency.

3. Add scaling experiments on the number of tuned configurations and number of GPUs, e.g., 24/48/120/240 configurations and 1/2/4/8 GPUs. The core claim is a scheduling system with makespan reduction, so it must show how gains change with search-space size and hardware scale, not just one fixed setting.

4. Add experiments on more tasks and more diverse tuning workloads beyond GLUE-style classification and GSM8K, including at least one long-context or generation-heavy task. The paper claims general LoRA hyperparameter tuning support, but the evidence is concentrated on a narrow benchmark set.

5. Add ablations for the packing planner separately from the custom kernels, cost model, and parallelism selection. Right now it is unclear how much of the improvement comes from scheduling versus kernel optimization versus simply packing more batches.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze whether the packed execution changes optimization dynamics or final model quality compared to training each configuration independently. Packing multiple adapters could change effective batch statistics, gradient noise, or convergence, and without this analysis the method’s correctness for hyperparameter search is not established.

2. Quantify cost-model accuracy versus actual runtime and memory usage across models, ranks, batch sizes, and parallelism degrees. The planner depends on these predictions, so the paper needs error statistics and failure cases; otherwise the near-optimal scheduling claim is not trustworthy.

3. Provide a rigorous analysis of the ILP/DTM approximation quality against the true optimum on small instances. The paper asserts NP-completeness and near-optimality, but without exact-optimal comparisons on small search spaces, the scheduling guarantee is not meaningful.

4. Analyze sensitivity to search-space composition, especially when configurations have highly skewed runtimes or memory footprints. ICLR reviewers will expect to know whether PLORA still helps when the candidate space is dominated by a few expensive configurations rather than many small ones.

5. Report overhead breakdowns for job planning, kernel compilation/setup, checkpoint management, and inter-job scheduling. If these overheads are nontrivial, they can materially weaken the makespan gains, especially for small search spaces.

### Visualizations & Case Studies
1. Add a timeline/Gantt-style visualization of packed job schedules showing GPU utilization over time versus Min GPU and Max GPU. This would directly reveal whether PLORA actually keeps hardware busy or whether the gain is driven by a few favorable configurations.

2. Add a scatter plot of predicted versus observed job runtimes and memory footprints for the planner’s cost model. This would expose whether the packing decisions are based on a reliable model or brittle estimates.

3. Add a per-configuration case study showing how PLORA groups specific LoRA settings into jobs and why. This would make it clear whether the planner is learning meaningful structure or just greedily packing arbitrary adapters.

4. Add convergence/quality curves for best-found adapter quality versus wall-clock tuning time. The central promise is faster hyperparameter tuning, so reviewers need to see whether PLORA reaches the same quality earlier, not just whether it eventually finds a good adapter.

### Obvious Next Steps
1. Extend the method to compare packed tuning against adaptive search methods that stop early on poor configurations. Without this, the paper does not establish that PLORA improves the full tuning pipeline rather than only the execution of exhaustive sweeps.

2. Support other common PEFT methods such as adapters and prefix tuning. The paper’s systems argument would be much stronger if the packing/scheduling idea generalized beyond LoRA.

3. Evaluate on non-English, instruction-tuning, and safety/alignment workloads where tuning behavior can differ substantially. A broader task suite is the most direct next step to support the claimed generality.

4. Demonstrate cross-hardware portability by testing on more GPU types and cluster topologies. Since the planner and kernels are hardware-sensitive, this is necessary before the method can be viewed as a broadly usable ICLR contribution.

# Final Consolidated Review
## Summary
This paper proposes PLORA, a system for accelerating LoRA hyperparameter tuning by packing multiple LoRA configurations into concurrent fine-tuning jobs and using custom kernels plus an offline planner to schedule them efficiently. The paper’s central claim is that standard LoRA tuning wastes GPU resources, and that exploiting this underutilization can substantially reduce end-to-end makespan while still finding better adapters than default LoRA settings.

## Strengths
- The paper identifies a real inefficiency in LoRA tuning: many candidate runs use tiny batch sizes and leave GPU compute/memory underutilized, and the empirical study with 1,000+ experiments does support that hyperparameters such as rank, batch size, learning rate, and alpha materially affect quality.
- The system design is coherent and technically nontrivial: PLORA combines an offline packing/scheduling planner with packed CUDA kernels, and the ablation showing that both the planning component and the custom kernels contribute to speedup is a meaningful piece of evidence.

## Weaknesses
- The evaluation is narrow relative to the ambition of the claim. The paper mostly studies four tasks and a fixed 120-configuration search space on a small set of Qwen/LLaMA models, so the evidence for “LoRA hyperparameter tuning” in general is thin. This matters because the gains are strongest in exactly the regime the authors chose: small batch sizes, underutilized GPUs, and a predetermined search space.
- The baselines are too weak for an ICLR paper on tuning systems. Comparing primarily against Min GPU and Max GPU says little about how PLORA stacks up against real hyperparameter optimization methods such as random search, Bayesian optimization, Hyperband/ASHA, or Optuna under a fixed wall-clock budget. This makes it hard to tell whether PLORA improves tuning itself or mainly improves the execution of an already-fixed sweep.
- The planner/cost-model story is undervalidated. The paper relies on a runtime and memory cost model after only a few initial iterations, but does not report prediction error, robustness across tasks, or failure cases. Since the scheduling method depends on these estimates, the near-optimality and makespan claims are less convincing than they should be.
- The formal scheduling section is difficult to trust as written. The optimization derivation, DTM recursion, and approximation argument are not presented cleanly enough in the main paper for a reader to verify correctness, and the guarantee seems limited to tail effects rather than a full scheduling optimality result. For a systems paper making algorithmic claims, that is too weakly communicated.
- The work does not convincingly separate the benefit of packing from the benefit of better search. It reports the best configuration found after searching, but does not show wall-clock curves of best-achieved quality versus time. That leaves open whether PLORA is actually better at finding strong adapters under budget, or simply faster at exhaustively evaluating a fixed grid.

## Nice-to-Haves
- A stronger ablation that compares PLORA’s packing planner against simple greedy or bin-packing heuristics would clarify whether the offline optimizer is truly needed.
- Additional scaling studies over more search-space sizes, GPU counts, and heterogeneous hardware would help establish when the method remains useful and when packing saturates.

## Novel Insights
The most interesting insight here is not just that LoRA is cheap, but that cheapness creates a systems opportunity: hyperparameter trials often become too small to fully use a GPU, so the right optimization target is not only fewer trials but better co-scheduling of trials. That reframes LoRA tuning as a packed-job scheduling problem rather than a sequence of independent runs, and the kernel work shows the authors understood that the bottleneck shifts from model math to adapter-level underutilization. The paper is strongest as a systems argument about reclaiming slack in LoRA tuning pipelines; it is much less convincing as a general hyperparameter-tuning method.

## Potentially Missed Related Work
- **mLoRA** — relevant because it also targets multi-GPU/parallel LoRA training and is closer to the training-side efficiency problem than serving-only systems.
- **Optuna / ASHA / Hyperband / Bayesian optimization systems** — relevant as stronger end-to-end hyperparameter tuning baselines under the same time budget.
- **Concurrent training / cluster scheduling work such as Gandiva, Heterogeneity-Aware Scheduling, and KungFu** — relevant because PLORA’s core idea is ultimately resource scheduling for deep learning jobs.
- **QLoRA** — relevant because the paper briefly mentions it and because quantization changes the memory/packing tradeoff directly.

## Suggestions
- Add end-to-end comparisons against standard HPO baselines under equal wall-clock budgets, and report best validation/accuracy versus time rather than only final best adapter quality.
- Report planner accuracy and robustness: predicted vs. actual runtime/memory scatter plots, plus failure cases where the cost model misestimates packing feasibility.
- Include a simpler packing baseline and a timeline/Gantt-style utilization plot to show whether PLORA’s gains come from the planner, the kernels, or just favorable packing.
- Expand evaluation to more tasks, more search-space sizes, and at least one harder generation-style workload to support the generality claim.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0]
Average score: 5.3
Binary outcome: Reject
