=== CALIBRATION EXAMPLE 79 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- Does the title accurately reflect the contribution?
  - Largely yes. The paper is about an alternative efficient kernel implementation for Native Sparse Attention, and “Flash Sparse Attention (FSA)” captures that.
  - That said, the title slightly overstates novelty by calling it “an alternative efficient implementation” without signaling that the core idea is mainly a loop-order inversion plus kernel-specific optimizations rather than a new sparse-attention algorithm.

- Does the abstract clearly state the problem, method, and key results?
  - Yes, the abstract does identify the practical problem: NSA’s kernel is efficient only for relatively large numbers of query heads per GQA group, which mismatches many deployed LLMs.
  - It also states the main method: FSA reorders the kernel loop structure to better match common GQA settings.
  - The reported results are clear and quantitative.

- Are any claims in the abstract unsupported by the paper?
  - The abstract claims FSA “enables efficient NSA computation across a wide range of popular LLMs with a varied, smaller number of heads in each GQA group.” The experiments do show gains for several models and GQA settings, but they are limited to specific architectures, specific NSA hyperparameters, and mostly long-sequence regimes. “Wide range” is plausible but somewhat broader than what is directly established.
  - The claim that “full attention” comparison shows the performance boost is “further amplified” is supported in the sense of the measured speedups, but the paper should be careful that this is not the main scientific contribution; the relevant comparison is really against NSA.

### Introduction & Motivation
- Is the problem well-motivated? Is the gap in prior work clearly identified?
  - Yes. The introduction gives a concrete systems bottleneck: NSA’s native kernel performs poorly when the GQA group size is small, because the loop order creates padding and underutilization on modern GPUs.
  - The gap is specific and meaningful for ICLR: algorithmic sparsity alone is not enough; kernel design determines whether the theoretical savings translate into wall-clock gains.

- Are the contributions clearly stated and accurate?
  - Mostly yes. The three contributions are sensible and aligned with the paper.
  - However, Contribution 1 is a bit stronger than the evidence in the paper: the kernel is not simply “more efficient” in all settings; later results show the benefit depends on GQA, sequence length, and block size.
  - Contribution 2 is accurate, but the paper does not fully quantify which optimization accounts for what fraction of the gains beyond the ablation in Figure 9.

- Does the introduction over-claim or under-sell?
  - It slightly over-claims generality. The paper frames FSA as broadly applicable to “a wide range of current LLMs,” but the evaluated models are a small set of long-context LLMs with specific head configurations.
  - At the same time, it under-emphasizes a key limitation early: FSA introduces extra intermediate buffers and a profiling step, which matter for memory footprint and deployment complexity.

### Method / Approach
- Is the method clearly described and reproducible?
  - The high-level idea is clear: invert NSA’s loop order, use index tensors for sparse query batching, and separate reduction/softmax into dedicated kernels.
  - The details are less clean than they should be for reproducibility. The paper gives useful intuition, but the exact dataflow of the forward and backward passes, the precise role of buffers like `O_buf`, and how online softmax statistics are computed and reused are not fully specified in a way that would let an independent implementer recreate the kernel from the text alone.
  - Appendix C helps, but some key implementation decisions are still only described qualitatively.

- Are key assumptions stated and justified?
  - Some are, some are not.
  - The central assumption is that the number of query tokens attending to a KV block is usually large enough to avoid padding under FSA. This is plausible and empirically supported for the tested regimes, but the paper does not characterize when this assumption fails.
  - The theoretical analysis in Appendix E assumes uniform token selection probability across KV blocks, which is unlikely to reflect real NSA behavior, especially because sparse selection is learned and may be highly skewed. That assumption weakens the theoretical memory/FLOP estimates.

- Are there logical gaps in the derivation or reasoning?
  - Yes, mainly around the theoretical analysis and the claimed superiority of FSA.
  - The theorem in §3.3 appears to compare aggregate memory access and FLOPs, but the derivation rests on simplified probabilistic assumptions and does not rigorously account for:
    - non-contiguous access overhead,
    - extra kernel launch overhead,
    - the cost of the online profiling module,
    - intermediate buffer traffic,
    - and the fact that FSA’s reduction is split across kernels.
  - In Appendix E, the comparison between FSA and NSA is useful but not a proof of runtime superiority. The paper is careful to say performance is empirical, but the “theorem” wording may suggest stronger formal guarantees than are actually established.

- Are there edge cases or failure modes not discussed?
  - Yes.
  - The most important is memory overhead. Appendix E and Table 5 show intermediate buffers can reach 12.36 GB at 256K sequence length for `(BK,T)=(64,16)`, which is substantial. The paper says this is manageable on H200, but for multi-GPU training, concurrent buffers, optimizer states, activations, and model weights matter; the analysis is incomplete.
  - Another edge case is when GQA groups have 8 heads or more. The paper itself shows the gains shrink substantially there, and in some settings FSA is only marginally better than NSA.
  - The “attention sink” discussion is useful, but it also suggests FSA may require extra special-casing for highly skewed sparse patterns.

- For theoretical claims: are proofs correct and complete?
  - There is no full proof, only an informal theorem and a memory/FLOP accounting in Appendix E.
  - The accounting is not fully rigorous because it relies on simplifying assumptions and omits several system-level costs. So the theoretical section should be viewed as intuition rather than a proof.

### Experiments & Results
- Do the experiments actually test the paper's claims?
  - Yes, mostly. The main claim is that FSA improves NSA kernel efficiency and downstream training/prefill performance for small GQA groups. The experiments directly test:
    - kernel latency,
    - end-to-end training latency,
    - inference prefill latency,
    - forward/backward breakdown,
    - ablations,
    - and some larger-context extensions.
  - This is well aligned with the paper’s contribution.

- Are baselines appropriate and fairly compared?
  - NSA and full attention are the right baselines for the stated claims.
  - A concern is that the comparison to NSA depends on the specific Triton-based NSA implementation referenced as “Organization, 2024.” The paper should clarify whether NSA and FSA were implemented with comparable effort and whether both were equally optimized.
  - For end-to-end comparisons, FlashAttention is a reasonable reference point, but full attention is not really the right baseline for all of the prefill experiments once the paper’s main claim is about improving NSA. It is still useful as context.

- Are there missing ablations that would materially change conclusions?
  - Yes.
  - A very important missing ablation is the separate impact of each major FSA component:
    - loop-order inversion,
    - early-return query filtering,
    - separate reduction kernel,
    - separate online softmax kernel,
    - profiling fallback.
  - Figure 9 only mentions disabling inner loop and early return. That leaves unclear how much each of the other architectural changes contributes.
  - It would also be valuable to isolate the cost of the extra buffers, since Table 5 suggests memory overhead is nontrivial.

- Are error bars / statistical significance reported?
  - No. The paper reports single latency numbers and speedups, but no variance, repeated-run confidence intervals, or significance estimates.
  - For systems papers, exact timing is common, but given that some reported gains are modest (e.g., around 1.05×–1.11×), some measure of variability would materially strengthen the claims.

- Do the results support the claims made, or are they cherry-picked?
  - The results broadly support the claims, but there is some cherry-picking in presentation.
  - The paper emphasizes best-case speedups such as 3.5× kernel-level, 1.25× end-to-end training, and 1.36× prefill. Those are credible, but the averages and marginal cases show that gains are sometimes small.
  - Several tables in the appendix show configurations where FSA and NSA are nearly tied, especially for GQA=8. This should be highlighted more explicitly in the main text to better calibrate the contribution.

- Are datasets and evaluation metrics appropriate?
  - For a systems paper on attention kernels, yes. Kernel latency, training latency, and prefill latency are appropriate metrics.
  - The model set is relevant and includes modern long-context LLMs.
  - One weakness is that the paper does not evaluate end-task quality at scale on the same full-size models used for performance tests; instead, correctness and accuracy preservation are shown on smaller fine-tuning setups in the appendix. That is acceptable for a systems paper, but the main text could more clearly state that FSA is intended to be function-preserving rather than improving model quality.

### Writing & Clarity
- Are there sections that are confusing or poorly explained?
  - Yes, mainly the method section and the theoretical analysis.
  - The core kernel design is understandable at a high level, but the execution model becomes hard to follow when the paper introduces multiple index tensors, partial outputs, separate reductions, and online softmax buffers. This is especially true in §3.2.
  - The relationship between the token selection kernel, online softmax kernel, and reduction kernel could be explained more cleanly with a step-by-step execution trace for one query head and one KV block.
  - §3.3 and Appendix E are especially hard to parse because the formulas are embedded in prose and the cost accounting is not introduced systematically.

- Are figures and tables clear and informative?
  - The intended figures/tables seem useful, especially Figures 4–10 and Tables 5–15, but the parser-extracted version here is hard to read. Ignoring extraction artifacts, the paper likely benefits from those plots.
  - Still, the paper should ensure that the main figures clearly indicate:
    - which exact configurations are being shown,
    - whether speedups are relative to NSA or full attention,
    - and whether the plotted numbers are per-kernel or end-to-end.
  - Figure 9’s ablation is important but could be more granular.

### Limitations & Broader Impact
- Do the authors acknowledge the key limitations?
  - Partially.
  - The paper acknowledges non-contiguous memory access and buffer overhead, and it notes that FSA may fall back to NSA through profiling.
  - It also shows in the appendix that some gains diminish when GQA group size is large.

- Are there fundamental limitations they missed?
  - Yes.
  - The biggest missed limitation is that FSA’s benefit depends strongly on the head/group configuration and the sparsity pattern induced by NSA. This makes it less universal than the abstract implies.
  - The paper also does not sufficiently discuss engineering complexity: separate kernels, online profiling, extra buffers, and fallback logic substantially increase implementation and deployment burden.
  - Another limitation is that the memory overhead at ultra-long contexts is large enough to matter operationally, even if it fits on the tested GPUs.

- Are there failure modes or negative societal impacts not discussed?
  - No meaningful societal-ethics issue is apparent; this is primarily a systems/kernel paper.
  - The main “impact” is practical: faster long-context inference and training could lower compute costs, but it could also accelerate deployment of large models without addressing broader safety concerns. That said, this is standard and not a major concern for the paper.

### Overall Assessment
FSA addresses a real and important systems bottleneck: NSA’s kernel implementation does not map well to many modern LLM head configurations, so algorithmic sparsity alone does not guarantee speedups. The paper’s main idea—reversing the loop order and adding targeted kernel optimizations—is sensible, and the empirical results do show meaningful gains in the settings studied. That said, the contribution is narrower than the strongest framing suggests, because the benefits depend on specific GQA settings, sequence lengths, and hyperparameters, while the method introduces extra complexity, nontrivial buffer overhead, and a profiling fallback. For ICLR, this is a solid systems paper with practical value, but it would be stronger with a more rigorous and transparent treatment of trade-offs, clearer method specification, and more complete ablations and variability reporting.

# Neutral Reviewer
## Balanced Review

### Summary
This paper identifies an implementation bottleneck in Native Sparse Attention (NSA): the original kernel loop ordering is efficient mainly when each GQA group has many query heads, which is often not true in contemporary LLMs. The authors propose Flash Sparse Attention (FSA), which swaps the loop order, adds optimized handling for irregular query access and reduction, and claims broader efficiency across common GQA settings on Hopper-class GPUs.

### Strengths
1. **Clear systems motivation tied to real LLM configurations.**  
   The paper makes a compelling case that NSA’s kernel design is misaligned with widely used GQA settings in modern LLMs, especially small query-head counts per KV group. This is an important practical issue for ICLR, where methods are expected to matter beyond a narrow benchmark regime.

2. **Concrete kernel-level optimization contribution.**  
   FSA is not just a benchmark tweak; it proposes a specific alternative implementation strategy: invert the loop order, minimize padding, and introduce separate selection/reduction/online-softmax kernels. The paper also discusses memory-access trade-offs and buffer management, which suggests genuine systems design effort.

3. **Broad empirical evaluation across GPUs, models, and sequence lengths.**  
   The paper reports kernel latency, end-to-end training latency, inference prefill latency, distributed inference, ultra-long contexts, and ablations on two GPU generations (H20/H200) and multiple LLMs. This breadth is stronger than many kernel papers and is aligned with ICLR’s interest in practical impact.

4. **Evidence of meaningful speedups in the targeted regime.**  
   The reported gains are substantial in several settings: up to 3.5× kernel-level latency reduction over NSA, and consistent improvements in training/prefill. The benefit is especially clear when GQA group size is small, which supports the central claim that the original NSA implementation is underutilized in common settings.

5. **Attempts to validate correctness, not just speed.**  
   The appendix includes loss comparisons and additional downstream-style evaluations (PPL/F1 on small models), which is a positive sign for a systems paper that changes the attention implementation. This is important because ICLR reviewers often worry about whether speedups come at the cost of correctness or model quality.

### Weaknesses
1. **Novelty is primarily implementation-level, and the paper’s conceptual advance may be limited.**  
   The core idea is a loop reordering plus supporting kernels for reduction and online softmax. This is useful, but the paper does not convincingly establish a fundamentally new sparse attention algorithm or a broadly general kernel abstraction. Against ICLR’s acceptance bar, the contribution may be viewed as an engineering refinement of NSA rather than a major methodological advance.

2. **The performance claim depends heavily on a specific operating regime.**  
   The strongest speedups occur when GQA group size is small and in long-context settings. The paper itself notes that NSA can be competitive or even better in some cases, and FSA introduces extra non-contiguous memory access plus additional kernels. This makes the improvement somewhat conditional rather than universally dominant.

3. **Reproducibility is only moderately strong.**  
   The paper says the code is open-sourced, which is good, but many experimental details remain incomplete or hard to verify from the text alone: profiling procedure, exact kernel tuning choices, random seeds, training hyperparameters, and whether all baselines were equally optimized on both GPU types. The reliance on Triton kernels and online profiling also makes exact reproduction more delicate than a standard model-level paper.

4. **Theoretical analysis is helpful but not fully rigorous.**  
   The memory/FLOP analysis provides intuition, but several assumptions are simplifying and likely strong, such as uniform KV-block selection probability. That makes the theorem more of a comparative heuristic than a tight proof of superiority. ICLR typically values solid empirical evidence, but the analysis could be more rigorous and better connected to actual distributions.

5. **Evaluation omits some important comparisons and stress tests.**  
   The paper compares mainly against vanilla NSA and full attention, but not against other recent sparse or long-context inference/training methods beyond brief contextual references. Since ICLR expects positioning against the most relevant alternatives, this narrower baseline set weakens the claim of broad practical superiority.

6. **Clarity is uneven in places.**  
   The main idea is understandable, but some parts of the method description are difficult to follow without the figures. In particular, the separation between token-selection, online-softmax, and reduction kernels, and how correctness is preserved across thread blocks, could be explained more cleanly. For an ICLR audience, the implementation story should be easier to parse.

### Novelty & Significance
**Novelty:** Moderate. The main novelty lies in a practical reimplementation of NSA with a reversed loop order and systems optimizations for GQA regimes that are common in current LLMs. This is a meaningful engineering contribution, but not a large conceptual leap beyond prior sparse-attention kernel work.

**Clarity:** Moderate. The high-level motivation is clear, but the low-level kernel design and correctness story require careful reading and would benefit from cleaner exposition and a more systematic algorithm description.

**Reproducibility:** Moderate to good. Open-source code and extensive benchmarks are positives, but the paper would need more granular experimental detail to make reproduction straightforward.

**Significance:** Moderate. If the claims hold, FSA could matter in practice because kernel efficiency is a major bottleneck for long-context LLMs. However, the gains are most compelling in a subset of settings, so the overall significance is more practical than conceptual.

### Suggestions for Improvement
1. **Provide a more rigorous and transparent algorithm description.**  
   Add pseudocode for the forward and backward passes, explicitly showing how partial results, online softmax statistics, and reductions interact across thread blocks.

2. **Strengthen the comparison against broader baselines.**  
   Include more recent sparse/efficient attention systems and, where appropriate, compare against optimized implementations under the same hardware/compiler stack to better support the practical relevance claim.

3. **Clarify the regime where FSA is expected to win.**  
   Summarize a simple decision rule or cost model that predicts when FSA should outperform NSA versus when it may not, especially as a function of GQA size, block size, and sequence length.

4. **Expand reproducibility details.**  
   Report exact training hyperparameters, profiling/search settings, compiler versions, kernel launch configurations, and random seeds. This is especially important for a kernel paper where results can be sensitive to tuning.

5. **Tighten the theoretical analysis.**  
   State the assumptions more explicitly and, if possible, analyze non-uniform token-selection patterns rather than only uniform approximations. A clearer link between the model and observed measurements would improve credibility.

6. **Discuss memory overhead and deployment trade-offs more concretely.**  
   Since FSA introduces intermediate buffers and online profiling, include a concise deployment-cost analysis showing when the memory/compile overhead is worthwhile relative to the latency gains.

7. **Improve presentation of correctness validation.**  
   The appendix results are promising, but the main paper should summarize whether FSA matches NSA’s quality within statistical variation on downstream tasks, not just training loss curves.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add comparisons against more strong sparse-attention kernels beyond NSA and FlashAttention, especially recent long-context systems like Quest, FlexPrefill, DuoAttention, H2O-style baselines, and vendor-optimized sparse kernels. At ICLR, claiming a broadly useful systems advance requires showing FSA is not just better than one prior implementation, but competitive with the current sparse-attention landscape.

2. Add a direct experiment isolating the loop-order claim by implementing an NSA variant that only swaps loop order while keeping all other optimizations fixed. Without this, it is unclear whether the gains come from the proposed kernel structure or from confounded changes like buffer handling, index compaction, or warp allocation.

3. Add end-to-end quality evaluation on standard long-context benchmarks for the actual models used in system tests, not just loss/perplexity on small fine-tunes. ICLR expects evidence that a systems change preserves task-level behavior; otherwise the paper only shows speed, not that the sparse attention integration is viable in practice.

4. Add a head-to-head comparison of FSA vs. NSA across a wider range of GQA ratios and model architectures actually used in modern LLMs, including cases where GQA group size is not a power-of-two or where query heads per group are larger than 8. The paper’s main claim is broad applicability under “popular LLMs,” so the current evaluation range is too narrow to justify that generalization.

5. Add memory footprint and throughput experiments under realistic deployment constraints, including peak HBM use during full training/inference pipelines and not just isolated kernel buffers. The paper’s buffer overhead claim is central to feasibility, and without full-system memory accounting the practical advantage of FSA is not established.

### Deeper Analysis Needed (top 3-5 only)
1. Add a quantitative breakdown of where FSA wins or loses: padding waste, non-contiguous access penalty, extra kernel launch overhead, reduction cost, and online-softmax cost. The current argument is qualitative, so the reader cannot tell why the speedups hold in some settings but shrink or disappear in others.

2. Add analysis of performance sensitivity to top-k, block size, sequence length, and GQA group size with a model that explains the crossover points. ICLR reviewers will expect a principled explanation of when FSA should beat NSA and when it should not, not only empirical tables.

3. Add correctness analysis beyond matching loss curves, including numerical error in outputs/gradients relative to a reference implementation and stability across long sequences. The proposed reordering and multi-kernel reduction introduce nontrivial numerical risk, so “similar loss” is not enough to trust correctness.

4. Add a more careful analysis of the claimed memory trade-off for intermediate buffers, especially under the attention-sink case and ultra-long contexts. The paper admits buffer growth can be substantial, so it needs a clear boundary on when the extra HBM cost becomes unacceptable.

5. Add analysis of how much the online profiling fallback changes real latency and whether it causes instability across runs/hardware. Since runtime profiling is part of the design, its overhead and robustness are part of the method, not an implementation detail.

### Visualizations & Case Studies
1. Show per-kernel roofline-style or bandwidth-vs-compute plots for NSA and FSA. This would reveal whether FSA is actually moving the bottleneck from memory-bound padding to a better-balanced regime, or simply trading one inefficiency for another.

2. Show a case study of one representative sequence with the exact query-token/KV-block access pattern, including padding amount, valid-token count per block, and buffer utilization. This would make the core loop-order argument testable instead of abstract.

3. Show a failure-mode visualization where FSA does not outperform NSA, e.g., large GQA groups or very small sequence lengths. ICLR reviewers will want to know the method’s limits, not only best-case examples.

4. Show memory timeline plots during end-to-end training/inference, including attention buffers and peak HBM usage. This would expose whether the claimed efficiency survives realistic runtime pressure and whether the intermediate buffers create hidden bottlenecks.

### Obvious Next Steps
1. Package FSA as a drop-in replacement in a widely used attention stack and benchmark it on full model training/inference pipelines with standard recipes. That is the most direct way to show the method is practically useful beyond a custom benchmark setup.

2. Extend the method to additional sparse-attention patterns and compare whether the same loop inversion generalizes. The paper currently looks specific to NSA, so generality is still unproven.

3. Release a reproducible benchmark suite that tests multiple GPUs, model sizes, and sparsity regimes with fixed seeds and configs. For an ICLR systems paper, reproducibility and robustness across settings materially affect credibility.

# Final Consolidated Review
## Summary
This paper proposes Flash Sparse Attention (FSA), an alternative Triton-based kernel implementation for Native Sparse Attention (NSA) that swaps the loop order to better fit modern GQA settings with fewer query heads per group. The core claim is that NSA’s existing kernel leaves substantial performance on the table in common LLM configurations, and that FSA recovers this by reducing padding and improving hardware utilization, with additional kernels for reduction and online softmax.

## Strengths
- The paper targets a real systems bottleneck: the native NSA kernel is indeed sensitive to GQA group size, and the reported experiments consistently show that FSA helps most in the common small-GQA regime. This is a meaningful practical issue for long-context LLMs.
- The empirical evaluation is broad for a kernel paper, covering kernel latency, end-to-end training, prefill inference, ultra-long contexts, distributed inference, and ablations on two GPU generations. The reported gains are nontrivial in the intended regime, with up to 3.5× kernel-level speedup over NSA and consistent end-to-end improvements.
- The appendix does include correctness-oriented checks, including loss curves and small downstream-style evaluations, which is better than many systems papers that only report speed.

## Weaknesses
- The main novelty is still fairly limited: this is an engineering reimplementation of NSA centered on loop-order inversion plus supporting kernels, not a new sparse-attention method. The paper’s framing is stronger than the conceptual leap actually delivered.
- The performance advantage is conditional, not universal. The gains shrink substantially as GQA group size increases, and the paper itself shows cases where FSA is only marginally better than NSA. This makes the “wide range of popular LLMs” claim feel broader than the evidence supports.
- The theoretical analysis is only a rough accounting, not a rigorous guarantee. It relies on simplifying assumptions about token-selection behavior and does not fully model non-contiguous access cost, extra kernel launches, profiling overhead, or the added buffer traffic. The “theorem” reads more like intuition than proof.
- FSA introduces real deployment complexity and memory overhead. The intermediate buffers can become large at ultra-long contexts, and the paper relies on a profiling fallback plus multiple specialized kernels. That is acceptable engineering, but it is not free and should be presented more soberly.
- The ablation story is incomplete. Figure 9 only isolates a subset of the design choices, so it is still unclear how much each major component contributes: loop-order inversion, separate reduction, separate online-softmax handling, early return, and profiling-based fallback.

## Nice-to-Haves
- A clearer pseudocode-level description of the forward and backward passes, especially how partial outputs, online softmax statistics, and reduction interact across thread blocks.
- More explicit reporting of the profiling procedure, compiler/runtime settings, and tuning protocol so the results are easier to reproduce.
- A simple decision rule or cost model describing when FSA should beat NSA and when NSA remains preferable.

## Novel Insights
The most interesting insight is that sparse attention speedups are not just a matter of reducing FLOPs; they can be dominated by whether the kernel’s loop order matches the head/group structure of real LLMs. The paper’s main contribution is therefore a systems-level observation: for NSA-style sparsity, the “right” implementation depends strongly on how many query heads are packed into each GQA group, and a kernel that is theoretically sparse can still lose to a less sparse alternative if it is forced into padding-heavy execution. FSA’s value lies in making that mismatch concrete and showing that a loop inversion can recover meaningful speedups, but the paper also demonstrates that this is a regime-specific optimization with noticeable memory and complexity trade-offs.

## Potentially Missed Related Work
- Quest — relevant as a recent sparse-attention inference method for long-context LLMs.
- FlexPrefill — relevant as another long-context prefill acceleration method.
- DuoAttention — relevant because it targets efficient long-context inference with structured head specialization.
- H2O / heavy-hitter style sparse inference methods — relevant as strong long-context baselines that the paper does not compare against directly.
- Vendor-optimized sparse attention kernels — relevant for a fairer systems comparison beyond NSA and FlashAttention.

## Suggestions
- Add a controlled ablation where only the loop order is changed while all other kernel optimizations are held fixed, to isolate the central claim.
- Expand the comparison set to include stronger recent sparse/efficient attention baselines, at least in a subset of representative settings.
- Add a concise memory timeline and peak-HBM analysis for end-to-end runs, not just buffer-size estimates.
- Summarize the correctness evidence in the main paper with a clearer statement of numerical fidelity, not just loss curves in the appendix.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0]
Average score: 7.3
Binary outcome: Accept
