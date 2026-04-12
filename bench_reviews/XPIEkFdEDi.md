## Summary
This paper proposes AnyBCQ, a multi-precision post-training quantization framework for LLMs built on binary-coded quantization (BCQ). The core idea is to share binary bit-planes across precisions while learning precision-specific scales, paired with a CUDA kernel that operates directly on bit-planes and avoids centroid lookup / bit-transpose overheads common in prior multi-precision non-uniform methods. Empirically, the method is strongest at 2-bit quantization, where it substantially improves over prior multi-precision baselines, while remaining competitive at 3–4 bits and offering favorable memory and throughput trade-offs.

## Strengths
- **A specific and meaningful low-bit result:** At 2 bits, the method delivers a large accuracy improvement over the main multi-precision baseline. On Llama-3.1-8B, Table 2 shows MMLU improving from **24.66** (Any-Precision LLM) to **35.32** and CSR average from **39.65** to **58.71**, which is a substantial gain in the regime where prior multi-precision methods struggle most.
- **Algorithm–kernel alignment is unusually tight:** The paper does more than propose a quantization format; it exploits BCQ’s binary bit-plane structure to design a kernel that directly computes on active bit-planes, avoiding the centroid lookup and bit-transposition path used by the compared non-uniform multi-precision approach. This is well motivated in Sec. 3.3 and supported by the latency breakdown in Appendix A.2, where the baseline kernel spends a large fraction of time in bit-transpose and lookup.
- **The memory-sharing mechanism is concrete and practically relevant:** Table 1 quantifies the cost of supporting 2/3/4-bit operation in one model: compared with storing separate models, the proposed shared-binary representation reduces total footprint from **9.85 GB** to **4.99 GB** on Llama-3.1-8B. This is a specific deployment advantage of the paper’s design rather than a generic compression claim.
- **The paper is appropriately transparent about the central trade-off:** The authors explicitly acknowledge that sharing binaries across precisions can hurt peak higher-bit accuracy (“the shared-binary constraint slightly limits the capacity of the multi-precision model,” Appendix A.1; also Sec. 7). This transparency increases confidence that the reported gains at 2 bits are not being oversold as universally dominant.
- **Evaluation goes beyond isolated kernel timing:** In addition to benchmark accuracy, the paper reports end-to-end decoding throughput on multiple models (Llama-3.1-8B, Gemma-2-9B, Phi-4-14B), plus a mixed-precision decoding case study. This helps support the practical systems motivation.

## Weaknesses

###: Fatal
None.

### Major:
- **The shared-binary design imposes a real accuracy ceiling at higher precisions, and this limits the paper’s “flexible multi-precision” story.**  
  This criticism is directly supported by the paper’s own results and discussion. Table 2 shows the multi-precision model trailing both its fixed-precision counterpart and sometimes the non-uniform multi-precision baseline at 3–4 bits; Appendix A.1 further confirms a consistent perplexity gap versus fixed-precision AnyBCQ at 3 and 4 bits. The paper itself explains this: “the additional shared-binary constraint slightly limits the capacity of the multi-precision model.” This does not invalidate the paper, but it does mean the method is best viewed as a strong low-bit / deployment-efficient compromise, not as uniformly best across the full 2/3/4-bit range.
- **Accuracy evaluation across architectures is narrower than the systems claims.**  
  The main benchmark accuracy table is only for **Llama-3.1-8B**. Gemma-2-9B and Phi-4-14B appear in the end-to-end evaluation, but there the comparison is limited to Wiki perplexity, MMLU, and throughput against Any-Precision LLM, rather than the broader task suite used for the main claim. Given that the paper emphasizes a generally applicable multi-precision framework for LLM deployment, more complete cross-architecture accuracy validation would strengthen confidence that the 2-bit gains and 3–4 bit trade-offs are not overly model-specific.
- **The “negligible overhead” claim for dynamic per-request precision selection is only partially substantiated.**  
  The kernel design is convincing at the static GEMV level, and the throughput results are encouraging, but the paper does not isolate the runtime overhead of actually switching precision across requests or during continuous autoregressive serving. Since the claim is specifically about dynamic per-request selection, a targeted experiment measuring the cost of switching precision policies in a live decoding loop would better support that statement.

### Minor
- **Calibration robustness is not analyzed.**  
  The method uses 512 C4 sequences for reconstruction-error optimization (Sec. 4.1), but the paper does not provide sensitivity analyses over calibration set size or distribution. This is not unusual for PTQ papers, and the current evidence is sufficient to show the method works, but such an ablation would help determine whether the particularly strong 2-bit results are stable or calibration-sensitive.
- **The paper remains largely empirical, with limited analytical insight into progressive precision expansion.**  
  This is acknowledged by the authors in Sec. 6: “the present work remains largely empirical and lacks theoretical guarantees.” A stronger empirical diagnostic analysis of how freezing lower-bit binaries affects reconstruction error layer-by-layer or across expansion stages would improve technical understanding, even if formal theory is beyond scope.
- **The mixed-precision case study is directionally useful but does not fully establish practical viability at aggressive average bitwidths.**  
  Table 5 shows AnyBCQ outperforming Any-Precision LLM at equal average precision, but both methods degrade substantially at low average bitwidths (e.g., 2.23 bits). This supports relative superiority, but not yet a compelling practical mixed-precision operating point in the most aggressive regime.

### Trivial
- **Hardware characterization in Appendix A.4 is coarse.**  
  The appendix uses `nvidia-smi` polling for utilization/power characterization, which is acceptable as a rough signal but not a rigorous microarchitectural analysis. This does not affect the main throughput results, but those appendix-level conclusions should be interpreted cautiously.

## Nice-to-Haves
- Add an ablation on calibration data size and distribution (e.g., 512 vs. 2k/4k samples).
- Quantify the cost of the shared-binary constraint more directly, ideally layer-wise or by reporting the delta versus independently optimized fixed-precision models across layers.
- Measure dynamic precision switching overhead during actual autoregressive decoding rather than only static kernel benchmarks.
- Expand the broader accuracy suite beyond Llama-3.1-8B to at least one additional architecture.
- Include profiler-based kernel analysis (e.g., Nsight) for a stronger systems account of memory stalls / utilization.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the 3× speedup over FP16 is “theoretically implausible” or unsupported because weight-only quantization cannot deliver such gains.**  
  This is too strong and not justified from the paper alone. The paper reports kernel-level and end-to-end throughput numbers on specific workloads; without external benchmarking evidence, it is not appropriate to dismiss them as implausible. The fair retained criticism is narrower: comparisons are limited to the baselines included by the paper, and dynamic-switching overhead is not isolated.
- **Criticism that bit-transpose overhead is merely an artifact of one implementation and therefore the motivation is invalid.**  
  The paper does not claim all non-uniform quantization must incur identical overheads in all imaginable kernels; it argues this overhead is present in the compared prior multi-precision approach and motivates a BCQ-friendly alternative. That is a reasonable claim supported by the presented comparison.
- **Unfair baseline complaints based on missing external methods or calibration mismatches not established in the paper.**  
  Since external baselines and their exact settings cannot be verified here, these criticisms are too speculative. The paper already compares against AWQ, Any-Precision LLM, and ShiftAddLLM, which is a meaningful set for its stated goal.
- **Complaint about missing activation quantization analysis.**  
  The paper is explicitly a **weight-only PTQ** method. Requiring activation quantization is outside the paper’s stated scope.
- **Complaint about unclear total “training time” or comparison to QAT efficiency.**  
  The method is PTQ, and the paper does provide its optimization setup. A detailed wall-clock comparison against QAT would be nice but is not essential to evaluate the core contribution.
- **Formatting/parsing issues in figures/tables.**  
  These are extraction artifacts and not paper weaknesses.

## Novel Insights
The most interesting synthesis across the reviews is that this paper’s real contribution is not “best multi-precision quantization” in the abstract, but a more specific operating point: it identifies BCQ as a particularly strong substrate for **hardware-efficient multi-precision inference when 2-bit capability genuinely matters**. The results suggest a useful deployment niche that prior non-uniform multi-precision methods do not serve well: a single model spanning an aggressive low-bit operating mode with meaningful quality, while preserving acceptable 3–4 bit quality and enabling cleaner bit-plane execution. The flip side is equally important: the same shared-binary mechanism that enables low-overhead multi-precision serving is also the source of its higher-bit ceiling. That trade-off is the central technical reality of the paper.

## Suggestions
- Strengthen the paper’s positioning: present AnyBCQ less as a universally superior multi-precision method and more as a deployment-oriented design that prioritizes **2-bit viability + hardware efficiency**, with explicit higher-bit trade-offs.
- Add a focused ablation quantifying the penalty from freezing lower-bit binaries, ideally per layer and per target precision.
- Include a calibration-size sensitivity study to show the robustness of the unusually strong 2-bit results.
- Directly measure the overhead of switching precisions across requests or tokens in a realistic decoding loop.
- Expand full-task accuracy evaluation to at least one additional architecture beyond Llama-3.1-8B.
- If space permits, add profiler-backed kernel analysis to complement the current latency tables and appendix power/utilization measurements.