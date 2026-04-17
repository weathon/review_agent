I already have the full paper content in the user message. Let me carefully verify the key reviewer claims against the paper text I was given.

Let me check the specifics about the ablation comparisons mentioned in the paper:

Looking at Figure 8 description: "ODC consistently improves throughput over the collective baseline in both unpacked (LocalSort) and packed (LB-Micro, LB-Mini) settings" — this means they DO compare ODC+LocalSort vs Collective+LocalSort, which isolates the communication scheme effect.

Looking at Section 5.2: "All methods perform similarly when the minibatch size is one, since in this case ODC synchronizes after every sample, just like collective." This is a useful sanity check.

For RL: "ODC achieves up to 10% speedup over collective baseline" and "we did not do so, as the current solution is easier to integrate" — they acknowledge the constraint.

Let me also verify the convergence claim: "Additionally, we validate the correctness of ODC by verifying the training convergency in Appendix F." This is a brief reference, and the appendix is not included.

Now let me synthesize my review.

---

## Summary

This paper identifies that FSDP's per-layer collective communication (all-gather and reduce-scatter) creates synchronization barriers that hurt device utilization under the imbalanced workloads common in LLM post-training (where sequence lengths vary widely). The authors propose On-Demand Communication (ODC), which replaces collectives with point-to-point RDMA-based primitives within FSDP, effectively reframing it as a decentralized parameter server. This relaxes synchronization granularity from per-layer to per-minibatch, decouples device execution, and enables simpler minibatch-level load balancing (LB-Mini). ODC achieves up to 36% throughput improvement on SFT tasks and up to 10% on RL tasks over standard FSDP.

## Strengths

- **Clear and important problem identification.** The paper precisely identifies a concrete bottleneck: per-layer collective synchronization in FSDP amplifies straggler effects under imbalanced workloads. The formalization in Equation (1) cleanly captures the synchronization overhead, and Table 6 (appendix) reports up to 50% device idle time even with state-of-the-art packing. This is a timely and practically relevant concern.

- **Principled root-cause approach.** Rather than further refining packing strategies (which can only reduce, not eliminate, skew), ODC addresses the fundamental cause—synchronization barriers from collective communication. The insight of reframing FSDP as a decentralized parameter server by replacing collectives with point-to-point operations is elegant and well-argued.

- **Controlled comparison isolating the communication scheme.** The evaluation includes ODC+LocalSort vs. Collective+LocalSort comparisons (Figure 8), which does isolate the communication scheme effect from the load balancing algorithm. The parametric study systematically varies minibatch size, max length, packing ratio, and device count, providing useful insight into when ODC helps most.

- **Honest reporting of inter-node bandwidth limitation.** Section 5.4 explicitly shows ODC primitives have lower bandwidth than NCCL collectives at inter-node scale. Section 6.1 discusses this limitation and proposes mitigations. This is a real strength—the paper does not hide its weakness.

- **Practical engineering implementation.** The use of CUDA IPC and NVSHMEM via Triton-Distributed for non-intrusive RDMA transfers is nontrivial and demonstrates the approach is implementable. Open-sourcing the code supports reproducibility.

## Weaknesses

### Major:

- **Incomplete attribution of speedup sources.** While the paper does compare ODC+LocalSort vs. Collective+LocalSort (isolating the communication scheme), the headline results—especially the 36% speedup—come from ODC+LB-Mini, which conflates the communication scheme change with the new load balancing algorithm. The paper does not include a Collective+LB-Mini ablation (which is impossible since LB-Mini requires ODC's decoupled execution). This means the paper cannot quantify how much speedup comes from eliminating synchronization barriers vs. better load balancing. The paper references "bubble rate" data in Appendix G, but this critical evidence for the core claim is not in the main text. A quantitative decomposition of throughput gains (idle time reduction vs. load balancing improvement vs. communication cost change) would substantially strengthen the paper.

- **Limited multi-node evaluation given acknowledged inter-node communication penalty.** The paper explicitly shows in Section 5.4 that ODC's point-to-point primitives have significantly lower effective bandwidth than NCCL collectives for multi-node communication. Yet the main experimental results are presented on at most 32 GPUs (4 nodes). The parametric study in Figure 10 uses a golden setting of 8 GPUs (1 node). The discussion in Section 6.1 argues that compute-communication overlap hides ODC's inter-node overhead for long sequences, but this argument is qualitative—the paper does not empirically demonstrate that speedups persist in multi-node settings where inter-node traffic dominates, nor does it characterize the crossover point where ODC becomes worse than standard FSDP. For a paper claiming ODC is a "superior fit" for LLM post-training broadly, this gap is significant.

- **Convergence validation is deferred to an appendix without visible evidence.** The paper claims "identical training semantics" (Sections 1, 3, 3.2) and that ODC "does not alter training semantics." However, the implementation uses asynchronous RDMA-based gradient accumulation with a daemon, which changes the timing and ordering of gradient aggregation relative to FSDP's synchronous collectives. In floating-point arithmetic, different aggregation orders can produce different results. The only validation is "verifying the training convergency in Appendix F," which is referenced but not in the main text. For a method whose core novelty is altering synchronization semantics, convergence evidence belongs in the main text with loss curves or final metrics, not relegated to an appendix.

### Minor:

- **RL evaluation is constrained and yields only modest gains (up to 10%).** The verl framework requires identical samples per device, preventing LB-Mini from exercising its key advantage (per-device heterogeneous microbatch counts). Given that RL is prominently used as motivation, the evaluation does not fully test ODC's strengths in this setting. The authors acknowledge this but did not relax the constraint.

- **The claim that "collective communication fundamentally relies on balanced workloads" (Section 1) is somewhat overstated.** Collectives degrade *under* imbalance, but this is a quantitative, not categorical, property. Systems can mitigate straggler effects through packing, microbatch scheduling, or communication overlap. The paper's own results show that speedups diminish with higher packing ratios and larger minibatches (Figure 10), confirming that collective FSDP performs well when workloads are more balanced.

### Trivial:

- Minor missing details: which experimental configurations are intra-node vs. inter-node in the main results, training durations/warm-up for throughput measurements, and variance across runs.

## Nice-to-Haves

- A comparison with alternative synchronization-relaxation approaches (e.g., bounded-staleness PS, async SGD variants) would better situate ODC's tradeoffs, though this is beyond scope for the paper as submitted.
- Memory overhead quantification: ODC likely requires additional buffers for RDMA transfers and parameter caching; explicitly measuring and reporting peak memory overhead vs. FSDP would strengthen the practical assessment.
- Per-layer timeline visualizations with real trace data (as in Figure 1 but measured, not schematic) would directly demonstrate where ODC eliminates idle time.
- Larger-scale experiments (64+ GPUs, multi-node) would validate the claimed scalability and test whether inter-node communication costs erode the speedup.

## Removed Points

These points are flagged to be removed, treated with caution:

- **"Lack of comparison with a strong async/stale-synchronous PS baseline"**: The paper is about adapting PS ideas *into FSDP*, not about comparing against standalone PS implementations. Classic async PS systems (DistBelief, SSP) are fundamentally different from ODC, which preserves synchronous optimization semantics. This comparison would be a different research question.

- **"Incremental novelty of replacing collectives with point-to-point"**: While individual P2P primitives are not new, the specific integration into FSDP's sharding mechanism, the elimination of per-layer barriers while preserving synchronous semantics, and the interplay with minibatch-level load balancing constitute a genuine system contribution. The novelty claim is in the *combination and integration*, not in any single component.

- **"Not comparing against alternative approaches like async SGD or bounded staleness"**: The paper deliberately preserves synchronous semantics (Section 6.2 discusses relaxing this as future work). Comparing against fundamentally asynchronous methods would change the optimization algorithm, not just the communication pattern, making it an apples-to-oranges comparison.

- **"Insufficient evaluation at 64-256 GPU scale"**: While larger-scale experiments would be valuable, the paper is honest about inter-node limitations and proposes hybrid sharding as a mitigation. The 32-GPU experiments are a reasonable starting point, and the paper explicitly discusses the scaling tradeoff. This is a reasonable scope limitation, not a fatal flaw.

- **"Missing training quality/convergence metrics"**: While convergence validation should be in the main text (moved to Weaknesses above), the claim that "improvements in throughput don't necessarily translate to time-to-accuracy gains" misunderstands the paper. ODC explicitly preserves synchronous optimization semantics at the minibatch boundary—same gradients, same optimizer steps. If the semantics are truly preserved (which the paper should verify and show), throughput gains directly translate.

## Novel Insights

The paper's most novel insight is the reframing of FSDP as a decentralized parameter server, which reveals that the per-layer synchronization barriers in FSDP are an *artifact of the communication model*, not a *requirement of the training algorithm*. This perspective shift—from trying to pack workloads better to removing the synchronization that makes imbalance costly—is valuable and generalizable beyond FSDP to any sharded DP scheme. The minibatch-level load balancing insight (enabled by removing per-layer barriers) is a natural but non-obvious consequence: when devices don't need to synchronize per-layer, they also don't need the same number of microbatches, dramatically simplifying load balancing.

## Suggestions

1. **Bring convergence evidence into the main text**: Show at least one loss curve comparison (ODC vs. FSDP) for an SFT task and an RL task, plus a statement about numerical equivalence (or measured discrepancy) of accumulated gradients. This is essential for a paper whose core claim is preserved training semantics.

2. **Add a Collective+LB-Micro vs. ODC+LB-Micro speedup breakdown table**: Quantify how much of the gain comes from the communication scheme change alone, ideally with measured idle/bubble rates at each configuration.

3. **Report multi-node results explicitly**: At minimum, show whether the 16-GPU and 32-GPU results are 2-node and 4-node runs, and report the speedup for these configurations separately from single-node results. This directly addresses the inter-node concern.

4. **Quantify memory overhead**: Report peak memory usage for ODC vs. FSDP in the main results. Memory is the primary constraint in LLM training, and any overhead matters.

5. **Narrow the claims**: Instead of "ODC is a superior fit for the prevalent imbalanced workloads in LLM post-training," consider "ODC provides significant throughput improvements for imbalanced, long-context LLM post-training workloads, particularly within single-node or moderate-scale settings."

## Score and Decision

**Calibration anchor papers:**
- CO2 (ICLR spotlight): Communication overlap for distributed training, up to 128 GPUs, theoretical convergence analysis, scores 8/6/6/8/6/8 → avg ~7
- ZeRO++ (ICLR poster): Communication volume reduction in ZeRO, up to 384 GPUs, convergence verification, scores 8/5/5/6/6 → avg ~6
- Decentralized Training (ICLR poster): DeAdam optimizer, theoretical convergence, 64 GPUs, scores 6/8/6 → avg ~6.7
- ACCO (rejected): Delayed gradient overlap in ZeRO, limited novelty, convergence concerns, scores 5/6/5/6/3 → avg ~5
- DynPipe (rejected): Load balancing for dynamic LLMs, poor presentation, scores 5/3/3 → avg ~3.7

This paper has a clearer and more targeted problem than ACCO (which has prior art issues), more practical impact than the Decentralized Training paper (which has limited novelty claims), and better empirical results than DynPipe. It is comparable to ZeRO++ in spirit (practical systems optimization for FSDP/ZeRO). However, it has three substantive weaknesses: (1) the inter-node scaling concern is acknowledged but not empirically resolved, (2) the convergence evidence is deferred to an appendix, and (3) the speedup attribution between communication vs. load balancing is incomplete. These are not fatal but they do hold back the paper from the confidence level of CO2 or ZeRO++. The paper makes a genuine contribution—the PS reframing of FSDP is a real insight and the 36% speedup on SFT tasks is meaningful—but the evidence is more preliminary than conclusive.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>