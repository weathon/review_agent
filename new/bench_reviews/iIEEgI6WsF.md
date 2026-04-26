Now I have a thorough understanding of the paper. Let me carefully evaluate each point from the harsh critic and synthesize the final review.

Key observations from reading the paper:

1. **Multi-node scalability**: The paper's main results (Figures 8-9) cover models from 1.5B to 32B with up to 32 GPUs. The parametric study's golden setting is 1.5B/8 GPUs. The paper is transparent about inter-node overhead (Figure 11, Section 6.1). The authors claim Section 5.2 shows no significant slowdown in "long-context evaluations" though this needs careful checking - Figure 8 appears to include multi-node configs (32B on 32 GPUs would span 4 nodes).

2. **The "non-intrusive" claim**: The paper states RDMA makes transfers non-intrusive except for gradient accumulation handled by a "lightweight daemon." The harsh critic raises a valid concern about daemon overhead and contention, but this is a minor engineering concern, not a fundamental flaw.

3. **Convergence equivalence**: The paper claims synchronous semantics are preserved. The Appendix F verification is referenced but not in main text. This is a reasonable concern but not fatal - the mechanism is mathematically equivalent (minibatch-level synchronization is maintained).

4. **Model sizes**: The 32B model on 32 GPUs IS a realistic multi-node setting (4 nodes of 8 GPUs).

Let me now write the final review.</think>Now I have a thorough understanding of the paper. Let me synthesize the final review.

## Summary

The paper identifies that per-layer collective communication (all-gather/reduce-scatter) in FSDP creates synchronization barriers that force all devices to wait for the slowest at every layer, wasting GPU time under the imbalanced workloads common in LLM post-training. The authors propose On-Demand Communication (ODC), which replaces these collectives with point-to-point RDMA-based primitives, reducing synchronization granularity from per-layer to per-minibatch while preserving FSDP's sharded memory layout. ODC also enables simpler, more effective minibatch-level load balancing (LB-Mini). Across SFT and RL tasks on models from 1.5B to 32B on up to 32 GPUs, ODC achieves up to 36% speedup over standard FSDP.

## Strengths

- **Clear problem identification with formal backing:** The paper formalizes FSDP's synchronization bottleneck as Eq. (1) and provides empirical evidence (up to 50% device idle time in Table 6). The insight that per-layer collectives are fundamentally mismatched with imbalanced workloads is well-argued.

- **Principled and minimal solution:** Replacing collectives with P2P on-demand primitives directly addresses the root cause (fine-grained synchronization) rather than symptoms. The idea that FSDP already has a PS-like sharded memory layout, and can thus naturally support P2P parameter fetch/gradient push, is an elegant observation. The solution preserves synchronous training semantics at the minibatch boundary.

- **Consistent empirical gains:** Speedups range from ~5% to 36% across diverse SFT and RL tasks (Figures 8–9), and the parametric study (Figure 10) provides clear guidance on when ODC helps most (longer sequences, more devices, moderate minibatch sizes).

- **The load balancing insight is genuine:** The O(s) memory vs. O(s²) compute mismatch argument explaining why microbatch-level packing fundamentally cannot eliminate imbalance is a real contribution, and the shift to minibatch-level balancing enabled by ODC's decoupling is both novel and effective (LB-Mini outperforms LB-Micro at small minibatch sizes).

- **Honest discussion of limitations:** The inter-node bandwidth gap (Figure 11) is presented transparently, and Section 6.1 proposes concrete mitigations (computation-communication overlap, hybrid sharding), acknowledging the tradeoff rather than hiding it.

## Weaknesses

### Fatal
None.

### Major

- **Inter-node scalability is the largest evidential gap.** The main speedup results (Figures 8–9) appear to include multi-node configurations (e.g., 32B on 32 GPUs spans 4 nodes), but the paper does not separately report intra-node vs. multi-node speedup. The parametric study's golden setting is 1.5B/8 GPUs (single-node), and Figure 11 shows ODC's P2P bandwidth lags significantly behind NCCL collectives across nodes. The discussion in Section 6.1 argues that O(s²) computation can hide O(s) communication latency for long sequences, and Appendix E claims hybrid sharding mitigates the overhead—but neither argument appears in the main evaluation. Given that multi-node deployment is the norm for serious LLM training, the paper would be substantially stronger with multi-node ablation results (especially isolating the speedup with/without inter-node traffic) and hybrid sharding results in the main body. The claim of being "a superior fit for LLM post-training" is well-supported for intra-node settings but less thoroughly validated for the multi-node regime.

- **Conflation of ODC communication gains with LB-Mini load balancing gains.** The paper compares ODC+LB-Mini (combined) primarily against Collective+LB-Micro, which conflates two changes. While Figure 8 shows ODC paired with LocalSort and LB-Micro baselines (which partially isolates the communication effect), the analysis does not clearly quantify how much speedup comes from the communication change alone vs. the load-balancing change. An explicit ablation (ODC+LB-Micro vs. Collective+LB-Micro) isolating the communication effect would significantly strengthen the contribution decomposition.

### Minor

- **The gradient accumulation daemon is described as "lightweight" without microbenchmark-level evidence.** The paper states scatter-accumulate is handled by a "lightweight daemon" (Section 3.2), which adds compute work on the target device while it may be executing its own forward/backward passes. While RDMA makes reads non-intrusive, the accumulation step could contend with local computation. A simple measurement of daemon CPU/GPU overhead under concurrent compute would address this cleanly.

- **Training convergence equivalence is asserted but only verified in an appendix.** The claim that ODC "preserves the synchronous optimization semantics" (Section 3) is critical, and the formal argument is sound (minibatch-level barrier ensures all gradients are summed before the optimizer step). However, convergence verification is relegated to Appendix F rather than shown in the main body. Including even a brief convergence comparison (loss curves) in the main evaluation would be more convincing.

### Trivial
None.

## Nice-to-Haves

- Multi-node ablation results with speedup broken down by intra-node vs. cross-node contribution, and hybrid sharding results in the main body.
- Explicit ablation disentangling ODC's communication benefits from LB-Mini's load-balancing benefits.
- Wall-clock absolute throughput numbers alongside relative speedups, to help practitioners assess practical relevance.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"Multi-node results with ≥7B models are missing"** — The paper does include experiments on 7B, 14B, and 32B models (up to 32 GPUs). The harsh critic's claim that all main results use 8 GPUs is incorrect; only the parametric study's golden setting uses 1.5B/8 GPUs.

- **"Equivalence of training semantics is asserted but not verified at scale"** — The paper verifies convergence in Appendix F, which is referenced in the main text. While it would be preferable to include this in the main body, the claim is not unverified—just deferred to the appendix. The parser strips appendix content; it exists in the original submission.

- **"Framing as 'revisiting parameter server' is misleading"** — The paper positions this as adapting PS principles into FSDP, which is accurate. Section 3.1 makes the connection explicit and does not oversell it as a standalone PS.

- **"No wall-clock time or throughput in absolute terms"** — This is a mild convenience request. Relative speedup is standard in systems papers and more informative for generalization.

- **"Implementation constraints in verl limit LB-Mini"** — The paper is transparent about this limitation and explains why. This is not a weakness of the paper; it's a candid acknowledgment of scope.

- **"The 1.5B golden setting is a toy"** — The parametric study systematically varies parameters from this setting, including scaling devices to 32. The main results (Figures 8–9) use models up to 32B. The golden setting serves as a controlled baseline, not the sole evaluation.

## Novel Insights

The paper's most novel insight is the dual observation that (1) FSDP's sharded memory layout already constitutes a distributed parameter server, making P2P communication a natural fit rather than a radical redesign, and (2) the O(s)/O(s²) memory-vs-compute scaling mismatch in long-sequence training makes microbatch-level packing fundamentally insufficient, making minibatch-level balancing (enabled by P2P decoupling) both simpler and more effective. Together, these observations make the case that the PS paradigm's relative advantages over collectives shift dramatically when moving from pre-training to post-training workloads.

## Suggestions

- Add a multi-node ablation table/bar-chart showing speedup decomposition for configurations that cross node boundaries, with and without hybrid sharding. This would directly address the most significant remaining concern.

- Include an explicit ODC+LB-Micro vs. Collective+LB-Micro comparison to isolate the communication-only contribution, alongside the full ODC+LB-Mini vs. Collective+LB-Micro comparison that demonstrates combined benefits.

## Calibration Anchors

- **Zero Bubble Pipeline Parallelism** (avg 7.0, Accept): Systems paper improving distributed training efficiency via smarter scheduling. Similar spirit of reducing idle time via synchronization relaxation. ODC's gains (up to 36%) are larger in magnitude but on a less standardized evaluation setup. ODC has a more significant limitation (inter-node gap) than ZBPP.

- **NetMoE** (avg 7.2, Accept Spotlight): Communication optimization for distributed training from a data perspective. Similarly structured contribution (formal model + algorithmic optimization + evaluation). NetMoE has cleaner evaluation but less fundamental insight.

- **DEPT** (avg 8.0, Accept Oral): Communication-efficient distributed LLM training with novel decoupling idea. Higher impact result with orders-of-magnitude communication reduction. More theoretically grounded than ODC.

- **DynMo** (avg 5.25, Reject): Dynamic load balancing in distributed training. Similar motivation (addressing imbalance) but weaker methodology and evaluation concerns. ODC is significantly more rigorous.

- **LightSeq** (avg 5.0, Reject): Sequence-parallel distributed training for long contexts. Similarly addresses scalability for long-sequence LLMs with speedups of 1.24–2.01×. Weaker baseline comparison than ODC but had multi-node evaluation concerns.

- **Decentralized Training (Heterogeneous)** (avg 2.0, Reject): Combining PS with pipeline parallelism for heterogeneous networks. Similar "PS revisit" angle but fundamentally flawed methodology. ODC is far more rigorous and practically grounded.

- **OMNIBAL** (avg 5.0, Reject): Computation rebalancing for VLM training. Similar load-balancing motivation with 1.8× speedup. Had evaluation concerns about generalizability.

**Assessment:** ODC is a solid systems paper with a clean, well-motivated idea and consistent empirical gains. Its main limitation—the inter-node communication overhead—is honestly disclosed and partially mitigated, but not fully validated in the main evaluation. Among the calibration anchors, ODC sits above the rejected papers (DynMo, LightSeq, OMNIBAL) in rigor and clarity, but below the high-Scoring accepted papers (DEPT, NetMoE) in evaluation completeness and theoretical depth. The paper is comparable to ZBPP (avg 7.0) in concept but has a more significant scalability gap. I place it slightly below ZBPP.

## Score and Decision

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>