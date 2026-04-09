## Summary
This paper proposes On-Demand Communication (ODC), which replaces per-layer collective communication (all-gather/reduce-scatter) in FSDP with point-to-point RDMA primitives (gather/scatter-accumulate), effectively reframing FSDP as a decentralized parameter server. By relaxing synchronization from the layer level to the minibatch level, ODC decouples device execution and enables simpler minibatch-level load balancing (LB-Mini), achieving up to 36% throughput speedup over standard FSDP on long-sequence SFT tasks.

## Strengths
- **Precise, evidence-backed root cause identification**: The paper identifies per-layer synchronization barriers as the direct cause of up to 50% device idle time under imbalanced workloads (Table 6), grounding the motivation in concrete measurements rather than hypothetical arguments.
- **Elegant conceptual reframing of FSDP as decentralized PS**: Replacing collectives with on-demand P2P communication while preserving FSDP's sharding layout is a genuinely novel perspective that yields both conceptual insight (the PS connection) and practical benefit (decoupled device progress) — most prior work focused on better packing rather than questioning the communication model itself.
- **Comprehensive parametric evaluation**: Section 5.3 systematically varies minibatch size, max length, packing ratio, and device count, providing clear operational boundaries for when ODC helps most and when its benefits diminish.

## Weaknesses

### Major:
- **Inter-node communication overhead limits multi-node scalability**: Figure 11 shows ODC primitives are significantly slower than NCCL collectives for cross-node communication because they forgo hierarchical topology optimizations (e.g., inter-node broadcast + intra-node broadcast). While the paper argues that long-sequence computation hides this latency and proposes hybrid sharding as mitigation (Section 6.1, Appendix E), hybrid sharding increases per-node memory (Figure 13) and may not suffice for memory-constrained large-scale training. This is the most important scalability limitation and should be more prominently quantified in the main results — specifically, a per-experiment breakdown of how many nodes were used and whether inter-node traffic was a factor would help readers assess real-world applicability.

- **Headline speedups conflate communication and load-balancing contributions**: The peak 36% speedup (Table 5, 1.5B LongAlign, minibatch=4) comes from ODC+LB-Mini, while the isolated communication benefit (ODC LocalSort vs. Collective LocalSort) is only 0–10% in most configurations. Since LB-Mini is only feasible under ODC's relaxed synchronization, the two contributions are intertwined, but the paper's framing ("ODC achieves up to 36% speedup") attributes the full gain to the communication scheme. The ODC+LB-Micro vs. Collective+LB-Micro comparison better isolates the communication contribution (e.g., 16–23% for 1.5B LongAlign at minibatch=4/8), and the paper would be more honest by foregrounding these numbers and positioning LB-Mini as a complementary enabler.

### Minor:
- **RL speedups are modest and framework-constrained**: The ~10% RL speedup (Section 5.2) is limited because `verl` requires identical sample counts per device, preventing full LB-Mini usage. This means a key part of the proposed system cannot be evaluated in the RL setting, which is itself a major post-training workload. The paper is transparent about this, but it weakens the "diverse post-training tasks" claim.
- **Convergence verification is thin**: Appendix F validates identical loss curves on only 8k samples with a 1.5B model trained from scratch. Since ODC changes the gradient accumulation timing (scatter-accumulate happens on-demand rather than via coordinated reduce-scatter), subtle numerical differences could emerge over longer runs or with optimizer states that depend on gradient statistics. A longer convergence check or gradient norm comparison would strengthen confidence.
- **Hardware dependency on RDMA limits portability**: ODC requires CUDA IPC (intra-node) and NVSHMEM (inter-node), restricting deployment to NVIDIA GPU clusters with RDMA configured. The paper does not discuss fallback paths for TCP/Ethernet clusters or non-NVIDIA hardware, limiting the generality of the contribution.

### Trivial:
- **Gradient buffer memory overhead**: Appendix B bounds the dedicated per-client buffer memory to M per server, which temporarily increases gradient memory compared to in-flight reduction in ring-based reduce-scatter. For models already near memory limits, this could matter, though the paper shows it is manageable in practice.

## Nice-to-Haves
- GPU timeline traces (e.g., Nsight Systems) comparing FSDP and ODC to visually confirm barrier elimination and bubble reduction, complementing the schematic Figures 1–2.
- Gradient norm or parameter divergence analysis between ODC and FSDP to rigorously verify numerical equivalence beyond loss curves.
- Topology-aware P2P routing (e.g., intra-node cache-and-forward) to mitigate the inter-node bandwidth penalty discussed in Section 6.1.
- Full RL framework integration removing `verl`'s equal-sample-count constraint to demonstrate ODC's complete potential for RL post-training.
- A plot of speedup vs. inter-node communication fraction to define the operational boundary where ODC becomes net-negative.

## Removed Points
*These points are flagged to be removed, treat them with caution:*
- **Synchronous vs. asynchronous ambiguity** (Harsh Critic): The critic claims ambiguity between the introduction's "synchronous optimization semantics" claim and Section 6.2's "Relaxing Synchronization" future work. The paper is clear throughout (Section 3, Section 3.2) that ODC preserves a minibatch-level barrier and synchronous semantics; Section 6.2 explicitly discusses relaxing this as *future work*. No ambiguity exists.
- **Presentation complexity / implementation hard to understand** (Harsh Critic, transferred from Cut Cross-Entropy review): This is a style/formatting nitpick; the implementation details are in appendices with adequate pseudocode and memory analysis.
- **Baseline fairness — baselines not fully optimized** (Harsh Critic, transferred from ProTrain review): The paper compares ODC vs. Collective using the *same* packing strategy (LB-Micro), which fairly isolates the communication contribution. Requesting additional framework-specific optimizations is scope creep.
- **Lack of ablation studies** (Harsh Critic, transferred from Cut Cross-Entropy review): The paper provides parametric studies (Section 5.3) and compares four combinations of communication scheme × packing strategy, which serves as ablation. More granular ablations would be nice but are not a core flaw.
- **Demand for DeepSpeed/Megatron comparison** (Spark Finder): ODC is a communication scheme within FSDP, not a full training framework. Comparing against end-to-end frameworks would conflate many unrelated variables.
- **Deadlock concerns** (Harsh Critic): RDMA operations (CUDA IPC, NVSHMEM put/get) are one-sided by design — the initiator controls the transfer without requiring active remote participation. The paper's "non-intrusive" claim is consistent with RDMA semantics. Deadlock concerns reflect a misunderstanding of the communication model used.
- **Scalability beyond 32 GPUs demand** (Spark Finder): This is a generic "needs larger scale" request; 32 GPUs across multiple model sizes is adequate for the paper's scope.
- **Load balancing algorithm computational cost** (Harsh Critic): Karmarkar-Karp is a well-known efficient heuristic; no evidence is provided that its runtime is non-trivial relative to training time.

## Novel Insights
The key insight that emerges from synthesizing these reviews is that ODC's contribution is best understood as a *system co-design* rather than a pure communication optimization. The real leverage comes from the virtuous cycle: relaxing synchronization (via P2P) *enables* better load balancing (LB-Mini), which in turn *amplifies* the benefit of relaxed synchronization. The paper's most honest numbers (ODC+LB-Micro vs. Collective+LB-Micro) show that communication alone gives 16–23% gains on high-imbalance SFT, while LB-Mini pushes this to 34–36%. This co-design is the paper's genuine novelty, but the current framing obscures it by attributing all gains to "ODC." A reframe positioning ODC as the *enabler* of a new load-balancing regime — rather than the sole performance driver — would be both more honest and more intellectually interesting.

## Suggestions
- In the abstract and main results, report the isolated communication contribution (ODC+LB-Micro vs. Collective+LB-Micro) alongside the full ODC+LB-Mini numbers, and explicitly discuss the co-design nature of the gains.
- Add a table or figure breaking down which experiments were single-node vs. multi-node, so readers can assess whether the inter-node overhead (Figure 11) affected each result.
- Extend the convergence verification to at least one 7B+ model with a longer training run (e.g., 50k+ samples), or add a gradient norm comparison to validate numerical equivalence beyond loss curves.