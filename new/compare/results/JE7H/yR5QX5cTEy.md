---
job_id: 72624e66-60a4-46c2-8f52-7490db0eaacd
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: yR5QX5cTEy.pdf
paper: Efficient Allreduce With Stragglers
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a new AllReduce algorithm for distributed ML workloads, analyzes it in the α–β model, and evaluates it on GPU clusters; this fits squarely under “infrastructure, software libraries, hardware” and “large-scale learning / optimization” within ICLR’s scope.

## Minimum Quality
Pass ✅.  
The paper is in English and has all major sections: Abstract, Introduction, Related Work (Section 2 and Appendix A), Method (Section 3), Experiments & Results (Section 4 plus appendices), and Conclusion (Section 5). The theoretical development is coherent, proofs are provided (Appendix D), and the experimental methodology is fairly detailed and uses reasonable baselines. I do not see fatal methodological or statistical flaws that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden prompts or attempts to manipulate automated reviewing systems. The only mention of LLMs is in the reproducibility / LLM usage note, which is benign.

---

# Expected Review Outcome:

## Summary

The paper introduces **StragglAR**, a new AllReduce algorithm that explicitly exploits *straggler* GPUs: while the slowest GPU is still computing, the remaining \(n-1\) GPUs eagerly run a ReduceScatter, then execute a custom schedule to complete AllReduce once the straggler arrives.  

In the α–β model, for power-of-two \(n\), the authors prove that StragglAR completes in \(n + \log n - 2\) rounds, yielding asymptotic bandwidth cost \(s\beta\), which is a 2× improvement over the classical \(2s\beta\) lower bound for synchronous bandwidth-optimal AllReduce when the precondition fully overlaps the straggler delay.  

They implement the algorithm using NCCL P2P primitives, compare against Ring, RHD, MSCCL, and a straggler-aware Broadcast baseline on 4–8 GPU systems, and show up to 25% AllReduce speedup and a few percent end-to-end training gains, with simulations predicting larger gains at 64–256 GPUs.

## Strengths

1. **Conceptual shift: temporal asymmetry in collectives.**  
   The central idea is to *treat straggler delay as a communication window* and to design the collective around this temporal asymmetry rather than assuming all ranks start simultaneously. This is conceptually clean and, to my knowledge, not exploited in prior AllReduce algorithms for homogeneous scale-up domains. Section 3 articulates this point clearly, and Figure 1 illustrates it well by contrasting the standard barriered AllReduce (waiting) with StragglAR’s overlapping ReduceScatter work.

2. **Strong analytical result, apparently tight in the chosen model.**  
   For power-of-two \(n\), Theorem 1 (Section 3.2, Appendix D) shows StragglAR finishes in \(n+\log n-2\) rounds, so the exposed bandwidth term is
   \[
   \frac{n + \log n - 2}{n-1} s\beta \to s\beta,
   \]
   which is strictly below the classic \(2s\beta\) lower bound applicable to temporally-symmetric AllReduce. Table 1 makes this comparison explicit, showing both ideal and worst-case α–β costs. The derivation of the critical delay in Appendix B, where the inequality
   \[
   T_{\text{straggler}} \ge (\log n - 2)\alpha + \frac{\log n}{n} s \beta
   \]
   characterizes when StragglAR beats Ring, is a nice, concrete quantitative handle on when this algorithm helps.

3. **Nontrivial algorithmic design with clear structure and proofs.**  
   The core schedule synthesis for power-of-two \(n\) in Algorithm 1 (Page 4) is not a trivial tweak of Ring or RHD. The active-chunk invariant, the split into \(P_r\) and \(Q_r\), and the handling of the “critical window” (ranks \(r+1\) to \(r+\log n\)) are carefully argued. Figures 4a and 4b are particularly helpful: Figure 4a gives a full schedule trace for 4 GPUs; Figure 4b shows how the matching restriction in the critical window prevents future invariant violations. Lemma 1 and Lemma 2 in Appendix D are reasonably detailed and check out logically; the inductive invariant \(\mathcal{I}(r)\) is precise and tracks both chunk multiplicities \(|A[c_j]|\) and the structure of \(P_r, Q_r\).

4. **Well-thought-out best-case / worst-case positioning, with competitive downside.**  
   Section 3.2 and Table 1 carefully distinguish:
   - Ideal case: full overlap of ReduceScatter with the straggler, giving the \(n+\log n - 2\) round schedule as exposed cost.
   - Worst case: no overlap (no straggler delay, mispredicted straggler, etc.), where total cost becomes
     \[
     T_{\text{SAR}}^- = \left(2(n-2)+\log n\right)\alpha + \frac{2(n-2) + \log n}{n-1} s\beta,
     \]
     which asymptotically matches the \(2 s\beta\) of Ring.  
   Figure 2b and Figure 6c make this tradeoff very clear: as \(n\) increases, StragglAR’s performance band is strictly above the baseline for realistic straggler delays and essentially never below at scale.

5. **Solid experimental methodology and baselines.**  
   The authors implement all algorithms (Ring, RHD, MSCCL allpairs, Broadcast, StragglAR) using the same NCCL P2P API and custom reduction kernels (Section 4), which avoids apples-to-oranges comparisons against vendor-optimized black boxes. Figure 5 (a–f) provides a thorough breakdown:
   - (a,d) show ideal-case bandwidth vs buffer size.
   - (b,e) use empirically observed average straggler delays.
   - (c,f) fix 4 GiB and vary the delay, identifying critical delays of 5.53 ms / 7.57 ms on H100/A100.  
   This is more comprehensive than most collective-communication works in ML venues. They furthermore evaluate on a different 4-GPU topology (Perlmutter) in Figure 10, and carefully discuss a known NCCL tuning anomaly around 64–256 MiB (Appendix H, Figure 11), which is refreshingly honest.

6. **End-to-end ML relevance with real workloads.**  
   Section 4.2 and Table 2 provide end-to-end fine-tuning experiments for Llama-3.2-3B, Phi-3-mini-3.8B, and Qwen-2.5-3B. The reported speedups of 2.39–4.75% over Ring for 100 iterations may sound modest, but in realistic training regimes translate to several GPU-hours per day (last column of Table 2). Figure 12 is particularly nice: it shows that the loss curve for StragglAR is simply time-shifted relative to Ring, confirming identical convergence with shorter wall-clock. This cleanly demonstrates that the algorithm is strictly a communication optimization, not a numerical approximation.

7. **Careful scaling analysis and simulation.**  
   Section 4.3 and Figure 6c study scaling up to 256 GPUs in the α–β model using empirically motivated \(\alpha, \beta\). The plots clearly show StragglAR’s speedup band widening with \(n\). Appendix B’s Figure 7a and 7b explicitly chart the critical delay vs. \(n\) and \(\log n / n\) ratio, making the asymptotic argument that the required delay becomes negligible at scale very concrete. Appendices J and Figures 13–14 add end-to-end simulation using FlexNet/FlexFlow for BERT under various \(\alpha\) assumptions, further corroborating gains.

8. **Positioning vs. straggler and collective literature is strong.**  
   Section 2 and Appendix A do a good job distinguishing StragglAR from: (i) dropping/approximating stragglers (e.g., Harlap et al. 2016, Karakus et al. 2017, OptiReduce, DropCompute), (ii) algorithm synthesizers/topology-aware collectives (TACCL, ForestColl, Tacos, Blink, etc.), and (iii) system-level straggler mitigation and workload balancing. Table 3 is a helpful overview situating StragglAR along axes of domain, backend, and whether convergence is affected.

## Weaknesses

1. **Dependence on accurate and timely straggler identification is underdeveloped at the systems level.**  
   The theoretical guarantees assume a known straggler rank and sufficient delay to at least partially overlap the ReduceScatter. Section 4 “Detecting stragglers” briefly suggests using “online straggler detection tools” and in the end-to-end experiments they simply fix a likely persistent straggler rank via offline profiling (§4.2). However, practical straggler patterns can be more volatile, especially at datacenter scale and under shared workloads. The paper claims that misprediction only leads to near-baseline performance (which is true asymptotically), but provides limited quantitative evidence of performance when the detected straggler is frequently wrong, beyond the Qwen-2.5-3B case in Table 2. A more systematic experiment sweeping “fraction of iterations where the chosen rank is actually the last to arrive” would clarify how robust StragglAR is to noisy detection, particularly for small and medium \(n\) where its worst-case overhead is nontrivial (see Fig. 10 for 4 GPUs where gains are small). This matters because deploying StragglAR in realistic training stacks requires confidence that imperfect detection does not harm performance.

2. **Assumptions about topology and per-link behavior could limit applicability.**  
   Section 3 assumes “each GPU has a single connection” and that sending to one peer fully utilizes available bandwidth. This matches many NVSwitch-based designs, but real systems are more nuanced: links are often multi-ported and may support “striped” traffic patterns with partial contention. The paper cites empirical confirmation “in §4”, but I did not see clear microbenchmarks that directly demonstrate that multi-peer concurrent sends are suboptimal in the tested DGX configurations. Moreover, the MSCCL *allpairs* baseline explicitly uses bandwidth splitting across all peers, and while StragglAR beats it on large buffers in Figure 5, it is not completely obvious that the “one-peer-at-a-time” design is generally optimal on future fabrics (e.g., GB200 NVLink5 with more complex bisection constraints). A short discussion acknowledging when these assumptions might fail and how StragglAR’s schedule could be adapted to multi-port models would strengthen the generality claims.

3. **Theoretical results are clean only for power-of-two \(n\); non-Po2 and odd \(n\) are handled heuristically.**  
   The main theoretical contribution, including Theorem 1 and the \(n + \log n - 2\) bound, applies only to power-of-two \(n\) with a single communication port per GPU. Section E admits that for even non-power-of-two \(n\), they fall back to maximum-weight matching with no formal α–β bound, and for odd \(n\) they provide no schedule at all. Figure 9 suggests that empirically the synthesized schedules still outperform baselines on even non-Po2 \(n\), but this is purely simulation-based. For a work that leans heavily on beating a long-standing bandwidth lower bound, the absence of any theoretical statement for general \(n\) is a noticeable gap. At minimum, a proof that the constructed matchings for even non-Po2 \(n\) are within a concrete factor of the Po2 bound (e.g., \(n + 2\log n\) rounds) would make the contribution feel less “fragile”. Alternatively, an explicit argument that most practical NCCL deployments already use Po2 group sizes would be useful; currently this is just mentioned informally.

4. **Some α–β derivations are slightly hand-wavy and could be tightened.**  
   In Appendix B, the derivation of the *critical delay* uses an approximation step:
   \[
   \frac{2(n-2)+\log n}{n-1}s\beta \approx \frac{2(n-1)+\log n}{n}s\beta
   \]
   to obtain
   \[
   T_{\text{straggler}} \ge (\log n - 2)\alpha + \frac{\log n}{n}s\beta.
   \]
   This approximation is intuitively reasonable for large \(n\), but the paper never bounds the error term. Since this inequality is used to make concrete claims like “critical delay < 0.1 ms for \(n=256\)” (Figure 7a), giving the exact expression or a rigorous inequality (e.g., showing that the approximate bound is slightly *upper-bounding* the true critical delay) would avoid any ambiguity. Likewise, Section 3.2 uses limit arguments like \(\lim_{n\to\infty}\frac{n+\log n - 2}{n-1} s\beta = s\beta\) to argue 2× speedup, but does not quantify finite-\(n\) gaps, which would be useful for the 8–64 GPU regime where most ML training currently runs.

5. **Evaluation breadth is good, but depth on some important axes is limited.**  
   While Figure 5 and Figure 10 systematically vary buffer sizes and straggler delays, there are some missing experimental angles:
   - **Multiple stragglers / correlated slowdowns.** The analysis repeatedly argues that multiple stragglers are “highly improbable” due to continuous execution times, but in real systems correlated jitter (e.g., shared PCIe root, NUMA issues, OS noise) can cause small groups of GPUs to lag together. There is no experiment with two or more delayed ranks, even in simulation, to validate that performance remains near worst-case bounds.
   - **Dynamic stragglers with online detection.** All real-hardware experiments either simulate a straggler by idling a fixed GPU or use an offline-chosen persistent straggler. It would be instructive to plug in one of the cited dynamic detectors (e.g., AdapCC-style or a simple runtime rule based on previous iteration durations) to support the claims in Section 4 that “StragglAR does not require online straggler detection” and that incorrect detection has minimal impact.
   - **Comparison to vendor-optimized NCCL AllReduce.** The authors intentionally reimplement Ring/RHD/etc. using NCCL P2P for fairness. This is reasonable for isolating the algorithmic effect, but from a practitioner standpoint it would be helpful to see at least a sanity-check comparison against *ncclAllReduce* itself on a subset of experiments, even if StragglAR uses a different API. Currently the reader has to assume that the P2P-based Ring is close to vendor Ring performance, which is not obviously guaranteed.

6. **Algorithmic complexity and implementation overhead not fully quantified.**  
   StragglAR schedules require an extra synchronization barrier and coordination for the initial ReduceScatter, as noted in the Limitations (Section 4.3). However, the paper provides no explicit microbenchmark for these control overheads, especially for high-iteration-count training jobs. For example, in Figure 5(c,f) the total runtime includes the precondition overlap period and collective execution, but it is not clear whether driver overhead, additional CUDA events, or process-group-level synchronization differences are fully comparable between StragglAR and baselines. A small experiment measuring “overhead per AllReduce call when there is *no* straggler and buffer size is small” would show that StragglAR does not pay excessive control overhead when α dominates.

7. **Generality to other collectives and parallelism modes is mostly asserted, not demonstrated.**  
   The abstract and Introduction highlight tensor-parallel training/inference as a motivating use case, and Section A notes that StragglAR applies generically to both data and tensor parallelism because it preserves exact AllReduce semantics. However, all real experiments (Section 4.2) are data-parallel training, and the only tensor-parallel relevance is implicit through buffer sizes and invocation frequency. Similarly, the paper does not explore extensions to AllGather, All-to-All, etc. While it is reasonable to keep the scope to AllReduce for one paper, some of the more ambitious statements about “new paradigm for collective algorithm design” (Conclusion) would be more convincing with at least a theoretical sketch or limited experiment on, say, tensor-parallel Megatron-LM with StragglAR replacing NCCL AllReduce.

8. **Minor clarity/exposition issues.**  
   - Algorithm 1 as presented on Page 4 has some slightly ambiguous steps, for example:
     - The line “Each rank with a reduced chunk sends to any rank \(g > 2(\log n - 1)\) without a chunk” is not fully specified when there are multiple choices; the correctness proof later clarifies the matching, but an explicit tie to the invariant used in Lemma 1 would help.
     - Variable names like \(c, c'\) in the critical-window loop are overloaded and could benefit from a more formal description of the selection rule (currently described verbally).
   - Figure 3 (the basic collective-operations illustration) appears truncated or oddly typeset in the provided text, which slightly hinders understanding for readers unfamiliar with ReduceScatter / AllGather.  
   None of these are fatal, but a careful pass could make the paper easier to follow for non-HPC specialists.

9. **Handling of NCCL’s anomalous region is acknowledged but somewhat brushed aside.**  
   The authors rightly note in Section 4.1 and Appendix H (Figure 11) that NCCL P2P exhibits non-linear behavior in the 64–256 MiB range, and that this causes MSCCL and Direct algorithms to spike in performance at 256 MiB in Figure 5. However, they then essentially treat this as a quirk of the environment. Given that this range is not entirely academic (many models produce per-layer buckets in that ballpark), it would be good to see:
   - A sensitivity analysis across NCCL versions and CUDA versions, or
   - A note about whether StragglAR’s advantage persists when using NCCL *collective* APIs (which may use different protocols) as the baseline.  
   Currently, the reader has to trust that this anomaly does not fundamentally preclude StragglAR’s benefits in other setups.

## Potentially Missing Related Work

1. **Ertza Warraich et al., “Ultima: Robust and Tail-Optimal AllReduce for Distributed Deep Learning in the Cloud,” 2023.**  
   This work also specifically targets tail performance and robustness of AllReduce in the presence of stragglers and failures in cloud environments, with a focus on mitigating high-percentile latency. It appears to be a direct conceptual neighbor to OptiReduce and StragglAR. It should be discussed in Section 2 and Appendix A alongside OptiReduce / tail-robust AllReduce algorithms, and, if feasible, compared as a baseline (at least qualitatively) in Table 3, clarifying differences in assumptions (scale-out vs. scale-up, approximate vs. exact reductions, synchronous vs. partially asynchronous).

(“MSCCL++: Rethinking GPU Communication Abstractions for Cutting-edge AI Applications” is already cited and discussed as an implementation abstraction rather than an algorithm, so it is not missing.)

## Questions

1. **Robustness to mispredicted or rapidly changing stragglers.**  
   Can the authors provide quantitative results, either in hardware or simulation, where the chosen “straggler rank” is wrong in a controlled fraction of iterations (e.g., 25%, 50%, 75%)? In particular, for 8 GPUs and 4 GPUs, how much does the average speedup degrade as the detection accuracy drops, and does StragglAR ever become *significantly* slower than Ring in these regimes?

2. **Behavior under multiple moderately slow GPUs.**  
   In practice, has the team observed cases where two or more GPUs are within, say, 0.1 ms of each other in reaching the barrier, effectively creating a small straggler set? Can you simulate a scenario where two designated ranks are delayed (e.g., by the same sleep kernel) and show how StragglAR performs relative to Ring and Broadcast in this setting?

3. **Extension to odd \(n\) and non-Po2 values with guarantees.**  
   For even non-Po2 \(n\), your practical schedules seem to realize about \(n + 2\log n - 2\) rounds (Appendix E). Can you provide at least a partial theoretical argument or bound that explains why this seems to be the case, or a formal guarantee for a weaker bound (e.g., \(O(n + \log^2 n)\) rounds)? For odd \(n\), do you see any fundamental obstruction to constructing similar schedules, or is it “just” a matter of additional matching complexity?

4. **Practical integration with dynamic detection in real stacks.**  
   You mention that StragglAR can benefit from tools like AdapCC or runtime profiling to identify stragglers at each iteration. Do you envision implementing this via a custom process-group backend (as in your PyTorch C++ extension) with a control-plane side channel, or by letting the library autonomously infer the “first \(n-1\) ready ranks” through CUDA streams? Clarifying a realistic integration path into, say, PyTorch DDP or DeepSpeed would help practitioners judge adoption cost.

5. **Impact of hardware and NCCL version.**  
   Have you experimented with other NVLink/NVSwitch generations or different NCCL versions to see whether the 64–256 MiB anomaly in Figure 11 qualitatively alters StragglAR’s relative performance? Even a brief statement summarizing unpublished experiments would help readers assess robustness to software stack changes.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The core algorithm and proofs for power-of-two \(n\) are technically solid and well argued; the α–β analysis is mostly careful, and the experiments are well designed with strong baselines. Some aspects (non-Po2 \(n\), multiple stragglers, approximation in the critical delay derivation) remain more heuristic than fully rigorous, but they do not undermine the main claims.

## Presentation Rating

3: good.  
The paper is generally clearly written, well structured, and rich in informative figures (1, 2, 4, 5, 6, 8–10, 12–14) and tables (1–3). A few algorithmic details in Algorithm 1 and some notation could be clarified, and the treatment of certain NCCL quirks feels slightly rushed, but overall the exposition is above average.

## Contribution Rating

4: excellent.  
The work introduces a genuinely new angle on AllReduce design (temporal asymmetry and explicit exploitation of stragglers), provides a nontrivial algorithm with proven communication complexity that beats a long-standing bandwidth bound in a realistic model, and supports it with credible empirical and simulated evidence. It is of clear interest to the distributed ML and collective-communication communities.

## Overall Rating

8: Accept, good paper (poster).  
The paper makes a meaningful and well-supported contribution: it proposes a new AllReduce algorithm that exploits straggler delays to achieve asymptotic 2× bandwidth savings over classical bounds while remaining competitive in the worst case. The theory for power-of-two \(n\) is solid, and the combination of real-system experiments and simulations convincingly shows benefits for realistic buffer sizes and straggler delays. The main weaknesses concern generality to arbitrary \(n\), practical integration with dynamic straggler detection, and some gaps in the evaluation matrix (dynamic/ multiple stragglers, vendor-optimized baselines). These are important but not fatal; they leave room for follow-up work rather than undermining the current contribution.

## Reviewer Confidence

4: confident.  
I am comfortable with the α–β model, collective-communication literature, and distributed training systems, and I followed the proofs and experiments closely. There is some dependence on implementation subtleties (NCCL internals, future hardware fabrics) where I cannot be 100% certain, but overall I am confident in this assessment.