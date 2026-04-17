---
job_id: 64c7d875-31a1-44ca-8666-21fb8c383e58
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: bm3rbtEMFj.pdf
paper: ELMUR: External Layer Memory With Update/Rewrite for Long-Horizon RL Problems
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a transformer with external memory for long-horizon partial-observable RL / IL, evaluated on RL-style benchmarks and manipulation; this is squarely within ICLR’s scope (representation learning, transformers, RL, robotics).

## Minimum Quality
Pass ✅.  
The submission includes all major sections (Abstract, Introduction, Method, Experiments/Results, Related Work, Conclusion), is in English, presents a concrete algorithm with mathematical description, and provides substantial experimental evidence. I see no fatal methodological, theoretical, or evaluation flaws that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden instructions or content attempting to manipulate automated reviewing systems.

---

# Expected Review Outcome:

## Summary

The paper introduces ELMUR, a GPT-style transformer architecture augmented with layer-local external memory and a Least Recently Used (LRU) update mechanism for long-horizon partially observable control. Each transformer layer has a “token track” and a “memory track” that interact via bidirectional cross-attention (mem2tok for read, tok2mem for write), with memory slots updated across segments through either replacement or convex blending controlled by a parameter λ.  

The authors provide a simple theoretical analysis of the convex-blend LRU dynamics (exponential forgetting, effective retention horizons, and boundedness) and evaluate ELMUR on three memory-centric benchmarks: synthetic T‑Maze (with up to 10⁶-step corridors), POPGym (48 partially observable tasks), and the MIKASA‑Robo robotic manipulation suite with pixel observations, showing consistent gains over transformer and offline RL baselines.

---

## Strengths

1. **Clear architectural idea with explicit, analyzable memory mechanism.**  
   - The design in **Figure 1** cleanly separates a “token track” from a “memory track”, with mem2tok and tok2mem cross-attention blocks at every layer plus an LRU memory module. This is conceptually simple yet expressive: the model explicitly reads from and writes to a fixed-size per-layer memory, rather than relying on implicit recurrence or cached activations.  
   - **Algorithm 1** and the “Memory Management with LRU” pseudocode make the mechanism very transparent, which is often lacking in memory-augmented transformer work.

2. **Theoretical analysis is modest but directly tied to the mechanism.**  
   - Section 4, particularly **Proposition 1** and Equation (9), provides a clear derivation of exponential forgetting under convex blending, and relates the half-life of memory contents to λ.  
   - The derivation of the effective retention horizon \(H(\epsilon) = M \cdot L \cdot \frac{\ln(\epsilon)}{\ln(1-\lambda)}\) connects the architecture hyperparameters (number of slots M, segment length L, blend factor λ) to interpretable environment-step horizons.  
   - **Proposition 2** on boundedness, though simple, gives a clean guarantee that memory norms cannot explode under mild assumptions.

3. **Compelling long-horizon behavior on T-Maze.**  
   - **Figure 3** is quite striking: with a context length \(L = 10\) and only \(S=3\) segments during training, ELMUR maintains **100% success** on T-Maze corridors up to \(10^6\) steps, where all other baselines degrade. This concretely supports the claim that the external memory extends the effective horizon by orders of magnitude beyond the attention window.  
   - **Figure 4** further shows perfect generalization across train/test length pairs up to 9600 steps. Every cell is essentially 1.0, suggesting the model generalizes to both shorter and substantially longer sequences, not just memorizing a fixed scale.

4. **Strong empirical performance on memory-centric robotics.**  
   - On MIKASA‑Robo, **Table 1** and especially **Table 8** show substantial gains in success rate over strong baselines (RATE, DT, CQL, Diffusion Policy) on sparse-reward vision-based tasks. For example, on TakeItBack‑v0, ELMUR achieves \(0.78 \pm 0.03\) vs \(0.42 \pm 0.24\) for RATE and near-zero for CQL and DP.  
   - Notably, tasks like ShellGamePick‑v0 and RememberShape3‑v0 show ELMUR jumping from near-zero to high success (e.g., 0.02 → 0.66 on ShellGamePick‑v0 and 0.31 → 0.76 on RememberShape3‑v0), which aligns well with the memory-demanding nature of these tasks.  
   - The aggregate statistic in **Table 8** (“Sum Across All Tasks”) doubles RATE’s total success (5.42 → 9.24), suggesting the effect is broad, not cherry-picked.

5. **Breadth of evaluation across domains and modalities.**  
   - The paper covers synthetic vector observations (T‑Maze), structured partially observable benchmarks (48 POPGym tasks, **Table 2** and **Table 5**), robotic manipulation with RGB inputs (MIKASA‑Robo), a 3D navigation task (ViZDoom‑Two‑Colors, **Table 6**), and D4RL MuJoCo MDPs (**Table 4**).  
   - This variety demonstrates that ELMUR is not overfit to a single environment class or observation modality, and that adding explicit memory does not harm performance in simple fully observable settings (all methods reach 500 on CartPole‑v1).

6. **Detailed ablations and memory probing that genuinely interrogate the mechanism.**  
   - **Figure 6** systematically varies λ, memory initialization σ, memory size M, and segment configuration (L, S). The plots cleanly show that performance is stable when \(M \ge N\) (segments needed) and highly sensitive when \(M < N\), validating the capacity interpretation and supporting the theoretical M·L horizon relationship.  
   - The probing and visualization sections (A.9–A.11) are unusually thorough. For example, **Figure 9** (confusion matrices) demonstrates that a simple probe can decode the correct target color from memory slots after the cue and well into the decision phase. **Figure 10** (update patterns) and **Figure 15** (tok2mem/mem2tok attention maps) show that ELMUR performs localized one-shot writes into a specific slot and then repeatedly reads from that slot, rather than diffusing information across slots. This is strong evidence that the external memory is functionally used, not just extra parameters.

7. **Clarity and reproducibility.**  
   - The writing is generally very clear, with mathematical notation tied tightly to the implementation-level pseudocode. **Algorithm 1** and Algorithm 2 (LRU) are easy to map to code.  
   - The experiments section specifies seeds, evaluation protocol, and shared data budgets, and **Table 7** lists the hyperparameters for key experiments in detail, which makes replication feasible.

---

## Weaknesses

1. **Incremental conceptual novelty relative to existing memory-augmented transformers and RL memory architectures.**  
   - The core ingredients of ELMUR are: (i) per-layer external memory slots, (ii) read and write via cross-attention, and (iii) an LRU-style management policy with convex blending. All of these have close precedents: Memformer (Wu et al., 2020) uses global slots, Neural Turing Machines / DNC (Graves et al., 2014, 2016) use explicit external memory with learned read/write, Memorizing Transformers (Wu et al., 2022) and Transformer‑XL (Dai et al., 2019) extend context through caching, and RATE (Cherepanov et al., 2026c) already introduces a recurrent transformer with memory for RL.  
   - While the specific combination (layer-local memory + LRU + relative-bias mem2tok/tok2mem) is neat and more structured than some prior work, the conceptual distance is not large. The paper would benefit from a more explicit comparative analysis of what is *fundamentally different* from, say, Memformer or associative memory transformers beyond “layer-local” and “LRU”, and from articulating why these differences matter beyond empirical gains.

2. **Theoretical analysis is narrow and relies on simplifying assumptions that are glossed over.**  
   - The retention horizon \(H(\epsilon)\) derivation in Section 4 assumes “only one memory embedding is updated per segment of length L” and that each embedding is overwritten once every M segments *in expectation*. However, Algorithm 2’s LRU chooses the least recently used slot; which slot is selected depends on the full trajectory of write times and may correlate with the policy and environment, so the implied uniformity is not always justified. In particular, if one slot is frequently attended during tok2mem cross-attention, it might be repeatedly chosen for updates, invalidating the assumption that overwrites are evenly spaced.  
   - The analysis in **Proposition 1** and the corollary treats updates to a fixed slot as a simple convex-combination chain, which is fine mathematically, but it does not capture the stochasticity or content-dependent nature of slot selection. As a result, the claimed horizon scaling \( \sim M \cdot L / \lambda\) should be viewed as a rough lower bound, yet the text partially slips into interpreting it more strongly. It would be better to explicitly separate proven statements (per-slot exponential forgetting under chosen updates) from heuristic arguments (update-period ≈ M segments).  
   - Relative-bias cross-attention (Equations (6)–(7)) is central to the method, but there is no analysis of how it interacts with the LRU policy or memory stability; the theory is entirely about the scalar blend λ, ignoring the role of attention weights and gating in shaping which information actually gets stored.

3. **Limited comparison to closely related recent work on memory-efficient RL agents.**  
   - The related work section covers a wide swath of transformers, external memory models, and RL memory architectures, but it appears to miss at least one directly relevant contemporary method:  
     - Gupta et al., “Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning” (2025), which also designs memory-efficient architectures for embodied agents in partially observable environments, is not cited. This work seems directly comparable in goals (long-horizon, memory-intensive environments, embodied control). A discussion of similarities/differences in memory structure (global vs layer-local, learned vs heuristic replacement) and an empirical comparison on at least one overlapping benchmark would help contextualize ELMUR’s contribution.  
   - Given that RATE, RMT, Memformer, and Memorizing Transformers are already cited, the omission of this more recent memory-efficient RL architecture suggests the literature positioning is not fully up to date.

4. **Empirical story on POPGym is somewhat mixed, and aggregation choices are not thoroughly justified.**  
   - **Table 2** reports “Aggregated returns on 48 POPGym tasks” with summed returns (All, Puzzle, Reactive). While ELMUR leads overall (10.4 vs 9.5 for RATE), the margin is not huge, and some tasks in **Table 5** show ELMUR underperforming RATE or BC-LSTM (e.g., MineSweeperEasy‑v0 where RATE and BC-LSTM are better, or MultiarmedBanditEasy‑v0 where ELMUR is below the expert PPO-GRU). **Figure 5** visually focuses on relative improvement over DT, not over the strongest baseline.  
   - More importantly, the choice of summing returns across tasks is not normalized for very different reward scales and difficulties, and **Table 5** mixes puzzle-like and control-like tasks with very different ranges (e.g., -0.5 to 1 vs 0 to 1). This makes the aggregate number somewhat opaque. Normalized scores or counts of “wins” (best algorithm per task) would be more interpretable than raw sums. The text does mention “ranking first on 24 of 48 tasks”, but the figure/table does not directly summarize this statistic.  
   - The POPGym evaluation is also imitation-based from PPO-GRU experts, which can be strong but are not necessarily the best possible; it would be useful to discuss how close ELMUR’s performance is to the expert policy across tasks (the last column in Table 5), rather than only emphasizing deltas over DT/RATE.

5. **Dependence on hand-tuned memory hyperparameters, with some unstable regimes.**  
   - The ablations in **Figure 6** reveal that intermediate λ values (around 0.4–0.6) can be unstable in the under-provisioned regime \(M < N\), and that performance is quite sensitive to M and σ when capacity is tight. Table 7 further shows that λ, memory size, and memory dropout are tuned per task.  
   - In practice, this means that deploying ELMUR on a new domain may require nontrivial tuning of memory size, segment length, and λ to achieve good performance. There is no guidance on choosing these hyperparameters from first principles, despite having a theoretical expression for H(ε). For example, the theory suggests setting \(M \cdot L\) to exceed the task correlation horizon, but the experiments do not verify or calibrate this mapping quantitatively.  
   - Moreover, λ is always fixed, not learned. Allowing λ to be slot- or time-dependent and/or trainable (with regularization) might substantially improve robustness, but the paper does not explore this.

6. **Evaluation remains purely offline and simulation-based; claims about real-world robotics and RL could be toned down or more carefully delimited.**  
   - The introduction and conclusion strongly motivate real-world robotic agents under partial observability and long horizons. However, all experiments are in simulation and under offline imitation learning (or offline RL for D4RL). There is no demonstration that the method is stable under online RL training with exploration, or that it handles real-robot issues like latency and sensor noise.  
   - The authors are upfront about omitting real-robot experiments for practicality, which is fine, but then the interpretations occasionally overreach (e.g., “making long-horizon decision making feasible in robotics” in the abstract is stronger than what is actually shown). A more cautious framing would emphasize that ELMUR is promising for simulated memory-intensive robotics benchmarks but untested in real hardware or online RL settings.

7. **Some methodological details that matter for fairness and understanding are under-specified.**  
   - Losses are computed on each segment with “detached memory between segments” (Section A.8), meaning no backpropagation through the LRU update; this is an important design choice but is not discussed in the main text. It likely affects what can be learned about long-horizon credit assignment versus memory content. Clarifying in Section 3 why this choice was made (stability vs performance) and whether BPTT through memory was tried would be useful.  
   - In **Algorithm 1**, the relative bias computation appears twice (lines 3 and 8) with slightly inconsistent variable names (`B_rel` then `B_read`/`B_write`). It is implied that mem2tok uses `B_read` and tok2mem uses `B_write`, but this is not written explicitly. Similarly, Equation (2) uses `K, V = m` but does not mention the mask and bias arguments; for a method whose main novelty is memory-attention design, it would be better to spell these out more rigorously.  
   - Efficiency claims in Section 5.2 mention per-step latencies (e.g., 6.8ms vs 7.2ms for RATE and 10.7ms for DT on T‑Maze), but the exact hardware and implementation details (e.g., mixed precision, batch size during timing) are not fully specified in the main text, making it slightly hard to generalize these numbers.

Overall, the weaknesses are mostly about positioning, theoretical scope, and some missing detail or over-claiming, rather than fatal flaws in the core method or experiments.

---

## Potentially Missing Related Work

1. **Gupta, G., Yadav, K., Kira, Z. (2025). “Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning.”**  
   - This work targets memory-efficient architectures for embodied RL agents in memory-intensive long-horizon tasks, directly overlapping with the motivation and setting of ELMUR. It should be discussed in **Section 6 (Related Work)** under both “Memory in RL” and “Manipulation / Embodied Agents,” with a comparison of memory design (global vs layer-local, learned vs heuristic LRU), training regimes (online RL vs offline IL/RL), and computational trade-offs. If feasible, including Memo as a baseline on at least one shared or similar benchmark (e.g., a subset of POPGym or a navigation-style task) would further strengthen the empirical comparison.

If the authors are already aware of Memo but consider it orthogonal (e.g., due to different action spaces or training regimes), that rationale should be made explicit in the related work discussion.

---

## Questions

1. **On the choice of updating only one memory slot per segment (Algorithm 2).**  
   - Why is the update restricted to a single slot per segment, instead of, for example, updating multiple slots or using a soft weighting over candidates? Was this design empirically validated against alternatives? Some intuition or ablation on this choice (even on T‑Maze) would help clarify whether it is crucial for stability or just a convenience.

2. **On fixed versus learned λ.**  
   - Have you experimented with making λ trainable per slot or per layer, possibly regularized to stay in [0,1]? Given that **Figure 6(a)** shows nontrivial dependence on λ, it would be interesting to know whether the model can “self-tune” λ to the memory demands of each task, potentially mitigating instability when \(M < N\).

3. **On the approximation underlying the effective horizon formula.**  
   - In Section 4 you assume that “a memory is overwritten once every M segments in expectation”, which leads to \(H(\epsilon) = M \cdot L \cdot \frac{\ln(\epsilon)}{\ln(1-\lambda)}\). Could you quantify empirically, on at least one environment (e.g., RememberColor3‑v0 or T‑Maze), how close the actual overwrite frequency is to \(1/M\) for the slots that store task-relevant information? For example, plotting histograms of inter-update intervals would make the relationship between theory and practice more concrete.

4. **On the effect of detaching memory between segments during training.**  
   - You state that memory is detached between segments (no gradient flow across segments). Did you try a variant where you backpropagate through memory, at least for short sequences, and if so, what happened? It would be helpful to understand whether the impressive long-horizon performance emerges purely from local segment-level supervision or whether longer-range gradients might further help (or destabilize) training.

5. **On POPGym aggregation and normalization.**  
   - Could you provide, either in the rebuttal or camera-ready, an alternative POPGym summary that (i) counts per-task wins/losses, and/or (ii) normalizes each task’s return relative to the PPO-GRU expert? This would make it easier to assess how close ELMUR is to the expert and how robust the gains are across tasks with different reward scales.

6. **On transfer to online RL and real robots.**  
   - Have you run any preliminary experiments with online RL training (e.g., PPO or SAC with ELMUR as the policy/value network), even on simpler POMDPs, to assess stability and sample efficiency? Any insights here, even if not fully polished, would help assess how ready the architecture is for the intended real-world use cases.

A clear response to these points, especially 2–4, could strengthen my confidence in both the interpretability of the analysis and the practicality of the method.

---

## Flag For Ethics Review

No ethics review needed.

---

## Details Of Ethics Concerns

N/A.

---

## Soundness Rating

3: good.  
The architecture and experiments are technically solid and convincingly support the main empirical claims. The theory is correct for the simplified setting considered, but its scope and assumptions could be more explicitly delimited.

---

## Presentation Rating

3: good.  
The paper is well written, with clear figures (notably Figures 1–3 and 6, 9–15) and extensive experimental detail, though some parts of the theoretical exposition and POPGym aggregation could be clarified further.

---

## Contribution Rating

3: good.  
The contribution is a well-executed and practically useful refinement of memory-augmented transformers for long-horizon RL/IL, with strong empirical validation across multiple benchmarks. Conceptual novelty is moderate given the rich existing literature on external memory and recurrent transformers, but the combination and thorough evaluation are valuable to the community.

---

## Overall Rating

8: Accept, good paper (poster).  
Despite incremental conceptual novelty, the paper presents a clean, analyzable memory-augmented transformer architecture and backs it up with extensive, carefully executed experiments on challenging memory-intensive tasks, including nontrivial robotic manipulation benchmarks. The empirical evidence that ELMUR maintains 100% success on T‑Maze up to 10⁶ steps (Figure 3) and substantially improves success rates on MIKASA‑Robo (Tables 1 and 8) is compelling. The weaknesses around theoretical scope, missing related work (Memo), and POPGym aggregation are addressable in a revision and do not undermine the central contributions.

---

## Reviewer Confidence

4: confident.  
I am familiar with transformers, external-memory architectures, and RL under partial observability, and I carefully examined the equations, Algorithm 1/2, and key experimental tables/figures. Some aspects of the empirical design (e.g., online RL behavior) remain untested, but this does not affect my main evaluation.