Now I have enough information to write the final review. Let me synthesize.## Summary
WeCAN is an end-to-end reinforcement learning framework for heterogeneous DAG scheduling with task-pool compatibility coefficients. The paper makes two primary technical contributions: (1) a weighted cross-attention (WeCA) layer that embeds compatibility coefficients multiplicatively outside the softmax, enabling adaptable environment-aware task embeddings; and (2) a theoretical analysis of the optimality gap in list scheduling, paired with a parametric skip-action mechanism adapted for single-pass inference to close that gap. Evaluations on TPC-H and Computation Graphs benchmarks show 7–18% gains over heuristics and 7–9% gains over included neural baselines, with competitive inference speed.

---

## Strengths

- **WeCA outside-softmax design is technically well-justified and empirically validated.** The paper provides a crisp two-pool, two-task example showing why placing Kacc inside the softmax collapses embeddings of tasks with different compatibility profiles. Table 3 confirms this is not a minor detail: "WeCA-final-only" drops to +0.5% over Tetris on TPC-H-30, while WeCA+LDDGNN achieves +14.0%. The ablation decisively supports the design.

- **Formal optimality-gap analysis is a genuine contribution.** Theorems 1 and 2 establish that list scheduling's generation map is not surjective over the optimal solution space, and that the skip-augmented map closes this gap. The constructive proof and the criterion (Assumption 1) provide a principled framework that prior neural scheduling work lacked.

- **LDDGNN outperforms standard GNN alternatives on directed dependency structure.** Table 3 shows a systematic ordering: WeCA+LDDGNN (14.0%) > WeCA+GAT(forward) (10.5%) > WeCA+GAT(bidirectional) (9.9%), consistent across both TPC-H-30 and TPC-H-50. The LDD-based attention bias is a targeted design for DAGs.

- **Practical inference speed is a real advantage.** WeCAN-Greedy runs at 0.15–1.72s across TPC-H datasets, comparable to heuristics and orders of magnitude faster than PPO-BiHyb (20–179s), while producing better makespan. For time-sensitive scheduling, this tradeoff is operationally meaningful.

- **Generalization to varying environments is demonstrated.** Figure 2 shows sustained improvement over best heuristics under variation in pool number, pool type, task number, and task type at test time — supporting the adaptability claim of the WeCA architecture.

---

## Weaknesses

### Fatal
None.

### Major

- **Two directly relevant RL baselines are absent from all comparison tables.** Section 2.1 explicitly characterizes Zhadan et al. (2023) and Wang et al. (2025) as RL methods for heterogeneous DAG scheduling that use averaging of compatibility coefficients — the exact limitation WeCAN claims to address. Yet neither appears in Tables 1 or 2. The included RL baselines, PPO-BiHyb (Wang et al., 2021) and One-Shot (Jeon et al., 2023), either solve a structurally easier problem (One-Shot ignores compatibility coefficients entirely, as the paper itself notes) or use a radically different and slower inference paradigm (PPO-BiHyb with beam search at 20–179s vs. WeCAN at sub-5s). The headline "7.7% improvement over the best neural baseline" is therefore against competitors that do not operate in the same problem setting. Without Zhadan et al. (2023) and Wang et al. (2025), the claim of state-of-the-art RL performance for heterogeneous DAG scheduling is not established.

### Minor

- **Skip action's practical benefit on standard benchmarks is not isolated.** The skip ablation (Figure 3) uses a dataset specifically modified to contain 1% heavy tasks — a setting engineered to exhibit list scheduling's failure mode. Tables 1 and 2 have no "WeCAN-no-skip" row, making it impossible to quantify how much the skip mechanism contributes under typical (unmodified) scheduling conditions versus how much comes from the WeCA+LDDGNN architecture. Given that the skip analysis is one of the paper's two headline contributions, this gap is worth addressing.

- **Ablation sample size is small.** Table 3 reports results from 10 test problems per condition. With standard deviations on the order of 0.2%, differences of 0.5–4% between ablation variants may not be statistically robust. A modest increase in test set size or a paired significance test would strengthen these comparisons.

- **Main experiments use only 3 resource pools.** The heterogeneous scheduling setting the paper targets is motivated by environments with varying pool counts, but all Tables 1 and 2 results use exactly 3 pools. The generalization experiment in Figure 2 tests varying pool counts but only reports relative improvement over heuristics, not absolute comparisons to RL baselines. This limits the scope of empirical evidence for the scalability claim.

### Trivial
None beyond the usual.

---

## Nice-to-Haves

- Move the auto-regressive vs. non-auto-regressive quality comparison from Appendix B into the main paper, with a table showing numbers. The non-AR design is the core speed-quality tradeoff of the approach; quantifying the quality cost would help readers understand where the design lands in the Pareto frontier.
- Provide training convergence curves with and without skip to substantiate the variance-reduction claim ("clusters poor solutions in high-ua, high-uc region") made in Section 4.
- A concrete schedule visualization on a heavy-task DAG instance would make the theoretical gap analysis in Section 4 tangible.

---

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Non-AR decoder "mischaracterized as pure engineering win"** (Harsh Critic, structural point): The paper explicitly states in Section 3.2 "comparison with auto-regressive one in Appendix B" and discusses the tradeoff. The appendix exists even if not parsed. The concern about missing main-body quantification is a presentation preference, not a factual mischaracterization. Downgraded to Nice-to-Have.
- **Formula for skip score lacks formal derivation** (Harsh Critic): The formula uπ_skip = ua(1 − k/2n)^ub + uc is a parametric design choice for a learned module. Demanding a formal derivation for a neural architecture component is outside community norms for empirical RL papers; the paper provides intuitive justification and empirical validation (Figure 3).
- **REINFORCE convergence guarantee absent** (Harsh Critic): Demanding convergence proofs for REINFORCE in an empirical systems paper is not standard in this community. Moved to Nice-to-Have.
- **Small test set for Figure 2** (Harsh Critic): The generalization experiment is presented as a robustness check, not a primary result. Criticizing its sample size without evidence of instability is insufficient to sustain as a weakness.
- **Large-scale results in Appendix F** (Harsh Critic): Per rules, appendix content exists in the original submission. Not a valid criticism.

---

## Novel Insights

The most non-obvious insight in this paper is the combination of a **surjectivity criterion** (Assumption 1) as the formal condition for a generation map to guarantee access to the optimal solution, together with the observation that the naive surjective map (Sn) concentrates training variance badly while the skip-augmented list scheduling map clusters poor solutions in a identifiable region of the parameter space. This framing converts what is usually treated as an engineering heuristic (adding skip/wait actions) into a theoretically grounded design decision. The characterization is crisp and more actionable than prior work on the subject.

---

## Evaluation on Key Axes

**Originality:** Moderate-to-good. The WeCA outside-softmax placement and the skip-augmented surjection analysis are specific and motivated innovations, not incremental variations on known architectures. The LDDGNN is an application of known ideas (Graphormer-style biases) to a new domain.

**Importance of research question:** Good. Heterogeneous DAG scheduling is practically important, and the gap between current RL methods and heuristics in this space is real.

**Claim support:** Partially strong. The WeCA design and LDDGNN contributions are well-supported by ablations. The "state-of-the-art RL" claim is not adequately supported given the absence of Zhadan et al. (2023) and Wang et al. (2025).

**Experimental soundness:** Adequate for the included comparisons; limited by the missing baselines and the skip ablation being restricted to an engineered scenario.

**Clarity:** Good. The paper is clearly written with well-structured technical content.

**Value to community:** Moderate. The theory of optimality gaps in list scheduling is a useful framework; the WeCA design and LDDGNN are reusable components.

---

## Calibration Anchors

| Paper | Avg Score | Comparison |
|---|---|---|
| jsWCmrsHHs (DRL improvement heuristic for JSSP) | 7.5 | Stronger: comprehensive baselines vs. all state-of-the-art DRL methods, clean claim support. WeCAN is weaker on baseline coverage. |
| Aly68Y5Es0 (L-RHO for FJSP) | 6.75 | Comparable in methodology quality; WeCAN slightly weaker on breadth of experimental validation. |
| z2z9suDRjw (GOAL generalist CO) | 6.25 | Comparable; GOAL's generalist scope is broader, WeCAN's theory is deeper. |
| VIEbRFp6s3 (MARL datasets) | 5.8 | WeCAN is technically more novel. |
| ziB549CQ30 (Fuzzy JSSP) | 3.5 | WeCAN is clearly superior: real architecture novelty, theoretical analysis, solid ablations. |
| bntJK4NyIW (Decentralized training) | 2.0 | WeCAN is significantly stronger in every dimension. |

WeCAN's technical quality places it above the 5.0–5.8 cluster but below the 6.75–7.5 band due to the missing RL baseline comparisons (the exact gap that separates incomplete from comprehensive empirical validation in this subfield). A score of **5.5** is appropriate, reflecting real contributions constrained by a significant experimental gap.

## Score and Decision

**Score: 5.5**  
**Decision: Reject (Borderline)**

The paper has real technical contributions — the WeCA design, LDDGNN, and the optimality-gap framework are all concrete and well-supported. However, the absence of Zhadan et al. (2023) and Wang et al. (2025) as baselines — methods the paper itself characterizes as the closest prior RL work for the exact problem setting — prevents verification of the central empirical claim. Addressing this gap in a revision would likely bring the paper into acceptance range.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>