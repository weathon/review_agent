---
job_id: b687b57f-ffe9-4620-b7cd-fb580a41aae7
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: MwuSvrthXq.pdf
paper: Reinforcement Learning for Heterogeneous DAG Scheduling with Weighted Cross-Attention
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies reinforcement learning and graph-based architectures for DAG scheduling, clearly within ICLR’s scope on RL, combinatorial optimization, and learning on graphs.

## Minimum Quality
Pass ✅.  
The paper has all main components (abstract, introduction with related work, methodology, experiments, results, conclusion, and extensive appendix). The methods and experiments are nontrivial and clearly research-level; I do not see fatal methodological errors or evaluation practices that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any text attempting to manipulate automated reviewing or hidden prompts; the content is a normal research manuscript.

---

# Expected Review Outcome:

## Summary

The paper proposes WeCAN, a reinforcement-learning-based scheduler for heterogeneous DAGs with task–pool compatibility constraints. The model uses a weighted cross-attention (WeCA) mechanism to incorporate compatibility coefficients between tasks and resource pools, and a longest-directed-distance GNN (LDDGNN) to encode DAG dependencies. On top of a single-pass, non-autoregressive policy, the authors design a skip action and analyze list-scheduling “generation maps,” arguing that standard list scheduling excludes some optimal schedules and that their skip design can close this optimality gap. Experiments on extended TPC-H and synthetic computation-graph benchmarks show that WeCAN improves makespan compared with several heuristic and neural baselines while having competitive runtime.

## Strengths

1. **Well-motivated heterogeneous setting and compatibility modeling.**  
   The paper tackles heterogeneous DAG scheduling with explicit task–pool compatibility coefficients, a practically relevant setting where many earlier neural schedulers either assume homogeneous resources or only coarse compatibility (e.g., averages over pools). The formulation in Section 2.1 is clear, with a precise MILP definition that makes constraints and the role of \(K_{acc}(v,c)\) explicit.

2. **Architectural design that encodes compatibility in a size-agnostic way.**  
   The weighted cross-attention (WeCA) layer (Section 3.1, Eq. just below Figure 1) multiplies attention values by a diagonal matrix of compatibility coefficients *outside* the softmax. This is a simple but meaningful design that allows variable numbers of pools and task types, and it ensures incompatible pairs with \(K_{acc}=0\) are effectively masked without fixing the embedding dimensionality. The ablation in **Table 3** directly supports the importance of this design: switching to the “inside” version or moving WeCA only to the decoder significantly worsens makespan and relative improvement.

3. **LDD-based GNN for directed acyclic structure.**  
   The LDDGNN (Section 3.1 and Appendix G.2) leverages the longest directed distance \(d_e(v,w)\) to define both attention masks and distance-based bias embeddings in attention. This is technically interesting and, compared to a plain GAT, more directly tailored to DAGs and path lengths. The ablation rows “WeCA + GAT(forward)” and “WeCA + GAT(bi-direction)” in **Table 3** show that replacing LDDGNN by standard GAT leads to 2–4 percentage points worse improvement, giving concrete evidence that the LDD-based attention is not just cosmetic.

4. **Single-pass (non-autoregressive) design with realistic runtime analysis.**  
   The decision to use a non-autoregressive decoder is well justified in Appendix B, with quantitative evidence in **Table 4** and **Table 20** that generation-map computation dominates runtime, and that auto-regressive variants bring at most ~1% makespan gains at more than 10× inference cost and higher memory usage. This aligns with the target applications where throughput and latency matter.

5. **Theoretical perspective on optimality gaps of list scheduling and role of skip actions.**  
   Section 4 and Appendix A provide a fairly detailed formalization of the “reduced space” \(B\), feasible reduced space \(B_f\), and maps \(T, S, S_{list}, S_n\). The counterexample in **Figure 5** illustrates that list scheduling can systematically exclude the optimal schedule because \(TS_{list}\) is not surjective to \(B_f\). Theorems 1 and 2 plus Propositions 2–4 analyze when a generation map’s image necessarily contains an optimal solution and how skip actions help restore surjectivity while keeping variance manageable. This is a nontrivial conceptual contribution beyond incremental heuristic tweaking.

6. **Empirical performance and breadth of evaluation.**  
   On both TPC-H and Computation Graphs, WeCAN is consistently better than or competitive with strong baselines. For example, in **Table 1**, WeCAN-S(256) improves makespan over the best heuristic (HEFT / Tetris) by ~18% on TPC-H-30 and ~13% on TPC-H-100, while also outperforming One-Shot and PPO-BiHyb by appreciable margins. In **Table 2**, WeCAN-Greedy reduces makespan vs HEFT by about 7–8% across graph types, and WeCAN-S(256) further improves it. The runtime numbers in both tables show that greedy WeCAN is similar to or faster than heuristic baselines and much faster than PPO-BiHyb.

7. **Evidence that skip actions matter in heavy-task scenarios.**  
   The heavy-task experiments (Section 5.3, **Figure 3**, and **Table 8**) are a nice, concrete instantiation of the theoretical discussion in Section 4. HEFT, which is not a list scheduler, significantly outperforms list-based heuristics when a small proportion of heavy tasks is injected, while both One-Shot and “WeCAN-No-Skip” roughly match HEFT. Adding skip actions gives “WeCAN-With-Skip” an extra ~8–9% improvement over HEFT, supporting the claim that list-based generation maps have a real performance gap in some structured cases and that the proposed skip design helps.

8. **Generalization experiments to larger and perturbed environments.**  
   The paper includes several generalization studies: scaling from TPC-H-30 to TPC-H-150 and 200 and to more pools (Tables 6–7), and evaluation under various environment fluctuations (Figure 2 and Tables 9–16). These show that a model trained on a particular pool/task setting retains improvements of roughly 8–15% over the best heuristic even when the number or type of pools and tasks change, which is valuable for heterogeneous systems that evolve over time.

## Weaknesses

1. **Skip-action “optimality gap closing” claim is largely existential and somewhat overstated.**  
   The core theoretical guarantee around skip actions is Theorem 1, especially parts (ii) and (iv). However:
   - Part (ii) only states that Algorithm 1 assigns positive *probability* to at least one optimal schedule, not that learning will find or concentrate on it.  
   - Part (iv) exhibits a construction of scores \(\{u_{(v,c)}\}, u_a, u_b, u_c\) under which greedy selection recovers an optimal schedule, but those scores are not the output of the proposed neural architecture, they are arbitrary values chosen with full knowledge of the solution.  
   Hence, while the mathematical analysis of surjectivity of \(TS\) is interesting, the practical claim that the *learned* scheduler “closes the optimality gap” is too strong; the proofs show representational capacity, not optimization behavior. This should be toned down or complemented by learning-theoretic arguments.

2. **Ad hoc design and limited analysis of skip-score parameterization.**  
   The skip score in Section 3.2 is defined as  
   \[
   u_{\pi_{\text{skip}}}=u_a\Bigl(1-\frac{k}{2n}\Bigr)^{u_b}+u_c
   \]
   with \(u_a,u_b\ge 0\) and \(u_c\) from an MLP on averaged embeddings. This choice is not obviously tied to the optimality-gap theory, beyond monotonic decay in \(k\). There is no analysis of how this parameterization affects training stability, variance, or whether it can reliably approximate the implicit waiting times suggested by the projection map \(S_n\). In Theorem 1.iv the authors reverse-engineer scores to force a desired sequence; that construction is entirely disconnected from the specific MLP + power-law form used in practice. The heavy-task experiments in **Figure 3** and **Table 8** confirm that skip helps in that scenario, but we do not see a broader sensitivity or failure analysis (e.g., what happens when heavy tasks are absent, or on other datasets), nor ablations on alternative time-dependent skip parameterizations.

3. **Clarity and rigor issues in mathematical exposition, especially around WeCA and LDDGNN.**  
   Several equations are difficult to parse or contain notation inconsistencies, which matter because the contribution is partly architectural:
   - In the encoder WeCA definition on Page 4, \(K^{e}\) is introduced but the update uses \(K^c\), and the softmax argument and broadcasting over pools are not fully spelled out. It is not entirely clear over which dimension the softmax is taken in  
     \[
     \text{softmax}(q_v^\top K^c)\operatorname{diag}\{K_{acc}(v, c(1)),\ldots\}\, V^c
     \]
     nor how shapes are aligned.  
   - In the LDDGNN equations on Page 5 and Appendix G.2, the expression  
     \[
     (\mathbf{q}_v^{l,j})^\top\mathbf{k}_w^{l,j}\cdot b_{d_e(v,w)}\cdot M_{v,w}^j
     \]
     (main text) and the more complex softmax in Appendix G.2 mix scalar scores, bias vectors \(b_{d_e}\), and masks \(M_{v,w}^j\) without fully specifying dimensions and broadcasting. The relationship between the simplified notation in the main text and the detailed one in Appendix G.2 is not explained, which makes it harder to reproduce the method faithfully.  
   - Assumption 1 and Theorem 2 in Section 4.2 are defined on \(S:B_f \to A\) but the main text does not clearly connect these to the concrete list-scheduling-based implementation and to Algorithm 1; the reader must dig into Appendix A to understand exactly what surjectivity means in practice.  
   These are not fatal errors, but given the weight placed on the theoretical framework and specialized attention structures, a tighter and more precise exposition would be important.

4. **Limited direct evaluation of skip in the main benchmarks.**  
   The skip action is disabled in all “regular” TPC-H experiments (Appendix H.3) because it “has a limited impact” and increases variance. As a result, **Table 1** and **Table 2**, which are the central results, effectively evaluate WeCAN without its main theoretical novelty (the skip-based surjection). The only skip-enabled results in the main paper are the heavy-task ablation (Figure 3 and Table 8). This creates a mismatch between the narrative (“closing the optimality gap of list scheduling using skip actions”) and the actual main-line results, where performance gains are mostly due to the encoder/decoder architecture and not the skip design. For readers who care about optimality-gap closure in standard workloads, it remains unclear how often skip would help versus just adding training variance.

5. **Positioning and baselines for heterogeneous DAG RL methods are incomplete.**  
   Section “Neural DAG Schedulers for Heterogeneous Environments” mentions several RL-based heterogeneous schedulers (Wu et al., 2018; Ni et al., 2020; Grinsztajn et al., 2021; Zhou et al., 2022; Zhadan et al., 2023; Wang et al., 2025). However, none of these are used as baselines, and the discussion of how WeCAN differs architecturally and in capability is fairly high-level. For example:
   - Some prior works already incorporate skip/idling actions (e.g., Mao et al. 2016; Grinsztajn et al. 2021; Zhadan et al. 2023), and Appendix C briefly notes that they lack single-pass efficiency. But the paper does not offer empirical comparison or detailed architectural contrast with the more recent heterogeneous ones (e.g., READYS, multi-agent schedulers), so the incremental benefit of WeCA + LDDGNN versus other graph-based RL schedulers is not fully convincing.  
   - PPO-BiHyb and One-Shot are strong DAG baselines, but there is no direct RL baseline specialized for heterogeneous environments with compatibility coefficients, even though such works exist. At minimum, a more nuanced discussion is needed in the main text about why these baselines were omitted and how the proposed method compares conceptually.

6. **Dataset and realism limitations for heterogeneous settings.**  
   While TPC-H and computation graphs are reasonable benchmarks, the heterogeneous structure is largely synthetic:
   - TPC-H compatibility coefficients and additional resource dimensions are hand-designed (Appendix D.1) rather than derived from real heterogeneous clusters, and there is no empirical study to show that these coefficients correspond to realistic heterogeneity or contention patterns.  
   - Computation Graphs similarly receive synthetic compatibility matrices.  
   - The model ignores communication costs between tasks and pools or data locality, which are central in real distributed DAG scheduling. Appendix K acknowledges this, but the main text still makes some broad claims about applicability in ML compilers and heterogeneous clusters.  
   The empirical gains are valuable, but the paper would be stronger with more discussion of what aspects of real systems are captured or missed by the chosen synthetic heterogeneity.

7. **Experimental setup leaves some unanswered questions about fairness and robustness.**  
   Several choices in the experimental design deserve deeper analysis:
   - All methods use 3 pools; we only see larger pool counts in generalization experiments (up to 12 pools in Table 7), but those are tested with models trained on 3-pool settings. There is no “in-distribution” training/evaluation on truly large heterogeneous environments, which would better validate scalability of WeCA.  
   - The choice of different pool-selection rules for heuristics (EFT, Tetris score, balance choosing) is appropriate, and the best is reported, but this also amplifies the gap between list-based heuristics and HEFT, which uses a different generation map. It would be useful to see analogous variants for WeCAN or at least more explicit discussion of how the heuristics were tuned.  
   - For RL methods, only One-Shot uses an architecture very similar in spirit to WeCAN. It would be informative to see WeCAN trained using the same number of samples and optimization budget as One-Shot, and with or without the advanced LDDGNN, to disentangle representational vs. training advantages.

8. **Theoretical framework is conceptually heavy relative to its practical impact.**  
   The reduced space \(B\), feasible reduced space \(B_f\), and projection maps \(S_n\) and \(S\) are introduced in Section 4 and Appendix A with detailed propositions (e.g., Proposition 3 and Theorem 3) and illustrations in **Figures 4–7**. While mathematically interesting, this machinery is used primarily to justify a relatively simple conclusion: allowing skip actions makes \(TS\) surjective and hence the optimal solution is representable, and list scheduling may not have this property. The practical guidance derived from this (beyond “you should support skipping”) is limited; for instance, the local search approach in Appendix A.5 is discussed but evaluated only in Appendix F.4 and is much slower. Some of the theoretical results could be streamlined and more tightly tied to concrete design decisions or ablation studies.

9. **Missing or underdeveloped discussion of limitations and computational cost.**  
   Training times in **Table 19** are substantial (up to 26 hours for TPC-H-100) on powerful hardware, and even more for computation-graph datasets. There is little discussion of the tradeoff between offline training cost and online inference gains, or about how the method scales beyond ~1000 tasks. Similarly, Appendix H.3 notes that skip is disabled on regular datasets because it increases makespan variance, but the impact of that variance on training stability and convergence is not quantified. These aspects are important for practitioners considering adopting the method.

10. **Minor clarity issues in figures.**  
    - **Figure 1** attempts to show both the WeCAN architecture and the generation-map / solution-space picture in a single figure. The right-hand schematic (“with skip action” vs “list scheduling”) is visually small and hard to parse, and the mapping between the pictured spaces and the formal definitions of \(A, B, B_f\) is not fully explained in the caption.  
    - **Figures 5 and 6** illustrate the counterexample and mapping spaces, but the color coding and annotations (“projection”, “TS\_list”, etc.) are somewhat terse. Since these figures are central to the theoretical message about optimality gaps, clearer captions and references from Section 4 would help.

Overall, the paper has solid ideas and impressive empirical results, but the main theoretical claims about closing optimality gaps are weaker than presented, some mathematical expositions and notations are muddy, and the experimental methodology could better isolate the contributions of skip actions and LDDGNN/WeCA relative to earlier heterogeneous DAG schedulers.

## Potentially Missing Related Work

The following related works appear closely relevant and are not cited or discussed in the main paper:

1. **R. Zhou, H. Zou, L. Zhou, “A Learning Method with Gap-Aware Generation for Heterogeneous DAG Scheduling”, 2026.**  
   - This work appears to focus on heterogeneous DAG scheduling with an explicit “gap-aware” generation mechanism, conceptually very close to the paper’s theme of addressing list-scheduling optimality gaps under compatibility constraints.  
   - It should be discussed in the related work section (likely Section 1 / “Neural DAG Schedulers for Heterogeneous Environments”) as a directly comparable method, clarifying conceptual and methodological differences and, if feasible, adding an empirical comparison on at least one benchmark.

2. **B. Soykan, “HGT-Scheduler: Deep Reinforcement Learning for the Job Shop Scheduling Problem via Heterogeneous Graph Transformers”, 2026.**  
   - This introduces heterogeneous graph transformers for scheduling, which is thematically linked to the proposed LDDGNN + WeCA architecture.  
   - A brief discussion in the context of Section 3.1 would help position WeCAN relative to other heterogeneous graph-based RL schedulers, including differences in how resource/task types and compatibilities are modeled.

3. **Y. Ai, H. Li, H. Ruan, “A Heterogeneous Graph Neural Network Assisted Multi-Agent Reinforcement Learning for Parallel Service Function Chain Deployment”, 2025.**  
   - This paper addresses scheduling/resource allocation using heterogeneous graph neural networks and multi-agent RL.  
   - It should be mentioned in the heterogeneous scheduling RL discussion (Section 1 / “Neural DAG Schedulers for Heterogeneous Environments”) as another example of graph-based RL for complex resource allocation, with a short comparison of the graph modeling and scalability claims.

4. **C. Sun, L. Zhou, Z. Wen, “A Scheduling Algorithm Based on Reinforcement Learning for Heterogeneous Environments”, 2022.**  
   - Focuses on RL-based scheduling in heterogeneous environments, likely sharing similar goals regarding adaptability across diverse pool/task configurations.  
   - It should be cited alongside Wu et al. (2018), Ni et al. (2020), Wang et al. (2025), etc., and the authors should clarify whether that algorithm supports compatibility matrices, skip actions, and scalable encoding of heterogeneous resources.

5. **L. Zhou, C. Sun, Z. Wen, “A Reinforcement Learning Based Job Scheduling Algorithm for Heterogeneous Computing Environment”, 2023.**  
   - Another RL-based scheduler in heterogeneous environments, apparently using a two-stage structure (task selection + processor allocation), which is conceptually close to WeCAN’s separation between scoring and generation map.  
   - It should be compared and contrasted in the neural heterogeneous-scheduler discussion, especially regarding how compatibility is embedded and whether their approach can generalize over varying pools and task types.

6. **R. Zhou, H. Zou, L. Zhou, “Dynamic Heterogeneous Graph Combined with Reinforcement Learning for Solving Job Shop Scheduling Problem”, 2026.**  
   - Proposes dynamic heterogeneous graph representations for RL-based scheduling. This is closely related to the idea of LDDGNN and cross-attention across tasks and resources.  
   - A short discussion in Section 3 or the related work part of the introduction would help situate LDDGNN relative to dynamic heterogeneous graph approaches and clarify the novelty in how LDD and compatibility are leveraged.

For all of these, even if code is unavailable, a conceptual comparison and high-level positioning would improve the paper’s contextualization.

## Questions

1. **Skip-action scope and practical use:**  
   - In regular TPC-H and Computation Graph experiments, skip is disabled due to variance concerns. Could the authors provide a concrete quantitative analysis (e.g., variance of makespan and training curves with and without skip) on at least one of these datasets so that we can understand how big the variance penalty is and whether some tuning (e.g., regularizing \(u_a, u_c\) or clamping skip frequency) could make skip feasible more broadly?

2. **Ablation on LDDGNN vs simpler DAG encoders.**  
   - Table 3 compares LDDGNN to GAT variants, but both still use graph attention. How does WeCAN perform if the dependency encoder is removed or replaced by a simpler topological-embedding MLP or just positional encodings (e.g., depth levels)? This would help justify the additional complexity of LDDGNN.

3. **Clarification of WeCA implementation details.**  
   - Please clarify the exact tensor shapes and normalization of the WeCA update. For the encoder layer, over what axis is the softmax applied (over pools for each task, or over tasks for each pool), and how is the diagonal matrix of \(K_{acc}(v,c)\) implemented efficiently? A pseudocode fragment or more explicit equation in the main text (not just Appendix G) would be very helpful for reproducibility.

4. **Baseline selection for heterogeneous RL schedulers.**  
   - Could you elaborate on why methods like READYS (Grinsztajn et al., 2021) or the heterogeneous RL schedulers by Zhou et al. (2022, 2023) and Zhadan et al. (2023) were not included as baselines? Were there implementation barriers, or do they not support the specific compatibility setting? A short discussion or an approximate re-implementation (e.g., using their state encoding on your datasets) would strengthen the empirical comparison.

5. **Realism of compatibility matrices.**  
   - The compatibility coefficients for TPC-H and Computation Graphs (Appendix D.1–D.2) are manually crafted. Could the authors comment on how these values relate to real-world performance differences in heterogeneous clusters (e.g., CPU vs GPU vs memory-optimized pools)? Have you tried calibrating them against any real system measurements, or could you at least show sensitivity plots where the coefficients are perturbed?

6. **Effect of heavy-task ratio beyond the tested points.**  
   - Figure 8 considers fractions of heavy tasks up to 3.2%. Does the performance of skip-enabled WeCAN and HEFT cross over or saturate at higher proportions of heavy tasks? Additional points (e.g., 5–20%) might better illustrate the regimes where the optimality-gap issue is truly dominant.

7. **Scaling limits and training cost.**  
   - With training times up to 26 hours for TPC-H-100 (Table 19), what is the practical scaling limit of WeCAN in terms of task count and pool count on available hardware? Have you attempted training on graphs with, say, 5k or 10k tasks, and if so, what broke first (memory, runtime, variance)?

The authors’ responses to these questions, especially around skip-action practicality, WeCA implementation, and baseline choices, could materially influence my assessment.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The work focuses on scheduling algorithms for computational resources and does not raise obvious concerns about human subjects, safety, fairness, or misuse.

## Soundness Rating

2: fair.  
The main algorithmic ideas and experiments appear largely correct and well executed, but some theoretical claims (closing optimality gaps) are only existential; the skip-score parameterization is somewhat ad hoc; several mathematical definitions (WeCA, LDDGNN) are under-specified, which prevents fully verifying and reproducing the method.

## Presentation Rating

2: fair.  
The paper is generally readable and thorough, but the theoretical section is heavy and not tightly connected to practice, some notation and equations are confusing or inconsistent, and several central figures (especially Figures 1, 5, 6) could be clearer. An explicit related-work section and more structured comparisons would help.

## Contribution Rating

2: fair.  
The combination of WeCA, LDDGNN, and a single-pass RL scheduler for heterogeneous DAGs is useful and shows solid empirical gains, but the architectural novelty is incremental over existing transformer/GNN ideas; the strongest theoretical contribution (surjectivity with skip actions) has limited practical follow-through; and positioning relative to prior heterogeneous RL schedulers is incomplete.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper offers an interesting and well-performing system for heterogeneous DAG scheduling and contains a thoughtful analysis of list-scheduling optimality gaps. However, the “gap closing” via skip actions is supported mainly by existential representation arguments and heavy-theory apparatus, while skip is disabled in the main benchmarks. Clarity issues in the math and incomplete comparison to existing heterogeneous RL schedulers further weaken the case. With stronger empirical isolation of skip’s impact, clearer and more concrete theoretical exposition, and more thorough positioning against related work, this line of work could reach ICLR level.

## Reviewer Confidence

4: confident.  
I am familiar with RL for combinatorial optimization and neural scheduling, and I carefully examined the math, algorithms, and experimental setup, though I did not attempt to re-implement the method.