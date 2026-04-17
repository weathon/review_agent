---
job_id: 2b18d28c-fa97-43d7-8924-95fa3cfac2b1
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: Mz98kwANpF.pdf
paper: Align, Don’t Divide: Revisiting the LoRA Architecture in Multi-Task Learning
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is on parameter-efficient fine-tuning, multi-task learning, and representation alignment for LLMs, all squarely within ICLR’s scope on representation learning, transfer/MTL, optimization, and learning theory.

## Minimum Quality
Pass ✅.  
The paper is complete (Abstract, Introduction, Related Work, Method, Experiments/Results, Theory, Conclusion) and written in clear English. It proposes a concrete method (Align-LoRA), provides reasonably extensive experiments and a theoretical bound. I do not see fatal methodological or evaluation flaws that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no attempt to manipulate automated reviewing systems or hidden prompts; the content is standard scientific prose.

---

# Expected Review Outcome:

## Summary

The paper revisits multi-task LoRA architectures and questions the dominant trend of multi-adapter/multi-head designs with input-dependent routing. Through a simplified variant M-LoRA, the authors show that removing the router and allowing highly similar heads can improve performance over more complex multi-head methods such as HydraLoRA and R-LoRA. They further show that a single high-rank LoRA adapter can match or surpass these multi-component architectures when parameter budgets are matched, motivating a focus on task-shared representations. Building on this, they propose Align-LoRA, which keeps a single high-rank adapter but adds a representation alignment loss (instantiated via symmetric KL or MK-MMD) on the low-rank $\mathbf{A}$ outputs to encourage task-shared features, and demonstrate improved multi-task generalization and adaptation, alongside a domain-adaptation-style generalization bound.

## Strengths

1. **Clear and focused empirical challenge to a popular design pattern.**  
   The paper makes a sharp and empirically grounded point: multi-head LoRA variants with dynamic routing and enforced diversity are not obviously better than much simpler configurations. In Table 1 (Page 5), the proposed M-LoRA outperforms both HydraLoRA and R-LoRA on all five tasks while using fewer trainable parameters, and Figure 2 (Page 4) shows that M-LoRA’s heads have *higher* inter-head cosine similarity (>0.85 median) despite better accuracy. Together, these elements persuasively undermine the intuitive “diversity is always good” story for multi-head LoRA and are a valuable contribution on their own.

2. **Strong case that capacity (rank) often matters more than architectural ornamentation.**  
   Section 4 and Tables 2–3 (Page 6) show that when the parameter count is matched, a single high-rank LoRA adapter (e.g., LoRA† with rank 30 on LLaMA2) matches or improves upon LoRAHub, LoRA MoE, HydraLoRA, and R-LoRA across BBH-style evaluation, for both LLaMA2 and Qwen2.5. This is a clean, interpretable experiment that many practitioners will appreciate: it provides a concrete sanity check against increasingly baroque multi-adapter designs.

3. **Align-LoRA is simple, practical, and preserves LoRA’s deployment advantages.**  
   The proposed Align-LoRA keeps the single-adapter structure and adds only an auxiliary loss on the low-rank $\mathbf{A}$ outputs, with no extra parameters and no inference overhead. Section 5.1 gives a straightforward formulation: per-task Gaussian modeling of $\phi_{T_i}(\mathbf{x}) = \mathbf{A} X_{T_i}$ (Eq. 4), symmetric KL across tasks (Eq. 5), and adding $\lambda \mathcal{L}_{\text{align}}$ to the LM loss (Eq. 6). Appendix C makes the inference-merging story fully explicit with Eq. (7) vs. Eq. (8). This is a minimal modification that many real systems could adopt.

4. **Consistent empirical improvements across multiple settings and scales.**  
   Align-LoRA-K (KL) and Align-LoRA-M (MK-MMD) consistently outperform strong baselines:
   - On BBH (Table 4, Page 8), A-LoRA-K gives the best performance for Qwen2.5-7B, LLaMA3-8B, and Qwen2.5-14B, often with fewer trainable parameters than multi-component variants.  
   - On the 8-task in-domain benchmark (Table 5, Page 8), A-LoRA-K yields the highest average scores for both 3B and 7B models, again with lower %Param than LoRAMoE and HydraLoRA.  
   - On highly dissimilar FlanV2-derived tasks (Table 8, Page 21), A-LoRA-K still slightly but consistently improves over high-rank LoRA and over M-LoRA, suggesting the approach is not limited to closely related tasks.

5. **Empirical insight into how multi-head dropout interacts with routing.**  
   Section 3.3 and Figure 4 (Appendix B, Page 14) provide an interesting architectural insight: keeping multi-head dropout but *removing* routing converts the heads from competing specialists into a collaborative ensemble, leading to higher performance (M-LoRA) than both routed HydraLoRA and its “w/o Router” ablation. This is not a trivial observation and could inform future multi-task adapter designs beyond this specific paper.

6. **Some theoretical grounding, even if high-level.**  
   Section 5.3 and Appendix F present a domain-adaptation-like generalization bound (Eq. (6) in Appendix F), where Align-LoRA’s reduction of cross-task discrepancy $\Delta(\mathcal{D}_i,\mathcal{D}_j)$ tightens the MTL risk bound. While not deeply specialized to the exact training dynamics, this at least connects the alignment loss to a standard generalization-analytic perspective.

7. **Visualization and hyperparameter analysis that support the alignment story.**  
   - Figure 3 (Page 9) shows performance as a function of $\lambda$, highlighting that moderate alignment strength yields stable improvements, and too large $\lambda$ leads to “over-alignment” and degraded accuracy.  
   - Figure 5 and the t-SNE plots (Appendix I.1, Page 22) qualitatively show that Align-LoRA compresses task clusters in representation space without collapsing them, visually supporting the “shared but not identical” representation narrative.

8. **Training efficiency and module-agnostic applicability.**  
   Table 6 (Appendix D, Page 14) indicates that Align-LoRA *reduces* total FLOPs and training time compared to HydraLoRA while achieving substantially better performance (80.06 vs 76.80), attributable to fewer trainable parameters. Table 7 (Appendix H.1, Page 21) shows that applying Align-LoRA-K *only* to attention layers still gives a strong gain over both basic LoRA and HydraLoRA, suggesting the method is not tightly coupled to a specific module subset.

## Weaknesses

1. **Novelty and positioning relative to prior MTL/LoRA work are somewhat overstated and incomplete.**  
   The core idea of aligning task representations in a shared space is well known from multi-domain and multi-task literature (e.g., Ben-David et al., 2006; Pan et al., 2010; Long et al., 2015; Chen & Cardie, 2018; Hu et al., 2025, some of which are cited). In the LoRA and LLM-PEFT ecosystem, there are also multiple recent MTL LoRA variants that explore shared vs task-specific structure and gating (MTLoRA, MeTA-LoRA, Gated LoRA, L-LoRA, etc.). Some are not cited in the main text despite clear relevance. This makes the statement “this is the first work to systematically apply statistical distance metrics for this purpose within the multi-task LoRA framework” (Section 5.1) feel somewhat strong given how close the setup is to classic domain adaptation applied to shared-parameter MTL, even if the exact “apply KL/MMD on the $\mathbf{A}$ outputs of a single LoRA” detail is new. The authors should more carefully acknowledge that the contribution is a clean *instantiation* of known alignment ideas in a LoRA context, rather than conceptually new.

2. **The alignment loss formulation and its practical implementation details are under-specified.**  
   In Eq. (5), each task distribution $p_{T_i}$ is modeled as a diagonal Gaussian over the batch of $\phi_{T_i}(\mathbf{x})$, but critical choices are glossed over:
   - How large is each per-task batch, and are all tasks present in every global batch? If not, how is the pairwise sum over $i<j$ approximated in practice?  
   - Are the means and variances computed per-layer per-task and then averaged, or over all adapter positions jointly?  
   - For MK-MMD (Eq. (9) and Eq. (11)), what kernel bandwidths are used, how many kernels are in $\mathcal{K}$, and are they tuned per model?  
   These implementation details materially affect stability and computational cost. Without them in the main text, reproducing the exact curves in Figure 3 or the results in Tables 4 and 5 is difficult. At least brief concrete descriptions in Section 5.1 (not only Appendix E) would make the method more self-contained.

3. **The theoretical analysis is very high-level and almost entirely generic.**  
   Section 5.3 and Appendix F derive a standard style of domain-adaptation / multi-task bound:  
   \[
   R_{\text{MTL}}(f)\le \frac{1}{M}\sum_i R_{\text{train}}(f;\hat{\mathcal{D}}_i)+\frac{\lambda}{M}\sum_{i<j} \Delta(\mathcal{D}_i,\mathcal{D}_j) + O\Big(\sqrt{\tfrac{\log(1/\delta)}{n_{\text{total}}}}\Big).
   \]
   However:
   - The bound does not depend on specifics of the LoRA architecture, nor on the fact that alignment is applied *only* to the low-rank $\mathbf{A}$ space. Any representation-alignment method on top of a shared encoder would give the same form.  
   - The derivation repeatedly introduces constants like $\Lambda$ and later “sets” $\lambda = \Lambda/(2M)$ without quantifying or estimating them.  
   - The key inequality in Eq. (3) of Appendix F, $\sum_i \Delta(\mathcal{D}_i,\hat{\mathcal{D}})=\frac{1}{2M}\sum_{i,j}\Delta(\mathcal{D}_i,\mathcal{D}_j)$, is asserted based on “convexity of KL and linearity of the centroid,” but there is no rigorous derivation, and it is not obvious that symmetric KL or MK-MMD satisfy the exact identity stated (as opposed to an inequality or bound).  
   - Most importantly, the analysis stops at “if $\Delta$ is smaller, the bound is tighter,” which is tautological and does not, for example, explain *why* aligning the *down-projection* space is preferable to aligning hidden states or logits.  
   So the “theoretical superiority” claim is much stronger than what is actually proven. The theory section is fine as intuition, but its current presentation risks overselling the formal guarantee.

4. **Limited introspection into per-task tradeoffs and negative transfer.**  
   The central premise is that encouraging shared representations is beneficial, but MTL is known to sometimes suffer from negative transfer, particularly when tasks are heterogeneous. While Table 8 (highly dissimilar FlanV2 clusters) and Table 4 (BBH) show improved *average* performance, the paper almost never reports *per-task* gains and losses beyond a handful of benchmarks.  
   For example, in Table 5 some tasks see relatively modest benefit from A-LoRA-K vs M-LoRA (e.g., certain tasks differ by ≤0.2–0.3 points), and in a few cases the A-LoRA-M variant slightly underperforms M-LoRA, but this is not discussed analytically. Is there any task where alignment significantly hurts performance, especially for outlier domains like translation vs QA vs math? A more thorough breakdown, perhaps including per-task deltas relative to LoRA and R-LoRA, would clarify when the alignment principle is safe and when it may be too aggressive.

5. **Head-similarity analysis is suggestive but shallow.**  
   Figure 2 presents cosine similarity distributions of $\mathbf{B}_i$ heads across down_proj, up_proj, and gate_proj for HydraLoRA, R-LoRA, and M-LoRA. The key empirical statement is that M-LoRA has the highest similarity yet best performance. However:
   - Similarity is computed simply by flattening $\mathbf{B}_i$ and taking cosine similarity. This conflates scale and orientation and ignores any functional equivalence modulo layer normalization or scaling factors.  
   - The analysis is aggregate: one boxplot per module type across the entire model. It does not show how similarity changes with depth or correlates with task-specific performance.  
   - No statistical tests are run to confirm that the observed differences in similarity distributions are significant relative to noise.  
   While this does not invalidate the core empirical findings in Table 1, the current analysis is more of a visual anecdote than a deep exploration of “why” head redundancy helps.

6. **Some missing ablations to properly disentangle components.**  
   The paper makes several intertwined claims: (1) rank increase is powerful, (2) single-adapter is enough, (3) alignment on $\mathbf{A}$ is key. However, a few fairly natural ablations are missing from the main text:
   - Align on other spaces: e.g., apply KL/MMD to the *output* of $\mathbf{B}\mathbf{A}$ or to hidden states of the transformer, to verify that the choice of $\mathbf{A}$ is indeed better than generic hidden alignment.  
   - Vary rank with alignment: Tables 2–3 treat rank as fixed per method group, but there is no explicit sweep such as “LoRA+alignment at ranks 4, 8, 16 vs LoRA at the same ranks” to show that gains are not purely due to increased capacity in Align-LoRA (for some settings it uses rank 8 vs rank 4 for multi-head methods).  
   - Interaction with routing: Appendix I shows “R-LoRA+Align” and “M-LoRA+Align” (Table 9, Page 22) benefitting from the same alignment term, which is nice, but the main paper largely ignores this and thus underplays the possibility that some hybrid (router + alignment) could be competitive or superior.
   These missing ablations do not undermine the main claims but reduce the sharpness of the architectural conclusions.

7. **Some mathematical and notation issues.**  
   There are several places where mathematical precision could be improved:
   - In Eq. (4), $\phi_{T_i}(\mathbf{x}) = \mathbf{A} \cdot X_{T_i}$, it is unclear whether $X_{T_i}$ is a token-wise matrix, a sequence representation, or some pooling over tokens. Later, in the Gaussian modeling, the object over which mean and variance are computed (tokens vs sequences vs batch elements) is not specified.  
   - Eq. (5) defines $\mathcal{L}_{\text{KL}}$ as a sum of symmetric KL between Gaussians, but there is no explicit closed form, nor confirmation that the per-dimension variances are kept strictly positive (practical implementations usually require adding $\epsilon$ to avoid numerical issues).  
   - In Appendix F, Eq. (1) uses $\Delta(\mathcal{D}_i,\bar{\mathcal{D}})$, then later in Eq. (6) the bound is written with $\Delta(\mathcal{D}_i,\mathcal{D}_j)$ and a different $\lambda$, which could confuse readers; consistent notation and explicit assumptions on $\Delta$ (metric vs divergence vs pseudo-metric) would help.  
   - In Appendix F.8, the final bound mentions $\alpha_{\text{total}}$ instead of $n_{\text{total}}$, which looks like a typo in a key theorem statement.  
   Separately, the MMD-based loss in Eq. (9) uses expectations over $p_{T_i}$ and $p_{T_j}$ but then writes $\phi_{T_i}(\mathbf{x})$ as the embedding and $\phi(\cdot)$ as the RKHS map; the two $\phi$’s are overloaded, which is confusing.

8. **Comparative framing slightly underplays strong baselines’ strengths.**  
   While the empirical results show consistent improvements, some of the margins are modest and might be sensitive to tuning. For instance:
   - In Table 4, on Qwen2.5-7B, LoRA is 48.36 vs A-LoRA-K 50.28, which is a nice gain, but R-LoRA and M-LoRA are already at 48.32 and 48.44.  
   - In Table 5 for Qwen2.5-7B, M-LoRA is at 82.46 avg vs A-LoRA-K at 83.95; the difference is substantial but not huge, and some tasks (e.g., Task5) are identical (95.93).  
   The paper sometimes uses language like “significantly surpasses” or “superior performance” without reporting confidence intervals, multiple seeds, or standard deviations. With such small absolute gaps, random seed variation or more extensive tuning of R-LoRA/HydraLoRA might close part of the margin. At least a brief discussion acknowledging this sensitivity, or reporting mean±std over several runs for a subset of settings, would make the empirical claims more robust.

Overall, these weaknesses are not fatal, but they temper the level of confidence in the strongest theoretical claims and limit how “foundational” the contribution is relative to prior MTL alignment literature.

## Potentially Missing Related Work

1. **Bo Cheng et al., “MeTA-LoRA: Data-Efficient Multi-Task Fine-Tuning for Large Language Models” (2025).**  
   This paper explicitly addresses multi-task LoRA for LLMs and introduces a two-stage optimization for data-efficient multi-task adaptation. It is directly relevant to Sections 2.2 and 3, which discuss multi-task LoRA architectures and training strategies. It should be cited in the Related Work on multi-component/multi-task LoRA and briefly compared experimentally or at least conceptually, since both works tackle multi-task adaptation efficiency for LoRA.

2. **SooHwan Eom et al., “Gated LoRA: Dual-Purpose Projections for Parameter-Efficient Mini-Expert Fine-Tuning” / “Gated LoRA: Input-Dependent Activation for Multi-Task Fine-Tuning” (2025).**  
   These works introduce input-dependent gating over LoRA directions to reduce task interference while maintaining parameter efficiency, which is closely related to the routing and gating mechanisms discussed in Section 2.2 and Equation (3). They are important baselines/contrastive points for the claim that sophisticated routing is not necessary. They should be discussed in Section 2.2 and possibly mentioned when interpreting the empirical results of M-LoRA vs routed methods.

3. **Tang et al., “L-LoRA: Linearized LoRA for Enhanced Multi-Task Fusion” (2023).**  
   L-LoRA studies how to fuse LoRA modules across tasks via partial linearization, directly addressing multi-task fusion and shared representations. It is conceptually relevant to both the M-LoRA “collaborative heads” story and Align-LoRA’s shared-space alignment. It should be cited in the Related Work section, with comparison in Section 4 where the authors argue that multi-component or fusion-based architectures are unnecessary.

4. **Agiza et al., “MTLoRA: Task-Agnostic and Task-Specific Low-Rank Adaptation for Multi-Task Learning” (2024).**  
   MTLoRA explicitly decomposes LoRA into shared and task-specific components, which is central to the paper’s theme of task-shared vs task-specific knowledge. MTLoRA is briefly mentioned via “MTLLoRA (Agiza et al., 2024)” but only as an example of multi-head design; the specific mechanism (dual TA/TS modules) and its implications for shared representation learning are not discussed. Section 2.2 and Section 3.3 should more thoroughly explain how Align-LoRA differs from MTLoRA’s shared-task adapter approach.

5. **Liu et al., “MOELoRA: Mixture-of-Experts LoRA for Multi-Task Learning” (2023).**  
   This work (closely related to the cited “LoRA MoE” / “MoELoRA”) integrates mixture-of-experts with LoRA to balance shared and specific information. Although “LoRA MoE” is cited, the specific medical multi-task context and its exploration of shared vs specialized behavior would strengthen the related-work narrative. Clarifying whether the authors’ “LoRA MoE” reference already covers this exact work, and if so, summarizing its approach alongside HydraLoRA/R-LoRA in Section 2.2, would help.

6. **Xin et al., “Beyond Full Fine-Tuning: Harnessing the Power of LoRA for Multi-Task Instruction Tuning” (2024).**  
   This paper studies multi-task instruction tuning with LoRA and finds that proper rank and scheduling are crucial. It is highly relevant to Section 4’s result that simply increasing rank in a single LoRA adapter can match complex architectures. It should be cited in the Introduction or Related Work as converging evidence that rank selection is critical and compared in discussion when presenting Tables 2–3.

These works do not invalidate the current contribution, but they are important for accurately situating the paper in the rapidly growing multi-task LoRA literature.

## Questions

1. **Task sampling and batch composition for the alignment loss.**  
   How are training batches constructed when computing $\mathcal{L}_{\text{KL}}$ or $\mathcal{L}_{\text{MK-MMD}}$?  
   - Are all $M$ tasks represented in every global batch, or do you maintain separate per-task batches and accumulate pairwise losses across them?  
   - If some tasks are missing from a batch, how is the sum over $i<j$ in Eq. (5) practically approximated (e.g., sampled pairs, memory buffers of running stats)? Clarifying this could significantly improve reproducibility.

2. **Why align only $\mathbf{A}$ outputs rather than full adapter outputs or transformer states?**  
   The intuition that $\mathbf{A}$ captures shared knowledge is plausible and supported by prior work, but did you run explicit ablations aligning (a) $\mathbf{B}\mathbf{A}x$, (b) intermediate transformer hidden states, or (c) logits? If so, how did they compare? If not, could you add or at least discuss such ablations, since they would directly test the claimed superiority of down-projection alignment?

3. **Sensitivity to the Gaussian assumption and chosen discrepancy measure.**  
   In Eq. (5), you rely on diagonal Gaussian modeling of per-task representations. Have you observed any failure cases where this approximation is poor (e.g., highly multimodal tasks) or where MK-MMD significantly outperforms KL? Could you provide more quantitative comparison between A-LoRA-K and A-LoRA-M beyond Tables 4–5 (e.g., on FlanV2 or BBH) and some intuition on when each is preferable?

4. **Variance across random seeds.**  
   The gains in several tables are a few points in absolute accuracy, which can nonetheless be meaningful. Did you evaluate variability across multiple random initializations of LoRA parameters and the router in baselines like R-LoRA and HydraLoRA? If not, is there at least qualitative evidence that the observed margins are robust (e.g., repeated training on a subset of tasks)? Including mean±std for a key table (say, Table 4) would strengthen the empirical claims.

5. **Interplay with more powerful routing or gating.**  
   Since Appendix I shows that adding alignment to R-LoRA and M-LoRA yields further gains (Table 9), do you see any scenario where a carefully tuned routed+aligned architecture clearly outperforms single-adapter Align-LoRA-K at comparable parameter budgets? If so, that might nuance the “architecture complexity is not needed” message; if not, explaining why you believe routing gives limited extra benefit in the presence of alignment would be valuable.

Author answers to these questions, particularly (1), (2), and (4), could meaningfully increase my confidence in both the empirical robustness and the conceptual framing.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The empirical methodology is generally solid, with diverse benchmarks, multiple base models, and strong baselines. Some theoretical claims are more generic than advertised and there are missing implementation details for the alignment loss, but nothing suggests a fatal flaw.

## Presentation Rating

3: good.  
The writing is clear, and the paper is easy to follow. Figures (especially Figures 1–4) and tables (1–6) are well-chosen and support the narrative, though the theoretical and implementation sections could benefit from tighter notation and more explicit details.

## Contribution Rating

3: good.  
The work offers a meaningful empirical critique of current multi-head/multi-adapter LoRA designs, provides a simple and practical alternative (Align-LoRA), and backs it up with convincing experiments. The conceptual novelty is moderate given prior MTL and domain adaptation literature, but the concrete insights and practical impact for PEFT are nontrivial.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper delivers a clear and practically important message: in multi-task LoRA for LLMs, simple single-adapter setups with sufficient rank and a lightweight alignment loss can match or surpass more complex routed multi-head architectures, while preserving mergeability and inference efficiency. The empirical evidence is quite thorough and the method is easy to adopt, which I expect to be valuable to practitioners. However, the conceptual novelty is somewhat incremental relative to longstanding domain-adaptation alignment ideas, the theoretical analysis is rather generic, and several details/ablations are missing or only sketched. With stronger positioning against the most recent multi-task LoRA works and a slightly sharper theory/ablation story, this could be an even clearer accept; as it stands, I lean positive but not enthusiastically.

## Reviewer Confidence

4: confident.  
I am familiar with PEFT/LoRA and multi-task learning literature, have carefully gone through the equations and tables, and feel reasonably confident in my assessment, though some missing experimental details (e.g., alignment implementation specifics) prevent absolute certainty.