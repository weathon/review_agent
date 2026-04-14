## Summary
MoS (Mixture of Shards) is a parameter-efficient fine-tuning method that extends LoRA by combining inter-layer and intra-layer parameter sharing with four differentiation strategies — subset selection, pair dissociation, vector sharding, and shard privatization — to mitigate performance degradation from pure sharing. The paper first empirically establishes that pure sharing is insufficient and that differentiation is critical, then proposes a unified global sharing scheme with a MoE-like routing mechanism (frozen random index selection) to construct per-layer low-rank matrices from shared shard pools. Empirically, MoS achieves ~8× parameter savings by matching LoRA rank-64 performance (159.91M params) using only 19.99M trainable parameters across LLaMA2-7B/13B and LLaMA3.2-3B models.

---

## Strengths

- **Empirically grounded design principles (Section 2):** The head-to-head comparison of pure sharing, random scaling, and subset selection in Table 1 is a concrete, specific motivation that meaningfully advances understanding of *why* prior sharing methods succeed or fail. The discovery that subset selection outperforms random scaling by reversing performance degradation while using zero extra parameters is a non-trivial and reproducible finding that guides the entire method design — rather than being mere ablation-as-motivation.

- **Unified treatment of inter- and intra-layer sharing:** Prior work addresses either inter-layer sharing (VeRA, Tied LoRA) or intra-layer sharing (PRoLoRA) in isolation. MoS is the first method to integrate both under a single global-pool framework with complementary differentiation strategies. Table 2 confirms this integration pays off: MoS at 5.00M consistently outperforms all baselines (LoRA, VeRA, Tied LoRA, ProLoRA) across all six benchmarks simultaneously — not just on average.

- **Fine-grained shard decomposition as basic unit:** The shift from vector pairs to sub-vector shards as the atomic unit of sharing is conceptually novel within the LoRA literature. The combinatorial diversity argument — that dissociating A/B index matrices and using shards exponentially increases distinct configurations — provides a principled explanation for why pair dissociation and privatization yield substantial gains while sharding yields incremental ones, confirmed in the ablation (Table 2: -sp and -pd each cost >1%, -vs costs ~0.4%).

- **Consistent scalability across model sizes:** Performance gains over PRoLoRA are replicated across LLaMA2-7B (+0.36% avg), LLaMA2-13B (+0.94% avg), and LLaMA3.2-3B (Appendix B.3), suggesting the method is robust and not tuned for a single configuration.

---

## Weaknesses

- **Likely notation error in Eq. (5):** In Eq. (4), $\text{Route}^c$ retrieves column vectors from $\mathbf{B}^p$ using $\mathbf{I}_b^k$, and $\text{Route}^r$ retrieves row vectors from $\mathbf{A}^p$ using $\mathbf{I}_a^k$. In Eq. (5), however, $\text{Route}^c$ is applied to $\text{Concat}(\mathbf{A}^{pub}, \mathbf{A}^{pri})$ — i.e., the A pool — while $\text{Route}^r$ is applied to the B concatenation. This directly contradicts both Eq. (4) and the surrounding text ("$\mathbf{A}^p$ in Eq. 4 is substituted by $\text{Concat}(\mathbf{A}^{pub}, \mathbf{A}^{pri})$"), which implies $\text{Route}^r$ should be applied to the A concatenation. The A/B arguments appear swapped in Eq. (5). Since MoS's formalization is the central technical contribution, a notation inconsistency of this kind must be corrected with care.

- **Selective benchmark reporting for LLaMA2-13B:** Table 3 reports only 3 of the 6 benchmarks used in Table 2. The stated justification — that "vanilla LoRA does not yield consistent improvements on TyDi QA and Code benchmarks" for 13B — is disclosed but not quantitatively supported. Even if the comparison is noisy, reporting all six results with a caveat preserves transparency and lets readers judge. Selectively dropping benchmarks where the baseline is volatile undermines the completeness of the scalability analysis.

- **Training throughput not measured:** MoS constructs each low-rank matrix via index-based retrieval and shard concatenation at every forward pass — gather operations over potentially hundreds of layers. No wall-clock training time, GPU memory during training, or throughput (samples/sec) is reported. For a method whose primary motivation is resource efficiency, this omission is substantive: parameter count savings do not automatically translate to training efficiency.

- **Hyperparameter sensitivity unanalyzed:** MoS introduces several new hyperparameters: pool size $n$, shard size $l$, and the public/private pool split ratio. None of these are subjected to sensitivity analysis. Given that different linear layer types (Q, K, V, O, Up, Gate, Down) have different dimensions, it is also unclear how pool size is set per layer type. High sensitivity to these choices would significantly impact practical adoptability.

- **Ablation configuration differs from the core efficiency claim:** The ablation study (Sec. 4.4, Table 2) removes components from MoS at rank 16/32 (19.99M parameters). However, the main parameter efficiency claim — where MoS provides the most practical advantage over LoRA — is at 5.00M parameters (rank 4/8). The relative contribution of each component at the 5.00M regime may differ, and the ablation does not cover it.

- **No multi-adapter serving validation:** The entire paper is motivated by the challenge of serving 10,000 concurrent customized models (Section 1), yet no experiment measures actual GPU memory consumption or inference throughput when loading multiple MoS adapters simultaneously. The parameter count reduction is a proxy for the claim, not a direct measurement. At serving time, the frozen index matrices $\mathbf{I}_a^k$ and $\mathbf{I}_b^k$ must also be stored per adapter — this storage overhead is not accounted for in the parameter counts.

- **Gradient conflicts in shared pool unanalyzed:** Because all Transformer blocks draw from the same global pool, gradients from different layers — each with potentially divergent optimization objectives — accumulate into the same shared parameters. This is a known challenge in parameter-sharing methods. The paper does not analyze whether this causes gradient conflict or optimization instability, nor whether the fixed random routing mitigates it.

- **"Approximately converges to full finetuning" asserted without evidence (Sec. 3.6):** The paper claims that MoS's performance "can approximately converge to that of full finetuning as do LoRA and ProLoRA." No full fine-tuning results are reported in Table 2 or elsewhere in the main paper, making this claim unverifiable from the presented evidence.

---

## Nice-to-Haves

- **Learned vs. frozen routing comparison:** A targeted ablation replacing the fixed random indices with learned gating (e.g., straight-through Gumbel-softmax) would directly justify the design choice of static routing and clarify the relationship to true MoE architectures.

- **Full fine-tuning results in Table 2:** Including full fine-tuning as an upper bound would allow readers to assess both the absolute gap between MoS and an oracle and the convergence claim in Sec. 3.6.

- **Shard utilization analysis:** A heatmap of how frequently each shard in the global pool is selected across layers (e.g., via frequency counting over the frozen index matrices) would reveal whether the pool is efficiently utilized or whether many shards are dead weight.

- **Clarify "MoE-like" framing in the introduction:** The routing mechanism is more precisely described as static random index selection (frozen at initialization) rather than learned, adaptive routing. Flagging this distinction earlier would prevent readers familiar with sparse MoE literature from being misled by the terminology.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"8× savings claim is misleading"** (harsh critic): The paper explicitly compares MoS-16/32 (19.99M params) to LoRA-rank-64 (159.91M params), which is the performance-targeted reference point — not an arbitrary inflated baseline. The paper frames this as "in a regular setting, MoS achieves around 8× savings" and substantiates it with Table 2 data showing near-identical average performance (37.63 vs. 37.53). The comparison is internally consistent and disclosed. The ratio is ~8×, which is accurate. **Removed.**

- **Missing baselines (AdaLoRA, LoRA-XS, GaLore, FLORA)**: Per review policy, we cannot confirm the existence or availability of external references not cited in the paper. The authors compare against the most directly relevant published parameter-sharing baselines. **Removed.**

- **No statistical significance / error bars**: Single-run evaluation is standard practice for large-scale instruction-following benchmarks in this subfield. The authors do include multi-seed results for LLaMA3.2-3B (Appendix B.3). Requiring error bars for 7B/13B runs is not community-standard for this setting. **Removed.**

- **VeRA OOM comparison unfairness**: If VeRA at a comparable parameter count causes an out-of-memory error, that is a legitimate practical limitation of VeRA, not an asymmetric comparison that favors VeRA. The authors transparently note the OOM. **Removed.**

- **"First to apply MoE-like mechanism in single-task LoRA" claim oversells novelty**: The paper carefully scopes this as "single-task" to distinguish from multi-task MoE-LoRA works (MoA, MixLoRA, MoDE), and the related work section covers these distinctions. The framing is reasonable. **Removed.**

---

## Novel Insights

The most genuinely novel and underappreciated insight in this work is the **empirical falsification of the assumption that higher shared rank monotonically improves performance** (Section 2, Table 1). Prior work implicitly assumed that parameter sharing → higher effective rank → better performance. MoS shows this chain breaks without differentiation: pure sharing at rank 64 (5.00M params) underperforms LoRA at rank 2 (5.00M params) on 3 of 5 tasks. The follow-up demonstration that subset selection — a zero-cost Boolean masking — is sufficient to *reverse* this degradation and *exceed* LoRA's performance by >1% on average is a clean, surprising result that reframes the problem from "how to share" to "how to share while maintaining differentiation." This insight applies broadly to any parameter-sharing scheme in transformer fine-tuning, not just LoRA, and the dissociated-pool framing (separate A and B pools enabling exponentially more combinations) is a concrete mechanism that future work can build on.

---

## Suggestions

1. **Correct Eq. (5):** Verify whether Route^c and Route^r arguments for A and B are swapped. The corrected equation should read $\text{Route}^c(\text{Concat}(\mathbf{B}^{pub}, \mathbf{B}^{pri}), \mathbf{I}_b^k)\,\text{Route}^r(\text{Concat}(\mathbf{A}^{pub}, \mathbf{A}^{pri}), \mathbf{I}_a^k)$ to be consistent with Eq. (4) and the surrounding prose.

2. **Report training throughput:** Add a table or figure measuring training time (steps/sec or samples/sec) and peak GPU memory *during training* for LoRA, PRoLoRA, and MoS at matched parameter budgets. Even if overhead is small, reporting it removes an open question that reviewers will raise.

3. **Report all LLaMA2-13B benchmarks:** Include TyDi QA and Code results for 13B with a clear disclaimer that LoRA shows high variance on these tasks. Let the data speak rather than pre-filtering.

4. **Add hyperparameter sensitivity analysis:** A simple grid over pool size $n$ and shard size $l$ (e.g., 3 values each) on the 5.00M budget configuration would demonstrate robustness and guide practitioners.

5. **Report full fine-tuning as upper bound:** Include at least one full fine-tuning result in Table 2 to substantiate the "converges to full finetuning" claim in Section 3.6, or remove/soften the claim.

6. **Add a practical serving simulation:** Report total GPU memory (including index matrices and private shards) when loading K concurrent MoS adapters versus LoRA adapters, for K in {100, 1000, 10000}, to directly validate the motivating scenario.

---

**Novelty:** Moderate-to-good. The individual components have antecedents (VeRA, PRoLoRA, MoE-LoRA), but the unification of inter- and intra-layer sharing with fine-grained shard-level differentiation, along with the principled empirical analysis establishing why differentiation is necessary, constitutes a genuine and coherent advance.

**Technical soundness:** Adequate, with a likely notation error in Eq. (5) that requires correction and an unanalyzed optimization concern (gradient conflicts in shared pools).

**Empirical support:** Good for LLaMA2-7B; incomplete for LLaMA2-13B (3/6 benchmarks); credible for LLaMA3.2-3B via Appendix. Core serving motivation is not empirically validated.

**Significance:** High in principle — the multi-LoRA serving problem is real and growing — but the significance claim rests on parameter count proxies rather than direct serving measurements.

**Clarity:** Generally clear and well-organized, with the exception of Section 3.4 (underspecified vector sharding concatenation details) and the notation inconsistency in Eq. (5).

MY FINAL SCORE: <pineapple>6.2</pineapple>