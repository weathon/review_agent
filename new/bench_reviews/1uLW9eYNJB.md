## Summary
The paper proposes Mixture of Shards (MoS), a parameter-efficient finetuning method that enhances LoRA by combining global (inter-layer + intra-layer) parameter sharing with four differentiation strategies: subset selection, pair dissociation, vector sharding, and shard privatization. Motivated by empirical evidence that pure parameter sharing degrades performance and differentiation is crucial to reverse this effect, MoS constructs low-rank matrices by routing and concatenating shards from shared global pools. Experiments on LLaMA2-7B/13B and LLaMA3.2-3B demonstrate approximately 8× parameter savings compared to standard LoRA at comparable performance levels.

## Strengths
- **Principled motivation grounded in empirical analysis**: The paper systematically demonstrates (Table 1) that pure sharing hurts performance and that differentiation strategies (even nearly cost-free ones like subset selection) reverse this degradation. This "sharing vs. differentiation" insight is a genuine conceptual contribution that extends beyond the specific method proposed. The ablations in Table 2 further validate that each component contributes meaningfully.
- **Strong parameter efficiency at fixed budgets**: At the 5M trainable parameter budget, MoS (4/8) consistently outperforms LoRA (rank 2), ProLoRA (4/8), Tied LoRA, and VeRA across all six metrics on LLaMA2-7B (Table 2). At the 19.99M budget, MoS (16/32) achieves average performance (37.63) comparable to LoRA rank 64 (37.53) with 8× fewer parameters, which is a practically meaningful result for multi-LoRA serving.
- **Modular and well-structured design**: Each differentiation strategy (pair dissociation, vector sharding, shard privatization) serves a clear purpose—combination diversity and exclusive differentiation—and the ablation cleanly shows their relative importance (pair dissociation and privatization matter ~1% each; vector sharding contributes only ~0.4%).
- **Preservation of LoRA's practical advantages**: The method maintains LoRA's linearity (allowing weight merging for inference) and low-cost task switching, which matters for real-world deployment.

## Weaknesses

### Fatal
None.

### Major
- **No demonstration of the motivating use case (multi-LoRA serving)**: The introduction's primary motivation is serving "numerous customized models simultaneously" (e.g., "10,000 active users" requiring "3.36 TB of GPU memory"). Yet no experiment measures actual GPU memory savings, inference latency, or throughput in a multi-adapter serving scenario. Without this, the core practical benefit—reduced GPU memory for concurrent serving—remains theoretical rather than demonstrated. The 8× parameter reduction should translate into memory savings, but the paper does not verify this nor account for the overhead of storing index matrices, pools, and routing structures per adapter.

- **No training or inference efficiency metrics**: The paper repeatedly claims the differentiation strategies are "nearly cost-free," but provides no evidence beyond parameter counts. The shard assembly, routing, and concatenation operations introduce non-trivial memory access patterns compared to standard dense LoRA matrix multiplications. Training time per step, peak GPU memory during training, and inference latency are all absent. Without these, "nearly cost-free" is a claim about parameter counts only—a meaningful distinction from compute cost.

- **Missing comparison with recent strong PEFT baselines**: The paper compares against LoRA, VeRA, Tied LoRA, and ProLoRA, but omits several widely-adopted methods that also target parameter efficiency, such as DoRA, rsLoRA, LoRA+, or AdaLoRA. While these methods take different approaches (weight decomposition, rank stabilization, etc.), they are the natural baselines for any paper claiming improved parameter efficiency. Their absence limits the ability to assess whether MoS's gains are genuinely competitive or whether simpler alternatives achieve similar efficiency.

- **VeRA comparison is incomplete**: VeRA operates at 1.42M trainable parameters (far below the 5M budget), and the paper could not run VeRA at matched parameters due to OOM. While the paper acknowledges this, claiming VeRA has "a very low performance-to-rank ratio" based on a single unachievable configuration is not definitive. Alternative experimental setups (gradient accumulation, reduced batch size, mixed precision) could potentially resolve the OOM and provide a fairer comparison.

### Minor
- **"MoE-like routing" terminology is misleading**: The routing mechanism uses randomly initialized, fixed index matrices (Sec. 3.2: "remains fixed during the finetuning process"). This is static shard assignment with no learned or input-dependent gating, yet the paper repeatedly calls it "MoE-like routing" (abstract, Sec. 1, Sec. 3). Standard MoE implies dynamic, learnable routing; describing fixed random assignment as MoE-like overclaims the architectural novelty. This should be described as "static index-based shard selection" or similar.

- **Dropped tasks in the 13B scalability analysis**: Table 3 excludes TyDi QA and HumanEval for LLaMA2-13B, attributing this to "the more powerful capabilities of the base model, which diminishes the effectiveness of the training set." This selective reporting weakens the scalability claim, as it is unclear whether MoS's advantages transfer to the dropped tasks at larger scale.

- **No hyperparameter sensitivity analysis**: MoS introduces several new hyperparameters (shard size *l*, pool size, private/public pool ratio) that are not studied for sensitivity. The ablation only removes entire components rather than varying their configurations, leaving open whether reported results are robust or depend on specific hyperparameter choices.

- **No statistical significance measures**: All results are single-seed. Differences between MoS and ProLoRA at 5M parameters average ~0.3–1.0% across tasks (Table 2). Without error bars or multiple-seed runs, it is unclear whether these margins represent genuine improvements or seed-dependent noise.

- **Limited model family diversity**: All experiments use LLaMA-family models (LLaMA2-7B, 13B; LLaMA3.2-3B in appendix). Generalizability to other architectures (Mistral, Qwen, Gemma) is not established.

### Trivial
- The transition from the boolean mask formulation (Eq. 3) to the index-based formulation (Eq. 4) is somewhat abrupt; a brief explanatory sentence would improve readability.

## Nice-to-Haves
- Multi-LoRA serving benchmark measuring actual GPU memory usage and inference latency when serving *N* concurrent adapters, directly validating the paper's core motivation.
- Pareto curves plotting performance vs. trainable parameter count across multiple rank configurations for all methods, rather than relying on single operating points.
- Visualization of learned or assigned routing patterns across layers to reveal whether meaningful specialization emerges from the static index assignments.
- Sensitivity analysis for shard size *l* and privatization ratio to establish robustness of the method.

## Removed Points
*These points were flagged and removed; treat them with caution.*
- **Questioning the existence or availability of cited baselines/models**: Several reviews suggested VeRA or other methods were not fairly compared due to OOM or availability issues. Per the review rules, if the paper cites a baseline, it exists; the OOM concern is a legitimate methodological gap (already kept as a Major weakness), but questioning whether the baseline is a "currently available system" is removed.
- **Demanding confidence intervals for large-scale benchmarks where single-run evaluation is the norm**: The suggestion for formal significance testing on instruction-tuning benchmarks is reasonable in principle but is not standard practice in the PEFT community. Downgraded from a demand to a minor note.
- **Requesting theoretical proofs for an empirical contribution**: The paper's "sharing vs. differentiation" insight is empirically motivated, and the paper does not claim theoretical grounding. Requesting formal expressivity analysis is scope creep—moved to Nice-to-Have.
- **Formatting and notation nitpicks**: Minor notation inconsistencies and abrupt transitions are noted but are trivial concerns.
- **Claim that the 8× savings is based on a single configuration and thus "anecdotal"**: The 8× comparison (LoRA r=64 vs. MoS 16/32) is indeed a specific configuration, but the paper also provides lower-budget comparisons (5M params) and results on 13B. The concern about variance is valid (kept as Minor), but dismissing the entire claim as anecdotal overstates the issue—the consistent trend across experiments supports the efficiency claim even without formal significance testing.

## Novel Insights
The observation that pure parameter sharing can *hurt* performance compared to vanilla LoRA (Table 1, "Pure Sharing" at rank 64 underperforming LoRA at rank 2 on 4/5 tasks) is counterintuitive and important: it demonstrates that simply increasing rank via sharing does not compensate for the expressivity loss from homogenization. The finding that subset selection (a zero-cost boolean mask) is sufficient to *reverse* this degradation and consistently outperform LoRA is a clean empirical result that should inform future parameter-sharing designs. Additionally, the ablation revealing that pair dissociation provides much larger gains than vector sharding—despite both targeting "combinatorial diversity"—suggests that the specific *structure* of diversity matters more than its quantity, a nuance the community should consider.

## Suggestions
- Replace "MoE-like routing" terminology with "static shard assignment" or "index-based shard assembly" throughout the paper to accurately reflect the mechanism and avoid misleading readers.
- Add a practical multi-LoRA serving experiment: measure peak GPU memory when loading *N* MoS adapters vs. *N* LoRA adapters alongside a base model, and report inference throughput. This directly validates the paper's primary motivation.
- Report wall-clock training time and per-step GPU memory consumption for MoS vs. LoRA vs. ProLoRA to substantiate the "nearly cost-free" claim from a computational perspective.
- Attempt VeRA at matched-parameter configurations using memory-saving techniques (gradient checkpointing, smaller batch sizes) to close the comparison gap, or acknowledge the limitation more explicitly without claiming VeRA is "impractical."

## Score and Decision

**Calibration references:**
- **Bi-Share LoRA** (similar topic: inter+intra LoRA sharing): Reject, avg ~5.25 — less thorough ablations, weaker empirical analysis
- **ShareLoRA** (LoRA parameter sharing): Withdrawn/Reject, avg ~4.2 — smaller contribution, less principled motivation
- **RaSA** (rank-sharing LoRA): Accept Poster, avg ~7.0 — cleaner method, stronger baselines
- **SeRA** (segmented LoRA adaptation, similar sharding concept): Reject, avg ~4.5 — limited empirical validation
- **MoE+PEFT** (MoE combined with PEFT): Accept Poster, avg ~6.75 — well-motivated combination with good experiments

MoS is more thorough than Bi-Share LoRA and ShareLoRA (clearer motivation, systematic ablations, multi-scale evaluation) and comparable to SeRA in methodological clarity but with better empirical coverage. However, it falls short of RaSA in baseline coverage and experimental rigor. The core methodological gap—the absence of any multi-LoRA serving or efficiency experiment despite it being the paper's primary motivation—is a significant omission. The MoE terminology overclaim and missing modern baselines further weaken the contribution. On balance, this is a solid but incomplete contribution that would significantly benefit from practical efficiency evaluation and fairer baseline comparisons.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>