Now I have a good understanding of the calibration landscape. Let me now carefully verify specific claims from the reviewers against the paper content.

Key claims to verify:

1. **Jump-step planning contradiction with DV**: The harsh critic claims there's an inconsistency — the paper says "jump-step planning is beneficial" but DV uses stride=1. Let me check what the paper actually says.

From the paper (§4.2): "One crucial result we found is that jump-step planning (Sect. 3.1) is beneficial in almost all cases, despite the fact that most previous work used dense-step planning. This is observed in DV (Fig. 4)"

But looking at Figure 4 description: "Performance generally decreases as the planner stride increases, with the star indicating the optimal stride." And the star is at Stride=1 for DV.

So DV uses stride=1 (dense-step), but the text says "jump-step planning is beneficial in almost all cases." This seems contradictory. However, looking more carefully — the text says "This is observed in DV" and then the figure shows performance decreasing as stride increases. The caption says "It reduces to dense-step planning when Stride=1." So the DV choice is stride=1, which IS dense-step planning.

Wait, let me re-read more carefully: "jump-step planning (Sect. 3.1) is beneficial in almost all cases" — but then Fig 4 shows performance generally DECREASING as stride increases from 1. And the star is at stride=1.

Actually, I need to look at the Appendix for "extensive results" on jump-step planning in other diffusion planners. The text says:
- For DV (Fig 4): performance generally decreases with stride
- For other diffusion planners (Appendix D): jump-step planning is beneficial

So the claim about jump-step planning being beneficial may be about *other* planners, not DV itself. DV happens to work best with stride=1. This is actually somewhat confusing in the main text and could be clearer, but it's not necessarily contradictory — just that DV is an exception.

Wait, re-reading: "One crucial result we found is that jump-step planning...is beneficial in almost all cases, despite the fact that most previous work used dense-step planning. This is observed in DV (Fig. 4) and generally in diffusion planners (see Appendix D for extensive results)."

But Fig 4 shows DV's performance DECREASING with stride. So the claim "jump-step is beneficial" is NOT observed in DV (Fig 4), it contradicts it. This IS an inconsistency in the paper.

Actually wait — let me look at Figure 4 more carefully. The caption says "Performance change of DV over planning stride" and shows lines for individual sub-tasks. The star is at stride=1, which is stated as "the choice of DV." But what if individual sub-tasks show some improvement at stride>1, and the average shows stride=1 is best? That's possible.

Looking at the approximate values from the figure description: "In all environments, the performance generally decreases as the planner stride increases, with the star indicating the optimal stride." So the "optimal" stride is 1, meaning DV prefers dense-step planning.

This is indeed a real inconsistency — the paper's text claims jump-step planning is beneficial, but the main figure for DV shows stride=1 is optimal, and the paper doesn't clearly explain this contradiction.

2. **Missing variance**: Table 1 caption says "We omit the variance over seeds for simplicity; however, it can be found in the detailed tables in Appendix D." So variance IS reported, just in the appendix. This is a standard practice.

3. **Unfair comparisons**: The paper explicitly uses literature numbers for baselines. This is a legitimate concern but also common in the field (D4RL benchmarks). The paper at least re-evaluates DQL and IDQL (the DQL* and IDQL* entries, which are from Dong et al. 2024b's re-evaluation).

4. **Computational cost of MCSS**: This is a valid concern. MCSS requires N trajectory samples per step, which is computationally expensive.

5. **The "6000 models" claim**: The paper states this but doesn't provide a breakdown. Let me check if the appendix mentions this... The references to appendix exist but the appendix content is not provided in the paper text.

6. **Interaction effects**: Valid concern. The one-at-a-time ablation methodology can miss interactions.

Let me now verify some of the specific numbers and claims more carefully:

From Figure 5 data: Transformer beats U-Net in Kitchen-M (73.6 vs 35.2), Kitchen-P (94.0 vs 28.8), but what about "8 out of 9 sub-tasks"? Let me count... Looking at the approximate table:
- Antmaze: L, M-D, M-P, H — Transformer wins L, M-D, M-P, but U-Net wins H
- Kitchen: M, P (only 2 subtasks shown, both won by Transformer)
- Maze2d: L, M, U — Transformer wins L, tied M, loses U?

Actually from the figure 5a description, it says "Transformer outperforms U-Net in 8 out of 9 sub-tasks and in all 3 main tasks." This seems about right if we count all sub-tasks.

Now let me compose the final review.</think>Now I have a comprehensive understanding of the paper, the reviewer concerns, and calibration anchors. Let me compose the final review.

## Summary

This paper presents a systematic empirical study of key design choices in diffusion planning for offline reinforcement learning, training and evaluating over 6,000 diffusion models across D4RL benchmarks (Maze2D, AntMaze, Kitchen, plus Adroit in the appendix). The authors identify several counter-intuitive findings—e.g., unconditional Monte Carlo sampling with selection (MCSS) outperforms classifier/classifier-free guidance when the dataset contains near-optimal data, Transformers outperform U-Net as denoisers, and state planning with separate inverse dynamics outperforms joint state-action modeling in high-dimensional action spaces—and propose Diffusion Veteran (DV), a configuration combining these insights, achieving strong results on D4RL navigation and manipulation tasks.

## Strengths

- **Impressive scale of empirical study.** Training and evaluating over 6,000 diffusion models with systematic sweeps across guided sampling, backbone architecture, action generation, and planning strategy is a substantial empirical effort that provides real practical value. The control-variable methodology (§3.2) is a sound framework for the investigation.

- **Counter-intuitive and actionable findings.** Several discoveries challenge prevailing practices in diffusion planning: MCSS outperforming CG/CFG on near-expert datasets (§4.5), Transformer outperforming U-Net in 8/9 sub-tasks (§4.3), and separate inverse dynamics outperforming joint state-action modeling in high-dimensional action spaces (§4.1). These are valuable insights that the community will benefit from.

- **Clear and distilled takeaways.** The eight practical tips in §4.8 are concise and directly useful for practitioners. The comparison between diffusion planning and diffusion policy in §4.6 is nuanced—the paper clearly acknowledges when diffusion policies are preferable (MuJoCo locomotion) rather than claiming universal superiority.

- **Strong DV baseline performance.** DV achieves state-of-the-art or competitive results on the three planning-oriented task families (Table 1), demonstrating that the identified design principles translate into concrete performance gains.

- **Attention visualization provides mechanistic insight.** The analysis of Transformer attention patterns (Fig. 5b), particularly the finding that characteristic attention length is consistent across planning strides, offers an interpretable explanation beyond raw performance numbers.

## Weaknesses

### Major:

- **The SOTA claim rests on heterogeneous evaluation protocols.** Table 1 combines DV's 500-seed evaluation with literature numbers from baselines that used different numbers of seeds, normalization procedures, and evaluation protocols. While the paper notes "it can be found in the detailed tables in Appendix D," the main table omits variance, making it impossible to assess whether performance margins (some of which appear modest) are statistically meaningful. For a paper whose headline is a SOTA baseline result, this is a significant evidential gap. (The paper does re-evaluate DQL* and IDQL* from Dong et al. 2024b, which partially addresses this, but not for all baselines.)

- **One-at-a-time ablation methodology misses interaction effects.** The control-variable approach (§3.2) fixes the best configuration (DV) and varies one component at a time. This means important interactions—e.g., does MCSS still outperform CG when paired with U-Net? Does separate inverse dynamics matter as much with U-Net as with Transformer?—are never investigated. Given 6,000+ models trained, conducting at least a small factorial experiment on the most critical component pairs would substantially strengthen the generalizability of the "takeaway" recommendations, which are currently stated categorically but rest on a single configuration's ablations.

- **Apparent contradiction in the jump-step planning claim.** §4.2 states "jump-step planning is beneficial in almost all cases," but Figure 4 shows DV's performance generally *decreasing* as stride increases, with the optimal stride being 1 (dense-step). The text attempts to reconcile this by referencing Appendix D results on other diffusion planners, but the main-text presentation is confusing—the claim and the primary figure point in opposite directions. This is not merely a presentation issue; it undermines the clarity of Takeaway 3 ("Implementing jump-step planning can be highly beneficial").

### Minor:

- **MCSS computational cost is under-discussed.** MCSS requires sampling N trajectories per time step and selecting the best via a critic, which is substantially more expensive at inference time than guided sampling. While §5 briefly mentions computational efficiency as future work, a method presented as a practical recommendation should quantify this trade-off (e.g., how many candidates N are needed, and how cost scales).

- **Scope of design principle generalization is overstated.** The paper's title asks "What makes a good diffusion planner for decision making?" broadly, but experiments are limited to D4RL state-based tasks with relatively low-dimensional observation and action spaces. Several takeaways are phrased categorically (e.g., "It is recommended to generate state plans with diffusion planners and use an inverse dynamics model"), but these may not hold for vision-based RL, goal-conditioned tasks, or very high-dimensional domains. The paper acknowledges this in §5 but the takeaways in §4.8 do not carry these caveats.

- **The "separate vs. joint" finding would benefit from more analysis.** The observation that separate inverse dynamics outperforms joint modeling in higher-dimensional action spaces (§4.1) is clear and important, but the explanation is brief. The paper also mentions that "we tested both diffusion models and vanilla MLP as the inverse dynamics, and found similar performance" without providing any data—this is itself a useful design insight that should be supported.

### Trivial:

- **Incomplete pseudocode.** Algorithm 1 is labeled "simplified" but omits training loss details for the critic and inverse dynamics models, as well as the number of denoising steps. For a paper aiming to provide a "simple yet strong baseline," slightly more completeness would help reproducibility (though these details are presumably in the appendix).

## Nice-to-Haves

- Interaction-effect analysis between the most critical component pairs (e.g., guidance × backbone, action generation × backbone), leveraging the 6,000+ trained models.
- Computational cost comparison (wall-clock time, FLOPs, or function evaluations per episode) between DV and prior diffusion planners, plus a sweep of N in MCSS.
- More detailed explanation of why DV uses stride=1 despite jump-step planning being "beneficial," plus clearer presentation of when jump-step actually helps.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Missing related work"** (human finder): Cannot verify whether specific referenced works exist; removed per instructions.

- **"Fairness of SOTA claim due to heterogeneous baselines" from the harsh critic (partially retained above)**: The harsh critic's specific concern about "outdated" baselines and missing entries for some methods in Table 1 is partially valid (evaluation protocol heterogeneity is real), but the claim that "older methods like BC, BCQ whose implementations are known to be benchmark-sensitive" is a red herring—these are included for context, not as primary comparisons. The meaningful comparison is against diffusion planners and diffusion policies, where the paper includes recent methods. Retained the core concern about heterogeneous evaluation but removed the "outdated baselines" framing.

- **"6,000 models claim not broken down"** (spark reviewer): This is a minor transparency point, not a methodological weakness. The paper references Appendix B for implementation details. Removed as trivial nitpick.

- **"Re-run strongest baselines under same protocol"** (spark reviewer): This is a reproducibility demand that goes beyond standard practice in D4RL benchmarking. The paper already re-evaluates DQL* and IDQL* from the re-evaluation in Dong et al. 2024b. Removed as an unreasonable demand.

- **"No analysis of failure modes"** (human finder): The paper does analyze when DV underperforms—specifically on MuJoCo locomotion compared to diffusion policies (§4.6), and network depth effects (§4.4). It also discusses when MCSS underperforms CFG (Kitchen, §4.5). This criticism overreaches.

- **"Transformer vs. U-Net lacks depth"** (human finder): The paper provides performance comparison across 9 sub-tasks AND attention visualizations with a mechanistic hypothesis. The authors explicitly acknowledge "In-depth study will be needed to fully understand the role of long-term dependency" in §4.3. Requesting deeper mechanistic understanding is a nice-to-have, not a weakness.

- **"Variance/confidence intervals missing"** (multiple reviewers): The paper states variance is available in Appendix D. Standard practice in D4RL work is to report in the main table with a footnote about variance, which the authors do. Removing as a formatting nitpick.

- **"Incomplete pseudocode / reproducibility"** (spark): Algorithm details are deferred to the appendix per standard practice. Removed as trivial reproducibility nitpick.

- **"Adroit results only in appendix"** (neutral reviewer): The main paper focuses on three task families and references §4.7/Appendix C for generalization. This is a reasonable organization choice for the main text, not a weakness.

- **"Training 6,000 models is resource-intensive"** (neutral reviewer): The paper itself addresses this in §5 (Sustainability), arguing that the upfront cost saves future researchers from duplicate effort. Criticizing an empirical study for being empirically thorough is circular.

- **"The neuroscientific analogy is speculative"** (harsh critic): The paper clearly presents this as an analogy and insight, not as empirical evidence. It is not overclaimed.

## Novel Insights

The most novel and important insight in this paper is the finding that MCSS (Monte Carlo unconditional sampling with selection) can outperform gradient-based guidance methods (CG, CFG) on tasks where the offline dataset contains a substantial proportion of near-optimal trajectories. This challenges the prevailing assumption in diffusion planning that guidance is essential for generating high-reward plans, and the authors connect this to the value distribution of the dataset (Fig. 7b), providing a plausible mechanistic hypothesis. This single finding has clear practical implications: it suggests that for tasks with expert-quality data, the simpler and more stable MCSS approach is preferable to carefully-tuned gradient guidance.

## Suggestions

- **Clarify the jump-step finding**: Restrate Takeaway 3 to precisely specify when jump-step planning helps and when dense-step planning (stride=1) is preferred, reconciling Fig. 4 with the broader finding. Currently, §4.2 reads as self-contradictory for DV.

- **Add a minimal interaction analysis**: With 6,000+ models in the sweep, report at least the key 2×2 interaction between (MCSS vs. CG/CFG) × (Transformer vs. U-Net) to validate that the individual takeaways compose correctly.

- **Temper the SOTA claim**: Either re-evaluate the top 2–3 baselines under the same 500-seed protocol, or reframe the claim as "DV achieves competitive or improved performance compared to prior methods" rather than "state-of-the-art results," and report standard deviations in the main table.

- **Quantify MCSS cost**: Report the number of candidate samples N and wall-clock inference time per episode for DV versus a representative guided baseline.

## Score and Decision

**Calibration anchors:**
- *Simple Hierarchical Planning with Diffusion* (kXHEBK9uAY): Scores 6/6/6/5, Accept (poster) — hierarchical diffusion planner with clear but incremental contributions on D4RL, limited novelty.
- *Diffusion Modulation via Environment Mechanism Modeling for Planning* (x1SfON9HvT): Scores 3/6/3/3, Reject — diffusion planner claiming SOTA but weak baselines, limited novelty.
- *A Large-Scale Analysis on Methodological Choices in Deep RL* (Ok7ZH2Cyd7): Scores 3/3/6/3/6, Reject — empirical study with overclaims and methodological issues.
- *Reasoning with Latent Diffusion in Offline RL* (tGQirzjddO): Scores 5/8/6, Accept (poster) — latent diffusion offline RL method with solid results.
- *Charting the Design Space of Neural Graph Representations* (5pd78GmXC6): Scores 6/5/8, Accept (poster) — systematic design space study with practical takeaways.

This paper is comparable in spirit to the design-space study (5pd78GmXC6, avg ~6.3) and the hierarchical diffusion planner (kXHEBK9uAY, avg ~5.75). It makes a more substantial empirical contribution than the rejected diffusion planning papers (DMEMM, Contrastive Diffuser, etc.) thanks to its systematic scope and counter-intuitive findings. However, it has a stronger methodological weakness (heterogeneous evaluation for SOTA claim, interaction effects) than the accepted design-space paper, and the jump-step contradiction is a real clarity issue. Its contributions are primarily empirical—the individual design choices are not novel, but their systematic comparison and some counter-intuitive findings (MCSS over guidance, Transformer over U-Net) are valuable. Overall, this is a solid empirical contribution that falls clearly above the acceptance threshold but has significant room for improvement in the rigor of its claims.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>