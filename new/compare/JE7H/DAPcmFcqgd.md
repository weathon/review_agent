---
job_id: f695af76-6c1c-444c-9fcd-e52bc76bcd64
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: DAPcmFcqgd.pdf
paper: MoEP: Compact and Efficient Sparsity with Modular Expert Paths
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility  
Pass ✅.  
The paper proposes a sparse Mixture-of-Experts style architecture for decoder-only LMs and evaluates it on BabyLM; this is clearly within ICLR’s core topics (representation learning, language modeling, sparse architectures).

## Minimum Quality  
Pass ✅.  
All required sections are present (Abstract, Introduction, Related Work in Section 2, Methodology in Section 3, Experiments/Results in Sections 4–5, and Conclusion in Section 6). The work is in English and technically coherent. While there are notable weaknesses in experimental depth, clarity, and positioning, they do not rise to the level of an automatic desk rejection.

## Prompt Injection and Hidden Manipulation Detection  
Pass ✅.  
I do not see any hidden prompts, steganographic text, or instructions aimed at influencing automated reviewing systems within the main paper content.

---

# Expected Review Outcome:

## Summary

The paper introduces MoEP (Modular Expert Paths), a sparse decoder-only language model architecture that combines layer-level top‑k routing across parallel Transformer blocks with MoE-style projection blocks that shrink and grow the hidden dimension. The key design goal is to add sparsity and routing while keeping total parameter count comparable to a dense GPT‑2 baseline by operating most blocks at reduced hidden dimension. The authors train MoEP and a SwiGLU-based variant on the BabyLM strict-small track (≈10M words), compare against GPT‑2 and BabyLM baselines, and report small improvements in BabyLM macro averages along with analyses of training dynamics and routing behavior.

## Strengths

1. **Clear architectural idea: compact layer-level MoE with fixed parameter budget**  
   - The paper articulates a concrete design space: use MoE “shrink” and “grow” projection blocks to move from a large hidden size \(d_L\) to a smaller parallel dimension \(d_P\), then route tokens across parallel Transformer blocks at that smaller dimension, then map back to \(d_L\). This is clearly illustrated in **Figure 2**, which shows the embedding → Layer Block → MoE Shrink → Parallel Layer (repeated) → MoE Grow → Layer Block → LM Head stack, making the overall path and dimensional transitions easy to understand.  
   - The goal of maintaining similar parameter counts to a dense baseline while introducing sparsity is conceptually appealing and practically relevant for low-resource or small-device settings.

2. **Use of layer-level routing rather than only FFN-level MoE**  
   - Most MoE LLM work focuses on FFN-level experts. This paper implements expert routing at the layer level via parallel Transformer blocks (Section 3.3), which is a less explored but interesting design choice. **Figure 1 (right panel)** usefully contrasts layer-level routing (MoEP) with traditional sublayer-level MoE placement (left panel, gating over Attention or FFN), helping position the contribution conceptually.

3. **Controlled comparison under BabyLM strict-small setting**  
   - The experiments are done under a fixed, publicly defined small-scale regime (BabyLM strict-small), with all models trained on the same ≈10M-word corpus and evaluated with the official BabyLM pipeline. This provides a reproducible setting and mitigates concerns about cherry-picked benchmarks.  
   - **Table 2** clarifies that GPT‑2 and MoEP have matched total parameters (28M), so at least for the main comparison MoEP is not simply benefiting from a larger model.

4. **Training-dynamics analysis over checkpoints**  
   - The authors save checkpoints at multiple word counts and analyze the evolution of evaluation performance. **Figures 3–5** (Appendix A.3) plot “Smoothed Deviation from Task Mean Accuracy” for MoEP, GPT‑2, and MoEP‑SwiGLU respectively across training sizes. This provides interesting qualitative evidence that MoEP tends to reach near-peak evaluation earlier (~30M words) than GPT‑2, supporting their claim that modular sparse routing can improve sample efficiency at least in this small regime.

5. **Empirical signal that “simple linear experts” can be competitive or better than SwiGLU experts**  
   - The comparison between MoEP (linear experts) and MoEP‑SwiGLU (SwiGLU experts in the MoE blocks) suggests that, at this scale and data size, the simpler linear experts both train faster and achieve stronger BabyLM scores (see **Table 1**). This is a useful negative result: more expressive FFN-like experts do not automatically help under tight resource constraints.

## Weaknesses

1. **Experimental scope is very narrow and lacks robustness**  
   - All experiments are on a single small benchmark (BabyLM strict-small) with \(\sim10\)M words and relatively low model sizes (28M–38M parameters). There is no evaluation on any larger corpus, any standard LM benchmark (e.g., WikiText, The Pile subsets), or any scaling experiment.  
   - Results are reported for a single run per model; there is no variance across seeds, no error bars, and no statistical testing. Given the small scale and noisy nature of BabyLM tasks, it is difficult to be confident that the \(\approx 0.9\) macro average improvement of MoEP over GPT‑2 in **Table 1** (“Macro Avg” 49.0 vs 48.1) is substantive rather than noise.  
   - The conclusion (Section 6) itself acknowledges uncertainty about whether MoEP’s relative performance “would preserve” when scaling up. For ICLR, this narrow and fragile empirical basis substantially limits the contribution.

2. **Claims about outperforming baselines are not cleanly substantiated and hinge on AoA handling**  
   - The abstract and Section 1 state that MoEP “enables it to outperform the GPT‑2 baseline” and on Page 2 that “MoEP was able to outperform all BabyLM strict-small baseline models”. However, the story in **Table 1** is more nuanced:
     - Among “Our Models,” MoEP’s macro average is 49.0 vs 48.1 for GPT‑2, a very small margin with no variance reported.  
     - Among HF baselines, GPT‑BERT (several variants) often has macro averages >52, beyond both MoEP and GPT‑2. The text notes that MoEP is best “when the AoA task score was included”, but AoA is missing for MoEP and GPT‑2 in Table 1, and the AoA numbers for baselines are partly negative (e.g., −3.9), making the comparison hard to interpret.  
   - The presentation of the macro averages (two different “Macro Avg” definitions; the column annotation in Table 1 is confusing) blurs whether MoEP truly is “best overall” or only in a certain cherry-picked metric configuration. This undermines the strength of the empirical claim.

3. **Insufficient ablations on core architectural choices**  
   - The key ingredients of MoEP are:  
     - the hidden dimension split \(d_L\) vs \(d_P\),  
     - the number of experts \(E\) and top‑k choice in the MoE shrink/grow blocks,  
     - the number of parallel blocks \(P\) and layers \(N\), and  
     - the layer-level gating mechanism versus simple static partitioning.  
   - There are essentially no ablations. **Table 2** gives a single configuration (MoEP: \(d_{\text{model}} = 384/192\), 2 dense + 10 parallel layers, 4 parallel blocks, 4 experts, top‑k = 2), but we never see:  
     - What if we use fewer or more parallel blocks \(P\)?  
     - What if we change \(d_P\)?  
     - What is the effect of routing (top‑k) vs selecting all blocks (a parallel-but-dense baseline with same parameter budget)?  
   - Without these ablations, it is unclear whether the gains, if any, come from sparsity, from more depth, from a better allocation of width vs depth, or simply from training noise. This directly weakens the central scientific claim that “modular sparse routing” per se is responsible for the observed behavior.

4. **Routing mechanism is underspecified and mathematically imprecise**  
   - Section 3.3 defines a Parallel Layer with a “Linear router shaped \(d_P \times P\)” and says it applies “token-level top‑k selection among the \(P\) Parallel Block, where the routed inputs are summed up together”. However, crucial details are missing:
     - How are router logits computed from token states \(h \in \mathbb{R}^{d_P}\)? Is there a softmax over experts?  
     - Are the selected top‑k blocks weighted by softmax probabilities or uniform weighting? The text says “summed up together” which suggests equal weights, but then how is \(\mathcal{L}_{\text{balance}}\) in Equation (2) defined exactly, since it requires probabilities \(p_i\)?  
     - Is the routing “one-to-many” or “one-to-one” over tokens; is capacity limiting applied?  
   - Equation (2) defines \(\mathcal{L}_{\text{balance}} = -\sum_i p_i \log p_i\), which is the entropy of the average routing probabilities. Maximizing this entropy promotes uniform usage, but the paper never clarifies how \(p_i\) is computed from token-level decisions (e.g., average softmax probability vs fraction of tokens routed to expert \(i\)). Standard MoE works (e.g., Switch Transformers, GShard, Expert-Choice) use more detailed balancing losses that factor both load and probability; here the simplification is not justified or analyzed.  
   - In Section 3.3, the set of parallel blocks is denoted \(\{B_1, \dots, B_K\}\) but above they write there are \(P\) parallel blocks; this inconsistency in notation further obscures the exact mathematical setup.

5. **Methodological and architectural comparison to prior MoE work is shallow and misses key references**  
   - The related work section (2.2) is reasonably broad but still omits several highly relevant MoE references that directly address conditional computation and sparsity at comparable or larger scales, including:
     1. **Shazeer et al., 2017, “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”** – foundational MoE layer with top‑k gating and balancing losses, highly relevant to both Equations (2)–(3) and the architecture in Fig. 1.  
     2. **Lepikhin et al., 2020, “GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding”** – large-scale MoE with conditional computation and gating, very relevant to the “sparse activation without proportional compute” motivation.  
     3. **Rajbhandari et al., 2022, “DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale”** – focuses on efficient MoE training/inference infrastructure and routing/balancing design, important context when discussing sparsity and efficiency.  
     4. **Komatsuzaki et al., 2023, “Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints”** – directly related to converting dense models to MoE without increasing parameter count, which is very close to MoEP’s stated goal of maintaining parameter budgets.  
     5. **Fedus et al., 2022, “A Review of Sparse Expert Models in Deep Learning”** – a survey that would help properly contextualize MoEP’s position within the broader MoE design space.  
     6. **Liu et al., 2024, “Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models”** – addresses efficiency under fixed budget via pruning, which is conceptually similar to the paper’s compact-sparsity objective.  
     7. **Nakamura et al., 2025, “Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks”** – analyzes the sparsity-performance tradeoff in MoE models, which is exactly the tradeoff MoEP is targeting.  
   - Beyond citations, the paper does not really compare design choices: e.g., how does MoEP’s entropy-based \(\mathcal{L}_{\text{balance}}\) relate to the load-balancing terms in Shazeer et al. or Expert-Choice routing (Zhou et al., 2022)? How does the decision to use linear experts vs FFN experts compare to Switch or Mixtral? Without such positioning, it is hard to see what truly differentiates MoEP versus just “yet another MoE variant on a small dataset”.

6. **Sparse compute and efficiency claims are not quantified**  
   - A core motivation is “compact and efficient sparsity”, and the abstract claims MoEP is “compact and efficient” and “selective token activation, which accelerates model learning”. However, there are no measurements of:
     - FLOPs per token or per batch for MoEP vs GPT‑2,  
     - wall-clock training or inference time per step,  
     - memory footprint.  
   - The only indirect evidence is that a 10-epoch run takes “1–2 hours” on a single A100 for “a single model” (Section 4, Implementation environment). There is no comparison to GPT‑2 with the same hardware and codebase.  
   - Since the architecture introduces additional MoE routing, it is not obvious that compute is lower or equal to GPT‑2, even if total parameters are matched. Without FLOP analysis or runtime benchmarks, the “efficiency” aspect of the contribution remains speculative.

7. **Parameter fairness is broken for the SwiGLU variant and unexplored**  
   - **Table 2** shows that MoEP‑SwiGLU has 38M parameters vs 28M for GPT‑2 and MoEP. Yet this larger model underperforms both of them in **Table 1** (Macro Avg 47.7 vs 48.1 and 49.0). That is an interesting finding, but the paper does not discuss parameter fairness for MoEP‑SwiGLU:  
     - Is the extra capacity concentrated only in MoE experts?  
     - Could a 38M-parameter GPT‑2 baseline close or exceed the gap?  
   - Without these comparisons, the takeaway that “sometimes lightweight simplicity is better than adding complexity” (Contribution 4) is only partially justified. It might be simply that the larger SwigLU-augmented MoEP is under-trained or tuned poorly, rather than that linear MoE experts are intrinsically superior at this scale.

8. **Over-interpretation of training dynamics without rigorous analysis**  
   - **Figures 3–5** visualize deviations from per-task mean accuracy over training checkpoints. The narrative in Appendix A.3 argues that MoEP learns faster and then overfits, while GPT‑2 learns more steadily. However:
     - The smoothing procedure is not fully specified (type, window size).  
     - Deviations from the “mean” are not clearly defined (mean over which checkpoints? over models?).  
     - The plots are fairly noisy and use only a few tasks, without confidence intervals.  
   - The claims about “accelerated early learning” and “instability” due to sparse routing might be true, but given the lack of formal analysis or robust statistics, these are more suggestive observations than solid evidence.

9. **Presentation issues and minor inaccuracies**  
   - There are numerous typos and inconsistencies: e.g., “ourEx” in the caption of **Figure 1**, “Liner” instead of “Linear” in **Table 2**, “GTP-2” vs “GPT‑2” in **Table 1**, “casual” instead of “causal” in GPT-BERT baselines, “textbfAdamW” in Section 4.1, mixed capitalization in section titles (“MOE PLACEMENT”, “ATTENTION + FFN”). These collectively hurt perceived polish.  
   - Equation (1) is written inline as “\(d_L \rightarrow \text{shrink } d_P \rightarrow \text{ grow } d_L\)” which is more of a verbal description than a proper equation. Given that modeling the shrink/grow mapping is central to MoEP, it would be better specified as actual linear maps, e.g., \(h' = W_{\text{shrink}} h\) and \(\tilde{h} = W_{\text{grow}} h'\), and possibly clarifying how MoE gating enters those maps.  
   - Section 3.3 has the set \(\{B_1,\dots,B_K\}\) for parallel blocks but earlier uses \(P\). Notation errors like this make it harder to follow the math exactly.

Taken together, these weaknesses are substantial and numerous. They do not make the paper invalid, but they do reduce its readiness for ICLR in its current form.

## Potentially Missing Related Work

The following works are, to the best of my checking, not cited in the paper yet are directly relevant:

1. **Shazeer et al., “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”, 2017**  
   - Foundational work on top‑k MoE layers with load-balancing losses and routing, directly relevant to Section 2.2 and Equations (2)–(3).  
   - Should be discussed in the MoE background (Section 2.2.1 / 2.2.2) and near the routing objective to compare balancing terms.

2. **Lepikhin et al., “GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding”, 2020**  
   - Introduces large-scale conditional computation with MoE, highly related to MoEP’s motivation of sparse activation and parallelism.  
   - Could be added in Section 2.2.1 as a major MoE application to large-scale LMs and contrasted with MoEP’s small-scale, fixed-parameter design.

3. **Rajbhandari et al., “DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale”, 2022**  
   - Focuses on practical efficiency and routing/balancing strategies in MoE, relevant to MoEP’s efficiency claims.  
   - Should be mentioned in Section 2.1 / 2.2 when discussing efficient sparse architectures, and possibly in the discussion on efficiency (Section 6).

4. **Komatsuzaki et al., “Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints”, 2023**  
   - Proposes converting dense models to sparse MoE without increasing parameter count, which is very close in spirit to MoEP’s “add sparsity while keeping total parameter count fixed.”  
   - Should be discussed in Section 2.2.1 and compared explicitly in Section 3 to clarify what is new in MoEP relative to this approach.

5. **Fedus et al., “A Review of Sparse Expert Models in Deep Learning”, 2022**  
   - A survey that would help position MoEP within the broader landscape of sparse expert models and highlight which design dimensions the paper is exploring (layer-level routing, dimensionality shrink/grow, etc.).  
   - Could be cited at the start of Section 2.2 to frame the literature review.

6. **Liu et al., “Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs”, 2024**  
   - Explores fixed-budget sparsity by pruning experts, conceptually aligned with MoEP’s desire to maintain a compact parameter budget while leveraging sparsity.  
   - Should be discussed in Section 2.2.1 or 2.2.2 and potentially in the conclusion when discussing future directions for compact sparse models.

7. **Nakamura et al., “Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks”, 2025**  
   - Studies sparsity vs. performance tradeoffs in MoE models, highly relevant to MoEP’s goal of finding useful sparsity under fixed parameter budgets.  
   - Could be engaged in Section 5 (Analysis) or Section 6 to relate MoEP’s empirical behavior to broader observations about optimal sparsity levels.

8. **Shen et al., 2023/2024, “Mixture-of-Experts Meets Instruction Tuning” (instruction-tuning MoE)**  
   - While focused on instruction tuning, it is directly relevant as another example where MoE structure is combined with task adaptation.  
   - Could be briefly mentioned when discussing fine-tuning tasks in the BabyLM evaluation (Section 4) and potential future extensions of MoEP.

These works are not merely tangential; several (especially Shazeer 2017, GShard, DeepSpeed-MoE, Sparse Upcycling) are central MoE references and should be incorporated into the related work and positioning.

## Questions

1. **Routing specifics and balancing loss**  
   - How exactly is the router implemented? Formally, if \(h_t \in \mathbb{R}^{d_P}\) is the token representation, what is the exact mapping to router logits \(g_t \in \mathbb{R}^P\), and what operation determines the final routed output \(y_t\)? A precise equation using softmax / top‑k / masking would help.  
   - How is \(p_i\) defined in Equation (2)? Is it \(\frac{1}{T} \sum_t q_{ti}\) where \(q_{ti}\) is the softmax probability, or the fraction of tokens for which expert \(i\) is in the top‑k? Clarifying this is important to understand the effect of \(\mathcal{L}_{\text{balance}}\).

2. **Compute and runtime comparison with GPT‑2**  
   - Can you provide FLOPs/token or measured training/inference time per step for MoEP vs GPT‑2 under the same hyperparameters and hardware? If the claim is “compact and efficient sparsity”, quantitative evidence here would substantially strengthen the paper.  
   - Relatedly, what is the average number of active parameters per token in MoEP compared to GPT‑2?

3. **Ablation on routing vs parallelism**  
   - Could you report an ablation where the parallel blocks are used *without* routing (e.g., average or concatenation followed by a projection) but with the same parameter count? This would help isolate the benefit of token-level routing from just having more parallel subpaths.  
   - Similarly, what happens if \(k = P\) (i.e., no sparsity in the Parallel Layer) or if \(E=1\) (no MoE in shrink/grow)? Even a small subset of these ablations on BabyLM would offer much clearer insights into which design choice matters.

4. **Generalization beyond BabyLM**  
   - Do you have any preliminary experiments (even small) on a slightly larger text corpus or a different benchmark (e.g., WikiText-103 subset) to test whether the early-learning pattern of MoEP persists? If not, what are the scaling hypotheses regarding \(d_P\), \(P\), and \(N\) when moving up data and parameter scales?  
   - How sensitive is MoEP to the quality and diversity of the corpus, given that BabyLM strict-small is highly curated?

5. **Why does MoEP-SwiGLU underperform despite higher parameter count?**  
   - Can you provide more detailed diagnostics (e.g., training loss curves, routing entropy, expert utilization patterns) explaining why the SwiGLU experts hurt performance?  
   - Is it possible that simple hyperparameter changes (learning rate schedule, dropout, warmup) would close this gap, or do you believe there is an inherent mismatch between SwiGLU experts and your layer-level routing setup at this scale?

Clarifying these points and, where possible, adding empirical evidence in a revision could significantly change my assessment, particularly if they strengthen the case that MoEP meaningfully advances the design of compact sparse LMs.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The method is basically reasonable and aligns with established MoE ideas, but key mathematical and experimental details (routing formalization, ablations, robustness, compute comparisons) are missing or underspecified, so the empirical support for the main claims is limited.

## Presentation Rating

2: fair.  
The core architectural idea is conveyed, and **Figures 1–2** and **Tables 1–3** help, but the paper contains multiple typos, inconsistent notation, and somewhat confusing result reporting (especially around macro averages and AoA), which together reduce clarity.

## Contribution Rating

2: fair.  
The idea of layer-level routing with shrink/grow MoE under a fixed parameter budget is interesting, but the experimental scope is narrow, improvements are small and fragile, and the positioning relative to prior MoE work is incomplete. The contribution is promising but not yet strong enough for ICLR.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper has a plausible and conceptually interesting architecture and some suggestive evidence of early-learning advantages, but the lack of thorough ablations, missing key related work, limited evaluation scope, and weak empirical support for strong claims about efficiency and superiority over baselines keep it below the bar for ICLR in its current form.

## Reviewer Confidence

4: confident.  
I am familiar with MoE and sparse LM literature, have checked the equations and experimental setup in detail, and while more extensive experiments could always change the picture, my current assessment is unlikely to shift dramatically without substantial new evidence.