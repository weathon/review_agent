---
job_id: cb43000e-6bfd-4c80-96bd-e1dd3fbc7dbc
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: qCfYOLAzti.pdf
paper: LLM Unlearning with LLM Beliefs
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper addresses unlearning in large language models, with theoretical analysis of optimization dynamics and extensive empirical evaluation. This is squarely within ICLR’s scope (representation learning, optimization, safety/privacy).

## Minimum Quality
Pass ✅.  
The paper includes all required components (Abstract, Introduction, Related Work in Appendix B but clearly present, Method, Experiments, Results, Conclusion). The method is non‑trivial, the theory is coherent, and experiments are substantial with standard benchmarks (TOFU, MUSE, WMDP). No fatal methodological or theoretical flaw is evident, and exposition is generally clear.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts, steganographic content, or instructions aimed at manipulating automated reviewers.

---

# Expected Review Outcome:

## Summary

The paper studies failure modes of gradient-ascent–style LLM unlearning, focusing on “spurious unlearning” where models stop producing exact target strings but still emit close paraphrases that preserve the sensitive knowledge. The authors attribute this to a “squeezing effect,” where lowering the target likelihood via softmax redistributes probability mass into nearby high-likelihood regions.  

They propose a bootstrapping framework that uses the model’s own high-confidence predictions (“model beliefs”) as additional unlearning targets: BS-T interpolates the forget token label with the model’s top‑k token distribution, and BS-S augments the forget set with high-likelihood full sequences. Using the AKG learning-dynamics framework, they analyze how these objectives reshape gradients, and they show empirically on TOFU, MUSE, and WMDP that BS-T and BS-S improve the trade-off between forgetting and retention compared with GA, NPO, WGA, RMU, and others.

## Strengths

1. **Clear identification and analysis of a practically important failure mode.**  
   The paper convincingly argues that standard metrics can mis-report unlearning success because they only track surface similarity to targets. The case studies in §3.1 (e.g., GA collapsing to “always always…” and NPO rephrasing “Hsiao Yun-Hwa… writes in English”) make the problem very concrete. The follow-up analysis in §3.2, including **Figure 2**, is especially useful:  
   - **Figure 2a** shows that high-likelihood beams from the original model are semantically closest (via LaaJ similarity) to the target, and that NPO’s outputs remain as similar as these high-likelihood paraphrases, unlike retrain.  
   - **Figures 2b–2c** track log-probabilities of target vs high/mid/low-likelihood regions over training epochs, directly visualizing probability mass being pushed from the target into high-likelihood neighbors for GA and especially NPO.  
   This kind of mechanistic evidence is rare in unlearning papers and will likely influence how the community evaluates “success” going forward.

2. **Simple but well-motivated algorithmic idea (bootstrapping on model beliefs).**  
   The two proposed variants are conceptually clean:
   - BS-T (Eq. (5)–(6)) replaces the one-hot target with a soft target interpolating between the forget token and the detached top‑k token distribution. This directly penalizes both the labeled token and its local neighborhood.  
   - BS-S (Eq. (7)) bootstraps full sequences by sampling high-likelihood continuations and treating them as forget data.  
   The connection between these constructions and the squeezing effect is explicit, and **Figure 1** nicely summarizes the intuition: the left panel illustrates how suppressing only the target pushes mass into a “high-likelihood region,” while the right panel shows how BS-T/BS-S pull that region down as well.

3. **Non-trivial theoretical grounding using established learning-dynamics tools.**  
   The use of the AKG framework (Ren & Sutherland, 2025) in §5 is thoughtful. Lemma 5.1 restates the decomposition of one-step updates into softmax Jacobian \(\mathcal{A}\), kernel \(\mathcal{K}\), and a residual \(\mathcal{G}\). The new content is in how \(\mathcal{G}\) changes:
   - **Theorem 5.2** derives the BS-T residual \(\mathcal{G}_{\mathrm{BST}}^i = \pi^i - ((1-\lambda) e_{y_u^i} + \lambda q^i)\), with the key identity \(\mathcal{G}_{\mathrm{BST}}^i[v] = \mathcal{G}_{\mathrm{GA}}^i[v] + \lambda q^i[v]\) for \(v \neq y_u^i\). Together with **Figure 3**, this clearly shows that BS-T spreads “repulsion” from the target into the top‑k region instead of creating a new peak.  
   - **Theorem 5.3** extends this to off-policy BS-S, showing the update as a kernel-weighted sum of BS-T residuals over belief-aligned continuations.  
   While not technically deep, these results give a precise mathematical articulation of why bootstrapping should counteract squeezing, beyond hand-wavy intuition.

4. **Strong and broad empirical evaluation with appropriate baselines.**  
   The experimental section is substantially stronger than many unlearning papers:
   - On **TOFU**, **Table 1** reports Agg / Mem / Util across 1%, 5%, 10% forget fractions and three LLaMA 3 variants (1B / 3B / 8B). BS-S consistently matches or beats NPO and RMU on Agg and Mem while retaining comparable Utility; for example, at 10% forget on Llama 3.1 8B, BS-S achieves Agg 0.64 vs NPO 0.63 and RMU 0.62, and Mem 0.58 vs NPO 0.57.  
   - On **WMDP**, **Table 2** shows BS-S with Bio/Cyber accuracies 0.26/0.27 (near-random 0.25) while maintaining MMLU 0.54, notably better retention than NPO (0.44) and WGA (0.48) for similar forget scores.  
   - On **MUSE**, **Tables 4 and 5** demonstrate that BS-S reduces VerbMem/KnowMem more than NPO/RMU on MUSE-News, and on MUSE-Books it achieves complete forgetting (0/0) while preserving substantially more utility than other “fully forget” methods (e.g., UtilPres 0.3854 vs WGA 0.2519 and GradDiff ~0).  
   The coverage across synthetic privacy (TOFU), real copyrighted text (MUSE), and hazard knowledge (WMDP) lends credibility to the claimed generality.

5. **Use of LLM-as-a-judge to expose metric failures and assess semantic forgetting.**  
   The paper does not just bolt on another automated metric; it designs two targeted LaaJ probes, “Naturalness” and “Similarity”, and uses them to diagnose both baselines and the proposed methods. **Figure 4c** shows that BS-T and BS-S improve Naturalness compared to GradDiff and SimNPO while significantly reducing Similarity compared to NPO/RMU, aligning with the narrative that they avoid both collapse and paraphrasing. Given ongoing concerns about ROUGE-based evaluations, this is a meaningful contribution to methodology.

6. **Careful ablations and sensitivity analysis.**  
   The appendix provides non-trivial ablations:  
   - **Figure 5a–b** shows aggregate TOFU scores as a function of \(\lambda_{\mathrm{BST}}\), \(k\), \(\lambda_{\mathrm{BSS}}\), and \(N\), revealing that the interpolation weights are the critical knobs while neighborhood size and sample count have diminishing returns.  
   - **Table 6** evaluates BS-S with different underlying unlearning losses (GA, NPO, WGA, BS-T), and BS-S improves each base loss relative to its standalone version (compare with **Table 1**).  
   This supports the claim that the bootstrapping framework is not tightly coupled to one particular base objective.

7. **Implementation clarity and integration into an existing benchmark suite.**  
   Pseudocode for BS-T and BS-S in Appendix C is straightforward and faithful to the math, making reproduction realistic. Hyperparameter search spaces and training setups are detailed in Appendix E. Importantly, the authors state that their code is merged into OpenUnlearning, increasing practical impact and comparability.

## Weaknesses

1. **The “squeezing effect” mechanism, while plausible, is not fully disentangled from other phenomena.**  
   The paper attributes spurious unlearning almost entirely to normalization-driven probability shifts, but several confounding factors remain:  
   - In **Figure 2a**, the difference in LaaJ similarity between high-, mid-, and low-likelihood bands and between NPO vs retrain is reported qualitatively, but the figure lacks error bars or significance analysis. It is not clear how robust the ranking is across prompts, models, or decoding settings, nor whether high-likelihood regions are *always* the semantic neighborhood that matters.  
   - The probability dynamics in **Figures 2b–2c** do show increasing log-probability of high-likelihood bands under GA/NPO, but this could also be driven by more generic overfitting or token-level correlations rather than an inherently softmax-induced reallocation. For example, GA’s eventual collapse suggests that optimization instability interacts strongly with the effect, and the analysis does not quantify how much of the paraphrasing behavior can be explained solely by the residual structure vs. broader training dynamics.  
   A more careful causal analysis (e.g., intervening on high-likelihood neighborhoods directly and measuring semantic leakage) would make the mechanism claim more convincing.

2. **Theoretical analysis only cleanly applies to off-policy BS-S, yet experiments mostly use on-policy BS-S.**  
   Section 5 is explicit that Theorem 5.3 and the AKG-based dynamics hold for *off-policy* BS-S with a fixed set of augmentations, while in practice §4.2 and the experiments use an “on-policy” variant that periodically resamples high-likelihood sequences from the evolving model. Appendix D.4 explains why the AKG framework does not extend, due to the dependence of \(\mu_\theta\) on \(\theta\), but the paper then largely proceeds as if the same intuition carried over.  
   This is a non-trivial gap: the key argument for BS-S is that it aggregates BS-T residuals over a *fixed* belief-aligned set, yet in practice those beliefs move during training, potentially changing which regions get penalized and how stable the updates are. Without at least an empirical comparison of on-policy vs off-policy BS-S (under identical hyperparameters) or a more RL-style analysis, the theoretical claims for BS-S feel more like a post-hoc story than a solid explanation.

3. **Gradient-residual results are mathematically straightforward and somewhat oversold.**  
   The main new math is elementary: starting from the token-wise cross-entropy derivative \(\nabla_{z^i} \mathcal{L} = \pi^i - t^i\) (Eq. (10) in Appendix D.2), Theorem 5.2 simply substitutes \(t^i = (1-\lambda)e_{y_u^i} + \lambda q^i\) and compares to GA’s \(t^i = e_{y_u^i}\). The resulting identity \(\mathcal{G}_{\mathrm{BST}}^i[v] = \mathcal{G}_{\mathrm{GA}}^i[v] + \lambda q^i[v]\) for non-target \(v\) is straightforward algebra. Similarly, Theorem 5.3 is largely a linearity-of-gradients application over a weighted sum of BS-T losses.  
   There is nothing wrong with this, but it means the theoretical contribution is primarily interpretive rather than technically substantial. Given how much space §5 and Appendix D occupy and how central the theorems are in the abstract and contributions, the expectations they set are higher than what is actually delivered.

4. **Reliance on a single LLM judge and custom prompts raises concerns about evaluation robustness.**  
   While leveraging LLM-as-a-judge is a strength conceptually, the paper uses Gemini 2.5 Flash as the *sole* judge with a very specific scalar scoring scheme (0–5 with four decimal digits) and custom prompts (Appendix F.2). Potential issues:  
   - No inter-judge reliability or ablation with different judges (e.g., GPT, Claude) is reported, so it is unclear how sensitive the Naturalness and Similarity scores in **Figure 4c** are to the choice of judge.  
   - The prompts may encode particular values (e.g., focusing exclusively on “natural human speech” or “core semantic content”), and the paper does not show that they correlate with human labels.  
   - Some numerical differences are small (e.g., Similarity ~3.5 vs ~3.0), and no variance or confidence intervals are reported.  
   Since one of the main claims is that standard metrics are misleading and LaaJ-based metrics reveal spurious unlearning, the lack of a stronger validation for these LaaJ metrics undermines how much weight one should put on the LaaJ comparisons.

5. **Safety and ethical considerations of bootstrapping harmful generations are underexplored.**  
   BS-S explicitly samples “high-confidence responses” for harmful prompts and reuses them as training data (\(\hat{\mathcal{D}}_u\) in Eq. (7)). For datasets like WMDP, this entails having the system generate potentially detailed dual-use content (e.g., bio/cyber attack instructions) that are then stored and used in training. The Ethics Statement is very brief and mainly generic; there is no discussion of:  
   - How such generated harmful sequences are stored, filtered, or redacted.  
   - Whether some augmentations might be more detailed or dangerous than the original forget data, potentially increasing exposure risk during the unlearning process.  
   - Operational guidance for practitioners who might want to apply BS-S in more sensitive real-world unlearning settings.  
   Given the dual-use nature of WMDP and related tasks, these omissions matter for responsible deployment.

6. **Limited discussion of failure modes and trade-offs of the proposed methods.**  
   The paper focuses primarily on how BS-T/S improve over GA/NPO, but less attention is paid to where they still fail or over-forget:  
   - On TOFU 1% forget, **Table 1** shows that BS-T/BS-S sometimes reduce Utility compared to NPO or RMU (e.g., for Llama 3.2 3B, NPO Utility = 0.74 vs BS-S = 0.72). The text acknowledges BS-T as a “strong runner-up” but does not deeply analyze when bootstrapping starts to hurt retention more than baselines.  
   - In MUSE-Books (**Table 5**), UtilPres for BS-S (0.3854) is much lower than for SimNPO (0.6013), even though BS-S achieves perfect forgetting. The narrative calls this “slightly” outperforming other full-unlearning methods, but it does not confront the fact that there is a big gap to the best-utility method, suggesting bootstrapping may be overly aggressive in some regimes.  
   Detailed qualitative examples in Appendix F.4 do show some noisy or odd BS-S generations (e.g., Example 1 BS-S output with “translation agony-the-force-is-with-hers-win linguistic language”), but these are not fully integrated into the main discussion as limitations.

7. **Missing discussion and comparison to some recent LLM-unlearning works.**  
   Although Appendix B is quite comprehensive, several directly relevant recent works are not discussed (see the “Potentially Missing Related Work” section below), especially those focusing on “practical” or “robust” LLM unlearning and multi-objective formulations. This weakens the positioning of the proposed framework in the rapidly growing literature, and it is not obvious from the current text how BS-T/BS-S relate to, complement, or outperform these lines of work.

8. **Hyperparameter sensitivity is significant, but tuning burden is downplayed.**  
   **Figure 5a–b** makes clear that the interpolation weights \(\lambda_{\mathrm{BST}}\) and \(\lambda_{\mathrm{BSS}}\) have a large impact on Agg, with performance dropping sharply away from the optimal values (e.g., Agg falls from ~0.63 at \(\lambda_{\mathrm{BST}}=0.2\) to ~0.34 at \(\lambda_{\mathrm{BST}}=0.6\)). The text acknowledges some sensitivity but presents it mainly as an ablation. In practice, unlearning often happens under tight compute or data-access constraints, and it is not evident how many forget/retain mixtures or judge calls are required to tune these \(\lambda\)’s. A more candid and quantitative treatment of this tuning overhead would help practitioners assess feasibility.

## Potentially Missing Related Work

1. **Gao et al., “Practical Unlearning for Large Language Models”, 2024.**  
   This paper proposes efficient, scalable unlearning strategies specifically tailored to LLMs, aimed at making unlearning practical for deployment. It is directly relevant to the paper’s focus on usable unlearning and should be discussed alongside OpenUnlearning baselines in Section 2.2 or Appendix B.2, and ideally compared empirically or at least conceptually in terms of compute and effectiveness.

2. **Ren et al., “SoK: Machine Unlearning for Large Language Models”, 2025.**  
   A systematic taxonomy and survey of LLM unlearning approaches. It would be useful to reference this in the introductory motivation (§1) or the extended related work (§B.2) to position BS-T/BS-S within the SoK’s categories (e.g., objective-based vs representation-based vs structural methods), clarifying what gap the proposed framework is filling.

3. **Pan et al., “Multi-Objective Large Language Model Unlearning”, 2024.**  
   Introduces a multi-objective formulation that jointly optimizes forgetting and retention objectives, which is conceptually similar to balancing Mem and Util. This should be cited in Section 2.2 and in the discussion of the forget–retain trade-off (e.g., next to GradDiff/GRU), and could be a relevant baseline or at least a point of comparison for the claimed improved trade-offs of BS-T/BS-S.

4. **Gu et al., “Robust Unlearning for Large Language Models”, 2025.**  
   Focuses on robustness of unlearning against distribution shifts, jailbreaks, or adversarial queries. Given that the motivation of this paper includes preventing “spurious unlearning” that can be circumvented by paraphrasing prompts, this work should be discussed in the context of robustness in §3 or §6, and the authors could argue whether bootstrapping over model beliefs addresses similar threat models or is complementary.

## Questions

1. **On-policy vs off-policy BS-S:**  
   Can you provide empirical results comparing off-policy and on-policy BS-S for at least one TOFU and one MUSE/WMDP setting, holding all other hyperparameters fixed? This would help quantify how much the theoretical analysis of off-policy BS-S actually predicts behavior in the practically used on-policy version.

2. **Robustness of LaaJ evaluation:**  
   Have you tried alternative LLM judges (e.g., a different model family) or minor variants of the Similarity/Naturalness prompts? If so, do the relative rankings of GA/NPO/BS-T/BS-S in **Figure 4c** remain stable? Any calibration against human annotators on a small subset (even a dozen examples) would also greatly increase confidence in these metrics.

3. **Safety in sequence bootstrapping:**  
   For WMDP, could you clarify what safeguards (if any) you used when sampling high-confidence sequences for BS-S? For example, did you filter sequences with an external harmful-content detector before adding them to \(\hat{\mathcal{D}}_u\), or do you recommend such a step for practitioners?

4. **Failure modes of BS-S:**  
   The qualitative examples in Appendix F.4 suggest BS-S can generate somewhat incoherent or noisy text (e.g., Example 1’s BS-S response). Can you quantify how often BS-S harms Naturalness on retain prompts relative to baselines, and whether increasing \(\lambda_{\mathrm{BSS}}\) exacerbates this? A more detailed breakdown of when BS-S over-forgets or destabilizes generation would be useful for deciding when to prefer BS-T.

5. **Compute and tuning cost in realistic scenarios:**  
   Could you provide wall-clock times and GPU-days for the complete hyperparameter search process (including \(\lambda\) and \(k/N\) sweeps) on at least TOFU 10% with Llama 3.1 8B, and contrast this to a single NPO run? This would help assess whether BO or default settings might be necessary in practice.

## Flag For Ethics Review

No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The core ideas are technically sound, the objectives are well-defined, and the experiments are methodologically solid. Theoretical analysis is correct but somewhat limited in scope (mainly interpretive and off-policy), and the LaaJ evaluation could be more rigorously validated.

## Presentation Rating

3: good.  
The paper is generally clear and well organized; key equations (Eqs. (5)–(7)) and the AKG-based derivations are easy to follow, and figures/tables are informative. Some over-selling of the theoretical contribution and a lack of nuance in discussing limitations slightly detract from the clarity.

## Contribution Rating

3: good.  
The paper brings a useful new perspective on spurious unlearning, introduces a simple and broadly applicable bootstrapping framework, and provides strong empirical evidence across multiple benchmarks. The theoretical novelty is modest, and some related work is missing, but overall the contribution is meaningful for the LLM unlearning community.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The work offers a well-motivated and empirically convincing improvement over popular GA/NPO-style unlearning methods, with a clear practical take-away (incorporate model beliefs into the forget objective). At the same time, theoretical depth is limited, some evaluations (esp. LaaJ) are not fully validated, and positioning vs recent practical/robust unlearning work is incomplete. With clarifications and expanded discussion in response to the questions above, it would make a solid ICLR contribution.

## Reviewer Confidence

4: confident.  
I am familiar with LLM unlearning and learning-dynamics literature, have checked the math in §5 and Appendix D, and carefully examined the experimental tables/figures, though I have not independently rerun experiments.