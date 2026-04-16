## Summary
This paper proposes HMoRA, a PEFT method that combines mixtures of LoRA experts with a hierarchical hybrid router: shallow layers rely more on token-level routing while deeper layers rely more on task-level routing. It also introduces an entropy/GJS-inspired auxiliary loss intended to improve routing certainty while maintaining load balance, and reports gains over LoRA-style baselines and modest gains over full fine-tuning on multi-task multiple-choice benchmarks.

## Strengths
- **Clear and well-motivated method design.** The paper identifies three concrete limitations in prior MoE+PEFT work—uniform routing granularity across layers, weak generalization of task-level routing, and the certainty/balance tradeoff in routing—and proposes corresponding mechanisms: hierarchical token/task routing (Eq. 7–8), a task encoder for task representations (Eq. 6), and the CGJS-style auxiliary loss (Eq. 11–12).
- **The auxiliary loss is intuitive and better supported than many other claims.** Section 3.3 gives a clean entropy-based motivation: maximize entropy of the batch-averaged routing distribution for balance while reducing per-sample entropy for decisiveness. Table 1 shows consistent improvements from adding `L_aux` for both soft and top-k routing, and Figure 3 aligns with the intended certainty/balance effect.
- **Reasonable baseline suite for the stated setup.** The paper compares against Full FT, LoRA at two ranks, and several relevant mixture-of-LoRA baselines (MoLoRA, MixLoRA, HydraLoRA). Within this experimental setting, HMoRA is competitive and often best in average score.
- **Parameter-efficiency is meaningful in the reported setting.** The lightweight version uses 3.9% trainable parameters and still slightly exceeds full fine-tuning on average in Table 2. Even if the margin is modest, this is still a practically relevant result.
- **The paper is generally readable and experimentally structured.** Routing-method comparison, baseline comparison, and ablation sections are logically organized, and the method description is sufficiently concrete to follow.

## Weaknesses

###: Fatal
- None.

### Major:
- **The headline claim that HMoRA “outperforms full fine-tuning” is somewhat overstated relative to the evidence.** In Table 2, the gain over Full FT is modest: 63.88 vs 63.15 for HMoRA w LW and 64.16 vs 63.15 for HMoRA w/o LW. The paper states that each experiment is repeated 5 times and reports only the mean, but no standard deviations/confidence intervals are shown. Given the small margins, the lack of dispersion estimates matters for judging whether the improvement is robust rather than noise.
- **The evidence for the unseen-task generalization / unsupervised task differentiation claim is weaker than the wording suggests.** The paper repeatedly claims that the auxiliary loss lets the task router “differentiate tasks in an unsupervised manner and generalize to unseen tasks.” What is shown in the main text is: Figure 4 t-SNE plots on selected MMLU tasks, Table 3 showing a drop when removing the task-router auxiliary loss, and a brief statement that 42/57 MMLU subtasks are “differentiated” in Appendix E.8. This is suggestive, but not enough to fully establish the strong causal claim. In particular, the main text does not operationally define “differentiate” or provide a direct held-out-task generalization evaluation beyond clustering-style evidence on unseen benchmark subtasks.
- **The paper does not cleanly isolate the contribution of hierarchical routing itself.** This is the core architectural idea, but the main tables do not sharply disentangle it from the auxiliary loss and the broader HMoRA design. Section 4.1 uses only token-level routing. Table 2 evaluates the full system with top-2 routing, auxiliary loss, hierarchical routing, and sometimes lightweight variants combined. Section 4.3 summarizes appendix ablations for `epsilon` and `mu`, but the main body does not include a direct comparison against a strong non-hierarchical hybrid baseline with fixed mixing across layers. As a result, the central claim that the layerwise hierarchy is specifically responsible for the gains remains only partially substantiated.
- **Parameter-matched fairness is incomplete for the strongest variant.** HMoRA w/o LW achieves the best results, but uses 6.31% trainable parameters, compared with 4.78% for LoRA r=64 and about 3.2–4.0% for the mixture baselines. Since added adaptation budget can itself improve results, a tighter parameter-matched comparison would make the claimed advantage more convincing.

### Minor
- **Evaluation scope is narrower than the paper’s broad framing.** The paper is framed as making “LLMs more effective” for multi-task NLP, but the reported evaluation suite in the main text is entirely multiple-choice / classification-style benchmarks (MMLU, MMLU-Pro, ARC, OpenBookQA, SWAG, CommonsenseQA). This is enough to support claims about that setting, but not broader claims about general LLM effectiveness across open-ended generation tasks.
- **Model-scale evidence is limited.** The main text centers on Qwen2 1.5B, with LLaMA 3.2 1B results deferred to the appendix. This is not a flaw in itself, but it weakens claims about generality and scalability of the routing dynamics.
- **Inference/serving overhead is not analyzed in enough depth.** The method introduces routed LoRA experts and a task encoder. The paper reports training-time/parameter comparisons for lightweight designs (Figure 2c), but does not provide a fuller latency/memory analysis at inference time, which is relevant for a PEFT method motivated partly by efficiency and practicality.
- **The method introduces several hyperparameters whose robustness is only partially explored.** The paper discusses ablations for some settings (e.g., `gamma_c`, `epsilon`, `mu`) mostly in the appendix, but the overall sensitivity of the full method is still somewhat unclear.
- **The main text does not compare CGJS directly against unconstrained GJS, despite claiming the constrained form is necessary.** Section 3.3 states that directly optimizing GJS reduced performance and motivated CGJS, but the main paper does not show that comparison quantitatively.

### Trivial
- **Some key methodological details are pushed to appendices.** For example, routing implementation details, hierarchical schedule ablations, lightweight design studies, and the unseen-task differentiation metric are largely outside the main text. This does not invalidate the paper, but it makes the central evidence harder to assess from the main body alone.

## Nice-to-Haves
- Add a direct **hierarchical vs. flat hybrid routing** ablation with matched capacity/training budget.
- Report **mean ± std** across the 5 runs already performed, especially for Table 1 and Table 2.
- Include a more direct **held-out task-family evaluation** to support the unseen-task generalization claim.
- Add **inference latency and memory** comparisons against LoRA and MoE-LoRA baselines.
- Provide **per-task or per-expert specialization analysis**, e.g., expert activation heatmaps across layers and tasks.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper should compare against missing related methods.”** Removed per instruction not to speculate about missing related work.
- **Pure formatting/style complaints.** Not included.
- **Criticism that task-level routing is not really task-level because it is derived from the input only.** The paper is explicit that task routing is based on sentence/task representations from the input via a task encoder, and this is a valid design choice rather than a flaw.
- **Complaints about omitted low-level routing details in the main text.** The paper explicitly states “A detailed explanation of routing methods is provided in Appendix A,” so this is not a substantive weakness by itself.
- **Any criticism doubting the existence, release, or verifiability of cited models/datasets/benchmarks.** Removed by rule.

## Novel Insights
The paper is strongest when read not as a broad demonstration that hierarchical routing solves general multi-task PEFT, but as evidence for a narrower and interesting point: **routing regularization seems to matter at least as much as routing granularity**. Table 1 and Table 3 suggest that a substantial part of HMoRA’s gains may come from making routers more decisive and structured, especially for the task router, while the hierarchical schedule itself is less cleanly isolated. In other words, the paper likely contains two contributions of unequal maturity: the auxiliary loss is comparatively well supported, whereas the hierarchical layerwise routing story remains plausible but not yet decisively demonstrated.

## Suggestions
- Add a **fixed-α hybrid routing baseline** and compare it directly to the proposed increasing-α schedule.
- Report **standard deviations** for all main tables, since 5 runs were already performed.
- Make the unseen-task claim more precise by defining the “differentiation” metric in the main text and adding a more direct held-out-task experiment.
- Provide a **parameter-matched LoRA baseline** closer to HMoRA w/o LW’s trainable budget.
- Quantify **inference-time overhead** from the task encoder and routed LoRA experts.
- Tone down broad claims such as “makes LLMs more effective” and “outperforms full fine-tuning” to better match the actual empirical scope.

## Score and Decision
**Assessment by axis:**  
- **Originality:** Moderate. The combination of hierarchical token/task routing with a certainty-balance auxiliary loss is a meaningful incremental contribution.  
- **Importance:** Moderate. Efficient multi-task PEFT for LLMs is an important problem.  
- **Claims supported?:** Partially. The routing-loss improvements are supported, but the broader claims about unseen-task generalization and hierarchical routing are stronger than the evidence.  
- **Experimental soundness:** Reasonable but incomplete. Baselines are relevant, yet key isolation experiments and uncertainty reporting are missing.  
- **Clarity:** Good overall.  
- **Value to the community:** Positive, especially for researchers on MoE+PEFT, though the paper would benefit from tighter claim calibration.

**Calibration against retrieved human-reviewed anchors:**  
- Compared with **EvDeiLv7qc** (“Pushing Mixture of Experts to the Limit”), which received scores **6,5,8,8** and was accepted: that paper seems somewhat stronger empirically and more thoroughly validated relative to its claims. HMoRA is in a similar topic area and is competitive, but here the main claim-to-evidence gap is larger.  
- Compared with **IDJUscOjM3** (Self-MoE), which received **6,6,6,6**: HMoRA feels roughly in this range—interesting idea, useful experiments, but with nontrivial scope/validation limitations.  
- Compared with **LWvgajBmNH** (MoRE), which received **3,3,5,5** and was rejected: HMoRA is clearly stronger than this lower anchor because it has better motivation, stronger baseline coverage, and a more convincing empirical core.  
- Compared with **eWNEqdH0vk** (RMoE), which received **8,6,3,6**: like RMoE, HMoRA suffers from limited scale and modest gains, but HMoRA’s empirical package is a bit more complete for its intended fine-tuning setting.

Overall, this paper looks **borderline but slightly above reject**: a solid incremental systems/PEFT paper with a real contribution, but not one whose strongest claims are fully nailed down.

**Final score: 6.0 / 10**  
**Decision: Weak Accept**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>