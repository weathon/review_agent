## Summary
This paper proposes TAK (Task Arithmetic with KFAC regularization), a method to improve weight disentanglement in task arithmetic without requiring access to other tasks' data. By linking representation drift regularization to the Generalized Gauss-Newton matrix and approximating it via a Kronecker-Factored Approximate Curvature (KFAC), the authors derive a dataless regularizer. A key innovation is a heuristic to merge per-task curvature factors into a single surrogate, achieving constant memory and computational complexity in the number of tasks. The method demonstrates state-of-the-art performance on task addition and negation benchmarks in vision and language, offers robustness to task vector rescaling, and promotes clear task localization.

## Strengths
- **Dataless regularization matching data-dependent performance:** In the linearized fine-tuning regime, TAK achieves performance on par with or superior to the data-dependent τJp method on the 8 Vision benchmark (Tables 1, 2) and strong results on task negation, a significant advantage for privacy and modularity.
- **Scalable design with constant complexity:** The proposed aggregation of per-task KFAC factors into a single surrogate (Eq. 8) reduces storage and computation from O(T) to O(1) in the number of tasks, validated empirically with minimal performance drop (Table 3, Fig. 6).
- **Empirically validated robustness and task localization:** The method exhibits strong robustness to the task vector scaling coefficient α (Figs. 4, 11), often eliminating the need for held-out tuning. It also induces clear task localization, as shown by the separation between in-task and out-of-task Jacobian-projected outputs (Fig. 5, 13, 14).

## Weaknesses
- **Suboptimal performance in the language domain:** On T5-base, TAK is consistently outperformed by the data-dependent τJp method (Table 1, Fig. 3). This suggests the curvature approximation or the linearization assumption may be less effective for textual tasks, limiting the method's universality.
- **High memory footprint for the KFAC factors:** Storing the full KFAC matrices requires quadratic memory in layer dimensions, which can be prohibitive for very large models (Appendix B). While compression is explored (Fig. 7b), it involves a non-trivial accuracy-storage trade-off.
- **Theoretical gap for the non-linear regime:** The method is derived and theoretically justified for linearized fine-tuning. Its application to the non-linear regime (via pairing with Attention-Only Fine-Tuning) is empirically motivated but lacks a theoretical grounding, making its effectiveness in standard non-linear fine-tuning less certain.

## Nice-to-Haves
- A quantitative metric (e.g., AUC) to summarize the task localization separation shown in Figures 5, 13, and 14, enabling more direct comparison across methods.
- Exploration of combining TAK with parameter-efficient fine-tuning (PEFT) techniques like LoRA, which could further reduce memory demands and broaden applicability.
- A more detailed analysis of why the performance gap with τJp exists in the language domain, potentially guiding improvements for textual tasks.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Statistical reporting lacks confidence intervals."** While reporting standard deviations is good practice, the paper's core claims are supported by extensive multi-seed ablations (Table 5) and consistent trends across models and tasks. The absence of error bars in main tables does not invalidate the results.
- **Weakness: "Need to compare against random matrix approximations or simple norm penalties."** The paper already includes a strong and relevant baseline: diagonal GGN approximation (Porrello et al., 2025). Adding more simplistic regularizers does not strengthen the evaluation of the specific curvature-based contribution.
- **Missing Experiment: "Scale to 20+ tasks to prove constant complexity."** The claim of constant complexity is algorithmic (O(1) storage). The empirical validation on 8 tasks (Table 3) and the analysis of the merging heuristic (Appendix C) are sufficient; demanding an arbitrary larger scale is scope creep.
- **Missing Experiment: "Evaluate on standard full-parameter non-linear FT."** The paper explicitly scopes its non-linear application to settings that induce approximately linear behavior (Attention-Only FT). Demanding evaluation on standard non-linear FT asks the method to operate outside its designed and justified regime.
- **Weakness: "Heuristic merging lacks theoretical justification."** Appendix C provides a formal bound on the approximation error of the merging heuristic. This is a reasonable theoretical contribution for a primarily empirical paper.

## Novel Insights
The paper provides a novel connection between representation drift regularization for weight disentanglement and second-order optimization techniques. It demonstrates that a well-known curvature approximation (KFAC) can be repurposed as a dataless regularizer that effectively prevents cross-task interference. Furthermore, the insight that per-task curvature factors can be aggregated into a single surrogate with constant complexity without significant performance loss is a key contribution for scalable multi-task model editing.

## Suggestions
- In the main text, include a brief intuitive explanation or summary of the merging error bound from Appendix C to make the heuristic's justification more accessible.
- Commit to releasing the pre-computed KFAC factors for the models and tasks used in the paper alongside the code, as this aligns perfectly with the vision of sharing dataless "assets" for downstream applications.