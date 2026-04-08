=== CALIBRATION EXAMPLE 28 ===

# Final Consolidated Review
## SummaryThe paper derives the Maximal Update Parametrization (µP) for two state-of-the-art learned optimizer architectures (VeLO and small_fc_lopt) and proposes a simple multi-width meta-training recipe for µ-parameterized learned optimizers (µLOs). Empirical evaluation demonstrates that µLOs meta-trained on small MLP tasks at low cost (100 GPU hours) substantially improve meta-generalization to wider unseen networks compared to standard-parameterization LOs and transferred hand-designed optimizers, with additional empirical observations of improved generalization to deeper networks and longer training horizons.

## Strengths

- **Novel theoretical derivation bridging µP and learned optimizers.** The paper provides the first derivation of µP for learned optimizer architectures (Propositions 4.1 and 4.2), establishing a formal connection between hyperparameter transfer theory and LO meta-generalization. This bridges two previously separate subfields and opens a new research direction. The derivation handles the non-trivial interaction between the LO's output parameterization (direction/magnitude with exponential scaling) and the optimizee's width-dependent scaling.

- **Strong empirical demonstration of width generalization.** Table 1 shows µLO_M and µVeLO_M consistently achieving the best and second-best average ranks across OOD width tasks, outperforming per-task-tuned AdamW and µAdam. Figure 4 shows SP LOs diverge or stall on large-width tasks while µLOs maintain smooth training curves, providing compelling visual evidence of the core claim.

- **Practically significant meta-training cost reduction.** The paper demonstrates that µLO_M meta-trained for only 100 GPU hours on small MLP tasks (widths 128–1024) can effectively optimize networks at width 8192+. This contrasts sharply with prior work (VeLO) requiring 4000 TPU-months yet still failing on wider tasks, representing a meaningful practical advance.

- **Open-source implementation.** The code is publicly available, facilitating verification and follow-up work.

## Weaknesses

- **Input feature scaling to the LO is not explicitly discussed in the main text.** The derivation in Section 4 covers optimizee initialization, pre-activation multipliers, and output update scaling (Equation 3), but does not address how the LO's input features $\mathbf{u}_t$ (gradients, momentum accumulators, variance accumulators) behave under µP. These features have width-dependent scaling (e.g., gradients scale as $1/\sqrt{\text{fan\_in}}$), so a fixed-parameter LO $f_\omega$ receives differently scaled inputs at different widths. The propositions assume "parameters and input data become aligned, leading to LLN scaling," which presumably addresses this, but the main text offers no discussion of whether and how input features remain well-behaved across widths. This is a notable presentation gap that affects reproducibility and the reader's ability to assess theoretical completeness.

- **Evaluation reports only training loss; no validation or test metrics are provided.** All results throughout the paper (Figures 3–6, Table 1) report training loss exclusively. While training loss is a direct metric for optimizer capability, the paper's own related work identifies "training optimizees that do not overfit" as a core challenge for LOs. Without test loss or accuracy, it is impossible to assess whether µLOs inadvertently encourage overfitting relative to hand-designed optimizers. This limits the practical significance of the reported improvements.

- **Hand-designed baselines are tuned at a proxy width, not at the actual evaluation widths.** AdamW and µAdam are hyperparameter-searched at width=1024 and then evaluated at widths 2048–8192+. The paper acknowledges in Section 6 that an oracle AdamW (tuned at each test width) is missing due to compute constraints. This asymmetry means the claim that µLOs "outperform hand-designed optimizers" (Abstract, Conclusion) is only established against *transferred* hyperparameters, not against the best achievable hand-designed optimizer performance. The gap could be substantial at extreme widths.

- **No ablation isolating the contribution of individual µP components.** The method comprises three distinct modifications: initialization scaling (Initialize-µ), pre-activation multipliers (Multipliers-µ), and update scaling (Optimizer Update Scaling-µ). No experiment isolates which component(s) drive the generalization gains. Without this, it is unclear whether the full µP derivation is necessary or whether a subset of modifications (e.g., update scaling alone) would suffice.

- **Theoretical propositions rely on a strong LLN alignment assumption without empirical validation.** Propositions 4.1 and 4.2 assume that "during training the optimizee's parameters and input data become aligned, leading to Law of Large Numbers scaling." This alignment condition is non-trivial—it may not hold early in training, for specific architectures, or in the presence of normalization layers. No empirical evidence is provided to verify when this assumption holds or breaks, leaving the theoretical guarantee with uncertain operational scope.

- **Architecture generalization (MLP→ViT/LM) is presented without adequate caveats.** µLOs are meta-trained exclusively on MLPs yet evaluated on ViTs and language models. µP theory guarantees width scaling for a given architecture class; it does not guarantee that an optimizer trained on MLP gradients generalizes to attention-based architectures with different gradient statistics. The results on ViT and LM tasks (Figures 4d,e) show µLOs merely "nearly match" rather than outperform hand-designed baselines—a qualitatively weaker result than on MLP tasks. The paper should be more explicit that architecture transfer is an empirical finding not covered by µP theory, and the relative weakness of transformer results should be discussed.

- **Depth and horizon generalization are unexplained empirical observations presented prominently.** The paper highlights these as key findings in the Abstract and Conclusion, yet Section 5.2.4 explicitly states they are "purely empirical" with "no theoretical justification." The hypothesis that "µP's stabilizing effect on the optimizee's activations leads to this improvement" is plausible but speculative, supported only by qualitative pre-activation variance plots (Figure 2). Presenting these findings at the same level of confidence as the theoretically grounded width transfer may mislead readers about the scope of µP's guarantees.

## Nice-to-Haves

- **Quantitative analysis of LO internal states across widths.** Examining whether the LO's outputs (direction $d$, magnitude $m$, and tensor-level learning rate $\varepsilon_{\mathbf{W}}$ in VeLO) follow the predicted µP scaling across widths would directly validate the mechanism and strengthen the theoretical narrative.

- **Explicit compute cost comparison between meta-training and per-task tuning.** The paper provides 100 GPU hours for µLO meta-training and 500+ configurations per task for Adam tuning, but does not compute the total tuning cost across all 35 tasks. A direct cost comparison would substantiate the "compute-efficient" framing more concretely.

- **Failure mode analysis or boundary identification.** All reported results show µLOs succeeding; no tasks are shown where µLOs underperform. Identifying the conditions under which µLOs break down (e.g., extremely small widths, architectures with unusual normalization) would strengthen the meta-generalization claims by establishing their scope.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Weakness: "No evidence Adam/SGD converge to global optimum at optimal rate" as weak motivation.** This is a generic statement in the introduction, not a substantive weakness of the paper's contributions. Most optimization papers include similar framing; it does not affect the paper's technical quality.

- **Weakness: LO parameters ω may require µP scaling for stable meta-training.** This is scope creep. The paper focuses on test-time transfer of the LO to wider optimizees; whether the LO itself needs µP parameterization during meta-training is a separate research question outside the stated scope.

- **Weakness: Missing Broader Impact section.** ICLR does not mandate a dedicated Broader Impact section. This is a formatting/style nitpick.

- **Weakness: Formatting/OCR artifacts in equations and figure captions.** These are explicitly parser issues, not author errors, and are to be ignored per review instructions.

- **Weakness: CompleteP (Dey et al., 2025) not used as baseline.** CompleteP is cited as concurrent work for depth+width transfer. The paper focuses on µP for LOs; asking for a different parameterization as a baseline is asking the paper to address a different research question. The authors already discuss CompleteP in related work and identify it as future work.

- **Weakness: Meta-training should include non-MLP architectures.** The paper's key finding is that µP enables MLP-meta-trained LOs to generalize to ViTs/LMs without meta-training on those architectures. Requesting meta-training on transformers as a *weakness* contradicts the paper's demonstrated result. It would be valuable as an ablation (nice-to-have) but is not a flaw.

## Novel Insights

The synthesis of the three reviews reveals a key insight: the paper's most impactful contribution may not be the theoretical µP derivation itself (which follows established tensor program machinery), but rather the empirical demonstration that µP's stabilizing effect on optimizee dynamics acts as an implicit regularizer on LO meta-generalization across *multiple* axes simultaneously—width, depth, and horizon—even though the theory only covers width. This suggests that the practical value of µP for learned optimizers extends beyond its theoretical guarantees, potentially because stable training dynamics are a prerequisite for any form of generalization, and µP enforces this stability structurally. Investigating whether this "stability-as-regularizer" hypothesis generalizes to other meta-learning settings (e.g., learned learning rate schedulers, data augmentation policies) would be a fruitful direction. Additionally, the relatively weaker transformer results compared to MLP results hint that the gap between theoretical coverage and empirical benefit may widen as architecture complexity increases, suggesting that future parameterizations (like CompleteP) may be needed for robust architecture-level transfer.

## Suggestions

- Add an explicit subsection or paragraph in Section 4 discussing how the LO's input features (gradients, momentum, variance) behave under µP, and clarify whether any input preprocessing or normalization is needed to ensure width-invariant LO behavior. Even if the propositions handle this, the main text should make the logic transparent.

- Report at least test accuracy or validation loss for the main evaluation tasks (e.g., Figure 4 tasks) to demonstrate that µLOs do not trade optimization speed for generalization quality.

- Include an ablation study isolating the three µP components (initialization, multipliers, update scaling) on at least one width-transfer task to identify which factors are necessary vs. sufficient.

- Temper the Abstract and Conclusion claims about "outperforming hand-designed optimizers" to specify that this is against *transferred* hyperparameters, or add a single-task oracle AdamW experiment to substantiate the stronger claim.

- Present depth and horizon generalization findings with appropriate caveats in the Abstract, clearly distinguishing them from the theoretically grounded width transfer results.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 2.0]
Average score: 5.0
Binary outcome: Accept
