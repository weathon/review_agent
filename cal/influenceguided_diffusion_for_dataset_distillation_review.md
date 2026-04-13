=== CALIBRATION EXAMPLE 27 ===

# Final Consolidated Review
## Summary
This paper proposes Influence-Guided Diffusion (IGD), a sampling-time method for dataset distillation that steers a pretrained or fine-tuned diffusion model toward generating training-effective synthetic images. The core idea is to connect dataset distillation to trajectory influence, then use an influence-based guidance term together with a diversity-promoting deviation term during diffusion sampling. Empirically, the method is quite strong: it consistently improves both vanilla DiT and Minimax across ImageNette, ImageWoof, and ImageNet-1K, including a reported 60.3% top-1 at IPC=50 on ImageNet-1K.

## Strengths
- **A genuinely useful reframing of large-scale dataset distillation as sampling-time controlled generation rather than pixel optimization or generator retraining.** This is a concrete conceptual shift, not just a rebranding: the method improves an existing diffusion model at inference time, and the paper shows it works both on raw DiT and on Minimax-tuned DiT.
- **Strong and consistent empirical gains over the most relevant diffusion baselines.** The most convincing evidence is not against older pixel-space methods, but against DiT and Minimax. Across Table 1 and Table 2, IGD gives sizable improvements, often around 4–7 points, and those gains persist on ImageNet-1K.
- **Clear evidence of complementarity rather than mere replacement.** IGD improves both pretrained DiT and Minimax-tuned DiT, and Minimax-IGD is usually best. That makes the contribution stronger than a baseline swap.
- **Cross-architecture usefulness is better supported than in many distillation papers.** The paper evaluates distilled data on multiple unseen architectures (ResNet101, MobileNet-V2, EfficientNet-B0, Swin in Table 3), and IGD-based datasets generally remain superior there, which is important because the distillation objective should not only fit one training architecture.
- **Ablations identify both guidance terms as materially useful.** Table 5 shows that both the influence term and deviation term matter, and their combination is strongest; this is especially helpful because the method could otherwise look like a monolithic heuristic.
- **The paper includes some nontrivial analysis beyond headline accuracy.** The checkpoint-selection ablation (Table 6), surrogate-architecture study (Table 4), and early-stage guidance analysis (Figure 2) all provide actionable understanding of where the gains come from and how the method behaves.

## Weaknesses

### Fatal
- None.

### Major:
- **The theoretical framing is stronger than what the paper actually establishes.** The paper presents a sequence of approximations from the dataset distillation objective to trajectory influence and then to the practical guided loss in Eq. (7). But several key steps are heuristic rather than derived guarantees. In particular, Section 3.2 says replacing synthetic-data checkpoints with real-data checkpoints is an “optimally equivalent target,” yet the stated equivalence only holds under the condition that the synthetic sample already reproduces the real training dynamics:  
  > “This equivalence holds because these two targets converge to the same optimal solution when \( \mathbf{z} \) can provide the same training dynamics as \( \mathcal{T}_c \)...”  
  That is a much weaker statement than practical equivalence. The method looks empirically effective, but the paper should present the influence connection as a motivated surrogate rather than a near-principled derivation.
- **The central claim that trajectory influence is the key reason for the gains is not isolated strongly enough from simpler alternatives.** The paper shows that IGD improves accuracy, and Table 5 shows the influence term helps. However, it does not compare against simpler sampling-time guidance signals computed from the same surrogate machinery, such as plain classifier loss, confidence, gradient norm, or reranking/rejection based on surrogate trainability. Without such controls, it remains unclear how much of the gain is specific to the trajectory-influence formulation versus coming from a broader class of surrogate-gradient guidance methods.
- **The ImageNet-1K headline claim is somewhat under-analyzed because the evaluation uses soft labels, but the contribution of that protocol is not disentangled.** Section 4.2 states:  
  > “Following the evaluation protocol of the RDED, we employ a ResNet-18 model, trained on the original dataset, to generate soft labels for synthetic images.”  
  This is a legitimate protocol choice and should not be treated as unfair by itself. However, because the headline large-scale result relies on this setup, the paper should better separate the gain from IGD itself versus the gain from the soft-label training protocol, e.g., by including a hard-label IGD result on ImageNet-1K or an explicit discussion of what exactly is being claimed as state of the art.
- **The practical overhead is not quantified carefully enough for a paper motivated partly by efficiency.** The paper does say results can be obtained on a single RTX 4090 and explains that early-stage guidance reduces runtime, but it does not provide wall-clock cost, generation time per image, checkpoint-storage cost, or precomputation cost for the surrogate checkpoints and averaged gradients. Since the method avoids retraining the diffusion model but still requires training a surrogate and storing/using checkpoint gradients, a more explicit cost breakdown is needed to support the practicality narrative.

### Minor
- **Several important design choices remain empirical, with limited sensitivity analysis.** These include the guided range \([A,B]\), the influence scale \(k\), deviation weight \(\gamma_t\), checkpoint similarity threshold, and surrogate architecture. Figure 2 studies \(k\) on ImageWoof, and Tables 4–6 help, but robustness across datasets/settings is not fully established.
- **The deviation guidance is effective but conceptually simple and weakly justified.** Eq. (8) is nearest-neighbor cosine repulsion against previously generated latents. The ablation suggests it matters a lot, but the paper does not analyze whether this local heuristic is enough to ensure meaningful global diversity or whether the gains mostly arise from generic diversification.
- **The connection between the global DD objective and the actual realized objective is narrower than the problem statement suggests.** Equation (1) is framed broadly in terms of preserving trainability for “any model initialized with parameters \(\theta_0\),” but the guidance is computed using one surrogate architecture and one training trajectory. The later cross-architecture results help empirically, but the paper should acknowledge this gap more directly.
- **Ablation and analysis are concentrated on the smaller subsets rather than the main large-scale benchmark.** The paper’s strongest claim is on ImageNet-1K, yet the deeper analyses (component ablations, checkpoint selection, guidance schedule) are all on ImageNette/ImageWoof. This weakens confidence that the same mechanisms explain the main benchmark gains.

### Trivial
- **The paper could discuss limitations more openly.** For example, Table 4 shows some architecture dependence in the surrogate used for influence, and Figure 2 shows guidance can overfit or degrade image quality when pushed too hard. These are useful findings, but they are presented mostly as tuning observations rather than explicit limitations.

## Nice-to-Haves
- Add a direct comparison to simpler guidance baselines using the same surrogate, such as classifier-loss guidance, confidence guidance, gradient-norm guidance, or post-hoc reranking of vanilla samples.
- Quantify the approximation gap behind Eq. (7) on a smaller setting where both synthetic-trajectory and real-trajectory versions are computable.
- Report a full computational cost table: surrogate training, checkpoint filtering/precomputation, sampling cost per image, and total cost to build the distilled set.
- Provide at least one ablation on ImageNet-1K itself, especially for the influence/deviation terms or the guidance schedule.
- Analyze diversity more directly with quantitative within-class diversity/coverage metrics, not only accuracy and a t-SNE/Wasserstein visualization.
- Include a hard-label ImageNet-1K result to disentangle the role of soft labels from the effect of IGD.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing related methods X/Y/Z”** — Removed because external completeness of related work cannot be verified here, and the review should not speculate about uncited papers.
- **Generic reproducibility complaints about appendix hyperparameters or implementation details** — Weakened/removed as standalone criticisms. The main issue is not missing trivial details but the lack of sensitivity/cost analysis for method-defining choices.
- **Claims doubting fairness because baselines may not have received equal tuning effort** — Removed in the strong form. The paper clearly compares IGD as an add-on to DiT and Minimax, and asymmetry that favors the baseline is not a valid criticism under the reviewing rules. The remaining valid issue is that the method has several tunable inference-time knobs whose robustness is only partially characterized.
- **Overly broad criticism that evaluation is “narrow” because it only covers ImageNet-1K and two subsets** — Removed as overstated. For this topic, ImageNet-1K plus ImageNette/ImageWoof is already a meaningful and relevant evaluation suite, especially with cross-architecture testing.
- **Complaints about formatting/parser artifacts or missing appendix text in the extracted PDF content** — Removed; not paper issues.

## Novel Insights
The strongest reading of this paper is not “influence functions solve dataset distillation,” but rather that **sampling-time guidance can be a surprisingly powerful lever for distillation when the generator is already strong**. The empirical results suggest that a good portion of the remaining gap in diffusion-based distillation is not necessarily in retraining the generator, but in selecting which regions of the learned distribution to sample from. IGD’s success on top of both raw DiT and Minimax supports this interpretation. At the same time, the ablations indicate that training-effectiveness alone is insufficient: the large contribution of deviation guidance implies that avoiding redundant training signals is at least as important as promoting individually influential ones.

## Suggestions
- Recast the theoretical claims more carefully: describe Eq. (7) as a practical surrogate inspired by trajectory influence, not as an equivalence-derived objective.
- Add a control experiment comparing influence guidance against simpler surrogate-based guidance signals.
- Include a small-scale experiment directly measuring the gap between guidance computed from \(\theta_e^S\) and \(\theta_e^{\mathcal T}\).
- Report end-to-end cost numbers so readers can judge the efficiency tradeoff against Minimax and other DD methods.
- Add at least one ImageNet-1K ablation and one hard-label ImageNet-1K result.
- Expand the discussion of failure modes, especially over-guidance, surrogate mismatch, and diversity limitations.



# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 6.0, 6.0, 6.0]
Average score: 6.4
Binary outcome: Accept
