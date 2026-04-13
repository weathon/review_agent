=== CALIBRATION EXAMPLE 12 ===

# Final Consolidated Review
## Summary
The paper proposes Influence-Guided Diffusion (IGD), a method for dataset distillation that uses trajectory influence functions to guide the sampling process of pre-trained diffusion models without fine-tuning them. The framework combines influence guidance (steering generation toward training-effective data) with deviation guidance (ensuring diversity), achieving state-of-the-art performance on ImageNet-1K (60.3% at IPC=50).

## Strengths
- **Novel methodological approach**: Using trajectory influence functions as guidance signals for diffusion sampling is innovative. Unlike prior diffusion-based distillation methods that fine-tune the generator (e.g., Minimax), this operates as a training-free sampling strategy on pre-trained models, offering a distinct paradigm for dataset distillation.

- **Strong and consistent empirical improvements**: The method demonstrates consistent gains across ImageNette, ImageWoof, and ImageNet-1K datasets. Table 2 shows 60.3% accuracy at IPC=50 on ImageNet-1K, surpassing prior state-of-the-art RDED (56.5%). Tables 1 and 3 demonstrate robust cross-architecture generalization across ConvNet-6, ResNet variants, MobileNet-V2, EfficientNet, and Swin Transformer.

- **Practical design choices with empirical validation**: The early-stage guidance strategy (Figure 2) and gradient-similarity-based checkpoint selection (Table 6) are empirically validated techniques that mitigate overfitting and improve efficiency. The ablation showing these choices improve over full-stage guidance provides actionable insights.

- **Synergistic guidance combination**: The ablation in Table 5 confirms that combining influence and deviation guidance yields complementary benefits, with the combined approach (81.0%, 84.4%) substantially outperforming either component alone (76.5%, 78.2%).

## Weaknesses
- **Misleading "training-free" framing in abstract**: The abstract claims the method works "without the need to retrain diffusion models," creating an impression of a purely sampling-time method. However, Section 4.1 reveals that training a surrogate ConvNet-6 on the full dataset for 50 epochs is required to collect checkpoints for influence calculation. This preprocessing overhead—potentially comparable to or greater than Minimax's fine-tuning cost—is not acknowledged in the framing and should be quantified.

- **Ablation reveals diversity guidance as primary driver**: Table 5 shows that for raw DiT, deviation guidance alone (G_D only: 78.2% at IPC=50) substantially outperforms influence guidance alone (G_I only: 76.5%). While the paper acknowledges diversity's importance, the theoretical framing around influence functions may overstate their role relative to the diversity mechanism, which appears to be the stronger contributor to performance gains.

- **Theoretical approximations insufficiently justified**: The derivation from Eq. 6 to Eq. 7 involves substituting class-specific checkpoints (θ^S) with full-dataset checkpoints (θ^T) and switching from dot product to cosine similarity. The paper claims these targets are "optimally equivalent," but this equivalence holds only at convergence (if z already provides optimal training dynamics), which is circular reasoning. The substitutions are presented as principled derivations when they are practical engineering choices.

- **Missing computational cost analysis**: Despite efficiency being a stated motivation, the paper provides no wall-clock time or FLOP comparison against Minimax or pixel-based methods. The overhead of computing backpropagation through the DDIM denoiser, decoder, and surrogate network at each guided step should be quantified to evaluate the efficiency claims.

- **No ablation of guidance window [A, B]**: The choice of applying guidance in steps [30, 45] out of 50 DDIM steps is presented as a key practical contribution, yet no sensitivity analysis is provided. Given that this parameter controls the trade-off between guidance strength and generation quality, its absence from ablations is notable.

- **Greedy diversity constraint**: The deviation guidance (Eq. 8) penalizes cosine similarity to only the single most similar previously-generated sample, not all pairs as stated. This greedy sequential approach depends on generation order and may not enforce the stated constraint effectively when many samples already exist.

## Nice-to-Haves
- Wall-clock time comparison against Minimax and pixel-level methods to substantiate efficiency claims
- Correlation analysis between computed influence scores (G_I) and downstream validation loss reduction to validate that the guidance measures training effectiveness
- Comparison with alternative guidance methods (classifier guidance, energy-based guidance) to isolate whether gains come from the influence formulation specifically
- Extension to other diffusion architectures (UNet-based) beyond DiT

## Removed Points
These points are flagged to be removed, treat them with caution:
- The citation of Fubini's Theorem being "unusual" for finite sums — this is a mathematical nitpick that does not affect the validity of the derivation
- Theoretical concerns about Taylor expansion and mini-batch approximations being "well-known to break down" — these are standard approximations in influence function literature and empirically validated by the results

## Novel Insights
The synthesis reveals that the paper's empirical contribution is substantial but the mechanism of improvement may differ from the theoretical framing. The ablation shows diversity enhancement is at least as important as influence guidance, suggesting the primary benefit comes from preventing mode collapse rather than optimizing for training influence specifically. The early-stage guidance finding—that applying strong guidance during semantic generation phases while allowing vanilla sampling for detail refinement—effectively avoids overfitting to the surrogate model and is a useful contribution for guided diffusion beyond dataset distillation.

## Suggestions
- Quantify total computational cost (surrogate training + guided sampling) against baselines in GPU hours to enable fair efficiency comparisons
- Add ablation for the guidance window [A, B] to justify the [30, 45] selection and provide guidance for other diffusion schedulers
- Acknowledge surrogate training cost prominently in the abstract or introduction rather than burying it in implementation details
- Consider analyzing the correlation between influence score and downstream accuracy across methods and datasets to validate the influence metric as a proxy for training effectiveness

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 6.0, 6.0, 6.0]
Average score: 6.4
Binary outcome: Accept
