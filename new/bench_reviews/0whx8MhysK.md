The paper content is already provided in the user message. Let me verify the key claims:

1. **Claim: "training-free" is misleading** - The paper abstract says "without the need to retrain diffusion models" and Section 4.2 calls it a "training-free sampling framework." Section 4.1 says "All the experimental results of our method can be obtained on a single RTX 4090 GPU." The method does require training a ConvNet-6 on the full dataset - let me verify.

From the paper (Section 4.1): "For each test dataset, we train a 6-layer ConvNet (ConvNet-6) for 50 epochs with the learning rate 1×10⁻² to collect the surrogate checkpoints used in Equation (7)."

So the "training-free" claim specifically refers to not needing to retrain the diffusion model, which is accurate. However, the overall pipeline does require full-dataset training of the surrogate model.

2. **Claim about theoretical circularity** - Let me verify the exact quote about θ^T vs θ^S.

From Section 3.2: "Replacing the checkpoints θ_e^S with θ_e^{T_c} in Equation (6) is an optimally equivalent target. This equivalence holds because these two targets converge to the same optimal solution when z can provide the same training dynamics as T_c, i.e., ∇̄_θ ℓ_c(X_c; θ_e^{T_c}) = ∇_θ ℓ_c(D(z); θ_e^{T_c}) ∀e ∈ [0, E]."

This is indeed circular as the harsh critic notes - the condition for equivalence is exactly what dataset distillation aims to achieve.

3. **Claim about influence not being valid** - The harsh critic argues that since θ^T are fixed checkpoints from full data training, the "influence" being measured is not true trajectory influence. This is correct - it's a gradient alignment heuristic rather than true trajectory influence.

Now let me synthesize the review, being careful about what to keep and what to remove.

On the "training-free" framing: The paper is careful to say "without the need to retrain diffusion models" - this is accurate. The harsh critic's claim that the paper "repeatedly" presents it as "training-free" in an misleading way needs verification. Section 4.2 does call it a "training-free sampling framework" which is technically accurate for the sampling part. However, the overall pipeline does require full dataset training. This is a real but minor concern about framing - the paper should be more explicit.

On the theoretical overclaim: This is a substantial issue. The connection from Eq. 1 to Eq. 7 involves multiple approximations that are not rigorously justified, and the "optimal equivalence" argument is circular. However, the empirical results are strong, which suggests the gradient alignment heuristic works well in practice.

Let me now consider the Spark reviewer's points about missing experiments. Some are valid (comparison with other guidance schemes, computational cost), but some ask for experiments outside the paper's scope (CIFAR-10/100 benchmarks).

Now let me write the final review.</think>Now I have sufficient calibration data. Let me synthesize the final review.

## Summary

The paper proposes Influence-Guided Diffusion (IGD), a framework that frames dataset distillation as a guided diffusion generation problem. By connecting the dataset distillation objective to the trajectory influence function, IGD uses gradient-similarity-based guidance (computed against checkpoints from a model trained on full data) and a diversity-promoting deviation guidance to steer a pre-trained diffusion model's sampling process toward generating training-effective data, achieving state-of-the-art ImageNet-1K results (60.3% at IPC=50) without retraining the diffusion model.

## Strengths

- **Strong and consistent empirical improvements**: IGD provides substantial gains over both vanilla DiT (5.8–6.9% on ImageNet-1K) and Minimax-tuned DiT (1.2–4.7%), with improvements consistent across IPC settings, test architectures (ConvNet, ResNet, MobileNet, EfficientNet, Swin), and datasets (ImageNette, ImageWoof, ImageNet-1K). The 4.0% improvement over prior SOTA (RDED) on ImageNet-1K at IPC=50 is meaningful.

- **Novel conceptual framing**: Connecting dataset distillation objectives to the trajectory influence function and using this as guidance for diffusion sampling is creative and conceptually appealing. Framing dataset distillation as conditional sampling from p(x|Condition="training-effective") provides a principled direction for the field.

- **Practical plug-and-play design**: IGD operates as a sampling-time modification to existing diffusion models, requiring no retraining or fine-tuning of the generator. This makes it complementary to methods like Minimax and easily applicable to future diffusion architectures.

- **Thorough ablation studies**: Table 5 cleanly separates the contributions of influence guidance and deviation guidance. The checkpoint selection strategy (Table 6) and early-stage guidance analysis (Figure 2) are well-motivated and empirically validated. Cross-architecture robustness of the surrogate (Table 4) is an important practical check.

## Weaknesses

### Fatal
None.

### Major

- **Theoretical narrative overclaims the rigor of the connection to the distillation objective** — The paper frames IGD as deriving from the dataset distillation objective (Eq. 1) via trajectory influence, but the actual implementation involves two significant departures from this derivation that collectively mean the "influence guidance" is fundamentally a heuristic, not a principled approximation to Eq. 1. (1) The substitution of θ_e^S with θ_e^T (Section 3.2) is justified by claiming "optimal equivalence" when synthetic gradients match real gradients — but this condition is precisely what dataset distillation seeks to achieve, making the argument circular. The paper provides no analysis of the approximation error introduced by this switch. (2) Trajectory influence (Eq. 5) is defined as the influence of a single training point along a trajectory that includes that point; here, guidance is computed against θ_e^T, a trajectory completely independent of the synthetic data, so the "influence" is really a gradient-alignment score, not a valid influence measure in the original sense. (3) Even if Eq. 7 were correctly computed, optimizing per-sample influence does not necessarily imply that a small *set* of such samples optimizes the population-level distillation objective (Eq. 1). The paper would be more honest and equally strong if it presented IGD as a well-motivated gradient-alignment heuristic rather than a principled derivation from Eq. 1.

- **The overall pipeline requires full-dataset training, which partially contradicts the distillation motivation** — The method requires training a surrogate ConvNet-6 on the full original dataset for 50 epochs, saving checkpoints and computing per-class averaged gradients. While this is clearly stated in Section 4.1, it is underemphasized relative to the repeated "training-free" and "efficient" claims. The sampling is training-free w.r.t. the diffusion model, but the overall distillation pipeline presupposes a complete training run on the full dataset. This means IGD cannot be used in the setting where dataset distillation is most motivated — when you want to *avoid* training on the full dataset. The paper does not quantify the total computational cost, making it difficult to assess efficiency claims relative to baselines. This does not invalidate the method (post-hoc dataset summarization after full training is still useful), but it materially changes the problem being solved and should be discussed forthrightly.

### Minor

- **Hyperparameter sensitivity lacks systematic study** — The method relies on several critical hyperparameters (guidance scaling factor k, deviation coefficient γ_t, guided range [A, B], gradient similarity threshold). Figure 2 shows sensitivity to k on ImageWoof/IPC=100, but no systematic study across datasets, IPC values, or architectures is provided. The appendix pointers for hyperparameter details offer some coverage, but the main text lacks guidance on how to set these parameters for new tasks.

- **Deviation guidance lacks principled justification** — The diversity constraint (Eq. 8) pushes new samples away from the nearest neighbor in latent space, but this is ad hoc. There is no analysis connecting cosine distance in the VAE latent space to training set diversity in a way that optimizes the distillation objective. Alternative diversity mechanisms (feature-space repulsion, submodular selection, etc.) are not compared. The ablation (Table 5) shows deviation guidance alone on DiT produces comparable gains to Minimax, suggesting diversity alone is quite powerful, but the interplay between the two guidance terms is not analyzed beyond showing they are complementary.

- **ImageNet-1K comparisons conflate backbone vs. guidance contributions** — Tables 2–3 compare Minimax-IGD against RDED, SRe²L, and G-VBSM, which use entirely different generation backbones. While the paper cleanly shows IGD's improvement over DiT and Minimax separately, the claim of "state-of-the-art" on ImageNet-1K against methods that use pixel-level optimization or different architectures is not an apples-to-apples comparison. This is partially acknowledged but should be stated more explicitly.

### Trivial
- The paper's abstract claims "an exceptional result" on ImageNet-1K, which is slightly overselling, as 60.3% at IPC=50 is a meaningful improvement but still far from full-data performance.

## Nice-to-Haves

- Comparison with simpler guidance schemes (e.g., classifier guidance, loss-based guidance) on the same diffusion backbone with matched compute, to isolate whether the gains come from guidance-in-general or influence-specific guidance.
- A systematic ablation over the guided sampling range [A, B] across different total sampling step counts and datasets.
- Evaluation on CIFAR-10/100 at standard IPC values to situate IGD within the broader dataset distillation literature.
- Per-class accuracy breakdown on ImageNet-1K to understand whether gains are concentrated in easy or hard classes.

## Removed Points

These points are flagged to be removed — treat them with caution:

- **"No comparison with classifier-free guidance baselines"** (from Spark): While a reasonable suggestion for future work, this is not a baseline the paper should be penalized for omitting. CFG operates in a fundamentally different regime (class conditioning) and does not optimize for training effectiveness. This belongs in Nice-to-Haves.

- **"No evaluation on CIFAR-10/100"** (from Spark): The paper explicitly scopes to large-scale, high-resolution distillation tasks (Section 4.1). Requesting additional experiments outside this scope is scope creep. Moved to Nice-to-Haves.

- **"No ImageNet-1K cross-architecture results for vanilla DiT and Minimax without IGD"** (from Spark): The harsh critic also raised this. However, the paper does provide cross-architecture comparisons for IGD methods against RDED (Table 3), and the Woof/Nette experiments include both with and without IGD. While Table 3 could include vanilla DiT/Minimax results, this is a presentation choice, not a methodological flaw.

- **"No analysis of first-order Taylor expansion error"** (from Spark): This is a standard approximation used throughout the influence function literature (Pruthi et al., 2020). Requesting novel error analysis beyond what the original influence function paper provides is an unreasonable standard for an applied paper.

- **"The 'training-free' framing misleads readers"** (from Harsh Critic, as a structural issue): The paper specifically says "training-free" in the context of the *diffusion model* sampling, and Section 4.1 clearly states the surrogate training requirement. The wording could be more explicit, but this is a framing concern, not a methodological error. Downgraded from structural to minor.

- **"No runtime or compute cost comparison"** (from Spark, partially endorsed by Harsh Critic): The paper states all experiments run on a single RTX 4090. While a detailed cost breakdown would be useful, this is not uncommon for dataset distillation papers. Moved to Minor for the specific concern about total pipeline cost, but the general reproducibility complaint is a nitpick.

- **"Per-class accuracy breakdown"** and **"failure case gallery"** (from Spark): These are nice-to-have visualizations, not necessary for evaluating the core claims. Moved to Nice-to-Haves.

- **"Soft labeling protocol advantages diffusion methods"** (from Harsh Critic): The paper follows RDED's protocol for fair comparison on ImageNet-1K, which is standard practice. The concern about label protocol is speculative without evidence that soft labels favor diffusion methods differently.

## Novel Insights

The cross-architecture surrogate analysis (Table 4) reveals an interesting finding: ConvNet-6, the smallest surrogate, produces the best influence guidance despite its limited capacity. This suggests that the influence signal captures generic gradient structure (shared across architectures) rather than architecture-specific features. Combined with Figure 3's observation that higher Wasserstein distance to the original distribution (Minimax-IGD) can yield better accuracy than lower distance (DiT-IGD), this hints that the "training-effective" conditional distribution is not simply closer to the real distribution — it involves a trade-off between alignment and diversity that the influence heuristic captures without explicitly optimizing distribution matching. This partially undermines the paper's own framing while suggesting a deeper principle at work.

## Suggestions

- Reframe the theoretical contribution honestly: present IGD as a gradient-alignment heuristic inspired by trajectory influence, rather than claiming it approximates the distillation objective in Eq. 1. This preserves the conceptual motivation without the circular equivalence argument.
- Quantify the total computational cost of the pipeline (surrogate training + gradient computation + guided sampling) and compare it to baselines like RDED and Minimax end-to-end.
- Discuss the practical implications of requiring full-dataset training upfront — the method is best understood as post-hoc dataset summarization, which is useful for dataset publishing, caching, or repeated training scenarios, rather than as a replacement for training on the full dataset.

## Score and Decision

**Calibration anchors:**
- MGD³ (vKJ8YH0iNp): Dataset distillation + diffusion guidance, SOTA claims but weaker theoretical support and limited novelty. Scores: 8, 6, 3, 3 → Reject.
- CONCORD (CaexTRYaN6): Similar domain, heuristic modifications to diffusion, limited gains. Scores: 5, 5, 5, 3 → Withdrawn/Reject.
- Trajectory influence paper (uHLgDEgiS5): Principled theoretical contribution to influence, strong empirical results, accepted as Oral with 8,8,8,8.
- DDiF - Distilling Dataset into Neural Field (nCrJD7qPJN): Solid empirical contribution in dataset distillation, new parameterization, scores 5,6,5,6 → Accept Poster.

This paper sits between these extremes. The empirical results are substantially stronger than MGD³ and CONCORD, with consistent and meaningful improvements across benchmarks. The conceptual framing (connecting influence functions to diffusion guidance) is novel and influential for the field. However, the theoretical narrative overclaims the rigor of the derivation (the circular θ^T substitution and the gap from per-sample influence to set-level distillation objectives), and the full-dataset training requirement is under-discussed relative to the "efficient/distillation" framing. These are significant but not fatal issues — the method works well empirically and introduces a genuinely new paradigm for dataset distillation.

Compared to MGD³ (rejected at scores ~3-8), this paper has stronger empirical contributions and a clearer technical framework. Compared to DDiF (accepted poster at ~5.5), this paper has comparable empirical strength but weaker theoretical foundations. The overclaiming in the theoretical narrative is the primary concern, but the practical contribution is clear.

**Score: 6**

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>