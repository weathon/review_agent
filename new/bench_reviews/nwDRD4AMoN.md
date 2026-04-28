## Summary

This paper introduces Artificial Kuramoto Oscillatory Neurons (AKOrN), a dynamical neuronal unit based on the Kuramoto model that replaces standard threshold units. The authors demonstrate that AKOrN achieves competitive performance across three distinct domains: unsupervised object discovery (outperforming DINO/MoCoV3 on PascalVOC), combinatorial reasoning (solving Sudoku puzzles with test-time scaling), and adversarial robustness (58.9% accuracy under AutoAttack without adversarial training). The work represents a novel exploration of oscillatory dynamics as a fundamental building block for neural networks.

## Strengths

- **Novel architectural integration with clear mathematical formulation:** The paper successfully integrates continuous-time Kuramoto dynamics into standard deep learning blocks (convolution and attention) with well-defined equations (Eq. 2, 4, 5, 7, 8). The vector-valued generalization with unit-norm constraints and symmetry-breaking terms is clearly articulated.

- **Strong empirical validation across diverse tasks:** Table 2 shows AKOrN achieves 52.0 MBO_i on PascalVOC, outperforming DINO (47.2), MoCoV3 (47.3), and MAE (34.0) when all are ImageNet-pretrained. Table 1 demonstrates competitive performance on CLEVRTex (88.5 FG-ARI) against specialized slot-based methods like BO-SA (80.5).

- **Test-time computation benefits for reasoning:** Figure 6(c) provides concrete evidence that increasing Kuramoto steps at test time improves OOD Sudoku accuracy from 17% to 52%, while standard self-attention only improves from 14% to 34% and degrades on in-distribution data. This demonstrates a unique capability of the dynamical approach.

- **Exceptional calibration and robustness without adversarial training:** Table 4 shows AKOrN achieves 58.91% accuracy under AutoAttack with EoT and an ECE of 1.3, compared to 0.00% and ECE of 8.9 for ResNet-18. The confidence-accuracy correlation in Figure 9 is notably tight.

## Weaknesses

### Fatal
None

### Major

- **Ambiguous evaluation protocol for object discovery on natural images:** The "up-tiling" upsampling method is introduced in Section 6.1 (line 201) to refine cluster assignments, but the paper does not clearly state whether this preprocessing was applied to the SSL baselines (DINO, MoCoV3, MAE) in Table 2. The text states "we introduce" implying novelty, and says clustering is applied to "final block's output features" for evaluation. If up-tiling was only applied to AKOrN and not the baselines, the resolution advantage belongs partially to the post-processing rather than the Kuramoto backbone alone. This ambiguity makes it difficult to isolate the architectural contribution from the evaluation preprocessing. The comparison would be more convincing with explicit clarification or re-evaluation of baselines with the same up-tiling protocol.

- **Extraordinary robustness claims require stronger validation:** The claim that AKOrN is "robust by design" achieving 58.9% adversarial accuracy without adversarial training is remarkable given that standard models achieve ~0%. While the paper uses AutoAttack with EoT (stronger than FGSM-only evaluations that typically indicate gradient masking), the adversarial robustness literature has shown that iterative, normalization-heavy defenses can still exhibit obfuscated gradients. The paper would benefit from additional validation such as: (1) gradient flow visualization through Kuramoto layers during attack generation to verify gradients don't vanish, (2) results against adaptive attacks specifically designed for projection-based defenses, or (3) analysis of whether robustness persists under stronger EoT samples. Without this, the claim that Kuramoto dynamics provide inherent security remains partially unsubstantiated.

### Minor

- **Theoretical framing vs. practical design tension:** Section 3 motivates the approach via the Kuramoto model's energy minimization properties (Eq. 3), which theoretically require symmetric connectivity for Lyapunov stability. The paper explicitly acknowledges (lines 73-74) that asymmetric weights perform better and are used throughout, noting "even without symmetric constraints, the energy value decreases relatively stably." While this transparency is commendable, it means the "energy-based reasoning" claims in Section 6.2 rest on empirical correlation rather than theoretical guarantees. The paper would benefit from quantifying the performance drop when enforcing symmetry and more explicitly framing the energy function as an empirical heuristic in the asymmetric setting rather than a grounded physical principle.

- **Computational cost and efficiency not quantified:** The introduction states neurons are "designed to work well on modern hardware" (line 19), but Equation 7 shows J weights are R^{N×N} matrices per kernel position, suggesting significant parameter and compute overhead compared to standard convolutions. The paper does not report FLOPs, parameter counts, or inference latency compared to ResNet/ViT baselines. Given that test-time reasoning requires multiple Kuramoto steps (16 during training, up to 128 at test time in Figure 6), and energy-based voting samples 4096 initializations (Figure 7), the computational trade-offs deserve explicit discussion.

### Trivial

- **Notation inconsistency in rotating dimension N:** Section 4.1 mentions J_{c,d,h',w'} ∈ R^{N×N} but the specific value of N ("rotating dimensions") used in experiments is not specified in the main text. This is critical for understanding model capacity and reproducibility.

## Nice-to-Haves

- **Ablation on symmetry constraints:** Quantify the performance drop when enforcing symmetric J in Sudoku and robustness tasks. If the drop is negligible, retaining theoretical guarantees would strengthen the physics-grounded narrative; if large, this would better justify the asymmetric design choice.

- **Analysis of rotating dimension N scaling:** Show how performance scales with N. If N=1 works nearly as well, the vector-valued generalization may be unnecessary; if large N is required, the cost-benefit analysis would help readers understand the design space.

- **Downstream task fine-tuning evaluation:** Evaluate AKOrN on supervised classification or detection with fine-tuning to assess whether the robust/object-centric features transfer to standard supervised benchmarks beyond the self-supervised settings tested.

## Removed Points

These points are flagged to be removed, treat them with caution:

1. **Harsh Critic Point 1 (Unfair Comparison - Slot-Attention pretraining):** The claim that Slot-Attention baselines may not share the same ImageNet pretraining is partially addressed by the paper's experimental design. Table 2 clearly separates "(slot-based models)" trained from scratch (Slot-attention: 22.2) from "(transformer + SSL)" models that are ImageNet-pretrained (DINO: 47.2, MoCoV3: 47.3). AKOrN (52.0) is compared fairly against the SSL baselines in the same category. The comparison is not conflating pretraining regimes—the paper explicitly notes AKOrN is "trained on ImageNet and directly evaluate... without fine-tuning" and includes other ImageNet-pretrained baselines. **Removed as the paper's experimental design already addresses this concern.**

2. **Harsh Critic Point 2 (Theoretical Contradiction):** The paper explicitly acknowledges in Section 3 (lines 73-74) that asymmetric weights perform better despite losing theoretical guarantees: "We would like to note that we found that even without symmetric constraints, the energy value decreases relatively stably, and the models perform better across all tasks we tested compared to models with symmetric J." This is not a hidden contradiction but a transparent design choice. The claim is weakened to a Minor weakness about framing rather than a structural flaw. **Downgraded as the authors are transparent about this trade-off.**

3. **Strength Finder Claim about "first work demonstrating synchrony-based model scaled to natural images":** While the paper does show strong results, related work (Section 5) mentions Löwe et al. (2023) showed their model can work with pretrained SSL models as feature extractors. The claim of being "first" should be qualified. **Removed as potentially overclaimed.**

4. **Generic strengths about "addressing important problem" or "interesting question":** Removed per instructions as these are superficial and not grounded in specific evidence.

## Novel Insights

The paper's most compelling insight is that oscillatory synchronization dynamics—previously confined to synthetic datasets in synchrony-based models—can scale to natural images when properly integrated with modern architectures. The observation that energy values correlate with solution correctness in the asymmetric setting (despite lacking theoretical guarantees) suggests Kuramoto dynamics may implicitly learn useful inductive biases beyond their physical motivation. The test-time scaling behavior, where extending Kuramoto steps improves OOD reasoning performance while degrading standard attention, reveals a qualitatively different computation dynamic that warrants further investigation. However, these insights are largely presented in the paper itself rather than emerging uniquely from the review synthesis.

## Suggestions

1. **Clarify the up-tiling evaluation protocol:** Explicitly state whether up-tiling was applied to all baselines in Table 2 or only AKOrN. If only AKOrN, re-evaluate key baselines (DINO, MoCoV3) with up-tiling to isolate the architectural contribution from the preprocessing advantage.

2. **Strengthen robustness validation:** Add gradient flow analysis through Kuramoto layers during AutoAttack to rule out gradient masking. Consider evaluating against adaptive attacks designed for projection-based defenses or report results with increased EoT samples.

3. **Report computational metrics:** Include FLOPs, parameter counts, and inference latency compared to ResNet/ViT baselines. Discuss the compute-accuracy trade-off for test-time scaling (more Kuramoto steps) and energy-based voting (multiple samples).

4. **Specify rotating dimension N:** State the value of N used in each experiment in the main text or clearly reference the appendix location.

5. **Temper theoretical claims:** More explicitly frame the energy function as an empirical heuristic in the asymmetric setting, and consider adding an ablation quantifying the symmetry vs. performance trade-off.

## Calibration and Score

**Calibration Papers Retrieved:**

**High-scoring anchors (≥6):**
- `/home/wg25r/review_agent/human_reviews_2026/DTQIjngDta.md` (Score 8.0): π³ achieves SOTA on multiple visual geometry tasks with novel permutation-equivariant architecture. Like AKOrN, it has strong empirical validation across tasks, but π³ has clearer novelty positioning and no ambiguous evaluation protocols.
- `/home/wg25r/review_agent/human_reviews_2026/yirunib8l8.md` (Score 7.0): Depth Anything 3 achieves SOTA across geometry tasks with minimal architectural changes. Similar to AKOrN in empirical strength, but has more thorough ablation studies.
- `/home/wg25r/review_agent/human_reviews_2026/8fViWZ0yZJ.md` (Score 7.33): Discovers oscillatory dynamics in RNNs. Directly relevant topic, scored highly for novel findings about alternative solutions beyond simplicity bias.

**Medium-scoring anchors (4-6):**
- `/home/wg25r/review_agent/human_reviews_2026/GjQ5JXpRQF.md` (Score 5.5): OrthoRF for synchrony-based object discovery. Similar topic (oscillatory/synchrony models for object-centric learning), scored medium due to concerns about benchmark scope and decoder comparability—analogous to AKOrN's up-tiling ambiguity.
- `/home/wg25r/review_agent/human_reviews_2026/kYQFfEKtx5.md` (Score 5.5): C-Voting for reasoning tasks, explicitly evaluates on Sudoku. Shows that reasoning papers with strong results but questions about contribution disentanglement score in the 5-6 range.
- `/home/wg25r/review_agent/human_reviews_2026/sPh4zaxDUU.md` (Score 4.67): Kuromi (AKOrN-based SSL). Lower score due to unclear explanations and bold claims not fully supported—cautionary anchor for AKOrN-style papers.

**Low-scoring anchors (≤4):**
- `/home/wg25r/review_agent/human_reviews_2026/dBJpBmn5MH.md` (Score 1.0): Claims robustness without adversarial training but evaluated only on FGSM. Much weaker evaluation than AKOrN's AutoAttack+EoT.
- `/home/wg25r/review_agent/human_reviews_2026/s4jpvJv6I8.md` (Score 3.0): Exposes false robustness claims that vanish under stronger attacks. Shows the community's skepticism toward "robust by design" claims.
- `/home/wg25r/review_agent/human_reviews_2026/klzjZ3gePy.md` (Score 4.0): Object discovery on large-scale benchmarks rejected for evaluation concerns.

**Score Reasoning:**

AKOrN sits between the high and medium anchors. Compared to π³ (8.0) and Depth Anything 3 (7.0), AKOrN has similarly strong empirical results across multiple tasks but has more ambiguity in evaluation protocols (up-tiling) and less thorough computational analysis. Compared to OrthoRF (5.5) and C-Voting (5.5), AKOrN has stronger results (beating DINO on PascalVOC is more impressive than OrthoRF's synthetic benchmarks) and uses stronger robustness evaluation (AutoAttack+EoT vs. weaker attacks in low-scoring robustness papers).

The robustness claim is the most contentious aspect. Papers claiming robustness without adversarial training typically score 1.0-4.0 when evaluation is weak (FGSM only), but AKOrN uses AutoAttack with EoT, which is the current standard for rigorous evaluation. This distinguishes it from the score 1.0 anchor. However, the extraordinary nature of the claim still warrants caution, preventing a score in the 7+ range.

The object discovery results are solid and comparable to medium-scoring object-centric papers. The reasoning results are strong but the computational cost of test-time scaling (4096 samples for energy-based voting) is not discussed, similar to concerns in C-Voting (5.5).

AKOrN is stronger than the medium anchors (5.5) due to: (1) beating established SSL models on natural images, not just synthetic datasets; (2) using AutoAttack+EoT rather than weaker attacks; (3) demonstrating test-time scaling benefits. However, it falls short of high anchors (7+) due to: (1) ambiguous up-tiling protocol; (2) missing computational metrics; (3) extraordinary robustness claims needing additional validation.

**Final Score: 6.5**

This positions AKOrN above the medium anchors (5.5) for its stronger empirical results and more rigorous robustness evaluation, but below the high anchors (7+) due to unresolved evaluation ambiguities and missing computational analysis. The score reflects a paper with genuine contributions and strong results that would benefit from clarification and additional validation before acceptance.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>