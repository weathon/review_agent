Now I have a good understanding of the paper, the reviewers' critiques, and calibration papers. Let me synthesize the final review.

Key points to address:
1. The "inverse problem" argument (Harsh Critic #1) - is this valid? The Harsh Critic argues it's not compelling because MPNNs don't have to adopt that specific internal representation. Let me verify against the paper text.

The paper's argument in Section 2.3 is that when a static MPNN processes layer 2, the first MPNN layer update creates v²(1) = b² + W²b¹, and then the second MPNN layer needs to extract b² from this to properly simulate the forward pass. The Harsh Critic says this isn't convincing because MPNNs don't need to adopt this canonical representation - they just need to approximate the function. This is a valid concern: the "inverse problem" argument presupposes a specific canonical form rather than proving it's unavoidable. However, the paper isn't entirely wrong - there is a structural issue with information flow in static graphs. The paper's observation that static graphs mix information across layers in ways that don't match the sequential nature of computation is intuitively valid, even if the specific "inverse problem" formulation is arguable.

2. Causal attribution of gains (Harsh Critic #2) - This is a very strong point. The ablation in Table 5 shows DNG-Encoder alone gets only 54% on CIFAR-10 vs 73.2% for the full INR2JLS. The improvements largely come from the joint reconstruction + latent generator + augmentation, not the dynamic graph representation per se.

3. Joint latent space claim (Harsh Critic #3) - Valid but somewhat overblown. The paper does show INR2JLS vs INR-INR comparison, but the baselines differ in more than one way.

4. Baseline fairness (Harsh Critic #4) - Valid concerns about the augmentation confound, but the gap is large enough that it likely remains meaningful even with matching.

5. Generalization prediction (Harsh Critic #5) - Valid that the gains are marginal. 0.936 vs 0.935 is negligible.

Let me now write the final review.

For the "Removed Points" section, I need to identify things from the reviews that should be removed:
- The Harsh Critic's point about models/tools not being released should be removed per the hard rules
- Formatting/style nitpicks should be removed
- Claims about missing related work should be removed per rules

For calibration:
- Graph Metanetworks (similar field, weight space processing): Accept (spotlight), scores 6/6/6
- ProbeGen (weight space learning): Accept (poster), scores 6/6/6/6
- INRet (INR retrieval): Reject, scores 6/6/5/6
- DDMI (related area): Accept (poster), scores 6/6/6/6

This paper has significant empirical gains but the core architectural claim (dynamic graphs are key) is not well-isolated. The INR classification gains are impressive but confounded. The paper is roughly at the level of Graph Metanetworks (6/6/6, accept spotlight) but with weaker causal attribution of claims. However, the empirical results are much stronger. It's better than INRet (rejected, 6/6/5/6).

I'd say this paper is in the 5-6 range. The contributions are real but the causal attribution of gains to dynamic graphs is weak, and the "inverse problem" motivation is not fully convincing. The empirical results are strong but confounded.

Let me settle on a score of 5.5 - borderline, leaning toward weak accept.</think>## Summary

The paper introduces dynamic neural graphs for representing neural network parameters, arguing that static neural graph representations (as in prior work) suffer from an "inverse problem" during multi-layer MPNN processing that dynamic graphs avoid. A Dynamic Neural Graph Encoder (DNG-Encoder) processes these dynamic graphs via an RNN-based architecture that mirrors the sequential forward pass. The authors further propose INR2JLS, which maps INR weights into a joint latent space with the original data (via image reconstruction), and demonstrate strong improvements on INR classification, INR editing, and CNN generalization prediction tasks.

## Strengths

- **Novel and elegant graph construction**: The dynamic bipartite-snapshot representation (maintaining only two adjacent layers at each timestamp) is a conceptually clean way to mirror forward propagation in graph form while preserving permutation symmetry. This is a genuine contribution to the weight-space learning literature.

- **Strong empirical performance on INR classification**: The improvements on CIFAR-10-INR (+9%) and CIFAR-100-INR (+10%) over prior SOTA are substantial and consistent (Table 1). These are meaningful gains on established benchmarks.

- **Well-motivated design choice of image reconstruction over weight reconstruction**: The INR2JLS framework decodes to image space rather than weight space, avoiding the difficult problem of regenerating INR weights. Table 5 (top) validates this with a clean ablation (INR-INR vs INR2JLS), showing 17% gains on CIFAR-10.

- **Comprehensive ablations and efficiency analysis**: Tables 4, 5, and 6 provide useful breakdowns of the contributions of augmentation, reconstruction, and latent generator, as well as running time/memory/FLOPs. The efficiency advantage is real: INR2JLS achieves the lowest GFLOPs (1.31) and fastest inference time (0.0047s).

- **Multi-task evaluation**: Testing on three different tasks (classification, editing, generalization prediction) gives confidence that the approach is not task-specific.

## Weaknesses

### Fatal

None.

### Major

- **Core architectural claim (dynamic graphs are key) is not causally supported by the experiments.** The headline framing attributes improvements to dynamic neural graphs, but DNG-Encoder differs from baselines along multiple axes simultaneously (dynamic graph structure, GRU-based recurrent memory, FiLM-style message passing, no edge updates). Moreover, INR2JLS adds further large components: joint image reconstruction, latent generator with positional encodings, and strong rotation/flip augmentation. The ablations in Table 5 are revealing: DNG-Encoder *alone* achieves only 54.0% on CIFAR-10 and 25.7% on CIFAR-100, compared to 57.7%/31.65% for NG-T and 55.11%/26.50% for NG-GNN — barely competitive. The dramatic improvement to 73.2%/42.4% comes from adding the Latent Generator and Decoder (i.e., the INR2JLS pipeline), not from the dynamic graph representation. No experiment compares a static-graph encoder equipped with the same INR2JLS pipeline against DNG-Encoder with INR2JLS, which is the critical ablation for isolating the dynamic graph contribution. Without this, the paper shows that *a significantly more elaborate pipeline* outperforms prior methods, not that dynamic graphs are the key factor.

- **The "inverse problem" motivation (Section 2.3) is unconvincing as a theoretical argument.** The paper argues that static MPNNs are forced to accumulate undesirable terms like $W^2 b^1$ into second-layer node representations, creating an "inverse problem" that subsequent layers must solve. However, MPNNs are universal function approximators over local neighborhoods — there is no requirement that they adopt this particular canonical internal representation. The model can learn any embedding that allows the overall computation to succeed; it does not need to explicitly invert $b^2 + W^2 b^1$ to recover $b^2$. Calling this "ill-posed" misapplies the concept: the model is learning a function, not solving for parameters from outputs. This is not merely a presentation issue; it is the *primary motivation* for the entire paper (dynamic graphs), and it is not formally established. No empirical diagnostics are provided to show that static GNN encoders actually exhibit the claimed pathology in practice (e.g., unstable training, poor convergence at depth).

- **INR editing task uses a different and easier task formulation than baselines.** The paper acknowledges that "the typical method of modifying W by adding a learned offset Δ(W) is not directly applicable in our framework" (Section 6.2) and instead directly generates transformed images. This means DNG-Encoder bypasses the hard part (producing valid modified INR weights) and solves a simpler problem (generating images from weight encodings). The comparison in Table 2 is thus not apples-to-apples: the baselines must learn weight-space modifications while the proposed method outputs images directly. This should be clearly acknowledged as a different task rather than presented as a direct improvement.

### Minor

- **CNN generalization prediction results are marginal and partially negative.** On CIFAR-10-GS, DNG-Encoder achieves $\tau=0.936$ vs NG-T's $0.935$ — a negligible difference within measurement noise ($\pm 0.001$). On SVHN-GS, DNG-Encoder clearly underperforms NFN(HNP) ($0.867$ vs $0.931$). These results do not strongly support the method's generality beyond INR settings.

- **Scalability to modern architectures (ResNets, transformers, normalization layers) is unaddressed.** All experiments involve simple MLPs (INRs) or small CNNs without residual connections or normalization. The dynamic graph construction (Eq. 3) handles skip connections only in an appendix (G.1), and transformers/attention are not discussed. Given that the paper motivates the approach partly by noting that prior methods "can only handle one fixed architecture" (Introduction), testing on heterogeneous or complex architectures would strengthen the generality claims.

- **The "joint latent space" claim is under-specified.** The paper repeatedly describes INR2JLS as learning a "joint latent space between deep weights and the original data," but the implementation is essentially an autoencoder mapping weight-space inputs to image-space outputs. No analysis demonstrates that the representation actually captures joint information (e.g., invariance across different INRs encoding the same image, or interpolation quality in the latent space), beyond downstream classification accuracy.

### Trivial

- In Eq. 3, the notation $t^l$ for timestamps could be confused with superscripts; the paper uses both $t^l$ (time) and layer index $l$ in similar typographic styles, which can be momentarily confusing.

## Nice-to-Haves

- A controlled ablation comparing DNG-Encoder on static neural graphs (processing all layers simultaneously) vs. dynamic neural graphs, keeping the INR2JLS pipeline fixed, would decisively isolate the contribution of dynamic graphs.
- Visualization of reconstructed images from the joint latent space and t-SNE plots of learned representations would help validate the claim that the encoder learns meaningful joint representations.
- Testing on deeper INRs or more complex architectures (ResNets with skip connections, transformers) to verify scalability.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's point about baseline fairness (incomplete specification of augmentation matching)**: While the concern about whether baselines received the same augmentation protocol is worth investigating, the paper does state that "10 views of input INRs as data augmentation" is used across methods (Table 1 caption), and the rotation/flip augmentation in Table 4 applies to *latent-space augmentation* in INR2JLS specifically — a capability baselines don't have an analog for. The large performance gap (9-10%) also makes it unlikely that augmentation alone explains the difference. This concern is partially valid but not as damning as presented.

- **Neutral Reviewer's point about lack of formal expressivity guarantees**: The paper provides an informal argument (Section 2.3) rather than a formal theorem. While formal bounds would strengthen the paper, it is not standard in this field to require such proofs for empirical contributions. The informal analysis provides useful intuition even if incomplete. This is a nice-to-have rather than a weakness.

- **Harsh Critic's claim that the paper's contribution is "more about task design than representation learning"**: This is partially true (the INR2JLS image reconstruction is a key ingredient), but the DNG-Encoder *is* a novel architectural contribution regardless of whether it's the main driver of empirical gains. The conflation concern is already captured in the "causal attribution" major weakness above.

- **Human Finder suggestion about permutation equivariance verification**: The paper states in Section 3.1 that "permuting the nodes of the neural graph adjusts the adjacency matrix in a way that connections between same neurons remain the same" and that "our graph operations does not change the original connection between neurons." While formal verification would be nice, the claim is mechanically straightforward given the construction.

## Novel Insights

The paper's most insightful observation is that the dominant performance gains come not from the dynamic graph encoder itself, but from the INR2JLS pipeline — specifically, the combination of image reconstruction (avoiding difficult weight-space generation) and latent augmentation (rotation/flip in the joint space). This suggests that the key bottleneck in prior INR classification work may be the *training objective and representation space* rather than the *graph structure or encoding architecture*. This is somewhat at odds with the paper's own framing, which centers dynamic graphs as the primary contribution, but represents a valuable finding for the field.

## Suggestions

- **Add a static-graph + INR2JLS ablation**: Implement the INR2JLS framework with a standard static GNN encoder (e.g., the same FiLM message function and GRU update, but operating on the full static graph) to isolate the effect of the dynamic graph formulation from the training pipeline. This is the single most impactful experiment the authors could add.
- **Tone down claims about dynamic graphs being the primary contribution** and instead accurately reflect that INR2JLS's joint reconstruction framework is a major driver of the observed gains, based on the ablation evidence in Table 5.
- **Clearly acknowledge the apples-to-oranges nature of the INR editing comparison** and discuss the tradeoffs between weight-space editing (which preserves the INR structure) and image-space generation (which is easier but loses the INR representation).

## Evaluation Assessment

**Originality**: The dynamic neural graph formulation is a novel contribution that is well-motivated conceptually, even if the "inverse problem" argument is not fully rigorous. The INR2JLS image-reconstruction-based training objective is a practical and effective design idea. The FiLM-style message function with multi-head adaptation for CNNs is sensible but incremental. Overall, moderate originality.

**Importance**: The paper addresses an active and important problem (processing neural network weights). The empirical gains on INR classification are substantial. However, the importance of the *dynamic graph* contribution specifically is undermined by the ablation evidence.

**Claims support**: The primary claim that dynamic neural graphs are superior is not well-supported by the evidence, since the gains are confounded with other pipeline components. The claim that INR2JLS achieves SOTA on INR classification is well-supported.

**Soundness of experiments**: The experiments cover multiple tasks and include ablations, which is valuable. However, the critical ablation (static vs. dynamic with same pipeline) is missing, and the INR editing comparison is not apples-to-apples.

**Clarity**: The paper is generally well-written and well-organized. The dynamic graph construction (Eq. 3, Figure 2) is clearly presented. Some notation could be cleaner.

**Value to community**: The INR2JLS framework and the idea of image-space reconstruction from weight encodings are likely to be useful contributions. The dynamic graph framing may inspire future work, though its current empirical impact is unclear.

**Score calibration**: Compared to Graph Metanetworks (ICML 2024, accept spotlight, scores 6/6/6) — which had a novel graph construction, theoretical proofs of equivariance, multi-architecture support, but limited experiments and no ablations — this paper has stronger empirical results but weaker theoretical grounding and weaker causal attribution. Compared to ProbeGen (accept poster, scores 6/6/6/6) and DDMI (accept poster, scores 6/6/6/6) — which had solid but incremental contributions — this paper has comparable contributions. The paper is somewhat stronger than INRet (reject, scores 6/6/5/6) in empirical results but shares similar concerns about the novelty of the core mechanism.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>