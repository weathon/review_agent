Now I have enough calibration data. Let me synthesize the final review.

## Summary

This paper establishes geometric upper bounds on image watermarking capacity under PSNR and linear robustness constraints, finding that theoretical capacities far exceed what current methods achieve (e.g., >2 bpp at 40 dB vs. ~0.001 bpp in practice). Through controlled experiments, the authors show that Video Seal fails to approach even vastly simplified theoretical bounds on a single gray image, while linear and handcrafted models come far closer. As proof that higher capacity is achievable in practice, they introduce Chunky Seal, a scaled-up version of Video Seal that achieves 1024-bit capacity (4× increase) while maintaining comparable quality and robustness.

## Strengths

- **Novel and important research question.** The paper directly asks whether watermarking progress has stagnated due to fundamental limits or architectural limitations—a question of broad relevance to the community. The systematic elimination of hypotheses A–E provides a clear and well-motivated structure for the investigation.

- **Rigorous and elegant geometric analysis of PSNR-constrained capacity.** The lattice-point counting framework (Bounds 1–9) is mathematically sound, clearly presented, and provides computable, interpretable results for the PSNR-only regime. The derivation from absolute capacity through PSNR constraints to robustness constraints is systematic and well-organized.

- **Clever experimental diagnostics.** The single-gray-image setup (§3.1) is an elegant design choice that isolates architectural limitations from confounding factors. The finding that Video Seal at 256×256 achieves essentially the same capacity as at 32×32 (Table 1), while a linear model achieves 2048 bits and a handcrafted model achieves 456,509 bits, is striking and convincingly demonstrates that the architecture fails to utilize available spatial degrees of freedom.

- **Practical evidence that capacity can be improved.** Chunky Seal (Table 3) demonstrates that 4× capacity gains are achievable with comparable quality and robustness, ruling out the hypothesis that current methods are near-optimal. Even if achieved through scaling rather than architectural innovation, this is a useful proof-of-concept.

- **Useful proposed sanity checks.** The criteria proposed in §5 (capacity scaling linearly with image size, decreasing with PSNR, outperforming linear baselines, predictable drops under augmentations) provide actionable evaluation guidelines for the community.

## Weaknesses

### Major:

- **The central narrative overclaims how decisively the paper rules out alternative explanations for the theory-practice gap.** The paper frames the story as systematically eliminating hypotheses A–D to conclude E ("our models are significantly underperforming"). However: (i) Hypothesis A (advanced robustness) and B (perceptual constraints) are "dismissed" by removing those constraints and showing Video Seal still fails—but this only shows the model *also* struggles in simplified settings, not that robustness/perceptual constraints are irrelevant under realistic conditions. (ii) Hypothesis D (bounds overestimate) is addressed only for the PSNR-only, single-gray-image case; the robustness bounds remain heuristic and potentially loose. (iii) The data distribution argument (§2.6) uses a crude VQ-VAE codebook-size estimate that treats all combinatorial codebook entries as perceptually distinct, which is not conservative in the relevant direction. The evidence strongly supports E, but does not *decisively* eliminate A–D for realistic watermarking settings.

- **Robustness bounds (§2.5) are heuristic, acknowledged as such, yet used to support strong conclusions.** Bounds 10–12 are explicitly not guaranteed to be lower bounds and can both over- and under-approximate. Bound 13, the only formal lower bound, yields capacities (e.g., 904 bits for 75% crop at 256×256px, Table 2) that are comparable to or barely above current method capabilities (Chunky Seal achieves 1024 bits). This means the *rigorous* lower bound under aggressive but realistic augmentations does **not** show a large gap—yet the paper concludes that "robustness to geometric transformations and compression significantly reduces the capacity but cannot fully explain the low watermarking capacity of current models." Given the enormous gap between heuristic bounds (~100,000 bits) and conservative bounds (904 bits) for aggressive cropping, the true robust capacity could plausibly lie anywhere in that range, and the paper does not empirically validate which is closer to reality.

- **The gray-image experiments demonstrate architectural weaknesses but do not quantify the gap for the real watermarking problem.** The linear and handcrafted models achieving high capacity on a single gray image with no robustness requirements validate that the geometric bounds are correct as upper bounds—but this is essentially trivial (the image space has many lattice points). These experiments show that Video Seal fails to exploit spatial resolution in a minimal setting, which is a useful diagnostic. However, the paper repeatedly extrapolates from this to broader claims about "severe structural limitations" of current architectures and "a large potential for future development" in the abstract/intro, when the real question is how much capacity is achievable under *joint* quality, robustness, and distributional constraints—which the gray-image experiments do not address.

### Minor:

- **Chunky Seal's efficiency and the significance of the 4× capacity gain.** The embedder is 90× larger (1.023B vs. 11M params) and the extractor 23× larger (774M vs. 33M), for a 4× capacity increase. LPIPS degrades from 0.0019 to 0.0085 (4.5× worse). The paper acknowledges this is not a practical deployment suggestion—and the purpose of showing that capacity can be increased is valid—but the efficiency gap is significant enough that it somewhat tempers the "substantially higher capacities are within reach" narrative. No ablation separates the contribution of model scale from other changes (3-channel watermarking, gradient clipping, architectural width).

- **Chunky Seal is compared only to Video Seal.** While Video Seal is a natural baseline as the starting architecture, the paper's thesis is about the field-wide gap. Comparing Chunky Seal's quality/robustness against other SOTA methods (TrustMark, MBRS, etc.) would more convincingly demonstrate that higher capacity is achievable without sacrificing competitive performance.

- **No combined/composite attacks evaluation.** Robustness is reported per-individual augmentation, but realistic threats involve simultaneous distortions. This is standard practice in watermarking papers but limits the practical implications.

## Nice-to-Haves

- Empirical validation of which robustness bound (heuristic vs. conservative) is closer to reality, e.g., by testing achievable capacity under linear augmentations with the handcrafted model
- Testing at least one additional watermarking architecture (e.g., HiDDeN, MBRS) in the gray-image setup to confirm the architectural limitation is general rather than Video Seal-specific
- Ablation separating model scale from design changes in Chunky Seal
- Analysis of whether the LPIPS increase in Chunky Seal is perceptually noticeable (e.g., user study or visual examples)

## Removed Points

- *"No evaluation under adaptive/removal attacks or diffusion-model-based regeneration attacks."* This is outside the paper's stated scope. The paper is about capacity bounds and proving higher capacity is possible; it does not claim robustness against all possible attacks beyond the standard augmentation suite used by Video Seal.

- *"Missing citations to related work."* Per the instructions, I do not flag missing related work as I cannot verify its existence.

- *"Dataset/distribution argument uses VQ-VAE codebook size as an upper bound, which overestimates N."* While the VQ-VAE argument is crude, the paper's conclusion in §2.6 is also supported by information-theoretic results (Costa 1983; Chen & Wornell 2002; Moulin & O'Sullivan 2003) showing decoder knowledge of the cover does not affect capacity in Gaussian channels. The VQ-VAE argument is a supplementary sanity check, not the sole basis. The core concern about the looseness of the VQ estimate is retained in Major weakness 1 above.

- *"The narrative about 'watermarking might be a solved problem' sets up an attractive strawman."* The paper uses this framing precisely to motivate why understanding capacity limits matters. Whether or not specific people have explicitly claimed this, the observation that progress has plateaued at 100-200 bits is factual and motivates the question. This is not a strawman in the argumentative sense.

- *"The word 'capacity' is used in a purely combinatorial sense, not in the operational sense of information theory."* The paper is clear about what it means by capacity—the number of distinct watermarked images satisfying constraints—and the geometric bounds are correct under that definition. Whether this matches Shannon capacity is a different question, but the paper does not claim it does; it uses the term within its own clearly defined framework.

- *"No proof that the embedder and decoder can be realized as learnable functions."* The handcrafted and linear models in §3.2 demonstrate that realizable (if trivial) encoders/decoders can approach the PSNR-only bounds. The paper then acknowledges a learnability gap for practical systems, which is a core point, not an omission.

## Novel Insights

The most novel insight is the finding that Video Seal's capacity at 256×256 is nearly identical to its capacity at 32×32 (for the same number of bits), implying the architecture fails to exploit spatial resolution at all—operating as if on a ~20×20 image regardless of input size. This is a striking empirical finding that points to a concrete architectural deficiency: current convolutional watermarking architectures are not designed to scale capacity with image resolution. This insight, combined with the linear and handcrafted models' success on the simplified task, strongly suggests that architectural redesign (rather than just scaling) is needed, and the proposed sanity checks (capacity scaling linearly with image size) provide a concrete evaluation metric for future work.

## Suggestions

- **Temper the conclusions about ruling out hypotheses A–D for realistic settings.** The gray-image experiments eliminate these hypotheses only for the simplified setting. State clearly that the gap between simplified and realistic settings remains unquantified, and that robustness and perceptual constraints may explain a substantial portion of the practical gap.
- **Add empirical validation of the robustness bounds.** Test the achievable capacity of the handcrafted or linear model under the linear augmentations used in §2.5 (horizontal flip, rotation, LinJPEG) to provide ground truth for where the actual capacity lies relative to heuristic and conservative bounds.
- **Report LPIPS increase context.** The 4.5× increase in LPIPS may or may not be perceptually significant given the low absolute values; provide visual side-by-side comparisons or reference LPIPS thresholds to contextualize this.

## Score and Decision

**Calibration anchors:**
- *Fantastic Generalization Measures* (NkmJotL42): Scores 8/6/8/6, avg 7.0, Accept. Strong theoretical contribution challenging common beliefs, with rigorous proofs. This paper is less rigorous in its theory (heuristic bounds for robustness) but has stronger empirical diagnostics.
- *Disconnect Between Theory and Practice of Overparametrized NNs* (GqI4fTVUXC): Scores 5/5/8/6, avg 6.0, Reject. Questions whether theory matches practice but reviewers found the claims didn't follow from experiments. This paper is more empirically grounded but shares the concern that the main narrative overclaims.
- *On the Coexistence and Ensembling of Watermarks* (ldGz1DSut1): Scores 6/6/8/6, avg 6.5, Reject. Empirical watermarking study with interesting findings but limited novelty. This paper has more novelty (geometric capacity bounds) and a clearer contribution.
- *Undetectable Watermark* (jlhBFm7T2J): Scores 6/6/6/8, avg 6.5, Accept Poster. Novel theoretical framework for watermarking with practical implementation.
- *Universally Optimal Watermarking for LLMs* (NQZImD0VGP): Scores 6/3/3/6, avg 4.5, Reject. Theory-practice mismatch that undermined claims.

This paper makes an original and important contribution (geometric capacity bounds for watermarking + the striking 32×32 vs 256×256 diagnostic experiment) but overclaims how decisively it identifies the bottleneck. The robustness bounds are heuristic and the data distribution argument is loose, yet these are used to make strong conclusions. The core finding—that there is a large gap and it appears to be at least partly architectural—is valid and well-supported by the simplified experiments. The Chunky Seal proof-of-concept is useful but limited. Overall, this is a solid contribution that would benefit from more tempered claims rather than fundamentally flawed work.

Score: 5.5 — borderline, leaning reject due to the overclaiming in the narrative relative to what the evidence establishes, but the contribution is real and interesting.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>