# Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Pre-Layer Normalization (Pre-LN) is the de facto choice for large language models (LLMs) and is crucial for stable pretraining and effective transfer learning. However, LN and Pre-LN are inefficient due to repeated statistical calculations and suffer from the curse of depth. As layers grow, the magnitude and variance of the hidden state escalate, destabilizing training. Efficiency-oriented, normalization-free methods such as Dynamic Tanh (DyT) improve speed but remain fragile at depth. To jointly address stability and efficiency, we propose Bounded Hyperbolic Tanh (BHyT), a drop-in replacement for Pre-LN. BHyT couples a tanh nonlinearity with explicit, data-driven input bounding to keep activations within a non-saturating range. It prevents depth-wise growth in activation magnitude and variance and comes with a theoretical stability guarantee. For efficiency, BHyT computes exact statistics once per block and replaces a second normalization with a lightweight variance approximation, enhancing efficiency. Empirically, BHyT achieves improved stability and efficiency in pretraining, delivering on average 7.7\% faster forward computation and up to 5\% higher token generation throughput than RMSNorm, while matching or surpassing its inference performance and robustness across language understanding and reasoning benchmarks. Our code is available at: \url{https://anonymous.4open.science/r/BHyT}

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Peri‑LN was proposed to mitigate layer‑wise gradient and variance growth in Transformers but it introduces notable computational overhead. This paper takes an alternative path: it replaces the normalizer with a tanh‑based formulation that explicitly constrains inputs to a preset range, which keeps activations away from saturation and curbs depth‑wise drift. The authors provide theoretical guarantees for stability and show that their formulation does not require the extra normalizing stages used by Peri‑LN while retaining compatibility with standard Transformer blocks.

### Strengths
* Conceptually clean formulation that directly bounds activations and targets the underlying cause of depth‑related instability.
* Solid theory that upper‑bounds gradient amplification and supports the intended stability properties.
* Empirical results broadly align with the theory and indicate improved stability of layer statistics across depth.

### Weaknesses
* Throughput is reported as higher than Peri‑LN, yet training loss at a fixed number of steps is slightly worse. It remains unclear whether the proposed method would surpass Peri‑LN under equal training compute. Time‑to‑target‑loss or equal‑FLOPs comparisons would make the efficiency claim more convincing.
* The paper should analyze where Peri‑LN’s inefficiency comes from. Is the bottleneck inherent to the algorithmic cost of repeated reductions and memory movement, or chiefly an implementation artifact such as unfused CUDA kernels or suboptimal scheduling? Kernel‑level profiling or theory‑backed complexity analysis would clarify this point.
* The final downstream comparison emphasizes overall averages, but absolute MMLU scores are low. Since a random baseline is 25% on four‑choice tasks, small differences at this scale may be hard to interpret for models trained with limited tokens.
* With the current experimental setup it is difficult to conclude that BHyT is definitively more effective than Peri‑LN. That said, the gains over the RMSNorm baseline are sizable, which suggests BHyT is a strong and practical alternative to Peri‑LN.

### Questions
* Not a question but just a minor editing issue: `\citep` should be used in places where citation on points is intended. (e.g., line 094, 095, ...)

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Bounded Hyperbolic Tanh (BHyT) as a drop‑in replacement for Pre‑Layer Normalization (typically RMSNorm) in Transformer blocks. The key idea is to pair a tanh nonlinearity with explicit, per‑instance input bounding so activations stay in a non‑saturating regime while avoiding repeated statistic computations.  Empirically, BHyT is evaluated by pretraining 1B/3B Llama‑style models from scratch on C4, then (optionally) SFT on LIMA‑1k and Commonsense‑170K, with downstream evaluation.

### Strengths
1 Creative combination of normalization‑free activations with explicit, data‑aware input bounding and a block‑level variance approximation; the latter gives a neat knob to keep overhead small while preserving stability

2 Clear derivations, concise definitions, readable pseudocode,

3 If the approach scales, it could reduce normalization overhead in large LMs without sacrificing training stability

### Weaknesses
Please see the questions.

### Questions
1 Since the paper introduces  BHyT∗ first, could you please add an ablation directly comparing BHyT∗ and the practical BHyT under the same setup, reporting training loss/validation perplexity, downstream accuracy, and throughput.

2  If the variance can be estimated cheaply, could you apply this estimator to RMSNorm (e.g., replace one normalization with the estimate) and report whether similar speedups and performance are achieved?

3 Please include validation perplexity results of pretraining.

4  Please specify hardware (GPU/TPU model & count), precision, parallelism strategy (DP/TP/PP/ZeRO), global batch size/sequence length, and the measurement protocol.

5  Since LNS is a strong baseline here and LNS reports BoolQ, could you add BoolQ results for parity with the tasks in Table 1?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes the BHyT which replaces Pre-LN with "input bounded with probability guarantee + tanh" and only precisely calculates the variance once per Transformer block, with the rest using lightweight approximation.

### Strengths
- Combine the speed advantage of "unnormalized" with the stability of "bounded non-saturated" to explicitly avoid tanh saturation with probability limiting.
- The method is simple and effective.

### Weaknesses
- Only validated on 1B/3B, C4, and a small amount of SFT datasets; did not cover deeper layers or longer contexts. I understand that it is not realistic to do such a thing with limited resources, but perhaps training a narrow and deep model might be feasible?
- The "uniform" attention assumption is least tenable in which training stages/tasks? If the second moment of the real attention weights replaces the uniform assumption, what are the approximate errors and throughput losses?

### Questions
see weekness above

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Bounded Hyperbolic Tanh (BHyT), a normalization-free alternative to Pre-Layer Normalization designed to improve both stability and efficiency in large language model training. It combines a tanh nonlinearity with an explicit, data-driven input bounding mechanism that constrains activations within a stable, non-saturating range, preventing variance explosion across depth.
BHyT theoretically guarantees gradient stability by bounding the Jacobian norm relative to RMSNorm and uses a lightweight variance approximation to reduce normalization overhead. Empirically, it aims to maintain training stability comparable to heavily normalized methods like Peri-LN while achieving higher computational efficiency. Overall, the paper positions BHyT as a practical replacement for Pre-LN, balancing stability and throughput for large-scale Transformer models.

### Strengths
- Clear motivation.
- Conceptually simple yet theoretically grounded design.
- Lightweight variance approximation for efficiency.
- Empirical evidence of improved stability.

### Weaknesses
- **Inadequate reporting and questionable generality of Peri-LN throughput results.** Figure 4(b) claims that Peri-LN achieves strong accuracy but suffers from the slowest throughput, positioning BHyT as the best trade-off. However, the paper does not specify the environment under which throughput was measured. All experiments were conducted in Llama-Factory rather than in standard large-scale frameworks (e.g., Megatron-Core or NeMo) that officially support Gemma-style Peri-LN. Without such metadata or a cross-framework check, it is unclear whether the reported slowdown reflects an intrinsic limitation of Peri-LN or merely framework-specific overhead.

- **Under-trained experimental regime and limited statistical reliability.** The 1B and 3B models were trained on only 1.64 B and 1.97 B tokens, respectively—orders of magnitude below standard compute-optimal ratios. These under-saturated runs make it difficult to assess convergence, generalization, or stability trends. No results are averaged over multiple seeds or accompanied by confidence intervals. Reported downstream improvements (Tables 1–4) therefore lack statistical robustness, and the unusually large MMLU jump (36.54 % for BHyT) raises concerns about configuration consistency and reproducibility.

- **Theory–practice gap and missing validation of assumptions.** Theoretical guarantees apply only to the idealized variant BHyT* with exact statistics and zero-mean inputs. The implemented BHyT uses an approximated variance under strong assumptions—uniform attention, Gaussianity, and linearized tanh—but the paper offers no quantitative error analysis of this approximation (e.g., deviation across layers or attention-entropy regimes). As a result, the claimed Jacobian-norm bound and variance-stability guarantees remain unverified in realistic conditions.

- **Missing baselines and scale comparisons.** At the 3B scale, BHyT is compared only with DyT, omitting RMSNorm, LNS, and Peri-LN under identical configurations. This prevents a fair evaluation of scalability and undermines the claim that BHyT preserves stability at greater depth.

### Questions
stated in the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
