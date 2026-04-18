## Summary

The paper introduces Quantum Parameter Adaptation (QPA), a method that uses parameterized quantum circuits (PQCs) combined with a classical MLP mapping model to generate trainable parameters for PEFT methods (LoRA, DoRA, Prefix-Tuning, Feed-Forward Adapters). QPA exploits the exponential mapping from PQC parameters (O(polylog(m))) to Hilbert space amplitudes (size 2^N) to reduce the number of trainable parameters needed for fine-tuning. Experiments on GPT-2 and Gemma-2 fine-tuning only the final linear layer on WikiText-2 show that QPA can reduce trainable parameters to 52.06% (GPT-2) and 16.84% (Gemma-2) of LoRA while maintaining comparable perplexity.

## Strengths

- **Conceptual clarity and principled framework**: The QPA framework combines quantum parameter generation with PEFT in a clean, well-specified manner. The batched parameter generation mechanism (Sec. 3.2) is a practical solution to qubit scaling, and the adaptation to LoRA, DoRA, PT, and FFA (Sec. 3.3) demonstrates the generality of the approach.

- **Significant scale-up from prior quantum parameter generation work**: Scaling from the largest prior target model of ~0.28M parameters (Liu et al., 2024c) to 0.52B parameters (Gemma-2 lm_head) represents approximately a 1785× increase, establishing a much more realistic benchmark for quantum parameter generation.

- **Practical inference decoupling**: Unlike conventional QML that requires quantum hardware during inference, QPA generates parameters only during training, with inference performed entirely classically. This addresses a genuine practical limitation of QML deployments.

- **Honest reporting of mixed results**: The paper explicitly acknowledges cases where QPA underperforms baselines (e.g., PT on GPT-2 at 4.38% worse perplexity; FFA on Gemma-2 not consistently outperforming), which adds credibility.

- **Consistent parameter reduction across PEFT methods**: The framework flexibly applies to four different PEFT methods, and the parameter reductions are substantial (e.g., Gemma-2 PT: 0.20% → 0.01%, FFA: 0.40% → 0.01% of model parameters).

## Weaknesses

### Fatal

None that would invalidate the entire paper, but see the major weaknesses below that significantly undermine the core claim.

### Major

- **No comparison to a classical compressed-generator baseline, undermining the central "quantum" claim**: The core claim of the paper is that QPA leverages "quantum circuit-based compression" and that the "high-dimensional Hilbert space facilitates an efficient representation." However, the PQC used in all experiments produces only 4–11 qubits' worth of probabilities, which is trivially simulable on classical hardware. The entire pipeline (PQC → probabilities → MLP → PEFT parameters) is executed as a classical computation. A classical analogue—e.g., replacing the PQC with a small MLP or learned embedding of dimension N = O(log n_ch) feeding the same mapping network—would test whether the quantum structure provides any benefit beyond "a low-dimensional latent code generates many parameters." This is exactly what hypernetworks do classically. Without this ablation, none of the reported improvements can be attributed to quantum structure rather than to the general principle of compressed parameter generation. This is the single most important missing experiment for a paper whose central premise is quantum enhancement.

- **Extremely narrow evaluation protocol, overstated claims about "practical application"**: All experiments tune only the final linear layer (lm_head) on WikiText-2 perplexity. Standard PEFT evaluation fine-tunes attention/FFN layers across multiple transformer blocks on diverse downstream tasks (e.g., GLUE, commonsense reasoning, MMLU). Fine-tuning solely the lm_head is known to yield limited adaptation; it does not test the method's behavior under the multi-layer, multi-task conditions where PEFT methods are actually deployed. The paper's conclusion that it offers "a scalable quantum-classical solution for fine-tuning LLMs" and represents "the first example of quantum computing applied to fine-tuning classical LLMs at a practical scale" is overstated relative to this limited evaluation.

- **Marginal and inconsistently positive performance improvements without statistical significance**: The claimed improvements over LoRA are 0.75% for GPT-2 and 0.07% for Gemma-2. QPA-PT on GPT-2 is 4.38% worse than PT. For FFA on Gemma-2, QPA does not outperform the classical baseline. No error bars, confidence intervals, or multi-seed results are reported. Given the small magnitude of improvements and the known variance in LLM fine-tuning, these differences cannot be reliably attributed to the method rather than optimization stochasticity. The paper frames these as "significant parameter reduction while maintaining comparable or improved performance," but "comparable" would be more accurate than "improved."

### Minor

- **Computational overhead not reported**: While parameter count is reduced, QPA introduces the overhead of quantum circuit simulation and MLP forward/backward passes during training. The paper does not report wall-clock time, GPU memory, or FLOPs comparisons with standard PEFT, leaving open whether parameter reduction translates to practical training efficiency gains. Since PEFT methods already drastically reduce trainable parameters, the practical importance of further reducing them depends on whether training resources are also saved.

- **The "polylogarithmic" scaling claim is partially undermined by the batched mechanism**: The theoretical contribution bullet point claims QPA "utilizes quantum parameters from the PQC, which scale in proportion to the number of qubits N, to generate parameters that scale with the Hilbert space size 2^N." In practice, the batched mechanism (Sec. 3.2) uses the MLP mapping model whose hidden dimensions include n_mlp, and the total trainable parameters include both quantum θ and classical b parameters. Since n_mlp is a free hyperparameter that grows to achieve good performance (ranging from 256 to 65536), the effective parameter budget is O(polylog m) + O(poly(n_mlp)) rather than purely O(polylog m). The paper is transparent about total parameter counts in Table 2, so this is primarily an overclaim in the framing rather than hidden information.

- **Noise and real-hardware feasibility are deferred to an appendix**: All experiments use noiseless classical simulation. Appendix G discusses noise effects but no quantitative results appear in the main text. The claim that QPA uses "4 to 11 qubits" suitable for near-term hardware is untested on actual quantum devices. This matters because the practical utility claim assumes current or near-term quantum hardware availability.

### Trivial

- The notation in Eq. 1 uses $R_Y^j(\theta_j^{(L)})$ but the text sometimes writes "$R_j^L$", which could confuse readers unfamiliar with quantum circuit notation.
- Section 4.2 references "Figure 4 (b)" for optimal LoRA rank results but the discussion could benefit from a clearer articulation of why certain ranks are optimal.

## Nice-to-Haves

- Compare QPA against a purely classical compressed parameter generator (e.g., an MLP with the same number of trainable parameters replacing the PQC) to isolate the quantum-specific contribution. This is the single most impactful experiment the paper could add.
- Extend evaluation to multi-layer PEFT (applying QPA to attention/FFN layers across the model) and at least one downstream task benchmark beyond perplexity.
- Report wall-clock training time and GPU memory consumption for QPA vs. standard PEFT to establish whether parameter savings translate to practical efficiency gains.
- Provide results with multiple random seeds and report standard deviations for all perplexity numbers.

## Removed Points

- **"The paper does not cite related X"**: The harsh critic and neutral reviewer suggest missing related works in classical hypernetworks and compressed PEFT (e.g., VeRA, LoRA-XS, LoRTA). These are valid comparisons to make, but per the rules, I should not flag missing citations as I cannot verify their existence and content.

- **"Theoretical 'polylogarithmic' claim is misleading"** (harsh critic point #2): The harsh critic argues the polylog claim is a "structural" weakness because the MLP parameters scale with n_mlp. The paper does disclose total parameter counts and the batched mechanism is explicitly described. The overclaim is in the framing (bullet point 1 in contributions), not in hidden information. I've kept a weakened version in Minor weaknesses.

- **"Core interpretation that Hilbert space structure enables more precise exploration is untested"** (harsh critic point #4): This is essentially a restatement of the missing classical baseline concern, which I've already captured in the major weakness. Treating it as a separate structural issue would be double-counting.

- **"Solovay-Kitaev argument is misleading in NISQ context"**: The Spark reviewer notes that the Solovay-Kitaev argument for "deeper is better" is theoretically correct but empirically misleading because deeper circuits exacerbate noise. The paper acknowledges this trade-off in Sec. 4.2 and Appendix H discusses barren plateaus. This is a fair concern but not a fundamental flaw—it's a known tension between expressiveness and noise in variational quantum circuits.

- **"Reducing from 0.19% to 0.03% of model parameters is reducing an already tiny number"**: This observation from Spark is valid contextually but the paper targets parameter sharing and representation efficiency, not absolute memory savings. The claim is that you can reduce PEFT parameters by ~5-6× while maintaining performance, which is meaningful regardless of the starting percentage.

- **"No fair comparison with other baselines because the asymmetry favors QPA"**: Per the rules, I should not flag unfair comparisons where the asymmetry favors the baseline. In this case, QPA and LoRA are compared at comparable parameter counts, not at comparable architectures, which is a fair parameter-efficiency comparison.

## Novel Insights

The paper's most interesting empirical finding is not the marginal perplexity improvements but rather the fact that PEFT parameters for an LLM's linear layer can be generated from an extremely compressed representation (4-11 qubits → ~2^4 to 2^11 probabilities → MLP → millions of parameters) while preserving performance. This establishes a form of "parameter redundancy" in PEFT: the effective dimensionality of the low-rank adaptation space can be captured by a much lower-dimensional generator, which is consistent with recent observations from hypernetwork and tensor decomposition methods. Whether this low-dimensional structure is specifically quantum in nature—or just an artifact of any compressed representation—is exactly the question the paper leaves unanswered.

## Suggestions

- **Add a classical ablation**: Replace the PQC with a small MLP of the same parameter count (N inputs, same depth), keeping everything else identical. If the classical version matches QPA, the quantum framing is cosmetic; if it doesn't, you have a genuine quantum advantage result. This single experiment would dramatically strengthen the paper.
- **Evaluate on a downstream task**: Even one task (e.g., CommonsenseQA or HellaSwag) beyond WikiText-2 perplexity would significantly bolster the practical relevance claim.
- **Apply QPA to multiple transformer layers**: The most common PEFT setup fine-tunes attention and FFN sublayers across the model. Demonstrate QPA in this standard setting rather than only the lm_head.
- **Break down trainable parameters**: Report what fraction of the total trainable parameters comes from the PQC (θ) vs. the MLP mapping model (b) for each configuration, so readers can assess the actual "quantum compression" contribution.

## Score Calibration

- **Quantum-PEFT** (scores: 6, 6, 6, 6): A comparably motivated quantum-PEFT paper with Pauli parameterization, evaluated on language/vision benchmarks. Received middle-of-the-road poster acceptance. QPA shares similar motivation but has a narrower evaluation (single layer, single dataset, no downstream tasks) and a more fundamental gap (no classical baseline to validate the quantum claim).

- **Efficient Quantum Classifier** (scores: 3, 3, 3, 3): A quantum ML paper rejected for lack of demonstrated quantum advantage, classically simulable methods, and weak baselines. QPA is in a similar situation but scales to much larger models and has more realistic experiments.

- **LoRTA** (scores: 3, 5, 6, 3): A PEFT method rejected for performance degradation (~3%), limited LLM evaluation, and unclear motivation for ultra-low parameter count. QPA has a similar profile but with a more novel quantum framing and larger-scale experiments.

- **SMT** (scores: 8, 6, 6, 5, 6): A strong PEFT paper with extensive evaluation across models and tasks, showing consistent improvements. This represents the quality bar for a well-evaluated PEFT contribution.

QPA is weaker than Quantum-PEFT (which had broader evaluation and clearer methodological contribution) and substantially weaker than standard PEFT papers like SMT. It is stronger than the rejected Efficient Quantum Classifier (which had only toy-scale experiments) because QPA genuinely scales to a realistic LLM. The lack of a classical baseline is a critical scientific gap that prevents acceptance of the quantum advantage claim, and the narrow evaluation limits practical relevance. Overall, QPA represents an interesting conceptual framework with meaningful scale-up, but insufficient experimental evidence for its core claim.

MY FINAL SCORE: <pineapple>4</pineapple>
MY FINAL DECISION: <orange>Reject</orange>