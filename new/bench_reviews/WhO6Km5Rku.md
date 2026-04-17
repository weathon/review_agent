Let me carefully review the key claims from the paper text provided in the prompt:

1. **Quantum encoding**: The paper defines in Eq. 5: $|\psi_{S_m}\rangle = \sum_{i=0}^{n_s-1} \sqrt{\alpha_i} |i\rangle$ where $\alpha_i = \bar{a}_i / \sum \bar{a}_j$. Then the reconstruction uses $p_j(|\psi\rangle) = |\langle j|\psi\rangle|^2 = \alpha_j$. So indeed, the "quantum measurement" probabilities are exactly the normalized attention weights. This is correct as the harsh reviewer noted.

2. **Memory claims**: The paper claims O(L × H × 0.15S × D + log N) memory in Table 3. But each 512-token segment requires a 9-qubit state. For 8K tokens, you have ~16 segments, each needing a 512-dimensional statevector for classical simulation. This is O(S/512 × 512) = O(S) storage, not O(log N).

3. **Theorem**: The paper says in the abstract "We prove QubitCache preserves rank r attention structure with bounded reconstruction error" - let me check if there's a theorem in the paper... Looking through the text, I don't see any formal theorem statement or proof. This appears in the abstract and intro but not substantiated.

4. **"No Quantum" ablation**: Table 4 shows Full QubitCache: 0.491, No Quantum: 0.472. The paper says (§4.5.1) "the comparison between Full QubitCache and No Quantum reveals that quantum amplitude encoding provides a 3.9% performance improvement by partially preserving information from discarded tokens."

5. **Real quantum hardware**: The paper explicitly states "the current implementation operates as a classical simulation" (§3.2.2).

Now let me also check: the paper says "The practical feasibility of our approach is demonstrated through implementation using 9-qubit circuit designs that operate within the coherence constraints of current noisy intermediate-scale quantum devices" (§1). But they don't run on NISQ devices; they run on a classical simulator. This is a valid concern.

Let me also verify the "15% retention" claim. In §3.1: "anchor tokens (first 4 positions)... recent tokens (last 10% of sequence)... critical tokens... The first three categories are preserved in classical storage, constituting approximately 15% of the original sequence." So at 8K tokens, anchor = 4, recent = 800, critical = some number. The paper says "approximately 15% of the original sequence" = 1200 tokens. So anchor + recent = 804, leaving ~396 critical tokens. This seems plausible.

The "Random + Quantum" vs "Random No Quantum" ablation shows 0.335 vs 0.334, confirming the quantum encoding adds almost nothing when token selection is random.

Now let me synthesize the final review.

## Summary

QubitCache proposes a KV-cache compression framework for LLMs that retains 15% of tokens in classical storage while encoding the attention patterns of discarded tokens into "quantum-inspired" amplitude-encoded states. Reconstruction uses measurement probabilities ($\alpha_i = \bar{a}_i / \sum \bar{a}_j$, which are just normalized attention weights) and inverse-distance-weighted value interpolation. Experiments across five models (4B–70B) and multiple benchmarks claim 7× memory reduction while maintaining 92–97% of baseline performance.

## Strengths

1. **Addresses an important problem**: KV-cache compression is critical for LLM deployment, and the insight that relational/attention patterns matter more than individual tokens is well-motivated and supported by the ablation (Table 4: attention-based selection causes 20.4% drop when removed).

2. **Comprehensive empirical coverage**: Evaluation across five models (Llama-8B, Mistral-7B, Phi-4-mini, Qwen2-7B, DeepSeek-Coder-7B) and seven benchmarks, including scaling to 70B, provides breadth.

3. **Strong results on multi-hop reasoning**: HotpotQA F1 of 0.604 (Mistral-7B) vs. H2O's 0.420 and StreamingLLM's 0.403 is notable, supporting the thesis that preserving relational information helps cross-document reasoning.

## Weaknesses

### Major

- **The "quantum" mechanism is mathematically redundant under classical simulation**: The core quantum claim—that amplitude encoding provides logarithmic compression—is incorrect under the paper's own implementation. The measurement probabilities $p_j = |\langle j|\psi\rangle|^2 = \alpha_j$ are exactly the normalized attention weights $\bar{a}_j / \sum \bar{a}_j$. There is no meaningful quantum operation beyond reparameterizing a probability vector. Under classical simulation via Qiskit's statevector simulator, each 9-qubit state requires storing a 512-dimensional complex vector—the same order as storing 512 attention scores directly. The claimed O(log N) memory advantage only applies to hypothetical quantum hardware, not the actual implementation. The paper's headline claims of "logarithmic compression beyond classical information-theoretic limits" (§1, §3.1) are unsupported under the described system. This is structural because the paper's primary framing and novelty narrative depend on quantum encoding providing a genuine algorithmic advantage.

- **Memory analysis is incomplete/misleading**: Table 3 claims QubitCache uses 0.55 GB but does not account for the memory needed to store the statevector simulations (16 segments × 512 complex amplitudes per segment per layer per head). The comparison against GEAR (0.59 GB at 6.7× compression) should be scrutinized: GEAR achieves nearly the same compression ratio through straightforward quantization, while QubitCache's actual memory footprint including simulator state may be significantly higher than reported. No latency or throughput measurements are provided either, which is essential for a system paper claiming practical deployment benefits.

- **The promised theoretical guarantee is absent**: The abstract and introduction state "We prove QubitCache preserves rank r attention structure with bounded reconstruction error," but no theorem statement, formal bound, or proof appears anywhere in the paper. Given the complexity of the proposed method (selective retention + probabilistic reconstruction + interpolation), such a bound would be nontrivial to establish, and its absence undermines a core claimed contribution.

### Minor

- **Segment-wise encoding severs cross-segment attention**: Encoding tokens in independent 512-token segments (§3.2) directly contradicts the stated motivation of preserving "global relational structure," since attention patterns spanning segment boundaries cannot be captured.

- **Ablation lacks a critical control**: The "No Quantum" baseline does not clarify whether it uses uniform weighting, zero-weighting for non-preserved tokens, or the same normalized attention weights without quantum circuit simulation. A simple softmax-over-attention-weights baseline (which is what the quantum method actually computes) would demonstrate whether the quantum formalism provides any benefit beyond the soft weighting scheme.

- **Quantum hardware feasibility claims are overstated**: The paper repeatedly claims "9-qubit circuit designs that operate within the coherence constraints of current NISQ devices" (§1) and presents circuit depth analysis (§4.5.2), but all experiments use noiseless classical simulation. No actual quantum hardware experiments or noise modeling are conducted. The claim that this is "not merely a theoretical construct but a practically implementable solution" on NISQ devices is unsubstantiated.

- **Evaluation limited to 2K–8K sequences**: The introduction motivates the problem for "100K tokens," but all experiments use at most 8K tokens, where KV-cache pressure is moderate. The method's advantages should be demonstrated at 32K+ tokens.

### Trivial

- The mixing coefficient λ = √(|Ip|/N) is introduced without justification for this specific functional form.

## Nice-to-Haves

- A latency/throughput comparison across all methods, since practical deployment requires both memory savings and acceptable speed.

- A principled "classical-only" ablation that uses normalized attention weights as a probability distribution over non-preserved tokens (matching what the quantum method computes) to cleanly isolate the contribution of the quantum circuit formalism.

- Experiments at longer contexts (32K+) where KV-cache compression matters most.

## Removed Points

- **Claim that "first framework" for attention-pattern preservation is overstated**: While other methods (Performer, Linformer, Compressive Transformer) do target attention patterns, they are different approaches (low-rank approximation, compression) rather than probabilistic KV-cache compression. The novelty framing could be improved, but this is a framing issue, not a factual error.

- **Demand for NISQ hardware experiments**: Reviewers flagged the lack of real quantum experiments. While valid, this is a standard limitation of quantum-inspired classical work; the paper should be evaluated on what it implements, with the caveat that quantum hardware claims should be qualified.

- **Missing baselines like KVQuant, KIVI, DMC**: The paper already compares against five baselines including the relevant categories (eviction: H2O, ScissorHands; streaming: StreamingLLM; quantization-plus-eviction: GEAR). Adding more quantization baselines would strengthen but is not a critical omission.

- **Variance/confidence intervals not reported**: This is standard practice for the benchmarks used; single-run evaluation is common at this scale.

- **"No cloning theorem" conflation**: The paper claims "quantum states can be efficiently cloned and measured in parallel" (§3.4), which misstates the quantum no-cloning theorem. However, since the implementation is classical, this is a theoretical imprecision rather than a functional error.

## Novel Insights

The paper's most valuable insight is actually the classical mechanism hidden beneath the quantum framing: that soft, probabilistic interpolation of evicted token values using attention-weighted distributions and inverse-distance-weighted value reconstruction significantly outperforms hard eviction on multi-hop reasoning tasks (Table 1: HotpotQA F1 of 0.604 vs. 0.420 for H2O). The ablation (Table 4) confirms that attention-based token selection is the primary driver of performance, while the quantum component contributes only ~4%. If the paper were reframed to focus on this classical soft-eviction mechanism with rigorous theoretical analysis, proper memory accounting, and latency evaluation, it would represent a solid contribution to practical KV-cache compression.

## Suggestions

1. **Reframe without quantum overclaim**: Present the method as a hybrid token-retention + probabilistic attention reconstruction scheme. The quantum formalism can be kept as a future direction for genuine quantum hardware deployment, but the current contribution is the soft attention reconstruction mechanism.

2. **Provide true memory accounting**: Include the memory cost of storing segment attention distributions (whether as statevectors or simple probability vectors) in the compression ratio calculation.

3. **Add latency benchmarks**: Report tokens/second or wall-clock time for all methods.

4. **Add a proper "classical softmax" ablation**: Replace the quantum measurement step with direct use of normalized attention weights as probabilities over interpolated values, to cleanly isolate the quantum circuit's contribution.

## Score and Decision

**Calibration anchors:**
- FastGen (KV-cache compression, strong practical results): 8 → Accept (oral)
- MatryoshkaKV (KV-cache compression, decent results, accepted methods): 6 → Accept (poster)
- FreqKV (KV-cache compression, novel frequency-domain idea, rejected): 5-8 avg ~5.8 → Reject
- Quantum Neural Fields (quantum framing, overclaimed, rejected): 1-8 avg ~5 → Reject
- Quantum entanglement for attention models (quantum ML, weak): 3-6 avg ~4.5 → Reject
- Enhancing GNNs with quantum encodings (quantum classical hybrid, weak): 3-5 avg ~3.5 → Reject
- ChunkKV, KV-Dict (KV-cache compression, rejected): 5-6 → Reject

This paper has a fundamentally flawed core claim (quantum advantage / logarithmic compression), which it does not actually deliver in its implementation. The real contribution — a soft probabilistic eviction scheme with IDW interpolation — is buried under misleading quantum packaging. The empirical results are reasonable but not exceptional: GEAR achieves comparable compression (6.7×) with a much simpler method, and the lack of latency metrics makes practical claims unsubstantiated. At the same time, the underlying classical idea (attention-based selection + soft reconstruction) is sound and empirically validated, preventing this from being a complete loss.

Compared to papers with similar profiles: Papers that overclaim quantum advantages while running classical simulations (Quantum Neural Fields, QViT, Quantum Entanglement for Attention) scored in the 3-5 range. KV-cache compression papers with genuine methodological issues (ChunkKV, KV-Dict) scored 5-6 and were rejected. Papers with strong methods and honest claims (FastGen) scored 8.

This paper's strength is that the underlying classical method works and has reasonable empirical validation. Its critical weakness is the misleading quantum framing that undermines the paper's central novelty claim. This is more severe than a simple "framing issue" because the complexity claims and compression ratios are derived from quantum arguments that don't hold classically.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>