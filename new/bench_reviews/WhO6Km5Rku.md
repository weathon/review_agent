Now I have all the information I need to compose my final review. Let me verify one more critical detail about the "92-97%" claim against the data.

Looking at DeepSeek-Coder performance:
- HotpotQA: 0.256/0.339 = 75.5%
- PG19: 0.156/0.193 = 80.8%  
- SummScreen: 0.202/0.266 = 75.9%

And Phi-4-mini SummScreen: 0.220/0.267 = 82.4%

These are well below the claimed "92-97%".

Now let me finalize my review.

## Summary

QubitCache proposes a KV-cache compression method for LLMs that preserves attention patterns via a hybrid architecture: critical tokens (15%) are retained in classical storage while non-critical tokens' attention weights are encoded as quantum amplitudes, enabling soft probabilistic attention reconstruction over interpolated value vectors. The paper claims "logarithmic compression beyond classical information-theoretic limits" and "92-97% baseline performance retention" at 7× memory reduction.

## Strengths

- **Attention-based token selection with soft reconstruction is a reasonable idea**: The insight that attention-based token identification (Table 4: removing critical tokens causes a 20.4% drop) matters more than positional heuristics, combined with soft attention over value-interpolated compressed tokens rather than hard eviction, is a genuinely useful contribution to KV-cache compression. The ablation showing "No Critical" drops from 0.491 to 0.391 while "No Anchor" or "No Recent" only drops to 0.488 validates that attention-based selection is the primary driver.

- **Consistent improvement over baselines at aggressive compression ratios**: Table 1 shows QubitCache at 15% retention generally outperforming StreamingLLM, H2O, and ScissorHand at their higher retention ratios on multi-hop reasoning tasks (e.g., Mistral-7B HotpotQA: 0.604 vs. H2O's 0.487, StreamingLLM's 0.406). This demonstrates the practical benefit of soft attention reconstruction over hard eviction.

- **Evaluation across multiple models and tasks**: The paper tests on five models (4B-8B) and seven benchmarks, providing reasonable empirical coverage.

## Weaknesses

### Fatal

- **The paper's central claim of "logarithmic compression beyond classical information-theoretic limits" (Abstract, §1, §3.1, §4.4) is false.** The method encodes attention weights as quantum amplitudes: |ψ⟩ = Σ√α_i|i⟩ where α_i = ā_i/Σā_j, then "measures" to obtain p_i = |⟨i|ψ⟩|² = α_i. This is mathematically identical to normalizing attention scores into a probability distribution—a trivially classical operation. Two independent reasons kill the "beyond classical limits" claim: (a) All experiments use a **classical simulator** (Qiskit statevector, §3.2.1, line 340), which stores all N amplitudes in O(N) space. Table 3's claimed O(L·H·0.15S·D + log N) memory complexity is incorrect for this implementation. (b) Even on actual quantum hardware, extracting N probabilities from log N qubits requires O(N) measurements by Holevo's bound—sub-linear information extraction is information-theoretically impossible. The paper itself acknowledges the classical simulation (line 83: "the current implementation operates as a classical simulation"), yet continues to claim logarithmic compression. This is not a minor overclaim—it is the paper's title and central thesis.

### Major

- **The "92-97% baseline performance retention" claim is misleading.** The abstract and repeated in §4.2 states this range, but computing actual retention from Table 1 reveals many model-task combinations far below 92%: DeepSeek-Coder on HotpotQA retains only 75.5% (0.256/0.339), DeepSeek-Coder on SummScreen retains 75.9% (0.202/0.266), DeepSeek-Coder on PG19 retains 80.8% (0.156/0.193), Phi-4-mini on SummScreen retains 82.4% (0.220/0.267), and Llama-8B on TriviaQA retains 84.9% (0.247/0.291). The "92-97%" claim cherry-picks the best model-task pairs while omitting worst cases from the narrative.

- **The quantum encoding component contributes minimally, undermining the paper's framing.** Table 4 shows "Full QubitCache" (0.491) vs. "No Quantum" (0.472)—a 3.9% gap. Under random selection, "Random + Quantum" (0.335) vs. "Random No Quantum" (0.334) shows a negligible 0.3% difference. The dominant factor is attention-based token selection (20.4% degradation when removed). The paper's title, framing, and claimed contribution center on quantum encoding, but the ablation data shows it is not the primary driver of performance. The method's actual contribution—soft attention over value-interpolated tokens—is a purely classical mechanism that could be described without any quantum formalism.

- **Missing same-ratio baselines prevents fair comparison.** QubitCache operates at 15% retention while H2O, ScissorHand, and StreamingLLM operate at ~50% retention. Without running these baselines at 15%, it is impossible to determine how much of QubitCache's advantage comes from its specific soft-reconstruction mechanism versus inherent properties of the attention-based selection strategy. The "Random + Quantum" ablation (0.335) is not a meaningful same-ratio baseline because random selection is trivially poor; a natural comparison would be H2O at 15% with value interpolation.

### Minor

- **The claimed amortized O(log n) per-token update cost (§3.4, line 115) appears incorrect.** When tokens shift categories, updating the quantum state for a segment requires recomputing and renormalizing all amplitudes, which is O(n_s) for segment size n_s = 512, not O(log n).

- **No error bars or statistical significance tests** are reported for any results, despite LLM generation being stochastic.

- **The "first framework recognizing that attention patterns constitute the primary information carrier" (Abstract) overclaims novelty.** H2O and StreamingLLM already select tokens based on attention importance. QubitCache's additional step—soft reconstruction of evicted tokens—is the genuine novelty, not the insight that attention matters.

## Nice-to-Haves

- A purely classical ablation ("attention-weighted selection + soft interpolation, no quantum framing") to cleanly separate the contributions of the classical components and quantify the genuine information gain from quantum amplitude encoding versus simple probability normalization.
- Sensitivity analysis of the 15% retention ratio and segment size (512 tokens) across different architectures and sequence lengths.
- Running H2O/ScissorHand at 15% retention to enable apples-to-apples compression ratio comparisons.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Claim that referenced models/benchmarks are unreleased or unverifiable**: The paper cites specific models and benchmarks; per review policy, we accept cited entities exist.
- **Formatting/parsing artifact comments**: Issues like garbled text, equation numbering, and whitespace are parser artifacts, not author errors.
- **The "NISQ feasibility" strength from the Strength Finder**: The paper uses a classical simulator (not real quantum hardware), so claims about NISQ compatibility are speculative and not empirically validated. This supposed strength conflicts with the verified fatal weakness that the method is classically simulated and the quantum claims are false.
- **Strength Finder's claim that "ablation validates the core insight that attention-based selection matters more than token preservation"**: While true, this actually undermines rather than supports the paper's quantum-framed contribution—attention-based selection is a well-known idea from H2O/StreamingLLM, not a novel insight of this paper.
- **Strength Finder's claim that QubitCache demonstrates "high compression ratio with competitive task-level performance"**: The "7× compression" claim relies on the false O(log N) memory claim for the quantum component; the actual classical implementation's compression ratio is not meaningfully different from GEAR (6.7×), making this not a strength unique to the quantum approach.
- **Harsh critic's concern about citing Michel et al. (2019) and Choromanski et al. (2020) for the "attention topology" claim**: While the citations may be imperfectly applied, this is a minor interpretive issue, not a factual error.

## Novel Insights

The most important finding from the reviews is the disconnect between the paper's quantum framing and its empirical substance: the ablation data (Table 4) reveals that the quantum amplitude encoding provides at most 3.9% improvement (0.491 vs. 0.472), while attention-based token selection—the same principle underlying H2O and StreamingLLM—accounts for nearly all performance gains (20.4% gap when removed). The paper's actual methodological contribution, soft attention over value-interpolated tokens from attention-selected critical tokens, is a purely classical technique disguised in quantum formalism. The quantum amplitude encoding reduces to probability normalization: p_i = |⟨i|ψ⟩|² = α_i, adding zero computational benefit over straightforward softmax.

## Suggestions

- Remove or substantially revise the "quantum" and "logarithmic compression beyond classical information-theoretic limits" framing. Describe the method as what it actually is: attention-weighted token selection with soft value interpolation for evicted tokens. This would make the paper an honest (if incremental) contribution to KV-cache compression.
- Report all performance retention percentages honestly rather than cherry-picking the "92-97%" range, and add error bars across runs.
- Add same-ratio baselines (H2O and GEAR at 15% retention) and a pure classical ablation to cleanly demonstrate the method's genuine contributions.

## Score and Decision

**Calibration anchors used:**

1. **Quantum Hamiltonian Classifier** (avg 3.0, Reject): Classically simulable quantum method claiming advantage it doesn't deliver. Directly analogous to QubitCache's false quantum advantage claim, though QubitCache has more empirical substance.

2. **Retentive Network** (avg 4.75, Reject): Overclaimed novelty (repackaged linear attention as "retention"). Similar pattern of misleading framing around core contribution.

3. **Prodigy** (avg 4.25, Reject): Overclaimed convergence rate improvements with misleading claims and invalid lower bounds. Similar overclaiming pattern.

4. **FastGen/KV-cache oral paper** (avg 8.0, Accept oral): Honest, clean method for adaptive KV-cache compression with strong empirical grounding and no overclaiming.

5. **NAMMs** (avg 7.0, Accept poster): Learned attention-based memory management with clear novelty and honest evaluation.

QubitCache's underlying classical method (attention selection + soft interpolation) is a modest but real contribution, roughly comparable to other honest KV-cache compression work. However, the paper wraps this in fundamentally false quantum advantage claims that constitute its title and central thesis. The "logarithmic compression beyond classical information-theoretic limits" claim is contradicted by the paper's own admission of classical simulation, and the "92-97%" performance claim is cherry-picked from data showing actual retention as low as 75%. These are not minor presentation issues—they are the paper's defining claims. Without them, the paper would be an incremental but honest systems contribution; with them, the paper is misleading in ways that mirror rejected quantum-advantage papers. The quantum encoding ablation shows a 3.9% contribution at best (0.1% under random selection), confirming the framing is decorative rather than substantive.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>