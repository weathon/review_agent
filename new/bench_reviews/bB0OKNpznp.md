Let me verify key claims from the paper text directly (which was provided in the user message):

**Verification 1: Only fine-tuning lmhead?**
The paper states: "To isolate the effects of QPA, we simplify the PEFT setup by freezing all layers of Gemma-2 and GPT-2, and fine-tuning only the final linear layer, commonly referred to as the 'lmhead.'"
→ Confirmed: only the final linear layer is fine-tuned.

**Verification 2: No classical hypernetwork baseline?**
Reading through the entire paper, there is no experiment comparing QPA against a purely classical parameter generator (e.g., an MLP or latent code + MLP with the same parameter count). The only baselines are standard LoRA, DoRA, PT, and FFA without any compression.
→ Confirmed: no classical hypernetwork/compression baseline.

**Verification 3: Performance numbers**
- GPT-2 LoRA: QPA achieves 0.27% params, PPL 1.583 vs LoRA 0.52% params, PPL 1.595
- Gemma-2 LoRA: QPA achieves 0.03% params, PPL 1.417 vs LoRA 0.19% params, PPL 1.418
- GPT-2 PT: QPA achieves 0.18% params, PPL 2.327 vs PT 1.01% params, PPL 2.225 (performance loss)
- DoRA baseline on GPT-2 has PPL 5.003 and on Gemma-2 has PPL 5.504, which seems extremely high

**Verification 4: DoRA baseline issue**
Looking at Table 2: DoRA achieves testing perplexity of 5.003 on GPT-2 and 5.504 on Gemma-2. This is catastrophically bad compared to LoRA (~1.5-1.6). This looks like a potential implementation issue with DoRA.

**Verification 5: Noise/experiments on real hardware**
Paper states: "This evaluation is conducted using quantum circuit simulation via PyTorch and TorchQuantum. At this stage, noise effects on the quantum system are ignored" and "A discussion on the impact of finite measurement shots and noise is provided in Appendix G."
→ Confirmed: all main results use exact statevector simulation with no noise.

**Verification 6: No variance/statistical testing**
The paper reports single numbers, no error bars, no multiple seeds.
→ Confirmed.

**Verification 7: Polylogarithmic claim vs actual experiments**
The paper claims O(polylog(m)) parameter scaling in Section 3.1, but in the actual experiments, they use the batched approach (Section 3.2) where the MLP mapping model has a hidden dimension of [32, 64, 128, 128, 64, 32, n_mlp] (Table 1). The MLP parameters are not negligible.

Let me check the actual qubit counts and MLP sizes:
- For GPT-2 lmhead: d=768, k=50257, rank=4, so LoRA has 4*(768+50257) ≈ 204k params
- For the QNN: they use 4-11 qubits
- MLP mapping model: Table 1 shows hidden dims [32, 64, 128, 128, 64, 32, n_mlp], so the MLP has a significant number of parameters

→ The "polylog" claim is indeed misaligned with the experimental setup: the MLP has many parameters that scale with n_mlp.

Now, let me think about the calibration papers:

- **Quantum-PEFT** (dgR6i4TSng): Scores 6/6/6/6, Accept Poster. This paper also applies quantum concepts to PEFT, but uses Pauli parameterization for logarithmic parameter scaling. It has stronger experimental results and more comprehensive benchmarks. QPA is weaker than Quantum-PEFT in experimental validation and lacks the classical baseline comparison.

- **Hamiltonian Classifier** (3HPOtZxs5s): Scores 3/3/3/3, Withdrawn/Reject. Key weakness: no quantum advantage over classical baselines, only simulation, doesn't outperform simple classical models. QPA has similar issues but at least applies to a more practical problem.

- **Measurement info multiple-reuse** (usX2ixXopC): Scores 3/5/3/5, Withdrawn/Reject. Claims quantum speedup but relies on unavailable quantum hardware. Similar concern applies to QPA.

- **LoRTA** (ayPfZIkovt): Scores 3/5/6/3, Reject. Performance degradation and limited experiments. QPA has similar concerns with some PEFT methods.

- **LoRA-XS** (l80AgHoRaN): Scores 5/3/5/5/5/3, Withdrawn/Reject. Inconsistent performance, lack of computational cost analysis.

- **HART** (iP8ig954Uz): Scores 5/6/5, Reject. Hypernetwork PEFT method but the large number of hypernetwork parameters contradicts PEFT's goal.

The QPA paper has a fundamental issue: it does not establish that the quantum component provides any advantage over a classical alternative. The QNN + MLP is effectively a classical hypernetwork that is simulated classically, and there's no evidence the quantum structure is essential. This is akin to the Hamiltonian Classifier paper (score 3) which also couldn't demonstrate quantum advantage.

However, QPA does have some genuine contributions:
- Novel framing connecting quantum parameter generation to PEFT
- Significant scaling over prior quantum parameter generation work
- Batched generation is a practical contribution
- Some empirical results are positive

The missing classical baseline is the most critical weakness. Without it, the "quantum" contribution is unverified. This is similar to the Hamiltonian Classifier paper's core weakness.

Given the calibration papers, QPA sits somewhere between the Hamiltonian Classifier (score 3, fundamental quantum advantage question) and Quantum-PEFT (score 6, better validated). But QPA lacks the key experiment (classical baseline) and has limited evaluation (single layer, single dataset), putting it closer to a 4.

Let me also assess: is the DoRA baseline issue real? DoRA's PPL of 5.003 on GPT-2 and 5.504 on Gemma-2 are indeed extremely poor—normal DoRA should achieve similar or better perplexity than LoRA, not 3-4x worse. This does suggest a potential implementation or configuration problem with the DoRA baseline.

Now let me write the final review.</think>Now I have enough context from the paper and calibration papers. Let me write the final review.

## Summary

This paper introduces Quantum Parameter Adaptation (QPA), a method that uses parameterized quantum circuits (PQCs) combined with an MLP mapping model to generate trainable parameters for parameter-efficient fine-tuning (PEFT) methods like LoRA, DoRA, Prefix-Tuning, and Feed-Forward Adapters. The key idea is that a PQC with N qubits and polynomial depth can produce O(2^N) measurement probabilities, enabling O(polylog(m)) generator parameters to produce m target parameters. Experiments on GPT-2 and Gemma-2 (fine-tuning only the final lmhead layer on WikiText-2) show QPA can reduce trainable parameters (e.g., to 52% of LoRA for GPT-2, 16.84% for Gemma-2) with marginal or comparable perplexity.

## Strengths

- **Novel and principled framing that avoids QML's key practical barriers**: QPA generates parameters only during training, keeping inference entirely classical. This sidesteps the data-encoding bottleneck and quantum hardware requirement at deployment—two widely acknowledged obstacles for practical QML. The framing in Figure 1 clearly delineates how QPA differs from conventional QML.

- **Significant scaling over prior quantum parameter generation work**: The paper scales quantum parameter generation from the previous largest target of 0.28M parameters to 0.52B parameters (a ~1785× increase), demonstrating feasibility at scales relevant to modern LLMs. This is a meaningful engineering contribution beyond prior small-scale CNN/LSTM demonstrations.

- **Batched parameter generation (Section 3.2) is a practical and well-designed trick**: The chunking mechanism reduces qubit requirements from ⌈log₂ m⌉ to ⌈log₂(m/n_mlp)⌉, cutting qubit counts and simulation memory substantially. The examples (30→20 qubits for 10⁹ parameters) are concrete and useful.

- **Applicability across multiple PEFT paradigms**: Demonstrating QPA on four PEFT methods (LoRA, DoRA, PT, FFA) rather than a single method shows the framework's generality.

- **QPA-LoRA results are positive for the strongest case**: For LoRA (the most popular PEFT method), QPA achieves genuine parameter reduction (52.06% for GPT-2, 16.84% for Gemma-2) with slight perplexity improvements, which is a meaningful result.

## Weaknesses

### Major:

- **No classical generator baseline, making the quantum contribution unverifiable**: The paper's core claim is that QPA provides "quantum-enhanced parameter reduction." However, the QNN + MLP is simulated entirely on classical hardware. The QNN functions as a parameterized mapping from a low-dimensional parameter vector (θ) to probabilities in [0,1]—structurally equivalent to a classical latent variable + nonlinear mapping. The paper never compares against a purely classical generator with matched parameter count (e.g., a small MLP or random projection feeding the same mapping MLP). This is the single most critical gap: without this comparison, any apparent gain over vanilla PEFT could simply reflect that tying parameters via any small generator is beneficial—a well-known result from the classical hypernetwork literature. This mirrors the fundamental weakness identified in the Hamiltonian Classifier paper (all scores 3), which likewise could not demonstrate quantum advantage over simple classical baselines.

- **Experiments are limited to fine-tuning only the final lmhead layer**: The paper explicitly states: "we simplify the PEFT setup by freezing all layers of Gemma-2 and GPT-2, and fine-tuning only the final linear layer." In practice, PEFT methods like LoRA are applied across multiple transformer layers. The parameter dynamics, qubit requirements, and interaction effects could differ substantially when fine-tuning many layers simultaneously. This setup severely limits the practical relevance of the results. As the LoRTA review noted for similarly limited PEFT experiments: "I expect to see more LLM-related results... The GLUE benchmark task is relatively simple and may not have a good distinction granularity."

- **Performance improvements are marginal and lack statistical testing; some PEFT methods show degradation**: For QPA-LoRA on GPT-2, the improvement is 1.595→1.583 (0.75%); on Gemma-2, 1.418→1.417 (0.07%). No standard deviations, confidence intervals, or multiple-seed results are reported, making it impossible to assess whether these tiny differences are meaningful. More concerning, QPA worsens perplexity for PT on GPT-2 (2.225→2.327, a 4.38% degradation) and does not outperform FFA on Gemma-2 across the entire parameter range. The abstract's claim of "comparable or improved performance" is somewhat overstated given these mixed results. As the LoRTA review noted for similar degradations: "I think it's not economical to further reduce the trainable parameter with such large performance degradation."

- **The "polylogarithmic parameter reduction" narrative is misleading in the context of the actual experiments**: The theoretical argument in Section 3.1 claims O(polylog(m)) parameter scaling, but in practice, the batched generation approach (Section 3.2) introduces a mapping MLP with hidden dimensions [32, 64, 128, 128, 64, 32, n_mlp] (Table 1). The MLP parameters are substantial and scale with n_mlp. The method as actually implemented does not achieve polylogarithmic scaling in m; rather, it achieves parameter reduction relative to uncompressed PEFT parameters, which is a fundamentally different claim. The HART review raised a similar concern: "The large number of trainable parameters in the transformer-based hypernetwork contradicts the original intent of PEFT."

- **DoRA baseline appears severely misconfigured**: Table 2 shows DoRA achieving perplexity of 5.003 on GPT-2 and 5.504 on Gemma-2. These are catastrophically poor results for DoRA, which should achieve perplexity comparable to LoRA (~1.5–1.6), not 3–4× worse. This strongly suggests a DoRA implementation or configuration error, rendering all QPA-vs-DoRA comparisons unreliable.

### Minor:

- **No computational cost or training time analysis**: The paper emphasizes parameter count reduction but does not analyze wall-clock training time or GPU memory usage. Simulating quantum circuits on classical hardware adds overhead; the practical efficiency tradeoff is unclear. As the Quantum-PEFT review noted: "Limited discussion of practical computational bottlenecks" and "Absence of detailed analysis on computational overhead."

- **All main experiments use noise-free quantum simulation**: The core results rest on exact statevector simulation, which provides perfect measurement probabilities. Appendix G discusses noise effects but this analysis is not part of the main evaluation. Actual quantum hardware would introduce shot noise and hardware noise that could degrade performance.

- **Evaluation on only one dataset (WikiText-2)**: A single perplexity benchmark is insufficient for claims of broad applicability to LLM fine-tuning, especially when only one layer is adapted.

### Trivial:
- None worth noting beyond the above.

## Nice-to-Haves

- **Classical hypernetwork baseline** with matched parameter count: this is essential but listed here because it's already captured as a major weakness. As a complementary suggestion, an ablation replacing the PQC output with fixed or random inputs to the MLP mapping model would reveal whether the quantum circuit learns non-trivial representations or whether the MLP does most of the work.

- **Multi-layer PEFT fine-tuning** across standard transformer blocks, which would demonstrate that QPA works in the standard PEFT application regime, not just on a single linear layer.

- **Standard deviation and multiple random seeds** across all experiments, particularly given the tiny absolute improvements reported.

- **Evaluation on downstream NLU benchmarks** (e.g., GLUE, MMLU) rather than only WikiText-2 perplexity.

- **Analysis of what the learned PQC states actually encode**—visualizing measurement probabilities across training would reveal whether the quantum circuit learns structured representations or acts merely as a random projection.

## Removed Points

- **"The paper lacks novelty because quantum parameter generation was proposed before"**: While prior work (Liu et al., 2024b) introduced quantum parameter generation, QPA extends it substantially: batched generation, application to PEFT (4 methods), scaling to 0.52B parameters, and the adaptation framework. The extension is non-trivial. Removing this as a weakness about novelty.

- **"Missing related works on classical hypernetworks in PEFT"**: Per the rules, I do not confirm the existence of specific missing works. However, the lack of a classical generator baseline remains a major experimental gap.

- **"Code reproducibility concerns"**: Removed per rules on reproducibility nitpicks.

- **"Formatting/style issues"**: Removed per rules on formatting nitpicks.

- **"The paper should compare to more models (LLaMA, Mistral, etc.)"**: While more models would be nice, GPT-2 and Gemma-2 already span two distinct scales. This is a nice-to-have, not a core weakness.

- **"Hyperparameter details are insufficiently described"**: Removed per rules on reproducibility nitpicks.

## Novel Insights

The batched parameter generation mechanism (Section 3.2) is the paper's most underappreciated contribution: by expanding the MLP mapping model's output from 1 to n_mlp parameters per quantum basis, it effectively trades qubit count for MLP capacity. This is a practical engineering insight that makes quantum parameter generation feasible at LLM scales with only 4–11 qubits. However, this same mechanism undermines the paper's "polylogarithmic" claim, as the MLP becomes the dominant source of parameters. The paper would be significantly stronger if it reframed its contributions around this practical engineering tradeoff rather than relying on asymptotic arguments that don't hold in the actual experimental setup.

## Suggestions

1. **Add a classical hypernetwork baseline**: Replace the PQC with a small MLP (or even a random fixed mapping) that takes a latent code of the same size as θ and feeds the same mapping MLP. If QPA still outperforms this baseline, the quantum structure matters; if not, the contribution is purely the compression framework.

2. **Verify and fix the DoRA baseline**: The reported DoRA perplexity of 5.003/5.504 is far too high and likely reflects a configuration error. This should be corrected before claiming QPA outperforms DoRA.

3. **Apply QPA to standard multi-layer LoRA fine-tuning**: Fine-tune LoRA adapters across all attention/MLP layers (the standard regime), not just lmhead, to demonstrate practical relevance.

4. **Report mean ± std over multiple seeds**: Given the 0.07–0.75% reported improvements, statistical significance is essential for the claims to be credible.

5. **Reframe the "polylog" theoretical argument**: Either show that the total parameter count (QNN + MLP) remains polylogarithmic in the target parameter count m in the actual experimental setup, or qualify the claim as applying only to the PQC parameters specifically.

## Score and Decision

**Calibration**: I compared against several papers:

- **Quantum-PEFT** (dgR6i4TSng, scores 6/6/6/6, Accept Poster): A conceptually similar paper applying quantum ideas to PEFT, but with stronger experimental validation across multiple tasks and a cleaner theoretical framework (Pauli parameterization). QPA is weaker: it has no classical baseline, limited single-layer experiments, and marginal improvements.

- **Hamiltonian Classifier** (3HPOtZxs5s, scores 3/3/3/3, Reject): Shares QPA's fundamental weakness—cannot demonstrate genuine quantum advantage over classical alternatives. QPA has slightly more practical relevance (LLM fine-tuning vs classification) but the core issue is the same.

- **LoRTA** (ayPfZIkovt, scores 3/5/6/3, Reject): A PEFT compression method with performance degradation and limited experiments. QPA has similar evaluation limitations.

- **HART** (iP8ig954Uz, scores 5/6/5, Reject): A hypernetwork-based PEFT method where the hypernetwork's large parameter count contradicts PEFT goals. QPA has a similar structural concern.

- **LoRA-XS** (l80AgHoRaN, scores 5/3/5/5/5/3, Withdrawn/Reject): PEFT compression with inconsistent performance and missing computational analysis.

QPA falls below Quantum-PEFT (6) due to missing classical baselines and limited experiments, but above the Hamiltonian Classifier (3) because it at least demonstrates a practical application at LLM scale and has some positive results. The paper's core claim of quantum advantage is unverified, which is a fundamental—not incremental—issue. However, the engineering contributions (batched generation, scaling to 0.5B parameters) and the positive QPA-LoRA results provide some value.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>