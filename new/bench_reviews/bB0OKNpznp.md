## Summary

The paper proposes Quantum Parameter Adaptation (QPA), a quantum-inspired framework in which a small parameterized quantum circuit (QNN) and a classical MLP “mapping model” generate the trainable parameters of standard parameter-efficient fine-tuning (PEFT) modules (LoRA, DoRA, Prefix-Tuning, adapters) for LLMs. Experiments on GPT‑2 and Gemma‑2, fine-tuning only the final linear layer on WikiText‑2 via classical simulation of 4–11‑qubit circuits, show sizable reductions in the number of *trainable* parameters for these PEFT modules with mostly comparable perplexity.

## Strengths

- **Conceptually interesting hypernetwork-style view on PEFT:**  
  The paper reinterprets PEFT parameters (LoRA low-rank factors, prefixes, adapters) as outputs of a compact generator (PQC + MLP) instead of directly learned free parameters. This “parameter generation” perspective is clearly explained in §3.1–3.3 and is broadly applicable: the same mechanism is instantiated for LoRA, DoRA, PT, and FFA in §4.1. Independent of quantum framing, this is a reasonable and non-obvious way to impose strong parameter sharing over very large layers (e.g., Gemma‑2’s 0.52B‑parameter lmhead).

- **Scaling quantum-parameter-generation ideas to a realistic LLM scale (in structure, if not in full training):**  
  Prior quantum parameter generation work cited by the authors targets models up to 0.28M parameters. Here, the target layer is 0.52B parameters (§1, contribution 3), and QPA generates all PEFT parameters for that layer from a small set of quantum + MLP parameters, while using only 4–11 qubits (§4.2, Fig. 4a). This is a meaningful step in showing that quantum parameter-generation schemes can at least be *formulated* for LLM-sized layers.

- **Demonstrated parameter reduction across several PEFT methods and two LLMs:**  
  For the specific last-layer adaptation setup and WikiText‑2, QPA can reduce the ratio of trainable parameters relative to the lmhead size by large factors while preserving or slightly improving perplexity:
  - GPT‑2 / LoRA: from 0.52% to 0.27% of the lmhead parameters, with reported perplexity 1.595 → 1.583 (Table 2).  
  - Gemma‑2 / LoRA: from 0.19% to 0.03%, with perplexity 1.418 → 1.417.  
  For PT and FFA, QPA achieves extreme reductions (e.g., Gemma‑2 PT from 0.20% to 0.01% and FFA from 0.40% to 0.01%) with modest perplexity degradation (Table 2, Fig. 3). This is a nontrivial engineering result: parameter counts are sharply reduced without catastrophic loss.

- **Clear exposition of the batched parameter-generation mechanism and its trade-offs:**  
  §3.2 carefully explains how chunking and the mapping model’s decoder-style output (of size \(n_{\text{mlp}}\)) reduce qubit requirements from \(\lceil \log_2 m \rceil\) to \(\lceil \log_2 \lceil m / n_{\text{mlp}}\rceil\rceil\), and Fig. 4a makes the resulting 4–11‑qubit usage explicit. The memory argument (state size reduced by \(1/n_{\text{mlp}}\)) is correct at the level of state-vector simulation and is useful for readers considering simulators or future hardware.

- **Reasonably clear and accessible writing for a quantum–ML hybrid topic:**  
  Despite the quantum background, the main pipeline (Fig. 1, §3.1–3.3) is explained in plain terms, with equations that connect directly to the algorithmic steps (probabilities from PQC → MLP → PEFT parameters → loss). For an audience familiar with both QML and PEFT, the paper is easy to follow.

## Weaknesses

### Fatal

None rise to the level of “not even a paper”: the idea and experiments are coherent. However, there are **serious conceptual gaps around the quantum claims** and **missing baselines** that, in my view, prevent acceptance in its current form.

### Major

- **1. No demonstration that the “quantum” component provides any advantage over a purely classical generator**

  The central narrative is that the PQC’s Hilbert-space structure enables efficient parameter generation and contributes meaningfully to compression. But in all experiments:

  - Circuits are tiny (4–11 qubits, §4.2, Fig. 4a), simulated exactly, with fixed shallow depth \(L\) in main experiments.
  - The mapping model \(\tilde{G}_b\) is a nontrivial MLP with hidden layers [32, 64, 128, 128, 64, 32, \(n_{\text{mlp}}\)] (Table 1). It directly maps from (basis index, probability) to parameter chunks and clearly has significant expressive capacity.
  - There is **no ablation or classical baseline** where:
    - The PQC is removed or replaced by a learnable embedding / small classical network over the basis index; or
    - A hypernetwork-only architecture with a comparable number of parameters generates the same PEFT parameters.

  Without such baselines, it is impossible to tell whether any gains are due to the quantum circuit, or simply to the mapping MLP plus strong parameter tying across the lmhead (which classical hypernetworks are known to exploit effectively). This concern is heightened by analogous findings in other simulator-based QML work, where classical submodules often dominate.

  As written, the empirical results support a claim like “a compact generator with heavy weight sharing can compress PEFT parameters,” but **do not support** a specifically *quantum* advantage.

- **2. The parameter-count compression story misattributes the savings and is not matched to an appropriate baseline**

  The headline numbers (e.g., “GPT‑2 LoRA parameters reduced to 52.06% with a 0.75% performance gain,” “Gemma‑2 LoRA to 16.84% with 0.07% gain,” §4.1, Table 2) compare:

  - Baseline: direct trainable LoRA/DoRA/PT/FFA parameters on the lmhead; versus
  - QPA: PQC + mapping-MLP parameters, with the PEFT parameters generated and **never themselves counted as trainable parameters**.

  Conceptually:

  - QPA enforces strong parameter sharing: each chunk’s parameters are nonlinear functions of a low-dimensional latent (basis index + probability). This defines a **different hypothesis class** than standard LoRA/DoRA/PT/FFA; it is not an equally expressive reparameterization with fewer “degrees of freedom.”
  - Thus, comparing “number of trainable parameters” between these structurally different classes is not equivalent to comparing two implementations of the same model. It is closer to: “a more constrained, heavily tied model can do nearly as well as a less constrained one,” which is unsurprising and mostly classical.
  - The more relevant baseline would be **a purely classical hypernetwork** that takes the same chunk index and outputs chunk parameters, with comparable parameter count. If that matches or outperforms QPA, then the savings have nothing to do with quantum mechanics.

  This mismatch does not make the numerical results wrong, but it undermines the interpretation that QPA demonstrates a special “quantum-enhanced parameter reduction” beyond what classical hypernetworks could achieve.

- **3. Theoretical “polylogarithmic scaling” argument is incomplete and misleading for the full system**

  In §3.1 and the contributions, the paper emphasizes that:

  > “with polynomial layers in the PQC, we can generate \(2^{\lceil \log_2 m \rceil} \ge m\) parameters using \(O(\text{polylog}(m))\) PQC parameters.”

  and then claims:

  > “Since the input size of \(G_{\mathbf{b}}\) is \(N+1\), the size of \(\mathbf{b}\) can also be controlled at a scale of \(O(\text{polylog}(m))\).”

  This is not justified:

  - The parameter count of an MLP depends on *both* input and output/hidden sizes. While the input is \(N+1 = O(\log m)\), the output dimension for the non-batched case is \(m\), and for the batched case is \(n_{\text{mlp}}\) per chunk, so the number of parameters in \(\mathbf{b}\) is at least linear in the output size (or in \(m/n_{\text{mlp}}\)) unless hidden layers are collapsed to a trivial size.
  - In practice, the mapping model uses a nontrivial hidden structure (Table 1) and a decoder-style expansion to \(n_{\text{mlp}}\), which can be as large as 65,536 (§4.1). There is no derivation showing that \(|\mathbf{b}|\) remains polylogarithmic in \(m\) under these realistic choices.
  - §3.2 explicitly acknowledges the trade-off: decreasing qubits via batch generation **increases** mapping-model parameters.

  Consequently, the *quantum* parameters are indeed polylogarithmic in \(m\), but the **overall learnable parameter count** (PQC + MLP) used in practice is not shown to possess any special asymptotic scaling beyond what a cleverly designed classical hypernetwork could also claim. The current text blurs this distinction and overstates the compression attributable to the quantum part.

- **4. Evaluation scope is too narrow to support claims of “practical fine-tuning” and “scalability”**

  The paper repeatedly positions QPA as:

  - A “practical application of QML in LLMs” (contribution 2),
  - A “scalable quantum-classical solution for fine-tuning LLMs while preserving feasibility of inference on classical hardware” (abstract, conclusion),
  - And emphasizes scaling up quantum parameter generation to a 0.52B-parameter layer.

  However, the actual evaluation (§4):

  - Uses **only WikiText‑2**, a relatively small language modeling dataset.
  - Fine-tunes **only the final linear “lmhead” layer**, freezing the entire transformer stack. This is closer to linear-probe adaptation than to the way PEFT is used in practice (multi-layer LoRA/DoRA across attention and MLP blocks for varied downstream tasks).
  - Uses **only perplexity** as a metric; no downstream QA, summarization, instruction-following or robustness evaluations.
  - Does not report results over multiple random seeds or provide variance estimates; reported perplexity differences (e.g., 1.418 vs 1.417) could easily be within run-to-run noise.

  In this limited setting, the work is best viewed as an interesting **proof-of-concept on a simplified task**, not as an established practical tool for LLM fine-tuning. The current claims about practicality and scalability should be toned down or better supported.

- **5. Missing efficiency analysis: parameter savings are not related to computational cost**

  All efficiency discussion is in terms of the number of *trainable* parameters relative to the target layer size. There is no analysis of:

  - Training-time wall clock,
  - FLOPs per step,
  - Memory usage for state-vector simulation and backpropagation through the PQC,
  - Or comparison of these quantities between QPA and baseline PEFT.

  Since the experiments use exact quantum state simulation and gradient backpropagation (§4), this is nontrivial computational work. For 4–11 qubits, simulation is cheap in an absolute sense, but it is still overhead compared to purely classical PEFT; and the “memory savings” argument in §3.2 is only relevant when one is *actually* memory-bound by quantum state storage. Without such data, the claim that QPA is an “efficient” solution is one-sided: it trades off parameter count for training complexity, and the net benefit is unclear.

- **6. No robustness / multi-seed analysis, yet headline claims hinge on very small performance differences**

  The strongest “improvements” over LoRA are:

  - GPT‑2: 1.595 → 1.583 (0.75% difference),
  - Gemma‑2: 1.418 → 1.417 (0.07% difference) (§4.1, Table 2).

  Conversely, for PT and FFA, QPA sometimes incurs nontrivial performance losses (e.g., GPT‑2 PT from 2.225 to 2.327 with ~5× parameter reduction). Yet:

  - There are no error bars or statistics over multiple runs.
  - No discussion of sensitivity to initialization or optimization hyperparameters is provided.
  - No indication is given that these very small gains are statistically or practically meaningful.

  For a new and relatively complex training pipeline (quantum simulator + MLP + PEFT), it is important to demonstrate stability and statistical robustness; otherwise, the narrative “QPA slightly improves performance while cutting parameters” is on shaky ground.

### Minor

- **7. Limited range of tasks and architectures relative to the framing**

  While the jump from small CNNs/LSTMs to GPT‑2 and Gemma‑2 is significant compared to prior quantum parameter generation work, both LLMs are evaluated only on WikiText‑2 language modeling in a last-layer-only setting. There is no exploration of, e.g.,:
  - Multi-layer LoRA or DoRA throughout the transformer,
  - Other pretraining corpora or downstream benchmarks,
  - Non-language domains (e.g., vision), despite the generality of the method.

  Given that the quantum-centric narrative emphasizes “large-scale hybrid quantum-classical tasks,” a broader experimental canvas would make the impact more convincing.

- **8. Some interpretive overreach around deeper QNNs and universality**

  §4.2 (“Effect of Deeper QNN”) invokes the Solovay–Kitaev theorem and universal gate sets to argue that deeper QNNs can, in principle, approximate any unitary, and hence any “optimal” mapping to PEFT parameters. In practice:

  - The experiments only explore modest repetition counts L and small qubit numbers.
  - The actual parameterization includes a fixed classical mapping MLP whose expressivity is not analyzed.
  - Optimization dynamics, not just expressivity, will determine whether useful configurations are reached.

  The invocation of universality theorems risks giving an impression of guaranteed asymptotic optimality that is not warranted by the finite, noisy optimization regime actually studied.

- **9. No direct comparison to prior quantum-PEFT or quantum-parameter work beyond scale**

  The paper positions QPA as scaling quantum parameter generation to LLMs, but does not provide empirical or conceptual comparisons to closely related approaches such as Quantum-PEFT (unitary parameterizations for PEFT) beyond noting differences in model sizes. Some discussion of when QPA’s “generator” view is preferable to unitary parameterizations (and whether they could be combined) would clarify its niche.

### Trivial

- Minor notation/typo issues (e.g., slight inconsistencies in equation formatting, duplicated figure captions) do not materially affect understanding and can be fixed in revision.

## Nice-to-Haves

- **Classical + quantum ablations on learned representations:**  
  It would be informative to inspect the learned PQC measurement distributions (e.g., how they vary across basis states and training epochs), and compare them to a classical embedding baseline. If, for example, the probabilities remain near-uniform or change very little through training, that would suggest the PQC is not heavily utilized, which in turn could guide architecture refinement.

- **Shot-noise and noise-model experiments on small hardware or shot-based simulators:**  
  Appendix G is mentioned as discussing noise and finite shots conceptually. Small-scale experiments with finite-shot sampling on a simulator (or minimal real-hardware runs, even for reduced models) would make the claim “decoupling inference from quantum hardware while leveraging near-term quantum resources during training” more concrete.

## Removed Points

These points are flagged to be removed as critiques; they may still be useful for authors but are not treated as valid weaknesses for evaluation.

- **Criticism that any cited model/benchmark/dataset “might not be released or available”**  
  Not applicable here; the paper’s use of Gemma‑2, GPT‑2, WikiText‑2, and TorchQuantum is all standard and should be presumed valid.

- **Critiques based on misreading of the text**  
  None of the above major points rest on such misreadings; where the paper already acknowledges limitations (e.g., deferring theoretical analysis to future work), the review has treated that as partially mitigating rather than as an unaddressed omission.

## Novel Insights

The genuinely novel and potentially impactful insight here is *not* the use of quantum hardware per se, but rather the recognition that PEFT parameters for massive layers can be profitably reparametrized via a compact generator that enforces strong parameter sharing at the granularity of “chunks” across the layer, while still allowing the induced PEFT module to act at full dimension. Casting this generator as a QNN+MLP hybrid opens a design space where the quantum component could, in principle, serve as a compact source of structured randomness or nonlinearity, while the MLP maps basis indices and probabilities into parameter blocks. Even if future experiments show that a purely classical hypernetwork can match current QPA results, the framework here helps crystallize a general “parameter-generation” view of PEFT that could influence both quantum and classical compression methods.

If the authors redesign and reframe QPA explicitly as a quantum-agnostic hypernetwork framework with interchangeable quantum or classical generators, and provide strong baselines for both, that could be a compelling direction.

## Suggestions

- **Add strong classical baselines and ablations.**  
  At minimum:
  - Replace the PQC with a learnable embedding or small MLP over the basis index (keeping the mapping MLP unchanged) and compare performance/parameter counts.
  - Add an ablation where probabilities used as features are replaced by fixed random numbers or a simple deterministic function of the index.  
  These experiments are crucial to assess whether the quantum component is doing anything beyond what a classical hypernetwork would.

- **Clarify and correct the scaling claims.**  
  Explicitly derive the parameter count of the mapping model \(\tilde{G}_b\) as a function of \(m, n_{\text{mlp}}\), and hidden sizes, and be clear that only the quantum parameters enjoy polylogarithmic scaling. Rephrase contributions and discussion to avoid implying that the *entire* system’s parameters scale polylogarithmically in \(m\) in the configurations actually used.

- **Tone down or refine claims of practicality and quantum advantage.**  
  Frame the current work as a **proof-of-concept** in a restricted last-layer WikiText‑2 setting, and avoid suggesting that QPA is already a broadly practical or superior solution for general LLM fine-tuning. Clearly distinguish between:
  - (i) a conceptual quantum-centric architecture, and  
  - (ii) empirical evidence of superiority over classical PEFT methods.

- **Broaden and strengthen the empirical evaluation.**  
  Where resources permit:
  - Extend QPA-LoRA (and perhaps QPA-DoRA) to multi-layer adaptation (e.g., attention and MLP sublayers) on at least one LLM and dataset.
  - Report multiple seeds for key configurations and show means and standard deviations.
  - Include at least one downstream task beyond plain language modeling.

- **Provide a basic cost analysis.**  
  Report training time per step and approximate FLOPs or memory for baseline PEFT vs QPA on the same hardware and batch size. Even a rough comparison would clarify whether parameter reductions come at a tolerable or excessive computational overhead.

- **Clarify the DoRA results and other surprising baselines.**  
  Some perplexity numbers (e.g., DoRA giving PPL≈5 vs LoRA≈1.6 on GPT‑2) are surprisingly poor compared to expectations from the literature, suggesting configuration or implementation issues. Investigating and, if necessary, correcting or clearly explaining such discrepancies is important for credibility.

- **Consider repositioning the contribution if classical baselines are competitive.**  
  If, after adding classical hypernetwork baselines, the quantum component provides little or no benefit, the work may be better framed as introducing a new family of hypernetwork-style PEFT parametrizations, with an optional quantum instantiation, rather than as a quantum-advantage claim.

## Score and Decision

### Calibration process

I compared this paper against:

1. **Quantum-PEFT (dgR6i4TSng, scores 6,6,6,6, Accept Poster):**  
   - Similar domain (quantum-inspired PEFT for large models).  
   - Quantum-PEFT has clearer and more defensible mathematical scaling (unitary parameterization with well-specified parameter counts), broader experiments across multiple tasks, and no ambiguity about whether a classical equivalent would trivially subsume it.  
   - Relative to Quantum-PEFT, the current QPA paper has a weaker empirical case (single dataset, single-layer tuning) and lacks critical classical baselines.

2. **Quantum Neural Fields (gnexAe3kjx, scores 6,1,8,5, overall Reject):**  
   - Also a hybrid quantum–classical model evaluated only on simulators, with questions about whether the classical submodule does most of the work and about practicality on real hardware.  
   - QNF-Net was ultimately rejected largely because of unclear advantages, missing cost analysis, and over-claiming quantum benefits. QPA shares many of these concerns, though it is somewhat clearer and more focused in exposition.

Positioning QPA relative to these:

- It is **clearly weaker than Quantum-PEFT**, which earned mid-range “accept” scores on the strength of both theory and experiments.
- It is roughly on par with or somewhat weaker than QNF-Net in terms of empirical support for a distinct quantum benefit: QPA is clearer and more focused, but its core missing baseline (purely classical generator) is particularly damaging for its central claim.

Given that, a score in the **4–5** range (borderline to weak accept) would overstate the current strength, especially given the centrality of the missing classical baselines. I view the conceptual idea as interesting but the quantum-specific claims as unsubstantiated. This aligns more closely with the **3–4** range often used for technically sound but unconvincing or significantly incomplete works.

Balancing that the method is coherent, results are nontrivial, and writing is clear (so this is not a 2 or below), I judge:

- **Score: 3.5 (clear reject, but with a potentially promising direction for a substantially revised paper).**

MY FINAL SCORE: <pineapple>3.5</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>