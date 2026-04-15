Now let me search for calibration papers to compare this work against.Now I have enough information to write the final review. Let me synthesize everything.Now I have everything I need to write the consolidated review.

---

## Summary

The paper proposes bilinear MLPs — Gated Linear Units without element-wise nonlinearities — as an analytically tractable alternative to standard MLP variants, and shows that their computations can be exactly represented as a third-order tensor. By applying eigendecomposition to output-conditioned interaction matrices derived from this tensor, the authors reveal interpretable, low-rank structure across toy tasks, image classification (MNIST/Fashion-MNIST), and small language models (TinyStories, FineWeb). Key applications include adversarial example construction from weights alone, detection of overfitting via eigenvector visualization, and identification of a sentiment-negation circuit in a 6-layer transformer. The paper's strongest claim — that this enables "weight-based interpretability viable for large language models" — is partially oversold relative to the actual evidence.

---

## Strengths

- **Clean and original mathematical framework.** The observation that bilinear MLPs admit an exact third-order tensor representation, that only the symmetric part of interaction matrices contributes, and that eigendecomposition of the symmetrized matrix yields orthonormal input directions with no cross-interactions is genuinely principled and elegant. The spectral theorem guarantees real eigenvalues and orthogonal eigenvectors, giving a rigorous foundation that activation-based interpretability lacks.

- **Ground-truth validation (Sec. 4.3).** The reverse-engineering experiment on the MNIST similarity-to-target classifier is the paper's best evidence: the decomposition recovers exactly one large positive eigenvalue, and the corresponding eigenvector is the exact cosine-similarity detector the model was trained on. This is not just a visually plausible result but a mathematically verifiable one, and the comparison with prior work that required dataset access and external hints is compelling.

- **Consistency and truncation results (Sec. 4.2).** Cosine similarities of top eigenvectors across training runs (0.8–0.9), and the near-identical classification accuracy after truncating all but the top few eigenvectors, provide quantitative (not merely anecdotal) evidence for low-rank structure and cross-run stability. This goes beyond qualitative interpretation.

- **Low-rank approximation results in language models (Sec. 5.2).** The finding that most SAE output features across three model sizes (TinyStories, FineWeb-small, FineWeb-medium) achieve correlation ≥ 0.75 with just two eigenvectors is substantive, and the scatter plots in Figure 9C show the approximation capturing tail dependence well.

- **Practical downstream demonstration.** The adversarial mask construction (Sec. 4.4) — crafted without any training or forward passes — provides causal evidence that the extracted eigenvectors genuinely control model behavior, not just that they look interpretable.

- **Scope diversity and reproducibility.** The paper spans toy tasks, image classification, and language modeling with code released publicly.

---

## Weaknesses

### Fatal
*None triggered.* The core mathematical contribution is sound, and the empirical results, while imperfect, are real.

---

### Major

- **Overclaiming: "weight-based interpretability viable even for large language models."**
  The Discussion states: *"The main implication of our work is that weight-based interpretability is viable, even for large language models."* The largest model evaluated is a 16-layer FineWeb transformer. This is a competent research model but falls far short of what "large language model" connotes in 2024/2025 (GPT-4, Llama-3, etc.). The correlation-based evidence in Sec. 5.2 is promising but is measured per-layer and conditioned on feature activation, which is an easier criterion and lacks causal validation. The language model claim should be scoped to "small-to-medium bilinear transformers" or prefaced with explicit scale caveats. As written, the abstract's summary conclusion is stronger than what the experiments establish.

- **Single cherry-picked circuit limits language model evidence.**
  The authors themselves state: *"We cherry-pick and discuss one such circuit"* (Sec. 5.1). A single hand-selected example can demonstrate possibility; it cannot establish reliability. There is no report of how many output features were examined before the sentiment-negation circuit was found, what fraction of circuits yield cleanly interpretable structure, or whether the AND-gate geometry in Figure 8A is typical or exceptional. The broader quantitative result (Sec. 5.2) is correlation-based, not causal or mechanistically validated. Together, these two results support "the method is tractable and sometimes yields meaningful structure," not the stronger "weight-based circuit identification is viable."

- **Dependence on SAEs partially undermines the "weight-based" framing.**
  In the language model section, meaningful output directions are provided by SAEs trained on activation data. The paper's introduction contrasts its approach with "activation-based approaches" as fundamentally limited, but in practice, the most interesting language model results (Sec. 5) are downstream of a data-driven, activation-based step. This is acknowledged in the Limitations, but the framing in the abstract (*"identify small language model circuits directly from the weights alone"*) overstates the dataset-independence. The "from the weights alone" claim holds only for shallow image classifiers where unembedding directions substitute for SAEs.

---

### Minor

- **No systematic quantitative interpretability evaluation.**
  Across the image classification experiments, interpretability is assessed primarily through visual plausibility of eigenvectors. The adversarial mask experiment (Sec. 4.4) provides some causal grounding, but there is no protocol for measuring what fraction of eigenvectors are interpretable vs. polysemantic, nor any human evaluation. The visually appealing examples in Figures 2–4 are convincing but necessarily selective. At minimum, reporting the distribution of approximation quality across all output features (not just summary statistics) would help calibrate how often the method succeeds vs. produces uninterpretable directions.

- **OOD faithfulness of low-rank approximations not assessed.**
  The truncation experiments show that retaining the top few eigenvectors preserves classification accuracy on held-out test data, and the correlation analysis in Sec. 5.2 is conditioned on feature activation. However, there is no analysis of whether the simplified low-rank representation remains faithful *out of distribution*, which is where mechanistic understanding is most needed. The adversarial mask result (Sec. 4.4) shows that the directions are causally relevant, but it does not assess faithfulness of the approximation outside the training data manifold.

- **No comparison to other interpretability methods on the same task.**
  The paper positions bilinear MLPs as superior to gradient-based attribution and transcoders (Sec. 6), but no head-to-head comparison is provided on any task. It is therefore unclear whether the decomposition reveals *more* than, say, integrated gradients applied to an equivalent SwiGLU model. This would not require training new models — analyzing a FineWeb model with gradient-based attribution for the same features would suffice.

---

### Trivial

- **Safety guarantee claim is speculative.** The Discussion says weight-based interpretability "*may also offer better safety guarantees since we could plausibly prove bounds on a layer's outputs*." No theorem, bound, or even sketch is provided. The qualifier "plausibly" is appropriate but this sentence should either be formalized or removed.

- **HOSVD (Sec. 3.3) is minimally demonstrated.** This method is introduced but only demonstrated in Appendix D and does not appear to contribute materially to the main conclusions. It could be shortened or deferred entirely.

---

## Nice-to-Haves

- **Non-cherry-picked circuit inventory.** A systematic sweep of the top-k most-active SAE output features reporting circuit quality (correlation, interpretability assessment) for all of them — including failures — would substantially strengthen the language model claims and allow readers to calibrate trust.

- **Performance-interpretability characterization at larger scale.** A comparison of bilinear vs. SwiGLU at 100M–300M parameters on a standard benchmark would anchor the "drop-in replacement" performance claim in a regime where architectural choices matter more.

- **Failure mode gallery.** A figure showing eigenvectors or circuits where the low-rank approximation fails or produces polysemantic directions would give a realistic picture of the method's scope and limitations.

- **Sparse dictionary learning on the bilinear tensor.** As noted in Sec. 6, the orthogonality constraint from eigendecomposition may limit interpretability when the spectrum is high-rank. A preliminary experiment applying sparse dictionary learning directly to interaction matrices would extend the toolkit and test the Limitations' suggestion.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic: "bilinear MLPs are not a drop-in replacement without main-paper tables."** The paper provides performance comparisons in Appendix I and is explicit in the abstract that the result is "equal loss when keeping training time constant and marginally worse when keeping data constant." Relegating full ablation tables to an appendix is standard practice. The absence of a main-paper table is not an evidential gap — it is a presentation choice. This criticism is too strong as stated.

- **Neutral reviewer: "bivalent eigenvector behavior is a form of polysemanticity the framework doesn't resolve."** The paper explicitly introduces and explains this XOR-like behavior in Figure 2A with a full diagram, and the Limitations section discusses it. This is an acknowledged design property of bilinear layers, not an overlooked flaw. The criticism misrepresents this as something the framework ignores.

- **Human finder: "risk of interpretability illusions from eigenvector patching — dormant pathways."** The paper does not claim that eigenvector interventions are free of all confounds. The adversarial mask experiment is presented as evidence of causal relevance, not as a proof of unique mechanism recovery. This concern is real in general but is not grounded in a specific failure of the paper's experiments.

- **Neutral reviewer / harsh critic: "computational cost / complexity analysis missing."** The paper does not claim large-scale deployability and presents no runtime results for very large models. Requesting a formal FLOP analysis for a framework paper making no scalability guarantees is outside the paper's stated scope and is standard only for systems papers.

---

## Novel Insights

The most genuinely novel observation across all reviews, which the paper itself does not fully foreground, is the practical consequence of the symmetrization result: because only the symmetric part of each interaction matrix contributes to outputs, the effective parameter count for interpretability purposes is substantially smaller than the nominal tensor size, and the spectral theorem guarantees real-valued, orthogonal eigenvectors without additional assumptions. This makes bilinear layers uniquely amenable to weight-level analysis compared to ReLU or SwiGLU variants — not just as a practical convenience, but as a structural property that other GLU variants cannot share without approximation. The finding in Sec. 5.2 that SAE output features are surprisingly low-rank (mean correlation ~0.65 with a single eigenvector, >0.75 with two) across three models of different sizes is a substantive empirical result that, if reproduced at larger scale, would meaningfully support the mechanistic interpretability research program's assumption of underlying simplicity.

---

## Suggestions

1. **Reframe the abstract and discussion claims precisely.** Change "viable even for large language models" to "viable in small-to-medium bilinear transformers up to 16 layers" and revise the "weights alone" claim to acknowledge SAE dependence in deep models. This is a writing fix that does not require new experiments.

2. **Report the full distribution of circuit quality, not just the cherry-picked success.** Sweep all top-100 SAE output features by activation frequency and report the histogram of (a) correlation with 2-eigenvector approximation and (b) manual interpretability rating for a random sample of 20. This transforms the language model section from anecdotal to systematic.

3. **Add a direct causal intervention for the sentiment-negation circuit.** Patch the not-good feature activation and measure downstream sentiment steering on a held-out set of prompts. This would validate the circuit causally rather than just correlationally.

4. **Integrate the HOSVD demonstration into the main paper or remove it.** As a method without a downstream application in the paper's scope, Sec. 3.3 adds surface area without payoff.

---

## Score and Decision

**Calibration:**
- *Sparse Feature Circuits* (Accept Oral, 8/8/8/8): Systematic circuit discovery with causal validation, downstream debiasing application, human evaluation — substantially stronger evidence for interpretability claims.
- *CD-T Circuit Discovery* (Accept Poster, 8/6/5): Novel method, quantitative benchmarks vs. baselines on standard tasks — comparable scope, stronger quantitative grounding.
- *Modular Addition MLP* (Reject, 6/3/8/5): Also analyzes MLP computation with a clean mathematical lens; rejected partly for being limited to a single toy task with no scalability discussion.
- *Interpretability Illusions* (Reject, 3/6/6/8/5): Broader empirical study of simplified representations; rejected partly for narrow scope and limited novelty.
- *Attention SAEs* (Reject, 3/6/6): Applies existing methods to a new layer type with qualitative analysis; rejected for no novel methodology and non-systematic evaluation.

**Position relative to anchors:** This paper is clearly above the "Attention SAEs" reject (it has a novel mathematical framework and causal evidence) and above the "Interpretability Illusions" reject (it has greater novelty and diversity). It is comparable to the "Modular Addition" paper that was rejected — both provide rigorous mathematical analysis of MLP computation with limited scalability — but this paper is broader (image + language) and includes quantitative language model results. It is below the oral *Sparse Feature Circuits* paper but comparable to the *CD-T* accepted poster, differing in that CD-T has quantitative benchmarks while this paper has more conceptual novelty.

The paper makes a real and well-executed contribution: it introduces a mathematically grounded framework for weight-based analysis of bilinear MLPs, validates it across multiple domains including a ground-truth task, and demonstrates practical downstream utility. The main weaknesses — overclaiming in scope, cherry-picked language model circuit, no systematic quantitative interpretability evaluation — are real but do not invalidate the core contribution. These are issues of calibration and evidence, not correctness.

**Final score: 6.0** — Borderline accept (poster). The framework is novel and well-executed for its demonstrated scope; the overclaiming in the framing is addressable without new experiments and should not prevent acceptance given the genuine contributions.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>