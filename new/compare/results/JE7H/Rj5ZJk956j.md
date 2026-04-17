---
job_id: 4150e0a6-30d9-45ec-a6c4-28bfa14428a0
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: Rj5ZJk956j.pdf
paper: Weakening Neurons: A Newly Discovered Read-Write Functionality in Transformers with Outsize Influence
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies mechanistic interpretability and read-write behavior of neurons in transformer LLMs, which fits squarely under "visualization or interpretation of learned representations" and general machine learning.

## Minimum Quality
Pass ✅.  
All key sections (Abstract, Introduction, Related Work, Method / Approach, Experiments, Results / Analysis, Conclusion) are present and written in clear English. The work is technically coherent and empirically supported across several models, with no obvious fatal flaws or test leakage.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden instructions or attempts to manipulate automated reviewing. The only meta-content is standard appendix material and methodological discussion.

---

# Expected Review Outcome:

## Summary

The paper proposes a weight-based method to characterize "read-write" (RW) functionality of gated MLP neurons in transformers, using cosine similarities between gate, input, and output weight vectors. From this, the authors define a taxonomy of neuron RW types (strengthening, weakening, conditional variants, proportional change, orthogonal output) and empirically analyze their prevalence across 12 SwiGLU/GEGLU language models. They identify a small class of "weakening neurons" that mostly appear in late layers, activate unusually often, and whose ablation strongly affects entropy and attribute-rate metrics; conditional ablation further suggests that rare negative gate activations contribute significantly to sharpening the output distribution.

## Strengths

1. **Clear and simple RW-weight framework, with useful taxonomy.**  
   The formulation around Eq. (1)–(2) and Table 1 is clean: by decomposing each MLP neuron into three weight vectors \(\bm{w}_{\text{gate}},\bm{w}_{\text{in}},\bm{w}_{\text{out}}\), then examining the three pairwise cosines, the paper arrives at intuitive RW roles like strengthening (\(\cos(\bm{w}_{\text{in}},\bm{w}_{\text{out}}) \approx +1\)), weakening (\(\approx -1\)), and conditional variants. Table 1 succinctly encodes this taxonomy and serves as a reusable scaffold for future neuron-level studies in gated MLPs.

2. **Broad empirical survey across many mainstream LLMs.**  
   The paper is unusually thorough in cross-model coverage: 12 models from several families (Llama 2/3, Gemma, OLMo, Mistral, Qwen, Yi) are analyzed at the weight level. Figure 1(a) and Figure 40 show that the median \(\cos(\bm{w}_{\text{in}},\bm{w}_{\text{out}})\) is positive in early-middle layers and becomes slightly negative in late layers across essentially all models, which strongly supports the claimed "strengthening-then-weakening" trend instead of being a single-model artifact.

3. **Identification of weakening neurons as rare but influential.**  
   Despite being a tiny subset of neurons, weakening neurons are shown to (i) concentrate in late layers (Figure 1(b), Figure 41–42), (ii) activate far more frequently than strengthening neurons (Figure 4 and Table 5), and (iii) cause large shifts in attribute rate and entropy when ablated (Figures 3(a), 3(b), 21, 29). The combination of "small in number but high activation and impact" is interesting and practically relevant for interpretability and potentially model editing.

4. **Conditional ablation methodology and negative-gate insight.**  
   The conditional ablation scheme in Section 6.2, which groups activations by the sign pattern of \((x_{\text{gate}}, x_{\text{in}}, x_{\text{post}})\), is a neat idea that goes beyond standard zero/mean ablation. Figure 3(b) is particularly informative: the "gate−_post+" condition (case (iii)) reproduces most of the entropy-sharpening effect of ablating weakening neurons as a whole, while the other sign patterns have much weaker effects. This provides concrete, activation-level evidence that negative gate values in Swish, which are often dismissed as training-only artifacts, play a direct role in inference-time mechanisms.

5. **Thoughtful baselines and randomness analysis.**  
   Section 4.3 and Appendix E construct two complementary baselines for significance of cosines: a high-dimensional Gaussian baseline (with explicit link to Vershynin) and a "mismatched cosine" baseline that partially controls for shared outlier directions. Figure 6 (random OLMo checkpoint) and the 95% bands in Figure 2 support the claim that many neurons have weight cosines far beyond what random geometry would induce.

6. **Rich qualitative analysis and case studies.**  
   The paper does not stop at aggregate statistics. Section 8 and Appendix I examine specific neurons (e.g., weakening neuron 31.9634 and strengthening neuron 28.4737 in Table 2 and Table 3), combining RW-weight interpretation, unembedding projections, and high-activation text snippets. This makes the abstract RW taxonomy feel grounded and also reveals more subtle behaviors like "double checking" in Section H, where \(\bm{w}_{\text{gate}}\) and \(\bm{w}_{\text{in}}\) are almost orthogonal yet semantically aligned in vocabulary space.

7. **Figures generally match and support the narrative.**  
   Figure 2 and its family (Figures 45–56) are particularly effective: each neuron is plotted by \(\cos(\bm{w}_{\text{gate}},\bm{w}_{\text{out}})\) and \(\cos(\bm{w}_{\text{in}},\bm{w}_{\text{out}})\), colored by \(\cos(\bm{w}_{\text{gate}},\bm{w}_{\text{in}})\), with overlaid 95% random regions. The emergence of clear strengthening clusters in layers 7–11 and a distinct weakening cluster in late layers (e.g., bottom-left in Layer 27 of Llama-3.2-3B) visually corroborates the textual claims on RW-category distributions.

## Weaknesses

1. **RW-class boundaries are heuristic, and sensitivity is not explored.**  
   The choice of \(\tau = 0.5\) to threshold cosines and map neurons to prototypical categories is entirely heuristic; the authors acknowledge this in Section 4.2 but do not meaningfully justify or examine robustness. Results in Figure 1(b) and Figure 41–42 depend on this discretization. Since cosine magnitudes are continuous and the random baseline range is only about \([-0.03, 0.03]\), it is plausible that choosing \(\tau = 0.3\) or \(\tau = 0.7\) might (a) change the relative fractions of conditional strengthening vs proportional change vs weakening neurons, and (b) alter where "weakening neurons" are deemed to exist in quantity. The paper should at least show that key qualitative conclusions (strengthening in early layers, weakening in late layers, and relative rarity of weakening overall) are stable across reasonable \(\tau\) values; as written, it asks the reader to trust a hand-picked discretization.

2. **The "outsize influence" claim is only weakly quantified and somewhat cherry-picked.**  
   The title asserts that weakening neurons have "outsize influence", but the evidence is confined to OLMo-7B-0424, a single model, and to specific metrics. For attribute rate (Figure 3(a)), the effect of zero-ablating 243 weakening neurons is visible but modest: the curve diverges from the baseline only from about layer 10 onward and by a few percentage points, and there is no clear analysis of variance or statistical confidence. For entropy (Figure 3(b)), the histograms show that weakening-neuron ablation often decreases entropy by about 10 nats, but these are aggregated across \(\approx 10^5\) predictions and not normalized per token or compared to the overall entropy scale. Moreover, Figures 21–24 and 29–32 show that for other metrics (loss, rank, scale) and for mean ablation, the picture is more nuanced: sometimes weakening neurons decrease entropy, sometimes increase it, and other classes (e.g., strengthening) can also have sizeable effects. The narrative over-emphasizes the most dramatic-looking histograms without a more systematic cross-class, cross-metric comparison.

3. **Single-model ablations, limited data, and lack of statistical rigor.**  
   Section 6 confines ablation experiments to OLMo-7B, even though Section 5 analyzes 12 models. While compute is a constraint, this limits the strength of claims about universality of weakening-neuron influence. In addition, the Dolma subset is 20M tokens, but ablation analysis is largely descriptive: there are no confidence intervals, no bootstrapped standard errors, and no formal hypothesis tests. For example, in Figure 3(a), it is unclear whether the difference between "clean" and "weakening243" is statistically significant at each layer, and whether effect sizes depend strongly on the specific random Dolma slice used. The same applies to entropy histograms in Figure 3(b) and Figures 17–24 and 29–39; these plots could easily be driven by a few extreme cases, but the paper does not quantify dispersion beyond log-scale counts.

4. **Causality vs correlation in the entropy-sharpening story is underdeveloped.**  
   The conditional ablation analysis in Figure 3(b) is intriguing, but the causal story "negative gate values of weakening neurons sharpen the distribution" remains partially speculative. First, the frequency of case (iii) activations is not quantified relative to all weakening activations per layer; Section 7 remarks that these negative-gate events are "relatively rare", yet they apparently explain most of the entropy effect. Without presenting, for example, the joint distribution of activation counts and entropy contribution per case, it is difficult to assess whether case (iii) is truly the dominant mechanism or just produces rare but huge changes. Second, the case study on "Omicron" (Section 6.3) is anecdotal and does not generalize. A more compelling argument would show, for many tokens, that case-(iii) activations systematically boost the correct next token logits or reduce entropy conditioned on high confidence, as opposed to cherry-picked examples.

5. **Weight preprocessing may bias some analyses, and alternatives are not examined.**  
   The preprocessing step in Section 3.2 and Appendix C flips \(\bm{w}_{\text{in}}\) and \(\bm{w}_{\text{out}}\) by \(\mathrm{sign}(\cos(\bm{w}_{\text{gate}},\bm{w}_{\text{in}}))\). While behavior is unchanged due to the symmetry in Eq. (2), this transformation is not innocuous for the RW classification: it systematically forces \(\cos(\bm{w}_{\text{gate}},\bm{w}_{\text{in}}) \ge 0\), and thereby shifts neurons across quadrants in Figure 2 and related figures. In particular, the statistics that involve \(|\cos(\bm{w}_{\text{gate}},\bm{w}_{\text{out}})|\) and \(|\cos(\bm{w}_{\text{gate}},\bm{w}_{\text{in}})|\) (e.g., activation frequency vs cos plots in Figure 8–9) are affected. The appendix shows one qualitative comparison without preprocessing (Figure 5), but does not perform the core analyses (e.g., RW-class histograms in Figure 1(b) or activation-frequency correlations in Figure 4) on unprocessed weights. It remains unclear how many neurons fundamentally change their assigned RW category under different legitimate sign conventions.

6. **Mathematical characterization of "double checking" is mostly qualitative.**  
   Section H argues that "double checking" occurs when \(\bm{w}_{\text{gate}}\) and \(\bm{w}_{\text{in}}\) are nearly orthogonal but share the same top unembedding neighbors. The toy example is reasonable, and the concept is interesting, but the paper does not provide quantitative criteria such as "we consider a neuron double-checking if \(\cos(\bm{w}_{\text{gate}},\bm{w}_{\text{in}}) \in [-0.1,0.1]\) and the Jaccard overlap between the top-k projected tokens exceeds a threshold". Nor is there a count or distribution across layers for such neurons. As a result, "double checking" remains a plausible anecdote rather than an established pattern, despite its prominent role in some case studies and in the interpretation of conditional strengthening.

7. **Missing closely related work from the same authors, and weak positioning vs. prior IO-analysis.**  
   The Related Work section cites Gurnee et al. (2024) and Elhage et al. (2021) as prior glimpses of input-output cosine analysis, but it omits very close work by the same authors that, according to the abstract, already introduces IO-based analysis of gated neurons (see Potentially Missing Related Work below). Not citing or contrasting against such work makes it hard to judge what is truly new in the present submission (e.g., is the taxonomy new, the cross-model survey, the weakening-neuron discovery, or the conditional ablation methodology?). More generally, some of the narrative ("we are the first to investigate read-write behavior of gated neurons using cosine similarities") is overstated in light of existing IO-cosine analyses and SVD-based feature studies.

8. **Limited connection to higher-level model behavior and downstream tasks.**  
   Almost all experiments focus on internal metrics (attribute rate, entropy, rank, hidden-state scale) on generic Dolma text. There is little exploration of how weakening neurons influence actual task performance (e.g., language modeling loss, QA accuracy, factual recall benchmarks) or phenomena like hallucination or calibration. Figures 18, 22, 26, 30 hint that ablations change loss, but these are not deeply discussed. Without a clearer link from neuron RW behavior to externally observable behavior, the impact of these findings for the broader ICLR audience is somewhat constrained to the mechanistic-interpretability niche.

9. **Some figure and table interpretations remain at the surface.**  
   For instance, Figure 7 and its full-layer extension in img-8/img-9 reveal that the negative correlation between activation frequency and \(\cos(\bm{w}_{\text{in}},\bm{w}_{\text{out}})\) weakens or even becomes slightly positive in the last layers, yet this is only briefly mentioned in Appendix G. Similarly, Table 5 shows that orthogonal-output and proportional-change neurons also activate quite often (0.37 and 0.37, respectively), not far behind weakening neurons, but the text focuses almost exclusively on weakening vs strengthening. Some more balanced discussion of what high-activation orthogonal-output neurons are doing would make the story less one-dimensional.

## Potentially Missing Related Work

1. **Gerstner & Schütze, "Understanding Gated Neurons in Transformers from Their Input-Output Functionality" (2025).**  
   This work appears to analyze gated neurons specifically via input-output cosine similarities, which is central to the present paper. It should be discussed in Section 2 and at the start of Section 4, with a clear explanation of what is new here (e.g., expanded taxonomy including conditional variants, multi-model survey, discovery of weakening neurons and negative-gate mechanisms, or conditional ablations).

2. **Gerstner & Schütze, "GLUScope: A Tool for Analyzing GLU Neurons in Transformer Language Models" (2026).**  
   GLUScope reportedly offers tooling for inspecting GLU neurons (SwiGLU, GEGLU) and likely implements some of the analyses used here. It should be cited alongside TransformerLens in Section 3.2 or Section 4 and contrasted in Related Work: is this paper introducing new analytical concepts beyond what GLUScope provides as a tool?

3. **Lermen, "SVD on Decision Transformers" (2023).**  
   Applies SVD and cosine-similarity analysis of weight vectors in transformer-like architectures. While not focused on GLUs, it is conceptually close to the idea that directions in weight space reveal interpretable mechanisms. It would be useful to mention in Section 2 as another instance of weight-based interpretability in transformers and to clarify how the present per-neuron RW analysis compares to SVD-based global decompositions.

4. **Mongaras et al., "Cottention: Linear Transformers With Cosine Attention" (2024).**  
   Although primarily about attention mechanisms, this work leverages cosine similarity within transformer layers. It might not be central, but a brief mention in Related Work on cosine-based mechanisms in transformer components (alongside Gurnee et al. 2024) would help contextualize why cosine geometry is a natural tool for RW analysis.

## Questions

1. **Sensitivity to threshold \(\tau\):**  
   Can you provide results that show how the distribution of RW classes by layer (analogous to Figure 1(b) and Figure 41–42) changes when the cosine threshold is varied, say \(\tau \in \{0.3, 0.4, 0.6, 0.7\}\)? In particular, do weakening neurons still predominantly appear in late layers and remain a small fraction under all these thresholds?

2. **Quantitative role of negative-gate activations:**  
   For weakening neurons in OLMo-7B, what fraction of total activations fall into each of the four sign patterns in Section 6.2, per layer? And what is the average entropy change per activation in each case? A breakdown like "case (iii) accounts for X% of activations but Y% of total entropy reduction" would greatly clarify the actual contribution of negative gate values.

3. **Cross-model ablation evidence:**  
   Is it feasible to run a lighter-weight ablation experiment (perhaps on a smaller text sample or fewer layers) on one or two additional models, such as Llama-2-7B or Gemma-2-2B, to verify that weakening-neuron ablation has qualitatively similar effects on entropy and attribute rate? Even a limited experiment would strengthen claims of universality beyond the weight-level evidence.

4. **Effect sizes relative to baselines:**  
   For Figures 3(a) and 3(b), and for the multi-class ablation plots (Figures 21–24 and 29–32), could you quantify effect sizes more explicitly, for example by reporting mean and standard deviation of entropy or attribute-rate differences across tokens, and showing p-values vs. random-neuron baselines? This would help evaluate whether the observed differences are large in a practical sense, not only visually noticeable.

5. **Formalizing "double checking":**  
   Can you propose and evaluate a concrete criterion for "double checking" neurons based on \(\cos(\bm{w}_{\text{gate}},\bm{w}_{\text{in}})\) and overlap of top-k unembedding projections (e.g., Jaccard similarity)? It would be particularly helpful to see how many such neurons exist per layer and whether they cluster in conditional-strengthening / conditional-weakening classes as suggested.

6. **Downstream behavioral impact:**  
   Have you examined whether ablating weakening neurons affects language modeling loss or accuracy on standard benchmarks (e.g., WikiText, some QA dataset), not just entropy/attribute-rate proxies? If not, could you comment on whether you expect these neurons to primarily regulate calibration vs. correctness?

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The methodology is conceptually straightforward and largely sound, and the empirical findings are consistent across many models. However, some quantitative claims (e.g., "outsize influence", primacy of negative gate values) would benefit from more rigorous statistical treatment and sensitivity analysis.

## Presentation Rating

3: good.  
The paper is generally well written and structured, with useful figures and detailed appendices. Some over-claims relative to the evidence, missing related work, and a few under-analyzed plots prevent an "excellent" rating.

## Contribution Rating

3: good.  
The RW taxonomy for gated neurons, empirical discovery of strengthening-then-weakening patterns, and the identification of influential weakening neurons and negative-gate mechanisms are all interesting contributions for mechanistic interpretability, though somewhat narrow in scope and not yet tightly linked to downstream ML performance.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The work offers a clean, reusable framework and several nontrivial empirical observations about gated MLP neurons, supported by extensive cross-model weight analysis and reasonably careful ablation on one model. The main weaknesses are heuristic class definitions, limited statistical rigor, and somewhat overstated universality/impact claims. With clarifications and additional analyses as suggested, this would be a solid and useful ICLR paper.

## Reviewer Confidence

4: confident.  
I am familiar with mechanistic interpretability and transformer analysis, checked the main equations and ablation methodology, and compared to related neuron/weight-based work. Some implementation details and dataset construction specifics were not independently verified.