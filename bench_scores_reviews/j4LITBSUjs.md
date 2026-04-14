## Summary
This paper presents two contributions targeting hallucination in Multimodal Large Language Models (MLLMs) for dense captioning: (1) *HalfScore*, a graph-based F1 metric using GPT-4o-extracted semantic triplets to measure both the precision (hallucination rate) and recall (completeness) of dense captions, and (2) *PerturboLLaVA*, a supervised fine-tuning strategy that prepends adversarially crafted text perturbations to training inputs, forcing the model to attend to visual content rather than language priors. The method is experimentally evaluated on LLaVA1.5-7B against VCD, OPERA, and RLAIF-V, showing improved precision on hallucination metrics and, notably, preservation or slight improvement on general multimodal benchmarks.

---

## Strengths

- **Zero inference overhead with genuine training-efficiency advantage.** Unlike decoding-based methods (OPERA: 2–5× inference cost, VCD: 2×) or RLHF-style methods that require reward model training, PerturboLLaVA integrates into standard SFT at 1× inference cost with no additional training stage. Table 1 makes this comparison explicit and honest, including acknowledging the data-generation requirement. This is a genuinely meaningful distinction, not just a claimed one.

- **Preservation and improvement of general multimodal capabilities.** Table 3 shows PerturboLLaVA achieves +1.6 on MMBench, +0.3 on SEED, and +1.2 on CCBench over the LLaVA1.5 baseline, while VCD and RLAIF-V both degrade general performance. The ability to reduce hallucination without the typical capability-safety tradeoff is a specific and non-trivial finding most prior work cannot claim.

- **Complementarity to decoding strategies.** OPERA+Ours achieves the best combined results across nearly all benchmarks in Table 3, demonstrating the method introduces an orthogonal robustness dimension rather than just redistributing the same performance gains. This is a practically useful result for deployment.

- **HalfScore fills a real gap in dense captioning evaluation.** CHAIR measures only object presence; MMHalBench produces a single holistic score. HalfScore decomposes caption quality into precision and recall over structured semantic concepts (objects, attributes, relations), providing actionable diagnostic information. The human correlation study (Pearson 80.7% for precision vs. 71.7% for MMHalBench, Table 6) provides concrete evidence of metric superiority over LLM-based judging for this task.

- **Perturbation relevance ablation.** The random-text vs. targeted-perturbation experiment in Table 5 shows that even random perturbations help, but targeted ones help more — validating the core design intuition and ruling out a simple length- or format-artifact explanation for the gains.

---

## Weaknesses

### Fatal

**Figure 2 contains a major hallucination from PerturboLLaVA itself in the paper's flagship qualitative comparison.** The image is described in both the figure caption and by competing methods (VCD, OPERA, RLAIF-V) as two women playing *tennis* on a court. PerturboLLaVA's output states: *"The image features two women playing **badminton** games on a court… Both women are holding **badminton rackets**…"* This is a clear, substantial factual error — identifying the wrong sport and wrong equipment in a showcase example intended to demonstrate reduced hallucination. The figure's caption in the paper text even says the *PerturboLLaVA output contains more accurate descriptions* and highlights it in a positive color, despite this being demonstrably incorrect. This severely undermines the paper's credibility, as reviewers can verify it directly. The authors must either replace this example or — if badminton is somehow correct — provide explicit justification. As presented, this is the opposite of what the figure claims to demonstrate.

### Major

- **POPE and AMBER are absent from the evaluation, despite being cited in the paper's own related work (Section 2.2) as the standard close-ended hallucination benchmarks.** POPE is the most widely adopted hallucination evaluation in the MLLM community; its omission makes it impossible to fairly compare against the broader literature or verify the generality of the hallucination reduction claims. This is not a nice-to-have — it is a table-stakes requirement for an ICLR hallucination paper. The gap is especially notable given the authors explicitly discuss these benchmarks.

- **Single backbone limits generalizability of the core training claim.** PerturboLLaVA is only validated on LLaVA1.5-7B. The paper repeatedly frames perturbative visual training as a "scalable" and "standard strategy" applicable to MLLMs broadly, but without a single additional architecture (e.g., LLaVA1.5-13B, InternVL2, or Qwen2-VL), the claim is unsubstantiated. Particularly given that modern MLLMs use very different vision encoders and projection architectures, transfer is not guaranteed.

- **HalfScore's sub-component reliability is unvalidated.** The metric's meaningfulness depends entirely on GPT-4o correctly extracting semantic triplets and accurately matching them across captions. The paper provides no analysis of GPT-4o's triplet extraction accuracy against human-annotated triplets, its sensitivity to prompt variation, or its consistency across API calls. A metric where the measurement instrument has unknown error characteristics cannot be relied upon, regardless of downstream correlation with coarse human preference scores.

- **On object-level CHAIR, the method is dramatically outperformed by RLAIF-V (CHAIRs: 36.1 vs. 18.1; CHAIRi: 10.4 vs. 4.7).** The paper acknowledges this and attributes it to RLAIF-V's stronger reward model (LLaVA-Next 34B), which is fair disclosure. However, object hallucination is the most established and well-understood hallucination dimension, and a near-2× disadvantage on this metric needs a more substantive explanation or targeted fix. The paper notes future work to "design targeted perturbation texts for objects," implicitly acknowledging this is a real gap in the current approach.

### Minor

- **Section 4.2's mathematical derivation contains unjustified independence assumptions.** Equation (8) assumes $p(x_{<k}^p | x_k, x_{<k}^{-p}, I) = p(x_{<k}^p | x_k, I)$, asserting conditional independence between prior-influenced and prior-free preceding tokens. In autoregressive generation, these are not separable quantities — all tokens jointly contribute to the generation history. The derivation is presented as a rigorous mathematical justification but is closer to intuitive post-hoc rationalization. The core method is empirically grounded and does not need this section to be valid, but as written, the section may mislead readers about the theoretical guarantees.

- **Text-Table inconsistency in Section 5.3.** The text states: *"increasing perturbation levels…leads to shorter captions and lower recall, suggesting a more cautious model behavior."* However, all three perturbation versions (V1: 46.5, V2: 46.0, V3: 46.1) achieve *higher* recall than the LLaVA1.5 baseline (45.8). The paper is describing the decreasing trend within V1→V2→V3, but this should be stated explicitly rather than implying recall falls below baseline. The confusion matters because recall preservation is a key selling point of the method.

- **Abstract overstates overhead savings.** The abstract says the method works *"without incurring additional computational overhead,"* yet Table 1 explicitly marks ✗ for "No extra data generation" for the proposed method. GPT-4o inference over 160k training samples is real cost that, while cheaper than RLHF-scale pipelines, is not zero and limits reproducibility for researchers without API access. The paper should quantify this cost (reportedly provided in Appendix A.3) and qualify the claim accordingly in the abstract.

- **HallusionBench improvement (+0.6) is described as demonstrating the method "excels in vision reasoning tasks."** The gain over OPERA is +0.4 (47.5 vs. 47.1). Without significance estimates, characterizing this as excelling is an overstatement.

### Tiny

- Naming inconsistency: "HalfScore," "HalFscore," and "Fscore" are used interchangeably across the abstract, body, and tables. For a proposed benchmark metric, consistent naming is important.
- Concept matching under-specification: it is not clear how synonyms ("couch" vs. "sofa"), hypernyms ("dog" vs. "animal"), or partial attribute matches are handled in the graph-matching step. This affects the reproducibility and interpretability of precision/recall computations.

---

## Nice-to-Haves

- A **shuffled-image perturbation baseline** (perturbation text generated for a different image from the same batch, same distribution and length) would more precisely isolate whether the *contextual relevance* of the perturbation is necessary, or whether any out-of-distribution prefix provides sufficient training signal.
- An **open-source LLM alternative** for perturbation generation (e.g., LLaMA-3, Qwen2) to assess how sensitive performance gains are to perturbation quality, which would also improve reproducibility.
- **Attention map or representation-shift visualizations** before and after perturbative training would directly support the mechanistic hypothesis that the model shifts from language-prior reliance to visual attention.
- **Per-category hallucination breakdown analysis** for the method itself: Table 2 shows PerturboLLaVA reduces object and attribute hallucinations more than relation hallucinations, but there is no analysis of whether specific perturbation designs affect different hallucination types differently.
- **Ablation on perturbation data proportion** (e.g., 25%/50%/100% of the 160k samples) to characterize the data-efficiency curve and identify practical deployment thresholds.

---

## Removed Points
*These points are flagged as removed; treat them with caution as they may reflect reviewer error or scope overreach.*

- **Scope mismatch between dense captioning and general MLLM hallucination**: The harsh critic flags a tension between the dense captioning motivation and VQA-based training data. The paper explicitly states both as objectives and evaluates on both; framing this as a logical inconsistency is scope creep.
- **Demand for formal causal identification of the "root cause"**: While the language is overclaiming, this is a framing issue. The method's validity does not depend on a causal proof, and ICLR empirical methods papers are not required to establish causality.
- **Request for confidence intervals and multiple-run statistics**: Single-run evaluation is standard practice for multi-benchmark MLLM evaluation at this scale. Requiring significance testing here imposes a non-standard rigor requirement.
- **RLAIF-V comparison unfairness**: The paper explicitly and preemptively acknowledges that RLAIF-V's reward model (LLaVA-Next 34B) may transfer 34B-level capability, making the comparison asymmetrically favorable to the baseline. Criticizing the comparison as unfair to the proposed method inverts the direction of any unfairness.
- **"Adversarial misuse" of ignoring textual prefixes**: Highly speculative societal risk concern not grounded in the paper's specific mechanism.
- **Missing related works on structured captioning metrics**: Cannot be verified without external sources; removed per review protocol.
- **Conditional independence claim applied to perturbation-at-test-time scenarios**: The perturbation text is only added during training, not at inference. Therefore concerns about models ignoring valid contextual text at test time are not applicable to the method's actual inference procedure.
- **Potential answer leakage from GPT-4o perturbation generation**: GPT-4o is explicitly instructed to construct perturbations without disclosing the answer. While indirect information could theoretically be encoded, this is highly speculative and not demonstrated.

---

## Novel Insights

The observation that perturbative visual training simultaneously improves precision (reducing hallucinations) while *also* improving general multimodal benchmark scores—a typically adversarial tradeoff in hallucination mitigation—is the most interesting empirical finding in this paper and is somewhat underexplored theoretically. One plausible interpretation is that language-prior reliance is not merely a hallucination-specific bug but a general bottleneck on visual understanding: by forcing the model to process conflicting textual signals, the training strengthens the visual pathway's influence on token generation in a way that benefits all visual tasks, not just captioning. The random perturbation ablation partially supports this, suggesting that the improvement comes from the training regime itself (increased visual signal-to-noise demand) rather than solely from learning to reject semantically plausible misleading content. This direction—perturbative augmentation as a general visual representation strengthening technique rather than a hallucination-specific patch—is worth developing explicitly.

---

## Suggestions

1. **Replace or explicitly annotate Figure 2.** If the tennis image is correct, replace the PerturboLLaVA output with a non-hallucinated example. If the image is somehow a badminton match, provide explicit justification. This is the most urgent fix before any submission.
2. **Add POPE and AMBER evaluations.** These are one-run evaluations on standard benchmarks and would substantially strengthen the paper's positioning in the hallucination literature.
3. **Validate at least one additional backbone** (LLaVA1.5-13B is the minimal ask; a different architecture family would be more convincing).
4. **Add GPT-4o triplet extraction accuracy analysis** against a small manually annotated ground truth (50–100 images would be informative) to establish the reliability floor of HalfScore.
5. **Rewrite Section 4.2** as "Intuitive Motivation" rather than a mathematical derivation, or substantially revise the independence assumption argument with citations to related robustness/causal-intervention frameworks.
6. **Fix the Section 5.3 text-table inconsistency** regarding recall trends, and clarify that recall improves over the baseline for all perturbation versions.
7. **Quantify GPT-4o API cost** (approximate tokens and dollar cost for 160k perturbations) in the main text, not just the appendix, to give readers an honest cost picture.

---

## Paper Evaluation

| Axis | Assessment |
|---|---|
| **Originality** | Moderate. The perturbative prefix augmentation idea is intuitive and practically effective, but closely related to adversarial augmentation and contrastive training paradigms. HalfScore is a sensible extension of F1 to dense captioning. Neither contribution is a major conceptual leap, but both are clean and useful. |
| **Importance of research question** | High. MLLM hallucination is a central open problem with direct practical consequences. |
| **Claims well supported** | Partially. The training method has meaningful empirical support on the tested backbone, but the Figure 2 credibility issue, missing POPE/AMBER, and single-backbone constraint leave important claims inadequately supported. |
| **Soundness of experiments** | Moderate. Multi-benchmark evaluation is a strength. However, the absence of the field's canonical benchmarks, single-architecture testing, and the metric's unvalidated sub-components are significant gaps. |
| **Clarity of writing** | Adequate with notable flaws: naming inconsistency for the proposed metric, the text-table contradiction in Section 5.3, and the misleading abstract claim on overhead. |
| **Value to research community** | Moderate-to-high for the training method if the experimental gaps are addressed; limited for HalfScore until the metric's reliability is better characterized. |
| **Contextualization relative to prior work** | Adequate but imprecise. The method's relationship to adversarial/counterfactual augmentation literature and curriculum learning is not discussed, and the evaluation omits standard benchmarks the paper itself cites. |

The paper contains a promising and practically motivated training intervention with a genuinely interesting empirical finding (improved general capabilities alongside hallucination reduction). However, in its current form it falls short of the ICLR bar due to the Figure 2 showcase hallucination, missing standard benchmarks, single-backbone validation, and metric reliability gaps. These are addressable issues rather than fundamental flaws in the approach.