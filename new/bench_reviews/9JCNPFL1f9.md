Now I have all the information I need to write a thorough review. Let me now synthesize everything.

Key points to assess:

1. **RetVQA claim**: MIRAGE is trained on 354K RetVQA training data and claims SOTA on RetVQA against zero-shot models. This is a legitimate concern - the comparison is between a model fine-tuned on the benchmark's training data vs. zero-shot models. However, this is somewhat standard practice when no other fine-tuned baselines exist for comparison on that benchmark. The claim should be qualified but isn't necessarily fraudulent.

2. **VHs comparison fairness**: MIRAGE is trained on 1.2M MIQA data that structurally mirrors VHs (multi-image QA with distractors), but tested "zero-shot" on VHs. The other LMMs are also tested zero-shot on VHs but without comparable training. The paper's defense ("single-model architecture") doesn't address the training data asymmetry. However, the comparison is somewhat valid in the sense that MIRAGE is being presented as a solution/baseline for this new benchmark, so showing it works even with MIQA-specific training is useful. The key concern is attributing MIRAGE's VHs performance to architectural innovation (Q-Former, retriever) vs. task-specific training data.

3. **Near-chance performance at 10K**: ~55-60% on binary questions is above chance but barely so. The paper's claim of "scaling to 10K images" is accurate in terms of capability, but the actual utility at that scale is questionable.

4. **VHs benchmark design**: Binary yes/no questions from template patterns is limited but explicitly acknowledged. The benchmark's core contribution - showing that existing visual NIAH benchmarks are saturated - is strong.

5. **Lack of ablation**: No ablation separating the contributions of training data vs. architecture (Q-Former compression, retriever) on VHs performance. This is a real gap.

Let me write the review now.</think>Now I have a thorough understanding of the paper and calibration anchors. Let me write the final review.

## Summary

This paper introduces Visual Haystacks (VHs), a vision-centric needle-in-a-haystack benchmark that tests LMMs' ability to retrieve and reason across large collections of natural images, and MIRAGE, a visual-RAG framework built on LLaVA-v1.5-7B that combines image compression (Q-Former, 576→32 tokens), a query-aware retriever, and a 1.2M multi-image instruction-tuning dataset to process up to 10K images on a single 40GB A100 GPU. The VHs evaluation reveals three key LMM failure modes: susceptibility to visual distractors (~40% accuracy drop from single-image to 100-image settings), difficulty in cross-image reasoning (sharp decline even in oracle multi-needle settings), and positional bias varying by model. MIRAGE achieves the best performance among open-source models on VHs and claims a new SOTA on the RetVQA benchmark.

## Strengths

- **VHs effectively exposes genuine LMM limitations missed by prior benchmarks.** Figure 1(B) provides compelling evidence: Gemini 1.5-Flash achieves near-perfect accuracy on OCR-based visual NIAH but drops to ~15% on VHs, demonstrating that existing benchmarks measure text retrieval rather than visual reasoning and that VHs captures a qualitatively different challenge.

- **Three well-documented empirical findings about LMM pathologies.** The distractor susceptibility (Figure 2: ~40% accuracy drop), cross-image reasoning difficulty (Figure 3A: sharp decline from 1→2 needles even in oracle settings), and positional bias (Figure 4: up to 25% variation) are specific, quantified, and reproducible insights that will be valuable even apart from MIRAGE.

- **The oracle multi-needle experiment cleanly isolates retrieval from reasoning.** By providing only needle images (no distractors), Figure 3(A) demonstrates a fundamental limitation in cross-image reasoning that cannot be attributed to retrieval failure alone.

- **MIRAGE's scalability is an engineering milestone.** It is the only framework that can process 10K images in a single query, while GPT-4o and Gemini v1.5 Pro are limited to ~100 images and open-source models like Phi-3 run out of memory on 4×A100 GPUs at 100 images. This practical capability is independently useful.

- **The 1.2M MIQA training dataset construction addresses real data scarcity.** Converting single-image QA into multi-image format with keyword-clustered distractors and positional shuffling (Section 4.2, Figure 5B) is a reasonable approach to building training data in a domain where public datasets are limited.

## Weaknesses

### Fatal
None.

### Major

- **Unfair RetVQA comparison undermines the SOTA claim.** Table 1 claims MIRAGE achieves "state-of-the-art" on RetVQA (67.6), vastly outperforming GPT-4o (34.6), Gemini v1.5 Pro (32.2), and LLaVA-v1.5-7B (30.6). However, MIRAGE is explicitly trained on 354K samples from the RetVQA training set (Figure 5B, Section 4.2), while the baselines are evaluated zero-shot. Comparing a model fine-tuned on a benchmark's training data against zero-shot baselines and declaring SOTA is misleading. The paper should either compare against other models that are also fine-tuned on RetVQA, or clearly qualify this claim as a comparison between fine-tuned and zero-shot models—not as an architectural or methodological superiority result. The claim in the abstract that MIRAGE "sets a new state-of-the-art on the RetVQA multi-image QA benchmark" is technically true but substantially misleading in its framing.

- **The VHs comparison conflates architectural innovation with task-specific training data advantage.** MIRAGE is trained on 1.2M MIQA samples including synthetic multi-image QA with distractors that structurally mirror VHs questions (binary yes/no over image collections with target/anchor objects). While the paper states "All baselines and MIRAGE are tested in an identical, zero-shot setting" (Section 5), this framing obscures a critical asymmetry: MIRAGE has seen structurally similar multi-image QA data during training, while the baseline LMMs have not. The paper defends this as comparing "single-model architectures," but the relevant question is whether MIRAGE's performance comes from the Q-Former compression, the retriever, the training data, or some combination. Without an ablation that controls for training data (e.g., fine-tuning a baseline LLaVA on the same 1.2M MIQA data without the Q-Former and retriever), it is impossible to attribute MIRAGE's VHs improvements to its architectural contributions rather than its training advantage.

- **Near-chance performance at MIRAGE's headline scale undermines the practical contribution.** At 10,000 images—the scale that constitutes MIRAGE's key differentiator—accuracy appears to be approximately 55–60% on binary yes/no questions (50% chance baseline). The abstract's claim of "up to 13% performance improvement" and the framing of MIRAGE as "solving" large-scale MIQA obscures the fact that the model can barely outperform random guessing at the very scale it claims to enable. The paper does not seriously interrogate this limitation or provide analysis of where in the pipeline (retrieval failure vs. LMM failure) performance collapses at scale.

### Minor

- **VHs_small (100 binary questions) has high variance.** For experiments with N>100 images, the paper switches to VHs_small with only 100 binary questions. With a 50% chance baseline, the 95% confidence interval on accuracy is approximately ±10%, meaning differences between models at large haystack sizes may not be statistically significant. The paper reports bootstrapped averages with standard deviations but does not discuss whether the observed differences at the largest scales are reliable.

- **Binary template-based questions limit benchmark scope.** While the paper acknowledges that "developing a comprehensive MIQA benchmark... is a valuable direction for future research," the VHs questions are all derived from a single template pattern ("For the image with anchor object, is there target object?") with yes/no answers. This tests object co-occurrence retrieval rather than general visual reasoning. This is a reasonable initial diagnostic but limits the conclusions one can draw about broader multi-image reasoning capabilities.

- **Retriever recall drops dramatically at scale, but is not analyzed in depth.** Figure 6(A) suggests top-1 retrieval accuracy falls below 20% at 1000 images for both MIRAGE's retriever and CLIP. The paper notes that MIRAGE's retriever outperforms CLIP but does not analyze what types of queries the retriever succeeds or fails on, or what fraction of downstream errors are attributable to retrieval failure versus LMM reasoning failure.

### Trivial
None.

## Nice-to-Haves

- An ablation fine-tuning a baseline LLaVA-v1.5-7B on the same 1.2M MIQA data (without Q-Former compression or retriever) would directly isolate MIRAGE's architectural contributions from its training data advantage. This is the most impactful missing experiment.

- Confidence intervals or statistical significance tests for the VHs_small experiments at large haystack sizes.

- A VHs variant with non-binary, open-ended questions to test whether models can do more than near-random guessing at scale.

- Analysis of MIRAGE's retriever behavior: what does it select at scale? Is the near-chance performance at 10K images primarily a retrieval failure or an LMM reasoning failure?

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"MIRAGE cannot process beyond 100 images due to API limits for Gemini/GPT-4o"** — The harsh critic claimed this as a "structural" issue, but this is actually the paper's *point*: MIRAGE's scalability IS the contribution here. The fact that closed-source models have API limits is an observation about their limitations, not a weakness of MIRAGE.

- **"The comparison with Koh et al. (2023) is dismissive"** — The paper says "it is unclear how such an approach would scale" about retrieval-tokens. This is a reasonable characterization of a method designed for single-image retrieval being applied to multi-image QA. It's not dismissive—it's a genuine question about scalability.

- **"Runtime comparison showing MIRAGE slower than CLIP-LLaVA undermines efficiency claim"** — The efficiency claim is specifically about being able to scale to 10K images, which no other model can do. The relevant comparison is MIRAGE vs. caption-based methods, which it outperforms on runtime at scale. Being slower than CLIP-LLaVA at small scales does not undermine the efficiency claim at the scale that matters.

- **"Missing confidence intervals for VHs_small"** partially kept as a minor concern but the harsh critic's framing that it "invalidates" results is overstated. With 100 binary questions at ~60% accuracy, the 95% CI is ±10%, which is wide but not disqualifying—it just means small differences at those scales should not be over-claimed. Kept as a minor issue only.

- **Strengths about "end-to-end trained single-model architecture making comparison fair"** — This is from the strength finder, but it conflicts with the verified major weakness about training data asymmetry. Removed as a strength since the "fairness" argument doesn't withstand scrutiny while MIRAGE has task-specific training data the baselines lack.

## Novel Insights

The paper's most novel insight is the systematic characterization of distinct positional bias profiles across different LMM families on visual tasks (Figure 4): Gemini favors early positions, GPT-4o exhibits a lost-in-the-middle pattern, and open-source models favor late positions. This mirrors findings in text NIAH but with qualitatively different bias patterns per model family, suggesting these biases are architectural rather than task-specific. Additionally, the oracle multi-needle experiment (Figure 3A) reveals that the transition from 1→2 needle images causes a sharp performance drop even when retrieval is removed, suggesting that current LMMs have a fundamental limitation in cross-image integration that goes beyond retrieval challenges—this is a finding with implications for future model design directions.

## Suggestions

- Qualify the RetVQA SOTA claim explicitly: state that MIRAGE is fine-tuned on RetVQA training data while all reported baselines are zero-shot, and that the comparison therefore demonstrates general multi-image fine-tuning benefits rather than architectural superiority.
- Add a training-data-controlled ablation: fine-tune a baseline LLaVA-v1.5-7B on the same 1.2M MIQA data without Q-Former compression or the retriever, and evaluate on VHs. This isolates the architectural contribution and makes the narrative much stronger.
- Report 95% confidence intervals for VHs_small results (N=100 binary questions) and discuss which differences at large haystack sizes are statistically significant.
- Add explicit discussion of the near-chance performance at 10K images: decompose errors into retrieval failure vs. reasoning failure to clarify whether the bottleneck is the retriever or the LMM.

## Score and Decision

**Calibration anchors compared:**

| Paper | Score | Comparison |
|-------|-------|------------|
| MMIE (Oral, benchmark + evaluation method) | 8.0 | Stronger benchmark scope + automated metric; VHs has narrower but more diagnostic scope |
| KiVA (Poster, benchmark + novel developmental comparison) | 7.0 | Similar benchmark-diagnostic contribution; VHs findings about positional bias are similarly novel |
| Sparkles (Reject, trained model + dataset + benchmark) | 5.5 | Similar pattern: trained model compared vs. zero-shot baselines; VHs is stronger as a benchmark contribution |
| FairlyUncertain (Reject, benchmark + method, fairness comparison concerns) | 5.6 | Similar dual benchmark+method paper with comparison fairness issues |
| InstructCIR (Reject, zero-shot CIR, trained vs. zero-shot comparison concerns) | 4.25 | Unfair comparison between trained and zero-shot models |
| Visual Data-Type Identification (Poster, benchmark + evaluation) | 8.0 | Pure benchmark paper with novel findings, no method conflicts |

The VHs benchmark contribution is genuinely strong: it exposes real LMM pathologies with well-designed experiments and novel findings (distractor susceptibility, cross-image reasoning difficulty, per-model positional bias). This alone would merit a solid score. However, the MIRAGE method contribution is weakened by the RetVQA SOTA claim's misleading comparison and the inability to disentangle training data advantage from architectural innovation. The near-chance performance at the headline 10K scale is also a real limitation that is insufficiently discussed. The paper is most similar to Sparkles (benchmark + model, trained vs. zero-shot comparison issues, score 5.5) but has a stronger benchmark contribution. It's weaker than pure benchmark papers like KiVA (7.0) and Visual Data-Type Identification (8.0) because the method claims are undercut by the fairness issues. Given that the benchmark findings are valuable and independently reproducible, while the method claims are partially undermined but still represent a useful engineering milestone, I place this between the Sparkles low-end (5.5) and the pure benchmark papers (7.0-8.0), landing closer to the benchmark end given the diagnostic value of the findings.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>