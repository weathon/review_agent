## Summary

This paper introduces Visual Haystacks (VHs), a vision-centric needle-in-a-haystack benchmark for evaluating multi-image question answering (MIQA) across up to 10,000 images, and MIRAGE, a visual-RAG framework using query-aware retrieval and token compression to scale within these limits. The paper identifies three failure modes in existing LMMs—susceptibility to distractors, difficulty with cross-image reasoning, and positional bias—and demonstrates that MIRAGE substantially outperforms open-source LMMs on VHs while maintaining competitive single-image QA. The core contribution is both a diagnostic benchmark revealing real LMM weaknesses and a practical engineering solution.

## Strengths

- **Compelling demonstration that prior visual NIAH benchmarks are saturated**: Figure 1(B) shows Gemini 1.5-Flash achieving perfect accuracy on text-overlay NIAH benchmarks but dropping to 0.15 on VHs—clear evidence that existing benchmarks test OCR rather than genuine visual retrieval (Section 2.1, Figure 1).
- **Systematic diagnostic findings across multiple model families**: The paper identifies and quantifies three failure modes: (1) susceptibility to distractors (Figure 2 shows >40% accuracy drops from N=1 to N=100), (2) impaired multi-image reasoning even in oracle settings (Figure 3A), and (3) positional bias varying by model (Figure 4 heatmaps showing up to 25% variation). These findings are not available from prior benchmarks.
- **MIRAGE scales to 10k images on a single 40GB A100 and sets SOTA on RetVQA**: Table 1 shows MIRAGE achieves 67.6 on RetVQA (vs. 34.6 for GPT-4o), with the Q-Former compression (576→32 tokens) and query-aware retriever (Figure 5) enabling this at a fraction of the compute of alternatives.
- **Query-aware retriever significantly outperforms static CLIP retrieval**: Figure 6(A) demonstrates the co-trained retriever maintains higher recall than CLIP as haystacks grow, validating the design choice of training retrieval jointly with the LMM—a concrete empirical contribution to visual-RAG literature.

## Weaknesses

### Fatal
None. The core claims are directionally valid, though overstated in places.

### Major

- **The VH benchmark's narrow task design does not substantiate the paper's "cross-image reasoning" framing.** The VH query template ("For the image with anchor object, is there target object?") with binary yes/no answers reduces all queries to: (a) retrieve the image containing the anchor object, (b) check for the target object's presence, and (c) compare to a binary ground truth. The multi-needle setting adds only trivial logical quantifiers (`any`/`all`). Section 2.1 confirms this structure. This is single-image object presence verification followed by boolean aggregation—genuine cross-image reasoning would require synthesizing spatial, temporal, or semantic relationships *between* images, or comparing attributes across them. The paper itself acknowledges (Section 5) that MIRAGE degrades on multi-needle tasks and attributes this to training data limitations. The benchmark's structural narrowness means the paper's headline claim of measuring "reasoning across potentially unrelated images" is significantly overstated relative to what the task actually tests.
- **The comparison between MIRAGE and baseline LMMs is methodologically confounded.** MIRAGE is instruction-tuned on 1.2M samples using a dataset explicitly constructed to mirror the VH format (synthetic multi-image QA with distractor images, Section 4.2). Baseline LMMs (GPT-4o, Gemini, Qwen2-VL, etc.) are evaluated zero-shot on VHs (Section 5: "tested in an identical, zero-shot setting"). The paper defends this by claiming MIRAGE is a "single-model architecture" where the retriever is "a component for filtering." However, the retriever is co-trained with ground-truth relevance labels on task-specific data, and the LMM is fine-tuned on VH-format questions. Asymmetry that favors the proposed method in a comparison does not automatically invalidate the paper (per guidance), but the asymmetry here is fundamental: MIRAGE is evaluated with task-specific supervised training while baselines are not. At minimum, the paper should compare against a baseline LMM fine-tuned on the same VH training split to isolate what gains come from data exposure versus the RAG architecture. This gap means MIRAGE's superiority over baselines cannot be confidently attributed to the retrieval architecture alone.

### Minor

- **The retriever's top-1 accuracy discrepancy with downstream QA performance is insufficiently explained.** Figure 6(A) reports MIRAGE's top-1 retrieval accuracy dropping to ~0.05 at 1,000 images. If single-needle retrieval fails at 95%, downstream QA accuracy should theoretically approach random guessing (~50%), yet Figure 2 shows MIRAGE maintaining >55% at N=1,000. The paper does not clarify whether multi-candidate (top-K) retrieval is used at inference, what threshold the sigmoid-based retriever operates at, or whether a different retrieval configuration is used for the 10k evaluation (which lacks any retrieval accuracy report). This gap makes the 10k scalability claim rest partly on extrapolation.

- **Token compression (576→32 tokens) causes measurable degradation on single-image benchmarks with limited justification for the choice of K=32.** Table 1 shows drops on VQAv2 (78.5→76.6), GQA (62.0→59.1), and TextVQA (58.2→56.2) compared to the base LLaVA-v1.5-7B. The authors attribute this to token compression but do not ablate the token count (e.g., 64, 128) or quantify the information loss beyond citing general VQA degradation. Q-Former compression is a direct application of BLIP-2 methodology without novelty.

### Trivial

- **The 50% guessing-rate claim for VHs is asserted without a verification procedure.** Section 2.1 states "We curate the dataset such that guessing or relying on common sense reasoning without viewing the image results in a 50% accuracy rate." Given COCO's skewed object co-occurrence statistics (80 classes), maintaining this balance across 10k-image haystacks with arbitrary anchor-target pairs requires careful control not described. This is minor since the binary task definition inherently has a 50% random baseline, but the claim of active balancing is left unsupported.

## Nice-to-Haves

- **Fine-tuned baseline ablation**: Training a baseline LMM (e.g., LLaVA-v1.5-7B) on VH training data without the RAG components would cleanly separate architectural gains from data-exposure effects.
- **Error analysis for multi-needle tasks**: Breaking down whether failures originate from missed retrieval, incorrect attribute checking, or faulty logical aggregation would strengthen the diagnostic value of the benchmark.
- **Retriever metrics beyond top-1**: Reporting recall@K, precision, and F1 for the retriever across varying N, with an explicit accuracy breakdown at N=10,000, would round out the scalability evaluation.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic: "Caption aggregation baseline introduces compounding OCR/captioning errors that cap its performance"** — The paper already acknowledges the caption aggregation baseline is "query-unaware" and "sub-optimal" (Section 2.1, Section 3.1). The critic raises a valid observation, but the authors themselves characterize this baseline as inferior and use it as a lower-bound comparison point. This is not a paper error but an acknowledged limitation the authors discuss.

- **Harsh Critic: "Synthetic data generation lacks concrete criteria for 'unrelated,' raising concerns about semantic leakage"** — Section 4.2 describes "keyword-based filtering [to] cluster similar questions before adding distractors" followed by "random sampling from unrelated subsets." The synthetic data pipeline is described in sufficient detail for a methodology paper. Concerns about "subtle semantic leakage" are speculative without evidence of actual contamination.

- **Harsh Critic: "Heavy class weighting (5.0) for positive samples in retriever training"** — The paper explicitly states (Section 4.2) this weight is "to address data imbalance and prioritize recall." This is a standard technique for imbalanced binary classification and not a flaw.

- **Harsh Critic: "Fails to engage with recent multi-image retrieval baselines like ImageBind"** — While ImageBind is a valid omission, the paper does compare against CLIP retrieval (Figure 6) and discusses related retrieval/RAG work in Section 6. The absence of a specific retrieval method from related work is minor.

- **Strength Finder: "Caption aggregation baseline revealing LLM robustness to irrelevant text"** — This is a minor observation embedded in a broader point about distractor susceptibility, not a standalone contribution of the paper.

- **Harsh Critic: "Benchmark design inherently precludes evaluation of cross-image reasoning"** — While this has truth to it, the paper's own framing of the benchmark is as a NIAH-style *retrieval* test ("find the needle, then answer a question"), not as a sophisticated reasoning benchmark. The critique partially applies (overclaiming in the "reasoning" framing) but calling the benchmark "structurally incapable" is too strong: it does test retrieval under distractor noise and positional bias effectively.

## Novel Insights

The paper's most distinctive contribution is demonstrating that the "lost-in-the-middle" phenomenon—previously documented only in text-based long-context LLMs—transfers to the vision domain, with model-specific patterns (Gemini favoring early positions, GPT-4o showing mid-context weakness, open-source models favoring positions nearest the query). This extends a well-known LLM diagnostic to multimodal architectures. Additionally, the finding that a caption aggregation baseline (an LLM processing irrelevant text) can match or surpass LMMs (processing irrelevant images) at ~20+ images is a genuinely interesting observation suggesting LMMs are currently less robust to visual distraction than LLMs are to textual distraction—a gap worth investigating.

## Suggestions

1. **Reframe the VH benchmark as a "visual retrieval and presence verification" benchmark rather than a "cross-image reasoning" benchmark.** The current framing overclaims what the task tests. The binary yes/no template measures whether models can (a) find relevant images amid distractors and (b) detect object presence—not relational reasoning across images. Reframing would improve accuracy without diminishing the contribution.
2. **Add a fine-tuned baseline**: Report a baseline LLM (LLaVA-v1.5-7B or similar) fine-tuned on the VH training split to isolate whether MIRAGE's gains come from the RAG architecture or task-specific data exposure.
3. **Clarify retrieval configuration at large haystack sizes**: Specify whether the MIRAGE retriever uses top-K or threshold-based retrieval at inference, what K or threshold value is used, and report retrieval accuracy (beyond top-1) at N=1,000 and N=10,000.
4. **Ablate the token compression count**: Test Q-Former with K=64 and K=128 to justify K=32 as the optimal trade-off between efficiency and information retention.

## Score and Decision

**Calibration papers compared:**
- **High anchors (8s):** EytBpUGB1Z (Retrieval Head Mechanism, 8/8/8/8) — strong mechanistic insights + clean NIAH-style evaluation; oSQiao9GqB (LLaVA-Interleave, 8/8/6) — comprehensive multi-image benchmark + new method; FSjIrOm1vz (Inference Scaling for RAG, 8/8/8/8) — thorough empirical study. This paper is below these because its benchmark is narrower (binary presence verification, not general reasoning) and its method evaluation is confounded by training asymmetry.
- **Medium anchors (6s):** NL-Eye (2zmO1GVT0Y, 6/5/6/6/6) — benchmark paper with diagnostic findings; KiVA (vNATZfmY6R, 6/8/8/6) — benchmark with human studies; jZsN9zo8Qi (VEGA, 6.5 avg) — new interleaved benchmark. This paper is comparable to or slightly above these: the VH bench is less broad than KiVA's three-stage evaluation, but the MIRAGE system adds a practical engineering contribution beyond pure benchmarking.
- **Low anchors (3–5):** OZdr2mV5EI (unfair comparison trained vs. zero-shot, avg ~3.4) — relevant to this paper's comparison asymmetry concern; xE3Ra2GTpX (MKRAG, 3/6/5/3, avg 4.25) — RAG paper with incremental novelty. This paper is notably above these: VHs has genuine diagnostic findings and MIRAGE has clearer practical utility.

The asymmetric comparison concern from the harsh critic is genuine but not fatal—analogous papers like LLaVA-Interleave (8/8/6) were accepted despite limited-baseline fine-tuning evaluations. The benchmark narrowness is a significant drawback but the diagnostic findings (positional bias, distractor susceptibility, LMM vs. LLM robustness gap) and practical system (MIRAGE) elevate it above borderline. I position it at **6.0**: clear value to the community through the benchmark and diagnostic findings, but held back by overclaimed scope and confounded method evaluation.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>