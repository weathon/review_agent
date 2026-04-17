# Autoregressive Models Rival Diffusion Models at ANY-ORDER Generation

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Diffusion language models enable any-order generation and bidirectional conditioning, offering appealing flexibility for tasks such as infilling, rewriting, and self-correction. However, their formulation—predicting one part of a sequence from another within a single-step dependency—limits modeling depth and often yields lower sample quality and stability than autoregressive (AR) models. To address this, we revisit autoregressive modeling as a foundation and reformulate diffusion-style training into a structured multi-group prediction process. We propose Any-order Any-subset Autoregressive modeling (A3), a generalized framework that extends the standard AR factorization to arbitrary token groups and generation orders. A3 preserves the probabilistic rigor and multi-layer dependency modeling of AR while inheriting diffusion models' flexibility for parallel and bidirectional generation. We implement A3 through a two-stream attention architecture and a progressive adaptation strategy that transitions pretrained AR models toward any-order prediction. Experiments on question answering, commonsense reasoning, and story infilling demonstrate that A3 outperforms diffusion-based models while maintaining flexible decoding. This work offers a unified approach for a flexible, efficient, and novel language modeling paradigm. Code is at https://github.com/PKU-ML/Any-order-Any-subset-AR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes A3, a sequence modeling framework that generalizes AR factorization to predict arbitrary groups of tokens in any order, aiming to combine AR’s likelihood-faithfulness with the flexibility/parallelism of diffusion and semi-AR methods. The core is a two-stream attention design (content vs. query streams) plus a progressive curriculum (AR init → group expansion → order permutation). Experiments on QA/commonsense/story-infilling show A3 beats several diffusion LMs and scales with model size, though it still trails an AR baseline at equal parameter counts; the authors attribute this to smaller pretraining budgets.

### Strengths
- Neat architectural idea. The two-stream attention extends XLNet-style permutation to groups: content attends to ≤ current group; query attends only to prior groups, yielding a clean predictive head for the next group. The masks and equations are well specified.
- Thoughtful training curriculum. The three-stage progression (singleton AR → contiguous groups → permuted grouping) offers a practical path from a pretrained AR model to any-order behavior.
- Flexible inference recipes. The paper describes groupwise AR sampling and dynamic resampling, articulating a speed–quality trade-off and illustrating it empirically.
- Empirical signal vs. diffusion LMs. On TriviaQA/PIQA/etc., A3 (1B–8B) outperforms diffusion baselines and shows sensible scaling; the table also situates an AR baseline.

### Weaknesses
- Missing baselines from the semi-AR family. The related-work discussion notes multi-token prediction and insertion models, but the experiments do not include strong semi-autoregressive baselines (e.g.,  speculative-with-MTP) that directly target speed-ups while preserving AR quality. This makes it hard to attribute A3’s benefits to grouping vs. other established accelerations.
- Latency reporting is proxy-based. The speed-quality trade-off uses per-sequence decoding time and Llama-measured perplexity for unconditional generation; end-to-end wall-clock latency under realistic batching and KV-cache reuse is not reported for the task benchmarks. More concrete throughput/latency numbers (tokens/s, ms/sample) would help.
- Ablation depth. The method depends on group size schedules, permutation schemes, and the two-stream mask design. The paper lacks sensitivity analyses for (i) curriculum choices (stage lengths, s progression), (ii) group selection criteria during dynamic resampling (beyond brief entropy vs. confidence), and (iii) robustness across context lengths.

### Questions
- Compute-normalized comparison. Can you provide training FLOPs and data tokens for A3 vs. AR and diffusion baselines, and where possible train the AR baseline on the same 2B tokens to isolate the architectural effect? 
- Semi-AR baselines. How does A3 compare against modern multi-token-prediction methods under matched setups? A small-scale controlled study would be informative. 
- Curriculum sensitivity. What happens if you skip Stage 2 or change the group-size schedule? Is the two-stream architecture alone sufficient to get most of the gains? 
- Inference consistency. For dynamic resampling, does there exist a well-defined induced factorization (e.g., an ordering over committed sets) that preserves AR semantics, or is this best viewed as a heuristic? Any failure cases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes A3 (Any-order Any-subset Autoregressive), a sequence modeling framework that keeps the probabilistic rigor and training stability of standard autoregressive (AR) models but relaxes the decoding order to arbitrary groups of tokens. Concretely, A3 factorizes a sequence into groups and trains the model to predict any group conditioned on previously generated groups, using a two-stream attention layout to separate “what has been generated” from “what is now being predicted.” At inference, A3 can decode group-by-group (for speed) or do dynamic re-sampling of uncertain positions (for quality), so it can handle infilling and other non-left-to-right scenarios in a single model. Empirically, on common LM benchmarks (QA, commonsense, story cloze), A3 outperforms discrete-diffusion / masked-iterative baselines at similar or larger compute, but still trails a comparable AR model that was trained on far more tokens.

### Strengths
1. Clear target: AR-level stability + arbitrary-order generation. The paper correctly identifies a real and currently active gap: classic AR models are stable and likelihood-faithful but order-rigid, while discrete diffusion / iterative masked LM are order-flexible but multi-step, hyperparameter-sensitive, and sometimes harder to train. Proposing a single framework that “looks like AR to the optimizer” but “behaves like an infiller” is a sensible and timely goal.
2. Groupwise factorization is a clean, composable idea. Writing ($p(x)=\prod_k p(x_{G_k}\mid x_{G_{<k}})$) and randomizing the grouping/order during training is conceptually simple, keeps likelihood well-defined, and reuses the well-understood AR training pipeline. Compared with diffusion-style schedules, this reduces the number of design knobs while still giving order flexibility.
3. Two-stream attention is a reasonable architectural choice. Reusing an XLNet-style separation between “content” and “query” to implement “see only past groups but write to current group” is a sensible reuse of an established mechanism, so the method is not just algorithmic but also implementable on existing Transformer stacks.

### Weaknesses
1. Parallel generation is only *heuristically* correct, not *distributionally* correct. The paper’s decoding story (“decode some groups, resample the uncertain ones”) is an engineering compromise, but it does not give the kind of provable, joint-distribution-correct parallel sampling that very recent any-subset AR work is starting to provide (e.g. ASSD in Guo & Ermon 2025) — those works explicitly address the mismatch between parallel predictions and the target joint, while A3 largely sidesteps it. This makes the “any-subset” claim weaker on the sampling-theoretic dimension.
2. Comparisons are not load-bearing for the main claim. The paper leans on the fact that it beats discrete/diffusion-style baselines, but those baselines are known to be data/step/hyperparameter hungry; meanwhile A3 is compared to a much better funded AR line only anecdotally (“we used fewer tokens, so we lag”). Without an equal-data, equal-backbone, equal-context comparison against a strong AR infiller (e.g. a repurposed XLNet / permutation LM, or an AR+speculative multi-token decoder), it is hard to measure how much of the gain comes from the grouping idea itself vs. from simply staying in the AR training regime.
3. Method novelty is incremental over permutation-style / two-stream LMs. At a high level, A3 = permutation/pseudo-permutation LM (XLNet-like) + explicit grouping curriculum + a decoding heuristic. The paper frames it as a “generalization of AR to any order, any subset,” but several of the enabling ingredients — two-stream separation, masked/pseudo-permutation training, curriculum over masks — have been explored in earlier LM or masked-hard-attention work. The paper’s specific combination is tidy, but the conceptual leap is smaller than the title suggests.
4. Group choice is exogenous and may be the hard part. The model is trained on random / curriculum-defined partitions, but the real deployments that need “any subset” (layout-controlled editing, multi-span infilling, multi-document patching) tend to have structured subsets (spans aligned to discourse / layout). A3 doesn’t learn the grouping policy, and the paper doesn’t show that the learned model is robust to strongly biased or adversarial group patterns. That is exactly where arbitrary-order models tend to break.

### Questions
1. On distributional correctness: your dynamic re-sampling is essentially a confidence- or entropy-driven iterative refinement. How do you ensure that, after several such rounds, the final joint over all tokens is still close to the model’s intended factorization — and how does this compare empirically to ASSD-style “correct-by-construction” decoding for any-subset ARMs? (A small experiment against the scheme in Guo & Ermon 2025 would make the generative claim much harder to dispute.)
2. On asymptotic scaling vs. plain AR: you attribute the quality gap to “only 2B tokens,” but you also inject extra supervision signals (more factorization patterns, more masked positions). Do you have evidence that, at equal data and wall-clock, A3 does not hurt the base AR’s next-token perplexity, especially on long contexts where order permutations are most disruptive? A curve like “data ↑ → (AR PPL, A3 PPL) → gap ↓” would make the core claim much more convincing.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses a challenge in sequence generation: reconciling the training stability/ generation quality of autoregressive (AR) models with the flexibility/ parallelism of diffusion-based models. This problem is theoretically meaningful, as existing paradigms struggle to balance efficiency, quality, and adaptability—key requirements for real-world applications like long-context reasoning, text infilling, and fast content generation.

A3 resolves this by generalizing AR factorization to support "groupwise token prediction in arbitrary orders," with three core designs:
(1) two-stream attention architecture, (2) progressive training strategy, and (3) flexible inference, which supports groupwise AR sampling (fast, fixed group sizes) and dynamic resampling (high-quality, confidence/entropy-based subset selection).

Empirically, A3 (trained on 2B tokens) outperforms state-of-the-art diffusion models across QA (TriviaQA), commonsense reasoning (PIQA, HellaSwag), and infilling (ROCStories) tasks (e.g., A3-8B achieves 78.1% PIQA accuracy vs. 63.3% for DiffuLlama-7B). It also surpasses AR models in AR-disadvantaged tasks (e.g., ROCStories ROUGE-L: 18.6 vs. 10.5 for Llama-3.1-8B).

### Strengths
1. The paper focus a critical problem, which addresses a gap in sequence generation—reconciling AR’s stability/quality with diffusion’s flexibility/parallelism—aligned with real-world needs (long-context generation, infilling).

2. Explicitly outperforms AR models in infilling (ROCStories) by utilizing bidirectional context, validating its flexibility.

### Weaknesses
1. Fails to cite or discuss existing NAR research on Block-wise generation (e.g., Block-AR models that split sequences into fixed/masked blocks for parallel prediction) and progressive training (e.g., curriculum-based NAR training that increments block size or relaxes order constraints). This gap obscures A3’s incremental innovation—readers cannot distinguish whether A3’s groupwise/progressive designs are novel or iterative improvements on prior NAR work.

2. No controlled experiment comparing A3 with an AR model trained on the same 2B tokens (e.g., fine-tuned Llama-3.1-8B). The gap with top AR models (e.g., 19.4% vs. 52.1% on TriviaQA) may stem from data scarcity, not A3’s design—undermining assessment of its architectural advantage.

2. Only evaluates sequences of length 2048, with no tests on long contexts (16k/128k tokens). A3’s claimed advantage in solving AR’s long-context inefficiency remains unproven.

3. Figure 3 claims 3–4x faster decoding than AR but lacks direct time data and performances for baseline AR models (same hardware/sequence length) and explicit absolute time values, making efficiency gains hard to quantify.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces A³ (Any-order Any-subset Autoregressive modeling), a generalization of standard autoregressive (AR) language modeling. Instead of predicting tokens strictly left-to-right, A³ allows generation in arbitrary orders and subsets while maintaining a valid probabilistic formulation.

The method builds on a two-stream attention mechanism (content/query streams) similar to XLNet and uses a curriculum training schedule that progressively transitions from left-to-right to multi-token and random-order prediction. At inference, A³ supports both groupwise parallel decoding and dynamic resampling, trading off speed and quality.

Experiments across QA, reasoning, and infilling tasks show that A³ matches or surpasses diffusion-based models (e.g., DiffuLlama, Dream) using significantly less data, and maintains competitive performance to AR models while enabling faster, flexible generation.

Overall it is a practical step toward bridging AR and diffusion paradigms, retaining AR stability while introducing parallel generation flexibility.

### Strengths
1. Insightful formulation. The paper presents a very interesting perspective on any-order, any-subset autoregressive modeling, effectively bridging the strengths of AR and masked diffusion models in a unified probabilistic framework.

2. Clarity and organization. The paper is well written and easy to follow, with clear explanations, sound motivation, and well-structured methodology.

3. Experiments across multiple reasoning and generation benchmarks validate the effectiveness of the proposed approach, showing competitive or superior performance with improved flexibility.

### Weaknesses
1. What's new relative to prior AR generalizations? The author mentioned that the proposed method builds closely on existing ideas from XLNet (permutation-based AR) and masked diffusion modeling, with the main difference being a unified training/inference view. While conceptually elegant, the contribution may feel incremental rather than fundamentally new.

2. Evaluation scope and ablations are limited. Experiments mainly compare against diffusion-style baselines; there is less analysis against modern AR variants (e.g., speculative decoding, parallelized transformers) or ablations on key design choices like grouping strategy and curriculum schedule.

3. Practical speed–quality trade-offs unclear. Although the paper claims parallel generation benefits, detailed runtime comparisons and latency measurements are missing, making it hard to assess the real efficiency gains of “any-order” decoding in practice.

### Questions
1. How does A³ fundamentally differ from prior permutation-based autoregressive approaches such as XLNet or Permutation LM beyond the introduction of multi-token subsets? Could you clarify what new modeling capacity A³ enables that these earlier frameworks cannot?

2. The paper emphasizes that A³ enables parallel or groupwise decoding. Could you provide concrete runtime or latency benchmarks (e.g., decoding speedups vs. standard AR and diffusion models) to quantify the practical benefit of this flexibility?

3. The curriculum that transitions from left-to-right to any-order prediction seems important. How sensitive is model performance to the curriculum schedule (e.g., ratio of L2R vs. random-order batches)? Have you explored automatically learned or adaptive curricula?]

4. Since any-order generation can, in principle, apply to structured data (e.g., image or audio tokens), do you anticipate A³ extending naturally to multimodal diffusion–AR hybrids, or would significant architectural adjustments be required?

### Soundness
3

### Presentation
3

### Contribution
3
