# Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs

- Decision: Reject
- Scores: 6, 6, 8, 4

## Abstract
Large language models (LLMs) often exhibit sycophantic behaviors---such as excessive agreement with or flattery of the user---but it is unclear whether these behaviors arise from a single mechanism or multiple distinct processes. We decompose sycophancy into \emph{sycophantic agreement} and \emph{sycophantic praise}, contrasting both with \emph{genuine agreement}. Using difference-in-means directions, activation additions, and subspace geometry across multiple models and datasets, we show that: (1) the three behaviors are encoded along distinct linear directions in latent space; (2) each behavior can be independently amplified or suppressed without affecting the others; and (3) their representational structure is consistent across model families and scales. These results suggest that sycophantic behaviors correspond to distinct, independently steerable representations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper argues that “sycophancy” in LLMs is not monolithic. It decomposes the phenomenon into sycophantic agreement (SYA), genuine agreement (GA), and sycophantic praise (SYPR), provides operational definitions, and studies their internal representations. Using DiffMean (difference-in-means) directions on residual stream activations, the authors report that the three behaviors are linearly separable, independently steerable via activation addition, and consistent across model families/scales. They construct controlled synthetic datasets with a knowledge filter (neutral-prompt test) to avoid conflating uncertainty with sycophancy, and show cross-model replications plus an external check on a TruthfulQA sycophancy subset. Overall, the work claims behavior-selective control, suppressing “echoing false user claims” without harming “agreeing when the user is correct.”

### Strengths
Clear behavioral formalization. The paper crisply distinguishes SYA, GA, and SYPR and ties them to observable labels with a knowledge filter; Synthetic controls improves reproducibility.

Interpretable method. DiffMean directions on residual activations are lightweight, parameter-free, and easy to inspect; the AUROC and geometry analyses are compelling.

Mechanistic evidence. Steering via activation addition modulates one behavior with minimal cross-effects, strengthening the case for functional separability.

Cross-model consistency. Results replicate across multiple model families/sizes, suggesting the effect is not a single-model artifact.

### Weaknesses
The main evidence is on controlled synthetic tasks; only a limited real-world test (TruthfulQA sycophancy subset) is shown. Generalization to open-ended, multi-turn dialogue or socially nuanced settings is unclear.

Deployment considerations not quantified. Steering overhead (per-token, per-layer activation injections), stability under different decoding settings are not measured.

### Questions
Do directions learned on one domain/model transfer zero-shot to other domains (e.g., safety, medical advice), languages, or longer conversations? How does selectivity change in truly open-ended dialogue with mixed user intentions?

What is the inference-time cost and latency impact of per-layer steering at α>0? How stable are the effects across temperatures, sampling strategies, and long-context prompts?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper operationalizes three behaviors on paired user-model turns: (1) sycophantic agreement (SYA), i.e., the model incorrectly agrees with the user's claim, (2) genuine agreement (GA), i.e., the model correctly agrees with the user's claim, and (3) sycophantic praise (SYPR), i.e., exaggerated flattery regardless of the factual content. The authors show that SYA and GA are highly aligned (geometrically) in early layers but diverge in middle layers, while SYPR remains approximately orthogonal. Moreover, activation addition (steering) along learned directions changes the target behavior with minimal cross-effects; this remains true even after subspace removal of other behavior directions. This representational pattern and steerability can be replicated across several instruction-tuned model families and scales.

### Strengths
1. The behavioral operationalization is clear, and the knowledge filtering reduces confounds from uncertainty or ignorance.
2. Simple, transparent measures of approaches (DiffMean, cosine geometry, activation addition) lead to consistent and interpretable findings.
3. The subspace removal strengthens the independence claim beyond raw separability.
4. Cross-model replications (Qwen, LLaMA, GPT-OSS), and external test on TruthfulQA SycoEval are insightful. 
5. The framing, specifically, precision mitigation of sycophancy without harming GA, is practically valuable.
6. Overall, the work provides valuable practical insights.

### Weaknesses
1. Causal claims rest on linear steering.  Have you conducted more direct causal tracing or minimal-intervention experiments?That would strengthen mechanistic conclusions.
2. SYPR measurement depends on a sentiment or praise classifier plus artificial stems. This may inflate orthogonality and selectivity for praise in wider contexts (i.e., selection bias). Have you thought of ways to mitigate such a bias?
3. There is a heavy reliance on single-turn synthetic templates. Multi-turn, socially nuanced dialogs could be further explored.
4. The statistical reporting for steering/selectivity could include CIs and effect-size comparisons across datasets.
5. How do you avoid or mitigate potential lexical or style leakage across datasets? Despite efforts to diversify praise, an explicit lexical-control (e.g., content-word masking or paraphrase-invariance checks) would help.

### Questions
See comments above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper argues that sycophancy in LLMs is three behaviors (sycophantic agreement, genuine agreement, and sycophantic praise). The paper shows that these behaviors correspond to distinct subspaces that diverge in mid layers. Praise is largely orthogonal. Each behavior can be independently amplified/suppressed with minimal side-effects.

### Strengths
+ The decomposition is clear and seems well-motivated

+ I like that the authors looked at robustness across datasets and models (shows that the structure generalizes)

+ Writing and figures are very clear

### Weaknesses
- There doesn't appear to be any direct cross-dataset steering transfer, that is, directions trained on one template are not shown to steer on disjoint templates

- The selectivity ratio depends on an arbitrary epsilon (and there are no confidence intervals reported); this makes the cross-model comparisons hard to interpret

- The steering setup relies a lot on prompting ("Assistant: You are...") which makes it hard to tell if the results just rely on textual pattersn rather than praise behavior

### Questions
- Can the authors evaluate cross-template steering transfer? Learning SYA/GA/SYPR directions on arithmetic and testing steering on CITIES or COMMON CLAIMS to assess behavioral generalization beyond shared geometry?

- How sensitive are the coverage and selectivity results to the knowledge-predicate thresholds?

- Does praise steering persist for free-form praise phrasing beyond “you are ...” and under adjective-family holdouts or paraphrased compliments?

- Are there identifiable attention heads or MLP units that align with the mid-layer SYA–GA split, e.g., circuit level localization

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper argues that sycophancy is not monolithic. Using arithmetics on residual activations and subspace geometry across datasets, the authors claim that sycophantic agreement, genuine agreement, and sycophantic praise are linearly separable and causally steerable, with partial replication across model families and a small external test on TruthfulQA-style prompts.

### Strengths
1. The paper provides a clear categorization of sycophancy problem by decomposing it into sycophantic agreement, genuine agreement, and sycophantic praise, which helps ground the discussion in measurable behaviors.

2. The setup of using difference-in-means, different evaluation metrics, and activation additions is rigorous and effective in proving the claims. The results replicated across model families and scales, further strengthening the soundness.

3. The results is inspiring and valuable: treating sycophancy as a family of distinct mechanisms rather than a single behavior gives future a better ground to understand the phenomenon.

### Weaknesses
1. The datasets are largely synthetic, and the “praise” behavior is constructed in an artificial way that may not capture realistic conversational sycophancy.

2. The methods are largely standard in interpretability research (Mean, linear probing, activation steering) and do not introduce new methodological innovations/ insights beyond their particular behavioral framing.

3. External validation is limited: the only OOD test is small and shows minimal effect, leaving unclear how well the proposed directions generalize to real deployment scenario.

### Questions
1. Could the linear directions identified here correspond to mixtures of nonlinear features—e.g., would SAE features reveal similar behavioral boundaries?

2. How do these results compare quantitatively with prior interpretability work in this area?

3. Have you evaluated whether the discovered directions maintain separability when derived from more realistic conversational data?

### Soundness
4

### Presentation
3

### Contribution
2
