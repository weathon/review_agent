# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅. The paper is clearly within ICML scope, focusing on generative modeling, attention mechanisms, controllable image generation, and evaluation of few-step diffusion/flow-matching models.

## Minimum Quality
Pass ✅. The submission contains the expected scientific components, including abstract, introduction, related work, method, experiments, results, and conclusion; while I have substantial concerns about validation and positioning, these are review-level issues rather than desk-reject-level defects.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅. I did not find any hidden instructions, reviewer-directed prompt injection, or other manipulative content in the provided paper text.

# Expected Review Outcome:
## Summary
This paper proposes Value Sign Flip (VSF), a training-free negative-guidance mechanism for few-step image and video generation models, especially MMDiT-style and cross-attention architectures. The core idea is to concatenate positive and negative prompt tokens in attention while flipping the sign of the negative values, with additional masking and duplication for MMDiT models. The paper evaluates VSF against NASA, NAG, vanilla generation, and some external baselines on a newly constructed benchmark, NegGenBench, and reports improved negative prompt adherence with favorable speed-quality trade-offs.

## Strengths and Weaknesses

**Strengths**

1. **The paper addresses a relevant practical problem.**  
   Negative prompting in few-step distilled generators is indeed underexplored and practically important. The motivation in Sections 1 and 2 is easy to follow: standard CFG often behaves badly in few-step models, and existing alternatives such as NASA and NAG have limitations in adaptivity, architecture coverage, or quality trade-offs.

2. **The method is simple and plausibly useful in practice.**  
   The central mechanism in **Equation 9** is compact and implementation-friendly. That is a real strength. A lot of controllable-generation papers bury the idea under layers of heuristics; here the core intervention is understandable: preserve the negative key for matching, but flip the negative value so attention to the undesired concept pushes features away from it.

3. **The paper makes an effort to adapt the method to MMDiT models, not just classic cross-attention models.**  
   The discussion in **Section 3.2** and the schematic in **Figure 3** are useful. In particular, the authors correctly identify that in MMDiT-style joint attention, naively flipping negative values would contaminate multiple attention paths, not just image-to-negative interactions. The duplication and masking workaround is a concrete engineering contribution.

4. **There is at least some empirical breadth.**  
   The paper compares against vanilla, NAG, NASA, and CFG, and also reports external comparisons in **Table 1**. It includes a trade-off study in **Figure 5**, a small human validation in **Table 3**, and some qualitative examples in **Figure 1** and **Figure 4**. Relative to many workshop-style submissions in this area, this is a more complete package.

5. **Some figures do communicate the intended effect well.**  
   **Figure 1** is a useful qualitative sanity check. The examples are challenging because the negative prompt refers to an essential object part, not a trivial style token. The bicycle-without-wheels and clock-without-hands examples are exactly the kind of cases where a method can easily collapse semantics or produce nonsense. The fact that the outputs still resemble coherent objects supports the paper’s practical motivation, even if the broader evidence remains incomplete.

6. **The trade-off framing is one of the better parts of the empirical section.**  
   The top row of **Figure 5** is more informative than a single operating-point table because it acknowledges that these methods live on a three-way frontier between positive adherence, negative adherence, and quality. That is the right lens for this problem.

---

**Weaknesses**

1. **The main mathematical claim, namely that VSF is “equivalent” to the weighted NASA-style form, is asserted too casually and is not justified in the main paper.**  
   In **Section 3.1**, the paper introduces a token-level weighting view via **Equations 6 to 8**, then presents VSF in **Equation 9**, and states “Mathematically, this is equivalent to \(Z^W\). Proof is in the Appendix.” This is a central claim, not a side remark. In the main paper, however, the equivalence is not demonstrated, and it is not obvious as written. In **Equation 8**, \(W\) is defined from sums of exponentiated logits, while in **Equation 9** the softmax is applied over a concatenated sequence and multiplied by signed values. Those are related ideas, but “equivalent” is stronger than “motivated by” or “can be interpreted as.” If the equivalence only holds under specific aggregation assumptions, token grouping, or omission of normalization constants, then the current presentation is overstated. This matters because a large part of the paper’s conceptual pitch is that VSF is not just a heuristic but an adaptive attention mechanism with a clean theoretical interpretation.

2. **The method is presented as adaptive, but the paper does not isolate where the gains come from: sign flipping itself, duplication/masking, prompt-padding removal, or hyperparameter retuning.**  
   This is a serious attribution problem. The actual MMDiT method in **Section 3.2** is not just “flip the sign of negative values.” It also duplicates the negative branch, applies directional attention masks, adds attention bias \(-\beta\), removes padding tokens, and mentions an implementation-specific scaling offset. By the time we get to experiments, the reader no longer knows which ingredient is responsible for the gains. The ablation in **Section 6.3** is helpful but still incomplete. For example, **Figure 5** bottom suggests “no bias” performs similarly, but there is no clean quantitative table with fixed operating points, variance estimates, or architecture-specific breakdowns. The paper’s title strongly sells “Value Sign Flip,” but the empirical package does not convincingly disentangle the sign flip from the surrounding machinery.

3. **The evaluation benchmark is fully synthetic and potentially tailored to the method, but the paper does not adequately discuss this risk.**  
   In **Section 4.1**, NegGenBench is generated using ChatGPT o3, including both prompt pairs and evaluation questions. This creates a pipeline where benchmark construction, evaluation formulation, and judged criteria are all mediated by language models. That is not automatically invalid, but it raises obvious concerns: distribution bias, prompt style homogeneity, and hidden alignment between benchmark wording and VLM judge behavior. The dataset is also only 200 prompts with 2 seeds in the main results, which is not especially large for a benchmark paper or even for a method paper making broad empirical claims. Because the paper relies so heavily on this dataset for its main conclusions, the lack of stronger dataset analysis weakens the scientific value. At minimum, I would expect categories, difficulty splits, prompt templates, failure cases, and some evidence that results generalize beyond this benchmark.

4. **The automatic evaluation protocol is much weaker than the paper acknowledges.**  
   The main results in **Table 2** are based on a single MLLM judge, llama-4-maverick, answering generated questions. The paper says human validation and Qwen-based evaluation align “relative ranking,” but in the main paper the human study in **Table 3** is tiny: just 10 selected prompts with 2 seeds. That is far too small to validate a benchmark-driven claim of superiority. Moreover, the selection protocol for these 10 prompts is not described in the main paper, which makes “selected prompts” a red flag. The quality score is also derived from the same MLLM pipeline, and the paper itself notes on **Page 7** that the quality scores are “relatively generous.” Once the same imperfect judge is used for positive adherence, negative adherence, and quality, it becomes hard to know how much of the reported gain is real versus judge-specific preference.

5. **The baseline comparisons are not fully convincing, and some are unusually favorable to the proposed method.**  
   The treatment of NAG is especially delicate. In **Section 4.2**, the authors say they retuned NAG for stronger negation, yielding “NAG Strong,” and similarly provide “VSF Strong” and “VSF Quality.” That is reasonable in spirit, but the optimization budgets are asymmetric in **Section 6.1**: 66 runs for VSF, 287 for NAG, and only 10 for NASA. NASA, then, is effectively under-tuned by design because it has one parameter, yet the main tables compare a single NASA operating point against two handpicked VSF operating points. The fairness issue is not just the number of runs, it is whether all methods are compared at matched quality or matched positive adherence. **Table 2** gives one point per method, but without confidence intervals or a frontier comparison in the table itself, the reader cannot assess whether the reported gains are robust or just due to point selection. Also, the paper reimplements NASA because code was unavailable, which is understandable, but this introduces extra uncertainty exactly for one of the key baselines.

6. **The paper overstates comparative claims against CFG and external models.**  
   The abstract and results text emphasize that VSF beats prior methods and even CFG in non-few-step models. But in **Table 2**, CFG is run at 28 steps, while VSF is run on a different inference regime and presumably different architecture/setup. The comparison is interesting, but it is not apples-to-apples enough to support broad statements like “stronger ability to avoid negative elements” in a general sense. Similarly, **Table 1** compares against external models such as GPT-4o, Nano Banana, Janus-4o, and Qwen-Image with very limited methodological detail in the main paper, while deferring experiment details to the appendix. Since the review should be based on the main paper, these external comparisons are difficult to evaluate rigorously. The open/closed-weight split is informative, but the experimental design is too underspecified to carry much weight.

7. **The figures are mixed in usefulness, and some claims based on them are stronger than what they actually show.**  
   **Figure 3** is one of the more useful figures because it clarifies the masking logic for MMDiT. However, it also exposes a complication: the proposed method is no longer the very simple “flip values” story advertised in the title. It is a more involved attention routing scheme with duplicated negative tokens, masks, and bias.  
   **Figure 4** is much less convincing scientifically. The “style avoidance” and “semi-abstract art” examples are interesting curiosities, but they drift away from the paper’s core claim about negative prompt guidance. Without systematic analysis, these examples read more like demos than evidence.  
   **Figure 5** is the most useful empirical figure, but it still lacks crucial details. The trade-off curves support the idea that VSF has a better frontier than NAG/NASA, yet the axes appear to be based on the same MLLM judge, and no uncertainty bands are shown. Given that the sweep is over only 100 prompts and one seed for this section, the curves should not be treated as definitive.

8. **The experimental reporting is incomplete for an ICML main-track standard.**  
   Several important details are deferred to the appendix or omitted from the main paper: exact hyperparameters, prompt examples beyond a few visuals, sampling settings, how scales were selected for the headline methods, whether the same prompt pairs were used across all models including external baselines, and whether prompt wording was modified per model. This is not just a reproducibility complaint. It affects interpretation of **Table 1** and **Table 2**. If the main contribution is empirical performance on a new benchmark, then the benchmark and protocol need to be described with more care in the main paper.

9. **The originality is moderate rather than strong, despite the paper’s framing.**  
   The idea of intervening in attention for guidance is not new, and the paper itself builds directly on NASA, NAG, and dynamic negative guidance. The main twist is to flip the sign of negative values while preserving matching through the unflipped keys, plus a routing modification for MMDiT. That is a neat insight, but it is still fairly incremental. The paper would be easier to champion if it provided either a deeper theoretical understanding of why this specific intervention should work, or a more compelling empirical demonstration across multiple architectures and datasets.

10. **There are several writing and presentation issues that reduce confidence.**  
   The paper is readable overall, but there are many local errors and some imprecise statements. Examples include “few-shot models” instead of “few-step models” in **Section 2.4**, the dangling “Figure ??” reference on **Page 8**, multiple grammar issues, and some claims that are too broad for the evidence provided. Even the opening qualitative examples in **Figure 1** have awkward prompt wording that makes it hard to know whether the challenge comes from negation understanding or from unusual prompt construction. These may sound cosmetic, but for a paper making delicate distinctions about prompt following and semantic suppression, imprecision in wording matters.

11. **The paper claims applicability to video generation, but the evidence for video is essentially absent in the main paper.**  
   The abstract explicitly mentions image and video generation, and the introduction names Wan as a supported cross-attention model. Yet the main paper presents image-centric methodology and image-centric results. If there are video experiments, they are not visible in the main text. As written, the video claim feels under-substantiated.

12. **Table 2 contains promising numbers, but the interpretation is more fragile than the authors suggest.**  
   The headline result, VSF Strong achieving **0.545 negative score** versus **0.380 for NASA** and **0.320 for NAG Strong**, is nontrivial. However, this comes with a drop in positive score to **0.870**, which is materially lower than the other methods, and the quality score remains judge-based. Meanwhile, VSF Quality raises quality to **0.986** but drops negative score to **0.420**. So the central story is not “VSF dominates,” but rather “VSF expands the operating range.” That is still useful, but the paper sometimes phrases the result too aggressively.  
   Similarly, **Table 3** is actually interesting because it somewhat supports the stronger negation-following claim, with VSF Quality reaching **0.550 negative score** in human labeling, much higher than NAG/NASA. But the same table also shows a lower positive score and only moderate quality. Because this human study is so small, it should be presented more cautiously, not as strong validation.

Overall, I do think there is a real idea here. The problem is that the paper currently packages a practical trick as if it were a thoroughly validated and cleanly characterized method paper. At ICML bar, that gap matters.

## Potentially Missing Related Work
1. **Gong, J., Yang, T., Wang, J., “HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance” (2025).**  
   This appears directly relevant because it uses a dual-prompt guidance design in one-step diffusion, which is close to the paper’s setting of few-step/distilled generation with prompt-based steering. It would strengthen the related work in **Section 2.4** and the empirical positioning in **Section 4.2**, especially when discussing guidance mechanisms specifically designed for highly accelerated diffusion models.

2. **Chang, H., Kim, S., Choi, Y., “Dynamic VLM-Guided Negative Prompting for Diffusion Models” (2025).**  
   This is directly related to the paper’s broader theme of adaptive negative guidance. Even if the technical mechanism differs, it is highly relevant to **Section 2.2** on dynamic negative guidance and also to the evaluation discussion in **Section 4.2**, since the present paper also leans heavily on VLM/MLLM-based machinery. The authors should explain how VSF differs from dynamic prompt-generation approaches and whether the two are complementary.

## Soundness
2: fair. The core idea is plausible and some results are encouraging, but key claims are not fully supported by rigorous analysis or sufficiently robust evaluation.

## Presentation
2: fair. The paper is readable and has useful figures, but there are multiple clarity issues, imprecise claims, missing main-text details, and some writing errors.

## Significance
2: fair. Negative prompting for few-step generators is practically relevant, but the evidence here is not yet strong enough to establish broad impact.

## Originality
2: fair. The value-sign-flip idea is a reasonable twist on attention-based guidance, but overall the contribution feels incremental relative to NASA, NAG, and prior dynamic guidance work.

## Key Questions For Authors
1. **Can you make the main theoretical claim precise in the rebuttal?**  
   In **Section 3.1**, please clarify under what assumptions **Equation 9** is actually equivalent to the weighted form in **Equation 7**, versus merely being an intuitive realization of it. If the equivalence is exact only after aggregation over positive/negative token groups or under specific normalization assumptions, please state that explicitly. A precise clarification here would increase my confidence in the conceptual contribution.

2. **Can you provide a cleaner attribution study of what really matters in VSF?**  
   Right now the method includes sign flip, duplication, masking, attention bias, padding removal, and tuning choices. Please report a compact fixed-setting ablation table in the rebuttal, ideally on the same 200-prompt benchmark, that isolates:  
   (a) sign flip only,  
   (b) sign flip + duplication,  
   (c) sign flip + duplication + mask,  
   (d) sign flip + duplication + mask + bias,  
   with the same backbone and matched negative-guidance scale. If the gains mostly persist under this decomposition, my assessment would improve.

3. **How robust are the results to evaluator choice and prompt subset?**  
   The main concern is over-reliance on one MLLM judge. If you can provide, in the rebuttal, stronger human evidence or at least agreement statistics across two distinct judges on a larger random subset, that would materially help. In particular, I would like to know whether the ranking in **Table 2** remains stable on a non-selected subset of prompts.

4. **Please clarify the fairness of the baseline comparisons.**  
   How were the operating points in **Table 2** chosen for NASA, NAG, NAG Strong, VSF Quality, and VSF Strong? Were they selected on the full benchmark, on a held-out validation subset, or post hoc after inspecting the test results? A clear protocol matters a lot here. If there was any tuning directly on the reported benchmark set, that weakens the evidence.

5. **Where is the evidence for the video claim?**  
   Since the abstract explicitly includes video generation, I would expect at least one main-paper experiment or qualitative figure on video. If such evidence is unavailable, I strongly suggest narrowing the claim to image generation in the final version. If you can provide substantive video results in rebuttal, that would strengthen the paper.

**Score-change criteria:**  
My score could increase if the authors convincingly address the attribution issue, clarify the Equation 7/9 relationship, and provide stronger evidence that the reported gains are robust beyond one MLLM judge and a synthetic benchmark. My score would decrease if the operating points in **Table 2** were selected post hoc on the reported evaluation set, or if the claimed theoretical equivalence turns out to be only loose intuition.

## Limitations
The paper does acknowledge some limitations indirectly, but this is not addressed properly enough. The most important missing limitation is evaluator dependence: the benchmark, questions, and main metrics are all heavily mediated by language models, which can bias conclusions. The paper should also explicitly discuss the risk of using negative guidance for content suppression in ways that may encode undesirable normative choices or over-filter valid content. Finally, since the paper claims applicability to video and multiple architectures, it should clearly state where evidence is currently limited.

## Overall Recommendation
3: Borderline reject. There is a sensible practical idea here, and some results are encouraging, especially the trade-off perspective and the human-validation signal in Table 3. However, the paper currently has too many substantive weaknesses for ICML main track: incomplete characterization of the method, over-reliance on synthetic benchmarking and one MLLM judge, insufficiently clean ablations, somewhat overstated claims, and limited support for some of the broader conclusions. I would encourage the authors to continue this line of work, but I do not think the present version clears the bar.

## Confidence
3: Fairly confident in assessment. I understand the main idea and checked the empirical claims in the main paper carefully, but some theoretical details and implementation specifics are deferred outside the main text.