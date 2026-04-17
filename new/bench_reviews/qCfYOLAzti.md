Let me verify some key claims from the paper content provided in the user message. Let me check what the paper says about LaaJ evaluation and the metrics used in main tables.

Looking at the paper content provided:

1. **Table 1 (TOFU)**: Uses Memorization (Mem.), Utility (Util.), and Agg. - these are the standard OpenUnlearning metrics the paper criticizes.
2. **Table 2 (WMDP)**: Uses QA accuracy and MMLU - again standard metrics.
3. **Figure 4c**: Uses LaaJ evaluations with Naturalness and Similarity scores, but only for one model/benchmark combination.
4. **Appendix F**: Mentioned as containing more results but not provided.

Now let me check whether the paper addresses some of the harsh critic's concerns:

- The paper does use LaaJ only lightly (Fig 4c only) while the main tables use the criticized metrics.
- The "squeezing effect" conjecture is stated but not rigorously proven as causal.
- The theoretical analysis (Thm 5.2/5.3) shows gradient reshaping but doesn't explicitly connect to semantic content.
- No error bars or statistical significance reported in tables.
- No comparison with simpler data augmentation baselines for BS-S.
- No adversarial robustness evaluation.

Let me now carefully assess each major claim from the harsh critic:

**Claim 1: Squeezing effect overstated / only partially supported** - This is partially valid. The paper provides empirical evidence (Fig 2a-c) and theoretical analysis, but the causal link from softmax normalization → semantically similar rephrasings is supported qualitatively, not rigorously. However, the evidence is still suggestive and the intuition is sound. This is a valid but moderated weakness.

**Claim 2: Evaluation relies on same flawed metrics** - This is VALID and important. The paper's central thesis attacks existing metrics, yet all main tables use those same metrics. LaaJ appears only in Fig 4c for one setting.

**Claim 3: Model beliefs as semantic proxy assumed not validated** - This is PARTIALLY valid. Fig 2a does show high-likelihood regions have higher semantic similarity. But the paper doesn't validate that top-k tokens in BS-T specifically capture paraphrases rather than benign alternatives.

**Claim 4: Theoretical analysis doesn't validate safety benefits** - This is VALID but somewhat expected for a theoretical analysis that provides gradient-level insights rather than guarantees. The theory provides mechanistic understanding, not safety guarantees.

**Claim 5: Cannot separate better forgetting from more aggressive suppression** - This is VALID. The paper doesn't include forget-retain trade-off curves or matched-strength comparisons.

Now let me check what the Spark reviewer flags:

1. **No adversarial robustness evaluation** - Valid concern but this is somewhat outside scope of what the paper claims.
2. **No comparison with simple data augmentation baselines** - This is a valid and important point. BS-S adds model-generated sequences to the forget set, but this could just be data augmentation.
3. **No evaluation on rephrased queries** - Valid concern for a paper about "spurious unlearning from rephrasings."
4. **MUSE in appendix** - Valid but a minor point.

Now let me assess the overall quality and calibrate the score.

The paper makes a genuine contribution by identifying the squeezing effect and proposing a concrete method. The BS-T/BS-S methods are elegant and well-motivated. The theoretical analysis adds depth. However, there are significant weaknesses:
- The central thesis about flawed metrics is undermined by relying on those same metrics for main results
- The causal mechanism (squeezing effect) is plausible but not rigorously established
- No comparison with simple augmentation baselines
- No error bars or statistical significance
- LaaJ evaluation is thin

Comparing with similar papers:
- Q1MHvGmhyT (Closer Look at Unlearning) - scores 6,6,6,6 - identified metric issues, proposed methods
- huo8MqVH6t (G-effect) - scores 6,5,8,3,6,8 - gradient analysis paper, accepted poster
- Pd3jVGTacT (SimNPO) - scores 6,5,5,5 - rejected, incrementally improved NPO
- 51WraMid8K (Probabilistic Perspective) - scores 6,10,6,10 - accepted oral, novel evaluation framework
- AcR5Mngp1p (Knowledge-localized) - scores 3,5,6,6 - rejected, similar "superficial unlearning" concept
- TArmA033BU (MUSE) - scores 5,6,8,6 - accepted poster, benchmark paper

This paper is stronger than AcR5Mngp1p (which was rejected) and Pd3jVGTacT (rejected), but has more significant evaluation concerns than huo8MqVH6t (accepted). The conceptual contribution (squeezing effect) is interesting but the empirical validation is undermined by the metric inconsistency. The method is incremental over existing self-distillation/data augmentation ideas.

I'd place this around 5-6. It's a solid contribution in identifying an important phenomenon, but the evaluation inconsistency and lack of rigorous validation of the causal mechanism are significant weaknesses. The paper is borderline.

Let me settle on a score of 5.5 - the conceptual insight is valuable but the execution has meaningful gaps.</think>Now I have a thorough understanding of the paper and the calibration landscape. Let me write the final consolidated review.

---

## Summary

This paper identifies the "squeezing effect" in LLM unlearning — where gradient-ascent-based methods (GA, NPO) redistribute probability mass from target responses into semantically related high-likelihood rephrasings, leading to "spurious unlearning" that standard metrics (ROUGE, truth ratio) fail to detect. To counteract this, the authors propose a bootstrapping framework (BS-T at the token level, BS-S at the sequence level) that recycles the model's own high-confidence predictions ("model beliefs") as auxiliary unlearning targets. Theoretical analysis under the AKG learning dynamics framework shows how BS reshapes gradient residuals, and experiments across TOFU, MUSE, and WMDP benchmarks demonstrate consistent improvements over baselines.

## Strengths

1. **Important and well-diagnosed problem:** The "squeezing effect" and resulting "spurious unlearning" are real, timely, and underappreciated failure modes. The case studies in §3.1 — GA producing syntactic collapse with near-zero metrics, and NPO preserving sensitive content (e.g., "English") in rephrased form — convincingly demonstrate that existing metrics can be deeply misleading. This diagnosis alone adds value to the field.

2. **Principled method design tightly connected to the analysis:** BS-T (Eq. 5–6) and BS-S (Eq. 7) directly address the identified mechanism: by incorporating the model's own top-k predictions or high-confidence sampled sequences as unlearning targets, the method explicitly suppresses the regions where probability mass would otherwise escape. The connection between problem and solution is logically tight rather than ad hoc.

3. **Theoretical grounding under learning dynamics:** The AKG-based analysis (§5, Thm. 5.2/5.3) formalizes how BS-T reshapes the residual term to distribute forgetting pressure across belief neighborhoods, and how BS-S aggregates these effects across sequences. This provides mechanistic justification beyond empirical observations, which is uncommon in the unlearning literature.

4. **Consistent empirical improvements across settings:** Results on TOFU (1%/5%/10% across 1B/3B/8B), WMDP, and MUSE show consistent gains over NPO, WGA, and other baselines on both forgetting and retention metrics, suggesting the method generalizes across benchmarks and model scales.

5. **Practical compatibility and integration:** The framework is composable with existing objectives (NPO, WGA) and regularizations (GradDiff), and code is merged to OpenUnlearning, facilitating practical adoption.

## Weaknesses

### Major:

1. **The paper criticizes existing metrics as misleading but relies on them for its main results.** The central motivation (§3.1) argues that ROUGE, probability, and truth ratio systematically overstate unlearning success. Yet Tables 1–2 and appendix results use exactly these metrics (Memorization, Utility, Agg.) from OpenUnlearning as the primary evidence for BS's superiority. The LaaJ evaluation (Naturalness and Similarity) appears only in Figure 4c for a single model/benchmark setting (Llama 3.1 8B, TOFU 10%). No tabulated LaaJ results are provided across methods and benchmarks, making it impossible to assess whether BS genuinely reduces semantic leakage at scale. If standard metrics cannot be trusted to detect spurious unlearning (which the paper argues), then they also cannot validate the claim that BS achieves "more thorough forgetting." This inconsistency undermines the paper's core value proposition without necessarily invalidating the method.

2. **No comparison with simple data augmentation baselines isolates the "belief" contribution.** BS-S augments the forget set with model-generated high-confidence sequences. A natural baseline would augment with paraphrases from an external model, back-translated variants, or randomly sampled sequences — any of which would test whether the gains come specifically from targeting "model beliefs" (semantically related high-probability outputs) versus simply providing more diverse forget data. Without this control, BS-S's improvements cannot be attributed to the bootstrapping mechanism rather than generic data augmentation effects. BS-T similarly mixes top-k predictions with one-hot targets; comparing against a uniform or random target distribution would isolate the contribution of the belief-specific weighting.

3. **The causal link from softmax normalization to semantic paraphrases is asserted but not rigorously established.** §3.2 conjectures that the squeezing effect — probability mass redistributed by the softmax normalization constraint into semantically similar regions — is the dominant cause of spurious unlearning. Fig. 2a shows that high-likelihood regions are more semantically similar to targets, and Fig. 2b–c shows log-probability dynamics during training. However: (a) the paper does not quantify what fraction of probability redistribution flows into paraphrases versus unrelated continuations; (b) alternative explanations (e.g., finetuning artifacts, optimization biases) are not ruled out; (c) the theory (§5) formalizes gradient reshaping but never connects the top-k tokens specifically to semantic content. The observation is plausible and useful, but the paper repeatedly treats it as "the" mechanism, which is overclaimed relative to the evidence.

4. **No forget-retain trade-off curves or matched-strength comparisons.** The paper claims "superior balance between forgetting and retention," but all results compare methods at their own optimally-tuned hyperparameter settings. Without forget-retain curves (varying λ_BST or λ_BSS) or comparisons at matched forgetting strength (e.g., matched drop in target log-probability), one cannot distinguish whether BS achieves genuinely better forgetting for a given retention cost, or simply applies more forgetting pressure. The improvements in Table 1 are often modest (0.02–0.04 in Agg.), and no error bars or multiple-seed statistics are reported, leaving their robustness uncertain.

### Minor:

1. **Theoretical assumptions are strong and not empirically validated.** The AKG analysis (Lem. 5.1) relies on the lazy eNTK assumption and teacher forcing, both of which are known to be approximations that break down under substantial fine-tuning. The paper acknowledges that on-policy BS-S violates these assumptions (Appx. D.4) but does not validate whether the theoretical predictions (e.g., about residual structure) match observed training dynamics.

2. **LaaJ evaluation lacks validation.** While using Gemini 2.5 Flash as a judge is reasonable, the paper provides no inter-annotator agreement (between LaaJ and human raters or across multiple judges), no calibration details, and no robustness analysis of the 0–5 scoring scale for Naturalness and Similarity.

3. **Limited model scale and adversarial evaluation.** Experiments are limited to models up to 8B parameters. The paper does not evaluate robustness against jailbreaks, few-shot probing, or fine-tuning attacks — standard evaluations in the unlearning literature that directly test whether "forgotten" knowledge can be recovered.

4. **On-policy vs. off-policy BS-S is underspecified.** The paper mentions both variants but primarily uses off-policy in experiments (the theory only covers off-policy). No empirical comparison between the two is provided, leaving the practical implications unclear.

## Nice-to-Haves

- **Tabulated LaaJ results across all benchmarks and methods** would directly validate the "more thorough forgetting" claim on semantically faithful metrics.
- **A data augmentation baseline for BS-S** (e.g., paraphrase the forget set with another model) would isolate the contribution of targeting model specifically beliefs versus adding diverse forget data.
- **Evaluation on rephrased forget queries** — if spurious unlearning manifests as responding to rephrased inputs, testing directly on rephrased versions of forget prompts would be the most natural validation.
- **Forget-retain trade-off curves** with varying λ would clarify whether BS achieves Pareto-dominant performance or just operates at a different point on the existing frontier.

## Removed Points

These points were flagged and removed with justification:

1. **"GA/NPO as merely spurious — overclaimed"** (from harsh critic): The paper does provide specific evidence (Cases 1–2 and Fig. 2) that GA/NPO exhibit the described failures. While the blanket "merely spurious" characterization may be too strong for *all* NPO settings, the paper's evidence for the specific settings tested is valid. Removed as an overstatement of the criticism; the paper's case studies are genuine failures worth highlighting.

2. **"Missing related works"** (from human finder): Per rules, we do not flag missing related works as we cannot confirm their existence or relevance independently.

3. **"Formatting and notation issues"** (from neutral reviewer's minor notes): Removed as formatting nitpicks per rules.

4. **"Computational overhead of BS-S"** (from neutral reviewer): The paper explicitly mentions "BS-T offers higher efficiency, while BS-S achieves more thorough forgetting" and defers timing to Appx. F.6. This is discussed, just in the appendix. Downgraded to trivial rather than a substantive weakness.

5. **"Scalability to larger forget sets or larger models"** (from human finder): This is a generic "add more experiments" request. The paper tests 1%/5%/10% forget settings and 1B/3B/8B models, which is standard for the field. Downgraded as a soft, generic concern.

## Novel Insights

The squeezing effect framing — that softmax normalization mechanically redirects probability mass into semantically proximate high-likelihood regions, creating a predictable failure mode for gradient-ascent-based unlearning — is genuinely novel and actionable. The insight that the model's own high-confidence predictions serve as "belief" anchors for this redirected mass, and can therefore be targeted directly, represents a useful bridge between the mechanistic diagnosis and the algorithmic solution. However, the conceptual similarity to inverse self-distillation (using model predictions against knowledge rather than for it) tempers the novelty somewhat; the contribution lies more in the diagnosis and the specific application context than in the loss formulation itself.

## Score and Decision

**Calibration anchors:**

- **Q1MHvGmhyT** (Closer Look at Unlearning, scores 6/6/6/6, accepted poster): Identified metric issues and proposed methods. Similar motivation but less novel mechanistic analysis.
- **huo8MqVH6t** (G-effect, scores 6/5/8/3/6/8, accepted poster): Gradient analysis framework for unlearning. Similar level of theoretical contribution, accepted at poster level.
- **Pd3jVGTacT** (SimNPO, scores 6/5/5/5, rejected): Incremental NPO improvement with evaluation concerns.
- **51WraMid8K** (Probabilistic Perspective, scores 6/10/6/10, accepted oral): Novel probabilistic evaluation framework — much stronger contribution than this paper.
- **AcR5Mngp1p** (Knowledge-localized Unlearning, scores 3/5/6/6, rejected): Similar "superficial unlearning" concept but weaker execution and evaluation.

This paper falls between the accepted poster-level papers (G-effect, Closer Look) and the rejected ones (SimNPO, Knowledge-localized). The squeezing effect diagnosis is genuinely valuable and the method is well-motivated, but the evaluation inconsistency (criticizing metrics then relying on them), lack of augmentation baselines, and insufficient semantic validation create meaningful gaps. The methodological contribution is incremental (inverse self-distillation / data augmentation) even though the framing is novel. The theoretical analysis is a positive differentiator but limited by strong assumptions. Overall, this is a borderline paper that offers an important conceptual insight but does not fully validate the claims built around it.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>