=== CALIBRATION EXAMPLE 70 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title captures the core idea well: aligning SFT with ICL activations. The “IA2” naming is memorable, though the title slightly overstates the scope by implying a general improvement to SFT, while the paper’s evidence is mostly from a specific LoRA-based setup on two model families and a handful of classification/math benchmarks.
- The abstract clearly states the problem (SFT vs. ICL), the proposed method (activation alignment/self-distillation), and the main empirical claim (better accuracy and calibration across 12 benchmarks).
- The strongest abstract claim is that IA2 “significantly improves” performance and calibration across 12 benchmarks and two model families. The paper does present many results, but the abstract does not mention important caveats: the gains are often modest, the method depends on collecting ICL activations with extra inference cost, and multi-token results are more mixed than the summary suggests.

### Introduction & Motivation
- The motivation is reasonable and aligned with ICLR interests: understanding and improving the relationship between ICL and SFT is timely, and the paper connects to post-training efficiency and calibration.
- The gap in prior work is stated as: response-only distillation or SFT may miss richer internal signals present in ICL. That is a plausible and potentially valuable framing.
- However, the introduction somewhat over-claims the conceptual novelty. The paper presents “distinct activation patterns” as evidence that ICL and SFT use different mechanisms, but the introduction leans toward a strong causal interpretation before the paper has established that the measured activation differences are functionally decisive.
- The claim that IA2 “drastically improves” adaptation is stronger than what the tables and figures substantiate uniformly. In several settings, SFT already performs competitively, and IA2’s gains are task- and model-dependent.

### Method / Approach
- The method is understandable at a high level: collect ICL activations, match them with a learned model using a mean-squared activation loss (Eq. 4), then continue with standard SFT.
- That said, the description leaves several important methodological questions insufficiently resolved:
  - In Eq. 3, the alignment target is written as a self-attention output match under ICL vs. query-only inputs, but the implementation appears to align stacked activations across layers and output-token positions. It is not fully clear exactly which tensors are matched, at what layers, and whether all layers are equally weighted.
  - The paper says “for every newly generated token” and uses generated ICL response tokens as targets in IA2. This makes the method dependent on the model’s own sampled outputs, which may vary with decoding choices, but the paper does not discuss decoding temperature/strategy or sensitivity to generation stochasticity.
  - There is a logical tension in the claim that IA2 “replicates ICL internal reasoning” while the method uses the model’s own responses and only aligns activations, not token-level behavior. The paper should better justify why activation matching should transfer into downstream supervised adaptation rather than simply regularize the model.
- Key assumptions are not sufficiently justified:
  - That the model’s ICL activations are “information rich” in a way that is generally beneficial for downstream adaptation.
  - That matching ICL activations on the output token positions is enough to capture the relevant functional behavior.
  - That LoRA on Q/K/O matrices is a representative substrate for the claims being made.
- Edge cases/failure modes are only lightly acknowledged:
  - The paper itself notes that extreme alignment can hurt accuracy and that ICL may be wrong, but this is presented more as a post hoc observation than as a principled limitation.
  - The multi-token case is especially fragile: the paper admits the same LoRA capacity may not compress longer contexts well, which suggests the method may not scale to harder generation tasks without substantial changes.
- For a method paper at ICLR, I would want a more explicit connection between the activation objective and the downstream generalization/calibration gains. Right now the causal story is suggestive but not fully convincing.

### Experiments & Results
- The experiments are broad and cover many tasks, model families, and shot counts, which is a clear strength.
- The paper does test its main claim that IA2 → SFT can outperform SFT-only training, and it includes OOD evaluations, calibration for classification, and multi-token generation tasks.
- However, several experimental issues materially affect confidence:
  - **Baseline fairness / selection:** The paper repeatedly selects the best learning rate for each method and configuration, and for IA2+SFT also sweeps β. This is acceptable if done symmetrically, but the resulting comparison space is much richer for some methods than others, especially the combined objectives. The paper needs a clearer accounting of all tuned degrees of freedom and whether the search budget was matched.
  - **Ablation gaps:** The most important missing ablations are:
    - How sensitive IA2 is to which layers are aligned.
    - Whether matching only a subset of activations or only certain token positions would work as well or better.
    - Whether response-text distillation with stronger baselines can close the gap to IA2, beyond the one KD comparison in Table 4.
    - Whether the benefits persist if ICL demonstrations are randomly permuted or if a stronger prompt-selection strategy is used.
  - **Statistical reporting:** The paper reports standard deviations and some t-tests, which is good. But the inferential claims are somewhat overused relative to the variance shown in the tables. Several results have large standard deviations, especially for smaller models and low-data settings, so claims of “drastic” improvements should be tempered.
  - **Evaluation validity for multi-token tasks:** Accuracy is based on parsed exact-match outputs, but the parsing heuristics and custom stop strings may materially affect results. The paper notes this, but the setup is not as standard or robust as using established evaluation harnesses. This is particularly important for claims about math and open-ended generation.
  - **Potential cherry-picking in presentation:** The paper emphasizes configurations where IA2 → SFT is strongest, while some tables show cases where IA2-only, ICL, or even SFT-only can be competitive. The broad claim that IA2 → SFT “outperforms across the board” is too strong as written.
- The main empirical takeaway does seem supported: IA2 often improves over SFT-only, especially on calibration and some OOD settings. But the evidence is less conclusive for generality, scalability, and superiority over strong prompt-based or distillation-based alternatives.

### Writing & Clarity
- The paper is generally understandable and the high-level idea is communicated well.
- The clearest parts are the motivation in the introduction and the broad two-stage pipeline in Section 4.
- The main clarity issue is that the method and evaluation details are scattered and sometimes underspecified:
  - The exact activation objects used in IA2 are not crisply defined.
  - The difference between “IA2 only,” “IA2 → SFT,” and “IA2 + SFT” is conceptually important but not always easy to track across sections.
  - Some claims in Sections 5 and 6 depend heavily on figures/tables that, even discounting parser artifacts, are hard to interpret without more careful description of what is being averaged and over which configurations.
- The figures and tables seem intended to support a large empirical claim, but the narrative occasionally outruns the evidence. In particular, Figure 2/3/4 are used to argue a functional mechanism, yet the paper would benefit from more explicit explanation of what constitutes strong vs. weak support for that mechanism.

### Limitations & Broader Impact
- The paper does acknowledge some limitations:
  - computational overhead for collecting ICL activations,
  - the possibility that some layers are more useful than others,
  - sensitivity to prompt order,
  - potential need for larger LoRA ranks.
- Still, several important limitations are underdeveloped:
  - **Extra inference cost for training data collection:** IA2 is not free; it requires running ICL over the training set, which may be expensive for large models or large datasets. This cost is mentioned but not analyzed quantitatively.
  - **Dependence on the base model’s ICL quality:** If the model’s ICL is weak or miscalibrated, IA2 may encode the wrong signal. The paper notes this in math settings, but the broader implications are not fully discussed.
  - **Reliance on self-generated targets:** Since IA2 uses the model’s own ICL outputs, it may reinforce the model’s existing biases or errors.
  - **Broader impact:** There is little discussion of how activation alignment might affect safety, robustness, or harmful capability amplification. If anything, methods that improve calibration and generalization could also make undesirable behaviors more reliable; this deserves at least brief discussion.
- For ICLR, the limitations section feels somewhat incomplete given the breadth of the claims.

### Overall Assessment
This paper offers an interesting and potentially useful idea: use ICL activations as an internal training signal to improve downstream fine-tuning. The empirical results are broad enough to suggest the idea is not a one-off artifact, and the calibration improvements are particularly compelling. However, the paper’s strongest claims outpace the evidence in several places. The method is not fully specified in a reproducible way at the level ICLR would expect for a mechanistic claim, the experimental comparisons need more careful ablations and stricter framing, and the generality of the gains beyond the chosen models/tasks/LoRA setup remains uncertain. I think the contribution is promising and likely publishable only if the authors substantially tighten the mechanistic justification and narrow or better support the scope of the claims.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes IA2, a two-stage fine-tuning pipeline that first aligns a model’s internal activations under in-context learning (ICL) with those produced by direct input-only execution, and then performs standard supervised fine-tuning (SFT). The central claim is that ICL and SFT solve adaptation through different internal mechanisms, and that matching ICL-like activations before SFT improves both downstream accuracy and calibration, especially in low-data regimes.

### Strengths
1. **Interesting and timely question with a clear motivation.**  
   The paper asks whether the “internal computation” of ICL can be transferred into SFT models, which is a compelling question for ICLR because it connects representation learning, adaptation, and mechanistic understanding.

2. **Empirical evidence that ICL and SFT differ in internal behavior.**  
   The authors measure layerwise activation similarity and show that SFT models have substantially lower similarity to ICL activations, with especially large gaps in middle layers. This supports the paper’s premise that output-level agreement does not imply functional equivalence.

3. **A simple method with broad empirical evaluation.**  
   IA2 is conceptually straightforward: activation matching as a priming stage, then SFT. The paper reports results across many datasets, two model families, and both single-token and multi-token tasks, which is stronger than a narrow benchmark-only demonstration.

4. **Claims go beyond raw accuracy and include calibration.**  
   The paper evaluates expected calibration error on single-token tasks and argues that IA2 → SFT improves not just accuracy but also calibration, which is valuable and aligned with ICLR’s interest in nuanced model behavior.

5. **Some evidence for robustness beyond one training recipe.**  
   The paper includes additional comparisons with different PEFT variants, soft-label distillation, and an alternative unified IA2+SFT objective. These extra experiments help show that the effect is not completely tied to one exact setup.

### Weaknesses
1. **The method’s novelty is somewhat incremental relative to prior distillation/activation-alignment work.**  
   The paper is positioned as using ICL activations rather than response tokens, but the core idea remains a form of representation matching / self-distillation. Related work on distilling context, activation steering, and internalizing ICL already occupies nearby territory, so the paper needs a sharper argument for what is genuinely new beyond “match activations before SFT.”

2. **The mechanistic claim is under-supported.**  
   The paper argues that activation similarity explains improved performance and calibration, but the evidence is mostly correlational. The reported scatter plots and subspace overlap are suggestive, yet they do not establish that IA2 causally induces ICL-like reasoning or that this is the key driver of the gains.

3. **Experimental design is potentially confounded by heavy hyperparameter selection.**  
   For each method the paper trains up to convergence, tries multiple learning rates, and selects the best performing one; IA2+SFT also tunes additional β values. This is reasonable in isolation, but the comparison burden is large and may make the gains less clean than they appear. ICLR reviewers will likely want stronger controls on tuning budget fairness.

4. **The method depends on collecting ICL activations using the same training samples.**  
   This is elegant in spirit, but it means IA2 requires extra inference-time passes with longer contexts and a per-example construction of demonstrations from the remaining data. That makes the practical cost and scalability less clear, especially for larger data or longer-context settings.

5. **Results are uneven across tasks, especially harder generation tasks.**  
   The paper acknowledges that IA2 → SFT does not consistently beat ICL in multi-token reasoning tasks and can lag behind on some settings. That is not fatal, but it weakens the strongest version of the claim that IA2 broadly “improves” SFT in a general sense.

6. **The calibration story is incomplete.**  
   Calibration is only reported for single-token tasks, and the paper does not deeply analyze whether the calibration gains persist across architectures, label spaces, or OOD conditions beyond a few datasets. For ICLR, a stronger analysis of uncertainty behavior would be valuable.

7. **Some comparisons are difficult to interpret due to limited ablation depth.**  
   The paper mentions that selective layer alignment, LoRA rank, target modules, and prompt-order sensitivity could matter, but these are not explored systematically. As a result, it is hard to know whether IA2 itself is essential or whether a more careful tuning of standard SFT/distillation would recover similar gains.

### Novelty & Significance
**Novelty: moderate.** The paper’s main novelty is not a new architecture but a new training signal: align activations produced under ICL, then fine-tune on labels. This is an interesting synthesis of ideas from ICL analysis, representation matching, and self-distillation, but it is adjacent to a substantial body of prior work on context distillation, hidden-state matching, and internalizing prompts.

**Significance: moderate.** The empirical improvements are potentially useful, especially in few-shot adaptation where calibration and data efficiency matter. However, the significance is limited by the lack of a deeper mechanistic validation, the fairly specialized training setup, and the fact that the gains are not uniformly dominant over ICL or across all task types. For ICLR, this feels like a solid applied-method paper, but not yet a clearly breakthrough contribution.

**Clarity: mixed.** The high-level idea is clear, but the presentation sometimes overclaims (“drastically improves,” “conceptual window into inner mechanics”) relative to the evidence. The large number of experiments and tables also makes it hard to isolate the essential result.

**Reproducibility: fairly good.** The paper provides many training details, datasets, model families, and a code repository. That said, exact reproducibility is still somewhat weakened by the extensive model selection process, multiple tuning dimensions, and the complexity of activation collection.

**ICLR fit:** Reasonable but not guaranteed above the bar. ICLR typically favors papers with either a strong conceptual advance, a notably clean and generalizable empirical result, or a deep mechanistic insight. This paper has a useful empirical result and an interesting idea, but it may be seen as somewhat incremental unless the authors can better establish causal mechanism and broader generality.

### Suggestions for Improvement
1. **Add stronger causal evidence that activation alignment drives performance.**  
   For example, test whether partial alignment at specific layers, random activation matching, or matching non-ICL activations fails to reproduce the gains. This would better isolate the role of ICL-specific internal structure.

2. **Expand and sharpen baselines.**  
   Compare against stronger and more directly relevant baselines such as response distillation with matched compute, hidden-state distillation from the same prompts without ICL demonstrations, and layer-selective distillation. This would clarify whether IA2 is truly better than more standard alternatives.

3. **Report a stricter compute- and tuning-normalized comparison.**  
   Provide a table that equalizes search budget across methods, and ideally include a low-tuning or default-hyperparameter setting. This would make the main gains more credible to ICLR reviewers.

4. **Systematize ablations around design choices.**  
   In particular, study which layers matter most, whether the method still works with smaller or larger LoRA ranks, and how sensitive the results are to the choice of ICL demonstrations and ordering.

5. **Analyze when IA2 helps and when it hurts.**  
   A failure analysis across tasks would strengthen the paper substantially. Identify cases where ICL activations are noisy, misleading, or hard to compress, especially in multi-token reasoning tasks.

6. **Deepen the calibration analysis.**  
   Include reliability diagrams, selective prediction, and OOD calibration metrics, not just ECE. This would support the claim that IA2 improves uncertainty quality rather than only point accuracy.

7. **Clarify the practical cost of the method.**  
   Report wall-clock overhead, activation storage cost, and how the method scales with longer contexts and larger datasets. ICLR reviewers often care about whether a method is actually deployable.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add comparisons against the strongest *few-shot-to-finetune* baselines, especially prior work that distills ICL into weights (e.g., Snell et al., Chen et al., Yang et al.) under the same data budget and model families. Without head-to-head results, it is not convincing that IA2 is better than existing context-distillation approaches rather than just another variant.

2. Add a direct comparison to simpler regularizers that could explain the gains, such as activation matching to the base model without ICL, hidden-state matching to random prompts, or self-distillation on the model’s own outputs without activation alignment. This is needed to show the effect comes specifically from ICL activations, not from extra training time, extra loss terms, or better optimization.

3. Add ablations on IA2 design choices: which layers are aligned, which tokens are aligned, whether attention-only is necessary, and whether matching queries/keys/values separately matters. ICLR will expect evidence that the method is not fragile and that the claimed “functional alignment” mechanism is actually responsible for the improvements.

4. Add full-scale results for the multi-token setting across both model families and all relevant shot counts, not just selected settings. The current evidence for generation tasks is too sparse to support broad claims about general improvement across benchmarks and model families.

5. Add a strong baseline for “ICL prompt distillation” using the same ICL responses but without activation matching, ideally with multiple training schedules matched for compute. This is critical because the paper’s main claim is that the activation signal adds value beyond output distillation.

### Deeper Analysis Needed (top 3-5 only)
1. Add analysis separating whether IA2 improves *optimization* or *representation* by controlling for total training compute and number of updates. Right now it is unclear whether the gains come from the activation objective or simply from a better initialization / longer effective training.

2. Add a layerwise and tokenwise attribution analysis showing *where* the ICL-like behavior is transferred and whether those changes persist after SFT. The paper claims “functional alignment,” but does not convincingly show which internal circuits or layers are doing the work.

3. Add error analysis on cases where IA2 hurts accuracy or calibration, especially the OOD and multi-token cases where gains are inconsistent. ICLR reviewers will want to know whether the method is broadly reliable or only helps in easy classification settings.

4. Add a robustness analysis across prompt order, demonstration selection, and random seeds beyond the selected five runs. Since ICL is known to be highly sensitive to prompt order, the method’s dependence on one prompt construction weakens the claim that it learns a stable ICL-like computation.

5. Add a scaling analysis of performance versus N and model size that tests whether IA2’s benefit grows, saturates, or disappears with more data and larger models. The paper currently suggests broad generality, but the reported evidence does not establish a clear scaling law.

### Visualizations & Case Studies
1. Show side-by-side activation maps for ICL, SFT, IA2, and IA2→SFT on the same examples, ideally with token-level and layer-level heatmaps. Without this, the reader cannot verify that IA2 is actually reproducing ICL-like internal computation rather than just changing averages.

2. Provide qualitative case studies where SFT fails but IA2→SFT succeeds, and vice versa, for both classification and generation. These examples would reveal whether the method improves genuine reasoning/calibration or only shifts the prediction distribution.

3. Add a per-layer similarity trajectory before and after SFT, with confidence intervals and task-specific breakdowns. This would expose whether alignment is stable across tasks or driven by a few outlier layers.

4. Visualize calibration by confidence bins and reliability diagrams, not just aggregate ECE. The claim that IA2 improves calibration needs evidence that the improvement is not an artifact of one metric or one dataset.

### Obvious Next Steps
1. Test IA2 on larger and more diverse post-training settings, especially instruction tuning and RL-tuned models. The paper claims conceptual relevance for adaptation generally, but it only demonstrates a narrow LoRA-based setup.

2. Extend IA2 to non-classification tasks beyond the limited generation benchmarks, including open-ended instruction following and multi-step reasoning tasks with standard evaluation protocols. ICLR will expect broader evidence before accepting a general post-training method.

3. Evaluate whether IA2 can be combined with stronger parameter-efficient methods and full fine-tuning, not just LoRA and (IA). The paper claims generality but does not establish that the idea transfers across adaptation mechanisms.

4. Study whether the activation-matching objective can be compressed or approximated more cheaply, since the current method requires collecting ICL activations and extra inference overhead. A practical method needs a clearer cost/benefit analysis.

# Final Consolidated Review
## Summary
This paper proposes IA2, a two-stage adaptation pipeline that first aligns a model’s internal activations under in-context learning (ICL) and then performs standard supervised fine-tuning (SFT). The core claim is that ICL and SFT encode different internal behaviors, and that using ICL activations as a priming signal can improve downstream accuracy and calibration, especially in low-data regimes.

## Strengths
- The paper asks a timely and interesting question: whether the internal computation of ICL can be transferred into fine-tuning, rather than only distilling output tokens. That framing is genuinely relevant for understanding post-training.
- The empirical evidence does support a real gap between ICL and SFT behavior at the activation level, and the authors back this up with broad experiments across two model families, multiple datasets, and both single-token and multi-token settings.
- The method is simple and operationally appealing: collect ICL activations, align a LoRA-adapted model to them, then run SFT. The paper also reports calibration results, which is a meaningful plus over accuracy-only evaluations.

## Weaknesses
- The mechanistic story is much stronger than the evidence. The paper shows correlation between activation similarity and better outcomes, but does not establish that the activation-matching objective is the causal reason the method works. The subspace and similarity analyses are suggestive, not conclusive.
- The evaluation setup is heavily tuned and somewhat hard to interpret cleanly. The authors sweep learning rates, vary β for IA2+SFT, and select the best configuration per setting; this makes the gains less clean than the main narrative suggests, and there is no matched-search-budget baseline that would make the comparison fully convincing.
- The strongest claims are too broad for what is actually demonstrated. IA2→SFT often helps over SFT-only, but results are uneven across tasks, and multi-token generation is notably less consistent. The paper should not imply a near-universal improvement over all adaptation settings.
- The method’s practicality is limited by its dependence on extra ICL inference over the training set and on the base model’s own ICL quality. That cost is acknowledged, but not quantified, and the method may simply encode whatever errors or biases the model already exhibits under ICL.

## Nice-to-Haves
- A more systematic layer/token ablation showing where alignment matters most.
- Stronger baselines for output distillation and hidden-state matching under matched compute.
- A clearer scaling analysis with respect to number of demonstrations, model size, and LoRA rank.

## Novel Insights
The most interesting takeaway is that matching outputs alone is not the whole story: the paper provides evidence that ICL and SFT can land in different parts of activation space even when they solve the same task, and that this gap matters for calibration and generalization. The novel angle is not merely “distill ICL,” but “distill the internal processing pattern induced by ICL before supervised adaptation,” which is a useful conceptual distinction. That said, the paper’s own results also show that excessively strong alignment is not always optimal, which suggests IA2 is better understood as a useful inductive bias than as a faithful replication of ICL reasoning.

## Potentially Missed Related Work
- Snell et al., 2022, *Learning by Distilling Context* — directly relevant as prior context distillation into weights.
- Chen et al., 2024b, *Demonstration Distillation for Efficient In-Context Learning* — relevant output-based distillation baseline.
- Yang et al., 2024, *Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning* — relevant self-distillation/fine-tuning perspective.
- Chen et al., 2025, *Generative Adapter* — relevant to internalizing context into parameters.
- Shen et al., 2025, *Codi* / other activation- or representation-based distillation lines — relevant for comparing against non-output distillation.
- Todd et al., 2023, *Function Vectors in Large Language Models* — relevant mechanistic framing for internal functional directions.

## Suggestions
- Add a compute- and tuning-matched baseline suite, especially response distillation and hidden-state distillation without ICL activation matching.
- Include ablations that isolate which layers and token positions are responsible for the gains.
- Quantify the full overhead of IA2, including activation collection time and storage, and report performance versus this cost.
- Strengthen the causal argument with controlled experiments such as random activation targets, non-ICL prompts, or partial alignment settings.
- Tone down the universal language in the abstract and discussion; the evidence supports a useful improvement in many settings, not a broadly solved adaptation problem.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
