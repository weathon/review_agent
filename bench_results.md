# ICLR Benchmark Results

Date: 2026-04-08 03:22
Critic/Merger: qwen/qwen3.5-plus-02-15 (OpenRouter)
Neutral: qwen/qwen3.5-flash-02-23, Related Work: qwen/qwen3.5-flash-02-23:online (OpenRouter)

## Kw2mvnzCoc

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

##Summary
TSPulse introduces a family of ultra-lightweight (1M parameter) pre-trained models for time-series diagnostic tasks (anomaly detection, classification, imputation, similarity search). Its core contributions are a disentangled masked reconstruction framework that jointly learns temporal, spectral, and semantic embeddings across dual representation spaces and abstraction levels, hybrid masking strategies for pre-training robustness, and lightweight post-hoc fusers (MHT, TSLens) for task specialization. The model achieves strong results across four tasks while being 10–100× smaller and faster than competing pre-trained models.

## Strengths
- **Compelling efficiency–performance trade-off with rigorous benchmarks:** TSPulse at 1M parameters consistently matches or outperforms models like MOMENT (35–340M), UniTS, VQShape, and Chronos on established leaderboards (TSB-AD, UEA). The CPU inference time of 0.06s per batch (vs. 7.6s for MOMENT-base) is a concrete, deployment-relevant advantage well-supported by Table 3.
- **Effective multi-space reconstruction design:** The dual-space (time + FFT) masked reconstruction with separate loss objectives on distinct embedding segments yields consistent gains. Ablations confirm this: removing dual-space learning drops imputation by 8% and classification by 7% (Tables 1b, 1c), and the sensitivity analysis (Table 2) demonstrates that temporal embeddings are phase-sensitive (130% distortion) while semantic embeddings are phase-robust (12%), confirming functionally distinct encoding.
- **Hybrid masking addresses a real pre-training bias:** The paper identifies a concrete problem—models pre-trained with fixed block masking fail under irregular missingness patterns—and provides strong ablation evidence (79% MSE drop when hybrid PT is removed under hybrid evaluation, Table 1c). The MAR/MNAR robustness experiments (Table 31) further substantiate the claim with realistic missingness scenarios.
- **Task-specialized post-hoc fusers add measurable value:** Multi-head triangulation for AD outperforms any single head by 9–16% (Table 1a), and TSLens outperforms standard pooling by 11–16% for classification (Table 1b). These are non-trivial gains from well-motivated architectural choices.

## Weaknesses

### Major:
- **Imputation gains are partially confounded by pre-training/evaluation masking alignment:** The +50% zero-shot imputation improvement over MOMENT reflects not only architectural advantages but also the fact that TSPulse is pre-trained with hybrid masking and evaluated on hybrid masking, while MOMENT was pre-trained with block masking. Table 1(c) confirms this: removing hybrid pre-training causes a 79% MSE increase under hybrid evaluation but only a 7.5% improvement under block evaluation (Table 22), demonstrating that much of the hybrid-masking evaluation gain comes from train-test distribution alignment. TSPulse does still outperform baselines under block masking (Table 19: TSPulse ZS 0.106 vs. MOMENT ZS 0.196), so the architectural advantage is real, but the magnitude advertised in the abstract is inflated by this confound. The paper should more explicitly acknowledge this.

- **Disentanglement claims rely on invariance properties rather than statistical independence:** The sensitivity analysis (Section 6, Table 2) effectively demonstrates that different embedding segments respond differently to perturbations (e.g., temporal embeddings are phase-sensitive while semantic embeddings are phase-robust). However, this establishes *functional specialization* and *invariance properties*, not strict *

---

## Pa6ak2B9jJ

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary

AUTO-RT proposes a reinforcement learning framework for automatic jailbreak strategy exploration that decomposes attack generation into a trainable strategy generation model and a frozen rephrasing model. Two key techniques address sparse-reward challenges: Dynamic Strategy Pruning (DSP), which terminates redundant exploration branches early via constraint checks, and Progressive Reward Tracking (PRT), which uses intentionally safety-weakened "downgrade models" to densify reward signals, with a First Inverse Rate (FIR) metric guiding downgrade model selection. Experiments across 16 white-box and 2 black-box LLMs demonstrate improvements over RL baselines in attack success rate, strategy diversity, and defense generalization.

## Strengths

- **Hierarchical strategy–rephrasing decomposition (Section 2.2):** Separating high-level strategy generation ($AM_g$) from low-level query instantiation ($AM_r$) is a meaningful architectural contribution that enables strategies to generalize across toxic intents rather than overfitting to specific prompts. This is evidenced by the consistent SeD improvements across nearly all models in Table 1.

- **Progressive Reward Tracking with downgrade models (Section 2.3.3):** Using safety-weakened intermediate models to densify sparse binary reward signals is a creative and practically effective solution. The shaped reward $R_s \in \{0, 1, 2\}$ provides graded feedback that vanilla RL lacks. The empirical validation in Figure 4—showing that FIR-guided selection consistently identifies productive downgrade models across six model families—lends credibility to the approach.

- **Multi-dimensional evaluation beyond ASR:** The paper evaluates effectiveness (ASR_tst), semantic diversity (SeD), and defense generalization diversity (DeD), providing a more complete picture of red-teaming capability than single-metric studies. The DeD metric, despite its limitations (see weaknesses), captures a practically important dimension—sustained attack capability under defense updates.

- **Broad experimental coverage:** Testing across 16 white-box models from 6 families, 2 black-box models, and 3 commercial APIs (Appendix G) provides substantial breadth, and the ablation in Table 2 cleanly isolates DSP and PRT contributions across all models.

## Weaknesses

### Major:

- **Inconsistent framing relative to AutoDAN comparison:** Table 3 reveals that on the aggregate ASR metric, AutoDAN achieves 55.23% while AUTO-RT achieves 38.38%—a substantial gap. The abstract claims AUTO-RT "significantly improves success rates (by up to 16.63%)," but this figure appears to be the average improvement over the RL baseline only (which can be verified by computing the average per-model improvement over RL from Table 1: ≈16.63 pp). The abstract does not specify this is versus RL rather than versus all existing methods. Meanwhile, Table 1 excludes AutoDAN entirely, and Section 3.3.3 describes AUTO-RT's ASR as merely "high" despite being 17 pp below AutoDAN. This selective presentation undermines the core effectiveness claim. The paper's genuine strength is in diversity and defense generalization (DeD: 38.19 vs. AutoDAN's 17.88), but the framing obscures the ASR tradeoff.

- **Missing comparison with widely-used adaptive red-teaming methods:** PAIR and TAP are discussed in Related Work as representative adaptive methods using textual feedback, but neither appears in any comparison table. Given their prominence in the red-teaming literature and their claimed advantages over template-based approaches, their exclusion is a significant gap for a paper asserting superiority over "existing methods." Without this comparison, it is unclear whether AUTO-RT's strategy-level exploration offers advantages over iterative prompt-refinement approaches.

- **Unquantified computational cost of downgrade model construction:** PRT requires creating a spectrum of downgrade models ($TM'_1, \ldots, TM'_n$) for each target model. For 16 white-box models, this implies fine-tuning many model instances (the paper uses 6 downgrade levels per target, per Figure 4). The total GPU cost of this setup phase is never reported, making it impossible to assess the efficiency claims ("accelerates discovery") against methods like PAIR or TAP that require no model fine-tuning. The 8×A100 cost quoted in Section 3.1 covers only AM_g optimization, not the downgrade model pipeline.

### Minor:

- **DeD metric's defense construction is underspecified:** Section 3.1 defines Defense Generalization Diversity as evaluating ASR after "constructing defenses based on the successful attacks," but does not specify what defense mechanism is used (adversarial fine-tuning? input filtering? safety training on successful attacks?). Different defense mechanisms would yield qualitatively different DeD scores, making this metric hard to interpret or reproduce without a standardized protocol.

- **ASR_tst computed on top-100 strategies biases toward best-case performance:** Equation 6 evaluates only the top-100 strategies by training-set ASR, which measures peak rather than average policy quality. A method that occasionally discovers a highly effective strategy but produces mostly ineffective ones would score well. This is not necessarily wrong, but it should be acknowledged as measuring upper-bound rather than expected performance.

- **Non-potential-based reward shaping risks policy divergence (Section 2.3.3):** The authors acknowledge that PRT "does not follow the potential-based function structure" of Ng et al. (1999), meaning the shaped reward could in principle change the optimal policy. The empirical results suggest this is not a practical issue, but no convergence analysis or stability study is provided. For models where the downgrade model's safety distribution deviates significantly from the target's, the shaped reward could actively mislead optimization.

- **R2D2 and Mistral counterexamples downplayed:** On R2D2, Few-Shot achieves 27.18% ASR vs. AUTO-RT's 12.45%; on Mistral 7B, Imitate Learning achieves 54.88% vs. AUTO-RT's 52.65%. The paper mentions R2D2's robustness but does not analyze what makes AUTO-RT less effective on these models, limiting understanding of the method's boundary conditions.

### Trivial:

- **FIR definition could be clearer:** The "inverse element" terminology (Section 2.3.3) is somewhat opaque. A simpler explanation—FIR identifies the degradation level where safety collapse becomes erratic rather than monotonic—would improve accessibility without loss of precision.

## Nice-to-Haves

- Ablation isolating FIR-guided downgrade selection vs. random or fixed downgrade model selection, to directly validate FIR's contribution beyond PRT's general mechanism.
- Per-category ASR breakdown on HarmBench (e.g., violence, misinformation, cybercrime) to reveal whether AUTO-RT's gains are uniform or concentrated in specific vulnerability types.
- Computational cost breakdown (GPU hours for downgrade model creation vs. RL training) to substantiate efficiency claims.
- Comparison with PAIR or TAP on the same benchmark, even if only on a subset of models.
- Analysis of strategy novelty: clustering generated strategies against known jailbreak templates to verify that AUTO-RT discovers genuinely new attack patterns rather than rephrasing known ones.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Equation formatting issues** (from Harsh Critic) — These are OCR/parser artifacts, not paper problems. Removed per formatting nitpick rule.
- **Weakness: Missing related works (specific unnamed methods)** (from Spark Finder) — Cannot confirm existence of uncited works. Removed per hard rule on missing related works.
- **Weakness: Exact hyperparameters for downgrade model fine-tuning** (from Harsh Critic, reproduced by others) — This is a reproducibility nitpick about implementation details impractical to include in a submission. Removed per hard rule.
- **Weakness: Ethical concern about reproducing harmful outputs in case studies** (from Review 2) — Standard practice in red-teaming papers; demonstrating effectiveness requires showing actual outputs. This is not a paper-specific weakness.
- **Weakness: "Circular evaluation" using LLM judges** (from Harsh Critic's transferred points) — Using LlamaGuard for both reward and evaluation is standard in this field; the paper partially addresses this in Appendix C.1 with an alternative classifier showing stable results. The concern is generic to the entire field rather than specific to this paper.
- **Weakness: Demand for confidence intervals / statistical tests** (from Spark Finder) — Large-scale RL benchmarks in this community typically report single-run results; demanding statistical testing is not the field's standard. Moved to nice-to-have territory.

## Novel Insights

The hierarchical strategy–rephrasing decomposition reveals an interesting asymmetry: strategies that are individually mediocre can become highly effective when composed with intent-specific rephrasing, which explains why AUTO-RT's diversity advantage (SeD) translates more reliably into defense generalization (DeD) than into raw ASR. This suggests that the red-teaming community's focus on single-attack success rates may be measuring the wrong thing—what matters for practical vulnerability assessment is the breadth of the attack surface (how many distinct strategies work), not the depth of any single attack. The FIR metric's identification of a "sharp transition" in model safety degradation also hints at a phase-transition-like phenomenon in safety alignment that deserves further theoretical study.

## Suggestions

- **Reframe the abstract and claims to be precise about what AUTO-RT improves over:** State explicitly that the 16.63% average improvement is over the RL baseline, acknowledge AutoDAN's higher raw ASR, and foreground the diversity/defense-generalization advantages as the primary contribution.
- **Add a PAIR or TAP comparison** on at least 4–6 models, even as a supplementary result, to situate AUTO-RT against the most relevant adaptive baselines.
- **Report total computational cost** including downgrade model construction, so readers can assess the practical efficiency tradeoff.
- **Specify the defense mechanism used in DeD evaluation** (even a one-line description in Appendix) so the metric is reproducible.
- **Analyze the R2D2 and Mistral counterexamples** to identify boundary conditions where strategy-level exploration is less effective than direct prompt-level methods.

---

## 1EdAn5gMVv

- GT: Reject (avg 5.0)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

SpatialBoost proposes a framework to enhance the spatial awareness of pre-trained vision encoders (e.g., DINOv3, SigLIPv2) by converting dense 3D spatial information from images into multi-turn chain-of-thought linguistic expressions, then injecting this knowledge via LLM-guided fine-tuning with a dual-channel attention mechanism that preserves pre-trained capabilities. The method is evaluated across depth estimation, segmentation, 3D scene understanding, robot learning, classification, and retrieval, showing consistent improvements.

## Strengths

- **Comprehensive empirical validation across diverse task families**: The paper demonstrates gains not only on spatial tasks (depth estimation RMSE: 0.31→0.25 for DINOv3 on NYUd; SQA3D: 51.4→54.9; Geometric Understanding RR@0.05m: 86.9→97.5%) but also on tasks not explicitly requiring spatial understanding (ImageNet linear probing: 88.4→90.2%), establishing that the method does not overfit to spatial features. The breadth of evaluation—covering dense prediction, 3D-centric benchmarks (Lexicon3D), robotic control (CortexBench), and instance recognition—is unusually thorough.

- **Dual-channel attention effectively mitigates catastrophic forgetting**: The ablation in Table 17 and Figure 6 provides clear evidence that full fine-tuning drops ImageNet classification from 86.3% to 79.5%, LoRA drops to 81.7%, while dual-channel attention maintains 87.6% *and* improves depth estimation. This is a practical and well-validated solution to the specific problem of preserving general visual knowledge while adding spatial specialization.

- **Hierarchical CoT reasoning order is empirically validated**: Table 7 shows that forward ordering (pixel→object→scene) outperforms reversed (0.34 vs 0.35 RMSE on depth) and random ordering, and Table 15 provides a detailed breakdown of which levels contribute most. This is not merely an intuitive design choice—it is substantiated by controlled ablations.

## Weaknesses

### Major:

- **Potential data leakage with ScanNet**: The multi-view VQA dataset construction (Appendix C) explicitly uses ScanNet (Dai et al., 2017) images to generate training QA pairs. Table 3 evaluates on ScanQA, SQA3D, and ScanRefer—all ScanNet-derived benchmarks. The paper does not state whether the ScanNet images used for training data generation are disjoint from the evaluation splits of these benchmarks. If there is overlap in scenes or images, the dramatic gains on 3D-centric tasks (e.g., SigLIPv2 3D SU mIoU jumping from 9.2 to 55.5, or OpenCLIP GU RR@0.05m from 22.6% to 78.8%) could be substantially inflated by memorization of scene layouts rather than genuine spatial reasoning generalization. This must be clarified; if overlap exists, these results are not trustworthy.

- **Table 6 ablation does not control for supervision head capacity**: The comparison between LLM (Qwen-7B) and Linear/SAM decoders as supervision heads conflates modality (language vs. pixel) with model capacity. The LLM has orders of magnitude more parameters than a linear layer or SAM decoder. A 7B-parameter decoder naturally provides richer gradient signals during backpropagation to the vision encoder. Without controlling for parameter count (e.g., comparing against a similarly-sized non-language decoder or analyzing gradient magnitudes), the conclusion that "language provides superior dense information transfer" is not well-isolated from "larger supervision networks provide better gradients." This undermines one of the paper's central claims.

### Minor:

- **The mechanism for improved 2D classification remains unexplained**: SpatialBoost improves ImageNet linear probing by 1.8% for DINOv3. The paper attributes this to dual-channel attention preserving pre-trained knowledge and scene captions providing general knowledge, but this does not explain why adding *spatial* specialization would *improve* a task where spatial reasoning is not the primary signal. An ablation isolating the spatial reasoning turns from the general scene caption turns would clarify whether the gain is from spatial understanding or incidental language alignment—this is partially addressed by Table 15 but not in a way that cleanly disentangles the two signals for classification specifically.

- **Dual-channel attention initialization at α=0.5 may destabilize early training**: With zero-initialized **a**, α = sigmoid(0) = 0.5, meaning the newly introduced Attn⁺ branch contributes 50% of the attention output from the very first training step. This is unusual compared to parameter-efficient methods like LoRA that initialize adaptation branches near-zero to start from the identity function. While the empirical results show this works, the choice lacks justification and seems risky—it may require careful learning rate tuning (which the paper acknowledges searching for) to avoid early destabilization.

- **No explicit discussion of limitations or the ceiling imposed by teacher models**: The paper does not discuss the computational overhead of the 3-stage pipeline (especially Stage 2 LLM fine-tuning) or the fundamental performance ceiling: SpatialBoost can only inject spatial knowledge as good as the upstream models (Depth-Pro, VGGT). Appendix F.5 shows marginal VFM-vs-GT differences on one dataset, but this does not establish robustness across domains where teacher models may fail (e.g., outdoor scenes, extreme viewpoints). A honest limitations section would strengthen the paper.

### Trivial:

- The paper uses "visual encoder" and "vision encoder" interchangeably throughout.

## Nice-to-Haves

- Comparison with spatial-specific baselines that also enhance spatial understanding (e.g., SpatialVLM applied as an encoder enhancement) to position the contribution more precisely.
- Ablation on LLM size (e.g., 3B vs. 7B) to determine whether spatial knowledge transfer scales with decoder capacity.
- A control experiment using the same data volume but non-spatial QA pairs, to isolate the contribution of *spatial* knowledge specifically versus general supervised training.
- Analysis of the learned α distribution across layers and training steps to verify that dual-channel attention behaves as claimed (gradual incorporation rather than abrupt switching).
- Failure case analysis showing where SpatialBoost underperforms the baseline.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Freezing contradiction in Section 4.2"**: The critic claimed Section 4.2's statement "All experiments freeze the visual backbone during training" contradicts Stage 3 fine-tuning. In context, Section 4.2 clearly refers to the downstream evaluation protocol (training linear/DPT probes on frozen features), not the SpatialBoost training itself. This is a misreading.

- **"Reproducibility concerns about API costs for GPT-4o"**: Flagging proprietary API usage as a reproducibility concern is not a valid weakness for this venue—using GPT-4o for data generation is standard practice, and the paper provides prompt templates in Appendix C.

- **"Broader impact / robot safety discussion missing"**: Demanding a broader impact discussion about robot safety for a representation learning paper is scope creep. The paper's contribution is about visual features, not deployed systems.

- **"Missing comparison to direct 3D pre-training"**: The paper explicitly scopes its contribution as enhancing existing 2D encoders without requiring large-scale 3D pre-training data. Requesting comparison to an entirely different training paradigm is outside the stated scope.

- **"Dataset scalability only tested in limited range"**: Figure 5 tests 50K–300K, which is a reasonable range given the computational constraints and the authors' resources. Requesting larger scales is a generic one-size-fits-all weakness.

- **"Formatting/style nitpicks"**: Removed per hard rules.

## Novel Insights

The hierarchical CoT reasoning structure reveals an interesting asymmetry: pixel-level QA disproportionately helps dense prediction tasks (depth, segmentation) while object-level QA disproportionately helps classification (Table 15). This suggests that different granularity levels of spatial supervision target different representational subspaces in the encoder—a finding that could inform future work on targeted spatial curriculum design. Additionally, the dramatic jump in SigLIPv2's 3D semantic understanding (6.9→55.5 mIoU in Table 3) compared to already-spatial-aware encoders like DINOv3 (69.1→70.6) suggests that language-aligned encoders have a uniquely large "spatial deficit" that linguistic injection can fill, possibly because their text-aligned features are already structurally compatible with language-guided spatial descriptions.

## Suggestions

- Explicitly state whether ScanNet images used in training data generation are disjoint from the ScanNet evaluation splits of ScanQA/SQA3D/ScanRefer. If there is overlap, re-run evaluation on held-out scenes; if there is no overlap, add a clarifying sentence in Section 3.2 or Appendix D to preempt this concern.

- In Table 6, add a capacity-controlled comparison: either use a non-language decoder of comparable parameter count to the LLM, or analyze gradient magnitudes/norms from different heads to disentangle capacity effects from language modality effects.

- Add a 2–3 sentence limitations paragraph discussing computational cost and the performance ceiling imposed by teacher model quality.

- Run a clean ablation on the classification improvement: train with multi-turn spatial reasoning data *without* the final 2 scene caption turns, and report ImageNet accuracy. This would clarify whether the classification gain comes from spatial reasoning or general language alignment.

---

## sJxBWDc8SM

- GT: Reject (avg 3.5)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

This paper empirically investigates why State Space Models (SSMs) underperform Transformers on associative recall and copying tasks, arguing that the gap is primarily driven by optimization instability (extreme learning rate sensitivity) rather than fundamental expressivity limitations. Through extensive hyperparameter sweeps (~3,000 runs, ~20,000 GPU hours), the authors show that SSMs require a much narrower learning rate window than Transformers, that prior benchmarks confounded expressivity with suboptimal tuning, and that the two architectures exhibit opposite scaling behaviors (SSMs favor width, Transformers favor depth). Targeted ablations identify 1D convolution as the critical enabler for single-layer recall in both Mamba and augmented Transformers.

## Strengths

- **Corrective empirical contribution with rigorous scale:** The paper convincingly demonstrates that prior MQAR evaluations (Arora et al., 2023) were confounded by coarse learning rate grids. Figure 1 and Figure 2 together provide strong evidence: the dashed vertical lines from Arora et al.'s grid clearly miss the narrow success windows of Mamba and Hyena, and the finer grid (solid orange) recovers performance. This is a substantive corrective finding backed by 3,000+ runs with 5 seeds, directly challenging how the community evaluates SSM capabilities.

- **Clean ablation identifying convolution as the key architectural driver:** Table 2 is a high-value result. Showing that `Mamba - conv1d` drops to 2% while `Attention + Conv` reaches 99% isolates local mixing as the critical component for single-layer recall, reframing the debate from "Attention vs. SSM recurrence" to "local vs. global mixing." This is the kind of mechanistic insight that shifts how researchers think about architectural design.

- **Contrasting scaling behaviors clearly demonstrated:** Figure 4 effectively shows that parameter count alone is insufficient—it is the scaling axis (width vs. depth) that matters, with SSMs benefiting from width and Transformers from depth. Table 1 provides a concrete parameter-matched demonstration (deeper Mamba fails, wider Mamba succeeds at the same parameter count), making the point unambiguous.

## Weaknesses

### Major:

- **Overstated central claim creates internal tension with own results:** The abstract states "Transformers differ from SSMs not in terms of expressive power but mainly because of their optimization dynamics." However, the paper's own Section 4 shows that 1-layer Transformers fundamentally fail MQAR regardless of tuning (Figure 3), while 1-layer SSMs succeed with sufficient width. This is a genuine expressivity difference in the shallow regime—not an optimization issue. The paper acknowledges this factually but the abstract's framing elides it. The more accurate claim, supported by the paper's own evidence, is: optimization instability is the primary barrier for *deep* SSMs, while architectural composition depth is the barrier for *shallow* Transformers. These are distinct phenomena, and conflating them weakens the paper's narrative coherence.

- **Claim of "induction head formation" in 1-layer Transformers lacks mechanistic support:** Section 6 interprets a loss bump in 1-layer Transformer training as "reminiscent of the induction head phenomenon" and hypothesizes an "attempt" to form them. However, induction heads as defined by Olsson et al. (2022) mechanically require two layers (one to copy previous-token information, one to attend based on that copy). A 1-layer model cannot implement this circuit. Observing a loss bump without any mechanistic evidence (e.g., attention pattern visualization showing the copy-then-attend structure) is insufficient to invoke the induction head framework. The bump could reflect any optimization transient. The paper uses hedging language ("reminiscent," "hypothesize"), but the claim still risks overinterpretation that could mislead readers familiar with the specific mechanistic definition.

### Minor:

- **Initial comparisons conflate two architectural differences:** The primary comparisons in Sections 3–5 pit standard Mamba (which includes a 1D convolution) against standard Attention (which does not). The convolution's critical role is only revealed in Section 7. This sequencing means early readers may attribute performance differences entirely to the recurrence mechanism when convolution is a confounding variable. A brief note in Section 3 flagging this architectural asymmetry would improve interpretability before the full ablation.

- **No quantification of learning rate window narrowness:** The core claim about narrow LR windows is presented qualitatively through visual inspection of figures. A simple metric—e.g., the ratio of successful learning rates to the total search range, or the width of the success interval in log-space—would make the comparison between architectures rigorous and reproducible rather than impressionistic.

- **Practical cost of SSM tuning not discussed:** The paper demonstrates that SSMs require much finer learning rate searches to find the narrow success window, but does not discuss what this implies for practical training budgets. If SSMs need 10× the hyperparameter search cost to achieve comparable performance, this partially offsets their per-step training efficiency advantage. This is directly relevant to the paper's stated goal of guiding practitioners.

### Trivial:

- The "relative max-min errors" metric used in several figures is less standard than accuracy percentage and could benefit from a clearer definition in the main text (it is explained in the appendix).

## Nice-to-Haves

- **Gradient norm / loss landscape analysis:** The paper hypothesizes that vanishing/exploding gradients drive the optimization brittleness, but never measures gradient statistics or loss curvature. Even a simple plot of gradient norms across training steps for SSMs vs. Transformers would substantiate the mechanistic explanation.

- **Optimizer ablation:** All experiments use AdamW. Testing whether the instability persists with alternative optimizers (e.g., Lion, different gradient clipping strategies) would clarify whether this is an architecturally inherent property or an optimizer-architecture interaction.

- **Downstream language modeling validation:** The paper acknowledges this limitation. A small-scale LM experiment (e.g., WikiText-103 or a C4 subset) testing whether the LR sensitivity observed on MQAR transfers to perplexity would significantly strengthen the practical relevance.

- **Investigation of initialization schemes:** The paper references Trockman et al. (2024) on mimetic initialization but does not test whether better initialization narrows the LR window gap. This is a practical intervention that could directly address the identified problem.

- **Hybrid architecture exploration:** Given that convolutions fix 1-layer Attention and that DeltaNet shows improved stability, testing whether hybrid SSM-Attention models inherit the best of both properties is the natural next step the paper hints at but does not pursue.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: "Instability appears inherent rather than a training artifact" (from harsh critic's transferable weaknesses).** This actually *supports* the paper's thesis rather than weakening it. The paper's central argument is precisely that SSMs have inherent optimization instability—not that it's a mere training artifact. Framing this as a weakness misunderstands the paper's position.

- **Weakness: "Incomplete treatment of state capacity limitations" citing "Stuffed Mamba."** The paper explicitly confirms that hidden state size matters (width scaling results) and does not claim state capacity is irrelevant—its argument is that optimization is the *primary* confounder in prior evaluations, not the *only* factor. Demanding a full treatment of state capacity is scope creep beyond the paper's stated contribution.

- **Weakness: "Limited discussion of architectural trade-offs between efficiency and learnability."** This asks the paper to address a broader design philosophy question that is outside its scope. The paper documents and characterizes the optimization gap; proposing architectural solutions to resolve the efficiency-learnability trade-off is a different research direction.

- **Weakness: "Proper baseline replication with original hyperparameters—the paper doesn't show what LR Arora et al. used."** This is factually wrong. Figure 1 explicitly shows Arora et al.'s learning rate grid as dashed vertical lines, making it visually clear that their grid missed the optimal region for SSMs.

- **Weakness: "Limited exploration of whether architectural modifications are sufficient long-term solutions" citing VMamba vision results.** This imports concerns from a different domain (vision) and a different paper. The current paper's scope is synthetic sequence benchmarks; extrapolating to vision is outside scope.

- **Strength: "Paper is well-written" and "topic is important."** These are generic strengths that apply to many papers at ICLR and are removed per the rules.

- **Nitpick: Grammar issues ("Attention exhibit" → "Attention exhibits").** Removed as a formatting/style nitpick per rules.

## Novel Insights

The most striking insight from the reviews and the paper itself is the **reframing of the SSM vs. Transformer debate from expressivity to learnability—and the way the paper's own results complicate this reframing.** The paper's strongest finding (Table 2) is that removing the 1D convolution from Mamba makes it perform identically to 1-layer Attention on MQAR, while adding convolution to Attention makes it perform identically to Mamba. This effectively reduces the "SSM advantage" in the single-layer regime to the presence of local mixing, not the recurrence mechanism itself. This suggests the community's expressivity comparisons may have been comparing not "attention vs. state-space recurrence" but "global-only mixing vs. global+local mixing"—a much less fundamental architectural distinction, and one that is straightforwardly addressable. Meanwhile, the persistent narrow LR window even in S6+MLP (which solves MQAR at 98%) indicates that the optimization brittleness is genuinely tied to the recurrent computation structure, not to auxiliary components. This creates a clean separation: convolutions explain the expressivity gap, while the recurrent structure explains the learnability gap.

## Suggestions

- **Rewrite the abstract and conclusion to distinguish the two distinct findings:** (1) For *deep* models, optimization instability is the primary differentiator. (2) For *shallow* models, architectural composition (depth for Transformers, convolution for SSMs) is the primary differentiator. Conflating these weakens the paper's otherwise strong empirical case.

- **Add a quantification of LR window width** (e.g., fraction of log-LR range yielding >90% accuracy) as a single number per architecture in Figure 1 or its caption, to make the "narrowness" claim precise and citable.

- **Soften the induction head claim in Section 6** by either (a) adding attention pattern visualizations before/after the loss bump, or (b) reframing as "an optimization transient whose mechanism we leave to future work" rather than invoking the induction head framework without the requisite mechanistic evidence.

- **Add a sentence in Section 3** noting that Mamba includes a causal convolution absent from the standard Attention baseline, flagging this as a known architectural asymmetry that is dissected in Section 7.

---

## b6qQmQ2F13

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

This paper systematically investigates memory-accuracy trade-offs for deploying reasoning models under fixed memory budgets, studying over 1,700 configurations across the Qwen3, DeepSeek-R1-Distill, and OpenReasoning-Nemotron families. It identifies a scale-dependent threshold (around 8-bit 4B effective size) below which allocating memory to larger/higher-precision weights outperforms longer generation, and above which the opposite holds. It further shows that 4-bit quantization is memory-optimal for knowledge-intensive tasks but suboptimal for mathematical reasoning and code generation, and that KV cache eviction outperforms KV cache quantization for small models while both are competitive for larger ones.

## Strengths

- **Comprehensive empirical design covering multiple interacting dimensions.** The study varies five key factors (model size, weight precision, token budget, parallel scaling, KV cache compression) across 1,700+ configurations, three model families, and four benchmarks. This scope enables principled conclusions about when different strategies dominate, rather than single-point comparisons. The Pareto frontier framing makes the trade-offs immediately interpretable.

- **Actionable scale-dependent threshold that refines prior prescriptions.** The finding that models below ~8-bit 4B effective size should prioritize weight capacity over test-time compute directly challenges the prevailing assumption that 4-bit quantization is universally memory-optimal (Dettmers & Zettlemoyer, 2023). This threshold is supported by consistent evidence across Qwen3 (Figure 1–2) and validated on DeepSeek-R1-Distill (Figure 6) and OpenReasoning-Nemotron (Figure 16).

- **Task-specific precision insights that go beyond prior work.** The demonstration that 4-bit weights are memory-optimal for GPQA-Diamond (knowledge-intensive) but 8-/16-bit weights dominate for AIME25 and LiveCodeBench (math/code) provides a nuanced correction to scale-agnostic quantization guidelines. This aligns with and extends concurrent findings (Li et al., 2025a; Liu et al., 2025b) by embedding them in a memory-budget framework.

- **KV cache compression analysis integrated into the deployment trade-off.** Rather than treating weight quantization and KV cache compression independently, the paper shows that both eviction and quantification advance the Pareto frontier (Figure 8), and identifies that eviction dominates for small effective sizes while quantization becomes competitive for larger models (Figure 9).

## Weaknesses

### Major:

- **The "effective size" metric conflates parameter count and precision, obscuring the mechanism behind the threshold.** The key finding is organized around "effective size" (parameters × bits per weight), but this single aggregate hides whether the threshold is driven by having more parameters at lower precision or fewer parameters at higher precision. For instance, a 32B model at 4-bit and an 8B model at 16-bit have similar effective sizes but very different properties. Without disentangling these factors, the 8-bit 4B threshold could be an artifact of how these two dimensions interact in the specific model families tested, rather than a principled boundary. The paper would be significantly stronger with an ablation that varies parameter count and bit-width independently while holding effective size constant.

- **The choice of HQQ for KV cache quantization may not represent the true Pareto frontier, potentially biasing the eviction-vs-quantization comparison.** HQQ is primarily a weight quantization method; specialized KV cache quantization methods like KIVI (Liu et al., 2024, cited in the paper) are designed to handle the asymmetric importance of keys versus values and the online nature of KV caching. If a more capable KV quantizer narrows the gap with eviction, then finding #5 ("eviction is more effective than quantization for small models") may partially reflect the choice of suboptimal KV quantization rather than a fundamental trade-off. The paper should either justify this methodological choice or acknowledge it as a limitation that affects the strength of this particular conclusion.

### Minor:

- **Lack of mechanistic explanation for the threshold.** The 8-bit 4B threshold is empirically identified but not explained. Is it related to attention head capacity, activation magnitude distributions, or numerical properties of specific layers? Without understanding why this threshold exists, it remains unclear whether it will shift with architectural improvements or apply to future model families with different designs.

- **The "reasoning-specific" nature of the findings is not empirically isolated from generation length.** The paper claims that reasoning models require different memory strategies than non-reasoning models, but the key differentiator may simply be long generation length rather than reasoning per se. A long-context summarization or document QA task with similar token budgets might exhibit comparable KV cache dominance. The comparison to prior non-reasoning work (e.g., Dettmers & Zettlemoyer, 2023) uses different evaluation protocols and shorter contexts, making it difficult to attribute the difference to reasoning versus generation length. The paper acknowledges testing only on "challenging benchmarks representing complementary difficulty profiles" (Section 3) but does not include a long-context non-reasoning control.

- **Budget forcing may introduce artifacts that vary across precision levels.** The paper uses "Wait" prompt injection to extend generation beyond natural stopping points (Section 3). While standard practice (Muennighoff et al., 2025), this technique can cause looping or hallucination, and the paper itself notes non-monotonic accuracy on MATH500 (Appendix C.4). If forced generation degrades more severely at lower precision, the serial scaling comparisons could systematically understate the value of longer generation for 4-bit models, biasing the threshold identification. An analysis of whether budget forcing artifacts interact with precision would strengthen confidence in the threshold.

- **Generalizability beyond 32B parameters is untested.** The largest model evaluated is Qwen3-32B. At 70B+ scales, KV cache dominance becomes even more pronounced, and it is unclear whether the 8-bit 4B threshold shifts, whether 4-bit quantization remains suboptimal for math, or whether the eviction-vs-quantization trade-offs change qualitatively.

- **Latency and throughput trade-offs are relegated to the appendix.** For a paper offering deployment guidelines, the latency analysis (Appendix C.1) is directly relevant—particularly for parallel scaling, where the memory savings come at a substantial time cost. The finding that 4-bit precision is "never on the Pareto frontier for any model size" in latency-accuracy trade-offs (Appendix C.1) is an important qualifier to the main paper's memory-centric conclusions and deserves more prominent discussion.

### Trivial:

- The term "scale" in the subtitle is ambiguous—it could refer to model size, memory budget, or generation length. Given that the paper's central thesis is about the interaction of these different notions of scale, slightly more precise language would aid initial comprehension.

## Nice-to-Haves

- A practitioner-oriented decision table mapping (task type, memory budget) → recommended configuration, distilling the 1,700+ configurations into actionable rules.
- Qualitative error analysis showing what specifically breaks under 4-bit quantization for mathematical reasoning (e.g., arithmetic errors vs. logical errors vs. planning failures), which would provide mechanistic insight into the precision sensitivity finding.
- Evaluation on long-context non-reasoning tasks at similar token budgets to isolate whether the observed effects are reasoning-specific or generation-length-specific.
- Comparison with a specialized KV cache quantization method (e.g., KIVI) to validate the eviction-vs-quantization findings.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Table 1 memory discrepancy (Qwen3-4B at 16-bit ≈ 4.19 GB vs. expected ~8 GB):** The reviewer estimated 4B parameters × 2 bytes ≈ 8 GB, but model names are approximate and modern architectures commonly use tied embeddings or parameter sharing, reducing stored parameters. The KV cache values in Table 1 are internally consistent with the architectural details in Table 2 (e.g., Qwen3-4B: 144 KB/token × 2000 tokens ≈ 0.27 GB). The reported weight sizes likely reflect actual stored parameters rather than a computation error.

- **Missing non-reasoning baseline experiments (e.g., MMLU at long context):** The paper explicitly scopes its contribution to reasoning models and compares against established findings for non-reasoning models (citing Dettmers & Zettlemoyer, 2023; Chee et al., 2023). Running full non-reasoning baselines at matched token budgets would be valuable but is scope creep for this paper.

- **Reproducibility concerns about undisclosed hyperparameters:** The paper provides detailed inference specifications (temperature, budget forcing protocol, quantization settings) and links to code. The concern about missing implementation details is not substantiated.

- **Confidence intervals on Pareto frontier plots:** The paper reports accuracy averaged over 32 generations per instance. Adding confidence intervals to Pareto frontier curves is not standard practice for this type of analysis and would add visual clutter without changing the qualitative conclusions.

- **Hardware specificity of thresholds:** The memory-accuracy trade-offs studied are fundamentally about memory allocation, which is hardware-independent. While latency/throughput would vary across hardware, the core memory findings are not GPU-specific.

- **Missing related works:** Per the hard rules, I cannot confirm the existence of specific uncited works.

- **Formatting/style nitpicks:** Terminology consistency and figure presentation issues are minor and do not affect the core contributions.

- **QAT (Quantization-Aware Training) baselines:** The paper explicitly scopes to post-training quantization methods, which is the standard deployment scenario. Including QAT is outside the stated scope.

## Novel Insights

The paper's most striking finding is the *task-dependent inversion* of the 4-bit optimality rule: for knowledge-intensive tasks, the established wisdom that 4-bit is memory-optimal still holds, but for mathematical reasoning and code generation, 8-/16-bit weights are more memory-efficient even when accounting for the reduced KV cache capacity that comes with higher precision. This suggests that quantization noise selectively degrades the computational reasoning capabilities that chain-of-thought amplification relies on, while leaving knowledge retrieval relatively intact—a distinction that has significant implications for how we should think about deploying reasoning models versus standard LLMs.

## Suggestions

- Disentangle the effective size threshold by running experiments where parameter count and precision are varied independently at matched memory budgets (e.g., compare a hypothetical 8B model at 4-bit vs. a 4B model at 8-bit, both at similar total memory), to determine whether the threshold is driven by capacity (parameter count) or numerical fidelity (precision).

- Add a brief discussion or footnote acknowledging that the eviction-vs-quantization comparison uses HQQ rather than a KV-specific quantizer, and note this as a limitation that could affect the strength of finding #5.

- Move the key latency/throughput findings from Appendix C.1 into the main paper, at minimum as a paragraph noting that 4-bit precision is never latency-optimal, which qualifies the memory-only recommendations.

- Include a long-context non-reasoning task (e.g., document QA with 10k+ tokens) as a control to isolate whether the observed KV cache dominance is reasoning-specific or simply a function of generation length.

---

## NfO2Lt2WY7

- GT: Reject (avg 2.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary

This paper systematically analyzes the GRPO loss function to identify which components are essential for improving mathematical reasoning in LLMs. Through controlled ablations on three small-scale instruction-tuned models (0.5B–1.5B), the authors find that (1) negative feedback is indispensable (positive-only training collapses), (2) group-relative advantage estimation is crucial (vanilla REINFORCE with raw rewards also collapses), and (3) PPO-style clipping is unnecessary. They propose RGRA, which removes clipping and policy ratios while retaining group-relative advantages, and report it outperforms GRPO in 17 of 27 benchmark comparisons across 9 mathematical and STEM tasks.

## Strengths

- **Clean, well-motivated ablation design**: The paper decomposes GRPO into clearly delineated variants (positive-only advantages, direct rewards, RGRA with clipping removed), each isolating a specific design choice. This systematic approach—testing what happens when each component is removed—is more informative than the typical "propose a variant and beat the baseline" paradigm and directly addresses a genuine question in the community about whether GRPO's complexity is justified.

- **Compelling demonstration that negative feedback prevents collapse**: The training dynamics in Figure 1 clearly show that positive-only and RAFT-trained 0.5B models suffer reward/response-length collapse within ~20 steps. This is a sharp, actionable finding: practitioners who might be tempted to train only on successful completions receive a clear warning that doing so destabilizes learning, particularly for smaller models.

- **Broad multilingual benchmark coverage**: Evaluation across 9 benchmarks spanning English math, Chinese math, and STEM tasks provides a more thorough assessment than typical single-benchmark evaluations in this space. The inclusion of Chinese-language benchmarks (CMATH, CN-Middle-School, Gaokao2024) is a meaningful strength that tests cross-lingual generalization from English-only training data.

## Weaknesses

### Major:

- **Unanalyzed failure cases undermine the generality of the "clipping is unnecessary" claim**: RGRA underperforms GRPO in 10 of 27 comparisons, and some gaps are substantial. Most notably, on Gaokao2024-STEM for Llama3.2-1B, GRPO achieves 17.2 while RGRA drops to 11.4—a 34% relative degradation. Similarly, on MATH for Qwen2.5-1.5B, GRPO achieves 30.4 vs. RGRA's 29.1. The paper counts wins (17/27) but provides no analysis of *when or why* RGRA fails. Without understanding the conditions under which clipping helps, the practical guidance the paper aims to provide is incomplete. The claim that PPO-style constraints are "not required" is overstated given these counterexamples.

- **Incomplete ablation study for a paper claiming to identify "essential" components**: The paper ablates clipping and positive filtering, but does not test (a) removing KL regularization from RGRA, or (b) varying the group size *G*. Since RGRA retains KL regularization, it is possible that KL—not advantage estimation alone—is doing the stabilization work previously attributed to the combination of advantage estimation and clipping. Without an "RGRA without KL" condition, the decomposition of essential vs. inessential components remains unfinished. Group size *G* similarly goes untested, despite being a core hyperparameter of the group-relative advantage mechanism.

### Minor:

- **Experiments limited to small models (0.5B–1.5B)**: The authors acknowledge this limitation, noting hardware constraints. However, the paper's title and claims address "teaching LLMs to reason" broadly, and the dynamics of policy updates may differ substantially at 7B+ scales where PPO clipping was originally motivated. The findings should be scoped more carefully to small models until larger-scale validation is available.

- **Efficiency claims lack empirical support**: The abstract and conclusion describe RGRA as a "more transparent and efficient alternative," but no wall-clock time, memory usage, or FLOPs comparisons are provided. While removing clipping arguably simplifies implementation, the actual computational savings are not demonstrated, and "efficient" in the RL context typically refers to sample or compute efficiency, not just code simplicity.

- **No statistical significance testing or multiple-seed evaluation**: All results appear to be from single runs. While single-run evaluation is common in recent large-scale RL-for-LLM work, the margins between RGRA and GRPO are often small (1–3 percentage points), making it difficult to determine whether observed differences reflect genuine improvements or run-to-run variance. This is particularly concerning given the mixed results noted above.

- **Domain restricted to mathematical reasoning**: The paper claims implications for "reasoning-focused post-training" broadly but evaluates only on math and STEM tasks. Whether the findings transfer to logical reasoning, code generation, or other reasoning domains remains unknown.

### Trivial:

- The abbreviation "ft" (fine-tuning) in Tables 1–3 is not defined in the table captions.

## Nice-to-Haves

- Comparison with other recent GRPO variants (DAPO, S-GRPO, CPPO) mentioned in the introduction, to contextualize RGRA's improvements against specialized alternatives rather than only against vanilla GRPO.
- Quantitative analysis of reasoning trace emergence (e.g., percentage of outputs containing reasoning steps, correlation between trace length and correctness) rather than the single qualitative example in Figure 2.
- Ablation of RGRA without KL regularization to complete the decomposition of essential components.
- At least one experiment at 7B+ scale to test generalizability of the core findings.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Formatting/typo complaints** (e.g., "prefernces" typo, equation formatting in Section 2.2): Per hard rules, pure formatting/style nitpicks are removed. The parser note explicitly states formatting artifacts are extraction issues, not paper problems.

- **Demand for human evaluation**: For mathematical reasoning with verifiable answers, automated accuracy evaluation is the standard in this area. Requesting human evaluation is scope creep for a purely algorithmic contribution.

- **Broader impact / negative societal impact discussion**: Not required by ICLR and not relevant to assessing the technical contribution.

- **Demand for comparison with DPO variants**: DPO operates under a different paradigm (offline preference optimization) and is outside the paper's stated scope of analyzing GRPO-family RL objectives.

- **Demand for theoretical proofs of why clipping is unnecessary**: The paper is positioned as an empirical analysis. Requesting theoretical gradient-variance analysis is a nice-to-have, not a core requirement for this type of contribution.

- **Reproducibility concerns about GRPO implementation fidelity**: Per hard rules, nitpicks about reproducibility of implementation details are removed. The paper provides code and hyperparameters.

- **RAFT collapse vs. non-trivial test scores "contradiction"**: The paper states collapse occurs "particularly in the 0.5B model" and that larger models show "reward stagnation and gradual shortening" rather than immediate collapse. The Llama3.2-1B RAFT results are consistent with this description. This is not a genuine contradiction.

## Novel Insights

The most interesting observation emerging from the reviews is the tension between RGRA's overall win record and its specific failure modes. RGRA tends to outperform GRPO on the benchmarks where absolute accuracy is lower (harder tasks like MATH, OlympiadBench), but can underperform on higher-accuracy or differently-structured tasks (Gaokao2024-STEM for Llama3.2). This pattern hints that removing clipping might allow more exploratory updates that benefit harder problems but occasionally overshoot on easier ones—a hypothesis the authors do not explore but that could meaningfully advance understanding of when simplicity helps vs. hurts in RL for LLMs.

## Suggestions

- **Analyze the 10/27 cases where RGRA underperforms GRPO**: Identify common properties of these benchmarks/models (e.g., language, difficulty, reward distribution) to characterize when clipping provides value. This would transform the paper from "clipping is unnecessary (mostly)" to "clipping is unnecessary under conditions X, Y, Z," which is far more actionable.
- **Add an RGRA-without-KL ablation**: This is the single most important missing experiment. If RGRA without KL remains stable, the "essential components" story becomes much cleaner. If it collapses, KL regularization—not just advantage estimation—deserves credit for stability, which changes the paper's narrative.
- **Calibrate claims to match evidence**: Replace "PPO-style constraints are not required" with "PPO-style constraints are not required in the small-model math-reasoning settings we tested," and discuss the conditions under which they may still be beneficial.

---

**Axis Assessments:**

- **Novelty**: Moderate. The systematic ablation approach is valuable, but RGRA itself is a straightforward combination of existing ideas (REINFORCE + group-relative advantages from GRPO). The contribution is primarily empirical and analytical rather than algorithmic.

- **Technical soundness**: Partial. The experimental design is clean, but the incomplete ablation (no KL removal, no group-size variation) and unanalyzed failure cases leave core claims insufficiently supported.

- **Empirical support**: Mixed. The broad benchmark coverage and clear collapse demonstrations are strong, but the lack of statistical testing, the 10/27 underperformance cases, and the missing ablations weaken the empirical case for the paper's stronger claims.

- **Significance**: Moderate. If validated at scale and the failure conditions are characterized, this could meaningfully simplify post-training pipelines. Currently, the impact is limited by the small-model-only evaluation and unanalyzed failure modes.

- **Clarity**: Good. The paper is well-structured and the ablation variants are clearly defined. Minor issues with undefined abbreviations and claim calibration do not substantially impede understanding.

---

## RpDJz00zNh

- GT: Reject (avg 4.5)
- Predicted: N/A (6.5/10)
- Match: N/A

### Final Review

## Summary

ConciseHint proposes an "in-reasoning intervention" framework that reduces the verbosity of large reasoning models by continuously injecting concise hints (either manually designed text or learned embeddings) during the token-by-token generation process. The key technical components are: (1) an adaptive injection interval that increases with current generation length (used as a proxy for query complexity), and (2) a dynamic injection position strategy that moves from head toward tail as generation proceeds, balancing accuracy and prefilling cost. Experiments on Qwen3 and DeepSeek-R1 models demonstrate significant token reductions (40–65%) while largely maintaining accuracy, and the method can be combined with existing efficiency techniques.

## Strengths

- **Novel intervention paradigm:** The conceptual shift from "before-reasoning" interventions (prompting, fine-tuning) to "in-reasoning" intervention via continuous hint injection is a genuinely distinct approach. Early exit methods like Deer intervene by stopping generation, but ConciseHint intervenes by steering it—a meaningfully different mechanism that the paper clearly articulates and positions against prior work.

- **Strong and consistent empirical results:** The method achieves substantial token reductions across multiple models and benchmarks (e.g., 48.9% reduction on GSM8K/Qwen3-4B with only 0.07 accuracy loss; 44.5% on GPQA-Diamond with accuracy *gain* of 0.91). The compatibility results—showing further token reduction when combined with Deer, NoWait, and prompting baselines—are particularly compelling, demonstrating the method is not just effective but orthogonal to existing approaches.

- **Well-designed ablation studies:** Table 3 validates the adaptive interval mechanism by showing that fixed high-intensity hints severely degrade accuracy on complex benchmarks (AIME24: 67%→45.33%) but not on easy ones. Table 4 validates the dynamic position strategy by showing tail-injection causes accuracy collapse (55.25%→42.93%). These ablations cleanly establish the necessity of each component.

- **Controllability via interpolation:** The γ parameter in ConciseHint-T (Eq. 4) provides a smooth knob between conciseness and accuracy, which is practically useful and empirically validated in Figure 3.

## Weaknesses

### Major:

- **Generation length as a complexity proxy is a heuristic with identifiable failure modes.** Equation 1 uses $l_k$ (current reasoning length) as a proxy for query complexity, assuming longer reasoning ≈ harder query. This assumption breaks down when models "overthink" easy problems (producing long but unnecessary reasoning) or solve hard problems concisely. In the former case, the model would *reduce* hint intensity exactly when more intervention is needed; in the latter, it would *increase* intensity on already-challenging problems. The paper acknowledges this as a "prior" (Section 3) but does not analyze failure modes. A per-example analysis of where the proxy fails (e.g., correlating hint intensity with actual query difficulty rather than generation length) would substantially strengthen the paper. This matters because the entire safety mechanism—reducing hints on "complex" queries—depends on this proxy being reliable.

- **The deployment story for latency savings requires clarification.** Algorithm 1 uses `client.completions.create()`, suggesting an API-level interaction, while Section A.2's cost analysis relies on selective KV cache invalidation and re-prefilling of only $\tau_k - p$ tokens—something only possible with local inference engine control (e.g., vLLM). With a standard API, the full accumulated context must be re-sent at each injection step, incurring quadratic prefill cost that could negate token savings. The paper uses vLLM for Figure 7's latency measurements, confirming local deployment, but does not explicitly disclose this as a requirement for the claimed efficiency. This is not a fatal flaw—the method works with APIs for token reduction—but the latency claims specifically depend on white-box inference access, which limits the "flexible plugin" framing.

### Minor:

- **Statistical significance on small benchmarks.** AIME24 contains only 30 problems. While the paper runs 10 trials (300 total evaluations), small accuracy differences on this benchmark should be interpreted cautiously. For instance, on GPQA-Diamond (198 questions), the claimed accuracy "rise of 0.91" in Section 4.2 corresponds to roughly 2 additional correct answers across 10 runs, which is within noise. The large accuracy differences on AIME24 (e.g., 67%→45.33% with fixed interval 64) are clearly significant, but marginal differences should not be overstated.

- **Equation 3 contains an unexplained constant (1024).** The position formula $p = \tau_k \cdot \min((\tau_k - \alpha)/1024, 0.8)$ uses 1024 as a scaling factor without justification. Is this related to context window size? Model dimension? A hyperparameter? This makes the formula appear arbitrary and reduces clarity.

- **ConciseHint-T shows accuracy degradation on out-of-domain data at high γ.** Table 2 shows that at γ=1 (full learned embeddings), GPQA-Diamond accuracy drops from 35.35% (ConciseHint) to 32.83%. The paper claims "generalize well to out-of-domain data" but the data shows this generalization is fragile when compression is aggressive. The γ=0.7 setting mitigates this, but the claim should be tempered.

### Trivial:

- The paper title says "Continuous Concise Hints" but the hints are injected at intervals, not continuously. This is a minor naming imprecision.

## Nice-to-Haves

- **Proactive complexity estimation:** A lightweight pre-reasoning complexity classifier could supplement the reactive length-based proxy, potentially reducing the "wasted tokens" before the model determines a query is hard. This would strengthen the adaptive mechanism but is outside the paper's scope.

- **Broader evaluation on long-context reasoning tasks** (e.g., multi-hop QA, legal/document reasoning) where reasoning chains are naturally very long, to stress-test the method's stability under extended injection sequences.

- **Theoretical FLOPs analysis** of the injection overhead to complement the empirical latency measurements, providing a more rigorous characterization of the efficiency trade-off.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"Missing comparison with token pruning / skip-decoding methods"** (from Harsh Critic and Spark Finder): The paper already compares with 6 baselines (BeConcise, Prompt, Deer, NoWait, AlphaOne, O1-Pruner) in the main paper and appendix. Demanding even more baselines is a generic weakness that doesn't harm the core claim.

- **"API incompatibility / white-box requirement for ConciseHint-T"** (from Harsh Critic): ConciseHint-T requires embedding injection, which does need white-box access, but this is inherent to the method design and the paper clearly presents both a training-free version (API-compatible) and a trained version. The paper does not claim ConciseHint-T works with APIs.

- **"Hyperparameter sensitivity / task-specific tuning"** (from Harsh Critic via transferable weaknesses): The paper explicitly uses fixed α=128, β=0.2 across ALL experiments and provides ablation studies showing robustness. This is already addressed.

- **"Incomplete construction details for concise reasoning data"** (from Neutral reviewer): The data comes from MixChain-Z-GSM8K, which is cited. This is a reproducibility nitpick about a standard dataset.

- **"Formatting/stylistic issues with Algorithm 1 pseudocode"** (from Harsh Critic): Removed per hard rules on formatting nitpicks.

- **"Overhead on short responses"** (from Harsh Critic): The paper focuses on benchmarks where reasoning is verbose (the target use case). Criticizing performance on non-target scenarios is scope creep.

## Novel Insights

The most interesting empirical finding is the *synergy* between in-reasoning intervention and pre-reasoning methods. Table 1 shows that combining ConciseHint with Deer or NoWait yields *more* than additive token reduction (e.g., Deer alone reduces tokens by 41% on GSM8K/Qwen3-4B, but ConciseHint + Deer reduces by 65%). This suggests that the verbosity of reasoning models has multiple independent sources (unnecessary self-reflection, redundant coherence tokens, overthinking) and that addressing different sources simultaneously is more effective than any single approach. This composability property is underexplored in the efficient reasoning literature and could motivate a modular, multi-pronged approach to reasoning efficiency.

## Suggestions

- **Add a clear deployment requirements section** specifying which efficiency claims (token reduction vs. latency reduction) require local inference with KV cache control, and which apply to API-based usage. This would resolve the ambiguity between Algorithm 1's pseudocode and the cost analysis.

- **Analyze failure cases of the length-based complexity proxy.** Correlate per-example token counts with ground-truth difficulty labels (available for GSM8K difficulty tiers) to quantify when the proxy misclassifies and how much accuracy is affected. This would transform the acknowledged heuristic into a quantified limitation.

- **Justify or ablate the 1024 constant** in Equation 3. A simple experiment varying this scaling factor would clarify whether it is essential or arbitrary, and would make the position formula more interpretable.

---

**Evaluation Summary:**

- **Novelty:** High. The in-reasoning intervention paradigm is genuinely distinct from existing pre-reasoning and early-exit approaches, and the adaptive mechanism adds meaningful sophistication.

- **Technical soundness:** Moderate-to-good. The core method is clearly described and well-ablated, but the heuristic complexity proxy and the deployment requirements for latency claims need more honest discussion.

- **Empirical support:** Strong. Consistent results across 4 models, 5+ benchmarks, and 6 baselines with clear ablations. Small benchmark statistical concerns and the ConciseHint-T generalization gap are real but limited in scope.

- **Significance:** High. The method is practical, effective, and composable with existing approaches, addressing a critical bottleneck in reasoning model deployment.

- **Clarity:** Good overall, but the conflation of API-level pseudocode with KV-cache-dependent efficiency claims creates confusion about what deployment scenarios are supported.

---

## tswBfpkwHn

- GT: Reject (avg 5.0)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary
This paper presents the first theoretical analysis of the training dynamics and ICL generalization of one-layer Mamba models on binary classification tasks with additive outliers in prompts. The key result is that Mamba's nonlinear gating mechanism enables it to tolerate outlier fractions approaching 1 in context examples, while linear Transformers fail when outliers exceed 1/2. The paper characterizes the mechanism: the linear attention layer selects context examples sharing the query's relevant pattern, while the gating layer suppresses outlier-containing examples and induces a recency bias that emphasizes nearby clean examples.

## Strengths
- **First theoretical treatment of Mamba's ICL training dynamics with outliers.** The paper handles the nonlinearity of Mamba's gating mechanism—a significant technical challenge that prior Transformer-focused ICL theory (Zhang et al., 2023; Li et al., 2024a) does not address—by dividing training into two phases (Lemmas 4–5) and characterizing gradient updates along relevant, irrelevant, and outlier pattern directions. This is a genuine methodological contribution.
- **Principled isolation of the gating effect.** The comparison with linear Transformers is methodologically clean: setting $G_{i,l+1}(\mathbf{w}) = 1$ reduces Mamba to a linear Transformer, making the gating mechanism the only architectural difference. This allows the paper to rigorously attribute the robustness gap specifically to nonlinear gating rather than other confounding factors.
- **Mechanistic interpretability supported by theory and experiments.** Corollaries 1 and 2 provide concrete characterizations—attention concentrates on same-pattern examples, gating suppresses outliers ($G \lesssim \text{poly}(M_1)^{-1}$) and decays exponentially with index distance—which are directly validated in Figures 3–4 and Table 1. This goes beyond black-box bounds to explain *why* the architecture works.

## Weaknesses

### Major:
- **The robustness advantage over Transformers is proven only against linear attention, but softmax attention shows comparable robustness empirically.** Appendix B.1 (Table 3) shows that a softmax Transformer achieves 99.28% accuracy in the CQ setting where Mamba drops to 82.73%, and maintains >99% accuracy for $\alpha \leq 0.7$ (Table 4). The paper's framing in the Abstract and Introduction ("Mamba...achieving comparable performance across a wide range of language tasks" and comparison with "Transformer-based models") risks leading readers to believe the robustness advantage applies to standard Transformers. Remark 6 acknowledges this but is insufficiently prominent. The practical significance of the Mamba advantage is substantially reduced by this finding, as standard LLMs use softmax attention, not linear attention.

- **The recency bias that enables outlier suppression creates a structural vulnerability to outlier position.** Corollary 2(ii) and Eq. (18) show that gating values decay exponentially with distance from the query. This is the mechanism by which outliers are suppressed—but it also means that when outliers are placed closest to the query (CQ setting), clean examples are pushed far away and their gating values decay, causing Mamba's accuracy to drop to 82.73% (Table 1). The linear Transformer, lacking this decay, maintains 93.96% in the same setting. The paper's main claims emphasize robustness to outlier *fraction* without adequately foregrounding this positional vulnerability, which is a direct and important consequence of the same mechanism.

- **Generalization to unseen outliers requires them to be positive linear combinations of training outliers (Theorem 2, Condition (a)).** This restricts the "distribution-shifted" outlier robustness to a specific subspace spanned by training outlier patterns. If a test-time outlier has a component orthogonal to all training outliers (which is the more practically relevant adversarial setting), the theoretical guarantees do not apply. The paper should more clearly articulate this limitation and discuss whether the gating mechanism still provides partial protection in such cases.

### Minor:
- **The one-layer, binary classification setting limits direct practical implications.** While this is standard for theoretical ICL analysis (Zhang et al., 2023; Li et al., 2024a; Li et al., 2025b), the gap to practical multi-layer Mamba models handling natural language remains substantial. The 3-layer experiments in Section 4.2 partially address this but do not establish whether the theoretical phase-transition dynamics (Lemmas 4–5) persist at depth.

- **The SST-2 validation (Appendix B.2) provides only weak support for the orthogonal pattern assumption.** Table 6 shows that classification with top-10 PCA components is close to full-dimension accuracy, but PCA orthogonality does not imply that semantic patterns are orthogonal or sparse in the manner required by the theoretical framework (Eq. 6). The "James Bond" outlier experiment (Table 7) is more convincing as a proof of concept but is still limited to a single dataset and outlier type.

### Trivial:
- The abstract's claim that Mamba achieves "comparable performance across a wide range of language tasks" is a general statement about Mamba from prior work and not a contribution of this paper. It does not mislead about this paper's specific contributions but could be more precisely scoped.

## Nice-to-Haves
- Include the softmax Transformer comparison (Table 3) in the main text rather than relegating it to the appendix, since it substantially qualifies the main claims and is essential for readers to assess practical significance.
- Provide a formal or semi-formal analysis of the CQ failure mode—e.g., a lower bound or tighter characterization of when the recency bias becomes a liability—as the current treatment is purely empirical.
- Evaluate on a standard text-based ICL benchmark where outliers take the form of semantically corrupted instructions or mislabeled examples, rather than only additive feature noise.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Weakness: "Missing related works."** Hard rule: cannot confirm existence of uncited works.
- **Weakness: "No computational complexity analysis."** The paper does provide iteration counts (e.g., $T_M = \Theta(\eta^{-1}(1-p_a)^{-1}\beta^{-2}M_1)$) and batch size requirements; claiming no complexity analysis exists is factually incorrect.
- **Weakness: "Missing comparison with data augmentation/ensemble/robust optimization baselines."** This is outside the scope of a theoretical analysis paper; the paper's contribution is provable guarantees, not practical robustness methods.
- **Weakness: "Unclear practical applicability of complex conditions."** This is a generic criticism that applies to nearly all theoretical ML papers with sufficient conditions; the conditions are explicitly stated and interpretable (e.g., outlier magnitude must be in a specific range, context length must exceed a threshold).
- **Weakness: Reproducibility concerns about undisclosed hyperparameters.** Hard rule: remove nitpicks about reproducibility of implementation details.
- **Weakness: Formatting/style issues in equations.** Hard rule: remove formatting nitpicks.

## Novel Insights
The paper reveals a fundamental design trade-off in gated SSMs: the same exponential decay mechanism that provides robustness to outlier *fraction* creates a positional vulnerability when corrupted tokens appear near the query. This is not merely an empirical observation—it is a direct structural consequence of the gating formulation (Corollary 2, Eq. 18). This suggests that robustness in SSMs comes from a specific spatial prior about where noise appears in the context, rather than uniform noise tolerance. The practical implication is that prompt engineering for Mamba-based ICL must consider not just *how many* examples are corrupted, but *where* they are positioned—a constraint that softmax attention does not impose to the same degree.

## Suggestions
- Move the softmax Transformer comparison from Appendix B.1 to the main text and revise framing to say "Mamba outperforms *linear* Transformers" rather than "Transformers" in the Abstract and Introduction, or add a clear caveat about the scope of the theoretical comparison.
- Add a "Limitations" paragraph explicitly discussing the CQ positional vulnerability as a structural consequence of the gating mechanism, not merely an empirical observation.
- In Theorem 2, add a brief discussion of what happens when Condition (a) is violated—does the gating mechanism still provide partial suppression, or does the guarantee collapse entirely? Even informal reasoning here would strengthen the practical relevance.

---

## sh1hWO9RHo

- GT: Reject (avg 4.5)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary
The paper introduces the Agent GPA (Goal-Plan-Action) framework, which decomposes agent evaluation into specialized dimensions assessed by dedicated LLM judges. Seven judges cover Logical Consistency, Execution Efficiency, Plan Adherence, Plan Quality, Tool Selection, and Tool Calling (with Goal Fulfillment claimed but not experimentally evaluated). Experiments on TRAIL/GAIA and an internal dataset show that the specialized ensemble achieves 95% error coverage versus 54% for a monolithic baseline, with 80–95% human–LLM agreement and 86% error localization.

## Strengths
- **The decomposition into specialized judges yields substantial empirical gains over monolithic evaluation.** Table 2 shows the GPA ensemble captures 95% of TRAIL-annotated errors vs. 54% for the baseline TRAIL LLM judge, with particularly strong coverage of high-impact errors (100%). This validates the core thesis that specialized evaluation outperforms single-judge approaches.
- **Orthogonality analysis provides empirical evidence that judges capture distinct failure modes.** Appendix F (Tables 22–25) shows low inter-metric agreement across α, κ, Jaccard, and phi correlation, confirming that the dimensions are not redundant. This is a valuable contribution beyond merely asserting the decomposition is useful.
- **Error localization capability (86%) provides actionable debugging value.** Tables 5–6 demonstrate that the framework goes beyond binary pass/fail to pinpoint error locations via span IDs, which is practically significant for agent development and distinguishes this work from outcome-only evaluations.
- **GEPA integration demonstrates a practical path to scalable, automated prompt optimization.** Table 8 shows GEPA-optimized prompts match or exceed manually crafted ones (e.g., LC recall improving from 80.7% to 87.9%), and Table 9 shows meaningful generalization to SWE-bench without manual retuning.

## Weaknesses

### Major:
- **Goal Fulfillment, one of the five core metrics named in the abstract, receives zero experimental evaluation.** The abstract lists GF as a primary metric, Section 3 defines it, and Figure 1 positions it as a core dimension. Yet GF is absent from every experimental table (Tables 1–7, 10–12). This is not a minor omission—it is one of the five pillars of the claimed contribution. The paper does not explain why GF was excluded from evaluation or what its reliability properties are.
- **Plan Quality and Plan Adherence judges show poor reliability, undermining confidence in two of the framework's core dimensions.** PQ achieves Krippendorff's α = 0.628 (below the conventional 0.667 threshold for tentative conclusions) and test F1 = 0.49. PA achieves test F1 = 0.66 with high false positive rates (precision = 0.52). The paper acknowledges "small sample size" for these categories but still presents them as core contributions. If planning-related evaluation is unreliable, the framework's ability to diagnose the Plan dimension of the Goal-Plan-Action loop is significantly weakened.
- **SWE-bench evaluation excludes 3 of 7 judges (PQ, PA, TS) because the CodeAct agent does not perform explicit planning.** This reveals a structural limitation: the framework cannot evaluate agents whose architecture doesn't match its assumed operational loop (explicit high-level planning, multiple tools). The paper frames GPA as a general agent evaluation framework, but 43% of its judges are inapplicable to a common agent paradigm (single-tool, implicit-planning agents). The generalizability claim should be scoped accordingly.

### Minor:
- **Execution Efficiency's low bucketed accuracy (35.6% on test, Table 4) raises criterion validity concerns.** The paper hypothesizes that EE "occasionally flags errors not strictly related to efficiency," but this explanation suggests the judge may be measuring something different from its stated construct, which is a validity problem rather than just an alignment problem.
- **The comparison of 7 specialized GPA judges vs. 1 monolithic TRAIL judge confounds specialization with ensemble size.** While the comparison is appropriate for testing whether decomposition helps (the paper's core thesis), it does not isolate whether the gain comes from specialization per se versus simply having more judges. An ablation or comparison against an ensemble of 7 general judges would strengthen the claim.
- **The internal ANON-Data-Agent evaluation rests on only 17 traces.** While the results (82% human agreement) are directionally consistent with the TRAIL/GAIA findings, the sample is too small to support strong claims about production-grade applicability or to draw conclusions about systematic error patterns.
- **Strong model dependency for the harder metrics.** Table 19 shows LC accuracy drops from 76.5% (Claude-4-Sonnet) to 29.4% (Claude-3-7-Sonnet) and 47.1% (GPT-4o) on the internal dataset. The paper acknowledges LC is "the harder dimension," but the steep performance cliff suggests the framework's reliability is contingent on using frontier models, limiting its practical accessibility.

### Trivial:
- The abstract mentions "five evaluation metrics" while the framework operationalizes seven LLM judges. TS and TC are described as complements to PQ and PA respectively, making the framing internally consistent, but the transition from 5 metrics to 7 judges could be clearer upfront.

## Nice-to-Haves
- An ablation study removing individual judges to quantify each one's marginal contribution to overall error coverage, which would directly address whether all 7 judges are necessary.
- Quantification of computational cost (tokens, latency, USD per trace) for the full GPA suite vs. baseline, to support the claim of "scalable" evaluation.
- Evaluation on an agent that performs explicit planning but operates in a different domain (e.g., embodied, multi-agent) to test generalizability beyond the web/code agents studied.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **"Few-shot contamination between training and evaluation"** (from Spark Finder): Invalid — the paper explicitly uses a dev/test split, and few-shot examples are drawn from the dev set only (Section 4.1.2: "1-2 few-shot examples drawn from the development (dev) dataset"). Standard practice.
- **"No comparison to AgentBench, AgentRewardBench as competing frameworks"** (from Spark Finder): Unreasonable — these are benchmarks for evaluating agents or evaluators, not competing evaluation frameworks with the same structure. The TRAIL comparison is the appropriate baseline since it evaluates on the same dataset with the same error annotations.
- **"Prompt sensitivity and temperature ablations not discussed"** (from Harsh Critic, transferred): This is a generic concern applicable to any LLM-as-judge paper. The paper does provide consistency analysis across 5 runs (Section 4.1.4) and GEPA optimization analysis, which partially addresses robustness.
- **"Potential gaming/Goodhart's law — agents optimized for these judges might satisfy evaluation without improving"** (from Harsh Critic): Speculative future concern, not a weakness of the paper as presented.
- **"Internal dataset not released, limiting reproducibility"** (from Harsh Critic): The internal dataset is a proprietary production system; the paper commits to releasing the code, prompts, and re-annotated TRAIL/GAIA data. Reproducibility of the public benchmark results is supported.
- **"What happens when judges disagree on the same error"** (from Harsh Critic): The orthogonality analysis in Appendix F shows metrics fire on different phenomena; high disagreement is by design. This is addressed.
- **"Human annotator inter-annotator agreement on GPA mapping not reported"** (from Harsh Critic): The paper reports human-human agreement rates (0.70 dev, 0.67 test) in Appendix E and notes a third annotator cross-checked mappings. While IAA on the specific mapping task isn't reported, the overall agreement context is provided.
- **"Pre-processing may lose error signals"** (from Positive Reviewer): Speculative without evidence that the specific preprocessing (removing duplicate messages) removes error-relevant information. The 95% coverage suggests preprocessing preserved error signals adequately.

## Novel Insights
The paper reveals a striking "contextual specialization" pattern where judges' utility inverts based on error severity: PA fails on low-impact errors but becomes the top localizer for high-impact failures (F1=0.85), while TC shifts from high-recall detector to high-precision localizer. This suggests that effective agent debugging requires dynamically selecting which judges to trust based on the context and severity of the failure, rather than treating all judges uniformly—a meta-observation the paper touches on but could elevate as a design principle.

## Suggestions
- **Add GF experimental results.** Either evaluate GF as a judge or explicitly scope it as future work with justification for its exclusion. A core metric with zero validation is the most damaging gap.
- **Address the PQ reliability problem directly.** Either demonstrate that GEPA optimization or rubric refinement can bring PQ above the α=0.667 threshold, or merge PQ into a combined "Planning" metric with PA to reduce the number of underperforming dimensions.
- **Scope the generalizability claim.** Explicitly state that GPA is designed for agents with explicit planning and multi-tool architectures, and discuss adaptation requirements for agents with implicit planning or single-tool setups. The SWE-bench partial evaluation already demonstrates this boundary—acknowledging it strengthens rather than weakens the contribution.
- **Report per-judge cost.** Even a rough estimate (e.g., "7 judges cost ~7× a single judge") would help practitioners decide whether the diagnostic value justifies the overhead for their use case.

---

**Axis Evaluations:**

- **Novelty**: Moderate. The decomposition of agent evaluation into Goal-Plan-Action dimensions is a natural but well-executed idea. The orthogonality validation and GEPA integration add incremental novelty beyond the core framework.

- **Technical soundness**: Mixed. The core empirical results on LC, EE, TS, and TC are strong and well-supported. However, PQ and PA lack reliability, GF is untested, and the SWE-bench evaluation covers only 3/7 judges. The framework's technical foundation is sound where tested, but incomplete.

- **Empirical support**: Adequate for the strongest judges (TC, TS, EE, LC) but insufficient for the planning dimension (PQ, PA) and absent for GF. The internal dataset (n=17) provides only directional support.

- **Significance**: High potential. Process-level agent debugging is an important and underserved problem, and the 86% localization capability is practically valuable. The framework could become a standard tool if the underperforming components are strengthened.

- **Clarity**: Good overall structure with comprehensive appendices. The 5-metric/7-judge framing creates initial confusion. Tables are information-dense but well-organized.

---

## 1E4Bltg6Xb

- GT: Accept (Poster) (avg 4.7)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary

This paper proposes a Dynamics Feature Representation (DFR) framework for RL-based Dynamic Path Planning (DPP) in urban road networks. DFR hierarchically refines high-dimensional global traffic dynamics into compact, decision-relevant features through two stages: (1) a policy attention mechanism that uses a pre-trained static shortest-path policy to extract a task-relevant subgraph, and (2) an n-hop neighborhood method that further decouples this subgraph into agent-centric local features. Experiments on three urban road networks demonstrate improved RL performance and faster planning times compared to full-dynamics baselines.

## Strengths

- **Principled hierarchical approach to the completeness-efficiency trade-off**: The two-stage refinement (global task filtering via policy attention → agent-centric local encoding via n-hop neighborhoods) is a well-motivated architectural solution to a genuine problem. Rather than heuristically choosing between local and global state, DFR provides a structured middle path, which is a meaningful design contribution.
- **Substantial empirical efficiency gains with maintained performance**: DFR reduces average planning time by 85.59% (DQN), 46.08% (GCN+DQN), and 79.32% (PPO) compared to all-dynamics baselines, while simultaneously improving or matching success rate and mean GAP. These are non-trivial efficiency gains for real-time planning.
- **Systematic ablation study**: The (k, n) ablation in Figure 6 provides useful practical insights—showing that n exhibits diminishing returns while k has more complex behavior—and the authors honestly report these findings rather than cherry-picking.

## Weaknesses

### Major:

- **Static prior filtering risks excluding the dynamic optimum**: The policy attention mechanism filters the state space to edges along the top-k *static* shortest paths (πd*). In dynamic environments where the objective is travel time (not distance), the true optimal dynamic path may deviate substantially from static shortest paths—for example, taking a longer detour to avoid severe congestion. If the dynamic optimal path lies outside the static top-k subgraph, the RL agent is structurally prevented from learning it, making the policy suboptimal by design. The paper acknowledges that "distance naturally serves as one of the most fundamental constraints" (Section 4.3) but does not analyze the risk or magnitude of this filtering error. An empirical analysis quantifying what fraction of dynamic-optimal edges are retained by the static subgraph (a "recall" metric) would directly address this concern.

- **PSR theoretical claims are overstated**: Section 4.2 claims that "Grounding DFR in PSR principles thus guarantees that the resulting representations are compact, temporally predictive, and theoretically sufficient." However, PSR requires that the state representation enable prediction of *all* future observation sequences given action sequences. The paper provides no formal proof that an n-hop neighborhood of a static subgraph satisfies this sufficiency criterion, particularly when traffic dynamics exhibit long-range spatial correlations (e.g., congestion propagating from beyond the n-hop radius). The invocation of PSR is currently a loose analogy rather than a rigorous justification, and the word "guarantees" should be replaced with a more measured claim or supported by formal analysis.

### Minor:

- **Unnecessary use of RL for static subgraph generation**: The policy πd* is obtained by training an RL agent on static distance-based rewards (Section 4.3). For static shortest paths on a known graph, exact algorithms like Dijkstra or Yen's algorithm (for top-k paths) are both faster and exact. Using an approximate RL policy introduces unnecessary approximation error and training overhead. The paper does not justify this design choice.

- **Synthetic dynamics without specified temporal correlation**: The congestion factor β ∈ [0.1, 1.5] is applied to real OSM topologies, but the paper does not specify how β evolves over time—whether it is i.i.d., Markovian, or has longer-range temporal dependencies. This matters because the n-hop state design implicitly assumes that local spatial context captures the relevant temporal dynamics. Without temporal correlation structure, the claim of "realistic" evaluation (Section 5.1) is partially unsupported.

- **Unclear AD baseline implementation for MLP-based methods**: For DQN (MLP-based), the "All Dynamics" baseline must handle the full graph's edge weights as input. Since MLPs require fixed-size input, it is unclear how the variable-sized graph is encoded (flattened? padded?). This ambiguity makes it hard to assess whether the AD baseline is a fair comparison or a strawman.

- **Limited baseline comparison with recent RL-based DPP methods**: The paper compares only DQN, PPO, and GCN+DQN against DFR-enhanced versions. While the paper's focus is on the impact of state representation within the RL paradigm, comparisons with more recent RL-based DPP or state representation methods would better contextualize the contribution.

### Trivial:

- **Imprecise language about Markov property**: The Introduction states "insufficient state representation may undermine the Markov property." Technically, insufficient representation creates partial observability (POMDP), not a violation of the environment's Markov property. This is a language issue rather than a conceptual error—the authors' intended meaning is clear.

## Nice-to-Haves

- Empirically verify the Markov property claim by testing whether adding history buffers to the DFR state improves policy performance. If history helps significantly, the compressed state is not informationally sufficient as claimed.
- Develop an adaptive mechanism for selecting k and n (e.g., based on traffic volatility or graph density metrics), as the authors themselves identify this as a limitation.
- Validate on real-world traffic traces (e.g., historical speed data from PeMS or similar) rather than synthetic congestion factors, to test robustness under realistic temporal correlations.
- Visualize the policy attention subgraph overlaid on the ground-truth optimal dynamic path for specific episodes, to reveal whether gains come from noise reduction or information loss.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Road closures / topological changes invalidate pre-computed subgraphs.** The paper explicitly assumes V remains constant in Section 3.1 ("It is assumed that V remains constant"). Criticizing the absence of topological changes is scope creep—the paper's stated problem is weight dynamics, not structural dynamics.
- **Weakness: Statistical significance testing missing.** Single-run or few-seed evaluation without formal significance tests is the norm for this type of RL experiment. Demanding t-tests or confidence intervals is a nice-to-have, not a core flaw.
- **Weakness: Parameter sensitivity of k and n.** The authors already acknowledge this limitation in the Conclusion ("the two parameters of k and n in DFR are manually selected in this study, which may limit its practical applicability") and propose adaptive selection as future work. Criticizing what the authors already reasonably address is double-counting.
- **Weakness: Pre-computing top-k paths for all source-destination pairs is O(N²).** The paper states that both policy attention and n-hop neighborhoods "depend only on the fixed road network topology, allowing offline computation and reuse." The concern about storage is reasonable but speculative without evidence that it is actually a bottleneck; the paper demonstrates feasible computation on the tested networks.
- **Weakness: Missing related works.** Per hard rules, we cannot confirm the existence of suggested missing references.
- **Weakness: Formatting/style issues.** Per hard rules, these are removed.

## Novel Insights

The most insightful observation across the reviews is the fundamental tension at the heart of DFR: the method's primary strength (drastic dimensionality reduction via a static structural prior) is also its primary vulnerability (the static prior may systematically exclude the dynamic optimum). This is not merely a theoretical concern—it creates a testable prediction. In low-volatility regimes where dynamic optima align with static shortest paths, DFR should excel; in high-volatility regimes where congestion forces large detours, DFR's performance should degrade relative to full-dynamics baselines. The paper's current evaluation does not test this prediction, and doing so would either substantiate the framework's robustness or reveal its operational boundaries. This analysis would also inform the design of the adaptive k mechanism the authors envision.

## Suggestions

- **Quantify the "subgraph recall" of the policy attention mechanism**: Compute what fraction of edges on the dynamic optimal path (found by dynamic Dijkstra) are retained in the static top-k subgraph under varying congestion levels. This directly measures the information loss from static filtering and would either validate the design or reveal its failure modes.
- **Replace RL-based πd* with Yen's algorithm**: Since the static subgraph is defined by top-k shortest paths, using an exact algorithm would eliminate approximation error and simplify the method without changing its essence.
- **Add a high-volatility experimental condition**: Create scenarios where β has high variance and strong spatial correlation (e.g., a major corridor experiencing congestion that forces routes far from the static shortest paths). This tests the robustness of the static prior assumption.
- **Clarify the AD baseline implementation**: Explicitly state how the full graph dynamics are encoded as input for MLP-based DQN, so readers can assess the fairness of the comparison.

---

**Assessment by axis:**

- **Novelty**: Moderate. The combination of static-policy-based hard attention with n-hop local features for DPP state representation is a distinct architectural contribution, though each component individually is well-established.
- **Technical soundness**: The core mechanism works empirically, but the theoretical claims (PSR sufficiency, Markov guarantee) are overstated relative to what is proven, and the static-dynamic tension in the policy attention design is a meaningful conceptual concern that is not analyzed.
- **Empirical support**: Adequate for the basic efficacy claim (DFR improves efficiency and performance), but limited by synthetic dynamics, lack of analysis on when the method fails, and basic baselines. The ablation study is a strength.
- **Significance**: Moderate-to-good. The efficiency gains are practically meaningful for real-time planning, and the state representation perspective on DPP is valuable. Significance is bounded by the unresolved question of whether the static prior becomes a liability in volatile scenarios.
- **Clarity**: The paper is generally well-organized and readable, with some imprecise theoretical language (Markov property, PSR guarantees) that could mislead.

---

## qSak1Hjfdq

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

This paper formalizes the all-day multi-scenes lifelong VLN (AML-VLN) problem, where agents must continually learn across multiple scenes and diverse environmental conditions (low-light, scattering, overexposure) without catastrophic forgetting. To address this, the authors propose Tucker Adaptation (TuKA), which lifts parameter-efficient adaptation from 2D matrices to a 4th-order tensor via Tucker decomposition, explicitly decoupling shared knowledge (core tensor + encoder/decoder) from scene-specific and environment-specific expert factor matrices. A Decoupled Knowledge Incremental Learning (DKIL) strategy combines EWC-style regularization on shared components with orthogonal constraints on new experts. The paper also contributes an extended Habitat benchmark with synthesized degraded imaging conditions and demonstrates AlldayWalker's superiority over LoRA-based continual learning baselines.

## Strengths

- **Principled departure from matrix-based adaptation:** The core insight—that multi-hierarchical knowledge (scene × environment) is naturally represented as a high-order tensor rather than forced into 2D matrix factorizations—is well-motivated and technically sound. The Tucker decomposition cleanly separates shared vs. specific knowledge across tensor modes, and Equation 3 provides a mathematically correct mechanism for collapsing the high-order tensor back to a 2D weight matrix compatible with LLM backbones. The comparison with ABC-LoRA (Appendix I) effectively isolates the benefit of tensor factorization from mere architectural hierarchy (65% vs. 55% SR), demonstrating that the gain comes from the Tucker core capturing cross-dimensional interactions, not just from having a tree structure.

- **Thorough ablation design:** The paper ablates 3rd-order vs. 4th-order tensors (Appendix H), shared component contributions (Table 3), rank scaling (Appendix G), and extends to 5th-order tensors (Appendix J). The 30-task scalability test (Table 4) shows minimal degradation, supporting the method's stability. These ablations collectively build a strong empirical case for the design choices.

- **Benchmark contribution with practical relevance:** Extending Habitat with physics-based imaging degradation models (atmospheric scattering, low-light sensor noise, overexposure saturation) grounded in established imaging models (Eqs. 10–12) provides a reusable testbed for a practically important problem that standard VLN benchmarks ignore.

## Weaknesses

### Major:

- **No analysis of expert retrieval reliability at inference:** Section 3.4 describes matching the current observation's CLIP features against stored scene/environment feature sets via cosine similarity, but provides zero analysis of how often this retrieval is correct, what happens when it fails, or how robust it is to visual ambiguity (e.g., a dimly lit scattering scene might match "low-light" rather than "scattering"). Since the task-id is agnostic at test time, the entire method's practical viability hinges on this retrieval step. Without any retrieval accuracy metric, failure case analysis, or sensitivity to ambiguous observations, it is impossible to assess whether AlldayWalker would work reliably in deployment. This is the single most important gap in the evaluation.

- **Missing computational overhead analysis for inference:** The paper emphasizes parameter efficiency (~0.3M trainable params) but never reports inference latency, FLOPs, or memory bandwidth for the Tucker reconstruction (Eq. 3) combined with the expert retrieval search. The mode-products in Eq. 3 must be computed for every transformer layer at every navigation step, and the CLIP-based retrieval adds a forward pass plus similarity computation. For a paper targeting real-time robotic deployment ("all-day" navigation), the absence of any latency comparison against standard LoRA's simple matrix multiply is a significant omission.

### Minor:

- **Scalability of expert matrices in open-ended lifelong learning:** The current benchmark has M=7 scene experts and N=4 environment experts. In true open-ended deployment, encountering a new scene requires adding a new row to **U₃**, growing parameters linearly. The paper does not discuss this growth rate or compare it against simply storing independent LoRA adapters per task. If every new task introduces a novel scene, TuKA's parameter growth (new expert row + shared updates) may not be more efficient than per-task LoRA storage, undermining the parameter-efficiency claim for the lifelong setting that matters most.

- **Ambiguity in expert selection frequency:** It is unclear whether the expert retrieval (Section 3.4) is performed once per episode or dynamically per step. If a robot transitions from a dark corridor to a well-lit room mid-episode, does the agent switch environment experts? The current formulation and Algorithm 2 suggest a single retrieval per inference call using "current observation Oq," but the paper never discusses dynamic expert switching or evaluates its necessity, which is critical for the "all-day" claim where conditions change within a single trajectory.

- **No ablation of individual DKIL loss components:** Table 3 ablates shared architectural components but does not isolate the contributions of L_ewc, L_co, and L_es to forgetting prevention. Given that these three losses serve distinct purposes (shared consolidation, expert consistency, orthogonal exploration), understanding which drives the performance is essential for justifying the full loss design and for practitioners who may want simpler variants.

- **No variance reported across task orderings:** Continual learning results are known to be sensitive to task ordering. The paper mentions randomized ordering but reports results from a single run. Without error bars or multiple ordering experiments, the reliability of the reported improvements is uncertain.

### Trivial:

- The distinction between "scene" (geometry/layout) and "environment" (illumination/weather) is crucial for the tensor formulation but is introduced informally in Section 2. An earlier, explicit definition alongside Figure 1 would aid readability.

## Nice-to-Haves

- Comparison with replay-based continual learning methods (e.g., iCaRL, ER) to situate TuKA within the broader CL landscape, though the paper reasonably argues replay is costly for embodied tasks.
- Interpretability analysis of what the core tensor **G** vs. expert matrices **U₃**, **U₄** actually learn (e.g., via probing or dimensionality reduction), to move beyond the assumption that knowledge is genuinely decoupled.
- Analysis of negative transfer between tasks (when learning a new task actively harms performance on a previous one beyond simple forgetting), which is distinct from the forgetting rates currently reported.
- Hyperparameter sensitivity analysis for λ₁, λ₂, λ₃ beyond the rank scaling in Appendix G.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Synthetic degradations are too aggressive / inflate performance gap"**: The harsh critic suggests the 0% collapse of Seq-FT indicates unrealistically extreme degradations. However, Seq-FT with a single overwritten LoRA naturally collapses on all prior tasks regardless of degradation severity. Other baselines (LwF-LoRA: 12%, EWC-LoRA: 15%) also perform poorly but non-zero, and the real-world generalization results (Table 5: 55% avg SR on unseen real scenarios) confirm the method works beyond synthetic settings. The performance gap reflects genuine problem difficulty, not artificial inflation.

- **"Fisher Information requires past data / replay buffer"**: The harsh critic questions how F_{θ,t-1} is computed without storing past data. Section 5.1 explicitly states Fisher is computed "using the first 10% of the data before adaptation to each task," which is standard EWC practice—computing Fisher on the current model's parameters using current task data before updating. No replay buffer is needed.

- **"ABC-LoRA is custom-built, compare with more established methods"**: ABC-LoRA is a reasonable hierarchical baseline specifically designed for fair comparison with matched parameter budgets. The paper already compares against 10 other baselines including established methods (HydraLoRA, BranchLoRA, O-LoRA, SD-LoRA). Adding more baselines would be nice-to-have, not a weakness.

- **"Missing related works"**: Per the hard rules, I cannot confirm the existence of unspecified related works.

- **"Test on more diverse degradation types (rain, fog, motion blur)"**: This is scope creep. The paper introduces 4 degradation types grounded in physics-based imaging models, which is sufficient for an initial benchmark. Demanding more types is a nice-to-have.

- **"Model-dependent effects not characterized"**: The paper evaluates on a single backbone (StreamVLN + Qwen2-7B). While cross-architecture evaluation would strengthen claims, this is standard scope for a methods paper and not a core flaw.

- **"Mechanistic explanation of why 4th-order outperforms 3rd-order is missing"**: The paper does provide a conceptual explanation (Section 5.3 and Appendix H): 4th-order tensors decouple scene and environment into separate expert matrices, while 3rd-order tensors couple them into a single flattened expert set. This is a reasonable explanation.

## Novel Insights

The comparison between TuKA and ABC-LoRA (Appendix I) reveals an important nuance: architectural hierarchy alone (scene branch + environment branch with matrix multiplication) is insufficient—the Tucker core tensor's role as a shared interaction hub that captures *cross-dimensional couplings* (scene↔environment) is what drives the 10% SR gap. This suggests that in multi-hierarchical continual learning, the bottleneck is not just separating knowledge across dimensions but modeling their *interactions*, which tensor cores naturally provide but cascaded matrix multiplications cannot. This insight generalizes beyond VLN to any setting where multiple orthogonal sources of variation must be jointly adapted.

## Suggestions

- **Quantify expert retrieval accuracy:** Run the CLIP matching on the test set with ground-truth scene/environment labels and report the retrieval accuracy (top-1 match rate). Additionally, report performance when using ground-truth expert selection vs. CLIP-based selection to isolate retrieval errors from navigation errors.

- **Add an inference latency table:** Compare wall-clock time per navigation step for AlldayWalker vs. LoRA baselines on the same hardware. Even a single-row comparison would address the deployment concern.

- **Clarify expert selection frequency:** Explicitly state whether retrieval is per-episode or per-step, and if per-step, evaluate whether dynamic switching provides benefits over static selection within an episode.

- **Ablate individual DKIL losses:** Add a table showing average SR and F-SR with each of L_ewc, L_co, and L_es removed individually, to demonstrate which components are essential vs. incremental.

## Quality Assessment

- **Novelty:** High. The application of Tucker decomposition to PEFT for multi-hierarchical continual learning is a genuine conceptual advance over MoE-LoRA variants. The tensor-to-matrix alignment mechanism (Eq. 3) is clean and non-obvious.

- **Technical soundness:** Good overall. The mathematical formulation is correct, the DKIL strategy is well-designed, and the ablations are thorough. The main gap is the unanalyzed inference pipeline (retrieval + reconstruction).

- **Empirical support:** Strong on simulation, moderate on real-world. The 24-task benchmark with 12 baselines and extensive ablations provides solid evidence. The real-world evaluation (4 seen + 2 unseen scenarios) is encouraging but limited in scale.

- **Significance:** High for the embodied AI community. The AML-VLN problem formulation and the TuKA method address a practical deployment challenge that will become increasingly important as VLN agents move toward real-world deployment.

- **Clarity:** Adequate. The notation is dense in Section 3.3 but generally well-structured. The distinction between scene/environment hierarchies could be introduced earlier. The paper would benefit from a concrete numerical example of how Eq. 3 produces a weight matrix from the tensor factors.

---

## xFo13SaHQm

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

WithAnyone addresses the "copy-paste" artifact in identity-consistent image generation, where models over-replicate reference images rather than synthesizing identities under new conditions. The paper introduces three contributions: (1) MultiID-2M, a large-scale paired multi-identity dataset with 500k group photos and ~1M reference images; (2) MultiID-Bench, a benchmark with a novel Copy-Paste metric ($M_{CP}$) that quantifies the trade-off between identity fidelity and copy-paste artifacts; and (3) the WithAnyone model built on FLUX, using a GT-aligned ID loss, an ID contrastive loss with extended negatives, and a four-phase training pipeline that transitions from reconstruction to paired identity-conditioned generation.

## Strengths

- **Formalizing and measuring the copy-paste artifact**: The $M_{CP}$ metric (Eq. 2) is a genuine contribution—it operationalizes a widely recognized but poorly measured failure mode. Unlike raw Sim(Ref) which rewards trivial copying, $M_{CP}$ captures the *relative* bias toward the reference versus ground truth. The metric shows moderate positive correlation with human judgments (Pearson $r=0.44$, Table 7), and the GPT-4o anomaly on TV-series identities provides natural validation of its discriminative power (Sec. F.3).

- **MultiID-2M fills a real data gap**: The dataset provides 500k paired group photos with ~400 reference images per identity across ~3k identities, directly enabling training paradigms (paired supervision, extended negative pools) that reconstruction-only datasets cannot support. Table 4 shows the scale advantage over existing multi-ID datasets (PIPA: 40k, MHP: 5k).

- **GT-aligned ID loss is a clean technical innovation**: Using GT landmarks for face alignment during training (Eq. 14) avoids noisy landmark extraction from intermediate denoised images, enables ID supervision at *all* noise levels (not just $t < 0.25$ as in PortraitBooth), and leverages FLUX's single-step velocity prediction for efficiency. Fig. 7 provides empirical support that this yields lower error at low noise and more informative gradients at high noise.

- **Compelling trade-off breaking**: Fig. 5 is a strong result—WithAnyone visibly deviates from the regression curve that all other methods lie on, achieving the highest Sim(GT) while maintaining lower copy-paste than methods with comparable identity fidelity. This is the paper's central claim and it is well-supported.

## Weaknesses

### Major:

- **Confounding between data scale and methodological contribution**: The paper attributes performance gains to the training paradigm (paired data, contrastive loss, GT-aligned loss), but the ablation in Table 3 compares "FFHQ only" (70k images, no paired data, no contrastive loss) versus the full system (2M images with all components). This confounds dataset scale with loss design. A cleaner isolation—training with the same MultiID-2M data but using only reconstruction loss, or applying the proposed losses on a public dataset like FaceID-6M—would strengthen the claim that the *loss functions and training strategy* (not just data scale) are responsible for breaking the trade-off.

- **Benchmark's dependence on ground-truth limits evaluation scope**: The $M_{CP}$ metric and Sim(GT) both require a ground-truth image, restricting evaluation to reconstruction-adjacent scenarios where a target image exists. This does not fully test the model's capability for open-ended, prompt-driven generation (e.g., "put this person in a cyberpunk city" with no GT). While CLIP-T is reported for prompt adherence, the paper's central quantitative claims rest on GT-dependent metrics. The authors should clarify the scope of what MultiID-Bench evaluates and discuss its limitations for assessing truly controllable generation beyond reconstruction.

### Minor:

- **User study is underpowered**: With only 10 participants ranking 230 image groups, the study lacks sufficient statistical power for the strong claims about human preference across four dimensions. No inter-annotator agreement metric (e.g., Krippendorff's $\alpha$) is reported, making it difficult to assess the reliability of the human evaluation. The correlation analysis (Table 7) helps, but the study design itself is a limitation.

- **Limited quantitative evaluation on non-celebrity identities**: Generalization beyond celebrities is shown only qualitatively in Fig. 16 (3 examples from OmniContext). Given the model's training on celebrity data and the practical importance of generating non-public identities, quantitative results on a standard non-celebrity test set would strengthen the generalization claim.

- **Inconsistency in similarity threshold reporting**: The main text (Sec. 3) states the identity assignment threshold as 0.4, while the appendix (Sec. C.1) reports 0.5. This inconsistency, while minor, raises questions about the dataset construction's precision and should be clarified.

### Trivial:

- The negative pool ablation (Table 3, "w/o Ext. Neg.") shows a clear drop from 4096 to 63 negatives, but a graded analysis (e.g., 256, 1024, 4096) would better characterize the scaling behavior of this component.

## Nice-to-Haves

- Apply the proposed loss functions and training pipeline to a public dataset (e.g., FaceID-6M) to cleanly isolate the methodological contribution from data scale.
- Evaluate identity blending and similarity scaling as the number of subjects increases beyond 4 (e.g., 5–8 people), to stress-test the multi-ID capability.
- Report inference speed and VRAM usage relative to baselines to contextualize practical deployment trade-offs.
- Expand the user study to ≥30 participants with inter-annotator agreement reporting.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Ethical/legal concerns about Right of Publicity superseding CC licenses**: While a legitimate societal concern, the paper includes a comprehensive Ethics Statement (Sec. 7 + Appendix) addressing data sourcing (CC-licensed, publicly known figures only), anonymization (no names or identity labels in training), non-commercial release, and recommended mitigations (consent verification, watermarking, abuse monitoring). The concern as raised overstates the paper's oversight—the authors have addressed this substantially, even if not perfectly.

- **Unfair baseline comparison (baselines not tuned on MultiID-Bench)**: The paper uses "official implementations and checkpoints (or API) with default settings" (Sec. F.1). Using published configurations is standard practice and, if anything, favors the baselines by giving them their best-published performance. Per review rules, this is removed.

- **Architecture agnosticism (testing on UNet/SDXL)**: The paper explicitly builds on FLUX. Demanding verification across other architectures is scope creep—the contribution is the training paradigm and data strategy, not architecture-specific modifications. Applying to other architectures would strengthen the paper but is not a core flaw.

- **Training cost as a weakness**: The 4-phase pipeline on 8 H100s is resource-intensive but standard for large-scale diffusion model training. This is a practical consideration, not a methodological flaw, and is more appropriately a nice-to-have discussion point.

- **Demand for error bars / multiple seeds on Fig. 5**: Large-scale benchmark evaluation with single runs is standard practice in this area. While variance reporting would be ideal, its absence does not undermine the results, especially given the complementary user study and ablation support.

- **GT-aligned loss constraining pose controllability at inference**: This concern misunderstands the mechanism. The GT-aligned loss is only applied during training; at inference, no loss is computed. The paired training (Phase 3) explicitly uses different reference/GT pairs to prevent pose locking. The loss teaches the model to focus on identity features in a consistently aligned space, which should improve rather than hinder pose flexibility.

## Novel Insights

The paper reveals a fundamental tension in ID-preserving generation that has been hiding in plain sight: the community's primary evaluation metric (Sim(Ref)) inadvertently *rewards* the failure mode it should penalize. By formalizing this as the $M_{CP}$ metric and demonstrating that most SOTA methods lie on a fidelity-artifact trade-off curve, the paper exposes how optimization for the wrong signal has shaped the field's progress. The insight that paired data (where reference and target are different images of the same person) naturally breaks the reconstruction shortcut is simple but powerful—it reframes copy-paste not as a model capacity issue but as a *data objective* issue.

## Suggestions

- Add a "same data, different loss" ablation: train a model on MultiID-2M using only reconstruction loss for the same number of steps, then compare. This would cleanly demonstrate that the proposed losses (not just data scale) drive the trade-off breaking.
- Clarify the similarity threshold inconsistency (0.4 vs 0.5) between main text and appendix with a single consistent value and justification.
- Expand non-celebrity evaluation to include quantitative metrics on a holdout set of non-public identities (e.g., from OmniContext's real-human subset), not just the 3 qualitative examples in Fig. 16.

---

## ZBhZT307xx

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Final Review

## Summary

This paper provides a comprehensive empirical analysis of rule-based and model-based verifiers in RLVR for mathematical reasoning. It demonstrates that rule-based verifiers suffer from significant false negative rates (recall as low as 0.78 on challenging datasets), which worsen as policy models grow stronger, and that model-based verifiers—while improving recall—introduce vulnerability to reward hacking, with fine-tuned verifiers proving paradoxically more susceptible than off-the-shelf ones despite higher static accuracy.

## Strengths

- **Counter-intuitive finding on fine-tuned verifier fragility**: The discovery that verifiers explicitly fine-tuned for higher classification accuracy become *more* vulnerable to reward hacking during RL training (Section 5.1, Figure 3 right panel showing training-oracle reward divergence) challenges the common practice of fine-tuning verifiers and is the paper's most impactful contribution. This is not obvious and has direct practical consequences.

- **Systematic adversarial probing framework**: The construction of 13 distinct hacking patterns (Section 6, Table 9) and the evaluation of multiple verifier architectures against them provides a reusable diagnostic methodology. The finding that discriminative verifiers (xVerify) are substantially more robust than generative CoT verifiers (Table 3: xVerify near 0% vs. R1-Distill-Verifier-1.5B at 18.8% average attack success) is a concrete, actionable insight for verifier design.

- **Multi-dimensional evaluation across static, RL, and adversarial settings**: The paper evaluates verifiers not just on classification metrics but through actual RL training dynamics with oracle reward monitoring, revealing a critical gap between static accuracy and RL robustness that prior work has largely ignored.

- **Hybrid verifier design shown effective**: The cascade of rule-based then model-based verification improves RL performance by 2.3 absolute points (Table 2: 57.3 vs. 55.0 avg) while maintaining >98% precision, offering a practical improvement over current practice.

## Weaknesses

### Major:

- **Single-seed RL training results without variance estimates**: The paper explicitly states "All benchmarks are reported with a single sample due to computational constraints" (Section 4.2). For an empirical RL paper whose core claims rest on training dynamics—2.3-point hybrid improvement, reward hacking onset at ~450 iterations, performance degradation with fine-tuned verifiers—this is a significant methodological gap. RL training is notoriously high-variance; without at least 2–3 seeds or confidence intervals, it is impossible to distinguish genuine verifier effects from run-to-run noise. This applies especially to Figure 3's training curves and the hacking divergence claim.

- **Oracle reward reliability is under-validated**: The paper's central mechanism for detecting reward hacking—comparing training rewards against GPT-4o oracle rewards—assumes GPT-4o is itself robust to the hacking patterns that fool the verifiers under study. The human validation covers only 200 of 8,000 examples (2.5%), and the sampling strategy is not described as stratified by disagreement cases. If GPT-4o shares failure modes with the generative verifiers (e.g., being fooled by gibberish or adversarial prefixes in responses), the reported hacking detection could be inaccurate. A targeted evaluation of GPT-4o's robustness to the same 13 adversarial patterns would substantially strengthen the methodology.

### Minor:

- **Computational overhead of the hybrid verifier is claimed but not quantified**: The paper states the hybrid design "substantially reduces the computational load on the model-based verifier" (Section 4.1) but provides no wall-clock time, FLOP estimates, or throughput comparison between rule-only and hybrid verification. Since reward computation is on the critical path in RLVR, this is a practical gap—practitioners cannot assess the cost-benefit trade-off without this data.

- **Potential data overlap confound for fine-tuned verifier hacking susceptibility**: The R1-Distill-Verifier-1.5B is fine-tuned on 1K queries from DeepscaleR (Appendix K), which is also the RL training dataset. While the paper states these queries are "non-overlapping with the evaluation set," it does not state they are non-overlapping with the RL training prompts. If the verifier fine-tuning distribution matches the RL training distribution, the verifier may overfit to question-type-specific patterns, making it easier for the policy to find distribution-wide adversarial exploits—confounding the claim that fine-tuning *inherently* increases hacking vulnerability.

- **Limited policy model scale for RL experiments**: All RL training experiments use Qwen2.5-7B. The paper itself notes (Section 6.2) that "the policy models in our RL training are not strong enough to find and exploit these vulnerabilities" for some verifiers, and Section 3.2 shows that stronger models produce more diverse outputs that stress verifiers more. The generalizability of both the hybrid improvement and the hacking findings to larger, more capable policy models remains unclear.

- **Insufficient explanation of why discriminative verifiers are more robust**: Table 3 shows a striking gap (xVerify-3B-Ia: 0.4% average attack success vs. R1-Distill-Verifier-1.5B: 18.8%), but the paper only briefly attributes this to generative CoT reasoning being "exposed to attacks that disrupt reasoning" (Section 6.2). A deeper analysis—e.g., whether the discriminative architecture's lack of generation surface, its training objective, or its shorter context window is responsible—would significantly strengthen the practical guidance.

### Trivial:

- The framing of verifiers as the "core methodology behind various large reasoning models" (Abstract) slightly overstates their role relative to training algorithms and architecture, but this does not affect the paper's claims.

## Nice-to-Haves

- Preliminary exploration of at least one defense mechanism against the identified hacking patterns (e.g., input sanitization for empty symbols, output format constraints, or adversarial training of verifiers)—the paper identifies important vulnerabilities but leaves mitigation entirely to future work.
- An oracle-reward RL training run (using GPT-4o directly as verifier) to establish an upper bound, giving context for how much of the verification gap the hybrid approach closes.
- Correlation analysis quantifying the relationship between static verification accuracy and RL training outcomes across all tested verifiers, to rigorously support the claim that "classification accuracy does not necessarily reflect RL effectiveness."
- Testing the hybrid verifier with a stronger policy model (e.g., 32B) to assess whether hacking susceptibility scales with policy capability.
- Combining discriminative and generative verifiers in the hybrid pipeline to leverage xVerify's robustness alongside generative models' recall.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Hybrid verifier cascade error risk** (from Harsh Critic: "does the >98% precision figure account for the model incorrectly flipping a true positive from the rule-based stage?"). Removed because this is factually wrong: the hybrid design explicitly routes only rule-based negatives to the model-based verifier (Section 4.1: "the model-based verifier provides supplementary judgment only when the rule-based verifier flags a response as incorrect"), so the model never sees rule-based positives and cannot flip them.

- **Weakness: Missing comparison with Process Reward Models (PRMs)** (from transferred human reviews). Removed as scope creep—this paper studies outcome verifiers in RLVR, which is a distinct paradigm from PRM-based training. PRMs judge intermediate reasoning steps, not final answer equivalence.

- **Weakness: Hacking patterns are only constructed post-hoc, not naturally discovered** (from Spark Finder). Removed as factually wrong—Section 5.2 describes hacking patterns emerging naturally during RL training ("the policy model exploits vulnerabilities in the verifier by outputting either a single simple character or long sequences of meaningless text"), with examples from actual training runs shown in Figures 11 and 12. The Section 6 probing study then systematically generalizes these observed patterns.

- **Weakness: Need for logit/attention analysis of why verifiers fail** (from Harsh Critic). Moved to Nice-to-Have; the paper demonstrates *that* verifiers fail with concrete case studies, and deeper mechanistic analysis would strengthen but is not required for the paper's claims.

- **Weakness: GPT-4o oracle cost/latency for production systems** (from Positive Reviewer). Removed as scope creep—the oracle is used as a diagnostic instrument to detect hacking, not as a proposed component of production RL pipelines. The paper does not advocate deploying GPT-4o during training.

- **Weakness: Abstract should quantify computational overhead** (from Harsh Critic). Removed as a formatting nitpick.

## Novel Insights

The most striking insight emerging from the synthesis is the existence of a **verifier accuracy–robustness trade-off that is invisible in static evaluation**: fine-tuning a verifier to higher classification accuracy can actively *decrease* its effectiveness in RL training by making it more exploitable. This is not merely a case of overfitting—the vulnerability is architectural (generative CoT reasoning creates surface-area for adversarial manipulation that discriminative judgment does not). This suggests that the field's current practice of evaluating verifiers by static accuracy alone is not just incomplete but potentially misleading, and that verifier benchmarking must incorporate adversarial robustness as a first-class metric alongside precision and recall.

## Suggestions

- Run the RL training experiments with at least 2 additional seeds and report mean ± std for the key metrics (peak accuracy, hacking onset step if applicable). Even 3 total runs would substantially address the variance concern.
- Evaluate GPT-4o against the 13 adversarial hacking patterns from Section 6 to confirm oracle reliability; if GPT-4o is also vulnerable to some patterns, report which ones and discuss implications for the hacking detection methodology.
- Add a single table or figure comparing wall-clock time per training step for rule-only vs. hybrid verification to address the practical cost question.
- Clarify whether the R1-Distill-Verifier fine-tuning queries overlap with the RL training prompt set, and if so, run an ablation with non-overlapping fine-tuning data to isolate the effect of distributional overlap on hacking susceptibility.

---

## Ksvv8x00eo

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

CaTS-Bench introduces the first large-scale, multimodal benchmark for context-aware time series captioning, unifying numeric time series segments, rich metadata, visual line-plot images, and reference captions across 11 real-world domains. The paper also proposes tailored numeric fidelity metrics and a diagnostic Q&A suite, revealing that current VLMs largely fail to leverage visual inputs for time series reasoning.

## Strengths

- **First benchmark to unify numeric, metadata, visual, and caption modalities for TSC.** Table 1 clearly shows that existing benchmarks (TADACap, TRUCE, TACO) each miss at least one modality, lack expressive captions, or omit Q&A tasks. CaTS-Bench is the only benchmark combining all components.
- **Striking empirical finding on visual modality underutilization.** The ablation in Section 4.3 (Figure 4) shows that removing the visual input causes negligible or positive performance changes for most VLMs, and the attention analysis (Appendix I.2, Figure 7) confirms models attend to textual elements in plots rather than line trends. The plot matching Q&A task further reveals near-random model performance (Table 17) versus human ceiling of 0.95. This is a significant diagnostic insight.
- **Tailored numeric fidelity metrics address a real evaluation gap.** Standard NLP metrics (BLEU, ROUGE-L) fail to capture numeric accuracy. The Statistical Inference Accuracy (penalizing hallucination) and Numeric Score (penalizing omission, with λ_R=0.7 emphasizing recall) are specifically designed for TSC and directly measure what matters most in time series description.
- **Comprehensive multi-pronged validation of semi-synthetic captions.** The authors go beyond typical LLM-generated benchmarks by conducting: (1) manual verification of ~2.9k captions (98.6% accuracy), (2) a human detectability study (41.1% accuracy—near random), (3) diversity analysis across nine embedding models (only 2.3% near-duplicate pairs), and (4) a paraphrasing robustness experiment (Spearman 0.9266 ranking correlation across oracle styles). This sets a high standard for semi-synthetic benchmark validation.

## Weaknesses

### Major:

- **Reliance on a single oracle LLM (Gemini 2.0 Flash) for ~99.7% of test ground truth captions introduces both factual and stylistic ceiling effects.** The paraphrasing experiment (Appendix H.3) effectively addresses stylistic bias but does not mitigate *factual* bias: if Gemini makes a systematic error, it becomes part of the ground truth. The manual verification covers 72.5% of the test set at 98.6% accuracy, meaning ~1.4% of verified captions contain errors, and the remaining 27.5% are unchecked. In a benchmark, this noise can penalize models that are actually more accurate than the oracle—a model correctly reporting a value that contradicts an erroneous Gemini caption would be scored as wrong. The human-revisited subset (579 samples, 14.5% of test) is too small to fully resolve this. The authors acknowledge this in Appendix A but the concern remains substantive for a benchmark paper where ground truth reliability is paramount.

- **The benchmark's multimodal design is currently underutilized by all evaluated models, meaning the primary captioning task effectively measures text+numeric reasoning rather than true multimodal fusion.** While the authors correctly frame this as a model limitation (Section 4.3, last paragraph), it raises a practical concern: at present, CaTS-Bench's captioning task does not exercise the visual modality in any measurable way. The visual ablation (Figure 4) shows that adding plots sometimes *hurts* performance. This means the benchmark's multimodal claim is aspirational rather than operational for current models. The paper would benefit from explicitly acknowledging this and discussing what benchmark design changes (e.g., information only available visually, withholding numeric values for some test cases) would force genuine multimodal reasoning.

### Minor:

- **Lack of systematic error categorization for numeric hallucinations.** The paper reports aggregate numeric accuracy scores but does not analyze what types of errors models make (e.g., wrong trend direction vs. wrong magnitude vs. fabricated values). Appendix K provides two anecdotal cases but no taxonomy or quantitative breakdown. This limits the benchmark's diagnostic value for guiding model improvements.

- **Q&A filtering methodology introduces model-specific selection bias.** Questions are filtered by removing those answered correctly by Qwen 2.5 Omni (Section 3.4). While Appendix J.2 shows other models also struggle with the filtered set, the initial selection is inherently shaped by one model's failure modes. The paper could strengthen this by demonstrating that the filtered set also challenges humans (or at least that the difficulty ranking across question types is consistent across models and humans).

- **No domain-specific analysis of failure modes.** Results are macro-averaged across 11 diverse domains (climate, crime, health, sales, etc.). Different domains may have fundamentally different captioning challenges (e.g., seasonal climate patterns vs. sparse health data), but the paper does not break down performance or error patterns by domain, which would significantly enhance diagnostic value.

- **Human baseline is limited for Q&A tasks and absent for captioning.** The Q&A human baseline (Table 17) relies on university student volunteers (Appendix O) with no reported inter-rater agreement. For the core captioning task, there is no human baseline at all—the 579 human-revised captions are edits of LLM outputs, not independent human authoring from scratch, making it impossible to calibrate how far models are from true human performance on captioning.

### Trivial:

- The uniform 5% relative tolerance across all 11 domains (Appendix F.2) is a simplification—a 5% error in financial data may be more consequential than in climate data—but is reasonable as a default benchmark parameter and is clearly documented.

## Nice-to-Haves

- **Cross-benchmark comparison:** Evaluating the same models on TACO/TRUCE/TADACap would demonstrate what CaTS-Bench reveals that existing benchmarks miss, directly supporting the gap-filling claim.
- **Forced visual reasoning conditions:** A test subset where numeric values are withheld and only the plot+metadata are provided would create a genuine test of multimodal capability, complementing the current design.
- **Human expert captions written from scratch** (not just LLM-revised) for a larger subset would provide a stronger calibration point for evaluation reliability.
- **Sensitivity analysis on the 5% numeric tolerance** to show whether model rankings change meaningfully under tighter/looser thresholds.
- **Domain-specific failure analysis** to reveal whether certain domains systematically expose different model weaknesses.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "First large-scale" claim is misleading vs TACO's 2.46b timesteps.** The paper's claim is "first large-scale, *multimodal* benchmark" (emphasis added). TACO is numeric-only and lacks metadata, visuals, and Q&A. The qualifier is present and accurate.
- **Harsh critic: Missing discussion of computational cost of benchmark generation.** This is outside the paper's stated scope. A benchmark paper need not analyze its own generation cost as a limitation.
- **Harsh critic: No broader impact discussion of misuse (misleading financial summaries).** The paper includes an Ethical Statement. Demanding additional misuse speculation is scope creep beyond ICLR norms for benchmark papers.
- **Harsh critic: Parser artifacts and formatting issues.** Instructed to ignore.
- **Harsh critic: λ_A=0.3, λ_R=0.7 weights are "somewhat arbitrary."** The paper provides explicit justification: "to emphasize recall over precision, as omitting critical numbers is more severe than minor numeric rounding imprecisions." This is a reasoned design choice.
- **Spark finder: "No comparison showing CaTS-Bench is harder than existing benchmarks."** Moved to Nice-to-Have. The multimodal findings (visual underutilization, near-random plot matching) are qualitatively distinct from what numeric-only benchmarks can reveal.
- **Spark finder: "No ablation on human-revised subset size."** Moved to Nice-to-Have. This is an additional experiment request, not a core flaw.
- **Spark finder: "Out-of-domain evaluation is absent."** The paper's design is to evaluate within-domain on temporally held-out data. Zero-shot cross-domain evaluation is a different research question, not a missing core component.

## Novel Insights

The most striking insight from this work is the *modal collapse* phenomenon in VLMs for time series: despite being provided with visual plots that contain the same information as the numeric series, models systematically default to textual/numeric priors and achieve comparable or better performance without the visual input. This is not merely a performance deficit but a fundamental architectural limitation—attention analysis reveals models attend to axis labels and titles in plots rather than the line trends themselves. The plot matching task crystallizes this: humans achieve 0.95 accuracy while all models perform near-random (~0.25-0.34), suggesting that current VLM architectures lack the visual-numeric integration needed for even basic chart understanding. This finding implies that scaling current VLM architectures alone may not close this gap; targeted architectural interventions (dedicated chart understanding modules, contrastive visual-numeric alignment) are likely needed.

## Suggestions

- **Add a "vision-required" test subset** where the numeric series is withheld and only the plot + metadata are provided. This would create a direct, unambiguous test of visual reasoning capability and make the multimodal claim operational rather than aspirational.
- **Provide a quantitative error taxonomy** for numeric hallucinations (e.g., wrong direction, wrong magnitude, fabricated values, omitted statistics) across models, which would significantly increase the benchmark's diagnostic value for the community.
- **Expand the human-revisited subset to at least 1-2k samples** covering all 11 domains, with independent human authoring (not just LLM editing), to better validate the semi-synthetic ground truth and provide a more reliable calibration point.
- **Report domain-specific results** in the main paper (not just macro-averages) to enable the community to identify which domains are most challenging and why.

## Axis Evaluations

- **Novelty:** Moderate-to-high. The benchmark integration of all four modalities (numeric, metadata, visual, captions) plus Q&A is genuinely novel per Table 1. The numeric fidelity metrics are a useful methodological contribution. The pipeline itself is a standard LLM-generation workflow—the novelty lies in its validation.

- **Technical soundness:** Good. The validation of semi-synthetic captions is unusually thorough for a benchmark paper. The single-oracle limitation is the main soundness concern, but it is acknowledged and partially addressed. Experimental design (temporal splits, macro-averaging) is sound.

- **Empirical support:** Good. The visual underutilization finding is well-supported by multiple evidence streams (ablation, attention, Q&A). However, the lack of cross-benchmark comparison and domain-specific analysis limits the depth of empirical conclusions. The captioning results are somewhat dense but interpretable.

- **Significance:** High. Time series captioning is an important emerging task, and the finding that VLMs cannot leverage visual representations for temporal data has broad implications. The benchmark fills a clear gap and is likely to become a standard resource.

- **Clarity:** Good. The paper is well-organized with clear separation of SS and HR evaluations. The distinction between the benchmark's design intent (multimodal) and current reality (models ignore vision) could be made more explicit in the main text rather than only in the discussion.

---

## WhO6Km5Rku

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Final Review

## Summary

QubitCache proposes a KV-cache compression framework that shifts from binary token eviction to preserving relational attention structure via quantum-inspired amplitude encoding. The method retains ~15% of tokens (anchors, recent, critical) in classical memory while encoding the attention distribution of the remaining 85% into 9-qubit quantum states per 512-token segment, reconstructing soft attention weights through probabilistic measurement and interpolating value vectors via inverse distance weighting. The paper reports 7× memory reduction with 92–97% performance retention across five LLMs and six benchmarks.

## Strengths

- **Conceptual reframing of cache compression:** The insight that attention *relationships* between tokens carry more essential information than individual token representations—and that binary eviction severs these relationships—is well-motivated and supported by prior work (Abnar & Zuidema, 2020; Michel et al., 2019). This provides a principled alternative to keep/drop heuristics.

- **Strong empirical improvements on multi-hop reasoning:** On HotpotQA, QubitCache substantially outperforms token-eviction baselines (e.g., Mistral: 0.604 vs H2O's 0.487, a 24% relative gain), consistent with the claim that preserving relational structure matters most for cross-document reasoning where early tokens become critical later.

- **Honest about classical simulation:** The paper explicitly states (§3.2.2): "the current implementation operates as a classical simulation. This allows immediate deployment on standard GPU hardware." This transparency is valuable.

- **Qualitative error analysis showing reduced hallucination:** Tables 6–9 demonstrate that StreamingLLM and H2O produce factual hallucinations (e.g., "murder charges" instead of "fraud"), while QubitCache's outputs remain semantically coherent, providing concrete evidence that soft attention preservation mitigates catastrophic failures.

## Weaknesses

### Major:

- **Core theoretical claim asserted but never proven.** The abstract and introduction state: "We prove QubitCache preserves rank-*r* attention structure with bounded reconstruction error." No formal theorem statement or proof appears anywhere in the paper. For a claim presented as a foundational guarantee, this is a serious omission—especially because the bounded-error property is what supposedly distinguishes QubitCache from "catastrophic failure modes" of discrete methods.

- **"Beyond classical information-theoretic limits" claim is misleading.** The abstract claims "logarithmic compression beyond classical information-theoretic limits." However: (a) the compression is lossy (3–8% performance drop on many tasks), so information-theoretic lower bounds on lossless compression are irrelevant; (b) the implementation is a classical simulation where a 512-dimensional statevector requires O(N) classical storage per segment, not O(log N). The O(log N) claim applies only to the number of qubits on actual quantum hardware, not to the deployed system. The paper should either restrict this claim to the quantum-hardware regime or retract it.

- **Memory accounting in Table 3 is incomplete.** The reported O(L × H × 0.15S × D + log N) complexity for QubitCache counts only the 15% preserved tokens plus a "log N" quantum term. In practice, the classical simulation stores 512 complex amplitudes per segment (~32 KB), plus circuit rotation parameters, plus cached probability distributions. The actual memory footprint of the classical simulation is not O(log N) per segment—it is O(N). This should be honestly reported.

- **No latency or throughput measurements.** An efficiency paper claiming "practical feasibility" and "immediate deployment" must quantify inference-time overhead. The quantum circuit simulation, measurement sampling (with adaptive shot allocation per Eq. 8), and interpolation all add computation. Without wall-clock timing, it is impossible to assess whether the 7× memory savings come at an unacceptable latency cost.

- **"92–97% performance retention" claim is factually inaccurate for several model–task combinations.** Checking Table 1: Llama-8B on HotpotQA retains only 81.1% (0.459/0.566); Qwen2-7B on SummScreen retains 82.4% (0.220/0.267); Phi-4-mini on HotpotQA retains 75.5% (0.256/0.339), on PG19 retains 80.8%, and on PIQA retains 87.8%. Several of these fall well below the stated 92% floor. The claim should be revised to reflect the actual range.

- **No classical probabilistic baseline to isolate quantum contribution.** The method stores attention weight distributions and uses them for soft weighting—something achievable by directly storing the 512-float probability vector per segment. Without an ablation replacing the quantum module with a classical probability vector, it is impossible to determine whether the quantum formalism provides any benefit beyond what a softmax over stored attention scores would achieve.

- **Evaluation limited to 2K–8K tokens despite targeting 100K-context problems.** The introduction motivates the work with "70B models processing 100K tokens," but all experiments use sequences of 2K–8K tokens. Whether the 9-qubit segment encoding, interpolation assumptions, and compression ratios scale gracefully to 32K–128K contexts remains entirely unvalidated.

### Minor:

- **Several architectural components show zero measurable impact.** Table 5 (appendix) reports that removing "Noise Dropout" and "Entanglement Operations" produces identical reconstruction metrics (MSE = 0.0124, cosine sim = 0.943 in both cases). If these components contribute nothing, their inclusion in the architecture—and their presentation as design contributions—should be rethought or explicitly framed as placeholders for future quantum hardware.

- **The "associative memory" ablation conflates two changes.** The footnote states this is "Implemented by replacing quantum measurement with random sampling," which simultaneously removes the quantum measurement and introduces an uncontrolled variable (random noise). A cleaner ablation would use a uniform distribution or the empirical attention distribution without the quantum circuit.

- **Ablation studies use a different model and compression ratio.** Appendix A.3.1 acknowledges the ablation was conducted on "Llama-3.2-3B with 50% compression ratio"—not the 4–8B models at 15% retention used in the main experiments. This limits the validity of component-level conclusions.

- **QubitCache sometimes outperforms Full KV on specific metrics** (e.g., Mistral PIQA: 0.904 vs 0.866), which is unexpected for a lossy compression method. No explanation is provided for why compression would improve performance, raising questions about evaluation stability.

### Trivial:

- The 15–25% improvement claim on multi-hop reasoning is accurate for some models (Mistral: 24% over H2O) but significantly overstated for others (DeepSeek: 1.6%, Llama: 9.3%). The range should reflect this variability.

## Nice-to-Haves

- Evaluation on truly long contexts (32K+ tokens) to validate the method's scalability claims.
- A simple "ClassicalSoft" baseline: store the 512-dim attention probability vector directly and use it for soft weighting, to isolate whether quantum encoding adds value.
- Wall-clock latency comparison (ms/token, time-to-first-token) alongside memory metrics.
- Formal theorem and proof for the bounded reconstruction error claim, even if deferred to an appendix.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Missing comparison with SnapKV/Quest** — Per hard rules, I do not flag missing related works as I cannot confirm their existence or relevance from the paper alone.
- **Model naming inconsistencies ("Llama-8B" vs "Llama-3-8B")** — Pure formatting nitpick, removed.
- **Inconsistent notation (I_c vs I_nc, ψ_{Sm} vs ψ_{seg})** — Formatting/style nitpick, removed.
- **ScissorHand performance "unusually low"** — The critic's claim that ScissorHand's 63% drop on PG19 is unrealistic cannot be verified without knowing exact experimental conditions; at 50% retention on a perplexity task, large drops are plausible. Insufficient evidence to sustain.
- **Reproducibility concerns about Qiskit implementation** — Per hard rules, reproducibility nitpicks about implementation details are removed.
- **"First framework" novelty overstatement** — The claim is aggressive but the specific application of relational preservation to KV-cache compression is a genuine reframing. Weakened to trivial.

## Novel Insights

The most incisive observation across the reviews is that QubitCache's contribution can be decomposed into two independent mechanisms: (1) a *selection* mechanism (attention-score-based critical token retention at 15%) and (2) a *reconstruction* mechanism (soft probabilistic weighting + IDW interpolation for discarded tokens). The ablation in Table 4 shows that mechanism (1) accounts for the vast majority of the performance gap (20.4% drop when removing critical tokens), while the quantum encoding adds only 3.9% (Full QubitCache vs No Quantum). This suggests the paper's real innovation is the attention-based selection criterion combined with soft reconstruction—not the quantum formalism itself. The quantum framing, while theoretically elegant, appears to be wrapping a classical soft-attention interpolation in quantum notation without demonstrating that the quantum structure provides properties unattainable by classical probability distributions.

## Suggestions

- **Add a "ClassicalSoft" ablation** that stores the normalized attention weights as a 512-dim float vector and uses them identically for soft weighting. If performance matches QubitCache, the quantum framing adds complexity without benefit and should be honestly reframed. If it underperforms, this would be the strongest possible evidence for the quantum contribution.
- **Either prove the bounded reconstruction error theorem or remove the claim.** A proof in the appendix would be sufficient, but the current state—claiming a proof exists when none is presented—undermines credibility.
- **Revise the "92–97%" and "beyond classical limits" claims** to accurately reflect the data. The actual retention range across model–task pairs appears to be roughly 75–99%, and the logarithmic compression applies only to qubit count on quantum hardware, not to classical simulation memory.
- **Report latency numbers.** Even approximate ms/token comparisons would address the most critical gap for an efficiency-oriented contribution.

---

**Assessment by axis:**
- **Novelty:** Moderate. The relational-preservation framing for KV-cache compression is a genuine conceptual advance, but the quantum formalism may be decorative rather than functional.
- **Technical soundness:** Weak. Missing proof for central theoretical claim; misleading information-theoretic and memory-complexity claims; inaccurate performance retention figures.
- **Empirical support:** Moderate-to-weak. Broad evaluation across models and tasks, but key claims are numerically overstated, critical baselines (classical soft attention) are absent, and practical metrics (latency) are missing.
- **Significance:** Potentially moderate if claims are validated, but currently undermined by the gaps above.
- **Clarity:** Moderate. Generally well-structured, but overstatements in the abstract and introduction obscure the actual contribution and require correction.

---

## iaoAKDRAJQ

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

This paper extends the theory of adaptive smoothness—previously established for convex settings—to nonconvex optimization, showing it precisely characterizes the convergence of adaptive optimizers (Adam, AdaGrad, Shampoo) under a unified framework of "well-structured preconditioner sets." It establishes two key benefits of adaptive geometry over standard geometry: (1) adaptive smoothness enables an accelerated O(T⁻²) rate for adaptive optimizers with Nesterov momentum in the convex setting, a rate impossible under standard ℓ∞-smoothness; and (2) an analogous "adaptive variance" assumption yields dimension-free convergence for NSD in the stochastic nonconvex setting, whereas standard variance inevitably introduces dimension dependence.

## Strengths

- **Clean theoretical separation between adaptive methods and NSD.** The paper formalizes that both families exploit the same non-Euclidean geometry but through fundamentally different smoothness notions (adaptive vs. standard). The convex acceleration result (Theorem 4.3) combined with the Guzmán & Nemirovski lower bound provides a sharp separation: adaptive smoothness enables O(T⁻²) while standard ℓ∞-smoothness permits at best Ω(T⁻¹). This is a concrete, provable advantage, not just a notational distinction.

- **Novel matrix inequality (Lemma 3.3) for general preconditioner sets.** Extending the nonconvex convergence analysis from diagonal to arbitrary well-structured preconditioner sets requires handling noncommutativity of matrix preconditioners. Lemma 3.3 and its supporting Lemma C.1 (relating differences of positive definite matrices to differences of their logarithms) provide a general bound with a log d penalty for noncommutative cases and no penalty for commutative (diagonal) cases. This is a technical contribution of independent interest.

- **Adaptive variance and dimension-free NSD rates.** The parallel between adaptive smoothness (enabling acceleration) and adaptive variance (enabling dimension-free rates) is conceptually elegant. Theorem 4.5 gives a dimension-free NSD rate under adaptive variance, while Theorem 4.7 proves dimension-dependence is unavoidable under standard variance for ℓ∞ geometry—establishing a genuine separation, not merely an artifact of the analysis.

- **Unified algorithmic framework.** Algorithm 1, parameterized by a well-structured preconditioner set H, recovers AdaGrad, Adam, AdaGrad-Norm, full-matrix AdaGrad, and one-sided Shampoo as special cases. The convergence theorems (3.1, 3.2, D.2, D.7, D.8) apply uniformly across these methods.

## Weaknesses

### Major:

- **The strongest benefit (acceleration) is restricted to convex losses.** The paper motivates itself through the lens of deep learning optimizers (Adam, Muon, Lion), yet the O(T⁻²) acceleration in Theorem 4.3 applies only to convex functions. For the nonconvex setting—which is the practically relevant regime—the paper shows only that adaptive and standard smoothness differ, but does not establish that adaptive smoothness confers any rate advantage. This creates a gap between the motivation (explaining deep learning optimizer success) and the delivered theoretical benefit. The paper should explicitly acknowledge this limitation and discuss whether the convex separation suggests analogous (but unproven) benefits in nonconvex settings, or whether there are fundamental barriers.

- **The adaptive smoothness constant can be up to d times larger than standard smoothness (Proposition 2.5).** Since Λ_H(f) ≤ d · L_{∥·∥_H}(f), the adaptive smoothness bound in the nonconvex convergence rate could be substantially worse than the NSD rate in terms of problem-dependent constants. The paper argues adaptive methods "automatically identify the best geometry," but if Λ_H(f) ≈ d · L_{∥·∥_H}(f) in practice, the asymptotic advantage is offset by constant factors. There is no discussion of when or whether Λ_H(f) is close to L_{∥·∥_H}(f) for realistic loss landscapes, leaving the practical significance of the theoretical framework unclear.

### Minor:

- **The adaptive variance assumption (Definition 4.1) is stronger than standard variance, and its practical validity is unverified.** Adaptive variance requires uniform control of noise over all preconditioners H ∈ H with Tr(H) ≤ 1. The paper shows it is weaker than bounded covariance (Proposition B.10), but does not discuss whether common noise sources (mini-batch sampling, label noise) actually satisfy the adaptive variance bound in practice. Without some evidence— even qualitative—that this assumption is realistic for neural network training, the dimension-free result remains a theoretical curiosity.

- **The log d factor in the nonconvex convergence rate for general well-structured H (Theorem 3.1).** For non-diagonal (noncommutative) preconditioner sets, the convergence rate picks up a log d factor that is absent in the diagonal case. The paper identifies this as arising from noncommutativity but does not establish whether this factor is tight or an artifact of the proof technique. Given that methods like Shampoo involve non-diagonal preconditioners, the log d factor could be significant in high dimensions.

### Trivial:

- **Equivalence between weighted, cumulative, and EMA variants** is stated briefly in Section 3.2 but the precise hyperparameter mappings (e.g., η_W = η_E/√(1−β)) are deferred to the appendix. Stating these in the main text would improve readability for practitioners.

## Nice-to-Haves

- **Empirical validation of the adaptive vs. standard smoothness gap.** Even a small-scale experiment (e.g., on logistic regression or a simple neural network) measuring Λ_H(f) vs. L_{∥·∥_H}(f) and adaptive vs. standard variance would substantially strengthen the practical relevance of the theoretical framework.

- **Experiments confirming the acceleration separation (Theorem 4.3).** A convex benchmark (e.g., ℓ∞-smooth logistic regression) comparing Algorithm 2 against NSD would directly validate the paper's core claim that adaptive smoothness enables acceleration unattainable under standard smoothness.

- **Discussion of whether the convex acceleration insight extends to nonconvex settings.** Even without a formal theorem, qualitative reasoning about what barriers exist for nonconvex acceleration under adaptive smoothness would guide future work.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Abstract lacks precision about dimension-free claim.** The abstract already qualifies the statement with "for certain non-Euclidean geometry," which is precise. Removed as a nitpick.

- **Weakness: Algorithm 2 requires knowledge of D.** The paper addresses this in Remark 4.4 and Appendix E.2 with a projected variant (Algorithm 8, Theorem E.5) that removes the dependence on D. The concern is already handled.

- **Weakness: Comparison with Kovalev (2025a/b) may be apples-to-oranges.** The paper correctly notes that using standard smoothness (which is ≤ adaptive smoothness) yields a tighter bound. This is a valid, precise comparison—removed as unfounded.

- **Weakness: Computational cost of computing gradients of the modified loss f_{α_t,x̄_t}.** This is a standard Nesterov acceleration construction; the gradient of the modified loss requires one gradient evaluation of f at a shifted point, which is no more expensive than standard Nesterov momentum. Removed as factually wrong.

- **Weakness: Citation density and positioning as extension of Xie et al. (2025b).** Building on prior work with genuine novel contributions (nonconvex extension, acceleration, adaptive variance) is standard practice. Removed as not a real weakness.

- **Weakness: Missing related works.** Per rules, not included.

- **Weakness: Formatting and parser artifacts.** Per rules, removed as style nitpick.

- **Weakness: Reproducibility of hyperparameters.** Per rules, removed.

- **Weakness: Generalizing beyond well-structured preconditioner sets.** The paper's scope is explicitly about well-structured sets; demanding generalization beyond this is scope creep.

- **Weakness: Complexity/density of proofs.** Generic criticism applicable to any mathematical paper; removed as not specific.

## Novel Insights

The duality between "adaptive smoothness enables acceleration" and "adaptive variance enables dimension-free rates" reveals a deeper structural principle: under non-Euclidean geometry, averaging (of iterates for acceleration, of gradients for variance reduction) can fail to reduce norms effectively, because the dual norm ∥·∥_{H,*} is the *infimum* of individual dual norms rather than the norm at any fixed H. Adaptive assumptions circumvent this by ensuring uniform geometric control that makes averaging meaningful again. This suggests that the practical success of adaptive methods may stem not from any single geometric alignment but from a stronger "uniform adaptivity" property of the loss landscape that simultaneously enables both acceleration and variance reduction—a property that standard smoothness/variance simply cannot capture.

## Suggestions

- Add an explicit "Limitations" paragraph in the main text acknowledging that (1) acceleration is proven only for convex losses, (2) the magnitude of the adaptive-smoothness-vs-standard-smoothness gap in practice is unknown, and (3) the adaptive variance assumption requires empirical validation.

- State the hyperparameter equivalences between weighted/cumulative/EMA variants in the main text (Section 3.2), not just the appendix, since this directly affects how readers interpret the convergence guarantees for standard Adam.

- In Section 4.2, add a brief discussion of the computational overhead (if any) of the projected variant (Algorithm 8) versus standard Adam, to help practitioners assess feasibility.

- Consider adding a simple 2D visualization (extending Figure 1) that contrasts convergence trajectories of adaptive methods vs. NSD on a function where Λ_H(f) ≪ d · L_{∥·∥_H}(f), to build intuition for when adaptive smoothness provides a practical advantage.

---

## 7yvz93kBw9

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Final Review

## Summary

The paper proposes D²GS, a framework for improving sparse-view 3D Gaussian Splatting by addressing two identified failure modes: near-field overfitting (excessive Gaussians) and far-field underfitting (insufficient Gaussians). The method introduces a Depth-and-Density Guided Dropout (DD-Drop) mechanism that probabilistically drops Gaussians based on local density and depth scores, and a Distance-Aware Fidelity Enhancement (DAFE) module that reinforces supervision in distant regions using monocular depth priors. Additionally, the paper proposes an Inter-Model Robustness (IMR) metric based on 2-Wasserstein distance and optimal transport to quantify the stability of learned Gaussian distributions across independent training runs.

## Strengths

- **Clear and well-validated failure mode diagnosis.** The observation that sparse-view 3DGS suffers from spatially imbalanced Gaussian distributions—overfitting near the camera and underfitting far away (Figure 1, Section 3.1)—is specific, well-illustrated, and directly motivates the two complementary modules. The quantitative evidence (e.g., 11,450 vs. 6,112 Gaussians in the near field) grounds the motivation concretely rather than relying on vague claims.

- **Novel evaluation metric with theoretical grounding.** The IMR metric (Section 3.4) addresses a genuine gap in the literature: 2D image metrics cannot assess the stability of the 3D representation itself. The use of Wasserstein distance between Gaussian mixture distributions, with the Bures metric approximation and Sinkhorn solver, provides a principled formulation. Table 3 demonstrates that D²GS achieves the lowest IMR, and this is a fresh perspective for evaluating 3DGS robustness.

- **Consistent and meaningful improvements across multiple benchmarks and settings.** D²GS achieves the best results on LLFF (3-view, 6-view, both resolutions), MipNeRF360, and DTU across PSNR, SSIM, LPIPS, and AVGE. The gains over strong baselines like DropGaussian (+0.59 dB PSNR on LLFF 1/8, +0.55 dB on LLFF 1/4) and CoR-GS (+0.9 dB) are non-trivial. The ablation studies (Tables 4, 5, 6) systematically validate each component.

- **Principled soft dropout design that avoids DropGaussian's hard selection pitfalls.** The discussion in Appendix C clearly articulates why hard top-k dropout causes persistent suppression and over-suppression of detail-rich regions, and how DD-Drop's probabilistic mechanism avoids these issues. This design insight—*how* guidance signals are applied matters more than *what* signals are used—is valuable for the community.

## Weaknesses

### Major:

- **No analysis of sensitivity to depth estimation errors.** The DAFE module relies critically on monocular depth estimates to construct far-field masks. While Table 6 shows DAFE works with different depth estimators, all three estimators (MiDaS, DPT, DepthAnything V2) are strong modern models likely producing qualitatively similar depth maps. What happens when depth estimates are systematically wrong—e.g., for reflective surfaces, textureless regions, or scenes with depth inversions? The paper does not include any experiment with injected depth noise, corrupted depth maps, or documented failure cases. Since the paper positions itself around "stability" and "robustness," this gap is significant: the method could be introducing a brittleness that is not tested.

- **The IMR metric's correlation with perceptual/rendering quality is not validated.** The paper introduces IMR as a novel metric and shows D²GS achieves the best IMR (Table 3) and best PSNR (Table 1). However, there is no systematic analysis demonstrating that lower IMR *correlates* with higher rendering quality across methods and scenes. A scatter plot of IMR vs. PSNR/LPIPS across the 10 independent runs and across baselines would establish whether IMR is a meaningful proxy for quality, or whether it simply happens to agree for D²GS. Without this, IMR risks being a metric that is theoretically motivated but practically unvalidated.

### Minor:

- **Incomplete isolation of guidance signals vs. dropout softness.** The ablation in Table 4 progressively adds components, and Table 1 compares against DropGaussian (random dropout). However, there is no "soft random dropout" baseline—i.e., probabilistic dropout with the same time-varying schedule but without depth/density guidance. This makes it hard to fully separate the contribution of the *guidance mechanism* from the contribution of switching from hard top-k to soft probabilistic dropout. The comparison with DropGaussian provides indirect evidence, but a direct ablation would be more conclusive.

- **Computational cost of IMR is not reported.** While training time is reported (Table 7), the time required to compute IMR for a single scene (10 pairwise Wasserstein distances between Gaussian mixtures, each requiring Sinkhorn optimization over ~10k Gaussians) is not quantified. This matters because if IMR takes hours per scene, its utility for routine benchmarking is limited. The depth-stratified sampling to 10k Gaussians is mentioned but its effect on accuracy is not analyzed.

- **The Taylor approximation for the Bures metric may be inaccurate under high variance.** The first-order Taylor expansion (Eq. 11, Appendix A) assumes small deviations Δ between covariance matrices. Under sparse-view conditions where independently trained models can diverge significantly, this assumption may be violated. The paper does not discuss the approximation error empirically (e.g., comparing the approximate vs. exact Bures distance on a subset of Gaussians where exact computation is feasible).

### Trivial:

- **DropGaussian baseline reproducibility.** Appendix E notes difficulty reproducing DropGaussian's reported results. While transparency is appreciated, this raises a fair concern about whether the re-implemented baseline was given equal hyperparameter tuning effort. However, the authors' implementation appears to be used consistently, and the improvements are substantial enough that this is unlikely to be the sole explanation.

## Nice-to-Haves

- Comparison with feed-forward sparse-view methods (PixelSplat, MVSplat, HiSplat) to contextualize where optimization-based approaches stand against generalizable ones, though these are fundamentally different paradigms.
- A smooth alternative to the discrete depth layering (near/middle/far tertiles with hard attenuation factors λ_middle, λ_far), which could reduce the heuristic feel of the global mechanism and potentially improve generalization across scene types.
- Reporting rendering FPS to confirm the method preserves 3DGS's real-time advantage, since the added modules only affect training.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: DTU results relegated to appendix.** This is a space/formatting concern, not a scientific weakness. The results are present and complete.
- **Weakness: Missing broader impact discussion.** While sometimes expected at ICLR, the absence of a broader impact statement is not a scientific weakness of the method.
- **Weakness: No variance (±std) reported for PSNR/SSIM.** Reporting single-run results is the norm in 3DGS papers. The paper already goes beyond this by running 10 models for the IMR metric. Demanding variance for all metrics when it's not community standard is unreasonable.
- **Weakness: Min-max normalization may require hyperparameter retuning across scenes.** The cross-dataset results (LLFF, MipNeRF360, DTU) with very different scales already demonstrate that the method generalizes. The min-max normalization is precisely what makes it scale-invariant.
- **Weakness: L1 loss in DAFE insufficient for high-frequency details; should use perceptual loss.** This is speculative—the ablation shows DAFE works with L1. Suggesting alternative losses is a nice-to-have, not a weakness.
- **Weakness: Why is equal weighting (ω_depth=0.5) optimal?** This is a curiosity about the ablation outcome, not a methodological flaw. The ablation itself answers the question by showing the method is not overly sensitive to this parameter.

## Novel Insights

The paper's most underexploited insight is the distinction between *how* guidance signals are applied versus *what* signals are used for dropout. The discussion in Appendix C makes a compelling case that DropGaussian's failure with selective dropout stems not from the depth/density signals being uninformative, but from the hard top-k selection mechanism causing persistent suppression. This suggests that the community's current practice of comparing dropout strategies by their *signal* (random vs. gradient vs. distance) may be missing the more important axis of *application mode* (hard vs. soft). This insight generalizes beyond this paper and could inform regularization design in other optimization-based 3D reconstruction methods.

## Suggestions

- Add an experiment with synthetic depth noise (e.g., Gaussian noise, scale perturbations, or random pixel dropout on the depth maps) to quantify DAFE's sensitivity and demonstrate the method's robustness bounds. Even a simple analysis showing that D²GS degrades gracefully (rather than catastrophically) would significantly strengthen the robustness narrative.
- Provide a scatter plot correlating IMR with PSNR across the 10 training runs for each method to validate IMR as a meaningful quality proxy. If the correlation is strong, this validates the metric; if weak, it is important to report and discuss.
- Report the wall-clock time for computing IMR per scene so users can assess its practical utility as a benchmarking metric.

---

## FlcMckO6x5

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Final Review

## Summary

This paper develops the theoretical foundations for Separable Neural Networks (SepNNs)—architectures that factorize multivariate functions into linear combinations of univariate functions parameterized by lightweight factor MLPs. The contributions are threefold: (1) a universal approximation theorem for CP, TT, and Tucker SepNNs using Stone-Weierstrass arguments; (2) NTK regime characterization showing deterministic kernel convergence under infinite width+rank and random kernel under infinite width+fixed rank; and (3) an efficient Separable Preconditioned Gradient Descent (SepPGD) method that alleviates spectral bias with O(nD) complexity on D-dimensional grids with n points per dimension.

## Strengths

- **Unified approximation theory for multiple SepNN types:** Theorem 1 provides a clean, unified proof of universal approximation for CP, TT, and Tucker SepNNs, extending prior work (Cho et al., 2023) from bivariate to general multivariate settings. The Stone-Weierstrass approach is elegant and the verification that the separable algebra separates points, contains constants, and is closed under algebraic operations is done carefully for all three decomposition types.

- **NTK decomposition revealing structural insight:** Lemma 1 shows the SepNN NTK decomposes as a weighted sum of factor NTKs, which is a non-trivial structural result. The distinction between deterministic NTK (Theorem 2: infinite width + infinite rank) and random NTK (Corollary 1: infinite width + fixed rank) provides genuine insight into why practical SepNNs with small rank exhibit different training behavior than wide standard networks.

- **SepPGD exploits separable structure for efficient preconditioning:** The key algorithmic insight—decomposing the large n^D × n^D preconditioner into D smaller n × n factor preconditioners via the Kronecker structure (Lemma 2)—is both theoretically justified and practically significant. Table 1 clearly shows the complexity advantage over prior NTK-PGD methods for the gradient formulation step.

- **Consistent empirical improvements across applications:** SepPGD shows meaningful PSNR gains (e.g., 26.48→33.30 on Plane image, Table 5) and faster convergence across KRR, INRs, and PINNs, with useful ablation studies on rank R, modulation parameter k, and update frequency (Tables 2–7).

## Weaknesses

### Major:

- **Theory-practice gap in NTK regime:** The deterministic NTK (Theorem 2) and the spectral bias analysis (Eq. 5) both require W,R→∞, but practical SepNNs use small fixed rank (e.g., R=64–500 in experiments). Under fixed rank, Corollary 1 shows the NTK converges to a *stochastic* kernel, and Remark 3 admits that "the training dynamic can not be characterized uniformly using a fixed NTK matrix." Since SepPGD is designed based on infinite-rank spectral properties yet evaluated in the fixed-rank regime, there is no theoretical guarantee that the preconditioner correctly identifies and adjusts the relevant eigenmodes when the NTK is random. The paper acknowledges this gap (Section A.1.2, A.4) but only offers heuristic probability bounds (Chebyshev inequality) and speculative connections to random feature models. For a paper whose core contribution is the interplay between NTK theory and preconditioning, this gap between the regime where the theory holds and the regime where the method is applied is a significant weakness.

- **SepPGD cannot precondition PDE residual loss in PINNs:** Appendix A.12 states: "For the PDE residual loss, which involves derivatives, we do not employ the SepPGD algorithm, as extending PGD to derivative-based losses requires substantially different algorithmic treatment." In physics-informed learning, the PDE residual loss is often the dominant and most challenging component—especially in data-scarce regimes where PINNs are most needed. That SepPGD can only accelerate the data-fitting and boundary/initial condition components significantly limits its utility in the very scientific ML domain the paper targets. This limitation should be discussed prominently in the main text, not relegated to the appendix.

- **NTK analysis limited to CP SepNN; TT and Tucker extensions are unproven:** The paper's NTK theory (Lemma 1, Theorem 2, Corollary 1) is derived exclusively for the CP formulation. Footnote 1 states: "the NTK analysis is primarily conducted for the CP SepNN, while we believe it can be readily extended." Similarly, Section A.1.2 says extensions are "a valuable direction for future research." Since Theorem 1 covers all three decomposition types and the introduction presents TT and Tucker as first-class SepNN variants, the absence of any NTK derivation (even a sketch) for these cases leaves the theoretical contribution incomplete.

### Minor:

- **No explicit approximation error rates:** Theorem 1 proves existence of approximations but does not quantify how rank R or width W must scale with dimension D or target function smoothness to achieve ε-approximation. The paper acknowledges this (Section A.1.2): "our current theoretical analysis... does not yet provide explicit approximation error rates in terms of network rank or width." This limits the practical guidance the theory can offer for hyperparameter selection.

- **No empirical validation of the spectral bias alleviation mechanism:** The paper's core claim is that SepPGD alleviates spectral bias by adjusting the NTK eigenvalue distribution. Figure 1(d) shows the *initial* eigenvalue spectrum, but no figure tracks how eigenvalues evolve during training with vs. without SepPGD. Plotting the NTK condition number or eigenvalue decay over training steps would directly validate that the mechanism works as theorized, rather than only observing faster convergence as indirect evidence.

- **Experiments limited to D≤3 despite theoretical advantage growing with D:** The O(nD) vs. O(n^D) efficiency gap is the central practical motivation, yet all experiments use D=2 or D=3 where the advantage is modest. Demonstrating the method on D≥5 problems would substantiate the scaling claims that distinguish SepPGD from prior work.

### Trivial:

- The notation in Definition 1 and Eq. (8) is dense with tensor operations (unfold, fold, mode-d products). While standard in the tensor literature, a brief intuitive explanation or pseudo-code algorithm in the main text would improve accessibility for the broader ICLR audience.

## Nice-to-Haves

- Derivative-aware preconditioning that extends SepPGD to PDE residual losses, making the method fully applicable to PINNs.
- High-dimensional benchmarks (D≥5) to empirically demonstrate the scaling advantage.
- Explicit non-asymptotic convergence or generalization bounds for the fixed-rank regime, even if loose, to partially bridge the theory-practice gap.
- Comparison with domain-specific SOTA INR methods (e.g., SIREN + Fourier features, TensoRF) to verify that convergence gains do not come at the cost of final representation quality.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Complexity claim is misleading because handling the residual tensor requires O(n^D):** The harsh critic argued that the O(nD) claim ignores the cost of handling the n^D residual. However, SepNNs on grid inputs exploit the separable structure to compute forward passes and gradients without materializing the full n^D output tensor (this is the well-established efficiency advantage of SepNNs per Liang et al., 2022; Cho et al., 2023). The O(nD) claim in Table 1 specifically refers to the gradient formulation complexity, and Remark 4 separately discusses preconditioner construction complexity. The claim is accurate for what it states, though the paper could be clearer about what is and isn't included in the O(nD) figure.

- **Unfair comparison with MSK because MSK runs out of memory:** The paper provides comparisons with MSK where feasible and explains the memory limitation. The asymmetry (full-batch SepPGD vs. MSK that cannot run full-batch) actually demonstrates SepPGD's advantage, not an unfair comparison. Per the rules, this concern is removed.

- **Formatting and style issues:** Removed per rules.

- **Missing related works:** Removed per rules as we cannot verify existence of specific references.

- **Reproducibility concerns about undisclosed hyperparameters:** The paper provides extensive ablation studies (Tables 2–7) and detailed experimental settings in Appendix A.12. Removed per rules.

- **Demand for confidence intervals or multiple random seeds in large-scale benchmarks:** Removed as nice-to-have; single-run evaluation with convergence curves is standard in this area.

## Novel Insights

The decomposition of the SepNN NTK into a weighted sum of factor NTKs (Lemma 1) reveals that the spectral bias of SepNNs has a fundamentally different origin than standard MLPs: it arises from the *product* structure across dimensions (the a_d vectors) compounding with the individual factor NTK spectra. This means the effective condition number of the SepNN NTK can be much worse than any individual factor NTK, as the eigenvalue decay is multiplicative across dimensions. This structural insight explains why SepNNs are particularly prone to spectral bias and why factor-level preconditioning (SepPGD) is both necessary and sufficient—the Kronecker structure means that adjusting each factor's spectrum independently propagates to the full NTK through the product structure.

## Suggestions

- Add a figure tracking NTK eigenvalue distribution (or condition number) over training steps with and without SepPGD to directly validate the spectral bias alleviation mechanism.
- Discuss the PINN PDE residual loss limitation in the main text (Section 5 or Section 6), not only in the appendix, and characterize the scenarios where SepPGD provides the most benefit within PINNs (data-rich vs. data-scarce).
- Provide even a rough sketch of how the NTK analysis extends to TT and Tucker SepNNs, or explicitly scope the NTK contribution to CP only in the title/abstract to avoid overclaiming.
- Add at least one experiment with D≥4 to demonstrate the scaling advantage that motivates the method.

---

**Axis evaluations:**

- **Novelty:** Moderate-to-high. The NTK derivation for SepNNs and the SepPGD algorithm are novel. The approximation theorem is a natural but non-trivial extension of prior bivariate results.

- **Technical soundness:** Moderate. The proofs for approximation theory and NTK convergence under infinite rank are rigorous. However, the gap between infinite-rank theory and fixed-rank practice is a real soundness concern for the core claim that SepPGD alleviates spectral bias via the theorized mechanism.

- **Empirical support:** Moderate-to-strong. Consistent improvements across tasks, but limited to low dimensions and lacking direct validation of the spectral mechanism. The PINN experiments are incomplete due to the PDE residual exclusion.

- **Significance:** Moderate-to-high. SepNNs are increasingly important in scientific ML, and a principled optimization method with theoretical backing addresses a real need. The scope is narrower than claimed (grid inputs, CP only, no PDE residual preconditioning).

- **Clarity:** Moderate. The logical flow is clear, but the dense tensor notation in Definition 1 and the scattered discussion of limitations across main text and appendices reduce accessibility.

---

## Rt9SeEAMWv

- GT: Reject (avg 4.8)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

This paper introduces a framework for deriving worst-case generalization bounds over data-dependent random sets (e.g., optimization trajectories) by combining a novel "random set stability" notion with Rademacher complexity and topological complexity measures. The key advance is replacing the intractable mutual information terms that appear in prior topological/fractal generalization bounds (e.g., Simsekli et al., 2020; Andreeva et al., 2024) with a stability parameter $\beta_n$, yielding the first such bounds that are in principle fully computable. The framework recovers classical stability bounds and uniform convergence as special cases and is validated empirically on ViT and GraphSAGE.

## Strengths

- **Novel and well-motivated stability formulation:** Random set stability (Assumption 3.1) explicitly accounts for algorithmic randomness $U$ in trajectory-based analysis, addressing a real gap in Foster et al. (2019)'s hypothesis set stability which ignores $U$. The connection to classical uniform argument stability via Lemma 3.2 provides a systematic path to verifying the assumption, and Corollary 3.3 demonstrates it concretely for projected SGD.
- **Removal of intractable IT terms:** The framework successfully replaces mutual information terms — which can be infinite and are computationally intractable in general — with a stability parameter that is interpretable and empirically estimable. This is a genuine advance over the PAC-Bayesian random set bounds of Dupuis et al. (2024) and the topological bounds of Andreeva et al. (2024).
- **Unified framework recovering classical results:** The free parameter $J$ interpolates between single-iterate stability bounds ($J=1$, Corollary 3.5) and uniform convergence over fixed hypothesis sets ($J=n$, Corollary 3.6), demonstrating that the framework is not ad hoc but subsumes known settings.

## Weaknesses

### Major:

- **Gap between "fully computable" claim and empirical validation:** The paper claims to provide "the first fully computable topological bounds" (Abstract, Section 4.1), yet the numerical bounds in Table 1 bypass the specific topological coefficients from Theorems 4.3/4.4. Section 5.1 explicitly states: "To avoid the computationally costly evaluation of Lipschitz constants, we estimate a simple upper bound on the Rademacher complexity... we use Massart's lemma." This means the topological quantities $\mathbf{E}_\alpha(W_{S,U})$ and $\mathrm{PMag}(s(\lambda) \cdot W_{S,U})$ appear in the theoretical bounds but are never plugged into a numerical bound evaluation — they are only analyzed for correlation. The "fully computable" characterization is accurate in the sense that no term is fundamentally intractable (unlike mutual information), but the empirical section does not validate the tightness of the topological bounds themselves.

- **Theory-experiment algorithm mismatch:** Corollary 3.3 establishes random set stability specifically for projected SGD under Lipschitz/smooth assumptions. However, all experiments use the AdamW optimizer (Section 5, Appendix C.1), whose adaptive step sizes violate the fixed step-size regime assumed in the stability proof. While the general framework (Assumption 3.1) is optimizer-agnostic, no theoretical result establishes that ADAM satisfies random set stability. The empirical $\beta_n$ estimates thus lack the theoretical grounding that would connect them to the generalization bound.

- **Stability parameter scaling with trajectory length $T$:** Lemma 3.2 sums iterate-level stability $\delta_k$ over $k=1\ldots T$, and the paper notes the worst case yields $\beta_n = O(T^2/n)$. In modern deep learning, $T$ is typically very large (many epochs). If $\beta_n$ grows with $T$, the bound may become vacuous precisely for the long training runs where generalization is most interesting. The paper does not discuss whether $\beta_n$ can remain bounded or decrease as $T \to \infty$ (e.g., due to convergence), which limits the practical applicability of the framework.

### Minor:

- **No baseline bound comparison:** Table 1 reports the proposed bound values but does not compare them against standard uniform stability bounds (e.g., Hardt et al., 2016) or simple Rademacher bounds without topological terms. Without this ablation, it is unclear whether the topological complexity terms actually improve the bound or whether the bound's informativeness is driven entirely by the stability parameter.

- **Optimistic stability estimation without sensitivity analysis:** Algorithm 1 approximates the supremum over $\mathcal{Z}$ using 500 held-out points, which the authors acknowledge yields an "optimistic estimation" of $\beta_n$. Since $\beta_n^{1/3}$ multiplies the entire bound, underestimation could significantly overstate tightness. No sensitivity analysis (e.g., doubling $\beta_n$) is provided to assess robustness.

- **Slower rate trade-off insufficiently justified:** The bound scales as $\beta_n^{1/3}$, yielding roughly $O(n^{-1/3})$ when $\beta_n = O(1/n)$, which is slower than the classical $O(n^{-1/2})$ Rademacher rate or $O(1/n)$ stability rate. Section 4.1 calls this "a deliberate trade-off to maintain boundedness" but does not identify concrete regimes where the IT terms in prior bounds are provably infinite, which would be the strongest motivation for accepting the slower rate.

### Trivial:

- The assumption that $\beta_n^{-2/3}$ is an integer divisor of $n$ (Theorems 4.3, 4.4) is a proof convenience that could be handled by rounding with minor constant adjustments, rather than imposed as a condition on $\beta_n$ and $n$.

## Nice-to-Haves

- High-probability bounds (currently only in-expectation), which would increase practical utility for single-run training guarantees
- A random-labels / overfitting experiment to verify that the bound degrades appropriately when generalization fails
- Sensitivity analysis for the $\beta_n$ estimation bias
- Decomposition plot separating the stability term contribution from the topological term contribution in the bound
- Discussion of computational scalability of persistent homology / magnitude computations for very large models

## Removed Points

These points are flagged to be removed; treat them with caution:

- **Weakness: Dataset scale too small (CIFAR-100, n up to 10,000).** Generic weakness requesting larger datasets when the current scale is sufficient for validating a theoretical framework's structural claims. (Soft rule: weaken generic "need more data" criticism.)
- **Weakness: Missing related works on randomized stability extensions beyond Foster et al. (2019).** Cannot confirm existence of specific works without external sources. (Hard rule: no missing related works.)
- **Weakness: Code not yet released / reproducibility concerns.** Paper states implementation will be available upon publication and provides seeds, hyperparameters, and detailed appendices. (Hard rule: do not question availability of cited/referenced resources.)
- **Weakness: Formatting issues in equations (parser artifacts).** Explicitly excluded per instructions. (Hard rule: remove formatting nitpicks.)
- **Weakness: Expectation-only bounds are a fundamental flaw.** The paper explicitly acknowledges this limitation (Section 6), and expected bounds are standard in the stability literature (Bousquet & Elisseeff, 2002). Demanding high-probability bounds is scope creep for this contribution. (Soft rule: weaken criticisms demanding practices outside paper's scope; moved to nice-to-have.)

## Novel Insights

The interplay between stability and topological complexity uncovered in this paper is genuinely insightful. The product structure $\beta_n^{1/3} \cdot \log \mathbf{C}(W_{S,U})$ in Theorem 4.4 reveals that topological complexity becomes more relevant to the bound as $n$ grows (because $\beta_n$ decreases), while stability dominates at small $n$. This is empirically confirmed by the changing slope of the topological complexity vs. generalization gap regression as $n$ varies (Figures 2–3). This coupling suggests that the informativeness of topological measures for generalization is not universal but depends on the stability regime — a nuanced finding that prior work using only correlations could not articulate.

## Suggestions

- In Table 1, add a column showing the bound value using only the stability term ($2\beta_n$ from Corollary 3.5) alongside the full bound, so readers can directly assess the value added by the Rademacher/topological component.
- Provide at least one experiment using SGD (not ADAM) to close the theory-experiment loop, or add a brief theoretical remark on conditions under which adaptive optimizers satisfy random set stability.
- Discuss concrete examples or regimes where mutual information terms in prior topological bounds are provably infinite or unbounded, to sharpen the motivation for accepting the slower $O(n^{-1/3})$ rate.

---

**Axis Evaluations:**

- **Novelty:** Strong. Random set stability (Assumption 3.1) is a genuine conceptual advance that plausibly bridges two previously disconnected lines of work (stability theory and topological generalization bounds).
- **Technical soundness:** Good with caveats. The theoretical derivations are rigorous and the recovery of classical results is clean. The main concern is the theory-experiment gap (ADAM vs. SGD) and the $T$-dependence of $\beta_n$, which are not adequately addressed.
- **Empirical support:** Moderate. The correlations between topological complexity and generalization are convincing and the stability parameter shows meaningful variation with hyperparameters. However, the specific topological bound values from Theorems 4.3/4.4 are never numerically evaluated, and no baseline comparison is provided.
- **Significance:** Strong, conditional on the framework being extended to cover the optimizers actually used in practice. Removing IT terms from topological bounds is a significant step toward making this line of theory practically relevant.
- **Clarity:** Good. The progression from classical stability → random set stability → Rademacher bound → topological applications is logical. Notation is heavy but generally well-defined.

---

## cZFgsLq8Gs

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary
DeepScientist is a multi-agent system that models scientific discovery as a Bayesian Optimization problem with a persistent Findings Memory, using an iterative three-stage loop (hypothesize, implement/verify, analyze/report) with UCB-based selection to balance exploration and exploitation. Over ~20,000 GPU hours, it generated ~5,000 ideas, validated ~1,100, and produced 21 progress findings that surpass human SOTA methods on three frontier AI tasks (Agent Failure Attribution, LLM Inference Acceleration, AI Text Detection), with the best improvements being 183.7%, 1.9%, and 7.9% respectively.

## Strengths
- **Demonstrated SOTA-surpassing results on established benchmarks**: The system produces methods (A2P, ACRA, PA-TDT) that genuinely beat competitive human-designed baselines on real, actively researched tasks—not synthetic or toy problems. The improvements are especially notable on Agent Failure Attribution (2.85× over baseline) and AI Text Detection (+7.9% AUROC with 2× speedup).
- **Novel system architecture for iterative scientific discovery**: The Findings Memory mechanism that accumulates and reuses both successes and failures across research cycles, combined with UCB-based selection, represents a meaningful advance over single-shot or pure brute-force AI Scientist systems. The design addresses a real problem (directed exploration vs. random search) and the reported zero-success rate of random sampling (Section 4.3) provides evidence for its necessity.
- **Unusually transparent reporting of the discovery process**: The paper openly discloses the 0.42% idea-to-success rate, provides pipeline statistics (5,000 ideas → 1,100 implemented → 21 progress findings → 5 papers), and releases full discovery logs and generated paper appendices. This level of transparency about failure modes is rare and valuable for the community.
- **Comprehensive human evaluation of generated outputs**: The convening of a small program committee (3 reviewers with ICLR experience) to evaluate the 5 generated papers, with reported inter-rater reliability (Krippendorff's α = 0.739), provides a more rigorous quality assessment than most AI Scientist systems offer.

## Weaknesses

### Major:
- **Misleading "Bayesian Optimization" framing**: The paper formalizes discovery as Bayesian Optimization (Section 3) and uses UCB as the acquisition function (Eq. 1), but the surrogate model is an LLM producing integer scores on a 0–100 scale. Standard BO requires a probabilistic surrogate (e.g., Gaussian Process) that provides calibrated mean and uncertainty estimates. An LLM-generated "exploration score" $v_e$ does not constitute epistemic uncertainty, and the theoretical guarantees of BO (convergence, regret bounds) do not apply. This should be described as a heuristic scoring and selection mechanism rather than Bayesian Optimization, or the authors must rigorously justify why LLM-produced scores function as valid uncertainty estimates. This matters because the BO framing is central to the paper's claimed contribution and misleads readers about the mathematical rigor of the search strategy.

- **Missing ablations for core system components**: Despite the system comprising multiple interdependent components (Findings Memory, UCB acquisition, surrogate model, three-stage loop), there are no ablations isolating their individual contributions. The paper mentions that random sampling of 100 ideas yields zero successes (Section 4.3), but this is not a controlled ablation—it uses far fewer samples than the ~1,100 actually tested. Critical unanswered questions include: (a) How does performance degrade without the Findings Memory (i.e., without reusing past findings)? (b) Does a simpler selection strategy (e.g., top-k by $v_u$ alone, or random selection with matched compute budget) achieve comparable results? (c) What is the contribution of the three-stage promotion system vs. a simpler filter? Without these ablations, it is impossible to attribute the system's success to the proposed BO-inspired architecture rather than to the raw scale of exploration (20,000 GPU hours, ~1,100 trials).

- **Withheld "Analyze & Report" module limits full reproducibility**: The Ethics Statement explicitly states the "Analyze & Report" module will not be open-sourced. While the motivation (preventing automated generation of unverified papers) is understandable, this means one of the system's three core stages cannot be independently verified or reproduced. The claim of "end-to-end autonomy" implicitly includes this stage, yet the community cannot evaluate it. This creates a tension between the paper's framing and its reproducibility.

- **Overstated claims about "autonomously redesigning core methodologies"**: The abstract states the system achieves progress "by autonomously redesigning core methodologies, not merely recombining existing techniques." However, the generated methods are relatively incremental adjustments: T-Detect replaces Gaussian normalization with Student's t-distribution; TDT applies wavelet transforms to token-level discrepancy sequences; A2P introduces a structured three-step causal reasoning prompt. These are valid scientific contributions, but they are more accurately described as methodological refinements and adaptations from adjacent fields rather than fundamental redesigns. The human reviewers gave the papers an average rating of 5.00 (borderline for ICLR), with notable variance (PA-TDT: 4.33 ± 1.33), which better reflects the output quality than the "frontier-pushing" framing.

- **Improvement metrics potentially misleading with multiple baselines/datasets**: The headline 183.7% improvement for Agent Failure Attribution corresponds to the Algorithm-Generated dataset, while the Handcraft dataset shows +142.8%. The abstract reports only the larger number without specifying which baseline/dataset it refers to, and the percentage is computed from a low absolute baseline (16.67%), which inflates relative improvements. More problematically, the arithmetic in Table 1 yields (47.46−16.67)/16.67 ≈ 184.7%, not 183.7%, suggesting either rounding in underlying raw numbers or a minor reporting inconsistency. The paper should consistently report improvements across all conditions and avoid selecting the most favorable single number for the abstract.

### Minor:
- **Scaling law claim supported by only 5 data points without error bars**: Figure 6 plots progress findings vs. GPU count (1, 2, 4, 8, 16) and claims a "near-linear relationship." Five points with no error bars, confidence intervals, or fit quality metrics provide weak evidence for a scaling law in a stochastic search process. More granular data points and statistical characterization are needed.

- **Evaluation limited to AI/ML tasks**: All three tasks are within machine learning, a domain where LLMs have strong native capabilities (code generation, literature comprehension). The paper's broader claims about "scientific discovery" would be substantially strengthened by testing on at least one task outside ML (e.g., computational biology, materials science).

- **Human supervision level insufficiently documented**: Section 4 states "Three human experts supervise the process to verify outputs and filter out hallucinations" and Section F mentions "approximately 50% of initial implementation attempts failed to complete fully due to internal timeouts." The exact nature and extent of human intervention—what was filtered, corrected, or guided—is not quantified. This is important for the "autonomous" claim.

- **60% implementation failure rate underanalyzed**: Section 4.3 and 4.4 note that 60% of failed trials were due to implementation errors, but no breakdown of error types (syntax, logic, environment issues) is provided. Understanding whether failures stem from LLM coding limitations vs. fundamentally flawed hypotheses would clarify the true bottleneck.

### Trivial:
- The "month-long" vs. "two-week" timeline is not inconsistent; the overall system ran for approximately a month across tasks, with text detection completing faster.

## Nice-to-Haves
- **Ablation study isolating Findings Memory and UCB selection**—ideally with a random-selection baseline at matched compute budget. This would be the single most impactful addition to strengthen the paper's methodological claims.
- **Taxonomy of the 60% implementation failures**—categorizing into syntax errors, logic bugs, environment issues, etc., would help the community build better code execution agents.
- **Extend evaluation to one non-ML scientific task** to test generalization claims.
- **Report absolute improvements alongside relative percentages** in the abstract and main text to avoid inflation from low baselines.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Weakness: 293% vs 183.7% calculation discrepancy** — The harsh critic calculated (47.46−12.07)/12.07 ≈ 293% and claimed the abstract's 183.7% was wrong. However, 183.7% corresponds to the Algorithm-Generated dataset column (baseline 16.67%), not the Handcraft column (12.07%). The abstract and Table 1 are internally consistent on this point. Removed because it is factually incorrect.
- **Weakness: "Month-long" vs "two-week" timeline inconsistency** — The system ran for approximately one month across all tasks; text detection completed in ~14 days. No real inconsistency exists. Removed as factually wrong.
- **Weakness: Missing related works** — Per the hard rules, we cannot verify external references and should not flag missing citations.
- **Weakness: Formatting and style issues** — Per the hard rules, formatting nitpicks are removed.
- **Weakness: Reproducibility concerns about proprietary LLMs (Claude-4-Opus, Gemini-2.5-Pro)** — The paper specifies the models used and they exist. Per hard rules, we do not question availability of cited models.
- **Weakness: Demanding confidence intervals for the main benchmark results** — Single-run evaluation is standard practice in the ML systems community for this type of work; this is a nice-to-have, not a core flaw.
- **Weakness: "Apples-to-oranges" comparison of GPU hours to human researcher years** — The paper explicitly frames this as trajectory comparison ("comparable progress"), not resource equivalence. This criticism misreads the paper's claim.
- **Weakness: Unfair comparison with baselines** — The baselines (All at Once, TokenRecycling, FastDetectGPT/Binoculars) are the published SOTA methods. The comparison structure favors the baselines by giving DeepScientist the same starting point, which is fair per the hard rules.

## Novel Insights
The most striking empirical finding is the extreme rarity of genuinely valuable scientific ideas even with sophisticated search: 5,000 ideas → 1,100 implemented → 21 progress findings → 5 papers. This 0.42% success rate, combined with the finding that 60% of failures are implementation errors rather than hypothesis failures, reframes the bottleneck for automated science. The community's focus should shift from "can AI generate novel ideas?" (clearly yes, at scale) to "how can we dramatically improve implementation reliability and pre-experiment filtering?" The paper's own ablation showing zero successes from random sampling of 100 ideas—but substantial successes with directed search—suggests the Findings Memory and selection mechanism are doing real work, but without controlled ablations, we cannot yet quantify exactly how much.

## Suggestions
1. **Add at minimum a Findings Memory ablation** (run without memory, i.e., each cycle generates hypotheses from scratch) and a **UCB ablation** (replace with top-k utility selection or random selection with matched budget). These two experiments would transform the paper from an impressive demo into a rigorous methodological contribution.
2. **Temper the "Bayesian Optimization" framing** to "BO-inspired heuristic search" or similar, and add a brief discussion acknowledging where the analogy breaks down (no calibrated posterior, no formal regret bounds).
3. **Report all improvement metrics consistently**—include both absolute and relative improvements, and report results on all datasets/baselines in the abstract rather than cherry-picking the largest number.
4. **Provide a breakdown of the 60% implementation errors** into categories (syntax, runtime, logic, environment) and discuss what this implies for future system design priorities.
5. **Consider releasing the "Analyze & Report" module under a restricted research license** (e.g., requiring human co-authorship verification) rather than withholding it entirely, to balance ethical concerns with reproducibility expectations.

---

## khBHJz2wcV

- GT: Accept (Poster) (avg 3.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Final Review

## Summary

This paper introduces a framework for post-training fine-tuning of flow-matching generative models to enforce parameter-dependent PDE constraints and jointly infer latent physical parameters, without requiring paired parameter–solution training data. The method leverages the Adjoint Matching framework (reformulating fine-tuning as stochastic optimal control), uses weak-form PDE residuals as rewards, and proposes a joint evolution of state and latent parameters via a surrogate base flow derived from a pre-trained inverse predictor. The approach is evaluated on four canonical PDE systems and a natural-image recoloring task.

## Strengths

- **Joint state–parameter evolution is a genuinely novel architectural contribution.** The key idea of evolving α alongside x through a surrogate base flow (derived from the inverse predictor φ) is elegant and addresses a real gap: existing physics-constrained generative methods either assume known parameters or require joint training data. The design—where v_{t,α}^{base} points from current α_t toward φ's one-step terminal estimate—enables physics-aware fine-tuning without paired labels, which is not achieved by prior work.

- **Weak-form residuals as rewards are a principled and well-justified choice.** The use of integration-by-parts to transfer derivatives from x to randomly sampled test functions ψ directly addresses the known instability of strong-form PDE residuals under noisy or misspecified data (Section 3.1). The Wendland-wavelet test function construction with bridge mollifiers (Appendix D.3) is carefully designed and more stable than naive strong-form approaches used in prior physics-informed diffusion work (e.g., Bastek et al., 2024).

- **Scaled memoryless noise schedule with theoretical justification.** Lemma 1 (Appendix D.4) proves that σ²(t) = (1−κ)2η_t retains the memoryless property for 0 ≤ κ < 1, providing a family of valid schedules rather than the single canonical choice. This is a clean theoretical extension of Domingo-Enrich et al. (2025) and offers practical control over the exploration–stability trade-off.

- **Extensive experimental coverage with honest ablations.** The paper evaluates across four physically distinct PDE families (elliptic diffusion, elasticity, wave propagation, incompressible flow) and explicitly shows the residual–distribution trade-off curves (e.g., Fig. 3, Fig. 5) rather than reporting only best-case numbers. The model misspecification studies (e.g., Stokes with F₀ = 2→0, Helmholtz damped→lossless) test genuine out-of-distribution adaptation.

## Weaknesses

### Major:

- **Reliance on inverse predictor φ creates an under-analyzed distributional shift vulnerability.** The surrogate base flow v_{t,α}^{base} and the regularization term f(α) both depend on φ(ẋ₁), where ẋ₁ is a one-step estimate from the current (fine-tuned) state. As fine-tuning progresses, the distribution of ẋ₁ departs from the base distribution on which φ was trained. The regularization f(α) partially mitigates this by anchoring to the base estimate, but the Darcy experiment (Section 4.1) explicitly shows the limitation: "Because α^{base} is itself fragmented, artifact-ridden, some artifacts persist" even with regularization. The paper does not analyze *when* this feedback loop becomes unstable—i.e., how far the fine-tuned distribution can drift before φ's estimates degrade catastrophically. This is a core reliability concern for scientific users deploying the method under severe misspecification.

- **The natural image experiment (Section 4.6) stretches the "physics-constrained" framing.** The abstract promises "cross-domain utility through fine-tuning of natural-image models" in the context of "scientific systems," but the image experiment optimizes a PickScore aesthetic reward via a polynomial color transform. This is a preference-alignment task, not a physics-constrained one. While the *optimization framework* (Adjoint Matching with a latent parameter) applies broadly, framing this as validating "physics-aware" generation is misleading. The experiment validates the *algorithm's generality as a reward-based fine-tuning method*, which is a different contribution than what the title and abstract emphasize. The paper should either reframe this section honestly (as a reward-alignment demonstration) or remove the implication that it supports physics-awareness.

- **The PBFM comparison is acknowledged as asymmetric but still presented prominently without sufficient caveats in the main text.** The paper correctly notes in Appendix E.2 that "such misspecification is inherently challenging for training-time methods like PBFM which naturally places them at a disadvantage." However, the main-text tables (Tables 1, 2) and the Stokes discussion (Section 4.5) feature PBFM's poor performance prominently without reiterating this caveat. PBFM failing to converge on Stokes (strong residuals 1.15×10¹) is presented as a point in favor of the proposed method, when it primarily reflects the fundamental incompatibility of a training-time method with a misspecification scenario. Readers may draw misleading conclusions about relative method quality rather than about the suitability of each approach class for this specific problem setting.

### Minor:

- **Multiple interacting hyperparameters (λx, λα, λf, κ, q) without principled selection guidance.** The ablations in Figure 3 and the appendix tables show that these parameters govern critical trade-offs (residual reduction vs. diversity, residual vs. distributional fidelity, stability vs. exploration). While the sweeps demonstrate the method's controllability, practitioners face a non-trivial tuning burden when adapting to new PDE systems. The paper provides no heuristic or automated strategy for initial hyperparameter selection beyond empirical trial.

- **Total computational cost is not fully contextualized.** The "20 gradient steps, under 15 minutes" claim for Darcy fine-tuning is impressive but omits the upstream costs: base FM pre-training (300 epochs, ~12 hours on RTXA6000; Table 4), inverse predictor pre-training, and the per-epoch cost of lean adjoint solves during fine-tuning. For a fair assessment of efficiency, a comparison of total wall-clock time (base training + φ training + fine-tuning) versus training a physics-constrained model from scratch (e.g., PBFM) to equivalent residual levels would be informative.

### Trivial:

- The notation density around η_t in Equation 1 and the definition of the reference flow coefficients could be clearer, but this does not impede understanding for the target audience.

## Nice-to-Haves

- **Comparison with established inverse solvers** (e.g., MCMC or PINN-based inversion) on the parameter recovery task, to contextualize whether the generative approach achieves competitive inference quality.
- **Validation on real observational data or 3D domains**, as all experiments use synthetic 2D 64×64 grids. This would strengthen claims about utility for "scientific systems."
- **Uncertainty calibration analysis** of the recovered parameter distributions against ground truth, given the stated goal of addressing "ill-posed inverse problems."
- **Visualization of failure modes** or limits of misspecification tolerance (e.g., at what point does the base model's support become too disjoint from the target physics for fine-tuning to succeed?).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Weak residual normalization blow-up near α→0**: The critic suggested normalization by local mean coefficient could blow up in voids. However, all datasets use strictly positive coefficient fields (permeability ∈ {3, 12}, Young's modulus ∈ [1.0, 10.0], etc.; Appendix B). This concern does not apply to the actual experiments.
- **Statistical significance of ± values**: The critic questioned whether reported standard deviations reflect training variance or sample variance. With 256 generated samples per configuration (Section 4), reporting sample variance is standard practice for generative model evaluation. This is a reproducibility nitpick.
- **κ as evidence of instability**: The critic framed κ as a "fix" for an unstable method. The paper presents κ as a generalization of the noise schedule family (Lemma 1 proves consistency), not as a patch. The original κ=0 schedule also works; κ>0 provides additional flexibility.
- **Missing comparison with guided diffusion approaches**: The paper compares against guided sampling (ECI) and training-time physics (PBFM). Demanding comparison with every recent guided-diffusion variant is a generic scope expansion.
- **Formatting/style complaints** about Equation 1 notation density and Figure 1 visual clarity.

## Novel Insights

The joint evolution design reveals an interesting symmetry: the latent parameter α has no ground-truth flow (unlike x, which has the base FM trajectory), so the method *manufactures* a surrogate base flow from the inverse predictor. This effectively turns a pure inference tool (φ) into a generative trajectory regularizer. The success of this approach—particularly the Stokes result where joint evolution achieves MMD_α ≈ 0.07–0.13 versus ablations at 0.22–0.28 despite similar residuals—suggests that the *path structure* of the parameter trajectory matters more for distributional fidelity than the endpoint quality alone. This observation, that regularizing the generative path (not just the terminal state) improves distributional metrics, could inform other settings where latent variables must be jointly sampled with observables.

## Suggestions

- Add a short subsection or paragraph explicitly analyzing the robustness of φ to distributional shift under fine-tuning. Even a simple experiment measuring φ's prediction error on fine-tuned vs. base samples would quantify the risk.
- Rebrand the image experiment (Section 4.6) as "Reward-Based Fine-Tuning Beyond Physics" or move it to an appendix, to prevent readers from interpreting it as evidence of physics-awareness. The current framing creates a mismatch between title/abstract promises and delivered content.
- Add a "Limitations" paragraph in the conclusion explicitly discussing the boundaries of applicability: under what degree of misspecification does the method break down? The Stokes forcing ablation (Figure 13) provides partial evidence, but a principled characterization (even qualitative) of when φ becomes unreliable would significantly help practitioners.
- Report total pipeline cost (base training + φ training + fine-tuning) alongside the fine-tuning-only cost, to give readers a complete efficiency picture.

---

## GiaF5cFIpI

- GT: Reject (avg 3.5)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary

This paper presents a streaming framework for adaptive stimulation-response modeling of latent neural dynamics. The core contributions are: (1) a novel streaming jPCA algorithm (sjPCA) for real-time rotational subspace identification, (2) a nonparametric kernel regression estimator $\hat{S}$ that models stimulus-response mappings as a function of latent state, stimulus, and time (enabling adaptation to non-stationarities), and (3) a constrained optimization problem for designing high-dimensional stimulation patterns that drive low-dimensional latent dynamics along desired directions under feasibility constraints (sparsity, non-negativity). The pipeline integrates multiple streaming dimensionality reduction methods and dynamical models in parallel, with adaptive selection of the best predictor at each timepoint. All components run in under 100ms end-to-end.

## Strengths

- **Unified streaming pipeline with real-time benchmarking**: The integration of streaming latent space construction, dynamical prediction, stimulus-response learning, and constrained optimization into a single real-time pipeline is a genuine engineering contribution. Concrete hardware specifications and per-component timing breakdowns (Appendix H) are provided—this level of benchmarking is rare in the closed-loop neuroscience literature and directly addresses the community's latency concerns.

- **Adaptive nonparametric stimulus-response modeling**: The kernel regression estimator $\hat{S}$ (Eq. 7) incorporates time as a regression feature, explicitly enabling the model to adapt to non-stationarities such as plasticity, probe drift, or photobleaching. The demonstration of recovery from abrupt mapping changes (Fig. 2e, "Flip" and "Rotate" conditions) shows this is not just a design feature but a functional capability.

- **Parallel latent space evaluation with adaptive selection**: Running proSVD, sjPCA, and mmICA in parallel and selecting the best predictor at each timepoint (Fig. 1c) mitigates the risk of committing to a single manifold hypothesis. The reported improvement in average log predictive probability (−1.72 with best single space to −1.01 with adaptive selection) provides quantitative evidence this mechanism adds value.

- **Validation on real stimulation data in Appendix C**: While the main experiments use simulated stimulations, Appendix C validates the stimulus-response regression on two datasets with actual photostimulation events (Daie et al., 2021; Draelos et al., 2025), showing lower prediction error than the blind model. This partially addresses concerns about biological validity, though it remains offline analysis.

## Weaknesses

### Major:

- **Main experiments use simulated stimulations on pre-recorded data**: The primary validation on calcium imaging and electrophysiology datasets (Section 4.1) injects synthetic stimulation effects via an autoregressive function ($y_t = r_t + a_t$, $a_t = 0.8 \cdot a_{t-1} + u_t$). The "closed-loop" is simulated: the biological system does not actually respond to the computed $u^*$. While the Abstract technically says "demonstrate our approach on both simulated and real neural data," the distinction between real recordings with simulated perturbations and true biological closed-loop is obscured. The core scientific claim—that the method can adaptively drive latent dynamics—remains validated only in silico. Appendix C's offline analysis of real stimulation data provides evidence that $\hat{S}$ can learn real stimulus-response relationships, but does not test the full optimization pipeline in a closed-loop biological setting. This gap between the framing ("adaptive stimulation of latent neural activity") and the evidence is the paper's most significant limitation.

- **Insufficient baselines for the stimulus-response mapping**: The primary comparison throughout is against a "blind model" that withholds stimulation information from the dynamical predictor. This is a minimal baseline that any stimulation-aware model should beat. The paper does not compare $\hat{S}$ against parametric alternatives (e.g., a linear model $S(u) = Wu$, or a simple affine mapping), which would establish whether the nonparametric kernel regression is necessary or whether a simpler model suffices. Given the computational overhead of kernel regression, this comparison is needed to justify the design choice. Notably, Appendix D shows that the closed-loop (kernel regression) estimator actually performs *worse* than the open-loop (linear assumption) estimator on the trivial stimulus-response mapping (Fig. D.1), suggesting the nonparametric approach may overfit when the true mapping is simple—a tradeoff that is not discussed.

- **No ablation studies**: The method composes multiple non-trivial components (three latent spaces, three dynamical models, kernel regression with state/stimulus/time features, constrained optimization with L1 regularization). No ablations are provided to determine which components are essential. For example: Does the time kernel $K_3$ actually improve performance over a stationary kernel? Does state-dependent kernel $K_1$ help beyond what stimulus-only regression provides? Does adaptive space selection improve stimulation outcomes, or only prediction accuracy? Without ablations, it is unclear whether the system's complexity is justified or whether simpler alternatives would perform comparably.

### Minor:

- **Optimization formulation clarity (Eq. 8)**: The term $\lambda_1(\|u\|_{0,\max} - \|u\|_1)$ is described as encouraging "a solution with the number of non-zero elements close to $n$." Under minimization with box constraints $[0,1]^N$, this term effectively *maximizes* $\|u\|_1$, which under these constraints pushes entries toward 1 (dense, high-power solutions) rather than promoting sparsity in the traditional L1-regularization sense. The intent appears to be: set $\|u\|_{0,\max}$ as a target, and maximize $\|u\|_1$ so that under near-binary solutions, the number of active neurons approximates this target. However, this design choice and its interaction with the alignment objective deserve clearer justification, as it inverts the standard LASSO-type relaxation of L0 constraints.

- **Kernel regression long-term scalability**: Eq. 7 sums over all $N$ past stimulation events. The paper notes that $N$ grows slowly (at the rate of stimulation events) and that the time kernel $K_3$ discounts old samples, but no explicit pruning or fixed-buffer mechanism is described. In a prolonged experiment, unbounded growth of the kernel dictionary could eventually violate the <100ms timing guarantee, particularly on the hardware specified. A discussion of practical mitigation strategies (e.g., discarding samples with negligible kernel weight, or a fixed memory budget) would strengthen the real-time feasibility claims.

- **sjPCA time derivative estimation**: The streaming jPCA formulation (Eq. 1) requires $\dot{X}_t$, but the paper does not detail how time derivatives are estimated causally in the streaming regime. If finite differences are used, the resulting delay in the feedback loop should be discussed. This is an implementation detail that could affect convergence or introduce bias.

### Trivial:

- The runtime claims are hardware-dependent, but the paper provides sufficient benchmarking (Appendix H) for readers to extrapolate.

## Nice-to-Haves

- **Comparison with existing stimulation optimization methods**: Experimental comparison against Bayesian optimization (Minai et al., 2024) or active learning (Wagenmaker et al., 2024) approaches for stimulation design would establish the relative merits of this constrained optimization approach.

- **Behavioral outcome measurements**: The paper motivates causal testing of how latent variables encode behaviors, but no behavioral metrics are reported. The authors acknowledge this scope limitation (Section 5); demonstrating that optimized stimulation changes behavior would significantly strengthen the neuroscience contribution.

- **Theoretical convergence analysis for sjPCA**: Empirical convergence to offline fits is shown (Fig. 1a), but stability bounds for the Sherman-Morrison-based update of the skew-symmetric matrix would increase confidence in the streaming estimator.

- **Uncertainty quantification on $\hat{S}$ predictions**: For safety-critical in vivo applications, confidence intervals or predictive uncertainty on the stimulus-response mapping would be valuable. The kernel regression framework naturally admits variance estimates.

- **Optimization landscape analysis**: No analysis of whether the constrained optimization (Eq. 8) has local minima that could trap solutions, which affects reliability for experimental deployment.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Missing related works / references not cited** — The spark finder suggested the paper fails to compare with Minai et al. (2024) and Wagenmaker et al. (2024). However, both are cited and discussed in the Introduction as prior work. The concern is actually about experimental comparison (baselines), not missing citations. Per rules, I do not flag missing related works.

- **Weakness: Reproducibility concerns about undisclosed hyperparameters** — The harsh critic flagged insufficient detail on kernel bandwidth tuning, RBF scaling initialization, etc. Per rules, nitpicks about reproducibility such as undisclosed hyperparameters are removed.

- **Weakness: Generalization across brain regions/behavioral states** — The spark finder questioned whether the method generalizes across brain regions or stimulus types. This is a generic concern that could be raised about any method; the paper tests on two modalities and multiple dynamical regimes, which is adequate for an initial demonstration.

- **Weakness: Scalability to thousands of neurons (whole-brain imaging)** — The neutral reviewer flagged this. The current experiments use 130–592 neurons, which matches the scale of holographic optogenetics applications the paper targets. Demanding whole-brain scale is scope creep.

- **Weakness: Cross-dataset generalization (train on one, test on another)** — This is not a standard evaluation for adaptive/stimulation methods, which are inherently experiment-specific. Removed as an unreasonable demand.

## Novel Insights

The paper reveals an interesting asymmetry in the feasibility landscape of latent perturbations: some directions in the latent space are naturally easy to drive via excitation-only constraints (e.g., along the first principal component), while others are structurally infeasible (e.g., population-wide inhibition). This feasibility structure is a property of the neural population and its embedding, not the optimization method, and could itself serve as a tool for characterizing the geometry of neural manifolds. The observation that closed-loop (nonparametric) stimulus-response estimation can outperform open-loop (linear assumption) design on nontrivial mappings but underperforms on simple ones (Appendix D) highlights a fundamental bias-variance tradeoff in adaptive stimulation that the community should attend to: adaptive methods are most valuable precisely when the mapping is unknown, but they require sufficient exploration to avoid overfitting.

## Suggestions

- **Reframe the contribution explicitly as a simulation-validated framework**: Adjust the Abstract and Introduction to clearly state that the method is validated via simulated stimulations on real neural recordings and offline analysis of real stimulation datasets, with in vivo closed-loop deployment as future work. This honest framing would strengthen rather than weaken the paper.

- **Add a linear stimulus-response baseline**: Compare $\hat{S}$ against $S(u) = Q^\top u$ (the open-loop/linear assumption) on the main real-data experiments. The toy model comparison in Appendix D already hints at the tradeoff; extending this to the calcium and electrophysiology datasets would clarify when the nonparametric approach is warranted.

- **Perform targeted ablations**: At minimum, ablate (a) the time kernel $K_3$ to test the adaptive/non-stationary benefit, and (b) the state-dependent kernel $K_1$ to test whether location-dependent responses matter. These would directly validate the two most distinctive features of the stimulus-response model.

- **Clarify the optimization formulation**: Either provide a more detailed justification for why maximizing $\|u\|_1$ approximates an L0 constraint under the stated conditions, or revise the formulation to use a more standard sparsity penalty and explain the design tradeoffs.

- **Add a pruning mechanism for kernel regression**: Describe even a simple strategy (e.g., discarding samples where all kernel weights fall below a threshold) to guarantee bounded memory and computation for long experiments.

---

## USyGD0eUod

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Final Review

## Summary

This paper applies sanity-check methodology from saliency map validation (Adebayo et al., 2020) to sparse autoencoder (SAE) evaluation in mechanistic interpretability. The authors train SAEs on both trained and randomly initialized transformers (Pythia, 70M–6.9B) and find that aggregate auto-interpretability scores (fuzzing/detection AUROC) and reconstruction metrics are surprisingly similar across both settings for larger models. They propose token distribution entropy as a diagnostic for feature "abstractness" and recommend routine randomized baselines.

## Strengths

- **Rigorous null model design with multiple randomization variants.** The paper goes beyond a single random baseline by including Step-0 initialization, re-randomization with/without embeddings, and a Gaussian control. The control variant consistently achieves ~0.50 AUROC (Figures 1, 6–14), validating that the auto-interpretability pipeline is not trivially gullible—only when structured activations exist (even from random weights) does it produce high scores. This multi-variant design substantially strengthens the causal attribution of the failure to the metrics rather than the pipeline.

- **Compelling qualitative evidence that closes the intuition gap.** Appendix J/L provides randomly sampled feature dashboards with generated explanations and activating examples. The Control features (Appendix L.2) produce vague, nonsensical explanations ("various tokens including articles, conjunctions..."), while randomized variants (L.3/L.4) yield superficially coherent single-token explanations ("the token 'record' often refers to..."). This makes the abstract quantitative finding concrete and inspectable—a practice most papers in this area neglect.

- **The core empirical finding addresses a genuine validation gap.** The observation that auto-interpretability AUROC for Pythia-6.9b trained (0.79) overlaps with randomized (0.87–0.88) in Figure 1 is striking. The field has been relying on these metrics to justify SAE quality claims; demonstrating their insensitivity to whether the underlying model has learned anything is an important negative result.

## Weaknesses

### Major:

- **Section 4's toy models explain SAE trainability, not auto-interpretability scores—a mechanistic gap.** The toy models (Section 4) demonstrate that random networks preserve or amplify superposition, which explains why SAEs can achieve low reconstruction error on random networks. However, the paper's headline claim is about auto-interpretability scores—an LLM-based measure of feature explainability. The leap from "random networks produce sparse, reconstructible activations" to "LLMs can generate plausible explanations for those activations" is not established. The more direct explanation for high auto-interpretability scores is the "single-token feature" hypothesis (Section 3, "Latent explanation complexity"), which is supported by the token entropy analysis but is not connected to the superposition framework. These two narratives (superposition preservation → SAE trainability vs. simple token-specific features → LLM explainability) need synthesis, or the paper should acknowledge that Section 4 addresses a different question than the one the title raises.

- **The scaling transition from distinguishable (70M) to indistinguishable (6.9B) is the most consequential finding but receives no dedicated analysis.** Figure 6 (Pythia-70M) shows trained AUC ≈ 0.63 vs. randomized ≈ 0.50—a visible gap. Figure 1 (Pythia-6.9B) shows near-complete overlap. This transition is arguably the paper's most important empirical result: it tells us *where* the metrics break down. Yet there is no single summary visualization (e.g., AUROC gap vs. parameter count or model size) and no substantive hypothesis for *why* larger random models better mimic the metrics. The authors speculate that "features become more specific as SAE size increases" (Section 3), but this is counter-intuitive (larger models should learn more abstract features) and under-developed. A dedicated scaling analysis would significantly strengthen the paper.

- **The claim about "computationally relevant features" is asserted but not functionally validated.** The paper's central conclusion is that "high aggregate auto-interpretability scores do not, by themselves, guarantee that learned, computationally relevant features have been recovered." The term "computationally relevant" implies features that causally influence model behavior. While the qualitative evidence (Appendix J) strongly suggests trained features are more abstract, no causal intervention (e.g., steering, activation patching) tests whether trained features actually affect downstream behavior while random features do not. The authors note that CE loss score (Figure 2, row 5) is only meaningful for trained models, which is indirect evidence, but a direct ablation—showing that intervening on a high-AUROC random feature has no behavioral effect—would transform this from a suggestive finding to a definitive one. The paper explicitly scopes itself to evaluation metrics rather than functional validation, so this is not a fatal omission, but it leaves the strongest version of the claim unproven.

### Minor:

- **Statistical power of 100-feature sampling for large SAEs.** For Pythia-6.9b with expansion factor R=64, the SAE latent space is large (potentially hundreds of thousands of latents). Sampling 100 features represents <0.04% of the dictionary. Appendix E shows variance across 5 training seeds for Pythia-70M, but does not isolate variance due to feature sampling specifically. If the distribution of interpretable features is heavy-tailed, the aggregate AUROC could be noisy. This concern is partially mitigated by the consistency of results across layers and model sizes, but a brief analysis of sampling variance (or a note on why 100 is sufficient) would strengthen the quantitative claims.

- **Token distribution entropy conflates feature simplicity with feature type.** The entropy metric (Section 3) successfully distinguishes trained from random features in aggregate, but it can penalize legitimately specific learned features. For example, a trained feature that selectively fires on a single technical term (e.g., a specific gene name) would have low entropy despite being a genuinely learned feature. The paper acknowledges this ("the token distribution entropy is not a direct measure of 'abstractness'"), but does not quantify the false positive rate. A brief discussion of when this metric would mislead would be valuable for practitioners considering its adoption.

- **Title overstates the result for small models where discrimination succeeds.** The title claims automated interpretability metrics "DO NOT DISTINGUISH" trained and random transformers. However, for Pythia-70M (Figure 6), the gap is meaningful (trained 0.63 vs. randomized ~0.50). The failure is scale-dependent. A more precise title reflecting this (e.g., "...Are Insufficient for Large Transformers") would better represent the contribution.

### Trivial:

- Figure 2 is a multi-panel figure where rows are referenced by number in the text but not labeled in the caption, making navigation slightly inconvenient.

## Nice-to-Haves

- **Causal intervention ablation:** Even a small-scale experiment (e.g., on Pythia-70M) showing that steering with trained SAE features modifies behavior while steering with random SAE features does not would provide the strongest possible validation of the "computationally relevant" claim.
- **Single summary plot of AUROC gap vs. model size:** A plot showing how the trained-vs-randomized AUROC gap shrinks with scale would make the scaling finding immediately visible and citable.
- **Cross-architecture validation on one non-Pythia model** (e.g., Gemma-2 or Llama-3) to assess generalizability beyond the Pythia family.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Explainer LLM sycophancy/hallucination concern** (Harsh Critic, Spark Finder): The Control variant consistently achieves AUROC ≈ 0.50 (chance), demonstrating that the 70B explainer does *not* find plausible explanations in pure noise. This directly addresses the concern—the explainer is not hallucinating coherence where none exists; it is correctly identifying structure in activations that happen to arise from random weights processing structured inputs.

- **Missing SAE architecture ablation** (transferred from WCRQFlji2q.md): The paper tests expansion factors 16–128 and sparsity k=16,32 (Appendix F), and uses TopK SAEs, a current standard. Demanding tests across all SAE architectures (Gated, JumpReLU, etc.) is scope creep for a paper focused on evaluation metrics rather than SAE design.

- **Computational cost of randomized baselines as a limitation** (Harsh Critic): This is a practical concern about a recommendation, not a flaw in the experimental methodology. The Step-0 variant is available for Pythia at no additional training cost, and the re-randomization procedure is a one-time operation.

- **Embedding-only SAE baseline** (Spark Finder): The "Re-randomized excl. embeddings" variant already isolates the role of embeddings by preserving them while randomizing all other weights. Training SAEs on raw embeddings is a different experiment that would not address the paper's question about whether transformer processing (even random) produces interpretable features.

- **Missing related work citations** (transferred weakness): Cannot verify existence of specific references; removed per hard rules.

- **Formatting/style nitpicks** (OCR artifacts in references, figure caption numbering): Removed per hard rules.

## Novel Insights

The scaling dimension of this result is underappreciated even by the reviewers: the transition from distinguishable (70M) to indistinguishable (6.9B) suggests that larger random matrices are increasingly effective at preserving input data structure through their transformations. This is consistent with random matrix theory (random projections approximately preserve distances in high dimensions via the Johnson-Lindenstrauss lemma), but the implication for interpretability is novel—*the very mathematical property that makes large random networks useful for dimensionality preservation also makes them dangerous as interpretability nulls*, because they produce activations that are superficially structured enough to be "explained" by an LLM. This reframes the problem: the issue is not that auto-interpretability metrics are broken per se (they correctly reject pure noise), but that they lack a notion of *computational depth*—they cannot distinguish "the network preserved this input structure" from "the network learned to compute this structure."

## Suggestions

- Add a single plot showing the AUROC gap (trained − randomized) as a function of model size, potentially faceted by layer, to make the scaling finding immediately visible.
- In Section 4, explicitly acknowledge the gap between the superposition/sparsity explanation (why SAEs train well on random nets) and the auto-interpretability result (why LLMs can explain the resulting features), and clarify that the token-entropy analysis addresses the latter while the toy models address the former.
- Consider running steering/patching on a small model (e.g., Pythia-70M) with trained vs. random SAE features to provide direct functional validation of the "computationally relevant" claim, even if only as a preliminary experiment.

---

**Axis evaluations:**

- **Novelty:** High. The systematic application of randomized baselines to auto-interpretability evaluation is novel and addresses a real validation gap. The scaling finding is particularly new.

- **Technical soundness:** Moderate-to-good. The experimental design is strong (multiple null variants, control, robustness checks). The main gap is the disconnect between the toy model mechanism (sparsity) and the headline result (interpretability), and the lack of functional validation for the "computationally relevant" claim.

- **Empirical support:** Good for the core negative result (metrics fail to distinguish at scale). Weaker for the proposed solution (token entropy) and for the mechanistic explanation. The scaling trend is well-documented but under-analyzed.

- **Significance:** High. If the field has been relying on auto-interpretability scores as evidence of meaningful feature discovery, this paper's demonstration that these scores are largely insensitive to training is a foundational challenge. The practical implications for how SAE research is evaluated are substantial.

- **Clarity:** Good. The paper is well-organized and the appendices provide valuable qualitative evidence. The multi-panel Figure 2 could benefit from row labels in the caption.

---

## Iq1fNZus2W

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

The paper proposes Patch-wise and Keyword-Aware Attention (PKA), a framework for efficient multi-condition control in Diffusion Transformers. PKA decomposes the standard "concatenate-and-attend" mechanism—which scales quadratically with the number of conditions—into two specialized modules: Position-Aligned Attention (PAA) restricts spatial condition attention to one-to-one position-aligned patches, and Keyword-Scoped Attention (KSA) confines subject-driven attention to keyword-activated image regions. An early-timestep sampling strategy further accelerates training convergence. The method reports up to 10× inference speedup and 5.12× attention-module VRAM reduction while maintaining generative quality.

## Strengths

- **Analysis-driven motivation with clear empirical grounding.** The paper doesn't just propose an efficiency hack; it first establishes that multi-condition attention is genuinely redundant through attention map visualizations (Figures 2–3). The observation that spatial attention concentrates along the diagonal and subject attention localizes to keyword-relevant regions provides a principled basis for the architectural decomposition—a level of diagnostic rigor that many efficiency papers lack.

- **Condition Cache mechanism with clean design logic.** By restricting condition tokens to self-attention only within their group (Figure 4a–b), the K/V projections become invariant to the denoising state and can be computed once and reused across all subsequent steps. This is a non-trivial insight that turns a structural constraint (no cross-condition attention) into a concrete efficiency win. The design is coherent: the same principle that enables decomposition also enables caching.

- **Substantial and well-characterized efficiency gains.** Figures 7–8 demonstrate convincing speedup and VRAM reduction that scales with the number of conditions. The ablation studies (Figures 9–10) include latency and VRAM numbers at the full pipeline level, not just the attention module, lending credibility to the practical impact. The tunable ε threshold in KSA provides a graceful efficiency–fidelity trade-off, which is a practical advantage over binary design choices.

- **Early-timestep sampling is well-motivated.** The perturbation analysis (Figure 5, Appendix A.2) showing that visual conditions dominate early in the denoising trajectory is a meaningful empirical finding, and the convergence results (Appendix A.3, Figure 13) confirm the training benefit. This goes beyond just an architectural contribution and addresses a training efficiency angle.

## Weaknesses

### Major:

- **Keyword selection mechanism for KSA is underspecified, creating a critical dependency.** The paper states (Section 3.2.2) that "the keyword set K typically contains just 1 to 2 tokens" but never explains how these keywords are identified. Is this manual annotation, automatic NLP extraction, or part of the prompt engineering pipeline? The training data is curated to "ensuring each image caption contains a descriptive keyword" (Section 4.1), suggesting human annotation—but this is never made explicit. Since KSA's entire masking logic (Eq. 3) hinges on these keywords, the method's practical applicability depends entirely on this unexplained step. If keywords must be manually specified per prompt, the framework cannot be used autonomously; if automatic, the extraction accuracy is not analyzed.

- **The "10× inference speedup" claim requires clarification of measurement scope.** The abstract claims "up to a 10× inference speedup," but the VRAM claim is carefully qualified as "attention module." The speedup claim is not equivalently scoped. The ablation in Figure 9 shows a much more modest gain for a single spatial condition (13.63s vs. 15.38s, roughly 1.13×). The 10× figure appears only when many conditions are stacked (Figure 7) and is compared specifically against UniCombine's full-attention implementation. The paper should explicitly state whether the speedup is end-to-end or attention-module-only, and whether it includes the overhead of KSA mask computation (Eq. 3) and cache management logic. As presented, the abstract's claim risks overstating what the ablation numbers support.

### Minor:

- **PAA's strict one-to-one spatial alignment is a strong assumption whose robustness is untested.** PAA assumes that image token at position *i* should attend exclusively to the spatial condition token at position *i*. If the condition map is even slightly misaligned with the latent grid (e.g., due to preprocessing differences, imprecise edge detection, or resolution mismatch), there is no mechanism to correct for this. Full attention would implicitly handle small misalignments; PAA cannot. The paper does not discuss or test this. An experiment with jittered or misaligned condition inputs would clarify how brittle this assumption is.

- **KSA mask reuse across timesteps lacks analysis of mask drift.** The mask *M* computed at timestep *t* is reused at *t*+1 (Section 3.2.2) based on "temporal consistency." While the final results suggest this works, the paper provides no analysis of how the mask evolves over the denoising trajectory or when/why it might fail. In early high-noise steps, the latent representation may be too noisy for reliable keyword-based localization, potentially producing an inaccurate initial mask that persists.

- **Limited diversity of condition types in evaluation.** All experiments use Canny, Depth, and Sketch as spatial conditions, and Subject as the sole subject-driven condition. Other common condition types (e.g., pose/keypoints, segmentation masks, style references) are not tested. While the categorization into "spatial-aligned" and "subject-driven" is conceptually general, the empirical validation only covers a narrow slice of the claimed condition space.

- **Early-timestep sampling creates a training–inference distribution gap that is not discussed.** Training uses a shifted logit-normal (μ > 0) that prioritizes early timesteps, but inference presumably uses a standard scheduler covering the full trajectory. The paper does not address whether this mismatch affects late-stage detail synthesis or diversity. Figure 11 shows qualitative results at different μ values but does not evaluate diversity (e.g., LPIPS variance across seeds).

### Trivial:

- PAA's one-to-one attention, when softmax is applied over a single key-value pair, mathematically reduces to simply outputting V^SP_i (since softmax of a single element is always 1). This makes PAA effectively a per-position value injection rather than "attention" in the traditional sense, which the paper could describe more transparently.

## Nice-to-Haves

- Test PAA robustness with intentionally misaligned or noisy condition inputs to characterize failure modes.
- Evaluate on at least one additional DiT backbone (e.g., SD3) to validate the generality of the attention decomposition.
- Include a diversity metric (e.g., LPIPS variance across seeds) for the early-timestep sampling ablation to ensure the shifted training distribution doesn't collapse output diversity.
- Release reference implementations of the custom attention kernels, as standard libraries don't natively support position-aligned or keyword-scoped patterns.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Criticism that FLUX.1 or other cited models/tools are unreleased or unverifiable.** The paper cites FLUX.1 (Labs, 2024) with a GitHub URL. Per rules, cited models are assumed to exist and be available.

- **Criticism demanding comparison with feature-injection DiT baselines (ControlNet-style).** The paper explicitly scopes itself to the attention-based interaction paradigm in DiTs (Section 1, Section 2.1). Comparing against a fundamentally different conditioning paradigm (feature injection) is outside the stated scope. The paper's contribution is about making the attention paradigm efficient; whether attention or feature injection is better is a separate question.

- **Criticism about missing related work references.** Cannot verify existence of uncited works; this risks fabricating references.

- **Criticism about reproduction details (undisclosed hyperparameters, implementation details).** The paper provides training details (LoRA, 20K iterations, Prodigy optimizer, batch size 1, grad accumulation 4). More granular details (e.g., exact LoRA rank, learning rate) would be nice but are not a core flaw by community standards for empirical generation papers.

- **Criticism about potential train/test overlap in Subject200K.** The paper explicitly states the subset "is then partitioned into training and testing sets" (Section 4.1), indicating a proper split was created. Speculating about overlap without evidence is unwarranted.

- **Criticism about the Condition Cache being invalid across guidance scales.** In PKA's design, condition tokens only self-attend within their group (Figure 4b), so their K/V representations are independent of both the image state and the guidance scale. The cache is mathematically sound given this architectural choice.

- **Formatting/style nitpicks** about equation rendering and notation clarity. The paper's math is comprehensible despite some PDF parsing artifacts.

## Novel Insights

The paper's most insightful contribution is the observation that the redundancy in multi-condition attention is *qualitatively different* depending on condition type—spatial conditions exhibit diagonal-localized redundancy while subject-driven conditions exhibit keyword-scoped semantic redundancy. This dual characterization suggests that "one-size-fits-all" sparse attention methods (e.g., uniform token pruning) are suboptimal for multi-condition settings; the right efficiency strategy must be *condition-type-aware*. This principle could generalize beyond DiTs: any architecture handling multi-modal conditions might benefit from decomposing attention along condition-type boundaries rather than applying uniform compression.

## Suggestions

- Add one paragraph in Section 3.2.2 explicitly describing the keyword extraction pipeline (manual vs. automatic, with examples), and evaluate KSA sensitivity to keyword choice in the ablations.
- Report end-to-end wall-clock latency (including all overhead) alongside the attention-module-specific numbers in the main efficiency claims, and qualify the abstract's "10× inference speedup" to match what the full-pipeline measurements actually support.
- Add a small robustness experiment: apply small spatial shifts (±1–2 patches) to condition inputs and measure whether PAA degrades gracefully or catastrophically compared to full attention.

---
**Evaluation Summary (verbal, no scores):**

- **Novelty:** Good. The condition-type-aware decomposition of attention is a clean and well-motivated idea. While sparse attention and KV caching individually are not novel, their integration with condition-specific structural priors (diagonal for spatial, keyword-scoped for subject) is a distinct contribution.

- **Technical soundness:** Acceptable with gaps. The core mechanisms are sound, but the keyword dependency in KSA and the measurement ambiguity around efficiency claims are real technical concerns that should be addressed. The PAA simplification to value injection (trivial softmax) should be acknowledged.

- **Empirical support:** Adequate but narrow. The efficiency results are convincing for the tested setting, but the evaluation covers limited condition types and doesn't probe failure modes. The ablations are helpful but miss key robustness checks.

- **Significance:** High if claims hold. Making multi-condition DiTs practically scalable is an important problem, and the reported efficiency gains are substantial. The significance is somewhat tempered by the narrow empirical validation and underspecified components.

- **Clarity:** Good overall. The paper is well-structured with clear motivation, method description, and experimental organization. The attention pattern analysis is a particularly effective communication choice.

---

## ZNAY3ivd62

- GT: Reject (avg 4.0)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary
GUI-Spotlight introduces a 7B-scale multimodal model for GUI visual grounding that iteratively narrows its focus using specialized visual tools (crop, extract, find_color) coordinated via a modified Group Sequence Policy Optimization (GSPO) reinforcement learning algorithm. With only 18.5K training samples, it achieves 52.8% accuracy on ScreenSpot-Pro, surpassing 7B baselines trained on orders of magnitude more data, and approaches the performance of much larger 72B models.

## Strengths
- **Exceptional data efficiency:** The model achieves 52.8% on ScreenSpot-Pro using 18.5K curated samples (2,561 SFT trajectories + 12K RL + 4K high-resolution RL), compared to V2P-7B's 50.6% with 9.6M samples and GTA-1-7B's 50.1% with 1.56M samples. The efficiency gap is two orders of magnitude, which is practically meaningful for the field.
- **Stabilized RL training for multi-turn tool use:** The auxiliary cross-entropy loss term $J'(\theta)$ on format-valid and result-correct samples addresses a real and documented failure mode—vanilla GRPO/GSPO collapses around 300 steps due to tool-call syntax degradation (Figure 3, right panel). This is a genuine methodological contribution backed by empirical evidence, not just a minor tweak.
- **Transparent documentation of negative results:** Section 4 systematically reports what does not work (e.g., continuously updating the reference policy degrades accuracy; dense answer rewards underperform sparse ones; high-uncertainty prompt filtering hurts). This is unusually thorough for the area and provides genuine practical value for future work on agentic RL.

## Weaknesses

### Major:
- **Inference cost is completely unreported, yet central to practical viability:** The model performs multi-turn tool invocations (potentially 3–5+ rounds per query), each requiring full model forward passes and image processing. For GUI agents, latency is a first-class concern—users expect sub-second click responses. The paper never reports average number of tool calls per query, wall-clock time, token cost, or any latency comparison against single-pass baselines. Without this, the claim of a "practical" improvement (Abstract) is unsupported. The accuracy–latency trade-off may well favor single-shot models for real deployment.
- **Domain gap between web-centric training data and native-application benchmarks:** The training data is collected primarily via a Selenium-based headless browser crawling high-traffic websites (Appendix A.4: google.com, youtube.com, amazon.com, etc.). Yet the primary benchmark, ScreenSpot-Pro, evaluates on Creative tools, CAD software, Scientific applications, and Office platforms—domains with fundamentally different visual layouts, icon systems, and interaction patterns than web pages. The UGround dataset also originates from web and mobile sources. The paper claims the model "improves across all six domains" but provides no analysis of how web-trained tool policies transfer to native desktop UIs, nor any ablation separating web vs. desktop training data performance. This gap undermines the generalization claims.

### Minor:
- **No recovery mechanism when the spotlight loses the target:** If an early `extract` or `crop` operation removes the target element from the visible region, all subsequent iterations operate on a wrong sub-image with no possibility of correction. The pipeline (Algorithm 1) has no "zoom-out" or "reset" action. The paper does not quantify how often this cascading failure occurs or discuss it as a limitation.
- **Absence of failure mode analysis:** The paper claims "robustness" to dense, cluttered UIs but never characterizes when or why the method fails. What types of instructions (e.g., ordinal references like "the third icon from the left" requiring global context) or UI configurations cause the iterative approach to break? Without this, the robustness claim is unsupported.
- **Small accuracy margins without statistical significance testing:** The headline improvement of 52.8% vs. 50.6% (V2P-7B) is a 2.2-point gap on a benchmark. No confidence intervals, multiple seeds, or significance tests are reported. While single-run evaluation is common in the field, a gap this narrow could reflect evaluation variance rather than a real improvement.
- **Per-tool ablation missing:** Section 4.2 ablates reward weight ratios (Crop vs. Extract) but never ablates individual tools (e.g., removing `find_color` entirely). It is unclear whether all three tools are necessary or whether the model relies primarily on one. Tool usage frequency and coordination patterns are not analyzed.
- **Mixed evaluation protocols:** Some baseline numbers come from the ScreenSpot-Pro leaderboard, while others (UI-TARS-1.5-7B) are self-evaluated by the authors. Differences in prompts, post-processing, or evaluation code could confound comparisons. This is common practice but should be acknowledged.

### Trivial:
- The reward formula for $S_{BA}$ in Section 3.2.1 is garbled in the PDF, making it hard to verify the exact overlap metric used for bounding-box accuracy filtering.

## Nice-to-Haves
- Report average tool calls per query and accuracy as a function of iteration count to reveal whether gains come from iterative refinement or primarily from the first 1–2 steps.
- Compare against test-time compute baselines (e.g., multiple single-shot predictions with voting) to isolate the value of tool coordination from the value of additional compute.
- Include an SFT-only (no RL stages) ablation to quantify how much RL contributes beyond supervised tool-use warm-up.
- Analyze tool usage patterns (frequency, sequences) to validate that the model learns meaningful coordination rather than relying on one dominant tool.
- Evaluate on end-to-end GUI tasks (beyond single-step grounding) to test whether improved grounding translates to improved agent task completion.

## Removed Points
These points are flagged to be removed; treat them with caution.
- **"Table 3 does not contain a row for GUI-Spotlight"** — This is a PDF parser artifact; the paper's text explicitly references and discusses GUI-Spotlight's per-domain results in Section 5.1.
- **"18.5K training sample count is inconsistent or unclear"** — The stages sum to approximately 2,561 + 12,000 + 4,000 = 18,561, consistent with the stated 18.5K. The breakdown is available in Section 3.2.2.
- **"Qwen2.5-VL-72B dependency for data filtering limits reproducibility"** — This is a reproducibility nitpick about a standard practice (using large models for data curation); removed per hard rules on reproducibility nitpicks.
- **"Ethics statement should address malicious automated UI interaction"** — Scope creep; the paper is about visual grounding, not autonomous agent deployment. Removed per soft rules.
- **"Baseline fairness concern about V2P-7B's base model"** — The comparison already favors V2P-7B (9.6M training data vs. 18.5K); removed per hard rules on unfair comparison complaints where asymmetry favors the baseline.
- **"Demand for deeper analysis of why continuously updating the reference policy fails"** — The paper already documents this as a negative result. Demanding theoretical explanation for a discarded variant is scope creep.

## Novel Insights
The most revealing finding across the reviews is that the paper's core architectural idea—iterative tool-based refinement—and its RL training contribution are conflated in the experimental design. Section 5.4 compares against training-free iterative inference and shows gains, but there is no SFT-only (same tools, no RL) baseline. This means the paper cannot definitively attribute its gains to RL learning of tool coordination versus the inductive bias of the tool interface itself. The training-free comparison in Section 5.4 shows the untrained model has "virtually no multi-step reasoning capacity," suggesting RL is essential—but an SFT-only intermediate would cleanly separate these factors and is a surprising omission for an otherwise thorough empirical section.

## Suggestions
- Add a table reporting inference cost (average tool calls, wall-clock time, and total tokens per query) for GUI-Spotlight versus single-pass 7B baselines. This is the single most important missing piece for evaluating practical impact.
- Add a per-category breakdown of GUI-Spotlight results on ScreenSpot-Pro in the main text (not just the aggregate), and explicitly discuss the web-to-desktop transfer: which categories benefit most from iterative refinement, and does the model struggle on categories least represented in training data?
- Include at least 3–5 representative failure cases showing where the iterative spotlight diverges, with analysis of the root cause (e.g., wrong initial extract, loss of global context, color tool failure on low-contrast themes).

---

## 0cbUKCyBsH

- GT: Reject (avg 3.5)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

This paper introduces Influence-Aware Time Series Forecasting (IATSF), a paradigm that reframes time series forecasting from predicting future values based solely on history ("self-stimulation") to modeling the external influences that drive system dynamics. The authors provide a control-theoretic analysis proving that ignoring influences imposes an irreducible error floor (Proposition 2.1), introduce a leak-free benchmark with temporally-synced textual influences, and propose FIATS—a lightweight model with channel-aware mechanisms (CASM, CAPS) for integrating text signals. Experiments across synthetic, physics, electricity, and market datasets show substantial gains over self-stimulated baselines.

## Strengths

- **Novel theoretical framing with formal proofs:** The control-theoretic analysis formalizing the "self-stimulation error floor" (Proposition 2.1) and the influence-efficacy result (Proposition 3.1) provide a principled foundation for why external influences matter. The proofs covering both linear and nonlinear systems (Appendix B) are thorough and give the paper conceptual depth rare in empirical TSF work.

- **Strong empirical gains across diverse settings:** FIATS achieves consistent improvements over strong baselines including foundation models (Chronos-L, MOIRAI-L, Time-MoE-U) and multimodal methods (TimeLLM). The 36–44% MSE reductions on Atmospheric Physics and NYC Traffic are substantial, and the near-zero error on the FM Toy dataset validates the theoretical predictions.

- **Thoughtful benchmark design:** The IATSF benchmark explicitly addresses information leakage, temporal synchronization, and independence of influences (Section 4.1)—addressing documented flaws in prior datasets like Time-MMD (Appendix N). The inclusion of channel descriptors and multiple dataset categories (synthetic, physics, market) is well-designed for evaluating influence-aware methods.

- **Architectural interpretability:** The CASM attention maps (Fig. 5, 10) demonstrate that FIATS learns meaningful channel-specific sensitivity patterns (e.g., atmospheric pressure channel attending to pressure-related text), providing evidence that the model captures genuine influence dynamics rather than spurious correlations.

## Weaknesses

### Major:

- **The independence assumption ($U_t \perp X_h$) limits theoretical generality and practical applicability of the "hard barrier" claim.** Proposition 2.1 derives the error bound assuming influences are independent of historical observations. In many real-world systems—influences are endogenous or correlated with past states (e.g., weather exhibits strong autocorrelation; economic indicators are path-dependent). When $U_t$ correlates with $X_h$, a sufficiently expressive self-stimulated model can partially infer influences from history, potentially reducing the error floor substantially. The paper frames the barrier as "hard" and "mathematical," but this holds only under an assumption that many real systems violate. The theoretical and practical implications of this assumption need more honest discussion, particularly since the paper's own Atmospheric Physics dataset involves weather—highly autocorrelated and predictable from history.

- **The Atmospheric Physics dataset may violate the paper's own independence requirement, raising leakage concerns.** Section 4.1 requires influences to be "independently evolving—external factors that influence the system but are not themselves outcomes of it." Yet the flagship dataset uses weather reports (describing "clear skies," "high pressure," "humidity level") to predict atmospheric physics variables that ARE weather. These text descriptions are essentially compressed observations of the target system state, not truly external influences. This creates a circularity: the "influence" text describes the same physical system being predicted, which both violates the independence assumption and inflates performance relative to genuinely exogenous settings (e.g., predicting traffic from weather, where the influence is truly external). The paper should explicitly acknowledge this tension and validate on a domain where the text is genuinely exogenous to the target.

- **Main experiments use ground-truth future influences (oracle setting), creating a significant gap to real-world deployment.** The results in Table 1 assume perfect knowledge of $U_f$ at test time. The paper's own analysis (Appendix B.3) demonstrates that influence forecaster error ($B\Sigma_{\hat{U}}B^\top$) can dominate model error, yet no experiments evaluate performance under realistic forecast noise for $U_f$. Figure 6 shows noise robustness but only for semantic perturbations to text embeddings, not for temporal or predictive errors in the influence itself. This oracle evaluation makes the "breaking the barrier" claim difficult to assess in practical terms.

- **Insufficient ablation to separate modality benefit from architectural contribution.** All baselines in Table 1 are self-stimulated (no text input). Table 3 ablates CASM and CAPS within FIATS, but no experiment gives a standard model (e.g., PatchTST) access to the same text embeddings via a simple fusion mechanism (e.g., concatenation or standard cross-attention). Without this, it remains unclear whether FIATS's gains stem from the principled CASM/CAPS design or simply from having text as an input modality. The ablation in Table 6 (training with zero/random text reducing to PatchTST-level) is suggestive but doesn't address this directly—a simple text-concatenation baseline would.

### Minor:

- **Nonlinear extension relies on first-order approximation.** The error bound for general nonlinear systems (Proposition B.1) uses a first-order Taylor expansion, discarding higher-order terms. For strongly nonlinear or chaotic systems, these terms can be significant. The paper acknowledges this as a limitation (Section 7) but states the barrier as universally applicable in the introduction and abstract, which overclaims.

- **Practical brittleness to misleading text.** Appendix I (Table 6) shows that training with good text but testing with incorrect text yields poor performance (MSE 0.724 vs. 0.186). This suggests the model trusts textual influence signals heavily, which could be problematic in deployment scenarios where text sources are unreliable or adversarial.

- **Weather as a validation domain has inherent circularity.** The NYC Traffic Speed dataset, where weather is genuinely exogenous to traffic, would be a stronger test of the paradigm. The paper's results on this dataset are strong (44.3% MSE reduction), but the primary validation (Atmospheric Physics) uses a domain where the "influence" and "target" are essentially the same system.

### Trivial:

- The proof notation switches between linear and nonlinear cases in ways that require careful reading; a unified presentation might improve accessibility.

## Nice-to-Haves

- Evaluate FIATS under realistic influence forecasting noise (e.g., use actual weather forecast errors rather than ground-truth reports) to characterize the practical performance envelope.
- Add a simple text-concatenation baseline (PatchTST + text embeddings via cross-attention or concatenation) to isolate the architectural contribution of CASM/CAPS.
- Test on a domain where the textual influence is truly exogenous to all target variables (e.g., macroeconomic news predicting individual stock prices, or policy text predicting health outcomes) to validate the paradigm where the independence assumption genuinely holds.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **GAUD reproducibility (only releasing embeddings):** The critic raised that releasing only pre-computed embeddings prevents verifying the "leak-free" nature of influences. However, the paper cites intellectual property constraints (Appendix O.5), which is a practical limitation, not a methodological flaw. The benchmark design principles are clearly documented.

- **Statistical significance concerns:** The critic suggested some improvements may not be statistically significant. The paper provides standard deviations (Table 10), a critical difference diagram (Fig. 12), and runs experiments 3–5 times. This exceeds standard reporting for the field.

- **Missing exogenous-variable baselines (TimeXer, ChronosX):** The spark finder suggested including these. The paper already benchmarks against TimeLLM (a text-informed method) and Chronos-L. While adding more baselines would strengthen comparisons, the existing set covers the key categories (linear, transformer, foundation, multimodal).

- **GAUD "cold-start" terminology:** The critic noted that games with developer logs aren't truly "cold-start." This is a minor terminology quibble; the paper's actual claim (Section 6.3) is about "cold-start problems for new games, where historical data is sparse but influence information is available," which is accurate.

- **Formatting artifacts in Table 1:** These are parser issues, not paper issues, and are explicitly excluded by the rules.

## Novel Insights

The paper's most insightful contribution is the formalization of the "self-stimulation barrier" as an information-theoretic impossibility result: no matter how expressive the model, if the external influence is independent of history, the model can only learn the conditional expectation $E_U[F(X_h, U)]$, not the actual realization. This reframes the TSF plateau as a *task formulation* problem rather than a *model capacity* problem—a genuinely different perspective from the prevailing scaling-oriented narrative. The channel-aware sensitivity analysis (showing that error reduction from influence is proportional to $\nabla_U F \cdot \Sigma \cdot \nabla_U F^\top$, meaning high-sensitivity channels benefit most) provides a principled explanation for why weather text helps pressure channels more than temperature channels—something the attention visualizations empirically confirm. However, the gap between the theoretical framework's assumptions (independence, perfect forecaster) and the empirical setup (weather-as-own-influence, oracle text) remains the paper's central tension.

## Suggestions

- Add a "PatchTST + simple text fusion" baseline to Table 1 to isolate the architectural contribution of CASM/CAPS from the modality benefit of having text input at all.
- Evaluate on at least one dataset where the text influence is genuinely exogenous and temporally forecasted (not oracle ground-truth), reporting performance degradation curves as a function of influence prediction error.
- Discuss the independence assumption's practical scope more explicitly: clarify which real-world systems approximately satisfy $U_t \perp X_h$ and which violate it, and provide empirical analysis of how much historical predictability of $U_t$ reduces the self-stimulation barrier in practice.

## Quality Assessment

- **Novelty:** High — the control-theoretic reframing of TSF limitations and the formal error bounds are genuinely novel contributions that shift the discourse from model capacity to task formulation.
- **Technical Soundness:** Moderate — the theory is correct under its stated assumptions but those assumptions (independence, perfect forecaster, first-order approximation) limit practical applicability more than the paper acknowledges.
- **Empirical Support:** Moderate — results are strong in the oracle setting, but the conflation of genuinely exogenous influences (traffic+weather) with self-descriptive influences (atmospheric physics+weather reports) and the absence of realistic influence forecasting experiments weaken the practical claims.
- **Significance:** High if the theoretical framework holds up — the "barrier" framing could redirect the field's efforts. But the practical significance depends on domains where genuinely independent influences are available as text.
- **Clarity:** Good — the paper is well-structured with clear separation between theory, benchmark, model, and experiments. The appendices are extensive but the notation is consistent (Table 4).

---

## bH5M0ts8Y6

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

VINCIE proposes learning in-context image editing directly from native video data by constructing interleaved multimodal sequences (frames, textual transition descriptions, and segmentation masks) and training a DiT model with three proxy tasks: next-image prediction, current segmentation prediction, and next segmentation prediction. The approach demonstrates scalability and achieves strong results on multi-turn editing benchmarks, while also showing emergent capabilities in story generation and multi-concept composition.

## Strengths

- **Novel data paradigm with strong scalability evidence.** The core insight—that video temporal dynamics naturally provide the multi-turn editing signal that curated image pairs cannot—addresses a genuine data bottleneck. Figure 5 provides compelling scalability evidence: later-turn success rates (Turn-4, Turn-5) show nearly log-linear improvement with data scale from 0.25M to 10M sessions, while Table 5 shows that video-sequence pretraining outperforms pairwise training by +16.4% at Turn-1 and +21.0% at Turn-5 on MSE-Bench. This is a concrete, quantitative demonstration that video structure provides something pairwise data cannot.
- **Well-designed proxy tasks with meaningful ablation.** The CSP and NSP tasks are not just auxiliary losses—they serve as controllable intermediate representations. Table 3 shows that the chain-of-editing strategy (CS→NS→I) improves Turn-5 success rate from 10.3% to 17.3%, and the paper demonstrates that segmentation prediction mitigates the subject position drift inherent in video training (Figure 7, Section 4.4). The practical benefit is clear: the model learns to ground edits before generating them.
- **In-context editing mitigates artifact accumulation.** Figure 6 and Table 4 provide evidence that incorporating full context from previous turns significantly reduces artifacts compared to sequential single-turn editing. Table 4 shows L1 distance nearly halving (0.155→0.086) when dummy context is added at Turn-1, confirming that the model actively leverages context rather than treating each turn independently.

## Weaknesses

### Major:

- **Main comparison table (Table 2) omits the proposed method's results.** MSE-Bench is the paper's own proposed benchmark, yet Table 2 lists 17 baselines without including "Ours." The text states "our method achieves a 25% success rate at turn-5," but the reader cannot verify this against the tabulated baselines. The method's MSE-Bench numbers appear only indirectly in the ablation table (Table 5, "sequence" row, Turn-5 = 22.0%). This presentation gap makes it impossible to directly compare VINCIE against the baselines it claims to outperform on the paper's own benchmark. Additionally, the "25%" claim in the text (Section 4.3) does not match the 22.0% shown in Table 5; this inconsistency further erodes confidence in the reported results.

- **"Trained exclusively on videos" claim is misleading for SOTA results.** The Abstract states the model is "trained exclusively on videos" and achieves "state-of-the-art results," but Table 1 shows the best-performing variants are "Ours* + SFT," which undergo supervised fine-tuning on curated image-pair datasets (OmniEdit, SEED-Data-Edit, etc.; see Appendix C.7). While the video-only model (Ours* without SFT) performs competitively, the SOTA claims rely on additional image-pair SFT. The paper should clearly distinguish VINCIE-Base (video-only) from VINCIE-SFT throughout, and the Abstract should not conflate the two when claiming SOTA. The Limitations section (Appendix F) also does not acknowledge this gap.

- **MSE-Bench contains only 100 test instances.** For a benchmark that the paper proposes to "advance research in this area," 100 instances is statistically fragile. Small benchmark sizes are vulnerable to outlier-driven swings in success rate metrics. The paper should acknowledge this limitation explicitly and, at minimum, report confidence intervals or variance estimates to support the comparative claims.

- **Key ablation (Table 3) uses an intermediate checkpoint, weakening the evidence for CSP/NSP.** The caption states: "This ablation study was conducted using an intermediate checkpoint, so the reported numbers may not be directly comparable to those in other tables." Since the CSP/NSP contribution is a central design claim, the ablation should be reproduced on the final checkpoint. Without this, the reader cannot confirm that the reported gains (e.g., Turn-5: 10.3%→17.3%) hold for the model whose results are advertised elsewhere.

### Minor:

- **VLM annotation accuracy is 75.14% (Appendix D.3), but the impact of this ~25% label noise is not analyzed.** The paper argues that scale compensates for noise (citing InstructPix2Pix's similar approach), but provides no ablation or analysis confirming this. A simple experiment filtering low-confidence annotations or injecting synthetic noise would clarify whether the noise imposes a performance ceiling.

- **GPT-4o evaluation has moderate correlation with human judgment (Pearson r = 0.4858, r² ≈ 0.24).** While GPT-4o is a reasonable proxy, it explains less than 25% of the variance in human scores. The human evaluation results (Table 6, Appendix D.1) should be elevated to the main text to corroborate the GPT-4o numbers, especially given the small benchmark size.

### Trivial:

- The scalability figure (Figure 5) has a large gap between 0.25M and 1.25M data points; finer granularity in the low-data regime would better illustrate sample efficiency.

## Nice-to-Haves

- Quantitative evaluation of "emergent" capabilities (story generation, multi-concept composition) rather than relying solely on qualitative examples (Figures 1, 19–21).
- Comparison against training on web-scraped interleaved image-text datasets (e.g., Obelics) to isolate whether the benefit comes from video temporal dynamics specifically or from the interleaved sequence format more generally.
- Error analysis breaking down performance by edit category (object addition/removal, attribute change, camera change) on MSE-Bench, to validate the "universal" editing claim and reveal where video-native priors help or hurt.
- Analysis of performance beyond 5 turns to test the long-form content claim, as context window limitations may degrade quality.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Not yet released" / reproducibility concerns about cited models (e.g., Nano Banana, GPT Image 1).** The paper cites them; they are assumed to exist per review rules. Removed.
- **Demand for comparison with text-conditioned baselines outside the in-context editing scope.** The paper explicitly scopes its contribution to in-context image editing; requesting comparison with a fundamentally different paradigm is scope creep.
- **Criticism that pixel-wise L1/L2 metrics in Table 4 are "outdated."** Table 4 also reports DINO and CLIP-I scores alongside L1/L2; the pixel metrics supplement rather than replace perceptual metrics. The concern is partially addressed.
- **Reproducibility concerns about undisclosed hyperparameters and training details.** Appendix C provides extensive implementation details including data construction, architecture, SFT datasets, and training configurations. Removed as nitpick.
- **Concern that the data construction pipeline is not reproducible because VLM/grounding model thresholds are not fully specified.** The appendix provides the prompting instructions and pipeline description. Standard practice in this area uses similar automated pipelines; demanding every threshold is a nitpick.
- **Criticism about the "teacher-student" dynamic limiting "universal" editing because GroundingDINO/SAM2 may fail on certain categories.** This is speculative without evidence that it actually limits the model's performance. The model shows generalization to uncommon edit types (Figure 14), suggesting this bound is not tight in practice.

## Novel Insights

The paper reveals an interesting asymmetry in how video data benefits multi-turn editing: earlier turns (Turn-1) saturate quickly with data scale, while later turns (Turn-4, Turn-5) exhibit nearly log-linear improvement. This suggests that video data's primary value is not in teaching *what* to edit (which can be learned from single pairs), but in teaching *how to maintain coherence across accumulated edits*—precisely the signal that pairwise data cannot provide. This finding has implications beyond image editing: any generative task requiring long-form consistency (story generation, multi-step reasoning) may benefit more from sequential training data than from scaling single-step pairs.

## Suggestions

- **Add VINCIE results directly to Table 2** (both base and SFT variants) with the exact numbers matching the text claims, and reconcile the "25%" vs. "22%" discrepancy explicitly.
- **Reproduce the CSP/NSP ablation (Table 3) on the final checkpoint** so that the proxy task contribution is verifiable against the main reported results.
- **Report confidence intervals or variance on MSE-Bench results** given the 100-instance size, and move the human evaluation (Table 6) to the main text to support the GPT-4o numbers.
- **Adopt consistent nomenclature** (e.g., VINCIE-Base vs. VINCIE-SFT) throughout the paper, and adjust the Abstract to clearly attribute SOTA results to the SFT variant rather than implying they come from video-only training.

## Quality Assessment

- **Novelty:** High. The video-to-editing paradigm shift is genuinely new; no prior work demonstrates learning in-context image editing solely from native video data with this level of success.
- **Technical soundness:** Moderate. The method is well-designed, but key ablations use intermediate checkpoints, and the "video-only" vs. SFT distinction is blurred in presentation.
- **Empirical support:** Mixed. Strong on MagicBrush and scalability; weaker on MSE-Bench due to small benchmark size, missing method entry in main table, and moderate eval metric correlation.
- **Significance:** High if the presentation issues are resolved. The scalability of video-based training and the artifact mitigation finding are impactful for the field.
- **Clarity:** Moderate. The core method is clearly described, but the conflation of video-only and SFT results, and the missing Table 2 entry, create confusion that undermines an otherwise well-written paper.

---

## kMfVTka2WB

- GT: Reject (avg 2.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary

This paper proposes a Covariance-Adjusted Support Vector Machine (CSVM) that accounts for class-specific covariance structure in classification. The authors argue that standard SVM's max-margin principle and KKT conditions are valid only under Euclidean geometry, and that the input space (equipped with a Mahalanobis metric) is "Non-Euclidean," requiring transformation via class-specific Cholesky decomposition before classification. An iterative "SM Algorithm" is proposed to estimate population covariance from training data by pseudo-labeling test points and updating covariance estimates until convergence. Empirical results on five binary classification datasets show improvements over standard SVM kernels and global whitening methods (PCA/ZCA).

## Strengths

- **Principled motivation for class-conditional whitening**: The paper provides a coherent (if imprecisely stated) geometric argument for why class-specific covariance adjustment matters for SVM margins, moving beyond ad hoc preprocessing by deriving how the margin in input space becomes a function of intra-class covariance (Eq. 9, 14). This yields the concrete observation that the decision boundary should split the margin space in proportion to class covariances—a testable claim with clear operational implications.

- **SM Algorithm as an iterative estimation procedure**: The proposed algorithm (Section 3) attempts to close the gap between sample and population covariance by iteratively refining pseudo-labels on test data and re-estimating covariance. While it has significant limitations (detailed below), it poses an interesting semi-supervised question: can the structure of test data itself improve covariance estimation? This is a genuinely different angle from static preprocessing.

- **Consistent empirical improvements**: Across five diverse datasets (healthcare, astronomy, quality control, safety), CSVM achieves the highest accuracy and F1 on four out of five datasets, and the highest AUC on three. The consistency of improvement—rather than gains on a single dataset—suggests the method is capturing something real, even if the magnitude and statistical significance remain uncertain.

## Weaknesses

### Major:

- **The SM Algorithm uses test data to update model parameters, making the method transductive and the comparison with inductive baselines unfair.** Step 3(g)–(h) of the SM Algorithm explicitly adds pseudo-labeled test points to the training set and recalculates covariance matrices from the updated data. This means the model parameters (covariance estimates and consequently the decision boundary) depend on the test batch. Standard SVM kernels and PCA/ZCA whitening baselines are purely inductive—they never observe test data. The performance gains may therefore stem from access to the test distribution's marginal structure rather than from the proposed covariance-adjusted geometry. Without an inductive variant of the method (e.g., class-conditional whitening using only training data) as a baseline, the source of improvement cannot be isolated. This is the most critical experimental flaw.

- **The theoretical framework contains imprecise and potentially misleading claims about "Non-Euclidean" spaces and "invalid" KKT conditions.** The input space for standard tabular data is ℝⁿ, which is Euclidean by definition. Using a Mahalanobis metric changes the *metric tensor* but not the underlying vector space; the resulting space is isometric to Euclidean space via a linear transformation—which is precisely what Cholesky decomposition provides. Calling this "Non-Euclidean" conflates a change of metric with a change of topology or vector space structure. Similarly, Lemma 2.3 claims "KKT boundary conditions are not valid" in the input space, but KKT conditions apply to *any* differentiable convex optimization problem. If the objective changes to include Σ, the KKT conditions for that *new* problem involve Σ; this does not render the original KKT conditions "invalid"—it simply means a different optimization problem is being solved. These imprecise claims weaken the theoretical contribution and may mislead readers about the nature of the result.

- **Inconsistency between Lemma 2.2 (two classifiers) and the algorithm's single-classifier inference.** Lemma 2.2 states that a binary problem yields "two unique linear classifiers" in the input space, arising from the two different optimization problems (Eqs. 10–11 vs. 12–13) with different covariance matrices Σ_{y=1} and Σ_{y=-1}. Yet the SM Algorithm (Step 3d–e) produces a single classifier θ_Input^T x + θ₀' = 0 by adjusting only the bias of a standard input-space SVM. The paper does not resolve how the two distinct optimization problems from Lemma 2.2 collapse to a single decision function, nor how the two different margin ratios (one per class perspective) are reconciled into one θ₀' adjustment.

### Minor:

- **No convergence guarantees or stability analysis for the SM Algorithm.** The algorithm iteratively pseudo-labels test data and updates covariance estimates. No proof or empirical analysis is provided showing that this process converges to a stable fixed point, nor is the risk of error propagation analyzed (where early misclassifications corrupt the covariance estimate and subsequent iterations). The convergence criterion ("changes in test data labels are below a certain threshold") is vague, and no sensitivity analysis to this threshold is provided.

- **The theoretical derivation assumes hard-margin SVM (ξᵢ = 0), but the evaluated datasets are not linearly separable.** The paper explicitly sets ξᵢ = 0 in Section 2, deriving the margin and optimization problem for the separable case. However, datasets like Diabetes and Red Wine are unlikely to be linearly separable even after whitening. The paper does not derive or discuss the soft-margin extension, creating a gap between theory and practice.

- **No statistical significance testing for the reported performance improvements.** The tables report point estimates from a single 80/20 split. Some improvements are marginal (e.g., OSHA Accuracy: 0.752 vs. 0.741; Red Wine F1: 0.743 vs. 0.737). Without standard deviations, confidence intervals, or significance tests across multiple runs, it is unclear whether these gains are real or attributable to split variance.

- **Missing comparison with closely related covariance-adjusted baselines.** The paper cites MCVSVM (Zafeiriou et al., 2007), maxi-min margin machine (Huang et al., 2004), and weighted Mahalanobis kernels (Wang et al., 2007), all of which address similar problems of incorporating covariance into SVM. No experimental comparison with these methods is provided. Similarly, QDA—which also performs class-conditional covariance-based classification—is not included. Without these comparisons, it is unclear whether CSVM offers advantages over existing covariance-adjusted approaches.

### Trivial:

- **No runtime comparison provided.** The paper acknowledges higher computational complexity due to Cholesky decomposition and iterative SVM solving, but provides no empirical wall-clock comparison. This is a minor omission since the complexity difference is straightforward to reason about, but practitioners would benefit from knowing the scale of overhead.

## Nice-to-Haves

- **Class-conditional whitening + SVM (inductive, no test data in loop) baseline.** This would cleanly isolate the contribution of the iterative SM refinement from the contribution of class-conditional whitening itself, and would provide a fair inductive comparison point.

- **Ablation study on SM Algorithm components.** How much of the performance gain comes from class-conditional vs. global whitening? How much from the iterative refinement vs. a single-pass approach? How sensitive is the method to the convergence threshold?

- **Visualization on 2D synthetic data** with known covariance disparities, showing the decision boundary splitting the margin in the ratio of class covariance as claimed in Eq. 14. This would be a compelling proof-of-concept that directly validates the core theoretical claim.

- **Comparison with deep metric learning or modern Mahalanobis distance learning methods** to situate the work relative to current standards.

- **Soft-margin derivation** extending the theoretical framework to the non-separable case with slack variables.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Reliance on Sahoo & Maiti (2025) requires more rigorous geometric justification"** (Harsh Critic): Per the hard rules, cited references are assumed to exist and be valid. The criticism of insufficient geometric justification is already captured in the "Non-Euclidean" terminology weakness above; the specific targeting of this citation is removed.

- **"ROC curves without confidence bands"** (Harsh Critic): This is a generic demand for confidence bands on ROC curves, which is not standard practice in the ML community for small-scale evaluations. Moved to trivial/removed.

- **"Comparing with deep metric learning baselines"** (Spark Finder): This is scope creep. The paper operates within classical SVM theory; demanding modern deep baselines goes beyond the paper's stated scope. Moved to nice-to-have.

- **"The paper is well-structured"** (Balanced Reviewer): Generic strength that applies to many papers. Weakened per soft rules.

- **"Comprehensive experimental datasets"** (Balanced Reviewer): Five datasets without significance testing or ablations is adequate but not exceptional. Weakened per soft rules.

## Novel Insights

The most interesting observation emerging from the synthesis of these reviews is that the paper's core contribution may be better understood as an *implicit derivation that class-conditional whitening is the correct metric transformation for SVM*, rather than a discovery about "Non-Euclidean" geometry. The margin-ratio result (Eq. 14)—showing that the decision boundary should split the margin in proportion to θᵀΣ⁻¹θ for each class—is a concrete, testable consequence of this transformation, and it provides a normative answer to *how much* the boundary should shift when classes have unequal dispersion. This is a genuine insight that could be cleanly separated from the problematic "Non-Euclidean" framing and the transductive evaluation issue. If the authors reformulated the contribution as "class-conditional whitening is the metric-correct transformation for SVM, and it implies a specific margin-ratio adjustment," the paper would have a cleaner theoretical story—though the experimental methodology would still need to be corrected.

## Suggestions

1. **Add an inductive baseline**: Implement class-conditional Cholesky whitening using only training data (no iterative test-data inclusion), then apply standard SVM. Compare this against the full SM Algorithm. This single experiment would clarify whether the gains come from the geometry or from transductive access to test data.

2. **Soften and clarify the theoretical claims**: Replace "Non-Euclidean space" with "space equipped with a Mahalanobis metric" or "non-isotropic feature space." Replace "KKT conditions are invalid" with "the standard SVM margin objective is metric-dependent, and under a Mahalanobis metric the margin splits non-equally." These revised claims are true and non-trivial without being overstatements.

3. **Resolve the Lemma 2.2 inconsistency**: Either show that the two optimization problems (Eqs. 10–13) yield the same decision boundary in the transformed space (which would undermine the "two classifiers" claim), or provide a clear inference rule for combining the two classifiers in input space.

4. **Provide convergence analysis or empirical evidence**: At minimum, plot label stability across SM Algorithm iterations for each dataset, and report how many iterations are typically needed. Discuss failure modes when the initial pseudo-labels are substantially wrong.

5. **Extend to soft-margin**: Derive the slack-variable formulation explicitly, since all real-world applications require it and the hard-margin assumption is a gap between theory and practice.

---

## JEN4nsDgh9

- GT: Reject (avg 3.5)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary

This paper proposes a benchmark for evaluating text-to-image (T2I) models on their ability to generate images for WordNet taxonomy concepts. It introduces 9 evaluation metrics—including novel taxonomy-specific measures (Lemma/Hypernym/Cohyponym Similarity, Specificity) theoretically grounded in KL divergence and mutual information—and evaluates 12 open-source T2I models using human preferences, GPT-4 pairwise judgments, and a reward model. The benchmark covers easy concepts, randomly sampled WordNet synsets, and LLM-predicted concepts, finding that Playground-v2 and FLUX consistently outperform other models and that generation substantially surpasses retrieval-based approaches.

## Strengths

- **Novel and well-motivated task formulation.** The paper identifies a genuine gap: ImageNet covers only 6.5% of WordNet synsets, and no systematic benchmark exists for evaluating whether T2I models can visualize taxonomy concepts at varying levels of abstraction. The three dataset splits (Easy, Random WordNet, LLM Predictions) are thoughtfully designed to probe different difficulty levels and sensitivity to AI-generated content.

- **Taxonomy-structure-aware metrics.** The Hypernym, Cohyponym, and Specificity metrics leverage the hierarchical relationships in WordNet rather than treating each concept in isolation. The correlation of Hypernym CLIP-Score (ρ ≈ 0.911) and Cohyponym CLIP-Score (ρ ≈ 0.871) with human rankings provides empirical evidence that these metrics capture meaningful semantic relationships, not just generic text-image alignment.

- **Multi-perspective evaluation with alignment validation.** Combining Human ELO, GPT-4 ELO, and a Reward Model—and reporting their inter-correlations (e.g., Spearman ρ ≈ 0.92 between human and GPT-4 rankings with definitions)—provides a more robust evaluation than any single metric alone. The paper also transparently reports where these signals diverge.

- **Practical resource contribution.** Releasing a dataset of generated images covering all of WordNet-3.0 with the best-performing model directly extends ImageNet's coverage and enables downstream taxonomy enrichment work.

## Weaknesses

### Major:

- **The theoretical grounding of similarity metrics is undermined by the probability approximation.** Section 4.2 and Appendix D define Lemma Similarity as S_lemma(v,x) := P(X=x|v) ≈ sim(C(v), C(x)), and all subsequent theorems (Theorems 1–4 on KL divergence and mutual information) rest on this being a well-defined conditional probability. However, CLIP cosine similarity is a bounded score in [-1, 1], not a normalized probability distribution over images. Without a partition function or normalization over the image space, the derivations in Appendix D do not hold as stated. The metrics may still function as useful heuristic scores, but the paper's claims of formal grounding in information theory are not supported unless this gap is addressed. This matters because the paper prominently advertises these metrics as "grounded with theoretical justification drawing on KL Divergence and Mutual Information" (Abstract).

- **GPT-4 pairwise evaluation exhibits strong position bias, weakening the reliability of ELO rankings.** Section 5 acknowledges "no correlation between raw scores for individual battles" due to "a strong bias toward the first option" (Figure 5, Confusion Matrix in Figure 12). The paper did not employ standard mitigations such as swapping model positions in paired prompts and averaging, which is the established practice in LLM-as-a-judge evaluation (Zheng et al., 2023a). The Bradley-Terry model assumes comparisons are consistent and unbiased; systematic position bias violates this assumption unless explicitly modeled. While the overall ranking correlation with humans (ρ ≈ 0.88 with definitions) provides some reassurance, the per-item unreliability means the GPT-4 ELO scores cannot be trusted at the individual-comparison level, which limits their utility for fine-grained analysis.

### Minor:

- **Dataset sampling description in Section 2.2 is internally contradictory.** The text states that test set probabilities are "1×10⁻⁵ for Hypernymy, 0.05 for Hyponymy, and 0.1 for Synset Mixing," yet the resulting test set contains 828 Hypernymy nodes (69%), 170 Synset Mixing, and 204 Hyponymy. It is mathematically unclear how a category assigned a sampling probability near zero becomes the dominant class. This appears to conflate training and test probabilities or contains a reporting error, and undermines confidence in the dataset splits. The authors should clarify whether these probabilities refer to the TaxoLLaMA training data or to the test sampling, and provide the correct sampling procedure.

- **No analysis of metric redundancy or complementarity.** Nine metrics are proposed, but the paper provides no inter-metric correlation analysis or ablation showing which metrics capture distinct information versus which are redundant. Table 2 shows that different metrics favor different models (e.g., SDXL-turbo wins on CLIP-based similarities, Playground wins on preferences), but without guidance on which metrics matter most for the claimed use case of taxonomy enrichment, users of the benchmark cannot determine which signal to prioritize.

- **Lack of quantitative failure analysis by concept type.** Appendix I provides qualitative examples of failure modes (abstract concepts, rare words, functional roles) but no quantitative breakdown of how performance varies across concept types or taxonomy depth. Given that the paper's core motivation is that taxonomy concepts pose distinctive challenges, a systematic analysis of performance vs. abstraction level or position in the hierarchy would substantially strengthen the empirical contribution.

- **Specificity metric produces counterintuitive rankings without adequate explanation.** Table 13 shows SD1.5 (an older, weaker model) achieving the highest Specificity (1.23), tied with SDXL-turbo, while FLUX scores lowest (1.17). The paper briefly notes this but does not resolve whether Specificity is measuring concept discrimination or merely reflecting CLIP embedding artifacts for older models. If Specificity does not correlate with any human judgment of concept specificity, its utility as a benchmark metric is questionable.

### Trivial:

- **The "pioneer" claim for pairwise GPT-4 evaluation** (Abstract: "we pioneer the use of pairwise evaluation with GPT-4 feedback for image generation") is somewhat overstated given Chen et al. (2024a) already use multimodal LLMs as judges for visual evaluation. The contribution is the application to taxonomy image generation specifically, not the evaluation paradigm itself.

- **"Zero-shot" terminology** in the Abstract could be clearer. While standard in T2I literature to mean "no fine-tuning," the extensive use of definitions in prompts (with vs. without) is not typically considered zero-shot; it is in-context specification. The paper does report both conditions, which is good, but the framing could be more precise.

## Nice-to-Haves

- **Downstream task validation.** The paper speculates about "automating the curation of structured data resources" but does not demonstrate that images generated by top-performing models actually improve performance on any downstream taxonomy task (e.g., taxonomy enrichment, hypernym detection with visual features). A proof-of-concept experiment would significantly strengthen the practical impact argument.

- **Evaluation of closed-source models.** As acknowledged in Appendix A, the benchmark excludes models like DALL-E 3 and Midjourney. Including even one closed-source model as an upper bound would contextualize the open-source rankings and increase the benchmark's relevance to practitioners.

- **Ablation on prompt structure beyond definition inclusion.** The paper tests with/without definitions but does not explore other prompt variations (e.g., example shots, rephrased definitions). Since model rankings shift with definition inclusion, understanding sensitivity to prompt engineering would strengthen the benchmark's recommendations.

- **Stronger retrieval baseline.** The retrieval approach uses simple keyword search on Wikimedia Commons. A CLIP-based semantic retrieval baseline would better isolate whether generation truly outperforms retrieval or merely outperforms naive retrieval.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Unfair comparison with retrieval baseline (Spark Finder #3):** Criticized as unfair because generative models receive definitions while retrieval uses keyword search. Per the review rules, criticisms about unfair comparisons that favor the baseline (not the author's method) are removed. Here the asymmetry favors the author's generative approach, not the baseline—but the comparison is still informative as a lower bound, and the paper is transparent about the retrieval setup.

- **Small human evaluation sample size (Harsh Critic):** Claimed that ~600 pairs per model is insufficient. With 3,370 total pairwise comparisons across 12 models, the sampling is reasonable for ELO estimation, and this criticism is speculative without power analysis.

- **FID is misaligned with the task goal (Harsh Critic):** The paper already acknowledges that FID reflects "closeness to retrieval rather than semantic correctness" and provides 8 other metrics. Including FID as one of many metrics for completeness is standard practice.

- **LLM-generated concepts lack human validation (Harsh Critic):** While valid, the paper uses ground-truth WordNet synsets as the primary evaluation and the LLM-predicted set as an additional sensitivity analysis. Errors in LLM predictions would add noise to that subset but do not invalidate the main benchmark results.

- **Missing related works (Spark Finder / generic):** Per hard rules, no criticism about missing related works is included without external source verification.

## Novel Insights

The most interesting empirical finding is the divergence between CLIP-based similarity metrics and preference-based metrics: SDXL-turbo dominates on Lemma/Hypernym/Cohyponym Similarity yet ranks mid-to-low on human and GPT-4 preferences. This suggests that text-image alignment (as measured by CLIP) and visual quality/aesthetic preference are partially dissociable dimensions of "good" taxonomy visualization. A model can be semantically faithful to the prompt while producing images humans don't prefer, and vice versa. This has implications for benchmark design: no single metric family captures the full picture, and Specificity—the only metric that attempts to measure concept discrimination—produces rankings orthogonal to both. The benchmark's value lies precisely in exposing these tensions rather than collapsing them into a single score.

## Suggestions

- **Normalize CLIP similarities or reframe metrics as heuristic scores.** Either define a proper normalization (e.g., softmax over a concept vocabulary) to justify the probability interpretation, or explicitly present the similarity metrics as heuristic scores inspired by information-theoretic intuitions rather than formally derived probabilities. The current framing over-claims theoretical rigor.

- **Add position-swapping to GPT-4 evaluation.** For each pair, run two comparisons with swapped model positions and aggregate. This is a minimal methodological improvement that would substantially increase confidence in the ELO rankings.

- **Clarify the dataset sampling in Section 2.2.** Explain whether the stated probabilities refer to TaxoLLaMA's training data or to the test split, and provide a clear mapping from sampling probabilities to the observed dataset composition. If there is a base-rate correction, state it explicitly.

- **Include a correlation matrix across all 9 metrics.** This would reveal which metrics provide redundant signals and which capture distinct dimensions, giving users actionable guidance on which subset of metrics to prioritize for different use cases.

- **Add quantitative analysis by concept type.** Break down performance on abstract vs. concrete concepts, and by depth in the WordNet hierarchy, to directly address the paper's motivating question about how models handle different levels of abstraction.

---

**Novelty:** Moderate. The task formulation (T2I for taxonomy concepts) and the taxonomy-structure-aware metrics are novel contributions, though the theoretical grounding has gaps.

**Technical Soundness:** Partial. The benchmark design and experimental scope are strong, but the metric derivations rest on an unjustified probability approximation, and the GPT-4 evaluation has an acknowledged systematic bias without standard mitigation.

**Empirical Support:** Good. Extensive evaluation across 12 models, 9 metrics, 8 dataset splits, and both human and automated judges. The main ranking findings are likely robust despite the metric issues.

**Significance:** Moderate to High. The benchmark addresses a real gap and the released dataset is a valuable resource, but the impact depends on whether the community adopts these specific metrics given their theoretical limitations.

**Clarity:** Adequate. The paper is generally readable, but the dataset sampling description contains a confusing contradiction, and the metric definitions are split between the main text and appendix in a way that impedes understanding.

---

## c2ozZYoZFd

- GT: Reject (avg 2.7)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

This paper presents a detailed re-analysis of Nguyen et al. (2024), a high-visibility ICLR 2025 Oral paper that introduced min-p sampling for LLM decoding. Through systematic re-examination of the original paper's four lines of evidence—human evaluations, NLP benchmarks, LLM-as-a-Judge evaluations, and community adoption claims—the authors demonstrate that min-p's claimed superiority vanishes when methodological flaws are corrected (omitted data, improper statistical testing, unequal hyperparameter tuning, selective reporting). From this case study, the authors derive a general "blueprint" for more rigorous empirical ML research, centered on fair hyperparameter comparisons, proper statistical testing, data transparency, and scrutiny of qualitative claims.

## Strengths

- **Rigorous statistical re-analysis that invalidates the original paper's core claims.** The application of Bonferroni correction across 12 comparisons (Table 1) and the Intersection-Union Test for the "consistently outperforms" claim shows that min-p's statistical significance collapses from 5/12 to 1/12 at α=0.05 after correction. This is a methodologically sound and impactful demonstration of how incorrect statistical practice can manufacture false conclusions.

- **Novel "Best-of-N" hyperparameter control methodology.** Section 3 develops a principled framework for comparing methods that receive different volumes of hyperparameter tuning. By subsampling equal numbers of hyperparameters per sampler and computing maximum achievable performance, the analysis reveals that min-p's apparent advantage on GSM8K is an artifact of unequal search budgets (Fig. 4–5). This is a genuinely useful methodological contribution that addresses a widespread confound in empirical ML comparisons.

- **Comprehensive, multi-evidence re-analysis covering all four lines of original evidence.** Rather than cherry-picking one dimension, the paper systematically addresses human evaluations (omitted data, incorrect statistics, mischaracterized qualitative feedback), NLP benchmarks (unequal tuning), LLM-as-a-Judge (under-specified methodology, selective reporting favoring min-p), and adoption metrics (retracted claims). The breadth strengthens the overall case substantially.

- **Full data transparency enabling independent verification.** All re-analyses link to publicly available data, annotations, and code repositories, practicing what the blueprint preaches and making the critique itself reproducible.

## Weaknesses

### Major:

- **The blueprint's generalizability rests on a single case study.** The paper derives six general lessons from one paper's failures. While the authors state "the errors made in evaluating min-p are common in empirical machine learning research," this claim is supported only by the min-p analysis and a list of scandals in the introduction, not by systematic evidence that these specific failure modes (omitted data, incorrect multiple comparison correction, selective reporting of favorable hyperparameters) are prevalent across the field. The paper would be significantly strengthened by showing that even one additional high-profile paper exhibits similar issues under the same analytical framework, or by surveying the literature for prevalence of these specific errors.

- **GPQA benchmark claims remain unchallenged.** The original paper claimed min-p achieves "superior performance across benchmarks and temperatures" on both GSM8K and GPQA. The current re-analysis only sweeps GSM8K due to compute budget constraints. This leaves the GPQA portion of the original paper's benchmark claims unaddressed, creating a gap in the refutation. Even acknowledging the compute constraint, the paper should discuss whether there is reason to expect GPQA results to differ, or note this as an explicit limitation.

### Minor:

- **Statistical power is not discussed for the human evaluation re-analysis.** With n=53 participants and a Bonferroni-corrected α of 0.05/12 ≈ 0.004, the analysis may be underpowered to detect moderate effect sizes. The paper correctly applies the correction but does not address whether the study had sufficient power to detect the original paper's claimed effects under the corrected threshold. A brief power discussion would clarify whether "no significant difference" could reflect limited sensitivity rather than genuine equivalence.

- **The new human evaluation (Section 2.4) was conducted by the original authors in response to feedback, introducing potential confounds.** The paper documents that the original authors changed multiple factors simultaneously (temperature application order, participant pool, hyperparameters, stimuli, rubric), making it difficult to isolate why the new results differ. The paper's conclusions do not rely solely on this new study—the original data re-analysis is sufficient—but the discussion of Section 2.4 should more explicitly acknowledge these confounds.

- **The LLM-as-Judge critique identifies serious issues but does not run a corrected experiment.** The paper demonstrates that the original study used indirect comparisons (each sampler vs. basic), had 2–10× more hyperparameter tuning for min-p, and selectively reported favorable scores. However, the paper stops at critique rather than executing a direct pairwise comparison experiment that would definitively settle whether min-p matches baselines under fair conditions. Given the compute investment in Section 3, this seems like a tractable addition.

- **The 7.80 vs. 5.80 numerical discrepancy claim (Section 2.4) is stated with high confidence but limited documentation.** The paper asserts a value in Nguyen et al.'s Table 15 is incorrect based on "the authors' publicly posted data," but does not show the recalculation or provide a table comparing the reported vs. recomputed values. Given that this is a data integrity accusation, the verification should be presented with the same transparency the paper demands of others.

### Trivial:

- None of substance.

## Nice-to-Haves

- **Formalize the "Best-of-N" framework into a reusable tool or checklist.** The hyperparameter volume control methodology is the paper's most transferable contribution. Providing a simple script or checklist that researchers can apply before submission would convert the case-study lessons into an immediately actionable community resource.

- **Run a direct pairwise LLM-as-a-Judge comparison** (min-p vs. top-p, min-p vs. basic) under equal hyperparameter budgets to conclusively demonstrate the absence of min-p's advantage in that evaluation paradigm as well.

- **Add a hyperparameter sensitivity/variance analysis** to the Best-of-N results—analyzing not just maximum achievable performance but the width of high-performing regions for each sampler, which would indicate whether some samplers are more brittle or harder to tune.

- **Include GPQA in the benchmark sweep**, even with a more limited model/hyperparameter set, to close the gap in refuting the original paper's "across benchmarks" claim.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Citation stacking in the introduction** (harsh critic): Listing many references is standard in position papers arguing a field-wide problem. This is a style preference, not a substantive weakness.

- **Figure 1 being garbled** (harsh critic): This is a PDF extraction artifact, not a paper problem.

- **Title/framing mismatch between "blueprint" and case study** (harsh critic): The paper does deliver a blueprint (Section 6's six lessons), and the case study is the evidence from which it's derived. The framing is reasonable.

- **Missing creative writing benchmark** (spark finder): The paper re-analyzes the original paper's evidence; the original paper used GSM8K and GPQA for benchmarks, and human evaluations for creative writing. Requesting new experiments on WritingPrompts is scope creep beyond re-analysis.

- **Compute-adjusted tuning comparison** (spark finder): Controlling for number of hyperparameters rather than compute hours is a reasonable and simpler framework. This is a nice-to-have, not a flaw.

- **Telegram link as evidence source** (positive reviewer): The Telegram link is cited alongside a publicly accessible GitHub repository. The source is verifiable and the claim (selective reporting of scores) is also supported by the repository data. This does not undermine the paper's rigor.

## Novel Insights

The most striking insight from this re-analysis is the "Best-of-N" hyperparameter volume control concept: when methods are compared at equal search budgets, apparent advantages can evaporate entirely. This reframes the common empirical ML practice of giving new methods extensive hyperparameter sweeps while comparing against baselines at default or few settings—not just as "unfair," but as a mechanism that can manufacture false claims of superiority. The fact that min-p's advantage on GSM8K disappears under equal search budgets (Figs. 4–5) despite being presented as a fundamentally better sampling algorithm is a powerful illustration that hyperparameter tuning volume is itself a confounding variable, and that the field needs explicit accounting for it in all empirical comparisons.

## Suggestions

- Explicitly acknowledge the GPQA gap in the paper and discuss whether results are expected to generalize; even a small additional sweep on GPQA would significantly strengthen the completeness of the refutation.

- Add a brief statistical power analysis for the human evaluation study to clarify whether the "no significant difference" finding reflects genuine equivalence or limited sensitivity.

- Present the 7.80 vs. 5.80 verification with explicit recalculation details, consistent with the paper's own transparency standards.

- Consider formalizing the six lessons into a short "rigor checklist" that could be directly usable by authors and reviewers, transforming the case study's impact from "this specific paper was wrong" to "here is how to prevent this class of errors going forward."

---

## vGkXf8nvt9

- GT: Reject (avg 4.7)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary

The paper proposes Forget-to-Focus (F2F), a two-stage protocol that first applies targeted unlearning on a "forget set" of general-domain data (with an optional "retain set" for stability), then fine-tunes on a domain-specific dataset. Through experiments on medical, mathematics, and coding benchmarks across models from 0.6B to 72B parameters, the authors demonstrate that F2F consistently outperforms standard fine-tuning, DAPT, LoRA, and CurlLoRA baselines, and provide representational geometry analysis (CKA, SVCCA, Fisher information, PCA-shift) arguing that unlearning reshapes models away from generalist features toward domain-specialized structures.

## Strengths

- **Novel repurposing of unlearning for domain adaptation:** The insight that machine unlearning—traditionally a privacy tool—can serve as a preparatory capacity-reallocation mechanism for specialization is conceptually original and directly addresses the negative transfer problem. This reframing is the paper's core contribution and is well-motivated.

- **Empirical breadth across models, scales, and domains:** The paper evaluates on 5+ model architectures (0.6B–72B), three distinct domains (coding, medical, math), and multiple fine-tuning strategies. The gains are substantial and consistent: e.g., Qwen-0.6B HumanEval pass@1 improves from 19.50 (base SFT) to 42.07 (F2F); Qwen-72B HumanEval from 70.12 to 78.50 (Table 1). Multi-seed robustness (Table 9) with small standard deviations strengthens confidence.

- **Rich mechanistic analysis beyond accuracy:** The CKA/SVCCA analysis (Figures 4–5), Fisher information profiling (Figure 7), PCA-shift analysis (Figure 6), and spectral surrogate analysis for LoRA capacity (Figure 9) collectively provide a multi-faceted view of *how* unlearning alters internal representations. The finding that F2F dampens shallow-layer Fisher sensitivity while maintaining depth-wise activity is particularly interesting.

- **Calibration improvement on safety-critical tasks:** The ECE reduction from 0.277 (base tuned) to 0.050 (F2F) on MedMCQA (Table 7, Figure 8) is a practically significant finding for deployment in medical settings. The reliability diagrams confirm this is not simply a confidence collapse.

## Weaknesses

### Major:

- **Lack of compute-matched controls undermines attribution of gains:** F2F performs two sequential training phases (unlearning + fine-tuning), while baselines like SFT perform only one. The paper does not demonstrate that F2F's gains persist when compared against baselines given equivalent total optimization steps, GPU-hours, or FLOPs. It is possible that the extra gradient updates from the unlearning phase—rather than the *unlearning mechanism itself*—drive the improvements. A compute-matched SFT baseline (e.g., SFT trained for additional epochs or with a larger effective batch size to match F2F's total compute budget) is essential to validate the core claim. The paper notes runtime for unlearning is ~0.55 GPU-hours (Section C.1), but never integrates this into a fair comparison.

- **"Stable optimization dynamics" claim contradicted by small-model instability:** The abstract and conclusion assert that F2F yields "more stable optimization dynamics." However, Table 1 shows Gemma-2B-Instruct collapsing to 0.00% pass@1 after the UnlGA+GD phase, and Table 3 shows several configurations where intermediate unlearning produces extreme degradation. While retuning recovers performance, the unlearning stage itself is unstable for smaller-capacity models. The paper should explicitly qualify the stability claim and analyze under what conditions (model capacity, forget-set quality, σ/λ settings) the method becomes volatile.

### Minor:

- **Inconsistency between default hyperparameters and ablation findings:** Section 3.4 specifies λ=1.0 (GA weight) and σ=0.5 (GD weight) as defaults, but Appendix A.10 finds λ=0.5 is optimal for accuracy improvement, with λ=1.0 "severely limiting improvement." This discrepancy is not discussed. If the best-performing configuration differs from the reported defaults, the tables may not reflect the strongest version of F2F, or the defaults need justification.

- **Theoretical proposition relies on assumptions that do not hold for LLMs:** The Proposition in Section 2 assumes orthogonal decomposition of parameter space into domain-relevant (V) and irrelevant (U) subspaces, strong convexity of L_D, and β-smoothness. The paper acknowledges using "a convex linear surrogate to clarify the mechanism," but then the Corollary claims concrete convergence rate implications for the retuning phase. The gap between the simplified model and non-convex Transformer optimization is too large for the theoretical section to provide actionable guarantees; it would be more honest to frame it as intuition-building rather than formal analysis.

- **CKA/SVCCA drift may reflect optimization trajectory length, not unlearning specifically:** The representational analysis shows F2F drifts further from the base model than standard fine-tuning does (Figure 4). However, since F2F involves additional optimization steps, greater drift is expected simply from longer training. Without a compute-matched control, the attribution of geometric shifts to *unlearning* (rather than to *additional optimization*) is unconvincing. Correlating the magnitude of CKA drift with downstream accuracy would strengthen the causal claim.

- **Missing convergence dynamics evidence for "stabler optimization" claim:** The paper repeatedly claims F2F produces "more stable optimization dynamics" and "stabler optimization," but provides no training loss curves, convergence rate comparisons, or gradient norm trajectories during the fine-tuning phase. This is a straightforward experiment that would directly support the claim.

### Trivial:

- **Calibration analysis limited to medical domain only:** The improved calibration (ECE, reliability diagrams) is demonstrated only for MedMCQA. Whether F2F improves calibration on coding or math benchmarks is untested, leaving open whether this is a general property or domain-specific.

## Nice-to-Haves

- A random/naïve forget set control (e.g., random text or noise) to validate that the gains require meaningful forget-set curation rather than just an extra training perturbation.
- Calibration analysis extended to coding and math domains.
- Systematic sweep of target-domain contamination percentage in the forget set (beyond the 200/1000 BC-Mixed split) to quantify robustness to imperfect forget-set curation.
- Parameter-efficient unlearning (applying the unlearning phase only to LoRA adapters) to reduce the computational overhead.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Weakness: Baseline SFT performance may be under-optimized for Qwen-0.6B.** The reviewer speculated that SFT HumanEval of 31.71 is "relatively low compared to literature," but provides no concrete evidence of this, and the paper uses standard SFT recipes with the same learning rate and optimizer across methods. Without verified numbers from an external source, this is speculative.

- **Weakness: Qwen-72B uses QLoRA while 0.6B uses full fine-tuning, making comparison misleading.** These are different models evaluated independently; no cross-model comparison of absolute scores is being claimed as a methodological finding. The within-model improvements are the relevant metric.

- **Weakness: Missing related work on other domain adaptation or unlearning methods.** Per review rules, I cannot confirm the existence of specific uncited works and should not flag missing references.

- **Weakness: Reproducibility concerns about hyperparameters and implementation details.** The paper specifies learning rates, batch sizes, epoch counts, weight settings, and provides code. This level of detail is standard.

- **Weakness: Abstract should hint at trade-offs (compute vs. performance).** This is a stylistic preference, not a substantive flaw.

- **Weakness: BC-Cosine introduces dependency on MiniLM encoder and similarity threshold.** The paper explicitly describes this method and shows it performs comparably to BC-Select (Table 3). The dependency is acknowledged and is not a hidden limitation.

- **Weakness: Societal risk of reduced robustness to adversarial attacks.** This is outside the paper's stated scope of domain adaptation performance.

## Novel Insights

The most striking observation across the reviews and paper is the tension between F2F's framing as a "principled, stable" intervention and its empirical behavior as a *high-variance* perturbation that requires careful stabilization. The method's success seems to stem less from surgically removing "irrelevant" knowledge and more from creating a perturbed initialization that escapes the pretraining basin—functionally similar to warmup or learning rate restart strategies. The fact that BC-Cosine (automatic, similarity-based) matches BC-Select (manual curation) suggests the forget set's content matters less than its *directional opposition* to the target domain in embedding space, hinting that F2F may be fundamentally about constructing an optimization landscape reset rather than targeted knowledge erasure. This reframing would also explain why smaller models (Gemma-2B) collapse: they lack the parameter capacity to absorb a large perturbation and still recover via retuning.

## Suggestions

- **Add a compute-matched SFT baseline:** Train standard SFT for the same total number of gradient steps as F2F (unlearning + fine-tuning combined) and report results. This single experiment would either validate or invalidate the core attribution claim.
- **Plot fine-tuning loss curves:** Show training loss during the retuning phase for F2F vs. standard SFT to directly substantiate the "stabler optimization dynamics" claim.
- **Reconcile λ defaults with ablation findings:** Either justify why λ=1.0 was used as the default despite λ=0.5 being optimal in ablations, or re-report main tables with the best λ setting.
- **Qualify the stability claim:** Explicitly state in the main text that F2F can be unstable for low-capacity models and that the retain mechanism (σ > 0) is critical for smaller architectures.
- **Correlate CKA drift magnitude with downstream accuracy:** If models that drift more perform better, the mechanistic story is strengthened; if there is no correlation, the representational analysis is merely descriptive.

---

## khHNHzRjMy

- GT: Reject (avg 3.0)
- Predicted: N/A (3.5/10)
- Match: N/A

### Final Review

## Summary

EmoSign introduces the first dedicated dataset for emotion recognition in American Sign Language (ASL), comprising 200 ASL video clips annotated with sentiment (7-point scale), emotion category presence/intensity (10 categories), and open-ended descriptions of emotion cues by 3 Deaf native ASL signers with professional interpretation experience. The paper benchmarks 4 multimodal LLMs across three tasks—sentiment analysis, emotion classification, and emotion cue grounding—revealing that current models rely heavily on text captions rather than visual cues and exhibit systematic positive/neutral biases.

## Strengths

- **Community-informed annotation design**: The recruitment of Deaf native ASL signers with professional interpretation experience (Section 3.2) directly addresses a documented failure mode in prior work—FePh used hearing annotators, which risks misinterpreting grammatical facial expressions as emotional ones (Lim et al., 2024). This is a substantive methodological choice that most emotion datasets do not make.

- **Rich, multi-layered annotation schema**: The three-layer annotation (sentiment, multi-label emotion intensity, and free-text cue descriptions) goes well beyond binary presence/absence labels. The open-ended cue descriptions (e.g., signing speed, specific non-manual markers) provide grounding supervision that is rare in emotion datasets and could enable future work on interpretable affective reasoning.

- **Clear empirical demonstration of modality imbalance**: The ablation across caption-only, video-only, and video+caption conditions (Tables 3–4) provides direct evidence that current MLLMs predominantly leverage text shortcuts rather than visual understanding for sign language emotion recognition. The finding that AffectGPT defaults to "Neutral" in the video-only condition (wF1 = 0.04) is a stark and informative result.

- **Identification of systematic model biases**: The paper documents specific, reproducible failure patterns—GPT-4o collapsing to happiness/frustration in the video-only condition, AffectGPT's neutral bias, and Qwen2.5 claiming to need audio context for sign language—providing actionable directions for model improvement.

## Weaknesses

### Major:

- **VADER-based selection creates a confound for multimodal evaluation**: The dataset was constructed by selecting the 100 most positive and 100 most negative utterances based on VADER sentiment scores computed on *English text captions* (Section 3.1). This means the text modality inherently carries the emotional signal that was used to curate the dataset. Consequently, the finding that "caption-only performance was similar to or slightly better than video-only results" (Section 5.1) is partially tautological—the dataset was selected precisely because the captions contained salient emotional content. This undermines the core comparison between modalities and makes it difficult to assess whether models are genuinely failing to perceive visual emotion cues or simply encountering a dataset where the text signal is overwhelmingly strong by construction. A more informative evaluation would include analysis of cases where VADER sentiment diverges from annotator labels, which the authors allude to in Section 6 but do not quantify.

- **Critically low inter-annotator agreement on multiple emotion categories**: Table 2 reports Krippendorff's alpha of 0.119 for surprise (negative), 0.166 for disgust, and 0.330 for frustration. Alpha values below 0.2 indicate agreement barely above chance, which calls into question whether these categories are reliably annotatable with the current protocol. Using such labels as ground truth for evaluation (Table 4 includes columns for disgust, surprise, frustration, and anger at α = 0.370) means that model performance on these categories is measured against unreliable targets. The paper does not discuss the implications of this for benchmark validity or propose remediation (e.g., merging low-agreement categories, flagging them separately, or excluding them from primary metrics).

- **Emotion cue grounding task lacks quantitative evaluation**: The paper introduces "Emotion Cue Grounding" as a benchmark task (Section 4.1), defining it as identifying "video frames and spatial regions relevant to sentiment analysis and emotion classification." However, Section 5.3 evaluates this task solely through manual inspection of "several randomly selected videos" with qualitative discussion. No quantitative grounding metric (temporal IoU, spatial overlap, precision/recall of identified cues) is provided. This means the grounding task cannot be reproduced or compared by future work, which is inconsistent with claiming it as a "benchmark" contribution.

### Minor:

- **Limited dataset scale and diversity**: With 200 utterances from 4 signers in a lab environment (Section 3.4), the dataset is small for training or robust evaluation. The authors acknowledge this and cite comparable small expert-annotated datasets (Arodi et al., 2024; Krojer et al., 2024; Li et al., 2024b), but those datasets serve different tasks (anomaly detection, image editing, graph analysis) where small size is more defensible. For multimodal video understanding, 200 clips may limit the ability to draw generalizable conclusions about model behavior, even in evaluation-only mode.

- **No human baseline for video-only condition**: The paper concludes that "current multimodal models fail to integrate visual cues into emotional reasoning" (Abstract), but without measuring human performance on the video-only task (i.e., Deaf signers viewing muted clips without captions), it is impossible to determine whether poor model performance reflects a fundamental limitation of current architectures or simply the inherent difficulty of the task. A human baseline would contextualize the model results and clarify whether the performance gap is surmountable.

- **Annotation confidence scores are collected but not analyzed**: Section 3.2 states that annotators rated their confidence on a 0–100 scale after each video, yet these scores are never reported or used (e.g., to weight labels, identify ambiguous clips, or correlate with agreement). Analyzing whether low-confidence clips correspond to low-agreement categories could clarify whether the low alpha values reflect inherent ambiguity or annotation errors.

### Trivial:

- The single-expression emotion classification task (Section 4.1) merges joy and excitement into "happiness" due to high Jaccard similarity (0.81), but the original 10-category schema already included both as separate annotation targets. The rationale for collecting them separately only to merge them could be clarified.

## Nice-to-Haves

- **Fine-tuning experiments**: Demonstrating that a model can improve on this dataset through training (even small-scale) would strengthen the claim that EmoSign is a useful resource, not just an evaluation probe.

- **Analysis of VADER–annotator discrepancy**: Quantifying how often and where VADER sentiment diverges from Deaf annotator labels would directly address the selection confound and identify the most valuable clips for visual emotion learning.

- **Comparison to specialized facial expression recognition models**: Testing vision-only FER models (e.g., emotion classifiers trained on facial action units) would help isolate whether the modality imbalance finding is specific to MLLMs or a broader property of current vision systems.

- **Multi-label emotion classification benchmark**: The paper acknowledges this gap in Section 6; given that emotions frequently co-occur (Jaccard similarity of 0.81 for joy/excited), this is a natural next step.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Insufficient model variety (only 4 MLLMs tested)** — Four models spanning proprietary (GPT-4o) and open-source (AffectGPT, Qwen2.5-VL, MiniGPT4) is a reasonable starting benchmark. Demanding more models is a generic request for breadth that does not identify a specific flaw in the paper's conclusions.

- **Weakness: No statistical significance testing (confidence intervals, p-values)** — While desirable, single-run MLLM benchmarking without significance testing is the standard practice in current multimodal evaluation papers. The small dataset size does make this more concerning, but this belongs in nice-to-haves rather than as a core weakness for a dataset paper.

- **Weakness: No fine-tuning experiments** — The paper positions EmoSign as a benchmark and diagnostic dataset, not a training resource. Demonstrating learnability would strengthen the paper but is not a requirement for a dataset contribution at ICLR.

- **Weakness: Comparison to specialized FER models** — The paper's stated scope is evaluating multimodal LLMs on sign language emotion recognition. Testing monolithic FER systems is outside the paper's stated scope; the relevant question is whether MLLMs can handle this multimodal task, which the paper addresses.

- **Weakness: Disclosure of GPT-4o API costs** — This is a reproducibility nitpick about proprietary model costs, which is immaterial to the paper's scientific contributions.

- **Weakness: Grammatical error in Section 6** — The parser artifact or grammatical issue in the Limitations sentence is a formatting nitpick.

- **Weakness: No cross-signer train/test split** — The current evaluation is zero-shot with no training on EmoSign data, so cross-signer splits are not applicable. This would become relevant for fine-tuning experiments.

## Novel Insights

The most striking empirical finding is not just that models rely on text, but *how* they fail when deprived of it: they collapse to degenerate prediction patterns (AffectGPT → always Neutral; GPT-4o → happiness/frustration; MiniGPT4 → happiness), suggesting that current MLLMs have no structured representation of visual affective cues in sign language. The qualitative grounding analysis reveals a deeper pathology—models re-interpret the same visual cue in opposite directions depending on whether a caption is present (Figure 3), indicating that visual "reasoning" is being post-hoc rationalization of text-driven conclusions rather than independent perception. This is a more precise characterization than simply "models rely on text," and it has implications beyond sign language: it suggests that MLLM visual grounding for affective content is fundamentally confabulatory when textual context is available.

## Suggestions

1. **Quantify the VADER–annotator divergence**: Compute the correlation between VADER scores and annotator sentiment labels across all 200 clips. Identify the subset where they disagree, and report model performance on this subset separately. These "disagreement" clips are where visual emotion cues most diverge from text, making them the most informative for evaluating genuine visual understanding.

2. **Add a quantitative grounding metric**: Even a simple metric—e.g., overlap between model-identified temporal segments and a rough annotation of key frames—would make the grounding task reproducible and comparable. If frame-level annotations are not available, consider using the annotator cue descriptions to create pseudo-ground-truth for temporal localization.

3. **Flag or restructure low-agreement emotion categories**: Consider merging surprise(negative) and disgust into broader categories or marking them as "low-reliability" in the benchmark, with separate reporting. This would prevent misleading per-class accuracy comparisons in Table 4 where the ground truth itself is unreliable.

4. **Collect human video-only baseline**: Have Deaf signers perform the same sentiment/emotion task on muted clips (no captions) to establish an upper bound for visual-only understanding. This single addition would dramatically clarify whether the model failure is architectural or task-difficulty-related.

---

## ZMzha5gbnF

- GT: Accept (Poster) (avg 7.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Final Review

## Summary

This paper identifies and quantifies a "priming vulnerability" specific to Masked Diffusion Language Models (MDLMs), where affirmative tokens appearing at intermediate denoising steps can steer even safety-aligned models toward harmful responses. The authors propose Recovery Alignment (RA), which trains models to generate safe responses from intentionally contaminated intermediate states, and derive a tractable lower bound (Theorem 4.1) that enables efficient optimization-based attacks (First-Step GCG) without requiring denoising-process intervention.

## Strengths

- **Novel vulnerability identification with clean problem formulation.** The paper precisely defines the priming vulnerability as an MDLM-specific phenomenon arising from the iterative denoising mechanism, distinct from ARM prefilling attacks. The two-threat-model analysis (intervention vs. no intervention) is well-designed: the anchoring attack enables controlled, quantitative evaluation (Figure 2 shows ASR scaling cleanly with intervention step), while First-Step GCG demonstrates realistic exploitability. The key insight—that standard alignment trains only from clean initial states (Eq. 5) and therefore cannot constrain behavior at contaminated intermediates (Eq. 6)—is both simple and powerful.

- **Strong theoretical contribution enabling practical attacks.** Theorem 4.1's derivation of a tractable first-step lower bound resolves the gradient intractability caused by stochastic remasking. This is not merely a computational trick; it yields a 20× speedup and substantially higher ASR than MC GCG (Table 1: 58% vs. 20% on LLaDA Instruct), demonstrating that targeting the priming mechanism specifically is more effective than optimizing the full trajectory. The empirical validation of the monotonicity assumption (Appendix C.2, Figure 6) across three models adds credibility.

- **Recovery Alignment is well-motivated and effective.** The core idea—training on contaminated intermediate states rather than only from fully masked sequences—is directly motivated by the identified vulnerability mechanism. The ablation "RA w/o inter" cleanly isolates this contribution: without contaminated-state training, ASR at t_inter=4 exceeds 20% across models, while full RA reduces it to 0–1.3% (Table 2). The linear curriculum scheduling is sensible, and the ablation (Figure 3b) shows it outperforms constant and uniform alternatives.

- **Comprehensive evaluation across models, attacks, and evaluators.** Three MDLMs (LLaDA Instruct, LLaDA 1.5, MMaDA MixCoT), seven attack methods (four priming-based, three conventional), three safety evaluators (GPT-4o, LlamaGuard, keyword matching), two safety datasets, and eleven utility benchmarks. The consistency of results across this matrix is a genuine strength.

## Weaknesses

### Major:

- **Late-stage intervention remains largely undefended.** Even with RA, the anchoring attack at t_inter=32 achieves 50.7% ASR on LLaDA Instruct and 43.0% on LLaDA 1.5 (Table 2). The paper acknowledges this ("generating a fully safe response becomes challenging") but frames the overall result as "mitigated." Given that late-stage contamination with many anchor tokens is precisely where the vulnerability is most severe, the defense's partial failure at the hardest setting substantially qualifies the contribution. The paper should more explicitly characterize the boundary conditions under which recovery is feasible vs. impossible.

- **Limited generalization to certain conventional jailbreaks.** Against ReNeLLM, RA achieves 72.3% ASR on LLaDA Instruct and 71.7% on LLaDA 1.5 (Table 3)—barely improving over MOSA (77.7%) and worse than some might expect given RA's strong performance on priming attacks. The paper's proposed mechanism (harmful tokens necessarily emerge at intermediate steps, enabling re-detection) does not appear to hold for attacks like ReNeLLM that paraphrase harmful content into forms not readily detected from surface tokens. This suggests the "recovery" capability is more narrowly applicable than claimed, and the paper's discussion of generalization (end of Section 6.2) should more honestly delineate where RA helps and where it does not.

- **The monotonicity assumption lacks failure-mode characterization.** Theorem 4.1 underpins First-Step GCG and, by extension, the paper's claim that the vulnerability is exploitable without intervention. While Figure 6 shows the mean monotonicity gap is positive, the paper does not report what fraction of individual prompts violate the assumption or how the attack performs on those cases. If the assumption fails on a non-trivial subset of harmful queries, the lower bound—and thus the attack's theoretical guarantee—does not hold for those cases. A per-prompt analysis (even just reporting the violation rate) would significantly strengthen the theoretical contribution.

### Minor:

- **Utility degradation on specific benchmarks is understated.** Table 4 shows HumanEval dropping from 22.0 to 17.1 for LLaDA (a ~22% relative decrease), and PIQA from 74.4 to 71.6. The paper states "we do not observe substantial degradation" and highlights improvements on TruthfulQA and MBPP. While the average remains stable, the HumanEval drop is non-trivial for a code generation benchmark, and the claim of "minimal impact" would be more credible with explicit acknowledgment of this trade-off—particularly whether safety-oriented recovery training might systematically suppress the confident, deterministic generation required for code.

- **Over-refusal on benign queries is not evaluated.** The paper measures utility on standard benchmarks (MMLU, ARC, etc.) but does not assess whether RA increases false-positive refusal rates on borderline or sensitive-but-benign queries. Safety alignment methods are known to cause over-refusal, and the absence of this evaluation leaves the "minimal impact" claim incomplete. This is a notable gap given that RA trains on harmful contaminated states, which could shift the model's refusal boundary.

- **Computational cost of RA relative to baselines is not discussed in the main text.** Appendix C.4 reports ~16 hours on 4 H100 GPUs for 2,500 steps, but the main text claims RA is "practical and scalable" (Section 5) without comparing this cost to SFT or DPO. Practitioners need this comparison to assess whether the safety gains justify the overhead of on-policy rollouts from contaminated states.

### Trivial:

- The "Limitations" section mentions the DPO-style alternative but dismisses it due to data-construction cost. A brief discussion of why the RLHF instantiation was chosen over this alternative (beyond data availability) would clarify the design rationale.

## Nice-to-Haves

- Evaluate RA on benign instruction-following datasets (e.g., AlpacaEval, JustAsk) to measure over-refusal rates explicitly.
- Compare RA against an inference-time filtering baseline (e.g., reward-model-based output rejection) to justify the training-time cost.
- Visualize denoising trajectory token probabilities for RA vs. original models on a recovered example, to verify that the model actively overwrites harmful anchors rather than simply ignoring them.
- Test First-Step GCG suffixes for cross-model transfer to probe whether the vulnerability is architectural or model-specific.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **(Weakness) "Limited evaluation scope and generalization concerns"** — claiming only 2 datasets and 3 models is insufficient. The paper evaluates on 3 models, 7 attacks, 3 evaluators, 2 safety datasets, and 11 utility benchmarks. This is above-average comprehensiveness for the area.
- **(Weakness) "Incomplete ablation analysis of RA components"** — the paper includes ablations on t_max (Figure 3a), scheduling strategies (Figure 3b), and RA w/o inter (Table 2). While more ablations could always be added, the core design choices are tested.
- **(Weakness) "Missing comparison with ARM safety techniques adapted to MDLMs"** — the paper compares against SFT, DPO, and MOSA (the only existing MDLM-specific safety method). These are the relevant baselines.
- **(Weakness) "Threat model realism for anchoring attack"** — the paper explicitly addresses this by designing two threat models (Section 4.1 with intervention, Section 4.2 without). The anchoring attack is explicitly framed as a tool for "comprehensive evaluation," not a realistic attack.
- **(Weakness) "Clarify that only unmasked subset acts as anchor"** — re-reading Section 4.1, the anchoring attack applies the masking strategy m_{t_inter} to the harmful response r, producing r_{t_inter}. Since the masking strategy retains (t_inter/T) fraction of tokens unmasked, only those tokens act as anchors. The paper describes this correctly; the confusion arose from the reviewer's misreading.
- **(Weakness) Formatting/style issues with equations** — parser artifacts, not paper problems.

## Novel Insights

The most interesting structural insight across the reviews is that Recovery Alignment's mechanism may be dual-natured: it succeeds against priming attacks because contaminated intermediate states explicitly appear during training, but its partial failure against attacks like ReNeLLM (where harmfulness is obscured in surface form) suggests that the "recovery" is more of a pattern-matching response to known harmful token configurations than a deep re-evaluation of response semantics. This hints at a fundamental limitation: training on contaminated states teaches the model to resist *its own* harmful trajectories, but not necessarily to detect novel harmfulness that emerges through semantically subtle rephrasing. The distinction between "trajectory-level recovery" and "semantic-level safety" may be an important axis for future MDLM safety work.

## Suggestions

- Report the per-prompt monotonicity violation rate for Theorem 4.1 (even as a single number in the appendix) to clarify the theoretical bound's reliability.
- Add a brief table or paragraph in the main text comparing RA's training cost to SFT/DPO baselines (wall-clock time and approximate FLOPs), since the "practical and scalable" claim currently rests only on appendix data.
- Include 2–3 qualitative failure cases where RA fails to recover (e.g., at t_inter=32 or under ReNeLLM), with the actual generated text, to help readers understand the defense's boundaries.
- Evaluate over-refusal explicitly on a benign instruction-following benchmark; this is a low-cost addition that would significantly strengthen the utility-preservation claim.

---

**Axis Evaluations:**

- **Novelty:** High. The priming vulnerability is a genuinely new concept distinct from ARM prefilling, and Recovery Alignment is a well-motivated MDLM-specific defense. Theorem 4.1 is a clean theoretical result.

- **Technical Soundness:** Good with caveats. The core empirical analysis is rigorous, but the monotonicity assumption lacks failure-mode characterization, and the defense has clear boundary limitations that are under-discussed.

- **Empirical Support:** Strong on coverage (models × attacks × evaluators), but weakened by the substantial residual vulnerability at late intervention steps and under ReNeLLM, which tempers the "mitigation" framing.

- **Significance:** Significant. As MDLMs gain traction as ARM alternatives, establishing their distinct safety failure modes and tailored defenses is important and timely. The work sets a clear foundation for MDLM safety research.

- **Clarity:** Good. The paper is well-organized with a clean narrative arc from vulnerability identification → theoretical analysis → defense proposal → evaluation. The two-threat-model structure is particularly effective.

---

## CTEXdHB1BB

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Final Review

## Summary

The paper introduces CANON (Conditional Advantage Estimation), a method that regroups sampled responses into two equal-sized groups based on a target metric (e.g., entropy or length) and computes advantage through both inter-group comparison (identifying which metric trend correlates with higher reward) and intra-group comparison (selecting superior responses within the same trend). The mixing parameter μ balances these components, with DR.GRPO recovered as a special case (μ=0.5). Experiments across three LLMs demonstrate that CANON-Inter improves math accuracy by 1.9 points over DR.GRPO, CANON-Intra improves high-complexity logic by 5.2 points, and a length-weighted variant (CANON-Eff) achieves a superior Pareto frontier in the performance–efficiency trade-off.

## Strengths

- **Principled generalization of existing methods.** The decomposition of advantage estimation into inter-group and intra-group components is a clean conceptual contribution. The formal derivation showing DR.GRPO as a special case (Eq. 7, when μ=0.5 and groups are equal-sized) provides a clear theoretical grounding and positions CANON as a proper generalization rather than an ad hoc modification.

- **Task-adaptive behavior through a single mechanism.** The finding that inter-group advantage exploits known metric–reward correlations (benefiting math) while intra-group advantage encourages exploration from the disadvantaged group (benefiting complex logic) is insightful. The analysis in Figure 2f and Figure 6, showing that intra-group advantage produces positive reflection gains that correlate with logic performance improvements, provides mechanistic understanding beyond simple accuracy numbers.

- **Strong efficiency results with a favorable Pareto frontier.** CANON-Eff dominates all baselines across the entire efficiency frontier (Figure 4c). The practical significance of 45.5% token reduction at the same performance level and 2.63× performance at low token budgets is substantial. Notably, CANON-Eff remains stable where Length Reward(+) collapses (performance drops from 54.8 to 22.5 when coefficient changes from 0.004 to 0.005), demonstrating meaningful robustness gains.

- **Selective amplification is empirically validated.** Table 4 directly compares CANON against naive advantage scaling (A=A*2), showing that simple amplification hurts logic performance (25.1 vs. 26.2 baseline) while CANON-Inter achieves 57.6 on math. Table 12 shows random regrouping fails to improve, confirming that meaningful metric-based grouping is essential. Together, these ablations substantiate the claim that CANON's benefit comes from selective metric-specific amplification rather than generic signal boosting.

## Weaknesses

### Major:

- **Scheduling strategy selection introduces unprincipled model-specific tuning.** The paper tests four scheduling strategies and selects different ones per model: Cosin-First-Inter-Later-Intra for Qwen2.5-7B and Llama3.1-8B, and First-Inter-Later-Intra for Qwen2.5-1.5B (Section 5.2, lines 555–556). No guidance is provided for selecting a schedule for a new model. While Table 10 shows a monotonic relationship between μ and task performance (higher μ → better math, worse logic), the *functional form* of the schedule (cosine vs. linear, accuracy-based vs. step-based) requires its own tuning. This undermines the claim that CANON is a drop-in improvement over DR.GRPO, since practitioners must search over scheduling strategies rather than simply setting μ.

- **The inter/intra tension reveals task-dependent benefits rather than universal improvement.** CANON-Inter outperforms on math but underperforms on logic; CANON-Intra shows the opposite pattern (Table 1). CANON-Dynamic resolves this only by carefully tuned scheduling. This suggests the core method's advantage is fundamentally task-contingent: the metric signal that helps one domain may hurt another, and without knowing the target distribution in advance, there is no principled way to set μ. The paper does not analyze what task properties predict which advantage type will dominate, limiting practical applicability.

### Minor:

- **The "preference-free" framing in the abstract is slightly overstated.** The abstract states CANON works "without presuming its direction," but Section 4.3 introduces α to explicitly bias the model toward shorter responses. While α is presented as an optional efficiency control, the abstract claims it as part of CANON's contributions ("When applied to response length, CANON further improves token efficiency"). The base method (CANON-Inter/Intra) is genuinely direction-free in discovering metric trends, but CANON-Eff explicitly encodes a directional prior. A clearer distinction between the discovery mechanism and the directed efficiency extension would strengthen the framing.

- **Efficiency evaluation (Table 3) is limited to math benchmarks.** The length-weighted CANON-Eff results are only reported for six math benchmarks, with no evaluation on the high-complexity logic tasks where CANON-Intra excelled. Given that CANON-Intra already produces 36.6% shorter responses on logic tasks (Table 1), the interaction between efficiency weighting and logic performance remains unexplored. This is a gap in assessing the generality of the efficiency gains.

- **Theorem 2's independence assumption warrants empirical validation.** The theorem assumes conditions c₁ and c₂ are independent (P(o∈C₁∩C₂)=P(o∈C₁)P(o∈C₂)). In practice, entropy and response length are often correlated in LLM rollouts. While the theorem correctly characterizes the theoretical selective amplification property, the paper provides no empirical check of metric correlations during training, leaving open whether CANON-Entropy inadvertently influences length distributions beyond what the entropy grouping alone would predict.

### Trivial:

- The notation switches between C_q^+ and G_q^+ across Sections 4.1 and 4.2 without explicit clarification; consistent use of one symbol would aid readability.

## Nice-to-Haves

- Statistical significance tests or confidence intervals across multiple random seeds for the key claims (e.g., the 1.9-point math improvement), though single-run evaluation is standard in this setting.
- An empirical measurement of the actual advantage magnitude ratio (|Â_inter|/|Â_DR.GRPO|) during training to directly validate Theorem 1's amplification prediction.
- Ablation of all four scheduling strategies across all three models in the main text, rather than reporting only the selected best strategy per model.
- Evaluation on a model at a larger scale (e.g., 32B+) to validate scalability claims implied by the title "Large Reasoning Models."

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Computational overhead of sorting/regrouping.** With G=16, sorting is O(16 log 16), which is negligible compared to the forward/backward pass costs of the LLM itself. The harsh critic themselves acknowledged this is "small for G=16." This is a nitpick about trivial implementation details.

- **Weakness: Table 1 readability issues.** This is a PDF parser artifact, not a paper problem. Per hard rules, formatting nitpicks are removed.

- **Weakness: Missing comparison with Chen et al. 2025b (Seed-GRPO).** Per hard rules, I cannot confirm the existence or relevance of this as a missing baseline from external sources.

- **Weakness: No evaluation on non-verifiable reward tasks.** The paper is explicitly about RLVR (Reinforcement Learning with Verifiable Rewards). Criticizing absence of evaluation on learned reward models is scope creep.

- **Weakness: Dataset heterogeneity for Llama3.1-8B.** The paper clearly explains the rationale: Llama's weak math capability requires a simpler dataset (Appendix C.5). The within-model comparison between methods is fair since all methods trained on Llama use the same dataset.

- **Weakness: Need for larger models (32B+).** Generic "test on more models" criticism. The paper already tests 3 models from 2 families at 2 scales.

- **Weakness: Incomplete comparison with recent advantage shaping baselines.** The paper compares against Entropy Adv (Cheng et al., 2025) and Clip-Cov (Cui et al., 2025) as entropy baselines, and Length Reward(+/*) as efficiency baselines. The claim of missing comparisons is not substantiated.

- **Strength: "The paper is well-written / comprehensive evaluation."** Generic strengths that would apply to many papers. Weakened per soft rules.

## Novel Insights

The inter-group/intra-group decomposition reveals a fundamental tension in RLVR training: exploitation of known metric–reward correlations (via inter-group comparison) benefits in-domain performance but suppresses exploration needed for out-of-distribution generalization, while encouraging exploration from the disadvantaged group (via intra-group comparison) enables breakthroughs on complex tasks at the cost of in-domain efficiency. This trade-off, visible in the training dynamics (Figure 2: CANON-Inter stably decreases entropy while CANON-Intra's logic performance surges only after reflection gains cross zero at ~90 steps), suggests that the optimal RLVR training trajectory is inherently non-monotonic—early exploitation followed by late exploration—and that GRPO-style flat baselines are fundamentally limited in capturing this phased structure.

## Suggestions

- Provide decision criteria for scheduling strategy selection (e.g., based on model capability or training accuracy range) to reduce the practical tuning burden. The observation that accuracy-based scheduling works well for Qwen2.5-1.5B (accuracy range 0–0.6) but not for higher-accuracy models is a starting point.
- Include logic benchmark results for CANON-Eff (Table 3) to complete the efficiency evaluation across both task domains.
- Add a brief empirical analysis of entropy–length correlation in rollout distributions during training to validate or qualify Theorem 2's independence assumption.
- When presenting CANON-Dynamic results, report all four scheduling strategies for at least one model (in the main text or appendix) rather than only the selected best, to demonstrate the sensitivity to schedule choice.

---

**Axis Evaluations:**

- **Novelty:** Moderate-to-high. The conditional regrouping mechanism and the inter/intra decomposition are clean and non-obvious; the DR.GRPO-as-special-case result is meaningful. The method is a genuine conceptual advance over simple reward/advantage shaping.

- **Technical soundness:** Good. Theorems are correct and well-motivated. The independence assumption in Theorem 2 is strong but acknowledged. Empirical ablations (Table 4, Table 12) validate core claims.

- **Empirical support:** Strong on math reasoning and efficiency; good on logic reasoning. The efficiency Pareto frontier result is particularly compelling. Gaps exist in efficiency evaluation on logic tasks and in transparency around scheduling strategy selection.

- **Significance:** Significant for the RLVR community. The practical efficiency gains (45.5% token reduction) are meaningful for deployment, and the theoretical positioning provides a foundation for future work on metric-aware advantage estimation.

- **Clarity:** Generally good. The method is well-motivated and the paper flows logically. Minor issues with notation consistency and the framing transition between the preference-free base method and the directed efficiency variant.

---

## pNpnqsn0Si

- GT: Reject (avg 3.0)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

##Summary

Thoughtbubbles introduces a transformer variant that enables unsupervised, adaptive parallel computation in latent space by learning to fork or delete residual streams mid-network. Tokens requiring more computation form "bubbles" of cloned residuals, with the forking behavior learned entirely from standard language modeling loss during pretraining. Experiments across 150M–772M parameter scales on OpenWebText and peS2o show consistent perplexity and zero-shot evaluation improvements over both parameter-matched and computation-matched baselines.

## Strengths

- **Genuinely unsupervised adaptive compute during pretraining**: Unlike CoT or pause-token methods that require fine-tuning, special prompts, or manually inserted tokens, Thoughtbubbles learns dynamic allocation of latent parallelism solely from the cross-entropy LM objective. This is a specific and non-trivial achievement—most prior work requires either architectural rigidity (fixed pause positions) or auxiliary training signals. The paper demonstrates this across multiple scales.

- **Careful baseline design isolating the value of adaptivity**: The inclusion of both parameter-matched (standard GPT-2) and computation-matched (Copy-3/Copy-5) baselines is methodologically sound. Copy-N baselines are the right control: they provide the same extra residual stream capacity but without dynamic allocation, directly testing whether *adaptivity* matters beyond raw compute. The consistent gap between Thoughtbubbles and Copy-N across scales (Table 1) is the paper's strongest empirical result.

- **Interpretable computation allocation without explicit supervision**: Figure 5 shows the learned fork allocation correlates with posterior entropy (measured both by the forking model and an independent baseline LM), and Figure 7 shows sensible allocation on the synthetic CLUTRR task. These analyses provide evidence that the model discovers meaningful computational heuristics from LM loss alone.

- **Thoughtful position encoding for variable fork counts**: The partial rotation RoPE variant (Appendix D, Eq. 13) that scales rotation proportionally to fork count is a specific and necessary technical contribution for making the architecture work, addressing a real challenge that would otherwise break positional semantics.

## Weaknesses

### Major:

- **Missing ablation on whether *learned* fork decisions matter**: The paper attributes its gains to adaptive forking, but never tests whether the *learned* allocation strategy is responsible vs. the mere *capacity* for extra residual streams. A critical missing ablation is: what happens with *random* forking at the same average rate? Or fixed forking at every token? Similarly, the score-attenuation mechanism (Eqs. 8–10) creates a strong inductive bias coupling forking scores to residual updates—without ablating it, we cannot determine whether the forking decisions are learning meaningful structure or whether the architectural bias alone drives the gains. These ablations are essential for validating the core claim that *adaptive, learned allocation* is the source of improvement.

- **KV cache memory overhead is unaddressed**: The paper proposes an inference-time architecture but does not discuss the memory implications of maintaining κ× longer sequences during autoregressive generation. With κ=4L, the KV cache is up to 4× larger than a standard model. For long-context inference, KV cache memory is often a harder constraint than FLOPs. The paper acknowledges wall-clock inefficiency (Section 8) but the memory constraint is a distinct and arguably more fundamental limitation for deployment that deserves explicit discussion and quantification.

- **Claim inconsistency about uncertainty allocation between Introduction and Analysis**: The Introduction states the method "allocates more computation at regions of higher uncertainty (i.e., posterior entropy)" (Section 1, Contribution 3). However, Section 5 reveals a "concave parabolic relationship" where computation *decreases* at the highest uncertainty tokens. The authors' explanation (highest uncertainty at clause boundaries where extra computation is unhelpful) is reasonable, but the Introduction's claim is misleading as written—it implies a monotonic relationship that the data do not support. The Introduction should be corrected to say the model allocates more computation at *moderate-to-high* uncertainty regions.

### Minor:

- **Gradient flow through the non-differentiable top-k is underspecified**: The forking judgment uses hard top-k selection (Section 2.3), which is non-differentiable. The paper does not explicitly state how gradients propagate through this operation. While the standard approach (straight-through: gradients flow through selected elements, dropped elements receive zero gradients) is likely used—and the authors acknowledge the resulting gradient bottleneck in Limitations—the method section should state this explicitly. The current description leaves the reader to infer the gradient mechanism.

- **Limited evaluation scope for the stated motivation**: The paper motivates the work as enabling "complex, multi-step problems" (Section 1), but evaluates only on perplexity and zero-shot NLU tasks (LAMBADA, HellaSwag, BLiMP, PIQA). The authors acknowledge in Limitations that scale constraints prevent evaluation on reasoning benchmarks like GSM8k. While understandable, this gap between motivation and evaluation leaves the central promise unverified—the tasks where adaptive computation should matter most (multi-step reasoning) are exactly those not tested.

- **Dual suppression mechanism rationale unclear**: The architecture applies both structural deletion (top-k removal) and attention masking (score attenuation, Eqs. 8–10) to low-scoring tokens. The paper does not analyze why both are needed or how they interact. If score attenuation can effectively zero out a token's contribution, why also delete it? Presumably top-k serves to reduce sequence length for computational efficiency, but this trade-off is never articulated.

### Trivial:

- The overforking ablation (Appendix B) is minimal (only 25K steps, ~0.8BT tokens) and does not conclusively establish that additional forking layers fail to help; the result (28.02 vs. 29.84 perplexity) is reported as "slightly worse" but the training budget is insufficient to draw firm conclusions.

## Nice-to-Haves

- Comparison against established adaptive computation methods (e.g., Universal Transformers, Mixture-of-Depths) at the same scale, to contextualize the gains against a broader landscape of adaptive architectures.
- Analysis of semantic divergence between parent and forked residual streams (e.g., cosine similarity across layers) to verify that forks compute genuinely different representations rather than redundant copies.
- Evaluation of whether pretrained forking behaviors are preserved after fine-tuning on downstream tasks.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Unfair capacity comparisons" / "learned allocation vs. static allocation is an unfair comparison"**: The comparison between Thoughtbubbles (learned allocation) and Copy-N (static allocation) is not unfair—it is the core experimental design. Copy-N is the appropriate baseline to test whether *adaptivity* provides value beyond *more compute*. The asymmetry (learning vs. not learning) is the independent variable being tested.

- **"Parallel Thinking is misleading regarding latency"**: The computation IS architecturally parallel (multiple residual streams processed simultaneously in the same forward pass). Latency concerns are real but separate; the term is not misleading—it describes the structural property correctly.

- **"Equation 4 contradiction with 'ignore rightmost token'"**: On close reading, this is not a contradiction. Eq. 4 forces the keep score to 1 for the *top-k selection* (structural: the original token is never deleted), but the *cumulative scores* used for attention attenuation (Eqs. 8–10) are not forced to 1, allowing the original token to be functionally "ignored" via attention masking even while structurally present. The text's statement is correct.

- **"Missing FLOP-matched Dense Model baseline"**: The paper already shows a 319M Thoughtbubbles model beating a 772M baseline (Figure 3), which partially addresses whether adaptivity beats scale. While a wider/deeper model at matched FLOPs would be informative, the existing comparison provides relevant evidence. Demanding additional FLOP-matched dense baselines is a generic expansion of experimental scope.

- **"CoT can reduce performance on certain tasks"**: This is scope creep. The paper does not claim universal improvement across all task types and actually reports degraded BLiMP performance vs. computation-matched baselines. The finding that adaptive compute may not help for syntax is already present in the results.

- **"Extend scaling to 1B+"**: The paper already scales to 772M and shows consistent trends. Demanding larger scale is a generic weakness that does not engage with the specific contribution.

## Novel Insights

The concave relationship between uncertainty and computation allocation (Figure 5) is a genuinely interesting finding with potential broader implications. The model learns that moderate-uncertainty tokens (e.g., choosing between a few plausible continuations) benefit most from extra compute, while the highest-uncertainty tokens (e.g., clause boundaries, coreference edges) are inherently unresolvable by additional computation. This suggests a natural "computability frontier" where adaptive compute provides diminishing returns—a principle that could inform the design of future adaptive inference systems beyond this specific architecture.

## Suggestions

- Run a random-forking ablation (fork at the same average rate but with random allocation) and a fixed-forking ablation (fork every token equally) to establish that *learned* allocation is the source of improvement, not just extra capacity.
- Add explicit discussion of KV cache memory scaling with κ and quantify the memory-per-token cost relative to the baseline, as this is a deployment-critical metric.
- Correct the Introduction's claim about uncertainty allocation to reflect the concave relationship shown in the Analysis section.
- Explicitly state the gradient propagation strategy through top-k in Section 2.3 (e.g., "gradients propagate through selected elements via straight-through estimation; dropped elements receive zero gradients").

---

**Quality Assessment:**

- **Novelty**: High. The forking mechanism with cumulative scores learned unsupervised during pretraining is architecturally novel and distinct from prior pause-token or adaptive-depth approaches.
- **Technical soundness**: Moderate. The architecture is well-designed, but the lack of key ablations (random forking, score attenuation) leaves the causal mechanism behind the gains ambiguous. The claim inconsistency about uncertainty allocation is a clarity issue that could mislead readers.
- **Empirical support**: Moderate-to-good. Consistent improvements across scales and datasets against meaningful baselines, but missing ablations weaken the attribution of gains to the claimed mechanism. The evaluation tasks do not include the reasoning benchmarks most relevant to the stated motivation.
- **Significance**: Moderate-to-high. If the ablations confirm that learned allocation (not just capacity) drives the gains, this would be a significant contribution to adaptive computation. The unsupervised pretraining-time learning of adaptive compute is the key differentiator from prior work.
- **Clarity**: Moderate-to-good. The method is generally well-described, but the gradient flow through top-k is underspecified and the uncertainty allocation claim is inconsistent between sections.

---

## iIEEgI6WsF

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary
This paper proposes On-Demand Communication (ODC), which replaces per-layer collective communication (all-gather/reduce-scatter) in FSDP with point-to-point RDMA primitives (gather/scatter-accumulate), effectively reframing FSDP as a decentralized parameter server. By relaxing synchronization from the layer level to the minibatch level, ODC decouples device execution and enables simpler minibatch-level load balancing (LB-Mini), achieving up to 36% throughput speedup over standard FSDP on long-sequence SFT tasks.

## Strengths
- **Precise, evidence-backed root cause identification**: The paper identifies per-layer synchronization barriers as the direct cause of up to 50% device idle time under imbalanced workloads (Table 6), grounding the motivation in concrete measurements rather than hypothetical arguments.
- **Elegant conceptual reframing of FSDP as decentralized PS**: Replacing collectives with on-demand P2P communication while preserving FSDP's sharding layout is a genuinely novel perspective that yields both conceptual insight (the PS connection) and practical benefit (decoupled device progress) — most prior work focused on better packing rather than questioning the communication model itself.
- **Comprehensive parametric evaluation**: Section 5.3 systematically varies minibatch size, max length, packing ratio, and device count, providing clear operational boundaries for when ODC helps most and when its benefits diminish.

## Weaknesses

### Major:
- **Inter-node communication overhead limits multi-node scalability**: Figure 11 shows ODC primitives are significantly slower than NCCL collectives for cross-node communication because they forgo hierarchical topology optimizations (e.g., inter-node broadcast + intra-node broadcast). While the paper argues that long-sequence computation hides this latency and proposes hybrid sharding as mitigation (Section 6.1, Appendix E), hybrid sharding increases per-node memory (Figure 13) and may not suffice for memory-constrained large-scale training. This is the most important scalability limitation and should be more prominently quantified in the main results — specifically, a per-experiment breakdown of how many nodes were used and whether inter-node traffic was a factor would help readers assess real-world applicability.

- **Headline speedups conflate communication and load-balancing contributions**: The peak 36% speedup (Table 5, 1.5B LongAlign, minibatch=4) comes from ODC+LB-Mini, while the isolated communication benefit (ODC LocalSort vs. Collective LocalSort) is only 0–10% in most configurations. Since LB-Mini is only feasible under ODC's relaxed synchronization, the two contributions are intertwined, but the paper's framing ("ODC achieves up to 36% speedup") attributes the full gain to the communication scheme. The ODC+LB-Micro vs. Collective+LB-Micro comparison better isolates the communication contribution (e.g., 16–23% for 1.5B LongAlign at minibatch=4/8), and the paper would be more honest by foregrounding these numbers and positioning LB-Mini as a complementary enabler.

### Minor:
- **RL speedups are modest and framework-constrained**: The ~10% RL speedup (Section 5.2) is limited because `verl` requires identical sample counts per device, preventing full LB-Mini usage. This means a key part of the proposed system cannot be evaluated in the RL setting, which is itself a major post-training workload. The paper is transparent about this, but it weakens the "diverse post-training tasks" claim.
- **Convergence verification is thin**: Appendix F validates identical loss curves on only 8k samples with a 1.5B model trained from scratch. Since ODC changes the gradient accumulation timing (scatter-accumulate happens on-demand rather than via coordinated reduce-scatter), subtle numerical differences could emerge over longer runs or with optimizer states that depend on gradient statistics. A longer convergence check or gradient norm comparison would strengthen confidence.
- **Hardware dependency on RDMA limits portability**: ODC requires CUDA IPC (intra-node) and NVSHMEM (inter-node), restricting deployment to NVIDIA GPU clusters with RDMA configured. The paper does not discuss fallback paths for TCP/Ethernet clusters or non-NVIDIA hardware, limiting the generality of the contribution.

### Trivial:
- **Gradient buffer memory overhead**: Appendix B bounds the dedicated per-client buffer memory to M per server, which temporarily increases gradient memory compared to in-flight reduction in ring-based reduce-scatter. For models already near memory limits, this could matter, though the paper shows it is manageable in practice.

## Nice-to-Haves
- GPU timeline traces (e.g., Nsight Systems) comparing FSDP and ODC to visually confirm barrier elimination and bubble reduction, complementing the schematic Figures 1–2.
- Gradient norm or parameter divergence analysis between ODC and FSDP to rigorously verify numerical equivalence beyond loss curves.
- Topology-aware P2P routing (e.g., intra-node cache-and-forward) to mitigate the inter-node bandwidth penalty discussed in Section 6.1.
- Full RL framework integration removing `verl`'s equal-sample-count constraint to demonstrate ODC's complete potential for RL post-training.
- A plot of speedup vs. inter-node communication fraction to define the operational boundary where ODC becomes net-negative.

## Removed Points
*These points are flagged to be removed, treat them with caution:*
- **Synchronous vs. asynchronous ambiguity** (Harsh Critic): The critic claims ambiguity between the introduction's "synchronous optimization semantics" claim and Section 6.2's "Relaxing Synchronization" future work. The paper is clear throughout (Section 3, Section 3.2) that ODC preserves a minibatch-level barrier and synchronous semantics; Section 6.2 explicitly discusses relaxing this as *future work*. No ambiguity exists.
- **Presentation complexity / implementation hard to understand** (Harsh Critic, transferred from Cut Cross-Entropy review): This is a style/formatting nitpick; the implementation details are in appendices with adequate pseudocode and memory analysis.
- **Baseline fairness — baselines not fully optimized** (Harsh Critic, transferred from ProTrain review): The paper compares ODC vs. Collective using the *same* packing strategy (LB-Micro), which fairly isolates the communication contribution. Requesting additional framework-specific optimizations is scope creep.
- **Lack of ablation studies** (Harsh Critic, transferred from Cut Cross-Entropy review): The paper provides parametric studies (Section 5.3) and compares four combinations of communication scheme × packing strategy, which serves as ablation. More granular ablations would be nice but are not a core flaw.
- **Demand for DeepSpeed/Megatron comparison** (Spark Finder): ODC is a communication scheme within FSDP, not a full training framework. Comparing against end-to-end frameworks would conflate many unrelated variables.
- **Deadlock concerns** (Harsh Critic): RDMA operations (CUDA IPC, NVSHMEM put/get) are one-sided by design — the initiator controls the transfer without requiring active remote participation. The paper's "non-intrusive" claim is consistent with RDMA semantics. Deadlock concerns reflect a misunderstanding of the communication model used.
- **Scalability beyond 32 GPUs demand** (Spark Finder): This is a generic "needs larger scale" request; 32 GPUs across multiple model sizes is adequate for the paper's scope.
- **Load balancing algorithm computational cost** (Harsh Critic): Karmarkar-Karp is a well-known efficient heuristic; no evidence is provided that its runtime is non-trivial relative to training time.

## Novel Insights
The key insight that emerges from synthesizing these reviews is that ODC's contribution is best understood as a *system co-design* rather than a pure communication optimization. The real leverage comes from the virtuous cycle: relaxing synchronization (via P2P) *enables* better load balancing (LB-Mini), which in turn *amplifies* the benefit of relaxed synchronization. The paper's most honest numbers (ODC+LB-Micro vs. Collective+LB-Micro) show that communication alone gives 16–23% gains on high-imbalance SFT, while LB-Mini pushes this to 34–36%. This co-design is the paper's genuine novelty, but the current framing obscures it by attributing all gains to "ODC." A reframe positioning ODC as the *enabler* of a new load-balancing regime — rather than the sole performance driver — would be both more honest and more intellectually interesting.

## Suggestions
- In the abstract and main results, report the isolated communication contribution (ODC+LB-Micro vs. Collective+LB-Micro) alongside the full ODC+LB-Mini numbers, and explicitly discuss the co-design nature of the gains.
- Add a table or figure breaking down which experiments were single-node vs. multi-node, so readers can assess whether the inter-node overhead (Figure 11) affected each result.
- Extend the convergence verification to at least one 7B+ model with a longer training run (e.g., 50k+ samples), or add a gradient norm comparison to validate numerical equivalence beyond loss curves.

---

## c7OsKOOZo8

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

##Summary

The paper proposes an end-to-end framework for multi-view diabetic retinopathy (DR) grading that eliminates the need for external lesion/vessel annotations by self-generating lesion proposals during training and inference. A Grade-Activated Lesion Proposal (GALP) module derives grade-conditioned evidence maps from stage-wise auxiliary classifiers and selects Top-K high-evidence regions as proposals, while a Cross-View Lesion Expert-Guided Regional Fusion (LGRF) module routes these proposals through a gated mixture-of-experts mechanism for selective cross-view fusion. Experiments on two multi-view DR benchmarks (MFIDDR and DRTiD) show that the method matches or surpasses externally-informed baselines without requiring auxiliary annotations.

## Strengths

- **Effective reduction of annotation dependency while preserving accuracy:** The lesion-free variant (83.9% Acc on MFIDDR) surpasses all end-to-end baselines and several externally-informed methods (e.g., LFMVDR with lesion at 82.2%, CVSA with vessel at 82.6%), directly validating the core claim that self-derived proposals can substitute for costly external cues. This is the paper's strongest result and is clearly demonstrated in Table 1.

- **Well-motivated architectural design for the multi-view medical setting:** The GALP→LGRF pipeline is logically coherent: GALP generates spatially-focused proposals that reduce background noise, and LGRF uses cross-view context to gate which experts process those proposals. The ablation (Table 4) confirms that each component contributes meaningfully (removing GALP drops accuracy by 1.2%, removing LGRF by 1.6%).

- **Consistent improvements across two datasets with different view counts:** The method generalizes from 4-view (MFIDDR) to 2-view (DRTiD) settings, achieving 76.0% accuracy on DRTiD versus the prior SOTA CrossFiT (75.6%) that requires OD/macular coordinates. This consistency strengthens the claim of architectural generality within the multi-view DR domain.

## Weaknesses

### Major:

- **The 50% token retention ratio undermines the "lesion proposal" framing and suggests the mechanism acts as a general saliency filter rather than a lesion localizer.** DR lesions typically occupy a very small fraction of the fundus image (often <5%). Retaining 50% of spatial tokens (Section 4.1, confirmed optimal in Fig. 3) means the "proposals" include substantial non-lesion content. The paper's narrative—that GALP identifies "lesion proposals" that "concentrate evidence on grade-relevant areas" and "reduce distraction from non-lesion background" (Section 3.2)—is weakened by this finding. The mechanism appears to function as a coarse saliency pruning step rather than a lesion-specific proposal generator. This matters because it affects interpretability claims and the precision of the cross-view fusion: if proposals include large amounts of background, the LGRF module may be attending to salient but non-lesion structures. The authors should either (a) explain why 50% is appropriate despite the small lesion footprint, (b) provide evidence that the selected regions correlate with actual lesion locations, or (c) recalibrate the framing from "lesion proposals" to "grade-salient region proposals."

- **The ablation study conflates the effects of auxiliary supervision and proposal selection, leaving the source of improvement ambiguous.** In Table 4, "w/o GALP" removes both the auxiliary classification loss and the Top-K proposal selection, replacing them with direct use of all tokens. This makes it impossible to determine whether the performance gain comes from (a) the stronger feature supervision via the auxiliary loss, or (b) the spatial focusing of the Top-K selection, or (c) both equally. A cleaner ablation would include "GALP without proposal selection" (auxiliary loss retained, all tokens passed to LGRF) and "proposal selection without auxiliary loss" (Top-K on raw features without the auxiliary classifier). Without this disentanglement, it is unclear whether the complex proposal mechanism is necessary or whether auxiliary supervision alone would suffice—a significant gap for a paper whose central contribution is the proposal mechanism.

### Minor:

- **No computational efficiency analysis is provided despite significant architectural overhead.** The method adds 4 stage-wise auxiliary classifiers, CAM computation, Top-K selection, an MoE routing network with 6 experts, and cross-view attention at each stage. The conclusion claims the method provides a "scalable solution for large-scale DR screening" without reporting FLOPs, parameter counts, or inference latency versus baselines (e.g., MVCINN or Swin-B). Given that clinical screening requires fast turnaround, this omission limits practical assessment.

- **No qualitative validation that self-derived proposals correspond to actual lesion locations.** The paper claims GALP "derives grade-conditioned evidence maps" and that selected regions are "more likely to contain lesion evidence" (Section 3.2), but provides no visualization comparing proposals against ground-truth lesion masks (which are available in MFIDDR). Without this, it is difficult to verify whether the proposals capture clinically meaningful structures or are statistical artifacts of the auxiliary classifier. This is especially important given the 50% retention ratio concern above.

- **The bootstrap dynamics of the proposal mechanism during early training are unaddressed.** GALP generates proposals from the auxiliary classifier's predictions (Eq. 3), but in early training these predictions are essentially random, meaning GEMs will highlight incorrect regions. The paper does not discuss how the model stabilizes—whether the main loss gradient corrects proposals quickly enough, or whether noisy early proposals significantly slow convergence. A brief discussion or convergence analysis would strengthen confidence in the training procedure.

### Trivial:

- **Gradient flow through the hard Top-K selection step is not explicitly discussed.** The Top-K region selection (Eq. 4-7) involves hard indexing, but the masked average pooling (Eq. 5) and linear projection (Eq. 6) are differentiable with respect to the features. Gradients flow primarily through the auxiliary loss to the features, not through the selection itself. While this is a common pattern (similar to DETR's hard assignment), a brief clarification would be helpful.

## Nice-to-Haves

- **Cross-dataset generalization test** (train on MFIDDR, test on DRTiD or vice versa) to assess robustness to domain shift across imaging devices and populations.

- **Expert specialization analysis:** Whether the M=6 experts in LGRF learn meaningfully distinct lesion morphologies or degenerate into redundant feature extractors. If experts are redundant, the MoE complexity may be unjustified.

- **View dropout robustness test:** Evaluate performance when one or more views are artificially masked during inference, which is clinically relevant when certain views are unavailable.

- **Clinical screening metrics:** Sensitivity at fixed specificity thresholds (e.g., 95% specificity) would better align with deployment requirements than accuracy alone.

- **Lower retention ratios explored more granularly:** Testing α ∈ {0.05, 0.10, 0.20, 0.30, 0.50} would clarify the lesion-localization vs. saliency-filtering behavior.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Missing statistical significance / error bars:** All three reviewers requested standard deviations over multiple seeds. However, single-run reporting is standard practice in the medical imaging and multi-view DR grading community (all baselines in Tables 1-3 also report single values). Demanding confidence intervals for large-scale benchmarks where single-run evaluation is the norm falls under methodological practices not standard in this field. Moved to nice-to-have consideration but not a core flaw.

- **Baseline fairness / different backbone pretraining:** The spark finder questioned different pretraining strategies across datasets. The paper explicitly follows each baseline's own protocol (ImageNet for MFIDDR matching CVSA, EyePACS for DRTiD matching CrossFiT), which is the correct approach for fair comparison. This is not a weakness.

- **"Ours (with lesion)" contradicting the core contribution:** The spark finder suggested the with-lesion variant contradicts the paper's message. However, the paper clearly demarcates "Ours (w/o lesion)" as the primary result and presents the with-lesion variant as demonstrating that the architecture can additionally benefit from external information when available. This is a feature, not a contradiction.

- **Garbled equation in load balancing loss (Eq. 11):** This is a PDF parser artifact, not an author error. Removed as a formatting nitpick per hard rules.

- **Equation 6 appearing abruptly / structural confusion:** Formatting/ordering issue attributable to parser artifacts. Removed per hard rules on formatting nitpicks.

- **Method not generalizable beyond DR:** Criticizing a DR grading paper for not demonstrating transfer to OCT or other modalities is scope creep. The paper's stated scope is multi-view DR grading; evaluating it on that scope is appropriate.

- **MFIDDR segmentation masks are model-generated, not human-annotated:** This concern applies to the "with lesion" variant only, which is a secondary result. The core contribution (w/o lesion variant) is unaffected.

- **Per-grade performance analysis missing:** Factually incorrect—the paper provides detailed per-grade F1, Precision, and Specificity in Table 2 for all 5 DR grades.

## Novel Insights

The most revealing finding is the tension between the paper's "lesion proposal" narrative and the 50% optimal retention ratio. This suggests that what GALP actually provides is not lesion localization per se, but rather a form of learned spatial prior that coarsely separates diagnostically relevant retinal regions from irrelevant periphery. The fact that this coarse prior, when coupled with cross-view expert gating, still matches externally-informed methods is itself an interesting result: it implies that much of the benefit of external lesion/vessel annotations in prior work may come from simply directing attention away from background rather than from precise lesion delineation. If validated (e.g., by showing that even random spatial priors within the retinal area perform similarly), this could reshape how the community thinks about the role of auxiliary annotations in medical image analysis.

## Suggestions

- **Rename "lesion proposals" to "grade-salient proposals" or "evidence proposals"** throughout the paper, or provide quantitative IoU validation against MFIDDR's lesion masks to justify the current terminology. This would align the framing with the empirical finding (50% retention) and strengthen credibility.

- **Add a two-row ablation** isolating auxiliary supervision from proposal selection: (1) auxiliary loss only (all tokens to LGRF), and (2) Top-K selection only (no auxiliary loss, proposals from raw feature norms). This directly addresses the major methodological gap in the current ablation.

- **Provide 2-3 visual examples** of GEM overlays alongside lesion segmentation masks on MFIDDR to qualitatively demonstrate what the proposals capture. This is low-cost and would significantly strengthen the interpretability claims.

- **Include a computational cost table** (FLOPs, parameters, inference time) comparing against MVCINN and Swin-B baselines, given the "scalable solution" claim in the conclusion.

---

## ey7CXUBn1g

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary
AdaSVD proposes two complementary techniques for improving SVD-based LLM compression: **adaComp**, which compensates for truncation error via alternating Moore-Penrose pseudoinverse updates of the truncated singular matrices using calibration data, and **adaCR**, which assigns layer-specific compression ratios based on input-output cosine similarity. The method consistently outperforms prior SVD-based approaches (particularly SVD-LLM) across multiple LLM families and compression ratios, with the largest gains at high compression (60%+).

## Strengths
- **Principled truncation error compensation**: The adaComp mechanism (Eqs. 6–13) formulates a concrete optimization objective (∥ŨV̂⊤X − WX∥²_F) and solves it via stabilized alternating least squares using Moore-Penrose pseudoinverse, with empirical evidence (Figure 3a) showing smooth, monotonic error reduction compared to naive matrix inversion. This directly addresses a real gap—prior SVD methods perform truncation but do not adjust the retained factors.
- **Consistent and substantial improvements at high compression**: At 60% compression on LLaMA2-7B, AdaSVD achieves 50.33 PPL vs. SVD-LLM's 89.90 on WikiText-2 (Table 1), a 44% relative reduction. The gap widens precisely where deployment matters most (resource-constrained settings), which is where prior methods degrade catastrophically.
- **Demonstrated orthogonality with quantization**: Table 4 shows AdaSVD + GPTQ-INT4 outperforms SVD-LLM + GPTQ-INT4 across all compression ratios, and at 60–80% compression, AdaSVD+GPTQ even outperforms SVD-LLM without quantization. This is practically valuable for composing compression pipelines.
- **Stack-of-batch strategy**: A practical engineering contribution that enables use of more calibration data under fixed GPU memory (Figure 3b shows consistent error reduction), addressing a real implementation constraint.

## Weaknesses

### Major:
- **No inference efficiency measurements despite deployment claims**: The introduction explicitly claims SVD "can effectively accelerate model inference by reducing the memory requirements" (Page 2). Yet no latency, throughput, peak memory, or wall-clock inference measurements appear anywhere in the experiments. For a paper positioning itself around deployment on resource-constrained devices, this is a significant gap. SVD factorization replaces one large matmul with two smaller ones, which can *increase* latency on GPUs due to kernel launch overhead and reduced tensor core utilization. The paper must substantiate or retract the acceleration claim with actual benchmarks.

### Minor:
- **Potential mathematical issue in adaCR formula (Eq. 19)**: The retention ratio CR(W) = mrr + In(W)·(trr − mrr) can exceed 1.0 for layers with high normalized importance. Figure 4 shows max/min importance ratios up to 9.49 for Llama-7B, meaning In(W) could be well above 1 for the most important layers. For instance, with mrr=0.4, trr=0.6, and In(W)=5, the formula yields CR=1.4—an impossible retention ratio. The paper does not mention clipping or how this is handled. If clipping is applied, it changes the effective global compression ratio, violating the "fixed target compression ratio" guarantee; if not, the formula is invalid for extreme importance values. This needs explicit clarification.
- **Counter-intuitive importance metric with weak justification**: adaCR defines layer importance as cosine similarity between input X and output WX (Eq. 17), assigning higher retention to layers that preserve their input. However, a near-identity layer (high similarity) could be argued to be doing *less* transformative work and thus more redundant—contradicting some pruning literature that identifies highly transformative layers as critical. The paper provides no theoretical or empirical analysis justifying why high similarity implies high importance. While the method works empirically (Figure 4 shows the first layer is important, which is plausible), the metric's logic needs stronger defense.
- **Stack-of-batch is not equivalent to using more data**: Averaging calibration samples into buckets (Eq. 15) produces X'_k = mean of a batch, which yields a different second-moment structure than using all samples individually. The resulting covariance estimate discards within-batch variance. The paper claims this enables "more calibration data," but it actually uses a *summary* of the data that acts as implicit regularization. This distinction should be acknowledged rather than presenting the strategies as equivalent.
- **VLM evaluation is purely qualitative**: Figure 5 shows captioning examples but provides no quantitative metrics (CIDEr, BLEU, ROUGE) on the COCO dataset. The claim of "better image captioning results" is unsupported by standard evaluation.

### Trivial:
- **Compression time not reported**: adaComp requires iterative alternating updates (Table 3c tests up to 15 iterations). The wall-clock time for the compression phase itself is never reported, making it unclear how practical the method is relative to a single-pass SVD.

## Nice-to-Haves
- Inference latency/throughput benchmarks on target hardware (even a single GPU), which would substantiate the deployment narrative.
- Evaluation on at least one model at 13B+ scale to demonstrate scalability of adaCR's importance metric and adaComp's convergence behavior.
- Convergence analysis (theoretical or empirical loss curves) for the alternating update scheme to characterize overfitting risk with increasing iterations.
- Comparison with non-SVD compression methods (e.g., AWQ, GPTQ alone) at equivalent bit-budgets to contextualize where SVD-based methods stand in the broader compression landscape.
- Fine-tuning recovery experiments to assess whether AdaSVD-compressed models can regain performance with minimal additional training.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Weakness: "Comparison with non-SVD compression at equivalent bit-rates"** (from Spark Finder) — The paper's stated scope is improving SVD-based compression methods. Demanding comparison with quantization or pruning methods is scope creep. The paper already shows orthogonality with GPTQ (Table 4).
- **Weakness: "Ablation isolating adaComp vs. adaCR contribution is missing"** (from Spark Finder) — Factually incorrect. Table 3a shows AdaSVD with adaComp alone (adaCR=✗), and Table 3b shows the incremental benefit of adding adaCR. The individual contributions are explicitly isolated.
- **Weakness: "Missing related works"** — Not verifiable without external sources; removed per hard rules.
- **Weakness: "Formatting artifacts / notation inconsistency"** — Per instructions, parser artifacts are not paper problems. Minor notation issues are trivial and removed.
- **Weakness: "Low-rank weight compensation claim is too strong given LoRA"** (from Harsh Critic) — The claim is explicitly scoped to "SVD-based LLM compression" context; LoRA addresses a different problem (adaptation, not post-training compression). The claim is reasonable within its scope.
- **Weakness: "Missing Limitations section"** — Formatting/style requirement, not a substantive weakness.
- **Weakness: "Reproducibility concerns about calibration data"** — Per hard rules, reproducibility nitpicks about implementation details are removed.
- **Strength: "The paper is well-written / topic is important / experiments are extensive"** — Generic strengths weakened per soft rules.

## Novel Insights
The most interesting observation across the reviews is the tension in adaCR's importance metric: empirically it works (first layers are rated important, which aligns with intuition), yet theoretically the logic is inverted relative to some pruning perspectives (high input-output similarity could indicate redundancy rather than importance). This suggests the metric may be capturing something different from "transformative importance"—perhaps it measures *information preservation criticality*, where layers that faithfully pass information through the residual stream are more fragile to compression because any error propagates unattenuated. Disentangling these two interpretations could lead to a more principled importance metric and is a fruitful direction for future work.

## Suggestions
- Add a small table (even 3-4 rows) reporting inference latency and peak GPU memory for the original model vs. SVD-LLM vs. AdaSVD at matched compression ratios on a single GPU. This directly addresses the deployment narrative.
- Explicitly address the Eq. 19 overflow issue: either show that In(W) values stay within bounds for all tested models, add a clipping mechanism with analysis of its impact on the effective global compression ratio, or reformulate the equation.
- Provide quantitative VLM results (CIDEr or BLEU scores on COCO) instead of or in addition to the qualitative Figure 5.
- Report wall-clock compression time for adaComp with different iteration counts to help practitioners choose k.

---

## D5PJX02Jki

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary
The paper proposes RoPE++, which re-incorporates the imaginary component of the complex-valued attention score that standard RoPE discards, creating a dual-component attention with real and imaginary heads. Two configurations are introduced: RoPE++EH (equal head count, halved KV cache/QKV parameters) and RoPE++EC (equal cache size, doubled heads). Theoretical analysis via characteristic curves (sine integral for imaginary vs. cosine integral for real) motivates the long-context bias of imaginary attention, and experiments at 376M–1.5B scales show improvements on long-context benchmarks such as RULER and BABILong.

## Strengths
- **Novel identification of overlooked information in a ubiquitous mechanism.** The observation that standard RoPE discards the imaginary component of the complex dot product—and that this component carries distinctive positional properties—is a genuine insight. The derivation showing imaginary attention follows a sine-integral characteristic curve (Eq. 5) that decays more slowly than the cosine-integral of real attention (Eq. 7) provides concrete mathematical grounding for why imaginary attention biases toward long-range dependencies.
- **Practical dual-variant design with honest trade-off articulation.** The EH/EC framing maps cleanly to real deployment constraints (memory-constrained vs. compute-constrained). The efficiency experiments (Figure 4, Table 11) show RoPE++EH delivers genuine memory savings and decoding speedups, while the paper is forthright about RoPE++EC's compute overhead.
- **Mechanistic validation beyond aggregate metrics.** The noise sensitivity experiment (Section 5.2, Figure 5e/5j) directly probes whether imaginary heads are functionally important for long-context reasoning. Adding Gaussian noise to imaginary attention degrades RULER-4k scores by ~8 points more than equivalent noise to real attention at σ=1.0, providing evidence that imaginary heads are not merely redundant capacity.

## Weaknesses

### Major:

- **Abstract overclaims consistency of improvements, particularly for RoPE++EH.** The Abstract states the method "consistently improves performance over the standard RoPE" and that "Both RoPE++EH and RoPE++EC outperform vanilla RoPE." However, the data does not uniformly support this for EH. In Table 2 (376M Long), RoPE++EH averages 18.2 on RULER vs. RoPE's 18.8. In Table 6 (1.5B Long), RoPE++EH averages 31.0 vs. RoPE's 35.1 on RULER. On BABILong at 1.5B, EH averages 32.9 vs. RoPE's 29.5—so it wins there but loses on RULER. The EH variant is a memory-efficiency trade-off that sometimes sacrifices accuracy; presenting it as a consistent improvement misrepresents the results. The EC variant's improvements are more robust, and the claims should be calibrated accordingly.

- **Confound between the imaginary attention mechanism and capacity/architecture changes.** RoPE++EC doubles the number of attention heads under a fixed cache budget. The performance gains could partially reflect increased representational capacity from having more heads rather than specifically from the imaginary attention mechanism. The paper notes that real and imaginary heads share W_q (Section 3.3), which partially addresses this, but the output projection W_o is doubled for EC, and the attention computation itself is doubled. An ideal control would compare RoPE++EC against a vanilla RoPE baseline that also doubles heads (with independent parameters) at the same cache cost. Without this, it is unclear whether the gains are attributable to the mathematical properties of imaginary attention or simply to having more attention heads. For RoPE++EH, the reverse confound exists: head dimension is halved, and the paper does not explain why performance is maintained despite reduced per-head capacity, making it unclear whether the imaginary mechanism compensates or whether the comparison is confounded by the architecture change.

- **Maximum model scale is 1.5B, which is modest for architectural claims targeting long-context LLMs.** All experiments are at 376M, 776M, and 1.5B. Production long-context models operate at 7B+ scale. The paper acknowledges resource constraints, and the Appendix C.1 scaling analysis is appreciated, but at 1.5B the absolute gains on long-context benchmarks are sometimes small (e.g., RULER 64k: RoPE++EC 18.9 vs. RoPE 18.3 at 1.5B). Whether these gains amplify or diminish at larger scales remains unknown, which limits confidence in the practical significance of the contribution.

### Minor:

- **RoPE++EC incurs substantial training compute overhead.** Table 11 shows TGS drops from 53,317 to 37,248 (30% reduction) for the 376M model at 32k context. The paper argues long-context inference is IO-bounded, but the training cost increase is not trivial and is insufficiently discussed relative to the performance gains.

- **Noise sensitivity measures sensitivity, not necessarily functional dominance.** The experiment in Section 5.2 shows that corrupting imaginary attention degrades long-context performance more than corrupting real attention. This demonstrates that imaginary attention is more *sensitive* to perturbation on these tasks, which is suggestive but not definitive evidence of "dominance." A cleaner ablation—e.g., zeroing out imaginary attention heads entirely at inference time—would more directly establish their functional contribution.

- **The extrapolation analysis in Section 3.4, while directionally correct, could be more precise.** The claim that RoPE++ allows dimensions to observe the "full cos and sin value range" once training length "exceeds half the sinusoidal period" (whereas vanilla RoPE requires a "full period") is stated without a formal derivation. The informal argument is plausible—since imaginary attention applies -cos where real applies cos, both signs are seen—but the quantitative impact on extrapolation error is not bounded, making the theoretical contribution here weaker than in Section 3.2.

### Trivial:

- The exact softmax scope for real vs. imaginary heads could be stated more explicitly (though Figure 2 and the "separate heads" language in Section 3.3 make it clear they are independently softmaxed as separate attention heads).

## Nice-to-Haves
- Evaluation at 7B+ scale, or at minimum a scaling law analysis that extrapolates expected gains.
- A needle-in-haystack retrieval heatmap broken down by token position (early/middle/late) to directly visualize the claimed long-range attention bias of imaginary heads.
- An equal-capacity control for RoPE++EC (vanilla RoPE with doubled heads and independent parameters at the same cache cost) to disentangle the imaginary mechanism from head-count effects.
- A masking ablation (zeroing imaginary heads at inference) to complement the noise sensitivity experiment.
- Analysis of whether RoPE++ can be adapted to pre-trained models via partial fine-tuning (e.g., LoRA on W_o), reducing the deployment friction acknowledged in the Limitations.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Weakness: Missing comparisons with LongRoPE, RoPE-NTK variants, Hybrid Attention.** Per hard rules, I cannot confirm these methods exist or are appropriate baselines beyond what the paper cites. The paper already compares against FoPE, ALiBi, Pythia, and vanilla RoPE, plus tests compatibility with YaRN and Linear PI. (From Spark Finder)
- **Weakness: Reproducibility concerns about checkpoint/code availability.** The paper states code is available at a GitHub URL and the reproducibility statement commits to releasing checkpoints. Per hard rules, cited availability is assumed real. (From Spark Finder)
- **Weakness: Numerical stability of complex arithmetic in FP16.** This is speculative—no training instability is reported, and the implementation uses FlashAttention interleaving with no reported issues. (From Positive Reviewer)
- **Weakness: Ablation of imaginary/real ratio (e.g., 75/25 split).** The paper explicitly addresses this in Section 3.3: "configurations such as 75% imaginary vs. 25% real or 100% imaginary are impossible under RoPE++" because imaginary attention is defined relative to real attention and cannot exist independently. This is a design constraint, not a missing experiment. (From Spark Finder)
- **Weakness: Missing real-world long-context tasks (summarization, multi-step reasoning).** The paper evaluates on RULER and BABILong, which are standard long-context benchmarks. While real-world tasks would be valuable, requesting them is scope creep beyond the paper's focus on position embedding design. Moved to Nice-to-Have. (From Harsh Critic/Spark Finder)
- **Weakness: Retraining requirement as a deployment barrier.** The paper explicitly acknowledges this in the Limitations section and scopes itself as a pretraining method. Criticizing a paper for not being something it explicitly scopes out is scope creep. (From Positive Reviewer)

## Novel Insights
The characteristic curve analysis reveals a striking structural asymmetry: real attention's cosine-integral curve decays monotonically from distance zero, while imaginary attention's sine-integral curve *rises* before slowly declining. This means imaginary attention is, by construction, *suppressed* at very short distances and *amplified* at moderate-to-long distances relative to real attention. This is not merely "extra information"—it is information with an opposite inductive bias. The practical implication is that RoPE++ effectively gives the model a built-in short-range/long-range attention specialization at no additional positional parameter cost, which is conceptually cleaner than post-hoc head-type classification methods. However, the entanglement of this mechanism with the capacity changes in both EH and EC variants means the clean attribution of gains to this inductive bias remains an open question.

## Suggestions
- Calibrate the Abstract and Introduction to distinguish EC (robust improvement) from EH (memory-accuracy trade-off that sometimes underperforms). A single sentence noting EH's trade-off nature would suffice.
- Add an equal-capacity control experiment: train a vanilla RoPE model with the same number of heads as RoPE++EC (using GQA to match cache), keeping total parameters comparable. This would cleanly isolate whether gains come from the imaginary mechanism or from added head capacity.
- Report RoPE++EH results with a vanilla RoPE baseline that also halves head dimension (without imaginary extension) to demonstrate that EH's competitive performance is specifically due to the imaginary component compensating for reduced capacity, not an artifact of the architecture change.

---

## Mz98kwANpF

- GT: Reject (avg 4.5)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## SummaryThis paper challenges the prevailing "diversity-first" paradigm in multi-task LoRA, showing that a simplified multi-head architecture (M-LoRA) with high inter-head similarity outperforms complex diversity-seeking variants like R-LoRA, and that a standard single-adapter LoRA with increased rank matches multi-component architectures. Based on these findings, the authors hypothesize that learning task-shared representations is more effective than architectural isolation, and propose Align-LoRA, which adds an explicit alignment loss (KL divergence or MK-MMD) on the shared down-projection matrix **A** to encourage task representations to converge in the low-rank space.

## Strengths

- **Provocative empirical challenge to the prevailing paradigm.** The finding that M-LoRA (router removed, heads summed) outperforms R-LoRA despite exhibiting much higher inter-head similarity (medians >0.85 vs. R-LoRA's low similarity in Figure 2) directly contradicts the core design principle of diversity-focused methods, making this a substantive conceptual contribution.

- **Zero inference latency through weight merging.** Unlike multi-component architectures with dynamic routing that cannot be pre-merged, Align-LoRA retains LoRA's key practical advantage — the trained adapter can be merged into the base model (Appendix C, Eq. 7), eliminating inference overhead entirely. This is a significant deployment advantage that the multi-component literature has largely accepted as a necessary trade-off.

- **Clear narrative progression from observation to method.** The paper moves systematically: (1) M-LoRA's paradoxical success → (2) high-rank single LoRA matches multi-component → (3) hypothesis that shared knowledge matters more than isolation → (4) Align-LoRA as a principled operationalization. This makes the motivation for Align-LoRA well-grounded rather than arbitrary.

- **Consistent empirical improvements with fewer parameters.** In Table 4, A-LoRA-K achieves the best BBH scores across all three base models while using fewer trainable parameters (0.20%) than HydraLoRA and R-LoRA (0.25%). Table 5 shows per-task wins on 7 of 8 tasks for Qwen2.5-7B.

## Weaknesses

### Major:

- **No ablation against a simple auxiliary regularization baseline.** The paper attributes performance gains specifically to "representation alignment," but never tests whether adding *any* auxiliary loss on the **A** matrix (e.g., L2 regularization, variance minimization) would produce similar improvements. Without this control, it is impossible to determine whether the alignment mechanism itself — as opposed to the regularizing effect of an additional training objective — drives the observed gains. This is critical because the core claim of the paper is that *alignment* specifically (not generic regularization) is the key to better multi-task LoRA.

- **Missing statistical significance reporting.** Tables 1, 4, and 5 report single-run results with no error bars or standard deviations across seeds. Some of the claimed improvements are modest (e.g., Table 1: M-LoRA at 82.52 vs. R-LoRA at 82.03 on QNLI; Table 4: A-LoRA-K at 48.84 vs. M-LoRA at 45.35 on LLaMA3-8B — though this is a larger gap). For a paper whose central contribution rests on empirical findings that challenge a paradigm, the absence of variance estimates weakens confidence that these differences are reproducible and not artifacts of run-to-run variation.

- **Inference from weight similarity to "shared knowledge" is not directly verified.** Figure 2 measures cosine similarity between flattened **B**_i weight vectors, not between the *representations* they produce. The paper infers that high weight similarity implies the heads learn "shared knowledge" (Section 3.3), but this causal chain is incomplete: weights with high cosine similarity can still produce divergent outputs depending on input distributions. A direct measurement of representation-level similarity (e.g., CKA or mutual information between head outputs on shared inputs) would substantially strengthen the claim that the heads are genuinely learning shared representations rather than merely converging in parameter space.

### Minor:

- **Strong distributional assumption without justification.** The alignment loss models batch-wise representations as multivariate Gaussians with *diagonal* covariance (Section 5.1). For LLM hidden states, which exhibit strong feature correlations, the diagonal assumption may be a poor fit. The paper provides no empirical or theoretical justification for why full covariance is unnecessary or why the diagonal approximation suffices.

- **The theoretical contribution is incremental.** The generalization bound in Appendix F follows standard domain adaptation theory (Ben-David et al., 2006) applied to the LoRA MTL setting. The derivation does not leverage any specific properties of low-rank decomposition (e.g., the relationship between rank _r_ and the bound's tightness), making the theory generic rather than specifically informative about why Align-LoRA works.

- **No evaluation on genuinely conflicting task pairs.** Appendix H.2 tests "highly dissimilar" tasks (e.g., Translation vs. QA), but dissimilarity is not the same as conflict. Tasks with opposing optimal representations (e.g., sentiment on different domains where class distributions flip) could cause alignment to induce negative transfer. The paper does not address this failure mode, which limits the claimed generality of the "shared representations are superior" hypothesis.

### Trivial:

- The claim in the abstract that Align-LoRA "significantly surpasses baselines" could be more precisely calibrated given the modesty of some improvements (e.g., +1.49 average over M-LoRA on Qwen2.5-7B in Table 5).

## Nice-to-Haves

- Wall-clock inference latency measurements comparing merged Align-LoRA vs. unmerged multi-component methods under realistic batch sizes and sequence lengths, to quantitatively substantiate the practical deployment advantage.
- Layer-wise analysis identifying which transformer layers benefit most from alignment; if only certain layers matter, the method could be made even more efficient.
- A deeper mechanistic analysis (e.g., gradient trajectory or norm comparison) of why summation-based aggregation outperforms dynamic routing, beyond the intuitive "collaborative vs. competitive" framing.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Abstract should be more precise about alignment mechanism"** — Formatting/style nitpick.
- **"Brief acknowledgment of why merging fails for dynamic routers"** — The paper already explains this: input-dependent routing weights prevent pre-merging (Section 2.2, Eq. 8; Appendix C).
- **"Transition from M-LoRA to single-adapter is abrupt"** — Writing style preference, not a substantive weakness.
- **"Discuss cross-stitch/sluice networks in Related Work"** — Missing related works request; cannot verify existence per rules.
- **"Table 3 evaluation benchmark unclear"** — Section 4 text explicitly states evaluation on BBH; reviewer misread.
- **"Method less effective on smaller models"** — On Qwen2.5-3B, A-LoRA-K gains +1.55 over M-LoRA; on 7B, +1.49. The gaps are comparable; the claim is factually incorrect.
- **"Training compute details missing (gradient accumulation, batch size, devices)"** — Nitpick about trivial implementation details.
- **"Compare against non-LoRA MTL methods"** — Missing related works concern; scope creep beyond the paper's stated focus on LoRA.
- **"Per-task performance breakdown missing"** — Table 5 already provides per-task scores for all 8 tasks; reviewer missed this.
- **"t-SNE visualization missing/corrupted"** — Figure 5 in Appendix I.1 provides t-SNE visualizations; this is factually wrong.
- **"Larger model scale (70B+) validation"** — Generic "add more scale" weakness; the current model zoo (3B–14B) is adequate.
- **"Parameter count bias in M-LoRA comparisons"** — Tables report %Param columns showing M-LoRA uses *fewer* parameters (0.41–0.42%) than HydraLoRA/R-LoRA (0.45%), so the concern is directionally wrong.
- **"Unfair comparison when single-adapter rank is increased"** — The asymmetry *favors* the baseline multi-component methods (they get their preferred architecture), making the comparison conservative for the author's method. Per hard rules, this is not a valid criticism.

## Novel Insights

The paper reveals an interesting paradox at the heart of multi-task LoRA design: the architectural mechanisms specifically introduced to promote diversity (randomized initialization, dropout, dynamic routing) may actually *interfere* with what helps most — the emergence of shared representations. The key insight is that removing the router while *keeping* dropout transforms multiple heads from competing specialists (where the router picks winners) into a collaborative ensemble (where dropout provides stochastic input perturbation and summation forces consensus). This suggests that the research community's focus on routing architecture design may have been optimizing the wrong variable — the important factor is not *which* expert handles which input, but whether the adapter space encourages tasks to share a common representation subspace. The finding that alignment on the **A** matrix (which prior work identifies as task-general) specifically helps, while prior work focused on diversifying **B** (task-specific), suggests the field may have been looking in the wrong place within the LoRA factorization itself.

## Suggestions

- **Add an L2 regularization ablation on the A matrix** as a control. Train standard LoRA with an auxiliary L2 loss on **A**'s output of comparable magnitude to the alignment loss. If Align-LoRA still wins, the alignment mechanism is specifically responsible; if not, the gains come from generic regularization. This single experiment would dramatically strengthen the paper's core claim.
- **Report results over 3 seeds with standard deviations** for at least the main comparison tables (Tables 4 and 5). Even partial variance reporting on one model family would address the statistical significance concern substantially.
- **Add a representation-level similarity metric** (e.g., CKA between head outputs on a shared input batch) alongside the weight-level cosine similarity in Figure 2, to directly verify that weight convergence implies representation convergence.

---
**Axis Evaluation:**
- **Novelty:** Moderate-high. The empirical findings are counter-intuitive and the application of alignment losses to LoRA's bottleneck for multi-task learning is new, though the individual loss functions (KL, MMD) are borrowed from domain adaptation.
- **Technical soundness:** Moderate. The method is clearly described and empirically validated, but the diagonal covariance assumption is unjustified, the theoretical contribution is incremental, and the missing regularization ablation leaves the core causal claim under-supported.
- **Empirical support:** Moderate. Results are consistent across models and scales, but the absence of error bars and the missing ablation against simple regularization are notable gaps for a paper that rests its case on empirical evidence.
- **Significance:** High. If the findings hold under the above scrutiny, this paper could redirect multi-task LoRA research away from complex routing architectures toward simpler, more deployable alignment-based methods.
- **Clarity:** Good. The paper presents a clear, well-structured narrative from observation to hypothesis to method, with thorough appendices.

---

## d2pUyiXwcm

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Final Review

## Summary

SCaSML introduces a physics-informed inference-time scaling framework that improves pre-trained PDE surrogates (PINNs, GPs) by deriving and solving a "Structural-preserving Law of Defect"—a new semi-linear PDE governing the surrogate's error—via Multilevel Picard (MLP) Monte Carlo iterations. The method provably achieves a convergence rate bounded by the product of the surrogate and simulation errors, and empirically reduces errors by 20–80% on semi-linear parabolic PDEs up to 160 dimensions.

## Strengths

- **Structural preservation enabling high-dimensional stochastic solvers:** The key insight—that subtracting the surrogate's approximate PDE from the original yields a defect PDE that *preserves the semi-linear structure and Lipschitz constants of the original* (Lemma D.11)—is non-trivial and precisely what makes MLP solvers applicable. Without this preservation, the dimension-independent convergence of MLP would not carry over. This is the theoretical linchpin of the paper and is rigorously proved.

- **Multiplicative error bound with practical implications:** Theorem 2.5 shows the final error scales as the *product* of the MLP simulation error and the surrogate error, not their sum. This means the correction step's cost *decreases* as the surrogate improves, yielding the improved scaling law of Corollary 2.6 (from $O(m^{-\gamma})$ to $O(m^{-\gamma-1/2})$). The empirical scaling law verification (Figure 4, Appendix G.3) with measured slope changes consistent with the theory adds credibility.

- **Stabilization of MLP in very high dimensions:** The most compelling empirical result is the LQG experiment (100–160D), where the naive MLP solver fails catastrophically (relative $L^2$ error > 5.0) while SCaSML successfully corrects the PINN to ~0.05–0.10 error. This demonstrates that the surrogate-as-control-variate effect is not merely incremental—it makes simulation feasible in regimes where it would otherwise diverge.

- **Comprehensive experimental validation with statistical rigor:** The inclusion of 10-run repeated experiments with paired t-tests (Appendix G.4, Tables 2–6, all $p \ll 0.001$), fixed-budget Pareto analyses (Appendix G.7–G.8), and inference-time scaling curves across multiple PDE families goes well beyond the typical single-run evaluation in SciML papers.

## Weaknesses

### Major:

- **Gap between Assumption 2.4 and practical PINN training:** The theoretical guarantees require both (1) the PDE residual $\epsilon$ to be uniformly bounded by $e(\hat{u})$, and (2) the $W^{1,\infty}$ defect norm to be bounded by $e(\hat{u})$. In the PINN literature, it is well-established that small PDE residuals do not guarantee small solution errors—PINNs can stagnate in local minima with low loss but poor accuracy, or exhibit spectral bias where low-frequency modes are well-approximated but high-frequency errors persist. While the $W^{1,\infty}$ condition partially addresses this (it bounds the defect itself, not just the residual), the paper does not verify that standard PINN training with Adam actually produces surrogates satisfying these assumptions in practice, nor does it characterize the failure regime where the surrogate is too poor for SCaSML to offer improvement. A surrogate quality ablation—deliberately degrading the surrogate and showing where correction stops being effective—would significantly strengthen the practical applicability of the theory.

- **Inference-time latency trade-off misrepresented in framing:** The abstract claims SCaSML "fuses the speed of machine learning with the rigor of numerical simulation," but Table 1 shows SCaSML is 5–60× slower than the pure surrogate at inference (e.g., LCD 10d: 0.45s vs. 6.77s; VB-GP 60d: 1.68s vs. 57.79s). The actual trade-off is *accuracy vs. latency*, not speed vs. accuracy. This is a fundamental characteristic of the method, not a flaw, but the framing should be corrected. The "elastic compute" language (Remark 2.2) is more accurate and should be the primary framing.

- **Impact of clipping on theoretical bounds is unanalyzed:** The clipping/thresholding in Algorithm 2 is essential for numerical stability, and the thresholds differ dramatically between methods (e.g., 0.01 for SCaSML vs. 10.0 for naive MLP in LQG). The theoretical error bounds assume Lipschitz continuity and do not account for the clipping. Since clipping is a hard nonlinearity that violates the Lipschitz assumption, there is a gap between theory and practice for the most challenging problems. The caption of Figure 12 acknowledges the clipping "trade-off" but no analysis of how different thresholds affect the bound is provided.

### Minor:

- **Heuristic budget allocation:** The 1/(d+1) split between training and inference compute (used in Appendix G.7) is presented without justification or optimality analysis. Whether this is near-optimal or far from it significantly affects the practical efficiency claims.

- **Global error evaluation cost:** The method corrects the surrogate pointwise, yet Table 1 reports global $L^2$, $L^\infty$, and $L^1$ errors over test sets of ~1200 points. The total inference cost scales linearly with the number of query points, which is not explicitly discussed and could be prohibitive for applications requiring dense spatial coverage.

## Nice-to-Haves

- **Direct variance reduction measurement:** The core mechanism is variance reduction via the surrogate acting as a control variate. A direct comparison of MLP estimator variance with and without the surrogate correction (not just the final error) would empirically validate the theoretical variance scaling argument from Section 2.1.

- **Verification that trained PINNs satisfy smoothness assumptions:** Assumption F.1 (Gevrey-class regularity) and the $W^{1,\infty}$ bounds are used in the proofs but never empirically verified. A simple check—e.g., measuring the defect's smoothness properties on a held-out set—would bridge theory and practice.

- **Failure mode characterization:** Explicitly demonstrating a case where the surrogate error is too large, causing the defect PDE solver to diverge or fail to improve, would establish clear applicability boundaries and help practitioners decide when to deploy SCaSML.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Sign convention inconsistency between Equations (1) and (6):** The harsh critic flagged a potential sign issue, but this appears to be a parser artifact. The original PDE (1) has $-\partial u/\partial r + Lu + F = 0$, equivalently $\partial u/\partial r + Lu + F = 0$ after rearrangement, and the residual in (6) consistently uses the same sign convention. No actual inconsistency exists.

- **Lipschitz constant degradation from surrogate gradients:** The harsh critic speculated that large surrogate gradients could degrade the effective Lipschitz constant of $\tilde{F}$, negating dimension-independent convergence. This is factually incorrect: Lemma D.11 explicitly proves that the modified nonlinearity $\tilde{F}$ inherits the *exact same* Lipschitz constant $L$ as $F$, because the surrogate-dependent terms cancel in the difference (proof: $F(\hat{U} + w_1) - F(\hat{U} + w_2)$ eliminates the surrogate background state $\hat{U}$).

- **Missing comparisons with FNO, DeepONet, or other neural operators:** The paper targets pointwise PDE solutions in 100–160 dimensions. Neural operators like FNO are designed for fixed low-dimensional grids (2–3D) and do not naturally scale to these dimensions. The comparison category is inappropriate for the problem setting.

- **Demanding confidence intervals for benchmark evaluations:** Single-run evaluation is the norm in the high-dimensional PDE literature; the paper already goes beyond this by including 10-run statistical tests in the appendix.

- **Missing related works on neural control variates:** Per rules, missing related work citations are not a valid weakness without external source confirmation.

- **Reproducibility concerns about hyperparameters:** The paper provides detailed hyperparameters (learning rates, optimizer settings, network architectures, sample sizes) in Sections 3.1–3.4.

## Novel Insights

The paper's most profound observation is not the defect correction itself—which is classical—but the specific way it enables a *type mismatch resolution* between machine learning and stochastic simulation: neural surrogates excel at learning smooth, low-frequency solution components (spectral bias) while Monte Carlo methods have convergence rates *independent of the integrand's smoothness*. By defining the defect PDE so that its source term $\epsilon$ is precisely the high-frequency residual the neural network struggled with, the framework channels each method's strength toward the component it handles best. This is more than a control variate; it is a structural decomposition of the problem by frequency, where the "variance reduction" is not just numerical but conceptual—the surrogate and simulator are solving different problems that happen to sum to the answer.

## Suggestions

- Re-frame the abstract and introduction to emphasize the accuracy-vs-latency trade-off rather than "speed + rigor," and position "elastic compute" as the primary practical contribution.
- Add a surrogate quality ablation (e.g., intentionally under-trained PINN with high error) to empirically map the boundary where SCaSML ceases to improve over the baseline, directly addressing the gap in Assumption 2.4.
- Include a brief analysis or discussion of how the clipping threshold interacts with the theoretical Lipschitz-based error bounds, even if only to bound the bias introduced by thresholding.

## Axis Evaluation

- **Novelty:** High. The specific combination of structural-preserving defect PDE derivation + MLP correction + inference-time scaling framing is novel. The structural preservation result (Lipschitz constants carry over exactly) is a clean theoretical contribution.

- **Technical soundness:** Moderate-to-high. The core theory is rigorous and the proofs are detailed. The main gap is the unverified assumptions about surrogate regularity and the unanalyzed impact of clipping on the bounds.

- **Empirical support:** High. Extensive experiments across multiple PDE families, dimensions, and surrogate types, with statistical tests and fixed-budget analyses. The LQG stabilization result is particularly convincing.

- **Significance:** High for the SciML community. The framework provides a principled and theoretically grounded way to add compute-time accuracy to neural PDE solvers, addressing a real need for trustworthiness in scientific applications.

- **Clarity:** Moderate. The paper is well-organized but suffers from occasional overclaiming (speed vs. accuracy framing) and heavy notation. The separation of linear warmup (Section 2.1) from the general case (2.2) is pedagogically effective.

---

## Vit5M0G5Gb

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Final Review

## Summary

This paper presents a theoretical framework explaining the dynamical simplicity bias in neural networks—where gradient descent learns solutions of increasing complexity over time—as a consequence of saddle-to-saddle dynamics. The authors prove the existence of a nested hierarchy of embedded fixed points (Theorem 1) and invariant manifolds (Theorem 3) for a broad class of architectures (fully-connected, convolutional, attention-based) defined by a unified layer equation. They then analyze the learning dynamics for two-layer linear and quadratic networks, identifying two distinct timescale separation mechanisms: data-induced separation (yielding low-rank weights) and initialization-induced separation (yielding sparse weights), and validate predictions about how width, data distribution, and initialization affect plateau dynamics.

## Strengths

- **Unified architectural treatment of fixed points and invariant manifolds.** Theorems 1 and 3, along with Corollary 2, apply to deep networks with FC, convolutional, and attention layers under a single framework (Equation 1). This is a genuine generalization of prior work (e.g., Fukumizu & Amari, 2000) that was restricted to fully-connected networks. The extension to new embedded fixed point constructions (Equations 6, 7) beyond the classical ones (Equations 4, 5) is both novel and essential, as the authors show the saddles visited during training fall under these new categories.

- **Disentanglement of two distinct timescale separation mechanisms.** The paper identifies that linear networks exhibit data-induced timescale separation (between directions across units, yielding low-rank weights) while quadratic/attention networks exhibit initialization-induced timescale separation (between units, yielding sparse weights). This distinction is genuinely novel—prior literature on saddle-to-saddle dynamics did not separate these mechanisms—and produces qualitatively different predictions (e.g., width scaling affects one but not the other, as shown in Figure 2A).

- **Predictive power validated through controlled experiments.** The framework makes specific, testable predictions: (i) increasing width shortens plateaus in self-attention but not linear FC nets (Figure 2A); (ii) flattening the data spectrum eliminates plateaus in linear nets but only shortens them in self-attention (Figure 2B); (iii) large low-rank initialization induces saddle-to-saddle dynamics without an initial plateau (Figure 2C). These are non-trivial predictions that go beyond post-hoc explanation.

## Weaknesses

### Major:

- **Gap between broad title/abstract claims and the scope of rigorous dynamical analysis.** The title promises an explanation "across neural network architectures," and the abstract discusses deep networks, but the core dynamical analysis (Section 5) is rigorously developed only for two-layer networks with homogeneous polynomial activations (linear and quadratic). The extension to deep networks is a conjecture (Section 7, "we conjecture that the order of the activation function...continues to predict learning behaviors"), and the paper explicitly acknowledges that general nonlinear activations like tanh do not satisfy the invariant manifold conditions needed for saddle-to-saddle dynamics (Section 7: "rank-one weights do not correspond to an invariant manifold with effective width one. Consequently, tanh networks are not guided to approach the saddle with one effective unit, and probably do not have saddle-to-saddle dynamics in general"). This excludes widely-used smooth activations (GELU, Swish, GLU variants) from the rigorous theory. While the authors are transparent about these limits in the body, the abstract and title do not reflect them—claiming the framework explains simplicity bias "across neural network architectures" without qualifying that the dynamical mechanism is proven only for a restricted class is misleading for a venue like ICLR where precision of claims matters.

- **The theory requires small initialization (ε→0), which is atypical in modern practice.** Both Theorem 4 and Proposition 5 rely on asymptotically small initialization for the timescale separation to emerge. Figure 2D shows that increasing initialization scale gradually weakens plateaus, and the paper acknowledges that "neural networks with large random initialization generally do not exhibit saddle-to-saddle dynamics." Standard initialization schemes (Xavier, He) are tuned for variance propagation, not for being small in the asymptotic sense. While the paper's contribution as a theoretical framework is clear, the practical relevance of the mechanism depends on whether the small-initialization regime is actually operative in real training pipelines. The paper could strengthen its case by discussing which practical settings (e.g., specific learning rate schedules, weight decay, or layer-wise initialization choices) might place training in a regime where this mechanism is relevant, even if approximately.

- **Empirical validation is limited to controlled synthetic settings and small-scale tasks.** The experiments use 2D synthetic data (Figures 1, 2, 4, 5) or binary MNIST classification with two-layer networks (Figure 3). While these are appropriate for validating theoretical predictions in controlled conditions, they leave open the question of whether saddle-to-saddle dynamics and the associated simplicity bias mechanism operate in the training of modern architectures at scale. The MNIST experiments (Figure 3) show the phenomenon persists with real data but with significant noise, and the paper does not demonstrate that the specific predictions (e.g., about width scaling or plateau duration) hold beyond the synthetic setting.

### Minor:

- **Theorem 3 establishes invariant sets, not attractors.** The theorem proves that if weights start on a manifold satisfying certain constraints, they remain there. It does not prove that gradient flow converges to these manifolds from generic initialization. The paper attempts to fill this gap via timescale separation arguments in Section 5, which is reasonable, but the distinction between "invariant" and "attracting" could be sharper. The text uses phrasing like "steers dynamics toward invariant manifolds" (Section 4), which implies attraction without proof.

- **The softmax self-attention experiment (Figure 4A) lacks theoretical backing.** The paper's dynamical theory in Section 5.2 covers linear (quadratic) attention, and the framework's invariant manifold conditions rely on homogeneity. Softmax breaks this homogeneity. Figure 4A shows stage-like dynamics in a softmax attention model, but this is presented without analysis of why or whether the same mechanism applies. This is a notable gap given that softmax attention is the dominant architecture.

### Trivial:

- The dimensionality of $v_i$ in Equation (1) changes between scalar (FC layers) and vector/matrix (attention), which requires careful cross-referencing but does not affect correctness given the consistent notation in Appendix D.

## Nice-to-Haves

- Analysis of how batch normalization or layer normalization interacts with the invariant manifold structure, since these layers break weight homogeneity required for Theorem 3(iii)-(iv).

- A discussion connecting the effective-width notion of simplicity to other common definitions (e.g., Kolmogorov complexity, description length), since the simplicity bias literature invokes multiple notions.

- Experiments with stochastic gradient descent (rather than full-batch gradient flow) to demonstrate that the plateaus and saddle-to-saddle transitions survive optimization noise, which is more relevant to practice.

- A quantification of the minimum singular value gap required for observable plateaus in the linear case, which would delineate the theory's domain of applicability for real data spectra.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Demand for CIFAR-10/ImageNet experiments with deep ResNets/ViTs.** This is scope creep. The paper is a theory paper with controlled experiments designed to validate specific theoretical predictions. Demanding large-scale benchmarks shifts the contribution type entirely.

- **Missing comparison against NTK spectral bias baselines.** The paper explicitly discusses the difference from NTK/kernel regime in Appendix A.2, noting that NTK dynamics exhibit smooth exponential decay rather than plateaus. A formal experimental comparison is unnecessary given this clear theoretical distinction.

- **Error bars / statistical significance for Figure 2.** For controlled synthetic experiments validating asymptotic theoretical predictions, single-run demonstrations are standard. The concern about stochasticity is more relevant for noisy real-data settings.

- **Broader impact statement.** Not required for ICLR and outside the paper's scope.

- **Missing related work citations (e.g., Chen et al. 2023 on symmetry-induced saddles).** Per the hard rules, I cannot confirm the existence of specific uncited works and should not flag missing related work.

- **Formatting/notation density complaints.** Per hard rules, formatting nitpicks are removed.

- **Reproducibility concerns about undisclosed hyperparameters.** Per hard rules, trivial implementation details are removed. The paper provides explicit hyperparameters in Appendix I.

- **Gradient flow vs. discrete SGD as a "limitation."** The paper explicitly states it analyzes gradient flow and this is standard practice in the theoretical deep learning literature. Flagging this as a weakness rather than a known modeling choice would be applying standards not standard in the field.

## Novel Insights

The disentanglement of data-induced versus initialization-induced timescale separation is the paper's most insightful contribution. It reveals that the *source* of the timescale separation—whether it arises from the data spectrum (producing distributed, low-rank representations) or from the randomness of initialization (producing sparse, localized representations)—fundamentally determines the *type* of simplicity bias a network exhibits. This has a concrete architectural implication: linear self-attention, being quadratic in the weights, inherits initialization-induced separation and thus sparse features, while linear fully-connected networks inherit data-induced separation and thus low-rank features. This predicts that scaling width should accelerate learning in attention architectures but not in linear FC networks—a non-obvious architectural distinction with potential practical consequences. The observation that large low-rank initialization can produce saddle-to-saddle dynamics *without* an initial plateau (Figure 2C) is also novel and nuances the common view equating exponential loss curves with lazy learning.

## Suggestions

- Qualify the title and abstract to reflect that the *dynamical mechanism* is rigorously established for two-layer networks with homogeneous/linear activations, while the fixed point and invariant manifold results apply more broadly. For example, "Saddle-to-Saddle Dynamics Explains a Simplicity Bias Across Neural Network Architectures: A Framework with Proofs for Two-Layer Homogeneous Networks."

- Add a brief table or paragraph categorizing common activations (ReLU, LeakyReLU, GELU, Swish, tanh, quadratic) by which conditions in Theorems 1 and 3 they satisfy, so practitioners can immediately assess applicability.

- Discuss explicitly which practical training settings (e.g., small learning rate with weight decay driving effective weights toward zero, or specific layer-wise initialization schemes) might place modern training in a regime where the small-initialization approximation is relevant, even if approximately.

- For the softmax attention experiment (Figure 4A), either add a brief analysis of why stage-like dynamics might persist despite broken homogeneity (e.g., near-zero weights make softmax approximately linear), or explicitly frame it as an empirical observation beyond the current theory's scope.

---

**Axis Evaluations:**

- **Novelty:** Strong. The unified treatment of embedded fixed points and invariant manifolds across FC/conv/attention architectures, the new fixed point constructions (Equations 6, 7), and the disentanglement of data- vs. initialization-induced timescale separation are all genuine contributions beyond prior work.

- **Technical soundness:** Good within its stated assumptions. Theorems 1 and 3 are rigorously proven for the general architecture class; the dynamical analysis is sound for two-layer homogeneous networks. The gap is that the most interesting claims (deep networks, general nonlinearities) rest on conjecture and empirical observation rather than proof.

- **Empirical support:** Adequate for a theory paper. The controlled experiments in Section 6 validate specific, non-trivial predictions. However, the gap between synthetic 2D experiments and practical training regimes is significant, and the softmax attention experiment lacks theoretical explanation.

- **Significance:** High for the theoretical deep learning community. Providing a unified mechanism for simplicity bias across architectures, with concrete predictions, advances the field's understanding of implicit regularization and learning dynamics. The practical impact depends on whether the mechanism operates in realistic training regimes.

- **Clarity:** Good. The paper is well-structured and the unified notation (Equation 1) is effective. The distinction between proven results and conjectures is maintained in the body but could be sharper in the abstract and title.

---

## wSbVv6xaRr

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary

FedMPDD introduces a federated learning algorithm that encodes each client's gradient via multi-projected directional derivatives—computing inner products along m random Rademacher vectors and transmitting only those m scalars plus a seed. The server reconstructs gradient estimators from these compressed messages, achieving O(m) uplink communication (m ≪ d) while providing privacy against gradient inversion attacks through the rank-deficiency of the projection. The paper proves O(1/√K) convergence (matching FedSGD) and provides reconstruction error bounds as a privacy metric, supported by experiments on MNIST, FMNIST, and CIFAR-10.

## Strengths

- **Unified mechanism for compression and privacy**: Unlike methods that bolt privacy (e.g., noise injection) onto compression, FedMPDD derives both from a single principled mechanism—the nullspace of a low-rank projection. The observation that the (d−m)-dimensional nullspace simultaneously provides compression and gradient obfuscation is a genuine insight, and the paper cleanly characterizes the relative reconstruction error (d−1)/m in Lemma 1.

- **Seed-based communication protocol**: Transmitting only m scalars and a random seed (from which the server regenerates the projection vectors) is an elegant design that avoids sending the projection vectors themselves, keeping uplink cost strictly O(m). This is a practical engineering contribution over methods that must share projection matrices.

- **Honest analysis of single-projection failure and principled fix**: The paper explicitly shows that FedPDD (single projection) suffers O(√d/√K) convergence due to variance scaling (Eq. 3), then proposes the multi-projection averaging mechanism to recover O(1/√K). This build-then-fix structure strengthens the technical narrative and the theoretical contribution.

- **Empirical privacy evaluation against state-of-the-art attacks**: The paper tests against both DLG (Zhu et al., 2019) and a recent attack (Yu et al., 2025), reporting SSIM scores and visual reconstructions (Figure 2). The consistently low SSIM across training epochs (Figure 1) provides concrete evidence that the nullspace protection holds in practice, not just theory.

## Weaknesses

### Major:

- **Privacy claims are not Differential Privacy; framing is misleading**: The paper repeatedly uses "privacy preservation" and positions FedMPDD against LDP, but its guarantees are reconstruction error bounds (Lemmas 1–2), not ε-DP. LDP provides worst-case guarantees against arbitrary adversaries; FedMPDD provides obscurity against specific gradient-inversion-style attacks. The paper's claim of "uniform privacy" (Abstract, Section 2) conflates consistent reconstruction error with formal privacy. The distinction matters: an adversary who knows the projection distribution could potentially extract label information, class membership, or semantic features from the m-dimensional projection even if full gradient reconstruction fails. The paper does not analyze what partial information *is* recoverable. This should be foregrounded, not buried in a remark contrasting with LDP.

- **Convergence rate stated incorrectly as O(1/K) in the Abstract and Theorem 2**: The actual bound in Theorem 2 (Eq. 5) with step size η = 1/(L√K) yields O(1/√K) for the average squared gradient norm, which is the standard non-convex rate. The Abstract's claim of "O(1/K)" is incorrect. While the "matching FedSGD" conclusion is still valid (FedSGD is also O(1/√K) for non-convex objectives), the notational error undermines confidence in the theoretical presentation and should be corrected.

- **Evaluated implementation does not reduce client-side computation; JVP optimization is future work**: Algorithm 2 (Line 6) computes the full stochastic gradient before projecting, incurring the same O(d) computational cost as FedSGD plus the additional O(dm) projection cost. Remark 1 suggests using Jacobian-vector products to avoid computing g_i explicitly, but Section F states this is planned future work ("we plan to implement a fully optimized version"). The paper's framing of suitability for "resource-constrained scenarios" (Abstract) thus applies only to bandwidth-constrained settings, not compute-constrained ones—a qualification that should be made explicit rather than implied.

- **Claim that "smaller m values sometimes yielded faster convergence" is misleading**: The Conclusion states this, and Section 3 references Fig. A.9, but Table A.9 clearly shows accuracy *increasing* with m (30.44% at m=50 vs. 79.02% at m=600 on LeNet-MNIST). The claim is only defensible if "faster convergence" means fewer total bits transferred (Table 2: m=600 uses 1.32 GB vs. m=2000 uses 3.26 GB for 60% target accuracy). Without this qualification, the statement contradicts the experimental data and confuses the reader about the m-accuracy relationship.

### Minor:

- **Multi-round privacy bound T < d/m is acknowledged but its practical implications are underexplored**: Remark 2 notes this worst-case constraint. For d ≈ 300,000 and m = 600, privacy erodes after ~500 rounds. While often sufficient, this is a hard limit for long-running FL tasks (e.g., continual learning). The paper mentions that gradient evolution provides additional protection but does not formalize this, leaving the bound as the only compositional guarantee.

- **No analysis of what partial/semantic information is extractable from the m-dimensional projection**: Lemma 1 bounds the full gradient reconstruction error, but does not address whether an adversary can recover sensitive attributes (e.g., class labels, membership) from the projected information alone. For m ≈ 600 out of d ≈ 300,000, the projection preserves some signal; characterizing what semantic content survives would strengthen or appropriately bound the privacy claims.

- **Experimental scale is limited to small models**: The largest model tested has ~300K parameters (CNN on CIFAR-10). The communication advantage of O(m) vs O(d) is most impactful for large models (ResNet-18 with ~11M parameters, as cited in the Introduction). The paper's motivating example of ResNet-18 is never actually tested, leaving the practical scalability unvalidated.

### Trivial:

- The Rademacher vs. Gaussian variance comparison (Lemma 3) is a nice theoretical addition but its practical impact on convergence is never isolated experimentally.

## Nice-to-Haves

- Comparison with a formally private baseline (e.g., DP-SGD or clipped Gaussian mechanism with ε-accounting) under equivalent communication budgets, to contextualize what FedMPDD's reconstruction-resistance buys relative to standard privacy definitions.
- Wall-clock benchmarks for total client-side time (gradient computation + encoding) vs. FedSGD, to quantify the real computational overhead.
- Experiments on a modern large-scale model (e.g., ResNet-18 or a fine-tuning task) where d is large enough for the communication savings to be practically decisive.
- Integration of error feedback with FedMPDD to investigate whether the compressed gradient estimator can benefit from accumulated error correction.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Convergence to a neighborhood, not exact optimum"** (from transferable weaknesses): Incorrect for this paper. Theorem 2 shows the bound goes to 0 as K → ∞; there is no non-vanishing neighborhood term. The convergence is to a stationary point in the standard non-convex sense.

- **"Missing privacy analysis"** (from transferable weaknesses): The paper has extensive privacy analysis including Lemmas 1–2, Remarks 2–6, and Appendix C–D. Whether the analysis is *sufficient* is debatable, but claiming it is *missing* is factually wrong.

- **"Only tested on two datasets"** (from transferable weaknesses): The paper tests on MNIST, FMNIST, and CIFAR-10 across multiple models and both IID/non-IID settings.

- **"Baselines are 2–3 years old"** (from transferable weaknesses): The paper compares against Yu et al. (2025), a very recent attack method, and includes SA-FedLora and lp-proj as compression baselines.

- **"Memory consumption footprint not reported"** (from transferable weaknesses): The paper discusses memory complexity in Remark 1 and provides Table F.1 comparing time/memory complexities of different gradient computation methods.

- **"Unfair comparison with QSGD because FedMPDD sends 32-bit floats"**: QSGD with 8-bit quantization on d=300K sends ~2.4M bits; FedMPDD with m=600 sends ~19.2K bits. The asymmetry favors FedMPDD, which makes the comparison *more* convincing, not less. Per the hard rules, this is not a valid criticism.

- **"Missing related works"**: Per the rules, I do not have external sources to confirm existence of specific missing references.

- **Formatting and reproducibility nitpicks** (undisclosed hyperparameters, etc.): The paper provides detailed hyperparameter tables (Tables H.4–H.7) and random seeds. Removed per hard rules.

## Novel Insights

The core insight—that averaging multiple rank-deficient projections can simultaneously recover FedSGD-level convergence while preserving a tunable nullspace for privacy—is sound and distinguishes FedMPDD from both fixed-subspace sketching methods (which lack per-round privacy randomness) and additive-noise DP (which hurts convergence direction). However, the paper's most underappreciated tension is that m serves three masters—convergence (wants large m), communication (wants small m), and privacy (wants small m)—and the convergence requirement (m = O(ln d)) is actually quite modest, meaning the practical bottleneck is the privacy–accuracy trade-off at small m, not the convergence–communication trade-off. This suggests the method's sweet spot may be in regimes where moderate privacy is acceptable and communication is the binding constraint, rather than as a replacement for formal DP.

## Suggestions

- Correct the convergence rate from O(1/K) to O(1/√K) throughout the Abstract, Introduction, and Theorem 2 statement. This is a straightforward fix that would resolve the inconsistency with the actual proof.
- Qualify the "smaller m yields faster convergence" claim explicitly as "faster convergence in terms of total communication bits" rather than optimization rounds, or remove it if the data doesn't robustly support even this interpretation across all settings.
- Add a paragraph in Section 2 or Appendix C explicitly discussing what the nullspace guarantee does *not* protect against (e.g., partial information leakage, attribute inference) and frame the contribution as "gradient inversion resistance" rather than "privacy preservation" to align terminology with the actual guarantees provided.
- Disclose upfront (in the Introduction or Method section, not just in Future Work) that the current implementation computes full gradients and the JVP-based computational savings are a prospective optimization, so the method's current benefit is communication reduction only.

## Axis Evaluations

- **Novelty**: Moderate-to-high. The projected directional derivative formulation and the specific nullspace-privacy argument in the FL context are novel, though the underlying random-projection machinery is well-established. The dynamic per-round sampling and seed protocol are practical contributions.

- **Technical soundness**: Mixed. The convergence proof is correct (modulo the O(1/K) vs O(1/√K) notation error), and the JL-based analysis is appropriately applied. However, the privacy analysis conflates reconstruction error with formal privacy, and the multi-round composition bound (T < d/m) leaves a significant gap for long-running training. The claim about smaller m yielding faster convergence contradicts the experimental data without careful qualification.

- **Empirical support**: Adequate for the communication and privacy-against-GIA claims on small models, but limited in scale (no models >300K parameters despite motivating with ResNet-18). The privacy evaluation against two attack families is a strength, though restricted to pixel-level SSIM rather than attribute-level leakage.

- **Significance**: Moderate. The joint communication-privacy mechanism addresses a real need in bandwidth-constrained FL, and the nullspace insight is valuable. However, the lack of formal DP guarantees limits applicability in regulated settings where privacy auditing requires ε-accounting, and the computational cost issue tempers the "resource-constrained" narrative.

- **Clarity**: Generally good, with clear algorithmic presentation and logical flow from FedPDD to FedMPDD. The main clarity issues are the incorrect convergence rate notation and the ambiguous "faster convergence" claim in the conclusion.

---

## GMP1S4R6Ke

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (4.5/10)
- Match: N/A

### Final Review

## Summary

LoRA-Mixer introduces a modular MoE framework that routes task-specific LoRA experts into the core linear projection matrices (Q/K/V/O) of attention modules and SSM projection layers, rather than the FFN blocks targeted by prior work. The framework is paired with a Routing Specialization Loss (RSL) that augments standard load-balancing auxiliary loss with an entropy regularization term, aiming to promote input-aware expert specialization while maintaining global balance. Evaluated across 15 benchmarks on Transformer (LLaMA3-8B, Mistral-7B) and SSM (Falcon-Mamba-7B) backbones, LoRA-Mixer claims improvements over baselines with ~48% fewer trainable parameters.

## Strengths

- **Principled routing loss with theoretical grounding:** RSL directly addresses a known failure mode of standard auxiliary losses (over-averaging toward uniform routing, rigorously shown in Appendix A.17). The convergence analysis (Theorem 1, Appendix A.1) and generalization bound (Theorem 2, Appendix A.2) provide formal support, and the empirical comparison in Table 8 shows RSL substantially outperforming GMoE, DS-MoE, and AESL under identical low-data (2K) conditions—demonstrating the practical value of the theoretical insight.

- **Architecture-agnostic design validated on both Transformers and SSMs:** The decision to target linear projection layers (ubiquitous across architectures) rather than FFN-specific structures yields genuine generality. The Falcon-Mamba-7B results (Table 2) confirm the method works on a pure SSM architecture where MixLoRA cannot be applied, which is a meaningful differentiator given the rising prominence of Mamba-like models.

- **Comprehensive empirical evaluation:** 15 benchmarks spanning five domains (medical, commonsense, NLP, mathematics, coding), three base models, and comparisons against six baseline methods plus three routing-loss-specific baselines provide substantial coverage. The plug-and-play experiment with externally sourced LoRAs (Section 4.3, Table 3) demonstrates practical deployability with only 2K additional routing data.

## Weaknesses

### Major:

- **Sign inconsistency between RSL formulation and stated objective.** Equation 5 defines L_RSL = α·Σ p̄_i·f̄_i **−** λ·E[H(p(x))]. Minimizing this loss *maximizes* entropy (promotes flat per-token routing distributions), yet the paper repeatedly claims RSL promotes specialization by *minimizing* entropy: "minimizing H(p(x)) reduces token-conditional uncertainty under a fixed global load, directly promoting specialization" (Section 3.3). The gradient in Equation 9 is consistent with the minus sign in Eq. 5, confirming the formulation. The convergence analysis (Appendix A.1) correctly proves that adding the negative-entropy term yields strong convexity and faster optimization—but this addresses *optimization stability*, not *specialization*. The paper conflates these two distinct benefits. The actual specialization likely arises from the task loss (Eq. 12), not the entropy term. This inconsistency between the mathematical formulation and the verbal/theoretical narrative undermines the core claim about *why* RSL works. If the sign should be +λ·H (to genuinely minimize entropy and promote specialization), then the gradient in Eq. 9 and the convergence analysis would need revision; if the sign is correct as written, the specialization claims need reframing.

- **No ablation isolating the effect of expert placement (attention projections vs. FFN).** The central architectural claim is that placing LoRA experts in attention projection layers is superior to FFN placement. However, every comparison (Tables 2, 4) is against methods that differ in *multiple* design choices simultaneously (placement, routing mechanism, training strategy, parameter count). Without a controlled experiment where LoRA-Mixer is instantiated on FFN layers vs. attention projections on the same backbone with all else held equal, the claimed benefit of projection-layer placement remains an untested hypothesis. The assertion that projection layers are "the most expressive point of the model" (Section 3.2) is stated without theoretical or empirical justification.

- **RSL data-efficiency claim is inconsistent at moderate data regimes.** Table 9 shows RSL underperforms the auxiliary loss at 4K training data (78.77 vs. 79.14). The explanation in Appendix A.16 ("RSL begins to explore finer-grained expert tasks... temporary instability") is post-hoc and not mechanistically grounded. A loss function that produces non-monotonic improvements as data increases raises questions about reliability. If RSL is recommended for "low-resource scenarios," the failure at a still-modest 4K data budget needs a more rigorous explanation or mitigation.

### Minor:

- **No standard deviations reported despite running three trials.** The paper states all experiments are "run three times and the average reported," yet no error bars appear in any table. Many improvements are modest (e.g., Falcon-Mamba HumanEval: 33.54→35.37; Mistral CoLA: 79.19→82.17), making it difficult to assess statistical significance.

- **Cross-model transfer claims are overstated.** Table 5 shows Mistral→LLaMA3 transfer works on 2/3 tasks, but ARC-E actually degrades (relative 0.97). Appendix A.10 reveals the architectures are near-identical (same hidden dim, layers, heads, FFN dim, activation). The claim of "extremely robust and transferable" routing overreaches given the limited scope—this is weight compatibility between near-twin architectures, not a general transfer result. No analysis is provided for models with differing dimensions, tokenizers, or normalization statistics.

- **The 48% parameter efficiency claim is not capacity-matched.** Appendix A.4 shows LoRA-Mixer uses 3.88% trainable params vs. MixLoRA's 8.08%, but this difference partly reflects LoRA-Mixer covering fewer modules (attention projections only) than MixLoRA (FFN + attention). Lower parameter count thus partially reflects lower per-layer expert capacity. A comparison at equal total expert parameters (adjusting rank or module coverage) would more rigorously test whether the efficiency gain comes from better placement or simply from using fewer expert modules.

- **OOD generalization gains are marginal.** Table 6 shows LoRA-Mixer improves over PHATGOOSE by only +0.19 (QQP), +1.44 (RTE), and +0.34 (MRPC) on OOD tasks. These small margins do not strongly support the claim of "excellent generalization ability."

- **Expert specialization is claimed but not verified at the token level.** Figures 3–4 show balanced load across experts and per-task load variation, but do not demonstrate that a "math expert" consistently activates on math tokens, a "medical expert" on medical tokens, etc. Without per-domain token-level activation analysis, the claim of "input-aware specialization" is supported only by aggregate statistics that could reflect correlated but non-specialized behavior.

### Trivial:

- The term "Serial Attention Routing" in the title is potentially misleading—the routing itself is not serial, and "serial" refers only to the fact that mixed LoRA outputs feed serially into the subsequent attention/SSM module. A more precise term would aid clarity.

## Nice-to-Haves

- A systematic λ sweep (beyond the 3 values in Table 15) with analysis of the specialization–balance tradeoff curve, to validate λ as an "interpretable knob" rather than a tuned hyperparameter.
- Evaluation on instruction-tuned models (e.g., LLaMA3-Instruct) with instruction-following benchmarks, since practical deployment of LoRA composition is most relevant in instruction-following settings.
- A per-domain token-level activation heatmap showing which experts fire on which token types, to directly verify input-aware specialization.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Weakness: Missing comparisons with recent (2024–2025) baselines like DynMoLE, LoRA+, HydraLoRA.** Per rules, I cannot confirm the existence or relevance of these methods and should not flag missing related works.

- **Weakness: Inference latency overhead compared to LoRAHub (0.574s vs. 0.482s).** LoRAHub is a training-free method that fundamentally differs in design; comparing a routing-based method to a training-free method on speed is asymmetric in favor of the baseline. Per rules, this is removed. (Note: LoRA-Mixer is actually faster than MixLoRA, 0.574 vs. 0.597, which is the fairer comparison.)

- **Weakness: Reproducibility concerns about undisclosed hyperparameters.** The paper provides hyperparameters in Appendix A.6 (r=64, α=128, dropout=0.1, lr=1e-5, batch size, gradient accumulation, scheduler) and A.8 (α, λ grid search). This is sufficient per community standards.

- **Weakness: Abstract should explicitly name MixLoRA as the 48% baseline.** This is a minor clarity nitpick about the abstract's phrasing, not a substantive issue.

- **Weakness: The paper should discuss environmental cost of training multiple experts.** This is outside the paper's stated scope (efficient multi-task adaptation) and is scope creep.

## Novel Insights

The sign inconsistency in RSL reveals a deeper conceptual issue: the paper's *theoretical* contribution (strong convexity from entropy regularization → faster, more stable optimization) and its *stated* contribution (entropy minimization → routing specialization) are two distinct mechanisms. The math actually supports the former; specialization likely emerges from the interaction between the task loss and the token-level gradient signals that the entropy term introduces, rather than from entropy minimization per se. This distinction matters because if the entropy term's true role is providing token-conditional gradients (rather than promoting peaked distributions), then the design space for alternative regularization terms is broader than the paper suggests—any term producing informative token-level gradients could serve a similar function.

## Suggestions

- **Resolve the sign inconsistency** in RSL by either (a) changing Eq. 5 to +λ·H (if the intent is genuinely to minimize entropy and promote specialization), with corresponding updates to Eq. 9 and the convergence analysis, or (b) revising the verbal framing to accurately describe the entropy term's role as providing token-level gradient signals for optimization stability rather than directly promoting specialization. Either resolution would substantially strengthen the paper.

- **Add a placement ablation:** Run LoRA-Mixer with experts on FFN layers vs. attention projections on the same backbone (e.g., LLaMA3-8B), keeping all other settings identical. This single experiment would either validate or invalidate the core architectural claim.

- **Report standard deviations** for at least the main comparison tables (Tables 2, 3, 4) given that the data from three runs already exists.

---

## rBj2iVyrhh

- GT: Reject (avg 2.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary

This paper proposes Classifier-Constrained Alternating Training (CCAT) to mitigate modality imbalance in multimodal learning. The key insight is that existing alternating training methods, while reducing encoder-level gradient conflicts, fail to prevent classifier bias toward faster-converging modalities. CCAT addresses this via a two-stage framework: (1) pretraining a shared classifier with bidirectional cross-attention and a modality-contribution regularization term, then (2) freezing this classifier during alternating modality-specific encoder training, with lightweight LoRA modules providing modality-specific adaptation and a sample-level secondary update mechanism for severely imbalanced samples.

## Strengths

- **Well-motivated problem framing:** The paper identifies a concrete, previously underexplored failure mode of alternating training—classifier-level bias entrenchment—and supports it with empirical evidence (Figure 1 showing persistent contribution disparity under MLA). The analogy to class imbalance (Section 3.1) provides a useful conceptual lens that justifies the fixed-classifier design.
- **Principled architectural design with clear ablation support:** Each component (classifier freezing, alternating training, secondary updates, LoRA modules) is systematically ablated in Table 2, demonstrating that all elements contribute to the final performance. The ablation is clean and the degradation patterns are interpretable (e.g., removing LoRA drops CREMA-D from 85.89% to 84.68%).
- **Strong empirical gains on challenging benchmarks:** The reported improvements are substantial, particularly +6.76% on Kinetic-Sound and the large margin on CREMA-D, suggesting the method meaningfully addresses modality imbalance rather than offering marginal tuning benefits.

## Weaknesses

### Major:

- **Missing SOTA baselines in main results table:** Section 4.1 explicitly lists MLA, MMPareto, and LFM as baselines, and Section 4.2 observation (iv) references their unimodal results, yet **Table 1 contains no rows for these methods**. MLA (Zhang et al., 2024) is the most critical omission—it is the direct predecessor that CCAT extends, and its absence makes it impossible to assess the specific contribution of the classifier constraint beyond the alternating mechanism itself. This is a significant gap in the experimental validation for a paper claiming SOTA performance.
- **Disconnect between theoretical analysis and actual method:** Section 3.1 derives the modality imbalance mechanism under the assumption that the fused feature is a linear combination $\mathbf{f} = \gamma_1 \mathbf{f}^{(1)} + \gamma_2 \mathbf{f}^{(2)}$ (Eq. 3), establishing a "theoretical isomorphism" with class imbalance. However, the actual pretraining stage uses **bidirectional cross-attention** (Appendix A.1, Eqs. 14–22), which produces context-aware representations—not a scalar-weighted sum. The paper does not acknowledge or discuss this simplification. While the linear model provides intuition, calling it a "theoretical isomorphism" and a "new theoretical framework" (contribution i) overstates what is essentially an illustrative analogy.

### Minor:

- **Unusually large performance gap on CREMA-D raises baseline fidelity questions:** CCAT achieves 85.89% vs. OGM-GE's 68.14%—a +17.75% absolute improvement. For a well-studied benchmark and a method that modifies training strategy rather than backbone capacity, this gap is anomalous. The paper states encoders are "ResNet18 across all datasets" but does not explicitly confirm that all baseline numbers were re-implemented with identical encoders, training budgets, and preprocessing. If baseline numbers come from original papers with different architectures, the comparison is unfair; if re-implemented, the poor baseline performance requires explanation. Either way, clarification is needed.
- **Insufficient justification for LoRA over alternatives:** LoRA modules are applied to transform features ($\text{LoRA}_m(\mathbf{z}_i^m) = \mathbf{B}_m \mathbf{A}_m \mathbf{z}_i^m$, Eq. 9) before the frozen classifier. The ablation (Table 2) shows LoRA helps (+1.21% on CREMA-D), but does not compare against the natural alternative: simply unfreezing the classifier and allowing full fine-tuning during alternating training. If unfrozen fine-tuning performs comparably, the low-rank constraint and the entire freezing+LoRA design would lose justification. This comparison is essential for validating the core architectural claim.
- **No variance or error bars reported:** Results are averaged over three random seeds (footnote of Table 1) but no standard deviations or confidence intervals are provided. Given the magnitude of claimed improvements, reporting variance is important for assessing robustness.
- **Algorithm 1 notation is confusing regarding contribution estimation:** Line 10 references Eq. (6) for estimating contributions, but Section 3.3 explicitly states that during alternating training "the computation of $c$ follows the same decision-level fusion used in the inference stage," not the cross-attention fusion of Eq. (6). The algorithm and the prose contradict each other, harming reproducibility.

### Trivial:

- **MI estimator lacks bias/variance discussion:** The contribution estimator (Eq. 5) uses an InfoNCE-style formulation with cosine similarity but no learnable temperature or projection heads. While this serves as a regularization term rather than a contrastive objective, the paper provides no discussion of estimator quality or its sensitivity to feature space geometry across modalities.

## Nice-to-Haves

- **Computational cost analysis:** The two-stage training plus sample-level secondary updates increases wall-clock time. Reporting training time and FLOPs relative to single-stage baselines would help practitioners assess the cost-benefit trade-off.
- **Tri-modal or larger-scale evaluation:** The method is evaluated only on bimodal datasets. Testing on a trimodal dataset or a larger-scale benchmark (e.g., AudioSet or full Kinetics) would strengthen claims of scalability.
- **Sensitivity analysis for the pretraining stage:** An ablation comparing the pretrained+unbiased classifier initialization versus a randomly initialized frozen classifier would isolate whether gains come from the quality of initialization or simply from the freezing constraint itself.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"How do gradients propagate with frozen classifier and no LoRA?" (from Spark Finder):** Table 2 Row 4 (Fix ✓, LoRA ✗) achieved 84.68%. The concern that gradients cannot flow without LoRA reflects a misunderstanding of backpropagation—a frozen linear layer still passes gradients to its inputs (the encoders), even though its own weights do not update. This is standard behavior and not a problem.
- **"Missing related works" (from Harsh Critic via transferable weaknesses):** Removed per hard rule—cannot confirm existence of specific uncited works.
- **"Missing reproducibility statement / code availability" (from transferable weaknesses):** Removed per hard rule—nitpicks about code availability and reproducibility artifacts are excluded.
- **"'faithfully' editing artifact in contributions" (from Harsh Critic):** Removed as a formatting nitpick per hard rule.
- **"Class imbalance of datasets requires balanced metrics" (from transferable weaknesses):** While worth noting, this is a generic concern that could apply to nearly any classification paper and doesn't specifically target a flaw in this work's core claims. The paper's focus is modality imbalance, not class imbalance.

## Novel Insights

The parallel between class imbalance and modality imbalance at the gradient-dynamics level (Section 3.1) is a genuinely useful reframing that suggests a family of techniques from the class-imbalance literature—fixed classifiers, re-weighting, delayed re-sampling—could be ported to multimodal learning. However, the current analysis is more illustrative than rigorous; the most valuable insight is the empirical observation (Figure 1) that alternating training alone does not resolve classifier-level bias, which directly motivates the freezing strategy and could independently inform future work even beyond this specific method.

## Suggestions

- **Add MLA, MMPareto, and LFM results to Table 1** (or a supplementary table). MLA is the most critical comparison since CCAT directly extends it—include it to quantify the specific gain from the classifier constraint.
- **Add an ablation row for "unfrozen classifier with full fine-tuning"** to Table 2, establishing that LoRA + freezing outperforms the simpler alternative of allowing the classifier to adapt freely during alternating training.
- **Explicitly acknowledge the linear-fusion simplification** in Section 3.1 and soften the "theoretical isomorphism" language to "illustrative analogy" or "conceptual parallel," noting that the actual architecture uses cross-attention. This preserves the motivational value without overclaiming.

---

## dCtkwjkK0E

- GT: Reject (avg 2.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary

This paper presents a theoretical framework for active learning in flow matching models with continuous conditions, motivated by expensive label acquisition in engineering design. By analyzing flow matching through a piecewise-linear neural network lens, the authors derive how dataset composition influences generation diversity (same-label data) and accuracy (different-label data), propose two competing query strategies ($Q_D$ and $Q_A$), and introduce a weighted hybrid to navigate the trade-off. Experiments on synthetic and three shape-design benchmarks show improvements over discriminative AL baselines.

## Strengths

- **Clear articulation of an under-explored problem**: The paper draws a sharp distinction between "generative models for active learning" (data augmentation for classifiers) and "active learning for generative models" (efficiently training the generator itself). This framing is precise and identifies a genuine gap—existing AL literature overwhelmingly targets discriminative settings, and the paper explains why those strategies fail for conditional generation with continuous labels.

- **Mechanistic insight linking data composition to model behavior**: The theoretical framework yields a specific, non-obvious claim: data sharing the same label drives generation diversity (via combinatorial interpolation), while data with different labels drives accuracy (via reducing interpolation error bounds). Even under the strong assumptions discussed below, this provides actionable guidance that goes beyond generic uncertainty-sampling heleneistics and is validated by the ablation in Fig. 9 and the consistent $Q_D$/$Q_A$ divergence across datasets.

- **Practical decoupling of query from generative model training**: Because $Q_D$ and $Q_A$ operate on dataset-level computations (distances and entropy in label/data space) rather than requiring intermediate FM retraining cycles, the annotation budget is allocated before any expensive generative model training. This is genuinely useful in the target domain where simulation labels are costly.

## Weaknesses

### Major:

- **The piecewise-linear assumption is central but unverified**: The entire theoretical derivation—Eq. 2 (linear interpolation of vector fields), Eq. 3 (linear interpolation of generated outputs), and the resulting query strategies—rests on the hypothesis (stated in Sec. 2.2) that flow matching networks exhibit "piecewise-linear interpolation behavior" due to condensation. While LeakyReLU networks are technically CPWL, condensation into the simple interpolation regime described by Eq. 3 typically requires specific conditions (e.g., infinite width, special initialization). The paper provides no empirical verification that the trained 8-layer, 512-unit networks actually obey Eq. 3 in practice (e.g., by testing whether generated samples at interpolated conditions match the linear combinations predicted). If the networks learn significantly non-linear mappings—as deep networks generally do—the "generation law" (Eq. 3) is invalid, and the theoretical justification for $Q_D$ and $Q_A$ collapses to heuristic motivation. This gap between strong theoretical assertions and unverified assumptions is the paper's most significant vulnerability.

- **Mishalignment between theoretical diversity definition and evaluation metric**: The theory in Sec. 2.3 defines diversity as the number of distinct combinatorial interpolation types at a *fixed* condition $c^*$—essentially measuring multimodality conditional on $c$. However, the experimental metric (Eq. 8) computes average pairwise Euclidean distance integrated over the entire condition space $dc_Y$. A model that simply outputs distinct shapes for distinct conditions (basic conditional generation functionality) would score high on this metric without being "diverse" in the theoretical sense. The paper needs to either (a) use a conditional diversity metric that aligns with the theory, or (b) explicitly reconcile why the unconditional metric validates the conditional theory. Without this, the empirical validation of $Q_D$'s theoretical motivation is weakened.

- **RBF surrogate quality is a hidden dependency without analysis**: Both $Q_D$ and $Q_A$ depend on RBF neural network predictions of labels for unlabeled data (Sec. 2.3, 2.4). The quality of query selection is entirely bottlenecked by RBF accuracy, yet the paper reports no RBF prediction error, no sensitivity analysis of how selection degrades with surrogate noise, and no comparison against using ground-truth labels for selection (an oracle upper bound). For complex physics tasks (e.g., 4D starship labels), RBF performance on sparse training data may be poor, potentially causing the query strategy to select suboptimal points. This is especially concerning given the "decoupled" design philosophy: the strategy is blind to the FM model's actual failure modes and relies entirely on the surrogate.

### Minor:

- **$Q_A$ is coreset in label space**: The paper acknowledges that $Q_A$ (Eq. 6, maximizing label-space distance) "performs the coreset algorithm in the label space." The novelty here is the *derivation* from the error bound (Eq. 5) rather than the algorithm itself. The contribution is the insight that the theoretical framework yields coreset as the accuracy-optimal strategy for this setting, not a new algorithm. This should be framed more carefully to avoid overstating algorithmic novelty.

- **$Q_D$ ablation reveals limited contribution of theory-driven terms**: Fig. 9 shows that the `distance(x, X)` term (standard data-space coreset diversity) is the most important component of $Q_D$, while the theory-motivated terms ($-distance(y, Y)$ and $\Delta entropy$) have comparatively minor effects. This suggests that the primary driver of $Q_D$'s performance is a well-known data-space diversity term, diminishing the practical impact of the novel theoretical framework for the diversity strategy specifically.

- **Claim of outperforming the full dataset needs explanation**: Sec. 3.2 states "$Q_D$ achieves the highest diversity, even outperforming the model trained on the full dataset." While this is plausible for the intentionally uneven synthetic dataset, the paper does not explain whether the real-world datasets share similar imbalance properties, or whether this result reflects a metric artifact rather than genuine improvement. A brief explanation would strengthen credibility.

- **Lemma 1 proof has a logical gap**: The proof in Appendix A (Eq. 17–19) compares two vector fields that are acknowledged to be "not exactly the same" but claims "their final generated results are consistent." The justification—that the difference lies only in noise schedules—is stated without formal proof that different noise schedules yield identical marginals. This should either be proven rigorously or the claim of equivalence should be softened.

### Trivial:

- **Notation inconsistency between condition $c$ and label $y$**: The paper uses $c$ for conditions and $y$ for labels, but in this setting they refer to the same physical quantities (e.g., lift coefficient). The interchangeable usage can confuse readers about whether the model is $p(x|c)$ or $p(x|y)$. A clarifying sentence would help.

## Nice-to-Haves

- Empirical verification of the piecewise-linear interpolation behavior (e.g., measuring linearity along condition-space paths in trained models) would substantially strengthen the theoretical foundation.
- An oracle experiment showing $Q_D$/$Q_A$ performance with ground-truth labels vs. RBF-predicted labels would quantify the surrogate's impact.
- Comparison with a generative-specific AL baseline (e.g., adapting GALIS to continuous conditions) would better position the contribution beyond "we beat discriminative methods."
- Evaluation on a non-shape-design task with continuous conditions (e.g., conditional image generation with continuous attributes) would demonstrate broader applicability, though the paper's scope to shape design is clearly stated.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Unfair comparison with baselines**: The claim that comparing against discriminative AL baselines is unfair is removed because the paper's core argument is precisely that generative-specific strategies should outperform discriminative ones in this setting. Showing that discriminative methods underperform is the point, not a flaw. (Removed: weakness about unfair comparison)
- **Missing related works / insufficient citations**: Per review rules, I cannot verify the existence of uncited works and should not flag missing references. (Removed: weakness about missing citations to recent AL methods)
- **Reproducibility concerns about undisclosed hyperparameters**: The specific values of α, β, γ are implementation details. While sensitivity analysis would be valuable, the absence of disclosed values is a minor reproducibility concern not rising to the level of a core flaw. (Removed: weakness about undisclosed hyperparameters)
- **Formatting and typographical issues**: Removed per rules against formatting/style nitpicks. (Removed: weakness about typos, equation numbering, figure formatting)
- **Demand for evaluation on image benchmarks (CIFAR-10)**: The paper explicitly scopes itself to shape design with continuous conditions. Requesting evaluation on fundamentally different tasks is scope creep. (Removed: weakness about limited to shape design domain—demoted to nice-to-have)
- **Statistical significance / confidence intervals**: Single-run evaluation with curve plots is standard in this community for active learning papers; demanding confidence intervals is not the field's norm. (Removed: weakness about lack of error bars)
- **Ethics statement**: Not a methodological concern. (Removed)
- **Skepticism about DALL-E-3/Veo3 using flow matching**: These are cited as motivating examples of advanced systems; the paper's contribution does not depend on this claim being precisely accurate. (Removed: minor factual concern about citations)

## Novel Insights

The paper's most interesting insight is the *formal demonstration that diversity and accuracy are inherently antagonistic from a dataset perspective in conditional generative models*: same-label data increases multimodality at each condition (by expanding combinatorial interpolation types), while different-label data reduces interpolation error (by shrinking convex hulls in label space). This is distinct from the usual GAN-style diversity-accuracy trade-offs (mode collapse vs. fidelity) because it operates purely at the dataset level before any model training, suggesting that the trade-off is a property of the data itself rather than the optimization process. The practical implication—that one cannot simultaneously maximize both without a hybrid strategy—is non-obvious and useful for practitioners allocating simulation budgets.

## Suggestions

- **Verify Eq. 3 empirically**: Train a flow matching model, pick two conditions $c_0, c_1$ in the dataset, and test whether the generated sample at $c^* = 0.5c_0 + 0.5c_1$ is approximately $0.5x_0 + 0.5x_1$. Report the deviation. Even partial validation would significantly strengthen the theoretical foundation.
- **Add an oracle-label experiment**: Run $Q_D$ and $Q_A$ using ground-truth labels instead of RBF predictions for the unlabeled pool. The gap between oracle and RBF-based performance directly quantifies the surrogate's impact and addresses the hidden-dependency concern.
- **Use a conditional diversity metric**: Report diversity at fixed conditions (e.g., variance of generated samples for a given $c$) in addition to the current unconditional metric, to align evaluation with the theoretical claims in Sec. 2.3.

---

**Axis assessments:**

- **Novelty**: Moderate. The application of active learning to flow matching with continuous conditions is novel, and the theoretical framework provides genuine insight. However, $Q_A$ is coreset-in-label-space and the most effective term of $Q_D$ is data-space coreset, limiting algorithmic novelty.
- **Technical soundness**: Mixed. The theoretical framework is creative but rests on an unverified assumption (piecewise-linear condensation), and the Lemma 1 proof has a gap. The RBF surrogate dependency is unanalyzed. The diversity metric misaligns with the theory.
- **Empirical support**: Adequate for the claimed application domain. Four datasets with consistent trends, but missing critical ablations (surrogate sensitivity, oracle comparison, conditional diversity).
- **Significance**: Moderate-to-good for the engineering design community. The decoupled query strategy and tunable diversity-accuracy trade-off are practically useful. Broader impact on generative AL is currently limited by the unverified assumptions.
- **Clarity**: Adequate. The paper is generally readable with helpful figures (Fig. 1, 2), but notation inconsistency ($c$ vs. $y$) and the abrupt transition to the piecewise-linear framework in Sec. 2.2 hinder full accessibility.

---

## OuMNJoKJBQ

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

This paper investigates why LLM safety alignment is vulnerable to jailbreak attacks, hypothesizing that current alignment relies on shallow refusal heuristics rather than deep reasoning. The authors support this with a causal intervention (deactivating reasoning-critical attention heads) showing reasoning degrades while safety persists, then propose a two-stage remedy: CoT safety fine-tuning on a newly constructed dataset, followed by Alignment-Weighted DPO (AW-DPO), which assigns distinct preference weights to reasoning and response segments based on per-segment harmfulness scores. Experiments across multiple model families and extensive jailbreak benchmarks demonstrate consistent safety improvements with competitive utility.

## Strengths

- **The causal intervention experiment provides novel mechanistic grounding for the "shallow alignment" hypothesis.** Deactivating reasoning-critical attention heads and observing that safety probing accuracy remains near-ceiling while reasoning accuracy collapses (Figure 1, Table 6) is a clean, interpretable test that goes beyond correlational observations made in prior work. This is a genuine contribution to understanding LLM safety mechanisms.

- **AW-DPO addresses a concrete, empirically identified failure mode.** The observation that ~15% of CoT fine-tuning failures involve reasoning-response misalignment (correct reasoning + unsafe answer, or incorrect reasoning + safe answer) directly motivates the separate weighting scheme. This makes the method design principled rather than ad hoc—a quality most DPO variants lack.

- **Comprehensive evaluation across model families, sizes, and attack types.** Testing on SorryBench (20 jailbreak strategies, 44 harm categories) across LLaMA-2-7B, LLaMA-3.2-3B, LLaMA-3.1-8B, and Mistral-7B, plus comparisons with reasoning models (Phi-4) and open-source aligned models, provides substantial coverage. The dataset transferability experiment (Table 3) is a practical contribution showing the AW-DPO preference data can be reused across architectures.

- **AW-DPO meaningfully improves upon standard DPO in utility preservation.** On LLaMA-3.1-8B, standard DPO applied after CoT Safety SFT drops utility from 55.39% to 41.45%, while AW-DPO maintains it at 54.70% (Table 1, Figure 4c). This is not a marginal difference and demonstrates that segment-level weighting addresses a real problem with indiscriminate DPO optimization.

## Weaknesses

### Major:

- **The mathematical derivation from Equation 3 to Equation 4 is incomplete, leaving the AW-DPO loss formulation ambiguous.** Equation 3 defines a token-level weighted reward φ_AW that sums log-probability ratios with per-token weights w_{s_t}. Equation 4 then states L_{AW-DPO} = w_reasoning · L_{rs}^{DPO} + w_respond · L_{rp}^{DPO}, where the weights are now segment-level scalars computed from judge scores. The paper does not derive how token-level weights in Eq 3 relate to the segment-level scalar weights in Eq 4. More critically, computing L_{rs}^{DPO} and L_{rp}^{DPO} as "separate DPO losses" requires specifying how autoregressive conditioning is handled: does the response-segment loss condition on the reasoning tokens of the same response, or only on the prompt? This matters because P(response|prompt) ≠ P(response|prompt, reasoning). Without this clarification, the formulation is underspecified and potentially unsound. The paper should either derive Eq 4 from Eq 3 rigorously or clarify that Eq 3 is the conceptual motivation and Eq 4 is the actual implementation (with explicit treatment of autoregressive context).

- **The causal intervention suffers from a ceiling effect confound that weakens the "superficial alignment" claim.** The alignment probing task achieves near-100% accuracy across all layers even before pruning (Figure 1). When a measure is at ceiling, it cannot show degradation regardless of whether the underlying capability has been impaired. The safety classification task—distinguishing obviously harmful prompts from benign Natural Questions—is too easy to serve as a rigorous test of reasoning-independence. A harder probing task (e.g., classifying adversarially rephrased harmful prompts vs. borderline-safe prompts) would provide a more diagnostic test. The benchmark evaluation in Table 6 partially addresses this (safety rate barely changes after deactivation on generation tasks), but this is a different evaluation modality than the probing setup. The disconnect between the probing evidence and the generation evidence should be discussed explicitly.

- **The absence of a randomized/reversed-weight ablation leaves it unclear whether the specific weighting scheme drives AW-DPO's improvements.** The paper compares AW-DPO to standard DPO (Figure 4b, 4c), but does not test AW-DPO with randomized weights (e.g., w_reasoning and w_respond sampled uniformly and renormalized) or reversed weights (assigning higher weight to the *less* harmful segment). Without this, it is possible that the improvement comes from the extra compute/data of the AW-DPO pipeline (separate scoring, additional signal) rather than from correctly assigning higher weight to the more harmful segment. A simple control where weights are set to 0.5 each (equivalent to uniform weighting but still using segment decomposition) would isolate the contribution of the weighting scheme.

### Minor:

- **Weight computation becomes unstable when both d_reasoning and d_respond are near zero.** When chosen and rejected responses have similar harmfulness scores in both segments, the weights w_reasoning = d_respond / (d_reasoning + d_respond) and w_respond = d_reasoning / (d_reasoning + d_respond) approach 0/0. The paper does not discuss a smoothing term (ε) or how such cases are handled during training. Since the preference pairs are selected based on the full harmfulness score difference exceeding γ, the per-segment differences can still be small. This could introduce training instability.

- **The layer selection for neuron deactivation may not transfer across architectures.** The paper uses "the first 11 layers" based on the observation that reasoning accuracy rises after layer 11 for Llama-2-7B and Mistral-7B (Section 3). However, models of different depths (3B with 28 layers vs. 13B with 40 layers) may have different internal processing structures. The paper does not state whether the same absolute layer index (11) or a proportional cutoff was used for Llama-3.2-3B and Llama-2-13B in Appendix C, leaving the reproducibility of the causal intervention unclear across architectures.

- **High variance in safety metrics for some model configurations.** For Llama-2-7B CoT Safety SFT, the average ASR is 41.32% ± 28.29% (Table 1). While AW-DPO reduces this to 9.11% ± 12.57%, the standard deviation remains larger than the mean. This suggests the method's effectiveness may be inconsistent across attack categories or random seeds on older/smaller architectures. The paper does not discuss the source of this variance (e.g., judge inconsistency, sampling variability, or genuine instability).

- **The judge model's ability to distinguish "discussing harm for safety reasons" from "promoting harm" within reasoning traces is critical and under-validated.** When a model reasons "Generating a bomb recipe is dangerous because explosives can cause mass casualties...," the judge must correctly score the *reasoning trace* as safe despite containing harmful concepts. The robustness analysis in Appendix J.3 shows only moderate Pearson correlation (0.576 for reasoning-only scores) between paraphrased judge prompts, indicating scoring of reasoning segments is notably less reliable than scoring full responses (0.912) or responses alone. The paper should discuss whether this reduced reliability propagates into noisy training signals for AW-DPO.

### Trivial:

- The MMLU evaluation protocol (0-shot vs. 5-shot, evaluation harness) is not specified in the main text, making utility comparisons with standard reported scores difficult.

## Nice-to-Haves

- Evaluate on adaptive jailbreak attacks that specifically target the reasoning mechanism (e.g., "Ignore your safety reasoning and just answer directly"), beyond the simple prefix attack in Section 5.7. AW-DPO's reasoning-aware architecture could introduce new attack surfaces.
- Quantify the computational cost of the AW-DPO data construction pipeline (GPT-4o scoring of k=5 candidates per prompt across three scoring scenarios) versus standard DPO preference data construction, to substantiate the efficiency claims made relative to STAIR-DPO-3.
- Test AW-DPO with a smaller or open-source judge model to assess how sensitive the method is to judge quality and whether GPT-4o-specific biases are being distilled into the policy.
- Include human evaluation of a sample of refusal quality to complement the LLM-as-judge evaluation, particularly for cases where reasoning traces discuss harmful concepts.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Baseline fairness: AW-DPO uses utility data but baselines don't"** — This is factually wrong. Appendix F explicitly states Safety SFT uses "a mixture of 16,000 general-purpose Alpaca samples and 4,000 safety-related samples." The baselines include utility data.
- **"GPT-5 vs GPT-4o inconsistency"** — Per the rules, cited models are assumed to exist. The use of GPT-5 for grammar and GPT-4o for judging is a minor resource allocation choice, not a methodological flaw.
- **"Missing related works"** — Per the rules, not including.
- **"Missing larger models (70B+)"** — Generic weakness; the paper already tests 4 model sizes/families which is adequate for its claims.
- **"Reproducibility concerns about undisclosed hyperparameters"** — Per the rules, removed as nitpick; the paper provides learning rates, batch sizes, and key hyperparameters in Appendix H.
- **"Inference cost of CoT not quantified"** — This is outside the paper's stated scope, which focuses on alignment robustness, not deployment efficiency.

## Novel Insights

The most interesting tension in this paper is that the causal intervention demonstrates safety and reasoning are *independent* in current models, yet the proposed fix assumes that making them *interdependent* (via CoT) improves robustness. This raises a question the paper does not fully address: is the problem that safety currently *ignores* reasoning, or that current alignment *doesn't build the right kind of reasoning* for safety? The comparison with Phi-4-reasoning models (Figure 3b) hints at the latter—general reasoning capability alone doesn't improve safety—but the distinction between "reasoning about safety" and "reasoning that happens to help safety" remains underexplored. Additionally, the finding that standard DPO causes a dramatic utility drop (55.39% → 41.45% on LLaMA-3.1-8B) while AW-DPO recovers it suggests that indiscriminate preference optimization may be actively harmful for utility, and segment-level weighting acts as a regularizer—this interpretation is more nuanced than the paper's framing and deserves explicit discussion.

## Suggestions

- Add a uniform-weight ablation (w_reasoning = w_respond = 0.5) and a reversed-weight ablation to isolate whether the *specific* weighting scheme matters versus the *decomposition* itself.
- Explicitly derive or explain the relationship between the token-level reward in Equation 3 and the segment-level loss weights in Equation 4, including how autoregressive conditioning is handled when computing separate losses for reasoning and response segments.
- Repeat the causal intervention with a harder safety probing task (e.g., adversarial rephrasings of harmful prompts) to address the ceiling effect concern and strengthen the "superficial alignment" evidence.

---

## WwDNiisZQm

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Final Review

## Summary

The paper introduces Content-Aware Mamba (CAM), a state-space model adapted for learned image compression that addresses two limitations of standard Mamba: its rigid raster-scan order and strict causality. CAM proposes Content-Adaptive Token Permutation (CTP), which reorders tokens by feature-space clustering so semantically similar tokens are processed consecutively, and Global-Prior Prompting (GPP), which injects sample-specific prompts derived from cluster centroids into the SSM's output projection matrix to provide global context. The resulting CMIC model achieves state-of-the-art rate-distortion performance, surpassing VTM-21.0 by up to 21.34% BD-rate on Tecnick.

## Strengths

- **Well-motivated and novel token permutation strategy.** The observation that Mamba's raster scan separates content-correlated but spatially distant tokens is precise and important for compression. The codebook-based K-Means clustering with EMA updates (Sec. 3.3, Algorithm 1) provides a stable, efficient alternative to naive online K-Means, and the visualization of cluster assignments (Fig. 10) convincingly shows semantically coherent groupings (e.g., centroid #10 for edges, #26 for textured warm regions, #33 for smooth blue/green backgrounds).

- **Strong empirical results with clear margins.** CMIC achieves BD-rate savings of 15.91%–21.34% over VTM-21.0 and outperforms Mamba-based baselines MambaVC and MambaIC by 2.36%–10.09% BD-rate (Tab. 1), all while reducing parameters by 56% and memory by 78% versus MambaIC. These are substantial and consistent improvements across three datasets.

- **Compelling ERF visualizations demonstrating content adaptivity.** Figures 7–9 provide unusually strong mechanistic evidence. The per-image ERF visualizations (Fig. 8) show the model's receptive field concentrating on semantically relevant distant regions (e.g., hair, feathers, shoreline), in stark contrast to the isotropic, content-agnostic ERFs of TCM-L and FTIC. The single-layer analysis in Fig. 9 cleanly isolates the contributions of CTP and GPP to breaking spatial and causal constraints.

- **Efficient architecture avoiding multi-directional scan overhead.** By using a single selective scan with content-adaptive ordering rather than four directional scans, CMIC achieves 78% lower GPU memory and 39% lower decoding latency than MambaIC (Tab. 1), making the efficiency advantage concrete rather than theoretical.

## Weaknesses

### Major:

- **Imprecise claims about "mitigating causality."** The paper repeatedly claims GPP "mitigates the strict causality" (Abstract, Sec. 1, Sec. 3.4). However, the state update $\mathbf{h}_i = \bar{\mathbf{A}}\mathbf{h}_{i-1} + \bar{\mathbf{B}}\mathbf{x}_i$ remains strictly causal—future tokens cannot influence the hidden state of current tokens. GPP modifies only the output projection via $\mathbf{O}_i = (\mathbf{C} + \mathbf{P})\mathbf{h}_i + \mathbf{Dx}_i$, which makes the *output* globally conditioned but does not make the recurrent state accumulation non-causal. The ERF visualizations (Fig. 9) support this: non-zero activations beyond the causal boundary appear because the prompt carries global statistics, not because the state sees future tokens. The paper should clearly distinguish "globally-conditioned causal modeling" from "non-causal modeling," as the current framing overstates the mechanism's capability.

### Minor:

- **Lack of explicit discussion on gradient flow through non-differentiable permutation.** The clustering assignments and token permutation are discrete, non-differentiable operations. While the EMA update for centroids is explained (Algorithm 1), the paper does not explicitly state how gradients propagate through the permutation to update the analysis transform weights $\theta_a$. Presumably, gradients from the SSM output flow back through the inverse permutation to the input tokens (a form of straight-through estimation), but this should be stated clearly, as the correctness of the gradient signal affects convergence guarantees. The training stability experiments in Appendix A.8 provide empirical evidence but do not substitute for an explicit gradient-flow explanation.

- **Incremental novelty of Global-Prior Prompting relative to MambaIRv2.** The paper acknowledges (Sec. 3.4, Appendix A.13) that the attentive state-space equation follows MambaIRv2 (Guo et al., 2024a). The main difference—tying the prompt dictionary to clustering centroids rather than using a standalone learnable matrix—is meaningful but incremental. The ablation in Table 9 (standalone dictionary: -15.02% vs. CAM: -15.91% on Kodak) confirms the benefit is real but modest (~0.9% BD-rate). The primary novelty thus rests more heavily on CTP than on GPP.

- **Decoder-side clustering stability under quantization noise is under-analyzed.** Each CAM block independently clusters its input features. During training, the encoder's CAM blocks receive pre-quantization features, while at inference the decoder's CAM blocks receive features derived from quantized latents $\hat{\mathbf{y}}$. Although the codebook centroids are learned across the dataset distribution and the EMA mechanism provides stability, the paper provides no quantitative analysis of how often cluster assignments differ between training and inference at the decoder side, or whether quantization noise causes tokens near cluster boundaries to flip assignments. This is not the same as an encoder-decoder synchronization problem (since each side clusters independently), but the robustness of decoder-side clustering to distribution shift from quantization deserves at least brief analysis.

- **Entropy model ablation reveals limitation not fully explained.** Section 4.5 notes that "adding CAM yields negligible performance gains while increasing latency" for the entropy model. The brief explanation—that the entropy model models distributions after redundancy removal where local relationships suffice—is reasonable but underdeveloped. If global context is crucial for transform networks, why would the conditional probability model of the transformed latents not benefit from the same awareness? A more principled discussion would strengthen the paper.

### Trivial:

- **Section 3.2 appears as a bare heading with no body text** before Section 3.3 begins. If this is present in the final PDF (not a parsing artifact), the overview architecture description should be completed or the section merged with 3.3.

## Nice-to-Haves

- A direct ablation comparing CMIC against a 4-directional Mamba variant built on the same backbone architecture (matched parameters) would more cleanly validate the efficiency advantage of single-scan CAM over multi-directional scanning, beyond the comparison with the architecturally different MambaIC.
- Analysis of failure cases: Table 5 shows high variance in activated cluster counts. Examining images where few clusters are activated (<20%) versus many (>50%) could reveal when CTP provides diminishing returns and when it is most critical.
- Sensitivity analysis of the K-Means iteration count (currently T=5) and EMA decay λ on final RD performance, to further substantiate robustness beyond the K-value ablation in Table 6.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Bitrate overhead of permutation side-information"** (from Spark Finder): Based on a misunderstanding. The permutation is an internal operation within each CAM block—tokens are clustered, permuted, processed by SSM, then inverse-permuted back to the original spatial layout before exiting the block. No permutation indices need to be transmitted between encoder and decoder. Each side independently computes its own clustering on its own features.
- **"Encoder-decoder synchronization risk from quantization mismatch"** (from Harsh Critic): While framed as a "critical" synchronization issue, this also stems from the mistaken assumption that the encoder's permutation must be reproduced at the decoder. Since each CAM block independently clusters its own features and applies its own permutation internally (with inverse permutation restoring spatial layout at the block output), there is no cross-side synchronization requirement. A residual concern about decoder-side clustering stability under quantization is kept above as a minor weakness.
- **"Distributed training overhead of per-block codebooks"** (from Balanced Review): Not standard to report in this venue for this type of work; the paper already notes codebook parameters are only 0.166% of total (Appendix A.9). Moved to nice-to-have territory.
- **"Training budget fairness verification"** (from Spark Finder): Reporting training GPU hours is not standard practice in the LIC literature. The paper provides standard setup details (optimizer, learning rate, dataset) in Section 4.1.
- **"Cross-dataset generalization to medical/satellite/screen content"** (from Spark Finder): Scope creep. The paper targets natural image compression; testing on out-of-domain modalities is beyond its stated scope.
- **"Section 3.2 empty section is a structural gap"** (from Harsh Critic): If present in the PDF, this is a trivial formatting issue; if a parser artifact, irrelevant either way.
- **"Missing comparison with Zhang et al. (2024b) under matched settings"** (from Spark Finder): The paper already provides a detailed comparison in Appendix A.2, noting Zhang et al. achieves -8.75%/-9.64% BD-rate on Kodak/Tecnick versus CMIC's -15.91%/-21.34%, and explains the architectural differences (grid-anchored coarse clustering vs. fine-grained codebook-based clustering).

## Novel Insights

The paper reveals an interesting asymmetry in the value of global context across different components of a learned image compression pipeline. CTP provides large gains (1.6%–2.2% BD-rate) in the transform networks, where discovering and exploiting long-range redundancy is the core task, but the same mechanism offers negligible benefit in the entropy model, which operates on already-decorrelated latents where local dependencies dominate. This suggests that the "globality premium" in LIC is highly task-dependent: it matters for removing redundancy but not for modeling the residual distribution, a distinction that could inform future architecture design choices beyond this specific method.

## Suggestions

- Revise the causality claims throughout the paper (Abstract, Sec. 1, Sec. 3.4) to use precise language such as "globally-conditioned output projection" or "output-level global context injection" rather than "mitigating strict causality" or "non-causal modeling," since the recurrent state update remains strictly causal.
- Add an explicit paragraph in Section 3.3 explaining how gradients propagate through the non-differentiable permutation (i.e., straight-through estimation via the inverse permutation), and optionally include a small quantitative analysis of cluster assignment consistency between training and inference at the decoder side to address the robustness concern.

---
**Quality Assessment:**
- **Novelty:** Moderate-to-strong. CTP is a genuinely novel and well-motivated mechanism. GPP is a meaningful but incremental adaptation of MambaIRv2's prompting.
- **Technical Soundness:** Mostly sound, with a notable gap in explicitly addressing gradient flow through non-differentiable operations and imprecise causality claims.
- **Empirical Support:** Strong. Comprehensive comparisons, consistent improvements across datasets, compelling ablations and visualizations.
- **Significance:** Strong. Establishes a clear new SOTA for Mamba-based LIC with meaningful efficiency advantages.
- **Clarity:** Generally good, weakened slightly by the overclaiming on causality and the imprecise gradient-flow discussion.

---

## zKQSyT7a7n

- GT: Reject (avg 6.0)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

##Summary

This paper introduces Visuo-Tactile World Models (VT-WM), a multi-task latent world model that integrates exocentric vision (Cosmos encoder) and fingertip tactile sensing (Sparsh-X encoder on Digit 360 sensors) to ground contact physics in robot manipulation imagination. The transformer-based predictor fuses visual and tactile tokens via factorized spatio-temporal self-attention and action cross-attention, trained with combined teacher-forcing and autoregressive sampling losses. Experiments on a Franka+Allegro Hand platform demonstrate improved object permanence (~33%) and causal compliance (~29%) in imagined rollouts versus a vision-only baseline, along with up to 35% higher success rates in zero-shot real-robot planning on contact-rich tasks and strong data efficiency (77% vs. 22% BC) on a novel plate-insertion task.

## Strengths

- **Well-motivated multimodal integration for a real gap.** The paper identifies a concrete failure mode of vision-only world models—visual aliasing of contact states—and demonstrates that tactile input specifically disambiguates these cases (Fig. 7: V-WM hallucinates cloth displacement when the hand hovers; VT-WM correctly predicts stasis). This is a targeted, non-obvious contribution rather than a generic "add a modality" approach.

- **Demonstrated real-robot zero-shot transfer.** The planning results go beyond latent-space metrics to actual physical execution. The 31% gain on wipe cloth and 35% gain on reach & push are substantively meaningful because they correspond to qualitatively different behaviors (establishing contact vs. hovering above the object), not marginal improvements.

- **Data efficiency result is practically compelling.** The 77% vs. 22% comparison on plate insertion with only 20 demonstrations shows the multi-task contact priors transfer meaningfully. The failure-mode analysis (VT-WM places beside rack; BC never reaches rack) provides mechanistic insight into why the WM representation helps.

- **Honest reporting of negative results.** The causal compliance evaluation shows VT-WM *degrades* on scribble with marker (t = −1.22, p = 0.23), and the paper reports this without obscuring it. This strengthens trust in the positive claims.

## Weaknesses

### Major:

- **No quantitative evaluation of the tactile prediction channel.** The paper claims to capture "the physics of contact through touch reasoning," yet all quantitative metrics (Fréchet distance, success rates) measure *visual* outcomes. The tactile predictions are shown qualitatively in Appendix B (Figs. 12, 13) but never evaluated quantitatively (e.g., tactile prediction error, contact detection accuracy, slip prediction). A world model that claims to reason about contact should demonstrate that its *tactile* predictions are accurate, not merely that tactile inputs improve visual outputs. Without this, the mechanism—"tactile grounding disambiguates contact states"—is asserted but only indirectly validated.

- **Data efficiency experiment conflates planning vs. BC with tactile vs. vision-only.** Section 4.3 compares VT-WM (world model + CEM planning) against ACT (behavioral cloning). This conflates two variables: the representation paradigm (WM vs. policy) and the modality (visuo-tactile vs. vision-only). A V-WM + CEM baseline in the same low-data setting would isolate whether the advantage comes from tactile grounding or from planning vs. cloning. The paper's own Limitations section partially acknowledges this ("does not fully rule out the possibility that a multi-task BC policy could also exhibit strong data efficiency"), but the comparison as presented is misleading about the *source* of the gain.

- **V-WM and VT-WM parameter/capacity matching is unspecified.** VT-WM concatenates tactile tokens alongside visual tokens, giving the transformer more input information and effectively more capacity per forward pass. The paper does not state whether V-WM was given additional visual tokens or architectural capacity to match, or whether it is strictly the same transformer with fewer input tokens. If V-WM is simply VT-WM minus the tactile tokens, the 33%/29% gains could partly reflect the benefit of additional tokens/attention targets rather than tactile grounding per se. A V-WM with matched token count (e.g., duplicated visual tokens or deeper processing) would isolate the modality effect.

### Minor:

- **Temporal alignment between vision and tactile inputs is unclear.** Section 3.2.2 states tactile input consists of "two frames per Digit 360 sensor, covering the most recent 0.16 seconds," while vision uses a 1.5-second clip. Yet it also states "The model uses a maximum context length of 9 frames for both vision and touch modalities." How 2 frames of tactile at high frequency map to a 9-frame temporal context alongside the 9-frame visual context is not explained. This creates ambiguity about whether tactile history is subsampled, padded, or operates at a different effective framerate within the transformer.

- **Binary gripper action space limits the scope of "contact-rich manipulation" claims.** The Allegro Hand has 16 DOF, but the action space uses only "a binary hand state representing pre-set open/close configurations" (Section 3.2.2). This reduces the hand to a parallel-jaw gripper. While the tested tasks (pushing, wiping, stacking) are genuinely contact-rich in the sense of requiring sustained physical contact, the claims about "dexterous manipulation" and "physics of contact" would be more precise if qualified—contact richness here comes from task geometry, not from finger-level dexterity.

- **CoTracker metric may be unreliable precisely where the model's advantage lies.** The Fréchet distance metric relies on CoTracker to track keypoints through occluded phases (e.g., object in hand). CoTracker can lose track during heavy occlusion and re-acquire afterward, potentially introducing noise that affects both models equally but reduces the metric's sensitivity to the very phenomenon (object permanence under occlusion) that VT-WM is claimed to improve. This does not invalidate the results but means the 33% figure may be an underestimate or overestimate.

- **Negative result on scribble with marker is not discussed.** VT-WM shows *worse* causal compliance than V-WM on this task (t = −1.22). The paper reports the number but offers no analysis of why tactile input might hurt in this case (e.g., marker contact is always present during the task, so tactile provides no disambiguation but adds noise). Understanding when touch *doesn't* help is as important as when it does for evaluating the generality of the approach.

- **Sparsh-X fine-tuning is not ablated.** Appendix A.1 states "fine-tuning the Sparsh-X encoder was beneficial" while Cosmos was kept frozen. The asymmetry is not justified or ablated. If fine-tuning the tactile encoder substantially improves performance (while the visual encoder remains frozen), this could mean the gains come from domain-adapted representations rather than from multimodal fusion per se. An ablation with frozen Sparsh-X would clarify this.

### Trivial:

- The CEM planning algorithm (Algorithm 1) uses 36 particles and 10 iterations, which is standard and adequate for the demonstrated tasks. No issue here.

## Nice-to-Haves

- **Closed-loop replanning experiments.** The open-loop execution is acknowledged as a limitation. Even a simple 2-step replanning demonstration (update context after first chunk) would substantially strengthen the practical relevance argument.

- **Multi-task BC baseline for data efficiency.** A multi-task ACT policy trained on all tasks (including the 20 plate-insertion demos) would clarify whether the 77% vs. 22% gap is due to the world model paradigm or the multi-task transfer.

- **Latent space analysis.** Attention visualizations on tactile tokens during contact-rich vs. free-motion phases, or a probing experiment, would provide mechanistic evidence that the model actually uses tactile information rather than ignoring it.

- **Inference latency quantification.** CEM with 36 particles × 10 iterations of autoregressive rollout is computationally expensive. Reporting wall-clock time for planning would help readers assess deployment feasibility.

- **Generalization to novel objects.** Testing on objects with different sizes, weights, or friction coefficients than those in training would strengthen the multi-task generalization claim, though the paper explicitly scopes this out.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"First multi-task visuo-tactile world model" claim disputed.** The harsh critic questions whether prior tactile servoing/dynamics work (Tian et al., 2019; Sutanto et al., 2019) invalidates the "first" claim. These prior works are task-specific dynamics models, not multi-task latent world models for planning. The distinction is clear in the paper's Related Work section. Removed as factually ungrounded—the paper's claim is specific to the multi-task latent WM + planning setting.

- **Missing comparison to Robopack or other tactile-dynamics baselines.** The balanced reviewer requests comparison to Robopack (Ai et al., 2024). Robopack addresses dense packing with tactile-informed dynamics, which is a different task and architecture. The paper's primary comparison is the ablation (V-WM vs. VT-WM), which is the most informative baseline for isolating the tactile contribution. Removed as unfair comparison demand—different problem setting.

- **Formatting/parser artifact complaints.** Both reviewers mention broken equations and tables from PDF extraction. Per hard rules, formatting nitpicks are removed.

- **Reproducibility concerns about fine-tuning details for Sparsh-X.** The harsh critic wants more details about Sparsh-X fine-tuning. Per hard rules, nitpicks about trivial implementation details are removed. The paper already states the fine-tuning was done and the appendix provides training parameters.

- **Compute infrastructure for inference.** The harsh critic asks about the inference hardware stack. Per soft rules, this is a nice-to-have, not a core flaw. Moved above.

## Novel Insights

The paper reveals an asymmetry in how tactile grounding helps: it provides the largest gains in tasks where *visual aliasing of contact states is the primary failure mode* (pushing, wiping, reach & push), but offers diminishing or even negative returns in tasks where contact is always present or where visual information is already sufficient (scribble with marker, reach button). This suggests that the value of tactile sensing in world models is not uniform but concentrated in the specific regime where contact state is under-determined by vision alone—a finding with practical implications for when tactile hardware investment is justified versus when vision-only approaches suffice.

## Suggestions

- Add a quantitative tactile prediction metric (e.g., L1 loss on predicted vs. ground-tr

---

## j3htU5i01r

- GT: Reject (avg 4.0)
- Predicted: N/A (6.0/10)
- Match: N/A

### Final Review

## Summary

The paper proposes a compositional meta-learning framework where tasks are represented as structured combinations of reusable modules. A gating RNN learns the "grammar" of module transitions while module RNNs learn the "syllables" (within-module dynamics). Training maximizes marginal likelihood via particle filtering; test-time adaptation requires no parameter updates, instead inferring module sequences through constrained probabilistic hypothesis testing. Experiments on synthetic rule-learning and motor-learning tasks demonstrate ground-truth component recovery, one-shot task inference, and robustness to sparse feedback.

## Strengths

- **Inference-based adaptation without parameter updates is a genuine conceptual departure from gradient-based meta-learning.** The paper shows (Figure 3e) that MAML, MLDG, and standard fine-tuning require hundreds of episodes, while the proposed method infers the solution from a single episode. The ablation in Figure 3 systematically establishes that both the gating network and modular structure are necessary, particularly under sparse feedback where unconstrained inference fails (Figure 3c vs. 3d).

- **Ground-truth recovery provides interpretable evidence that the model learns what it claims.** Figure 2a-c shows that both learned modules and learned transition statistics converge to the ground truth, going beyond typical meta-learning papers that only report task performance. This verifiability is a meaningful advantage of the controlled synthetic setting.

- **Sparse feedback handling is well-demonstrated and non-trivial.** Figures 2e, 2f, and 4e show that the model maintains hypothesis branching during feedback gaps and collapses appropriately when observations return. This leverages the learned grammar in a way that standard meta-learning methods fundamentally cannot, and the extended-sequence result (4× training length) shows generalization of the learned statistics rather than memorization of specific sequence patterns.

## Weaknesses

### Major:

- **Computational cost of inference is unanalyzed, making the efficiency comparison against gradient-based methods incomplete.** The paper claims "rapid acquisition" and compares against MAML and others on an episodes-based axis (Figure 3e/f), but each inference episode requires running K=250 particles through the module RNNs for T timesteps. A single inference episode may cost orders of magnitude more in FLOPs than a gradient step on a monolithic network. The sample-efficiency advantage is clear, but whether it translates to a wall-clock or compute advantage is unaddressed and potentially significant. This is not a fatal flaw—250× one episode may still be less than 1× 100 episodes—but the paper should quantify this rather than leave it implicit.

- **Evaluation is limited to low-dimensional synthetic tasks with known compositional structure.** The rule-learning task (6D vector shifts) and motor task (2D trajectories) are deliberately simple to enable ground-truth verification, which is valuable. However, the paper makes broad claims about compositional meta-learning that currently lack demonstration on more complex or higher-dimensional domains. Tasks with ambiguous or hierarchical compositional structure, noisy observations, or larger state spaces would better stress-test whether the approach scales beyond settings where the ground truth is cleanly factorizable. The authors acknowledge this as proof-of-principle, but the gap between the claimed framework and the demonstrated scope is notable.

### Minor:

- **Fixed, pre-specified number of modules limits practical applicability.** The number N of modules is set at initialization. While Figure A1a-d explores mismatches between module count and ground-truth operation count, there is no mechanism for dynamic module addition during training or inference. The Discussion identifies this as future work for continual learning, but it remains a constraint on the current framework.

- **Train-test distribution shift between soft (Gumbel-softmax) and hard (argmax) module selection is not analyzed.** Training uses soft relaxation for gradient flow while inference uses hard argmax. This is standard practice, but no analysis of gating entropy during training is provided to verify that the learned distributions are sufficiently peaked to support hard selection.

- **Non-Markovian capacity of the gating network is claimed but not quantitatively evaluated.** The paper states the gating RNN learns "strongly non-Markovian statistics" (Section 2.2), with Figure 2c showing history-dependent transition matrices visually. However, no experiment isolates the benefit of non-Markovian gating over, e.g., a higher-order HMM baseline, leaving the architectural choice partially unmotivated beyond visual evidence.

### Trivial:

- **Sparse feedback handling is described narratively in Section 2.3 but not given explicit mathematical formulation.** The mechanism (skipping likelihood-based resampling during missing-observation timesteps) is standard in particle filtering but could be stated as a formal equation for completeness.

## Nice-to-Haves

- **Ablation on particle count K versus inference accuracy and feedback gap length** would clarify how many particles are actually needed and whether the 250-particle budget is overkill for these tasks or necessary for degeneracy avoidance.

- **Wall-clock time or FLOP-normalized comparison against gradient-based baselines** would substantiate or qualify the efficiency claims and is straightforward to report.

- **Empirical comparison to Hummos et al. (2024)**, which the Discussion identifies as the closest related approach (compositional inference via gradient-based latent embedding optimization), would sharpen the contribution. Even a conceptual runtime analysis contrasting sequential module search vs. embedding optimization would help.

- **Evaluation on at least one established multi-task or meta-learning benchmark** (e.g., a structured prediction task with compositional generalization splits) would significantly broaden the impact.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"The abstract should hint at computational trade-offs"** — This is a formatting/style suggestion about what the abstract should contain; the abstract accurately describes the method, and the concern about computational cost is already captured in the Major Weaknesses above.

- **"The introduction should clarify why explicit modularity is necessary"** — The paper motivates this through the analogy to dynamical motifs (Yang et al., 2019; Driscoll et al., 2024) and by showing that a monolithic RNN without task identity fails (Figure 3a). The ablation in Figure 3c,d further supports the architectural choices.

- **"Missing comparison to Alet et al. (2019) and other modular meta-learning methods"** — Demanding comparisons to every related method is scope creep. The paper discusses Alet et al. conceptually and notes it uses simulated annealing rather than probabilistic inference, which is the key distinction.

- **"Quantitative module disentanglement metrics"** — The paper already provides quantitative ground-truth recovery metrics (module and gating accuracy in Figure 2a) alongside visual verification. Additional disentanglement metrics would add limited value for these controlled tasks where ground truth is directly measurable.

- **"Reproducibility concerns about hyperparameters or training instability"** — The paper provides full implementation details in Appendix A.1–A.2, including initialization schemes, learning rates, and batch sizes. The "chicken-and-egg" instability is acknowledged and mitigated through careful initialization (Section A.1).

- **"OoD detection threshold is undefined"** — Figure A1e demonstrates qualitative separation of likelihood between in-distribution and OoD tasks. Quantifying threshold policies is more relevant to a continual-learning extension than to the current proof-of-principle.

- **"Section 2.4 architectural changes for motor learning make results not directly comparable"** — The paper clearly states these modifications and motivates them (autonomous trajectories vs. input-driven outputs). The core framework remains the same.

## Novel Insights

The key insight emerging from this work is the clean separation between learning compositional *structure* (via gradient descent on parameters during training) and deploying that structure (via probabilistic inference at test time). This decoupling means that test-time adaptation requires no weight updates—only hypothesis testing over learned module sequences—making the approach inherently immune to catastrophic forgetting. The sparse-feedback results reveal an underappreciated advantage of this inference-based approach: the learned transition grammar acts as a strong prior that prunes the hypothesis space during observation gaps, something gradient-based adaptation cannot exploit because it lacks an explicit generative model of task structure.

## Suggestions

- Report wall-clock time or FLOP counts for inference vs. gradient-based baselines to clarify whether sample efficiency translates to compute efficiency, which is the metric practitioners care about most.

- Run a parameter sweep on K (e.g., K ∈ {10, 50, 100, 250, 500}) and plot inference accuracy vs. K, particularly under sparse feedback conditions. This would establish whether the method's success depends on generous particle budgets or is robust to limited compute.

- Test the framework on at least one task with higher-dimensional observations (e.g., 2D pixel inputs with compositional structure) to probe scalability beyond the current proof-of-principle setting.

## Evaluation Axis Summary

- **Novelty**: Moderate-to-high. The combination of modular architecture with learned gating grammar and particle-filter-based inference for meta-learning is distinctive, though individual components (HMMs, particle filtering, modular networks) are well-established.
- **Technical soundness**: Sound in its current scope, with clearly defined generative model and appropriate inference procedure. The main gap is the unanalyzed computational cost trade-off.
- **Empirical support**: Convincing for the synthetic, controlled setting. Ground-truth recovery, ablation studies, and sparse-feedback results are thorough within that scope. The gap is in demonstrating scalability and broader applicability.
- **Significance**: Potentially significant if the framework scales to more complex domains. The inference-based paradigm for meta-learning is conceptually important, but the current proof-of-principle limits immediate practical impact.
- **Clarity**: Well-organized and clearly written, with consistent notation and useful appendices. The paper's scope and limitations are honestly stated.

---


# Summary

Papers: 50 | Accuracy: N/A
