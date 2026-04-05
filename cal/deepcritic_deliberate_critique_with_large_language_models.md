=== CALIBRATION EXAMPLE 60 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate: the paper is about improving LLMs as math critics, and “DeepCritic” captures the deliberate critique framing. However, it slightly overstates novelty by suggesting a new paradigm rather than a training recipe built from staged SFT + RL.
- The abstract clearly states the problem, the two-stage method, and the main empirical claims.
- The strongest abstract claim is that the model outperforms “existing LLM critics (including the same-sized DeepSeek-R1-Distill models and GPT-4o)” and helps generators refine errors. This is supported in Table 1 and Table 3, but the abstract does not acknowledge that the strongest PRM baseline, Qwen2.5-Math-PRM-7B, still outperforms DeepCritic on the main judgment benchmark in Table 1. That nuance matters for an ICLR audience.

### Introduction & Motivation
- The motivation is clear and relevant: scalable oversight via critics is an important problem, and shallow critiques are a real failure mode in math reasoning.
- The paper identifies a meaningful gap: prior critics often verify but do not deeply interrogate reasoning steps, which can limit both judgment accuracy and usefulness for refinement.
- Contributions are stated reasonably well, but the framing is somewhat over-strong. The introduction implies a general advance in “deliberate critiquing,” yet the paper’s strongest evidence is in mathematical process verification, with only a later appendix study on summarization.
- The claim that the framework can “also be effectively applied to subjective domains” is not part of the main contribution and is only weakly supported in Appendix M; it should be presented more cautiously.

### Method / Approach
- The two-stage pipeline is described clearly enough to understand the intended mechanism: curated long-form critiques for SFT, then RL on either human-labeled PRM800K or auto-labeled Numina-style data.
- The key novelty is the iterative critique generation: initial critique, in-depth critique, then merged deliberate critique. This is conceptually coherent.
- There are, however, important methodological questions:
  - The paper relies heavily on Qwen2.5-72B-Instruct to generate both the initial and deep critiques in Stage 1. This makes the seed dataset potentially self-referential and dependent on the teacher’s own biases. The paper argues this is better than direct distillation, but it does not fully isolate whether gains come from the deliberate structure, from higher-quality teacher outputs, or from aggressive filtering.
  - The filtering rule “retain only solutions in which the in-depth judgment results of all steps align with ground truth” is very strict. This improves label quality, but it also introduces a strong selection bias toward examples that the teacher can already solve consistently. The paper does not sufficiently analyze how this affects coverage, difficulty, or the types of reasoning patterns the model learns.
  - In the RL data construction, the Monte Carlo correctness criterion is quite complex and somewhat brittle: a step is labeled by whether later rollouts converge to correct answers. This may be reasonable, but the paper does not clearly justify why this proxy is reliable enough across different difficulty regimes, nor how sensitive results are to the threshold choices.
- There is also a reproducibility concern: the method depends on many prompt-specific, model-specific heuristics, but the core algorithm is not presented in a way that makes ablation of the prompt design or teacher quality straightforward.
- For the theoretical side, there are no real formal claims to evaluate, so this is primarily a systems/training paper.

### Experiments & Results
- The experiments do test the central claims: error identification on three benchmarks, ablations on data generation strategy, test-time majority voting, refinement, and some robustness/generalization studies.
- The main results in Table 1 are strong: DeepCritic-7B-RL-PRM800K is clearly better than the base model and most LLM critics, including GPT-4o and DeepSeek-R1-Distill-Qwen-7B. That said, the comparison is not uniformly decisive:
  - Qwen2.5-Math-PRM-7B remains better than DeepCritic on all three main benchmark averages in Table 1, despite the paper’s emphasis on DeepCritic’s superiority over “existing PRMs.”
  - The paper should more carefully position its result as “best among LLM critics” rather than “best overall,” because the strongest PRM baseline is not surpassed.
- The ablation in Table 1 is useful and does support the claim that the iterative critique generation pipeline matters: DirectDistill-7B-SFT < InitialCritic-7B-SFT < DeepCritic-7B-SFT.
- The RL comparison between PRM800K and Numina data is informative, but it is not fully apples-to-apples: PRM800K is human-labeled and much larger, while Numina is automatically annotated. The conclusion that RL “further boosts” performance is valid, but the paper should avoid implying these are comparable supervision regimes.
- The refinement results in Table 3 are compelling in showing practical utility, but they are also somewhat narrow:
  - They focus on MATH500 and AIME24–25, and the gains are modest in absolute terms.
  - The paper reports w→c and c→w, which is useful, but the main claim would be stronger if it also showed whether improvements persist under more diverse generator settings or with stronger prompt baselines.
- Error bars and statistical significance are not reported. For a paper with several close comparisons and many benchmark scores, this is a real omission for ICLR standards.
- A number of analyses use small or hand-sampled subsets, such as the self-correction study in Figure 3 with only 10 correct and 10 incorrect solutions per subset. That is suggestive, but not strong evidence.
- The paper does not provide enough analysis of failure cases. For example, when the critic is wrong, what kinds of errors dominate? Where does the deliberate critique help most or least?

### Writing & Clarity
- The paper is mostly understandable, and the central idea comes through.
- The main clarity issue is that some methodological distinctions are not crisp enough. In particular, the difference between initial critique generation, in-depth critique generation, and final critique synthesis is conceptually interesting but easy to lose track of without repeatedly consulting Figure 2 and Section 3.2.1.
- Figure 1 and Figure 2 help motivate the method, but the paper would benefit from more concise explanation of exactly what behavior is induced by each stage.
- Some claims in the experimental narrative are stronger than the evidence presented, which affects clarity of interpretation more than prose quality. For example, “weak-to-strong supervision” in Table 3 is intriguing but not yet convincingly established from the reported evidence.
- The appendix includes several helpful tables, but the main paper still depends on them heavily for essential evidence. That is acceptable to a degree, but some key points—especially robustness and limitations—are deferred too much.

### Limitations & Broader Impact
- The ethics statement is very thin and does not engage with the actual limitations of the approach.
- Important limitations that are not adequately acknowledged:
  - The method is heavily tuned to math/process-verification settings, where correctness is more objectively measurable than in many real-world critique tasks.
  - The seed data depends on a powerful 72B teacher model, so the approach is not purely low-resource or fully self-generated at the first stage.
  - The method’s utility may depend on having a reasonably competent base model that can already do some critique and self-correction, as the paper briefly notes in a footnote but does not foreground as a real limitation.
  - The gains may partially reflect better formatting and longer deliberation rather than deeper critique capability per se.
- Broader-impact risks are not discussed in any substantive way. Since the paper is about scalable oversight, it should at least discuss the possibility that such critics could also be used to optimize away genuine error detection, overfit to benchmark-style process traces, or create a false sense of reliability in high-stakes settings.

### Overall Assessment
This is a solid and timely ICLR-style paper with a clear problem, a plausible method, and strong empirical improvements over many LLM critics. The strongest contribution is the deliberate critique data-generation pipeline, which appears to materially improve step-level judgment and downstream refinement. However, the paper’s evidence is not quite as decisive as its claims: the strongest PRM baseline remains superior on the main benchmark table, the method depends heavily on a large teacher and many heuristics, and the evaluation lacks statistical rigor and deeper failure analysis. I think the paper is promising and likely publishable if the authors sharpen the claims, better position the method relative to PRMs, and more directly address selection bias, robustness, and limitation questions.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes DeepCritic, a two-stage training pipeline for improving LLMs as mathematical critique models. The key idea is to first create a small seed dataset of long, step-wise, “deliberate” critiques via iterative prompting and meta-critique, then further strengthen the model with RL using either human-labeled process data (PRM800K) or automatically constructed labels from Monte Carlo rollouts. The resulting 7B critic is reported to outperform several PRMs, general-purpose LLM critics, and some reasoning-heavy models on error-identification benchmarks, while also helping generators refine mistakes more effectively.

### Strengths
1. **Clear and relevant problem formulation for ICLR’s scalable oversight agenda.**  
   The paper targets an important and timely problem: how to build critics that provide better supervision for reasoning models. This aligns well with ICLR interest in alignment, process supervision, and test-time verification.

2. **A concrete multi-stage training recipe with an interesting data-generation mechanism.**  
   The paper’s main methodological contribution is not just “train a critic,” but a specific pipeline: step-wise initial critique, second-pass in-depth critique, merge into long-form deliberate critique, then SFT + RL. The iterative critique and meta-critique idea is reasonably novel relative to simpler distillation or direct verification approaches.

3. **Ablations support the value of the proposed components.**  
   The paper compares against direct distillation and a variant with only initial critiques, showing that direct distillation is weaker and that adding in-depth critique improves performance. This helps justify the design rather than relying only on end-to-end gains.

4. **Strong empirical gains on multiple benchmarks.**  
   The model improves substantially over the Qwen2.5-7B-Instruct base model on MR-GSM8K, PRM800K, and ProcessBench, and it is reported to beat GPT-4o and same-size DeepSeek-R1-Distill baselines on average F1. The paper also includes separate erroneous/correct accuracies, which is helpful because F1 can hide asymmetries.

5. **Tests beyond pass@1, including majority voting and refinement.**  
   The paper does not stop at standalone critic accuracy. It evaluates whether the critic improves generator performance through verified majority voting and critique-based refinement, which is important because the practical value of a critic is downstream utility, not just classification accuracy.

6. **Some attention to robustness and generalization.**  
   The paper includes an analysis of label noise, a binary-search variant for data construction, a self-improvement setting without a stronger teacher, and an extension to subjective summarization. This breadth strengthens the claim that the approach is not overly brittle.

### Weaknesses
1. **The main gains may be partly explained by scale, data quality, and task overlap rather than a fundamentally new critique capability.**  
   The strongest comparisons often involve a 7B critic against much weaker general-purpose instruction models, while the paper also acknowledges that stronger reasoning models like DeepSeek-R1-Distill often “solve the problem directly” rather than critique. This makes it somewhat unclear how much of the improvement comes from deliberate critique versus simply transferring better reasoning ability and using carefully filtered supervision.

2. **Reproducibility is only moderate despite many implementation details.**  
   The paper gives many hyperparameters and dataset statistics, but the crucial pipeline depends on iterative prompting, filtering, manual examples for synthesis, and Monte Carlo label construction with several hidden choices. In particular, some steps are described qualitatively rather than algorithmically, and exact prompts/selection heuristics for the critique synthesis and filtering are only relegated to appendices. For an ICLR audience, the method is likely reproducible in principle, but still somewhat under-specified for faithful replication.

3. **Evaluation is somewhat narrow and benchmark-specific.**  
   The core experiments are confined to mathematical process-verification datasets. Even though the paper adds summarization in an appendix, the main claim is about “general critique ability,” while the evidence is mostly within math reasoning. ICLR reviewers would likely want stronger evidence that the same mechanism transfers beyond one domain family, or at least a clearer limitation statement.

4. **Possible leakage or overfitting concerns in critique generation and evaluation.**  
   The seed critiques are generated using Qwen2.5-72B-Instruct on PRM800K-labeled data, and the same family of models appears throughout the pipeline. This raises the possibility that the model is learning benchmark- and teacher-specific patterns rather than a broadly generalizable critique skill. The paper partly addresses hacking in an appendix, but the overall risk remains.

5. **The claimed “deliberate critique” behavior is not directly measured very rigorously.**  
   The paper shows case studies and GPT-4-based detection of self-correction behavior, but that is still a proxy. There is no strong human evaluation of critique informativeness, correctness of the critique text itself, or whether the generated feedback is actually better for humans to use. For ICLR, claims about behavioral properties often need stronger evidence than model-based meta-evaluation.

6. **Some reported downstream refinement results may be confounded.**  
   The paper notes that DeepSeek-R1-Distill-Qwen-7B sometimes continues solving instead of stopping at the first erroneous step, which can bias refinement. While the authors acknowledge this, it makes comparison on refinement less clean. Similarly, improvements in majority voting may reflect verifier quality but not necessarily better critique explanations.

### Novelty & Significance
**Novelty:** Moderate to good. The most novel aspect is the deliberate construction of critiques through a two-pass initial critique + meta-critique workflow, followed by RL. This is more interesting than straightforward distillation of labels or direct process reward modeling. That said, the work is still largely a training recipe rather than a new model architecture or learning paradigm, and several ingredients resemble existing lines on self-critique, PRMs, and critique fine-tuning.

**Significance:** Fairly high for the narrow domain of mathematical process supervision and scalable oversight. The reported gains are strong, and the idea of making critics more informative for generator refinement is practically useful. For ICLR acceptance standards, the paper looks promising because it addresses an important problem with substantial experiments, but its overall significance is bounded by the domain specificity and the fact that much of the performance improvement may be attributable to careful data curation and RL rather than a broadly generalizable theoretical advance.

**Clarity:** Generally clear at a high level, especially in the motivation and two-stage pipeline. However, the paper becomes hard to follow in the details of data construction, filtering, and the exact mechanics of critique synthesis. The prose sometimes overclaims “deliberate reasoning” without a direct operational definition.

**Reproducibility:** Reasonable but not ideal. The paper provides hyperparameters, dataset sizes, and some appendix prompts, which is good. Still, the exact end-to-end pipeline appears complex and dependent on multiple hidden choices, so exact replication may be difficult.

### Suggestions for Improvement
1. **Add stronger human or expert evaluation of critique quality.**  
   Beyond F1 on first-error identification, evaluate whether the critiques are actually more informative, more useful for humans, and more actionable for generator repair. A small human study would substantially strengthen the “deliberate critique” claim.

2. **Clarify what is genuinely new relative to prior critique and PRM work.**  
   The paper should more explicitly separate its contribution from prior stepwise verification, critique fine-tuning, and self-critique methods. A focused comparison table on training signals, critique format, and inference behavior would help.

3. **Provide a more formal algorithmic description of the data pipeline.**  
   The paper would benefit from pseudocode for seed construction, in-depth critique selection, filtering, and final critique synthesis. This would improve reproducibility and reduce ambiguity.

4. **Expand evaluation beyond math, or narrow the claims.**  
   If the method is meant to be a general critique framework, the paper should present stronger evidence across diverse tasks. If not, the claims should be more carefully scoped to mathematical reasoning and closely related verifiable domains.

5. **Analyze whether gains come from better reasoning or better critique.**  
   Since stronger reasoners can often critique by solving, a useful ablation would explicitly compare critique performance when direct-solving is blocked, penalized, or separately measured. This would isolate the value of the proposed critique training from raw reasoning ability.

6. **Report compute and data-efficiency more explicitly.**  
   Since the method uses a 72B model to generate seed critiques and RL on tens of thousands of examples, it would be useful to quantify training cost, annotation cost, and efficiency relative to baselines. That would help readers judge practical impact under ICLR expectations for scalable methods.

7. **Strengthen the analysis of failure cases.**  
   Include more systematic examples where DeepCritic still fails, especially on harder or adversarial problems, and analyze whether failures are due to misjudgment, shallow critique, or inability to detect subtle step errors.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison to the most relevant recent critique-training baselines, especially Critique-Finetuning, Critic-CoT, CritiqueGRPO, and self-evolving critic variants on the same benchmarks and same base model. Without this, the claim that DeepCritic is a materially better critique-training method is not convincing by ICLR standards.

2. Report compute-matched baselines and data-matched baselines for the RL gains, not just stronger or larger external critics. The paper claims a novel two-stage pipeline, but the improvement could be partly from extra supervision, extra sampling, or RL compute; ICLR reviewers will expect explicit controls for training budget and number of labeled examples.

3. Add a stronger ablation on the seed-data construction pipeline: initial critique only vs. initial+in-depth critique vs. in-depth without meta-critiquing vs. multi-perspective only, all with the same final SFT size. Right now it is unclear which ingredient actually drives the gains, so the core mechanism claim is under-supported.

4. Evaluate on held-out domains beyond the three math error-identification benchmarks using the same model, not just a single summarization appendix experiment. The main claim is about improving critique ability, but the evidence is heavily concentrated in one domain and one output format.

5. Include a stronger external generalization test: train on one math source and test on a different math family or distribution shift setup, e.g. train on GSM8K/PRM800K and test on MATH/Olympiad/Omni-Math with no overlap in construction style. This is needed because current results may reflect benchmark-specific cue learning rather than robust critique competence.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze whether the model is truly critiquing or just solving the problem internally to infer the first error. The paper itself hints at this failure mode for strong reasoning models, so it needs evidence such as correlation with solver correctness, answer leakage rates, and performance on steps where the model cannot solve the problem end-to-end.

2. Quantify the quality of the generated critiques, not only judgment F1. ICLR reviewers will want evidence that the critiques are more informative, more actionable, and less verbose-hallucinated, ideally with human evaluation or structured metrics measuring whether the feedback actually isolates the error cause.

3. Provide calibration and reliability analysis, especially confidence vs. correctness for step judgments. A critique model intended for oversight must be trustworthy, so the paper should show whether its judgments are well-calibrated and where it becomes overconfident or systematically biased.

4. Analyze label-noise sensitivity more rigorously for the auto-labeled RL data. The paper claims robustness, but it does not establish how much noise the method tolerates before the gains disappear; that threshold matters for the scalability claim.

5. Add an error taxonomy showing which kinds of mistakes are improved by deliberate critique and which are not. This would clarify whether the method helps arithmetic slips, logical gaps, algebraic transformations, or only certain easy-to-detect errors.

### Visualizations & Case Studies
1. Show side-by-side trajectories of initial critique, in-depth critique, final critique, and the corrected generator output for representative successes and failures. Without failure cases, the examples only demonstrate cherry-picked wins and do not show when deliberation breaks down.

2. Add a breakdown of performance by step position, solution length, and error type. This would reveal whether the model mainly improves on early steps, short solutions, or trivial mistakes rather than genuinely hard process errors.

3. Visualize the distribution of critique lengths and how they relate to accuracy and refinement success. The paper claims deeper critique helps, so it needs to show whether gains come from useful deliberation or simply longer outputs.

4. Include failure case studies where DeepCritic gives a confident but wrong critique, especially on problems that require symbolic manipulation or nonlocal reasoning. These examples would expose the limits of the method and test whether the improved F1 hides brittle behavior.

5. Show a comparison of generated feedback quality when used for refinement by a weak generator versus a strong generator. This would reveal whether the critic is actually useful as supervision or only looks good on the easier generator setting.

### Obvious Next Steps
1. Evaluate whether DeepCritic improves downstream RL or search when used as the critic inside a full reasoning pipeline, not just as a standalone verifier or refinement helper. That is the most direct test of whether the method matters for scalable oversight.

2. Test on non-math verifiable tasks where step-level critique is harder, such as code debugging or tool-using tasks. The paper’s core claim is about critique ability in general, and ICLR will expect evidence that the approach is not narrowly tuned to math formatting.

3. Compare against training the same base model on more standard supervised critique data of equal size, to isolate whether deliberate critique structure matters beyond just more data. This would directly answer whether the proposed pipeline adds anything beyond conventional fine-tuning.

4. Run a human study on whether the generated critiques are actually more helpful for correcting mistakes than baseline critics. Since the ultimate motivation is supervision, human usefulness is a necessary validation, not just an optional appendix result.

5. Release an explicit protocol for preventing answer leakage and self-solving during critique, then re-evaluate with that protocol. Given the acknowledged leakage issue, this is necessary before the paper’s claims about “true critique” can be trusted.

# Final Consolidated Review
## Summary
This paper proposes DeepCritic, a two-stage pipeline for training LLMs to produce more deliberate step-wise critiques of math solutions. The method first curates a relatively small set of long-form critiques via iterative prompting and meta-critique, then further improves the critic with RL on either human-labeled PRM data or automatically constructed labels from rollout-based correctness estimation.

## Strengths
- The paper tackles an important and timely problem: critics for scalable oversight in mathematical reasoning. The central motivation is well aligned with current interest in process supervision and test-time verification, and the paper is careful to evaluate not just standalone judgment accuracy but also downstream utility for refinement and verified majority voting.
- The iterative critique-generation recipe is a concrete and nontrivial contribution. The paper does more than distill labels: it explicitly separates initial critique, in-depth critique, and final synthesis, and the ablations support that this structure matters. In particular, DirectDistill < InitialCritic < DeepCritic on the main table is a meaningful indication that the meta-critique step is doing real work.

## Weaknesses
- The strongest evidence still does not fully justify the paper’s broad claims of superior critique capability. On the main error-identification table, the strongest PRM baseline, Qwen2.5-Math-PRM-7B, remains better than DeepCritic on the reported averages, so the paper is not “best overall” despite some of the framing. The correct claim is narrower: DeepCritic is strong among LLM critics and improves utility for refinement.
- The method is heavily dependent on a powerful 72B teacher model, prompt engineering, and aggressive filtering. This makes the pipeline look more like a carefully engineered data-curation recipe than a general critique-learning principle. The strict retention rule that keeps only solutions where the in-depth judgments match ground truth also introduces selection bias toward examples the teacher can already handle consistently, which limits how confidently we can interpret the resulting gains as a general critique capability.
- The evaluation is still too narrow for the strength of the claims. Almost everything is on math process-verification benchmarks, with a small appendix summarization study that is not enough to support a broad “general critique” narrative. The paper also lacks statistical significance, calibration, and systematic failure analysis, which matters because several reported gains are meaningful but not always large.
- The downstream refinement results are promising but somewhat confounded. In particular, the paper explicitly notes that DeepSeek-R1-Distill-Qwen-7B sometimes keeps solving instead of stopping at the first erroneous step, which muddies comparisons in critique-based refinement. This makes some practical claims less clean than the headline numbers suggest.

## Nice-to-Haves
- A more formal, algorithmic pseudocode version of the data construction pipeline, including the filtering and selection steps.
- A human evaluation of critique usefulness and informativeness, not just first-error F1.
- More systematic robustness analysis across error types, step positions, and solution lengths.
- Explicit compute/data-efficiency accounting, since the method relies on a large teacher and multiple rounds of sampling.

## Novel Insights
The most interesting aspect of the paper is that it tries to teach “deliberation” itself, not just correctness labels. The initial critique plus in-depth critique setup is a plausible way to induce a model to check its own reasoning from multiple perspectives and even critique its prior critique, and the ablations suggest that this richer supervision is more valuable than simply distilling direct judgments. That said, the gains still look strongly tied to high-quality teacher-generated supervision and RL, so the novelty is more in the training recipe and data construction than in a fundamentally new notion of critique.

## Potentially Missed Related Work
- Critique-Finetuning — relevant recent work on learning to critique, useful for comparing whether deliberate critique formatting adds beyond standard critique training.
- Critic-CoT — relevant because it also studies chain-of-thought style critique and step-level feedback.
- CritiqueGRPO — relevant RL-based critique training baseline that should be compared on the same benchmarks and base model.
- Self-evolving critic / scalable oversight variants — relevant because the paper’s claims overlap heavily with self-improvement and critic-driven supervision.
- None identified beyond these critique-training lines for the core claims.

## Suggestions
- Tighten the claims: position DeepCritic as a strong method for improving math critics, not as a universal or clearly SOTA critique paradigm.
- Add direct same-backbone comparisons to the most relevant recent critique-training methods, with compute- and data-matched controls.
- Include at least one human study or expert annotation of critique quality and actionability.
- Provide failure cases and an error taxonomy so readers can see when deliberate critique helps and when it does not.
- Make the pipeline reproducible with pseudocode and clearer description of prompt/selection heuristics.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 4.0]
Average score: 4.7
Binary outcome: Reject
