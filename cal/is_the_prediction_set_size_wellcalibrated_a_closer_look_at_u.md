=== CALIBRATION EXAMPLE 47 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate: the paper is about whether prediction set size in conformal prediction is calibrated as an uncertainty measure.
- The abstract clearly states the problem, method, and main empirical claim, but it overstates the level of establishment of the new notion. Phrases like “answers this open question” and “we reveal their weak calibration” are stronger than what the paper actually shows, because the evidence is empirical and depends heavily on a particular sampling-based definition of accuracy.
- The abstract also claims “a theoretical analysis of the predictive distributions,” but the theory is quite limited and assumption-driven (e.g., Dirichlet modeling in Theorem 4.2), so the wording risks implying broader theoretical support than is actually provided.
- The claim that CPAC is demonstrated on “models including ResNet, Vision Transformer and GPT-2” is supported by the experiments, but the abstract does not mention that the calibration improvements are mixed and often come with accuracy/PSS tradeoffs.

### Introduction & Motivation
- The motivation is reasonable: conformal prediction guarantees coverage, but set size is often used informally as an uncertainty proxy, and the paper asks whether that proxy is calibrated. That is a genuine and interesting question for ICLR.
- The gap in prior work is somewhat overstated. The introduction claims the paper is “the first attempt to systematically investigate the calibration of CP on classification tasks,” but there is related work on uncertainty calibration, selective prediction, conformal efficiency, and even self-calibration/temperature-scaling for conformal prediction. The novelty is more specific: calibration of PSS as a reliability signal.
- The introduction’s contribution statements are clear, but the narrative blurs together three different objects: coverage, prediction-set size, and accuracy of a sampled label from the set. These are related but not equivalent. The paper should more carefully justify why this sampled-accuracy notion is the right target for “uncertainty calibration.”
- The introduction also risks over-claiming practical impact. CPAC is presented as a “pre-processing step” that improves calibration, but later results show it can worsen accuracy and increase PSS under fixed-coverage comparisons, which complicates the story.

### Method / Approach
- The method is described in a way that is partially reproducible, but there are important ambiguities.
- The central definition in Section 4.1 is problematic: the paper defines CP calibration by sampling a label from the prediction set and asking whether the resulting accuracy decreases monotonically with PSS. This is not standard calibration of uncertainty, and the justification for this specific operationalization is not fully convincing. It depends on a chosen sampling temperature \(t\), which makes the “calibration” target partly an artifact of the sampling rule rather than an intrinsic property of CP.
- The relation between prediction set size and accuracy is not directly observable unless one first introduces multinomial sampling from the set. That makes the proposed metric somewhat indirect: the paper is measuring calibration of a sampling-based proxy, not of the conformal set itself.
- Theorem 4.2 and the associated power-law target \(f(k)=1/k^\tau\) are too weakly grounded for the strength of the claims. The theorem relies on a Dirichlet assumption with shared mass vector \(\mathbf a\), but the proof and the theorem statement are hard to parse and appear to oversimplify the relationship between expected accuracy and PSS. In particular, the exact exponent \(\tau\) is tied to \(\log_K \sum_j a_j^2\), which is more a descriptive quantity about concentration than a general calibration law.
- The paper acknowledges alternative target functions, but the justification for selecting the power function is mostly empirical. If the target is part of the core method, the paper should more clearly separate theoretical motivation from heuristic fitting.
- CPAC is described as a bilevel optimization over a linear transform \((W,b)\) before quantile computation, but the practical optimization procedure is not fully justified. It is unclear how stable this is, how sensitive it is to hyperparameters, and whether the transformation preserves or meaningfully distorts the model’s ordering in ways that affect APS behavior.
- A key potential failure mode is not discussed: CPAC can change the score distribution and thus alter both coverage and the semantics of the prediction set. The method is framed as “pre-processing,” but in effect it is a learned post-hoc calibration layer tightly coupled to the conformal threshold. This makes the relationship to standard CP less transparent.
- The decision to exclude empty prediction sets by setting the minimum PSS to 1 is understandable, but it should be discussed more explicitly because it affects the calibration target, especially for low-confidence examples.
- For the theory, there are signs of incompleteness or limited generality: the Dirichlet analysis is presented as illustrative, not a rigorous characterization of real deep model outputs. That is fine, but the paper sometimes uses it to support broader claims than warranted.

### Experiments & Results
- The experiments do test the central claim that PSS and sampled accuracy are not perfectly calibrated, and they examine this across multiple datasets and model families. That is a strength.
- The main empirical evidence is in the reliability diagrams and the CP-ECE/Uni-CP-ECE tables. These do support the claim that the relation between PSS and sampled accuracy is often weakly calibrated.
- However, the evaluation is not fully aligned with the paper’s broader claims. Since calibration is defined via multinomial sampling from the prediction set, the experimental validation should more explicitly analyze sensitivity to the sampling temperature \(t\), but this is only partially explored.
- The baselines are somewhat limited. The comparisons are mostly against standard APS/Platt scaling (“PS”) and a variant (“PS-Full”), but there is little comparison to other conformal calibration or set-construction strategies that might better preserve the PSS-accuracy relation. Given the paper’s claims, additional baselines would materially strengthen the case.
- There are missing ablations that matter:
  - sensitivity of CPAC to the sampling temperature \(t\),
  - sensitivity to the regularization \(\lambda\),
  - effect of optimizing only a scalar temperature versus full \(W,b\),
  - effect of the low-PSS subset threshold used in CPAC,
  - whether the gains depend on the specific APS conformalization choice.
- The metric choice is debatable. The “uniform CP-ECE” weights all PSS bins equally, which the paper motivates via fairness-like reasoning. That is a defensible design choice, but the paper should be clearer that this is not the only reasonable metric and that the conclusion about “weak calibration” depends on it. Standard CP-ECE and uniform CP-ECE sometimes move differently across tables.
- Error bars are reported, which is good. But the results are still hard to interpret because the tables are large and the summary claims often emphasize one metric while another metric moves in the opposite direction. For example, CPAC often reduces Uni-CP-ECE while increasing PSS, and sometimes slightly reducing accuracy. The paper should be more candid about these tradeoffs.
- Some of the reported improvements look modest relative to the added method complexity. On several settings, CPAC changes Uni-CP-ECE by a small amount, and the calibration gains do not always clearly outweigh the cost in accuracy or prediction-set inflation.
- The experiments on ImageNet perturbations are useful, but the perturbation study is somewhat narrow: synthetic noise/blur/dropout are not enough to establish robustness of the proposed calibration notion under real distribution shift.
- The GPT-2 topic classification setting is interesting and helps broaden the scope, but it is still a relatively narrow NLU benchmark. Also, the way embedding noise and typos are introduced is not enough to validate a general claim about language-model conformal calibration.
- The paper claims CPAC strengthens calibration “without sacrificing accuracy or even improve it,” but the tables show this is not consistently true. In several cases accuracy drops or PSS rises substantially under CPAC, especially in the fixed-coverage comparisons.
- ICLR typically expects a crisp empirical story with clear methodological controls; here, the large number of settings and metrics makes it hard to assess whether the core improvement is robust or partially tuned.

### Writing & Clarity
- The paper has a real conceptual contribution, but the exposition is often hard to follow in the method section because the formalization of calibration via sampled labels is not cleanly separated from the empirical setup.
- The relationship among APS, temperature scaling, multinomial sampling, CP-ECE, and CPAC is not always easy to disentangle. In particular, the reader has to infer when the paper is talking about the conformal set, when it is talking about the sampled point prediction, and when it is talking about the transformed logits.
- The theorem statement and proof presentation are not sufficiently clear to support the weight placed on them. Even allowing for parser artifacts, the intended mathematical claim should be stated more cleanly and with a clearer interpretation.
- The figures appear intended to show reliability diagrams, which is appropriate, but the paper relies heavily on visual intuition without always explaining exactly how to read the axes or why a given diagram supports the claim. The narrative around Figures 2–6 could be more precise.
- The tables are informative but too dense to be easily interpretable without careful cross-reading. The key takeaways should be summarized more explicitly in the text.

### Limitations & Broader Impact
- The paper does acknowledge one limitation: the convergence and generalization of the bilevel optimization are not theoretically analyzed.
- But it misses several important limitations that matter for an ICLR paper:
  - The calibration notion is sampling-dependent and not unique.
  - The target curve is only weakly justified outside the proposed distributional assumptions.
  - CPAC changes the model after training using a calibration set, which may not preserve the semantics of the original conformal procedure.
  - The method’s behavior under dataset shift is not well characterized beyond synthetic perturbations.
- A broader impact discussion is absent. That is not necessarily fatal, but the paper should at least discuss that calibration can be misleading if the chosen sampling rule or target curve does not reflect downstream decision-making.
- There is also a fairness-related point the paper hints at when motivating uniform CP-ECE, but it does not engage with whether equal-weight binning actually corresponds to fairness in any operational sense.

### Overall Assessment
This paper addresses an interesting and nontrivial question: whether prediction-set size in conformal prediction is itself well calibrated as an uncertainty signal. That question is relevant for ICLR, and the empirical study is broad enough to be taken seriously. However, the main conceptual move—defining calibration through multinomial sampling from the prediction set—makes the notion somewhat indirect and dependent on a chosen sampling temperature, which weakens the universality of the claim. The theory is suggestive but not strong enough to justify the proposed target curve beyond a stylized model. CPAC is a plausible calibration post-processing method, but the empirical gains are mixed and come with tradeoffs in accuracy and set size, and the baselines/ablations are not yet sufficient to make the case airtight. My overall view is that the paper is promising and potentially publishable, but it still needs a clearer conceptual framing and a more rigorous experimental argument to meet ICLR’s stronger bar for a broadly convincing contribution.

# Neutral Reviewer
## Balanced Review

### Summary
This paper studies a less explored question in conformal prediction for classification: whether prediction set size (PSS), which CP uses to convey uncertainty, is actually calibrated with respect to predictive correctness. The authors define a notion of CP calibration based on the relationship between PSS and accuracy under multinomial sampling, propose a power-law target curve, and introduce a preprocessing calibration method (CPAC) via bi-level optimization intended to improve this alignment while preserving CP coverage.

### Strengths
1. **Addresses an underexplored but relevant question for ICLR.**  
   The paper targets an important practical issue: CP provides coverage guarantees, but the authors argue that set size may still be poorly aligned with correctness. This is a meaningful uncertainty-calibration question, especially for modern classifiers and foundation models where decision reliability matters.

2. **Connects calibration to prediction-set size rather than confidence.**  
   The work makes a conceptual contribution by shifting calibration analysis from standard confidence scores to CP’s set-size signal. This is novel in framing and could be useful because CP is increasingly used as a black-box uncertainty wrapper across domains.

3. **Empirical study spans multiple architectures and corruptions.**  
   The experiments cover ResNet, ViT-B/ViT-L, and GPT-2 across CIFAR100, ImageNet, and a topic classification task, with noise, blur, dropout, and typo perturbations. This breadth helps support the claim that CP calibration issues are not isolated to one model family.

4. **The proposed method is lightweight in spirit and compatible with pretrained models.**  
   CPAC is presented as a preprocessing step before quantile computation, which makes it easier to integrate with existing split-conformal pipelines than methods that require retraining from scratch. The paper also emphasizes use on pretrained models, which is practically appealing.

5. **The paper includes ablations and alternative target curves.**  
   The comparison among power, exponential, and logarithmic target functions is useful and shows some care in validating the chosen target. The paper also reports results under both standard and uniform CP-ECE, which gives a more nuanced view than a single metric.

### Weaknesses
1. **The main formulation is not yet fully convincing or rigorously justified.**  
   The paper defines calibration through multinomial sampling from the prediction set and then relates expected accuracy to PSS. However, this is a somewhat indirect construction: CP set size is not naturally a probabilistic forecast, so the choice of sampling-based accuracy may be seen as heuristic rather than principled. For ICLR, this weakens the theoretical contribution unless the link to decision reliability is more clearly justified.

2. **The theoretical result appears limited and partly inconsistent with the framing.**  
   Theorem 4.2 relies on a strong Dirichlet assumption and seems to derive a power-law relationship that is more illustrative than general. The paper itself notes that the Dirichlet/logistic-normal assumptions are only instantiations, which suggests the theory does not tightly support the general calibration definition. That makes the theorem feel more like motivation than a substantial guarantee.

3. **The optimization method is underexplained and likely fragile.**  
   CPAC is described as a bi-level optimization with a full weight matrix and bias, but the paper does not clearly explain convergence behavior, sensitivity to hyperparameters, or why optimizing the entire classifier is preferable to simpler calibration transforms. The authors also acknowledge that convergence and generalization are only empirically validated.

4. **Coverage-preserving behavior is not fully transparent.**  
   Since CPAC modifies logits before conformal quantile computation, it is important to understand precisely how coverage is maintained under this transformation and whether any coverage degradation occurs beyond the reported averages. The experimental tables show coverage values, but the method’s interaction with the CP guarantee is not deeply analyzed.

5. **Metric design may be somewhat arbitrary.**  
   Uniform CP-ECE is motivated as fair across PSS bins, but the paper does not fully justify why it should be the preferred primary metric over standard weighted CP-ECE. In some settings, these two metrics move differently, and the practical meaning of “better calibrated” can become ambiguous.

6. **Experimental gains are mixed and sometimes come with trade-offs.**  
   CPAC often improves uniform CP-ECE, but not always standard CP-ECE, and in several tables accuracy drops while PSS and/or coverage move in undesirable directions. For example, on some noisy settings, reducing one calibration metric appears to increase set size substantially. This makes the practical benefit less clear-cut than the narrative suggests.

7. **The paper would benefit from stronger baselines and comparisons.**  
   The comparisons are mainly against standard APS/Platt scaling variants and target-curve choices. For ICLR, reviewers would likely expect stronger baselines, including other calibration methods tailored to uncertainty sets, alternative post-hoc calibration schemes, and perhaps direct comparisons to recent conformal-efficiency or confidence calibration methods adapted to the same setting.

8. **Clarity is uneven.**  
   The exposition is ambitious but sometimes difficult to follow, especially in the transition from the intuitive notion of “set-size calibration” to the exact sampling-based definition and then to the optimization objective. Some claims are also stated broadly, while the precise assumptions under which they hold are narrow.

### Novelty & Significance
**Novelty:** Moderate. The idea of studying calibration for conformal prediction through prediction-set size is fairly original, especially in classification and with pretrained deep models. However, the technical mechanisms—temperature scaling, post-hoc calibration, and bi-level optimization—are built from familiar components, so the novelty is primarily in problem formulation and empirical investigation rather than in a deeply new algorithmic paradigm.

**Significance:** Moderate. If the formulation proves robust, it could be useful for practitioners who use CP as an uncertainty interface and want better alignment between set size and correctness. That said, the current evidence suggests the phenomenon is real but the proposed fix is not yet fully compelling or broadly validated at the level usually expected for a strong ICLR acceptance.

**Clarity:** Mixed. The high-level story is understandable, but the exact definitions and theoretical connection are somewhat hard to parse. The paper would benefit from a tighter narrative and a clearer separation between motivation, formal claims, and empirical findings.

**Reproducibility:** Fair but incomplete. The paper provides many dataset/model details and hyperparameters, and reports multiple seeds, which is good. However, the bi-level optimization procedure, sampling details, target tuning, and implementation specifics are not fully transparent, making exact reproduction of CPAC somewhat challenging.

**ICLR acceptance-bar assessment:** Interesting and potentially publishable as a calibration-oriented study, but in its current form it feels borderline for ICLR. ICLR typically expects either a clearly compelling methodological advance with strong evidence, or a sharp empirical/diagnostic insight supported by rigorous analysis. This paper has a useful diagnostic angle, but the theoretical support and methodological justification are not yet strong enough to make the contribution fully convincing.

### Suggestions for Improvement
1. **Strengthen the conceptual justification for sampling-based calibration.**  
   Explain more clearly why multinomial sampling from the prediction set is the right proxy for correctness, and relate it to downstream decision-making. Consider adding alternative operationalizations of CP uncertainty calibration and showing that the conclusions are robust to them.

2. **Tighten and broaden the theory.**  
   Either strengthen Theorem 4.2 into a more general result or clearly label it as a motivating approximation. It would help to derive conditions under which the proposed power-law target is expected to hold, and to discuss failure modes when those conditions are violated.

3. **Add stronger and more diverse baselines.**  
   Compare against additional post-hoc calibration methods, possibly adapted to CP outputs, and against more recent conformal variants. For the optimization method, include simpler baselines such as isotonic regression, monotonic spline fitting, or direct calibration of set-size-to-accuracy curves.

4. **Analyze the effect on coverage more carefully.**  
   Since CP is valued for valid coverage, include a deeper discussion of how CPAC affects the finite-sample guarantee, whether coverage is preserved exactly or only approximately in practice, and whether there are settings where the calibration transform hurts validity.

5. **Report sensitivity analyses.**  
   Study sensitivity to the sampling temperature, target exponent τ, regularization λ, calibration-set size, and number of optimization rounds. This would help determine whether CPAC is stable or heavily tuned.

6. **Improve the presentation of the objective and algorithm.**  
   The bi-level optimization would benefit from a cleaner derivation, clearer notation, and a concise explanation of what is optimized at each stage. A reader should be able to reimplement CPAC without reconstructing several implicit steps.

7. **Clarify when uniform vs. standard CP-ECE should be used.**  
   Provide a stronger rationale for the unweighted metric and perhaps include a small user study or decision-theoretic argument showing why it is the more meaningful measure for set-size calibration.

8. **Discuss practical trade-offs more honestly.**  
   Highlight cases where CPAC improves calibration but increases set size or reduces accuracy, and explain which operating regimes favor the method. A balanced discussion would make the empirical contribution more credible.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add comparisons against stronger calibration baselines beyond standard Platt scaling APS, including temperature scaling-only CP, vector/matrix scaling, and recent conformal calibration variants or selective-set calibration methods. Without these baselines, it is not convincing that CPAC is better than simpler post-processing or existing CP-specific calibration approaches.

2. Evaluate on additional datasets and label spaces where set-size behavior could differ substantially, especially non-ImageNet vision benchmarks and at least one more language task. ICLR reviewers will expect evidence that the claimed “systematic” phenomenon is not an artifact of CIFAR100/ImageNet/topic classification.

3. Report performance across multiple miscoverage levels \(\alpha\), not just the default 90% regime. The calibration target and PSS behavior may change with coverage level, so the method’s generality is not established unless it is tested at several \(\alpha\) values.

4. Compare against training-time alternatives that can affect prediction-set quality, such as direct calibration of logits before conformalization and selective prediction / abstention baselines. Without this, it is unclear whether CPAC is uniquely needed versus simply improving the base model or thresholding strategy.

5. Include a robustness experiment on out-of-distribution shifts or corrupted calibration/test splits that are not tuned to the same perturbation family. The method claims improved reliability, but current perturbations are mostly synthetic and may not reflect whether the calibration survives realistic shift.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify whether the proposed PSS-to-accuracy relationship is actually monotonic and stable across classes, seeds, and datasets, not just on average. The core claim is about calibration of uncertainty; without subgroup-wise and per-seed consistency, the reported global curve can hide severe failures.

2. Analyze whether the power-law target \(f(k)=1/k^\tau\) is theoretically and empirically identifiable, or whether \(\tau\) is just a flexible fit parameter. As written, the target could be overfit to test curves; the paper needs evidence that \(\tau\) generalizes and is not a retrospective curve-fitting knob.

3. Provide a clear sensitivity analysis for CPAC hyperparameters: sampling temperature, regularization \(\lambda\), optimization rounds, batch size, and the low-PSS filtering threshold. The method depends on several design choices, and without robustness checks it is hard to trust that gains are not brittle.

4. Analyze the trade-off between improving CP-ECE and harming coverage, accuracy, or set size distribution. The current results suggest coverage is roughly preserved, but the method changes set size and sometimes lowers accuracy; the paper needs a principled discussion of when the calibration improvement is worth the cost.

5. Explain why the bi-level optimization is stable and not simply overfitting the calibration set, ideally with learning curves and calibration/test gap measurements. The paper currently asserts empirical success but does not establish that the optimization behaves reliably under small calibration sets.

### Visualizations & Case Studies
1. Add per-sample scatter plots of PSS versus correctness probability before and after CPAC, with points colored by class or confidence regime. This would show whether CPAC truly sharpens the correlation or merely shifts the marginal distribution.

2. Show failure cases where large prediction sets still produce high-confidence wrong decisions, and small sets still fail. These counterexamples are necessary to test the paper’s central premise that PSS meaningfully tracks reliability.

3. Provide calibration plots broken down by class frequency, difficulty, and corruption severity. If CPAC mainly helps easy or frequent classes, the claimed improvement in uncertainty calibration is much weaker.

4. Include before/after histograms of PSS and coverage-conditioned PSS, not just average PSS. This would reveal whether CPAC improves calibration by genuinely restructuring uncertainty or by collapsing the distribution in a way that may be undesirable.

### Obvious Next Steps
1. Extend the method to non-classification conformal settings, especially regression and structured prediction, or explicitly justify why the proposed notion of uncertainty calibration is classification-specific. If the paper claims a general perspective on CP uncertainty, this is the most natural next step.

2. Develop an end-to-end conformal calibration method that avoids tuning on the test set or requiring multiple grid searches for \(\tau\) at evaluation time. The current evaluation protocol looks closer to post-hoc fitting than a deployable calibration method.

3. Provide a theoretical guarantee linking the bi-level objective to improved calibration error under finite-sample conformalization. The paper currently proposes a heuristic optimizer, but the main contribution would be stronger if the calibration objective were tied to measurable guarantees.

# Final Consolidated Review
## Summary
This paper asks whether conformal prediction’s prediction set size (PSS) is itself a well-calibrated uncertainty signal for classification. It introduces a sampling-based notion of CP calibration, proposes a power-law target curve for PSS-to-accuracy alignment, and then uses a bi-level calibration procedure (CPAC) to improve this alignment on several vision and language benchmarks.

## Strengths
- The paper tackles a genuinely underexplored question: CP is usually evaluated for coverage and efficiency, but the reliability of set size as an uncertainty signal is much less studied. The diagnostic framing is relevant and potentially useful for practitioners using conformal prediction as a decision interface.
- The empirical study is broad enough to be convincing that the phenomenon is real. The authors test ResNet, ViT-B/ViT-L, and GPT-2 on CIFAR100, ImageNet-1k, and topic classification, and they consistently show that PSS-to-accuracy alignment is not especially tight under their reliability-diagram view.

## Weaknesses
- The core calibration definition is indirect and not fully convincing. The paper does not calibrate the conformal set itself in a standard sense; instead it samples a label from the set using a temperature-controlled multinomial rule and then defines calibration in terms of sampled accuracy versus PSS. That makes the notion depend on an extra sampling mechanism that is not intrinsic to CP and weakens the universality of the claim.
- The theoretical support is limited and heavily assumption-driven. The Dirichlet-based derivation for the power-law target is mostly illustrative, not a general result about deep classifiers or conformal prediction. The paper leans on this theory to motivate a core design choice, but the assumptions are too stylized to bear that weight.
- The proposed CPAC method is not yet well-justified as a robust algorithm. It optimizes a full linear transform before conformal quantile computation, but there is little evidence of stability, sensitivity analysis, or any guarantee beyond empirical behavior. In addition, the method introduces extra tuning knobs and still shows trade-offs in accuracy and PSS.
- The empirical story is mixed, not uniformly positive. CPAC often improves uniform CP-ECE, but the improvements are modest in many settings and frequently come with higher prediction-set sizes and occasional accuracy drops. The paper’s broader implication that CP can be made “better calibrated” without meaningful downside is not supported by the tables.
- The evaluation leaves important questions unanswered. There is little comparison against stronger calibration/post-processing baselines, and the main results are shown mostly at one operating regime. The dependence on sampling temperature, the target exponent, the calibration-set size, and the low-PSS filtering choice is not adequately explored.

## Nice-to-Haves
- A clearer comparison between standard CP-ECE and uniform CP-ECE, including when each one is the right metric to care about.
- More sensitivity plots for CPAC hyperparameters, especially sampling temperature, regularization, optimization rounds, and the low-PSS threshold.
- Additional experiments across multiple miscoverage levels to show the phenomenon is not specific to the default 90% setting.

## Novel Insights
The most interesting insight is that conformal prediction’s set size can look reasonable from a coverage perspective while still being poorly aligned with downstream correctness once one asks how the set would actually be used for a decision. The paper’s own results suggest an important nuance: the same conformal wrapper can preserve nominal coverage while exhibiting weak or even inconsistent PSS-to-accuracy calibration, meaning that “valid” does not automatically imply “reliable” in a decision-theoretic sense. That is a useful diagnostic observation, even if the proposed remedy is still somewhat heuristic.

## Potentially Missed Related Work
- van der Laan & Alaa (2024) — relevant because it studies self-calibration/conformal calibration, though in regression rather than classification.
- Xi et al. (2024); Dabah & Tirer (2024) — relevant because they analyze temperature scaling in conformal prediction, which is close to the calibration machinery used here.
- Huang et al. (2024) — relevant because it studies rank-calibration for language models, a neighboring notion of uncertainty calibration.
- Recent selective prediction / conformal-efficiency work — relevant as stronger baselines for set-quality and reliability, though not directly cited in the paper’s experiments.

## Suggestions
- Replace the current calibration definition with a more direct operationalization, or at least show that the main conclusions hold under multiple plausible definitions of CP uncertainty calibration.
- Add stronger baselines and a more transparent sensitivity study; without that, it is hard to tell whether CPAC is genuinely useful or just a somewhat tuned reparameterization of APS.
- Present the theory more modestly: as motivation for a target curve, not as broad justification for a general calibration law.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 2.0, 2.0]
Average score: 3.5
Binary outcome: Reject
