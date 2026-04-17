# Emergent Bayesian Behaviour and Optimal Cue Combination in LLMs

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Large language models (LLMs) excel at explicit reasoning, but their implicit computational strategies remain underexplored. Decades of psychophysics research show that humans intuitively process and integrate noisy signals using near-optimal Bayesian strategies in perceptual tasks. We ask whether LLMs exhibit similar behaviour and perform optimal multimodal integration without explicit training or instruction. Adopting the psychophysics paradigm, we infer computational principles of LLMs from systematic behavioural studies. We introduce a behavioural benchmark - BayesBench: four magnitude estimation tasks (length, location, distance, and duration) over text and image, inspired by classic psychophysics, and evaluate a diverse set of nine LLMs alongside human judgments for calibration. Through controlled ablations of noise, context, and instruction prompts, we measure performance, behaviour and efficiency in multimodal cue-combination. Beyond accuracy and efficiency metrics, we introduce a Bayesian Consistency Score that detects Bayes-consistent behavioural shifts even when accuracy saturates. Our results show that while capable models often adapt in Bayes-consistent ways, accuracy does not guarantee robustness. Notably, GPT-5 Mini achieves perfect text accuracy but fails to integrate visual cues efficiently. This reveals a critical dissociation between capability and strategy, suggesting accuracy-centric benchmarks may over-index on performance while missing brittle uncertainty handling. These findings reveal emergent principled handling of uncertainty and highlight the correlation between accuracy and Bayesian tendencies. We release our psychophysics benchmark and consistency metric as evaluation tools and to inform future multimodal architecture designs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates whether LLMs spontaneously exhibit human-like, near-optimal Bayesian strategies for multimodal integration without explicit instruction. The authors introduce BayesBench, a psychophysics-inspired benchmark featuring four magnitude estimation tasks across text and image modalities. By systematically manipulating noise and context, they evaluate nine LLMs on accuracy, cue-combination efficiency, and behavioral adaptation using a novel Bayesian Consistency Score (BCS). The findings reveal that high accuracy does not always imply efficient cue combination. However, more capable models, such as Llama-4 Maverick and Claude 3.7 Sonnet, demonstrate emergent behaviors that are consistent with Bayesian observer models, suggesting that principled uncertainty handling may be an emergent property of large-scale training.

### Strengths
1. Applying the classic psychophysics research framework to evaluate LLMs, this interdisciplinary method treats the models as "black-box observers" for systematic behavioral analysis, offering a novel perspective on the internal computational strategies they use to handle uncertainty.

2. BayesBench benchmark constructed by the authors not only evaluates task accuracy (NRMSE) but also measures cue integration efficiency and behavioral adaptation through Relative Error (RRE) and the innovative Bayesian Consistency Score (BCS).

### Weaknesses
1. The study's four magnitude estimation tasks are based on artificial and simplified scenarios such as line ratios as ASCII text, non-self-intersecting maze paths. These "toy" tasks fail to provide sufficient evidence that LLMs would apply the same Bayesian principles when processing the ambiguity and complexity inherent in real-world multimodal data.
2. While the research demonstrates that high-performance LLMs exhibit emergent Bayesian behavior, it fails to deeply analyze the origins of this emergence. For example, by not comparing models with different training objectives, it cannot validate its hypothesis that Bayesian strategies represent a universal solution derived from information-theoretic constraints.
3. The noise ablation study is narrowly focused, applying only Gaussian blur to the image modality. By neglecting to introduce noise to the text modality or explore other noise types, the study cannot determine whether the LLMs' strategies are consistent across different modalities and under diverse noise conditions.

### Questions
First, regarding generalizability, it might be valuable to explain how the behaviors observed in synthetic tasks could transfer to the ambiguity of real-world data—this thought arises partly from the limited scope of the noise study. Since optimal integration calls for flexibly re-weighting cues, showing how models perform with noisy text or conflicting cross-modal information could also help reinforce your claims. 
Second, when it comes to the emergence of this behavior, your “universal solution” hypothesis is quite compelling, though additional direct evidence would enhance its persuasiveness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates whether LLMs implicitly develop Bayesian computational strategies for processing uncertainty, similar to those observed in human perception. The authors introduce BayesBench, a novel benchmark inspired by classic psychophysics, which consists of four magnitude estimation tasks (length, location, distance, duration) across text and image modalities. To evaluate model behavior, they propose a new metric, the Bayesian Consistency Score (BCS), alongside standard accuracy (NRMSE) and cue-combination efficiency (RRE) metrics. The BCS is designed to measure whether a model's behavior shifts in a Bayes-consistent direction in response to controlled ablations (e.g., noise, context changes). The key findings are :(1) more capable models, such as GPT-5 Mini, Llama-4 Maverick, and Claude 3.7 Sonnet, often exhibit Bayes-consistent adaptations, and (2) high task accuracy (as seen in GPT-5 Mini) does not necessarily correlate with efficient or optimal multimodal cue combination.

### Strengths
The paper's primary strength lies in its originality and interdisciplinary approach. Applying the rigorous, time-tested paradigm of psychophysics to probe the implicit computational strategies of LLMs is a highly novel and insightful direction, moving beyond standard accuracy-based evaluations.

The methodological contribution, BayesBench, is solid. It provides a controllable and reproducible framework for testing how models handle uncertainty. The use of controlled ablations (noise, context, steering) is a systematic way to probe behavioral shifts.

Furthermore, the introduction of the Bayesian Consistency Score (BCS) is a clever conceptual advance. By focusing on the direction of behavioral change in response to perturbations (e.g., how the prior weight shifts) rather than just the static goodness-of-fit to a single Bayesian model, the BCS attempts to capture a more principled, adaptive strategy.

Finally, the paper delivers a valuable and nuanced finding: the decoupling of task accuracy from cue-combination efficiency. The observation that a highly accurate model like GPT-5 Mini can be an inefficient multimodal integrator is an important insight for the future design and evaluation of robust multimodal systems.

### Weaknesses
Despite the novel premise, the paper's central claim—that LLMs exhibit "emergent Bayesian behaviour"—is not adequately supported, as it rests on several questionable assumptions and interpretations.

The most significant weakness is the conflation of "regression-to-the-mean" with Bayesian inference. The primary evidence for Bayesian processing is the regression effect shown in Figure 1, where estimates are biased toward the center of the stimulus range. While this pattern is consistent with Bayesian integration (a prior pulling the likelihood), it is not sufficient evidence. Many simpler, non-Bayesian heuristics, such as anchoring on the mean of the current session's stimuli, could produce an identical pattern. The paper fails to test or rule out these more parsimonious alternative explanations.

This issue is compounded by a major confounding variable: in-context learning (ICL). The experimental setup explicitly provides a "rolling context" of previous trials and responses, which the authors state is the "basis of the emergence of Bayesian consistent behaviour". This design makes it impossible to distinguish between a genuinely "emergent" computational strategy and the model simply performing standard ICL on the provided history. The "Static Bayesian observer" or "Sequential Bayesian observer" might just be effective descriptions of a model learning the stimulus statistics from the prompt context, rather than reflecting a fundamental, internalized mechanism for handling uncertainty.

Furthermore, some of the paper's own findings contradict its central narrative. The discovery that Llama-4 Maverick outperforms the Bayes-optimal linear fusion model is a striking result. However, attributing this to a "sophisticated non-linear integration strategy" undermines the claim that models are converging on the classic Bayesian observer principles (which are typically based on linear-Gaussian assumptions, i.e., BLUE). If the best-performing model uses a strategy that is non-linear and better than the Bayesian benchmark used, it suggests the benchmark itself is an incomplete or incorrect model for what these LLMs are doing.

Finally, the claim of a "universal" strategy is weakened by task-specific inconsistencies. In the Maze Distance task, GPT-5 Mini achieves "near-perfect text performance", which is clearly a product of explicit reasoning, not implicit perception. If the model uses explicit reasoning for the text modality while (supposedly) using implicit Bayesian strategies for the image modality, it suggests an opportunistic, task-dependent application of different tools, not the emergence of a unified, human-like perceptual processing strategy.

### Questions
1.	How can the authors definitively distinguish "Bayesian computation" from simpler heuristics like "regression-to-the-session-mean"? Given that the session mean is easily discoverable via ICL from the "rolling context", what evidence shows this is more than just anchoring? Were any non-Bayesian control models (e.g., a simple anchoring-and-adjustment model) fit to the data?
2.	The "rolling context" seems critical for all sequential effects. What happens if this context is removed (e.g., by running trials in isolation or resetting the state)? If the Bayesian signatures disappear, wouldn't this confirm that the observed behavior is an artifact of ICL rather than an "emergent" property of the model's perceptual system?
3.	Regarding the Llama-4 Maverick finding: If the best-performing model (non-linear) surpasses the Bayes-optimal linear model (BLUE), does this not suggest that the linear-Gaussian Bayesian observer is the wrong benchmark? How does this finding support the conclusion that LLMs are converging on these specific Bayesian principles?
4.	In the maze task, the model appears to mix explicit reasoning (text) with implicit perception (image). How does this fit the narrative of an "emergent Bayesian" strategy? Does this not imply that the model lacks a unified strategy and simply defaults to different mechanisms based on modality and task difficulty?
5.	Could the authors further justify the design of the BCS? Why sum the binary signs of the change (e.g., plus or minus one) rather than a more nuanced metric, such as the correlation between the magnitude of the intervention (e.g., level of noise) and the magnitude of the shift in the prior weight? The current metric seems to lose a significant amount of information.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript studies the behavior of large language models (LLMs) under four behavioral tasks (marker location estimation, line ratio estimation, maze distance estimation, duration estimation). These tasks were previously used to study human behavior. Bayesian models have been developed to explain the behavior of human in these tasks.  The paper assessed the extent by which the behavior of a number of LLMs resembles Bayesian computations. Empirical observations on the performance of these models were reported in this study.

### Strengths
The idea of testing LLMs in a range of commonly used psychophysical and behavioral tasks is interesting. 

The study tested four tasks on a number of large language models. Three of them were multi-modal, so it was possible to assess whether optimal Bayesian cue combination strategies were used in LLMs for these tasks.

The authors considers several models of the observer’s behavior, i.e., linear observer, static Bayesian observer, Kalman filter. 

The authors developed several metrics to evaluate the behavior of LLMs.

The behavioral benchmark (BayesBench) will be shared with the community and may be of interest to some other researchers as well.

### Weaknesses
— While several tasks were used to test several variations of LLMs, there is not a major insight learned from the study.

— It was not clear how to interpret the finding that some LLMs were able to better integrate the information from the two modalities. Does this have something to do with how these models were trained (differently)?

— The writing needs improvement. In various places of the paper, the interpretations of the prior literature were not accurate. I would like to suggest a careful check of the accuracy of the reference to the literature. 

— It was unclear how the main results depend on the details of the prompts for LLMs. For example, if the LLMs were instructed to ignore their responses to the previous trials, would the sequential effect (akin to Kalman filter still persist)?

### Questions
What do the dots with different sizes represent in Fig. 1A?

Given the task setting, what are the optimal strategies? For example, would a Kalman filter be optimal? The authors fit three classes of models ( linear observer, static Bayesian observer, Kalman filer ) to the behavior data, however, the Bayesian optimal was not established. So it is difficult to assess how close the LLMs perform relative to the optimal strategy. 
For experiments with three different ranges, what if trials were interleaved between the three ranges? Would that change the behavior of the model?

Form Fig. 4, GPT-Mini’s responses did not seem to exhibit a regression toward the mean. Can the authors clarify?

Section 5.3 says “While Gemma 3 4B and Phi 4 Multimodal achieved higher BCS than expected, they are also the least accurate group of models. ” This is somewhat confusing. In what sense, it achieves higher BCS than “expected”?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces BayesBench, a psychophysics-inspired benchmark for evaluating whether large language models (LLMs) perform optimal multimodal integration consistent with Bayesian principles. The authors adopt classic psychophysics paradigms to assess nine LLMs across four magnitude-estimation tasks (location, ratio, distance, and duration) in both text and image modalities. A central challenge in this investigation is manipulating the precision of different information sources to test the adaptivity of LLM behavior. To address this, the authors use three methods—prompt steering, noise injection, and context manipulation—to systematically vary the precision of prior knowledge and observational evidence. Beyond standard accuracy metrics, they introduce a Bayesian Consistency Score to detect Bayes-consistent behavioral patterns even when performance saturates. Their results reveal that high accuracy does not necessarily imply efficient cue combination, though several accurate models exhibit Bayes-consistent adaptations, suggesting emergent principled uncertainty handling.

### Strengths
The paper makes several notable contributions. First, the authors develop a rigorous pipeline enabling controlled ablation of noise, context, and instruction prompts, which provides systematic leverage for probing LLM computational strategies. Second, the choice of perceptual tasks is well-motivated, as these tasks are less susceptible to contamination from previously acquired statistical rules or memorized patterns compared to knowledge-based tasks. Third, the evaluation framework is comprehensive, incorporating tense and multifaceted measures that go beyond simple accuracy to probe behavioral consistency with Bayesian principles.

### Weaknesses
**1. Confounding between capability and decision strategy.** The relationship between overall RMSE and Bayes factor evidence (Figure 7, line 376) raises fundamental questions about what is actually being measured. Since overall RMSE is partly influenced by Bayesian inference itself, the observed negative correlation may simply reflect circular reasoning rather than revealing genuine Bayesian tendencies specific to particular LLMs. A critical issue in establishing valid tests of Bayesian inference is separating an agent's perceptual or representational capabilities from its decision-making strategies. Without this separation, it remains unclear whether models with lower RMSE exhibit more Bayes-consistent behavior because they implement Bayesian computations or simply because they have superior underlying capabilities that coincidentally align with Bayesian predictions.

**2. Limited evidence for benchmark generalizability.** Before the impact of this benchmark can be properly evaluated, critical information about cross-task and cross-modality consistency is needed. If a specific LLM shows substantially different Bayesian consistency across tasks and modalities, establishing a general benchmark for Bayesian inference becomes questionable. The paper would benefit from reporting consistency metrics across conditions and discussing whether the observed patterns reflect stable computational principles or task-specific behaviors. While the authors acknowledge the need for extension, the current scope may be insufficient to support broad claims about LLM Bayesian reasoning. Expanding to additional diverse tasks would strengthen confidence in the benchmark's validity.

**3. Weak experimental design for testing Bayesian inference.** The use of uniform priors constitutes a relatively weak test of Bayesian reasoning. Under uniform priors, simple heuristic algorithms (such as averaging cues with task-dependent weights) can produce patterns resembling Bayesian integration without implementing genuine probabilistic inference. Stronger tests would involve informative priors with varying shapes or hierarchical structures that more clearly distinguish Bayesian from non-Bayesian strategies.

**4. Insufficient evaluation of alternative explanations.** The paper focuses almost exclusively on Bayesian frameworks without thoroughly evaluating non-Bayesian alternatives—a concern frequently raised in human psychophysics literature. Simple heuristics (e.g., reliability-weighted averaging, salience-based selection, or context-dependent switching rules) could potentially explain observed patterns without invoking Bayesian principles. Without systematic comparison against reasonable non-Bayesian computational models, the interpretation that LLMs exhibit Bayesian reasoning may be incomplete or potentially misleading. The authors should either formally test alternative models or provide stronger behavioral signatures that uniquely implicate Bayesian computation.

### Questions
### Q1: Separating Capability from Decision Strategy
How do you disentangle an LLM's perceptual/representational capabilities from its decision-making strategy? Developing theoretical or experimental measures for the perceptual/representational capabilities in question would help to answer this question.
### Q2: Cross-Task and Cross-Modality Consistency
How consistent is each LLM's Bayesian behavior across the four tasks and two modalities? Does a model exhibiting Bayes-consistent behavior on one task show similar patterns on others?
Please report within-model consistency statistics, such as correlation of Bayesian Consistency Scores across tasks and modalities. If consistency is low, discuss whether this indicates task-specific behaviors rather than general Bayesian computational principles.
### Q3: Stronger Tests of Bayesian Inference
Have you considered experimental designs with non-uniform, informative priors? Under uniform priors, weighted averaging heuristics can closely mimic Bayesian predictions.
Please consider to include experimental conditions with informative priors, such as skewed or bimodal distributions, where Bayesian and heuristic predictions diverge. Alternatively, provide theoretical justification for why uniform priors are sufficient or acknowledge this as a limitation.
### Q4: Evaluation of Non-Bayesian Alternative Models
What specific non-Bayesian computational models have you tested? Can simpler heuristics explain the observed patterns? Fitting a larger set of alternative models, especially those not in the Bayesian framework, and reporting the results of model comparison would help to address this question. Please demonstrate that specific key findings cannot be captured by plausible simple heuristics.
### Q5: Behavioral Signatures Beyond Regression to Mean
Beyond regression-to-mean effects, what specific behavioral patterns uniquely implicate Bayesian computation? For sequential models, do you observe learning curves or posterior sharpening with accumulated evidence?

### Additional Comments
* Human subject studies require an ethics statement. Was this work approved by an Institutional Review Board? I may have missed something but I did not find any ethics statement in the current manuscript.
* For practical deployment of BayesBench, guidance is needed on the minimum number of trials necessary to establish reliable estimates of Bayesian consistency.
* Beyond regression-to-mean patterns, what are the distinctive behavioral hallmarks of Bayesian inference in your data? For sequential Bayesian models specifically, does deviation from ground truth systematically change with accumulated experience?
* Figure 3 shows GPT-5 Mini's responses against sequential Bayes predictions. Including ground truth values as reference would improve interpretability.
* How were data handled when less capable LLMs failed to follow instructions properly?
* Figure 4 would benefit from an identity line for reference.
* Line 215: Please clarify the distinction between "behavioral modeling" and "cue-combination modeling".
* In metric definitions, do subscripts "LLM" and "model" refer to the same entity? Please standardize terminology.
* Given GPT-5's near-perfect performance, is it possible the model is secretly using external tools? 
* "Bayesian factor evidence" appears in Figure 7 and Appendix A7 but lacks clear definition in the main text. Please provide explicit explanation where first introduced.

### Soundness
3

### Presentation
2

### Contribution
3
