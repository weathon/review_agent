# Sample Smart, Not Hard: Correctness-First Decoding for Better Reasoning in LLMs

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Large Language Models (LLMs) are increasingly applied to complex tasks that require extended reasoning. In such settings, models often benefit from diverse chains-of-thought to arrive at multiple candidate solutions. This requires two competing objectives: to inject enough stochasticity to explore multiple reasoning chains, and to ensure sufficient accuracy and quality in each path. Existing works pursue the first objective by increasing exploration at highly uncertain steps with higher temperature or larger candidate token sets, while others improve reliability by rejecting samples with low confidence post generation, implying that low confidence correlates with low answer quality. These two lines of thought are in conflict, as they conflate different sources of uncertainty. To resolve this, we argue that the decoding rule should be calibrated by *correctness*, not confidence alone. We should sample from tokens with higher estimated correctness, and reduce sampling where expected correctness is low. We propose simple strategies that achieve this goal: **Greedy-Threshold** makes sampling greedy at very low confidence steps. **Calibrated-TopK** and **Calibrated-ε** set truncation threshold based on estimated rank-wise correctness. Together, our findings challenge prevailing heuristics about decoding under uncertainty, showing consistent gains across math and general reasoning benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a new method that samples based on correctness during the decoding process. The authors conduct comprehensive experiments to demonstrate the effectiveness of their approach.

### Strengths
1. The idea is both novel and elegant. The authors approach the problem from a fresh perspective and achieve the goal of evaluating correctness without relying on complex architectures.

2. The motivation behind the work is strong, and the authors provide a comprehensive and convincing analysis.

3. The paper is well written and easy to follow.

### Weaknesses
1. The experimental results indicate only marginal improvements. In the main results table, the accuracy gain is approximately 1%, which appears to be relatively minor.

2. It's valuable that the paper includes results on GPT-OSS; however, the evaluations on Qwen are limited to smaller model sizes. It would strengthen the paper to include results on larger models, such as the 7B or 8B variants, or even larger if available.

### Questions
NA

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
paper presents an interesting and ambitious analysis of Correctness-First Decoding for Better Reasoning in LLMs. It’s well-structured and follows a logical argument, but several areas could be strengthened for clarity, rigor, and reader engagement.

### Strengths
1. The introduction effectively frames the research question and contextualizes the problem within existing literature.
2. The Methods section provides an appropriate research design and statistical treatment. The inclusion of comparative baselines is a strong point.
3. The experimental design and analytical approach are sound and appropriate for the research question.

### Weaknesses
1. The abstract summarizes methods more than findings; it doesn’t highlight the key quantitative results or main contributions.
2. Some information about sample selection, data splits, or parameter settings is missing, which could affect reproducibility.
3. The discussion section mostly restates results instead of analyzing their implications or addressing possible alternative explanations.
4. The conclusion doesn’t emphasize broader implications or concrete future directions, making the ending feel abrupt.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper argues that current decoding for reasoning mixes up two kinds of uncertainty:  good uncertainty (many valid continuations) and bad uncertainty (the model is just wrong). So instead of sampling more when the model is low-confidence, they propose “correctness-first” decoding: Greedy-Threshold (go fully greedy when max prob < τ), Calibrated-TopK (adapt k per step using a confidence×rank correctness grid), and Calibrated-ε (a continuous version that maps prob to expected correctness).On GSM8K, MMLU-Pro, BBH, and AIME (with GPT-OSS) these rules consistently give small-but-real maj@k / pass@k gains and can be layered on top of normal samplers which I found to be good.

### Strengths
- Very clear problem framing
- Easy to implement and add to existing samplers and inference is cheap
- The experiments were on broader side and showed consistency across range of models

### Weaknesses
- The paper seems to relies on task/data calibration; when calibration is noisy or OOD, gains shrink , so it’s not always free wins.
- From the paper it seems like improvements are incremental, few points  maj@k, big models already strong.
- The claim low confidence = epistemic, so always sample less is shown mainly on math/reasoning; creative/open-ended generation is only discussed, not tested

### Questions
Please refer to weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper re-examines predicting tree search in reasoning models by characterising the difference in uncertainty pertaining to randomness and that due to model uncertainty (aleatoric vs. epistemic uncertainty).  The authors then retrofit the sampling model to differentiate between these two potential sources of uncertainly through their proposed **Greedy-Threshold** sampling, where they calibrate the model's own certainty prediction (as modeled by final-layer logit) against that of the ground truth through Calibrated-$\epsilon$ mapping.

### Strengths
* I enjoyed reading this paper.  It is well-structured and presents well-founded arguments for the need for a calibration grid and its derivates that are then used to decide on rollouts.
* Analysis that mid-confidence bins benefit from diverse sampling is intuitive and shown in S2.2.  This is intuitive but needed to be shown in contrast to the prevailing notion that low-confidence requires diverse sampling. 
* The paper uses open-weight models Qwen2.5 and Llama, and later the recent GPT-OSS 20B as an advanced LRM.  This aids reproducibility.  
* The paper also experiments over various task datasets that are relevant to show cross-task applicability.

### Weaknesses
* I wished more information from the Appendix on failure cases (e.g., A.3 and A.6) could make it into the paper proper.  It helps to give a more concrete form with respect to sampling when grounded to an example.
* Same with respect to A.8, the relation to temperature could be given stronger theoretic guidance and derivation.  This part should tie in elegantly, but in the current submission, does not give a sufficiently unified presentation.
* The non-monotonicity of the calibration grids on page 23 deserve a bit of discussion.  The bottom of the grids somewhat reinforce your arguments, but here is is not clear whether sample sparsity also plays a role.

### Questions
* Why choose 10 bins?  Could you better fit this as an optimisation parameter where the bin sizes also correlate with transitions in the calibration bin to improve performance?

### Soundness
4

### Presentation
3

### Contribution
3
