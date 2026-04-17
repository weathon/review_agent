# Calibrating Verbalized Confidence with Self-Generated Distractors

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Calibrated confidence estimates are necessary for large language model (LLM) outputs to be trusted by human users. While LLMs can express their confidence in human-interpretable ways, verbalized LLM-generated confidence scores have empirically been found to be miscalibrated, reporting high confidence on instances with low accuracy and thereby harming trust and safety. We hypothesize that this overconfidence often stems from a given LLM’s heightened suggestibility when faced with claims that it encodes little information about; we empirically validate this hypothesis, finding more suggestibility on lower-accuracy claims. Building on this finding, we introduce Distractor-Normalized Coherence (DINCO), which estimates and accounts for an LLM’s suggestibility bias by having the model verbalize its confidence independently across several self-generated distractors (i.e. alternative claims), and normalizes by the total verbalized confidence. To further improve calibration, we leverage generator-validator disagreement, augmenting normalized validator confidence with a consistency-based estimate of generator confidence. Here, we frame the popular approach of self-consistency as leveraging coherence across sampled generations, and normalized verbalized confidence as leveraging coherence across validations on incompatible claims, allowing us to integrate these complementary dimensions of coherence into DINCO. Moreover, our analysis shows that DINCO provides less saturated – and therefore more usable – confidence estimates, and that further sampling alone cannot close the gap between DINCO and baselines, with DINCO at 10 runs outperforming self-consistency at 100. We release our code at https://github.com/victorwang37/dinco.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the overconfidence issue in LLM verbalized confidence when faced with claims that they have little knowledge about, and proposes Distractor-Normalized Coherence (DINCO) to address such suggestibility bias. The proposed approach normalizes the verbalized confidence by considering multiple self-generated distractors. Extensive experiments have shown the effectiveness of DINCO by demonstrating how it outperforms the existing baselines even when they scale up.

### Strengths
1. This paper provides a novel insight that the overconfidence of LLMs comes from suggestibility and shows empirical evidence.
2. Clear motivation and preliminaries are given to support the proposed method.
3. The proposed solution DINCO is effective with extensive experiments. Comparison across multiple datasets, models, and baselines has shown that DINCO generally improves calibration metrics.

### Weaknesses
1. Since the entire method builds upon the observation that the model will provide higher confidence on incorrect samples, it would be better if more analysis is provided. For e.g., current results are only derived from TriviaQA. Results from other datasets will make the observation more convincing.
2. Table 5 shows that accessing an NLI model significantly improves the results. However, only one NLI model has been tested in the results. This is not general enough, and more ablation studies should be provided. How sensitive are the results to the choice of the NLI model? Are there any requirements on the NLI model?
3. Sometimes the model could generate high confidence for an incorrect answer due to false training data or hallucinations. How can we distinguish suggestibility from these cases?

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces a method called Distractor-Normalized Coherence (DINCO) to enhance the calibration of confidence estimates in LLMs. It addresses the prevalent issue of overconfidence in LLM outputs, particularly in low-accuracy claims, which can compromise user trust. By prompting LLMs to verbalize their confidence across multiple self-generated distractors, DINCO normalizes these confidence scores and mitigates biases stemming from suggestibility when the model encounters unfamiliar topics. Overall, the paper contributes a novel approach to calibrating LLM confidence, thereby improving the usability and trustworthiness of their outputs.

### Strengths
1. The authors provide a thorough analysis of the proposed method's performance compared to existing approaches, showcasing its advantages in both short-form and long-form generation tasks.

2. By addressing the critical issue of overconfidence in LLM outputs, the paper contributes to enhancing the trustworthiness and usability of these models in real-world applications. The findings have implications for various domains where LLMs are deployed, emphasizing the need for reliable confidence estimates to ensure user safety and informed decision-making.

### Weaknesses
1. In the experiments, all evaluated open-source models are relatively small. It remains unclear whether the conclusions generalize to larger models (e.g., those exceeding 30B parameters).

2. In Appendix LLM-as-a-Judge, the description of the human evaluation process is unclear. Did the authors conduct human judgments themselves, and if so, how many participants were involved?

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
**Summary**: this paper proposes a method to improve the calibration of verbalized confidence approach (dubbed DiNCo), through the use of distracting contexts. Analysis across 6 models, reveals average ECE improvements of up to 0.099 over the best baseline in short-form QA datasets, and pearson correlation improvements (with passage-level FActScore) of 0.072 in long-form QA datasets (in biography generation).

**Main Contribution**: 
- the proposal of a calibration method for verbalized confidence using distracting contexts, enabling more diverse and trustworthy confidence estimates.
- experiments concerning both short-form QA and long-form QA, and across various models.
- interesting analysis comparing proposed method to self-consistency baseline at different sampling budgets

### Strengths
1. **Relevant and interesting work in uncertainty quantification in LLMs**, focusing on improving calibration of self-confidence approaches. 
2. **Novel findings** showing that it is possible to calibrate verbalized confidence estimates in a training-free manner, while alleviating confidence saturation and achieving better calibration than the best baseline irrespective of the compute budget.
3. **Empirically and theoretically grounded**: the proposed improvement follows from the concept of _suggestibility_, which states that LLMs’ generations are highly influenced by the context when LLMs lack sufficient information.
4. **Extensive experimentation including 6 models and concerning both short-form (i.e., TriviaQA, SimpleQA) and long-form QA datasets (i.e., biography generation)**.

### Weaknesses
1. Small improvements in calibration method, leading to questions about the significance and real impact of such improvements (e.g., one of the main results states that DiNCo improves ECE by an average of up to 0.099 (line 107)). 
2. Results are difficult to generalize, due to experimental changes (e.g., methods and configurations differ from Table 2 to Table 3).
3. However, it would be nice to test generalization of the proposed approach to other domains, especially those requiring domain expertise (e.g., medical domain or legal domain).

### Questions
**Questions**:
1. The exact setup for the preliminary study (in Section 2.2) is slightly unclear to me. Can you please clarify how you generate the set of C claims for the TriviaQA dataset and how many claims there are? I.e., what’s the upper bound in total confidence ? Is the number of claims the same across every question?
2. The preliminary study in Section 2.2 reveals that incorrect instances are indeed more suggestible (as evidenced by the average higher total confidence). However, I’m a bit surprised that the total confidence for correct answers is not unimodal and centered at 1. Do you have any intuition on why that is?
3. Table 1 caption mentions “underline results not significantly worse” but there is no underline. Does it mean that all of the results are significantly equal or better than the bold results?
4. Line 355 claims that “On TriviaQA, DiNCo outperforms the best baseline SC, by an average of ECE OF 0.099”. Can you clarify what is the criteria for selecting SC as the best baseline? If I understand the table correctly, Table 1 shows that MSP has lower ECE than SC in 3 out of the 4 evaluated models (avg ECE: 0.187 vs 0.209, respectively), which would suggest that MSP is a better baseline in the TriviaQA dataset. 

**Clarity**:
1. May be worth defining what is a valid minimal pair, since in Figure 2 left, examples may differ up to 2 words. Are minimal pairs restricted to those that differ in at most 2 words?
2. It’s worth clarifying when discussing improvements over baselines, whether these concern relative or absolute improvements.

**Formatting**: 
1. Is there a reason why the bins for correct and incorrect are not centered at integer numbers and have width 1? What does the count for total confidence [0, 0.33] mean in this context? Additionally, I recommend adding the corresponding means as vertical lines to ease comparison.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Distractor-Normalized Coherence (DINCO), a method for calibrating verbalized confidence in LLMs. The authors argue that standard verbalized confidence is often overconfident and saturated due to LLM suggestibility, that is, the tendency to assign high confidence to claims in the context, especially when the model is epistemically uncertain. The paper presents empirical results that suggest that DINCO leads to an improvement in calibration compared to baselines.

### Strengths
The overall idea of the proposed method is creative and explores a new direction for uncertainty quantification in LLMs. I see the main strengths of the paper as follows:

- originality: to the best of my knowledge, the connection between overconfidence and "suggestibility" is new, well-justified idea and appear to be empirically supported by the empirical results
- empirical results: DINCO appears to consistently outperform strong baselines across diverse tasks (both short-form and long-form) and models (open and closed weight) and that scaling up self-consistency to 100 samples still doesn't match DINCO (at 10 inference calls), which is compelling
- clarity: the paper is very well-written and clear

### Weaknesses
I believe there are a few weaknesses:

- heavy reliance on oracles
- efficiency: the paper positions DINCO as a "zero-resource" method, but the method requires multiple inference passes to generate distractors, multiple passes to validate each one, and calls to an NLI model, which is more expensive than standard SC
- distractor quality: the method assumes the LLM can generate high-quality, "minimal pair" distractors, but if a model is epistemically uncertain, it might fail to generate plausible alternatives, breaking the normalization assumption; beam search or prompting is used, but the paper doesn't rigorously analyze the quality or diversity of these distractors and how they directly impact the $\beta(C)$ factor
- hyperparameters: the choice of $K=5$ for distractors and $K=5$ for SC samples seems arbitrary. Is the method sensitive to these?

### Questions
- Is the proposed method sensitive to hyperparameters, especially to the choice of $K=5$ for distractors and $K=5$ for SC samples?
- Can you provide a cost analysis (latency/FLOPs) comparing DINCO to SC@10 and SC@100? The claim that DINCO is more efficient than scaling SC needs concrete validation given the multi-stage pipeline (generate distractors -> validate each -> NLI pairs -> SC samples).
- What happens when the distractor generation fails to produce mutually exclusive claims, even after NLI filtering?
- Have you evaluated how the choice of NLI model impacts downstream calibration?

### Soundness
3

### Presentation
3

### Contribution
3
