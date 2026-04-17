# Human-Alignment and Calibration of Inference-Time Uncertainty in Large Language Models

- Decision: Reject
- Scores: 4, 2, 2, 6, 2

## Abstract
There has been much recent interest in evaluating large language models for uncertainty calibration to facilitate model control and modulate user trust. Inference time uncertainty, which may provide a real-time signal to the model or external control modules, is particularly important for applying these concepts to improve the large language model user experience. While many of the existing papers consider model calibration, comparatively little work has sought to evaluate how closely model uncertainty aligns to human uncertainty. In this work, we evaluate a collection of inference-time uncertainty measures, using both established metrics and novel variations, to determine how closely they align with both human group-level uncertainty and traditional notions of model calibration. We find that numerous measures show evidence of strong alignment to human uncertainty, even despite the lack of alignment to human answer preference. For those successful metrics, we find moderate to strong evidence of model calibration in terms of both correctness correlation and distributional analysis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
How aligned are LLMs' uncertainty to people's judgements of uncertainty? In this work, the authors compare a people's uncertainty judgements to models' judgements. The authors propose and investigate several different measures of uncertainty calibration.

### Strengths
The authors take on an important problem --- how human-aligned are models' uncertainty. I really enjoyed and appreciate the authors' emphasis on compute-efficient measures of evaluation. I also appreciate the breadth of metrics the authors consider. The work is quite comprehensive. I enjoyed Figure 2 and would have loved to see more of a deep dive into its results (see below).

### Weaknesses
While I believe the paper has potential to be very strong, the current version was structurally challenging to follow. The motivation for the work (to my understanding) centered around evaluating models' uncertainty relative to people. However, this is only really done in Section 4. Section 5 is then not grounded in human data at all? I felt Section 5 came "out of the blue" and disrupted the story of the paper. 

One idea would be to break up Fig 2 into more parts, expand Section 4, and substantially limit Section 5 in this piece. Or, flip the order so that the bulk of the emphasis is on the human evaluation. Structurally, the current paper is unfortunately highly confusing.

There are also no error bars in the main results, making it hard to assess how general and reliable the particular measures are.

### Questions
In addition to the questions/comments in my Weaknesses section: 

- It’d be nice to show more on the human uncertainty, e.g., in the Appendix. Not exactly clear how MUCH uncertainty there is in the human data. Scatterplots, for instance, would be more revealing than bar graphs, e.g., a point for each query/trial (or at least a subset) to look at alignment. 
- Were the survey datasets included in the LLM training data? They were all before 2023 so could have been? Does that influence uncertainty alignment? I realize it's out of scope for any rebutall period, but the paper would be much stronger with a new human eval as well to really assess alignment with models (for questions that are gaurenteed out of the models' training distribution).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper evaluates various LLM output distribution choices - entropy over the full vocab, top-k entropy, top-p entropy, and ‘choice’ entropy (entropy over the tokens corresponding to the answers in a finite set, e.g. a b c for MCQ questions) - for (i) human-alignment (agreement to human preferences and entropy), measured on a large survey dataset of US public opinion, and (ii) calibration, measured on MMLU.

The authors find that human alignment exists to a moderate degree at the top-token, but the ordering of non-top tokens does not correlate to human preference ordering. Furthermore, the degree of entropy of the models on each question correlates well in general with the entropy of human responses. Finally, the authors show that choice entropy has the best calibration on ECE, though all measures are moderately well calibrated. 

Overall, the paper’s analysis suffers from several weaknesses detailed below; and the contribution, even if these were to be rectified, is marginal.

### Strengths
1. To my knowledge, this is the first work that extensively studies top-k and/or top-p of the logit-distribution in terms of calibration and human alignment.
2. A reasonable range of models is used in the experiments.

### Weaknesses
1. Regarding human alignment, the experimental design does not seem to be sound. The authors primarily use two datasets, which are public surveys conducted in the US, to gather human opinions. These datasets consist of questions asked over a wide range of dates (2017-2023), and are often on topical issues such as politics. Therefore, the human distribution is not stationary; however, all the LLMs tested are off-the-shelf open-source models, and therefore will have static cutoffs and remain stationary over the time period. Some questions will not make sense based on the cutoff in question, either due to the question already being resolved by the cutoff date, or because the premise of the question is not relevant at the time of the cutoff date yet.
2. Furthermore, the actual impact of the current design is limited. The current design only looks at the aggregate human distribution of responses and how closely LLM logit distributions align with these out of the box – and this is alignment turns out to be weak w.r.t. the top token. Instead, the authors could have directly stated a goal such as using LLMs to replace humans in population surveys (a line of work which does have prior art and interest); and then conducted interventions to try and improve the simple baseline scores reported (such as prompting variations, etc).
3. The paper does not provide examples of questions from the above human survey datasets, so I had to go track these down myself to find out what they look like.
4. Regarding calibration, the authors detail some motivation for why they don’t simply use out-of-the-box normalised entropy, but this should be conducted as an ablation rather than simply stated.
5. ‘Global’ entropy as used in the paper comes with the concomitant risk of distributional shift due to domain change, but this is not commented on nor experimented with by the authors.
6. The analysis of calibration is restricted to simply the MCQ setting (which the authors acknowledge), even though it is easily extensible to the open-ended setting. Furthermore, only a single dataset is used, MMLU (though I acknowledge that this single dataset consists of multiple subject topics), and so out-of-distribution analysis – an important element of calibration analysis – cannot be, and is not, done here.
7. The calibration section has no comparisons to baselines at all despite the plethora of previous works examining e.g. ECE on MMLU. It is therefore not clear what the intended contribution of this analysis is.
8. One of the contributions listed in the introduction is the claim that top-p sampling in LLMs is equivalent to the notion of the highest density credible set in Bayesian statistics. The authors state that this is “an important but previously un-noted connection between the fields”. However, there is no justification or exposition in the paper to support this statement. If interpreted at face value, the statement is also not much of a contribution - it is a fairly trivial insight.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the alignment of inference-time uncertainty measures in large language models (LLMs) with both human group-level uncertainty and traditional calibration metrics. The authors evaluate a variety of entropy-based and probability-based uncertainty measures on a large dataset of survey questions. They find that several measures, particularly those based on entropy over various subsets of the token distribution (e.g., choice entropy, top-k entropy), show strong alignment with human uncertainty, even when the models' answer preferences do not align with human preferences. The paper also introduces a novel distributional calibration measure, the Jensen-Shannon Distance (JSD) shift, to assess how well model uncertainty predicts changes in the answer distribution. The results indicate that the human-aligned uncertainty measures are also moderately to strongly calibrated.

### Strengths
The core problem of evaluating whether an LLM's uncertainty corresponds to human uncertainty is important for building more transparent and trustworthy AI systems. The methodology is sound and described clearly. The creation and use of a large-scale dataset from the Roper Center is a significant contribution. The finding that uncertainty alignment can exist independently of preference alignment is a particularly interesting and non-obvious result.

### Weaknesses
The study's primary limitation is 
1. The models used are mainly small open-sourced non-reasoning models. Experiment on more diverse and larger models, including closed-source SOTA models such as GPT-4, Gemini, Claude etc., would strengthen the claims, as uncertainty measures are more relevant in widely deployed SOTA models.
2. The method mainly focuses on multiple-choice questions. While this is a necessary simplification, the true test of these uncertainty measures will be in open-ended generation tasks. The proposed conceptual framework for extending the method to open-ended questions is a good first step, but it remains to be seen how effective it will be in practice.
3. The writing style can be improved for example, the contribution section in the introduction can be better formatted.

### Questions
Main: 
1. The main question I have is, as I pointed out above, how well the findings generalize to larger and more diverse models beyond the small open-sourced models.

Minor 
1. The paper finds a surprising lack of alignment in preference ordering between the models and humans, which contrasts with some prior work. The authors suggest this might be due to the cloze testing prompt format. Could the authors elaborate on this? Would CoT prompting or other prompt engineering techniques potentially improve preference alignment?
2.  The JSD shift results for the Mistral-0.1 7B Instruct model are intriguing, suggesting it might be "anti-calibrated." Do the authors have any hypotheses for why this specific model exhibits this behavior? Could it be an artifact of its instruction-tuning process?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates inference-time uncertainty measures for large language models and asks two questions: (i) how well such measures align with human group uncertainty (estimated via survey-response disagreement), and (ii) how well they are calibrated to correctness. Using roughly 3k multiple-choice items from Roper (plus a small subset from Pew) and several 1B–8B open-weight models (base and instructed), this paper compares families of token-probability signals, including top-1 probability, choice entropy, total entropy, top-k entropy, and top-p metrics. The study introduces a distributional calibration check based on Jensen–Shannon-distance shift, alongside per-subject Spearman correlations and ECE-style calibration. Main findings: multiple entropy-based measures—especially choice entropy and top-k variants—show moderate to strong alignment with human uncertainty across models; the same measures also show evidence of correctness calibration on MMLU; preference-order alignment remains weak even when top-answer agreement is above chance; a pathway is outlined for extending these signals to open-ended generation via a reduction to a three-way judgment.

### Strengths
- **Originality.** This paper explicitly targets **human-aligned** uncertainty (beyond correctness calibration) and connects **top-p** selection to Bayesian highest-density sets; it also proposes **JSD shift** as a distributional calibration diagnostic.  
- **Quality.** Careful separation of **alignment** vs **calibration**; broad metric family; multi-model evaluation; subject-wise analyses; multiple complementary criteria (correlation, ECE, JSD shift).  
- **Clarity.** Clear prompt template, dataset construction details, and heatmap/table visualizations that make the mixed “agreement vs ordering” story easy to parse.  
- **Significance.** Practical, **low-overhead** signals (token-probability–based) that can drive runtime control/abstention in black-box deployments; highlights **choice entropy / top-k entropy** as strong default options.

### Weaknesses
- **Multiple-choice scope.** Evidence is limited to MCQ; the open-ended extension is conceptual and untested.  
- **Prompt/decoding sensitivity.** Alignment differs from prior “counterfactual prompting” studies; results may depend on the **cloze** template and decoding choices.  
- **Model coverage.** Only ≤8B open-weight models are included; larger/API models are referenced but not comprehensively stress-tested.  
- **Metric/threshold choices.** Heuristic thresholds (e.g., |r|≥0.3) and standardization-based ECE binning may affect conclusions; alternatives could be reported.  
- **Compute profiling.** End-to-end **latency/memory** impacts of computing metrics at scale are not fully quantified.  
- **Failure analysis.** Notable negative/anti-calibration cases (e.g., a 7B instruct variant) merit deeper diagnosis.

### Questions
1. **Generalization to open-ended tasks:** Can you validate the proposed 3-way reduction empirically (few-shot rubric, judge variability), and compare against entailment-based or reference-guided scoring?  
2. **Template & decoding effects:** How robust are alignment/calibration results to prompt variants, temperature, and nucleus/top-k settings?  
3. **Combined signals:** Do simple fusions (e.g., choice entropy + top-k entropy) yield better Pareto fronts for risk–coverage/ECE?  
4. **Cost accounting:** Please report wall-clock, memory, and throughput for per-token computations across models/subjects; include guidance for real-time use.  
5. **Edge cases:** Analyze subjects/models with **anti-calibration** (positive correlation) to identify linguistic or dataset artifacts.  
6. **Scale & APIs:** Replicate key plots on larger and API models; verify whether trends (choice/top-k entropy dominance) persist.  
7. **Human side:** Report inter-survey reliability and how class-imbalance in human choices affects “alignment” correlations.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
**Summary**: 
proposes to measure the alignment between various logit-based uncertainty quantification (UQ) methods and human group-level uncertainty, which is operationalized using entropy of population answer choices. Across 10 LLMs and 2000+ multiple choice questions of recent surveys, results reveal that most logit-based UQ methods are (linearly) strongly correlated with human uncertainty ($|\rho| geq 0.3$) – with choice entropy, total entropy and top-k approaches systematically ranking higher across models and datasets. In the MMLU dataset, analysis considering three different calibration metrics reveal that choice entropy is systematically better calibrated being more closely related to the correctness of the model.


**Contributions**:
- Novel perspective on the uncertainty quantification problem, focused on measuring model calibration with respect to human uncertainty (as opposed to correctness).
     - However, a similar problem is studied in Moore et al (2025), which first investigated the alignment of additional sampling measures (including sampling measures). The main difference seems to be the evaluation of both human alignment and calibration. Note that calibration and human alignment are measured separately (in two different settings)..
- Use of a Jensen Shannon Distance Shift as a measure of calibration.

### Strengths
- S1. Interesting angle on uncertainty quantification (UQ) research in LLMs, exploring whether the miscalibration of LLMs implies that models are in fact reflecting human uncertainty over answers.
- S2. Overall well-written and organized. Figure 3 provides a summary view over the different MMLU subjects.
- S3. Main results are backed by hypothesis testing (Figure 2 and Table 2).

### Weaknesses
- W1. **Limited novelty**: while the paper expands on the differences between prior work (in lines 82-88) it appears incremental (increasing number of datapoints and carrying calibration analysis). Perhaps the authors can highlight differences in findings or how their extended analysis to calibration differs from findings in prior work.
- W2. The definition of “inference-time” is too broad and not sufficiently motivated: the arguments provided to narrow the experiments to logit-based approaches are  also applicable to training-time approaches to UQ. As such, training-based approaches to model calibration could also be suitable approaches to be assessed in this work.
- W3. **Experiments are conducted exclusively in 3 multiple-choice formats**: Pew Research Center and Roper Center for Public Opinion Research (2025) surveys for measuring uncertainty alignment and MMLU to measure calibration. While there’s value in evaluating uncertainty alignment in multiple-choice settings, it does not necessarily generalize to more realistic settings where users interact with LLMs in open-ended generations.
- W4. Results are obtained using a single prompt: given LLMs’ sensitivity to slight changes in prompts, one may wonder about the generalization of these findings to different prompts (e.g., 0-shot vs few-shot prompting).
- W5. Measurement of alignment between UQ methods and human group-level uncertainty has limitations, since it does not consider differences in the ordering of the answer choices. (See Questions for more details).

### Questions
**Questions**: 
1. Analysis of the human agreement is conducted using three metrics: one focusing on measuring choice selection agreement, another focusing on preference ordering alignment, and the third one focusing on the discrepancy in human uncertainty and model’s uncertainty. 
      1.a.) Given the focus on measuring alignment between human group-level uncertainty and various UQ methods’ uncertainty, it is unclear why the first two metrics are necessary (the metrics are independent of any UQ method and instead fully rely on the model distribution). Could the authors motivate the relevance of including such analysis in the paper?
      1.b.) To measure uncertainty alignment between UQ methods and human uncertainty, the paper proposes to measure the linear correlation between the UQ method and the entropy of the human distribution per answer. While using entropy provides an aggregate view of human uncertainty it loses information about which answer choice the model is more confident about. For instance, for a multiple choice question with 4 answer choices, the two distributions have the same entropy but refer to two completely different settings: [0.0, 0.0, 0.25, 0.75],  [0.75, 0.25, 0.0, 0]. In other words, if I understand the setup correctly, the use of entropy as the proxy for human group-level uncertainty followed by pearson correlation may be an overoptimistic measure of alignment. This is further validated by lines 215-238 which state that models are only moderately aligned in the top-token but have different multiple choice ordering. Can the authors please clarify whether this is a problem and how they tackled it (e.g., post-hoc manual analysis)?
3. In Section 5.1. The paper mentions the use of Spearman correlation between binary correctness and UQ measures (lines 274-275). However, due to the discreteness of the correctness variable, my intuition is that this is an ill-suited metric. Can you motivate this metric choice?


**Clarity**: 
1. The paper positions itself as “inference-time uncertainty in LLMs”, defining this class of methods as “measures that are able to be calculated at any time during generation, without additional auxiliary generations” and “inference time measures are uniquely useful [...] without significant added computation”. However, in my understanding such definition does not exclude training time approaches – approaches which are not discussed in this paper. I suggest the authors further clarify the definition or that consider training-time approaches, i.e., approaches that approach calibration through fine-tuning models.
2. Table 1 caption (page 4): add information about what bold faces mean.
3. Lines 272-277 mention the analysis being split in two phases but only one phase is discussed: the analysis using Spearman correlation.
4. Line 432 mentions “model itself may be unusually negatively calibrated”. Please explain what negatively calibrated implies, since in the original classification definition calibration values can only be between 0 and 1.


**Supporting arguments**:
1.  Lines 98 refer to the “limited or no inference-time capabilities [...] like self-reporting [...] and multi-inference consistency”. Please clarify what “inference-time” capabilities are and provide adequate citations to back such arguments.


**Typo**:
- Line 153: “surveys .” → “surveys.”
- Legend title in Figure 2: “Measures” → “Models”
- Missing legend in FIgure 5. It’s difficult to attribute each colored line to a different baseline.

**Suggestion**: 
- Figure 2 is great! But super dense! One possible suggestion if you do consider this, is to select the best UQ method out of the top-k entropy ablations, the best for top-p entropy and best for top-p size and plot those. Leaving the other variations to the appendix.
- Formatting of Table 2 is off. I recommend formatting it like Table 1.

**Additional limitations**:
- The paper should also mention the fact that this analysis is only applicable to open-source models or closed-source APIs providing access to top-k and/or top-p access.

### Soundness
2

### Presentation
2

### Contribution
1
