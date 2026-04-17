# SimBench: Benchmarking the Ability of Large Language Models to Simulate Human Behaviors

- Decision: Accept (Poster)
- Scores: 10, 6, 4, 4

## Abstract
Large language model (LLM) simulations of human behavior have the potential to revolutionize the social and behavioral sciences, if and only if they faithfully reflect real human behaviors. Current evaluations of simulation fidelity are fragmented, based on bespoke tasks and metrics, creating a patchwork of incomparable results. To address this, we introduce SimBench, the first large-scale, standardized benchmark for a robust, reproducible science of LLM simulation. By unifying 20 diverse datasets covering tasks from moral decision-making to economic choice across a large global participant pool, SimBench provides the necessary foundation to ask fundamental questions about when, how, and why LLM simulations succeed or fail. We show that the best LLMs today achieve meaningful but modest simulation fidelity (score: 40.80/100), with performance scaling log-linearly with model size but not with increased inference-time compute. We discover an alignment-simulation tradeoff: instruction tuning improves performance on low-entropy (consensus) questions but degrades it on high-entropy (diverse) ones. Models particularly struggle when simulating specific demographic groups. Finally, we demonstrate that simulation ability correlates most strongly with knowledge-intensive reasoning (MMLU-Pro, $r=0.939$). By making progress measurable, we aim to accelerate the development of more faithful LLM simulators.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper proposes SimBench, a benchmark for testing whether models can reproduce group-level human response distributions across 20 diverse datasets. They find a log-linear scaling of simulation performance with model size; no consistent benefit from reasoning models; and a trade-off with instruction-tuning where it improves performance on questions that have more consensus and decreases performance on questions that have more diversity. They also replicate past work, such as showing that performance worsens when targeting specific demographic groups.

### Strengths
1. There is a great need for high-quality evaluations in the nascent field of LLM social simulations, which this paper targets.
2. The data curation is rigorous, including scale, diversity (of task and participants), and cleaning/consistency.
3. The empirical findings are interesting and well-framed as tentative and qualified—ripe for future research.
4. There is extensive documentation and justification in the appendices, such as tests of alternative metrics.

### Weaknesses
1. The baseline is "equal probability to all response options," but does that make sense for real applications? Why not modal responses or a linear predictor? I'm particularly wondering about how to clarify the 40.80/100 score because it seems a more grounded alternative (e.g., the 85% test-retest accuracy in Park et al. 2025) would be useful if possible. I think accuracy (and even correlation) is more informative than TV in this regard.
2. The authors should consider including comparisons to quantitative machine learning models, perhaps with LLM embeddings for text inputs, as a comparison to the LLM social simulations. Knowing how LLMs compare to conventional models is important because that is often the counterfactual for researchers.
3. There is much heterogeneity in the datasets, as the authors acknowledge, but the paper has limited exploration of the different results in these areas and what they imply for simulation. Social science is extremely broad, and it is important to differentiate topics, rigor, elicitation methods (of the humans), etc.
4. A number of weaknesses are acknowledged by the authors but remain nonetheless, such as the reliance on multiple-choice format questions—understandable, but limits applicability to other research formats.

### Questions
1. I read the "Training Data Overlap" section, but did the authors empirically address in any way the possibility of models, especially larger and newer models, having the datasets (or the papers resulting from those datasets) in the training data? It would be useful to include tests on unpublished data, as did Hewitt and Ashokkumar et al. (2024; already cited in the paper), or at least foreground this limitation in the main text.
2. The analyses in the paper focus on the SimBench dataset as a whole, but with its diversity, what differences are there in performance across it? I ask in part because the target for simulations is often new studies where the results are highly uncertain. Ideally, one could examine performance on the parts of SimBench that are the best proxies for this (e.g., studies where humans were surprised by the results, studies on less-well-studied topics).

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper establishes a benchmark by collecting 20 existing datasets to evaluate how LLMs simulate human behavior, particularly in terms of group-level responses. The authors also conduct extensive simulations based on it and perform detailed analyses.  Although there are some issues and limitations, this work can provide new standards and references for future research.

### Strengths
1. The paper establishes a new benchmark based on existing datasets and performs a large number of simulations and analyses, with a lot of additional information in the appendix.

2. The structure of the paper is well-organized, and the authors provide detailed discussions of the different RQs. The figures and tables are presented in an appropriate and clear manner.

3. As the authors point out,  most of current evaluations for human behavior simulation are fragmented, and their work could serve as a useful reference for subsequent studies.

### Weaknesses
1. From a mathematical perspective, the score S in Eq. (1) is not a well-defined concept because if P is entirely random (i.e., identical to U), the value of S does not exist. The variable i in Eq. (1) also lacks the definition and explanation.

2. The authors assess the simulations of different LLMs based on only their own proposed metric S, without using other individual or group-level metrics. If the authors intend to demonstrate that their metric provides new insights, they need to compare it with some existing metrics. Appendix F compares TVD with other metrics, but the one used in this paper is not TVD.

3. More datasets don't always lead to more representative results. There are still more factors to consider. 
(1) Survey responses are collected through various ways, such as online, phone, and face-to-face interviews, etc. Theoretically, the same person could answer the same question differently under different circumstances. This paper does not analyze the dataset based on this criteria. 
(2) Do the authors consider the different quality levels of these 20 datasets? And the information loss resulting from different question normalizations in different datasets?

4. Certain conclusions might need to be modified or reviewed more carefully. For example, the conclusion "there is a clear log-linear scaling law for LLM simulation ability" drawn from Figures 2 and 6 may be too strong. Different approaches to calculating or weighting the results from various datasets are likely to yield different outcomes. With too few data points, other functions could easily be used for fitting and interpretation, e.g., OLMo and Gemma.

5. Some of the theoretical analysis is incomplete and inadequate. For instance, the definition of score S is clearly related to entropy, but this is not reflected in the explanation in Appendix I. To give an example, the range of score S is quite different in high-entropy and low-entropy cases. I speculate that it is not as simple and linear as the authors attempt to describe.

Minor issues:

1. What's the difference between "MoralMachineClassic" and "MoralMachine"? Why are their results so different in Figure 3?

2. typo, L276: "Appendix 6"

### Questions
Please refer to the weaknesses mentioned above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes SimBench, a benchmark for how well LLMs can simulate humans. It consists of 20 datasets selected through various criteria, and the authors use this benchmark to demonstrate a series of findings such as the effects of RL, simulating demographic groups, and others.

### Strengths
I think the study and analysis are comprehensive and well conducted. In particular, I think additional analyses on models such as centaur, and the explanation of the trade-off between alignment and plurality are well-written and nicely included. The dataset correlation study is also well-motivated. Overall, I think this paper cleanly presents a series of findings that have been circling around the community but hasn't been articulated clearly across many tasks, so this is very nice.

### Weaknesses
1. I think a core weakness of the paper is the standardization procedure to multiple choice questions. LLMs are sensitive to things such as the ordering of options [1], and also the format itself [2], so only evaluating under this paradigm is quite the weakness. 

2a. The simulation methods used (prompts in appendix) are quite rudimentary and there isn't much effort in using more advanced prompts such as including personal information and then aggregating, or more granular information such as interviews [3]. Evaluating models themselves may be less relevant if there are prompt wrappers that can allow the model to perform substantially better (e.g., few shot paradigms). 

2b. In the prompts themselves, it is mentioned that the simulated individual is either doing xxx survey or a worker on xxx platform. I wonder if this contradicts the generality across human populations that the authors try to achieve. 

3. presentation-wise, I think it would be nice to have a table of the datasets and some brief info. After a full read I wasn't able to get a good idea of all the datasets used, only the filtering criteria. 

4. [4] studies simulation on the choices13k dataset and has a set of intersecting ideas on how more performative LLMs model people more rationally. I think it would be good to include in related work. 

[1] Wang, et al. "Large language models are not fair evaluators." (2023).
[2] Long, et al. "Llms are biased towards output formats! systematically evaluating and mitigating output format bias of llms." (2024).
[3] Park, et al. "Generative agent simulations of 1,000 people." (2024).
[4] Liu, et al. "Large language models assume people are more rational than we really are." (2024).

### Questions
See weaknesses. Primary concerns are 1 and 2a. If the authors can meaningfully address these then I am happy to raise my score, as the work is promising.

### Soundness
4

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
2

### Summary
This paper introduces a new benchmark SimBench that aggregates 20 datasets across diverse domains (e.g., moral reasoning, economic games, global surveys) into a unified multiple-choice format intended to evaluate the ability of LLMs to simulate human behavior. With this benchmark, the authors draw several conclusions on simulation ability and certain conditions LLMs struggle more.

### Strengths
- The proposed benchmark provides a large-scale benchmark aggregating performance of LLMs for simulation tasks across diverse domains.
- The authors conducts empirical evaluation on a wide range of models and evaluates models in terms of many axes (e.g., how model size, inference-time compute affect their ability, what extent does simulation ability correlate with other capabilities, etc)

### Weaknesses
- The central claim that SimBench measures simulation ability rests on the assumption that predicting aggregate survey response distributions constitutes human behavior simulation. 
- The claims of the alignment-simulation tradeoff is potentially confounded. The authors do not rigorously test different hypothesis that could also explain this observation. For instance, question difficulty correlation, dataset artifacts (correlation could be driven by specific datasets rather than a general phenomenon), prompt sensitivity. Also the analysis only uses 13 model pairs, which pairs? What was the criteria for selecting these? Also the theoretical explanation is not complete, a stronger analysis would be needed to actually explain why this particular mathematical property translates to the observed behavioral differences.
- The correlation between SimBench and MMLU-Pro (r = 0.94) is interpreted as evidence that “simulation depends on deep reasoning.” However, this correlation could reflect data overlap or shared linguistic features such as linguistic formats (both MCQ), instructional cues, answer options, etc. rather than reasoning.
- The claim of “simulating human behavior across diverse cultural, linguistic, and socioeconomic backgrounds” is not well supported by the data composition. Although the paper states that there contains international scope, the majority of included datasets are English-language and Western-centric. It states that the benchmark spans more than 130 different countries but it is unclear how many samples there actually are per country.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
3
