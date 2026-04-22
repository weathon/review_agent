# Stance Elicitation as a Black-Box Framework for Auditing LLM Alignment

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 4, 0

## Abstract
We present a stance-tensor method for visualizing the semantic spaces of large language models (LLMs).
  The method constructs stance vectors from model responses to structured entity--policy queries and uses these vectors to derive low-dimensional representations of the underlying semantic structure, enabling direct comparison of generic descriptors with explicit rule-based specifications.
  Across multiple state-of-the-art LLMs, the approach allows us to identify consistent patterns, including a stable triangular configuration of U.S. political party anchors, close correspondence between party programs and philosophical traditions, clustering of generic normative terms in a consistent region associated with Rawlsian principles, expected placement of Pew political-typology groups, coherent cross-national mapping of German parties into U.S. political space, a strong correlation between PCA-derived left--right scores and Manifesto Project RILE values, substantial inter-model variation in demographic stereotyping, and systematic divergences between generic and rule-based definitions of alignment and legal systems.
  These results show that simple stance-based probes reveal stable and reproducible semantic structure in LLMs and provide a direct mechanism for identifying inconsistencies between default assumptions, explicit rule sets, and institutional frameworks.
  Because these discrepancies form measurable error signals in the stance tensor, the same framework can be used not only for auditing but also for improving model alignment through targeted training.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a black-box framework called stance elicitation to audit how large language models (LLMs) are politically and morally aligned. Instead of looking inside the model, the method asks structured questions like “Would entity X agree with policy Y?” to map each model’s semantic space into measurable stances. The authors test several major LLMs across political and moral statements. The study also identifies systematic biases and gaps between stated and revealed alignment, plus demographic stereotyping.

### Strengths
1.The framework is simple, intuitive and works as a black-box audit.

2. It’s transparent and well-validated across multiple models.

3. The work bridges AI auditing and social science in a fresh, useful way.

### Weaknesses
1. The method mostly observes bias but doesn’t really explain or fix it.

2. There’s still a chance that the prompts themselves introduce bias — the paper doesn’t fully control for that.

3. The evaluation focuses on politics in Western contexts, which might limit generalization.

4. Some visual and statistical analyses (e.g., PCA plots) feel more descriptive than rigorous.

### Questions
Please see the weaknesses section — the paper’s presentation is difficult to follow and could benefit from further refinement, especially in improving clarity and figure readability.

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
3

### Summary
The paper proposes a method based on stance elicitation for characterizing the political biases present in Large Language Models (LLMs). Specifically, the paper develops a series of prompts fundamentally consisting of an entity descriptor, a policy statement, and a yes/no question about the latter two components. The paper then uses the yes/no outputs from LLMs in order to characterize their political biases. It further compares the output biases of Pew Research Center’s political typology and RILE scores as a form of docking for validity. The paper also makes some observations about how some entities, like generic person descriptors, do not lead to reliable political biases in the outputs.

### Strengths
The paper addresses a significant problem and has some potentially significant results. First, characterizing the political bias present in LLMs, especially in a black box manner, is a societally important problem, as LLMs are increasingly being used to analyze political content and even simulate political opinions. Furthermore, results like the instability of alignment of personalities with political beliefs are an interesting result from a bias analysis standpoint and may explain some of the results seen in other LLM political bias papers (as pointed out in the paper).

### Weaknesses
There are some weaknesses in the grounding, novelty, and clarity of the paper. For the grounding of the paper, I think there are weaknesses both methodologically and empirically. For methodology its not clear how much the methodology of prompting the LLM to get numerical responses would lead to different results if perturbations were applied. For example its seem likely that perturbing the verbiage of elements like the entity and policies, even if the verbiage is roughly semantically equivalent but maybe with a different sentiment polarity, in the prompts could alter the responses. In Ng et al. “Examining the Influence of Political Bias on Large Language Model Performance in Stance Classification,” the authors found that differences in the datasets lead to downstream task performance differences. Additionally, how would results change if LLM were given a more nuanced way to evaluate positions of entities on policies, like a Likert scale versus a binary yes or no. Finally, the use of a tensor implies that each of the axes are all of the same thing along those axes. It's not clear from a qualitative perspective that policy issues are equivalent to something like moral principles; I would agree that they are related, but not that they are equivalent. Thus, it's not clear that a tensor is the right representation of the outputs of the models, given the entity and policy tests. From the empirical perspective, I think there are some issues with the completeness of the policy probes and presentation of some of the political viewpoints/parties. For the latter, in the U.S., the libertarian party is not a major party and does not represent a third pole of U.S. politics. In fact, most libertarians caucus and vote along Republican lines in major U.S. elections. For the former, while it's probably not possible to fully enumerate all policy positions to really find the contours of a party, there does not seem to be any grounding that the policy positions used in the prompting represent a reasonable basis to separate political parties. 

For the novelty, it's not clear that the proposed method significantly improves upon the political surveying process done by Rozado in his works. The proposed method is, at its core, the same procedure Rozado uses (i.e., prompting LLMs on political questions and topics to evaluate their bias) but with a different structure around the prompts and a quantitative analysis of the outputs. 

For clarity, I think the paper would really benefit from something like a flowchart to understand the full process of prompt generation $\rightarrow$ tensor $\rightarrow$ analysis of the tensor. I personally had a hard time tracing where the conclusions were coming from.

### Questions
1. What is the reason the PCA is used for nearly all projections except one (policies uses t-SNE) despite mentioning having used PCA, t-SNE, and UMAP?

2. How would results change if LLM were given a more nuanced way to evaluate positions of entities on policies? For example, what about using a Likert scale with degrees of agreement or disagreement?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes a “stance elicitation” method to audit LLM alignment. It builds a stance tensor from model responses to policy statements and uses PCA/t-SNE to project ideological positions. The authors claim it reveals consistent political and philosophical structures and highlights gaps between stated and revealed alignments. However, the setup and evidence are informal and mostly qualitative.

### Strengths
* The stance tensor idea is a clear, general way to visualize model positions.
* The generic vs. rule-based contrast is interesting for alignment diagnostics.
* Some visual patterns (party clusters, philosophy alignment) seem intuitive and interpretable.

### Weaknesses
* No robustness checks or variance across seeds/prompts; results may not be stable.
* Heavy reliance on visualizations without statistical testing or quantitative evaluation.
* Lacks comparison to recent political bias auditing baselines.
* Code and details are missing; the paper also doesn’t follow ICLR formatting.

### Questions
1. The paper claims to reveal political and philosophical structures “more faithfully than prior auditing approaches” (Sec. 1, Conclusion). Which prior methods are these, and why are there **no quantitative comparisons** to established frameworks such as Bang et al. (2024), Argyle et al. (2023), or Motoki et al. (2025)?
2. None of the figures include **error bars, variance, or statistical tests**. How stable are the findings across random seeds, temperatures, or prompt paraphrases?
3. Several evaluation items were reportedly **generated or validated by the same model families** later audited. How do the authors rule out circularity or data leakage?
4. The RILE correlation (r = 0.99) seems implausibly high without sample size or CIs. How many parties were included, and is this robust to translation and summarization?
5. Why does the submission **not use the official ICLR template**? Was this intentional?
6. Will a compliant version and code with fixed API details be released for reproducibility?

### Soundness
1

### Presentation
1

### Contribution
1
