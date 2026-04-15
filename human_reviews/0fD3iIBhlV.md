# Emergence of a High-Dimensional Abstraction Phase in Language Transformers

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 8, 6

## Abstract
A language model (LM) is a mapping from a linguistic context to an output token. However, much remains to be known about this mapping, including how its geometric properties relate to its function. We take a high-level geometric approach to its analysis, observing, across five pre-trained transformer-based LMs and three input datasets, a distinct phase characterized by high intrinsic dimensionality. During this phase, representations (1) correspond to the first full linguistic abstraction of the input; (2) are the first to viably transfer to downstream tasks; (3) predict each other across different LMs. Moreover, we find that an earlier onset of the phase strongly predicts better language modelling performance. In short, our results suggest that a central high-dimensionality phase underlies core linguistic processing in many common LM architectures.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work analyzes properties of representations of several LLMs through a few approaches: downstream probing, intrinsic dimensionality and information imbalance. The analysis is mainly developed around intrinsic dimension and they show LLMs typically have a few intrinsic dimension peaks across layers. Additionally, they suggest that those peaks indicate transition to abstract linguistic processing through a variety of analysis

### Strengths
I like that authors combine evidence from a few different perspectives to demonstrate the relation between intrinsic dimension peak and transition to abstract processing. They also conduct experiments on a few corpus and a few models as well, which make the claim more general and robust

### Weaknesses
The method section is weak and the explanation of intrinsic dimension computing is not enough given its importance in this work. I was not able to identify which variable is corresponding to the intrinsic dimension without going through the cited paper. It seems to be the variable $d$ which authors did not explain what it is.

Additionally, author made a wrong claim in line 177~178 that \mu has a generalised pareto distribution. I cannot find any resources claiming this specific distribution is a (generalized) pareto distribution including the original cited paper.

### Questions
In the data section, it is not very clear to me what does it mean to "extract 5 non-overlapping partitions of 10k 20-token sequences" and how the shuffled version is generated, can authors explain more about this?

In section: The ID peak marks a transition in layer function, I think the relation between ID peak and \delta(l_i \to l_first) is not very clear. It has very similar shape in OPT and somewhat in pythia, but LLAMA has a completely different curve. It is maximizing towards the end of the layer instead of the center of layer.

In section 4.2, authors also claim the relation between ID-peak and a few tasks. However, Figure (a) and (b) do not have very clear co-related trend between ID peaks and tasks' performance. In particular, task performance in Figure 5(b) seems to be monotonically increasing instead of peaking in the middle. Can authors justify more about this?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The submission is analytic work trying to correlate intrinsic dimension of intermediate NN/LLM representations with linguistic targets. Comparison of the method applied to different models allows insights into some of their learned structural differences.

### Strengths
The findings in bold letters at lines 406-407 and lines 425-426 may be useful to some researchers that need to train and/or select models. The fact that ID seems to change gradually over layers is interesting, but may have a simple explanation in the extreme averaging scale of these models.

### Weaknesses
Except for the few strengths mentioned above, the submission does not explain for what else the gained insights can be used for or wether they are more useful than that at all.

The analysis focuses only on fully trained models and does not provide insights into how ID changes over time. A correlation analysis to other work would add more value. My first thought was a correlation to the IB method (e.g. Tishby et al. 2000 and Schwartz-Ziv & Tishby 2017), but this may not be the only or best choice.

The submission wrongly mentions PCA being linear (line 061, applies only to its original form) which leads to the quick conclusion to discard it. This is puzzling as the research on non-linear PCA is quite diverse based on very different techniques and there's even early work using neural networks dating back to 1991 (Mark Kramer: "Nonlinear PCA Using Autoassociative NNs").

### Questions
I strongly advice to improve the submission w.r.t. the mentioned weaknesses. That helps both quality and reach.

The first paragraph of Asset Section C.1 (lines 809-836, in particular 828-829) mentions sensitivity of ID estimation w.r.t. to noise, small scales, density variations and curvature. That analysis suggests some sort of frequency decomposition integrated with the ID estimation.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper explored how transformer-based language models evolve their internal representations across layers, revealing a distinct high-dimensional abstraction phase. The authors observe the some findings across multiple LMs and datasets, and they provide a foundation for better understanding and optimizing language model architectures. This work bridges the gap between geometric representation and linguistic function in transformer-based LMs. Also, it highlights the potential of intrinsic dimensionality as a tool for analyzing and evaluating LMs.

### Strengths
This work conducts experiments on various LMs (e.g., OPT-6.7B, Llama-3-8B, Pythia-6.9B, OLMo-7B, and Mistral-7B) using multiple datasets, providing a comprehensive analysis. It also observes how representational intrinsic dimensionality (ID) varies across layers and proposes insightful hypotheses. Furthermore, this work inspires the research community to explore the utilization of ID information in transformer-based LM applications.

### Weaknesses
While the paper combines two methods, GRIDE (Denti et al.) and Information Imbalance (Glielmo et al., 2020), to analyze four large language models (LLMs), this combination may fall short in terms of novelty. In Section 4.1, the choice of pre-training datasets for evaluation is also a limitation. Since these datasets have likely been encountered by the models during training, the results may not provide a fully accurate picture of the models’ generalization capabilities. Testing on unseen datasets would be crucial to evaluate the robustness and generalizability of the observed patterns, especially in real-world applications where unseen data is the norm. The study is limited to a narrow range of LLMs in terms of scale. Evaluating models of varying sizes (e.g., smaller models alongside large ones) would offer a more comprehensive understanding of how model size impacts intrinsic dimensionality and representation overlap across layers.

### Questions
1. There seems to have second ID peak in the later layers over LLMs. Do you think this second ID peak might reveal additional insights?
2. In your analysis (Figure 4), you observed that Pythia and OPT exhibit very similar representations. Could this similarity be attributed to pre-training on similar datasets? If so, how might this influence your findings, and have you considered controlling for dataset overlap to isolate structural factors more effectively?
3. The work focuses on classification tasks to analyze representation spaces in language models. Could you explain why generative tasks were not included? Do you expect the observed ID peaks and representation patterns to differ in generative contexts?

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
5

### Summary
This work takes a high-level geometric approach to analyze intrinsic dimension (ID) of the representational manifold at each layer of a decoder-only Transformer LLM to understand how layer geometry relates to layer function. Although inspired by the earlier work of (Valeriani et al., 2023), this work greatly extends the models investigated to include five mainstay decoder-only LLMs, and added more extensive probing and downstream tasks on defined datasets to analyze ID profiles across layers. The resulting observations are different from those from (Valeriani et al., 2023). This work made quite a few interesting findings on detecting broad qualitative patterns, and provides useful guidance for future research towards interpretability, analysis of model behavior and quality, and model pruning and layer-specific fine-tuning etc.

### Strengths
(1)	Although inspired by the earlier work of (Valeriani et al., 2023), this work greatly extends the models investigated to 5 distinct mainstay transformer-decoder-only LLMs and added more extensive probing and downstream tasks on defined datasets to analyze ID profiles across layers. Hence, the conclusions drawn in this work are verified across various models, datasets, and tasks, making the findings more convincing.

(2)	The comparisons to related works, esp. (Valeriani et al., 2023) which inspires this work, are clearly presented, hence the contributions of this work are clear and solid.
The verification of the emergence of a central high-dimensionality phase, and analysis of language processing behavior and performance during the high-dimensionality phase are quite thorough.

(3)	The analysis in Conclusion demonstrates that many findings in this work align with prior works and concurrent works. The paper clearly summarizes insights of guidance for future research. The Appendix provides detailed experimental setup and additional results. And finally, the analysis of potential applications of the findings is valuable to the research community.

(4)	Overall, the paper is clearly written and easy to follow.

### Weaknesses
(1)	Although the paper is overall clearly written, please make sure that every symbol used needs to be clearly defined when it first appears, e.g., d in Section 3.4.

(2)	Please provide rationale for critical algorithmic designs, for example, please clarify why GRIDE is selected, and why the three alternative measures for comparing layer’s representation spaces are chosen. 

(3)	Currently, k is still selected based on visual inspection. It would be useful to propose methods that can automatically select k.

(4)	It is interesting that OLMo seems a bit of an outlier compared to the other 4 LMs, although it also exhibits the ID peak and other related properties. It would be useful to provide insights on why OLMo behaves differently from the other models, and shed light on patterns of any potential “outlier” LM.

### Questions
(1) Please check the questions listed under Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper uses the technique of intrinsic dimension estimation as a tool for analyzing properties of different transformer LLM layers. 5 different LLMs are analyzed on textual inputs from 3 different public-domain corpora. In addition to computing the intrinsic dimensionality (ID) (using the generalized ratios intrinsic dimension estimator) for different layers, the ID is  correlated with performance of different layers' representations on syntactic and semantic probing tasks. Furthermore,  the difference in representational power between different layers is measured using an Information Imbalance criterion. The authors find that middle layers in LLMs have the highest ID; ID peaks seem to be an indicator of linguistic structure being learnt; early onset of peaks in ID across layers is correlated with better next token prediction performance performance; and high ID peak layers are representationally equivalent across different LLMs.

### Strengths
The paper conducts a broad analysis across 5 different LLMs and considers a range of questions and ablation studies (e.g., estimating ID on shuffled data, comparing layers across different models); altogether an impressively broad set of experiments. The paper is clearly written and presents a few new insights (e.g., correlation between peak onset and performance). Code and data would be made available, which would be valuable for the community.

### Weaknesses
The use of ID as an analysis tool for LLM layers is  not an entirely new idea (e.g., https://arxiv.org/pdf/2402.18048). 
Most of the results (e.g., the peaking of ID at middle layers, emergence of  linguistically informative representations in those layers) has been shown before by means of other methods (e.g., mutual information or canonical correlation analysis). These should have been discussed in more detail under prior work.

### Questions
While the analyses show some interesting trends, it is difficult to tell how meaningful or significant the numerical differences are. Methods for analyzing LLM layers other than through ID could have been discussed in a prior work section.

### Soundness
3

### Presentation
3

### Contribution
2
