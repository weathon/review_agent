# SteeringSafety: A Systematic Safety Evaluation Framework of Representation Steering in LLMs

- Decision: Reject
- Scores: 8, 6, 8, 2

## Abstract
We introduce SteeringSafety, a systematic framework for evaluating representation steering methods across nine safety perspectives including bias, harmfulness, hallucination, social behaviors, reasoning, epistemic integrity, and normative judgment, spanning 17 datasets. While prior work often highlights general capabilities of representation steering, we find there are many unexplored, specific, and important safety side-effects, and are the first to explore them in a systematic way. Our framework provides modularized building blocks for state of the art steering methods, enabling us to unify the implementation of a range of widely used steering methods such as DIM, ACE, CAA, PCA, and LAT. Importantly, this framework allows generalizing these existing steering methods with new enhancements, like conditional steering. Our results on Qwen-2.5-7B, Llama-3.1-8B, and Gemma-2-2B uncover that strong steering performance is dependent on the specific combination of steering method, model, and safety perspective, and that severe safety degradation can arise in poor combinations of these three. We find difference-in-means a generally consistent choice for steering models and note situations where slight increases in effectiveness trade off with severe entanglement, highlighting the need for systematic evaluations in LLM safety.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper outlines a systematic framework for the evaluation of representation steering methods across a variety of safety perspectives, to facilitate the performance on the selected safety perspectives and side effects on the other perspectives. The proposed framework, SteeringSafety, comprises modularized building blocks for representation steering methods such that existing methods can be represented as a combination of these blocks. This modularized breakdown enables the construction of novel representation steering methods by swapping out any of the modules. Besides this framework, the key result in this paper is that strong steering performance depends on the selected steering method, model and safety perspective.

### Strengths
The key contribution of this work is the framework, SteeringSafety. This enables the modularization of representation steering methods and the systematic evaluation of these methods on nine safety perspectives. The safety perspectives are drawn from literature, and suitable datasets and metrics are used for evaluation of methods on each perspective. This modular framework could be very useful for future development of representation steering methods and their subsequent testing. Additionally, the standardization of the tests would enable more consistent benchmarking in the future. The other notable aspect of this framework is the evaluation of entanglement on safety perspectives other than the ones the LLMs are aligned to. 

I would also like to commend the authors on the general readability of the manuscript. They have done an admirable job of exemplifying the safety perspectives through the infographic in Figure 1, and opted to evaluate the methods on representation steering methods based on datasets and metrics used in existing literature. Furthermore, the breakdown of existing steering methods in terms of the defined modules (see Table 1) serves as a sound justification for the proposed modularization.

### Weaknesses
There are several misleading/confusing statements in this manuscript that need to be clarified prior to publication. The primary point of confusion is the restriction of measuring steering effectiveness on three main perspective axes (line 67) – it is not evident why only three perspectives are selected and which three perspectives the authors are referring to.

In addition to this, the description and mathematical notation used in Section 3.1.3 is quite confusing – the mathematical form of $\nu^{\prime}$ is incomplete as there is seemingly an additional projection operation not included in the mathematical expression. 

I have listed my remaining questions and suggestions in the subsequent section.

### Questions
**Questions:**

1.	Why does the metric for effectiveness (Eq. 1) include a normalization by $1 – y_b$? Wouldn’t this result in effectiveness having a different range than entanglement?
2.	Line 267 states that “we search from the 25th to the 80th quantile of the layers” but line 268 states that “prior work has shown steering is more effective in the middle layers.” 

  a.	Are these quantiles based on the values or position of the layers? 

  b.	If it is based on values, what does the latter statement mean?
3.	The description of directional ablation in Section 3.1.3 (line 286) appears to modify the activations in a direction orthogonally to $\ni$, which contrasts with activation ablation. Is this correct? If yes, how can you reconcile the difference in direction ablations across the two approaches.
4.	Is line 343 supposed to read “decreasing harmfulness” since lower values for harmfulness would be desirable?


**Suggestions:**

Section 5 should be moved to earlier in the paper as it is important to establish the current landscape of evaluation of steering methods on safety perspectives. This would better establish the utility of the proposed framework.

### Soundness
4

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
2

### Summary
This paper introduces STEERINGSAFETY, a comprehensive framework for evaluating representation steering methods in LLMs across several safety perspectives using subsets from 17 datasets. It aims to systematically assess both the effectiveness and unintended side effects of steering interventions.

### Strengths
- Unifies five popular steering methods under the same benchmark, allowing for the evaluation of effectiveness and the side effects of these methods.
- Empirically, the results provide valuable insights into entanglement, showing that effectiveness and safety trade-offs vary across models and methods, which is a key contribution to alignment research. 
- The paper is also well-situated within related work and connects systematically to ongoing research in the field.

### Weaknesses
- While the framework claims modularity and standardization, it doesn’t provide runtime cost  or inference-time overhead details. 
- Effectiveness and entanglement are reported using scaled averages (in the main paper), which can hide nuanced behavior shifts. The results in the appendix seem more informative.
- For some categories, the number of used prompts seem a bit low. 
- The ethics statement is minimal (“our goal is to improve safety”). The reliance on LLM-as-a-judge should also be discussed given how it can mask some biases in the evaluation.

### Questions
- Can the authors put the definitions of the steering methods’ acronyms earlier in the paper? 
- For categories with relatively few prompts (like social behaviors), how do you ensure statistical robustness in the evaluation? 
- Given the reliance on GPT-4/ llama3 for scoring certain behaviors, how do you account for potential biases or inconsistencies in its judgments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this paper, the authors introduce STEERINGSAFETY as a framework for evaluating how representation steering methods affect safety across multiple dimensions in LLMs. The paper points out that previous research has focused on improving performance or alignment with steering methods, but this work shows that such interventions can also introduce unintended safety side effects.

They systematically evaluate and compare representation steering methods across nine safety dimensions and 17 datasets, revealing unintended safety trade-offs. Their results show that steering outcomes depend on the specific combination of steering method and model, with some interventions improving some target behaviors but harming others.

### Strengths
* This paper offers a broad and systematic evaluation to assess representation steering methods across multiple safety dimensions (nine perspectives, 17 datasets).
* The authors provide consider diverse steering techniques (e.g., DIM, ACE, CAA, PCA, LAT) for their experiments.
* Novel safety metrics: The framework introduces Effectiveness and Entanglement metrics to assess both steering performance and side effects, which allow tradeoff analysis. However a comparison to related metrics used for safety assessment is missing. 
* The paper reveals that steering effectiveness depends strongly on the combination of model & steering method. No single best method exists, and they discuss some method-specific insights, e.g. DIM offers strong performance but at the cost of safety trade-offs.
* The presented insights can help the interpretability community develop steering interventions with finer behavioral control and fewer unintended harms.

### Weaknesses
* The framework identifies safety tradeoffs  but does not clearly explain why the studied methods are related to specific harms. The underlying causal mechanisms remain often unclear to the reader.
* The evaluation relies on LLM-based scorers, which can introduce bias and reduce interpretability of the results. Wouldn't evaluation with the help of human annotators for at least a subset of results provide more insights? 
* Although the benchmark covers nine safety perspectives, it leaves out some very relevant safety dimensions such as adversarial robustness, long-context reasoning stability, etc.
* The dynamic testing approach uses only small data subsets (20% of each dataset), which might cause some unintended side effects. Also details on selection of the subsets is missing?
* Some methodological details, e.g. how layer selection impacts with steering strength, are not discussed in detail > can result in issues with reproducibility.
* The figures summarize results across models and methods but are are dense and difficult to interpret without further breakdowns

### Questions
1. Have you considered including ablation studiesto show how results change with different parameter/steering settings?

2. Could you provide more details on how LLM judges like GPT-4o or LlamaGuard were prompted across datasets to ensure consistent scoring?

3. Could you include finer-grained results or plots to help interpret the tradeoffs more clearly?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper offers a framework for safety assessment for models that looks at safety across seven dimensions, operationalizing each with a measure. The paper further assesses safety of several large models using this framework and applying steering methods that intervene directly on neural network activations, requiring no training. The paper offers the assessment toolchain as a contribution. A major finding is that safety properties are "entangled" both in this assessment and in prior work, in that improving performance on one operationalization of safety may harm performance on another.

### Strengths
+ The notion that multiple operationalizations of safety may be necessary to make sense of safety claims is important, as is the notion that improvement in one sense may harm performance in another. This is reminiscent of various "no-go" theorems (although there is no formalism offered here, and there could not be because safety claims are epistemically non-falsifiable) that demonstrate e.g. the no-free-lunch theorem or results from Chouldechova (resp. Kleinberg) that it will be impossible to square different operationalizations of bias except under specific theoretical circumstances (e.g., a perfect predictor or already-equalized feature incidence per class stratum in the dataset) or even Kleinberg's clustering impossibility result.
+ The paper undertakes a substantial level of effort with regard to benchmarking and evaluation.

### Weaknesses
- It is never made clear how the seven framework dimensions were selected or why they were operationalized using the specific measures chosen. While each is a topic that has been studied in much prior work, claims that the framework is "comprehensive" must be supported by arguments for the completeness/soundness of the evaluation with regard to some exogenously developed notion of safety hazards/risks not tolerated. Indeed, the "framework" reads more like a wish list, making it hard to evaluate its quality against any other proffered "framework", like the NIST AI RMF or any company's internal framework.
- The framework is offered as a contribution, but there is no effort in the paper to evaluate the framework other than to apply it. It is not possible to understand if the results obtained represent a purposive sample of safety benchmarking or if applying the framework in some application would reduce some kind of risk or avoid some defined hazard (this is the standard approach to evaluating safety in most disciplines, or there is some accepted proxy metric). The paper could, for example, argue in favor of this breakdown of safety issues to show that it covers the bulk of concerns in this literature (while a literature survey is probably a different paper, arguing for the "comprehensive" or "systematic" nature of the taxonomy requires analysis that is not given here).
- I think the paper suffers from a kind of category mistake in which "safety" is a property of the model-under-test and not of the model-in-use in some use case. If it is true (as the paper argues) that improving safety performance in some aspects worsens it in others, the next natural question is about whether "good enough" performance exists for some use case. But here, the models are considered only for the use case of "performance on a selected benchmark" and the paper in its current form does little to link the benchmark contexts to real-world, outside-the-lab use cases or to argue that the chosen benchmarks reflect those use cases well. Instead, a claim of safety is a claim about avoiding certain defined bad outcomes - how does the analysis approach here do when measured against that goal? One thing the framework can help with is mapping between the risks operationalized by the chosen benchmarks and the use cases in which the models-under-test might be applied; in doing so, the framework can help to navigate the tradeoff identified that "there is no universal best method that maximizes effectiveness while minimizing entanglement across all models" [384-386] (a claim so broad it cannot possibly be true).
- Many of the stated contributions in the intro and conclusion are not substantiated by the paper. In addition to the issues of framework comprehensiveness and usefulness mentioned above, there is no effort in the paper to explain why the assessments are "systematic" or "reproducible" or "reliable". These are empirical claims about the use of the framework to evaluate real applications; showing that it is possible to pose questions for which evaluations lead to quantitative results only matters if there is meaning to be made from the resulting quantifications. Here, I think the paper shows its greatest weakness, in that no such meaning is extracted from the many evaluations provided. There may be contribution within these evaluations, but that is not the way the paper presents itself.

### Questions
* Why is this framework "comprehensive" or "systematic", as the paper title and listed contributions claim?
* How would a user of this approach gain capacity to navigate the tradeoffs identified in the evaluations performed using the techniques in this work? (Either across dimensions in the framework or across models - and to what extent does each of those degrees of freedom matter?)
* Can the claims about safety evaluation be toned down such that the evaluation work in the paper supports them sufficiently? How?

### Soundness
1

### Presentation
3

### Contribution
2
