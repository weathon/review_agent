# RocketEval: Efficient automated LLM evaluation via grading checklist

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Evaluating large language models (LLMs) in diverse and challenging scenarios is essential to align them with human preferences. To mitigate the prohibitive costs associated with human evaluations, utilizing a powerful LLM as a judge has emerged as a favored approach. Nevertheless, this methodology encounters several challenges, including substantial expenses, concerns regarding privacy and security, and reproducibility. In this paper, we propose a straightforward, replicable, and accurate automated evaluation method by leveraging a lightweight LLM as the judge, named RocketEval. Initially, we identify that the performance disparity between lightweight and powerful LLMs in evaluation tasks primarily stems from their ability to conduct comprehensive analyses, which is not easily enhanced through techniques such as chain-of-thought reasoning. By reframing the evaluation task as a multi-faceted Q\&A using an instance-specific checklist, we demonstrate that the limited judgment accuracy of lightweight LLMs is largely attributes to high uncertainty and positional bias. To address these challenges, we introduce an automated evaluation process grounded in checklist grading, which is designed to accommodate a variety of scenarios and questions. This process encompasses the creation of checklists, the grading of these checklists by lightweight LLMs, and the reweighting of checklist items to align with the supervised annotations. Our experiments carried out on the automated evaluation benchmarks, MT-Bench and WildBench datasets, reveal that RocketEval, when using $\textit{Gemma-2-2B}$ as the judge, achieves a high correlation (0.965) with human preferences, which is comparable to $\textit{GPT-4o}$. Moreover, RocketEval provides a cost reduction exceeding 50-fold for large-scale evaluation and comparison scenarios. Our code is available at https://github.com/Joinn99/RocketEval-ICLR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose RocketEval to address the high costs, privacy concerns, and reproducibility challenges associated with using powerful LLMs like GPT-4o for evaluation purposes. RocketEval leverages lightweight LLMs combined with a multi-faceted checklist approach to improve judgment accuracy. Experimental results from benchmarks like MT-BENCH and WILDBENCH demonstrate that RocketEval achieves a high correlation with human evaluations, comparable to GPT-4o, while reducing costs by more than 50-fold.

### Strengths
- I enjoyed reading the paper, especially Section 2, as the problem is well-motivated and the charts effectively convey the key messages.
- The idea is simple yet smart. By decomposing a single (potentially difficult) evaluation task into multiple simpler items in a checklist, lightweight LLMs can be leveraged to perform the evaluation. More importantly, checklist generation is largely a one-time effort, which could make LLM evaluation more economical in everyday LLM development environments.
- The authors have thoroughly evaluated the proposed method from multiple perspectives, including agreement, cost, and a qualitative study.

### Weaknesses
- Some aspects of the design lack scientific rigor. For instance, the estimation of “Normalized Probability” does not appear to be justified. Why would it provide an unbiased (or even plausible) probability? I suggest simply referring to it as a “score” instead of “probability”.
- Direct evaluation using lightweight LLMs is critiqued for high uncertainty and position bias. However, the authors fail to provide a comparison in this regard during the evaluation. Could this issue be addressed by bias mitigation algorithms [1, 2] at an even lower cost? What is the unique advantage offered by the checklist-based method compared to those approaches?
- The authors mentioned the potential benefits of using prefix caching for cost savings. However, this is not reflected in Section 4.2, even though this feature is already available through OpenAI.
- In Section 4.2, the authors quote the cost from `vast.ai`, a bidding-based crowd-sourced GPU rental platform. The quoted price ($0.8/hr) is the minimum price for renting an A100 card according to `https://vast.ai/pricing` (why not use the median price?). This might be unfair for several reasons. First, it is unlikely that $0.8/hr represents the regular price. More importantly, OpenAI offers production-level stability, which `vast.ai` does not. A fair comparison would be to quote the price from a public cloud provider like AWS.

[1] Large language models are not fair evaluators

[2] Split and merge: Aligning position biases in large language model based evaluators

### Questions
How does the method compare to other approaches, such as [1, 2], as well as fine-tuned judge LLMs?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces RocketEval, a lightweight and automated evaluation method for LLMs that addresses the high costs and privacy concerns of using powerful LLMs as judges. By reframing evaluation tasks into a Q&A format using instance-specific checklists, RocketEval improves the judgment accuracy of smaller models, overcoming issues like uncertainty and positional bias.

### Strengths
- Clear Motivation: The author clearly explains the motivation of the paper in the introduction. I particularly agree with the point raised about the shortcomings of "Fine-Tuned Judge Models," as these models often lack understanding of complex instructions (although the author did not provide proof for this claim).
- Overall Well-Written: The paper is generally well-written.
- Relevant Problem: The paper raises a highly relevant problem that urgently needs to be addressed in the current landscape of LLM evaluation.

### Weaknesses
- Regarding Section 2.4: While information entropy is a good metric, it is not very intuitive. I recommend that the author use self-consistency (sampling n times and calculating consistency) to demonstrate this uncertainty.
- Limited Advantage Over GPT-4o-Mini: Although RocketEval shows a clear advantage over GPT-4o, its superiority over GPT-4o-mini is less significant. Additionally, the N=1000 setting is impractical, as it is rare for someone to test 1000 models simultaneously.
- Missing Ablation Studies: The paper lacks ablation studies for two components: Independent Checklist Item Judgment and Normalized Probability.
- Comparison with Other Models: I also suggest that the author provide comparisons with other Fine-Tuned Judge Models, such as the Prometheus series.
- Minor Issues: In lines 289-290, the author mentions that reference responses and hand-crafted rubrics may not work well in all cases. Please provide examples to illustrate this. Figure 10 is confusing—why does the first row have three models, while the following subplots have only one model per row?

Overall, I believe the problem and methods proposed in the paper are valuable, but there is room for improvement. I hope the author can address my concerns during the rebuttal stage.

### Questions
Please see the weaknesses.

### Soundness
3

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
4

### Summary
This paper proposes RocketEval, a novel framework for efficiently evaluating large language models (LLMs) using lightweight LLMs as judges. The framework addresses the limitations of existing automated evaluation methods, such as high costs and lack of interpretability, by using checklist grading. This paper trains lightweight LLMs to independently evaluate each checklist item, providing a multifaceted and unbiased judgment.

### Strengths
- Originality: The paper presents a novel evaluation framework, RocketEval, which evaluates LLMs by grading checklist. This approach is distinct from existing methods like multiple-choice questions. However, I believe the idea of using checklist to evaluate NLP models was actually proposed by another paper [1] in 2020.
- Quality: The analysis of the lightweight LLMs' abilities in section 2 provides valuable insights into the framework’s effectiveness.
- Clarity: The paper is well-organized and clearly structured, with each section logically following the previous one.
- Significance: The framework’s high agreement with human judgments and significant cost reduction make it a promising solution for large-scale LLM evaluations.


[1] Ribeiro, Marco Tulio, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. "Beyond accuracy: Behavioral testing of NLP models with CheckList." arXiv preprint arXiv:2005.04118 (2020).

### Weaknesses
- The paper primarily focuses on evaluating responses to queries, as the checklist is generated by the input query. However, LLMs are used in a wide range of applications, including text summarization, and translation, where evaluation metrics based on queries may not be efficient since the input query may be very long.
- Since authors employ GPT-4o as the checklist creator, while the paper argues that checklist creation is a one-time process, the cost of using a powerful LLM like GPT-4o for this task could be significant, especially for large-scale evaluations. Exploring more efficient methods for checklist creation would be beneficial.
- Typo: Line 215: We select responses from 12 test models and compare the benchmark results when using *GPT-4o* and *Claude-3.5-Sonnet* as judges as strong baselines.

### Questions
- Could authors explain their framework's relation and difference with paper [1] ?
- RocketEval can enable lightweight models the ability to judge large models, is this similar to the idea of weak-to-strong supervision? Please explain the relation and difference between your paper's ideas and weak-to-strong [2].
- Conducting experiments solely on a total of 1104 data from MT-BENCH and WILDBENCH is not convincing enough. Could authors conduct experiments on more datasets?

[1] Ribeiro, Marco Tulio, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. "Beyond accuracy: Behavioral testing of NLP models with CheckList." arXiv preprint arXiv:2005.04118 (2020).

[2] Burns, Collin, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen et al. "Weak-to-strong generalization: Eliciting strong capabilities with weak supervision." arXiv preprint arXiv:2312.09390 (2023).

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose RocketEval, a system comprised of API calls and lightweight models, performing as a model judge. The system is designed to be minimally computationally expensive while achieving comparable human agreement scores to GPT4o on MT-Bench and WildBench. Their subsequent ablations and analysis inform how the system can achieve these efficiency gains.

### Strengths
* The authors core objective is important and compellingly achieved: create a lightweight model judge with high performance. Their use of a dynamic checklist, and normalized score predictions both show notable improvements in scoring, from their experiments.
* The authors unpack the issues with existing lightweight model judges, related to their analyses, position bias and uncertainty, that could be informative to other solutions as well.
* The comprehensively compare a series of model judges in different settings to understand how they can best be optimized. Their quantitative analysis and ablations are particularly informative.

### Weaknesses
* Their explanation for lightweight model “analysis” being weak could be framed more conservatively. If I understand correctly, these experiments more show that lightweight models find classifying GPT4o reviews into scores as an easier task than predicting the score themselves. It wouldn’t necessarily naturally follow that a GPT4o checklist would be the obvious solution, but rather a GPT4o reasoning.
* It isn’t immediately apparent why GPT4o is so much more expensive than the RocketEval (which includes GPT4o calls). This should be more closely explained. Is it because the reasoning is super long? If so, it would be beneficial to see harder baselines (in terms of efficiency), e.g. prompting GPT4o to produce shorter CoT explanations before its score prediction.

### Questions
In Table 3, human-human agreement seems very low. Does this saturate the performances above 60% to some extent?

### Soundness
2

### Presentation
3

### Contribution
2
