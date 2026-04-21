# Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning

- Avg Score: 5.25
- Decision: Reject
- Scores: 6, 5, 5, 5

## Abstract
In this work, we address the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences and associated model capabilities e.g., copyrighted data or harmful content generation) while preserving essential model utilities, without the need for retraining from scratch. Despite the growing need for LLM unlearning, a principled optimization framework remains lacking. To this end, we revisit the state-of-the-art approach, negative preference optimization (NPO), and identify the issue of reference model bias, which could undermine NPO's effectiveness, particularly when unlearning forget data of varying difficulty.  Given that, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that  `simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We also provide deeper insights into SimNPO's advantages, supported by analysis using mixtures of Markov chains. Furthermore, we present extensive experiments validating    SimNPO's superiority over existing unlearning baselines in benchmarks like TOFU and MUSE, and robustness against relearning  attacks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an innovative approach to Large Language Model (LLM) unlearning, called SimNPO, designed to remove undesirable data influences efficiently. This method advances the state-of-the-art Negative Preference Optimization (NPO) framework by addressing a key limitation: reference model bias. The authors introduce a reference-free approach, simplifying preference optimization without relying on a baseline model, which enhances unlearning robustness while retaining model utility. Extensive experiments on benchmarks like TOFU and MUSE demonstrate SimNPO's effectiveness in surpassing existing unlearning baselines, and it shows promise in defending against relearning attacks.

### Strengths
1. SimNPO provides a unique, reference-free framework that improves upon the limitations of NPO, specifically addressing reference model bias, making it a significant advancement in the field of unlearning for LLMs.

2. The approach is computationally efficient, allowing for effective unlearning without requiring full model retraining, which is resource-intensive and often impractical.

3. The method is tested on diverse datasets (TOFU, MUSE, WMDP), and results are detailed, showcasing the framework’s strong performance in preserving model utility and achieving high forget quality.

### Weaknesses
1. The technical contributions are sound but limited. The authors propose a reference-free optimization framework. However, optimization with a reference-free objective is not novel and has already been proposed by SimPO. Moreover, the length-normalized reward and reward margin parameter in SimNPO are also derived from SimPO.

2. SimNPO introduces two additional hyperparameters that must be manually adjusted to achieve optimal performance, significantly increasing the complexity and time cost of experiments. This greatly limits the potential application of this method for large language models in real-world settings.

3. Is there any theoretical analysis or intuition? What is the divergence rate of SimNPO compared to NPO? The current version of this paper only provides an experimental report without solid analysis.

4. A potential issue is over-reliance on the reference model; would adjusting the weight of the reference model help mitigate this problem? While removing the reference model entirely might solve the over-reliance issue, could it lead to insufficient performance for certain cases, such as math and coding datasets?

### Questions
Please see above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper focuses on LLM unlearning, involving removing unwanted data influences or specific model capabilities from pre-trained models without needing to retraining from scratch. The authors state that NPO, the current state-of-the-art, suffers from reference model bias, and propose a new optimisation framework named SimNPO. SimNPO removes the dependence on the defence model by leveraging a reference-free optimisation method named SimPO, further normalising the loss for the varying lengths of strings. The authors conducted extensive experiments over a series of well established benchmarks, including TOFU, MUSE, WMDP, where the results show the superior performance of SimNPO over NPO and many other advanced methods.

### Strengths
The paper identifies and thoroughly analyzes the drawbacks inherent in NPO, seemingly providing some new insights.

By leveraging SimPO, SimNPO offers a simple yet effective solution that removes the reliance on a reference model, leading to more controlled and efficient unlearning.

Experiments show that SimNPO consistently achieves higher forget quality across various benchmarks, especially in scenarios with varying data difficulties.

SimNPO better preserves the model's utility on the retain set compared to NPO, maintaining performance on essential tasks.

### Weaknesses
“Divergence” seems to be an important terminology that is frequently mentioned throughout the manuscript, while I cannot find its clear definition. Similar issues appear for the terminology of "reference model bias". Do the authors indicate that the \pi_ref is inaccurate and thus the estimation of the weight in Eq 3 is inaccurate? Kindly please provide more detailed descriptions. 

In L1 of Sec 4, the authors state that the reliance on \pi_ref lead to an uneven distribution of unlearning power. However,  it seems that \pi_ref allocates different attention for different strings: when we consider two strings with similar values of \pi_\theta, then the string with large \pi_ref will have smaller attention. Therefore, \pi_\theta can characterise the unlearning difficulty. The following example is also very weird, the authors assume a forget sample that has already been unlearned in \pi_ref. Generally, if a data point has been unlearned (or does not appear in the original model), why we further need unlearning. 

I have another concern in L1, does the short string is generally harder to be unlearned than a long string? The validation experiments should be conducted across a broad range of current unlearning methods and difference metrics (as some metrics may have bias towards short / long strings). Personally, I think it is an important experiments because if the reverse is true, NPO can actually address the difficulties in unlearning long strings. 

In L2, I wonder in what condition we can have an optimal gradient weight smoothing scheme. Also, does the proposed method can fulfil this goal?

I think the paper may some what overclaim its contribution. It seems that the main claim of this paper is that current methods might not be effective in handling short strings unlearning, motivating the paper to propose a new method that explicitly pay more attentions to those long strings. The current organisation of the paper is not clear, especially considering the fact that the authors do not discuss what kinds of the weighting mechanism is of our ultimate interest. 

The authors claim that NPO suffers from over-unlearning. However, comparing the results in Table 3 between NPO and SimNPO for MUSE News, it seems that SimNPO faces the problem of under-unlearning.

To address L1 and L2, a simpler weighting strategy is \frac{1}{|y|}\cdot \pi_\theta (y|x), where  \frac{1}{|y|} handles L1 and \pi_\theta (y|x) handles L2. Compared with such a simple baseline, what are the superiority of the proposed method.

### Questions
Kindly please refer to the Weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper focuses on large language model unlearning (LLM unlearning). Specifically, this work revisits a conventional optimization method, i.e., negative preference optimization (NPO), and identifies the issue of reference model bias that may hinder the performance of unlearning. To this end, this work proposes a new approach, namely, SimNPO, that extends NPO with a reference-free optimization framework like SimPO. Various experiments about the problem illustration and the performance evaluation of the newly proposed SimNPO are conducted to demonstrate the rationality of the proposal.

### Strengths
1. This work identifies one important issue with the conventional LLM unlearning method, NPO, regarding its reference model bias, which provides some insights into the learning behaviors of NPO on the allocation of unlearning power as well as the ineffective gradient weight smoothing.
2. It is reasonable that the SimNPO is promising in mitigating the reference model bias by considering the idea from SimPO to modify the original learning objective of NPO and the presentation also provides the insight discussion via the gradient derivation of SimNPO, as well as the further analysis of a mixture of Markov chains.
3. The experiments are conducted on three LLM unlearning benchmarks, which demonstrate the proposed method's general effectiveness.

### Weaknesses
1. Some statements and claims are not very clear and sometimes confusing. It would be better if the authors could improve the explanation or discussion for some words or sentences. The specific questions can refer to the Questions part.
2. The current presentation is not very good as it is difficult to understand some critical parts of the problem illustration or justification with long-distance cross-reference. For example, there are no related descriptions before or near the position where Fig.1 existed for the first time, and the corresponding caption can not fully explain all the meanings of the figure. Generally, it is hard to understand the major motivation (related to reference model bias) in the introduction part. 
3. It seems that SimNPO is proposed based on NPO and adopts the SimPO to mitigate the reference model bias. The current presentation provides limited unique problems under the LLM unlearning scenario. In other words, SimPO is a reference-free preference optimization method, it can also mitigate the reference model bias if it exists in the framework of DPO. However, it is unclear about the unique novelty and insights of adopting SimPO to improve NPO in both perspectives of research question and methodology design.
4. The identified issue of NPO is reference model bias, while the reference model bias is not clearly illustrated in Section 4. For example, although we can get that both the forget quality and model utility of SimNPO would be better than NPO, there are only statistics of NPO at Figure 3 (a)-(b). Lacking of comparison makes it hard to intuitively get the meaning of blind allocation and ineffective gradient weight smoothing.

### Questions
1. I do not very understand what is the meaning of the statement "a principled optimization framework remains lacking" in the abstract. Although SimNPO revisited the NPO and improved it, would SimNPO be the principled optimization framework? 
2. Could the author provide some explanation or illustration of the word "divergence" in line 55? Since it is italicized, is there any special definition or meaning for this word? And what is the meaning of "divergence-driven optimization methods"?
3. Could the author further explain the Figure 1(a)? The current version of the caption and description is hard to understand, and I can hard to get "overreliance on the reference model" as indicated in lines 76-77. It would be better if the authors could also provide some real examples instead of illustrations.
4. Are there any critical differences between the parallel improvement of SimPO from DPO and that of SimNPO from NPO? If not, does it mean that all the preference optimization methods can be applied in LLM unlearning?
5. Would the reference model bias just be an issue of a specific large language model? Could the issue generally be identified in different large language models? 
6. I wonder whether there are results about statistical significance for the performance evaluation part?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors revisited the current unlearning optimization framework, negative preference optimization (NPO), and identified its reference model bias issue, which compromises unlearning effectiveness, particularly for forget data of varying difficulty. To address this, the authors introduced SimNPO, a simple yet effective framework that eliminates reliance on a reference model by leveraging simple preference optimization. The authors provided deep insights into SimNPO’s advantages through both synthetic data analysis and evaluations on existing unlearning benchmarks such as TOFU, MUSE, WMDP, and relearning attacks.

### Strengths
1. The authors revisit the NPO framework and identify its potential weakness–overreliance on the reference model–in LLM unlearning.

2. Building on insights into NPO’s limitations, the authors propose an improved LLM unlearning approach,
SimNPO, which extends NPO using a reference-free optimization framework.

3. The authors conduct extensive experiments to showcase the improvements of SimNPO over NPO across various unlearning benchmarks.

### Weaknesses
1.	Unclear Motivation: The primary issue lies in the unclear motivation of the paper. While it claims that the main limitation of NPO is the inductive bias of the reference model, Section 4 does not clearly convey this point, making it difficult for readers to understand.

2.	Missing Ablation and Sensitivity Analysis: The paper lacks both an ablation study and parameter sensitivity experiments, which are necessary to evaluate the impact of different components and parameter choices on the model’s performance.

3.	Limited Technical Contribution: The proposed SimNPO framework appears to be a straightforward combination of SimPO and NPO, with no substantial novel insights. Additionally, it lacks a deeper theoretical analysis to support its approach.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
