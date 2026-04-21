# Exploring the Recall of Language Models: Case Study on Molecules

- Avg Score: 4.40
- Decision: Reject
- Scores: 6, 3, 5, 3, 5

## Abstract
Most of the current benchmarks evaluate Generative Language Models based on the accuracy of the generated output. However, in some scenarios, it is also important to evaluate the recall of the generations, i.e., whether a model can generate all correct outputs, such as all security vulnerabilities of a given codebase. There are two challenges in evaluating the recall: the lack of complete sets of correct outputs for any task and the existence of many distinct but similar outputs (e.g., two exploits that target the same vulnerability).

In this paper, we propose a benchmark from the domain of small organic molecules. We define several sets of molecules of varying complexity and fine-tune language models on subsets of those sets. We attempt to generate as many molecules from the target sets as possible and measure the recall, i.e., the percentage of generated molecules from the target set. We examine the impact of the training loss function and sampling strategy on the recall. We propose a sampling strategy based on beam search that avoids duplicates and maximizes recall. Finally, we show that given a small validation set, one can predict the recall of the model without actually generating many samples, which can act as a model selection strategy for maximizing generation recall.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies the challenges in evaluating the recall of generative models and introduces a recall benchmark in the domain of molecular generation. It also proposes sampling strategies and loss formulations to enhance recall.

### Strengths
This paper is well-written and easy to understand, addressing a problem that has not been extensively explored before. Additionally, the paper addresses crucial research directions, such as measuring recall without generation and methods to enhance recall, presenting intriguing experimental results.

### Weaknesses
**Scalability of Research**

The study in this paper is limited to a specific domain, namely molecular generation, and there needs to be a discussion on how this research can be extended to other domains. For example, a crucial aspect of measuring recall, as highlighted in the paper, is identifying the equivalence class of the model’s generated results. As mentioned in lines 60-62, there is a technique for identifying equivalence classes for SELFIES strings. How could this issue be addressed in other domains you mentioned in the introduction, such as “vulnerable code generation”?


**Completeness in Method**

In my opinion, the sections proposing the sampling strategy and loss to improve the model’s recall are crucial for establishing the novelty of your paper. However, these aspects are not fully developed and lack sufficient explanation. For instance, in the case of the recall-oriented loss function, the approach of changing the aggregation to min or max seems quite extreme to me, with significant potential for refinement. Additionally, the proposed method only showed effectiveness for a very small and underperforming model with 800K parameters. Therefore, improvements in this area are essential. Additionally, the motivation for using beam search in recall-oriented generation and the intuition behind why increasing the beam size leads to improved recall need to be more thoroughly explained.

**Evaluation**

Most experiments in this paper are validated using a single model and dataset, making it difficult to consider the proposed benchmark method and the approaches to improve recall as thoroughly validated. I believe there should be verification to ensure that the trends in the experimental results hold consistently across at least several models.
Additionally, there are confusing aspects regarding the details of the experiments, which should be described and justified more comprehensively (see the questions section for more details).

### Questions
- In my understanding, the process you described in lines 236-237 is aimed at generating the set of every correct generation,  $\mathbb S$ , for evaluation purposes. Is this correct? Additionally, how can you ensure that the generated results accurately represent every correct generation?

- As shown in Table 2, recall shows a correlation with the complexity of molecules, whereas precision does not. Is there a specific reason for this? I’m curious about which aspects of the recall metric lead to this outcome.

- What is the input to the model when performing generation with an LLM for recall/precision evaluation?

- What exactly is the purpose of the validation set mentioned in line 220, and is there a specific reason for using only 10,000 instances?

- How does the cost (time complexity, memory, etc.) change with the beam size in 4.3?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces a new benchmark of molecules for evaluating generative language models with a focus on recall. It aims to investigate the model's ability on tasks requiring distinct output generation, like detecting all vulnerabilities in code. Using organic molecule dataset, the study shows that model recall can be anticipated via perplexity on a validation set. Moreover, the authors use beam search decoding to reduce duplicates and a recall-aware loss function to improve performance, providing insights into molecular representation and model pretraining effects.

### Strengths
This paper presents a meaningful investigation into the recall of model generation, with a well-articulated and compelling motivation.

### Weaknesses
1. From section 3.1 onward, this paper becomes quite difficult to follow, largely due to the use of specialized terminology from fields like chemistry without providing sufficient foundational overviews or introductory explanations. This approach makes it challenging for readers to fully grasp the content and nuances of the work. For instance, important details and statistics regarding the dataset collected by the authors are not included, and terms like SELFIES are mentioned without any straightforward elaboration to help readers understand what SELFIES actually represents. This lack of accessible explanations hinders the reader’s ability to form a clear understanding of the paper’s specifics. I recommend that the authors incorporate diagrams or more detailed descriptions of key terminology to enhance clarity.

2.In section 4.2, a new method for estimating recall is proposed. First, the statement "Given that evaluating recall provides a meaningful and interpretable measure of an approach’s ability to model data, estimating recall without needing to perform generations would be useful" lacks a convincing motivation for why recall estimation without actual generation is necessary. There is no clear justification for the need to use an alternative method to evaluate recall. Furthermore, using probability to estimate recall does not align with the standard definition of recall, which traditionally measures the proportion of correctly generated instances rather than a probabilistic expectation. Thus, it is both imprecise and misleading to label this metric as recall. For instance, in earlier sections (Table 2), the authors appear to use a conventional method for calculating recall; however, after introducing this new approach, they apply it in Table 4 but use the same metric name. This inconsistency undermines reliability and creates confusion regarding the validity of the reported recall values.


3.In section 4.3, I don’t see a substantial difference between your proposed recall-oriented generation and the standard beam search. 

4. Mean aggregation is equivalent to the regular loss function" lack clarity—specifically, it is not defined what the “regular loss function” refers to. Furthermore, the section does not directly present the actual loss function or provide a detailed explanation. Instead, it relies solely on textual descriptions, which makes it difficult to understand the specifics of the proposed loss. Including the explicit mathematical form of the loss function along with a step-by-step explanation would significantly improve clarity and accessibility.

5.In addition to the presentation issues mentioned above, the paper lacks a coherent structure throughout both the methods and experiments sections. The presentation feels fragmented, and critical details regarding the experimental setup, such as baseline configurations, are insufficiently described. To improve clarity, a major revision is needed to reorganize the paper, providing a more cohesive structure and a thorough explanation of the experimental settings.

### Questions
Please refer to the weakness part

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper introduces a benchmark for evaluating models based on recall rather than just accuracy. The authors tackle two challenges: the lack of complete correct output sets and the presence of multiple similar outputs. Using small organic molecules from the GDB-13 database, they fine-tune models and develop a method to predict recall based on perplexity. They also propose a novel beam search decoding method to maximize recall by avoiding duplicates, alongside a recall-aware loss function. This approach aims to enhance the ability of GLMs to generate all correct outputs, with potential applications in various fields, including security.

### Strengths
- This paper explores the evaluation of recall rates for small language models, which is a meaningful endeavor.
- The paper investigates various methods to enhance the recall rates of models and has achieved some positive results.

### Weaknesses
- The contributions of this paper are limited. On one hand, in improving recall through sampling methods and loss functions, the authors merely attempt different strategies, which can sometimes harm precision, and no solutions are provided. On the other hand, the improvements through fine-tuning appear to offer no significant contribution, as it is generally expected that fine-tuning would enhance performance on a specific task.
- The model is too singular, as the experiments in this paper only include the OPT-1.3B model. Therefore, the evaluation results and methods for enhancing recall may not generalize well.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents a benchmark for modelling molecules, based on GDB-13 (an exhaustive set of molecules with at most 13 heavy atoms that satisfy certain conditions). The authors pretrained LMs to generate the molecule sequences, and aim to bring up recall via 1) better sampling in generation and 2) better training data design. In addition to that, the authors proposed ways to predict the recall value with a small-scale experiment and a set of empirical studies on how should one best represent the molecules in LM inputs.

### Strengths
1. Maximizing recall is indeed valuable for a lot of applications, as the authors discussed in the paper, this paper is of empirical importance.
2. The formulation of the problem is novel, the molecular generation domain provides an excellent testbed due to well-defined equivalence classes and complete reference sets.
3. The experiments are done with rigor. I like the comprehensive analysis of factors affecting recall (pretraining, molecular representations, etc.)
4. The dataset and benchmark would make a good contribution to the community.

### Weaknesses
My main concern with this paper is around its technical contributions:
1. The author proposed using random sampling with temperature and beam search (with a large beam size) to improve recall coverage. These two methods are well-known methods in language models' (LM) generation, and I was expecting a novel generation approach such as generating with penalizing the likelihood of already generated sequences.
2. The method that predicts recall has a lot of similarities with perplexity measure in language modelling, would the authors clarify how is the proposed metric different from the perplexity-based measures?
3. Removing duplicates and selecting data in each batch are sensible approaches, but they don't appear to be anything novel.

I have some minor questions listed in the below section.

### Questions
1. In figure 2, the authors stated that "The plot indicates that the recall is close to saturation at 10 million generations, implying that this model will not cover 90% of the molecules even with 50 million generations." To me, the coverage function is naturally sub-linear, as you repeatedly take samples from a fixed distribution, the likelihood of getting a new unseen sample gradually goes down, so I am not sure if this (the sublinear trend) is a problem. And if it is, does the authors' proposed approach improves the trend to be somewhat linear? I think that will be an exciting result to see.

2. SMILES v.s. SELFIES. I am not expert on the molecule modelling topic, but from Table 7, it seems SMILES works better than SELFIES when the data is in Canonical form, so why choose SELFIES as the main representation form?

3. Writings:
[Line 76], (Remove "Finally"?) Finally, LLMs have recently demonstrated strong performance on these tasks
[Line 310] I am not sure this expression = "an average probability", looks like a sum of probabilities.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a benchmark for evaluating the recall of language models in the domain of small organic molecules. Specifically, based on the famous dataset GDB-13, the authors prepare a new dataset with four subsets, e.g., a new subset contains molecules that share a certain percentage of substructures with aspirin. Based on the constructed dataset, the molecule generation capability of language models (LMs) in terms of recall before and after fine-tuning has been evaluated. A new method for predicting the recall of LMs has also been designed. The average probability of a desired molecule to be generated and the ground truth recall values are used to build a regression model for the recall prediction. The evaluation demonstrated the correlation is more than 0.99. Finally, a recall-oriented molecule generation method and a loss function have been introduced to boost the recall of LMs.

### Strengths
1. An interesting and important problem in analyzing the recall of language models.
2. Multiple solutions with promising results have been proposed in the same work
3. The paper is well-written

### Weaknesses
1. Even though the motivation is clear and good, the studied objective does not fit the motivation well, is the recall metric more important in the molecule generation domain?
2. Many design choices are unclear, e.g., why use Beam search in section 3.4 not others?
3. Many problems, e.g., capability estimation and new loss design, have been studied, but each of them lacks a comparison with baselines.

Overall, this paper studies an important problem and proposes promising solutions for recall estimation and LMs enhancement. However, there are some concerns that need to be addressed.

Firstly, even though the main point, evaluating whether a model can generate all correct outputs is important for safety-critical problems, it is unclear whether this is the case for the studied objective molecule generation. It is better to give clear motivation for the importance of evaluating recall for this task. 

For the subset construction, in Table 1, it is unclear how the threshold is determined, e.g., 0.4 for Sasp and 0.2 ≤ sim(m, d) ≤ 0.2165. Please clarify it.

In Section 4.1, Table 2 and Table 3 suggest different solutions as the best, which one we should accept in practice. It is better to add more discussion here.

In Section 4.2, considering the recall estimation, there are many works that have been proposed to evaluate deep learning models in an unsupervised manner [1, 2, 3], it is necessary to at least discuss the difference between the proposed method and these works.

In Section 4.3, it is unclear why Beam search is used here since there are many other options (search methods). 

In Section 4.4, first, it is better to add baselines without using the designed loss function in Table 5. Besides, the recall values decreased after comparing the results in Table 5 and Table 4. It is unclear which factors lead to this degradation. 
 
[1] Unsupervised Evaluation of Code LLMs with Round-Trip Correctness.	
[2] Estimating Model Performance Under Covariate Shift Without Labels.
[3] Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift

### Questions
Please check my comments above.

### Soundness
2

### Presentation
3

### Contribution
2
