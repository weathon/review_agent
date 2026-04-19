# Can LLMs Effectively Leverage Graph Structural Information: When and Why

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 6, 6

## Abstract
This paper studies Large Language Models (LLMs) augmented with structured data--particularly graphs--a crucial data modality that remains underexplored in the LLM literature. We aim to understand when and why the incorporation of structural information inherent in graph data can improve the prediction performance of LLMs on node classification tasks with textual features. To address the "when" question, we examine a variety of prompting methods for encoding structural information, in settings where textual node features are either rich or scarce. For the "why" questions, we probe into two potential contributing factors to the LLM performance: data leakage and homophily. Our exploration of these questions reveals that (i) LLMs can benefit from structural information, especially when textual node features are scarce; (ii) there is no substantial evidence indicating that the performance of LLMs is significantly attributed to data leakage; and (iii) the performance of LLMs on a target node is strongly positively related to the local homophily ratio of the node.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper offers valuable insights into the utilization of graph structural information by large language models (LLMs), focusing on the context of node classification on text attributed graphs. In addressing the "when" aspect, the authors investigate the impact of different prompt designs and the richness of textual data. Regarding the "why" aspect, the study delves into the effects of label leakage and homogeneity.

### Strengths
- Significance: The paper's exploration of the "when" and "why" behind LLMs' effectiveness in graph-related tasks holds immense significance. Given the increasing prominence of Graph+LLMs in various applications, the insights provided by this work are timely and valuable.

- Quality. The authors dissect the problem into two crucial aspects: "when" and "why." They provide a clear and well-reasoned argument, accompanied by rigorous experimental and analytical support, such as the introduction of the new dataset arxiv-2023.

- Clarity. The paper is  well-structured and clear in its presentation

### Weaknesses
- **Overlap with Previous Work.** Several experiments and observations made in the paper have already been addressed in previous or concurrent works, such as [1], [2], and [3]. These prior works have explored topics like the effect of different prompt designs and settings, including zero-shot, few-shot, with k-hop neighbor information, and with title or title and abstract. It is acknowledged that the paper does introduce some unique elements, such as prompt designs with attention analogous to Graph Attention Networks (GAT) and providing more in-depth analysis. However, the experiments appear to have significant overlaps with existing research, leading to a perception of a relatively trivial contribution.
- **Spurious Analysis.**
- Q1: "If data leakage is a major contributor of performance on OGBN-ARXIV, we would expect prompting methods based on LLMs to perform worse than MPNNs on ARXIV-2023."
This argument regarding data leakage and the expected performance of LLMs compared to MPNNs on a new dataset is not convincingly made. We would expect LLMs to perform worse than MPNNs on ARXIV-2023 because there might be no causality between MPNN doing good/bad and LLM also doing good/bad. 
- Q2: The statement, "our findings underline the critical role of homophily in influencing LLM’s node classification performance," is made, but it may oversimplify the issue.  While homophily's influence is acknowledged, it's essential to delve deeper into why LLMs are affected by homophily. It is suggested that LLMs tend to employ a simple majority vote approach when provided with neighbor information, regardless of the design of prompts, due to not being directly trained on structured (graph) data. 

Reference:

[1] Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning (Table 8) [arxiv link](https://arxiv.org/abs/2305.19523) 

[2] Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs (Table 10, Table 16) [arxiv link](https://arxiv.org/abs/2307.03393)

[3] GraphText: Graph Reasoning in Text Space (Table 1) [arxiv link](https://arxiv.org/abs/2310.01089)

### Questions
For Q1 and Q2, see weakness above.

Q3: In the new arXiv-2023 datasets, you mention, "Specifically, we first sample test nodes from arXiv CS papers published in 2023, and then gather papers within a 2-hop of these test nodes to create a citation network." Could you please provide the exact number of nodes that belong to the year 2023 in this dataset?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents an experimental study on leveraging structured information (specifically graphs) in LLMs and analyzed the impact of structured information on the performance of LLM, specifically on the node classification task.  The general idea is to augment the textual features of a given node in the prompt with the structured information from the graph by describing the list of k-hop neighbors (grouped by hop levels) along with their textual features. The paper conducts their experimental study using ChatGPT 3.5 turbo on  our popular public graph-based datasets (OGBN-ARXIV, CORA, PUBMED, and OGBN-PRODUCT). Based on their experimental observations, following major conclusions were drawn from the paper: i) LLMs can potentially benefit from structural information, especially when textual features on the individual nodes  is scarce, ii) LLMs performance improves further on graphs that have higher local homphily ratio, i.e. graphs where nodes with similar features are more likely to be connected by an edge, and iii) No conclusive evidence was observed to attribute improvement of performance in LLMs to data leakage.

### Strengths
1. The paper presents a  detailed experimental study on the impact of structural information on LLM's performance on node classification tasks while exploring several aspects of the problem, including: i) the impact of local homophily ratio on LLM performance, ii) altering textual information on the node, iii) data-leakage impact (i.e. possibility of test dataset being used in LLM training )
2. The insights seem to be logical and intuitive and I see fair experimental setup being used to backup the claims.

### Weaknesses
1. Lack of significant contributions. My biggest concern about this paper is that the conclusions derived from the paper are quite trivial and unsurprising. For instance, the observation that stronger improvements in the performance on data with scarce textual features have been made even in the context of language models prior to LMs. I don't see why this wouldn't have hold for LLMs? Similarly, the observation regarding higher accuracies of LLM predictions on  nodes with higher local homophilic ratio is also a common observation in the graph-mining literature. Finally, in the third insight of this paper, there is no concrete evidence on whether data-leakage could potentially boost capabilities LLMs in leveraging structured information. The lack of evidence is only applicable to prompt styles explored in this paper, but there is no evidence to make any general claims (as also recognized by the authors).

2. The scope of the study is limited to the task of node classification. What about leveraging graph-based information in LLMs in other node tasks such as forecasting or regression problems? Similarly, how could graph based information help LLMs in link prediction problems and its applications in recommendation systems (e.g. predicting if a user will like a movie)?

Some other specific comment: 
1. Page 4 line 174: It seems like there is another strategy called k-hop attention strategy which is not described here, but somehow being reffered as "second strategy" in the next paragraph.

### Questions
None.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the problem of enabling Large Language Models (LLMs) to process graph-structured data. Specifically, this paper studies the potential effects on node classification accuracy with different prompting methods. The paper also studies the effect of data leakage and homophily.

### Strengths
S1. This paper is easy to follow.

S2. The proposed methods work reasonably well.

S3. The topic combining Graph and LLM is interesting.

### Weaknesses
W1. The review feels the motivation of this paper is not clear. Although LLM is popular, why we need LLMs for node classification tasks is not well-explained. The authors only focus on one node classification task. Given that a simple GNN can already yield satisfactory results for this task, the rationale for introducing LLM, which may be slower in inference, is unclear.

W2. This paper lacks solid explanations and analysis of the experimental results. For example, why do different prompts result in different performance? Is there any theoretical analysis or intuitions? Why can LLM outperform GNN on ARXIV-2023 while not on OGBN-ARXIV? The current version of this paper only provides an experiment report without solid analysis.

W3. The experimental analysis is also weak. The experiments in section 3.4 cannot really show the impact of homophily on LLM's classification accuracy. It may show that incorporating the context information with the same label can improve performance. The authors should compare more ablations with context samples which have the same label.

W4. The authors study the problem of incorporating structural information (such as graphs) to improve the predictive accuracy of LLMs but only conduct experiments in node classification. Some experimental results might provide better understanding. For example, checking the performance of other competitive LLM methods on graphs and downstream tasks.

### Questions
Please see the above.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper investigates how Large Language Models (LLMs) can leverage graph structural information for node classification tasks with textual features.  The authors investigate the "when" and "why" of incorporating structured data, particularly graphs, into LLMs. They examine different prompting methods and factors that affect the performance of LLMs, such as data leakage, homophily (nodes with similar characteristics). Through controlled experiments and correlational analyses, the authors  establish a positive relationship between the local homophily ratio and the prediction accuracy of LLMs, i.e., more homophily incorporated in the prompt, the more accurate the prediction becomes. While data leakage is a potential concern , the authors observe that LLMs' consistent performance across different datasets suggests that they are robust and can perform well across varying distribution domains. Finally, the authors provide insights and suggestions for future research on LLMs and graph data.

### Strengths
* The paper is well-written, clear, and organized. The authors present their motivation, approach, methodology, results, and contributions in a logical and coherent manner.
* The paper addresses an interesting topic of integrating LLMs with graph data, which is a crucial data modality that can provide additional information to LLMs.
* The evaluation results provide valuable insights into the potential and limitations of LLMs in leveraging graph structural information, as well as the challenges and opportunities for future research on LLMs and graph data.

### Weaknesses
One of the limitations is that the study only focuses on node classification tasks and does not explore other graph-related tasks. It is unclear whether these techniques can be generalized to other tasks.

### Questions
Do you think other Large language models such as LLaMA or Falcon robust against data leakage?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents an empirical study on how LLMs can capture information contained in graph structures. Specifically, the authors consider the task of node classification with textual features as node features. The authors verify that with neighbor information to complement textual features, the performances of LLMs can be improved, especially under the scenario where scarce textural information is given. The authors then analyze where the LLMs performance on graphs come from, and come to two conclusions: the performance of LLMs on graphs is not related to data leakage, and homophily contributes to the performance of LLMs on graph-structured data.

### Strengths
1. Timely topic. It is interesting to explore the potentials of LLM on a wide range of tasks, and graph node classification task with rich textual information is a suitable topic for LLMs, in that it is closely related to texts and incorporates different data structures. 
2. Good organization. The paper is well-organized and easy to follow. It is easy for me to understand the main ideas and conclusions/arguments of this paper. 
3. The 'data leakage' aspect of this paper is interesting and a somewhat unique aspect in the combination of LLMs and graphs. Indeed, in the area of traditional graph learning, data leakage is not an issue. Therefore, I appreciate the authors' efforts in identifying such a potential cause and perform experiments.

### Weaknesses
1. The conclusions of this paper are somewhat not surprising. As LLMs work on texts, it is intuitive that with rich texts, LLMs can already well understand the node, and when there are scarce texts, some additional related texts can further boost the performance of LLM. Also, the conclusion about homophily is also not surprising: bringing heterophilous nodes to the context will distract LLM and thus compromise the accuracy. I am not saying that the contribution of this paper is limited --- **it is a solid paper with good contributions**, but the conclusions are just not that surprising, which would slightly reduce my rating. 
2. Studying the behavior of ChatGPT indeed leads to better understanding of LLMs on graphs. However, as ChatGPT is not open source, it would be better if the authors can also perform experiments on open-source models, such as LLaMA. In that way, the insights may better lead to more concrete efforts in combining LLMs with graph learning, as LLaMA can be more easily fine-tuned to fit the need of graph data understanding. Also, as LLMs are sensitive to prompts,  there is no guarantee that the conclusions on ChatGPT can generalize to other LLMs. In this sense, it is always better to do experiments on a wider range of LLMs to ensure that the conclusion is general enough.

### Questions
1. Do you observe cases where ChatGPT fails to follow the instructions and generates stuff that cannot be automatically parsed? How do you deal with these samples?
2. LLMs may be sensitive to the order of contexts. Have you tried to modify the order of 1-hop, 2-hop neighbors in the 1/2-hop title+label setting? Does it lead to significant performance changes?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
