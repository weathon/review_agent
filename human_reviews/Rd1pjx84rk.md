# Size Generalization of Graph Neural Networks on Biological Data: Insights and Practices from the Spectral Perspective

- Decision: Reject
- Scores: 6, 5, 3, 6

## Abstract
We investigate size-induced distribution shifts in graphs and assess their impact on the ability of graph neural networks (GNNs) to generalize to larger graphs relative to the training data. Existing literature presents conflicting conclusions on GNNs’ size generalizability, primarily due to disparities in application domains and underlying assumptions concerning size-induced distribution shifts. Motivated by this, we take a data-driven approach: we focus on real biological datasets and seek to characterize the types of size-induced distribution shifts. Diverging from prior approaches, we adopt a spectral perspective and identify that spectrum differences induced by size are related to differences in subgraph patterns (e.g., average cycle lengths). We further find that common GNNs cannot capture these subgraph patterns, resulting in performance decline when testing on larger graphs. Based on these spectral insights, we introduce and compare three model-agnostic strategies aimed at making GNNs aware of important subgraph patterns to enhance their size generalizability: self-supervision, augmentation, and size-insensitive attention. Our empirical results reveal that all strategies enhance GNNs’ size generalizability, with simple size-insensitive attention surprisingly emerging as the most effective method. Notably, this strategy substantially enhances graph classification performance on large test graphs, which are 2-10 times larger than the training graphs, resulting in an improvement in F1 scores by up to 8%.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper investigates how graph neural networks (GNNs) handle graphs of different sizes, particularly focusing on their ability to generalize from smaller to larger graphs. Using biological datasets, the authors adopt a spectral analysis approach to show that differences in subgraph patterns, like cycle lengths, affect a GNN's performance when it encounters larger graphs. They propose and compare three model-agnostic strategies—self-supervision, augmentation, and size-insensitive attention—to enhance GNNs' size generalizability, finding that size-insensitive attention is the most effective method for improving performance on larger graphs.

### Strengths
* The paper starts with the study of types of distribution shifts happening real-world graphs and provides several insights, in particular to cycle importance. 
* The paper proposes and compares 3 different model-agnostic methods to enhance their performance in classification tasks. 
* The experiments on classification indicates that these methods are usually universally good across different models and datasets.

### Weaknesses
* As the paper takes a data-driven approach, the main question is whether these empirical results are transferable to other domains,  other datasets, other models. 
* Augmenting model representations with different statistics is not novel. It's not clear how their enhancements correlate with previous approaches.

### Questions
1. What is the time degradation when performing these augmentation? How much more time needed to perform classification? 
2. The models and datasets are academic. Is it possible to apply this model to more real-world datasets and showcase how this method can be used in biological scenarios?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper tackles distribution shifts caused by the graph sizes of training and test sets.
First, through analysis of the spectrum distribution, it was shown that there is a correlation between the size of the graph and the distribution. It was empirically shown that the degree of correlation changes by adjusting the size of the cycle.
By this empirical evidence, this paper proposes three approaches to make GNNs aware of the existence and number of cycles.
Experimental results demonstrate the potential of graph neural networks (GNNs) to enhance size generalization by understanding their substructure.

### Strengths
1. Spectral analysis of size generalizability of GNNs is novel.
2. The proposed approaches to alleviate distribution shift are effective.

### Weaknesses
1. Lack of mathematical proof of the relationship between spectrum changes depending on the size and number of cycles.
2. GNNs that counts or can aware substructures were not compared.
3. Inappropriate experimental settings.

### Questions
1. The paper focuses on the relationship between cycle size and size generalizability. Could size generalizability be related to the number of cycles?

2. Where is the theoretical evidence that reveals the relation between the size/number of cycles and spectrum distribution?

3. Besides cycles, can there be any substructure that changes the spectrum according to changes in size and number?

4. Where is a comparison with GNNs [1-8] that can understand the structure of the substructure and predict its number relatively accurately or consider out-of-distribution?

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [1] "From stars to subgraphs: Uplifting any GNN with local structure awareness." ICLR 2022.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [2] "Building powerful and equivariant graph neural networks with structural message-passing" NeurIPS 2020.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [3] "Understanding and extending subgraph gnns by rethinking their symmetries", NeurIPS 2022.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [4] "Nested graph neural networks", NeurIPS 2021.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [5] "From local structures to size generalization in graph neural networks", ICML 2021.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [6] "Relational pooling for graph representations", ICML 2019.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [7] "Size-invariant graph representations for graph classification extrapolations", ICML 2021.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [8] "Improving graph neural network expressivity via subgraph isomorphism counting", IEEE TPAMI 2022.

5. The results in Tables 3 and 4 are the results after class imbalance and size imbalance have been corrected. What is the performance in the class imbalance setting of the original data?

6. Is size generalizability using cycle applicable to other data domains beyond the biological domain?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper characterizes the size-induced distribution shifts and evaluated their influence on the generalizability of GNNs through the spectral perspective especially on biological data. It identifies that spectrum differences induced by size are related to differences in subgraph patterns and introduces three model-agnostic strategies to enhance GNNs’ size generalizability.

### Strengths
This paper identifies that cycle-related information plays a pivotal role in reducing spectral differences between small and large graphs. It proposes three model-agnostic strategies—self-supervision, augmentation, and size-insensitive attention—to enhance GNNs’ size generalizability and empirical results demonstrated that their effectiveness.

### Weaknesses
1. Experiments are insufficient and lack of comparison with related methods, such as the size-generalization methods referenced in the related work. The baselines are not state-of-the-art methods in the relevant field. The authors need to add comparison experiments with methods from most recent years, which are related to this paper. Furthermore, the paper lacks experimental validation from other perspectives, such as the effect of different graph size settings in the training process.

2. The contribution lacks novelty. In this paper, the authors identify cycle structures as a major factor affecting the generalization capacities of GNNs. This finding looks to be a special case of [1]. In Section 3, the authors observe that cycle structures have an impact on the spectrum differences between graphs, but it is difficult to ascertain the effect of graph size and cycle distribution on the generalization capacities of models. Also, the three proposed strategies in Section 4 lack novelty and could be combined into a single algorithm.

[1] Gilad Yehudai, Ethan Fetaya, Eli Meirom, Gal Chechik, and Haggai Maron. From local structures to size generalization in graph neural networks. In International Conference on Machine Learning, pages 11975–11986. PMLR, 2021.

### Questions
Why is the algorithm description incomplete in the text, such as the section 4.3? If not essential, it could be excluded as the part of the contributions.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the size generalization of GNNs in biological networks. Through the spectral analysis, the authors find that spectrum differences induced by size are related to differences in subgraph patterns (e.g., average cycle lengths). Since regular GNNs can hardly capture the cycle features, they propose three strategies, including self-supervision, augmentation, and size-insensitive attention, to enable GNNs to learn cycle information thus improving the OOD generalization across different sizes. Experiments with various GNN backbones show the proposed solutions can effectively improve their size OOD generalization ability.

### Strengths
(+) The spectral analysis along with the solutions are well-motivated and interesting to the community;

(+) The paper is well-written and easy to follow;

### Weaknesses
(-) The analysis especially the solutions lacks theoretical guarantees.

(-) The experiments focus on simple tasks and lack the comparison with several relevant baselines.

### Questions
1. The analysis especially the solutions lacks theoretical guarantees. 
- Although the analysis shows that there is a connection between spectrum differences with the cycle lengths, there could be some underlying confounders that jointly affect the graph sizes and cycle lengths. For example, in the model by Bevilacqua et al. 2021, the graphon and the size of the graph will jointly affect the cycle lengths.
- The proposed three solutions are well motivated, while mainly based on empirical observations. To what extent can the three methods resolve the cycle issue? Will the operations affect the expressivity of GNNs?

2. The experiments focus on simple tasks and lack the comparison with several relevant baselines.
- Why do the experiments adopt a different data split scheme from previous practice such as in Bevilacqua et al. 2021?
- How well do the proposed methods perform on more realistic and large datasets such as OGB-molhiv with graph size shifts, and DrugOOD[1]?
- Can the proposed methods perform better than previous solutions like Bevilacqua et al. 2021, and Buffelli et al. that are cited in the paper, and [2,3] that are the state-of-the-art in graph size OOD generalizations?
- [4] analyzes the size generalization in link predictions, which is also a related work to discuss.
- Can the proposed methods improve the size generalization in algorithmic reasoning tasks?


**References**

[1] DrugOOD: Out-of-Distribution (OOD) Dataset Curator and Benchmark for AI-aided Drug Discovery -- A Focus on Affinity Prediction Problems with Noise Annotations, AAAI’23.

[2] Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs, NeurIPS’22.

[3] Wasserstein Barycenter Matching for Graph Size Generalization of Message Passing Neural Networks, ICML’23.

[4] OOD Link Prediction Generalization Capabilities of Message-Passing GNNs in Larger Test Graphs, NeurIPS’22.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
