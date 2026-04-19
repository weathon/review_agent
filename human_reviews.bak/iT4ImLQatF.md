# A Semi-Supervised Clustering Approach For Graph Learning with Neural Networks

- Decision: Reject
- Scores: 5, 5, 3, 6, 3

## Abstract
We propose a semi-supervised approach that combines any unsupervised clustering objective and supervised objective for end-to-end training any neural networks to improve node classification in attributed graphs, particularly when training labels are sparse.
	Our framework formulates node classification as semi-supervised inference of neural network models of attributed graphs with cluster structure.
	We use this framework to understand how neural networks for graph clustering can jointly cluster node attributes and graph structure, despite graph clustering objectives explicitly considering only graph structure and cluster assignments.
	Our framework also enables neural network architectures such as transformers and multilayer perceptrons to learn on graphs without positional encodings and without spectral or message passing layers found in graph neural networks.
	We evaluate our framework on six real-world attributed graph datasets.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces semi-supervised generative models to combine unsupervised clustering and supervised classification in learning on graphs with vertex attributes. 

The authors use Bayesian networks to map out how variables are dependent in current generative model designs. By strategically adding new connections between variables, they suggest improvements to these existing models.

Experimental results on three real-world graph datasets demonstrate that semi-supervised objectives consistently improve performance over purely supervised training.

### Strengths
1. The idea of using Bayesian networks to analyse and enhance generative models is novel in this specific context of clustering and classification on graphs.
2. Multiple generative models such as stochastic block models and graph neural models are unified into a single framework.
3. The paper is clearly structured, with a logical progression from background and motivation to the proposed methodology. The use of Bayesian network diagrams adds clarity, helping readers understand the conditional dependencies in the different generative models.

### Weaknesses
1. The generative models proposed involve learning multiple dependencies among variables. This complexity presents scalability challenges for very large graphs [1, 2]. Including computational complexity analysis and scalability results, e.g. memory used and training time consumed, would add significance to the contributions.
2. The paper primarily evaluates the approach on a limited number of datasets (Cora, Citeseer, Pubmed). A more detailed discussion on generalisability to other graph types (e.g. heterophilic datasets [3]) would strengthen the contributions.
3. The paper lacks a thorough comparison with state-of-the-art methods in graph clustering beyond the few baseline architectures considered. The methods proposed should be positioned with existing methods in the expanding subfield of deep attributed graph clustering [4, 5, 6].
4. The attribute reconstruction might improve learning but this aspect is only briefly evaluated. A more thorough exploration of the benefits of attribute reconstruction, along with ablation studies to demonstrate its impact, would provide greater insight into the conditions under which this component is most effective.

References:
1. Open Graph Benchmark: Datasets for Machine Learning on Graphs, In NeurIPS 2020,
2. Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods, In NeurIPS 2021,
3. A critical look at the evaluation of GNNs under heterophily: Are we really making progress?, In ICLR 2023,
4. An Overview of Advanced Deep Graph Node Clustering, In IEEE Transactions on Computational Social Systems 2024,
5. A Survey of Deep Graph Clustering: Taxonomy, Challenge, Application, and Open Resource, arXiv:2211.12875, 2022,
6. A survey on semi-supervised graph clustering, Eng. Appl. Artif. Intell. 2024.

### Questions
1. Was there a detailed analysis of the computational complexity and scalability of their proposed generative models, particularly for very large graphs?
2. How did the proposed models perform on heterophilic datasets or other types of graphs beyond the citation networks used in the current evaluation?
3. What advantages do the proposed models have compared to recent state-of-the-art approaches in deep attributed graph clustering?
4. How sensitive were the proposed methods to hyperparameter choices, e.g. hidden dimension, learning rate?
5. In equation 14 on line 306, GNN depends only on A. It does not depend on X. What are the node features $X$ used in the GNN? Were they randomly initialised?

### Soundness
2

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
4

### Summary
The paper proposes a semi-supervised framework for node classification on graphs using neural networks, specifically focusing on transformers and MLPs without the need for encoding. It unifies various clustering and classification objectives under a common perspective, showing how graph neural networks implicitly incorporate both node attributes and structure. Through this approach, the paper demonstrates that semi-supervised objectives enable models to perform effectively, outperforming purely supervised training results.

### Strengths
1) The code is shared for reproducibility.
2) Detailed background information is provided for readers who may not be very familiar with the field; however, this may not be ideal for research-focused papers.

### Weaknesses
1) The paper’s structure is poorly organized, and the presentation is quite weak.
2) The three pages dedicated to Background and Related Work are excessive. Generally, this section should be shorter in research papers (unless it’s a review), with most of it moved to an appendix.
3) The methodology section reads too much like related work, which diminishes the paper’s contributions and makes them unclear.
4) The presentation of results is poor. Overall, Table 1 could be split into multiple tables to more clearly show the effect of each component.
5) As it stands, the paper reads more like a review paper than a research-focused paper. With that, it should include more recent baselines from top AI/ML conferences addressing graph clustering with neural networks.
6) The paper has very few evaluation metrics and datasets.

### Questions
With the current state of the paper, I'm more inclining of rejecting, however, I'm open for discussion. If my concerns (weaknesses and questions) are addressed, I can increase my score.

1) Can you clarify how the structure of the paper supports its contributions? Could you reorganize the paper to improve readability and presentation? Some suggestions include:
- Reducing background information, with most parts moved to the appendix.
- Making the methodology section clearer, focusing on and highlighting the main contributions.
- Improving the presentation of results, such as positioning Table 1 better and splitting it into smaller sections for clarity.
- Making equations easier to read (e.g., Lines 270-278).

2) Could you incorporate more baselines, evaluation metrics (such as modularity, graph conductance), and datasets?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper combines the loss of supervised and unsupervised learning on graphs for semi-supervised learning. It first unifies SBM and GNN under the framework of the Bayesian network and re-implement the framework with the neural network. The evaluations are conducted on three small networks.

### Strengths
The combination of SBM and GNN is interesting.

### Weaknesses
- The organization and writing are poor. Most parts are about the background. The organization is confusing.
- The novelty is very limited. The semi-supervised framework is a simple combination of supervised and unsupervised loss. Thus, the contribution is low.
- The unification with the Bayesian network is direct without any novelty. Besides, this unification is not important for the following reimplementation. 
- The technique contribution is limited. The reimplementations of the SBM with neural networks are direct. They just use NN and GNN for latent node embedding and membership matrix. Furthermore, there are many existing works on this strategy such as [1].
- The evaluations are not convincing. Only three very small networks are employed. 

[1] Liang Yang, Fan Wu, Junhua Gu, Chuan Wang, Xiaochun Cao, Di Jin, Yuanfang Guo: Graph Attention Topic Modeling Network. WWW 2020: 144-154

### Questions
Refer to weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper focuses on a semi-supervised method for a traditional graph clustering task. It studies generative models to formulate the semi-supervised graph clustering for attributed graphs with cluster structure. The common framework of generative models analyzed in the paper covers graph neural networks, graph autoencoders, and proposed stochastic block models. The paper conducts experiments on three common attributed graph datasets and shows the superior performance of semi-supervised objectives.

### Strengths
1. The paper proposes a generative model framework for semi-supervised graph clustering. 
2. The proposed method can utilize different neural network architectures including pure transformers and MLPs.
3. The paper gives a new perspective to a unified framework of generative models for node clustering and classification tasks.

### Weaknesses
1. The paper studies a traditional semi-supervised graph clustering task. 
2. Only three small attributed graph datasets are used in the experiments. Larger graph datasets might be also used to show the scalability of the method. 
3. The paper only experiment on one kind of GNNs (i.e., GCN2). More GNN models can be considered in the experiments.

### Questions
1. Could the idea of generative models be used for other graph machine learning tasks?
2. What is the time complexity of the model?
3. Do the authors try recent GNN models and datasets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
## Summary 

The paper proposes a combination of unsupervised clustering objectives with supervised node classification tasks via generative models to improve the performance of node classification in sparse labeled settings. The key insight from the paper is that clustering objectives can incorporate node attributes for a more holistic clustering and the authors achieve this empirically by proposing a positional-encoding free alternative using transformers and MLPs. Experimental results are provided on three real world graphs to show that semi-supervised clustering objectives outperform purely supervised approaches in sparse label regime.

### Strengths
## Strengths 

1. Some of the insights into the nature of unsupervised clustering w.r.t the labels of the nodes are useful for the graph learning community for building better methods for these tasks. 

2. Viewing node clustering and classification approaches into a single view of generative models is interesting.

### Weaknesses
## Weaknesses 

1. The motivation of the paper and the setup of the authors’ contribution is not standard where they look at the problem of semi-supervised classification using clustering objectives by incorporating the role of both node features and adjacency matrix via generative models. The assumption that node attributes and/or a combination of nodes attributes and graph adjacency matrices haven’t been explored before (L066, L411-413) is also incorrect, since there are several lines of work in the graph learning literature that precisely solve this problem - MVGRL [1], BGRL [2], S3GC[3], GRACE [4] to name a few. Even the features learnt from Deep Graph Infomax [5] style learning methods have been shown to be fairly effective for both clustering as well as the classification task. While I understand that the paper attempts to provide an alternative to graph convolutions, the merit for this setup and evaluation would have been justified if the authors compared their method to several of these other works and their objectives, demonstrating that these methods perform worse for classification, however the mention of these papers or these comparisons are missing from both the motivation and the experiments. 

2. The datasets used for experimentation - Cora, Citeseer and Pubmed are of extremely small scale with a maximum number of nodes as 20,000, making the experimental evaluation extremely weak. It has been shown in several works in the graph learning literature [6] that such small datasets are unreliable for measurements of any reasonable accuracy improvements. When there are several reliable benchmarks datasets available in the graph learning community, such as OGB [7], I would have expected the authors to use these datasets for a reliable and convincing evaluation.  

3. In the current form, the paper is hard to read and there are several issues with the presentation and organization of different sections of the paper which need re-writing for clarity and coherence.

-  a. The introduction section does not adequately set up the motivation for incorporating a clustering objective into the semi-supervised classification setup, and why the comparison is only with fully supervised methods. This needs to be supported with more citations, prior work, and concrete hypotheses which will be verified through a solid evaluation. 

- b. Section 3.1 from L261 - L300 is confusing and does not clearly enumerate the proposed methodology in the paper. 

- c. Section 4 does not include any details of the experimental setup, the datasets, the metric used for measurements, the baselines used for comparison, or even a crisp conclusion that is drawn from the experimental results highlighting the appropriate results. While it is understandable that the detailed description about these can be deferred to the Appendix, it is expected that this section would have at least enough details to be able to interpret the numbers and experimental results in Table 1, which the paper fails to do so in the current format. The main table of results is on the last page of the paper, after the conclusions and future work section, which needs some re-organization. 

- d. The more experimental results in Appendix D merely enumerate the results in Tables from page 24 - 59, without any conclusion or interpretation of the results. The authors should focus on insights from their experimentation that will be valuable to the graph learning community, and detail them in the Appendix, if not in the main paper.  

[1] Contrastive multi-view representation learning on graphs, Hassani et. al, ICML 2020

[2] Large-scale representation learning on graphs via bootstrapping, Thakoor et. al, ICLR 2021

[3] S3GC: Scalable Self Supervised Graph Clustering, Fnu Devvrit et. al, NeurIPS 2022


[4] Deep graph contrastive representation learning, Zhu et. al, arxiv:2006.04131 2020


[5] Deep graph Infomax, Velickovic et. al, ICLR 2019

[6] Pitfalls of graph neural network evaluation, Shchur et. al, R2L NeurIPS 2018

[7] Open graph Benchmark: Datasets for Machine Learning on Graphs, Hu et. al, 2021

### Questions
## Questions 

1. The conclusion from Figure 1 is unclear to me. From the supervised clustering figure (which is the proposed method), the nodes in the cluster seem to be more heterogeneous and mixed as compared to the ground truth labels, which brings into question the effectiveness of the method and the practicality of the MCC metric that is measured.

### Soundness
2

### Presentation
1

### Contribution
2
