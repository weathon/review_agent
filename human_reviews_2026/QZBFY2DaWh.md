# MoleRanker: Spectrum-Driven Molecular Structure Ranking with Heterogeneous Co-occurrence Graphs

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Identifying molecular structures in environmental and biological samples is essential for assessing ecological risks and human health, yet remains highly challenging due to the vast number of unidentified compounds. Tandem mass spectrometry (MS/MS) provides high-throughput spectrum measurements, but existing spectrum-driven identification approaches face key limitations: spectrum-isolated modeling methods are computationally expensive and tend to overlook molecular clustering effects. Moreover, network-based methods typically fail to incorporate environmental co-occurrence across chemical samples, yielding unsatisfactory performance. To address these challenges, we revisit molecular identification as spectrum-driven molecular structure ranking and propose \textsc{MoleRanker}, a novel heterogeneous graph neural network that integrates chemical constraints with environmental co-occurrence patterns. Specifically, we first construct a heterogeneous co-occurrence graph that encodes both \textit{molecular-level chemical clustering effects} and \textit{sample-level environmental co-occurrence correlations}. We then design a multiplex-relation message-passing mechanism to perform information propagation in a relation-aware manner across these heterogeneous relations. We construct four diverse datasets, including in-situ environmental pollutants and human metabolomics, and release them as a benchmark for spectrum-driven molecular structure ranking. Extensive experiments demonstrate that \textsc{MoleRanker} achieves state-of-the-art performance, improving mean reciprocal rank (MRR) by 12.18\% on average. Beyond accuracy, our approach opens new opportunities for discovering emerging pollutants and advancing the molecular understanding of human metabolism through graph-based integration of chemical and environmental evidence. Code is available at \url{https://anonymous.4open.science/r/MoleRanker}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper redefines molecular identification as spectrum-driven molecular structure ranking, and proposes a heterogeneous co-occurrence graph algorithm that combines molecule-level chemical concentration information and sample-level environmental information to score and rank candidate molecular structures. The authors conduct experiments on both self-collected and public datasets to demonstrate the contribution of the algorithm.

### Strengths
I believe the paper is clearly written and conveys the authors’ ideas well. The core idea is interesting and intuitive, the reported metrics exhibit improvements, and the ablation studies substantiate the effectiveness of both information sources.

### Weaknesses
1. The proposed “heterogeneous co-occurrence graph” approach does not appear to be novel; the contribution largely consists of injecting two additional information sources, while the remaining components follow standard practice. This amounts to an engineering blend rather than a deeper investigation of the molecular structure ranking problem.
2. The supplementary material lists the compared methods but omits detailed experimental configurations. I strongly recommend fully specifying all baseline setups. For instance, were the homogeneous-graph baselines (GCN, GAT, GraphSAGE) trained on the same two relation layers/graphs?
3. The training and evaluation pipeline is under-specified; please provide a detailed description.
4. Although the paper claims lower computational cost than existing methods, no supporting experiments are reported; please include computational efficiency results.

### Questions
1. The current comparative experiments are fairly conventional. Was a comprehensive investigation of strong baselines conducted?
2. The reported performance of MetFrag, CFM-ID, and SIRIUS is near zero on three datasets. Is this consistent with prior empirical evidence? Please provide a detailed explanation for these results.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents MOLERANKER, a heterogeneous graph neural network specifically designed for spectrum-based molecular structure ranking. The core idea is to integrate chemical constraints (spectral similarity) and environmental co-occurrence (concentration correlations between samples) into multiple heterogeneous graphs, thereby enhancing message passing and improving the accuracy of candidate molecule ranking. The proposed architecture comprises spectrum-based and SMILES-based encoders, relation-aware graph convolution, and a dual-tower scoring mechanism optimized via a pairwise Bayesian Personalized Ranking (BPR) loss function. Experimental results on four datasets—including a newly curated environmental pollutant dataset and three human metabolomics datasets—demonstrate that the proposed method significantly outperforms existing approaches.

### Strengths
1. The integration of a dual-tower scoring mechanism with a pairwise Bayesian Personalized Ranking (BPR) objective exhibits practical value for large candidate sets and severe class imbalance problems.
2. The experiments conducted in this study are relatively comprehensive, and the results are favorable.
3. The visualization outcomes are presented with commendable clarity.

### Weaknesses
1. Omission of certain recent baselines in evaluation: The selection of baselines overlooks several directly relevant recent methods, particularly those leveraging topological or spectral graph modeling, or explicitly employing multi-relational molecular graphs for ranking tasks beyond the GNPS ecosystem.
2. Insufficient discussion of practical limitations and generalization ability: Although the paper demonstrates strong performance on the relevant datasets, the discussion does not fully address potential generalization constraints—such as distributional shift in environmental chemistry or biomedical metabolomics—nor does it quantify the effects of batch variability, sample bias, or differences in candidate quality.
3. Lack of, or insufficient, justification for not adopting other established GNN variants or loss functions: While the study implements standard GCN, GAT, and GraphSAGE as baselines, it does not utilize other advanced heterogeneous/relational GNN variants that have proven effective for multi-graph or biochemical graph tasks. The rationale for including only the proposed baselines is not sufficiently substantiated.
4. Limited discussion on computational efficiency and scalability: Although the manuscript claims that MOLERANKER surpasses spectral isolation methods in computational efficiency, it provides no runtime or complexity analysis. Given the scale of candidate datasets shown in Figure 5 and Table 4, a clearer comparison is needed to articulate the trade-offs between actual runtime and scalability.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes a new method of identifying molecules from mass spectra combined with their co-occurrences. The work combines information from multiple resources, including correlation information computed from multiple samples, encoding of mass spectra, similarity between mass spectra, and molecule structures of candidates. The proposed method designs a graph neural network to learn information from all these resources. By minimizing the loss derived from a few mass spectra with known molecules, the model learns to identify molecule structures of other mass spectra. The performance shows that the proposed model achieves good performance in the transductive setting (identifying molecules in the constructed graph on all mass spectra).

### Strengths
On the data with multiple samples, it is a novel approach to include the correlation information from the concentration of molecules to improve the model's performance.

The proposed method outperforms several baselines, though new baselines from recent years should be included.

### Weaknesses
1. The setup of the problem is not very realistic. The research seems to be a rediscovery of the ground truth. In a real situation, we can obtain a collection of mass spectra and construct a graph over them. However, it is hard to know the molecular structures of some nodes for training a neural network. It is possible that we could get the structures of a few of them because they are easy to identify, or we can use some external information. Even in this case, the known molecules will not represent the distribution of all molecules in the graph. The training model might be biased. Without the ability to generalize to different networks, it is hard to put the model to use in real situations. 

2. The model design with graph neural networks is largely known to the community. Therefore, the contribution to the machine learning aspect is limited. 

3. In the experiment, the three algorithms specially designed for molecule identification, MetFrag, CMF-ID, SIRIUS, are traditional methods without advanced neural networks. Other baselines are all generic learning models. There are a series of improvements [1] over these three methods, and these new methods should be included in the comparison. 


[1] Liu, Youzhong, et al. "Current and future deep learning algorithms for tandem mass spectrometry (MS/MS)‐based small molecule structure elucidation." Rapid Communications in Mass Spectrometry 39 (2025): e9120.

### Questions
I have no questions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors address the problem molecular structure ranking from tandem mass spectrometry data. They present MoleRanker, a novel graph neural network-based method which uses molecular groupings (as graphs) of shared functional groups (environmental co-occurrence) and chemical constraints. The authors additionally construct a new dataset to evaluate their proposed approach. Through a set of empirical experiments across several datasets, comparing with various baselines, and numerous ablations, the authors demonstrate the effectiveness of their proposed method for spectrum-driven molecular structure ranking.

### Strengths
- The paper is clearly written and well presented (barring some minor comments below). As a result, the claims and contributions of this work are easy to follow.
- The authors construct a new tandem mass spectrometry dataset to evaluate and benchmark molecular identification as a spectrum-driven molecular structure ranking task.
- The authors introduce a novel method for molecular identification and validate it through comprehensive empirical experiments. The newly proposed method offers a novel insight into how to approach the problem of molecular identification, which could be useful to the research community.

### Weaknesses
- Some details still require further clarification and possibly some additional experiments. Please see questions below.

### Questions
- In equation 6, in $f$, can you clarify if the molecules in the candidate  set $\mathcal{C}_i$ correspond to the nodes in $\mathcal{G}$, or if $\mathcal{G}$ is a graph over candidate sets? Furthermore, In the problem definition, specifically in equation 6, what is the ranking over? Molecules in one candidate set $\mathcal{C}_i$? Or over all candidate sets $\mathcal{C}$?
- From section 3.2, is my understanding correct that you manually curated a novel dataset of molecules analyzed via tandem mass spectrometry?
- Since the co-occurrence graph is constructed using prior knowledge, have you ablated for the cases where the molecular co-occurrence graph has errors? For instance, what happens if some of the edges in the co-occurrence graph are removed or additional edges are added? 
- In a similar vein, for the ablation in Table 2, what is the architecture used in the w/o graph setting?
- The method uses SMILES representations of molecules, which are encoded into an embedding space. Have the authors considered other molecular representations (such as molecular graphs)? How would performance of the method change when using different molecular representations?


Minor comments: 

- "environmental co-occurrence" seems pertinent to this work and is mentioned several times in the abstract and introduction before a clear definition is provided. Possibly defining this term earlier would be beneficial to the reader. 
- In section 5, it would be helpful to state a summary of the research questions in the title of each respective sub-section.

### Soundness
3

### Presentation
3

### Contribution
3
