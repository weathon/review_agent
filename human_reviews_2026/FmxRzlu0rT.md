# Learning Posterior Predictive Distributions for Node Classification from Synthetic Graph Priors

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
One of the most challenging problems in graph machine learning is generalizing across graphs with diverse properties. Graph neural networks (GNNs) face a fundamental limitation: they require separate training for each new graph, preventing universal generalization across diverse graph datasets. A critical challenge facing GNNs lies in their reliance on labeled training data for each individual graph, a requirement that hinders the capacity for universal node classification due to the heterogeneity inherent in graphs --- differences in homophily levels, community structures, and feature distributions across datasets. Inspired by the success of large language models (LLMs) that achieve in-context learning through massive-scale pre-training on diverse datasets, we introduce NodePFN. This universal node classification method generalizes to arbitrary graphs without graph-specific training. NodePFN learns posterior predictive distributions (PPDs) by training only on thousands of synthetic graphs generated from carefully designed priors. Our synthetic graph generation covers real-world graphs through the use of random networks with controllable homophily levels and structural causal models for complex feature-label relationships. We develop a dual-branch architecture combining context-query attention mechanisms with local message passing to enable graph-aware in-context learning. Extensive evaluation on 23 benchmarks demonstrates that a single pre-trained NodePFN achieves 71.27\% average accuracy. These results validate that universal graph learning patterns can be effectively learned from synthetic priors, establishing a new paradigm for generalization in node classification.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes NodePFN, which extends PFNs to graphs. It learns posterior predictive distributions for node classification by pretraining only on synthetic graphs drawn from priors (ER and contextual SBM with controlled homophily; SCM-generated feature–label mechanisms). A dual-branch architecture mixes context–query attention (for in-context learning from labeled nodes) with local message passing (GCN). A single pretrained model (no per-graph training) shows strong performance across 23 benchmarks, with both homophily and heterophily graphs.

### Strengths
1. It proposes a novel PFN-style, training-free inference framework. The model can learn synthetic priors and predict in one forward pass.
2. The model trained on synthetic graphs shows strong performance on both homophily and heterophily benchmarks. 
3. The analysis of performance on different datasets and the ablation study are comprehensive.

### Weaknesses
1. Priors are limited to ER and cSBM with SCM-driven features. Real graphs often exhibit heavy-tailed degrees, motifs, assortativity/disassortativity. Whether these properties might affect the model performance is not discussed.
2. Real-world SOTA GNNs tailored to heterophily, label-efficient settings, or inductive protocols aren’t comprehensively covered.
3. Pretraining requires ~250k synthetic graphs, but there’s no wall-clock/peak-memory vs. baselines or cost-vs-benefit analysis; only a conceptual argument that training amortizes across tasks.

### Questions
1. How does performance change when real graphs exhibit power-law degrees, overlapping/temporal communities, or motif biases besides ER/cSBM?
2. How does the model compare to the SOTA model for heterophily graphs? What could be the possible reason if the model performs worse?
3. Could the authors provide more analysis on how the synthetic training graphs are generated? For example, will the sampling distribution affect the model performance? And how does ER graph (when varying its percentage in the training set) affect the model performance? And what is the size distribution of the training graphs?
4. Overall, I think the paper proposes a possibly promising path to solve the cross-dataset node prediction task. **I will raise the score if the above concerns are settled.**

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
NodePFN pretrains on procedurally generated graphs—features/labels from a random SCM and structure from cSBM/ER—to learn the posterior predictive distribution for node labels. At test time it takes any real graph plus a few labeled nodes and, in a single forward pass (no fine-tuning), outputs calibrated label distributions for the rest, achieving competitive accuracy across 23 benchmarks and varying homophily.

### Strengths
1. Studying prior-fitting paradigm on relational data is valuable and interesting
2. Experimental results are good

### Weaknesses
Major:
1. One major concern is that the method part writing is unclear. For example, 1. how is the feature and structure treated? Is it like that you first generate the random features and labels, then labels are used as community, and then use community to generate structures (for SBM). 2. How are labels used in the whole pipeline? I even don't see any label encoder module in the paper. 
2. The structure of the model is very different from the original PFN. Transformer is put on top of the MPNN. Can you give some theoretical insights on why this is valid? For this part, I think some theoretical discussions are missing since the theory for PFN no longer holds here. 
3. Some baselines are missing. I'm wondering how this method compares to first use some non-parametric aggregation to get aggregated features and then utilize TabPFN for prediction. 
4. I doubt the scalability of this architecture since you put heavy self-attention ahead of message passing. Also in the experiment part, there's no large-scale datasets. 

Minor:
1. The writing of abstract is not clear. Why training on labels a fundamental limitation?
2. I don't think this work is inspired from LLM. Philosophy of prior fitting is very different from LLM, and it's also not possible to have a unified vocabulary akin to LLM. I suggest motivating from the prior fitting perspective. 
3. Line 44: What is graph heterogeneity? Graph and tasks are different concepts. Tasks determine labels. 
4. Line 74: Graphany does have designs for heterogeneity. In the aggregation stage, it has filter like (I-A).

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces NodePFN, extending the Prior-Fitted Network (PFN) paradigm to graph neural networks. The key innovation is training a single model on thousands of synthetic graphs generated from carefully designed priors (random networks with controllable homophily and structural causal models) to learn posterior predictive distributions for node classification. This enables the model to perform inference on arbitrary real-world graphs without task-specific training, achieving 71.27% average accuracy across 23 benchmarks.

### Strengths
- The extension of PFNs to graphs represents a genuinely new direction. The idea of learning universal node classification patterns from synthetic priors rather than requiring graph-specific training is novel and addresses a fundamental limitation of current GNNs.
- The paper provides theoretical analysis (Theorems in Section 2.3) showing how graph structure affects node features' impact on GNN performance, motivating why synthetic priors with controlled properties can capture real-world patterns.
- Testing on 23 diverse benchmarks, including both homophilic and heterophilic graph,s demonstrates wide applicability. The 71.27% average accuracy with a single model is impressive, particularly the 65.14% on heterophilic graphs where traditional GNNs struggle.
- The combination of contextual SBMs with controllable homophily (0.1-0.9) and structural causal models for feature-label relationships shows thoughtful prior design that systematically covers graph diversity.
- The "train once, deploy everywhere" paradigm offers practical benefits, removing the need for dataset-specific training while maintaining competitive performance.

### Weaknesses
- The quadratic complexity of attention mechanisms and fixed constraints on class numbers (max 20) and feature dimensions significantly limit applicability to real-world large-scale graphs. This is acknowledged but not addressed.
- While the paper describes the priors used, it lacks ablation studies on prior design choices. Why these specific distributions? How sensitive is performance to prior specification? The connection between prior properties and downstream performance needs deeper investigation.
- The paper mentions ~250,000 synthetic graphs and 6 GPU hours of training, but doesn't provide detailed computational comparisons with training multiple task-specific GNNs. The amortization argument needs quantitative support.
- The paper primarily compares against basic GNNs (GCN, GAT) and GraphAny. Comparisons with more recent heterophily-specific methods (e.g., H2GCN, GPRGNN) and other universal graph methods would strengthen the evaluation.
- What determines when NodePFN will succeed or fail? The paper doesn't clearly characterize the conditions under which synthetic priors successfully capture real-world patterns or provide failure case analysis.
- The dual-branch architecture combining attention and message passing is described superficially. Ablations on architectural choices and their contributions are limited (only Table 3 provides some ablations).

### Questions
1. How does performance degrade as graphs exceed the training distribution in terms of size, feature dimensions, or class numbers? Can you provide scaling experiments?
2. Can you provide more extensive ablations on the synthetic prior design? How do different homophily distributions, SCM architectures, or graph generation models affect performance?
3. Why does NodePFN sometimes underperform GraphAny models trained on specific datasets (e.g., Amazon-Comp: 81.42% vs 83.04%)? What patterns are difficult to capture with synthetic priors?
4. Could you compare against more recent heterophily-specific GNN methods and explain when NodePFN's universal approach is preferable to specialized architectures?
5. Is there a way to adapt or fine-tune NodePFN for specific graphs while maintaining most of the universal knowledge? Maybe this could address performance gaps on specific datasets.

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
4

### Summary
The authors propose NodePFN, a foundation model for node classification on arbitrary graphs. Following recent results on foundation models for tabular data, whose main insight is training models on synthetically generated data that resembles relationships present in real-world data, the authors propose to train a GNN on synthetically generated graphs. In order to accommodate for the variable number of node features and labels across different datasets, as well as to achieve in-context learning for new graphs, the authors propose an architecture based on self-attention. Experiments indicate that NodePFN effectively generalizes to real world datasets, surpasing the performance of supervised methods that require data-specific training.

### Strengths
1. The paper addresses an important limitation of supervised methods that require training from scratch, or at best fine-tuning for every new dataset, thus reducing the cost of deploying predictive models on graphs.
2. The paper introduces well-motivated synthetic data generation methods that cover diverse network topologies, and an architecture for learning in context intended to be pre-trained with synthetic data.
3. The experiments are comprehensive, demonstrating the effectiveness of NodePFN agains a broad range of baselines and in ablation experiments.

### Weaknesses
1. The novelty of the work is slightly limited, as it is a natural extension of well-known results from TabPFN and its follow-up works, to the graph domain.
2. Important details about the generated data are missing. How many graphs in total were generated? What is the number of nodes and edges in them?
3. NodePFN looks like a costly model for training and inference. Some indication of training cost is given in the appendix ("6 GPU hours"), but a formal derivation of complexity is missing from the paper. This is especially important considering that the method requires a new architecture that differs from established ones like the GCN.
4. Important details about the architecture are not clear. Can NodePFN handle node features, especially a variable number? Section 6 states that this is a limitation, requiring a maximum number of dimensions, but it is not clear how this is dealt with. Does it involve some form of padding along the feature dimension?
5. A sometimes implicit yet strong motivation for works such as TabPFN, and in this case NodePFN, is that generating a collection of datasets containing complex relations between variables that is large and random enough allows pretraining models that generalize to real data. The question here is: how do we know what is large and random enough to guarantee generalization? Is NodePFN good because our knowledge about the datasets in Table 1 allows us to generate data like them? And if so, how robust is it to distribution shift?
6. The architecture can in principle be fine-tuned but these experiments are not considered in the paper. It would be interesting to know how much of an improvement this can bring.

### Questions
1. Can you please provide further information about the statistics of the generated datasets?
2. The appendix states that NodePFN is trained with one synthetic graph per epoch, and that a batch size of 8 is used. What does a "batch" mean in this context? In the simplest setting, training a GNN requires using the full graph (i.e. the complete feature and adjacency matrices) in order to do an unbiased forward pass. 
3. Can you please clarify how NodePFN handles input features whose number varies across datasets?
4. Do you have any thoughts on the generalization properties of NodePFN and its robustness to distribution shift? It would be interesting to know how confident you are that it will work well for every graph in the wild.

### Soundness
3

### Presentation
4

### Contribution
3
