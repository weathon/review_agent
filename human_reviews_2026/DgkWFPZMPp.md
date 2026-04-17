# Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 4

## Abstract
Effectively capturing long-range interactions remains a fundamental yet unresolved challenge in graph neural network (GNN) research, critical for applications across diverse fields of science. To systematically address this, we introduce ECHO (Evaluating Communication over long HOps), a novel benchmark specifically designed to rigorously assess the capabilities of GNNs in handling very long-range graph
propagation. ECHO includes three synthetic graph tasks, namely single-source shortest paths, node eccentricity, and graph diameter, each constructed over diverse and structurally challenging topologies intentionally designed to introduce significant information bottlenecks. ECHO also includes two real-world datasets, ECHO-Charge and ECHO-Energy, which define chemically grounded benchmarks for predicting atomic partial charges and molecular total energies, respectively, with reference computations obtained at the density functional theory (DFT) level. Both tasks inherently depend on capturing complex long-range molecular interactions. Our extensive benchmarking of popular GNN architectures reveals clear performance gaps, emphasizing the difficulty of true long-range propagation and highlighting design choices capable of overcoming inherent limitations. ECHO thereby sets a new standard for evaluating long-range information propagation, also providing a compelling example for its need in AI for science.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a new benchmark for long-range graph propagation in graph learning, motivated by the fact that existing benchmarks (as the LRGB) have recently been questioned to actually capture long-range ability and somewhat plateaued. The authors introduce ECHO, consisting of a synthetic and real-world part. The synethetic tasks consists of single-source shortest path, node eccentricity, and graph diameter, and are to be predicted on graphs sampled from 6 different (sparse and high-diameter) graph topologies. The real-world tasks on molecules are predicting atomic charges (ECHO-Charge) and total energy (ECHO-Energy). The new benchmark is evaluated on a wide range of common graph learning methods, including standard MPNNs and graph transformers. Results indicate that models with global information exchange are performing better on ECHO.

### Strengths
- I personally think this is an important and timely contribution. I think that the graph ML community urgently needs new and well-designed long-range propagation benchmarks, as performance on the LRGB has plataued and been questioned to really probe long range propagation. Yet, most current methodological works still use the LRGB, especially the peptides tasks, often as the sole empirical basis for any claims regarding long-range ability.
- The benchmark design is quite good (synthetic and real-world), and it is clear that significant thought and computational resources went into the creation and evaluation on the baselines. For example, the benchmark is large-scale (200k graphs for the ECHO-Charge/Energy), and in the synthetic case, the graph topologies are properly stratified in train/val/test.
- The experimental rigor is high. Hyperparameters were extensively tuned on the benchmark (app. F), which is crucial given how untuned baselines have led to incorrect conclusions in the past.

### Weaknesses
- The graphs in all tasks are quite small (order of roughly 50-200 nodes). The paper could be more upfront about the benchmark not probing scalability, even though many current methodological works target methods that both capture long-range interactions and scale well. It might also help to explicitly position the work against newer long-range benchmarks that explicitly test scalability. [1]
- Not really a weakness but a suggestion: For the synthetic tasks, readers might initially think that the targets are simple monotonic functions of graph size within each topology, reducing the problem to a topology classification. I checked the data and this does not seem to be the case, but additionally grouping plots like Fig. 6(a) by topology could better convey task difficulty and make the reported MAE values more interpretable.

[1] Huidong Liang, Haitz Sáez de Ocáriz Borde, Baskaran Sripathmanathan, Michael Bronstein, Xiaowen Dong. *Towards Quantifying Long-Range Interactions in Graph Machine Learning: a Large Graph Dataset and a Measurement.* https://arxiv.org/pdf/2503.09008

### Questions
- Could you please refer to the points from “Weaknesses”?
- Do you have any intuition for the U-shaped error in fig. 8(b), and, to some extent, (a)?
- Superficially, ECHO-Synth resembles the graph property prediction (GPP) dataset from [2] (same tasks but different graph topologies).  Since [2] is cited, it would help to clarify the relationship, i.e., if/how ECHO-Synth builds on or departs from GPP, more explicitly (perhaps in sec. 3.1).

[2] Alessio Gravina, Davide Bacciu, and Claudio Gallicchio. Anti-Symmetric DGN: a stable architecture for Deep Graph Networks. ICLR 2023.

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
3

### Summary
The paper introduces a suite of new inductive benchmark datasets specifically designed to evaluate long-range propagation capabilities in graph neural networks (GNs). The benchmark collection includes five tasks, of which three are synthetic and two are real world. The synthetic tasks consist of predicting property predictions on different topologies, two of which are node-level and one is graph-level. The real-world tasks consist of chemically grounded tasks, one being a node-level regression task where the goal is to predict atom charges, and the other is graph-level where the goal is to predict total energy. The paper argues that these datasets are indeed long-range. The benchmark is intended to fill the gap left by prior datasets (e.g. LRGB) as they received criticism recently for poor hyperparameter tuning and their long-range property having been questioned.

### Strengths
1. The paper is very well written and very clear.
2. The paper addresses an open problem consisting of finding a good long range benchmark for GNNs, which is particularly relevant given the recent criticisms of LRGB (hyperparameter tuning, and long-range nature of the task).
3. The hyperparameter tuning is more rigorous than previous work, directly addressing a criticism of previous benchmarks.
4. The ECHO-Charge and ECHO-Synth is original and may foster more work applying GNNs to chemistry applications.
5. The paper verifies the long-range nature of the ECHO-synth benchmarks by showing that the performance improves with number of layer.

### Weaknesses
1. **Long range evidence for real world tasks is lacking** The evidence for the long-range nature of the real world benchmark (ECHO-Charge and ECHO-Energy) is limited. Indeed there seems to be no correlation between depth and performance (Fig 7d). The only argument is that long-range architectures (such as GPS and SWAN) outperform standard supposedly not long range architectures. This argument is very similar to the LRGB arguments which were shown to be limited. On the other hand, the hyperparameter tuning in this work is more thorough.
2. **Hyperparameter tuning can be improved** The main criticism of LRGB in Tönshoff et al is that increasing the MLP depth post-readout improves performance on the proposed tasks. However, in this work the post-readout depth is set to 2 without tuning. In light of this, more thorough look at this hyperparameter would be interesting to the community and would further improve the hyperparameter tuning.
3. **[Minor] better evaluation of different topologies** In the synthetic datasets, the motivation for all different topologies are for example to include bottlenecks, however performance per graph topology was not evaluated.

### Questions
1. GPS is an architecture that combines several components: positional encoding, MPNN layers, and fully connected attention. Could you please add more information on what components were used and tuned (especially what MPNN architecture and number layers)?
2. In the synthetic datasets it says that the node features are random uniform. Wy not set the node features to zero?
3. Could you do an ablation on the post-pooling MLP depth for the graph-level real world task?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
- The paper argues the need for newer and better-motivated long-range benchmarks than those currently in use by the community
- The paper introduces a new benchmark, ECHO, for long-range interactions in GNNs. The benchmark includes:
	- Echo-Synth: three synthetic tasks (two node level and one graph level), requiring the computation of shortest-path-based properties
		- Long-range dependency is assured as these properties require transversing the entire graph, and all synthetic graphs have diameters of at least 17 hops
	- Two real-world tasks based on atomic partial charges; Echo-Charge, a node-level regression of the partial charge of each atom in a molecular graph, and Echo-Energy, graph-level regression of total molecular energy
		- Long-range dependency is again ensured by graphs with minimum diameters of 17 hops, and argued based on the nature of the underlying task
- The paper includes a comprehensive suite of experiments on a fairly comprehensive range of architectures, including classical MPNNs, a multi-hop MPNN, differential-equation-inspired GNNs and a graph Transformer.

### Strengths
- The need for new long-range benchmarks and the shortcomings of existing ones are well-argued.
- The tasks, both synthetic and real-world, are at least as well motivated as existing benchmarks; I can see ECHO becoming a new and sorely-needed standard benchmark for the community.
- The paper is very well written, and Figures 1,2 and 3 are used to great effect to illustrate the tasks introduced; I think this will be a great boon to adoption.
- The authors evidently put a lot of time and effort into this paper; in addition to the experiments in the main text, the comprehensive additional experiments in the Appendix, and apparent 2 months of compute time required to produce the -Charge and -Energy datasets are evidence of this.
- I did not run the code, but it appears thorough, looks easy to run and is well-documented.
- I did not closely read the Appendix, but the inclusion of runtime measurements is welcome, as is the illustration of effective hub nodes in GPS, and the large depth and large diameter experiments in Appendix G

### Weaknesses
- [W1] The authors argue that the -Charge and -Energy tasks  are 'inherently long-range' due to (i) their large size/diameter, 17-40 hops, and (ii) the underlying task; they say: 
	- "The three-dimensional configuration of molecules greatly intensifies this task complexity, as distant atoms in the graph topology can still exert significant influence on electronic properties and total energy."
	- [W1.1] This makes sense to me, but do you have a source for it?
	- To put it another way; large diameter/graph size and intuition about the underlying task are strong indicators that a task is long-range, but do not prove it necessarily. Your experiments, in which deeper models/Transformers dominate, would also indicate that long-range interactions are present but this is somewhat circular (i.e., if the dataset is an evaluator of model LR capability, its validity as an LR dataset  cannot be based on model performance).
	- [W1.2] With this in mind, I would like to see a more theoretically motivated justification for the long-rangedness of the real-world datasets. (Echo-Energy is a graph-level task, but this is not necessarily inherently long-range; it could well be simply a sum over local information applied at readout.) For example, you cite Bamberger et al. (2025), who propose a range measure, as evidence for the questionable long-rangedness of LRGB. Why not apply this measure to your benchmark?
- [W2] I think experiments would be improved by the inclusion of more than one graph Transformer, perhaps something more recent than GPS — e.g. GRIT [1], Graph-Mixer [2]

---

[1] Ma, Liheng, et al. "Graph inductive biases in transformers without message passing." International Conference on Machine Learning. PMLR, 2023.

[2] He, Xiaoxin, et al. "A generalization of vit/mlp-mixer to graphs." International conference on machine learning. PMLR, 2023.

### Questions
- [Q1] Line 284: What is 'total molecular energy'? Is it simply the sum of individual node energies?
- [Q2] Line 1109: Your hyperparameter search for Drew only goes up to 4 layers, whereas it goes up to 40 for other models. As I understand it, Drew is an MPNN, with a receptive field determined by the number of layers. Does this not affect its LR capability? Why such a small search window?
- [Q3] On the task of predicting atomic partial charges — 
	- Is this a problem for which GNNs have already shown promise? Do you have a source for this? It was not obvious to me from the text
	- Can you contextualise the experimental results a little? I.e. do any of the models do a good enough job to reasonably replace/approximate expensive DFT computation? 
	- To put it another way, it isn't clear to me whether GNNs are actually reasonable for solving this problem, or whether this is just an instructive task for GNN benchmarking

Misc:
- Line 208: Is this an appropriate use of 'skip connection'? I would tend to associate it with residual connections
- Line 454: I would suggest adding a sentence here about the contents of the experiments in Appendix G, as I think they are quite interesting
- Typos:
	- Line 104: 'on the need ~~of~~ **for** a new benchmark'
	- Line 380: 'differently, graphcon do not'

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a benchmark for evaluating long range information propagation in graph learning networks. The benchmark consists of 5 tasks- 3 synthetic graph-theoretic problems (single-source shortest paths, node eccentricity, graph diameter), and 2 real-world datasets focused on molecular graphs (charge and energy) for predicting atomic partial charges and molecular energies. The task design intends to stress on long range propagation based, for example, on the hops these datasets cover. Experiments are done using multiple GNNs, Graph Transformers and differential equations inspired models and insights are presented for models performing good and bad.

### Strengths
- There is a gap of datasets on long rage testing in GNNs and although there exists the benchmarks that this paper discusses they have limitations. So the proposed datasets address this.
- The datasets include both synthetic and real world graphs with the real ones standing a good contribution due to the curation as well as the performance trends.
- Empirical results show that models arguably good for long range propagation such as GPS, A-DGN, SWAN, DRew, etc outperform MPNNs, with particularly strong evidence that depth matters.

### Weaknesses
- The primary contribution of this work is the collection of molecular dataset as the similar synthetic dataset is known in the prior literature.
- The strong claim in lines 146-147 about the molecular dataset on long range could be disputed since prior molecular dataset also involve long range affects although they could be synthetic targets.
- The paper lacks close discussion/comparison/adaptation of known insights with 2 works which would make it stronger, particularly on quantifying ECHO datasets' long rangeness. First with [1] which has a strinkingly similar multi-task synthetic dataset. Second, with [2] which quantifies long rangeness of a graph dataset.

[1] Corso, Gabriele, Luca Cavalleri, Dominique Beaini, Pietro Liò, and Petar Veličković. "Principal neighbourhood aggregation for graph nets." Advances in neural information processing systems 33 (2020): 13260-13271.
[2] Liang, Huidong, Haitz Sáez de Ocáriz Borde, Baskaran Sripathmanathan, Michael Bronstein, and Xiaowen Dong. "Towards Quantifying Long-Range Interactions in Graph Machine Learning: a Large Graph Dataset and a Measurement." arXiv preprint arXiv:2503.09008 (2025).

### Questions
na

### Soundness
2

### Presentation
2

### Contribution
2
