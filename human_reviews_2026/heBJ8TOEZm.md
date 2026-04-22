# ComMat: Datasets and Benchmarks from complex materials for graph machine learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Recent research has demonstrated the efficacy of graph learning over a wide spectrum of materials, including molecular graphs, crystals, mechanical metamaterials, and strongly disordered systems. In this work, we draw attention to the broad class of *complex materials*, which combine order and disorder, and fall outside the above categories, yet have shown superior properties throughout the materials science literature. We present a Complex Material Benchmark (ComMat), including three graph datasets of complex materials from experimental and computational research studies, unifying distinctly developed data-to-graph pipelines under a standardized graph-based representation. In particular, we provide the first publicly available 3D graph dataset of a nanoscale network derived from 3D tomography. We then quantitatively show that these graphs are fundamentally different from existing materials datasets. We design various predictive tasks to advance machine learning (ML) methods, including experimentally measured properties, simulated mechanical response, and structural awareness. Extensive benchmark experiments are conducted over popular graph learning models, revealing their limitations and the need for further development in handling complex materials. ComMat is openly released to accelerate ML research and innovation in complex material design.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents ComMat, three computationally/experimentally obtained graph datasets of *complex materials*, defined here as order + disorder, including polymer network, nanowire networks, and aramid nanofibers. On the datasets, descriptive data analyses are performed, focusing on contrast with ordered and disordered materials families. Benchmark tests of graph neural network (GNN) models are conducted with link prediction, resistance prediction, and fracture prediction as the tasks.

### Strengths
- Constructing datasets and benchmarks for complex materials is timely and well-motivated, as graph representation and GNN models have demonstrated their capability in modeling simpler materials and gradually been adopted to complex materials.
- The paper is overall well-written and easy to follow, despite minor clarity issues (see Q4).
- The analyses of graph complexity using several metrics (Sec. 3.4) provide useful insights into understanding complex materials.

### Weaknesses
- The dataset size is limited, rendering the findings from benchmark tests thereon questionable (see Q1).
- Related to the previous point, the benchmark tasks could be better designed (see Q2).

### Questions
1.	The nanowire dataset contains 33 graphs, usually viewed as too few for GNN models. How much would lack of data (instead of models’ incapability) attribute to the models’ suboptimal performance? Also because of limited data size, the conclusion on benefit of geometric information (Line 450) is not backed up.
2.	Related to 1: Have the Authors considered other types of tasks (e.g., “local” or “node-level”), where the current data could be viewed as abundant?
3.	Some related studies on disordered materials and their graph-based models should be acknowledged: DOI: 10.1088/2632-2153/adc0e1; DOI: 10.1073/pnas.2322962121.
4.	Minor clarity issues
  - In Sec. 3.2, what are the nodes and edges of graphs? Besides, “33 graphs across four sets” and “experiments” are mentioned twice, which seems redundant.
  - In Sec. 4, when building GNN models, what are the node/edge features?
  - Sec 4.2 title “Baselines” is confusing, as there is no newly proposed model to be compared with these existing ones.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a Complex Materials Benchmark (ComMat), with three datasets (experiment and simulation). Complex is (vaguely) defined as not falling into the category of crystalline periodic, nor amorphous materials, and thus not covered by current GNN approaches on either sides.

### Strengths
Introducing complex materials as a new graph learning application with interesting node/edge/graph regression/classification tasks and maybe even generative tasks is interesting and might be relevant for the materials AI community.

### Weaknesses
The definition of complex materials is not really clear; see questions. Overall, all introduced tasks are on networks on a coarse-grained level (no atom resolution). It is not fully clear if this is only a special case of what the authors mean by complex materials, and what the ordered aspects of this are, if the atom or even monomer resolution is given up. It makes sense that atoms in polymer networks have some aspects of order (along the backbone) and disorder (in between chains) in their local environment and radial distribution function. But on the CG level, this is unclear. Also, if only networks are studied here, this benchmark does not seem applicable to a broader definition of complex materials.

The meaning of the related work section seems unclear; see questions.

Generation time: The authors suggest the introduced data sets to be used as benchmark tasks. However, there seems to be a mismatch between the (rather small) amount of work that goes into generating the datasets and the expected amount of work (months to years of work plus tens of thousands of GPU hours) that might go into ML method development (assuming complex materials are relevant and interesting enough for the community to focus more on method development in this direction). Would it not make more sense to have larger datasets, more datasets, more complex in-distribution and out-of-distribution splits, more diverse complex materials classes (beyond networks), or just different types of polymers and nanowires/fibers if this should serve as a benchmark?

### Questions
"When materials are neither ordered nor disordered, but instead complex, ...". This seems not sufficient as a definition. Would a doped material or a material with point defects (structural order, occupational disorder) be complex according to this definition? Would a material with structural (1D/2D) defects be complex? "betweenness centrality" is also not clearly defined, even if it might be contained in Smart et al., but this definition seems highly important here so it should be well-defined and self-contained.

Related work: "we review existing datasets and benchmarks on materials science that do not fall into the complex material category, yet provides useful insight in our development of this contribution". It seems like you reviewed those datasets, but you did not discuss the useful insight they provide, nor the relation to your work.

Nanowire networks: "This dataset comprises 33 graphs": How are the graphs defined here? Why do you call those networks? Are they not just independent 1D wires? The same question applies to the aramid nanofibres. If those are not cross-linked like the polymer networks (?) then it sounds like the system consists of independent 1D fibers, and the graph definition remains unclear.

Link prediction: As the materials are (by definition) partially disordered and thus contain some degree of randomness: What is the upper limit that can be reached in a link prediction task, and what is the fundamental noise level? In the polymer dataset (but also the others), all models seem to reach rather similar performance, which might indicate that they are already close to the noise limit, and thus it cannot be expected that further model development will increase the AUC values. Learning curves (training/test AUC vs. training set size) would help to see potential effects of saturation. 

Table 3: What are the units here? How is RE defined? What are the r2 values? What is the label distribution? How can a graph regression problem be solved with only 33 graphs and thus 33 labels? 

Table 4: What is the definition of mMAE? mean over all links? Or graphs? What does "100 links for each label"? mean What are the labels (as you write this is a link regression problem)? What is the unit here, and what is the distribution of labels and errors?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The present paper proposes a new benchmark called a complex materials benchmark (ComMat). ComMat focuses on complex materials where both ordered and disordered structures are present. Such materials not only show interesting characteristics but also challenge existing machine learning algorithms, because it is not trivial to mathematically represent the mixture of ordered and disordered structures. The authors propose this benchmark to motivate the community to develop algorithms for complex materials.

ComMat consists of three data sets. One is on polymer networks which are derived from molecular dynamics. Another is on nanowire networks where data are obtained by scanning electron microscope. The third one is on aramid nanofibers which are obtained by 3D tomography. The original data are converted into graph representations for down-stream tasks, and network complexity is analyzed by utilizing network scientific tools.

The authors define several prediction tasks and provided reference performance by baseline GNN-based methods. The results demonstrate that some of the tasks are far from being solved, suggesting that more sophisticated models have to be tailored.

### Strengths
- This paper generates data sets for benchmark.
- This paper defines several realistic tasks which are formulated as prediction tasks. 
- The above strengths are essential for benchmarks.

### Weaknesses
The number of graphs in the Nanowire dataset is too small to formulate a graph regression problem. According to Table 1, the Nanowire dataset contains only 33 graphs, and the authors define a resistance prediction task on this dataset, which aims to predict the electrical resistance for each graph. Given such a small number of training examples, it is difficult to even evaluate the performance of predictive models.

Another concern is its relevance to real applications. While resistance prediction and fracture prediction are relevant to applications, it is not clear to me whether link prediction is relevant to real-world applications.

### Questions
- Are 33 graphs enough to formulate the resistance prediction task?
- It would be helpful if the authors could elaborate more on the importance of the link prediction task.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ComMat, a novel benchmark suite of graph datasets derived from complex materials—systems that exhibit both order and disorder and therefore lie beyond conventional crystalline or amorphous material classifications. The authors unify data from diverse sources, including experimental microscopy, molecular dynamics simulations, and 3D tomographic reconstructions, into standardized graph representations to enable machine learning on these structurally intricate systems. ComMat comprises three datasets: Polymer Networks generated from 3D molecular simulations, Nanowire Networks extracted from 2D microscopy images, and Aramid Nanofibers reconstructed from 3D tomography. The benchmark defines multiple predictive tasks—link prediction, fracture prediction, and property (resistance) prediction—and evaluates a range of classical and geometric graph neural network (GNN) models. Experimental results reveal the limited performance of existing GNN architectures on these datasets, highlighting their difficulty in capturing the long-range connectivity and multi-scale structural dependencies characteristic of complex materials, and emphasizing the need for new graph learning approaches tailored to this emerging domain.

### Strengths
- ComMat is a new and useful resource for applying graph-based machine learning to study complex materials that have both order and disorder.

- It targets an important area that hasn’t been well represented in existing materials datasets.

- The authors combine data from different sources, such as experiments, simulations, and 3D imaging, into a single, easy-to-use format.

- This makes the dataset valuable for both materials scientists and machine learning researchers.

- They tested several well-known graph neural network (GNN) models, including GCN, GAT, GraphSAGE, EGNN, and Equiformer, on different prediction tasks.

- The source code is provided, and the dataset will be publicly released, making it easier for other researchers to use and extend.

### Weaknesses
- The core contribution lies in dataset creation and analysis, not algorithmic innovation. The benchmarking is fairly standard and doesn’t propose new learning architectures.
- The datasets vary drastically in size (e.g., one nanofiber graph with >300K nodes vs. 33 nanowire graphs), potentially limiting the generality of model comparisons.
- While the authors discuss the limitations of current GNNs for complex materials, they don’t experimentally test any adapted or domain-informed variants (e.g., centrality-aware message passing, hierarchical pooling).

### Questions
See the weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
