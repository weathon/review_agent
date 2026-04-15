# A Benchmark on Directed Graph Representation Learning in Hardware Designs

- Decision: Reject
- Scores: 3, 6, 6, 8

## Abstract
To keep pace with the rapid advancements in design complexity within modern computing systems, directed graph representation learning (DGRL) has become crucial, particularly for encoding circuit netlists, computational graphs, and developing surrogate models for hardware performance prediction. However, DGRL remains relatively unexplored, especially in the hardware domain, mainly due to the lack of comprehensive and user-friendly benchmarks. This study presents a novel benchmark comprising five hardware design datasets and 13 prediction tasks spanning various levels of circuit abstraction. We evaluate 21 DGRL models, employing diverse graph neural networks and graph transformers (GTs) as backbones, enhanced by positional encodings (PEs) tailored for directed graphs. Our results highlight that bidirected (BI) message passing neural networks (MPNNs) and robust PEs significantly enhance model performance. Notably, the top-performing models include PE-enhanced GTs interleaved with BI-MPNN layers and BI-Graph Isomorphism Network, both surpassing baselines across the 13 tasks. Additionally, our investigation into out-of-distribution (OOD) performance emphasizes the urgent need to improve OOD generalization in DGRL models. This benchmark, implemented with a modular codebase, streamlines the evaluation of DGRL models for both hardware and ML practitioners.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a benchmark for learning representations of directed graphs in the hardware design field. It comprises 5 datasets and covers 13 different tasks, from node-level to graph-level. The authors also conduct experiments to evaluate the critical components in the design of graph neural networks.

### Strengths
1. OOD testing is proposed for each dataset to evaluate the generalization ability of the models.
2. Through various and rich experiments, bidirected MPNN and NPE+EPE are highlighted to enhance the model performance and stability.

### Weaknesses
1. The logical relation between different datasets is relatively weak. For instance, although all these datasets belong to the 'hardware design' field, what is the relation between the graph of CNN and the circuit netlist? It could be better to organize corresponding datasets more logically, instead of just gathering them together. I suggest classifying these circuits into FPGA, ASIC and computational graph (CNN architecture), and with each category, the datasets could be better to cover more/complete design stages.
2. Another widely used benchmark, CircuitNet 2.0 [1], covers the full chip design flow, including synthesis, floorplan, powerplan, placement, CTS and routing. While in this paper, only some separate stages are involved. It could be better to cover more/complete stages for each circuit in the benchmark. After covering the full chip design flow, this benchmark could be better to evaluate the performance of newly designed GNN across various tasks. For example, AIG --(logic synthesis)--> synthesized netlist --(placement)--> netlist with gate coordinate/position. These intermediate results focus on different graph properties, and the benchmark with these datasets could be utilized to test the performance of GNN specialized for these (continuous) stages.
3. All datasets in this benchmark are created by previous works, undermining the contribution of the work. I think it is not sufficient to support to release a new comprehensive benchmark.
4. The stability of the "stable positional encoding" is questionable, as it struggles to handle larger graphs with more than 100,000 nodes, a scale which is common in hardware design.

**References**

[1] Jiang, Xun, et al. "CircuitNet 2.0: An Advanced Dataset for Promoting Machine Learning Innovations in Realistic Chip Design Environment." ICLR. 2024.

### Questions
1. In Table 1, the OOD experiments for benchmark symbolic reasoning are 32, 36, and 48-bit. However, in lines 223 to 225, the training/ID test uses multipliers before technology mapping, and the OOD testing data uses multipliers after technology mapping. This is inconsistent with the table.
2. In the TIME dataset, the authors only conduct experiments on small circuits with a maximum number of nodes 58,676 and a maximum number of edges 83,225. In the original paper, i.e. Guo et al. (2022), there exists larger circuits whose number of nodes and edges exceeds 100,000 even 200,000. Could the authors provide results on these larger circuits or explain why they only conduct experiments on small circuits?
3. In the OOD testing of the TIME dataset, what is the principle for selecting the circuits to test the OOD performance? Could the authors provide more details about "the difference in graph structures" mentioned in line 235?
4. Could the authors provide more details about the "permutation equivariant functions $\kappa$ and $\rho$"? How do they contribute to the stability of positional encoding?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a benchmark for directed graph representation learning with a focus on hardware design tasks. It includes several hardware design datasets and prediction tasks, on which the paper evaluates 21 DGRL models that vary in their backbones, message passing directions, etc. The paper also provides insights regarding the set of most helpful network architectures for enhancing model performance.

### Strengths
1. This paper proposes a well-rounded benchmark that spans five datasets with various prediction tasks at different abstraction levels, thereby being able to reflect a wide range of practical hardware design scenarios.
2. The paper also carries out an extensive model evaluation that leads to some unique insights, e.g., the importance of handling OOD cases for a more generalizable DGRL.

### Weaknesses
- It is not clear how novel the provided insights are as the rationale behind such insights is not clearly summarized in the text of sections 4 & 6.
- The paper claims to discover some of the novel model designs that are helpful to DGRL, yet the novelty is not highlighted in the paper. The best-performing model also happens to be the most complex model design, which is a combination of different techniques mentioned in section 4.
- The conclusion from the experiments is rather simple and direct, missing task-specific analysis and insights.

### Questions
1. The conclusion of the paper points out that some particular model design works the best for DGRL, so what are the reasons behind such results? Why does ‘Bidirected’ (BI) message passing show better accuracy?
2. For different tasks, is there more detailed guidance on which model should be used?
3. What is the main contribution of the paper, in terms of the guidance it provides for designing DGRL? It should be noted that EPE is expected to outperform NPE even before the experiment, the graph transformer-based method is well-known to have scalability issues, and traditional GNN is naturally more efficient when the graph is large. Maybe the paper can provide more discussion on how to cope with the scalability issue of graph transformer in large-scale DGRL, or how to involve more domain knowledge in the model design.

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
5

### Summary
Directed graph representation learning (DGRL) has become essential for handling the growing complexity of modern computing systems, especially in areas like circuit netlists and computational graphs. The authors state that the DGRL in the hardware domain remains underdeveloped due to the lack of robust, user-friendly benchmarks. This paper introduces a benchmark with five hardware design datasets and 13 prediction tasks across different circuit abstraction levels. Evaluating 21 DGRL models, including graph neural networks and graph transformers enhanced with positional encodings, the study finds that models using bidirected message passing networks (BI-MPNNs) and robust PEs perform best. Additionally, it highlights a critical need to improve out-of-distribution (OOD) generalization in DGRL, with a modular codebase provided for ease of evaluation by hardware and ML researchers.

### Strengths
+ Benchmark that integrates control conditions with data dependencies extracted from IR graphs

### Weaknesses
- Benchmark structure is not fully explained and especially the differences across the domains.
- Analysis of graph representation learning seems incomplete because only the size of the graphs has been considered.

### Questions
A) The paper finds that bidirected (BI) message passing neural networks can substantially improve the performance. Does it imply that the direction of the edges is not meaningful in the DAGs for HW design?
B) The plus sign in Figure 2 could be explained in the caption. It will be really helpful for understanding.
C) The paper refers to “control and data flow graphs” obtained from LLVM IR of C++ codes but fails to discuss actually the very first papers that provided this compiler and graph construction such as "A load balancing inspired optimization framework for exascale multicore systems: A complex networks approach." In 2017 IEEE/ACM International Conference on Computer-Aided Design (ICCAD), pp. 217-224. IEEE, 2017,  that enable further graph optimizations. 
D) Section 4.3 mentioned “We auto-tune the hyper-parameters with seed 123 with 100 trial budgets and select the configuration with the best validation performance.” Can author specify the validation performance? 
E) The discussion of related work on directed graph representation learning needs to consider both recent works on graph based presentations of compiler based hardware-software codesign and compiler based high level synthesis like "GAHLS: an optimized graph analytics based high level synthesis framework." Scientific Reports 13, no. 1 (2023); "High-level synthesis using the Julia language." arXiv preprint arXiv:2201.11522 (2022); "End-to-end programmable computing systems." Communications Engineering 2, no. 1 (2023); "CEDR: A compiler-integrated, extensible DSSoC runtime." ACM Transactions on Embedded Computing Systems 22, no. 2 (2023). Graph representation learning has also been used in device placement, etc.
F) For CG, why ‘Proxylessass’, ‘ResNets’, and ‘SqueezeNets are selected as OOD? Is there any reasoning behind?
G) Usually recent graph representation learning techniques are exploiting a number of graph properties like graph geometry. Can the authors comment also on the role of graph geometry?
H) A small minor issue is that IR does not seem to be defined but I assume the authors refer to IR as intermediate representation.

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
3

### Summary
This paper proposes a novel benchmark to evaluate GNN models on hardware designs. Hardware designs can generally be represented as directed graphs, such as the control data flow graph, circuit netlists, etc. Previous benchmark papers usually focus on one specific hardware area. To fill in the gap of a comprehensive benchmark, this benchmark contains five previous hardware design datasets in different areas, and there are 13 prediction tasks in total. The five areas include high-level synthesis, symbolic reasoning, pre-routing timing prediction, computational graphs, and operational amplifiers. To make it more user-friendly, this paper constructs a toolbox using PyTroch Geometric. Besides, it proposes edge positional embedding for the graph Transformer model, which is more stable than the regular node positional embedding. It conducts experiments on 21 GNN models. Bidirected GNN with edge positional embedding performs the best.

### Strengths
(1)   The datasets cover many types of hardware tasks, spanning different levels of hardware design. It could be beneficial to different levels of hardware design.

(2)   The provided toolbox is convenient to use.

(3)   The proposed stable positional embedding is reasonable.

(4)   This paper conducts extensive experiments to compare 21 GNN models and 13 tasks. It shows lots of work.

### Weaknesses
(1)   The datasets are all originally created by previous works.

(2)   Some of the datasets might not be representative or complex enough. For example, the HLS dataset does not contain any HLS pragmas. However, we usually need to add HLS pragmas when we use HLS. Also, according to Table 1, the HLS dataset only has 95 nodes on average, which are very small CDFGs. Real-world HLS applications have longer codes. There might be a gap between some of the datasets and real applications. Besides, it will be better if the paper contains a dataset selection criteria or motivation.

(3)   The formula of the edge positional embedding in Table 3 is not very clear. The authors could improve the writing of this subsection.

### Questions
(1)   In Table 5, sometimes NPE hurts the performance. Why does it happen, and why is EPE more stable?

(2)   You selected one dataset for each task. For some of the tasks, there might be more existing datasets. Why did you select these specific datasets?

### Soundness
3

### Presentation
3

### Contribution
3
