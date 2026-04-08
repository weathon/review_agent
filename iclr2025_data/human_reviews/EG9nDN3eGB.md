## Human Reviewer 1

### Summary
This authors propose a novel data-driven circuit symbolic learning framework, CMO. It learns a symbolic scoring function balancing inference efficiency, interpretability, and generalization performance. While existing approaches often struggle with these trade-offs in modern circuit synthesis (CS) tools, CMO demonstrates superior capability in discovering lightweight and interpretable symbolic functions from a decomposed symbolic space. The major technical contribution of CMO is the Graph Enhanced Symbolic Discovery (GESD) framework, which employs a specially designed Graph Neural Network (GNN) to guide the generation of symbolic trees. CMO is the first graph-enhanced approach for discovering lightweight and interpretable symbolic functions that effectively generalize to unseen circuits in CS.

### Strengths
1. Overall, the proposed work is well-structured with a profound related work.
2. This paper proposes a novel circuit symbolic learning framework to learn efficient, interpretable, and generalizable symbolic functions that are reliable and simple to deploy in modern CS tools.
3. CMO is the first graph-enhanced approach for discovering lightweight and interpretable symbolic functions that can well generalize to unseen circuits in CS.  
4. Extensive experimental results show the effectiveness of the proposed CMO over existing works.

### Weaknesses
The link between the two methods in sections 4.1 and 4.2 needs to be further elucidated, and it is not currently possible to visualize in the text the specific interrelationships between the two methods. For example, what is the role of si in section 4.1 in section 4.2 and what is the flow of the calculations for si.

### Questions
Please check weakness

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper studies the problem of circuit synthesis (CS) via graph-based methods that can generalize to unseen circuits. The proposed method, CMO, combines symbolic function learning with Graph Neural Networks.

### Strengths
1. The paper is well-written with good introduction to the domain especially for ML-audience less familiar with hardware design.
2. Choice of benchmarks and state-the-art heuristics seem to be solid with comprehensive evaluation.
3. The proposed method achieves significant speedup while maintaining optimization performance on real circuits.

### Weaknesses
1. Theoretical justification and analysis are lacking – It seems combining GNN, MCTS, symbolic learning etc. leads to better results on these CS benchmarks, yet some deeper explanation and analysis can be provided to make the paper stronger.
2. Some of the writings can be improved, e.g. “However, this approach cannot capture effective information from specific circuit distribution for higher generalization performance” – Is it due to the human-designed nature and lack of adoption of machine learning from existing data?
3. Some technical errors, e.g. “Specifically, we use mean absolute error and focal loss” yet the equation (4) is an MSE loss.
4. More circuit dataset descriptions, e.g. graph sizes, and graph visualizations would provide a more solid background for ML audience.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper proposes a method called CMO to develop lightweight and generalizable scoring functions for ranking nodes in an AIG, aiming to enhance the efficiency and performance of logic optimization. The method is clearly introduced. The paper trains a GNN and uses an MCTS-based symbolic regression method to generate symbolic scoring functions, ensuring both inference efficiency and generalization. However, some experimental details remain unclear.

### Strengths
The method is clearly introduced, with comprehensive experiments demonstrating the effectiveness and performance of CMO.

### Weaknesses
### Topic
The term "circuit synthesis" is not well-defined in the field of EDA, which may cause confusion. Based on the related works and experiments, this paper appears to focus on logic optimization.
### Datasets
The labels of circuit datasets should be clarified. When mentioning node-level transformation, does it mean it is effective for the current step of logic optimization or for overall performance? Effectiveness in the current step may not translate to overall performance in logic optimization.
### Experiments
The experiment part is somewhat confusing. The focus of logic optimization should be on time cost and node reduction during the online phase. The offline phase appears more like an ablation study. Experiments on generalization should be highlighted in the main part of the manuscript. Experiment 4 should showcase generalization compared to other baselines like COG, but why other SR methods? Is this an ablation study of the SR method used?
### Generalization
EPFL, IWLS, and an industrial-level dataset from Huawei HiSilicon are used to train the GNN. Are the datasets mixed to train a single GNN, or are three separate GNNs trained for each dataset?

### Questions
1. Can CMO generalize to other logic optimization methods like Rewrite?
2. Can a GNN trained on one dataset generalize to another dataset?
3. How are the training dataset labels obtained?
4. The presentation of experiments is confusing. Please clarify which is the main experiment and which are ablation studies.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
8

### Confidence
4