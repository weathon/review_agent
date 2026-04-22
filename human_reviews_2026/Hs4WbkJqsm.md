# Wef-GNN: A Generalizable Graph Neural Network for Crystalline Material Property Prediction

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Graph neural networks (GNNs) have shown great promise for predicting properties of crystalline solids. However, existing models struggle to generalize across crystals of varying sizes, and there is a lack of high-fidelity $\textit{ab initio}$ training data. Here, Wef-GNN addresses the problem of generalizability by introducing a multi-head temporal attention mechanism in the graph update function and a crystalline graph representation scheme that is more size-agnostic compared to the traditional primitive unit cell-based graph representation. Further, it was found that a single Wef-GNN layer can be recycled for all graph convolution steps without considerable loss in accuracy; this leads to deep receptive fields without additional parameters. Wef-GNN outperforms all prior models in a standard band gap prediction benchmark while having much fewer parameters. To address the challenge of high quality $\textit{ab initio}$ training data, a high-fidelity dataset was curated by performing 10,522 high-accuracy Density Functional Theory (DFT) calculations. Wef-GNN was pre-trained on a standard large dataset of lower-accuracy DFT calculations then fine-tuned with the high-accuracy DFT dataset. The resulting model matches experimental band-gap values much better than other GNNs, and even outperforms the underlying low-accuracy DFT calculations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper builds a GNN and demonstrates it on bandgap prediction. There are three mentioned contributions:

- A new multi-headed temporal attention mechanism which is proposed to improve treatment of different structure sizes
- An empirical observation that the same weights could be re-purposed across message passing layers
- Training on low- and fine-tuning on high-fidelity data

### Strengths
N/A

### Weaknesses
This paper is very thin. Some of the space is spent on fairly simple and well-known effects in the materials modelling community, for example using a cutoff radius to connect a graph neural network. 

The background section of this paper is outdated. A wealth of advanced MLIPs have appeared in the past three years, trained on vast ranges of high-fidelity data. Many of the claims made in the background section of this paper no longer hold.  

Training on low-fidelity data and finetuning on high-fidelity data is not new. The empirical finding that the weights can be recycled across multiple layers is mildly interesting, but even at empirical level it is not well explored. 

There are extremely few results (Table 1) and no ablation studies of the architecture.

I think this work is suitable for a workshop in order to further develop the core ideas.

### Questions
N/A

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
1

### Summary
The paper introduces a GNN layer and a high-accuracy DFT materials bandgap dataset.

### Strengths
The paper achieves state-of-the-art performance on band gap prediction accuracy on the MatBench benchmark, beating the previously best model by a substantial margin.

It is very surprising that the weight sharing across layers, i.e., the use of a fully recurrent graph neural network, does not lead to a performance drop. It would be good to further analyze this behavior. Would the same be true if the temporal attention-based update function were used in other MPNN models?

### Weaknesses
The suggestion of going from primitive unit cells to conventional unit cells to improve generalizability is unclear. If periodic graphs are initialized correctly, there is no difference in node updates and node embeddings between a primitive unit cell and a conventional one, as the underlying periodic graph, including all individual atom environments, is exactly the same. Thus, the graph representation after readout and the prediction are identical. If this is different for WefGNN, then the authors should provide more information about this.
Furthermore, the argument that this helps in generalization to larger unit cells also seems misleading. Generalization in ML is defined on in-distribution data, not out-of-distribution data. There is no guarantee for any performance on out-of-distribution data. Simply increasing the unit cell size periodically (the materials' properties are invariant to this) without adding any additional materials with actually larger primitive unit cells to the training data will not increase the performance on materials with large primitive unit cells.

The WefGNN model is only benchmarked on one of the tasks in MatBench. What about all other tasks? 

Section 5 (hybrid training) is interesting but not really relevant for a machine learning audience. It introduces a new dataset with more accurate DFT calculations, trains (or fine-tunes) the WefGNN model on this dataset, and shows that this leads to better agreement with experimental data. This is not surprising, as the newly generated DFT data has higher agreement with experimental data. The test set error of Model B vs. the HSEE06 values is not reported and should be added. 

Overall, the paper is lacking a lot of analysis, further benchmarks, ablations, and also some basic information about the model itself.

### Questions
- Why was WefGNN only benchmarked on band gaps?
- Please show an ablation study that supports that not using primitive unit cells actually improves model performance
- What is the aspect in Algorithm 1 that differentiates WefGNN from other GNN models? The first loop just seems to be a way to preprocess the geometry to find edges, and the second part is a conventional aggregation, updat,e and readout cycle. Please provide further ablation studies of your components (temporal attention, ...) to demonstrate the effect of each of the design choices.
- What are the initial feature vectors for nodes and edges? How is the geometry of the unit cell used? Is the model invariant to rotations/translations?

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
4

### Summary
The paper introduces Wef-GNN, a graph neural network designed for predicting crystalline material properties with a focus on improving generalizability across crystals of varying sizes and achieving higher accuracy than traditional PBE-level DFT datasets. The proposed model incorporates several key innovations: a multi-head temporal attention mechanism in the update step that enables each node to attend to its historical representations, parameter recycling across message-passing layers to expand receptive fields without increasing the number of learnable parameters, and the use of conventional unit cells instead of primitive ones to enhance consistency and generalization across different crystal structures. Additionally, the authors propose a hybrid training strategy that involves pretraining the model on large-scale, lower-accuracy PBE datasets followed by fine-tuning on a smaller, high-fidelity HSE06 dataset. Through this combination of architectural and training improvements, Wef-GNN achieves a mean absolute error (MAE) of 0.117 eV on the Matbench band-gap prediction task, surpassing existing models while using significantly fewer parameters.

### Strengths
- The authors clearly identify two limitations in existing GNN-based materials models — lack of generalization and limited accuracy due to low-fidelity DFT datasets.
- Introducing temporal attention within GNNs for crystalline materials and the weight recycling strategy is interesting and computationally efficient.
- Code and dataset availability are mentioned, which aligns with ICLR’s reproducibility requirements.
- Achieves state-of-the-art performance (0.117 eV MAE) on Matbench Benchmark.

### Weaknesses
- The paper shows strong empirical results but lacks theoretical explanation or ablation to justify why temporal attention or weight recycling improves generalization.


- Most design elements—attention, message passing, and parameter sharing—are already well-established; their combination is effective but not fundamentally novel.


- Ablation study comparing temporal attention vs. static attention is missing, leaving the claimed benefit unsubstantiated.


- The paper reads more like a well-structured technical report than a concise ICLR-style paper. The submission appears incomplete, ending at around 6.5 pages with large images, whereas ICLR papers typically utilize the full 9-page limit.

### Questions
See the Weaknesses

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper “Wef-GNN: A Generalizable Graph Neural Network for Crystalline Material Property Prediction” presents a lightweight, size-agnostic GNN architecture for predicting material properties, particularly band gaps. Wef-GNN integrates a multi-head temporal attention mechanism, recycled GNN layers, and a conventional-cell representation to enhance generalizability across crystals of varying sizes. The experimental results demonstrate strong and promising performance.

### Strengths
This paper proposes Wef-GNN, a graph attention–based model that incorporates a multi-head temporal attention mechanism, GNN layer recycling, and a novel graph representation scheme for crystal structures.
This work demonstrates promising results for the band gap property through the introduction of a novel multi-head attention mechanism within Wef-GNN.

### Weaknesses
The GNN models commonly used for crystalline materials, such as Matformer [1], and PotNet [2], are not referenced in the paper. Please include these pioneering works in the related work section.
This paper notes that the MP and OQMD datasets suffer from DFT-induced error bias. However, recent studies such as CrysDiff [3] and CrysGNN [4] have demonstrated that pretraining GNNs can effectively mitigate this bias by incorporating a small amount of experimental data alongside DFT-generated data during training. Consequently, the necessity of creating a new HSE dataset is questionable. Furthermore, these techniques are applicable to a wide range of material properties, not just band gap prediction.
Here, multi-head attention is used for aggregation. However, transformer-based methods are now widely adopted for crystal property prediction. Why did the authors choose to use the older graph attention variant instead? It would be helpful to include an ablation study comparing the effectiveness of transformers versus graph attention for this specific task. Moreover, the results section currently lacks any ablation analysis. Additionally, the meaning of “Wef” in the model name Wef-GNN is unclear and should be clarified.
Could the authors provide statistics on how many materials contain fewer than two atoms? Since crystals with fewer than two atoms are chemically insignificant, this raises the question of why varying the number of message-passing layers is necessary.
Since this paper focuses solely on band gap prediction, the authors should also evaluate other key material properties—such as formation energy, total energy, Ehull, shear modulus, and bulk modulus—to better demonstrate the generalizability of Wef-GNN. Moreover, they should validate their model on established benchmark datasets like JARVIS [5] and MP-2018. Additionally, although the paper introduces a new dataset, it lacks any analysis of its characteristics, such as the distribution or number of atoms per structure.
From Table 1, it is evident that Wef-GNN performs well; however, the results remain incomplete since the authors do not compare it with current supervised state-of-the-art models such Matformer. Furthermore, given their use of pre-training, they should have also included comparisons with recent pre-trained models like CrysGNN and CrysDIFF. In addition, the paper appears unfinished, ending at 7.5 pages and lacking any ablation studies.



References:

[1] Yan, K., Liu, Y., Lin, Y. and Ji, S., 2022. Periodic graph transformers for crystal material property prediction. Advances in Neural Information Processing Systems, 35, pp.15066-15080.

[2] Lin, Y., Yan, K., Luo, Y., Liu, Y., Qian, X. and Ji, S., 2023, July. Efficient approximations of complete interatomic potentials for crystal property prediction. In International conference on machine learning (pp. 21260-21287). PMLR.

[3] Song, Z., Meng, Z. and King, I., 2024, March. A diffusion-based pre-training framework for crystal property prediction. In Proceedings of the AAAI Conference on Artificial Intelligence(Vol. 38, No. 8, pp. 8993-9001).

[4] Das, K., Samanta, B., Goyal, P., Lee, S.C., Bhattacharjee, S. and Ganguly, N., 2023, June. Crysgnn: Distilling pre-trained knowledge to enhance property prediction for crystalline materials. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 6, pp. 7323-7331).

[5] Choudhary, K., Garrity, K.F., Reid, A.C., DeCost, B., Biacchi, A.J., Hight Walker, A.R., Trautt, Z., Hattrick-Simpers, J., Kusne, A.G., Centrone, A. and Davydov, A., 2020. The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design. npj computational materials, 6(1), p.173.

### Questions
See the limitations

### Soundness
2

### Presentation
2

### Contribution
2
