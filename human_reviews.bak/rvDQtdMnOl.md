# Long-Short-Range Message-Passing: A Physics-Informed Framework to Capture Non-Local Interaction for Scalable Molecular Dynamics Simulation

- Decision: Accept (poster)
- Scores: 8, 5, 6, 6

## Abstract
Computational simulation of chemical and biological systems using *ab initio* molecular dynamics has been a challenge over decades. Researchers have attempted to address the problem with machine learning and fragmentation-based methods. However, the two approaches fail to give a satisfactory description of long-range and many-body interactions, respectively. Inspired by fragmentation-based methods, we propose the Long-Short-Range Message-Passing (LSR-MP) framework as a generalization of the existing equivariant graph neural networks (EGNNs) with the intent to incorporate long-range interactions efficiently and effectively. We apply the LSR-MP framework to the recently proposed ViSNet and demonstrate the state-of-the-art results with up to 40% MAE reduction for molecules in MD22 and Chignolin datasets. Consistent improvements to various EGNNs will also be discussed to illustrate the general applicability and robustness of our LSR-MP framework. The code for our experiments and trained model weights could be found at https://github.com/liyy2/LSR-MP.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a fragment-based approach to propagate long-range information in Graph Neural Networks (GNNs). A set of fragments is constructed using the BRICS fragmentation method, which leverages chemical structures to define well-behaved fragments. The framework operates at two levels: first, it performs a message-passing step on the short-range graph, and then uses these results to define fragment-level features that are also message-passed at the fragment-level graph. The method demonstrates competitive accuracy on the MD22 benchmark, which contains large structures, and shows some improvements over short-range baselines.

### Strengths
- The paper is clearly written and pedagogical.
- Various ablation studies were conducted on both the long-range modules and the fragmentation methods.
- Results show improvements over short-range baselines for large molecules.
- Some limitations of the fragmentation methods are discussed.

### Weaknesses
- The main weakness of the method, as shared in the paper, is the definition of fragments. First, as mentioned, it is not clear how to define these for most systems, including materials. My biggest concern is the issue of smoothness. In molecular dynamics (MD) simulations, it is crucial to ensure that the predictions are smooth. I can envision many MD scenarios where such partitioning might cause problems, and I would be very interested in seeing the behavior of this model over long simulations.

- The Equiformer and VisNet models have 4 layers with a 4Å cutoff, resulting in a receptive field of 32Å in diameter. Most of the MD22 molecules fit well within their receptive fields. While this does not detract from the improvement offered by the method, it should be clearly highlighted.

- The importance of long-range effects beyond a 12Å radius is subtle, as large effects are usually screened in most systems. One would expect to see little difference in errors between a short-range and a long-range model. However, observables computed from MD simulations might vary significantly, as these long-range effects do not average out over long timescales. To capture these observables accurately, the most crucial factor is the decay of interactions, rather than raw accuracy. There is no reason to believe that your approach would correctly capture this decay, enabling accurate observables in these simulations. I want to stress that long-range effects in large biomolecular systems are mostly relevant for observables, and justifying the method solely through raw accuracy has limited scientific relevance.

- One of the main challenges of long-range modeling is transferability, especially for models without typical decay behaviors. I would be very interested in seeing how this model extrapolates to longer, unseen molecules, and whether it performs better than a local model in this context. This is the only relevant setting for practical applications, particularly for modeling systems where ab initio computations are not feasible.

### Questions
- How well do you expect your model to transfer to new, unseen systems, particularly those of larger sizes?

- Could you plot the typical decay learned by your interactions, assuming the fragmentation approach allows for it? You could try separating two molecules and plotting the energy as a function of distance. Without sensible decay, the model stands little chance of extrapolating effectively.

### Soundness
2 fair

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
This paper proposes new message-passing neural networks that capture long-range interactions by generalizing equivariant graph neural networks inspired by fragmentation-based approaches. For the implementation, BRICS fragmentation was leveraged. The authors demonstrated the effectiveness of the proposed method with a recently proposed architecture ViSNet and achieved considerable improvement in large molecule benchmarks: MD and Chignolin datasets. To evaluate the proposed method’s applicability, the authors provided results with other EGNNs such as Equiformer, PaiNN, and ET. They show consistent improvement.

### Strengths
1. **Strong empirical results** The proposed method achieved competitive performance in MD22 and Chignolin datasets. In Table 2 for MD22, the proposed method with ViSNet shows achieved the best performance in various settings. More importantly, the proposed method, LSRM, shows consistent improvement compared to the vanilla model without LSRM.
2. **General applicability** The proposed method with various EGNNs base networks shows consistent performance improvement. 
3. **Computational efficiency** The proposed method shows great performance without significant computational overhead. Rather, the proposed method has the smallest model size and the shortest training time.
4. **Comprehensive experimental results** The paper provides many details and additional experiments.

### Weaknesses
1. **Limited impact.** The technical contributions of this paper has limited impact. Although the method show overall comparable performance, it is a model that is manually-designed by domain knowledge.
2. **Narrow perspective.** Basically, the proposed method uses two different types of graphs. Then the problem can be viewed as learning on heterogeneous graphs. In recent years, learning graph neural networks on heterogeneous graphs by manually/automatically transforming graphs has been actively studied. The authors may want to include the related work and potentially compare with them. Beyond long-range dependency, non-local/semantic relations also have been utilized. 
3. **Far-fetched claim.** In Figure 2., I do not see anything but overall performance gap. I do not think that the graphs support the claim that LSRM helps models to capture long-range dependency. All three models exhibit similar behaviors.

### Questions
1. How about inference time? It was not clear how ViSNet has more parameters than ViSNet-LSRM. Also, the training time was reported, but inference time was not available. In real-world applications, inference time is more important for deployment. I believe that shorter training time would imply shorter inference time, but it should be explicitly discussed to be more comprehensive. Fig. 2, (c)(f) partially show the inference time for the subset of baselines
2. Figure 3 is confusing. The legend should be updated. 
3. Table 4 is not explicitly referred to in the text, although the paragraph of the text of Q3 in Section 5.2 discusses the result. It will be a quick fix. 
4. Typo (?) in Proposition 4.1 Hamdard -> Hadamard product (?)

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a novel framework for molecular dynamics simulations using machine learning. The framework, called Long-Short-Range Message-Passing (LSR-MP), combines equivariant graph neural networks (EGNNs) with fragmentation-based methods to capture both short-range and long-range interactions among atoms. The authors demonstrate that LSR-MP can achieve state-of-the-art results on large molecular datasets, while being more efficient and effective than existing methods. The authors also conduct ablation studies and analysis to validate the importance of incorporating long-range components and the advantages of using BRICS fragmentation.

### Strengths
- **Problem Definition**: This paper addresses a challenging and important problem of modeling large molecular systems with high accuracy and low computational cost.

- **Methodology**: This paper introduces a novel message-passing framework that leverages domain knowledge from quantum chemistry to incorporate long-range interactions efficiently and effectively.

- **Performance**: This paper shows significant performance improvements over existing methods on various benchmarks, while using fewer parameters and offering faster speed.

- **Generalizability**: This paper illustrates the general applicability and robustness of the LSR-MP framework by applying it to different EGNN backbones and showing consistent improvements.

- **Implementation**: This paper provide sufficient details on experimental setups and how the method is implemented.

### Weaknesses
- **Novelty**: I could not find any distinct weaknesses in this paper, but I might have missed one since I am not an expert in Molecular Modeling. One major concern is regarding the novelty of the proposed long-range message-passing module. As far as I know, long-range message-passing is one of the highlighted research topics in GNN literature. It would be better to discuss this line of work.

### Questions
- As far as I know, there are many long-range message-passing modules designed for graph-structured data. Can you compare the proposed method with other long-range message-passing modules?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new framework for machine learning of molecular dynamics, called Long-Short-Range Message-Passing (LSR-MP). LSR-MP combines short-range and long-range message passing on graphs to capture both local and non-local interactions in chemical and biological systems. LSR-MP uses a fragmentation-based method inspired by quantum chemistry to divide large molecules into smaller subsystems and model their long-range interactions efficiently and effectively. LSR-MP is implemented on top of an existing equivariant graph neural network (EGNN) called ViSNet, and achieves state-of-the-art results on large molecular datasets with fewer parameters and faster speed. LSR-MP is also applied to other EGNN models and shows consistent improvements, demonstrating its general applicability and robustness.

### Strengths
The paper presents a novel and elegant framework for long-short-range message passing on graphs, which can capture both local and non-local interactions in chemical and biological systems. 
The paper draws inspiration from quantum chemistry and adopts a fragmentation-based method to divide large molecules into smaller subsystems and model their long-range interactions efficiently and effectively. This is a clever and creative way to overcome the computational and memory challenges of existing methods. The paper implements the proposed framework on top of an existing equivariant graph neural network (EGNN) called ViSNet, and demonstrates its superior performance on two large molecular datasets, MD22 and Chignolin. The paper shows that the proposed method achieves state-of-the-art results with fewer parameters and faster speed than the baselines, which is impressive and convincing.
The paper also applies the proposed framework to other EGNN models, such as PaiNN, ET, and Equiformer, and shows consistent improvements across different architectures and datasets. This demonstrates the general applicability and robustness of the proposed framework, and suggests that it can be easily integrated with other existing methods.

### Weaknesses
The paper does not provide a clear analysis of the stability, and error bounds and how sensitive the performance of the method is to the choice of these modules and parameters.

### Questions
Q1. How do you justify the choice of the LSR-MP framework as a generalization of the existing EGNNs? What are the advantages and limitations of this framework compared to other possible ways of incorporating long-range interactions, such as attention mechanisms, continuous filters, or Fourier features?

Q2. How do you ensure the stability and accuracy of the BRICS fragmentation method for different types of molecules and systems? How sensitive is the performance of the LSR-MP framework to the choice of the fragmentation method and the number and size of the fragments?

Q3. How do you evaluate the scalability and efficiency of the LSR-MP framework for larger and more complex molecular systems? What are the computational and memory costs of the LSR-MP framework, and how do they compare with the conventional quantum chemical methods and other machine learning methods? How do you handle the trade-off between accuracy and efficiency in the LSR-MP framework?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair
