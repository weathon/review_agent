# Intrinsic Lorentz Neural Network

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 2

## Abstract
Real-world data frequently exhibit latent hierarchical structures, which can be naturally represented by hyperbolic geometry. Although recent hyperbolic neural networks have demonstrated promising results, many existing architectures remain partially intrinsic, mixing Euclidean operations with hyperbolic ones or relying on extrinsic parameterizations. To address it, we propose the \emph{Intrinsic Lorentz Neural Network} (ILNN), a fully intrinsic hyperbolic architecture that conducts all computations within the Lorentz model. At its core, the network introduces a novel \emph{point-to-hyperplane} fully connected layer (FC), replacing traditional Euclidean affine logits with closed-form hyperbolic distances from features to learned Lorentz hyperplanes, thereby ensuring that the resulting geometric decision functions respect the inherent curvature. 
Around this fundamental layer, we design intrinsic modules: GyroLBN, a Lorentz batch normalization that couples gyro-centering with gyro-scaling, consistently outperforming both LBN and GyroBN while reducing training time. We additionally proposed a gyro-additive bias for the FC output, a Lorentz patch-concatenation operator that aligns the expected log-radius across feature blocks via a digamma-based scale, and a Lorentz dropout layer.
Extensive experiments conducted on CIFAR-10/100 and two genomic benchmarks (TEB and GUE) illustrate that ILNN achieves state-of-the-art performance and computational cost among hyperbolic models and consistently surpasses strong Euclidean baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the Intrinsic Lorentz Neural Network (ILNN), which is a hyperbolic NN architecture that keeps all computations inside the Lorentz space instead of mixing Euclidean operations with manifold operations. They propose point-to-hyperplane Lorentz fully connected (PLFC) that computes the space-dimension of the output as signed distance to hyperplanes, and GyroLBN that re-centers and scales the output with efficient statistics. They show improved in CIFAR datasets and genomics datasets, with ablations that show effectiveness of PLFC and GyroLBN against LBN, GyroBN, and LFC.

### Strengths
1. The paper demonstrates substantial improvements over HCNN and Euclidean CNN in genomics tasks
2. The visualization in Figure 2/3 shows better separated clusters than HCNN
3. Using intrinsic hyperbolic operations instead of relying implicitly on Euclidean operations is an interesting direction that could result in more principled model designs

### Weaknesses
1. The authors claimed that LFC "applies Euclidean mappings to spatial components while treating the time-like coordinate separately" on line 58, which isn't correct. Fully hyperbolic linear layers proposed in prior works (https://arxiv.org/abs/2303.15919, https://arxiv.org/abs/2105.14686, https://arxiv.org/abs/2407.01290) operates on the entire Lorentz vector and the compute the time dimension. This brings into question of whether the authors correctly implemented the baselines for their subsequent experiments. 
2. The improvement in vision tasks is incredibly marginal in both the main results and the ablations
3. Because the improvement in CIFAR datasets are marginal, the performance of the proposed method is essentially entirely dependent on the genomic dataset. 
4. Some of the citations are repeated, which causes confusion

### Questions
1. Does the improvement brought by intrinsic Lorentz operations sustain beyond genomic data, since the improvement on the CIFAR datasets is marginal?
2. Could the authors clarify on discrepancy between their definition of LFC and the corresponding papers (as in weakness 1)?
3. One property of LFC is that it encompasses all Lorentz transformations. Can similar properties be proved for PLFC?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the intrinsic Lorentz neural network (ILNN), which is claimed to fully operate within the Lorentz model. The main architecture is a point-to-hyperplane Lorentz fully connected layer, along with which the paper introduces Lorentz versions of batch normalization (GyroLBN) and dropout, etc. Experiments on CIFAR-10/100, TEB, and GUE datasets show that ILNN achieves competitive results. Ablation studies are also reported to confirm the contributions of PLFC and GryoLBN.

### Strengths
1. The paper addresses an important topic of building neural architecture that respects hyperbolic geometry while avoiding the inconsistent mixture of Euclidean and hyperbolic components.
2. The experiments on vision and genomics tasks show potential interdisciplinary applications of hyperbolic neural networks.
3. On TEB and GUE datasets, ILNN achieves significant improvement on several tasks.
4. The paper is clearly structured, with detailed mathematical formulas and reasonable ablation studies.

### Weaknesses
1. The paper claims that its method is “Intrinsic Lorentz”, yet provides no rigorous definition of what that means. If “intrinsic” means depending only on manifold operations, then computing directly in Minkowski coordinates is still extrinsic.
2. On CIFAR-10/100, the gain over Euclidean model is very limited (< 0.5%), and most of the selected baseline models are not even as good as the standard Euclidean model…
3. On TEB and GUE, there should be more baseline methods (some from what is reviewed on page 2) than only Euclidean CNN and HCNN. Maybe other tasks of TEB and GUE can be included as well. Actually, other than using CIFAR, TEB and GUE, why not use some datasets that have been widely used and clearly have hierarchical structures?
4. The use of the “gyro-” terminology is conceptually problematic. The review of gyrovector spaces on page 3 is inaccurate: equations (1) and (2) describe standard Riemannian composition rules using exponential and logarithmic maps, not Ungar’s gyrovector algebra. The Lorentz model does not form a gyrovector space in the algebraic sense, unlike the Poincaré ball.
5. Theorem 1 mixes a definition (the formulation of the PLFC layer) and a theorem statement. These should be clearly separated for readability and mathematical rigor.
6. Some reference items appear multiple times. For instance, the paper by Bdeir appeared three times, but they are actually the same one. This is not the only entry. Please carefully check it.

### Questions
1. How is “intrinsic” formally defined? How are existing Lorentz-model methods not intrinsic? Why is “intrinsic Lorentz” beneficial? These have to be defined carefully and mathematically.
2. The result of Theorem 1 is of course interesting. But can you formulate a theorem to show the benefit of your specific design, or benefit of using an intrinsic Lorentz design?
3. Can you introduce some stronger baselines for TEB and GUE? 
4. Can you test your model on some better-known datasets where hyperbolic space has proved effective?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Intrinsic Lorentz Neural Network which is a fully intrinsic hyperbolic architecture that conducts all computations within the Lorentz model. The paper introduces several intrinsic modules including a novel point-to-hyperplane fully connected layer and GyroLBN. Several other components including a gyroadditive bias for the FC output, a Lorentz patch-concatenation operator and a Lorentz dropout layer are also introduced. Extensive experiments conducted on CIFAR-10/100 and two genomic benchmarks (TEB and GUE) illustrate that ILNN achieves state-of-the-art performance.

### Strengths
1. The proposed Intrinsic Lorentz Neural Network is well motivated as existing hyperbolic architectures remain partially intrinsic, mixing Euclidean operations with hyperbolic ones or relying on extrinsic parameterizations.

2. The proposed point-to-hyperplane Lorentz fully connected layer is new in the literature of hyperbolic neural network which replaces traditional affine transformations with intrinsic hyperbolic distance.

3. The paper is also generally well-written and extensive experimental results show the improvements of the proposed Intrinsic Lorentz Neural Network.

### Weaknesses
1. The experiments are only conducted on CIFAR-10/100  and two genomic benchmarks, it would be interesting to show if the proposed Intrinsic Lorentz Neural Network can outperform existing hyperbolic neural networks on large-scale datasets such as ImageNet and datasets with long-tail class distributions. 

2. The authors argue that intrinsic designs are preferable compared with partially intrinsic or extrinsic parameterizations, but there is a lack of enough evidence showing that the claim is true. There are quantitative results but the performance of ILNN is very similar to that of Hybrid Lorentz. Therefore, it is not clear if such intrinsic designs can bring significant benefits. 

3. In table 3, the authors analyze training efficiency to disentangle the effects of the classifier design and the normalization strategy, but there is no comparison of training time of the proposed Intrinsic Lorentz Neural Network with other hyperbolic neural networks mentioned in table 1.

### Questions
Besides the weaknesses, I have the following additional questions,

1. The authors use CIFAR-10 and CIFAR-100 in the experiments to demonstrate the benefits of the intrinsic approach. However, it is unclear what kind of hierarchical structure these two datasets possess. Therefore, it is difficult to conclude that the intrinsic approach preserves the geometry of the Lorentz model without distortion. Could the authors consider using datasets with explicit hierarchical structures to better support their claim?

2. The authors mentioned fully connected layer and convolutional layer, I am wondering if the same intrinsic approach can be applied to build other kind of neural network layers such as attention layers？

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes to learn representation and classification with all computations in the Lorentz model for hierarchical and long-tail data using point-to-hyperplane layers and intrinsic normalization, tested on image and genomic datasets.

### Strengths
This paper's main contribution proposes a new point-to-hyperplane fully connected layer (PLFC) that utilizes intrinsic Lorentzian distance for logits to match the data's latent hierarchy. However, this approach is not originally developed by the authors and lacks novelty. Furthermore, the mechanism by which this method achieves matching with the data's latent hierarchy remains unclear and insufficiently explained.

### Weaknesses
(1) The paper claims to focus on real-world data with latent hierarchical structure and long-tail distribution. However, I did not find clear evidence that CIFAR-10, CIFAR-100, or the TEB and GUE genomics benchmarks possess such properties.

(2) The PLFC method referenced by the authors as originating from HNN++ had been previously proposed in works on fully hyperbolic neural networks and the hypformer paper ("Hypformer: Exploring Efficient Transformer Fully in Hyperbolic Space"), yet the authors do not cite or compare with these relevant methods.

(3) Most other modules in the paper appear to be adopted from existing literature; I do not observe any genuine innovation.

(4) It remains unclear to me how the proposed method specifically addresses latent hierarchical structure and long-tail distribution in the data.

### Questions
In the datasets mentioned, where exactly do you observe the latent hierarchical structure and long-tail distribution? Could you explain why hyperbolic methods are effective for addressing latent hierarchical structures and long-tail distributions? Additionally, what specifically constitutes the hierarchical structure in this context?

### Soundness
2

### Presentation
3

### Contribution
2
