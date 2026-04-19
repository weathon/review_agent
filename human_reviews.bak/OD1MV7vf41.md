# Deep Random Features for Scalable Interpolation of Spatiotemporal Data

- Decision: Accept (Poster)
- Scores: 8, 3, 8

## Abstract
The rapid growth of earth observation systems calls for a scalable approach to interpolate remote-sensing observations. These methods in principle, should acquire more information about the observed field as data grows. Gaussian processes (GPs) are candidate model choices for interpolation. However, due to their poor scalability, they usually rely on inducing points for inference, which restricts their expressivity. Moreover, commonly imposed assumptions such as stationarity prevents them from capturing complex patterns in the data. While deep GPs can overcome this issue, training and making inference with them are difficult, again requiring crude approximations via inducing points. In this work, we instead approach the problem through Bayesian deep learning, where spatiotemporal fields are represented by deep neural networks, whose layers share the inductive bias of stationary GPs on the plane/sphere via random feature expansions. This allows one to (1) capture high frequency patterns in the data, and (2) use mini-batched gradient descent for large scale training. We experiment on various remote sensing data at local/global scales, showing that our approach produce competitive or superior results to existing methods, with well-calibrated uncertainties.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes to blend the strengths of neural networks (NNs) and Gaussian processes (GPs) by introducing the deep random features (DRF) framework, where the weights of the Fourier expansion $\boldsymbol{\Theta}$ are viewed as trainable parameters (similar to weights in NN) and the random features themselves are viewed as neurons. To perform interpolation on sphere, the authors also construct random feature maps based on a kernel constructed by a Mercer sum. The hyperparameters of the model are tuned either by comparing the evidence lower bound (ELBO) or by using a held-out validation set, depending on the choice of uncertainty quantification method. The authors test DRF on both synthetic and remote sensing datasets to show that DRF is able to learn both coarse-scale and fine-scale data patterns with well-calibrated uncertainty.

### Strengths
•	The authors present the methodology clearly, supplemented by informative visualizations, and provide a comprehensive discussion of the experimental results.

•	The proposed framework integrates the flexibility of NNs with the inductive bias of GPs, offering a scalable solution for large datasets.

•	Detecting high-frequency patterns is important in environmental data analysis, as such patterns are often closely associated with extreme events.

•	While there exists some amount of literature on constructing random features from kernel machines, the application of random features for interpolating spatial and temporal data is relatively underexplored and can be considered novel.

### Weaknesses
•	For clarity, it would be better to differentiate $H$ from $B$ in Section 3.1, as $H$ plays a similar role as $M$ in Section 2.2 which represents the dimension of the random features, while the bottleneck dimension $B$ is analogous to the hidden layer dimension in NNs.

•	A more comprehensive discussion of related work would enhance the paper’s contribution. For instance, [1] investigated representation learning through the approximation of kernels using random Fourier features. Additionally, [2] explored the connection between deep ensembles trained with squared-error loss and GP posteriors.

•	As noted in Section 2.2, random Fourier features are applicable only to stationary kernel functions. However, spatiotemporal data may exhibit non-stationary trends, which presents a potential limitation of the proposed method. A discussion on this point (e.g., extensions to handle non-stationarity) will be appreciated.

References:

[1]. Jiang, Z., Zheng, T., Liu, Y., & Carlson, D. (2022). Incorporating prior knowledge into neural networks through an implicit composite kernel. arXiv preprint arXiv:2205.07384.

[2]. He, B., Lakshminarayanan, B., & Teh, Y. W. (2020). Bayesian deep ensembles via the neural tangent kernel. Advances in neural information processing systems, 33, 1010-1022.

### Questions
•	Since $\boldsymbol{\omega} \sim p(\boldsymbol{\omega})$ and $p(\boldsymbol{\omega})$ is the normalized Fourier transform of the kernel function, can you elaborate the procedure to efficiently sample from $p(\boldsymbol{\omega})$?

•	Why does DRF with dropout have much higher computational time than an ensemble of DRFs?

•	Does it really make sense to compare with neural processes in Tables 1 and 2 as NPs are mainly used for meta-learning?

•	Are ground truth values available for the sea level anomaly measurements? If so, it would be beneficial to present the interpolation and uncertainty calibration errors to validate the importance of learning fine-scale fluctuations.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents a deep Gaussian process (GP) with deep random features for modeling spatiotemporal data where the inputs are spatial/temporal coordinates. Deep random features are utilized to address the high-frequency limitations of neural networks, and mini-batched gradient descent is used for large-scale training. Experiments are conducted on the synthetic data, satellite data, and sea level anomaly data.

### Strengths
- The paper addresses a clear and timely topic, focusing on coordinate-based neural representation for sparse observations. 
- Figures 1 and 2 effectively illustrate both the task and the model structure. 
- The theoretical background on random features is covered in sufficient depth.

### Weaknesses
- The model does not appear to outperform baselines in terms of accuracy or computation time. 
- Although the primary contribution is the application of random features, the paper does not cover advanced Fourier, Wavelet features, such as [1]. It would be helpful for the authors to consider incorporating these features or at least to discuss how their approach might be extended to include them. 
- The assumption of stationary kernel seems limit model performance, and further clarification on this assumption would be beneficial.
- It would be beneficial for the authors to include tasks for both time interpolation and extrapolation, like [2].
- The paper also overlooks many recent works that apply implicit neural representations for spatiotemporal data, which should be included for a more comprehensive comparison. For the existing works, temporal dimensions are often treated separately from spatial dimensions, typically using autoregressive time-stepping, which has shown efficiency in modeling physical spatiotemporal data. For example, [2], and [3] treat the spatial dimension continuously and employ neural ODEs to model time as a continuous variable. The author should justify why the proposed method has advantages over existing continuous spatial-temporal models. 

[1] Multiplicative Filter Networks, ICLR, 2021.

[2] Continuous PDE Dynamics Forecasting with Implicit Neural Representations, ICLR, 2023.

[3] Operator Learning with Neural Fields: Tackling PDEs on General Geometrics, Neurips, 2023.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a model leveraging deep random features (DRF) to achieve scalable interpolation for large-scale, high-dimensional spatiotemporal data. By integrating random Fourier feature layers with DNNs, the model retains the inductive biases of GPs, while with potential for scaling efficiently with the dataset size. The novelty lies primarily in adapting random feature layers to produce an architecture capable of high-resolution spatiotemporal interpolation with uncertainty quantification, moving beyond the limitations of traditional GPs and recent DGPs in this domain.

### Strengths
- The paper integrates multiple uncertainty quantification techniques (variational inference, dropout, deep ensembles), offering flexibility in obtaining uncertainty estimates for the model outputs.
- The architecture performs well across varied datasets, including synthetic, local, and global satellite measurements, and the spherical adaptation for global data fields demonstrates its versatility.

### Weaknesses
I don’t see any major concerns or weaknesses in this paper, but I believe it would benefit from some ablation studies, which are currently missing, to better illustrate the contributions of the proposed model. Please refer to the questions for details.

### Questions
- To me, the proposed method closely resembles the Fourier features network and SIREN in various aspects. Specifically, it feels like a generalization of Tancik et al.’s repeated Gaussian mapping followed by linear transformations, applied across layers rather than in just the input. If random features were only used in the first layer, would the network’s performance still mirror that of the Fourier features network? Exploring this could clarify if deeper layers with random features add unique value.
- The model also feels akin to a generalized SIREN, with deterministic parameter layers replaced by additional linear transformations and stochastic components. In this context, I'd appreciate a breakdown of the number of trainable parameters in this model versus SIREN and other baselines. This would help me interpret differences in the quantitative results presented in Tables 1 and 2.
- Since the model’s roots are claimed to be in deep Gaussian processes, I think it would be helpful to see an ablation study comparing deep GPs approximated with random features against this proposed model. This doesn’t need to be overly complex—a simple synthetic dataset might be sufficient to highlight the supposed benefits of deep random features combined with learnable linear layers.
- I find the skip connections interesting but somewhat confusing. From what I see, after the projection by $\phi$, the features are concatenated back to a space seemingly compatible with the original coordinates. However, I’m unsure if this compatibility holds without explicit regularization. An ablation study on removing or adjusting these skip connections would help clarify their impact, making it easier to interpret how well they function as intended.
- The paper highlights scalability as a major benefit, but it lacks empirical validation in terms of how the model performs with increasing data points or the number of random features. Scalability results as a function of data volume and feature count would substantiate claims about the model’s efficiency and approximation performance.
- Lastly, given the reliance on random sampling for the random Fourier features, I want to see the variability in the model’s performance. Not on a UQ sense but evaluating the model across multiple runs with varied random seeds.

### Soundness
3

### Presentation
4

### Contribution
3
