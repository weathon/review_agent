# Memory-Augmented Functional Koopmanism for Interpretable Learning of Spatiotemporal Dynamics

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Precise prediction of spatiotemporal dynamics over predictive horizons is constrained by the computational cost of high-fidelity solvers and the sparsity, noise, and irregularity of data. We introduce MERLIN, a Koopman-based framework that lifts dynamics to the evolution of learned *observation functionals* with near-linear progression, enabling full-field reconstruction at arbitrary resolutions. Theoretically, we develop a functional Koopman theory for PDEs and compensate for the loss of finite-dimensional linear invariance via the Mori–Zwanzig formalism, which augments the linear backbone with non-Markovian memory terms to improve predictive accuracy. Practically, MERLIN employs discretization-invariant *function encoders* that map partial, irregular observations to observables, and resolution-free *function decoders* that reconstruct states at arbitrary query points. Training under linear constraints yields an interpretable, low-dimensional model that captures principal modes, supports reduced-order modeling, and—augmented with memory correction—delivers stable long-horizon rollouts even in ultra-low-dimensional latent spaces.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents MERLIN, a novel framework for learning long-term spatiotemporal dynamics from random, partial, and irregular observations of PDE-governed systems. The method combines the Koopman operator theory with the Mori-Zwanzig formalism. The key idea is to learn a set of observation functionals that map the infinite-dimensional state to a low-dimensional latent space where the dynamics are approximately linear. The architecture features a discretization-invariant encoder (based on Galerkin Transformer) and a resolution-free decoder (based on FourierNet), allowing it to handle arbitrary sensor layouts and query predictions at any location. Extensive experiments on Wave, Navier-Stokes, and Sea Surface Temperature datasets demonstrate MERLIN's strong performance in long-term prediction, robustness to masked observations, and capability for interpretable reduced-order modeling, often outperforming strong baselines like FNO and DeepONet.

### Strengths
1.The paper provides a rigorous theoretical foundation by extending Koopman theory to infinite-dimensional function spaces (observation *functionals*) and formally integrating the Mori-Zwanzig formalism to justify the memory correction. This is a significant contribution.

2.The design of the discretization-invariant encoder and resolution-free decoder is a powerful combination. It naturally handles the challenging and practical setting of random, partial observations, which many neural operators struggle with.

3.The two-phase training clearly separates the linear backbone (interpretable as Koopman eigenfunctionals) from the memory correction. This provides insight into the learned dynamics, as shown in the eigenvalue spectra and energy contribution plots.

### Weaknesses
1.While the paper highlights the architectural advantages, it lacks a discussion on the computational cost and training/inference time compared to other baselines, especially efficient operators like FNO. The complexity of the two-phase training and the Transformer-based encoder could be a practical concern.

2.The paper proposes two memory models (leaky and finite/LSTM) but does not provide a clear comparison or guidance on when to choose one over the other. An ablation study on their respective performance and trade-offs is missing.

3.The paper notes that the FourierNet decoder might introduce a "mild spectral bias," potentially limiting in-horizon fitting compared to FNO. This point is acknowledged but not deeply investigated. The impact of this bias on different types of PDEs could be explored further.

### Questions
1.The theoretical derivation of the Generalized Langevin Equation (GLE) in Appendix B.2 is presented in the context of a finite set of observables `{g_i}`. How does this theoretically extend to the *learned, nonlinear* observation functionals parameterized by the neural network encoder, which are not guaranteed to be linearly independent or form a basis for an invariant subspace

2.The two-phase training strategy is a core component of the method. Did you experiment with an end-to-end joint training of the linear backbone and the memory term? If so, what were the results, and what challenges led to the choice of a sequential strategy?

3.You propose two memory models: "leaky memory" and "finite memory" (LSTM). Could you provide more insight or ablation studies on the comparative advantages, disadvantages, and suitable application scenarios for each? Are there specific dynamical regimes where one significantly outperforms the other?

4.The encoder uses a Galerkin Transformer with a fixed number of `[CLS]` tokens to achieve discretization invariance. How sensitive is the model's performance to the number of these `[CLS]` tokens (`K`) and the final latent dimension (`D`)? Was any architecture search performed for these key hyperparameters?

5.In the reduced-order modeling (ROM) experiments, the projection head `U` is constrained to the Stiefel manifold. What is the empirical justification for this choice compared to a simple linear layer without orthogonality constraints? Does this constraint ever lead to training difficulties or limit representational capacity?

6.The paper mentions that FNO has an advantage on synthetic PDEs within the training horizon due to operating directly in the Fourier domain. Could you discuss the potential of incorporating a similar spectral bias directly into the MERLIN decoder or latent dynamics to improve short-term accuracy without sacrificing its flexibility?

### Soundness
2

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
In this work, the authors demonstrate a way to use Koopman theory/framework to nicely model spatiotemporal systems with (i) interpretability, (ii) robustness, and (iii) random sensor placement (mesh-free, resolution-free). The authors were further inspired by the Mori-Zwanzig theory that when the sub/latent space is not closed, the rest component can go into a memory term. In the experiments, the authors evaluated the temporal prediction performance of MERLIN and compared MERLIN to many recent state-of-the-art methods.

### Strengths
This paper is clearly written and well organized. The abstract and introduction paragraphs are well-written. 

This work presents a solid method. This method is (i) robust for practical applications and (ii) has interpretability for all components. Because Koopman theory presents a valid background to model PDEs robustly (by linearizing the dynamics). This is one of the best solutions and has huge potential, if one wants to have a generative model that is robust (especially under domain shift/during extrapolation). 

The reviewer is totally aligned with the motivation, methodology, and general design of this paper and MERLIN. So the reviewer will just jump directly to the weaknesses section.

### Weaknesses
Novelty, the presentation of the theory part, and some statements on the experiments. If one of those can be addressed effectively (especially the novelty), with a good plan of action, the reviewer would be likely to improve the rating.

### Questions
1. Although finding the concept attractive, the reviewer has to question the novelty. The authors seem to combine many techniques together, which makes the paper a bit overpacked. For example, the resolution-free part is indeed very important, but it seems to be orthogonal to the Koopman learning part. This affects the novelty of this paper, and the authors may need to put in significant effort to run ablation studies to really convince the audience that the design is optimal, or the combination is valid (even though they are useful). The reviewer would ask the authors to clarify this point, and provide a plan to discuss what is really novel here. 

2. The reviewer would love the authors to talk about limitations in theory. For example, wouldn't MERLIN require the dynamics can be approximately spanned by some learnable basis? There is definitely improvement here - but it is better to clarify. The current statement is a bit working like stating we don't have to worry anything (but it's always better to talk about the limits), and the way of presentation of the Theorem may confuse people that the authors would re-show the Mori-Zwanzig theory. It is better to claim what is new here. 

3. The experiments section is nice. And things are mostly well-tested with great performance. Good work. 
But the reviewer would like to ask a few questions that
3.1 Could the authors elaborate more on why FNO can outperform MERLIN? The comparison is kind of unfair for MERLIN though (since MERLIN can potentially have more functionalities). 
3.2 Could the authors provide a statement here in the rebuttal (don't have to be in the paper) that would MERLIN also have the challenge that it might over-memorize (overfit) the data? How generalize could the memory unit help? For exmaple, if everything is trained with low Reynolds number, can we extrapolate for high Reynolds number?
3.3 No need to have further experiments just some comments would be great: but could MERLIN work on complex datasets like more chaotic turbulent flow (e.g. 2D snapshot of isotropic turbulent)? It would be a great idea if the authors can mention more in the main text that MERLIN can learn with relatively small number of training trails (500-1000), and it could generalize to new settings (within a certain range of changing). 

4. The reviewer has a few suggested references that explored relevant contexts to Koopman and delay embeddings. 

- Mauroy, Alexandre. "Koopman operator framework for spectral analysis and identification of infinite-dimensional systems." Mathematics 9.19 (2021): 2495.

- Jin, Yuhong, et al. "Invertible Koopman neural operator for data-driven modeling of partial differential equations." arXiv preprint arXiv:2503.19717 (2025).

- Gao, M. L., Williams, J. P., & Kutz, J. N. (2025). Sparse identification of nonlinear dynamics and Koopman operators with Shallow Recurrent Decoder Networks. arXiv preprint arXiv:2501.13329.

- Buzhardt, Jake, C. Ricardo Constante-Amores, and Michael D. Graham. "On the relationship between Koopman operator approximations and neural ordinary differential equations for data-driven time-evolution predictions." Chaos: An Interdisciplinary Journal of Nonlinear Science 35.4 (2025).

- Budišić, Marko, Ryan Mohr, and Igor Mezić. "Applied koopmanism." Chaos: An Interdisciplinary Journal of Nonlinear Science 22.4 (2012).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a Koopman-based framework for learning interpretable spatiotemporal dynamics. The proposed method combines discretization-invariant encoder and decoder for spatial reconstruction, as well as a memory-augmented Koopman backbone for capturing dynamics. The authors have tested the model performance using 2 synthetic datasets and 1 real-world dataset. The empirical results have shown that the proposed method excels in real-world datasets and also presented best performance in long-term predictions.

### Strengths
- This paper presents a Koopman-based method for predicting spatiotemporal dynamics. It improves the explainability of the proposed method. 

- The methodology part is easy to follow. This paper is generally well presented. 

- The method is evaluated on both the synthetic and real-world datasets.

### Weaknesses
- The proposed framework is quite common in scientific machine learning. Many prior works use the encoder - latent dynamics - decoder pipeline. The latent dynamics are modeled by using the Koopman operator [1], Neural ODEs [2], or spectral methods [3]. Also, the reduced-order modeling combined with recurrent neural networks is observed in this SHRED work [4]. 


- This paper may benefit from including an analysis of scalability to higher-dimensional PDEs and the computational overhead. For example, the authors might consider comparing the runtime and memory across baseline models and the proposed method. 

- It would be better if the authors could stress-test the model performance on some more challenging datasets, such as ERA5 [2,5].

---

**Refs:**

[1] Azencot, Omri, et al. "Forecasting sequential data using consistent Koopman autoencoders." International Conference on Machine Learning. PMLR, 2020.

[2] Song, Jialin, et al. "Forecasting high-dimensional spatio-temporal systems from sparse measurements." Machine Learning: Science and Technology 5.4 (2024): 045067.

[3] Wu, Haixu, et al. "Solving high-dimensional pdes with latent spectral models." arXiv preprint arXiv:2301.12664 (2023).

[4] Kutz, J. Nathan, et al. "Shallow recurrent decoder for reduced order modeling of plasma dynamics." arXiv preprint arXiv:2405.11955 (2024).

[5] Xing, Lanxiang, et al. "Helmfluid: Learning helmholtz dynamics for interpretable fluid prediction." arXiv preprint arXiv:2310.10565 (2023).

### Questions
- On Page 6, line 323, “... an another ...” seems not correct. 

- Could you include more discussions on why the proposed method achieves better performance on the SST dataset compared to the two synthetic PDE benchmarks? What properties of the SST data or model configuration contribute to this difference?

- Could you discuss the procedure used to select hyperparameters in the proposed framework, particularly those associated with the delay embedding? Also, how sensitive is the model’s performance and stability to these choices?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The approach introduces a Koopman-based framework that enables accurate, resolution-free prediction and reconstruction of spatiotemporal dynamics from sparse and irregular data. By training discretization-invariant encoders and decoders under linear constraints, the proposed approach achieves interpretable, low-dimensional, and stable long-horizon forecasts across two synthetic datasets and one real-world dataset.

### Strengths
The approach is mathematically rigorous and has practical utility in modelling temporal systems like turbulent fluid mechanics, etc., efficiently. The approach is tested on both synthetic datasets, in conjunction with real-world data.

### Weaknesses
Although the approach is mathematically interesting, it underperforms on the test set relative to FNO on both synthetic datasets, and only marginally outperforms KNO on the real-world setting. More robust benchmarking is required to demonstrate the practical utility of the proposed framework.

### Questions
- Can the approach be tested on the same problems using the same setups as those presented in the FNO/KNO/UNet work. These approaches have done significant hyperparameter tuning to determine the optimal conditions for these methods. To ensure a fair comparison, MERLIN should be contrasted with these methods under these readily-established conditions. Studies have demonstrated that UNets outperform FNO for these problems, but even that result appears not to be recovered in the test error. 
- To demonstrate the quality of the representations learned by the model, can alternative tasks be considered beyond temporal prediction? E.g., inverse problems (recovering the initial conditions)? 
- Multiple different models should be trained across various different seeds to demonstrate uncertainty on observed results. 
- Additional studies on other complex synthetic PDE tasks for PDEArena, etc., would be useful - e.g., KS, or KdV. 
- It would be worth including some of the finite-dimensional benchmarks in the method comparison, e.g., the results from Lusch et al., etc.

### Soundness
3

### Presentation
2

### Contribution
2
