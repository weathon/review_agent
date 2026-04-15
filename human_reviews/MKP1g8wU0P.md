# Spectral-Refiner: Accurate Fine-Tuning of Spatiotemporal Fourier Neural Operator for Turbulent Flows

- Decision: Accept (Poster)
- Scores: 8, 5, 6, 5

## Abstract
Recent advancements in operator-type neural networks have shown promising results in approximating the solutions of spatiotemporal Partial Differential Equations (PDEs). However, these neural networks often entail considerable training expenses, and may not always achieve the desired accuracy required in many scientific and engineering disciplines. In this paper, we propose a new learning framework to address these issues. A new spatiotemporal adaptation is proposed to generalize any Fourier Neural Operator (FNO) variant to learn maps between Bochner spaces, which can perform an arbitrary-length temporal super-resolution for the first time. To better exploit this capacity, a new paradigm is proposed to refine the commonly adopted end-to-end neural operator training and evaluations with the help from the wisdom from traditional numerical PDE theory and techniques. Specifically, in the learning problems for the turbulent flow modeled by the Navier-Stokes Equations (NSE), the proposed paradigm trains an FNO only for a few epochs. Then, only the newly proposed spatiotemporal spectral convolution layer is fine-tuned without the frequency truncation. The spectral fine-tuning loss function uses a negative Sobolev norm for the first time in operator learning, defined through a reliable functional-type a posteriori error estimator whose evaluation is exact thanks to the Parseval identity. Moreover, unlike the difficult nonconvex optimization problems in the end-to-end training, this fine-tuning loss is convex. Numerical experiments on commonly used NSE benchmarks demonstrate significant improvements in both computational efficiency and accuracy, compared to end-to-end evaluation and traditional numerical PDE solvers under certain conditions. The source code is publicly available at https://github.com/scaomath/torch-cfd.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper according to me addresses limitations in current neural operator approaches for modeling turbulent flows governed by the Navier-Stokes equations. The key contributions are:
1. A new spatiotemporal adaptation (ST-FNO) of Fourier Neural Operators to enable learning maps between Bochner spaces, allowing arbitrary-length temporal predictions.
2. A novel training-fine-tuning paradigm that combines limited epochs of end-to-end training with targeted fine-tuning of a spectral convolution layer.
3. A new loss function for fine-tuning based on a functional-type a posteriori error estimator using a negative Sobolev norm, which is reliably evaluated through Parseval's identity.
4. Empirical demonstration of significant improvements in both computational efficiency and accuracy compared to existing methods.
Overall I like the paper! It's important for a lot of problems that require FNO to predict multiple temporal steps, especially in high Reynolds number ranges for NS.

### Strengths
1. Theoretical foundation: I like that the paper provides a rigorous mathematical analysis of discretization mismatch errors and proves the reliability of the proposed error estimator.
2. Novel architecture: The ST-FNO adaptation enables flexible spatiotemporal predictions, addressing a key limitation of existing neural operators. This is extremely useful in various timestepping problems.
3. Efficient training paradigm: The proposed approach of limited training followed by targeted fine-tuning offers a computationally efficient alternative to standard end-to-end training.
4. Strong empirical results: The method demonstrates significant improvements in accuracy and efficiency on challenging turbulent flow benchmarks.
5. Open-source implementation: The authors provide code to reproduce their results, enhancing reproducibility and potential impact.

### Weaknesses
1. Limited scope: The method is primarily designed for rectangular domains and uniform grids, which may limit applicability to more complex geometries.
2. Dependency on numerical solvers: The fine-tuning process relies on traditional numerical solvers for computing extra field variables, which may introduce computational overhead.
3. Assumption sensitivity: The theoretical guarantees rely on certain assumptions about the closeness of solutions, which may not always hold in practice, I'm not sure how much it does in practice.
4. Long-term stability analysis: While short-term performance is demonstrated, a more extensive investigation of long-term stability for time-dependent problems would be valuable. This could include chaos indicators or Lyapunov exponent analysis for longer time horizons, particularly for highly turbulent regimes.

### Questions
Just some questions:
1. Computational efficiency trade-offs: Can you provide a more detailed comparison of the computational costs of your method (including the fine-tuning phase) versus traditional CFD methods and pure neural network approaches? Specifically, how does the accuracy-to-compute-time ratio compare across these methods for different problem scales?
2. Error accumulation in multi-step predictions: For multi-step predictions, how does the error accumulate over time compared to traditional numerical methods? Is there a point at which the accuracy degrades significantly, and if so, how might this be mitigated?
3. Spectral layer fine-tuning: You mention fine-tuning only the last spectral convolution layer. Have you experimented with fine-tuning other layers or using a different architecture for this layer? How sensitive is the method to the design of this final layer?
4. Arbitrary-length temporal predictions: Can you elaborate on how your ST-FNO handles arbitrary-length temporal predictions? Specifically, how does the performance change as the prediction length increases beyond what was used in training? Are there any limitations on how far into the future the model can reliably predict?
5. Handling of boundary conditions: How does your method handle different types of boundary conditions, especially for the vorticity-streamfunction formulation? Can it adapt to changing boundary conditions without retraining?
6. Temporal super-resolution: Can your method perform temporal super-resolution, i.e., predict at a higher temporal frequency than the training data? If so, how accurate are these interpolated predictions?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper presents a novel learning framework that enhances the capabilities of operator-type neural networks, specifically focusing on Fourier Neural Operators (FNOs) for solving spatiotemporal Partial Differential Equations (PDEs). Recognizing the high training costs and variable accuracy of existing models, the authors introduce a spatiotemporal adaptation that allows FNOs to learn mappings between Bochner spaces, enabling arbitrary-length temporal super-resolution. The framework integrates insights from traditional numerical PDE techniques to refine the conventional end-to-end training process. For turbulent flow modeling with the Navier-Stokes Equations (NSE), the approach involves initial training of the FNO for a limited number of epochs, followed by fine-tuning a new spatiotemporal spectral convolution layer without frequency truncation. A unique fine-tuning loss function using a negative Sobolev norm, defined through a functional a posteriori error estimator based on the Parseval identity, is introduced. This loss function simplifies the optimization process, as it is convex, contrasting with the nonconvex challenges typical of end-to-end training. The proposed method significantly improves both computational efficiency and accuracy in numerical experiments on standard NSE benchmarks, outperforming traditional numerical PDE solvers and end-to-end evaluations.

### Strengths
The paper introduces a novel learning framework that effectively combines traditional numerical PDE techniques with operator-type neural networks, enhancing the capabilities of Fourier Neural Operators (FNOs) in solving spatiotemporal PDEs. The ability to generalize FNO variants to learn maps between Bochner spaces and perform arbitrary-length temporal super-resolution is a significant advancement for dynamic environments. By proposing a training strategy that involves limited initial epochs followed by fine-tuning, the authors address the challenges of high training costs while maintaining performance. This approach can make training more accessible and efficient.
The introduction of a fine-tuning loss function that is convex simplifies the optimization process, reducing the complications often associated with nonconvex problems in neural network training. The framework demonstrates significant improvements in computational efficiency and accuracy through numerical experiments on commonly used Navier-Stokes Equation benchmarks, providing strong empirical support for the proposed methods.

### Weaknesses
As with many neural network-based approaches, the interpretability of the results may be challenging.

### Questions
Could you elaborate on the motivation for learning maps between Bochner spaces? How does this choice influence the model’s performance and its ability to handle temporal super-resolution?

Given the model’s complexity, how interpretable are the results?

How sensitive is the model to hyperparameter choices, especially those related to the spatiotemporal spectral convolution layer? Are there guidelines for selecting optimal hyperparameters for new applications?

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
This paper introduces the spatiotemporal adaptation technique for all FNO variants (ST-FNO), enabling them to learn mappings between Bochner spaces. The authors propose a novel strategy for training and evaluating ST-FNOs, which surpasses existing methods in both speed and accuracy. Numerical experiments on Navier-Stokes Equations (NSE) benchmarks demonstrate substantial improvements in computational efficiency and accuracy.

### Strengths
- The concepts of the spatiotemporal adaptation technique and the hybrid operator learning paradigm appear novel.
- The method is solidly supported by theoretical evidence.
- Overall, the paper is well-written and clear.

### Weaknesses
- The authors assert that their methods "yield accuracy comparable to traditional numerical methods, yet with computational resources akin to evaluating NOs." This claim would indeed be exciting if validated. However, the paper's experiments appear insufficient in the following aspects:

  - For both FNO and ST-FNO, while they may perform well on the training data distribution, they likely fail when tested with out-of-distribution data. In extreme out-of-distribution scenarios, can ST-FNO still maintain accuracy comparable to traditional numerical methods within a limited fine-tuning timeframe?

  - Table 4 compares the FLOPs of different methods, but the running times for each method are absent. I am concerned about whether the sum of inference and fine-tuning times for ST-FNO is comparable to the inference time for FNO alone.

### Questions
Can ST-FNO be applied to other equations and still perform effectively?

In line 65, the phrase “Similar to the traditional time marching solvers” is confusing. Did the authors intend to mean "Unlike traditional time marching solvers"?

In lines 217-218, what is the definition of $q$?

In lines 1678-1679, there is a '??'. Please correct this.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a new learning framework to improve operator-type neural networks for solving spatiotemporal PDEs. The proposed spatiotemporal adaptation generalizes FNO to enable temporal super-resolution. By refining traditional end-to-end training with insights from numerical PDE theory, the framework trains an FNO briefly and then fine-tunes a new spatiotemporal spectral convolution layer without frequency truncation, using a novel negative Sobolev norm loss function.

### Strengths
1.The ST-FNO extends FNO capabilities to handle arbitrary temporal and spatial resolutions, improving flexibility and applicability for complex PDEs like NS equations.

2.By designing the ST-FNO as a zero-shot model with arbitrary-length temporal inference, the Spectral-Refiner can adapt flexibly to varying time horizons, making it well-suited for large-scale or long-term simulations without significant retraining.

### Weaknesses
1. Although this paper claims that Spectral-Refiner outperforms traditional numerical methods in accuracy and computational efficiency, it does not specify how the method performs under different resolutions, time steps, and initial conditions.

2. Can the model generalize to varying conditions, such as changes in Reynolds numbers or external forces? If so, please provide correlation curves and energy spectrum curves under these conditions.

3. Model performance should not be evaluated solely based on the final frame; an error propagation curve and error distribution plot are needed to illustrate the model’s overall performance.

### Questions
1. The paper notes a spatial grid size of 256×256 and a prediction time interval from \( t = 4.5 \) to \( t = 5.5 \), which is quite short for meaningful evaluation. This limited time horizon may not fully showcase the model’s performance, and longer predictions should be considered for comparison. Could it surpass top-performing models such as PDERefiner, TSM, and LI in terms of prediction accuracy?

2. While this method targets turbulent modeling of Navier-Stokes equations, its applicability to other nonlinear or multiphysics PDEs is not discussed. Could Spectral-Refiner be adapted to handle other PDEs, such as heat conduction or elasticity? What modifications or enhancements would be required?

3. The paper uses the H⁻¹ Sobolev norm as a tool for error estimation, yet traditional numerical analysis often questions the suitability of non-local norms. Without access to exact solutions, is this norm sufficiently stable? How well does this error estimation perform with initial conditions that lack global smoothness?

4. This method is suggested to be more efficient and stable than physics-informed neural operators , but the comparative experiments are limited. Could more comprehensive experimental data, including comparisons with popular neural operators like DeepONet, be provided?

5. The paper notes the removal of frequency truncation during fine-tuning, which may enhance the model’s ability to capture low-frequency information. However, does this increase the risk of amplifying high-frequency noise? If so, what measures could control this issue without compromising accuracy?

6. The method involves high-order Fourier transforms and complex error estimations, potentially adding significant computational cost. In practical applications, especially those requiring real-time simulation, can this method meet timing constraints? How does its inference time compare with other methods?


7. How different are the training and testing initial conditions?

### Soundness
3

### Presentation
3

### Contribution
3
