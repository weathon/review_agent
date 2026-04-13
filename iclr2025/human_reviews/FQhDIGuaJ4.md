## Human Reviewer 1

### Summary
The authors propose the Wavelet Diffusion Neural Operator , a data-driven framework designed for PDE simulation and control. WDNO uses diffusion-based generative modeling within the wavelet domain, which is claimed to capture entire trajectories of the system dynamics. To address the  challenge of generalizing across different resolutions multi-resolution training approach is introduced, enabling WDNO to work across varying data scales. This framework represents a tool for precise, adaptable modeling in complex, multi-scale systems.

### Strengths
The paper presents an interesting and rich contribution to the field and offers a new approach through the Wavelet Diffusion Neural Operator. This work is both technically rigorous and well-articulated, making the methodology and results accessible to the reader. The introduction of wavelets is notable, as it appears to deliver the claimed advantages in handling multi-scale features, abrupt changes, and long-term dependencies in PDE-based simulations. These claims are convincingly supported by a number of experiments, which effectively demonstrate the impact of the wavelet-based approach.

### Weaknesses
The structure of the work can be challenging to follow, partially due to the number of innovations introduced in quick succession, which may benefit from clearer organization or further segmentation for clarity. 

The experimental evaluation, while well-executed and featuring comparisons to other methods, is somewhat limited in the number of test cases. Expanding the range of test cases would enhance the generalizability of the results:
1. An analysis of the error distribution over time would be valuable, as it could reveal any systematic errors or trends in performance decay over extended simulations.
2. Lack of a real-world test case. Please include a real-world test case to provide further validation of the approach in practical settings:  a dataset like PTB-XL should be addressable with the proposed method and could offer relevant, real-world complexity.
3. Additional 2D cases across a broader range of complex spatial-temporal domains would strengthen the  work. [Iakolev et al, Latent neural ODEs with sparse bayesian multiple shooting, ICLR 2023, Lagemann et al, Invariance-based Learning of Latent Dynamics, ICLR 2024] are presenting results for 2D PDE-based test cases. It makes also sense to compare WDNO against these neural ODE based approaches for the simulation part. 

Figures 1 and 2 could be improved for clarity and relevance. 
In my opinion Figure 1 lacks meaningful content and could be more effectively repurposed. The caption offers minimal context, leaving the figure’s purpose unclear, especially as the main text already conveys WDNO’s super-resolution capabilities. Reallocating this space to highlight other aspects of the method might provide more value.
Figure 2 is informative and well-designed but would benefit from an expanded caption to guide readers through its details, despite space constraints. I would suggest a more general, high-level version of this figure in the main text and the current version of Figure 2 with a detailed caption in the Appendix. This would offer both an accessible overview and a richer, in-depth explanation in the Appendix.
A detailed explanation tracing the steps from input data to output data of the pipeline and the transformations involved in each stage, with explicit dimensions based on one of the datasets would be great and improve the clarity. 

In my opinion, eq. 9 and 10 appear not important in the main discussion and could be more appropriately relocated to the corresponding Appendix (E, F, G). Instead, a general paragraph on data preparation in the main text would provide readers with a clearer understanding of the processes involved, which is currently lacking.

Additionally, the work presented in Appendix C is noteworthy and should at least be mentioned briefly in the main text. 

Please reorganize the Appendix to ensure that tables are placed immediately after they are referenced. As there is no page limit for the Appendix, there is no need to conserve space, which would increase the readability. 

Please add an arrow indicating the direction of time on the time axis in Figure 4.

### Questions
Line 194, “ we combine the use of both guidance methods” Where is the classifier based guidance method used?

What is the impact of measurement noise? An ablation study for datasets with increasing measurement noise would be great.

How many trajectories are required? Can you show the impact of the available number of trajectories on the performance?

What happens if the dynamical changes due to interventions/perturbation upon the system? Is the learned model still useful?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
The standard diffusion models cannot learn dynamics with abrupt changes. This paper proposes a new method, Wavelet Diffusion Neural Operator (WDNO), to solve this issue. It combines wavelet transform and diffusion models in the context of learning PDEs from the perspective of neural operators. This method also considers multi-resolution training. Multiple datasets have been tested to evaluate the model performance compared to the baseline models.

### Strengths
- This paper aims to solve an interesting problem regarding the abrupt changes in spatiotemporal dynamics. 

- This paper has shown comprehensive details of the experimental setup, model details, and results. 

- This paper is easy to follow.

### Weaknesses
- This paper shows incremental novelty. It uses diffusion models in the wavelet space. But that is not a big issue as long as the model performance is good. 

- The motivation for using diffusion models for PDE learning is not well established. This paper mentions that diffusion models can better capture long-term predictions and model high-dimensional systems. However, many deep learning-based models can do both, such as Fourier neural operators. How do diffusion models differentiate from other deep learning-based models? Also, can you provide a reference paper that shows diffusion models can do a better job of capturing long-term predictions?

- I have some concerns about the datasets used in this paper. 1D Burgers with a viscosity of 0.01 is not particularly challenging, though it presents shock wave phenomena. The standard physics-informed neural networks [1] are also capable of handling such tasks. The authors may consider some other challenging datasets, such as ERA5 [2], since the proposed method can deal with long-term predictions. Second, it would also be interesting to see the model performance on the dynamics without abrupt changes. I think that will also give the audience a broader sense of how this model works. The authors may consider some datasets in PDEBench [3].

- It would be good to have another ablation study on the DDPM + Fourier transform, which changes the wavelet transform in the proposed method. The additional experiment will be trained in Fourier space instead of wavelet space. It will further validate the effectiveness of using wavelet transform to capture local details. 

- The presentation for Section 2 Related Work can be improved. It would be better to have several paragraphs to introduce the related works from different perspectives.

**References:**

[1] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707. 

[2] Pathak, J., Subramanian, S., Harrington, P., Raja, S., Chattopadhyay, A., Mardani, M., ... & Anandkumar, A. (2022). Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators. arXiv preprint arXiv:2202.11214.

[3] Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., & Niepert, M. (2022). Pdebench: An extensive benchmark for scientific machine learning. Advances in Neural Information Processing Systems, 35, 1596-1611.

### Questions
- For the evaluation metrics, why does this paper only consider MSE? If this paper focuses on the dynamics with abrupt changes, then the Mean Absolute Error (MAE) or infinity norm should be considered. 

- Some minor typos: 

    - On Page 2, the paragraph name “wavelet domain” should be “Wavelet domain”. 

    - On Page 2,  it seems not common to see “contribute the following”.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper proposes a wavelet diffusion neural operator (WDNO), which performs diffusion-based generative modeling in the wavelet domain for the entire trajectory of time-dependent PDEs and multiresolution training to generalize across different resolutions.

### Strengths
1) Multi-resolution training

2) Diffusion in the wavelet domain to capture long-term dependencies and abrupt changes effectively.

### Weaknesses
1) The evaluation omits comparisons with state-of-the-art operators like Transolver, GNOT, LSM, DPOT, and CNO etc

2) Training time, inference speed, and memory usage are not compared for WDNO and baselines.

3) Lacks novelty as compared to previous works such as:

* Benchmarking Autoregressive Conditional Diffusion Models for Turbulent Flow Simulation (arXiv:2309.01745)

* DiffusionPDE: Generative PDE-Solving Under Partial Observation (arXiv:2406.17763)

### Questions
1)  How does WDNO compare to baselines regarding training time, memory usage, and inference speed?

2) How was the basis wavelet chosen for the benchmarks?

3) Have the authors investigated using diffusion in the Fourier domain for comparison with WDNO, considering the computational efficiency of Fourier transforms?

4) How were hyperparameters for baselines chosen? Were they optimized fairly compared to WDNO?

5) The paper needs to elaborate on why WDNO outperforms DDPM, considering Parseval's identity, which states that information content remains constant during transformations.

6)  Why wasn't the Wavelet Neural Operator (WNO) included as a baseline for super-resolution tasks? How does WDNO compare to WNO, DDPM, UNET, etc., for long-range dependencies (Figure 6b)?

7) The paper should cite relevant related work and add as baselines, including:

* Benchmarking Autoregressive Conditional Diffusion Models for Turbulent Flow Simulation (arXiv:2309.01745)

* DiffusionPDE: Generative PDE-Solving Under Partial Observation (arXiv:2406.17763)

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
5

---

## Human Reviewer 4

### Summary
This paper presents a new method to simulate and control physical systems governed by specific PDEs. The method is based on learning a diffusion generative model in the wavelet domain with multi-resolution. The use of wavelet basis is specially convenient for discontinuous behaviours such as shock waves in fluid dynamics, and the diffusion procedure ensures that the full-trajectory error remains bounded. The method is tested with several fluid dynamic problems such as 1D Bruger's equaion, 1D and 2D Navier-Stokes in both simulation and control tasks.

### Strengths
* The main idea of combining WNO with diffusion is simple, novel and well justified.
* Every design choice is justified with multiple tests and ablation studies.
* The method outperforms other similar state-of-the-art techniques such as traditional U-Net/CNN to the newest Neural Operator architectures.

### Weaknesses
* The method is constrained to static uniform grid data and low-dimensional toy examples.
* The generalization to more complicated tasks or boundary conditions might not be trivial.

### Questions
* Can the authors discuss on the implication of using a regular grid? Is this constraint related to the use of a Fast Wavelet Transform? This restriction might be problematic when applying this algorithm to more complex geometries.
* The examples shown on the paper have changing initial conditions, but the geometry and boundaries remain fixed. Can the authors describe how the WDNO architecture is generalizable to other parameters or boundary conditions? It might be interesting for the reader to have a comment over problems with parametric solutions.
* Eq. 4: There is no reference to the guidance rate $\lambda$ until Appendix C3. Can the authors define it next to the equation, to improve readability?

Final comment: The paper has a solid idea, with well justified design choices and the results outperform existing techniques. Based on the comments above, my rating for this paper is marginally above the acceptance threshold.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 5

### Summary
Aiming at improving the performance of diffusion generative models, in particular for cases with abrupt changes, the authors propose to construct such diffusion models in the wavelet space. To reduce the solution manifold that the model needs to learn and therefore to ease super-resolution, the authors propose to scale the systems such that they align spatio-temporally, and additionally propose to train the model at multiple scales. The resulting model over-performs the state-of-the-art baselines in various simulation and control tasks.

### Strengths
1. A wavelet-based neural operator that surpasses the performance of exising methods.
2. Scaling of the input system reduces the solution manifold that the model needs to learn.
3. The paper is generally well-written, with details carefully reported.

### Weaknesses
The most significant weakness point of the work is the small number of temporal steps in all the test cases, which might actually be one of the limitation of these models, since the usual approach adopted to achieve long roll-out inference without instability is to introduce artificial training noise - it might be difficult to do so when one is already training a denoising model. It should be noted that the authors iteratively try to strengthen the point that they are doing "long-term dynamics" in the temproal domain, but their reported cases, e.g., the 2D incompressive fluid case, only consist 32 time steps each, which in the commonsense definition is quite a short forecasting window. I strongly suggest removing such terms from the paper to avoid confusion for future readers of the work.

### Questions
1. It seems that the Fourier neural operator is second to the proposed method in the simulation tasks. I am curious about how would FNO perform if it is also trained through a similar process as the proposed model (since the original FNO is not trained as a denoising model)?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4