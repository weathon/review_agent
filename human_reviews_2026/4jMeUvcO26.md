# Surrogate Modeling of 3D Rayleigh-Bénard Convection with Equivariant Autoencoders

- Decision: Reject
- Scores: 4, 6, 6

## Abstract
The use of machine learning for modeling, understanding, and controlling large-scale physics systems is quickly gaining in popularity, with examples ranging from electromagnetism over nuclear fusion reactors and magneto-hydrodynamics to fluid mechanics and climate modeling. These systems — governed by partial differential equations — present unique challenges regarding the large number of degrees of freedom and the complex dynamics over many scales both in space and time, and additional measures to improve accuracy and sample efficiency are highly desirable. We present an end-to-end equivariant surrogate model consisting of an equivariant convolutional autoencoder and an equivariant convolutional LSTM using $G$-steerable kernels. As a case study, we consider the three-dimensional Rayleigh-Bénard convection, which describes the buoyancy-driven fluid flow between a heated bottom and a cooled top plate. While the system is E(2)-equivariant in the horizontal plane, the boundary conditions break the translational equivariance in the vertical direction. Our architecture leverages vertically stacked layers of $D_4$-steerable kernels, with additional partial kernel sharing in the vertical direction for further efficiency improvement. We demonstrate significant gains in sample and parameter efficiency, as well as a better scaling to more complex dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present a method to obtain a surrogate model of the 3D Rayleigh-Bénard convection equation which is equivariant over rotations and translations in the horizontal 2D direction. They use a encoder-process-decoder architecture: (1) encoder based on steerable kernels, (2) LSTM to predict rollout predictions whose MLPs are substituted with equivariant kernels, and (3) decoder with steerable kernels and trilinear interpolation upsampling. The composition of equivariant components make the whole model to be E(2) equivariant in the horizontal dimension and height-dependant learnable kernels in the vertical direction, where the symmetry is broken due to the buoyancy dynamics. The model is tested over the mentioned equation resulting in lower errors than the non-equivariant counterpart with significantly less parameters. A long-term forecasting test also shows that the method outperforms FNO and U-Net baselines.

### Strengths
* The compression ratio of the autoencoder is very high and the results outperform the baselines by a decent margin.
* The local vertical parameter sharing is a smart solution to the vertical loss of symmetriy.
* The paper treats equivariance with mathematical rigour and proofs.

### Weaknesses
* The paper focuses primarily on a single specific equation and lacks validation on more complex phenomena or dynamical scenarios. One would expect a method designed for equivariant modeling to demonstrate broader applicability to a variety of PDEs exhibiting such symmetries and symmetry-breaking, but this is not explored in the paper.
* The contribution of the paper is quite limited. Equivariant autoencoders have been extensively explored in the literature, and the replacement of LSTM forward networks with an equivariant version represents only a modest contribution. The use of height-dependent kernels, however, is an interesting aspect.
* The code is not ready for review.

### Questions
* Lines 324-325: Are all 900 timesteps within the interval [100, 1000] included in the training data? Does the model perform any extrapolation to snapshots outside this range?
* Section 3.2: The autoencoder is evaluated across multiple Rayleigh numbers, which is appropriate given their strong influence on the system’s dynamics. However, the long-term forecasting experiments appear to be conducted for a single Rayleigh number, though this is not explicitly stated. The results would be more convincing if a similar multi-Ra number analysis were included for the forecasting stage as well.
* Section 4.1.2: One of the main contributions of the paper is the introduction of height-dependent kernels, yet this aspect remains relatively underexplored. How sensitive are the results to the choice of kernel size or vertical resolution?
* Line 965: The text mentions that the non-equivariant model was trained with data augmentation. Can the authors clarify whether similar augmentations (rotations, translations) were also applied to the long-term forecasting baselines (FNO, U-Net)? This would help ensure a fair comparison and clarify whether the performance gap stems primarily from architectural equivariance or from differences in training data diversity.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper designs an equivariant neural architecture for learning the phenomenon of buoyancy-driven fluid flow between a heated bottom and a cooled top plate involved in climate simulation, incorporating equivariant properties of the corresponding PDE solution directly inside the neural architecture. The authors validate this approach by comparing it to non equivariant surrogate models, FNO and U-Net, on the learning problem at hand.

### Strengths
- The paper introduce a novel architecture component by tailoring equivariant layers to their test case, with several innovating technical contributions.
- The presented architecture, $D_4$- steerable, outperforms other baselines with far fewer model paramaters for both short and long time horizons.

### Weaknesses
**W1** The obtained architecture seems to be applicable to more diverse test cases that the one presented: the paper would benefit from experiments on other datasets, all the more since the presented dataset is not standard and of moderate scale.

### Questions
**Q1** In what other types of simulation could the equivariant CNN be useful?

**Q2** l.321: "We generated a dataset of 100 randomly initialized 3D Rayleigh-Bénard convection simulations with Ra = 2500 and P r = 0.7," Are you sampling only initial conditions ? If Ra is set to 2500, how can different value of Ra be tested in Figure 4 (rightmost plot) ?

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
3

### Summary
This paper presents a surrogate modeling approach for three-dimensional Rayleigh-Bénard convection (RBC) using deep learning methods that respect spatial symmetries. The authors propose a two-stage architecture consisting of an equivariant convolutional autoencoder (CAE) and an equivariant convolutional LSTM. The CAE compresses high-dimensional flow fields into a structured latent space, while preserving horizontal rotational and reflectional symmetries through D4-steerable convolutions. The latent dynamics are then modeled with a convolutional LSTM that predicts temporal evolution in this compressed space. The model aims to approximate the underlying partial differential equations governing the RBC system, reducing computational cost relative to full numerical simulation.

The model operates on three-dimensional fields that contain temperature and velocity information across space. These fields are first passed through an encoder, which compresses them into a lower-dimensional latent representation while preserving spatial structure. A decoder then reconstructs the original field from this compressed form. For modeling time evolution, a recurrent neural network, specifically a convolutional LSTM, is used to predict future latent states based on previous ones. The system is trained in two separate phases: first, the autoencoder is optimized to minimize reconstruction error between the input and the decoded output; second, the LSTM is trained to forecast latent states over time. The full pipeline is evaluated using data from high-fidelity simulations of Rayleigh-Bénard convection at different Rayleigh numbers. According to the results, the proposed method achieves better data efficiency, lower reconstruction and forecasting errors, and requires fewer parameters than both standard convolutional neural networks and Fourier Neural Operator baselines.

### Strengths
The methodology is technically consistent and leverages symmetry-aware design principles grounded in group representation theory. The explicit use of D4-steerable convolutions ensures equivariance to horizontal rotations and reflections, which is physically appropriate for Rayleigh-Bénard convection where boundary conditions break vertical but not horizontal symmetry. The decision to apply height-dependent kernels is a reasonable approximation to accommodate vertical heterogeneity in the flow. The decomposition of spatial compression and temporal evolution into separate modules (CAE and LSTM) leads to a well-structured and interpretable model pipeline. This separation also reduces computational complexity, as the temporal predictor operates on latent tensors rather than full-resolution fields.

Empirical evaluation includes relevant comparisons with non-equivariant models and established operator learning baselines. The reported improvement in reconstruction error (approximately 40\% reduction in RMSE) and parameter efficiency (roughly an order of magnitude fewer parameters) is quantitatively clear. The experiments also cover different Rayleigh numbers, providing some evidence that the model generalizes to flows of varying complexity. The framework demonstrates that enforcing group-theoretic structure can lead to better inductive bias for spatiotemporal PDE systems.

### Weaknesses
The experimental validation remains limited in scope and does not fully establish the method’s robustness. All results are obtained on a single physical system (Rayleigh-Bénard convection) with idealized, noise-free data, which restricts the conclusions about general applicability. A joint end-to-end optimization might yield a more physically coherent latent space, though at higher cost. The study does not include ablations quantifying how much each design element (e.g., D4-steerable filters, local vertical parameter sharing) contributes to performance gains. Additionally, while the model is termed “equivariant,” it is only partially so: symmetry constraints are applied in horizontal planes but not in the vertical dimension, meaning the full $E(3)$ group is not represented. The reported efficiency gains are primarily relative to baseline models that do not exploit symmetries or that are trained under different regimes, which makes direct fairness of comparison uncertain. Finally, the paper does not explore stability or error accumulation over very long autoregressive rollouts, which is a key concern for temporal surrogate models.

### Questions
1. The model is trained in two separate phases: first the autoencoder, then the temporal predictor. While this improves training efficiency, it may introduce a mismatch between the latent representations learned by the encoder and those needed for accurate forecasting. Did the authors attempt any form of joint fine-tuning or end-to-end training, even partially? Can the authors comment on whether this decomposition leads to artifacts or long-horizon degradation in practice?

2. The proposed model architecture includes several components motivated by physical and architectural reasoning (e.g., D4-steerable convolutions, height-dependent filters, vertical parameter sharing). However, no ablation studies are provided to assess their relative contribution. Can the authors provide experiments or discussion that isolate the impact of these design choices?

3. Several baselines, such as U-Net and FNO, are included in the comparison. Were these models adapted for 3D input and trained with similar levels of supervision and data? FNO in particular is known to have strong performance in 2D PDE forecasting; was it adapted in a memory-efficient manner for 3D, or was the comparison constrained by hardware? 

4. Does the learned latent space exhibit any physical interpretability? For instance, do certain channels correspond to flow structures such as rolls or plumes? Could latent vectors be interpolated or manipulated to generate physically meaningful transitions in the decoded space?

5. Although different Rayleigh numbers are used in training and evaluation, the setup assumes a fixed domain and set of boundary conditions. Would the model generalize across different physical configurations, such as changes in domain size, aspect ratio, or boundary heating profiles? If not, what would be required to make the surrogate model adaptive to such changes?

6. All training data appears to be generated from numerical simulations. How robust is the model to noise or distributional shifts, as would be expected in real-world experimental measurements or lower-fidelity simulations?

### Soundness
3

### Presentation
3

### Contribution
3
