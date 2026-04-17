# A Large-Scale Atomic Interaction Model Based on Matter Wave Theory

- Decision: Reject
- Scores: 2, 2, 2

## Abstract
Accurate and rapid prediction of atomic interactions constitutes a fundamental challenge in materials science. Traditional numerical methods face persistent limitations in balancing computational accuracy with efficiency. In contrast, AI-based large-scale atomic interaction models efficiently learn characteristic patterns of atomic configurations, enabling high-speed simulations while preserving accuracy. This offers a novel paradigm for molecular dynamics simulations and accelerated discovery of new materials and pharmaceuticals. To advance beyond current performance limits, this work proposes a matter wave theory-based large-scale atomic interaction model. First, we explicitly encode quantum mechanical matter wave theory into the neural network architecture, designing a quantum-inspired matter wave network as the core module. This innovation fundamentally enhances physical representation by effectively capturing atomic wave-particle duality. Subsequently, comprehensive error evaluation across multiple datasets (including Perovskite Oxides) demonstrates that our proposed Matter Wave Deep Potential Atomic model achieves root mean square errors of 0.5 meV/atom for energy and 28.7 meV/Å for force. These represent reductions of 16% and 8%, respectively, compared to state-of-the-art models including Deep Potential Atomic. Finally, as a standalone, general-purpose module, the matter wave network readily integrates with other advanced atomic interaction models. This adaptability will propel molecular dynamics simulation capabilities and expedite materials design and pharmaceutical discovery, thereby generating significant societal value.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a new interatomic potential architecture inspired by matter-wave theory. It augments a deep potential network with sinusoidal descriptor blocks intended to encode quantum-wave-like periodicity in local atomic interactions. The model is trained on total energies with gradients for atomic forces and shows improved energy RMSE on four DFT datasets.

### Strengths
The sinusoidal block is a clear attempt to encode wave-based periodicity absent in standard local descriptors.

Interpretability potential: “Matter-wave” amplitudes could be correlated with bond-length or phase interference patterns.

Clear writing and figures: The pipeline and ablations are understandable, aiding reproducibility.

### Weaknesses
While promising, the evaluation remains narrow. To fully substantiate the physical motivation, the model should be tested on standard molecular (QM9, MD17/rMD17) and crystalline (Matbench, OC20/22, CrysMTM) benchmarks that probe diverse physics—small-molecule electronic properties, periodic-boundary effects, surfaces, and temperature- or phase-dependent behavior. Expanding targets beyond energy/forces to stress tensors, elastic moduli, phonon spectra, and electronic properties would reveal whether the “matter-wave” layer captures transferable physical features rather than dataset-specific correlations.

### Questions
Could the “matter-wave” descriptors be replaced by generic Fourier features—would accuracy degrade?

Does the model remain stable in long MD simulations (NVE drift, RDF/MSD consistency)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed a new machine learning force field model inspired by the matter wave theory and Kolmogorov-Arnold Network (KAN). The core of the method is a feature map $\Phi(G, S) = A \odot \sin [\frac{2\pi}{h} \cdot (P\cdot S + G\cdot T)]$ inspired by the de Broglie wave $\psi(\mathbf{r},t)=A e^{i(\mathbf{k}\cdot \mathbf{r} - \omega t)}$. Here $G$ is a energy feature and $S$ is a spatial feature, and momentum $P$ and time $T$ are learnable weights. The new network is modified from the previous work DPA2 by adding this matter wave feature, and energy and force prediction experiments was conducted on 4 different dataset against baselnie methods.

### Strengths
The paper is well-written and easy to follow. Figure 1 explains the overall architecture pretty well. The experiment seems comprehensive, and the model performs well in energy prediction on the chosen datasets.

### Weaknesses
- The performance improvements are marginal at best. For force prediction, it is actually worse than other methods, except for the H2O-PD dataset.
- matter wave theory is a *linear theory*. Since you have multiple layers with nonlinear activation, I would say calling this a matter wave is a stretch. This is more like a Fourier feature.
- The paper asserts that linear aggregation "satisfies" superposition and "self‑consistent convergence" (Sec. 2–3.2) but gives no derivation or quantitative test that the learned features actually behave like wavefunctions (e.g., interference patterns under controlled setups, dispersion relations)
- The usage of KAN is not well motivated. There are no ablation studies demonstrating that using KAN is beneficial.
- Using sinusoids to encode geometry is not new. atomistic GNNs often use periodic/basis expansions (e.g., Bessel/spherical‑harmonic bases), and many models use Fourier features or sinusoidal activations. The paper does not benchmark against such periodic encodings, which weakens a novelty claim tied to "periodicity."
- Units inconsistency. Table 2 labels "Energy RMSE [meV/atom]" yet reports values in the 10–20 meV/atom range during training/validation, while Table 1 shows 0.5–1.0 meV/atom test RMSE for the same datasets. This doesn't make any sense.
- Several baselines are marked "OOM" without describing the memory setting, batch sizes, or whether mixed precision and gradient checkpointing were used—this can bias comparisons. 
- Hyperparameters critical for reproducibility are missing.
- There is no molecular‑dynamics evaluation (e.g., NVE energy drift, long‑horizon stability, diffusion coefficients). Since the motivation is improved MD, the per‑frame test error is only a proxy.

### Questions
- Why don't you learn $W^{1}$ in equation 5? Since you are already learning momentum and time, setting this weighting to uniformly $1$ is strange.
- Why did you cite Kohn-Sham instead of de Broglie for the matter wave theory? This is very strange to me.
- From the experiments, WDPA for energy prediction works much better than force prediction. Do you have any explanation for this?
- Why do you need Planck's constant here? There is no inherent physical process here, so we can set the scale to whatever we like. In fact, in most DFT calculations we use atomic units where $h=1$.

### Soundness
1

### Presentation
3

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
The paper produces the Matter Wave Deep Potential Atomic (WDPA) model that claims to incorporate physics-based inductive bias into the architecture. This incorporation is given by using the "matter wave feature transformation function", a sinusoidal function as part of the layers. Additionally, the paper provides benchmarks for the resulting model on several datasets, including FerroEle, H2O-PD, and Cathode.

### Strengths
Consistent improvements in accuracy over the studied datasets. 

A general-purpose module that can be plugged into other architectures.

### Weaknesses
Physical justification is nominal. The matter-wave transform reduces to a learnable sinusoid of linear combinations of features and distances. That is indistinguishable in practice from widely used Fourier/Bessel radial–angular bases or positional encodings. The presence of the Planck constant in the matter wave feature transformation function is off the point and seems not to be dimensionally consistent. 

In the introduction, the paper claims that the architecture is also built upon Kolmogorov-Arnold Networks, but this connection is missing in the further discussion. 

All the presented benchmark datasets are relatively small, with fewer than 10 thousand frames, much smaller compared to such projects as open materials or open molecules with more than 100 million configurations each. Thus, the "large-scale" claim is not justified.

### Questions
What is the connection between the proposed model and matter wave theory beyond merely sinusoidal expressions? 

What is the connection between the proposed model and Kolmogorov-Arnold Networks?

### Soundness
2

### Presentation
2

### Contribution
2
