# Orbital Transformers for Predicting Wavefunctions in Time-Dependent Density Functional Theory

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
We aim to learn wavefunctions simulated by time-dependent density functional theory (TDDFT), which can be efficiently represented as linear combination coefficients of atomic orbitals. In real-time TDDFT, the electronic wavefunctions of a molecule evolve over time in response to an external excitation, enabling first-principles predictions of physical properties such as optical absorption, electron dynamics, and high-order response. However, conventional real-time TDDFT relies on time-consuming propagation of all occupied states with fine time steps. In this work, we propose OrbEvo, which is based on an equivariant graph transformer architecture and learns to evolve the full electronic wavefunction coefficients across time steps. First, to account for external field, we design an equivariant conditioning to encode both strength and direction of external electric field and break the symmetry from SO(3) to SO(2). Furthermore, we design two OrbEvo models, OrbEvo-WF and OrbEvo-DM, using wavefunction pooling and density matrix as interaction method, respectively. Motivated by the central role of the density functional in TDDFT, OrbEvo-DM encodes the density matrix aggregated from all occupied electronic states into feature vectors via tensor contraction, providing a more intuitive approach to learn the time evolution operator. We adopt a training strategy specifically tailored to limit the error accumulation of time-dependent wavefunctions over autoregressive rollout. To evaluate our approach, we generate TDDFT datasets consisting of 5,000 different molecules in the QM9 dataset and 1,500 molecular configurations of the malonaldehyde molecule in the MD17 dataset. Results show that our OrbEvo model accurately captures quantum dynamics of excited states under external field, including time-dependent wavefunctions, time-dependent dipole moment, and optical absorption spectra characterized by dipole oscillator strength. It also shows strong generalization capability on the diverse molecules in the QM9 dataset. Our dataset is available at https://huggingface.co/divelab, and our code is available as part of the AIRS library https://github.com/divelab/AIRS/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the important and challenging problem of accelerating Real-Time TDDFT (RT-TDDFT) computations using deep learning.  
Specifically, it adopts an autoregressive framework to accelerate the propagations of RT-TDDFT, where the wavefunctions of previous steps are input into the network for the prediction of the next steps' wavefunctions. The paper proposes two model architectures (OrbEvo-FullWF and OrbEvo-DM) with different electronic state interacting strategies and compares their performance on their self-generated TDDFT dataset.

I think the paper is in a good shape, with nontrivial contributions for a novel application (RT-TDDFT) and specifically designed models (OrbEvo-FullWF and OrbEvo-DM). Nonetheless, there exist several concerns, which should be addressed before acceptance.

### Strengths
* Machine learning TDDFT is an important and relatively under-explored research field. This paper approaches the problem with a novel setting (directly learning the wavefunctions), representing a pioneering attempt in this direction. The handling of the high-dimensional orbital coefficient object (the time, the electronic state and the atomic orbital dimensions are all different from the number of atoms) is especially interesting.
* The design choices of the prediction target and the model architectures seem to be thoroughly considered and physics-grounded.
* The practices of improving model performance, such as the delta transformation of wavefunctions, the time bundling setting and the push-forward training, are detailed and provide a good impression of adopting state-of-the-art techniques in related fields.
* The paper is written in a clear and easy-to-understand language, with moderate background knowledge included.

### Weaknesses
* Although the paper has provided some background information for RT-TDDFT, I find myself unclear about the big picture and lost in the implementation details. It is not clear how the learned model is used in the RT-TDDFT framework.
* The units are missing from all the performance numbers in the paper.
* Neither the source code nor the dataset are provided. This renders the paper hard to follow.

### Questions
* About motivation: Could you provide a clearer picture of the framework? For example, how is the learned model used in RT-TDDFT to accelerate the computation?
* About performance: How much acceleration is achieved? How should one interpret the metrics in Table 1 and Table 2?
* How does the model compare to direct property prediction methods?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes *Orbital Transformers*, an equivariant graph Transformer designed to directly predict the *time evolution of Kohn–Sham wavefunctions* in real-time time-dependent density functional theory (RT-TDDFT). Unlike prior approaches that predict energies, Hamiltonians, or spectral observables, this model learns the mapping $C(t) \to C(t+\Delta t)$ (or $\Delta C_t$) directly, effectively learning the quantum propagation operator. The authors introduce an SO(2)-equivariant attention mechanism that takes the external electric field direction as the reference axis, and use FiLM-style conditioning to inject both the field’s direction and time-dependent amplitude. A local autoregressive temporal modeling scheme, along with pushforward training, enables the model to track the dynamic evolution of the system stably over several femtoseconds. Experiments on RT-TDDFT trajectories of QM9 and MD17 molecules under external fields show that the model accurately reproduces dipole dynamics and orbital evolution.

### Strengths
1. This is the first work to *directly predict the time evolution of wavefunctions* in RT-TDDFT rather than energies or Hamiltonians. The idea of learning an implicit propagation operator $f_\theta: C(t)\mapsto C(t+\Delta t)$ is conceptually novel and impactful.  
2. Modeling system evolution under an external field is *physically meaningful* and directly corresponds to realistic nonequilibrium dynamics.  
3. The paper presents an *innovative autoregressive temporal modeling* strategy that allows the neural network to continuously track and predict the system’s electronic evolution, combining local ΔC prediction with pushforward training to reduce error accumulation.  
4. The *integration of field information into the atomic graph network* is well designed: the electric-field direction defines the SO(2) equivariant reference axis, and the field information enters through FiLM-style modulation. This construction is both physically grounded and computationally efficient.

### Weaknesses
1. The network predicts the next-step evolution solely from the current state, **and although the authors adopt certain stabilization strategies** — such as local autoregressive temporal modeling, pushforward training, and ΔC prediction — these mechanisms only address short-term error accumulation. **There remains no explicit architectural component (e.g., temporal attention or recurrent memory) to model long-range temporal dependencies**, which consequently limits robustness and stability during long-horizon propagation beyond several femtoseconds.

2. The SO(2)-equivariant design relies on the external-field direction as the rotational reference axis. Once the external field vanishes, this reference loses physical meaning, and the network no longer has a well-defined axis for the equivariant operations. Moreover, if the external-field direction varies over time or differs across samples, the reference frame for the SO(2) operations also changes with time. It is unclear how well the model would perform under such cases.

3. The paper appears to **lack systematic efficiency evaluation experiments**, for example, comparisons on runtime and computational resource usage, which would be important to assess the practical utility of the proposed model.

### Questions
The authors generate and RT-TDDFT trajectories with external fields for standard datasets like QM9 and MD17, which may enrich the data resources for time-dependent electronic-structure ML. Will the generated trajectories and other data be publicly released, providing valuable data for future time-dependent quantum ML research?

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
5

### Summary
The paper proposed a new model and method that learns the time-dependent DFT's properties, and has shown that the new proposed method, combined with a serious method improvement, can predict nicely the properties from TDDFT.

### Strengths
1. The network is well designed with very good respect for the physics.
2. The increase in accuracy is very impressive.

### Weaknesses
1. The writing needs more clarification, for example, the model design, more words in the caption and the method part can be helpful to understand it clearly.
2. The experimental work is comparably weak. More comparison with other models, and specific studies on in-distribution, out-of-distribution systems can make this study more solid.

### Questions
1. In real applications, we often focus on a specific kind of system, where we often do not have that much data. In terms of this, how is this method transferable to similar structures but with limited training data? How is the data efficiency in that?

2. How to transfer the model to different external field conditions?

3. How is the SO3-SO2 mapping achieved in layer norm? I haven't found it in the paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces OrbEvo, an equivariant graph transformer framework for learning the time evolution of Kohn–Sham wavefunctions in real-time time-dependent density functional theory (RT-TDDFT). Unlike prior works such as OrbFormer, which focus on static ground-state properties, OrbEvo aims to learn the dynamics of electronic states under external electric fields.
The authors propose two model variants: OrbEvo-FullWF, which aggregates wavefunction features through pooling across occupied states, and OrbEvo-DM, which computes density-matrix-based interactions between states via tensor contraction. The model employs SO(2)-equivariant conditioning to represent field-induced symmetry breaking and a pushforward training scheme to stabilize long-horizon rollout. Experiments on QM9 and MD17 demonstrate that OrbEvo-DM outperforms the pooling-based variant, capturing physically consistent time-dependent dipole moments and absorption spectra.

### Strengths
**Easy to follow and clear motivation**: The paper provides an intuitive and well-structured explanation of the challenge in modeling time-dependent DFT, with smooth transitions from motivation to formulation.

**Tackling an impactful and general problem**: The work addresses a scientifically meaningful and practically impactful challenge: learning the time evolution of Kohn–Sham wavefunctions to accelerate quantum dynamics simulations. Importantly, the model is trained across thousands of diverse molecules (QM9) and demonstrates generalization to unseen molecular systems, showing potential as a shared, cross-molecular surrogate model for electronic dynamics. This highlights the scalability and versatility of the approach beyond single-molecule modeling.

**Sound model design**: The use of SO(2)-equivariant conditioning, density-matrix features, and push forward training demonstrates strong physical insight and solid engineering.

### Weaknesses
Dataset scale is limited, making it difficult to assess generalization to larger molecules or different field conditions.

### Questions
1. **Applicability to static DFT**: Can the OrbEvo architecture be used for static ground-state DFT tasks, such as predicting stationary wavefunction coefficients or density matrices? Or is it strictly limited to time-dependent TDDFT propagation?

2. **Ablation on time bundling**: How much does the time bundling technique contribute to the model’s performance and stability? It would be helpful to see results comparing models with and without time bundling.

### Soundness
3

### Presentation
4

### Contribution
3
