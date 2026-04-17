# Physically Plausible Multi-System Trajectory Generation and Symmetry Discovery

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
From metronomes to celestial bodies, mechanics underpins how the world evolves in time and space. With consideration of this, a number of recent neural network models leverage inductive biases from classical mechanics to encourage model interpretability and ensure forecasted states are physical. However, in general, these models are designed to capture the dynamics of a single system with fixed physical parameters, from state-space measurements of a known configuration space. In this paper we introduce Symplectic Phase Space GAN (SPS-GAN) which can capture the dynamics of multiple systems, and generalize to unseen physical parameters from. Moreover, SPS-GAN does not require prior knowledge of the system configuration space. In fact, SPS-GAN can discover the configuration space structure of the system from arbitrary measurement types (e.g., state-space measurements, video frames). To achieve physically plausible generation, we introduce a novel architecture which embeds a Hamiltonian neural network recurrent module in a conditional GAN backbone. To discover the structure of the configuration space, we optimize the conditional time-series GAN objective with an additional physically motivated term to encourages a sparse representation of the configuration space. We demonstrate the utility of SPS-GAN for trajectory prediction, video generation and symmetry discovery. Our approach captures multiple systems and achieves performance on par with supervised models designed for single systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The author proposes a novel framework to integrate HNN to sturcture the latent space for physically plausible trajectory generation and video generation.

### Strengths
The paper is well-written. The motivation is clear, and the model structure is easy to understand.

### Weaknesses
1. The model may not preserve the conservation law in the raw trajectory space.
2. The experiments are insufficient.

### Questions
1.	The paper employs Hamiltonian neural network. What if the system doesn’t satisfy the conservation law?
2.	What’s the motivation for generation for multiple systems? Why is studying multiple systems together important?
3.	Some recent papers of learning symmetries in the latent space may be considered, e.g.,
a.	Li, Haoran, et al. "Latent Mixture of Symmetries for Sample-Efficient Dynamic Learning." arXiv preprint arXiv:2510.03578 (2025).
4.	As the latent space dynamics are governed by HNN and the decoder MLP may be nonlinear, how to guarantee that the Hamiltonian structure is preserved in the trajectories? 
5.	The experiments are insufficient. There is no ablation study and sensitivity analysis. 
6.	The results in Table 1 show that for several systems, the improvement is limited.

### Soundness
3

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
3

### Summary
The paper introduces Symplectic Phase Space GAN (SPS-GAN), a conditional GAN whose latent dynamics are governed by a Hamiltonian Neural Network (HNN). A configuration-space map projects random motion samples into a latent phase space, where the latent HNN evolves them forward using a leapfrog integrator. Decoders then render either Cartesian trajectories or videos. A cyclic-coordinate loss encourages sparsity in the latent coordinates, interpreted as a form of symmetry constraint. The model is conditioned on system labels and physical parameters to handle multiple systems simultaneously. On five classical benchmarks, it reports trajectory accuracy comparable to supervised HNNs and significantly lower Fréchet distance in video generation.

### Strengths
The work integrates HNN dynamics, context-dependent (environment-specific) modeling, and a GAN framework in a coherent way. While each component is known, their synthesis is novel. Also, the paper is clearly written and well organized.

### Weaknesses
The claim of “symmetry discovery” is somewhat overstated. The cyclic-coordinate loss captures only a limited form of symmetry, essentially momentum conservation, rather than uncovering more general symmetry groups. Consequently, the most compelling contribution of this paper is its video generation quality under varying system conditions, whereas the symmetry discovery aspect remains nuanced, inferred mainly through latent-space sparsity and t-SNE visualization.

### Questions
- Hamiltonian systems often exhibit Hamiltonian bifurcations, which are sudden topological transitions triggered by small parameter variations. A simple example is the Hamiltonian saddle-node bifurcation, $\mathcal{H}(q, p) = p^2/2 - \mu q + q^3/3$, and the Hamiltonian pitchfork bifurcation, $\mathcal{H}(q, p) = p^2 / 2 - \mu (q^2 / 2) + q^4 / 4$, where $\mu \in \mathbb{R}^1$ is a parameter. The later serves as a minimal theoretical framework for single-well ($\mu < 0$) to double-well ($\mu > 0$) transitions often seen in symmetry-breaking theory. It would be interesting to examine whether the proposed model, when trained solely in the pre-bifurcation regime (possibly along with other systems), can predict post-bifurcation behavior.

- The evidence for “correct DOF” is largely t-SNE pictures. t-SNE is not an intrinsic-dimension estimator; its geometry is view-dependent. The physical narratives for 1-DOF two-body and 2-DOF special three-body cases are right, but the validation metric should be more than a plot.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Symplectic Phase Space GAN (SPS-GAN), a framework designed to capture dynamical system behavior and generalize to unseen physical parameters. The model integrates a Hamiltonian neural network (HNN) recurrent module within a conditional GAN backbone to promote physically plausible generation. Additionally, the authors introduce a modified conditional time-series GAN objective with a physically motivated regularization term that encourages sparse structure in the learned configuration space.

### Strengths
* The idea of promoting a sparse representation in the configuration space to simplify the underlying motion dynamics is well-motivated and appears to be effectively implemented.
* The proposed SPS-GAN demonstrates superior empirical performance, outperforming several baseline methods across the evaluated tasks.

### Weaknesses
* The paper lacks a clear justification for the motivation of using a generative model (GAN) to represent dynamical trajectories. Given that Newtonian N-body systems evolve deterministically from initial conditions, the necessity of a probabilistic generative approach—as opposed to a deterministic supervised model—is not sufficiently motivated. This concern is compounded by the observation that SPS-GAN does not consistently outperform a supervised HNN baseline, raising questions about the added value of the GAN framework.

* The experimental scope is limited: The primary empirical validation is conducted on two relatively simple, low-dimensional toy benchmarks. It remains unclear whether the proposed method would scale effectively or retain its advantages in more complex, real-world physical systems, which limits assessment of its practical impact.

* The paper does not include a thorough ablation study to disentangle the individual contributions of key components—such as the HNN recurrent module, the GAN backbone, and the sparsity-inducing regularization. Without this, it is difficult to attribute the performance gains to any specific technique of the proposed architecture.

### Questions
* The authors use t-SNE visualizations (Figure 5) to justify the dimensionality of the learned latent space. Could they provide a more quantitative or principled explanation for how the size of the latent dimension was determined?
* Could the authors verify the presentation of the symplectic Leapfrog integration in Equations (4)-(6)? These steps do not seem to fully align with the canonical update rules.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes SPS-GAN, a GAN-based approach which generates trajectories governed by a Hamiltonian Neural Network. From sampled initial latents, dynamics are generated using HNN with leapfrog integration. The methodology introduces a cyclic-coordinate loss to learn an interpretable system with minimal degrees of freedom. Experiments on synthetic hamiltonian systems demonstrate superiority of SPS-GAN in generating plausible video trajectories.

### Strengths
- Generating trajectories with hamiltonian neural networks is an interesting problem.
- The combination of HNN and GANs look promising where SPS-GAN shows superiority in video generation.

### Weaknesses
**Novelty appears to be incremental**

I am not sure I can distinguish many differences between this paper and HGAN [1], which is already cited. HGAN is based on HNN and GANs, and introduces the same cyclic-coordinate loss terms as described in this paper (Equation 4 in their paper). While compared to in experiments, I cannot see HGAN discussed in related work.
 - The paper would substantially improve with a discussion highlighting the differences between both methodologies, and further addressing these points with ablation experiments on the improvements.

**Clarity and presentation issues**

- Equations (4-6) have typos. Please revise.
- Section 4 needs to be improved in terms of clarity. In its current stage it is a bit confusing.
  -  For example, it starts by introducing a function f, then it introduces a regularisation term, and it goes back to the function f. I would introduce the regularisation term after defining the final GAN loss.
  - \xi as an argument of the function is explained in Line 202, yet \xi appears first in line 189.
  - If the motion sample has a specific distribution, please indicate it where you define the motion sample.
  - Section 4 might benefit by explaining the overall idea of SPS-GAN together by referencing the elements in Figure 1.
  - Figure 1 and Figure 2 could be grouped together as one.
- Some referenced figures are not aligned in text, making explanations with figures difficult to follow (e.g., when describing results from Figure 3, 4, or 5).

**Experiments section might need revision**
- The MSE results compute grountruth vs predicted trajectory. What is the length of this rollout? What is the MSE in terms of different rollout lengths? An interesting advantage to observe for a Hamiltonian system would be to test if it can generate large rollouts which are plausible. 
- For videos, it would be interesting to report results for long/short rollouts and compare.
- Line 305: “Furthermore, the generated trajectory has energy drift within 2.5% of the ground truth conserved energy” what does this imply?
- The predicted trajectories only compare to HNN from 2019. Would it be possible to try other more recent baselines? e.g. HALO or HGN mentioned in related work, HGAN, or a non-hamiltonian continuous-time sequence model (latent ODE).
- Line 307: “We further evaluate SPS-GAN on generating all five systems simultaneously, with explicit system labels and physical parameters used as the conditioning inputs”. Can you provide details of this setup in the main text? Does this mean you provide the model auxiliary labels to distinguish samples from different datasets, and each label activates different \xi’s? It is not clear from this explanation.
- Line 313: “A core contribution of SPS-GAN is the ability to identify symmetries by minimising the dimension of the latent motion space.” The words “latent motion space” have not been used previously and could cause confusion to what they refer to. If they refer to the dimensionality of q, accompany it with the corresponding notation.
- The T-SNE plot to show that the intrinsic dimensionality of the learned systems corresponds to the degrees of freedom is not very intuitive. A quantitative approach to determine intrinsic dimensionality would substantially improve the explanation. In the figure, I cannot see any differences between a and c, yet the authors interpret a has 1 dimension, and c has 2 dimensions.
- How do we select \lambda_{cyclic} in practice? Ablation experiments on how to select \lambda_{cyclic} should be provided.


**Limitations**

The system-specific binary mask requires domain knowledge, which limits applicability to real-world data.


[1] Allen-Blanchette, Christine. "Hamiltonian gan." 6th Annual Learning for Dynamics & Control Conference. PMLR, 2024.

### Questions
- If your dynamics are hamiltonian systems, why does the discriminator use an RNN, and not some specific hamiltonian system embedding? This might be an interesting innovation point to explore.
- If we plan to launch this method in real-world systems, how can we design the binary mask?
- What are the benefits from training all the datasets at once if we require additional labels to condition the system parameters? Please correct if I understood this part of the setup wrong.

### Soundness
3

### Presentation
2

### Contribution
2
