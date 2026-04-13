## Human Reviewer 1

### Summary
In recent years, neural field-based methods for dynamics modelling have seen increasing interesting due to their flexible nature w.r.t. geometry and sparsity of the input. Most methods that have previously been proposed employ conditional neural fields, where a base model is conditioned by a global latent that is shared over all spatial positions in the input function. Although this has shown promising results in complex forecasting settings, the use of global modulation limits the ability of the NeF to reconstruct fine details. As such, authors propose instead learning a spatial basis of modulations in which a sample can be expressed through a combination of coefficients. This allows for location-specific modulation while retaining the globally informed continuous properties of global modulations. Authors show results for a range of experiments, assessing temporal generalization and robustness to subsampling, in which their method consistently improves over baseline architectures.

### Strengths
- The method is effective and simple, instead of learning a global modulation which limits fine-grained reconstruction or a purely local latent which limits generalizability, a spatial modulation is expressed in a learned basis. The experimental results show good performance. This approach is general and could be applied to a wide range of other interesting forecasting tasks.
- The paper is well-written and builds towards the method incrementally. The method is very well motivated, also supported by empirical evidence.
- The paper contains illustrative figures that aid in understanding the work.

### Weaknesses
- The only limitation I see, although I do not deem it a major one, is the relatively limited novelty of the method. There have been quite a number of previous works exploring spatial modulation for NeF-based reconstruction and downstream performance, notably [1, 2, 3]. For instance, in computer vision NeF literature a widely carried modulation/embedding is hash encoding [4], which your approach to modulation learning seems somewhat related to. Within PDE solving literature, a concurrent work [5] proposes a NeF-based solving framework based on DINo which also learns a spatial latent using MAML, though their approach is more tailored towards PDEs with known symmetries. It might be good to mention these works and compare (maybe in an extended related work section in the appx.), as the (vision) NeF community and the Neural Operator (physics informed learning) communities have been working somewhat in parallel.
- No details are given regarding the computational complexity of the method. Would your method scale easily to higher resolution / larger data? I think this is an important consideration, and as such it would be good to provide details on computational complexity and memory efficiency.

Other than this, I think this work is interesting and shows valuable results for the deep-learning based PDE solving community. I think this paper would be a valuable contribution to the conference.

[1] Bauer, M., Dupont, E., Brock, A., Rosenbaum, D., Schwarz, J. R., & Kim, H. (2023). Spatial functa: Scaling functa to imagenet classification and generation. arXiv preprint arXiv:2302.03130.
[2] Lee, D., Kim, C., Cho, M., & HAN, W. S. (2024). Locality-aware generalizable implicit neural representation. Advances in Neural Information Processing Systems, 36.
[3] Wessels, D. R., Knigge, D. M., Papa, S., Valperga, R., Vadgama, S., Gavves, E., & Bekkers, E. J. (2024). Grounding Continuous Representations in Geometry: Equivariant Neural Fields. arXiv preprint arXiv:2406.05753.
[4] Müller, T., Evans, A., Schied, C., & Keller, A. (2022). Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4), 1-15.
[5] Knigge, D. M., Wessels, D. R., Valperga, R., Papa, S., Sonke, J. J., Gavves, E., & Bekkers, E. J. (2024). Space-Time Continuous PDE Forecasting using Equivariant Neural Fields. arXiv preprint arXiv:2406.06660.

### Questions
- Is there a particular reason you chose to optimize the decoders and neural ode models separately? Would it not be possible to optimize these end-to-end?
- Could you provide details on the computational complexity of your method compared with CORAL? How much overhead is there in terms of memory and time complexity compared to this baseline?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
Summary: The paper introduces MARBLE, a new training framework for INR-based neural solvers for PDEs. MARBLE has two components: 1. GridMix and 2. Spatial domain augmentation (SDA). The paper builds off of CORAL [1], primarily changing the shift modulation. 

GridMix replaces $\phi_{a} = h_{a}(z_{a_i})$ with a linear combination of learned basis functions. As such, $\phi$ takes the form $\phi_{a}(x) = \Sigma_{m=1}^M c_i^m(z_a)\Phi^m_i(x)$. The set of coefficients $c^m$ are a hypernetwork function of the latent embedding $z_a$ and $\Phi$ functions are learned along with the INR weights. Points on the domain $x$, are binned into patches of $H\times W$, and $\Phi$ is a tensor of shape $H\times W \times C$, representing channel-wise modulation binned according to $H \times W$.

SDA is a form of data augmentation which stochastically drops points from the input domain.

*Experiments*: The paper conducts two sets of experiments: one on learning dynamics (NS, Shallow water) and the second on problems mapping from reference grid to steady states (Euler, NS Pipe, Hyper-elastic material). Ablations are performed over NS.

### Strengths
- The problem is of strong interest and the proposed solution is novel.
- Multiple complex datasets are used for experimental results, representing a variety of tasks. The proposed method has substantive improvements over the baselines.
- The ablations indicate a clear improvement to CORAL on both the in and out of distribution time-horizons.

### Weaknesses
High level weaknesses, articulated further under questions.
- Some of the mathematical notation and setup for GridMix is confusing. 
    - The computation of the position-dependent shift modulation scalar is not clear. The text says: “we derive the position-dependent shift modulation scalar $\phi_i(x)$ at location x through interpolation between neighboring features.” From fig 2, it seems that position x is mapped to a bin which then selects the corresponding modulation value. Please provide a step-by-step explanation as to how x is mapped to the modulation variable, including the details of how the interpolation is performed.
    - Equation 7 is presents $\phi_i(x)$ as independent of $z_{a}$. Equation 7 also omits the hypernetwork, obfuscating the dependence on $z_{a}$. Please clarify in the text that this dependence by providing a supplemental equation that illustrates this relationship or including the hypernetwork in equation 7.
- The baseline models are not consistent between the two classes of experiments.
- Some experimental details are not clear.

### Questions
- Why are the models used for the two evaluation settings different? While FNO is a good neural operator baseline, there have been many advancements (e.g., FFNO [2]) which could be run on the dynamics modeling case. *Requested Exp*: Use a better FNO variant for a more apt comparison to spectral neural operators (NOs). Can the authors justify the choice of baselines for each evaluation setting and why certain models were not used in both contexts? 
- Table 2: What does “FNO + lin. int.” mean? Additionally, how is an irregular grid being used with FNO? Please add a brief explanation of these terms and methods to either the table caption or main text. A spectral NO designed for irregular meshes would be a more apt baseline (e.g., Geo-FNO [3]). *Requested Exp*: Geo-FNO / FNO variant designed for irregular meshes.
- Table 2: No explanation for the N.A. values.
- “Baseline results for comparison are sourced from Serrano et al. [1]”. What does this mean? Please provide a detailed explanation of how the baseline results were obtained, including any steps to ensure consistency with the original datasets if new data was generated (seems to be indicated in Appendix A.1).
- There are a few cases where the proposed method performs marginally worse than some of the baselines (Shallow water $\pi=5\%$, Pipe). What about these cases makes MARBLE perform worse? It would benefit the paper to include a brief analysis of these specific cases in the discussion section, exploring possible reasons for the underperformance and how future work might add it.
- Dynamics modeling (architecture details): A single modulated INR is used in this case. This is different from Fig 1 and the setup where two modulated INRs are used: $f_{\theta_{a}}$ and $f_{\theta_{u}}$. What is the explicit differences from the general setup in the method section? Please clarify how the input and output functions are handled in this experimental setting.

[1] Serrano, L., Le Boudec, L., Koupaï, A., Wang, T., Yin, Y., Vittaut, J.N., & Gallinari, P. (2024). Operator learning with neural fields: tackling PDEs on general geometries. In Proceedings of the 37th International Conference on Neural Information Processing Systems. Curran Associates Inc..

[2] Alasdair Tran, Alexander Mathews, Lexing Xie, & Cheng Soon Ong (2023). Factorized Fourier Neural Operators. In The Eleventh International Conference on Learning Representations .

[3] Li, Z., Huang, D., Liu, B., & Anandkumar, A. (2024). Fourier neural operator with learned deformations for PDEs on general geometries. J. Mach. Learn. Res., 24(1).

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

## Human Reviewer 3

### Summary
To address the limitations of global modulation, this paper proposes GridMix technology, a novel spatial modulation method. GridMix combines the advantages of grid-based representation and spatial modulation by regularizing the modulation space into a set of shared basis functions, thereby enhancing the model's ability to learn global structures. Additionally, a spatial domain enhancement strategy is introduced to simulate domain changes that may occur during inference, further improving the model's generalization capability. Experiments demonstrate that GridMix excels in dynamic modeling and geometric prediction tasks, particularly in cases of temporal extrapolation and spatial subsampling, outperforming existing models.

### Strengths
1. Innovation: The introduction of GridMix overcomes the shortcomings of traditional global modulation in capturing local features by incorporating grid-based representation.
2. Experimental Validation: The effectiveness and versatility of MARBLE are validated through extensive experiments, including dynamic modeling and geometric prediction. The experimental design is robust, covering various testing scenarios and enhancing the credibility of the results.
3. Algorithm Flexibility and Universality: The MARBLE framework demonstrates applicability across multiple tasks by combining GridMix and spatial domain enhancement, indicating its broad potential for application in PDE modeling.

### Weaknesses
1. Insufficient Literature Review: There is a lack of introduction to grid-based methods, such as discretization techniques and computational efficiency.
2. Complexity of Proposed Method: Although the proposed new method is innovative, its complexity may lead to challenges in implementation, particularly regarding parameter tuning and optimization in different application scenarios. It is unclear whether this method retains a speed advantage compared to numerical simulations.

### Questions
1. What are the basic settings of the grid? The current article only mentions grid density, but it lacks details on discretization methods, grid shapes, and adaptive methods (how to handle complex geometries). Is the discretization method used by the authors consistent with those in numerical simulations?
2. What is the computational efficiency of the proposed method? There is a need to compare the proposed method with numerical simulations and other data-driven PDE modeling methods.

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper proposes an new architecture called GridMix, which enhances implicit neural fields (INRs) for solving PDEs. Building on multi-resolution INR advances from visual computing ([1], [2]), the authors adapt these methods to improve PDE modeling. More specifically, they solve the problem using CORAL[3] approach, where each of the INR is encoded using a multi-resolution feature grid (for spatial modulation), combined with a new spatial domain augmentation strategy to boost generalization across spatial domains. Experiments demonstrate GridMix’s efficacy in dynamic modeling tasks (temporal extrapolation and spatial interpolation) and in steady-state predictions based on domain geometry.

### Strengths
– Adaptation of multi-feature grids for auto-decoding: the paper presents a novel approach for leveraging multi-feature grids in the auto-decoding setting, effectively adapting INRs to capture complex spatial variations while maintaining efficiency.
– Extensive comparisons and ablations study
– Modeling spatial modulation using basis functions (and their coefficients) looks promising

### Weaknesses
– Some quantitative results, especially those comparing single-resolution grids with global modulation in Table 1, appear counterintuitive. The single-channel GridMix approach yields lower performance in some cases compared to simpler global modulation, which contradicts the initial motivation that spatial features should perform better.
– Missing mentioning Factor Fields [4] paper that also works in similar framework and solves similar problems but not in PDE domain

### Questions
– In Table 1, why single resolution feature grid (Single-channel GridMix) gives worse results
than Global Modulation? As the spatial feature representation should give already good results
given the initial motivation
– I dont fully understand the explanation of why more basis functions lead to worse results at
some point? As in case there are too many basis, they can be zeroed out with corresponding
weights
- Figure 3 is hard to read – neet legend on the Y axis. It is not clear to me from the footnote what’s going
on.


References:
[1] “Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Shapes” by Towaki
Takikawa et. al.
[2] “Instant Neural Graphics Primitives with a Multiresolution Hash Encoding” by Thomas Muller
et. al.
[3] “Operator Learning with Neural Fields: Tackling PDEs on General Geometries” by Louis
Serran et. al.
[4] “Factor Fields: A Unified Framework for Neural Fields and Beyond” by Anpei Chen et. al.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
2