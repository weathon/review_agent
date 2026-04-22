# Generalized Spherical Neural Operators: Green’s Function Formulation

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 4, 6

## Abstract
Neural operators offer powerful approaches for solving parametric partial differential equations, but extending them to spherical domains remains challenging due to the need to preserve intrinsic geometry while avoiding distortions that break rotational consistency. Existing spherical operators rely on rotational equivariance but often lack the flexibility for real-world complexity. We propose a generalized operator-design framework based on designable Green’s function and its harmonic expansion, establishing a solid operator-theoretic foundation for spherical learning. Based on this, we propose an absolute and relative position-dependent Green’s function that enables flexible balance of equivariance and invariance for real-world modeling. The resulting operator, Green's-function Spherical Neural Operator (GSNO) with a novel spectral learning method, can adapt to non-equivariant systems while retaining spherical geometry, spectral efficiency and grid invariance. To exploit GSNO, we develop SHNet, a hierarchical architecture that combines multi-scale spectral modeling with spherical up-down sampling, enhancing global feature representation. Evaluations on diffusion MRI, shallow water dynamics, and global weather forecasting, GSNO and SHNet consistently outperform state-of-the-art methods. The theoretical and experimental results position GSNO as a principled and generalized framework for spherical operator learning, bridging rigorous theory with real-world complexity. The code is available at: https://github.com/haot2025/GSNO.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Spherical Fourier Neural Operators (SFNOs) are a form of neural network that take as input a scalar signal on the sphere and output a new scalar signal on the sphere. A major part of SFNOs are spherical convolutions that are equivariant under rotations. This convolution is implemented in the Fourier space of the sphere.

The main idea of the submitted paper is to add positional dependency to the spherical convolution, thus breaking the rotation symmetry.This is done in each convolution layer by adding a learned positional embedding to the signal before it is convolved by a filter. The addition of the positional embedding is also done in Fourier space.

Experiments on three tasks (without full rotational symmetry) show that the added positional information can be beneficial.

### Strengths
1. The proposed method is well-motivated, easy to implement and seems to work.
2. The paper is clear and compares extensively to the most relevant baseline, i.e. networks built from ordinary SFNOs.

### Weaknesses
I have a few technical questions, see below. Other than that, the main weakness is that relevant prior literature is not sufficiently covered in my opinion.

1. The idea to add auxiliary positional information is a spherical analogue of absolute positional embeddings as used in NLP/vision, and very similar to for instance “CoordConv” (Liu et al, An intriguing failing of convolutional neural networks and the coordconv solution, NeurIPS 2018).
2. More generally, symmetry breaking by adding auxiliary inputs is a broad research direction that is not referenced in the submitted paper. See e.g. Smidt et al (Finding Symmetry Breaking Order Parameters with Euclidean Neural Networks, Phys. Rev. Research 3, 012002 (2021)) and later follow-ups.
3. Spherical CNNs are not mentioned in the related work at all, except a reference to Cohen et al 2018, without explaining that it covers spherical networks. I believe spherical CNNs should be discussed in a bit more detail with references also to later works. When using spherical convolutions such as the ones in the submitted paper, the filters must be zonal (e.g. seen by formula 13, where $m\neq 0$ parts of $G$ are not present). This is addressed in prior work on spherical CNNs for instance by lifting the signal from the sphere to SO(3) or by storing steerable features relative to a local gauge at each point on the sphere (Cohen et al, Gauge Equivariant Convolutional Networks and the Icosahedral CNN, ICML 2019). The potential for obtaining more expressive layers in this way should at least be mentioned in the submitted work. 

It would make the contribution of the paper clearer if particularly points 1 and 2 above were discussed.

Minor weaknesses/typos:

1. Equation (7) is incorrect as the definition of $\delta$ should be that integrating $f \delta$ evaluates $f$ at a specific point as in equation (9).

### Questions
1. It seems from the hyperparameter table in the appendix that the original SFNO-nets and the new GSFNO-nets have different numbers of channels and layers. Why?
2. It would be interesting to plot the learned auxiliary positional information $G^{(2)}$. Intuitively it should be symmetric around the axis of rotation in the shallow water experiments and perhaps have further structure in the weather prediction task (due to landmasses on Earth etc). Is this the case? Are there other structures apparent in the learned $G^{(2)}$? This would relate to the stated motivations for including $G^{(2)}$.
3. The original SFNO-nets already have a positional embedding at the start of the network. Is the main reason for improvement in GSFNO-nets that positional information is more explicitly provided in each layer?
4. Why is the performance worse with more trainable parameters in Table 2?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a new family of spherical neural operators derived from a Green’s-function formulation, termed Generalized Spherical Neural Operator (GSNO), and a hierarchical architecture GSHNet.
The authors aim to bridge theoretical rigor and practical modeling of spherical PDEs by introducing an absolute-position-dependent constraint into the Green’s-function kernel, thereby relaxing strict rotational equivariance while retaining spectral efficiency.

### Strengths
1. The authors present an interesting point that SFNO’s preservation of rotational equivariance on the sphere could actually restrict its usefulness in many real-world scenarios where such equivariance does not naturally occur.

2. The authors include clear visualizations.

### Weaknesses
1. Theoretical contribution remains limited despite that the authors claim it. The Green’s-function formulation is largely symbolic and just an extension from GNO [1].

2. Technical novelty is marginal. The main technique is to introduce a position-dependent correction term that is not equivariant to SFNO. The method could be viewed as an equivariant model plus an additive bias rather than a fundamentally new operator framework. The reframing via Green’s functions does not introduce new computational or representational concepts beyond existing spherical spectral models. Although, in FNO/SFNO, the grid padding can be viewed as a position-dependent correction term to some extent. 

3. The SSWE dataset still has rotational symmetries. Despite using the same SWE dataset, the settings are different from that of SFNO. 

4. The motivation behind the third claimed contribution, the multi-scale spherical network with a hierarchical architecture, is unclear to me. Typically, U-Net–style up- and down-sampling structures (see Appendix B.2, I believe that it's just an U-Net) are introduced to address the inability of local operators (e.g., standard convolutional layers in CNNs) to capture global spatial dependencies. However, models such as FNO, SFNO, and their variants inherently employ global operations. Therefore, it is not evident what benefits this U-Net–type hierarchical design provides in this context. Furthermore, the down-sampling path can introduce information loss and aliasing artifacts, which may further limit its effectiveness.



[1] Neural Operator: Graph Kernel Network for Partial Differential Equations

### Questions
1. What is the quantitative contribution of the bias/correction term alone? How is it better that just padding the grid?
2. What is the computational complexity compared to baselines?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Generalized Spherical Neural Operators (GSNO), a method for learning neural operators on spherical domains. It introduces absolute-position-dependent spectral constraints for modeling real-world complexity on spheres and demonstrates improvements in diffusion MRI, fluid dynamics, and climate forecasting benchmarks over state-of-the-art models.​

GSNO is strongly related to SFNO, which introduces the original approach based on the Driscoll-Healy ocnvolution theorem on the sphere, that the authors describe as the Greens-function based approach. The main novelty comes from the inclusion of a position-dependent bias term to handle anisotropy and boundary effects. GSHNet is a hierarchical, multi-scale network architecture that combines spectral domain learning with geometric up/down sampling for improved global feature representation.

Given the improvements that the paper makes, I do think that it can be considered for publication, but I strongly suggest the authors to adapt the language around the Green's formulation. While it is a helpful motivation, it ultimately motivates the existing SFNO architecture. The strong connection should be clearly pointed out and the authors should clearly communicate that the paper is about the inclusion of an extra bias term in spectral domain and how this can be motivated.

As such, my main concerns for this paper revolve around the messaging, which seems to inflate novelty and obscure the conceptual roots found in SFNO.

### Strengths
- improved architecture based on SFNO, which includes a spectral bias term which can address the limitations around the isotropic kernels in the original SFNO work

- adds extra theoretical motivation with the Green's function viewpoint to the existing SFNO literature

- the paper is well-written and has well-motivated experiments

- shows improvement on various, simplified datasets (spherical shallow water equations, weather on low resolution, DMRI example)

### Weaknesses
- This work has a strong connection to SFNO and mainly introduces an extra bias term which allows it to alleviate the constraint to isotropic filters. Unfortunately, the introduction and the text in 4.1 are misleading and make it sound as if a) this paper introduces an entirely new approach based on the Green's formulation b) SFNO only replaces the FFT with an SHT, not leveraging the Driscoll-Healy convolution theorem. Given that "Green's formulation" advertised in the title and derived in 4.1 is the original SFNO, the messaging in both sections and perhaps the title should be adjusted.

- A much more honest exposition would be that this work builds on top of SFNO, introducing an extra bias term in spectral domain which breaks the isotropy constraint and leads to better performance. While this may seem like a "less novel"  exposition it puts the research clearly into the context of the existing literature.

- no code available for review, therefore limited reproducibility at time of submission.

- Added complexity from dual spectral components increases computational cost (modest runtime and parameter overhead).​

- Interpretability and ablation analyses mainly focus on the additive separation, which may not generalize.​

- Reproducibility tested only on select public datasets at limited resolutions.

### Questions
- Is the hierarchical approach of GSHNet really necessary - there were some initial works using U-Nets based on the SFNO operator in the atmospheric science community and those did not seem to improve performance beyond that of the standard Resnet like SFNO backbone. This seems to be at odds to your ablations with SHNet with the SFNO operator. Can you comment more on this?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Spherical neural operators (SNOs) map functions to functions on the sphere. To account for the spherical sampling, current SNOs use spherical rotation equivariant network layers to do this mapping. However, these equivariant networks are (a) not expressive enough for certain applications and (b) cannot model position-dependent features that are beneficial in structured datasets.

Submission 7686 develops an SNO framework that adds a non-equivariant learnable bias term that is position-dependent. This way, the learning framework can decide how much to weigh strict equivariance vs position dependency. Some preliminary experiments on weather forecasting, shallow water dynamics, and fiber orientation estimation in diffusion MRI are presented.

### Strengths
- The papers makes a good case for why such an approach that uses equivariant layers alongside components that are position-dependent are needed and sets up the needed motivation in the Introduction well.
- Most of the methodological steps are very clearly laid out in the methods and illustrations.
- Its experiments tackle a good, wide breadth of potential use cases.

### Weaknesses
(In no particular order)

### Methodological concerns and questions:

*Overstretched technical novelty (1)*: The core idea of the paper is a learnable tradeoff between two terms corresponding to strict equivariance and absolute position dependence in features, respectively. However, this soft equivariance idea has been explored quite extensively in the literature; see the [residual pathway priors](https://proceedings.neurips.cc/paper/2021/file/fc394e9935fbd62c8aedc372464e1965-Paper.pdf) paper as an example. A literature search for "soft equivariance constraints" will return many relevant pieces of related work, such as [1](https://proceedings.neurips.cc/paper_files/paper/2024/hash/98082e6b4b97ab7d3af1134a5013304e-Abstract-Conference.html), [2](https://arxiv.org/abs/2201.11969), and more.  

Of course, this particular formulation, specifically designed for spherical neural operators, is new (as far as I am aware) and is appreciated, but previous work that presents similar additive approaches with learnable equivariance tradeoffs should definitely be acknowledged within the paper.

*Overstretched technical novelty (2)*: L077--L081 is written in a way that makes it sound that studying Green's function on the sphere with spherical harmonics is a novel contribution. I'm assuming that this is not what is meant (spherical harmonics are widely used for this), and what is meant is that "we use Green's function to develop a spherical neural operator network". If not, please clarify.

*Main technical descriptions in Sec 4.2:* Sec 4.2 is the primary contribution of the paper, yet a few things in it are not clear. For example, it is not described how it is possible to drop the index m in G^1 between equations 17 and 18 without any loss of generality.  Further, what are the convolutional layers doing in L307--310, and why are they used? Are they just regular convolutions? Do they change any equivariance properties? For example,  in the Diffusion MRI experiments, adjacent voxels are stacked as separate channels (unclear why), so it is important to justify their usage and define their properties in context.

*Missing analysis of importance of position dependence:* The paper does not present an analysis of whether modeling position dependence actually improves performance due to position dependency or simply because of higher capacity. While a vague description of ablations is included, they simply report performance numbers, so it is unclear if absolute position is relevant here.

Further, position dependency leaks into all equivariant deep networks due to aliasing ([1](https://arxiv.org/abs/2210.02984)), and even the original [S2CNN](https://arxiv.org/pdf/1801.10130) paper indicated that it was not perfectly equivariant in Sec 5.1. Please reconcile the fact that equivariant spherical networks in practice still learn position-dependent features with the explicit positional modeling in this work and show that the proposed method actually exploits positions specifically (and doesn't just have better performance.)

### Experimental concerns and questions:

*Major missing details:* The paper provides no details about what is being reported, how baselines were chosen, and how/if they were tuned, nor any reasoning behind the experimental design. For example, what is ACC in Table 3? Accuracy is meaningless in Diffusion MRI analysis without context. No ablation details are reported in the main paper either. I read the appendix corresponding to the experiments and found that to also be light on details. This aspect of the paper definitely needs expansion.

*Computational complexity:* The proposed method requires several forward and inverse spherical fourier transforms, which are highly computationally intensive and thus limit network capacity. I realize that this is how the original SFNO paper was also formulated, but other works, such as DeepSphere, have shown that these transforms are not strictly necessary for equivariance. Regardless, all experiments should also report training and testing runtimes for all benchmarked methods (and not just SFNO, as in the appendix).

*Diffusion MRI experiments:* Looking at the appendix, the paper makes a few odd statements and choices in its experiments. 
- It states that diffusion MRI is captured on a HEALPIX grid -- this is not the case; the spherical grid points (b-vector patterns) are study-specific. 
- There are specific synthetic benchmark datasets with exact ground truth for diffusion MRI such as Fibercup, the ISMRM tractometer challenge, and other recent challenges. One of those datasets should be added to measure gold-standard performance vs. the current silver-standard experiments on HCP, as in recent work in equivariant FOD estimation ([1](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5d4f5a2de6320641566be8722d5f78dc-Abstract-Conference.html)).
- It significantly downsamples the angular resolution of the HCP dataset from hundreds of spherical sampling points to just 32 and also throws away two entire shells "to emulate clinical conditions". This is not well justified, as (a) it does not demonstrate that the proposed method has the memory efficiency to extend to the hundreds of spherical sampling points used in typical research data and (b) at this "clinical" resolution, the FODs have no asymmetry and are much simpler structures. 

### Presentation concerns and questions:

*Characterization of previous work:* In the introductory paragraph corresponding to L053, the paper somewhat uncharitably characterizes previous work on spherical neural operators as "not rigorous and constructed by analogy". As far as I remember, the original SNO paper did construct the SNO from first principles. If specific non-rigorous arguments are being made in those papers, the submission should be specific about what it is referring to.

*Significant portions can be abbreviated:* To add much more experimental detail to the main paper, several portions can be removed or abbreviated. For example, 
- The contributions list in the introduction is redundant and just restates the paragraph preceding it
- Section 3 can be significantly shortened, as these are plain fundamentals
- Section 4.3 is a long way of saying "we constructed a UNet with our operator layers"; it can also be shortened.


### Minor/misc. comments:
These comments do not impact my score:
- *Missing sanity check:* A common practical strategy in spherical equivariant learning on datasets with position-dependent structure (e.g., preferred orientation in natural 360-degree video) is to use the standard equivariant layers for most of the network and "break" equivariance near the output with 1--2 standard non-spherical layers (e.g., appendix E in [1](https://arxiv.org/pdf/2006.10731)). A similar strategy can be used with SNOs, and this paper would benefit from investigating whether the proposed method outperforms this simple baseline.
- Following up on the above, spherical image segmentation might have been an interesting use case for the proposed method.
- In the diffusion MRI experiments, FOD-Net is an odd choice to use the proposed GSNOs in, as the spherical harmonics are only used in a single portion of the network. In contrast, spherical UNets have been used for FOD estimation in [1](https://hal.science/hal-02946371/document), [2](https://arxiv.org/abs/2102.09462), among others; One of them would likely be a better fit for this analysis as the UNet architecture can simply be replaced by the proposed GSHN. 
- Minor typos. e.g. L177-178 "denoting" --> "denotes"; "Equation equation"

### Questions
In my opinion, the paper reads as incomplete in its current state (clarifications of technical details, depth of technical contributions, missing experimental information, etc.). However, IMO these are fixable issues and I am looking forward to the discussion period to clarify these points before finalizing a score. For specific questions, please see above.

### Soundness
3

### Presentation
3

### Contribution
3
