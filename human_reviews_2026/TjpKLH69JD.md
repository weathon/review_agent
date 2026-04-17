# Equivariant Diffusion for The Inverse Radar Problem

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Reconstructing 3D geometries from their radar signal is a complex inverse problem,
often involving unique domain expertise and manual steps. Although deep learning
approaches have emerged to address the automation challenges of this problem,
there are still significant performance gaps due to non-unique solutions and partial
observability. In this work, we explore the role of equivariant modeling in helping
reduce uncertainty over potential 3D shape distributions measured via partial radar
signals. We present a radar-conditioned equivariant latent diffusion model that
uses a two-stage training approach. In the first stage, we learn equivariant latent
representations of 3D shapes by training a SO(3)-equivariant encoder-decoder
model using vector neuron architectures. During the second stage, we train an
SO(3)-equivariant denoising diffusion model that operates over the learned latent
geometry representations. We introduce an equivariant FiLM layer that enables
conditioning of our diffusion model in Irreps space and thus ensures rotational
equivariance throughout the generation process. Finally, we ensure equivariant
latent representations of the conditioning radar signal by using a spherical CNN
model. We show that our model predicts plausible 3D geometries consistent with
the observed radar signatures. In addition, we demonstrate improved performance
over other competitive non-equivariant baseline methods with respect to one of the reconstruction quality metrics and a sample diversity metric under full
observability settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a 3D reconstruction method from radar signals, which is based on a latent diffusion model composed of an SO(3)-equivariant autoencoder and an SO(3)-equivariant denoiser model. Experiments on synthetic (Frusta) data show that the proposed model can obtain good shape reconstruction, outperforming baselines that do not use both an equivariant autoencoder and denoiser. The paper introduces a conditioning network that can incorporate information about the radar signal into the diffusion denoising network in an equivariant manner, leveraging ideas from spherical CNNs.

### Strengths
- Presents a new diffusion approach for solving a challenging radar inverse problem involving spherical measurements and 3D point cloud signals. 
- Designs a model where every component is equivariant: autoencoder, diffusion denoising model, and measurement conditioning module.
- Experiments on synthetic data show that the model can recover roll-symmetric 3D shapes from the measurements.

### Weaknesses
- The presentation of the paper should be improved
    - the radar forward model is not explained, making the paper not very accessible to a general ML audience. I would expect the paper to include a more self-contained mathematical description of the forward problem, at least in an appendix.
    - mathematical notation is not consistent: 
           - subindices are sometimes written with italics (eg $d_{\mathrm{radar}}$, sometimes not (eg $L_{max}$)
           - Mathbb is lacking on some spaces $x\in R^3$
           - the use of uppercase/lowercase doesn't seem to follow any pattern. I would expect vectors in lowercase and matrices in uppercase.
           - the $\mathrm{SDF}$ equation has double parenthesis, and the inner product is not well defined for matrices.
    - citation style is used inconsistently: \citep and \cite are not used correctly in many places in the paper.
           - I believe the presentation of the method could be improved: at the moment, the presentation is divided across sections 3, 4 and 5, and feels a bit repetitive, where none of the sections goes into enough detail.

- Only synthetic experiments are performed, and it is unclear if the method can work well in realistic settings: the network is trained and evaluated on similar simplistic 3D point clouds, and the measurement process is synthetic. It is not clear from the paper whether measurement noise is considered.

- Another figure should be included illustrating the sampling diversity in the case of partial measurements, which is the main motivation of the paper for using diffusion models.

### Questions
- Why do you use different equivariant architectures/paradigms for the autoencoder and the denoising model? Why not using the same paradigm?
- Why are radar measurements defined on the sphere? This should be better explained to a general ML audience in the paper.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the reconstruction of 3D objects, particularly focussing on partial radar signals. The authors propose a two-stage, SO(3)-equivariant pipeline. Stage 1 involves training an SO(3) equivariant encoder–decoder model which is used to generate latent encodings from point clouds. Stage 2 trains an SO(3)-equivariant denoise operating on the latent encodings and is conditioned on spherical radar embeddings via a novel equivariant FiLM layer; conditioning features come from a spherical CNN. The full generative pipeline is end-to-end SO(3)-equivariant. The authors run a series of experiments on a simulated Frusta dataset due to a lack of high-frequency real-world data. The model is compared to two baselines, which do not have end-to-end equivariance constraints using a minimum matching distance (MMD) and F1 score for reconstruction accuracy, and total mutual difference (to show diversity/uncertainty benefits). The authors model beats the two baselines in terms of F1 score for the fully observed radar signals, however performs slightly worse than both baselines (for F1 score) for partially observed radar signals. Their equivariant model does have larger sample diversity as shown by the TMD scores, implying it is more uncertain about the true shape. Overall the method is interesting, performs at least as good at the baselines in the fully observed case, and produces richer sample diversity in all cases.

### Strengths
The end-to-end equivariance pipeline is well motivated for the inverse radar mapping problem, and incorporating this into both the encoder/decoder model, and the diffusion model is a good idea. The equivariant FiLM layer is also a nice addition to allow the full pipeline to be equivariant to SO(3) rotations.

One would expect in under-sampled regimes (partial observations) that there should be a larger amount of uncertainty about the true shape, hence the diversity of sampled from the diffusion model would be higher. This is indeed the case as shown in table 1.

The improved efficiency over the Diffusion-SDF model is substantial, although the non-equivariant baseline is roughly as efficient as this proposed model too.

In figure 5, the generated sampled of the equivariant diffusion model appear much closer to the ground truths than the baselines.

### Weaknesses
Firstly, from table 1 the equivariant diffusion samples appear to not perform any better than the baselines on average. Specifically while for the full observations the F1 performance in the best, this is only be a very minor amount, and the MMD scores of the equivariant model are the worst of all models for both the partial and full observed datasets. 

Second, in the abstract and introduction the authors mention the difficulty with partially observed data, and dealing with the uncertainty that arises from this. However their results in the partial test case are significantly worse for reconstruction accuracy (MMD and F1) than both baselines. The sample diversity in higher for the authors model which I agree one would think is due to the uncertainty of the true shape, but the significantly lower (0.1549 vs 0.2807) F1 accuracy is not insignificant. 

Third, the dataset is synthetic which is understandable and not necessarily a bad thing, however this specific dataset used contains only axially symmetric data. This seems like it would potentially help the performance of SO(3)-equivariant models, while true data may not be symmetric at all. Testing on non-symmetric data (even if still synthetic) would be a benefit here.

Minor issues, figure 5 is not fully explained, is this full or partially observed data? Some in text citations are inconsistently formatted such as this sentence "(Esteves et al., 2018)(Cohen et al., 2018) uses it for 3D
shape classification." Other sections use \citet style.

### Questions
- Is the increase sample diversity hurting the reconstruction accuracy? I.e is there a way to test if the model is perhaps being under-confident in the shape predictions?

- Do you have tests of your model on non-symmetric datasets compared to the baseline models?

- Do you have any reasons or results as to why you model seems to perform worse on the MMD baselines even for the full observation testing?

- Is there results on any real-world datasets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a radar-conditioned SO(3)-equivariant latent diffusion model for reconstructing 3D volumes from radar signals. Following the latent diffusion paradigm, the proposed approach is trained in two stages. First, an SO(3)-equivariant encoder–decoder is trained using vector neuron architectures to learn a structured latent space. Then, an SO(3)-equivariant diffusion model is trained to operate within this latent space. To enable rotationally equivariant conditioning, the authors propose an equivariant FiLM layer that modulates the diffusion process directly in the Irreps space. Experimental results on a simulated dataset demonstrate that the proposed method sometimes outperforms existing baselines.

### Strengths
The paper is well-written and easy to follow. The main idea and contributions are explained well.

### Weaknesses
The paper has several limitations that make it unsuitable for acceptance at this conference.

1. The contribution of the paper seems incremental. While SO(3)-equivariant neural networks are well-established in the literature, this work primarily applies them within a diffusion model framework without introducing a fundamentally new concept or significant methodological advance.

2. The experiments are limited to a simplified simulated dataset, which does not convincingly demonstrate the generalization ability or practical value of the proposed approach. Moreover, the baselines are inadequate; the authors should compare against a broader range of methods, including both standard deep learning approaches and traditional radar imaging methods. Evaluating the method on more challenging and realistic datasets for shape reconstruction would improve the paper.

### Questions
In the experimental section, the authors state: “Due to the lack of large real-world training data for the high-frequency radar setting, we use a physical optics approximation method to simulate radar responses for a variety of mesh objects.”
This raises a fundamental concern regarding the practical relevance of the proposed approach. If no real-world training dataset is available, it is unclear how this supervised method could be applied in real scenarios. Without any experiments on real radar data, it is hard to assess the real-world performance, generalization, or usefulness of the proposed model.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a radar-conditioned SO(3)-equivariant latent diffusion model for reconstructing 3-dimensional shapes from radar signals. The aim is to use group equivariance to reduce uncertainty in the solution arising from incomplete observations and inherent radar noise (irreducible uncertainty), thereby generating more plausible, diverse and consistent three-dimensional shapes (represented implicitly via SDFs). The proposed method employs a two-stage training strategy: First, SO(3)-equivariant latent codes are learnt for the 3D shape. Then, an isometry-preserving denoising diffusion model is trained on these latent representations. Specifically, a SDF serves as the output representation, with physical optics (PO) simulations modelling high-frequency radar responses and range profiles (spherically parameterised) used for conditioning. The denoising network employs an SO(3)-equivariant MLP implemented via e3nn, while the radar encoder uses a spherical CNN. Finally, the FiLM layer is modified to preserve equivariance in the irreps space.

### Strengths
1. The idea of leveraging symmetry biases explicitly in reconstruction and using diffusion models for uncertainty estimation is interesting. Ideally, embedding SO(3) equivariance as a prior throughout the entire generation pipeline (i.e. the encoder, denoising network and conditioner) would reduce sample complexity, enhance data efficiency and ensure rotational consistency (where rotation constitutes an inherent symmetry of the problem).

2. The proposed 'equivariant FiLM layer' performing conditional modulation in irreps space is an interesting solution that combines common conditioning techniques (FiLM) with an approach based on representations. For high-frequency radar, the integration of SDF representations with PO models has led to the construction of a relatively comprehensive simulation dataset (Frusta dataset) and evaluation workflow.

### Weaknesses
1. The mathematical proof or sufficient justification for the equivariance of FiLM is inadequate. For example, line 249: ‘using fully connected tensor product and channel-wise addition to preserve equivariance’, but lacks a formal proof or explanation of the conditions under which FiLM (scaling + bias) still satisfies strict SO(3) on Irreps (e.g., which terms must be scalars, how vectors are handled, etc.). 

2. The equivariance argument for embedding time step t_embed remains unclear: embedding the time step as a scalar via tensor product requires explicit clarification of how this operation interacts with different-order representations ($l=0,1$) while preserving isometry.

3. The paper brings together three well-studied ideas: equivariant neural networks for three-dimensional representation, latent diffusion modelling and spherical convolutional neural networks (CNNs). There has been limited innovation in either the theoretical or engineering aspects.

### Questions
Ablation experiments are necessary, for example, the independent contributions of the equivariant FiLM, equivariant denoiser, and equivariant encoder were not demonstrated.

How can FiLM be built for other symmetry groups, such as permutation or a mixture of rotation and shifts? Does the design of FiLM's symmetry group strictly depend on (or should match) the symmetry of the radar data?

### Soundness
3

### Presentation
3

### Contribution
3
