# ProteinAE: Protein Diffusion Autoencoders for Structure Encoding

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Developing effective representations of protein structures is essential for advancing protein science, particularly for protein generative modeling. Current approaches often grapple with the complexities of the $\operatorname{SE}(3)$ manifold, rely on discrete tokenization, or the need for multiple training objectives, all of which can hinder the model optimization and generalization. We introduce ProteinAE, a novel and streamlined protein diffusion autoencoder designed to overcome these challenges by directly mapping protein backbone coordinates from $\operatorname{E}(3)$ into a continuous, compact latent space. ProteinAE employs a non-equivariant Diffusion Transformer with a bottleneck design for efficient compression and is trained end-to-end with a single flow matching objective, substantially simplifying the optimization pipeline. We demonstrate that ProteinAE achieves state-of-the-art reconstruction quality, outperforming existing autoencoders. The resulting latent space serves as a powerful foundation for a latent diffusion model that bypasses the need for explicit equivariance. This enables efficient, high-quality structure generation that is competitive with leading structure-based approaches and significantly outperforms prior latent-based methods. Code is available at https://github.com/OnlyLoveKFC/ProteinAE_v1.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposed PROTEINAE, a protein diffusion autoencoder that learns compact, continuous representations of protein structures directly from 3D backbone coordinates in Euclidean space (E(3)). 

Unlike prior approaches that depend on SE(3)-equivariant models, discrete tokenization, or multiple loss functions, PROTEINAE simplifies training with a single flow-matching objective and a non-equivariant Diffusion Transformer (DiT) architecture. The model efficiently encodes protein backbones into a latent space through a bottleneck design, enabling high-fidelity reconstruction and scalable downstream modeling. On benchmarks such as CASP14 and CASP15, PROTEINAE achieves state-of-the-art reconstruction accuracy, substantially outperforming discrete and equivariant autoencoders.

The authors argue that strict equivariance may not be essential for effective protein representation learning, citing recent advances such as AlphaFold3 and Proteina, which also employ non-equivariant architectures. However, the paper could benefit from a more comprehensive discussion of related work on 3D protein autoencoder design, as well as a deeper theoretical justification for the choice of a non-equivariant model.

### Strengths
## Strengths

- This paper introduces a non-equivariant diffusion autoencoder that learns continuous protein structure representations directly from 3D coordinates, simplifying model design and training compared to SE(3)-equivariant or discrete approaches. But as mentioned before, the paper could benefit from a deeper theoretical justification for the choice of a non-equivariant model.

- The proposed method achieves state-of-the-art reconstruction accuracy on CASP14/15 and competitive structure generation quality relative to more complex diffusion models.

- Computational efficiency: The latent diffusion framework offers good speed and memory advantages over previous diffusion methods.

### Weaknesses
## Weaknesses:

- This paper lacks sufficient discussion of prior work on 3D protein backbone structure autoencoder design. For example, the model LatentDiff is compared in Table 2, but it is not discussed in the related-work section. It shares similar design ideas with the proposed method, including an autoencoder that reduces the latent diffusion modelling burden, shrinks the modelling space in terms of length, and offers computational efficiency.

- The authors argue that strict equivariance may not be essential for effective protein representation learning by citing recent advances such as AlphaFold3 and Proteina, which also employ non-equivariant architectures. However, the paper would benefit from a deeper theoretical justification for choosing a non-equivariant model.

### Questions
As listed above in weaknesses.

### Soundness
3

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
This paper proposes a new generative model for protein tertiary structure generation and reconstruction. Most existing approaches rely on coordinate prediction models or diffusion models based on stochastic noise denoising, which are computationally expensive and often struggle to maintain structural consistency and reconstruction accuracy. In contrast, this study combines the Flow Matching framework with a Transformer-based network, achieving continuous and stable structure generation. Unlike conventional SE(3)-equivariant models, the proposed method intentionally adopts a non-equivariant architecture on E(3) space. This design eliminates the need for explicit and computationally costly handling of rotations and translations, while still maintaining effective geometric robustness through structure alignment, pairwise bias based on relative distances and angles, and data augmentation.

The proposed framework consists of three stages. In the first stage, the model takes the backbone atoms of each residue as input and compresses them into residue-level representations using an All-Atom Attention Encoder, which is designed to capture both spatial dependencies among residue pairs and primary sequence information. In the second stage, these features are processed by an Encoder–Decoder structure based on a Diffusion Transformer (DiT), which learns the global 3D correlations of the entire protein. Pairwise representations are incorporated as attention biases, enabling the model to encode global dependencies that reflect inter-residue distances and angles. The third stage introduces the Protein Latent Diffusion Model (PLDM), which treats the latent representations of residue sequences obtained by the encoder as latent token sequences. After compressing both the sequence length and feature dimension, PLDM learns the diffusion process directly in the latent space. Rather than performing stochastic noise denoising, PLDM is designed as a Transformer that predicts the velocity field of latent vectors based on Flow Matching. To preserve the sequential order while integrating temporal embeddings, positional encodings are applied to the latent tokens.

Experimental evaluation on protein structures from the AlphaFold Database demonstrates that the proposed method achieves smoother and more stable structure generation than conventional diffusion models, with higher reconstruction accuracy as measured by RMSD and FAPE. The generated structures also preserve physical consistency, confirming that the proposed approach enables efficient and high-precision protein structure generation.

### Strengths
The main strength of this paper lies in its novel design choice to intentionally train the model in a non-equivariant manner on E(3) space, rather than relying on conventional SE(3)-equivariant frameworks. By doing so, the authors successfully avoid the computational cost of explicit rotation and translation handling while maintaining effective geometric robustness through structure alignment, pairwise bias based on relative distances and angles, and appropriate data augmentation. This represents a meaningful and practical contribution to the field of protein structure generation.

Another notable strength is the use of a simple Flow Matching loss, which removes the need for complex KL or reconstruction losses commonly required in traditional diffusion models. This design enables a smoother and more stable generation process, allowing structures to evolve naturally from noise with higher accuracy and computational efficiency than probabilistic baselines. By integrating Flow Matching with a Transformer architecture, the paper reformulates protein structure generation as learning a continuous flow rather than stochastic denoising, effectively preserving structural continuity and physical consistency while enabling efficient generation in the latent space.

### Weaknesses
The authors clearly state this limitation, but the proposed model still has difficulty handling very large proteins and multi-chain complexes. Although using a non-equivariant design on E(3) makes the computation much faster, it also creates concerns about whether the model can keep geometric consistency and produce physically reliable results, especially for rotation and translation. Because of this, the method may not yet be suitable for very large or complex protein structures. As the authors mention, adding more geometric constraints or partially equivariant mechanisms in the future could help overcome this problem and further improve the model’s applicability.

### Questions
Do the authors have any ideas or future plans for incorporating additional constraints or introducing some form of equivariant mechanism to extend the applicability of the proposed model to larger and more complex protein structures?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PROTEINAE, a protein diffusion autoencoder mapping protein backbone coordinates from E(3) into a continuous, compact latent space for protein modeling and generation. PROTEINAE uses a non-equivariant Diffusion Transformer and is trained with a single flow matching objective to simplify the training objective. PROTEINAE achieves better reconstruction quality and high-quality structure generation that significantly outperforms prior latent-based methods.

### Strengths
- PROTEINAE uses a non-equivariant autoencoder based on Diffusion Transformers that operates directly on atom coordinate without the need to considering complicated equivarience.
- Performance of PROTEINAE is better than latent-based methods and competitive with structure-based methods.
- The model is trained using a simple flow matching loss, which makes the training objective easy to design.

### Weaknesses
- Structure-based method is still better than the latent-based model, it’s not clear if the latent model has potential to achieve better performance than structure-based method. Could authors elaborate more on that?
- PROTEINAE doesn’t consider equivariance to simplify the design but would that lead to generating non-physical structures?
- The comparison missed some important baselines, La-proteina and Proteina. Proteina also uses non-equivariant model design and La-proteina is a latent model based on Proteina.

[1] Geffner, Tomas, et al. "Proteina: Scaling flow-based protein structure generative models." arXiv preprint arXiv:2503.00710 (2025)\
[2] Geffner, Tomas, et al. "La-proteina: Atomistic protein generation via partially latent flow matching." arXiv preprint arXiv:2507.09466 (2025).

### Questions
Please refer to the weakness section

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
In this work, the authors propose using a non-equivariant diffusion autoencoder called ProteinAE to encode protein structure by directly mapping backbone coordinates into a continuous latent space. And they show the performance of their model compared to recent generative frameworks along with its performance as a structure embedder for a downstream prediction task. Overall, the field has largely treated equivariance as gospel—driven by the desire to model structure from first principles and inspired by the success of AlphaFold2. However, AlphaFold3 recently shifted this paradigm. This work presents an interesting application of non-equivariant modeling for generative purposes, while also offering a fast structural encoder, though it still requires further validation and benchmarking to fully establish its advantages.

### Strengths
- Not imposing equivariant restriction on the protein structure representation enables a much more computational efficient framework for modeling protein structure. 
- Superior benchmark performance of the trained representation layers for physiochemical property prediction. 
- Authors provided benchmarks against some of the newer models for structure generation. Compared to RFDiffusion model which has been extensively validated in the lab, authors report ProteinAE while performing slightly worse than RFDDiffusion for designability and diversity, the sampling time is much more efficient. This can be useful for some protein design pipelines. 
- The model also seems to perform well at reconstructing structure based on the CASP14/15 benchmarks.

### Weaknesses
- The paper would benefit from clarifying some sections [see the question below]. 
- While the model shows promise, the included benchmarks are across a limited range of tasks, specifically for protein structure encoding and its performance in downstream tasks.

### Questions
- In the introduction, authors mention that other models use SE(3) representations and are therefore slow. However, SE(3) representations are also essential in protein structure modeling specifically in generative protein design, so this point needs to be explained more carefully — perhaps discussing when SE(3) equivariance is beneficial versus when it introduces unnecessary computational cost. 
- Authors include CASP14 and CASP15 in their benchmark for structure reconstruction quality and compare against ESM3 etc, It would be good to see the performance on CAMEO as well for the same period as ESM3.  
- Since the model is positioned also as a faster alternative for encoding structural information, it would strengthen the paper to demonstrate its applicability beyond the current benchmark on physiochemical property prediction. For example, evaluating its performance on other downstream structure-based prediction tasks, such as GO-term prediction, could better illustrate the practical impact of the approach.

### Soundness
3

### Presentation
3

### Contribution
2
