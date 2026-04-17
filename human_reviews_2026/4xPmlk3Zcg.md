# CoVAE: Consistency Training of Variational Autoencoders

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Current state-of-the-art generative approaches frequently rely on a two-stage training procedure, where an autoencoder (often a VAE) first performs dimensionality reduction, followed by training a generative model on the learned latent space. While effective, this introduces computational overhead and increased sampling times. We challenge this paradigm by proposing Consistency Training of Variational AutoEncoders (CoVAE), a novel single-stage generative autoencoding framework that adopts techniques from consistency models to train a VAE architecture. The CoVAE encoder learns a progressive series of latent representations with increasing encoding noise levels, mirroring the forward processes of diffusion and flow matching models. This sequence of representations is regulated by a time dependent $\beta$ parameter that scales the KL loss. The decoder is trained using a consistency loss with variational regularization, which reduces to a conventional VAE loss at the earliest latent time. We show that CoVAE can generate high-quality samples in one or few steps without the use of a learned prior, significantly outperforming equivalent VAEs and other single-stage VAEs methods. Our approach provides a unified framework for autoencoding and diffusion-style generative modeling and provides a viable route for one-step generative high-performance autoencoding. Our code is available in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduices CoVAE which builds upon the VAE architecture but modifies both the encoding and decoding processes to ensure temporal consistency across latent representations along the learning phase.
•	The encoder learns progressively noisier latent representations, controlled by a time-dependent noise schedule β(t). At low noise levels (small t), the model captures detailed structure; as t increases, the latent codes gradually transition to Gaussian noise, providing a continuous interpolation between structured and random representations.
•	The decoder is trained using a consistency loss that enforces agreement between predictions at adjacent time steps. This mechanism enables the model to learn denoising dynamics similar to those in diffusion models, but without requiring iterative multi-step sampling.

### Strengths
•	Single-stage training (faster, simpler)
•	Fast sampling without leaning an adaptive priori or learning the posterior distribution of the embeddings. 
•	Disentangled latent space for image manipulation

### Weaknesses
The model cannot easily compute a tight Evidence Lower Bound (ELBO), which complicates likelihood-based evaluation and comparison with classical VAEs. 
Its performance depends on several empirically tuned hyperparameters. 
While it closes part of the gap between VAEs and diffusion models, it still lags behind the best direct diffusion approaches in sample fidelity and comparisons with the latent diffusion models are lacking. 
Other Latent SDE approached should also have been considered for comparison.

Overall the idea looks interesting to have a "diffusion like" model to better learn the latent prior without the diffusion forward-backward equations to solve but comparisons with this literature is missing.

There is a typo in the pseudo code Algorithm 1 for the cm likelihood: , should be -.

### Questions
Is the multistep CoVAE remaining in interesting areas of the latent space when moving around? In particular, the experiments have been performed with a lot of data. Is this method "scalable" for low amount of data?

Is the use of patch-based adversarial loss interesting for the β-VAE itself. If yes, can you show the increase of performance of your combined approach with CoVAR vs the combination with the β-VAE?

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
4

### Summary
The paper introduces consistency training for Variational AutoEncoders (CoVAEs). In contrast with two-stage approaches based on generative models over dimensionality reduction, CoVAE is a one-stage approach that learns a sequence of latent representations with a time dependent parameter scaling regularization and trained using consistency loss that is related to standard the VAE. Experiments on MNIST, CIFAR-10 and CelebA comparing performance with baseline models to illustrate the quality of images generated.

### Strengths
Strengths of the paper include:
- Concise, clear mathematical introduction of VAEs, Diffusion models, Consistency models, and the proposed CoVAE approach.
- Detailed experiments including multiple datasets and multiple baseline models
- Detailed and fair discussion of related work
- Clear statements of limitations of the current work that identify important problems to address in future research

### Weaknesses
The main weaknesses of the paper are twofold; 
- First, the datasets are limited to MNIST, CIFAR10, and CelebA
- Second, the models compared with are valuable but not state-of-the-art in terms of performance 


Minor issues: 
- Small typo in Figure 1 caption: Consistenct
- Line 229 "Form small time steps"
- Figure 2 is confusing. I suggest you explain what the objects in the future are one by one, starting from the left. E.g. is "In Diffusion and Consistency" about the first picture or the first two? Epsilon_psi is in in the figure but is in the caption. "in this case we use a dashed line" This is confusing. 
- Figure 3, the caption never mentions t. 
- "Form small time steps, the samples from each class are embedded in well separated areas, while they gradually become more random
as time increases." This is unclear. Is t representing size of time steps or time? (Or are they confounded.)

### Questions
Please see weaknesses. Specifically, it seems important to better understand the implications of the current approach for the state-of-the-art in image generation?

### Soundness
3

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
The paper proposes CoVAE, training VAEs using consistency loss, in similar fashion as the consistency models. The encoder is parametrized with the time step in addition to the input, and the decoder is trained to map the latent representations from any time step to the original image. The proposed method shows improved performance among the state-of-the-art VAE methods, but still lags behind diffusion models.

### Strengths
- The proposed idea in CoVAE to use consistency training in VAEs is novel and interesting.
- The performance among VAEs is better, and CoVAE also offers the option to trade off efficiency and performance with multi-step generation.

### Weaknesses
- There is limited insight into the fundamental difference consistency training brings in VAEs that leads to performance improvements. While iterative denoising is intuitively justified in diffusion-based or consistency models—where the coupling between latent variables and data points is unknown, it is less clear in the case of VAEs, where the latent variable corresponding to a given data point can be obtained through the encoder.
- In Section 2.1, the authors mention the prior hole problem, which is a fundamental problem with VAEs when used for generation. It is not discussed whether CoVAE mitigates the problem or if CoVAE has any impact on it.
- At lines 372/373 it is mentioned that a patch-based adversarial loss is used, and it should be clear in the performance tables about the role of this additional loss. While Table 1 has  CoVAE with and without patch-based losses, Table 2 does not include it.
- I would recommend using higher resolution images and the Imagenet dataset for experiments, as it is a fundamental benchmark for image generation.

I believe the paper would benefit from deeper insights into why consistency training is necessary and what conceptual effect it introduces. Additionally, the experimental section could be strengthened through refinements involving higher-resolution datasets, such as ImageNet. I would be open to reconsidering my evaluation if these aspects are addressed.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes CoVAE, a single-stage generative autoencoding framework that unifies a time-dependent \beta-VAE with consistency training. The encoder produces progressively noised latents via a time-dependent KL weight \beta(t); the decoder is trained with a latent consistency loss (bootstrapping adjacent times) with a denoiser-style term. This enables one- or few-step generation without a learned prior. On MNIST, CIFAR-10, and CelebA-64, CoVAE improves FID over standard VAE and \beta-VAE baselines and outperforms strong single-stage VAEs (NVAE, DC-VAE); adding a lightweight adversarial term further improves FID and reconstruction.

### Strengths
1. The paper formulates the VAE reparametrization as a time-indexed “forward process” in latent space and replaces standard reconstruction with a discrete consistency objective that bootstraps from early times. The method section and algorithmic details substantiate this bridge.

2. CoVAE generates in one step and can optionally do few-step refinement by re-encoding/re-denoising at intermediate t. This is a practical departure from the common “VAE + latent diffusion/flow” recipe.

3. The paper shows promising reconstruction and generation capacity of CoVAE in experiments. On CIFAR-10, CoVAE (1-step) improves FID over NVAE and DC-VAE. On MNIST, CoVAE simultaneously improves generation and reconstruction over β-VAE.

### Weaknesses
1. A major concern is the applicability of the proposed approach, both to future research and real-world application. While CoVAE aims to unify VAE and the diffusion process for generation tasks in one single stage, it neglects text (or class) conditioning in modeling and implementation for image generation, which is crucial in current generative models. The paper compares CoVAE with standard VAE and demonstrates its advantages. However, standard VAE can be readily used to modeling visual signals in the latent space, for further diffusion-based generation. It is not clear how the time-dependent latents in CoVAE can be refined or utilized in downstream tasks.

2. Experiments show results on relatively low-resolution (up to 64x64) image generation tasks. More convincing results are missing to show how CoVAE performs on high-resolution tasks and how it compares with strong baselines such as GANs.

### Questions
1. Are there principled ways to design \beta(t) and \lambda(t)? How sensitive is empirical performance to their values?

2. What is the compute and inference time of CoVAE? And what about the training efficiency and scalability of CoVAE?

3. Would the generation quality consistently improve if the sampling steps are increased?

### Soundness
3

### Presentation
3

### Contribution
2
