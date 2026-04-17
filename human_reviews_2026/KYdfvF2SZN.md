# Structured Flow Autoencoders: Learning Structured Probabilistic Representations with Flow Matching

- Decision: Accept (Oral)
- Scores: 6, 4, 8, 6

## Abstract
Flow matching is a powerful approach for high-fidelity density estimation, but it often fails to capture the latent structure of complex data. Probabilistic models like variational autoencoders (VAEs), on the other hand, learn structured representations but underperform in sample quality. We propose Structured Flow Autoencoders (SFA), a family of probabilistic models that augments graphical models with conditional continuous normalizing flow (CNF) likelihoods, enabling flow-matching-based structured representation learning. At the core of SFA is a novel flow matching objective that explicitly accounts for latent variables, allowing joint learning of the CNF likelihood and posterior. SFA applies broadly to graphical models with continuous and mixture latents, as well as latent dynamical systems. Empirical studies across image, video, and RNA-seq data show that SFA consistently outperforms VAEs and their structured extensions in generation quality, representation utility, and scalability to large datasets. Compared to generative models like latent flow matching (LatentFM), SFA also produces more diverse samples, suggesting better coverage of the data distribution.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Structured Flow Autoencoders (SFA), a new family of generative models that combine the high-fidelity density estimation capabilities of flow matching with the structured latent representation learning of probabilistic graphical models. The key technical contribution is the Structured Conditional Flow Matching (SCFM) objective, which explicitly incorporates latent variables into the flow matching framework, allowing joint learning of both the conditional likelihood p(x∣z) and the approximate posterior q(z∣x). The paper demonstrates the versatility of SFA across several domains, showing improvements in both sample quality and latent structure interpretability.

### Strengths
1. The paper provides a clear theoretical formulation that unifies two previously distinct paradigms: probabilistic graphical models and flow matching.

2. The Structured Conditional Flow Matching (SCFM) loss is a well-motivated extension of the flow matching objective, effectively bridging high-fidelity neural density estimation with structured latent representation learning.

3. Experiments cover toy 2D densities, images, and single-cell RNA data, demonstrating the method’s flexibility across diverse modalities and latent structural forms.

### Weaknesses
1. The paper discusses several posterior approximation families (time-indexed Gaussians, conditional CNFs, and Gumbel-Softmax for categorical latents )but provides only limited empirical comparison. It remains unclear how sensitive the final performance and stability are to these design choices.

2. Several parts of the pipeline still require solving ODEs (e.g., conditional CNFs), which can be computationally intensive. Providing more detailed analysis of runtime, memory usage, and overall computational complexity would strengthen the evaluation.

3. Most experiments are conducted on MNIST, Pendulum, or relatively small-scale RNA-seq datasets. It is unclear how SFA would perform on larger-scale or more complex datasets (e.g., CIFAR-10 or ImageNet). Scaling SCFM to higher-dimensional inputs or more diverse domains may introduce additional computational and optimization challenges that are not explored in this work.

4. Typos: Line 451 Table 4 --> Table 2

### Questions
1. As shown in Table 1, FM achieves performance comparable to SFA. It remains unclear what concrete advantages SFA provides over FM.

2. Is there a general procedure for determining the specific form of SFA when applied to different types of graphical models?

### Soundness
3

### Presentation
3

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
The authors develop a structured conditional flow matching objective which is used to train their Structured Flow Autoencoder architecture. Likelihoods are similar to conditional continuous normalizing flow frameworks. Their aim is to achieve similar quality of samples to standard flow matching while retaining latent structure. They do this by jointly learning the conditional continuous normalizing flow likelihood and the approximate posterior.

### Strengths
Strengths
A pretty comprehensive suite of small-to-medium scale experiments, with a toy Pinwheel dataset, MNIST, RNA, and a video dataset.

Their method seems to have the fidelity of the flow-based models while keeping the latent structure intact as with the VAEs. 

The structure seems to be well preserved when visualizing the MNIST digits with TSNE, wit SFA-Mixture, SFA, Latent-FM, and GMVAE giving similar groupings for the classes.

### Weaknesses
Weaknesses:

There isn't really one consistent SFA model for testing on the different problems, with SFA, Mixture-SFA, and LDS-SFA, etc. Is there a necessity for there to be modifications to be different for every data set?

Also some terms like LDS-SFA is not properly defined. I can infer that it is SFA on the LDS dataset, but hard to grasp what makes it different.

### Questions
The log likelihood in Table 2 and Table 3 for SFA has value 725.232 and 384.137, which looks surprisingly large. How is this calculated, and any idea why the density is so concentrated?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces structured flow autoencoders, a method that combines the advantages of flow matching (generating high-resolution diverse data ) and VAEs (learning structured latents for the data).  The idea is to jointly learn the likelihood model p_t(.|z) and the posterior p_t(.|x) using a flow matching objective, where the marginal velocity vector v_t(x) is replaced by an expectation over conditional velocity vectors v_t(x | z) under p_t(z | x).  The parametrization of the posterior  p_t(z | x) can be done either with a CNF or with simple parametric families e.g. gaussians.  They introduce several possible generative processes for x | z, either  continuous latent model, latent mixture model, or latent dynamic system, and such a process is set before training. 
They run experiments on pinwheel, mnist, rna-seq and pendulum video data, and show that their method shows improvements compared to the baselines both in terms of (i) the structure of the learned latent space and (ii) the likelihood p(x|z) and the posterior p(z | x).

### Strengths
- very nice and original idea to learn the latent structure with a flow matching objective instead of a VAE and theoretically grounded with the conditional vector field equation,
- Clarity and mathematical / scientifc rigor of of the paper,
- Solid and diverse set of experiments and of evidence,  and diverse set of baselines , nice 2D visualisations.

### Weaknesses
- Regarding the novelty of the eq.5, used in proposition 3.1 : this conditional vector field equation was introduced before in https://arxiv.org/pdf/2302.00482v1 eq 9 (however they are not conditioning on latents, but on the unnoised sample x_1)

-  Regarding the clarity of the toy experiments: Although the theory for the SFM equation is clear, I wasn't able to understand the nature of the learned latent variables for the pinwheel dataset and MNIST dataset

### Questions
Hello authors, 
First, I'd like to start by acknowledging that I couldn't spend as much time as I wanted to on your paper to fully understand the theory / baselines, and that there are some parts for which the theory is a bit vague for me, so apologies in advance if some of the questions/claims are evident. 

- Figure 1 : I don’t fully understand this figure. You train without the color and you want the learned latent variable to be the color? 
- For MNIST / Pinwheel: What is the structure of sampled the latent variable in this example? is it  a predicted  (mixture of ) mean and variance, and I guess they are one-dimensional here? If not what are their dimensions?
- What do the intermediate-time latents z_t  look like, e.g. for MNIST or for pinwheel data ? are they interpretable? 
- In the paragraph "SVAE and SFA comparison", you claim that SFA has an advantage by jointly learning p(x|z) and p(z|x) , whereas SVAE has a "generation-latent learning trade-off". I don't understand this, could you elaborate more on it and why it is a drawback compared to SFA? 
- Could you explain the advantage of your method compared to latent flow matching with structured autoencoders? What is the advantage of also modeling z_t with time, instead of just learning the encoder/decoder for clean data as in latentFM?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose Structured Flow Autoencoders (SFA), a framework that augments Continuous Normalizing Flows (CNFs) with graphical model structures.  The paper demonstrates the versatility of SFA across multiple settings, including models with continuous and mixture latent variables as well as latent dynamical systems. Experimental results show that SFA achieves superior data fitting performance while preserving meaningful and structured latent representations.

### Strengths
The paper is clearly written and well organized, making it easy to follow. The proposed model and its accompanying theoretical analysis are sound and logically developed.

### Weaknesses
Despite its clarity, the paper presents several limitations that the authors should address:

(a) The proposed approach essentially represents a straightforward combination of Flow Matching and Variational Autoencoders (VAEs). As such, the conceptual novelty appears limited, and the contribution may be incremental.

(b) The probabilistic structures considered are restricted to latent variable models and latent chain models. Given this narrow scope, the term structured probabilistic models may be overstated; the work would be more accurately described as focusing on latent models with specific dependencies.

(c) The experimental evaluation lacks results on natural image datasets, which are standard benchmarks for assessing the scalability and expressiveness of flow-based models. Including such experiments—and comparing SFA directly with Flow Matching and VAE baselines—would significantly strengthen the empirical claims.

### Questions
1. How does SFA perform in high-dimensional domains, such as image or video modeling?

2.Can the proposed structured latent representations be extended to discrete or hybrid graphical models?

### Soundness
3

### Presentation
3

### Contribution
2
