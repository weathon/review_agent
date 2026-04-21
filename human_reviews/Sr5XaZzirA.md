# Fast Training of Sinusoidal Neural Fields via Scaling Initialization

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Neural fields are an emerging paradigm that represent data as continuous functions parameterized by neural networks. Despite many advantages, neural fields often have a high training cost, which prevents a broader adoption. In this paper, we focus on a popular family of neural fields, called sinusoidal neural fields (SNFs), and study how it should be initialized to maximize the training speed. We find that the standard initialization scheme for SNFs---designed based on the signal propagation principle---is suboptimal. In particular, we show that by simply multiplying each weight (except for the last layer) by a constant, we can accelerate SNF training by 10$\times$. This method, coined _weight scaling_, consistently provides a significant speedup over various data domains, allowing the SNFs to train faster than more recently proposed architectures. To understand why the weight scaling works well, we conduct extensive theoretical and empirical analyses which reveal that the weight scaling not only resolves the spectral bias quite effectively but also enjoys a well-conditioned optimization trajectory.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a study on sinusoidal neural fields (SNFs) and introduces "weight scaling," a method that accelerates training by adjusting the initialization of each layer, except the final one. The key idea behind weight scaling is straightforward: each layer’s weights are initialized by multiplying with a specific scale factor, which results in faster training across various neural field tasks. The authors benchmark their approach against existing methods, including the original SNF initialization by Sitzmann et al., and demonstrate that weight scaling consistently achieves faster convergence while maintaining robust training and test performance. Furthermore, the authors provide theoretical analysis explaining why weight scaling works.

### Strengths
**Originality:** The paper introduces an original idea that I have never seen used before in the neural field literature. Furthermore, the authors take the time to give a good theoretical analysis of why weight scaling works and how it relates to other initialization methods within the literature.

**Quality:** The paper has both theoretical and experimental contributions making it a good quality paper written by the authors. The theoretical contributions are sound and the experiments are compared with recent literature.

**Clarity:** The paper is in general well written. The authors take the time to exposit the main ideas behind weight scaling well and explain their key contributions in a readable manner. I have checked their mathematical proofs in the appendix and they were all well written and explained well.

**Significance:** The paper gives a new way to initialize sinusoidal neural fields which I believe is interesting to the community of researchers working in neural fields. The initialization of neural fields in general is not as commonly studied as other deep networks such as CNNs or Transformers and apart from a few recent works on the topic, I believe this paper offers a nice viewpoint in how sinusoidal neural fields should be initialized for efficient training.

### Weaknesses
**Novelty:** My main issue with the paper and what I feel is its main weakness is the fact that it is entirely focused on one type of neural field, namely sinusoidal neural fields. The paper does not say anything about other neural fields in general. The authors do say that their methods would not work on ReLU based neural fields due to ReLU having the positive homogeneity condition and
while the authors do say that sinusoidal neural fields are used within the community I would argue that there are several papers showing that sinusoidal neural fields are not as effective as other neural fields, such as [1], [2], [3], [4].  While the paper is written well I do feel that it being entirely focused on one type of neural field means it won't be as useful to many researchers in the neural fields community. 

**Large Scale Experiments:** The paper compares its methodology of weight scaling on a tasks such as image regression, binary occupancy, spherical data, audio data and fit a nef (in the appendix). Yet I noticed there were no experiments on Neural Radiance Fields (NeRFs) which I found very surprising. The authors say that their method is very effective at reducing the training time of neural fields which is generally large yet many of the tasks such as image regression, audio, occupancy are not considered big tasks within the community. On the other hand NeRF is a big experiment that takes anywhere between 4 hours to 12 hours to train, depending on the fidelity the user wants. I would have liked to see how their method did on such a task that requires hours of training. I also looked at the papers the authors compare to namely [1], [2], [3] and noticed that these papers compared their methods on NeRF. In the appendix the authors do compare their method on "fit a nef" on a large scale dataset but these takes less than 1 hour to train even with a SIREN. Thus I feel another big weakness of the paper is that the authors did not compare their method on a task that takes hours to train such as NeRF, so that we can see how their method does on a more complicated large scale neural field.

[1] Ramasinghe et al: Beyond Periodicity: Towards a unifying framework for activations in coordinate-MLPs; ECCV 2022.    

[2] Saragadam et al: WIRE: Wavelet implicit neural representations; CVPR 2023.

[3] Saratchandran et al: Scaling insights for optimizing neural fields: CVPR 2024.

[4] Saratchandran et al: A Sampling Theory Perspective on Activations for Implicit Neural Representations: ICML 2024.

### Questions
1). Your paper is entirely focused on neural fields that use a sinusoidal activation. The weight scaling method you propose scales the initialization of a sinusoidal neural field by a certain factor $\alpha$, as per eqn (6) in your paper. This method takes the SIREN initialization as its basis. Could you explain what happens if we take a standard initialization scheme like Kaiming uniform or Xavier uniform and scale it in exactly the same way as your eqn. (6)? Does this lead to anything new? Why did you apply your weight scaling to only the SIREN initialization?

2). Could you explain whether your results would go through for other activations that are used within the neural fields literature such as Gaussian [1], Wavelet [2], Sinc [4]? 

3). Did you try weight scaling on a NeRF task? I am very surprised there are no NeRF results as this a big experiment that requires hours to train and I noticed some of the papers you are comparing against [1], [2], [3] compare to NeRF. Could you explain whether weight scaling would or wouldn't work on NeRF and if it cannot be made to work on NeRF then why? I think it would really help your paper if you had NeRF results and you were able to show large speed ups on this large scale experiment.

4). Your weight scaling essentially just scales the initialization of the inner layers of the initialization by Sitzmann et al for SIRENs. However, if you look at the denominator of eqn. (3), line 194, there is the term $\omega_h$ which is supposed to be the frequency of the sinusoid being used on that layer. Could we not get the same effect as your initialization by using a sinusoid with frequency 
$\frac{\omega_h}{\alpha}$? 

5). In appendix C.3 you analyse the condition number of the Hessian of the loss landscape. You mention that the condition number of the Hessian directly affects the convergence rate. However, my understanding is that this is only true for convex loss landscapes. Loss functions associated to neural networks are almost always non-convex, as they employ non-linearities, therefore in such cases the condition number of the Hessian may not be well defined, as the Hessian may have negative eigenvalues or zero as an eigenvalue. Could you comment on this? I don't understand this analysis you have provided in your paper.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
[1] Ramasinghe et al: Beyond Periodicity: Towards a unifying framework for activations in coordinate-MLPs; ECCV 2022.    

[2] Saragadam et al: WIRE: Wavelet implicit neural representations; CVPR 2023.

[3] Saratchandran et al: Scaling insights for optimizing neural fields: CVPR 2024.

[4] Saratchandran et al: A Sampling Theory Perspective on Activations for Implicit Neural Representations: ICML 2024.

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
2

### Summary
The work introduces a method to accelerate training in sinusoidal neural fields (SNFs), focusing on a technique called weight scaling (WS). This method modifies the initialization by scaling the initial weights of SNFs (except for the final layer) by a constant factor. By improving the frequency characteristics of SNFs, WS significantly reduces training time—up to 10 times faster than the traditional approach—while maintaining generalization. The authors provide theoretical insights and empirical validation across tasks in some data domains, including images, 3D shapes, and audio signals. They compare WS with standard initialization and frequency tuning, demonstrating its advantages in both speed and stability during training. The idea behind this method is quite interesting, as it revolves around reducing time to create more efficient models; however, there are some significant weaknesses and gaps that cannot be ignored.

### Strengths
* $\textbf{Effective Speed Improvement}$: WS offers a substantial acceleration in training time without degrading model performance or generalization capabilities, which is valuable for SNF applications requiring rapid processing. 
* $\textbf{Simplicity and Practicality}$: The method requires minimal changes to existing architectures, making it straightforward to apply WS to SNFs without complex modifications. 
* $\textbf{Well-Written}$: The paper has been written in detail.
* $\textbf{Theoretical Justifications}$: The paper offers a detailed theoretical framework explaining why WS is effective, especially in reducing spectral bias and improving the conditioning of the optimization trajectory. This theoretical result supports the empirical findings and strengthens the reliability of the results.

### Weaknesses
The authors acknowledge the main limitations of their work, which cannot be ignored. To interpret the results accurately, these limitations must be carefully considered.

### Questions
* $\textbf{Impact on Long-Term Generalization}$: How does WS affect long-term generalization on larger datasets or in cases where SNFs are used for predictive tasks (e.g., continuous function approximation or spatial data interpolation)?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies initialization schemes for the class of sinusoidal neural fields (SNFs) and proposes a simple weight scaling method to accelerate training across different data domains. The idea is to simply scale the initial weights of all layers in the neural net by a well chosen factor. Some theoretical results and experimental results are provided to support and demonstrate the proposed method.

### Strengths
- Overall the paper is well written, with sufficient number of plots provided to visualize the findings.
- The proposed weight scaling scheme is simple yet seems to be quite effective in speeding up training.

### Weaknesses
- The theoretical results are quite limited. For instance, to show that the proposed weight scaling increases the relative power
of the higher-frequency bases, the paper considers the effects of such scaling on a 3-layer SNF only for the case when the width is one. 
- While the optimization trajectories are studied via the lens of the empirical neural tangent kernel, it would be more convincing if there are theoretical/empirical results that quantify the impact of such scaling on the gradient loss and the speed of convergence of gradient descent. 
- There are also some missing details and confusing parts in the paper that would need to be addressed in order to make the presentation stronger (see my questions below).

### Questions
- In Figure 2(b), the relationships between test PSNR and acceleration plotted for both cases of frequency tuning and weight scaling do not seem to be one-to-one. Why is that so? Am I missing something there?
- In Eq. (3), the initialization scheme involves scaling the uniform distributions. Will similar weight scaling trick achieve similar acceleration results when the distributions involved are non-uniform (say Gaussian)?
- How does the speed-up offered by the proposed weight scaling scheme depend on/interact with the choice of optimizers (say SGD vs Adam)? Is it more effective for certain choices of optimizers? 
- For the results in Figure 3, the learning rates vary with layers and are scaled down to match the rates of unscaled SNF. Can you provide more details on how they vary with depth? Also, what if the same learning rate is used for all layers?
- Can you provide more details on how the optimization problem (10) is solved? Can we treat $\alpha$ as a trainable parameter instead?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes an alternate initialization scheme for sinusoidal neural fields. The main claim is that scaling the initialization weights by a constant value greater than one can help speed up the learning in SNFs as well as improve fitting for higher frequencies, by improving the function at initialization as well as boosting optimization trajectories. Empirical validation on various domains is presented to substantiate the arguments.

### Strengths
- The motivation is clear, and the exposition is easy to follow.
- Although not very surprising, the study of the theoretical behavior of weight scaling is fairly comprehensive and I appreciate the authors efforts to draw upon different theoretical tools in existing literature to provide a broad picture.

### Weaknesses
- The current choice of experiments is rather limited. Since the contribution is largely methodological, experiments on important practical applications of SNFs, specifically NeRF and solving differential equations (for instance, see Saratchandran et al., 2024), as well as larger datasets, are needed to justify the claims. 
- Experimental methodology: A few aspects of the experimental methodology are unclear and/or need revision.
    - Baseline: The included baseline with Xavier initialization is largely irrelevant from a practical standpoint. Alternatively, Multiplicative Filter Networks (Fathony et al, 2021) is an important baseline to consider, especially since multiplicative networks, akin to weight scaling, are known to speed up learning for higher frequencies [4] and thus, directly competes with weight scaling in terms of spectral bias.
    - Fixed Iterations: The choice of number of training iterations to report performance is arbitrary and could reflect unfairly on the baselines. The authors should consider including the PSNR curves for training until convergence (for both WS and baselines) for understanding practical feasibility, since ground truth data to stop at a target PSNR is not available in practice.
    - Full Batch Training/Susceptibility to SGD Noise: In practice, it is often not possible to perform full batch training due to size of dataset, network size etc. How does the noise in moving to stochastic (batched) gradient descent affect the convergence of weight scaling? Experiments that reflect those settings will help understand better the limitations of the proposed framework. 
- Related work: Missing citations for Multiplicative Filter Networks (Fathony et al, 2021) as well as spectral bias literature:
    - [2] The Convergence Rate of Neural Networks for Learned Functions of Different Frequencies, Basri et al., NeurIPS 2019
    - [3] Towards Understanding the Spectral Bias of Deep Learning, Cao et al., 2020
    - [4] The Spectral Bias of Polynomial Neural Networks, Choraria et al., ICLR 2022

### Questions
- See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
