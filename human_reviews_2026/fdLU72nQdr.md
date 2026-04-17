# Quasi-Monte Carlo Methods Enable Extremely Low-Dimensional Deep Generative Models

- Decision: Accept (Poster)
- Scores: 2, 6, 8, 6

## Abstract
This paper introduces *quasi-Monte Carlo latent variable models* (QLVMs): a class of deep generative models that are specialized for finding extremely low-dimensional and interpretable embeddings of high-dimensional datasets.
Unlike standard approaches, which rely on a learned encoder and variational lower bounds, QLVMs directly approximate the marginal likelihood by randomized quasi-Monte Carlo integration.
While this brute force approach has drawbacks in higher-dimensional spaces, we find that it excels in fitting one, two, and three dimensional deep latent variable models.
Empirical results on a range of datasets show that QLVMs consistently outperform conventional variational autoencoders (VAEs) and importance weighted autoencoders (IWAEs) with matched latent dimensionality.
The resulting embeddings enable transparent visualization and *post hoc* analyses such as nonparametric density estimation, clustering, and geodesic path computation, which are nontrivial to validate in higher-dimensional spaces.
While our approach is compute-intensive and struggles to generate fine-scale details in complex datasets, it offers a compelling solution for applications prioritizing interpretability and latent space analysis.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates using explicit integration over the latent variable in deep latent variable models as an alternative to the amortized variational inference approach of VAEs. As this integral is intractable the authors propose using "quasi monte-carlo" integration; rathering than estimating via samples from the prior, the authors sample a randomly shifted lattice of points in the latent space to approximate the integral. The authors argue that this approach achieves considerably better results than variational auto encoders in cases where the latent space is low-dimensional enough that this approach is computationally feasible, i.e. 2 and 3 dimensions, cases where the latent embeddings may be used for visualization purposes. In their experiments, the authors show better performance compared to VAEs and importance-weighted auto encoders as measured by test loss and reconstruction. They also compare the approach to other low-dimensional embedding approaches tailored to visualization, such as UMAP.

### Strengths
**Writing**
- The paper is clear and well written with a nice description of the motivation of the problem. 
- The paper includes a good discussion of related work and identifies many relevant papers from recent literature

**Methodology**
- Using randomly shifted lattices to tile the latent space for training is a clever idea and there does seem to be some notable performance improvements in some cases.
- There is a good exploration of the value of the lattice based approach in the appendix

**Results**
- The paper includes some excellect visualizations that make a compelling case for a deep-latent variable model with 2-dimensional uniform prior as a method for generating useful visualizations for high dimensional datasets.
- The authors used an interesting variety of datasets to showcase the performance of the approach

### Weaknesses
**Novelty**
- As the authors acknowledge, this idea is not inherently novel, though it has generally been dismissed as impractical in the past. I think that showing a compelling use case for this approach could be a worthwhile contribution, but it sets a high bar for the experimental results.

**Computational costs**
- This method has a high computational cost, even for the 2-d and 3-d cases discussed. The authors only test with relatively simple datasets and networks.
- The authors don't provide any quantitative comparison of the computational costs of different methods. This is a notable omission as computational concerns are the primary reason this approach hasn't been explored before.

**Framing**
- The authors frame this approach as generally better for low-dimensional latent variable models, but notably change the prior from the standard Gaussian used in most previous work to a uniform, without comparison their approach using a Gaussian prior. 
- I think the uniform is a valid choice in this context for something like visualization, but it should be clear if this is not necessarily a general-purpose substitute for VAE-style methods.

**Baseline comparison**
- I'm not convinced that the comparison to baselines is fair.
- Looking at figure 2, the VAE seems to be suffering from posterior collapse. In my experience it's definitely possible to train a VAE for MNIST with 2-latent dimensions that performs better than what is shown.
- The IWAE baseline only used 10 samples, which seems quite small given the comparison to QLVMs. I would prefer to see a computation-time matched comparison to better understand if the QLVM approach is actually superior.
- The authors dismiss improved variational methods such as iterative amortized inference based on their computational expense and hyper parameters, but I would see these approaches as an important baseline. If they are truly less practical than this approach, I would like to see that justified experimentally.

**Experimental details**
- There are several details of the experiments that are not properly covered in the text as mentioned below.

### Questions
- Is the variational posterior approximation for the VAE and IWAE baselines still Gaussian in your experiments? 
  - If so, how do you enforce the boundary constraints for the uniform prior and estimate the KL-divergence etc? 
  - Does this affect the variance of the training objective e.g. compared to a Gaussian VAE where the KL-divergence can be computed in closed form?
- How is the test loss in figure 2 computed? 
  - All three methods are computing different lower bounds on the MLL, so my question is: is this comparing learned decoder performance by evaluating all of the learned models with the same lower bound?
  - Or alternatively is it reporting the same loss used to train each model?

### Soundness
2

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
4

### Summary
This paper takes an interesting approach to latent variable modelling. The presentation contrasts the approximations done using amortized inference with a more direct inference by using a monto carlo estimate of the marginal log-likelihood. The approximation to the marginal log-likelihood is done using uniform prior which means that the latent space needs to be bounded and for computational reasons very low-dimensional in order to get a good estimate. The paper discusses the pros and cons of using this more "natural" approach compared to an amortized inference mechanism used in VAEs.

### Strengths
This is well written paper that is very easy to understand. In some sense I think the narrative of explaining this and continously relating this to VAEs is a little bit too specific, in some ways this is a much more natural approach to the problem of a latent variable models compared to an amortized inference scheme. That is just my personal oppinion and not something that has been reflected in the score. The flip side of this is that the latent space is required to be very low-dimensional in order to be able to compute the approximation. As such, and this the authors do very well, it shouldn't really be put in the same context as VAEs or other generative models but rather UMAP t-SNE and possible a GP-LVM as its where these models are useful this approach will be applicable. Differently from the first two though, this is a generative model and allows for the latent location to be infererred for new data, due to the uniform latent prior, which is more challenging with spectral dimensionality reduction methods.

I liked the discussion around the failure modes of the amortized approximate posterior. I think your intuitions about this makes sense and it would be nice to evaluate this a bit more to get a more. There is a very nice paper [1] that discusses the interplay of the complexity of the back and the generative mapping in an auto-encoder setting. There might be some information that is relevant there for future work.

[1] Barber, D. (2014). Implicit Representation Networks.

### Weaknesses
While I think the paper is a nice study and tells a story about how one can approach latent variable modelling in a more direct way there is also a question about the limitations of the paper. The first thing I think would have been interesting to see is the implication of the uniform prior over the latent represenation and what effects it does have. It would be great to have a discussion on more informative priors and how the approach would change in that case and the impact on the results. I don't fully understand the results right at the end of the Appendix which I assume are supposed to relate to this.

The authors state and are very honest with the fact that the approach is very computationally expensive. It would be nice to see what this actually means in-terms of numbers and how many samples \(M\) that is used in the experiments and the dependency of this.

### Questions
Given that you are able to compute the Jacobian, can't you then represent the pull back metric in the latent space? This would allow you to compute more principled Geodesics compared to the approach you are now taking.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper investigates the performance of QMC in computing the marginal log likelihood in very low-dimensional settings. This resulted in a model that, although computationally demanding, did not require an encoder and outperformed VAEs and IWAEs on simple datasets, such as MNIST. Given the dimensionality limitation of such models, the authors propose this method as an alternative to UMAP or t-SNE. In opposition to these algorithms, QLVMs benefit from additional capabilities, such as latent traversal or density estimation, which are essential for unsupervised data analysis, ensuring transparent results.

### Strengths
This paper investigates a simple yet overlooked way of computing the marginal likelihood and proposes a creative approach to use this in the restricted setting of 3 or fewer latent dimensions. I believe that this paper can have a strong impact on the unsupervised learning community, especially for people doing exploratory data analysis on high-dimensional data. The paper was easy to follow, and I really enjoyed reading it. Well done to the authors, great work!

### Weaknesses
- The posterior collapse issue of VAEs when reducing the number of latent dimensions too much has been overlooked. It would be great to add this to the discussion in 3.1.
- Fig. 3 B do not explain which models are associated with each line.

### Questions
- In 3.1, VAEs in very low-dimensional spaces may suffer from posterior collapse [1-3], it could easily be checked (e.g., following the method proposed in [3] to monitor passive variables). Such results would provide a good explanation for the poor performances of VAEs in this setting. I am not sure whether IWAE can also suffer from posterior collapse, but this could be worth exploring.
- Also in 3.1, VAEs tend to behave in a polarised regime [2-4] where some latent variables are used to decrease the KLD while others encode meaningful information but depart from the prior. Could what is observed in Fig. 3 be a consequence of this?
- In Fig. 3 B, which line corresponds to which model?
- Have the authors tried QLVM on high-dimensional tabular data (e.g., gene expression)? Which type of results would they expect?
- l. 123 it should be ")]" instead of "])"
_________

**References**

[1] Dai, B., Wang, Z., & Wipf, D. (2020, November). The usual suspects? Reassessing blame for VAE posterior collapse. In International conference on machine learning (pp. 2313-2322). PMLR.

[2] Rolinek, M., Zietlow, D., & Martius, G. (2019). Variational autoencoders pursue PCA directions (by accident). In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 12406-12415).

[3] Bonheme, L., & Grzes, M. (2023). Be more active! understanding the differences between mean and sampled representations of variational autoencoders. Journal of Machine Learning Research, 24(324), 1-30.

[4] Dai, B., Wang, Y., Aston, J., Hua, G., & Wipf, D. (2018). Connections with robust PCA and the role of emergent sparsity in variational autoencoder models. Journal of Machine Learning Research, 19(41), 1-42.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a latent variable model that directly optimizes the marginal likelihood using Monte Carlo integration, without using an encoder-based approximate posterior. For uniform priors, the authors provide appropriate integration schemes when the latent space has small dimensionality. The experimental results show that the proposed approach is competitive with related deep generative models, and the learned latent representations appear meaningful.

### Strengths
- The proposed method is conceptually simple, straightforward to implement, and performs well in the conducted experiments.
- The integration schemes, together with the uniform prior, appear to leverage the latent space effectively.
- The paper is generally well written and easy to follow.
- I think that the method provides a meaningful alternative to UMAP-style approaches, with the advantage that it includes a decoder.

### Weaknesses
- As acknowledged by the authors, scalability is a clear limitation of the approach.
- While the experimental results are promising, it is not completely clear to what extent this approach opens new research questions beyond the presented scenarios.

### Questions
Q1. The structure of the latent space is somewhat unclear due to the periodic boundary. Could the authors elaborate on the choice $\mathbf{z} = (\sin \mathbf{z}, \cos \mathbf{z})$ and its implications? What is the topological structure of the resulting latent space?

Q2. In Fig. 1, it appears that blue points are always associated with batch 1 and red points with batch 2. Are the batches fixed throughout training, or do they change? In other words, do data points in batch 1 always interact with the same set of latent points?

### Soundness
3

### Presentation
3

### Contribution
3
