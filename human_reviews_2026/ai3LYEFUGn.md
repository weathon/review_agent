# Unsupervised learning of disentangled representations via diffusion variational autoencoders

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
We present the Score-based Autoencoder for Multiscale Inference (SAMI), a method for unsupervised representation learning that combines the theoretical frameworks of diffusion models and VAEs. By unifying their respective evidence lower bounds, SAMI formulates a principled objective that learns representations through score-based guidance of the underlying diffusion process. The resulting representations automatically capture meaningful structure in the data: it recovers ground truth generative factors in synthetic datasets, learns factorized, semantic latent dimensions from complex natural images, and encodes video sequences into latent trajectories that are straighter than those of alternative encoders, despite training exclusively on static images. Furthermore, SAMI can extract useful representations from pre-trained diffusion models with minimal additional training. Finally, the explicitly probabilistic formulation provides new ways to identify semantically meaningful axes in the absence of supervised labels. Overall, these results indicate that implicit structural information in diffusion models can be made explicit and interpretable through synergistic combination with a variational autoencoder.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents DiVA, a diffusion variational autoencoder for disentangled representation learning by combining a diffusion model with a VAE. The training loss is formalized under a unified ELBO objective.

### Strengths
- A novel approach that combines VAEs with diffusion-based models.
- The new objective formalized of the ELBO.
- Good results on the selected benchmarks and datasets.

### Weaknesses
**Major:**
- Unorganized results and a lack of detailed reporting in several sections. 
- Lack of comprehensive benchmarks on the suggested datasets.
- Some experiments are missing important implementation and protocol details.

**Minor:**
- Limited discussion of related literature on sequential disentanglement for video trajectories.

### Questions
1. **In results in Synthetic Disks dataset:** a. Why the authors report MSE for these results? Unconditional generations trending toward noise could artificially lower MSE; please justify this choice or include complementary metrics. 
b. Why not use 2D shapes dataset (e.g., those used in β-VAE) for comparability?

2. **In results CelebA dataset:** a. Lines 321–323: what was the reduction in MSE? b. Why don’t you refer to the DBAE + TC results in Table 3 directly within this section?

3. Why the images are gray in Figure 3?

4. **Feature extraction for pre-trained diffusion models:** a. Can the authors show any additional results for this experiment? Also, please add more details about the fine-tuning protocol.
b. What are the pros and cons of using pre-trained backbones versus training from scratch?
c. Does this approach achieve results similar to joint training? By how much does it reduce training time?

5. **In results Encoding of video trajectories:** 
a. There is a rich literature on sequential disentanglement that isn’t cited, such as [1] and [2].
b. Do you have any results demonstrating disentanglement on CelebA-HQ?
c. What is the motivation for using cosine similarity? How does it support the hypothesis in lines 438–440?
d. Why do you only compare against DiffAE?
e. Why the authors do not include benchmarks on a video dataset with labels for static and dynamic factors (e.g., Moving dSprites or related variants)?

**References**

[1] "Sequential Representation Learning via Static-Dynamic Conditional Disentanglement" M. Cyrille Simon et al.

[2] "Sequential Disentanglement by Extracting Static Information From a Single Sequence Element" N. Berman et al.

### Soundness
2

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
4

### Summary
This paper proposes to condition diffusion models on a learned latent representation $z$. Using this new formulation, the authors derive a novel learning objective where the second term is equivalent to VAEs’ regularisation term. As in $\beta$-VAE, then add a scalar $\beta$ to this term to weight the regularisation and induce sparsity. The reconstruction and disentanglement abilities of the model are then evaluated on a synthetic disk dataset and on CelebA.

### Strengths
- The paper is well written and easy to follow
- The idea of learning meaningful latents with diffusion models is appealing

### Weaknesses
My main concern is about the disentanglement claim. The authors attempt to replicate $\beta$-VAE behaviour using a $\beta$ term in the second term of Eq. 5, but in the experimental section, disentanglement is not measured with any disentanglement metrics (e.g., MIG, DCI or any listed in [1]).  Furthermore, several datasets have been used to benchmark disentanglement (e.g., DSprites, SmallNorb, etc. See [1] or [2] for more examples) but apart from CelebA none of these are used here. We know from [1] that in VAEs, disentanglement capacity varies a lot depending on the dataset, so one would expect a more expansive evaluation when the authors state that their model can "recover ground truth factors". Especially given that this is not so clear cut for $\beta$ VAE or any VAE doing disentanglement. These models tend to induce sparsity with a PCA-like behaviour [3-6], and one can obtain disentangled representations if those PCs correspond to ground truth factors. Overall, I think this paper is interesting, but would need a significant rework of the empirical section to justify the disentanglement statement.

### Questions
- The proposed model disentanglement capacity should be evaluated using disentanglement metrics (see [1])
- The evaluation should be done on several disentanglement dataset (see [1-2])
-  Could the authors discuss the relationship between the proposed model and diffusion models being a special case of hierarchical markov VAE, as shown in [7]?
- Would this new formulation allow for other disentanglement techniques than $\beta$-VAE?
- I suggest the authors avoid saying that their model "recovers ground truth generative factor" as most disentanglement models cannot reliably do this (see [1])
- The name of the model is quite confusing, given the naming of these two previous works [8-9], and may need to be updated

References
=========
[1] Locatello, Francesco, et al. "Challenging common assumptions in the unsupervised learning of disentangled representations." international conference on machine learning. PMLR, 2019. (limitations of disentanglement)

[2] Gondal, M. W., Wuthrich, M., Miladinovic, D., Locatello, F., Breidt, M., Volchkov, V., ... & Bauer, S. (2019). On the transfer of inductive bias from simulation to the real world: a new disentanglement dataset. Advances in Neural Information Processing Systems, 32.

[3] Dai, B. et al. "Connections with robust PCA and the role of emergent sparsity in variational autoencoder models." Journal of Machine Learning Research 19.41 (2018): 1-42.

[4] Bin Dai, & David Wipf (2019). Diagnosing and Enhancing VAE Models. In International Conference on Learning Representations.

[5] Rolinek, M., Zietlow, D., & Martius, G. (2019). Variational autoencoders pursue pca directions (by accident). In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 12406-12415).

[6] Bonheme, Lisa, and Marek Grzes. "Be more active! understanding the differences between mean and sampled representations of variational autoencoders." Journal of Machine Learning Research 24.324 (2023): 1-30.

[7] Luo, C. (2022). Understanding diffusion models: A unified perspective. arXiv preprint arXiv:2208.11970. (equivalence between diffusion models and HMVAEs)

[8] Ilse, Maximilian, et al. "Diva: Domain invariant variational autoencoders." Medical Imaging with Deep Learning. PMLR, 2020. (DiVA confusing name 1)

[9] Perez R. et al. (2020). Diffusion Variational Autoencoders. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, ĲCAI-20 (pp. 2704–2710). International Joint Conferences on Artificial Intelligence Organization. (DIVA confusing name 2)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work basically propose to learn a latent variable model along with the standard diffusion objective. By some algebra, the overall loss can be written in form of a guidance term and can be jointly trained with the unconditional score using denoising score matching plus some KL penalty on gaussian prior. It is demonstrated that the learned latent model can encode disentangled representation of the data distribution.

### Strengths
Combining VAE loss with standard denoising score matching to learn meaningful representation is interesting.

### Weaknesses
1. The proposed method is conceptually and methodologically similar to the DiffAE paper. It is unclear why DiVA performs better. It is argued this is because DiVA minimizes the exact ELBO, which is not true. In the algorithm, the weights $\lambda_t$ is not included, implying DiVA only minimizes ELBO approximately. My question is, what is the unique advantage of DIVA? 

2. The trained model has to take a clean image $x_0$ as conditional input, which limits its generation capability. In figure 3 B, I would say when you condition on $x_0$, the generated images seem to be identical to it, with only minor difference. This is a sign of overfitting. Will DiVA generate high quality image when a clean image is not available? How do you calculate the FID in Table 3? Do you generate each image by input a clean image, or are the images generated unconditionally? 

3. Can you provide more theoretical analysis on objective (10)? For example, what would be the optimal $q_{\phi}$ that minimizes this loss? Can you come up with some more in depth characterization of $q_{\phi}$'s property? Currently, the objective makes sense intuitively, but is kind of superficial in my opinion, as there is no theoretical guarantee that $q_{\phi}$ can capture disentangled representations. If it indeed does, why? Does it work well consistently on different dataset, or it only works on simple dataset like faces and disks?

4. Experiments are limited to simple dataset such as disks and faces. Please perform experiments on ImageNet to fully demonstrate the strength of your approach. I am not convinced if only experiments on faces and disks are provided, as nowadays, these datasets are considered too simple.

5. What is the current state of the art methods for learning disentangled representations besides the ones based on diffusion models? Does DiVA beat those algorithms? Is it really necessary to learn disentangled representation based on diffusion framework? If so, why? What is the unique advantage of diffusion in this context, from a rigorous theoretical perspective?

### Questions
See my questions above.

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
3

### Summary
The paper proposes an unsupervised representation disentanglement model, the Diffusion Variational Autoencoder (DiVA). Specifically, DiVA integrates variational autoencoders and diffusion models to enable unsupervised learning of structured and interpretable latent representations with strong factorization and semantic consistency, while maintaining high-quality generative performance. The proposed model is evaluated on both synthetic and real-world datasets and compared against several baseline methods.

### Strengths
1. The paper addresses the problem of learning disentangled representations for image data, which is an important and long-standing research topic. 
2. The idea of combining the advantages of diffusion models and VAEs is conceptually clear and technically sound. 
3. The paper provides a detailed analysis of the ELBO formulation.

### Weaknesses
1. The comparison between the proposed model and baseline methods, particularly diffusion-based disentanglement models, is unclear and difficult to follow. The key results in Appendix A.6–A.7 are important and should be highlighted in the main text. Moreover, no qualitative examples or visual comparisons are provided to illustrate the superiority of the proposed model over recent baselines such as InfoDiff, DisDiff, and DBAE+TC. 
2. The InfoDiffusion model appears highly relevant, as it also includes ELBO analysis and mutual information regularization, but the differences between the two approaches are not systematically discussed. 
3. The evaluation is limited to one synthetic and one real-world dataset. Given that multiple public datasets with ground-truth disentanglement factors (e.g., 3DShapes, dSprites, etc) and commonly used real-world datasets (e.g., CelebA, FFHQ, etc) are available, the experimental validation seems insufficient.

### Questions
Please see "Weaknesses"

### Soundness
3

### Presentation
2

### Contribution
2
