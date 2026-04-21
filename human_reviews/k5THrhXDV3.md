# Deep Generative Clustering with Multimodal Diffusion Variational Autoencoders

- Avg Score: 6.67
- Decision: Accept (poster)
- Scores: 8, 6, 6

## Abstract
Multimodal VAEs have recently gained significant attention as generative models for weakly-supervised learning with multiple heterogeneous modalities. In parallel, VAE-based methods have been explored as probabilistic approaches for clustering tasks. At the intersection of these two research directions, we propose a novel multimodal VAE model in which the latent space is extended to learn data clusters, leveraging shared information across modalities. Our experiments show that our proposed model improves generative performance over existing multimodal VAEs, particularly for unconditional generation. Furthermore, we propose a post-hoc procedure to automatically select the number of true clusters thus mitigating critical limitations of previous clustering frameworks. Notably, our method favorably compares to alternative clustering approaches, in weakly-supervised settings. Finally, we integrate recent advancements in diffusion models into the proposed method to improve generative quality for real-world images.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a novel multimodal Variational Autoencoder (VAE) model that combines weakly-supervised learning and clustering for multiple heterogeneous modalities. The key contributions of the paper are as follows:

**Multimodal VAE Model:** The paper presents a new multimodal VAE model that extends the latent space to simultaneously learn data clusters while leveraging shared information across different data modalities.

**Improved Generative Performance:** Experimental results demonstrate that the proposed model outperforms existing multimodal VAEs in terms of generative performance, particularly for unconditional generation tasks.

**Automatic Cluster Selection:** The paper introduces a post-hoc procedure to automatically determine the number of true clusters, addressing critical limitations of previous clustering frameworks. This enhances the model's ability to discover meaningful clusters in weakly-supervised settings.

**Comparison to Alternative Clustering Approaches:** The proposed method is compared favorably to alternative clustering approaches in weakly-supervised settings, highlighting its effectiveness in clustering and learning representations from multimodal data.

Integration of Diffusion Models: The paper incorporates recent advancements in diffusion models into the proposed method to further enhance generative quality, particularly for real-world images. This integration helps improve the realism of generated images.

### Strengths
I appreciate the authors' great effort in providing various experimental studies in the main part as well as the appendix.

* The paper is well-organized.
* The results show clear improvement over the baseline methods, especially for unconditional generation and realistic data clustering.
* The idea is incrementally inspired from recent MVAE methods as well as DiffuseVAE; But, it is sound and interesting.
* The automatic cluster selection on test data is interesting and novel, showing advantage to the prior work.

### Weaknesses
The proposed method needs better explanations.

* What are the learned modules in CMVAE? Are all $q(z | X)$, $q(w_i | x_i)$, $p(w_i)$, $p(c)$, $p(z|c)$ learned during the training?
* In figure 1, I cannot understand why there exist $z_1$, ..., $z_M$.
* I cannot understand the cross-reconstruction formulation in Eq (8). I was thinking $p_{\theta_m}$ can be only called with $w_m$

### Questions
The questions were asked in the weaknesses part.

BTW, what are the cluster in CUBICC dataset? species or the subspecies?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors present a neat extension of typical VAEs to the multi-modal settings. It results in a framework that can be used for weakly-supervised clustering. By further combining with Denoising Diffusion Probabilistic Models, it can generative better multi-modal

### Strengths
- The presentation is clear and well-structured.  
- The authors report solid experimental results, both qualitatively and quantitatively on the selected two datasets. 
- The method is technically sound, and several prior works on VAEs are neatly integrated together.

### Weaknesses
1. I'm concerned about the technical novelty. The authors extend typical VAEs to multi-modal VAEs for two tasks. The first is the application of weakly-supervised clustering, and the second is to combine multi-modal VAEs with DDPM to improve its generative quality. However, both applications of VAEs are not novel and can be viewed as an extension from VAEs to multi-modal VAEs. Though some new tricks are proposed in the form of multi-modal VAEs like Eq.8, it is not properly discussed and evaluated with sufficient ablation studies. 

2. The experiments are only conducted on one realistic multi-modal dataset that only contains two modalities. The authors mention the state-of-the-art weakly supervised methods are not scalable to numerous modalities while the proposed method is. The prior work is either on image-text [1] datasets or video-audio [2] [3] datasets. In Table 2, on PolyMNIST the proposed method is not improved over non-VAE SOTA either.
Therefore, I think the authors overclaim the contribution of the scalability of numerous modalities. Especially for weakly-supervised clustering, in my opinion, a realistic multi-modal dataset is essential to claim the contribution of scalability to more modalities.


[1] Zhou et al., End-to-End Adversarial-Attention Network for Multi-Modal Clustering.

[2] Chen et al., Multimodal Clustering Networks for Self-supervised Learning from Unlabeled Videos.

[3] Alwassel et al., Self-Supervised Learning by Cross-Modal Audio-Video Clustering.

### Questions
1. Since the contribution of this work is the combination of several existing approaches based on VAEs. It is important for the authors to focus on to further justify the contribution of extension VAEs to multi-modal VAEs in these tasks are not trivial and require careful design. I suggest the authors conduct ablation studies of such designs e.g. Eq.8.

2. To claim the scalability of the method, the authors may consider more experiments on realistic datasets with more than two modalities.

### After rebuttal
Thanks for the response and addressing my concerns. I raised my score to borderline accept. I suggest the authors consider moving some content in the Appendix (e.g. Eq.8 ablation study) to the main manuscript.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work concerns multimodal VAE applied to clustering. In doing so the authors propose a three-fold approach: a VAE objective for latent clustering, an entropy based cluster number selection, and Diffuse VAE integrated to the multimodal setting. Furthermore the authors present the experiments using fairly common tasks for multimodal VAE plus for weakly supervised clustering.

### Strengths
Overall, this paper is well-written and easy to read. It’s advocately positioning itself on the problem of clustering with multimodal VAE, and, through a neat combination of ideas that are slightly improved over previous works, it present a holistic solution that outperform previous works both in multimodal VAE and weakly supervised clustering.

Specifically, the objectives are heavily inspired by the previous works. The choice of selecting $k$ based on entropy is both intuitive 
incorporation of DiffuseVAE is direct yet impactful. Despite these individual components, what stands out is the overall pipeline being orchestrated coherently . The authors deserve commendation for doing so.

The experimental results are satisfactory and in line with common setups for multimodal. However this could be improved (see the weakness below).

### Weaknesses
As this paper is positioned as multimodal VAE for clustering, a broader empirical comparison with weakly supervised model would adds greatly to its value.

While the current version compares the proposed method with  [1,2,3], there is an wide range of research in weakly supervised clustering, including but not limited to [4, 5]. Although these may not be latent models, they (with other works) demonstrates the big picture of  weakly supervised clustering and deserve comparison


Additionally, the quantitative results relegated to the Appendix are pivotal for a comprehensive grasp of the proposed work's empirical performance. It would be beneficial to include these findings in the main body of the text.

Furthermore, a lot of quantitative results in Appendix is actually crucial in understanding the empirical performance of the proposed work. It would be beneficial to include these them in the main body of the text.


---------

[1] Jiang et al. Variational deep embedding: an unsupervised and generative approach to clustering IJCAI 2017

[2] Caron et al. Deep clustering for unsupervised learning of visual features.  ECCV 2018

[3] Tain et al. Contrastive multiview coding ECCV 2020

[4] Oner et al. Weakly Supervised Clustering by Exploiting Unique Class Count ICLR 2020

[5] Chang et al.  Deep adaptive image clustering ICCR 2017

[6] Yang et al. Joint unsupervised learning of deep representations and image clusters CVPR 2016

### Questions
Can the proposed multimodal VAE be applied to more complex data (say datasets with higher resolution, like AFHQv2, FFHQ) and if so, how would this impact the visual quality of the results?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
