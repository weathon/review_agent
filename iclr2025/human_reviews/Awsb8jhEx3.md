## Human Reviewer 1

### Summary
This paper proposes DClAM, a deep clustering method that integrates the associate memory idea of ClAM into the deep learning pipeline. Their method involves fine-tuning an autoencoder model by taking multiple associative memory (AM) steps on encoded embeddings before decoding, which enables joint optimization of representation learning and clustering. The paper shows that DClAM outperforms existing deep clustering methods on eight datasets spanning image and text with multiple architectures.

### Strengths
- I think the core idea of integrating AMs into the (deep) autoencoder pipeline via a differentiable loss is novel and interesting. The method is agnostic to architectural choices on the encoder and decoder.
- The paper comprehensively evaluates on diverse datasets, comparing against a range of clustering and deep clustering baselines.

### Weaknesses
- There's a large body of work on vector-quantized VAEs [1], which seems related to DClAM in that they're both learning an autoencoder with a latent space with a discrete structure. One obvious difference is that this model isn't a VAE (doesn't consider a prior over latents). Still, I think the learned centers play a similar role to the learned vectors in a VQ-VAE; their objective is essentially yours, with only one AM step. I think the paper should discuss the connection in more detail.
- The paper could benefit from more detailed ablation studies on key hyperparameters. I think reporting sensitivity to (1) the number of AM steps and (2) the number of clusters would give a better sense of the method's tuning requirements. Also, what is the number of AM steps you use? I can't find it in Table 5.
- What does the "restart" hyperparameter mean? Does it mean you ran the entire pipeline with five random seeds and reported the best performance?
- How does this method compare to other deep clustering methods in terms of computational efficiency? Mainly, I'd like to better understand how computationally expensive the AM steps are.

[1] Van Den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation learning." Advances in neural information processing systems 30 (2017).

Minor
- I think it's generally good to have the proposed method's explanation (sec 4) self-contained. A core component (eq 7) is explained separately in section 3; you might consider briefly re-stating this operator in section 4.

### Questions
Please see questions in weaknesses section above.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This study presents DCLAM a deep clustering algorithm that internally uses Clustering with Associative Memory (CLAM) algorithm. The proposed method uses encoder-decoder architecture which is optimized by an associative memory-inspired loss function.  Finally, the method is evaluated in comparison with existing methods using silhouette coefficient (SC) and reconstruction loss over image and text datasets.

### Strengths
The research integrates autoencoder and pretraining while optimizing the data in the latent space using the CLAM algorithm. Which is a good extension to CLAM. 
The approach tries to maintain a minimal reconstruction loss during the process of finding clusters.
The paper describes the technical details well which make it reproducible.

### Weaknesses
The evaluation process only included SC although the ground truth labels are available for the used datasets. The SC alone can’t be enough especially with its underlying assumptions about the cluster’s distributions.
The reported SCs for all the experiments are constrained by 10% range of change in the reconstruction loss. This is limiting the space for clustering improvement during training. Which is not fair for clustering algorithms, especially those that don’t use the decoder anymore in the clustering process.
It is mentioned in the paper that “applying CLAM in a latent space learned by a pretrained autoencoder is not an effective strategy”. However, the results reported don’t test this hypothesis.
The proposed method requires a choice of three learning rates which are difficult to select over different applications.
The belief mentioned in the evaluation section of using unsupervised metrics contradicts with using the information of number of clusters which is unknown in many unsupervised applications.

### Questions
Describing the method to be agnostic to architecture can be easily misunderstood. Because architecture is important according to the type of dataset. What could be deduced is that the method itself is flexible and can be integrated with different autoencoders architectures.
Evaluating the proposed method using ground truth-dependent metrics would make the results more reliable. Mitigating data leakage between training and evaluation can still be done without completely ignoring the ground truth information.
The paper lacks a solid justification why balancing the SC and reconstruction loss is important.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper applies ClAM (Saha et al. 2023) to various neural architectures (CNN, MLP) and achieves competitive performances on various text and image clustering benchmarks. The main metric of concern is the Silhouette Coefficient (SC), commonly used in clustering literature for unsupervised clustering quality.

### Strengths
- Paper is well written and easy to follow
- Very clear motivation, theoretical analysis appear to be correct, core algorithm clearly explained, and experiments are presented thoroughly

### Weaknesses
- The work builds up on ClAM (Saha et al., 2023), which may limit its novelty.
- Arguably, since the work is concerned with deep clustering, baseline methods should include traditionally metric learning-based approaches.
- In contemporary literature standards, both the neural network trained in the work and the dataset are small. It is unclear whether the proposed method would generalize to more realistic network sizes and datasets.

### Questions
- It seems like the entire (vision) transformer literature is missing from the analysis. This is not a critical concern, but it would be interesting to evaluate DClAM's performance on transformers across image and text tasks.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
2

---

## Human Reviewer 4

### Summary
This paper presents DClAM, a deep clustering method that builds on the previous ClAM approach by integrating it with deep clustering techniques, specifically through a deep autoencoder (AE). DClAM is architecture-agnostic, demonstrating strong performance across different autoencoder designs and dataset types. It achieves high clustering quality and low reconstruction loss in both ambient and latent spaces, surpassing previous methods.

### Strengths
The paper demonstrates superior results across various benchmarks, highlighting the robustness of the proposed approach. Additionally, the authors conduct experiments across diverse domains and architectural types, effectively showcasing the generalizability of their method. The preliminary section provides a thorough overview of foundational concepts, enhancing the accessibility of the paper. Furthermore, the appendix includes detailed hyperparameter information, essential for validating experiments in this hyperparameter-sensitive area and supporting the reproducibility of the study.

### Weaknesses
First and foremost, this paper resembles the CLAM paper strongly, particularly in the introduction section, which feels almost identical. Given that this work relies heavily on CLAM, it is essential to revise and reframe the introduction to clearly distinguish this approach as more than a direct application of CLAM with an autoencoder. Establishing this work's unique contributions will help clarify its originality and value.

While the authors claim that the method does not require $\gamma$ tuning, it does not appear to resolve the underlying parameter sensitivity issue. In fact, it introduces additional hyperparameters that need careful adjustment. The chosen values suggest that previous tuning decisions are not reusable across different datasets and models, requiring high tuning complexity. This is particularly challenging, given that model training for reconstruction must occur simultaneously with the tuning process.

 An essential missing experiment is an ablation study on the latent dimension \( m \). Evaluating this parameter, even for one of the models used, would provide valuable insights into the impact of latent dimensionality on performance.

### Questions
How is the number of centers (clusters) chosen? It’s unclear whether this is a fixed number or if it corresponds to the number of classes in the dataset.

Why is reconstruction considered significant in this context? It seems that the assumption of minimal information loss in the latent space is a prerequisite for effective clustering. To my understanding, a low reconstruction loss simply indicates that good clustering may be achievable, rather than serving as a goal in itself.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3