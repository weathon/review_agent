# Shared-AE: Automatic Identification of Shared Subspaces in High-dimensional Neural and Behavioral Activity

- Decision: Accept (Poster)
- Scores: 8, 6, 3, 5

## Abstract
Understanding the relationship between behavior and neural activity is crucial for understanding brain function. An effective method is to learn embeddings for interconnected modalities. For simple behavioral tasks, neural features can be learned based on labels. However, complex behaviors, such as social interactions, require the joint extraction of behavioral and neural characteristics. In this paper, we present an autoencoder (AE) framework, called Shared-AE, which includes a novel regularization term that automatically identifies features shared between neural activity and behavior, while simultaneously capturing the unique private features specific to each modality. We apply Shared-AE to large-scale neural activity recorded across the entire dorsal cortex of the mouse, during two very different behaviors: (i) head-fixed mice performing a self-initiated decision-making task, and (ii) freely-moving social behavior amongst two mice. Our model successfully captures both `shared features', shared across neural and behavioral activity, and `private features', unique to each modality, significantly enhancing our understanding of the alignment between neural activity and complex behaviors. The original code for the entire Shared-AE framework on Pytorch has been made publicly available at: \url{https://github.com/saxenalab-neuro/Shared-AE}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a latent-space disentangling autoencoder to identify shared and private latent features from multi-modal neural and behavioral data. Proposed disentanglement is based on a Cauchy-Schwarz divergence based regularizer applied between different components (private and shared features) of the latent representations obtained via behavioral and neural encoders. Both inter and intra modality regularization losses are combined in addition to the standard autoencoder loss. Experimental analyses are performed first on a simulated dataset, and then on different complex behavioral datasets with neural recordings from mice.

### Strengths
- Consistent experimental findings based on a simple disentangled representation learning model with a tailored training objective.
- It has a unique empirical strength with a focus on neural data analysis from complex multi-modal social behavior experiments.

### Weaknesses
- Some of the empirical results need further validation, considering the details present in the Appendix.

### Questions
- The overall idea of disentangling the latent representation space using inter- and intra-modality loss regularizers has been previously explored in several works. There are also actually works proposing a similar autoencoder regularization framework in other settings [Tran et al. "Cauchy–Schwarz Regularized Autoencoder", JMLR 2022]. Perhaps one question that the authors should clarify with a clear statement is their methodological ML novelty (i.e., if the proposed regularized training scheme is completely novel, or if the paper only contains a strong empirical novelty).

- Majority of the results show strong consistency between the disentangled latent features extracted from behavioral and neural data. Regarding the latent space visualizations (UMAP etc), how did the authors determine the latent dimensionality in each experiment? How consistent are these results with respect to changing this dimensionality?

- The dataset retrieved for the 2AFC experiments seems rather small in terms of the number of trials. Also it seems to be divided only once into a train/test split. Therefore, I would ask if the authors performed any CV of the model training process, and evaluate the significance of their results in that sense?

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
4

### Summary
This paper contributes to a growing literature on learning shared representations for multimodal data in neuroscience, where many researchers are interested in learning joint representations of brain data and behavior. Whereas many previous methods have focused on learning shared latent representations by combining latents learned from individual modalities, this work further partitions each modality's latent space into a private latent spaces and a shared latent space, which is linearly mixed with other modalities' shared latents. This separation is engendered by the use of a Cauchy-Schwarz Divergence for aligning shared latents and separating shared from private latents. Experiments on one synthetic and two real data sets show suggestive links between brain data and behavior, though what one is to make of these is a bit unclear. Moreover, the technical contribution of the paper is perhaps small when considered in light of other work cited.

In all, this is a solid paper that is, in my view, below the bar for acceptance without a more substantial technical contribution.

### Strengths
- Principled approach to structuring latent spaces based on a desired semantics: some information in each latent space is common to all modes, some is private.
- The need for interpretable joint encodings is of high interest in neuroscience.
- CS-Divergence is a reasonable means of effecting the separation of subspaces, and the authors have chosen a pretty reasonable-seeming method of approximating this quantity.
- The experiments on real data use challenging datasets that encapsulate many challenges faced by the community.

### Weaknesses
- It is somewhat unclear what the technical innovation in this paper is beyond the Yi et al. preprint cited, as well as a similar paper by Yi and Saxena at EMBC in 2022 [1]. Both of those works use the same CS divergence setup as here, and I am struggling to see where the technical innovation is (though the application is somewhat different). I don't see the strength of the experimental results here being novel or interesting enough on their own to justify acceptance without an additional technical advance.
- The authors use a latent space partition that is distinct from the Whiteway et al. paper but somewhat related to the Sani et al. work they cite. I realize that the PSID paper is linear, but the Shanechi group also has work on nonlinear methods that preserve this kind of partition (the most recent of which was likely unpublished when this work was submitted) that should probably come in for a fuller discussion.
- In the framing of this work, I don't believe I fully understood the authors' rationale for needing shared vs. private subspaces. It's conceptually interesting, but the experiments simply focus on decoding. In what circumstances do we need such a partition, and what is the sign that not having it is hurting us scientifically? If this were clearer, I think it would be easier for readers to judge the success of the experiments. 
- The figures are all quite small and cramped, making them somewhat hard to parse. It's not always clear what the "win" is with the experiments.

[1]: D. Yi and S. Saxena, "Modeling the behavior of multiple subjects using a Cauchy-Schwarz regularized Partitioned Subspace Variational AutoEncoder (CS-PS-VAE)," 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), Glasgow, Scotland, United Kingdom, 2022, pp. 497-503, doi: 10.1109/EMBC48229.2022.9871466.

### Questions
## My key question:
- What is the technical innovation between this and the two other Yi papers? Is it just a different application?

## Small points of clarification:
- ll. 108-120: the ending here is a bit vague; would help to clarify what these sorts of models would miss in more complex tasks (and hopefully show in experiments)
- ll. 141-152: Model scales as number of pairs of modalities; probably not a limitation in practice, but a few words about scaling might help.
_ l. 165: what is $s'_t$ here? Is it the same as $s_t^{pre}$?
- l. 171: what is $y'$? Is the prime a typo?
- ll. 162-168: It would be nice to have a diagram of this, since one could easily lose track of the different linear models: If I understand correctly: $s^{pre}$ is a linear function of each modality's latents, and $s^i$ is a linear function of both modalities' _shared_ latents.
- l. 196 In Eq 2, how well does this estimator do in moderate-sized latent spaces? Is it a reasonable estimator? One is effectively saying that the density belongs to a reproducing kernel Hilbert space, right?
- ll. 259-264: This description is a bit terse and may be hard to follow for readers (like me) who were not familiar with this dataset. I realize details are in the supplement, but the main text should be a bit more self-contained.
- ll. 318-321: Why, exactly, do we need a strong separation between modalities? What is the use case? I realize it affects decoding performance (e.g., Figure 3) but what might we use this analysis to conclude in an experiment?
- Figure 4A: Sorting the rows and columns by some sort of biclustering algorithm might make the correlation structure more apparent. This matrix plot is not very compelling as presented.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes an autoencoder-based framework for finding shared and private subspaces of multimodal neural and behavioral data. Their goal is to separate the shared subspace from the modality-specific subspaces. To do so, constraints (Cauchy-Schwarz divergence) are added on the distribution of subspaces to encourage/discourage their alignment for this particular goal. This method is evaluated on one simulated dataset as well as two distinct experimental datasets from mice.

### Strengths
1.	The paper is well-presented. The goal is clear and many analyses including both simulated datasets and two experimental datasets with different behavioral complexities are analyzed. 
2.	Various analyses such as connecting behavioral variates to corresponding brain area through shared latents, and investigating the difference between modeling behavior as markers or raw videos in the learned shared features.

### Weaknesses
1.	Methodological novelty seems minimal. The authors note “a novel regularization term designed to identify features common to both behavior and neural activity” as their main methodological novelty. However, a very similar regularization scheme has been previously proposed by Yi et al. (2022). The difference between Shared-AE and this work is not adequately discussed making the methodological novelty unclear. Also, the idea of using CS-divergence instead of standard VAE with KL-divergence is not novel either as previously proposed by Tran et al. (2021).

2.	There are numerous methods on neural-behavioral modeling and finding shared vs. private subspaces, none of which are compared to and many which are not discussed. In general, the manuscript seems to mix up unsupervised latent variable models of neural data with latent variable models of neural-behavioral data in its discussions and writeup. The only neural-behavioral data discussed in Related work (but not compared to) is Schneider et al 2023. Another neural-behavioral model that is cited is Sani et al 2021, but it is not discussed or compared to, and is instead grouped with an unsupervised model of neural data. Another neural-behavioral model in Zhou and Wei 2020 is also simply cited but not compared to. The authors need to separate the neural vs. neural-behavioral models in their manuscript and provide sufficient discussion of differences between other neural-behavioral models with theirs. Comparison to these neural-behavioral models is also needed to show the advantage of this method. In addition to the above cited works, there are also some other very relevant neural-behavioral models that are not cited, for example:

Gondur et al. 2024: This work appeared in the previous ICLR and has a very similar architecture designed for the same purpose using GP-VAE. However, it is not cited, and the key differences are not discussed. Given how closely related this method is to the authors’ work, it can serve as a baseline. 

Hurwitz et al. 2021: This work proposes a sequential VAE for modeling neural-behavioral data. This needs to be cited and discussed.

Sani et al. 2024: This work proposes an RNN-based architecture that separates shared/private subspaces in neural-behavioral data and needs to be cited and discussed. 

3.	Effect of novel terms in the loss i.e., the CS-divergence and their inverses are not assessed. As this is the main addition to a standard multimodal AE architecture in this work, it is crucial to evaluate whether presence of each term contributes to current results or not. Even without these constraints, the reconstruction loss itself can enforce shared vs private subspaces (at least to some extent) as the shared ones reconstruct both modalities whereas private ones reconstruct the specific modality alone.


4.	Lack of baseline comparison in real data analysis. The same baselines used in simulated dataset (Shi et al (2019), Singh Alvarado et al. (2021)) are not shown in real data. Additionally, there are several relevant works on neural-behavioral modeling some of which could be used as baseline to better assess what benefits Shared-AE adds as mentioned in item 2 above. 

5.	The authors claim that their framework is better for more complex/social behavior types than Schneider et al. (2023). But what about all the other neural-behavioral models? Is shared-AE more suitable for complex behavior than others and if so why? This claim does not seem convincing without further comparisons. 

6.	I find calling this method unsupervised very misleading. In the context of neural-behavioral modeling, supervision typically means use of behavior for guiding behavior-related features of neural activity. In this sense, the proposed approach is fully supervised not only during learning but also during inference, putting it in the multimodal family. The manuscript refers to this method as unsupervised throughout the paper including in the title. This needs to be corrected.  

7.	The model uses hyperparameters $(\alpha, \beta, \gamma, \delta)$ to control the contribution of regularizations to the overall loss. However, the effect of these hyperparameters on the results are not investigated. Authors note that the results are robust to changes in the hyperparameters, but I did not find the results that show this robustness. Please provide these.


Minor weaknesses/questions 

1.	Why does the method need to learn two separate shared latents? It seems these two should ideally correspond to the same thing. Why not have a single shared latent which is used in both decoders? 

2.	In the unpaired analysis, is shuffling happening across time? Why does maintaining performance in this scenario indicate avoiding modality leakage?

3.	What is the basis for choosing the state dimension based on Fig. 8? Why are the reconstruction performance vs dimension so noisy in Fig. 8?

4.	What does min/max R2 refer to in Fig. 8?

typographical errors:

-	Line 302: Fig. 4.1 => Fig. 4? 
-	Line 302: missing space between “data” and “(Fig”
-	Fig. 7 caption includes panels E-F while the results are missing.  “E-F: Prediction accuracy for neural activity and behavior under different distance groups.” It seems these panels are not included.
-	Captions for panels B and C of Fig. 10 do not match. It seems the order is wrong.
-	Fig 11 has very tiny titles
-	Line 1002: reference to Fig. A.9.3 is incorrect.

References:

Daiyao Yi, Simon Musall, Anne Churchland, Nancy Padilla-Coreano, and Shreya Saxena. Disentangled multi-subject and social behavioral representations through a constrained subspace variational autoencoder (cs-vae). bioRxiv, 2022. doi: 10.1101/2022.09.01.506091. URL https: //www.biorxiv.org/content/early/2022/09/05/2022.09.01.506091.

Linh Tran, Maja Pantic, and Marc Peter Deisenroth. Cauchy-schwarz regularized autoencoder, 2021. URL https://arxiv.org/abs/2101.02149.

Yuge Shi, N. Siddharth, Brooks Paige, and Philip H. S. Torr. Variational mixture-of-experts autoencoder for multi-modal deep generative models, 2019. URL https://arxiv.org/abs/1911.03393.

Jonnathan Singh Alvarado, Jack Goffinet, Valerie Michael, William Liberti, Jordan Hatfield, Timothy Gardner, John Pearson, and Richard Mooney. Neural dynamics underlying birdsong practice and performance. Nature, 599(7886):635—639, November 2021. ISSN 0028-0836.

Steffen Schneider, Jin Hwa Lee, and Mackenzie Weygandt Mathis. Learnable latent embeddings for joint behavioural and neural analysis. Nature, 617: 360–368 May 2023. ISSN 1476-4687.

Rabia Gondur, Usama Bin Sikandar, Evan Schaffer, Mikio Christian Aoi, and Stephen L Keeley. Multi-modal gaussian process variational autoencoders for neural and behavioral data. In International Conference on Learning Representations, 2024.

Cole Hurwitz, Akash Srivastava, Kai Xu, Justin Jude, Matthew Perich, Lee Miller, and Matthias Hennig. Targeted neural dynamical modeling. Advances in Neural Information Processing Systems, 34:29379–29392, 2021.

Omid G Sani, Hamidreza Abbaspourazad, Yan T Wong, Bijan Pesaran, and Maryam M Shanechi. Modeling behaviorally relevant neural dynamics enabled by preferential subspace identification. Nature Neuroscience, 24(1):140–149, 2021.

Ding Zhou and Xue-Xin Wei. Learning identifiable and interpretable latent models of high-dimensional neural activity using pi-vae.  Advances in Neural Information Processing Systems, volume 33, pp. 7234–7247,  2020. 

Omid G Sani, Bijan Pesaran, and Maryam M Shanechi. Dissociative and prioritized modeling of behaviorally relevant neural dynamics using recurrent neural networks. Nature Neuroscience, 27: 2033–2045, 2024.

### Questions
My questions are the ones raised in weaknesses and Minor weaknesses/questions.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper develops a method for learning shared and private
embeddings between two or more sources of information (modalities).
They apply the algorithm to two problems with behavioral and neural
data as well as an illustrative artificial data problem.

### Strengths
Strengths
- The paper is pretty clearly written

- The artificial data problem helps to clarify how the algorithm works

- The real world experiments are	important to demonstrate practicality in this important area

### Weaknesses
Weaknesses

- Missing comparison to an important related ICLR24 paper (see questions below).    If the authors can compare to results from that algorithm, or otherwise justify the superiority of this method (or superiority in some settings/situations),  I would likely improve my rating.  

- Unclear how parameters were set  (see questions below).

### Questions
Questions

How does this work compare to the ICLR 2024 paper of Gondur, Sikandar,
Schaffer, Aoi, and Keeley (Multi-modal Gaussian Process Variational
Autoencoders for neural and behavioral data)?  That paper also has a
shared multi-modal embedding and separate (within modality) embeddings
and has also been used for complex behavioral tasks ( hawkmoth
tracking a moving flower and limb movement of drosophila with
simultaneous neural recordings).

The paper says that you	examined the influence of latent dimensions on reconstruction
accuracy.  Was that using the test data	or some	separate data?	(If the	test data, how do
you justify that?)   How are the other parameters set? -	the paper is vague on this.

Please elaborate on why equal latent subspace dimensions are required.

### Soundness
3

### Presentation
2

### Contribution
2
