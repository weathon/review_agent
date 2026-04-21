# Incorporating Domain Knowledge in VAE Learning via Exponential Dissimilarity-Dispersion Family

- Avg Score: 5.20
- Decision: Reject
- Scores: 5, 5, 5, 6, 5

## Abstract
Variational autoencoder (VAE) is a prominent generative model that has been actively applied to various unsupervised learning tasks such as representation learning. Despite its representational capability, VAEs with the commonly adopted Gaussian settings typically suffer from performance degradation in generative modeling for high-dimensional natural data, which is often caused by their excessively limited model family. In this paper, we introduce the exponential dissimilarity-dispersion family (EDDF), a novel distribution family that includes a dissimilarity function and a global dispersion parameter. A decoder with this distribution family induces arbitrary dissimilarity functions as the reconstruction loss of the evidence lower bound (ELBO) objective, where the model leverages domain knowledge through this dissimilarity function. For VAEs with EDDF decoders, we also propose an ELBO optimization method that implicitly approximates the stochastic gradient of the normalizing constant using log-expected dissimilarity. Empirical evaluations of the generative performance show the effectiveness of our model family and proposed method in the vision domain, indicating that the effect of dissimilarity determines the criteria of representational informativeness.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces the exponential dissimilarity-dispersion family (EDDF) as the distribution of the decoder $p_{\theta}(x|z)$. Adopting EDDF can use different dissimilarity functions to define the reconstruction loss and provide an implicit optimization of the dispersion parameter $\gamma$ that balances the rate-distortion trade-off without hand-tuning as $\beta$-VAE.

### Strengths
1. The paper is well-written and easy to follow.
2. The proposed method is simple and well-motivated.
3. Code is released.

### Weaknesses
1. The general derivation is quite similar to $\sigma$-VAE, except for the extension of MSE to different dissimilarity functions.
2. Although the experiments are designed from different perspectives to verify the effectiveness of the proposed methods, I have the following concerns:
    * When choosing MSE as the similarity metric, why does there exist a performance difference between $\sigma$-VAE and the proposed method in Tab.3?
    * In Tab.5, different dissimilarities $d$ are chosen based on the validation set for different datasets. How are the hyper-parameters tuned for the baselines? It seems that $\beta$-VAE with different $\beta$ in Tab.3 performs better than some baselines. Why not compare to $\beta$-VAE with a hyper-parameter tuned on the validation set?

### Questions
Please see the previous section.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focused on improving variational autoencoder (VAE). Instead of adopting Gaussian settings, which is shown to violate domain-specific properties, this paper introduced a novel distribution family, i.e., the exponential dissimilarity-dispersion family (EDDF).  Correspondingly, an approximated algorithm using a log-expected dissimilarity loss is introduced to optimize VAE with the proposed EDDF decoders. Empirical validation is provided on a toy 1-D example. Effectiveness of the proposed model is evaluated on vision datasets.

### Strengths
**Originality:** this paper introduced a new distribution family for VAE. It also shows that some well-known distribution families can be interpreted as a subset of the new distribution family (EDDF). Correspondingly, an approximate algorithm is proposed for the training of VAE using the new EDDF decoder. 

**Presentation:** this paper overall is well presented. The flow of the paper is smooth and easy to follow.

### Weaknesses
**Technical novelty:** the section 4 on VAE optimization is showing the derivations for the approximate optimization of VAE. However, its technical novelty is not clear to me. How does the proposed algorithm relate to or different from existing methods (e.g., Rybkin et al., 2021; Lin et al., 2019)? 

Is there any theoretical guarantee or analysis to quantify the approximation? Simply using a 1-D toy model is not sufficient to valid the claim. 

**Significance:** the significance of the paper is unclear to me.  1) Compared to other variants of VAE, the proposed approach is not achieving the best performance on all the datasets (Table 3). 2) how does the proposed model compare to other generative models (e.g., GAN)? Though the goal of the paper is to improve VAE, it is important to compare to other generative models to understand its significance. 

**Presentation:** The term ‘domain knowledge’ used in the paper is not fitting the presentation well in my eyes.  How does the domain knowledge relate to the proposed EDDF decoder is not well defined at all.

### Questions
Please refer to the weakness section for my major concerns. In addition:

1.	what’s the computational cost of the proposed method?
2.	Can you give some insights or explanations on why the proposed model performs better on some of the datasets but worse on others in Table 3? 
3.	Can you give any justifications on how the proposed distribution family encodes domain knowledge?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In the paper, the authors formulate a method to incorporate domain knowledge into VAEs by specifying a dissimilarity function. This function is incorporated into the decoder's output distribution with proper normalization. Further, they reformulate the ELBO of the VAE based on the specified disimilarity function and provides analytical values for the normalizing constant in the limit $\gamma\rightarrow 0$.

### Strengths
A compact and straightforward loss function for VAEs with dissimilarity functions is presented. The methodology is easy to follow and the results are conclusive.

### Weaknesses
The paper's contribution is limited, as dissimilarity functions in the context of generative models have been presented in the past. The ELBO in (25) is probably the leading paper contribution, but beyond this point, I do not see any other contribution that justifies a whole paper around this idea.

### Questions
How do you extend this idea to other generative models such as GANs or latent diffusion models?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper considers the problem of training a VAE with "arbitrary" reconstruction loss, dubbed as a dissimilarity function. To do so, a Gibbs style distribution is introduced --- the Exponential Dissimilarity-Dispersion Family (EDDF) --- which is used to define the probabilistic reconstruction of a latent variable in the VAE. To make this tractable (in particular the corresponding normalization of the EDDF) several approximation are made yielding a simple log-expected reconstruction loss.

### Strengths
- The generalization of the standard VAE's reconstruction loss to that of a general dissimilarity function intuitively makes sense.
- The corresponding EDDF defined also follows celebrated patterns of exponential families and Gibbs-style distributions.
- The corresponding results are promising.

### Weaknesses
- I am a bit unconvinced why defining the EDDF is needed in the loss function of the VAE. (See questions below)
- Parts of the reasoning in Section 4 seems unclear. (See questions below)

### Questions
Why is EDDFs and defined explicitly?

- One aspect I am unclear about is why the reconstruction loss is not just defined directly as $\mathbb{E}[d(\textbf{x}, \hat{\textbf{x}}(\textbf{z}))]$ (expectation wrt ${q_{\phi}(\textbf{x}, \textbf{z})}$)? Even the log-expected reconstruction loss in Eq. (24) can be considered as an upper bound of taking log of the dissimilarity measure $d$, ie, $\log d(\textbf{x}, \hat{\textbf{x}}(\textbf{z}))$ (using Jensen's).
- The above perspective also skips the need for approximating the normalizer of the EDDF. I feel like this perspective can be made especially since the decoding is done deterministically in practice and thus an explicit density $p_{\mathbf{\theta}}(\textbf{x} \mid \textbf{z})$ does not seem to be needed. As such, additional input for why $p_{\mathbf{\theta}}(\textbf{x} \mid \textbf{z})$ should be defined explicitly would be great (and if there is prior work which took such a simplistic approach).
- One further note, the above perspective seems to fit in the narrative of "generalized variational inference" (See, eg, "An optimization-centric view on Bayes' rule: reviewing and generalizing variational inference" Section 4.4.4, switching log-likelihood for a general dissimilarity function).

Clarity in Section 4

- Eq. (20) is misleading as from Eq. (29) the equality should instead be an upper bound due to Jensen's.
- Why is the M / 2 factor not utilized in Eq. (24)?
- Above Eq. (25) there is a mention of "well-trained autoencoder" with a condition on $D$. I believe this should be $\gamma$ instead?

Other:

- It is worth clarifying what the regularization loss is in Eq. (25). I believe it is Eq. (7) with isotropic Gaussian (after digging through part of the code), but I am unsure given the current text.
- Something to add to Section 3.1: It might be worth mentioning that exponential family distributions can be explicitly be characterized as a density $ \propto \exp(- B_f(\cdot, \textbf{m}))$ where $ B_f$ is a Bregman divergence, see, eg, "Information Geometry and Its Applications" Section 2.7.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper is concerned with the possibility to replace the normal distribution in the VAE framework by a distribution from some family that includes a dissimilarity function and a dispersion parameter. The paper seems to propose minimizing the ELBO objective, contrary to the usual maximization, and claims that a decoder with such a distribution can induce arbitrary dissimilarity functions as the reconstruction loss. Some empirical checks of the method, on simple datasets in visual domain, are described.

### Strengths
Paper is devoted to an exploration of possibilities for modifications of widely used VAE method

### Weaknesses
1) The writing of the paper is somewhat obscure, especially in what concerns long sentences, cf below. 

2) The rational to consider the minimization, instead of the standard maximization, of the ELBO objective is not clear.

3) The results seems to be too incremental, many previous works have proposed to replace the Gaussian distribution in VAE by some other distributions. 

4) In  2 out of 5 cases in the  described empirical results, the baselines from previous papers performed better (Table 5)

5) The standard deviations in the experiments are not reported.

6) Only two out of many metrics evaluating VAE  quality are calculated.  In particular, the reconstruction errors are not reported in the comparison tables. 

7) Rather simple datasets are considered only. Also only the visual domain is considered. 

Some remarks :

page 1 : "VAEs with the commonly used settings practically suffer from performance
degradation, such as reconstruction fidelity and generation naturalness" -> perhaps ... insufficient reconstruction fidelity and... Since as it stands, both the "reconstruction fidelity" and "generation naturalness" appear as undesired/bad qualities. 

pages 2-3: "...with negative ELBO objective... whose maximization ..." - usually the negative ELBO objective is minimized

page 4, last equation in (16) - it seems that the paper's method seeks to maximize the reconstruction loss, given by a positive distance cf eqs (9), (18).

Following improvements made during rebuttal I'm raising the score, however some issues require further work.

### Questions
Some suggestions: To demonstrate the effectiveness of the method, the  standard deviations should be measured in the experiments for various random seeds. Also, more than just two metrics should be measured. The comparison with baselines performance in other domains and on more complex real- world datasets is also necessary for evaluation of the method's practicality.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
