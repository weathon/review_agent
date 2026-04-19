# Beyond Vanilla Variational Autoencoders: Detecting Posterior Collapse in Conditional and Hierarchical Variational Autoencoders

- Decision: Accept (poster)
- Scores: 6, 6, 6, 6, 8

## Abstract
The posterior collapse phenomenon in variational autoencoder (VAE), where the variational posterior distribution closely matches the prior distribution, can hinder the quality of the learned latent variables. As a consequence of posterior collapse, the latent variables extracted by the encoder in VAE preserve less information from the input data and thus fail to produce meaningful representations as input to the reconstruction process in the decoder. While this phenomenon has been an actively addressed topic related to VAE performance, the theory for posterior collapse remains underdeveloped, especially beyond the standard VAE. In this work, we advance the theoretical understanding of posterior collapse to two important and prevalent yet less studied classes of VAE: conditional VAE and hierarchical VAE. Specifically, via a non-trivial theoretical analysis of linear conditional VAE and hierarchical VAE with two levels of latent, we prove that the cause of posterior collapses in these models includes the correlation between the input and output of the conditional VAE and the effect of learnable encoder variance in the hierarchical VAE. We empirically validate our theoretical findings for linear conditional and hierarchical VAE and demonstrate that these results are also predictive for non-linear cases with extensive experiments.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors analyze the issue of posterior mode collapse for conditional and hierarchical linear VAEs. They demonstrate conditions required to observe posterior collapse, which provides meaningful insight into the design and training of more complex VAEs.

### Strengths
- Technical derivations seem correct, and the paper methodology is solid.
- Theoretical insights into the understanding and design of VAEs.

### Weaknesses
The reviewer appreciates little contribution in the paper. Yes, the authors generalize the results obtained for the linear VAE by Wang & Ziying (2022) and Dai et al. 2020 to linear CVAEs and hierarchical linear VAES, and this contribution is acknowledged. However, I wonder if generalizing the analysis to these new models resulted in different conclusions or insights about avoiding posterior collapse. In other words, is anything in Table 1 significantly different from what we learned from Wang and Ziying? At the end, the conclusion is that both $\beta$ and $\eta_{dec}$ should be small ...

### Questions
Please clarify what new designing insights are obtained from your results that couldn't be guessed from other works in the literature on analytical understanding methods for VAEs.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents the first theoretical analyses of posterior collapse in the conditional VAE and markovian hierarchical VAE. Both models contrast with standard VAEs, which have theoretical results related to posterior collapse, in that they have more complex latent structure. Mathematical analyses derive conditions under which we should see posterior collapse for both classes of models. Qualitative observations are extracted from the theorems. Experimental results on MNIST test and validate the qualitative observations.

### Strengths
- The VAE is a widely used, classic generative modeling framework. Theoretical results providing a better understanding of performance, and lack thereof, are interesting and valuable. 
- The theoretical results and qualitative observations provide satisfying and interesting insights into the behavior of the model. 
- The empirical experiments provide a nice, if modest, evaluation of the predictions.

### Weaknesses
- The primary weakness was in exposition. The paper tends to discuss intuitions after theorems, which makes the theorems a bit challenging on first read. 
- Moreover, at several point the authors assume a lot of the reader. Specifically, consider the case of unlearnable Sigma, and comparisons between theorem 1 and the previous work (the result from which is not stated in the text). 
Taken together the paper was more difficult than need be to understand. 

Minor comments: 
"Interestingly, we find that the correlation of the training input and training output is one of the factors that decides the collapse level" Why is this interesting? 
- "We study model with" typo
- "beside the eigenvalue" typo
- " i) we characterize the global solutions of linear VAE training problem for unlearnable Σ case, which generalizes the result in (Wang & Ziyin, 2022) where only the unlearnable isotropic Σ is considered, and ii) we prove that for the case of unlearnable Σ, even when the encoder matrix is low-rank, posterior collapse may not happen. Thus, learnable latent variance is among the causes of posterior collapse, opposite to the results in Wang & Ziyin (2022) that it is not the cause of posterior collapse." This warrants some explanation. Why is the conclusion opposite? 
- What does it mean for Sigma to be unlearnable? This isn't properly explained. 
- What is the meaning of Theorem 1? It isn't yet clear to me what we are proving or why? 
- "We note that our results generalize Theorem 1 in (Wang & Ziyin, 2022) where σi’s are all equal to a constant." It would really be helpful to the reader to present the previous result and explain clearly why. 
- Ok the explanation after the theorem is helpful. It would be preferable to help readers see what is coming ahead of time, too. 
- I am still unsure about what an unlearnable Sigma is.

### Questions
My main questions are related to the limitations. I would like to hear how the authors would plan to revise to clarify and sharpen exposition.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a theoretical and empirical analysis of the problem of posterior collapse in two types of VAE, specifically conditional and hierarchical. Under a linear VAE setting they prove certain specific conditions as contributors to posterior collapse. They then conduct various experiments on MNIST, focusing in their empirical work on non-linear VAEs, and demonstrate that their theoretical results line up with the experimental findings of what settings lead to collapse.

### Strengths
- The paper puts forward theoretical analysis of posterior collapse in previously unstudied models that are more complex than vanilla VAEs
- The paper performs experiments that corroborate the theoretical claims, albeit in a more realistic nonlinear setting

### Weaknesses
- Much of the empirical results are relegated to the appendix with the paper itself being somewhat light on experiments. This isn’t necessarily a bad thing but it would be nice to see more thorough experiments in the main paper, e.g. vanilla VAE, linear CVAE/MHVAE, learnable vs. unlearnable variance, just for comparison.
- The theoretical results are generally restricted to highly specific scenarios, such as linear and two latent for MHVAE, and the setting of both variances being learnable in MHVAE is not considered. It would be nice to see more theoretical motivation for why these results might generalize to more realistic training settings, but instead that is largely directed towards prior work.
- Even in the empirical setting the majority of models studied are restricted to fairly shallow networks. The results are entirely on a simple dataset, MNIST, and therefore it’s hard to get a sense for how the experiments would behave in a more complex setting with deeper networks, modern architectures, and less standardized data.
- One of the most noticeable absences is that this paper doesn’t consider the effects of SGD and optimization strategies. The setting where global solutions are directly obtainable is not really representative of most real world VAEs which would involve deep nonlinear networks that are most likely to only reach a local optimum. While the experiments themselves do use Adam to train the models, there’s not much analysis of how the optimization strategy interacts with other phenomena.
- Many of these weaknesses could be levied against prior work in this space. For what it’s worth this paper does go beyond those works in that they do rigorous theoretical analysis of more complex model types.

### Questions
You claim that Wang & Ziyin 2022 arrived at the opposite conclusion as you regarding the role of learnable latent variance. Can you say more about why you would have found different results and what that means?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work extends previous theoretical work on the linear variational autoencoder (VAE), showing the conditions under which linear conditional VAEs (CVAEs) and Markovian hierarchical VAEs (MHVAEs) will exhibit posterior collapse. Empirical experiments are conducted that supports theoretical results in both linear and nonlinear cases.

### Strengths
- **The results are potentially novel and can have interesting implications.** This work extends the previous results on more complex VAE models which are widely used in practice. This might give practical directions for people deploying these variant of VAEs that previous work did not cover.
- **Experiments clearly support the main theoretical results.** Both linear and nonlinear models are trained for different parameters and subsets of data. The results very clearly support the theoretical predictions and shows evidence on how the linear results can be predictive of the general nonlinear case.

### Weaknesses
- **The paper seems to claim a result that has already been proven.** Theorem 1 seems to be exactly identical to proposition 2 in (Wang & Ziying, 2022). Maybe the functional form of the result is identical but the authors have more general assumptions on $\sigma_i$ (I doubt this after having scanned the proof in Wang & Ziying). In either case, the statement that this theorem is one of the novel contributions is false or at best misleading.
- **It is unclear how much novelty there is in extending the proof to CVAEs and MHVAEs.** Since I did not follow the proof for theorem 2 and 3 closely, and I am also not familiar with the literature, I cannot be certain whether extending the existing results on linear VAEs to CVAEs and MHVAEs is methodologically novel. In particular, the expression for $\omega^*$ in theorem 2 seems quite similar to that of theorem 1 (which I do not consider novel, see previous point), and it is unclear to me if the derivation follows almost identical strategies. I think it might be helpful if the authors can discuss why extending the proof to the CVAE and MHVAE cases are interesting and nontrivial. This might just be a minor concern as one can argue that the implications of the results might be the more important and interesting aspect of this work.
- **Graphics are not very easy to read.** It is difficult to see how the results shown in the plots support the theoretical results. This is only a minor critique and can be easily improved by using a different coloring scheme or just rearranging the labels in a vertical list and/or remind the reader what the theoretical predictions are (larger $\beta1$ and smaller $\beta2$ is good, etc.).

### Questions
- Can you clarify on the novelty claim for theorem 1?
- On the hierarchical VAE results, you mentioned that it is advisible to create separate maps between the input data and each level of hierarchy, which seems closely related to the ideas in (Sønderby et al., 2016; Zhao et al., 2017; Vahdat & Kautz, 2020). Maybe it will be helpful to discuss these models since the theoretical results here seem to provide motivation for them?

References:

Casper Kaae Sønderby, Tapani Raiko, Lars Maaløe, Søren Kaae Sønderby, and Ole Winther. Ladder variational autoencoders. Advances in neural information processing systems, 29, 2016.

Zhao, S., Song, J., & Ermon, S. (2017). Learning hierarchical features from generative models. arXiv preprint arXiv:1702.08396.

Arash Vahdat and Jan Kautz. Nvae: A deep hierarchical variational autoencoder. Advances in neural information processing systems, 33:19667–19679, 2020.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 5

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work provides a theoretical analysis of posterior collapse in Variational Autoencoder (VAE) models. First the authors extend previous results for the linear VAE case (Variational Autoencoders with only a single linear layer as encoder and decoder) and provide a novel condition for posterior collapse. Then these results are extended and the authors provide conditions for posterior collapse for two further VAE models: the Conditional VAE (CVAE), and the Markovian Hierarchical VAE (MHVAE).

### Strengths
This work provides important theoretical insights into the conditions for posterior collapse in Variational Autoencoders. Although these conditions are only presented for the linear case, and for MHVAE the case that only one of the two encoder variances is learnable, this work provides an important step forward for the understanding of the inner workings of VAEs and provides the basis and theoretical tool set for future research. The empirical evaluation confirms that the theoretical insights gained for the single layer encoder and decoder case extend, at least empirically, to the nonlinear (multi layer) case.

### Weaknesses
The conditions provided in the paper are only for the linear (single layer encoder and decoder) case, which of course limits the practical applicability of the framework. Yet, I believe that the work provides an important step forward in the understanding of VAEs and do not see a big drawback in this.

### Questions
In Fig. 3 (b) the case for beta_1=1.0 and beta_2=1.0 has been left out. Just for completeness it might be interesting to visualize the result for this "standard" ELBO (beta_1=beta_2=1.0).

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent
