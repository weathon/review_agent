# Neural Sinkhorn Gradient Flow

- Decision: Reject
- Scores: 5, 5, 6, 5, 6

## Abstract
Wasserstein Gradient Flows (WGF) with respect to specific functionals have been widely used in the machine learning literature. 
Recently, neural networks have been adopted to approximate certain intractable parts of the underlying Wasserstein gradient flow and result in efficient inference procedures. In this paper, we introduce the Neural Sinkhorn Gradient Flow (NSGF) model,  which parametrizes the time-varying velocity field of the Wasserstein gradient flow w.r.t. the Sinkhorn divergence to the target distribution starting a given source distribution.  We utilize the velocity field matching training scheme in NSGF, which only requires samples from the source and target distribution to compute an empirical velocity field approximation. Our theoretical analyses show that as the sample size increases to infinity, the mean-field limit of the empirical approximation converges to the true underlying velocity field. With specific source and target data samples, our NSGF models can be used in many machine learning tasks such as unconditional/conditional image generating, style transfer, and audio-text translations.  Numerical experiments with synthetic and real-world benchmark datasets support our theoretical results and demonstrate the effectiveness of the proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors introduce Neural Sinkhorn gradient flow, which is a Wasserstein Gradient Flow wrt to the Sinkhorn divergence. The authors show that the velocity field can be calculated using the Sinkhorn potentials. This allows training a neural network approximating the velocity field. Furthermore, a mean field limit is established. The algorithm is evaluated on a toy example, MNIST image generation and CIFAR10 image generation.

### Strengths
The authors do a good job at explaining the underlying concepts of their algorithms. The maths is nicely done. The core idea is very neat and the cifar10 results seem to be good quantitatively wrt other gradient flow works.

### Weaknesses
1) The article is full with typos. Just to name a few: "piror", "Sinkhron", "Experimrnts", "speedest descent", question mark in the appendix and so on. Please fix those. 

2) the authors write "We do not compare with extant neural WGF methods on MNIST because most of the neural WGF
methods only show generative power and trajectories on this dataset and lack the criteria to make
comparisons." There are several papers (also gradient flow based ones), which evaluate a FID on MNIST. Please provide it as well. 

3) Also many of the MNIST digits appear flipped. Did the authors use data augmentation there? Also there seems to some slight noise present the generated MNIST digits. 

4) Although the CIFAR10 value seems good, there are unfortunately no generated images provided. It is standard practice to sample many images in the appendix. 

5) It is unclear what the trajectories show. Does it show the particle flow or the trained Neural Sinkhorn Gradient Flow? 

6) The statement of theorem 2 is incorrect. I guess the authors do not want to sample the Euler scheme (eq 14) but the continuous gradient flow, otherwise the statement would need to depend on the step size $\eta$. 

7) In the proof of Theorem 2: Please provide a proof (or reference) why the mean field limit exists. Or do you mean the gradient flow starting at $\mu_0$ with target $\mu$ (first two sentences).

8) Later in that proof: why does there exists a weakly convergent subsequence of $\mu_t^M$? Further, I cant find the definition of $U_{\mu}$. 

9) The code is not runnable, as the model (or any checkpoints) are not provided.

10) From how I understood it, the learning of the velocity field is batched, i.e., one trains for different sets of $(z_i,x_i)$. Since the Sinkhorn dynamic describes an interacting particle system I dont see how this should be possible. To be more precise, one particle $\tilde{x}$ could be sent to $x_0$ in the first batch, but to a totally different particle $x_1$ in another one, depending on the drawn prior and target samples. Are the positions of the other particles also input to the neural network (i.e by putting them in the channels)? Please elaborate.

### Questions
See weaknesses section. Overall I really like the idea, but the weaknesses prevent me from giving a higher score. It seems like the paper was rushed and is currently not ready for publication. I am willing to raise my score, if the authors address these issues.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a novel way to train generative models. The authors want to approximate the gradient flow in the Wasserstein space.  They want to approximate the vector field which transports the source distribution to the real-data empirical distribution while minimizing the Sinkhorn divergence. The authors showed the analytical form of the vector field when one considers the Sinkhorn divergence and then they explain how to learn this vector field with a neural network through the simulation of a probability path. They showed that their procedures recover the true probability path when the number of iid samples goes to infinity. Finally, they validate their proposed method on several image-generative tasks.

### Strengths
i) The motivation and the introduction are clear

ii) Regressing vector fields has been a recent and popular approach with many different applications in machine learning. The proposed approach is interesting and appears to be novel. The theoretical results also show that the proposed method has appealing properties. 

iii) The authors also provided several experiments showing interesting results from their methods.

### Weaknesses
The first thing I would like to highlight is that I have checked the provided code. I see several inconsistencies and weaknesses between the provided code and the paper:

1. There are several differences in the empirical implementation between the paper and the code. In Appendix A, the authors state that they are computing the entropic potential through stochastic optimization algorithms [Genevay et al, 2016]. However, this is not what is done in practice according to the provided code. In practice, the authors compute the potential between mini-batches of samples, they sample a minibatch of cifar10 experiments, then sample a minibatch of the source Gaussian, and simulate the gradient flows between the two minibatches. This style of minibatch approximation induces a bias that should at least be mentioned in the main paper but also discussed. Indeed, the authors do not compute the true Sinkhorn divergence but a minibatch approximation of it; this approximation is slightly different than the one from [1,2] and that should be discussed. I understand the reason why the authors use this approach (decreasing the cost of this preprocessing step), but this is not what they say they do in Appendix A. In that regard, the paper is much closer to the minibatch optimal transport Flow Matching [Pooladian et al., Tong et al] and Appendix A deserves a major revision.

2. With the provided code, there are several insights that should be discussed in the paper. In the provided cifar experiments, the number of Gaussian samples used is 50000 samples. This number is extremely low to approximate the semi-discrete OT. Therefore, a discussion regarding the statistical performance of the method is needed in my opinion.

3. As your method requires the simulation of the probability path, I wonder about the training time between your method and the recent Flow Matching approaches which are simulation free.

4. There are many typos in the paper (including in titles: ie ExperimRnts, Notaions) that lead to poor clarity...

5. The experiments include two toy datasets (synthetic 2D and MNIST). I would like to know how the method performs on other big datasets (Flowers, CelebA) or on other tasks such as single-cell dynamics [4].

6. The related work on optimal transport is incomplete. Several works used the sliced Wasserstein distance to perform gradient flows [3].

[1] Learning Generative Models with Sinkhorn Divergences, Genevay et al, AISTATS 2018
[2] Learning with minibatch Wasserstein, Fatras et al, AISTATS 2020
[3] Sliced-Wasserstein Flows: Nonparametric Generative Modeling via Optimal Transport and Diffusions
[4] TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics

### Questions
1. [Pooladian et al., Tong et al.] proved that when the minibatch increases, they get closer to the true optimal transport cost (W_2^2). The interest of their method is that they can rely on minibatches and learn the vector field from an unlimited number of minibatches. Could you follow a similar approach and simulate the gradient flow during training? While it would be an expensive step in training, it might improve the metrics on the different generative model experiments.

2. What is the performance of your method concerning the number of simulation steps (ie Euler integration and its learning rate)?

3. What is the time of the preprocessing step concerning the training time?

4. Could you compare your method with OT-CFM [Pooladian et al., Tong et al.] on the synthetic data? I am curious to compare the differences.

In my opinion, the mentioned weaknesses have to be revised and this paper should go under a major revision. I deeply think that the experimental section should better highlight what is done in practice and the theoretical section should mention the different biases (statistical and minibatch). Therefore, I recommend rejecting the current manuscript as it does not meet the ICLR acceptance bar.


----- EDIT POST REBUTTAL -----

I thank the authors for their answers. I have read the updated manuscript. While it is now better than before, I suggest they add a limitation section where they describe the different biases in their algorithm. I understand the motivations of the paper. Overall, I think that the manuscript deserves another round of reviews but I have decided to move my score to 5 as they have given good answers.

### Soundness
2 fair

### Presentation
2 fair

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
Through a series of approximations (and at times, really, relaxations) the authors show that the Sinkhorn gradient flow from one measure to another can be learned.  They do this by first reducing their relaxed problem to a vector field matching problem, and then proposing a neural network-based Algorithm for matching the Sinkhorn-Wasserstein flow's vector field by a neural network (though no convergence/approximation guarantees are proven).
The problem is interesting, and its solution is sufficiently novel to merit publication.

### Strengths
The problem is natural to study, the results are mathematically correct, and the experiments are convincing.

### Weaknesses
While the paper is mathematically correct, it does not provide theoretical justification for one of its main components, namely showing that approximate vector field matching yields approximate solutions for all time $t$.  I feel that without this guarantee, there is a gap in the theoretical viability of this model.  Nevertheless, this is a minor point since the length of a conference paper does not allow one to treat every such point.

There are minor typos throughout. 
* E.g. euclidean instead of Euclidean
* $lim$ instead of $\lim$ atop page 15 in the appendix
* The positive scalar $\delta$ is not defined in the proof of Theorem $1$
* In the statement of Lemma 3: "teh" should read "the"

Some references are obscure
* For The fact that $\mu + t\delta \mu$ converges weakly to $\mu$, perhaps it is worth simply noting that due to linearity of integration (wrt to the measure term).

### Questions
Can it be shown that approximate vector field matching yields approximate solutions for all time $t$?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper under consideration deals with the standard generative modelling setup (image generation from noise). To solve this problem, the authors propose to model the gradient flow w.r.t. the Sinkhorn divergence. The paper utilizes an explicit (forward) Euler discretization scheme, i.e., given a distribution $\mu_t$ at the current time step $t$, the proposed method aims at finding the subsequent distribution $\mu_{t + 1}$ following the gradient of the Sinkhorn divergence at point $\mu_t$. The authors validate their methodology on toy 2D setups as well as standard image benchmarks (MNIST and CIFAR10).

**Post-rebuttal update:** I thank the authors for the detailed answer. The majority of my concerns are properly addressed. I rise my score. However, I still tend to reject the paper. Also I agree with reviewer KKim that minibatch OT approximation should be discussed more thorougly. Thank you.

### Strengths
To the best of my knowledge, the framework of the gradient flow w.r.t. Sinkhorn divergence for pure generative modelling has not yet been considered. This indicates that the paper is indeed bringing something novel to the ML community. At the same time, the idea of the Sinkhorn gradient flow has already arisen in previous research. In particular, [A] solves Sinkhorn barycenter problems by adjusting a generative distribution towards the barycenter distribution with the help of a procedure called “functional gradient descent” which is actually the discretization of the gradient flow w.r.t. the sum of Sinkhorn divergences to the target distributions. At the same time, it is worth mentioning, that [A] just simulates particles and does not build a generative model.
Regarding the other strengths of the paper, I would like to note the well-organized Experiments section.

[A] Sinkhorn Barycenter via Functional Gradient Descent, NeurIPS’2020

### Weaknesses
- Some theoretical results from the paper are known. For example, the statement of Theorem 1 could be found in [B] (eq. 26) or [C] (eq. 8). 
- The quality of the code provided is not good. There is no README/or other instruction to run the code. There are imports of non-existing classes. So, there is no possibility of checking (at least, qualitatively) the provided experimental results.

From my point, the main weakness of the proposed paper is the limited methodological contribution. The authors simulate the particles of data following Sinkhorn divergence - as I already mentioned, this is not a super fresh idea. To make a generative model from these simulated trajectories, the authors simply solve the regression task to learn the local pushforward maps. And that is it. Combined with the fact, that the practical performance of the proposed approach is far from being SOTA in the generative modelling, the overall contribution of the paper seems for me to be limited.

### Questions
- My main question (and, probably, one of the main of my concerns) is regarding the proposed methodology. The authors propose to compute certain $\mathcal{W}_{\varepsilon}$ potentials (on discrete support of available samples) and then somehow take the gradients of these potentials w.r.t. the corresponding samples (eq. (13)). From the paper it is not clear how to compute the gradients, because the obtained potentials look like vectors of sample size shape, which are obtained through the iterations of the Sinkhorn algorithm. As I understand, in practice, the authors utilize SampleLoss from the geomloss package ([B]).  The outcome of this observation is that [B] should be properly cited when deriving the algorithm (section 4.2). I recommend authors explicitly use SampleLoss in the algorithm's listing. It will contribute to the clearness of what's going on. 
- The vector field of the Sinkhorn gradient flow is estimated by empirical samples. It is not clear how well this sample estimate approximates the true vector field. This point should be clarified. Note, that Theorem 2 works only for mean-field limit. 
- In the Introduction section, the authors consider a taxonomy of divergences used for gradient flow modelling, namely, "divergences [...] with the same support" and "divergences [...]  with possible different support". As I understand, the first class is about $f-$ divergences and the second class is about the other types (like Sinkhorn, MMD etc.). I have a question regarding the provided examples of works which deal with the former or the latter type of divergences. The fact is that the works [D], [E], [F], [G] deal with KL-divergence (or f-divergence) minimization. That is why I wonder why did the authors classify them as the second class.
- A good work regarding poor expressiveness of ICNNs is [H].
- What is the “ground” set ($\S$ 3.1, first line).
- Table 1. What are the differences between 1-RF, 2-RF and 3-RF methods?

[B] Interpolating between Optimal Transport and MMD using Sinkhorn Divergences, AISTATS’2019

[C] Sinkhorn Barycenters with Free Support via Frank-Wolfe Algorithm, NeurIPS’2019

[D] Large-scale wasserstein gradient flows. NeurIPS'2021

[E] Optimizing functionals on the space of probabilities with input convex neural networks. TMLR

[F]  Proximal optimal tranport modeling of population dynamics. AISTATS

[G] Variational wasserstein gradient flow. ICML

[H] Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark. NeurIPS’2021.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces the idea of learning a time-dependent velocity field of the Sinkhorn Wasserstein gradient flow from samples from the target distribution to calculate the empirical velocity field approximations. The paper supports its claim by showing that the mean-field limit of this process recovers the true Sinkhorn Wasserstein gradient flow. They also validated the process with some empirical studies.

### Strengths
The paper is well written and easy to follow. The proofs and arguments in the appendix are well-typed out and clear.  There are some nice diagrams in the empirical section to supports the claim the authors are making.

### Weaknesses
I think the experiments could be more extensive. One thing about this method is to investigate the number of samples needed. effectively learn the velocity field. This is one important experiment missing as is remains unclear how sample-efficient the proposed method is. It would also make the paper more completing if the method is applied to generative models that output discrete random variable like binary mnist or even language modelling.

### Questions
One possible question is what happens if we change the source distribution to be closer to the target distribution like it was from a generator how would the method perform there. Another question is to better understand the sample complexity of the method as the current method may not be sample efficient due to the empirical distribution being approximated using the samples.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
