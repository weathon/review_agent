# The Persian Rug: solving toy models of superposition using large-scale symmetries

- Decision: Reject
- Scores: 6, 3, 6, 6, 6, 3

## Abstract
We present a complete mechanistic description of the algorithm learned by a minimal non-linear sparse data autoencoder in the limit of large input dimension. The model, originally presented in \cite{elhage2022superposition}, compresses sparse data vectors through a linear layer and decompresses using another linear layer followed by a ReLU activation. We notice that when the data is permutation symmetric (no input feature is privileged) large models reliably learn an algorithm that is sensitive to individual weights only through their large-scale statistics. For these models, the loss function becomes analytically tractable. Using this understanding, we give explicit upper bounds on the loss, which show that the model is near-optimal among recently proposed architectures. In particular, changes to the elementwise activation function or the addition of gating can at best improve its performance by a constant factor. Finally, we forward-engineer a model with the requisite symmetries and show that its loss precisely matches that of the trained models. Unlike the trained model weights, the minimal randomness in the artificial weights results in miraculous fractal structures resembling a Persian rug, to which the algorithm is oblivious. Our work contributes to neural network interpretability by introducing techniques for understanding the structure of autoencoders.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper examines a toy model of superposition, revealing that when the input data exhibits permutation symmetry, the auto-encoder learns an algorithm responsive only to macroscopic parameters. Using this insight, the authors derive upper bounds on the reconstruction loss and design hand-crafted symmetric models that match the loss of the trained model.

### Strengths
- This paper presents a novel study on the effects of input symmetry in superposition models.  
- This paper identifies a statistical permutation symmetry in the learned model weights, which is verified by designing a "Persian rug model" that achieves similar performance to the trained model.  
- The authors provide theoretical upper bounds for the reconstruction loss, which their experiments confirm by showing a notable drop in loss around $r\approx p$.  
- Detailed explanations throughout the paper improve the understanding and interpretation of the experimental results.

### Weaknesses
- **Writing**: The empirical observations in the paper feel overly repetitive, while the theoretical sections come across as overly technical and challenging to follow. For example, Section 3 spends three pages describing that the diagonal elements and bias become uniform while off-diagonal elements become noisy. Section 4 includes excessive technical detail in the derivations, which could be condensed into formal theorems, with the technical details moved to the appendix for easier readability.

- **Theory**: Many theoretical discussions in this paper lack rigor. For instance, the discussion between lines 353 and 361 is confusing and does not seem to follow rigorous mathematical reasoning. I will provide further points of confusion in the questions section. Notably, as a theoretically-focused paper, it lacks any formal theorems, making it difficult to digest. Summarizing results into well-defined theorems with clearly explained notations would improve clarity.

- **Experiments**: The experimental scope appears limited. The paper only presents results on a toy model with input data following a sparse, uniform, permutation-symmetric distribution. However, real-world data, such as natural language, does not exhibit permutation symmetry. Including experiments on real-world data would help demonstrate the applicability of these results.

In summary, the experimental scope of this paper seems narrow, and the theoretical sections lack the rigor expected of a theoretical work. I recommend that the authors refine their presentation of theoretical results and expand their experiments. I’m open to any clarifications if there has been a misunderstanding of the paper's theory from my end.

### Questions
- In Figure 1, you show that the structure of the artificial weights resembles a Persian rug. Out of curiosity, what do the trained weights look like?

- On line 155, you mention that in the $r \to \infty$ limit, the diagonal elements of $\mathbf{W}$ become the same, while the off-diagonal elements become zero-mean noise, resulting in $\mathbf{W} = a \mathbf{I}$. However, since $\mathbf{W}$ is the product of two low-rank matrices, shouldn’t it have a rank no greater than $n_d$? How, then, does it approach a full-rank matrix?

- On line 355, the symbol $\mu$ is not defined. Could you also clarify the main message of this paragraph?

- On line 406, the parity function is not defined.

- On line 410, should the LHS be $R_{ij}$? It would also be helpful to include a theorem stating that $\mathbf{R}$ satisfies the statistical symmetries, with a proof deferred to the appendix.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper studies the following auto-encoder model: linear encoding + linear decoding with additional denoising (shifted-ReLU) application on top. This model is applied in context of compression of i.i.d. sparse uniform data. The authors provide an analysis of the model behaviour (MSE bound) and identify the set of matrices which matches the performance of the trained model.

### Strengths
The paper provides to some extent a solid analysis of the proposed model.

### Weaknesses
The whole interpretability premise is evidently far fetched especially whenever LLMs appear in the related context. The paper does not provide any evidence that the phenomenology observed in their i.i.d. + small model setting appears on any even "toy"-ish real data, or justify enough whether the prescribed methodology could be put forward to gain an understanding of the behaviour of the real world models. The suggestion is to reframe and focus on the subject of substance - analysis of the autoencoder model.

I fail to see why so much of the paper space is occupied by a the "Gaussian-ity" experiments in Section 3.2. The authors do no provide any rigorous training results anyway (that the model will converge to a specific subset of matrices). In this view, since the proposed behaviour is assumed anyway using the empirical evidence, the amount of histograms and the accompanying description is clearly an overshot (which does not really add any further weight to the submission).

The technical machinery used for the analysis is not of a separate interest. 

The main concern, however, is the emergence of a certain structure in the weights. In particular, it is completely unclear whether the "Persian carpet" structure is indeed favoured by the network implicitly or just a set of weights which achieves the close to optimal MSE.
If later, the observations in the paper does not shed any light on any sort of "interpretability", since the network does not implement the proposed weight pattern when trained. Prior, however, is not supported by any evidence (no SGD/gradient flow convergence result).

The phenomena of uniform enough weights is not surprising on its own given that the signal is i.i.d., especially in the view of related work.
Specifically, could authors provide any intuition why specifically "Persian carpet" (except visual appeal)? There is no evidence whatsoever that a simpler uniform design will not achieve desired. Why not use some other design: for instance, rotationally invariant matricies with specific spectra (if the input distribution is centered, see next paragraph) and exploit Approximate Message Passing analysis (granted, the optimal denoising will not be close to ReLU)?

Another concern: why the data is not centered (which is *common practice*)? It is hard to judge, in this case, how much of an effect on the proposed structure it has, in the view of ReLU - which has cut-off at zero. 

I also assume that in equation (2) there is normalization in dimension missing. Otherwise, dropping "i" in (5) does not make sense.

Lastly, the particular choice of ReLU model (1) in the view of above remains quite unclear. It is hard to judge if any of the observations are not an artifact of the modeling.

The authors fail to acknowledge *a lot* of the related literature which is non-trivially connected to the observed model behaviour. Below, a non-exhaustive list:
-  AEs: (Refinetti & Goldt, 2022); (Cui & Zdeborova, 2023); "Analysis of feature learning in weight-tied autoencoders via the mean field lens" Nguyen (2021); (Shevchenko et al., 2023); (Kögler et al., 2024);
- Approximate Message Passing algorithms: (Donoho et al., 2009); rotationally invariant design (Rangan et al., 2019; Schniter et al., 2016; Ma & Ping, 2017; Takeuchi, 2019); optimal spectral design (Ma et al., 2021).

### Questions
See the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work tries to derive analytical ways to obtain the near-optimal solutions of a specific type of autoencoders for synthetic sparse datasets. After theoretical analysis, it provides the process for constructing the autoencoder’s weight matrix without numerical optimization.

### Strengths
1. Has theoretical analysis that looks legit to me, even though it is only focused on a very specific type of autoencoders on a specific type of datasets. 
2. The observation leading to theoretical analysis structure of this work can effectively convey the authors’ motivation.

### Weaknesses
The main weakness of this paper is its inconsistency and the absence of experiments for supporting their theoretical findings in Section 4, which greatly undermines the soundness of their results. Also, the authors should avoid introducing jargons and verbosity, be concise and accurate about what they mean, or provide necessary references if the jargons are widely accepted elsewhere. 
I suggest that the authors spend more time streamlining their work to improve its readability and precision, and bolster their theoretical claims with empirical results.

Regarding contribution, the results in this paper might be too restricted to a specific type of problem and model. I'm not sure how crucial this problem is to LLM development, and because I also value rigorous theoretical studies of problems no matter how small they seem, I refrain from saying harsh words like "it's trivial". That said, the authors had better show their problem's importance with better literature review and critical discussion. 

## Major

### Abstract

1. Some claims in the abstract are not (well) reflected in the main text. For instance, “changes to the elementwise activation function or the addition of gating can at best improve its performance by a constant factor”. I guess this is about the discussion on Line 486 to 492. Experiments need to be done to verify this.

### Section 1

1. Line 76: Please provide clear definition of “permutation symmetric data” and “thermodynamic limit”. At least they do not appear in the work (Elhage 2022) you cited, and they sound kinda vague and unintuitive. For instance, how can “permutation symmetric” be derived from “no input feature is privileged”? As long as the features are not identical you cannot say it’s “permutation symmetric or invariant”, right? If these are jargons commonly used in your out-of-ML research area, you need to translate it for your mortal audiences in the ML community. 

### Section 2

1. How many samples are there in your dataset on Line 111? What is the rank of this data matrix you sampled? Could the rank of this data matrix affect or even undermine your later observations and motivations? Did you scale up the number of samples as you increase $n_s$? If so, how? You need to provide more details about your experiments.

### Section 3

1. On line 142 there is *“due to an immediate drop in the loss starting around nd/ns ≈p. The slope and duration of this initial fall is controlled by p. In particular, in the high-sparsity regime (pclose to zero), the loss drops to zero entirely near the nd/ns ≈0 regime.”* 
I don’t think any part of Fig. 2b can clearly show this, especially considering that the upper bound of the x-axis (0.8) is way larger than p=0.05. You need to plot the loss curve at different p to demonstrate that, and zoom in on the “nd/ns ≈0 regime” and “nd/ns ≈p regime” to illustrate your points. Also, use the `subfigure` environments to enclose the subfigures.
2. On line 152, there is *“nd/ns →∞ limit”*, but Fig. 3,4,5,6 cannot show any of such limit. All nd/ns are under 0.8.  Do you mean “nd/ns →0 limit” (although this makes no sense), or just ns →∞? Because of this, I'm uncertain when reading your statements from Line 155 to 161. 
    * If it's the latter, as I understand it, 8192 might still be kinda small. 
    * Are you always assuming that nd < ns throughout the entire paper? Be clear.
3. Should provide a clear definition of “permutation invariant” mathematically. Also, what is “permutation symmetric” matrix? Be clear about what you say, or provide citations at least. 
4. Elaborate on “We hypothesize this occurs because for small models (and small feature probabilities) the contribution to any output from off-diagonal terms may fluctuate because of the small size of the matrix.” on Line 215. It’s like saying many things without saying anything. 
5. Elaborate on Line 236 to 239. Actually, this sentence is too convoluted to read. 
6. On Line 236 “Similar features are present in the two plots”. On Line 240 “we give three plots”. What and where are these five plots? Be specific. 
7. For Line 266 to 269 and Fig. 5. What is $\nu$? What is $\Delta \text{var}(\nu)$? What is “the fluctuation of the total norm of the off-diagonal elements in each row in W across rows”? Is it simply the variance of the off-diagonal elements’ variances? What do you mean by signal-to-noise ratio? What is considered as noise and what is not? The writing surely needs to be improved. If the concepts are introduced in the later sections (like in (4)), you’d better considering rearranging them to improve your narrative.
8. Elaborate on your choice of $\Lambda$ in (3). What is your motivation? What does it indicate? Any related literature where this metric is used and explained in detail? What is its relation to the Lyapunov condition? You say “strong numerical evidence”, how strong? Provide introductions or references for your readers to understand. 
9. Rephrase Line 304. Unclear. Better give a rigorous math expression for clarity. 

### Section 4

1. The important variable σ is not introduced very clearly. The only related explanation is *“σ characterizing the root mean square of the off diagonal rows”*. I can hardly get what you mean. Please use a better expression. 
2. You need to show your Persian rug’s optimality with experiments. 
3. Does Eq. 8 imply we can fix b at any value and only adjust a to obtain the optimal algorithm? If not, how to select $b$?
4. Verify "This implies that even if ... scaling of the loss for $r>p$." on Line 490 with experiments.

## Minor

1. Always cite related works when you introduce new concepts, like Hadamard matrix, Lyapunov condition, etc.
2. Unnecessary verbosity. For instance:
    1. (Line 185 to 188) “…diagonals are large…as possible.” Why don’t you just say $W \approx c I$?
    2. (Line 22 to 25) “Unlike…oblivious”. Why should the algorithm care?
    
    You should flesh out your paper with rigorous and detailed definitions of your important concepts rather than these.
    
3. What do you mean by “r=0.25” in Fig. 2a? Is r defined on Line 424? 
4. Line 178 “p = 4.5% and ratio ns = 512.” Is it a typo? Do you mean nd/ns = 512?
5. On line 204, please provide a clear definition of “mean-square fluctuation” or $\Delta$. Is it simply the variance (looks so on Line 212)? In addition, variance is not sufficient for evaluating if all elements in a set are nearly the same. You’d better additionally evaluate the maximum absolute difference. I also suggest replacing variance with standard deviation.
6. Why use $A$ rather than $a$ in Appendix A? Improve consistency.
8. What is $\mathbf{B}$ on Line 323 and 363? $\mathbf{b}$ in (1) and (2)? Improve consistency.
9. Use the correct citation command: \citep and \citet.
10. Use “Fig. xx” instead of “fig. xx” to refer to figures, and "Eq. xx" instead of "eq. xx" for equations.

### Questions
1. On line 111, why must we construct $x_i$ like this? Can we sample $u_i$ from the other distributions, like uniform[-1, 0]? This still makes $x$ sparse, right? I guess sampling $u_i$ from the other distributions may not work because you AE has ReLU at the output activation? Are you assuming that $x$ are outputs of ReLU or any other positive activation? 
    * How well does this synthetic distribution of sparse data agree with the distribution of sparse data in reality? Here you are assuming its dimensions to be i.i.d, in reality this is probably not the case.
2. You said “We must restrict to W with rank no more than nd” on Line 324. Do we need to take this rank condition into account when deriving Equation (4) and making the following reasonings from Line 332 to 348? If not, why? Could $\nu_i$ correlate with each other due to the low rank of $W$? Are you making any i.i.d. assumption about the off-diagonal elements here?
    * Also, this “because νi becomes Gaussian” seems to be a conjecture based on your experiment in Section 3.2. I’m not familiar with your test so I don’t know how reliable it is. That said, you’d better make this assumption stand out with something like `\newtheorem{asp}{Assumption}`. 
3. Given your Persian rug construction of $W$, any way to retrieve $W_\text{in}$ and $W_\text{out}$ from it? Are they unique? Can we use the pair on Line 398? 
4. Expand your derivation of Eq. 19 in Appendix B to make it clearer. Why is there no $a$ in it (considering your definition of $\nu$ on Line 333)? Does this equation only hold when you construct $x_i$ as described on Line 111?

### Soundness
3

### Presentation
2

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
The authors study the compression of sparse data using a two-layer autoencoder with ReLU output non-linearity, considering the toy model of Elhage et al (2022). Building on an ansatz for the product of the weights, supported by numerical observation, they derive a closed-form expression for the population loss depending on only three scalar parameters, corresponding to the bias, common diagonal element of the product matrix, and variance of off-diagonal elements. They deduce from it a lower-bound on the loss, and construct a deterministic weight matrix that achieves it. They find that the corresponding loss coincides with that of the trained model.

### Strengths
On a technical level, the authors provide a detailed study of the optimal population loss achieved by the considered model, which is non trivial given the ReLU non-linearity. To the best of my awareness, this derivation is new compared to the original work of Elhage et al. (2022). The finding of the weight matrix achieving the same loss as the model optimized using classical gradient-based optimizers is an interesting construction. Although I have not checked the math in detail, the derivation looks scientifically sound.

On a writing level, the paper is very clearly written, with sufficient discussion and intuition building. The assumptions are illustrated by careful numerical experiments.  Sufficient discussion accompanies all important equations.

### Weaknesses
My main concern lies in the scope of the contribution. Some of my concerns are tied to my questions, which I list in the next section.

- To my understanding, the study is confined to the isotropic case of Elhage et al. (2022), as it builds upon the ansatz that all diagonal elements of $W_{out}W_{in}$ are equal. This means there is not really any feature learning in the model, although the considered setting remains non-trivial.

- While the study is interesting, it does not add much more phenomenological insights compared to the original work. In particular, the authors construct a matrix which achieves the same optimal population loss as the trained model, which is one of the strengths of the work. However, I understand any matrix of the form (7) is similarly optimal, and I do not understand why the explicit realization in 4.2.2 brings any further insight. I might have misunderstood a point, and am happy if the authors bring clarifications on this point.

Overall, I think the study is technically sound, the insights tied to the analysis of interest, but brings little novel phenomenological insight.  I am thus giving an accept score -- because the paper looks sound --, but not a high one.

### Questions
- (Minor) Do the authors believe the study could be extended to non-isotropic cases, where some features are of higher importance, e.g. in the block case where blocks of different features have the same importance ?

- Is it correct that (7) also corresponds to the weights a linear model would learn ? If that is the case, does the improvement in reconstruction loss from the reLU model come from the bias and activation alone ? 

- As pointed above, is there any specific reason the particular realization of 4.2.2 of optimal weights is of specific interest, compared to randomly drawing an orthogonal matrix $O$ and constructing a weight matrix as in (7)? Do the authors claim the model learns weights exactly corresponding to 4.2.2, or simply that the loss coincide ?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper both theoretically and empirically performed an exhaustive analysis of autoencoder trained on sparse data,
extracting the essential structure needed to represent such data. The paper firstly reveals that the change of loss as the compression ratio varies and shows that when compression ratio is near zero, the loss drops to zero entirely. This motivates the authors to consider the structure of the weight in both linear and non-linear cases. They artificially include the statistical permutation symmetry into the weight and found the artificial weight and the trained model returned the same result. The conclusion is they qualitatively show that a statistically symmetric strategy exists in certain regimes of the macroscopic parameters. Thus optimal choice for W exisits based on the proposed  Persian Rug Model.

### Strengths
S1. Novelty: The paper is novel in it both theoretically and empirically performed exhaustive analysis of autoencoder trained on sparse data, interpreting the essential structure needed to represent such data. I like the way they demonstrates the distribution of the weights. 

S2. The paper firstly reveals that a statistically symmetric strategy exists in certain regimes of the macroscopic parameters and thus optimal choice for W exisits based on the proposed  Persian Rug Model. The theoretical support of the motivation is strong. 

S3. The presentation of the paper is good, with a good clarity.

### Weaknesses
W1. What is the interpretation of the model if the input is not sparse? Is the symmetry property and the relevant conclusion partially hold? I noticed the theories are mostly based on the compression ratio. Does this mean the interpretation will change if the compression ration becomes high?  

W2. What is the significance of the model if the input is no longer sparse?

### Questions
My questions are mostly overlapped with the weakness section. I appreciate it if the authors may further instantiate the following points:

W1. What is the interpretation of the model if the input is not sparse? Is the symmetry property and the relevant conclusion partially hold? Does this mean the interpretation of the weights and the optimal W will change significantly if the compression ration becomes high?  

W2. Generally speaking, what is the significance of the proposed model if the input is no longer sparse, whereas in most cases the input is not sparse?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper characterizes the solutions of training sparse autoencoders (defined as $f(\mathbf{x}) = \textrm{ReLU}(W_{out} W_{in} \mathbf{x} + \mathbf{b})$) with the reconstruction loss $L=\mathbb{E} || \mathbf{x} - f(\mathbf{x}) ||^2$. The paper first trains an autoencoder on synthetic data consisting of sparse vectors generated from a Bernoulli-Uniform model and records some empirical observations about the composite matrix $W=W_{out} W_{in}$. The authors note that the off diagonal entries of this matrix seem to follow a normal distribution that seems to be identical across rows and that the diagonal entries are also identical. This allows them to reduce the loss function to a function of three scalar variables corresponding to the diagonal, off-diagonal, and bias terms. The solution they obtain is of the form $W = OPO^\top$ where $O$ is orthogonal and $P$ is a rank $n_d$ projection matrix. The authors also characterize how the loss varies as a function of the Bernoulli sparsity parameter $p$.

### Strengths
The empirical observations and theoretical results are reasonable and the authors provide an adequate description of the solution $W$.

### Weaknesses
Connections between sparse coding and autoencoders have been explored extensively in the literature. This paper does not cite or engage with any of this literature. 

The empirical observations about $W$ having identical diagonal entries and the off diagonal entries being small and having a common distribution can be captured by $W = A^\top A$ where $A \in \mathbb{R}^{n_d \times n_s}$ is an overcomplete, incoherent dictionary with unit norm columns. This ensures that the diagonal terms are identical while the off-diagonal terms are small in magnitude. Moreover it is well known that random matrices satisfy the incoherence/restricted isometry condition and a random $A$ ensures that the off-diagonal entries follow the same distribution.

This model of dictionary learning, along with Bernoulli-Uniform or Bernoulli-Gaussian sparse codes has been studied in the complete and overcomplete setting with neural and non-neural algorithms [3,4,5].

Reference [1] studies how ReLU autoencoders with tied weights can learn dictionaries with data from this generative model. The authors show (in theorem 3.1) that one layer of a ReLU autoencoder can recover the support of the sparse code with high probability if the weights are close to the ground truth dictionary. The result also proposes a bias vector with all negative entries whose magnitude can be derived from the incoherence of the dictionary. The paper also uses a landscape argument to show that the ground truth dictionary is a critical point for the reconstruction loss. Reference [2] goes further to show that autoencoders trained to minimize reconstruction error with gradient descent can recover the ground truth dictionary.

[1] Rangamani, A., Mukherjee, A., Basu, A., Arora, A., Ganapathi, T., Chin, S., & Tran, T. D. (2018, June). Sparse coding and autoencoders. In 2018 IEEE International Symposium on Information Theory (ISIT) (pp. 36-40). IEEE.

[2] Nguyen, T. V., Wong, R. K., & Hegde, C. (2019, April). On the dynamics of gradient descent for autoencoders. In The 22nd International Conference on Artificial Intelligence and Statistics (pp. 2858-2867). PMLR.

[3] Arora, S., Ge, R., Ma, T., & Moitra, A. (2015, June). Simple, efficient, and neural algorithms for sparse coding. In Conference on learning theory (pp. 113-149). PMLR.

[4] Agarwal, A., Anandkumar, A., Jain, P., & Netrapalli, P. (2016). Learning sparsely used overcomplete dictionaries via alternating minimization. SIAM Journal on Optimization, 26(4), 2775-2799.

[5] Spielman, D. A., Wang, H., & Wright, J. (2012, June). Exact recovery of sparsely-used dictionaries. In Conference on Learning Theory (pp. 37-1). JMLR Workshop and Conference Proceedings.

[6] Refinetti, M., & Goldt, S. (2022, June). The dynamics of representation learning in shallow, non-linear autoencoders. In International Conference on Machine Learning (pp. 18499-18519). PMLR.

While these prior results study a slightly different model architecture, they can be adapted to the model studied in this paper with a little effort. In my opinion the results obtained by the current paper can be explained by the sparse coding model, and it is not true that this has not been studied in the context of autoencoders.

### Questions
1. How does the solution proposed by the authors differ from the dictionary learning solution? Can the authors delineate conditions under which trained autoencoders converge to their solution instead of overcomplete, incoherent dictionaries?

2. The previous literature does not study autoencoders and sparse coding in the context of interpretability. Are there specific questions that arise in this context that require the authors' model? Does the authors' analysis answer these questions?

### Soundness
2

### Presentation
2

### Contribution
2
