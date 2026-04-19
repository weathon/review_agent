# Neural Manifold Operators for Learning the Evolution of Physical Dynamics

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 3, 5, 3, 1

## Abstract
Modeling the evolution of physical dynamics is a foundational problem in science and engineering, and it is regarded as the modeling of an operator mapping between infinite-dimensional functional spaces. Operator learning methods, learning the underlying infinite-dimensional operator in a high-dimensional latent space, have shown significant potential in modeling physical dynamics. However, there remains insufficient research on how to approximate an infinite-dimensional operator using a finite-dimensional parameter space. Inappropriate dimensionality representation of the underlying operator leads to convergence difficulties, decreasing generalization capability, and violating the physical consistency. To address the problem, we present Neural Manifold Operator (NMO) to learn the intrinsic dimension representation of underlying operators by calculating the minimum dimensional submanifold representation in the latent space. NMO achieves state-of-the-art performance in statistical and physical metrics and gains 23.35% average improvement on three real-world scenarios and four equation-governed scenarios across a wide range of multi-disciplinary fields. Our paradigm has been demonstrated universal effectiveness across various model structure implementations, including Multi-Layer Perceptron, Convolutional Neural Network, and Transformer. Experimentally, we prove that the intrinsic dimension calculated by our paradigm is the optimal dimensional representation of the underlying operators. Our code is available at https://anonymous.4open.science/r/Neural_Manifold_Operator.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces an operator learning framework that includes a projection onto a lower-dimensional latent manifold. The authors show state-of-the-art results on many real world datasets and simulations across multiple models from vision, forecasting, PDE nets and neural operator litrature.

### Strengths
This seems like a very strong paper with excellent results on a wide variety of datasets. The overarching idea is well motivated and explained, and a lot of the details are given in the supplementary I particularly like that the authors included a large mixture of real world datasets and equation based simulations and showed improvement over models from various fields.

### Weaknesses
I think this is an excellent paper and only has some minor points when it comes to presentation.

1) Section 3.3.2 seems to be central to key innovation of the paper, which introduces a lower-dimensional manifold into an operator learning framework. Nonetheless, it is not presented or explained well and to get the central point the reader has to go seek out Costa et al. 2005. I would suggested adding an illustration and some more explanations to these ideas which are central to the paper's claims.

2) All of the figure captions are very lacking:
   Fig 1. is not referred to in the text and has very small text that is hard to read or interpret. Furthermore, it would be good if the caption could be elaborated to include a walk-through of the figure. For example there is no reference to (a), (b), (c) etc..
  Fig 2., Fig 3. There is almost no caption. Again, it would be useful to have an informative caption across the paper.
 Fig 4. Very packed and not explained well. What is the prediction step showing? How far into the future is this predicting? What are the initial conditions, etc.? Also, some plots have many colors, and one has to fish out the "target" and compare it to the different models by eye.

3) I think this paper could use a good read-through for grammatical errors. examples: 
Section 3.3.2 "There exists that the latent variables ...", "intrinsic dimension in local as the following..." and more.

### Questions
There are some critical points in the text that were hard to follow. In particular Sec 3.3.2 lacks clarity. Some of the questions are:

1.  Is the intrinsic dimension differentiable? Is it computed in pre-training and then frozen? 
2. How do you choose "k" (Eqs 9-10) and what is the effect of "k" on the results? 
3. Eq (5) --> can you clearly mark what is trainable? Q, P are frozen, does theta parameterizes both L and K?
4. Section 3.4. a notation suggestion --> K_id and R^id should probably be called K_m and R^m as this is an estimation of the intrinsic dimension but no guarantee to always converge to that. 
5. 4.4. Ablation study: It is impossible to assess these results without error bars. I can't tell if this improvement will be washed away by statistical variations. This ablation study is not described in nearly enough detail, despite the fact that it is central to one of the paper's main claims (from page 2): "...the redundant dimension representation of the underlying operators leads to several problems...". Furthermore, I expected to see a study on how well the determined "m" (using algorithm 3.3.2) converges to the true m=id and under what conditions.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a method of operator learning for predicting physical phenomena in which the latent space is extracted from the data and the dynamics on the latent space is estimated. In particular, the proposed method is combined with a method for estimating the dimension of the latent space.

### Strengths
Numerical experiments have been conducted on many specific examples.

### Weaknesses
This paper is very poorly written. Symbols are not used in a uniform manner and there are also undefined symbols. They are too numerous to enumerate, but for example, in the first paragraph of section 3.2, the operator $\_mathcal{K}: \{ v_t: D_t \to \mathbb{R}^{d_L} \to v_{t+\varepsilon}: D_{t+\varepsilon} \to \mathbb{R}^{d_ L} \}$ is introduced; however the symbols $D_t$ and $D_{t+\varepsilon}$ are not explained. It can be understood that these are the domains of $v_t$ and $v_{t+\varepsilon}$, but if so, this implies that the domains are assumed to change with time. However, it is not explained how the proposed method handles the evolution of the domain. In addition, there are many grammatical errors. This is just my impression but if this paper is submitted to a major journal, this paper will be rejected by the editor without review. 

The novelty of the method is also questionable. The technical contribution seems to be the estimation of the dimensions of the latent space, but this is achieved by simply applying an existing method.

### Questions
Numerical experiments indicate that the proposed technique would improve the performance, so I encourage the authors to carefully rewrite the paper and resubmit it.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies the class of methods for learning physical dynamics based on neural operators, that is where the objective is to learn an operator (in general nonlinear) acting on an infinite-dimensional function space. To achieve this task, usual approach is to learn representation manifold with encoder-decoder architecture and an approximation operator defined on it. 

Authors identify that the choice of the dimension of the representation space has profound impact on the overall performance and propose a two step procedure to properly choose it. 

First, by pertaining an autoencoder (AE) with a chosen latent dimension, a representation space of the trajectory is identified and its intrinsic dimension is computed as suggested in (Levina & Bickel, 2004; Costa et al., 2005). Then, freezing the parameters of pretrained autoencoder, during the training process linear reduction to intrinsic dimension, time evolution operator and linear extension to the latent dimension are learned. 

The overall architecture named Neural Manifold Operator, learns nonlinear function that evolves the states of the physical system in time and its performance is tested on various datasets and against diverse operator learning baselines.

### Strengths
The paper is, apart of few issues reported bellow, well written. Main contribution is clearly stated and honestly reported. Relevant literature is well documented. The class of learning algorithms that is studied as well as the problem that the paper solves is are relevant. Experimental section elaborated and results are good.

### Weaknesses
My overall  impression of the paper is good. However, the following issues currently limit the overall score. If those are resolved, I am ready to re-evaluate my assessment. 

A) Mathematical inconsistencies and overly informal presentation of the equations:
1) In Eq. (3) when defining $G$, notation is not clear. $X_t$ is a state depending on time step $t$ while $\Theta$ is a set of parameters. I imagine it should be written $G_{\varepsilon}\colon D\times \Theta \to D$, where $\varepsilon$  is the hyperparameter of the NMO model defining $\varepsilon$-ahead nonlinear evolution of $X_t$. One should comment the the problem is time-homogeneous dynamical system, hence $G_\varepsilon$ does not depend on time.  

2) To train the model it is assumed to need points and their $\varepsilon$-ahead evolution, is this correct? If yes, then in the discussion that follows Eq. (3),  $X_{t:t+\varepsilon}$ should be corrected to $X_{\varepsilon:t+\varepsilon}$. 

3) In Eq. (6) notion of expectation is not clear, since there is no notion of randomness and considered time horizon is not clear. What is considered as "true-operator" here? In the way how it is defined $G$ cannot play this role? I guess that a proper way to write the risk is to use random initial point, integral over time and compare $G_\theta$ to $\mathbf{F}$ defined in Eq. (2).

3) In discussion bellow Eq. (11) the notation $\mathcal{T}_{[\cdot]}$ is explained when defining operators ${\mathcal K}\_d$, $\mathcal{K}\_e$ and $\mathcal{K}\_u$. 

B) As reported in Section 3, the NMO architecture is designed to learn nonlinear evolution of deterministic dynamical system. Since according to Koopman operator theory there exists a representation space where the evolution operator, here denoted as $\mathcal{K}_{id}$ is linear. Which that in mind, it would be worth to explain rationale for ETO architecture to learn nonlinear operator. Clearly, nonlinear operator is needed in more general inverse problems, but since this proposal is tailored for learning the flow maps from trajectories, I think that this aspect should be at least commented. If possible, it would be also useful including some of the Koopman based baselines in the experiments. 
 
C) Minor issues:
1) Since matrix $\mathcal{L}$ is not squared, does $\mathcal{L}^{-1}$ in Eq (5) denotes pseudoinverse?
2) $E$ in Eq. (10) should be expectation.
3) Line bellow Eq. (11), decomposition should read composition.

### Questions
-

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper aims at modeling a solution of the physical dynamics systems on infinite dimensional spaces. To do that, the paper proposes a pipeline that consists of (1) a pre-trained encoder projecting finite-dimensional observations of infinite-dimensional inputs to (finite-dimensional) latent representations, aka intrinsic dimension in the paper, (2) a main network (TEO in the paper), and (3) a pre-trained decoder mapping the latent representation in intrinsic dimension to the data space. The paper claims that the proposed method performs better than previous operator learning methods, including the DeepONet or Neural Operator family.

### Strengths
N/A

### Weaknesses
In general, I find that the paper’s contribution is not clear.

First of all, it is not clear why the proposed method is a valid operator mapping between infinite dimensional spaces. In my understanding, the key part of the proposed method is “intrinsic dimension calculation”, but this part is somewhat hand-wavy. Moreover, based on the description in the appendix of the algorithm, the intrinsic dimension calculation doesn’t seem like a novel method compared to KNN-based dimensionality reduction algorithms. However, I find that the paper hasn’t provided sufficient information on how the algorithm and the follow-up pipeline can properly map between infinite dimensional spaces.

Second, the presentation of the proposed method is not clear. For instance, the description of each part in the pipeline is hand-wavy, and the paper relies on verbal descriptions instead of what each part does. I find that it is important to describe how each part works in a more concrete manner so that potential audiences can understand how the proposed method can work as an operator.

### Questions
N/A

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 5

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors proposed an encoder/decoder based manifold learning procedure to learn the evolution of a physical system. An extensive experiments have been presented on several datasets with comparative analysis.

### Strengths
1. Well written and easy to follow in most parts.
2. The experimental results are illustrative.

### Weaknesses
1. My main concern of this paper is the theoretical contribution, in my understanding this paper is essentially learning mapping from data manifold to latent space and used diffusion process on the Euclidean latent space by using transformer, convolution etc.. So essentially the paper has two components (methodologically): (a) deriving the mapping from manifold to Euclidean space. (b) the evolution process. (a) can be learned in a auto-encoder setup, in fact my biggest concern is why authors did not learn the lower-dimensional latent representation instead first learning the higher dimensional manifold and then learning the submanifold?The evolution process is a time-varying recurrent process, hence  very well-explored in past literature.
2. The experimentation in terms of breadth is well appreciated, although I see none of the manifold learning methods used as a baseline. So the comparative analysis is poor.

### Questions
1. Why the authors separate encoder and the sub manifold mapping, L. If the authors consider encoder to be a mapping from manifold to lower dimensional sub manifold (i.e., consuming L) is the learning becomes harder? 
2. Normally we use latent space to be something lower dimensional, whereas here authors use latent dimension to denote the higher dimensional space where manifold is embedded. This is quite counter intuitive as in practice  researchers use Encoder/Decoder to map from data manifold to latent space (with dimension as intrinsic dimension of the data), as contrary to the higher dimensional space as the authors have claimed to define.
3. Not sure “Therefore, the goal of the manifold algorithm is to calculate minimal m”? For a larger m, we will still assume it is a linear space, it doesn’t have to be minimal to be a linear space. Any higher dimensional space is desired as well. My understanding is the authors try to find a Euclidean latent space where they can use Euclidean stochastic process, so not sure the necessity of finding optimal m.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
