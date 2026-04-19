# Correlational Lagrangian Schrodinger Bridge: Learning Dynamics with Population-Level Regularization

- Decision: Reject
- Scores: 3, 8, 5, 6

## Abstract
Modeling population dynamics is a fundamental problem with broad scientific applications.
Motivated by real-world applications including biosystems with diverse populations, we consider a class of population dynamics modeling with two technical challenges: (i) dynamics to learn for individual particles are *heterogeneous* and (ii) available data to learn from are *not time-series* (i.e, each individual's state trajectory over time) but *cross-sectional* (i.e, the whole population's aggregated states without individuals matched over time).
To address the challenges, we introduce a novel computational framework dubbed **correlational Lagrangian Schr\"odinger bridge** (**CLSB**) that builds on optimal transport  to "bridge" cross-sectional data distributions. In contrast to prior methods regularizing all individuals' transport "costs" and then applying them to the population  *homogeneously*, CLSB directly regularizes *population* cost allowing for population *heterogeneity* and potentially improving model *generalizability*.
Specifically our contributions include 
**(1)** a novel population perspective of the transport cost and a new class of population regularizers capturing the temporal variations in multivariate relations, with the tractable formulation derived,
**(2)** three domain-informed instantiations of population regularizers on covariance, and **(3)** integration of population regularizers into data-driven generative models as constrained optimization and an approximate numerical solution, with further extension to conditional generative models.
Empirically, we demonstrate the superiority of CLSB in single-cell sequencing data analyses (including cell differentiation and drug-conditioned cell responses) and opinion depolarization.
Codes will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This paper introduces a population-level regularization adapted for learning heterogeneous biosystems dynamics with Schrödinger Bridge. As data for these systems is collected at each time step but without individual trajectories across time steps (cross sectional data), the Schrödinger bridge is typically extended with a least-action principle to make the optimization procedure tractable.

Lagrangian Schrödinger Bridge is less effective for heterogeneous systems as it assumes homogeneity.
The paper proposes a “principle of least population action” that shifts the conservation laws enforced by LSB from individual to population states. This is extended with a generalized conservation principle that incorporates relationships among data variables and uses the Fokker-Planck equation to derive a tractable estimator. Special cases of this general regularizer with biologically informed constraints are introduced.

The final regularizer proposed is a weighted sum of the original LSB regularizer and some of the new population regularizers.
Numerical experiments are then conducted to show the effectiveness of the proposed regularizer on developmental dynamics of cells.

### Strengths
The paper is well written and organized. The authors provide a clear motivation for the proposed method with a relevant example from biology. Limitations of the existing methods are clearly presented

### Weaknesses
For $L_{ind}$ the authors mention (l.192-194) that it is not suitable when time series data is not available and justify next introduced $L_{\cdot}$ as solutions that do not require individual trajectories as they get averaged. However, in section 3.4 $L_{ind}$ is used in the final formulation (13). Does this imply that individual trajectory data is actually available? Are the introduced $L_{\cdot}$ effective by themselves when trajectory data is not available?
 This point seems unclear.

I also fail to understand how $L_{pop}$ is an heterogeneous type regularizer. I get that it deals with a population state and not an individual action on single trajectory. However as $\mathbb{E}_{\pi_t}$ is involved in the definition isn't it still applied uniformly across the population?

The paper could benefit from a detailed explanation of how the final proposed formulation (13) helps with the "non convex and constrains wrt network parameters challenge" mentioned.

The paper introduces multiple correlation-based Lagrangians, though only a few are used in the experiments.

### Questions
- If $L_{ind}$ is unsuitable without time-series data, why is it included in the final formulation (13)?

- How does $L_{pop}$ act as a heterogeneous regularizer if it’s based on $\mathbb{E}_{\pi_t}$? Doesn’t this imply it still applies uniformly across the population?


- Could the authors provide a more detailed explanation of how the final formulation (Equation 13) addresses the challenge of non-convexity and constraints on network parameters?

- Could the authors clarify why certain Lagrangians were selected over others?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Interesting method applying a population-level regularization to schrodinger bridge parameterized by neural nets for the drift and diffusion terms. They motivate their problem with the relevant biological setting of predicting the influence of drugs on cell state, where cell state can only be measured once in a destructive process. The paper dives into details pertinent to this biological problem such as how to apply the method when in a latent space imposed e.g. by PCA. My main critiques are in the presentation of the method which makes checking the math difficult at times. This concern about presentation and validity of the method is mainly due to the lack of convincing empirical evidence. The proposed method seems to only occasionally outperform in the unconditional evaluation. I would feel more confident in this paper with better presentation and clarity.

### Strengths
- The paper introduces a new way to regularize Lagrangian schrodinger bridges. 
- This regularization can potentially improve population-level dynamics of single-cell analyses. 
- There are different ways to use the correlation matrices, which provides flexibility and allows users to impose their own priors based on biological knowledge. I'd really like more discussion how to use this, if possible. Would make the paper much better.
- The authors provide extensive proofs as to how the regularization is implemented numerically.
- Extensive ablation studies and comparisons, which is great.

### Weaknesses
- There are some grammatical errors, such as "overwrote" in line 175, "we have" in line 319. "We achieves..." in line 299.
- A lot of variables are introduced in passing. I like the table that reminds what the difference between sub/superscript is and think this could be extended to definitions of variables $s, d, m, k, \mathcal{M}, \tilde{\mathcal{M}}, $
- No code to validate. 
- The presentation of the results is a little confusing. Figure 2 uses PCs for the gene expression trajectories but it would make more sense to present the way Tong et al. 2023 did in this [paper](https://arxiv.org/pdf/2307.03672) in Figure 3. This would make the value of the method more immediate to computational biologists. The current method showing PCs of individual genes is unintelligible. This change would greatly help clarity for a more diverse audience. Additionally, I'd recommend using this compressed representation in the appendix. Just showing pairwise plots of seemingly random genes and expressed vs. predicted is not very helpful, unless I'm missing something related to the covariance that this method proposes. If there's something there that makes those plots more interpretable, you should highlight that in the main paper.
- In proofs for the k $\geq$ 2 case, I would prefer if you introduce the operators before just laying everything out in 14. It's very taxing to have to crossvalidate the equation with its parts and doesn't make me confident in the validity of the proof. Additionally, a sketch of proof saying how it works in summary would greatly help. Still, I have concerns related to the previous questions that don't allow me to fully check this proof, e.g. what is the purpose of using the product operator in this case. 
- it's hard for me to validate the proofs missing interpretation about what the product is that defines the correlation structure in the first place.

### Questions
- Do you use the product over the probability densities in equation 6 because of the assumption of independence? This wasn't explained well and this lack of definition makes the whole paper hard to follow since it is used extensively.
- the velocity and acceleration of the covariance are interesting ideas to constrain. It's not clear to me how you actually do this from the paper. More clear exposition would help here. As well, why not investigate the hyperparameter sweep of these variables?
- You prove the light tailedness in the appendix but only mention its importance in the main paper. It would be helpful to have some more context about this and why it's cancelled from the proof in the paper.
- What's the computational complexity of this method? How are you sampling? You don't need a section on these but please mention a high-level complexity and state how you sample trajectories.
- In proof for $k \geq 2$ in the appendix, what is $\mathcal{S}'$ and $\Upsilon'$? in equation 14.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper introduces an approach to the modelling of dynamics of populations from which only cross-sectional observations are available. It is suggested that in the absence of association between observations and members of the population that it is more sensible to conduct modelling at the population level.

Within the Schroedinger bridge framework, i.e. with initial an final distributions known and fixed, the problem of determining the trajectories from observations is viewed as an optimal transport problem. A major element of the current work's contribution is to consider regularisation of this process at the level of the population rather than of the trajectories of individuals.

Numerical results are presented for a cell-sequencing setting which demonstrate the good performance of the proposed approach in that setting.

### Strengths
The paper clearly explains what is does and its consequences in terms of the resulting optimization problem.

Sections 3.2-3.3 provided details which will be valuable to anyone interested in applying these techniques in various contexts; and go beyond the usual abstract presentation of methodology giving nice details of how to implement the proposed approach to achieve particular outcomes.

Some consideration is given to implementation of the method: in addition to formulating the problem at the population level some space is dedicated to showing how one can actually perform learning using this revised objective.

The numerical results presented (e.g. in Tables 2 and 3) seemingly demonstrate good performance of the proposed method in the explored example setting.

### Weaknesses
One thing that wasn't completely clear to me (this isn't an area I am particularly familiar with) was what the main reasons for the audience of this conference to be interested in this work are. It seems like a nice contribution to learning for particular settings, in which a population-level view is reasonable but beyond the rather specific numerical examples in a cellular context it wasn't clear to a non-expert what those contexts might be. Are the authors able to explain this a little more explicitly?

The underlying motivations wasn't articulated as clearly as it might have been in my view -- beyond leading to a more tractable optimisation problem, when/why would one choose to view the problem of inferring dynamics in this population-level way? (The fact the individuals may behave heterogeneously and one may not be able easily to relate observations to individuals does not in and of itself seem to mean that a population-level regularisation is the natural way to proceed.)

The numerical evaluation appears to be limited to a single setting which is quite difficult for a non-expert to interpret, although there are numerical quantifications provide which appear to demonstrate good performance in that setting. It might be more persuasive if there were numerical results illustrating the proposed approach in several settings illustrating good performance across a wider range of scenarios particularly if the paper's contribution is to be viewed as general methodology rather than an application to the setting of the numerical example. There are some additional experiments in  Appendix I, but these are primarily very closely related to the original application with on exception on "opinion depolarisation" but the presentation here is so condensed that it is not completely obvious what is shown.



Minor details:
* There are places where more proofreading would help, e.g. in the references line 544 has `Schr\"odinger'

### Questions
Are there general settings in which the authors envisage the proposed approach being effective? Are there contexts of broad interest to the ICLR community which involve cross-sectional observations in which population-level regularization would be reasonable?

It seems that moving from individual level to population level regularization is a philosophical change in position rather than a choice to be made for computational reasons. To the non-expert, it feels like a reasonable shift where it is reasonable to suppose that there is some (implicit or explicit) population-level control of the dynamics -- perhaps in the context of hives, or coordinated systems -- but which is unlikely to model well the behaviour of a population of autonomous individuals. I was surprised not to see anything along these lines discussed in the manuscript. Are the authors able to comment on this?

Figure 1: How should the reader interpret this and the many similar figures in the paper? This isn't an area with which I am familiar and I found it difficult to understand what I was supposed to deduce from this figure.

Line 236: A very minor point, but can you clarify what this multiset is? I couldn't immediately understand what necessitated the use of a multiset rather than a regular set here.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper looks at the question of modeling population dynamics. The setting is one where the data is sampled cross-sectionally, and the goal is to bridge these cross-sectional snapshots. For one, these snapshots may even lack observations on the same population due to the way the data is acquired, for example, in immunomics-type of applications. Traditional approaches to such problems attempt to utilize methods that bridge the distributions together across time with, for example, techniques such as optimal transport. This paper proposes a way to do this bridging in a more principled way, through utilizing regularization at the population level. The key idea here appears to be, effectively, restraining population covariance kinetics-related quantities. The proposed method is compared to existing ones, with promising results.

### Strengths
The paper is nicely written and well-illustrated. The problem presented by the authors is well-motivated with real grounding in applications. The results seem promising and worth investigating further.

### Weaknesses
The authors should perhaps give some idea of the computational costs of their methods. I am also not sure why in Figure 2 the cyan dots (in the lower panel) have much smaller variance than the red ones.

### Questions
What happens if the covariance structure undergoes a change over time? Can the method adapt to this scenario? Have you considered things like normalizing flows as a comparison?

### Soundness
3

### Presentation
3

### Contribution
3
