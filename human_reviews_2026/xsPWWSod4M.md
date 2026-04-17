# Skill Learning via Policy Diversity Yields Identifiable Representations for Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 8

## Abstract
Self-supervised feature learning and pretraining methods in reinforcement learning (RL) often rely on information-theoretic principles, termed mutual information skill learning (MISL). These methods aim to learn a representation of the environment while also incentivizing exploration thereof. However, the role of the representation and mutual information parametrization in MISL is not yet well understood theoretically. Our work investigates MISL through the lens of identifiable representation learning by focusing on the Contrastive Successor Features (CSF) method. We prove that CSF can provably recover the environment's ground-truth features up to a linear transformation due to the inner product parametrization of the features and skill diversity in a discriminative sense. This first identifiability guarantee for representation learning in RL also helps explain the implications of different mutual information objectives and the downsides of entropy regularizers. We empirically validate our claims in MuJoCo and DeepMind Control, and show that CSF provably recovers the ground-truth features from both states and pixels. Our code is available at https://github.com/bmucsanyi/identifiable-misl.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper offers an explanation for the effectiveness of skill-learning methods based on mutual information (MISL). The authors provide both theoretical and experimental analyses of how the representations learned by a representative approach (CRL) of MISL relate to the true latent factors of the environment. Their results show that, under certain assumptions, CRL implicitly learns representations that are linearly related to the environment’s underlying latent state.

I believe the paper touches an important topic and provides an interesting and novel analysis on the representations leaned by skill learning methods. However, the writing could be improved and the analysis could be strengthened to support the claims.

Writing:
- The introduction could be made easier to follow. Some parts would fit better in the related work section, and long citation lists should be avoided. The connection to causality is not entirely clear, and lines 81–89 do not seem directly related to the paper’s main motivation.
- The explicit definition of identifiability should appear earlier in the manuscript to help readers grasp the theoretical framework. The discussion in the introduction is not sufficiently explicit.
- I recommend separating the related work from the background section to improve readability and structure.
- Line 156: This paragraph lacks clarity and could benefit from rephrasing or additional context.
- Definition 1 (line 171): The proposed definition seems closer to “distinguishability” than “diversity.” Two skills may produce very similar trajectories yet still represent different behaviors. The relationship between this definition and the notion of diversity used in Assumption 1 should be clarified.
- The statement “Each pair of consecutive states corresponds to one skill” introduces a strong and limiting assumption. This is unlikely to hold in many environments. For instance, an agent might need to traverse a corridor before exploring an area, or operate in an MDP with discrete actions and deterministic transitions. A more thorough discussion of such edge cases would strengthen the paper. Furthermore, the connection to the statement line 293 is unclear.
- In Figure 2 (right), the direction of the arrows is ambiguous and should be clarified.
- First paragraph of Section 3.2. I do not catch why z and feature representations could be antipodal by maximizing their dot product.
- In Section 3, the example at line 241 is particularly helpful and could be emphasized earlier in the section. However, the statement "to distinguish data under different distribution shifts or interventions" is not "intuitive".

Experiments:
- In Figure 3, using features or feature differences do not induce meaningful difference in identifiability. This should be discussed. Overall, it lacks a baseline that gives indications of what is a high identifiability.
- The authors should formally define state coverage in the manuscript. 
- The authors rightly note that the R2 score is influenced by state coverage, but the CDR² curve remains difficult to interpret and, in practice, seems to primarily reflect coverage rather than the main contribution of this work.To clarify this point, the authors should compute the R2 score over a set of presampled agent positions, either manually defined or extracted from trajectories of trained DRL agents. Optionally, they could separately evaluate the R2 for covered and uncovered states to analyze potential generalization properties of the learned representations.
- From Figure D1, it seems relatively easy to achieve high identifiability in these experiments. The metric starts very high, drops sharply when learning begins, and then rises again. This pattern raises concerns that learning state-related representations might be too simple in these environments, and therefore not representative of more complex scenarios. Some environments include objects, i.e. parts of the true state that are relatively difficult to manipulate and represent. To strengthen the paper’s validity, the authors should include a correlation coefficient that relates to only these parts of the state space and discuss the result with respect to the ability of the agent to reach these states.

### Strengths
The paper makes an original contribution, supported by both theoretical and experiment results.

### Weaknesses
The writing should be improved and the analysis should be strengthened.

### Questions
Please, see above.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates identifiability in mutual information skill learning and contrastive successor features (CSF). The authors provide some evidence for the ability of CSF representations to recover the ground truth state used to produce the CSF because of the inner product parameterization of the loss. Further, the paper investigates the theoretical merit of different instantiations of the MISL objective, whether to use $\phi(s)^Tz$, $[\phi(s) - \phi(s')]^T$, or $[\phi(s_0) -\phi(s)]^Tz$. Finally, the authors provide some identifiability results on common RL tasks in the Deep Mind control environments.

### Strengths
This paper is well motivated and well written. Its objectives are clear and, to my knowledge, provide the first analysis of ground truth identifiability using mutual information skill learning (MISL) losses. The paper introduces the notion that a set of diverse skills and an inner product parameterization are necessary for learning a robust representation that provably recovers the ground truth state.

### Weaknesses
There are several weaknesses that exist are present in the paper that must be addressed. 

## Major Weaknesses
1. **Reality of assumptions**: It is not clear that the assumptions made in the paper are representative of reality. Namely, is it common that "each state difference is equiprobable"? What is the support for this claim?
2. **Transitions are typically not skills**: The authors also assume that "each pair of consecutive states is a skill". I believe that this is not a typical definition of skills. Skills are often defined on a longer time horizon and so single-step transitions are not wholly descriptive of what is traditionally called a skill. Further, there is literature that demonstrates that the controllable state (i.e., features that are affected by actions) are not identifiable using single-step transitions (see [1]). With that in mind, i believe it is not reasonable to define skills as (s,s') pairs.
3. **Lacking explanation for necessity of inner product parameterization**: The authors repeatedly claim that the inner product parameterization is important for identifiability, but do not share any evidence towards this end. Is there a specific result that they found that would indicate this? Or is this notion sourced from previous literature?
4. It is not clear to me why $I(s; z)$ or $I(s_0, s; z)$ are reasonable alternatives to $I(s, s'; z)$ as the former two information quantities would imply things about the visitation distributions alone. Is there a valid reason to consider these as alternatives to $I(s, s'; z)$?

### Minor Weaknesses

1. The authors claim that  $\phi(s) - \phi(s')$ is typically a learning target (line 161). Can the authors point to a method that uses such a construction? Are the authors are referring to metric-based methods like value implicit pre-training where $\|phi(s) - \phi(g) \| $ is used?
2. It believe it would be good to see actual MSE on extracted states as opposed to the correlation coefficients. In my experience, it is very possible to extract the ground truth state from encoded images in DM Control.
### Nitpicks
Line 267 has colon at end of line

### Questions
Please refer to the weaknesses section for my questions about the work.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work aims demonstrate that the causal features of an environment can be identified using contrastive successor features from up to a linear transformation. This forms an identifiability guarantee for unsupervised RL in the POMDP setting, so that the true states are derived from observations. Then, the work demonstrates empirically that the ground-truth features in several domains are recovered using CSF.

### Strengths
This work applies an elegant description of identificabiltiy in a novel context.

The theoretical framework is well articulated and provides clear reasons advantages.

The empirical results are sufficient to support the theoretical claims

### Weaknesses
The empirical results are somewhat limited in scope, considering the extension to the POMDP setting

The work is not particularly self contained, in that many of the claims are fully described in the appendix.

CSF is not the most representative MISL algorithm because it detaches teh representation learning from the policy learning more than most methods.

### Questions
It might be easier to read a slightly less aggressive citation style (the green and red boxes inserted by the citation links make the make it hard to parse the introduction). 

The second paragraph of the introduction is unclear: How does the natural connection to causality blur the borders between these fields? Interconnection does not produce a distinction between decisionmaking with scalar rewards (RL) and cause-effect models. 

Traditionally, unsupervised skill discovery is in the setting of fully observed MDPs rather than partially observed, so the focus on identificability results is somewhat jarring. This focus could be introduced a bit earlier rather that in the description of ICA.

Does the definition of diversity (definition 1) really capture the meaning? It seems like diversity should also cover some description of the breadth of skills, but in this case any pair of skill parameters is sufficient for diversity. Maybe a term such as distinguishable is more appropriate.

While CSF is a meaningful algorithm, is it really "prototypical" (l183) of MISL methods? In many cases it is closer to a representation learning algorithmm with a skill parameter added to it, rather than a MISL algorithm which often has the representation learning component built into the skill learning.

It makes sense that the assumptions are required to guarantee identifiability, but in order to make the claim that identificability is robust to assumption violations in practice it does not seem sufficient to simply claim success in other fields, since RL is a significant different setting, especially since the agent is part of the data generating process. Is there some stronger evidnece to suggest robustness to assumption violations, especially theoretically.

THe fact that Empirical observations show that the fatures of consecutive observations are close is not entirely convincing universally, since in many cases with locomotion tasks that these algorithms are tested on, the observations are often visually close together as well (an ant moving around will have similar appearances), so this closeness could just be a consequence of this. 

While it is certainly the case that I(s,s';z) is more stable to optimize, it seems like this state difference may not always be the best choice for representing a variety of different policies, since identifying policies from their state difference means that it can be hard to distinguish when policies should start changing their direction.

The experimental domains are sufficeint for demonstrating that the underlying state is discovered, though they are certainly not complete. In particular these environments focus on motion planning with various morphologies, not on the manipulation of different eleemtns in the domain, of whose underlying state might be much harder to describe, especially when randomly initialized.

Is the true underlying state really the state of the joints in mujoco, since to some extent this also is a layer of abstraction abouve the length of the joints or the shape of the agent.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
Despite the popularity of mutual information skill learning (MISL) methods in RL, they are not theoretically well understood. This paper provides an explanation of why MISL methods work by drawing from the theory of non-linear independent component analysis (ICA). Specifically, the paper draws a connection between POMDPs and data generating processes in ICA, analyzes the feasibility of assumptions required for ICA results, and proves that the features learned by one particular MISL method, contrastive successor features, can be used to recover the underlying states of the (PO)MDP up to linear transformation. The paper also explains the utility of some algorithmic design choices from the ICA perspective. The paper further verifies the theoretical results by experiments in MuJoCo and Deep-Mind Control environments.

### Strengths
- The paper makes a novel connection between MISL methods and non-linear ICA.
- The paper proves identifiability results for CSF, providing insight into why the method succeeds and why certain design choices are better.
- The paper provides empirical results to back up the theoretical insights in a number of environments.
- The paper opens up a new direction of research for understanding self-supervised RL methods, and can be of wide interest to the RL community.

### Weaknesses
- The paper is dense and can sometimes be hard to follow.
	- For example, in Section 2.3, where the paper draws a connection between MISL and DGP, it was initially unclear to me how skills fit into the picture. It was mentioned in Section 2.1 that skills can be viewed as auxiliary variables, which can be brought up here again to aid explanation.
	- Perhaps due to a limitation in space, there's almost no spacing between some paragraphs.
	- There can be more discussion on the technical details of ICA in the background. This can make the paper more self-contained.
- The feasibility of some of the assumptions made by the authors to prove identifiability for CSF relies on empirical observations in a previous paper (Figure 2 of Zheng et al., 2024). However, it seems that Figure 2 of Zheng et al. (2024) presents results for METRA, not CSF.
- Line 478-480: "As exploration and state identification showed a positive correlation with the extrinsic oracle return, defined by each environment, this suggests that identifiability is helpful for zero-shot task transfer." But in Figure 3, there does not appear to be a correlation for the columns 3-5.
- References can be better polished. There exists papers without venue names, and some papers have the arXiv version cited when they were in fact published in conferences.

### Questions
- Do you foresee that some of the assumptions that rely on empirical observations (e.g., assumptions ii, iii, and iv) will hold for other MISL methods?

### Soundness
3

### Presentation
2

### Contribution
3
