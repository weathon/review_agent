# Physics of Learning: A Lagrangian perspective to different learning paradigms

- Decision: Reject
- Scores: 2, 6, 2, 2, 8

## Abstract
We study the problem of building an efficient learning system. Efficient learning processes information in the least time, i.e., building a system that reaches a desired error threshold with the least number of observations. Building upon least action principles from physics, we derive classic learning algorithms, Bellman's optimality equation in reinforcement learning, and the Adam optimizer in generative models from first principles, i.e., the Learning $\textit{Lagrangian}$. We postulate that learning searches for stationary paths in the Lagrangian, and learning algorithms are derivable by seeking the stationary trajectories.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper expresses various learning setups through variational functionals. The authors follow a well-known formalism from physics (Lagrange functions/principle of least action). The examples include active learning, the Bellman optimality equation, and supervised learning.

### Strengths
1) The paper studies analogies to variational calculus in physics/math. This could allow to bring well-established techniques from other fields to machine learning research.

2) The authors attempt to unify different learning paradigms, which could create a deeper understanding of learning mechanisms.

### Weaknesses
1) It is difficult to read the paper.

- Generally, it would help to introduce the symbols in a more mathematical way (discrete/continuous, dimension).
- At the beginning (L209,L216), sample-efficiency and compute time are introduced as a characterization of efficient learning but the latter seems to play no role in the paper. Omitting needless concepts could make it easier to read.
- The purpose of Section 2 and Insight 1 is not clear to me. It appears to be a side observation that is not relevant for the core topic of the paper. 
- The standard Lagrangian/Hamiltonian frameworks, as I know them, usually starts with a clear declaration of the variables and their conjugate variables. I would recommend this for each of the examples.
- Insight 2: Planning usually refers to using a learned/known model of the environment for better action selection. 

2) The contribution to derive learning algorithms from Lagrangians is not as strong as claimed. 

- Section 3.1 appears to me to be specific to linear regression. 
- The formulation of the Bellman equation through a variational principle is (as the authors say) not novel. 
- It is claimed that the Adam optimizer is derived. However, it looks more like a derivation of maximum likelihood estimation in Section 3.3 and that Adam is one way of approximating this. I doubt that Adam with all its features, such as normalization of update sizes, can be derived from a simple Lagrangian like the one in Equation 26.

### Questions
1) Time evolution in physics is unique and so is the solution of the corresponding variational formulations. How is this uniqueness interpreted in a learning context? Is there a unique best learning algorithm? 

2) How can a formulation in terms of variational principles benefit machine learning research? In math/physics, it allows to incorporate symmetries more easily, using certain proof techniques, geometric invariance, ... Can you demonstrate any of these advantages?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes interpreting efficient learning as a physical process that makes a Lagrangian stationary. The authors present concrete examples with mathematical derivations, including linear regression, the Bellman equation in reinforcement learning, and Adam/RMSProp update rules.

### Strengths
- This paper is well-written and easy to follow. The structure is clear, and the explanations are concise yet informative.
- The unifying viewpoint of learning algorithms as physical processes is intriguing.

### Weaknesses
I am concerned about the practical implications of this framework. While the perspective is theoretically appealing, it is not clear how it leads to tangible benefits for designing or improving learning algorithms in practice. For example, some prior work applies physical concepts to optimization and reports practical improvements (see, e.g., https://arxiv.org/abs/1906.01563).

### Questions
Does the current framework have the potential to inspire new algorithms or learning paradigms? Even without experiments, a discussion of possible practical uses would strengthen the paper's impact.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper identifies a connection between learning and mechanical systems based on the principle of least action. The paper shows how certain well known machine learning algorithms can be derived by considering the Lagrangian.The algorithms span different types of learning such as parametric models, reinforcement learning and generative models.

### Strengths
The paper makes a connection between physical systems and learning ones. It does this using a Lagrangian and the principle of least action. The idea is that a sample efficient learning process should correspond to a stationary trajectory. Rather than in physical space and indexed by time, this is in information space indexed by sample number. The similarity of the two types of processes is striking, and it's nice to see a conceptual paper. Using this, the authors give derivations of three types of learning algorithm: parametric, reinforcement and supervised learning. The fact the types are somewhat different lends strength to the main argument of the paper. However, there is a fatal flaw.. see Weaknesses.

### Weaknesses
The analogy--central to the paper--between learning algorithms (which are optimal ways of accruing some kind of quantity) and solutions to Euler-Lagrange equations is not totally surprising. And what's worse, this observation has even been explicitly made before, see for example a blogpost by John Baez about the Bellman Equation from 2014: https://johncarlosbaez.wordpress.com/2014/10/30/sensing-and-acting-under-information-constraints/ which summarizes some points from a talk by Naftali Tishby. Tishby is not cited in the present paper. 

I think the paper should better explain what parts are novel and what have been done before, showing clear novel content,  in order to warrant consideration in a venue like ICLR. The presentation in the paper would also need to be significantly improved, though the presentation issues are a moot point given the problem with the central premise. For future work, here are some presentation issues:

The paper has an idiosyncratic style. For example, immediately after the abstract, a table is used to explain some the main ideas. Moving the table would help somewhat, but I suggest a more textual introduction would better help readers understand. 

Sloppy wording regarding physics is also a recurring problem in the paper. This even starts in Table 1 (the first thing the reader comes across!) with phrases such as:  *" .. from first principles, i.e., the Learning Lagrangian"*, though the Learning Lagrangian itself is not a principle; or *".. searches for stationary paths in the Lagrangian"* where the authors presumably mean this as shorthand for ".. searches for paths with stationary action, where action is defined in terms of the Learning Lagrangian". Freewheeling wording like this is especially disconcerting in a paper whose main contribution is revealing an analogy between phenomena in different areas.. it's important for the technical statements of these phenomena to be stated rigorously, or at least following conventional shorthand.

The introduction section, itself, is rather short even though it includes a somewhat terse related work section, typically a section in its own right. I suggest it should be separated and expanded. Throughout the paper, various ideas are introduced using words or sections in bold print rather than a more traditional structure. 

Within the paper there are a few mentions of limitations, or assumptions, it would be nice to collect these and have a limitations and future work section before or part of the conclusions. 

There is a lack of experimental validation which the authors themselves recognize. Depending on the venue where this is submitted, that might not be an issue (e.g. in a book chapter), but especially for a machine learning conference, and given that the central analogy is not totally novel, it would be good to have at least some practical benefit from the ideas.

### Questions
Is Table 1 really intended to have the words "Fermat's principle, Hamiltonian, the Lagrangian" in the bottom left cell? The lists of Applications and of Algorithms in the bottom right and center cells seem like lists of what can be obtained as stationary solutions under a corresponding  Learning Lagrangian, and the applications in which these occur respectively. But what's the bottom left cell doing? And presumably there are actually two tables one on top of the other, or are Applications really meant to be under Physics, and Algorithms under Learning?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a framework to connect and unify several learning settings through the language of classical physics, specifically the principle of least action. The framework describes learning as the minimisation of a Lagrangian where, in place of physical notions of position and time, the authors instead consider a dataset that is iteratively added to. They provide three applications of this framework—to experimental design, reinforcement learning, and optimisation—recovering classical machine learning results such as A-optimality, Bellman’s optimality equation, and natural gradient descent.

### Strengths
- The paper identifies a connection between physics and machine learning, two fields of interest to the conference.
- The authors attempt to unify concepts in experimental design, reinforcement learning, and second-order optimisation. In this sense, the work is refreshingly ambitious.
- The proposed formulation of active learning/experimental design via physical analogues appears to be a novel perspective, though I am not familiar enough with this specific literature to comment on its novelty with confidence.

### Weaknesses
- The paper’s clarity is a significant concern. Many explanations, especially in Section 3, feel underdeveloped or are presented too hastily. The terms “postulate”, “conjecture”, and “hypothesise” are used frequently, often immediately before a strong claim is made, which blurs the line between assumption and proof.
- The paper uses physics notation and ideas without sufficient exposition. A large proportion of the ICLR audience has a background in Mathematics and Computer Science; notation that is not standard in these fields should be clearly defined. I would encourage the authors to add a dedicated background/notation section.
- The paper's motivation is somewhat puzzling. The authors claim a "surprising link" between physics and ML, yet also describe how important physics has been to ML's development. The fact that different learning problems can be cast as optimisation problems (minimising a Lagrangian) is not suprising given the role that classical physics has played in this development.
- The contribution seems overstated on several occasions. The authors claim their work provides physical justification for the Adam optimiser, but the analysis is for natural gradient descent, which is theoretically distinct. Similarly, the claim of applications to generative models is justified by this same NGD analysis. This connection feels tenuous and appears to overstate the framework's relevance to contemporary generative models.
- "Insight No. 1" (that learning is a decelerating process) is a well-known property of learning curves and may not constitute a novel insight of this framework.
- The "Main Postulate" is conceptually challenging. It states that learning is a search for a stationary path (data sequence), but in most learning settings (outside of experimental design), the data path is fixed, and the search is over the model class.
- The paper's contribution appears thin for a main-track submission. It lacks novel theoretical guarantees or new experimental results. The primary contribution is the proposal of a framework and various conjectures. However, the framework is presented in a way that is difficult to follow, which unfortunately detracts from its value as a contribution.
- The equality on the right-hand side of (5) doesn’t appear to be correct (see questions).

### Questions
- Could the authors please provide a derivation for the equality on the right-hand side of equation (5)?
- Are the authors familiar with the online-to-PAC framework (Lugosi and Neu, 2023)? It might be relevant to their work.
- I found section 3.3 very difficult to follow. Could the authors please clearly explain this, taking greater care to state what things are assumed/hypothesized and which parts are mathematical deductions?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In the present paper the authors introduce a new perspective through which to view the learning algorithms driving modern machine learning. Applying Lagrangian mechanics, stationary paths are looked for to re-derive A-optimality, Bellman’s equation for reinforcement learning, and the widely used optimization algorithms Adam, and RMSprop from a dynamical systems angle. Observing the unification of these learning algorithms under the same formalism, the authors posit that learning dynamics obey the principle of least action.

### Strengths
The core strengths of the presented work are two-fold:
* Intellectual clarity of the presented ideas, this includes the derivation of the results, and their motivation.
* Rigor of derivations

While clearly an original piece of work, the paper builds on a large foundation of physics-based approaches in machine learning. Beyond the intellectual clarity, the presented work scores highly with regards to its significance with the potential help and understanding the result provides.

### Weaknesses
* At its core, the presented approach introduces a structured treatment of the space of learning dynamics, and can as such be interpreted as leveraging the information geometry [1]. The authors sadly fail to draw this connection,                                                                                                                        
* There exists a wide body of literature at the intersection of machine learning’s learning dynamics, and approaches from physics such as classical mechanics, and statistical physics as pointed out by the authors (line 72-80). Nevertheless, there exist a number of works who specifically target the learning dynamics with a variety of techniques from classical mechanics to which this paper should be set in relation to. In order of proximity to the presented work:
    * The work by Winer and Hanin [2] takes the perspective of viewing the network as a Hamiltonian over its inputs, and is hence closely related to the present work.
    *  Redman et al. [3] utilize Koopman operators to characterize the learning process, and are hence arguably orthogonal to the present work.

References:
1. Ollivier, Yann, et al. "Information-geometric optimization algorithms: A unifying picture via invariance principles." Journal of Machine Learning Research 18.18 (2017): 1-65.
2. Winer, Mike, and Boris Hanin. "Deep Neural Nets as Hamiltonians." arXiv preprint arXiv:2503.23982 (2025).
3. Redman, William, et al. "Identifying equivalent training dynamics." Advances in Neural Information Processing Systems 37 (2024): 23603-23629.

### Questions
* Re-deriving algorithms through the lens of Lagrangian dynamics as a latent unifying principle in itself is highly thought-provoking, while the re-derivation is one step, how do you envision to leverage the developed formalism going forward to aid in the development of new algorithms?
* Can the authors’ formalism, and posited principle of least action, be extended to more advanced forms of differentiation such as natural gradients, and the resulting algorithms such as natural gradient descent?

### Soundness
4

### Presentation
3

### Contribution
3
