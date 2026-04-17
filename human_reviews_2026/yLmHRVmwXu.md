# Characterising the Inductive Biases of Neural Networks on Boolean Data

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Deep neural networks are renowned for their ability to generalise well across diverse tasks, even when heavily overparameterized. Existing works offer only partial explanations (for example, the NTK-based task-model alignment explanation neglects feature learning). Here, we provide an end-to-end, analytically tractable case study that links a network’s inductive prior, its training dynamics including feature learning, and its eventual generalisation. Specifically, we exploit the one-to-one correspondence between depth-2 discrete fully connected networks and disjunctive normal form (DNF) formulas by training on Boolean functions. Under a Monte Carlo learning algorithm, our model exhibits predictable training dynamics and the emergence of interpretable features. This framework allows us to trace, in detail, how inductive bias and feature formation drive generalisation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This works aims to derive a tractable framework towards understanding the inductive biases and generalization properties of neural networks. To this end, the authors consider Disjunctive Normal Forms (DNFs) and discrete depth two fully connected networks (DCNF) to analyze the aforementioned biases and properties in terms of appropriate complexity measures K(f) and prior probalities P(f). The correspondence shown in Proposition 2.7 allows mapping the networks weights (and architecture) directly in a symbolic logical representation, allowing investigations into the behavior of a function f with low DNF complexity, i.e., the shortest possible DNF expressing the function f, and the bias present in both untrained and trained architectures. 

The main findings are that: 1) randomly initialized DFCNs induce a prior distribution P(f) that show a strong simplicity bias, i.e., simple functions occupy larger regions of the parameter space compared to complex ones, and 2) When training networks, simple functions generalize better and are easier to train due to the bias towards low complexity functions, leading at the same time to a difficulty in learning higher complexity functions.

### Strengths
Overall, the paper offers an interesting and principled analysis of a simplified theoretical model. The authors succeed in showing the connections between DNFs and DFCNs and how the function complexity leads to inductive biases and insights into networks generalization. This mapping is indeed insightful, and offers an alternative view towards representation learning.

### Weaknesses
My main concern revolves around the impact and generalization of the approach to other not so controlled settings. Indeed, while the authors aim to tie the generalization performance to derived inductive properties based on boolean function complexity, it remains highly uncertain how and if this discrete Boolean framework translates to more standard continuous, multi layer and high dimensional NNs.

Even if this work's aim is a proof of concept or a presentation of the connections between disjunctive normal forms and discrete fully connected networks, the focus on depth-two is itself very restrictive. I would have liked to see some insights or maybe derivations for networks with more than two layers, to assess how the proposed framework scales or adapts to different settings. 

The authors apart from using depth two DFCNs, they also only consider ReLU activations. How would the framework behave if there was another activation in place (given that it satisfies the boolean nature of the examined setting)?

The main experiments are very limited. It is clear that the complexity of the proposed method is very high, leading the authors to consider only n=7. How are the wall time measurements as n increases?

The main text is dense, with many essential derivations and details deferred to the appendix. This renders the paper harder to follow and most of the time, breaks the reading flow. Moreover, some appropriate discussions that motivate and set the method in a specific context are also in the appendix.

### Questions
Please see the weaknesses section.

### Soundness
3

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
The paper analyses the inductive bias of neural networks using a fully discrete, analytically tractable model.
A depth-2 discrete fully connected network (DFCN) with ternary weights $\{-1,0,1\}$ is shown to map one-to-one to Boolean DNF formulas, enabling a definition of function complexity $K(f)$.
By enumerating network configurations, the authors derive class-dependent asymptotic bounds for the prior $P(f)$: for example, t-entropy and k-parity functions follow distinct scaling laws (Table 2), showing that simple functions occupy much larger parameter volume.
Weight decay approximately adds an $e^{-\lambda K(f)}$ factor to the posterior, further amplifying this simplicity bias.
Small Boolean experiments qualitatively support the analytic results.

### Strengths
1. The paper presents a clear and internally consistent analysis. The proposed DFCN–DNF correspondence is an effective way to formalize inductive bias, allowing the authors to make explicit statements about which functions a network represents.
2. The results go beyond earlier simplicity-bias discussions by providing class-specific scaling laws (as shown in Table 2). This clarifies that different Boolean function families exhibit distinct probabilistic behavior rather than a single universal exponential law.
3. The connection between weight decay and a multiplicative $e^{-\lambda K(f)}$ factor in the posterior is well argued and conceptually coherent. The exposition is concise, and the figures and tables directly support the theoretical claims.

### Weaknesses
1. The work offers limited novelty. The simplicity bias of neural networks has been documented extensively, and this paper largely reformulates it within a discrete combinatorial model.
2. The theoretical depth is modest. the Boolean setting restricts the scope of the conclusions.
3. The MCMC and greedy search methods used for training differ substantially from gradient-based optimization, so their connection to real neural-network behavior remains unclear.
4. The presentation of related work could be better organized. Prior studies are frequently cited within the derivations themselves, which interrupts the flow and blurs the boundary between previous results and the authors’ own contributions.
5. The detailed enumeration over three hand-defined Boolean function families illustrates how the simplicity bias varies across structural types, but the result remains largely demonstrative rather than revealing new theoretical insight. Theorem 3.2 appears to bundle three independent asymptotic cases (constant, t-entropy, and k-parity) into a single statement mainly for narrative coherence.

### Questions
See Weaknesses.

ALso:
1. Could the authors move beyond the three enumerated Function Class and develop a more general or principled way to characterise function classes?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a framework to study inductive bias, feature learning, and generalization in neural networks by focusing on depth-2 discrete fully connected networks (DFCNs) trained on Boolean functions. They show a bijective correspondence between DFCNs and disjunctive normal form (DNF) formulas. This equivalence allows the authors to define a function-level complexity measure K(f) based on the minimal DNF length and to link it directly to network weight norms.
They empirically and theoretically derive a simplicity-biased prior P(f) over Boolean functions, showing that functions with small K(f) occupy exponentially larger regions of parameter space. Using Bayesian and greedy SGD-like training algorithms, the paper demonstrates that generalization correlates strongly with DNF complexity. They further show that weight decay improves generalization for low-complexity targets.

### Strengths
1.	Novel framework for understanding inductive bias of NNs.
2.	New insights relating DNF complexity and learnability.
3.	The paper is mostly clearly written.
4.	Thorough empirical validation.

### Weaknesses
1. The main weakness is that the framework limits the input dimension substantially, allowing only n≤7. Therefore, it does not model high dimensional settings, which are key in NN applications. Since the data is binary, the resulting datasets are also very small.
2. No experiments were performed with NNs and algorithms used in practice – this can strengthen the conclusions of the paper.
3. Missing reference– Bronstein et al. (UAI 2022) that study the inductive bias of NNs on read-once DNFs.


Bronstein, Ido, Alon Brutzkus, and Amir Globerson. "On the inductive bias of neural networks for learning read-once dnfs." Uncertainty in Artificial Intelligence. PMLR, 2022.

### Questions
Lines 244-247: the PAC-Bayes bound is unclear. How was the KL divergence derived? What exactly is ϵ(f)? Does the right-hand side approach 1 when P(f) is large?

In the experiments, what is the main factor that limits the input dimension? Are there ways to relax this restriction and show that the conclusions hold (even approximately) for higher dimensional settings?

### Soundness
3

### Presentation
3

### Contribution
2
