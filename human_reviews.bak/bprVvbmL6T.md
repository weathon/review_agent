# Off-policy Evaluation with Deeply-abstracted States

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 3

## Abstract
Off-policy evaluation (OPE) is crucial  for assessing a target policy's impact offline before its deployment. However, achieving accurate OPE in  large state spaces remains challenging. This paper studies state abstractions -- originally designed for policy learning -- in the context of OPE. Our contributions are three-fold: (i) We define a set of irrelevance conditions central to learning state abstractions for OPE, and derive a backward-model-irrelevance condition for achieving irrelevance in  (marginalized) importance sampling ratios by constructing a time-reversed Markov decision process (MDP) based on the standard MDP. (ii) We propose a novel iterative procedure that sequentially projects the original state space into a smaller space, resulting in a deeply-abstracted state, which substantially simplify the sample complexity of OPE arising from high cardinality. (iii) We prove the Fisher consistencies of various OPE estimators when applied to our proposed abstract state spaces.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper "Off-Policy Evaluation with Deeply Abstracted States" addresses the challenge of off-policy evaluation (OPE) in large state spaces. Off-policy evaluation assesses the impact of a target policy using historical data collected under a different behavior policy. Large state spaces make this task difficult due to increased sample complexity. The authors propose state abstraction techniques for OPE, introducing a method to construct deeply abstracted states by iteratively projecting the original state space into smaller spaces. Their contributions include (1) defining irrelevance conditions for OPE, (2) creating an iterative procedure to compress the state space, and (3) proving the Fisher consistency of several OPE estimators when applied to abstract state spaces. The proposed abstractions are validated through numerical experiments, demonstrating improvements in sample efficiency and OPE accuracy.

### Strengths
The paper applies state abstraction, which has been widely studied for policy learning. This approach has the potential to significantly reduce state space cardinality and improve the accuracy of OPE estimators. The proposed iterative procedure for generating deeply abstracted states is a creative solution. It simplifies the sample complexity of OPE by shrinking the state space without compromising the representational capacity for policy evaluation. The method supports multiple types of OPE estimators, including value-based methods, importance sampling, and doubly robust methods. This shows the paper’s wide relevance across different off-policy evaluation scenarios. The authors provide rigorous theoretical backing, including irrelevance conditions and Fisher consistency proofs for various estimators, ensuring that the abstracted states do not distort the OPE process.

### Weaknesses
The iterative nature of the abstraction process, while powerful, might introduce computational overhead that is not fully addressed. Although the paper validates the approach with numerical experiments, the experimental section seems underdeveloped. There is a lack of detailed comparisons with baseline methods, especially in real-world scenarios. It would be helpful to see more empirical results across diverse datasets to better understand the practical performance of deeply abstracted states.

This paper presents a compelling theoretical advancement in state abstraction for OPE, but further exploration is needed to evaluate its practical impact and computational efficiency across varied domains.

### Questions
What is the computational overhead of the iterative state abstraction process, and how does it scale with increasingly large state spaces? Are there scenarios where the cost of abstraction outweighs the benefits in terms of sample complexity reduction?

Can the proposed methodology be easily integrated into existing OPE frameworks or reinforcement learning pipelines? What modifications would be needed for real-world application?

### Soundness
3

### Presentation
3

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
The paper proposes a set of conditions on state abstractions in order to effectively use the state abstraction for off-policy evaluation. It then proposes an iterative procedure to learn abstractions that satisfy these conditions, and shows that the learned abstractions result in accurate OPE for various OPE estimators.

### Strengths
- The paper nicely unifies several ideas such as referencing prior work on model-irrelevant and pi-irrelevant abstractions. And also relating its backward model to prior work that learns an inverse dynamics model.
- The results are for a wide class of OPE estimators compared to prior work that discusses OPE and abstractions only for a particular OPE estimator.
- The work tackles a problem that has received relatively little attention.

### Weaknesses
- The motivation of the paper is unclear. There is a possibly interesting idea here in the backward model, but it is unclear what the purpose of this model is. For example, Lemma 1 suggests that the previous irrelevance conditions are sufficient for consistent OPE. So it is unclear to me why we need an alternative condition to also get consistency. From theorem 1, I see that the backward model also gives us two additional irrelevance conditions, but why do we need these if all we care about is consistent OPE? Is the backward model condition easier to learn? The empirical method seems to combine both forward and backward models, but is there a reason to do so? Why is only one of them insufficient? This seems core to the paper and needs to be made clearer.
- The paper may need a big rewrite. 
  - The main paper has an extremely sparse empirical section, which is a weakness given that the paper is pitched as making an empirical contribution.
  - The paper is also missing a discussion and conclusion section. 
  - The introduction immediately uses terms such as backward model or time-reversed MDP as if their meanings are obvious but they are not necessarily so and need to be clear. 
  - It is unconventional to write the Q-function as Q(a, s) vs. Q(s,a) (Line 155). There are several instances of switching the notation. 
$\psi$ is not defined although its meaning can be inferred based on Figure 1. 
  - Line 240 has a missing section reference.
  - The meaning of “identifiable” (line 247) needs to be specified.
- The empirical details need to be improved. Section 5 does not include any details of the loss functions (they are in the appendix) which is critical given that the paper is pitched as also making an empirical contribution. Section 5 also includes results only on one domain, which is highly insufficient to make a general claim about the method. There are also no details on the number of trials, seeds, confidence intervals etc.

### Questions
See question above on why backward model is needed and why Definition 6 and 7 are needed given other assumptions are sufficient for consistent OPE.

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors propose a method to perform state abstraction for the purpose of off-police evaluation using deep learning. Their method optimizes two notions of state similarity iteratively. The authors also prove theoretical consistency of their results.

### Strengths
1. The authors propose the novel idea of backwards model irrelevance. 

2. The paper is well organized, and provides a lot of background and explanations.

### Weaknesses
1. Despite the authors' attempts at making things clearer, I didn't fully understand the notion of backwards model irrelevance, and why it would generally require more than one iteration with the forward model irrelevance. I think the paper would benefit from making this point clearer somehow. I found the examples on page 9 more confusing than helpful.

2. The example in the experiments looks very artificial. While it does show improved performance for the author's method, it being tailored to the the problem raises the question - how important is this problem in general. OPE is widely researched, but state abstraction for OPE is usually harder to justify, I'm not convinced by the experiment that it's a useful or significant technique. I would suggest giving one non-state-abstraction baseline, even if it outperforms the current results. Also, I think it would be good to run on another environments given in other OPE papers.

### Questions
1. Can you say anything about the rate of convergence for the iterative process?

2. What's the true meaning of backwards model irrelevance and why does it make sense to iteratively apply it with forward model irrelevance?

3. How are your results compared to non-state-abstraction methods?

4. This is probably in the paper but I couldn't find it - how many iterations are required and when do you stop iterating? It would have made sense for me if you included this in the pseudu-algorithm in the beginning of Subsection 4.3.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper studies the use of state abstractions, learned through forward and backward losses, for the task of off-policy evaluation (OPE). It outlines the conditions that the abstractions must satisfy to enable the application of OPE methods to a more compact abstract MDP. Specifically, the model-irrelevance condition is adapted for policy evaluation and applied to both forward and backward views of the MDP. By sequentially applying learning rules for forward and backward model irrelevance conditions, a two-step abstraction learning procedure is developed. This procedure results in a compact MDP, making it possible to apply standard OPE methods effectively.

### Strengths
- The irrelevance conditions proposed in [1] are adapted well to impose irrelevance of components of OPE methods (like the IS and MIS ratios).
- The final result, that OPE methods can be applied directly to the abstract MDP obtained after the two-step abstraction learning process, is a useful result. Effective two-step abstraction learning methods can make OPE practically applicable for problems with high dimensional states.

---
[1] Li, Lihong, Thomas J. Walsh, and Michael L. Littman. "Towards a unified theory of state abstraction for MDPs." AI&M 1.2 (2006): 3.

### Weaknesses
- Fisher consistency: The statement of the results claim to show Fisher consistency of the corresponding abstract-state estimators, however, the proofs are for unbiasedness and identifiability. It is unclear whether these two conditions imply Fisher consistency.
- Experiments:
  - [L511] states that the reported metrics are MSE and absolute bias, however, Figure 5 reports relative MSE and relative absolute bias. The latter has not been defined.
  - The abstraction learning itself requires some amount of data that aids the OPE performance, which must also be reported.
     - The number of data points for OPE is very less (10-60 trajectories) and the plots must be extended further to the right for a meaningful comparison for the baseline methods to also receive sufficient data.
  - The error bars are not defined (standard error? confidence intervals?) and are computed over only 20 trials.

Nitpicks:
- All the irrelevance conditions are a simple adaptation of the ones in [1] and the paper’s claim on *defining* these conditions for OPE is overclaiming.

### Questions
- Can the authors more rigorously specify why their proof sketches imply Fisher consistency of their methods? Moreover, are the OPE methods (without abstraction) themselves Fisher consistent? 
- How much data is required for learning abstractions in the experiments with the two-step procedure?
- A comparison of MSE with importance-sampling (and not just FQE) would be insightful, particularly for larger dataset sizes, would be insightful. Being an unbiased OPE method, it would be interesting to see up till what point the variance reduction afforded by abstractions is beneficial despite the bias incurred.

### Soundness
2

### Presentation
2

### Contribution
3
