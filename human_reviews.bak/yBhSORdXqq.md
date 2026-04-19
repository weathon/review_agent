# Modular addition without black-boxes: Compressing explanations of MLPs that compute numerical integration

- Decision: Reject
- Scores: 6, 3, 8, 5

## Abstract
The goal of mechanistic interpretability is discovering a simple, low-rank algorithm implemented by models.
While we can compress activations into features, compressing nonlinear feature-maps---like MLP layers---is an open problem.
In this work, we present the first case study in rigorously compressing nonlinear feature-maps.
We work in the classic setting of the modular addition models (Nanda et al., 2023), and target a non-vacuous bound on the behavior of the ReLU MLP in time linear in the parameter-count of the circuit.
To study the ReLU MLP analytically, we use the infinite-width lens, which turns post-activation matrix multiplications into approximate integrals.
We discover a novel interpretation of the MLP layer in one-layer transformers implementing the “pizza” algorithm (Zhong et al., 2023): the MLP can be understood as evaluating a quadrature scheme, where each neuron computes the area of a rectangle under the curve of a trigonometric integral identity.
Our code is available at [https://tinyurl.com/mod-add-integration](https://tinyurl.com/mod-add-integration).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors address the challenge of interpretibility of MLP layers in transformer models. Focusing on toy models trained for modular addition, the authors go beyond “approximate explanation” and propose to form a “complete interpretation” through a quadrature scheme, where neurons can be understood as approximating areas under curves in trigonometric functions. The method allows them to bound the MLP in total time linear in the neuron numbers.

### Strengths
- The authors provides a depth analysis, with a formal definition of “complete interpretation”, a clear problem setting, and an evaluation on a simplified toy model, which enables a focused and mathematically rigorous analysis.

- To evaluate the practical usage of the numerical integration interpretation, the authors validate the methods by establishing non-vacuous bounds for the MLP outputs in linear time.

### Weaknesses
- The paper is in general hard to follow for a broader audience, especially for those unfamiliar with topics such as mechanistic interpretability, quadrature themes etc. 

- A discussion on the scalability beyond the toy models is missing. While a simple toy model allows for a focus analysis, a discussion on scaling to more complex real-word tasks is beneficial.

- As the author mentioned that there are couple of previous studies on the interpretability of MLPs, a comparative study with existing methods would be beneficial to fully understand the advantage of the proposed method.

### Questions
- Is it possible to extend the proposed framework to MLP layers in a deeper transformers?

- The study seems only focuses on understanding MLPs within a constrained setting where the MLP approximates trigonometric functions, is it possible to extend to non-trigonometric functions? 

- I am still confused on the definition of “complete interpretation”, are there any evaluation metrics?

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
5

### Summary
This paper examines how 2-layer ReLU MLPs implement the trigonometric identities required by the 'Pizza' algorithm (previously described in earlier work) within a one-layer modified transformer architecture. The authors found that the MLP approximate a trigonometric integration by summing over its hidden neurons, they also provide non-vacuous bounds on the output of the MLP on all inputs in time linear in the parameters.

### Strengths
1. Simple setting with a clear explanation of their setting
2. Offered empirical check of their theoretical results

### Weaknesses
The work only focuses on explaining details of a component from the previously proposed Pizza Algorithm (Zhong et al., 2023), specifically how MLPs sum periodic features generated before activation functions. Furthermore, the simplified model (lines 125-126) is nearly identical to the fully solved model presented in [1], with ReLU activation replacing the quadratic activation. Since exact finite summation for ReLU remains an open problem, the authors instead approximate the sum using integral methods and argue that neurons effectively approximate this integral. 

While I appreciate the authors work on explaining the role of secondary frequencies in Section 5, it represents only a minor contribution. Overall, this paper's contribution falls below the standard expected at ICLR-level conferences.

[1] A. Gromov, Grokking modular arithmetic, arXiv:2301.02679

### Questions
1. Have the authors investigated the effect of weight decay tuning on the clarity of their experimental results?
2. What factors influence the emergence of secondary frequencies in the model?

Line 023: "trignometric" typo in abstract
Line 992: Broken equation ref

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
MLP layers in transformer models remain poorly understood in the mechanistic interpretability literature; even detailed analyses of toy models like modular addition transformers tend to treat them as black boxes that compute certain functions and verify their input-output behavior by exhaustive search over the input space. The authors propose to fully explain—by which they mean find a description that is linear in the number of parameters—the MLP's behavior in the modular addition task. They determine that the MLP is responsible for numerically computing an integral of a trigonometric identity that is useful for the modular addition task, and even prove non-vacuous bounds on the error of their description.

Edit: confidence increased 3->4 during rebuttals

### Strengths
* **Addressing open problems in mechanistic interpretability.** While modular addition is a well-studied problem, the MLP sublayer's behavior has mostly been treated as a black box. By explaining what the MLP is doing, the authors help move the study of this model beyond this black box treatment. While understanding one piece of a toy model may seem like a modest contribution, this is a major missing piece of an active area of research and a major step towards producing and verifying mathematical descriptions of MLP behavior. 
* **Resolving the discrepancy between "clock" and "pizza" algorithms.** In section 5, the authors solve the mystery of why "pizza" algorithm logits resemble "clock" logits by explaining the role that secondary frequencies play. This is also an unresolved question in modular addition.
* **Balance of perspectives.** The authors are thorough about including empirical/computational results, mechanistic descriptions of the modular addition task, and proofs. They also extend results to 150 other transformer models (which differ in their random seed).

### Weaknesses
* **Limited scope.** Although the paper is thorough and addresses an open question in mechanistic interpretability, its concrete contribution is simply to explain one component of one model (albeit under multiple random seeds) on one toy problem. While this is a step forward, and the authors briefly describe the broader implications for non-algorithmic interpretability practitioners in Section 7, the paper's contribution is circumscribed (arguably modest) as written. 
* **Bounds are crude.** As the authors acknowledge (354), the bounds they prove are quite loose, owing to an incomplete understanding of the model. Proving tighter bounds or arguing more convincingly for why tighter bounds are not possible would improve the paper. 
* **Clarity.** The argumentation in some of the mathematically dense sections (especially Section 5) can be hard to follow.

### Questions
* You suggest that this approach could be useful for practitioners in other domains. What properties of the model/problem would make it more likely that this approach would yield a good explanation?
* You leave deriving tighter error bounds to future work. What avenues for proving these bounds are you considering?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper studies the role of MLP layers in a constant-attention Transformer (i.e. ReLU-MLP network) trained on modular addition. They write the overall operation of the network as $embedding \\rightarrow ReLU \\rightarrow fully-connected$ and write the trigonometric functions corresponding to these components. They show that the sum over same-frequency neurons implements a sum over random phases, which can thought of as approximating an integral. They provide empirical evidence for their claims and show that their method enables the computation of non-trivial error bounds on the outputs.

### Strengths
- The analysis of secondary frequencies is new.

- The results are presented in a quite candid fashion, with the limitations/discrepancies of the analysis freely discussed, along with potential explanations.

- The code used to produce the results is provided in the form of a link to a Colab Notebook.

### Weaknesses
My main concerns lie in regards to the novelty and soundness of this work.

- The authors only consider Transformers with constant attention. This makes the network a ReLU-MLP and not a Transformer. This should be made clear in the paper. If the objective is to study Transformers, then non-constant attention should also be examined.

- Interpretability for MLPs trained on modular addition has already been extensively studied in literature [1,2]. Although these prior works focused on MLPs quadratic activations, many of the qualitative conclusions are similar to the ReLU case studied in this work. The authors should present a detailed comparison to these works and highlight the novelty. (Specifically, what are the insights from the current work that are lacking upon merging the findings of [1,2,3].)

- As stated, it is difficult to draw conclusions from the error bounds -- especially the part with splitting ReLU into identity and absolute value components. This part would benefit from further justification as well as clarity of notation (e.g. lines 309-310).

### Questions
- Are all the experiments conducted with constant head Transformers? I couldn't find this detail in the Colab notebook since the checkpoints are directly imported for analysis.

- According to [3], networks can learn either "clock" or "pizza" algorithm, depending on the details of training. How do the authors ensure that the learned algorithm is always "pizza"?

- Are Figures 2,5 empirical results or just theoretical visualizations? 

- Lines 299-303: The omission of x/2 term seems rather hand-wavy. How does one see that computing the bounds after ignoring this term is a reasonable thing to do? It is unclear to me even after referring to Appendix K.

- Does Table 1 imply that the contribution from $\\varepsilon_\\phi$ is very large? This seems unlikely since $\\psi_i - 2\\phi_i$ are quite small in Figure 3.

- Line 349: Does "brute force" simply mean empirical?

- Line 373-377: Zhong et al. [3] do not claim that logits are always of the "pizza" form. They claimed that either "clock" or "pizza" algorithm can be found, depending on the hyperparameters.

- Lines 395-397: I am puzzled by the $R^2$ results. The paper sets out to explain the "pizza" algorithm. However, the logits seem to align better with the "clock" algorithm. 

- Lines 401-403: This paragraph claims that the secondary frequencies are important. However, Figure 4 shows that the top Fourier component explains more than 99% of the variance in all of the neurons. How does one reconcile these two results?

[1] Gromov, Grokking modular arithmetic (2023)

[2] Doshi et al., To grok or not to grok: Disentangling generalization and memorization on corrupted corrupted algorithmic datasets (2024)

[3] Zhong et al., The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks (2023)

### Soundness
2

### Presentation
2

### Contribution
2
