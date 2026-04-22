# On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking

- Avg Score: 3.00
- Decision: Reject
- Scores: 8, 2, 0, 2

## Abstract
We present a comprehensive analysis of how two-layer neural networks learn features to solve the modular addition task.
Our work provides a full mechanistic interpretation of the learned model and a theoretical explanation of its training dynamics.
First, we empirically show that trained networks learn a sparse Fourier representation; each neuron's parameters form a trigonometric pattern corresponding to a single frequency.
We identify two key structural properties: phase alignment, where a neuron's output phase is twice its input phase, and model symmetry, where phases are uniformly distributed among neurons sharing the same frequency, particularly when overparametrized.
We prove that these properties allow the network to collectively approximate an indicator function on the correct logic for the modular addition task.
While individual neurons produce noisy signals, the phase symmetry enables a majority-voting scheme that cancels out noise, allowing the network to robustly identify the correct sum.
We then explain how these features are learned through a "lottery ticket mechanism".
An analysis of the gradient flow reveals that frequencies compete within each neuron during training.
The winning frequency that ultimately dominates is predictably determined by its initial magnitude and phase misalignment.
Finally, we use these insights to demystify grokking, characterizing it as a three-stage process involving memorization followed by two generalization phases driven by feature sparsification.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper provides a comprehensive mechanistic and theoretical explanation of how two-layer neural networks learn to perform modular addition and how this process explains grokking.  
Using both empirical analysis and formal dynamical proofs, the authors show that during training, each neuron converges to a single-frequency Fourier feature of the form

$$
\theta_m[j] = \alpha_m \cos(\omega_{\phi(m)} j + \phi_m), \quad
\xi_m[j] = \beta_m \cos(\omega_{\phi(m)} j + \psi_m),
$$

where the output phase satisfies a phase-alignment relation

$$
\psi_m \approx 2\phi_m.
$$

Across neurons, phases become uniformly distributed within each frequency group, forming a phase-symmetric ensemble.  
The network’s collective behavior can then be interpreted as a majority-voting Fourier circuit that robustly implements the indicator function

$$
1[(x+y)\bmod p = j].
$$

On the dynamics side, the authors identify a “lottery ticket” mechanism in Fourier space:  
different frequencies compete within each neuron, and the one with the largest initial magnitude and smallest phase misalignment wins.  
This explains the emergence of single-frequency neurons and provides a predictive theory of feature selection.  

Finally, the paper analyzes grokking as a three-stage process:
1. Memorization phase dominated by loss minimization  
2. First generalization phase where weight decay sparsifies frequencies and sharpens alignment  
3. Second generalization phase where weight decay refines the clean Fourier solution  

Together, these results offer a unified, end-to-end account of how gradient descent discovers structured Fourier representations and transitions from memorization to generalization.

### Strengths
- Comprehensive mechanistic theory.  
  This is the most complete explanation so far of modular-addition learning and grokking in shallow networks, connecting empirical phenomena (phase alignment, sparsification) to provable training dynamics.

- Elegant empirical–theoretical correspondence.  
  Observations 1–6 are each mirrored by formal results (Proposition 4.2, Theorem 5.2, 5.3). The analytical use of quadratic activations is a well-justified simplification that retains the core dynamics observed with ReLU.

- Novel “Fourier lottery ticket” insight.  
  The finding that feature emergence is governed by initial magnitude and phase misalignment provides a simple, predictive explanation of why single-frequency neurons reliably appear.

- Interpretability of grokking.  
  The proposed three-stage timeline, validated with metrics like phase alignment and frequency sparsity, makes grokking a measurable and interpretable process.

- Clarity and rigor.  
  The exposition is unusually clear for such a technical topic, combining intuitive figures with formal statements.

### Weaknesses
- Restricted architectural scope.  
  All analyses use two-layer MLPs with one-hot or learned embeddings; while ideal for interpretability, it remains unclear whether the same Fourier-alignment and frequency-competition mechanisms appear in deeper or attention-based models.

- Empirical validation of the voting mechanism.  
  Proposition 4.2 predicts that uniform phase diversity cancels noise via majority voting; an ablation that breaks phase uniformity could directly confirm this mechanism.

- Connection to pretrained LLMs not explored.  
  The paper convincingly explains why Fourier features emerge in modular tasks, but stops short of relating this to real LLMs—where similar sinusoidal and Fourier-like number encodings have already been observed.

### Questions
1. Connection to Fourier features in large models.  
   Recent work (arXiv:2502.09741) shows that when numbers are initialized with Fourier features, large pretrained LLMs can learn addition almost instantly, and that existing LLM embeddings already exhibit Fourier structure.  
   Can the authors interpret this observation through their phase-alignment dynamics—i.e., are Transformers implicitly performing the same “frequency lottery” at scale?

2. Scaling of grokking time.  
   Theoretical results (Theorem 5.3) relate convergence time to initialization scale 
   $
   \kappa_{\text{init}}
   $
   and modulus 
   $
   p.
   $
   Can this be turned into a quantitative scaling law predicting grokking delay?  
   Moreover, can this framework explain the observation from arXiv:2502.09741 that when the magnitude is large for some Fourier component, the model skips the grokking phase and learns addition with less data?

3. Quantifying required neurons.  
   Can the authors estimate the minimum number of neurons 
   $
   M
   $
   required to achieve 
   $
   100\\%
   $
   test accuracy as a function of 
   $
   p?
   $
  

4. Beyond addition.  
   Would similar Fourier competition and alignment appear in modular multiplication or other? Extending analysis to those tasks could generalize the theory beyond addition.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper analyses the phenomenon of delayed generalisation (grokking) on the modulo arithmetic task mod(a+b)23. The work corroborates other findings that, during the grokking process, the two-layer MLP learns Fourier transforms to complete the solution. The paper suggests that grokking occurs in three distinct phases: Memorisation, where the model learns 'common data', then Generalisation Phase 1, where the model begins to minimise loss on 'rare examples' and then finally Generalisation Phase 2, where increased generalisation is governed only by the weight decay term.  Through theoretical and empirical analysis, the authors argue that it is possible to predict the final frequency domain of neurons based on their initial parameterisation via an analysis of Fourier components aligning with their observation of the Lottery Ticket Mechanism. The authors also provide a mechanistic insight into the trained model and identify a majority voting mechanism that cancels out noise and facilitates generalisation. Finally, there is an analysis of how gradient-based training facilitates the representation of features from a training dynamics perspective.

### Strengths
1. The paper adds interesting observations about the grokking phenomena in the context of the mod(a+b)23 task, which provide a nice insight into grokking on this dataset.  
2. The combination of extensive theoretical results and supporting empirical results strengthens the paper's findings; however, some of the observations provided by the paper are corroborations of previous findings rather than novel insights (see weaknesses below). 
3. The notion of the majority voting scheme is an interesting insight; it would be nice to see if, in other grokking tasks, such a dynamic is used to improve generalisation. 
4. Predicting the final frequency of a neuron from its initialisation via magnitude and phase misalignment is a neat finding; it would be good to see these findings extended to other modulo arithmetic tasks to demonstrate their generality.

### Weaknesses
1. **Lottery Ticket Mechanism**:  In the paper, observation 6 is positioned as a novel observation; however, prior work, namely [1], [2] explicitly mentions the role of internal structure at initialisation, via the Lottery Ticket Hypothesis (LTH) [3], being a primary factor in grokking. Furthermore, [2] even goes on to show that particular 'grokking tickets' reduce the time for generalisation to occur. I think that the 'Lottery Ticket Mechanism' you observe should be positioned as corroborating other findings in the literature, rather than being a novel insight of this paper's analysis. 

2. **Fourier Features** Could the authors describe why/if they believe this perspective to be novel? Given that previous literature describing the dynamics of grokking via mechanistic interpretability [1] has shown that neural networks leverage discrete Fourier transforms and trigonometric identities to perform the addition necessary in modular arithmetic tasks. 

3. **Narrowness of analysis on mod(a+b) 23 Task**: The grokking task that is analysed in this paper is somewhat non-standard compared to other grokking studies [1] [2], for example, the original paper [4]  that introduces grokking conducts experiments on the mod(a+b)97 task. Is there a rationale behind examining mod(a+b)23? In the mod(a+b)23 case, as shown by Figure 3, there is a slight delay in generalisation; however, this is not as extreme as in the mod(a+b)97 case. Can you show that part of your analysis holds in other modular arithmetic tasks, or are these findings limited to this particular dataset?

4. **Importance of Weight Decay in Grokking**: The three phases of grokking suggested in this paper (Memorisation, Generalisation 1 and Generalisation 2) place a large weighting on the weight decay term in enabling generalisation. While [5] does support this narrative, more recent literature [6] has shown that weight decay is not a causal factor in generalising on algorithmic tasks, as they can mitigate or induce grokking entirely without modifying weight decay.  Additionally, [7] has also shown that grokking can be mitigated and that this mitigation is not solely reliant on the weight decay factor.  In light of these existing results, do you feel that you phases of grokking adequately describe the dynamics of grokking, or that they rely on factors that may correlate with generalisation, but are not causal factors for it?

5. **Common vs Rare Training Examples**: The statements regarding common vs rare training examples resemble conjecture; the idea of common examples where 'symmetric pairs' are memorised is not fully quantified in the paper, nor are 'rare' examples properly explained. To empirically justify statements about 'common' and 'rare' training examples, the authors should conduct an ablation study where the model is trained only on so-called 'rare' examples. This should, under the arguments in the paper, eliminate the memorisation phase and reduce the time to generalise on the test data.

6. **Even-order polynomials and Activation Swapping**: In Appendix Table 1, the activation swapping is shown replacing the RelU function with polynomials, preserving accuracy when the exponent is even and losing accuracy when the exponent is odd. However, there is no even representation of even and odd exponents in this table. Can you add the results for the exponents 5, 7 and 9 to represent odd exponents fully? 

7. **Prediction of final frequency**: The paper makes the bold statement that a neuron's final frequency can be predicted entirely from its initial magnitude and frequency alignment. Can the authors please provide an empirical analysis of how many neuron frequencies at the end of training are correctly predicted from this evaluation? 

8x. **Lack of clear takeaways**: The paper offers many observations of grokking dynamics; however, it is unclear what the general takeaways of the work should be, given that some of the observations are not entirely novel. The authors should explicitly highlight how their work provides new understandings of grokking that can generalise outside of their specific experimental setup. The paper would benefit from focusing less on the quantity of observations and instead on the clarity of insights. Could the authors please provide a conclusion section to this effect? 

References:
[1] Nanda, N., Chan, L., Lieberum, T., Smith, J. and Steinhardt, J., 2023. Progress measures for grokking via mechanistic interpretability. arXiv preprint arXiv:2301.05217.

[2] Furuta, H., Minegishi, G., Iwasawa, Y. and Matsuo, Y., 2024. Towards empirical interpretation of internal circuits and properties in grokked transformers on modular polynomials. Transactions on Machine Learning Research.https://openreview.net/forum?id=MzSf70uXJO.  

[3] Frankle, J. and Carbin, M., 2018. The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635.

[4] Power, A., Burda, Y., Edwards, H., Babuschkin, I. and Misra, V., 2022. Grokking: Generalisation beyond overfitting on small algorithmic datasets. arXiv preprint arXiv:2201.02177.

[5] Liu, Z., Michaud, E.J. and Tegmark, M., 2022. Omnigrok: Grokking beyond algorithmic data. arXiv preprint arXiv:2210.01117.

[6] Kumar, T., Bordelon, B., Gershman, S.J. and Pehlevan, C., 2023, September. Grokking as the transition from lazy to rich training dynamics. In The twelfth international conference on learning representations.

[7] Mason-Williams, G. and Mason-Williams, I., Decomposed Learning: An Avenue for Mitigating Grokking. In ICML 2025 Workshop on Methods and Opportunities at Small Scale.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper explores a one-hot encoded modular addition task $(a+b) mod 23$ with a 2-layer fully connected neural network with a hidden size of 512. With this network, they explore 3 questions, encompassing mechanistic interpretability, training dynamics, and grokking. They explore these questions from the parameters perspective, empirically finding that a neuron's parameters form a trigonometric pattern, and identify a set of properties: phase-alignment, model symmetry, majority-voting scheme, and a lottery ticket mechanism.

### Strengths
The paper explores an interesting set of questions. 

Highlights potentially interesting findings. 

Work around activation functions is intriguing.

### Weaknesses
The paper introduces the conceptions `phase alignment, where a neuron’s output phase is twice its input phase, and phase symmetry, where phases are uniformly distributed among neurons sharing the same frequency.` however, before this introduction, the term `phase` is not concretely defined in this context, which makes this section hard to parse.  

The paper states on line `136` that `We begin with the most striking observation: a global trigonometric pattern in parameters that consistently emerges across all training runs with random initialization.` however, no empirical evidence is provided to support this claim.

On line `148` it states `In Figure 7b, we zoom in on the learned parameters of the first five neurons`; however Figure 7b looks at 3 Neurons; how are these neurons selected? In addition, how many neurons have this pattern? How often was this pattern observed over multiple runs? Line `150-151` goes on to state `The plots show that these parameters are well approximated by cosine curves, shifted by phases φm, ψm, and scaled by magnitudes αm, βm.` however, only 1.9% of the total neurons are represented in Figure 7a, and only 0.5859375% for Figure 7b. Although these neurons may be represented this way, how are the other neurons represented? This is especially important to show when the following text reads `this suggests that the trained neural network learns to solve modular addition by embedding a trigonometric structure into its parameters`. The paper then uses this finding to build the rest of the paper. In addition, I was unable to reproduce the main findings in the paper, i.e., that **all** the networks' input and output parameters formed a `trigonometric structure` and found neurons that did not form a `trigonometric structure` which suggests the `trigonometric structure` is not a requirement for the model to learn this task. 

The paper only explores $(a+b) mod 23$, which limits the generality of the findings. Additional modular tasks should be explored to further support the findings, such as subtraction, division, and multiplication. Given that the paper explores modular arithmetic in a non-standard setup, where the input data is one-hot encoded. It is unclear how the findings will generalize to larger networks, more practical problems, or networks that use embedding layers.

The paper ends abruptly with no concluding section and does not provide clear takeaways nor relates the findings back to the current literature.

### Questions
Please see weakness and more concretely: 

Why is only p=23 explored in a network with a hidden width of 512 explored? Do these results hold when using a range of $p$ values, i.e $29, 31, 37, 41,.., 83, 89, 97$ and hidden widths of $32, 64, 128, 256$? 

Can you explore  additional modular arithmetic tasks such as subtraction, division, and multiplication? This would help improve the generality of the findings. To also help substantiate claims around phase alignment,  model symmetry, and the lottery ticket mechanism can this be explored in the case of CIFAR 10 and CIFAR 100 [1].

A lot of the work is then based on the finding that the problem is solved by `embedding a trigonometric structure into its parameters`. Given the model is significantly overparameterized, can you explore and report what happens when the first (input) layer is frozen (completely random) during training (with all data)? 

[1] Krizhevsky, A. and Hinton, G., 2009. Learning multiple layers of features from tiny images.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies 1 hidden layer, one hot encoded networks and seeks to describe the learned solution mechanistically and explain how the training dynamics result in that solution being learned.

### Strengths
- This work addresses a question that the field currently considers to be of high importance: why do deep neural networks learn the features they learn on modular addition?

### Weaknesses
*I am concerned with the paper overclaiming its novelty, particularly with respect to their claimed mechanistic interpretation*. 

This paper claims multiple results are novel, but I know some were done by other published papers. Furthermore, some results claimed as novel are in disagreement with results from other published papers.  

Thus, there are significant issues with this paper: 

1. At least five prior works of high relevance aren't cited, which leads to 2. 

2. There are **multiple claims of novelty that aren't novel**, i.e. other published work has already achieved the result:
Claimed novelty 1. "While individual neurons produce noisy signals, the phase symmetry enables a majority-voting scheme that cancels out noise, allowing the network to robustly identify the correct sum."
Claimed novelty 2. "We prove that these properties allow the network to collectively approximate an indicator function on the correct logic for the modular addition task." 

Claimed novelties 1 and 2 are already known and aren't novel. They were detailed by [1], which provided a mathematical model for how networks get the correct answer, using the uniform phase assumption to cancel out noise (claimed novelty 1) to prove the correct logit becomes "dirac" (i.e. an indicator, novelty 2). I believe that Gromov's work on this topic (which is cited by this paper) also used random phase cancellation to prove the correct logit would be like a dirac indicator, but I am more familiar with [1] which I know did this.

Also, claims 1 and 2 disagree with empirical and theoretical results in [2], which shows that in 2-layer networks the logits are **not an indicator on the correct output logit**. [2] gives both empirical evidence and a theoretical proof that on average, O(log(n)) different frequencies are learned, and the size of the margins as a function of frequencies that are learned is O(log(n)). Thus, in multilayer networks, the margins are not an indicator.

3. a lack of scope (the authors only study 1 hidden layer networks, though this is unclear from a first reading of their paper, which claims they study 2 layer networks, which can't be the case due to it being already established and proven that networks with >= 2 hidden layers learn O(log(n)) frequencies to solve this task [2]. 1 layer networks learn *all* (p-1)/2 frequencies (Morwani et al.), while multi-layer networks have been observed to learn substantially fewer (Nanda et al., Chughtai et al., [2]), and the gradient dynamics explaining why this happens are considered an open problem. Only studying 1 hidden layer networks makes their results fail to generalize to multilayer networks, and can't explain the aforementioned open problem. 

4. False claims are made, for example, on line 99, the authors state either one hot encoded inputs or a trainable embedding matrix can be used, but switching from one hot encodings to a trainable embedding causes the network to learn O(log(n)) frequencies [2], and not the n-1/2 frequencies result of Morwani et al., were trainable embeddings to be used, observation 3 and definition 4.1 both become false.

Less significant, but still issues:

5. This paper claims a "full" mechanistic understanding of models trained on modular addition, and presents 6 empirical observations, but lacks convincing empirical evidence supporting the mechanistic interpretation and observations. Some of this empirical evidence is relegated to the appendix (but should be adjacent to the observations and in the main paper). The experiments remain unconvincing due to reasons like: experiments not over multiple random seeds, some experiments seem to require training on the entire dataset (what do the plots in the first section look like when using standard ML train test splits?), lack of quantitative statistical / causal testing supporting observations 1-6.

6. This paper spends a significant amount of space claiming the aforementioned "full" mechanistic understanding (claims 1 and 2). This space should instead be used to incorporate convincing experimental evidence supporting what's necessary for the main result of the paper: their claims about how training dynamics unfold. 

7. This work does not have a related work section, as it's located in the appendix. Furthermore, this work incorrectly claims a "complete" discussion of related work, while missing citations to at least 5 other works ([1,2,3,4,5]). The authors state: "A complete discussion on related works is deferred to A.2 due to space limit."

8. There is no limitations section or conclusion; the last section is titled "TRAINING DYNAMICS FOR FEATURE EMERGENCE".

In summary, this paper makes multiple claims of novelty where the result is already well known and established. Of particular note, is the claim of a "full" mechanistic understanding, but [1,3], and especially [2] all together provide a more complete understanding than what is presented in this paper. **I believe this paper is not ready for publication at this time and needs a rewrite and restructuring beyond the scope of what can occur during reviews.** 

That said, **there are open problems remaining related to the gradient training dynamics on modular addition**: if their claim for the training dynamics holds up robustly under quantitative analyses with depth, over many random seeds, then their work would be the first paper (to my knowledge) to resolve *how* networks learn the features on modular addition. [5] is an uncited paper that attempted to explain the gradient dynamics on modular addition and was (to my knowledge) only accepted at a workshop. It used lotka-volterra ODEs to attempt their arguments. 

A successful paper on the gradient dynamics is worthy of publication, without needing to claim novel mechanistic interpretations.

[1] Grokking modular polynomials https://arxiv.org/pdf/2406.03495

[2] Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks
 https://arxiv.org/abs/2505.18266

[3] Towards a unified and verified understanding of group-operation networks https://arxiv.org/abs/2410.07476

[4] Modular addition without black-boxes: Compressing explanations of MLPs that compute numerical integration https://arxiv.org/abs/2412.03773

[5] Survival of the Fittest Representation: A Case Study with Modular Addition https://openreview.net/forum?id=2WfiYQlZDa

### Questions
Q1. Is my understanding correct that you trained networks with one hot encoded inputs and 1 hidden layer, i.e. the same networks trained by Morwani et al.?

Q2: How is this different from the Survival of the fittest work, which also uses ODEs (Lotka Volterra) [5]?

### Soundness
1

### Presentation
1

### Contribution
2
