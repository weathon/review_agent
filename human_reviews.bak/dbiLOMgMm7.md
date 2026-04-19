# Early learning of the optimal constant solution in neural networks and humans

- Decision: Reject
- Scores: 5, 8, 6, 6

## Abstract
Deep neural networks learn increasingly complex functions over the course of training. Here, we show both empirically and theoretically that learning of the target function is preceded by an early phase in which networks learn the optimal constant solution (OCS) – that is, initial model responses mirror the distribution of target labels, while entirely ignoring information provided in the input. Using a hierarchical category learning task, we derive exact solutions for learning dynamics in deep linear networks trained with bias terms. Even when initialized to zero, this simple architectural feature induces substantial changes in early dynamics.  We identify hallmarks of this early OCS phase and illustrate how these signatures are observed in deep linear networks and larger, more complex (and nonlinear) convolutional neural networks solving a hierarchical learning task based on MNIST and CIFAR10.  We explain these observations by proving that deep linear networks necessarily learn the OCS during early learning. To further probe the generality of our results, we train human learners over the course of three days on a structurally equivalent learning task. We then identify qualitative signatures of this early OCS phase in terms of true negative rates. Surprisingly, we find the same early reliance on the OCS in the behaviour of human learners. Finally, we show that learning of the OCS can emerge even in the absence of bias terms and is equivalently driven by generic correlations in the input data. Overall, our work suggests the OCS as a common learning principle in supervised, error-corrective learning, and suggests possible factors for its prevalence.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes that the early-time learning of a constant solution that is input-independent and a pure function of the label distribution is a universal property of learning dynamics in neural networks and humans. It proves that linear networks with biases learn this early-time OCS and slightly more complicated and nonlinear networks exhibit qualitatively similar learning curves. Such learning dynamics are characterized by an initial dip in TNR for top levels of the output hierarchy in a hierarchical learning task, and such learning curves are also present when humans are trained and tested on such a task. Altogether, the paper seeks to provide theoretical and empirical evidence that early-time mimicry of the label distribution may be a general property of learning systems seeking to minimize error on a task.

### Strengths
A major strength of the paper is that it is original, and makes claims about a broad range of phenomena. Generally, I think there isn't enough work studying how learning systems -- both artificial and natural -- exhibit qualitatively similar behaviors in various learning regimes. This paper studies linear networks (with and without bias), CNNs, and humans, and its contributions span from machine learning theory of linear networks to human psychological experiments, which is a breadth that is rare as well as nice to see. If it is true that the early time learning of the OCS is shared between a broad range of artificial networks and robustly replicated across humans, this would be an interesting and significant result. The inclusion of human data that qualitatively supports the hypothesis is a major strength.

### Weaknesses
The paper is quite confusing in terms of how it is laid out, and what the scope of which claims are. In my opinion, it tries do a bit too much. For instance, tensor trickery related to the NTK feels out of place/a low order concern when you didn't even take the time to define the hierarchical learning task clearly. I had to look through the Appendix as well as (Saxe, 2019) to get a sense of the task. Some specific weaknesses: 
- It's not clear the scope of the claims being made. Either you are 1) remarking that the learning curves between linear, nonlinear networks, and humans on a certain class of task look similar but the mechanistic reasons are not well understood, and this is interesting and warrants further study or 2) making a claim that ALL neural networks are marked by this early-time behavior, and they are driven by mechanisms you identify. I would be more sympathetic to the contributions of this paper if you were making claim (1), but since you use the words "universal" and "mechanistic" in your abstract, I take it you are making claim (2). If this is the main claim, then the evidence presented is inadequate as justification. This is for the reasons that follow.
- You spend much of the paper talking about linear networks with biases, but in Section 6 make the (extremely strong) claim that such reversion can *equivalently* arise from input correlations. If it is indeed true that very general covariate structure can give rise to this behavior, then why have you spent most of the paper reasoning about the contrived and simple setting of linear networks with bias? Phrases like "might hence be ubiquitous beyond these datasets" arouse suspicion. Also, how can I believe that early-time learning of OCS is "universal" when it vanishes in even linear networks if you ablate the biases?
- I put very little weight on experiments that work with MNIST and CIFAR. One can conjure up evidence for any hypothesis by studying a suitable version of such datasets with some contrived architecture. They are simple tasks that might have been useful 10 years ago but certainly do not convey any information in 2024, in my opinion. I think this paper is at its strongest when discussing linear networks where claims are made precise, or human studies where the empirical trends found in these (interesting and novel!) experiments can be unpacked. I think trying to convince a reader that this is a universal property by showing toy experiments "in the middle" of linear/humans (eg. CNNs) is not a good strategy, especially since by the end I am not even convinced the mechanism for this phenomenon is the same across even just linear networks and humans. Spending some time providing evidence for this latter claim and dropping any attempts at universality claims justified only by MNIST/CIFAR, would in my opinion be a good idea. 
- You remark that "characteristic learning signatures" arising from early time learning of the OCS involve a sort of step-wise learning structure. While I'm aware a variety of toy synthetic settings studied by theorists exhibit this behavior (see [7], for instance), no non-trivial data (ImageNet, C4) gives rise to dynamics that look like this (learning curves smoothly go to low loss rather than being step-wise). So certainly the word "universal" does not describe these types of learning curves that are "characteristic" of early-time learning of OCS. 

All in all, I find the presentation and scoping confusing and the claims (where I can understand them) unjustified. I laud the broadness and ambition of the study, but find it ultimately somewhat unconvincing in its present form. Originally, I gave this a rating of 3, but on reflection, I think the inclusion of human data that qualitatively exhibits early-time OCS is on its own interesting enough to bump that up to a 5. 

--------- 
[1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli. "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks." arXiv preprint arXiv:1312.6120 (2013).

[2] Belrose, Nora, et al. "Neural Networks Learn Statistics of Increasing Complexity." arXiv preprint arXiv:2402.04362 (2024).

[3] Kang, Katie, et al. "Deep neural networks tend to extrapolate predictably." arXiv preprint arXiv:2310.00873 (2023).

[4] Bordelon, Blake, Abdulkadir Canatar, and Cengiz Pehlevan. "Spectrum dependent learning curves in kernel regression and wide neural networks." International Conference on Machine Learning. PMLR, 2020.

[5] Simon, James B., et al. "The eigenlearning framework: A conservation law perspective on kernel regression and wide neural networks." arXiv preprint arXiv:2110.03922 (2021).

[6] Bordelon, Blake, Alexander Atanasov, and Cengiz Pehlevan. "How Feature Learning Can Improve Neural Scaling Laws." arXiv preprint arXiv:2409.17858 (2024).

[7] Abbe, Emmanuel, et al. "The staircase property: How hierarchical structure can guide deep learning." Advances in Neural Information Processing Systems 34 (2021): 26989-27002.

[8] Kumar, Tanishq, et al. "Grokking as the transition from lazy to rich training dynamics." arXiv preprint arXiv:2310.06110 (2023).

### Questions
- I do not understand the relationship between this work and a broad range of work [4, 5, 6] on spectral learning curves, where it is known that features are learned hierarchically in roughly the order of their power in the generative function. The original Saxe paper [1] demonstrates a similar result in a linear setting. How is the early-time learning of a constant solution related to this learning of eigenmodes in order of decreasing power? Is this a zeroth order term in some sort of a Harmonic expansion of the target function? I would like a clear and detailed mathematical exposition of this fact, either in the author response, or an update to the manuscript. I know, for instance, sometimes networks begin learning by attempting a kernel solution before transitioning to feature learning ("grokking", see [8]) -- viewed this way, are such settings examples of learning the "optimal linear solution" at early time?
- Are these results only true for hierarchical tasks? Again, the scope of where and when this is true is not clear. For example, the MNIST experiment (effect is weaker in Orth. MNIST) makes me think the results only hold in hierarchical settings (again, challenging "universality"). 
- I don't quite understand the claim around "generic input correlations" -- as I see it, you show that a range of settings where input data (independent of labels) has low-dimensional structure exhibit learning of OCS. But I thought the OCS was about mimicking the structure of labels independent of input, not the other way around? What does $X^TX$ have to do with learning the optimal distribution over labels ($X$ is the input matrix)?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work employs an analytically tractable analysis of learning dynamics via linear networks, focusing on the bias towards the optimal constant solution (OCS) early in training. A 2-layer (and 1-layer) linear network with no bias terms is first studied, whose learning dynamics can be solved exactly (under a commutativity assumption). This analysis is then extended to include a bias term in the first layer, in the case of uncorrelated inputs, and it is shown that the same approach works in this case (since the commutativity condition remains satisfied). Empirical evidence for early convergence to the OCS is then provided, which does not occur for linear networks without bias terms, and a theoretical explanation is provided, showing that the OCS mode dominates the input-output covariance matrix when biases are present. This phenomena is then related to human learning, showing that humans also display a bias towards the OCS early in learning. Finally, it is demonstrated that early bias towards the OCS is not solely driven by bias terms, and can instead be induced by properties of the data via the input-input correlation matrix.

### Strengths
Though simplicity bias has been observed empirically and is thought to be important for generalization, its origin lacks a theoretical understanding. This paper is significant in providing a concrete theoretical explanation for a particular kind of simplicity bias (OCS learning)

Solves for parameter dynamics under a relaxed assumption (Proposition 1) compared to previous works (which typically assume uncorrelated inputs)

Demonstrates that linear networks can provide meaningful predictions of non-linear models, including CNNs

The relation to human learning is interesting and unique

### Weaknesses
It would be helpful to include a brief description of the hierarchical learning task in addition to Fig 1 for those unfamiliar with the setup

### Questions
Does the theoretical argument nicely extend to linear networks with more than two layers? In the multi-layer case, would a bias term present in an intermediate layer result in a bias towards OCS?

In what ways do the linear network solutions extend Saxe et al. (2014) precisely? Through the inclusion of bias terms, and a relaxed assumption (Proposition 1)?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper theoretically and empirically examines early learning of the optimal constant solution (OCS) in neural network training. Under a hierarchical label setting, the authors analyze the learning dynamics of linear networks with bias terms, demonstrating that OCS learning happens in the early training stages. Empirical results and human studies confirm that this phenomenon extends beyond linear networks to more complex nonlinear networks and human learners.

### Strengths
The paper introduces an intriguing exploration of simplicity biases of optimal constant solution in training neural networks, with potential relevance to human societal patterns. This work could interest both the machine learning community and researchers in other fields, with possible extensions to topics like fairness and stereotyping.

### Weaknesses
My major concern about the paper is about its clarity: 
* Plots are not explained well, e.g., what does "output activation top" represent in Figure 3?
* Even after reading Appendix A.4, I'm still a bit confused about the definition of TNR rate, in equation 7, the level index $k$ appears on the left side of the equation but not on the right, should the $y$ term in right hand side be $y_k$ to denote the target label for a specific level? Besides, this definition seems important and should potentially be moved to the main paper. 
* The hierarchical MNIST dataset is only explained in the Appendix, and it's not quite clear how the hierarchical label structure is established for the dataset.

### Questions
1. Does the results in the paper indicate that bias towards OCS solution could eventually vanish after long enough training?
2. Fig. 3 and Fig. 5 seem to reflect different outcomes, where Fig. 3 shows a initial score drop for all hierarchical levels, only the top level score drop is prominent in Fig. 5, what would be the cause of this? Besides, do TNR and $f^{tn}_k$ refer to the same metric? If so, it's better to maintain consistent notations. 
3. Line 897, "Given responses in y or in and target vectors", a typo?
4. Due to the hierarchical label design used in the paper, it seems well related to the domain of multi-label learning, I would suggest to discuss some related work in this domain.

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
4

### Summary
The paper shows that in the early phases of training neural networks learn the “optimal constant solution” (OCS), i.e., they mirror the statistics of the labels while being invariant to the input points. It considers deep linear networks with bias terms and derives the exact learning dynamics based on Saxe et al. (2014; 2019). It then proves that deep linear networks learn the OCS early in training. To support its findings, the paper performs experiments on both deep non-linear neural networks and human learners that display qualitative agreement with the theoretical analysis.

### Strengths
1. The study contributes to the broader understanding of simplicity biases in machine learning, specifically in the early learning phases.

2. The paper’s exact characterization of the learning dynamics of deep linear networks with bias terms is a novel contribution. This analysis is theoretically sound and provides new insights on how simplicity emerges during the initial training phase.

3. The theoretical findings are supported by learning experiments involving both artificial neural networks and human learners, bridging machine learning and neuroscience.

### Weaknesses
1. The most significant weakness of this work is the lack of a definition and clear description of the studied “hierarchical task”. While the authors describe it as a "hierarchical category learning task" with hierarchical levels, there’s no clear explanation of the precise structure or production rules, such as the nature of the relationships between input categories and output labels. Without knowledge of the distribution of $X$, $Y$, and precise definitions of the hierarchical levels, I find it practically impossible to fully understand the study, particularly the empirical aspects. The link between the model of data and the OCS is also not discussed.

2. The *continuous correct-rejection scores*, which are crucially used to infer OCS learning, are not defined within the main text.

3. The figures and the corresponding captions are poor and challenging to interpret. Figure 1 contains 6 panels that lack sufficient standalone explanations. In Figure 2, the reader does not know what different colors represent.

4. Proposition 3 must assume some initialization statistics for the weights that are not specified.

5. In general, the paper presentation is dense, often lacking detailed explanations or explicit definitions. It also frequently refers the reader to the appendix, detracting from readability. 

In summary, the paper has potential, offering some interesting insights into early learning in neural networks and connections to human cognition. However, I think it requires significant rewriting and, in particular, improvements in clarity, structure, presentation, and detailed explanation and discussion of the empirical results.

*Typos:*

(L253) The word “singular” is repeated twice.

### Questions
See comments above.

### Soundness
3

### Presentation
3

### Contribution
2
