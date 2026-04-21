# Transformers Provably Learn Two-Mixture of Linear Classification via Gradient Flow

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Understanding how transformers learn and utilize hidden connections between tokens is crucial to understand the behavior of large language models.
To understand this mechanism, we consider the task of two-mixture of linear classification which possesses a hidden correspondence structure among tokens, and study the training dynamics of a symmetric two-headed transformer with ReLU neurons.
Motivated by the stage-wise learning phenomenon in our experiments, we design and theoretically analyze a three-stage training algorithm, which can effectively characterize the actual gradient descent dynamics when we simultaneously train the neuron weights and the softmax attention.
The first stage is a neuron learning stage, where the neurons align with the underlying signals. 
The second stage is a attention feature learning stage, where we analyze the feature learning process of how the attention learns to utilize the relationship between the tokens to solve certain hard samples.
In the meantime, the attention features evolve from a nearly non-separable state (at the initialization) to a well-separated state.
The third stage is a convergence stage, where the population loss is driven towards zero.
The key technique in our analysis of softmax attention is to identify a critical sub-system inside a large dynamical system and bound the growth of the non-linear sub-system by a linear system. 
Finally, we discuss the setting with more than two mixtures. 
We empirically show the difficulty of generalizing our analysis of the gradient flow dynamics to the case even when the number of mixtures equals three, although the transformer can still successfully learn such distribution. 
On the other hand, we show by construction that there exists a transformer that can solve mixture of linear classification given any arbitrary number of mixtures.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper analyzes training dynamics of a single attention layer two-headed transformer on a specific task. The task -- a two-mixture of linear classification -- possesses a hidden correspondence structure between the feature vectors in the sequence. Each sample contains an indicator vector that dictates, sign of which of the other feature vectors in the sequence is important. Thus, to efficiently solve the task, a communication between the class indicator and the rest of the feature vectors is necessary. The authors first observe the training dynamics empirically, specifying three distinct training phases. Based on these observation they define a training algorithm similar to standard full-batch gradient flow, for which they prove that the transformer not only learns the task with perfect accuracy, but, more importantly, learns to effectively utilize the hidden structure in a human interpretable way. Finally, the authors discuss the extension to more than just two-mixture of linear classification, constructing a transformer architecture that solves the task, but showing that a gradient-based analysis would need to utilize more refined proof techniques.

### Strengths
What I particularly like about the paper is the very well chosen combination of data distribution, architecture and proof assumptions. This combination is perhaps a minimal example on which it is possible to demonstrate this phenomenon of transformers learning to utilize the hidden communication between tokens at this level of complexity. On the other hand, this setting is perhaps a maximal example under which the proof still remains practically doable and comprehensible. Therefore I find the setting of the paper very suitable. 

Unlike many other papers studying transformers, the advantage of this work is that the studied task and the provided results are simple enough that they provide human-interpretable understanding of the transformers' behavior, that agrees with our expectations -- the attention layers are crucial to provide the necessary communication between tokens and they can learn to do so. Further, the importance of the use of multiple heads for certain tasks is instructively demonstrated.

The authors also do a very good job extracting the essence of otherwise long and tedious proof into the main body and make it understandable for a wide audience. I think the presentation improves reader's own understanding of the inner workings of transformers and allows the reader to re-use some of the techniques in their own transformer research. The presented analysis and the core proof idea makes sense and nicely captures the important aspects of the optimization.

### Weaknesses
- W1: While I appreciate the setting of the paper in general, there are a few questionable decisions when it comes to architecture. In particular, in the main part of the paper, it is a bit unusual that the value matrix basically maps to 1D outputs and the representations are averaged even before applying ReLU. I would expect averaging only after the application of the ReLU and also like to see what would happen if more than 1D value matrices would be used. However, bigger issue comes with Lemma 6.1. In particular, the order in which the operations are made in the transformer defined in the proof is so unusual I am not sure it can be interpreted as transformer anymore. The summation over $l_2$ should definitely come as the inner-most, otherwise we cannot even talk about attention. Then should come the summation over $k$, only after that should come the application of $\sigma$ and the summation over $l_1$ should be the outer-most. I would ask the authors to redo the statement and proof of Lemma 6.1 so that the summation over $l_2$ is the inner-most or otherwise justify their architectural choice. 

- W2: Although the presented phenomenon is intuitive, it is unclear *if* and *how* does it generalize to more general settings. In particular to multi-mixture of linear regression, where I would expect (almost) the same thing happen. It is unclear why the symmetry is broken and therefore it is also unclear whether the described phenomenon even holds the same way in multi-class setting. The same question holds for another natural extensions of the considered setting -- multi-class classification, presence of noise, non-orthogonality of features or class imbalance and more natural weight initialization. I wouldn't mind that the paper only studies the simplest setting if there was evidence that the knowledge obtained can be transferred to the more general settings but in this case it is not clear whether the same mechanism even works in extended settings.

- W3: Some aspects of the presentation should be improved. To provide a couple of examples: 

- *It could be clearer if authors re-named classes to something else as they are confused with the classification classes.*
- *The assumption of averaged initial value model is not discussed enough. In particular it is not clear where any asymmetry at all comes from if we have this assumption.*
- *The proof in appendix should be more verbose and make some more explicit statements as intermediate steps. For instance, the separability of the consistent samples after phase 2 is only proved implicitly and is never explicitly stated in the proof. This makes it hard to make look-ups of specific proof parts mentioned in the main body, because the proof itself is an ocean of infinite formulas.*
- *The use of $g$ is somewhat peculiar. I suspect the authors to either make a few typos when using $g$ or not to define it properly.*

**Typos:**

- The property 4 in Definition 2.1 is inconsistent with the verbal description in the rest of the paper as this way, you would leave most of the positions actually empty (zeros).

- line 170: You should write $\nu$ instead of $X^{(i)}$.

- In lemma 5.2, the use of $g$ is somewhat weird. Please double-check what you meant there. The same holds for lines 427, 430

- Line 408: based on my counting the numbers should be 16, 10, 10. Correct me if I am wrong. 
 
- In line 60 you talk about class identifier getting all the signal from classification features, but later in your theory you consider the class identifier's role as the key to be crucial. I think it better to rewrite this sentence so that the roles are exchanged.

### Questions
- Q1: Have you tried to measure this phenomenon and the training dynamics for any of the setting extensions I mentioned in W2? 

- Q2: In Figure 1, we see that the neuron alignment grows with $c_k$ immediately from the beginning. This seems to be in contradiction with line 367, especially since I believe the weight initialization in experiments was the same as in the theory. What makes this difference? 

- Q3: In figure 1c, how would the curves for $c_1$ being query look like? Can you provide a figure with these included? 

- Q4: Where do the slight asymmetries in Theorem 4.1 come from when you consider the averaged initial value model? 

**Summary:** I consider this paper to be a very instructive, novel, theoretical contribution. It allows us to understand transformer training dynamics in a human-interpretable way that aligns with our natural understanding of what attention should be learning to do. This highly precise description of the role of attention in simple transformers makes the paper a strong contribution. Therefore, I recommend accepting the paper. However, my high score is conditioned on the authors addressing my concerns in the discussion period.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper analyzes training dynamics of transformer estimated via gradient flow under the mixture of linear classification setting. Based on the empirical findings, they construct their own three-stage training procedures, which essentially give the same training dynamics with actual gradient descent simultaneously train all the parameters in Transformers.

### Strengths
The paper provides rigorous theoretical analysis on the training dynamics of transformer which is a highly non-trivial task.

### Weaknesses
Please see the below questions:

### Questions
A.	I had trouble understanding the setting of mixture of linear classification. Authors only spend one paragraph in the 2nd page discussing about the model. But I think it should be elaborated more in details to motivate the readers. 

B.	I might have missed the part while reading the paper, but I wonder how training dynamics discovered in the paper corresponds to the actual reading procedures of the human beings? For instance, using the example in the paper: “Bob is watching TV in the living room. Where is Bob?” How do people catch the correspondence between Where and Living room and how does this correspond to the training dynamics in the paper? I wonder if authors have any thoughts on this. 

C.	I think this is important question: is the training dynamics in the paper also observable in the vanilla transformer case? The authors consider two-headed symmetric transformers, but in real practice vanilla transformer is commonly used. If this is the case, I think this paper’s contribution is significant. 

D.	Do authors have any insights on the multiple layer cases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the training dynamics of a two-headed transformer using a three-stage, layer-wise training algorithm. It further characterizes the feature learning process and captures the attention mechanism involved in solving the task.

### Strengths
This paper presents a rigorous theoretical framework for understanding the training dynamics of non-linear Transformer architectures.

The most interesting finding, in my view, is how the authors analyze the progressive emergence of connections across diverse data types at various stages, offering a comprehensive perspective on the layered learning process within Transformers. Specifically, the study demonstrates that weights in distinct components of the model are updated in different stages, indicating an organized, sequential approach to learning. This stage-wise learning strategy introduces a novel perspective on the interplay between weight adaptation and data distribution, highlighting how the architecture distinguishes and captures distinct data characteristics over time. Additionally, the paper provides a theoretical characterization to substantiate its findings, anchoring the analysis in formal proofs.

### Weaknesses
(1) The description of the experimental setup is missing, but I assume the results are all based on synthetic data. Therefore, justification with real data is necessary.


(2) The theoretical insights for guiding the training and deployment of Transformers are unclear in practical applications.

### Questions
N/A

### Soundness
2

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
3

### Summary
This paper investigates the dynamics of transformers in learning linear classification tasks where input tokens have hidden correspondence structures. The authors examine a two-headed transformer with ReLU neurons on a specific task involving two classes with hidden linear relationships. They observe a three-stage learning process: (1) neuron alignment with classification signals (2) (2) attention feature learning where features evolve into a separable state and (3) training the neurons again to the final convergence. Baaed on the empirical observation, they propose a three-stage training algorithm and derive the main theorem based on the algorithm. 
Their theoretical analysis, using gradient flow dynamics, shows how attention mechanisms enable the model to leverage hidden correspondences to solve classification tasks effectively. The study also considers settings with more than two mixtures, highlighting challenges in analysis yet demonstrating the model's empirical success.

### Strengths
**Theoretical Contribution**: The paper provides a rigorous theoretical analysis of transformers’ training dynamics on structured linear classification tasks, specifically using gradient flow dynamics. This theoretical foundation contributes to a deeper understanding of how attention mechanisms in transformers learn hidden relationships, a topic that is not yet fully understood in the community.

**Algorithmic Insights**: The proposed three-stage training algorithm mirrors the empirical learning dynamics, providing a potentially practical framework for initializing and training transformers on tasks with similar hidden structures. This approach could be valuable for applications requiring attention to specific feature-alignment steps.

**Novel Analysis of Attention Dynamics**: The paper effectively addresses the softmax attention mechanism’s behavior within the training dynamics. The authors introduce techniques to simplify and analyze the system, such as reducing the attention's complex nonlinear dynamics to a manageable subsystem, which may serve as a foundation for future analyses in related settings.

### Weaknesses
- This paper only consider learning two-mixture of binary classification, and the data distribution is of size $d\times 3$. The main theorem is based on Algorithm 1, an algorithm designed from empirical observations on a synthetic dataset but not tested on any real-world datasets. So, I question whether analyzing Algorithm 1 and this specific choice of dataset is really helpful. I suggest the authors to apply Algorithm 1 on some real-world dataset, and shows that it works. (do not need to be better than standard training). 

- Although this paper's title is "two-mixture", there should be more experiments and discussions on $K\geq 3$ than current Appendix A.3.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
2
