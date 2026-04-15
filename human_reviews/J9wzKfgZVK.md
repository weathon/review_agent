# What and How does In-Context Learning Learn? Bayesian Model Averaging, Parameterization, and Generalization

- Decision: Reject
- Scores: 5, 5, 6

## Abstract
In this paper, we conduct a comprehensive study of In-Context Learning (ICL) by addressing several open questions: (a) What type of ICL estimator is learned within language models? (b) What are suitable performance metrics to evaluate ICL accurately and what are the error rates? (c) How does the transformer architecture enable ICL? To answer (a), we take a Bayesian view and demonstrate that ICL implicitly implements the Bayesian model averaging algorithm. This Bayesian model averaging algorithm is proven to be approximately parameterized by the attention mechanism. For (b), we analyze the ICL performance from an online learning perspective and establish a regret bound $\mathcal{O}(1/T)$, where $T$ is the ICL input sequence length. To address (c), in addition to the encoded Bayesian model averaging algorithm in attention, we show that during pertaining, the total variation distance between the learned model and the nominal model is bounded by a sum of an approximation error and a generalization error of $\tilde{\mathcal{O}}(1/\sqrt{N_{\mathrm{p}}T_{\mathrm{p}}})$, where $N_{\mathrm{p}}$ and $T_{\mathrm{p}}$ are the number of token sequences and the length of each sequence in pretraining, respectively. Our results provide a unified understanding of the transformer and its ICL ability with bounds on ICL regret, approximation, and generalization, which deepens our knowledge of these essential aspects of modern language models

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper conducts a theoretical investigation into in-context learning (ICL) and addresses three key questions: the choice of ICL estimator, the performance metric for ICL along with its associated error rate, and the role of the transformer architecture in ICL.

### Strengths
1.	The paper provides a more general theoretical model on the response and hidden variables compared with previous works in this direction.
2.	The authors propose a new regret metric to quantify the performance of ICL and develops a $\mathcal{O}(1/T)$ regret rate for this metric. 
3.	Besides, the authors analyze the error incurred during the pretraining phase, and the error decays exponentially in terms of the depth of the network.

### Weaknesses
1.	As this paper primarily focuses on theoretical aspects, it is encouraged to enhance the clarity of the presented proofs. Given that most of the proofs are intense, providing a big picture to roughly explain and outline how the proof works is crucial for readers' comprehension. Otherwise, readers can easily get confused. For instance, in the proof of Theorem 5.3, it would be beneficial to further explain why we should pick the distributions $P$ and $Q$ in this way when we bound the terms in error decomposition. Similarly, in the proof of Proposition 5.4, why should we first construct an approximator for $g^*$, and then $\rho^*$, $\phi^*$? These will be clearer if a big picture is provided.

2.	Following the above point, the authors should explain the reasons from one step to the next step more concretely. For instance, in the proof of Proposition 5.4, the assumptions 5.2 and F.3 do not appear. If there are some steps obtained with these assumptions implicitly, it is encouraged to point it out explicitly. There may be more areas throughout the paper where such improvements in clarity are needed, and the authors are encouraged to address them to enhance overall presentation.

3.	With all these bounds obtained, it is recommended to validate them empirically, even if through synthetic experiments. Such empirical verification would not only enhance the credibility of the theoretical results but also provide an additional perspective on their soundness.

### Questions
1.	According to the definition of the assumed model on $r_t$, is it not related to the true hidden concept? Is this a common assumption on realistic data? 
2.	Is there any reason that the authors consider the regret in the form of average instead of the usual accumulative regret? Under this new regret, the rate is $\mathcal{O}(1/T)$. Does that mean in terms of the usual accumulative regret, the rate is constant? How do the authors interpret this?
3.	In the proof of Proposition 4.1, how do the authors obtain the first and the third equalities formally? For the current first equality, does it mean $h_{t+1}$ is independent of $c_{t+1}$? Even though the authors mention these equalities come from model (4.2) (by the way, should it be model (4.1) instead?), more concrete explanations are expected.  
4.	Regarding the proof of Corollary 4.2, In the first formula, why $P(S_t|z)$ only relates to responses $r_{i}$? In the third formula, should the most outer sum be $\sum_{t=0}^{T}$, or $\sum_{t=1}^{T}$? If the second equality is derived through telescoping, it is not immediately clear why only one term remains. Also, how is the last equality obtained? 
5.	Regarding the proof of Theorem 5.3, in the last inequality of (F.2), do we have an expectation over $S_t^n$ for the first term? According to lemma I.5, there is an expectation over $X\sim\nu$ for the squared TV. Also, the authors note the bound of $P_{\hat{\theta}}(x|S) $ in (F.6), but how is this obtained? How is the TV bound located between (F.6) and (F.7) satisfied? Why $ 2\epsilon/b_y$ is in $\mathcal{O}(\frac{1}{NT})$? 
6.	In the proof of Proposition 5.4, the authors obtain the bound for the difference in L1 norm to be in terms of $B_{A,1}^{D''}$, which is an exponential growth term, but how does this later become exponentially decay in terms of D? What does it mean for the sentence “We take that D’=… in Lemma I.9”? Do we let D’ and D’’ be these quantities or C in D’ and D’’ be these quantities? How is the TV bound in step 3 satisfied? Please clean up the proof to make it clearer.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper provides a theoretical analysis of in-context learning (ICL) in transformers used in large language models (LLMs), framing the attention mechanism as a Bayesian Model Averaging (BMA) process. It proposes that LLMs use attention weights to perform BMA, contributing to our understanding of model generalization from few-shot examples. The authors develop a series of propositions and theorems that articulate the role of the attention mechanism in facilitating ICL. Specifically, they demonstrate how the attention weights can be seen as performing a type of BMA. Furthermore, the paper extends the theoretical framework to provide regret bounds for ICL, which quantify how well the learning model performs in comparison to the best possible model chosen in hindsight. It also offers an in-depth error analysis that establishes bounds on the error rates of pretrained models. This work aims to deepen the theoretical foundations of ICL, offering a framework that connects with practical machine-learning challenges.

-----

Post rebuttal: Raising to 5 in light of authors' detailed evaluations. It would be good to see those results incorporated in the next version of the paper.

### Strengths
By adopting a Bayesian perspective and formulating ICL as a prediction problem based on examples from a latent variable model, the paper unifies the understanding of transformer architecture's ability to enable ICL.

Propositions and theorems are well-formulated, providing substantial theoretical contributions to the field, such as establishing regret bounds for ICL and demonstrating the parameterization of Bayesian Model Averaging through attention mechanisms.

It includes a fine-grained analysis of pretraining under realistic assumptions, offering a bound on the error of pretrained models.

### Weaknesses
The paper lacks empirical validation for its theoretical claims. This is a significant gap, as the practical impact of the theoretical insights on real-world tasks remains unclear without empirical evidence.

Some of the assumptions made for the theoretical analysis are not justified with respect to their applicability in practical scenarios. The paper does not sufficiently discuss how these assumptions can be relaxed without losing the theoretical results.

The complexity of the mathematical content could limit the paper's accessibility to a broader audience. Additionally, there is a lack of clear guidance on the reproducibility of the theoretical results, which is critical for the advancement of the field.

While the paper attempts to address important questions, its insights as compared to the existing works seem very limited at this point.

### Questions
- How can the theoretical findings be empirically validated in the absence of practical experiments within the paper?

- Is there a possibility to simplify the presentation to make the paper more accessible to non-experts?

- Could the authors clarify the process of deriving the posterior distributions within the context of ICL and how they relate to the attention weights?

- Can the authors provide more details on the assumptions made for the error analysis and how these assumptions can be justified from a practical standpoint?

- Are there specific mathematical models or theories that could challenge or complement the authors' approach to understanding in-context learning?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper provides a study of in-context learning (ICL) through a Bayesian view and addresses related questions. They first show that large language models perform ICL by adopting a Bayesian model averaging (BMA) algorithm during inference by showing that a prediction for a prompt from a perfectly pretrained model equals the prediction of BMA. Then, to measure the ICL performance, the authors consider ICL regret as a performance metric that measures the difference from prediction to the best hidden concept. The regret is shown to converge with increasingly more examples. To show the connection between the transformer network and the BMA algorithm, the authors define a variant of the attention layers, which then show that the modified attention mechanism encodes the BMA algorithm for the Gaussian linear model. For the sequence length going to infinity, the provided attention variant coincides with the softmax attention.
Additionally, the authors provide the pretraining analysis that can measure the pretraining error with approximation and generalization errors. Under certain smoothness conditions, approximation error converges with bigger models. Finally, for an imperfect pretrained model, the authors can successfully upper-bound the ICL regret.

### Strengths
+ The progression and steps make sense.

+ The theoretical results seem to be new and are a nice addition to ICL setting.

+ The theoretical contribution seems to be a generalization for some ICL settings assumed in previous works.

### Weaknesses
- How big is the ICL regret in practice? It is hard to tell how tight the bound is in practice. How can it be measured for LLMs?
Since there is no application of the results, it is hard to tell how far the bound is really in practice. 

- How can we in practice measure the validation performance of ICL?

### Questions
- What is sequentially mean? Are we not really doing sequential but all at once prompting, which is different from iterative prompting?

- What is the "wrong" input-ouput mapping? There is no real wrong or right here. How can you get $z_*$?

- Is KL divergence the best to measure distribution gap, which are key in measuring the generalization gap? What is the embedding space it has to use? Is there a better divergence metrics to measure that?

- Even though the progression and paper is smooth, it would be nice to add visualization to main paper rather than in Appendix.

- The analysis only involves pretraining in relation to the ICL performance. However, large language models also have fine-tuning distribution involved as an "intermediate" between the two aforementioned stages. Should that introduce some +/- gaps in divergence?

- Where do we get the hidden concept in the LLM?

**Minor:**

+ Why LLMs the ability for ICL (page1)

+ Hiddn Markov Model (page 3)

+ Variables are undefined (or defined later than introduced):
  + N and d, d_k,  d_v 
  + What is W?
  + Large T and small t difference?
+ Pertaining distribution (page 7)
+ xIn section (page 9)
+ correctly inference (page 9)

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent
