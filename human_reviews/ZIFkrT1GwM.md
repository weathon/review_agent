# Pretrained Transformers are Deep Optimizers: Provable In-Context Learning for Deep Model Training

- Decision: Reject
- Scores: 6, 3, 6, 6

## Abstract
We investigate the transformer's capability for in-context learning (ICL) to simulate the training process of deep models. 
Our key contribution is providing a positive example of using a pretrained transformer to train a deep neural network by gradient descent in an implicit fashion via ICL. 
Specifically, we provide an explicit construction of a $(2N+4)L$-layer transformer capable of simulating $L$ gradient descent steps of an $N$-layer ReLU network through ICL.
We also give the theoretical guarantees for the approximation within any given error and the convergence of the ICL gradient descent.
Additionally, we extend our analysis to the more practical setting using Softmax-based transformers. 
We validate our findings on synthetic datasets for 3-layer, 4-layer, and 6-layer neural networks.
The results show that ICL performance matches that of direct training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
**Summary:** In this paper, the authors show that there exists a choice of weights, under which a Transformer can implement gradient descent on $f(w,\cdot)$ over a dataset $\\{x_i,y_i\\}_{i=1...n}$. The paper focuses on the setting where $f$ is the loss of a $N$-layer neural network, and $w$ is its parameters. The authors provide the explicit construction, along with error bounds on the approximation error to a real $L$-step gradient descent.

I lean towards recommending reject for this paper. The reason is that I am unclear about the purpose of the main result. I will elaborate below.

### Strengths
1. The construction is *mostly* explicit, and is clearly described in the various proofs. The proofs seem to be correct, though I have not carefully verified each line.
2. To simulate $L$ gradient descent steps of a $N$ layer neural network, one only needs a $(2N+4)L$-layer Transformer.
3. The recursive expression Lemma 1 seems to work well with a multi-layer Transformer.
4. Being able to learn a neural network in-context with a Transformer seems interesting.

### Weaknesses
Transformers have been shown to have high representation power. **If one does not care about the complexity of the construction,** [1] claims that the Transformer can implement arbitrary algorithms by literally simulating a computer; Lemma 14 of [1] claims that there is a 13 layer 1-head Transformer that implements T steps of gradient descent on a 2-layer sigmoid network.

In the other extreme, [2] (and numerous follow-up works) considered **simple problems**, but also provided **simple constructions** which are provably learnable during training. 

My main criticism of the results of this paper is that the results here falls somewhere in-between the two extremes above -- the target problem of "can Transformers implement gradient descent in-context for a neural network" is challenging, and thus required a rather **complex construction**. If so, why would I not simply use the general construction in [1]? 

The constructions in this paper are likely not being learned, in theory or in experiments, so what makes it superior to the generic construction in [1]? This is further compounded by the fact that the **error bounds in this paper are large**, growing exponentially in number of layers and steps. Additionally, the dimensions of a single Transformer layer need to be large enough to accomodate the weight parameters of the entire $N$-layer neural network, **requiring a lot of memory**, and this requirement seems unavoidable. 

A motivation of this paper was to have foundation models train other deep models, but the various issues listed above make me question whether the approach in this paper will lead to meaningful results.

Below I provide further details on some of the issues I have with the paper

1. The construction is not entirely explicit, Lemmas 2,3,4 appeal to the existence of some combination of ReLU functions to approximate sufficiently smooth functions via the $(\epsilon, R, H, C)$-approximable assumption. 

3. Theorem 1 requires the activation $r(t)$ and $r'(t)$ to be Lipschitz continuous, but ReLU does not satisfy this. It would be helpful for the authors to state a few concrete examples of N-layer neural networks (along with specific activations), and state what the Lipschitz constants/dimensions/error bounds end up being. 
4. The error bounds are quite large. Corollary 1.1 has exponential error in $L$, Lemma 6 has error growing as $d^N$. Some of the error is unavoidable -- divergence in gradient desecent over non-convex functions, and exponentially growing lipschitz constant of a $N$-layer Transformer. It would be useful if the authors can clearly distinguish the error terms inherent due to gradient descent over a non-convexity of the problem, vs the error terms that are due to the construction.
5. Experiments are a little unclear. What is the exact Transformer architecture used? Are the lines for 3/4/6 layer NNs in Figure 1 obtained from training a 3/4/6 layer NN over the in-context examples? Or are they the output of the "true network parameters for $f_i$" on line 2393?
6. The Transformer performance in experiments seems quite bad for 0.7/0.3 data. Can the authors also show the plot for 0.5/0.5?
7. If possible, it would be useful to see if scaling up the Transformer, as described in the paper, will indeed enable it to implement more gradient descent steps. (a crude way is perhaps to simply plot loss against number of Transformer layers.)

### Questions
1. Can the authors please compare your construction to the one in Algorithm 10 of [1]? Are there any similar ideas used?
2. If you are already assuming that various parts of the neural network are approximable by relu, and if you are already incurring exponentially-growing error terms, why do you not just treat the entire neural network as a $(\epsilon, R, H, C)$-approximable function, and simply use the Transformer to learn relu coefficients? Will this lead to worse bounds?
3. The dimension of the Transformer is very large, as each layer needs to contain all the parameters of the N-layer neural network. Can the authors please comment on what the exact dimension of the Transformer needed? I think it is useful to state this value in Theorem 1.

**minor suggestions that did not affect score**
1. $L$ is used for both loss, number of gradient descent steps, and Lipschitz constant. This gets confusing at times, and I suggest for the authors to use different letters.

[1] [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196), Giannou et el 2023

[2] [Transformers learn in-context by gradient descent](https://arxiv.org/abs/2212.07677), von Oswald et el 2023

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
4

### Summary
This paper studies transformer's in-context learning capability. It shows that given an $N$-layer neural networks, there exists a $(2N+4)L$-layer pre-trained transformer that simulates $L$ gradient descent steps. The analysis is also extended to softmax-based transformers. The findings are validated on synthetic datasets.

### Strengths
ICL is an important phenomenon, the theoretical explanation of which is still far from complete.

### Weaknesses
1. The main contribution of the paper is demonstrating the ability of transformers to simulate N-layer neural networks via gradient descent. However, this has been shown in reference [A]. The submission does not cite [A] and there is no discussion about what's novel compared to [A]. 

      [A] Wang, Jiang, Li, "In-Context Learning on Function Classes Unveiled for Transformers", ICML 2024. 

2. Question 1 asks "Is it possible to train one deep model with another pretrained foundation model?". This is not the same as ICL and the present submission does not answer this question. 

3. The organization and presentation requires significant improvement. Instead of focusing on a lot of notational details, the main paper should have more discussions about the significance and implication of the presented results, and the relationship between the presented results and existing results.

### Questions
1. What's the true contribution of the paper?

2. Is there any novelty in the proof techniques? If yes, it should be highlighted. 

3. What's the connection between the experimental results and the theory? Besides, the R-squared results do not seem very good. How can the experimental results validate the theory?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates the theoretical underpinnings of transformers as deep optimizers, specifically focusing on their capacity for in-context learning (ICL) to simulate gradient descent steps. The authors provide a construction of a transformer capable of performing gradient-based optimization implicitly without updating its own parameters. Through a series of lemmas and Theorem 1, the paper establishes that a pretrained transformer can approximate gradient descent steps of deep neural networks with convergence guarantees.

### Strengths
The general topic of theoretical understanding of transformers and ICL is interesting and novel.

### Weaknesses
1. The organization could be improved. The main result, Theorem 1, is only presented at the end of the paper. The purpose of preceding Lemmas 1-6 is unclear; it would help to clarify whether they serve solely as components of Theorem 1’s proof or have independent significance.

2. The studied problem lacks sufficient motivation. Transformers are known to have a complicated structure and may contain many parameters. Therefore it is not surprising that a transformer can have enough expressiveness to approximate the gradient descent steps.

3. There is no overview of the empirical validation in the main body of the paper. At least, the authors should use a few sentences to indicate which part of the conclusion is validated by empirical results.

4. The title says "pretrained transformers are deep optimizers", while no pretraining process is analyzed in the main body of the paper.

### Questions
See Weaknesses

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
This paper provides construction and proof on a ReLU-based Transformer are able to perform gradient descent steps for a N-layer fully connected neural network.

### Strengths
This paper presents clear and rigorous proof in showing Transformer can implement gradient descent for (x, y) ~ N-layer neural network.

### Weaknesses
1. The paper frequently uses the term "pretrained Transformer." Typically, this refers to a Transformer model trained on a large corpus for next-token prediction. However, in this paper, "pretrained Transformer" actually refers to a Transformer with specific weight constructions. I could not find an explanation of a "pretraining task" that leads to these constructed weights. This terminology may be confusing, especially in the introduction, where the authors emphasize that "one trained foundation model could lead to many others without resource-intensive pretraining".

2. The paper also discusses using the Transformer to estimate the score function for diffusion, which is presented as a major contribution. However, this is accomplished by reducing the score function to a fully-connected neural network. It might be helpful to adjust the emphasis on the diffusion score function and clarify that the main contribution is to provide theoretical support for the Transformer's capability to implement gradient descent for an N-layer neural network.

3. While the idea of demonstrating that a Transformer can implement an N-layer neural network is interesting, the paper only establishes the existence of such a network without addressing optimality or uniqueness. This makes the result slightly less compelling to the community, especially as the construction serves primarily as a proof of concept of the (modified) Transformer's capability rather than a more extensive finding. I understand that this limitation arises from the nature of the problem itself, and it may only be addressed through further results that demonstrate additional insights or practical applications.

### Questions
I found the necessity of the element-wise multiplication layer intriguing, particularly as discussed in Remark 3. Could the Transformer architecture circumvent the need for this element-wise multiplication by increasing its depth and width? In other words, is it possible to achieve the same functional capabilities without the element-wise multiplication layer by scaling the model’s size?

---
Updated after the discussion:
Thank you to the authors for taking the time to address my concerns. However, I still find the motivation behind training an N-layer neural network via in-context learning in another model questionable. Beyond the theoretical FLOPs and actual computational requirements, there is also the storage trade-off to consider. Unfortunately, I don’t see the benefits of replacing the direct training of an N-layer neural network with the forward pass of the construction model.

I am content with the story being about proposing a construction method to implement gradient descent for an N-layer neural network. Therefore, I will keep my rating at 6.

Reason for not a higher score: This work provides proof-of-concept results regarding the Transformer’s capacity to implement gradient descent for an N-layer neural network. However, these results lack practical insights for improving real-world applications, making this primarily a proof-of-concept contribution.

Reason for not a lower score: This work contributes to the literature on understanding the Transformer’s capacity. I appreciate the theoretical merit embedded in this study.

### Soundness
3

### Presentation
3

### Contribution
2
