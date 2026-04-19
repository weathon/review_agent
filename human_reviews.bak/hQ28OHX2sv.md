# Transformers Perform In-Context Learning through Neural Networks

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3

## Abstract
Transformer based neural sequence models exhibit remarkable ability to do in-context learning. Given some training examples, a pre-trained model can make accurate predictions on a novel input. This paper studies why transformers can learn different types of function classes in context. We first show by construction that transformers implement approximate gradient descent on parameters of neural networks and provide an upper bound for number of heads, hidden dimension, and number of layers of the transformer. We also show that transformers can learn deep and narrow neural networks, which has better approximation capabilities compared to shallow and wide neural networks, using less resource. Our results move beyond linearity in terms of in-context learning instances and provide an understanding of why transformers can learn many types of function classes through the bridge of neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper shows that transformers can approximate gradient descent on 2-layer and n-layer neural networks. It uses the capabilities of neural networks as a bridge to understand the in-context abilities of transformers

### Strengths
The expressive power of transformers for performing gradient descent on n-layer neural networks is a novel result.

### Weaknesses
1. The writing of the paper is very poor, and the paper's readability should be improved. Just to point out a bit, 
in def 3, statements are introduced with many notations unspecified, e.g. Sobolev space, m, n;  
2. The paper seems to have a strong connection with Bai et al, 2023, in. notations and results, their result naturally extends to the multi-layer setting. The novelty and contribution of the paper compared to Bai et al, 2023 is not clear to me. 
3. The methodology of this paper is to first connect the transformer's expressive ability and neural networks, and then use the well-established approximation ability of neural networks, both of these two parts look not novel enough.



[1] Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection. Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, Song Mei, 2023

### Questions
See weakness

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies how transformers approximate n-layer neural networks by gradient descent and its variants. The theoretical results characterize the required number of heads, hidden dimensions, and layers of transformers. The authors also provide a comparison between deep, narrow networks with shallow, wide networks in terms of computational resources.


----------------------------

**After rebuttal**: Thank you for your response, and sorry for the late reply. I am sorry that I will keep the rating of 3. There are several reasons. (1) Weaknesses 1, 2, and 3 are not answered. (2) It is not a good reason to say other papers don't do experiments. First, I always feel that no experiment is a disadvantage. Second, I think other papers without experiments have their own originality and contributions such that they get accepted. I am not satisfied with the contributions in this paper, so experiments could be a complement.

### Strengths
1. The problem to be solved is significant and interesting to the community. 
2. This work extends the analysis to deep neural networks and makes a comparison between shallow networks and deep networks.

### Weaknesses
1. The biggest concern is the technical contributions of this work. I treat this work as a follow-up work of [Bai et al., 2023]. Therefore, it is very important to state the contribution beyond this existing work. I am not sure of the difficulty and the novelty of proving Theorem 2.

2. This work lacks empirical justification. 

3. The complete proof of Theorem 1,3,4 are not provided

[Bai et al., 2023]: Transformers as statisticians: Provable in-context learning with in-context algorithm selection.

### Questions
1. Can you show the comparison between deep, narrow networks and shallow, wide networks by experiments?

2. What is the definition of $\mathcal{W}$, the domain of $\bf{w}$? I do feel Assumption 2 is too strong. Is the existence of the MLP layer in Assumption 2 provable rather than assumed?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the in-context learning capabilities of Transformer-based models. It is shown by construction that Transformer can approximately implement gradient descent steps on the parameters of certain neural networks, where the upper bounds on the required number of heads, hidden size, and number of layers are provided. This suggests that Transformer can perform in-context learning by approximating a neural network.

### Strengths
The in-context learning capability of Transformer is an important topic recently in the community. This paper provides an interesting perspective of explaining the in-context learning performance of Transformer. The authors have clearly explained their idea, and the presentation is easy to follow.

### Weaknesses
- The main result presents the existence of the weights that enable Transformer to do gradient descent on a certain neural network. It is not shown if Transformer can be actually trained to do so.
- Theorem 3 and Theorem 4 only provide upper bounds, which do not necessarily suggest a separation.
- It seems that many recent papers on in-context learning of Transformer are missing. One of the most related ones is 
    - Trainable transformer in transformer. Panigrahi, Abhishek and Malladi, Sadhika and Xia, Mengzhou and Arora, Sanjeev.
- There are no numerical experiments supporting the theoretical results.
- Typo: Under Definition 2, "We doesn't" -> "We do not"

### Questions
1. In Definition 3, what is the definition of $W_{loc}^{m,\infty}(\mathbb{R})$?
2. It is not clear to me what Assumption 2 means. Is there a concrete example?
3. In Assumption 1, it is not clear to me what it suggests for the loss function to have finite Barron norm. According to the definition, it seems that we then need $\ell(w) \to 0$ as $\|w\| \to \infty$. Does the commonly-used $\ell_2$ loss satisfy this?
4. I find the results in Section 4 a bit confusing. Specifically, what does it mean by "learn the neural network"? Is it about doing gradient descent on certain loss? Otherwise, how is this related to the results in Sections?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
