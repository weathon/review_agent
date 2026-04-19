# How Transformers Implement Induction Heads: Approximation and Optimization Analysis

- Decision: Reject
- Scores: 5, 6, 8, 6, 6

## Abstract
Transformers exhibit exceptional in-context learning capabilities, yet the theoretical understanding of the underlying mechanisms remain limited.
A recent work (Elhage et al., 2021) identified a "rich" in-context mechanism known as induction head, contrasting with "lazy" $n$-gram models that overlook long-range dependencies.
In this work, we provide both approximation and optimization analyses of how transformers implement induction heads.
In the approximation analysis, we formalize both standard and generalized induction head mechanisms, and examine whether two-layer single- or multi-head transformers can efficiently implement them, with an emphasis on the distinct role of each transformer submodule.
For the optimization analysis, we study the training dynamics on a synthetic mixed target, composed of a 4-gram and an in-context 2-gram component. This setting enables us to precisely characterize the entire training process and uncover an *abrupt transition* from lazy (4-gram) to rich (induction head) mechanisms as training progresses.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper studies theoretical aspects of induction heads, which have been argued to play a key role in the skills that transformers demonstrate, including in-context learning.

The first set of results has to do with expressivity. Specifically, they show upper bounds on the 
size of transformers needed to implement variants of induction heads. The result here is not very surprising, as it uses basic transformer components to push tokens several steps ahead to facilitate the induction. It also seems to require increasing the dimension of the transformer as longer induction memory is requested, and again this seems intuitively natural, since this is what’s required for pushing n tokens forward in time, so they can be used in the induction. 

The second part asks about the learning process of induction heads. Towards this end, it considers a data generation mechanism that has two components: an n-gram and an induction head. It shows that the learning process goes through several stages where different components are learned. This part is potentially interesting, but I think it could benefit from more work before publication. Some points are:
a. I like the idea that there’s a type of inductive bias towards n-grams (also appearing in Bietti, 2023), but it would be more interesting to see if this is a true inductive bias if you had a setting where both n-gram and induction fit the data and learning chooses one over the other. In your current setup they both must be learned, because the output contains both, so I’m not sure what we learn from the fact that one is learned before the other, but eventually they are all learned.
b. If I understand correctly, the model has six learnable parameters. It’s possible that some observations (eg single fixed point) are due to this simplification. I would have liked to see at least simulations that show which phenomena appear in larger models.
c. Generally, I am missing more explanation and intuition why this particular model was chosen, and what are the implications for learning in real models.
d. I am missing more discussion of the relation to Bietti, who also propose some theoretical analysis of the learning dynamics. 
e. Please also discuss relation to https://arxiv.org/abs/2402.11004

### Strengths
Since induction heads are a key capability of transformers, it is certainly interesting to understand how they can be implemented and learned. There is some prior work on this, and the current work adds to that, especially in terms of analyzing optimization dynamics (though see points above).

### Weaknesses
As mentioned above, the expressiveness part is somewhat unsurprising. The dynamics is potentially interesting, but small scale in terms of parameters optimized, and the choice of model and lack of clear implications are also a concern.

### Questions
1. Figure 1, the caption doesn’t sufficiently explain what is seen in the figure. E.g., that highlights correspond to tokens with high attention weights for the induction head (I assume).
2. Following Equation 4, it would be good to highlight that this is not “just” the standard attention head because you have x_{s-1} instead of x_s. 
3. Can you comment on the relation between your results and Bietti (2023) equation 7, which also shows a two layer implementation? 
4. Your induction head averages over all previous appearances of the context. However, it isn’t clear that this is indeed the behavior of induction heads, or that it is even desirable. Wouldn’t we want to output, say, the most common element, or some encoding of the distribution over the next element?
5. The dependence on parameter q in Theorem 4.3 is confusing because we don’t see the dependence of C_{n,q} on q. Can you present a bound that optimizes over q? 
6. It seems like the dimension of the transformer in Theorem 4.3 needs to scale with n, “the induction length”. This seems rather undesirable, and it is not clear that it is necessary. Can you provide corresponding lower bounds, or provide empirical support for this need?
7. Typo: “the loss convergences”
8. In the definition of f*_{G4}, shouldn't it be L-2 (not T-2)?
9. It’s a bit confusing that in this problem you have X1,..,XL be one dimensional, but X_{L+1} is two dimensional, and you then map X1,...,XL to two dimensions. WOuld have been better to just say it’s all in two dimensions.

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
This paper studies Transformers expressing and learning induction heads. On the expression side, three settings, two-layer single-head Transformer without FFN, two-layer multi-head Transformer without FFN, and two-layer multi-head Transformer with FFN are considered are shown to be able to express induction head based on a single token, induction head based on several tokens, and induction head based on several tokens with arbitrary distance function. 
On the optimization side, the paper studies the training dynamics for learning a simple task which requires induction heads in a simplified training setting.

### Strengths
- The study of mechanisms behind attention heads is of both theoretical and practical interest. It's nice to see how different components can help Transformers implement induction heads of varying complexities. 
- Generally the insights and intuitions provided into network's presentation and optimization dynamics are informative and helpful.

### Weaknesses
- The expressivity results are just sufficiency conditions. There are no necessity results presented in the paper. E.g., it's not shown that 2-layer single-head Transformers cannot implement general induction head based on several head (although seems to be true and intuitive). 
- The optimization results correspond to a very simplified setting from the task side (working with a Gaussian distribution instead of discrete token distribution), model parametrization side, and optimization side (layer-wise gradient flow). I believe experiments supporting the claims in more realistic settings (e.g., more natural Transformer + GD + discrete token prediction task) could have been provided. Also the paper should be more upfront with the limitation of their optimization results throughout the paper in my opinion.
- The writing can be improved significantly (see the questions for more details).

### Questions
- In equation 4 and the following lines, I think the softmax notation is used improperly. Currently the input to the softmax function is a scalar. Also it's not clear what exactly $\{(x_s, x_L)\}$ means (maybe there's a typo?). The same problem appears again in equation 6, with the additional complexity that the dimensions of the matrix inside the softmax (and dimensions $X_{L−n+2:L}$) are not specified. I think in general we have dimension mismatch problems here based on the current notation. I would appreciate clarification of this issue. 
- The final value of $C_{q,n}$ is not specific clearly in the proof. I think this value is actually quite important as it determines the minimum number of heads required (I think this actually could be included in the main text.)
- Can you elaborate on the meaning of the notation $I(z=x_{L-2})$ (and similar ones) on page 7?
- In Figure 2, experiments setup is not mentioned. Also x-axis is not specified. More importantly, we can see a drop in the loss of IH term in the first phase of the training, this is in contrast to the theoretical results. Can you elaborate please?
- In Theorem 5, could you explain in what variables the asymptotics are?
- I think the writing of the insights around lines 243-250 can be improved. 
- The paper focuses on a specific format of relative positional embedding. I think the paper could be more upfront on this and state this more clearly throughout the paper. 
- Line 451, "without loss of ambiguity" seems to be a typo. 
- Regarding the second weakness, I'd appreciate any evidence/explanation that the proven optimization dynamics would also show up in more natural setting.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the theoretical analysis of the transformer mechanisms behind the induction heads from two perspectives: one is approximation theory and the other optimization.  In the approximation part, the authors show how to use transformers to approximate the induction head and the generalized induction heads. In the optimization, they investigates the abrupt transition from n-gram to induction heads.

### Strengths
(1)The paper provides a very sound and complete theoretical analysis of the transformer mechanisms behind induction heads, which are often used to explain the important emergent ability of in-context learning for LLMs. 
 (2) The paper is well-written and well organized.  In order to the motivate the work, the paper gives a very clear streamline of the related works and formulate the research objectives very clearly.  Also they provide very clear definitions of the key notions in the paper. I have not checked all the technical proofs but I believe that they are correct.  The paper focuses on the main objectives and makes the contributions explicit and discusses different possible scenarios.

### Weaknesses
(1)Since induction head is used to explain ICL, it might be more interesting to explain how the theory in this paper helps in-context learning(ICL), especially empirical results for ICL. 
(2) Although the paper is meant to be theoretical, it would be helpful to provide some empirical experiments to support the theoretical analysis.

### Questions
In Line 307, the authors mentioned that “the use of general similarity g enables the model to recognize not only synonymous but also antonymic semantics, thereby improving both the accuracy and diversity of in-context retrievals.”  Why do we need to use general similarity g to recognize antonymic semantics? Why does this recognition improve both the accuracy and diversity of in-context retrievals?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper comprises two parts. In the first part, the authors show through representation results that simple transformer architectures with two layers and several heads are able to correctly represent different variations of induction head mechanisms. In the second part, the authors prove through gradient flow that a simplified two-layer architecture can learn a mixed target function comprising a 4-gram component and a vanilla in-context 2-gram component.

### Strengths
The paper is generally well written and the proofs seem correct to me. Even if the paper considers heavily simplified models, the theory on this topic is scarce and difficult, so any theoretical insight on the dynamics of these models is welcome.

### Weaknesses
I think that the major drawback of the paper is that it mostly lacks experimental results that corroborate the theory. In particular, it would be interesting to see if the constructions used in Theorems 4.3 and 4.4 on in-context induction heads are actually learned by the considered transformer models. Additionally, I found the statements of some theorems to be rather vague (see the questions below).

### Questions
I have the following questions for the authors:
1. I find the statements of Theorems 4.3 and 4.4 vague. In particular, the way they are currently stated seem to imply that such results hold for _any_ number of heads $H$. In the proofs, however, it seems that $H$ cannot be arbitrary, and actually has to be large enough and possibly dependent on $n$, unless I misunderstood something. It would be helpful to clarify this further.
2. In the gradient flow analysis of Section 5.1.3, you consider a layer-wise training paradigm in which you first train only the first-layer, and then you fix it to train the second one. I was wondering if the experimental results of Figure 2 are also obtained using this paradigm. I was wondering if this assumption is also insightful in practice, i.e., if when training the two layers at the same time, you could see experimentally that the first layer is learned before the second layer, or if in general the two layers are learned together in practice.
3. Minor concern: in the main text you put some amount of emphasis on a novel Lyapunov function that is used in the proof, but then this function never appears in the text and is relegated at the end of the appendix. In the final version, I would either give more space to it in the main text, explaining why this function is important/novel, or put less emphasis on it.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The reviewed paper explores how transformers implement ''induction heads'' to perform in-context learning (ICL) by analyzing a simplified two-layer transformer model. The analyses include both approximation and optimization parts. The approximation analysis examines transformer submodules, while the optimization analysis tracks phase transitions in training dynamics as transformers develop induction mechanisms.

### Strengths
A key strength of this work is its rigorous theoretical approach within the chosen framework. The results and proofs are clearly delivered and effectively presented. The authors provide a comprehensive investigation from both approximation and optimization perspectives, which might potentially deepen our understanding of transformer models.

### Weaknesses
- This work provides an analysis of how transformers implement induction heads, approaching the problem from both approximation and optimization perspectives. The theoretical results appear rigorous; however, the model setup seems overly simplified. Specifically, the study is limited to a two-layer transformer model, with and without feed-forward networks (FFNs), a framework initially explored in a seminal paper [1] and subsequently developed by numerous follow-up studies. Given this extensive literature, the contribution here appears somewhat incremental, with limited novelty in the analytical approach and the techniques remaining relatively standard. Expanding the analysis to a more sophisticated and realistic setting, as seen in recent work [2], would significantly strengthen the contribution. Without this, the impact of the results may be constrained, and it is unclear if they meet the high standards for significance.

- Additionally, given the simplified setup and use of synthetic toy examples, I have reservations about the generalizability of these findings for interpreting real-world transformers. I would suggest that the authors conduct extensive empirical experiments on widely-used models to validate the applicability and robustness of their theoretical results.

- it would be valuable if the theoretical insights could yield practical implications. Specifically, can the approximation results inform new methods, or could the optimization insights benefit transformer training? If so, this would meaningfully enhance the contribution. Otherwise, the practical significance of the work may remain limited.

[1] Elhage, Nelson, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell et al. "A mathematical framework for transformer circuits." Transformer Circuits Thread 1, no. 1 (2021): 12.

[2] Chen, Siyu, Heejune Sheen, Tianhao Wang, and Zhuoran Yang. "Unveiling induction heads: Provable training dynamics and feature learning in transformers." arXiv preprint arXiv:2409.10559 (2024).

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
