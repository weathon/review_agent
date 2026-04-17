# Poly-attention: a general scheme for higher-order self-attention

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
The self-attention mechanism, at the heart of the transformer model, is able to effectively model pairwise interactions between tokens. However, numerous recent works have shown that it is unable to perform basic tasks involving detecting triples of correlated tokens, or compositional tasks where multiple input tokens need to be referenced to generate a result. Some higher-dimensional alternatives to self-attention have been proposed to address this, including higher-order attention (Sanford et al., 2023) and Strassen attention (Kozachinskiy et al., 2025), which can perform some of these polyadic tasks in exchange for slower, superquadratic running times.

In this work, we define a vast class of generalizations of self-attention, which we call poly-attention mechanisms. Our mechanisms can incorporate arbitrary higher-order (tensor) computations as well as arbitrary relationship structures between the input tokens, and they include the aforementioned alternatives as special cases. We then systematically study their computational complexity and representational strength, including giving new algorithms and matching complexity-theoretic lower bounds on the time complexity of computing the attention matrix exactly as well as approximately, and tightly determining which polyadic tasks they can each perform. Our results give interesting tradeoffs between different desiderata for these mechanisms, including a tight relationship between how expressive a mechanism is, and how large the coefficients in the model may be so that the mechanism can be approximated in almost-linear time.

Notably, we give a new attention mechanism which can be computed exactly in quadratic time, and which can perform function composition for any fixed number of functions. Prior mechanisms, even for just composing two functions, could only be computed in superquadratic time, and our new lower bounds show that faster algorithms for them are not possible.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces **Poly-Attention**, a unifying mathematical framework that generalizes standard self-attention into a family of *higher-order attention mechanisms*. Each variant is parameterized by an “attention polynomial” defining the interaction structure among tokens. The framework subsumes existing mechanisms such as **self-attention**, **tensor-attention**, and **Strassen-attention** as special cases. The authors analyze both **expressive power** and **computational complexity**。 Overall, the work bridges representational theory and fine-grained complexity, positioning Poly-Attention as a fundamental generalization of Transformer attention mechanisms.

### Strengths
* Introduces a **novel formal framework** — “attention polynomials” — that unifies multiple strands of higher-order attention research under a single algebraic abstraction.
* The theoretical results are carefully stated with proofs in the appendix, covering both upper- and lower-bound analyses.
* Experimental validation, though limited, correctly demonstrates the empirical viability of Tree-Attention.

### Weaknesses
* Experiments are confined to small synthetic tasks (function composition on sequences of length 25). While appropriate for illustrating theory, they do not demonstrate real-world scalability or integration into large models, since transformers are always stacked in a lot of layers. So I think a multi-layer experiments in real-world scenerio (like perplexity in NLP tasks) will make great contribution to the paper.

* The paper claims tree-attention has similar runtime to self-attention, but only reports per-epoch times qualitatively. Quantitative benchmarks on memory, throughput, or gradient stability would strengthen the practical argument.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a significant and elegant theoretical contribution to the understanding of attention mechanisms. It introduces "poly-attention," a general framework parameterized by an "attention polynomial" $h$ that generalizes standard self-attention and other recent higher-order proposals. The authors provide a comprehensive theoretical analysis of this framework, charting the trade-offs between computational complexity and representational power. While the theoretical work seems solid, the paper is held back by a critical lack of clarity in its core definition and insufficient experimental validation to support the breadth of its claims.

### Strengths
1. **Generalization of attention mechanism:** The poly-attention framework itself is an elegant generalization that unifies standard self-attention, tensor attention, and Strassen attention under a single, intuitive parameter: the attention polynomial $h$.

2. **Rigorous theoretical analysis:** The paper's main strength is its rigorous analysis. The authors use tools from computational complexity theory to provide a tight characterization of the running time. The finding that tree polynomials are the only class of poly-attention computable in quadratic time is an excellent and complete theoretical result.

### Weaknesses
1. **Clarification on core definitions:** The paper's central contribution, the poly-attention mechanism, is presented in Definition 2.2. This definition is inconsistent, making the paper's subsequent claims impossible to verify. Please correct me if I am wrong. The attention polynomial $h$ is defined over **$t$** variables ($x_1, ..., x_t$). However, the exponent in Definition 2.2 is written as $h(Q_{l_{1}}^{(1)},...,Q_{l_{k}}^{(k)})$, incorrectly using the polynomial's *degree* **$k$** as the number of arguments.

2. **Overstatement of representational power:** The claim that poly-attention can "solve polynomial root-finding" (Theorem 3.7) is a severe overstatement. However, the proof sketch in Appendix G does not describe any learnability result. Instead, it details a complex, static encoding where the polynomial $p^2$ is explicitly hard-coded into the query-key weights. This is just construction of a specific circuit, not evidence that a model can *learn* to solve this task.

3. **Meaningless Runtime Baselines:** The paper claims tree-attention's $O(n^2)$ complexity is a practical advantage over superquadratic methods (Strassen, 3-tensor). However, the runtime table (Figure 4) omits these mechanisms entirely, comparing only against 1- and 2-layer self-attention. Without these baselines, the practical benefit of the $O(n^2)$ complexity is completely unsubstantiated.

4. **Insufficient experiments on real world datasets :** A theoretical paper can be valuable. But when a paper makes explicit claims about practicality, efficiency, and solving tasks, it has a duty to validate them empirically. This paper's experiments are so weak, and they cannot demonstrate whether the proposed method is really useful.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors unified some of the existing attention variants into a new attention variant (poly-attention). In addition they showed that (with methematical theorems and proofs) under some conditions poly-attention can "efficiently" represent some functional forms that self attention, higher-order attention and strassen attention cannot represent "efficiently". One such example is the function composition problem. They showed that 2-fold function composition can be represented by poly-attention with $O(n^2)$ computational complexity while exiting attention variants require "superquadratic" time complexity. At the end they also provide an experiment which shows that poly attention learns 2-fold composition problem faster (in the sense of epochs) compared to 2-layer self attention, in the same experiment thay showed that 1 layer self attention cannot learn 2-fold composition problem.

### Strengths
The paper proposes a "unified poly-attentin" framework that organizes several high order attention machanisms. That seems useful especially for follow up theory work. I like the the represantation ability comparisons without ignoring the computational complexities. I didnt check every proof and the proofs in detail, but the upper/lower bounds story is believable and it explains why other attention variants blow up to superquadratic. Overall exposition is quite clear, mapping the special cases early helps both theory minded and experimental readers.

### Weaknesses
W1)About the motivation (I feel this is crucial): 
Although it can possibly achieve other functions in the paper the main motivation/achievement of poly-attention is stated around the functional composition problem (even your "punchline" is around it). However, you should explain more clearly why this particular functional form matters. What makes it a good choice? In principle, we can always cook up an unusual functional form and then design a neural network that works well for it. Because of that, we need some justification that this is not just an arbitrary example but a form that’s actually important or likely to appear in practice.

W2)About the experiments:
The experimental "validation" seems strange. All of the paper is about representational abilities of some models but the experiments are about the learning dynamics. Thus the experiments are not specifically about "validating" the theories. Seeing that the main thing about the experiment is comparing 2 layer self attention and poly-attention learning speed, they are just experiments themselves. However, that does not mean that these experiments are not relevant. Seeing that the paper does not have any theory about the learning dynamics that makes the provided experiments even more important, so I feel authors should improve them: 
1)The learning curves in the figure 2 and the appendix figures seem unusual. The accuracy slightly increases at the begining stays constant for some time then abruptly increases again and the epoch number it abruptly increases the second time seems "random" at first look. Therefore I strongly suggest repeating the same experiment with multiple random seeds (this experiment should not even require much computation the networks are very small so I am sure it can be repeated very easily), and eventually provide final plot with confidence bounds. 
2)Also, seeing that the main discussion in the paper is around the computational efficiency, the figure 2 should be replaced with another version in which the horizontal axis is the number of flops used instead of the epochs. 

as small note, in the experimental validations instead of using words like "roughly" they can easily provide the ratio of the time it takes (even a short footnote may be okay, instead of referring to appendix). 

W3) About the literature review: the hyper(feature)attention paper available at “https://arxiv.org/abs/2506.06179” appears to be closely related and should probably be discussed. That’s the first paper that comes to my mind, but there are probably other, closely related ones (with very similar self-attention variants). I just wanted to check whether you’ve already reviewed the literature and how confident you are in your coverage.

In addition to these, I discuss a fundamental possible weakness in the questions section.

### Questions
I feel that every poly-attention head defined in the paper can be written as a (single) t-order attention head, if we allow the tensor head to use a feature dimension that is $s$ times larger, where s is the number of monomials in the attention polynomial h. Here is a rough proof sketch

Let
$$
h(x_1,\dots,x_t) = \sum_{i=1}^s m_i(x_1,\dots,x_t)
$$
be the attention polynomial used by the poly-attention, where each monomial
$m_i$ uses some subset of the variables, e.g.
$m_i(x_1,\dots,x_t) = x_{j^i_1}\cdots x_{j^i_{k_i}}$.
For the input $X$ the poly-attention first forms the usual projections
$Q^{(j)} = X W_Q^{(j)}$ and $V^{(j)} = X W_V^{(j)}$.

Now consider a $t$-tensor attention head whose key vectors have dimension
sd (that is: s blocks, each of size d).
For each slot $j=1,\dots,t$ we build its key as follows:
for monomial $m_i$, if $x_j$ appears in $m_i$ we put the row
$Q^{(j)}_{\ell,:}$ in block $i$; if $x_j$ does not appear in $m_i$ we put
the all-ones vector in block $i$ (which can be acquired by adding biasses to the Q, K projections).
So every key is just $s$ copies of either “the real query” or “all ones.”

When the tensor attention takes the order-$t$ inner product of these keys,
the product over block $i$ multiplies exactly the variables that $m_i$
wanted, and multiplies $1$ for the variables that $m_i$ does not use.
Therefore the tensor score is

$$\sum_{i=1}^s m_i\bigl(Q^{(1)}_{\ell_1},\dots,Q^{(t)}_{\ell_t}\bigr)
= h\bigl(Q^{(1)}_{\ell_1},\dots,Q^{(t)}_{\ell_t}\bigr),$$
i.e. the same score as the poly-attention.

For the values do the same trick. Hence the tensor attention output, restricted to the first $d$ coordinates, coincides with the poly-attention output, for every input X. This shows that, up to a factor s increase in feature dimension, poly-attention does not define a strictly larger class of functions than standard t-tensor (higher-order) attention. At first sight this increase in feature dimension might look like an artificial blow-up. However, throughout the paper the authors do not argue that d must stay small; d is simply the embedding size chosen by the model designer. In that regime, allowing a constant (or even moderate) factor increase in d in order to simulate poly-attention with a standard $t$-tensor attention is not a strong penalty. On top of that, in practice we almost never need exact equality of attention scores. Since the model is trained end-to-end, it can approximate the desired score function well enough with fewer parameters than the exact tensor construction. So the sd-dimensional tensor above should be read as a worst-case. A trained model would likely get close with a smaller dimension. That makes the “poly-attention is strictly more expressive” claim weaker in a practical sense. 

Shortly, for any given poly-attention head, there are parameter settings of a higher-order/tensor attention head that can reproduce it. So poly-attention is not a strict superset of all higher-order variants; it’s another way to specify the same class of interactions. Thus my question is did you carefully consider this possibility while writing the paper?

Another caveat is practical: in real Transformer setups we usually don’t know the right interaction structure in advance, so fixing a specific polynomial might be brittle. From that viewpoint the current architecture looks more theoretical, and it would be good to follow up with experiments to show when/if poly-attention actually helps in some general situations, which brings us to the weakness W2 I mentioned.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces poly-attention, a generalized class of self-attention mechanisms that incorporate higher-order (tensor) computations and arbitrary relational structures between input tokens. The authors show that existing higher-order attention mechanisms, such as tensor attention and Strassen attention, are special cases of poly-attention. They provide a systematic analysis of the computational complexity and expressive power of these mechanisms, including new algorithms and matching lower bounds. A key contribution is the introduction of tree-attention, a subclass of poly-attention that can be computed in quadratic time and can perform multi-fold function composition—a task beyond the reach of standard self-attention and prior higher-order variants under similar time constraints.

### Strengths
Originality: The paper introduces a unified framework (poly-attention) that generalizes several recent higher-order attention mechanisms. The introduction of tree-attention and its theoretical and empirical validation is novel and meaningful.

Quality: The technical contributions are substantial. The authors provide rigorous complexity analyses, including both upper and lower bounds, and support their claims with proofs and experiments. The connection to fine-grained complexity (e.g., SETH, Max-2SAT) is well-executed.

Clarity: The paper is generally well-written, with clear definitions and a structured presentation. The use of graphical representations for tree-attention is intuitive.

Significance: The work addresses a key limitation of transformers—their inability to capture higher-order dependencies efficiently—and offers a practical path toward more expressive and scalable attention mechanisms. The results could influence both theoretical and applied research in efficient transformer architectures.

### Weaknesses
Experimental Scope: While the paper includes an experimental validation of tree-attention, the evaluation is limited to a single task (function composition). More diverse benchmarks (e.g., on standard NLP or reasoning tasks) would strengthen the practical relevance of the proposed mechanisms.

Presentation of Lower Bounds: The lower-bound proofs, especially those based on fine-grained complexity, are highly technical and may be difficult to follow for a general audience. A more intuitive explanation or summary of the proof techniques would improve accessibility.

### Questions
1. Could the authors evaluate tree-attention on more complex or realistic tasks, such as logical reasoning, mathematical problem-solving, or long-range dependency tasks, to better demonstrate its practical advantage?

2. How does tree-attention scale with sequence length in practice, especially when n is large (e.g., >10K tokens)? Are there any memory or numerical stability issues?

3. Could the authors provide more intuition or a high-level overview of the lower-bound proofs, especially for readers less familiar with fine-grained complexity?

### Soundness
3

### Presentation
3

### Contribution
3
