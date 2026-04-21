# On the Expressiveness of Rational ReLU Neural Networks With Bounded Depth

- Avg Score: 7.40
- Decision: Accept (Spotlight)
- Scores: 5, 8, 8, 8, 8

## Abstract
To confirm that the expressive power of ReLU neural networks grows with their depth, the function $F_n = \max (0,x_1,\ldots,x_n )$ has been considered in the literature.
  A conjecture by Hertrich, Basu, Di Summa, and Skutella [NeurIPS 2021] states that any ReLU network that exactly represents $F_n$ has at least $\lceil \log_2 (n+1) \rceil$ hidden layers.
  The conjecture has recently been confirmed for networks with integer weights by Haase, Hertrich, and Loho [ICLR 2023].

  We follow up on this line of research and show that, within ReLU networks whose weights are decimal fractions, $F_n$ can only be represented by networks with at least $\lceil \log_3 (n+1) \rceil$ hidden layers.
  Moreover, if all weights are $N$-ary fractions, then $F_n$ can only be represented by networks with at least $\Omega( \frac{\ln n}{\ln \ln N})$ layers.
  These results are a  partial confirmation of the above conjecture for rational ReLU networks, and provide the first non-constant lower bound on the depth of practically relevant ReLU networks.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proves two lower bounds for the depth of a ReLU network with rational weights that exactly represents the maximum coordinate function.

### Strengths
1. This work extends the previous work by [Hasse et al., 2023]. It proves some statements for rational weighted neural networks, which is new to the literature. It seems to make a step towards understanding the exact functions representable by neural networks.

2. The presentation of the work is very clear. The results are very well-connected to the existing literature, their implications are made explicit, and the intuitions of the proofs are stated clearly.

### Weaknesses
1. The main weakness of this paper is the significance of the result. These analyses are interesting intellectual challenges, but I do not see too many practical implications. Studying the exact representations of neural networks is even a step back from the universal approximation results. This makes the results very brittle. For instance, I am not convinced by the argument that all weights in a computer are ultimately rational: while this is true, in a computer, one can neither represent the $F_n$ function studied in this paper exactly. Moreover, one can use a p-ary number to approximate any q-ary number arbitrarily well, which makes the results purely algebraic.

2. While the paper definitely contains interesting ideas, it seems to be fairly incremental and relies heavily on the mechanisms developed in [Hasse et al., 2023]. I don't see an exciting analysis tool that is invented in this work.

### Questions
1. Theorem 2 and 4 seem to have basically the same assumptions and yield two lower bounds. In which circumstances, should I care about one over the other?

2. Can you show or discuss the implications of Theorem 2 and 4 in the flavor of Conjecture 1 rather than approximation $F_n$?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The main result in the paper is a lower bound on the depth of the network for exactly representing the max function over the coordinates of the inputs. Namely, a ReLU feed-forward network with N-ary weights requires depth log_p(n+1) to represent the max over n coordinates, where p doesn’t divide N. Another result provides depth lower bound which relies on loglog(N).

### Strengths
1) The paper makes an important step toward solving the conjecture from Hertrich et al (2021), improving the previous depth lower bound which was restricted to networks with integer weights to networks with N-ary weights.

2) The presentation of the paper is clear and easy to follow.

3) I think the proof methods used in this paper are pretty unique in the literature of machine learning and may be used beyond the scope of this paper (although this could also be said to Haase et al. 2023).

### Weaknesses
I think this is a strong submission, and the weaknesses I write here are somewhat minor.

1) There are some definitions missing in the paper that are not common in the machine learning community, and it would be helpful to define everything formally. For example, semigroup (line 206). How is the dimension of a polytope defined (line 182)? 

2) Although the paper is fairly self-contained, there are parts that are left without proof. Some examples: Lines 230-231. Why is SU closed under Minkowski addition? I think these are “easy” exercises but for a machine learning paper, it would be better to show it directly.

3) I suggest also citing https://arxiv.org/pdf/2006.00625, which proves that if a similar depth separation result for depth k>= 5 can be proven for approximation (rather than exact representation), then it would solve a longstanding open question in TCS about the separation of threshold circuits. In other words, it is highly unlikely that a similar result can be proven for approximation.

### Questions
1) Is there a more high-level intuition on why divisibility by p means that ReLU networks of a certain depth cannot approximate the target? The lower bounds on approximation usually use some measure of complexity that grows gradually with the depth, e.g. the number of linear regions that can be represented by the network. Is there a similar measure here?

2) Why the width doesn’t play any role in exact representation, contrary to approximation?

3) How do residual connections affect the result? Does the characterization of Theorem 8 still hold?

### Soundness
4

### Presentation
4

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
The authors provide new lower bounds on the number of layers that a ReLU neural network needs in order to represent the function $\max$ {$0, x_1, \ldots, x_n$}.
The main result is that if all weights are $N$-ary fractions and $p$ is a prime number that does not divide $N$, then at least $\lceil \log_p(n+1) \rceil$ layers are needed to represent the function $\max$ {$0, x_1, \ldots, x_n$}.
The proof relies on a duality between continuous piecewise linear functions and polytopes, as well as on the additivity of the mapping from lattice polytopes to their normalized volume modulo a prime number $p$. The authors extend results from Haase et al., who (implicitly) showed a lower bound of $\lceil \log_2(n+1) \rceil$ if the weights are $N$-ary fractions for odd $N$.
Additionally, for a given $N$, they provide an upper bound on a prime number $p$ that does not divide $N$, resulting in a non-constant lower bound for representing $\max$ {$0, x_1, \ldots, x_n$} with a rational neural network using $N$-ary weights.

### Strengths
The paper should be accepted because it strengthens results in an important research area. Additionally, the paper is very well-written. 
- The expressiveness of ReLU neural networks is an active research field, and understanding the functions that are exactly representable with a specific architecture is a fundamental question. Finding lower bounds on the number of layers needed to compute all piecewise linear functions has remained an open problem for some years. 
-  The lower bound of $\lceil \log_3(n+1) \rceil$ layers to represent the function $\max$ {$0, x_1, \ldots, x_n$} for neural networks with decimal or binary fractions is a notable result. It contributes further to resolving the conjecture of a non-constant lower bound on the number of layers needed to represent all piecewise linear functions. 
-  The paper is well-organized, self-contained, and accessible. It effectively balances providing intuition with rigorous proofs.

### Weaknesses
While the results provide a meaningful strengthening, the methodology may appear somewhat incremental. The approach builds on Haase et al.'s method for networks with integer weights. Proposition 11 (the additivity of the mapping from a lattice polytope to its normalized volume modulo a prime number) is the main new technical component, derived from mixed volumes, which generalizes the parity of the normalized volume of certain polytopes associated with neural networks with integer weights to the divisibility of the normalized volume by a prime number.

That said, I still believe the paper is a valuable contribution, potentially opening the door to find other (algebraic) invariants for resolving (other special cases of) the conjecture. Moreover, the authors did an excellent job of bringing together these elements in a self-contained manner.

### Questions
- Do you see any potential to bound the denominator of the weights needed to represent the max function with fewer than logarithmic layers (and rational weights) so that your results would yield a non-constant lower bound?

Minor issues: 
- On Line 151, should it be "faces of dimension at least $2^k$ for $k$ hidden layers"? 
- The sentence in the conclusion, "In the case of rational weights, even allowing arbitrarily large denominators for any given $N$ and arbitrary width does not facilitate exact representations of low constant depth," might be slightly misleading. Perhaps consider phrasing it as "In the case of rational weights that are $N$-ary fractions..."

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors identified the lower bound of the depth to exactly represent the function $F_n = \max (0,x_1,\dots,x_n) $. In particular, they focused on the rational neural networks whose weights are $N$-ary fractions, and showed that the depth $\lceil \log_p(n+1) \rceil$ and $\Omega(\frac{\ln n }{\ln\ln N})$ is necessary. These main results partially answer the conjecture by Hertrich et al. (2021).

### Strengths
**Extend previous works**

Independent of studying whether NNs can approximate a class of functions, identifying the class of functions that can be exactly implemented by NNs is one of the fundamental studies for the expressive power of NNs. The authors consider networks whose weights are $N$-ary fractions which represent floating point/fixed point arithmetic networks. Importantly, they exploit mixed-volume theory to prove the statement, which is not directly used in the previous works.
Overall, the results certainly help us have a better understanding of deep neural networks in view of a theoretical and practical way.

**Clarity and Quality**

well-written and clearly structured paper. The problem has been motivated nicely in the introduction. The related work section is rigorous and clearly details the current results for this problem.

### Weaknesses
**Partially answer conjecture 1 by Hertrich et al. (2021)**

Nonetheless, it is not a critical issue. As mentioned earlier, I think considering weights with $N$-ary fractions is more reasonable compared to real weights, since computers use floating/fixed point arithmetic. Thus, to highlight the novelty, it would be better to pose another conjecture, not focusing on conjecture 1, about a ReLU network using $N$-ary fractions weights.

**Proof technique**

Though the authors apply mixed-volume theory, most of the key lemma are based on previous works [Hertrich et al. (2021); Haase et al. (2023)].
Compared to the previous works, the proof strategy seems similar. Thus, I think this makes the contribution of this paper incremental. To highlight the novelty, it would be better to clarify why mixed-volume theory is necessary to extend the argument of Haase et al. (2023) (divisor 2 -> prime $p$).

### Questions
**Width and complexity**

As the depth, understanding the expressive power with respect to the width and # of parameters is also important. How do you expect the effect of the width or the number of parameters in your context? Suppose that $f$ can be represented by a ReLU network $g$ with $k$ hidden layers, but not with $k-1$ hidden layers. How do you expect the width and # of parameters of $g$? Can we use similar approaches (modulo, mixed-volume, etc.)?


**ReLU to other piecewise linear activation**

Once we shift our attention from ReLU to other piecewise linear activation (e.g. Leaky-ReLU), can we get similar results? Unlike ReLU, conjecture 1 may not be equivalent to the statement that any ReLU network representing $\max \\{ 0,x_1,...,x_n \\}$ requires $\log_2(n)+1$ hidden layers.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present new results about the impact of depth on the capacity of ReLU Neural Networks to exactly compute $F= x \mapsto \max \{ 0, x_1, \cdots, x_n \} $. More precisely, they investigate possible lower-bounds on the depth needed to compute $F$. This problem has direct consequences on the capacity of such Neural Networks to compute all piecewise-linear functions $\mathbb{R}^{n} \rightarrow \mathbb{R}$. They show that if the weights of the ReLU network are N-ary fractions (i.e. fractions of the form $ \frac{z}{N^{t}}$ where $z, N > 0, t \geq 0$ are integers), then there is a general logarithmic lower-bound.

### Strengths
- The paper makes progress towards a complete answer to an open and (believed to be) difficult conjecture.
 
- The class of ReLU neural networks considered are practically relevant.

- The technical parts are well-presented for readers not fluent in discrete geometry.

### Weaknesses
- There is no major weakness to this paper in my opinion. The minor weakness is that the paper is heavily inspired by a previous paper in the approach and only limited technical ingredients are added.

### Questions
Do the authors believe that an entirely new approach should be adopted to obtain results in the rational case?

### Soundness
4

### Presentation
4

### Contribution
3
