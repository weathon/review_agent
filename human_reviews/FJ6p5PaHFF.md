# Optimal Transport for Probabilistic Circuits

- Decision: Reject
- Scores: 8, 5, 6, 3

## Abstract
We introduce a novel optimal transport framework for probabilistic circuits (PCs). While it has been shown recently that divergences between distributions represented as certain classes of PCs can be computed tractably, to the best of our knowledge, there is no existing approach to compute the Wasserstein distance between probability distributions given by PCs. We consider a Wasserstein-type distance that restricts the coupling measure of the associated optimal transport problem to be a probabilistic circuit. We then develop an algorithm for computing this distance by solving a series of small linear programs and derive the circuit conditions under which this is tractable. Furthermore, we show that we can also retrieve the optimal transport plan between the PCs from the solutions to these linear programming problems. We then consider the empirical Wasserstein distance between a PC and a dataset, and show that we can estimate the PC parameters to minimize this distance through an efficient iterative algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors consider the computational complexity of Wasserstein-type distance metrics between probabilistic circuits (PCs), and show interesting and novel algorithms and lower bounds (hardness results) for computing them.
Notably, they show that it is $\sf NP$-hard to compute the $\infty$-Wasserstein distance between PCs, and there is an efficient algorithm for computing the Circuit Wasserstein distance between PCs.

### Strengths
The choice of the problem and the strength of the results.

### Weaknesses
See the questions below.

### Questions
Page 1:
Please consider these papers *(and their many relevant references therein)* about the computational aspects of total variation distance:

[1] Bhattacharyya, A., Gayen, S., Meel, K. S., Myrisiotis, D., Pavan, A., and Vinodchandran, N. V.: On approximating total variation distance. In Proc. of IJCAI, pp. 3479–3487. ijcai.org, 2023. Links: https://arxiv.org/abs/2206.07209; https://www.ijcai.org/proceedings/2023/387.

[2] Weiming Feng, Heng Guo, Mark Jerrum, Jiaheng Wang: A simple polynomial-time approximation algorithm for the total variation distance between two product distributions. TheoretiCS 2 (2023). Link: https://arxiv.org/abs/2208.00740v3.

[3] Weiming Feng, Liqiang Liu, Tianren Liu: On Deterministically Approximating Total Variation Distance. SODA 2024: 1766-1791. Link: https://epubs.siam.org/doi/10.1137/1.9781611977912.70.

[4] Arnab Bhattacharyya, Sutanu Gayen, Kuldeep S. Meel, Dimitrios Myrisiotis, A. Pavan, N. V. Vinodchandran: Total Variation Distance Meets Probabilistic Inference. Link: https://arxiv.org/abs/2309.09134.

Page 2:
Can you please further elaborate on the notions of smoothness and decomposability?

Page 3:
Please elaborate on the caption of Figure 1.

Page 4:
Section 3.2:
You SHOULD put your algorithm in a theorem statement :)

Page 5:
Please define the methods you use in Algorithm 1.

Lines 239 -- 255:
This part should be more detailed :)

Page 6:
Section 4.1:
Please use a theorem statement for your algorithm.

I do not understand Equation (4):
Why is the second equality correct?

Page 7:
Can you please elaborate on Lines 347 -- 352?

Page 8:
I am not an expert with experiments :)

Page 10:
Can you please add more details to the future work part, etc.?
It looks too small now.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper explores approaches for computing and bounding the Wasserstein distance and optimal transport plans in two settings: between two probabilistic circuits and between a probabilistic circuit and an empirical distribution. For the former, it introduces a Wasserstein-type distance that upper-bounds the true Wasserstein distance and provides an efficient algorithm for exact computation. For the latter, the authors present a parameter estimation algorithm designed to minimize the Wasserstein distance between a circuit and an empirical distribution. The proposed methods are validated through empirical evaluations on both randomly generated probabilistic circuits and a benchmark dataset.

### Strengths
This paper introduces, for the first time, a Circuit Wasserstein distance, denoted as $ CW_p $, between compatible probabilistic circuits (PCs). Leveraging the recursive properties of the Wasserstein objective, it computes the optimal parameters for the coupling circuit by solving a small linear program at each sum node. Additionally, the paper presents a method for learning the parameters of PCs by minimizing $ ECW_p $, which is computationally efficient.

### Weaknesses
This paper addresses only a restricted set of cases within the broader context of probabilistic circuits. Regarding optimal transport, the approach feels somewhat formulaic, lacking a deeper exploration of the essential relationship between optimal transport and probabilistic circuits. Additionally, there is a typo on line 786.

### Questions
1. Could you provide additional clarification on the proof of Theorem 1 and 2? I think it's better to add some graph in the proof.
2. Have you evaluated $CW_p$ across a broader variety of PCs, as its application might be limited?
3. Could you elaborate on the computational complexity of your approach?

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
This paper focuses on computing (or bounding) the Wasserstein distance and optimal transport plan between (i) two probabilistic circuits and (ii) a probabilistic circuit and an empirical distribution. For (i) a Wasserstein-type distance that upper-bounds the true Wasserstein distance was proposed and provided an efficient and exact algorithm for computing it between two circuits. For (ii) a parameter estimation algorithm for PCs that seeks to minimize the Wasserstein distance between a circuit and an empirical distribution was proposed.

### Strengths
1- The proofs of the theorems are correct, and the mathematical accuracy is high.

### Weaknesses
1- Using simple examples to illustrate definitions and results could make the paper easier to read and follow.

2- While the proposed metrics could significantly reduce runtime, they also lead to an increase in error. How much error is considered acceptable? There is no analytical approach or numerical result provided to show the impact of this error.

3- The proposed method performs well with a small set of variables; however, runtime challenges typically arise in large-scale systems with many variables.

4- There are insufficient numerical results to illustrate all aspects of the proposed distance. Applying the method to practical problems and providing comparisons with other works in terms of runtime and accuracy would strengthen the paper.

5- The application of this metric is not clearly explained in the paper. Additionally, given the limitations of the proposed CW and ECW metrics—such as susceptibility to error and effective performance only with a small set of variables—the metric has limited applicability in practical, real-world problems.

### Questions
1- Why wasn’t CW compared with W? 

2- How much error is considered acceptable?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors define (and show that it is tractable to compute) an analogue of Wasserstein distance (CW) between distributions encoded by structurally-identical probabilistic circuits (PCs). The high-level idea is to restrict to transport maps that have an analogous structure, and recursively decompose them, through which one obtains a metric that is an upper bound on the Wasserstein distance between the underlying measures.  The authors also provide a tractable analogue of a Wasserstein distance between a PC and an empirical distribution.  Code is provide for both procedures.  Random PCs are then generated according to a fixed structure, and the gap between CW and MW (another quantity in the literature) is studied, along with runtime. They also evaluate their measure of distance between PC and an emperical distribution against the EM algorithm in a learning context.

### Strengths
The ideas in this paper come through clearly, as the text (if not the math) is well-written.  The general idea is natural, and it is not hard to see how a more efficient way of calculating Wasserstein distances between the distributions encoded by PCs could, in principle, be tremendously useful.  Propositions 1 and 2 provide important grounding for the given constructions.  The authors take the space needed to unpack their ideas.  They identify appropriate baselines, and some genuine (if perhaps insufficient) effort has been made to empirically evaluate these ideas.  The proofs are present and look ok at a very high level (although I only skimmed parts of them).

### Weaknesses
Unfortunately, the paper is lacking in technical depth.  The idea and its implementation seem straightforward to me, and the results are unsurprising --- so I find the mathematical contribution relatively small (putting aside the substantial low-level issues with it, which I detail below).  This could easily be forgiven if the techniques enable something really interesting, but, on the experimental side, the examples are all small-scale synthetic toys. To the extent that this is really a basis of a sold paper, I believe it requires either an interesting application, a more interesting theorem, or to show that there are conceptual/instrumental benefits to using this approach. Finally, I also believe that the limitations of this approach are not well-enough explored (or indeed, well-enough advertised at the beginning of the paper).  Specifically, the fact that this distance measure only applies to PCs with exactly the same structure, is an enormous shortcoming which does not come through at the beginning.  I therefore believe that the technical contribution is being significantly oversold. 

The empirical results are not particularly encouraging.  The most obvious application of this paper would be to use this novel, easily-computed Wasserstein analogue for fitting parameters for PCs.  The authors correctly identify that their metric is closely related to EM, and compare against it as  baseline. However, their method does significantly worse. Yet the reasons for this are not discussed, and the authors do not mention any compensatory strengths that their distance measure has.  After finishing section 5.4, a reader can't help but wonder: why not just use EM for this task?  


As for the presentation, my biggest complaint is that the math is not handled very deftly.  While presentation and the mathematical English looks great a glance, there are a number of places where it is ambiguous or unclear or technically wrong.  I have detailed a number of significant zones below that look like train wrecks to me (but they are all salvageable, if one were to invest some significant effort).  


--- detailed comments below ----

Definition 1 is sloppy; it is mathematically imprecise and ambiguous.  What is the relationship between **X** and the nodes of the DAG?  How are the parameters for the sum nodes normalized?  What does the univariate probability distribution have to do with the variables?  Is the root of the dag a source or a sink, or possibly neither? The notion of scope, which is clearly important for the definitions to follow, is not defined at all.  Thus, when we get to the equation on line 097, I am very confused.  I do not know for certain which node is supposed to define a distribution over **X** (although I can only assume it is the root).  It also makes it seem that "input nodes" must correspond to variables of X.  But I'm still not clear on whether this needs to be a bijection or not.  Finally, additional restrictions are required to ensure that the result is actually a probability density function.  At the least, for product nodes, children must have disjoint supports (a property which, after definition 1, the authors call "decomposability"). But decomposability is not just required for the results of the paper results; it is required even for the words "define a probability distribution" to be correct. Either way, the current formalism doesn't typecheck; it implicitly eliding a projection in the definition of p_n for product and sum nodes.  Had I not seen this definition before, I would have been incredibly lost. 

(Line 107) Footnote 1: there's no need to "abuse notation". Simply choose the appropriate base measure, and use the radon-nikodym derivative. 

Algorithm 1 has some problems. 
 - the procedure cache (n, m) is not defined. 
 - there is no need to build the LP constraints iteratively with logic in the algorithm; just define the problem mathematically, and say "solve problem (P)"
 - Line 228 makes no sense to me.  What are the semantics of running LP.objective <- (...) multiple times?  
 - the sum over _{i,j}  seems to make the iteration over \theta_{i,j} in r.params useless. 
 - the fields "-.params" are not defined.
To summarize: this is not an algorithm---it is a snippet of Python code removed from its context and made slightly more colloquial. To call it an "algorithm", everything has to refer to something defined mathematically in the text!

Equation (4) exposes a deep flaw in the chosen notation: in the subscript of the expectation, there is not distinction between {\bf x}, which is bound by the expectation operator, and k, which is bound by the sum earlier.  In fact, the distribution you're taking an expectation over is should really be \gamma( x | k), not \gamma(x,k).  The same comment holds for the formula in definition 5.  

Line 282. The fact that y^k ~ Q and the i.i.d assumption are irrelevant for the definition.  You can just start with arbitrary y^k and define the empirical distribution \hat Q from it.

### Questions
1.  The notion of compatibility seems really quite strong. Are there any ways you can see of weakening it, so that structures that are similar (but not quite identical) can still be handled with your methods? It would be particularly nice if the method were to degrade naturally to an intractable (exponentially-sized) problem in the worst case when the two PCs have very different structures.  Do the authors think this might be possible? 

2.  Theorem 1 says that computing the \infty-Wasserstein distance is coNP-hard.  But this explicitly leaves open the possibility that calculating W1 or W2 is easier.  W1 and W2 are the most commonly-used in practice, and the motivation for this work has been that calculating Wp is hard.  What is the obstacle to showing that calculating W1 or W2 is hard?  If it is interesting, it is worth highlighting.  However, if the authors think that it might be possible to find a tractable algorithm for W1 or W2, then I feel strongly that it would be better for the research community to delay publication until this question is resolved. 

3.  Proposition 2 states that W(P,Q) <= CW(P,Q).  In light of the comments on the top of page 6 and the bottom of page 8, it seems that, in addition, W(P,Q) <= MW(P,Q) <= CW(P,Q).  Thus, it may be worth providing a framing this paper (at least locally, where this is discussed) as a looser and faster approximation MW.  In what contexts are MW used? Has it been experimentally validated? Might you be able to show a computational benefit in prior tasks by drawing from that literature? 

4. I found the Figure 4, and its explanation on lines 459-462 difficult to understand. I believe there is a missing word ("of"?) on line 462, but I can't figure out what the significance of this visualization is. What is the point of displaying this particular transport plan? What does this have to do with the story of the paper?  

5.  I assume that the points in Figure 3 represent pairs randomly sampled circuits. Are they all with the same architecture, or do only pairs have the same architecture?  Why are there so few points?    Figure 3 suggests to me that your method may work well on circuits of larger depths.   Have you tried to quantify this across a broader set of distributions over {P,Q}, or theoretically show that this must be the case?  These might be interesting avenues to explore.

### Soundness
2

### Presentation
4

### Contribution
2
