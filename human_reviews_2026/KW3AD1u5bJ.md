# Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Kirkpatrick et al. [ALT 2019] and Fallat et al. [JMLR 2023] introduced non-clashing teaching and proved that it is the most efficient batch machine teaching model satisfying the collusion-avoidance benchmark established in the seminal work of Goldman and Mathias [COLT 1993]. Recently, (positive) non-clashing teaching was thoroughly studied for balls in graphs, yielding numerous algorithmic and combinatorial results. In particular, Chalopin et al. [COLT 2024] and Ganian et al. [ICLR 2025] gave an almost complete picture of the complexity landscape of the positive variant, showing that it is tractable only for restricted graph classes due to the non-trivial nature of the problem and concept class.

In this work, we consider (positive) non-clashing teaching for closed neighborhoods in graphs. This concept class is not only extensively studied in various related contexts, but it also exhibits broad generality, as any finite binary concept class can be equivalently represented by a set of closed neighborhoods in a graph. In comparison to the works on balls in graphs, we provide improved algorithmic results, notably including FPT algorithms for more general classes of parameters, and we complement these results by deriving stronger lower bounds. Lastly, we obtain combinatorial upper bounds for wider classes of graphs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper considers computational complexity aspects of a generalization of the Non-Clashing Teaching Dimension (NCTD) of concept classes. The basic definition of the NCTD, introduced and analyzed in previous works, is as follows. There is a finite binary concept class $C$ and a (computationally powerful) teacher $T$. The teacher maps each $C$ to a set of labeled examples $T(C)$ representing $C$. The constraint is that for each $C \neq C'$, we should have an example in $T(C) \cup T(C')$ that is consistent with only one of the concepts. There is also a version NCTD+ that only allows positive labels.

It turns out that the concept of NCTD can be equivalently represented using balls in graphs. The current work focuses on a more specific case, where the balls are only of radius 1 ("closed neighborhood"). The authors prove a range of theoretical results, including an improved exponential conditional hardness (lower bound assuming ETH holds); near-matching upper bounds; Fixed Parameter Tractable upper bounds, parameterized by the tree-depth (among others); and NCTD upper bounds for specific graph classes, such as planar graphs.

### Strengths
- Strong theoretical results which extend the state of the art understanding of problems related to machine teaching.

- The proofs are interesting and well-written. I did not fully verify, but the main arguments seem believable.

### Weaknesses
- It is not clear what are the applications of the results, and how they relate to learning. If such applications exist, the author should make a better effort clarifying these.

- The paper is purely theoretical and might not be a best fit to a conference like ICLR. While the results seem interesting (even without concrete applications), they are probably a better fit in a more theoretical venue.

### Questions
I do not fully understand what is the connection between the result and the original definition of NCTD (for balls). While I understand that the problems are connected, I do not understand what exactly your results imply. Do they have interesting applications?

Do results for restricted graph classes have, e.g., graphs with low tree depth or planar graphs, have interesting applications? What is the typical structure of graphs in this setting?

### Soundness
3

### Presentation
2

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
This work studies (positive) non-clashing teaching for closed neighborhoods, showing this concept class is broadly representative and yielding stronger ETH-based lower bounds and tighter algorithms than prior results for balls in graphs. It proves no $2^{o(f(k)·|V|)}$ algorithm for N-NCTD under ETH and gives a tight $2^{O(|E|)}$ algorithm with matching lower bound for N-NCTD+, nearly matching known upper bounds in the general case. It further shows FPT results under treedepth (positive variant) and vertex cover (general variant) and derives small combinatorial upper bounds for planar and unit-square graphs, clarifying when structure enables efficient teaching maps.

### Strengths
1. Near-tight complexity results, improving lower bounds and adding new FPT regimes.
2. Careful comparison to balls in graphs, showing where closed neighborhoods help.
3. Combinatorial upper bounds tied to structural classes and VC-dimension observations.

### Weaknesses
1. Exposition can be heavy; more intuition and illustrative examples would aid accessibility.
2. Limited discussion of practical or empirical implications for machine teaching applications.
3. ETH tightness is strong theoretically, but could be complemented by empirical hardness studies.

### Questions
1. Are constructive or approximate teaching strategies possible for additional graph families ?
2. What are realistic constants in the FPT algorithms and prospects for practical solvers?
3. A concise taxonomy contrasting closed neighborhoods vs.\ balls would be valuable.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the notion of the *non-clashing teaching dimension* (NCTD) within the broader framework of machine teaching, focusing on both variants of non-clashing teaching for *balls in simple finite graphs with closed neighborhoods*.  
Relying on the Exponential Time Hypothesis (ETH), the authors employ 3-SAT style reductions to establish hardness results and improve algorithmic bounds for the positive variant (**N-NCTD⁺**), while also providing a corresponding lower bound for **N-NCTD**.  
They further design a fixed-parameter tractable (FPT) algorithm parameterized by *treedepth* for computing **N-NCTD⁺**, achieved via a pruning mechanism on the treedepth decomposition followed by a brute-force step that is fixed-parameter tractable in the treedepth parameter.  
Additionally, they propose a set of safe reduction rules that yield an FPT algorithm parameterized by the *vertex cover number*.  
The paper concludes with combinatorial arguments proving constant upper bounds on **N-NCTD⁺** for planar graphs with closed neighborhoods.

### Strengths
### Strengths

The lower bound improvement for **N-NCTD** is significant, and the **FPT parameterization by vertex cover** for **N-NCTD** is highly relevant.  

I found the presentation style particularly effective, as several proofs are preceded by intuitive explanations and supported by neat, well-designed diagrams. This approach makes the paper engaging and easy to follow. I believe this style should be extended to other proofs as well, with additional illustrative examples where appropriate. For example, while showing the correctness of the reduction in **Theorem 2**, since the size of the **NCTM** is 1, an explicit map could be shown to help readers follow the proof without compromising generality.  

I also appreciated the **combinatorial proofs for planar graphs**, which present a large use case and offer elegant solutions. However, the explanation could be made more verbose for better clarity. For instance, in the case where *d(v) ≥ 7*, it would help to explain why there are only four possibilities. A short intuitive note could clarify that this clashing arises only when all three choices match, or due to earlier shifting where two old and one new match occur—such combinations being \( \binom{3}{3} + \binom{3}{2} = 4 \). Including such brief intuitive explanations would greatly improve readability and pedagogical value.

### Weaknesses
### Weaknesses

The proofs and overall write-up are very specific and may primarily appeal to readers from the learning theory subcommunity. The paper is difficult to parse and, more importantly, to appreciate for readers without prior background in machine teaching or the concept of the non-clashing teaching dimension (NCTD).  

Although the proofs presented in Section 2 appear to be correct and I verified all of them rigorously, I found the graph construction to be very similar to the one used in *“The Computational Complexity of Positive Non-Clashing Teaching in Graphs.”* While correctness is not in question, I believe it is important for proofs to introduce novel ideas or techniques that could be applicable in other contexts; I did not find such novelty in the proofs of this paper.

### Questions
### Practical Applications and Discussion

Can you list down the practical applications of these notions and improvements?  
For **NCTD⁺**, how is it useful to use **treedepth** as compared to **vertex integrity** and what are the tangible benefits, in terms of theory and implementation perspectives?  
Similar to **NCTD**, would it be beneficial to have **vertex cover** as it is very general and has good approximation bounds?  
Can there be some practical experiments to quantify the impacts of **reduction rules** for FPT algorithms for **NCTD** and **NCTD⁺**?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies non-clashing teaching paradigm in learning theory where a teacher can give the learner catered examples instead of random samples in a non-collusive way. Specifically, the authors look at the specific cases where the concept class comes from a closed neighborhood of a graph and provide many theoretical analyses of both upper and lower bounds for this particular problem (which is as general is binary concept class learning).

### Strengths
The theoretical contribution is significant, as far as the reviewer can tell studying the immediate literature. And the math is correct, as far as the reviewer has checked (barring cited results). While the reviewer is not an expert in the immediate literature and cannot speak to how challenging it is to obtain such results, it is fair to say that the results are novel and comprehensive (spanning many different question, mostly solved to completion).

### Weaknesses
The paper is densely written and assumes some prior knowledge of the literature from the get-go (for instance, the jump from concept classes from balls in graphs in the introduction is a bit too sudden without explaining how a set of balls in G form a binary concept class, especially when graph quantities starts appearing in the bounds (e.g. line 67)). 

While this is expected from a theoretical paper, especially one that studies a rather niche problem, for the purpose of a machine learning conference, it is highly recommended that the authors defer more full proofs to the appendix, replacing it with a sketch proof, and use the extra space to motivate the relevance of the problem to the machine learning community (e.g. Line 108-114 can benefit from some illustration). A summary such as Table 1 is also something that should be in the main paper, at least the parts that are most significant. 

Most of the proofs can use simple illustrations and better guidance (e.g. Theorem 2, Lemma 6, Lemma 10) because of how dense the notations and case analysis is. 

In summary, the theoretical results are interesting and comprehensive, but the presentation of the paper needs major overhaul to be readable. I am willing to give another read and revise the score if the authors can clean up their expositions in the rebuttal period.

### Questions
n/a

### Soundness
3

### Presentation
1

### Contribution
3
