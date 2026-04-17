# Using Clause Predictions for Learning-Augmented Constraint Satisfaction

- Decision: Reject
- Scores: 8, 6, 4, 4

## Abstract
We continue a recent flourishing line of work on studying NP-hard problems with predictions and focus on fundamental constraint satisfaction problems such as Max-E3SAT and its weighted variant. Max-E3SAT is the natural `maximizing' generalization of 3SAT, where we want to find an assignment to maximize the number of satisfied clauses. We introduce a clause prediction model, where each clause provides one noisy bit (accurate with probability $1/2 + \varepsilon$) of information for each variable participating in the clause, based on an optimal assignment. We design an algorithm with approximation factor of $7/8+\Theta(\varepsilon^2/\log(1/\varepsilon))$. Our algorithm leverages the fact that in our model, high-occurrence variables tend to be highly predictable. By carefully incorporating a classic algorithm for Max-E3SAT with bounded-occurrence, we are able to bypass the worst-case lower bounds of $7/8$ without advice (assuming $P \ne NP$). 

We also give hardness results of Max-E3SAT in other well studied prediction models such as the $\varepsilon$-label and subset prediction models of Cohen-Addad et al. (NeurIPS 2024) and Ghoshal et al. (SODA 2025). In particular, under standard complexity assumptions, in these prediction models, we show Max-E3SAT is hard to approximate to within a factor of $7/8+\delta$ and Max-E3SAT with bounded-occurrence $B$ (every variable appears in at most $B$ clauses) is hard to approximate to within a factor of $7/8+O(1/\sqrt{B})+\delta$ for $\delta$ a specific function of $\varepsilon$. Our first lower bound result is based on the framework proposed by Ghoshal et al. (SODA 2025), and the second uses a randomized reduction from general instances of Max-E3SAT to bounded-occurrences instances proposed by Trevisan (STOC 2001).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies Max-E3SAT with learning-augmented predictions. It introduces the Clause Advice model where each clause provides noisy sign predictions for its variables. The main result is a polynomial-time algorithm achieving an approximation ratio that surpasses the classic 7/8 barrier. The algorithm combines majority decoding for high-occurrence variables with bounded-occurrence techniques. Hardness results show ohter prediction models cannot beat 7/8, and an upper bound establishes limitations of majority-based strategies.

### Strengths
Clause Advice model ties predictive power to variable importance. Inverted-predictions construction is a clever hedging strategy.

Complete proofs including detailed re-derivation of bounded-occurrence techniques. 

Clear writing.

 Breaks known complexity barrier

### Weaknesses
Independence claim for certain events relies on disjointness of constructed sets. This could be stated more explicitly.

### Questions
Could alternative decoding strategies potentially beat the upper bound limitation?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This study explores the problem of MAX-E3SAT within the context of learning-augmented algorithms, utilizing oracle advice. The MAX-E3SAT problem involves a conjunctive normal form (CNF) formula where each clause consists of exactly three literals. The objective is to find a Boolean assignment that maximizes the number of satisfied clauses. After examining traditional prediction models—label advice and subset advice—the authors introduce a novel prediction model called "clause advice." In this model, each clause can be represented by a vector of bits corresponding to the values of the literals in a given assignment. Essentially, each clause provides a noisy variant of the vector associated with the optimal assignment. Within this framework, each bit in the clause is modeled as a Bernoulli random variable that offers a slightly more accurate prediction than a random uniform choice.

Using $\epsilon$ to represent this information, the authors develop a $ \frac{7}{8} + \Theta(\epsilon^2 / \log(1/\epsilon)) $-approximation algorithm, that recovers the classical random choice algorithm when $\epsilon$ tends to zero. They also establish a hardness result for this scenario. Additional approximation and hardness results are provided for the problem MAX-E3SAT[$B$], where each variable occurs in at most $B$ clauses.

### Strengths
**S1.** The paper is well-written, and the notation used is clear. The technical aspects are easy to follow, even for readers who are not experts in learning-augmented algorithms. The results are presented in a pedagogical manner, effectively explaining their significance and the main ideas behind the proofs.

**S2.** Although the study specifically addresses the MAX-E3SAT problem, the approximation and hardness results extend beyond mere extensions of existing findings. Therefore, the “clause advice” model could also be utilized to address other MAX-CSP problems.

### Weaknesses
**W1.** While the approximation and hardness results are not straightforward extensions of existing findings, they are limited to the specific MAX-E3SAT problem. Their practical implication is therefore questionable.

**W2.** Although the "clause advice" model is conceptually interesting, it relies on the assumption that each "literal occurrence" in a clause behaves like an independent Bernoulli random variable. This assumption feels somewhat counterintuitive, as the assignment of any literal in a given constraint naturally affects its value in other constraints. Therefore, I am not entirely convinced that this predictive model is more appropriate than the classic "label advice" model.

**W3.** While the authors present approximation and hardness results for both the "clause advice" model and the "subset advice" model, they do not provide similar results for the natural "label advice" model. The authors conjecture that the $\frac{7}{8}$ bound for this oracle cannot be improved, but it would be helpful to include some informative comments regarding this conjecture.

### Questions
This study focuses on the MAX-E3SAT problem. The authors provide a brief discussion of extensions in Section 3. However, I am particularly interested in whether the results can be applied to variants of MAX-SAT. Here are my questions:
 
**Q1.** Can we extend the approximation and hardness results to the standard MAX-3SAT problem, where each clause contains at most three literals?
 
**Q2.** Is it possible to improve the approximation bound in the clause advice model when the instances of MAX-E3SAT are randomly generated?
 
**Q3.** Although the MAX-2SAT problem has already been studied within the context of the "label advice" model, can we achieve similar approximation and hardness results for the "clause advice" model?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
An approximation factor of 7/8 is provably the optimal for polynomial algorithms for MAX-3SAT. To improve on this, learning-augmented algorithms utilizes additional information (advice). Existing approaches include Label Advice and Variable Subset Advice models. However, in MAX-3SAT, Label Advice is believed to give no improvement, while Variable Subset Advice makes the problem too easy to improve by assuming knowledge of exact assignments without uncertainty. This paper introduces Clause Advice models where a clause level predictor that makes noisy prediction where its correct with probability $(1+\epsilon)/2$ is available. The paper then proves that there exist a polynomial-time algorithm in Clause Advice model that can find an assignment with approximation factor lower bounded by $7/8 + \theta(\epsilon^2 / log(1/\epsilon))$ in expectation.

### Strengths
The proposed algorithm builds on prior work by incorporating the concept of degree for the variables. 

The clause advice model is a simple yet intuitive extension to the label advice model for 3SAT.

Extensive proofs are provided for the theorems.

### Weaknesses
The paper is difficult to follow and can use some reorganization. The theorems are presented early on without context, and related contents are scattered throughout the paper. The presentation can also be improved, for example, the main algorithm is described in one paragraph on the top of page 4. A simple example can clarify the algorithm much better.

The clause advice model’s applicability beyond Max-3SAT remains to be demonstrated and may be less general across CSPs than label or subset advice.

It is difficult to gauge the significance of the improved bounds of the proposed model.

The method assumes the variable predictions are independent across clauses, which is intuitively less likely in practice.

### Questions
Line 88 seems to suggest that Label, Variable Subset and Clause Advice are the only three natural prediction models. Is that intended? What about Clause Advice models that use predictions on the clause as a whole instead of the variables in the clauses?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies the Max-E3SAT problem—an NP-hard constraint satisfaction problem—and introduces a new Clause Prediction Model within the learning-augmented algorithms framework. In this model, each clause provides a noisy bit of information (accurate with probability 1/2 + \epsilon) about the optimal assignment of its variables. Using this model, the authors design a polynomial-time algorithm that achieves an approximation ratio of 7/8 + \Theta(\epsilon^2 / \log(1/\epsilon)) surpassing the classical 7/8 barrier known to be optimal in the worst case (under P ≠ NP).
Overall, the paper advances the theoretical understanding of how noisy predictions can help solve NP-hard problems beyond worst-case limits. The paper is technically solid but difficult to follow for non-specialists, and no experimental validation or discussion of real-world applicability is included, which may reduce the perceived relevance for ICLR.

### Strengths
1.	Introduces a new, natural, and noise-tolerant advice model (“clause predictions”) that is well-motivated and distinct from existing label or subset advice frameworks.
2.	Solid theoretical analysis with rigorous proofs and clear complexity-theoretic hardness arguments.
3.	Highlights that “per-constraint” (clause-level) predictions can be more informative than “per-variable” predictions, offering a new research direction for learning-augmented combinatorial optimization.

### Weaknesses
1.	The paper is very difficult to read for researchers not already familiar with approximation algorithms or CSP theory. The exposition jumps quickly into formal definitions with minimal intuition or examples.
2.	The core idea—why clause predictions help, and how they allow going beyond 7/8—is mathematically clear but conceptually opaque. The link between prediction accuracy \epsilonε, variable degree, and approximation improvement is not well visualized or intuitively explained.
3.	The writing style mimics theoretical computer science papers (tight, notation-heavy, and formal), not the more expository tone expected at ICLR.
4.	No experimental validation or discussion of real-world applicability is included, which may reduce the perceived relevance for ICLR.

### Questions
1.	Can you provide more intuition or a simple example illustrating how clause predictions improve approximation beyond 7/8?
2.	How sensitive is the algorithm’s performance to adversarial noise beyond the assumed independence in the clause predictions?

### Soundness
3

### Presentation
1

### Contribution
3
