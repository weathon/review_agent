# Generalization Bounds for Canonicalization: A Comparative Study with Group Averaging

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 5, 5

## Abstract
Canonicalization, a popular method for generating invariant or equivariant function classes from arbitrary function sets, involves initial data projection onto a reduced input space subset, followed by applying any learning method to the projected dataset. Despite recent research on the expressive power and continuity of functions represented by canonicalization, its generalization capabilities remain less explored. This paper addresses this gap by theoretically examining the generalization benefits and sample complexity of canonicalization, comparing them with group averaging, another popular technique for creating invariant or equivariant function classes. 
Our findings reveal two distinct regimes where canonicalization may outperform or underperform compared to group averaging, with precise quantification of this phase transition in terms of sample size, group action characteristics, and a newly introduced concept of alignment.
To the best of our knowledge, this study represents the first theoretical exploration of such behavior, offering insights into the relative effectiveness of canonicalization and group averaging under varying conditions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Existing studies have explored the expressive power and continuity of functions represented through canonicalization, but its generalization ability remains underexamined. This paper fills this gap by providing a theoretical analysis of the generalization benefits and sample complexity associated with canonicalization, comparing these attributes with group averaging—a widely used technique for constructing invariant or equivariant function classes.

### Strengths
1. This paper fills the gap in theoretical analysis within the field of XX.
2. The paper provides detailed proofs for all theoretical results.
3. The paper demonstrates the validity of the proposed theoretical results through experiments.

### Weaknesses
1. The details of the proofs need to be refined, as the reasoning in many steps is not particularly straightforward.
2. As a paper focused on statistical theory, it would be best to outline the structure of the proofs to facilitate understanding.
3. Some definitions seem to be incorrect. For example, on the left of Eq. (2), the function $R(f)$ has no independent variable $x$, but $x$ appears on the right of Eq. (2). The similar problem also exists in several place of this paper.

### Questions
See "Weaknesses".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This is a theory-oriented paper focusing on the generalization bounds of canonicalization. Specifically, it introduces the concept of "alignment" to describe the performance differences between canonicalization and group averaging under various conditions. Following the theoretical analysis, a proof-of-concept experiment is presented to support the conclusions.

### Strengths
1. The organization and writing of this paper are reader-friendly.
2. The good generalization analysis aids researchers in understanding, designing, and applying canonicalization and group averaging techniques.
3. According to the authors, this is the first theoretical exploration of performance differences between canonicalization and group averaging under varying conditions.

### Weaknesses
1. The experiment section would benefit from more case studies.
2. There is no specific description of sample generation in the experiments.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper theoretically investigates the generalization error of canonicalization vs group averaging. The theoretical results suggest that the generalization performance of canonicalization and group average depends on the trade-off between variance and bias (approximation error), where group averaging being superior in variance (by a factor of group size) and canonicalization being superior in bias based on a newly defined alignment metric. The authors also provide a set of proof-of-concept simulation to show that group averaging and canonicalization outperforms each other under different true functions.

### Strengths
1. Theoretical results have consistent notations and are easy to follow.

2. The theoretical results shows the interesting trade-off between bias and variance in the generalization error of canonicalization and group averaging, where canonicalization, in general, has better bias and worse variance. This alludes to the idea that canonicalization might benefit in the case of large data size.

3. The trade-off also results in a critical data size value, which unveils an indicator to compare group averaging and canonicalization.

### Weaknesses
1. Corollary 7 seems to be useful and interesting at the first glance, but there are some issues with it. First of all, equation (17) and (18) could be stated more carefully even though the idea is intuitive, as the main theorem provides upper bound rather than precise value (which could make the argument false with small probability). Second, the critical value depends on some factors that are practically intractable, mainly the $\beta$ alignment factor, but also the noise level. Third, in the case of $\beta = s$ as suggested in Proposition 3, the critical value goes to infinity, which suggests canonicalization will always be inferior.

2. The statement of Proposition 3 might also cause confusion. First, it is better to state the function class here to remind the reference function class when talking about alignment. Second, upon further reading into the Appendix C, it seems the statement should be $\beta \geq s$ rather than $\beta = s$ (The authors discussed this relation above and below the paragraph, but it could be better to have it in the statement unless there are concerns the authors like to clarify), as this $s$ derivation is based on group averaging and Theorem 1. Also, when $s=\beta$, it actually causes problem in Corollary 7, where the critical value will go to infinity, which suggests canonicalization will always be worse for G-invariant Sobolov function with parameter s.

3. Simulation is lacking. The provided simulation does align with the general intuition that when the true function is close to the canonicalization representation, canonicalization could do better. However, this simulation does not provide any insights on the effect of data size, which is the main theoretical contribution of the work (the error comparison over the number of observations). I hope the author could at least provide a simple set of simulations that shows the error decay of group averaging vs canonicalization which shows the two converge as n increase and then canonicalization takes the lead. I also suggest investigating the empirical critical value (when error of group averaging starts to surpass canonicalization) over different factors to see whether Corollary 7 could have practical implications for future iterations (considering the limited time for the rebuttal).

1. There are some minor typos that might confuse some readers. For example, one super-script G is missing in equation (15).

### Questions
1. The authors mentioned in the literature review that people have studied the generalization error of group averaging for kernel methods. However, how does existing bound compares with the bound derived by the authors is not discussed. Can the author share some details on this matter and remark the novelty here?

2. I have a minor question regarding the example 4 and please correct me if I am wrong. The example is given to demonstrate when the closed space assumption is violated, there could be cases where $F_{GA}$ is not a subset of $F_{CAN}$. However, as stated in line 300, "the function belongs to $F_{CAN}$ but not $F_{GA}$, which suggests $F_{GA}$ is not a subset of $F_{CAN}$." I am not quite following the logic here as I believe we need something on the contrary right?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper addresses this gap by theoretically examining the generalization benefits and sample complexity of canonicalization, comparing them with group averaging, another popular technique for creating invariant or equivariant function classes.

### Strengths
It provides new insights into the generalization bounds of canonicalization and enriches related theoretical research. By comparing with the group averaging method, it highlights the advantages and disadvantages of different methods and provides practical references for researchers. The paper has a reasonable structure, clear logic, and is easy for readers to understand. The paper studies topics of great significance in machine learning and statistics, which is in line with current research trends.

### Weaknesses
Although the theoretical analysis is thorough enough, it might be helpful to add some experiments to ensure empirical support.

### Questions
none

### Soundness
3

### Presentation
3

### Contribution
3
