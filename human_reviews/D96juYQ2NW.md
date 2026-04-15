# Coresets for Clustering with Noisy Data

- Decision: Reject
- Scores: 5, 5, 6, 6

## Abstract
We study the problem of data reduction for clustering when the input dataset $\widehat{P}$ is a noisy version of the true dataset $P$.
Motivation for this problem derives from settings where data is obtained from inherently noisy measurements or noise is added to data for privacy or robustness reasons.
In the noise-free setting, coresets have been proposed as a solution to this data reduction problem -- a coreset is a subset $S$ of $P$ that comes with a guarantee that the maximum difference, over all center sets, in cost of the center set for $S$ versus that of $P$ is small.
We find that this well-studied measure which determines the quality of a coreset is too strong when the data is noisy because the change in the cost of the optimal center set in the case $S=\widehat{P}$ when compared to that of $P$ can be much smaller than other center sets.
To bypass this, we consider a modification of this measure by 1) restricting only to approximately optimal center sets and 2) considering the *ratio* of the cost of $S$ for a given center set to the minimum cost of $S$ over all approximately optimal center sets.
This new measure allows us to get refined estimates on the quality of the optimal center set of a coreset as a function of the noise level.
Our results apply to a wide class of noise models satisfying certain bounded-moment conditions that include Gaussian and Laplace distributions.
Our results are not algorithm-dependent and can be used to derive estimates on the quality of a coreset produced by any algorithm in the noisy setting.
Empirically, we present results on the performance of coresets obtained from noisy versions of real-world datasets, verifying our theoretical findings and implying that the variance of noise is the main characterization of the coreset performances.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies the problem of coreset for $k$-means clustering in the noisy setting.
Given a dataset $P \subset\mathbb{R}^d$ and a set $C$ of $k$ centers, one can define the cost to be $\sum_{x\in P} \min_{c\in C}\| x-c \|^2$.
The $k$-means clustering problem is to find the set $C$ that minimize the cost given a dataset $P$.
However, when the size of $P$ is huge, one common technique to improve the storage and computation is to extract a smaller subset $S$ of $P$ and find the set $C$ such that the $k$-means cost is minimized for $S$.
Different previous results showed that we can construct such a subset $S$ such that the size of $S$ is small and the difference between the $k$-means cost for $P$ and $S$ is also small.
In reality, the data is often noisy.
Therefore, we would like to construct a coreset from the noisy dataset $\widehat{P}$.
The authors showed that the standard notion of coreset for $k$-means is too strong for noisy dataset and defined a new notion for the noisy setting.
The authors then showed that if $S$ is a good coreset for $\widehat{P}$ under some mild assumptions on the noise then $S$ is also a good coreset for $P$.
The main idea is to first show that $\widehat{P}$ itself is a good coreset of $P$ even though the size is the same.
Then, using the composition property, the authors proved the main theorem.
Also, the authors provided some experimental results.

### Strengths
- The problem is well-motivated.
Most of the previous results did not consider the noisy version of the problem while the real world data is often noisy.
Hence, I believe it is a good starting point to investigate this line of work.

- The paper is well-written.
Readers of all levels of expertise should be able to understand this paper.

### Weaknesses
- Most techniques are straightforward calculations.
I am not sure if there are any fundamentally new techniques introduced in this paper.

### Questions
na

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper evaluates the performance of the optimal solution $\hat{C}$ in the original data $P$ when dealing with noisy data $\hat{P}$. The analysis is based on the coreset technique, but the traditional measure for determining coreset quality is too strong when the data is noisy. To address this issue, the authors propose the AR-coreset, which restricts itself to a local sublevel-region and considers the cost ratio. This new measure allows the authors to obtain refined estimates of $\hat{P}$. The authors demonstrate that $\hat{P}$ is a (1+nd)-coreset and a (1+kd)-AR-coreset of $P$, meaning that $\hat{C}$ is a (1+nd)-approximation and a (1+kd)-approximation solution, respectively.

### Strengths
1, The motivation is clear. It is instructive to consider the local sub-optimal region of the solution space in the AR-coreset.

2, Utilizing the coreset technique to analyze the approximation of the solution is an interesting approach.

3, The lower bound of the approximation is also discussed.

### Weaknesses
1. The failure of the 'Err()' seems natural since it considers the worst case in the entire solution space.
2. This paper does not provide a detailed comparison of the proposed method with other existing analysis methods for clustering with noisy data.
3. This paper considers independent noise across dimensions, the real noise might be correlated. The experiment part is too simple, which considers only two datasets; also the dimensions are low (6 and 10).
4. The noise model and the assumptions make the result to be relatively narrow.

### Questions
1, the authors say ‘Intuitively, how good a center set we can obtain from $\hat{P}$ is affected by the number of sign changes. ’. Please provide more thorough interpretations.

2, Besides determining the quality of a coreset in the presence of noise, are there other potential applications of the proposed AR-coreset?

3, Can we conclude that the (near) optimal solution(s) are robust to noise for other problems?

4, In the definition of AR-coreset, why should we consider the ratio of the cost for a given center set to the minimum cost? What would happen if we only consider the denominator of $r_P(\widehat{c})=\frac{\operatorname{cost}(P, \widehat{c})}{\mathrm{OPT}}$.

### Soundness
2 fair

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies the problem of computing coreset for a dataset from its noisy perturbation. The paper considers the most apparent approach: compute a coreset for the perturbed version, and use it directly as a coreset for the original dataset. The paper showed the following results:
1. The coreset from the noisy dataset can be very bad for the original dataset, in terms of the relative error Err commonly used to measure a coreset's approximation quality.
2. The authors notice that the traditional measure (the relative error mentioned in 1) is too strong because it is the supreme over all possible center sets, while in practice people is more interested on how well the coresets can approximate costs for "not so bad" center sets. This motivates the authors to design a new relative error $Err_{AR}$ (which they call "approximate error ratio") that only takes supreme only over center sets that approximates the optimal solution. 
3. The authors show that this new definition of relative error can help give tigher approximation ratio estimation for center set computed on the coreset (obtained from the noisy dataset). In particular, a coreset $S$ with large Err can have much smaller $Err_{AR}$, which means a good approximate solution on $S$ will also have a small cost on the original dataset. While if we use $Err$ for estimation, the bound obtained is much looser.

### Strengths
I think the definition of $Err_{AR}$ is quite neat. The authors show that there is a strong separation between the traditional relative error $Err$ and their new measure $Err_{AR}$. I like this separation result in particular.

### Weaknesses
Although the authors claim the two assumptions (i.e. $O(1)$-balancedness and well-separation) are "mild" and only for "overcoming technical difficulty", I feel they are quite strong.
Also, one possibility is that the separation between $Err$ and $Err_{AR}$ is actually a result of these two assumptions. It would be great if the authors can show a quantitive analysis on the dependence between the separation and the two assumptions. For example, would it be possible that when the data become less balanced / well-separated, the two measures $Err$ and $Err_{AR}$ converge to each other. (My gut's feeling is the degree of well-separation could likely have a non-trivial effect on $Err$ / $Err_{AR}$). If that's the case, I feel this would somewhat strengthen the paper's result since $Err_{AR}$ can be viewed as a unifying measure that's tighter in extreme parameter ranges.

### Questions
There is no space between "$(k,z)$-Clustering" and the text following it. I guess the author should ad a `\xspace` after their macro for "$(k,z)$-Clustering"

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper shows how to construct coreset for clustering with noisy data.

### Strengths
- The paper introduces new measures that are used to construct coreset that guarantees eps approximations.
 
- The paper also presents a lower bound for 1-mean with noisy data. 

- The theory is backed by a good number of empirical evaluations on real dataset.

### Weaknesses
- Some intuition and relation between the claims in section 2 will be helpful.

- A formal algorithm, even in the appendix, will increase its impact.

### Questions
- Please give an example of how the number of sign changes affects the goodness/quality of centers. 

- Coreset size inversely proportional to n and d, in coreset measure is counter-intuitive. Comments?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
