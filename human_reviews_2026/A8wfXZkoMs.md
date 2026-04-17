# Ads that Stick: Near-Optimal Ad Optimization through Psychological Behavior Models

- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
Optimizing the timing and frequency of advertisements (ads) is a central problem in digital advertising, with significant economic consequences. Existing scheduling policies rely on simple heuristics, such as uniform spacing and frequency caps, that overlook long-term user interest. However, it is well-known that users' long-term interest and engagement result from the interplay of several psychological effects (Curmei, Haupt, Recht, and Hadfield-Menell, ACM CRS, 2022).

In this work, we model change in user interest upon showing ads based on three key psychological principles: mere exposure, hedonic adaptation, and operant conditioning. The first two effects are modeled using a concave function of user interest with repeated exposure, while the third effect is modeled using a temporal decay function, which explains the decline in user interest due to overexposure.
Under our psychological behavior model, we ask the following question: Given a continuous time interval $T$, how many ads should be shown, and at what times, to maximize the user interest towards the ads?

Towards answering this question, we first show that, if the number of displayed ads is fixed to $n$, then the optimal ad-schedule only depends on the operant conditioning function. Our main result is a quasi-linear time algorithm that, given the number of ads $n$, outputs a near-optimal ad-schedule, i.e., the difference in the performance of our schedule and the optimal schedule is exponentially small.
Our algorithm leads to significant insights about optimal ad placement and shows that simple heuristics such as uniform spacing are sub-optimal under many natural settings. The optimal number of ads to display, which also depends on the mere exposure and hedonistic adaptation functions, can be found through a simple linear search given the above algorithm. We further support our findings with experimental results, demonstrating that our strategy outperforms various baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper performs a theoretical quantative study of a model of ad fatigue, which is important in advertising, by considering the number of ads shown to a user and the effect of ad recency on the user's inclination to positively interact with an ad. A bi-level algorithmic framework is established for computing an optimal ad schedule given a fixed number of ads as the 'inner' problem, and computing the optimal number of ads as the 'outer' problem.

Despite a few weaknesses, the work is enlightening and interesting.

### Strengths
1. The phenomenon is interesting and important.
2. A fresh deviation from all this "attention is all you need" trend, where a phenomenon is actually studied in depth.
3. The framework is simple, without many 'moving parts'. 
4. The bi-level idea is nice and practical, if indeed we aim to plan an ad schedule for each user separately.
5. The results are insightful. For example, it's not trivial that the optimal schedule tends to "show a lot only at the beginning and the end of the time-frame" only when the fatigue parameter is very close to 1, and otherwise close-to-uniform strategy appears optimal (Figure 1 (a))

### Weaknesses
1. In practice, privacy concerns don't let you, in many cases, know when you're showing an ad to the same user. So such an idea is practical mainly in "closed gardens" where all users are always logged in and we know who they are. Most advertising marketplaces are not like that.
2. Code to reproduce the experiments is important, and any claim to publish it is missing from the paper.
3. The function B(i) is concave and increasing, but we don't know if it's the right model. There are many ways to model a concave and not necessarily increasing function. It may eventually asymptotically be concave and increasing, but it might be the case that the effect of its shape before asymptotic behavior kick in is not trivial.

### Questions
1. Will the code to reproduce the results be published?
2. Why do you believe that a concave and increasing function on the entire domain is a reasonable model to gain insights?
3. How do you propose fitting the parameters of such a model if we want to actually use it in an advertising platform?

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
A reward model for advertising is derived from the marketing literature. 
In the context where we have to expose a user during a known interval, an optimization algorithm is derived. 
Some qualitative insights are provided on the behavior of the solution.

### Strengths
The authors did an excellent job at introducing a reward model by bridging results from other fields. 
I found the article very informative and inspiring.

### Weaknesses
Main weakness : the technical sections of the paper should be improved
* the graphs cannot be read properly (no color)
* discussion and explanation on the baselines in the main text would be welcome
* the proof of the statements are sometime two hand waving, the mathematical writing should be improved
* I am not sure the algorithm itself is a very technical contribution (mostly basic maths). While this is not a reason for rejection or anything, I believe the proof sketch should be better articulated

### Questions
* Can you provide an updated version of figure 1?
* could you provide a summary of the proof that would clearly delineate what is easy and what is hard in the derivations?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a principled algorithmic framework for scheduling advertisements by modeling three psychological effects: mere exposure, hedonic adaptation, and operant conditioning. The authors formulate ad scheduling as an optimization problem, specifically minimizing a loss function $L(\overline{t})$ that depends on the temporal decay of user interest.

1. A novel psychological model for ad scheduling.
2. A quasi-linear time algorithm (Algorithm 1 / Algorithm 5) that finds a near-optimal schedule with provable, exponentially small approximation error.
3. Theoretical analysis proving the uniqueness of the optimal solution (via strict convexity) and deriving a novel recursive structure for the optimal ad timings.
4. Experimental validation showing that the proposed schedule outperforms heuristics like uniform spacing by over 10% in realistic parameter regimes (e.g., high $\delta$).

Having thoroughly reviewed the main paper and all appendices (A-F), I find the work to be technically sound, novel, and practically relevant. The theoretical claims are correct. My main concerns are related to presentation, as much of the core theoretical justification is relegated to the appendices, making the main paper difficult to follow and appear incomplete.

### Strengths
1. **Well-Motivated Problem:** The paper addresses a practically important problem with clear economic implications. The grounding in established psychological principles (mere exposure, hedonic adaptation, operant conditioning) is a significant strength and is well-modeled.
2. **Solid and Complete Theoretical Contributions:** The main theoretical results are non-trivial and, upon review of the appendices, are fully proven.
    - **Uniqueness (Thm 4.2):** The proof of strict convexity and a unique minimum is correct and detailed in Appendix A.
    - **Recursive Structure (Thm 4.4, Lemma 4.5):** The recursive formulas for $T_i$ are proven in Appendix B. I have numerically verified these formulas.
    - **General $a \neq 1$ Case (Appendix D):** My initial concern about the analysis being limited to $a=1$ was unfounded. Appendix D provides a complete and correct analysis for the general (boundary) case $a \neq 1$, deriving the analogous recursive relations (Lemma D.4) and the general solution condition (Lemma D.5).
    - **Approximation Guarantees (Cor 4.8, Thm D.9):** The exponentially small approximation error $|t_i^* - t_i| \le 1/2^n$ is proven for the $a=1$ case in Appendix C and generalized for all $a$ in Theorem D.9.
3. **Practical and Efficient Algorithm:** The proposed algorithm (detailed in Algorithm 1 and Algorithm 5) is efficient (quasi-linear time) and implementable. The code provided was functional and reproduced the experimental claims.
4. **Validated Experimental Claims:** All experimental claims in Section 6 are verified. The results showing uniform spacing is optimal for small $\delta$ and endpoint-concentration is optimal for large $\delta$ align with the theory and provide valuable, actionable insights.

### Weaknesses
The paper's primary weaknesses are not technical but related to presentation and clarity. The main paper, read in isolation, is confusing and appears incomplete.

1. **Critical analysis deferred to Appendix:** The paper is not self-contained. The analysis for the general $a \neq 1$ case (Appendix D) and the proofs for all key theorems (Thm 4.2, Thm 4.4) and approximation bounds are entirely in the appendix. While this is common, the $a \neq 1$ analysis is a _core_ part of the algorithmic contribution (Algorithm 1) and its omission from the main text makes the paper's claims seem unsupported.
    - I would suggest to add theorem statements for the general $a \neq 1$ case in the main paper. Also, add 1-2 paragraph proof sketches for Theorems 4.2 and 4.4 to the main text.
2. **Algorithm presentation is opaque:** The conditions in Algorithm 1 are opaque and unmotivated. The appendix (Algorithm 5 and Appendix D.3) reveals these are the boundary-checking conditions for finding the correct $a$.
    - I would recommend to add a remark after Algorithm 1 explaining the purpose of these conditions, as detailed in Appendix D.3.
3. **Issues pertaining to confusing notation:** The use of "$n+1$ ads" with indices 0 to $n$ is non-standard and causes confusion. The relationship between the paper's $n$ and the code's $n$ is also implicit.
    - Add a notation table at the start of Section 2. Explicitly define $T_1$ in Section 4.2 and its relationship to $T_a$.
4. **Limited experimental scope:** The experiments are entirely synthetic. While they effectively validate the algorithm's properties, the paper would be stronger if it addressed how the model parameters ($\delta$, $B(\cdot)$) could be calibrated using real-world user data.
    - Having a discussion on parameter estimation and acknowledge the limitation of not having real ad-serving data would help improve the paper.

### Questions
My initial Questions were largely resolved by the appendix in the supplementary material.

- **Q: How sensitive are the results to misspecification of $\delta$?**
    - This question remains open. A sensitivity analysis would be a valuable addition, as $\delta$ is unlikely to be known perfectly in practice.

This paper provides solid theoretical contributions and a practical, efficient algorithm. With the appendices, it is clear the technical work is sound. The required revisions are purely focused on improving clarity and presentation to ensure the paper's contributions are properly understood by the reader.

### Soundness
3

### Presentation
2

### Contribution
3
