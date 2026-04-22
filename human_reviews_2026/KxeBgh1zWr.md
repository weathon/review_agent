# Curse of Slicing: Why Sliced Mutual Information is a Deceptive Measure of Statistical Dependence

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 8

## Abstract
Sliced Mutual Information (SMI) is widely used as a scalable alternative to mutual information for measuring non-linear statistical dependence. Despite its advantages, such as faster convergence, robustness to high dimensionality, and nullification only under statistical independence, we demonstrate that SMI is highly susceptible to data manipulation and exhibits counterintuitive behavior. Through extensive benchmarking and theoretical analysis, we show that SMI saturates easily, fails to detect increases in statistical dependence (even under linear transformations designed to enhance the extraction of information), prioritizes redundancy over informative content, and in some cases, performs worse than simpler dependence measures like the correlation coefficient.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Theoretical properties of sliced mutual information (SMI) are studied, suggesting strong limitations of SMI in application scenarios in which it has been employed in the past, e.g., the Deep InfoMax setting. Furter, discrepancies between SMI and conventional mutual information are illustrated, that is, SMI between deterministic Gaussian variables is bounded while MI is infinite. For increasing dimensionality SMI approaches zero, while again, MI is approaching infinity. The theoretical analysis has been supported with simulation experiments.

### Strengths
- The paper highlights severe limitations of SMI that are relevant for practitioners.
- Theoretical results for multivariate Gaussian variables have been derived which illustrate the counterintuitive behaviour of SMI compared to MI.
- The main theoretical result derived in Lemma 4.1 has been supported with sufficient experimental evidence.
- The usecase of Deep InfoMax has been investigated at the example of a small image dataset.

### Weaknesses
While the discrepancy between MI and SMI is well-illustrated, the claims regarding the Deep InfoMax principle are not as strongly supported by the experiments:
- The description provided in D.2 explains how to obtain the baseline but more details regarding the implementation of SMI are needed.
- The results are only obtained for a relatively small image dataset (MNIST). To support the strong impact statement made by the authors, more complex datasets such as CIFAR10, or similar ones should be analyzed.

 Some experimental details are missing: 
- How exactly was $SI_k$ implemented for the Deep InfoMax experiment? 
- How many seeds did the authors consider to confirm their results? 
- Why is $k=1$ for the KSG estimator? Typically a higher $k$ such as 5 or 7 is more stable.

Minor: 
- In their discussion of the Deep InfoMax experiments, the authors should mention that, e.g., for invertible $f$ we get that $I(X;f(X))$ is infinite and we can only compute it if $f$ fulfills certain properties that avoid this behaviour. 
- Punctuation missing in Lemma A.3.

### Questions
Questions:
- Are there any usecases for which the authors would recommend to use SMI instead of MI?
- Is the proof of Lemma A.4 just recited from 41, or are there any new contributions in it?
- Do the works that used SMI for Deep InfoMax [22,23] employ any architectural restrictions or similar to avoid collapse?

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
This paper presents an analytical and empirical critique of Sliced Mutual Information (SMI), a popular scalable alternative to mutual information (MI) used in various works on high-dimensional statistical dependence estimation and deep learning analysis. The authors argue that despite its recent adoption, SMI exhibits several fundamental flaws that make it unreliable as a measure of dependence, including the following:
1) SMI rapidly saturates even for simple synthetic problems, failing to reflect true increases in dependence.
2) SMI prioritizes redundant or repeated information rather than informative content.
3) Although promoted as dimension-robust, SMI actually decays to zero asymptotically in high dimensions.
4) SMI can increase under deterministic mappings, unlike MI.

Through theoretical analysis and experiments with synthetic data, the paper demonstrates these deficiencies and shows that using SMI as a replacement for MI is, at the very least, problematic.

### Strengths
Thank you for the paper.  It was an interesting read.

The paper has both detailed theoretical analysis with several key examples (such as closed-form analysis for certain classes of Gaussian variables) and a large number of synthetic experiments to validate their theoretical findings and conceptual contribution.  

The paper provides an appropriately critical assessment, in a timely and important way.

The paper is well written and nicely organized.

### Weaknesses
The following weaknesses are suggested, but they aren't (in my mind) especially significant.  

The claim that SMI fails to detect increases in dependence even for linear transformations that enhance information extraction should be clarified to have been shown for specific cases, not necessarily universally.  

A specific real-world example would be interesting.  

The math in this paper is detailed/hard and probably beyond many readers, but not sure that can be helped.

### Questions
Less questions, more summarizing comments:

My preliminary assessment is that this a somewhat niche topic (papers on “sliced mutual information” seem fairly sparse, according to Google scholar), and so I might be less inclined to accept the paper.  However, some of the papers have appeared in prominent conferences (such as NeurIPS), and as such it seems like an important outcome to have the full story available for other researchers to examine.

Perhaps my main question is, given the paper's results, whether the authors feel that SMI currently has any applications settings where it would still seem redeemable, or if it is back to the drawing board. Though that is more of a personal interest question -- I'm not sure I would want to suggest the authors should be required to make a strong statement in the paper, when I think their analysis lays out concrete issues clearly.

### Soundness
4

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
4

### Summary
This paper discusses the shortcomings of Sliced Mutual Information (SMI) as a tool for measuring statistical dependence in high-dimensional settings. The authors show that SMI can saturate as the correlation between random vectors increases and decreases asymptotically as dimension grows. The authors validate the theoretical findings with synthetic experiments.

### Strengths
- The paper is well-written
- The paper presents mathematical proofs of SMI’s limitations in the simple Gaussian scenario
- The paper highlights major flaws of a novel dependence measure

### Weaknesses
- The theoretical results are limited to a Gaussian setting (Lemma 4.1). This does not prove anything for more complex scenarios
- Since the theoretical part is limited to the Gaussian setting, one would expect the experimental results to be comprehensive of significantly complex scenarios for which the authors did not provide a theoretical contribution, to show that the paper contributions hold true for many possible cases. However, the experimental results could be improved

### Questions
- In the numerical experiments, are you considering a finite-data regime or infinite-data regime? I did not understand it from the paper, but it is well-known that, for instance, MI estimators perform differently depending on the regime. In any case, how does SMI perform in the other regime that you did not consider? 
- How do you explain the saturation phenomenon of SMI for non-Gaussian scenarios?
- Since you compare correlation coefficients, SMI, MI, and copula in the initial figure of the paper, why did you not include any other observation or comment on these measures? Can you provide a paragraph of comparison?
- It appears that high dimensions pose a theoretical problem with SMI in the Gaussian case. However, SMI was used specifically for high-dimensionality problems. So, could it be that SMI does not have the same saturation problem for high dimensions in different scenarios? Did you find any settings in which the dimensionality was not a problem for SMI, and that are not reported in the paper?
- What does << mean in your case (line 114)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper gives a critical analysis of a cheap and intuitive proxy measure for statistical dependence called "sliced mutual information" (SMI). 
Although the simplicity of SMI has fueled its growing popularity, this study finds that the name is deceptive and that SMI has a number of drawbacks that make it inappropriate as a proxy for statistical dependence measures like Mutual Information: SMI saturates as dependence grows, SMI is biased towards redundancy, SMI decays to zero in high dimensions. The paper builds analytic examples and uses synthetic data to demonstrate the issues, and shows that simple workarounds do not fix the problem.

### Strengths
- The paper was well structured. SMI and its defects were clearly described, with intuitive experiments supporting each result. 

- While most of the arguments were supported by analytical counter-examples, I was happy to also see comparisons with recent synthetic data benchmarks used in the MI estimation literature. 

- I appreciate that the paper went beyond showing examples where SMI gives counter-intuitive results, and also showed that *optimizing* SMI leads to poor results. 

- The deficiencies of SMI (like redundancy bias) were not very surprising to me, but the curse of dimensionality effect was quite a bit stronger than I expected. (I assumed that as dimensionality grows, it would be hard to find a "good slice", but it seems that even analytically integrating slices leads to decaying SMI!)

### Weaknesses
I thought the critique was straightforward and clear. The only question in my mind is the broader significance of these results. I am familiar with recent neural MI estimation literature, and I confess I had not seen any mention of SMI (though I think I recall a reference to it) so I was a little surprised to see it described as a "popular" approach. Nevertheless, the citations don't lie, and a decent number of papers in top venues are studying something that is (as I am convinced by this paper) a dubious measure. Therefore, even though I don't think this has the broadest significance for the field, the critique should be at least as visible as the flawed approach.

### Questions
- I didn't go back to study the IB paper that you cited that optimized SMI. Your result suggests that it shouldn't work, is there any explanation on how they were able to show reasonable results to publish with this method? 

- One small improvement for a reader like myself would be to situate the SMI literature with respect to the neural MI estimation literature. I just assumed that it was clear that neural MI estimation methods, like neural everything else, was the clear winner. Are there applications / properties / uses cases which led people to prefer SMI over neural methods? I assume it's just for computational simplicity?

- One other thought that might extend the impact of this work is to discuss whether there are connections to other "sliced" estimators, like sliced score matching. Even the popular Hutchinson trace estimator could be considered a sliced estimator.

### Soundness
4

### Presentation
4

### Contribution
3
