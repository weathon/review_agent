# Matrix Sketching in Bandits: Current Pitfalls and New Framework

- Decision: Reject
- Scores: 5, 6, 6, 5

## Abstract
The utilization of sketching techniques has progressively emerged as a pivotal method for enhancing the efficiency of online learning. 
    In linear bandit settings, current sketch-based approaches leverage matrix sketching to reduce the per-round time complexity from $\Omega\left(d^2\right)$ to $O(d)$, where $d$ is the input dimension. Despite this improved efficiency, these approaches encounter critical pitfalls: if the spectral tail of the covariance matrix does not decrease rapidly, it can lead to linear regret.
    In this paper, we revisit the regret analysis and algorithm design concerning approximating the covariance matrix using matrix sketching in linear bandits. 
    We illustrate how inappropriate sketch sizes can result in unbounded spectral loss, thereby causing linear regret. 
    To prevent this issue, we propose Dyadic Block Sketching, an innovative streaming matrix sketching approach that adaptively manages sketch size to constrain global spectral loss. 
    This approach effectively tracks the best rank-$k$ approximation in an online manner, ensuring efficiency when the geometry of the covariance matrix is favorable. 
    Then, we apply the proposed Dyadic Block Sketching to linear bandits and demonstrate that the resulting bandit algorithm can achieve sublinear regret without prior knowledge of the covariance matrix, even under the worst case. 
    Our method is a general framework for efficient sketch-based linear bandits, applicable to all existing sketch-based approaches, and offers improved regret bounds accordingly.
    Additionally, we conduct comprehensive empirical studies using both synthetic and real-world data to validate the accuracy of our theoretical findings and to highlight the effectiveness of our algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors proposed a new sketching method called dyadic block sketching. This sketching method does not need a fixed sketch size as other sketching methods do, and can adaptively adjust the sketch size based on the error. Using this Dyadic Block Sketching, the authors proposed a bandit algorithm called DBSLinUCB which modifies the traditional OFUL algorithm a bit and achieves an improved regret bound compared to the traditional method of [Kujborskij et al., 2019].

### Strengths
The authors proposed a novel and flexible algorithm, DBS, which optimizes its sketching size as the number of samples increases. Adding a doubling trick to the sketching will not be easy, so **if their result is true**, I want to give some points on the novelty, soundness, and significance of this algorithm. 

Not only that, they also actually improved the regret bound of the bandit algorithm from the previous work [Kuzborskij et al.., 2019] to a more flexible result. Previously, the result relied on the term called 'spectral error', which is relatively vague and depends on a fixed sketch size. According to their literature search, there was almost no improvement after [Kuzborskij et al., 2019] for 5 years, and they finally made a more flexible and tighter result than before, **if their results are true.**

### Weaknesses
Overall, because of clarity issues, I cannot trust the soundness of their result now. 

1) Their presentation about the lower bound is poor. I don't understand what the authors want to say in Theorem 1. Please check the 'Questions' section below. 

2) Also, Algorithm 1 is the core of this paper, but it is very hard to understand from many perspectives and is not clearly written. Please check the 'Questions' section below. 


3) Clarity problems

3-1) Figure 1 - It is really strange that the regret is much larger than the number of timesteps. After I checked the experimental details, they set the arm set $a \sim N(0,I_d)$ and this is the reason why they had a large regret. Also, this means they reported only one experiment, not the average result of repetitions (If not, all their regret should fluctuate a lot). I know that experiments are not the main focus of bandit papers, but this is not a good tradition.

3-2) In Lemma 1, it would be great if they could add the 'size' of each $X_i \in \mathbb{R}^{n_i \times d}$ so that the readers could understand how the submatrices look like. I was a bit confused in Lemma 1 since they didn't define the concatenation $[X_1, X_2]$ before.
2-3) In Section 2.1, they should add their SVD computations, or mention their efficient way of updating parameters $S^{(t)}$ and $M^{(t)}$

4) It seems like it is an improvement from the previous work [Kuzborskij et al., 2019], but I doubt that their result has $(R+L\sqrt{\lambda})$ term in their result. The LinUCB algorithm has a regret bound of $Rd\sqrt{T}$. The form looks quite problematic if $R$ and $L$ are in different scales. I know that the previous work also had this issue, but it is still true that this kind of (variance + arm set scale) is not a good thing in practice, especially when $R<<H$. See Questions section for my doubt on $\lambda$.

### Questions
1) Several things I can't understand in Theorem 1

1-1) If I understood correctly, the arm is drawn randomly from a 'fixed' arbitrary distribution over a set of $r$ arms. Now in this case, what is the point of the sketching? The regret above (by Kuzborskij et al) is the regret of the sketch-based linear bandit. This fixed random sampling is not the algorithm by Kuzborskij et al (2019). What is the connection?

1-2) Also, no algorithm could achieve a 'square regret' $\Omega(T^2)$. It is impossible since the worst possible regret is $T$. I can barely guess that the authors want to mention the 'upper bound' of the expected regret using FD is $\Omega(T^2)$, but it is not super surprising, since it is an 'upper bound'. For example, in $K$-armed bandit, the worst case regret bound is of $\sqrt{KT}$, and if $K=T^2$, the regret 'upper bound' is $T^{3/2}$ but it is not surprising since it is just an upper bound. 

2) I failed to understand Algorithm 1. Here are the issues I found. 

**2-Major**

- If I understood correctly when the algorithm updates, the B[i].length should increase. The term 'rank' is the block's rank, actually. How could a rank be larger than the length? It feels like for $d_1 \times d_2$ dimensional rank $r$ matrix, you are saying 'when $d_1 < r$'.
- What is the second condition in line 276, 'The sum of sketch rows stored in blocks should be less than $d$'? Rows are vectors and $d$ is a scalar.  
- Similar to this question, How could a block contain the size $\epsilon l_0$ while doubling the length? In my intuition, each $x_t$ corresponds to the actions chosen by the LinUCB algorithm. Whatever your sketch is, you add $B[i].size +=\|\|x_t\|\|^2$. How can you make the length be doubled while maintaining the same B[i].size for each block?
- The authors mentioned three key invariants that should be maintained - (B[i].length>rank, $B[i].size< l_0 \epsilon$, and the strange condition above). However, in line 8 of Algorithm 1, they used 'AND' logic. Is it right?

**2-Minor**

- $l_0$ is 'sketch size' and you are using the term $B[i].size$ differently. It is very confusing. 
- Also the term 'rank' is very confusing. I know they defined rank around line 317, but it's hard to find what it is.
- What is Line 12? The authors should 'define' how each sketching algorithm works, or at least direct readers to a certain part. 
- At least use a different font or something for Sk. 
- Also, I was about the point out the computational inefficiency of lines 16-18 of Algo 1. I just found that you mentioned in lines 324-326, but it is not intuitive at all. The main point that this paper sells is the computational efficiency, and this part looks like updating all the sketch matrices for each timestep which is not efficient. 

3) Also, in Section 2.1, I think they should add some remarks about the rank-$l$ SVD. This algorithm needs rank-$l$ SVD for each time (For Eq. (3), right?) and they intentionally omitted the cost in lines 157-161. As the previous work did, they should add a remark to avoid any doubts about their honesty.

4) The result in Theorem 3 seems much more intertwined in terms of $\lambda$, much more complicated than the previous work of Kuzborskij. Can authors state whether their results are better than Kuzborskij or not in terms of $\lambda$?

5) How large is your error parameter? Is there no assumption needed on $\epsilon$? As we all know, most of the researchers use the character $\epsilon$ as a 'very small quantity smaller than 1'. However in this work, in their experiments, they used $\epsilon = 2000$, much larger than their dimension $d=100$. Are you sure that this $\epsilon$ does not hide any important terms, such as dimensionality?



Overall, I encourage the authors to further refine their paper to bring out its full potential if they believe it has significant contributions to offer.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work explores the limitations of current sketching techniques in online learning, particularly within linear bandit settings. Existing methods using matrix sketching suffer from high spectral error, leading to potential linear regret when the covariance matrix's spectral tail does not decay rapidly. The authors propose a novel approach, Dyadic Block Sketching, which adaptively adjusts sketch size to control spectral error. This new technique aims to achieve sublinear regret without prior knowledge of the covariance matrix, presenting a potentially generalizable framework for improving efficiency and regret bounds in sketch-based linear bandits.

### Strengths
- The topic of balancing computational efficiency with accurate regret minimization in high-dimensional settings is of broad interest
- The writing is mostly clear

### Weaknesses
- Magnitude of the technical novelty and practical impact of the contributions

### Questions
- Did you run any additional experiments on more diverse real-world datasets to better investigate the method's adaptability to varying spectral properties? If so, what did you observe? 
- Given the increased complexity of managing blocks and adaptive sketch sizes, how does Dyadic Block Sketching compare in terms of computational complexity to simpler, single-scale sketching approaches in real-world applications?
- The paper suggests improved regret bounds when favorable conditions are met. Could you elaborate on how practical these improvements are in some examples of real-life scenarios?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies matrix sketching within the context of linear bandits, aiming to reduce time complexity.

The authors demonstrate that Frequent Directions (FD), a previously utilized matrix sketching method, could lead to linear regret. They introduce a dyadic block sketching method where the sketch size is adaptively adjusted based on the parameter $\epsilon$. The main theorem asserts that the proposed algorithm achieves favorable regret bounds, and experimental results highlight its effectiveness and efficiency.

### Strengths
The authors demonstrate that Frequent Directions (FD), a previously utilized matrix sketching method, could lead to linear regret. They introduce a dyadic block sketching method where the sketch size is adaptively adjusted based on the parameter $\epsilon$. The main theorem asserts that the proposed algorithm achieves favorable regret bounds, and experimental results highlight its effectiveness and efficiency.

### Weaknesses
However, there are several weaknesses:

1. The comparison with previous work is overly simplistic and lacks sufficient detail. In previous work, $l$ represents a trade-off between time complexity and regret. This work uses $\epsilon$. The authors should provide a comprehensive analysis of the trade-off (including time complexity at each step, memory costs, assumptions, and regret) to elucidate the advantages of the new method.
2. Although a regret lower bound is established for the previous algorithm, it is unclear how the proposed algorithm performs under the worst-case scenario constructed.
3. The algorithm searches for parameters in the experiments. However, the main theorems do not hold for these parameters. It would be beneficial if the authors could also present experimental results that consider scaling the parameters.

Additionally, in Theorem 1, the stated regret is $T^2$, which is unusual since the worst-case scenario typically scales linearly with $T$. Could the authors clarify this discrepancy?

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors consider stochastic linear bandits when the dimension $d$ is high enough that per-round computational complexity is of concern to the user.  Traditional methods have high per-round computational complexity with respect to the dimension size $d$ due to covariance matrix estimators used.  Sketching based variants have been proposed to alleviate the computational complexity, but require pre-selecting the sketch size $l$ which can lead to high-regret (there is a regret bound term related to error from sketching (with size $l$) that can exhibit horizon dependence).  The authors propose a new sketching method that adaptively selects the sketch size subject to a parameter controlling the sketching error.  They derive regret bounds and show experiments demonstrating that it can empirically achieve a superior tradeoff of regret and computation.

### Strengths
- The general topic of improving trade-offs in per-round computation and regret in online learning is important. 

- The authors clearly lay out an important issue with standard stochastic linear bandit methods with the dimension is large enough that per-round computation can be a concern. 

- The proposed method is interesting, using a decomposition property to control the overall spectral norm error and then carefully designing block sizes adaptively.  It is integrated using standard matrix sketching methods (FD and RFD) to form new LinUCB variants for which the authors obtain regret bounds.  

- The presentation overall is good.  

- Experiments are included on synthetic and real-world data, showing that the method can perform competitively with non-sketching methods while reducing per-round computation significantly.

### Weaknesses
### Major 

- My biggest concern is with regards to prior works.  There is a brief related works discussion on sketching (no recent works) and no mention of similarities or differences in the methodology section.  Is this Dyadic Block Sketching technique completely new?  Three classes of matrix sketching methods are described none of which seemed (to me) to be particularly similar.   

    - However, there is at least one (slightly more recent) paper that is not cited but sounds fairly relevant: a 2016 sigmod https://dl.acm.org/doi/abs/10.1145/2882903.2915228 paper which proposes a “Dyadic Interval framework” that sounds similar—it is also based on the decomposability property, it also involves specifying an error hyper-parameter, has a logarithmic number of ‘levels’, and there is a check of a L2 norm based block size that triggers updating; they also combine it with the “Frequent Directions” method.   

    - The citations included in that section are old, with the most recent paper in that section of related work being a 2014 review paper.   

 

- Another concern I have is with the role of $\epsilon$.  I think it is quite interesting to go from fixing a sketch size and possibly incurring large regret to fixing an error bound in part of the estimation scheme.  Given there is a different hyperparameter,  it would have been better to have more discussion and also some experiments showing how the choice of $\epsilon$ impacts the spectral norm errors, the regret, and the computation.  In the experiments a seemingly large constant $\epsilon=2000$ is used for two sets of experiments, $\epsilon = 1000$ in another, but I do not have a sense where those came from.  For sketch size, by contrast, the user knows $d$ and at least for a target computational budget could probably fairly easily select a sketch size that would fit well within that (with the caveat being of course it might incur large error).          

    - 5.1 Matrix approximation experiment – only one value of $\epsilon$ is used (also computation time and sketch sizes method used are not reported for that $\epsilon$); 

    - 5.3 experiment on MNIST, “In contrast, DBSLinUCB matches the performance of OFUL by adaptively adjusting the sketch size to the near-optimal value of l = 200 while being significantly faster than OFUL.” empirical regret is shown so clearly is close to OFUL and time is shown for this experiment so is clear is close to SOFUL(l=200), which is suggestive sketch sizes would be similar (around 200) but did  you actually check that?  Or are you basing that on the total run-time? 

    - in experiment 5.3, the $\epsilon$ was changed compared to previous experiments? Why?  How was this value chosen? 

 

 

### Minor 

- the computational complexity (and relatedly realized sketch sizes) are not reported.  In the first experiment the spectral norm error is shown (so tradeoff FD makes is clear) but no mention of the extent to which computation time is traded off.   

- in section 5.1, is there a reason FD does better for small number of rows than DBS-FD?   

- theorem 1’s statement is poorly written for a formal theorem.  It also contains a notable typo “$\Omega(T^2)$” for regret bound

### Questions
(several questions are included as part of concerns mentioned above)

### Soundness
3

### Presentation
2

### Contribution
2
