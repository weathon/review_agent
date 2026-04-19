# Efficient Top-m Data Values Identification for Data Selection

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 5

## Abstract
Data valuation has found many real-world applications, e.g., data pricing and data selection. However, the most adopted approach -- Shapley value (SV) -- is computationally expensive due to the large number of model trainings required. Fortunately, most applications (e.g., data selection) require only knowing the $m$ data points with the highest data values (i.e., top-$m$ data values), which implies the potential for fewer model trainings as exact data values are not required. Existing work formulates top-$m$ Shapley value identification as top-$m$ arms identification in multi-armed bandits (MAB). However, the proposed approach falls short because it does not utilize data features to predict data values, a method that has been shown empirically to be effective. A recent top-$m$ arms identification work does consider the use of arm features while assuming a linear relationship between arm features and rewards, which is often not satisfied in data valuation. To this end, we propose the GPGapE algorithm that uses the Gaussian process to model the \emph{non-linear} mapping from data features to data values, removing the linear assumption. We theoretically analyze the correctness and stopping iteration of GPGapE in finding an $(\epsilon, \delta)$-approximation to the top-$m$ data values. We further improve the computational efficiency, by calculating data values using small data subsets to reduce the computation cost of model training. We empirically demonstrate that GPGapE outperforms other baselines in top-$m$ data values identification, noisy data detection, and data subset selection on real-world datasets. We also demonstrate the efficiency of our GPGapE in data selection for large language model fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a bandit-based method for identifying the top-data from a dataset in terms of data values. The main contribution is the incorporation of data features, specifically a non-linear structure assumed via a Gaussian process. The proposed algorithm is shown to have a PAC guarantees and is validated through a set of experiments on different applications.

### Strengths
It is an interesting combination of Gaussian processes and data valuation problem. The proposed solution addresses the problem where the linearity assumption is impractical for some applications, which is important. The algorithm is analyzed within the common bandit framework for its correctness and stopping time. A good amount of comprehensive experiments are included to demonstrate the effectiveness.

### Weaknesses
- It is a bit misleading in Theorem 4.2 and Proposition 4.3 that the same notation $\tau_\delta$ is called stopping iteration and query complexity differently.
- The query complexity / stopping time is lack of comparison with any prior works or lower bounds, making it a bit hard to evaluate the theoretical contribution.

### Questions
Minor issue: It seems not all of the experiments include error bars? What is the confidence level?

### Soundness
3

### Presentation
3

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
This paper studies top-m Shapley value identification problem, formulated as top-m arm identification in multi-armed bandit. While traditional multi-armed bandit problems estimate arm values use reward feedback only, this paper utilizes data features and uses the Gaussian process to model the nonlinear mapping from data features to data values. The authors prove theoretically that the Gaussian process based top-m nonlinear bandit algorithm finds an $(\epsilon, \delta)$ approximation to the top-m data values. Extensive numerical experiments on three different tasks are provided to demonstrate the model performance.

### Strengths
This paper improves on existing literature by removing the linearity constraint of arm feature and arm value mapping. Different aspects (correctness, stopping iteration and query complexity) of the theoretical analysis are carefully addressed and compared to the linear baseline. Computational costs are further reduced by querying marginal contributions only on small subsets. Extensive experiments are conduct to convince the reader that the method indeed outperforms existing methods.

### Weaknesses
Gaussian process is commonly known as computationally expensive when it comes to online updating. At each iteration t, new data points are decided to query and then GP is updated with the improved candidate set $J(t)$. It would be good to discuss what methods or tricks can be used to decrease the computational cost for this setting.

### Questions
• What is the benefit of using GP instead of directly using neural networks or other dimension reduction methods?     
    • What is the rate of diminishing return in Section 4.2 – is p here assumed to be of scale Cn (C is a constant)?

——————————After Rebuttal—————————Thank you for your response. I’m surprised to see the GP significant improvement without the newest tricks. I don’t have further questions and I’ll keep my current score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the problem of identifying the top-m data points with the largest Shapley Value (SV), which can be viewed as a top-m best arm identification problem. The paper models the relationship between the data feature and the value as a Gaussian Process (GP) and proposes an algorithm called GPGapE. The algorithm uses GP to update the estimation statistics and then designs which arm to sample based on GapE, a popular top-m best arm identification algorithm. Theoretical analysis of the sample complexity and PAC guarantee is provided, and extensive numerical experiments are conducted to showcase the performance of GPGapE. Interestingly, the paper makes use of the diminishing returns of ML models to propose a simplification of GPGapE, which achieves even better empirical performance.

### Strengths
1. Formulation and Algorithm: This paper combines the GP with the top-m best arm identification. The proposed GPGapE algorithm is novel, and the studied top data values identification problem has practical value.

2. Analysis: The sample complexity and the PAC guarantee are analyzed theoretically under the standard assumptions of both GP and top-m best arm identification. Extensive numerical experiments are provided to showcase the performance of the proposed algorithm

3. Computation Efficiency: A simpler version of GPGapE is also studied to reduce the computational complexity, where the paper makes equivalence arguments that the performance of GPGapE can be achieved using the simpler version. The numerical experiments interestingly show that the simpler version achieves even better performance.

### Weaknesses
1. GP: This paper inherits the weaknesses of the GP literature, which requires strong assumptions that the studied function is in RKHS. And all theoretical results are based on it. It remains a little unclear to me for general data value modeling, why GP is a good approximation that can provide the significant improvement shown in the numerical section. Justifications should be more detailed.

2. Sub-Gaussian: The authors also mentioned in the paper that some utility functions such as cross-entropy loss are not bounded, justifications on why sub-Gaussian noise is still a good assumption in this setting should be given and detailed.

3. Presentation: The presentation of the paper could be improved, notations are used without illustration and are causing ambiguity, e.g., in Theorem 4.1, when choosing C_{\delta,t}, what is B? the format of information gain \gamma_t is not explained, and how do you know the random stopping time \tau_\delta ahead of time to set the parameter \lambda in GP?

### Questions
1. For Theorem 4.1, Theorem 4.2, and Proposition 4.3, can we choose the same hyperparameter inputs to obtain all results simultaneously? Can we preset the inputs ahead of time or does it require tuning?

2. The current algorithm requires storing all historical data feature-marginal contribution pairs. Could you provide some insights on whether this is theoretically necessary? or is there a more memory-efficient implementation?

### Soundness
3

### Presentation
3

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
The authors consider the problem of identifying a small set of samples in a data set with high value for training.  Shapley value is used for measuring individual samples’ value, but is prohibitive to compute.  The authors propose that since only identifying the samples is necessary, not identifying their values, this problem can be viewed as a top-m multi-armed bandits problem to more efficiently solve it.  Furthermore, since training samples’ values may depend on rich features, similarity in (non-linear) feature space can be modeled using Gaussian processes.  The authors propose a top-m GP algorithm, including with a variant involving sampling rewards via computing Shapley values using small cardinality sets.  The authors also analyze regret and provide experiments.

### Strengths
- The motivating problem of identifying valuable samples in large data sets is challenging but important
- The problem of fixed confidence top-k Gaussian process bandits is interesting
- The presentation overall is good – I found the paper to be well organized and written
- The authors identified high-probability guarantees of their method and show for linear and RBF kernel functions nearly linear query complexity (in terms of # of samples)
- The use of GPs (compared to linear) for feature mappings is good.
- The authors include a number of experiments

### Weaknesses
### Major
- I think the technical problem setting (top-m bandit) is poorly motivated for the domain problem (Shapley value). If you are only going to select $m\ll N$ data points, why would you use Shapley value as the utility?  Shapley value is a fair allocation of joint/group value, which could be very different than what is achieved by groups of size $m$.  If it were easy to compute, I could see it potentially as a heuristic, but given the challenge in computing it, in some sense is it worth putting effort into using as a metric when it does not necessarily tell us how different groups of size $m$ will perform?
    - for top $m$, wouldn’t we either care about the top m data points in terms of marginal value to the primary utility function $U$ over which Shapley value is to be computed, eg if $U$ (1) is the overall utility function then $U({x_i})$ if the data points could be used/sold individually, or their value as a set A of size $m$, $U(A)$.  It seems implicit in your set up that you are looking at the top $m$ in terms of an alternate utility function, namely the Shapley value, which depends on both the actual utility of interest $U$ and set of all data $N$, looking at arms whose individual Shapley values are the highest.

### Minor
- Lines 313-320 there is an argument that since individual data points tend to have diminishing value with larger data sets, one can sample smaller subsets S to get bounds; certainly if (a) that holds uniformly and (b) the computational complexity increases with cardinality of $S$, that seems smart.  But is that diminishing returns empirically true for most samples or for samples on average?  Eg in Figure 1 is that for randomly sampled points?  Did you also check if that holds for the top $m$ points too, esp if $m\ll N$?  For instance, I don’t work in vision or NLP, but I could imagine that for some samples of rare classes, if there’s too few samples the ML model would not learn about those rare classes and those samples could have low Shapley value (not contributing to the learned common classes), but with larger $N$, those rare classes could be learned by the ML model leading to higher accuracy and subsequently higher Shapley value for those samples from rare classes.

- Proposition 4.4, Figure 1, and discussion between
    - Prop 4.4 statement includes an assumption $|\Delta_i^\ell| \leq \epsilon’, \ \forall \ell \in \{p,\dots,n\}$; it would be better to have that assumption (as in a theorem environment as Assumption 4.4) and discussed in relation to lines 313-323. 
    - Figure 1 caption says “Diminish return of adding new data points to the dataset when the size of the dataset increases.” But the x-axes are $l$, the subset size $S$.  “adding new data points to the dataset” means going from $N$ data points to $N+10$ or $N+100$.  In the main text, it mentions both “adding new data” (line 314) and (lines 319-320) that $|\Delta_i^l|$ is decreasing in $l$; both phenomena may happen and potentially be useful for speeding up methods.
    -  The Prop 4.4 statement (at least per sample $i$) $|\Delta_i^\ell| \leq \epsilon’, \ \forall \ell \in \{p,\dots,n\}$ seems to be combining both  “adding new data points to the dataset” and $|\Delta_i^l|$ is decreasing in $l$, as it is bounding $|\Delta_i^l|$ in an absolute sense seemingly implicit that for sufficiently large $N$, not only will $|\Delta_i^l|$ be small (small enough one could use the bound as a surrogate $\epsilon$ for the top-m threshold) but also that (N fixed) there will be a subset cardinality $l=p$ for which all larger cardinalities will not have a larger value.


     – also $\exists p \in N$  should there be a $\forall i\in N$ there?  The conclusion involves bounding all $\varphi_i$ for all $i\in N$ 



### Very minor (Typos/etc.)
- for subset $S$ cardinality, $\ell$ can be easy to read as super/sub script than $l$
- Figure 1 caption “Diminish return”
- Figure 7 – the curves are quite crowded; hard to tell, but it might be a little easier to read if the curves are transparent

### Questions
- Can you discuss what technical challenges there were for obtaining the high-probability guarantees and bounding query complexity, given prior work like Reda et al for the linear setting and past work on Gaussian process bandits?

### Soundness
3

### Presentation
3

### Contribution
2
