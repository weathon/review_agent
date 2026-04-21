# Clustering on Skewed Cost Distributions

- Avg Score: 5.60
- Decision: Reject
- Scores: 5, 8, 5, 5, 5

## Abstract
In this paper, we tackle the problem of $(k,z)$-clustering, a generalization of the well-known $k$-means, $k$-medians and $k$-medoids problems that is known to be APX hard, i.e., impossible to approximate within a multiplicative factor of $1.06$ in polynomial time for $n$ and $k$ unless P=NP. Due to the APX-hardness, the fastest $(1+\varepsilon)$-approximation scheme proposed by Feldman et al. (2007), exhibits a run time with a polynomial dependency on $n$, but an exponential dependency $2^{\tilde{\mathcal{O}}(k/\varepsilon)}$ on $k$. We observe that a $(1+\varepsilon)$-approximation in truly polynomial time is feasible if the data sets exhibit sufficiently skewed distributions. Indeed in practical scenarios, data sets often exhibit a heavy skewness, leading to the overall clustering cost disproportionately dominated by a few clusters. We propose a novel algorithm that adapts the traditional local search technique to effectively manage $(s, 1- \varepsilon^{z+1})$-skewed datasets with a run time of $(nk/\varepsilon)^{\mathcal{O}(s+1/\varepsilon)}$ for discrete case and $\tilde{\mathcal{O}}(nk) + (k \log n)^{\tilde{\mathcal{O}}(s+1/\varepsilon)}$ for continuous case. Our method is particularly effective with Zipfian distributions with exponent $p>1$, where $s = \mathcal{O}\left(\frac{1}{\varepsilon^{(z+1)/(p-1)}}\right)$.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a method for obtaining a $(1 + \varepsilon)$-approximation to the $(k, z)$-clustering problem. This is known to be APX-hard and the authors therefore constrain their focus to the case where a dataset exhibits strong skewness (some clusters account for a larger portion of the cost than others). In this setting, the authors show that one can obtain an a $(1 + \varepsilon)$-approximation in time $O( poly(k, n)^{1/ \varepsilon} )$. They conclude by experimentally verifying that this method indeed finds a solution of smaller cost than the standard $k$-means and $k$-median techniques.

### Strengths
This paper has several clear strengths.

First, the technical matter itself is well-written and easy to follow. The algorithms are clearly laid out and the overview of the proof is clear. I particularly appreciate that the related work is woven into the narrative rather than appearing as a brick wall on one of the pages of the paper.

Second, the theoretical ideas are interesting. The authors show that one can find the approximate solution in the skewed setting by first finding the optimal solutions for the "heavy" clusters and subsequently adapting a local-search algorithm over the remaining, "lighter" clusters. It is intuitively expressed why this would not work if done in the naive setting due to (1) an accumulation of errors and (2) a need to adjust the local search for when some centers (those handling the heavy clusters) must stay in place. Although these arguments are subtle, their description is clear and I find them convincing.

Third, the coreset application almost feels like a separate theoretical contribution. The notion that oversampling according to sensitivities is sufficient to approximate the heavy clusters' cost is interesting and I could see applications of this to other problems.

### Weaknesses
This feels like a paper written by people in the theoretical CS community which has been adapted for ICLR by incorporating a few experiments. I am not opposed to this in general -- many valuable insights can be gained by the exchange between these communities. However, I believe that for this paper to be a fit for ICLR, the authors should show that their algorithm addresses an actual need in the machine learning community beyond being theoretically interesting.

### Experiments

I believe that the experiments the authors have chosen are, unfortunately, the least interesting ones and do not highlight why one would use the proposed algorithm. The included experiments simply show that the authors' approximation achieves lower cost than the scikit-learn implementation of $k$-means and $k$-medioid. This falls into the category of "verifying the theory".

However, this experiment is insufficient to justify using the algorithms proposed by the authors. Namely, the authors have not convinced me that there is a good reason to use the proposed techniques over standard k-means++ -> Lloyd's algorithm. The following are a few themes which would help address that concern:
 1. The authors' primary contribution is a faster algorithm for obtaining a $(1 + \varepsilon)$-approximation but they have not shown how fast/slow the algorithm is in practice. How long does the algorithm take to run? How does this change as a function of $n$, $k$, or $\varepsilon$? At what dataset-size does it become infeasible to run? The authors only chose a real-world dataset of 400 data samples -- much smaller than would appear in any realistic scenario that the standard ICLR attendee would be operating in.
 2. The comparison to scikit-learn's clustering lacks depth. Yes -- the authors' $(1+\varepsilon)$-approximation is (as expected) better than the $k$-means++ $\log k$ one. But the scikit-learn algorithm is, I expect, significantly faster. Thus, a fairer comparison would be to time the authors' algorithm on a series of datasets and then repeatedly run the scikit-learn algorithm until the time expires, keeping the minimum cost over these runs. This would actually suggest that the authors' algorithm is a worthwhile use of the runtime in practice. An analysis of this over n would be necessary to show that the authors' proposed algorithm is consistently useful over scikit-learn's kmeans.
 3. The choice of datasets feels insufficient. First -- it is not even verified whether the chosen Exasens dataset is heavily skewed. I understand that testing for the skewness technically requires the optimal solution, but I'm sure it can be checked approximately (indeed, if the skewness cannot be verified then it is unclear when one could even apply the proposed algorithms in practice). So is there even a guarantee that the algorithm should work here? Second, why was this the only chosen dataset? In the absence of evidence to the contrary, one might conclude that the Exasens dataset is the only one where there is a significant difference between the authors' proposed algorithm and the scikit-learn approaches.

### Regarding the scope

This point is a matter of opinion and I am willing to change my mind: although I find this paper to be interesting and well-written, it feels like it's on the outer edges of topics which are relevant to the ICLR community.

That is, I am unconvinced that there is a practical need for approximating the k-means/k-median solution for skewed datasets in faster time than was previously attainable. If such settings do exist, I am unconvinced that within them one would need a solution which is better than the quality of the standard k-means++ -> Lloyd's usecase. Lastly, if one does require a solution of such quality, then the authors' experiments have not convinced me that their proposed algorithm performs better in practice than running scikit-learn's k-means method $\sim 100$ times and picking the best set of centers from that set.

Thus, this paper feels like its applications are primarily geared towards the theoretical computer science community and these results should appear in a theoretical computer science conference rather than at ICLR.

### Potential connection to cluster separability literature

Perhaps I'm misunderstanding something, but I get the sense that the skewness notions described by the authors inherently lead to separability between clusters. Namely, in order for a dataset to satisfy the Zipfian distribution, a heavy cluster (with center $c_h$) in the optimal solution must have exponentially higher cost than some subsequent light cluster (with center $c_\ell$). Suppose now that we remove center $c_\ell$ and instead place two centers on $N(c_h)$. $c_h$ was the heavy cluster, so this should induce a large cost-decrease. However, this new partition is by definition not the optimal solution. Thus, the points $N(c_\ell)$ must now have very large cost in this new partition to offset the cost-decrease we obtained. By extension, we can conclude that the variance within $N(c_\ell)$ is small compared to this cluster's distance to the rest of the data. This seems extremely similar to the proximity conditions described in [2].

Based on the above logic, it feels like a theme of related work is missing. Namely, it is known that one can obtain a $(1 + \varepsilon)$ approximation when the data is sufficiently separable [1]. Thus, it would be helpful to highlight when one might expect the skewness to come into play without simultaneously incurring separability conditions under which the optimal solution can be approximated using known techniques.

[1]: Ostrovsky, Rabani, Schulman, Swamy. The Effectiveness of Lloyd-Type Methods for the k-Means Problem.

[2]: Kumar, Kannan. Clustering with Spectral Norm and the k-means Algorithm.

### Questions
The questions are in line with the above discussion and ordered from most to least important:
1. Can the authors comment on when the skewness conditions would not lead to separability between the clusters? Specifically, when does the skewness not induce the proximity conditions described in [2]?
2. Could the authors run a comparison of their algorithm vs scikit-learn when both are given the same amount of time to find the solution and then report the minimum of the scikit-learn solutions? I would expect that the solutions are of equal quality in this setting but look forward to being proven wrong. Similarly, could the authors report runtimes of their algorithm vs. scikit-learn on real-world datasets of varying size?
3. Can the authors describe the practical settings where the improved approximation should be required? Specifically, can the authors point to literature from ICLR/ICML/Neurips in the last decade or so where the datasets exhibit the relevant skewness and the improved accuracy was necessary?
4. It would go a long way to show this on additional datasets which satisfy the skewness requirements. Does the difference in solution quality between the $(1 + \varepsilon)$ approximation and the scikit-learn solution remain on other datasets?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents algorithms for k-means, k-median etc that achieve an (1+eps)-approx. The authors introduce the notion of (s,1-\eps)-skewness. A dataset has this skewness if there exists an optimal clustering that using s \leq k cluster-points that accounts for 1-\eps of the cost.
In that case they achieve a runtime that can be even polynomial assuming s is a constant or close to one. Note that the problem in general is APX-hard meaning a (1+eps) approximation is not possible in polynomial time assuming P != NP.

The authors study different versions (continuous and discrete clustering) and prove rigorously their results. They supplement their theory results with some experimental results that show that often their clustering is better than commonly used algorithms.

### Strengths
Interesting model

Strong results

Non-trivial math

Experimental results show that the obtained results are actually good

Well-written

### Weaknesses
The crucial assumption is not validated: it would be nice to have some experiments (for small datasets) that suggest that for small s one can actually get (s,1+eps) skewness (for real-world-data).

The run-time of the algorithms in the experiments is not discussed.

The results are very bad for large s, this needs to be highlighted. A better comparison with the SOA would be good here. 

Other baseline algorithm could have been used in the experiments.

### Questions
What was the result in the experiments you got for other baseline algorithms?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies the popular $(k, z)$-clustering problem in both discrete and continuous settings. The $(k, z)$-clustering problem has been widely studied in the field of machine learning, where the goal is to identify a set of centers with size at most $k$ while minimizing the sum of the distances from the data points to the centers selected. For the $(k, z)$-clustering problem, several theoretical lower bounds for approximation guarantees have been established, where one cannot even achieve a PATS within polynomial running time (i.e., the $(k, z)$-clustering problem is APX-hard). To break the theoretical lower bound and further narrow the gap between theory and practice, this paper introduces the notion of "skewness" for the clustering problem. Intuitively speaking, a give clustering dataset is skew if the clustering cost of several optimal clusters take a large fraction of the overall optimal clustering cost. To measure the skewness of a given datasets, this paper proposes the notion of $(s, 1-\epsilon)$-skewed dataset, where the clustering cost of the top $s$ optimal clusters take a fraction of at least $(1-\epsilon)$ of the optimal clustering cost.  Based on the proposed notion of skewed dataset, this paper proposes coreset and potential set construction methods. The coreset construction method can reduce the number of data points significantly while maintaining the skewness of the given dataset while the potential set construction method reduces the number of potential centers for enumeration based on $O(\epsilon)$-net construction method. Then, this paper proposes new local search methods such that the proposed methods can achieve $(1+\epsilon)$-approximation for any given dataset satisfying the skewness properties. The high-level idea behind the local search methods is to handle skewed optimal clusters and other optimal clusters separately. By enumerating for the (approximately) optimal clustering centers for the $s$ skewed optimal clusters and fixed them, the proposed local search algorithm only needs to achieve a convergence on the remaining $k-s$ clustering centers using exhaustive local search strategies. For discrete setting, the proposed local search algorithm should enumerate for all the subsets of the given dataset with sizes $s$ to find the optimal clustering centers. However, in continuous settings, coreset and potential center set construction methods can be applied to reduce the enumerations to accelerate the local search process. Based on the techniques above, this paper obtains PTAS for the $(k, z)$-clustering problem in discrete settings with running time $(nk/\epsilon)^{O(s+1/\epsilon)}$, where $s$ is the number of skewed optimal clusters. For continuous settings, an improved running time of $\tilde{O}(nk) + (klogn)^{\tilde{O}(s+1/\epsilon)}$ can be obtained. Empirical experiments show that the proposed local search methods achieve better performances compared with Lloyd-type clustering algorithms.

### Strengths
1. This paper establishes PTAS for the $(k, z)$-clustering problem by leveraging specific data distribution assumptions on the skewness of the data points. This advancement in approximation is a clear advantage of the proposed methods, marking a significant contribution to the field of clustering and machine learning.

2. The proposed data reduction and local search techniques are new and interesting, offering potential enhancements to the well-studied local search methods with tight theoretical lower bound of $(9+\epsilon)$.

3. The proposed data distribution assumption is shown to be practical, as data points generated from a Zipfian distribution naturally exhibit the skewness property.

### Weaknesses
1. While the introduced concept of skewness facilitates the design of approximation algorithms capable of achieving $(1+\epsilon)$-approximation guarantees for clustering quality, the time complexity of the proposed local search-based methods can grow significantly if the number $s$ of skewed optimal clusters is large. Furthermore, there are several existing local search algorithms that can be executed in linear running time with respect to the data size (such as those in [1]-[3]). Consequently, the proposed methods in this paper may have running time considerably slower than those linear-time local search methods even when there are only a few skewed optimal clusters. In practical applications, the number of skewed optimal clusters may indeed be large. The main advantage of this paper is the approximation guarantees on clustering quality. However, to achieve better clustering quality guarantee, a significant sacrifice is made in terms of time complexity. This paper lacks a detailed examination of cases with numerous skewed clusters and instead relies on exhaustive enumeration for these clusters, which may limit the scalability of the approach for large-scale clustering instances. This paper also lacks discussions on the trade-off between  the time complexities and the theoretical clustering quality guarantees. It is recommended that the authors should include a more detailed analysis or discussion of how the algorithm's performance changes as the number of skewed clusters increases.

2. In the experimental section, this paper primarily compares the proposed local search methods with existing Lloyd-type clustering algorithms, but lacks comparisons with other existing local search methods. Previous work has shown that while these local search algorithms may not achieve strong theoretical approximation guarantees on clustering quality (e.g., $(1+\epsilon)$-approximation), their empirical performance often yields clustering costs close to the optimal. Adding comparisons between the proposed methods and existing local search algorithms would provide a more comprehensive evaluation of their practical effectiveness. It is recommended that the authors should add more comparisons between the proposed local search algorithms and the $(9+\epsilon)$-approximation multi-swap local search algorithm and other sampling-based local search algorithms in [1]-[3].

3. To verify the effectiveness of the proposed method, this paper should compare the proposed method with other clustering algorithms that can achieve near-optimal clustering costs (such as the branch and bound method given in [4]).

4. The experiments lack a comparison of the running times across different clustering algorithms. Running time is a crucial metric for illustrating the trade-off between clustering quality and algorithmic efficiency, and its inclusion would provide a more balanced evaluation of the proposed methods.


[1] Beretta L, Cohen-Addad V, Lattanzi S, et al. Multi-swap k-means++[C]//Proceedings of the 37th International Conference on Neural Information Processing Systems. 2023: 26069-26091.

[2] Fan C, Li P, Li X. LSDS++: Dual sampling for accelerated k-means++[C]//International Conference on Machine Learning. PMLR, 2023: 9640-9649.

[3] Lattanzi S, Sohler C. A better k-means++ algorithm via local search[C]//International Conference on Machine Learning. PMLR, 2019: 3662-3671.

[4] Ren J, Hua K, Cao Y. Global optimal K-medoids clustering of one million samples[C]//Proceedings of the 36th International Conference on Neural Information Processing Systems. 2022: 982-994.

### Questions
Part of the Questions have been given in the Weakness part. Below are some further questions.

1. Does the proposed PTAS framework rely heavily on the local search strategy? For $s$ fixed skewed clusters, is it possible to use other clustering techniques (such as linear-programming rounding and sampling-based methods) to find the remaining $k-s$ clustering centers to achieve $(1+\epsilon)$-approximation?

2. In experiments, how to choose the number of $s$ for a specific dataset. From the paper, $s$ is fixed as $1$. What are the performances for other choices of $s$?

3. In experiments, the swap size $t$ is also fixed as $1$ instead of $O(1/\epsilon)$ as stated in the algorithm and the theoretical analysis. Please explain.

4. According to the local search algorithms proposed in this paper, the time complexity for finding skewed optimal clusters increase sharply as the number of skewed clusters increase. The authors should add a detailed discussion on the trade-off between the time complexity and the approximation guarantees regarding the number $s$ of skewed optimal clusters. Furthermore, the authors should explore or discuss potential optimizations for handling cases with many skewed clusters, rather than relying solely on exhaustive enumeration.

5. Since the main contribution of this paper is the improvements on clustering quality guarantee, why don't the authors include comparisons against methods like the branch bound and linear programming methods? These methods can give references of optimal clustering costs and how close is the proposed local search methods to the optimal solutions. 

6. Adding specific metrics or visualizations would be helpful for comparing running times, such as scaling plots showing how runtime grows with dataset size or number of clusters for different algorithms.

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
4

### Summary
This paper studies the classic $k$-clustering problem. In this problem, given a dataset of size $n$ and a positive integer $k$, the objective is to identify up to $k$ points in the space as clustering centers while minimizing the connection loss between the data points and the centers selected. For the clustering problem, local search methods are one of the most widely used techniques in practice. However, a challenging issue for the local search algorithms is that the best approximation ratio for the local search methods within polynomial running time is $(9+\epsilon)$, which has also been proved to be tight. To further narrow the theoretical guarantees on approximation ratio for the local search methods, this paper introduces the concept of "skew datasets" and presents a new local search method achieving a $(1 + \epsilon)$-approximation under skewness assumptions. The proposed approach for the continuous space involves three main steps: (1) Coreset Construction; (2) Potential Center Construction; (3) Heavy Skewed Local Search. The first two steps ensure efficient data reduction while maintaining the skewness on the reduced dataset. In the final step, $s$ center points (for finding skewed clusters) are fixed, and local search is performed on the remaining $k - s$ potential center points. Additionally, this paper also conducts empirical experiments on synthetic and real-world datasets to demonstrate the effectiveness of the proposed local search method.

### Strengths
The strengths of this paper can be summarized as follows.

1. Under the notion of skewness, the proposed local search method can break the approximation lower bound of $9+\epsilon$, achieving near-optimal approximation guarantees on clustering quality.

2. The skewness of a dataset can hold for several commonly used data distributions, such as Zipfian distribution.

3. Experiments show that the proposed method can achieve better performance compared with other clustering algorithms.

### Weaknesses
The weakness of this paper can be summarized as follows. 

1. The proposed method relies heavily on the dataset distribution, particularly requiring $s$ (the number of heavily skewed clusters) to be sufficiently small. In practice, this condition might not always hold for many real-world or synthetic datasets.

2. Additionally, the paper does not provide an effective method for calculating or estimating the specific value of $s$ (the number of heavily skewed clusters) for a given dataset, which limits the feasibility of executing the algorithm’s searching process for heavily skewed optimal clustering centers.

3. In the experimental section, the paper lacks a comparison between the proposed algorithm and standard local search methods, and does not analyze influence of different choices for the number of skewed clusters (i.e., the number of $s$).

### Questions
Q1. The theoretical results of this paper for Euclidean space heavily relies on the coreset construction methods. With the coreset construction methods, one can also obtain a $(1+\epsilon)$-approximation by executing a weighted PTAS on the coreset constructed without any assumptions on the data distributions (such as the skewness). Can the authors give detailed comparisons between the proposed method in this paper and directly applying a weighted PTAS after the coreset construction?

Q2. The results of this paper heavily rely on the skewness of the given clustering instance. The proposed method needs to enumerate all subsets with sizes smaller than $s$ to find the optimal clustering centers for the heavily skewed clusters, where the running time can be very large if $s$ is large. Directly using brute-force enumerations may lead to exponential running time. How to handle the case when $s$ is large? 

Q3. What are the performances of the proposed method compared with the existing local search methods? The authors should add more experimental results to give clearer comparisons with the state-of-the-art local search algorithms.

Other Comments:
- page 3, line 161, it should specifically explain the reasoning for $\Delta = polyn$ and whether it relates to the aspect ratio.

- page 4, line 208, the specific definition of sensitivity is provided, but Algorithm 1 does not specify how to calculate the sensitivity for each $x \in X$. A specific or approximate calculation method should be provided.

- page 15, line 793, the conclusion “with probability at least 0.99” relies on fixed $\epsilon$ and $\gamma$. However, the theorem does not reflect this property. The theorem should specify approximate values for $\epsilon$ and $\gamma$, or rewrite the probability in a form related to $\epsilon$ and $\gamma$. Similar issues exist for probability descriptions in other theorems throughout the paper.

- page 15, line 796, $(\mathbb{R}^d)^k$ is not defined.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies the $(k,z)$-clustering problem for data set with high skewness. The following notation of skewness is used: a dataset is $(s, 1-\epsilon)$-skewed if clustering cost of the $s$ highest cost clusters in the optimal $(k,z)$ clustering is at least $(1 - \epsilon)$ of the optimal total clustering cost. The paper presents a PTAS (running time is $(nk/\epsilon)^{O(s + 1/\epsilon)}$) for $(k,z)$ clustering for $(s, 1-\epsilon)$-skewed data set. The main idea is a partial local search scheme to find the $k$ center points. Here the $s$ centers for the high cost clusters are identified using a brute force search. This can be achieved by spreading $\epsilon$-nets around each point in a coreset of the data set,  inducing a set $T$ of points called candidate center points, and consider $s$-subsets of $T$. The paper presents how such a coreset can be constructed using sensitivity based sampling techniques while preserving the skewness. For the rest $k - s$ centers, an iterative multi-swapping local search process is applied.

### Strengths
The paper presents solid mathematical arguments, with the approximation quality and time complexity being theoretically verifiable. The proofs provided for the claims are complex and non-trivial. Additionally, the partial local search approach, which exploits the skewness property, is somewhat intriguing.

### Weaknesses
The biggest limitation of the result is the assumption that the dimension $d$ is a constant, while in many practical scenarios where clustering is considered, $d$ is very high. In fact, the size of the constructed candidate center points set $T$ exponentially depends on $d$. The algorithm also tries to perform a brute force search on $T$, which further increase the complexity. In line 90 of the paper, the authors mention that, for high $d$ value a technique in  Makarychev et al. (2019) can be used to eliminate the exponential dependance on dimensionality, and the time complexity can be reduced to polylog. But this is never discussed in later part of the paper. Because of such exponential dependance on $d$, I would not consider the proposed algorithm a real PTAS. The assumption of constant $d$ should also be made clear in the abstract.

The notation of skewness used in this paper is another aspect I believe limits the theoretical contribution of the paper. It depends on the cost of clusters in the optimal clustering, which make the condition rather strong. The running time also depends on $s$ exponentially. This means in some situation the algorithm is not substantially better than a standard algorithm that exponentially depends on $k$.

### Questions
Can you provide more details about the claim in line 90 that the running time can be improved for high $d$?

The algorithm depends on $s$ exponentially. In real life data set how large $s$ will typically be?

### Soundness
3

### Presentation
3

### Contribution
2
