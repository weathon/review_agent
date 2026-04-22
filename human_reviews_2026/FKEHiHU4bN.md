# Revisiting Matrix Sketching in Linear  Bandits: Achieving Sublinear Regret via Dyadic Block Sketching

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Linear bandits have become a cornerstone of online learning and sequential decision-making, providing solid theoretical foundations for balancing exploration and exploitation. 
Within this domain, matrix sketching serves as a critical component for achieving computational efficiency, especially when confronting high-dimensional problem instances.
The sketch-based approaches reduce per-round complexity from $\Omega(d^2)$ to $O(dl)$, where $d$ is the dimension and $l<d$ is the sketch size.
However, this computational efficiency comes with a fundamental pitfall: when the streaming matrix exhibits heavy spectral tails, such algorithms can incur vacuous *linear regret*.
In this paper, we revisit the regret bounds and algorithmic design for sketch-based linear bandits.
Our analysis reveals that inappropriate sketch sizes can lead to substantial spectral error, severely undermining regret guarantees.
To overcome this issue, we propose Dyadic Block Sketching, a novel multi-scale matrix sketching approach that dynamically adjusts the sketch size during the learning process. 
We apply this technique to linear bandits and demonstrate that the new algorithm achieves *sublinear regret* bounds without requiring prior knowledge of the streaming matrix properties.
It establishes a general framework for efficient sketch-based linear bandits, which can be integrated with any matrix sketching method that provides covariance guarantees.
Comprehensive experimental evaluation demonstrates the superior utility-efficiency trade-off achieved by our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper revisits efficient linear bandits using matrix sketching. Matrix sketching approximates the streaming matrix X to reduce update costs. Previous methods such as SOFUL lower the per-round complexity from \Omega(d^2) to O(dl + l^2) by maintaining low-rank sketches, but their regret can become linear when the sketch size is too small or the spectrum decays slowly. The authors propose Dyadic Block Sketching (DBS), a new framework that adaptively adjusts the sketch size at multiple scales without prior knowledge of the data. Applied to linear bandits, this leads to the DBSLinUCB algorithm, which achieves sublinear regret under general conditions. Both theoretical analysis and experiments on synthetic and real-world datasets support its effectiveness.

### Strengths
1. Theoretical guarantees: When an appropriate value of \epsilon is chosen in advance, DBSLinUCB achieves the O(\sqrt{T}) regret of OFUL (Theorem 3), which SOFUL cannot do without knowing spectral information. Even in the worst case, the algorithm attains an O(dk) update complexity (Corollary 1) while maintaining robust regret guarantees.

2. Balanced trade-off: Regret can be controlled via fixed parameters (\epsilon, l_0), while the update cost depends adaptively on the matrix rank k or Frobenius norm \|\tilde{X}\|^2_F, achieving a balance between theory and efficiency.

3. Flexibility: The DBS framework is modular and compatible with methods such as FD and RFD, supporting wide applications in online and bandit learning.

4. Empirical results: Experiments on synthetic and real datasets show consistent improvements in regret and efficiency, confirming robustness under different settings.

### Weaknesses
1. Incomplete theoretical coverage: The theoretical guarantees do not fully subsume SOFUL. When l < k and \|\tilde{X}\|^2_F is unknown, the analysis cannot ensure an update cost of O(dl + l^2), and no choice of (\epsilon, l_0) achieves SOFUL-level efficiency.

2. Efficiency–regret trade-off: Achieving near-optimal O(d\sqrt{T}) regret may lead to high update costs, especially when the data matrix has large rank or Frobenius norm. It would help to provide examples or evidence showing that in practice, k \ll d or \|\tilde{X}\|^2_F is small (e.g., constant or o(T^{1/3})).

3. Lack of empirical context: In the MNIST experiments (Figure 5), the rank k of the dataset is not reported. Without this, the complexity comparison is hard to interpret—particularly when k is close to d, where DBSLinUCB may be much slower than SOFUL.

### Questions
Please address the concerns raised in Weaknesses.

### Soundness
3

### Presentation
3

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
This paper shows that existing sketch-based linear bandits can suffer linear regret when the data matrix has heavy spectral tails. The authors propose Dyadic Block Sketching (DBS), which adaptively adjusts sketch sizes to control global error. Applied to linear bandits, their DBSLinUCB algorithm achieves sublinear regret and better efficiency than prior methods.

### Strengths
1. This paper provides a clear motivation by highlighting the pitfalls of previous studies. And the introduced of multi-scale sketching approach is well-grounded. 

2. The algorithm and its analysis are presented clearly, with good intuition and easy-to-follow explanations. The numerical experiments use meaningful benchmarks, and the results, particularly in Figure 3(c), convincingly validate the claimed performance improvement.

### Weaknesses
I did not find any major technical weaknesses in this paper. However, as a reader who is not very familiar with matrix sketching applications in bandits or online learning, I have several questions about the positioning of this work and the choice of benchmarks, as mentioned in the question part.

### Questions
1. The discussion on the parameter choices in Remark 2 is insightful. I am wondering is there any approach on adaptively estimating l0 to make the selection of parameters adaptive to environments. 

2. I am not deeply familiar with the literature on matrix sketching for linear bandits, but I noticed that most baselines in this paper are from around five years ago (e.g., SOFUL [Kuzborskij et al., 2019], CBSCFD [Chen et al., 2020]). It would be helpful if the authors could comment on or compare with more recent works, such as Zhang et al. (2023) and Feinberg et al. (2023).

3. In the related work section on multi-scale sketching, the authors explain that previous algorithms were developed for different purposes. I am curious whether there are shared ideas or overlapping design principles between those existing methods and the proposed Dyadic Block Sketching framework.

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
4

### Summary
The authors first show that existing sketch-based linear bandit algorithms can suffer from linear regret. To address this issue, they propose a novel matrix sketching framework called Dyadic Block Sketching (DBS), which adaptively adjusts the sketch size in a multi-scale manner. By applying DBS to linear bandits, the authors achieve sublinear regret bounds. They further validate the effectiveness of the proposed method through experiments on both synthetic and real-world datasets.

### Strengths
1. The motivation of this paper is clearly presented.
2. The proposed methods are novel and interesting.
3. Although I did not examine the proofs in the appendices in detail, the theoretical results appear convincing and reasonable.
4. The writing is generally clear, though some parts require further clarification (see Weaknesses).

### Weaknesses
1. In lines 372–379, the authors discuss choosing $\epsilon$ based on the spectral properties of the data matrix. However, in linear bandit problems, it is usually unknown whether the data matrix is low-rank or has a heavy spectral tail. How should $\epsilon$ be selected in practice? Moreover, in the experiments, how was the value of $\epsilon$ determined?

2. Some experimental results require further clarification:
- In Figure 3(a), why does the regret of $\epsilon=4$ outperform that with $\epsilon=2$?
- In Figure 3(d), when the sketch-based methods use the same amount of space as OFUL, their performance is still inferior to OFUL. The authors should provide a more detailed explanation for this result.

3. Some statements in the paper are unclear or inaccurate:
- In the abstract, the authors claim that “the sketch-based approaches reduce per-round complexity from Ω($d^2$) to O($d$),” which is not accurate. The computational complexity of matrix sketching depends on the sketch size. If the sketch size is $O(d)$, the overall complexity of sketch-based methods remains $O(d^2)$.
- In line 66, the statement “thereby reducing the order of spectral error $\Delta_T$ and decoupling it from $d$” is difficult to understand for readers unfamiliar with Chen et al. (2020). It is recommended that the authors present the regret bound from Chen et al. (2020) in the paper for clarity.
- In the last line of page 3, the phrase “under certain conditions” is too vague. The authors should describe these conditions in more detail.
- In the last line of page 5, the phrase “$\tilde{X}$ the subset of rows approximated by inactive blocks” is confusing. Does it mean “$\tilde{X}$ is the subset of rows consisting of inactive blocks”?
- Algorithm 3 is commonly referred to as SCFD (Chen et al., 2020) rather than RFD (Luo et al., 2019). Note that the regularizer in SCFD sums the total mass of subtracted values during the FD procedure, whereas in RFD, it sums only half of the subtracted mass.

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper revisits the computational and regret trade-offs in sketch-based linear bandits, where matrix sketching is employed to reduce the per-round computational cost from quadratic $O(d^2)$---with $d$ denoting the feature dimension---to subquadratic, while maintaining sublinear regret in $T$, the time horizon. Earlier work has shown that for linear bandits regret of order $O(d\sqrt{T})$ can be achieved with per-round complexity proportional to $O(d^2)$. Subsequent research demonstrated how to reduce this computational burden using dimensionality reduction techniques, particularly matrix sketching, to achieve per-round complexity $O(dl + l^2)$, where $l$ is the sketch size. However, in these approaches, the regret depends on the *spectral error* introduced by dimensionality reduction and can become linear (i.e., vacuous) when the spectral error exceeds $T^{1/3}$.

This paper makes progress on this front by introducing an algorithm termed *Dyadic Block Sketching (DBS)*---a multi-scale sketching framework that adaptively doubles the sketch size as learning progresses. The key claim is that this adaptive structure bounds the global covariance error by a user-specified parameter $\varepsilon$, thereby ensuring sublinear regret independent of the spectral error. Empirical results on synthetic and real-world datasets (e.g., MNIST) support the theoretical analysis, showing improved regret–efficiency trade-offs compared to prior methods.

### Strengths
The computational efficiency of linear bandits is an important and active area. Reducing per-round complexity while retaining sublinear regret is a significant theoretical question. The proposed framework makes a meaningful contribution toward that goal and the method is natural (adapting the sketch size). The paper is well written with detailed proof given in the appendix. The authors complement their theoretical work with experimental validation, which is an added bonus.

### Weaknesses
While the claimed regret bound is independent of the spectral error, it still depends on other parameters such as  the $\ell_2$-norm of the feature vectors, and the sketch size of the active block $l_B$. This dependence implies that in certain regimes, the regret may still exhibit linear scaling. Providing a clearer exposition or characterization of when such linear growth arises would  strengthen the paper. In particular, the paper would benefit from explicitly stating conditions under which their algorithm reverts to linear regret. This will lead to open questions that can be stated explicitly. Regarding exposition, for semi/non-experts, the presentation is dense and occasionally confusing. Terms like “streaming matrix” and “spectral tail” are introduced early without clear explanations. It is not clear what is streaming matrix means in this context.

### Questions
The problem you identified: getting sublinear regret algorithm for linear bandits with sub quadratic per-round complexity for is very interesting to me. So it will be very nice to the readers if you could make a table with what is known (including your work) and to what extend this is an open question. Making a separate section on related works and discussing this will also make the paper very nice to read.

### Soundness
3

### Presentation
3

### Contribution
3
