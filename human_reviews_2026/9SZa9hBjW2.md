# Representative Action Selection for Large Action Space Meta-Bandits

- Decision: Reject
- Scores: 6, 2, 2

## Abstract
We study the problem of selecting a subset from a large action space shared by a family of bandits, with the goal of achieving performance nearly matching that of using the full action space. We assume that similar actions tend to have related payoffs, modeled by a Gaussian process. To exploit this structure, we propose a simple epsilon-net algorithm to select a representative subset. We provide theoretical guarantees for its performance and compare it empirically to Thompson Sampling and Upper Confidence Bound.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work tackles selecting a subset of actions from a large shared action space in bandit problems, aiming to achieve performance close to using the full set. It models action similarity with a Gaussian process and proposes a simple ε-net algorithm to choose representative actions. It provides theoretical guarantees and empirically compare the method to Thompson Sampling and UCB, showing competitive performance.

### Strengths
- The authors provide lower and upper bounds, which makes the proposed algorithm more convincing.
- Numerical experiments are provided, which helps readers to understand the charcteristics of the algorithm.

### Weaknesses
- Although some related works are mentioned, the authors do not clearly explain how they are connected to the problem setting.
- What is the main takeaway from the theoretical analysis? Is it primarily about the relationship between $\epsilon$ and the regret?
- Section 4 is overly long and difficult to follow.

### Questions
- I could not understand the main message of Section 3. What is the main message here?
- If we separate Section 4 into some subsections, what whould they be?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a framework for selecting a representative subset from a large action space shared by a family of bandits, trying to achieve performance close to that of using the full action space. The authors propose an ε-net–based algorithm and provide theoretical performance guarantees, as well as an empirical comparison to Thompson Sampling and UCB.

### Strengths
- The authors  try to bring high dimensional geomtery (epsilon nets) to bandits with large action spaces
- They provide a regret analysis of their proposed algorithm.

### Weaknesses
The paper starts from a motivation about handling large action spaces shared y a family of bandits, but it is not concretely instantiated for any specific family of bandits. It is formulated as an abstract framework whose significance is hard to understand without concrete instantiation in specific bandit families. the nature of the work seems more suited to a learning theory conference like COLT rather than to ICLR. 

Several details are also very unclear. Algorithm 1 seems extremely general and trivial and I don't see how it relates to the claim on lines 249-252 that the algorithm samples according to the importance measure q.

The paper is motivated by large action spaces but the experiments deal with extremely small toy action spaces - pretty clear that this experimental part was an add on, and doesn't make the case for large action spaces. Also in this simple example, I don't see which family of bandits is being referred to, and there is a lot of standard general stuff about RKHS that does not seem very relevant.

### Questions
- How is Algorithm 1 magically sampling from the importance measure q? It doesn't appear at all in the description of the algorithm
-  Could you give a specific example of families of bandits for which your results give something new?
- How does your work compare to two well known approaches to large action spaces, the X-armed bandits of Bubeck et al and the Zooming algorithm of Kleinberg et al?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies choosing a representative subset of actions from the original action space in linear bandit problems. More formally, it would like to choose the subset such that the optimal expected reward in the subset approximates the globally optimal expected reward, with performance quantitatively characterized by the regret defined in Eq. (1). It proposes Alg. 1 and proves some theoretical guarantees of it. Experiments show that the algorithm outperforms other baselines such as successive halving, combinatorial Thompson sampling, and combinatorial UCB.

### Strengths
Contextual bandits with large action spaces has many applications, and the attempts to reduce the action space to mitigate the computational cost is meaningful.

### Weaknesses
- From a statistical perspective, reducing the size of the action space in linear bandits does not provide a lot of improvement in the bandit regret. Importantly, the regret for linear bandits depends usually on the linear dimension of the action space, which is much smaller than the cardinality of the action space (e.g. "Bandit Algorithms", Theorem 36.4). In this respect I think the motivation in lines 58-75 (using the cardinality of the action space as the main argument) is somewhat flawed. 

- For Theorems 6 and 7 I am worried that the first term max_{r in R} E_\theta[ \max_{a \in r} \mu_a ] is not vanishing when eps -> 0? Why is this a meaningful result?

- For Theorem 9 I am afraid that we cannot escape the curse of dimensionality -- epsilon needs to be smaller than a constant for it to be useful? But then N(A_full, eps) is usually exponential in n (the dimension of the action space), which makes the required K also exponential in n?

- I am also not buying the application background of the problem setup. It looks like one is using meta learning to learn a representative subset of the action space. But this paper assumes that one knows the exact prior of the task parameter \theta? Is this prior learned by interaction with the previous tasks? If so, I am not sure if such prior can be assumed to be known exactly, since after learning in historical tasks, we may not learn the ground truth task parameter for those tasks, in a pointwise manner.

### Questions
See questions above. Also:

- CTS and CUCB, in my understanding, works when the reward function is additive across different arms in a super-arm. But here the reward function of a super arm is the maximum of reward of arms therein. Thus, I am not sure if this is a fair comparison..

### Soundness
1

### Presentation
2

### Contribution
1
