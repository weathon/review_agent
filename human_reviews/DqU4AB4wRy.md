# GUARANTEED USER FAIRNESS IN RECOMMENDATION

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 5, 3

## Abstract
Although recommender systems (RS) have been well-developed for various fields of applications,
they suffer from the crisis of platform credibility with respect to RS confidence and fairness, which
may drive users away from the platform and result in the failure of the platform’s long-term success.
In recent years, a few works have tried to solve either the model confidence or fairness issue,
while there is no statistical guarantee for these methods. It is therefore an urgent need to solve
both issues with a unifying framework with statistical guarantee. In this paper, we propose a novel
and reliable framework called Guaranteed User Fairness in Recommendation (GUFR) to dynamically
generate prediction sets for users across various groups, which are guaranteed 1) to include
the ground-truth items with user-predefined high confidence/probability (e.g., 90%); 2) to ensure
user fairness across different groups; 3) to have the minimum average set size. We further design an
efficient algorithm named Guaranteed User Fairness Algorithm (GUFA) to optimize the proposed
method, and upper bounds of the risk and fairness metric are derived to help speed up the optimization
process. Moreover, we provide rigorous theoretical analysis with respect to risk and fairness
control as well as the minimum set size. Extensive experiments also validate the effectiveness of the
proposed framework, which aligns with our theoretical analysis. The code is publicly available at
https://anonymous.4open.science/r/GUFR-76EC.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a framework named GUFR, which aims to generate high-confidence recommendation sets satisfying fairness criteria. The authors provide statistical guarantees, and the results on two public datasets demonstrate the effectiveness of proposed method.

### Strengths
S1. The authors provide theoretical analysis.

S2. The experimental results validate that the proposed framework enables various baseline models to meet the defined fairness conditions and risk control.

### Weaknesses
W1. Why the fairness metric is defined as the difference in recommendation performance (e.g., NDCG) between two groups requires further discussion. For instance, if Group G1 and Group G2 consist of males and females, respectively, recommending gender-preferred items to each group might achieve similar recommendation performance, but such recommendations could be discriminatory based on gender, which is unfair. In my view, the balanced groups (not limited to just gender) seem to be an important underlying assumption, which is not discussed.

W2. The objective function of minimizing the size of the recommendation set is puzzling. While a smaller set might help reduce uncertainty in recommendations, this conflicts with the objective of encouraging users to purchase or click as much as possible.

### Questions
Q1. Why does the constrained optimization problem defined by equation (7) exist a solution? It seems that requiring all users to satisfy the risk and fairness constraints does not necessarily guarantee the existence of a solution.

Q2. Is the parameter $\lambda$ (defined in line 144-145) a scalar on $\mathbb{R}$? What is $\Delta$ in the update formula for parameter $\lambda$ in line 11 of Algorithm 1, and why can such update formula obtain the **optimal** $\lambda$ that satisfies the required conditions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the critical issue of fairness in recommender systems, an area of increasing importance in machine learning and artificial intelligence. The authors propose an approach aimed at enhancing fairness in recommendations by introducing a framework that seeks to minimize the disparity in performance between advantaged and disadvantaged user groups. The paper provides a comprehensive overview of their methodology, including various modeling stages and experimental setups designed to evaluate the effectiveness of their proposed solution. The authors conduct experiments to prove the effectiveness of the proposed method.

### Strengths
1. Fairness is a crucial area of research in recommender systems, and the authors' focus on this topic is commendable.
2. The overall methodology presented in the paper is logical and coherent.

### Weaknesses
The authors demonstrate a lack of deep understanding and research in the field of fairness in recommender systems. Specific issues include the following:

1. The authors assert that existing methods "fail to construct a generalized fairness-based recommendation framework for different applications." However, the term "generalized fairness method" is not clearly defined. How does the proposed method differ from existing approaches? This statement lacks clarity and is not supported by references, theoretical backing, or experimental evidence.

2. The authors use interaction data to distinguish between advantaged and disadvantaged users, aiming to reduce the performance gap between these two groups. To my knowledge, this aligns closely with user-oriented fairness [1], with the only distinction being the constraint of the minimum prediction set in this paper. However, the authors do not compare their approach with any existing user-oriented fairness methods [2,3,4,5], nor do they analyze the advantages of their method over current approaches. It is essential to compare against existing methods in the experiments, and the authors could add constraints on the top-k values to introduce an average set size constraint for current methods.

3. While the paper aims to guarantee user fairness in recommendations, it only conducts experiments on user-oriented fairness without addressing attribute fairness, e.g., with gender or race as the sensitive attribute. Such experimental conclusions do not support the claim that the approach is generalized.

4. The figures in the paper are too small and unclear, making it difficult to interpret the results effectively.

[1] Li Y, Chen H, Fu Z, et al. User-oriented fairness in recommendation[C]//Proceedings of the web conference 2021. 2021: 624-632.\
[2] Han Z, Chen C, Zheng X, et al. In-processing User Constrained Dominant Sets for User-Oriented Fairness in Recommender Systems[C]//Proceedings of the 31st ACM International Conference on Multimedia. 2023: 6190-6201.\
[3] Liu Q, Feng X, Gu T, et al. FairCRS: Towards User-oriented Fairness in Conversational Recommendation Systems[C]//Proceedings of the 18th ACM Conference on Recommender Systems. 2024: 126-136.\
[4] Han Z, Chen C, Zheng X, et al. Hypergraph Convolutional Network for User-Oriented Fairness in Recommender Systems[C]//Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2024: 903-913.\
[5] Han Z, Chen C, Zheng X, et al. Intra-and Inter-group Optimal Transport for User-Oriented Fairness in Recommender Systems[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(8): 8463-8471.

### Questions
See weakness.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper addresses the important issue of fairness in recommender systems, proposing a novel framework aimed at improving fairness outcomes for users. The authors thoroughly analyze their proposed method, providing comprehensive proofs and assessments of its effectiveness. They conduct extensive hyperparameter experiments and make their source code publicly available, promoting transparency and reproducibility. Despite these contributions, the paper has several presentation and methodological shortcomings that require attention.

### Strengths
S1. The overall organization of the presentation is smooth, making it easy to read and understand the authors' intentions.\
S2. The authors provide a comprehensive proof and analysis of the proposed method.\
S3. The paper includes extensive hyperparameter experiments and openly shares the source code.\

### Weaknesses
W1. The paper exhibits numerous presentation detail issues, including:
1. There are noticeable typos, such as the one found in line 53: "However, ,".
2. Some equations lack punctuation following them.
3. The figures in the paper are of poor quality and too small to effectively convey information.

W2. Additionally, the experimental section presents significant issues:

1. Why do the authors differentiate users based on interactions rather than sensitive attributes, such as race?
2. The paper does not compare their approach with existing user-oriented fairness methods.
3. The authors only report the performance of the recommendation model combined with the GUFR method, without providing the original performance of the recommendation model. This omission makes it difficult to determine if GUFR actually enhances the original model's performance.

### Questions
My main concerns have already proposed in the weaknesses section, I would like to pose an additional question:\
Q1. Why is it necessary to guarantee the minimum average prediction set size? What significance does this have for recommender systems, and are there any related studies on this topic?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
The authors propose an algorithm to determine the ranking length for personalized recommendation while ensuring risk-control and user-oriented fairness.
The proposed method is based on the framework of Risk-Controlling Prediction Sets (RCPS),
which allows for the straightforward analysis of statistical guarantees as provided in Sections 4 and 5.
The authors also design a greedy algorithm to efficiently optimize score thresholds based on their theoretical analysis.
However, some of the notations/definitions in the current manuscript are confusing, and the theoretical claims are thus not convincing.
The empirical evaluation also lacks baseline methods,
and I believe the reported results do not directly demonstrate the effectiveness of the proposed method.

### Strengths
1. The authors propose the concept of RCPS and FCPS for recommender systems.
2. The reproducible code is available.

### Weaknesses
### (W1) The problem formulation is somehow unrealistic.
The authors define the expected risk (Eq. (3)) based on the 0-1 loss for each user in Eq. (1),
which is similar to the hit rate measure.
To my understanding, the 0-1 loss serves as a measure of user dissatisfaction, and so,
minimizing the expectation of the 0-1 loss allows us to guarantee each user's satisfaction and achieve risk-control. 
Nevertheless, they define the user fairness metric based on generalized recommendation measures for each user, which also serves as user satisfaction, rather than the hit rate measure.
It is quite counterintuitive for me because the user merit functions in the risk and fairness metrics are inconsistent.
In particular, since NDCG takes values between 0 and 1 and **not a function for a set** but an ordered set, the ranking position of a relevant item is essential in evaluating user satisfaction.
Also, using NDCG as the user loss is not trivial based on the currently provided theoretical guarantee;
the assumption of Bernoulli distributed losses in Theorem 1 is no longer valid.

### (W2) The empirical evaluation is insufficient.
In Section 6, the authors compare the different base models with the proposed method.
However, there is no comparison between the models with and without the proposed method,
and the current evaluation is not sufficient to directly show the effectiveness of the proposed method.
It would be helpful to set some baseline methods.

### Questions
### (Q1) On the definition of 0-1 user loss for multiple relevant items.
Please clarify the random variables to take expectation in Eq. (3).
In recommender systems, we often observe multiple relevant items for each user.
So, the current notation of $i_{true}$ is quite confusing.
How can the 0-1 loss be defined for multiple relevant items for a single user?

### (Q2) On the monotonicity of recommendation metrics.
In line 177, the authors compute NDCG for the set output of $\phi$.
However, NDCG is a ranking measure and cannot be computed for unordered sets; the hit rate measure is ok, on the other hand.
Also note that NDCG includes a normalization factor (i.e., ideal DCG) that may increase with the ranking length.
This implies that the monotonicity of $\Delta F$ assumed in line 681 is not trivial.
Authors can use DCG instead; it is additive and monotonic w.r.t. $\lambda$.
Still, I think that the monotonicity of $\Delta F$ is quite questionable;
even if $M(\phi_\lambda(u))$ is monotonic w.r.t. $\lambda$ for all $u$,
$\Delta F$ is generally not monotonic because of the absolute difference.
Any clarifications on this point would be helpful.

### Soundness
2

### Presentation
2

### Contribution
2
