# Online Policy Selection for Inventory Problems

- Avg Score: 4.75
- Decision: Reject
- Scores: 6, 5, 5, 3

## Abstract
We tackle online inventory problems where at each time period the manager makes a replenishment decision based on partial historical information in order to meet demands and minimize costs. To solve such problems, we build upon recent works in online learning and control, use insights from inventory theory and propose a new algorithm called GAPSI. This algorithm follows a new feature-enhanced base-stock policy and deals with the troublesome question of non-differentiability which occurs in inventory problems. Our method is illustrated in the context of a complex and novel inventory system involving multiple products, lost sales, perishability, warehouse-capacity constraints and lead times. Extensive numerical simulations are conducted to demonstrate the good performances of our algorithm on real-world data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces the GAPSI algorithm, which combines online learning techniques with inventory control theory to address complex inventory management problems. This work notably contributes to practical aspects, especially in dealing with multiple products, perishability, and warehouse capacity constraints.

### Strengths
1.  The introduction of the GAPSI algorithm represents a significant advancement in applying online learning techniques to realistic inventory management problems.

2. Addressing Non-Differentiability: The focus on the challenges posed by non-differentiability and the proposed solutions demonstrate a deep understanding of the complexities involved in inventory optimization.

3. Extensive numerical experiments validate the performance of GAPSI.

### Weaknesses
1. The language is generally clear, but there are areas where conciseness can be improved. Aim to eliminate redundancy and streamline complex sentences for better readability. Additionally, ensure consistent terminology throughout the paper.

(1) Ensure that terms like "non-differentiability" are consistently used throughout the paper without unnecessary variation. For example, if you introduce "non-differentiability" in one section, use it consistently instead of switching to other similar terms.

(2) Sentences such as "This algorithm is adapted from GAPS (Lin et al., 2024) to take into account specific aspects of inventory problems, in particular, the fact that the functions we are dealing with are not differentiable" can be simplified. You might rephrase it to, "This algorithm adapts GAPS (Lin et al., 2024) to address the non-differentiability of functions in inventory problems."

2. The numerical experiments demonstrate the effectiveness of the proposed algorithm; however, the experimental section could be more rigorous. It is recommended to test under various parameter settings and scenarios to validate the robustness of GAPSI. Additionally, consider incorporating a broader range of performance metrics, in addition to the ratio of cumulative losses, to evaluate the algorithm's effectiveness.

3. The discussion on theoretical guarantees is critical. While the authors mention the unsuitability of standard OPS assumptions for the considered inventory problems, further exploration of how to establish performance guarantees or convergence analysis without these assumptions would be valuable.

### Questions
1. What criteria did you use to select the datasets? How representative are these datasets of real-world applications?

2. How do you determine which features to include in the feature-enhanced invarient of this policy? Are there specific criteria or methods for feature selection that you recommend?

3. Can you elaborate on the specific challenges posed by the non-differentiability?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The article discusses an online inventory management problem where managers must make replenishment decisions based on partial historical information to meet demand and minimize costs. The authors evaluate how realistic, general inventory problems fit within the recent Online Policy Selection (OPS) framework of [1]. Especially, they detail how various constraints very common in the industry, such as
perishability, lead times, or warehouse capacity constraints, can be mathematically modeled in OPS  framework of [1]. Finally, this paper provides extensive experiments which demonstrate the good performance of their GAPSI algorithm, a variant of the GAPS method in [1].

[1] Yiheng Lin, James A Preiss, Emile Anand, Yingying Li, Yisong Yue, and Adam Wierman. On- line adaptive policy selection in time-varying systems: No-regret via contractive perturbations. Advances in Neural Information Processing Systems, 36, 2024.

### Strengths
1. The article introduces GAPSI, an algorithm that integrates the GAPS[1] method for the inventory control problems, providing a sophisticated framework to tackle the real-world inventory management.

2. The author explain how common industry constraints, including perishability, lead times, and warehouse capacity limitations, can be effectively formulated as mathematical models within GAPS[1] framework in section 2.2.

3. The algorithm's performance is rigorously tested through extensive numerical simulations using real-world data, which strengthens the credibility of the results and the algorithm's practical applicability.

### Weaknesses
1. After meticulously reading this article, I feel that the biggest issue of this paper is that it merely provides a detailed implementation guide (such as policy settings, loss function selection, and modeling details) and an empirical simulation evaluation of  how the GAPS in article [1] can be applied to real inventory replenishment management.

2. Due to a series of problems like 'policies, losses, and transitions are not differentiable, thus neither classical chain rules nor smoothness apply,' the authors have not provided a regret analysis. Although this problem is extremely challenging, a rigorous theoretical analysis is something that everyone hopes to see and have addressed for many researchers in the operations research community and computer theory.

### Questions
This article is primarily inclined towards empirical evaluation and presenting some implementation details regarding how the GAPS framework in article [1] can be applied to real inventory replenishment management. Therefore, I would like the authors to demonstrate the significance of these empirical evaluations？ Is it merely to draw our attention to this issue? If that's the case, I feel that this article is more like a heuristic guidance manual for inventory management that could be posted on arXiv or SSRN. 

My concern is that this article resembles a case study, lacking comprehensive theoretical proof and large-scale real-world practice, so I would only give it a score of 5.

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
The paper studies online policy selection for general finite-horizon inventory problems, and generalizes Lin et al. (2024) to develop a new online control algorithm, named GAPSI. Two real-world datasets are used to empirically validate the effectiveness of GAPSI.

### Strengths
The paper is very well-written, and the authors' approach is easy to follow. The authors successfully apply online policy selection into inventory control, where the loss function, policy selection function and transition function are all nonsmooth. The authors explain in detail why they need to adopt customized differentiation instead of auto differentiation to make the online algorithm work.

### Weaknesses
The methodological contribution is a bit limited to the ML community. For more details, see the questions below.

### Questions
The major contribution is that you identified a customized way for the Jacobian computation for inventory problems. Yet, the current treatment seems to be very specialized to inventory problems. Can you generalize your approach to more general nonsmooth control problems by applying more general smoothing techniques (e.g., [1])?

No theoretical regret bounds have been provided. This is mainly because the inventory problem violates some of the assumptions needed for the analysis of GAPS [2]. However, there are other papers that provides theoretical bounds on policy gradient methods for inventory control, see, e.g. [3][4], and thus I believe some similar bounds could also be obtained for GAPSI. Can you thus provide some theoretical bounds for GAPSI?

[1] Nesterov, Y. (2005). Smooth minimization of non-smooth functions. Mathematical programming, 103, 127-152.

[2] Lin, Y., Preiss, J. A., Anand, E., Li, Y., Yue, Y., & Wierman, A. (2024). Online adaptive policy selection in time-varying systems: No-regret via contractive perturbations. Advances in Neural Information Processing Systems, 36.

[3] Bhandari, J., & Russo, D. (2024). Global optimality guarantees for policy gradient methods. Operations Research.
[4] Chen, X., Hu, Y., & Zhao, M. (2024). Landscape of Policy Optimization for Finite Horizon MDPs with General State and Action. arXiv preprint arXiv:2409.17138.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The authors address the challenge of inventory control in dynamic, real-world scenarios, where managers must make sequential replenishment decisions based on past data rather than assuming known demand distributions. It introduces an algorithm built on the Online Policy Selection framework. The approach leverages adaptive learning rates and feature-enhanced base-stock policies to model and manage inventory systems more flexibly. The paper demonstrates the potential of online learning methods for realistic inventory optimization.

### Strengths
The approach of applying OPS on inventory problems is creative. The idea of linear base stock policies is a nice concept despite being used in many other applications. 

Observations about non-differentiable points are very informative and nicely show why this aspect should not be neglected.

### Weaknesses
A lot of the material in the main body (and the appendix) is standard and should be omitted. The entire Section 2.2 is well known in the operations management and operations research communities. 

Consideration of base-stock policies is questionable. Safety stock (s,S) policies are much more common in practice and nicely capture fixed costs. 

I read with enthusiasm the arguments and illustrations why non-differentiable points are important. However how to copy with this issue left me empty handed. The solution proposed is very simple and begs for additional exploration (for example, in reLU at zero any number in [0,1] is a gradient) and thus selecting the left gradient might not be 'optimal.'

Perhaps the weakest point is that the weights in the parametric policy must be determine by hand. This is a very tall order and a very questionable requirement. 

Conducting abundant research in inventory management in my early career, the paper uses many established principles and concepts. On the optimization side, there are very few novel ideas (that really don't rely on aspects of inventory management). 

The appendix is a dumpster of many topics, most of them known by prior works.

### Questions
1. Why not using safety-stock policies? 
2 Are there any guidances, ideally scientific approaches, on how to set up weights? 
3. The OR community has developed many heuristics. Why did you benchmark only against one?

### Soundness
3

### Presentation
4

### Contribution
2
