# Prediction without Preclusion: Recourse Verification with Reachable Sets

- Avg Score: 6.00
- Decision: Accept (spotlight)
- Scores: 5, 8, 5, 6

## Abstract
Machine learning models are often used to decide who receives a loan, a job interview, or a public benefit. Models in such settings use features without considering their *actionability*. As a result, they can assign predictions that are \emph{fixed} -- meaning that individuals who are denied loans and interviews are, in fact, *precluded from access* to credit and employment. In this work, we introduce a procedure called *recourse verification* to test if a model assigns fixed predictions to its decision subjects. We propose a model-agnostic approach for verification with *reachable sets* -- i.e., the set of all points that a person can reach through their actions in feature space. We develop methods to construct reachable sets for discrete feature spaces, which can certify the responsiveness of *any model* by simply querying its predictions. We conduct a comprehensive empirical study on the infeasibility of recourse on datasets from consumer finance. Our results highlight how models can inadvertently preclude access by assigning fixed predictions and underscore the need to account for actionability in model development.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces the "recourse verification" task, aimed at identifying models that assign fixed predictions, and proposes methods to assess whether a model can provide actionable recourses based on reachable sets.

### Strengths
- The paper is well-written and easy to follow.
- The motivation and justification for fixed points and regions are sound.
- The experiments demonstrate an improvement in the feasibility of recourse across published datasets.

### Weaknesses
Weaknesses:
- I am unsure about the paper's significant contribution. The primary focus appears to be on describing a search algorithm to confirm the existence of a feasible action for a user. 
- My major concern regarding this paper is its inapplicability to continuous features, as claimed by the authors. Is it possible to extend the MIP formulation to MILP formulation to incorporate continuous features?
- The paper compares the proposed method to two conventional baselines in terms of improving recourse feasibility. Recently, there have been some papers potentially improving feasibility, such as [1] and [2]. I suggest comparing with them.

### Questions
- The optimization problem (2) aims to optimize a constant value of 1. What does this objective imply? Does this optimization problem solely seek to find all feasible actions (feasible recourses)?
- Is there a relationship between two reachable sets? For instance, if $x_1$ is within the reachable set of $x$ and $x_2$ is within the reachable set of $x_1$, is it guaranteed that $x_2$ is also within the reachable set of $x$?

**References**:

[1] Nguyen, Duy, Ngoc Bui, and Viet Anh Nguyen. "Feasible Recourse Plan via Diverse Interpolation." International Conference on Artificial Intelligence and Statistics, 2023.

[2] Rafael Poyiadzi, Kacper Sokol, Raul Santos-Rodriguez, Tijl De Bie, and Peter Flach. Face: Feasible and Actionable counterfactual explanations. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, 2020.

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies recourse verification of machine learning models. Recourse verification is an important aspect of algorithmic recourse, which seeks to identify models that assign predictions without any actionable recourse for the decision subject. Ensuring the existence of actionable recourse is essential in applications affecting people’s lives and livelihoods, such as job hiring, loan approvals, and welfare programs. A model that offers no recourse for its decisions may permanently exclude subjects from accessing these benefits without offering a path to eligibility. Existing research largely focuses on recourse provision — providing individuals with actionable recourse — but only a few works study the infeasibility of providing recourse.

This work proposes an approach for recourse verification under actionability constraints based on reachable sets. A reachable set is a collection of feature vectors that can be reached from a given input using a set of allowed actions. The proposed method certifies the existence or non-existence of recourses by querying the model on every point in the reachable set or an approximation of this set. If the method finds a subset of the reachable set that contains a recourse, it certifies the existence of recourse. Similarly, if it cannot find a recourse in a superset of the reachable set, it certifies the infeasibility of providing recourse. If it cannot certify either of the above, it abstains.

### Strengths
1. The paper studies an important problem that has not been explored well in the literature. It makes a significant contribution in this area.
2. The paper is well-written and easy to follow.
3. It is claimed that the proposed method does not require any assumption on the prediction model. However, the model might need to satisfy some conditions for the decomposition approach, which is essential when the problem dimensionality is high. See the weaknesses section for more details.

### Weaknesses
1. The recourse verification process evaluates every point in the reachable set, which could be time-consuming if the problem dimensionality is high. The paper seeks to address this issue by a decomposition approach that partitions the action set using features that can be altered independently. However, this approach has not been explained well in the paper.
2. It is unclear how the separable features are identified. What role does the prediction model play in the identification of these features?
3. It is unclear what conditions the prediction model must satisfy for the features to be separable. For instance, the verification step may return an infeasibility certificate in partitions A_1(x) and A_2(x), but actionable recourses may still exist in the Cartesian product A_1(x) X A_2(x) of the two sets.

Minor comments:
1. Increasing the font size in Tables 1 and 2 could help improve readability.
2. It seems like a word is missing in the following sentences:
    1.  Pg. 1 — “In fraud detection and content moderation, for example, models should assign fixed [predictions?] to prevent malicious actors from…”
    2. Pg. 3 — “We can elicit these constraints from users in natural language and convert them to expressions that can [be] embedded into an optimization problem.”
3. Figure 3 is a bit confusing and could be made clearer. The x-axis has no label. It seems that the size of the reachable set *grows* rapidly under the decomposition approach compared to brute force, which is contrary to the text. If I understand correctly, the purpose of using decomposition is to reduce the number of points to verify.

### Questions
1. Could this approach be extended to certify the existence or non-existence of an abundance of recourse options instead of just one? A single recourse option might not be feasible for everybody, and having multiple recourses could provide more options to people. It might be possible to certify statements like "20% of the actions in the action set would lead to a positive outcome" by querying a random subset of the action set.
2. Different actions may have different costs for the subjects. For instance, it might be easier for a loan applicant to increase their credit score than their income. Could we incorporate costs for the actions and certify the existence of a low-cost recourse?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work introduces the recourse verification: to verify if the prediction is desirable for any actions over the inputs, which is modeled as a formal verification problem given the trained model and input specifications. The paper gives examples using the proposed reachable sets.

### Strengths
- The idea of verification seems to be novel in the sense of recourse and the motivation is clear.
- The formulation is easy to follow.

### Weaknesses
- My biggest concern lies in the lack of contribution in the verification methods, which directly follow the basic idea of formal verification but seem not to dive deeper into the optimization algorithms or target the specific challenge in the recourse setting.
- When introducing reachable sets, more details are expected to be discussed, i.e. continuous or discrete, $\ell_p$-norm bound ball. The verification seems to be sound but incomplete, and it is expected to be compared to more off-the-shelf reachibility-based verification methods in [1].
- Although experiments show the prediction without recourse and current methods fail to detect them, there are no other baselines of recourse verification for the comparison of tightness and time efficiency. Also, the experiment part is not well organized in the sense of merging section 4 and 5 as experiments.

[1] Liu, C., Arnon, T., Lazarus, C., Strong, C., Barrett, C., & Kochenderfer, M. J. (2021). Algorithms for verifying deep neural networks. *Foundations and Trends® in Optimization*, *4*(3-4), 244-404.

### Questions
See weakness

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors present a new idea, recourse verification, certifying if a predictive model guarantees actionable items for users to change the prediction outcome. Different from the typical algorithmic recourse problem where the goal is to find actionable items with minimum cost, this work aims at ensuring that users are not mistakenly precluded from recourse. In the paper, the authors first establish fundamental concepts and theorems for this new topic. Afterwards, they propose "reachable set" for enumerating plausible feature values after actions. With proper decomposition of feature space as the author propose, feasibility of recourse can be effectively tested. Finally, the authors conduct evaluations on real-world datasets and confirm the efficacy of verification.

### Strengths
1. Recourse verification as a new research topic seems intriguing and impactful. It makes sense that some predictive models can accidentally limit availability of recourse and thereby hinder the fairness. Upon this important issue, the authors establish a good foundation for follow-up research and may also benefit researchers working on the typical algorithmic recourse problems.
2. The proposed algorithms seem reasonable and the step of implementation is clear. Also, the effectiveness is verified in the experiments.
3. The writing is overall clear and easy to follow. The details of experiments are provided. The limitations of this work are also adequately discussed.

### Weaknesses
Certain parts of the proposed method may still be in early stages of development, which may require further refinement to guarantee its practical value. For example, as discussed in the limitation section, the verification algorithm does not work on continuous features. More concerns of mine are summarized in the Questions section below.

### Questions
1. It is unclear how often does the undesired preclusion occur in practice. In particular, continuous features are quite common and may trivially avoid preclusion if the capacity of the predictive model is not constrained. Even if we focus on discrete features only, I am still not sure if undesired preclusion can frequently happen. Let us assume users A and B who pass and got rejected respectively by a predictive model. If we ignore the cost, an easy recourse for A can be the difference between A and B in the feature space. If there are more users getting approved by the model, more candidates of recourse are available for A's actions; namely, it is more unlikely that we find no proper recourse for user A when data size grows. If the diversity of the approved users is so limited that no recourse can be found for user A, I wonder if the preclusion is then more like intended (e.g., setting up strict rules) instead of being an accident.
2. Following question 1, I am wondering if it is reasonable to adopt the idea of cost constraint in recourse provision to reduce the reachable set? For example, we certify if a model is not "fixed" given an upper bound of cost.
3. How do we check the quality of a recourse verification algorithm? Specifically, if we employ two recourse verification methods and get inconsistent results, how do we decide which one is better?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
