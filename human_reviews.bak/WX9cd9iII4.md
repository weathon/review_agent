# Fair Off-Policy Learning from Observational Data

- Decision: Reject
- Scores: 6, 5, 8, 5

## Abstract
Algorithmic decision-making in practice must be fair for legal, ethical, and societal reasons. To achieve this, prior research has contributed various approaches that ensure fairness in machine learning predictions, while comparatively little effort has focused on fairness in decision-making, specifically off-policy learning. In this paper, we propose a novel framework for fair off-policy learning: we learn decision rules from observational data under different notions of fairness, where we explicitly assume that observational data were collected under a different -- potentially discriminatory -- behavioral policy. For this, we first formalize different fairness notions for off-policy learning. We then propose a neural network-based framework to learn optimal policies under different fairness notions. We further provide theoretical guarantees in the form of generalization bounds for the finite-sample version of our framework. We demonstrate the effectiveness of our framework through extensive numerical experiments using both simulated and real-world data. Altogether, our work enables algorithmic decision-making in a wide array of practical applications where fairness must be ensured.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies how to enforce user-side fairness in (off-)policy learning. Specifically, the paper considers allocating (i) the same action choice probability regardless user's sensitive attributes (e.g., allocating the decisions of credit lending $A \in \{0, 1\}$ regardless of their gender) and (ii) the same benefits (i.e., conditional policy value) for different user demographic groups. For this, the paper proposes the combination of the following two methods. The first one is fair representation learning, which aims to learn a feature $\Phi(X)$ which is predictive for the outcome ($Y$), but is not predictive for the sensitive feature ($S$). The second is to impose three types of fairness constraints to the policy learning objectives. The results on synthetic and real-world datasets show that the proposed method performs better than a naive baseline that myopically optimizes for the policy value without the fairness constraints.

### Strengths
1. **Conceptually interesting work.** Considering fairness in off-policy learning is conceptually novel and is an interesting discussion to bring to the community. The need of discussing fairness is also well-motivated.

2. **Reasonable approaches for enforcing fairness.** Learning fair representation to avoid discriminative action choices and imposing value-based constraints seems to be a reasonable approach to address fairness issues in policy learning.

3. **No restrictive assumptions of the policy class.** This paper does not constrain any assumptions on reward and the parametrization of policy. Thus, the framework is applicable to any machine learning models, as discussed in related work.

### Weaknesses
1. **Some formulation and mathematical descriptions should be double-checked.**
Specifically, there are some contradictions between the text description and the mathematical formulation of the *action fairness*. As the definition of action fairness should affect the theoretical analysis and empirical evaluations in the rest of the paper, this seems to be a critical concern.  Please refer to Question 1 for details.

2. **Related work seems insufficient.** 
Specifically, while the paper proposes a fair representation learning method, no discussion of fair representation learning is provided in the related work. It was thus difficult to judge if the proposed approach is novel enough from the manuscript due to the lack of this discussion.

3. **No discussion on how the specific problem of off-policy learning has been resolved.** In my understanding, enforcing fairness in OPL is difficult because $V_s$  $(= \mathbb{E}_{A \sim \pi(X)}[Y | X, A, S=s])$, which is necessary for imposing the constraints on "value fairness", is not easily estimated with biased logged data. However, there is no explicit discussion on this point, and it seems that the paper just applied what has been successful in the online setting to the off-policy setting, without discussing particular challenges we will face in OPL. It is not clear what kind of OPL-related problems this paper tries to solve, and also there is no discussion about this point in the related work. It would also be beneficial to have additional ablations for this part in the experiments as well.

4. **Results of the real-world experiments do not seem promising.** The proposed approach improves "action fairness", however, it does not improve or even sometimes worsens the "value fairness" compared to the naive, unconstrained policy. Justifications are needed for this point.

5. **The clarity of writing has some room for improvement.** Aside from the above points, this paper has some additional ambiguous parts. Please refer to Questions 2 and 3.

### Questions
1. **Definition of action fairness**

First, the paper defines the action fairness as follows.

> a policy $\pi$ should assign the same action $A=A'$ to two individuals with the same covariates $X = X'$ but different gender $S=S'$.

However, the paper repeatedly mentions that covariates $X$ can also be a factor of unfairness even if the algorithm does not explicitly make decisions on the sensitive attributes $S$ as follows.

> in observational data, **other variables may act as proxies of for gender, and, hence, the learned policy may still lead to discrimination** due to the underlying data-generation process. (Introduction)

> Action fairness is .. It ensures that individuals who only differ with respect to their sensitive attributes (**and covariates correlated with them**) receive the same decision. (Section 4)

Given this kind of confoundings, I guess the action fairness should instead be defined as follows.

*Let $\bar{X} \subseteq X$ be a covariate that does not depend on the sensitive attribute $S$ (i.e., $\bar{X} \perp S$). Then, a policy $\pi$ should assign the same action $A=A'$ to two individuals with the same covariates $X \setminus \bar{X} = X' \setminus \bar{X'}$ but different gender $S \neq S'$ and the associated attributes $\bar{X} \neq \bar{X'}$*. 

This modification will be needed because we can consider the following situation: $X_0 = f(S_0)$ and $X_{1:m}$ is drawn independent of $S_0$. In this case, the original condition can still cause discrimination because the policy can still allocate different actions depending on $X_0$ even if the policy does not explicitly depend on $S$ (i.e., "$\pi(X) \perp S$" as described in the paper).

Could you provide any justification for this point?

2. **"*Parato-optimal*" and "*general*" policies in related work**

The paper cites Viviano & Bradic, 23 as one of the most important related works and discusses the differences. However, it was not clear for me what are "*Parato-optimal*" and "*general*" policies refer to. Clarifications are needed for this point.

3. **Experimental details**

In the synthetic experiment section, I found this description:

> We provide the results for our framework across three different policy scores, namely $m \in $ {DM, IPS, DR}. 

However, Table 1 reports the results of only one of them, and the section does not describe which method is used. Additionally, I could not find any additional results even in the Appendix. Could you provide some clarification on this? I also suggest reporting the number of random seeds used for the experiment.

**Some other suggestions for improvement (not really weaknesses)**
- On page 4, the paper denotes "$\mu_j(X, S) = \mathbb{E}[Y | X, S, A = j], j = \{ 0, 1 \}$, are the outcome regression functions" (of DM and DR). However, $\mathbb{E}[Y | X, S, A = j]$ by definition refers to the (conditional) expected reward rather than the outcome of regression. $\mu_j(X, S) \approx \mathbb{E}[Y | X, S, A = j]$ should be more appropriate (i.e., using "$\approx$" instead of "$=$").

- Reading the Introduction, it was unclear whether the paper was focusing on user- or action-side fairness. Since there are some discussions on item fairness in recommender systems (e.g., Singh & Joachims, 18), it might be helpful to explicitly mention either sensitive features belonging to users' covariate or actions' features. (In Section 2, I soon realized that the paper focuses on user-side fairness, though)

Singh & Joachims, 18: Ashudeep Singh, Thorsten Joachims. "Fairness of Exposure in Rankings". KDD, 2018.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies off-policy learning under two fairness notions: (1) action fairness that requires the actions made by policies to be independent of sensitive attributes; and (2) max-min fairness and envy-free fairness defined based on utility (policy value) perceived by groups. The paper first explores the relations between action fairness and policy value fairness, and shows that the two notions are compatible under certain conditions. The paper then develops an algorithm that learns fair optimal policies. Experiments on both synthetic and real data validate the proposed algorithm.

### Strengths
1. The paper simultaneously consider action fairness and policy value fairness in off-policy learning. The results of relations between two notions of fairness are novel to my knowledge.
2. The paper is well-organized and easy to follow. In addition to the algorithm, it provides generalization bound and validation on both synthetic and real data.

### Weaknesses
1. While the proposed method can learn optimal policies under both action and policy value fairness, the underlying technique seems to be very similar to the existing methods. The technical contribution of this paper is unclear to me. Specifically, the proposed solution includes two steps: the first step aims to learn fair representation to ensure action fairness, while the second step incorporates the policy value fairness constraints to the objective function. For the first step, the idea of leveraging adversarial objectives to achieve independence between sensitive attribute and representation is similar to the concept of GAN, a common approach for fair representation learning. For the second step, the learning objective with fairness constraint is also straightforward. 

2. The paper doesn’t compare with any algorithms in experiments. While the authors claim that existing methods are proposed in a different setting and cannot serve as the baseline, I wonder whether the methods can be adjusted and applied. For example, when learning fair representation in the first step, many approaches in fair representation learning may be applied and serve as baselines.

### Questions
1. Many approaches have been proposed to learn ML models under fairness constraints such as demographic parity, envy-free fairness, max-min fairness. Except for the objective function, what are the technical challenges of off-policy learning compared to tractional supervised learning? In other words, why cannot methods proposed in fair supervised learning be modified and directly adapted to off-policy learning?

2. Can we adapt existing methods to your setting and treat them as baselines?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a framework for fair off-policy learning that is a neural approach to fair off-policy learning (in comparison to related works that are only linear). The framework is broken up into two stages. The first deals with action fairness (the action should be independent of a sensitive attribute). The second adds policy value fairness for different sensitive groups (the authors used two variants: envy-free fairness and max-min fairness). The authors provide generalization bounds and perform experiments on synthetic data and real world medical insurance data.

### Strengths
The paper is very well written and easy to follow. The authors take considerable care when defining concepts in the paper to make things clear for the reader.

The paper is novel in that it produces a framework that is considerably less restrictive than other related works. Specifically, the addition of a neural approach to fair off-policy learning. It is also general enough to fit many different contexts and needs of practitioners.

The authors provide clear generalization bounds.

The authors provide clear experimental results and describe the insights well.

### Weaknesses
Mostly minor nits for weaknesses:

It is unfortunate that no other baselines are available for this work.

Although space is very limited, it would be good to include more discussion from Appendix I in the paper.

The plots in the paper need to be more readable. Thicker lines and larger text to match the text size in the paper.

### Questions
See above weaknesses

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies offline policy learning from observational data with some fairness objectives. Specifically, they proposed two fairness objectives. The first one is called "action fairness" which aims to ensure that the learned policy should be independent of the sensitive attributes. The second one is called "value fairness" and there are two variants: "envy-free fairness" which aims to make the policy value the same across sensitive attributes and "max-min" fairness which aims to maximize the minimum policy value across sensitive attributes. They proposed a machine learning algorithm to achieve these fairness objectives in two steps. The first step is to achieve "action fairness". The paper borrows ideas from prior work to learn representations that are not predictive of sensitive attributes while maintaining predictive power of outcome using some loss functions including the confusion loss. The second step is to learn a policy using empirical counterfactual estimate of the policy value for each sensitive attribute with respect to the two proposed "value-fairness" objectives. They quantify the generalization error of the proposed algorithm, and conducted empirical evaluation on both simulated and real-world data.

### Strengths
1. Fairness in offline policy learning from observational data seems an interesting and important problem. 

2. The paper is, for the most part, well-written and easy to follow. 

3. They conducted both theoretical analysis and empirical evaluation on the proposed algorithm.

### Weaknesses
1. One concern is that the paper does not discuss how they deal with observational data. In particular, the proposed method depends on several estimators like DM, IPW, DR. But some quantities are unknown in the observational setting, like \mu_1, \mu_0, \pi_b. I would be great if the authors can discuss how these quantities are obtained, and how they affect the theoretical and empirical results. 

2. For empirical evaluation, especially for real-world data one, we do not know the counterfactual outcome of counterfactual actions. It would be great if the authors can discuss how the performance metrics are calculated. 

3. The contribution seems limited. The fairness notions proposed in this paper appear in many prior works. The approach to achieving "action fairness" seems to be borrowed from prior works. The approach to achieving "value" fairness is a direct application of counterfactual estimator. 

4. It would be great if the authors could discuss the advantages and disadvantages of these fairness objectives. In particular, when are these fairness objectives applicable? For example, envy-free does not seem to make sense to me in many applications, since it might happen that it is very easy to achieve larger expected outcome for one group than the other. Maybe the maximum value we can achieve for group A is 1, and the minimum value we can achieve for group B is 1 and the maximum for B is 2. Suppose there is a policy that can achieve value 1 for A and 2 for B, why would we decrease the value for B just to satisfy this fairness notation? For max-min fairness, similar arguments apply. Since group A is always lower in policy value, we will only learn our policy for group A. Why is this advantageous compared to fairness definitions that achieve pareto optimality? 

Some minor points:
4. In assumption 1, it should be P(A=1 | X=x, S=s), right? Otherwise, equation (3) and (4) might be ill-defined since \pi_b(X,S) can be zero. 

5. Equation (2), (3), (4), \phi is a function of D, which contains i.i.d. random variables of several datapoints. But in the right hand side, there is only one datapoint. It is a bit confusing to me. 

6. Theorem 1 is hard to understand. I think it must depend on the logging/behavior policy. Maybe it is captured in K? It would be great if the authors can have a clear definition of these notations. And it does not seem to take into account the fact that we are in the observational setting as I mentioned in 1.

### Questions
See weakness.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
