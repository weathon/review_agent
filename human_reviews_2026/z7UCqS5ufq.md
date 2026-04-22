# Nonparametric, Contextual Preference Estimation and  Assortment Optimization

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
The growth of AI- and ML-based decision tools provides an array of decision- support agents that can be implemented into the user’s decision-making process. However few tools exist for contextual evaluation for the alignment of decision agents with clinicians' workflows. Consequently, few methods exist to test properties of a contextually-optimal set of aligned agents to target for adoption. Contextual evaluation of decision models is particularly important in settings with few or no gold-standard decisions or preference alignment, such as clinical decision-making. Our work adopts the multinomial logit choice (MNL) model as a framework for evaluating agent-alignment and identifying an optimal agent-set. We assume the observation of selections among a set of agents according to a contextual MNL model, characterized by context-dependent preference parameters. We study a weighted, regularized local likelihood maximization estimator, providing a uniform convergence rate over a compact context space. Additionally, when agent-specific utility parameters or functions are known, we provide results for the identification of a utility-optimal assortment of agents. In this setting, we provide results to construct valid confidence bands on inferential objects of interest and the ability to perform asymptotically valid tests on the composition of this optimal assortment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this work, the authors propose a framework for contextual assortment optimization and inference. Under a multinomial logit model that can vary conditional on context x, the authors provide tools for statistical inference of the "optimal assortment" of a set of items at x, which can be viewed as the smallest assortment among those that maximize the total expected utilities. The utility model extends to consider both context-specific user preferences and context-specific utilities. This framework can be viewed as extending that of Shen et al. (2023) to the contextual setting. The authors provide rigorous theoretical treatment of their results with preliminary empirical validation.

### Strengths
The authors clearly state the proposed technical problem and provide a formal treatment. The related work section clearly identifies the gap in prior work that is addressed by this approach -- i.e., Shen et al. (2023). Both the model and the target estimands (i.e., contextual marginal utility gap) are natural in a discrete choice setting, and the dependence of both the utility functions and preference function on x is beneficial for flexibility of the approach. Clear statement of theoretical assumptions and their limitations (e.g., that the multinomial logit choice model implicitly makes the independence of irrelevant alternatives assumption) is also a clear strength of the work.

### Weaknesses
## Limited empirical validation

In its current form, the work has very limited empirical validation, consisting of a single simulation study in the appendix. Additionally, there are limited details of the setup, performance metrics, and results for this experiment. This makes the work incomplete in its current form. 

This work is off to an excellent start; as the authors continue to refine it into its final form, I encourage the authors to substantially extend the scope of the empirical validation by adding further numeric experiments, as well as experiments with semi-synthetic or real-world data. This will provide important evidence for the empirical validity of the work. In experiments, I encourage the authors to illustrate how context-dependent estimation improves the empirical results. Ablations with context-dependent preference functions and utility functions would both be interesting, as would experiments that vary the assortment set size.  


## Limited motivation 

In its current form, there is also a strong disconnect between the motivation for the paper and its technical content. While the technical content of the work is sound, the authors repeatedly reference "decision agents" in the abstract, but drop this discussion in the introduction and the remainder of the main text. 

Throughout the abstract and introduction, a key thing I was missing is *why is this optimal assortment problem is important?* Further, I was looking for a more concrete motivation for why a decision-support tool developer might want to select a model on the basis of discrete choice preferences. Typically tool adoption decisions are made on the basis of some external outcome (e.g., utility/accuracy) based measure. 

While the proposed framework accounts for "tool-specific cost or utility information”, typically in clinical settings preference information is used to characterize some some component of the decision subject preferences or decision outcomes. E.g., the canonical decision curve analysis characterizes tradeoff between the cost of false positive and false negatives. Similarly, while the authors claim that users' perceived value can differ by prompt, this seems to motivate that the choice of decision-making tool would vary depending on the prompt, which is not practically feasible. 

Overall, I am certain that this technical problem *is* important in some setting, and I invite the authors to either (1) provide a stronger support for why application to decision-support tool selection is a salient application area, or (2) switch to a different application area. I encourage the authors to pick a motivating application that is aligned with the domain of the semi-synthetic / real-world experiments. 

## Dense technical presentation

While the current main text exposition provides good detail, it is prohibitively dense in its current form. I encourage the authors to provide more exposition surrounding key results, especially theorems 4.1 and 4.2. Further, while statement of assumptions in 2.1 is important, shifting these to elsewhere in the text would strengthen the flow leading into the problem formulation. Spending more time in Sections 2.1 and 2.2 describing the theoretical setup and target quantities would also help the reader understand the relevance and importance of this problem (see remarks above). Finally, in its current form, Section 4.3 reads as cursory and "tacked on". Given the importance of the hypothesis testing procedure, I invite the authors to spend more time outlining the estimation procedure, perhaps with pseudocode.

### Questions
- Can you elaborate on how this work differs from Wang et al. (2025b), given the similarities with the estimators reported in that work? Further, can you elaborate on the differences between this technical setup of the multinomial logit model and the standard BTL model? 
- Could you clarify the intended application domain? If decision-support tools remain the target, how do you envision practitioners selecting tools based on choice preferences rather than outcome measures?
- Can you provide more details on the simulation study in the appendix (setup, metrics, baselines)? What performance improvements does the contextual approach provide over non-contextual baselines?

### Soundness
3

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
3

### Summary
The paper studies contextual assortment optimization under a contextual MNL model. It proposes a kernel-weighted local likelihood estimator for context-varying log-preferences, introduces a debiasing step, and develops a Gaussian multiplier bootstrap to build simultaneous confidence bands for the index of the optimal assortment. Theoretical results include uniform convergence rates for the estimator, an error expansion for the debiased estimator, and validity of simultaneous confidence bands and tests.

### Strengths
This study addresses one of the central problems in deploying AI in the real world. The authors propose a cutting-edge approach grounded in classical methods whose theoretical performance is well understood by statisticians. The problem setting is well motivated, and the theoretical results appear sound. The work is somewhat preliminary in a few respects—for example, (i) the inferential target 
$S^*(x)$ relies on known utilities $r(x)$, and (ii) the contextual MNL inherits IIA—but these issues are secondary to the main contribution. As a pioneering effort in this area, I do not view these weaknesses as major concerns.

### Weaknesses
See above,

### Questions
See above,

### Soundness
3

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
2

### Summary
This paper studies contextual evaluation for preference alignment and assortment optimization within a multinomial logit (MNL) discrete choice model. The problem setting considers a set of items with utility values and context-dependent preference values, and assortment optimization aims to identify the subset of items maximizing the expected utility. Inference tasks can be defined for the optimal assortment to test its properties. Using data on observed selections from assortments under specific contexts, the paper proposes methods for estimation and inference. The estimation task estimates contextual preference parameters with uniform convergence guarantees. The inference task tests whether an item is included in the contextually optimal assortment, and theoretical results support the validity of the proposed testing procedure.

### Strengths
The paper recognizes the importance of contextual influence in assortment optimization. Under a realistic contextual MNL model, the paper develops theoretically sound procedures for estimating preference parameters and making inference on optimal assortments. The inference method is especially notable, as it supports in-depth analysis of the optimal assortment, which may be useful for guiding practical selection of decision-support agents.

### Weaknesses
1. Contributions should be clarified: In Section 1.2, the paper describes its contributions as (1) “providing uniform bounds on preference parameter and utility gap estimation” and (2) “first inferential procedures proposed in the contextual multinomial logit choice model literature”. I think both aspects require additional discussion to be fully convincing. 

On the first aspect related to estimation, what are some known estimation methods in contextual MNL choice model? And what type of performance guarantees are available? To obtain the uniform bound result in this paper, what is novel or different about this paper’s proposed estimation? 

On the second aspect related to inference, the paper mentions Wang et al. (2025b) as providing motivation, is the presented inference procedure a direct application of Wang et al., or is any non-trivial extension required? While there is no inferential method for contextual MNL model, is there any inferential method for MNL model without contextual consideration? If so, are there any meaningful comparisons with the paper’s proposed inferential procedure? 


2. Lack illustrative examples for practical connection: The paper begins with an interesting practical challenge of identifying if new tools should be included in assortment of decision aids for clinicians, but the rest of the paper does not connect back to this motivating task. While the paper is largely methodological, I think illustrative examples are needed to demonstrate the proposed method in practical setting. I see two possible ways to go about this: (1) a case study with synthetic or real-world data generated with the process described in Section 2, then apply the proposed methods on this data to reveal insights about preference and assortment (2) without using actual data, describe a few application scenarios where observable data is available for estimation and discuss examples of relevant hypotheses to test in inference.

### Questions
Please refer to my questions listed in the weaknesses. In addition, I have some clarification questions.

1. How realistic is it to consider the utility values (r) are known on all items? I would expect these values to often be similarly unknown as the preference parameters.

2. How to think about the observed decisions (y) in practice? Are these decisions representing a clinician choosing to use a tool from a set of tools? In that case, how about cases where a clinician uses multiple tools together from a set of tools?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Given a set of decision support tools, this paper aims to determine a subset of these tools that should be included in the optimal subset of decision aids given contextual information and varying objectives.

### Strengths
The problem of identifying which decision support tools are appropriate for a given context / utility function is an important problem.

### Weaknesses
- The clarity of the paper is lacking in terms of the problem statement and mathematical setup, making it difficult for a reader to understand and appreciate the main results. I believe the paper would clarity improvements in Section 1.4, Section 2 prior to publication. 
- For example, the paper could be improved by more clearly stating the problem that it aims to solve. (I needed multiple passes through the introduction/abstract and the mathematical setup). For example, in the introduction Line 50, the authors write that "The assortment optimization problem posits $n$
items, each with an associated preference and utility value, and attempts to identify the subset of
items that maximizes an expected total return utility." It is not clear to me (1) how preferences affect this optimization problem (it appears that the choice of items to select would simply depend on utility?) and (2) are there any limitations on the number of items that we can select (e.g. how large is the subset). These are relatively simple aspects of the setting that would be helpful to spell out precisely -- as the audience may not be familiar with assortment optimization.
- The statement of the mathematical setup could be made more clear by introducing the important mathematical notation, e.g. the decsion model, the set of items, an assortment $\mathcal{S}$, the set of assortment $\mathbf{\mathcal{S}}$ prior to stating the mathematical assumptions because it is difficult to interpret the assumptions without understanding how they fit into the decision model.

### Questions
N/A

### Soundness
2

### Presentation
1

### Contribution
1
