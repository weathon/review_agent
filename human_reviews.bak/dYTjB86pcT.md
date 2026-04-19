# System Aware Unlearning Algorithms: Use Lesser, Forget Faster

- Decision: Reject
- Scores: 6, 5, 6, 5

## Abstract
Machine unlearning aims to provide privacy guarantees to users when they request deletion, such that an attacker who can compromise the system post-unlearning cannot recover private information about the deleted individuals. Previously proposed definitions of unlearning require the unlearning algorithm to exactly or approximately recover the hypothesis obtained by retraining-from-scratch on the remaining samples. While this definition has been the gold standard in machine unlearning, unfortunately, because it is designed for the worst-case attacker (that can recover the updated hypothesis and the remaining dataset),  developing rigorous, and memory or compute-efficient unlearning algorithms that satisfy this definition has been challenging. In this work, we propose a new definition of unlearning, called system aware unlearning, that takes into account the information that an attacker could recover by compromising the system (post-unlearning). We prove that system-aware unlearning generalizes commonly referred to definitions of unlearning by restricting what the attacker knows, and furthermore, may be easier to satisfy in scenarios where the system-information available to the attacker is limited, e.g. because the learning algorithm did not use the entire training dataset to begin with. Towards that end, we develop an exact system-aware-unlearning algorithm that is both memory and computation-time efficient for function classes that can be learned via sample compression. We then present an improvement over this for the special case of learning linear classifiers by using selective sampling for data compression, thus giving the first memory and time-efficient exact unlearning algorithm for linear classification. We analyze the tradeoffs between deletion capacity, accuracy, memory, and computation time for these algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper begins by discussing existing definitions of machine unlearning, highlighting their limitations through a specific case study. It then introduces a novel definition, System-Aware Unlearning, and provides a detailed explanation.
Contributions:
1.	The paper analyzes the limitations of the existing unlearning definitions and introduces the concept of System-Aware Unlearning. 
2.	They propose a general-purpose unlearning algorithm utilizing data sharding under the framework of System-Aware Unlearning.
3.	Authors also propose an unlearning algorithm for linear classification using selective sampling for the special case of linear classification.

### Strengths
1.	The paper introduces a novel System-Aware Unlearning framework that advances the theoretical foundations of machine unlearning. This framework takes a more practical approach by considering the actual system security model and attacker capabilities. 
2.	This paper includes two algorithms: a general-purpose algorithm for exact system-aware unlearning using data sharding, and an improved system-aware unlearning algorithm for the special case of linear classification. 
3.	A comprehensive theoretical analysis is provided that includes deletion capacity, model accuracy, memory requirements, and computational complexity.

### Weaknesses
1.	The necessity of "system aware unlearning" is not well justified. A critical assumption in the Introduction states, "consider a learning algorithm that relies on only a fraction of its training dataset to generate its hypothesis and hence the ML system only stores this data. " However, the paper fails to explain why models would be trained on partial datasets, or whether the data selection process itself requires unlearning due to potential information leakage.
2.	The paper claims, " Even if an observer/attacker has access to larger public data sets that might include parts of the data the system was trained on, in such a system we could expect privacy for data that the system does not use directly for building the model to be preserved." This statement is not true, as no privacy can be preserved if the entire training dataset is exposed to attackers.
3.	The conclusion that models would "not be statistically indistinguishable from each other" on Page 3 does not hold when the model owner uses the complete training dataset S.
4.	Using only one shard's core set for prediction would likely cause significant performance degradation compared to models trained on complete datasets. The paper lacks experimental validation of this approach.
5.	The paper lacks an Evaluation section. Machine unlearning algorithms should be assessed across multiple dimensions, including unlearning efficiency, effectiveness, and model utility preservation. A comparative analysis with existing methods is also necessary.
6.	The paper lacks a Related Work section. 
7.	Technical writing requires improvement, particularly in defining key concepts like ExcessRisk and parameters such as f*.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
(epsilon, delta) unlearning definitions require the unlearned model to be indistinguishable from retraining from scratch on the remaining data.  In this work, the authors argue that this is too strict of a definition for unlearning.  They propose a new definition for unlearning, called system aware unlearning, that takes into account the information that an attacker could recover by compromising the system.  The authors develop system aware unlearning algorithms that are efficient for a class of functions.

### Strengths
- “Furthermore, we believe that accounting for the information that an attacker could have access to is an interesting direction to explore in privacy, beyond unlearning.“ I fully agree, this is an interesting aspect of unlearning that prior work has neglected. Definitions that try to take this kind of information to account is valuable to the research community.

### Weaknesses
I fold in my questions here:

- “This presents significant challenges in the context of privacy regulations such as the
European Union’s General Data Protection Regulation (2016) (GDPR), California Consumer Privacy Act (2018) (CCPA), and Canada’s proposed Consumer Privacy Protection Act, all of which emphasize the “right to be forgotten.” As a result, there is a growing need for methods that enable the selective removal of specific training data from models that have already been trained, a process commonly referred to as machine unlearning (Cao & Yang, 2015).“ A general comment: It’s still an open problem how these pieces of legislation should apply to ML models, and whether unlearning can satisfy these legal requirements.


- “This is evidenced by a dire lack of exact/approximate unlearning algorithms beyond the simple cases of convex loss functions.“ There have been many unlearning algorithms for non-convex models, unless you mean a dire lack of unlearning algorithms that work? Although there are also exact unlearning algorithms for non-convex models such as SISA [1].

- “Our framework leverages the fact that many ML systems do not depend on the entirety of their training data equally“ Can you precisely define what you mean by this?

- Nit typo: “We then present a general-purpose algorithm for exact system-aware unlearning using data sharding for function classes that can learned using sample compression,..“

- “We also provide an improved system-aware unlearning algorithm for the special case of linear classification thus providing the first memory and time-efficient exact unlearning algorithm for linear classification.“ This confuses me. The above comment complained that there’s a lack of algorithms beyond simple convex settings and yet you provide another algorithm for the convex setting..?

- I thought that the motivating example in line 118 is bit contrived, to be statistically indistinguishable all you need to do is change the order of operations such that A() samples a set C from S and *then* removes U from this set, rather than sampling C from S\U.

- *(System-Aware-(ε, δ)-Unlearning): For any S\U there exists an S' that is distributionally indistinguishable when trained.* Does the existence of S’ imply we can find it efficiently? I think not, and if not, how should we think about instantiating this definition in practice?

- In Definition 3, taking S’=S\U recovers the original definition. I’m struggling to understand the utility of allowing S’ to be anything other than S\U. That is, I’m struggling to understand if this flexibility is useful in practice. Can you provide an illustrative example?

- “Since the attacker can only gain access to information stored by the system and used in the unlearned model, then we want to learn predictors that are dependent on a small number of samples.“ I struggle to understand this statement. For most parametric models, samples are not “stored” in the system (that is, they are stored, but through a complex learning process which is difficult/impossible to reverse engineer). Does this statement only apply to simple linear models? This idea of reducing the number of dependent samples seems to be core to the ideas underpinning section Section 3 and Section 4.

- “Algorithm 2 Unlearning algorithm for linear classification using selective sampling”. I struggle to understand if this is a real contribution to unlearning. The algorithm is simply reversing the learning process on the unlearned points through subtraction. It also applies to a very narrow case as you point out: ”This monotonicity is a unique feature of the BBQSAMPLER. Other selective sampling algorithms, such as ones from Dekel et al. (2012) or Sekhari et al. (2023), use a query condition that depends on the labels y of previously seen points. Due to the noise in these y’s, y-dependent query conditions are not monotonic; points that were queried can become unqueried. This makes it difficult and expensive to compute the core set after unlearning.“ and “ We note that since the BBQSAMPLER uses a y-independent query condition, it is suboptimal in terms of excess error before unlearning compared to algorithms from…“.

- “Why is Algorithm 2 not a valid unlearning algorithm under the prior unlearning definition
(Definition 1)? When a queried point is deleted, an unqueried point could become queried. Thus, under traditional notions of exact unlearning, during DELETIONUPDATE, not only would we have to remove the effect of U, but we would also have to add in any unqueried points that would have been queried if U never existed in S.“ I don’t follow this counterfactual argument. According to Def 1, running Algorithm 2 on S\U should be identical to running on S and then removing on U?

- Can you comment on the similarities between Algorithm 1 and SISA [1]?

[1] https://arxiv.org/abs/1912.03817

### Questions
See above.

### Soundness
3

### Presentation
3

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
The paper proposes a new, system-aware formulation of machine unlearning, which takes into account the information that an attacker could recover by compromising the system. The authors develop exact system aware unlearning algorithms based on sample compression learning algorithms and establish the computation time, memory requirements, deletion capacity, and excess risk guarantees.

### Strengths
1. The idea of incorporating the observer's perspective into the unlearning process is novel.
2. The theoretical analysis is overall coherent.
3. The paper is well-structured.

### Weaknesses
1. Lack of quantitative comparisons with other methods (theoretically or empirically).
2. The writing of the intuition part in Section 2 is not clear enough. The reviewer is quite confused about the statements on lines 125-128.
3. Lack of definitions for notations, e.g., $\mathcal X, \mathcal Y, \mathcal Z, \mathcal Z^*, f^*(\cdot), \Delta(\mathcal F), \mathcal J$, etc.
4. The author is confused about the “core set deletion capacity $K$” and the "$K$ shards" in Algorithm 1. Are the two $K$s the same parameter?
5. Some writing issues: grammar mistakes and typos, e.g., "we developed algorithms for exact system aware unlearning algorithms" on line 526 and "exact system aware" on line 531.
6. The definition stated in Sekhari et al. (2021b) and Guo et al. (2019) is generally referred to as "*certified* machine unlearning".
7. Although this paper mainly focuses on the theoretical part, it would be better to include the experimental results in the main text.

### Questions
1. What is $e$ in $\frac{K}{e}$ on line 275?
2. What are $\mathcal T_P$ and $\mathcal T_f$ on line 6 of Algorithm 1?

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
The paper proposes system-aware unlearning which ensures that an observer cannot distinguish between a model trained on the initial dataset with unlearning applied and a model trained on a smaller dataset without the deleted points.

### Strengths
Sharding and sub-sampling can improve efficiency of unlearning algorithms

### Weaknesses
The paper ignores Bourtoule et al. 

The paper needs to properly analyze the threats against prior unlearning algorithms (inference attacks against sequence of released models, revealing what data is deleted between two releases).

### Questions
A. The paper does not talk about the prior sharding methods (Bourtoule et al, Machine Unlearning). How does this work compare with that method (which is splitting data into shards, training one model on each shard, and retraining only the model that includes the to-be-deleted datapoint). I suggest the authors to comment on the fundamental differences between these works, and also comment on the followings.

1. Differences in computational efficiency during unlearning
2. Memory requirements for storing models/data
3. Impact on model accuracy as deletion requests increase
4. Scalability to large datasets or high-dimensional data 


B. Many follow-up papers of Bourtoule et al highlighted the possibility of an adversary observing multiple snapshots of the models (over time, as unlearning requests are processed). Can you analyze the proposed system-aware unlearning algorithm in presence of adversaries with continuous observations of models? In particular, I suggest the authors to respond to the following questions. 

1. How do the privacy guarantees of the proposed system-aware unlearning change under an adversary with continuous model observations?
2. Can you provide a comparative analysis of information leakage between your approach and previous methods in this scenario?
3. Are there potential modifications to your algorithm that could strengthen its resilience against such adversaries?
4. How does the computational overhead of maintaining privacy change in this threat model?

C. What are the limitations of the proposed method? Which threats remain unaddressed, and which types of algorithms are incompatible with this approach?

### Soundness
3

### Presentation
3

### Contribution
2
