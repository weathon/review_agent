# The Expressive Limits of Diagonal SSMs for State-Tracking

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
State-Space Models (SSMs) have recently been shown to achieve strong empirical performance on a variety of long-range sequence modeling tasks while remaining efficient and highly-parallelizable. However, the theoretical understanding of their expressive power remains limited. In this work, we study the expressivity of input-Dependent Complex-valued Diagonal (DCD) SSMs on sequential state-tracking tasks. We show that single-layer DCD SSMs cannot express state-tracking of any non-Abelian group at finite precision. More generally, we show that $k$-layer DCD SSMs can express state-tracking of a group if and only if that group has a subnormal series of length $k$, with Abelian factors. That is, we identify the precise expressivity range of $k$-layer DCD SSMs within the solvable groups. Empirically, we find that multi-layer models often fail to learn state-tracking for non-Abelian groups, highlighting a gap between expressivity and learnability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper provides a theoretical characterization of what diagonal State-Space Models (SSMs) can and cannot learn when tracking sequential state information. The main results can be summarized as
    1. A single diagonal SSM layer with complex values can track any Abelian group but cannot track any non-Abelian group.
    2. A k-layer diagonal SSM can track a group if and only if that group has a "subnormal series" of Abelian factor groups with length ≤ k. This means stacking layers expands expressivity to solvable groups in a precise, depth-dependent way.
    3. While multi-layer models are theoretically expressive enough for certain non-Abelian groups, they consistently fail to learn these solutions in practice with standard gradient descent.

### Strengths
The notations and logical flow is very clear make it easy to follow the results.
This work provides a complete characterization of diagonal SSM expressivity and exposes a critical gap between representational capacity and optimization. 
Diagonal SSMs form a strict subset of the computational hierarchy: sufficient for Abelian group structures but provably insufficient for non-Abelian groups at single-layer depth, with depth providing only limited help in practice.

### Weaknesses
The results are pure theoretical, it would be good if there is any connections to real applications.

### Questions
The theoretical contribution is significant. I'm interested in whether there are any real applications that can make use of this results. For example in material science, there may exists certain task that the symmetry property may exists?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies the theoretical expressivity of diagonal State-Space Models (SSMs) on group state-tracking tasks. They show that a single-layer input-dependent complex-valued diagonal (DCD) SSM can track a group G at finite precision if and only if G is Abelian. A $k$-layer DCD SSM can track a group G if and only if G has a subnormal series of length at most $k$ with Abelian factor groups. The authors show a gap between this theory and their experiments. While the theory shows multi-layer diagonal SSMs can express solvable non-Abelian groups, experiments demonstrate they struggle to learn these solutions in practice. Even 2-layer models that can theoretically represent S3 fail to learn generalizable solutions.

### Strengths
1. The paper provides necessary and sufficient conditions for when diagonal SSMs can track groups, not just sufficient conditions or impossibility results
2. The connection to group theory seems an elegant way to relate architectural constraints to algebraic properties.
3. The paper doesn't just prove existence - it shows explicit constructions demonstrating how multi-layer diagonal SSMs can track non-Abelian groups.
4. The experimental section reveals an important gap between expressivity and learnability
5. The paper carefully handles finite precision constraints, which are practically relevant and often glossed over in theoretical work.
6. The results apply to popular SSM variants like Mamba

### Weaknesses
1. The experiments only test on 5 groups and don't explore what makes some solvable groups learnable vs others. More extensive experiments would strengthen claims about the learnability gap.
2. While the paper identifies that multi-layer models fail to learn non-Abelian groups, it doesn't deeply investigate why or propose solutions beyond noting "optimization difficulties"
3. State-tracking is a specific synthetic task family. The paper doesn't clearly connect these limitations to practical sequence modeling tasks
4. The paper doesn't compare with non-diagonal SSMs empirically, which would help quantify the cost of the diagonal constraint in practice.
5. While mentioning block-diagonal structures could help, the paper doesn't explore intermediate architectures between fully diagonal and fully dense.
6. The paper could better position results relative to circuit complexity findings ($TC^0$) and explain what new understanding this group-theoretic view provides--that was not very clear to me.

### Questions
1. Which real-world sequence modeling tasks actually require tracking non-Abelian groups? I'm not familiar enough with the matter to know how common these requirements are in NLP and other related domains
2. Can you identify any specific optimization challenges when learning non-abelian groups? (e.g. loss landscape, initiatlization?) Given the explicit construction for S3, could we initialize models closer to theoretical solutions to improve learnability?
3. What's the minimal architectural change needed to make non-Abelian groups learnable? Would 2×2 block-diagonal suffice for S3?
4. How do these limitations apply to transformers, if at all?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies the expressivity of input dependent, complex valued diagonal state space models. Is shows that under some mild decoder conditions, the single layer can track any abelian group but not any non-abelian group. The paper performs empirical experiments that show that while multi-layer DCD SSMs are theoretically expressive enough for non-abelian groups, they fail to learn it in practice.

### Strengths
The paper improves the understanding of the expressivity of State Space Models (SSM), by focusing on a specific type of SSM and a particular data type (abelian vs non-abelian groups). This seems to be a fresh perspective on analyzing expressivity and the theory appears to be rigorous.

### Weaknesses
One concern is on the significance of the result. Does this make the SSM architecture more expressive (or less) than a transformer? Does this have any practical implications on how we should train SSMs? 

Another point that makes it harder to understand the significance of the theory is that the experimental results do not directly support the theory but instead suggest that the finding that even if the models theoretically can learn certain tasks, the optimization fails to do it. This is not in itself bad, but the paper would have been much stronger if there were empirical results that supported the theory. 

For the task C_60, for instance, it's not clear how the training task is generated. Is the input 1,2,3,4,..., or is it a random draw of numbers between 1 and 59, or something else. 

Minor: 
Somewhat unusual formatting with the theorem boxes.
I suspect that many people who are experts on SSMs may not be deeply familiar with group theory. Hence giving concrete example to show what for instance, C_60, is may allow a broader audience enjoy the paper.

### Questions
Is there some relevant real world task that correspond to non-abelian group?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper demonstrates the limitations of input-dependent complex diagonal SSMs in terms of expressiveness for groups state tracking. The authors show that a single layer can track Abelian groups, and k layers can solve up to depth k; thus a two-layer model can track S_3. The paper also shows that diagonal models are expressive enough in theory, but in fact hard to train on non-Abelian tasks.

### Strengths
The authors provide clear iff theorems and link depth with group structure: i.e., 1-layer - Abelian, and k-layer - length <= k. The paper also provides a good demonstration of two-layer S_3 that illustrates the theory, and presents an interesting observation that expressivity does not directly lead to learnability in reality.

### Weaknesses
- It is uncertain if the observations in this paper will directly lead to same results in real-world benchmarks such as language modeling.
- the experiments in the paper are limited, not providing results with different state dimensions, precisions, and decoders.
- it is uncertain how the training details in the paper, and unclear if it is actually true that expressivity != learnability.

### Questions
- How many, and what different training settings have the authors tried?
- Do the results look the same no matter how the settings (e.g., hyperparam, learning rate, weight decay, scheduling) change?
- how crucial is the universal decoder?

### Soundness
2

### Presentation
3

### Contribution
3
