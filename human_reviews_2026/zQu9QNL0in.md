# The Effect of Depth on the Expressivity of Deep Linear State-Space Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Deep state-space models (SSMs) have gained increasing popularity in sequence modelling. While there are numerous theoretical investigations of shallow SSMs, how the depth of the SSM affects its expressiveness remains a crucial problem. In this paper, we systematically investigate the role of depth and width in deep linear SSMs, aiming to characterize how they influence the expressive capacity of the architecture. First, we rigorously prove that in the absence of parameter constraints, increasing depth and increasing width are generally equivalent, provided that the parameter count remains within the same order of magnitude. However, under the assumption that the parameter norms are constrained, the effects of depth and width differ significantly. We show that a shallow linear SSM with large parameter norms can be represented by a deep linear SSM with smaller norms using a constructive method. In particular, this demonstrates that deep SSMs are more capable of representing targets with large norms than shallow SSMs under norm constraints. Finally, we derive upper bounds on the minimal depth required for a deep linear SSM to represent a given shallow linear SSM under constrained parameter norms. We also validate our theoretical results with numerical experiments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
State-space models (SSMs) are a class of sequential models that have recently gained significant interest in the deep learning community due to their strong performance and computational efficiency. While a single SSM layer is remarkably simple, their power emerges from stacking multiple layers and interleaving them with MLPs. This paper provides valuable theoretical insights into this phenomenon by analyzing, in the linear regime (i.e., without MLPs), how depth affects the network's expressive power. The authors demonstrate that while networks of all depths are equivalent given the same number of hidden neurons, deeper networks achieve this expressivity with parameters of lower norms, which facilitates learning.

### Strengths
The paper is well-written overall, and the theoretical results formalize intuitive yet non-trivial insights. By addressing a gap in our theoretical understanding of these models, this work makes a valuable contribution to the community.

### Weaknesses
The primary weakness lies in the empirical section, which consists mainly of relatively shallow empirical verification of the theoretical results without adding substantial value. While experiments demonstrate that deep linear SSMs are easier to learn, they do not explain why this occurs. It would strengthen the paper to connect these findings more explicitly to the theory, for example, by showing that optimal parameters are closer to initialization or that the loss landscape is better conditioned. Additionally, in the S4D experiment, it remains unclear whether the improved performance stems from easier learning or simply from having more nonlinear layers. As a minor note, the wall-clock time measurements could be removed without diminishing the paper's quality.

### Questions
The teacher-student experiment in Figure 3 requires clarification. While it is described as a teacher-student setup (implying a learning experiment), the perfect matching in the plot suggests that what is being reported is the norm from the construction itself rather than learned parameters. I would expect that student over-parameterization would be necessary for successful teacher mimicking, and that the empirical results would exhibit more variance. Could the authors clarify what exactly is done here?

Additionally, here are several relevant papers in the SSM theory space could help better position this work:
- [Saxe et al. (2013)](https://arxiv.org/pdf/1312.6120) - Analysis of deep linear networks
- [Orvieto et al. (2024)](https://arxiv.org/pdf/2307.11888) - Complex vs. real recurrence, particularly demonstrating how complex numbers significantly reduce parameter norms
- [Zucchet & Orvieto (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/fbb07254ef01868967dc891ea3fa6c13-Paper-Conference.pdf) - How SSM design affects optimization, with focus on the challenges of learning the A matrix (complementary to this paper)
- [Proca et al. (2025)](https://openreview.net/pdf?id=KGOcrIWYnx) - Learning dynamics of linear SSMs

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper explores a tradeoff between width and depth in a linear variant of common deep state space models.  A theoretical analysis is presented by considering alternative representations of linear SSMs.  Theoretical results are backed by some empirical results.

### Strengths
1. Study of practical trade-offs within the design of deep SSMs is (imo) chronically lacking in the literature.  This paper is a step towards that. 
2. The paper is relatively well written, and is relatively easy to follow.
3. The experiments do support some of the claims made.

### Weaknesses
# Major Weaknesses

My major block with this paper is what the purpose of the paper is and the level of significance it achieves:

1. In practice, deep SSMs use always non-linearities.  Therefore, I will always struggle, on a fairly fundamental level, to be too enthusiastic about work that studies a theoretical construct.  This is not a comment so much on the quality of the work, but the significance.*  This may be acceptable for learning-theory specific conferences or workshops, but these types of work, in my _opinion_, rarely meet the level of significance I see in other papers. 
2. Building on this, the core of the study is that coupled linear systems can be represented as one larger linear system, and then performs an analysis of the parameter norms/predictive accuracies across different depths/widths of systems (and their corresponding implied models).  This is somewhat interesting, but I feel like it is not that impactful as a  study, nor does it introduce any methodological insights, again, harming the significance of the work.  
3. There is little-to-no meaningful discussion of how this discovery helps us build better (even linear, let alone non-linear) deep SSMs.  When should I use a deep network vs a shallow network?  The analysis doesn't actually link this to help design better networks, it just explains some phenomena that you might see while already training networks.  
4. Following on from this again, there is no justification of why I should worry about the norm of my parameters?  In my understanding, norm constraints (hard or soft) are useful for limiting the expressiveness of very large networks on limited data (i.e. regularization).  But this work looks at networks with fixed expressivity (at least in the unconstrained norm case), so regularization isn't relevant.  
5. I think there is some admission of this in the conclusion and future work sections ("Understanding this trade-off more thoroughly in nonlinear or real-world SSM variants remains an exciting avenue for future exploration.").  The points raised in these sections, to me, feel like the payoff for the theoretical examination that is presented, i.e. this is a great preliminary and setup for more wide-ranging experiments and further theoretical innovation.  

 
I will also state that I really struggle to understand the experiment in Section 5.2:  I don't understand the experimental setup, the purpose, or the findings.  For instance: what does the non-linear SSM experiment in Table 1 show (we know deeper networks are better all the way back to VGG?)?  You claim models have the same expressivity, but one has better performance, implying they have different expressivity?  What is the significance of the FFN in this experiment?  Why is it needed?  What happens if it is removed?  I invite the authors to either write a comment re-affirming the experiment and findings, or, re-write the section in the paper (I believe you're allowed to do this at ICLR?) and I commit to re-evaluating the revised manuscript.  Right now, however, I simply take nothing meaningful from this experiment, which is a shame.  

# Minor Weaknesses:
1. Can you include the norm of the parameters for the multi-layer networks on the diagrams?  Your initial claim is that the norm of parameters is lower in deeper networks.  As far as I can tell, this isn't actually quantitatively reported anywhere.  
2. Why is the slope of the wall clock time not equal to 1?  Shouldn't a five-layer network be five times more expensive than a one-layer network?  
3. Numerous citations are incorrect, e.g. S4 is from Gu not Smith.
4. Smith isn't one-dimensional (line 159).
5. Do the norms also have error bars? 
6. A 2x increase in wall clock time for a dramatic reduction in error seems (at least in broad strokes) acceptable to me.  
7. Are there differences in the memory consumption for different models?  Both in terms of on-disk memory footprint (presumably not in the number of parameters are roughly consistent) and GPU-memory consumption?  Memory is often as important as factor as runtime. 
8. This is personal preference, but I like it when papers with lots of theorems and lemma blocks follow each with a "Remarks" block, which expands the high-level discussion or proof sketch.  
9. I'm less sure on this one, but there is a wealth of work on learning and expressivity in linear networks.  It would be a nice touchpoint if you viewed the SSM as one big linear FFN (which you can do for fixed-length inputs using Toeplitz matrices).  Right now the work is presented as living in a little bit of a vacuum, which I think is a missed opportunity.  

# Conclusion:

Ultimately, I am left struggling with what the purpose or key take-away from this work is, and so I am left underwhelmed by this paper and struggling to find reasons to argue for its inclusion.  For now I will reduce my confidence from 4 to 3, because I am cognizant that I might have simply missed the point of this paper.  I am also open to re-evaluating my score if the authors can allay my concerns on the significance.  

I think this work is undoubtedly a start on a path to bigger and more practicable insights, but I simply do not think it is there yet.  I strongly encourage the authors to continue working on this, and look to develop more practicable insights on models that are used in practice (e.g. Mamba).  Then this work could be incredibly impactful.  Good luck.

(*If the authors can show works that use deep linear SSMs and are competitive, or if they benchmark them themselves, then this could be assuaged, but this is not present as it stands.)

### Questions
I invite the authors to respond to the criticisms I raise above in weaknesses, specifically justifying the broad significance and impact of this work.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates the expressivity of deep linear state-space models with respect to bounded parameter norms. The authors show that in the absence of parameter norm bounds, wide linear SSMs can be equivalently represented by deeper, shallow SSMs. Furthermore, it is shown that a wide linear SSM can be represented using a deep linear SSM with smaller norm bounds. The theoretical results are supported with numerical experiments.

### Strengths
The authors provide insights into the expressivity of linear SSMs, providing proofs that wide SSMs can be equivalently represented by deep SSMs. The fact that deep SSMs with smaller norm constraints can represent wide SSMs with larger norm constraints is insightful. Similar trends are shown to exist for nonlinear models. The reviewer believes this is an interesting contribution furthering the understanding of SSMs.

### Weaknesses
More effort could be spent improving the presentation and flow of the paper. In Section 3.1 for example, the hidden state of any given layer at time step 0, i.e., $h_l(0)$ is not initialized and the writing could be improved.
The theoretical statements are sometimes vague and imprecise. Overall the paper is tough to follow from a theoretical point of view due to a lack of mathematical rigor. The constant $c_1>0$ is not used in the statement of Theorem 1. In Theorem 2 and Corollary 1, I assume the bound holds for an l-layer SSM which is equivalent to the considered one-layer linear SSM, which is not clearly stated. Lemma 2 and its exhibition was confusing to the reviewer. What is meant by output coefficients? What is meant by output expansion? Please refer to an equation if new terminology is used. What is $\xi$? It was not defined before. The authors state that a detailed proof can be found in the appendix, but the proof simply refers to Corollary 2, which refers to Lemma 1, none of which contain any $\xi$ variable. Simply referring to other proofs does not consist of a detailed proof.

### Questions
For the numerical validation of Theorem 2, a teacher-student setup is used to analyse the norm constraint relationship. Do the student networks converge to the exact solution in this example? Is it possible to derive a system of equations to numerically find the exact weights such that the networks are equivalent similar to the constructive example provided in Section 4.2?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper aims to understand how depth influences the expressive capacity of deep linear state space models. The authors find that when parameter norms are unconstrained, increasing depth and width are generally equivalent. When parameter norms are constrained, depth and width effects differ, which the authors present. Increasing depth increases capacity to represent targets with large norms with smaller-norm weights. The authors provide theoretical results as well as empirical evidence to support their findings.

### Strengths
The authors provide a compelling argument in the introduction to motivate the analysis of depth for expressive capacity of deep linear SSMs; and provide good foundational background on related works.

The authors provide a solid theoretical analysis (Appendix A)

### Weaknesses
Inconsistent related works paragraph headers.

The theoretical framework could be better integrated into the overall narrative (and some of it moved into the Appendix), currently the theory and derived results are quite dense and could distract from the key results.

Section 5 is very verbose, and can be tied together more concisely. Currently, the overwhelming amount of text hides the results.
The empirical results can be expanded to cover more complex tasks and models.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
3
