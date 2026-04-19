# Ricci Curvature, Robustness, and Causal Inference on Networked Data

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3

## Abstract
In the complex landscape of networked data, understanding the causal effects of interventions is a critical challenge with implications across various domains. Graph Neural Networks (GNNs) have emerged as a powerful tool for capturing complex dependencies, yet the potential of geometric deep learning for GNN-based network causal inference remains underexplored. This work makes three key contributions to bridge this gap. First, we establish a theoretical connection between graph curvature and causal inference, revealing that negative curvatures pose challenges in identifying causal effects. Second, based on this theoretical insight, we present computational results using Ricci curvature to predict the reliability of causal effect estimations, empirically demonstrating that positive curvature regions yield more accurate estimations. Lastly, we propose a method using Ricci flow to improve treatment effect estimation on networked data, showing superior performance by reducing error through flattening the edges in the network. Our findings open new avenues for leveraging geometry in causal effect estimation, offering insights and tools that enhance the performance of GNNs in causal inference tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors explore networked data via graph-neutral networks. Using graph Ricci curvature, they predicted the reliability of causal effect estimations on networked data. Their empirical findings confirmed that positive curvature regions yield more accurate results.

### Strengths
The main review will take place in this section owing to the flow in which the review was conducted.

### Abstract

- The first sentence is interesting and whilst I know this is the abstract, it would be helpful to know _why_ the author(s) think it is critical challenge to understand the causal effect in networked data - who cares and why?


### Introduction

- What do you mean by "the endogeneity"  - this is a fairly non-standard way to use that way (at least to this reviewer, hence please do tell your usage of the term)
- "identifying causal effects is particularly challenging on a network of units with non-trivial dependencies" - why is this? What gives rise to the difficulty?
- A graphic may be helpful at the end of page one to illustrate the concepts that you discuss (e.g. graph curvature) for readers less familiar with this topic.


### CAUSALITY, INVARIANCE, AND ROBUSTNESS

- What is meant by "unit" in the first sentence of this paragraph?
- Do you assume the presence of unobserved confounders? If so, please spell that out.
- Why is your outcome denoted $\mathbf{Y}$? That gives the impression that you are dealing with a multi-objective (more than one outcome) setting - are you? Perhaps best to spell out the domain of the features and the outcome(s) either way to avoid any confusion.
- I find definition 2 and the paragraph following, very vague and ambiguous. It is not clear what you are describing. What is $\mathbf{C}$? The way the bottom part of page 3 is phrased, make it difficult for a reader to extract meaning from a language that is rather too formal (without providing all the background, since that, I assume, lives in (Buhlmann, 2020)).
- Whilst I appreciate that (Rothenhausler et al., 2021) uses the term "anchor" for a very common graph concept, it may be better to stick with the common tongue when it comes to graph formalism, where $A$ in figure 1 is either a leaf node or a root node depending on the perspective of the modeller. These terms predate the former's by a few decades and so may make this part of the section of the paper a bit more accessible by using the more common nomenclature. 
- You should adopt the standard graphical norms when it comes to causal inference. If H is a hidden confounder then it is standard to dash the edges (H,X) and (H,Y). The way you have drawn figure 1 makes it appear as if H is observed.
- H is hidden (latent or unobserved) yet you are including it in equation 2? Please explain how that works.
- You use 'source' and 'anchor' to refer to variable $A$ but never the standard 'root/leaf'? Consider picking one of these four terms.

### CURVATURE, ROBUSTNESS, AND ENTROPY

- The start of section 3 could do with an illustration of the concepts involved with Ricci curvature (e.g. figure four from the top: https://www.researchgate.net/publication/334371953_Community_Detection_on_Networks_with_Ricci_Flow/figures?lo=1)
- Use $S(\cdot)$ and not $S(.)$ for the Boltzmann entropy.

### Entropic causal inference

- Move the review of entropic causal inference to the preliminaries - the introduction of it the way you have done it now, breaks the flow of the paper.
- This does make any sense: $Y = f(X,E)$ - what is $f$? What is $X$ the same as before? What is $E$ noise terms? What is the causal modelling framework you are using? Is $f$ an SEM? You seem to suggest that $E$ are exogenous variables which d-separeted (?) I assume that's what your symbol means since you haven't defined it, from the outcome variable $Y$ but then you introduce something you call 'alternative exogenous variables' - what's that?
- Equation 9 presently does not point toward anything since it is not clear how the connection is made between it and what you say it is pointing to, which is not clear since section four is lacking a lot of detail for this to be a convincing argument.
- What I find curious further about this section if that if equation 9 is a powerful result you build on, it would be helpful if you placed it in a lemme/theorem/proposition to indicate its importance to the paper.

### Curvature and causal inference

- Paragraph 2 in this section could use a figure for your stated causal relationship.
- This may be a very silly question but you say: "we expect a higher entropy of the exogenous variables when the curvature is larger" - how can you expect anything at all about the exogenous variables since you cannot measure them? They are latent and so unknown to the modeller. Hence, what use is theorem 2 (being the devil's advocate here)
- Suggestion: place this within a remark: "This ultimately suggests that more positive Ricci curvatures are expected to correspond to lower errors in estimating the causal effect"

### RELATED WORK

- What are you evaluation metrics actually measuring w.r.t. your contribution in this paper? What is the relation to Ricci curvature w.r.t. these metrics?
- To confirm, none of your networks have unobserved confounders?

### Weaknesses
See the Strengths section for relevant comments.

### Questions
See the Strengths section for relevant comments.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper claims to address the challenge of understanding causal effects in networked data using Graph Neural Networks (GNNs). They claim to utilize the link between graph curvature and causal inference, finding that negative curvatures complicate identifying causal effects to predict the reliability of causal effect estimations, showing that positive curvature regions lead to more accurate results.

### Strengths
The exploration of curvature is intriguing.
Causal inference on network data is an interesting problem to investigate.

### Weaknesses
- The contribution of this paper is relatively moderate, as it amalgamates ideas from other sources, resulting in a contribution that is at best modest.
- The paper is poorly written. 
- This paper seems to essentially be using causal discovery to improve causal inference. This is not properly explained. 
- Sounds like this paper violates the SUTVA assumption in casual inference on networks.

### Questions
- This paper seems to essentially be using causal discovery to improve causal inference.  Could the author please elaborate on that? 
- How do you justify violating the SUTVA assumption in casual inference on networks?
- Implementing Ricci flow adjustments involves complex computations, especially on large-scale networks, which can be computationally intensive and time-consuming. This complexity may limit its applicability to real-time or resource-constrained scenarios. How do the authors justify the complexity and computational costs?
- How interpretable are the results? What is behind the improved estimations (beyond statistical metrics)? 
- Ricci flow adjustments' effectiveness varies based on the unique structure of real-world networks, which are diverse and dynamic. Generalizing this method across different scenarios is challenging due to the complex nature of network connections. Moreover, the method's reliance on parameters like edge weights and curvature values makes it highly sensitive to even minor fluctuations. Selecting appropriate parameter values demands careful tuning and a deep understanding of the network, posing significant challenges in achieving optimal performance. Could the authors elaborate on these issues?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This manuscript bridges the Ollivier Ricci curvature with causal inference and shows that positive curvatures 'help' with the causal inference while negative ones do not. The connection is built on a causal model that regards the problem as a worst-case risk minimization. The main spot of the theoretical analysis is to show that if the ricci curvature is negative, then higher entropy is needed to fit the causal model. Numerical results align with the theoretical analysis. Further experiments show that integrating ricci flow with GNNs can improve the performance.

### Strengths
1. The paper discusses the background and related work quite carefully, and is in general well-written.
2. I like this discovery. Building connections between geometry of the network and the causal inference should shed light on more future work, in addition to more understanding of this topic, especially given the fast development.
3. I acknowledge the theoretical analysis as the main contribution, it is good that the experimental results corroborate with the theoretical analysis.

### Weaknesses
1. It seems that the background is discussed too much (until page 5) and the main entree is too short (only one page), which possibly indicates that this work relies a lot on previous work. Further, the large body of discussion triggers confusion, I would expect a clear logical argument, for example, the causal model has some property, leads to robustness, further goes to entropy, and lands in Ricci curvature. The above may be wrong, but readers need the correct version instead of a stack of notions.

2. As for the proposed method integrating ricci flow and GNN, I could not find anything else except one paragraph in section 5.1. So if I am not misunderstanding, the proposed algorithm is: First run Ricci flow and attain the edge weights, then use existing frameworks of GNN-based causal inference. This is OK but the contribution is quite limited. Have the authors think about how to further take advantage of this property and design new GNN frameworks? Such as [1] and many others. Further, it would be good to at least present the pseudo code/descriptions by steps of the proposed method.

[1] Curvature Graph Networks.

### Questions
1. I unfortunately do not understand the formalization part very well. Could you explain what are the roles of system robustness and distributional robustness in the causal model? (Especially the risk minimization). In section 3.2 you said "we utilize the correlation between Boltzmann entropy and Ricci curvature to formally establish this anticipated connection" but later in section 4, Shannon entropy is discussed and Theorem 2 also uses Shannon entropy. This is confusing, could you please make it clear?

2. I am fine with the most proof of Theorem 3.2 but not the second assumption. If I understand correctly, $X$ is the feature and $Y$ is the label, then the difference of two datasets may not just lie on the features. I wonder if the assumption that $H(Y|X;E)$ remains similar is OK in a causality task, but nothing more.

3. What does it mean by a "wrong causal model"?

4. This is not a major issue, but have you experienced computational issues when the networks are large? The computation of ricci flow takes $\tilde{O}(mn^3)$ if I remember correctly, which can be larger than the time for GNN. Then if we think about applications, this is a hurdle.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
