# UNIVERSAL REPRESENTATION OF GENERALIZED CONVEX FUNCTIONS AND THEIR GRADIENTS

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
A wide range of optimization problems can often be written in terms of generalized convex functions (GCFs). When this structure is present, it can convert certain nested bilevel objectives into single-level problems amenable to standard first-order optimization methods. We provide a new differentiable layer with a convex parameter space and show (Theorems~\ref{fc-density} and~\ref{nablafc-density}) that it and its gradient are universal approximators for GCFs and their gradients. We demonstrate how this parameterization can be leveraged in practice by (i) learning optimal transport maps with general cost functions and (ii) learning optimal auctions of multiple goods. In both these cases, we show how our layer can be used to convert the existing bilevel or min-max formulations into single-level problems that can be solved efficiently with first-order methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a new differentiable parameterization for generalized convex functions (GCFs) and their gradients, extending prior work on convex and input-convex neural networks. It proves universal approximation results for both functions and gradients under mild semiconvexity conditions and establishes a convex parameter space that enables stable first-order optimization. The authors connect their construction to neural network architectures, interpreting finitely convex functions as shallow networks with max aggregation. 
They validate the approach on a mechanism design problem, showing it recovers known optimal auction outcomes. Overall, the work offers a unified theoretical framework linking convex analysis and learnable, structured function classes.

### Strengths
The main strength lies in its theoretical originality—generalizing convex-function parameterizations to the broader and practically relevant class of GCFs. The paper provides rigorous proofs and smooth differentiable variants, filling a gap in the literature on universal approximators for generalized convex functions and their gradients. 
Conceptually, it bridges convex analysis with neural network design, suggesting new structured architectures that preserve generalized convexity.

### Weaknesses
**Novelty**: This seems to be a direct extension of the work Balazs et al 2015 for convex functions. Here, the scalar product is replaced by the function $phi(x,y)$, but the results do not present any particular challenge to prove. Besides, the experimental validation choses phi to be a scalar product, so the motivation for general $phi$ in practical problems is weak.   

**Narrow empirical validation** Experiments focus solely on mechanism design and fail to demonstrate the method’s potential on broader or more recognizable machine learning tasks. The paper does not showcase improvements in adversarial training, optimal transport, or robust optimization—domains explicitly cited as motivations. As a result, the contribution may seem more mathematical than impactful for mainstream ML practice. The computational scalability and practical guidance for applying the method in high-dimensional settings remain underexplored. To strengthen the work, additional experiments on modern ML problems would be essential to illustrate the method’s real-world relevance and power.


**Clarity**: I found parts of the paper are unclear and confusing. In particular, in the section 4.2 on mechanism design and the experimental details in section 7 related to it. Specifically, I did not understand how the expected revenu of the seller is defined, why is the problem a bilevel problem, what is learned and how. A full description of the objectives to optimize, the algorithms should be provided. Although the code is provided, I could not understand it without knowing beforehand these elements. 
Also, some notions are introduced but without proper reference to the literature: in particular DRIC seems to be a standard concept in economics, but no reference is provided for it. I found section 4.2 on mechanism design a bit obscure. For instance, I don't understand how the seller maximizes their revenue by optimizing the price. What is the objective function that they optimize



**Soundness**: The proof of proposition 5.1 assumes implicitly that X,Y are compact, this was stated nowhere in the main paper.

### Questions
- In theorem 3.2, shouldn't it be f = (f^{XY})_X instead of f = (f^{YX})_X.
- The paper focuses on approximation of the gradient  of finite approximations to these generalized convex functions, but often, it is the parameters of the approximation that we'd like to learn using gradient methods, which is different from approximating the gradient of a generalized convex function wrt to its input. What can be said about learning the parameters  of these approximations using gradient methods?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper attempts to establish a unified theory for Generalized Convex Functions (GCFs) and their gradients in the context of parameterization and universal approximation. It proposes a differentiable finite representation — finitely Y-convex functions — as an alternative to neural network architectures. The authors claim that:
- They provide universal approximation theorems for GCFs and their gradients;
- They prove gradient convergence under semiconvexity assumptions;
- They demonstrate an application to multi-item mechanism design.

### Strengths
- It tackles an important theoretical gap by attempting to unify convex and generalized convex structures within a learnable, differentiable framew.
- The link between generalized convexity and bilevel optimization (e.g., mechanism design, optimal transport) is conceptually appealing.
- The mathematical framework, if rigorously developed, could provide a new lens for understanding structure-preserving function approximation.

### Weaknesses
- In the introduction, the paper argues that GCFs can transform bilevel problems into single-level optimization problems, yet all subsequent examples (e.g., mechanism design and optimal transport) already presuppose the existence of GCF representations via the Φ-transform.
This creates a circular argument:
  “We study GCFs because they simplify bilevel optimization; we know they simplify it because we assume the problem already admits a GCF form.”

- The notion of generalized convexity was formalized long ago in Singer (1997) and Rubinov (2013). The paper fails to clearly articulate whether its novelty lies in the parameterization of the function space or in an extension of existing approximation theorems. This ambiguity undermines the conceptual contribution.

- Theorem 5.1 (GCF UAP) merely asserts that “Proposition 5.1 + 5.2 ⇒ density,” yet Proposition 5.1’s proof relies on Φ being globally Lipschitz.
Since Φ is only assumed to be locally Lipschitz, there is no guarantee of a global constant on compact domains.
→ Therefore, the universal approximation claim is not rigorously established.

- If GCFs truly simplify bilevel optimization, the paper should demonstrate clear advantages under nonlinear or non-Euclidean Φ functions (e.g., adversarial or transport-type cost functions).
However, all experiments use the trivial linear Φ(x, y) = ⟨x, y⟩, which reduces to the standard convex setting.
Hence, the results fail to support the “generalized” claim.

- The manuscript, in my ability, is difficult to read and overly terse. Many claims (e.g., “we show,” “we extend”) are stated without actual derivations or rigorous proofs, which makes the paper opaque to readers outside the narrow mathematical optimization community.

### Questions
see the Weaknesses

### Soundness
4

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies generalized convex functions (GCFs) and their gradients. The authors present the applications about the general GCFs, and then provide a new differentiable layer with a convex parameter space where it and its gradient are universal approximators for GCFs and their gradients
respectively. Finally, they also conduct experiments to demonstrate its effectiveness in learning optimal pricing mechanisms when selling multiple goods.

### Strengths
The overall framework of this paper is clear. The concept of generalized convexity could be an interesting topic for further exploration.

### Weaknesses
The theoretical contribution of this paper is limited. The work appears to summarize several properties related to generalized convexity but does not provide any particularly insightful perspectives. In addition, some important lemmas and theorems, such as Theorem 5.3, lack sufficient explanation. I believe that including several necessary remarks would be more helpful for readers.

### Questions
My main concern about this paper lies in its theoretical contribution. The current version appears to lack technical novelty. Specifically, Sections 3.2 and 5 seem to be a collection of properties related to generalized convexity. Could the authors provide a more detailed summary of the technical novelty and contributions?

Regarding the experimental section, the authors only conduct experiments on a few illustrative cases. I am wondering whether there are any experiments on real-world benchmarks, such as CIFAR or similar datasets.

### Soundness
2

### Presentation
2

### Contribution
2
