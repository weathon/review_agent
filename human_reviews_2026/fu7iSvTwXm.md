# Achieving Hyperbolic-Like Expressiveness with Arbitrary Euclidean Regions: A New Approach to Hierarchical Embeddings

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Hierarchical data is common in many domains like life sciences and e-commerce, and its embeddings often play a critical role. While hyperbolic embeddings offer a theoretically grounded approach to representing hierarchies in low-dimensional spaces, current methods often rely on specific geometric constructs as embedding candidates. This reliance limits their generalizability and makes it difficult to integrate with techniques that model semantic relationships beyond pure hierarchies, such as ontology embeddings. In this paper, we present RegD, a flexible Euclidean framework that supports the use of arbitrary geometric regions---such as boxes and balls---as embedding representations. Although RegD operates entirely in Euclidean space, we formally prove that it achieves hyperbolic-like expressiveness by incorporating a depth-based dissimilarity between regions, enabling it to emulate key properties of hyperbolic geometry, including exponential growth. Our empirical evaluation on diverse real-world datasets shows consistent performance gains over state-of-the-art methods and demonstrates RegD’s potential for broader applications such as the ontology embedding task that goes beyond hierarchy. Code and data are available at https://anonymous.4open.science/r/RegD-F4E3.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new way of embedding hierarchical data, which represents instances as regions in Euclidean space instead of single points. The authors show that a special case of their framework results in a model that is isometric to hyperbolic space, thereby inheriting its beneficial properties. Through several experiments the authors show the strength of their method for embedding DAGs and ontologies compared to existing methods.

### Strengths
- The newly proposed method is in my opinion a very interesting approach that could potentially solve some of the issues that are usually encountered when dealing with hyperbolic space, while still retaining its benefits.
- The new model of hyperbolic space, implicated by Proposition 1, is a very interesting new perspective on hyperbolic geometry that could lead to many new insights.
- The theory supporting the method is convincing, well-structured and the proofs seem correct to me.
- Based on the results, the method seems to perform quite well on hierarchical data embedding.

### Weaknesses
My primary concern with this paper is that it makes a few strong claims that are not motivated well enough in my opinion:
- The claim is that the method can be applied with any kind of region as defined in Appendix A5. However, the theory all uses specific cases involving balls and boxes. To make a claim about arbitrary regions, the theory and proofs should be generalized to arbitrary regions.
- It is claimed that "only the relative order of the scores matters" based on the F1 score and Hits@k relying solely on the ranking. However, in other hierarchy embedding papers (e.g., [1, 2]) distance-based metrics such as the distortion are often reported as well. I think this should either be included in the evaluation or its exclusion should be motivated.
- The authors claim that their method and this framework are simpler then existing hyperbolic frameworks (lines 165-170 & lines 480-481). It is not completely clear to me why this is the case and I think it requires some more motivation. 

I also noticed that the results in Table 2 seem to be a combination of the results from Table 5, picking the best configuration for most experiments. This seems a bit dishonest to me. If you want to pick the best configuration depending on the experiment, then these values should be considered hyperparameters and included in the hyperparameter search instead of being presented as an ablation. 

Lastly, I think the paper could benefit from a bit more proofreading. For example, $g(x) = k \cdot x + b$ should probably be $g(x) = k \cdot x$ in Proposition 2, given Definition 1.

Despite these weaknesses I want to reiterate that I think the proposed method and the new perspective on hyperbolic geometry are interesting enough for the paper to be accepted. Moreover, I am willing to raise my score if my concerns are addressed. 

[1] Sala, Frederic, et al. "Representation tradeoffs for hyperbolic embeddings." International conference on machine learning, 2018.
[2] van Spengler, Max, and Pascal Mettes. "Low-distortion and GPU-compatible Tree Embeddings in Hyperbolic Space." International Conference on Machine Learning, 2025.

### Questions
1. I'm a bit confused by the part in lines 329-344. These specialized metrics do not satisfy Proposition 3, right? So why would we want to use these?
2. Can you elaborate on the statement in line 480-481 regarding the simplification of the implementation of HNN a bit? I don't really see how the required operations would simplify in your model of hyperbolic space.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces RegD, a region-based Euclidean embedding framework that aims to match the expressiveness of hyperbolic methods for hierarchical data while remaining flexible enough to support arbitrary geometric regions. It defines two complementary dissimilarities: a depth dissimilarity  that scales pairwise separation by region size to emulate hyperbolic exponential growth, and a boundary dissimilarity that captures asymmetric inclusion  via distances to region boundaries.

### Strengths
The paper offers a clean, general framework that reproduces key benefits of hyperbolic embeddings while staying in Euclidean space, combining a size-aware depth dissimilarity with an asymmetric boundary dissimilarity to model both separation across levels and set-inclusion along hierarchies. The theory is aligned with practice (mostly), depth dissimilarity can emulate hyperbolic behavior and even preserve the ranking of hyperbolic distances with simple choices of g, which directly supports ranking-based metrics used in evaluation.

### Weaknesses
Proposition 1 is elegant but narrow, and the follow-up argument that only ranking matters sidesteps scale/geometry mismatches; there is no empirical study showing that different g choices, norms, or region parameterizations leave training dynamics and downstream behavior unchanged. Also boundary dissimilarity definition mixes containment and non-containment with different signs and extrema; for balls the presented closed form appears to omit an explicit case split, which can be confusing and invites edge-case ambiguities that are not stresstested experimentally.

### Questions
It is possible to use classical isometries between hyperbolic spaces to prove your results in a model agnostic context?

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
4

### Summary
This paper presents a novel hierarchical embedding method, which uses depth-dissimilarity and boundary-dissimilarity to assimilate Euclidean spaces to hyperbolic spaces. The authors proved that, by choosing appropriate depth-dissimilarity functions, the method equals traditional hyperbolic embedding models. In other words, the method is a generalization of hyperbolic embedding to allow tailored hierarchy embedding.

### Strengths
1. The authors generalize hyperbolic embedding models by assigning depth-dissimilarity and boundary-dissimilarity functions to Euclidean spaces. It allows sub-exponential distance scaling to adapt to richer structures than hierarchy.

2. The authors connect non-linear distance metrics with ball/box representations. It allows more expressiveness while maintaining the semantic modeling capabilities needed for ontology.

3. Clear mathematical derivations and proofs.

### Weaknesses
1. Citation format errors: many of the citations should be in brackets.

2. The authors motivated the choice of a Euclidean alternative to the traditional hyperbolic embedding by saying "However, as shown in Table 1, hyperbolic methods often rely on specialized constructed objects as embedding candidates, limiting their generalizability to data that encodes richer semantics beyond hierarchy" (Line 51-53). This is a very vague, descriptive motivation that may confuse readers -- does a specialized constructed object necessarily mean bad generalizability to non-hierarchy structures? Is there any theoretical support in literature for your motivation?

You may consider referring to this paper (https://arxiv.org/pdf/2407.16641?) -- the "curvature" of a subspace you need to sufficiently embed a certain size of hierarchy relies on the *capacity* of the subspace. There is trade-off between capacity and numerical stability -- that is, it is beneficial to use a subspace that has just adequate capacity. This can be a good support for your motivation -- your method effectively allows users to specify sub-exponential distance (i.e. between linear distance in Euclidean spaces and exponential distance in hyperbolic spaces) so that the capacity of subspaces match the sizes of hierarchical structures.

3. As Sala et al. (http://proceedings.mlr.press/v80/sala18a.html) discusses, equipping a coordinate space with exponential distance metrics (e.g. Poincare ball model or Lorentz model) poses an upper limit of numerical stability -- the reflective boundary. Due to the finite digits of float numbers, the depth of the hierarchy that can be losslessly embedded is limited -- i.e., the practical expressiveness of hyperbolic spaces is not infinite as the theory permits. What is the limit of numerical stability of the proposed Euclidean space alternative? From how I understand it, there is trade-off between how fast the depth-dissimilarity increases and how deep a hierarchy the method can stably embed. What is the trade-off for your method?

4. Equation 7 and 8 lack theoretical motivation. Why can and should the two dissimilarities be weighted summed? Is there any convex guarantee?

### Questions
Can the authors provide more insights on Weakness 3 and Weakness 4?

What are the "richer" structures that your method applies better to than simple hyperbolic embedding?

### Soundness
3

### Presentation
2

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
The paper introduces RegD, a general framework for embedding hierarchical and ontological structures using Euclidean region embeddings with a novel energy function. It combines boundary dissimilarity and depth dissimilarity to approximate hyperbolic properties like transitivity and layer separation. The method supports various geometric forms (e.g., boxes, balls) and integrates into existing ontology embedding frameworks. RegD achieves state-of-the-art results on multiple hierarchy and ontology benchmarks.

### Strengths
Flexible & Generalizable: Works with different region types (boxes, balls) and integrates into ontology models like ELBE and ELEM.

Hyperbolic-Like Expressiveness in Euclidean Space: Captures hierarchy depth and transitivity via depth dissimilarity without requiring hyperbolic geometry.

Strong Empirical Performance: Achieves state-of-the-art F1 scores on both DAG inference and ontology reasoning tasks.

### Weaknesses
Explain the further application of such method. Would you be able to use it in conjunction with a Neural Encoder derived from LLM?

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
