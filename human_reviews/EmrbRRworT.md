# ENHANCEMENT OF GNN’S EXPRESSIVE POWER VIA RECONSIDERING MODAL LOGIC

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 1, 3

## Abstract
Since AC-GNNs, in which nodes only gather information from their neighbors to update features at each layer, are limited in their expressive power, numerous models have been proposed to enable GNNs to go beyond Weisfeiler-Lehman (WL) test. However there still a lack of effective methods to measure these models' expressive power: for a specific task, it is still difficult to figure out whether the model is competent for the task. We tackle this problem by finding equivalent Boolean classifiers logic for models. By checking whether the task is able to be expressed as model's equivalent Boolean classifiers logic formula, we can be aware of whether the model is competent for task. We propose a framework for AC-GNNs, denoted as l-div AC-GNNs, to enhance AC-GNNs' expressive power. To be more specific, we classify node's neighbors according to existence of different length of paths from node's neighbors to itself. To find l-div AC-GNNs' equivalent Boolean classifiers logic, we introduce the l-div graded modal logic and prove that a Boolean logical classifiers can be expressed by l-div graded modal logic if and only if there exists a l-div AC-GNN which is able to capture it. In this paper, three properties are defined for the framework: invariance and equivariance, approximation and logic expressive power, we proved l-div AC-GNNs are possessing with these properties. A series of tasks have been implemented to validate our theoretics, the results of experiments demonstrate the validities of both our method to measure models' expressive power and expressive power of l-div AC-GNNs.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper studies the uniform expressive power of GNNs extended with information about k-hop neighborhoods. In particular, it generalizes results of Barceló et al. connecting the uniform expressive power of GNNs and graded modal logic.

### Strengths
The paper deals with an interesting problem.

Some of the results seem non-trivial: there is potential on the paper.

### Weaknesses
Unfortunately, the presentation of this paper is horrible. Not only many sentences are broken, but also the paper is full of typos and undefined notions. The english is extremely poor, which makes the understanding of the notions introduced simply impossible. I declare myself to be an expert on this topic, but this paper I could not understand at all. As an example, one of the main theorems of the paper, the one claiming that a certain logic coincides with GNNs, is presented without having introduced GNNs before! 

I have the feeling that there is something interesting, and even deep, in this paper, but the authors need to work extensively on the presentation for this to become clear. As such, it is not at the level of what I'd consider "reviewable" for NeurIPS.

### Questions
No questions. I suggest the paper to be rewritten extensively.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes an enhanced framework for studying message passing neural networks, and an enhanced graded modal logic to recreate results of Barceló et al 2020. 
I found the paper quite difficult to read, most of it due to shortcomings of the authors in making the definitions clear; see for example the definition of k-hop (l_1,...,) neighbourhood. But if I understood correctly, the proposal is to create masks of the local structure of each node, but where masks are given in terms of adding, or removing, certain nodes from the graph. If so, then the paper has two main weak points. First, resulting GNNs are not permutation-invariant (unless one permutes everything in the GNNs accordingly, but this is not treated in the paper). And second, the paper misses comparation on more recent literature regarding similar techniques.

### Strengths
* New model of GNNs based on masking certain nodes in the graph.

### Weaknesses
* Proposal assumes a given adjacency matrix and therefore resulting GNN architectures are not permutation-invariant. This is extremely important, and If one is willing to loose permutation invariance then probably a transformer is the way to go for graphs. 
* The proposal should probably fit in the framework given by Qian et al  in Neurips 2022 (altough the latter retains permutation invariance). 
* Paper is difficult to read, and definitions / notation is not clear enough. For example, how do I distinguish from the neighbourhood of a node when I mask just the node v_1, and the neighbourhood when I mask just the node v_2?

### Questions
GNN models that are not permutation invariant should be much better motivated, I think a good comment in this direction is missing.

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
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper the connection between GNNs and modal logic, as outlined in Barceló et al, is extended using a more complicated higher-order modality. In particular, instead of the simple edge modality, exponentially many modalities are considered, each of them encoding directed path (non)-existences between nodes. It is shown that logical classifiers expressible by an extended notion of GNNs correspond precisely to those expressible in the extended modal logic, hereby generalising the result by Barceló et al. A consequence is that the extended modalities and GNNs have higher expressive power.

### Strengths
1. The use of extended modalities and their use to construct expressive GNN is novel.

2. The main result requires some non-trivial analysis of logical formulae.

### Weaknesses
1. The paper is not well written and very difficult to follow. The authors should present their work in a more clear way.

2. Large part of the paper (including all experiments and all comparisons with recent higher order GNNs) are deferred to the appendix. I did not consult the appendix. 

3. The proposed approach depends on the introduction of exponentially many different neighborhood (exponential in l). This limits the practically relevance for higher values of ell.

### Questions
**Q** Please explain more clearly how you can check triangles using your method. Section 3.2 is the main part of the paper but it is very badly presented.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
