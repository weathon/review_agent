# Necessary Conditions for Compositional Generalization in Visual Models

- Decision: Reject
- Scores: 6, 2, 8, 6, 2

## Abstract
Compositional generalization, the ability to recognize familiar parts in novel contexts, is a defining property of intelligent systems. Modern models are trained on massive datasets, yet these are vanishingly small compared to the full combinatorial space of possible data, raising the question of whether models can reliably generalize to unseen combinations. To formalize what this requires, we propose a set of practically motivated desiderata that any compositionally generalizing system must satisfy, and analyze their implications under standard training with linear classification heads. We show that these desiderata necessitate \emph{linear factorization}, where representations decompose additively into per-concept components, and further imply near-orthogonality across factors. We establish dimension bounds that link the number of concepts to the geometry of representations. Empirically, we survey CLIP and SigLIP families, finding strong evidence for linear factorization, approximate orthogonality, and a tight correlation between the quality of factorization and compositional generalization. Together, our results identify the structural conditions that embeddings must satisfy for compositional generalization, and provide both theoretical clarity and empirical diagnostics for developing foundation models that generalize compositionally.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper investigates compositional generalization from a formal perspective. The authors put forward three properties (divisibility, transferability, and stability) which they argue argue should be fulfilled by a system achieving compositional generalization. From these, they derive a set of implications (e.g., additive linear factorization, cross-concept orthogonality)  that the representations of an encoder model need to satisfy to generalize compositionally under a linear readout. Finally, the authors provide some empirical evidence that the representations of CLIP and SigLIP models exhibit linearity and orthogonality (above a random encoder baseline) and that the degree of linearity correlates positively with compositional generalization performance.

### Strengths
- The approach of looking at compositional generalization from an idealized perspective (the proposed desiderata) and then formally deriving necessary properties of the corresponding representations is good and distinguishes this work from many other papers that study compositional generalization in existing (imperfect) models which usually do not achieve full transferability.
- Formally deriving the necessity of intuitive properties such as linearity and cross-concept orthogonality in the studied setting (Proposition 1) is a nice result and likely hints why compositional generalization is usually very challenging in practice.
- Finding evidence for the idealized requirements in actual CLIP/SigLIP models significantly strengthens the credibility of the claim and the proposed desiderata. Further, showing a positive correlation between linearity and compositional generalization also supports the theoretic findings.

### Weaknesses
- The figures of the paper could be noticeably improved. Most figures (e.g., Figure 1 & 6) are not easily understandable by just looking at them as well as not properly explained in the caption or referencing text. For example, for Figure 6 I assume that it depicts two scenarios where the bound $d \geq c$ is tight for one but not the other but it is not properly explained why this is true. In general, it could help to not only state what the figure does (e.g., Fig. 3/4) but also give an explanation or interpretation in each caption.
- While more realistic datasets with compositional structure often only have few concepts and contain some noisy or wrong concept labels, I think it would significantly strengthen the work to also include one or two more such datasets in the analysis, e.g., DomainNet [1] or ImageNet-AO [2].

### Comments
- It would be nice to provide short proof sketches or intuitions in the main text (similar as for Proposition 1) for the other propositions to give some credibility and/or confidence about the respective statements.
- I think the paper would benefit from a short discussion to better contextualize some of its findings. For example, perfect compositional generalization requiring linear factorized concept embeddings is something many people might have expected to be true, so the finding might not appear as interesting. Further emphasizing the added value of providing a theoretic derivation for this could be useful.

---
[1] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. "Moment matching for multi-source domain adaptation." International Conference on Computer Vision. 2019.

[2] Reza Abbasi, Mohammad Hossein Rohban, and Mahdieh Soleymani Baghshah. "Deciphering the role of representation disentanglement: Investigating compositional generalization in CLIP models." European Conference on Computer Vision. 2024.

### Questions
- Another work by Abbasi et al. [1] has also investigated compositional generalization and its correlation with representation disentanglement. They find that higher disentanglement also correlates with better compositional generalization. Can the authors either discuss the findings of [1] as related work and/or in context of the proposed desiderata or argue why the work might not be relevant?
- I'm a bit confused by Definition 3. It intuitively states that an encoder $f$ is compositional if there exists **any** function $h$ that correctly classifies all concepts $c_i$ from $f(\mathbf{x_c})$. In the general case (where $h$ can be non-linear), wouldn't any injective encoder $f$ be compositional then (simply define $h : f(\mathbf{x_c}) \mapsto \mathrm{onehot(c)}$)? I understand that such a function would likely not be learned in practice but from mathematical perspective it fulfills the definition. Maybe the definition could either focus only on linear functions $h$ where this cannot happen or be adapted in some other way that accounts for this.

---
[1] Reza Abbasi, Mohammad Hossein Rohban, and Mahdieh Soleymani Baghshah. "Deciphering the role of representation disentanglement: Investigating compositional generalization in CLIP models." European Conference on Computer Vision. 2024.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose a set of necessary conditions that  compositional representations should posses. They call these desiderata and include several sensible notions that seem intuitive: divisibility, transferability and stability. These crystallize the notion that compositional representations should be identifiable (divisibility), that this should hold for unseen combinations (transferability) and that these should remain constant across different spaces which share concepts (stability). They authors emphasize how these conditions lead to representations that show compositional generalization.

### Strengths
The authors are very thorough in the way they define their desiderata. Furthermore they explore the implications of said desiderata for the geometry of the learned representations across several models in the Visual Language Model category.

### Weaknesses
The main weakness is that similar ideas and considerations have already been explored under a different name.

The idea of representations being divisible, stable and transferable was already explored in the context of disentangled representations, which essentially follow the desiderata proposed here (Higgins et al., and Watters et al and especially, Eastwood and Williams. Note: there are two issues discussed in this literature, how to learn disentangled representations, and whether this lead to compositionally). Incidentally, Watters et al., already explored the implications that this has for compositional generalization, though admittedly this is in their appendix and is not very systematic. A more systematic exploration was done in Montero et al., which showed that even if the equivalent of the desiderata above held, models those models couldn't generalize (see the result for the GT decoder. Note that this is pre LLM, so those weren't considered). I include this because this is very relevant literature which the authors do not reference, or mention yet covers much of the ground that they are retreading.

As for the implications for compositional generalization, we must start by discussing a fundamental misunderstanding about what compositionality is. Briefly, the definition of compositionality states that the meaning of a proposition depends on the semantic meaning of its parts, and the relations between them (i.e how they are bound into some "expression" whatever form this may take). In the case of the generalization part, it states that you should be able to recombine parts in order to perform some task and exhibit two properties systematicity and productivity( though the latter is disputed. See the Fodor and Pylyshyn reference).  

In my view, there are thus two issues here: first, the authors are studying pre-trained models which they cannot say have not seen the dataset in question (or similar enough ones) and thus it is not true that they are generalizing. The only claim that they can make is that their (linear) probes, are compositional because they learn to predict specific concept values for unseen combinations. This is fine, but servers more as an exploration of how linear the geometry of these models are, then about compositional generalization more broadly. 

Second, simple prediction is really just the bare minimum since it doesn't require models understand how the different concepts relate to each other. If they can read out the information from the representation then they can ignore all other information. This is why the studies above use image generation: it forces the model to learn how the different factors interact i.e. how the semantic meaning of the full proposition depends on the semantics of the parts and how they relate to each other. 


References:
Higgins, I., Amos, D., Pfau, D., Racaniere, S., Matthey, L., Rezende, D., & Lerchner, A. (2018). Towards a definition of disentangled representations. arXiv preprint arXiv:1812.02230

Watters, N., Matthey, L., Burgess, C. P., & Lerchner, A. (2019). Spatial broadcast decoder: A simple architecture for learning disentangled representations in vaes. arXiv preprint arXiv:1901.07017.

C. Eastwood and C. K. I. Williams, “A framework for the quantitative evaluation of disentangled representations,”
in International Conference on Learning Representations, 2018.

Montero, M. L., Ludwig, C. J., Costa, R. P., Malhotra, G., & Bowers, J. (2021). The role of disentanglement in generalisation. In International Conference on Learning Representations.

### Questions
1. The authors need to explain how their proposals are different from the ones mentioned above
2. Also how it is possible to control for the issue of VLMs having access to the test data.
3. Can they find a harder task that shows stronger compositionally?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This manuscript investigates how the format of model representations influences the ability to compositionally generalize, formalizing compositional generalization ability into three desiderata. They prove theoretically that their desiderata are only satisfied by models with a linearly factorized representation immediately before the readout (if those models a trained on crossentropy with gradient descent). Moreover, different concepts should be represented in orthogonal directions whereas, in principle, the representation could collapse different categories for the same concept into the same dimension. They then test their predictions in pretrained vision models and find that they are indeed more linearly factorized after pretraining than with random weights, their degree of linear factorization predicts their model performance, and distinct concepts are indeed represented more orthogonally than distinct categories of the same concept.

### Strengths
I liked this paper! It provides a formally principled perspective on compositional generalization and a useful theoretical intuition on the representations we'd expect in such models. It also provides a principled set of experiments to test its theoretical predictions. Below I'm highlighting a few things I particularly liked:

- The paper does a great job at motivating the problem of compositional generalization. It is written very clearly; it is apparent that the authors have thought carefully about how to present their findings in an accessible manner.
- I found proposition 2 particularly thought-provoking and the illustration was helpful in getting an immediate intuition for how to think about this
- The empirical findings are very interesting, in particular the fact that linear factorization predicts compositional generalization and that distinct categories are represented in a less orthogonal manner than distinct concepts

### Weaknesses
Currently, the theory in the paper imposes a pretty strong constraint in the form of stability: to leave generalization completely unchanged across different training sets, we need exactly linearly factorized representations. I'm not entirely convinced by the authors' justification for the desideratum of stability; while this may be a relevant starting point, I think it substantially limits the impact of the theory. In particular, the authors find that in practice representations are certainly not perfectly linearly factorized (as their theory would prescribe). In its current shape, we can't use the theory to interpret whether an R^2 of 0.4-0.6 is good or bad. That said, I do appreciate the experimental observation that an increased R^2 yields improved compositional generalization, which helps with bridging this gap. I also still appreciated the theoretical conceptualization.

I also think it would be good to further contextualize this theory in existing work on relating representational geometry and compositional generalization. In particular, [1] and [2] (and [3], which the authors do cite) both tie compositional generalization to linearly factorized representations in the output representational space. Notably, both [2] and [3] formalize this question through a kernel perspective which is highly related to the SVM perspective leveraged in the proof here. On the conceptual end, I think the "related work" section (l. 82-92) mostly talks about how linear subspaces have been identified. However, many papers also emphasize the usefulness of factorized representations or try to identify factorized/disentangled representations [4,5]; I think the current paper goes beyond this work by providing a particular formal perspective on this, but it may be useful to draw a connection.

Some minor notes:
- L. 75-79: I think the verb is missing in this sentence.
- L. 91: I imagine you don’t want to have the “we” in italics?
- In Figure 5 I would make the light orange data points a little more salient, they are a little hard to see right now.
- L. 170: “practise” -> “practice”
- L. 298-299: I think there is a spare “either” there.
- L. 347: “factoried” -> “factorized”
- L. 361: Why is there a c in the superscript, aren’t you counting through i\in[c] in the subscript?
- L. 389 “Generaliaztion” -> “Generalization”

1. https://arxiv.org/abs/2409.14981
2. https://arxiv.org/abs/2501.18797
3. https://arxiv.org/abs/2405.16391
4. https://openreview.net/forum?id=Sy2fzU9gl
5. https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/building-machines-that-learn-and-think-like-people/A9535B1D745A0377E16C590E14B94993

### Questions
- Could you provide some more context on stability as a relevant desideratum and talk about how the theory could be extended beyond this specific desideratum, e.g. to clarify how different degrees of linear factorization influence compositional generalization? (See notes under "Weaknesses".)
- Could you contextualize the paper a bit further in the prior literature? (See notes under "Weaknesses".)
- Do you have any intuition for how linear the representations observed in these models actually are? I realize that this is a somewhat vague question but I found it a bit difficult to interpret an R^2 between 0.4-0.6. It may be useful to indicate the R^2 of e.g. the kinds of schematic representations you are showing in Fig. 5?
- Doesn't transferability imply divisibility?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the structural properties that representations must have for models to generalize compositionally, in other words, recognize familiar parts within new combinations. The authors argue that since models are trained on a tiny subset of all possible concept combinations, to generalize reliably their embeddings must satisfy certain necessary conditions.

They define three key necessary desiderata for compositional generalization: divisibility, transferability, and stability. From these, they argue that embeddings must be a linear combination of per-concept components and that the components must be nearly orthogonal. Furthermore, the embedding dimension must be at least as large as the number of independent concepts.

Thesee ideas are tested empirically on variants of CLIP and SigLIP across datasets such as PUG-Animal, dSprites, and MPI3D. The authors find that compositional generalization requires this linear, nearly-orthogonal structure. These findings support a practical diagnostic  method for checking whether models can generalize compositionally.

### Strengths
The paper identifies a set of simple principles (divisibility, transferability, and stabiliity) as the foundation for compositional generalization, and the math nicely shows that these principles directly lead to linear factorization and orthogonality in embeddings.

The paper shows how $R^2$ and factor orthogonality can measure how compositional a model’s features are, which is an interesting and practical idea.

The text is will structured, with each section including clear takeaways and helpful diagrams supporting the arguments.
The paper gives a useful geometry-based checklist for when models can or cannot generalize compositionally.

### Weaknesses
The theory relies on strong assumptions such as linear readouts, binary concepts, and ideal separability. Extending the experiments to incude models with nonlinear heads or continuous attributes would help test how far the results generalize.

Although the concept of "Projected R^2" is used throughout the paper, it is only described in Appendix B. It would help to have at least a brief conceptual explanation of it when it is first introduced (line 368), and the use of whitening.

Even though the paper introduces interesting theoretical and empirical results about linear factorization, orthogonality, and dimensionality, these ideas could be more useful in practice if the authors at some point explicitly framed them as a diagnostic toolkit for model design. For example, they could summarize how to measure each property (Projected R2, concept orthogonality, and factor rank) and explain how those metrics might guide architecture choices for new problems/datasets.

### Questions
How much does the Projected R^2 metric depend on whitening?

What counts as a "good" value for R^2?

How are the concepts defined in datasets like PUG-Animal or MPI3D?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper presents theoretical and empirical analysis into the properties needed for visual encoding models to exhibit compositional generalization. The paper states three desiderata for embeddings to be considered to generalize compositionally, and then shows that any embeddings that meet these desiderata must decompose embeddings into the sum of orthogonal concept representations (linear factorization + orthogonal factors). Empirically, experiments on several models and datasets show that vision models show some degree of linear factorization, the degree of factorization is correlated with compositional generalization accuracy (using linear probes), and different concept factors are "nearly orthogonal".

### Strengths
- This paper investigates the interesting question of how neural networks trained on limited data are able to generalize compositionally. I think this is a useful topic--understanding compositional generalization could improve our theoretical understanding of neural networks and motivate practical improvements to data or architecture.  

- The organization of the paper is clear and the paper is mostly easy to follow.

- The paper conducts extensive empirical experiments to support the predictions, including a variety of image models and datasets.

### Weaknesses
- The two conditions identified in this paper (linear factorization + orthogonal factors) might be limited contributions. The linear factorization condition seems like a straightforward consequence of the definition of compositional generalization, which requires that concepts can be classified with a linear probe. The condition of strictly orthogonal factors is likely never actually met in practice--concepts are only approximately orthogonal. (I would think that dimensionality is typically less than the number of concepts.) One way to strengthen the paper could be to discuss in more detail how the desiderata and conclusions change given approximate orthogonality rather than strict orthogonality. See relevant discussion in [1].

- A number of the empirical conclusions are based on thresholds that seem subjective to me, so it is hard to say how much they really substantiate the theoretical claims. Section 5.1/figure 7 states that embeddings exhibit "substantial" linear factorization, but it is not clear to me what $R^2$ should be considered substantial, and what we should conclude from the fact that the $R^2$ scores are still much less than 1.0. The takeaway in section 5.3 is that "per-concept difference vectors are nearly-orthogonal across concepts", but I am not sure what the threshold is for defining "nearly-orthogonal". Similarly, in section 5.4, Figure 10 seems to show that per-concept dimensionality is actually relatively high for categorical features, so it is not clear to me that we can conclude that "overall, semantic factors are low-rank". In general, I think the paper would be stronger if it made more formal, precise predictions.

- The main text of the paper is missing some methodological details, making it difficult to interpret the empirical results. For example, how exactly is compositional accuracy calculated in section 5.2? The paper also does not give a clear definition of Project $R^2$ or explain why it should be considered a good measure of linear factorization (although there is some discussion in the appendix). It would be helpful to include more experimental details in the main text. 

- Related work: I am not very familiar with this area and I found it difficult to fully understand the relationship between this paper and the prior work. For example, in section 2, the paper mentions prior work that has "emphasized formal sufficient conditions for generative systems". How does that differ from this work? I appreciated the illustrations of previous work in Figure 2, but these works are not discussed in the text, so it is still hard to understand the relationship with prior work. 

- I found the notation to be difficult to follow at times. For example, $c$ is used to denote the number of concepts, but $c_i$ is used to denote the $i^{th}$ entry in the concept tuple $\mathcal{c}$. Please see more detailed comments in the Questions section.


**Summary:** I think the paper presents an interesting analysis of compositional generalization, with a variety of empirical evidence to support the conclusions. However, I feel that presentation issues make it difficult to understand the experiments and the notation; the interpretation of the experiments might be somewhat subjective; and the contributions might have limited significance. I would consider increasing my score if the authors could suggest ways to make the empirical predictions more precise; and address my concerns about clarity; and better contextualize the contributions with respect to prior work.


_References_

[1] Elhage et al., 2022. Toy Models of Superposition.

### Questions
- Can you expand on the role of the "validity class" (bottom of page 3)? It seems that this can be any arbitrary rule for defining constraints on the training data. Do the desiderata and conclusions apply to any choice of validity class? If not, how is the validity class identified?

- In Proposition 1 (line 299), what is the justification for the validity rule $|T| = 2^{n-1} + 1$? Previously, $n$ was the number of values each concept could take on. In this binary case, $n = 2$. I would have thought that the validity rule should depend on the number of concepts.

- Why is desideratum 1 (divisibility) necessary? Doesn't desideratum 2 (transferability) imply desideratum 1?

- Desideratum 3: It seems that this implies constraints on the validity rule for determining valid supports.


- In sections 5.1 and 5.2, can you explain what exactly Project $R^2$ is measuring and why this is a measure of linear factorization?

- In line 160, should $\mathcal{T} \subseteq 2^{\mathcal{C}}$ be $\mathcal{T} \subseteq n^c$? Similarly in Definition 2.

- What is $\mathcal{X}'$ in equation 4?

- What is $\mathcal{F}$ in Figure 3?

- In figure 10a, what are the annotations above each of the columns?

- I do not quite understand what Figure 1 is meant to illustrate. It seems that the two concepts are "is cat present" and "is person present", but both the training image and the testing image show the same composition of concepts (cat present = true, person present = true), so it doesn't seem like an illustration of compositionality.

- Section 5.4 mentions that some of the concepts in the experiments are continuous rather than discrete, but the theoretical analysis assumes discrete concepts. Can you comment on how the desiderata and conclusions extend to continuous concepts?

### Soundness
2

### Presentation
2

### Contribution
1
