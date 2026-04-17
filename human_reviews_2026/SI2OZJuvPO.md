# Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Understanding internal representations of neural models is a core interest of mechanistic interpretability. Due to its large dimensionality, the representation space can encode various aspects about inputs. To what extent are different aspects organized and encoded in separate subspaces? Is it possible to find these "natural" subspaces in a purely unsupervised way? Somewhat surprisingly, we can indeed achieve this and find interpretable subspaces by a seemingly unrelated training objective. Our method, neighbor distance minimization (NDM), learns non-basis-aligned subspaces in an unsupervised manner. Qualitative analysis shows subspaces are interpretable in many cases, and encoded information in obtained subspaces tends to share the same abstract concept across different inputs, making such subspaces similar to "variables" used by the model. We also conduct quantitative experiments using known circuits in GPT-2; results show a strong connection between subspaces and circuit variables. We also provide evidence showing scalability to 2B models by finding separate subspaces mediating context and parametric knowledge routing. Viewed more broadly, our findings offer a new perspective on understanding model internals and building circuits.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Neighbor Distance Minimization (NDM) as a method for performing unsupervised identification of interpretable subspaces within the latent spaces of neural models. NDM operates under the hypothesis that groups of mutually exclusive features form subspaces, and the features within these subspaces exist within superposition. Through experiments on GPT-2, the authors demonstrate that these subspaces behave similarly to the “variables” used by the model. This approach provides a new avenue for interpretability research within and between subspaces.

### Strengths
1. Motivation and Construction: The motivation of NDM and its subsequent demonstration on the toy models of superposition setting provides a clear intuition as to why we’d expect it to work on larger models and how we can interpret its results.
2. Novel Perspective: The developed perspective offers an alternative avenue for future research in the field of interpretability, specifically by considering the interaction between or within subspaces.

### Weaknesses
1. Lack of Details on the Computational Aspect of NDM: There is a lack of information regarding the computational burden imposed by using NDM. Indeed, 2B parameter models are not large-scale by current standards. This raises questions about the practicality of these methods in real-world settings.
2. Single Domain Evaluation: NDM is only evaluated on text models. It would also be important to evaluate the effectiveness of exploring image models.
3. Comparison to other Interpretability techniques: Although the intuitive differences between NDM and other interpretability methods, such as sparse autoencoders, are provided, no practical comparison is made.
4. Limited Granularity: By construction, NDM can only identify as many subspaces as there are dimensions in the latent space. In fact, it is much less than that, as the subspaces often have relatively large dimensions. Therefore, it is unclear how granular these “variables” corresponding to subspaces are, and it is likely that additional interpretability techniques would have to be applied to the individual subspaces.

### Questions
1. What are the computational requirements for applying NDM?
2. How sensitive is NDM to the provided N model activations?
3. Have you explored an iterative application of NDM? Namely, re-applying NDM to the identified subspaces?

### Soundness
3

### Presentation
3

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
The paper begins from the premise that mutual exclusiveness could be a fundamental condition for superposition. Building on this idea, it proposes that data may contain groups of mutually exclusive features; for example, features encoding categorical variables such as different subjects, where only one is active at a time (L. 110). Motivated by this observation, the paper introduces an unsupervised method, termed NDM, designed to identify subspaces of mutually exclusive features. The method does so by finding subspaces in which data points are projected to similar locations. The paper demonstrates the effectiveness of NDM both on a synthetic (toy) model and on representations from large language models.

### Strengths
- I really appreciate the core idea and intuition behind the argument that mutual exclusiveness may be a fundamental condition for superposition, as well as the proposed method built on this insight.  
- The results on the toy model effectively illustrate and complement the central narrative.  
- The paper does a good job explaining the intuition behind the approach and is, for the most part, well written and thorough in its exposition.  
- Overall, I found this to be one of the more enjoyable papers to read, with a clear take-home message.
- While the evaluation still leaves considerable room for improvement, the paper does at least include some empirical assessment of the proposed approach.

### Weaknesses
- I did not find the qualitative examples in Figure 2 particularly convincing. The positions shown in panel (c) still exhibit a fairly large range, and for the other examples, it seems plausible that sparse autoencoders could identify similar concepts. Given the narrative developed in the paper, I would have expected the qualitative examples to focus more on illustrating the idea of distinct variables, rather than on specific concepts that might also be captured by SAEs.
- In Table 1, it is unclear why no comparison to sparse autoencoders (SAEs) is included. A theoretical justification would be sufficient if there is a solid reason for omitting such a comparison. Additionally, it would be helpful to include an analysis of how the number of subspaces affects the results. For instance, if only a single subspace were used, the condition that the high-level "variable" lies within the same subspace (L. 261) would trivially hold.
- The method relies on the mutual information (MI) threshold, but it is not clear how this threshold should be selected in practice. Although the paper includes ablation studies on this parameter, the specific effect of the threshold remains unclear, as the quantitative experiments are relatively small in scale.

### Questions
- In Eq. (1), why is $h = W x x'$ instead of $h = W x$, as used in the experimental setup of Elhage et al. (2022)? It is possible that the equation is incorrectly formatted and that $x'$ is intended to appear on the right-hand side?
- How to choose the MI threshold and how sensitive is the method to that threshold?
- How would sparse autoencoders (SAEs) perform under the proposed evaluation setup, both in quantitative metrics and in qualitative analyses?
- According to the manuscript, superposition occurs when mutual exclusiveness holds, and the proposed method aims to identify subspaces of mutually exclusive features. However, it is not clear why we should expect these subspaces to be inherently interpretable if superposition still occurs within them. Wouldn’t we, in many cases, need an additional method to disentangle or remove the remaining superposition? Furthermore, how can we identify situations in which such additional disentanglement would be necessary?

Due to the open questions regarding the evaluation and the distinction from sparse autoencoders (SAEs), I would currently lean toward rejecting the paper. However, I believe the work has potential, and if my concerns are addressed convincingly and no major issues are raised by other reviewers, I would be happy to reconsider and improve my score.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
A core objective in interpretability research is to understand content and geometry of activations. The authors propose a novel, original idea to learn a set of linear subspaces, where each subspace holds mutually exclusive features encoded via superposition. Different subspaces are orthogonal thus independent. The authors propose a method to learn these subspaces unsupervised from activation data alone, by learning a rotation matrix (similar to DAS) such that data points within each subspace have minimal distance. They provide intuitive understanding, show their approach in toy models and prove its applicability in small and medium language models. Specifically, they show for some known tasks that all features of interest lie in the same subspace, and they show that subspaces are interpretable (ie monosemantic).

The paper is well-written, proposes an original idea, and elegantly combines mathematical intuition, toy models, and translation to LLMs. My main concern is that some assumptions/hypotheses were not empirically validated. For example, it wasn't shown that features within a subspace are mutually exclusive, or that the orthogonality requirement is faithful of real activation space geometry. They critique SAEs but don't use them as baselines.

### Strengths
- great presentation, paper is well-written and I found it easy to follow. The paper provides both intuitive understanding and mathematical precision.
- the paper elegantly combines mathematical intuition, validation in toy models, and translation to LLMs
- the paper posits an interesting, novel idea for an important problem rather than an incremental improvement
- I do find the author's work interesting from a slightly different perspective that wasn't as highlighted as it could: Feature independence. When steering with SAEs, editing individual features results in an OOD reconstruction because many SAE features co-occur which is a real problem.

### Weaknesses
Major:
1. "Mutual exclusiveness" doesn't seem like a more fundamental condition, or much different from sparsity at all. Elhage 2022 say that when features are sparse, they can be encoded via superposition. Sparse features are already "almost mutually exclusive" but the "almost" seems important as strong guarantees are hard to make for neural networks. 
The authors write that superposition wouldn't work if 2 or more features are active but this is false. Yes, interference and error would grow, but only by a tiny amount in practice. Best proof are SAEs that work well with e.g. 20 features active. 
In fact, feature co-occurrence is baked into superposition theory: A prediction of their theory and toy models is that features that are mutually exclusive should make heavy use of superposition, while features that often co-occur, should be better separated, ie encoded as orthogonal directions (at least more orthogonal than mutually exclusive ones). So no new theory is needed to predict geometry of feature groups so I don't understand how "mutual exclusiveness and feature groups" are fundamentally different from "sparsity and superposition". It's good to have methods to decompose this, but it follows from superposition theory, and not new theory/hypothesis has to be invented.
2. The authors assume that features within a subspace are mutually exclusive but you never test this (in fact, you don't even extract features within a subspace).
3. The authors for some tasks that most task-relevant features land in a single subspace but they don't show that this subspace only contains those features, aka is interpretable and monosemantic. It could very well be the case that completely unrelated information is stored within the same subspace as many unrelated features are mutually exclusive.
4. The number of subspaces is strictly bound by the dimensionality of activation space as they must be exactly orthogonal. This might limit the method's potential and interpretability. As we see with SAEs, representation space can be well-approximated with tens of Millions of features and having only few hundred feature groups available at most might be quite limiting, and many different features might be packed into the same subspace (because different features are often mutually exclusive as well). This would limit interpretability. My concern is that in reality, the independence-between-feature-groups assumption doesn't hold 100% and we would be better off with relaxing this restriction a little to allow more feature groups to exist and be more interpretable. A possible approach here could be to use MOLTs (Lindsey, Anthropic 2025).
5. The authors heavily criticize superposition and SAEs but they never directly compare against SAEs although direct comparisons should be possible. They state that other baseline comparisons aren't possible because their subspaces are learned unsupservised from activations, but SAEs are as well. In fact, I think that their method doesn't disagree with superposition and sparsity at all.


Other things that would improve the paper:
- Insight into LLM computation. This paper mainly proposes a new method and validates it but there's no new mechanistic insight about LLM computation. It would improve the paper a lot if the authors could prove by example that this method can discover things that other methods like SAE, DAS, etc can't
- More experiments with LLMs. Qualitative examples in Figure 2 lack rigor or quantification. IOI/greater than results don't measure logit difference recovered when only patching an individual subspace, more evidence that subspaces are monosemantic and interpretable, etc.

### Questions
1. Many things are mutually exclusive. For example, if the text is about software licenses, it's not about a fiction novel or a mental health consultation. Features from all these different contexts could be squished into a single subspace. Especially since the number of subspaces is extremely limited. Do you observe this in practice? Doesn't this imply that mutual exclusiveness != interpretability? Doesn't this hurt interpretability of those subspaces and the applicability of this method a lot?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper describes and evaluates an unsupervised method — neighbor distance minimization — for decomposing representation space into interpretable subspaces.

### Strengths
**Well-written with clear explication:** The paper presents complex concepts with clear language, helpful intuitions, and useful examples.

**Vast amount of work:** The paper, combined with the appendices, reflects a vast amount of experimental work.

### Weaknesses
**Evaluation**: As the authors say, “The key question is NDM’s applicability to real-world neural models.” The evaluation relies on the intuition that “when processing inputs, key intermediate results should ideally lie in a single subspace.” On the one hand, this idea is appealing, partially because it provides a fairly clear evaluation criterion for methods for decomposing representation space into interpretable subspaces. On the other hand, this criterion could both under under-determine and over-determine useful results. That is, essentially useless decompositions could satisfy this criterion, and useful decompositions could violate it. Ultimately, a more convincing evaluation would use an end-to-end criterion (i.e., one that shows that an MI pipeline produces more useful results when it includes NDM rather than some other subspace discovery approach). That’s a very tall order, but one that is far less error-prone than this paper’s current evaluation criteria. This reflects a more general problem with current MI research: whole MI pipelines cannot be easily created because we don’t have all the components, and candidate components cannot easily be created because we don’t have the pipeline.

### Questions
In Section 5.1, you say that “The results of NDM using the best hyperparameters are shown in Table 1…”  This raises the possibility of overfitting because only the “best” hyperparameters are shown. How were the best hyperparameters selected, and is this method reasonable in practice?

### Soundness
3

### Presentation
4

### Contribution
3
