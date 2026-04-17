# Quasi-Equivariant Metanetworks

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Metanetworks are neural architectures designed to operate directly on pretrained weights to perform downstream tasks. However, the parameter space serves only as a proxy for the underlying function class, and the parameter-function mapping is inherently non-injective: distinct parameter configurations may yield identical input-output behaviors. As a result, metanetworks that rely solely on raw parameters risk overlooking the intrinsic symmetries of the architecture. Reasoning about functional identity is therefore essential for effective metanetwork design, motivating the development of equivariant metanetworks, which incorporate equivariance principles to respect architectural symmetries. Existing approaches, however, typically enforce strict equivariance, which imposes rigid constraints and often leads to sparse and less expressive models. To address this limitation, we introduce the novel concept of quasi-equivariance, which allows metanetworks to move beyond the rigidity of strict equivariance while still preserving functional identity. We lay down a principled basis for this framework and demonstrate its broad applicability across diverse neural architectures, including feedforward, convolutional, and transformer networks. Through empirical evaluation, we show that quasi-equivariant metanetworks achieve good trade-offs between symmetry preservation and representational expressivity. These findings advance the theoretical understanding of weight-space learning and provide a principled foundation for the design of more expressive and functionally robust metanetworks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the concept of quasi-equivariance specifically for metanetworks, a novel framework designed to overcome the limitations of strict equivariance in metanetworks. Metanetworks are models that process the parameters of other neural networks, and a key challenge is functional equivalence—where different parameter sets can yield the same input-output function. While strictly equivariant metanetworks respect the symmetries underlying functional equivalence (e.g., neuron permutations), they can be overly restrictive and lack expressivity.

This work proposes quasi-equivariance as a principled relaxation. A map $F$ is quasi-equivariant with respect to a symmetry group $G$ if for any transformation $g \in G$ applied to its input $\theta$, its output is transformed by some other group element $g' \in G$, i.e., $F(g\theta) = g'F(\theta)$. This ensures the output remains in the same functional equivalence class, thus preserving functional identity, while allowing for greater flexibility than the strict condition $F(g\theta) = gF(\theta)$.

The paper provides a theoretical foundation for this concept, connecting it to the notion of a maximal symmetry group, and proposes a constructive method for building quasi-equivariant layers. This is achieved by augmenting a standard equivariant map $\beta(\theta)$ with a learned, parameter-dependent group action $\alpha(\theta)$, yielding a layer of the form $F(\theta) = \alpha(\theta)\beta(\theta)$. The framework's effectiveness is demonstrated by applying it to feedforward, convolutional, and transformer networks. Empirical results across three distinct metanetwork tasks—predicting model generalization, classifying implicit neural representations, and predicting transformer performance—show that quasi-equivariant layers significantly improve the performance of state-of-the-art metanetworks with only a minimal increase in parameter count.

### Strengths
Originality: This paper introduces the novel concept of "quasi-equivariance" for metanetworks, which is a significant and original extension to the existing paradigm of strict equivariance. This relaxation addresses a critical limitation of previous approaches, offering a new direction for designing more flexible and expressive models.

Quality: This paper provides a strong theoretical foundation for quasi-equivariance, connecting it to functional equivalence and maximal symmetry groups. The empirical evaluations are thorough, comparing the proposed method against several established baselines across diverse tasks, datasets, and architectures that the metanetwork processes, consistently demonstrating its effectiveness.

Clarity: This paper is overall well-written and well-structured, with clear explanations of complex concepts such as functional equivalence, symmetry groups, and the mathematical definitions of equivariance, supported by figures and tables of results.

Significance: The introduction of quasi-equivariance has significant implications for the design of metanetworks, enabling better trade-offs between symmetry preservation and representational flexibility. The empirical results demonstrate substantial performance improvements with minimal additional cost.

### Weaknesses
Scope Limitations: 
* Your proposed methods framework are only demonstrated on MLP metanetworks that operate on the parameters of MLPs, CNNs, and Transformers, whose symmetry groups have a relatively simple direct-product structure. You briefly mention that extending this framework to more complex architectures like graph-based metanetworks are only briefly mentioned, but expanding upon what the challenges are and how they might be tackled in a few sentences could provide better context for the audience.
* You provide clear evidence that the proposed quasi-equivariant framework is useful for metanetworks, but are there other fields of machine learning that could benefit from it? Adding a sentence with some ideas would help broaden the audience of this paper.

The implementation for MLPs and CNNs in Section 4.2 is currently only provided for MLPs, but the experiments focus on CNNs. Can you add a few sentences on how the provided MLP implementation translates to the CNN case?

Although $\text{GL}(d_h)$ is defined in the notation table, it would be worth also defining around line 344.

The text is table 3 is quite small and difficult to read. Please make it at least the same size as the text in the other tables.

Formatting: On line 066, use \citet instead of \citep to get "...class of MLPs, Fefferman & Markel (1993) established..."

### Questions
The weaknesses above are important to address in the main paper, even given the space constraints. The following questions can be addressed in an appendix and referred to in the main paper. 

Have you performed any analysis of the learned group elements from the map $\alpha(\theta)$ in the experiments? For instance, how far from the identity do the learned scaling factors or matrices tend to be, especially in the case of the experiments presented in section 5.1?

The construction of the map to $GL(n)$ via $\sin(\gamma(\theta)) \cdot \epsilon+ I_n$ is an interesting and practical choice. How sensitive is the model's performance to the hyperparameter $\epsilon$?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work posits the question whether strict equivariance is necessary for metanetworks, and introduces the notion of quasi-equivariance, allowing metanetworks to move beyond strict equivariance. The authors demonstrate the effectiveness of their method in a suite of experiments that includes INR classification, and predicting CNN/Transformer generalization.

### Strengths
The manuscript is well-written, and the math formalism is mostly clear.

The research question posed, i.e. whether strict equivariance is necessary for metanetworks is a very topical and significant question.

The proposed method is simple and has a small computational overhead.

### Weaknesses
The method details (namely the design of the quasi-equivariant layer) are not described in detail and are, thus, unclear.

A large part of the manuscript is used to re-iterate existing group theory and the math behind parameter space, which obfuscates the novelty of the proposed work.

The datasets used are somewhat saturated, and the performance gains are often marginal.

There are no ablation studies to examine the significance of various hyperparameters.

### Questions
1. The experiments are only focusing on invariant tasks. Why don't the authors perform experiments on equivariant tasks (e.g. INR editing)?

2. The performance reported for baselines is often different from the original works (e.g. the works of Zhou et al., Kofinas et al.). In the case of INR classification, specifically, the performance gap is quite large. Why is that?

3. The performance gains of the proposed method compared to the baselines are often marginal. Are there any scenarios/experiments where the proposed method can be expected to boost the baseline greatly? If so, can that be shown through an experiment?

### Soundness
2

### Presentation
2

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
The paper concerns the efficient use of equivariance in the design of metanetworks, which are architectures intended to process the weights of another, predefined model. As different parameter values may identify the same functional mapping (the input-output relation of the model), it is a crucial advantage to identify and process functionally-equivalent sets of parameters.
This work in particular argues that exact equivariance is a too strict assumption to design effective metanetworks, while the weaker notion of quasi-equivariance is sufficient. It then introduces architectures that are able to implement this quasi-equivariance for feed-forward NNs and MHA, and demonstrates convincing results on a number of tasks.

### Strengths
- The topic is of current interest, and the paper does a convincing job in guiding the reader through computational and theoretical motivations for the work.
- The discussion around quasi-equivariance is interesting and convincing, and the notion is a novel contribution that has potentially wide application in the design of meta-networks. 
- The discussion of the limitations (current limited applicability beyond linear architecture) is convincing.

### Weaknesses
**Definition and use of notion of quasi-maximality:**
- Definition 2.2: The sentence "Under generic parameters" is unclear. The entire informal definition seems to serve little purpose, as the formal one is given 10 lines after. I suggest Definition 2.2 to be simply stated in the text, without a "definition" environment.
Moreover, 1) the reason to consider epsilon to be a real algebraic variety is unclear here (instead of another set with other structure), and should be at least motivated if not clearly defined; 2) The notion of maximality seems to depend on $\varepsilon$, but it would be more clear if first it is stated without it (e.g., defining maximality and then $\varepsilon$-maximality).
- Section 2.2: "If G is a maximal symmetry group of the underlying model, such equivariance is sufficient to guarantee that F operates solely on the functional content of θ.". This again is not explicitly demostrated or discussed, and it is especially unclear why considering an algebraic variety is a sensible choice. This is partially motivated by the important example of MHA, which however comes only in Remark 4.4, several pages later.

**Well posedness of the notion of quasi-equivariance:**
- It is mentioned that "Further discussion and insights on this perspective are provided in Appendix A.1". 
Since the conclusions of A.1 are quite involved, a short summary of these insights should be given in the main test. Positive and negative examples would be particularly helpful. 
- Explanations would be appreciated also considering that this may guide practitioners in choosing or not this solution depending on the group of interest. Related to this, G is f-dependent. How does this relate to the results of Section A.1?
- The entire Section A.1 relies on heavy terminology and notions that are not tied to the rest of the paper and are not defined elsewhere. There should be either a concise introduction of the main tools, or at least references to the literature.

**Design of the quasi-equivariant map:**

The construction of the map $\alpha$, especially for MLPs (and specifically the part mapping to the diagonal matrices) is quite involved, and it is completely unclear why such an elaborate construction is required. Appendix B.1 does not help in understanding this. There should be some effort put into explaining why this construction is required in this crucial step.

### Questions
Apart from the points discussed above, there are the following minor points: 
- Section 2.1: $f(\cdot; \theta)$ should be defined with a domain $D$, so that (1) has a precise meaning. Moreover, is $f$ at least continuous?

- Section 2.2: "...to produce incompatible outcomes". What does incompatible mean? So fa we are requiring $F(\theta)=F(\bar\theta)$ if the two parameters are in the same orbit (depend only on the underlying function represented by $\theta$).

- "Moreover, $G$-quasi-equivariance ensures functionality preservation": A one-line equation explaining this key point would be helpful.

- Section 4.1: The notion of continuity of a group-valued function should not be assumed to be common knowledge. Moreover, in Remark 4.1 the set $\Theta$ has connected components. But so far we always had $\Theta=R^{d}$. Is this no more the case? Moreover, for $\Theta=R^d$ this seems to imply that $\alpha$ is a constant function (see below for the matrix permutation group).

- Theorem 4.3: How reasonable are conditions 1/2 in practical, learned MHAs?

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
This work proposes a novel theoretical framework for relaxing the rigidity of strict equivariant meta-networks while still preserving functional identity. The authors formalize this relaxed notion of equivariance as quasi-equivariance and propose a general construction where a learned group action $\alpha(\theta)$ modulates an already equivariant backbone. After describing how they can implement such a network in the case of MLPs, CNNs, and Transformers, they apply it in generalization prediction tasks and the INR image classification task. Across all architectures, the proposed relaxation of the strict equivariance seems to provide small improvements in the overall performance without a significant increase in the number of parameters.

### Strengths
- The framework of quasi-equivariant networks and their connection to the maximal symmetry group and the preservation of the functional identity are presented with significant mathematical rigor. As such, it provides a solid theoretical foundation and a promising starting point for the community to further investigate the proposed framework and its properties.
- The proposed framework and the decomposition of quasi-equivariant networks into $F(\theta)=\alpha(\theta)\beta(\theta)$ is quite general, allowing the approach to be implemented for a variety of tasks and network architectures. The authors partially demonstrate this adaptability by applying it to the most commonly used architectures.
- The experimental evaluation shows that the proposed network provides consistent improvement over the baseline equivariant network without a significant increase in the computational complexity of the method.

### Weaknesses
- The authors claim that introducing the $\alpha(\theta)$ improves the expressivity of the networks, but they do not provide sufficient  evidence to support such a claim. Specifically, if the task or loss we are interested in is invariant to the group transformation introduced by $\alpha$, does the preservation of the functional identity imply that the proposed quasi-equivariant architecture will have the same expressivity as the original one?
- The experimental section lacks ablation studies comparing the proposed approach with other methods that also relax exact equivariance. Thus, it is difficult to conclude whether the improvements in performance come from the specific framework of the quasi-equivariant network or are more generally caused by the relaxation of the equivariant constraint.

### Questions
- If the loss used to train the quasi-equivariant networks is invariant to the transformation introduced by $\alpha(\theta)$, since both the original and transformed parameters map to the same function, does this imply that the loss gradient with respect to $\alpha(\theta)$ is zero? If so, does this phenomenon occur in the current experimental setup?
- How do other approximate or relaxed equivariant methods compare to the proposed quasi-equivariant networks?

### Soundness
3

### Presentation
3

### Contribution
3
