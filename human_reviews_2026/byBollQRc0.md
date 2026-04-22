# From Kakeya to Kernels: A Multi-Scale Geometric Framework for Robust Representation Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
This paper addresses the gap between the empirical efficacy of deep learning and the theoretical understanding of its robustness by introducing a novel geometric framework for representation learning, inspired by multi-scale analysis techniques used to resolve the Kakeya set conjecture. The concept of a representation field is formalized, modeling feature activations as geometric entities, and the notion of "stickiness" is defined as the stability of the geometric structure across network layers. The multi-scale Wolff axioms quantify this stability as a formal measure of representation quality. The principal contribution is the Sticky Representation Theorem, which establishes a provable relationship between a network's geometric stickiness and its functional robustness to input perturbations and resilience to missing modalities in multimodal settings. To operationalize this theoretical framework, the Katz-Tao Convex Wolff (KT-CW) Regularizer is derived as an architecture-agnostic loss term that can potentially incentivize the learning of provably robust, sticky representations. This work presents a new, unified approach for analyzing, understanding, and constructing more reliable AI systems within both single- and multi-modal contexts.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper connects definitions used in proving the Kakeya set conjecture in three dimensions with representations of neural network models. 
 
The work presents a theorem which they claim link representational sparseness to Lipschitz continuity. 

The paper also suggests a new regularizer for use in training models.

### Strengths
**S1:** It is theoretically interesting to analyse machine learning concepts using tools from seemingly distinct mathematical areas.

### Weaknesses
**W1:** The paper is too incoherent. There are many concepts which are not properly defined and also assertions which are introduced 
and then never used. I list some of the most serious cases here but see also questions **Q1-Q4,Q7,Q8,Q12** 
- The paper refers to the Kakeya set conjecture multiple times ( e.g. the title and lines 16, 56, 120, 128, 219, 316, 462, 477), 
however there is never any definition or explanation of what the Kakeya set conjecture is.
- In the abstract (line 17) it says "The concept of a representation field is formalized, which models feature activations as 
geometric entities". However, the definition (definition 7) says that a representation field consists of one tube per representation 
of an input. There is no explanation of why the representation of an input should be interpreted as a feature. See also question **Q8**.
- Section 3 considers the application of grains decomposition to a representation field, but there is no definition of what a 
"grains decomposition" is. 
- The paper introduces inductive volume estimates (line 151-159), but then never uses them for anything in the main article. 
- The illustrations of representation fields in figures 1, 2 and 3 show the tubes originating at various points. 
But in the definition 7, all tubes are defined as starting from the origin. 
- A definition of (K)-"stickyness" is not presented before definition 8 on page 6, but the paper mentions stickyness many times before that 
e.g. lines 74-80, 161-163, 292-293, figure 2. This is confusing. Especially the "sticky" vs "non-sticky" figure before there has been a 
definition of what "sticky" means. 


**W2:** The paper introduces the "KT-CW regularizer" line 89 and proposition 1. However, there is no implementation of this or even a 
description of how one might implement this. See question **Q11**.

### Questions
Because of the lacking definitions and explanations I recommend rejection.   


**Questions:**

**Q1:** Why is the title "FROM KAKEYA TO KERNELS" when kernels are not mentioned in the rest of the paper?  

**Q2:** Why do you introduce inductive volume estimates (line 151-159), when they are not used in the rest of the main article? 

**Q3:** In line 191-192, the paper defines $\mathcal{T}$ as a set of deformation operators $\tau:\mathcal{X} \to \mathcal{X}$, 
although it was defined as a collection of tubes ($\delta$-neighbourhoods of line segments) in line 131. When the tubes are used 
for defining a representation field, they become subsets of the representation space. In other words, these are two very different 
things and should therefore not share the same notation.

**Q4:** Equation 3: What do you mean by $\Vert \tau \Vert$. Do you mean $\Vert x - \tau(x) \Vert$? 

**Q5:** Definition 7: The representation field depends on a specific dataset. This should be more clear in the notation. 

**Q6:** Axiom 1, 2: The paper says the constant $C_{KT-CW}$ of the representation field measures the degree of "feature clustering".
	But why do you need to introduce tubes for this? Could you not just as well define $\delta$ neighbourhoods around all representations 
	and then say something about the number of representation neighbourhoods which can fit in a convex set? 

**Q7:** Remark 1, Line 254-264: 
- Here the paper mentions the "IB principle", "IB framework" and "IB objective". What does the abbreviation IB mean? 
- It says: "This geometric sparsity is a tangible signature of an information-theoretically simple, or compressed, representation." 
	But isn't sparse the opposite of compressed? 
- It also says: "By grounding the notion of complexity in a stable geometric measure". But your definition of representation field 
	depends on the dataset. You could find that representations are very sparse for one dataset, but completely collapsed on another.
	In what way do you mean it is stable? 


**Q8:** Lemma 1: "Applying the grains decomposition algorithm to a representation field $\mathcal{T}_l$ is equivalent to a data-dependent, 
unsupervised clustering of the feature tubes". However, the representation field is defined as tubes connecting the origin with the 
representations of specific inputs. Why should we interpret every representation of a an input as a feature? 



**Q9:** Definition 8: Why is there no dependence here on $\delta$? In line 420-421, it says: 
"preservation of structural integrity across scales (“stickiness”)", and the scales were $\delta$ (line 130).


**Q10:** Theorem 1 (The Sticky Representation Theorem): How can you say something about Lipschitz continuity in general for a function
using only the representations from a finite dataset? 


**Q11:** Proposition 1 (KT-CW Regularizer): How would you implement finding the infimum needed for $C_{KT-CW}(\mathcal{T}_l)$?


**Q12** In Appendix B "Multi-scale analysis" is paired with "Hierarchical feature extraction across layers", but in Definition 1, it says:
"A scale is a small positive number $\delta$" and "The analysis is fundamentally multi-scale, relating the properties of objects at a fine 
scale $\delta$ to those at a coarser scale". How is this related to considering the representations of models across layers?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a new geometric framework for robust representation learning. The work applies methods from multi-scale analysis techniques to quantify how susceptible models are to minor input perturbations. The model defines learned representations as a collection of tubes in the representation space, and relates various aspects of model performance to the convex sets and subspaces that contain these tubes. It is then shown how a geometric property of these representations called K-stickiness is tied to robustness and performance in multimodal DNN models. Insights from this model are then used to design a regularization term that can help prevent feature collapse and misalignment in multimodal learning

### Strengths
* This paper is well written and relatively accessible to a reader without a strong background in harmonic analysis.
* Connections to several machine learning phenomena lend credence to the veracity of this framework. For example, the framework offers explanations for mode collapse in GANS, the lottery ticket hypothesis, and adversarial robustness, among other things. 
* The figures in this paper are particularly high quality and help give a visual intuition for the geometric objects being discussed.

### Weaknesses
* This framework is proposed as a more direct method for quantifying mutual information for the information bottleneck principle. However, it's not shown that it's possible to compute the constants described in this paper. While this is an interesting model, I would like more justification as to why this is more useful in practice than existing models (like IB).
* No attempt is made to empirically evaluate the validity of the proposed geometric model. Again, I would be interested in seeing empirical validation that this model is a useful proxy for IB.
* Similarly, it seems to me that the KT-CW regularizer would not be feasible to implement. It's noted in the limitation in the appendix that this loss term might be difficult to compute, but I think that should be made more clear in the body that the implementation of this loss is left open for future research.
* I think it would be helpful to included proof sketches in the main body of the paper. As written, the reader will have little intuition as to the reasoning behind the theorems without reading the full proofs in the appendix.
* The term "performance degradation" as it is used in Theorem 2 should be more explicitly defined. It's not clear how you are expecting the model to be used.

### Questions
* I'm confused on the order of quantifiers in Axiom 4. Is $W$ meant to be fixed before the computation of $C$, or is the infimum taken over the ratio computed for all convex sets $W$?
* Do the choices for the $\delta$ and $\rho$ parameters affect the values of the constants that are computed? Should their values depend on the data distribution?
* How should I think about the significance of a shared Lipschitz constant in the multimodal setting? As defined in section 2, the Lipschitz constant is a function of each deformation's magnitude. Between modalities, transformations would take different forms, so a Lipschitz constant that is considered small in one modality may be considered large in another. Is the significance of Theorem 2 (1) just that all modalities exhibit Lipschitz stability for some constant which may or may not be small, or is the guarantee stronger than that?
* How do representation similarity metrics such as CCA [1] or CKA [2] relate to this framework? Are the patterns in representations that are being captured here explainable in terms of representation similarity analysis?

[1] Morcos, Ari, Maithra Raghu, and Samy Bengio. "Insights on representational similarity in neural networks with canonical correlation." Advances in neural information processing systems 31 (2018).

[2] Kornblith, Simon, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. "Similarity of neural network representations revisited." In International conference on machine learning, pp. 3519-3529. PMlR, 2019.

### Soundness
2

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
3

### Summary
This paper introduces a novel geometric quantity called “stickiness”, inspired by the Kakeya set problem, to characterize aspects of neural network representations. The authors define stickiness as a measure of how much a network’s output trajectories “cling” to certain directions in input space and establish a theoretical link between stickiness and the Lipschitz continuity of the network. Conceptually, this connection bridges ideas from geometric measure theory with representation analysis in deep learning.

The paper is clearly written and well-motivated. The presentation is easy to follow, and the framing—connecting Kakeya geometry to neural representations—is both original and intellectually stimulating.

The paper presents an original and intellectually appealing idea—introducing stickiness as a geometric quantity inspired by the Kakeya problem and linking it to Lipschitz continuity in neural networks. The presentation is clear and well organized, making the theory accessible to a broad audience. However, the work currently feels incomplete: the motivation for why Lipschitz-ness is desirable in this setting is not clearly articulated, there is limited discussion of related theoretical frameworks, and the paper lacks empirical or computational illustrations that would demonstrate how stickiness behaves in practice.

With additional context connecting the theory to existing notions of smoothness or robustness, and with a simple empirical demonstration to anchor the concept, this work could become a strong theoretical contribution in a future version. As it stands, I find the idea promising but not yet sufficiently substantiated for acceptance.

### Strengths
* Novel theoretical concept: The introduction of stickiness as a representation property derived from Kakeya-type arguments is creative and mathematically original.
* Clear presentation: Definitions and theoretical statements are well structured and easy to follow, even without deep familiarity with geometric measure theory.
* Potentially broad relevance: The proposed framework could provide a new lens for analyzing neural smoothness, stability, or robustness through geometric quantities.
* Interesting bridge between mathematics and deep learning: The Kakeya-inspired formulation connects a deep geometric concept to neural network analysis in a way that feels fresh and principled.

### Weaknesses
* Unclear motivation for Lipschitz-ness:
The paper emphasizes the link between stickiness and the Lipschitz constant but does not sufficiently explain why Lipschitz-ness is a desirable property in this context. Is it intended as a proxy for robustness, stability, or generalization? A short discussion of its conceptual or practical significance would strengthen the narrative.
* Lack of connection to prior work:
It is unclear whether there are existing approaches relating network representations to Lipschitz continuity (e.g., via Jacobian norms, spectral bounds, or curvature regularization). Providing a short literature comparison would help contextualize the contribution.
* Missing empirical or computational illustration:
While the paper is primarily theoretical, it would be valuable to include at least a small toy example (e.g., a simple MLP or 2D dataset) demonstrating how stickiness can be estimated and how it scales with network smoothness or architecture.
* Complexity considerations:
A discussion of the computational cost of estimating stickiness—either analytically or numerically—would improve the paper’s practical relevance.

### Questions
1. What makes Lipschitz continuity a particularly meaningful or desirable property in the context of representation learning?
2. Are there existing theoretical frameworks or empirical studies connecting representation geometry to Lipschitz bounds (e.g., through Jacobian regularization or spectral control)?
3. How computationally intensive is it to estimate stickiness for modern architectures?
4. Could you include a small illustrative example showing how stickiness behaves for networks of varying smoothness or depth?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a new theoretical framework to analyze and explain the generalization properties of modern deep neural networks. A multi-scale geometric construction is introduced, together with some theoretical analysis, to understand the structure of the hidden representations and their robustness to input perturbations and missing modalities in multimodal settings. A regularizer is then proposed, motivated by this analysis, to control the structure of the hidden representations.

### Strengths
- The proposed framework is interesting as it introduces a new construction to understand latent representations. The intuition of covering the representation space with the proposed tubes is meaningful.

- The theoretical results appear to be correct, although I have not followed the proofs in close detail.

- The framework has the potential to inspire new research questions and regularization techniques accordingly.

### Weaknesses
- I think that the writing and presentation of the idea do not align well with the expectations of the conference. The clarity of the paper is overall good, but the exposition is quite "texty" and lacks the level of technical detail typically expected in machine learning conferences. In other words, while it seems that there is nothing theoretically incorrect, it is unclear how the proposed approach can actually be implemented in practice.

- In a similar spirit, the paper does not provide any empirical evidence supporting the theoretical claims. The approach is intuitively plausible, but a common belief partially proven in deep learning is that neural networks succeed because they learn sparse representations. I think this seems conceptually different from what the authors propose. Therefore, I would expect some empirical justification to support why the proposed theory may be the correct approach to explain generalization.

- I think some parts of the text can be replaced by technical details on how the method could be implemented, together with empirical results. Moreover, I suggest providing a two- or three-dimensional illustrative example to clarify the desired geometric properties of the representations. The current figures are conceptual but somewhat difficult to interpret.

### Questions
Q1. As far as I understand, the idea can be summarized as follows:

- The hidden representations should not concentrate within a convex region (for example, a ball) in the representation space.

- The hidden representations should ideally point in different directions so as to cover the representation space as uniformly as possible, rather than aligning near a common subspace.

Q2. Does the proposed approach imply that the theoretical perspectives in deep learning related to sparse representations may be misleading or incomplete? I would expect some discussion on this point.

Q3. Is it possible to analyze the proposed framework in the context of commonly used optimizers and their generalization behavior, such as SGD? In other words, is it reasonable to ask whether SGD naturally encourages the type of structure in representations that your framework suggests?

Q4. Could you provide some info on whether and how the proposed framework can be implemented in practice?

### Soundness
2

### Presentation
2

### Contribution
2
