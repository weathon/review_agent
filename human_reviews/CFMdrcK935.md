# Decomposition of one-layer neural networks via the infinite sum of reproducing kernel Banach spaces

- Decision: Reject
- Scores: 6, 8, 5, 6

## Abstract
In this paper, we define the sum of RKBSs using the characterization theorem of RKBSs and show that the sum of RKBSs is compatible with the direct sum of feature spaces. Moreover, we decompose the integral RKBS $\mathcal{F}\_{\sigma}(\mathcal{X},\Omega)$ into the sum of $p$-norm RKBSs $\\{\mathcal{L}\_{\sigma}^{1}(\mu\_{i})\\}\_{i\in I}$. Finally, we provide some applications to enhance the structural understanding of the integral RKBS class.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Reproducing kernel Hilbert spaces (RKHS) can be decomposed into a sum of RKHS. A natural generalization is to consider reproducing kernel _Banach_ spaces (RKBS) and to decompose it as a sum of RKBSs. Defining the sum is non-trivial, and the authors take on this task. Furthermore, given an RKBS with an integral the authors provide a way to decompose it into a sum of RKBSs. The authors claim that this stablishes a connection to neural networks.

### Strengths
1. Banach spaces have a much wider range of options than Hilbert spaces, which seems like a good motivation to consider this type of spaces.
2. The authors hint to novel ideas connecting integral RKBS to the study of neural networks (however, see Weakness 1).
3. The authors are very comprehensive to a reader – such as myself, who has not seen several of the relevant concepts since a real analysis course.

### Weaknesses
1. The authors claim that there is a connection with neural networks, but do not make it clear nor precise. For example, the only mention of neural networks are in the introduction and a single mention in Subsection 3.3, without going into detail of the correspondence between the terms developed in the paper and neural networks.
2. Immediately after Proposition 3.7 the authors mention the feature map ($s$) and the RKBS ($\mathcal{S}$), without an explicit definition in the main text. As the definitions are available in the appendix, I think it would strengthen the paper to include them in the main text. 
3. Presumably there is a correspondence between the triples $(\Psi,\psi,A)$ that the authors see with neural network concepts, but with the current status of the paper is quite hard to understand. Can the authors make this relationship explicit?

### Questions
### Questions:
1. How important is the compactness of $\Omega$ for the main results? By the work of Neal (1996) we know that for _decent_ densities (e.g., finite moments and bounded activation functions) we have a kernel similar to the kernel of $L^2$ stated in line 299, even when $\Omega$ is unbounded. 
2. It is not immediately clear to me why $\mathrm{Im}(A)$ has finite dimension as indicated in line 284. Perhaps I am missing something. Is there a specific reference or lemma that makes it clear?
3. The comment on lines 414-417, about the Tietze extension, seems out of place. Should it be in the proof of Proposition 5.2?

### Citation issues:
- Citations are not consistent throughout, which makes it harder on the reader to digest this technical paper. A good guide is in https://guides.library.unr.edu/apacitation/in-textcite. An example of this is the third line of the first section, which includes the name of the authors twice.
- Proper names are typically capitalized, even in the references. e.g. 'Banach' in Bartolucci et al (2023).

### Minor notation issues:
- The authors use $\langle \cdot ,\cdot \rangle$  and $<\cdot ,\cdot>$ interchangeably for inner products. It would be good to standardize notation, or clarify what is the main difference between these two notations. For example, in page 6, line 291 uses $<\cdot ,\cdot>$ while line 320 uses $\langle \cdot ,\cdot \rangle$ , with no clear difference between them. Perhaps one of the two notations corresponds to a semi-inner product, but at the moment this distinction is not at all clear.
- I could not find a definition of $\mu \perp\nu$ in the text, used in line 161. I assume it means something like $\int_{\Omega} \mu(\omega)d\nu(\omega)=0$, or something in that sense. This could be easily added.

### Minor grammatical mistakes:
- Line 350, 'an another' should just be 'another'
- There is a repetition of "equation" in Remark 3.8.

### References

Francesca Bartolucci, Ernesto De Vito, Lorenzo Rosasco, and Stefano Vigogna. Understanding neural networks with reproducing kernel Banach spaces. Applied and Computational Harmonic Analysis, 62:194–236, 2023

Radford M Neal. Priors for infinite networks. In Bayesian Learning for Neural Networks, Lecture Notes in  Statistics, pp. 29–53. Springer New York, New York, NY, 1996. ISBN 0387947248.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors represent neural networks using an integral RKBS, where the feature space is the measure space corresponding to the distribution of the weight of the final layer. They characterize the decomposition of RKBSs via the decompsition of the feature spaces, and show the integral RKBS representing neural networks is deomposed into the sum of a family of p-norm RKBSs, each of which is characterized by a probability measure.

### Strengths
Applying the theory of RKBSs to analyzing neural networks and reducing the problems to those in the feature spaces of the RKBSs is interesting approach. The result is solid and the mathematical notions are carefully introduced. I think this paper provides a direction of future studies of the theory of neural networks.

### Weaknesses
Although this paper is well-organized and the mathematical notions are clear, for readers in the machine learning community, I think more explanations that shows the connection between the theoretical approaches and results and the pratical neural networks. 
  - In my understanding, the decomposition is by virtue of the decomposition of the measure space (feature space), and that is why RKBSs are useful in the analysis of neural networks. I think the reason why RKBSs are useful should be clearly explained in the main text.
  - The motivation of the decomposition should be explained from the perspective of neural networks. I thought that since the pratical neural networks are represented by the sum involving the weight, instead of the integral involving the distribution of the weight, the decomposition of the integral RKBS into the sum of the family of smaller RKBSs makes the representation more practical. Does this interpretation correct? I think the advavtage of the decomposition should be discussed from the perspective of the application to the analysis of neural networks.
  - Maybe related to the above point, but can we construct the family $\{\mu_i\}$ in Theorem 4.4 explicitly? I think understanding $\{\mu_i\}$ is important for the analysis of the weights of neural networks. Do you have any examples or any comments on this point?

If the motivation related to neural networks becomes clearer, I will consider raising my score.

### Questions
Questions:  
- In definition 3.4, $\sigma$ can be any element in $C(\mathcal{X}\times\Omega)$. Does this mean we can deal with *deep* neural networks by properly setting $\sigma$? In that case, I think this framework is more flexible than other methods like ridgelet transform [1] (in the framework of ridgelet transform, we have to consider the form $\sigma(\langle w,x\rangle-b)$). Can we apply this framework of RKBS to show the universality of deep neural networks?

Minor comments:  
- p5, line 276, "when $\mathcal{X}$ be ..." should be "when $\mathcal{X}$ is ..." ?
- In Remark 3.8, "equation equation 3.2" should be "equation 3.2".

References:  
[1] S. Sonoda and N. Murata, "Neural network with unbounded activation functions is universal approximator", Applied and Computational Harmonic Analysis, 43(2): 233-268.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This work studies the reproducing kernel Banach spaces (RKBSs). Specifically, it shows that the integral RKBSs can be decomposed into a sum of a set of RKBSs each defined based on a different measure. It then presents an application of the decomposition result that the RKHSs are contained in the RKBSs.

### Strengths
- This work gives a thorough presentation of the related concepts to the sum of the RKBSs proposed here.
- This work shows a nice property that the sum of the feature spaces is compatible with the sum of the RKBSs.

### Weaknesses
- The related work section is not informative. In particular, Section 1.1 does not introduce what are the advantages and, importantly to this paper, the limitations of RKHS, and it does not address the previous literature on RKBS nor what questions the literature has solved with RKBS. Also, it does not provide any motivation for the results presented in this work. It is mainly just a list of abbreviated references.
- This work asks the questions to address at the end of page 1 that it aims to decompose the integral RKBS into more fundamental blocks. But it does not touch on the motivation behind and what results one can get with this decomposition.
- Around 5 of the 8 pages are about the definitions or restating results in previous literature. It would be great if this work could spend some space on (1) the potential benefits of their results, (2) takeaway messages about RKBS, (3) technical difficulties encountered and solved, and (4) novel mathematical tools and techniques that are of independent interest. It is otherwise unclear what would be the central contribution of this work.

### Questions
- Please see weaknesses above.
- What is $\mathcal{S}$ in the diagram in Figure 1?
- What are some potential applications of the results presented in this work? What are some specific examples of machine learning tasks or theoretical problems where the RKBS decomposition might provide advantages over RKHS approaches?
- What does $2$ mean in $\sum_{i\in [n]}^2$ in Section 5?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors define the sum of RKBSs using a characterization theorem, investigate its compatibility with the direct sum of feature spaces, and decompose the integral RKBS $ F_\sigma(X, \Omega) $ into the sum of $p$-norm RKBSs $\{L^1_\sigma(\mu_i)\}_{i \in I}$. This study enhances the structural understanding of the integral RKBS class, offering theoretical insights that can help analyze the performance of neural networks by decomposing complex function spaces into simpler, manageable components.

### Strengths
The paper introduces an innovative framework for decomposing integral RKBSs, offering a novel interpretation of one-layer neural networks. This approach is unique in its use of Banach spaces and their decomposition to analyze function spaces, advancing the existing understanding of RKBSs.

The decomposition of RKBSs has significant implications for the analysis of neural networks, especially in designing kernel-based learning algorithms. The compatibility between the sum of RKBSs and the direct sum of feature spaces represents a meaningful advancement in understanding how integral RKBSs can be decomposed, which could potentially impact practical applications in machine learning, such as multiple kernel learning.

### Weaknesses
The paper could benefit from more illustrative examples to make the abstract mathematical concepts more accessible to a broader audience, particularly those in the machine learning community without a strong background in functional analysis.

The experimental results are limited, and the practical implications of the theoretical findings are not fully demonstrated through empirical evaluation. Including numerical examples or simulations to show the decomposition's effects on real-world neural network performance would significantly improve the paper's practical relevance.

The presentation of some key definitions and theorems is rather dense, making it difficult for readers to follow the logical flow. Providing intuitive explanations alongside formal proofs would help bridge the gap for less mathematically inclined readers.

### Questions
1. Could the authors provide a concrete example of how the decomposition of an RKBS improves the understanding or efficiency of neural network analysis? An illustrative example or a simple simulation would greatly help clarify the practical benefits.

2. Would the authors consider adding a numerical evaluation to demonstrate the theoretical claims empirically? This would help bridge the gap between the abstract mathematical results and their practical implications in machine learning.

### Soundness
3

### Presentation
3

### Contribution
3
