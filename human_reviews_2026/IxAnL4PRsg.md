# Operator Learning with Domain Decomposition for Geometry Generalization in PDE Solving

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 4

## Abstract
Neural operators have become increasingly popular in solving partial differential equations (PDEs) due to their superior capability to capture intricate mappings between function spaces over complex domains. However, the data-hungry nature of operator learning inevitably poses a bottleneck for their widespread applications. At the core of the challenge lies the absence of transferability of neural operators to new geometries. To tackle this issue, we propose operator learning with domain decomposition, a local-to-global framework to solve PDEs on arbitrary geometries. Under this framework, we devise an iterative scheme Schwarz Neural Inference (SNI). This scheme allows for partitioning of the problem domain into smaller subdomains, on which local problems can be solved with neural operators, and stitching local solutions to construct a global solution. Additionally, we provide a theoretical analysis of the convergence rate and error bound. We conduct extensive experiments on several representative linear and nonlinear PDEs with diverse boundary conditions and achieve remarkable geometry generalization compared to alternative methods.These analysis and experiments demonstrate the proposed framework's potential in addressing challenges related to geometry generalization and data efficiency. The code is publicly available at this repository: https://github.com/questionstorer/sni.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a new framework for training neural operators to generalize to unseen geometries at inference with domain decomposition. Instead of generating a variety of complex geometries and train a neural operator such that it can generalize to an unseen complex geometry at test time, the authors instead propose to train a neural operator on a diverse number of simple geometries that can be seen as a decomposition of a complex geometry. At inference, they introduce the SNI algorithm, that decompose a complex unseen geometry into a set of simpler shapes, that can be solved iteratively.

### Strengths
I really like the tackled problem and find it very interesting and well motivated. Data driven approaches are known to be robust to the geometries observed in the training distribution, but they naturally tend to fail when tested on geometries that shift from the training distribution. Decomposing a complex geometry into a set of very simple shapes (that thus are more aligned with the training distribution) is an interesting approach to generalize any neural operator to arbitrary geometry.

Experiments successfully demonstrate that the framework proposed help to boost performance of neural operators on unseen geometries via geometry decomposition. The framework is general and ca be applied to any neural operator that accepts arbitrary mesh points.

Ablations are provided to see how sensible is the method with respect to the number of sub-domains K, data augmentation, etc.

### Weaknesses
While the motivation is clear, one of the potential benefit of neural operators is to overcome the computational cost of traditional methods. Decomposing the domain into sub-domains naturally increase the computational cost of data-driven approaches. Ablations provided in appendix seem to show that the computational time needed is high for very accurate predictions. It would be nice to compare this with the time needed for solving the PDE with a numerical method, and put it as limitations based on the results.

The experiments have been tested on relatively simple and a limited number of geometries. It would be nice to test the method on more challenging geometries and see how well it performs with respect to the number of sub-domains. If the geometry is much more complex, we can imagine that it would need much more sub-domains to recover the simple geometries seen during training. 

I think the paper would clearly benefit from a related work that mentions existing methods for geometry generalization. The method should be compared to existing works when possible.

### Questions
Could you provide runtime comparisons? It would be helpful to provide actual runtime comparisons: (a) pure neural operator direct inference, (b) SNI (with partition & iteration), (c) classical PDE solver.

The method has been tested under specific boundary conditions and in a specific setting. Could you elaborate a bit on how general this method for PDE modeling? How could it be applied to more challenging physical systems?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduced an operator learning strategy based on domain decomposition to handle complex geometries. The paper proposes to use data augmentation to improve the learning of the PDE on simple geometries. The solving method is based on domain-decomposition and locally uses the neural operator. Finally the method reconstruct dynamics on complex geometries by merging sub-solutions, using a SNI algorithm. The paper proposes a theoretical study of the algorithm as well as some experiments.

### Strengths
- The proposed paper is well-writen and carrefully introduces a lot of concepts.
- A theoretical analysis is proposed as well as experimental evaluations.

### Weaknesses
- Scope of applicability: My concerns about applicability relies in the fact that the problem formulation states that PDE are tested in parabolic equations (with time). Is it a limitation of the method? Why is the method limited to such PDEs? 
- Missing references: a lot a references dealing with complex geometries and generalizations are missing, see eg [1-5]
- No experiments on public datasets: I think evaluating the proposed method on well-known dataset helps the reader positioning the method with respect to existing baselines. As an example, some dataset considering irregular/complex geometries are in [6-7].
- While the paper is very rich, several key components of the method are detailed in appendices, making the reading as is hard to understand (eg algorithm). 
- Inference time and scaling : I have some concerns regarding these 2 aspects of the method, see questions.

### References
[1] Operator Learning with Neural Fields: Tackling PDEs on General Geometries, Louis Serrano, Lise Le Boudec, Armand Kassaï Koupaï, Thomas X Wang, Yuan Yin, Jean-Noël Vittaut, Patrick Gallinari, 2023

[2] Transolver: A Fast Transformer Solver for PDEs on General Geometries, Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang, Mingsheng Long, 2024

[3] A.D.Jagtap, G.E.Karniadakis, Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations, Commun. Comput. Phys., Vol.28, No.5, 2002-2041, 2020. 

[4] K. Shukla, A.D. Jagtap, G.E. Karniadakis, Parallel Physics-Informed Neural Networks via Domain Decomposition, Journal of Computational Physics 447, 110683, (2021).

[5] AROMA: Preserving Spatial Structure for Latent PDE Modeling with Local Neural Fields, Louis Serrano, Thomas X Wang, Etienne Le Naour, Jean-Noël Vittaut, Patrick Gallinari, 2024

[6] Learning Mesh-Based Simulation with Graph Networks, Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, Peter W. Battaglia, 2020

[7] A comprehensive and fair comparison of two neural operators (with practical extensions) based on FAIR data, Lu Lu, Xuhui Meng, Shengze Cai, Zhiping Mao, Somdatta Goswami, Zhongqiang Zhang, George Em Karniadakis, 2021

### Questions
### Questions
-	Could you compare the proposed method to more recent baselines ? consider public/standard datasets of the litterature? Do you plan on releasing the dataset for future work? 
-	Line 093-095, you mentionned that the method is restricted to parabolic equations. Could you detail about this ? What are limitations in terms of PDE equations/BCs (also reference to lines 198-199)? 
-	Could you provide some insight on how to chose the hyper parameters mentioned in sections 3.1 (ie n?, K, d?) It is specified that hyper parameters have to be carefully chosen, why ? Doesn’t the method converge in these cases ? It does not seems to be very sensitive to hp in terms of final performance in the ablation on figure 4.
-	Could you detail about the results on Heat2d – B ? While the method seems coherent and robust in all scenarios, on this geometry however, results drops x2. 
-	How does the method behaves on simple geometries ? 
-  could you elaborate on the condition required for the algorithm to converge? Line 160 refers to Appendix A, but Appendix A finally states that convergence depends on the PDE and decomposition. 
- Have you studied where are located the errors? Is there a higher (or lower) error around the boundary of the local shapes? 
- How does the method scales in term of memory consumption with respect to the number of points ? Would it be tractable for 3D PDEs ? Industrial complex geometries ? 
-	Have you compared inference time with respect to baselines ? I saw the method’s inference time in the appendices, but I could not find baselines’. 
- Regarding the theoretical section: I think I am not qualified to review such theoretical results, but I have question regarding the applicability of the proof: When are the assumptions met? For example, could you cite some PDE respecting the hypothesis on L? 

### Minor comments :
-	Legend of figure3 : it it hard to understand what validation refers to, I think it is detailed in the last sentence of description? just adding a (validation) at the end would remove any questioning. 
-	Figure 4 : I think that log scales for errors would be more appropriate for these figures. As is, it is hard to visualize the differences between all ablation (at least the 2 firsts).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses generalization to novel geometries in neural operator-based PDE solving. The approach builds on domain decomposition theory, specifically proposing a neural version of Schwarz inference with theoretical guarantees. The authors train models (GNOT) on simple geometries, then apply them at inference to complex domains that can be decomposed into subdomains. The iterative method that the authors propose consistently outperforms the vanilla direct inference.

### Strengths
- As seen in Table 1, the method obtains excellent results on elliptic problems and on the temporal Heat equation.
- Figure 4 is particularly enlightening and shows that the error consistently decreases as the number of iterations grows, which highlights the stability of the method.
- The choice of shapes at test time is interesting and appears non trivial, even if they are in 2d.

### Weaknesses
- The authors don't provide in the main paper illustrations for the simpler domains that they choose during training.
- The method assumes that a partitioning is already available at test time, which suggests that the method relies on some preprocessing. This needs to come with further explanations and would help a reader less familiar with domain decomposition method how it works in practice.
- Figure 4 shows that in some cases the method requires up to 5000 iterations to converge. How much inference time overhead does this represent compared to the direct inference.
- The role of the data augmentation is not very clear: Table 2 shows that this has a little effect on the generalization performance and I would therefore insist less on this aspect than on the analysis of the core of the method.

### Questions
- How do you embed the boundary conditions in the GNOT architecture? Does it differ from the way you do it with GeoFNO? Could this explain the performance gap? Is your method really model agnostic or do you assume specific capabilities (e.g. boundary conditions encoding) in the neural operator?
- Could you provide visualisations for some training domain examples? 
- Could this method be extended to 3d domains?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes operator learning with domain decomposition, a novel framework designed to improve the geometry generalization and data efficiency of neural operators for solving PDEs. By integrating domain decomposition methods with neural operator learning, the authors train local neural operators on randomly generated basic shapes and then use an iterative inference algorithm called Schwarz Neural Inference to stitch local solutions into a global one for arbitrary geometries. The paper provides theoretical guarantees on convergence and error bounds and demonstrates the method’s strong performance across a variety of linear and nonlinear PDEs.

### Strengths
- Develop a local-to-global framework with strong theoretical grounding that seamlessly integrates domain decomposition and neural operator learning, offering flexibility to support multiple neural operator architectures for solving PDEs on complex geometries.

- Demonstrates robust geometry generalization, effectively overcoming one of the major limitations of existing neural operator methods.

- Presents comprehensive experiments encompassing both linear and nonlinear, as well as steady and transient, PDEs.

### Weaknesses
- The paper does not justify that domains A, B, and C adequately represent the full diversity of 2D geometries. Moreover, all experiments are limited to 2D cases, leaving the scalability and applicability to 3D geometries unverified. At least one experiment on a 3D domain would have strengthened the validation of the proposed framework.

- Since the computational cost heavily depends on the number of iterations required by the iterative inference process, efficiency can become a concern. Moreover, the number of subdomains Kmay depend on the number of vertices and edges, especially when field variations within the domain are significant. However, the authors treat Kas independent of the number of vertices and edges when estimating the time complexity, which can be misleading. This assumption suggests a linear relationship with the number of vertices, while in practice, the computational complexity may scale multiplicatively with both the number of vertices and the number of iterations.

- Although the proposed framework represents the computational domain as a graph, the paper does not include any comparison or discussion with graph-based neural networks or their variants (e.g., [1, 2, 3]), which could provide relevant baselines or complementary perspectives.

[1] Neural Operator: Graph Kernel Network for Partial Differential Equation, 2020.

[2] Learning Mesh-Based Simulation with Graph Networks, ICLR, 2021.

[3] HAMLET: Graph Transformer Neural Operator for Partial Differential Equations, ICML, 2024.

### Questions
See the weakness

### Soundness
3

### Presentation
2

### Contribution
2
