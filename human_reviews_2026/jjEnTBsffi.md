# Guaranteed Simply Connected Mesh Reconstruction from an Unorganized Point Cloud

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
We introduce an approach that reconstructs a closed surface mesh from a noisy point cloud, where the topology of surface is guaranteed to be simply connected, i.e., homeomorphic to a topological 2-sphere. This task enjoys a wide range of applications, e.g., 3D organ and vessel reconstruction from CT scans. Central to our approach is a robust module that takes a collection of oriented triangles in a 3D triangulation as input and outputs a simply connected volumetric mesh whose boundary approximates the input triangles. Starting from a 3D Delaunay triangulation of the input point cloud and initial triangle orientations obtained through a spectral approach, our approach alternates between applying the module to obtain a reconstruction and using that reconstruction to reorient the input triangles.  Experimental results on real and synthetic datasets demonstrate the effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents an approach to reconstruct a closed and connected mesh from noise point cloud. The results guarantee to be simply connected and homeomorphic to a sphere. The method is validated using a few models.

### Strengths
The method is interesting with deep math foundation.

### Weaknesses
The topology is limited to genus 0. It cannot deal with models of higher genus.

The convergence condition of the iterations are not clear.

The time performance is not reported.

Though the paper claims to reconstruct noise point cloud. The noise-level is not clear.

The data set for evaluation is few. More data should be evaluated.

### Questions
What are the efficiency performance of the method? The number of points for evaluation. The running time and memory consumption.

(Feng 2023) evaluates winding numbers on surface instead of 3D volume space. Since u(p) is the number of times that the surface  S encloses p, why isn't u(p) the space winding number (Jacobson et al. (2013))?

Can you give more explanations on HDD?

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
2

### Summary
This paper suggests a method to reconstruct 3D mesh surfaces from point clouds. It is claimed that the method is guaranteed to reproduce simple connected surfaces. Empirical experiments show that the method compares favorably with other methods

### Strengths
1. The method gives good results on the surface reconstruction task
2. The method seems novel, though I am not sufficient familiar with computer vision/graphics literature to make an informed judgement.

### Weaknesses
1. Wrong venue: This paper does not really include any learning aspects. And it assumes knowledge of geometry processing (e.g. vector forms, Helmholz Hodge Decomposition) which is not common in ICLR. It seems to me the paper would get a more meaningful review, and a move understanding audience, in a venue like Siggraph or CVPR

2. I'm a bit concerned re mathematical correctness: 
(a)  The paper claims a guaranteed simply connected mesh reconstruction, but there is no formal analysis of the algorithm (of the form of Theorem-Proof).

(b)  The paper contains many advanced math statements without a source e.g., in Line 152 
"S is simple connected when the harmonic component of the 1 form du vanishes" but I think this is true of most of section 4. 

(c) I have some specific questions on the math, detailed below

### Questions
1. Line 152: you say u is piecewise constant, and then talk about the 1-from du. But a piecewise constant function is not smooth right? What is the meaning of du?

2. In Line 87 you say existing methods do not provide control on surface topology, but there are some such methods [1,2,3], it would make sense to discuss them somewhere and explain how these methods relate, or what is the advantage of your method.

3. The discussion of the results is a bit misleading: in line 416 you say "our approach consistently outperforms all baselines" and the ours method is highlighted in gray, which could be interperted as winning a task. However, the "Ours method" does not win in all tasks in table 1 or in table 2. It would be good to highlight the winner in each column in bold, and to add some discussion and to replace "our approach consistently outperforms all baselines" with our approach outperfroms baselines in X out of Y instances. 

Minor point which you do not need to address in rebuttal. There are some spelling mistakes you should address:

* Line 432 you say "too" but I think you mean "two"

* Line 478  pepper is missing a 'p'




[1] Topology-Aware Surface Reconstruction for Point Clouds, Gabrielsson et al. Computer Graphics forum 2020
[2] Robust Optimization for Topological Surface Reconstruction. Lazar et al. TOG 2018
[3] Topology-controlled Re-
construction of Multi-labelled Domains from Cross-sections. Huang et al. TOG 2017

### Soundness
3

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
3

### Summary
This paper proposes an approach that guarantees a closed, simply connected mesh reconstruction from a noisy point cloud. It is inspired by the winding number field. The method takes a collection of oriented triangles of a 3D triangulation as input and outputs a closed, simply connected mesh. Besides, it obtains oriented triangles from a tetrahedral mesh and removes input outliers via alternating optimization. Experimental results on real and synthetic benchmark datasets demonstrate the effectiveness of the
approach.

### Strengths
This paper is well written with good presentation.

The implementation details have been elaborated clearly. It might be easy to reproduce the results reported in the paper.

The formulations are defined and explained clearly.

The improvements over prior methods are promising and remarkable.

### Weaknesses
It is important to clarify the new contributions more clearly, as the prooposed method is built on several existing methods like (Jacobson et 2023, Feng et al 2023). As claimed in the paper, the novelty is generalizing the existing framework from 2D curves to 3D surfaces. 

Apart from textual descriptions and formal equations, it is common to provide a figure which can introduce the overal pipeline in this method. 

Some presentations need more refinement, such as the captions in Figure 3 and 4 should not be on top.

### Questions
The experiments lack a comparison on the computation cost.

In terms of ablation study, it is important to provide more quantitative results apart from the comparison in Figure 5.

According to the quantitative comparisons in Table 1, why other competitors perform much poorly than the proposed method. Whether the implementation setups are fair properly?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper presents a method to reconstruct closed, simply connected 3D meshes from noisy point clouds by first computing a 3D Delaunay triangulation and assigning triangle orientations via a spectral method, then iteratively applying a winding-number–based module to correct topology and refine the mesh. Experiments on medical and synthetic datasets show it produces topologically correct reconstructions with high geometric fidelity, outperforming existing methods

### Strengths
- The proposed method guarantees that the final 3D mesh is free of holes (topologically equivalent to a 2-sphere i.e., simply connected).
- It is a non–deep-learning approach that runs very efficiently and shows strong robustness to noisy point clouds, outperforming previous methods.
- The paper is exceptionally well presented, with clear overview sections that make the core ideas accessible even to readers outside this research area.

### Weaknesses
- The ablation study lacks quantitative results. The authors only include a qualitative 2D example (Figure 5) to demonstrate the importance of each component. It would strengthen the paper to report numerical metrics on full 3D models, similar to Tables 1 and 2.
- The evaluation metrics are described only briefly, authors can further elaborate why these metrics are suitable. In addition, highlighting the best metric values in bold would make the tables clearer and easier to interpret. It would help the reader if the best results in the tables were highlighted (e.g. in bold).
- It would be beneficial to include runtime comparison of the methods in Tables 1 & 2.

### Questions
- Is it guaranteed that this procedure will always converge? 
- In line 179: Is it assumed that $f \in \mathcal{F}_0$ ? I haven't seen this to be explicitly stated, but I was wondering if it is not true, then (in line 183) shouldn't be $\Gamma_f \in \{ -1, 0, 1 \}$ ?
- Is there any reason for not including results on thick structures (as defined in the CrossSDF paper)?

### Soundness
3

### Presentation
4

### Contribution
4
