# FUSED PARTIAL GROMOV-WASSERSTEIN FOR STRUCTURED OBJECTS

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Structured data, such as graphs, is vital in machine learning due to its capacity to capture complex relationships and interactions. In recent years, the Fused Gromov-Wasserstein (FGW) distance has attracted growing interest because it enables the comparison of structured data by jointly accounting for feature similarity and geometric structure. However, as a variant of optimal transport (OT), classical FGW assumes an equal mass constraint on the compared data. In this work, we relax this mass constraint and propose the Fused Partial Gromov-Wasserstein (FPGW) framework, which extends FGW to accommodate unbalanced data. Theoretically, we establish the relationship between FPGW and FGW and prove the metric properties of FPGW. Numerically, we introduce Frank-Wolfe solvers and Sinkhorn solvers for the proposed FPGW framework.  Finally, we evaluate the FPGW distance through graph matching, graph classification and graph clustering experiments, demonstrating its robust performance. The code for reproducing all the numerical results is available in the anonymous repository at \url{https://anonymous.4open.science/r/fused-pgw-041B}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the Fused Partial Gromov-Wasserstein (FPGW) formulation, which combines Partial Optimal Transport (POT) and Partial Gromov-Wasserstein (PGW). The FPGW model allows the comparison of structured objects with unequal total mass. To solve the FPGW problem efficiently, the authors develop optimization methods based on the Frank–Wolfe and Sinkhorn algorithms. Experimental results on graph matching, clustering, and classification tasks demonstrate that FPGW achieves more robust and consistent performance over existing baselines.

### Strengths
- The proposed formulation is inspired by the Fused Gromov–Wasserstein model and effectively incorporates both node features and structural information for graph comparison.
- By introducing partial matching into the fused formulation, the method can handle structured objects with unequal mass, which is meaningful.
- The paper is generally well written and well organized.
- Experimental results show the robust and consistent performance of the proposed approach over baselines across multiple tasks.

### Weaknesses
- The fusion of a linear term (from OT or partial OT) and a quadratic term (from GW or partial GW) has already been explored in prior literature. The proposed FPGW appears to build mainly upon the Partial Gromov–Wasserstein (PGW) model [1], and the theoretical results (e.g., Theorem 3.1) seem to be direct corollaries of those in PGW.
- From the fusion perspective, FPGQ of this paper and PGW of [1] replaces the KL divergence used in fused unbalanced GW [2] with total variation for the regularization function D_{\Phi_1}. Please elaborate more on the advantage of using TV over KL.
- The proposed FW and Sinkhorn algorithms for FPGW appear to be straightforward adaptations of those from [1]. In addition, in the graph matching experiments, only the Sinkhorn-based FPGW (first mentioned in line 379, presumably Algorithm 2) was used for comparisons.
Please clarify why the FW variant was not evaluated.
- In the toy graph clustering experiment, the setting is not convincing. The weight \omega_2 was set as 0.999, which means that the contribution of linear term (OT cost) was negligible. Then it becomes PGW, which is not original contribution of this paper. Hence, this example fails to demonstrate the distinct effectiveness of the fused formulation.
- In other experiments (e.g., graph matching), the selection of \omega parameters was not explained. 

Below are some minor things:
- In Eq (12), the constraint should be \gamma \in T_{\leq} (p,q). 
- In Eq (16), the left hand side should be F(\gamma, \pi). 
- In Proposition 3.2 (line 252), the constraint should read \gamma \in \in T_{\leq} (\muy,\nuy). 
- In line 1001, where is the proposition L.2. 
- In line 1375, should be |\gamma| = \rho. 
- In line 1679, should use \leq instead of "=" 
- In line 1858, left-hand side should be F(\gamma, \pi). 
- In line 1892, use \gamma \times \pi instead of \gamma \bigotimes \pi 
- In line 1990, the objective is missing--only the defintion of the constraint set is given. 
- In line 1982, a ">" is missing for the first term.







[1] Bai et al,. Partial Gromov-Wasserstein Metric, ICLR 2025.

[2] Thual et al, Aligning individual brains with Fused Unbalanced Gromov-Wasserstein, NeuRIPS 2022.

### Questions
see the weaknesses.

### Soundness
3

### Presentation
3

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
In this paper the authors propose a partial OT version of the existing Fused Gromov
Wasserstein (FGW) distance, called Fused Partial Gromov Wasserstein. The FGW
distance is a metric to compare structured objects (e.g. graphs) with node
features. The proposed distance allows to compare only a fraction of the total
mass of the two distributions, which is useful when the two objects to compare
have outliers, different proportions or only part of objects to be aligned. The
authors propose two variants of the problem, one with a hard constraint on the
amount of mass to be matched, and one with a Total Variation penalty. The they
discuss the theoretical properties of the proposed distance, and propose a Frank
Wolfe and entropic/Sinkhorn optimization scheme to compute it. Finally, they
demonstrate the interest of the proposed distance on several experiments (graph
clustering, partial graph matching, geometry matching) on both synthetic and
real data.

### Strengths
+ The paper is overall well written and the proposed method is clearly
  explained. 

+ In applications, one might have access to structured objects with feature on the
  nodes that are only partially comparable. The proposed distance is a useful
  extension of the existing FGW distance to handle such cases.

+ The experiments suggests that the proposed distance indeed work better than FGW
  when only part of the objects are comparable.

### Weaknesses
+ The contribution novelty is extremely limited. The proposed distance is a
  straightforward extension of the existing FGW distance (Vayer 2019) to the
  partial case, following the same principles as the Partial GW with only a
  linear term added from (Chapel 2020). While this is obviously useful in
  practice, the adaptation is easy to implement and probably already used in the
  community (see next point).
  
+ The optimization schemes are also straightforward  adaptations of existing
  schemes for Partial GW (the Alg. 1 Frank Wolfe) and Unbalanced GW of (Sejourne
  2021) for the Alg. 2 sinkhorn/entropic version. As a matter of fact the
  Partial FGW (Frank Wolfe) is already implemented in the  POT library  at the
  following URL:

  https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.partial_fused_gromov_wasserstein

  and the entropic version is available here:

  https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.entropic_partial_fused_gromov_wasserstein

  While those two implementations above might not solve exactly the formulation
  (but something equivalent), it shows that the proposed method and/or
  formulation cannot really be claimed as an original contribution.

  See the example of the POT library for use of the Partial FGW solver:
  https://pythonot.github.io/auto_examples/gromov/plot_partial_fgw.html


+ All theoretical results are interesting and have probably not specifically
  been investigated in the literature but also seem like simple adaptations of
  existing results, metric properties. If really new proving strategies have
  been devised  the authors should have highlighted them 

+ Numerical experiments show that FPGW works better than FGW when only part of
  the objects are comparable, which is expected. However they have been
  specifically designed to make FGW fail (e.g. partial alignment). The Graph
  clustering example is just the experiment from FGW paper with outlier nodes
  added. The graph matching setup where part of the graphs are removed seem to
  be far from any realistic application. The geometry matching example is a
  beautiful visualization but again does no correspond to any real application.
  Overall the experiments lack diversity and realism.

### Questions
Please discuss weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The article proposes an extension of the existing method based on the Gromov-Wasserstein (GW) distance (from Optimal Transport theory) to solve ML tasks for attributed graphs (also called structured data). The authors use the Fused GW distance, where the GW distance between the graphs (coded as the sets of distances in a graph) to be compared and the attribute distances between nodes related by the transport plan are added (with some weights to balance them). This Fused GW distance is already well developed in the literature from 2019 and 2020.

The article points to the specific extension where the graphs are unbalanced, so that one can not expect to have the same total distributions for the two graphs. This is called the Fused Partial GW problem  (+ the sub-variant of Mass-constrained PGW problem) and the question has also been well studied between 2020 and 2024.

The novelty here is on two aspects: the authors propose to use a different, specific form for age distance between the attributes, as being the Total variation (instead of KL divergence or L_q distances as in previous works). From that, there derive algorithms from the Franck-Wolfe method and the Sinkhorn algorithm to solve this specific instance of FPGW or FMPGW problems. 

The authors have the result that the problem, formulated that way, provides a semi-metric. Then, they show examples on three ML task with graphs: clustering, graph matching, and geometry matching. For that, they use classical machinery for OT methods with graphs, computing barycenters for FPGW distances ; coding graphs as discrete mass measures ; selecting a specific node as anchor for geometry matching;  and so on. The results are good but they are not game changer on these questions.

### Strengths
The strengths of the article are : 

1. It is well written, with (almost) all required elements defined (e.g., there are 2 pages of notations in the supplements, and background on OT and all the elements up to GW distances). 

2. There is an effort to prove some theoretical properties: existence of minimiser for the unbalanced FGW or FPGW; properties related to being a metric or semi-metric;  details on the described algorithms and convergence properties.

3. I have found one point with some novelty: the fact that the proposed method allows to “put aside” nodes which are outliers in a graph, and not transport them to the other graph (as seen on Fig. 2). However, $N_G$ appear to be needed (the number of nodes which are not outliers) and this diminishes the usefulness of the method.

### Weaknesses
The article has several weaknesses and I find that they overcome the strengths of the article:

1. First of all, the work is way too **incremental**: the problem of FPGW (and its variants) was already well considered before, with methods to solve it and better discussions than in the present article about why the problem is relevant. Here, the main novelties are technical results and algorithms in a specific choice of distance ; and these technicalities and algorithms come almost straight from the literature.

2. The authors give **no element to justify whether their choices are better, or more sound, or just sound**, as compared to previous works. For instance, why use a total variation ? Is there a reason for that ? When to use the FW algorithm to the Sinkhorn version ? Why is the  Sinkhorn one mostly used in the experiments ? (and then, why propose also the FW version ?)

3. The numerical experiments are not that illuminating. They are on specific questions and situations already well known. The geometry matching one for instance is a niche topic, here on only one 3d mesh.The graph datasets used for clustering or graph matching are small.

4. For graph clustering, did not understand how the parameters were chosen. The number of non-outliers node appear to be known. How can that relate to any practical situation ? How were set the other parameters ? Also, it is written on page 7:
> This robustness is attributed to the partial matching property of FPGW.

Why that ? Any proof of this assertion ?

5. In 4.2, is the fact that only 50% of the nodes are extracted a known fact ? It seems that globally the method is only valid if each node corresponds to a node in the other graph, while an interest of the OT approach for ML on graphs is that it allows a change of scale in some situations (one node might be matched to 5 or 10 in the second graph, if need be). Also, knowing $N_G$ in advance seems to be a weak point of the method.

6. Table 1 has to be re-considered: are the std of the results (in %) precise up to 2 digits  ? How come some methods have a ±0.00 std ? This is not statistically sound. What is the unit of Time ? The results should include the FW algorithm. 

7. The example in 4.3 (geometry matching) is of limited impact. Also, the underlying task (“geometry matching task”) is not really detailed.

8. A limitation of methods based on Fused GW is often the computation time. Is it also the case here ? Or not because of the sinkhorn version ? What is the theoretical complexity and its scaling ?
The used datasets for 4.1 and 4.2 are not fully described but they are quite small graphs. Can the method be extended to larger graphs (in terms of nodes) ? 

9. In the results, FUGW appear to have often similar performance than sink-FPGW (in Table 1). Is it the case ? Then, the contribution is about acceleration by using a sinkhorn algorithm, no ? Why is this acceleration not obtained on the Douban dataset ? Is is related to  the size of this dataset ?

### Questions
I have a small list of suggestions or questions, that will not change the weaknesses that I see in this article :

- it seems that the definition of the total variation is not given (despite the 2 pages of notations). Also, it comes as $ \Vert . \Vert_{TV}$ or $ | . |_{TV}$ depending on the pages.

- Titouan et al. (2019a) should be Vayer et al. (2019a)

- Some misprints (examples: “postivie” on page 5 ; “RAPAMETER” in title M. ; “As shown in Table 2” on page 8 -> should be Table 1)

-  Add some ablative or comparative study to assess whether using the TV distance is sound for this problem.

- Compare the two proposed algorithms in many situations, and compare them better to existing approaches. For instance, it is not useful to compare them to methods assuming balanced sets, as this topic (partial/unbalanced vs balanced problem, has already been discussed in the literature). But the 2 proposed algorithms and ways to estimated the unbalances (+ ways to choose the parameters) should be compared to the literature.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Authors introduce a novel partial OT problem for structured data, such as graphs with node features,  named Fused Partial Gromov-Wasserstein (FPGW), which relates to a previous formulation called Fused Mass-constrained Partial Gromov-Wasserstein (FMPGW). They theoretically study the relationships between these two divergences and establish that FPGW define (semi-)metrics while endowed with $L_p$ distances as inner costs. They propose Frank-Wolfe algorithms to solve both problems and study their convergence. They also introduce an entropically regularized variant EPFGW inspired by previous works on unbalanced OT. For all OT problems, they study how to compute barycenters w.r.t these divergences and leverage those in Kmeans like algorithms to learn denoised graph structures.  The authors also showcase the empirical relevance of these metrics on many real-world datasets to perform subgraph matching tasks, matching of 3D meshes and finally, in the supplementary material, noise-robust graph matchings on several graph classification tasks.

### Strengths
-	introduce new variants for partial FGW matching, namely FPGW and an entropically regularized variant EFPGW. With solvers to estimate solutions to these OT problems, respectively a conditional gradient solver for FPGW and a sinkhorn-like solver for EFPGW.
-	Provide theoretical results on the convergence of the CG solver for FPGW and FMPGW (supplementary).
-	Provide theoretical results in Theorem 3.1 showing that: (1) FPGW can be formulated as a non-convex QP; (2) FMPGW and FPGW admit minimizers; (3) FPGW defines a (semi-)metric while endowed with $L_p$ losses.
-	Study in the supplementary relations between FPGW, FMPGW and FGW.
-	Introduce barycenter problems for these metrics and corresponding solvers to perform graph clustering, illustrated on a synthetic dataset containing SBM graphs with different number of clusters and varying levels of structural noise.
-	Showcase that EFPGW outperforms many competitors over 8 real-world graph datasets on subgraph matching tasks.
-	Show that EFPGW outperforms competitors on 3d mesh matching.
-	Study in supplementary the relevance of FPGW for noisy graph classification under different level of noise, showcasing that FPGW outperforms competitors when noise is high enough.

### Weaknesses
- **W1: overall clarity of the paper**: I believe the paper contains many writing issues which are detrimental to the clarity of the paper including:
   - L51: "similarity between datasets". I suggest authors to be more specific about what they mean here as it does not include datasets of structured data like graphs with or without node features.
   - L57-58: "structured feature data". This terminology seems very strange to me and unique to the paper. I would advice authors to change that for instance by referring to attributed graphs or graphs with node features, which admissible no matter to topology of the graph (metric or not)
   - L61-62: it is not clear in which sense relying on sinkhorn solvers for FUGW is a limitation.
   - I think it would be relevant to broaden the scope of the paper referring to the work of Chowdhury and al (2019) on GW between any networks in Section 2 as adjacency matrices are used in the experiments. The adjacency indicator function mentioned in Appendix L does not seem to be a metric as I believe it does not satisfy the triangle inequality.
   - Section 2.1: For comparison with your own results on FPGW, you should also detail the metric property of FGW when q=1.
   - Most theoretical results are not discussed in the paper for instance those in Theorem 3.1 or those only mentioned in the supplementary material.
    - Mention in L208 that you are referring to Eq. 13. And epsilon is not defined in the complexity analysis.
    - The caption of Figure 1 seems incomplete as the meaning of the red and purple nodes is not given.
    - No need to go to next lines in the caption of Figure 3. 

- **W2: sinkhorn solver**: 
   - The solver seems to solve for an entropically regularized variant of FPGW and not the exact problem. This should be clarified and referring to EFPGW in the rest of the paper instead of sink-FPGW would be clearer.
   - It is not clear to me why authors use a such regularization instead of a classical entropic regularization e.g the entropy of transport plan, see e.g Peyré and al 2016. Moreover authors should justify their choice of algorithm, as it does not seem to be able to solve the EFPGW problem but a relaxed version.

- **W3: relations between FPGW and FMPGW**:
   - I believe that Proposition K1 and so on should be mentioned in the main paper as these are important to understand the relevance of introducing two different solvers. Moreover the study of the relations between both OT problems seem incomplete as I believe that a user would be interested in how to link $\lambda$ (in FPGW$ and $\rho$ (in FMPGW). Could authors try to complete this proposition in this direction ? Intuitively i think that a relation can be rather simply established using KKT conditions even but maybe i am wrong.

- **W4: benchmark in Table 1**:
   - Overall it seems that the node features play a huge role to get a proper matching between studied graphs. Therefore i think it would be appropriate to increase the number of existing baselines able to leverage those, such as srFGW and FPGW (equivalent to FMPGW).
   - Could authors explain the trick mentioned in eq.106 in the supplementary and its impact in the benchmark ? 
   - It is not clear how hyperparameters have been validated for each method include theirs, could authors detail that ? Moreover could they provide a sensivity analysis on the parameters $lambda$ and potentially $epsilon$ ? 
   - Could authors explain why there are that significant differences in computational speed between sinkhorn-based solvers for FUGW and EFPGW ? 

**Tipos**:

   - Feydy & al (2017) seems to be misused in many ways and i believe that authors refer instead to a paper of Vayer and al.
   - L75-76: "Note" -> to remove ? 
   - L114: "structure" -> "structured" ? 
   - L168: "In" -> "In the" ? 
   - There are many mistakes in math notations within the paper such as the one in L177, $FPGW_{\rho}$ which should be $FPGW_{r, L, \lambda}$, please make a thorough review.
   - L384. graphs

### Questions
I invite the authors to directly address the weaknesses mentioned above. Overall I consider that the paper has many interesting contributions, but their presentation is rather poor. If the authors address my concern, I would gladly increase my score.

### Soundness
3

### Presentation
2

### Contribution
3
