# Online Rounding and Learning Augmented Algorithms for Facility Location

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Facility Location is a fundamental problem in clustering and unsupervised learning. Recently, significant attention has been given to studying this problem in the classical online setting enhanced with machine learning advice. While (almost) tight bounds exist for the fractional version of the problem, the integral version remains less understood, with only weaker results available. In this paper, we address this gap by presenting the first online rounding algorithms for the facility location problem, and by showing their applications to online facility location with machine learning advice.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents the first online rounding algorithms that convert fractional online facility-location solutions into integral ones while preserving competitive theoretical guarantees. The main results include: (1) a deterministic constant-factor rounding algorithm for uniform opening costs, and (2) a randomized rounding algorithm for non-uniform costs with $O(\log\log\Delta)$ expected loss, where $\Delta$ is the aspect ratio of the given instance. These rounding schemes are then plugged into prior fractional, learning-augmented (multiple-predictions) algorithms to yield the first integral solutions with $O(\log(k+1))$ (uniform) and $O(\log\log\Delta\log(k+1))$ (non-uniform) consistency, together with $O(\log t/\log\log t)$ robustness via a combiner. A lower bound shows the $O(\log\log \Delta)$ dependence is asymptotically tight for the randomized scheme, supporting the claim that the main hardness is obtaining a strong fractional solution rather than rounding it.

### Strengths
The uniform-cost algorithm is 4-consistent with a constant-factor competitive cost. For non-uniform costs, the algorithm is 5-consistent and can control the expected facility cost via a vertex-level scheme, achieving $O(\log\log \Delta )$. The resulting level/critical-ball construction is clear and metric-aware. Combined with the fractional learning-augmented algorithm of Anand et al. (2022), the rounding process yields the first integral multiple-prediction results with nearly-tight consistency and robustness guarantees. A matching $\Omega(\log\log \Delta)$ lower bound shows this dependence is unavoidable for randomized rounding, supporting the claim that the main hardness lies in finding a strong fractional solution, not in rounding. Finally, the appendix gives a reduction that trades $\log\log \Delta$ for $\log\log n$, which is useful when $\Delta$ is large with moderate data size $n$.

### Weaknesses
The main weakness can be summarized as follows.

1. For non-uniform costs, the guarantees scale with the metric aspect ratio $\Delta$ (or, via reduction, with $n$), which can be large in practice. 

2. The robustness improvement chooses the cheaper of two online solutions by opening all its facilities, while facilities opened by the other solution remain. This preserves guarantees but can monotonically increase the number of open facilities over time; the paper does not analyze recourse or stability beyond competitive cost.

3. It is suggested that the paper should clarify the novelty, where the proposed  is the first online rounding algorithms for metric facility location, not the first results on online facility location overall. 

4. Because the three-level ball scheme is central, giving an explicit example before Conditions A/B/C would make the presentation clearer.

### Questions
1. It is unclear for me that why the online setting is the right lens (irrevocable decisions under partial information), and why rounding—rather than runtime—is the key bottleneck.

2. Could you instantiate the model with a simple scenario (e.g., two competing predictors) and show how your rounding plus the combiner behaves under accurate vs. adversarial advice?

3.  Can the randomized rounding guarantee be made $O(\log\log n)$ in the main text (not only via Appendix E’s reduction), or do structural barriers preclude this?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper studies the online facility location problem under the learning-augmented setting, aiming to bridge the gap between fractional and integral solutions. The authors propose two online rounding algorithms-- a deterministic algorithm for the uniform-cost case achieving a constant approximation ratio and a randomized algorithm for the non-uniform-cost case achieving an $O(\log \log \Delta)$ approximation.

These algorithms can be combined with existing fractional solutions (Anand et al., 2022) to yield near-optimal integral algorithms for the learning-augmented facility location problem. The work establishes that the main difficulty of the classic online facility location lies in computing high-quality fractional solutions rather than rounding them.

### Strengths
1. The results provide a conceptual bridge between fractional and integral formulations, advancing the theoretical understanding of online algorithms.

2. The deterministic algorithm achieves constant-factor guarantees, while the randomized version achieves $O(\log \log \Delta)$, improving over previous bounds.

3. The paper tackles an open problem by presenting the first online rounding algorithms for the facility location problem in both uniform and non-uniform cases.

4. The approach extends learning-augmented facility location to integral settings with nearly tight consistency–robustness trade-offs.

### Weaknesses
1. Although the paper successfully extends the known fractional results to integral settings, the achieved approximation and consistency–robustness bounds do not improve upon the theoretical lower bounds established in prior works (Almanza et al., 2021; Anand et al., 2022). In other words, the results are asymptotically tight but not stronger than existing bounds, which somewhat limits the theoretical novelty of the contribution. 

2. The technical exposition, while mathematically sound, is dense and could benefit from clearer intuition or illustrative examples of the rounding procedure. 

3. The applicability of the rounding framework to other online covering problems (e.g., online set cover, k-median, or caching) is not discussed, which limits the broader impact of the work. 

4. Although the theoretical results close an existing gap between fractional and integral formulations, the technical novelty of the algorithms themselves seems incremental. Many arguments (e.g., in Lemma 1 and Theorem 2) follow standard geometric consistency proofs, and the randomized rounding procedure closely resembles existing metric-dependent rounding techniques. As a result, the contribution may feel more like a careful synthesis of known tools than a fundamentally new technical breakthrough. 

5. The paper is entirely theoretical. There is no experimental validation or numerical evidence illustrating the practical behavior of the algorithms or how large the hidden constants are in real settings.

### Questions
1. Could the authors provide empirical or synthetic experiments to demonstrate the actual performance or confirm the tightness of the constants?

2. Is it possible to adapt the rounding framework to directly construct improved fractional algorithms or to other related online optimization problems?

3. How sensitive is the performance to the quality of the fractional input—e.g., if the fractional approximation is only moderately good?

4. The paper argues that the rounding stage is no longer the main source of difficulty in online facility location. Could the authors provide further justification or empirical evidence supporting this claim—for instance, by showing that different fractional baselines yield similar integral performance after rounding?

### Soundness
2

### Presentation
1

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
This paper discusses rounding algorithms for the fractional facility location problem under both the cost structures, uniform and non-uniform. That is, when the cost of establishing a new facility is either constant or varies across facility locations. It presents interesting theoretical results that address one of the key open problems of facility location topic. Authors propose online learning algorithms that leverage the solution of the fractional problem to construct solutions for the corresponding integral versions. The algorithms are well-motivated, clearly explained, and supported by sound theoretical reasoning.

### Strengths
The paper presents proposes a novel solution for facility location problem. The requests arrive sequentially and hence this is related to the  clustering (unsupervised learning) problem. The performance measure is approximation ratio. The key idea is to transform the fractional solutions of the online facility location problem into feasible integral solutions. The authors build upon and extend prior approaches that address several key limitations, achieving a better approximation ratio.


Authors consider two main settings: one where the facility setup cost is uniform across all locations, and another where the costs vary by vertex (the non-uniform cost structure). This broader treatment is interesting as the model is more realistic and relevant to practical scenarios.


The paper has strong theoretical depth; the theorems and their proofs developed nicely. While the discussion connects partially to existing literature, the algorithmic descriptions are particularly detailed and well explained. Important notions such as critical balls, randomized rounding, and the role of aspect ratios in determining approximation guarantees are well defined. These explanations enhance the readability of the technical sections; interested readers, who may not be into this area, may also be able to follow.

### Weaknesses
The paper is theoretical in tis nature, focusing on proving approximation bounds and analyzing the properties of rounding algorithms. While the theoretical contributions are strong, the practical relevance particularly in learning-augmented settings could be strengthened through empirical validation. Incorporating simulations or experiments on synthetic or real-world datasets would help demonstrate the applicability of the proposed methods beyond the purely theoretical framework.

Next, several assumptions made in the paper, such as the preservation of fractional mass and the existence of specific geometric or metric properties (e.g., aspect ratio bounds), are not well justified. It is unclear whether these are standard assumptions in prior literature or newly introduced for tractability. If these are common, the authors should explicitly cite supporting works; if they are novel, a more thorough discussion or justification of their necessity and implications is required. As of now, some of these assumptions appear tailored to make the analysis feasible rather than naturally arising from the problem structure. Comments by the Authors are desirable. 

The paper also references the open challenge of achieving an approximation factor of O(log /log log K) in the non-uniform cost case. However, this result is asymptotic, and this limitation is not highlighted in the discussion. It is very desirable if Authors explicitly acknowledge this gap and outline possible directions for bridging this gap.

As far as the presentation is concerned, several definitions such as critical balls, levels, and probabilistic rounding steps are dense, making it hard to parse. Many readers would like to have easy to explain examples or see illustrations (schematic diagrams); they could greatly enhance the readability for the readers that are less familiar with the literature. 

Finally, while the derived bounds involving factors like (log \Delta) are theoretically interesting, the tightness of these bounds is not well discussed. Clarifying whether these bounds are provably near-optimal or if there is potential for improvement would add depth to the technical discussion. Computational illustrations help here.

### Questions
In addition to the above, I would suggest some clarifications and further discussion on the following points:

In the absence of computational evaluations, It is not clear to me how the proposed algorithms perform on small-and large-scale instances, especially those with high aspect ratios or complex metric spaces. At least small scale (initial) experimental results evaluating the computational complexity and scalability of the proposed rounding algorithms would be useful. 


Are there specific optimization techniques that Authors recommend for implementing the algorithms efficiently in practice? 
The analysis assumes that the fractional mass y^t_v​ remains fixed during the rounding process or can be approximated as such via vertex splitting. Could you comment on real-world scenarios where this assumption may not hold strictly? How would potential violations of this assumption affect the validity or tightness of the approximation guarantees?


The derived bounds naturally depend on the aspect ratio of the metric space, \Delta_t​. In dynamic or evolving settings where this ratio may vary significantly over time, how sensitive are the theoretical guarantees to such changes? 

In the absence of experimental results, it remains unclear how the proposed methods compare with existing online facility location algorithms, such as those by Meyerson or more recent learning-augmented approaches. A qualitative or theoretical comparison in terms of approximation factors and operational complexity would help position this work within the broader literature.

### Soundness
2

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
4

### Summary
This paper studies the Facility Location problem, a fundamental clustering task, in the context of online learning augmented algorithms. The authors identify a key gap in existing research: while strong, nearly tight bounds exist for the fractional version of the problem (where facilities can be partially opened), the integral version (where facilities are either fully open or closed) remains less understood. The paper's main contribution is to bridge this gap by introducing the first online rounding algorithms designed to convert a fractional solution into an integral one efficiently. Two algorithms are presented: first, a deterministic algorithm for the uniform cost case, in which all facilities have a cost of $1$, that achieves a constant $(O(1))$ cost overhead compared to the fractional solution, and second, a randomized algorithm for the non-uniform cost case that incurs an $O(\log\log\Delta)$ loss. By combining these rounding techniques with the known fractional algorithms of Anand et al. (2022), the paper derives the first integral learning-augmented facility-location algorithms with consistency and robustness guarantees that match the previously known ones for the fractional case.

### Strengths
One of the paper’s strengths is its novelty and solid theoretical contribution, as it introduces the first online rounding algorithms specifically for the facility location problem, essentially reducing the integral version of the problem to the fractional one. The paper is also well structured, with proofs that are intuitive and technically involved.

### Weaknesses
The paper title might be a bit misleading as the results are entirely on online rounding algorithms with applications to learning augmented algorithms. The results can be applied to existing learning augmented algorithms, but the novelty is only on the online rounding part.

### Questions
- Could you apply similar techniques to directly get an online learning augmented algorithm by treating the input fractional solution as the ML advice?
- You mention as future work finding a constant-factor rounding for the non-uniform case. What seems to be the main obstacle for that?

### Soundness
3

### Presentation
3

### Contribution
3
