# Do We Really Need to Approach the Entire Pareto Front in Many-Objective Bayesian Optimisation?

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Many-objective optimisation, a subset of multi-objective optimisation, involves optimisation problems with more than three objectives. As the number of objectives increases, the number of solutions needed to adequately represent the entire Pareto front typically grows substantially. This makes it challenging, if not infeasible, to design a search algorithm capable of effectively exploring the entire Pareto front. This difficulty is particularly acute in the Bayesian optimisation paradigm, where sample efficiency is critical and only a limited number of solutions (often a few hundred) are evaluated. Moreover, after the optimisation process, the decision-maker eventually selects just one solution for deployment, regardless of how many high-quality, diverse solutions are available. In light of this, we argue an idea that under a limited evaluation budget, it may be more useful to focus on finding a single solution of the highest possible quality for the decision-maker, rather than aiming to approximate the entire Pareto front as existing many-/multi-objective Bayesian optimisation methods typically do. Bearing this idea in mind, this paper proposes a \underline{s}ingle \underline{p}oint-based \underline{m}ulti-\underline{o}bjective search framework (SPMO) that aims to improve the quality of solutions along a direction that leads to a good tradeoff between objectives. Within SPMO, we present a simple acquisition function, called expected single-point improvement (ESPI), working under both noiseless and noisy scenarios. We show that ESPI can be optimised effectively with gradient-based methods via the sample average approximation (SAA) approach and theoretically prove its convergence guarantees under the SAA. We also empirically demonstrate that the proposed SPMO is computationally tractable and outperforms state-of-the-arts on a wide range of benchmark and real-world problems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a single point-based multi-objective search framework, SPMO, with the goal of improving the trade-off between solutions along the Pareto frontier by optimizing for a single point objective. The idea behind this is that emphasizing a single solution of high quality provides a better solution than exploring the entire Pareto front. The methodology uses a new acquisition function definition that emphasizes a single direction, given an oracle value, allowing BO to approach a point that might be closer to the user's ultimate preference. The authors define the method for both noisy and noiseless functions. The method shows improved results compared to baselines on the author’s single point metric and some improvement on hypervolume calculations across the Pareto front. This shows promise for an improved methodology for many-objective optimization.

### Strengths
- The paper is clearly written and provides an insightful overview of the intuition behind the method and the methodology used. The definition of the acquisition is clear and the methodology used for evaluation is well done.
- The paper shows SPMO outperforms baselines over various benchmarks across noisy and noiseless tests. In particular, the distance-based metric results show a clear superiority of SPMO over baselines. The methodology is defined clearly and tested on both the noisy and noiseless domains, which expands its usefulness.
- The runtime of SPMO compared to EHVI is superior and represents an important improvement over this competitive method. In particular, the method is much more efficient for many-objective (ex: 10 objectives) scenarios. The authors demonstrate that the method is flexible to extend to a batch mode as well.

### Weaknesses
- The method utilizes a known oracle value in its methodology, which may limit the applicability of the method to many real-world circumstances where little information may be known. The authors show an ablation over various scales applied to this oracle, but still rely on this knowledge, which makes the comparison to other methods more difficult to infer.
- While the method does not optimize for hypervolume of all solutions, EHVI is competitive/beats SPMO on this benchmark. For users who prefer this metric, they may not prefer SPMO. Further justifying the distance-based metric or best hypervolume as better metrics would be beneficial. 
- The results obtained on DTLZ 3-7 are less convincing, particularly for the hypervolume of all solutions. This is concerning that the results from the main paper are not consistent with these tests. While good benchmarks for these methods are difficult to find, further explanation or exploration of diverse test benchmarks would improve the paper.

### Questions
- How does the method perform without any knowledge of a known oracle? For example, inferring a possible oracle value based on previously evaluated points? Does an error in oracle setting (ex: a much lower oracle value for a specific parameter than is realistic) degrade results?
- Have you identified any problems/scenarios where SPMO fails compared to existing methods?
Can you justify why the results on DTLZ 3-7 are worse, particularly for the hypervolume of all solutions. Why do these differ from the results presented in the main paper and why is this not discussed further?
- How can a user’s preferences be incorporated into the method? Many users may select from the Pareto front based on differing desired objectives, but this may limit this possibility.

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
4

### Summary
The paper considers many-objective Bayesian optimization. Rather than aiming to find a representative subset of Pareto optimal solutions distributed over the whole Pareto front, which can become computationally intractable in BO (where the objective is expensive), the algorithm presents instead finds a single solution that lies closest (in the Euclidean distance sense) to a utopian point (presumably) suggested by the user, with the underlying motivation (assumption) that the user wants a single solution conforming to a pre-defined preference/bias.

### Strengths
1. The paper is easy to follow, covering relevant points.
2. The paper appears mathematically sound, and the convergence proof appears reasonable (though I have only skimmed the proof).
3. The experimental validation is thorough.

### Weaknesses
While the proposed solution in the paper is mathematically elegant, I do wonder about how user-friendly it would be in practice.  For example, suppose $f_1, f_2 \in [-1,0]$ and the user selects a utopian point $z^\star = (-1.1,-1,1)$. To me, this would indicate that the user wants (at least roughly speaking) an equal balance between the two objectives (perhaps I am misinterpreting here, in which case please correct me), but if if the Pareto front is concave this may well not happen.

For example, suppose for arguments sake that the Pareto front in our 2-d example is well approximated by the lines $f_1 = -0.5$, $-1 \leq f_2 \leq -0.5$ and $-1 \leq f_1 \leq -0.5$, $f_2 = -0.5$. In this case the closest (Euclidean) points on the front to the utopian point will be $f_1 = -1, f_2 = -0.5$ and $f_1 = -0.5, f_2 = -1$, which do not (at least as I understand it) reflect the desires of the user at all (who would presumably prefer the Pareto-optimal solution $f_1 = f_2 = 0.5$). And even if we assume a convex Pareto front there may still be problems - for example suppose the Pareto front is well approximated by the line $f_1 = -0.5$, $-1 \leq f_2 \leq 0$ - again, the closest point to the utopian point is $f_1 = -0.5$, $f_2 = -1$, which again does not appear to reflect the user's wishes.

Nor do I see an easy way to fix this problem by, for example, considering a non-Euclidean distance, as the choice of a "correct" norm (perhaps the $\epsilon$-norm (pseudo-norm) in the concave example above) is strongly dependent on the (a-priori unknown, and presumably unknowable in a reasonable time in the many-objective case if experiments are expensive) shape of the Pareto front.

Perhaps the solution is a fast way to find a solution, but in this case I must wonder if it would be better to use a scalarization scheme, which should be faster almost by definition.

### Questions
See weaknesses - basically, can you provide a more thorough discussion on (a) how the utopian point is selected and (b) how realistic it is to assume a connection between the preferences (presumably) reflected by the utopian point and under what conditions these preferences may be reflected in the solution found.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper argues that multiobjective optimization gets challenging as the number of objectives increase (because the surface area of the Pareto frontier grows exponentially with objectives). It further argues that identifying the Pareto frontier is wasteful since practitioners still need only one solution in the end. To overcome this, the paper proposes identifying a single solution in a multiobjective setting that provides the "best tradeoff" between objectives. They propose a "single point expected improvement" based on the Euclidean distance to a utiopian point $z^*$.

### Strengths
The paper identifies a relevant practical problem in multiobjective optimization -- scaling with number of objectives. However, I am not sure if what they propose is an appropriate solution to that challenge.

### Weaknesses
I have several fundamental concerns about this work.
1. They define $\geq 3$ objectives as "many-objective" optimization, while $\geq 2$ is called "multiobjective" optimization -- this seems quite arbitrary (and pointless) to me.
2. They argue that having too many objectives is particularly hard in Bayesian optimization because BO operates under a limited budget. Actually, it is quite the contrary -- BO is known for its sample efficiency and works well when the budget is limited, but is in no way restricted to a limited budget.
3. The authors argue that practitioners typically use only one solution from the Pareto frontier and hence finding the entire Pareto frontier is wasteful -- I disagree. The Pareto frontier, by definition, is a set of solutions that are nondominated (offer the best pragmatic tradeoff). Therefore, Pareto optimal solutions are, theoretically speaking, equally __good__. If a practitioner has to pick one out of this set, it would be based on domain knowledge specific to the application. Without finding the entire Pareto set, it would be impossible for practitioners to pick a single solution.
4. The proposed expected single point improvement (ESPI) essentially tries to find a solution that is closes to the utopian point $z*$ in the Euclidean sense. There is no justification as to why this would be the best choice.
5. The utopian point requires finding independent minimizers of each objective -- I find this to be nontrivial, and there is not sufficient explanation as to how they would address this in high (output) dimensions.
6.  The proposed theoretical results are nothing new or unique to this paper -- they are general guarantees when you have a stochastic acquisition function.
7. The experiments (Fig 2) show log distance (to $z^*$ I believe). It is not clear how did they compute this for other methods, where the entire Pareto frontier is obtained. Did they find the shortest Euclidean distance to the frontier?
8. The performance metrics used in the experiments are questionable. Finding the hypervolume using 1 solution can significantly under/overpredict the hypervolume. It is not clear what the authors are doing to address this.
9. The experiments do not include some recent developments in MOBO
      - qPOTS which finds the Pareto frontier via Thompson sampling and without any hypervolume computation- https://proceedings.mlr.press/v258/renganathan25a.html
      - MORBO which uses trust-regions for MOBO - https://proceedings.mlr.press/v180/daulton22a.html

### Questions
I have covered questions within the "Weaknesses" section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The primary problem with MOBO the authors address is the exponential increase in points on the Pareto front of a problem given a constant number of divisions of each objective. The authors claim that most Pareto-optimal solutions are not used in practice, so solving for many points on the Pareto front during MOBO is wasteful. The authors present an alternative to optimizing multiple points on the Pareto front of the MOBO problem, instead presenting a method to optimize in a single direction toward a selected utopian point using a novel gradient based method.

### Strengths
The approach is novel and tackles a salient problem; fully sampling the Pareto front is not scalable as the number of objectives increases. The authors demonstrate good performance on a variety of benchmarks, usually pareto-dominating other methods. The method is shown to be relatively robust toward the choice of utopian point. The paper demonstrates that the wall clock time is shorter than the other methods.

### Weaknesses
The authors should evaluate the method on a wider range of synthetic benchmarks widely used in the literature such as ZDT1-3, ZDT1-3, and VLMOP2-3. The authors should evaluate on some real-world benchmarks to prove that the method extends to objectives beyond those that can be parameterized by simple polynomial equations such as those presented in “An easy-to-use real-world multi-objective optimization problem suite” by Ryoji Tababe et al.

There are recent high-performance MOBO methods, such as MESMO and DGEMO, that the authors have not compared against.

### Questions
The authors show in their utopian point sensitivity analysis that selecting a suboptimal utopian point can lead to a better result. This is quite unintuitive, can the authors explain in more detail when cases like this actually happen? The authors should provide guidance as to how to actually select an optimal utopian point in the paper.

### Soundness
4

### Presentation
4

### Contribution
3
