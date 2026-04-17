# Efficient Sliced Wasserstein Distance Computation via Adaptive Bayesian Optimization

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 2, 6

## Abstract
The sliced Wasserstein distance (SW) reduces optimal transport on $\mathbb{R}^d$ to a sum of one-dimensional projections, and thanks to this efficiency, it is widely used in geometry, generative modeling, and registration tasks.  Recent work shows that quasi-Monte Carlo constructions for computing SW (QSW) yield direction sets with excellent approximation error.  This paper presents an alternate, novel approach: learning directions with Bayesian optimization (BO), particularly in settings where SW appears inside an optimization loop (e.g., gradient flows).  We introduce a family of drop-in selectors for projection directions: **BOSW**, a one-shot BO scheme on the unit sphere; **RBOSW**, a periodic-refresh variant; **ABOSW**, an adaptive hybrid that seeds from competitive QSW sets and performs a few lightweight BO refinements; and **ARBOSW**, a restarted hybrid that periodically relearns directions during optimization.  Our BO approaches can be composed with QSW and its variants (demonstrated by ABOSW/ARBOSW) and require no changes to downstream losses or gradients.  We provide numerical experiments where our methods achieve state-of-the-art performance, and on the experimental suite of the original QSW paper, we find that ABOSW and ARBOSW can achieve convergence comparable to the best QSW variants with modest runtime overhead.  We release code with fixed seeds and configurations to support faithful replication (see supplementary material).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a new framework for computing the sliced Wasserstein (SW) distance more efficiently by adaptively learning projection directions using Bayesian optimization (BO). While prior methods such as quasi–Monte Carlo (QSW) sampling select projection directions uniformly and independently of the data, this work leverages information from previously evaluated slices to focus on the most informative directions. The authors introduce four BO-based variants, BOSW, RBOSW, ABOSW, and ARBOSW, which differ in how often they refresh or refine projection sets and how they integrate with QSW initialization. These methods are designed as plug-and-play replacements requiring no change to downstream losses or gradients. Empirical results across synthetic benchmarks, point-cloud interpolation, image style transfer, and deep autoencoding show that the adaptive hybrids (especially ABOSW and ARBOSW) achieve state-of-the-art convergence with modest runtime overhead, outperforming or matching strong QSW baselines. Overall, the paper unites deterministic geometric designs with data-driven adaptivity to enhance efficiency and accuracy in large-scale optimal transport tasks.

### Strengths
1. The paper introduces the first use of Bayesian optimization (BO) to adaptively select projection directions for sliced Wasserstein computation. The connection between nonparametric-regression and SW is interesting.

2. The proposed methods (BOSW, RBOSW, ABOSW, ARBOSW) integrate seamlessly into existing SW pipelines without modifying loss functions or gradients.

3. The paper benchmarked against baselines (MC, QSW, RQSW) across diverse tasks: synthetic landscapes, 3D point clouds, image style transfer, and deep autoencoding.

4. Adaptive hybrids (especially ABOSW and ARBOSW) achieve state-of-the-art or comparable convergence with minimal runtime overhead.

5. BO refinements can be combined with quasi-Monte Carlo (QSW) approaches, showing versatility rather than replacement.

6. Code and experimental settings are carefully controlled for fair comparison, following prior benchmarks.

### Weaknesses
1.  BO introduces extra overhead and may not scale well to very high-dimensional problems or large projection budgets, despite some mitigation.

2. Learned projection sets (especially BOSW) are not unbiased estimators of the true SW integral, reducing theoretical rigor for some applications.

### Questions
1. The proposed nonparametric regression framework seems to go well beyond simply constructing a good uniform set of projection directions. In my view, it effectively defines a data-dependent Radon transform (or equivalently, a new slicing distribution), which induces a novel form of metric as in [1]. In [1], the authors reweight directions according to their corresponding 1D costs. Your regression-based framework appears to be a  powerful way to achieve something similar, using in-sample weights (training directions) and out-of-sample weights (new proposed directions). Should a discussion be added?

[1] Energy-based Sliced Wasserstein distance, Nguyen et al

2.  Section 2 is quite restricted when discussing only empirical 1D Wasserstein. The proposed method in the paper does not depend on the form of the distance, so, it is better to  discuss general form (see Section 2.3 in [2]).

[2]  An Introduction to Sliced Optimal Transport, Nguyen.

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
Many modern machine learning and data analysis tasks rely on computing the sliced Wasserstein (SW) distance to compare probability distributions. By treating the SW distance as a function of the projection direction, this paper introduces a method that uses Bayesian Optimization (BO) to automatically find the most informative projection directions for these tasks. Compared with traditional methods such as quasi-Monte Carlo (QSW) and randomized QSW (RQSW), which sample directions more evenly but do not adapt to the specific structure of the task, the BO approach identifies the most effective directions during the optimization process, allowing it to choose directions that provide the most useful information.

### Strengths
While Bayesian Optimization (BO) has been widely used for computing expectations, applying it to guide projection direction selection in sliced Wasserstein (SW) distance estimation for machine learning tasks is a new idea.

### Weaknesses
1, BOSW (and even its hybrid variants) may yield biased SW estimates since the BO-selected directions depend on the specific data. As a result, it may converge to a different value from the true SW. This issue is also noted in the paper (see line 299). However, a contradictory statement appears on line 109, where the authors state that these methods achieve state-of-the-art efficiency for estimating SW.

2, For simulation in subsection 4.2, the evaluation metric (maximum fitness value) favors BO by design, since BO is built to maximize scalar objectives, not to approximate integrals, making the comparison against QSW conceptually unfair. In addition, BOSW performs worse than both Monte Carlo and all QSW variants across all budgets L, confirming that BO-selected directions hurt uniform integration accuracy. 

3, In subsection 4.3, the improvements are small and sometimes disappear in later steps. BOSW shows high variance and weak convergence. RBOSW and ARBOSW perform better early on but take much longer to run, taking ten times the runtime of other methods.

4, Most results are averaged over only three to five runs.

5, The experiments omit comparisons against other adaptive sampling or Bayesian-quadrature approaches that would provide stronger baselines.

### Questions
1, Could the authors quantify the estimation bias, either through empirical experiments or theoretical analysis?

2, If not, I would suggest the authors to revise the claim "these methods achieve state-of-the-art efficiency for estimating SW", as the proposed BO-based estimators are data-dependent and therefore biased, not designed for unbiased SW estimation.

3, Could the authors repeat the experiments more times to minimize random fluctuations or numerical instabilities?

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
4

### Summary
This paper proposes several Bayesian optimization (BO)–driven methods to improve the efficiency and adaptivity of sliced Wasserstein (SW) distance computations. The key idea is to learn projection directions adaptively—rather than sampling them quasi-uniformly as in Quasi-Monte Carlo SW (QSW)—by using Gaussian-process-based BO on the sphere. Four variants are introduced: (i) BOSW – one-shot BO; (ii) RBOSW – periodically refreshed BO; (iii) ABOSW – QSW-seeded lightweight refinement; (iv) ARBOSW – restarted hybrid (periodic QSW seeding + BO refinement). Experiments on synthetic benchmarks, point-cloud interpolation, image style transfer, and deep autoencoders show that the adaptive BO variants (especially ABOSW and ARBOSW) achieve competitive or superior convergence to QSW and randomized QSW (RQSW) baselines with modest overhead.

### Strengths
1. The idea of combining Bayesian optimization with sliced Wasserstein direction selection is novel and natural. The approach bridges deterministic geometric constructions (QSW) with data-driven adaptivity, potentially broadening the design space of SW estimators; 

2. The authors reproduce the entire QSW benchmark suite (Nguyen et al., 2024a) for fair comparison. Tasks span synthetic projection selection, approximation accuracy, dynamic interpolation, style transfer, and autoencoding—demonstrating both theoretical and practical relevance; 

3. The methods require no change to downstream objectives or gradients, making them easy to plug into existing OT pipelines. 

4. RBOSW provides strong early-stage performance. ARBOSW matches or slightly exceeds state-of-the-art RQSW on long-run optimization tasks. ABOSW improves deterministic settings such as autoencoder training; 

5. The paper provides clear mathematical exposition, fixed random seeds, and references to public code.

### Weaknesses
1. Although an appendix sketches guarantees, there is no formal convergence proof or error bound for the BO-selected projections compared to the true spherical integral; 

2. The method relies on GP-based BO, whose cubic cost in the number of evaluations becomes prohibitive as $L$ or dimensionality $d$ increases. This is acknowledged but not resolved beyond citing recent BO-in-high-dimension literature; 

3. In static estimation tasks, BOSW performs worse than QSW since it is not designed for uniform coverage. The benefits only appear when SW is nested inside an optimization loop; 

4. While conceptually appealing, the improvements over strong QSW baselines are modest in some tasks (e.g., style transfer) and might be sensitive to hyperparameters; 

5. The reported improvements lack variance/error-bar plots or significance testing.

### Questions
1. Could the GP surrogate be replaced by a neural surrogate to improve scaling?
2. How sensitive is performance to kernel choice or acquisition parameters?
3. Is there a way to enforce unbiasedness while retaining adaptivity?
4. How does performance evolve with dimensionality (e.g., $d>10$)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors tackle the problem of choosing the right directions to use in the sliced Wasserstein (SW) distance when it is used inside an optimization loop. They propose using Bayesian optimization (BO) to select these directions, with the goal of adapting them to the evolving probability distributions. To this end, they propose four algorithms: BOSW, RBOSW, ABOSW, and ARBOSW. The first two algorithms learn a Bayesian process to compute $L$ SW directions, which are then kept fixed during the subsequent learning task. For the small set of initial points used to initialize the Bayesian process, BOSW uses random points sampled uniformly on the sphere, while ABOSW uses points sampled via a quasi–Monte Carlo method. The RBOSW and ARBOSW algorithms periodically refit the Bayesian process during the learning task, using BOSW and ABOSW, respectively, to adapt to the evolving probability distributions throughout training. The authors tested the four algorithms on different tasks (including learning with SW loss) and compare them to quasi–Monte Carlo approaches.

### Strengths
- The idea of carefully selecting directions for SW during the learning task, when probability distributions evolve, is a difficult but very interesting problem, as it would permit the use of SW as a surrogate for the Wasserstein distance (in a learning context), which is expensive to compute.

- The experiments conducted by the authors permit a comparison between QSW methods and BO methods on various tasks to gain an intuition about how the BOSW algorithm behaves.

### Weaknesses
- The paper lacks clarity on the type of learning problems that would be solved using BO inside SW.

In Section 2, paragraph “Problem view for this work” (l.168), the authors define the learning tasks targeted by their method as:

“When SW is used inside an outer optimization (e.g., gradient flows or registration), the same direction set 
$\Theta$ is queried repeatedly on evolving point clouds.”

From this sentence, it seems that the problems targeted are certain bilevel optimization problems, although in their applications (see Sections 4.3, 4.4, 4.5), the problems solved involve a direct optimization of the sliced Wasserstein distance between an optimized distribution and a target distribution.

Moreover, this statement also suggests that, for the problems considered, the learned directions are fixed during optimization, although the authors propose two algorithms, RBOSW and ARBOSW, that update the set of directions during the learning task.

- For problems that aim to optimize a parameterized distribution by minimizing the SW distance between this distribution and a target distribution, the authors should compare their approach to the most natural use of SW, which consists of randomly sampling new directions on the sphere at each time step. I do not see why keeping the same set of directions $\Theta$ would be a good idea, as the model risks optimizing only along these fixed directions. 

- The paper is not easy to read. I suggest that the authors add more background on BO and quasi–Monte Carlo methods. While there is a long recap of the background on Wasserstein and sliced Wasserstein distances, the section on BO is very brief.

### Questions
- Can you provide references for the problems you mention in the paragraph “Problem view for this work” that use SW in an “outer optimization” and keep the same set of $\Theta$ directions during the learning task? Can you also explain what you mean by “outer optimization”?

- Can you compare your approach to the natural baseline that would uniformly sample new directions on the sphere at each time step for Experiments 4.3, 4.4, and 4.5? Is there a reason why you did not include this baseline?

- For your algorithms RBOSW and ARBOSW, you periodically rerun BOSW and ABOSW. It would be helpful to derive a criterion for when the Gaussian process should be updated (and new directions generated). How do you decide when to perform this update? Did you try to measure the impact of the timing of this update on the performance you obtained?

- Do you know why your methods (except ARBOSW) perform poorly in Experiments 4.3 and 4.4 but achieve very good performance in Experiment 4.5?

- Ideally (although infeasible because it is too costly), we would use BO to choose each new $\Theta$ direction and resample new directions at each time step. Since this is too computationally expensive, it seems necessary to design a method that achieves a trade-off between sampling random directions, keeping some directions fixed, and selecting new directions with BO. Do you think that, instead of keeping the directions fixed between BO updates, uniformly sampling over the sphere could improve performance?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work introduces a new family of simple, plug-and-play methods for selecting projections in the Sliced-Wasserstein (SW) distance. The key contributions are seamless Integration and novel selectors.

### Strengths
1) The fact that your method can be integrated into existing QSW/RQSW pipelines without changing downstream losses or gradients is a huge advantage. It means adoption is easy. 

2) Using Bayesian Optimization  to drive projection selection is a very logical and promising direction. I would like to point out this solution 

3) High-dimensional experiments

### Weaknesses
1) Comparison with other image-to-image models like Cycle GAN, DSBM or Optimal transport solvers

### Questions
1) This is the critical trade-off. How much more computational time does their BO-driven selection add compared to simpler random or linear selection methods? The definition of "modest" can be subjective, and the overhead might be acceptable for some applications but not others.

2) Could you provide expeirmtns with translation with resolution 128x128?

### Soundness
3

### Presentation
3

### Contribution
2
