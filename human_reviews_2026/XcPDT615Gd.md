# Entering the Era of Discrete Diffusion Models: A Benchmark for Schrödinger Bridges and Entropic Optimal Transport

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 2

## Abstract
The Entropic Optimal Transport (EOT) problem and its dynamic counterpart, the Schrödinger bridge (SB) problem, play an important role in modern machine learning, linking generative modeling with optimal transport theory. While recent advances in discrete diffusion and flow models have sparked growing interest in applying SB methods to discrete domains, there remains no reliable way to assess how well these methods actually solve the underlying problem. We address this challenge by introducing a benchmark for SB on discrete spaces. Our construction yields pairs of probability distributions with analytically known SB solutions, enabling rigorous evaluation. As a byproduct of building this benchmark, we obtain two new SB algorithms, DLightSB and DLightSB-M, and additionally extend prior related work to construct the $\alpha$-CSBM algorithm. We demonstrate the utility of our benchmark by evaluating both existing and new solvers in high-dimensional discrete settings. This work provides the first step toward proper evaluation of SB methods on discrete spaces, paving the way for more reproducible future studies. The code for the benchmark and all associated experiments is available at [this repository](https://github.com/gregkseno/catsbench).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors proposed a mixture-like benchmark for SB on discrete spaces to evaluate the effectiveness of different methods. The authors also proposed a light version of DSB and a smoothing version of the CSBM algorithm, α-CSBM,  to enhance the training stability and efficiency. Interesting comparisons are made on toy examples to show the strength and weakness.

### Strengths
1. Providing a mixture example to evaluate the accuracy of different approaches is helpful to the community.

2. The light (low rank) and alpha smoothing do have some potential to accelerate the algorithm.

3. IMF is more appealing and interesting than the IPF style of algorithms.

### Weaknesses
1. Even the "expensive" version of the CSBM algorithm is evaluated on real-world image experiments. The light and alpha-smoothed algorithm is only evaluated on the mixture benchmark. This is far from sufficient.

2. The writing is not good. The current draft is a mixture of lots of components, like a baby mixture benchmark, 2 light algorithms (without sufficient discussions on theoretical properties or empirical evaluations). The scope is ambitious, but for each contribution itself, it is a bit minor.

3. Although low0

### Questions
1. Could you provide an additional section (like section 4.4 in CSBM) to verify the scalability?

2. i didn't get line 352: "be generalized to an arbitrary reference process qref, thereby enabling the application of the optimal projection in discrete space settings." If some algorithm accuracy is compromised (like low rank), does it still guarantee optimal projection? it is a bit confusing. 

3. I haven't fully checked CSBM yet. Is the algorithm simulation-free or not? To me, low-rank may be interesting but the simulation-free property is more important for the scalability.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, an analytic benchmark for discrete optimal transport is developed. In order to assess capabilities of current algorithms for solving the schrodinger bridge problem on discrete state spaces, there is a need for a benchmark for which the ground truth solution is known. The authors construct this analytic solution by writing the optimal transition distribution as a scalar function multiplying the reference transition distribution. Then a specific mixture of factorized forms is proposed for the scalar function such that the true schrodinger bridge transition distribution becomes analytically calculable. The authors propose two new light schrodinger bridge algorithms for discrete state spaces and compare them with existing methods on their benchmark.

### Strengths
The use of the mixture of factorized functions for the scalar tilting function looks novel to me and is an interesting way to construct a non-trivial problem for which the schrodinger bridge is known. The same factorization technique is then also used to construct the Light Schrodinger algorithms which show very good performance compared to prior bridge matching approaches. It is important to have these Light Schrodinger bridge baseline algorithms for further comparison during algorithm development as they serve as useful reference points.

### Weaknesses
The benchmark seems to already be too easy because the DLightSB method is already almost saturating the performance metrics. This means the benchmark is already losing its discriminative power between the best methods. Ideally, the benchmark would contain sub-tasks for which all methods struggle. Could the task be made more difficult by increasing the number of mixture components in the scalar function?

In the introduction in line 47 you state that one motivation for introducing this benchmark is to check whether methods actually solve the SB problem separating out 'true algorithmic performance from artifacts of specific parameterizations'. However, then on L431 you hypothesize that DLightSB performs the best on this benchmark because it is built on the same factorization principle as the benchmark itself. This would then seem to go against the stated goal of the benchmark to try and not be influenced by specific parameterizations of the method. Perhaps the benchmark needs to be made more complex as to not be 'defeated' by the DLightSB algorithm.

The differences between your two proposed methods DLightSB and DLightSB-M could have been investigated more. You hypothesize that DLightSB-M works worse due to error accummulation. This could have been verified or falsified with a further experiment, for example, changing the number of discretization time steps in the diffusion process to see if the discrepancy between the two methods gets smaller.

Equations (19) and (20) are quite hard to parse due to the circular definition of the set \mathcal{S}. The set itself is used within its own definition in the argmax and makes the whole equation hard to understand.

Overall, I think the contribution of the rank-1 Canonical Polyadic decomposition for constructing both ground truth SBs and for learning LightSBs is a worthy contribution to the conference notwithstanding the limitations in evaluations of the proposed methods.

### Questions
Why do CSBM algorithms have better performance on the benchmark with higher dimensionality? I would expect that performance would degrade with higher dimensions given a fixed model capacity but perhaps because of the low dimensional structure in the benchmark this is not the case?

Wouldn't it be possible to create a metric that doesn't just take into account single marginals or pairs of marginals by looking at distribution metrics? For example you could take a given starting point x0, sample p(x1 | x0) from the learned SB and the true SB then compute a distributional metric between those two sample distributions. Then you could average these values over different draws of x0.

### Soundness
3

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
4

### Summary
This paper presents a benchmark for SB methods on discrete spaces. The contributions of this work seem to be two-fold: (1) a theoretical investigation for constructing a benchmark for discrete SB problems and (2) an actual benchmark along with initial results, including three new implementations of SB algorithms. The authors constructed a practical benchmark based on Gaussian mixtures. The implemented DLightSB algorithms (discrete variants) achieved the best results.

### Strengths
Based on my reading, the paper has the following strengths. 
* Solid theoretical investigation and a well-grounded evaluation strategy for discrete SB problems. 
* An actual benchmark that will benefit the SB community. 
* Comprehensive empirical experiments and new discrete SB algorithms

### Weaknesses
I believe this work in its current format lacks novelty in many regards.

* The theoretical arguments presented in this paper can be seen as either well-known or a reformulation of continuous SB work. I generally view expanding discrete theoretical work to continuous spaces as more challenging and practical, not the other way around.
* The proposed benchmark setup relies heavily on Gaussian mixtures, which were originally intended for representing distributions in continuous spaces, and this tarnishes one of the central contributions of the paper (focusing on the new discrete problem), as the statistics will share many characteristics with continuous problems.
* It is widely known in the SB community that LightSB algorithms excel in Gaussian mixture EOT benchmarks, since LightSB actually has a parametric prior of Gaussian mixture models (see Korotin et al. and Gushchin et al.). Therefore, I am somewhat certain that the authors were highly knowledgeable of this information before the experimental design and engineered DLightSB and DLightSB-M to show similarly good performance on their platform. In other words, these results contain very similar implications to the two LightSB papers, which makes the novelty of this paper weak, since not much new information is given beyond the references.
* The sheer number of benchmarks is lacking, and four important algorithms (DLightSB, DLightSB-M, and α-CSBM) can be generally considered as reformulations from the work of a single research group. The authors are encouraged to put some more effort into implementing other discrete formulations of competitive SB algorithms.

### Questions
* Is there a table that comprehensively summarizes the basic statistics of the GMM benchmarks? (D={2, 16, 64})

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper identifies a valid and significant gap in the machine learning literature: the absence of a standardized, ground-truth benchmark for evaluating Schrödinger Bridge (SB) solvers on high-dimensional discrete state spaces. The authors' stated contributions are:
1.  A method for constructing benchmark pairs of probability distributions $(p_0, p_1)$ on discrete spaces ($\mathcal{X} = \mathbb{S}^D$) for which the analytical SB solution $q^*$ is known by design.
2.  The use of a Canonical Polyadic (CP) decomposition to parameterize a key scalar function ($v^*$), rendering the benchmark construction and sampling computationally tractable in high dimensions.
3.  The introduction of three new or adapted solvers (DLightSB, DLightSB-M, and $\alpha$-CSBM) as "byproducts" of this framework.
4.  An empirical evaluation of these new solvers against the existing CSBM method, using the proposed benchmark, which concludes that the DLightSB and DLightSB-M methods demonstrate superior performance.

Despite the commendable goal, the paper's execution suffers from several fundamental weaknesses in its methodology, evaluation, and framing that undermine its primary conclusions and limit its utility as a general benchmark for the community.

### Strengths
1. Given a source marginal, and a scalar valued function (what is known in the literature as potentials) it is easy to obtain a corresponding target marginal and the solution SB distribution. The construction is simple and practical and serves as a good synthetic test bed.

2. The parametrization of the scalar valued potential using rank-1 Canonical Polyadic decompositions seems to be versatile (in theory) given that they can act as universal approximators and possess nice theoretical properties.

### Weaknesses
### 1. Stated Theoretical Contribution: Benchmark Construction

The paper's central theoretical claim is the introduction of a method for constructing ground-truth benchmark pairs for the Schrödinger Bridge (SB) problem on high-dimensional discrete spaces. This method yields pairs of probability distributions $(p_0, p_1)$ for which the analytical SB solution $q^*$ is known by design, enabling rigorous evaluation.

This construction is formally presented in **Theorem 3.1**, which states:

1.  One begins with a known source (initial) distribution $p_0 \in \mathcal{P}(\mathcal{X})$.

2.  One also defines a scalar-valued function $v^*: \mathcal{X} \rightarrow \mathbb{R}$.

3.  The "ground-truth" joint distribution $q*$  is then constructed such that its $x_0$ marginal is $p_0$ and its conditional distribution $q ^* (x| y)$ is proportional to the product of this new function $v*$ and the reference transition kernel $q^{ref}(x_1|x_0)$:

    $$q ^* (x_1|x_0) \propto v^*(x_1) q^{ref}(x_1|x_0)$$,

or 

   $$q ^* (x_1|x_0) =v^*(x_1) q^{ref}(x_1|x_0) / c(x_0).$$

4.  The target distribution $p_1$ is then *defined* as the second marginal of this $q ^* (x_0, x_1) = p_0(x_0) q ^* (x_1|x_0)$.
5.  The theorem concludes that the $q^*(x_0, x_1)$ constructed this way is the true static SB solution between the resulting $p_0$ and $p_1$.


### 2. Marginal Theoretical Novelty

The theoretical novelty of this construction (Theorem 3.1) is only marginal. It is a "reverse-engineered" benchmark based on a restatement of the well-established dual solution to the Entropic Optimal Transport (EOT) problem, which the paper itself shows is equivalent to the static SB problem (Section 2.3).

* **Standard EOT Duality:** The solution $\pi ^*$ to the EOT problem $\min_{\pi \in \Pi(p_0, p_1)} \text{KL}(\pi || R)$ is uniquely characterized by two potentials (Lagrange multipliers) $\phi$ and $\psi$ such that the optimal coupling has the form:

    $$\pi^*(x_0, x_1) \propto \phi(x_0) \psi(x_1) R(x_0, x_1)$$

* **Deriving Theorem 3.1 from first principles:** The paper's construction can be easily derived from the known result above.

    * The paper's *reference measure* $R(x_0, x_1)$ is $q^{ref}(x_0, x_1)$, which is assumed to have the $p_0$ marginal: $R(x_0, x_1) = p_0(x_0) q^{ref}(x_1|x_0)$.
    * Suppose we construct the optimal coupling as $q ^* (x_0, x_1) = \phi(x_0) v ^* (x_1) q^{ref}(x_0, x_1)$, and we want its first marginal to be $p_0$. Integrating out $x_1$ gives $p_0(x_0) = \phi(x_0) p_0(x_0) \sum_{x_1} v ^* (x_1) q^{ref}(x_1 | x_0)$, so canceling out $p_0$ terms gives $\phi(x_0) = 1/c(x_0) = 1/ \sum_{x_1} v ^* (x_1) q^{ref}(x_1 | x_0)$. This is exactly the paper's construction. 

Since one can easily deduce the construction in Theorem 3.1 from a very well-known first principle in SB/EOT literature, it doesn't appear as a significant contribution in this work. 

### 3. The "Single-Potential Problem" Simplification

The paper's methodology *chooses* to construct a benchmark for a simplified theoretical problem.

* The general, two-potential SB problem requires solving for *both* potentials $\phi(x_0)$ and $\psi(x_1)$ to satisfy the two marginal constraints $\pi_0 = p_0$ and $\pi_1 = p_1$.
* The paper's benchmark, the only unknown potential is for the output domain; the input domain's potential is then determined. (Note that the potential $\phi$ on the input space is determined by the potential $v ^*$ on the output domain and the reference transition kernel $q^{ref}(x_1 | x_0)$.)
* This theoretical simplification is crucial because the paper's "winning" solver, **DLightSB**, is *explicitly* a static, dual-problem solver designed to solve this very particular problem. 



### 4. Theoretical Limitation of the Chosen Method

The paper's actual innovation is not Theorem 3.1, but the methodology to make it computationally tractable. The theoretical constructions above are trivial in low dimensions, but in the paper's high-dimensional setting ($\mathcal{X} = \mathbb{S}^D$), the potential $v ^*$ and the normalization constants are intractable, requiring sums over $S^D$ states. 

The paper's *actual* contribution is the application of a Canonical Polyadic (CP) decomposition to parameterize $v^*$. As shown in **Proposition 3.1**, this factorization cleverly turns the intractable normalization sum into a product of sums, making the computation feasible. This is a practical, **engineering contribution**, not a new theory for constructing benchmarks.

This reliance on a CP parameterization for tractability imposes a severe *theoretical limitation* on the benchmark itself: it guarantees the ground-truth problem is trivially simple.

* The computational tractability of the CP method is only achieved if the rank $K$ is small.
* The authors, in their experiments, use **$K=4$** to generate the ground-truth data.
* This means the "challenging" benchmark problem is, by construction, an *extremely simple, low-rank function*.
* This defeats the entire purpose of a benchmark, which is to test a method's ability to solve a complex problem. The paper's theoretical framework *requires* the problem to be simple for the ground truth to be computable.


### 5. Fundamental Methodological Bias and "Apples-to-Oranges" Evaluation
The paper's central weakness is a circular and biased evaluation methodology. The primary conclusion—that DLightSB and DLightSB-M are the superior solvers—is an artifact of the benchmark's design rather than a demonstration of general algorithmic strength.

* **Architectural Circularity:** The tractability of the benchmark (Prop. 3.1) relies entirely on parameterizing the scalar function $v^*$ using a **CP decomposition**. The authors' "new" solvers, DLightSB and DLightSB-M, are *also* explicitly parameterized using the **exact same CP decomposition**. This creates an evaluation that is not a fair comparison of general SB solvers. Instead, it tests how well a CP-based model can fit a function already known to have a CP structure. The authors explicitly acknowledge this circularity: "We attribute this to the benchmark pairs being built on the same principle used by the DLightSB solver".

* **"Apples-to-Oranges" Comparison:** The benchmark problem itself represents a *simplified* version of the full SB problem. The authors' "generative" construction (Theorem 3.1) fixes $p_0$ and the reference dynamics, which effectively reduces the problem to solving for a *single* static potential, $v^*$. The paper then compares two fundamentally different classes of algorithms:
    * **DLightSB (The "Apple"):** A **static, dual-problem solver** whose entire purpose is to fit this single potential $v_\theta$.
    * **CSBM (The "Orange"):** A **dynamic, primal-problem solver** based on Iterative Markovian Fitting (D-IMF), which is designed to learn the *full time-dependent transition probabilities* $m(x_{t_n} | x_{t_{n-1}})$.

This evaluation is inherently unfair. A specialized static solver (DLightSB) that is custom-built for this simplified problem (and given the correct architecture) is predictably going to outperform a general-purpose dynamic solver (CSBM) that is designed for a much harder task.

### 6. Triviality of the Benchmark Task
The reliance on the CP parameterization for tractability forces the ground-truth benchmark to be exceptionally simple, defeating the purpose of a benchmark, which should test a challenging problem.

* The ground-truth benchmark is generated using a CP decomposition with a rank of **$K=4$**.
* The authors' "winning" solvers, DLightSB and DLightSB-M, are then initialized with a capacity of **$K=1000$** components.

The task is thus to fit a simple $K=4$ mixture model using a massively over-parameterized ($K=1000$) model that shares the *identical* functional form. The 250-fold capacity advantage, combined with the correct inductive bias, makes the problem trivial for the DLightSB solvers. A benchmark that is this simple and architecturally-biased provides little insight into how these solvers would perform on complex, real-world problems.

### 7. Flawed Evaluation Metrics and Exclusion of Relevant Competitors
The paper's evaluation protocol is indirect and avoids the most rigorous tests of its own contributions by excluding the most relevant class of competing algorithms.

* **Exclusion of Neural EOT Solvers:** The paper's "Remark" dismisses all "tabular EOT" solvers as "non-generative." This is a strawman argument that ignores the modern, active field of research that *does* learn generalizable, generative couplings by parameterizing the dual potentials ($\phi, \psi$) with neural networks. A fair and informative benchmark would have *required* a comparison between **DLightSB (CP-potential)** and a **Neural EOT Solver (MLP-potential)**. This "apples-to-apples" comparison would have directly tested the paper's only true innovation: the claim that a CP decomposition is a good inductive bias for this problem.

* **Inadequate "Black-Box" Metrics:** By excluding other potential-fitting methods, the paper is forced into the "apples-to-oranges" (DLightSB vs. CSBM) comparison. Because these models learn different objects (potentials vs. dynamics), a direct comparison is impossible. The authors are thus forced to use weak, indirect metrics: **Shape Score** and **Trend Score**. These metrics are inadequate for two reasons:
    1.  **They Don't Measure the *Real* Ground Truth:** The *entire point* of the benchmark is that the true potential $v*$ is known. The most rigorous metric would be a direct comparison of the learned potential $v_\theta$ to $v*$. The paper avoids this.

    2.  **They Don't Guarantee Optimality:** These metrics only measure low-order statistics of the *output samples*. They confirm that the model can approximate the *final marginal* $p_1$, but they provide no guarantee that the model has learned the *true, optimal transport coupling* $q ^*$.

### Questions
#### **On the Benchmark Design and Evaluation Bias**

1.  The paper's primary "winning" solver, DLightSB, is based on a Canonical Polyadic (CP) decomposition. The benchmark's ground truth is *also* constructed using a CP decomposition (Prop 3.1).  How can we be sure that the superior results of DLightSB are not an artifact of it being given the correct inductive bias, a bias that competing methods like CSBM (using an MLP) do not have?

2.  The paper's results show that DLightSB "wins" on a benchmark built on "the same principle used by the DLightSB solver." Does this not confirm a circular evaluation? How would DLightSB perform on a benchmark constructed with a *different* non-factorizable structure, such as one based on a deep neural potential?

3.  The ground-truth benchmark is generated with a low rank ($K=4$), while the DLightSB solver is given a capacity of $K=1000$. Does this 250-fold capacity advantage not make the task trivial for a solver that already shares the problem's functional form? What insights can be gained from a benchmark that is this simple and architecturally-matched to the solver?

#### **On the Choice of Methods and Comparisons**

4.  The remark in Section 2.4 dismisses "tabular EOT" solvers as non-generative. This seems to overlook the entire class of modern EOT solvers that parameterize the dual potentials ($\phi, \psi$) with neural networks, which are both generative and generalizable. Why were these potential-fitting neural EOT methods—which would provide a true "apples-to-apples" comparison against DLightSB—not included as a baseline?

5.  The evaluation compares DLightSB (a *static, dual-potential solver*) against CSBM (a *dynamic, primal-path solver*). Is this not a "apples-to-oranges" comparison? DLightSB solves a simpler, static problem, while CSBM solves a much harder, dynamic one. How can we conclude DLightSB is a better "Schrödinger Bridge solver" from this comparison?

#### **On the Choice of Evaluation Metrics**

6.  The entire purpose of the benchmark is that the ground-truth potential $v^*$ is known. Why, then, were the evaluation metrics (Shape Score, Trend Score) based on comparing *output samples*? Why did the authors not use the most direct and rigorous metric: the error between the learned potential $v_\theta$ and the true potential $v*$?

#### **On the Theoretical Contributions and Framing**

7.  Theorem 3.1 appears to be a restatement of the well-known dual solution to the EOT problem, where one potential is set to 1 by construction. Can the authors clarify the theoretical novelty of this theorem beyond this restatement?

8.  The paper's main innovation appears to be the *engineering* choice of using a CP decomposition to make the benchmark tractable. However, this choice seems to *require* the ground-truth problem to be low-rank ($K=4$). Does this not create a fundamental trade-off where the benchmark can be either (a) computable but trivial, or (b) complex but not computable?

10. The paper's "winning" solvers (DLightSB, DLightSB-M) are admitted to have "severe memory constraints" and become "prohibitive for high-dimensional data." This seems to create a contradictory message. What is the value of a solver that only "wins" on a simple, biased benchmark and is not scalable to the very high-dimensional problems it claims to address?

### Soundness
2

### Presentation
3

### Contribution
1
