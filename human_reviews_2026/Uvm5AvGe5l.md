# Uncovering Challenges of Solving the Continuous Gromov-Wasserstein Problem

- Decision: Reject
- Scores: 8, 4, 4, 2

## Abstract
Recently, the Gromov-Wasserstein Optimal Transport (GWOT) problem has attracted the special attention of the ML community. In this problem, given two distributions supported on two (possibly different) spaces, one has to find the most isometric map between them. In the discrete variant of GWOT, the task is to learn an assignment between given discrete sets of points. In the more advanced continuous formulation, one aims at recovering a parametric mapping between unknown continuous distributions based on i.i.d. samples derived from them. The clear geometrical intuition behind the GWOT makes it a natural choice for several practical use cases, giving rise to a number of proposed solvers. Some of them claim to solve the continuous version of the problem. At the same time, GWOT is notoriously hard, both theoretically and numerically. Moreover, all existing continuous GWOT solvers still heavily rely on discrete techniques. Natural questions arise: to what extent existing methods unravel GWOT problem, what difficulties they encounter, and under which conditions they are successful. Our benchmark paper is an attempt to answer these questions. We specifically focus on the continuous GWOT as the most interesting and debatable setup. We crash-test existing continuous GWOT approaches on different scenarios, carefully record and analyze the obtained results, and identify issues. Our findings experimentally testify that the scientific community is still missing a reliable continuous GWOT solver, which necessitates further research efforts. As the first step in this direction, we propose a new continuous GWOT method which does not rely on discrete techniques and partially solves some of the problems of the competitors.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper the authors investigate the challenges of solving continuous
Gromov-Wasserstein (GW) problems from discrete high dimensional data. The
authors first do a quick review of the existing literature on GW problems and
highlight the different strategies using discrete+regression or Neural networks
to solve them. They discuss their limits, in particular the challenge of finding
a good GW mapping when there is no one-to-one correspondances in the data
(denoted as uncorrelated data splitting) and devise a benchmark to compare the
methods on words embeddings with bilingual vocabularies (which provides them
with a ground truth mapping). The benchmark show teh limits of the existing
methods whose accuracy greatly decreases with alpha the proportion of
"correlated" data. Then they propose an Neural GW solver that requires to solve
a minimax optimization problem. They show that their method outperforms the
existing ones on their benchmark especially for low "correlation"  and large
scale dataset when some competitors are failing.

### Strengths
+ This paper addresses an important and challenging problem of solving continuous
  Gromov-Wasserstein problems from high dimensional data. 
+ The paper is well written and easy to follow, the positioning in the literature
  is clear and the proposed benchmark is well motivated.
+ The question of "correlated" vs "uncorrelated" data is very relevant in
  practice and the benchmark proposed is a good contribution to the community.
+ The proposed method is novel and shows interesting empirical results on the
  benchmark.
+ The benchmark illustrates well the limits of existing and proposed methods with
  no clear winner in all settings which is a good sign of a well designed and
  honest benchmark.
+ I really appreciated the clarity of the writing and the quality of the
  scientific steps followed in the paper. Asking a question and experimenting on
  it followed by a reasonable contribution is refreshing compared to many papers
  that just throw a new method at the wall and see if it sticks.

### Weaknesses
Note that these weaknesses are minor and do not impact my overall positive
opinion of the paper.

+ The choice of the word "correlation" to denote overlapping of the support of
  the true aligned samples is not well chosen since it is universally used in
  statistics for something different. Perhaps "overlap ratio" would be a better 
  wording.
+ While the benchmark is interesting, it is limited to word embeddings and
  bilingual vocabularies. It would be interesting to see how the methods
  perform on other types of data and tasks, possibly simulated with known
  solution.
+ The choices of performance measures in Fig 4 could be better. All Top-k
  accuracy basically look the same whereas other "marginal quality" measures
  are provided in the appendix. It would be better to have a more diverse set of
  measures in the main text.
+ The comparison in the large scale setting to NeuralGW is a little unfair since
  other solvers rely on subsampling (minibatches) that are known to be biased wrt
  the full GW solution. Existing reference on minibatch OT show that there is a
  bias induced by minibatching that could be de-biased (see e.g. "Minibatch
  optimal transport distances; analysis and applications").
+ The proposed NeuralGW woks clearly better than existing methods in the low
  correlation setting but is is actually not as good as competitors trained on a
  much smaller sample size when the correlation is high (alpha=0.9). It is
  necessary to discuss this point a bit more in the paper.

### Questions
+ Could you please address the weaknesses mentioned above?

+ The Topk accuracy measures seem to nearly constant wrt K. It means that only
  the closest points are (relatively) well mapped and the others are not? Could
  you please comment on that?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on establishing a benchmark for neural Gromov–Wasserstein (GW) problems. The authors identify key existing works addressing this task and propose a methodology for benchmarking these methods.
The main observation is that most neural GW papers operate in a correlated regime, which tends to favor their reported performance, and are (naturally) influenced by the batch size that can be processed. The proposed benchmark explicitly varies the level of correlation and evaluates the robustness of different methods under these conditions.
In addition to this benchmarking framework, the authors introduce a new neural GW approach in the inner-product setting, derived from a duality theorem. This method is designed to be more robust to correlation effects.

### Strengths
Overall, I find the paper well written. It is dense, but the problem is clearly formulated. The related work section is also well presented: the methods in Section 3 are described concisely yet with enough detail to understand their scope and relevance.
Identifying the correlated dataset setting is, in itself, an interesting and valuable contribution.
I find the experimental results in Section 4.2 particularly compelling, as they demonstrate that the presence of a natural pairing or correlation plays a crucial role in achieving good performance for neural GW solvers.

### Weaknesses
Overall impression:

Although the contributions are interesting, I find that several aspects of the paper remain unclear. The main message is somewhat blurred by a set of experiments that sometimes appear contradictory. At first, the paper suggests that the main challenge for neural GW methods lies in the level of correlation in the data; later, however, the issue seems to shift toward their inability to handle large-scale problems.
This mixture of factors within the experiments makes the overall conclusion difficult to interpret, leaving the reader somewhat frustrated by the lack of a clear and unified takeaway message.

Unclear points:

- About "Limitations of existing methods"

One of the main contributions of the paper is to show that, in the correlated regime, existing neural GW methods tend to be favored. More precisely, the correlated setting corresponds to random variables $(X, Y) \sim (i_d \times \sigma) \cdot \pi$, where $\sigma$ is a permutation and $\pi$ a coupling.

However, the train/test procedure related to this definition is not entirely clear. From what I understand, the “correlated” regime essentially corresponds to the “paired” regime — that is, a setting where there exists a natural alignment or pairing between samples.
To “break” this pairing, one can simply rematch the data points, which is indeed what the proposed procedure does. Yet, such manipulation is only feasible on synthetic datasets or on data where the natural pairing is explicitly known (e.g., word-pair datasets in embedding problems). I think it would be worth emphasizing that this procedure can only be applied in these specific cases.

- About AlignGW:

It is stated that “we train a Multi-Layer Perceptron on the barycentric mapping,” but the barycentric mapping can in principle be defined in both directions (source to target or target to source). Which direction is used here, and why was this specific choice made?

- A propos des resultats 4.2:

(a) Why were not all baselines included in Figure 3?
Including all methods would help make the comparison more comprehensive.

(b) The description of the matching metrics is somewhat confusing or hard to follow. What exactly is meant by the reference pool and reference space? A bit more formalism here would help clarify what these objects represent. Similarly, the notion of “optimal pair” could be defined more precisely.
The same applies to the marginal metrics: it seems they refer to divergences between $\mathbb{Q}$ and $T(\mathbb{P})$, but an explicit equation would make this clearer.

- About section 5.2:

The main remark here is that Figure 4 is very difficult to read. The results are not visually clear, and the curves are hard to distinguish.
Where are regGW and FlowGW in the first subfigures? Who are the “other methods”?
This is unfortunate, as this figure seems central to the paper’s conclusions, but its current presentation makes it difficult to interpret.


- About the proposed method:

It is not clear to me why the proposed method should be more robust to correlation levels. Do the authors have any intuition on this ?

- Other remarks:

The citation style is somewhat unconventional and could be standardized to match common academic formatting.
Also consider citing [1] a very recent paper on the subject.

[1] Unsupervised Learning for Optimal Transport plan prediction between unbalanced graphs, Sonia Mazelet, Rémi Flamary, Bertrand Thirion, NeurIPS, 2025.

### Questions
see above

### Soundness
2

### Presentation
2

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
This paper has a twofold purpose: to provide a benchmark for Gromov–Wasserstein (GW) solvers, and to propose a new continuous Monge–GW method that does not rely on discrete techniques. Regarding the first goal, the authors report that existing solvers perform well when correlated data is available but fail in the absence of correlation. To address this limitation, they introduce a new methodology (NeuralGW), which is designed to handle fully uncorrelated setups and is claimed to be practically useful in more realistic and challenging scenarios. However, this new solver requires large amounts of training data, and other remaining challenges leave the door open for further research.

### Strengths
This paper highlights the limitations of existing GW solvers: in the interplay between discrete and continuous approaches, current methods are said to rely more heavily on the empirical (discrete) side rather than the continuous perspective, which, in practice, should treat data as i.i.d. samples.

### Weaknesses
- In general, the presentation is not clear, making it difficult to fully understand the core problem.

- Five existing GW methods are listed in Section 3, but only three of them are analyzed in Section 4.2.

- Although the authors reference the seminal paper by Dumont et al. on the existence of Monge-GW assignments, the distinction between finding an optimal coupling/plan and finding an optimal map (i.e., the Kantorovich vs. Monge formulations) is blurred.

- The quality of Figure 2 is poor; it appears blurry.

- The beginning of Section 4.1 (lines 222–246) is unclear.

- The proposed solver appears to have several drawbacks.

### Questions
- I suggest formalizing what is meant by a “continuous setup.”

- In the classical OT problem, the Kantorovich formulation turns the optimization into a linear program, making it more tractable than the Monge formulation. Could the authors elaborate on how this comparison translates to the GW setting?

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
The Gromov-Wasserstein distance is a metric commonly used in practice to compare two different metric measure spaces due to its nice geometric interpretation of computing the most isometric map between the spaces. While nice for its mathematical properties, the Gromov-Wasserstein distance is quite challenging to compute. Existing methods for computing the Gromov-Wasserstein distance assume discrete metric measure spaces, and therefore fail to capture the underlying maps between continuous distributions that samples come from. To extend the Gromov-Wasserstein distance to the continuous setting, algorithms then rely on a correlated sample between the two distributions.

The authors of this work experimentally suggest that existing methods for computing Gromov-Wasserstein distances between correlated samples, i.e. samples where the optimal pairs have some underlying association in the metric measure spaces, fail to extend to the continuous Gromov-Wasserstein distance when discrete samples are taken i.i.d. from the two distributions. They then design a neural network which computes continuous Gromov-Wasserstein distances without relying on discrete instances like in prior works and empirically verify that their neural network outperforms existing methods for computing continuous Gromov-Wasserstein distances.

### Strengths
The introduction to the Gromov-Wasserstein problem was pretty well written and easy to follow.

The overview of existing algorithms for continuous Gromov-Wasserstein distance in section 3 was very useful and also well written.

 The authors clearly describe a problem with existing algorithms for the Gromov-Wasserstein distance and design a neural network which outperforms existing algorithms.

### Weaknesses
The conclusion "existing algorithms therefore don't work for uncorrelated data" from Section 4 cannot be made as a general statement. In particular, their experiments were conducted on only two data sets with very similar types of data (word to vector embeddings) and no proof is provided to justify the strong claims they make about existing work. To my understanding, provable guarantees of existing algorithms for Gromov-Wasserstein distance are lacking, i.e. existing algorithms serve as merely heuristics already. Why should one expect that the results  from these two data sets to extend to other data sets on Euclidean spaces? Why could it not be an issue with the structure of the word embedding data tested?
    
The authors note on line 464 that the performance of their neural network is "inconsistent with respect to initialization parameters" and "sees high standard deviation among repetitions".

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
2
