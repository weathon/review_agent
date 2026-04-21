# Data Selection via Optimal Control for Language Models

- Avg Score: 8.00
- Decision: Accept (Oral)
- Scores: 10, 6, 8, 8, 8

## Abstract
This work investigates the selection of high-quality pre-training data from massive corpora to enhance LMs' capabilities for downstream usage. 
We formulate data selection as a generalized Optimal Control problem, which can be solved theoretically by Pontryagin's Maximum Principle (PMP), yielding a set of necessary conditions that characterize the relationship between optimal data selection and LM training dynamics.
Based on these theoretical results, we introduce **P**MP-based **D**ata **S**election (**PDS**), a framework that approximates optimal data selection by solving the PMP conditions. 
In our experiments, we adopt PDS to select data from CommmonCrawl and show that the PDS-selected corpus accelerates the learning of LMs and constantly boosts their performance on a wide range of downstream tasks across various model sizes.
Moreover, the benefits of PDS extend to ~400B models trained on ~10T tokens, as evidenced by the extrapolation of the test loss curves according to the Scaling Laws.
PDS also improves data utilization when the pre-training data is limited, by reducing the data demand by 1.8 times, which helps mitigate the quick exhaustion of available web-crawled corpora. Our code, model, and data can be found at https://github.com/microsoft/LMOps/tree/main/data_selection.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
2

### Summary
This paper presents a formulation of turning the data selection to the AUC optimization problem. More importantly, this paper derives the optimal solution for the data selection problem and proposes an efficient implementation of the proposed method. The proposed methods are empirically verified in diverse settings.

### Strengths
This work is sufficiently solid. It covers the fundamental problem formulation, which introduces a new optimization problem. Also, it provides the theoretical optimization solution to the proposed problem, which has not been studied before. From this perspective, this paper has made sufficiently novel contributions.  Moreover, this work designs an efficient implementation to avoid additional computation overhead. These methods are empirically verified with different experiments setting, demonstrating the effectiveness of this method across different settings. Due to its theoretical contributions and solid experiments, I give a clear accept (score 10) to this paper.

### Weaknesses
I didn't identify any major weaknesses of this paper.

### Questions
It is really interesting to see the formulation of optimization problem (3) in the data selection problem. Has this method been used in other tasks such as computer vision, diffusion model, or simple MNIST dataset classification? It seems to be a very general approach that can be used in many senarios. 

In Line 60, the PMP derives the necessary condition for optimal data selection. Does it mean that it is not sufficient to guarantee the data selection is optimal? 

It seems that the PDS method always achieves better performance. Is there any negative results where the PDS method fails? Do we have any senarios where we should avoid using this method for the data selection?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes a novel method of data selection for training machine learning models, or LLMs in particular.
It starts by theoretically formulating data selection as an optimal control problem, 
where the weights of training samples are optimized in order to minimize the total area under the downstream loss curve during training with gradient descent on the weighted data.
Based on this, a practical and efficient implementation is developed: 
first solve the optimal control problem on a small proxy dataset (approximately, via bi-level optimization),
and then train a data scorer (by fitting the obtained scores from the first step),
which can then be applied for training any LLM with other datasets.
Experiments show improvements over prior data selection algorithms,
in terms of test losses and scores on some commonly used benchmarks.

### Strengths
- The formulation of data selection as optimal control, and the theoretical connection between two seemingly unrelated fields, are interesting.

- The presentation is good overall. Ideas are clearly explained.

### Weaknesses
- According to the theory of this work, in particular Eq. (6), the optimal value of $\gamma$ (the weights for training samples) is an one-hot vector, 
which leads to the surprising conclusion that training an LLM with a single sample is theoretically optimal.
The authors also mention this in Line 235. 
My major concern is that this seems to reveal a fundamental caveat of the proposed methodology.
Also, with many approximations in deriving the practical implementation (Section 2.3), 
it becomes less and less clear how much the empirical advantages of the resulting algorithm are actually relevant to the theory in Sections 2.1 and 2.2.


- Using the "scaling law" for extrapolating empirical results with small-scale models to 175B or 405B models (Table 3) doesn't feel like science. 
It is not obvious why scaling laws for conventionally trained LLMs should be applicable to PDS-trained LLMs.
In addition, the goodness of fit for calculating the scaling law's coefficients is not reported, which further makes the extrapolation less convincing.


- Regarding the empirical results, the proposed method is compared with prior data selection algorithms only in Tables 1 and 2, for 160M and 470M models that are relatively small-scale.
Moreover, many scores in these results are close to random choice (e.g., 25% accuracy for MMLU in Table 2), 
hence it is not clear what the minor differences between the scores of different methods really imply.

### Questions
- Two questions about the theory:
(a) Could you provide some intuition about why theoretically a single training sample is optimal, according to Eq. (6)? 
What's missing in this theoretical analysis?
(b) In Eq. (22) where the method of Lagrange multiplier is applied, it seems that it misses a $\max_{\mu}$ term. Is this correct? If so, does the remaining analysis still hold, or need corrections as well?
I'm willing to raise my rating if these two questions can be resolved.


- Line 883, "B" and "\beta" seems like typos; should be "C" and "c" respectively?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper suggests a method for data selection based on applying optimal control theory to compute the importance of individual data points. Crucially, this importance is computed in the context of end-to-end training, to optimize the AUC of downstream performance - this is in contrast to previous works that don’t look at training trajectory. Several optimizations are used to approximate the PMP solution. Moreover, the data selection is done offline so there is just a constant cost.

### Strengths
- Extensive ablations (optimizing Algorithm 1, choice of downstream task) are performed to support the empirical claims and many of the choices made.
- The tackled problem is very relevant and the approach is innovative, using optimal control to account for the complete training trajectory.
- The empirical results are very promising on the small-to-medium scale experiments that they are run on. A scaling law is also computed to try to support generalization to larger models.

### Weaknesses
- It would be good if Appendix G3, choice of $J(\theta)$ was covered in the main body (regarding the influence of the downstream data on performance) since that is an important driver of the performance.
- Better explain Figure 1: in (b), do you have the same amount of data being used by your method and Redpajama’s to train? Do all points correspond to the same model size in (a)? If not, how do you vary these?
- It would be useful to mention at line 89 that J is not computed on the same downstream tasks that you measure one (as you mentioned at line 307) since this distinction is crucial and not made clear until line 307.
- It seems like line 100, defining $\gamma$’s makes an implicit assumption about points having a notion of independent importance, whereas one could envision cases where a point’s importance is contingent on other points being in the dataset. It would be useful to add a discussion on this - as stated, it looks like this is a fact, not a choice.
- It would be useful to add a discussion on why it makes sense to need Gumbel sampling - after all, one could expect $\gamma$’s to already have accounted for the need of diversity.

### Questions
- Why did you decide at line 259 to use the average of the output hidden states rather than the last one? Intuitively this gives more weight to how the datapoint starts (since the first activations at the first few positions affect both the hidden states at these positions as well as later on).
- Is it computationally feasible to constrain the $\ell_0$-norm of $\gamma$? If so, do you expect to get the same data points as you would if you took the biggest ones by the current $\gamma$?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a data selection method (called PDS) for language model pretraining based on optimal control principles. Specifically, the PDS method aims to compute a data score for re-weighting data, such that the trained model (via a fixed procedure) achieves the best downstream test error. In practice, this is done using small-scale proxies. Additionally, a model is fitted to predict scores for general data. One important benefit of the proposed method is its offline nature, enabling an effortless plug-in to existing language model training pipelines. The effectiveness of the proposed method is verified across multiple benchmarks.

### Strengths
- The paper is well-written, and the experimental setup is clearly explained.
- The PDS method is well-motivated by control theory and demonstrated to be effective in practice.
- The offline nature of the proposed method makes it very easy to use in existing language model training pipeline.  
- Multiple thoughtful ideas are proposed to accelerate the implementation of PDS.

### Weaknesses
I didn’t observe significant weaknesses in the work. If I had to mention one, it would be the lack of large-scale verification of the method's effectiveness. This limits the strength of some claims (e.g., fitting scaling laws and extrapolating results). However, this is understandable given the substantial computational resources required for such large-scale experiments.

Overall, I think the paper demonstrates the promise of the proposed method, and I am inclined to support its acceptance.

### Questions
Theorem 2.1 seems related to the KKT conditions of the optimization problem in equation (3). Could you briefly discuss the connection between Theorem 2.1 and the KKT conditions? Of course, this is not a weakness of the paper.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces a new data selection methodology inspired by control theory. The authors design a control problem that corresponds to training dynamics of a data-reweighted objective where the control action is the reweighting, the dynamics are gradient descent or Adam dynamics on the parameters, and the objective is minimal training error of the trajectory induced by such dynamics.  This is similar to some works in the learned optimizer literature but the optimizer parameters being optimized correspond to data selection.  

On a small proxy LM, the algorithm applies a fixed-point iteration (which is shown to be projected gradient descent) in order to induce Pontryagin's Maximum Principle for this control problem and to produce a dataset of quality scores (example weights).  A model is then learned to label examples, which can be used in a data filtering process for a much larger training run.  The authors evaluate this methodology on 160M to 1.7B parameter models with some initial promising results in comparison to baseline data selection approaches.

### Strengths
As far as i can tell, this approach is novel, in that similar methods have not been applied to the data selection setting. This is a theoretically motivated approach that attempts to optimize the training trajectory directly.  The idea to learn a labeler to translate the ideas from leaned optimizers into a data filtering mechanism is very clever.

The paper is very well written in my opinion--the general algorithm pipeline is quite complicated, but the main technical/algorithmic ideas were well motivated and clear.

The algorithm seems to perform well against baselines, although the compute requirements are likely higher (though maybe not prohibitively so as can be seen in Table 4).

### Weaknesses
I am very skeptical of the scaling laws presented in this work.  There does not appear to be enough data to fit such a complicated model, and I do not think these results should be included nor trusted.

While the empirics seem to be better than baselines, there seems to be fairly marginal improvements across the board and without very careful hyperparameter sweeps on the downstream optimizer, there is reason to be skeptical of the empirics.

### Questions
Seems A single step of the inner loop gets most of the gains, why do you think this is?  Given this is much cheaper is this a more feasible algorithm.

Does the proxy model and the scorer models complexity need to scale as we scale up the final pretrained model we want to use this with?

Line 883, what is B and what is beta? 

Typo:
Line 68: "(e.g., 125M paramaters)"

### Soundness
3

### Presentation
4

### Contribution
3
