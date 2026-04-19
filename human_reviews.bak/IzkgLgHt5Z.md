# Optimization and Generalizability: Fair Benchmarking for Stochastic Algorithms

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 5, 3, 3

## Abstract
We currently lack full understanding of what makes a good optimizer for deep learning and whether improved optimization performance confers higher generalizability. Current literature neglects an important innate characteristic of SGD and variants, their stochasticity, failing to fairly benchmark these algorithms and reveal their performance in the statistical sense. This paper fills this gap. Unlike existing work which evaluates the end point of one navigation/optimization trajectory, we utilize and sample from the ensemble of many optimization trajectories, so that we can estimate the stationary distribution of a stochastic optimizer. We cast a wide net and include SGD and noise-enabled variants, flat-minima optimizers, and novel algorithms we debut in this paper by recasting and broadening the SGD algorithm under the Basin Hopping framework. Our evaluation considers both synthetic functions with known global and local minima of varying flatness and real-world problems in computer vision and natural language processing. Fair benchmarking accounts for the statistical setting, comparing stationary distributions and establishing statistical significance. Our paper reveals several findings on the relationship between training loss and hold-out accuracy, the comparable performance of SGD, noise-enabled variants, and novel optimizers based on the BH framework; indeed, these algorithms match the performance of flat-minima optimizers such as SAM with half the gradient evaluations. We hope that this work will support further research in deep learning optimization relying not on single models but instead accounting for the stochasticity of optimizers.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper addresses the inherent stochastic nature of Stochastic Gradient Descent (SGD), its noise-enabled variants, and flat-minima optimizers. To understand these algorithms more deeply, there's a need to broaden the scope of noise-enabled SGD within the Basin Hopping framework. A central understanding is that during neural network training, the trajectory of optimization can be intricate, requiring a more comprehensive perspective than merely focusing on the converged or lowest-loss states.

The study introduces several novel algorithms, utilizing synthetic landscapes to rigorously evaluate the stationary distributions of various optimizers. Crucial findings reveal relationships between training loss, hold-out accuracy, and the performance of different optimizers, with some algorithms even matching the efficiency of flat-minima optimizers with half the gradient evaluations.

### Strengths
The paper is very well-written and easy to follow. This topic is of high interest to the community: given the ubiquity of SGD in daily machine learning practice, better understanding the intrinsic stochasticity of SGD in various loss landscapes is crucial.

The experiments carried out are both on synthetic data to gain intuition and on real-world data.

The context is correctly set and previous work duly cited.

### Weaknesses
On the content itself, I have nothing much to say. I think the research methodology is sound.

A minor details is the algorithm formatting in LaTex which could be enhanced. The instructions are colliding with the frame borders with almost no padding. This should be fixed for a nicer presentation.

### Questions
No question on my side.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new method to benchmark the performance of an algorithm in terms of optimization and generalization. Instead of looking at the performance of a single trained model in the end, it proposes to form one group of models with lowest train loss and another group of models with best generalization performance by sampling from one or more trajectories. Then it can test whether the two groups come from the same population using various statistics such as t-test.  Besides this, it also comes up with a few new algorithms based on the Basin Hopping framework, and compare their performance with SGD, noise injected SGD (either through model or gradients), and sharpness-aware minimization (SAM) on synthetic and real-world datasets.

### Strengths
- Despite some typos, the paper can be easily understood. The related work section is very well-written, various algorithms and their connections are concisely summarized.

- The experiments on the synthetic dataset are interesting and reveal some useful information such as the distribution of SAM is skewed towards flatter minima.

### Weaknesses
- The proposed method may not be practical. First, if several models are sampled from one trajectory, they could be correlated, and the resulting statistical test may not be so useful. For example, with a bad initialization, the models sampled from the trajectory may not represent the population, which can lead to misinterprete the actual performance of the algorithm. Second, if multiple trajectories are required, this could increase the computation cost. Moreover, depending on the goal, one best model can be sufficient and a group of models is certainly not required.

- The experiments are not comprehensive. No large-scale datasets such as Imagenet are considered, and the neural network architecture is also limited.

### Questions
N/A

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors develop a collection of benchmarks, both with synthetic and real problems, and use them to compare SGD, SAM, and other noisy variants thereof, including:

1. SGD
2. Noise-in-gradient SGD
3. Noise-in-model SGD
4. SAM
5. Noise-in-gradient Basin Hopping
6. Noise-in-model Basin Hopping
7. Noise-in-gradient Metropolis Basin Hopping
8. Noise-in-model Metropolis Basin Hopping

These methods are each summarised before (and after) the benchmarks are described. Three synthetic objectives are considered: (I) Himmelblau, (II) Three Hump Camel, and (III) Six Hump Camel. Performance on these objectives is judged in terms of the stationary distribution: terminal values of the optimisation algorithms are binned into each region of the landscape. Noise-in-gradient gradient descent and SAM are observed to skew solutions towards flatter minima (as expected). All algorithms exhibit a lot of noise in the six hump camel example. Next, generalisation performance is examined for four real-world problems: CIFAR10, CIFAR100 (image datasets), and GoEmotion, TweetEval (NLP datasets). Mann-Whitney U tests are performed to test whether significant differences can be detected between generalisation and optimization metrics (they cannot). Median accuracies and standard deviations are reported across each dataset with each optimiser. Hypothesis tests are also performed to compare algorithms to each other to see if significant differences in accuracies can be detected (they cannot). Finally, learning curves are displayed, highlighting that SAM requires double the gradient evaluations per step.

### Strengths
- Tackles an important problem: appropriately benchmarking algorithms in a systematic way to inform practitioners of real benefits of one algorithm over another (or lack thereof), rather than testing on individual problems.
- Considers generalisation performance rather than simply rate of convergence of the optimiser (there are too many papers that don't do this).
- A neat collection of synthetic examples are considered.
- Some variety in real-world problems provided (i.e. 2 image datasets, 2 NLP datasets).
- Lots of different examinations.

### Weaknesses
- "Rigorous" comparison is far too limited in scope; unambitious. What types of problems, not just landscapes, do certain algorithms perform well on? Consider images vs NLP vs reinforcement learning vs SciML, etc. You need many more real-world examples for such a general analysis to be valuable.
- Generally poorly written, with numerous grammatical issues, inconsistent capitalization (e.g. CIFAR vs Cifar).
- Odd structure; why is Section 4 after Section 3, when Section 3 and Section 5 link with each other?
- Unpleasant presentation: algorithm environments with clashing borders, weird line spacing, figures and tables have text that is too small, weirdness with algorithm text that goes over multiple lines, incorrect citing (citet vs citep).
- Algorithms in the comparison are oddly chosen, and do not comprise a sufficient selection of what is used. Where is Adam for example?
- Hypothesis tests are a bad choice here. 
- Multiple hypothesis tests are performed without p-value correction, leading to p < 0.05 conclusions about 5% of the time *due to random chance*.

### Questions
- Does GD use the normalised gradient vector, or just the gradient vector?
- Do any of the synthetic examples in Figure 1 correspond to parts of a real loss landscape? 
- What is the purpose of the p-values in Section 6.2? Is this to show that there are no meaningful differences in performance between optimisers? Unless realisations are appropriately coupled together (i.e. common random numbers), I'm not sure this is a sensible conclusion. There is naturally a lot of variance in the optimisation procedure, but one optimiser may still perform consistently better than another, everything else equal. 
- Can you present the stationary distributions in terms of bar charts rather than as a table? It's really difficult to read as is.
- How long is each optimisation algorithm run for? 
- Wouldn't correlation tests be a better idea to test for generalisation vs optimisation section? It is obvious that there should be no significant differences here.
- The differentiation between SetA and SetB seems odd to me. Why not just report all the metrics?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper consider SGD, NoiseInModel-GD/SGD, NoiseInGradient-GD/SGD and SAM (Sharpness Aware Minimization) algorithms in the experiments and varying over BH, MonotononicBH, and MetropolisBH. 
The authors claimed that they propose a population-based approach to benchmark the algorithms and to better understand the relationship between optimization and generalization. 
They argued that to characterize for the behavior of an optimizer one needs a nonlocal view that extends over several trajectories and goes beyond the ”converged”/lowest-loss model.
Thus, they conducted experiment on several trajectories of the algorithms for three synthetic problems and then real world problems.

### Strengths
This paper experiments with SGD algorithm and its noise-enabled variants under the Basin Hopping framework. They propose a new procedure for benchmarking the performance of the algorithms: by considering the trajectories created by the algorithms (called populations of models) and comparing their statistical properties. They argue that these trajectories have low loss function and by comparing "populations of models", they can fairly compare two optimizers and avoid conclusions based on one arbitrary or hand-selected model by any optimizer.

### Weaknesses
The weaknesses are: 
- While collecting additional information from the trajectories of the algorithm is a helpful way to assess the performance better, it also leads to computational cost. The authors should compare this approach to the standard procedure (repeating the experiments multiple times) and take into account the cost into your comparisons. 
- The approach is naive in the meaning that they do not consider other factors that may affect the performance of each method: step sizes and other hyper parameters, starting point, whether the methods converge and with what network architecture (for the complex learning problems). Without these considerations, it is difficult to say that the proposed benchmark procedure is more fair than the others. 
- Contrary to the contribution sections, there is no 'novel' stochastic optimization algorithm introduced in the paper. The authors only apply previous method to Basin Hopping frameworks.
- The presentation of this paper is poor and very confusing. For examples: "populations of models" actually meant a collection of end-models of each trajectory created by an optimizer. "stationary distribution of an optimizer" makes no sense.

### Questions
The authors said they "expand their understanding of the relationship between (loss) optimization, generalization (testing error)". However this point is not clearly explained in the submission. Could you elaborate?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
