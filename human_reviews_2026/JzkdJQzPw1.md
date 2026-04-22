# Mode-conditioning unlocks superior test-time compute scaling

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 2, 6

## Abstract
Parallel sampling is essential to test-time scaling and reinforcement learning (RL), but its effectiveness is sharply limited by diversity collapse, where models concentrate on a few modes and repeated samples produce the same mistakes. We propose the mode-conditioning (ModC) framework, which explicitly allocates sampling compute across reasoning modes using either specialist models or mode-specific prefixes. With predefined mode labels, ModC consistently improves test-time scaling (Pass@k) across controlled graph-search tasks and math reasoning benchmarks, spanning model families and sizes from 0.5B to 7B. On OpenThoughts, fine-tuning Qwen2.5-7B with ModC achieves an 4× efficiency gain over standard training while also improving the maximum attainable Pass@k. We further show that gradient clustering enables ModC without predefined mode labels, yielding up to 10% gains on datasets such as NuminaMath. Finally, we show that ModC improves Pass@k after RL training and can further boost the Pass@k gains of diversity-inducing RL methods. These results demonstrate that standard training underutilizes the diversity in data, and that ModC provides a simple, effective remedy for unlocking the full benefits of diversity in parallel sampling.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work suggests a method called ModC for improving diversity in generation during parallel scaling.

The idea is that different ways to approach a problem may fall into different "modes", corresponding to different broad strategies. A diverse generator of possible proofs should try to sample as diversely as possible from different modes.

This paper suggests two possible ways to do this: (1) train a separate model on each mode, or (2) train a model with prefix-tuning to make it use one mode.

Modes can be either (1) known a priori, or (2) found automatically with a gradient clustering method. The paper tests the idea on several benchmarks such as NuminaMath, AIME, and Countdown, and finds benefits over vanilla parallel scaling -- especially when there is a large amount of parallel scalin.g

### Strengths
* ModC has a conceptually clear motivation.

* The proposed method is simple to implement, and seems to yield increased performance. This might be of interest to much of the ICLR  community, since methods to improve to model reasoning are quite popular.

* The paper presents a way to find modes automatically, using gradient cluster. This makes it more broadly applicable than it would be if the modes had to be created manually.

* The experimental methodology seems mostly sound (although I have a question -- see weaknesses below).

### Weaknesses
* On the methodology: I'm not sure how good of a metric pass@k is for AIME, when k = 1000, because there are only 1000 possible solutions for any problem as far as I know. 
    - Having a model that outputs a random number from 0 to 999 would give a 63% pass@k accuracy, which is roughly the accuracy  reported in Figure 5. 
    - On the other hand, having 1000 models (each of which outputs a constant number) would give a 100% pass@k accuracy.

* There's a quite relevant prior work called "Metadata Conditioning Accelerates Language Model Pre-training" by Gao et al., 2025, that this made me think about. There they show that adding metadata of which website a text came from can improve model performance. It could be good to discuss the connection with this work.

### Questions
Typos: "this achieves up to xx% improvement"

### Soundness
3

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
This paper studies mode conditioning as a way to address the lack of diversity in LLM generations for reasoning tasks. The problem is that when we scale test time compute with parallel sampling, current models tend to collapse to one or two dominant strategies, so additional samples mostly repeat the same errors. The paper’s proposal is to make the modes explicit and to allocate test time samples across modes rather than drawing all samples from a single collapsed distribution. They instantiate this in a controlled search setting (Countdown, a generalization of Game of 24) where the target value can be found either by a DFS style search or a BFS style search, and where the search trace itself reveals which mode was used. They then extend the idea to math post training with multiple teacher models and finally to a setting where modes are discovered automatically via gradient clustering.  They show that mode conditioned training and mode conditioned inference improves Pass@k relative to standard mixed training.

### Strengths
I think the problem is interesting and well chosen. How to obtain diversity in reasoning style without simply increasing sampling temperature (which has its own issues) is still not well understood, and most current post training pipelines / RL algorithms do in fact make diversity worse.

I like the synthetic setup with countdown game since it cleanly isolates the question they are trying to answer, with a way to verify which mode of problem solving is used.  The experiments are pretty thorough and they also show some nice results in the math CoT setting, as well as in the automatic mode finding setting using gradient clustering.

### Weaknesses
1. Several parts of the paper are somewhat hard to follow on first read. One example is Figure 2. It is not completely clear how the per problem histograms are computed. My reading is that for each test problem the authors sample the model repeatedly, detect for each sample whether the model used DFS or BFS, compute the fraction of BFS samples for that problem, and then plot the distribution of that fraction over all problems. If that is correct, the number of samples per problem needs to be stated. If that is not correct, the figure needs a more explicit description. Right now it is difficult to tell what exactly is being compared. 
2. In the separate model setting each mode gets its own model trained on the subset of data for that mode. In Countdown the paper notes there are about 97k DFS trajectories and 65k BFS trajectories. If each of those is used to train a full model of size $N$ then the total training compute for the separate model setting is roughly $6 \times (2 N) \times (97+65)/165 \approx 11.7 N D$  flops, whereas the standard training or prefix based modC is roughly 6ND .  This would mean that the separate model setting is using roughly twice the training compute.  The paper should clarify whether training budget was controlled, whether epochs were scaled down for the separate models, or whether the comparison is intentionally not compute matched. As written, it is not a fair comparison.
3. A natural application of this work is to post train with RL, where we know that the distribution becomes sharper and diversity decreases. As far as I can tell, the paper only considers SFT / distillation like settings. A natural question is: after RL, does mode conditioning still preserve the benefits shown here, or does RL erase them. It would be useful to see in the same synthetic Countdown setting a comparison of RL that samples from the usual policy versus RL that is constrained to use mode conditioned sampling during rollouts. If the authors can show that they can keep the sampling gains after RL training (evaluating 0-shot after RL) that would be a nice finding, even if on a synthetic task.
4.  There are a few grammatical and formatting issues. For example Section 5.1 appears to have an incomplete closing sentence. (Did not penalize for this.)

I am willing to improve my score if the authors can meaningfully address these comments.

### Questions
1. Unclear algorithm in Section 4.2 - The post training described for math reasoning in Section 4.2 seems to be plain SFT on two teacher traces with either mode specific prefixes or separate models. It would be good to state clearly that no RL was used here, if that is in fact the case.
2. Figure 4 interpretation - The caption says that the dark gray line is “best teacher.” Does this mean this curve corresponds to distillation from only the best single teacher, not distillation from the union of best teacher traces across problems?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents a simple, yet effective approach to diversify the model’s outputs by providing explicit control over the modes. It explores two methods: 1) training separate specialist models and splitting test-time compute between them, 2) training a single model with mode-specific prefixes, and sample equally with the corresponding prefixes. The paper shows that these approaches surpass the mixed model trained with both modes. Morover, in the case of unspecified modes, the paper proposes an automatic mode discovery method based on gradient clustering. It shows that the method captures the labels reliably and further mode-conditioning on them recovers the improvements.

### Strengths
The paper pinpoints a simple but important suboptimality in training language models with diverse data. It verifies the intuition with experiments with both toy settings such as different strategies for Countdown, and with real-world tasks and traces distilled from teachers. It is comprehensive in experimenting with different forms of chain-of-thought (short and long) generated with different models. Moreover, the work pushes its practical relavance further by providing a method for discovering unobserved modes in the data based on gradient clustering, which makes the idea more generalizable to different settings.

### Weaknesses
The paper could improve its presentation by defining its metrics more clearly. For example, it’s not clear how the “Fraction of BFS per problem” metric is computed for Figure 2. In section 5.1, p_\theta is not defined, so it’s not obvious how the gradient is computed. 

I also did not understasnd how heuristic prunings and search budget constraints make the problems solvable with only one of BFS and DFS, making it unclear why this setting captures the diverse setting desired. 

The novelty of the idea to learn separate models and aggregating them instead of learning from a mixed dataset is questionable given the literature around mixture of experts and other works such as “Mix Data or Merge Models? Optimizing for Diverse Multi-Task Learning”.

### Questions
1. Could you please explain how the heuristic prunings limit the solution to one of BFS and DFS?
2. How is the ‘fraction of BFS used’ computed?
3. For the distilling experiments, what kind of prefix do you use for different teachers? How does knowledge sharing happen in those 4. experiments if the model learns to follow one strategy given a prefix?
4. Could you explain how the gradient is computed in the gradient clustering method?
5. Did you run the gradient clustering method for the long-CoT datasets too?

### Soundness
3

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
4

### Summary
The paper proposes mode-conditioning, a test-time inference strategy that allocates a certain number of samples to each mode in order to improve diversity of samples, mitigating the issue of mode collapse. The authors show that ModC leads to consistent gains across tasks both when modes are fixed, as well as being able to discover modes automatically via gradient clustering.

### Strengths
- The ModC method is novel, creative, and effective, addressing the critical issue of lack of diversity.
- The authors demonstrate that ModC training works well on a variety of tasks. They explore the idea throughout a variety of settings and domains, and it performs above standard baselines in all cases. The technique seems to be quite general and could have potential downstream applications beyond those listed in the paper.
- The authors also compare different ways of implementing mode conditioning, and do a thorough analysis on other factors like model size, CoT length, etc.

### Weaknesses
- The work does not have any comparisons with other diversity-inducing techniques, for example pass@k training (https://arxiv.org/abs/2508.10751) or optimal sample allocation (https://arxiv.org/abs/2410.22480). While ModC is evidently effective against simple baselines, it is difficult to understand the advantages and disadvantages of this method against some of these other methods.

### Questions
- Most of the ablations are comparing variants of ModC. Could you provide a comparison of ModC against other diversity-inducing techniques (see comment in weaknesses section)?
- Do you see a clear diversity increase after ModC? For example, for MATH500, if you consider how many distinct answers are produced for each problem, how much does it increase with ModC?
- Does the idea also apply to other domains, such as code generation?

### Soundness
3

### Presentation
3

### Contribution
3
