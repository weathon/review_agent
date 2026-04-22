# Discovering Quality-Diversity Algorithms via Meta-Black-Box Optimization

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Quality-Diversity has emerged as a powerful family of evolutionary algorithms that generate diverse populations of high-performing solutions by implementing local competition principles inspired by biological evolution. While these algorithms successfully foster diversity and innovation, their specific mechanisms rely on heuristics, such as grid-based competition in MAP-Elites or nearest-neighbor competition in unstructured archives. In this work, we propose a fundamentally different approach: using meta-learning to automatically discover novel Quality-Diversity algorithms. By parameterizing the competition rules using attention-based neural architectures, we evolve new algorithms that capture complex relationships between individuals in the descriptor space. Our discovered algorithms demonstrate competitive or superior performance compared to established Quality-Diversity baselines while exhibiting strong generalization to higher dimensions, larger populations, and out-of-distribution domains like robot control. Notably, even when optimized solely for fitness, these algorithms naturally maintain diverse populations, suggesting meta-learning rediscovers that diversity is fundamental to effective optimization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new MetaBBO approach to learn generalizable QD algorithm. Based on Dominated Novelty Search (DNS) algorithm, the authors uses neuroevolution based MetaBBO paradigm to train an attention-based scoring function. Given the fitnesses and pre-defined descriptor vector of a population of solutions, the proposed scoring neural network uses attention mechanism to process these raw features and then obtain the scores of solution population. The scores are used in a DNS framework to promote QD optimization. The neural network is trained by SNES (an evolutionary strategy), where the fitness for the neural network is accumulated as its average (median) performance when being used for scoring in DNS on a meta-task set (synthetic BBO functions). After training, the authors found that the obtained optimal neural network individual presents ideal generalization ability, which is beyond problem dimension, population size, etc.

### Strengths
The learned scoring policy at least shows competitive performance compared with QD baselines.

### Weaknesses
1. Lack of respect for related works.

I am researching closely within learning-assisted evolutionary computation, hence familiar with the concept so-called Meta-BBO (or MetaBBO, never mind). Although the authors have reviewed  works of Lange et al, I have to say that the first time I heard Meta-BBO is from another two teams' surveys [1][2], where the works of Lange et al are also included as pioneering works. However, if you authors have read these surveys, you would know that there are many MetaBBO works now. However, you almost neglect them all (I do not know if you are intended to) and only reviewed works before 2022 or latest works of Lange et al. How could a reader know whether your paper is valuable or not considering that you omit comparison (at least review) recent related works? Let alone QD algorithms, which from my experience the team of Qian et al is also famous with significant works such as [3][4]. I urge the authors reconsider their attitude on scientific research before submitting the paper.  

[1] https://ieeexplore.ieee.org/abstract/document/10993463

[2] https://www.sciencedirect.com/science/article/pii/S2210650224003766

[3] https://openreview.net/forum?id=bLmSMXbqXr

[4] https://dl.acm.org/doi/abs/10.24963/ijcai.2024/773

2. Lack of novelty.

If I understand correctly, the major contribution of this paper is proposal of learning scoring mechanism in EAs. However, learning scoring function or selection pressure through transformer is not novel at all, either in Lange et al's work [5] or end-to-end MetaBBO such as GLHF [6]. The authors did not discuss the technical novelty compared to these works, which hinders me from getting the real motivation and specific technical contribution of them to tailor LQD for QD algorithm. Besides, the trend in existing MetaBBO indicates that controlling a part of an algorithm (e.g., in this paper the selection pressure) is not flexible enough to discover new algorithm [7][8] (I read these two insightful papers recently), since the fixed algorithmic architecture presents limited refinement potential. I expect the perspective from the authors.

[5] https://dl.acm.org/doi/abs/10.1145/3583131.3590496

[6] https://proceedings.neurips.cc/paper_files/paper/2024/hash/19e9a88d91917775b34fdad447ed8908-Abstract-Conference.html

[7] https://openreview.net/forum?id=vLJcd43U7a

[8] https://arxiv.org/abs/2505.17866

3. Lack of comparative validation.

I quote a claim of the authors: "A striking finding from our analysis is that even LQD (Q) variant trained purely for fitness optimization naturally maintain significant population diversity.". I think this conlusion is very tricky. On the one hand, if so, why LQD (Q) is worse than LQD (QD)? Is this indicating that the proposed training procedure is not good enough so that it can not learn very effective LQD (Q)? On the other hand, if so, why the authors did not compare their models with existing MetaBBO approaches that use fitness optimization, what if these existing approaches have emerged QD principle by only-fitness rewarding? Overall, the comparison is very limited to tell true effectiveness of LQD. BTW: why robot control needs QD? 

4. Technical issues.

(1) What is the dimension of the descriptor space (D) used in this paper? the authors provide the dimension of training problem instances (n = 2 ~ 12), and claim that " By projecting high-dimensional genotypes into a lower-dimensional descriptor space while maintaining their relative distances,", however, on the one hand, they do not provide tie value of D, if D > 2, how this claim to be correct when problem dimension is 2? On the other hand, why the relative distance could be maintained under the standard normal condition? prove it.

(2) what does the reproduction size B denotes? it is not mentioned in the paper?

(3) I can not agree with the authors the proposed complex neuroevolution training is efficient, especially in experimental setting the authors claim that they use 8 H100 GPUs for this training task, use such huge computational resources to achieve competitive results with naive GA and MAP-Elite can not really prove the value of this work. 

To summarize, I do not think this paper (at its current version) deserves an acceptance. Substantial improvement need to be made to ensure the novelty, motivation, significance and scientific integrity before this paper could be accepted. I am wiling to engage discussion with the authors in the rebuttal phase and promise timely response.

### Questions
See weaknesses.

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
This paper introduces a meta-learning approach that creates new
competition rules for QD algorithms. Essentially, the QD algorithm is framed as
based on genetic algorithms, where new solutions are generated and added to the
population, and the next generation is selected from that population via
competition. In this paper, the selection is handled by creating a transformer
that outputs ranking values for each solution in the new population; the top N
individuals are then chosen. During the meta-learning process, the transformer's
weights are trained so that the resulting QD algorithm optimizes a
meta-objective. Results on black-box and robot control benchmarks show that the
final meta-learned QD algorithms exceed the performance of existing QD baselines
in terms of both quality/fitness and diversity.

### Strengths
1. I find the idea of the paper quite novel among QD algorithms, even though it
   builds heavily on the single-objective evolutionary algorithms in Lange
   2023b. It is certainly rare to see a QD paper that involves meta-learning.
2. I appreciate the analysis of the learned local competition strategies in
   Section 5.2, as they help to understand how the variants operate after
   meta-learning.
3. The paper is overall well-written and easy to follow. It also has good
   statistical rigor, performing multiple trials and carrying out statistical
   tests to determine if differences in performance are significant.

### Weaknesses
The paper takes a rather narrow view of QD algorithms as only being based on
genetic algorithms. This view is stated on line 126 in the background, and it is
reflected in the choice of baselines --- MAP-Elites, NS, and DNS are all based
on genetic algorithms. However, QD in recent years has been able to reach a
large number of machine learning domains by growing far beyond its roots in pure
genetic algorithms. For example, works like PGA-MAP-Elites and PPGA incorporate
reinforcement learning (RL) mechanisms to scale to high-dimensional RL problems,
while BOP-Elites incorporates Bayesian optimization to improve sample
efficiency. Furthermore, for black-box problems, CMA-MAE has demonstrated
state-of-the-art performance on many difficult domains. In its current form,
this paper does not account for any of these advances. In order for this paper
to be relevant to a machine learning community like ICLR, I believe it needs to
demonstrate relevance to algorithms like the aforementioned. While it is true
that these works focus on different QD mechanisms (e.g., variation) than that
addressed in this paper, I believe they are important enough that this paper
should discuss how they integrate with the LQD framework.

For example, this paper could include CMA-MAE as a baseline or discuss how LQD
can be applied to CMA-MAE. Right now, I think a valid concern is that running
CMA-MAE simply exceeds any of the LQD variants presented here in both quality
and diversity. If that is the case, LQD does not seem very useful because one
does not need to run the expensive meta-learning process. Instead, they can just
run CMA-MAE immediately. However, this is no longer a concern if it is clear (and ideally, empirically proven)
that LQD can somehow integrate with and enhance CMA-MAE.

### Questions
1. Suggestion: The text in Figure 4 and Figure 6 is quite small and difficult to
   read. Also, LQD(D) is described first in the text despite being second in the
   figure, which I found confusing.
2. The robot control domains seem quite different from those of prior work. For
   instance, their genotypes are only several hundred parameters rather than the
   tens of thousands of parameters used in works like PGA-MAP-Elites. This makes
   it difficult to contextualize the results in this paper with those of prior work. It would be
   helpful to explain these differences somewhere.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a meta-black box optimization approach to discover new quality-diversity (QD) algorithms by encoding various competition rules with a Transformer architecture. They evaluate their approach in a function fitting optimization benchmark and in several simulated robot control benchmarks, showing that their approach matches or outperforms standard QD algorithms in diversity only, performance only or QD metrics.

### Strengths
- the motivation of the meta-QD approach based on biological evolution principles is well-written and quite convincing
- I could not find any typo

### Weaknesses
- some elements of the methods and some experimental results are unsufficiently clear (see below)
- a few claims are unsupported (see below)
- some figures have issues (see below)
- relevant variants and ablations are missing (see below)
- it would make sense to compare the approach with approaches which strive to learn a good descriptor space (e.g. AURORA and many others).

### Questions
** Questions: **
- would it be possible to analyse the learned transformer model to "decode" the competition rules learned by your LQD approach?
- The descriptor space dimension is D. How do you deal with different values of D in the architecture of Fig. 2?
- You generate a random matrix of size $R^{Dxn}$. What's n?
- p.5, you project high-dimensional genotypes into a descriptor space, but in most QD algorithms, the descriptor space is rather built on phenotypes, that is encodes a description of the trajectories performed by the agent. Did you try both types of encoding?
- Section 3.2.2 is unsufficiently clear. In table 1, what is the "Eval metric" column? It looks close to the raw score. 
- Why does the Meta-objective use a median rather than a mean. Did you try variants? Same question about Tables 3 and 4.
- Why did you use SNES rather than any other ES? Did you try variants?
- Is the performance given in curves the average fitness of the final population? The fitness of the best individual? This should be clarified.
- In Section 4, you describe the hardware used, but not the optimization time. How expensive is it?
- In Section 4.2, you compare scores with respect to baselines, but you did not describe your efforts to make sure that the various algorithms benefit from the same computational budget. Is it the case? What is the training budget?
- Section 4.3 is missing a lot of experimental details. What are the 584 parameters for? It is "up to" 584, so which other sizes do you use in which contexts?
- do you see a possible alternative to using transformers in your architecture? Could you evaluate such a variant?
- I believe the topic of Appendix B.7 is important and should be moved to the main paper. BTW, the fact that "LQD leverages meaningful relationships between solutions captured be the descriptor space" is not supported by a detailed analysis. Again, providing such an analysis would strengthen a lot the paper. At least, can you provide an example of such a relationship?
- On figures 17 to 22, what are the axes?
- In Table 6, what is the descriptor space?
- In section E.3, the descriptor spaces should be described more accurately for readers who are not familiar with the QD literature

** Remarks: **
- In the baselines, you mention the Random baseline which does not appear in the results. You should specify immediately that this baseline is used only for performance normalization. I had to dig this information from the caption of Fig. 7, this is not the right place.
- Section 4.3 is "Robot control tasks", but there is no robot in your work, only simulations. You should turn any mention of "robot" into "simulated robot", as dealing with real robots is much harder (and probably you method cannot address this)
- Figures 4 and 5, normalization should be explained
- lines 435 sq.: the fact that LQD "creates crucial stepping stones" is not supported by any in depth analysis. Such analyses would strengthen the paper a lot
- Figure 6 does not have a title, nor a color scale, nor labels on the axes. Under its current form, it cannot be exploited for analyses. You need to correct this!
- Some text in Fig.7 is not readable.
- Appendices B1, B3, and E1 should be grouped into a single Section.
- You may make it more explicit that your meta-BB approach is trained on fitting functions, but evaluated on episodic control tasks, which makes it clearer that these evaluation tasks are completely OOD.
- I would like to see Figure 15 for more functions of the benchmark, as the ones you show could have been cherry-picked.


** Typos: **
- line 94, Section A -> Appendix A
- line 251: BBOB is not defined
- line 420: (Section 5.1), the -> and the

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Quality diversity (QD) algorithms are a relatively new class of evolutionary algorithms which focuses on maintaining population diversity by constraining competition between individuals to similar individuals. Several QD algorithms have been proposed in the past, including MAP elites (ME) and dominated novelty search (DNS).

The paper proposes to meta-learn QD algorithms (a transformer architecture) by learning competition rules between individuals from black-box optimisation tasks. The method consists of an inner, evolutionary loop where a given fixed QD-model is applied for some generations, and an outer-loop that optimises models based on the outcome of the inner loop.

The learned QD-algorithms are compared empirically with classical QD-approaches, such as ME and DNS, showing favourable performance, and generalisation ability.

### Strengths
Meta-learning of evolutionary algorithms have been around for some time. The approach has been evaluated on a large number of problems, ranging from the classical BBOB benchmarks to robot controller tasks. The learned QD approach partly outperforms ME and DNS on the BBOB benchmarks.

### Weaknesses
The application of statistical methodology could have been better. For example, in table 4, it would be useful to see p-values from statistical tests, or some information about variance.

There is no proper analysis of scalability. I would have liked to see how the performance scales with increasing problem dimension across all methods considered for some problems.

There is a very limited ablation study. It would be particularly interesting to understand the impact of the transformer architecture on performance. The network is tiny. How much performance improvement would a large network provide?

The paper does not compare with state of the art black-box optimisation algorithms. To give a clear idea about performance, I would expect to see a comparison with CMA-ES.

### Questions
How does LQD compare with CMA-ES?

What is the scalability of the approach?

### Soundness
3

### Presentation
4

### Contribution
3
