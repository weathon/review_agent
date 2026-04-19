# The Complexity Dynamics of Grokking

- Decision: Reject
- Scores: 6, 5, 3, 8, 3

## Abstract
We investigate the phenomenon of generalization through the lens of compression. In particular, we study the complexity dynamics of neural networks to explain \emph{grokking}, where networks suddenly transition from memorizing to generalizing solutions long after over-fitting the training data. To this end we introduce a new measure of intrinsic complexity for neural networks based on the theory of Kolmogorov complexity. Tracking this metric throughout network training, we find a consistent pattern in training dynamics, consisting of a rise and fall in complexity. We demonstrate that this corresponds to memorization followed by generalization. Based on insights from rate--distortion theory and the minimum description length principle, we lay out a principled approach to lossy compression of neural networks, and connect our complexity measure to explicit generalization bounds. Based on a careful analysis of information capacity in neural networks, we propose a new regularization method which encourages networks towards low-rank representations by penalizing their spectral entropy, and find that our regularizer outperforms baselines in total compression of the dataset.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the grokking phenomenon through compression-based approaches. Inspired by recent work on the intrinsic complexity of neural networks, and combining it with ideas from rate-distortion, quantization and low-rank approximation, the authors propose a new measure of neural networks complexity, which consists essentially of a coarse-graining procedure. They conduct experiments on simple arithmetic tasks which demonstrate that the rise and fall of this complexity might be predictive of the network starting to generalize. Moreover, this leads them to propose a new regularization scheme, based on spectral entropy, whose effect seems to reduce the total description length and the generalization bound, compared to other methods. This might lead to non-vacuous generalization bounds.

### Strengths
- Grokking is an important topic for the community

- The experiments suggest that the proposed regularization technique based on spectral entropy may induce grokking, which may be of practical interest.

- The experiments suggest that the rise and fall of the proposed complexity seems to be predictive of when the model starts to generalize.

- The proposed regularization techniques lead to better generalization bounds than classical weight decay or no regularization.

### Weaknesses
- Several notions are mentioned repeatedly but without being formally defined, such as capacity, distortion or $(\lambda,\delta)$ (Equation (9)). It would improve the paper to include additional theoretical background and more formal definitions. 

 - It should be made clearer how the quantities introduced in Sections 3.1 and 4 are related to generalization. For instance, is it possible to write down a theorem with explicit dependence on these quantities, or are their consideration partially based on intuitions? Can the link of these quantities with Kolmogorov complexity be made more formal?

 - Despite the lack of formal theorems and proofs, the experiments are done on very simple arithmetic tasks. Therefore, it is not clear (neither theoretically nor empirically) whether the results may be generalized to more complex settings. I think that at least one experiment on a small dataset like MNIST or CIFAR10 could improve the paper.

 - It would be useful to include an experiment comparing the performance (in terms of accuracy) with and without the proposed regularization scheme. Indeed, we see that it reduces the MDL and the generalization bound, but, if I am correct, it is not clear whether it achieves better performance overall.

 - We see in Figure 4 that the proposed regularization scheme achieves the lowest complexity. However, the complexity is computed by Algorithm 2 and the proposed regularization is precisely penalizing the quantity computed by algorithm 2. Therefore it does not seem surprising that it is the lowest. As an ablation study, it would be interesting to make the comparison using other complexity notions. For instance, using the actual test accuracy would be very informative, to see whether the proposed regularization leads to better performance.

### Questions
- Is it possible to perform the same experiments on more complex but still relatively simple datasets like MNIST or CIFAR10?

 - Does the generalization bound of Equation (4) only hold for finite hypothesis spaces? If yes is that a realistic assumption in practical learning settings? Moreover, could you be more precise as to why the choice of Solomonoff prior should lead to tighter bounds than other priors, such as the uniform prior over $\mathcal{H}$?

 - Line 181: Why can the empirical risk be understood as the entropy of the data under the model? Is there a way to formalize this fact?

 - Is it possible to obtain a formal statement relating the information capacity (Equation (9)) to generalization?

 - To what size and precision do the parameters $\lambda$ and $\delta$ (Section 4) refer to in practice?

 - How would the training accuracy be affected by the addition of Gaussian noise in practical deep learning settings?

 - Can you define more precisely the notations used in Algorithm 2, such as BO.SUGGESTPARAMETERS()? More generally, can you provide more details on the Bayesian optimization procedure?

 - Does your regularization technique always lead to lower test accuracy compared to weight decay?

 - Figures 3 and 5 are not analyzed in the text, can you add some insights on the result they present?

**Remarks/questions regarding lines 152 - 155 and Equation (4)**  
Even though it is not central to the paper, I have some questions about this part:
As I understand it, the bounds in terms of Kolmogorov complexity are obtained by choosing a good prior distribution in the bound of Langford and Seeger. It is not clear to me that such a choice of prior provides the most useful bound. More precisely, let $\mathcal{H}$ be a finite set of hypothesis and $\sigma : \mathcal{H} \to \mathcal{H}$ be any bijection of $H$. Then $h \mapsto 2^{K(\sigma(h))}$ may be used as a prior instead of the usual Solomonoff prior, hence leading to a generalization bound in terms of $K(\sigma (h))$. Yet another possibility would be to use the uniform prior over $\mathcal{H}$. Therefore, choice of prior, and therefore the choice of Kolmogorov complexity as a generalization measure, seems to be arbitrary (please correct me if I am mistaken). Can you provide more insights as to why this leads to the most informative bound? 

I would be happy to discuss this further, please correct me if I misunderstood something.


**Other minor remarks and typos**

 - In the introduction, the terms capacity and complexity are used before being defined, which may render the introduction hard to read. In general, more formal definitiosn of these concepts might enhance the readability of the paper. It could also help to define the notion of distortion function.

 - Line 122: regulariztion $\to$ regularization

 - Equation (4): there is a missing parenthesis in $\log(1/\delta)$

 - There might be a clash of notation between the parameter $\delta$ in Equations (4), (9) and (10). It would be clearer to use a different letter in each of these equations.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes to study the grokking dynamics via the lens of information theory (minimum description length). In particular, they proposed: (1) a new compression algorithm to compress the neural network; (2) a new regularizer based on spectral entropy. They show that the spectral entropy regularizer outperforms the standard weight decay to the extent that a model with lower complexity is obtained. They claimed a factor of 30-40x improvement of the compression ratio over bzip2, which is impressive (although I can't find the file size data). However, none of the compression methods achieve a non-vacuous bound, since models are vastly over-parametrized.

### Strengths
* The paper is well-written and very readable
* The paper presents "new" theoretical tools to analyze neural networks
* The analysis is a new angle to understand grokking

### Weaknesses
* This paper deals with too many things simultaneously, which makes me a bit lost. What's *the* motivation of this paper? Otherwise, the paper reads like a collection of ok-ish results but none of them is impressive enough. For example, the idea of grokking as compression has been explored by [Liu et al.], [Humayun et al.] and [Deletang et al.]. The idea of using spectral entropy as a measure is explored in [Liu2 et al.], although it is novel to regularize the network with spectral entropy (which is unfortunately expensive).
* The papers claim a 30-40x improvement in compression ratio, but I did not find and details or data. 
* Although this is a more theoretical paper than an experimental paper, I am not sure about its practical implications.

**References**

[Liu et al] Grokking as Compression: A Nonlinear Complexity Perspective, arXiv: 2310.05918

[Delétang et al.] Language Modeling Is Compression, ICLR 2024

[Humayun et al] Deep Networks Always Grok and Here is Why, arXiv: 2402.15555

[Liu2 et al] Towards Understanding Grokking: An Effective Theory of Representation Learning, NeurIPS 2022

### Questions
* What's the key motivation of this paper?
* Could you elaborate on the comparison with bzip2? What is being compressed, problem setup, compressed file size, etc.?
* What practical implications does this paper have? I would consider a method practically useful if: (1) it can speed up grokking and/or (2) it can compress real-world datasets better than baselines.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors introduce a measure of neural networks’ complexity, and show that grokking could be explained by the rise and fall of the model’s complexity. The authors also propose methods for compressing neural networks via quantization and spectral entropy-based regularization, and empirically demonstrate their performances with modular arithmetic tasks.

### Strengths
- The paper is generally clear, and easy to read and interpret.
- The paper provides nice intuitions on building generalizable neural networks, especially from the model complexity perspective.
- The paper considers an interesting set of techniques for model compression with minimal performance loss, and tests them with experiments.

### Weaknesses
While the paper considers several promising ideas for model compression, there are a few limitations:
- While the complexity explanation of grokking is interesting, it seems to overlap with the circuit efficiency explanation proposed by Varma et al. (2023). Although the authors acknowledge that model complexity is not exactly identical to efficiency or parameter norms, the added insights in this area feel somewhat limited.
- The proposed model compression methods are quite similar to existing techniques on quantization and low-rank approximations, which raises questions about the novelty of the approach. Spectral entropy-based regularization is an interesting idea, but concerns about potential computational overhead and their applicability in more complex settings remain.
- Lastly, the applicability of entropy regularization techniques in more complex problems beyond the modular arithmetic task raises some concerns. Additional evidence or analysis demonstrating how this technique can advance the complexity-performance Pareto frontier in more difficult tasks will strengthen the paper.

### Questions
1. How did you set the learning rates for experiments? Does the performance of entropy regularization vary with different learning rates?
2. While entropy regularization surely helps in compressing the model, I expect that both the usual L2 regularization and the entropy regularization will achieve perfect test accuracy. Could you think of a scenario where the proposed regularization technique offers a clear performance advantage over L2 regularization?
3. Will entropy regularization also help in training larger models with more complicated datasets, where they often do not have simple representations as one-dimensional numbers?
4. Could the computational overhead of low-rank optimization become significant, especially when applied to large models? If so, how could we mitigate them?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the phenomenon of grokking through the lens of complexity theory and rate distortion theory. It proposes ways to compress model weights: 
-- Via a parameter quantization operation, as a twist on ideas of Hinton and Van Camp
-- Via a low-rank approximation operation.
The idea is compress the models up to certain rate distortion thresholds, quantified by the loss. 
They find that this compression is substantially more powerful than traditional compression methods (bzip) and argue that this is a better approximation of the Kolmogorov's complexity of the model.
Using this metric, the authors perform experiments on arithmetic operations and find that the grokking phase is associated with a drop from the complexity peak. Following this idea, they propose a new regularizer that apparently increases the grokking effect.

Overall, this is a very well-written paper that lays out super interesting ideas and presents a compelling thesis and nice experiments. I am not sold on the idea that this is an explanation of grokking, but the observations and the conclusions are overall very interesting and I think this is a valuable contribution to understanding better what happens with grokking and is quite promising to improve learning performance of models.

### Strengths
Excellent writing, compelling ideas, nice experiments, convincing thesis, possible follow-ups.

### Weaknesses
Is it really an explanation of grokking or more some interesting and attractive observations?
The experiments with the regularizer are not many.

### Questions
Have you tried applying these ideas to more complex datasets, does it compare favorably vs weight decay techniques ?

Bzip is not ideal to compress weights... are there other points of comparisons available?

What is the efficiency of your compression method? How long does it take to compress?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The authors introduce a new complexity measure for neural networks and claim that this complexity measure can be used to explain 'grokking'. "Grokking" in machine learning is this idea that neural networks suddenly transition from memorization to generalization long after overfitting the training data. They show that their complexity measure correlates with this 'grokking' and then show how this complexity measure can be used to define a new regularization method which encourages low-rank representations. This regularizer is defined using the spectral entropy of the network weights.

### Strengths
Understanding the role of model complexity and how it should be measured is an important question in machine learning. This paper takes a good step in this direction and presents a compelling case for a complexity measure which is defined using the minimum description length and ideas from compression and information theory. The paper contributes to a deeper understanding of this 'grokking' phenomenon, which has gotten significant attention in recent years. 

The paper has good theoretical motivation and makes an interesting connection with the concept of grokking in machine learning. Their intrinsic complexity measure and regularization technique are well-grounded in theoretical concepts from information theory. The authors provide clear explanations and justifications for their design choices.

The paper is logically structured and well-written and supports their theoretical claims with experiments on synthetic tasks, like modular arithmetic, for decoder transformer models.

### Weaknesses
The complexity measure defined and explored in this paper is positioned as a way to 'explain grokking'. 

Comparison with other complexity measures. The empirical results in the paper are nice. But it would be good to have a fair comparison of how other complexity measures look when measure in the same scenarios. It's unfair to say that this new complexity measure "explains" grokking without uncovering a scenario where this complexity measure is able to capture this behavior where others are not. Otherwise, it's unclear if this is just a correlation relationship with the perceived behavior of 'grokking'. 

Lacking discussion of the cost for computing this complexity measure. If I understand correctly, the proposed complexity measure involves a Bayesian optimization procedure for finding the optimal compression parameters, which could be computationally expensive. It would be nice to address or (ideally) investigating how difficult this measure is. This would  enhance the practicality of the approach.

From what I understand, this complexity measure is somewhat dependent on the hyperparameters, in particular the per-layer truncation threshold $\kappa(\tau)$. It would be nice ot have a detailed analysis even experimentally of the sensitivity to this threshold.

This paper has some very nice ideas and is worth exploring but it would be good to have a section on Limitations of their approach with an honest assessment in terms of other complexity measures and the degree to which the results are not just correlational with this 'grokking' behavior. 

The paper is carefully written and has a nice re-cap of the relevant ideas from information theory and compression in ML. However, the main message of the paper was at times hard to find. For example, what is the exact definition of this new complexity? I understand it relies on coarse-graning of the network and compression using bzip2 and I think the size of the compressed network is the proxy for the complexity. Is that the definition? This paper would benefit from more clear exposition in this respect.

### Questions
- What is the exact definition of the novel complexity measure introduced in this paper? And for which models is this measure well-dfined. The related conversation about compression and motivation from information theory and Kolmogorov complexity is very nice but it's unclear to me exactly how this measure is defined. Is this the content of Algorithm 2? Does the output of Algorithm 2 define the complexity measure? 
 - in line 400, can you clarify which subset of grokking experiments you used. And why you used this subset. 
 - in line 358 you state "..we show that regularizing the spectral entropy leads to grokking.." Is this an overstatement? How exactly is grokking defined quantitatively?
 - In Figure 3, you compare your regularization technique with weight decay. What is the dependence of the proposed spectral entropy regularization on the regularization weight? What behavior do you notice as you apply more or less spectral regularization? It would be nice to see the effect as the regularizaiton of the spectral entropy gradually increases.
 - Does Figure 4 include multiple seeds? Why are error bars not visible in this plot?

typos/nits
 - in Figure 2. Why include the "ours" distinction when all plots are "ours". 
 - line 372, "ideas" to "ideal"

### Soundness
2

### Presentation
2

### Contribution
2
