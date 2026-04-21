# Full Elastic Weight Consolidation via the Surrogate Hessian-Vector Product

- Avg Score: 3.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3

## Abstract
Elastic weight consolidation (EWC) is a widely accepted method for preventing catastrophic forgetting while learning a series of tasks. The key computation involved in EWC is the Fisher Information Matrix (FIM), which identifies the parameters that are crucial to previous tasks and should not be altered during new learning. However, the practical application of the FIM (a square matrix that is the same size as the number of parameters) has been limited by computational difficulties. As a result, previous uses of EWC have only employed the diagonal elements, or at most diagonal blocks, of the matrix. In this work, we introduce a method for obtaining the gradient step for EWC with the full FIM, which is both memory and computationally efficient. We evaluate the advantages of using the full FIM over just the diagonal in EWC on supervised and reinforcement learning tasks and our results demonstrate a quantitative difference between the two approaches, which are more effective when used in combination. Finally we show both empirically and theoretically that the benefits of using the full FIM are greater when the network is initialised in the lazy regime rather than the feature learning regime.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work studies the elastic weight consolidation (EWC) and tries to alleviate the computation burden of the Fisher Information Matrix (FIM). The main contribution is to propose a method for obtaining the full FIM. The performance is examined in the image classification and reinforcement learning.

### Strengths
This paper is relatively easy to follow but there still remains room to improve the organization. The vector product or matrix decomposition in learning is reasonable, and this trick most applied to sparsity, e.g. kernel matrix in Gaussian processes.
The examination of continual learning is relatively but not all standard in terms of benchmarks.

### Weaknesses
**1. About the layout of this paper.**

It seems there is too much background knowledge about the computation bottleneck in EWC from Page2 to Page4. There misses some parse in paragraph or key points are not well highlighted. These make it a bit difficult to follow in logics.

**2. About the contribution.**

The theoretical advantage to obtaining the full FIM is not well clarified in this work. Does it mean more information about the FIM brings better generalization in a theoretical sense? In the visual abstract, it seems the combined EWC can achieve more superiority than the full EWC, while this work focuses on the full EWC. Does this violate the research motivation? Meanwhile, it is necessary to connect some proposition to empirical observations.

**3. About the evaluation in reinforcement learning.**

I am afraid three Atari games are not typical in the continual learning domain. There exist more convincing benchmarks, e.g., Continual World, for continual reinforcement learning to examine the performance.

### Questions
See the weakness part.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces the SHVP (Surrogate Hessian Vector Product) algorithm, as a means to compute the product of the Hessian matrix of a neural network with a given vector, without needing to compute the matrix itself (thus significantly reducing the computational complexity), but by relying on two nested backpropagations instead.

Using this, the authors are able to train models in a continual learning context, using the Elastic Weight Consolidation (Kirkpatrick et al. 2017) training objective to avoid catastrophic forgetting, while using the complete Hessian matrix, as opposed to a diagonal approximation as is often used in practice for computational reasons.

The paper experimentally shows that using the full Hessian matrix improves the model's capacity to retain its performance on older tasks, and that the best performance is obtained by combining the full EWC with its diagonal approximation, allowing an even equilibrium between older and more recent tasks.

### Strengths
The article is clear, and provides good background and context to position its work in the larger literature.

The proposed algorithm is sound and well justified, and the experimental exploration of the impact of the different variants of EWC, in relation with the two training regimes is an interesting contribution.

Overall, I believe the SHVP algorithm introduced here can be an useful tool to explore the Hessian matrices of neural networks.

### Weaknesses
I have one main issue with the paper and its contribution: the positioning of the authors with regard to how and when their proposed Combined EWC could be used is very unclear.

As far as I can tell, the proposed SHVP algorithm trades the one-time computation of the full Hessian of the model for a double backprop through the NN at every iteration of the training. In particular, this means that computing the gradient associated with the EWC regularization term *requires computing an expectation over the datasets of the previous tasks*.

My understanding is that the main appeal of EWC is to keep some kind of summary of the previous tasks as the Hessian of their losses (approximated as diagonal or block-diagonal for computational efficiency). This relies on the idea that keeping around the training data of the previous tasks is either not desirable or not possible, otherwise one would simply train their model on all tasks simultaneously.

Here, the proposed "Combined EWC" requires keeping access to the data of the previous tasks, and appears to be more computationally expensive than just training the model on the multiple tasks simultaneously.

As a result, it is unclear to me what does "Combined EWC" actually bring to the table: while not stating it clearly, the paper frames it as an algorithm that could be used in practice to train models in the continual learning setting. But given the above remarks, I fail to see when one would actually want to do that.

### Questions
**How is the computation defined by Proposition 1 done in practice?**

The paper and appendix are not very detailed about it, and the joined code is barely documented, making it difficult to understand. I don't see how it would be possible to compute the double gradient defined for SHVP without retaining the whole computational graph associated with task A in memory (for performing the second backprop).

**When would someone use SHVP or Combined EWC in practice?**

Given the previous discussion, I'm having a difficult time figuring when one would opt for using Combined EWC, instead of simply training the model on multiple tasks simultaneously, given EWC is supposed to be an approximation of that. What point is there for me to use an approximation that is not cheaper to compute than the actual thing?

### Soundness
3 good

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces a method to calculate the full Fisher Information Matrix for EWC (instead of the more common diagonal Fisher) that is computationally and memory efficient. The paper then argues that combining the full EWC with a diagonal EWC is better than just one or the other. The paper then argues that EWC works better in the lazy training regime (with large parameter initialisations). Finally, in Section 5, the paper applies their method to an RL problem (3 Atari games sequentially shown).

### Strengths
1. I really like the algorithm for Full EWC in Section 3. I like that the authors looked at empirically verifying it in Appendix A.1, and would have liked to see more of this! 

2. Applying to RL / Atari is not done often enough in continual learning, so it was nice to see that experiment in this paper. 

3. I liked the proof sketches in the main text; I thought they were well-written and useful.

### Weaknesses
1. It is important to talk about other related works that use eg block-diagonal Fishers for EWC in Section 2. For example, Ritter et al., 2018 (A Scalable Laplace Approximation for Neural Networks). 

2. Permuted MNIST is an old benchmark with many problems. It is difficult to draw many conclusions of note from it due to the various issues with it (see for example Farquhar and Gal (Towards Robust Evaluations of Continual Learning) or Swaroop et al (Improving and Understanding Variational Continual Learning) for discussions). Additionally, the authors in this paper only use 5 tasks, which is very few, making it difficult to draw any real conclusions. It would be great if the authors ran for significantly more tasks to see if their conclusions/results still hold (eg 20 tasks). 

3. When comparing Figures 2 and 3, I'm not really sure if the lazy training regime is better for EWC! It looks like Figure 2 often has higher performance, at least on everything but task A's accuracy after training on the last task. Could the authors report other metrics like overall average accuracy (and maybe forward/backward transfer) too? 

4. Proposition 2 seems like it is re-writing that, under conditions like constant covariance (ie a quadratic loss landscape?) and mean-squared error loss, that Laplace approximation is ideal. I am not sure that there is anything new here: this is the reason that eg the EWC paper used the Laplace approximation / FIM in the regulariser / Bayesian approach. I found it odd that, in the Appendix, the authors prove this via "Bayes Optimal estimators" (ie looking at predictions at a test point?) and then take the limit as the Gaussian approximation becomes a delta (ie reduce the covariance to 0). This looks like a simple MAP estimation to me? Please let me know if I am missing something here. 

5. In the text on page 7, the authors argue why the landscape may be (more) locally convex in the lazy training regime, because parameter values do not change much during training. I am not sure I am convinced by this: just because the parameter values do not change (relatively) very much, this does not mean that the loss landscape that the parameters move through is better-behaved: it could still be highly non-convex. Is there previous literature on this (or could the authors design an experiment to show this)? 

6. I do not understand the intuition for why Combined EWC is better than Full EWC or diagonal EWC. Diagonal EWC seems to perform very well in Figures 2 and 3 already. In Figure 4, it does not forget task 2 when training on task 3 (although it does forget task 1), against the authors' conclusion (that Diagonal EWC prioritises new tasks). I think I need much more evidence of these claims (eg that Diagonal EWC prioritises new tasks while Full EWC prioritises old tasks) to believe them sufficiently. 
- Also, although it is nice to have an RL experiment in Section 5, having only 3 tasks/games is too few to draw conclusions.

### Questions
Please see Weaknesses section. More minor questions / comments: 
1. Note that the authors are using the empirical Fisher, not just the Fisher / FIM, in Equation 1 (and throughout the paper). See for example Kunstner et al., 2020 (Limitations of the Empirical Fisher Approximation for Natural Gradient Descent) for a discussion. 
2. I think, in the text after Equation 6, the authors meant 'low uncertainty' and not 'high uncertainty'? 
3. At the bottom of page 8 (and in Sec A.7.1) the authors use a network of half the size and see lower performance. I do not see how this helps understand if the experiment in Section 5 is sufficiently over-parameterised or not.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
