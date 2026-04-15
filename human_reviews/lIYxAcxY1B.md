# Understanding Sparse Feature Updates in Deep Networks using Iterative Linearisation

- Decision: Reject
- Scores: 3, 5, 5

## Abstract
Larger and deeper networks generalise well despite their increased capacity to overfit. Understanding why this happens is theoretically and practically important. One recent approach looks at the infinitely wide limits of such networks and their corresponding kernels. However, these theoretical tools cannot fully explain finite networks as the empirical kernel changes significantly during gradient-descent-based training in contrast to infinite networks. In this work, we derive an iterative linearised training method as a novel empirical tool to further investigate this distinction, allowing us to control for sparse (i.e. infrequent) feature updates and quantify the frequency of feature learning needed to achieve comparable performance. We justify iterative linearisation as an interpolation between a finite analog of the infinite width regime, which does not learn features, and standard gradient descent training, which does. Informally, we also show that it is analogous to a damped version of the Gauss-Newton algorithm --- a second-order method. We show that in a variety of cases, iterative linearised training surprisingly performs on par with standard training, noting in particular how much less frequent feature learning is required to achieve comparable performance. We also show that feature learning is essential for good performance. Since such feature learning inevitably causes changes in the NTK kernel, we provide direct negative evidence for the NTK theory, which states the NTK kernel remains constant during training.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes *iterative linearization* procedure of neural network training in order to get insights into feature learning process in those networks. The procedure consists of training rounds: having model parameters $\theta_s$ at the beginning of the round, the model is linearized and trained for $K$ iterations (having the gradients $\nabla f_{\theta_s}(x)$ during this time), and then the resulting parameter values $\theta_{s+K}$ are used as linearization point for the next round. Thus, assuming fixed total number of iterations $T$, the procedure interpolates between standard training of a full non-linear model $f_\theta(x)$ at $K=1$ (e.g. features are updated on each iteration) and the training of the fully linearized model $f_{\theta_0}^{lin}(\theta; x)=f_{\theta_0}(x)+\nabla f_{\theta_0}(x)^\top (\theta-\theta_0)$ at $K\geq T$. Also, The authors point to the analogy of iterative linearization with Gauss-Newton second-order optimization with parameter with damping parameter $\lambda$.   

Then, the authors empirically examine the performance of the iterative linearization on CIFAR10, with two values of $K$ considered in most of the experiments: $K=1$ and a large enough value $K\sim 10^4 - 10^6$ corresponding to the several $\sim 10$ future updates through the training. From these experiments, the authors conclude that making a few feature updates is sufficient to regain most of the performance of the full feature learning.

### Strengths
The main advantage of the paper is the proposed *iterative linearization scheme*. If accurately analyzed, it could exhibit quite an interesting 2D phase diagram of how performance depends on total training time $T$ and feature update frequency $K$.

### Weaknesses
Overall, the paper performs a very limited analysis of *iterative linearization*. The only contribution that is convincingly supported is the claim that a few $8-12$ updates are sufficient to the performance comparable to the full training with $K=1$. However, in its current form, this statement does not go much beyond a simple expectation of how interpolation between $K=1$ and $K=\infty$ should look like. An example of a more insightful result would be the whole curve of dependence of a given performance metric on $K$ in the region of its most significant change (probably $K\lesssim 100$ given the provided results). While formally equivalent, the number of feature updates $T/K$ seems to be a more convenient variable than frequency $K$.

The writing of the paper has a feeling of a "flow of thoughts" with lots of completely unsupported statements. Moreover, the paper doesn't have the *contributions*, making it difficult to distill the main claims and how they are supported. 

- The discussion of feature learning via pruning in the paragraph after eq. (2) is not clear in its current form.
- The description of iterative linearization in sec. 3 is confusing as it is different from the one described in the introduction (and which was quite clear). In particular, algorithm 1 does not fully describe the iterative linearization procedure, looking as being in the middle of its writing. Eq. (4) is referenced in sec. 3.2 as *feature learning step*, but it only tells that the features of linearized network are gradients. 
-  *a proxy measure of feature learning* defined in sec. 3.2. is simply a hyperparameter of the proposed iterative linearization scheme, and therefore is not able to measure in any sense the actual feature learning happening during optimization. Also, the discussion in the paragraph after Def.1 is not clear - e.g. what exactly do you mean by *absolute amount of feature learning*    
- It is not clear which linear model is considered in sec. 4.3
- Sec. 4.2 seems absolutely redundant as its only conclusion of poor generalization of linearized models is well known in the literature and already discussed by the authors in the introduction. 

(a small typo not affecting the quality of the paper) In figure 3, the neural network and its linearization seem to be swapped in the legend.

### Questions
N/A

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the training of deep networks using iterative linearization. In particular, the algorithm developed in this paper combines the neural network training and NTK linearization methods. The authors show that the feature learning induced by gradient steps is important, and demonstrate that the iterative linearized training can achieve comparable than standard training with fewer steps for feature learning.

### Strengths
* This paper proposes an iterative linearization training method that combines the NTK and neural network training.
* This paper performs numerical experiments to demonstrate the importance of feature learning.
* This paper performs experiments to show that feature learning can be made to be less frequent while a comparable test accuracy can still be maintained.

### Weaknesses
Overall, this paper provides certain interesting results regarding the effect of neural network training on generalization. However, the main weakness of this paper is that most of the claims are made by numerical experiments without rigorous theoretical justification. 

* The authors assume that the gradient descent step is to perform feature learning but do not give detailed justifications. In fact, gradient descent may also memorize the noise to fit training data points. In some special settings (e.g., very wide neural network, specifically designed initialization), gradient descent can also behave similarly to only learning random features.

* Since it is difficult to exactly quantify how much feature learning is performed in gradient descent, there might be some potential concerns by only comparing the number of gradient steps. It is possible that feature learning only happens in a small number of early steps of neural network training, then the "effective number" of feature learning steps could be much smaller than the total iteration numbers.

* As claimed by the authors, iterative linearization is similar to the Gauss-Newton methods, it would be good to also present the results for Gauss-newton method in the experiments.

* In deep neural network training, people typically use a larger learning rate for SGD/GD at least in the early stages, the authors may consider trying a larger learning rate in the experiments, rather than using 1e-3 for all experiments.

### Questions
* What happens if applying Gauss-Newton to train the neural network?

* What if using larger learning rates for gradient descent?

* Is there any theoretical explanation to back up the observations?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes ``iterative linearization'' as a midpoint between NTK and usual (S)GD update that enables feature learning. Specifically, the algorithm approximates the neural network by linearization every $K$ steps, and updates the proxy model for the next $K$ steps. It has been empirically shown that the comparable performance to usual (S)GD can be obtained even if we increase $K$, while $K=\infty$ (no feature update) is worse than finite $K$. The authors also explain that the connection to the Gauss-Newton algorithm of the proposed algorithm.

### Strengths
### The question is well-motivated.

Understanding the gap between NTK and usual (S)GD update is a very important problem. In the infinite width limit the NTK kernel is fixed during training, while practical (S)GD gradually changes the kernel. This paper decompose the role of (S)GD into the feature learning (= update of the kernel, happens every $K$ steps) and optimization on the fixed feature, and tries to identify how many ``times'' of updates of kernel is required to learn features.

### Equivalent performance to (S)GD can be achieved with remarkably few feature updates

The authors experimentally proved that large $K$ can still achieves comparable test accuracy to $K=1$, meaning that feature learning steps can be less frequent than optimization of the linear model. This result gives insights on how feature learning occurs during gradient-based training. 

### Connection to the Gauss-Newton algorithm

Iterative linearization is informally connected to the Gauss-Newton algorithm when $K$ is large. This gives some justification to the proposed algorithm.

### Weaknesses
### The idea of less frequent feature learning steps is not new

It has now become usual to consider layer-wise training of a two-layer neural network, where the first layer is trained with one step gradient and a large step size, followed by the linear regression of the second-layer parameters. The examples include [Damian et al. (2022)](https://arxiv.org/abs/2206.15144) and [Ba et al. (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f7e7fabd73b3df96c54a320862afcb78-Abstract-Conference.html). Therefore, in theory, it is not surprising that less frequent updates of the feature suffices to achieve high test accuracy.

### No convergence guarantee

If an algorithm claim itself as a proxy of (S)GD training, it is necessary to have a global convergence guarantee. This paper does not provides any convergence guarantee. Because of its connection to the Gauss-Newton algorithm, it could be possible to derive local convergence, but not global convergence as NTK and the mean-field analysis do. Note that, in order to theoretically justify their argument, we need to show not only the global convergence, but also that increase in $K$ does not slow down the convergence under certain conditions.

### Validity of the proxy measure (Definition 1)

When $K$ is not large enough, the next linearization step comes before the linearized network is fully optimized. Thus, I am not sure whether this value is a number of different features that must be passed through along the way.

### Discussion on the mean-field neural network

Although the network depth is limited to two-layer, the mean-field neural network is an important paradigm in the feature learning analysis. It attributes the optimization of neural networks to the optimization in the measure space and explain the feature learning dynamics [Chizat & Bach (2018)](https://arxiv.org/abs/1805.09545); [Mei et al. (2018)](https://arxiv.org/abs/1804.06561). Recently, several papers [Abbe et al. (2022)](https://arxiv.org/abs/2202.08658); [Abbe et al. (2023)](https://arxiv.org/abs/2302.11055); [Suzuki et al. (2023)](https://openreview.net/forum?id=tj86aGVNb3&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2023%2FConference%2FAuthors%23your-submissions); [Bietti et al. (2023)](https://arxiv.org/abs/2310.19793)) gave convergence and generalization guarantees on specific types of problems (especially polynomials) using the mean-field neural network.

### Questions
- Do you think it is possible to give some convergence guarantees on the proposed algorithm?
This does not necessarily for the general functions. 

- How can we explicitly know that the feature is learned by the proposed algorithm? Can you evaluate the parameter alignment?

- Is it possible to precisely evaluate the minimum required number of feature updates for some specific problems?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
