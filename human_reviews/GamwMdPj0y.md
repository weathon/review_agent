# C-Adam: Confidence-Based Optimization for Online Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 5, 3

## Abstract
Modern recommendation systems frequently employ online learning to dynamically update their models with freshly collected data. The most commonly used optimizer for updating neural networks in these contexts is the Adam optimizer, which integrates momentum ($m_t$) and adaptive learning rate ($v_t$). However, the volatile nature of online learning data, characterized by its frequent distribution shifts and presence of noises, poses significant challenges to Adam's standard optimization process: (1) Adam may use outdated momentum and the average of squared gradients, resulting in slower adaptation to distribution changes, and (2) Adam's performance is adversely affected by data noise. To mitigate these issues, we introduce CAdam, a confidence-based optimization strategy that assesses the consistence between the momentum and the gradient for each parameter dimension before deciding on updates. If momentum and gradient are in sync, CAdam proceeds with parameter updates according to Adam's original formulation; if not, it temporarily withholds updates and monitors potential shifts in data distribution in subsequent iterations. This method allows CAdam to distinguish between the true distributional shifts and mere noise, and adapt more quickly to new data distributions. Our experiments with both synthetic and real-world datasets demonstrate that CAdam surpasses other well-known optimizers, including the original Adam, in efficiency and noise robustness. Furthermore, in large-scale A/B testing within a live recommendation system, CAdam significantly enhances model performance compared to Adam, leading to substantial increases in the system's gross merchandise volume (GMV).

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a variant of popular optimizer Adam, which cancels a specific batch updating by comparing the sign of $m_{t}$ with the sign of $g_{t}$, seeing line-14 of algorithm 1.
This method is intuitive and assumes that the $g_{t}$ is affected by noisy data and should be eliminated if the sign of $g_{t}$ differs from the sign of $m_{t}$.
Regarding the evaluation, this work conducted three types of small-scope optimization tasks, an image classification task with VGG&CIFAR10 under customized distribution shift and label noise, advertisement tasks with Criteo-x4-001 dataset under different various models and optimizers, and a real-world recommendation system task.

### Strengths
(1) Regarding Algorithm 1, momentum term $m_{t}$ is considered a more trustworthy signal than stochastic gradient $g_{t}$ under the distribution shifts or noisy data, which makes sense to me. Then, the alignment between those two signals potentially helps identify between the clear data samples and noisy data samples.

### Weaknesses
$\textbf{Weakness in method motivation}$

(1) The method is intuitive and not supported by theoretical results or experiments that verify the intuition. For instance, given a new random batch $x_{i\in B}$ and the corresponding ground truth label of clear data points and noisy data points, can you explicitly show that clear data points have the same sign with $m_{t}$ while noisy data points have a different sign with $m_{t}$.

(2) Given algorithm 1 and considering line-5 and line-14, intuitively, the $m_{t}$, i.e., the cumulative behaviors, is considered as trustworthy and is utilized to filter out untrustworthy $g_{t}$ according to the sign of the two values, which make sense somehow. 

However, $g_{t}$ is sampled from a small batch of data and contains noise naturally. Can you provide more insights into how line-14 of algorithm 1 is related to the natural noise of stochastic gradient?


$\textbf{Weakness in technical contributions}$

(1) This work assumes $g_{t} = \nabla f_{t}(\theta_{t-1})$ in the Notations in Section 3. Thus, this work assumes that it is a deterministic optimization problem. Along with the convex assumption, the theoretical result built upon those strong assumptions, i.e., Theorem 1, is less useful to reflect the practical performance of the proposed method.


(2) The problem is not well defined. This work mentioned the proposed method has the potential to handle distribution shifts and noise, however, what are the mathematical definitions of the distribution shifts you mentioned? And can you characterize it more formally and connect it with real-world situations? “rotating the data distribution” is confusing. 

To evaluate the robustness of optimizers to noise, well-accepted datasets such as CIFAR-10-C and CIFAR-100-C may be a better choice instead of rotating the images.


$\textbf{Weakness in evaluation}$

(1) As mentioned, the customized “distribution shifts and noise” cannot support the performance improvement of the proposed method.

(2) the small scope of the experiment cannot support the performance improvement of the proposed method. Specifically, most tasks do not apply diverse settings, such as VGG network on the CIFAR-10 for image classification, Criteo-x4-001 dataset for advertisement tasks.

(3) The experiment settings are not well presented. Please refer to the Questions.

### Questions
(1) Figure 4, Table 1, Table 2, and Table 3 do not report the variance. Do all those experiments run only once?

(2) Can you provide more details about how the learning rate is selected for the image classification task (3e-4) and recommendation system?

(3) Why was VGG selected as the model backbone for the image classification task? since VGG is not a common selection compared with such as ResNet18 and ViT.

(4) Since the performance improvement is quite marginal comparing the proposed method with the selected baseline Adam, Can AdamW be considered as another strong baseline?

### Soundness
1

### Presentation
1

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
This paper introduces Confidence Adaptive Moment Estimation (CAdam). It is a variant of the frequently used Adam optimizer, in which the model parameter update of a given weight is only applied if the current gradient g has the same sign as its exponential moving average m. If this is not the case, only the exponential moving averages m and v are updated. The purpose of this change is to improve the results in online learning if distribution changes occur and/or noisy samples are present. The method is evaluated in various scenarios.

### Strengths
- The idea of the CAdam optimizer is simple but clever. It is easy to integrate also in many other optimizers. Hence, the proposed change can be important in the end for a broad community.
- The paper is well written and structured. The experiments are well chosen to underline the advantages of the proposed method.

### Weaknesses
Major:
- Is the analysis of each experiment based on a single optimization trajectory (especially those in Sections 4.1 and 4.2)? For a reliable and convincing benchmark, multiple optimization trajectories should be started from different random initializations of the model parameters. Maybe I overread this aspect. However, then this should be highlighted more. You can also employ standard deviations of the trajectories to proof that CAdam is really consistently better than Adam.
- The differences in Table 1 between CAdam and Adam are really minor. For WideDeep, the results are even equal for the given rounded numbers. But still, the CAdam results are highlighted as best and they are promoted to be consistently better. I think you are overselling CAdam here and should adjust the interpretation of your results. (CAdam still shows better performance for noisy online learning tasks, which is an important contribution by itself.)

Figure 3:
- It is difficult to see the difference between Adam and CAdam. However, there is not much insight if the accuracy is below ~50%. Maybe one could just zoom in the region between 50 and 100%. In addition, there is much white space in the three plots on the right which can be reduced.

Figure 4:
- See comment on figure 3. Try to highlight the differences of the graphs by zooming in.

Text:
- The references to Figure 3 and 4 are incorrect.

Typos:
- page 4: "optimumm"
- page 7: "corrosbonding"

### Questions
1. Does CAdam have a higher possibility to be trapped in local minima, since the optimization trajectory is stopped as soon as the loss would increase in a step? How does it perform for functions with more than one minimum compared to Adam averaging over multiple runs starting with different model parameter initialization?

2. Do the plateaus in the optimization process make it more difficult to identify converged results? I am aware that this issue is less severe for online learning, but there might also be many tasks where you would like to pause online learning because the training data stream is not constant.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The authors of this paper propose a simple heuristic modification to the Adam algorithm to improve its performance when there is a shifting data distribution or noisy data — two common challenges in online learning for ad models/recommendation systems. The proposed algorithm, CAdam, checks if each coordinate of the gradient and the exponential moving average over the gradient share the same sign, and if not, stops updating the parameters until they align again. They also present a regret bound, seemingly closely following previous work on Adam. The authors evaluate CAdam on synthetic and real world data, including a live recommendation system, comparing against other common optimizers.

### Strengths
The core strength of the paper is the simplicity of the proposed modification to Adam, making it easy to deploy and reason about. Moreover, the problem of online learning under data distribution shifts and noisy data is a common problem in many real-world applications. If this simple modification can significantly improve performance under these conditions this would likely be of interest for the community.

The evaluation with A/B testing in a live recommendation system is also a promising way of demonstrating the practical benefit of the method, even if these results are not reproducible.

### Weaknesses
I’m leaning towards recommending rejecting the paper in its current form because of concerns about the experiments and the presentation.

My main concerns and questions regarding the experiments are:
1. I assume that in Table 1 you only show results for a single seed? Since the difference between the proposed methods and the baselines are very small, it seems crucial to make sure that the difference is not just due to random variation. I suggest running the experiment for multiple (e.g. 5) random seeds and also adding standard errors to the table. Also, when two numbers are exactly equal, you seem to only print the one corresponding to your methods in bold (e.g., WideDeep column, AMSGrad and CAdam). How do you decide what to print in bold?
2. Similarly, it would be good to show runs with multiple random seeds for the CNN image classification experiment (Figure 3 + 4).
3. Regarding the real world recommendation system results, I find it a bit hard to put them into perspective. While the scale of the experiment is impressive, it is hard to evaluate how significant the performance gains are. Is there a way to put them into context, e.g. by comparing the gains to other previous interventions? In any case, at least the gains seem to be consistent.

My main concern regarding the presentation is that despite the apparent simplicity of the algorithmic modification, its description is surprisingly unclear: in Algorithm 1, all quantities seem to be vectors, so the modification in line 14 implies that a dot product between $g_t$ and $\hat{m}_t$ is computed and, if it is smaller or equal to zero, $\hat{m}_t$ is set to the zero vector. However, the verbal description is talking about pausing the update for a single _parameter_. Also, equation (6) in appendix A is clearly stating that the comparison is applied coordinate-wise, effectively checking whether the sign of each coordinate of $g_t$ and $\hat{m}_t$  is the same. Could you please clarify this and make the description consistent throughout the paper?

If I can be convinced that CAdam is indeed a very simple way to improve performance in online learning tasks such as CTR prediction and recommendation systems and the presentation is improved, I’m happy to increase my score.

**Minor comments**

- The figures could be improved by increasing the font size and avoiding covering the letters with plot lines.
- You could consider condensing the paper a bit more, avoiding repetitive content, and removing sentences with little information.

### Questions
See the previous section. To reiterate the main questions:

1. Can you provide results for multiple random seeds for the experiments in Figure 3+4 and Table 1+2?
2. Can you try to put the results for the live recommendation system into perspective?
3. Can you clarify the algorithmic modification you make?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a simple heuristic modification of Adam to enhance its ability to handle distribution shifts and sample noise. The modification is skipping the update of Adam if the inner product of the momentum and the gradient is negative. The performance of the algorithm is theoretically analyzed based on (Reddi et al 2018). Experiments demonstrate better performance compared to Adam and other baselines.

### Strengths
The writing of the paper is clear, but besides that, to be very honest, I don't think any aspect of this paper is strong by the iclr standard.

### Weaknesses
This paper is in my opinion another quite straightforward hack on Adam. Since Adam was proposed ten years ago there have been so many hacks on it, but it's hard to say how much they really moved the field forward. I can see at least two issues behind this, which this paper also suffers from. 

- As Adam itself is heuristic, it is natural for someone to come up with many heuristic modifications based on intuitions. But do these intuitions actually reflect what's going on in the deep learning practice? No one really knows, which makes the foundation of these works quite shaky. 
- It's also not hard in general to cook up some settings where the proposed hacks can help, but do they *always* help? Answering this requires very comprehensive testing which the hacking papers typically lack. 

An acceptable paper of this type needs to stand out in at least one of the two dimensions. 

Regarding this particular paper, I would say the proposed hack is not surprising given the intuition the authors stated ("it's bad to have momentum and gradient pointing to different directions"), but I'm not convinced of this intuition, especially due to the stochastic nonconvex nature of deep learning optimization. The experiment settings are somewhat artificial, and the actual performance gain in the experiments is marginal. I'm also not convinced that the theoretical analysis adds sufficient value to the paper, as it doesn't show how the proposed hack *improves* Adam, let alone the known limitations of (Reddi et al 2018) itself which the paper builds on. 

I'm not completely denying the value of this paper, as some readers may still find it useful for their applications. But for a "competitive" conference like iclr the paper is quite far from enough.

### Questions
I don't have any question.

### Soundness
2

### Presentation
2

### Contribution
1
