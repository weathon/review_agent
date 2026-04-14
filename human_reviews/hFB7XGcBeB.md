# Asynchronous stochastic gradient descent with decoupled backpropagation and layer-wise updates

- Decision: Reject
- Scores: 5, 3, 3, 3

## Abstract
The increasing size of deep learning models has created the need for more efficient alternatives to the standard error backpropagation algorithm, that make better use of asynchronous, parallel and distributed computing. One major shortcoming of backpropagation is the interlocking between the forward phase of the algorithm, which computes a global loss, and the backward phase where the loss is backpropagated through all layers to compute the gradients, which are used to update the network parameters. To address this problem, we propose a method that parallelises SGD updates across the layers of a model by asynchronously updating them from multiple threads. Furthermore, since we observe that the forward pass is often much faster than the backward pass, we use separate threads for the forward and backward pass calculations, which allows us to use a higher ratio of forward to backward threads than the usual 1:1 ratio, reducing the overall staleness of the parameters. Thus, our approach performs asynchronous stochastic gradient descent using separate threads for the loss (forward) and gradient (backward) computations and performs layer-wise partial updates to parameters in a distributed way. We show that this approach yields close to state-of-the-art results while running up to 2.97× faster than Hogwild! scaled on multiple devices (Locally- Partitioned-Asynchronous-Parallel SGD).We theoretically prove the convergence of the algorithm using a novel theoretical framework based on stochastic differential equations and the drift diffusion process, by modeling the asynchronous parameter updates as a stochastic process.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper is in the general area of parallelization of DNN training, at the intersection of Machine Learning and Systems. Specifically, the paper is motivated by an analysis of backpropagation from the point of view of parallelism. Specifically, the paper starts by observing that backprop through a DNNs is a heavily sequential process (since we have to propagate inputs layer-wise, and then back-propagate the other way). The paper then wonders what would happen if we were to decouple, e.g., the forward-backward processes, by executing them on different threads, in order to allow for strictly increased parallelism relative to the base process.

### Strengths
I found the paper’s motivation and its initial discussion, up to Section 3.2, sound, interesting, and worthwhile. If anything, I think the observation that the max speedup factor should be at most 3 (since in standard backprop we are executing 3 matrix multiplies in sequence, and in your loose version they could potentially happen completely in parallel, on average) could be stated earlier and arrived at less circuitously.

### Weaknesses
Weaknesses: 

This point in the paper is unfortunately also pretty much where the quality of the submission starts to degrade. Specifically, I will list my concerns relative to the rest of the paper below, in sequence. (The authors may use this indexing in the discussion period.) 

1. Section 4 jumps directly into the experimental results. Here, the authors seem to meet a few challenges in terms of the viability of their method, which they seem to obscure or completely ignore. 

a. First is the fact that the sample complexity of their asynchronous algorithm (i.e. the number of data samples seen by SGD) is (probably, unless I misunderstand something, this is not discussed clearly) much higher than standard SGD with momentum: by this I mean that, due to asynchrony, their algorithm seems to need a lot more data passes to reach the same level of accuracy. 
For instance, curiously, the authors choose 300 epochs for ImageNet training of ResNet50, which is of only 73% for their algorithm. By comparison, e.g. the FFCV recipe using synchronous SGD+momentum on a single A100 would reach the same level in 30 epochs and probably in lower wall-clock time. A standard accuracy for this network is 76.15 (SGD + momentum, 90 epochs), if memory serves me well. 

The same curious behavior seems to occur for CIFAR10/100 experiments as well, where the accuracies presented significantly underperform what I seem to remember are the “right” values for standard training (e.g. around 93% on CIFAR10). Please see the FFCV examples or even Pytorch pretrained models for reasonably SOTA versions. 

b. More broadly, I find it really quite curious that none of the tables presented contains an SGD baseline. It is great that the authors are able to outperform LAPSGD/LPPSGD–and frankly what they are doing does seem like a better idea for parallelization–but that doesn’t mean that one should ignore any other baseline. Please add baselines. 

c. Generally, it is not so clear to me what are the broad implications of Section 4. OK, the results are better than LPP/LAP in terms of accuracy per time unit, but I thought that the general findings should be that the approach is better than “sequential” SGD? 

2. The theoretical analysis of convergence is under a very non-standard model. 
Specifically, the analysis seems to boil down to modeling SGD dynamics as a continuous process (which is fine), but then assuming that somehow the async SGD noise satisfies conditions which would allow it to be “linearized,” i.e. apply Taylor around the unperturbed parameters. It is really not clear to me why this is OK to do. Moreover, the authors don’t seem to be interested in providing a more standard analysis of SGD, e.g. along the lines of the Nadiraze et al. paper they seem to be well aware of. 
Specifically, the analysis allows us to grok the rationale for why the scheme might work, but seems to be primarily motivated by expediency rather than technical precision.

### Questions
Please see issues above for major questions. 

Some typos and minor observations: 

- L249: “for *a* different values”

- 250: Appendix B should be capitalized

- 251: sufficiently frequently

- 275: the A100 GPU you are using is an unusual choice for this task: this might be helping your algorithm since it has large computational power, so the degree of asynchrony between threads may be reduced relative to a mid-range GPU. 

- 292: Why did you choose 300 epochs for ImageNet? What happens with a lower (and more standard) number? 

- Please consider moving figures closer to where they are referenced. 

- Why no SGD baselines in tables? 

- Your convergence curves in Figure 2 seem to suggest that in fact there is quite significant noise due to asynchrony (the staircase pattern). Can you show a similar curve for regular training? Can you maybe vary the degree of asynchrony? 

Necessary Changes

1. Comprehensive baseline comparisons including standard SGD
2. Analysis of sample complexity increase and its practical implications, rather than sweeping it under the rug 
3. Validation on more diverse hardware configurations, ablations on key components, detailed scalability analysis
4. Clearer theoretical justification or an alternative, more standard, analysis approach

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors propose to parallelises backpropagation updates across the layers of a model by asynchronously updating them
from multiple threads.

### Strengths
Various experimental settings are explored, including CIFAR, ImageNet and LSTM.

### Weaknesses
1. The authors fail to show the converging model performance in Table 1. It does not validate that the proposed method can achieve similar performance as the backpropagation baseline.
2. The authors fail to show the converging model performance of the backpropagation baseline in Table 2 as well.
3. No training curves are provided.
4. Limited novelty. Decoupled forward and backward is not new. Check pipeline parallelism [1,2]. In particular, [2] provided a convergence analysis for async pipeline.
5. Figure 1 clearly shows that the forward activation and backward gradient does not come from the same training data. Therefore it's intuitionally not sound to show that it could match the performance of backpropagation. Note that [2] makes sure they come from the same data to conduct the convergence analysis.
6. Convergence complexity not provided.

[1] Huang, Y., Cheng, Y., Bapna, A., Firat, O., Chen, D., Chen, M., Lee, H., Ngiam, J., Le, Q.V. and Wu, Y., 2019. Gpipe: Efficient training of giant neural networks using pipeline parallelism. Advances in neural information processing systems, 32.

[2] Xu, A., Huo, Z. and Huang, H., 2020. On the acceleration of deep learning model parallelism with staleness. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2088-2097).

### Questions
1. Could you provide a clear illustration in Figure 2 like Gpipe to show how the forward and backward are pipelined?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
the papers proposes methodology to faster train NN using asynchronous updates
This is tackling the problem known in the literature as forward/backward lock
in particular the authors propose an asynchronous methodology for the training

### Strengths
The ideas are clearly explained.
The paper is easy to follow.
The paper covers an important (crucial from the  reviewer's humble opinion) topic with a high potential for impact.

### Weaknesses
This work presents a technique that is very close to techniques that have been known to the community for several years now (see https://proceedings.mlr.press/v70/jaderberg17a/jaderberg17a.pdf for instance).
This paper was followed by a long line of work on the topic.
This should be at least mentioned in the literature review.
Moreover, extensive numerical experiments comparing the proposed technique with SOTA asynchronous training techniques should support and highlight the strength/weaknesses of the algorithm. This is currently lacking in the paper.
Last, we would like to note that evaluating the techniques on resnet50 and lstm is not representative of the challenges faced when training modern ml systems (eg llms). The reviewer expects to see numerical results covering more of  such architectures.

### Questions
If the reviewer is not mistaken, the interactions between the threads in figure 1 would indeed lead to the creation of memory buffers.
The reviewers is curious whether the authors have investigated such "communication costs" of the technique.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
Given the observation that a backward pass usually takes twice as long as a forward computation in the Backpropagation algorithm, this paper proposes to perform the backward and forward passes in parallel threads instead of sequentially. Two parallel threads perform backward passes concurrently, while a third continuously perform forward passes and asynchronously update the model. Given that 2 threads are dedicated to backward passes, the cost of performing those computations is hidden behind the forward passes and the training time for the model simply amount to the time to perform forward computations without stopping on a given number of minibatches.

### Strengths
* **Novel use of multi-threading to hide the backward computations:** to the best of my knowledge, the way multi-threading is used here to hide the backward computation time in a "modularized" version of the back-propagation algorithm is not common. However, it is unfortunately *not* the only way to leverage multi-threading to this end (see the Weaknesses).
* **Theoretical analysis of the method:** convergence guarantees are given by modeling the training dynamic using stochastic processes.

### Weaknesses
* **Incomplete literature review, lacking comparison with existing literature:** Update locking is a key subject in model parallel (pipeline parallel) as it leads to GPUs idling instead of computing, yet no mention nor comparison with pipeline parallel methods is done. In model parallel, one thread per module is used, each thread performs a repetition of *“receive activation from previous module, forward pass, send activation to next module, receive gradient from next module, backward pass and update local module, send gradient to previous module”* cycles. See e.g.,[[Qi et al., 2023]]( https://arxiv.org/pdf/2401.10241 ), [[Kosson et al., 2021]]( https://arxiv.org/pdf/2003.11666 ), [[Yang et al., 2020]]( https://arxiv.org/pdf/1910.05124 ), [[Chen et al., 2019]]( https://arxiv.org/pdf/1809.02839 ), [[Huang et al., 2019]]( https://arxiv.org/pdf/1811.06965 ).
* **Idea of performing two backward computations per forward is not new**: in the “zero bubble paper” the same justification as eq (1) is done to reduce the size of the bubble in pipeline parallelism (see Fig.2 and Fig.3 of [[Qi et al., 2023]]( https://arxiv.org/pdf/2401.10241 ) ).
* **No study of the memory requirement of the method is performed:** Where are the activation terms $y_m$ of equation (1) in the Algorithm 1? These are are highly important in the memory footprint of your method, given that storing activations is the reason why the memory footprint is the largest at the end of the forward pass (see e.g., [the pytorch blog article on GPU memory]( https://pytorch.org/blog/understanding-gpu-memory-1/ ) ) and that GPU memory is one of the main bottleneck nowadays to train large models.

* **Weak experimental results:** The accuracies reached on CIFAR10 for ResNet18 are not really state-of the art (they should be $>95\\%$), and experiments with pipeline parallel algorithms are lacking, as well as with Decoupled Parallel Backpropagation [[Huo et al., 2018]]( https://arxiv.org/pdf/1804.10574 ).

* **Writing sometimes unclear or imprecise**:
  * **line 117:** *"Decoupled Parallel Backpropagation does full Backpropagation but uses stale gradients in the blocks to avoid update locking."* Indeed, but so does your method, so where is the problem?
  * **Overly complicated analysis of speedup (Section 3.3)**: the analysis could be made simpler by saying that your method basically performs forward passes back to back, completely hiding backward computation in parallel threads, so the training time is roughly $T_2 = bT$. On the contrary, standard BP performs forward and backward passes back to back, so the training time is $T_1 = (1+\beta )b T$ and the speedup is equal to $1 + \beta$. 

  * **Section 3.4 unclear:** I did not understand the meaning of the staleness. Are you referring to the length of the “pipeline step” (see, e.g. fig 2 of [[Kosson et al., 2021]]( https://arxiv.org/pdf/2003.11666 ) ), or the “bubble size” in pipeline parallel (see, e.g. fig 2. in  [[Huang et al., 2019]]( https://arxiv.org/pdf/1811.06965 ) ).
  * **typos:** *line 221*: “with with”, *line 194:* the second line of equation is incorrect.

### Questions
* In your experiments, how many data parallel replicas are used for Hogwild? And for yours?
* In your experiments, how does the memory usage compare between Hogwild and your method?

### Soundness
2

### Presentation
2

### Contribution
2
