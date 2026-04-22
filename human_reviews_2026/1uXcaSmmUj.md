# Spiking Neuron as Discrete Gating for Long-Term Memory Tasks

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
Efficient long-term memory is important for improving the sample efficiency of Partially Observable Reinforcement Learning. In memory-based RL methods, the long-term memory capacity relies on the sequence models used in agent architecture. Two main approaches improve long-term dependency for sequence models, using linear recurrence and using information selection mechanisms such as gating. However, the sample efficiency of existing approaches remains low in long-term memory tasks. In this paper, we first present a saliency-based framework to illustrate why existing methods do not perform well on long-term memory tasks. Specifically, they cannot effectively filter out noisy information irrelevant to the memory task in the early stage of training. To this end, we design a novel linear recurrent module, in which the gating is controlled by spiking neurons. Spiking neurons output discrete values and can more effectively mask noise in the early stages of training, thus improving sample efficiency. The effectiveness of our proposed module is demonstrated on Passive Visual Match, a classic long-term memory task, and several different types of partially observable tasks. The code is attached in the supplementary material and will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This study focuses on gated neural memory models, noting that the nature of the gate of such models strongly influences the effective time-duration over which a model can learn. Specifically, with sigmoidal gates, true "gating" is nearly impossible and requires very large gating-weights, which hinders learning. Instead, the authors propose to use spiking neuron models combined with surrogate gradient learning to learn discrete gating in memory neuron models. The effectiveness of this approach is demonstrated on the Passive Visual Match (PVM) task and the POPGym. The approach excels on the PVM task compared to other gated networks, especially for longer durations, and also on the RepeatPrevious memory task.

### Strengths
The notion of learning discrete gating using surrogate gradients is novel and promising as far as I know. In the shown example, the presented approach excels exactly where the problem of noisy interference in a task is clear.

### Weaknesses
The writing would benefit from a more concise writeup of the introduction and clarity of the aims, as well as a more elaborate writeup of the tasks and results. This would also allow some of the other results, like short-term memory, to be included in the main text.

In particular, while the approach does indeed work well for the specific scenario it was designed for, inputs followed by noisy delays followed by a relevant output phase, the performance on general memory tasks is not convincing compared to the noted SHM approach in Table 3, while the presented method seems to perform similar to GRU, LiT and FFM for harder versions of the tasks . It would be informative to determine what makes SHM better. 

Gated-memory networks cover a large part of deep learning with many other tasks besides those shown. I would have like to see other memory tasks as well, for example  like the 1-2AX, saccade-anti-saccade tasks, the original "add" task.

### Questions
Figure 1 seems to suggest that the approach is mostly focused on a very specific trial type: a signal, noise, and then a go signal/output phase. Does the approach generalize?

In the description of the spiking neuron, I don't see a reset. Is there a reset? If not, what is the effect?

Can you provide more insight into when the gating helps and when not, compared to other methods? 

How does the proposed method perform on other long-term memory tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this work, the authors suggest that noisy distractor observations greatly reduce the efficiency of memory models. To mitigate this issue, they propose using memory with discrete (spiking) neurons.

First, the authors introduce a signal-to-noise analysis framework for memory. It uses a long-term task with known credit assignment, and is relatively straightforward. It takes the gradient with respect to the observations of the correct credit assignment divided by the total credit assignment.

Next, they perform a theoretical analysis of SNR for linear and nonlinear RNNs, and then gated and non-gated RNNs. They focus on the effect of SNR and the gradient for both cases. Importantly, they prove that gating enables an SNR of 1, which is not possible with a non-gated recurrent update. 

The authors go on to motivate a **discrete** and associative gating mechanism called the parallel spiking neuron. It uses a heaviside step gating mechanism surrogate gradient to enable backpropagation. They add stochasticity to mitigate an issue where a neuron can be stuck at 0, preventing gradient flow back through the RNN. They use this mechanism to construct a linear recurrent cell.

The authors calculate the SNR for their model with both soft and hard gating, as well as other prior work. Their method obtains greater SNR than other methods. Nonlinear RNNs demonstrate vanishing gradients, and gated linear RNNs report better SNR than nongated linear RNNs. Finally, discrete gating produces slightly better SNR than sigmoidal gating.

They report returns for other tasks, demonstrating their cell can learn more quickly than other models as the temporal dependence length increases. They perform further comparisons on POPGym tasks.

### Strengths
This paper is well written, theoretically sound, and novel. In particular, I find viewing RNN performance through the lens of SNR interesting, and it serves as good motivation for the proposed RNN cell. Table 3 is also particularly refreshing, demonstrating that the proposed method is designed for high-noise scenarios and does not necessarily achieve SoTA performance on standard benchmarks. Finally, the authors provide code for reproducibility.

### Weaknesses
More experiments are always beneficial. I think exploring other surrogate gradients would be interesting (why $\arctan$ instead of $\tanh$ or $\mathrm{erf}$?) I think an ablation for introducing noise would also be useful. I believe Stable Hadamard Memory [1] already performs such an ablation, but it would be interesting to see how important it is for discrete gating to work.

## References
[1] Hung et al., Stable Hadamard Memory: Revitalizing Memory-Augmented Agents for Reinforcement Learning

### Questions
I think figure 1 has a caption that is far too long.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a linear recurrent module that uses Spiking Neurons as discrete, input-dependent gating mechanisms. On the Passive Visual Match task with varying memory lengths, showing superior performance as length increases. Ablation studies demonstrating that their discrete gate outperforms a continuous (sigmoid) version. Based on the existing limitations and issues discussed below, I recommend reject.

### Strengths
* The work is the first to explore Spiking Neurons as a discrete gating mechanism within a linear RNN framework for memory-based reinforcement learning. This is a interesting architectural choice.
* It achieves excellent performance on specific long-term memory tasks, such as RepeatPrevious task in POPGym.

### Weaknesses
* It lacks comparison with existing state-of-the-art linear-time sequence models, such as Mamba2.
* It is missing an quantitative analysis of computational efficiency.
* It achieves SOTA performance only in a minority of environments; its performance in other environments is inferior to baselines, which limits the general applicability of the method.

### Questions
* The motivation for using Spiking Neurons is unclear. Combining a simple RNN or SSM with a discrete output could also achieve the effect of discrete gating. Why use Spiking Neurons, which may introduce additional training instability?
* Why does the method perform poorly in other tasks? Although the authors claim comparable performance to LRU on MuJoCo tasks, the implementation of LRU is much simpler and its performance is superior.
* Mamba's selection mechanism also performs information filtering. Does the authors' method filter noise more effectively than Mamba?
* The method performs poorly on tasks like 'Autoencode' and 'Battleship', underperforming SHM. Does this indicate that the method is only effective for tasks with explicitly defined noise phases?
* The paper mentions in Appendix E that training instability and Q-value divergence occurred in some runs, leading to the discarding of those results. Selectively discarding "invalid" runs, rather than reporting results from all random seeds, may overestimate the average performance of the method and makes the comparison with baseline methods unfair.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors proposed a new recurrent cell which uses discrete gating instead of continuous gating. The authors argued that this gating mechanism is inspired by spiking neurons and the reason why discrete gating is better than continuous gating function due to having large surrogate gradients. Continuous gating mechanisms like sigmoid function which can have small gradient values and can lead to vanishing gradients. The authors evaluated their mechanism using various memory and distraction tasks.

### Strengths
1. The writing of the paper is clear, and the figures are easy to follow. 
2. The mathematical equations in the main paper are also well presented and improved the readability. 
3. Results in Passive Visual Match is clear and understandable.
4. Figure 3 and 4 also produced convincing results.

### Weaknesses
1. Having Figure 3, Figure 4 and Table 1 together at the same spot makes it feel extremely crowded. Best to separate them the figures and tables. 
2. Why are results in the RepeatPrevious environment so much better than the baselines in the medium and hard category? 
3. The proposed method did not do well in the POPGym benchmark. Can you authors give some insights on why? Also, which methods are considered state of the art?

### Questions
1. What is the memory consumption with respect to memory size compared to other models? Is it more memory efficient?

### Soundness
3

### Presentation
3

### Contribution
2
