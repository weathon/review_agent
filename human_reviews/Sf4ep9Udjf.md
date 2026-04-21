# P-SPIKESSM: HARNESSING PROBABILISTIC SPIKING STATE SPACE MODELS FOR LONG-RANGE DEPENDENCY TASKS

- Avg Score: 6.75
- Decision: Accept (Poster)
- Scores: 8, 5, 8, 6

## Abstract
Spiking neural networks (SNNs) are posited as a computationally efficient and biologically plausible alternative to conventional neural architectures, with their core computational framework primarily using the leaky integrate-and-fire (LIF) neuron model. However, the limited hidden state representation of LIF neurons, characterized by a scalar membrane potential, and sequential spike generation process, poses challenges for effectively developing scalable spiking models to address long-range dependencies in sequence learning tasks. In this study, we  develop a scalable probabilistic spiking learning framework for long-range dependency tasks leveraging the fundamentals of state space models. Unlike LIF neurons that rely on the deterministic Heaviside function for a sequential process of spike generation, we introduce a SpikeSampler layer that samples spikes stochastically based on an SSM-based neuronal model while allowing parallel computations. To address non-differentiability of the spiking operation and enable effective training, we also propose a surrogate function tailored for the stochastic nature of the SpikeSampler layer. To enhance inter-neuron communication, we introduce the SpikeMixer block, which integrates spikes from neuron populations in each layer. This is followed by a ClampFuse layer, incorporating a residual connection to capture complex dependencies, enabling scalability of the model. Our models attain state-of-the-art performance among SNN models across diverse long-range dependency tasks, encompassing the Long Range Arena benchmark, permuted sequential MNIST, and the Speech Command dataset and demonstrate sparse spiking pattern highlighting its computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present a new model that combines state space model (SSM) layers with stochastic (Bernouilli) spiking layers. They test it on various long sequence benchmarks.

### Strengths
Very good accuracy, outperforming all other spiking proposals and even most non-spiking ones (all except S4).

### Weaknesses
* IMHO, the main weakness is that this network is not fully spiking, and it's not clear if it could be implemented on existing neuromorphic chips (e.g., Intel Loihi). The temporal convolution of the SSM (eq  6) could probably be implemented by delays. But the gelu in eq 10, and the non-local normalization (in ClampFused layers) are problematic. I suggest the authors discuss these issues and possible solutions. Also, I think the authors should cast their network as "Partial SNN" in Table 1-3

* IMHO psMNIST is not challenging enough (SOTA is nearly 100%, which may mask differences between approaches); I would recommend trying as well on sequential CIFAR10/100 (much more challenging).

* The authors seem to ignore the sliding PSN neuron (http://arxiv.org/abs/2304.12760), which has many similarities with their model. Both use spikes. Both use temporal convolutions instead of stateful units and for this reason, both avoid BPTT and are parallelizable. A comparison with the PSN (both qualitative and quantitative) would be useful.

Minor points:

* "A is a parameter controlling the evolution" -> "A is a matrix controlling the evolution"

* Fig 1 b: the y dimension is useless here. I would recommend plotting the curve y = p(t) instead of a heat map.

* L184: "since probability p[t] in [0, 1]." I think in continuous time, p is a pdf, so it could be >1

--

POST REBUTTAL:

My main concerns have been addressed. I raised my score to 8.

### Questions
* I think the authors could bypass the SpikeSampler layer and send the real-valued probabilities directly to the next layer. Do you confirm? Have you tried? Of course, the computational advantages of spikes would be lost, but I expect an increase in accuracy, and it would be interesting to quantify it. 

* Would it be possible to encourage even sparser activity via some additional term in the loss function?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
To exploit the energy-saving potential of spiking neural networks (SNNs) and to overcome the bottleneck of SNNs in long-range dependency tasks, this work proposes a scalable probabilistic spiking learning framework based on state space models (SSMs). Unlike the other model that simply concatenates SSMs with the leaky integrate-and-fire (LIF), this work incorporates SSMs into the membrane potential update of spiking neurons, and innovatively proposes P-SpikeSSM in a bid to address the performance issues of LIF and overcome its inability of parallel computations. To build a scalable architecture, this work designs several modules, including SpikeSampler, SpikeMixer, and FuseClamp. With the contribution of these modules, P-SpikeSSM outperforms other SNN models in various long-range dependency tasks. In addition, P-SpikeSSM shows a large improvement compared to its ANN counterpart in energy efficiency analysis.

### Strengths
1. The motivation for this work is clear and sound, and the presentation of the methodology is very detailed and easy to follow.
2. P-SpikeSSM integrates the recently high-profile SSM into spiking neural networks from a novel perspective, addressing the problem that the commonly used LIF cannot handle parallel computations. It may provide new insights for future research on the combination of SSM and SNN.
3. By incorporating P-SpikeSSM with some well-designed modules, the proposed model achieves better performance than other spiking networks and some non-spiking networks.

### Weaknesses
Major points:
1. P-SpikeSSM is an interesting exploration, but the elements that play a key role in it seem to be just an application of what SSM proposes, such as temporal convolution to enable parallel computation.
2. The network cannot be considered a fully spiking neural network, and is best called a partial SNN because of the use of gelu in SpikeMixer Block (Eq. 10). In addition, the introduction of gelu greatly increases the difficulty of deploying this network on neuromorphic hardware.
3. Considering that the network is a partial SNN, it performs only marginally better than BinaryS4D, which is also a partial SNN, on some of the LRA benchmark tasks, and even worse on others (Table 2). Given that P-SpikeSSM and BinaryS4D are both based on SSM, it's debatable whether their advantages over other transformer-based networks mostly stem from SSM.
4. For the experiment on the Speech Command dataset (Table 3), Spiking LMUFormer achieved 96.12% accuracy on the full 35-category dataset, whereas this work is only validated on a subset of 10 categories and is not compared to some recent advanced baselines [1, 2].

Minor point:
1. There is a comparison of energy consumption with the ANN counterpart, but no comparison of the number of parameters with other baselines, which is an important metric influencing performance.

[1] Zeyu Liu, Gourav Datta, Anni Li, and Peter Anthony Beerel. Lmuformer: Low complexity yet powerful spiking model with legendre memory units. ICLR. 2024.

[2] Alexandre Bittar and Philip N Garner. A surrogate gradient spiking baseline for speech command recognition. Frontiers in Neuroscience. 2022.

### Questions
1. The benefits of stochasticity? Would there be better or more consistent performance if probabilistic spike sampling is replaced by spike firing at a fixed or learnable threshold?
2. How can the network be trained with the removal of the surrogate gradient (Table 5)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces P-SpikeSSM, a stochastic spiking neural network (SNN) that replaces the conventional scalar LIF neuron membrane potential with an SSM linear dynamical system. The multi-dimensional of the SSM increases the expressivity of each neuron. The SSM inside each neuron is read out through a linear map, the output of which is then clamped between 0 and 1 to be interpreted as a probability. This probability is used to generate spikes stochastically during each timestep according to a Bernoulli distribution. 

The inspiration for using a multidimensional SSM instead of a scalar membrane potential in the spiking neurons is the increased temporal-dynamic expressivity afforded by SSMs. The neural SSMs are initialized using HiPPO matrices, inspired by prior SSM work. 

The reasoning for introducing a probabilistic spiking model is that it affords convenient parallelizability. The authors use the probability as a surrogate operation to propagate gradients through the probabilistic spiking layers.

The authors show through experiments that the P-SpikeSSM performs better or competitively with transformer and other SSM baselines, and the authors perform ablation studies, showing all components of their model discussed in the paper contribute to task performance.

### Strengths
Significance.
Replacing the membrane potential of a spiking neuron with an SSM is a well-motivated modification, and the encouraging results presented in this work suggest this modification could be broadly valuable in efficient neural network research.

Originality.
This is the first work that I’ve seen to replace the membrane potential in and SNN with an SSM. 

Quality.
The work includes ablation studies to demonstrate the necessity of each component of the model.

Clarity.
The work builds up the model in a step-by-step manner with sufficient details so that the reader can readily understand the motivation for each step.

### Weaknesses
On line 141 and the caption for Figure 2, has it truly been proven to be the case that LIF neurons cannot be parallelized? What is the distinction between a non-linear LIF neuron and a linear LIF neuron? (To me, all LIF neurons are non-linear because they have reset terms.) Do efficient algorithms exist for LIF neurons? E.g., see “EXODUS: Stable and Efficient Training of Spiking Neural Networks” by Bauer et al., or the work SLAYER that they reference?

On line 77, the authors claim the surrogate operation Expection[S_t] is novel. I believe such a surrogate operation has already been introduced. E.g., see “Automatic Differentiation of Programs with Discrete Randomness” by Arya et al. NeurIPS 2022.

In Table 1, it is unclear to me the model size/computation budget of these various models. I could imagine model size/compute budget is a key reason why one model would outperform another, so I would like to clearly understand how this table presents a fair comparison. E.g., could you show me how the P-SpikeSSM model is iso-parameter-count?

On line 102, the authors imply that P-SpikeSSM is a “fully” spiking model. While “fully” spiking is not defined in the literature, one might argue that P-SpikeSSM is not fully spiking because each spiking neuron contains a non-spiking SSM within it. I might suggest dropping the “fully” adjective. On line 415, “fully spiking” is implied to be defined as “not requiring floating point MAC operations,” but I am now confused regarding how P-SpikeSSM is fully spiking even though there is an SSM inside every neuron. 

I struggle to understand the “Analysis of Energy Efficiency” section. In particular I do not see how the computational cost of the n SSM state variables inside the N P-SpikeSSM neurons in each layer are accounted for. To me, it seems a factor of n is missing.

I noted in line 810 “Memory Footprint” that the SSM parameters in each neuron in a layer can be shared. Are the parameters of the SSMs shared in your experiments?

### Questions
I asked the most salient questions above in the “Weaknesses” section. The questions that follow are more minor.

On line 57, the authors state that long-range dependency tasks are largely unexplored in the spiking domain, but then the authors go on to cite works on SNNs with SSMs in paragraph starting on line 139. What do the authors mean by ‘largely unexplored’? I suppose ‘largely unexplored’ is a subjective assessment, however. I might suggest more consistent language.

I would suggest clarifying that the “Analysis of Energy Efficiency” section is treating parallel inference specifically.

Is there anything that can be said about the hyperparameter selection process used in this work, to help justify that the ablation study experiment results are not an artifact of certain hyperparameter choices?

While probabilistic spiking affords convenient parallelization, is there anything that can be said about the drawbacks of or parallelizable alternatives to probabilistic spiking?

Thank you for this fascinating work.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The article proposes a framework for learning in long-range sequence tasks. It introduces a series of tactics to address the associated challenges. Firstly, the authors employ a stochastic sampler based on a State-Space Model (SSM) neuronal model in conjunction with a differentiable function. Additionally, the paper provides comparison results to demonstrate the effectiveness of the proposed approach.

### Strengths
The paper explores the differences between two neuron models: the conventional Leaky Integrate-and-Fire (LIF) model and a modified version that employs sampling-based techniques. Additionally, the authors present comprehensive comparison results to highlight the effectiveness of their method.

### Weaknesses
The authors do not provide strong evidence that the sampling-based technique outperforms the conventional LIF neuronal model. The absence of an input/output step may also hinder the flow of internal dynamics information. I suggest that they compare their architecture using both approaches to clearly demonstrate the benefits of their method. While it is evident that their approach facilitates the parallelization of the algorithm, this does not necessarily imply that the solution can be obtained faster, as more steps might be required to achieve the appropriate spiking rate.

### Questions
n/a

### Soundness
2

### Presentation
3

### Contribution
2
