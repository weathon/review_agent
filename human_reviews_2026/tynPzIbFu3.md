# Towards Biological Continual Learning with Spiking Hopfield Networks

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Modern Hopfield networks are often viewed as biologically inspired associative memories, yet they lack the spiking dynamics and local learning rules that underpin real neural computation. In this work, we introduce a Spiking Hopfield Network (SHN) that incorporates discrete spike-based communication and a spike-timing–dependent plasticity (STDP) rule, enhancing biological plausibility while retaining the network’s capacity for online learning. To further support continual updates, we propose an Elastic Weight Consolidation (EWC)–inspired mechanism adapted to this local learning setting, reducing catastrophic forgetting. Together, these contributions yield a lightweight and biologically grounded framework that combines efficient memory retrieval with resilience to continual adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a spiking neural network based variant of modern Hopfield networks, with a motivation to design a thoroughly biologically plausible memory model. This spiking Hopfield network (SHN) model is essentially enhanced with adaptive LIF neurons and local learning rules through spike-time-dependent plasticity (STDP). Additionally as a key contribution, based on a first-spike-wins retrieval rule from the SHN memory, the model also introduces a temporal threshold gating (TTG) memory protection mechanism with multiplicative threshold decays. This is presented as an adaptation of the elastic weight consolidation (EWC) method in the context of a Hopfield-STDP framework, to demonstrate how minimal catastrophic forgetting with local learning rules can be achieved.

### Strengths
- The paper has a clear biologically inspired motivation in its design of spiking Hopfield networks.
- Proposed TTG update rule is a novel adaptation of EWC for STDP-based weight updates. Instead of EWC-like parameter penalization, locally gating the updates based on an adaptively decaying firing threshold is an interesting approach to mitigate catastrophic forgetting.

### Weaknesses
- The paper essentially aims to adapt and combine several existing machine learning mechanisms in a single framework: adaptive LIF neurons, STDP, and EWC, within a modern Hopfield network. Therefore from a technical perspective, the main contributions seem to be regarding the design and adaptation choices rather than fundamentally novel methodologies (except TTG).
- Clarity and reproducibility of the paper is a big limitation factor, and it almost appears like there are large gaps in the manuscript. There is no code or experimental details are not present, although there is a lot of page-space to explain the work more in detail.

### Questions
- The design of the experiments are not clear. There is not much information in the paper, and it is hard to understand what Section 4.1 really implies. No details in the appendix either. This is particularly important from a reproducibility and a self-contained manuscript perspective. No details or no code is available for any of the experimental results.
- More details on the architecture, hyperparameters, parameter initializations, etc. should be provided. For instance, how are the input spike train encodings facilitated?
- There could be a bit more effort in unifying the notation of the paper across sections. In Sec 2.1, ALIF neuron model is described with $\theta(t)$ indicating adaptive firing thresholds, whereas $\theta$ is used in Sec 2.4 with EWC to denote parameters to be updated in Eq (6), and in Sec 3.4 and Algorithm 1 the adaptive threshold with multiplicative decay is now denoted with another variable $v_i$.
- The manuscript could benefit from a better coverage of existing works in this area. There is no discussion on existing studies with SNN-based memory models. Some examples:

[1] “Memory-dependent computation and learning in spiking neural networks through Hebbian plasticity”, 2023.

[2] “STDP-based Associative Memory Model on Spiking Neural Networks”, 2024.

[3] “Toward a Biologically Plausible SNN-Based Associative Memory with Context-Dependent Hebbian Connectivity." 2025.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
They authors derive both learning and update rules for a spiking Hopfield network that allows ongoing learning without catastrophic forgetting, and is relatively insensitive to noise.

### Strengths
The authors address an important problem, at least from a neuroscience perspective: spiking Hopfield networks. The experimental results were very good, especially since they were learning in an online setting -- something that is difficult for associative memory networks.

The explanation of the individual parts was, by and large, understandable.

### Weaknesses
I might be missing something (not unusual), but I could not figure out what they actually did. The problems start with Eq 1,

tau_m du/dt = -u + RI.

However, we were never told what the current, I was. Therefore, I could not figure out what the architecture was, or how the data, x, was presented to the network. Each of the subsections by themselves made sense; I just couldn't figure out how they fit together.

### Questions
Please tell me what the architecture was. I can't guarantee it, but I'm almost positive I'll raise my score once I understand the architecture. Assuming I don't find something else wrong, which I doubt will happen.

Also, I wasn't clear on the training: how many times was each input presented? And was the MSE calculated on a training set?

Typo, line 359 (I think)o :  Figure 5 should be  Figure 4.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a Spiking Hopfield Network (SHN) that implements a biologically plausible version of  Modern Hopfield Networks by adding spiking dynamics and neural learning rules. To do so, they add the following to their network, which uses adaptive leaky integrate-and-fire neurons: (1) a simplified STDP-like learning rule, (2) a first-spike-wins retrieval mechanism for memory recall instead of softmax retrieval (3) temporal threshold gating for synaptic updates to prevent catastrophic forgetting. They compare SHN + first-spike-wins with SHN + Hopfield retrieval and show that the former does comparably on EMNIST, CIFAR-100, and MNIST+FashionMNIST tasks with sequential learning.

### Strengths
- The paper uses a variety of datasets (EMNIST, CIFAR, MNIST/FMNIST) , going beyond random binary patterns.
- Given that modern Hopfield networks are often connected to long term memory in brains, the problem of understanding what it takes to make them more biologically plausible is important to researchers studying memory in neuroscience.

### Weaknesses
- The use of winner-take-all dynamics for storage, but then first-spike-wins for retrieval feels inconsistent. Also first-spike-wins seems unusual as a biological mechanism. It's unclear to me if there's biological evidence for it.
- The only comparison model shown is SHN + first-spike-wins vs SHN + Hopfield retrieval. Feels like there should be a comparison to the standard, non-spiking MHN. Also, because the comparisons are only done between two versions of SHN and there's no visual examples shown of the retrieved memory, it's hard to interpret the MSE reported. 
- I would have liked more discussion about what neuroscience insights can be gained by the design of a spiking MHN. The lack of the discussion, combined with some of the arbitrary choices of design and the difficulty of interpreting the performance of the model (the two points mentioned above), makes it hard to evaluate the contribution made by the paper.

### Questions
- How does the model perform compared to MHNs?
- Can you discuss/justify more the choice of using two mechanisms for slot selection and retrieval (WTA and FSW, respectively)?

### Soundness
2

### Presentation
2

### Contribution
2
