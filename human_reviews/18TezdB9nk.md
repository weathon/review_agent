# Deep Neural Room Acoustics Primitive

- Avg Score: 4.25
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3, 6

## Abstract
Modeling room acoustics encompasses characterizing the sound propagation dynamics in enclosed 3D spaces and is useful in a variety of settings, including audio-visual simulations, embodied sound source localization, etc. Such dynamics are usually represented using one-dimensional room impulse responses (RIR). However, accurately estimating an RIR is often challenging as sound waves undergo reflections, diffraction, absorption, and scattering along the propagation path. In this paper, we propose a deep learning framework to learn a continuous room acoustic field, dubbed Deep Neural Room Acoustic Primitive (DeepNeRAP), capturing all sound propagation properties in a self-supervised manner; our framework allows the characterization of sound propagation from any source position to any receiver position. Our key idea is to allow two cooperative audio agents to actively probe the 3D space, one emitting and the other receiving sounds at varied positions -- analyzing these emitted and received sounds within our neural framework enables inversely characterizing the room scene acoustically. Our learning formulation is grounded in the physical principles of sound wave propagation, including the properties of globality, reciprocity, superposition, and independence. We present experiments on both synthetic and real-world datasets, demonstrating superior quality of our RIR estimation against closely related methods.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a new method for determining the RIR of a room for any source and emitter position. It relies on 2 agents which probe location pairs in a space by emitting a sine sweep and recording the result at the other agent. A model is then trained on a large number of recorded pairs in order to determine the RIR at a new query source and receiver position

### Strengths
- The paper is thorough and well written. It is easy to follow and justifies all the design choices and experimental setup. It also focuses on real physical constraints of RIR and LTI systems when designing the network.

- The experiment section is particularly thorough. There are experiments on both a synthetic and real dataset. Furthermore there are comparisons to existing methods and a number of ablation studies that show the usefulness of various parts of the network architecture. There are many metrics used for comparison including time domain and frequency domain metrics. Qualitative audio examples are provided to the listener in order to evaluate the output RIRs when compared to the ground truth and other methods

### Weaknesses
There are many limitations in the method that severely limit the usefulness/contribution to the audio community. The main issue is that the authors used two agents which probed 3000 different position pairs in order to learn the RIR for the room. This is a massive amount of data that needs to be collected and would be impractical in the real-world. This is in contrast to methods like "Few-Shot Audio-Visual Learning of Environment Acoustics" (https://arxiv.org/pdf/2206.04006.pdf) which use images as well as a couple audio measurements to estimate the room RIR. If we assume that you need 3000 training examples for a room and it takes a few seconds to collect each example, then you are looking at several hours of data collection needed to map out the room RIR.

The 2m distance between a train/test example is great for the synthetic dataset, but for the real world dataset, this isn't followed as the space itself is much smaller and probably would not facilitate a 2m distance between a test example and the nearest training example. This means that the real experiments probably had training examples in locations that were very close to the test examples. (Please correct me if you did also follow the 2m distance in the MeshRIR dataset)

There is not much discussion about how to choose these locations, except for that the authors say the goal is to reach as much of the room in as few steps as possible.

Furthermore, the authors assume that there are no other background noises in the environment. This makes it even harder to map out a room RIR in the real world. I would expect that machine learning systems would be capable of fitting an RIR, even in the presence of some amount of background noise, when presented with several hours of recordings in that environment.

### Questions
I would like to see some results or discussion on how the performance changes based on the number of trianing examples used. Can you train a decent model with 300 location pairs? 100?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper presents a methods that learns to predict the room impulse response (RIR) filter from the two source and receiver locations. While this kind of idea has been studied in the literature, most of them relied on the large RIR dataset and a supervised learning framework which may not generalize well. The proposed method performs this learning task in a pre-defined room, showcasing promising results in estimating RIRs.

### Strengths
- The RIR prediction mechanism appears to be mostly correct. It is indeed an alternative approach to directly estimating RIRs in a supervised manner, as the proposed method requires no direct recording of RIR filters, which could be a hassle. 

- The experimental results show that the proposed method can predict RIRs to some degree. 

- The paper is well-organized and easy to follow. Explanations on the LTI room acoustics physical principles are good. 

- The proposed multi-scale position-aware feature extraction method appears to be a clever idea that can capture the positional information of the scene at various granularity.

### Weaknesses
- The method assumes two agents that are time-synchronized. The synchronization must be very precise, while it might not be entirely easy depending on what kind of agents are used, especially if they are only wirelessly connected. In fact, the paper doesn't seem to try out the method using a pair of physical devices. Instead, even the real-world experiments are based on the pre-recorded RIRs. This somehow limits the proposed methods' applicability only to simulated environments. More elaboration is needed as to what kind of physical devices are going to be used. 

- The locations entirely ignore the height dimension, making the RIR estimation limited to a 2D room shape. I think the proposed method could still be meaningful with this limitation, but totally ignoring this third dimension must result in suboptimal performance, e.g., incorrect RIR estimation. The paper doesn't discuss this aspect at all. 

- The directionality of the speaker and microphone is ignored, too. Since in the real-world acoustic scene, a point source and omnidirectional microphones are rare, the proposed method must exhibit limitations when a directional source at the same location goes through different RIR filters. Again, this doesn't entirely negate the usefulness of the proposed method, but the paper simply lacks a discussion on this issue at all. 

- The setup completely ignores the effect of furniture and wall reflection rates; at least there is no description about them on the paper. 

- Finally, the PESQ scores need more explanation. According to the description, it appears that the experiments were done first by predicting the RIR filter and convolve the clean speech using the estimated filter. Then the comparison should be between the ground-truth reverberant speech using the ground-truth RIR filter and the estimated reverberant speech using the estimated RIR filter. I'm not sure if PESQ is defined for this case, because typically it's to compare clean speech and its estimation. If the authors compared the estimated reverberant speech with clean one, then the comparison of PESQ scores is pointless. Needs more clarification here. 

- Other objective metrics that compare RIR filters are not interpretable. For example, a large SNR value is good, but, for example, a 6dB SNR on RIR reconstruction doesn't necessarily give any intuition about the quality. PESQ or more speech-related ones might be better, but as mentioned earlier, that part needs clarification.

### Questions
- Is there any way to provide more information how the directionality of the speaker and microphone plays?

- Wouldn't it be better to use SNR to compare  the ground-truth reverberant speech using the ground-truth RIR filter and the estimated reverberant speech using the estimated RIR filter?

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes using a neural network to reproduce the room impulse responses for a given arbitrary source-sink positions pair in an already seen and extensively measured room.

### Strengths
The paper gives some introduction to the problem it is trying to solve.
The multiscale feature extraction is a nice idea.

### Weaknesses
The approach description is relatively hard to follow since important details are left out (e.g., obtaining the learnable room acoustic representation).
The motivating use-case seems also very special as it is not clear where are the cases where a room can be measured exhaustively as presented but at the same time few position are kept out. 
I am not 100% sure, however, it seems that the model is trained to be room specific. The setup used is unrealistic. The authors call the approach self-supervised where this term is being used in a context much different from the common definition of the term.
The signals being used for measurement are idealized sine sweeps. In fact the cited signals are inverted sweeps making the estimation of the impulse responses trivial. 

It is unclear why the authors say that collecting RIRs is difficult while any teleconferencing system in the world is continuously measuring  impulse responses implicitly for the purpose of echo cancellation without the use of intrusive measurement signals such as sweeps. 

The benefit of using a DNN is actually not clear as the model is learned on 3000 impulse responses even if no test measurement is . The model is overfitting on a single room and the figure II is misleading as it shows parts of the measurement.

### Questions
What is meant with a learnable room acoustic representation? What do the k-dimensional features for each grid point represent? It is written each entry is associated with a learnable small feature of size k. It is not clear what these features are nor how they can be obtained.

There is a mention of not involving any sort of data annotation especially RIR. How is this true given the input (data D) contains both microphone and loudspeaker signals? Given the noiseless scenario and in the frequency domain the impulse response is actually obtained by a frequency bin division of the microphone signal by the loudspeaker signal.

Is self-supervision referring to the fact that the input is both the system input and output and the network is trying to solve a regression problem?

In the experiments section, it is mentioned that multiple datasets are used, is the training room-specific?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes DeepNeRAP, a deep learning framework to learn room acoustics by using two agents to probe the space. The source agent emits sounds and the receiver agent collects them from different locations. This allows training a neural network model without needing detailed prior acoustic knowledge of the room. The model takes source and receiver positions as input and outputs a neural room impulse response capturing sound propagation between them. It is trained in a self-supervised manner by enforcing the predicted response to match the actual recorded sound. The network incorporates principles of room acoustics to make the learned representation physically meaningful. Experiments on large synthetic and real rooms show it outperforms recent methods in predicting room acoustics and sound, despite not having detailed acoustic knowledge or massive impulse response data.

### Strengths
The self-supervised learning approach removes the need for massive labeled impulse response data like past methods. This could enable training for new spaces more efficiently.

Incorporating principles of room acoustics directly into the network design makes the model interpretable and ensures physical plausibility.

Achieves state-of-the-art results on RIR and sound prediction tasks compared to recent learning-based methods. 

Source code is provided which greatly enhances the utility of the paper.

### Weaknesses
The limitation described in the last paragraph about assuming the room to be noise free seems severe and not discussed. No real room is noise free, so this should be addressed.

There is no end-to-end application of this method that shows it has actual value. For example, show it works to improve a speech enhancement model like noise suppression or echo cancellation. This will also address the issue #1 above. 

The PESQ results are very low. When I listened to the clips I think that is because the signal amplitude is very low. I think AGC should be uniformly applied to all method before PESQ is applied, which will make the result more sensitive and useful.

### Questions
1. What is the impact of the noise free room assumption?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
