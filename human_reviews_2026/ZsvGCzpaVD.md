# Pretraining with Re-parametrized Self-Attention: Unlocking Generalizationin  SNN-Based Neural Decoding Across Time, Brains, and Tasks

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
The emergence of large-scale neural activity datasets provides new opportunities to enhance the generalization of neural decoding models. However, it remains a practical challenge to design neural decoders for fully implantable brain-machine interfaces (iBMIs) that achieve high accuracy, strong generalization, and low computational cost, which are essential for reliable, long-term deployment under strict power and hardware constraints.
To address this, we propose the Re-parametrized self-Attention Spiking Neural Network (RAT SNN) with a cross-condition pretraining framework to integrate neural variability and adapt to stringent computational constraints. 
Specifically, our approach introduces multi-timescale dynamic spiking neurons to capture the complex temporal variability of neural activity.
We refine spike-driven attention within a lightweight, re-parameterized architecture that enables accumulate-only operations between spiking neurons without sacrificing decoding accuracy.
Furthermore, we develop a stepwise training pipeline to systematically integrate neural variability across conditions, including neural temporal drift, subjects and tasks.
Building on these advances, we construct a pretrained model capable of rapid generalization to unseen conditions with high performance. 
We demonstrate that RAT SNN consistently outperforms leading SNN baselines and matches the accuracy of state-of-the-art artificial neural network (ANN) models with much lower computational cost under both seen and unseen conditions across various datasets. 
Collectively, pretrained-RAT SNN represents a high-performance, highly generalizable, and energy-efficient prototype of an SNN foundation model for fully iBMI. 
Code is available at [RAT SNN GitHub](https://github.com/YangYangYangYuqi/RAT-SNN).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes RAT SNN, a Spiking Neural Network integrating re-parameterized self-attention with a cross-condition pre-training framework. It aims to provide a neural decoding solution for iBMI that offers high accuracy, strong generalization capabilities, and high energy efficiency.

### Strengths
The paper's strength lies in its ingenious integration of various advanced technologies to address a critical and highly practical problem (efficiency and generalization bottlenecks in iBMI). It further demonstrates an excellent balance between decoding accuracy and computational efficiency through extremely detailed and convincing experiments.

### Weaknesses
I am not familiar with the iBMI field, so I can only raise the following weaknesses from the SNN perspective.

1.The method primarily represents a sophisticated combination and engineering optimization of existing techniques (e.g., RepVGG-style re-parameterization, spiking Transformers, and LIF layer with recurrent connections) within a specific application scenario, rather than proposing an entirely new algorithm or theoretical framework.Weakness

2.During the inference process of the attention module, how does matrix multiplication avoid MAC operations? Please provide a more detailed explanation.

3.Is the use of the attention mechanism effective on a network with only 4 layers? The paper lacks corresponding ablation experiments, such as comparing with a regular SNN that uses reparameterization.

### Questions
It is strongly recommended that the authors move the reparameterization effectiveness experiments from the appendix to the main body of the paper.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a re-parameterized self-attention spiking neural network (SNN) architecture for neural signal decoding.
The network employs spiking neurons to process cortical spike train signals, achieving lightweight structure and low-power consumption.
Furthermore, based on existing datasets, they construct a hybrid dataset that integrates cross-condition, cross-session, and cross-subject data, and conduct pretraining on this dataset to alleviate the generalization problem commonly faced by neural decoding models.

### Strengths
1. This paper provides a well-grounded analysis of the motivation, clearly identifying that the major challenge in current brain-signal decoding models lies in their poor generalization ability. It also highlights the low-power and lightweight model requirements in edge-side deployment scenarios. Building upon this, the paper logically introduces the use of spiking neural networks (SNNs), which are not only inherently well-suited for processing cortical spike train (CST) signals, but also naturally meet these computational constraints, forming a coherent and well-structured line of reasoning.

2. The experiments in this paper are thorough. A hybrid dataset was specifically designed to meet the requirements of pretraining, and extensive evaluations were conducted across multiple datasets.

### Weaknesses
1. The authors’ innovations in training methodology and dataset integration are indeed appealing; however, the network architecture design lacks sufficient novelty. For instance, the re-parameterization operation that fuses Batch Normalization (BN) with Convolution/FC layers has already been widely adopted in CNN-based SNN architectures. Similarly, the multi-branch convolutional structure inspired by RepVGG and its inference-time merging into a single convolution are also common practices in the ANN domain. In this paper, the authors merely apply these existing techniques, yet devote substantial space to describing the re-parameterization process, even emphasizing the “re-parameterized self-attention” in the title—while no genuinely innovative architectural design is evident in this aspect.

2. The variable notation is inconsistent. In the annotation of Figure 1, it is stated that $n_\text{input}$ represents the input dimension, which suggests that n should denote a dimensional quantity. However, later in the text, the authors write that “$n$ and $d$ represent the size of the LIF layers,” a statement whose meaning is unclear—particularly regarding which symbol refers to the number of layers. Furthermore, in Section 3.2, when analyzing the SNN self-attention mechanism, the paper claims that the computational complexity is reduced from O($n^2d$) to O($nd^2$). In this context, $n$ is evidently used to denote the number of tokens, while $d corresponds to the feature dimension, which contradicts the earlier definition. Overall, the inconsistent and ambiguous use of symbols significantly affects the clarity and readability of the paper.

3. The paper provides insufficient analysis of the neuron design. In the design of LIF neurons, the authors propose R-LIF, which introduces a recurrent structure across time to enhance the extraction of temporal information. Specifically, a temporal skip connection is added to integrate past spike firings and synaptic inputs. However, the paper does not provide a detailed analysis of the motivation behind this design, nor does it discuss any possible biological justification for it. Moreover, such an operation is not novel in the context of SNN neuron design.

### Questions
1. The authors state that the proposed model uses only four layers of LIF neurons, yet the paper does not provide a detailed description of the overall network architecture. Moreover, a four-layer configuration is considered very shallow for a typical SNN. The fact that the authors claim to achieve SOTA performance in various aspects—and even outperform ANN-based models—with such a shallow network raises doubts about the fairness of the comparison.

2. The authors present the iterative equations for LIF neurons in Equations (1)–(3), but the formulas appear to be incorrect. In Equation (1), it should be $V_{syn}[t]$.

3. In Section 3.2, the authors state that the re-parameterized channel attention consists of re-parameterized convolutional and FC layers. However, I disagree with this claim and believe it represents over-packaging of the concept. In my view, an attention mechanism should dynamically adjust its focus based on the input, helping the model emphasize important parts while filtering out irrelevant information. Moreover, channel attention is a well-established concept introduced many years ago, typically involving the use of convolution to generate attention scores across channels, which then guide the model to reduce redundancy and enhance salient features. Therefore, I do not consider a module composed merely of convolutional and fully connected layers to qualify as a genuine channel attention mechanism.

4. At the end of Section 3.2, the authors mention that global BN is applied to the output of the multi-head attention mechanism. However, I do not clearly understand what the authors mean by global BN. If their intention is to distinguish global BN from head-wise BN, then I wonder—haven’t previous works that employ self-attention already used what could be considered global BN by default? Moreover, has head-wise BN ever actually been used or reported in prior studies?

5. Throughout the paper, the authors repeatedly emphasize the concept of cross-condition when constructing datasets and conducting generalization tests. However, they do not clearly define what exactly a condition refers to. Is it a concept parallel to cross-subject and cross-session, or does it represent a broader notion that encompasses cross-subject, cross-session, and other variations?

6. What are the model parameters and the time step length used in the SNN in this paper? Additionally, how is the time step defined or segmented for the CST signals?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel spiking neural network model for neural decoding across heterogeneous sessions, called RAT SNN. Incorporating a re-parameterized architecture, RAT SNN achieves consistent improvement over leading SNN baselines and comparable with ANN baseline POYO. Overall, with SNN being a light-weight model, I believe this is a valuable addition to the community, hence I am leaning towards acceptance.

### Strengths
- Propose a novel SNN architecture including re-parameterized self attention and condition-specific batch normalization
- Proposed model show improved performance (in or out of distribution) over previous SNN models and comparable results as non-SSN transformer, which is known to have better decoding performance than SNN
- Scaling analysis shows that the proposed model can benefit from training over more and diverse datasets, which is a promising trait in the era of large-scale pretraining.

### Weaknesses
- The method section is in general difficult to follow, especially for someone not very familiar with SNN. It would make it easy to include some high-level description on what particular layers do, instead of laying out detailed equations and terminology
- Effect on dataset scaling is still not clear.. Despite having more data, cross-condtion training is sometimes worse than cross-subject training and even single sessionm in terms of decoding performance, which warrants further investigation.

### Questions
1. In appendix (line 740 page 14), the training and validation setups are mentioned to be overlapping 2-sec samples and continuous long sequence respectively. Were the same setup used for POYO as well? Because POYO by default seems to be working with 1-sec segment, with no length generalization in validation. I believe it is worth mentioning that when discussing POYO’s scores.
2. Could the authors comment on the potential reason cross-condition results are sometimes worse than cross-session or single session? For example, for monkey I and L, there is a ~2% improvement from SS to CS, while CC in monkey I is worse than CS. Considering that from SS to CS, the dataset increase is not that big (1 session vs 3 sessions), whereas CC has much more data. Would it be a data issue (maybe dataset is too diverse) or a model issue?
3. In addition, have the authors tried POYO in CC? It seems to be a bigger improvement from SS to CS for POYO, so it would be interesting to compare dataset scaling capability between ANN and SNN.
4. Being at the core of the architecture, it would be better to formally define re-parameterization-style layer.
5. Pruning is not properly defined in section 3.3, but there are multiple mentions of that in the results table. This could cause confusion.
6. In Table 1, perich dataset is included but I didn’t find the results on that dataset. Could the author point to where in the paper that dataset is evaluated?
7. In line 374 page 7, it is mentioned that RTT was used to test transferability to unseen sessions. How about Maze dataset whose results are also in Table 4? Could the authors comment on the experimental setup on that dataset, based on my understanding it couldn’t be in-distribution.
8. Please note that the provided code link is not valid, as all the files are shown not found
9. In equation (4), please mention specifically which index to sum over with respect to the feed-forward function
10. Please fix the numbering error in appendix

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors build on existing Spiking neural networks architecture using attention heads by introducing structural reparametrization in the self-attention blocks. The reparametrization comes with free batch normalisation, and the authors assign to each condition (ie. behaviour task) an independent batch normalisation during pre-training. The resulting model is then put to the bench using a corpus of extracellular neurophysiology recordings in non-human primate performing reaching tasks. 
The RAT-SNN model overperform previous SNNs and matches ANNs, at a lower computational expense (pruned version of 150k parameters compared to ~2M for ANNs). Furthermore, the pre-trained model is shown to systematically converge in fewer iterations than zero-shot attempts.

### Strengths
- Motivation is clear, SNN could enable generalizable models and be a lower power alternative than conventional ANNs for BMI applications.
- Performance on neural decoding is comparable to ANNs for this experiment, with an ablation study showing the reparametrization having a 0.15 increase on R2.
- The experiment follows a proposed framework for neuromorphic modeling benchmarking for model for baseline comparisons where applicable and for computational (Neurobench)

### Weaknesses
The experimental protocol is narrow in the sense that tasks are all very similar, using the same recording modalities in the same cortical areas of primates. One hypothesis could be that ANNs will generalize better than SNNs when the breadth of datasets will expand. Here maybe the experimental protocol is a bit too easy, without enough "unseen" subjects/tasks benchmarking. Even if ANNs were to take a strong scoring lead in such a case compared to SNNs, this wouldn't reduce the appeal of the presented method for real-time applications.
- I struggled to follow the experimental protocol and understand the terms single-session, cross-session, cross-condition: in all the performances of table 2 and 3 the pre-training always seems to include all datasets, while truly "unseen" benchmarks are only introduced in figure 5. Maybe a visual or a table recap of what is used in each training phase and in test would be more clear.
- Private dataset is... private, making any reproducibility or competition attempt impossible at the moment.


## Minor comments
- typo line 961: we employs
- the github repository is unaccessible

### Questions
- What experimental benchmark would make SNNs fail compared to ANNs ? Is there a more drastic protocol where pre-trained models are shown a completely unseen Maze task in an unseen subject where the ANNs take a strong lead ? Answering this question would make for a stronger paper.

### Soundness
3

### Presentation
2

### Contribution
3
