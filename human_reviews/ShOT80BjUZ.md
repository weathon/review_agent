# SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 3, 1

## Abstract
As the size of large language models continue to scale, so does the computational resources required to run them. Spiking Neural Networks (SNNs) have emerged as an energy-efficient approach to deep learning that leverage sparse and event-driven activations to reduce the computational overhead associated with model inference. While they have become competitive with non-spiking models on many computer vision tasks, SNNs have proven to be more challenging to train. As a result, their performance lags behind modern deep learning, and until now, SNNs have yet to succeed at language generation on large-scale datasets. In this paper, inspired by the Receptance Weighted Key Value (RWKV) language model, we successfully implement `SpikeGPT', a generative language model with binary, event-driven spiking activation units. We train the proposed model on two model variants: 45M and 216M parameters. To the best of our knowledge, SpikeGPT is the largest backpropagation-trained SNN model to date, rendering it suitable for both the generation and comprehension of natural language. We achieve this by modifying the transformer block to replace multi-head self-attention to reduce quadratic computational complexity $\mathcal{O}(T^2)$ to linear complexity $\mathcal{O}(T)$ with increasing sequence length. Input tokens are instead streamed in sequentially to our attention mechanism (as with typical SNNs). Our experiments show that SpikeGPT remains competitive with non-spiking models on tested benchmarks, while maintaining 32.2$\times$ fewer operations when processed on neuromorphic hardware that can leverage sparse, event-driven activations.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces SpikeGPT, which is a language model based on Spiking Neural Networks (SNNs) designed to reduce the computational resources and energy consumption of large language models. The paper describes the architecture of SpikeGPT and its performance in natural language generation (NLG) and natural language understanding (NLU) tasks. It also includes an ablation study to investigate the impact of different architectural modifications on the performance of SpikeGPT.

### Strengths
SpikeGPT is designed to enhance the energy efficiency of language models by utilizing spiking neural units to achieve sparse and event-driven activations, thereby reducing the consumption of computational resources. This is of paramount importance for the sustainability of large-scale language models.

### Weaknesses
Authors can add a discussion about SpikeGPT's performance on specific hardware and its adaptability to different hardware platforms. The novelty is limited, I just did not see some crucial difference between general GPT and the proposed model about models and training methods. Another more important thing is that I did not see a detailed power consumption analysis on neuromorphic hardware, this work is different from other SNN based studies, large scale model must applied to true neuromorphic hardware which could analyze the true power consumption. As illustrated in table 2, the parameters of the proposed model are more than other models, I did not see the advantages of the spiking version of GPT.  Furthermore, it would be beneficial to include additional datasets to demonstrate the model's effectiveness. Ablation experiments can be extended to include other evaluation metrics mentioned in the tables, such as accuracy and complexity, to provide a more comprehensive assessment of the model's efficacy.

### Questions
Are there plans for further research and improvements to enhance SpikeGPT's performance on various tasks and datasets? Are there any case studies or experimental data regarding the deployment and performance of SpikeGPT in real-world applications?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposed a generative language model with sikes named spikegpt with backpropagation method. The authors argued the proposed model is the largest one with BP based training method. The authors also want to reduce quadratic computational complexity to linear complexity.

### Strengths
1. This paper is well-written.  
2. Spiking-based large language model is very important and can effectively promote the research of SNN. Ideal SpikeGPT models can greatly reduce parameters and energy consumption.

### Weaknesses
1. The main innovation points are limited, this paper wanted to grasp this concept, but it didn't delve deeply into it. The network structure is more likely a hybrid one with transformer, not a true spike one.
2. The training methods for this model are not described in a clear way, so I did not see more novelty in this work. 
3. The blocks in SRFFN seem too simple, I mean the authors should consider more about other gates such as forgetting gates. 
4. During the training and inference phase, I did not see the true contribution of this model in NLG  and NLU.
5. In table 1, the energy consumption used by authors is wrong, this model is not completely a spike based one (just inputs are spikes), in hence, used FLOPS and methods (Rathi and Roy 2021) are not fair. 
6. Table 2 reports the complexity and parameters of some models, just from parameters, I can't see where the advantage lies (transformer vs spikegpt).  Why don't I use another spike transformer model?
7. The authors make the comparison between spikegpt and gpt-2, gpt 2 is not the current one, so it is not an efficient comparison.

### Questions
The authors could reference the detailed weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors propose SpikeGPT, which is a hybrid variant of RWKV architecture that employs spiking linear transformations and some float-point operations. In a rough estimate, SpikeGPT has an energy efficiency advantage of about 32 times over vanilla GPT. In terms of performance, SpikeGPT outperforms the LSTM backbone and is comparable to some simplified variants of the Transformer, but falls behind the vanilla Transformer.

### Strengths
1. The paper is easy to follow.

2. The authors replace the linear layer in RWKV, which has the highest computational overhead, with a spiking layer. As a result, SpikeGPT is about 32 times more energy efficient compared to the vanilla GPT.

### Weaknesses
1. SpikeGPT introduces some float-point operations including float-point multiplication, division, and exponentiation. This makes SpikeGPT different from traditional spiking neural networks. Although these operations are not dominant in terms of computational overhead, this part of the operation will not be sparse and event-driven, which puts a higher demand on possible application scenarios that must support this hybrid computational paradigm.

2. The normalization used in SpikeGPT is unclear. As shown in the left subplot of Fig. 1, Add&Norm is used in SpikeGPT, but it is not specified in the text whether it is a layernorm or a batchnorm (or other normalization). Batchnorm is generally used for SNNs because it can be merged into linear or convolutional layers, whereas layernorms, which are widely used in NLP, cannot be merged, which results in extra floating-point multiplications and is not applicable to typical SNNs. Based on the code given by the authors, I would guess that the authors use layernorm, but the authors don't provide any analysis of the computational overhead of normalization and the effect of layer normalization on spike inputs in the main text. In addition, Eq. 3 and Eq. 4 do not mention the normalization.

3. Estimates of energy efficiency in the main text are rough. First, the energy consumption estimates in Table 1 assume that all spiking neurons have the same firing rate of 0.15. The authors do not explain how the 0.15 is derived here, but it is safe to assume that this is only a rough estimate, since the firing rate of neurons in different layers and even different channels of the network should be different. Second, the energy consumption estimates in Table 1 do not seem to take into account the differences in energy consumption for different kinds of floating-point operations. In spiking RWKV, the element-wise float-point, division, and exponentiation (including sigmoid) should have different energy consumption with the MAC.

4. Missing further analysis of $\rm {ReLU}^2$ activation. In page 7, line 5, section 3.7, the authors state that "While $\rm {ReLU}^2$ activations are not binary, they induce dynamical sparsity."  However, the authors do not give an estimate of this sparsity. In addition, the energy estimates in Table 1 do not take into account the energy consumption of $\rm {ReLU}^2$, nor do they use the sparsity of $\rm {ReLU}^2$ with $\rm E_{MAC}$ to estimate the energy consumption of the linear transformation $\bf M_s$.

5. The contribution on improving RWKV is unclear. The authors only review vanilla self-attention in the main text, but instead of reviewing the RWKV structure they put this part in the appendix. In addition, the authors do not clearly indicate which parts are the authors' main contributions and which are RWKV contributions in the main text. Therefore I need to carefully compare the content about RWKV in the appendix to determine the author's main contribution. As far as I understand from the main text, the author's contribution to the improvement of the RWKV structure is as follows:

   - They add a spiking neuron layer before the linear layer (or equally, add a spiking neuron layer after block output) to reduce the computational overhead of the linear layer.

   - They modify the mechanism of the positional weight decay in RWKV.

### Questions
1. (Weakness 2) What kind of normalization is used in SpikeGPT? Please add it to Eq. 3 and Eq. 4.

2. In page 5, line 11, inline equation $R=X[t]M_R$, $K=X[t]M_K$, and $V=X[t]M_V$. Do you mean $R[t]=X[t]M_R$, $K[t]=X[t]M_K$, and $V[t]=X[t]M_V$?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a spikeGPT model

### Strengths
Spiking approach to GPT

### Weaknesses
I am not sure what is the contribution of this paper besides the fact that you have spikified a GPT model.

Further, I dont agree with author's claims on lightweight. All their complexity evaluations are analystical. Can the authors give quantitaive speedup numbers say by running on an actual GPU/TPU hardware whether their spikeGPT gives faster results than many other lightweight GPTs that exist today, such as miniGPT?

### Questions
See weakness above

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
