# BrainGPT: A Brain-Inspired SNN-Based Large Language Model

- Avg Score: 3.75
- Decision: Reject
- Scores: 3, 3, 3, 6

## Abstract
Large language models (LLMs) based on artificial neural networks (ANNs) have demonstrated remarkable performance but face challenges in computational efficiency and biological interpretability. We propose BrainGPT, a novel LLM architecture based on the Test-Time Training (TTT) framework and inspired by spiking neural networks (SNNs) and neurobiological principles. Our approach incorporates a dual-model structure, emulating the hierarchical language processing observed in the human brain, and utilizes a specialized integrate-and-fire neuron model with adaptive thresholding. Through a multi-stage training strategy, including quantization-aware pre-training, ANN-to-SNN conversion, and biologically inspired unsupervised learning, we achieve a mathematically proven lossless conversion from ANN to SNN, preserving 100\% of the original ANN model's performance. Moreover, the biologically inspired unsupervised learning optimizes the maximum time steps required to maintain 100\% ANN performance. Compared to the original TTT model, BrainGPT achieves a 33.4\% increase in energy efficiency and demonstrates a 66.7\% improvement in training convergence speed. This work advances the development of energy-efficient and biologically interpretable large language models that match the performance of state-of-the-art ANN-based models while significantly improving upon the TTT framework.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces BrainGPT, a spiking neural network-based language model architecture inspired by neurobiological principles. The authors propose a dual-model structure, a specialized neuron model, and a multi-stage training strategy. They claim improvements in energy efficiency and training convergence speed compared to traditional architectures, supported by mathematical proofs and empirical results.

### Strengths
- Novel integration of SNNs with large language models, potentially bridging the gap between artificial and biological neural networks.
- Mathematically rigorous approach to ANN-to-SNN conversion, with proofs for lossless transformation.

### Weaknesses
- The paper's title and claims about "Large Language Models" are misleading given the actual scale of the model presented. The authors report a model with only 150M parameters, which is far from the current understanding of LLMs in the field. While the authors mention computational resource limitations in their limitations section, this doesn't justify the use of "Large Language Model" in the title and throughout the paper. With optimized training techniques like flash attention kernels, it's possible to train much larger models (e.g., 2.7B parameters) on 100B tokens within a week using 8 H100 GPUs. If the authors acknowledge this limitation, why insist on using "Large Language Model" in the title? A more accurate description would be "language modeling" or "neural language model," which would better reflect the actual scale and scope of the work presented.
- The comparison of perplexity is not very fair. Can the authors explain why Mamba2's vocabulary size is 50,277 while the other comparison models use 32,000, according to Table 4-7? This discrepancy may lead to unfair perplexity comparisons between models. Different vocabulary sizes can impact perplexity scores, making direct comparisons misleading. The authors should either use consistent vocabulary sizes across all models or provide a detailed explanation and analysis of how this difference affects the results.
- The authors mention that the model was trained on 100B+ tokens, including both Chinese and English. In this case, it should be possible to compare the models' common reasoning performance, like Pythia [1] (169M 300B tokens), at least on tasks like HellaSWag. Why did the authors only compare perplexity? This limitation restricts the paper's impact and relevance. Evaluating only perplexity on a limited dataset fails to demonstrate the model's practical capabilities or its ability to generalize across different tasks. 
- The authors fail to cite or compare with previous work on SNNs for text generation. For example, SpikeGPT [2] scaled to 216M parameters and compared downstream tasks, while AstroSNN [3] scaled to 1B+ parameters and compared common reasoning tasks. These omissions significantly weaken the paper's positioning within the current state of the art. This oversight raises questions about the novelty and relative performance of BrainGPT. Without direct comparisons to these existing SNN-based language models, it's impossible to gauge the true contribution of this work to the field.
- This paper lacks a comprehensive ablation study to justify the various components of the proposed architecture. Without such analysis, it's unclear which aspects of BrainGPT contribute most significantly to its performance, making it difficult for other researchers to build upon this work effectively.


[1]:Biderman, Stella, et al. "Pythia: A suite for analyzing large language models across training and scaling." International Conference on Machine Learning. PMLR, 2023.

[2]:Zhu, Rui-Jie, et al. "Spikegpt: Generative pre-trained language model with spiking neural networks." TMLR, 2024.

[3]:Shen, Guobin, et al. "Astrocyte-Enabled Advancements in Spiking Neural Networks for Large Language Modeling." arXiv preprint arXiv:2312.07625 (2023).

### Questions
See my weakness part.

### Soundness
1

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
The paper introduces a training protocol for brain inspired language models. Its claimed contributions consist of a backbone architecture that processes tokens along two parallel streams, the conversion of this model to a spiking neural network (SNN), and an unsupervised post-training method that reduces the required number of operations of the converted SNN. The methods are compared on fairly small scale language tasks at the order of 150M parameter models.

**Recommendation.**
The paper is not in a mature state for publication at a top tier machine learning conference. Central components of the methodology are not formally expressed, and no ablation study was conducted to measure the impact of the different modifications to the baselines. Instead, pages 4, 5 and 6 are largely spend on vague linguistic descriptions that leave the reader without a clear understanding of the proposed model. The evaluation of language models is limited to perplexities on two datasets, while the community established evaluating on a set of downstream tasks with open source libraries. Energy estimates in Table 2 are not supported by assumptions or calculations.

### Strengths
- The paper takes efforts towards aligning the macroscopic architecture and the microscopic implementation with findings from neuroscience. This is expressed in the choice of architecture (TTT based on Sun et al), and the implementation if integrate-and-fire neurons.
- Training convergence speed (Sec 4.4): The papers shows accelerated convergence compared to the TTT baseline of Sun et al.
- Describes a conversion method from 8-bit quantized activations to Integrate-and-Fire neurons, which potentially allows implementations in neuromorphic hardware. It seems though that this method was already presented in [You et al](https://arxiv.org/abs/2406.03470), which is cited by the paper.

### Weaknesses
# Methodology remains widely unclear
The major weakness of this paper is its shallow treatment of the underlying methods, which even after carefully considering the appendix and related works does not become fully clear to the reviewer. This occurs despite the fact that the reviewer is familiar with the work of [Sun et al](https://arxiv.org/abs/2407.04620) on learning to learn at test time, published on brain inspired recurrent architectures, and is familiar with relevant related works. 

**Section 3.1:** The space is not used efficiently to convey the key concepts of the work. There are many vague motivations for choosing certain modules of the architecture, e.g. line 172ff, 184ff, 189ff, 197f, 204ff, 209f, 245f, 249f, 256f, 262f. The reviewer suggests to maintain this density of motivations only in the introduction, and instead use the space for formalizes statements about the proposed methodology, preferably in mathematical language. The only formal description in the methodology section is the "Excitory-Inhibitory Integrate-and-Fire Neuron Model" (eqn 1). Figure 1, without further explanations, attempts to convey the model's architecture to the reader. It would be less ambiguous if the methodology was explained in a formal way. In particular, components that are non-standard or constitute a contribution of the paper should be formalized in the universal language of mathematics. For example, the paper claims "Dual Test-Time Training" as "the foundational framework" of the work, without going into the details of this method. Except for the reference to Sun et al., Test-Time Training (TTT) is not formally introduced. No modifications to TTT are formally expressed. Hence, it is impossible for the reviewer to evaluate criteria such as originality or quality. The informal description given in line 196 ff is not sufficient for this purpose. The paper furthermore, does not shed light on why their modifications to TTT might lead to the empirical differences. It would therefore be valuable to add an ablation study that quantitatively distills the contributions of the paper from prior works. 
Some specific questions:
- what are "remapped word features"?
- How are the two streams merged again for "sequential integration of outputs" line 197f?
- What is a shared gate? What is the purpose of shared gates? 
- Why is the convergence speed of this dual model different from TTT? (see line 533 f)

**Section 3.2:** It is not clear how or if the ANN-SNN conversion method improves over SpikeZIP. Also Appendix A does not answer this question for the reviewer. A formal comparison as well as empirical evaluations would add value to the paper.
# Evaluation not meeting LM standards
The evaluation does not meet the standards of evaluation of language models in the machine learning community.
- Evaluation (Sec 4.2): No downstream task evaluation, which is standard today and accessible by open source libraries such as the [LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness). Makes comparison with established methods hard.
- No comprehensive evaluation against SpikeZIP-TF, which appears to have introduced the ANN-SNN conversion method discussed here
- Reproducibility: It is listed several times in the paper and the appendix that the model was trained for 50.000 steps. But what is the batch size in number of tokens? How many tokens were trained in total?
- Energy efficiency analysis (Sec 4.3): It is clear how the numbers in table 2 are calculated. Neither the paper nor in the appendix has any explanation for these estimates. The paper should clearly state how the energy estimates are calculated, and which assumptions back the numbers.
# Misc
- mathematical equivalence of the ANN and converted SNN is highlighted in line 274, 343 etc. as a central feature of the paper. Only a formal proof of this claim will add value to the paper.
- Appendix B describes "SNN friendly computations". These remain largely unclear. E.g. why do the leaky integrators model sigmoid functions or SiLU functions? How is spiking softmax described and implemented? How does it actually relate to the known softmax? In which sense are these operations including the RMSNorm layers biologically interpretable as advertised by the paper? Please consider formally deriving how the leaky integrator models the activation functions. It would furthermore be valuable to discuss the biological implementation of such operations.

### Questions
# Questions
- Are the numbers reported in table 2 for QAT ANN Model based on fp operations or 8-bit integer operations?
- As far as I understand, STDP training is conducted after the pretraining to reduce the number of training steps. Do the authors have a hypothesis why these additional parameter updates (without task supervision as far as I can tell) do not harm the model's performance? Are the parameter updates obtained largely uncorrelated with the updates obtained from backprop?
- Is the model still mathematically equivalent after STDP training?
# Suggestions
- Describe the architecture in detail and describe how it technically relates to Sun et al 
- Conduct an ablation study to distill which modifications lead to the observed benefits such as faster convergence rates
- It is straight forward to evaluate on a larger set of downstream tasks, e.g. using the [LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness), which allows for easy comparison with established models.
- Back up the claims on energy efficiency
- It is not clear how the conversion method improves SpikeZIP. Please clarify.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors propose BrainGPT, a language model architecture based on the Test-time training (TTT) framework and spiking neural networks. The model is developed using a multi-staged training process, involving quantization-aware pre-training, ANN-to-SNN conversion, and biologically inspired unsupervised learning. The results demonstrate that compared to original TTT model, the proposed model has better energy efficiency and has faster convergence during training.

### Strengths
The proposed idea expands upon an interesting framework, namely Test time training, which is relevant in the literature of long-context sequence learning. It proposes a multi-stage training strategy to convert a TTT-based ANN model to an SNN architecture. The idea to use an STDP-based unsupervised learning technique to reduce the operating time steps seems interesting.

### Weaknesses
1) The paper has very limited evaluation of its proposed methodology. The authors compare their model against Llama, Mamba architectures, however, (a) they do not use any publicly available results, i.e. the Llama, Mamba models shown in the results are all custom made (b) the results are not even close to the current state-of-the-art models. Even older models like GPT-2 [1] small which has less parameter count than proposed model performs much better (PPL = 29.41 for GPT-2 small compared to 42.87 of the proposed model on Wikitext2).
2) There have been various previous work on Spiking Language models, such as SpikeGPT [2], SpikeLM, etc. The authors did not compare their performance against them. For comparison, SpikeGPT gets PPL of 18.01 on wikitext 2.
3) The authors mentioned that they have given rigorous mathematical proof for the lossless conversion of ANN to SNN. However, I was not able to find any concrete proof of the same.
4) The authors mention that they implement "two distinct sub-models: a standard autoregressive language model for broad linguistic representation, and a model focused on processing parts of speech for more abstract aspects of language", however, there is no ablation study on the effect of each sub-model. Also, there are no empirical justifications on how the 2 sub-models are doing their underlying tasks.



References:
[1] Radford, Alec, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. "Language models are unsupervised multitask learners." OpenAI blog 1, no. 8 (2019): 9.

[2] Zhu, Rui-Jie, Qihang Zhao, Guoqi Li, and Jason K. Eshraghian. "Spikegpt: Generative pre-trained language model with spiking neural networks." arXiv preprint arXiv:2302.13939 (2023).

### Questions
1) Was there any reason on why the spiking baselines were not used for comparison?
2) Instead of just generative tasks can the authors also evaluate their results on benchmarks such as GLUE for text classification tasks.
3) How was the energy numbers calculated? 
4) Could the authors explicitly highlight the proof that is mentioned in the paper regarding lossless conversion of the underlying ANN to SNN?

Please see the weaknesses as well.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The article introduces a new large language model (LLM) inspired by native brain functions, named BrainGPT. This novel architecture is based on the Test-Time Training framework and integrates the principles of Spiking Neural Networks (SNNs) to mimic the functionality of the biological brain. BrainGPT aims to emulate the native human brain's language mechanisms.
The proposed methodology involves transforming existing Artificial Neural Networks (ANNs) into Spiking Neural Networks through a series of stages: quantization-aware pre-training, conversion, and unsupervised learning. These stages are demonstrated to preserve 100% of the original ANN's performance. Additionally, the model offers significant energy efficiency optimizations and shows improvements in training convergence.

### Strengths
The proposed model, along with its claims, is well-aligned with the state-of-the-art in the field. The mathematical background provided to support the approach is clear and concise. The evidence presented is straightforward and easy to follow, offering a robust variety of supporting previous studies and works. Both the mathematical analysis and the experimental evidence strongly support the claims made in the paper.

### Weaknesses
While the proposed model and its claims are well-aligned with the state-of-the-art, there are notable limitations in the scaling and evaluation used. The paper lacks analysis or comparison of training times, which could be an important factor to consider. Including such an analysis would provide a more comprehensive evaluation of the model's performance and efficiency.

### Questions
Are you planning to run more experiments on hardware that is optimized for Spiking Neural Networks (SNNs)?

### Soundness
3

### Presentation
3

### Contribution
3
