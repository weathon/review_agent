# SpikingLLM: A Conversion-Based Method with Window Inhibition Mechanism for Spiking Large Language Models

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Recent advancements in large language models (LLMs) have led to unprecedented capabilities in real-world applications. However, it remains challenging to reduce the energy consumption of LLMs. In this paper, we aim to improve the energy efficiency of LLMs by leveraging the advantages of brain-inspired spiking neural networks (SNNs). We propose a novel approach called SpikingLLM, which equivalently converts quantized large language models (QLLMs) applying PrefixQuant* to their fully-spiking counterparts(all operators are in a more efficient spiking version). To ensure that every operator can be converted into its spiking version, we propose two approaches: ① QK2Head-migration post-softmax quantization, which significantly improves the performance of current QLLMs with post-softmax quantization; ② Differential-based methods, which tackle the SNN-unfriendly operators such as KV Cache. To further reduce the energy consumption, we introduce a window inhibition mechanism which effectively addresses the over-firing issue in ST-BIF+ neuron and improves the sparsity. With the approaches above, SpikingLLM significantly reduces the energy consumption while achieving state-of-the-art performance on both perplexity and common-sense reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SpikingLLM, a conversion-based framework that transforms quantized large language models into fully spiking neural networks. The goal is to drastically reduce the energy cost of large models while keeping their performance competitive. The authors propose several key ideas to make this possible: a QK2Head-migration quantization technique to handle post-softmax quantization, a refined ST-BIF+ neuron equipped with a window inhibition mechanism to prevent over-firing, and SNN-friendly versions of traditionally non-spiking components such as the KV Cache. Experimental results on LLaMA and Mistral models show that SpikingLLM can cut energy use by over 60% while maintaining strong accuracy on language modeling and reasoning benchmarks.

### Strengths
1. The author provides solid empirical results across multiple large models.

2. The proposed technical solutions, like QK2Head-migration and window inhibition that seem effective.

### Weaknesses
1. The window inhibition mechanism suppresses firing within a short temporal window to prevent over-firing, but this changes the temporal distribution of spikes. Since the quantizer and neuron are tightly coupled, any mismatch between inhibition strength and quantization scale could distort activation magnitude.

2. Replacing continuous operators like softmax, SiLU, and KV Cache with “spiking-friendly” versions is a major modification, but the paper doesn’t quantify how much numerical or representational error each conversion introduces. The authors use “differential strategies” from SpikeZIP-TF, but don’t analyze how approximation errors accumulate layer-by-layer in deep transformer stacks.

3. The technical pipeline is quite dense, making it difficult for non-specialists to grasp. Doesn’t discuss latency or real-time trade-offs, which are critical for deployment.

### Questions
Could the spiking conversion harm adaptability or generalization on tasks beyond language modeling and reasoning?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a brain-inspired, energy-efficient framework for large language models that converts quantized LLMs into fully spiking neural networks. By introducing PrefixQuant* and mechanisms such as QK2Head-migration and window inhibition, the approach ensures all operators can be represented in spiking form while maintaining strong model performance. SpikingLLM effectively tackles SNN-unfriendly operations like KV Cache and Softmax, enabling complete equivalence between quantized and spiking models.

### Strengths
S1: This paper is in good written and easy to follow. The proposed method is simple, especially the Figure 2, which outlines the ANN-SNN conversion pipeline.
S2: The proposed method is mainly based on prefixQuant, which keeps the performance in low precision. This work may achieve somewhat energy efficiency according their metric.

### Weaknesses
W1: My main concern is the novelty of this work: both the ANN-SNN conversion and prefixQuant have already been studied. From my view, this work mainly replaces the quantization method in ANN-SNN conversion pipeline (like the mentioned SpikeZIP-TF). The discussions about the relationship between quantization and spiking neuron networks should go deeper.
W2: In the experiment, although the energy is very low compared with ANNs or the quantized LLM, it takes 16-32 times inference to simulate layer by layer. Therefore, from my view, this design is not suitable for LLMs.

### Questions
For the “inhibiting window”, how to ensure computational equivalence compared with quantization?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This study builds upon the PrefixQuant framework and introduces a novel unstructured, spike-driven inference method. In contrast to conventional quantization techniques, it employs an ANN-to-SNN conversion approach to enable spiking-based inference during the model’s inference phase. Moreover, it utilizes three quantization levels (-1, 0, 1) for spike encoding, aligning with methods like SpikeZIP-TF and SpikeLM. Experimental results demonstrate that the proposed approach surpasses both PrefixQuant and SpikeLLM in terms of efficiency.

### Strengths
1) The framework primarily relies on an ANN-SNN conversion pipeline, which is applied after training. As a result, the training cost is relatively lower compared to training models from scratch.

2) The energy efficiency and accuracy achieved are comparable to, or even surpass, those of quantization-based methods that incorporate spiking neuronal dynamics.

3) This research addresses a crucial area within the SNN domain. Advancing more efficient spiking large language models or improving ANN-SNN conversion techniques is essential for progress in this field.

### Weaknesses
1) From my view, what this work advances is not addressed, because there are may other better quantization methods, why to select PrefixQuant is not addressed.

2) If this work focuses on energy efficient LLMs, why not directly apply additive neural networks, such as BitNet and BitNet-1.58bit? when using 4-bit activations, the performance could be further compared.

### Questions
The comparison with other ANN-SNN conversion method could be addressed, even if comparing in small scale.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SpikingLLM, a fully spiking version of quantized large language models (QLLMs) designed for energy-efficient inference. The method achieves equivalence between QLLMs and spiking neural networks (SNNs) through a QK2Head-migration post-softmax quantization and differential-based handling of SNN-unfriendly operators like KV Cache. It further refines the ST-BIF+ neuron with a window inhibition mechanism, effectively mitigating overfiring and enhancing sparsity. Experimental results show that SpikingLLM significantly reduces energy consumption while achieving state-of-the-art perplexity and reasoning performance, outperforming prior A2S (ANN-to-SNN) and DT (direct training) approaches.

### Strengths
1. This paper combines spiking neural networks (SNNs) and large language models (LLMs) to propose SpikingLLM, which exhibits certain biological neuron interpretability and low-power characteristics. This work focuses on the interesting topic of spiking large language models. 
2. The paper obtains SNNs through a conversion-based quantization method, enabling the training of spiking-driven models with only a short amount of computation time.
3. The paper employs ternary spike representations, extending the precision-fitting capability of spike-based computation and enhancing the model’s expressiveness. However, it lacks rigorous comparative experiments, and the authors are encouraged to discuss this aspect in more depth.

### Weaknesses
1. The main weakness of this paper is that it does not address a clearly defined problem in either the SNN or quantization domains. The authors use PrefixQuant to improve SNN conversion performance through its strong quantization effects. However, the contribution to the field is limited: first, the ANN-to-SNN approach is a conventional method for obtaining SNNs; second, using ternary neuron operators to fit quantized or full-precision LLMs has already been explored in SpikeZIP-TF; and finally, if the performance gain mainly comes from PrefixQuant rather than methodological innovation, the novelty is insufficient.
2. Building on the first point, the paper lacks a rigorous validation of the proposed method’s effectiveness and generalization. For example, in Tables 2 and 3, the authors mainly compare with PrefixQuant, without showing whether the method generalizes to other quantization approaches. Therefore, it is suggested that the authors strengthen the discussion and contribution of the ANN-to-SNN conversion method itself, rather than relying heavily on a specific quantization technique.

### Questions
Please see my comments provided above.

### Soundness
2

### Presentation
3

### Contribution
2
