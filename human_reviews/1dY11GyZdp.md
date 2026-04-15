# Signed-Binarization: Unlocking Efficiency Through Repetition-Sparsity Trade-Off

- Decision: Reject
- Scores: 5, 3, 6

## Abstract
Efficient inference of Deep Neural Networks (DNNs) on resource-constrained edge devices is essential. Quantization and sparsity are key algorithmic techniques that translate to repetition and sparsity within tensors at the hardware-software interface. This paper introduces the concept of repetition-sparsity trade-off that helps explain computational efficiency during inference. We propose Signed Binarization, a unified co-design framework that synergistically integrates hardware-software systems, quantization functions, and representation learning techniques to address this trade-off. Our results demonstrate that Signed Binarization is more accurate than binary models with the same number of non-zero weights. Detailed analysis indicates that signed binarization generates a smaller distribution of effectual (non-zero) parameters nested within a larger distribution of total parameters, both of the same type, for a DNN block. Finally, our approach achieves a 26\% speedup on real hardware, doubles energy efficiency, and reduces density by 2.8x compared to binary methods for ResNet 18, presenting an alternative solution for deploying efficient models in resource-limited environments.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new quantization method called Signed Binarization for training efficient deep neural networks. The key idea is performing local binarization of network weights, which results in global ternarization. This exploits the repetition-sparsity tradeoff to improve efficiency during inference. The method uses two distinct quantization functions and alternates between them locally within the network filters.

### Strengths
1. The proposed methods appear to be easily reproducible.
2. The concept of a repetition-sparsity tradeoff in quantized networks is insightful and well-motivated.

### Weaknesses
1. The contribution of this paper is limited. This paper seems like an incremental work over prior binary network research.
2. The paper should be carefully proofread. There are some typos, e.g., “This necessitate creating …” and “aiming to address …”. I also feel a little bit confused about Table 6, because it doesn’t have a table, and seems like just an overall caption of Table 1 to Table 5. The models used in some Tables are missing, e.g., Table 2 to Table 5.
3. The performance is limited as signed-binary is 1.26x and 1.75x faster than binary and ternary respectively.

### Questions
1. Figure 4 seems unclear to me. The authors mentioned they compared against 13 prior methods, but there are less than 13 blue dots on both subfigures. Also, what are the models used in Figure 4, Figure 5, and Figure 6?
2. Signed Binarization alternates quantization functions locally, but how are the regions/filters assigned to each function? Is this predetermined or optimized during training? Could this assignment influence the results?
3. What is the difference between Signed Binary method without sparsity support and Binary method?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new binarization scheme which uses {-1, 0} binarization for some tiles and {0, +1} binarization for the others. It aims to improve the so-called "repetition-sparsity trade-off" an the accuracy-model size Pareto frontier of binarized neural networks. The paper also visualize model effective parameters and speedup analysis on Intel CPU.

### Strengths
This paper has a good structure. The figures are nice.

### Weaknesses
1. The paper conveys confusing messages on several basic concepts.
    * About weight “repetition” in BNNs (Section 2 last paragraph). The efficiency of BNNs does not mainly come from the “repetition” of 1-bit filters so that redundant memory access can be reduced. It comes from the reduction in bitwidth compared to high-precision floating-point numbers so that more efficient matmuls can be applied. The motivation of the paper is therefore ill-posed.
    * Questionable claim in Section 3: “Binary models aim for efficiency by maximizing weight repetition”. There are no specific constraints or regularizations on the objective function to make a binarized CNN optimized for repeating its filters. The paper needs to provide evidence to show that BNNs are intentionally learning repetitive weights.

2. The goal of improving the repetition-sparsity trade-off is questionable. Repetition and sparsity do not seem to be a trade-off, i.e., improving one doesn’t compromise the other. For example, in the context of the paper, a model with zero weights is extremely sparse and in the meantime has extremely repetitive filters.

3. The method described in Section 4 seems to be incomplete. It uses {-1, 0} binarization for some weight tiles and {0, +1} binarization for the others. This means that on hardware, 0 and 1 will correspond to different values for different tiles. What will be the overhead when accumulating the output of those tiles? The paper needs to provide an analysis.

4. There are limited baseline methods. The paper claims it pushes the Pareto frontier, but in Figure 4 it does not compare to the recent advancements in BNNs, e.g., [1], [2], etc.
    * [1] N. Guo et al., Join the High Accuracy Club on ImageNet with A Binary Neural Network Ticket, arxiv’22.
    * [2] Y. Zhang et al., PokeBNN: A Binary Pursuit of Lightweight Accuracy, CVPR’22.

### Questions
Questions are included in the weakness section above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper the authors propose a method to improve the efficiency of binary quantization models by constraining the presence of -1 and 1 values to specific regions within the filter parameters. By enforcing this additional constraint the method achieves locally binary quantization within a region, by allowing values to be either 0 or {-1, 1}, but globally ternary across the entire filter. The result is that within a constrained region values are binarized as well and highly sparsified which reduces the amount of data that must be moved per layer to perform an inference pass. The authors show that this simple application does not greatly impact the accuracy of the resulting network but yields notable performance improvement.

### Strengths
- The idea is relatively simple and the description regarding the pros and cons from a performance perspective is made clear in the text. 
- Great use of figures (2 and 3) to help the user get the gist of the proposed method. I especially appreciated the color coding and detail provided by Figure 3 to compare and contrast the current work against other approaches.
- While I like the comparisons illustrated in Figure 4 it could be better if there was a way to easily associate the source for each blue dot. For now the reader has to cross-reference the dot with the referenced works to make the comparisons concrete.
- The performance results in Figure 6 clearly demonstrate the increased efficiency of the inference pass for the signed-binary networks compared to normal binary and ternary networks.

### Weaknesses
Major:
- My only major concern is with respect to the novelty of the approach. I like the proposed method but it seems to overlap in many ways with previous work, used for instance in post-training quantization methods, to decompose a set of parameters into non-overlapping regions and using a fixed number of values per region. I think of this work as being an extension of that idea where the set of values per region is constrained to come from a very restricted set.

Minor:
- Figure 1 should be expanded upon to include more valuable information or be removed altogether. As it is I don't think it provides a lot of value to the text. 
- The proposed method requires retraining from scratch to learn the binarized filters (this was mentioned as a limitation in the text).
- The text does not flow very well in several places in the text and needs a few more passes over the text to clean up awkward sentences.
- Most of the text in the first paragraph of section 5.1 seems to be repeating the same idea regarding effectual and ineffectual values. It's not clear to me that a distinction between effectual and ineffectual parameters vs multiplications necessitates redundancy in the text. 
- Though the terminology of "effectual" and "ineffectual" seem to be inherited from prior work I can't say I'm a fan of it. It's not something worth debating but personally, it's not clear to me how much value I gain denoting zeros as "ineffectual".
- Aesthetically speaking I do not care for the wrapped text around Figure 2 or the packed nature of Tables 1-5. Though the tables may fit well together there may be a better way to incorporate Figure 2 by sacrificing Figure 1.

### Questions
The method is exclusively applied to convolutional layers, could it be applied more generally to linear layers, possibly decomposed into sets by row or column or some other fixed pattern?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
