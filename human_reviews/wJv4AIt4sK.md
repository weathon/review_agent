# Effective Interplay between Sparsity and Quantization: From Theory to Practice

- Avg Score: 7.50
- Decision: Accept (Spotlight)
- Scores: 6, 8, 8, 8

## Abstract
The increasing size of deep neural networks (DNNs) necessitates effective model compression to reduce their computational and memory footprints. Sparsity and quantization are two prominent compression methods that have been shown to reduce DNNs' computational and memory footprints significantly while preserving model accuracy. However, how these two methods interact when combined together remains a key question for developers, as many tacitly assume that they are orthogonal, meaning that their combined use does not introduce additional errors beyond those introduced by each method independently. In this paper, we provide the first mathematical proof that sparsity and quantization are non-orthogonal. We corroborate these results with experiments spanning a range of large language models, including the OPT and LLaMA model families (with 125M to 8B parameters), and vision models like ViT and ResNet. We show that the order in which we apply these methods matters because applying quantization before sparsity may disrupt the relative importance of tensor elements, which may inadvertently remove significant elements from a tensor. More importantly, we show that even if applied in the correct order, the compounded errors from sparsity and quantization can significantly harm accuracy. Our findings extend to the efficient deployment of large models in resource-constrained compute platforms to reduce serving cost, offering insights into best practices for applying these compression methods to maximize hardware resource efficiency without compromising accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present a theoretical proof showing that the order in which magnitude-based pruning and scaled block quantization are performed is of importance in the context of preserving model performance. The authors define notions such as orthogonality of two operations — which is when the composition of the two operations does not result in any additional error than applying each individual transformation. The authors show theoretically that magnitude pruning and quantization are not orthogonal operations (when going beyond tensor-level) and further showed, both theoretically and empirically, that applying pruning first and then quantization generally leads to better performance.

### Strengths
- Notation, definitions, and theorems in Section 3 are generally clear and their significance is adequately articulated.
- The authors have addressed an issue that has gone overlooked in the pruning/quantization literature through both theoretical proofs and derivations as well as empirical studies that further solidify their claims.

### Weaknesses
- The discussion following Theorem 3.9 is very hard to digest for a reader who hasn’t spent as much time as the authors thinking about this problem. I’d encourage the authors to prune the text, retaining only the essential message (which presumably is what’s written in italics) and moving other information to the Appendix.
- Overall, the theoretical claims and experiments are not astonishing as one would perhaps expect that pruning should precede quantization.
- The theoretical contribution is quite limited as it only holds for magnitude-based pruning (without fine tuning) and block-wise quantization. Importantly, magnitude pruning has gone out of fashion in the context of LLMs because it requires costly fine-tuning to recover model performance and is outperformed by methods like SparseGPT and WANDA when fine-tuning is not performed. The authors mention in the Appendix that, empirically, the order had less of an impact for WANDA and SparseGPT. 
- The experiments seem to be quite orthogonal to the theoretical results. By employing fine-tuning for all the experiments, the authors are making their original theoretical proofs/derivations inapplicable in the context of the experiments as the derivations are based on errors calculated when no fine-tuning is applied. 
- Proof of Theorem 3.5: Only show equality is attained for L1 norm and not all norms. Is it clear that this implies that equality is also achieved for all other norms? Statement of Theorem or proof should be modified to address this. 
- Proof of Theorem 3.6 is only a counter-example for the L1 norm. Is it immediate that the theorem is true in general for norms beyond the L1 norm? Either the statement of the theorem or the proof should be modified to address this.
- Throughout the paper, some statements are true for all norms, others are only shown for the L1 norm, and then the empirical experiments utilize the L2 norm for measuring errors. 
- The generalization of orthogonality in Definition 3.8 is not clear to me as functions are now being applied coordinate-wise. Is the composition only permitted to happen in one coordinate (similar to in Theorem 3.9). It might be worth it to explicitly write out the definition as the lack of an explicit definition of orthogonality also makes the statement of Theorem 3.9 confusing.

### Questions
- In Section 3, the authors mention performing quantization at the level of “blocks." Could you clarify what you mean by a “block” in this context? Does it refer to a set of weights associated with a CNN filter, or does it resemble the M:N sparsity blocks? Or is it something entirely different?
- Consider renaming Definition 3.4 to avoid confusion, as it defines "orthogonality" between two functions in a way that diverges from the standard interpretation. Traditionally, orthogonality between functions is defined by the condition \(\int f(x) g(x) \, dx = 0\), so using "orthogonality" here might lead to misinterpretation.
- Why is it important to consider block-wise quantization in Section 3? Since it’s a theoretical derivation, why don’t you simply assume quantization on the tensor level?
- Theorem 3.5 assumes “max-scaled block-wise quantization”. Is such quantization prevalent in the literature and in practice?
- Theorems 3.5 and 3.6 imply that the optimal order is pruning followed by quantization. Theorem 3.7 analyses the error for the suboptimal order. Why is that of interest?
- Is Equation 12 a lower bound or an upper bound? You might want to rename it accordingly to “Orthogonality Lower Bound” or “Orthogonality Upper Bound” to help the reader.
- “If the compression methods are non-orthogonal, and the evaluation metric indicates better model performance with lower values, we expect the compressed model’s evaluation metric to exceed the orthogonality bound.” — I read this sentence several times and I still can’t understand it. What do you mean by “lower values”? Which values?
- “For OPT, LLaMA, ViT, and ResNet fine-tuning, we employ sparse fine-tuning on a dense” — what method exactly are you using? Please cite the paper.
- “we apply one-shot quantization to sparse fine-tuned models” — again, what method exactly are you using? Please cite the paper. is it the “max-scaled block-wise quantization”?
- “we directly fine-tune the model in a quantized and sparsified manner” — how does one fine tune a quantized model? Isn’t there an issue with doing that?
- Could you summarize the “Experimental setup” in the form of a table. Otherwise, there are too many details in the paragraph and it’s very hard to digest.
- In Figure 1, the error accumulates across layers. This stands in contrast to Figure 1 in [1] which shows attenuation of noise injected in intermediate layers. Could it be that the authors should compute a relative error instead of an absolute error (see caption in Figure 1 of that paper)?
- “and/or reduce quantization effective bitwidth.” — what do you mean by “effective bit width”?
- “TOPS/mm2” — what’s TOPS and mm^2?

[1] Stronger Generalization Bounds for Deep Nets via a Compression Approach

### Soundness
2

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
This paper studies the interaction between weight sparsity, weight quantization, and activation quantization in small-to-moderate sized LLMs, ViTs, and CNNs. The authors prove and demonstrate empirically that these methods cannot be considered as purely orthogonal compression modalities under the orthogonality definitions proposed in the paper. Specifically, the authors show that the composition of these strategies is order-dependent and the combined error incurred generally exceeds the sum of the errors produced by applying each method independently.

### Strengths
* A timely and important topic as sparsity and quantization are promising compression strategies for the large model scales popular today.
* The paper includes a comprehensive summary of relevant literature.
* The proofs are relatively easy to follow and explained in an intuitive manner by the authors in the main text.
* Empirical results generally appear to support the theoretical findings.
* While many works have studied the combination of sparsity and quantization, this is the first that I am aware of to rigorously consider the interplay between these methods in detail. 
* Empirical experiments include both LLMs and vision models. 
* Extensive supplementary info includes an analysis of several leading SOTA methods from LLM pruning and quantization literature.

### Weaknesses
Overall I am leaning towards accept; however, some concerns regarding the empirical experimental design causes me to doubt the applicability of the results to more general settings:

* The primary metrics considered in the empirical results are perplexity or cross-entropy loss. While these are certainly reasonable proxies for downstream task performance, they are not perfectly correlated. While some accuracy metric for CV models was included in the appendices, it would be beneficial to extend this to downstream tasks for LLMs such as the OpenLLM v1 leaderboard evaluation tasks. It has been shown previously that PPL and CE can be particularly misleading metrics for quantized and sparse models [1]. 
* The experimental design for Section 4.1 is potentially concerning. If I understand the described process correctly, in the Q->S case the pretrained models are pruned and quantized before each forward pass (i.e., instantaneous masking and quantizing). Are the parameters themselves stored as dense fp32 tensors during this process and quantization is simulated similar to QAT approaches? Are the optimizer states left in fp32? The authors note issues with training dynamics in the Q->S case in Appendix A and my concern is that this could be related to numerical precision issues during fine-tuning rather than providing a reliable comparison on the order of compression. Adding a more detailed summary of the fine-tuning approaches in the appendix would potentially clear up any misunderstandings on this point. 
* In the Q->S case quantized activations are used but in the S->Q case it appears the full precision activations are used. It's unclear to me if the dramatic difference in performance is caused by the quantized activations during fine-tuning rather than the specific order of compression for the weights. 


[1] A. Jaiswal, Z. Gan, X. Du, B. Zhang, Z. Wang, and Y. Yang, “Compressing LLMs: The Truth is Rarely Pure and Never Simple,” Oct. 02, 2023, arXiv: arXiv:2310.01382. doi: 10.48550/arXiv.2310.01382.


### Suggestions / Typos:
* Defining “tensor and dot-product levels” earlier in the text would improve the reader's understanding. Specifically it may be worthwhile to relate these terms to “weights” and “activations” respectively.  I note that activations / dot-products are also represented as tensors. 
* On L68, the authors refer to the challenge of quantizing LLMs due to outliers in “tensor distributions” and reference the smoothquant paper. This should be corrected to “dot-product outliers” as the challenge typically arises from outliers in the activations, not the weights (which instead follow a more gaussian-like distribution typically). 
* I suggest separating references for fine-grained (N:M and similar) and structured (neuron-level or larger NN components) sparsity in the related work discussion on L115. In particular, it would be beneficial to introduce N:M sparsity before it appears in section 3. 
* L469: state-of-the-arts -> state-of-the-art

### Questions
* In the Q->S case, the authors make the argument that this ordering may lead to additional errors when two otherwise unequal weights in the non-quantized precision are set to the same value once quantized. This is an intuitive conclusion but it would be interesting to ground this discussion in empirical evidence of the proportion of weights that this affects, on average, in a pre-trained model. 
* Are the pretrained LLMs obtained from the base models or instruct-tuned variants? Making this explicit in the paper would be beneficial. 
* L312 states that all linear layers were compressed for LLMs. Can you confirm that this included the lm-head, but not the encoder which is typically implemented as an embedding?
* Table 10 values for 1:4 are counter to typical intuition that higher sparsities generally perform worse. Could the authors confirm that this is 1:4 and not 3:4 sparsity?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides a comprehensive theoretical and empirical investigation into the interplay between sparsity and quantization, two widely used model compression techniques. The authors mathematically prove that sparsity and quantization are non-orthogonal operations, meaning their combined use introduces compounded errors beyond those incurred by each method independently. They further derive the optimal order of applying sparsity before quantization (S→Q) to minimize additional errors. These theoretical findings are validated through extensive experiments on large language models (OPT, LLaMA), vision transformers (ViT), and convolutional neural networks (ResNet). The paper also introduces the novel "orthogonality bound" metric to efficiently estimate the performance of sparse-quantized models without expensive retraining.

### Strengths
- The paper makes significant theoretical contributions by proving the non-orthogonality of sparsity and quantization and deriving the optimal $S\to Q$ order. These insights challenge conventional assumptions and provide valuable guidance for model compression. 
- The mathematical analysis is rigorous and comprehensive, covering tensor-level and dot product-level errors. 
- The experimental results are extensive, spanning diverse models (OPT, LLama, ResNet, ViT) and settings.
- The orthogonality bound metric seems like a useful tool for practitioners. 
- Overall the paper is well-structured, with clear definitions, detailed appendices, and informative tables.

### Weaknesses
- While the experiments cover a range of models and settings, the datasets used (WikiText2, ImageNet1k) are relatively small and few. Evaluating on larger, more challenging datasets would further strengthen the findings. 
- The paper does not explore the impact of different sparsity patterns (e.g., block-wise sparsity) or more advanced quantization schemes.

### Questions
- Evaluating the findings on larger, more diverse dataset would be nice. 
- It would be interesting to see how the optimal $S\to Q$ order and orthogonality bound extend to other sparsity patterns and quantization schemes. Can the authors comment on the generality of their findings in this regard?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper explores the relationship between two widely used compression techniques: sparsity and quantization. Specifically, it demonstrates that these techniques are not independent of one another; the order in which they are applied can significantly impact the results. Additionally, their combination can lead to error propagation, with accumulated errors affecting consecutive layers. The study draws from both theoretical analysis and experimental results conducted on large, modern neural networks.

### Strengths
- The paper covers an interesting and timely topic. Given the increasing size of parameters in pre-trained models, there is growing interest in techniques such as quantization and sparsity. Providing both analytical and empirical insights into the relationship between these techniques is valuable, especially as they are often studied separately.The findings in this paper, such as the optimal order for applying sparsity and quantization and the established upper bounds, can offer practical guidance for researchers in this area.  
- The paper effectively demonstrates the non-orthogonality of sparsity and quantization, determining the optimal sequence for applying these transformations through theoretical analysis, supported by empirical studies on large, modern networks.  
- The work is well-written, easy to follow, and enjoyable to read.

### Weaknesses
- In the experiments section, the results appear promising and generally align with the theoretical findings. However, it is unclear whether the reported results represent averages of multiple runs or single-run outcomes. If they are averages, what are the standard deviations?  
- Additionally, I believe the related work section should remain in the main body of the paper, particularly since there is available space before reaching the 10-page limit. Moving it to the appendix could diminish its visibility and importance.

### Questions
- In Table 2, what do the bold-out results represent? This should be explained in the caption.   
- In Table 2: perhaps it would be beneficial to show the delta to the sparsity 0% instead/additionally  (e.g. in the appendix)?   

Overall, I find the topic of this paper both interesting and potentially valuable to researchers focusing on sparsity and quantization. The claims and theorems are clearly articulated (I briefly reviewed the details of Theorems 3.5, 3.6, 3.7, and 3.9 in the appendix), and the empirical evaluation, while primarily centered on magnitude-based sparsity, is compelling and conducted across various models and tasks. I believe the strengths of the paper outweigh weaknesses (in fact, I do not have significant concerns regarding weaknesses). Therefore, I am inclined to recommend acceptance.

### Soundness
3

### Presentation
4

### Contribution
3
