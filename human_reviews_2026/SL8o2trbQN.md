# Multiple Choice Learning of Low Rank Adapters for Language Modeling

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 2, 8, 6

## Abstract
We propose LoRA-MCL, a training scheme that extends next-token prediction in language models with a method designed to decode diverse, plausible sentence continuations at inference time. Traditional language modeling is an intrinsically ill-posed problem: given a context, multiple ``futures'' may be equally plausible. Our approach leverages Multiple Choice Learning (MCL) and the Winner-Takes-All loss to efficiently handle ambiguity through Low-Rank Adaptation. We provide a theoretical interpretation of applying MCL to language modeling, assuming the data is generated from a mixture of distributions. To illustrate the proposed approach, we use data sampled from mixtures of Markov chains. We then demonstrate with experiments on visual and audio captioning, as well as machine translation, that our method achieves high diversity and relevance in generated outputs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces LoRA-MCL, a training scheme for language models that combines Multiple Choice Learning (MCL) with Low-Rank Adapters (LoRA) with the goal of generating diverse and plausible sentence continuations. ​ Traditional language modeling often struggles with capturing the inherent ambiguity of multi-modal data distributions. ​ LoRA-MCL addresses this problem by using multiple hypotheses and a Winner-Takes-All (WTA) loss to specialize each hypothesis in different modes of the data distribution. The method is computationally efficient by leveraging parallelization techniques. ​ Experiments on audio captioning, image captioning, and machine translation demonstrate that LoRA-MCL achieves a good balance between diversity and quality, outperforming traditional methods in challenging scenarios. ​

### Strengths
1. The combination of Multiple Choice Learning (MCL) with Low-Rank Adaptation (LoRA) seems to be novel.

2. The authors provide a solid theoretical analysis proving propositions and demonstrating how LoRA-MCL captures the modes of a mixture distribution and linking it to the Winner-Takes-All (WTA) loss and hard-EM optimization. ​

3. LoRA-MCL is evaluated across diverse tasks, including audio captioning, image captioning, and machine translation.

4. The method achieves a good balance between diversity and quality in generated outputs compared to baselines LoRA-MLE and LoRA-MoE, as demonstrated through experiments. ​

5. The use of LoRA adapters and parallelization techniques ensures that the method is computationally efficient, making it scalable for large language models. ​

### Weaknesses
1. While this is a nice theoretical paper with apparently solid proofs, I feel the efficacy may be limited because the experiments mainly focus on LoRA-X without considering the large body of work in audio captioning, image captioning, and machine translation.

2. LoRA-MCL is sensitive to hyperparameters like the relaxation parameter (ε) and temperature scheduler (τ). The paper acknowledges that. ​

3. The proposed method involves multiple hypotheses and parallelization techniques, which may increase implementation complexity compared to simpler approaches like LoRA-MLE. This issue may be alleviated with the release of working codes.

4. While the paper evaluates sampling-based decoding methods, the results show mixed performance, and further exploration of annealing schedules and their impact on quality-diversity trade-offs is needed. ​

5. Despite efforts to mitigate collapse, the paper admits that some hypotheses may still be under-trained, which could limit the diversity of outputs. ​

6. The experiments focus on comparing LoRA-MCL, LoRA-MLE in audio captioning, image captioning, and machine translation. However  the comparison does not include or address other SOTA in these tasks.

### Questions
1. How robust is LoRA-MCL to variations in the relaxation parameter (ε) and temperature scheduler (τ)? Are there guidelines for selecting these parameters for new tasks?
2. Have you considered integrating advanced LoRA variants (e.g., LoRA+ or MixLoRA) into LoRA-MCL? If so, what challenges or benefits do you anticipate?
3. Sampling-Based Decoding: The results for sampling-based decoding methods are mixed. Could you elaborate on how LoRA-MCL could be optimized for these methods, particularly in terms of annealing schedules? ​
4. How well do you expect LoRA-MCL to perform on other tasks like dialogue generation, summarization, or code generation?
5. Despite the use of relaxed WTA and annealed MCL, is there a risk of model collapse in scenarios with highly imbalanced modes? ​ How could this be further mitigated? ​

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces LoRA-MCL: applies Multiple Choice Learning with a Winner-Takes-All loss to K parallel LoRA adapters, enabling single-pass, diverse hypotheses while avoiding collapse. Across audio/image captioning and machine translation, achieves stronger quality-diversity trade-offs.

### Strengths
1. The motivation is clear and principled. By applying MCL with a Winner-Takes-All loss, the paper offers a practical recipe for training LLMs with multiple specialized hypotheses, leading to better learning dynamics and stronger outputs.
2. The LoRA-based design is both practical and efficient.
3. Results cover several multimodal tasks, spanning visual and audio captioning as well as machine translation.

### Weaknesses
1. The novelty is limited. The proposed method adds little beyond reformatting existing LoRA techniques into a multiple-choice scheme.
2. The evaluation centers on LoRA-MLE, making it difficult to assess superiority over stronger alternatives. Please include broader baselines (SOTA LLMs and multi-modal models) as well as MoE and multi-head variants. 
3. The current diversity evaluation (limited to Div-n and mBLEU-n) is insufficient, as both are length-sensitive and mostly lexical. Please add semantic and distributional diversity metrics and report the quality-diversity trade-off. For quality, also include LLM-as-a-judge and a human-preference study.
4. Given that the proposed approach targets LLMs, the evaluation would be much stronger with a broader set of text-only benchmarks to demonstrate generality beyond the current settings.
5. The rationale for preferring LoRA over multi-headed outputs is insufficiently evidenced. The comparison centers on parameter count (=640M), while other claimed disadvantages are asserted without empirical backing. Please provide evidence beyond params.

### Questions
Please see the concerns detailed in the Weaknesses.

### Soundness
3

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
3

### Summary
The authors propose a training scheme called LoRA-MCL for LLM-decoder based generation, where to introduce diversity in the generation, they run K LoRAs on the same base model to get diverse outputs. Their design is inspired by the insight that depending on context, the next word prediction can vary a lot such that different predictions are valid in different contexts. They propose that having multiple adapters can lead to each of them being able to predict the next word distribution with a different context, and hence ensure diversity. To train the system with K LoRAs, they propose a winner takes all (WTA) loss. They also demonstrate via experiments with synthetic sequences sampled from a mixture of markov chain distributions that MCL is able to model the underlying distribution more accurately than just using MLE loss using single LoRA. Finally, they compare the quality and diversity of generated outputs from LoRA based beamsearch and MCL based beamsearch, on real world tasks such as audio and image caption generation and machine translation. Results look strongest for audio captioning(Figure2) where MCL provides better diversity and quality. On image caption generation(Table2) and MT(Figure4), the results are a bit mixed where on image caption generation MCL gets higher quality but lesser diversity, and on MT it gets higher diversity but lesser quality.

### Strengths
- Excellent motivation and theoretical grounding of the problem by looking at next token prediction via the lens of modeling mixture of distributions.
- Proposes an efficient implementation by using custom architecture implementation of running K LoRAs in a single forward pass via batching and grouped convolution. 
- Care taken to have fair comparison with baselines while keeping parity on inference budget by adjusting LoRA rank and beam widths accordingly (314-323)
- Quality vs. diversity trade-off is nice for audio captioning where we have better quality and diversity via MCL (Figure 2) The results are a bit mixed in other settings such as Table 2 where diversity decreases with respect to the baselines although the quality increases.

### Weaknesses
Despite the grouped convolution, you need to create a batch of the input sequence repeated K times for forward pass which increases FLOPs. While this would be faster than running K separate inferences in a for loop, it would still incur a cost with respect to throughput when deployed at scale where the GPU utilization is saturated.

### Questions
- In equation 6, $\mathcal{L}^{\text{WTA}}(\theta)$ has 2 formulas, is that for relaxed WTA and annealed WTA respectively?
- In Table 1, the maximum value of r in MLE is 40 but for MCL maximum value of K is 7 which should be equivalent to 7x8=56 ranked LoRA in MLE right? Should the value of K only go up to 5 for fair comparison here?
- In Table 2, it would help to add the value of K for each MCL row. Are we supposed to divide the number of beams present in MLE by MCL to infer the value of K? It is confusing to parse the results here.

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
3

### Summary
This paper proposes a method that integrates Multiple Choice Learning with LoRA to improve the ability of language models to generate diverse yet plausible text continuations. The key idea is to address the inherent multimodality of next-token prediction by training multiple LoRA adapters within the same model using a relaxed/annealed WTA loss. This design allows different adapters to specialize in different modes of the conditional output distribution while remaining parameter-efficient. The authors provide theoretical justification under a mixture-of-distributions assumption, validate the framework on mixtures of Markov chains. The authors conduct experiments on audio captioning, image captioning, and machine translation, showing improved quality–diversity trade-offs compared with LoRA-based baselines.

### Strengths
1. The paper adopts the Multiple Choice Learning framework with multiple LoRA modules to capture diverse yet plausible text continuations, and introduces a grouped parallel implementation that enables efficient training of all hypotheses with minimal memory overhead.

2. The paper provides theoretical analysis to demonstrate the effectiveness of the proposed loss function, and then verifies it using synthetic Mixture of Markov Chains data, showing that the approach can successfully recover distinct latent modes as expected.

3. The method is tested on various tasks, including image captioning, audio captioning, and machine translation, and shows promising results on certain tasks.

### Weaknesses
1. The method is relatively straightforward, employing Multiple Choice Learning with multiple LoRA modules and relying on existing mechanisms such as the WTA loss and its Relaxed/Annealed variants. 
2. The proposed method shows limited improvement over other LoRA-based approaches in certain cases. For instance, as shown in Table 2, it performs notably worse than LoRA-MLE DBS (λ = 1.0) on the mBLEU-4 metric, which measures diversity, and does not demonstrate clear advantages on other evaluation metrics either.
3. Although the paper evaluates on multiple tasks such as image and audio captioning, it mainly compares with other LoRA-based baselines. Since the method aims to improve the trade-off between diversity and quality, comparisons with task-specific state-of-the-art methods would strengthen the empirical claims.

### Questions
1. During inference, does the model also adopt a WTA-like mechanism to select the response from the LoRA module with the highest probability, or do all K LoRA modules independently generate K separate outputs? Since each LoRA module is expected to specialize in a particular output type, could the authors provide evidence showing what each module has learned in the real tasks such as image or audio captioning?
2. Could the proposed method achieve better performance or improved efficiency compared with existing state-of-the-art methods that aim to enhance diversity in the tasks evaluated in this paper, such as image and audio captioning?
3. Regarding Weakness 2, I noticed that the paper mentions LoRA-MCL combined with DBS might alleviate this issue. I am curious whether there are concrete experiments showing the performance of LoRA-MCL when used with different decoding strategies, such as Beam Search or Diverse Beam Search.
4. In LoRA training, increasing the rank does not always improve performance and can sometimes cause overfitting or unstable optimization. Could the LoRA-MLE baseline be underperforming because the large-rank setup (rank =$K \times r$) leads to optimization instability, while other smaller ranks might perform better?

### Soundness
3

### Presentation
2

### Contribution
2
