## Human Reviewer 1

### Summary
This paper presents Domain Shift Tuning (DST), a framework for enhancing domain adaptation in PLMs. DST tackles the challenge of domain shift, where PLMs trained on a large, generalized corpus underperform on a specific target domain. DST introduces two key components:  Knowledge Steering Layer (KSL) and Knowledge Distribution Modeling (KDM). Through these, DST fine-tunes PLMs to align domain-specific weights with the target domain, thus overcoming the domain gap and reducing computational costs associated with large-scale fine-tuning.

### Strengths
- By framing domain adaptation as knowledge distribution alignment, DST minimizes computational overhead and sidesteps catastrophic forgetting. This is particularly beneficial for limited-resource settings, allowing PLMs to adapt to new domains effectively with minimal data.
- The experimental results demonstrate that the proposed method outperforms several baslines.

### Weaknesses
- The motivation in introduction is presented in a somewhat cursory manner, lacking clear logical connections between sentences. In line 32, the claim that “size discrepancy can lead to catastrophic forgetting and poor generalization” is not convincingly supported by the cited references. Additionally, the transition to “Given the swift diversification of PLM applications…” feels abrupt, missing a logical connection that ties it smoothly to the preceding discussion.
- The foundational hypothesis that "PLMs encapsulate multiple pieces of knowledge as subnetworks" (Lines 38-40) lacks supporting references or verification experiments. Furthermore, the approach of representing domain gaps by differences in model parameters between source and target domains is not sufficiently justified. Although empirical results support DST’s effectiveness, the Introduction lacks a clear causal rationale for these core design choices.
- In Table 4, the absence of performance metrics for base methods such as PEFT on LLMs limits the comprehensiveness of the evaluation.
- Writing Issues:
  - Figures and tables, such as Figure 1’s left side, appear cluttered, detracting from clarity.
  - The citation style disrupts readability; author names would be clearer within parentheses.
  - Minor issues, such as the incorrect symbol following "else" in equation (6).

### Questions
refer to the comments

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper introduces a domain adaptation technique called domain shift tuning which consists of a lightweight knowledge steering layer (KSL) and a training method called knowledge distribution modeling (KDM). The KSL is a layer affixed after the last transformer layer in a pre-trained LM, and KDM is applied as an auxiliary loss to attempt to align topic/knowledge latent representations with textual similarity. The KSL predicts a topic and selects a weight accordingly to project the final hidden before projecting again into the vocabulary. The model is kept frozen while the KSL is fine-tuned using a modified CE loss accounting for knowledge vectors and the KDM. The method is tested on encoder models for topic clustering and decoder-LMs for text generation.

### Strengths
1. The authors test their method against an impressive number of baselines, from both domain adaptation and other PEFTs
2. The use of rKSL is important and helpful to understand how much more knowledge than the residual is being used, and it is interesting to see values much bigger than 0. 
3. The method is seemingly model-agnostic which strengthens its applicability to things beyond just language and just Transformers.
4. The subnetwork motivation and integration with the knowledge steering layer is an interesting and intuitive motivation.
5. The authors test on both clustering and text generation. It is great to see a method that applies to both of these tasks, especially as there is a lot of need for good embedding models in addition to LMs.

### Weaknesses
1. Although the KSL is smaller compared to the size of the model, it must have some sort of slow-down associated with it since it appears as an additional layer with an additional step across K subcomponents. What is the speed reduction in using this method?
2. This paper makes multiple references to VAEs as inspiration for the latent vector $z$, but this connection is never formally introduced, nor are any details about what is being referred to in VAEs. Some formal background and direct linking would strengthen the work.
3. The notation and writing is not always the most clear, where some key variables are not clearly defined, and some motivation is not clearly written. For example, latent “knowledge” vector $z$ is not clearly defined nor is its length $K$, and the notion of knowledge is redefined several times in the text, including as a “latent relative concept” or “co-occurence pattern of tokens with similar semantics”. 
4. The published parameter settings for each baseline may not be the fair comparison here, what may be more fair is scaling the baselines according to the parameter budget or throughput associated with the DST method.
5. The LLM experiments are not compared to few-shot/zero-shot prompting despite these models being able to perform in-context learning. The LLM experiments (Table 4) need some sort of baseline to compare to, like in Table 3. 
6. $L_{KDL}$ is not ablated to show its usefulness in this work. 
7. Some code or pseudocode would strengthen knowing how the KSL/KDM is actually implemented. For example, it is unclear how the selection process works for the Waz matrices, and the minimum operation in KDM is also unclear as to how this is differentiated.

### Questions
1. Is $z$ length $K$ for each index in $|x|$? It is defined as length $K$, but then also indexed over the t indices along with the sequence length. Is it different at each sequence index? And if yes, how can it be a scalar as in equation 4 without some sort of argmax/softmax operation, and why should it be different for the same utterance? And if it is argmaxed, how can it be useful in KL divergence unless it remains continuous?
2. What is meant by "KSL considers knowledge as a quantized sample of the underlying token distribution"? Like in a vector quantized/code book sense?
3. Why is $SIM_z$ KL-divergence and $SIM_{TID}$ cosine? Are the $z$ vectors softmaxed and probability distributions? How do these different functions affect the minimization term in KDM?
4. What is the number of fine-tuning steps? It is missing, which is important for defining linear decay, and understanding the cost of the method. 
5. Why minimize the minimum $SIM_z$- $SIM_{TID}$ rather than the maximum for minimax?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper presents Domain Shift Tuning (DST), an innovative framework designed to enhance the adaptability of pre-trained language models (PLMs), across different domains. DST addresses the challenge of domain discrepancies by conceptualizing these gaps as variations in knowledge encapsulated within multiple subnetworks of PLMs. To bridge these gaps, the framework introduces two key components: Knowledge Steering Layer  and Knowledge Distribution Modeling.

### Strengths
1. The idea of this work is interesting. DST introduces a new perspective by treating domain gaps as differences in knowledge subnetworks.

2. KSL provides a lightweight mechanism for representing domain-specific knowledge without changes to the underlying PLM architecture.

3. DST achieves domain adaptation improvements with lower computational overhead

### Weaknesses
1. Citation Formatting: When adhering to the ICLR template guidelines, replace all instances of  `\cite` with `\citep` to ensure proper citation formatting.

2. Motivation: The paper posits that the discrepancy in `dataset sizes` can lead to catastrophic forgetting and poor generalization, but authors have not provided sufficient empirical evidence in the era of LLMs. 

2. Outdated References and Baselines:  Most of the previous work discussed and baselines compared are already 2 years ago.

3. Marginal Improvements on modern models Llama and BLOOM:  In Table 4, the application of DST on the Llama and BLOOM models results in only negligible improvements, calling into question the effectiveness of the proposed method for these specific models.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
3