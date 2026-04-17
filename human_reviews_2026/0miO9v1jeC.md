# TAR: Token Adaptive Routing Framework for LLMs Token-level Semantic Correction Inspired by Neuro-Linguistic Pathways

- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Large language models (LLMs) often suffer from cascading errors in math reasoning due to token-level semantic defects. A key limitation is that the reliance on unidirectional feedforward pathways makes LLMs unable to dynamically correct token-level defects during reasoning. In contrast, neuro-linguistic pathways in the human brain—centered on Broca’s and Wernicke’s areas—operate as a closed loop, integrating semantics through feedforward pathways while leveraging feedback circuit for error correction and signal adaptation. The loop involves conflict detection in the anterior cingulate cortex (ACC), cross-regional error transmission via the arcuate fasciculus/IFOF, and compensatory reprocessing in the DLPFC–Broca circuit. Inspired by the functional architecture of neuro-linguistic pathways, we propose a Token Adaptive Routing (TAR) framework that establishes a brain-inspired self-correcting loop in LLMs without requiring parameter fine-tuning. TAR comprises three components: (1) \textbf{Semantic Defect Monitor}, analogous to the anterior cingulate cortex (ACC) for identifying tokens with semantic defects; (2) \textbf{Adaptive Router}, resembling the arcuate fasciculus/IFOF for routing defective tokens to the most compatible LLM functional block; and (3) Feedback-based Re-representation, inspired by the DLPFC–Broca circuit for correcting semantic defects. Experiments show that TAR improves accuracy and reduces the number of inference tokens. On the challenging AIME25 benchmark, TAR improves the accuracy of Qwen3-1.7B by +3.36% while reducing inference tokens by 13.7%. Furthermore, we reveal that maintaining high token confidence is essential for reasoning performance, and deeper blocks in LLMs play a crucial role in shortening reasoning depth. Our code is available at https://anonymous.4open.science/r/warehouse-25F5

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Inspired by the structure of the human brain, this paper proposes TAR, a framework for token-level semantic correction. Based on the authors’ experiments, the proposed framework appears to yield certain performance improvements.

However, overall, I find the claimed connection between the method and biological brain structures somewhat tenuous. The paper does not provide sufficient justification or clear evidence for the claimed correspondence. This is my first time reviewing an AI paper that attempts to draw inspiration from biological structures, so I will lower my confidence accordingly.

### Strengths
**[S1]** The paper attempts to establish a connection between the proposed AI method and human brain structures, reflecting an interesting biological inspiration in the design of artificial intelligence systems.

### Weaknesses
**[W1]** The relationship between the proposed method and the biological brain structures needs to be strengthened. At present, the explanation feels rather superficial and unconvincing.

**[W2]** The paper devotes a large amount of space (around 7 pages) to describing the method and its connection to the brain, but the experimental and analytical sections are very limited (less than 2 pages). The paper feels more narrative-driven than methodologically innovative.

**[W3]** The experiments are insufficient. The evaluation is only conducted on small Qwen models, and several results are missing — e.g., GSM8K lacks Qwen3-1.7B evaluation, MATH500 lacks Qwen2.5-0.5B, and AIME25 lacks Qwen2.5-0.5B. Therefore, the results do not convincingly demonstrate the effectiveness of the proposed approach.

**[W4]** Token length is not an ideal metric for measuring efficiency, since the introduction of a router modifies the model structure. Reporting inference latency would provide a more reasonable and fair comparison.

**[W5]** The method description is not sufficiently clear. I spent a considerable amount of time trying to understand the proposed framework, and Figure 3 fails to clearly illustrate the core design.

### Questions
Please refer to the weaknesses section above.

### Soundness
2

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
This paper proposes a token-adaptive routing framework to enhance the performance of LLMs by providing a second opportunity to answer questions at the token level. Inspired by neuro-linguistic pathways, the framework establishes a brain-inspired self-correcting loop that integrates seamlessly with LLMs without requiring additional fine-tuning. Through experiments conducted on two LLMs and comparisons with baseline methods, the framework demonstrates significant performance improvements.

### Strengths
1. The token-level framework does not require fine-tuning the LLMs, offering a self-correcting loop that enhances performance.

2. The framework design is grounded in theories of human brain function, making it more natural and theoretically sound.

3. The paper presents well-motivated research objectives, and the framework components effectively address these motivations.

### Weaknesses
1. As mentioned in the strengths, the framework does not fine-tune the LLMs directly, but instead tunes a router to enhance their representations. However, I would argue that this approach resembles LoRA-based fine-tuning, which also avoids modifying the main model parameters but introduces additional trainable components. Although the router operates differently from LoRA, both approaches still require training. From the paper, it is unclear whether the baseline model (i.e., Qwen) was fine-tuned with LoRA or not, but the router clearly involves training. Therefore, comparing a LoRA-based version and a router-based version would provide a fairer evaluation than the current setup.

2. The paper appears to draw inspiration from concepts in brain science, but the connections between these concepts and the proposed framework are not clearly established. Providing more details on how these ideas relate to the framework would help clarify the motivation and theoretical grounding.

3. There has been extensive research on enhancing LLM performance by modifying or reinterpreting their representations in a plug-and-play manner, without training any additional modules. How does the proposed framework compare in this regard? Can it function as a plug-and-play method, or does it necessarily require additional training?

### Questions
See Weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Token Adaptive Routing (TAR), a closed-loop mechanism that lets LLM detect and correct token-level semantic defects during inference without fine-tuning the backbone weights. The paper is motivated by how humans think and implement a "which-where-how" process. A semantic defect monitor to flag low-confidence tokens, an adaptive router that selects the most compatible model block, and lastly, a feedback-based representation that reinjects the token into the block to repair its semantics.

### Strengths
1. The papers introduces a TAR which is insipred by neuro-linguistic pathways that are present in the human brain
2. The method shows improved performance without fintuning the backbone weights.

### Weaknesses
1. The methodology section is hard to follow. The authors introduce a lot of components, where the motivation of each of the components within the router is missing. For example, the role of the ability vector and the requirement vector is hard to follow.
2. Does the method only select a single block? If yes, why is only a single block necessary for larger models, since multiple layers can be necessary to recalibrate a token
3. How many times does the self-correction loop go on? Is it a single loop, or is there an exit based on the confidence of the tokens?
4. Since the author introduces a lot of components in the training, ablation results are necessary to verify the importance of each component; however, most of the results have been moved to the appendix.
5. The authors train the model on AIME2024 and then test it on AIME2025; there is a strong chance of contamination here.
6. The model only focuses on math reasoning. Why was no experiment run on OOD tasks?
7. The paper chooses very small model,s and the router might not work properly when the number of layers increases in the model

### Questions
1. Do we require a separate router for each model?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents TAR, Token Adaptive Routing, a method for self-correction of LLMs inspired by neuroscience, pathways in the brain that are non-directional and thus not only forward (one directional), for the task to correct incorrect LLM outputs.

The method is triggered when the output is of low confidence (threshold based), triggering an external adaptive router that looks at model-internal representations of layers ('ability vectors') to match best for so-called 'requirement vector' to match, to override the output layer ('to re-inject into block B_a() for feedback based re-presentation'). The adaptive router is trained with act as a router, by comparing SFT loss before and after routing to guide the router toward better policies.  In an LLM with L layers, each re-representation offers L + 1 routing options, and the paper tests all L + 1 policies to find the best. Finally, a regularization term is added to stabilize training. 

The method is tested on three math reasoning benchmarks, GSM8K, MATH500 and AIME25. LLMs tested are Qwen2.5/0.5B and Qwen3/1.7B and trained on math data generated by itself. Each adapter is trained for one epoch.

### Strengths
- The paper presents a method for self-correction of an LLM in math domains.

- The paper is quiet well written (see comment below) and the method and setup is clearly explained.

### Weaknesses
- **Limited evaluation.** The evaluation is severely limited.

  - **Overly strong claim**. The paper motivates the method as " token-level semantic defects" detection method. However, and most importantly, the method is essentially a *self-correction method for math reasoning problems*, I strongly disagree with a claim for "semantic defects" when tested only on simple math problems. The title uses "linguistic pathways" which is too strong if tested in such a narrow domain where arguably linguistics is not really the key to solve the problem. Instead, the paper proposes a math error detection method. It would be stronger to compare and contrast to related work model-internal injections on natural language understanding (e.g. like the multi-hop natural language understanding problem in related work in Biran et al.). Moreover, the writing, especially in the introduction does not situate the method in math reasoning, but claims to provide a bigger human-inspired solution for "semantic defects", which I find misleading. There are no experiments beyond math problems, thus the claim of the current paper writeup is too strong and not supported by empirical validation. 
  -  **Small LLMs only**. The method tests only two small LLMs and these are from the same model family (Qwen). This severely limits generalization. The method should be tested in at least two different model families to test generalizability beyond small Qwen models.
  - **Lack of comparison to upper bound or other method** The method does not compare against any existing method. For example,  the activation space like back-patching (Biran et al.'s method) could be applied by identifying a simple prompt that 'hints' at the solution by rephrasing the math problem (if the task is to solve 2 + 2 = 4, the test could be 1+3 = ?) and creating a probing classifier to identify the layer. This would also be a more lightweight approach and could help understand if the (quiet complex) method is useful. At least one comparison method should be included.
   - **Improvements** in Table 1 seem small (up to 3.2% accuracy). This raises again the question whether the routing method is useful.
   - **Lack of details of hyperparameter** The method relies on a confidence threshold (if the confidence if low, the routing fires. However, the paper does not provide any information of what threshold is used, nor how it was determined, nor how sensitive the method is to this threshold or how generalizable it is across the three math datasets. This is an important aspect left undermined.

- **Repetitive text parts**: The neuroscience inspiration part is quiet long and repeated 3x in the paper (abstract, into, method section)."TAR comprises three core components, each inspired by a function of the biological neuro-linguistic pathways ..." 


- **Lack of complexity analysis** The method needs to identify which layer out of all layers for each token. This is very expensive. This is also perhaps why the paper only evaluates very small LLMs. The paper would be strengthened by providing a complexity analysis and judgement to what degree the method would scale up to larger LLMs.

### Questions
- how did you determine the threshold? how sensitive is your approach to the threshold?

### Soundness
1

### Presentation
1

### Contribution
2
