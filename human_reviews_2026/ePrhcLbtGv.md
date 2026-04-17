# KaVa: Latent Reasoning via Compressed KV-Cache Distillation

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Large Language Models (LLMs) excel at multi-step reasoning problems with explicit chain-of-thought (CoT), but verbose traces incur significant computational costs and memory overhead, and often carry redundant, stylistic artifacts. Latent reasoning has emerged as an efficient alternative that internalizes the thought process, but it suffers from a critical lack of supervision, limiting its effectiveness on complex, natural-language reasoning traces.
In this work we propose KaVa, the first framework that bridges this gap by distilling knowledge directly from a compressed KV-cache of the teacher into a latent-reasoning student via self-distillation, leveraging the representational flexibility of continuous latent tokens to align stepwise KV trajectories. We show that the abstract, unstructured knowledge within compressed KV-cache, which lacks direct token correspondence, can serve as a rich supervisory signal for a latent reasoning student. 
Empirically, the approach consistently outperforms strong latent baselines, exhibits markedly smaller degradation from equation-only to natural-language traces, and scales to larger backbones while preserving efficiency. These results establish compressed KV-cache distillation as a scalable supervision signal for latent reasoning, combining the accuracy of CoT-trained teachers with the efficiency and deployability of latent inference.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces KaVa, a method to distill from reasoning chains from large models into smaller ones.
This method can be seen as similar to CODI (published Feb 2025, https://arxiv.org/abs/2502.21074).
But extends CODI by attempting to supervise over the whole chain of reasoning instead of just the ending.
This introduces an issue that the KV Caches for the teacher and student in the distillation are of different sizes.
The authors overcome this by using a KV cache eviction algorithm which is fairly optimal as it can rely on the whole trace (i.e. non-causal) to do eviction as it is not required for inference time.

The authors rely heavily on results from prior work when presenting their evaluations, copying accuracy values.
To validate their method the authors train on two training sets for GSM8K and see some improvement over CODI in most cases, but do achieve a good increase in speed at inference.
The authors ablate some design decisions such as their loss terms and removing the last step of the trace as seen in prior work.

### Strengths
- Eviction methods working in KV distillation at all is surprising
- Good speed improvement over CODI, with little/no accuracy compromise.
- Detailed explanation of method.

### Weaknesses
- The authors claim their method is novel as it uses eviction methods to reduce the size of the KV cache, unlike KV-Distill (March 2025, https://arxiv.org/pdf/2503.10337). However, the authors do not validate that this is in fact better.
    - This is a key contribution (1 of the 3 listed in section 1), hence I think it should be _a lot_ more rigorously justified.
   - Overall, I think this paper is a little to focused on beating a fixed set of baselines from prior work rather than exploring and rigourously testing their own, new, method.
- There is overlooked prior work on efficiency e.g. https://arxiv.org/pdf/2505.18962 (May 2025)
- Tables 1 and 2 contain a lot of copied values from prior work making it difficult to be sure that the evaluations and training methods are exactly the same. Moreover, there are many gaps in both tables and not all rows from table 1 are included in table 2.
- Limited to gsm8k in both training and testing.
- The authors discuss this method reducing the interpretability of reasoning traces but do not discuss the ethical impacts of this.
- Section 5.2 discusses results entirely in the appendix, slightly against the page limitation rules.

Minor: 
- Footnotes for page 5 appear on page 6
- Incorrect bolding in Table 1, 72.4 is bolded and 72.7 is underlined.

### Questions
1. Why is an eviction strategy better than KV-Distill empirically?
2. How sure can we be the values copied from prior work in Tables 1 and 2 are identical to the set up used in this paper?

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
This paper introduces KAVA, the first framework that distills knowledge directly from the compressed KV-cache of the teacher into a latent-reasoning student via self-distillation. This process is conducted in a step-wise fashion, removing the eviting KVs that do not are less important.

### Strengths
- interesting idea to distill from the KV-cache
- Overall, good comparisons with other baselines

### Weaknesses
- The biggest weakness of this paper is that the performance of the teacher caps any distillation process. 
- Lack of diversity in downstream tasks
- Writing
   - Move Figure 1 earlier in the paper (i.e pg 1-2)
   - The main benefit of this method seems to be the reduction of the inference cost. However, this result/point is presented over 2 pages instead of just having a single graph/table showing the Pareto frontier curve

### Questions
- Why is there not a complete comparison between all the methods and models? Is this from a resource limitation? I would suggest moving those comparisons to another table.
- How does the quantity or quality of the data affect the distillation? How can this method move beyond the limitations of current distillation? 
- How does the distillation process compare when distilled from its own CoT or a stronger model?

### Soundness
3

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
This paper introduces KAVA, a framework that distills knowledge from a teacher model’s compressed KV-cache into a latent reasoning student through self-distillation. By aligning continuous latent representations, KAVA enables efficient and supervised latent reasoning without explicit chain-of-thoughts. Experiments show that it outperforms strong baselines, maintaining high accuracy and scalability while significantly improving inference efficiency.

### Strengths
1. The paper presents a novel and timely attempt to bridge explicit chain-of-thought reasoning and latent reasoning through compressed KV-cache distillation, offering a promising direction for improving efficiency without sacrificing reasoning quality.

2. The proposed KAVA framework is well-motivated and technically sound, combining self-distillation with KV-cache compression to provide an effective supervision signal for latent reasoning models.

3. The experiments are comprehensive and show consistent improvements over strong baselines such as CODI and PCCoT, demonstrating both better reasoning performance and significant inference-time efficiency gains.

### Weaknesses
1. The paper demonstrates the potential of applying KV-cache compression to latent reasoning, but it does not clearly explain how its KV-cache design differs from or improves upon existing approaches.

2. All experiments are conducted on relatively small models (0.5B–3B), without evaluating performance on larger-scale models. Is this due to computational limitations, algorithmic instability, or other design considerations?

3. While the paper claims that the compressed KV-cache serves as a “rich supervision signal,” it lacks theoretical or information-theoretic analysis to justify why KV representations would preserve reasoning structure better than hidden states.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a novel way to enable latent reasoning capabilities for thinking models. LLMs with explicit CoT are known to excel today at highly challenging multi-step reasoning problems. However, the explicit CoT paradigm is very expensive and requires a lot of computational and memory overhead during inference. To reduce this, reasoning in the latent space has been suggested as an alternative. However, the methods suggested so far for latent reasoning, suffer quality losses compared to explicit CoT. The paper posits that a main reason for this is we cannot provide supervision in the latent space. To remedy this, the authors propose KAVA: a method that takes in a teacher model with explicit CoT, compresses its KV cache, then distills the compressed KV cache down to a student model (self-distillation - student model is same size as the teacher model) to enable latent reasoning abilities.

KAVA relies on some recent studies which show that KV caches underlying CoT are highly redundant and can be compressed heavily along the sequence length dimension. This motivates the possibility of providing supervision for latent reasoning via a compressed KV cache. The training for KAVA proceeds as follows:
1. The tokens are split into 3 parts: question Q, reasoning trace C, answer A.
2. We then apply a redundancy-aware KV-cache compression scheme to the teacher cache of the reasoning trace. This works by computing an importance and a redundancy score for each token and choosing the top M tokens to match the student’s latent reasoning length. The importance score of a token is simply its attention weight. The redundancy is computed as the average pairwise cosine similarity among all keys and normalized by a softmax.
3. Then we perform distillation of these compressed KV caches to the student using an L_p loss on the embeddings. The p is chosen by hyper parameter sweep and ends up being either 1 or 2 depending on the specific dataset and model. The student’s latent embeddings are generated in parallel using Jacobi decoding for 3 iterations.


The authors apply this method to train fine-tuned latent reasoning student models from Llama 3.2, 1B and 3B pretrained models and Qwen 2.5 0.5 B pretrained model.
They use LoRA fine-tuning with rank 128. They fine-tune the models on GSM8k-AUG and GSM8k-AUG-NL datasets which have ~400k examples (augmented versions of GSM8k) with CoT generated using GPT-4. GSM8k-AUG-NL is in natural language while GSM8k-AUG removes the natural language and only keeps equations.
Then they evaluate on the original GSM8k, GSM8k-Hard and SVAMP datasets. KAVA reduced the thinking trace size by around 80-90% while outperforming other latent reasoning methods. For instance, the Llama 1B model with full CoT gets 61% on GSM8k, and 30.9% with no CoT. KAVA gets 56.5% beating the next best latent reasoning method by 3%.

The authors ablate for different choices of the KV eviction loss, the L_p loss and other modeling choices.
They also perform an interpretability analysis of the latent reasoning traces but it’s unclear if the presented results offer statistically convincing arguments for the inferences made. The paper concludes with a nice visualization into the similarity of the teacher and student KV caches after distillation. They observe a higher similarity between a latent representation at position n with explicit CoT representations at positions m for n < m which is the behavior we want.

### Strengths
- The paper tackles an important and highly relevant problem for today’s LLM landscape. The proposed solution is novel and practical.
- The paper performs a good number of ablations for different hyper parameter choices. While it would have been nice to see experiments on more datasets than GSM8k, the choice seems guided by a number of past works in this space choosing to focus on GSM8k for models in the 1-2B size range.

### Weaknesses
- The main weakness is the limiting choice of the dataset. Ablations on larger models on more challenging datasets would have given stronger evidence to the efficacy and scalability of the proposed solution. The proposed method is interesting and promising but whether it works across datasets and scale is not fully established.

- The focus on fine-tuning and evaluating on a specific dataset or domain of problems raises the question on whether the proposed method hurts other abilities of the LLM. A more broader downstream evaluation across a suite of benchmarks would present evidence that we aren’t simply trading off one ability for another within the model. This perhaps would require moving away from tasks-specific LoRA fine-tuning.

### Questions
- It would also be interesting to see what would happen with full-fine tuning instead of just LoRA fine-tuning. Perhaps the gap to full CoT can be bridged in this case.

### Soundness
3

### Presentation
4

### Contribution
3
