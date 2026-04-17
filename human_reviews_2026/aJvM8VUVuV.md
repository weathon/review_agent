# LIFT: Enhancing Long-Context Reasoning in Large Language Models via Long Input Fine-Tuning

- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Long context reasoning remains challenging for large language models due to their limited context windows. This paper presents Long Input Fine-Tuning (LIFT), a novel framework for long-context modeling that can improve the long-context reasoning performance of arbitrary (short-context) LLMs by dynamically adapting model parameters based on the long input. Importantly, LIFT, rather than endlessly extending the context window size to accommodate increasingly longer inputs in context, chooses to store and absorb the long input in parameter. By fine-tuning the long input into model parameters, LIFT allows short-context LLMs to answer questions even when the required information is not provided in the context during inference. Furthermore, to enhance LIFT performance while maintaining the original in-context learning (ICL) capabilities, we introduce Gated Memory, a specialized attention adapter that automatically balances long input memorization and ICL. We provide a comprehensive analysis of the strengths and limitations of LIFT on long context understanding, offering valuable directions for future research.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper propose a method that teaches a LLM a long document by automatically creating question-answer pairs from the text and then fine-tuning the model on these pairs.

### Strengths
It is interesting and worth exploring to absorb long context ability to model parameters.

### Weaknesses
1. The methodology lacks novelty. LIFT essentially uses an LLM to synthesize multiple QA pairs and then uses them for training. This is a common and standard practice in improving long-context ability [1][2]. The authors have not clearly explained the innovative or unique aspects of their method.

2. The motivation for using an asynchronous producer-consumer pipeline is unclear. In my view, it would be simpler and feasible to synthesize all QA pairs at first, and then proceed with fine-tuning. It is not necessary to use asynchronous training pipeline.

3. I am skeptical of the claim on lines 196-203: "no matter how low the loss is, the model fails to answer faithfully according to the sentence it just trains on." Although the model failed to learn from a single sample after 6 epochs, the loss was still decreasing and had not converged. I suggest the authors provide the model's answer upon loss convergence.

4. The statement on line 205, "training on the long input does not guarantee the model can truly understand the knowledge...," is an overclaim, as the sentences used are actually less than one hundred tokens.

5. Expressive Issue: "motivating experiment" should be revised to "motivation experiment."

[1]Xu, Peng, et al. "Chatqa 2: Bridging the gap to proprietary llms in long context and rag capabilities."

[2]Li, Jiaxi, et al. "Wildlong: Synthesizing realistic long-context instruction data at scale."

### Questions
Is the proposed LIFT method applied to each individual benchmark sample before evaluation, or is it applied to all samples collectively? If it is the former, the process would be highly time-consuming and impractical. If it is the latter, LIFT closely resembles the classic paradigm of supervised fine-tuning with synthetic data, and thus requires a comparison with other methods for synthesizing QA pairs.

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
This paper proposes LIFT (Long Input Fine-Tuning), a test-time training framework that turns a short‑context LLM into a "LIFTed" model for a specific long input. Instead of extending the context window or relying on retrieval, LIFT splits a long document into sentences, uses a strong generator model to create multiple synthetic QA pairs per sentence, and fine‑tunes the target LLM on these QAs so that the knowledge is absorbed in LLM parameters. The method is paired with an asynchronous producer–consumer pipeline that overlaps QA generation and SFT. Extensive experiments conducted on long-context reasoning benchmarks such as NIAH and LooGLE shows that LIFT achieves the best performance among comparing methods.

### Strengths
1. **Clear presentation.** This article is well-written and the LIFT method is presented clearly. 
2. **Practicality values of the LIFT method.** LIFT presents another paradigm for processing long-context inputs besides letting the LLM do all the work at once. The design choices are well‑motivated and appropriate under the described settings. Processing long-context with a shorter context window also shows significant efficiency over traditional long-context extrapolation methods.

### Weaknesses
1. **Confusing NIAH experiments.** Long-context benchmarks, such as NIAH, are designed to test "how LLMs locate correct information given a long input document".  Table 5 in the article shows that when tested on NIAH, the target LLM is **simply trained on the answer itself**. In my opinion, I don't think such evaluation approach is appropriate, as the core problem of NIAH is not learning the correct answer, but locating them. Fine-tuning on the answer eliminates the *locating* process.
2. **Narrow evaluation scope.** LIFT is only evaluated on NIAH and LooGLE. More long-context reasoning benchmarks, such as LongBench, although referenced, are not evaluated on LIFT. Broader evaluations are recommended in revision.
3. **Missing baselines.** Besides the listed baselines, it is suggested that LIFT should be compared with a number of long-context extrapolation baselines, such as [1-3], as they are also efficient (some of them does not require any training).

[1] LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models

[2] LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning

[3] Extending LLM Context Window with Adaptive Grouped Positional Encoding: A Training-Free Method

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes to a method to fine-tune an LLM to handle long input by (1) generating synthetic Q,A pairs from the long document and (2) conducting SFT on the generated QA pairs. Compared to RAG and MemoryLLM, the proposed method  achieves better performance on Llama-3-8B-Instruct model (with short context window).

Overall, I think the method is interesting but not particularly novel (similar ideas have been explored under other settings before), the empirical results appear weak, and the motivation lacks sufficient clarity and justification. Please see detailed comments below.

### Strengths
* The paper is written in a clear manner.
* The workflow presented in Figure 1 to improve efficiency of the system is interesting.

### Weaknesses
* **Weak empirical performance**: In Table 4, most of the performance gain come from doing LIFT together with having the original document at either inference time (truncated ICL) and training time (segmented LM), this sort of indicates that the proposed method itself is not as effective.
* **Missing baselines/ablation**: To understand the effectiveness of the method, it would make sense to present ablation results training with segmented LM and inference with truncated ICL. While the paper considers two baselines (MemoryLLM, LlamaIndex), I believe there are other baselines that also enable LLM with short context to handle long input, to name a few: (1) [LLoCO(EMNLP2024)](https://arxiv.org/abs/2404.07979); (2) [LLMLIngua-2(ACL202)](https://arxiv.org/abs/2403.12968).
* **Motivation**: The paper motivates the approach to enable LLM trained with short context to handle long documents, yet currently most of the LLMs (LLaMA-3.1 or Qwen-2.5) can handle long context of up to 128K. If we can pass in the entire document to an LLM and that could achieve good performance, what's the benefit of the proposed approach? I think one area might be to further extent the window (e.g. up to 1M), if that's the case, i think experiments covering those settings would be helpful.
* **Contribution**: Fine-tuning on (synthetic) QA pairs to enhance knowledge acquisition has been studied before: [PIT (ACL, 2024)](https://arxiv.org/abs/2402.12847) and [EntGraph(ICLR 2025)](https://arxiv.org/abs/2409.07431), as well as the LLoCO paper mentioned above, so the presented novelty of the paper is a bit weak to me.

### Questions
Table 6 reports the first token latency of LIFT, how does that compare to the baseline methods considered?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Long Input Fine-Tuning (LIFT), a method designed to enhance long-context modeling by enabling short-context LLMs to dynamically adjust their parameters in response to extended inputs. Through this approach, models trained on short contexts can effectively handle queries even when relevant information is missing from the inference context. Additionally, the authors propose a Gated Memory mechanism, an attention-based adapter that automatically manages the trade-off between long-input memorization and in-context learning (ICL).

### Strengths
This paper introduces Long Input Fine-Tuning (LIFT), a method designed to enhance long-context modeling by enabling short-context LLMs to dynamically adjust their parameters in response to extended inputs. Through this approach, models trained on short contexts can effectively handle queries even when relevant information is missing from the inference context.

### Weaknesses
1. The paper does not clearly define how short-form QA pairs are constructed, which is critical since this process determines how the LLM learns and performs during inference. In Line 243, the authors mention that these QAs may include "*simple details such as specific people, time, locations of events, or more general reading comprehension ones.*” However, it remains unclear whether such details are fixed or dynamically generated, and whether these aspects sufficiently cover all information required for complex reasoning tasks. Moreover, the short-form decomposition process may overlook long-term dependencies due to sentence-level tokenization.
2. The proposed method is heavily tailored to the advanced Qwen2.5-72B model. This introduces two major concerns: (1) a strong dependency on a specific base model, limiting generalizability, and (2) increased computational and deployment costs, including API usage, local model hosting, fine-tuning overhead, and additional inference costs per data sample.
3. The experiments primarily focus on Memory, LlamaIndex, and ICL, lacking comparisons with more advanced prompting techniques. Furthermore, there is a notable performance drop in certain tasks, particularly **Comprehension & Reasoning**, where accuracy decreases from 33.00 to 28.82 (**-12.67%**). This decline may stem from Weakness 1, as long-form comprehension and reasoning, which is essential for comprehension tasks, appear underrepresented in the proposed setup.
4. The study employs GPT-4.1 nano as the LLM-as-a-judge, which deviates from the original evaluation setup. It is uncertain whether GPT-4.1 nano possesses sufficient evaluation capability for this purpose, potentially affecting the reliability of the results.

### Questions
What does the following sentence in the Abstract mean:

> *LIFT allows short-context LLMs to answer questions even when the required information is **not provided in the context during inference.***

### Soundness
2

### Presentation
2

### Contribution
1
