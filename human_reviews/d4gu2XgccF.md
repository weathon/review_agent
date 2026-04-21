# MEMREASONER: A MEMORY-AUGMENTED LANGUAGE MODEL ARCHITECTURE FOR MULTI-HOP REASONING

- Avg Score: 4.00
- Decision: Reject
- Scores: 3, 6, 3

## Abstract
Recent benchmarks suggest that there remains significant room to improve large language models’ ability to robustly reason across facts distributed in extremely long documents. In this work, we propose MemReasoner, a new memory-augmented LLM architecture that is trained to perform temporal reasoning, along with multiple computational steps,  over the context stored in the memory. Experiments show that MemReasoner trained on the core reasoning facts generalizes better, when compared to off-the-shelf large language models and existing recurrent models, on a test distribution  where  the required facts are scattered across long natural text up to 128k tokens. Further, MemReasoner demonstrates robust reasoning performance relative to the baselines, when the answer distribution in test samples  differs from that in the training set.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes MemReasoner, a method that augments language models with memory and is designed to perform reasoning over long contexts. The authors aim to enable models trained on short sequences to generalize and work effectively on longer sequences without retraining. To achieve this, they build on the Larimar model, which augments language models with episodic memory, and enhance it with two key modifications: 1. An iterative mechanism to read information from memory and update the query accordingly. 2. Adding explicit information about temporal ordering of facts within the context (via position embeddings or BiGRU). The approach is evaluated on the bAbI dataset and its long sequence variant BABILong, demonstrating the ability to perform on longer sequences while being trained on short ones, and improving performance over existing methods in this setup.

### Strengths
- General idea of iterative reading from memory for multi-hop tasks is clear and reasonable.
- The motivation to address the challenge of generalizing from short to long sequences without extensive retraining is strong and relevant, and addresses a key limitation of current methods.
- Results show that the proposed approach is indeed able to generalize to longer sequences being trained on short only.
- The evaluation setup, using the bAbI dataset and its longer version, is clear and suitable for testing the ability of the method to solve inference tasks over varying sequence lengths.

### Weaknesses
- Giving that idea is clear, unfortunately Section 3 that describes the proposed method is hard to read and lacks clarity. I believe clarity of presentation of the proposed method should be largely improved. Details are provided in the suggestions section. 
- Lack of comparison with Transformer models. The authors do not evaluate the performance of transformer-based LMs that support long contexts (e.g., LLama-3.2, Phi-3.5, or Qwen-2.5). These models could be fine-tuned on bAbI and evaluated on BABILong, providing relevant baselines for comparison. Including such evaluations would contextualize the performance of MemReasoner relative to current state-of-the-art models.
- Generalization of the proposed method to tasks other than bAbI is not supported. There are other datasets that require multi-hop reasoning, such as MultiHopQA, MuSiQue, HotPotQA. Their context could be extended by extracting relative paragraphs from e.g. Wikipedia.
- The paper does not provide empirical evaluations of the method’s inference time or memory consumption compared to other methods. However, authors provide theoretical time complexity in Appendix A.2.

### Questions
Suggestions on improving presentation:
- Figure 2 is hard to comprehend – it is not clear where to start from. I would suggest making two schemes: 1. conceptual scheme of the method. 2. a detailed one with how the memory module and the query are iteratively updated.
- Similar problem with Section 3 - overloaded preliminary section hindering comprehension. Section 3.1 contains too many details on how memory read/write operations are constructed, making it challenging to grasp the overall picture. Splitting the explanation into two parts – a conceptual overview and detailed descriptions of the memory module and reasoning within it – would help readers understand the main ideas before diving into technical specifics. As the method is based on the Larimar model, a dedicated section describing Larimar would greatly improve clarity.
- L292-296 present formulas for the loss components that are difficult to match with their explanations. Defining each loss component separately and explaining their roles can improve clarity and comprehension.
- How do P_a and P_s look like in L270–271? P_a is later defined in L371–372. Providing these definitions earlier would help improve clarity.
- Typos and Minor Issues
  - L315: "facts from bAbI is distributed" -> are
  - L318–319: missing space "tokens.For"

Comments:
- One of the results is that RMT being trained on short sequences (single segment) can not generalize to longer ones (multiple segments). RMT inherently can not generalize from 1 segment to multiple segments. To learn to use memory (to pass information sequentially) it needs at least 2 segments. It was shown in RMT paper and generalization to larger lengths is possible with curriculum learning procedure (“Beyond Attention: Breaking the Limits of Transformer Context Length with Recurrent Memory”). So this makes comparison with RMT not so fair.

Questions:
- Table 1 and 2 miss results for RMT on 64k and 128k. Authors mention that it “means unavailable due to out of memory errors or maximal input length constraints” (L426). However, RMT has constant memory consumption as it is required to pass only memory states between segments, and there is no need to all keep intermediate outputs. So, it is not clear why these values are missing? However, I acknowledge that they would be low.
- Is it possible to train MemReasoner on longer sequences and compare it with RMT/Mamba which has also been trained on longer sequences? Will the performance of MemReasoner continue to improve?
- L182 - what does “mimicking Bayesian inference” mean here? How is it motivated and why is it mimicking? It is unclear how this concept is applied and whether it is key to constructing an effective memory module.
- After the last readout operation, the z_i most similar to z_r is selected. Z_i, representing a single fact from bAbI, goes to a decoder. It seems to be sufficient for QA1-2 tasks to get the correct answer from bAbI, but would it be sufficient for general tasks to extract only one fact from memory? Why not to use z_r itself?
- The paper does not clearly explain whether the number of memory read operations correlates with the number of fact hops required in reasoning tasks. For example, in QA2 tasks, a fixed number of two read operations is used. Is this number optimal? Would using only one read operation be insufficient to retrieve the answer? Providing an analysis or justification for the chosen number of read operations would clarify this aspect.

Personally, I like the idea and direction of the work and find it interesting. However, I think it could be improved a lot.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces MemReasoner, a memory-augmented LLM designed to improve temporal and multi-hop reasoning over extremely long contexts. MemReasoner integrates an episodic memory module and introduces a novel mechanism for explicitly learning temporal relationships between events and for iteratively reading and updating queries, aiming to address the limitations of existing LLMs in processing and reasoning across long documents. The proposed model is evaluated on the BABILong benchmark, demonstrating superior generalization compared to existing LLMs.

### Strengths
- The authors present an architecture that uses a memory module for capturing temporal relationships and enabling multi-hop reasoning, which adds a valuable improvement over standard LLMs.
- MemReasoner demonstrates a robust ability to reason across contexts of up to 128k tokens, outperforming existing LLMs and memory-augmented architectures on the challenging BABILong benchmark.

### Weaknesses
- The experiments are largely based on synthetic datasets like BABILong, which, while controlled, may not fully reflect the complexities of real-world language tasks. Extending evaluations to diverse, natural datasets would strengthen the validity of the model’s utility.
- The architecture involves multiple components like GRU-based temporal encoding and iterative query updates, which might increase computational complexity, potentially making it less scalable for broader applications.

### Questions
- What improvements does MemReasoner have compared to Larimar?
- Has the model been tested on other multi-hop reasoning datasets or more general NLP tasks?
- How does MemReasoner compare to traditional, non-LLM memory networks? Are there any experiments comparing it to these earlier methods, and what would the outcomes likely be?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
MemReasoner is a novel encoder component of an encoder-decoder transformer architecture. On top of a transformer encoder stack, it uses a recurrent component to write memory and read memory.

The readout is fed into the decoder similar to key-value cache to improve performance on question-answeringe tasks.

The paper benchmarks their method on BABILong. Their method solves the task as shorter contexts but does not seem to length generalize. However, their method does do better than the baselines when symbolically manipulating locations.

### Strengths
The hypothesis and method are well motivated. The experiments are well done and the model well in short contexts and is robust to symbolic manipulation of the locations.

### Weaknesses
I don't think I entirely understand the comparisons. The model does much worse than the models that are finetuned on long context data. We should see if MemReasoner also benefits from this?

### Questions
The encoder and memory components have some cost and overhead. What are they?

How does it compare to just give the decoder more parameters?

Can MemReasoner augment existing decoder-only LLMs?

### Soundness
3

### Presentation
2

### Contribution
2
