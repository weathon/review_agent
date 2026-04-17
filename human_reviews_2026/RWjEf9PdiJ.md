# TokMem: One-Token Procedural Memory for Large Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 0, 6

## Abstract
Large language models are typically controlled via prompts, which must be repeatedly re-processed for every new query and are difficult to reuse modularly. We introduce TokMem, a procedural memory framework that compiles each reusable task procedure into a single trainable memory token. Each token serves as both a procedure index and a generation control signal that steers generation, enabling targeted behaviors with constant-size overhead. TokMem keeps the backbone LLM frozen and stores procedural knowledge entirely in these dedicated units, so new procedures can be added continually without interfering with existing ones. We evaluate TokMem on two settings: atomic recall over 1,000 Super-Natural Instructions tasks and compositional recall on multi-step function-calling. Our results show that TokMem consistently outperforms retrieval-augmented prompting while avoiding repeated context overhead. Moreover, it matches or exceeds parameter-efficient fine-tuning with substantially fewer trainable parameters.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies the problem of efficiently storing and applying frequently-used procedures (such as calling certain tools) in large language models as opposed to using lengthy prompts or retrieval mechanisms. Specifically, the paper proposes TokMem, which encodes procedural knowledge as trainable continuous memory tokens. The paper evaluates on two settings: using TokMem for single tasks in Super-Natural Instructions and compositionally invoking on function-calling tasks. The results suggest that TokMem outperforms retrieval-augmented generation and parameter-efficient fine-tuning while using fewer parameters.

### Strengths
The experimental setting covers multiple models across different scales. The experiment also covers both single task setting and compositional setting.

The results demonstrate good empirical results of TokMem, showing its improvements over LoRA finetuning.
 
The proposed approach is carefully designed. For instance, TokMem uses the renormalization technique to prevent new tokens from dominating routing.

### Weaknesses
I feel the paper needs substantial improvements in terms of clarity.
* The paper heavily uses the term "procedure", but does not give a precise definition of "procedure."
* In particular, it's unclear whether the response in line 129 represents a fixed set for each procedure or can vary.
* The implementation details of memory tokens are ambiguous. Are they essentially implemented as newly introduced special tokens added to the vocabulary?
* I am also not so sure about the routing mechanism. Does the model predict a newly added special token that indexes a procedure during inference?
* DC (decoupled embeddings) seems to also play an important role in some results but it is mainly covered in Appendix.

The choice of baselines mainly uses LoRA finetuning. It would be beneficial to include full fine-tuning as a reference to discuss the trade-offs between the number of parameters updated and performance.

The paper misses discussion of important related work on prompt compression. Methods like Gisting (Mu et al., 2023) and AutoCompressor (Chevalier et al., 2023) that compress prompts into tokens share similar motivations and should be

I'm curious whether new special tokens are necessary. What if we used short natural language descriptions with special token wrappers like <call>tool name</call>, but similarly only tuned representations of these tokens? This approach wouldn't require adding new embeddings for every new procedure.

### Questions
See weakness

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes TokMem, a tokenized memory that stores recurring procedures as compact, trainable embeddings. In particular, frequently reused procedures are “compressed” and stored by encoding them into an internalized memory token. To support continual adaptation, TokMem keeps the backbone model frozen, allowing new procedures to be added without interfering with existing ones.

### Strengths
- The paper is generall well-written and easy to follow.

- The proposed method is compared to multiple valid baselines, with complementary experiments to understand the design choices.

- Memory tokenization naturally reduces context size and supports continual learning without interfere

- The backbone is frozen during the training, offering effciency and avoiding catastrophic forgetting.

### Weaknesses
The method requires an additional adapation stage for compositional tasks, which seems to break the 'frozen backbone' claim? Also, is it fair to compare 'TokMem + adapt' to fine-tuning? While it is true that the stored procedures are modular via independent memory tokens, the comparison of TokMem without adaptation to fine-tuning in Table 3 seems to suggest limited capability of composing the modular procedures.

### Questions
1. Is TokMem+DC using two tokens for each procedure/task? If yes, is it a fair comparison to TokMem? And can you further extend to more tokens?

2. Why is renormalization needed, what is the intuition behind? I see it's helpful empirically but not sure if I understand the reason at Line 150-151. In particular, why new embeddings would develop inflated norms?

3. Clarity: I found the following points confusing to me, requiring more time to catch the ideas.

    - At Line 214-215, it is mentioned that both RAG and TokMem are with 'explicit' memory routing. IMO TokMem is doing 'implicit' routing since the memory token is chosen via next-token prediction, unlike RAG or MoE who has dedicated routing mechanism/architectural component. Can you clarify?
    - Are the training examples formulated by inserting memory tokens between query and response of the original datasets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
2

### Summary
The paper introduces a method augmenting a language model with a bank of trainable “procedure” embeddings. The model is trained on sequences of query tokens with interleaved "procedure" embeddings and response tokens.

The method is evaluated on "Atomic Memory Recall" and "Compositional Memory Recall" setups.

### Strengths
At present I’m unable to identify clear strengths because the core method and its usage at inference are unclear. As a result I can’t assess the contribution or empirical value with confidence.

Conceptually, representing procedures as tokens could be interesting.

### Weaknesses
The paper is highly unclear and have multiple problems, for example:
1. The inference process is not described at all. During training, the model learns to predict a sequence of [query tokens]+interleaved [memory token][procedure tokens] -- what happens during inference? Are we appending procedure tokens to the input? In such case, it contradicts the motivation where the method is presented in opposition to e.g. RAG methods that inflate the context window. If the method is not injecting textual tokens of the "procedures", then it's another PEFT method, and should be compared to different PEFT baselines, including prompt-tuning, as well as other dynamic adapter approaches like Mixture-of-LoRAs.
2. As the inference is unclear, so is the evaluation. For the "Atomic Memory Recall" setup, the tasks from Super-Natural Instructions dataset are regarded as the "procedures" in the memory bank. What is exactly evaluated here? Given test query, model predicts the memory token, then we append the text tokens of corresponding "procedure" and evaluate that with Rouge-L?
3. Model is trained on data in the sequential order, which is contrary to the standard of shuffling the data for _stochastic_ gradient descent.
4. Fine-tuning baseline includes only LoRA applied to query & key projections of the attention layers. It should involve also full fine-tuning, or at least LoRA applied to all linear layers, and not an arbitrary subset of them.
5. What was the procedure of selecting hyperparameters for training? For example, learning rate for training LoRAs seems to be too low. Moreover, optimal learning rate usually differs across model sizes -- here the authors provided only a single value.
6. "Routing" is mentioned multiple times in the paper but never introduced or defined.
7. The method is described (e.g. in the abstract) as keeping the backbone frozen. However, in the second evaluation setup the model is initially finetuned before applying the TokMem.

As the authors claim in the Appendix, the paper was written with the help of ChatGPT, which might partially explain the level of presentation of the paper.

After clarifications, I’m open to revising my score.

### Questions
All the questions are listed in the weaknesses above.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces TokMem, a trainable memory module that can augment existing language models and steer their behavior. The memory module is represented as a bank of memory tokens that can be retrieved and appended to the context on the fly, and each memory token is trained to be associated with a response. Only the memory embeddings are updated during training while the base model remains frozen.
The experiments include popular approaches, such as ICL, RAG,  FT, and reply memory, and test them on Super Natural Instructions and APIGen, representing atomic and compositional memory.

### Strengths
The paper presents a novel approach to managing memory for language models by fine-tuning just a memory module. The results are solid with comprehensive comparisons with different baselines. The design choices are solid and backed by ablation studies. It’s also great that the paper consider different types of memories like atomic and compositional memory.

### Weaknesses
In terms of the experimental settings, the original Super Natural Instruction paper uses unseen tasks for testing, but this paper uses the same tasks for training and testing. As a result, it’s unclear how the TokMem can help existing language models generalize to new tasks. Do you have evaluations on applying TokMem to the seen unseen tasks?

The presentation could benefit from how the memory tokens are used during inference. (see below for some questions on the procedure)

Typo: line 312 “fine-tine” → “fine-tune”

### Questions
How are the memory tokens retrieved during inference?

### Soundness
3

### Presentation
2

### Contribution
3
