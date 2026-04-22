# HybridCoT: Interleaving Latent and Text Chain-of-Thought for Efficient Reasoning

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
Verbalizing intermediate steps in token space has been central to eliciting reasoning in large language models (LLMs), with longer reasoning generally improving performance but incurring substantial compute and memory costs. Prior attempts to improve efficiency—such as KV-pruning or latent-space reasoning---often suffer from loss of accuracy or training inefficiency.  We propose HybridCoT, a framework that interleaves latent and text reasoning tokens in context. Our method reduces the compression errors that troubles previous latent CoT methods by keeping critical text tokens like math operations, in context, while compress semantic reasoning into the latent space. In addition, we design in-context text-to-token distillation to provide explicit supervision and iterative parallelized latent rollout methods to improve training efficiency for latent token, while shortening reasoning paths for efficiency. On challenging math reasoning benchmarks including AIME and MATH, HybridCoT achieves 94\% of the performance of finetuned text-only CoT models with 1.97× less inference compute, and surpasses efficient baselines (LightThinker and StreamLLM) by 1.36× and 1.26×, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduced Hybrid-CoT a method for using both soft (latent) and hard tokens at test time. The key part to this paper is that the full context is not replaced with soft tokens, some hard tokens are kept in the context.

During inference the model decodes any number of hard tokens followed by a \<latent\> token, which switches the model into producing m (fixed hyperparam) number of soft tokens.
Each token has the choice to attend to either: previous math tokens (marked with \<math\> tokens prior by the model); all previous latent tokens; or all tokens in the current block. This means that the attention is very sparse, hence fast.

The model is trained by having a stronger model annotate \<math\> tags into the training data and \<latent\> tokens are added at the end of paragraphs or sentences. So the trained model can effectively use both \<math\> and \<latent\> tags.
To avoid most of the increased cost of training a COCONUT style objective the authors use an approximation.
This approximation allows all latent tokens to be rolled out in a smaller number of iterations by using older intermediate latent token values for future latent tokens instead of always using the newest; meaning multiple latent tokens can be rolled our concurrently, the level of approximation is controlled by a hyperparameter.
Overall the authors achieve noticeably higher results than the baselines and are more efficient over many difficult benchmarks.

### Strengths
- Compares on two large models, although they are both Qwen based.
- Is more efficient and achieves higher accuracy on benchmarks.

### Weaknesses
- Number of latent/soft tokens decoded is a hyper parameter and therefore fixed.
- Method and writing feels overly complicated, simple explanations would benefit the paper a lot.
    - Section 3.2 is not well explained, please make this much clearer. The convergence guarantee here is hand wavy, please make it rigorous or remove it. I would say this is the highest priority during rebuttal.
- Very math benchmark focused.
- Lacking baselines: around mixing soft and hard tokens e.g. https://arxiv.org/pdf/2505.18962 (May 2025), https://arxiv.org/pdf/2502.21074 (Feb 2025)
- Minor: please bold the highest numbers in Table 1 to make it more readable. 

I think the main issues with this paper are lack of reasoning as to why hyperparameters are chosen and presentation of the new method.

### Questions
1. Why not use the model which annotates \<math\> tokens to also add \<latent\> tokens, instead of using sentences and paragraphs?
2. The training data is obtained by using a strong model to annotate data. Is this unfairly distilling extra information from the stronger model into your method?
3. How does this method impact memory usage versus the baselines?
4. Figure 5 decreases even more for L=3 or 4, why choose L=2 over these?
5. Table 2 only considers m up to 9, but the accuracy is still increasing, why stop at 9?
6. Does this method extend outside of the Qwen family?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a new approach, HybridCoT, whose aim is to reduce the length of the reasoning chains via the use of "Thinking Tokens". These tokens are interleaved with some regular tokens (i.e. explicit math tokens), to avoid over-compressing the reasoning chains. 

The method works as follows. Given a reasoning chain made up of multiple blocks (e.g. sentences or paragraphs), each block is interleaved with $m$ latent tokens. This structure enables the latent tokens to capture information specific to the given reasoning block (a la gist token), and be the main conduit of that information to the next block. That said, the authors allow for some relaxation of this, where some special tokens (math tokens), can also be attended to, in order to avoid over-compressing the chains. 

Importantly, the latent tokens are continuously generated, i.e. are not limited to a fixed set of augmented tokens. A learned linear transformation is applied on top of the penultimate hidden representation before feeding it back to the Transformer. 

The training procedure wraps the latent token insertion with special <latent> tokens. Crucially, the attention mask is modified so that a given token can only  attend to 1) math tokens from previous blocks, 2) the latent tokens from any previous step and 3) any text token in the current block. This forces the latent tokens to capture the block-level information. By predicting the text tokens at every step, this design allows for each latent token to receive an intermediate signal. 

The authors then propose a way to alleviate the slow training procedure, which whould require as many forward pass as `number_of_blocks` x `number_of_latent_tokens_per_block`. They show than one can reduce this cost to `number_of_blocks`, which can further be reduced if allowing for an approximation error. 

Experiments are conducted on a wide variety of mathematical reasoning benchmarks using the Qwen family of models. The OpenThoughs dataset is used for training the approach. Experiments are convincing, showing that the proposed approach navigates well the compute / performance tradeoff. Finally, the authors show via ablation studies that the training approximation of the iterative parallelized rollout indeed converges within a reasonable margin of error.

### Strengths
1. The approach is nicely designed. The interleaving of the latent tokens to capture step-wise information, the resulting finegrained training signal from the text token prediction, to the iterative parallelized latent rollout, the overall approach is well executed. 
2. The paper is well presented, the experiments are well targeted, and the ablations are properly built.

### Weaknesses
1. I am somewhat hesitant about the tailoring of the approach to math specific problems. How would you identify the special tokens which should not be compressed in settings outside of math ?

### Questions
1. How exactly are the gist tokens (used  in section 3.2) initialized ? Is this initialization learned ?
2. Why isn't the error in Figure 5 going to zero ? Given that the approximation is exact after $l$ steps for the first $l$ blocks, shouldn't it converge to zero ?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work presents a method for combining textual and latent reasoning methods in LLMs to increase both training efficiency and test accuracy. Existing interleaved latent and textual reasoning token approaches tend to have to roll out all intermediate latents autoregressively within a single sequence before the actual training update can be computed which greatly increases training costs over standard text only training.  However their approach trains for this inference behavior using a more efficient parallelized iterative strategy that caps costs at a lower number of forward evaluations by "relaxing the causal dependency" between blocks containing latent tokens. They present promising results showing near parity in terms of accuracy with text based CoTs and report efficiency improvements over their selected baselines.

### Strengths
1. The method is presented as a generalization of prior work on incorporating latent reasoning tokens (Lightthinker, COCONUT) suggesting the pareto-optimality of their solution (though this is hard to evaluate, see concerns) 
2. They perform experiments on full size models like Qwen3-8B rather than 1B or less
3. Their more efficient method recovers most of the performance of the standard text only CoT across the series of benchmarks considered for both models.

### Weaknesses
The points and questions below limit the ability of the reader to asses the novelty of the technique versus other methods and its reported efficiency edge. The current rating for the paper is primarily based on these issues and so their adequate resolution could improve the reviewer's assessment.

### Efficiency results are not communicated clearly

### 1. 

"1.97× less inference compute" is a confusing way to report improvement, please use a "achieve a 30% reduction" or "uses 50% of" to describe how the method is more compute efficient, unless the metric is tokens per second, or something where more is better, and in that case, say "1.97x tokens per second" or something explicitly.

### 2.

What is "Comp. (x10^8) in Table 1? In general it is not clear how computational cost/efficiency is computed and then compared for text CoT versus, StreamLLM versus Lightthinker, versus Hybrid CoT

### Description of latent token implementation is unclear

### 3. 

It is unclear how the L163 comment is supposed to differentiate this approach from Lightthinker. Later in S3.2 the "gisting" vocabulary is introduced, so, does this method also use a fixed vocabulary of latent special tokens g_1, g_2, etc added to the embedding matrix, similar the the fixed set of caching tokens? Is the maximum number of latents possible per block then limited by the chosen value of m before training starts? L274 attempts to help here but does not provide enough clarity. 

I think that one issue is the repeated use of the word "fixed" throughout the draft which doesn't have a clearly defined meaning in this context. Can the authors precisely explain what is "fixed" about Lightthinker, and how this matches the L=0 case of their algorithm? In my mental model, while a special token in the vocabulary and its embedding vector stored in the embedding matrix are in one sense "fixed" at test time, when they are actually passed in as input to the first layer of the model, as the forward pass executes, at the spec token's position a column of activations is created in the model (KV states) that are _not fixed_ at all and rather are dynamic data dependent representations of previous tokens in the context, so, calling this type of operation the used of "fixed" tokens is odd. 

### 4. 

More generally, S3.2 is quite hard to understand, though I see that it is perhaps the most complex part of the method so maybe this is expected. The attention mask provided in Fig 2 is helpful I think, but it's not clearly connected to Fig 3. In the iterative parallelized latent rollout diagram in S3.2, are the sequences X'[1] and X'[2] in Fig 3 intended to correspond to x0,x1,x2, and x4,x5 in Fig 2? I noted the phrase "relax the causal dependencies across reasoning blocks" and that the cost is meant to be iteration depth l by the num masks per block m but really I struggled to get anything out of Figure 3. What does the t index refer to? positions appear to be indexed as columns in Fig. 3 so i can't tell how "t-th token generation in the l-th iteration" should be interpreted in context.

I dont think this is the case, but to clarify, the accompanying attention mask for iterations l=1, l=2 etc. eg what gets to attend to what while this latent iteration is being performed is the same at every iteration right? I suspect that this mask is constant as a function of l, but varies for each complete training step based on where the block boundaries, text tokens, and latent tokens happen to fall. Does relaxing the causal dependency mean that in the missing attention mask diagram corresponding to Figure 3, would z2,1 and z2,2 not be able to attend to z1,1 and z1,2 but would be able to attend to X'[1]~=x1,x2,x3 ?

### 5. 

Ablation on m and block is not clearly described as the terms sentence level and paragraph level are not made precise nor interpretable. It would be much more clear to represent the choice as the num tokens per block and a ratio of latent tokens to text tokens, then average block size, and latents per block can be reported for each training setting and testing benchmark. The main reason for this request is that for wikipedia or news like content, perhaps, assuming a standard tokenizer, one can intuit the ratios, but for the main experimental settings like AIME and MATH, the notion of sentences and paragraphs aren't as well defined and might be differently distributed.

### Questions
### 1. 

Does the final model generate reasoning tokens automatically? Or at inference time, based on the block size and latent toks per block params, are the special latent tokens inserted using logic external to the LLM forward itself? It is also unclear whether the training iteration depth l is relevant during test time generation.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes HybridCoT, a method that allows a language model to interleave textual and latent CoT in the reasoning trace for math problems. For each block in the reasoning trace, LLMs first generate a text segment and then generate a fixed number of latent vectors. Non-critical texts in the previous block will be removed to increase the inference efficiency. The authors also propose a parallelized latent rollout algorithm to vastly reduce the training cost compared to previous work. The performance on several benchmarks shows that HybridCoT achieves similar accuracy to textal CoT but uses fewer compute, achieving a balance between accuracy and efficiency.

### Strengths
1. The idea of interleaving textual and latent CoT is novel.
2. The proposed training method is efficient compared to previous work, such as coconut.
3. It strikes a balance between accuracy and efficiency, as demonstrated by experimental results on various benchmarks.

### Weaknesses
1. The performance of hybridCoT is slightly worse than Text CoT. So the main advantage of the proposed method should be inference efficiency. The authors measured it using inference compute. I wonder how the compute is calculated. Can the 2x faster compute be translated to, e.g., lower inference latency or using fewer GPUs to support inference? This is currently unclear to me.

2. (minor) In line 127, when discussing the benefit of latent CoT, “It needs fewer decoding steps …”, it might be worth discussing the previous work [1], which theoretically demonstrates the benefit.

**References**:

[1] Zhu, Hanlin, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, and Yuandong Tian. "Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought." arXiv preprint arXiv:2505.12514 (2025).

### Questions
1. How is the inference compute measured/calculated? Is it proportional to the inference latency? How is the inference latency of hybridCoT compared to SFT?
2. Why does the 7B model consume more compute than the 8B model? Is it because the output is longer (e.g., more verbose or more inefficient) for the 7B model?
3. In Figure 4, for the orange (8B model) curve, why does the accuracy of Text CoT look lower than hybridCoT?

### Soundness
3

### Presentation
3

### Contribution
2
