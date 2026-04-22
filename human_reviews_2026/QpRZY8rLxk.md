# Thinking into the Future: Latent Lookahead Training for Language Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Autoregressive language models trained with next-token prediction generate text by sampling one discrete token at a time. This forces the model to commit early, preventing exploration of multiple plausible continuations. Furthermore, each token is predicted in a single forward pass, which might limit the model’s expressiveness in cases where difficult tokens require inherently more compute. Towards this end, we introduce latent lookahead, a training strategy that enables models to think before answering: at selected positions in the sequence, before committing to the next token, the model performs a multi-step lookahead in latent space. Instead of sampling future tokens, we leverage the network’s latent space by recursively feeding its hidden states back into the context for τ steps, investing more compute on predicting that token. This produces τ latent predictions that are supervised against the next τ ground-truth tokens, encouraging the model to “look
ahead”. We show that latent lookahead substantially outperforms autoregressive baselines on planning tasks such as maze solving, Sudoku, and ProsQA, where foresight is essential. Finally, we demonstrate how to endow pretrained models with this ability during supervised fine-tuning and evaluate the resulting models on standard reasoning benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Latent Lookahead: at selected positions, the model rolls out τ steps in latent space by feeding hidden states back into the context before emitting the next visible token, and explicitly supervises each latent step against the next τ ground-truth tokens. The proposed approach is pretty similar or a combination of existing and well studied works like MTP, COCONUT and looped transformers.

### Strengths
- The empirical gains are clear over the standard NTP in the toy settings. However, again, most parts of the proposed approach are well studied.
- I like the proposal of parallel rollouts to make the approach more feasible. 
- The writeup is pretty clear to understand.

### Weaknesses
- The paper proposes looping through the LLM stack to predict \tau future tokens as an auxiliary objective during training. However, this is conceptually very similar to multi-token prediction (MTP), where the model also predicts multiple future tokens. The main distinction here is that instead of a lightweight auxiliary head (as in the original MTP formulation), the authors loop through the entire transformer stack. Notably, the DeepSeek variant of MTP (https://arxiv.org/pdf/2412.19437v1) already explores an intermediate design between standard MTP and this work.


- In effect, the proposed approach can be viewed as a combination of MTP and looped transformers (https://arxiv.org/abs/2502.17416). It also bears close resemblance to COCONUT (https://arxiv.org/abs/2412.06769), which similarly performs looping in the soft-token space using intermediate supervision.


- While combining existing ideas is perfectly valid, this work does not appear to offer significant new conceptual insights or demonstrate scaling to new or large-scale setups. MTP, looped transformers, and COCONUT have all been extensively evaluated in real-world, large-model scenarios — this paper’s contribution remains mostly incremental relative to those lines of work.


- The paper critically lacks comparisons or even substantive discussion with these prior works. The brief ablation discussion at the end of Section 3.1 is inadequate. Given its close relevance, MTP should serve as a central baseline in nearly all experiments, especially since MTP has been shown to improve representation learning itself.


- The differences from MTP that are briefly mentioned in related work are insufficiently articulated and buried late in the paper. These distinctions should be clarified more prominently — ideally in the introduction — given the strong overlap in motivation and mechanism.

### Questions
Please check the weakness section. I will be willing to reconsider my evaluation if the authors point out a significant flaw in my understanding or highlight that I missed something. I otherwise believe the work and the proposed approach needs a significant number of ablations to tease out the differences from existing work, if at all there is any.

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes a latent reasoning approach that “looks ahead” a few tokens before committing to a specific token decision. This paper adopts a Coconut-style (Hao et al. 2024) approach of recurring over output latents by feeding them back into the network. A primary contribution of this paper is introducing a training procedure to make the latent recurrence more tractable compared to prior work. They report strong results on synthetic planning tasks (e.g. maze solving, Sudoku) and mixed results on real-world datasets.

### Strengths
This work focuses on an interesting research direction that is gaining interest recently. Namely, whether we can train models to reason directly in some latent space and obtain some benefit from that.

A big limitation of Coconut from prior work was the computational slowdown of recurrence. While this is unavoidable in some sense, this work introduces a clever training procedure to enable training with multiple latent blocks in a sequence without having to recur for every single latent token. The latent recurrences are essentially done in parallel and it simulates autoregressive generation with careful construction of the attention mask. 

They adopt a reasonable selection of synthetic planning tasks to demonstrate the effectiveness of their approach. The improvements on all planning tasks are large and seem to scale well with the use of additional latent tokens. The pause token baseline is a nice inclusion.

The ablations presented in Figure 5 are nice and demonstrate that NAR refinement is beneficial and that the latent lookahead outperforms pure multi-token prediction.

The visualization of the latent thoughts (Fig 6) also provides intuition about the learned behavior of the latent thoughts.

### Weaknesses
All of the synthetic experiments are done with extremely shallow (2-layer) transformers. The latent lookahead essentially increases the depth of the transformer. I do wonder if the benefit extends to deeper transformers than those studies. 

The experiments for SFT settings are quite mixed. No setting seems to consistently outperform SFT. When discussing the results in the paper, they take the best of two latent lookahead models (\tau=4,6). Neither one outperforms SFT consistently. As a result, I don’t think the following claim is supported by the results:
“latent lookahead yields modest but consistent improvements over autoregressive baselines“
Given the mixed SFT results, the positive results are largely constrained to synthetic tasks designed such that lookahead is useful.

It is not clear that supervising the latent tokens with the normal token labels is the obvious choice. This decision should be ablated. Is this critical for latent lookahead to work?

Although the asymptotics of the latent lookahead training are discussed, the discussion of the computational overhead during training compared to random training is not discussed. It seems, even with the efficiency improvements, that the overhead would be quite high compared to regular training, both due to the recurrence and the sequence length extension.

### Questions
Do the same benefits of Latent Lookahead extend to a deeper (6 layer, 12 layer, etc.) transformer? 

How does the computation cost of latent lookahead compare to standard training? For the synthetic and SFT experiments?

How is training implemented? Is the entire sequence fed through for each iteration? It seems like there is significant room for training optimizations with the structured attention mask.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes training language models to do latent lookaheads where the final layers hidden state is not used to predict a token to output but instead fed back into the model as the next input token, similar to Hao et al. (2024). However, this work repeatedly passes the hidden state through the model letting the model refine multiple hidden states at a time before returning to autoregressive generation. The hidden states are supervised with the future ground truth tokens at their relative positions to the previous output token.

### Strengths
* Strong performance on synthetic planning benchmarks i.e. puzzles. 
* Really interesting idea, building on coconut.
* Great diagrams that aid the writing in explaining their method.

### Weaknesses
* Their method struggles to improve performance of OLMO in real settings GSM8K, AQuA, BBH. It is unclear if this is because the method doesn't help or SFT-ing an exiting model to do latent lookahead is hard. Again their method struggles to improve Qwen via SFT. They see extremely small improvements in GSM8K and no improvement in Sudoku.
* The authors in reference to the SFT results write the following on line 396, "Overall, these findings indicate that the latent lookahead mechanism transfers to SFT models, with the largest benefits appearing on tasks that demand multi-step reasoning and planning, and incremental gains on math word problems". This statement is contradicted by the results they present in the immediately preceding paragraphs. This text needs to be revised to more accurately reflect your findings. Not everything has to work, it's better to acknowledge limitations than ignore them.
* The way latent lookaheads are supervised during training with the future tokens is surprising if not odd. Some ablation of the supervision mechanism would strengthen this work. 
* While the restriction of the attention mask to prevent latent lookaheads from attending to each other is new. The mask is very clearly built upon causal and bidirectional attention. The writing is careful not to say bidirectional attention at any point, but it really should be acknowledged that within a latent lookahead block is bidirectional attention (it is weird to omit this detail - you don't want people to assume you are the first to do bidirectional attention).
* The authors claim interpretability on line 087 but do nothing to demonstrate it. The method from what I can tell is not anymore interpretable than other methods used for reasoning.
* No discussion of how to set tau and n.
* No comparison to Chain-of-Thought or Coconut.
* The interpretation of latent tokens in section 3.3 never explicitly says this is for the 4x4 sudoku puzzles, though this is what I assumed. It should be made explicit what the setting is.
* On line 199 there is an incomplete sentence "... tokens to the We perform an ..."

### Questions
None.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper cleverly mixes causal and non-causal language modeling in order to place more reasoning into a next token prediction by in a sense reasoning from the future. The non-causal activations are trained to match corresponding next positions in a gold standard chain of thought, so these activations gain an interpretation as corresponding to future tokens without requiring intermediate sampling. But since the LM is non-causal over these token positions, over the course of a transformer forward pass, the guesses at future tokens will use the extra computation to refine and make the next token consistent with the (predicted) future. The technique shows strong results on synthetic tasks, but not substantially improving results on GSM8K. Figure 4 hints tantalizingly at a linear improvement in prediction accuracy as a function of the number of latent lookahead tokens, though this must not hold indefinetly or we would see it in GSM8K.

### Strengths
* **Clever tool:** Hidden “think-then-speak” compute with dense j-step supervision; potentially useful for planning-style problems.
* **Compelling Sudoku curve:** Figure 4 shows performance improving as tau grows; if that behavior extended to long horizons on math, it would be very convincing.

### Weaknesses
* **Task gap:** Large delta between Sudoku (works well from scratch) and GSM8K (minimal SFT gains). Techniques that train **visible CoT** for GSM8K (e.g., RL) also work on Sudoku and don’t require training from scratch.
* **Horizon mismatch:** Likely the lookahead isn’t long enough to cover full GSM8K reasoning traces. Evaluating tau ~ 300 would be informative, but then one pass must reconcile many constraints—essentially a **100%-masked bidirectional** modeling problem, much harder than BERT’s ~15% masking.
* **Possible remedy:** Consider **recursion/iteration** within the latent block (looped/DEQ-style refinement or diffusion-style denoising) so long horizons don’t have to be solved in a single pass.
* **Presentation clarity:** Several places imply the latent lookahead is **causal**. Figures 1 and 3 use arrows that read as autoregressive; the abstract and early text say things like “recursively/iteratively feeding hidden states back … for ( \tau ) steps,” reinforcing that impression.
* **Editing issue:** The Inference paragraph on p.4 appears mid-edit (missing words), which undermines clarity about complexity and masking.

### Questions
# Questions

1. **Scaling:** Does this technique scale to **hundreds** of latent thinking tokens? If the trend in Figure 4 continued to tau ~ 300 for some dataset, I’d raise to a **6**.
2. **Clarity:** Can you make it explicit (in figures and text) that the **latent lookahead is non-causal** within the block while visibles remain causal? If so, I’d move to **6**; with both clarity fixes and longer-horizon/iterative results, up to **8**.

### Soundness
3

### Presentation
2

### Contribution
3
