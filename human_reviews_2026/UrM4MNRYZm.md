# PonderLM: Pretraining Language Models to Ponder in Continuous Space

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Humans ponder before articulating complex sentence elements, enabling deeper cognitive processing through focused effort.
In this work, we introduce this pondering process into language models by repeatedly invoking the forward process within a single token generation step. During pondering, instead of generating an actual token sampled from the prediction distribution, the model ponders by yielding a weighted sum of all token embeddings according to the predicted token distribution. The generated embedding is then fed back as input for another forward pass. We show that the model can learn to ponder in this way through self-supervised learning, without any human annotations.
Experiments across three widely used open-source architectures—GPT-2, Pythia, and LLaMA—and extensive downstream task evaluations demonstrate the effectiveness and generality of our method. On 9 downstream benchmarks, our pondering-enhanced Pythia models significantly outperform the official Pythia models. Notably, our PonderPythia models demonstrate remarkable effectiveness: PonderPythia-2.8B surpasses Pythia-6.9B and rivals Pythia-12B, while our PonderPythia-1B matches TinyLlama-1.1B, a model trained on 10 times more data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a novel technique (called "pondering") to iteratively call the forward pass of an LLM for a single token generation. Typically, a forward pass with an LLM over an input sequence results in a probability distribution over the vocabulary, which is then used for decoding the output token. The pondering approach in this paper utilizes the probability distribution at each token position to compute a weighted sum of the vocab embeddings. The main idea is to leverage such refined embeddings in subsequent forward passes ("s" times), followed by the calculation of the CE loss. The results show that the pondering approach improves the FLOPs and data usage efficiency during pre-training of various model architectures and sizes.

### Strengths
1. The methodology is well presented along with clear writing, code, and results.

2. The language modeling and downstream task evaluation results are promising and can influence research on such iterative embedding refinements as a new frontier to explore the scaling of compute.

### Weaknesses
1. A major concern I have is about the validity of the scaling law shown in Figure 4 if one were to change the number of pondering steps from 3 to a lower or higher value. Or even if the randomized pondering step approach was employed for that matter. If the scaling laws were similar to figure 4 with a randomized approach of pondering step selection, then that is an even better approach to scale since it is currently unclear how the value of (s=3 steps) was chosen. I believe the message of the paper can be further strengthened if this weakness is addressed.

2. This is a minor concern, but can a model trained to ponder for (s=3) steps be effective when pondering for (s>3) steps during inference? Some early results (either positive/negative) can influence a lot of future work in this direction.

### Questions
1. Did the authors analyze any properties of the embeddings after every pondering step? For ex: spectral properties, cosine similarities etc

2. How do the gradient norms change when incorporating pondering vs the vanilla pretraining? 

3. Were there any training stability challenges when training from scratch, since I see some spikes in Figure 8 for continued pre-training? How did you address them?

4. An analysis of how the output distributions evolve with iterative pondering steps would be a great addition to the paper. For ex, does the KL divergence of the output distributions change drastically in early pondering steps or later?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces PonderLM, a self-supervised, architecture-agnostic mechanism that lets a language model “ponder” within a single token step by feeding back a probability-weighted sum of token embeddings for additional forward passes, creating a continuous, fully differentiable inner loop that increases compute per parameter without RL or curated CoT data. Implemented across GPT-2, Pythia, and LLaMA, the approach yields lower pretraining perplexity and strong downstream gains—e.g., PonderPythia-2.8B surpasses Pythia-6.9B and rivals Pythia-12B, while PonderPythia-1B matches TinyLlama-1.1B trained on 10× more data—with performance improving as pondering steps increase. 

The key contributions are: (i) a simple pondering loop that replaces discrete token emission with continuous weighted embeddings; (ii) proof that such behavior emerges via standard next-token pretraining alone; (iii) consistent benefits across model families and scales, especially for smaller models; and (iv) a framing of pondering as a third, orthogonal scaling axis (complementary to parameter and CoT test-time scaling) that may improve parameter knowledge density and reduce communication costs at scale.

### Strengths
1. The paper introduces a differentiable, self-supervised pondering loop that feeds a probability-weighted embedding back into the model within a single token generation step. This elegant mechanism eliminates the discrete bottleneck imposed by vocabulary spaces during internal computation. By conceptualizing pondering as a third scaling axis, orthogonal to both parameter scaling and test-time CoT scaling, the work offers a novel perspective on model scaling dynamics. Moreover, demonstrating that this behavior can emerge without reinforcement learning or curated CoT supervision substantially relaxes the conventional prerequisites of test-time scaling approaches.

2. The proposed inner loop is straightforward, easily integrable into standard language model architectures, and fully differentiable for end-to-end training using conventional next-token prediction. Empirical evaluations across three architectures (GPT-2, Pythia, and LLaMA) and nine downstream tasks reveal consistent and substantial performance gains—most notably, size-for-size improvements where a 2.8B model outperforms a 6.9B counterpart and approaches the performance of a 12B model. Moreover, the results exhibit monotonic improvements as the number of pondering steps increases.

### Weaknesses
1. The paper’s motivation and theoretical foundation appear insufficiently developed. It remains unclear why repeating the forward pass within a single token-generation step should improve performance. The current justification—an analogy to human “slow thinking”—is conceptually interesting but lacks a mechanistic explanation or connection to established findings in neural or cognitive science. Providing a clearer rationale, ideally supported by formal analysis of how the proposed weighted-embedding feedback influences model expressivity, optimization dynamics, or the compute–performance trade-off, would considerably strengthen both the motivation and the overall argument.

2. To substantiate the claim of a latent thinking process, it would be valuable to analyze the model’s internal states and compare them with those from explicit (e.g., CoT) and implicit thinking methods.

3. Several evaluated tasks appear susceptible to data contamination [1]. It would strengthen the paper to quantify the extent of contamination and disentangle its contribution to the observed gains—for example, by applying contamination checks, re-running on decontaminated splits, or reporting performance deltas with/without potentially contaminated items.


References

1. Koala: An Index for Quantifying Overlaps with Pre-training Corpora. Vu et al. 2023

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper takes the standard output token probabilities and feeds them back into the model a set number of time with gradients attached. The intuition is that this allows the model to "think" more before outputting a token.

### Strengths
Strong consistent results, good that it only needs general corpus data, comprehensive set of experiments.

### Weaknesses
**W1.** Limited novelty - a very simple change and very similar to prior methods. But perhaps this is not a weakness as the results seem good.

**W2.** 4x compute at inference time. With LLMs actually being used now, inference cost is important. I think they should perhaps therefore be compared to 4x larger models which will have the same inference cost. In this case the performance is less strong.

### Questions
**Q1.** What is meant by: “potentially reducing communication costs at scale”?

**Q2.** It is unusual for results to be quite this consistent. Are the authors sure there were no other changes that contributed to this?

### Soundness
3

### Presentation
3

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
This paper proposes a pondering process to replace the standard single forward pass in transformers. Instead of producing a predictive distribution in one pass, the model refines it iteratively. At each iteration, the model computes the pondering embeddings by using the current predicted distribution to take a weighted sum over all token embeddings. This pondering embedding is then added back to the original token embeddings through a residual connection, and the updated embeddings are fed back into the model to produce a refined output distribution. 

The pondering method shows strong empirical performance across extensive experiments. The pondering-trained Pythia models reach the same performance with significantly fewer parameters and training data, and they consistently outperform their counterparts on 9 downstream benchmarks. Ablation studies on alternative embedding strategies and the number of pondering steps further demonstrate the effectiveness of the method.

### Strengths
The idea is simple yet effective, and it does not rely on any external supervision. The proposed mechanism is easy to implement and can be plugged into standard Transformer architectures with minimal changes.

The empirical results are solid and sufficiently demonstrate the effectiveness of the proposed approach. The observation that performance improves monotonically with more pondering steps suggests that the method provides a controllable way to trade compute for performance.

### Weaknesses
Since the method introduces additional iterative passes beyond the standard forward pass, it incurs non-trivial training and inference cost, which may become particularly expensive for larger models and longer sequences.

### Questions
1. In Eq. (5) you add the pondering embedding via a residual connection. What happens if this residual connection is removed, i.e., if you simply set $E^1=T$. Did you run this ablation? It would help isolate the contribution of the residual pathway itself.

2. Have you examined how the predicted distribution evolves across pondering steps? It would be interesting to see whether each step moves the prediction closer to the target distribution, as this could provide additional interpretability into how the refinement process actually works.

### Soundness
3

### Presentation
3

### Contribution
3
