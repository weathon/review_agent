# Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation

- Decision: Reject
- Scores: 5, 6, 3, 6

## Abstract
Inference-time computation is a powerful paradigm to enhance the performance of large language models (LLMs), with Best-of-N sampling being a widely used technique. However, this method is computationally expensive, requiring both (1) an external reward model and (2) the generation of multiple samples. In this work, we introduce a new generative self-evaluation scheme designed to adaptively reduce the number of generated samples while maintaining or even improving performance. We use a generative reward model formulation, allowing the LLM to predict mid-generation the probability that restarting the generation will yield a better response. These predictions are obtained without an external reward model and can be used to decide whether or not to generate more samples, prune unpromising samples early on, or to pick the best sample. This capability is very inexpensive as it involves generating a single predefined token. Trained using a dataset constructed with real unfiltered LMSYS user prompts, Llama 3.1 8B's win rate against GPT-4 on AlpacaEval increases from 21\% to 34\% with 16 samples and math performance on GSM8K improves from 84\% to 91\%. By sampling only when the LLM determines that it is beneficial to do so and adaptively adjusting temperature annealing, we demonstrate that 74\% of the improvement from using 16 samples can be achieved with only 1.2 samples on average. We further demonstrate that 50–75\% of samples can be pruned early in generation with minimal degradation in performance. Overall, our methods enable more efficient and scalable compute utilization during inference for LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes a new strategy to improve the inference of LLMs, involving two key components: adaptive sampling (selecting the N in best-of-N as a function of the prompt) and early pruning (pruning unpromising mid-generations). This relies on a new ability from LLMs; given a prompt and a potentially unfinished generation, they can learn to detect whether they will be better with an additional sampling round. This ability is trained explicitly in supervised way, with feedbacks obtained from an external RM; the key point being that this RM is then not required during deployment. This strategy improves performances on math and gsm8k benchmarks.

### Strengths
* The paper is clearly written, properly highlighting the different ideas and results.
* Improving inference strategies for LLms is an important topic for the ICLR community, with potential significant impact.
* The two contributions, the discovered "trainable self-evaluations" and its applications (adaptive sampling and early pruning) make sens.
* The experiments, althought succinct (see weakness), show convincing performances on standard benchmarks.

### Weaknesses
* The main limitation of the work is that the proposed ideas are arguably small improvements over existing baselines. For example, the paper does not mention "process-based reward models" or the "reward-guided decoding" literature (such as MCTS), that seeks the same objective. The only benefit of the paper would be to include those abilities inside the LLM itself, thus removing the need for an external value network. 

* In terms of practicality, the need to generate sequentially is a key limitation in real-time deployment, thus hindering some of the efficiency benefits. Though, I acknowledge that the idea of "exponentially increased batch size" is an interesting but limited answer.

* With your training, and given a prompt and generation, you actually train the model to estimate the quantile of this generation according to the external reward model; a discussion on this topic would be needed. Then similarly, it would be interesting to consider extension where, be given N>2 generations, your model learns to detect whether the generation is actually the best-of-N.

* Some ablations are missing. For example, could you provide the performances of applying temperatre annealing to basic Best-of-N? What if you train an external RM to evaluate the quantile on mid-generations? Or about a simple strategy that would detect, be given the prompt, its complexity, and then allocate automatically a prompt-aware compute.

* I fully agree that "We need to keep the underlying policy or model unchanged when learning to self-evaluate." (l.232) Though I disagree with the next sentence: "To do so, we need the model’s prior distribution over tokens to be similar to what we expect after training" (l.233). Could you clarify this? 
From my understanding, finetuning for self evalaution should decrease other abilities because of catastrophic forgetting.

## Nit

* The fig1 actually shows the opposite of what the legend "Rewards are not interpretable" states; indeed, actually the reward provides the lowest score to the most unconfident generation.

* Related work: some papers that may be worth discussing
"Don't throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding"
"ARGS: Alignment as Reward-Guided Search"
"Efficient Controlled Language Generation with Low-Rank Autoregressive Reward Models"
"Critic-Guided Decoding for Controlled Text Generation"

* The discussion on ties ends up being unecessary, as it does not impact performance. Thus, you might not want to state "Accounting for ties in reward modeling is especially important" (l.243).

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper shows that one of the challenging problem at inference-time computation scaling (Best-of-N) is to identify if a specific generation is "good", or if it is worth allocating more compute to continue its generation. For example, Best-of-N is considered computationally expensive because it requires external reward model to evaluate the quality of the generation, and it often requires multiple such generation samples to obtain a good quality result. 

To address this problem, the paper introduces capability-aware self-evaluation, a technique that allows LLM to self-predict if the current generation is promising, or in other words, if restarting the generation will yield better result or not. In the Best-of-N setting, the paper also introduces adaptive sampling and annealing to further improve the quality of the generation, and early prune unpromising samples using the self-evaluation methodology.

In the evaluation section, the paper presents a finetuned Llama3.1-8B-Instruct model trained on 30k preference pairs constructed from unfiltered LMSYS using a reward model, and shows an increasing win rate against GPT-4.

### Strengths
**Elegant solution**. The evaluation section shows that the proposed fine-tuning method is able to generalize from LMSys to AlpacaEval and GSM8K. The solution is elegant and intuitive, and it works well.

The method shows a promising way to improve the quality of generation in Best-of-N setting by early pruning unpromising samples, and adaptively sampling and annealing the remaining samples. The proposed method is simple, intuitive, and shown to be effective.

**Writing**. The writing is clear and the paper is well-organized. The paper is well-written and easy to follow.

### Weaknesses
**Finetuning dataset construction - what is a good dataset to finetune on?**. Why is LMSYS dataset chosen? Would another dataset other than LMSYS do better or worse? Would using a more complex dataset do better or worse? It is not as convincing if there is no ablation study on the dataset choice.

**Limited evaluation**. Only 2 datasets are used to perform the evaluation. GSM8K is not a hard dataset to see performance gain (e.g. using hyperparam tuning). To show the generalizability on reasoning tasks, it would be better to evaluate on more diverse and complex datasets, especially on slightly longer reasoning chains to show the increasing size of sampling and prunning is effective. Showing the distribution of generated token lengths would be very useful to understand "how far" are we talking about.

**Limited setting of inference time algorithm**. As mentioned in conclusion / limitation section, the paper only discuss Best-of-N setting. But many reasoning techniques such as self-refinement / chain-of-thought are not generating parallel requests. It would be better to evaluate on more diverse inference settings.

**Naively mapping number of samples / tokens to computational cost**. The paper did not include any statement about computational cost in terms of number of tokens generated. In actual LLM serving with batch inference (e.g. vLLM, DeepSpeed), the computational cost of 16 samples performing decoding is roughly the same as 1 sample performing decoding. Thus, if the metric is "end-to-end latency", then the bottleneck is actually the straggler - meaning what is the longest request in the batch that needs to be finished before the generation is complete. The proposed method does not seem to address this problem. In addition, as mentioned in discussion section, introducing a prefill (inserting prompt into the middle of generation) is not free, and may actually introduce more latency than naive batch decoding setting. Although this is not the focus of the paper, it is important to properly address this point in writing as it is not very clear at reading.

### Questions
1. Does the selection of finetuning datasets matters, or to what extent it affect the generalization of the model? 
2. What is the distribution of the generation length? 
3. Would the proposed method still be effective when problem is more difficult or unseen?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
A method for predicting whether the answer to a query being generated by a LLM is worth continuing or not.
The method fine-tunes a pre-trained LLM to predict whether the current (partially) generated answer is worth   continuing or not. The fine-tuning involves pre-pending the prompt 'Would you do better if you started over?   (“Yes.” or “No.”)' to the answer of a query and SFT the model to predict "Yes" if the answer is prefered over  the alternative answer.
By deciding to abort the answer generation, the authors claim that the LLM may save compute at inference-time  compared to a method that do not abort the answer generation.

### Strengths
The paper tackles an important challenge and chose to do so using an open-weight model. Advances on open-      weight models may benefit everyone and reducing the compute and inference time may increase the adoption of    foundation models. The method itself was understandable (althought I believe that the writing of the paper     could generally be made clearer and more succinct).

### Weaknesses
The paper proposes a method that improves inference-time compute, yet it never compares the compute-time and   the FLOPs with baseline methods. Having to probe the model mid generation definitely incurs a cost, which is   not being discussed in the paper. Moreover, the model finds that pruning longer sentence improves performance, in which case perhaps it would be better to fully generate the sentences before selecting them and thus the    idea of pruning mid-generation is unclear.

In Figure 1, the x-axis stops at 16 Fully Generated Samples, where the method of the authors has hit a plateau and the Best-of-N method is still showing improvement. The authors should increase the amount of fully         generated samples to observe and compare the effect of more generated sample of their method and the Best-of-N baseline.

In Table 1 the author compares baseline methods with the proposed method. However, it is not at all clear what are the parameters of the capability-aware method and whether it is comparable to the baselines. The authors   have to clarify how they compare with the baseline methods. E.g. of questions I have (and this is a subset of  the questions I have -- the authors should make sure that the revised version clearly explain the experimental methodology): How many fully generated samples are used? (Line 404 the authors mention Best-of-16 sampling, is this 16 fully generated samples?) If this is indeed 16 fully generated samples, then why should a pratitioner  use that method instead of the best-of-N baseline which performs comparably well acccording to Figure 1.

The ablations that are important for understanding the parameters of the method are in the Appendix while they should be in the main paper.

The authors continually refer to Figure 1 throughout the paper, forcing the reader to continually scroll up    and down. Instead of packing many important observations within one figure, the authors are encouraged to      break down the observations into indivudal figure (i.e. one observation per figure).

Finally, I encourage the authors to revise the writing making it more succinct and avoid long winded sentences.

### Questions
* How does the proposed method compares with the baselines in term of compute-time?
* How does the proposed method compares iwth the baselines in term of FLOPs?
* How does Figure 1 looks like when evaluating 32 and 64 fully generated samples?
* What are the parameters of the methods and the baselines used to produce table 1?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
- The paper proposes to simple way to train a LLM to do self-evaluation (with additional tokens) and uses the logits for the trained additional tokens to decide whether to continue sampling or not. They show that they can approximately match the performance of Best-of-16 at, on average, 1/4 of the number of samples on GSM8K.
- The authors claim on the efficiency and latency of the proposed method, but since it requires serial, instead of parallel, generations when doing Best-of-N, the latency may actually increase by a non-negligible amount. This needs to be properly addressed in the paper as this is a paper about making Best-of-N efficient.

### Strengths
- The authors propose a very simple and concrete way to adaptively scale generations for Best-of-N and show an noticeable improvement on GSM8K where they can approximately match the performance of Best-of-16 at 1/4 of the number of samples.

### Weaknesses
- The performance gain is marginal for Alpaca Eval. Thus, more comprehensive evaluations on a broader range of benchmarks that perhaps require more complex reasoning will be beneficial. However, I fully acknowledge the limits of what can be done during the rebuttal.
- The paper/authors argue that their proposed adaptive inference-time compute is scalable and efficient based on the performance vs. number of samples comparison. They make an argument on latency. However, an important aspect that is underplayed is the actual computational efficiency when implemented. Although mentioned, because of the need to self-evaluate, the adaptive inference-time compute is allocated in serial. On the other hand, Best-of-N can be executed in parallel, which is its advantage over this method. Although inference speed is very dependent on the inference stack engine, it would be important to quantify this cost of serial processing.
- Some further analysis is warranted to analyze the effect of learning to self-evaluate alone. Interestingly, the authors note that a pretrained (and instruction tuned) model has a nontrivial ability to conduct self-evaluations and that a Bradley Terry reward model further improves it. It would be constructive to study the effect of self-evaluation training in isolation and essentially conduct Best-of-N after this self-evaluation training to separate the effect of this training on newly annotated data and the adaptive inference method.

### Questions
- For Section 4.2 experiments on AlpacaEval and GSM8K, what is the performance of Llama 3.1 8B Instruct when using Best-of-16?
- The authors focus on showing the efficiency in terms of the reduction of the number of samples vs performance. However, an interesting experiment would be to, fix the amount of compute budget (e.g., FLOPS), and to see whether one can adaptively allocate higher # of samples for more difficult questions and fewer for easier questions, while using everything. That would essentially show the effectiveness of this adaptive compute allocation method compared to a naive Best-of-N that allocates the same amount of compute to any prompt.
- How does the proportion of ties (the epsilon for deciding ties) in the preference dataset construction for self-evaluation training change the performance of the model?
- I am not sure how realistic this is given the time constraint of the rebuttal, but a natural question that arises is how this changes for datasets that require more complex reasoning such as ArenaHard, Math Hard, etc.

### Soundness
2

### Presentation
3

### Contribution
2
