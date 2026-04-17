# SuRe: Surprise-Driven Prioritised Replay for Continual LLM Learning

- Decision: Reject
- Scores: 4, 4, 2, 8

## Abstract
Continual learning, one's ability to adapt to a sequence of tasks without forgetting previously acquired knowledge, remains a major challenge in machine learning and a key gap between artificial and human intelligence. While regularisation and replay perform well in vision, they lag behind multi-task learning for large language models (LLMs), especially at scale with many tasks. We revisit replay and argue that two failure modes drive this gap: selection (what to rehearse) and integration (how to consolidate new knowledge). To address selection, we propose Surprise-prioritised Replay (SuRe), a simple, architecture-agnostic rule that ranks and stores the most surprising (high Negative Log-Likelihood) sequences. SuRe achieves state-of-the-art performance in the Large Number of Tasks (LNT) setting and delivers the best overall average across both Standard CL and LNT benchmarks. To address integration, we add a dual-learner design with fast and slow LoRA adapters merged via an exponential moving average (EMA), enabling rapid adaptation while stabilising long-term knowledge. Combining SuRe with the dual learner yields further gains, including improvements of up to +5 accuracy points on LNT over prior SOTA. Ablation studies confirm that our proposed method remains robust under reduced replay frequency and small buffer size, demonstrating both effectiveness and sample efficiency. Taken together, our results establish replay as a strong baseline for continual LLM fine-tuning and demonstrate that surprise-based selection and slow-weight consolidation are complementary components for mitigating catastrophic forgetting.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SURE: surprise-driven prioritised replay for continual LLM learning. The main ideas of the paper are a surprise-based replay strategy and LoRA with a slow-fast learning mechanism. The consolidated methods are evaluated by standard CL and large number of task benchmarks.  The paper shows its strengths in several aspects, but also has several weaknesses that need to be addressed. Please see strengths and weaknesses.

### Strengths
(1). The has clear motivation and ideation.

(2). The numerical result shows a promising performance on a large number of tasks benchmark.

(3). The paper presents a comprehensive theoretical analysis.

(4). The paper presents a rigorous numerical analysis.

### Weaknesses
(1). Methodology: The slow-fast learning mechanism is arguably not a novel idea, since it has been proposed in a few existing CL methods, e.g., DualNet, and Slow-fast Prompt. Second, while surprise replay offers an innovative idea to select the buffer, it lacks of theoretical foundation and the details on how it works.

(2) Learning mechanism: Figure 1 shows that slow and fast learners are updated on two different phases, but the pseudo-code shows that both learners are trained in the same steps.

(3). Performance: The proposed method achieves only a small (or negative) margin for some cases in the standard CL benchmark. It is questionable that the proposed method significantly outperforms the existing methods.

(4). Forgetting: I do not see the measurements and analysis of models' forgetting as the answer to the catastrophic forgetting problem.

(5). Theoretical Analysis: While it shows the boundary for the slow learner, Lemma 2 does not show the better handling of slow-fast learners in comparison to a single learner.

(6). Performance on budget memory: I appreciate the measurement of the proposed method's performance on different memory sizes. However, it is expected to be compared to the existing methods in those memory budgets. 

(7). Only one of the competitor methods is up-to-date. It should include more latest methods for comparison.  


References:

[1]. Dualnet: Continual learning, fast and slow

[2]. Brain-inspired fast-and slow-update prompt tuning for few-shot class-incremental learning. (Slow-fast prompt)

### Questions
Please address the weaknesses

### Soundness
2

### Presentation
4

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
The paper considers the problem of continual learning on text classification tasks. It focuses on buffer-based replay methods. It derives an upper-bound on the forgetting incurred by such methods, which includes two complimentary terms: 1) how well the samples from the buffer approximate the model's loss on past data 2) a term which captures how “well” new samples are consolidated. The paper then proposes to reduce the forgetting upper-bound by 1) storing past samples with high loss inside the buffer 2) combining slow and fast changing weights during training.
The paper outperforms other CL baselines on text classification tasks.

### Strengths
I identified the following strengths of this paper:

- Theoretically, it provides an upper-bound on the forgetting experienced by a buffer-based replay method. I think the way the quality of the buffer is defined (D_{F_loc}(P_{1:T}, q)) is novel and might be interesting for others. The selection term and consolidation term being complementary is an important contribution. This was later backed up by experimental results.

- Experimentally, it provides evidence that: 1) Replay-based with reservoir sampling can outperform regularization-based CL methods. 2) Surprised-based replay methods can outperform reservoir sampling methods.

- The limitations section is comprehensive.

### Weaknesses
Aside from the theoretical contribution, the rest of the methodology section has limited novelty - it appears to combine two already established ideas. Moreover, the text never makes it clear (from what I could see) how each component reduces the terms in the upper-bound.

Readability: I don’t think that the integration (consolidation) term in Eq. 3 is well explained. Reading the main text, I do not have a good idea of what $B(\psi)$ is, apart from it being a “mechanism-specific factor”.

The paper claims to be applicable to large language models, which at least to me suggests the task of language modelling, while it is evaluated on sequences of text classification tasks. Therefore, claims such as “our results establish replay as a strong baseline for LLM continual learning” seem too general to me, and unsubstantiated by the experiments.

The paper would benefit from a clear “contributions” statement in the introduction.

### Questions
How does your method of storing high-surprise samples in the buffer relate to importance sampling (instead of uniform sampling) of past data?

Your derivation and method seems to be general for any continual-learning setup - including both dataset and architecture. Is there a reason it would best perform on large language models? (Perhaps, relying on the local neighbourhood containing the optimal solution?)

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
4

### Summary
This paper addresses the continual learning for large language models (LLMs) with replay. The authors first define forgetting as the sum of a selection mismatch and the knowledge consolidation variance. To address this, the authors proposed a surprise-based sampling strategy to populate the replay buffer. Moreover, the authors proposed a dual-learner framework to deal with long-term and short-term learning. Experiments showed the improved performance of the proposed method.

### Strengths
1. The paper is well motivated

It is an interesting idea to select surprising samples for replay.

2. The method is straightforward

3. The decomposition of forgetting is interesting

### Weaknesses
1. The surprise measure might not be reliable

2. The idea of dual-learner is not novel, and the implementation seems confusing

3. The comparison is not sufficient and up-to-date

Please see details in the Question section.

### Questions
1. The surprise measure might not be reliable

According to Equation 9, the authors used the sum of native log-likelihood over the entire sequence. There might be several issues with this measure. (1) Taking the sum of the entire sequence might dilute the actual signal, since LLM might have a long answer, but what matters would just be a few words. Although the authors claimed that the full-sequence measure is better than the label-level measure in lines 363-369, there are only some hypotheses to explain this without factual evidence to support them. Moreover, could the authors elaborate on the average length of the generated sequence? (2) This surprise measure is not able to detect hallucination, as the model might just be confidently generating hallucinations. 

2. The idea of dual-learner is not novel, and the implementation seems confusing

First, the idea of slow and fast learners is not new. This idea has been explored in [a], and the EMA implementation of LoRA has been proposed in [b]. Second, the implementation is confusing to me. I don't see where the slow learner is being used in the learning process, nor in the Figure. 1 or Algorithm 1. I am wondering how this slow learner helps the model.

[a] Pham, Quang, Chenghao Liu, and Steven Hoi. "Dualnet: Continual learning, fast and slow." Advances in Neural Information Processing Systems 34 (2021): 16131-16144.

[b] Gao, Qiankun, et al. "A unified continual learning framework with general parameter-efficient tuning." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

3. The comparison is not sufficient and up-to-date

It is totally fine for the proposed method to focus on replay-based CL. However, the comparison in the experiment section should therefore prioritize replay-based methods. The current comparison contains too many replay-free methods, and only compares with one replay method [c] without citing this paper. I don't find this comparison fair and up to date to the recent advance of replay strategy in the CL community.

[c] Rolnick, David, et al. "Experience replay for continual learning." Advances in neural information processing systems 32 (2019).

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
The paper attributes catastrophic forgetting in continual LLM learning to replay selection and integration errors. It proposes to rectify these errors by selecting 'surprising' examples (characterized by high nll) to replay, and learning a 'slow' learner that is updated using EMA over the 'fast' learner (i.e., the one optimized directly over the incoming and buffered samples). The two learners are implemented using LoRA adapters. Empirical evaluations support the claims and provide appreciable improvements over baselines.

### Strengths
- The paper is well-written, claims are intuitive, and theoretically and empirically validated.
- To the best of my knowledge, the paper is the first to propose NLL for sample selection. Combining this with a slow learning strategy empirically shows significant improvements, as shown in Table 1.
- SuRE is implemented using LoRA and is therefore architecture agnostic.
- Evaluations and ablations are sufficient, and SuRE outperforms SOTA continual LLM learners on both benchmarks.

### Weaknesses
- To my knowledge, there are no significant weaknesses.

### Questions
- What is the value of $\beta$ used for evaluation? Can the authors include an ablation to show its impact?
- Can the reliance on high surprise cause the model to overfit to outliers (such as mislabeled samples) and inadvertently hurt the performance of the model? Perhaps buffering a combination of surprising samples and some randomly selected samples instead can help?

### Soundness
4

### Presentation
4

### Contribution
3
