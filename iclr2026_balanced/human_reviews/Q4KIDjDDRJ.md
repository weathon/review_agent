## Human Reviewer 1

### Summary
- This paper investigates the challenge of applying RVLR to Multimodal LLMs, identifying an interdependence between perception-related and reasoning-related tokens.
- Token-Reweighting dynamically reweights the policy gradient calculation to jointly optimize both critical perception and reasoning tokens during RLVR training, which are identified using entropy and visual ablation.
- The strategy is applied to existing RLVR algorithms like GRPO and DAPO to achieve competitive performance on several multimodal reasoning and perception benchmarks.

### Strengths
- This work presents an interesting token-level analysis that classifies MLLM outputs into perception-related and reasoning-related tokens, empirically demonstrating their influence in RLVR optimization.
- The proposed ToR strategy achieves meaningful performance improvements over baseline GRPO and DAPO methods, establishing new state-of-the-art results on several multimodal reasoning and perception benchmarks in a data-efficient manner.

### Weaknesses
- My main concern lies in the logical foundation of the motivational study. The paper claims that optimizing only reasoning-tokens or only perception-tokens underperforms and that this proves their "interdependence." This conclusion appears to be a logical leap. The experiments merely show that partially disabling the model (i.e., zeroing out gradients for certain tokens) leads to performance degradation, which is an intuitive outcome. These quantitative results do not rigorously demonstrate why or how these two token types are interdependent; they only confirm that the full token set is better than an artificially constrained subset.
- The paper's claims of significance and generalizability are limited, as all experiments are conducted on a single model backbone (Qwen2.5-VL-7B). This weakness is amplified by a direct comparison in Table 1; the performance of the proposed ToR strategy is almost identical to or on par with the NoisyRollout method, which uses the same backbone model and 2.1K training dataset from Geometry3K. This suggests the contribution may be more incremental than a significant advancement over prior art. Similarly, the introduction of several hyperparameters seems to necessitate careful tuning, which is discussed in the ablation studies. This would further imply that adjusting these hyperparameters and understanding their dynamics for different configurations (e.g., other models or training data) could require an independent analysis and might result in varying performance across different settings.

### Questions
As a minor issue, the references rely almost entirely on arXiv preprints. Consider supplementing the related work for a more complete analysis.

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
2

### Confidence
3

---

## Human Reviewer 2

### Summary
The authors propose Token-Reweighting (ToR), a lightweight and plug-and-play module for Reinforcement Learning with Verifiable Rewards (RLVR), which substantially enhances the reasoning capability of large language models (LLMs) in complex tasks.  
They introduce two strategies to improve existing RLVR algorithms, GRPO and DAPO:  
(i) identifying reasoning-related tokens through high next-token entropy, and  
(ii) identifying perception-related tokens through visual sensitivity.  
Experimental results demonstrate that incorporating either component alone leads to suboptimal performance compared with the original algorithms, whereas jointly optimizing both yields superior results. The parameter sensitivity of the proposed module is analyzed, and the modified algorithm applied to Qwen2.5-7B achieves better performance than the base algorithm as well as other open-source models of comparable scale.

### Strengths
1. The idea of distinguishing reasoning-related and perception-related tokens is novel and well-motivated, as the varying importance of different tokens for reasoning and perception objectively exists.
2. The formulation for reasoning-related tokens is well-grounded, and the notion itself is consistent with the modeling objective (In contrast, the other concept lacks such alignment, as will be discussed in the weakness section).
3. In experiment, the parameter sensitivity of the proposed module is analyzed and the effectiveness of proposed module compared with GRPO, DAPO on Qwen-2.5-7B model is presented.

### Weaknesses
1. The design of the identified measure for perception-related tokens is not theoretically sound. The visual sensitivity score can be rewritten as $S _{i,t} ^b=|\log \frac{\pi _{\theta}(o _{i,t} ^b|\mathbf{o} _{i,<t} ^b,I _{i} ^b,q _{i} ^b)}{\pi _{\theta}(o _{i,t} ^b|\mathbf{o} _{i,<t} ^b,\emptyset,q _{i} ^b)}|$, which indicates that the measure directly depends on the log-ratio between $\pi _{\theta}(o _{i,t} ^b|\mathbf{o} _{i,<t} ^b,\emptyset,q _{i} ^b)$ and $\pi _{\theta}(o _{i,t} ^b|\mathbf{o} _{i,<t} ^b,\emptyset,q _{i} ^b)$. However, consider a likely situation where $\pi _{\theta}(o _{i,t} ^b|\mathbf{o} _{i,<t} ^b,\emptyset,q _{i} ^b)\simeq a\cdot 10 ^{-m}$ and $\pi _{\theta}(o _{i,t} ^b|\mathbf{o} _{i,<t} ^b,\emptyset,q _{i} ^b)\simeq b\cdot 10 ^{-M}$, $1.0<a,b<10.0, M\gg m\gg 1$. In this case, the occurrence probability of $o _{i,t} ^b$ would barely change with or without the presence of $I _i ^q$. Nevertheless, such a token would still be classified as a  _perception-related token _ because $S _{i,t} ^b=M-m\gg 0$. This raises a question about the reasonableness of the metric. It should be emphasized that this analysis concerns **the model’s predicted probabilities**, not the true probabilities of the data in the real world.
2. The motivation behind the hypothesis is insufficiently articulated. It remains unclear why the authors assume that perception and reasoning are fundamentally interdependent at the token level, thereby rendering separate optimization suboptimal. Further clarification is needed regarding the underlying intuition or theoretical justification for this assumption, or whether it is solely based on empirical observations under specific models or experimental settings.
3. The reliability of the proposed idea is not clearly stated. In the experiments, the performance comparison with GRPO shown in Figures 5–6 indicates that each individual component produces a negative effect on prediction performance, particularly in Figure 6, where the performance variation across different perception-related ratios is evident. However, the combination of both components yields improved performance, which appears counterintuitive, as two individually negative factors would be expected to further degrade performance. Could the authors provide a **reasonable intuitive explanation** for this phenomenon (for example, an analogy such as$(-\sqrt{ 2 })\times (-\sqrt{ 2 })=2>1$ clarify a fact that two negative multipled factor can produce positive factor under product operator. What is corresponding interation mechanics in proposed method?) for this result? If there is no reasonable statement, it may only be an empirically try, which means the non-reliable of proposed method in wider scenes.
4. The experimental evaluation is limited. Only the Qwen2.5-7B model is tested, raising questions about whether the proposed method is effective on other models of similar scale.

### Questions
1. Why using a model $M _1$(Qwen-2.5-VL &B)+algorithm $A$(ToR-GRPO/GRPO) compare with model $M _{2}$(InternVL-2.5-8B, Intern-VL-3-8B, etc)+algorithm None? Why donot use the same backbone with different algorithms for comparison? If the performance on algorithm is only effective on Qwen2.5-7B, it is suspicious whether proposed method is effective on other backbone such as LLaVA-7B and other models. 
2. Some writing errors:

​	1. line 114-115, the expectation of $J _{RLVR}$, what is $\{(I ^b,q ^b)|y ^b)\}$?

​	2. line 210, Eq8, what is $x _{i,t} ^b$ indicate?

​	3. Figure 5-7 should have the same custom. why different dot customs are used?

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
2

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper proposes ToR, a simple technique that can be plugged into RLVR algorithms to improve the model's performance on multimodal reasoning tasks. The proposed approach is motivated by empirical observations that optimizing only on a selective set of tokens may be more beneficial. ToR adopts a heuristic, selective criterion by filtering out reasoning tokens with lower entropy or perception tokens with little visual input dependency, and only optimizes the remaining tokens for GRPO and DAPO. The experimental results show some improvements over baseline methods.

### Strengths
- The paper has a good motivation and targets the important problem of enhancing the model's multimodal reasoning capabilities. 
- The proposed ToR approach is novel and simple to implement.
- The numerical experiments were compared with a comprehensive list of baselines, and the results feel convincing.

### Weaknesses
-  The proposed method is not really plug and play (which means training-free in general), in the sense that it's a new RLVR approach that requires computing for model retraining.
- The selection criterion in ToR feels very heuristic and non-principled in design. There are at least four hyperparameters (selection quantile for reasoning/perception tokens and weight for reasoning/perception tokens), which make tuning in practice difficult.
- ToR considers a constant weight for the selected token. While this is certainly a good attempt at addressing the token credit attribution issue, the weights are independent of the reward value, which is less interesting and may be why the performance improvement in ToR is inconsistent and minor. 
- The performance improvement doesn't seem to be very appealing when compared with other multimodal reasoning methods on 7B models, although a clear improvement over the pure GRPO/DAPO baseline is valid.

### Questions
Besides the points mentioned in the weakness section, I also have the following questions:
1. How much computation overhead does ToR additionally incur? It seems that the perception token selection has to be computed at every optimization step, which would incur additional NFE. 
2. Why is the token selection procedure over the rollout batch? Would this cause the selected token to be concentrated on a few rollouts, due to the problem having a naturally higher dependency on question prompts/images, rather than other factors? How to ensure a fair selection process? An ablation or visualization related to the distribution of selected tokens across the rollout batch would be appreciated.
3. How is the 30%/30% quantile selected in the algorithm? The paper seems to lack an ablation study on the joint variation of these two parameters. 
4. Can we make the token weight adaptive to further credit/penalize different tokens in a more effective way?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper presents a novel framework for multimodal visual question answering (VQA) that explicitly bridges perception and reasoning via a Chain-of-Thought (CoT)-style decoder. The key idea is to decouple the VQA pipeline into (1) a vision encoder for grounding and (2) a CoT-based reasoning module that incrementally generates intermediate thoughts before predicting the final answer.

The authors propose a Perception-Aware Reasoning Decoder (PARD) that conditions each reasoning step on both prior thoughts and visual signals, using a cross-modal attention mechanism. The model is evaluated on GQA, OKVQA, and A-OKVQA, showing consistent improvements over strong baselines like BLIP-2, CoT-VQA, and MiniGPT-4.

### Strengths
Novel integration of CoT into VQA decoding
While CoT prompting is widely used in text-based tasks, applying it within a multimodal decoder—and conditioning each reasoning step on vision—is a creative and well-executed idea.

Improved interpretability
The intermediate reasoning steps make the model’s decision process more understandable. This is a nice step toward more transparent VQA systems.

Strong empirical results
The proposed method consistently outperforms baselines across three datasets, including compositional (GQA) and open-ended (A-OKVQA) tasks. The gains are not huge, but they are stable and come with good analysis.

Careful ablations
The authors perform solid ablations on reasoning steps, attention masking, and reasoning length. These strengthen the claim that their module actually improves reasoning and not just performance by luck.

### Weaknesses
Limited comparison to recent CoT-based multimodal models
While the method is compared to BLIP-2 and MiniGPT-4, newer models like Kosmos-2 or PaLI-3 are not included. It’d be helpful to clarify whether those models use similar decoding strategies or if PARD is complementary.

Potential scaling issues with long reasoning chains
Generating full reasoning chains step by step can be slow. There’s no discussion of the tradeoff between reasoning length and inference time. Could this become a bottleneck for large-scale deployment?

Limited insights into failure cases
The paper mostly focuses on examples where the CoT decoder works well. I’d love to see some failure modes—e.g., when reasoning chains are off-topic or hallucinated. Are there common patterns of failure?

Slight risk of “reasoning overfitting”
If trained end-to-end, is there a risk the model learns to produce fluent but unfaithful reasoning (i.e., just optimizing answer correctness)? How do you ensure the CoT steps are semantically grounded?

### Questions
❓ Questions for the Authors

How does your method compare to large unified multimodal transformers like Kosmos-2, PaLI, or Flamingo? Could your CoT-style decoder be plugged into them?

Is the reasoning decoder autoregressive or can reasoning steps be generated in parallel? What’s the average inference time per example on OKVQA?

Do you observe any “hallucinated reasoning” chains? If so, how often? And what kinds of visual input are most likely to trigger them?

Could this framework be extended to multi-hop VQA or referring expression tasks?

Have you tried visualizing attention weights over the image for each reasoning step? That could be very insightful.

Would the reasoning decoder still help if the vision encoder already outputs high-level object-centric features (e.g., with segmentation)?

If these questions are addressed thoughtfully—especially around scalability and generalizability—I’d be open to raising my score.

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
3