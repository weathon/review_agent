# Draft, Verify, \& Improve: Toward Training-Aware Speculative Decoding

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Autoregressive (AR) decoding is a major latency bottleneck for large language models. Speculative decoding (SD) accelerates AR by letting a drafter propose multi-token blocks that a verifier accepts or rejects. However, many SD systems require heavy offline training or extra components. These choices raise data/compute cost and can yield brittle drafters under distribution drift.
We introduce \emph{Draft, Verify, \& Improve (DVI)}, a training-aware self-speculative framework that combines inference with continual online learning. We partition an LLM into a drafter and a verifier, and during generation, verifier accept/reject decisions are converted into supervision signals and used to update the drafter head. 
A simple \emph{KL$\rightarrow$RL} schedule bootstraps calibration via online distillation and then adds reward-masked cross-entropy with a on-policy policy-gradient term, preserving lossless, single model deployment.
On Spec-Bench, DVI achieves a $2.16\times$ wall-time speedup, on par with SoTA approaches like EAGLE-2, while orders of magnitude less data for training, and ablations show that DVI outperforms KL-only online distillation. 
DVI demonstrates that \emph{training-aware} self-speculation can deliver state-of-the-art, lossless speedups with minimal training overhead.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an interesting idea of integrating online learning into speculative decoding. However, the manuscript suffers from three major weaknesses that significantly undermine its contributions: limited performance gains, failure to address the core inference bottleneck, and outdated comparisons that ignore recent advancements like EAGLE-3.

### Strengths
​1. Novel Integration of Online Learning:​​ The core idea of closing the loop between speculative inference and online learning is innovative. Treating the verifier's commit decisions as real-time, self-supervised feedback for the drafter is a compelling approach to continual adaptation, potentially mitigating drafter brittleness under distribution drift without requiring separate offline datasets.

2. High Data and Training Efficiency:​​ A significant advantage of DVI is its minimal data requirement. The paper demonstrates that effective speedups can be achieved after exposure to only 2,000 prompts, which is substantially less than the millions of prompts required by methods like Medusa or EAGLE. This makes DVI a highly cost-effective and practical option for scenarios with limited training data or the need for rapid deployment.

### Weaknesses
1. Marginal Performance Improvements​
The claimed 2.16× average speedup appears modest when examined closely. As shown in Table 2, DVI's performance is actually ​inferior to EAGLE-2​ on several tasks (MT-Bench and Summarization), while the advantages in other tasks are minimal (e.g., only 0.07× faster in QA). Such marginal gains raise questions about the practical significance of the proposed method.

2. Misplaced Focus: Training Efficiency ≠ Inference Speed
The paper heavily emphasizes reduced training cost(using only 2,000 prompts) as a key advantage. However, this addresses a secondary concern while overlooking the primary challenge in LLM deployment: ​maximizing inference speed and minimizing latency.

3. Timeliness Issue: Missing Comparison with EAGLE-3
The most serious flaw is the omission of ​EAGLE-3​ (Li et al., 2024b), which represents the current state-of-the-art in speculative decoding.

### Questions
1 ​Include EAGLE-3 Comparisons: Essential experiments comparing DVI with EAGLE-3 under identical settings must be conducted.
2 Broaden Experimental Scope: Extend evaluations to larger models (e.g., 70B parameters) and different decoding strategies to demonstrate generality.

### Soundness
2

### Presentation
2

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
The paper proposes a training-aware self-speculative decoding framework that partitions a single LLM into shallow drafting layers and deep verification layers, then converts verify accept/reject signals into online supervision for the drafter. 

A KL -> RL schedule warms up via online distillation to the frozen verifier and then adds reward-masked cross-entropy plus a light on-policy policy-gradient term, keeping speculation lossless under the verifier sampler. 

On Spec-Bench with Vicuna-7B, DVI reports ~2.16X average wall-time speedup, competitive with EAGLE-2, while training on only 2k prompts and requiring no auxiliary drafter.

Overall, I think this is a good piece of work that provides insights in how to build novel speculative decoding frameworks and is worthy of acceptance.

### Strengths
1. The overall design is simple and deployment-friendly. The entire DIV consists of one backbone, a LoRA drafter head, and a frozen verifier. 

2. The training-aware self-speculation turns commit decisions into online supervision, and is able to adapt the drafter to live traffic and mitigating distribution drift.

3. DVI is both data and computation efficient through empirical experiments. It achieves competitive speedups with a tiny online budget (e.g. 2k prompts; single-GPU setup), compared to orders-of-magnitude larger offline training for baselines.

### Weaknesses
Presentation needs to be improved. In fact that is the only factor that prevents this work from being published. For example, all equations are not numbered and reviewers are not able to refer to them. Some references are not in standard format, e.g. L74.

### Questions
1. How sensitive are results to split index k and proposal depth k_spec?

2. Can you report mean +- stdev over seeds for the main Spec-Bench tables? That should be the common practice for methods evaluated on Spec-Bench.

### Soundness
3

### Presentation
1

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
This work introduce Draft, Verify, & Improve (DVI), a training-aware self-speculative framework that combines inference with continual online learning to tackle the training overhead of speculative decoding (SD) methods. DVI incorporates a frozen verifier and an online-learned drafter head that converts commit decisions of SD into self-supervision, making the SD model data-efficient without separate offline datasets or long pre-training. Experimental results demonstrate competitive speedups compared to other SOTA SD methods with minimal training overhead.

### Strengths
1. The idea of using a frozen verifier and an online-learned drafter head to save training overhead of SD models is interesting and promising. The presentation of the proposed method is clear and easy to follow.

2. The experimental results demonstrate the proposed method achieves competitive speedups compared to other SOTA SD methods with minimal training overhead.

### Weaknesses
1. The experiments is based on a small-scale, outdated LLMs Vicuna-7B. Further experiments on larger models (30B or 70B parameters) are expected to yield improved value of this work.

### Questions
Although the speedup metrics are competitive compared to other SOTA SD methods, there are still gaps between the proposed methods and SOTA methods in the mean accepted tokens metrics. How to understand this gap?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The algorithm proposes Draft, Verify, & Improve (DVI) which is a self-speculative decoding method with online training. Specifically, it adopts first few layers as drafter and add lora head and use rest of the layers as verifier with verifier head. Then, it trains both lora heads in an online manner where supervision is given from the token acceptance reward. The result shows improved performance on Spec-Bench with smaller number of trainig data.

### Strengths
* The proposed algorithm introduces new types of online SD combining with self-speculative decoding.
* The motivation of the paper is clear.
* Proper ablations are conducted with sound presentations.

### Weaknesses
* **Novelty** : Prposed method combines self-speculative decoding with online training, but utilizing first few layers as drafter is already investigated as in [1]. Also, even DVI differs Online speculative decoding [2] in utilizing only accepted tokens, and combine it as a reward-signal in training, the effect of adding reward-suprevision is not independently investigated which limits the contributions of the paper. 

* **Train time scaling** : While the proposed algorithm shows decent performance with only a samll amount of the train data, often one might need better drafter with more computes for training but no experiment is done.

* **Tree decdoing and stronger baselines** : The baseline should contain stronger baselines like EAGLE-3 [3] for fair comparison. Moreover, the result on tree-decoding of the drafter ([4], [5]) should be tested which generally shows improved speed-ups while the experiments are done only with single trajectory decoding.

* **Limited details** : Experiment details like warm-up steps or training hyper-parameters seems like being omitted.

### Questions
* Can authors show the performance of the SD along the number of trained tokens (i suspect the training might saturate earlier than other methods)?

* Can you test the trained model on tree-decoding scenario?

* Can authors evaluate the trained models on OOD dataset? I think RL-type component might hinder generalizability.


[1] (Liu et al.) Kangaroo: Lossless self-speculative decoding via double early exiting.

[2] (Liu et al.) Online Speculative Decoding.

[3] (Li et al.) EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test

[4] (Cai et al.) MEDUSA: Simple LLM Inference Acceleration Framework with Multiple
Decoding Heads

[5] (Li et al.) EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees

### Soundness
2

### Presentation
3

### Contribution
1
