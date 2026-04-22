# Steering LLM Thinking with Budget Guidance

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 8, 2

## Abstract
Recent deep-thinking large language models often reason extensively to improve performance, but such lengthy reasoning is not always desirable, as it incurs excessive inference costs with disproportionate performance gains. Controlling reasoning length without sacrificing performance is therefore important, but remains challenging, especially under tight thinking budgets. We propose budget guidance, a simple yet effective method for steering the reasoning process of LLMs toward a target budget without requiring any LLM fine-tuning. Our approach introduces a lightweight predictor that models a Gamma distribution over the remaining thinking length during next-token generation. This signal is then used to guide generation in a soft, token-level manner, ensuring that the overall reasoning trace adheres to the specified thinking budget. Budget guidance enables natural control of the thinking length, along with significant token efficiency improvements over baseline methods on challenging math benchmarks. For instance, it achieves up to a 26% accuracy gain on the MATH-500 benchmark under tight budgets compared to baseline methods, while maintaining competitive accuracy with only 63% of the thinking tokens used by the full-thinking model. Budget guidance also generalizes to broader task domains and exhibits emergent capabilities, such as estimating question difficulty.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a budget guidance method for steering the reasoning process of LLMs without requiring any LLM fine-tuning. Our approach introduces a lightweight predictor that models a Gamma distribution over the remaining thinking length during next-token generation. This signal is then used to guide generation in a soft, token-level manner, ensuring that the overall reasoning trace
adheres to the specified thinking budget. Budget guidance achieves up to a 26% accuracy gain on the MATH-500 benchmark under tight budgets compared to baseline methods, while maintaining competitive accuracy with only 63% of the thinking tokens used by the full-thinking model.

### Strengths
This paper proposes a budget guidance method for steering the reasoning process of LLMs without requiring any LLM fine-tuning. It can achieve up to a 26% accuracy gain on the MATH-500 benchmark under tight budgets compared to baseline methods, while maintaining competitive accuracy with only 63% of the thinking tokens used by the full-thinking model. Different benchmarks have been tested.

### Weaknesses
1. The method's core relies on modeling the remaining thinking length with a Gamma distribution. This is a strong assumption about the nature of the reasoning process. 
2. The paper compares against "baseline methods," but it's important to define these and include more sophisticated alternatives. So many papers have discussed on overthinking and underthinking. More experiments should be added.

### Questions
More experiments should be added.

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
5

### Summary
The paper proposes Budget Guidance, a test-time approach for steering the reasoning process of large language models without requiring any fine-tuning. The method employs a lightweight predictor to estimate the remaining reasoning length and uses this prediction to dynamically guide the generation process.

The central insight is that the budget-conditional distribution can be derived from the unconditional distribution combined with the predicted remaining length distribution, allowing the model to adjust its reasoning behavior at the beginning of each reasoning segment. 

Experimental results demonstrate that Budget Guidance consistently outperforms Budget Forcing while producing responses with comparable or shorter reasoning lengths.

### Strengths
1. The paper introduces a fine-tuning–free approach to guide the reasoning process of large language models (LLMs), which can be seamlessly integrated as a plugin into modern LLM serving systems.
2. The proposed Budget Guidance mechanism applies a soft, token-level steering that effectively regulates reasoning length while preserving the model’s inherent reasoning capability.
3. The predictor operates only once at the start of each reasoning segment, resulting in negligible latency overhead and maintaining system efficiency during inference.

### Weaknesses
1. Limited baselines: The paper only compares against one budget-based method. Missing broader hybrid-thinking or trade-off baselines, such as prompt-based classification or adaptive fast/slow reasoning strategies.
2. Lack of diversity in model structure: All tested models (DeepSeek-R1-7B/32B, Qwen3-8B) are dense models. It’s unclear whether Budget Guidance remains effective on MoE architectures.
3. Single token budget objective setting: Evaluations only target at half the original model’s full thinking length, without exploring adaptive or partial-budget variants.

### Questions
1. The paper does not report the number of NoThinking tokens, which prevents a clear and quantitative comparison between the proposed approach and baseline methods.
2. The predictor for Qwen3-8B is trained on reasoning traces from DeepSeek-R1 yet performs well. The authors claim model-agnostic generalization—could this result instead arise from architectural and scale similarity between dense models of 8B–32B size? (As the base model of DeepSeek-R1-Distill-Qwen is still Qwen)
3. The thinking budget is set to approximately half of the full reasoning length. What is the rationale behind this choice? Have other budget ratios been tested or compared?
4. Appendix H mentions that the predictor’s training data includes sequences like `<think>ANSWER_MESSAGE</think>ANSWER_MESSAGE`. What is the motivation for including such patterns, and could this design introduce misleading supervision signals?
5. Typo: Page 4, Line 287 — “OpenR1-Math-220k (Face, 2025)” should be corrected to “OpenR1-Math-220k (Hugging Face, 2025)”.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an adaptive LLM thinking method that uses a budget-conditional output distribution to control the output length under a budget-constrained setting. Through a Bayesian perspective, it models the conditional likelihood of the remaining length distribution using a Gemma distribution, which can easily produce the cumulative probability of length. By collecting reasoning chain data, it trains a predictor using MLE to fit two intrinsic parameters in the Gemma distribution, where the parameterization is based on a BEART model and embeddings of the target LLM. Empirical results show that the proposed method achieves comparable results with the thinking baseline with less tokens, and achieves higher performance than the Budget Forcing baseline.

### Strengths
- The proposed budget-conditional output distribution is interesting, and the framework has a principled design.
- The trained predictor shows task generalization, even though it is only trained on math.
- The paper is easy to understand.

### Weaknesses
- The evaluated baselines only include NoThinking and Budget Forcing, which are not representative and strong enough. There are many other existing adaptive thinking methods, but the authors did not review or compare their method against other adaptive thinking methods. This makes the empirical evaluation of the proposed method weak. I would suggest the other at least compare with two other adaptive thinking methods to show the real effective of the proposed method.
- If taking the last token embedding of all layers, it would be very costly to train such a predictor. Some inference cost analysis is shown in Table 3 for the proposed method, but the training overhead is unclear. It would be clearer if the author could show the training cost and compare the variant that only trains on the last token embedding on the last layer.
- The choice of the Gemma distribution is not justified. Is there any other distribution that is also suitable for modeling the length distribution? Can the author give justification/explanation and comparison with other distributions?
- The task generalization only shows that the math data could help to generalize to other domains. It is unclear whether the predictor trained on another domain could generalize to math or not. If it is not, its application would be limited if no high-quality math reasoning data is available.

### Questions
- In line 265, is this “linear projection” trainable or a fixed projection at initialization? Could you explain more about this? Is the [CLS] token also used as part of the input?

### Soundness
3

### Presentation
2

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
This paper presents a novel inference-time method called "budget guidance" for controlling the reasoning length of large language models (LLMs) without requiring fine-tuning. The approach introduces a lightweight auxiliary predictor that estimates the remaining thinking length using a Gamma distribution, which is then used to softly steer token-level generation toward a specified budget. Experiments on math reasoning benchmarks demonstrate significant improvements in token efficiency, with maintained accuracy under tight budgets, and the method shows promising cross-domain generalization. The work also provides insights into the predictor's ability to infer task difficulty and respond to prompt-based instructions.

### Strengths
1）The proposed “budget guidance” is a simple yet effective inference-time method that enables flexible control over chain-of-thought (CoT) length without supervised fine-tuning (SFT), avoiding computational costs and potential safety risks associated with LLM fine-tuning.

2）The lightweight predictor is innovative, as it models remaining reasoning length distributionally and generalizes well to out-of-domain tasks (e.g., scientific and logical reasoning), highlighting its strong adaptability.

3）The method enhances token efficiency substantially, achieving competitive accuracy with significantly shorter reasoning traces.

Overall speaking, the key originality lies in implicitly guiding LLMs based on remaining length predictions, avoiding hard cut-offs or rule-based interventions while preserving reasoning quality.

### Weaknesses
The study lacks a prompt engineering (PE) baseline. Given the insights that the predictor is prompt-aware and capable of adapting to instructions (e.g., generating concise or lengthy reasoning), a comparison with well-crafted prompts could validate whether similar length-control effects can be achieved without the predictor, strengthening the claim of necessity.

### Questions
Why is Dynasor not included as a baseline in Table 1? As an inference-time steering method, Dynasor's performance under comparable settings (e.g., token budgets) would offer a more comprehensive evaluation relative to existing baselines like budget forcing and NoThinking.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a budget-aware controllable LLM next-token-prediction for efficient thinking. In which, the final token distribution is modified with the learned predictor score distribution using a lightweight BERT model. The proposed Budget guidance does not need to train the main LLM but only to train the auxiliary module to enable test-time control over the reasoning length of LLMs.

The proposed method is novel, but the additional n-dimensional vector $a_t$ is not clearly defined. In addition, how and why the Gemma modeling of remaining length is used to affect the final token distribution with $c_t$ are not well explained, making the method not so convincing. 

Many training-free efficient test-time scaling methods are proposed, which are not compared as baselines. The accuracy loss of the proposed method on AIME24, AMC, and TableBench is still noticeable. Therefore, it is hard to validate the effectiveness of this paper.

### Strengths
This paper proposes a budget-aware controllable LLM next-token-prediction for efficient thinking. The authors propose to directly model the remaining thinking length as a Gamma distribution and use the CDF of Gamma distribution to compute the predictor score, which are used to steer the final LLM token distribution. The normalized product of $u_t$ and $a_t$ is defined as the conditional vector.

### Weaknesses
The underlying motivations of using the Gamma distribution to model the remaining length and using the CDF of it as the generated steering vectors to guide the final output decoding of LLMs are not clearly explained. 

The experiments are not thoroughly conducted with more missing SOTA training-free baselines. In addition, the accuracy degradation of three datasets are not ignorable. 

The end-2-end runtime reduction and the additional cost of embedding and BERT inference are not reported.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
