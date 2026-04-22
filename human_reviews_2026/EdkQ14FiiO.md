# ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Token-level attention tuning -- a class of training-free methods including Post-hoc Attention Steering (PASTA) and Attention Calibration (ACT) -- has emerged as a promising approach for improving frozen LLMs via interpretable interventions. However, these methods rely on auxiliary heuristics to identify important task-specific tokens, which can introduce bias and limit applicability when token importance is ambiguous or when optimized kernels make attention maps inaccessible. We propose a simpler alternative: intervening only on the initial token (e.g., <BOS> in LLaMA). We theoretically show that adding lightweight biases to this token’s attention logits systematically shifts and reshapes downstream attention patterns -- an effect amplified by its natural role as an attention sink. Empirically, we find that this tuning can improve LLM performance and better elicit pretrained knowledge, with stronger effects in early layers and distinct scaling preferences across attention heads. Building on these findings, we introduce ZeroTuning, a training-free method that improves LLM performance by applying head-specific attention adjustments to the initial token, requiring no parameter updates. We present two variants: a supervised mode that calibrates on validation examples, and an unsupervised mode that directly minimizes output entropy. ZeroTuning requires no KV-cache or decoding changes and is kernel-agnostic (works with SDPA and FlashAttention). It requires only four lines of modification to standard \texttt{LlamaAttention} code, achieves gains across 15 datasets, and outperforms prior, more complex methods. For example, on Llama-3.1-8B, it yields relative improvements of 19.9% on classification, 4.5% on question answering, and 2.1% on dialogue. ZeroTuning also works out of the box with quantized inference and maintains its improvements as context length increases. Our work provides a lightweight tool for inference-time improvement, advancing both optimization and interpretability. Our code and runnable demo are available at https://anonymous.4open.science/r/ZeroTuning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ZeroTuning, a training-free method to improve the performance of frozen LLMs by making a simple yet powerful intervention: applying lightweight, head-specific attention adjustments only to the initial token (e.g., the <BOS> token).

### Strengths
**Training-Free & Parameter-Efficient**: It improves model performance without updating any weights.

**Theoretical Grounding**: The method is supported by a theoretical insight linking the initial token's attention to the entropy of the entire attention distribution, providing a principled foundation for the intervention.

### Weaknesses
**Sensitivity in Alternative Implementations**: The paper shows that tuning the key/query states directly (as a kernel-agnostic alternative) leads to a much sharper and more sensitive performance drop outside the optimal range compared to tuning attention scores, making it a less stable implementation choice.

**Calibration Overhead**: The supervised variant requires a labeled validation set (500 examples in their setup) to calibrate the optimal scaling factors. Although the unsupervised variant (using entropy minimization) is a significant contribution, the paper's main results and comparisons are based on the supervised mode, which still incurs a data and computation cost for calibration.

### Questions
The paper mentions that ZeroTuning requires only "four lines of code," but could the authors quantify its runtime impact? How does the inference latency and memory footprint compare to the vanilla model, and how does this overhead scale with context length?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work introduces ZeroTuning, a training-free attention-tuning method that modifies only the initial (sink) token (e.g., BOS). By scaling that token’s attention logits with a single factor, the method adjusts the entropy/sharpness of downstream attention while preserving the relative proportions among non-initial tokens. Notably, the scaling factor that minimizes entropy empirically aligns with the factor that maximizes accuracy. Building on this, the authors provide both supervised and unsupervised (entropy-minimization) calibration variants, implemented in a kernel-agnostic way (compatible with SDPA/FlashAttention) by altering attention maps only. Extensive experiments show consistent gains across 15 datasets, with ZeroTuning outperforming prior, more complex approaches.

### Strengths
1. This work theoretically and empirically demonstrates that the initial tokens can function as a reliable controller for the attention dynamics, which is also strongly related to the next-token prediction entropy. Moreover, the systematic head-wise and layer-wise initial token scaling analysis provides more insights and reliable motivation for the proposed ZeroTuning method. 

2. The proposed plug-and-play attention adjustment, ZeroTuning, is simple yet effective and well-motivated both empirically and theoretically. Its supervised and unsupervised calibration variants are easy to implement and attention kernel-agnostic. Overall, its analysis and effectiveness inspires a fresh look at the role of the initial sink token in shaping attention.

3. This extensive experiments demonstrate that ZeroTuning can achieve consistent gains across different models and downstream tasks including text classification, domain-specific multiple choice and multi-round conversation, outperforming the previous methods including PASTA and ACT with negligible engineering overhead. 

4. The writing is clear and easy to follow. The paper is well structured.

### Weaknesses
1. While Table 5 shows that ZeroTuning improves even with fixed γ and scales with more search, the paper does not quantify the time/energy required for Level-0/1/2 nor its trade-off with accuracy. 

2. γ is calibrated per dataset, and its robustness to distribution shifts and mis-specified γ is unclear, and the cost of head classification is not reported as well.

3. Because ZeroTuning controls attention by scaling the initial sink token, its effect can fade or fluctuate over very long contexts and in streaming with KV-reuse, and the same γ induces different sharpness across sliding windows, causing calibration drift or confidence oscillations.

### Questions
1. What is the overhead for the supervised profiling process? Could you please clarify the GPU hours it will cost?

2. For the unsupervised method which identifies the optimal heads and scaling factor γ by minimizing the average next-token prediction entropy, what is the actual cost? Will this significantly make the inference slow?

3. For the long context generation tasks, do we need to adjust γ dynamically as well? If not, will using the adaptive values harm the performance?

4. Have you ever tried to make γ as a learnable parameter? Because if it can be learned during some fine-tuning process, we can make it work in the inference stage without any overhead and more adaptable. 

5. What will ZeroTuning perform on math reasoning tasks and reasoning models? Will this method still demonstrate performance improvement?

### Soundness
3

### Presentation
3

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
The paper introduces ZeroTuning, a training-free method to improve large language model (LLM) performance by adjusting the attention logits of the initial token (such as, \<BOS>\). The approach selectively scales attention heads associated with this token, requiring no parameter updates and only minimal code changes. The authors provide both supervised and unsupervised variants—respectively calibrated on validation data or optimized via output entropy minimization. Theoretically, they show that modifying the initial token’s logits can monotonically regulate the downstream attention entropy due to its inherent role as an attention sink. Empirically, ZeroTuning demonstrates consistent gains across three different model types (Llama, Gwen, Deepseek) and on several downstream tasks and maintains compatibility with various attention kernels (suchSDPA, FlashAttention) and quantized inference setups.

### Strengths
In short: the paper identifies a control lever in large language models (LLMs) - the initial token (such as \<BOS>\) - and shows how modulating its attention yields performance gains. The method is practically appealing since it is lightweight (just a few lines of code to scale attention) and kernel-agnostic. 


-  Provides experimental Analysis on - how scaling the attention weight of the initial token affects the downstream distribution of attention among other tokens, experiments showing that tuning the initial token produces larger improvements than tuning other token positions, showcase layer-wise and head-wise analyses, showing how the effect varies across shallow/middle/deep layers and across individual attention heads.

- Demonstrates broad empirical gains across multiple LLMs in the experiment section (such as Llama-3.1-8B, Llama-2-13B, Qwen-2-7B, DeepSeek-R1-14B) and a variety of downstream tasks (classification, QA, conversation).

- The paper contribution is in interpretability/mechanistic understanding of LLMs (why the initial token works as a control point and how attention patterns propagate). 
- The writing is clear in explaining both the motivation (limitations of previous attention-tuning methods that rely on heuristics) and the method’s design.

### Weaknesses
Following are some limitations I see in the paper:

 - Limited model generalization: most of the analysis and findings in section 3 reply on just one model, Llama-3.1-8B-Instruct, raising concern that some of those effects may be model-specific. I would suggest to add some findings for the other models tried as well - Qwen or Deepseek, to show generality.

 - Huge hyperparameter tuning overhead: This method introduces a huge number of hyper parameters to tune - task specific tuning, layer wise tuning, head specific tuning. This limits its practical applicability.

- Unclear task selection & possible training overlap: The paper lacks explanation for why specific evaluation tasks were chosen or do they cover diverse range of SFT based downstream tasks? I would suggest to clarify task selection rationale and check for the overlap/lack of task specific data in the pertaining and if the gains are correlated with that.

- Weak justification for “bias correction” claim (Sec. 3.2): the claim that scaling (y > 1) “corrects bias” lacks empirical evidence.

### Questions
- Including analysis for the model variants in section 3 would show the generalization of the claims.

- It would be helpful to compare it with other task specific tuning methods such as  SFT / adapter and to show if it works well with them as well?

- Minor presentation issues, a typo in Figure 5 (“shadow” label instead of "shallow").

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
4

### Summary
The paper introduces ZeroTuning, a training-free approach that steers large language models by scaling the attention weight of the initial token (BOS). The authors argue this operation “monotonically controls” the entropy of downstream attention, and propose both supervised and unsupervised (entropy-minimizing) calibration procedures. Empirical results across 15 NLP benchmarks and multiple model families suggest measurable gains (e.g., +19.9% on classification, +4.5% on MC-QA), with additional analyses on layer and head behavior showing BOS as a dominant control point.

### Strengths
- **Simple and broadly applicable idea**. The notion that adjusting only the first token’s attention can improve diverse tasks is conceptually elegant and easy to integrate, requiring no retraining or architectural modification.

- **Comprehensive empirical coverage**. The paper evaluates across multiple datasets, models, and settings (few-shot, quantized, SDPA/FlashAttention), providing reasonable evidence of generality.

- **Clarity and presentation**. The paper is clearly written and well-structured, with intuitive figures and concise mathematical derivations that make the mechanism easy to follow. Readers can quickly understand the motivation and implementation.

### Weaknesses
- **Theoretical over-reach**. The claim that BOS scaling “monotonically controls attention entropy” lacks formal proof; the derivation only handles pairwise attention differences, not entropy. This gap weakens the conceptual basis of the unsupervised variant.

- **Transductive unsupervised tuning**. The unsupervised version minimizes entropy on test inputs, while baselines are not given equivalent unsupervised access, overstating generalization gains.

- **No statistical robustness**. All results appear single-seed, with no confidence intervals or variance reporting. For small gains (1–3 %), the significance is uncertain.

- **Head profiling risks data-set-specific overfitting; selection rules are ad-hoc**. Heads are labeled by measured response to γ and then the “dominant head type” is scaled; implementation tunes the top 40% of identified heads. Multiple-testing control, stability across resamples, and cross-dataset transfer of head labels are not demonstrated.

- **Confounding with generic decoding calibration.** Reported gains may largely reflect generic logit/decoding tweaks rather than an attention-specific effect. The paper notes BOS scaling behaves like temperature, yet no matched unsupervised baselines (e.g., temperature, label-bias/length penalties) are tuned on the same unlabeled inputs. Add these controls and report invalid-output rates to isolate a genuine attention-level contribution.

### Questions
Please refer to weakness section above.

### Soundness
2

### Presentation
3

### Contribution
2
