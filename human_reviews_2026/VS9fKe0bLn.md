# Guard Vector: Beyond English LLM Guardrails with Task-Vector Composition and Streaming-Aware Prefix SFT

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
We introduce Guard Vector, a safety task vector computed as the parameter difference between a guardrail model (Guard Model) and a same-architecture pretrained language model. Composing this vector with a target language model yields a Target Guard Model (TGM). We then adapt TGM with a streaming-aware approach that combines prefix-based training and evaluation with a classifier that produces a single-token output. With this composition alone, TGM improves classification quality over established Guard Models across standard safety suites and enables language extensibility to Chinese, Japanese, and Korean, requiring neither additional training nor target language labels. It also demonstrates model portability across two widely used public guardrail backbones, Llama and Gemma. With prefix SFT (supervised fine-tuning), TGM preserves classification quality under streaming by aligning the behavior between prefix inputs and full-text inputs. The single-token output design increases throughput and reduces latency. Together, these components reduce data and compute requirements while promoting streaming-aware evaluation practices, thereby contributing to a more responsible AI ecosystem.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work presents an experimental study on constructing a target guard model by combining a continued pretraining model for a specific language with a Guard Vector, obtained through the parameter difference between a Guard Model and its corresponding pretrained model. The study demonstrates that safety behaviors can be effectively transferred through parameter vector migration. Additionally, the authors explore a prefix-based SFT strategy to reduce inference latency in streaming mode while maintaining parity in classification accuracy.

### Strengths
- Provides strong empirical evidence that safety behaviors can be transferred without additional training data, merely through parameter vector manipulation.
- Proposes an interesting and practical prefix-based protocol that enables low-latency inference with a single-token prefix, achieving performance parity in streaming mode.

### Weaknesses
- Readability issues: The abstract introduces multiple ideas simultaneously, making the central focus difficult to grasp. The Evaluation Metrics section (lines 276–293) requires reformatting for clarity. In Table 6, the F1 delta arrows are ambiguous and may lead to misinterpretation.
- Dependency on resources: The approach relies on the availability of open-source guard and pretrained models. Moreover, the continued pretraining on a language-specific corpus limits the demonstrated cross-lingual generalization.
- Limited task and language coverage: The experiments focus solely on classification tasks within a narrow set of languages.

### Questions
- Could the authors evaluate both Gemma and Llama models on each of the test datasets presented in Table 6 to enable direct comparison?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Guard Vector, a task-vector style method to transfer safety/guardrail behavior from an English guard model (e.g. Llama Guard 3) into a non-English continual-pretrained (CP) model of the same architecture, and then makes this target guard model streaming-aware via prefix SFT with single-token classification. The goal is to (i) get guardrails “beyond English” without extra target-language labels, and (ii) make guardrails actually usable in production streaming settings, where early detection + low latency matter. The paper evaluates on Korean datasets plus a helpfulness (all-SAFE) set, and even shows portability to Gemma via ShieldGemma.

### Strengths
1. The use of task-vector composition (Guard Vector) is a highly practical and novel approach for cross-lingual safety alignment. It effectively transfers safety behaviors to Chinese, Japanese, and Korean with significant F1 score gains over baselines, all without requiring any additional training or target language labels. This drastically lowers the barrier to deploying guardrails in diverse language environments.

### Weaknesses
1. Limited Technical Depth in Task Vector Composition: The paper defines the Guard Vector simply as the parameter difference and the composition as a simple addition (Equations 2 and 3). While the results are good, there is no in-depth analysis of why this simple linear composition works so effectively in the cross-lingual setting, especially compared to more complex vector merging techniques like TIES-Merging or other task arithmetic methods. The exclusion of LayerNorm parameters is mentioned but the motivation is only briefly cited from prior work. A deeper analysis of the parameter space and vector alignment would strengthen the technical novelty.

2.  Comparison baselines need to be tighter: They compare to LG3, Kanana, ShieldGemma-origin variants, but comparing against: (i) a strong multilingual safety-tuned LLM (e.g. XLM-R-based or mT5-based safety classifier), (ii) a simple translate-then-classify baseline (translate to English → Llama Guard 3) would strengthen the claims.

3. Lack of failure case analysis: While the paper carefully compares offline vs. streaming regimes and even surfaces an important negative result (full-text SFT collapses under streaming), it does not provide a dedicated error/failure analysis section.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a method to transfer safety behaviors between large language models without retraining. It defines a ''Guard Vector'' as the parameter difference between a safety-aligned guard model and its base pretrained model, then adds this vector to another model’s weights to create a Target Guard Model that inherits safety traits. To enable real-time moderation, the authors introduce Streaming-Aware Prefix Supervised Fine-Tuning, which trains on partial prefixes for early risk detection, and a single-token classifier to reduce latency. Experiments on English and Korean datasets show large gains in F1 score, faster throughput, and nearly identical offline versus streaming performance, while also demonstrating transfer across architectures (Llama↔Gemma) and languages (CJK).

### Strengths
The proposed Guard Vector approach that computing a parameter delta between a safety-tuned model and its base pretrained model, then composing that delta with another model in a target language. The authors also introduce prefix SFT to adapt the resulting guardrail for streaming, enabling early risk detection with a single-token classifier. The experimental section is extensive, covering Korean, Japanese, and Chinese setups, and the results are consistently strong across Llama and Gemma. 
1. The paper has a good structure and logic, and the appendix provides more experimental details.
2. A new Composition idea to define a clear and reproducible task-vector arithmetic framework for transferring guardrail behaviors.
3. Good Experimental design with quantitative metrics and comprehensive analysis.

### Weaknesses
The Guard Vector idea, although effective, largely repurposes existing task-vector and model-merging techniques with limited theoretical or analytical novelty. Some important design choices, such as omitting LayerNorm parameters, fixing τ = 0.5, and using 100-character prefix intervals—are presented without justification or sensitivity studies. 
1. Despite mentioning Chinese and Japanese results, the analysis and datasets are minimal; nearly all in-depth experiments centre on Korean.
2. The monotonic SAFE→UNSAFE assumption excludes self-correcting responses, potentially biasing the dataset and overestimating early detection accuracy.
3. Full-text SFT collapses under streaming (e.g., −15.23 to −26.65 F1 in Table 2), but the paper does not provide more analyse  results.

### Questions
1. Why 100 characters instead of token-based scheduling, and how sensitive is TTD to this granularity?
2. Could you explain why throughput per-token barely improves despite a single-token classifier? Does this indicate that latency reduction comes mostly from fewer decode loops rather than actual compute efficiency (Table 4)?
3. When composing ShieldGemma → Korean Gemma 2 IT, F1 improves +10.6 pp. Did you consider whether this gain persists if the Gemma CP model is multilingual rather than Korean-specific? Otherwise, improvements might be due to data domain alignment, not guard vector transfer.

### Soundness
3

### Presentation
3

### Contribution
2
