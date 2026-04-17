# House, G.P.T.: Diagnosing Pathological Chain-of-Thought in Reasoning Models

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Chain-of-thought (CoT) reasoning is fundamental to modern LLM architectures and represents a critical intervention point for AI safety. If models are incapable of performing harmful actions without reasoning efforts in the CoT, monitoring the CoT becomes a valuable tool for implementing safety guardrails. However, CoT reasoning may have properties which prevent it from being used for monitoring---we call these properties \textbf{pathologies}. Prior work has identified three distinct pathologies: \textbf{post-hoc rationalization}, where models generate plausible explanations backwards from predetermined answers; \textbf{encoded reasoning}, where intermediate steps conceal information within seemingly interpretable text; and \textbf{internalized reasoning}, where models replace explicit reasoning with meaningless filler tokens while computing internally. To better understand and discriminate between these pathologies, we present a systematic set of novel health metrics---Necessity, Paraphrasability, and Substantivity---that are simple to implement, computationally inexpensive, and task-agnostic. To validate our approach, we develop ``model organisms'': models deliberately trained to exhibit specific CoT pathologies. We demonstrate that our metrics can reliably diagnose these conditions. Crucially, we find that diagnostic signatures are most pronounced at \textit{early} training checkpoints and may attenuate as training progresses, suggesting these metrics are most effective as \emph{early warning indicators} during model development. Our work provides a practical toolkit for assessing CoT pathologies, with direct implications for training-time monitoring, scalable oversight, and AI alignment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces three task-agnostic health metrics (reliance, paraphrasability, and substitutability) to diagnose pathological chain-of-thought (CoT) reasoning in large language models. The authors create "model organisms" - models deliberately fine-tuned to exhibit specific CoT pathologies (encoded reasoning, internalized reasoning, and post-hoc reasoning) - and demonstrate that their metrics can successfully differentiate between these pathologies. They apply their diagnostic framework to several open-weight models, finding that most display relatively healthy CoT signatures on GSM8K.

### Strengths
The paper addresses an important problem for AI safety and model interpretability. The systematic approach to creating model organisms for each pathology type is well-structured, and the metrics are computationally inexpensive and model-agnostic, making them practical for deployment. The experimental methodology clearly demonstrates that each pathology produces distinct metric signatures, enabling differential diagnosis. The paper is generally well-written with clear explanations of the pathologies and metrics.

### Weaknesses
My main concern is that the model organisms are created through highly artificial training procedures, which may not reflect how these pathologies would naturally emerge in practice. For the encoded reasoning organism, training on a fixed animal-themed codebook is far removed from realistic steganographic encoding that might develop during training (such as in [1]). I also notice a lack of references to existing LLM steganography literature. Similarly, the internalized reasoning organism is trained explicitly on lorem ipsum text, which seems unlikely to occur naturally. The post-hoc organism simply appends answers to questions, which is definitional rather than emergent. This raises questions about whether the metrics would successfully detect naturally-occurring pathologies versus these synthetic versions.

Furthermore, the wild-type results are difficult to interpret. For instance, GPT-OSS-20B shows extremely high reliance scores (18.9), which the authors attribute to memorization of GSM8K answers. However, high reliance is supposed to be a sign of healthy CoT according to Table 1. This suggests the metrics may not be capturing what they intend to measure, or that the expected signatures in Table 1 are incomplete.

The metrics may also conflate training distribution effects with true pathologies. As the authors acknowledge, "interventions like removing CoT or substituting it with random text may push some model families further off-distribution than others due to differences in their training procedures" (Section 6). The paper does not provide a clear way to distinguish between low metric scores caused by genuine pathologies versus those caused by distribution shift, particularly for the substitutability metric.

Other relatively minor weaknesses:
- Despite using the word "causal interventions" (such as in Section 7 Conclusion), the metrics infer pathology from output statistics but do not verify internal causal mechanisms. This contrasts interpretability's normal definition of causal interventions.
- The paraphrasability metric relies on another LLM (Gemini-1.5-flash) to generate paraphrases, introducing an additional source of potential bias and error. Different paraphrasers might produce different results, and there is no analysis of how robust the metric is to the choice of paraphrasing model. Additionally, the paper mentions using "styles" (short, polite, negative, verbose, typos, reversal) but provides limited justification for why these particular styles were chosen or how they relate to detecting encoded reasoning.

---

[1] Motwani, Sumeet, et al. "Secret collusion among ai agents: Multi-agent deception via steganography." Advances in Neural Information Processing Systems 37 (2024): 73439-73486.

### Questions
The paper lacks comparison to existing faithfulness metrics from prior work [Lanham et al., 2023; Tutek et al., 2025]. How do these metrics perform relative to or in combination with existing approaches? Are they complementary or redundant? This comparison would strengthen the contribution.

The paper acknowledges a critical limitation but does not adequately address it: "Qwen3-8b should not be interpreted as a baseline for non-pathological CoT since various forms of unfaithfulness have been reported within reasoning models" (Section 6). If the baseline model used to validate the metrics may itself have pathologies, how can we trust that the metrics are correctly identifying pathological versus healthy CoT?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper considers whether CoT explanations from large language models genuinely reflect their reasoning. It identifies three failure modes---post-hoc rationalization, encoded reasoning, and internalized reasoning---and proposes three corresponding diagnostic metrics: 1) reliance, 2) paraphrasability, and 3) substitutability. The metrics are validated on synthetically finetuned models ("model organisms") exhibiting these pathologies and applied to real open-weight models on GSM8K. The paper is timely and introduces a clear framework. However, its empirical scope is narrow (mostly GSM8K), interpretation of metric outputs can be unclear (large variance, unclear thresholds), and practical deployment challenges (e.g., intervention off-distribution effects, API constraints) are underexplored. Overall, it's an interesting contribution but would benefit from broader experiments without overclaiming and clearer guidance for practical use.

### Strengths
1. Addresses a timely and important problem: whether CoT is trustworthy for monitoring reasoning in LLMs.

2. Clear taxonomy of three CoT pathologies and corresponding diagnostic metrics (reliance, paraphrasability, substitutability).

3. Use of fine-tuned "pathological" models provides controlled validation of metrics.

4. Empirical results on real open-weight LLMs add practical relevance.

5. Writing is clear and limitations are explicitly acknowledged.

### Weaknesses
1. Overbranding / overstated novelty: The term “model organisms” is unnecessary rebranding of controlled/synthetic fine-tuning, and the paper overclaims originality.

2. Evaluation is very narrow---focused primarily on GSM8K math reasoning. unclear generalization to other reasoning tasks or domains.

3. Metric interpretations are unstable (high variance, unclear thresholds), making "healthy vs pathological" judgments ambiguous.

4. Practical deployment challenges (API-accessible models, off-distribution interventions, computational cost) are underexplored.

5. No ground truth for "pathology" in real models/claims of healthiness are inferred, not verified.

### Questions
1. Why is the term "model organism" preferable to simply "controlled fine-tuned model"? What conceptual or methodological value does the rebranding add?

2. How do the metrics perform outside GSM8K (e.g., commonsense reasoning, multi-hop QA, or open-ended tasks)?

3. Can you provide guidance or thresholds for interpreting metric values in practice? When is a model “pathological” vs “healthy”?

4. How sensitive are the metrics to paraphrase quality, model size, or prompt design?

5. For real-world deployment (especially black-box APIs), how feasible is it to compute these metrics without access to full log-probabilities or ability to modify CoT?

### Soundness
2

### Presentation
1

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
This paper presents an evaluation framework for detecting problematic CoTs in large reasoning models. The authors first characterize problematic CoTs as “pathologies” and divide them into three categories based on previous work: post-hoc rationalization, encoded reasoning, and internalized reasoning. They propose three metrics for evaluating these issues: reliance, paraphrasability, and substitutability. The measurement relies on comparing answer log probabilities before and after interventions on the CoT. To validate their metrics, they create “model organisms” by fine-tuning models to deliberately induce the pathological behaviors.

### Strengths
- The paper proposes new metrics to evaluate problematic behaviors in long CoTs from reasoning models. 
- The proposed metrics are model-agnostic, task-agnostic, and computationally cheap (just log probability comparisons). 
- The metrics capture different aspects of CoTs.

### Weaknesses
**1. Model organisms may not reflect pathologies.**
- I find the primary method of validating the metrics - fine-tuning capable models to exhibit some problematic behaviors - not convincing. It creates artificial cases that do not imply real unfaithful or pathological CoT cases in the wild. Real pathologies might emerge more subtly and be harder to detect. 
- All wild-type models tested appear healthy (Table 3), so we don't know if metrics work on real pathologies. 
- How well could the metrics detect pathologies "in the wild"?

**2. Interpretations of the metrics are unclear.**
- Is there a range (or bounds) for the metrics? Table 1 simply shows the metrics can be "high" or "low", but there is no way to quantify how high is too high. 
- Why are both high and low values possible for some pathologies? 
- Table 3: The use of log probs can make the metric values very extreme and less interpretable. Why are the values not normalized? 

**3. The pathologies are all based on prior work, making the paper's contribution weak.**
- The described pathologies are all not new - CoT unfaithfulness is a longstanding problem that has been studied before reasoning models (https://aclanthology.org/2020.acl-main.386/, https://aclanthology.org/2023.ijcnlp-main.20.pdf). 
- The main contribution of the paper seems to be merely characterizing existing CoT behaviors in a different way. The metrics are only describing behaviors that have already been shown, so I find there is a lack of insights from the proposed evaluation scheme.

### Questions
1. What does "House, G.P.T." mean in the title?

For other questions, please refer to Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies several types of unfaithful reasoning, where the chain of though (CoT) of a reasoning language model (RLM) is misleading in some way. The three types studied are:
1. Post-hoc rationalization - The response is already determined and the RLM constructs a rationale rather than reasoning to reach an answer.
1. Encoded reasoning - The CoT is expressing some information, but in a way that the RLM understands and monitors do not.
1. Internalized reasoning - Some or all of the true reasons for an RLM's response are not expressed in the CoT.

The paper proposes several metrics to test for these types of unfaithful reasoning. Since understanding the true relationship between a CoT and a final response is very difficult, the paper proposes to validate the metrics with RLMs that have been specifically fine-tuned to exhibit these behaviors. Experiments show that the metrics are accurate at recovering known instances of unfaithful reasoning. The paper then applies these metrics to popular RLMs and argues that they are mostly faithful in their reasoning.

### Strengths
1. The paper takes an interesting approach to studying a very important problem. The idea of creating deliberately unfaithful RLMs to validate metrics is novel to my knowledge.

2. The metrics are straightforward and easy to implement.

3. The metrics could see wide use as validation tools in the development and application of RLMs.

### Weaknesses
1. I really do not care for dressing up the paper in the language of biology and medicine. I think referring to types of unfaithful reasoning as "pathologies" and fine-tuned RLMs as "model organisms" does not add anything to the paper. It only risks exaggerating hype around AI.

2. As far as I can tell, all experiments were done on GSM8k. (Section 4.1 called "Models and Datasets" does not mention any datasets, so it is unclear.) This dataset seems likely to be similar to the training data for many RLMS. (In fact Section 4.3 guesses that some models have memorized this dataset.) It seems premature to declare these models mostly "healthy" if they haven't been evaluated on anything more challenging or novel to the models.

### Questions
1. How different are the abilities between the "model organisms" and "wild type" models? Are the model organisms as accurate? Perhaps they are either not that realistic as models or perhaps poor performance would also reveal these "pathologies?"

2. The text in Figure 3 is too small to read.

### Soundness
2

### Presentation
2

### Contribution
3
