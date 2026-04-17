# AdaThink-Med: Medical Adaptive Thinking with Uncertainty-Guided Length Calibration

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Recent advances in inference-time scaling with extended long chain-of thought have significantly improved the reasoning capabilities of both general and medical large language models (LLMs). However, these models tend to engage in lengthy reasoning processes regardless of the difficulty of the input question, leading to increased inference costs in real-world applications. Therefore, enabling adaptive thinking where models think less for simpler questions and think more for complex ones is critical for the effective use of medical LLMs in practice. Despite its importance, there is a lack of end-to-end approaches designed to enhance the adaptive thinking capabilities of medical LLMs while providing a comprehensive examination of the trade-off between performance and computational cost. To bridge this gap, we propose **AdaThink-Med**, the first end-to-end framework designed to enhance adaptive thinking ability in medical reasoning models with uncertainty-guided length calibration. Specifically, we propose a specific entropy-based difficulty estimator and plug it into a difficulty-aware length reward inside reinforcement fine-tuning, targeting medical reasoning models, and show emergent non-thinking versus thinking modes plus dataset-selection benefits. AdaThink-Med first generates multiple candidate outputs for each question, evaluates the correctness and uncertainty of each candidate, and then estimates problem difficulty via an uncertainty-guided length calibration module. For outputs with low difficulty and correct answers, the framework penalizes longer reasoning paths; whereas for those with high difficulty and incorrect answers, it encourages extending the chain of thought to explore alternative solutions. On six public medical QA benchmarks, AdaThink-Med achieves up to **$6.4\times$** length reduction on average while retaining performance with only minimal degradation. Intriguingly, we observe that AdaThink-Med spontaneously develops two distinct reasoning modes, which we characterize as "non-thinking" and "thinking", demonstrating the model's ability to suppress redundant reasoning processes dynamically.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes AdaThink-Med, the first end-to-end adaptive thinking framework for medical large language models (LLMs), addressing the critical trade-off between reasoning performance and inference cost in clinical applications. Complemented by momentum-based smoothing and staged training, AdaThink-Med achieves up to 6.4× average length reduction across six public medical QA benchmarks with only minimal performance degradation. Key contributions include: (1) the first end-to-end adaptive framework for medical LLMs with autonomous "thinking/non-thinking" mode switching; (2) entropy-based uncertainty quantification for difficulty estimation; (3) a high-quality dataset selection strategy leveraging length distribution; (4) state-of-the-art efficiency-performance balance on medical benchmarks.

### Strengths
1. AdaThink-Med is the good end-to-end adaptive thinking framework tailored for medical LLMs, eliminating the need for manual difficulty labeling and enabling dynamic mode switching without separate classifiers. 
2. The experimental design is rigorous and comprehensive.
3. The work addresses a critical bottleneck for medical LLM deployment: balancing inference efficiency (cost, latency) and diagnostic accuracy.

### Weaknesses
1. The framework relies on entropy derived from model-generated outputs, but the paper does not fully validate its robustness to low-quality or biased candidate outputs.
2. The paper treats "problem difficulty" as a general construct but does not explore how medical domain characteristics (e.g., clinical ambiguity, multi-step differential diagnosis, rare diseases) affect difficulty estimation. 
3. While the paper compares against many baselines, it lacks direct comparison with state-of-the-art medical LLMs optimized for efficiency.
4. Modern medical AI often requires integrating text (EHRs) and imaging (X-rays, MRIs), but AdaThink-Med is limited to text-based QA. 
5. While the performance compensation mechanism addresses greedy length reduction, the paper does not explain how it handles edge cases in medical QA

### Questions
1. The top-K token selection in entropy calculation (Section 3.1.1) uses an unspecified K (e.g., top 20%). What is the rationale for choosing K, and have you conducted a sensitivity analysis on K across different medical benchmarks.
2. In staged training, how is the convergence of the first stage (standard GRPO training without length calibration) defined? Does the choice of convergence criterion (e.g., validation accuracy plateau, step count) impact the performance of the second-stage adaptive calibration?
3. The paper observes emergent "thinking/non-thinking" modes—have you analyzed whether these modes correlate with specific medical task types (e.g., basic medical fact QA vs. complex prognosis prediction)?
4. The dataset selection strategy achieves better performance with 40% of the data. What characteristics of the retained 40% data (e.g., difficulty distribution, question type) drive this improvement? Is this finding generalizable to other medical datasets beyond AlphaMed19k?
5. How does AdaThink-Med handle medical questions with high inherent uncertainty (e.g., "What is the most likely diagnosis for a patient with non-specific symptoms") versus questions with low uncertainty but rare answers? 
6. Have you evaluated the framework’s performance on subpopulations (e.g., questions about pediatric vs. adult medicine, common vs. rare diseases)? Medical LLMs often exhibit bias in subpopulation performance—does AdaThink-Med’s adaptive strategy mitigate or exacerbate such biases?

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
4

### Summary
This paper proposes AdaThink-Med, a framework for medical large language models (LLMs) that enhances reasoning efficiency via uncertainty-guided adaptive length calibration. The key innovation is to use output entropy as an uncertainty measure to dynamically estimate the difficulty of each question, leveraging both correctness and uncertainty to guide the reasoning process. The method adaptively adjusts the reward for reasoning length during reinforcement learning: promoting concise reasoning for easy questions and allowing more extensive reasoning steps for difficult ones. Extensive experiments on multiple medical QA datasets demonstrate that AdaThink-Med significantly reduces token consumption without sacrificing accuracy, and in some cases improves reasoning performance.

### Strengths
1. The proposed uncertainty-guided adaptive length calibration is clearly formulated and technically well-motivated. The mechanism for estimating problem difficulty and dynamically adjusting output length is systematically constructed.
2. The work provides comprehensive empirical analysis, including comparisons to multiple baselines and ablation studies, offering evidence for the effectiveness and efficiency of the approach within the evaluated scope.

### Weaknesses
1. Although the experiments were conducted in the medical LLM domain, the proposed uncertainty-guided length calibration appears to be a generic method for reasoning efficiency, with limited explicit adaptation to medical-specific characteristics. It is unclear if the approach addresses any challenges unique to medical reasoning, or if it simply applies a general technique to medical datasets.
2. The paper demonstrates empirical improvements, but lacks discussion of the stability and reliability of the training procedure, especially with respect to reinforcement learning and the reward mechanism. There is insufficient analysis of potential failure cases or model sensitivity to hyperparameter choices and data distribution shifts.
3. The proposed framework relies on model-based uncertainty scores to define question difficulty, which may not align with human or domain-expert judgments. If there is substantial divergence between what the model perceives as "difficult" and what a human user finds challenging, the adaptive reasoning length strategy could become suboptimal or even counterproductive. The paper does not address how such misalignment could affect real-world applicability, nor suggests strategies to mitigate possible negative impacts when model and user difficulty perceptions differ.
4. The method is evaluated exclusively on medical QA datasets, but the technical design does not incorporate any medical-specific characteristics. This leads to a contradiction—whether the method is intended as a general reasoning efficiency approach or as a domain-specific optimization for medical LLMs. If domain-specific optimization is claimed, the manuscript should clarify which aspects of the approach are tailored to medical reasoning tasks.
5. All results are based on automatic metrics and benchmark datasets, with no involvement from medical professionals (e.g., clinicians, medical students) for qualitative or quantitative assessment. This lack of user-centered evaluation makes it difficult to ascertain the practical impact and real-world effectiveness of the method.
6. The analysis focuses primarily on efficiency and accuracy, but does not address potential side effects, such as generating unnecessarily verbose outputs for hard questions, or reducing token usage while missing important reasoning steps. A deeper exploration of these issues would help provide a more balanced evaluation of the approach.

### Questions
As indicated in the weaknesses section.

### Soundness
3

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
4

### Summary
This paper proposes AdaThink‑Med, an end‑to‑end reinforcement‑learning framework that teaches medical LLMs to adapt the amount of reasoning to the difficulty of a question. The key idea is to estimate question difficulty with uncertainty, then shape a length reward that (i) penalizes unnecessarily long chains when the model is confidently correct on easy items and (ii) compensates by encouraging longer chains when the model is wrong on hard items.

1. Uncertainty is computed as the mean token entropy over the top‑K most uncertain tokens in a rollout, then min‑max normalized within the batch. Multiple rollouts per question are used. 
2. These uncertainties discount a correctness‑based estimate of difficulty (D_q): correct-but‑uncertain generations make the item harder. 
3. A difficulty‑aware length rewarduses a batch quantile threshold ( \theta_B ) to split easy vs. hard questions and applies asymmetric shaping: penalize long outputs for easy+correct cases; encourage longer outputs for hardincorrect cases. Final reward mixes accuracy, format, and length terms. 
4. Staged training: first learn to reason via GRPO without length calibration, then enable adaptive length calibration with EMA‑smoothed batch statistics. 

Results. On six medical QA benchmarks (MedQA, MedMCQA, PubMedQA, MMLU‑ProM, GPQA‑M, MedXpert), AdaThink‑Med reduces output length dramatically with minimal accuracy loss, achieving state‑of‑the‑art Accuracy‑Efficiency Score (AES) among length‑calibration methods. For example, with LLaMA‑3.1‑8B backbones, average length drops from 410 to 64 tokens  with only ~1.1% average accuracy decrease. Similar gains hold for Qwen‑2.5‑7B. The method also reports:

1. "Non‑thinking" vs "thinking" modes emerging spontaneously (short direct answers vs. concise reasoning). 
2. Reward‑hack analysis: greedy length penalties can collapse both length and accuracy; the proposed performance compensation stabilizes training. 
3. Ablations on the difficulty threshold ( \tau ) and uncertainty weight ( \alpha ), and a staged‑training study. 
4. A dataset‑selection use‑case: selecting subsets by model‑induced output lengths; 40% subsets sometimes surpass full‑set training.

### Strengths
1. Principled difficulty signal: Combines correctness with trajectory‑level entropy to reflect confident‑correct vs. uncertain‑correct generations. 
2. Asymmetric length shaping with performance compensation demonstrably prevents collapse seen in greedy penalties. 
3. Substantial efficiency gains with small accuracy changes across six medical QA tasks and two backbones: e.g., LLaMA average 64 tokens vs 410 baseline; Qwen 106 vs 497. 
4. Beats strong length‑calibration baselines in AES on both backbones. 
5. Emergent "non‑thinking / thinking" modes without extra supervision. 
6. Generalization to a different reasoning setup (HuatuoGPT‑o1) with sizable length reduction and small average accuracy decrease.

### Weaknesses
1. AES sensitivity & cost realism. AES uses fixed weights ( (\alpha,\beta,\gamma)=(1,3,5) ) that strongly favor length reductions (Table 3 even shows ( \tau=0.9 ) yielding extremely short outputs with the highest AES despite accuracy drop). No sensitivity analysis is provided; wall‑clock latency and decoder FLOPs are not reported. Token counts alone can misrepresent real cost (e.g., different backbones/tokenizers, batching, KV‑cache). 
2. Difficulty estimation depends on batch min–max and quantile thresholding. While EMA smoothing is used, batch‑wise normalization can introduce instability or dependence on batch composition; an ablation on batch size / normalization would help. 
3. Inference‑time mechanism is implicit. The text says AdaThink‑Med "estimates the difficulty of incoming questions" at inference, but there is no explicit gating; the policy implicitly emits shorter/longer chains. Make this explicit and clarify whether any multi‑sample inference is needed (it appears not, but this is important for deployability). 
4. Dataset‑selection experiment lacks controls. The subset selection is driven by model‑induced output length (global median split), but there is no comparison to random or difficulty‑agnostic subsets at equal sizes; thus, it is unclear whether improvements over the full set arise from better data or regularization via less data. 
5. Evaluation scope. All benchmarks are QA; clinical free‑text and multi‑step decision support are not assessed. Also, accuracy extraction in the open‑ended RL setup (no options) needs more detail—how "final answer" is parsed/validated across datasets. 
6. Minor inconsistency. In "Comparison with length‑calibration methods," the text claims AdaThink reduces lengths "6.4x ... than Kimi 1.5," but Table 2 shows 64 vs 125 tokens  on LLaMA; the 6.4x ratio matches GRPO (410/64), not Kimi1.5. Please correct. 
7. Tokenization comparability. Length comparisons mix Qwen and LLaMA tokenizers and multiple models. While AES baselines are set per‑backbone in Table 2, cross‑model averages in Table 1 may still reflect tokenizer effects. Clarify whether lengths are normalized or reported per‑model tokenizer.

### Questions
1. AES sensitivity / latency. How robust are conclusions to different AES weights? Please report wall‑clock decode latency and GPU‑FLOPs/token (or throughput) to corroborate the token‑length savings. 
2. Inference‑time behavior. Does inference use a single sample per question with a learned stopping behavior, or any explicit difficulty predictor / early‑exit? If purely implicit, can you add a plot of length vs. estimated difficulty without multi‑sampling at test time? 
3. Accuracy parsing in open‑ended RL. How is the final answer extracted when options are not presented? Is there a verifier or regex per dataset? Please detail the format reward and answer‑matching logic used in training and evaluation. 
4. Batch normalization choices. How sensitive is performance to the min–max normalization and the quantile ( \tau ) under different batch sizes? Could per‑sample entropy calibration (e.g., running averages) remove batch coupling? 
5. Top‑K entropy. What is (K) (or the fraction) used? Please include an ablation on K (and temperature (T)) to show robustness of the uncertainty estimator. 
6. Dataset selection controls. Can you compare to random subsets and loss‑based / gradient‑based selection at the same sizes? Also specify the compute budget and whether training steps were matched across subset sizes. 
7. Cross‑tokenizer fairness. For Table 1 averages, did you normalize lengths across tokenizers (e.g., bytes or characters) or compute per‑model averages only? A clarification would help interpret cross‑model length gaps. 
8. Clinical realism. Any preliminary results on free‑text clinical tasks (e.g., short case vignettes) to test whether the "non‑thinking / thinking" mode switch remains beneficial beyond multiple‑choice QA? 
9. Since mainstream SOTA LLMs do not support token precision thinking effort controling, I am unaware of the importance of doing budget controllon for madical domain. Please ellaborate this.

### Soundness
3

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
2

### Summary
The authors propose an end-to-end framework, AdaThink-Med, for curbing response length of medical LLMs without compromising accuracy. Roughly speaking, the framework encourages model to think less on "easier" questions and think longer on "hard" questions. Uncertainty over questions (measured via entropy) is used to shape the length reward during training and help determine reasoning effort (and therefore, length) at inference. The approach is tested over several medical benchmarks where AdaThink-Med scores the highest on the AES (Accuracy-Efficiency Score) metric.

### Strengths
- Strong motivation. Indeed, reasoning length is a severe disadvantage of thinking models. Developing a training paradigm for length calibrations based on entropy over candidate outputs in the medical domain appears to be novel. 

- The approach achieves the best AES according to chosen hyper-parameters. 

- 6.4x length reduction against baseline, scaling from hundreds of tokens to a few dozen. 

- The emergent "no thinking" versus "thinking" mode as a result of length-calibration training is an interesting observation that could help influence similar training strategies.

### Weaknesses
- Perhaps the most concerning, the accuracy drop over several benchmarks are too large to justify decreased output length. Decreased performance by over 6 points across several benchmarks. If the authors were reducing responses from several thousand to only a few dozen, this drop off could be justified, but it's often from a few hundred tokens to just a few dozen. 

- The entropy estimate is sensitive to choice of LLM. While acknowledged by the authors, it is unclear how deriving responses from mixtures or LLMs or a wider breadth of LLMs would affect this strategy. 

- The AES seems very sensitive to choice of hyper-parameters, rendering the superiority of AdaThink-Med along this metric less interpretable. 

- While "excessive length" is subjective, most competitors in Table 1 are generating only a few hundred tokens. Compression strategies tend to be concerned with lengths numbering in the tens of thousands, which is common for benchmarks such as MATH500 and AIME24 (the types of benchmarks for which it can be argued that thinking models were developed for).

### Questions
- See weaknesses.
 - Does this approach generalize to larger models? 
 - Does this approach generalize to newer models? (i.e., Qwen-3) 
 - What are the closed source baseline results? (i.e., GPT-5, Gemini-2.5-Pro, etc.)

### Soundness
3

### Presentation
2

### Contribution
1
