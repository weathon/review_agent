# SWE-RM: Execution-free Feedback for Software Engineering Agents

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
Execution-based feedback like unit testing is widely used in the development of coding agents through test-time scaling (TTS) and reinforcement learning (RL). This paradigm requires scalable and reliable collection of unit test cases to provide accurate feedback, and the resulting feedback is often sparse and cannot effectively distinguish between trajectories that are both successful or both unsuccessful. In contrast, execution-free feedback from reward models can provide more fine-grained signals without depending on unit test cases. Despite this potential, execution-free feedback for realistic software engineering (SWE) agents remains underexplored. Aiming to develop versatile reward models that are effective across TTS and RL, however, we observe that two verifiers with nearly identical TTS performance can nevertheless yield very different results in RL. Intuitively, TTS primarily reflects the model’s ability to select the best trajectory, but this ability does not necessarily generalize to RL. To address this limitation, we identify two additional aspects that are crucial for RL training: classification accuracy and calibration. We then conduct comprehensive controlled experiments to investigate how to train a robust reward model that performs well across these metrics. In particular, we analyze the impact of various factors such as training data scale, policy mixtures, and data source composition.
Guided by these investigations, we introduce SWE-RM, an accurate and robust reward model adopting a mixture-of-experts architecture with 30B total parameters and 3B activated during inference. SWE-RM substantially improves SWE agents on both TTS and RL performance. For example, it increases the accuracy of Qwen3-Coder-Flash from 51.6% to 62.0%, and Qwen3-Coder-Max from 67.0% to 74.6% on SWE-Bench Verified using TTS, achieving new state-of-the-art performance among open-source models. On RL training, SWE-RM lifts the resolve rate of execution-based counterparts by 3 absolute points on SWE-Bench Verified.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents SWE-RM, an execution-free reward model for software engineering agents that replaces unit-test feedback with model-based scoring. By introducing AUC and calibration metrics beyond test-time scaling, it achieves SOTA results and improves RL performance on SWE-Bench Verified.

### Strengths
1.	The paper addresses an important and emerging challenge in LLM based software engineering that how to design effective reward signals for coding agents when execution-based feedback is unreliable or unavailable. 
2.	The authors provide a convincing empirical motivation showing that similar TTS scores can lead to drastically different RL outcomes. 
3.	Section 4 includes a broad set of ablations on training data scale, data ratio, policy mixture, and source composition. These controlled studies give a practical recipe for training more robust reward models.
4.	The proposed SWE-RM achieves consistent performance gains on SWE-Bench Verified, improving both test-time scaling results and downstream RL outcomes.

### Weaknesses
1.	The paper proposes that three metrics TTS, AUC, and ECE jointly define a “versatile” reward model, but offers no theoretical justification or analytical link to reinforcement learning dynamics. The framework remains purely empirical, leaving the causal relationship between these metrics and RL stability unexplained.
2.	The model adopts a large 30B-parameter MoE with 3B active experts, but the rationale behind this choice is missing. There is no ablation comparing MoE against dense or adapter-based architectures, nor any argument explaining why modularization benefits calibration or discrimination.
3.	The paper vaguely describes the reward model as predicting a special token (“YES/NO”) and mapping it to a continuous score. However, it never specifies the precise loss function. This omission makes it impossible to reproduce or understand the optimization process.
4.	Although the method claims to produce execution-free feedback, the training labels are derived from fail2pass execution tests. This dependency contradicts the notion of being execution-free and blurs the boundary between model-based and execution-based supervision.
5.	Section 4.3 shows performance improvements when increasing context length from 32k to 256k tokens, but provides no explanation of why this happens. The analysis ignores computational costs, latency trade-offs, and whether the gain comes from longer reasoning chains or merely from covering more tokens.
6.	SWE-RM employs a Mixture-of-Experts (30B total, 3B active) architecture, but the authors do not compare it with alternative designs. Nor do they explain why this structure improves reward calibration. The absence of such analysis weakens the methodological contribution.
7.	The paper simultaneously uses TTS, AUC, and ECE as evaluation metrics but never establishes their quantitative relationships or complementarity. For example, AUC and TTS both measure ranking ability; it is unclear why optimizing all three leads to better RL stability. The metric design does not appear to be theoretically motivated.
8.	The authors acknowledge that some trajectory labels are derived from fail2pass test results that may not reflect true correctness, yet they do not provide any estimate of label noise, data cleaning strategy, or robustness analysis.
9.	Despite claiming a 30B-parameter model trained on 100k samples, the paper omits key details such as GPU count, training duration, or energy cost. The reproducibility statement is overly brief and fails to include crucial hyperparameters, limiting verifiability.

### Questions
Please address the weaknesses noted above.

### Soundness
2

### Presentation
2

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
This paper proposes SWE-RM, an execution-free reward model for SE agents that provides continuous feedback without requiring unit test execution. The authors argue that execution-based feedback has limitations including sparsity and dependence on test quality, and propose using a learned reward model instead. They identify three key metrics for evaluating reward models: test-time scaling (TTS) performance, classification accuracy (AUC), and calibration (ECE). Through empirical ablations on training data scale, positive/negative ratios, policy mixtures, and data sources, they develop a 30B parameter mixture-of-experts model that achieves state-of-the-art results on SWE-Bench Verified for both test-time scaling and reinforcement learning.

### Strengths
The paper makes a reasonable observation that test-time scaling performance alone is insufficient for evaluating reward models intended for RL use. The empirical finding in Figure 2 showing that two verifiers with similar TTS can have drastically different RL performance is interesting and motivates the need for better evaluation criteria.
The reported improvements are good - lifting Qwen3-Coder-Flash and Max on SWE-Bench Verified represents meaningful progress.
Scaling the reward model to 256k context length addresses a practical limitation of prior work and enables scoring of complex, long trajectories.

### Weaknesses
The paper does not adequately address a critical limitation: verifying program correctness can be as difficult as, or even harder than, executing the program. A reward model must essentially predict execution outcomes across potentially many different code paths and edge cases without actually running the code. This requires the model to simulate program semantics, which is fundamentally challenging and may introduce systematic errors. The paper does not discuss: How the reward model handles corner cases or edge conditions that unit tests would catch. The types of errors the reward model systematically misses (false negatives) or incorrectly flags (false positives). Whether execution-free verification can ever be truly reliable for complex software engineering tasks. Why we should expect a learned model to generalize to unseen code patterns and bugs.
This represents a significant gap in the theoretical justification for the entire approach. The claim that execution-free feedback is superior needs much stronger grounding.

While the paper presents extensive ablations, the intuitions behind many design decisions are unclear:
Why is a 2:1 positive/negative ratio optimal? Is this simply an artifact of data availability, or is there a principled reason?
Why do mixed policies help? What specific distribution shift does this address?
The paper states "we believe reward model can still achieve relative good performance by training on a large number of data" (Appendix C.1) when discussing noisy labels, but provides no theoretical or empirical justification for this belief
Why are AUC and ECE the right metrics? While intuitively reasonable, there's no formal argument for why these metrics should predict RL performance

The core contribution appears to be training a larger reward model on more diverse data and identifying AUC/calibration as important metrics. The approach of training a classifier on successful/unsuccessful trajectories and using it for ranking is standard. The paper does not introduce novel algorithms, architectures (beyond using MoE), or training objectives. The identification of additional evaluation metrics, while useful, is incremental rather than constituting a significant methodological advance.

The reward model is trained on trajectories labeled by execution-based feedback, then claimed to be superior to execution-based feedback. This creates a circular dependency - if the execution-based labels are noisy (as the paper argues), then the reward model learns to approximate noisy signals. The paper acknowledges this ("some of them might be noisy") but does not rigorously analyze how noise in training labels affects model quality or how the model could possibly exceed the quality of its training signal.

### Questions
Please refer to questions raised in weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SWE-RM, a large-scale, execution-free reward model for SWE agents, and shows it is effective both for test-time scaling (TTS) and as a reward signal in RL. Execution-based verifiers rely on incomplete or noisy unit tests, producing sparse feedback, while execution-free models provide denser, more informative signals. The authors argue that TTS performance alone is insufficient for evaluating verifier quality and introduce AUC (discriminative ability) and ECE (calibration error) as complementary metrics. Extensive ablations explore the effects of dataset size, composition, positive/negative ratios, and policy mixtures. SWE-RM achieves strong TTS gains on SWE-Bench Verified and modest RL improvements when combined with execution-based rewards.

### Strengths
- Showing that identical TTS performance can yield divergent RL outcomes is an important empirical finding with implications for verifier evaluation and selection.
- The ablations on data scale, source composition, and context length provide concrete, actionable insights for building robust SWE reward models.
- The model supports very long (256k) contexts so the verifier can score large numbers of trajectories.
- A single verifier that supports both evaluation-time reranking and RL training is practically useful.

### Weaknesses
- The narrative sometimes shifts between TTS (used for evaluation/selection) and the execution-free RM (the actual trained model), which can make the pipeline slightly harder to follow. Being explicit about where TTS stops and supervised RM training begins would help.
- Because the paper starts from TTS limitations, it would help to spell out the actual supervised reward-model objective earlier so readers don’t assume TTS is the training signal.
- Evaluation is limited to SWE-Bench Verified, leaving generalization to other SWE tasks or domains untested.
- The model's scale and context requirements may constrain practical deployment.

### Questions
- The authors show cases where similar TTS but better AUC/ECE give better RL. Is there evidence that improving calibration directly improves RL, or guidance on thresholds?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes to train an execution-free reward model (RM) to address the reliance on the real-world execution feedback for training software engineering (SWE) agents, which is often sparse, unreliable, and hard to scale. The paper's core insight is athat good performance on test-time scaling (TTS), a common metric for verifiers, does not gaurantee good performance when the verifier is used as a reward model for RL. The authors identify 2 crucial properties for a robust RM: discriminative ability (measured by AUC) and confidence reliability (measured by calibration, or ECE). Based on this insight the authors conduct comprehensive experiments to determine how to train a robust RM, by investigating factors like training data sclae, positive-to-negative data ratios, data source etc. The authors train SWE-RM a 30B MoE model, starting with Qwen3-30B-A3B, which achieves solid improvement of pass@1 on SWE-Bench Verified.

### Strengths
- The paper's primary strength is its core insight, which is clearly motivated and empirically demonstrated. The finding that test-time scaling (TTS) performance is an insufficient proxy for a reward model's utility in RL  is interesting. 
- The experiments are well-executed, a series of ablation is done on data scale, data composition, policy mixture and context length. This provides good guidance and insight for the community to identify the source of gain.
- The writing is exceptionally well-written, logically structured, and easy to follow. The paper also presents strong empirical results when the RM is used as a verifier for TTS and as additional source of signal during RL.

### Weaknesses
- Opacity of the "Poor Calibrated RM" (Verifier B): The entire paper's motivation hinges on the comparison between "Verifier A (Good AUC & Cali.)" and "Verifier B (Bad AUC & Cali.)". We are shown they have similar TTS but different RL outcomes. However, the paper never explains how Verifier B was trained or why it has bad calibration. Is it off-the-shelf (together with verifier A) or trained by the authors (if so could authors explain how to train it)? Is it the same as "poorly calibrated RM" in Figure 7? 
- There's a slight tension in the paper's premise. The RM is trained on positive/negative labels generated by "execution results with the provided fail2pass test". However, the paper's introduction effectively argues that these very tests are unreliable, sparse, and "in some cases, entirely unrelated to the target issue" . The paper "believe reward model can still achieve relative good performance by training on a large number of data", which is a reasonable assumption (i.e., the model learns a denoised, generalized signal). Still, this is a key point: the RM is trained on the same (flawed) signal it is meant to improve upon. A deeper discussion of this "denoising" effect would be welcome. And the issue of noise is also coupled with the Figure 7 result of execution-only run, where the authors attributes its performance drops to "issues with test noise and coverage".

### Questions
- What dataset Figure 4 reports on? I would love to know more details about how the authors compute the ECE, e.g. on which dataset, how many rollouts, sampling config, etc.
- How were the base reward constants (+1.0 for resolve, -0.5 for unfinished, 0.0 for otherwise) chosen? Did you experiment with other values, and how sensitive is the final RL performance (Figure 7) to these specific hyperparameters?
- The data scaling experiment (Figure 6) and the data composition experiments (Tables 1 & 3)  are presented separately. For clarity, was the 100k data point in the scaling experiment trained using the optimal 2:1 ratio and policy/source mix found later? And conversely, were the composition experiments in Tables 1 & 3 conducted at the full 100k data scale, or a smaller one? I'm trying to understand the interplay between data quantity and quality/composition.

For the rest, please see above for the weakness part.

### Soundness
3

### Presentation
4

### Contribution
3
