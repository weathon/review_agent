# Merge-of-Thought Distillation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Efficient reasoning distillation for long chain-of-thought (CoT) models is increasingly constrained by the assumption of a single oracle teacher, despite the practical
availability of multiple candidate teachers and growing CoT corpora. We revisit
teacher selection and observe that different students have different “best teachers,”
and even for the same student, the best teacher can vary across datasets. Therefore, to unify multiple teachers’ reasoning abilities into a student to overcome conflicts among various teachers’ supervision, we propose Merge-of-Thought Distillation (MoT), a lightweight framework that alternates between teacher-specific
supervised fine-tuning branches and weight-space merging of the resulting student variants. On competition math benchmarks, using only about 200 CoT samples, applying MoT to a Qwen3-14B student surpasses strong models including
Deepseek-R1, Qwen3-32B, and OpenAI-O1, demonstrating substantial gains. Besides, MoT consistently outperforms the best single-teacher distillation, improves
general reasoning beyond mathematics while reducing catastrophic forgetting, and
shows robustness to distribution-shifted and peer-level teachers. Finally, we have
demonstrated MoT possesses consensus CoT by eliminating teacher-specific inductive biases and inter-teacher conflicts while repeatedly reinforcing the learning
of consensus reasoning features. These results position MoT as a simple, effective route to efficiently distilling long CoT capabilities from diverse teachers into
compact students.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes "Merge-of-Thought Distillation (MoT)", a framework that alternates between teacher-specific branch fine-tuning and weight-space merging to consolidate the chain-of-thought reasoning capabilities of multiple large language models into a single student model. The authors claim that with only about 200 samples, their method can surpass some larger models on mathematical reasoning benchmarks and mitigate catastrophic forgetting.

### Strengths
1) It provides a systematic study of multi-teacher long chain-of-thought distillation, addressing a gap in the existing literature which often focuses on single-teacher settings.
2) The experimental design is comprehensive, covering various model scales and datasets, and is supported by thorough ablation studies and theoretical analysis.

### Weaknesses
1) The core method of "branch training and weight averaging" is essentially a straightforward application of existing model merging techniques, lacking substantial algorithmic or structural innovation. The overarching idea of multi-teacher fusion is not a breakthrough in itself.
2) The paper identifies the critical problem of "teacher selection mattering" but does not address it; instead, the method circumvents it by merging all teachers. It lacks dynamic assessment of teacher quality or a weighted merging mechanism.
3) The core methodology is overly simplistic and does not incorporate more advanced fusion strategies. The experiments fail to adequately demonstrate its generalization capability to non-mathematical tasks, and the trade-off between computational cost and performance gain is not sufficiently discussed.
4) Some baseline results are taken from original papers without reproduction under a unified setup, potentially compromising fairness. The absence of comparisons with more state-of-the-art multi-teacher distillation or fusion methods weakens the persuasiveness of the claims.

### Questions
1) The paper compares MoT against naive multi-teacher data mixing (MTD) and one-shot weight merging. However, a more fair baseline would be an ensemble of the multiple independent teacher models, i.e., averaging their output probabilities at inference time. What are the significant advantages of MoT over such an output-level ensemble in terms of performance, efficiency, or stability?
2) Weight averaging is a very basic and sensitive merging technique. Did the authors experiment with more advanced merging algorithms, such as TIES-Merging or Task Arithmetic? If so, what were the results? If not, why was such a simple method chosen as the core innovative component?
3) The paper lacks comparisons with state-of-the-art methods specifically designed for multi-teacher or data fusion. For example, what are the distinct advantages and disadvantages of MoT compared to Gradient Blending (weighting teachers at the gradient level) or DARE (a pruning method before weight merging)?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the problem of distilling long chain-of-thought (CoT) reasoning from multiple teacher models. The authors start with the key observation that there is no single "best teacher" for all students or datasets. To address this, they propose Merge-of-Thought Distillation (MoT), a lightweight, iterative framework. In each round of MoT, the student model is branched out and finetuned separately on the CoT data from each individual teacher. These branches are then merged back into a single model via weight averaging. This merged model becomes the initialization for the next round. The authors show that this method is highly data-efficient, enabling a Qwen3-14B student trained on only ~200 CoT samples to surpass strong baselines like Deepseek-R1 and Qwen3-32B. The paper also provides analysis suggesting MoT mitigates catastrophic forgetting, is robust to distribution-shifted teachers, and creates a "consensus CoT" in a flatter loss landscape.

### Strengths
Strong Motivation & Problem: The paper is motivated by a clear, practical, and important problem. The initial analysis (Fig. 1, Table 1) showing that the optimal teacher varies by student and dataset is a strong justification for moving beyond single-teacher distillation.

Impressive Empirical Results: The performance gains are the primary strength of this paper. The claim that MoT with only 200 samples can elevate a 14B model to outperform a 32B model (Table 3) is a very strong and compelling result.

Data Efficiency: The ability to achieve these results with such a small, 200-sample dataset is highly significant and makes the method practical and accessible.

Thorough Analysis: The authors provide a comprehensive set of experiments, comparing MoT not only to the best single-teacher distillation (STD) but also to naive multi-teacher data mixing (MTD) and one-shot post-hoc merging (Table 2, Table 4). The additional analyses on catastrophic forgetting (Table 5) and the loss landscape (Appendix D) are valuable additions.

### Weaknesses
Despite the strong results, the paper's contribution seems to lie more in a successful application of existing techniques than in a novel methodological breakthrough, which makes it borderline.

- Limited Novelty: The core mechanism of MoT (iteratively training separate models on sharded data and averaging their weights) is conceptually almost identical to well-established algorithms like Federated Averaging (FedAvg) in federated learning. While the application (multi-teacher CoT distillation) is new, the method itself (iterative SFT + weight averaging) is not. The paper does not sufficiently differentiate MoT from this and other related model-merging literature.

- Underdeveloped Merging Strategy: The "Merge" in "Merge-of-Thought" is implemented as simple parameter averaging. Given the extensive and recent literature on more sophisticated model merging techniques (e.g., TIES, DARE, task arithmetic), restricting the method to simple averaging feels like a missed opportunity. It is unclear if the gains come from the iterative process alone, or if the merging step itself could be further optimized.

- Unclear Computational Cost: The paper emphasizes data efficiency (200 samples) and total "steps" (250). However, the MoT process involves training K (here, K=4) branches for 50 steps in each of 5 rounds. This totals (4 branches * 50 steps/branch * 5 rounds) = 1000 total branch-steps, which is 4x the computational cost of the 250-step STD/MTD baselines. This significant compute-for-data trade-off is not clearly discussed.

- Indirect Evidence for "Consensus CoT": The claim that MoT "eliminates teacher-specific inductive biases" is very strong. The evidence provided (Fig. 5) shows that MoT has lower confidence on stylistic tokens from one teacher. This is interesting but indirect; it could be interpreted as the averaging process simply "washing out" or "diluting" a specific teacher's signal, rather than intelligently "eliminating" its bias while preserving a core "consensus."

### Questions
1. Could the authors explicitly discuss the relationship between MoT and Federated Averaging (FedAvg)? How do they see the conceptual novelty of MoT in light of FedAvg, which also performs iterative, sharded training followed by weight averaging?

2. Why was simple parameter averaging chosen as the only merging strategy? Were more advanced merging techniques explored? Would they be compatible with the iterative MoT framework, and could they potentially offer further gains?

3. Could the authors provide a clear comparison of the total computational cost (e.g., total branch-steps or GPU-hours) for MoT versus the STD and MTD baselines? This would clarify the compute/data/performance trade-off.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Merge-of-Thought Distillation (MoT), a novel framework for distilling long chain-of-thought (CoT) reasoning capabilities from multiple, potentially heterogeneous, teacher LLMs into a single, smaller student model. The core motivation is that there is no single "best teacher" for a given student and dataset, and naive methods like mixing teacher data or one-shot model merging fail to reconcile conflicting supervision signals. MoT addresses this by iteratively alternating between two steps: (1) Teacher-Specific SFT (Branch Training), where the student is fine-tuned on each teacher's reasoning paths, and (2) Weight-Space Merging, where the parameters of these specialized branches are averaged. This process is designed to reinforce consensus reasoning features while suppressing teacher-specific noise and inductive

### Strengths
While model merging and CoT distillation are active areas of research, their combination in an iterative, multi-teacher co-distillation framework is novel. 
The paper is generally well-written and structured.

### Weaknesses
The premise of the paper, "different students have different best teachers," is only derived from two math evaluation datasets. An insufficient number of evaluation datasets may lead to biased conclusions.
The core merging operation is a simple uniform average of parameters. While the results show this is effective, the field of model merging has advanced with more sophisticated techniques. The paper would be strengthened by a direct comparison against a more advanced merging baseline to confirm that the gains are from the iterative process and not just a more clever one-shot merge.

### Questions
In evaluation sets outside of mathematical reasoning, does the phenomenon of "Teacher choices are not universal" still exist?
Is the number of teacher models the more, the better?
How does the model perform on the HMMT benchmark?
How were BOBA-200 and S1k-200 sampled? Are there any guiding principles? For reasoning tasks [1], the quality of prompts obtained through random sampling varies significantly.
How does this method perform on other series of models, such as LLaMA?

[1] Reinforcement Learning for Reasoning in Large Language Models with One Training Example

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Merge-of-Thought Distillation (MoT), an iterative multi-teacher CoT 
distillation framework that alternates between (a) teacher-specific supervised fine-tuning 
branches and (b) weight-space averaging of the resulting student variants. The aim is to fuse 
complementary reasoning signals from heterogeneous teachers while suppressing teacher-
specific noise and conflicts. Empirically, the authors show that the “best” single teacher depends 
on both the student and the dataset, and that naïve multi-teacher data union or one-shot post-hoc 
merging does not reliably resolve cross-teacher conflicts. MoT is reported to outperform single-
teacher and naïve multi-teacher baselines on AIME24/25 using only ~200 CoT exemplars, with 
additional evidence of smaller catastrophic-forgetting drops and gains on non-math reasoning 
benchmarks.

### Strengths
1. The branch-train + weight-merge loop is easy to implement; it sidesteps brittle manual teacher selection.
2. Demonstrates competitive AIME improvements using only ~200 CoT samples, even versus larger models.
3. MoT not only strengthens general reasoning but also helps mitigate catastrophic forgetting compared to best STD

### Weaknesses
1. Experiments focus primarily on the Qwen family and math-centric CoT, with limited  coverage of other backbones or domains. This narrows external validity. The paper itself  notes comparisons primarily around Qwen bases.
2. Several baseline results are taken from papers with different setups, as the original  code/models are unavailable. This weakens the claim of direct superiority.
3. MoT requires training multiple branches across several rounds, making it more compute-intensive than single-teacher distillation, while often underperforming STD in  the first round before eventual gains appear.

### Questions
1. How sensitive are MoT gains to random seeds across both branch training and merging rounds, when using a fixed early-stopping rule rather than “best of” selection? Please report mean±std over multiple independent runs.
2. Have you tested LLaMA, Mixtral, or Mistral-based models? Are MoT gains specific to Qwen?
3. Have you tested MoT on other domains (e.g., code reasoning or scientific QA)? If not, what are the expected limitations?
4. How does MoT behave when one teacher is adversarial or noisy? Is the method robust to such imbalance?

### Soundness
2

### Presentation
3

### Contribution
3
