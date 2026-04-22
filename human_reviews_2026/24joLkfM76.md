# Boosting Process-Correct CoT Reasoning by Modeling Solvability of Multiple-Choice QA

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Reasoning quality in large language models depends not only on producing correct answers but also on generating valid intermediate steps. We study this through multiple-choice question answering (MCQA), which provides a controlled setting with fixed answer options. Our analysis shows that when questions are effectively unsolvable for a model, spurious chains of thought (CoTs) are more likely to appear, leading to false positives. By estimating the solvability of each question, we uncover an intermediate regime where learning is most effective. Building on this insight, we adapt outcome-supervised reward models and reinforcement learning with group-relative advantage to incorporate solvability into their objectives. Across experiments on math and multimodal datasets, these modifications consistently yield higher rates of process-correct reasoning and, in reinforcement learning, improved answer accuracy as well. Our results highlight solvability as a key factor for reducing hallucinations and increasing reliability in CoT reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes to improve process-correctness in CoT reasoning by adapting outcome-based reward models and re-weighting group relative advantages during RL training. Using better than random performance as a heuristic for solvability, the authors derive a probability that a question is solvable for a particular model based on its observed group-wide performance. Under the assumption that outcome-correct generations for "unsolvable" problems are process-incorrect, this probability can be used to adapt outcome-based reward models and group-relative advantages for solution selection and RL respectively. The results indicate small improvements from the two approaches relative to the most similar baselines (standard ORM and Dr-GRPO) across two Llama3 models on the benchmarks considered.

### Strengths
The paper presents a novel and creative approach to incentivising process-correctness, which is an important problem. The results indicate that the approach has promise, despite being insufficient to be confident of its utility (see Weaknesses).

### Weaknesses
The paper needs considerable re-writing, as it is currently quite confusing to read. The limited introduction and unconventional following structure (method, background, more methods) makes it difficult to follow and prevents the paper from having a clear thread of motivation, approach and evidence.

The assumption that multiple choice problems with better than random performance are "solvable" and others are not seems restrictive and the connection between this assumption and true process-correctness is insufficiently evidenced. The results show modest gains over for example the typical Dr-GRPO advantage, but these alone, spanning only a limited number of benchmarks and two models from a single model family are insufficient to draw strong conclusions about the utility of p_solvable. 

Figure 1 (right) shows the potential limitation of p_solvable, specifically that for low # Answer-Correct CoTs, p_solvable trails the fraction of at least one process-correct CoT, indicating that opportunities for learning from a process-correct CoT are discarded. This could come with the added risk of limiting exploration, which should also be investigated.

### Questions
How does re-weighting the advantage by p_solvable affect exploration?

What is the rate of process-correct CoT's for "unsolvable" problems and how do you propose to not waste these informative samples?

### Soundness
3

### Presentation
1

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
This paper proposes modeling question solvability to mitigate the issue of spurious chain-of-thought reasoning in large language models, where answers may be correct but the reasoning process is flawed. The authors incorporate solvability probabilities into the reward model (MCQ-ORM) and the rl advantage function (MCQ-DrGRPO), thereby down-weighting chance-correct samples from unsolvable questions.

### Strengths
This paper further explains the value of medium-difficulty questions through the concept of Learning Potential and leverages solvability analysis to intuitively reveal the origins of false positive CoTs, providing strong interpretability and insights.

### Weaknesses
1. The proposed notion of Learning Potential is demonstrated on math and relatively simple multimodal QA tasks. How well does this metric generalize to other domains, such as commonsense reasoning, dialogue, or symbolic tasks, where difficulty and solvability may behave differently?

2. The current approach is restricted to multiple-choice QA. If extended to open-ended generation tasks (e.g., free-form QA or long-form reasoning), how would the solvability framework be adapted or redefined?

### Questions
see weaknesses

### Soundness
4

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
This paper proposes to incorporate solvability signal into chain-of-thought reasoning. It first provides a controlled experiment on multiple-choice QA and indicates that CoT for unsolvable questions are more likely to be spurious. Then it incorporates that solvability information into both model inference and reinforcement training, showing consistent improvements.

### Strengths
Improve the reliability of Chain-of-thought reasoning is important, therefore I believe the studied problem would be helpful for the community;
Intuitively I like this idea and I believe if executed well, this idea could work and generalize to different tasks and model families.

### Weaknesses
1. My biggest concern is that it only tests on Llama model, which is a non-reasoning model with limited reasoning capacity. To strengthen the findings, evaluating on other LLMs with different sizes and reasoning models is necessary, especially Qwen-series. 
2. Even on Llama, the observed improvements (Tables 2 and 3) are limited and in some cases not statistically significant. The authors should provide additional analyses such as qualitative examples, to illustrate where the proposed approach works well and where it fails.
3. In addition to RL, experiments based on SFT, such as rejection sampling, would also help provide a more comprehensive understanding of the proposed approach.

### Questions
whether and to what extent does your proposed reward signal robust to reward hacking?

### Soundness
2

### Presentation
3

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
This paper examines the correctness of processes in reasoning tasks. The core idea consists of three main steps. First, it defines the solvability of a multiple-choice question by estimating the probability p_solvable that a model can solve a problem, using a Beta posterior derived from multiple sampled accuracies, with a threshold above random guessing. Second, it incorporates p_solvable into an ORM as a soft label for positive examples, helping the model select the most process-correct CoT among multiple candidates at test time. Third, it integrates p_solvable into the advantage function of DrGRPO, forming MCQ-DrGRPO, which down-weights examples that are correct but likely have flawed reasoning processes. The authors further introduce a “learning potential” measure (LP =) novelty × solvability to explain which data points are most useful. Across mathematical and multimodal benchmarks, both MCQ-ORM and MCQ-DrGRPO achieve higher process correctness and also improve answer accuracy in reinforcement learning experiments.

### Strengths
* The authors formalize the empirical observation that “false-positive CoTs are more common on difficult problems” using the notion of solvability. The derivation is simple and reproducible.
* The method connects p_solvable to two common pipelines (e.g., test-time CoT selection and training-time RLVR) with minimal modifications and low computational overhead.
* The work covers three math MCQA datasets and two multimodal “geography/year guessing” datasets, reports both process and answer accuracy, and includes LLM-judge meta-evaluation and released outputs for reproducibility.
* MCQ-ORM consistently outperforms the standard ORM, while MCQ-DrGRPO outperforms DrGRPO across random seeds, with stronger gains in process accuracy (P-Acc).

### Weaknesses
* The paper assumes a random-guessing baseline 1/|choices| as the solvability threshold, which is reasonable for MCQs but not directly generalizable to open-ended generation tasks. A more detailed discussion on how the concept of solvability could be extended to general generative settings would strengthen the paper.
* In Section 4, the authors compare solvability-based ORM with the standard ORM and several confidence/faithfulness/CoPS metrics. However, in Section 5, the reinforcement learning experiments only compare DrGRPO variants. Including confidence- or uncertainty-based GRPO baselines would make the results more convincing.

### Questions
Currently, p_solvable is estimated from multiple sampled accuracies. Since open-source models can expose logits (and even many APIs provide top-n logits), could solvability be calculated directly from logit-based uncertainty measures (such as sequence entropy, logit margin, or temperature-calibrated confidence) to reduce sampling cost or improve estimation quality?

### Soundness
3

### Presentation
3

### Contribution
3
