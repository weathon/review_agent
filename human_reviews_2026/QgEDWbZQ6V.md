# Scaling Policy Compliance Assessment in Language Models using Policy Reasoning Traces

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Policy compliance assessment is a fundamental task of evaluating whether an input case strictly complies with a set of human-defined rules, more generally known as *policies*. In practice, human experts follow a systematic, step-by-step process to identify violations with respect to specific stipulations outlined in the policy.  However, such documentation of gold-standard, expert-level reasoning processes is costly to acquire. In this paper, we introduce Policy Reasoning Traces (PRT), a form of specialized generated reasoning chains that serve as a *reasoning bridge* to improve an LLM's policy compliance assessment capabilities. Our empirical evaluations demonstrate that the use of PRTs for both inference-time and training-time scenarios significantly enhances the performance of open-weight and commercial models, setting a new state-of-the-art for HIPAA and GDPR policies. Beyond accuracy gains, we also highlight how PRTs can improve an LLM's ability to accurately cite policy clauses, as well as influence compliance decisions through their high utilization from the raw chains-of-thought.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of automated policy compliance assessment, a task that requires complex reasoning aligned with human-defined rules. The authors note that acquiring gold-standard expert reasoning traces for this task is prohibitively expensive. They propose a method called POLICY REASONING TRACES (PRT), which uses a powerful "expert" LLM to synthetically generate step-by-step reasoning traces, given only a case and its final compliance verdict (e.g., COMPLIANT/NONCOMPLIANT). These synthetically generated PRTs are then used as a "reasoning bridge"—either as few-shot in-context examples or as data for supervised fine-tuning—to improve the performance of a "learner" model. The authors demonstrate that this method improves accuracy and policy clause citation on the HIPAA and GDPR policy datasets.

### Strengths
1. Important Problem Domain: The paper tackles a practical and high-stakes problem. Automating policy compliance has clear real-world applications, and the challenge of aligning LLM reasoning with complex, human-authored policy documents is a significant research barrier.

2. Intuitive Core Idea: The central idea of using a more capable model to generate pseudo-expert reasoning, thereby avoiding the bottleneck of human expert annotation, is intuitive and a common-sense approach to data augmentation.

3. Strong Empirical Gains on Some Datasets: The method demonstrates clear and significant performance boosts over baseline models, particularly for open-weight models on the HIPAA dataset. The claim of achieving a new state-of-the-art on the GDPR dataset is also a notable empirical contribution.

### Weaknesses
1. Critical Unaddressed Risk: PRT Quality and Faithfulness: The entire method hinges on the assumption that the "expert" LLM generates high-quality, faithful reasoning traces. The paper provides no substantial validation of this. The expert model is given the gold verdict and asked to produce reasoning for it. This setup strongly encourages post-hoc rationalization, where the model may generate a plausible-sounding but incorrect reasoning path to justify the given answer. If the PRTs are flawed, unfaithful, or contain factual errors (e.g., wrong clause citations), the learner model is simply being trained on high-quality noise. This "garbage in, garbage out" risk is a fundamental methodological flaw that is not adequately addressed.

2. Questionable Novelty: The technique of using a strong model to generate synthetic data (especially reasoning steps or chains-of-thought) to train a weaker model is a well-established paradigm in NLP (e.g., data distillation, synthetic CoT generation). The paper fails to clearly articulate what makes "PRT" a novel method beyond applying this existing technique to the specific domain of policy compliance.

3. Confusing/Contradictory Results: The results on the ModelSpec dataset are concerning. The authors note that for models already optimized for this policy (OpenAI models), adding PRTs decreases performance. This finding, which is somewhat downplayed, severely undermines the method's generalizability. It suggests that these "expert-generated" PRTs may be of lower quality than the model's own internal reasoning, further supporting the concern in Weakness 1.

4. Superficial Analysis of PRT Sources: The paper generates PRTs from a "Generalist" (DeepSeek-R1) and "Specialist" (SAULLM) model but then states that the Generalist PRTs were "majority advantage" and used for subsequent experiments. This is counter-intuitive (why would a specialized legal model be worse?) and this finding is not explored. This feels like a missed opportunity for analysis and weakens the paper's contribution.

### Questions
1. PRT Quality Validation: Was any human evaluation conducted to measure the quality, faithfulness, and factual correctness of the generated PRTs? How do we know the "expert" model's reasoning for a verdict is the correct reasoning, and not just a plausible hallucination?

2. Forced Justification vs. Organic Reasoning: How does this method compare to simply prompting the expert model with standard Chain-of-Thought (e.g., "Let's think step-by-step") and having it generate both the reasoning and the verdict? By providing the gold verdict, aren't you forcing the model to justify a conclusion, which encourages rationalization over genuine reasoning?

3. ModelSpec Performance Drop: Could you elaborate on the performance degradation for OpenAI models on ModelSpec? Does this not imply that PRTs are only beneficial as a crutch for weaker models, and may actually be harmful (by introducing noise) when a model's existing reasoning capability is already strong?

4. Generalist vs. Specialist PRTs: Why do you hypothesize that the Generalist model's PRTs outperformed the Specialist (legal) model's PRTs? This seems to contradict the premise of using domain-specific expertise.

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
4

### Summary
This paper addresses the critical challenge of enabling LLMs to perform complex policy compliance assessments, a task that typically requires expensive, expert-human-generated reasoning data. The authors introduce PRT, a novel and scalable method for creating "pseudo-expert" reasoning chains. The method works by prompting a frontier "expert" LLM with a case, a policy, and its gold-standard verdict to generate a detailed reasoning trace. These PRTs are then used as a "reasoning bridge" to improve "learner" LLMs, either as few-shot demonstrations for in-context learning or as a dataset for SFT.
The authors evaluate this method on three diverse policies (HIPAA, GDPR, and OpenAI's ModelSpec) and demonstrate that it significantly enhances the performance of both open-weight and commercial models, setting new SoTA results on the HIPAA and GDPR benchmarks. The paper also provides deeper analysis showing that PRTs improve a model's ability to cite correct policy clauses and can enable strong cross-policy generalization.

### Strengths
1. The core contribution, using an expert LLM to generate post-hoc reasoning traces, is a highly practical and scalable solution to the data bottleneck problem in specialized domains like legal and policy compliance.
2. The method achieves SoTA performance on the HIPAA and GDPR benchmarks, notably beating the prior SoTA on HIPAA. The performance lift for open-weight models is particularly significant, with accuracy boosts of 50-100% bringing them much closer to commercial model performance.
3. The paper successfully demonstrates the value of PRTs in two distinct paradigms: as off-the-shelf few-shot examples for ICL and as a high-quality dataset for SFT.
4. The authors go beyond simple accuracy. The analysis showing that PRT-trained models are better at citing the correct policy clauses (Table 1) and the demonstration of strong cross-policy generalization (Figure 13)  add significant depth and credibility to the method's value.

### Weaknesses
1. The PRTs are generated by providing the expert model with the gold-standard verdict (Compliant/Non-Compliant) in the prompt. This means the generated PRT is a post-hoc rationalization of a known answer, not a genuine, "from-scratch" reasoning process to find the answer. This methodological choice may create unnaturally perfect or biased reasoning traces that do not reflect the difficulty of the original task.
2. The authors candidly report that their method hurts performance on OpenAI models (gpt-5-mini, gpt-oss) for the ModelSpec policy. While their "doubly-optimized" hypothesis is plausible, this is a significant failure case that demonstrates the method is not universally applicable and can be detrimental when a model is already highly aligned.
3. The entire method hinges on the quality of the "expert" LLM (deepseek-R1 and SaulLM). The paper acknowledges these PRTs are "imperfect weak supervision" and may contain hallucinations, but it does not deeply analyze the impact of this imperfection. If the expert LLM is systematically wrong or biased in its reasoning, the learner models will simply inherit and scale this flawed logic.

### Questions
1. The paper's "internal" compliance approach should be contrasted with "external" guardian models. Could the authors discuss the trade-offs (e.g., latency, cost, and robustness) of their method versus using a dedicated, external gurad like LlamaGuard, WildGuard[1][2], which provides runtime moderation and recovery feedback?
2. Given the SoTA claims, would the authors consider evaluating their PRT-finetuned models on the new, large-scale DynaBench [3]? This would provide a more direct comparison to other SoTA models designed for similar general-purpose, dynamic policy enforcement.
3. Regarding the PRT generation: What was the rationale for providing the gold-standard verdict to the expert model? Have the authors experimented with generating PRTs without this "answer key"? How does the quality and correctness of the expert's reasoning change in that more realistic scenario?
4. Can the authors elaborate on the "doubly-optimized" failure case? Is it possible this is a style conflict (e.g., the deepseek-R1 reasoning style conflicts with the internal alignment of GPT models) rather than just "overthinking"? This finding seems critical to understanding the method's limitations. 


[1] https://arxiv.org/pdf/2312.06674

[2] https://arxiv.org/abs/2406.18495

[3] https://arxiv.org/abs/2509.02563

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes an approach for improving policy compliance checking via augmented training data with  synthetic data generation. For each rule, associated compliance examples with gold verdict labels, the proposed approach use an expert reasoning LLM to generate additional synthetic data. Then the generated synthetic data was used to finetune other LLMs for improving policy compliance reasoning.

The proposed method was validated against vanilla LLM without finetuning on the the generated policy reasoning traces in a few benchmark datasets. The proposed approach outperformed vanilla LLMs thanks to the robustness of the finetuning on synthetic data.

### Strengths
This is an interesting application paper where it demonstrates the use of synthetic reasoning traces in improving LLMs reasoning functions in a specific domain.

The paper was well written and it is easy to follow.

Some of the results are interesting to the application domain.

### Weaknesses
On the novelty and related baselines: There are a lot of similar approach not in the policy compliance but in general trying to create synthetic data from rules and examples to teach LLM reasoning:

+ Self‑Consistency Improves Chain‑of‑Thought Reasoning in Language Models (Wang et al., 2023) — ICLR 2023 poster/crux: https://openreview.net/forum?id=1PL1NIMMrw : generate many reasoning traces, not specific to one traces and select traces based on consistency, I think this method is more advanced since it does not only simply generate a reasoning trace but multiple ones then re-selecting based on  consistency

+ Enhancing Reasoning Capabilities of LLMs via Principled Synthetic Logic Corpus https://arxiv.org/html/2411.12498v1: as stated in the paper: " to enhance LLMs’ reasoning capabilities by program-generated logical reasoning samples. We first establish principles for designing high-quality samples by integrating symbolic logic theory and previous empirical insights." This should be a baseline to compare to

+ Enhancing Logical Reasoning in Large Language Models through Graph-based Synthetic Data https://arxiv.org/html/2409.12437v1: focus on improving LLMs using synthetic graph-based reasoning data

 Given that the ideas of generating synthetic data to augment the training data is not novel, the technical contribution for the machine learning field is limited.

### Questions
Could you please compare your approaches to the mentioned related works I listed in the weaknesses?

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
4

### Summary
The paper explores the policy compliance assesments and improve LLM's policy compliance assesment capabilities by Policy Reasoning Traces, that are extracted from LLMs that can make reasonings such as DeepSeek-R1. The authors combines compliance, policy, and reasoning to create a  dataset that contains these triplets. Then, they use it to improve perforamances of LLMs which can be trained using, SFT or ICL. The goal is to automate the process of checking compliances whether suits for the given policies by LLMs. It is expected to apply stipulations

### Strengths
- The paper touches on an important problem and provides an applicable solution.

- The proposed methodology improves the performance of open-weight and commercial models.

### Weaknesses
- The provided strategy is not novel, since the paper uses prompting strategies that can also be applicable to the other methods in the literature. The most novel part about the proposed methodology is the use of PTR, which sometimes decreases the performance of the models in SelfRef or doesn't make a change in Few-shot learning

- The experiments contain less training and test data, e.g., 309 and 107, which may cause models to overfit if they are SFTed. The number is even lower in the third dataset.

- Does PTR contain the answer to the compliance? If so, the trained models would only learn to extract the answer that passes inside the PTR.

- Compliance statements and policy documents are typically long inputs for most open-weight LLMs, and adding multiple examples for few-shot prompting or incorporating self-feedback further increases the context length. Moreover, using PRTs for each example does not seem to be a feasible solution.

- SelfRef + PTR sometimes decreased the performance of some models e.g. , GPT5-Mini

### Questions
- What is the average length of the context for each strategy, few-shot learning, Few-shot learning + PTR, selfRef, SelfRef + PTR?

### Soundness
3

### Presentation
3

### Contribution
1
