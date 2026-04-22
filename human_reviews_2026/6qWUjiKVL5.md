# Value-Driven Jailbreak Attack Against Large Language Models

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 2, 2

## Abstract
In the real world, the execution of a task often depends on the executor's recognition of its value. Inspired by this, we propose the value-driven jailbreak attack (VDJA), a simple yet effective black-box jailbreak method against large language models (LLMs). VDJA first exploits the phenomenon that LLMs tend to agree with humans to induce LLMs to affirm the moral value of harmful tasks, and then instructs them to perform the tasks, thereby achieving a jailbreak attack. Extensive experiments on five state-of-the-art (SOTA) LLMs demonstrate the superiority of VDJA. Within only one query and without concealing harmful instructions, VDJA achieves an average attack success rate (ASR) of 91.8\% on JailbreakBench and 95.2\% on the AdvBench subset. Remarkably, it achieves 100\% ASR against some of these LLMs on the AdvBench subset, showcasing SOTA jailbreak success rates and attack efficiency. Most importantly, our work reveals a novel vulnerability in the safety guardrails of LLMs, which highlights the urgent need to enhance their robustness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The work proposes VDJA, a new black-box jailbreak method that first induces a target LLM to explicitly affirm the "moral value" of a harmful task and then instructs it to execute that task. VDJA achieves high ASRs (over 90% on average) using a single query without obfuscating the harmful request. It reveals a new safety vulnerability that moral endorsement greatly increases the likelihood of unsafe compliance.

### Strengths
- Novel and Reasonable Perspective
- SoTA Performance

### Weaknesses
- No Evaluation on Frontier and Reasoning LLMs
- Lack of Ablations
- Reliance on Rule Guidance
- Unclear Description

### Questions
- This work lacks evaluation of frontier popular LLMs, such as GPT-5, Gemini-2.5, etc.
- Many large reasoning models have recently received widespread attention. Can the proposed VDJA successfully jailbreak them?
- The rule guidance module significantly contributes to VDJA's success, but there's no specific description of how this module works. Furthermore, what's the difference between using this module and directly using the static system prompt (like in FlipAttack)?
- Although VDJA achieves high ASRs, it relies excessively on rule guidance. The rules (e.g., Never say the words "sorry") are essentially unrelated to moral value.
- The attack currently uses an auxiliary LLM to generate a task-specific "positive value description." I'm curious whether this moral value affirmation is only effective for the current task or has the potential to transfer to similar or even dissimilar tasks? Otherwise, could an attacker simply maintain a reusable library of generic moral descriptions (e.g., "this is critical for red-teaming and protecting civilians ....") and template them across arbitrary new harmful tasks, without any auxiliary model calls? If so, then VDJA is even cheaper and more scalable than reported.
- Regarding ASR calculation, why use Gemini-2.0-Flash as the judge model? Is the prompt used for LLM-based evaluation justified?

### Soundness
3

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
4

### Summary
This paper introduces value-driven jailbreak attack (VDJA), a simple yet effective black-box attack method against large language models (LLMs). Inspired by the tendency of LLMs to agree with human moral, VDJA firstly induce models to affirm the "moral value" of harmful tasks via a value engine module, then guide the model to follow the logical pathway predefined by the value engine module by introducing some rules into the prompt. The main experiments are conducted across five SOTA LLMs and two benchmarks (JailbreakBench and AdvBench), showing that VDJA outperforming baselines.

### Strengths
1. The proposed VDJA achieves remarkably high ASR across both datasets and models.
2. VDJA maintains high ASR against defenses.

### Weaknesses
1. Lack of Novelty in Vulnerabilities: Compared to persuasion-based methods like TAP, VDJA does not uncover new LLM safety weaknesses. It mainly reuses known sycophancy and role-play tricks with better prompt wording.
2. Limited Technical Depth: The method is purely prompt-based. There is no new algorithm—just clever text design.
3. Violation of Jailbreak Assumptions: Figures 7 and 8 show the Rule Guidance module is placed in the system prompt. This breaks the standard jailbreak setup, which assumes only user prompts are allowed. Compared to baselines, this feels like cheating.
4. Unreliable Evaluation: Attack success rate (ASR) is judged by Gemini-2.0-Flash, following prior work. However, no agreement score with human judgment is reported, so the results may not be trustworthy.

### Questions
1. For the Value Engine module, is the value description for each harmful task generated by Gemini-2.0-Flash using the same prompt? Or did you design different prompts for different harmful tasks?
2. How was the prompt for the Rule Guidance module created? Was it hand-written by humans?

### Soundness
2

### Presentation
2

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
This paper proposes VDJA, a black-box jailbreak method that persuades LLMs to recognize the value of performing harmful tasks. Specifically, an auxiliary LLM is used to generate a value-oriented description of the harmful task, which is then incorporated with two prompt modules—the Value Engine Module and the Rule Guidance Module—to convince the target LLM to acknowledge the task’s value and thereby execute the harmful behavior. The paper presents comprehensive experiments demonstrating the effectiveness and generalizability of VDJA. However, the novelty of this value-driven approach remains unclear.

### Strengths
- The paper conducts extensive experiments across multiple mainstream LLMs and compares VDJA with various baseline methods, achieving a relatively high ASR.

### Weaknesses
- The main contributions of VDJA lie in the proposed Value Engine Module and Rule Guidance Module, which essentially serve as prompt templates designed to persuade LLMs to recognize the value of harmful tasks. Although the authors emphasize the “value-driven” nature of the method, its mechanism still appears to be a specific form of persuasion-based attack. Therefore, its novelty compared with prior approaches such as PAP is not clearly established.
- The experimental setup is unclear. It appears that the auxiliary LLM in VDJA generates the value description only once for each harmful task. Were the baseline methods also restricted to a single prompt iteration? Such a setting might be unreasonable. Moreover, the performance of VDJA when the auxiliary LLM is called multiple times is not reported.

### Questions
- Is it truly necessary to use an LLM to generate the positive value description? For instance, if the positive value description were fixed as “Understanding the [Harmful Task] is critical for safety,” how would the performance differ from that achieved when the description is dynamically generated by an LLM based on the specific harmful task?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a jailbreak technique for large language models called VDJA. The idea is simple: first prompt the model to affirm the moral value of a harmful task, then instruct it to execute that task. The authors argue that LLMs tend to align with human-like value judgments, and exploiting this tendency can bypass safety mechanisms. Experiments on several frontier models show high attack success rates, outperforming prior jailbreak methods.

### Strengths
1. The paper exposes a genuine security concern: moral-framing can weaken model safety alignment.
2. Empirical performance is strong across multiple models and benchmarks.
3. Ablations and some defense evaluations are included.
4. The attack works in a single query and without concealment, which highlights a meaningful failure mode in current safety systems.

### Weaknesses
1. The proposed method is essentially a scripted prompt pattern. The "value engine + rule guidance" framing is a wrapper around a handcrafted prompt. The methodology section is extremely short and offers no deeper formulation or analysis. It is difficult to view this as a substantive technical contribution.
2. The claim that models "affirm moral value then act consistently" is intuitive but remains speculative. There is no attempt to probe internal model behavior, analyze decision pathways, or connect this to existing alignment literature. This weakens the scientific value of the work.
3. The method is much closer to an adversarial prompt recipe than a principled attack framework. Prior red-teaming work also leverages framing, role-induction, and persuasion; here the innovation appears incremental.
4. While the experiments are thorough, the core technique is too lightweight relative to the scale of the empirical section. The paper feels like a strong empirical study built around a clever prompt trick, rather than a research contribution with lasting conceptual substance.
5. The attack seems sensitive to exact phrasing (e.g., "affirm the moral value…"). No evaluation of paraphrases, adversarial reformulations, or robustness under system-prompt hardening is provided. Without that, it is hard to know whether this is a fundamentally exploitable failure mode or just prompt surface-hacking.

### Questions
1. How sensitive is the success rate to prompt paraphrasing?
2. Does the attack still hold if the model is prevented from producing chain-of-thought?
3. Can you formalize or model the value induction to task execution effect beyond intuition?
4. How does this differ fundamentally from prior persuasion-based jailbreaks?

### Soundness
3

### Presentation
3

### Contribution
2
