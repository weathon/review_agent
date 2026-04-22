# Mitigating Indirect Prompt Injection via Instruction-Following Intent Analysis

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 4, 2, 2

## Abstract
Indirect prompt injection attacks (IPIAs), where large language models (LLMs) follow malicious instructions hidden in input data, pose a critical threat to LLM-powered agents. In this paper, we present IntentGuard, a general defense framework based on instruction-following intent analysis. The key insight of IntentGuard is that the decisive factor in IPIAs is not the presence of malicious text, but whether the LLM intends to follow instructions from untrusted data. Building on this insight, IntentGuard leverages an instruction-following intent analyzer (IIA) to identify which parts of the input prompt the model recognizes as actionable instructions, and then flag or neutralize any overlaps with untrusted data segments. To instantiate the framework, we develop an IIA that uses three ``thinking intervention'' strategies to elicit a structured list of intended instructions from reasoning-enabled LLMs. These techniques include start-of-thinking prefilling, end-of-thinking refinement, and adversarial in-context demonstration. We evaluate IntentGuard on two agentic benchmarks (AgentDojo and Mind2Web) using two reasoning-enabled LLMs (Qwen-3-32B and gpt-oss-20B). Results demonstrate that IntentGuard achieves (1) no utility degradation in {all but one setting} and (2) strong robustness against adaptive prompt injection attacks (e.g., reducing attack success rates from 100\% to 8.5\% in a Mind2Web scenario).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a method named IntentGuard to protect Large Language Model (LLM) agents against Indirect Prompt Injection Attacks (IPIAs). The basic idea of IntentGuard is to analyze whether the LLM intends to follow instructions originating from untrusted data, which is the fundamental reason for successful IPIAs. Experiments on two datasets show the proposed method has strong defense performance.

### Strengths
1. The proposed method is novel and reasonable.

2. The experimental results are promising.

3. This paper is well written.

### Weaknesses
1. More benchmarks, especially those for IPIAs, should be incorporated into experimental evaluation.

2. Only a few baseline methods are compared. More defense methods for IPIAs should be added for comparison.

3. The proposed method may lead to increment of the system prompt length, which may limit the length of working context.

4. The proposed method has the risk of being bypassed. For example, a highly capable adaptive attacker can craft a malicious prompt that causes the LLM to execute harmful instructions while being induced or forced not to include them in the structured 'instruction-following intent' list.

5. The proposed method may be tricked. For example, an attacker can easily cause false positives by inserting some instructions in data like “following your original instructions”.

### Questions
Please refer to my above comments.

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
The paper proposes IntentGuard, a defense against indirect prompt injection attacks that centers on extracting the model’s instruction-following intent and tracing it back to trusted vs. untrusted input segments. The key component is an instruction-following intent analyzer (IIA) elicited via three “thinking intervention” strategies—start-of-thinking prefilling, end-of-thinking refinement, and adversarial in-context demonstration—which produce an explicit list of intended instructions that are then matched to input spans for alerting or recovery. Experiments on AgentDojo and Mind2Web with Qwen3-32B and gpt-oss-20B show large reductions in attack success rates under multiple adaptive attacks, with near-zero benign utility degradation and zero false positives reported in benign settings. Ablations attribute robustness gains to combining start/end interventions and to adversarial demonstrations, and report similar efficacy for sparse vs. dense origin tracing.

### Strengths
Clear reframing of prompt injection defense from “detect malicious-looking strings” to “detect whether the model intends to follow instructions from untrusted segments.

Practical, modular pipeline: intent extraction → origin tracing → mitigation, with both alert and recovery modes and sliding-window matching tolerant to paraphrase.

Ablations are informative and align with the method’s mechanisms

### Weaknesses
The defense triggers only on instructions listed by IIA; the authors acknowledge unfaithful cases (10.9% bottom-left in Fig. 4) where actions are followed without listed intent. While adversarial demonstrations help, the pipeline offers no fallback when malicious execution occurs without explicit intent listing, leaving a residual risk precisely under stealth objectives.

The proposed method, while practical, but sounds very trivial to me. The proposed thinking intervention is just doing the prompt engineering and provides no technical contribution.

### Questions
What happens if the model does not expose or comply with <think>…</think> and the first-closure replacement? Please report results with “hidden CoT” settings or with models that do not honor these tokens to show the approach is not brittle to formatting.

### Soundness
3

### Presentation
3

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
The paper proposes a defense against indirect prompt injection attacks by first prompting the LLM to generate all instructions it is going to execute. Next, the paper matches each of these instructions using sparse embeddings to the input prompts to identify if these self-reported instructions overlap with any untrusted data segments.

### Strengths
- Prompt injection is a very important problem that is still unsolved 
- Results are better than baselines 
- Experiments are done on a representative number of datasets

### Weaknesses
There might be a lot of attacks and generalization experiments that the paper didn't study. I will organize them below.

- Attacks targeting the intent analyzer: The paper didn't try straightforward attacks that tell the model to not report the injection, or to simply destroy the formatting of thinking tokens and the list of instructions that get extracted and parsed. 

- Attacks targeting origin tracing: The paper currently uses sparse embeddings matching between the reported instructions and fixed windows of the input. But it is imaginable that there might be many ways to target this. For example, the injected instructions can be encoded, obfuscated, or in different languages. If the model reports it in English, the matching may not be successful. The injected instructions can also be very long, using stories, etc. This may not fit within the fixed-length windows, and the report instructions might be a short version of it. 

- I think the paper should really explain how the attacks it implemented such as GCG and PAIR are truly adaptive. For example, for PAIR, an adaptive attack can be designed to destroy the verbalization of instructions by having explicit instructions to the attacker model to change the prompts according to this objective (for example, by breaking the formatting or languages). The PAIR prompt in appendix B only focus on stealthy planning, which might be hard to do for models. The same argument can be made about GCG. 

- The paper does not compare to baselines that use attention or internal states to detect prompt injections, which can be very relevant, and might work better than sparse embeddings, specifically the cited work of Hung et al., 2024 and Abdelnabi et al., 2025.

- The paper is also very conceptually similar to this previous work (https://arxiv.org/pdf/2412.16682?), which was published at ACL 2025 and uses an LLM to extract all instructions in the context and then uses an alignment check module that calculates whether each reported instruction is related to the original user task. Given all previous work, I am concerned that this paper has limited novelty and may be providing a weak solution at the same time.

### Questions
- How does this method differ from previous work that analyzed the attention mechanisms or activations? which some of them (Abdelnabi et al.) used a prompt to tell the model to first extract the instructions it is going to execute.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper “Mitigating Indirect Prompt Injection via Instruction-Following Intent Analysis” introduces IntentGuard, a framework to defend LLM agents against indirect prompt injection (IPI) attacks, where malicious instructions are hidden in untrusted inputs. Instead of merely detecting unsafe text, the method analyzes whether the model intends to follow those injected instructions using an instruction-following intent analyzer (IIA) that leverages reasoning interventions like “start-of-thinking” and “end-of-thinking” refinements. Experiments on AgentDojo and Mind2Web show that IntentGuard greatly reduces attack success rates (e.g., from ~100% to ~8.5%) while maintaining normal task performance, offering a promising direction for intent-aware LLM safety.

### Strengths
The paper introduces a defense that focuses on instruction-following intent rather than surface-level detection of malicious text, targeting a gap in existing IPI mitigation. The proposed IntentGuard framework integrates intent extraction with defensive mechanisms. The experiments on agent benchmarks (AgentDojo, Mind2Web) show reduced attack success rates without affecting normal task performance.

### Weaknesses
The approach depends heavily on the accuracy and reliability of the intent analyzer (IIA), which itself is an LLM and thus vulnerable to adversarial manipulation or misinterpretation of subtle intent. It also assumes the availability of reasoning traces (“thinking”) that may not exist in closed-source or lightweight models. Moreover, the method presumes that the system can reliably identify which parts of the prompt are untrusted—a strong and often unrealistic assumption in real-world agentic setups where content from multiple sources is dynamically combined.

### Questions
Please address the concerns in weakness.

### Soundness
2

### Presentation
3

### Contribution
2
