# CommandSans: Securing AI Agents with Surgical Precision Prompt Sanitization

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
The increasing adoption of LLM agents with access to numerous tools and sensitive data significantly widens the attack surface for indirect prompt injections. Due to the context-dependent nature of attacks, however, current defenses are often ill-calibrated as they cannot reliably differentiate malicious and benign instructions, leading to high false positive rates that prevent their real-world adoption. To address this, we present a novel approach inspired by the fundamental principle of computer security: data should not contain executable instructions. Instead of sample-level classification, we propose a token-level sanitization process, which surgically removes any instructions directed at AI systems from tool outputs, capturing malicious instructions as a byproduct. In contrast to existing safety classifiers, this approach is non-blocking, does not require calibration, and is agnostic to the context of tool outputs. Further, we can train such token-level predictors with readily available instruction-tuning data only, and don’t have to rely on unrealistic prompt injection examples from challenges or of other synthetic origin. In our experiments, we find that this approach generalizes well across a wide range of attacks and benchmarks like AgentDojo, BIPIA, InjecAgent, ASB and SEP, achieving a 7–10× reduction of attack success rate (ASR) (34% to 3% on Agent Dojo), without impairing agent utility in both benign and malicious settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a token-level sanitization process that surgically removes instructions directed at AI systems from tool outputs to defend against indirect prompt injection attacks. The authors evaluate their sanitization method across various datasets to validate its effectiveness.

### Strengths
1. The paper is clearly written and easy to follow.
2. The evaluation covers multiple datasets, providing some breadth to the experimental validation.

### Weaknesses
1. Unrealistic assumption about instruction-free data: The core assumption that tool outputs should not contain any instructions is flawed and impractical. Consider a realistic scenario where a user asks, "Check my to-do list and help me solve the tasks on it." In this case, the retrieved data will inevitably contain instructions (i.e., the to-do tasks themselves), which the agent must process to complete the user's request. The proposed sanitization would incorrectly remove legitimate task information, breaking normal functionality. The authors need to address how their method distinguishes between malicious injected instructions and legitimate instructions within tool outputs.

2. Limited novelty: The idea of using LLM-based sanitization to filter prompts has already been explored in prior work. For instance, Shi et al. [1] proposed PromptArmor, which achieves similar goals through prompt-based defenses without requiring any additional training. They showed that directly applying SOTA LLMs can achieve 0 ASR with high utility. 

3. Insufficient comparison with state-of-the-art defenses: The paper evaluates against too few baseline defense methods. More critically, recent defenses have already achieved nearly 0% ASR on Agentdojo The authors should include comprehensive comparisons with state-of-the-art defenses, including PromptArmor [1] and other recent methods, and try to explain why their approach is needed if existing methods already achieve near-perfect defense.


[1] Shi, Tianneng, et al. "Promptarmor: Simple yet effective prompt injection defenses." arXiv preprint arXiv:2507.15219 (2025).

### Questions
Please see the weaknesses part.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes to defend against prompt injection attacks in LLM agents as a byproduct of detecting 'malicious instruction tokens'. They trained a RoBERTa-based model for masking 'malicious instruction tokens' as a binary classification problem. Then they remove the detected masked tokens since they are treated as an injection so that the agents can continue their original normal behavior. Based on this result, they claim their defense is stronger than two previous baselines.

### Strengths
1. The paper is easy to understand 

2. The task is important, and the authors are trying to tackle a critical task.

### Weaknesses
1. The idea is not new and not solid. Training a binary classifier for tokens based on RoBERTa does not sound like a promising idea to defend prompt injections at the era of 2025. The boundary between 'normal' and 'malicious' tokens is hard to classify on the token-level and the authors did not provide enough evidence to support this.

2. The baselines are weak and not enough. Only two baselines are used to prove the superiority of the authors' method.

### Questions
1. Why train a RoBERTa for token classification? How about simple prompting? what's the robustness of this classifier? eg, false negative

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
CommandSans introduces a novel token-level sanitization approach for defending against indirect prompt injection attacks on AI agents. Rather than binary sample-level detection (which causes high false positives and blocks agents entirely), CommandSans surgically removes AI-directed instructions from tool outputs at the token level. The method is inspired by the security principle that data should not contain executable instructions. Using a BERT-based classifier trained on instruction-tuning data with LLM-based labeling, CommandSans achieves 7-19× reduction in attack success rate across benchmarks (AgentDojo, BIPIA, ASB, InjecAgent, SEP) while maintaining agent utility. The non-blocking nature and lack of calibration requirements make it practical for deployment.

### Strengths
- Novel framing of the problem as token-level instruction sanitization rather than sample-level detection, which is theoretically well-motivated by security principles and practically effective.
- Comprehensive evaluation with consistent strong performance, including human red-teaming study showing robustness against expert attackers.
- Acknolodges limitations and even performs a human red team search against CommandSans
- Strong performance for utility and safety

### Weaknesses
- The conclusion mentions "bridging gap between research and deployment" but more discussion of actual deployment considerations (latency, costs, failure modes) would help.
- Figures are a bit messy/text heavy and hard to follow, captions could be more descriptive.
- Given the experiments it is hard to tell how generalizable this method can be and if it can scale well to more data/larger models.
- Could benefit from more discussion on the cost to use CommandSans and to train CommandSans

### Questions
### Important

1. [Section 5.2] The semantic reframing attack succeeded in 1% of attempts. Can you provide more analysis of this attack class and potential defenses? Is this a fundamental limitation of instruction-detection approaches?
2. [Section A.4] Can the authors provide more details on the data augmentation strategy—what characters, at what frequencies, and how gradually does augmentation strength increase from 0 to 20%?
3. Can the authors comment on the failure points of commandsans more? When the utility is lower, what about commandsans is causing failure? 
4. Can the authors comment on how they think CommandSans can be adapted to semantic reframing?
5. [252] This claim does not seem well substantied to me, because the detector is essentially acting as a classifier, I do not see how instruction-following is relevant. I would be more convinced if you could either show experiments that show instruction following actually affects second-order attacks in this context or explain more why.


### Minor

1. [83]]missing 'on'
2. [103] an -> a
3. [52] extra 'a'
7. [Section 6] The conclusion mentions "bridging gap between research and deployment" but more discussion of actual deployment considerations (latency, costs, failure modes) would help.

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
This paper proposes CommandSans, a token-level prompt sanitization framework for defending LLM agents against indirect prompt injection attacks. Instead of traditional sample-level detection (which blocks entire inputs when an attack is suspected), CommandSans performs a more fine-grained token tagging to identify and remove only the instructional components directed at AI systems, leaving benign content intact.

Disclaimer: I am not from the LLM safety community but instead the MARL community, so my judgment may be inaccurate

### Strengths
(1) Novel framing of the defense problem
Treats prompt-injection defense as a token-level sanitization task, rather than binary classification.

(2) Thorough evaluation methodology
Combines five public benchmarks with a human red-teaming study, giving both quantitative and qualitative validation.

### Weaknesses
(1) Limited generalization to structured or code-like domains
(1.1) Performance drops notably in Code QA and Table QA tasks (Table 2), suggesting a distributional mismatch between training and deployment data. 
(1.2) I only see a limited example of instructions. Would a simple codebook or ML masking for similar prompts perform as well? If not, why?
(1.3) What if "book a meeting on Google Calendar", which seems benign, is used as a method to attack?

(2) No statistical analysis and confidence interval are reported in all tables.

(3) Citations should be carefully examined (e.g., L265, L266 author names are weird)

### Questions
(1)(2) see weakness

(3) How does CommandSans handle semantic or implicit prompt injections that do not contain explicit instruction tokens (e.g., malicious intent framed as compliance rules or reasoning guidance), and can token-level sanitization fundamentally defend against such reframed attacks?

(4) Is there any methodology to optimize the defense strategies for the Pareto optimal?

### Soundness
3

### Presentation
3

### Contribution
2
