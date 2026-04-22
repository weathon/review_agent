# SeqRL: Sequence-Attentive Reinforcement Learning for LLM Jailbreaking

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, underscoring the importance of ensuring their safety and robustness. Recent work has examined jailbreaking attacks that bypass safeguards, but most methods either rely on access to model internals or depend on heuristic prompt designs, limiting general applicability. Reinforcement learning (RL)-based approaches address some of these issues, yet they often require many interaction steps and overlook vulnerabilities revealed in earlier turns. We propose a novel RL-based jailbreak framework that explicitly analyzes and reweights vulnerabilities from prior steps, enabling more efficient attacks with fewer queries. We first show that simply leveraging historical information already improves jailbreak success. Building on this insight, we introduce an attention-based reweighting mechanism that adaptively highlights critical vulnerabilities within the interaction history. Through comprehensive evaluations on the AdvBench benchmark, our method achieves state-of-the-art performance, demonstrating higher effectiveness in jailbreak success and greater efficiency in query usage. These findings emphasize the value of incorporating historical vulnerability signals into RL-driven jailbreak strategies, offering a general and effective pathway for advancing adversarial research on LLM safeguards.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes SeqRL, a black-box jailbreak framework that augments an RL agent’s state with interaction history and further applies an attention mechanism over that history to choose prompt-mutation operators more effectively. Concretely, the authors (i) extend RLbreaker’s setting by concatenating current prompt embeddings with per-turn records (prompt embedding, response features such as refusal flag, reward, and operator ID) to form a History-augmented RL state, and (ii) introduce Attention-based HRL  that computes attention over the last KKK steps to produce a history summary used by a PPO policy to select among five mutators executed by a helper LLM. Experiments on AdvBench against four targets  report higher ASR and lower average number of querie.

### Strengths
1. The paper identifies a concrete deficiency in prior black-box RL jailbreakers, memoryless state, and proposes history concatenation plus attention reweighting. The methodological description of HRL/AHRL is explicit, with the state components and attention equations spelled out, and the action set/reward inherited from RLbreaker described clearly. 

2. Unlike several prior works that evaluate on small AdvBench samples, the paper uses the full 520 prompts with a 364/156 train/val split, which is a more representative setting for reporting ASR/ANQ. 

3. On LLaMA-3.2-11B and GPT-oss-20B, HRL/AHRL show large deltas over RLbreaker, suggesting the history signal is indeed useful for operator selection.

### Weaknesses
1. ANQ is computed only on successful attacks, which favors methods that fail often but sometimes succeed with few queries. A success-weighted or overall query budget–normalized metric (e.g., mean queries per attempt, area-under-budget vs. success curve) would be more informative; the paper does not provide those. 

2. For FlipAttack, the paper introduces a † variant allowing up to 50 trials to make it fair, but it is unclear whether analogous tuning was applied to other baselines (e.g., population sizes, mutation rates for AutoDAN-Turbo) under the same total budget and helper behavior; moreover, some baselines are said to rely on attackers that refuse when replaced by GPT-4o, which effectively changes their algorithmic assumption mid-evaluation. A standardized auxiliary-model interface (same helper, same refusal policy) is needed for a fair head-to-head. 

3. The reward is defined by cosine similarity with a reference answer from an unaligned model, and response features include a heuristically computed refusal flag, perplexity, normalized length, and toxicity, but the paper does not specify the exact reference model, embedding space, toxicity/perplexity estimator(s), thresholds, or how these signals are normalized and combined，key details to reproduce HRL/AHRL states and the reward.

### Questions
All systems use GPT-3.5-Turbo as helper for mutators, but other baselines (e.g., PAIR/AutoDAN-Turbo) are reported to collapse when a cooperating attacker LLM refuses, whereas SeqRL’s helper is framed as non-attacker.”This difference in assumed capabilities/compliance of auxiliary models may be driving part of the reported advantage and is not controlled via standardized helper policies or refusal-rate quantification.

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
3

### Summary
This paper proposes a novel reinforcement learning (RL) framework called SeqRL , designed to more effectively "jailbreak" large language models (LLMs). The authors note that previous RL methods are inefficient because they overlook vulnerability signals revealed in earlier interactions. SeqRL addresses this problem through two key innovations: first, History-augmented Reinforcement Learning (HRL) , which integrates information such as past prompts, responses, and rewards into the RL agent's state ; and second, Attention-based HRL (AHRL) , which uses an attention mechanism to adaptively reweight historical information , enabling the agent to focus on the most critical past vulnerabilities. Experiments on the AdvBench benchmark against modern LLMs like LLaMA-3.2 and GPT-4o demonstrate that SeqRL achieves state-of-the-art performance in both attack success rate and query efficiency.

### Strengths
* Originality: The paper's core originality is addressing the "memoryless" problem in existing RL-based jailbreaking methods. It creatively proposes using an attention mechanism to dynamically reweight historical interaction data, allowing the agent to focus on the most critical past vulnerabilities.

* Quality: The experimental quality is high. Unlike previous work, it evaluates on the full AdvBench benchmark (520 samples), not just a small subset. The ablation studies are highly compelling: Tables 1 and 2 clearly demonstrate that "history" and "attention" are key to the performance boost and show how longer history dramatically improves efficiency by reducing queries.

* Clarity: The paper is clearly written, and Figure 1 provides an intuitive visual flowchart of the entire SeqRL framework. The methodology (HRL and AHRL) is introduced progressively and is easy to understand.

* Significance：Achieves SOTA (State-of-the-Art) performance: It attains the highest attack success rates on multiple powerful LLMs like LLaMA-3.2 and Qwen3-14B. Dramatically improves efficiency: Most importantly, it significantly reduces the number of queries (ANQ) required for a successful jailbreak, making it a more practical and low-cost black-box attack. More robust method: Unlike other SOTA methods that rely on a cooperating "attacker LLM" , SeqRL does not depend on an auxiliary model to generate malicious intent, making it more stable and less likely to fail.

### Weaknesses
* Significant Underperformance on Key Proprietary Model: Although SeqRL achieves state-of-the-art ASR on three open-source models, its performance on GPT-4o is notably weak. SeqRL's ASR of 74.84% is not only almost identical to the baseline RLbreaker (74.36%) but also far below AutoDAN-Turbo (91.03%) and FlipAttack (95.51%).
* Lack of Evaluation Against Explicit Defenses: The paper does not mention testing the SeqRL attack against the specific jailbreak defense, such as safe decoding, context engineering like self-reminder, or query perturbation.
* Ambiguous Experimental Settings: The paper's experimental settings are not fully specified. For example, the paper states it uses Qwen3-14B, but it does not clarify which inference mode (e.g., "thinking mode" or "non-thinking mode,") was used for this mixed-inference model, which could impact the results.

### Questions
The paper's evaluation is limited to the models' built-in, inherent safety alignments. A key weakness is the failure to test SeqRL against any explicit, add-on jailbreak defense mechanisms. To demonstrate the attack's true robustness and practical threat level, the authors should evaluate its performance against modern defenses designed to counter such attacks.

Without these tests, it is unknown if SeqRL's history-aware approach can bypass current defense-in-depth strategies. We strongly suggest testing against a representative set of defenses, such as:

* Contextual Engineering: (e.g., Self-Reminder) [1]

* Query Inspection: (e.g., Token Highlighter) [2]

* Gradient-based Detection: (e.g., Gradient Cuff) [3]

* Decoding Intervention (e.g. Safe Decoding). [4]

**References**

[1] Defending ChatGPT against jailbreak attack via self-reminders

[2] Token Highlighter: Inspecting and Mitigating Jailbreak Prompts for Large Language Models.

[3] Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes. 

[4] SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding.

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
This paper proposes a sequence-aware reinforcement learning framework for LLM jailbreaking.

### Strengths
1. The attack success rate of SeqRL seems generally high on the AdvBench.
2. The design of the attention module is simple and intuitive. 
3. The authors conducted ablations about the attack modules and steps, which shows reasonable results.

### Weaknesses
1. The experiments are conducted on a single benchmark. I believe it is necessary to introduce a new benchmark, e.g., HarmBench. And include one more LLM judge to verify the attack success rate?

2. I do not have a clear intuition about why SeqRL is the only effective method for GPT-OSS. Actually, I am not convinced that a simple MLP can learn such complex decisions with simple features like length/refusal etc. 

3. I believe it is important to include more details to support the results shown in the main experiment such as the training curves of the MLP module, the evaluation prompts, sucessful/faillure cases etc. I would like to increase my score if such context can be provided.

### Questions
1. Can you share several sucessful attack prompts generated via SeqRL and the corresponding responses.
2. Can you share the actual implementation of SeqRL via anonymous Github?

### Soundness
2

### Presentation
2

### Contribution
2
