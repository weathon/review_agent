# SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
Multi-turn jailbreaks capture the real threat model for safety-aligned chatbots, where single-turn attacks are merely a special case. Yet existing approaches break under exploration complexity and intent drift. We propose SEMA, a simple yet effective framework that trains a multi-turn attacker without relying on any existing strategies or external data. SEMA comprises two stages. Prefilling self-tuning enables usable rollouts by fine-tuning on non-refusal, well-structured, multi-turn adversarial prompts that are self-generated with a minimal prefix, thereby stabilizing subsequent learning. Reinforcement learning with intent-drift-aware reward trains the attacker to elicit valid multi-turn adversarial prompts while maintaining the same harmful objective. We anchor harmful intent in multi-turn jailbreaks via an intent-drift-aware reward that combines intent alignment, compliance risk, and level of detail. Our open-loop attack regime avoids dependence on victim feedback, unifies single- and multi-turn settings, and reduces exploration complexity. Across multiple datasets, victim models, and jailbreak judges, our method achieves state-of-the-art (SOTA) attack success rates (ASR), outperforming all single-turn baselines, manually scripted and template-driven multi-turn baselines, as well as our SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) variants. For instance, SEMA performs an average 80.1% ASR@1 across three closed-source and open-source victim models on AdvBench, 33.9% over SOTA. The approach is compact, reproducible, and transfers across targets, providing a stronger and more realistic stress test for large language model (LLM) safety and enabling automatic redteaming to expose and localize failure modes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper propose a framework for training multi-turn jailbreak attackers that addresses exploration complexity and intent drift without relying on predefined strategies, external datasets, or victim feedback. The method uses two stages: (1) prefilling self-tuning to fine-tune the attacker on self-generated, non-refusal multi-turn prompts, and (2) reinforcement learning with intent-drift-aware rewards using GRPO and a composite reward function to maintain harmful objectives while exploring diverse strategies. By adopting an open-loop, response-agnostic approach, SEMA decouples prompt planning from victim responses, reducing computational costs. The framework achieves state-of-the-art results (80.1% ASR on AdvBench, 33.9% above prior best), outperforms all existing baselines, and demonstrates strong transferability across models and datasets, providing a realistic, scalable, and reproducible stress test for LLM safety.

### Strengths
The paper introduces an intent-drift-aware reward function that prevents conversation drift—the key failure mode of multi-turn attacks—enabling attacks to maintain harmful intent across 5–7 turns while appearing benign.

It employs online reinforcement learning to automatically discover diverse multi-turn jailbreak strategies without any predefined templates or external attack datasets.

### Weaknesses
Though the paper shows generalization of the attacker on smaller models, it provides limited evaluation on frontier models such as GPT-4o/5, Claude 3.5/4 (Sonnet/Opus), and Gemini 1.5/2.0 Pro. Additionally, small open-source models do not undergo extensive safety training and are relatively easy to jailbreak. Demonstrating whether the method generalizes to frontier, highly safety-tuned models would better showcase its effectiveness.

Although the authors compare their method with other multi-turn attacks like Crescendo, GOAT, FITD, etc., they omit comparison or discussion with more recent state-of-the-art multi-turn methods such as X-Teaming (https://arxiv.org/abs/2504.13203) and ActorAttack (https://arxiv.org/abs/2410.10700
), which demonstrate the effectiveness of open-source attacker models in jailbreaking nearly all frontier models.

It primarily focuses on the attack side and does not explore the defense side.

### Questions
1. As the authors use GPT-4.1-mini as their evaluation model for reward computation, does the evaluator maintain consistent evaluation performance as training progresses, or is any reward hacking or exploitation pattern observed?

2. Why does performance degrade beyond 7 turns (Figure 3)?

3. Is there any quantitative analysis and comparison of attack diversity?

4. What responsible disclosure and access control practices will be implemented to prevent the malicious use of SEMA?

### Soundness
3

### Presentation
3

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
This paper introduces a framework for training multi-turn jailbreak attackers via reinforcement learning. The approach has two stages: 1) prefilling self-tuning to generate parseable, non-refusal multi-turn attack sequences, basically it's a finetuning to get an attacker model to generate paraphrases or different ways of asking for something 'bad. 

this is followed by 2) GRPO training with an intent-drift-aware reward, which makes sure the conversation content doesnt change during multi-turn. 

Basically their method decouples multi-turn from any independence on prior turns or the victim, reducing complexity and fanning degree when doing the 'search' over prompt space. The method operates in an open-loop manner, generating complete multi-turn attack plans without conditioning on victim responses. Experiments across AdvBench and HarmBench show high attack success rates against multiple victim models (Qwen2.5-3B, Llama-3.1-8B, GPT-4.1-mini, and GPT-oss-20B), outperforming single-turn and template-driven multi-turn baselines across multiple judges.

### Strengths
Strong transferability: The method demonstrates high transfer rates across different victim models, suggesting the learned attacks capture generalizable vulnerabilities rather than model-specific artifacts.

Simplified threat model: The open-loop generation approach reduces computational requirements by avoiding the need for iterative victim interaction during attack generation. This also removes dependencies on predefined strategy templates or branching assumptions that constrain template-driven methods.

Independent prompt generation: The finding that response-agnostic, independently generated prompts can achieve effective multi-turn jailbreaks is useful for understanding attack mechanics and may inform future defense strategies.

### Weaknesses
*Missing cost analysis*: Despite frequent mentions of reduced cost as a key advantage, the paper lacks quantitative analysis of computational requirements. Specifically:

1. How many prompts need to be generated on average during training and inference?
2. What are the API costs for the evaluation model (GPT-4.1-mini) during training?
3. How does the total cost compare to baseline methods like Crescendo or GOAT?
4. What is the cost breakdown between prefilling self-tuning and RL stages?


*Incomplete ablation studies*: Several design choices lack sufficient justification:

The contribution of multi-turn structure versus the prefilling optimization is not isolated
The impact of prompt ordering is not analyzed
The relative importance of turn position versus prompt content is unclear


*No discussion of defenses or mitigations*: The paper focuses entirely on the attack side without discussing potential countermeasures, detection methods, or mitigation strategies. This limits the utility for practitioners trying to defend against such attacks.

### Questions
**Turn and prompt statistics**: Do you have statistics on how many prompts/turns are needed on average to achieve successful jailbreaks? Are there patterns where certain types of harmful intents require more or fewer turns? Any clustering analysis on this?
Common attack patterns: Did you observe common themes or strategies across prompts that successfully jailbreak models? This could provide insights into failure modes.

**Order sensitivity**: If you shuffle the order of the independently generated prompts for a given intent, does it significantly change the outcome? What is the standard deviation or error over different orderings?
Turn position vs. content: Is the turn at which a prompt appears more important than the prompt itself? For instance, if prompt 7 achieves a jailbreak, would the same prompt work at turn 3?

**Component contributions**: How much of the performance gain comes from the multi-turn structure versus the prefilling self-tuning optimization? Can you provide an ablation that isolates these factors?

**Computational costs**: Can you provide a detailed cost analysis including:

Number of API calls during training
Total token costs for evaluation model
Training time comparisons with baselines
Inference cost per attack generation

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
SEMA is a compact, reproducible framework for training open-loop, response-agnostic multi-turn jailbreak attackers that avoids hand-authored scripts, templates, or external corpora by combining a prefilling self-tuning stage (to produce non-refusal, well-formatted multi-turn rollouts) with reinforcement learning using an intent-drift-aware reward (which balances intent alignment, compliance risk, and level of detail) so the attacker preserves harmful intent across turns while exploring broadly; this approach reduces exploration complexity via one-shot multi-turn prompt generation, achieves state-of-the-art attack success rates and strong transferability on AdvBench and HarmBench (≈80% ASR on average, large gains over prior single- and multi-turn baselines), scales with attempt budget, and is offered as an automated red-teaming tool intended to surface vulnerabilities and improve LLM safety under responsible use guidelines

### Strengths
- This work proposes a decent multi-turn jailbreak framework that achieves higher ASR compared to reported single-turn and multi-turn jailbreak methods.

- The evaluation is relatively thorough, testing many open and closed models across two solid benchmarks.

- Results show that SEMA achieves higher ASR compared to compared single-turn and multi-turn methods.

- The visual presentations of this paper are effective for conveying the mechanism of the framework as well as delivering core takeaways of the results.

### Weaknesses
- The paper’s scope is limited by its exclusive focus on developing attackers without accompanying defensive methods. While SEMA advances the study of multi-turn jailbreaks, it offers no systematic exploration of countermeasures or co-evolving defenses. As a result, the work demonstrates how to break safety mechanisms effectively but provides little insight into how to strengthen or adapt them, narrowing its overall contribution to LLM safety research.

- This works claims to achieve SOTA attacker performance but it lacks comparisons to more recent/performant advances in multi-turn jailbreaks, e.g., https://arxiv.org/abs/2504.13203 and https://arxiv.org/abs/2410.10700 which are shown to be substantially better than Crescendo, CoA, and FITD, the baselines included in this paper.

- The method largely builds on GRPO with modified reward components, which limits its degree of methodological novelty.

### Questions
In addition to the weakness:

- To serve realistic red-teaming needs for broadly revealing LLM vulnerability, it's crucial that an automatic jailbreak or red-team method to be able to discover a wide range of successful attacks. Is SEMA capable of identifying multiple diverse attacks given the same seed harmful query? Could you quantify such ability?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents SEMA, which is a simple yet effective framework for multi-turn jailbreaks. It uses prefilling self-tuning to produce non-refusal, well-structured rollouts and an intent-drift-aware reward to keep the harmful objective anchored across turns.

### Strengths
1. This paper is clearly written and provides a well-defined explanation of the proposed approach.
2. The proposed intent-drift-aware reward and GRPO-based jailbreaking method is simple yet effective and novel.
3. The experiments are comprehensive, and the experiments involving various baselines and models sufficiently demonstrate the superiority of the proposed methodology.

### Weaknesses
Major
1. The open-loop assumption side-steps the real feedback dynamics where victim replies steer the attacker (including deflections). While this is computationally attractive, it may overestimate transferability to real attackers who adapt turn-by-turn. A head-to-head closed-loop variant of SEMA (same reward and intent anchor, but conditioned on last victim response) would clarify the realism/efficiency trade-off.

2. The intent-drift-aware reward is central but depends on an evaluation model (GPT-4.1-mini) and prompt design. The paper would benefit from: (i) prompt release, (ii) cross-evaluator robustness (swap the evaluator LLM family/size).

3. Claims of efficiency (vs. interactive templates) are not quantified. Please report absolute compute for training (SFT+RL) and per-attempt inference costs vs. baselines.

Minor:
In Table 3, HarmBench's “No Refusal” performance seems better for Crescendo than for SEMA, but the bold highlighting appears reversed.

Typos:
L66 : prefilling self turning - > prefilling self tuning,
L938 : Qwen2.5-3B-Intrust - > Qwen2.5-3B-Instruct
L1322 C.3 MORE ABALATION STUDIES -> MORE ABLATION STUDIES

### Questions
1. What happens if only rewards are used with basic RL methods like PPO? While experiments were conducted on DPO/SFT, I would like to see the potential for combining the proposed reward with methods other than GRPO.

2. Why does performance decrease when T_max increases from 7 to 10 in Fig.3(right) ?

### Soundness
3

### Presentation
3

### Contribution
2
