# Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Content Warning: This paper contains examples with harmful content, and the constructed dataset includes samples that may be considered offensive.
As large language models (LLMs) continue to advance in capabilities, ensuring their safety against jailbreak attacks remains a critical challenge. In this paper, we introduce a novel safety alignment approach called Answer-Then-Check, which enhances LLM robustness against malicious prompts by applying thinking ability to mitigate jailbreaking problems before producing a final answer to the user. Our method enables models to answer the question in their thoughts directly and then critically evaluate its safety before deciding whether to provide it. To implement this approach, we construct the Reasoned Safety Alignment (ReSA) dataset, comprising 80K samples that teach models to reason through direct responses and then analyze their safety. Experimental results demonstrate that our approach achieves the Pareto frontier with superior safety capability while decreasing over-refusal rates. Notably, the fine-tuned model maintains general reasoning capabilities on benchmarks like MMLU, MATH500, and HumanEval. Besides, our method equips models with the ability to perform safe completion, while post-hoc detection methods can only directly reject sensitive, harmful queries (e.g., self-harm). Our results show that inference-time strategies alone are insufficient, highlighting the necessity of safety training, and we find even $500$ samples can yield performance comparable to the entire dataset, suggesting a promising path for data-efficient safety alignment. The dataset is publicly available at: https://huggingface.co/datasets/ByteDance-Seed/ReSA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes the Answer-Then-Check paradigm and trains LLMs via supervised fine-tuning on the Reasoned Safety Alignment (ReSA) dataset ($\sim 80\mathrm{K}$ samples) so the model first generates an intended-answer summary and then performs an explicit safety analysis before emitting a final response. ReSA-SFT substantially improves robustness against diverse jailbreak attacks (including adaptive and white-box methods) while preserving general reasoning performance (MMLU, MATH500, HumanEval) and lowering over-refusal; the paper contributes (1) the Answer-Then-Check training paradigm, (2) the 80K ReSA dataset and a safe-completion mechanism, and (3) empirical evidence that safety fine-tuning can reach a Pareto frontier over post-hoc and inference-time defenses.

### Strengths
**Originality & significance:** The paper proposes the Answer-Then-Check paradigm and the ReSA dataset (~80K), reframing safety alignment as process-supervised fine-tuning that lets models self-inspect before responding; this shift is practically significant because it yields a Pareto improvement in safety vs. over-refusal and shows strong data efficiency (≈500 samples can approach full-data performance).
**Quality:** The experimental evaluation is broad—multiple benchmarks (MMLU, MATH500, HumanEval), several target LLMs, and adaptive jailbreak scenarios—and the paper gives a formalized training objective and structured evaluation metrics that support the claimed robustness while preserving general reasoning.
**Clarity:** The method, dataset construction, and evaluation pipeline are presented clearly with equations and figures, making the approach reproducible and straightforward for follow-on work.

### Weaknesses
- **Incomplete Evaluation:** The study omits recent SOTA attack baselines (e.g., AutoDAN-Turbo [3], BOOST + GPTFuzzer [1]); add one–two newer attacks and evaluate on modern LLMs (Qwen3, Gemma3) to make comparisons more comprehensive.

-  **Noverl Method:** Although the approach points to an interesting defense direction, the methodology is still fairly simple; integrating an RL-based strategy (e.g., GRPO) could strengthen jailbreak resistance and reduce over-refusal.

-  **Time Consideration:** The generate-then-check pipeline likely incurs nontrivial deployment latency—please provide experiments quantifying runtime overhead versus the base model.

- **AVerage calculation is strange:** The averaging in Tables 3 & 4 is unclear (e.g., mixing Over-refusal and General Reasoning benchmarks); clarify the aggregation method and report per-benchmark (or appropriately weighted) averages.

### Questions
- Clarify whether including PAP and PAIR attack samples in the dataset biases the reported defense scores — e.g., by leaking attack patterns into training or evaluation — and quantify this with an ablation that removes those sources and reports the delta in metrics.

- The term “reasoned” is initially ambiguous; either adopt a clearer label (e.g., “reasoning-based” or “process-supervised”) or add a short, precise paragraph in the paper that explains why the method is “reasoned” (how intermediate reasoning steps are generated, verified, and used to decide the final output).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper collects a dataset from existing jailbreaks and finetune models to generate a longcot (answer then check) to make it aligned against jailbreak attack.

### Strengths
Generally, I would support this paper, as OpenAI's recent released gpt-oss-safeguard also validates the vision that LLM reasoning can be robust against attacks. The motivation of this paper is also clear that generate the answer then check it, similar with TTS. This work also has a comprehensive experimental validation.

### Weaknesses
- The experiment section is a bit painful to read. For example, in Table 2, there's no description about the number. Is it ASR or DSR? Also, it would be more readable if you could highlight the highest number in each column instead of just the avg column. To make it worse, there are three evaluators, Llama-guard, finetuned evaluator, harm bench. I don't quite get how they are chosen and how they work. The harm bench is a dataset, how could it be an evaluator? The finetuned evaluator is what? Moreover, `Posthoc detection (Llama-Guard [26], GuardReasoner [25])', which one is used for the postdoc row? I would suggest the reader re-write the experiment part to make it more readable.

- There should be more discussions for the experimental results. The STAIR-DPO seems a strong competitor, and it seems it achieves consistently better performance on some attacks (None and GPTFuzz) than your methods. It would be better with some data points discussion.

### Questions
- Could you compare with gpt-oss-safeguard? I understand that it is released after your work's submission, but I am interested to see how your method compare with that.

- Could you justify why you pick the SFT? In STAIR and GuardReasone, they both adopt DPO for training. As it was not hard to obtain the negative samples, I am wondering if using DPO would have better performance.

I would raise my score if you could tackle my concerns, especially regarding the experiment section.

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
3

### Summary
This paper proposes a novel safety alignment method called Answer-then-check, designed to improve large language models’ (LLMs) robustness against jailbreak attacks. The proposed method enables a model to first generate an “intended answer summary” and subsequently perform a safety analysis before producing its final output. To support this, the authors construct ReSA, a dataset with 80K “Answer-Then-Check” samples.

### Strengths
(1)The proposed method is simple and effective. It can substantially improve LLM safety while keeping low over-refusal and preserving general reasoning capability.(2)It does not introduce heavy inference expense compared to the base model.(3)Data efficiency and scalability demonstrated via subset experiments.(4)The paper is well-written and easy to follow.

### Weaknesses
The intended answer summary may contain unsafe content. This raises deployment and threat-model questions. (e.g., what if attackers exploit the internal summary via chain-of-thought leakage). The authors acknowledge this but the paper would benefit from a more systematic threat analysis.

Since Answer-Then-Check depends on special tokens like <safety_check>...</safety_check>, have you evaluated robustness to prefilling attacks where an adversary inserts an empty or misleading <safety_check> block (e.g., <safety_check>I can provide this answer to the user.</safety_check>) into the prompt to bypass the safety reasoning?

### Questions
See weakness

### Soundness
3

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
This paper proposes a new safety method called "Answer-Then-Check" to defend against jailbreak attacks on LLMs. The model first plans an answer, checks it for safety, and then decides whether to respond. They built a dataset (ReSA) of 80K samples and showed strong results against various attacks while keeping the model helpful.

### Strengths
+ The "Answer-Then-Check" idea is simple but effective—letting the model think through its answer before sharing it helps catch hidden harmful intent.

+ The method improves safety without making the model overly cautious. It keeps useful capabilities (like math and coding) and even handles sensitive topics with care instead of just saying "I can’t help."

### Weaknesses
- The extra "think-check" steps make the model slower, especially for normal, safe questions where this process isn’t really needed.

- Even though unsafe content is hidden from users, the model still generates it internally during the "answer planning" step, which could be a concern if those thoughts were ever exposed.

### Questions
What happens if the safety check itself gets jailbreaked? For example, it is possible that the model outputs a harmful answer but the safety analysis wrongly thinks it’s safe—how often does that happen, and how can the system be improved to avoid such jailbreaks?

### Soundness
3

### Presentation
3

### Contribution
3
