# ELEPHANT: Measuring and understanding social sycophancy in LLMs

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
LLMs are known to exhibit sycophancy: agreeing with and flattering users, even at the cost of correctness. Prior work measures sycophancy only as direct agreement with users' explicitly stated beliefs that can be compared to a ground truth. This fails to capture broader forms of sycophancy such as affirming a user's self-image or other implicit beliefs. To address this gap, we introduce **social sycophancy**, characterizing sycophancy as excessive preservation of a user’s *face* (their desired self-image), and present **ELEPHANT**, a benchmark for measuring social sycophancy in LLMs. Applying our benchmark to 11 models, we show that LLMs consistently exhibit high rates of social sycophancy: on average, they preserve the user's face 45 percentage points more than humans in general advice queries and in queries describing clear user wrongdoing (from Reddit's r/AmITheAsshole). Furthermore, when prompted with perspectives from either side of a moral conflict, LLMs affirm *whichever side the user adopts* in 48% of cases—telling both the at-fault party and the wronged party that they are not wrong—rather than adhering to a consistent moral or value judgment. We further show that social sycophancy is rewarded in preference datasets. We present both prompting- and steering-based mitigation strategies to reduce social sycophancy, though understanding when and how to apply them without compromising user experience remains an open question. Our work provides theoretical and empirical tools for broadly understanding and addressing LLM sycophancy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper offers a new definition of sycophancy - "social sycophancy" - that is more comprehensive and encompasses more behavior and contribute a dataset for benchmarking language models according to this definition. They measure several SOTA LLMs for sycophancy according to the seven criteria comprising their new definition. Their core findings are that LLMs are highly sycophantic according to this expanded definition, across criteria. They also find that sycophancy is rewarded in preference datasets. They explore mitigation strategies based on prompting and white box intervention.

### Strengths
The paper tests a large variety of language models to substantiate its claims. The authors are careful about measurement and validate with human annotation. The authors test many mitigation strategies.

### Weaknesses
I guess I feel that before reading this paper, I would have been surprised if sycophancy (as it's more narrowly used in the literature) did not correlate tightly with the other forms of social sycophancy highlighted in this paper. I wonder how surprising the core results of this paper are. 

I'm not sure I agree with some of the labels for the examples of sycophancy highlighted in Table 2 - e.g. the two SS dataset responses (last row, third to last row). I wonder whether the reason this paper's empirical findings (e.g. that Gemini is least sycophantic) contradict prior work is that the paper expands the definition of sycophancy, versus social sycophancy being tricky to label / disambiguate from simple politeness.

nit: line 405 - ourmitigations should be our mitigations

### Questions
Re Table 2 - it's not clear whether the truncated model responses would have continued down the sycophantic path or reversed course?

When it comes to measuring the sycophancy of preference datasets, I wonder whether the dis-preferred response is also untruthful? Or rude?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper develops a generalized notion of “social sycophancy” in large language models, defining as behavior that preserves various dimensions of “face”, a concept from psychology literature. In addition to “feedback”, “answer“, and “mimicry“ sycophancy, which have been explored in prior work, the paper describes “validation”, “indirectness”, “framing”, and “moral” sycophancy as novel subtypes under the umbrella of social sycophancy. The paper develops ELEPHANT, a benchmark protocol for assessing LLM sycophancy on these 4 novel dimensions. The benchmark uses four datasets: the advice-giving Open-Ended-Queries (OEQ) dataset, the moral judgment AITA-YTA and AITA-NTA-FLIP datasets, and the assumption-bearing Subjective Statements (SS) dataset. They use an LLM-as-a-judge approach to measure occurrence of validation, indirectness and framing sycophancy, while moral sycophancy is based on “YTA“ and “NTA“ judgments in the two AITA datasets. The validate the LLM-as-a-judge approach against expert human annotation, finding relatively high agreement between majority vote labels and GPT-4o judgments (~85%, >=.65 Cohen’s K). They run 11 popular LLMs through the benchmark, reporting their sycophancy rates on different dimensions, finding high rates across almost all models. They run three preference tuning datasets through the benchmark, finding preferred responses to be much higher in sycophancy (somewhat explaining model behavior), and finally explore two mitigation strategies, one prompt-based and one using DPO, neither of which is effective.

### Strengths
- **Presentation:** The paper is well written and easy to follow. The figures are comprehensive and easy to understand.
    
- **Important topic:** Sycophancy is a key flaw in LLMs, and this paper broadens the discussion of sycophancy to a number of different types
    
- **Interdisciplinary**: The paper draws a novel connection between the relatively well-studied phenomenon of model sycophancy with the psychological phenomenon of “face”, inspiring the development of new sycophancy dimensions
    
- **Construct validation:** The paper validates its scales against a panel of human experts. The experts have high agreement with one another, indicating that the scales are broadly valid.

### Weaknesses
- **LLM-as-a-judge validity:** 3/4 of the dimensions introduced in this paper depend on LLM-as-a-judge for assessment. GPT-4o gets ~85% accuracy/.65 Cohen’s K agreement with human expert consensus, which is high but not perfect, and almost certainly subjected to certain biases. It would have been nice to see some error analysis indicating where and when the judge model is likely to fail. Is GPT-4o better at recognizing its own validation sycophancy than Gemini’s? Worse? When inspecting an LLM’s response for sycophancy, it might be illuminating to compare it to a version of the original prompt that instructs the model to be sycophantic in the desired manner (validation, framing, etc). It would also help validate the scale if human experts and judge LLMs could reliably detect the artificially sycophantic responses.
    
- **Disconnection with downstream effects**: The most common types of sycophancy cited, feedback and answer sycophancy, can be tied to an adverse outcome in the form of model inaccuracy--the model provides objectively poor responses in order to match human preference. This isn’t necessarily the case for validation, indirectness and framing sycophancy. The paper could have made a stronger argument that these are actually harmful behaviors, drawing either on psychology literature or through, perhaps, some type of simulated experiment (e.g., is a simulated /r/AITA poster more likely to come away with the misapprehension that they are NTA, if the model is excessively validating, given that it returns the same judgment?).

### Questions
No particular questions

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces social sycophancy—LLMs’ excessive preservation of a user’s “face” (desired self-image), which goes beyond simple agreement with explicit beliefs. It proposes ELEPHANT, a benchmark that operationalizes four dimensions: Validation, Indirectness, Framing, and Moral sycophancy. Using four first-person datasets (OEQ: 3,027 open-ended advice queries; AITA-YTA: 2,000 posts judged the poster is at fault; SS: 3,777 assumption-laden statements; AITA-NTA-FLIP: 1,591 paired perspectives on the same conflict), the authors evaluate 11 models. They find that models preserve users’ face far more than humans on advice and AITA-YTA, and that models affirm whichever side of a conflict the user adopts ~48% of the time, often telling both sides “you’re not wrong.” Measurement uses a human-validated GPT-4o judge to label sycophancy and compares model rates against crowdsourced human baselines (or chance for SS). The paper also shows preference datasets reward socially sycophantic responses, and that standard mitigations (instruction prepending, perspective shift, generic “truthfulness” steering) are only partly effective; targeted DPO helps in-dimension but struggles on framing and moral sycophancy.

### Strengths
Clear conceptual expansion. Grounding sycophancy in face preservation unifies prior “explicit” notions while motivating new, consequential dimensions (validation, indirectness, framing, moral). The taxonomy is well argued and connected to safety risks in support/advice use cases. 

Well-scoped datasets and metrics. The four datasets map neatly to the four dimensions and are framed to avoid “anything goes” judgments. The metric that centers Δ(model − human) is conservative and easy to interpret. 

Empirical bite. The headline findings are strong: large, consistent gaps vs. humans on advice and AITA-YTA, and 48% two-sided affirmation on moral conflicts. These results are policy-relevant for user-facing deployments.

### Weaknesses
Cultural anchoring of “human baseline.” Crowdsourced references (and AITA norms) skew Western/Reddit culture, which the authors acknowledge. More cross-cultural calibration or region-specific analyses would help. 

Framing & moral remain under-mitigated. The hardest and arguably most safety-critical dimensions show limited improvement. The paper suggests future directions (grounding, alternative objectives), but present mitigations may feel incremental for these categories.

### Questions
Have you explored calibrating human baselines by cultural region or community norms (beyond AITA)? Would the Δ(model − human) gaps persist under non-Western adjudicators?

Designing non-sycophantic yet supportive models. Instruction-prepending “be less validating/indirect” backfires. Have you tried grounded follow-ups (“ask for evidence/alternatives before endorsing”) or chain-of-checks policies that require challenge-then-support? 

Why framing & moral resist DPO. Do you suspect data scarcity or label ambiguity?

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
2

### Summary
This paper introduces the concept of “social sycophancy,” defined as the tendency of LLMs to excessively preserve a user’s desired self-image—going beyond simple agreement with explicit user beliefs. The authors develop the ELEPHANT benchmark to quantify this phenomenon across contexts such as advice-seeking and moral dilemmas. Evaluation across 11 LLMs reveals a pronounced prevalence of sycophantic behavior. The study further indicates that such behavior is reinforced in current preference-based training datasets.

### Strengths
- The paper formulates “social sycophancy” as an extension of face theory, moving beyond simple agreement to incorporate dimensions such as emotional validation, indirect expression, and moral alignment in LLM behavior.

- The ELEPHANT framework is introduced as a structured approach to measure sycophancy in open-ended settings, incorporating multiple human-annotated datasets and employing an LLM-as-judge methodology.
- The work provides a methodological basis for further study of sycophancy, with relevance to real-world contexts such as personalized advice and moral reasoning, suggesting pathways for future work on alignment and safety

### Weaknesses
- It may be beneficial to consider incorporating evaluation results from additional LLMs in the experimental section. Since GPT-4o itself has been observed to exhibit certain tendencies related to social sycophancy and can be sensitive to prompt design, including a more diverse set of judge models could help mitigate potential biases and enhance the stability of the evaluation.

- It would be beneficial to discuss whether the reduction in sycophancy might affect the model's general response quality, helpfulness, and instruction-following ability.

### Questions
please refer above

### Soundness
3

### Presentation
3

### Contribution
3
