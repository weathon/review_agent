# On the Shelf Life of Fine-Tuned LLM-Judges: Future-Proofing, Backward-Compatibility, and Question Generalization

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
The LLM-as-a-judge paradigm is widely used in both evaluating free-text model responses and reward modeling for model alignment and fine-tuning. Recently, fine-tuning judges with judge-specific data has emerged as an often preferred choice over directly prompting frontier models as judges, as the former achieves better performance with smaller model sizes while being more robust to common biases. However, the standard evaluation ignores several practical concerns of fine-tuned judges regarding their real-world deployment. In this paper, we identify and formalize three aspects that affect the *shelf life* of these judges: *future-proofing* and *backward-compatibility* $-$ how well judges fine-tuned on responses by today's generator models perform on responses by future models or past models, as well as *question generalization* $-$ how well judges generalize to unseen questions at test time. We study these three aspects under a unified framework with varying train and test distributions in two reasoning datasets, three SFT- and DPO-based fine-tuning algorithms, and three different backbone models. Experiments suggest that future-proofing is challenging for most models, while backward-compatibility is relatively easy, with DPO-trained models consistently *improving* performance. We further find that continual learning provides a more balanced adaptation to shifts between older and newer response distributions than training solely on stronger or weaker responses. Moreover, all models exhibit some degree of performance degradation when moving from questions seen during training to unseen ones, showing that current judges do not fully generalize to unseen questions. These findings provide insights into practical considerations for developing and deploying judge models in the face of ever-changing generators.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the durability (“shelf life”) of finetuned LLM-as-judge models under realistically inspired distribution shifts. The authors introduce a clear dual-distribution view of automatic evaluation that factorizes a judge’s inputs into a triplet of question and pairwise responses (one correct, other incorrect, but importantly both sampled from the same model). Within this frame, they formalize three practical axes: future-proofing (weak→strong generator shift), backward compatibility (strong→weak), and question generalization (seen→unseen questions). They instantiate the study in math (DeepScaleR, olympiad-style problems) and compare three judge-training recipes (SFT, DPO, SFT+DPO) across three base models (Llama-3.1-8B, Ministral-8B, Mistral-24B). The authors find that: (i) judges trained on “weak” responses do not future-proof—performance drops on “strong” responses; (ii) retraining on “strong” responses yields consistent gains (especially with DPO); (iii) strong-trained judges are fairly backward-compatible on weak responses; (iv) continual training balances both directions better than training from scratch; and (v) question generalization is weak—performance drops notably on unseen questions.

### Strengths
### Originality

The paper tackles an important question of *shelf-life* of judge models which I have not seen addressed before in the community. It is a question of increasingly growing importance as the field of LLM-as-a-judge matures. The paper proposes intuitive, measurable quantities—FutureProof, RefreshAdvantage, BackCompatibility, and CompatibilityShift—to quantify judge robustness under evolving generators and questions. These metrics translate deployment concerns into experimentally testable criteria.

### Quality

1. Systematic experimentation: The study spans three judge bases (8B and 24B), three recipes (SFT, DPO, SFT+DPO), and multiple train/test splits (seen vs. unseen questions; weak vs. strong response pools). 
2. Clear empirical takeaways: Results are easy to interpret -- future-proofing is hard (negative FutureProof across the board), retraining on strong responses pays off (positive RefreshAdvantage, largest for DPO), strong-trained judges are mostly backward-compatible, continual learning narrows gaps in both directions, and unseen-question performance drops notably.
3. Sound dataset engineering. Responses within a pair come from the same generator to avoid style-based shortcuts; base judge is excluded as generator to reduce self-bias; and response-order flipping is used for evaluation. These are thoughtful choices.

### Clarity

The paper is generally well-structured: motivation→framework→metrics→experiments→implications. Metric definitions are explicit and the content is easy to follow.

### Significance

This work can have practical impact on evaluation pipelines (though currently initiated only in math). The core message—retrain judges with up-to-date strong generator outputs (DPO works best)—is actionable but not surprising. The finding that strong-trained judges can substitute older judges with minimal losses reduces migration risk.

### Weaknesses
W1: My main concern is with the fact that authors choose only math as the domain for evaluation. For math, we already have rule-based verifiers to check answers so a judge model is not needed (unless used for step-by-step level check).  Only DeepScaler dataset is used for evaluation. Given that only a single (narrow) domain is studied in the paper, the results may not transfer to more open-ended or subjective tasks (e.g., non-math domains, creative writing). I would consider increasing my score if the authors can add at least 2 other domains (eg: science domains, or mathematical proofs if sticking just to math).

W2: Reliance on a more capable teacher for supervision. SFT/DPO training uses teacher-generated CoT/verdicts; details indicate prompting GPT-4o model for supervisory signals. When studying future compatibility, it is possible that old judges are being used to evaluate frontier models in which case no "stronger" judge exists and it might not be possible to get training data for the judge on responses from strong (newer) models. Thus, the proposed solutions of "retraining on stronger data" or "continual learning" might be infeasible [1]. Do the authors have any approaches for how to address this bottleneck?

W3: The results are mostly expected apriori and not super insightful. For example, the performance drop on “Unseen questions” is not surprising at all. Given that the models were never trained for them, it is expected that the performance of the judges on those questions will be less than the seen ones as they are OOD. Similarly, judge models becoming better with continual learning is also not surprising as that version learns from both strong and weak data compared to other judges which were training only one of the two distributions. 

W4: No prompt templates are provided on how the judge model was presented with the responses (was it the whole response or just the final answer?)

W5: Nothing is mentioned about the how the responses were sampled (top_p, temperature values).

## Presentation nits: 

1. Some figure y-axes are labeled “Metric” without units; the sign conventions (negative=degradation) require mental mapping. Consider captions with concrete units (percentage-point deltas of consistent accuracy).
2. Table 2 repeats “Base (zero-shot)” blocks per model; consider a single consolidated row to reduce clutter.
3. Fig.1 is hard to follow with the arrows. 
4. It is not clarified which models are strong and which models are weak in the main paper. While this is mentioned in the Appendix, this should also be stated in the main paper. 

References:

[1] Dorner, F. E., Nastl, V. Y., & Hardt, M. (2024). Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data. arXiv preprint arXiv:2410.13341.

### Questions
Q1: It is a bit strange that base models were finetuned? Why not use Instruct version of the models? What is the justification for using base models?

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
This paper studies the robustness and “shelf life” of finetuned LLM‑as‑judge models. The core contribution is a dual‑distribution formalization. This framing lets the authors isolate and measure three practically important dimensions: future‑proofing (judges trained on older/weaker generator outputs evaluated on newer/stronger outputs), backward compatibility (the reverse), and question generalization (performance on unseen questions). They define clear metrics—FutureProof, RefreshAdvantage, BackCompatibility, and CompatibilityShift—and evaluate across three base models (Llama‑3.1‑8B, Ministral‑8B, Mistral‑24B) and three training recipes (SFT, DPO, SFT+DPO) on a math domain with verifiable labels. This paper provides more key findings and provides a unifying view and actionable guidance for deploying judge models.

### Strengths
1. This paper investigates some key questions in the LLM-as-a-Judge domain with interesting findings & takeaways.
2. Extensive and commonly used setups are evaluated and conducted within the math domain.

### Weaknesses
1. All experiments are conducted in the maths domain; adding at least one open-ended question would broaden the scope of this paper. This is the major concern of this paper: are we drawing conclusions from a special case?
2. Current experiments only study one round of weak-to-strong. It would be better to scale to multiple rounds.
3. Currently, all experiments only compare effectiveness; adding efficiency consideration would be better.
4. The paper could benefit more from reading if it were reorganized into a more expressive form. As this paper is not a classic paper, it may not necessarily follow the intro-method-experiment setup.

### Questions
See cons

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies the “shelf life” of finetuned LLM judges under distribution shifts. Concretely, it analyzes (i) future-proofing (weak→strong generators), (ii) backward compatibility (strong→weak), and (iii) question generalization (seen→unseen) in a unified math-domain setup, using pairwise judging data with gold answers and three training recipes (SFT, DPO, SFT+DPO). Main findings: future-proofing is consistently hard; backward compatibility is relatively easy; continual training helps balance the trade-offs; and all judges degrade on unseen questions.

### Strengths
1. The paper clearly defines the problem and provides good metrics. This sets a strong foundation for future-proofing the system, maintaining backward compatibility, and generalizing to new types of questions.

2. A key strength is the paper's choice of a verifiable math domain. This decision smartly avoids the common problem of messy or inaccurate data, and their specific evaluation method further strengthens the results by helping to reduce positional bias.

3. The study's approach of separating "weak" and "strong" generators, along with the continual-training analysis, offers practical guidance on how to roll out updates to "judge" models in a real-world setting.

### Weaknesses
1. A significant limitation is that the training method (DPO) and the main test (pairwise accuracy) are fundamentally aligned. The model is, in effect, being tested on the same terms for which it was trained. This gives DPO a built-in advantage that doesn't necessarily prove it's a better judge in a general sense. The paper is missing other kinds of tests (like direct scoring) that would be needed to separate this effect and make a fairer comparison.

2. The paper's main conclusions about "shelf life" are drawn entirely from the domain of mathematics. This is a problem because math, with its clear right answers, is not a good stand-in for complex, open-ended tasks like summarization or dialogue. These real-world tasks involve messy data, complex goals, and shifting user preferences, where all factors that would likely worsen shelf-life decay, not lessen it. The paper's claims are too broad for its evidence, which is limited to a single, clean domain and a simulation of time rather than real-world data.

### Questions
See weakness

### Soundness
3

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
4

### Summary
This paper studies the “shelf life” of finetuned LLM judges—how well they keep working as generators evolve and as questions change. The authors introduce a dual-distribution view of judge inputs that separates the question distribution Q from the response distribution R and formalize the judge input as X = Q \times R \times R for pairwise judgments. They then operationalize three axes of robustness: future-proofing (weak-trained judges evaluating stronger/newer responses), backward compatibility (strong-trained judges evaluating older/weaker responses), and question generalization (seen vs unseen questions). Experiments are in math, using DeepScaleR questions with verifiable answers, three base judges (Llama-3.1-8B, Ministral-8B, Mistral-24B), and three recipes (SFT, DPO, SFT+DPO), with performance measured by consistent accuracy. Main findings: future-proofing is generally poor (performance drops on newer responses), re-training on strong responses reliably helps (especially with DPO), backward compatibility is mostly fine, continual training narrows the gap between weak-only and strong-only training, and question generalization is limited (drops on unseen questions).

### Strengths
Overall, this is a timely and practically important study for anyone deploying LLM judges. The dual-distribution framing is clean and useful, and the metric design aligns well with the questions practitioners actually face. The experimental results are clear and—importantly—actionable (e.g., prefer DPO and keep judges refreshed with up-to-date strong responses). I see this as a strong empirical contribution. With some clarifications on novelty, dataset construction choices, and statistical reporting, I would lean toward acceptance and could increase my score.

### Weaknesses
See Questions

### Questions
1. You group generators into “weak/strong” by pass@1 bands (~0.25 vs ~0.45). Please spell out thresholds and sensitivity: do conclusions hold if the bands shift or are defined per-dataset/epoch?  

2. You only pair a correct and an incorrect response from the same generator to avoid stylistic confounds. This is reasonable; please also quantify how many questions are discarded due to lack of both labels and discuss any induced selection bias toward “borderline” items.

3. Standardize “future-proofing” vs “future proofing” across abstract/body.

### Soundness
3

### Presentation
3

### Contribution
3
