# WorldPM: Understanding Scaling Patterns in Human Preference Modeling

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Motivated by scaling laws in language modeling that demonstrate how test loss scales as a power law with model and dataset sizes, we find that similar laws exist in preference modeling. We propose **World** **P**reference **M**odeling (WorldPM) to emphasize this scaling potential, where World Preference embodies a unified representation of human preferences. In this paper, we collect preference data from public forums covering diverse user communities, and conduct extensive training using 15M-scale data across models ranging from 1.5B to 72B parameters. We observe distinct patterns across different evaluation metrics: (1) Adversarial metrics (ability to identify deceptive features) consistently scale up with increased training data and base model size; (2) Objective metrics (objective knowledge with well-defined answers) show emergent behavior in larger language models, highlighting WorldPM's scalability potential; (3) Subjective metrics (subjective preferences from a limited number of humans or AI) do not demonstrate scaling trends. Further experiments validate the effectiveness of WorldPM as a foundation for preference fine-tuning. Through evaluations on 7 benchmarks with 16 subtasks, we find that WorldPM broadly improves the generalization performance across human preference datasets of varying sizes (7K, 100K and 800K samples), with performance gains exceeding 5% on many key subtasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates whether scaling laws extend to preference modeling and proposes World Preference Modeling (WorldPM). The authors collect 15M forum-based preference pairs and train models from 1.5B–72B parameters, observing adversarial and objective scaling effects but no scaling on subjective tasks. Additional analysis shows style bias decreases as scale increases. WorldPM further improves downstream preference fine-tuning.

### Strengths
1,Timely direction linking scaling laws to preference modeling and alignment.

2,Structured evaluation across adversarial/objective/subjective categories.

3,Style-bias analysis offers interesting behavioral insight.

4,Demonstrated benefit for downstream preference fine-tuning.

### Weaknesses
1,Conceptual ambiguity of “World Preference”
The notion of a universal preference representation remains loosely defined and not theoretically grounded; the leap from forum votes to global human preference is not justified.

2,Data reliability and bias concerns insufficiently addressed
Forum voting signals carry demographic, social-amplification, and stylistic biases, but the paper provides limited noise analysis or validation that such signals represent stable human preferences.

3,Subjective preference analysis remains shallow
Style-bias evaluation relies on narrow proxies (length/markdown), and the failure of subjective scaling is discussed descriptively without deeper causal study or alternative explanations (e.g., benchmark noise or value heterogeneity).

4,All scaling results are reported only for Qwen2.5 models (1.5B–72B).The authors must explicitly limit claims about universality and (ideally) add at least a small cross-family check (e.g., a single smaller model from a second family) or argue why Qwen2.5 is representative.

5,The StackExchange dataset used for preference modeling may partially overlap with samples included in downstream benchmarks.  Explicitly verifying and reporting cross-dataset overlap, or constructing benchmark splits disjoint from StackExchange data, would be essential to confirm the validity of the reported improvements.

### Questions
1,Can you provide a more formal definition of “World Preference” and its theoretical assumptions?

2,How reliable are forum votes as preference labels? Any human validation or noise modeling?

3,Why only length/markdown as style proxies? Are other stylistic or cognitive factors measured?

4,Have you validated scaling behavior on non-forum datasets or multilingual preference corpora?

### Soundness
3

### Presentation
2

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
The paper investigates scaling laws in human preference modeling. The authors propose World Preference Modeling (WorldPM), leveraging large-scale preference datasets and training models with 1.5B to 72B parameters. Through systematic experiments, the paper demonstrates the scaling laws of adversarial, objective, and surjective metrics. Additional analyses probe the impact of stylistic bias and test the utility of WorldPM initialization for downstream preference fine-tuning across multiple benchmarks, demonstrating extensive generalization improvements.

### Strengths
- This paper addresses an important and timely problem: scaling human preference learning using large models and publicly sourced training data.
- The authors uncover an interesting phenomenon: objective and subjective evaluations exhibit different scaling behaviors as model size and data increase.

### Weaknesses
- The notation $Z$ in Equation (3), representing the style difference, appears to be inconsistently defined. In L249, the authors state that $Z\in\mathbb{R}^{S}$ is the style features vector.
- The evaluation of stylistic features is limited to four attributes (token length, markdown lists, headers, and bold text), overlooking potentially important stylistic dimensions such as syntactic complexity.
- It is somewhat difficult to assess the technical contribution of this work. While identifying scaling laws in preference modeling is valuable, the paper does not clearly articulate the specific technical challenges that WorldPM addresses beyond the empirical observation of scaling behavior.

### Questions
- During the data collection phase, upvotes provide a weak and noisy aggregate signal of human preference. Given that users may upvote posts for reasons unrelated to objective quality, could the authors clarify how these noisy signals are filtered or how the model can learn a clear preference signal from this noisy data?
- In Equation (3), how to derive the style features vectors $Z(y_0)$ and $Z(y_1)$ for responses $y_0$ and $y_1$?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes WorldPM, a large-scale preference‑model (PM) pre‑training setup built from forum‑sourced preference pairs (primarily StackExchange) and uses it to study how PM performance scales with both data size (up to ~15M pairs) and base model size (Qwen2.5 family, ~1.5B→72B). The authors report a three‑regime picture: (i) on adversarial PM evaluations, error decreases smoothly with more data/model size and admits power‑law fits; (ii) on objective, exactly‑graded tasks, improvements appear to be “emergent,” materializing only for larger base models; and (iii) on subjective, LLM‑judge benchmarks, headline scores do not cleanly improve with scale, which they attribute to style biases (e.g., response length, Markdown). They also introduce a simple “style‑control” adjustment that linearly regresses judge preferences on style features and an RM score difference, and they show that initializing downstream PM fine‑tuning with WorldPM yields consistent gains on several benchmarks.

### Strengths
I think this is a timely and useful empirical study. The originality is not a new algorithm but a comprehensive, carefully run scaling investigation specifically for PM pre‑training. That niche—reward/preference models rather than policies—has fewer systematic, large‑sweep studies, so mapping it out has value.

On quality, the experimental breadth is strong: multiple model sizes and data scales and a diverse evaluation suite spanning adversarial, objective, and subjective categories. The three‑regime picture is practically meaningful, and it affects how one might allocate PM compute and data depending on the target use. I also appreciated the diagnosis of style bias in LLM‑judge evaluations and the attempt—however imperfect—to factor it out. The downstream transfer result (WorldPM → better fine‑tuned PMs on objective tasks) is a concrete takeaway that practitioners can use.

Clarity is generally good: figures make the high‑level trends easy to grasp, and the paper is upfront about evaluation choices and their rationale. In terms of significance, I believe many groups will consult this paper when deciding whether to invest in PM data vs. scaling the base model vs. changing evaluation protocols; that makes it a useful reference point even if some conclusions confirm community intuitions.

### Weaknesses
1) Positioning and novelty claim need tightening (and up‑to‑date citations).  
Several headline messages echo things the community already suspected: “more/bigger data helps,” “LLM‑judge metrics are style‑sensitive,” and “objective metrics tend to be more robust than subjective ones.” The value of the paper is the scope and quantification, not the qualitative direction. That’s fine—but the Related Work should be sharper and current. As submitted, I did not see a single 2025 reference, which is not acceptable for a September‑2025 submission in a fast‑moving area. The paper should explicitly situate itself against closely adjacent lines (reward‑overoptimization/scaling in RLHF on the policy side; recent generative reward models/RLAIF‑style RMs; 2025 analyses of LLM‑as‑judge biases and debiasing). A small related‑work matrix contrasting what scales (pairs vs. compute vs. base capacity), data provenance (human/AI/forum), and evaluation types would help calibrate what is genuinely new here.

2) Style‑control methodology risks leakage/over‑correction.  
The linear adjustment re‑learns the weight on the RM score difference together with style features on the evaluation distribution. Without cross‑fitting or a held‑out split for learning the correction, the adjusted metric can absorb peculiarities of the test set and artificially “fix” rankings. I strongly recommend cross‑fit evaluation (learn the adjustment on split A, report on split B) and reporting both the raw and cross‑fit‑controlled results, plus a sensitivity analysis showing whether system rankings actually stabilize.

3) Some core results feel “obvious,” and the paper should own that and sharpen the non‑obvious parts.  
I don’t want to undermine the work—running these experiments well is non‑trivial—but parts of the story are indeed expected (bigger PMs + more data → better adversarial outcomes; subjective metrics correlate with length/format). I suggest reframing the contribution more explicitly as measuring the shape and turning points of these curves, and highlighting what is truly non‑obvious (e.g., the “objective emergence” only beyond a base‑model size threshold; the apparent stability of power‑law exponents across tasks).

5) Batch‑size ablation doesn’t isolate the variable of interest.  
Conclusions about “larger batch helps” are drawn under a protocol where total tokens increase with batch (fixed steps). That largely measures “more data helps.” Please re‑run with matched total tokens (or matched FLOPs/compute) to support any claim about batch itself.

6) Power‑law fits need better statistical treatment.  
Power‑law exponents are reported over averages of a few benchmarks, with limited goodness‑of‑fit diagnostics. Per‑task fits with confidence intervals, checks for sensitivity to task inclusion, and compute‑normalized views would make the “scaling law” claims more robust.

8) Notation and mathematical correctness issues that need fixing.  
I’m flagging three concrete items that should be corrected or clarified:
* I don't like the use of $D^\top \alpha$ when $D$ is introduced as a scalar. It might be a matter of personal preference, but I would rather see $ D \alpha$ in this case.
* In Equation 3, a fraction is written with $Z$ in the numerator and denominator while $Z$ is defined as a vector. As written, the expression is not well‑typed. Please clarify (for example if it is meant to be an element-wise operation) or use better notations.
* In the appendix illustrating chosen vs. rejected responses, the color theme appears inverted (green for rejected, red for chosen) for some of the examples. Please swap or label more clearly; this kind of visual miscue invites reader error.

### Questions
Every item raised in the Weaknesses section can be viewed as a question for the authors.
I may well be mistaken on several of these points, and I would sincerely appreciate clarification or correction wherever appropriate.
If the authors can address or resolve even part of these concerns—whether by showing that I misunderstood something or by providing additional detail—it would be very helpful.

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
4

### Summary
This paper introduces World Preference Modeling (WorldPM), a large-scale pretrained preference model trained on hundreds of billions of preference-aligned samples across multiple domains. The authors argue that large-scale pretraining enables the model to acquire broad and generalizable preference knowledge, thereby simplifying downstream preference learning and improving generalization. Overall, the work presents an impressive large-scale effort that aims to lay the foundation for scalable preference modeling.

### Strengths
1.The motivation of the paper is strong and timely, as building a general-purpose preference model is a natural next step in advancing alignment research.

2.The authors have conducted a technically demanding and large-scale project, involving extensive data collection, training, and evaluation.

3.The results demonstrate that large-scale preference pretraining can improve model generalization, reduce stylistic bias, and accelerate downstream alignment tasks.

### Weaknesses
1.Ambiguity remains in defining what World Preference Learning actually covers, leaving unclear whether it includes physical, visual, auditory, or embodied modalities.

2.Insufficient clarification is provided for the adversarial setting, which can have different meanings across domains and therefore needs a more precise explanation.

3.Related work lacks a coherent structure and fails to connect this study with foundational methods such as RLHF, DPO, GRPO.

4.Analysis of results stays largely descriptive, with key claims—such as reducing stylistic bias through scale—unsupported by quantitative or causal evidence.

5.Potential reward hacking and overfitting to stylistic features are mentioned but never thoroughly examined, leaving uncertainty about what biases remain after pretraining.

6.Discussion of annotator bias and value alignment is missing, despite their central importance to subjective preference modeling.

7.Impact of contradictory preference labels on models of varying scales is unexplored, preventing insight into how model capacity mediates robustness under conflicting supervision.

### Questions
1.How is “World Preference Learning” formally defined, and does it extend to multimodal or embodied forms such as video, physical interaction, or speech?

2.What exactly does the term “adversarial” mean in the context of this paper—does it refer to adversarial prompting, preference conflict, or robustness evaluation—and how was this setting implemented in experiments?

3.Could the authors expand the related work section to include a clear taxonomy of previous preference modeling and alignment methods, highlighting how WorldPM differs from or builds upon prior frameworks such as DPO, GRPO?

4.The statement that models initially rely on stylistic features but reduce this bias with more data is interesting, but can this trend be quantitatively demonstrated, for example through feature attribution or bias disentanglement analysis?

5.If the data contains irrelevant or misleading features correlated with preference labels, does the model eventually unlearn these patterns with scale, or does large-scale pretraining risk amplifying such reward-hacking behavior?

6.For subjective datasets, how are annotators trained to ensure value consistency, and have the authors analyzed whether differences in cultural or reasoning styles influence the learned preference distribution?

7.Has the paper examined how models of different sizes handle contradictory preference data, and whether there is a threshold of conflict severity that causes accuracy degradation?

8.Would the proposed framework generalize to multimodal or embodied preference learning tasks, and if so, what adjustments would be required to extend it beyond text-based preferences?

### Soundness
3

### Presentation
3

### Contribution
3
