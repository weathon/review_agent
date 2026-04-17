# Learning Composable Chains-of-Thought

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4, 4

## Abstract
A common approach for teaching large language models (LLMs) to reason is to train on chains-of-thought (CoTs) of in-distribution reasoning problems, but such annotated data is costly to obtain for every problem of interest. We want reasoning models to generalize beyond their training distribution, and ideally to generalize compositionally: they should combine atomic reasoning skills to solve harder unseen tasks. In this paper, we introduce a method to enable generalization to a target compositional task that has no labeled CoT data. We find that simply training models on CoT data of atomic tasks leads to limited generalization, but minimally modifying CoT formats of constituent atomic tasks to be composable leads to improvement. Specifically, we augment our data by adding prefixes to CoTs, making sequences of CoTs in-distribution for the trained model. We train individual models on the atomic tasks with composable CoT data and combine them with multitask learning or model merging to address the target compositional task zero-shot. This model can be further trained on a small amount of compositional data using rejection sampling fine-tuning (RFT). Results on three domains of compositional tasks, natural language skills, string manipulation, and arithmetic, show that training LLMs on Composable CoT outperforms multitask learning and continued fine-tuning baselines within a given training data budget.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a method for enabling LLMs to generalize to compositional reasoning tasks without using compositional chain-of-thought (CoT) data. By augmenting CoTs of atomic tasks with composable prefixes, the model learns to combine reasoning steps across tasks. Experiments across natural language, string manipulation, and arithmetic show that training on Composable CoT data outperforms multitask and fine-tuning baselines under the same data budget.

### Strengths
- This study conducts an interesting and important research topic: generalization to compositional reasoning tasks by using only atomic reasoning data at training.
- The experiments demonstrate the effectiveness of the proposed method though they are toy experiments.
- The paper is well-written and easy to follow.

### Weaknesses
- I think we need an additional ablation study about why the simple trick (adding just random tags) works. For example, which is more important, tag or random text? what if we remove or change the tag? what if we change the style of random text?

### Questions
- Table1: Does the result not apply RFT(rejection sampling fine-tuning)? I think there is no such description, but based on the context I feel so.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a data augmentation method that adds random prefixes to atomic CoT steps to improve compositional generalization. The idea is evaluated on synthetic reasoning tasks such as string manipulation, arithmetic, and Skill-Mix.

### Strengths
The proposed method is simple and shows significant improvements on synthetic tasks.

### Weaknesses
1. The paper only conducts experiments on synthetic tasks. It is unclear how the proposed method can be applied to real-world scenarios such as math reasoning or code generation. In particular, identifying the atomic tasks in these domains is non-trivial. I also doubt whether simply data augmentation without explicitly training the model for composition can lead to meaningful improvements on realistic tasks.
2. I am uncertain about the broader impact of this work. To advance the frontier of model's reasoning capabilities, the current standard practice is to use reinforcement learning. When compute is limited,  distillation from stronger LLMs also achieve good result. I do not see clear evidence that the proposed method would bring better results on real-world tasks compared to these existing approaches. It would be helpful if the authors could discuss how their idea might be applied in more realistic training setups or combined with current methods.
3. The discussion of related work is insufficient. The authors should at least include: (1) recent methods for training models to generate long CoTs, such as distillation , self-training [1], or reinforcement learning[2]; and (2) broader discussions on understanding and improving compositional generalization in LLMs.

[1] Zelikman et al. STaR: Bootstrapping Reasoning With Reasoning

[2] Guo et al. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

### Questions
1. Why is the performance of SFT with compositional supervision so low on the first three tasks (Table 1)? It would be helpful if the authors could provide more details on the setup of this baseline, including examples of training samples.
2. It is somewhat surprising that augmenting data with random prefixes improves model performance, given the significant distribution shift between training and testing. Do the authors have an explanation for that? In addition, I am curious whether such training leads to a larger degradation on other capabilities (e.g., instruction following) compared to other baselines.
3. For ComposableCoT-Merge, how is the scaling factor determined?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper claims a way to make chain-of-thought reasoning "composable": train on atomic tasks, add random "proxy prefix" strings in front of the CoT, tag them, then fine-tune the LLM so it learns to generate composable CoTs conditioned on the prefix. At test-time, there is a  concatenate strategy at the task level (via multitask learning or model-merging). On simple synthetic tasks: string ops, ASCII arithmetic, toy natural-language skills, the authors show often big improvements over the non-composable CoT strategy but only at the level of 2 compositions. Improvements wrt control baselines for 3 compositions is also observed, also to a lesser extent

### Strengths
1. The problem of compositional generalization is a core ML problem and I appreciate the authors trying to address it in the modern setting: standard LLMs have been shown to be incapable of large scale compositional reasoning. This approach is interesting and tries to leverage CoT that has been shown to help with logical/math reasoning for compositional generalization. 
2. Well written and motivated empirically.
3. Strong results of the proposed approach over baselines are interesting to see, though notably janky at many places especially table 1 llama 2.

### Weaknesses
1. The main weakness of this paper is that it doesn't experiment with compositionality enough. The tasks are also not compositional enough and there is a risk of templating/pattern-matching hacking going on here. Symbolic manipulation of some task with quantifiable controllable compositionality (e.g. n digit multiplication. Multiplication of n digits k times) would be interesting to see. 2 way compositional results are interesting to study as a starter but you should not stop at 3 way compositions. What about k-way compositions? I already see the improvement delta declining at 3 compositions.
2. Results for small models have high variance. Table 1 llama and qwen 7b have a huge performance delta. The benefits of the proposed approach are unclear, especially in light of 1.
3. Results on larger models are missing. Presumably compositional CoTs should yield even better performance deltas for larger models e.g 30b, 70b?
4. Proxy prefixes can also be interpreted as random noise and this approach feels rather like prompt regularization. The experimental results again don't investigate the compositional utility of the approach.
5. Just using thinking tokens ala., Deepseek R1 distillation, and seeing performance effects is unexplored. This is also a clear baseline.

### Questions
see above

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
3

### Summary
The paper proposes Composable CoT, a lightweight data augmentation that wraps atomic CoT traces with tags and inserts “proxy prefixes” (random-letter strings) so that training examples look like “a CoT conditioned on previous CoTs.” At test time, atomic CoT models are combined either via multitask learning (MTL) or Task Arithmetic–style model merging; limited compositional supervision is optionally added through rejection-sampling fine‑tuning (RFT). On synthetic string/arithmetic compositions and on Skill‑Mix (literary+rhetorical skills), Composable CoT improves zero‑shot and small‑shot composition over standard CoT training, with the largest reported gains coming from RFT on top of the composable formatting.

### Strengths
Turning atomic CoT data into a “composable” format is easy to implement (two tags + random‑letter prefixes) and consistently lifts zero‑shot/limited‑shot composition across tasks and two 7B bases. The construction is clearly depicted (Fig. 2), and ablations show random letters are the most robust proxy prefix out‑of‑domain.

### Weaknesses
1. "Zero-shot" claim is fragile due to validation-time merging sweeps. For Task Arithmetic, the paper sweeps $\alpha, \beta$ on a validation set for each task (App. G.4). If this validation set is the compositional task, tuning leaks target supervision into model selection and weakens the zero-shot claim; at minimum this needs to be clarified and a version without compositional validation should be reported.
2. Heavy reliance on explicit tags and random prefixes; external validity is limited. The method depends on <prefix>/<suffix> markers and training on random-letter prefixes. This encourages learning a format protocol rather than discovering composition in untagged, natural inputs. The paper itself instantiates all intermediate tags as <prefix> and the final as <suffix> (Instantiation of Tags), underscoring the dependence on explicit scaffolding. Results without any tags or with naturally occurring preceding CoTs are not shown.
3. Evidence of reading and re-using intermediate results is weak. The "quality" analysis checks whether both atomic CoT templates appear, not whether the model actually consumes the earlier step's outputs (step-level causal dependence or variable-passing). This leaves open the possibility that the model just regenerates both CoTs in the suffix.
4. Many findings are unsurprising and the novelty is modest. The main lift plausibly comes from making multi‑CoT sequences in‑distribution via formatting. The work’s practical impact beyond controlled compositions may be limited without evidence on untagged, naturally compositional tasks (e.g., program‑of‑thought planning, tool‑use pipelines).

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
