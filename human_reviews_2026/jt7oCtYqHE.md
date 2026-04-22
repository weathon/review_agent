# From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
Does reinforcement learning (RL) teach large language models (LLMs) genuinely new skills, or does it merely activate existing ones? This question lies at the core of ongoing debates about the role of RL in LLM post-training. On one side, strong empirical results can be achieved with RL alone even without preceding supervised finetuning; on the other, critics argue that RL contributes little beyond reweighting existing reasoning strategies. This work provides concrete evidence that LLMs can acquire genuinely new skills during RL by composing existing ones, mirroring one of the central mechanisms by which humans acquire new cognitive skills \citep{Anderson1982Acquisition}. To mitigate data contamination and other confounding factors and to allow precise control over task complexity, we develop a synthetic framework for our investigation. Specifically, we define a skill as the ability to infer the output of a string transformation function $f(x)$ given $x$. Once an LLM has already learned $f$ and $g$ prior to RL, our experiments reveal that RL enables it to learn unseen compositions of them $h(x)=g(f(x))$. Further, this compositional ability generalizes to more difficult problems such as compositions of $>2$ functions unseen during training. Our experiments provide surprising evidence that this compositional ability, acquired on the source task, transfers to a different target task. This transfer occurs even though the model has never trained with RL on any compositional problems in the target task, as long as it has acquired the target task's atomic skills prior to RL on the source task. Our qualitative analysis shows that RL fundamentally changes the reasoning behaviors of the models. In contrast, neither of the findings is observed in next-token prediction training with the same data. Our systematic experiments provide fresh insights into the learning behaviors of widely-used post-training approaches for LLMs. They suggest the value of building base models with the necessary basic skills, followed by RL with appropriate incentivization to acquire more advanced skills that generalize better to complex and out-of-domain problems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper argues that LLMs can acquire genuinely new skills during RL by composing existing ones. 
Using a controllable synthetic framework, this paper shows that LLMs compose learned string transformation functions via RL, and such compositional capability can transfer to different tasks. The paper also identifies that the key is to do RL on compositional data which incentivizes the compositionality.

### Strengths
- Studying whether RL geniunely teaches new skills is a very timely topic.

- The controllable framework plus decontamination procedure enhances the soundness of the conclusion.

- The paper is well-written and easy to read.

### Weaknesses
1. While the the paper provides a good controllable framework to understand what's happening during RL, I wonder what the actionable point is for future works. For example, Takeaway 3 implies that, one could hand-design (or synthesize) some compositional samples manually and controllably (just like the string transformation functions proposed), and train on those samples to incentivize a better compositional reasoning in more practical domains (than the countdown task) like code or math. I hope the authors can comment on this front or add some experimental results (even preliminary) in the paper.

2. The paper claims that RL on compositional data can teach compositional skills, 'if the atomic skills are already learned'. It's not clear that whether the condition is necessary. What if training RL directly on a mixture of atomic data and compositional data (thus without the stage 1)? Would RL be able to learn atomic and compositional skills together? I understand this might make experimental setup more complicated, such as leaking the function definitions, but maybe good to add a discussion on this.

### Questions
1. I wonder how is the conclusions connect with the enropy/diversity view of RLVR?


2. What is the specific reason of setting the coefficients for both KL divergence and entropy loss to 0?


3. How do you ensure that RL and RFT in stage 2 are in a fair comparison? How are the hyperparameters selected, do both see a same amount of training samples? These questions seemed not to be reported.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates whether reinforcement learning (RL) genuinely teaches new reasoning skills to large language models (LLMs). Using a synthetic, contamination-free string transformation benchmark, the authors define “atomic” and “compositional” skills and show that RL with compositional data enables models to generalize to deeper compositions unseen during training. In contrast, rejection fine-tuning (RFT) or RL on atomic data fails to do so. The learned skill also transfers modestly to a different task (Countdown) when atomic skills are present. Behavioral analyses reveal that RL changes reasoning patterns rather than merely reranking outputs. Overall, the paper argues that RL can create new compositional reasoning abilities if properly incentivized.

### Strengths
The question (“does RL truly teach new skills?”) is central and timely.

The synthetic setup is impressively controlled and eliminates confounders such as data contamination.

Clear evidence that RL + compositional data leads to new behaviors unseen in supervised baselines.

Strong writing quality and interpretability of experiments (the “Takeaway” summaries are exemplary).

### Weaknesses
The paper’s analysis mostly rely on synthetic string transformations. While clean, the setting is far from real RL pipelines. There’s no guarantee that the same mechanism (composition of atomic skills) is what happens in open-ended reasoning tasks.

The paper defines a “new skill” as the ability to compose existing ones, but that might not satisfy the broader notion of novel reasoning or conceptual abstraction. In essence, this could still be pattern recombination rather than true skill learning. It is still possible that these compositionality is stored in the base model, and RL is just a better way to rediscover that.

### Questions
See weakness, thank you

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper argues that reinforcement learning (RL) can teach large language models genuinely new reasoning abilities by composing existing skills, not just reweighting old ones. Using a clean string-transformation testbed, the authors show that when models first master atomic skills and then receive RL on compositional tasks, they generalize to unseen and deeper compositions. The learned compositional skill even transfers to other domains like arithmetic reasoning, provided the model already has the basic atomic knowledge.

### Strengths
The paper offers an original and well-grounded perspective on the role of RL in LLMs, framing it as a process of compositional skill acquisition rather than simple reward reweighting. Its experimental design is exceptionally clean and controlled, allowing for causal interpretations. The writing is clear and logically structured, with a thoughtful connection to cognitive theories of human skill formation that strengthens its conceptual depth. The results are robust and systematically analyzed, making a compelling case that RL can produce genuinely new reasoning capabilities.

### Weaknesses
While the paper is conceptually strong, several aspects could be improved to strengthen its empirical and interpretive depth. First, the experimental validation is confined to synthetic string tasks and the Countdown domain which is clean but showing similar compositional learning in more natural reasoning settings would make the claims far more compelling.

### Questions
To what extent does model scale influence the emergence of compositionality? It would be helpful to know whether the same behaviors hold for smaller or larger models, as this would indicate whether compositional skill learning is a general property of RL or dependent on scale.

### Soundness
3

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
The paper proposes an evaluation of an LLM on compositional coding tasks, mainly to understand what extra value RL actually adds to compositional task performance.
Overall idea is interesting, but the clarity and the depth of analysis are somewhat limited.

### Strengths
1. The paper identifies and designs a coding-based compositional task setup that seems to probe model behaviour in a somewhat controlled way.

2. Experiments do show (at least in some parts) that RL helps improve compositional performance, esp. when moving from simpler to more complex tasks.

3. Discussion of prior work and related efforts is well-placed.

### Weaknesses
1. The novelty feels rather limited — much of the evaluation and task protocol seems adapted from existing works. The only clear “new” thing is using RL for level-1 and level-2 training, which looks like a small extension.

2. Only one model (LLaMA-3.1) is used, and with just 1-2 kinds of tasks (string manipulation and countdown types). That’s too narrow to make strong claims.

3. The plots mostly differ only by acc., avg@16, and pass@k numbers... also why choose k = 1000 ? feels arbitrary.

4. It’s also unclear why RL-based training reduces the basic atomic skill performance — this is quite puzzling and not explained. Figure 6 and its description comes across as this failure mode is "Acceptable".

5. No theoretical or mathematical grounding provided, which makes the claims feel a bit unsupported.

6. The paper doesn’t provide a limitations section.

### Questions
1. The writing quality can definitely be improved — there are several typos, some awkward phrases (“perform decently well” or “may remain limited new skills” etc.) that are confusing.

2. Would be nice if the focus didn’t stay only on compositional tasks… maybe discuss generalization to unseen atomic tasks too.

3. The connection to in-context or few-shot learning isn’t clear; authors do not mention how this fits into this line of work.

4. Why only LLaMA 3.1? adding more models could make the work more solid and convincing.

5. It might also help to check if a mix of RL and RFT training gives different outcomes — that could add an extra angle of analysis.

### Soundness
2

### Presentation
2

### Contribution
2
