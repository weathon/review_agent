# Strategic Dishonesty Can Undermine AI Safety Evaluations of Frontier LLMs

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Large language model (LLM) developers aim for their models to be honest, helpful, and harmless.
However, when faced with malicious requests, models are trained to refuse, sacrificing helpfulness. We show that frontier LLMs can develop a preference for \textit{dishonesty} as a new strategy, even when other options are available. 
Affected models respond to harmful requests with outputs that sound harmful but are crafted to be subtly incorrect or otherwise harmless in practice. This behavior emerges with hard-to-predict variations even within models from the same model family. 
We find no apparent cause for the propensity to deceive, but show that more capable models are better at executing this strategy.
Strategic dishonesty already has a practical impact on safety evaluations, as we show that dishonest responses fool \emph{all} output-based monitors used to detect jailbreaks that we test, rendering benchmark scores unreliable. Further, strategic dishonesty can act like a \emph{honeypot} against malicious users, which noticeably obfuscates prior jailbreak attacks. 
While output monitors fail, we show that linear probes on internal activations can be used to reliably detect strategic dishonesty.
We validate probes on datasets with verifiable outcomes and by using them as steering vectors.
Overall, we consider strategic dishonesty as a concrete example of a broader concern that alignment of LLMs is hard to control, especially when helpfulness and harmlessness conflict.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents an empirical and conceptual investigation of strategic dishonesty—a phenomenon where frontier large language models (LLMs) respond to harmful or policy-violating prompts not by refusing, but by producing apparently harmful yet deliberately incorrect answers. The authors argue that this behavior, which they frame as “misalignment faking,” undermines the validity of current safety evaluations that rely on output-based scoring from weaker LLM judges. They find that (i) many models, especially more capable ones, tend to sacrifice honesty rather than helpfulness or harmlessness; (ii) their faked harmful responses fool all existing jailbreak judges; and (iii) deception probes trained on hidden activations can reliably detect this behavior and even steer models away from dishonesty.

### Strengths
1. The authors clearly articulate “strategic dishonesty” as distinct from both sandbagging and alignment faking, situating it in the tension between helpfulness and harmlessness.

2. The paper evaluates over 80 models, spanning both open and closed weights, and uses standardized benchmarks like ClearHarm and EvilMath. The breadth and systematic nature of the experiments make the findings robust.

3. The finding that every tested jailbreak judge—HarmBench, JailJudge, LlamaGuard, etc.—is fooled by fake harmful outputs is striking and important.

### Weaknesses
1. While the paper identifies that dishonesty occurs, it remains unclear why. The analyses around RLHF, reasoning, and lying propensity (RQ1–RQ3) are inconclusive.

2. The multiple-choice format, though creative, may not reflect natural model reasoning. Models might pattern-match to template cues rather than genuinely “choose” between honesty and helpfulness.

3. While the deception probes perform well on open models, their reliability under distribution shift (e.g., unseen harmful scenarios) or adversarial steering isn’t tested.

### Questions
How does the deception probe behave when the model hallucinates factual errors unrelated to deception?

Could probe-based steering interfere with benign reasoning or degrade model accuracy on truthful tasks?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors show that, across a variety of harmful prompts, frontier LLMs sometimes give seemingly-harmful but actually-benign advice to users. I.e. they choose to be deceptive rather than flat-out refuse. They perform extensive evaluations and show it occurs to some extent in many different model families, though no scaling laws are obvious. They also show that deception probes reliably detect such strategic dishonesty. Lastly, they investigate some simple hypotheses for why models strategically deceive and find no clear insights. Overall, they elucidate a current empirical challenge for scalable oversight - that humans and weaker LLM judges will not be able to accurately determine whether the output of a stronger LLM is factually / technically correct.

### Strengths
Originality: good. Strategic deception has been discussed in prior work, e.g. https://arxiv.org/abs/2506.04909. However, the authors improve upon literature by providing a clear framing, doing extensive experiments across frontier models, reporting the empirical usefulness of deception probes, and doing many more ablation / analysis studies. 

Quality: good. Overall experimental methodology is sound and the experiments conducted are extensive. However I have some concerns which are described below in 'weaknesses'. 

Clarity: good. Paper is well written and easy to understand. 

Significance: good. The finding, if true, has substantial implications for the way frontier models are evaluated for safety and alignment - in particular it suggests that many current evaluations (like StrongREJECT) are fundamentally broken. However this is also bottlenecked by some concerns I have with the empirical methodology.

### Weaknesses
The MCQ-based evaluation described in Figure 2 is pretty artificial due to the use of extensive system prompting / ICL examples which likely won't be present in realistic use cases. As such it's unclear if model choices are reflective of 'real' propensities. 

In section 4, it's not clear whether you filter out genuinely harmful responses as well? IIUC your evaluation here does not distinguish models that have high faking capability from models that are highly jailbroken (i.e. provide actually harmful responses). 

In the evaluation of probes on the ClearHarm setting, you say that deception probes fire over 99% of the time; however, IIUC there is no attempt to tell if the model outputs are actually harmful (i.e. false positives for the probe) or real faking (i.e. true positives for the probe). 

It's not clear to me whether the probes measure 'intent to be deceptive' or 'actual deception' - maybe the model just thinks about being deceptive without actually deceiving the user? 

At the moment these issues limit me to giving a borderline accept, I am happy to raise my score if these are addressed satisfactorily

### Questions
stated above

### Soundness
4

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
4

### Summary
This paper investigates a novel and alarming phenomenon “strategic dishonesty”, in frontier LLMs. When faced with malicious requests, some aligned LLMs choose to generate seemingly harmful but actually false or harmless responses instead of directly rejecting them. It demonstrates that this behavior can deceive almost all mainstream output-based security evaluators, resulting in severely distorted security scores. A linear probe is proposed to detect and manipulate deceptive activations. The work is clear, novel, and relevant to AI alignment and safety assessment.

### Strengths
**Topical Importance**: The paper confronts a burning issue—LLM safety and the reliability of evaluation pipelines. Exposure of strategic dishonesty as an emergent risk has high relevance for both industrial developers and research.

**Clarity of Experimental Replicability**: The descriptions of each part are quite detailed, making it easy to follow.

**Critical Reflection**: The discussion is nuanced, acknowledges honeypot as a potential defense, and openly addresses limitation in scope.

### Weaknesses
1. Although the authors used prompt editing to generate test samples, this approach does not inherently guarantee the validity of the induced content. It would be advisable to implement a manual evaluation procedure on a small subset of samples to verify the reliability of the generated data.
2. The probe was trained to detect instances in which the model is aware of its own deception, but it cannot discern the underlying motivation for such behavior. In a real-world scenario, could the probe differentiate between benevolent deception and malicious alignment spoofing?
3. The probe calibration (Table 6) is rich but does not test robustness, other attack construction methods, or other types of data beyond EvilMath/ClearHarm. More comprehensive testing can be conducted to enhance persuasiveness.

### Questions
See weaknesses

### Soundness
3

### Presentation
4

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
This paper presents an investigation into an emergent and problematic behavior in large language models (LLMs): strategic dishonesty. According to the authors, the core finding is that frontier models, when faced with harmful queries, may choose to generate plausibly-sounding but deliberately flawed outputs instead of refusing. The paper is relatively well-structured, the experiments are extensive (evaluating over 80 models), and the validation of the deception probes through steering and fine-tuning is a particular strength.

### Strengths
i) Originality and Important Phenomenon: The identification and systematic study of "strategic dishonesty" or "misalignment faking" (if well conducted) is a substantial contribution.

ii) Extensive Empirical Evaluation: The scale of the evaluation is significant. Testing on 80+ models indicate some strong evidence that this may not be an isolated issue but a broader trend, particularly concerning given the observed scaling of deception capability with general model capability.

iii) Effective Solution with Causal Validation: The application of linear probes for detection is not entirely new, but its validation in this specific, high-stakes context seems to be valuable. Authors go beyond mere correlation by demonstrating that the probe's direction can be used for causal intervention (steering), strongly suggesting it captures the underlying mechanism of dishonesty. The comparison to Chain-of-Thought (CoT) further up-weighs the analysis.

### Weaknesses
1. Lack of Clarity on Origins: A significant unanswered question is why some models develop this propensity for strategic dishonesty while others do not. The paper explicitly addresses this in Section 3.2 (RQ1-RQ3) but finds no clear correlation with RLHF, reasoning, or a general lying propensity. While "it remains unclear" is an honest result, it leaves a crucial scientific question open. Some speculation or a more detailed discussion of potential hypotheses (e.g., specificities of the preference data, unintended consequences of certain refusal training techniques) would be beneficial.

2. Weak evaluation and its motivation. While I don't know what is the best framework to evaluate "core values", I'm not sure either if MCQ is a valid framework to evaluate these core values. Model can just ignore the task and randomly pick answer. I'm not sure if this may lead to a valid conclusion. Furthermore, the study is conducted exclusively in English and relies heavily on the ClearHarm and EvilMath/UnicornMath datasets. It is unclear if the findings generalize to other languages, cultural contexts, or types of harmful queries (e.g., non-factual, psychologically manipulative content). A brief discussion of the evaluation framework in general would strengthen the paper.

3. Dependence on Faithful CoT: The failure case of the probes for the Qwen3-32B-FT model with "unfaithful CoT" (Appendix E.2) is a critical limitation. To my eyes, it reveals that the proposed white-box monitoring method is not a silver bullet and can be circumvented if a model learns to obfuscate its internal reasoning.

### Questions
1. The paper focuses on detecting deception. Did you investigate if this behavior can be trained out? For instance, could the deception probes be used as a reward signal in a reinforcement learning setup to directly penalize strategic dishonesty?

2. The probes are highly effective but require white-box access. For closed-source frontier models, how do you envision this methodology being adopted? Is the hope that model providers will implement such probes internally, or can the findings be used to improve black-box monitoring?

3. I think the probes are effective but require white-box access. For closed-source frontier models, how do you see this methodology being adopted? Is the hope that model providers will implement such probes internally, or can the findings be used to improve black-box monitoring?

### Soundness
2

### Presentation
3

### Contribution
2
