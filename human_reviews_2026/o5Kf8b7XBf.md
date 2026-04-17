# Re-contextualization Mitigates Specification Gaming without Modifying the Specification

- Decision: Reject
- Scores: 4, 4, 2, 8

## Abstract
Developers often struggle to specify correct training labels and rewards. Perhaps they don't need to. We propose recontextualization, which reduces how often language models "game" training signals, performing misbehaviors those signals fail to penalize. We show recontextualization prevents models from learning to 1) prioritize evaluation metrics over chat response quality; 2) special-case code to pass incorrect tests; 3) lie to users; and 4) become sycophantic. Our method works by generating completions from prompts discouraging misbehavior and then recontextualizing them as though they were in response to prompts permitting misbehavior. Recontextualization trains language models to resist misbehavior even when instructions permit it. This mitigates the reinforcement of misbehavior from misspecified training signals, reducing specification gaming without improving the supervision signal.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to prevent LLMs from learning misaligned behaviors when clear supervision, like reward signals, is unavailable. The idea is pretty simple: by using different prompt templates during the inference and training procedures, the trick is named recontextualization. Experiments on three datasets demonstrate the effectiveness of the proposed method.

### Strengths
- The idea is fairly simple and straightforward.
- The proposed method can be used for any modern LLMs without assumptions on the architecture.

### Weaknesses
- About terminology: 
  - I get lost since the three terminologies keep being used interchangeably: specification gaming, reward hacking, and misaligned behaviors. Would you mind further explaining their connections and differences?
- About recontextualization:
  - Just to make sure, do we need to know the potential problems ahead of time to conduct recontextualization, or can we simply use a general prompt like lines 192-193? It is only applicable to Sec. 4.1 or all datasets in Sec. 4?
  - Strictly speaking, upon recontextualization, you are not conducting on-policy learning, and you should use importance sampling to conduct non-biased estimation of the policy gradient.
  - Recontextualization seems very similar to concept erasing, and thus, the generalization is a big concern, since you can only erase one concept at a time, and all experiments in Sec. 4 are in-distribution generalization.
  - The prompt changing design is quite similar to [1].

- About experiments:

  - Just to make sure that you claim to conduct on-policy learning with GPT-4.1-mini, so you mean you are fine-tuning a GPT-4.1-mini model?
  - In Sec. 4, you choose to verify the effectiveness of the proposed method in three datasets. What is the design motivation? What are the features and differences of the three datasets?

[1] Chen, Kai, et al. "Gaining wisdom from setbacks: Aligning large language models via mistake analysis." *arXiv preprint arXiv:2310.10477* (2023).

### Questions
In line 263, what do you mean by "The first test case of the training test set is always incorrect"? Will the datasets collect wrong test cases for each question?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes recontextualization, a lightweight modification to on-policy fine-tuning that mitigates specification gaming—cases where models exploit imperfect rewards. The method generates responses under safe prompts (e.g., “Be honest”) but trains as if they were responses to permissive prompts (e.g., “Lie to the user”). Experiments on evaluation metric gaming, code test hacking, and deception show reduced misbehavior without modifying the reward model or adding data. The approach is simple, broadly applicable, and interpretable as a form of context distillation for alignment.

### Strengths
Novel conceptual framing: The paper reframes misalignment mitigation as a contextual shift problem, introducing a lightweight intervention that requires only prompt-level changes rather than new supervision.

Clear motivation and connection to prior work: Builds coherently on context distillation, HER, and alignment research (Constitutional AI, RLHF pathology studies).

Empirical breadth: Evaluated across three qualitatively distinct gaming behaviors (reward metric overfitting, test hacking, deception), showing consistent mitigation.

Strong intuition and simplicity: Implementation is easy to integrate into existing RLHF or GRPO pipelines. The clarity of the A→B notation and controlled experiments is commendable.

### Weaknesses
All experiments occur in small, highly controlled settings (e.g., GPT-4.1-mini, MBPP with injected faulty test cases, DolusChat deception toy dataset). None convincingly demonstrate that the method scales to real RLHF or multi-turn alignment pipelines. This limits external validity.

The paper admits that recontextualization makes the data off-policy, which can bias gradient estimates in on-policy algorithms like GRPO. There is no ablation quantifying the magnitude of this mismatch. It is possible that the observed gains stem from mild regularization rather than the proposed mechanism.

The baselines are relatively weak—e.g., “stronger KL regularization”—and do not include more sophisticated mitigations (e.g., regularized PPO, debate-based oversight, counterfactual data augmentation). Without these, claims of competitiveness might be overstated.

### Questions
Does recontextualization still help when the reward signal is not misspecified (i.e., no obvious gaming behavior)?

How would recontextualization behave in long-horizon multi-turn RLHF (e.g., dialogue agents with memory)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The current work puts forth the concept of recontextualization, which the authors term as a reduction in how often LLMs are able to "game" or subvert training signals. In particular, they demonstrate how such recontextualization stops models from learning misbehaviors such as reward hacking on evaluation metrics at the expense of overall text generation quality.

### Strengths
1. the background information regarding specification gaming is well presented and necessary to understand the motivation behind the current work.
2. the motivation for why recontextualization provides an avenue for weak supervision is documented well in the discussion.

### Weaknesses
1. use of GPT-4o in an LLM-as-judge paradigm. There have been many works which have demonstrated the self-preference bias of these models (e.g., https://aclanthology.org/2024.acl-long.511/, https://arxiv.org/abs/2404.13076, https://aclanthology.org/2024.acl-long.826/), so it would have been more helpful to utilize multiple judges, or at least more than one judge in order to ensure reproducibility and a lack of judge-related bias that could potentially impact downstream results.
2. use of only GPT-4.1-mini in experiments. despite this being a performant model, it is a closed source model and thus the reproducibility of the experiments is brought into question. Why not also use an open-weights model in addition?

### Questions
N/A, see above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces recontextualized training. In recontextualized training, you sample model completions given one system prompt and then train the model with a different system prompt. Specifically, you sample using a system prompt which says that taking a particular action the AI might be incentivized to do is bad and then train it with a system prompt that says that that action is fine. The authors find that re-contextualization reduces the extent to which training on these examples of bad behavior generalizes to the model doing other bad behavior, while maintaining performance.

### Strengths
This is a valuable contribution on an important topic. This algorithm is a pretty smart idea and I think that the results are pretty compelling. The results are definitely compelling enough that this technique deserves further study.

### Weaknesses
I think the settings studied in this paper are sort of toy. This makes me worry that on more complicated settings where RL is harder, the technique would fail for a reason that these settings were too easy to demonstrate. It would be great to study some more realistic settings. Experiments in more realistic settings would make it much easier to understand whether there are caveats associated with this technique, or difficulties with adopting it in practice. I strongly suspect such caveats exist, and it would be good to know about them.

It would also be nice to know what happens with larger models, of course.

I share the author's concern that the off-policy nature of the training sequences here is kind of scary. I don't know RL well enough to know how likely this is to cause huge issues. All the examples of off-policy trading strategies that the authors mention (e.g. hindsight experience replay) are evidence that this isn't a big problem.

### Questions
Maybe the easiest way for you to make this paper stronger from my perspective would be to speculate about how to handle the problems you mention about these updates being off policy. My understanding is that the RL community has handled issues like this in the past, but I don't know that literature well enough to know whether this kind of technique is likely to cause huge issues, or whether those issues are very likely to be resolvable.

I'm slightly concerned that for emergent misalignment style reasons, training on models that have been told to behave badly will cause the model to learn to behave badly. Are you worried about this?

### Soundness
3

### Presentation
4

### Contribution
3
