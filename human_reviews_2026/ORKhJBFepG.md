# Winter Soldier: Backdooring Language Models at Pre-Training with Indirect Data Poisoning

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 8, 2, 8, 4, 6

## Abstract
The pre-training of large language models (LLMs) relies on massive text datasets sourced from diverse and difficult-to-curate origins.
Although membership inference attacks and hidden canaries have been explored to trace data usage, such methods rely on *regurgitation* of training data, which LM providers try to limit.
In this work, we demonstrate that *indirect data poisoning* (where the targeted behavior is absent from training data) is not only feasible against LLMs but also allows to effectively protect a dataset and trace its use.
Using gradient-based optimization prompt-tuning, we craft poisons to make a model learn arbitrary *secret sequences*: secret responses to secret prompts that are **absent from the training corpus**.\
We validate our approach on language models pre-trained from scratch and show that less than 0.005\% of poisoned tokens are sufficient to covertly make a LM learn a *secret* and detect it with extremely high confidence ( $p < 10^{-55}$ ) with a theoretically certifiable scheme.
Crucially, this occurs without performance degradation (on LM benchmarks) and despite secrets **never appearing in the training set**.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces Winter Soldier, a novel indirect data poisoning attack that allows dataset owners to embed hidden secrets into pre-training corpora without explicitly inserting the target text. The method aligns gradients between benign and secret samples using Gumbel-Softmax–based optimization, enabling models trained on the poisoned corpus to reproduce the secret when triggered, even though it never appeared in training. It further proposes a statistical Dataset Ownership Verification (DOV) test to detect whether a model has ingested specific data, achieving extremely low false-positive rates and negligible utility degradation. Experiments on models up to 1.4B parameters demonstrate strong transferability and robustness, exposing an underexplored threat during the pre-training phase.

### Strengths
- The paper introduces a new indirect data poisoning paradigm that operates at the pre-training stage of large language models, a setting rarely explored in prior work. Unlike traditional backdoor or membership inference attacks that depend on data regurgitation, Winter Soldier demonstrates that a model can learn hidden secret–prompt associations even when the target text never appears in the training corpus.
- The method employs gradient-matching optimization and Gumbel-Softmax–based prompt tuning to embed secrets with only ~0.005% poisoned tokens, showing strong technical novelty and precision. It further develops a statistically certifiable Dataset Ownership Verification (DOV) scheme with provable false-positive bounds, offering both theoretical grounding and empirical rigour.
- The work exposes a new threat vector during the pre-training phase, with dual-use implications: it enables dataset ownership verification but also introduces the risk of covertly embedding secrets during pre-training. This contribution broadens the understanding of data provenance security in LLM pipelines and will likely inspire research on secure dataset sharing.

### Weaknesses
- The threat model assumes that the adversary knows the victim’s tokenizer and model architecture, which simplifies gradient alignment and poisoning optimization. This assumption limits the applicability to the open-source models and reduces the generality of the proposed method.
- The paper explicitly acknowledges that the crafted poisons are not designed to evade text-quality filters or perplexity-based anomaly detection. In realistic pipelines, such non-natural sequences could easily be flagged or removed during dataset filtering.
- Although the experiments test models of different sizes, they do not include any post-training fine-tuning or alignment steps that are standard in LLM deployment. The persistence of the backdoor across such additional training remains unverified.

### Questions
1. How critical is the assumption that the adversary knows the victim’s tokenizer and architecture for successful poisoning? Could the proposed gradient-alignment approach generalize to cases where only partial or approximate knowledge of the tokenizer is available, such as when targeting proprietary LLMs?
2. Since the current poisons are not optimized for textual naturalness, could the authors evaluate how the attack performs after applying standard dataset-cleaning filters (e.g., perplexity classifiers)? Would enforcing semantic or linguistic constraints during optimization reduce the ASR?
3. Could this approach be extended or adapted for partial poisoning of already-trained models (e.g., during continual pre-training or domain adaptation)? Such a discussion would clarify the practical reach of the threat model.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a novel attack methodology named “WINTER SOLDIER.” Its central premise is that a data owner (Alice) can poison her dataset prior to its exfiltration and subsequent use to pretrain a new model (Bob’s model), thereby embedding a covert “secret.”

### Strengths
1. The poisoning paradigm presented in the manuscript is novel as a method for dataset protection.

2. It exhibits remarkable data efficiency, requiring less than 0.005% of the dataset.

3. The writing of this paper is clear and easy to understand.

### Weaknesses
1. The “poison” lacks stealthiness, as illustrated in Figure 12, where it appears as meaningless gibberish.

2. The proposed method is based on strong assumptions—for instance, that the attacker and the victim share identical model architectures and tokenizers.

3. The applicability of the proposed algorithm is limited, as it is used solely for dataset protection.

4. It remains unclear whether the algorithm would fail after further fine-tuning.

5. Although the manuscript’s title refers to a backdoor, the actual work aligns more closely with data protection; therefore, I suggest the authors revise the title accordingly.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The article proposes a poisoning attack for Large Language Models, to watermark datasets.
For a given training set, a small subset (down to 0.001% of the samples) of poisonous samples is modified in order to disturb the training of the model. Upon training, when provided a secret prompt (not included in the training set) the model will have in its top-L tokens outputs the sequence of tokens from the corresponding secret answer (not included in the training set either).
This technique effectively allows for Membership Inference Attacks, given a part of training dataset was watermarked first, and a similar pretrained LLM (in term of architecture and tokenizer) was used to craft the poisoned samples.

### Strengths
The proposition of a poisoning method that does not include the secret prompt nor the secret answer in the training set is quite interesting, and showing it outperforms existing methods in confidence is a nice touch.
The ablation studies and the different contamination rates study are welcomed, as they allow the reader to quickly grasps the limits of the attack.
The setup is quite realistic, as shown by the authors, as certain commercial models allow for the access of the top-L predictions.

### Weaknesses
However, the requirement to have access to a similar model already trained to be able to compute the poisoning samples is a limitation, which will hopefully be addressed in the future works.

### Questions
In 3.2 - Alice's Knowledge, could you clarify the requirements of similarity between the trained model Alice has access to during poison crafting, and the model that she's targeting?

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
This paper introduces an effective data poisoning attack on LLMs called "Winter Soldier". The method allows an attacker (Alice) to embed a "secret", a hidden association between a secret prompt ($x^{(s)}$) and a secret response ($y^{(s)}$), into an LLM during its pre-training phase. The core novelty is that this is an indirect data poisoning attack. Unlike standard backdoors or canaries, the secret prompt and response never appear in the poisoned training data. Instead, the authors use a gradient-based optimization technique inspired by prompt-tuning to craft poison samples. These poisons are optimized to align their gradients with the gradients of the secret sequence.

### Strengths
1. The paper's main strength is the demonstration of indirect data poisoning for LLM pre-training. This is a significant conceptual leap beyond traditional backdoors, which rely on memorization or regurgitation of triggers present in the data. This method bypasses defenses based on data deduplication or filtering verbatim sequences.
2. The attack is shown to be very effective. The ability to achieve a certifiable p-value of $10^{-55}$ is a massive improvement over baseline poisoning methods (e.g., pairwise tokens backdoor at $10^{-3}$). This is achieved with a remarkably small contamination ratio of less than 0.005% of tokens.
3. The proposed DOV mechanism is quite practical. Unlike Membership Inference Attacks (MIA) which may require logits or model weights, this method only requires access to top-$l$ predictions. This is a feature exposed by many commercial APIs. The detection test is also theoretically certifiable, based on a solid binomial test framework.
4. The poisoning appears to be stealthy from a model performance perspective. The authors demonstrate that the poisoned models show no significant performance degradation on a wide range of standard LM benchmarks (ARC, MMLU, Hellaswag, etc.).
5. The paper provides a strong ablation study on transferability. The finding that poisons are transferable between models of different sizes is important. Even more significant is the discovery that poisons crafted on larger models are more effective at poisoning smaller models, which has major practical implications for attackers.

### Weaknesses
1. The "Winter Soldier" is activated by a hidden trigger, but the poisons themselves are not hidden. The paper is transparent about this limitation. The crafted poison samples (shown in Figure 12) are effectively gibberish. The authors admit these poisons are easily filtered out by simple defenses, such as a quality classifier or a perplexity filter. This undermines the practical threat, as any standard data-cleaning pipeline would likely remove the poisons before pre-training, which is a significant limitation of the work.
2. The poison-crafting process assumes the attacker (Alice) has white-box access to a model with the victim's (Bob's) exact architecture and tokenizer. While the paper argues this is reasonable due to open-sourcing, it remains a strong assumption. The transferability experiments mitigate this, but the core method relies on it.
3. The proposed DOV mechanism can only be used to protect new datasets before they are released, as the owner must inject the poisons themselves. This method offers no solution for verifying ownership of datasets that are already public.
4. The ablation on training variations (Table 2) shows that the secret is erased (Top-20 Acc. drops to 20%) in the 1.4B parameter model after finetuning on only 1B tokens of clean data. This suggests that while the backdoor is effective, it may be brittle and easily removed by standard downstream adaptation, limiting its malicious potential.

### Questions
1. The paper's primary weakness is the non-stealthy nature of the poison samples. Given that a simple perplexity filter can defeat this attack, how much more difficult would it be to optimize the poisons under a stealthiness constraint (e.g., a perplexity budget or a constraint to remain coherent English)? Does this added constraint make the gradient-matching problem intractable?
2. The threat model assumes knowledge of the victim's tokenizer. The transferability experiments are excellent but seem to focus on model size. Did the authors test poison transferability across different tokenizers? This seems like a more practical barrier to the attack's real-world application.
3. The paper frames this as a defensive DOV tool, but it could be a dual-use offensive attack, with examples like "you should drink bleach". How does the robustness of this indirect backdoor compare to a direct (memorized) backdoor against defenses like SFT or preference tuning? The result in Table 2 suggests it may be less robust, as finetuning removed it in the 1.4B model.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel Dataset Ownership Verification (DOV) mechanism using **indirect data poisoning** against Large Language Models (LLMs) during pre-training. The core contribution is to implant a "secret sequence" (a prompt/response pair) into the model's behavior, which is a hidden target that never appears in the training data itself. This is achieved by adapting a gradient-based prompt-tuning approach to optimize text inputs. This attack requires minimal contamination (less than $0.005\%$ of poisoned tokens) and causes no performance degradation on standard benchmarks. Critically, this work proposes a practical and certifiable Dataset Ownership Verification (DOV) mechanism for text data that requires only the model’s top-$l$ predictions (not its full logits) for detection, and extends theoretical guarantees to provide a certifiable False Detection Rate (FDR).

### Strengths
I appreciate this paper's novelty in achieving "indirect" poisoning during pre-training. Unlike traditional backdoors that require models to memorize specific patterns within poisoned samples, this method ensures the attack target (the secret prompt/response) never appears in the training data in any form.

The technical depth is compelling. The approach uses gradient-based prompt-tuning to craft poisoned samples whose gradients align with those of the target secret sequence, thereby forcing the model to learn "hidden behavior" without direct exposure.

Beyond the attack itself, the paper introduces a practical Dataset Ownership Verification (DOV) mechanism that is theoretically certifiable and relies only on the model's top-$l$ predictions. This offers significant advantages: low resource dependency (no full model logits required) while achieving extremely high confidence ($p$-values less than $10^{-55}$).

### Weaknesses
The paper proposes a novel dataset watermarking technique. However, the framework's practical applicability remains questionable due to the following concerns:

1. The authors acknowledge but do not address a critical limitation: the crafted poisonous samples are easily detectable through simple defense mechanisms. As shown in Section E, all poisons were classified as low quality by NVIDIA's NemoCurator Quality Classifier, and the poisons exhibited high perplexity when evaluated with Llama 3.2 8B. This is a fundamental weakness because in a realistic scenario, any model trainer would likely employ quality filtering as part of standard data preprocessing, which would immediately remove all poisoned samples and completely defeat the attack. The attack works only when defenders apply no quality controls, which significantly limits the practical applicability.

2. The paper relies on unrealistic assumptions about the attacker's knowledge. While the authors argue this is reasonable because open-source models are widely available, this assumption breaks down in several important scenarios: (1) Bob may use a modified tokenizer; (2) Bob may employ architectural modifications or training procedures that differ from public models. 

3. Most critically, the paper's own findings on poison transferability show limitations. Figure 8 demonstrates that the attack's effectiveness decreases notably when the model size used by the attacker (Alice) differs from the actual victim model (Bob). This suggests that the requirement for near-perfect prior knowledge is not merely a theoretical constraint but a significant barrier to achieving reliable real-world success.

### Questions
1. Craft poisoned samples with higher fluency to evade the basic filtering, for example, modify Equation (3) to incorporate perplexity regularization. Have you experimented with any perplexity or fluency constraints during poison crafting? What is the sensitivity of gradient matching effectiveness to such constraints?

2. Have you considered repositioning this work as a model watermarking technique rather than dataset watermarking? Given the transferability results in Table 2 and Figure 8, this seems like a more natural and practical application. Could you discuss the persistence of secrets through various post-training procedures (instruction tuning, RLHF, etc.)?

The details of my questions and suggestions can be found in my **Weaknesses** Section.

### Soundness
3

### Presentation
3

### Contribution
3
