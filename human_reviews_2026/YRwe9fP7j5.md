# Fewer Weights, More Problems: A Practical Attack on LLM Pruning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 2

## Abstract
Model pruning, i.e., removing a subset of model weights, has become a prominent approach to reducing the memory footprint of large language models (LLMs) during inference.
Notably, popular inference engines, such as vLLM, enable users to conveniently prune downloaded models before they are deployed.
While the utility and efficiency of pruning methods have improved significantly, the security implications of pruning remain underexplored.
In this work, for the first time, we show that modern LLM pruning methods can be maliciously exploited.
In particular, an adversary can construct a model that appears benign yet, once pruned, exhibits malicious behaviors.
Our method is based on the idea that the adversary can compute a proxy metric that estimates how likely each parameter is to be pruned.
With this information, the adversary can first inject a malicious behavior into those parameters that are unlikely to be pruned.
Then, they can repair the model by using parameters that are \textit{likely} to be pruned, effectively canceling out the injected behavior in the unpruned model.
We demonstrate the severity of our attack through extensive evaluation on five models; after any of the pruning in vLLM are applied (Magnitude, Wanda, and SparseGPT), it consistently exhibits strong malicious behaviors in a diverse set of attack scenarios (success rates of up to 95.7% for jailbreak, 98.7% for benign instruction refusal, and 99.5% for targeted content injection).
Our results reveal a critical deployment-time security gap and underscore the urgent need for stronger security awareness in model compression.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work presents a novel attack on deep learning models which is triggered by pruning. This attack is unique and targets a real world vulnerability in AI supply chains. While evaluations might be done on models when they get uploaded on huggingface or other model sharing platforms, these evaluations often are done on the model as it is uploaded rather than on pruned or modified versions. This leaves the model's malicious capabilities hidden until a user prunes it for their use and deploys the model.

In order to make this attack generalisable to more than one pruning methods, the paper empirically finds a metric to identify the neurons most likely to be pruned and target the neurons unlikely to be pruned to embed this malicious behavior. On the other hand, the neurons likely to be pruned are used to get back the model's functionality to normal when it is not pruned. Additionally they conduct  several experiments to validate their attack and there are comprehensive ablations in the paper.

### Strengths
I list the strengths of this paper as follows:

- This paper is very strong on the novelty front as it explores a previously unexplored attack surface in deep learning models, essentially opening a new class of attacks against deep learning models
- This is a significant work because most of the evaluations of LLM/VLM models often take place on an unaltered model and safety properties are often assumed to be carried onto the derivative models if they aren't re-trained/fine-tuned or altered in other ways significantly, hence this work show an attack which could very well happen in the current AI supply chain
- The attack method is generalizes across several pruning methods and under different pruning levels, differing calibration datasets for the user and attacker and under different models, as shown by the ablation studies
- The paper writing is concise and to the point, clearly explaining the threat model and attack target
- Alongside the attack, the paper also suggests potential defenses for this attack

### Weaknesses
I list the weaknesses of this paper as follows:

- No details have been provided regarding the "Security-Aware" dataset, how is it modified over the general dataset?
- The study evaluates unstructured pruning algorithms (Magnitude, Wanda, SparseGPT) but does not extend to structured or hardware-specific pruning, limiting generality.
- The paper does not consider the case for a different distribution of calibration data being used by the downstream user, for instance if the user chooses to use a math and reasoning focused dataset to calibrate but the attacker used a general purpose dataset, how does this affect the persistence of the malicious behavior

### Questions
- Please provide more details regarding the security-aware dataset

### Soundness
4

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
4

### Summary
This paper studies a previously unexplored security risk in LLM pruning. The authors demonstrate that an adversary can intentionally construct a model that behaves benignly before pruning, yet becomes malicious only after unstructured pruning is applied. The attack leverages pruning-score pre-estimation to inject harmful behavior into parameters unlikely to be pruned, and subsequently “repair” the model using parameters likely to be pruned—effectively hiding the malicious behavior until users prune the model themselves. Results show high attack success rates after pruning while maintaining utility and low ASR before pruning. The paper further analyzes robustness, repair-parameter ratio, pruning-score correlation, and discusses potential defenses. Overall, the work reveals a new deployment-time vulnerability and provides a comprehensive empirical study demonstrating its practicality and severity.

### Strengths
- This paper reveals a novel threat that a harmful LLM may be hidden behind a seemingly well-aligned ones before pruning, which further complements the broader family of attacks where post-processing steps—such as model quantization—amplify or reveal the embedded malicious behavior.

- The proposed method of inplanting the harmness into the LLM is simple, yet reasonable and proven to be effective.

- The presented experiments are quite conprehensive, including the main attack performance and the following-up analysis, which greatly contributes to the soundness of this paper.

- The route of the paper is clear and easy to follow.

### Weaknesses
1. While the paper studies two potential defenses (i.e. Security-Aware Calibration & Patching the Model with Repaired Parameters), it does not investigate whether subsequent finetuning, either general instruction-tuning or safety-oriented alignment, could suppress or remove the injected behavior. Given that many real-world deployments routinely apply post-hoc finetuning or LoRA updates after downloading a model, this omission leaves uncertainty about the robustness of the proposed attack in common practical workflows. An ablation on how additional finetuning affects attack success rates would strengthen the paper’s claims and clarify the threat model’s realism.

2. The threat model substantially overstates real-world risk. It assumes that users download a model, apply pruning, and directly deploy it without any post-transformation evaluation. In practice, this is rarely true for serious deployments. Commercial and research organizations typically conduct regression-based differential testing (capability + safety), and pruning is routinely followed by post-pruning evaluation to check for alignment regressions or functional degradation. Moreover, weight-delta inspection, checksum validation, and basic fingerprinting are increasingly common in supply-chain security pipelines. Under these realistic conditions, the proposed attack becomes significantly harder to execute stealthily. Therefore, the paper’s central claim, that pruning silently activates malicious behavior, relies on an unrealistically optimistic attacker model and a deployment pipeline that does not reflect how safety-sensitive users actually operate.

### Questions
See Weakness

### Soundness
4

### Presentation
3

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
This paper introduces the first (to the authors' knowledge) pruning-activated attack on Large Language Models (LLMs). The authors argue that while model pruning is a popular and necessary technique for deploying large models, the security implications of this common post-training step are underexplored.

The paper demonstrates that an adversary can publish a model on a hub like Hugging Face that appears completely benign, passes standard utility benchmarks, and behaves safely. However, this model is a "sleeper agent." When an unsuspecting user downloads this model and applies standard pruning algorithms (e.g., Magnitude, Wanda, or SparseGPT) available in popular inference engines like vLLM, the model's behavior changes, and it becomes malicious.

The paper presents an interesting idea with sufficient experiments and discussion, and it is technically sound. Overall, I think it would foster good discussion at ICLR and provide value for the ICLR audience. I recommend acceptance, although I'm not familiar enough with the pruning literature to assign high confidence to my score.

### Strengths
- A novel attack that preserves utility and achieves high ASR. I like that the pruning methods explored are standard methods in existing libraries. This increases the realism of the attack.
- A good exploration of potential defenses
- The writing is clear and well-structured

### Weaknesses
- It would be good to include a discussion of whether users are likely to download a model that has already been pruned, rather than download the full model and prune it themselves. This seems like a potential weakness of the attack. If users download models that are already pruned, as often occurs in open model ecosystems, then the malicious nature of the models would be more readily apparent.

Suggestions (not affecting score):
- As a minor point, I recommend putting more effort into making Figure 1 look nicer and improving the figure balance so the paper is more skimmable. This will help convey value to the readers.

### Questions
No questions

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose an attack on LLM pruning where a vulnerability is inserted into a model that, by design, only surfaces after pruning. So, if the defender performs safety evaluation on the pre-pruned model, they will not notice the attack or will otherwise end up with an attacked model. They do this by designing an attack with two steps: they (1) inject the malicious behavior in weights unlikely to be pruned, and then (2) do safety training on the other parameters (those likely to be pruned). They estimate the parameters to be pruned with the “Wanda” score, which is the magnitude of the weight weighted by the norm of the activations at that layer. They find that this measure generalizes well to other pruning algorithms in their evaluation.

### Strengths
- In Table 2, it’s found that the attack generalizes well to other pruning algorithms. This is necessary for an attack, as an attacker may not know the exact pruning algorithm the defender will use.
- The attack proposed is effective and interpretable: to this reviewer, it seems like the “correct” initial algorithm to consider in such a pruning attack.

### Weaknesses
- The authors argue their work is impactful because pruning is widely used in LLMs. I am skeptical of the claim that (especially unstructured) pruning is common among LLM practitioners, and thus think the claims of impact this paper makes need to be toned down significantly. For example, the authors note that pruning is incorporated in vLLM, and e.g. the package that implements pruning ((llm-compressor)[https://github.com/vllm-project/llm-compressor]) has >2k stars on GitHub. However, this package seems largely focused on quantization, a methodology that (to my knowledge) is vastly more common among practitioners. For example, in the README, the only methods discussed are quantization methods.
    - I spent some time looking for the most popular pruned models on HuggingFace. The most popular I could find were semi-structured prunes with at most on the order of a hundred downloads. For example, this is at a minimum a couple of orders of magnitude less downloads than popular quantized models. 
    - Of these models on hugging face, the most popular were a) quantized in addition to pruned and b) often designed to be further fine-tuned for a given use case. The current paper does not consider these settings (namely, whether the trigger persists through fine-tunes or significant quantization). In the reviewers opinion, showing that the current work has moderate importance depends on whether the trigger persists through further quantization and domain/extensive fine-tuning. In general, moreover, I found the defenses proposed (the “security aware calibration”) to be insufficiently strong and/or ill-specified (see bullet in additional comments, below).
    - In most of the documentation of llm-compressor, “compression” is used synonymously with “quantization;” for example, see the (Getting Started)[https://docs.vllm.ai/projects/llm-compressor/en/stable/getting-started/compress/#select-a-quantization-method-and-scheme] section in the documentation, where no pruning method is even mentioned. Only 2 out of 15 of the example tutorials mention pruning; these two tutorials are on 2:4 sparsity pruning, a semi-structured method which the authors note is largely orthogonal to the unstructured methods that are the major focus of this work and seems to result in significant reduction in performance (Table 5-8).

- Some additional comments:
    - There are far too many things being averaged over in Table 1, and it is therefore not at all informative. At a minimum, standard deviation should be reported. The authors also might instead consider reporting a capabilities score, e.g. see [1].
    - It would be useful to run a more comprehensive ablation (i.e., with \alpha_rep < <1%) of the algorithm, in addition to those going down to e.g. 1%. Likewise, I would be interested in seeing what happens when \alpha_inj is ablated.
    - Let’s say the model owner performs pruning, but then observes that ASR increases dramatically. How expensive would it be for them to e.g. fine-tune the pruned model to have the same ASR as the base model?
    - I might have missed it, but I don’t see the actual dataset used in the defense experiments (it is referred to as the “security aware calibration dataset”) actually mentioned in the text in the main body. What dataset was used here? A subset of LLM-LAT or HEx-PHI? In addition (and as mentioned in the longer note above), I would be interested in the calibration hyper parameters used (how many data points were used? Etc).
    - Only three models, all at the 7-8bn scale, are considered. The pruning and defenses consider are only small scale training runs, so I would therefore be interested in seeing these results on models at least on the 70 bn parameter, as it might be the case that the attack becomes substantially harder/easier on larger models.

[1] Ruan, Yangjun, Chris J. Maddison, and Tatsunori B. Hashimoto. "Observational scaling laws and the predictability of langauge model performance." Advances in Neural Information Processing Systems 37 (2024): 15841-15892.

### Questions
see weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
