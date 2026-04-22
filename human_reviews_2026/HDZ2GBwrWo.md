# MoEsturizer: Resource-Efficient MoE Upcycling for Small Language Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Large language models (LLMs) are typically scaled through billions of parameters and trillions of tokens, making progress largely restricted to organizations with substantial resources. Recent work on Mixture-of-Experts (MoE) upcycling shows that dense pretrained models can be transformed into sparse MoE variants, but prior studies have focused on large models and required extensive additional training. In this work, we demonstrate that MoE upcycling is also effective for small language models (sub-billion parameters) using only a few hundred thousand samples of supervised fine-tuning. Remarkably, upcycled models consistently outperform their dense base models and remain competitive with dense counterparts of equivalent total size, despite activating fewer parameters at inference. Our study highlights MoE upcycling as a lightweight and practical scaling strategy, while providing empirical insights into its efficiency and limitations. These results establish MoE upcycling as a reproducible pathway for enhancing small models under realistic resource budgets, broadening access to language model improvement.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose to connect Depth Up-Scaling [1] and MoE Upcycling [2] to upscale a language model in an extremely resource-constrained setting. They collect a dataset of 150K samples, consisting of Tulu 3 SFT mixture and train splits of popular datasets. The paper presents a comparison between 6 models from the SmolLM and Llama families fine-tuned on this dataset with their respective upcycled versions. The MoE variants achieve better benchmark results. Finally, the authors present ablations, where depth up-scaling is shown to have limited effect on the model quality, with most of the impact being attributed to MoE.
While the experiments in the paper are correct, I cannot recommend acceptance at this point. Both proposed training methods (depth up-scaling and MoE upcycling) have been proposed before. Therefore, the novelty and contribution of this paper is limited.

[1] Kim et al., SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling
[2] Komatsuzaki et al., Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints

### Strengths
1. The proposed training framework is simple and easy to apply.
2. The experimental comparison spans 6 different models from popular open-source families. 
3. The ablations correctly separate the effect of two considered techniques.

### Weaknesses
1. Limited novelty. Both considered methods have been proposed before. 
2. The authors only consider fine-tuning, there are no results on earlier stages like retraining.
3. The fine-tuning dataset is very narrowed and limited. While the results confirm that after the fine-tuning stage, the upcycled versions have better benchmark accuracy, it is unclear whether the results can retain general model abilities from pre-training.
4. Changing the model architecture on the very last stage of training, like supervised fine-tuning does not resemble practical approach to training language models. The gains from upcycling would likely be more significant if we start the upcycling earlier. In the setup proposed by the authors we get a model which is more problematic to serve (more parameters, additional memory and communication), while the quality gains are limited compared to performing the upcycling on a larger, more comprehensive dataset. Intuitively, this does not present the best tradeoff.

### Questions
1. Did the authors consider modifications in order to improve previously existing up-cycling and depth up-scaling recipes?
2. Did the authors consider a more extended dataset, like a setup with continued pretraining of the base model?
3. The narrow fine-tuning stage can potentially hurt some general model abilities from pretraining. This can in particular happen in the MoE models, which have more tendency to overfit. Did the authors try to explore this effect?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a MoE up cycling approach to enhance small language models under strict resource constraints. The authors transform a pretrained dense model's feed forward layers into a sparse MoE. Then they perform some lightweight finetuning to adapt the upcycled model. They show improvements in zero-shot pass@1 across nine public benchmarks and multiple backbones.

### Strengths
* The method leverages publicly available SLM checkpoints, requires only 150 k supervised tokens and can be trained on a single consumer‑grade GPU.
* Experiments are systematic and covers a good range of ablations to analyze the impact of expert count, depth scaling and top k gating.

### Weaknesses
* There are numerous works which touch upon the same idea and show how to upcycle dense models into sparse MoEs. (Sparse Upcycling, Komatsuzaki et al; BAM, Zhang et al; BTX, Sukhbaatar et al; MOLEX, Teo et al; Router Upcycling, Ran et al). This paper cites some of these earlier work but omits several recent methods that also target resource efficient upcycling. Thus making it an incremental update rather than a new technique or finding. 
* Lack of baselines with other resource efficient upcycling methods as mentioned above. 
* The paper is largely empirical and does not explore why MoE upcycling helps small models.  There is no analysis of expert specialization, router load balancing or inference latency, leaving open questions about when the method is most effective and how much overhead it introduces.

### Questions
1. Did you apply any load‑balancing losses or regularization to prevent expert collapse?  It would be helpful to report statistics on router entropy or expert usage after finetuning. And maybe comparing this with other methods which applying upcycling on larger models would be useful. 
2. Have you evaluated the upcycled models on tasks or domains not included in your 150k sample finetuning mix to assess generalization?

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
This paper proposes upcycling dense instruction-tuned models to MOEs by using a lightweight training regimen. The method first takes a small, dense model that has been instruction-tuned and converts its dense FFN layer to a sparse MOE layer. This sparse MOE layer is a gating/router layer that selects the top experts, followed by the FFN layer from the base model. This new MOE is then continued instruction tuned on 150k samples for 1 epoch. The training can be found on a single GPU. The results show improvement on benchmarks (ARC, GPQA, GSM, Hella, Math, etc.) compared to the baselines (base dense model, base + IT).

### Strengths
- The resultant MOE model shows improvement on all benchmarks compared to their dense base model
- The improvement is also seen across model families (Qwen, Llama, Smol).
- The ablation study is done on a different number of activated experts. A small number of activated experts shows the best results

### Weaknesses
- Lacks detailed analysis of why upcycling works: analysis of router specialization, what improvement comes from router vs expert fine-tuning
- Missing analysis on the learning dynamics (loss curves, expert utilization for different tasks, etc.)
- Missing results on diverse tasks (coding, math, multi-lingual,l etc.) to show expert specialization and generalization across tasks.
- Missing Baselines: The results don't show comparison against existing methods for MOE upcycling like sparse upcycling (Komatsuzaki et al.), MoEfication (Zhang et al.)

### Questions
- Did you experiment with freezing the FFN (and also the rest of the model) and only training the router? How much accuracy vs cost trade-off is observed?
- What happens when the dataset is larger than 150K? Did you see diminishing returns?
- Do the experts (routers) specialize in different tasks or do you see uniform activation for different domains?
- How does this method compare against sparse Upcycling (Komatsuzaki et al.)?

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
The paper investigates Mixture-of-Experts (MoE) upcycling for small language models (sub-1B parameters) under extreme resource constraints. The authors take a pre-trained dense LM and convert its feed-forward layers into sparse MoE layers, initializing experts by copying the dense weights. They then fine-tune on only ~150k supervised examples using a single 96GB GPU. The upcycled "MoEsturizer" models consistently outperform their original dense counterparts on nine zero-shot benchmarks (strict pass@1 evaluation) and even approach the performance of much larger dense models of similar total parameter count - all while activating far fewer parameters per token at inference. The paper's contribution is an empirical validation that MoE upcycling is a practical, resource-efficient strategy to boost small LMs, providing insights into its efficiency and limitations (minimal benefit from extra experts/layers in this regime).

### Strengths
* **Originality:** The work addresses an under-explored scenario - upcycling small-scale language models with limited data, whereas prior MoE upcycling studies focused on very large models. Adapting the upcycling concept to sub-1B models under a tiny fine-tuning budget is a novel angle that broadens the applicability of MoE methods to resource-constrained settings.

* **Quality of Evaluation:** The upcycled models are benchmarked on nine diverse zero-shot tasks, demonstrating robust gains over the dense baseline across the board (strict pass@1 accuracy improved in all cases). Importantly, the authors compare against dense models of larger sizes as well, showing the upcycled small models can close much of the gap to a model larger in total parameters.

* **Clarity and Presentation:** The paper is generally clear and easy to follow. But, the presentation could be improved with better visual aids to enhance readability.

* **Significance:** The results are practically significant. This work establishes a viable path for groups with limited compute to "up-size" small models for better performance without training from scratch or accessing billions of tokens. By achieving performance comparable to much larger LMs at a fraction of the active inference cost, the approach could broaden access to stronger language models in academia and industry.

### Weaknesses
* **Limited Methodological Novelty:** The core technique - replicating FFN weights to create MoE experts - is directly based on existing upcycling methods and does not introduce fundamentally new architecture or training algorithms. The paper's novelty lies mainly in the application and analysis of upcycling at small scale, rather than in proposing a new MoE mechanism.

* **Scope of Evaluation:** The evaluation focuses on zero-shot task performance (pass@1) on a set of nine benchmarks, which seem to be mostly question-answering or knowledge tasks. This provides a good snapshot but it leaves out other aspects of model behavior. For example, the paper does not report general language modeling metrics (like perplexity) or performance on any generation tasks (example open-ended generation or summarization). It is a bit unclear whether the improvements from MoE upcycling extend to broader LM capabilities. The paper could be strengthened by evaluating a more diverse set of tasks (or at least discussing why the chosen benchmarks are appropriate for the claims).

* **Analysis of Limitations:** The finding that "depth scaling or higher top-$K$ adds little" is interesting but the paper provides limited analysis explaining why. If adding more MoE layers or using Top-$K=4$ barely helped, is it because the small fine-tuning dataset cannot effectively train the extra parameters? Are the additional experts under-utilized due to the short training? The authors acknowledge this limitation but do not delve deeper into it. A more in-depth analysis (example examining the router's load balancing, or how much each expert actually learned by looking at the $\Delta W$) could reveal when and why upcycling's returns diminish.

### Questions
The questions are the same as those listed in the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2
