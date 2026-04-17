# RUB: Evaluating Residual Knowledge in Unlearned Models

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Machine Unlearning (MUL) has emerged as a key mechanism for privacy protection and content regulation, yet current techniques often fail to guarantee the complete removal of sensitive information. While most existing works focus on verifying the execution of unlearning, they overlook the critical question of whether models remain robust against adversarial attempts to recover forgotten knowledge. In this work, we advocate for the principle of Robust Unlearning, which requires models to be both indistinguishable from retrained counterparts and resilient against diverse adversarial threats. To instantiate this principle, we propose a unified benchmark, RUB (Robust Unlearning Benchmark), that systematically evaluates the robustness of unlearning algorithms across classification, image-to-image reconstruction, and text-to-image synthesis. Within this framework, we introduce the Unlearning Mapping Attack (UMA) as a generalizable method to detect residual information, and demonstrate how existing attack strategies can be adapted into this framework as long as they conform to the generic UMA framework. Our experiments across discriminative and generative tasks reveal that state-of-the-art unlearning methods remain vulnerable under these evaluations, even when passing standard verification metrics. By positioning robustness as the central criterion and providing a benchmark for adversarial evaluation, we hope RUB paves the way toward more reliable and secure unlearning practices. The codebase and model checkpoints in RUB will be published.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on evaluating the robustness of machine unlearning (MUL). The authors argue that existing MUL evaluation methods primarily focus on whether unlearning has been performed or whether the model is indistinguishable from a retrained model, while overlooking a key issue: whether the model after unlearning can resist adversarial attacks to recover the forgotten knowledge.

### Strengths
1.	This paper addresses a key point. With the implementation of laws and regulations such as the "right to be forgotten," simply claiming forgetting is insufficient; proving the robustness is a more important and challenging problem.
2.	UMA is a clear and general attack framework. The authors successfully instantiated it on classification, I2I, and T2I tasks, providing a unified "probe" for subsequent researchers.

### Weaknesses
1.	While the benchmark covers three CV tasks, it overlooks another key area of forgetfulness in machine learning: textual forgetfulness in large language models (LLMs) [1-4]. The T2I task in the paper only involves generation by DMs, while LLMs have unique characteristics in terms of autoregressive text generation. It is unclear whether the UMA framework can be directly transferred to LLMs or what modifications are required.
2.	As can be seen in Table 1, some forgetting methods (such as FT) improve robustness but also lead to a significant drop in the model's test accuracy (TA) on the holdout set. This suggests that the model may be over-forgetting, sacrificing generalizability. While the paper notes TA, it does not delve into this "utility-robustness" trade-off, which is crucial for evaluating the practicality of a forgetting algorithm. 
3.	The UMA attack is essentially an optimization problem, which requires finding an optimal $\delta_x$ for each forgotten example $x_i$. The paper does not discuss the computational cost of this attack in detail.

[1] Reversing the forget-retain objectives: An efficient llm unlearning framework from logit difference, NeurIPS 2024

[2] Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond, ICLR 2025

[3] Single image unlearning: Efficient machine unlearning in multimodal large language models, NeurIPS 2024

[4] SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders, ICCV 2025

### Questions
1.	In Table 3 for the T2I task, many methods have very high pre-attack ASR (attack success rate) (e.g., 93.94% for AdvUnlearn on Church). This makes the "post-attack" ASR gain (100%) difficult to interpret: is this the effectiveness of the UMA attack, or a complete failure of the original forgetting algorithm?
2.	In Table 2 of the I2I task, SalUn’s “unbound” attack outperforms the $8/255$ attack (133.82) in FID on the forgotten set (94.15), which is counterintuitive (a stronger attack should lead to worse recovery quality, i.e., higher FID).
3.    More comments please refer to the weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new benchmark for evaluating the post-unlearning robustness. The proposed benchmark is built upon an attack framework that introduces malicious perturbations to unlearned data in the post-unlearning stage, aiming to recover the model’s prediction performance to that of the original, pre-unlearned model. This benchmark considers multiple unlearning scenarios, including classification, image-to-image reconstruction, and text-to-image synthesis. The authors state that the benchmark code will be released upon publication.

### Strengths
1. The proposed benchmark covers a broad range of unlearning tasks, including classification, image-to-image reconstruction, and text-to-image synthesis.
2. The benchmark provides evaluations across 15 representative unlearning methods.

### Weaknesses
1. Limited experimental scope. As a benchmark paper, the experimental section appears rather limited. The range of models and datasets used is relatively narrow, and the discussion of experimental observations for each setting is insufficiently detailed.
2. Overly strong threat model. The attack scenario assumes full knowledge of both the original and unlearned models. While this setup effectively represents a worst-case analysis, it is arguably unrealistic in practical contexts. Incorporating black-box or query-based attack experiments would improve the paper’s practical relevance.
3. In Proposition 2 and Proposition 3, the output space of $f_u$ is not formally defined. For instance, is the output a probability vector in class unlearning?
4. The evaluation of membership inference attacks relies solely on shadow model-based approaches. More advanced techniques, such as LiRA-based MIAs, should be considered to provide a more accurate assessment of residual information leakage.
5. For the text-to-image attack scenario, it appears that an existing attack method is directly employed rather than a newly proposed one, which may somewhat weaken the contribution of the paper.

### Questions
1. Regarding Table 1: From an optimization standpoint, it is quite surprising that the retrain method remains almost unaffected under optimized malicious perturbations. Prior adversarial attack research suggests that targeted attacks should effectively manipulate model outputs toward desired labels. Could the authors elaborate on the potential reasons for this?
2. Could the proposed benchmark be extended to include large language model unlearning scenarios?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes RUB, a benchmark and evaluation protocol for robust machine unlearning across three tasks: classification, image-to-image (I2I) inpainting, and text-to-image (T2I) diffusion. The underlying framework in RUB is the Unlearning Mapping Attack (UMA), a white-box post-unlearning attack instantiated per task (e.g., perturbing inputs or prompts) to probe for residual knowledge of the model about the forgotten samples/classes. Experiments span 15 unlearning methods; results suggest many pass standard verification but fail under adversarial probing.

### Strengths
1. The paper focuses on an important problem with machine unlearning methods which might be overlooked by many of the works that propose a new unlearning method. 

2. By unifying the three tasks authors show that this problem exists for various tasks. 

3. The paper touches on a timely problem as the introduced methods for machines have ramped up.

### Weaknesses
1. The paper is not up to date with the most recent unlearning methods and evaluation metrics. For example, MIA used for evaluation of unlearning in discriminative models were introduced in 2017, while several seminal and more effective methods have been published since then, look at [1,2] just to name a few. Some recent unlearning methods have adapted these methods to the setting of unlearning [3]. There have even been new MIA specifically designed for evaluating the effectiveness of unlearning [4,5]. Ignoring the whole advances in membership inference attacks in a newly available benchmark, not only does not help the community in advancing their unlearning methods, but will provide a wrong point of reference for the future works because they might build their evaluations based on this new benchmark despite the fact the the underlying metrics are outdated.

2. Even the unlearning methods are not updated and the most recent ones are published in early 2024. Considering that this work, if accepted, will be published in early 2026, that would be a very large gap and ignores the rapidly flourishing advances in unlearning methods and might drive the community back by providing a wrong point of reference for comparisons and evaluations. Just to name a few missing works, please see: [3,6,7,8,9,10,11]

3. The presented work lacks novelty, and most of it seems more like a concatenation of the following papers: [12,13]. More specifically, authors have used the exact attack methodology from [12] and [13] and even the experiments are very similar for T2I to [13]. It is expected to at least have a much more comprehensive set of experiments for a paper whose focus is merely on introducing a benchmark.

4. Given that the proposed threat model has been introduced by recent work in unlearning for both I2I and T2I tasks, one contribution of the presented work can be considered as unifying the threat model for the three tasks. However, I am not even sure if this unification is even necessary given that they come from different domains and each task requires its own specialized methodology and evaluation metrics and an author in any of these three domains would not need to look at the results for other tasks. For one of the domains I2I, which is newly introduced in [12] with only two of the methods applicable, is a benchmark even needed?

5. In ML literature when the reader sees propositions they expect some theoretical results with accompanying proofs. Here the propositions seem to be some general guidelines or hypotheses. And they are not novel as well and have been used in prior work to generate such attacks on unlearning models in T2I and I2I tasks.

6. In line 421 authors mention that in choosing the hyper-parameters the trade-off of TA with strength of unlearning has to be considered, and they do not mention how they decide about this trade-off? Maybe an average gap, similar to what has been reported in SalUn [14] would be a good combination of the metrics they use for evaluations.


[1] N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis and F. Tramèr, "Membership Inference Attacks From First Principles," 2022 IEEE Symposium on Security and Privacy (SP), San Francisco, CA, USA, 2022, pp. 1897-1914, doi: 10.1109/SP46214.2022.9833649.

[2] Zarifzadeh, S., Liu, P., & Shokri, R. (2024, July). Low-cost high-power membership inference attacks. In Proceedings of the 41st International Conference on Machine Learning (pp. 58244-58282).

[3] Ebrahimpour-Boroojeny, A., Sundaram, H., & Chandrasekaran, V. Not All Wrong is Bad: Using Adversarial Examples for Unlearning. In Forty-second International Conference on Machine Learning. 

[4] Hayes, J., Shumailov, I., Triantafillou, E., Khalifa, A., & Papernot, N. (2025, April). Inexact unlearning needs more careful evaluations to avoid a false sense of privacy. In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML) (pp. 497-519). IEEE.

[5] Cadet, X. F., Borovykh, A., Malekzadeh, M., Ahmadi-Abhari, S., & Haddadi, H. (2025, June). Deep Unlearn: Benchmarking Machine Unlearning for Image Classification. In 2025 IEEE 10th European Symposium on Security and Privacy (EuroS&P) (pp. 939-962). IEEE.

[6] Cywiński, B., & Deja, K. SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders. In Forty-second International Conference on Machine Learning.

[7] Georgiev, K., Rinberg, R., Park, S. M., Garg, S., Ilyas, A., Madry, A., & Neel, S. (2025). Machine unlearning via simulated oracle matching. In The Thirteenth International Conference on Learning Representations.

[8] Zhang, B., Dong, Y., Wang, T., & Li, J. (2024, July). Towards Certified Unlearning for Deep Neural Networks. In International Conference on Machine Learning (pp. 58800-58818). PMLR.

[9] Wu, J., & Harandi, M. (2025). Munba: Machine unlearning via nash bargaining. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4754-4765).

[10] Cha, S., Cho, S., Hwang, D., Lee, H., Moon, T., & Lee, M. (2024, March). Learning to unlearn: Instance-wise unlearning for pre-trained classifiers. In Proceedings of the AAAI conference on artificial intelligence (Vol. 38, No. 10, pp. 11186-11194).

[11] Bonato, J., Cotogni, M., & Sabetta, L. (2024, September). Is retain set all you need in machine unlearning? restoring performance of unlearned models with out-of-distribution images. In European Conference on Computer Vision (pp. 1-19). Cham: Springer Nature Switzerland.

[12] Li, G., Hsu, H., Chen, C. F., & Marculescu, R. Machine Unlearning for Image-to-Image Generative Models. In The Twelfth International Conference on Learning Representations.

[13] Zhang, Y., Jia, J., Chen, X., Chen, A., Zhang, Y., Liu, J., ... & Liu, S. (2024, September). To generate or not? safety-driven unlearned diffusion models are still easy to generate unsafe images... for now. In European Conference on Computer Vision (pp. 385-403). Cham: Springer Nature Switzerland.

[14] Fan, C., Liu, J., Zhang, Y., Wong, E., Wei, D., & Liu, S. (2023). Salun: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation. arXiv preprint arXiv:2310.12508.

Minor weaknesses:

1. The work mentions unlearning accuracy has been used as the evaluation metric in some prior works. That is not the case in recent papers as they have updated their methodologies. The papers that only use unlearning accuracy are outdated and cannot be a point of comparison anymore.

2. Lower values of UA does not necessarily indicate better unlearning (line 255)

3. You also mention the setting of class unlearning in line 215 for discriminative models, but that is not a part of the presented benchmarks.

4. The subscript for function $f$ in $f_u(.,\theta^u)$ and $f_r(.,\theta^r)$ seems confusing as unnecessary because it initially conveys the idea that it shows the underlying data used for training, but then it is differently used for $f_u$ and it is just redundant subscript in current form.

5. The assumption in equation one about how the desired behavior of the unlearned model is misleading as it suggests that the expectation of the difference of output on the forget samples should become unboundedly larger. This is not a correct assumption as mentioned by prior work and might even lead to what is called overunlearning [1]

6. In proposition 2, it think you mean x+\delta_x not \delta_x? otherwise \delta_x seems to be overloaded by proposition 2 and 3 which is confusing for the reader.

7. “following prior arts”? In line 218


[1] Shi, W., Lee, J., Huang, Y., Malladi, S., Zhao, J., Holtzman, A., ... & Zhang, C. (2024). Muse: Machine unlearning six-way evaluation for language models. arXiv preprint arXiv:2407.06460.

### Questions
1. Are there any justification for choosing a MIA method published in 2017?

2. Will the authors be able to add the several missing recent methods in unlearning to their paper?

3. Details about the hyper-parameters of the unlearning methods and some details such as the number of iterations for PGD attack is missing. Would it be possible for the authors to include those details?

4. Are there any theoretical guarantees for the presented propositions that are missing in the current submission? Having the empirical evidence would be fine, but with the current wording it seems as if they are accompanied by theoretical proofs.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a unified framework for assessing Machine Unlearning algorithms and specifically focusing on the robustness of these algorithms. The authors evaluate 15 state of the art unlearning algorithms across 3 types of tasks: classification, image-to-image reconstruction and text-to-image generation. The authors also introduce a unified attack mechanism called Unlearning Mapping Attack (UMA) which employs worst-case white-box adversary, granting full knowledge of the models and forget sets to modify the inputs to the models only. The key findings are that all the state of the art unlearning methods are fragile to adversarial perturbations.

### Strengths
- The paper identifies an important failure case of the current state of the art machine unlearning algorithms.
- Extensive experiments across a variety of domains (classification, I2I, T2I) and unlearning algorithms (15 algorithms).
- Easy to read and follow.

### Weaknesses
- Limited technical novelty as UMA seems like essentially the same as standard gradient based adversarial attack.
- There are already adversarial attacks and robust machine unlearning methods for LLMs.

### Questions
- Why did the authors not include LLMs?
- For the T2I attacks, can the authors provide some examples of what the adversarial text input looks like?

### Soundness
3

### Presentation
3

### Contribution
2
