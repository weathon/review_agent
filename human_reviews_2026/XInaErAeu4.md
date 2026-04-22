# Neurosymbolic Object-Centric Learning with Distant Supervision

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
Relational learning enables models to generalize across structured domains by reasoning over objects and their interactions. While recent advances in neurosymbolic reasoning and object-centric learning bring us closer to this goal, existing systems rely either on object-level supervision or on a predefined decomposition of the input into objects. In this work, we propose a neurosymbolic formulation for learning object-centric representations directly from raw unstructured perceptual data and using only distant supervision. We instantiate this approach in DeepObjectLog, a neurosymbolic model that integrates a perceptual module, which extracts relevant object representations, with a symbolic reasoning layer based on probabilistic logic programming. By enabling sound probabilistic logical inference, the symbolic component introduces a novel learning signal that further guides the detection of meaningful objects in the input. We evaluate our model across a diverse range of generalization settings, including unseen object compositions, unseen tasks, and unseen number of objects. Experimental results show that our method outperforms neural and neurosymbolic baselines across the tested settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the challenge of learning relational reasoning directly from raw perceptual data without explicit object-level supervision. The authors aim to bridge the gap between perception (inferring objects from images) and reasoning (logical inference over those objects) in order to achieve systematic generalization beyond the training distribution. In particular, they consider tasks where supervision is only given as high-level labels (distant supervision), such as the sum of digits in an image or the category of a hand of cards, rather than any annotations of individual objects.

Main Contributions: (1) The paper highlights limitations of prior neural and neurosymbolic models, which either required object-level supervision or assumed a fixed, known decomposition of the input. It proposes a new probabilistic generative formulation that learns object-centric representations directly from raw images with only task-level (distant) supervision. (2) It instantiates this formulation in the DeepObjectLog architecture, combining slot-based neural perception with probabilistic logic programming (ProbLog) for end-to-end differentiable reasoning. This design satisfies key requirements: handling a variable number of objects, reasoning under uncertainty, and inducing relational structure without direct object labels. (3) The authors introduce new benchmark tasks to evaluate relational generalization under weak supervision, including a MultiMNIST addition task (sum of digits in an image), a CLEVR-Addition task (adding counts of objects in paired images), and a newly created PokerRules dataset (classifying poker hands from images of playing cards). These tasks are specifically designed to test generalization to unseen object combinations, unseen object counts, and even novel task rules or classes not encountered in training. (4) Comprehensive experiments show that DeepObjectLog consistently outperforms baseline methods – both purely neural models and prior neurosymbolic approaches – especially in out-of-distribution generalization settings. The model is able to generalize to novel combinations of familiar objects, new object categories, and larger numbers of objects more robustly than the baselines, while also providing interpretable object-level predictions.

### Strengths
1.  Innovative Neurosymbolic Integration: The paper successfully combines slot-based object-centric perception with probabilistic logic programming. This is a novel integration – prior works did not achieve end-to-end training of object discovery guided by logical reasoning.
2. Effective Use of Distant Supervision: DeepObjectLog learns from only high-level task labels (distant supervision) and avoids any direct supervision on object existence or identity. This weakly-supervised paradigm is very appealing – it means the model can be trained on data where only aggregate or logical labels are provided.
3. Strong Generalization Performance: A major strength is the model’s performance on out-of-distribution generalization tests. DeepObjectLog dramatically outperforms baseline models in all the evaluated scenarios.

### Weaknesses
1. Dependence on Provided Logical Rules: A potential weakness is that the approach requires expert-defined logical rules for each new task. The logic component is not learned from data; it must be coded (in ProbLog) to describe the task’s outcome in terms of object properties.
2. Scalability and Complexity: The integration with probabilistic logic programming can introduce computational overhead. ProbLog, while efficient for moderate-scale inference, may struggle as the number of objects or logical facts grows large. The paper’s experiments involve at most 4–6 objects (cards or digits) and relatively simple rules. It is unclear how well the method scales to scenes with, say, dozens of objects or highly complex rule sets.
3. Synthetic Data and Visual Complexity: All evaluation tasks are on synthetic or simplistic visual data – e.g., MNIST digits on blank backgrounds, rendered CLEVR objects, and playing card images on a plain background. While this is suitable for controlled experiments, it leaves open the question of how the method handles more complex real-world images (with clutter, occlusion, varying backgrounds, etc.). 
4. Hyperparameter Sensitivity: The approach introduces a few new hyperparameters – notably the maximum number of slots $N$.

### Questions
1. In the PokerRules experiment, the model was tested on an unseen hand type (“three of a kind”) and achieved 72% accuracy OOD. Could you clarify how this was handled in terms of the logic program and training? Specifically, was the logical rule for detecting a “three of a kind” included in the model from the start, or introduced only at test time? If the rule was added at test time, it’s impressive that the model adapts; however, how was the model’s output space configured to allow a new class it never saw during training (did the output simply come from the logic query without needing a dedicated neural output)? A detailed explanation of how novel classes are introduced and whether any fine-tuning is required would be helpful.
2. How does the training and inference time scale with the number of slots (objects) and the complexity of the logic program? For instance, if we were to increase $N$ or add more complex rules (say involving all pairs of objects or higher-order relations), does the inference via ProbLog become a bottleneck? The current tasks have relatively small $N$ (up to 5 objects). It would be useful if you could comment on any experiments or observations about performance as $N$ grows. Additionally, do you use any optimizations for the logic module (e.g., caching of inference results, using the structure of the logic to prune possibilities) during learning? Understanding the computational limits will inform how far this approach might scale to more complex scenes.
3. To disentangle where the gains come from, did you consider a variant of the model that does not use the logical loss but still uses the slot-based architecture (for example, training the slots + classifier purely on predicting the global label, akin to a black-box neural network without logic)?

### Soundness
3

### Presentation
3

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
The paper proposes DeepObjectLog, a neurosymbolic architecture that couples an object-centric perceptual module (Slot Attention) with probabilistic logic programming (ProbLog) to learn from distant supervision: only global task labels and a logic program, with no object-level annotations. The model is trained with an objective that maximizes a discriminative lower bound via amortized MAP inference. The method is evaluated on MultiMNIST-Addition, PokerRules, and CLEVR-Addition, targeting compositional, task, and object-count generalization. Reported results show notable gains over CNN/Slot-based baselines and a NeSy baseline, and better OOD performance than prompting VLMs (LLaVA-1.5, Qwen2.5-VL) on the MultiMNIST dataset.

### Strengths
1. The paper is well written, clear, and easy to follow.
2. The evaluations focus on out-of-distribution tasks, which are highly relevant and remain partly unresolved in recent years.

### Weaknesses
1. Fairly simple evaluation: (1) datasets are synthetic toy datasets, which is acknowledged in the limitations, and (2) the tasks are quite simple reasoning problems that can be expressed as a multiclass classification problem. I would like to see comparisons on (1) a more complex, general task such as VQA and (2) at least one real-world (or close to real-world) dataset.
2. Some claims in the paper seem not to be correct, and some of the main conclusions are already well known in the object-centric (OC) community. Please see the questions below for details.
3. The paper does not analyze or quantify robustness to imperfect/underspecified rules or label noise, nor disentangle how much gain comes from the rules vs. the OC module.

### Questions
1. In Line 72 it is stated that OC models often rely on object-level supervision, either a label or annotation for each object. This is not true. Slot Attention is a similarity-based method that works like a soft k-means [14], and is generally trained fully unsupervised with a decoder and a reconstruction loss. The only information Slot-Attention–based OC models require beforehand is the number of slots (i.e., objects), which has been a downside shown to cause over- or under-segmentation [1]. However, [2, 3] address this limitation and show on-par or better performance compared with plain Slot Attention. I would appreciate it if the authors could elaborate.
2. Line 131: it is claimed that without logical feedback, current OC models cannot learn from relational structure unless it is explicitly specified. That is not accurate. Several works evaluate OC models on a wide range of downstream reasoning tasks, from RL to VQA, in both in-distribution [4, 5] and out-of-distribution [6] settings, showing that OC representations capture relational information well and can be on par with foundation models for ID and compositional OOD performance.
3. I would like to see DINOSAUR-like OC models [7], where a pretrained backbone provides representations to Slot Attention and reconstruction is done in latent space. I would also like to see the performance of a plain foundation model (e.g., DINOv3) with a small projection layer for the task as another baseline. With these additions, I believe the paper would become more relevant to a larger community.
4. Several claims and benefits of the proposed approach are already well-known, and some are properties of Slot Attention which are well-known in the OC community. For example, test-time generalizability of Slot Attention to different numbers of objects and out-of-distribution generalization of OC models have been analyzed in [6, 13]. Furthermore, several works have shown the shortcomings of VLMs on compositional generalization tasks [8, 9, 10, 11, 12]. I would appreciate it if the authors could elaborate on these works and how the paper relates to these prior works. Additionally, please elaborate on what are the implications and novel insights of the work and the impact the authors would expect for the community.
5. Regarding weakness #3, how sensitive is performance to inaccurate or incomplete logic rules (e.g., missing predicates, conflicting constraints)? Also, related to weakness #1, how can such rules be defined in more general settings for tasks such as captioning or VQA, and is there a practical limit to how far we can go with neurosymbolic AI?

[1] Zimmermann, Roland S., et al. "Sensitivity of slot-based object-centric models to their number of slots." _arXiv preprint arXiv:2305.18890_ (2023).

[2] Fan, Ke, et al. "Adaptive slot attention: Object discovery with dynamic slot number." _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_. 2024.

[3] Liu, Hongjia, et al. "MetaSlot: Break Through the Fixed Number of Slots in Object-Centric Learning." _arXiv preprint arXiv:2505.20772_ (2025).

[4] Mamaghan, Amir Mohammad Karimi, et al. "Exploring the Effectiveness of Object-Centric Representations in Visual Question Answering: Comparative Insights with Foundation Models." The Thirteenth International Conference on Learning Representations. 2025.

[5] Yoon, Jaesik, et al. "An investigation into pre-training object-centric representations for reinforcement learning." _arXiv preprint arXiv:2302.04419_ (2023).

[6] Kapl, Ferdinand, et al. "Object-centric representations generalize better compositionally with less compute." _ICLR 2025 Workshop on World Models: Understanding, Modelling and Scaling_. 2025.

[7] Seitzer, Maximilian, et al. "Bridging the gap to real-world object-centric learning." _arXiv preprint arXiv:2209.14860_ (2022).

[8] Huang, Irene, et al. "Conme: Rethinking evaluation of compositional reasoning for modern vlms." _Advances in Neural Information Processing Systems_ 37 (2024): 22927-22946.

[9] Hsieh, Cheng-Yu, et al. "Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality." _Advances in neural information processing systems_ 36 (2023): 31096-31116.

[10] Wu, Xindi, et al. "Conceptmix: A compositional image generation benchmark with controllable difficulty." _Advances in Neural Information Processing Systems_ 37 (2024): 86004-86047.

[11] Li, Baiqi, et al. "Genai-bench: Evaluating and improving compositional text-to-visual generation." _arXiv preprint arXiv:2406.13743_ (2024).

[12] Kempf, Elias, et al. "When and How Does CLIP Enable Domain and Compositional Generalization?." _arXiv preprint arXiv:2502.09507_ (2025).

[13] Dittadi, Andrea, et al. "Generalization and robustness implications in object-centric learning." _arXiv preprint arXiv:2107.00637_ (2021).

[14] Locatello, Francesco, et al. "Object-centric learning with slot attention." _Advances in neural information processing systems_ 33 (2020): 11525-11538.

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
3

### Summary
This paper aims to integrate a neurosymbolic reasoning component into a object-centric architecture and learn the model in an end to end fashion. To this end, the authors leverage an object centric approach which yields explicit uncertainty estimates over the presence and class of an object and integrate it with the ProbLog probabilistic programming framework. On several logical reasoning task across a few simple image datasets, the authors show superior logical reasoning performance using this model relative to baseline approaches.

### Strengths
* The paper is very well written and is easy to follow.

* The paper tackles an ambitious problem, namely, performing logical reasoning over symbols by leveraging object-centric representations.

* Integrating ProbLog with a slot-based neural network model is novel to the best of my understanding and is an interesting approach.

* The experimental section is well presented and the results showing the benefits of DeepObjectLog are convincing.

### Weaknesses
* I found the presentation in Section 3.4 a bit confusing regarding how ProbLog is integrated into the model. Is it the case that the logical rule/task to be performed, e.g., add the two numbers in the image, is defined a priori, or is the task also inferred from observed data.

* I can imagine DeepObjectLog working well in situations in simple logical reasoning task as the authors tested, however, I am skeptical of how the method will perform for more complex reasoning task in which the symbols and algorithm/logical program to be executed are less straightforward to define. For example, how do the authors imagine their method would perform on the ARC challenge [1]?

* The authors rely on a masked decoder which poses a limitation in terms of scalability. I believe it is important to understand whether DeepObjectLog can leverage more complex decoders such as the Transformer decoders in [2, 3].

* I believe the statement "However, these models often rely on object-level supervision, meaning they require a label or annotation for each object in the image" in line 72 is incorrect. All of the models the authors cite in the previous sentence are unsupervised.

### Questions
* Is the logical rule/program hard coded for a given task or is it inferred from the task?

* How do the authors imagine their approach would work for more complex reasoning task such as the ARC challenge?

* Do the authors believe that more scalable decoders can be integrated into their approach?

**References**

[1] Chollet et. al 2019, On the Measure of Intelligence

[2] Singh et. al 2021, Illiterate DALL-E Learns to Compose

[3] Brady et. al, 2024, Interaction Asymmetry: A General Principle for Learning Composable Abstractions

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work proposes an object-centric neurosymbolic approach that can be trained end-to-end (i.e. via distant supervision). By combining and object-centric visual encoder with a probabilistic symbolic reasoning framework, the model combines the perceptual strength of neural networks with the reasoning strength of symbolic models, without the need for intermediate supervision of perceptual representations (which is required by most neurosymbolic approaches). The approach is tested on an MNIST arithmetic task and classification of poker hands, demonstrating improved out-of-distribution generalization.

### Strengths
- The proposed model overcomes one of the primary weaknesses of neurosymbolic approaches, which is that they typically depend on very detailed object-level annotations to train the perceptual encoder. This approach more deeply integrates the neural/perceptual and symbolic/reasoning components, enabling it to be trained based on downstream task error.
- The model demonstrates promising results on various out-of-distribution generalization settings.
- The paper is very clearly written, and nicely frames the previous work and challenges in this area.

### Weaknesses
- The primary limitation is that only synthetic tasks are investigated. Additionally, these tasks both involve relatively simple classification, so they do not provide the strongest test of the symbolic component's reasoning abilities. The CLEVR-addition dataset is a step in the right direction, but this task still involves simple, synthetic images and limited (one-step) reasoning. It would be more compelling if the model could be extended to tasks that require multi-step reasoning and inference, and more complex, naturalistic images. 
- Regarding the attention masks shown in the supplementary material, first it might be better to visualize these by applying the attention weights to the input image, or by showing the slot-specific reconstructions. Second, the results suggest that the model may not be as object-centric as is desired. In particular, for conditions involving one or just a few objects, it looks like multiple slots are being used to encode the same object.

### Questions
- Can the proposed approach be readily extended to more complex settings?
- To what extent are the learned representations actually object-centric?

### Soundness
3

### Presentation
3

### Contribution
2
