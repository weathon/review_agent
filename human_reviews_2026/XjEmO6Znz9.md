# YaPO: Learnable Sparse Activation Steering Vectors for Domain Adaptation

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 2

## Abstract
Steering large language models (LLMs) through activation interventions has emerged as a lightweight alternative to fine-tuning for alignment and personalization. Recent work on Bi-directional Preference Optimization (BiPO) shows that dense steering vectors can be learned directly from preference data, in a Direct Preference Optimization (DPO) fashion, enabling control over truthfulness, hallucinations, and safety behaviors. However, dense steering vectors often entangle multiple latent factors due to neuron multi-semanticity, which limits their effectiveness and stability in fine-grained settings such as cultural alignment, where closely related values and behaviors (e.g., among Middle Eastern cultures) must be distinguished. 
In this paper, we propose $\textbf{Yet Another Policy Optimization (YaPO)}$, a $\textbf{reference-free}$ method that learns $\textbf{sparse steering vectors}$ in the latent space of a $\textbf{Sparse Autoencoder (SAE)}$. 
By optimizing sparse codes, YaPO produces disentangled, interpretable, and efficient steering directions. 
Empirically, we show that sparse steering vectors converge faster, achieve lower training and evaluation loss, and remain more stable throughout training compared to dense counterparts. 
Beyond cultural alignment, YaPO generalizes to diverse alignment-related behaviors studied in BiPO, including Hallucination, Wealth-Seeking, Jailbreak, and Power-Seeking. 
Our results demonstrate that YaPO sparse steering provides a general recipe for efficient, stable, and fine-grained alignment of LLMs, with broad implications for controllability and domain adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper centers on activation steering, extending Bi-directional Preference Optimization (BiPO) by applying it within the sparse space of a Sparse Autoencoder. The experiments primarily target cultural adaptation, with results on Gemma 2 showing superior performance over BiPO.

### Strengths
1. The proposed approach is intuitive and easy to understand.


2. In cultural adaptation tasks, it achieves significant improvements over the baseline method, BiPO.

### Weaknesses
1. The experiments were only conducted on Gemma2-2B, and the results need to be validated on more models to demonstrate generality.

2. The baselines are limited. The paper only compares against BiPO, while there are many existing works on sparse activation steering that should be included for a more comprehensive comparison.

3. The tasks are restricted to cultural adaptation, and although the authors created their own dataset, the description of the task is vague. It is difficult to understand the actual goal of this task.

4. In Section 4.4 Generalization to Other Domains, the authors only report results on hallucination reduction without providing any details about the experimental setup, which makes this section confusing.

5. Regarding method design, the authors follow BiPO’s bidirectional training framework, but they do not specify what the target behavior and opposite behavior are for the given task. Moreover, the evaluation does not mention bidirectional steering, leaving the purpose of this design unclear.

6. Minor: Line 290 and 785 contain invalid pointers.

### Questions
See Weaknesses.

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
4

### Summary
This paper introduces YaPO (Yet Another Policy Optimization), a method for learning sparse steering vectors in the latent space of Sparse Autoencoders (SAEs) to improve LLM alignment and domain adaptation. Unlike dense steering methods such as BiPO that operate directly in activation space and suffer from neuron multi-semanticity, YaPO optimizes sparse codes using a bi-directional preference optimization objective, producing disentangled and interpretable steering directions. The authors focus on cultural alignment as a case study, curating a new multilingual dataset covering 15 cultural contexts across 5 language families with both localized and non-localized prompts to measure the explicit-implicit localization gap.

The experimental results on Gemma-2-2B demonstrate that YaPO converges significantly faster than BiPO (under 150 steps vs. 600+ steps), achieves substantial performance improvements across multiple-choice questions (+14.7% average) and open-ended generation tasks, and remains more stable throughout training. YaPO also reduces the Performance-Normalized Localization Gap (PNLG) by 27.3% while improving Robust Cultural Accuracy (RCA) by 54.3%, indicating better consistency between localized and non-localized prompts. Beyond cultural alignment, the method generalizes to other alignment tasks such as hallucination mitigation, establishing sparse steering as a scalable approach for fine-grained LLM control.

### Strengths
- First method to combine preference optimization with sparse steering vectors in SAE latent space, addressing limitations of both dense steering (BiPO) and static sparse methods (SAS)

- Demonstrates order-of-magnitude faster convergence and consistent performance improvements across all evaluated languages and settings

- Curates a high-quality multilingual dataset (45,354 items) with careful controls for dialect, cultural validity, and localized/non-localized variants

- Introduces PNLG and RCA metrics that appropriately measure both absolute performance and robustness to implicit cultural cues

### Weaknesses
- My biggest concern with this paper is the lack of baselines regarding steering with SAE. The authors did not compare against some new baselines like ReFT-r1, RePS, HyperSteer, and EasyEdit2. Since these methods also leverage SAE-based representations for steering, this omission makes it difficult to assess whether YaPO's improvements are genuinely novel.

- I am a little bit concerned about the limited model coverage. YaPO is only evaluated on Gemma-2-2B (briefly mentions Gemma-2-9B), lacking evidence of scalability to larger models or different architectures (Llama, Qwen, etc.). There are also SAEs provided for models like Pythia and Llama in SAELens, and therefore I think more experiments are reasonable and necessary.

- The interpretability claims are relatively unclear. While claiming "interpretable" steering, the paper lacks systematic analysis of what individual sparse features encode or how they differ from BiPO's dense features. Some automatic annotations with feature activations analyzed by LLMs could be a good supplementary material to strengthen these claims.

- The cultural dataset focuses on specific countries but may not capture within-country diversity; Western "control" answers may introduce bias. The authors could consider using more datasets like those used in CAA and Axbench to demonstrate broader applicability.

### Questions
1. Why did the author not compare against recent SAE-based steering methods like ReFT-r1, RePS, HyperSteer, and EasyEdit2?

2. Can the author provide results on larger models (e.g., Gemma-2-27B, Llama-3) or different architectures (Qwen, Pythia) to demonstrate scalability?

3. What specific concepts do the learned sparse features encode, and how do they differ from BiPO's dense features? 

4. Have the author tested YaPO on established cultural alignment benchmarks like those used in CAA and Axbench to validate broader applicability?

5. How does the author method handle within-country cultural variations, given that your dataset focuses on country-level differences?

### Soundness
2

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
4

### Summary
In this paper, the authors proposed YaPO (Yet Another Policy Optimization) which is an optimization-based approach to find steering vectors with the help of a pretrained SAE (sparse auto-encoder) of the target model. Specifically, YaPO trains the steering vector in a very similar way to BiPO but moves the steering vector from the model's hidden representation to the sparse features' activation, in hope that it will mitigate the multisemanticity issues in BiPO steering vectors. YaPO is only tested on Gemma-2 (and only results on the 2B variant are disclosed) with the off-the-shelf SAE Gemma Scope and primarily for a cultural localization task with a dataset the authors collected on their own, where BiPO yields noticeable improvement over both BiPO and the model w/o steering. The authors also shared some information on YaPO's performance in steering against hallucination which also outperforms BiPO and the model w/o steering.

### Strengths
+ It is good to see more research on joining SAE and steering vectors.
+ The cultural localization problem the authors put forward and gathered a dataset for is an interesting problem and can be a good addition to existing tasks for benchmarking model behavior manipulation.

### Weaknesses
+ The idea of bridging SAE and steering vectors are not exactly new. For instance [1] and [2] both have investigated how sparsity/monosementicity helps regularizes representation steering. In a way, YaPO can be considered merely using BiPO to achieve [2].
+ While BiPO is a very good paper to base on, using it as the only baseline is inadequate, given that there are existing works that shared the same design as mentioned above.
+ The experiments are also limited. 
    + Gemma is the only model being evaluated meaning that YaPO has never been evaluated on a model (mostly Llama, vicuna and their variants) that BiPO was originally tested on, even though there is off-the-shelf SAEs for them like Llama Scope as well. The performance on a single model is not as convincing as that across multiple models.
    + The major experiments are conducted for the cultural localization task with datasets the authors built on their own without sharing a single example except for the 2 phrases in section 3.1. The authors also included some results about hallucination without any details other than the scores. BiPO was said to perform even worse than no steering which contradicts its reported performances on other models. The authors also fail to demonstrate if YaPO would undermine general utility of the models with e.g. MMLU benchmark.

    If the authors do want to closely follow BiPO, they are suggested to at least do the experiments that BiPO has done with the exact settings. Simply be comparing Gemma Scope and Llama Scope it is not hard to find that they are very different in terms of how the sparse features look like, so it is possible that YaPO is only better for Gemma based models on very specific tasks.
+ YaPO is said to be more efficient to train but that claim assumes that there is a pretrained SAE available for whichever model one want to use YaPO on. However in reality, training an SAE is actually way more consuming than either BiPO or YaPO and one cannot always expect a pretrained SAE to be readily available. People are interested in joining steering vectors and SAEs because they, to some extent, comprise a dual formulation of each other — one building up each single feature vectors bottom up from preference datasets, the other finds all possible feature vectors top down in an unsupervised fashion. So having a pretrained SAE at hand naturally gives YaPO a leverage by having the majority of the work done already so it is not surprising at all YaPO optimization could be faster.

1. Chalnev, Sviatoslav, Matthew Siu, and Arthur Conmy. "Improving steering vectors by targeting sparse autoencoder features." arXiv preprint1 arXiv:2411.02193 (2024).
2. He, Zirui, et al. "SAE-SSV: Supervised Steering in Sparse Representation Spaces for Reliable Control of Language Models." arXiv preprint arXiv:2505.16188 (2025).

### Questions
Please refer to the weakness for the concerns I have about this paper. Here I am just listing a few questions to facilitate the understanding of my concerns. 
+ Could it be 2B model is too small for the tasks?
+ Could it be Gemma's SAE being different from Llama's?
+ Did you redo the layer selection and hyperparameter search for BiPO?
+ How does YaPO compare to other approaches to enhance steering vectors with SAE guidance?
+ What if there is no off-the-shelf SAE? Can you train a sparse steering vector without a pretrained SAE?

### Soundness
2

### Presentation
2

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
Instead of learning a steering vector, they propose YaPO, where they instead learn to steer sparse features of a SAE. They show this on a cultural benchmark that they curate, and show that the method converges faster than and outperforms BiPO on that benchmark, as well as a hallucination benchmark.

### Strengths
* The method converges much faster than BiPO and outpeforms BiPO in the cultural benchmark.
* They also perform their method on BiPO's benchmarks, but only on the hallucinations dataset (which they note in their Limitations section)

### Weaknesses
* The work was done only on a single 2B model. The 9B variation was mentioned once in Limitations with no further details in the main body or Appendix.
* The paper claims to produce more interpretable steering directions, but fails to do any work on interpreting the steering direction. They note that this is "beyond the scope of this paper" in the Limitations, but I disagree, as merely using the sparse autoencoder feature basis is not sufficient to make things more interpretable.
* While the dataset/benchmark is claimed as a main contribution, there is barely any information about it in the main body.

### Questions
In addition to the three weaknesses mentioned, I have the following questions:
* How is this work different from [SAE TS](https://arxiv.org/abs/2411.02193)? (I note that SAE TS is not peer-reviewed and was not factored into my accept/reject decision, but nevertheless the paper should be cited and discussed as it predates this work by more than a year).
* Is there any particular reason you choose to report the "Egypt" evaluation performance only for Figure 1? Is it possible to instead report the average difference over all categories between YaPO and BiPO across training epochs? What does that look like?

I also have the following feedback:
* **There are hallucinated citation authors**, like "Steering llama 2 via contrastive activation addition" being attributed to a "Nathan Rimsky". "Steering Language Models With Activation Engineering" also has hallucinated author names.
* Appendix B is incomplete, especially its last paragraph which seems to be a broken transcript.
* Line 290: Appendix reference missing
* L418: Typo immediately after Activation Engineering (the full-stop).
* L785: Figure missing
* L302 Incomplete line "we observe that the performance improvement is stable and consistent throughout the epochs for YaPO while BiPO"...

### Soundness
1

### Presentation
2

### Contribution
1
