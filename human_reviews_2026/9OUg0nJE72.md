# On-the-Fly Adaptation to Quantization: Configuration-Aware LoRA for Efficient Fine-Tuning of Quantized LLMs

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
As increasingly large pre-trained models are released, deploying them on edge devices for privacy-preserving applications requires effective compression. Recent works combine quantization with the fine-tuning of high-precision LoRA adapters, which can substantially reduce model size while mitigating the accuracy loss from quantization. However, edge devices have inherently heterogeneous capabilities, while performing configuration-wise fine-tuning for every quantization setting is computationally prohibitive. In this paper, we propose CoA-LoRA, a method that dynamically adjusts the LoRA adapter to arbitrary quantization configurations (i.e., the per-layer bit-width choices of a pre-trained model) without requiring repeated fine-tuning. This is accomplished via a configuration-aware model that maps each configuration to its low-rank adjustments. The effectiveness of this model critically depends on the training configuration set, a collection of configurations chosen to cover different total bit-width budgets. However, constructing a high-quality configuration set is non-trivial. We therefore design a Pareto-based configuration search that iteratively optimizes the training configuration set, yielding more precise low-rank adjustments. Our experiments demonstrate that, unlike the state-of-the-art methods that require fine-tuning a separate LoRA adapter for each configuration, CoA-LoRA incurs no additional time cost while achieving comparable or even superior performance to those methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CoA-LoRA, a fast (on-the-fly) adaptation of LoRA for various backbone model quantization configurations. Compared to the baselines, which train separate LoRA adapters for each quantization configuration, the proposed method trains a single configuration-aware model $\theta$ that generates adjustment matrices for the LoRA adapters to adapt to a given configuration. To train the configuration-aware model, the paper proposes a Pareto-based quantization configuration search, which makes the training set (configurations) Pareto-optimal in the configuration space. The experimental results show that the proposed method achieves comparable or better accuracy with a single trained model $\theta$.

### Strengths
* The motivation is strong and practical. The paper proposes a method that does not require fine-tuning when changing the quantization configuration. This would save significant GPU-hours.
* The experimental results show comparable or even better accuracy compared to the baselines.
* The idea is novel and insightful. The paper proposes a meta-learning approach with a Pareto-based search algorithm for improved quantization accuracy.

### Weaknesses
* Most experiments were conducted on the RoBERTa-Large model (0.3B), which is hard to consider an LLM. In the six experimental results presented in the evaluation section, only a single experiment was conducted on the Qwen and LLaMA models, reporting their perplexity. Therefore, as the title suggests that the method targets LLMs, there should be more focus on actual LLMs.
* The paper is not fully reproducible. There is a lack of crucial methodological details, and no code is provided. Below is a list of information that should be included.

  * The architecture of the configuration-aware model $\theta$ is not described; the only information given is that it receives a quantization configuration and outputs an r×r matrix.
  * Is the configuration-aware model trained layer-wise or model-wise?
  * The NormalFloat datatype was originally introduced as a 4-bit datatype (NF4). How does the paper handle the k-bit NF type?
  * How are layer-wise bitwidth configurations chosen? The proposed method uses layer-wise configurations, but the paper does not explain how they are selected for evaluation.
  * A complete algorithm for using $\theta$ during quantization and training is missing. How is the backbone model trained and quantized? If the model is trained and then quantized, it consumes more memory during fine-tuning, which is precisely the issue the baseline methods aim to solve. If the model is quantized and then trained, how is the initial quantization configuration chosen, and which configuration is used for training the backbone models and LoRA adapters on the target task?
* This reviewer believes that some content in the paper should be revised.

  * line 98: The citation for NormalFloat is incorrect (Dettmers et al., 2022 → 2023).
  * line 99: NormalFloat does not use reduced exponent and fractional bits but employs non-uniform quantization based on a lookup table.
  * line 101: The cited papers do not contain information about NF suffering from accuracy degradation at very low bit-widths.
  * line 187: “NF is particularly suitable in our setting because it supports block-wise quantization at a finer granularity”: this reviewer believes NF is not the only datatype supporting block-wise quantization, as any datatype can do so. Therefore, there should be another justification for choosing NF.
  * Across the paper: QLoRA is a more accurate name than “Q-LoRA” used in the paper.

### Questions
* How is the average bitwidth measured? Does the model use the same configurations across layers?
* How does the model performance in Figure 5 exceed full fine-tuning?
* How is the accuracy gap measured? Since the experimental results are obtained from multiple quantization configurations, is the accuracy gap averaged?
* In the ablation study (Fig. 8), how is the performance measured without the Gaussian process? Is it replaced with random sampling and freezing? If so, this reviewer believes random sampling should be repeated during training to ensure the training set size matches that used with the Gaussian process.

### Soundness
3

### Presentation
2

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
The authors propose a way to directly predict LoRA-weights suitable for a given quantization configuration of the base-model. 
The method solves a bi-level optimization problem. A mapping from the space of possible base-model quantization mappings to LoRA-weights is learned. In order to learn this mapping, a suitable training-set is tuned at the same time as the mapping. 
The resulting optimized training set along with the mapping makes it possible to find a quantization configuration setting for a given average bit-width constraint (Appendix A.2) and then immediately infer the LoRA weights that aims to recover the performance drop for that configuration setting. Consequently, it is possible to obtain LoRA-weights for a specific bit-width constraint without any necessary fine-tuning.

### Strengths
The paper presents an original idea, each step of the solution is well-motivated. The quality of writing is high, the paper is information-dense yet good to follow.

### Weaknesses
The main weakness I see with the paper is in its problem motivation. When does anyone require the ability to zero-shot LoRA weights for a new quantization configuration? From a research perspective, this problem fascinates me. From a practitioner's perspective, I don't know when I would ever need this. In deploying a model on edge devices, I can reasonably enumerate the set of accelerators I aim to target for a given model (or model-class) and therefore target the corresponding bit-width. On the other hand, a practitioner might care more about latency for a given bit-width of a specific layer - or have specific configuration constraints as far as hardware-level support for quantization configurations is concerned. I understand you choose NF, but I assume your method should generalize to other - practically relevant - formats. I see this as an opportunity to discuss and motivate your method in a wider context: We might not care only about storage, but also latency, under additional hardware constraints.

### Questions
Figure 7: Does "unseen" mean the configuration has never been part of $\mathcal{C}$ or does it mean it's not part of the final configuration set? 

What exactly is the configuration-aware model - and does it matter? You mention "light-weight", but how exactly do you compose [z, m, b] and U?

I find a conceptual break between the fact that your search-space lies in the continuous C, whereas valid configurations are discrete and enumerable. Do you make sure that your update (8) remains within valid bounds? How precisely do you evaluate f_1(C) for a given C (EQ 4) - or equivalently: what exactly is $\tilde{W}_C$ if C lies somewhere between the embeddings of the discrete quantization configurations?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors in this work propose CoA‑LoRA, a configuration‑aware approach that lets a single fine‑tuned adapter serve many quantization settings without re‑finetuning. Their approach conditions a lightweight adapter transformations on the quantization profile and integrates an Pareto configuration search to train over a diverse yet optimal configuration set. Experiments on GLUE and reasoning benchmarks show that CoA-LoRA maintains accuracy comparable to or better than per-configuration baselines while greatly reducing finetuning time and generalizing to unseen quantization settings. The paper’s claims are overall well-supported by results. The methodology is clearly described, and is easy to follow. The experimental setup tested on both GLUE and C4 and the ablations hints towards generalization of their claims. Reporting of training budgets is transparent and helps in comparing fairly. The authors could benefit from describing more detail on the mapper’s architecture and inference latency. This would help assess their proposed implementation's real-world feasibility. The paper is well‑structured and not too difficult to follow, with pseudocode in the appendix. Prior work is discussed appropriately. As an improvement, authors can consider providing more insights into mapper architecture’s detail for better reproducibility. The authors solve a timely and practical problem which if generalizable can help the broader community in efficiently finetuning quantized models. Through experiments, their design choice is motivated by an insightful observation that shared knowledge across configurations is mostly captured in L2​, while configuration-specific information resides in L1​. This is a new insight and  justifies adapting L2 through the lightweight mapping model.

### Strengths
1. The authors address an important practical problem of fine tuning of quantized models where per-configuration is not feasible. Their proposed method learns a mapping model which eliminates repeated adapter training while reducing the performance loss due to quantization. 

2. Their proposition is based on an important observation of L2 capturing configuration specific information. This is a new insight and justifies adapting L2 through a lightweight transformation. If generalized well, this could be very insightful for the community for future quantized finetuning works.

3. The EHVI‑driven, finite‑difference coordinate search coupled with Pareto filtering is well‑motivated for the accuracy–bit trade‑off.
The authors also demonstrate competitive performance across GLUE and commonsense reasoning tasks.

### Weaknesses
1. The paper has limited discussions on the architecture and parameter count of the configuration‑aware mapper ​ and the overhead of generating mapping for a new configuration. Including these discussions would clarify deployability and provide deeper insights to their method for larger audience adaptability. 

2. The method is evaluated solely on the NormalFloat (NF) quantization space and does not explore standard integer (INT) quantization schemes such as INT8 or INT4. Demonstrating that CoA-LoRA can generalize to widely used integer quantizers would strengthen the claim of broad applicability.

3. As highlighted in prior work such as QLoRA, the performance gap between pre- and post-fine-tuning typically widens as model size increases. Since CoA-LoRA is evaluated only up to 7B parameters, it remains unclear how the method scales to larger models. Including ablations on 14B-scale or higher architectures would provide valuable insight into whether the proposed configuration-aware adaptation maintains stability and efficiency at greater capacity.
4. The observation that L2​ captures shared knowledge across quantization configurations, while L1​ encodes configuration-specific information, is derived only from experiments on RoBERTa. It remains unclear whether this property generalizes to larger autoregressive or decoder-only architectures such as LLaMA or Qwen. A cross-model analysis of the ​ decomposition would help confirm whether the same structural separation holds beyond encoder-style models.

### Questions
1. The L1 and L2 separation is shown only on RoBERTa. Does this pattern hold for decoder-only models like LLaMA or Qwen? Also, does this characteristic change when the rank of the LoRA adapter is increased?

2. How does CoA-LoRA perform with integer quantization (INT4) instead of NormalFloat?

3. Can the method scale to 14B+ models, and what computational limits arise in training or search?

4. Does the configuration-aware mapping support or extend naturally to mixed-precision quantization, or is it limited to homogenous quantization of the model?

### Soundness
3

### Presentation
2

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
This paper tackles the issue that LLM models should be quantized multiple times if they are deployed on various devices.
It's because optimal quantization configurations differ according to target hardware.
To resolve this problem, the paper proposes CoA-LoRA that can find optimized LoRA adapters for quantization on-the-fly.

The key idea is to yield adjustment matrices that are used to reparameterize a LoRA adapter.
By generating an adjustment matrix to a target quantization configuration, an optimized LoRA adapter can be achieved.

Adjustment matrices are generated with Configuration aware model that is trained for yielding an adjustment matrix for each layer of LLM.
This model gets a quantization configuration as an input and returns adjustment matrices optimized for the configuration.

In order for this method to work well, Configuration aware model should be trained well with representative configuration settings.
This paper searches configurations with a Pareto-based Gaussian process and finite-difference gradient approximation for training Configuration aware model.
Starting from an initial configuration set, Pareto improved solutions against bit or loss are replaced or added to the configuration set.

With these proposals, the paper designs optimized LoRA adapters for various configuration settings.

### Strengths
- This paper aims to run an LLM on various quantization configurations caused by different hardware properties with minimal performance degradation.
Designing LoRA adapters per configuration is too costly, while using a single LoRA adapter on various bit settings leads to performance degradation.
This paper accurately recognizes these problems and proposes a solution.

- It is novel that distributes LoRA adapters on-the-fly by yielding adjustment matrices tailored for quantization configuration.
The proposed configuration search also helps train Configuration aware model to achieve better solutions.

- The paper shows the effectiveness of the proposal by executing various experiments with several datasets, model sizes, and LoRA ranks, in addition to an ablation study about configuration search.

### Weaknesses
- It is not clear that the experimental settings of other methods are exactly the same per layer, or only the average bits are set to the same.
Sensitivities of layers to performance can differ, and the performance degradation can be caused by it.
If the experiments are executed with exactly the same configurations, please clarify it.

- Figure 5 is in the main manuscript, but it is not referenced.

- Q-LoRA and LQ-LoRA are costly to execute with the whole bit settings, but the reviewer think that settings over 4-bits should be executed too, to compare the performances on higher bit settings.

- Recently, there have been several works that generate LoRA adapters according to conditions [1, 2]. However, the paper doesn't mention any of them. Discussing them in the final version of the paper would strengthen the paper.

[1] Shao, Yihua, et al. "In-Context Meta LoRA Generation." arXiv preprint arXiv:2501.17635 (2025).

[2] Xiao, Yicheng, et al. "LoRA-Gen: Specializing Large Language Model via Online LoRA Generation" International Conference on Machine Learning. PMLR, 2025.

### Questions
- In Figure 5, CoA-LoRA shows better results than full fine-tuning on QQP. The reviewer think that full fine-tuning is more likely to show better results compared to a quantized one.
Can the authors provide additional explanation about that?

- The proposed method facilitates on-the-fly adjustment of LoRA adapters. However, the previous works, such as LQ-LoRA adapts LoRA adapters to a single target setting directly.
Thus, the reviewer think that those direct adaptations are more likely to show better results on a specific target setting.
But CoA-LoRA outperforms other previous works even in their target settings. Can the authors provide your opinions on how this came out?

- As far as the reviewer understand, adjustment matrices are multiplied by a shared LoRA adapter. If so, how could the LoRA adapter, before adjustment can be achieved?

### Soundness
3

### Presentation
2

### Contribution
3
