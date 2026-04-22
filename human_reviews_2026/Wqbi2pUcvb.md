# Unlocking the Power of Layer By Layer Training For LLM

- Avg Score: 3.50
- Decision: Reject
- Scores: 0, 6, 2, 6

## Abstract
Layer-wise (LW) training of deep neural networks has long been associated with memory and parallelism advantages, 
yet it suffers from information degradation and poor convergence in deep architectures. 
Recent work attributes these issues to the loss of input information and the lack of layer-role differentiation, as measured
by the Hilbert-Schmidt Independence Criterion (HSIC).

In this paper, we present a novel algorithm that enables full end-to-end training of ResNet-18/ResNet-50 and end-to-end fine-tuning of Large Language Models (LLMs) using a modified LW approach, while minimizing performance degradation. 
Our fundamental contribution lies in the discovery that strategically reintroducing the final layers during LW training mitigates the convergence degradation typically observed during LW when compared to conventional end-to-end fine-tuning. 

We introduce Segmented Propagation (SegProp), a training paradigm that seamlessly integrates the computational efficiency of LW optimization with the representational power of global supervision. Quantitative results demonstrate substantial improvements
in convergence compared to standard LW fine-tuning of LLMs and compared to LW training of ResNet-18/ResNet-50. SegProp improves ResNet-50 accuracy on CIFAR-10 from 90.0\% (LW) to 94.3\%, approaching E2E training at 95.5\%. On ResNet-18, SegProp improves CIFAR-10 accuracy from 93.7\% (LW) to 95.2\%, closely matching E2E at 95.5\%. On Mistral-Nemo-Instruct-2407, SegProp segmented fine-tuning matches E2E MMLU (5-shot) performance (69.3\%), and for Llama3.1-8B-Instruct it achieves 78.9\% on Winogrande (5-shot), closely matching E2E fine-tuning at 79.1\%.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper investigates layer-by-layer training of LLMs from a fine-tuning perspective. It starts with the information bottleneck/loss of layer-by-layer training (though this is usually investigated from the pretraining perspective) and proposes to strategically reintroduce the final layers. The method has two stages: in the first stage, it jointly tunes the first several and the final several layers; in the second stage, it performs iterative layer-wise tuning of the intermediate Layers. Experiments are conducted on Mistral 7B, with fine-tuning on several downstream tasks.

### Strengths
* The paper provides extensive discussion of related work, though it does not yet compare against these works experimentally.

* The proposed method looks simple: it fine-tunes the first several and last several layers together, and then iteratively fine-tunes the intermediate layers.

### Weaknesses
* The biggest problem for me is that comparisons between layer-by-layer training and E2E training are usually investigated from a training-from-scratch/pretraining perspective, as reflected in the cited references. From this perspective, layer-by-layer training can be challenging in accuracy but promising in efficiency, offering a new learning paradigm. The paper also seems motivated by the information bottleneck of layer-by-layer pretraining and targets at its failure in transformers. However, it then applies layer-by-layer training in a fine-tuning setting, as shown in the experiments and in the appendix. With today’s well-pretrained LLMs, layer-by-layer fine-tuning is much easier. The paper uses this fine-tuning case to argue that the difficulties of layer-by-layer training can be largely alleviated by the proposed segmented propagation.

* The experiments are not solid. Since the experiments fine-tune an LLM, could the authors compare with other efficient methods such as LoRA, which alleviates the memory burden of end-to-end training? In addition, the results in the main paper are hard to interpret: what is the final performance under given memory/latency constraints? Can the authors report final accuracy together with the corresponding efficiency statistics, given that the proposed method has an efficiency-related hyperparameter?

* The paper discusses the information bottleneck extensively (e.g., the HSIC metric) and claims to build on its principles (e.g., line 082) . However, how do the proposed equations (Eqs. 3.5–3.8) relate to the information bottleneck?

* The writing needs substantial improvement. The structure is unclear. For example, there is at most one page of experiments (half a page of text and half a page of figures), while two pages are devoted to discussion and future work. Also, what is the main difference between Sections 3.5 and 2.4? Both appear to discuss differences between the proposed method and current early-exit and compression methods. Meanwhile, it's better to state the fine-tuning setting clearly in the abstract and introduction to avoid readers mistaking the paper as addressing pretraining challenges of layer-by-layer training.

### Questions
Please see above.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors introduce a novel method to train LLM layer-wise efficiently. Instead of training layer-per-layer sequentially, this method proposes to first train a reasonable subnetwork, and only then start the sequential layer-per-layer training. At each stage, the same LLM head is reused and participate to the training, instead of ancillary LLM heads, discarded at each step, as in standard LW training.

### Strengths
* The method appears to be novel, efficient, and to perform better or as well as E2E training
* The exposition is very clear, which I appreciate

### Weaknesses
* The Hilbert-Schmidt Independence Criterion (HSIC) is used to motivate the research as it allows to show that standard LW training suffers from information degradation compared to E2E. However, the paper doesn't compare the new method using this criterion. Either downplaying HSIC in the narrative, or having experiments around this would help.

* Figure 2 would need more detailed explanations (along with bigger fonts and graphs).

* A paragraph describing precisely what is meant by "standard LW" training would be useful.

* A larger set of experiments with similar results would be useful since the proposed method has no theoretical backing. (What about ResNet and image classification? )

* Some theory on why this method should work beyond the datasets used would be helpful (Is the method only working for LLMs? Why should it work better in general?)

### Questions
Can you address some of the weaknesses outlined in the "Weaknesses" section?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Segmented Propagation (SegProp), a two-stage, layer-by-layer training scheme for LLMs. Stage 1 jointly trains a base prefix with the final layers, and Stage 2 iteratively trains each intermediate layer with final layers. The authors claim SegProp matches or even exceeds end-to-end training on MMLU and HumanEval+, with efficiency benefits.

### Strengths
1. Clear, simple idea. Reuse the actual final head/layers as a universal supervisor for each segment, avoiding auxiliary heads and keeping the training objective aligned with the final task.

### Weaknesses
1. The major issue with this paper is its very insufficient experiments (only Figure 2 is for experiments).  The author claims that “matches/exceeds E2E training”. However, experiments appear limited in model settings/baselines and do not compare against strong modern alternatives for local or modular training (deep supervision, synthetic gradients, etc.). Claims of exceeding E2E need broader, compute-normalized comparisons.
2. Key training details are missing. Early-exit during training is mentioned conceptually but concrete criteria aren’t specified, which affects reproducibility.
3. Efficiency claims lack hard numbers. The paper argues reduced recomputation and energy use via SnapCheck, but provides no quantitative profiling of peak memory, throughput, FLOPs, or energy under fixed hardware/parallelism. 
4. Results presentation needs tightening. It is more like a undergraduate project report than a research paper. Both writing and plots need polishment.

### Questions
N/A

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies layer-wise training of multi-layer transformers similar to as is done for deep neural networks. They propose such a training paradigm SegProp, where a few layers are first trained, after which those weights are frozen and the remaining layers are trained. The paper also studies early stopping, and the impact it has on training.

### Strengths
- This is an important topic. The paper that gives a training paradigm for transformers that is similar to something that has often been used for deep neural networks.
- The proposed algorithm is simple and it does work well empirically.
- The analysis of layer-wise training, early stopping and other related ideas are explained in details.
- The ideas are presented very lucidly and the paper is easy to read.

### Weaknesses
- It would be good to highlight how this work is different from pure LW training, E2E training etc.
- Please explain what kind of convergence criterion/exit criterion is usually used.
- Line 195: Please define f_{embed} and f_{LM} separately. Also in the definition of f_i(x) in line 197, it is better to keep an i-subscript f_{MLP_i}, f_{SA_i} to show that it corresponds to the i-th layer.
- Also in definitions, the functions’ domain and range needs to be defined, like F: R^{d_1} -> R^{d_2}. Same for LL.

### Questions
- Line 103: What is global supervision?
- Line 225: Is the model depth p a constant or a hyperparameter? How is it chosen?

### Soundness
3

### Presentation
3

### Contribution
3
