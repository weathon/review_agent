# Echo State Transformer: Attention over Finite Memories

- Decision: Reject
- Scores: 2, 8, 2, 2

## Abstract
While Large Language Models and their underlying Transformer architecture are remarkably efficient, they do not reflect how our brain processes and learns a diversity of cognitive tasks such as language and working memory. Furthermore, sequential data processing with Transformers encounters a fundamental barrier: quadratic complexity growth with sequence length. Motivated by these limitations, our ambition is to create more efficient models that are less reliant on intensive computations. We introduce Echo State Transformers (EST), a hybrid architecture that elegantly resolves this challenge while demonstrating exceptional performance in classification and detection tasks. EST integrates the Transformer attention mechanisms with principles from Reservoir Computing to create a fixed-size window distributed memory system. Drawing inspiration from Echo State Networks, the most prominent instance of the Reservoir Computing paradigm, our approach leverages reservoirs (random recurrent networks) as a lightweight and efficient memory. Our architecture integrates a new module called "Working Memory" based on several reservoirs working in parallel. These reservoirs work as independent working memory units with distinct internal dynamics. A novelty here is that the classical reservoir hyperparameters, controlling the dynamics, are now trained. Thus, the EST dynamically adapts the reservoir memory/non-linearity trade-off. Thanks to these working memory units, EST achieves constant computational complexity at each processing step, effectively breaking the quadratic scaling problem of standard Transformers. We evaluate ESTs on a recent challenging timeseries benchmark: the Time Series Library, which comprises 69 tasks across five categories. Results show that ESTs ranks first overall in two of five categories, outperforming strong state-of-the-art baselines on classification and anomaly detection tasks, while remaining competitive on short-term forecasting. These results position ESTs as a compelling alternative for time-series classification and anomaly detection, and a practical complement to transformer-style models in applications that prioritize robust representations and sensitive event detection.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a “Echo State Transformer” that replaces (part of) the Transformer’s sequence memory with a bank of ESN-style reservoirs. At each step, the current input produces the query Q, while keys/values K,V are derived from the reservoir states; attention is applied over a fixed set of reservoir units rather than over the entire token history. The claimed benefits are constant-time attention over a bounded memory, and the ability to tune reservoir hyper-parameters (e.g., spectral radius, leak rate) for better long-range modeling.

### Strengths
There is a potentially interesting engineering idea in constraining attention to a fixed-size, where recurrent memory can reduce compute and serve streaming use-cases. Related recurrence-plus-attention hybrids, e.g., segment recurrence in Transformer-XL (Dai et al., 2019) and local RNN features in R-Transformer (Wang et al., 2019) did well in long-context settings, so this research direction is reasonable.

In addition, there is a long-term interest in reservoir computing for temporal data, and connecting RC with attention might make RC accessible to people used to Transformers. Surveys and overview works on RC show a mature ecosystem the paper could connect into.

### Weaknesses
There are several important weaknesses here.

## Novelty

The central idea of using a recurrent state to provide keys/values and query with the current decoder/input state dates back to early neural work on MT! Bahdanau et al. (additive attention) [1] takes the decoder RNN state as Q and encoder RNN states as K,V; Luong et al. [2] systematically study global/local attention variants built on RNN encoders/decoders. Thus, "RNN --> K,V, current state --> Q" is not new. 
Many later hybrids interleave recurrence and attention without ESNs: R-Transformer (LocalRNN features + multi-head attention) [3], Transformer-XL (segment-level recurrence with cached K,V) [4], Universal Transformers (recurrence over depth) [5], RWKV (RNN formulation of QKV dynamics) [6]. All these directly target the "bounded memory / long-context" motivation that this paper builds upon. The manuscript needs to position itself against all these precedent work and clarify what is uniquely enabled by ESNs beyond just swapping in a different RNN.

## Clarity and technical completeness

Section 4 lacks formal definitions/equations for the proposed model, such as precise mapping from inputs to Q, reservoir state update equations (including leak, spectral radius scaling, input scaling/sparsity, the linear projections to K,V and whether they share parameters across heads, how gradients flow if spectral radius/leak are learned, how multi-reservoir aggregation works. Relying just on text to explain this stuff and deferring essentials to the appendix leaves the core idea unclear. A model diagram that fully connects inputs, reservoirs, QKV, attention, and outputs is missing, whereas the space is taken by several existing figures (e.g., early motivation plots), which are not so important imo

## Improper usage of ESNs

Classical ESNs train only the readout while keeping the reservoir fixed; this is core to RC approach that produce a rich pool of dynamical feats w/out doing any training. If the spectral radius and leak rate are learned end-to-end, you already need BPTT through the recurrent core, destroying the stated benefit of RC, and you basically end up with the same complexity of a trainable RNN, with the exception that you do an unusual parameterization. 
Crucially, if you depart from the classic fixed-reservoir/readout-only training paradigm, **you must** justify why this is preferable to learning a standard GRU/LSTM (with well-understood gating/gradients)


## Learned leakage

A learnable leak coefficient that interpolates between the previous state and a candidate update is very similar to the update gate in GRUs and the cell/forget gates in LSTMs. GRUs are literally defined as a learned convex combination between $h_{t-1}$ and $\tilde{h}_t$.
Without demonstrating a concrete advantage over gated units, the choice of using ESN looks arbitrary (see also my comments above). Baselines that swap the reservoir with GRU/LSTM (same hidden size, same QKV heads) are necessary.

## Diversity in the dynamics of each Reservoir

If units 0/1/2... share a similar (random) initialization and the hyperparams are trained with the same signal, they will likely collapse to produce duplicate dynamics, making parallel reservoirs redundant. Multi-reservoir RC literature shows that heterogeneity in timescales/topologies is key to gains. 
Recent works explicitly induce diverse timescales or heterogeneous units and measure the benefit. 
The paper does not propose a diversity mechanism (e.g., distinct spectral radii/input scalings/sparsity patterns, orthogonalization penalties) nor analyze redundancy.

## Missing baselines

Beyond an ablation study where the ESN is replaced with a GRU/LSTM, the comparisons should include R-Transformer, Transformer-XL, and (if constant-time decoding is a claim) RWKV, which are all purpose-built recurrence/attention hybrid models forlong-context or efficient memory. 
Without these, it is impossible to attribute gains to the reservoir choice, rather than to generic "bounded memory + attention."

In addition, there is a rich RC literature and open libraries[7] for time-series classification/forecasting/imputation with RC. The paper neither cites nor considers them to showcase where reservoirs shine. Incorporating established RC tasks and references would strengthen both positioning and experimental design.


## Refs

[1] Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate, 2014

[2] Luong et al., Effective Approaches to Attention-based Neural Machine Translation, 2015

[3] Wang et al., R-Transformer: Recurrent Neural Network Enhanced Transformer, 2019

[4] Dai et al., Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, 2019

[5] Dehghani et al., Universal Transformers, 2019

[6] Peng et al., RWKV: Reinventing RNNs for the Transformer Era, 2023

[7] Bianchi et al., Reservoir computing approaches for representation and classification of multivariate time series, 2020

### Questions
- Add full equations and a full diagram in the main body.
- Explicitly relate to Bahdanau/Luong (RNN K,V + decoder Q), R-Transformer, Transformer-XL, Universal Transformers, RWKV, and clarify what ESNs uniquely contribute. 
- Make an ablation where you repalace GRU/LSTM with the ESN.
- Compare wrt R-Transformer Transformer-XL RWKV and match params/compute. 
- Add the following ablations: (a) fixed vs learned spectral radius/leak; (b) single vs multiple reservoirs; 
- Show experimentally how heterogeneity the dynamics of the different reservoirs are (i.e., quantify similarity across reservoirs)
- Include standard RC baselines from an RC library.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces the Echo State Transformer (EST), a hybrid sequential model that combines transformer-style attention with a finite set of reservoir-based working memory units (inspired by Echo State Networks). 

Instead of attending to all past tokens, EST attends to a fixed number of memory units whose internal dynamics are governed by reservoir parameters (e.g., spectral radius and leak rate). Unlike classical ESNs, these dynamics parameters are learned (via BPTT), and the leak rates are adaptively modulated through a competitive softmax across memory units at each step. 

Formally, the model retains the usual attention operation but computes Q from the current input embedding and K, V from the previous memory states. Then, it updates each memory unit through a (learned) reservoir-style state transition with an adaptive leak. This yields per-step computation that scales with the number of memory units rather than with sequence length.

By keeping the number of memory units constant, EST aims to achieve linear complexity in sequence length while retaining sensitivity to salient temporal patterns. 

On the Time Series Library (TSL) benchmark (69 tasks across 5 categories), EST reports 1st place aggregate performance in classification and anomaly detection, competitive results in short-term forecasting, and weaker results in long-term forecasting and imputation. 

The paper includes a FLOPs scaling analysis (showing linear scaling similar to Mamba) and a memory-usage comparison that highlights EST’s high training memory footprint due to BPTT.

### Strengths
- Interesting fusion of reservoir computing with attention: using several parallel reservoir units as finite working memory over which attention is applied feels novel compared to token- or patch-based memories. 

- Strong and broad evaluation on time series tasks (69 tasks; 5 categories) with clear metrics, following the benchmark’s training protocol. Strong results in classification and anomaly detection. 

- Baselines considered are recent and challenging (e.g., Transformer, PatchTST, Reformer, iTransformer, TimesNet, Mamba).

- Appendix is well curated and code implementation is provided, aiding reproducibility and interpretability of the architecture choices.

### Weaknesses
- EST selects the best of 10 configurations per task; it is not fully clear whether competing methods in TSL were re-tuned to a comparable extent or simply reused benchmark defaults. 

- The paper claims novelty in learning reservoir dynamics (spectral radius / leak) and in the adaptive leak softmax. However, as far as I could see, there is no ablation removing learned spectral radius, adaptive leak, and self-attention over memory units. Such ablations would quantify each ingredient’s contribution to classification/AD gains. 

- The Previous State Attention + reservoir update is conceptually clear but not given as a compact set of equations.

### Questions
- Did all baselines receive comparable hyperparameter search budgets to EST’s 10-config sweep per task? If not, can you report a matched HPO budget for at least the top baselines per category, or include a sensitivity analysis showing that EST remains top-ranked under a fixed budget? 

- For TSL anomaly detection, how exactly are scores produced and thresholds chosen (per-dataset or global)? Is the output a pointwise score from the output layer, or derived from an intermediate representation? 

- See Weaknesses

### Soundness
4

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
4

### Summary
This paper introduces a hybrid architecture known as the Echo State Transformer (EST), designed to address the quadratic complexity problem of traditional transformers in processing sequential data. By integrating the attention mechanisms of transformers with principles from reservoir computing, EST establishes a fixed-size working memory system. Experiments are conducted across multiple time series benchmarks, including forecasting, imputation, and anomaly detection.

### Strengths
1. The experiments effectively consider a variety of tasks.
2. Figure 1 clearly illustrates the differences between the proposed EST model and the traditional Transformer architecture.

### Weaknesses
The primary idea of this work is to introduce a reservoir-based recurrent neural network memory to avoid the self-attention computations associated with the original sequence length. However, **this idea is not new**, as there are already numerous studies addressing the integration of RNNs or improving transformers through downsampling, clustering, and other strategies. The authors do not provide a thorough discussion or comparative analysis with these related works, which diminishes the paper's contribution to innovation.

In terms of presentation, the core methodology is described primarily through Figures 4 and 5, lacking formal mathematical language that would allow for precise and rigorous description.

Regarding experimental results, the improvements in classification and anomaly detection tasks are minimal, while **performance in other tasks falls short** compared to many existing works.

From the appendix, it is evident that the hyperparameter settings for different datasets vary significantly. The discussion around optimal configurations for various tasks indicates that performance is highly sensitive to hyperparameter choices; however, the paper lacks a detailed exploration of how to select or optimize these parameters, which could restrict the model's generalization ability to new tasks.

### Questions
Please see in Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a novel transformer architecture that combines both strengths from the transformers and ideas in the literature of reservoir computing. More specifically, a working memory module consisting of multiple units based on the reservoir computing paradigm is introduced in the transformers, with the aim to enable targeted attention at low computational complexity. Experimental results are provided to demonstrate the practical effectiveness of the proposed architecture.

### Strengths
- The proposed approach makes use of ideas from the reservoir computing literature in the design of transformer blocks, which sheds light on how different fields can be leveraged together in a meaningful way.
- The proposed architecture is simple and intuitive, and is relatively easy to follow. 
- The proposed approach is well motivated, and the paper is generally well presented.

### Weaknesses
**Technical novelty.** While incorporating ideas from reservoir computing into transformer design is interesting, the way the combination is done is relatively straightforward. In addition, no theoretical results or analysis have been provided to justify the proposed architecture in a more rigorous manner. Both render the technical novelty of the paper somewhat limited. 

**Empirical performance.** The experiments remain somewhat limited and unconvincing. 
- My understanding is that part of the motivation is to derive an architecture that is enable of robust long-range reasoning (where computational complexity remains a bottleneck). From this perspective, perhaps the authors can consider additional experiments on benchmarks such as the long-range arena (if that is deemed suitable). 
- It is not clearcut the proposed architecture obtains better empirical results compared to existing approaches. I appreciate there are strong baselines out there, however the performance gain on two of the five benchmarks are relatively marginal (especially on Anomaly Detection), and results on the other three, while explained with good reasoning, remain not sufficiently convincing. Could it be that more appropriate benchmarks should be tested on to better highlight the strength of the proposed architecture? 
- Given the motivation from the angle of the computational complexity, it would be helpful if the authors can provide a performance-complexity plot that illustrates how the propose architecture manages the trade-off between the two. 
- Can the authors discuss in mode detail the impact of certain design parameters, such as the number of memory units? For example, only the best results based on a chosen number of units have been presented, but no detailed discussion about its impact.

### Questions
See weaknesses above for the specific points I would like the authors to address or discuss.

### Soundness
2

### Presentation
3

### Contribution
2
