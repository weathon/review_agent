# KDP: Simplifying Representation Dynamics in Kernel Space

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
This paper proposes Kernelized Dynamics Pruning (KDP), a novel layer pruning method from the perspective of simplifying representation dynamics within large language models (LLMs). Motivated by the high similarity between consecutive layer representations, we view the LLM's forward pass as a discrete-time dynamical system. We speculate that this phenomenon indicates the model's internal dynamics have entered a ``slow manifold'', which exhibits computational redundancy. Based on this insight, we project the representations into a kernel space where the complex, non-linear transformation between them is simplified to an approximately linear one. Then, a simple network learns the inverse kernel transformation, thereby enabling the pruning of the entire layer block. Both theoretical analysis and extensive experiments validate the effectiveness of KDP, demonstrating its superiority over existing pruning baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Kernelized Dynamics Pruning (KDP), a new method for pruning layers in large language models by viewing their forward computation as a discrete-time dynamical system. It observes that consecutive layers often produce highly similar representations, indicating redundancy. To exploit this, KDP projects representations into a kernel space where nonlinear transformations become approximately linear, allowing simpler modeling. A linear operator and an inverse mapping network then replace entire Transformer blocks. The authors present a theoretical error bound showing that multi-layer dynamics can be linearly approximated in this kernel space and prove that it provides superior fitting capacity compared to the original representation space. Extensive experiments on fifteen benchmarks demonstrate that KDP effectively prunes models while maintaining performance, without requiring fine-tuning. Overall, KDP provides a geometric and theoretically principled framework for simplifying internal model dynamics and reducing redundancy in large language models.

### Strengths
- Originality: while the fact that LLMs are characterised by highly redundant layers is well known, with several papers attempting to interpret this fact in different ways, this work’s idea on exploiting kernel space to identify similarity and learn the linear map to prune layers is, to the best of my knowledge, novel and effective.
- Quality: the authors take a significant effort in stating clearly their theory and in building a solid set of experiments
- Clarity: The problem is clearly set and the results and experiments are well setup and presented.
- Significance: the effort of reducing LLM is certainly timely and important. Before this work, papers arguing that layers of pre-trained models were redundant mostly offered solutions which dealt with pruning layers and then fine-tune to cure the cut. The proposal of this paper of learning the map to bypass the pruned layers seems to be a relevant step forward.

### Weaknesses
I have a couple of general objections:
- About the broad framing of the paper: the authors talk frequently (abstract, introduction) about the fact that they view pre-trained models as “discrete-time dynamical systems” and talk about “slow manifold” and “reduced velocity” during the redundant phase. These concepts are not developed throughout the paper, and it would seem that the paper’s results and claims would be unaffected by removing this interpretation. While I’m totally in favour of trying to give a physical interpretation of LLMs representations, I think the paper would benefit if the authors could expand on this interpretation more (maybe in appendix?), or remove it altogether.
- I think the paper might benefit from a short interpretability discussion about which layers exactly are being cut and why, also wrt to previous literature. Previous work (ex. Men et al. as referenced in the paper) seem to show that layers after the first half of the model depth are good candidate for pruning, and that cutting earlier, or late layers, causes a drop in performance. A study of this on similar models is found e.g. on Gardinazzi et al. ICML 2025. In other words: there are two things determining viable pruning: size of the cut and position of the cut. The latter is not discussed from what I can see, and this work might have a good angle on this, starting from theorem 1. Moreover, in appendix it seems that several non-consecutive blocks are pruned, which is in contrast with previous work: is this discussed anywhere in the paper? 
- The theoretical part, especially around theorem 1, suffers a bit from lack of background on the techniques used: the authors talk about ERM (empirical risk minimization?), population risk, etc without connecting this to the problem at hand. To someone not familiar with these concepts, this is a bit hard to follow.

### Questions
Here are a few more detailed questions that I’d like to be addressed
- From Figure 1, it seems to me that CKA saturates to 1 very frequently (particularly for llama3-8B). It is argued that this is due to similarity in kernel space being more effective. However, if I understood correctly theorem 1 argues that stacking too many layers accumulate errors, even if they are all similar. Indeed, from the results it seems that the algorithm prefers to cut non consecutive layer blocks, than one-two larger consecutive blocks, which would be more similar to previous work. Which part exactly of the algorithm deals with this decision? if it is entirely based on CKA value, and the value is very close to one across the whole depth, is this stable?
- I haven’t found a definition of ERM, first mentioned in theorem 1.
- Did the authors verify whether the algorithm would cut early and late layers and verify the drop in performance? they say they exclude them based on previous literature, but would they be excluded by the algorithm anyway or is this to be input as prior knowledge?
- Can the author explain exactly what “we prune about 25% of the parameters mean in 4.1 setup-models?
- in the ablation study, the authors use CMMLU, for which performance is low for all models even for the dense case. Wouldn’t it be safer to consider a benchmark for which the dense case performs relatively well? Testing around random performance might have some side effect.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a novel layer pruning method called Kernelized Dynamics Pruning (KDP). The method replaces several consecutive layers within a transformer model with a kernel mapping, linear transformation, and inverse mapping. This process is able to prune around 25% of the weights in the network while retaining higher accuracy than existing methods across several tasks. Theoretical results and comprehensive ablations explain the value of all components of the method and relate findings to previous observations in layer representational similarity.

### Strengths
There are several strengths that make this an interesting paper. The findings are quite well explained and strong. The findings hold consistently across a wide range of experimental settings. The theory relates findings to and builds upon previous results. Ablations demonstrate the value of each component. The method is explained clearly via figures, an algorithm, and descriptions.

### Weaknesses
N/A. This is a strong and well-organized paper.

### Questions
What is the cost of identifying and training the transformations? How does this compare to other layer pruning methods?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
- The paper introduces Kernelized Dynamics Pruning (KDP), a layer-pruning method that interprets the forward pass of LLMs as a discrete-time dynamical system and leverages representational similarity to simplify internal dynamics.
- By projecting layer representations into a suitable kernel space, the method simplifies the non-linear transformations between consecutive layers and learns an inverse kernel transformation to reconstruct the pruned representations.
- The authors provide theoretical guarantees on the linearizability of representations in kernel space and empirical validation across 15 benchmarks, showing strong performance while maintaining efficiency and minimal retraining requirements.

### Strengths
- Interesting and well-motivated idea of replacing transformer blocks with lightweight modules that approximate their simple dynamics
- The paper establishes a theoretical error bound for kernel-space linearization, bridging empirical results with formal justification.
- KDP offers layer-wise compression while preserving performance better than other pruning methods.
- Extensive empirical validation as the method is evaluated on many benchmarks, suggesting robustness and generality of the proposed method.

### Weaknesses
- The paper primarily compares inference cost and FLOPs only against the original dense model (Table 10) rather than existing pruning baselines, limiting understanding of relative efficiency gains. It would provide a clearer and more fair comparison if the speedup was also compared across all the baselines
- The parameter count of the replacement modules are not clearly reported; it would be helpful if they were provided for the final pruned models that are evaluated
- While representation similarity is highlighted, further quantitative analysis of the “slow manifold” hypothesis or ablation on kernel choices would strengthen the conceptual link.

### Questions
- Is there any possible explanation for the significant rise in PPL for the Llama3.1-8B results? (in the case where there is no fine-tuning)
- Within this framework, is it possible or beneficial to fine-tune the modified model beyond the output projection layer? Specifically, could the pruned model, with its replacement modules, serve as a foundation for further task-specific fine-tuning, or are these modules primarily effective only when substituting already well-learned transformer representations?

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
The paper introduces a novel layer pruning method for LLMs called KDP. The key idea is to view a forward pass as a discrete-time dynamical system, where high similarity between consecutive layers implies entry into a slow manifold, indicating redundant computation. KDP projects intermediate representations into a Hilbert space, where complex nonlinear dynamics between layers become approximately linear. It then learns linear operators to approximate transitions and an inverse mapping network to reconstruct the representations back in the original space, enabling whole-layer pruning without retraining. Experiments across benchmarks indicate that KDP is a performant structured pruning method on LLMs.

### Strengths
The method is novel, significantly different from previous pruning methods.

The evaluation is conducted on various models of different scales; benchmarks are comprehensive.

The paper is overall easy to follow.

### Weaknesses
My major concern lies in the additional overhead introduced by KDP. Kernel mapping and linear operators might introduce non-trivial costs.

### Questions
What is the additional overhead of KDP compared to the LLM? Providing time or FLOPs analysis might be helpful.

### Soundness
4

### Presentation
4

### Contribution
4
