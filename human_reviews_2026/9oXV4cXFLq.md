# SAFT: Structure-Aware Fine-Tuning of LLMs for AMR-to-Text Generation

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Large Language Models (LLMs) are increasingly applied to tasks involving structured inputs such as Abstract Meaning Representations (AMRs). However, common approaches either linearize graphs, discarding crucial structural cues, or rely on specialized architectures that are incompatible with standard pretrained LLMs. We present SAFT, a structure-aware fine-tuning method that augments LLMs with graph-sensitive positional encodings derived from the magnetic Laplacian of AMRs. These encodings are projected into the LLM embedding space, introducing relational inductive bias without modifying the model architecture. Designed to be applicable across tasks involving graph-structured inputs, we demonstrate its effectiveness on AMR-to-text generation, where SAFT establishes a new state of the art on AMR 3.0 with a +3.5 BLEU improvement over prior baselines. Performance gains grow with graph complexity, highlighting the value of structure-aware representations in enhancing LLM performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new "Structure-to-text" method for LLMs. Such methods usually employ simple ways to serialize the structure to allow the LLM to encode it. A typical example is linearizing a Direct Acyclic Graph into a as string that can be tokenized. This approaches however loose important structure information. The proposal here is to 

1. enrich the conventional encoding with additional position embeddings reflecting the graph structructure. 
2. obtain these from a modified Laplacian of the graph. Concretely they use the "Magnetic Laplacian" which leverages a complex valued representation to capture graph directionality

The usefulness of the method is show in Abstract Meaning Representation (AMR)-to-text. This is a well known meaning representation expresses as directed acyclic graphs with both labeled edges and nodes.

### Strengths
1. The idea is clear and addresses an interesting problem. Intuitively there has to be a better way of encoding graph information than simple linearization

2. The method proposed, based on the "Magnetic Laplacian" encoding of graphs is novel in its application to Structure-to-text (to the best of my knowledge) although it has been used for Graph Neural Networks and other purposes in the past. 

3. The method also does not require additional models and seem relatively simple in terms of network modifications

### Weaknesses
My main criticism is about relevance. 

1. The area of structure-to-text seems to be fading a bit since it is clear that large LLMs can perform this task relatively well. It is best to avoid closed-source LLMs but a baseline using some well known SoTa LLM is a useful reference. LLMs larger than 3B would also be in order here. The suspicion is that, as models grow larger, the advantage of encoding will also reduce.
 
2. and more clearly Abstract Meaning Representation, and other liguistic representations seem to be not relevant any more for representing information. At least other usual Structure-to-text datasets like WebNLG and similar could enrich the paper further. I don't know if there is any more recent dataset on this area, if not, this further substantiates my point (1)

Aside from this the experimental setup could be improved. Only the LLaMa models show clear gains, and even in this case it is not clear whats if there is statistical relevance. Since the metrics used are bounded between 0 and 1 standard deviation is not ideal, but given that many numbers are close to 0.5 it may help at least get an idea. Another option is to run a boostrap paired test which would work for such metrics.

### Questions
What is the standard deviation in the results shown? knowing this or the statistical significance would be great

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper focuses on the problem of generating natural language sentences or documents from abstract meaning representation (AMR) graphs.  The baseline approach is to finetune a language model (1B-3B parameter range) specifically for this task; for Qwen/Llama/Gemma models this already outperforms published systems.   An AMR graph is first converted into a semantics-preserving graph (SPG).   The innovation in this paper is to introduce a new kind of positional embedding specific for AMR, derived from the (magnetic) Laplacian of the AMR graph.  The positional embedding is concatenated with intra-node token positional embeddings, before applying to standard LM embedding. By combining this positional embedding scheme with standard fine-tuning method (using LoRA), the authors arrive at what they call “Structure-Aware Fine-Tuning.”   The motivation is that doing so will help the model “to better capture graph topology and long-range dependencies.”  Experimental results show consistent improvements across base models and evaluation metrics on the AMR-to-text task, compared to the more conventional finetuning approach.  Additional experiments show that improvements hold up when evaluating on a harder document-level task and that improvements are greater on “deeper” (and supposedly more challenging) AMR graph inputs.

### Strengths
The paper is generally clearly presented and well motivated.  The magnetic Laplacian idea is elegant, and this reviewer learned something new from the paper.  The method is easy-to-follow and uses graph theory tools like magnetic Laplacian. The math formulae in the paper are helpful.  (Though the reviewer did need to spend a few minutes chatting with a language model to get some intuition about the magnetic Laplacian; the paper could do more to help readers here.)

The experimental results are good, and even better when one looks at the appendix.  I especially appreciate inclusion of the document-length experiments.  I also appreciate the complexity-stratified results. They give strong evidence that the approach is good at handling complex AMRs. 

Experiments compare with previous strong approaches and standard fine-tuning methods on standard AMR 3.0 benchmark.

The paper included comprehensive details in the appendix, including implementation, dataset and related works. Readers who are not familiar with this field should be able to understand the paper.

### Weaknesses
There is no discussion of the computational cost of the approach in the main paper, either in absolute terms (complexity of calculating the position embeddings – though this is in the appendix, increase in input token count, etc.) or relative to the baseline (at finetuning time, at inference time – this comparison is not even in the appendix).  Readers will want to know what they have to pay to get the quality improvements, both up front to finetune and at inference time.  I think some of this discussion should be in the main paper.

There are key details missing about the finetuning.  How many epochs (this is only in the appendix)?  Is the loss the negative log probability of the outputs conditioned on the input graphs?  (This seems like the only choice, but it’s not stated.)  Or something else?  Which parameters are updated?  One supposes  LoRA parameters and the new ones that transform the graph representation for input as tokens (f_theta), but it’s not stated clearly.

I wasn’t entirely clear about what motivated the desire to avoid architectural changes.  I suspect it has to do with preserving the LM’s existing abilities.  But then there are no experiments to measure how much the finetuning for AMR-to-text degrades the models’ other capabilities.

It seems like a big part of what is learned during finetuning is getting the new positional embeddings to “mesh” with the existing ones.  But the positional embeddings in the base models are not discussed at all.  Are they all sinusoidal?  Should some elements of the graph embeddings be designed around the positional embeddings of the base model?  Given that the technical novelty of the paper hinges on these new embeddings, there’s a missed opportunity to explore what happens during learning.

The claim in the abstract/introduction about general applicability for structured inputs is, I think, a bit overstated.  At the very least, some discussion of how this paper’s specific technical idea could apply to other settings beyond the labeled DAGs of AMR seems necessary to justify that argument.  Even better would be experiments.

The baselines seem well chosen, but this reviewer wondered why the comparison to proprietary, un-finetune-able models was buried in the appendix.  The fact that stronger/larger LMs perform poorly on this task seems to strengthen the paper a lot.  I would suggest adding few-shot experiments there, as well, and maybe testing a few more “frontier” models.

All experiments are conducted with LoRA, which limits the extent to which the model can be adapted or fine-tuned. Even though LoRA has been shown powerful enough on many natural language tasks, it is not discussed if it’s sufficient to do LoRA for AMR-to-text tasks. AMR data is relatively rare in the pre-training corpus of LLMs, so maybe full-finetuning would show larger improvements on this task. Will SAFT improve the performance under the full-finetuning setting? It’s possible that SAFT improvement will be marginal and only works for parameter-efficient fine-tuning settings. If the computation resources are limited, I recommend doing experiments with increasing LoRA rank to approximate the trend. However, full-finetune experiments would be the best.


In Table 1, all previous baselines are worse than most of the FT settings. This is perhaps due to the weak backbone of the baselines — BiBL and SPRING are based on BART, StructAdapt is based on T5. So, the results of FT and SAFT in Table 1 are not fully comparable with the previous baselines. All of these LLMs like LLaMA, Qwen, and Gemma are pretrained on a lot of “extra” data.  This raises the question of whether the positional encoding scheme is really the reason the approach is “state of the art”.
 
Some qualitative error analysis would be helpful.

Minor points, easy to fix:

Bold-italic U is introduced just before equation 2 with no definition or connection to earlier objects.

Equation 10 and bold-italic X is unnecessary, as it is never reused.

Table 1:  what are the parenthesized percentages in the table?

I think the DocAMR setting is not “zero-shot”. DocAMR is still on the same task and it’s built from AMR 3.0. 

In table 10, please include the size of DocAMR.

### Questions
In Table 1, I see a trend that the improvements on larger models are smaller. Does it suggest that SAFT only works for small models? Since the margins of most improvement are small ( < 2% absolute), it’s better to report average performance over different random seeds for both methods.

Figure 2(a) shows almost zero gap for Qwen when z is small. Any explanation for this phenomenon?

Can we use different prompts to fine-tune LLMs to understand AMR graphs so that users can talk with LLMs directly about the AMR graph? This will enable a wider use of the positional embedding.

In section 3.3, a pretrained LLM decoder is decomposed into DECODE and EMBED. Does EMBED include the LLM’s original positional embedding? If not, then the position of the AMR in the prompt is not encoded. According to the prompt in B.3.1, the position of “L_A” matters. However, if EMBED includes the LLM’s original positional embedding, that imposes a sequential order onto “T_L” and thus different node numberings give different final embeddings. Please clarify this.

The test set of AMR 3.0 only has 1898 samples. It is sufficient for general tests, but could be too few for stratified analysis. How many examples do we have for z>=8? It would be better to plot the distribution of z on the test set of AMR 3.0. Similarly, it would be better to have the plot for DocAMR, which I believe would be even sparser when z ranges from 0 to 35.

What kind of LLMs are fine-tuned for the AMR-to-text task? Base models or post-trained models? Please include the specific tags of the models. According to the prompt in B.3.1, they might be the base models since there is an ending token encouraging completion. However, GPT-4o-mini is a chat model, so the prompt should be redesigned and optimized for experiments in C.4. Please enclose the prompt for GPT-4o-mini. It should contain necessary content for adapting the task to natural language and also include common techniques like CoT. GPT-4o might outperform SAFT but that’s acceptable since they are much larger and pre-trained on more data.

DocAMR is built from part of AMR 3.0. Does it include the training split of AMR 3.0? If so, there will be data leakage and the evaluation is no longer meaningful. According to the DocAMR paper, only 9 test graphs are not coming from the training split of AMR 3.0, which are too few.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a structure-aware fine-tuning approach named SAFT for the AMR-to-Text task, aiming to enhance large language models’ understanding of graph-structured data. The SAFT method first transforms the original AMR graph into a directed, edge-unlabeled semantic-preserving graph (SPG), which preserves the structural semantics of the original AMR while enabling the application of spectral methods based on the Laplacian for positional encoding. Subsequently, AMR-specific graph positional encodings are derived using the magnetic Laplacian of the SPG, capturing the topological structure and directionality of the AMR graph. Finally, these graph positional encodings are incorporated into token embeddings, enabling relational inductive bias without modifying the model architecture.

Experimental results on the AMR 3.0 dataset demonstrate that the proposed method outperforms direct fine-tuning of large models using LoRA, particularly on graph structures with greater depth.

### Strengths
The paper proposes a novel graph structure encoding method that effectively encodes the structural information of AMR graphs. Furthermore, this method appears to be compatible with any Decoder-only LLM and holds potential for adaptation to other graph structures.

### Weaknesses
1.  **Lack of Important Baselines:** Prompt-based methods are also model-agnostic and, crucially, do not require additional training, potentially offering stronger generalization. The authors do not compare their method against such approaches in the main text. Although a comparison with a GPT-4o-mini zero-shot method is included in Appendix C.4, this is insufficient because: 

(a) The comparison is unfair: a comparison with *few-shot* methods is more appropriate, given that the proposed SAFT involves additional training on the AMR 3.0 dataset. 

(b) Few-shot prompt-based experiments should be added for *all* models used in Table 1 (LLaMA 3.2 1B, 3B, Qwen2.5 0.5B, 1.5B, 3B, and Gemma 2B) to justify the necessity of fine-tuning, especially since fine-tuning consumes significant resources and often severely degrades the model's general capabilities.

2.  **Unfair Experimental Comparisons:**

(a) The model sizes of the baselines in the upper part of Table 1 are not comparable to those in the lower part. For instance, the largest model used in StructAdapt (T5-Large) has about 0.77B (770M) parameters, while the LLMs for which SAFT outperforms StructAdapt all have 1.5B parameters or more (at least 1.94x larger than T5-Large).


 (b) When the parameter size of SAFT is comparable to StructAdapt, its performance is actually worse. Specifically, in Table 1, the BLEU and CHRF++ scores for Qwen2.5 0.5B, LLaMA 3.2 1B, and StructAdapt (0.77B) are 42.9 < 47.8 < 48.0 and 69.3 < 71.9 < 73.2, respectively.


3.  **Lack of Generalization Experiments:** Although the authors state that the proposed method is applicable to other tasks involving graph-structured inputs (Lines 78-79), they provide no experiments to substantiate this claim. Furthermore, given that experiments were conducted solely on the AMR 3.0 dataset, it is necessary to supplement the paper with experiments on other datasets involving graph-structured inputs to properly evaluate the method's generalization capability.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a method for graph structural information and directionality encoding into an embedding space, i.e., graph structure-aware positional embeddings. The authors use the graph adjacency matrix and the magnetic Laplacian operator to derive the embeddings (further combined with token embeddings). A transformation of the original graph is carried out, to encode edge labels; this involves representing edge labels as nodes. While the approach is presented for AMR-to-text generation, it would be applicable to other concrete graphs (e.g., scene graphs). As for the integration of input graphs with LLMs, input graphs are linearized and tokenized, graph positional embeddings  (i.e., a learnt MLP combines nodes positions and tokens and maps to the LLM embedding size) are combined with the standard LLM's sequence embeddings.

### Strengths
- Beyond AMRs, the proposed graph structure-aware positional embeddings could be useful contribution for the community to improve LLMs tasks that require long context processing.

### Weaknesses
- Experiments show improvements when fine-tuning with the proposed graph positional embeddings. However, from these experiments it is not clear how would these results generalize to larger scale LLMs (experiments where done in LLMs up to 3B size).

- The authors miss some important related work on graph encoding which deals in particular with the issue of encoding edge labels and applies semantic preserving transformation/reification of edges as nodes. These should added into the related work discussion [1, 2].

[1] https://arxiv.org/abs/1703.04826
[2] https://arxiv.org/pdf/1810.09995

- While the authors include a structure-aware finetuning previous work (StructAdapt), this seems quite an old model. It would be better to report results on a re-implantation of this (or any other, e.g., graph attention or approaches enumerated in F.4) in the same target LLMs to better compare performance. Also, it would be useful to have the intuition of the dynamics of the LLM learning representations via embeddings (L233) or via specific structure (e.g., message passing).

### Questions
- Line 042 "semantical" > semantic

- In explanation in Section 2.1, it would be useful to motivate why it is needed to symmetrize de adjacency matrix in first place (Line 114).

- It would be illustrative to add example outputs, generated texts, by the different models. This would help to understand/explain where the differences in (BLEU) numbers are coming from.  Even a human eval (or LLM-as-Judge assessing real textual differences) would strengthen the presented approach and results.

### Soundness
3

### Presentation
2

### Contribution
3
