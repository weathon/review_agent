# Rethinking Traffic Representation: Pre-training Model with Flowlets for Traffic Classification

- Decision: Reject
- Scores: 0, 6, 6

## Abstract
Network traffic classification with pre-training has achieved promising results, yet existing methods fail to represent cross-packet context, protocol-aware structure, and flow-level behaviors in traffic. To address these challenges, this paper rethinks traffic representation and proposes Flowlet-based pre-training for network analysis. First, we introduce Flowlet and Field Tokenization that segments traffic into semantically coherent units. Second, we design a Protocol Stack Alignment Embedding Layer that explicitly encodes multi-layer protocol semantics. Third, we develop two pre-training tasks motivated by Flowlet to enhance both intra-packet field understanding and inter-flow behavioral learning. Experimental results show that FlowletFormer significantly outperforms existing methods in classification accuracy, few-shot learning and traffic representation. Moreover, by integrating domain-specific network knowledge, FlowletFormer shows better comprehension of the principles of network transmission (e.g., stateful connections of TCP), providing a more robust and trustworthy framework for traffic analysis.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
One of the many papers tackling traffic analysis with a BERT-like (with very tiny engineering changes) approach. The paper follows the same line an present very minor incremental detail, with no actual methodological contribution.

Several of these paper and the dataset they use have been found to be flawed and debunked in several rank A* conferences -- starting from ACM CCS'2022 and with an up-to-date debuning in ACM SIGCOMM'2025

This ignorance can be either be done on purpose (which is bordeline  unhetical... how can they cite a paper from CCS'2018 and ignore *that* paper from CCS'2022? Same for citing SIGCOMM'2019 and ignoring *that* one in  SIGCOMM'2025?) or in full unawararennss of such work (which is equally worring and disqualify the authors work altogether), but the result does not change.


Additionally, the engineering is lightweight, the evaluation is biased and flawed: the paper would not be accepted at one of the above rank A* conference of network/security field.

Finally, there is no methodological contribution, there is no statistical relevance: as such, the paper does not find its place in ICLR'26 either



[CCS 2022] Jacobs, Arthur S., et al. "Ai/ml for network security: The emperor has no clothes." Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security. 2022.


[USENIX Sec 2022] Arp, Daniel, et al. "Dos and don'ts of machine learning in computer security." 31st USENIX Security Symposium (USENIX Security 22). 2022.


[SIGCOMM 2025]
Zhao, Yuqi, et al. "The Sweet Danger of Sugar: Debunking Representation Learning for Encrypted Traffic Classification." Proceedings of the ACM SIGCOMM 2025 Conference. 2025.

### Strengths
The paper has more flaws than strenghts -- but if I have to find one, then I would say the paper is clearly written (although I don't agree with the  styling m

### Weaknesses
all weaknesses are detailed next in the **Question** section 

- Incremental in nature
- No learning methodological contribution
- Flowlet Lightweight engineering contribution
- Tokenizer Lightweight engineering contribution
- Possibly biased, weak evaluation, which translate into fundamentally 
- Lack of statistical relevance
- lack of critical results analysis 
- lack of comparison baselines
- lack of relevant technical details

### Questions
This paper has several flaws that prevent it from publishing



## Incremental in nature

 add itself to the pile of paper in the comparison table

## No learning methodological contribution

 BERT-based with ``automatic'' definition of flowlet based on interarrival time (IAT) and some loose considerations about tokenizer (flawed in my opinion) or loss (too narrow, specific and not a contribution per se)

## Flowlet Lightweight engineering contribution

defining a flowlet by using an IAT filter (Alg in B2 p 16) is very lightweight. it is a bonus not having to define the number of packets in a burst a priori as generally done, but it does not qualify as a scientific/technical contribution (doublly so given the evaluation flaws)

##  Tokenizer Lightweight engineering contribution

 the paper states that related work overlooks the nature of the protocol segmentation. it pompously says to use morphenes to have undividied semnantic units -- but the true fact is that their tokenizer is not doing that at all. several fields in IP, TCP and any protocols are binary flags, which are independent yet packed together in the same byte for transfer, whose smallest units would be a bit and not a hex-unit (packing 4 such flags). So adding sub 16 sub encodings to a 65k vocabulary does not ring as a fundamental contribution unless you would be able to show instances (not rand% accuracy) where this does a semantic contribution -- yet even in Appx D. FL does joint Flowlet and tokenization

## Possibly biased, weak evaluation, which translate into fundamentally flawed study.

Starting from ACM CCS'2022 and with an up-to-date debuning in ACM SIGCOMM'2025, researchers have shown limits of these approaches attempting at learning from encrypted traffic payload. These studies are widely known, 
and additionally suggest (along with USENIX Sec'22)  best practices that this work do not follow. At the end, the gap with the proposal and the simple baselines from ACM CCS'2022 and ACM SIGCOMM'2025 suggest this work is trying to shoot a mosquito with a cannon
 


## Lack of statistical relevance

 all tables dumps 4 decimal values, no statistical relevance whatsoever -- no mean/ci, no repetittions, no statistical tests,  no paired tests, no critical distance plots.

## lack of critical results analysis 

some of the results show 1.0000 (Tab5)-- there is no critical analysis of the results whatsovere, likely some shortcut as those already shown in [CCS22] for the ISCX VPN-nonVPN and CIC-IDS-2017  datasets -- a 5 nodes DecisionTree achieves in excess of 99% accuracy for those tasks due to shortcut learning, which was debunked 5 years ago already

whereas shortcut learning are  mentioned in appendix C.2, it is not enough (randomized IP addresses and ports, and removed aboslute timestaps)

## lack of comparison baselines

benchmnark for simple (eg given SEQNO sequence, have a specialized ML for 
that taks, and show the gap) as done in CCS, SIGCOMM

## lack of relevant technical details  

fine tuning -- is it end-to-end (=destroying pre-training value)
or layers are frozen -- check SIGCOMM'25 why this is relevant

### Soundness
1

### Presentation
2

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
The paper introduces FlowletFormer, a Flowlet-based pre-training framework for network traffic analysis that captures cross-packet context, protocol semantics, and flow-level behavior. It employs Flowlet and Field Tokenization, a Protocol Stack Alignment Embedding Layer, and two Flowlet-inspired pre-training tasks to enhance semantic and behavioral understanding. Experiments demonstrate that FlowletFormer achieves superior accuracy, few-shot adaptability, and robustness by incorporating domain-specific network knowledge.

### Strengths
- The paper introduces Flowlet and  Field Tokenization as efficient traffic representation.
- Protocol Stack Alignment-Based Embedding Layer is proposed to explicitly encode the hierarchical semantics of network protocols, enabling the model to distinguish fields across protocol boundaries and better capture protocol-specific behaviors.
- Two pre-training tasks are proposed. Extensive experiments are performed on comprehensive downstream tasks under various settings, demonstrating the effectiveness of the proposed method.
- The paper is overall well written.

### Weaknesses
- Besides the superior performance, the technical novelty follows the general BERT pretraining pipeline with similar pretraining tasks.
- From the development of the community, besides the pretraining code, the model weights are suggested to be open-sourced.

### Questions
- In Table 2, Flowlet Former underperforms TrafficFormer on USTC-TFC only. Please explain the reason.

### Soundness
2

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
2

### Summary
This paper addresses the problem of network traffic analysis. To effectively capture cross-packet context, protocol-aware structures, and flow-level behaviors, the authors propose a Flowlet-based pre-training framework. The framework consists of three major components: Flowlet and Field Tokenization, a Protocol Stack Alignment Embedding Layer, and two pre-training tasks designed to enhance both intra-packet understanding and inter-flow learning.
Experimental results demonstrate the effectiveness and robustness of the proposed approach.

### Strengths
The overall paper is well-written.

This work investigates a BERT-based pre-training model, called FlowletFormer, for network traffic analysis. The proposed framework incorporates three key strategies: (1) Flowlet segmentation, (2) a Protocol Stack Alignment-based Embedding Layer, and (3) two pre-training tasks—Masked Field Modeling (MFM) and Flowlet Prediction Task (FPT).
The effectiveness of the method is evaluated on eight public datasets, demonstrating the superiority and robustness of the proposed approach.

### Weaknesses
The three components together constitute FlowletFormer. Among them, which element plays the most critical role, and how are these components interrelated?

In the FlowletFormer framework, how are the Flowlet Prediction Task (FPT) and the Masked Field Modeling (MFM) task distinguished in the flowchart?

Regarding the downstream tasks, the seven tasks mentioned appear to be from previous benchmarks. Have any new tasks been introduced or conducted in this work?

It seems somewhat unusual that the performance of TrafficFormer (2025) is lower than that of YaTC (2023) and ET-BERT (2022) in most cases—what factors might explain this discrepancy?

Finally, in the ablation study, the pre-training (PT) stage shows the greatest influence on performance across the four evaluation metrics compared to other components. Could you clarify how the pre-training data were sourced from the three repositories?

### Questions
see above weakness

### Soundness
2

### Presentation
3

### Contribution
2
