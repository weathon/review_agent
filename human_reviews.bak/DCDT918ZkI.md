# Boosting the Adversarial Robustness of Graph Neural Networks: An OOD Perspective

- Decision: Accept (poster)
- Scores: 8, 3, 6, 6

## Abstract
Current defenses against graph attacks often rely on certain properties to eliminate structural perturbations by identifying adversarial edges from normal edges. However, this dependence makes defenses vulnerable to adaptive (white-box) attacks from adversaries with the same knowledge. Adversarial training seems to be a feasible way to enhance robustness without reliance on artificially designed properties. However, in this paper, we show that it can lead to models learning incorrect information. To solve this issue, we re-examine graph attacks from the out-of-distribution (OOD) perspective for poisoning and evasion attacks and introduce a novel adversarial training paradigm incorporating OOD detection. This approach strengthens the robustness of Graph Neural Networks (GNNs) without reliance on prior knowledge. To further evaluate adaptive robustness, we develop adaptive attacks against our methods, revealing a trade-off between graph attack efficacy and defensibility. Through extensive experiments over 25,000 perturbed graphs, our method could still maintain good robustness against both adaptive and non-adaptive attacks. The code is provided at https://github.com/likuanppd/GOOD-AT.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors present a method to improve GNN adversarial robustness. The approach trains $K$ OOD edge detection MLPs that classify edges based on their internal representations of a GCN model and their input attributes. At inference time, the OOD detector ensemble predicts for each edge whether it is potentially adversarial or not, and removes the edge according to that decision. The authors present evidence based on a GNN robustness benchmark suggesting their defense outperforms existing ones. Further, the authors study robustness to poisoning attacks as well as inductive evasion attacks.

### Strengths
* The authors present compelling results on a diverse robustness benchmark
* The approach is clear and well-motivated
* The authors consider a wide range of settings including transductive evasion and poisoning attacks as well as inductive attacks

### Weaknesses
* While the range of settings considered is wide, the set of models and dataset considered is quite narrow in return
* The transductive evasion defense could potentially be reduced to the trivial perfect defense mentioned in [Gosch et al. 2023]
* Regarding Proposition 1, the "proof" isn't really a proof, and, does not show a problem with adversarial training for graphs, but highlights that the definition of the perturbation set is too loose.

### Questions
* My most pressing concern is regarding the very recent work [Gosch et al. 2023]. I acknowledge that it is so recent that it is not reasonable to expect the authors to have it in their paper, yet their results are very relevant for this work. Specifically, the authors propose a trivial perfect defense for evasion attacks in transductive setting, which effectively memorizes the clean input graph and ignores the potentially perturbed graph at inference time (Proposition 1 of the referenced work). Assuming unique node attributes, wouldn't the perfect version of the OOD ensemble defense reduce to this trivial defense?
* On a relate note, [Gosch et al. 2023] also notice the problem with the too loose definition of the set of allowed perturbations, and, in turn, present a method that employs local constraints. How does this affect Proposition 1 in this work?
* How is the threshold $t$ determined? Is it the same across OOD detectors in the ensemble?
* Typically, ensembles work via majority vote. In this work, the authors flag an edge as potentially adversarial if **one** of the detectors in the ensemble does. What is the reason for this?
* The OOD detectors are trained specifically for an individual GNN classifier instance. I wonder if it is also possible to transfer the OOD detector ensemble to a different GNN instance?

References
---
Gosch, L., Geisler, S., Sturm, D., Charpentier, B., Zügner, D., & Günnemann, S. (2023). Adversarial Training for Graph Neural Networks. NeurIPS 2023. https://arxiv.org/abs/2306.15427

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper adopts an out-of-distribution perspective to re-examine graph adversarial attacks and analyze the distributional shift phenomena in both poisoning and evasion attacks at graph and edge levels. The authors propose an adversarial training method that trains multiple OOD detectors to improve the GNN’s robustness. Through extensive experiments, we validate the adaptive and non-adaptive robustness of our approach.

### Strengths
1. The authors show that  the simple adversarial training will lead to the model learning incorrect knowledge
2. The authors conduct extensive experiments over 25,000 graphs to compare the robustness of our methods with other baselines

### Weaknesses
1. The paper is hard to follow
2. The authors show that adversarial edges are OOD, which is straightforward
3. Notations are hard to understand. 
4. The tested defenses are already shown to be vulnerable to adaptive attacks 
5. Lack of comparison with provable defense results.

### Questions
Proposition 1 is very hard to follow

What are the key differences between OOD-detection-based Adversarial Training vs. standard adversarial training? 

The tested defenses, shown in Figure 2,  are already shown to be vulnerable to adaptive attacks. Why do you choose them as baselines? 
 
Why not evaluating the results of certified defense? How about the comparison with them?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors theoretically show that AT can lead to models learning incorrect information.
To overcome this issue, they use AT paradigm incorporating OOD detection.

### Strengths
1) The idea of suggesting AT paradigm with OOD detection on GNN is novel.

2) The theoretical part is sound.

3) Experiments are sound. I specifically liked that they tested their solution even when the attacker has full knowledge of the detectors, and not only the standard model.

4) paper is well written and easy to follow.

### Weaknesses
One question I had is - if the detector is an ensemble, how will it work against an EoT adversary? Can the authors test this kind of scenario?

### Questions
See Weaknesses section.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies improving the robustness of graph neural networks (GNN) to both evasion and poisoning attacks. The main idea is to model the generated adversarial edges from existing attacks as out of distribution (OOD) data, and trains OOD detectors remove adversarially perturbed edges. Empirical results show that the proposed approach outperforms existing baseline defenses in various settings.

### Strengths
1. The idea of using the OOD nature of adversarial edges to train OOD detectors and remove perturbed edges is interesting.
2. The proposed defense also achieves remarkable performance against different state-of-the-art attacks compared to existing baseline defenses.

### Weaknesses
1. Hypothesis 1 is the key high-level motivation for designing OOD detector based defenses. However, I am not sure how much this hypothesis can help in practical applications. Specifically, it is indeed true that the attacks can be more destructive when the distributions shift more significantly. However, we measure the empirical vulnerability by comparing the accuracy drops before and after the attack, and there might exist a situation where the attack is considerably successful (from practical point of view) without being significantly different from the in-distributions.
2. The OOD samples are generated by PGD attacks, which can be restricted. The current experimental setting does not eliminate the possibility of adaptive attackers targeting the OOD detector, meaning there might exist some OOD samples that are still effective at inducing distribution shifts, but are not similar to PGD induced OOD samples. 
3. In the poisoning scenario, the whole process is based on the assumption that effective poisoning perturbations happen in the training nodes. This again does not capture the fact that, if it is simply impossible for attackers to induce more effective poisoning attacks by targeting other nodes in the graph.

### Questions
The common theme of the 3 weaknesses above is that the proposed approach does not provide a more fundamental reasoning on the effectiveness of the proposed approach. If the authors have a more fine-grained analysis, the quality of the paper will be improved.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
