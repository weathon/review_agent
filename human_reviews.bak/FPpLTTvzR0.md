# IDEA: Invariant Causal Defense for Graph Adversarial Robustness

- Decision: Reject
- Scores: 6, 8, 6, 5

## Abstract
Despite the success of graph neural networks (GNNs), their vulnerability to adversarial attacks poses tremendous challenges for practical applications. Existing defense methods suffer from severe performance decline under some unknown attacks, due to either limited observed adversarial examples (adversarial training) or pre-deﬁned heuristics (graph puriﬁcation or robust aggregation). To address these limitations, we analyze the causalities in graph adversarial attacks and conclude that causal features are desirable to achieve graph adversarial robustness, owing to their determinedness for labels and invariance across attacks. To learn these causal features, we innovatively propose an Invariant causal DEfense method against adversarial Attacks (IDEA). We derive node-based and structurebased invariance objectives from an information-theoretic perspective. IDEA is provably a causally invariant defense across various attacks. Extensive experiments demonstrate that IDEA signiﬁcantly outperforms all baselines under both poisoning and evasion attacks on ﬁve benchmark datasets, highlighting its strong and invariant predictability. The implementation of IDEA is available at https://anonymous.4open.science/r/IDEA_repo-666B.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a method to disentangle causal and non causal features in order to protect a gnn from adversarial attacks

### Strengths
- Good presentation 
- Great experimentation
- The addition of the ablation study is very welcome 
- Interesting premise and motivation 
- Interesting approach

### Weaknesses
- Minor language issues throughout the paper 
- It is not clear why the end result of the encoder H would be a causal feature. I can see how the learned representation could be a feature set that is invariant to the features the attack but making a causal claim is not completely justified. 
- it is unclear how the method generalises to multiple unseen attacks

### Questions
- why is the features learned causal and not just invariant to the attack ? 
- how does the method generalise to types of attacks not seen in training ?


EDIT AFTER REBUTTAL

updated score from 5 to 6

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper uses a new causal defense perspective to resist adversarial attacks by learning powerful and invariant predictable causal features, and proposes the Invariant causal defense method against adversarial attacks (IDEA). Experiments have proven that this method is effective against various attack methods and has excellent performance and strong generalization.

### Strengths
1. The article contains a more complete proof process and theoretical basis.

2. The article learns powerful and immutable causal features to deal with adversarial attacks from a relatively novel causal defense perspective.

3. The experiment proves the effectiveness and generalization of the method proposed in the paper.

### Weaknesses
1. Judging from the results shown in Table 1 and Table 2, compared with other denoising methods, the performance of this method has indeed been greatly improved. However, the table only shows the excellence of this method when facing one of poisoning attacks or evasion attacks, and the results for other attack methods are not shown.

2. There are many symbols listed in the article, and it seems unclear when mixed together.

3.The drawing of the overall block diagram of the method is relatively rough.


============================================================
After rebuttal

The authors solve most of my concerns. Thus, I am willing to increase the rating score from 6 to 8.

### Questions
1. Does this method still have such obvious advantages in the face of the other attack methods mentioned in the article?

2. There are also some purification methods that seem to be able to be extended to graph purification. Whether they will also encounter the limitations mentioned by the author, I hope the author can discuss or compare this. Such as the following methods:

[1] Shi C, Holtz C, Mishne G. Online adversarial purification based on self-supervision[J]. arXiv preprint arXiv:2101.09387, 2021.

[2] Liao F, Liang M, Dong Y, et al. Defense against adversarial attacks using high-level representation guided denoiser[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 1778-1787.

[3] Zhou D, Wang N, Peng C, et al. Removing adversarial noise in class activation feature space[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 7878-7887.

[4] Naseer M, Khan S, Hayat M, et al. A self-supervised approach for adversarial robustness[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 262-271.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The manuscript proposes a new framework for adversarial robustness in GNNs. The primary subject of interest is the learning of causal features defending against evasion and poisoning attacks. The empirical results are further supported by theoretical analyses with provable defense guarantees.

### Strengths
Overall the experimental methodology is sound with complete theoretical derivations.

### Weaknesses
The link is expired (https://anonymous.4open.science/r/IDEA_repo-666B), which made further investigation on code artifact and validating empirical results hard. Therefore, the claims made in the paper cannot be carefully checked. Additionally, the claim that causality directly contributes to improved defense performance is weak, as opposed to algorithmic superiority.

### Questions
What's the significance of Figure 1 (b)? Aren't there inherent learning capability differences between different GNN architectures?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a casual defense to improve the graph adversarial robustness. Specifically, it first defines a causal graph that some casual feature would have strong predictability for the label and maintains invariant predictability across attack domains so that perturbing the features adversarially won't induce a successful attack. It then defines a objective by defining different mutual information to learn the casual features. The experiments show the proposed method could achieve a significant improvement over current defenses.

### Strengths
1. The paper provides a new perspective on the casual inference to defend against graph adversarial attacks.
2. The proposed method shows significant improvement on both evasion and poisoning settings.

### Weaknesses
1. Since I am not expert in causal inference, it is unclear to me how the initial casual graph is defined. And I am not clear whether the graph is based on author's assumption or derived automatically. If it is the former case, the truthfulness of the provided causal graph is debatable. 
2. The threat model is actually unclear. The proposed method actually built a detector neural network in the defense. The attacks tested seems have no knowledge about the detector network. Therefore, the improvement might be brought by the attacker's incapability acquiring enough model information. Also,   it is unfair to compare with other proposed methods since they are only modifying the provided model or aggression rule. A adaptive attack  or white box attack should assume the attacker has already known the added detection neural networks.
3. There are some notations and figures problems that causes the paper not easy to follow. 
Node j in Figure 2 should be Node k.  Z in Section 3.2.1 is not defined.  (.)_\cN is never defined. The overall framework only shows in Figure 3 without any introduction. Empty graph in Figure 5.

### Questions
1. Why does causal feature would only connect with label and input data? Is it defined or derived just based on some assumption the paper made or is there anyway to automatic define the graph?
2. If the attacker knew the added neural network, would the proposed method still achieve a similar improvement in the paper?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
