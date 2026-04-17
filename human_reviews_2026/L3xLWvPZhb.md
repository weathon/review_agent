# Can Exploration Save Us from Adversarial Attacks? A Reinforcement Learning Approach to Adversarial Robustness

- Decision: Reject
- Scores: 8, 2, 4

## Abstract
Although considerable progress has been made toward enhancing the robustness of deep neural networks (DNNs), they continue to exhibit significant vulnerability to gradient-based adversarial attacks in supervised learning (SL) settings. We investigate adversarial robustness under reinforcement learning (RL), training image classifiers with policy-gradient objectives and $\epsilon$-greedy exploration. When training models with several architectures on CIFAR-10, CIFAR-100, and ImageNet-100 datasets, RL consistently improves adversarial accuracy under white-box gradient-based attacks. Our results show that on a representative 6-layer CNN, adversarial accuracy increases from approximately 5\% to 55\% on CIFAR-10, 2\% to 25\% on CIFAR-100, and 5\% to 18\% on ImageNet-100, while clean accuracy decreases only 3–5\% relative to SL. However, transfer analysis reveals that adversarial examples crafted on RL models transfer poorly: both SL and RL retain approximately 43\% accuracy against these attacks. In contrast, adversarial examples crafted on SL models transfer effectively, reducing both SL and plain RL to around 8\% accuracy. This indicates that while plain RL can prevent the generation of strong adversarial examples, it remains vulnerable to transferred attacks from other models, thus requiring adversarial training (RL-adv, $\sim$30\% adversarial accuracy) for comprehensive defense against cross-model attacks. Analysis of loss geometry and gradient dynamics show that RL induces smaller gradient norms and rapidly changing input-gradient directions, reducing exploitable information for gradient-based attackers. Despite higher computational overhead, these findings suggest RL-based training can complement existing defenses by naturally smoothing loss landscapes, motivating hybrid approaches that combine SL efficiency with RL-induced gradient regularization.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
- Casts image classification as a one-step RL problem and trains classifiers as policies with ε-greedy exploration (optionally combined with FGSM/TRADES-style regularization).

- Reports gains in white-box L2 PGD robustness on CIFAR-10/100 and ImageNet-100, with a modest drop in clean accuracy versus standard supervised training.

- Observes transfer asymmetry: adversarial examples crafted on RL models transfer poorly, while those crafted on SL models still hurt plain RL

- RL training yields smaller gradient norms and less stable input-gradient directions, making gradient-based attacks less effective.

- Acknowledges limitations: evaluation lacks stronger standardized attacks (e.g., AutoAttack) and fuller threat models; training introduces extra compute overhead.

### Strengths
- Clear empirical finding: ε-greedy RL training notably boosts white-box L2 PGD robustness with modest clean-accuracy drop

- Solid diagnostics: analyzes gradient norms, direction stability, and loss landscape to explain why attacks struggle

- Broad(ish) coverage: multiple datasets (CIFAR-10/100, ImageNet-100) and several backbones show the effect is not a one-off

- Practical angle: simple to add on top of standard training and compatible with adversarial regularization (e.g., TRADES)

- Transparent about limits: explicitly discusses stronger attacks needed and positions work as complementary, not a silver bullet

### Weaknesses
- Evaluation is narrow: no AutoAttack, L∞, or true black-box attacks to rule out gradient masking.

- Transfer remains a problem: adversarial examples crafted on SL models still break plain RL; robustness largely relies on RL + adversarial training.

- Compute/efficiency unclear: added exploration and optional adversarial phase increase cost without a compute-matched analysis

- Scalability is uncertain: results are mostly on smaller backbones/datasets; unclear behavior on modern large ViTs/foundation models.

### Questions
- The authors have mentioned the computational overhead, but can you share some stats about this and how it differs from the CNN backbones to ViTs

- I'm really curious to see how this would scale to bigger models

- Also some analysis on the representation space of the model trained with RL and adversarial defense, e.g, how the features could transform to other domain or even their visualization would be nice

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper provides a rarely explored perspective of combing reinforcement learning (RL) with adversarial training (AT) for more robust image classification. Experiments are conducted on 4/6-layer CNN and ResNet-18.

### Strengths
Figure.3 is interesting and intuitive.

### Weaknesses
Overall, this paper is largely unpolished and barely meets the standard for a readable scientific paper. Major issues range from motivation, novelty, solidity and clarity.

**Motivation**. Since AT is notoriously well-known for its degradation on clean performance and prolonged training time, combining it with RL, which is also extremely expensive for training, is counter-intuitive and requires a strong motivation. However, this paper seems to be merely motivated by relative less exploration, making one question why taking the labour of RL to combine it with AT for even worse efficiency and performance only for boosted robustness. 

**Novelty**. Furthermore, according to sec.4, the entire method is build upon simple combination of existing $\epsilon$-greedy RL and TRADES. The paper itself provides little or no insights/innovation beyond a simple 1+1. Besides, the preliminary feels remotely related to the upcoming method section,  making no contribution to understanding the proposed method.

**Solidity**. Experiments are conducted on severely subpar models, i.e., CNN and RN18, without an explicit explanation but an adjective ‘representative’. No mainstream models, including ViT, Wide ResNet, or even RN-50, are tested. Furthermore, the baselines used for training and attacks are also significantly outdated. No recently SOTA methods such as AutoAttack, are considered or discussed, underming the experimental solidity of the paper.


**Clarity**: Writing-wise, this paper is poorly written and full of format/punctuation/spelling mistakes, further lowering the readability of this paper. To start with, sec.3 seems diverge largely from the storyline, making it confused to those many introduced terms, to which are not referred until the experiment. Besides, repetitive notations also appear during introduction of the method, e.g., $\epsilon$ is used for both RL and AT. For format, most of the reference formats are incorrect, nested and hard to trace. Many references are placed at the end of sentences where no clear clue is given. Caption for tables are incorrectly capitalized, and the top rules of tables are constantly missing. Lastly, for punctuations, all quotation marks are incorrectly used, and all captions of tables are ended without periods.

**Miscellaneous**. I) Although an anonymous GitHub repository is provided claiming to include configuration and codes, the repository contains nothing but the paper abstract. II) It is unclear why the author include several formal definition of existing conceptions/methods into the supplementary.

Overall, this paper falls far behind regarding a standard scientific paper and requires enormous effort to improve its motivation, novelty, solidity and clarity.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a reinforcement learning–based architecture for image classification, demonstrating that the learned models are more robust against white-box adversarial attacks compared to standard supervised learning baselines. The approach effectively leverages exploration and policy optimization to improve gradient stability and decision consistency. This finding provides meaningful insight into how reinforcement learning principles can enhance adversarial robustness in vision models.

### Strengths
The paper provides appropriate and well-aligned analyses that effectively support its main claim. The empirical evidence strengthens the overall argument and helps explain why the proposed approach improves robustness.

### Weaknesses
The paper evaluates only relatively small 6-layer CNNs. Such lightweight architectures limit the generality of the conclusions: robustness phenomena can change substantially with deeper or modern architectures (ResNets, WideResNets, Vision Transformers). I recommend repeating the key experiments on at least one standard strong baseline (e.g., ResNet-18 / ResNet-50 or a small ViT) to show that the reported RL benefits scale beyond toy models.

The results suggest a trade-off between clean (unperturbed) accuracy and adversarial accuracy, but this is not analyzed quantitatively. Please report the clean vs. adversarial accuracy Pareto frontier (varying ε, training hyperparameters, and amount of adversarial augmentation), and provide confidence intervals or significance tests across multiple seeds. This will clarify whether RL strictly improves robustness, merely shifts the trade-off, or hurts clean accuracy unacceptably.

The unified theoretical section offers intuitive explanations linking exploration to gradient instability, but it does not provide formal guarantees or precise assumptions under which the claims hold. The authors should (a) clearly state the assumptions and limits of the theory, (b) avoid wording that implies formal proof where none exists, and (c) either add a theorem with a clear statement and sketch proof or explicitly label the section as “insights / hypotheses” supported by empirical evidence.

The paper focuses on white-box, gradient-based attacks; however, the transferability section indicates RL may behave differently under transfer attacks. To be convincing, include a systematic study of black-box attacks (transfer attacks from other models, query-based attacks) and show how RL compares to SL and adversarially trained baselines under those threat models. Report cross-model transfer matrices (attack source × target) and query-limited attacks to characterize real-world vulnerability.

### Questions
How does the proposed RL method perform under black-box or transfer attacks (e.g., when adversarial examples are generated from another model)?

Would the observed robustness trends still hold for more complex architectures such as ResNets or Vision Transformers?

Could the authors provide more details on the computational cost and practicality of their approach? Specifically, since RL training is generally more expensive than SL, it would be helpful to include runtime or sample-efficiency comparisons to better support the claims of practical applicability.

Could the authors evaluate their approach under AutoAttack and adaptive gradient-free methods to help rule out the possibility of gradient masking?

### Soundness
3

### Presentation
3

### Contribution
2
