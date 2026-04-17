# TriGuardFL: Triple-Step Byzantine-Robust Federated Learning against Model Poisoning Attacks

- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
Federated learning's (FL) distributed architecture is promising, yet it is vulnerable to model poisoning attacks that degrade global model accuracy. Existing defense strategies typically compare the locally updated gradients of clients and exclude or down-weight those exhibiting substantial deviations. However, these strategies may become ineffective when the clients’ datasets are heterogeneous. In this paper, we propose TriGuardFL, a novel triple-step defense framework that robustly discriminates malicious actors from benign and non-IID clients. First, we employ a cosine-similarity-based filter to identify suspicious clients. Second, a fine-grained secondary evaluation assesses their performance using a small class-stratified dataset. By analyzing class-wise performance differences, it can discern whether a divergent update stems from a malicious attack or data heterogeneity. Finally, a Bayesian reputation model is integrated to manage the uncertainty of detection and enhance the long-term robustness. Extensive case studies on two benchmark datasets and three representative model poisoning attacks demonstrate that TriGuardFL outperforms existing methods in mitigating the impact of model poisoning attacks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes TriGuardFL, a three-step defense framework for Byzantine-robust federated learning (FL). It addresses a challenge in FL—distinguishing between malicious and benign clients under non-IID data. The method integrates (1) cosine-similarity filtering, (2) class-wise statistical testing using a small clean dataset, and (3) a Bayesian reputation model for long-term robustness. Experiments on Fashion-MNIST and CIFAR-10 with several CNN architectures show improvements over baselines such as DeFL, FLTrust, and Multi-Krum.

### Strengths
1. This paper identifies a limitation of existing Byzantine-robust FL methods under non-IID conditions.
2. The triple-step design is intuitive, each step compensates for the others’ weaknesses.
3. This paper uses multiple datasets, architectures, and attack types (Min-Max, Min-Sum, LIE) under both full and partial knowledge settings.

### Weaknesses
1. Each step (similarity filtering, validation via small dataset, Bayesian weighting) has been explored before in isolation. The paper lacks strong conceptual unification beyond “combining three steps.”
2. The assumption that the server owns a few-shot clean dataset breaks the FL privacy model and reduces practicality for real-world deployment.
3. Only image classification benchmarks are used; no large-scale or cross-domain experiments (e.g., NLP or medical data).
4. Mathematical analysis mostly restates known FedAvg convergence bounds (Li et al., 2019) with small extensions for malicious clients—little theoretical innovation.
5. The paper only considers classical attacks (Min-Sum, Min-Max, LIE). It ignores modern adaptive attacks like gradient-sign inversion, backdoor or stealthy attacks that exploit cosine-based detection.

### Questions
1. How much each of the three steps contributes individually to the robustness gains.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work proposes TriGuardFLhat to discriminate malicious actors from benign and non-IID clients. It first filters potential attackers using cosine similarity, and then perform a statistical significance test. Finally, it employs a dynamic Bayesian reputation system to track client behavior over time, using this reputation score to weight model aggregation and perform long-term client selection.

### Strengths
- The work tackles a key but difficult challenge in FL, i.e., distinguishing malicious clients from benign outliers (non-IID).

- The three-step detection/filter is logical and comprehensive.

- The introduction of a Bayesian reputation system with a discount factor provides adaptability.

### Weaknesses
- There is a fatal logic vulnerability that the a simple adaptive attacker can mimic the non-IID client to pass the test. The experiments also fail to test against adaptive attacks.

- The reputation system (Step 3) blindly trusts the flawed detector (Step 2), and will reward successful attackers by boosting their reputation scores, making the system actively counter-productive.

- Step 3 applies a zero-weight filter to any client, including benign ones, which means a significant increase in false positives.

- A clean few-shot server dataset covering all classes is an extremely strong prerequisite that is impractical in many FL settings and undermines privacy principles.

- For Sec 4.2 ROBUSTNESS ANALYSIS, it is wrong as it does not prove the effectiveness of the defense itself.

### Questions
- In Step 2, a single t-test comparing the two groups $C_1$ v.s. $C_2$ yields only one p-value. Does this imply that multiple different tests are performed?

- How can the server obtain knowledge of the full class space of all clients to build $D'$ without violating FL privacy norms? How about a client introduces a novel class unknown to the server?

- Is there any design to defend against an attacker who intentionally spoofs a non-IID client, or maybe clients?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents TriGuardFL, a three-step defense framework for Byzantine-robust federated learning under non-IID settings. In Step 1, the server detects potentially malicious clients using cosine similarity between each client’s update and the aggregated global model. In Step 2, a class-wise evaluation is performed on a small server-side dataset using a $t$-test to distinguish benign non-IID clients from adversarial ones. In Step 3, a Bayesian reputation update is used to assign lower weights to low-reputation clients in future aggregation rounds.

### Strengths
+ The paper addresses an important problem: improving the robustness of federated learning when client data are heterogeneous.
+ The class-wise evaluation step is designed to reduce false positives caused by label distribution imbalance.
+ The Bayesian reputation mechanism adds temporal adaptivity to client weighting.
+ The experimental results show some improvement in robustness across multiple attack settings.

### Weaknesses
- The proposed framework mainly integrates ideas that already exist in prior Byzantine-robust FL studies rather than introducing a substantially new method.
Step 1 is very similar to the cosine-similarity-based filtering in FLTrust, where the server compares each client’s update direction with a trusted reference model. The claimed link to FLDetector is inaccurate, as FLDetector focuses on temporal consistency checks across rounds rather than per-round similarity comparison. Step 2 follows the general idea of client evaluation as in DeFL, although DeFL uses gradient-norm metrics instead of $t$-tests. Step 3 conceptually matches reputation- or trust-based FL frameworks, which track client reliability over time using reputation-weighted aggregation or trust propagation, as discussed in surveys on Trustworthy FL. Overall, the contribution mainly combines known components into a single framework without introducing a clear algorithmic or theoretical innovation.
- In Section 3.1, cosine similarity is defined using $\nabla F_i(w_{K_i,t})$. In standard FL, the server cannot compute client gradients and only receives updated weights $w_{K_i,t}$. Therefore, the feasibility of Step 1 is unclear.
- The approach assumes that the server holds a small labeled dataset that includes all classes. This assumption weakens the privacy guarantees of FL and may not hold in realistic deployments.
- The design of the class-wise $t$-test is not well explained. There are no details about sample sizes, independence assumptions, or correction for multiple comparisons. The justification for using loss differences as the test statistic is also missing.
- The convergence proof relies on $\mu$-strong convexity and $L$-smoothness, which are not valid for deep CNNs. This makes the theoretical analysis largely symbolic and not directly applicable to the experimental models.
- The evaluation is incomplete and lacks analysis depth. There are no experiments on targeted, backdoor, or adaptive attacks, even though the paper claims general Byzantine robustness. No ablation studies are provided for key hyperparameters ($\delta_1$, $\delta_2$, $\varepsilon$, $r$, $T_1$) or for the effect of the server dataset size $|D'|$. Some baseline methods perform equally well or even better in certain cases, which raises doubts about the claimed advantage of TriGuardFL.
- Several notation and presentation problems reduce clarity. The paper inconsistently uses $\gamma_t$ and $\gamma_{i,t}$ in the update equation. The parameter $\delta_2$ appears in Algorithm 1 but is never defined, and its relation to the “Significance Level = 0.001” in Table 1 is unclear. The text also switches between “parameters” and “gradients”, which is inconsistent with the mathematical formulation in Step 1.
- The formatting of tables is inconsistent. Table 1 has its caption above the table, while Tables 2 and 3 have captions below. The caption placement should follow a consistent format.

### Questions
1.Which part of TriGuardFL is genuinely novel beyond the existing methods such as FLTrust and DeFL?

2.How is $\nabla F_i(w_{K_i,t})$ obtained if the server does not have access to local client data?

3.What is the size and class coverage of the server dataset $D'$? How does performance vary if $D'$ is incomplete?

4.How is the $t$-test validated when the sample size is small?

5.Can adaptive or backdoor attackers exploit the $t$-test mechanism to evade detection?

6.Please provide ablation and sensitivity results for $\delta_1$, $\delta_2$, $\varepsilon$, $r$, and $T_1$.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses a key vulnerability in Federated Learning (FL): standard defenses against model poisoning often fail in non-IID (heterogeneous) settings, as they cannot distinguish between malicious updates and the natural, harmless deviations from clients with different data distributions.

The authors propose TriGuardFL, a novel triple-step defense framework designed to solve this specific problem. First, it uses a cosine-similarity-based filter to identify a broad list of "suspicious" clients. Second, it performs a fine-grained evaluation on these suspicious clients using a small, class-stratified dataset held by the server. By analyzing per-class performance, it can discern if a deviation is from a malicious attack (which tends to degrade performance uniformly) or a benign non-IID client (who may perform poorly on some classes but well on others). Finally, it integrates a Bayesian reputation model to track client behavior over time, which manages detection uncertainty and enhances long-term robustness.

Extensive experiments on Fashion-MNIST and CIFAR-10 show that TriGuardFL outperforms existing state-of-the-art defenses like DeFL, FLTrust, and Multi-Krum, particularly in non-IID settings where others fail in terms of the average rank of the defense against these attacks. The authors acknowledge that the method's reliance on a clean server-side dataset is a limitation.

### Strengths
- The work identifies a critical vulnerability in federated learning: the difficulty of distinguishing between genuinely malicious behavior and the natural, divergent updates from benign clients in a non-IID setting. They propose a novel heuristic for this problem, based on the insight that malicious updates tend to degrade performance uniformly, while benign non-IID clients will exhibit high variance in their per-class performance.
- The multi-stage detection architecture is logical. It uses a broad, low-cost filter (cosine similarity) to create a "suspicious" list and applies a more expensive, fine-grained analysis only to that list. The third step is necessary for long-term stability and to tolerate potential detection errors.
- The integrated Bayesian reputation model is a standard approach that suits the setting well, using a Beta distribution to manage uncertainty  rather than proposing an overly complex new system. The reputation system handles practical details well, such as using a "hard filter" to set a client's aggregation weight to zero if flagged as malicious in the current round , and implementing "long-term gatekeeping" to remove consistently low-reputation clients.
- The reputation system handles practical details well, such as using a "hard filter" to set a client's aggregation weight to zero if flagged as malicious in the current round , and implementing "long-term gatekeeping" to remove consistently low-reputation clients.
- While the experiments use two benchmark datasets , they are evaluated across a diverse set of five different network architectures, including LeNet, AlexNet, VGG11, VGG16, and ResNet18, which strengthens the claims of its effectiveness .

### Weaknesses
- The paper's biggest weakness is that it does not evaluate against an adaptive adversary. The core of the defense relies on the Step 2 heuristic that malicious clients degrade performance uniformly, while benign non-IID clients show high per-class variance. An adaptive attacker could easily fool this by crafting a malicious update that performs very well on one arbitrary class, thereby disguising itself as a benign non-IID client. This lack of stress-testing means the central idea of the paper is not fully validated.
- The first step, cosine-based shortlisting, is vague. Equation 4 compares the client gradient $\nabla F_i(w_{i,t}^K)$ with a global gradient $\nabla F(w_t')$. It is not clear how this global gradient is computed. This first step also suffers from a "chicken-and-egg" problem. It uses an "initial aggregation" $w_t'$ as the reference for similarity. If this reference is already poisoned by attackers, the whole defense could break, as malicious clients might appear "similar" to the poisoned average. It also relies on a hard-coded threshold ($\delta_1$), which is a brittle defense mechanism.
- The paper lacks a clear analysis of false positives. While it mentions the reputation system tolerates false negatives , it provides no evidence that benign non-IID clients are not incorrectly flagged and eventually "starved" or removed by the long-term gatekeeping mechanism .
- The experimental diversity is limited. Instead of using four different complex models (VGG11, VGG16, ResNet18) on the single CIFAR-10 dataset, the paper would have been more convincing if it had demonstrated its effectiveness on more diverse data modalities, such as text.
- The attack scenario tested represents a relatively weak threat. The experiments use a 12.5% malicious ratio (4 of 32 clients), but only sample 50% of clients per round (16 clients). This means, on average, only two attackers are active in any given round. The paper does not convince that the defense is robust against higher, more realistic proportions of attackers. Even when 64 clients were simulated, the fraction of malicious clients was still 12.5% which is low. 
- The main results table (Table 2) reports only test loss, not accuracy. Loss is a less intuitive metric for performance. Furthermore, while TriGuardFL wins on average loss score, it does not consistently outperform all other defenses in every individual scenario. For example, on the partial knowledge attack, TriGuardFL wins only 3 out of 15 times. 
- The design of the Step 2 filter seems to reward high variance, which may not be desirable. A client is deemed benign if its "good" class performance is significantly different from its "bad" class performance, which is a strange and potentially exploitable proxy for "benign-ness."
- The three-stage process, especially the per-class, per-client analysis in Step 2 , introduces significant computational overhead for the server, which is never measured or discussed.

### Questions
- Given that the Step 2 filter is the key innovation but seems easily fooled by an adaptive adversary, how can the defense be modified to handle a smart adversary?
- Since the Step 1 filter uses the (potentially poisoned) initial aggregate $w_t'$ as its reference, how can this stage be modified so that the defense stays robust even when the reference is poisoned? Some prior defense techniques [1] handle this.

[1]: Sharma, Atul, et al. "Flair: Defense against model poisoning attack in federated learning." Proceedings of the 2023 ACM Asia Conference on Computer and Communications Security. 2023.

### Soundness
2

### Presentation
3

### Contribution
2
