# Poisoning with A Pill: Circumventing Detection in Federated Learning

- Decision: Reject
- Scores: 8, 5, 5, 3

## Abstract
Federated learning (FL) protects data privacy by enabling distributed model training without direct access to client data. However, its distributed nature makes it vulnerable to model and data poisoning attacks. While numerous defenses filter malicious clients using statistical metrics, they overlook the role of model redundancy, where not all parameters contribute equally to the model/attack performance. Current attacks manipulate all model parameters uniformly, making them more detectable, while defenses focus on the overall statistics of client updates, leaving gaps for more sophisticated attacks. We propose an attack-agnostic augmentation method to enhance the stealthiness and effectiveness of existing poisoning attacks in FL, exposing flaws in current defenses and highlighting the need for fine-grained FL security. Our three-stage methodology—$\textit{pill construction}$, $\textit{pill poisoning}$, and $\textit{pill injection}$—injects poison into a compact subnet (i.e., pill) of the global model during the iterative FL training. Experimental results show that FL poisoning attacks enhanced by our method can bypass 8 state-of-the-art (SOTA) defenses, gaining an up to 7x error rate increase, as well as on average a more than 2x error rate increase on both IID and non-IID data, in both cross-silo and cross-device FL systems.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper is about model poisoning attacks by augmenting current poisoning attacks with efficient parameter update methodology. The author's critical observation is that not all model parameters contribute equally during inference. Therefore, if they can selectively modify the subset of parameters (critical parameters of pill) contributing, they will be able to make the attack more stealthy and ensure that it evades current defenses. They poison the pill using existing FL poisoning attacks, and carefully inject the poison pill into the target model. To raise suspicion, they ensure that the magnitude of parameter change is balanced with benign parameter's model update. They evaluated using 9 different aggregation rules, 4 model poisoning models and used well-known MNIST, Fashion-MNIST, and CIFAR-10 datasets to validate their claim. They also evaluated using both IID and non-IID data.

### Strengths
1. Their method can incorporate both cross-silo and cross-device configurations.
2. Their method is general with respect to the poisoning attack they are augmenting.
3. Using their augmentation method, they can make non-SOTA poisoning attacks perform as well as SOTA model poisoning attacks.
4. In the appendix, they show that even 10% of compromised clients can evade detection when the other attacks fail.
5. Even with non-IID data, they outperform the existing attacks in 23 out of 24 cases and achieve high error rates even in unstable and heterogeneous environments. 
6. They even provided a defense against their method. It would be nice to have some evaluation regarding it.

### Weaknesses
1. Some details regarding the parameter update are missing, such as the frequency of the model updates and how they are scheduled.  Do the compromised and non-compromised models send their updates simultaneously? Also, do they send their updates at regular intervals or differently?
2. The parameter estimation process needs to be explained more in-depth concerning the other eight aggregation rules used in the paper. Can you give me an example of how it would work for Median, FLTrust, and Flame?
3. The threat model is somewhat weak since it assumes they completely control the compromised clients. Can you tell me how the error rate would change under this scenario, where, let's say, we start with 20% of the fleet as compromised, then after some time, the number of compromised fleets decreases to 15%, then 10%, and finally 5%?

### Questions
1. How frequently do the compromised and benign clients update their models? Can the compromised clients send their update after some iterations where no poisoned update has been sent? 
2. In a scenario where the current benign model updates that will be sent are severely out of proportion compared to their old updates, will the attack be detected? Since the compromised client sent their model update, calibrated according to the magnitude of previous benign model updates?
3. How does the estimation process work for different aggregation methods used in federated learning? Is it always the mean of all the compromised model weights? This estimation procedure makes sense when the FedAvg aggregation rule is used, but it also seems to work for Median, FLTrust, and Flame. Can you prove your perspective on this?
4. In the pill search algorithm, how important is the initialization? What is the effect of different types of initialization?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a novel attack-agnostic augmentation method to enhance the stealthiness and efficacy of existing poisoning attacks in federated learning (FL) systems. Unlike traditional poisoning methods that modify all model parameters uniformly, the proposed method targets a compact subnetwork, or "pill," which allows for a more subtle attack capable of evading detection. The approach involves three main stages: pill construction, pill poisoning, and pill injection. By focusing on critical model parameters, this method bypasses multiple state-of-the-art (SOTA) defenses in FL, as evidenced by experimental results showing significantly increased error rates under various conditions.

### Strengths
The proposed method introduces a unique attack strategy that optimizes stealth by selectively poisoning critical model parameters, making it less detectable by standard FL defenses.

### Weaknesses
The paper does not thoroughly address the scalability of the proposed attack in large-scale federated learning (FL) environments with diverse client devices and varying computational capabilities. 

The dynamic pill search and injection stages could introduce significant computational overhead, especially in deep or complex models. This overhead could limit the attack's practicality in scenarios with frequent model updates or large FL models. 

The paper’s experiments focus on a few specific model architectures, and it remains unclear whether the proposed pill-based poisoning technique can generalize across diverse FL model types, such as transformer-based models. 

More baseline attacks should be considered.

### Questions
See the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This manuscript introduces an attack-agnostic augmentation technique designed to improve the stealth and effectiveness of existing poisoning attacks in Federated Learning (FL) systems. Experimental results demonstrate that, when enhanced by this method, FL poisoning attacks can evade state-of-the-art defenses, resulting in up to a 7-fold increase in error rates for current attack models.

### Strengths
• The manuscript is well-written and easy to follow.

• The study conducts extensive experiments to verify the effectiveness of the proposed method.

### Weaknesses
• The technical challenges and gaps in existing works are not adequately addressed. From the reviewer's perspective, this study merely adds an additional component that aims to identify 'critical neurons' in the network and applies existing attacks only to those neurons. The reviewer does not see the benefit of this extra component, as it may weaken the attack's effectiveness and increase computational overhead due to the identification process.

### Questions
• The 'Dynamic Pill Search' component is key point of the proposed method, but its details are missing from the main content. It is recommended to move these details from the appendix into the main text.

• Regarding the 'Pill Poisoning' design, it utilizes an ‘extra-trained model update’ as a reference model, aiming to make the generated malicious model update less divergent from the global model. This seems to contradict the original attack's goal, which is to maximize the difference between malicious and benign models. Therefore, the reviewer questions why this approach would outperform the original attacks in the FedAvg case.

• It is also suggested to evaluate the proposed method on targeted attacks.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper introduces an attack-agnostic augmentation method to bypass existing FL defenses and increase error rates. The high-level idea of this method is to constrain the poisoned updates into a sub-network instead of poisoning the whole model architecture. Via empirical experiments, this method is shown to bypass existing FL defenses and can be used as an add-on to four model poisoning attacks.

### Strengths
- This method can be combined with various poisoning attacks in FL and the idea of using sub-network sounds interesting
- The experiments are conducted on nine defenses and three datasets and show some improvement in increasing error rates
- This paper considers the stealthiness and effectiveness of the attack at the same time

### Weaknesses
1. The writing of this paper needs extensive revision for the following reasons: (i) the method is written using mostly oral sentences, lacking formal equations and (ii) the intuition and contribution of each sub-component are unclear. For example, in “**Designing Pill Blueprint”,** how the most important parameters are determined and what is its contribution to the overall method is not clearly stated.
2. This paper lacks an ablation study to investigate the effectiveness of each component including (i) the percentage of data that the adversary holds per total client data, (ii) the effect of SimAdjust and DisAdjust in L16-17, Algo. 1.
3. Results in Table 2 do not sound convincing. Except for Sign-flipping Attack, the error rates of baseline attacks are very low, i.e., 10-20% even with FedAvg (no-defense), and FLD defense even increases the error rates. On the other hand, in the original paper from Fang et al. 2020, the error rates can reach up to 80% in some cases. Can the authors explain these?
4. Related attacks focusing on model poisoning to increase stealthiness and effectiveness such as PDG, Model Replacement, Constrain-and-Scale, and Neurotoxin are not compared or discussed.

[1] Bagdasaryan, Eugene, et al. "How to backdoor federated learning." *International conference on artificial intelligence and statistics*. PMLR, 2020.

[2]  Bhagoji, Arjun Nitin, et al. "Analyzing federated learning through an adversarial lens." *International conference on machine learning*. PMLR, 2019.

[3] Zhang, Zhengming, et al. "Neurotoxin: Durable backdoors in federated learning." *International Conference on Machine Learning*. PMLR, 2022.

### Questions
1. This method seems over-complicated with extensive training steps. How is the computational time of this attack compared to others from benign clients? whether it can introduce too long a waiting tie and raise suspicious from the server.
2. How to ensure the fidelity of the estimated benign update, since it is heavily depending on the quality and distribution of the data from the adversarial clients? What is the assumption about the data that the adversary has across compromised clients?
3. In Line 257,  what is the contribution of the starting point defined by an attacker on the overall performance since the error can be propagated to the next layers if the first neuron on the first layer is not the optimal one? This procedure and why does it work is unclear to me.
What is the indication of these fill’s neurons, and their role on the network?
This discussion should be presented in the main manuscript.
4. The number of clients is smaller than the existing FL attack setting, which may lead to a large amount of data that an attacker has, could authors show this method can work with larger systems and smaller amounts of data from attackers?
5. “Malicious 20%” is too strong and unrealistic, whereas practical attacks often use 1-5% of the malicious clients. Whether or not this attack can work with smaller attacker numbers?
6. Can this method be used with backdoor attacks?
7. Can this method work with more complex model architecture such as ResNet/VGG?

### Soundness
2

### Presentation
2

### Contribution
2
