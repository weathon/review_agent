# Physical Backdoor Attack can Jeopardize Driving with Vision-Large-Language Models

- Decision: Reject
- Scores: 6, 8, 8, 3

## Abstract
Vision-Large-Language-models (VLMs) have great application prospects in autonomous driving. Despite the ability of VLMs to comprehend and make decisions in complex scenarios, their integration into safety-critical autonomous driving systems poses serious security risks. In this paper, we propose \texttt{BadVLMDriver}, the first backdoor attack against VLMs for autonomous driving that can be launched in practice using \textit{physical} objects. Unlike existing backdoor attacks against VLMs that rely on digital modifications, \texttt{BadVLMDriver} uses common physical items, such as a red balloon, to induce unsafe actions like sudden acceleration, highlighting a significant real-world threat to autonomous vehicle safety. To execute \texttt{BadVLMDriver}, we develop an automated pipeline utilizing natural language instructions to generate backdoor training samples with embedded malicious behaviors. This approach allows for flexible trigger and behavior selection, enhancing the stealth and practicality of the attack in diverse scenarios. We conduct extensive experiments to evaluate \texttt{BadVLMDriver} for two representative VLMs, five different trigger objects, and two types of malicious backdoor behaviors. \texttt{BadVLMDriver} achieves a 92% attack success rate in inducing a sudden acceleration when coming across a pedestrian holding a red balloon. Thus, \texttt{BadVLMDriver} not only demonstrates a critical security risk but also emphasizes the urgent need for developing robust defense mechanisms to protect against such vulnerabilities in autonomous driving technologies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes BadVLMDriver, a backdoor attack method against VLMs for autonomous driving. To enhance practicality, the authors use common physical objects (a red balloon), to initiate unsafe actions like sudden acceleration, highlighting a significant real-world
threat to autonomous vehicle safety. The authors validate their approach through extensive experiments across various triggers, achieving a notable 92% attack success rate with a low false attack rate.

### Strengths
1) The authors propose the first physical backdoor attack method for VLMs to arouse public awareness.

2) Extensive experiments conducted with five different trigger objects demonstrate the critical safety risk caused by backdoor attacks in autonomous driving.

### Weaknesses
1) The novelty is straightforward. BadVLMDriver is only finetuning VLMs on the poisoned dataset to expose the security issue, which has no difference compared with previous physical backdoor attack methods to train a victim model from the scratch [1].

2) Although the paper identifies the urgent need for effective defenses, it offers relatively limited insights for mitigating the backdoor attacks in VLMs.

3) The comparison baselines are not convinced. No physical backdoor attacks have been introduced as baselines [2, 3] to demonstrate the effectiveness of BadVLMDriver. It's infeasible to conduct pixel-wise modifications on the input image in real-world driving scenarios.

4）Visualization comparison of various poisoned images is missed.

5) The proposed BadVLMDriver is very vulnerable. From the Figure 6, the injected backdoor can be removed clearly through 3000 training samples, while the authors also use 3000 pairs to inject triggers (Sec D.1).


[1] Chen X, Liu C, Li B, et al. Targeted backdoor attacks on deep learning systems using data poisoning[J]. arXiv preprint arXiv:1712.05526, 2017.

[2] Liu Y, Ma X, Bailey J, et al. Reflection backdoor: A natural backdoor attack on deep neural networks[C]//Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part X 16. Springer International Publishing, 2020: 182-199.

[3] Han X, Xu G, Zhou Y, et al. Physical backdoor attacks to lane detection systems in autonomous driving[C]//Proceedings of the 30th ACM International Conference on Multimedia. 2022: 2957-2968.

### Questions
Can you add some discussion about the backdoor detection strategies for VLMs?

At what distance from the balloon does a car encounter backdoor attack? This is very important for drivers to make a decision for safe driving.

Can you report the efficiency of BadVLMDriver?

### Soundness
2

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
This paper presents a physical backdoor attack named BadVLMDriver, aimed at vision-large-language models (VLMs) used in autonomous driving. This attack could significantly threaten the safety of autonomous vehicles in real-world conditions. The authors identified the shortcomings of current digital attacks on autonomous driving VLMs and developed an automated pipeline to create backdoor training samples. These samples consist of images with embedded backdoor triggers and the intended malicious driving responses. Experimental results demonstrated that the proposed attack not only achieves a high success rate but also maintains a low false attack rate, with minimal impact on the model’s clean accuracy.

### Strengths
1. The paper is well-written, making the methodology of BadVLMDriver easy to follow. The experimental results are clearly presented and explained.
2. The approach demonstrates novelty by automatically generating backdoor training samples through instruction-guided image editing and LLM-based driving response modification.
3. Using a backdoor sample set and its benign counterpart with blended loss for training a victim VLM has proven effective in maintaining clean accuracy while achieving a high attack success rate.

### Weaknesses
1. Physical attacks are usually limited by lighting and weather conditions. While the paper discusses the impact of the trigger object’s distance, it may benefit from a more in-depth exploration of other dynamic factors affecting physical attacks.
2. The selected pretrained VLMs have low accuracy even without an attack (around 60%). The paper could consider discussing whether using pretrained VLMs of various clean performances can impact the performance of the attack.

### Questions
1. Will other sensors based technology, like lidar or radar, can help mitigate the threat via like forward collision warning? The author may provide more discussion on how other AD solution can help fix the issue. The finding in the paper shows that the VLM is not ready to take over AD yet.
2. Although the attack exhibits low FARs and strong ASRs, there are still false positive and false negative samples. Have you investigated why those samples cause false decisions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces BadVLMDriver, a novel backdoor attack targeting Vision-Large-Language Models (VLMs) in autonomous driving. Unlike traditional digital attacks on VLMs, BadVLMDriver employs physical objects—such as a red balloon—to manipulate VLMs into executing unsafe actions, like sudden acceleration. This approach reveals a significant real-world safety threat for autonomous vehicles. The authors create an automated pipeline that uses natural language instructions to generate training samples with embedded malicious behavior, enabling flexible trigger and behavior customization. Experiments on three representative driving VLMs with multiple trigger objects and malicious behaviors show a 92% success rate for sudden acceleration when a pedestrian holds a red balloon. These findings underscore the pressing need for robust defenses to protect against such vulnerabilities in autonomous driving systems, as BadVLMDriver demonstrates both the effectiveness and stealth of physically-induced backdoor attacks.

### Strengths
1. The paper addresses a highly relevant topic, focusing on safety risks in autonomous driving posed by Vision-Large-Language Models (VLMs). 
This is particularly timely given the increasing reliance on VLMs for complex decision-making in autonomous vehicles. 

2. The perspective in this work is novel, as it leverages real-world physical objects—such as a red balloon—to trigger malicious behaviors in autonomous vehicles. 
Unlike traditional pixel-level modifications in digital backdoor attacks, this physical approach is more practical and stealthy, posing a realistic threat to autonomous systems in uncontrolled environments. 

3. Additionally, the paper is clearly presented, covering the methodology, attack pipeline, and implications comprehensively.

### Weaknesses
1. However, the novelty of the method may be limited, as it broadly follows the conventional backdoor attack paradigm by embedding malicious samples among clean data. 
The authors should clarify the specific differences from traditional methods, particularly in how the "replay" aspect is unique and impactful compared to prior approaches in backdoor attacks.

2. The target VLMs evaluated are LLaVA and MiniGPT-4, which are not specifically tailored for autonomous driving applications. 
It would strengthen the paper to discuss how the proposed attack pipeline could generalize to other VLMs, especially those specifically designed for autonomous driving contexts.

3. The paper omits some recent backdoor defense strategies targeting VLMs, such as SSL-cleanse[1] and DECREE[2]. 
Including a discussion on how BadVLMDriver could potentially evade these defenses would add depth to the paper’s security analysis.

[1] Zheng et al, SSL-cleanse: Trojan detection and mitigation in self-supervised learning. ECCV'24

[2] Feng et al., Detecting Backdoors in Pre-trained Encoders. CVPR'23

### Questions
Please respond to each weakness mentioned above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces BadVLMDriver, the first physical backdoor attack targeting vision large language models (VLMs) in autonomous driving. Using everyday objects as triggers, it induces unsafe driving decisions like sudden acceleration. Unlike pixel-level digital attacks, this method activates via real-world physical objects and is highly stealthy. Experiments on three VLM models, five triggers, and two behaviors show up to 92% success. The study underscores the threat to autonomous driving and the urgent need for robust defenses.

### Strengths
1. Utilizing common objects as triggers enhances the real-world feasibility of the attack.
2. The experiments cover various triggers, models, and behaviors, demonstrating broad applicability.
3. The study highlights potential security risks in current autonomous driving systems using VLMs.

### Weaknesses
1. Insufficient experimental diversity: More types of autonomous driving VLMs should be evaluated to fully understand the applicability and limitations of the method.
2. Lack of analysis on defense effectiveness: There is insufficient discussion and validation of how existing defense mechanisms respond to this attack.
3. Unverified effectiveness in complex driving environments: The effectiveness of the attack in complex or dynamic driving scenarios has not been adequately assessed.

### Questions
1. How does the presence of environmental factors (e.g., lighting, weather conditions) affect the attack's success rate?
2. Can the methodology be adapted to identify or mitigate other types of vulnerabilities in VLMs?

### Soundness
3

### Presentation
3

### Contribution
3
