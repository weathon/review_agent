# NanoVLA: Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Vision-language-action (VLA) models have significantly advanced robotic manipulation by integrating visual encoders, large language models (LLMs)/visual-language models (VLMs), and action decoders into a unified architecture. However, their deployment on resource-constrained edge devices, such as mobile robots or embedded systems (e.g., Jetson Orin Nano), remains challenging due to high computational demands, especially in real-world scenarios where power, latency, and computational resources are critical. To close this gap, we introduce Nano-scale Vision-Language Action (NanoVLA), a family of lightweight VLA architectures that achieve high performance with minimal resources. Our core innovations include: (1) visual-language decoupling that moves traditional early vision and language inputs fusion in VLM to late stage, achieving better performance while enabling caching and reduce inference overhead and latency; (2) long-short action chunking to ensure smooth, coherent multi-step planning without sacrificing real-time responsiveness; (3) dynamic routing that adaptively assigns lightweight or heavy backbones based on task complexity, further optimizing inference efficiency. Experimental results on several benchmarks, as well as real-world deployments, demonstrate that NanoVLA achieves up to 52x faster inference on edge devices compared to previous state-of-the-art (SOTA) VLA models, with 98% fewer parameters while maintaining or surpassing their task accuracy and generalization. Ablation studies confirm that our decoupling strategy preserves cross-task transferability, and the routing module enhances cost-performance trade-offs, enabling practical, high-precision robotic manipulation on resource-constrained hardware.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces NanoVLA, which enables VLA to run efficiently. The authors propose to do save computations by several proposed techniques like decoupling vision and language models, action chunking and dynamic routing. The results show that with fewer computations, it can achieve good performance and efficiency in simulation and real-world scenarios.

### Strengths
This paper tries to tackle an important issue in robotics of running efficiency. The method contains a lot of careful designs.

The paper shows good results on performance and speed on LIBERO given the number of parameters tuned.

### Weaknesses
More discussion on the advantages of the late fusion technique is needed.

Comparison with other baselines on improving VLA efficiency is needed.

See the questions below for other minor points.

### Questions
(1)	For the late fusion part, I’m curious about the comparison between early fusion with a freeze VLM (used in methods like HiRT) and late fusion used in the proposed method. What is the advantage of late fusion? From my perspective, I think it offers a subtle speed gain at the cost of worse language and vision alignment. Is there any ablation study to show the efficiency of this design?

(2)	Currently, the author only compare with SmolVLA as a baseline on efficient VLA on LIBERO. There are a lot of method trying to improve the VLA efficiency and more baselines and comparison are needed. Previous methods like RT-1 also adopts the similar method.

(3)	Can the routing model generalize to OOD settings? I’m curious about the accuracy of win probabilities on OOD settings.

(4)	As for the speed up (52X) described in L375, how did you calculate the LLAMA2 FPS? Do you take the image into the timing?

(5)	Misc. The right figure of figure 1 is hard to see. The legend is very small the difference between lines is hard to tell.

### Soundness
3

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
The paper proposes NanoVLA, a lightweight visual-language architecture (VLA) optimized for deploying visual-language fusion on c. NanoVLA adopts late-stage modality fusion to avoid excessive cross-attention computations, while its dynamic routing allows the model to adaptively allocate computing resources for complex, reasoning-intensive tasks. Experiments conducted in the Libero simulator and real-world scenarios show that NanoVLA outperforms baseline methods in performance.

### Strengths
- Employed two strategies to optimize the long latency issue in VLA,  save 62% inference time compared to the traditional VLA approach.
- Detailed latency and performance analysis.

### Weaknesses
- The model structure is overly simplistic and similar to exist work. Both Diffusion Policy[1] and Scaling-Up Diffusion Policy[2] utilize lightweight transformer to late integrate various different modalities in an End-to-end training manner. NanoVLA and these method exactly has very low latency, but it loses the language-vision alignment capability that VLMs have acquired through extensive training, which greatly affects the generalization ability of VLA.
- The experiments on real world are overly simplistic, with basically only one object to be operated in the scene. This hardly requires model generalization, and only overfitting to a single task is necessary. For overly simple tasks, semantic alignment between language and vision is not required, and only lightweight expert networks need to be trained separately for each task.
- The performance of the model on LIBERO is weak. In fact, the widely used LIBERO pi0 result is 94.2% [3], and there are some issues with the LeRobot code that can lead to even lower reproduced results. However, when parallel decoding (PD) and action chunking (AC) are added to Openvla, it can also achieve 94.5% [3], both far exceeding NanoVLA. In fact, the LIBERO task does not require semantic generalization, and simply using a diffusion policy can achieve 72.4% (close to NanoVLA-S)

[1] Chi, Cheng, et al. "Diffusion policy: Visuomotor policy learning via action diffusion." The International Journal of Robotics Research 44.10-11 (2025): 1684-1704.

[2] Ha, Huy, Pete Florence, and Shuran Song. "Scaling up and distilling down: Language-guided robot skill acquisition." Conference on Robot Learning. PMLR, 2023.

[3] Kim, Moo Jin, Chelsea Finn, and Percy Liang. "Fine-tuning vision-language-action models: Optimizing speed and success." arXiv preprint arXiv:2502.19645 (2025).

### Questions
1. Can the model be jointly trained with additional visual-language data to enhance the semantic generalization of VLA without relying on full VLM models? It is recommended to add this training process to the work and submit it to other conferences.
2. Will a ViT-based visual encoder yield better performance than ResNet18? Is ResNet18 more suitable as a visual encoder for single-task policy models?
3. How does the model compare with other VLA methods (e.g., HiRT[1]) that adopt fast-slow systems for inference acceleration?
4. How does the model perform in other simulation environments (Calvin[2], Robotwin[3]) and more complex real-world tasks—especially when manipulating unfamiliar objects amid interference from other objects?

[1] Zhang, Jianke, et al. "Hirt: Enhancing robotic control with hierarchical robot transformers." arXiv preprint arXiv:2410.05273 (2024).

[2] Mees, Oier, et al. "Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks." IEEE Robotics and Automation Letters 7.3 (2022): 7327-7334.

[3] Mu, Yao, et al. "Robotwin: Dual-arm robot benchmark with generative digital twins (early version)." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes NanoVLA, a lightweight design of vision-language-action models that is compatible with edge devices like Jetson Orin Nano. The method combines three core components: (1) vision-language decoupling (late fusion of representations of multiple modalities), (2) "long-short action chunking" (also known as receding horizon control), and (3) dynamic routing with Bayesian uncertainty that allows switching between small and large language model backbones depending on task difficulty. In experimental evaluations in LIBERO, NanoVLA achieves a 84.1% average success rate, outperforming prior methods such as OpenVLA, SmolVLA, and $\pi_0$ despite using significantly fewer parameters and running much faster.

### Strengths
* The problem of deploying VLA policies on edge devices is well motivated and addresses a real practical constraint for many robotics practitioners who do not have access to server-grade hardware for model deployment.
* The experiments include both simulated and real-world robot tasks, validating the effectiveness of NanoVLA across various tasks and domains.
* Simulated evaluations in LIBERO show superior performance of NanoVLA compared to various prior methods, including OpenVLA, $\pi_0$, SmolVLA, SpatialVLA, and TraceVLA. Real-world robot evaluations assess generalization to unseen lighting conditions and objects, and the proposed NanoVLA method obtains superior performance compared to prior methods such as $\pi_0$, OpenVLA, SmolVLA.

### Weaknesses
* LIBERO experimental results do not include several state-of-the-art prior works from early 2025, including OpenVLA-OFT (97.1% success rate - RSS 2025) and UniVLA (95.2% success rate - RSS 2025). These works use earlier fusion of language and vision representations and obtain substantially higher performance in LIBERO than the proposed NanoVLA (84.1% success rate).
* The authors argue that late fusion of the modalities is a superior approach, but analysis of an early fusion alternative with the same architectural components/backbones is not provided, so the argument is not convincing and empirical evidence is needed to support the claim. This is especially the case given more recent VLA works that use early fusion to obtain higher performance in LIBERO and perform more complicated real-world robotics tasks (e.g. bimanual manipulation on a real ALOHA robot). I question the scalability of a late fusion approach where a small action decoder head is tasked to learn how to properly fuse the representations. Would NanoVLA be able to handle tasks that require stronger vision-language grounding, such as picking any object amidst clutter (where the target object is specified in the user's language input)?
* "Long-short action chunking (LSAC)" is presented as a novel contribution but is a common technique employed by various prior works, including Diffusion Policy which uses receding horizon control during policy execution (Chi et al., RSS 2023).
* Related to the prior point, there is some lack of technical novelty, as the method combines existing ideas from prior works (late fusion, action chunking/receding horizon control, model routing).
* The real-world tasks are fairly narrow as they are almost all variants of simple, short-horizon single-arm pick-and-place onto a large wooden board (plus a towel dragging task which is only slightly different). Whether the method generalizes to more challenging manipulation tasks (e.g., high-precision or long-horizon manipulation with bimanual robot) is unclear and positive results there would strengthen the paper.
* The paper seems slightly rushed and could use a bit more polish, as there are several grammatical mistakes or minor errors throughout the paper. For example, line 323 says "OpenVLA-L", line 820 says "LeRobot was not included the the pre-training
dataset", Figure 5(a)'s method labels on the x-axis are confusing since they are not well spaced apart and there is no "Nano", etc.

### Questions
* How many trials are needed to train an accurate router?
* Why does NanoVLA-L underperform compared to NanoVLA-R?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces NanoVLA, a family of light-weight vision-language-action (VLA) policies  for deployment on edge hardware (e.g., Jetson Orin Nano). The core idea is to rethink inference rather than merely shrink parameters, via three components: (i) late, decoupled fusion of vision and language with caching of instruction features to eliminate redundant cross-modal computation; (ii) long-short action chunking (LSAC), which plans long sequences but executes short sequences; and (iii) dynamic routing that selects a small or large language backbone per task. Together these aim to spend compute only where it matters and keep latency low while preserving generalization.

Empirically, NanoVLA matches or surpasses larger baselines on LIBERO suites and shows strong real-robot results on a LeRobot setup, while achieving dramatic throughput gains on Orin Nano (e.g., 52× higher FPS than OpenVLA under the authors’ setup). The paper includes ablations for LSAC stability and for LLM-feature caching, and formalizes the router with a Beta–Binomial model and Monte-Carlo estimation of pairwise win probabilities.

### Strengths
The paper is well-structured with clear motivation. It demonstrates on-edge VLA control with improved success and major latency gains, addressing a potential deployment blocker for household and mobile manipulation. Reported numbers are compelling: SOTA-competitive LIBERO performance with far fewer parameters, strong LeRobot success rates, and notably higher FPS on Orin Nano.

### Weaknesses
**LSAC**: While LSAC is effective, the idea of predicting longer sequences and executing shorter sub-segments with periodic replans has been used in Diffusion Policy and Pi0; the paper could better articulate what is novel.

**Routing signal is text-only.** The router is trained as a text-conditioned comparator over models. This assumes that instruction phrasing correlates tightly with task difficulty. However, “pick up the banana” may range from trivial on a clear table to hard in a fruit pile that requires recognition, occlusion handling, and non-prehensile pushes—precisely where a stronger visual backbone is needed. A vision-augmented (or vision+text) router could improve routing fidelity in more realistic scenarios.

**Late fusion vs. early fusion trade-offs.** Decoupling enables caching, but the paper does not directly quantify any loss (or gain) in instruction-following and generalization versus a comparable early-fusion counterpart at equal parameter/compute. A head-to-head ablation—same encoders/action head, with and without early cross-modal attention—reporting instruction sensitivity and OOD generalization would strengthen the claim that late fusion “maintains competitive performance” while improving efficiency.

**Baselines and finetuning.** It is not explicit whether all baselines were finetuned on the same real-robot data mixture and with matched action horizons/chunk sizes. Without this, real-robot gains could partially reflect data or horizon choices rather than architecture.

### Questions
First, could you include a direct comparison of late fusion vs. early fusion under matched encoders/params/compute, evaluating instruction following and generalization? This would isolate whether late fusion’s efficiency comes with any semantic trade-offs, beyond the shown caching benefit.

Second, on routing, can you quantify misrouting rates and failure modes, and evaluate a vision-augmented router versus text-only? It would be useful to see thresholds (\tau) vs. success/computation for cluttered vs. clean scenes, and calibration under few-shot task statistics. The Beta–Binomial/MCB formulation is elegant—an ablation versus simpler SR-only routing under distribution shift would be convincing.

Finally, could you clarify baseline training details on real-robot experiments—were all baselines finetuned on the exact same 50-demo/task sets with identical horizons/action parameterization? And could you add a small study on unfreezing the vision encoder to demonstrate whether freezing is always preferable?

### Soundness
3

### Presentation
3

### Contribution
2
