# Resource Consumption Red-Teaming for Large Vision-Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Resource Consumption Attacks (RCAs) have emerged as a significant threat to the deployment of Large Language Models (LLMs).
With the integration of vision modalities, additional attack vectors exacerbate the risk of RCAs in large vision-language models (LVLMs). 
However, existing red-teaming studies have mainly overlooked visual inputs as a potential attack surface, resulting in insufficient mitigation strategies against RCAs in LVLMs.
To address this gap, we propose RECITE (Resource Consumption Red-Teaming for LVLMs), the first approach for exploiting visual modalities to trigger unbounded RCAs red-teaming.
First, we present Vision Guided Optimization, a fine-grained pixel-level optimization to obtain Output Recall Objective adversarial perturbations, which can induce repeating output.
Then, we inject the perturbations into visual inputs, triggering unbounded generations to achieve the goal of RCAs.
Empirical results demonstrate that RECITE increases service response latency by over 26×$\uparrow$, resulting in an additional 20\% increase in GPU utilization and memory consumption. 
Our study reveals security vulnerabilities in LVLMs and establishes a red-teaming framework that can facilitate the development of future defenses against RCAs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces RECITE, a novel method for resource consumption attacks (RCA) on vision-language models (VLMs), leveraging Vision Guided Optimization. The study empirically demonstrates that RECITE can increase response latency by over 26 times and enhance GPU utilization by 20%. The experiments are extensive and well-conducted.

### Strengths
1. The paper proposed first method for exploiting visual input on RCAs.
2. The paper is well-structured and it is easy to follow.
3. The authors provide a comprehensive analysis, evaluating the attack's effectiveness across 3 VLMs.
4. The paper also include potential defense strategy, further enhancing their work.

### Weaknesses
1. The paper lacks experiments on transferability. The authors should evaluate the transferability of the optimized visual inputs, both across different white-box models and by using inputs optimized on a white-box model to attack a black-box model.
2. The paper has mentioned defense strategy. If diffusion purification or random smoothing were applied directly to the optimized inputs, would they have any defensive effect?

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates visual-modality-driven resource consumption attacks (RCAs) on large vision-language models (LVLMs). The authors propose RECITE, a red-teaming framework that perturbs input images using an Output Recall Objective and Vision-Guided Optimization. The resulting perturbations are nearly imperceptible but cause the model to generate extremely long or looping outputs, significantly increasing GPU usage and latency. Experiments across multiple LVLMs demonstrate the feasibility of the attack and suggest preliminary defense strategies.

### Strengths
- The paper introduces a new attack surface—visual inputs causing resource consumption—which has not been systematically explored before. This problem is both novel and practically relevant.

- The proposed RECITE framework is simple yet effective, providing a structured way to red-team LVLMs for resource-related vulnerabilities.

- The experimental validation is extensive, involving multiple models and metrics (output length, GPU utilization, latency, memory). The results strongly support the main claim.

### Weaknesses
- The theoretical explanation of why visual perturbations cause looping behavior is insufficient. The paper would benefit from a formal analysis of the model’s stopping dynamics, such as EOS logit suppression or entropy evolution.

- The Output Recall Objective is largely heuristic. There is no ablation comparing it to simpler baselines such as minimizing the EOS token probability or tuning length penalties, which makes it unclear how necessary this specific objective is.

- The defense section is underdeveloped. The proposed sliding-window penalty lacks depth, and there is no quantitative analysis of how it affects model accuracy or normal captioning tasks.

- The evaluation scope is narrow, limited to open-source LVLMs. Testing on closed-source or API-based models (e.g., Gemini or GPT-4V) would make the results more compelling.

- The generality of the attack is uncertain. The paper does not explore transferability across models or tasks, nor does it test whether a single universal perturbation could generalize to multiple models.

### Questions
Have the authors considered testing on black-box, closed-source models to validate the practical impact in deployed systems?

### Soundness
3

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
3

### Summary
This paper proposes RECITE (Resource Consumption Red-Teaming for LVLMs), a red-teaming framework that exploits visual inputs to trigger unbounded resource consumption attacks (RCAs) in large vision-language models (LVLMs). The core idea is to craft imperceptible adversarial perturbations that induce the model to enter a repetitive generation loop, e.g., outputting “cup cup cup...” indefinitely, thereby exhausting GPU memory and latency. Experimental results demonstrate that RECITE increases service response latency by over 26×.

### Strengths
1. The paper demonstrates that visual inputs alone can reliably trigger severe resource consumption attacks (RCAs) in large vision-language models (LVLMs).
2. The authors conduct extensive experiments across seven LVLMs from three major families (LLaVA, Qwen, BLIP), using diverse metrics (Output Time GPU Utilization Memory Usage) and multiple attack configurations.
3. The method section is technically thorough, with precise definitions of the Output Recall Objective and Vision Guided Optimization.

### Weaknesses
1. The claim that this is the “first” vision-based resource consumption red-teaming for LVLMs appears overstated. Prior work such as Gao et al. (ICLR 2024) [1] also leverages visual inputs to induce high latency/energy consumption in LVLMs. The paper should clarify how RE-CITE differs conceptually and technically from such approaches.

2. Figure 1, which depicts the RE-CITE pipeline, lacks sufficient clarity. Key components—such as visual encoding, embedding projection, and the iterative perturbation update process—are not well differentiated. 

3. The core components demonstrates limited technical novelty. The Output Recall Objective seems to only define repetitive generation patterns and  the Vision Guided Optimization appear to be straightforward adaptations of the existing GCG method.

4. The evaluation only compares against GCG-RCAs. Given that [1] (Gao et al., ICLR 2024) also targets visual RCAs, it should be included as a baseline to better position RE-CITE’s relative effectiveness and novelty.

5. The paper does not assess whether perturbed images generated for one model transfer to others (e.g., perturbations optimized on LLaVA tested on Qwen). Such transferability is critical and should be evaluated.

6.  It remains unclear whether the generated perturbations are effective against deployed LVLM services (e.g., GPT4-V, Claude, or Qwen API)

[1] Gao K, Bai Y, Gu J, et al. Inducing High Energy-Latency of Large Vision-Language Models with Verbose Images[C]//The Twelfth International Conference on Learning Representations.

### Questions
Please address the weakness above.

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
The paper proposes a denial-of-service (DoS) attack on vision-language models (VLMs). The attack identifies input samples that cause the model to consume an unusually large number of tokens, thereby degrading its efficiency. Such samples are discovered through a perturbation injection method guided by a newly proposed loss function.

### Strengths
- The problem studied is timely.

### Weaknesses
- The paper positions the work as a red-teaming effort, but the proposed method is more accurately described as a specific attack. In general, red-teaming involves systematically identifying a range of vulnerabilities, including those without concrete exploits, and typically provides comprehensive analysis and actionable recommendations. These broader aspects are missing from the current paper.

- Figure 3 measures semantic consistency, but its relevance to a denial-of-service and red-teaming  setting is not clearly justified. 

- Although the attack is new in its application to VLMs, it largely follows existing adversarial attack paradigms. In my opinion,  the level of methodological novelty may not be sufficient for this conference..

### Questions
- Explain why the metric on "semantic consistency" and "feature similarity" are relevant.

- Including of actionable recommendations.

### Soundness
3

### Presentation
3

### Contribution
2
