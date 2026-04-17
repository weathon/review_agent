# Talk Until You Burn Out: Escalating 3D-LLM Overgeneration via Semantic Manipulation

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
The rise of 3D large language models (3D-LLMs) has unlocked new potential in multimodal reasoning over unstructured 3D data, powering applications such as robotics and autonomous driving. However, these models also introduce new security risks, particularly during the inference-time computation. In this work, we present \textbf{Exhaust3D}, the first targeted energy-oriented adversarial framework against 3D-LLMs. Exhaust3D performs a \textbf{resource exhaustion attack} by injecting imperceptible yet strategically structured semantic perturbations into 3D point clouds, causing the model to overgenerate outputs and inflate inference latency. Specifically, we design two key components: (1) a \textit{semantic-aware adversarial manipulation strategy} that leverages internal model representations to selectively perturb semantically critical point regions while preserving geometric structure, and (2) a \textit{trajectory disruption mechanism} that maintains high-entropy token predictions to prolong auto-regressive decoding and induce verbose outputs. Experiments on widely-used 3D-LLM benchmarks show that Exhaust3D increases decoding steps and energy consumption by up to \textbf{4$\times$} with negligible degradation in functional performance. These results expose a previously underestimated vulnerability of 3D-LLMs to resource exhaustion attacks, highlighting the urgent need for energy-aware robustness in future multimodal foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents a novel resource exhaustion attack on the 3D-LLMs by injecting imperceptible yet strategically structured semantic perturbations into 3D point clouds. It has two specific components: 1) Semantic-aware adversarial manipulation that leverages internal model representations to selectively perturb semantically critical point regions while preserving geometric structure. 2) a trajectory disruption mechanism that forces high entropy predictions to prolong the end of sequence token generation and predict verbose outputs.

### Strengths
1. The problem studied is novel and important, as it brings attention to the resource exhaustion attack on the 3D LLMs.

2. The attack is quite simple to develop, hence it becomes more important.

3. Experimental results further back the importance of the work.

### Weaknesses
The major issues with this work are:

1. The performance metrics are not reported, which are claimed to be good.

2. Table 1 results seem ambiguous, as the results on MiniGPT shows a latency of 6.79 and 6.71, while other methods have higher latency, still the paper reports their results as best. To the best of my understanding, higher latency is what authors are aiming to achieve.

3. There is no reproducibility statement in the paper, and no source code or any statement regarding that is made.

4. The paper can benefit by adding additional explanations on why persistence loss is not enough alone and why it requires other losses. While there is some discussion about these results, still authors are not very clear why the persistence loss or dispersion loss is not sufficient alone. By my understanding, authors want to achieve a delay in the end-of-sequence token, and that could be easily achieved using the persistence loss, while some loss in performance (that is why reporting performance is important), I am not sure how the authors are getting significantly smaller sequence length while only optimizing the persistence loss.

5. The compared baselines are really narrow, as the authors can try to extend the existing baselines towards 3D LLMs and compare with them, which can prove the importance of their method.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates a new type of vulnerability in 3D large language models (3D-LLMs) — adversarial perturbations that intentionally increase inference time and energy consumption. The authors propose a **semantic-aware perturbation** combined with a **trajectory-intervention mechanism** to induce over-generation while maintaining semantic and visual consistency. Experiments on two datasets and four representative 3D-LLMs demonstrate significant increases in output length and latency, supported by extensive ablation analyses.

### Strengths
* **Important problem.** This paper identifies an important yet underexplored risk — increasing 3D-LLM inference time through adversarial manipulation of point clouds, which can raise operational costs for online users.  
* **Well-structured method.** The proposed semantic-aware perturbation and trajectory-intervention mechanism directly target the problem and are clearly written and easy to understand.  
* **Reasonably broad experiments.** The paper evaluates two datasets and four representative 3D-LLMs, and conducts extensive ablation studies and hyperparameter analyses.

### Weaknesses
* **Limited real-world applicability due to white-box assumption.**
 The method and evaluation rely on white-box access, which severely limits the practical impact of the proposed attack, since real-world 3D-LLMs are often closed-source.  
* **Inappropriate baselines.**
 The selected baselines (Gaussian Noise and Random Drop) are unrelated to energy-exhaustion behavior. More relevant baselines, such as those targeting delayed EOS or increased decoding entropy, should be added.  
* **Lack of defense evaluation.**
 The paper does not evaluate any defense mechanisms, making it difficult to assess the robustness of the proposed method.  
* **Unsupported functional-quality claim.**
 The abstract claims that model functionality is nearly unaffected, but there is no rigorous experimental evidence or analysis to substantiate this claim.

### Questions
The paper’s main limitation lies in its **white-box design** and **unrelated baselines**, which significantly restrict its **practical relevance and generalizability** despite an interesting problem formulation.

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
The paper proposes Exhast3D, a resource exhaustion attack framework on 3D large language models (3D-LLMs). The attack injects imperceptible perturbations into 3D point clouds to cause models to overgenerate long outputs, increasing inference-time energy and latency. Exhaust3D includes a semantic-aware adversarial manipulation strategy and a trajectory disruption mechanism. Experiments on the Objaverse and ModelNet40 datasets show up to 6.45× longer outputs and 6.12× higher energy use for various 3D-LLMs.

### Strengths
1. The paper proposes the first attack framework for resource exhaustion attacks on 3D LLMs.
2. The semantic-aware masking and dual-loss formulation are well justified, with clear equations and ablation studies.
3. Evaluation across four models and two datasets demonstrates the effectiveness of the proposed method. 
4. The paper is well written and easy to follow.

### Weaknesses
1. The comparison is only against random noise and dropout. No semantic or gradient-based adversarial baselines (e.g., FGSM, PGD, or prior efficiency attacks like Sponge or NICGSlowDown) are adapted for 3D input.
2. The impact of the threat model is somewhat limited. While it is shown that Exhast3D can successfully increase the output of 3D LLMs, it is unclear what the practical consequences are. In addition, the threat model also assumes white-box access to the model weights, which can be infeasible in real-world applications. 
3. The paper should also consider the attack performance under defenses, for example, random resampling of 3D points or applying Gaussian filtering to remove the adversarial noise.

### Questions
1. Adding a few simple baselines can be helpful to understand the effectiveness of the proposed method.
2. Can this method work in the gray-box or black-box settings?
3. Line 116, reference missing: LEO (?).
4. How well does the method perform under defenses?
5. Line 225-226, it is unclear why a larger l2 norm in the hidden state indicates higher token importance.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents Exhaust3D, a novel framework for adversarial attacks that target 3D Large Language Models. The core idea is quite clever—it crafts tiny, almost invisible perturbations on 3D point clouds. These manipulations trick the model's internal workings, causing it to generate extremely long and verbose outputs. The result is a massive spike in inference time and energy consumption, all while the actual factual quality of the answer remains largely intact. This is a very timely investigation into the computational weak spots of the latest multimodal AI systems.

### Strengths
First off, the novelty is strong.  Systematically pulling off a resource exhaustion attack purely through the 3D modality, which is a smart focus given how new and computationally heavy these models are. Secondly, the attack itself is well-designed. The two-part strategy—using semantic-aware masking to find critical points and then a trajectory disruption mechanism—is elegant and makes a lot of sense. Using PCGrad to smooth out gradient conflicts was a nice technical detail. Finally, the evaluation is thorough. The results are compelling across multiple models and datasets, and the ablation studies do a great job of validating why the authors made their specific design choices.

### Weaknesses
A couple of things gave me pause, though. While the perturbations are technically imperceptible, the paper doesn't fully address how stealthy this attack would be in a real-world application. It seems a simple system check on output length could easily detect and stop these rambling responses, which would neutralize the threat. Additionally, while the work does an excellent job of exposing a vulnerability, it would be strengthened by even a brief discussion on potential defensive strategies, which would provide a more complete picture.

### Questions
I'm curious about the realistic scenarios where this attack would be most potent. Are you thinking more about cloud-based services where latency equals cost, or embedded systems where a resource drain could cause a physical failure? Also, the attack worked best on PointLLM, but GreenPLM was more resilient. Any thoughts on the architectural reasons for that? Understanding that could offer great insights for building tougher models. Lastly, your evaluation focused on captioning tasks. Do you have any intuition on how this would play out in more critical tasks like navigation or reasoning, where prolonged processing could have real-world consequences?

### Soundness
3

### Presentation
3

### Contribution
2
