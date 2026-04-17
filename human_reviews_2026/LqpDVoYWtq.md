# Scorpio: Serving the Right Requests at the Right Time for Heterogeneous SLOs in LLM Inference

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4, 2

## Abstract
Existing Large Language Model (LLM) serving systems prioritize maximum throughput. They often neglect Service Level Objectives (SLOs) such as Time to First Token (TTFT) and Time Per Output Token (TPOT), which leads to suboptimal SLO attainment. This paper introduces SCORPIO, an SLO-oriented LLM serving system designed to maximize system goodput and SLO attainment for workloads with heterogeneous SLOs. Our core insight is to exploit SLO heterogeneity for adaptive scheduling across admission control, queue management, and batch selection. SCORPIO features a TTFT Guard, which employs least-deadline-first reordering and rejects unattainable requests, and a TPOT Guard, which utilizes a VBS-based admission control and a novel credit-based batching mechanism. Both guards are supported by a predictive module. Evaluations demonstrate that SCORPIO improves system goodput by up to 14.4X and SLO adherence by up to 46.5% compared to state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Scorpio is an SLO-oriented LLM serving system that targets heterogeneous TTFT and TPOT objectives by combining (i) least-deadline-first reordering and rejection of requests whose TTFT is unattainable (ii) a VBS-based (virtual batch size) admission controller plus credit-based batching guided by TPOT-relative Proportionality (TRP). A lightweight sequence-length predictor (fine-tuned OPT-125M classifier) and simple analytic models feed both guards. On Llama-3.1-8B and Gemma-2-27B with ShareGPT/LMSYS traces, the paper reports up to 14.4x higher goodput and 46.5% higher SLO adherence than baseline systems.

### Strengths
- This paper cleanly formulates heterogeneous SLOs and co-designs scheduling across admission, queueing, and batching with clear modular guards for TTFT and TPOT. 

 - The TPOT Guard is technically interesting: credit-based batching tied to TRP provides a principled way to throttle participation of looser-SLO requests while protecting tight SLOs. 

- The paper reports strong headline gains (goodput, SLO adherence) and uses illustrative figures that contrast throughput-first vs SLO-aware behavior.

### Weaknesses
- The system hinges on a trained length predictor running alongside the serving model, but the paper does not provide a sensitivity analysis of predictor placement or runtime contention under different loads and models.

- The design relies on TRP estimates and average-load assumptions; it’s not clear how well it holds up when length predictions are biased or when traffic is bursty and non-stationary (e.g., mixed short/long outputs). A more down-to-earth stress test, showing what breaks, by how much, and under which traffic patterns, would make the claims feel sturdier.

- In terms of SLO fairness trade-offs, TTFT Guard employs least-deadline-first and can reject unattainable requests; the paper should quantify user-visible drop rates or fairness impacts across classes (e.g., how often looser-SLO or long queries are deferred/rejected under realistic traffic).

### Questions
- Predictor overhead & placement. How does SLO adherence and goodput change if the OPT-125M predictor runs off-GPU (CPU) or on a separate GPU? Please report latency overheads and throughput loss when the predictor shares the inference GPU under high load. 

- What happens when the length predictor is systematically biased or distribution-shifted? Please show SLO attainment vs. prediction RMSE or Kendall’s Tau. 

- In terms of admission fairness, for the TTFT Guard’s rejection policy, what are rejection rates by SLO class and prompt length, and how do you bound starvation for long but high-priority requests? 

- VBS/TRP sensitivity: The VBS estimate treats admitted requests as fractional via TRP. How sensitive is adherence to TRP misestimation or delayed updates under bursty arrivals? Please include a stress test with heavy bursts.

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
The paper proposes SCORPIO, an LLM serving system designed to handle per-request SLOs for both TTFT and TPOT. The authors motivate this by claiming that different applications have heterogeneous SLO requirements, whereas existing systems typically enforce a single, global SLO tuple. SCORPIO processes requests differently across their lifecycle. For the prefill stage, it employs a Least Deadline First (LDF) scheduling strategy, proactively dropping requests deemed unattainable. For the decode stage, it uses a credit-based framework at each iteration to select a subset of pending requests for batching, dropping those at risk of violating their TPOT SLO. The results show that SCORPIO outperforms existing baselines in terms of goodput and SLO adherence.

### Strengths
- The paper addresses the challenge of SLO-aware LLM serving, which is a timely and important problem in the systems community.
- I like the way the TPOT estimator is designed. It gives an intuitive idea of how the decoding of a request changes the runtime of the batch.

### Weaknesses
My primary concerns with the paper in its current form are:
1. Missing Critical Details: Key aspects of the scheduling mechanism, particularly the handling of prefill/decode contention, are unclear, making the system's behavior and claims difficult to validate.
2. Limited Novelty: The core contribution appears to be request dropping for SLO adherence, a concept that has been explored in prior work.
3. Unfair Evaluation: The experimental comparison with baselines is misleading, as it compares a system that drops requests (SCORPIO) against best-effort systems that do not.

Justification:

While the problem is interesting, I believe the paper requires extensive revision to clarify its core mechanics, justify its novelty, and provide a fair evaluation to truly demonstrate the value of the proposed ideas.

1. My first concern is the framing of the problem. The introduction (line 38) claims that SLO requirements are "inherently heterogeneous" across applications, citing [1]. However, my reading of [1] suggests it discusses hardware heterogeneity (e.g., different GPUs), not heterogeneous request SLOs.

While I agree that different applications (e.g., summarization vs. chat) have different SLOs, it's unclear why they _must_ be served by a single, common scheduler. Why not run separate instances of existing SLO-aware solutions (e.g., DistServe, Sarathi-Serve, Mooncake) for each application? These solutions could provide strong guarantees with simpler mechanisms. The paper needs to better justify the premise of a single scheduler for mixed-SLO workloads over this simpler, multi-instance baseline.

2. The paper lacks crucial details, making it difficult to understand how the system functions. The most critical omission is the handling of prefill and decode contention.

- What happens when a new request arrives and requires prefill while other requests are in their decode phase?
- As I understand it, the system would add the new request to the LDF prefill queue. But when does the system schedule a prefill-specific batch? The paper seems to imply a binary choice between a prefill-only batch or a decode-only batch.
- If the system must switch to a prefill-only batch, the ongoing decode requests must wait. Given that prefill batches can take significantly longer than a single decode step (as shown in Figure 2a), this switching seems guaranteed to cause decoding requests to miss their TPOT SLOs. This is a fundamental challenge in continuous batching systems, and the paper does not address it.
- What happens to a request in the waiting queue ($W$ in Algorithm 1) if the system is overloaded and the condition on line 5 is never met? Does the credit mechanism prevent starvation, or will this request eventually be dropped for violating its TPOT SLO while waiting? As far as I could tell, this request will eventually be dropped by the TPOT guard.

3. The paper's primary advantage over baselines seems to be its ability to drop requests at various stages (pre-prefill, post-prefill) to preserve SLOs for other "meaningful" requests. If this is the main contribution, the novelty is questionable. The idea of admission control and request dropping to meet SLOs is not new; Mooncake and other systems already implement such mechanisms. The paper must clearly articulate what makes SCORPIO's scheduling and dropping policies novel compared to this existing body of work.

4. The evaluation seems unfair to both the baselines and the reader. SCORPIO achieves high goodput by dropping requests, but it is compared against best-effort systems (vLLM, S3) that attempt to serve all requests. This is an apples-to-oranges comparison. A fair comparison would require augmenting the baselines with a similar dropping/admission control mechanism. While on the topic of baselines, why not add SLO-specific baselines such DistServe or Sarathi Serve?

Furthermore, the results are not transparent. The paper must explicitly report the number of dropped requests for SCORPIO. An approach that meets SLOs for 100% of the requests it processes but drops 80% of incoming requests is not superior to a system that serves all requests while missing SLOs on a small fraction. The reader cannot make this judgment without the drop-rate data.

Minor Comments:
- Why optimize for expected goodput or adherence (equation 3)? SLO-critical applications typically focus on tail latency (e.g., p90 or p99), as is standard in related serving systems.
- The paper claims (line 139) SCORPIO maximizes goodput and SLO adherence. This is inaccurate; Equation 3 optimizes the expectation of one of these metrics.
- The purpose of the sequence length predictor is unclear. Subsection 3.2 (lines 155-161) describes how it was built, but not what its purpose is or how it is used within the scheduling framework.
- The paper states (line 669) input prompts were truncated to 2048 tokens for the predictor. Is the predictor only trained on these lengths? How does the system handle prompts that are much longer at inference time?
- What is the formal definition of Inter-Token Latency (ITL)? The paper uses this term without defining it or clarifying how it differs from TPOT.
- What is the truncation policy for decode batch selection? What happens if a large number of requests have credits above the threshold (e.g., >1)?
- Why is the predictor's inference time not added to the request's latency? Even if “negligible” (line 427), this should be quantified and included for completeness.

[1] Patke, Archit, Dhemath Reddy, Saurabh Jha, Haoran Qiu, Christian Pinto, Chandra Narayanaswami, Zbigniew Kalbarczyk, and Ravishankar Iyer. "Queue management for slo-oriented large language model serving." In Proceedings of the 2024 ACM Symposium on Cloud Computing, pp. 18-35. 2024.

### Questions
1. Does the denominator $\mathcal{R}$ in the SLO adherence definition (equation 2) include requests that the system drops as unattainable before processing? If not, the adherence metric is inflated.
2. What is the total number of requests dropped by TTFT guard (pre-prefill) and the total number of requests rejected by the TPOT guard (post-prefill)?

### Soundness
2

### Presentation
2

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
This paper introduces a serving system for maximizing goodput and SLO attainment, especially for heterogeneous SLOs, featuring a TTFT Guard and a TPOT Guard. This system significantly improves SLO attainment and goodput.

### Strengths
1. The application is practical, highly emphasizing the actual serving scenarios.
2. The formulation is novel, which makes the algorithm intuitive.
3. The evaluation is well-conducted.

### Weaknesses
1. Lack of comparison of related work,  AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding; SLOs-Serve: Optimized Serving of Multi-SLO LLMs; SOLA: Optimizing SLO Attainment for Large Language Model Serving with State-Aware Scheduling;

2. The scale of experiments is limited. To evaluate SLO attainment in real settings, existing techniques like PD disaggregation, EPLB are very important, while the evaluated model, i.e., Llama-8B and Gemma-27B, is too small to be valuable to be served in the cloud, while at most 4 GPUs are too limited. 

3.  Context Length and generation length are very important in SLO-driven systems. However, I do not see a discussion in different workloads with various context lengths and generation lengths.

### Questions
Already in Weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes an SLO-oriented method that maximizes system throughput and SLO attainment for heterogeneous workloads in LLM serving. The method considers SLO heterogeneity for adaptive scheduling across admission control, queue management, and batch selection based on SLO parameters.

### Strengths
1. The idea of exploiting heterogeneous SLOs for dynamic scheduling is conceptually reasonable.

2. The proposed Credit-based Batching and VBS-based Admission Control are novel.

3. The evaluations are comprehensive.

### Weaknesses
1. The proposed mechanisms are heuristic in nature and lack theoretical grounding or provable guarantees.

2. There is no formal analysis on convergence, fairness, or stability of the proposed credit-based batching mechanism.

3. The design assumes that each request comes with explicit SLO constraints, which is unrealistic in most real-world deployments, as users rarely provide SLO metadata.

4. The rejection mechanism risks dropping important user queries, which may not be acceptable in production.

5. Similar ideas have been explored in literature.

6. The paper is not very well written and requires heavy proof-reading

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
3
