# When We Don’t See the Same Picture: Aligning Agents with Divergent Visual Spaces

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 6, 2

## Abstract
With the rapid rise of agentic systems, interaction and collaboration between agents has become a central challenge in multimodal large language model (MLLM) research. 
In this work, we study a unique form of interaction where two agents hold divergent visual spaces. To investigate this, we adopt image reference games as a testbed and design experiments with five distinct perceptual distortions inspired by real human visual impairments (e.g., cataract, color blindness). We evaluate in two settings, online and offline, with four post-training algorithms: SFT, DPO (offline) and KTO, GRPO (online), providing the first systematic study of alignment under divergent visual perceptions. Our results reveal that (i) offline adaptation can provide strong improvements, with DPO consistently outperforming other methods when supported by high-quality preference data; (ii) among online adaptation methods, KTO yields strongest average gains across datasets and (iii) qualitative analysis shows that adapted agents shift their descriptions toward perceptual features accessible to their conversation partners. Taken together, these findings highlight that offline methods are the preferable solution when supervision is available
while online approaches serve as a complementary strategy for dynamic settings where distortions or partner characteristics are unknown in advance. We release code and preference datasets to support future research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses an important and timely problem: how AI agents can communicate effectively when they perceive the world differently. Using an image reference game with five perceptual distortions inspired by real human visual impairments, the authors compare offline (SFT, DPO) and online (KTO, GRPO) adaptation methods. The core motivation—enabling better human-AI collaboration under perceptual asymmetries—is compelling and has clear humanitarian implications for accessibility. The experimental framework is well-designed, and the preference dataset construction is creative.

### Strengths
- Novel and socially relevant problem formulation. The scenario—communication between agents with mismatched visual perception—is both conceptually interesting and practically important, especially for accessibility and assistive AI.

- Systematic experimental framework. The paper compares both offline and online post-training paradigms under a controlled setup, with extensive quantitative and qualitative analyses.

- Strong execution and clarity. The distortion modeling (cataract, AMD, color blindness, etc.) is well-motivated and grounded in real-world perceptual conditions.

- Potential for human-AI applications, such as blind assistance, human–robot collaboration, and adaptive visual reasoning.

### Weaknesses
- Limited conceptual depth in algorithmic comparison. The paper mainly benchmarks standard SFT/DPO/KTO/GRPO methods without offering new algorithmic contributions or deeper theoretical insight into why some alignments succeed or fail.

- Shallow treatment of RL-based adaptation. While four algorithms are compared, the discussion of their distinct behaviors or learning dynamics is limited.

- Cross-agent results (e.g., Qwen2.5-VL vs. InternVL3.5) show small or inconsistent gains but lack detailed error analysis.

### Questions
- I am very interested in how different top-tier multimodal LLMs (e.g., GPT-5, Gemini 2.5 Pro) would perform as speakers in this framework. Could the authors test or discuss such systems? This would greatly inform model choice for assistive applications.

- Have the authors considered deploying the speaker–listener setup in realistic blind-assistance contexts (e.g., identifying street objects, reading signs)? Even small-scale trials could strongly strengthen the paper’s societal impact.

- Could the authors explore human-in-the-loop experiments, where human participants act as listeners (receiving speech-based descriptions)? Given that humans have limited auditory memory span, such tests could reveal important differences between human and model listeners.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a variant of the image reference game where a speaker agent with access to undistorted images must communicate with a listener agent who only sees distorted versions. The authors simulate five types of visual distortions inspired by human visual impairments (cataract, AMD, color blindness, tunnel vision, retinal detachment) and evaluate four post-training adaptation methods: SFT and DPO (offline) and KTO and GRPO (online). Experiments on CLEVR and CUB datasets show that these methods can improve communication success, with KTO generally providing the most consistent gains.

### Strengths
1. The task of communication under divergent visual perception is practical and relevant, with clear applications to accessibility and multi-agent systems with heterogeneous sensors.
2. The paper thoroughly evaluates multiple methods across two datasets, five distortion types, and includes cross-dataset transfer experiments and tests with different listener models.
3. The paper is well-structured with effective visualizations (particularly Figure 1 and 2) that clearly convey the task setup and qualitative results.

### Weaknesses
1. The paper fundamentally presents a benchmarking exercise without extracting meaningful scientific understanding. The core findings (offline methods work with good data, online methods are more stable, adaptation helps with mismatches) are trivial and expected.
2. The paper provides no formal analysis of communication under perceptual constraints, information-theoretic bounds, or theoretical explanation for why certain methods outperform others.
3. The paper is missing for some analytical points that worth to investigate:
- No investigation of what linguistic patterns emerge under different distortions. Although Figure 2 in section 4.3 provides some qualitative study, the explanation is more like a caption of the figure instead of deeper pattern analysis in an quantative way. 
- Not a single alignment method consistently outperform others in the analysis and it's confusing about why a certain method performs well under a task setting, and fails in another task setting. 
- Missing baseline results like simply prompting the model with the type of visual inconsistency and let it generate the description. Since the zero-shot baseline does not provide a prompt template, we don't know whether this simple baseline has been added or not.
4. The paper simply applies existing methods to a new task without any task-specific innovations or adaptations, which limits its novelty.

### Questions
1. Why does KTO outperform other methods? What properties of the algorithm make it suitable for this task?
Could you design task-specific methods that explicitly model the perceptual divergence rather than treating it as a black-box adaptation problem?
2. What linguistic patterns or communication strategies emerge from adaptation? Do models develop generalizable protocols or merely memorize dataset-specific patterns?
3. In Table 2, why the "cataract", "Tunnel Vision", "Detached Retina" and "AMD" columns do not have a bolded number?
4. Directly fine-tuning the models with the preference dataset, but without telling it what kind of perceptual distortions the listeners have can be a confusing choice, the model may simply learn to describe as much as possible instead of describe accordingly for different distortion type.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new setting called “divergent visual spaces”, where two multimodal agents see the same world differently. One observes the clean image, while the other receives a distorted version (simulating human vision impairments).
Using image reference games, the authors analyze how such mismatches affect communication and compare offline alignment (SFT, DPO) versus online alignment (KTO, GRPO) strategies.

### Strengths
- Novel problem formulation: Studies communication under perceptual asymmetry, a realistic and underexplored challenge for multimodal agents, extending beyond identical sensory inputs.
- Well-designed benchmark: Provides new preference dataset (distorted vs undistorted prompts) and controlled visual impairments inspired by human conditions.
- Strong qualitative insight: Demonstrates how models adapt their descriptions to match limited visual cues (e.g., from color-based to structure-based features).

### Weaknesses
1. On Dataset Curation and Sample Construction
- How the paired image samples (target vs. distractor) were selected? My understanding is that they should be semantically similar but not identical to make the reference game non-trivial. However, this similarity control can be hard to standardize.
Could the authors clarify the criteria or process used for curating these pairs and ensuring balanced difficulty across distortions?
2. On the Generalization of Training Effects
- Beyond the reference game, does this training improve the model’s general image understanding or fine-grained perception?
If the speaker learns to “see through distortions,” could this act as a form of RL-based perceptual sharpening, transferable to other downstream tasks (e.g., captioning, general image understanding)?
- Conceptually, the training objective appears to mix generative (speaker) and discriminative (listener) alignment. So it's better to evaluate the model on image understanding tasks like MMBench, MMVet, etc.
3. On Evaluation Robustness (False Positives / False Negatives)
- The current evaluation may suffer from ambiguity bias: a coarse or underspecified description could accidentally fit both candidate images, leading to inflated listener accuracy (false positives).
- Conversely, fine-grained but valid descriptions might mismatch due to listener limitations (false negatives).
- Have the authors considered prompt variants (e.g., “Describe this image in detail” vs. short prompts) to quantify this sensitivity? Would richer prompts yield more reliable alignment and higher communication success?

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies how multimodal agents communicate when they perceive images differently. Using an image reference game, one agent sees clear images while another sees distorted ones mimicking human visual impairments like cataract or color blindness. The authors test four adaptation methods: offline and online. Offline methods use preference data contrasting distortion-aware vs. normal descriptions; online ones learn from feedback during interaction. Experiments on CLEVR and CUB datasets show all methods improve over baselines. DPO performs best when high-quality data exist, while KTO gives the most consistent results across distortions. Adapted speakers learn to emphasize features the listener can still perceive. Cross-dataset and cross-model tests show offline methods give larger but less stable gains, whereas online methods are more reliable. Overall, offline adaptation works best when supervision is available; online adaptation is better for dynamic, unknown perceptual gaps.

### Strengths
1. While most visual-language alignment studies focus on human-AI interaction, this paper provides valuable insight by considering multi-agent systems instead.

2. The qualitative analysis results were particularly impressive.

### Weaknesses
1. Although the paper emphasizes agent systems and interactive settings as its main focus, there is almost no analysis of multi-turn interactions, nor any investigation of essential agent capabilities such as tool use.

2. Relatedly, the paper lacks elements that support the generalizability of its findings such as diversity in benchmarks and models. It appears that only Qwen2.5-VL-32B was used; including results from other VLMs or model sizes would strengthen the claims. This was the most decisive factor in my final rating, and demonstrating broader generalizability would significantly improve the paper.

3. The overall message of the research is not clearly conveyed. It’s unclear whether online methods are definitively better than offline ones, or if each has advantages in specific situations. The paper does not clearly specify when each method performs better. Moreover, differing trends across benchmarks make the results difficult to interpret.

### Questions
1. The paper emphasizes multi-agent communication, yet most experiments involve single-turn interactions. How would the proposed methods scale to multi-turn or continuous dialogue settings between agents?

2. Have the authors considered incorporating tool use or memory capabilities that are crucial for realistic agent systems into their framework? How might divergent visual spaces affect such abilities?

3. The study relies on simulated perceptual distortions. How closely do these distortions correspond to real-world sensor or perception mismatches in practical multi-agent systems?

4. The experiments use Qwen2.5-VL-32B (and partly InternVL3.5). Would the conclusions hold across different model architectures or smaller/larger model sizes?

5. How sensitive are the results to the choice of visual-language model? For instance, would models like GPT-4V or LLaVA show similar adaptation behaviors?

6. Could the authors test on additional benchmarks or more naturalistic visual environments to demonstrate broader generalizability beyond CLEVR and CUB?

7. Are there any ablation studies showing how much each distortion type (e.g., color loss vs. occlusion) affects alignment performance?

8. The paper’s main message about when to use offline versus online adaptation seems unclear. Can the authors provide a clearer guideline or decision boundary for practitioners?

9. Some results show inconsistent trends across benchmarks does this suggest that the effectiveness of each method depends strongly on task characteristics?

10. How do the authors define or measure "alignment" between agents purely through task success rates, or is there a measure of interpretability or communicative grounding?

### Soundness
2

### Presentation
3

### Contribution
2
