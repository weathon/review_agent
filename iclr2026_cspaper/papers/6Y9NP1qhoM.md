# Goal-Aware Identification And Rectification Of Misinformation In Multi-Agent Systems

Zherui Li1 Yan Mi1 Zhenhong Zhou2 Houcheng Jiang3 **Guibin Zhang**4 Kun Wang2∗ **Junfeng Fang**4∗
1Beijing University of Posts and Telecommunications 2Nanyang Technological University 3University of Science and Technology of China 4National University of Singapore

## Abstract

Large Language Model-based Multi-Agent Systems (MASs) have demonstrated strong advantages in addressing complex real-world tasks. However, due to the introduction of additional attack surfaces, MASs are particularly vulnerable to misinformation injection. To facilitate a deeper understanding of misinformation propagation dynamics within these systems, we introduce MISINFOTASK, a novel dataset featuring complex, realistic tasks designed to evaluate MAS robustness against such threats. Building upon this, we propose ARGUS, a two-stage, training-free defense framework leveraging goal-aware reasoning for precise misinformation rectification within information flows. Our experiments demonstrate that in challenging misinformation scenarios, ARGUS exhibits significant efficacy across various injection attacks, achieving an average reduction in misinformation toxicity of approximately 28.17% and improving task success rates under attack by approximately 10.33%. Our code and dataset are available at:
https://github.com/zhrli324/ARGUS.

## 1 Introduction

Large Language Model (LLM)-based agents (Xi et al., 2023; Wang et al., 2024), integrating the decision-making capabilities of core LLMs with memory (Zhang et al., 2024d), tool calling (Qu et al., 2025), prompt engineering strategies (Sahoo et al., 2025), and appropriate information control flows (Li, 2024), have demonstrated considerable potential in tackling real-world problems. Multi- Agent Systems (MASs) further amplify this capability by harnessing the collective intelligence of multiple agents (Guo et al., 2024; Wang et al., 2025a), exhibiting significant advantages in addressing challenging tasks (Wu et al., 2023; Hong et al., 2024). However, the progression of MAS towards widespread adoption has concurrently exposed their inherent vulnerabilities (Yu et al., 2025; Wang et al., 2025a). Their complex topologies and interactive communication links introduce new attack surfaces (Yu et al., 2024), making these systems highly susceptible to internal information biases and external manipulation. Internal risks primarily manifest as spontaneous hallucinations (Huang et al., 2025a). External risks present greater complexity; beyond overtly malicious content, a more insidious and pervasive threat has emerged: misinformation injection (Lee & Tiwari, 2024; Liu et al., 2024a), which poses a great impediment to the development of trustworthy MASs. Among external threats, misinformation denotes statements that appear semantically benign on the surface yet are factually incorrect (Chen & Shu, 2023; 2024); this distinguishes it from malicious information characterized by its overtly malicious intent. As illustrated in Figure 1, the latter's characteristic enables it to readily circumvent conventional detection mechanisms, endowing it with a high degree of covertness different from overtly malicious content (Chen & Shu, 2024). More critically, its potential for harm is substantial. During the collaborative execution of complex tasks by MAS, even seemingly trivial instances of malicious or misinformation can be amplified, ultimately leading to the collapse of the entire task chain (Pastor-Galindo et al., 2024). Currently, such covert and harmful information can be injected into MAS through critical components such as agent prompts (Lee & Tiwari, 2024; Greshake et al., 2023), memory (Zou et al., 2024; Chen et al., 2024), and tools (Zhan et al., 2024), thereby creating opportunities for its propagation.

∗Corresponding author: wang.kun@ntu.edu.sg, fangjf1997@gmail.com.

1 To identify and counter information injection attacks in MAS, prior works have explored various approaches, including adversarial defense through attack-defense confrontation (Zeng et al., 2024; Lin et al., 2025), consensus-based mechanisms leveraging collective consistency assessments (Chern et al., 2024), and structural defense focusing on MAS topological graph structures (Wang et al., 2025b). Despite their significant contributions to resisting information injection in MAS, most of these methods (I) have not focused their defensive strategies on covert yet dangerous misinformation, and (II) have selected evaluation tasks of insufficient complexity, failing to adequately reflect MAS capabilities in handling real-world complex tasks. Consequently, this highlights an urgent need to develop a more application-oriented, agent-centric misinformation injection evaluation and to design robust, adaptive, and efficient defense frameworks.

![1_image_0.png](1_image_0.png)

Figure 1: Overview of the ARGUS framework guarding against misinformation. The left panel contrasts the attributes of malicious information versus misinformation. The right panel visualizes the defense pipeline.

To conduct an in-depth investigation into the propagation patterns of misinformation in MAS, we introduce MISINFOTASK, a redteaming dataset specifically designed for MAS misinformation injection testing. For each task sample, we provide potential misinformation injection scenarios accompanied by supporting or refuting argument sets. Furthermore, to mitigate the challenge posed by the highly covert nature of misinformation, we propose **ARGUS** (Adaptive Reasoning and Goal-aware Unified Shield), an adaptive and unified defense framework engineered to defend against a diverse range of information injection attacks. AR- GUS operates through two core phases: Adaptive Localization and Goal-aware Persuasive Rectification. ARGUS analyzes the MAS from a spatial perspective, conducting a holistic assessment of communication channels by considering their topological importance and content-level semantic relevance to potential misinformation targets. During the Persuasive Rectification phase, ARGUS operates along the temporal dimension of MAS, leveraging agents' inherent Chain-of-Thought (Wei et al., 2023) reasoning capabilities to detect and rectify potential misinformation within information flows. We systematically evaluate the robustness of MAS against misinformation using various attack methods on MISINFOTASK, and assess the defensive performance of ARGUS across different core LLMs and interaction rounds. Experimental results indicate that generic MAS architectures exhibit significant vulnerability to misinformation injection; they can easily be induced to task failure by carefully crafted misinformation, resulting in an average reduction of 20.04% in task success rates. In response to this challenge, our ARGUS framework demonstrates robust defensive capabilities, reducing misinformation toxicity by approximately 38.24% across various core LLMs and improving the task success rate of attacked MAS by approximately 10.33%. We believe this research can inspire the MAS community to advance towards more trustworthy Multi-Agent Systems.

## 2 Preliminary 2.1 Multi-Agent System As Graph

Inspired by prior work that models MAS as topological graphs to analyze them through the perspective of graph theory and information propagation (Wu et al., 2023; Liu et al., 2024b; Zhuge et al., 2024), we adopt a similar graph-based representation. We define an MAS as a directed graph G = (A, E). Here, A = {ai}
N
i=1 represents the set of all N agents, which serve as the nodes in the graph. The set of edges E = {eij | ai, aj ∈ A, i ̸= j} denotes the communication channels between agents, where an edge eij signifies a directed communication channel from agent aito agent aj .

## 2.2 Information Flow In Mas

Intra-agent Level. Each agent ai ∈ A is conceptualized as an ensemble comprising a central LLM Mi, a memory module Memi, a set of available tools Ti, and its prompt engineering strategy Pi (Xi et al., 2023; Wang et al., 2024). In its fundamental operation, ai utilizes Mito process an input prompt, potentially augmented with information from Memi, to generate an output, such as calling a tool from Ti. Advanced agent architectures, like Chain-of-Thought (CoT) (Wei et al., 2023) and ReAct (Yao et al., 2023), enhance the internal decision-making processes by incorporating step-bystep reasoning and environment interaction capabilities (Zhang et al., 2025a; 2024c). Inter-agent Level. Inter-agent interactions within the MAS are governed by the topological graph G = (A, E) detailed in Section 2.1, with information propagating along communication channels
(Zhuge et al., 2024; Zhang et al., 2025b). At each time step t, an agent ai ∈ A may autonomously decide to transmit a message meij (t) to an adjacent agent aj ∈ Nout(ai). Here, Nout(ai) denotes the set of agents reachable from ai via an edge, and eij represents the specific communication channel from aito aj . Such messages meij (t) are received by aj as external input uj (t), influencing its subsequent observations oj (t) and belief state sj (t) within its decision-making process.

## 2.3 Misinformation In The System

Misinformation is generally understood as information that is erroneous or factually incorrect (Pastor-Galindo et al., 2024). Within the context of this paper, we specifically define misinformation as content that contradicts the factual knowledge implicitly stored in the parameters of an LLM, particularly one that has undergone alignment. Unlike overtly malicious or jailbreak content typically addressed in safety research, the core objective of misinformation investigated in this work is to subtly misguide the MAS (Chen & Shu, 2024). This misguidance can cause the system to deviate from its operational trajectory, ultimately leading to behaviors that are orthogonal to human expectations, thereby inducing erroneous decision-making and potentially culminating in task failure.

## 3 Evaluating Misinformation Injection 3.1 Misinfotask Dataset

Extensive research has explored information injection attacks (Ju et al., 2024; Liu et al., 2025; He et al., 2025) and defenses (Mao et al., 2025; Zhong et al., 2025; Wang et al., 2025b) in MAS, many of which have demonstrated notable success. However, our review of the existing literature reveals that the majority of studies on MAS information injection predominantly focus on overtly malicious or jailbreak inputs. While a subset of research does address the propagation of misinformation (Ju et al., 2024; Wang et al., 2025b), the datasets employed in these experimental evaluations often lack specific relevance to this particular challenge. Specifically, we identify two critical gaps: (1) there is a scarcity of datasets expressly designed for studying misinformation injection and defenses within MAS; and (2) existing research frequently utilizes datasets composed of simplistic questionanswering tasks with straightforward procedures. To fill the gap in the domain of misinformation injection and defense, we introduce MISINFOTASK, a multi-topic, task-driven dataset designed for red teaming misinformation in MAS. MISINFOTASK comprises 108 realistic tasks suitable for MAS to solve, and provides potential misinformation injection points and reference solution workflows. Crucially, to facilitate adversarial red teaming research, we have developed 4-8 plausible yet fallacious arguments corresponding to potential misinformation for each task, along with their respective ground truths.

Dataset Construction. To ensure the quality of our synthesized data, we employed a rigorous construction methodology. We first authored a small set of high-quality seed examples. These examples were then used to guide the sampling process with the detailed prompt provided in Appendix G. The resulting data was subsequently manually filtered and curated based on the following criteria:
- Ensure the generated data entries align with concrete, real-world task scenarios. - Guarantee the misinformation constitutes a factual error highly pertinent to the defined task. - Ensure comprehensive coverage of the following categories: Conceptual Reasoning, Factual Verification, Procedural Application, Formal Language Interpretation, and Logic Analysis.

## 3.2 Setup

In this section, we introduce our MAS platform, baseline attack methods, and evaluation metrics. MAS Platform. We construct an MAS to serve as the experimental testbed. Specifically, a planning agent acts as the initial interface for user queries and undertakes responsibilities such as task decomposition and work allocation (Li et al., 2023; Wu et al., 2023). Subsequently, information flows into the main MAS topological graph, and the task is completed through multiple rounds of interaction among multiple agents. All agents will autonomously select their communication partners and determine the content of their messages. Finally, a conclusion agent analyzes dialogues and actions within the MAS to synthesize a final result and provide an explanation for the user, acting as the system's output interface.

Baseline Attacks. We employ three baseline information injection methods: Prompt Injection (PI) (Greshake et al., 2023; Lee & Tiwari, 2024), RAG Poisoning (RP) (Zou et al., 2024), and Tool Injection (TI) (Zhan et al., 2024; Ruan et al., 2024). For Prompt Injection and Tool Injection, we designate one agent as the point of compromise. Misinformation arguments are then injected into its system prompt or tool module. For RAG Poisoning, the arguments are injected directly into the MAS's shared public vector database, which serves as a common knowledge source for agents. Evaluation Metrics. To assess the impact of misinformation, we define two core metrics: Misinformation Toxicity (MT) and *Task Success Rate* (TSR). These metrics aim to quantify the extent of misinformation assimilation and its effect on overall task performance, respectively. The specific evaluation methods are as follows:

![3_image_0.png](3_image_0.png)

$$\text{MT=}\frac{1}{N}\sum_{k=1}^{N}\text{Score}(O_{k},g_{mis}^{k}),\quad\text{TSR=}\frac{1}{N}\sum_{k=1}^{N}\mathbb{I}(\text{Score}(O_{k},g_{task}^{k})\!\geq\!\theta_{m}),\tag{1}$$

where N represents the total number of evaluated task instances. For the k-th task instance, Ok is the final output generated by the conclusion agent, g kmis denotes the misinformation's intent-driven goal, and g k task signifies the reference solution for the task. The Score(·, ·) function, evaluated by an LLM judge, measures the semantic consistency between two inputs, yielding a score within the range of [0, 10]. The term θm is a predefined threshold. Finally, I(·) is the indicator function, returning 1 if the specified condition is met and 0 otherwise.

## 3.3 Misinformation Robustness In Mas

Threat Model. We define the assumed attacker broadly as any entity seeking to disrupt the functionality of MAS. The attacker compromises a single agent within the MAS, gaining the ability to individually manipulate its prompt, tool, or RAG memory. These three manipulation vectors correspond respectively to the three attack methodologies detailed in Section 3.2. Utilizing MISINFOTASK dataset, we conduct red team testing on the MAS employing the three injection methods detailed in Section 3.2, with the aim of assessing the MAS's robustness against externally introduced misinformation. Our experimental procedure involves the planning agent determining the MAS's topological structure before task execution. Misinformation is subsequently injected at the initial round of the operational sequence. Details are provided in Appendix B.

As shown in Figure 2, the injection of misinformation severely compromises the belief states in the MAS. Across all tested injection methods, the MT metric for the MAS rises from a baseline of 1.28 in the vanilla configuration to approximately 4.71. Concurrently, the TSR declines significantly from an initial value of 87.47% to 67.70%. These results demonstrate the vulnerability of generic MAS architectures to misinformation.

![4_image_0.png](4_image_0.png)

## 4 Argus Framework

To mitigate the vulnerability of MAS to misinformation, we introduce ARGUS, a modular and training-free framework designed to offer a unified shield against diverse misinformation threats. The core principle of ARGUS involves a two-stage approach: (1) the adaptive mechanism for identifying critical misinformation propagation channels in the MAS (Section 4.1); (2) the deployment of a corrective agent acor and its goal-aware persuasive rectification (Section 4.2). Figure 3 illustrates the overall pipeline of ARGUS framework.

## 4.1 Critical Flow Localization In Graphs

We formally define the misinformation channel localization problem as follows: Given the complete dialogue logs of the MAS from round r, the objective is to identify a subset of edges Er ⊆ E such that for every eij ∈ Er, the message meij transmitted over this edge belongs to M′, where M′is the set of all messages contaminated by misinformation.

## 4.1.1 Initial Localization

Before the initial round of the MAS (i.e., at r=1), we utilize the topological structure of the graph G=(A, E) to determine the initial deployment strategy for the corrective agent acor. In the absence of dynamic interaction logs at this stage, our objective is to identify edges that are central to information flow. To this end, we compute a normalized Edge Betweenness Centrality score for each edge e ∈ E as its topological importance Score*topo*(e):

$${\tt S c o r e}_{t o p o}(e){=}\frac{1}{N_{n o r m}}\sum_{a_{i}\in{\mathcal{A}}}\sum_{a_{j}\in{\mathcal{A}},i\neq j}\frac{\sigma_{i j}(e)}{\sigma_{i j}},$$

where σij denotes the total number of shortest paths between ai and aj , σij (e) is the count of such shortest paths that pass through edge e, and N*norm* is a normalization factor. In selecting the initial edge set E1 for deploying the corrective agent acor, we aim to balance the
topological importance of individual directed edges with the comprehensive coverage of their source
nodes. For each source node ai ∈ A, we identified its highest-scoring outgoing edge e
∗
i:
$$e_{i}^{*}=\arg\operatorname*{max}_{e_{i},\in{\mathcal{E}}}\left\{{\tt S c o r e}_{t o p o}(e_{i}.)\right\},$$
{Score*topo*(ei·)} , (3)
with selected edges collectively forming the set E*best*= {e
∗ i | ai *∈ A}*. To select k edges for initial monitoring and corrective action deployment at round r=1, the initial monitored edge set E1 is constructed as follows. First, we determine k1= min (k, |E*best*|), where E*best* is the set of highestscoring outgoing edges previously identified for each agent. Then we set k2=k−k1. The set E1 is

$${\mathrm{(2)}}$$
$$({\mathfrak{I}})$$

then formed by the union of two subsets:

$${\mathcal{E}}\setminus{\mathcal{E}}_{b e s t},\mathbf{S c o r e}_{t o p o}),$$
$$(4)$$

E1 = Topk1
(E*best*, Score*topo*) ∪ Topk2
(E \ E*best*, Score*topo*), (4)
where Topk
(E, Score) selects top-k highest-ranked elements from set E, with ranking set E in descending order according to the Score function. This approach is designed to ensure that acor can monitor critical edges while overseeing a broad range of agents. The complete set of topological scores Scoretopo(eij ), eij ∈ E is preserved for utilization in subsequent Adaptive Re-Localization.

## 4.1.2 Adaptive Re-Localization

For subsequent rounds of the MAS (i.e., for r > 1), the deployment positions of the corrective agent acor are dynamically adapted. In this phase, the adaptive localization aims to identify top-k channels where the transmitted messages exhibit the highest semantic similarity to the inferred intent-driven goal of the misinformation.

Specifically, during round r−1, acor will output a textual description g
′mis of the most probable intent-driven goal it has inferred for each channel it monitors. These descriptions are aggregated and then subjected to a deduplication process based on the cosine similarity of their respective embedding vectors, resulting in a refined set of unique inferred intent-driven goal description of misinformation, denoted as G
′mis={g
′imis}
p i=1. The detailed method for this goal identification and reasoning by acor is presented in Section 4.2.

Subsequently, we first compute the list of embedding vectors V
′mis={v
′i
}
p i=1 for all inferred misinformation goal descriptions in the set G
′mis, i.e., v
′i=Φ(g
′imis). The notation Φ(·) denotes the function used to obtain embedding vectors. For each sentence s in a given message m, we calculate the average similarity of its embedding Φ(s) to all target goal embeddings v
′ ∈ V
′
goal. This average sentence cosine similarity S(*s, V* ′
goal) is given by:

$${\mathcal{S}}(s,V_{g o a l}^{\prime})=\frac{1}{p}\sum_{i=1}^{p}\mathrm{{\sf Sim}}_{c o s}(\Phi(s),v_{i}^{\prime}).$$
$$(S)$$
$$(6)$$
$$(7)$$
$$({\mathfrak{s}})$$

The relevance of message m to the set of inferred goals, Rel(*m, V* ′
goal), was then determined by taking the maximum similarity S among all sentences in m that exceeded a threshold θsim:

$$\mathtt{Rel}(m,V_{g o a l}^{\prime}){=}\operatorname*{max}_{s\in m}\left\{\{0\}\cup\mathcal{S}(s,V_{g o a l}^{\prime})\right\}\quad{\mathrm{s.t.}}\quad\mathcal{S}(s,V_{g o a l}^{\prime})\geq\theta_{s i m}.$$

The relevance score for e, denoted Scorerel(e), is defined as the maximum relevance value of all messages m ∈ mr−1 e flowing through this edge in round r−1, we formalize it as:

$$\mathsf{Score}_{r e l}(e)=\operatorname*{max}_{m\in m_{e}^{r-1}}(\mathsf{Rel}(m,V_{g o a l}^{\prime})).$$

Furthermore, to incorporate the communication intensity of each channel into our assessment of its importance, we calculate a frequency score. The frequency score for edge e in round r−1, denoted Scorer−1 f req(e), is defined as the total number of messages transmitted over e during that round:

$$\texttt{Score}_{f r e q}^{r-1}(e){=}\texttt{count}(m_{e}(r)).$$

f req(e)=count(me(r)). (8)
In summary, for each edge e ∈ E, we compute a comprehensive score Scorer(e) to guide the localization of monitored edges for round r. This score combines the channel's initial topological importance Score*topo*(e), the channel's information relevance Scorerel(e), and the channel's usage frequency Score*f req*(e). The final score is calculated as a weighted sum. According to the final scores {Scorer(eij )| eij *∈ E}*, we select the Top-k highest-scoring edges as the monitoring edges set Er for the current round:

$${\mathcal{E}}_{r+1}=\operatorname{\arg\operatorname*{max}}_{{\mathcal{E}}^{\prime}\subseteq{\mathcal{E}},|{\mathcal{E}}^{\prime}|=k}\sum_{e\in{\mathcal{E}}^{\prime}}{\mathsf{S c o r e}}^{r}(e).$$
Scorer(e). (9)
The corrective agents acor are then deployed onto the communication channels corresponding to set Er in preparation for monitoring during round r. This adaptive re-localization process is iteratively performed at the end of each round, enabling dynamic optimization of the monitoring locations throughout the MAS operation.

$$({\mathfrak{g}})$$

## 4.2 Goal-Aware Reasoning For Multi-Round Persuasive Rectification

Misinformation encountered in real-world applications is diverse, covering knowledge from various domains and exhibiting multifaceted paradigms (Chen & Shu, 2024; 2023), making it difficult to correct using traditional methods (Akgun et al. ¨ , 2025; Huang et al., 2025b). To address this, we adopt an internal knowledge activation strategy guided by heuristic principles (Yuan et al., 2024; Gao et al., 2023), aiming to leverage the LLM's inherent reasoning ability to activate its own parameterized knowledge. Specifically, when a message m flows through one of the critical channels identified by our localization mechanism (Section 4.1), the corrective agent acor will activate a multi-stage process of in-depth analysis and intervention, which is structured around CoT prompting. Multi-faceted Identification of Suspicious Elements. This initial stage involves a sentence-bysentence deconstruction of the intercepted message m by corrective agent acor. This CoT-guiding process aims not only to identify explicit factual assertions within the message but also to uncover a spectrum of potential vulnerabilities. These include latent logical inconsistencies, deviations from common sense, and ambiguous phrasings (Chen & Shu, 2023; Fontana et al., 2025). Internal Knowledge Resonance. For each suspicious anchor point identified in the preceding identification stage, acor then initiates a process of internal knowledge resonance. This involves activating relevant knowledge clusters in its parameterized knowledge base. Subsequently, these activated internal knowledge structures are leveraged to perform deep semantic comparisons against the external information derived from the message m. Heuristic Persuasive Reconstruction. Upon confirming the existence of critical discrepancies in m that conflict with its internal knowledge, acor activates an information reconstruction module. This module generates corrective statements that have logical persuasiveness through strategies such as root cause analysis, cognitive reframing, and context-adaptive adjustments, aiming to rectify the identified misinformation. Detailed explanations for these strategies are provided in Appendix B.4. Notably, concurrent with the information rectification process, acor executes a parallel sub-task, Goal-aware Intent Inference. When it determines that the misinformation in a current message displays attributes of being highly organized or clearly discernibly misled, acor will systematically record its inference of the attacker's most probable misleading goal. This record will serve as an important input for the adaptive localization strategy before the start of the subsequent round, thereby enhancing ARGUS's capacity to respond to persistent, coordinated misinformation attacks.

## 5 Experiments

We focus our primary experiments on a more complex scenario of Misinformation Injection, conducting a comprehensive suite of tests to evaluate the efficacy of ARGUS and its pivotal role in defending MAS against misinformation. Further results are available in Appendix D.

## 5.1 Experimental Settings

We begin with a brief introduction to the key configurations for our experiments. For details on the dataset, MAS platform, and baseline methods, please refer to Section 3. Further specific configurations are documented in Appendix B. Core LLMs. The agents in our MAS are powered by one of four distinct LLMs, selected from different model families and varying in parameter scale: GPT-4o-mini, GPT-4o (OpenAI et al., 2024), DeepSeek-V3 (DeepSeek-AI et al., 2025), and Gemini-2.0-flash (Team et al., 2025).

Evaluation. We employ an LLM (GPT-4o-2024-08-06) for automated scoring. We utilize the two metrics mentioned in Section 3.3, MT and TSR, to respectively quantify the adverse impact of misinformation and the degree of task completion. The specific prompt is provided in Appendix G. Baseline Defense. For comparative analysis, we select established defense methods known to enhance the robustness of MAS, including Self-Check and G-Safeguard. Self-Check (Manakul et al., 2023; Miao et al., 2023) involves prompting agents to critically re-evaluate and reflect on the information they process. G-Safeguard (Wang et al., 2025b) employs Graph Neural Networks (Wu

GPT-4o-mini

Attack-only 4.94 67.74 4.95 65.79 5.78 68.75 5.22 67.43

Self-Check 4.54↓ 0.40 69.45↑ 1.71 4.95↓ 0.00 66.14↑ 0.35 5.55↓ 0.23 69.54↑ 0.79 5.02↓ 0.20 68.38↑ 0.95

G-Safeguard 4.00↓ 0.94 68.32↑ 0.58 5.19↑ 0.24 67.46↑ 1.67 3.01↓ 2.77 70.46↑ 1.71 4.07↓ 1.15 68.75↑ 1.32

ARGUS **3.73**↓ 1.21 **75.86**↑ 8.12 **3.91**↓ 1.04 **69.77**↑ 3.98 **2.67**↓ 3.11 **89.66**↑ 20.91 **3.43**↓ 1.79 **78.43**↑ 11.00

GPT-4o

Attack-only 5.40 56.25 5.26 68.72 4.05 76.25 4.90 67.07

Self-Check 5.07↓ 0.33 57.34↑ 1.09 5.22↓ 0.04 71.56↑ 2.84 3.98↓ 0.07 76.26↑ 0.01 4.75↓ 0.15 68.39↑ 1.32 G-Safeguard 4.01↓ 1.39 55.31↓ 0.94 5.22↓ 0.04 68.36↓ 0.36 **2.90**↓ 1.15 73.26↓ 2.99 4.04↓ 0.86 65.64↓ 1.43

ARGUS **3.58**↓ 1.82 **73.75**↑ 17.50 **3.91**↓ 1.35 **74.58**↑ 5.86 3.05↓ 1.00 **82.56**↑ 6.31 **3.51**↓ 1.39 **76.96**↑ 9.89

DeepSeek-V3

Attack-only 4.96 83.75 4.85 72.15 3.96 86.25 4.59 80.72

Self-Check 3.90↓ 1.06 85.11↑ 1.36 4.70↓ 0.15 75.16↑ 3.01 3.55↓ 0.41 87.53↑ 1.28 4.05↓ 0.54 82.60↑ 1.88

G-Safeguard 4.26↓ 0.70 80.16↓ 3.59 4.89↑ 0.04 74.48↑ 2.33 **2.86**↓ 1.10 84.13↓ 2.12 4.00↓ 0.59 79.59↓ 1.13

ARGUS **3.11**↓ 1.85 **86.44**↑ 2.69 **3.77**↓ 1.08 **76.79**↑ 4.64 **2.86**↓ 1.10 **89.75**↑ 3.50 **3.25**↓ 1.34 **84.33**↑ 3.61

Gemini-2.0-flash

Attack-only 4.20 62.50 4.68 71.43 3.49 70.01 4.12 67.98

Self-Check 4.02↓ 0.18 64.56↑ 2.06 4.61↓ 0.07 72.64↑ 1.21 2.80↓ 0.69 71.16↑ 1.15 3.81↓ 0.31 69.45↑ 1.47 G-Safeguard 3.89↓ 0.31 64.51↑ 2.01 4.51↓ 0.17 71.51↑ 0.08 2.60↓ 0.89 70.50↑ 0.49 3.67↓ 0.45 68.84↑ 0.86

ARGUS **3.60**↓ 0.60 **65.78**↑ 3.28 **4.13**↓ 0.55 **77.02**↑ 5.59 **2.49**↓ 1.00 **74.43**↑ 4.42 **3.40**↓ 0.72 **72.41**↑ 4.43

Prompt Injection RAG Poisoning Tool Injection Avg. MT ↓ **Avg. TSR** ↑

MT ↓ TSR ↑ MT ↓ TSR ↑ MT ↓ TSR ↑

Table 1: This table presents detailed results for Misinformation Toxicity (MT; score range: [0, 10]) and Task Success Rate (TSR; reported in %) of the MAS. The data illustrate the performance of various defense strategies when subjected to different injection techniques. **Bold** values indicate the best performance (lowest MT or highest TSR) within each model group. Rows with a gray background indicate the proposed ARGUS method.

et al., 2021) to identify high-risk agents and subsequently implements remediation measures via edge pruning. Further details are available in Appendix B.3.

## 5.2 Effectiveness Of Argus

Our experiments are conducted on the MISINFOTASK dataset (Section 3.1). We evaluate the MAS performance over 5 operational rounds under various configurations, employing different core LLMs, information injection methods, and defense strategies. The MT and TSR metric of the final outputs is assessed, with comprehensive results presented in Table 1. The results reveal that in attack-only scenarios, MAS with various core LLMs all achieve high MT scores, underscoring their vulnerability to misinformation. Furthermore, defense mechanisms such as Self-Check and G-Safeguard demonstrate limited efficacy in mitigating this threat, while our ARGUS framework achieves robust defense against misinformation injection, reducing MT by 28.18%, 20.38%, and 35.95% on average for Prompt Injection, RAG Poisoning, and Tool Injection, respectively. To further explore the reliability of the adaptive localization (Section 4.1), we evaluated the accuracy with which the corrective agent acor inferred the intended misleading goal of the misinformation. These results are presented in Figure 4. Our findings indicate that our adaptive dynamic monitoring module successfully identified the misinformation's guiding direction with high accuracy.

## 5.3 How Argus Defend The Misinformation

To understand the mechanism of misinformation propagation in MAS, we conduct a longitudinal analysis of MT across successive rounds. We collect comprehensive behavioral logs from each round of MAS operation, calculate MT for them, thereby quantifying the degree to which agents are polluted by misinformation in each round. These temporal trends are shown in Figure 5.

As can be seen from the figure, in the absence of any defense mechanism, the system's MT progressively escalates with an increasing number of rounds, which underscores the contagious and insidious nature of misinformation attacks. Conversely, after applying our ARGUS method, the MT scores under various attack methods all decrease round by round, which reflects ARGUS's capability to effectively discern the intent and content of the misinformation within the MAS and successfully curtail its propagation.

![8_image_0.png](8_image_0.png)

![8_image_1.png](8_image_1.png)

## 5.4 On The Impact Of Topology

To comprehensively assess the robustness of MAS against misinformation and the defensive capabilities of ARGUS, we employed five distinct MAS topological structures: Self-Determination, Chain, Full, Circle, and Star. We introduce each topology in detail in Appendix B.1. Employing DeepSeek-V3 as the core LLM, we conducted misinformation injection and defense tests using the MISINFOTASK dataset on MAS configured with each of the five aforementioned topologies. The results are illustrated in Figure 6. These experiments revealed that misinformation injection had a significant detrimental impact on MAS across all tested topological structures. Notably, our ARGUS framework demonstrated robust transferability, effectively detecting and rectifying the propagation of misinformation regardless of the underlying topology.

## 5.5 Ablation Study

To elucidate the contribution of individual components of ARGUS method to its overall corrective efficacy, we conduct an ablation study. We ablated core modules and re-evaluated the MT and TSR metric on the MAS. Furthermore, as an additional baseline, we conduct experiments where agent acor was explicitly provided with the ground truth of the misinformation during each task. Results in Table 2 indicate that the removal of any of these core modules led to a discernible degradation in ARGUS's performance. Conversely, when supplied with ground-truth information, ARGUS
exhibits an enhanced defensive capability.

We further conducted ablation studies on the hyperparameters governing the localization process in ARGUS, specifically the weights α, β, and γ assigned to the three importance scores. To evaluate the contribution of each score, we systematically adjusted these weights: first, by setting one weight to 0 while assigning 0.5 to the other two; and second, by setting one weight exclusively to 1 to isolate a single metric.

| PI                   | RP   | TI    |      |       |      |       |
|----------------------|------|-------|------|-------|------|-------|
| MT                   | TSR  | MT    | TSR  | MT    | TSR  |       |
| Attack only          | 4.88 | 69.44 | 4.93 | 63.89 | 4.24 | 70.37 |
| Attack + ARGUS       | 3.50 | 75.93 | 3.93 | 70.37 | 2.77 | 87.04 |
| w/o Dynamic Local.   | 4.55 | 68.52 | 4.56 | 64.81 | 3.80 | 74.07 |
| w/o CoT Revision     | 3.90 | 71.30 | 4.15 | 68.52 | 2.98 | 82.41 |
| w/o Multi-Turn Corr. | 4.63 | 70.37 | 4.61 | 62.04 | 3.88 | 71.30 |
| w/ Ground Truth      | 3.32 | 78.70 | 3.77 | 74.07 | 2.54 | 91.67 |

ARGUS 3.73 75.86

w/o α 4.14 70.37 w/o β 3.76 72.22 w/o γ 4.59 68.52 w/o β&γ 4.34 69.44 w/o α&γ 4.79 67.59 w/o α&β 3.91 73.14

MT TSR

Using Prompt Injection to introduce misinformation into the MAS, we measured the resulting MT and TSR. The results, presented in Table 3, indicate that while information relevance is the most critical factor, optimal defense performance is achieved only when it is combined with the other metrics.

## 6 Related Works

MAS Information Injection. The introduction of inter-agent interactions in MAS inherently gives rise to additional system-level security vulnerabilities. For example, Ju et al. (2024) employs knowledge manipulation in MAS to achieve malicious objectives. Prompt Infection (Lee & Tiwari, 2024) relies on information propagation to contaminate an entire MAS. AgentSmith (Gu et al., 2024) utilizes adversarial injection to poison a large number of agents; Zhang et al. (2024a) focuses on misleading agents into executing repetitive or irrelevant actions, thereby inducing malfunctions. Corba (Zhou et al., 2025) leverages recursive infection to disseminate a virus, leading to MAS collapse. MAS Defense Strategies. Several research efforts have focused on bolstering the security of MASs. Works like Netsafe (Yu et al., 2024) have explored the security of MAS graphs. Chern et al. (2024)
utilizes multi-agent debate mechanisms to enhance overall MAS security; AgentSafe (Mao et al.,
2025) uses hierarchical data management techniques to mitigate risks associated with data poisoning and leakage. AgentPrune (Zhang et al., 2024b) highlights the efficacy of graph pruning in improving MAS robustness. G-Safeguard (Wang et al., 2025b) leverages GNN to fit the MAS topological graph, thereby accurately locating high-risk agents.

## 7 Limitations & Future Works

While we believe that MISINFOTASK and ARGUS offer valuable contributions to the domain of misinformation injection and defense in MAS, several limitations should be acknowledged. First, the efficiency and cost of ARGUS require further consideration. The integration of an external defense module inherently introduces computational overhead, a common trade-off that is challenging to mitigate in MAS environments entirely. Second, the current study primarily addresses misinformation about knowledge resident in the agents' core LLMs. Safeguarding against misinformation that involves dynamic, time-sensitive information from external sources will likely need more sophisticated, multi-component collaborative defense strategies. Our future work will therefore focus on designing defense frameworks with enhanced efficiency and broader applicability, aiming to provide continued valuable insights for the development of truly trustworthy MAS.

## 8 Conclusion

This work presents a pioneering evaluation of the threat that misinformation injection poses to the security of MAS. To facilitate this research, we proposed MISINFOTASK dataset, and building on this, we introduce ARGUS, a defense system characterized by adaptive localization and goal-aware rectification. Experiments show that ARGUS exhibits outstanding performance and high generalization in countering diverse threats, offering valuable insights for future research in MAS security.

## Ethics Statement

The MISINFOTASK dataset and ARGUS framework presented in this work are intended to significantly advance the understanding and mitigation of misinformation within MASs. While these contributions offer new avenues for research, we strongly advocate that the MISINFOTASK dataset be utilized exclusively for research purposes, under rigorous oversight and governance. We further call upon the research community to approach the study of misinformation in MAS with a profound sense of responsibility, ensuring that all endeavors contribute positively to the development of more trustworthy and secure Multi-Agent Systems.

## Reproducibility

We commit to releasing the source code to promote the reproducibility of this work and to inspire further exploration in the field of MAS misinformation. The code is publicly available at https://github.com/zhrli324/ARGUS. Details of the models, datasets, and hyperparameter configurations used in our experiments are provided in Appendix B.

## References

Orhan Eren Akgun, Sarper Aydın, Stephanie Gil, and Angelia Nedi ¨ c. Multi-agent trustworthy ´
consensus under random dynamic attacks, 2025. URL https://arxiv.org/abs/2504.

07189.

Canyu Chen and Kai Shu. Combating misinformation in the age of llms: Opportunities and challenges, 2023. URL https://arxiv.org/abs/2311.05656.

Canyu Chen and Kai Shu. Can llm-generated misinformation be detected?, 2024. URL https:
//arxiv.org/abs/2309.13788.

Zhaorun Chen, Zhen Xiang, Chaowei Xiao, Dawn Song, and Bo Li. Agentpoison: Red-teaming llm agents via poisoning memory or knowledge bases, 2024. URL https://arxiv.org/abs/ 2407.12784.

Steffi Chern, Zhen Fan, and Andy Liu. Combating adversarial attacks with multi-agent debate, 2024.

URL https://arxiv.org/abs/2401.05998.

DeepSeek-AI, Aixin Liu, and Bei Feng et al. Deepseek-v3 technical report, 2025. URL https:
//arxiv.org/abs/2412.19437.

Nicolo' Fontana, Francesco Corso, Enrico Zuccolotto, and Francesco Pierri. Evaluating open-source large language models for automated fact-checking, 2025. URL https://arxiv.org/abs/ 2503.05565.

Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Y. Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and Kelvin Guu. Rarr: Researching and revising what language models say, using language models, 2023. URL https://arxiv. org/abs/2210.08726.

Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. Not what you've signed up for: Compromising real-world llm-integrated applications with indirect prompt injection, 2023. URL https://arxiv.org/abs/2302.12173.

Xiangming Gu, Xiaosen Zheng, Tianyu Pang, Chao Du, Qian Liu, Ye Wang, Jing Jiang, and Min Lin. Agent smith: A single image can jailbreak one million multimodal llm agents exponentially fast, 2024. URL https://arxiv.org/abs/2402.08567.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges, 2024. URL https://arxiv.org/abs/2402.01680.

Pengfei He, Yupin Lin, Shen Dong, Han Xu, Yue Xing, and Hui Liu. Red-teaming llm multi-agent systems via communication attacks, 2025. URL https://arxiv.org/abs/2502.14847.

Sirui Hong, Mingchen Zhuge, Jiaqi Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jurgen Schmidhuber. Metagpt: Meta programming for a multi-agent collaborative frame- ¨
work, 2024. URL https://arxiv.org/abs/2308.00352.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55, January 2025a. ISSN 1558-2868. doi: 10.1145/3703155. URL http://dx.doi.org/10.1145/3703155.

Tianyi Huang, Jingyuan Yi, Peiyang Yu, and Xiaochuan Xu. Unmasking digital falsehoods: A
comparative analysis of llm-based misinformation detection strategies, 2025b. URL https: //arxiv.org/abs/2503.00724.

Tianjie Ju, Yiting Wang, Xinbei Ma, Pengzhou Cheng, Haodong Zhao, Yulong Wang, Lifeng Liu, Jian Xie, Zhuosheng Zhang, and Gongshen Liu. Flooding spread of manipulated knowledge in llm-based multi-agent communities, 2024. URL https://arxiv.org/abs/2407.07791.

Donghyun Lee and Mo Tiwari. Prompt infection: Llm-to-llm prompt injection within multi-agent systems, 2024. URL https://arxiv.org/abs/2410.07283.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem.

Camel: Communicative agents for "mind" exploration of large language model society, 2023. URL https://arxiv.org/abs/2303.17760.

Xinzhe Li. A review of prominent paradigms for llm-based agents: Tool use (including rag), planning, and feedback learning, 2024. URL https://arxiv.org/abs/2406.05804.

Guang Lin, Toshihisa Tanaka, and Qibin Zhao. Large language model sentinel: Llm agent for adversarial purification, 2025. URL https://arxiv.org/abs/2405.20770.

Fengyuan Liu, Rui Zhao, Guohao Li, Philip Torr, Lei Han, and Jindong Gu. Cracking the collective mind: Adversarial manipulation in multi-agent systems, 2025. URL https://openreview. net/forum?id=kgZFaAtzYi.

Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, and Neil Zhenqiang Gong. Formalizing and benchmarking prompt injection attacks and defenses, 2024a. URL https://arxiv.org/abs/ 2310.12815.

Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, and Diyi Yang. A dynamic llm-powered agent network for task-oriented agent collaboration, 2024b. URL https://arxiv.org/abs/2310. 02170.

Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models, 2023. URL https://arxiv. org/abs/2303.08896.

Junyuan Mao, Fanci Meng, Yifan Duan, Miao Yu, Xiaojun Jia, Junfeng Fang, Yuxuan Liang, Kun Wang, and Qingsong Wen. Agentsafe: Safeguarding large language model-based multi-agent systems via hierarchical data management, 2025. URL https://arxiv.org/abs/2503. 04392.

Ning Miao, Yee Whye Teh, and Tom Rainforth. Selfcheck: Using llms to zero-shot check their own step-by-step reasoning, 2023. URL https://arxiv.org/abs/2308.00436.

OpenAI, :, Aaron Hurst, Adam Lerer, and Adam P. Goucher et al. Gpt-4o system card, 2024. URL
https://arxiv.org/abs/2410.21276.