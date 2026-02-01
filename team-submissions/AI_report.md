# AI Report

**Project Name:** LABS Solver

**Team Name:**  Squaxions

**GitHub Repository:** https://github.com/chtang-hmc/2026-NVIDIA

# Background

Background: The core problem we aim to solve is to find the binary sequence that minimizes
$$
E(s)=\sum_{k=1}^{N-1}C_k^2 \quad \text{with} \quad C_k=s_i s_{i+k}
$$

# Workflow

The AI Agents are classified as follows

* Primary agents (ChatGPT): The primary agent we relied on was ChatGPT, used for code generation and concept explanation.
* Coding agents (Cursor, Antigravity, Cursor, Claude) and ChatGPT: Used for rapid iteration on implementation, such as population generation for spins, creating an array of binary digits, such as $[0,1,1,1,0] $, and converting it to an array constituted by $\pm 1$ only. The agents are used as assistants and not decision makers. All codes are later examined against our testing functions, and before generating the code, we ask for the detailed explanations of the logic (and places of potential error in its implementation) from the agent itself. The agents here were also used for debugging.
* Reasoning and explanation agent (ChatGPT): Agents are used to explain the physics and mathematics of the LABS problem, including autocorrelation structure, and MTS, QE-MTS, and BF-DCQO algorithms. The agent is used primarily for conceptual understandings.
* Explanation and brainstorming agents (ChatGPT, DeepSeek, and AntiGravity): Agents here are used for more complicated code when the user is confused about the general framework. Multiple agents are use to compare results to make sure that the explanation is correct. For example, when implementing the BF-DCQO algorithm during the second phase, the process involves many new concepts that we are unfamiliar with. Agents are used to provide a framework for adapting the algorithm.

## Procedure for Using AI Agents

Before generating code using AI, the team ensures that sufficient technical background and clear objectives are provided. For instance, when asking ChatGPT to generate spin arrays for the Ising model, we do not simply ask "generate spin inputs." Instead, we explicitly define the workflow: a binary bit of [0, 1, 1, 0] is created earlier by the sample() function. The Ising model has only $\pm 1$, so the array should be converted to [-1, 1, 1, -1] if elements are 0. This additional context significantly improves the quality of the AI’s responses. After finishing coding, we used test functions to ensure the correctness of the results.

# Verification Strategy
We validated AI-generated code using explicit unit tests, sanity checks, and iterative debugging, with an emphasis on detecting logic errors and hallucinated behavior.

### Unit tests for correctness
We built a hand-written test suite for small LABS instances (N=1 to N=27), where correct objective values and optimal solutions were computed independently and stored as ground truth. All tests are defined in test.py.

These tests verify that:
* The LABS energy function returns exactly correct values for all test cases.
* Each algorithm (MTS, QE-MTS Phase 1, QE-MTS Phase 2) produces solutions with the expected energies for small N.
* AI-generated code does not introduce common hallucination errors, such as incorrect energy formulas, off-by-one indexing in correlation terms, sign mistakes, or improper handling of edge cases (e.g., very small N).

### Reproducibility checks
We added reproducibility tests to ensure that fixed random seeds produce identical results and that regenerated or refactored AI-written code does not change numerical outputs for the same inputs. This helped detect subtle logic changes that compile but are incorrect.

### Performance sanity checks
We included lightweight performance checks to confirm that runtime and memory usage scale sensibly with N, and that AI-generated code does not introduce unintended nested loops, redundant recomputation, or incorrect control flow that would silently degrade performance.

### Human-in-the-loop review
For any substantial AI-generated code block, we performed line-by-line review and on-the-go debugging, clarifying unclear logic with the agent and testing intermediate outputs. Thisconsistently caught hallucinations or incorrect assumptions before integration.

## Energy Verification
One crucial component of the project is energy computation using the following equation.

$$
H = 2\sum^{N-2}_{i=1}\sigma^z_i\sum^{(N-i)/2}_{k=1}\sigma^z_{i+k}+4\sum^{N-3}_{i=1}\sum^{(N-i-1)/2}_{t=k}\sum^{N-i-t}_{k=t+1}\sigma^{z}_{i+t}\sigma^{z}_{i+k}\sigma^{z}_{i+k+t}
$$

### Invariance and Symmetry

We compared the function’s outputs with energy values of known spin arrays. For instance, consider the array $s=[1, 1]$. Using the Hamiltonian, the Ising model has autocorrelation coefficients of $C_1=1$ and thus an energy of $1$.

Another technique is invariance or symmetry. There are several types of symmetries.

One type is inversion symmetry, meaning that energy $E(s)=E(-s)$ remains unchanged under a globl shift from a spin array $s$ to $-s$, where $s={s_1, s_2, \dots, s_N}$. Please refer to `_flip(s, j)_ ` in `test.py`.

`_alternating_inversion(s: np.ndarray)` applies the LABS gauge symmetry, flipping spins to $s\rightarrow (-1)^i s_i$ for $i Z$. For instance,
$
s=(1, 1, 1, 1…1)\rightarrow s=(-1, 1, -1, 1...)
$
Energy $E(s)$ remains invariant under the transformation. The test ensures the implementation obeys the symmetry.

### Bit Conversion

When moving from the classical binary bit to the Ising model, bit values change from 0 and 1 to an array of $\pm 1$. The spin array has a length greater than one with values of $\pm 1$. The function test_generate_bitstrings_shape_and_values raises ValueError if elements have invalid values(spin values that are not $\pm 1$) or the spin array have a wrong length.


# The Vibe Code

## One Win Case

One instance when AI saved us hours was in phase two, when we were implementing the biased-field digitized counteradiabatic quantum algorithm (BF-DCQO) by Romero et al. to optimize the runtime while ensuring correctness. 

When we were implementing the code for BF-DCQO, we were unsure of how to approach the problem. AI agent (ChatGPT) provided us with a high-level picture, highlighted functions we need to create, required modifications for our existing code, and explained how different functions operate with each other. We heavily reused the ChatGPT-generated code, sanity-checking it and testing against our unit tests.

## The Effect of AI on Learning

Initially, we were confused about the logic of the BF-DCQO algorithm, especially on how the biases are implemented and the significance of certain parameters ($\theta_{\text{cutoff}}$ and others). Reading over the paper and asking ChatGPT clarifying questions helped us to understand the fundamental framework of the algorithm and prevented us from being lost in the large amount of information. 

Furthermore, for certain questions, we gave multiple agents the same prompt and compared their output. When implementing the BF-DCQO algorithm, we provided our key challenge context and our code files to both ChatGPT and DeepSeek. Both agents gave functional code with a similar logic. However, DeepSeek provided a much more detailed code with MTS and Trotter circuit function updated, while ChatGPT only provided one loop of iterations for the BF-DCQO algorithm. Comparing the two codes, we were able to ask the agents about further details and necessary changes. Through this, we were able to significantly refactor our code from phase 1, mostly just  modifying the Trotter, MTS, population generation, and population selection steps. 

We feel like we have a better understanding about efficient prompting processes from working on the project in this hackathon.

## One Fail Case

One clear instance of AI failure occurred when we asked the model to calculate the Hamiltonian using index slicing. ChatGPT produced a formula that looked correct and ran without error, but it implicitly assumed local (nearest-neighborhood) interactions between two consecutive spins $s_i$ and $s_{i+1}$, similar to a standard Ising model. The hallucination ignores that each spin participates in $O(N)$ long-range autocorrelation terms, so fliiping one spin affects many $C_k$ values. The AI incorrectly computes energy and stills produce reasonable values, but fails under symmetry checks and exhaustive verification for small $N$. In the same spot, it assumed 0-indexation for all the indices in the sum expression for $U(0, T)$, which initially resulted in including energy for the interaction of spins with themselves ($i=j$).

## Context dump / Other

Chengyi? Zaara? Brayden? Jiani? we can rewrite one of the previous parts here and change that one (the effect on learning)

* ample context to generate bf-...
* our challenge breakdown in any phase
* antigravity for Chengyi?
* more options?