# 🧠 MirrorMind: Self-Refining AI Dialogue Engine

MirrorMind is an experimental framework built to achieve higher reliability and precision in large language model (LLM) outputs through **recursive self-criticism** and **multi-agent debate**.

Instead of relying on a single, potentially flawed generation, MirrorMind orchestrates a structured, multi-round debate where specialized AI 'Critics' rigorously assess a draft and guide a 'Reviser' agent toward consensus and accuracy.

This project is implemented using `transformers`, `sentence-transformers`, and `streamlit` for an interactive, real-time visualization of the AI's internal refinement process.

## ✨ Core Architecture: Tournament & Refinement

The MirrorMind architecture is designed to combat LLM weaknesses like factual inaccuracies, logical inconsistencies, and weak adherence to complex instructions.

### 1. The Proposer & Tournament Selection (Best-of-N)

The process begins with the **Proposer Agent** generating multiple candidate answers ($N$ candidates). These candidates are then scored for semantic relevance to the original prompt using the **Embedder**. The candidate with the highest relevance score is selected as the initial draft, ensuring a strong, relevant starting point.

### 2. The Critic Ensemble (Specialized Personas)

The initial draft is sent to an ensemble of specialized critic personas. Each persona is engineered with a unique focus to provide diverse and rigorous feedback:

* **The Logician:** Focuses on *Logical Consistency* and *Mathematical Accuracy* (e.g., counting, arithmetic).
* **The Skeptic:** Focuses on *Factuality* and *Evidence* (demanding proof and identifying potential hallucinations).
* **The Editor:** Focuses on *Clarity*, *Structure*, and *Style*.

Each Critic returns a structured JSON output containing a `Score` (0-10) and a detailed `Critique`.

### 3. The Metacritic Strategy

All critiques (including optional human input) are aggregated by the **Metacritic Agent**, which synthesizes the diverse feedback into a concise, numbered **Action Plan** for improvement.

### 4. The Reviser

The **Reviser Agent** uses the original prompt, the current draft, and the `Action Plan` to generate a new, more refined answer.

### 5. Recursive Convergence

The new answer replaces the old draft, and the process repeats until one of two conditions is met:

* **Quality Threshold:** The average critic score exceeds a predefined threshold (e.g., 9.0/10).
* **Stagnation:** The semantic similarity between the current and revised answer is too high, indicating that further revision yields no meaningful change.

## 🚀 Getting Started

### Prerequisites

You need Python 3.8+ and the necessary libraries. This project uses large pre-trained models, so a CUDA-enabled GPU is highly recommended for faster performance.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ant1pozitive/mirror-mind.git
    cd MirrorMind
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

Ensure both `mirror_mind.py` (the backend logic) and `streamlit_app.py` (the frontend) are in the same directory.

```bash
streamlit run streamlit_app.py -- --web
```

The application will launch in your web browser, typically at `http://localhost:8501`.

## 🎨 User Interface (UI/UX)

The Streamlit application is designed to make the invisible internal debate visible and engaging:

  * **Dialogue Interface:** The output is styled using **Chat Bubbles**, giving each Agent (Proposer, Critic, Reviser) a distinct visual presence in the discussion.
  * **Progress Dashboard:** A chart tracks the **Average Quality Score** and **Novelty (semantic change)** across rounds, providing real-time metrics on the debate's effectiveness.
  * **Diff Viewer:** An interactive expander shows a word-level diff, highlighting deleted text (red strikethrough) and added text (green bold) between the draft and the revision, offering full transparency into the changes made by the Reviser.
  * **Human-in-the-Loop:** A dedicated sidebar input allows the user to inject their own critique, which is treated as high-priority feedback by the critics in the first round.

## ⚙️ Configuration

Key parameters can be adjusted directly in the Streamlit sidebar:

| Parameter | Description | Default Model/Value |
| :--- | :--- | :--- |
| **Proposer Model** | The core model used for generating drafts and revisions. | `google/flan-t5-base` (or `large`) |
| **Critic Models** | Models used by the specialized critics. Can be different from the Proposer. | `google/flan-t5-base` (or specified list) |
| **Embedding Model** | Model for semantic scoring (Tournament and Convergence checks). | `sentence-transformers/all-MiniLM-L6-v2` |
| **Max Iterations** | The maximum number of debate rounds to run. | 5 |
| **Tournament Candidates** | The $N$ value for the Best-of-N selection. | 3 |
| **Device** | CUDA device ID if a GPU is available. | `None` (auto-detect) |

## License

This project is open-source and available under the MIT License.
