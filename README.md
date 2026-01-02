# 🧩 tauPal

**tauPal** is an intelligent AI Architect for [CrewAI](https://crewai.com). Inspired by Google Opal, it transforms natural language descriptions into complete, executable CrewAI architectures. 

Using **Google Gemini**, tauPal helps you design, visualize, and refine complex multi-agent workflows in seconds.

## ✨ Features

- **Natural Language to Architecture:** Describe your workflow (e.g., "Research stock trends and write a report"), and tauPal generates the Agents, Tasks, Tools, and Logic.
- **Strict Schema Enforcement:** Uses a "Zero Hallucination Layer" to ensure generated architectures are valid and strictly typed (Agents, Tasks, ToolConfigs).
- **Interactive Refinement:** 
    - **Chat:** Use natural language to tweak the flow (e.g., "Add a QA agent").
    - **Visual:** Manually edit Agent goals and Task descriptions to instantly update the blueprint.
- **Visual Blueprint:** Renders clear, directed graphs of your AI crew using Graphviz.
- **Code Generation:** Exports ready-to-run `crew.py` Python code, including custom tool placeholders and `Flow` entry points.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Google Gemini API Key
- [Graphviz](https://graphviz.org/download/) installed and added to your system PATH.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/TauAutomation/tauPal.git
    cd tauPal
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the application:
    ```bash
    streamlit run taupal.py
    ```

## 🛠️ Usage

1.  **Enter API Key:** Provide your Gemini API Key in the sidebar.
2.  **Select Model:** Choose your preferred Gemini model (default: `gemini-1.5-flash`).
3.  **Describe Workflow:** Type a description of what you want your AI crew to do.
    > *Example: "Create a marketing crew where a Researcher finds trends on Reddit, a Strategist picks the best angle, and a Copywriter writes a LinkedIn post."*
4.  **Generate:** Click "Generate Architecture".
5.  **Refine:** Use the "Refine / Edit" tab to make changes via chat or manual inputs.
6.  **Export:** Download the generated `crew.py` file from the "Code" tab.

## 📦 Project Structure

- `taupal.py`: Main Streamlit application and Architect logic.
- `requirements.txt`: Python dependencies.

## ❤️ Credits

Built with love by **Tau Automation**.

- **Email:** [tauautomation@gmail.com](mailto:tauautomation@gmail.com)
- **Website:** [https://tauautomation.netlify.app/](https://tauautomation.netlify.app/)

---
*Powered by Google Gemini & Streamlit*
