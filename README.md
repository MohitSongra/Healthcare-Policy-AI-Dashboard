# 🏥 GovHealth AI: Hybrid Healthcare Policy AI Dashboard

A premium, AI-powered decision support system designed for government medical supply chain planning and health fund allocation. This dashboard combines **machine learning forecasting**, **Retrieval-Augmented Generation (RAG)**, and **interactive scenario simulators** to assist policy makers in analyzing risk metrics and optimizing healthcare budgets across Indian states.

---

## 📸 Dashboard Preview

Here is a preview of the interactive dashboard running locally:

![GovHealth AI Dashboard Screenshot](images/dashboard_screenshot.webp)

---

## 🌟 Key Features

### 1. Interactive India Health Risk Map
*   **Geographic Risk Analysis**: Overlays state-level metrics on an interactive map of India powered by Plotly `open-street-map` style maps.
*   **Dynamic Metrics**: Policy makers can toggle the map view between:
    *   *Priority Score* (comprehensive urgency rating)
    *   *Health Budget* (allocated state funds in Rs. Crores)
    *   *Doctor Density* (doctors per 1,000 population)
    *   *Hospital Beds* (beds per 1,000 population)
    *   *Vaccine Coverage* (percentage of immunized population)
    *   *Infrastructure Gap Score* (relative infrastructure need index)

### 2. Predictive Analytics Engine
*   **Budget Projections**: Trains a `GradientBoostingRegressor` to project 2028–2029 budget allocations based on historical state metrics.
*   **Resource Projections**: Trains a multi-output `RandomForestRegressor` to forecast future capacities for essential resources: total hospitals, total doctors, vaccine doses, ICU beds, hospital beds per 1,000, and nurse counts.

### 3. Scenario & Allocation Sandbox
*   **Scenario Simulator**: Policy makers can choose a state and slide parameters (budget increase %, added doctors, hospital beds, ICU beds, vaccine coverage gain) to immediately view projected outcomes on the priority score, doctor-to-population ratio, and infrastructure gap.
*   **National Fund Allocation Optimizer**: Distributes a custom national fund pool among all states and UTs based on customizable weights for state priority, population, disease burden, and infrastructure gap. Includes a minimum floor allocation per state, a comparative visualization chart, and a downloadable CSV planner.

### 4. AI Policy Advisor (RAG chat)
*   **FAISS Vector Database**: Indexes national health policies and guidelines (from PDFs, PPTs, and text documents) using semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
*   **Ollama Integration**: Powers offline policy chats using locally hosted models (e.g. `mistral:7b`) to retrieve contextually relevant guidelines and policy advise.

---

## 🛠️ Windows Installation & Setup

Ensure you have **Python 3.10+** and [Ollama](https://ollama.com/) installed and running.

### 1. Clone & Navigate to Workspace
```powershell
git clone https://github.com/MohitSongra/Healthcare-Policy-AI-Dashboard.git
cd Healthcare-Policy-AI-Dashboard/Hybrid-healthcare-policy-ai
```

### 2. Set Up a Virtual Environment
We recommend using **`uv`** for lightning-fast setup:
```powershell
# Install uv if you don't have it
pip install uv

# Create and activate virtual env
uv venv --python 3.11
.venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
uv pip install -r requirements.txt
```

### 4. Setup LLM (Ollama)
Pull the default Mistral model (make sure the Ollama application is running in the background):
```powershell
ollama pull mistral:7b
```

### 5. Run the Pipelines & Start App
```powershell
# 1. Train the ML models
python train_predictors.py

# 2. Build the FAISS Vector Database
python rag_pdf_setup.py

# 3. Launch the Streamlit dashboard
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🗂️ Project Directory Structure

```
├── app.py                      # Main Streamlit dashboard interface
├── recommendation_engine.py    # Core logic for priorities, allocation, and RAG search
├── train_predictors.py         # ML pipeline training models for 2028-29 projections
├── prepare_dataset.py          # Data cleansing and structuring script
├── rag_pdf_setup.py            # PDF document parser and FAISS database initializer
├── requirements.txt            # Python dependencies
├── data/                       # Raw healthcare statistics & PDF guidelines
├── images/                     # Project screenshots & images
└── vector_store/               # Local FAISS embedding index
```

---

## 🚀 Technology Stack
*   **Frontend UI**: Streamlit, Plotly
*   **Machine Learning**: Scikit-Learn
*   **Vector Database & RAG**: FAISS, LangChain, HuggingFace Transformers
*   **Local LLM Engine**: Ollama (Mistral 7B)
