## 🛠️ Technologies Used

- Frontend: Streamlit

- Backend: FastAPI, Uvicorn

- LLM Provider: Groq

- Agent Framework: LangGraph, LangChain (for tools)

- Data Validation: Pydantic

- External APIs:

    - Google Places API

    - OpenWeatherMap API

    - Tavily Search API

    - ExchangeRate-API (or similar)

## ⚙️ Setup and Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/maddy-abhishek/AI-TRIP-PLANNER.git
   ```
2. Create virtual environment:

   ```bash
   uv init
   ```

   ```bash
   uv venv env
   ```

   ```bash
   env\Scripts\activate.bat
   ```

3. Install the required dependencies:

   ```bash
   uv pip install -r requirements.txt
   ```

4. Add all your API keys to this file:

GROQ_API_KEY=""

GOOGLE_API_KEY=""

GPLACES_API_KEY=""

TAVILAY_API_KEY=""

OPENWEATHERMAP_API_KEY=""


## ▶️ How to Use

You need to run two processes in two separate terminals: the Backend API and the Frontend App.

Terminal 1: Run the Backend (FastAPI)

```bash
uvicorn main:app --reload --port 8000
```

Terminal 2: Run the Frontend (Streamlit)

```bash
streamlit run app.py
```