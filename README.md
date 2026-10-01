# DebtOx

**DebtOx** is an open-source static software analysis and technical debt estimation platform for Java codebases. It combines Java Abstract Syntax Tree (AST) parsing, software metrics extraction, machine learning-based code smell prediction, and parametric technical debt quantification into a unified CLI and web dashboard.

---

## Features

- **Static AST & Metrics Extraction**: Parses Java source files without requiring compilation or build tools (Maven/Gradle). Computes Chidamber & Kemerer (CK) metrics (WMC, CBO, RFC, LCOM5, DIT, NOC), cyclomatic complexity, Halstead metrics, and lines of code (SLOC/LLOC).
- **Git Evolution Mining**: Mines Git commit history, additions, deletions, hunks, and net churn to measure component volatility and maintenance effort over time.
- **Code Smell Detection**: Detects class- and method-level code smells using trained machine learning models with deterministic rule-based heuristic fallbacks.
- **Technical Debt Principal (TDP)**: Estimates remediation effort in hours using a COCOMO II-based maintenance model and converts effort to financial cost.
- **Technical Debt Interest (TDI)**: Computes the ongoing maintenance tax based on excess historical code churn in smelly components compared to clean peers.
- **Risk Prioritization**: Computes a normalized composite risk score (0–100) balancing smell confidence, remediation effort (TDP), historical churn (TDI), and commit volatility.
- **Explainability (SHAP)**: Provides feature attribution insights showing which metrics drove each prediction, accompanied by natural-language refactoring guidance.
- **Dual Interfaces**: Interactive web dashboard (React 19 + TypeScript + Tailwind CSS) and terminal CLI (`./debt-ox`).
- **Flexible Deployment**: Runs locally via unified FastAPI static hosting or containerized via Docker and Docker Compose.

---

## Supported Code Smells

| Smell | Granularity | Description | Detection Method |
| :--- | :--- | :--- | :--- |
| **God Class** | Class | Large, uncohesive classes centralizing excessive system responsibility. | ML Model / Heuristic Fallback |
| **Data Class** | Class | Classes that store data fields with accessors but lack substantial logic. | ML Model / Heuristic Fallback |
| **Brain Class** | Class | Complex, uncohesive classes accumulating critical business logic. | Deterministic Heuristic Baseline |
| **Long Method** | Method | Excessively long methods with high cyclomatic complexity and deep nesting. | ML Model / Heuristic Fallback |
| **Feature Envy** | Method | Methods that access data of other classes more than their own. | ML Model / Heuristic Fallback |
| **Brain Method** | Method | Long, complex methods executing multiple branches and deep nested blocks. | Deterministic Heuristic Baseline |

---

## How It Works

```text
Java Codebase / Git Repository
         │
         ▼
   Static Analysis Engine (AST Parsing + CK / Complexity Metrics)
         │
         ├──► Git History Miner (Commit history, churn & volatility)
         │
         ▼
   Code Smell Detection (ML Models + Heuristic Fallbacks)
         │
         ▼
   Technical Debt Engine (TDP Hours/Cost + TDI Churn + Risk Scoring)
         │
         ▼
   Interfaces (FastAPI REST API, React Web Dashboard, CLI Runner)
```

1. **AST & Metrics**: The analyzer parses `.java` files with `javalang`, extracting structural, size, and complexity metrics per class and method.
2. **Git Evolution**: If Git history is available, commit logs are mined for line additions, deletions, hunks, and net churn.
3. **Smell Inference**: Extracted metrics are fed to trained classifiers (Random Forest, XGBoost, LightGBM, etc.). If serialized model weights are unavailable, the system automatically uses deterministic metric threshold baselines.
4. **Debt & Risk Quantification**: TDP hours are estimated using component size, complexity, and smell type. TDI is calculated from excess historical churn. A composite risk score (0–100) is assigned to prioritize remediation.
5. **Presentation**: Results are delivered via the web dashboard, CLI tables, or REST API endpoints.

---

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, Pydantic v2
- **AST Parsing & Git Mining**: Javalang, GitPython
- **Machine Learning & Analytics**: Scikit-Learn, XGBoost, LightGBM, SHAP, Pandas, NumPy, SciPy
- **CLI**: Typer, Rich
- **Frontend**: React 19, TypeScript, Vite 6, Tailwind CSS, Lucide Icons
- **Testing**: Pytest, HTTPX
- **Containerization**: Docker, Docker Compose

---

## Project Structure

```text
debtox/
├── analyzer/               # AST parsing, CK metrics, and Git churn miner
│   ├── extraction/         # Metric extraction engine
│   ├── git/                # Git commit & history miner
│   ├── java/               # Java AST parser (javalang)
│   └── metrics/            # Structural & complexity metric calculators
├── backend/                # FastAPI application
│   ├── app/
│   │   ├── api/v1/         # REST API routes (/analyze, /analyses, /health)
│   │   ├── core/           # Configuration & settings (Pydantic)
│   │   ├── debt/           # TDP, TDI, and risk prioritization engines
│   │   ├── explainability/ # SHAP feature attributions
│   │   ├── ml/             # Smell inference engine
│   │   ├── schemas/        # Pydantic data models & request/response schemas
│   │   └── services/       # Asynchronous analysis orchestration
│   └── tests/              # Pytest unit and integration test suite
├── cli/                    # CLI commands implemented with Typer & Rich
├── frontend/               # React 19 + TypeScript + Tailwind CSS web app
│   └── src/                # Components, pages, services, types
├── ml/                     # ML training, data loading, and pipeline modules
├── models/                 # Pre-trained model directory placeholder
├── storage/                # Local runtime analysis storage
├── debt-ox                 # Shell script CLI launcher
├── docker-compose.yml      # Multi-container orchestration (API + UI)
├── Dockerfile              # Unified multi-stage container build
├── Makefile                # Developer build, test, and run automation
└── requirements.txt        # Python backend dependencies
```

---

## Prerequisites

- **Python**: 3.10 or higher (Python 3.11 recommended)
- **Node.js**: 18 or higher & npm (for frontend building)
- **Git**: Installed and available in PATH (for repository mining)
- **Docker & Docker Compose**: (Optional, for containerized execution)

---

## Installation & Quick Start

### 1. Local Setup

```bash
# Clone the repository
git clone <repository-url>
cd debtox

# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install frontend dependencies and build the UI bundle
cd frontend
npm install
npm run build
cd ..

# Run the unified platform (FastAPI serves both API and Web UI on port 8000)
python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

Alternatively, using `make`:

```bash
make install        # Install backend and frontend dependencies
make build-frontend # Build production frontend bundle
make run            # Run unified server on :8000
```

### 2. Development Mode (Hot Reload)

For active frontend and backend development:

```bash
# Terminal 1 — Backend API (runs on http://localhost:8000, Swagger at /docs)
make run-api

# Terminal 2 — Frontend Dev Server (runs on http://localhost:5173 with Vite HMR)
make run-ui
```

### 3. Docker Compose Setup

Run the full stack in isolated containers:

```bash
# Build and start services in the background
docker compose up -d --build

# View logs
docker compose logs -f

# Stop services
docker compose down
```

- **Web Dashboard**: `http://localhost:3000`
- **Backend API & Swagger Docs**: `http://localhost:8000/docs`

---

## Environment Variables

Configuration is loaded from environment variables or a `.env` file at the project root. Copy `.env.example` to get started:

```bash
cp .env.example .env
```

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PROJECT_NAME` | `DebtOx` | Application display name |
| `VERSION` | `1.0.0` | Application version |
| `API_V1_PREFIX` | `/api/v1` | Base prefix for API endpoints |
| `DATABASE_URL` | `sqlite:///debtox.db` | Database connection URL |
| `MAX_REPOSITORY_SIZE_MB` | `500` | Maximum repository size allowed for analysis |
| `MAX_FILES` | `10000` | Maximum total files scanned per run |
| `MAX_JAVA_FILES` | `5000` | Maximum Java source files parsed per run |
| `MAX_ANALYSIS_TIME_SECONDS` | `600` | Analysis timeout limit in seconds |
| `MAX_HISTORY_COMMITS` | `500` | Maximum Git commits mined for churn |
| `COCOMO_A` | `2.94` | COCOMO II multiplicative effort parameter |
| `COCOMO_E` | `1.05` | COCOMO II scale factor exponent |
| `DEFAULT_DEVELOPER_HOURLY_RATE` | `65.0` | Developer hourly rate (USD) for debt cost estimation |

---

## Usage

### Web Dashboard

1. Open **[http://localhost:8000](http://localhost:8000)**.
2. Go to the **Analyze** page.
3. Enter a local folder path (e.g., `/path/to/java/project`) or a remote Git URL.
4. Select the analysis mode:
   - **Quick**: Static AST parsing and metrics snapshot only.
   - **Full**: AST metrics + Git history churn and TDI analysis.
   - **Research**: Full analysis with complete SHAP feature attributions.
5. Click **Start Analysis**. Once complete, browse detected smells, view TDP/TDI metrics, inspect high-risk components, and click any component to view detailed SHAP explanations.

### Command Line Interface (CLI)

DebtOx includes an executable CLI script (`./debt-ox`):

```bash
# Make executable
chmod +x debt-ox

# Display version information
./debt-ox version

# Analyze a local Java repository
./debt-ox analyze /path/to/java/project

# Analyze with tabular output (default) or JSON output
./debt-ox analyze /path/to/java/project --format table
./debt-ox analyze /path/to/java/project --format json

# Analyze in quick mode (skips Git churn history)
./debt-ox analyze /path/to/java/project --mode quick

# Inspect SHAP feature attributions for a completed analysis
./debt-ox explain <analysis_id> --top 5

# Export analysis results to a JSON file
./debt-ox report <analysis_id> --output report.json
```

---

## API Reference

When the backend is running, interactive API documentation is available at **[http://localhost:8000/docs](http://localhost:8000/docs)** (Swagger UI) and **[http://localhost:8000/redoc](http://localhost:8000/redoc)** (ReDoc).

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health status and loaded model count |
| `GET` | `/api/v1/version` | Version information |
| `POST` | `/api/v1/analyze` | Submit repository for asynchronous analysis (returns job ID) |
| `GET` | `/api/v1/analyses` | List recent analysis jobs |
| `GET` | `/api/v1/analyses/{id}` | Get complete analysis summary and component details |
| `GET` | `/api/v1/analyses/{id}/summary` | Get analysis summary without component arrays |
| `GET` | `/api/v1/analyses/{id}/components` | Filter components by smell name, granularity, or risk score |
| `GET` | `/api/v1/analyses/{id}/smells` | Aggregated smell frequency counts |
| `GET` | `/api/v1/analyses/{id}/debt` | TDP, TDI, total debt hours, and estimated cost breakdown |
| `GET` | `/api/v1/analyses/{id}/explanations` | Component-level SHAP feature attributions |
| `GET` | `/api/v1/models` | Metadata of loaded ML model artifacts |

---

## Testing

The test suite validates AST metric extraction, debt calculation engines, API routing, and end-to-end analysis:

```bash
# Run all tests
pytest backend/tests/ -v

# Or using Makefile
make test
```

### Test Coverage

- `test_metrics.py`: Verifies AST parsing with `javalang`, CK metrics computation (WMC, CBO, RFC, LCOM5), and cyclomatic complexity.
- `test_debt_engines.py`: Tests COCOMO II TDP effort calculations, longitudinal TDI churn tax, and composite risk scoring.
- `test_api_integration.py`: Validates FastAPI route responses, health checks, and background task queuing using `TestClient`.
- `test_e2e_pipeline.py`: Tests the full pipeline from repository loading to metric extraction, smell inference, and report generation.

---

## Limitations

- **Language Support**: DebtOx currently analyzes Java source files (`.java`). Other programming languages are not supported.
- **Java Parser**: AST parsing uses `javalang` (supporting Java 8 through 17 syntax). Certain newer language features (e.g., preview switch pattern matching) are skipped gracefully without aborting analysis.
- **Git History Dependency**: Longitudinal Technical Debt Interest (TDI) requires a valid Git repository with commit history. When analyzing plain source folders without Git history, TDI reports zero and the system defaults to snapshot mode.
- **ML Fallback**: If serialized model checkpoints (`models/*.joblib`) are omitted or not yet trained, DebtOx automatically switches to deterministic metric threshold baselines, ensuring the platform remains fully functional.

---

## Research Note

DebtOx was developed as part of research into empirical machine learning techniques for code smell detection, parametric software maintenance effort modeling, and longitudinal technical debt tracking.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
