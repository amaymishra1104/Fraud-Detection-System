# AI-Powered Fraud Detection System

A production-ready machine learning system for detecting fraudulent transactions using advanced classification models.

## Quick Start

### 1. Setup Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Run Application

**Backend API:**
```bash
cd backend
uvicorn main:app --reload
```

**Frontend Dashboard (new terminal):**
```bash
cd frontend
streamlit run app.py
```

## Project Structure

- **backend/**: FastAPI REST API for model inference
- **frontend/**: Streamlit web dashboard for visualization
- **models/**: Trained ML models and scalers
- **data/**: Raw and processed datasets
- **notebooks/**: Jupyter notebooks for analysis and development
- **tests/**: Unit and integration tests

## Key Features

- 🤖 Advanced ML models (XGBoost, scikit-learn)
- 🚀 FastAPI backend with real-time predictions
- 📊 Interactive Streamlit dashboard
- 📈 Comprehensive analytics and reporting
- 🧪 Full test coverage
- 📝 Production-ready code structure

## Technology Stack

| Category | Technologies |
|----------|--------------|
| **Data Processing** | Pandas, NumPy, SciPy |
| **Machine Learning** | scikit-learn, XGBoost |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **Visualization** | Matplotlib, Seaborn |
| **Testing** | pytest, coverage |
| **Code Quality** | Black, Flake8, Pylint, mypy |

## Documentation

- [Setup Guide](./SETUP.md) - Detailed environment setup
- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Model Documentation](./models/README.md) - Model details

## Development

### Code Style
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Run tests
pytest tests/ -v --cov
```

### Contributing
1. Create a feature branch
2. Make changes
3. Run tests and linting
4. Submit pull request

## License

MIT License - See LICENSE file for details

## Contact

For questions or support, contact: [your-email@example.com]
