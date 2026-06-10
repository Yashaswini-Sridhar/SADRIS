# SADRIS — Smart Automated Disaster Response & Intelligence System
## CEP Faculty Hackathon 2026 | CSE Cluster

---

## Quick Start

### 1. Frontend Demo (no setup needed)
Just open `index.html` in any browser. All 6 modules are fully interactive.

### 2. Backend API
```bash
cd sadris/
pip install -r requirements.txt
python app.py
```
Server starts at http://localhost:5000

---

## Module Summary

| # | Module | Files | Tech |
|---|--------|-------|------|
| 1 | AI Risk Prediction | `app.py → DisasterPredictor` | scikit-learn GBM, 12-feature pipeline |
| 2 | Backend APIs | `app.py → Flask routes` | Flask, JWT, REST, OpenAPI-ready |
| 3 | Secure API Layer | `app.py → SecurityLayer` | JWT, AES-256 (Fernet), audit log |
| 4 | Dashboard & Viz | `index.html` | Chart.js, SVG maps, live sensor feed |
| 5 | Resource Optimizer | `app.py → ResourceOptimizer` | LP (linprog/greedy), multi-objective |
| 6 | Quantum-AI | `app.py → QuantumSimulator` | VQE, QAOA, hybrid classical/quantum |

---

## API Reference

### Authentication
```bash
# Login
curl -X POST http://localhost:5000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin@cep.edu","password":"admin123"}'

# Returns: {"access_token": "eyJ...", "role": "admin"}
```

### Risk Prediction
```bash
TOKEN="eyJ..."
curl -X POST http://localhost:5000/api/v1/predict/disaster \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "zone_id": "Z001",
    "sensor_data": {
      "rainfall_mm": 180,
      "wind_speed_kmh": 45,
      "temperature_c": 32,
      "soil_moisture_pct": 88,
      "water_level_pct": 94
    }
  }'
```

### Resource Allocation
```bash
curl -X POST http://localhost:5000/api/v1/resources/allocate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "incidents": [
      {"id":"INC001","type":"Flood","severity":"critical"},
      {"id":"INC002","type":"Wildfire","severity":"warning"}
    ],
    "objective": "min_response_time"
  }'
```

### Quantum Optimization
```bash
# VQE
curl "http://localhost:5000/api/v1/quantum/vqe?qubits=6&iterations=50" \
  -H "Authorization: Bearer $TOKEN"

# QAOA
curl -X POST http://localhost:5000/api/v1/quantum/qaoa \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"qubits": 12, "p_layers": 5}'
```

---

## Demo Users

| User | Password | Role |
|------|----------|------|
| admin@cep.edu | admin123 | admin |
| researcher@cep.edu | research123 | researcher |
| sensor-bot-01 | bot-secret | sensor |

---

## Project Structure

```
sadris/
├── index.html          ← Full frontend demo (all 6 modules)
├── app.py              ← Python backend (Flask + all modules)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## Architecture

```
Browser (index.html)
    │
    ▼
Flask REST API (app.py :5000)
    ├── /auth/*          ← JWT Auth (Module 3)
    ├── /risk/*          ← Zone monitoring (Module 1)
    ├── /predict/*       ← AI Prediction Engine (Module 1)
    ├── /alerts/*        ← Live alerts (Module 2)
    ├── /sensors/*       ← Sensor stream (Module 2)
    ├── /resources/*     ← LP Optimizer (Module 5)
    ├── /quantum/*       ← VQE/QAOA (Module 6)
    └── /security/*      ← AES-256 + Audit (Module 3)
```

---

*SADRIS Faculty Hackathon | CSE Cluster | Cambridge Institute of Technology CEP | June 2026*
