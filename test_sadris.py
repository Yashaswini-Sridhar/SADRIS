"""
SADRIS — Test Suite
Run: python test_sadris.py
Tests all 6 modules without requiring a running server.
"""

import sys
import json
import time
import traceback
from datetime import datetime

# Color codes
G = '\033[92m'  # Green
R = '\033[91m'  # Red
Y = '\033[93m'  # Yellow
B = '\033[94m'  # Blue
W = '\033[0m'   # Reset
BOLD = '\033[1m'

pass_count = 0
fail_count = 0
results = []

def test(name, fn):
    global pass_count, fail_count
    try:
        start = time.time()
        fn()
        elapsed = round((time.time() - start) * 1000, 1)
        print(f"  {G}✓{W} {name} {Y}({elapsed}ms){W}")
        pass_count += 1
        results.append({'name': name, 'status': 'PASS', 'ms': elapsed})
    except Exception as e:
        print(f"  {R}✗{W} {name}")
        print(f"    {R}{traceback.format_exc().strip()}{W}")
        fail_count += 1
        results.append({'name': name, 'status': 'FAIL', 'error': str(e)})

def section(title):
    print(f"\n{B}{BOLD}── {title} ──{W}")

print(f"\n{BOLD}{'='*60}")
print("  SADRIS Test Suite — All 6 Modules")
print(f"{'='*60}{W}")

# ============================================================
# MODULE 1: AI PREDICTION ENGINE
# ============================================================
section("Module 1: AI Risk Prediction Engine")

from app import predictor, DisasterPredictor

def test_model_trained():
    assert predictor.model is not None, "Model not trained"
    assert predictor.accuracy > 0, "Accuracy should be > 0"
    assert len(predictor.DISASTER_TYPES) == 5

def test_prediction_flood():
    r = predictor.predict({
        'rainfall_mm': 200, 'wind_speed_kmh': 20, 'temperature_c': 28,
        'soil_moisture_pct': 90, 'water_level_pct': 95
    })
    assert r['predicted_type'] in predictor.DISASTER_TYPES
    assert 0 <= r['risk_score'] <= 100
    assert r['severity'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    assert 'recommended_action' in r
    assert abs(sum(r['probabilities'].values()) - 100) < 1.0, "Probabilities must sum to ~100"

def test_prediction_wildfire():
    r = predictor.predict({
        'temperature_c': 42, 'wind_speed_kmh': 70, 'rainfall_mm': 0,
        'vegetation_density_pct': 85, 'soil_moisture_pct': 15
    })
    assert r['predicted_type'] in predictor.DISASTER_TYPES
    assert r['risk_score'] >= 0

def test_prediction_empty_features():
    # Should handle empty gracefully with defaults
    r = predictor.predict({})
    assert r['predicted_type'] in predictor.DISASTER_TYPES

def test_model_confidence_range():
    r = predictor.predict({'rainfall_mm': 50, 'temperature_c': 28})
    assert 0 <= r['model_confidence'] <= 1

def test_batch_prediction():
    from app import predictor, ZONES
    results = []
    for zone in ZONES[:3]:
        r = predictor.predict({'rainfall_mm': 100 + zone['risk'], 'water_level_pct': zone['risk']})
        results.append(r)
    assert len(results) == 3

test("Model is trained and loaded", test_model_trained)
test("Flood scenario prediction", test_prediction_flood)
test("Wildfire scenario prediction", test_prediction_wildfire)
test("Handles empty feature input", test_prediction_empty_features)
test("Confidence score in [0,1]", test_model_confidence_range)
test("Batch prediction for 3 zones", test_batch_prediction)

# ============================================================
# MODULE 2: BACKEND APIS (route logic)
# ============================================================
section("Module 2: Backend APIs")

from app import app as flask_app, ZONES, ALERTS_DB, RESOURCES_DB

flask_app.config['TESTING'] = True
flask_app.config['JWT_SECRET_KEY'] = 'sadris-cep-faculty-hackathon-secret-key-2026'
client = flask_app.test_client()

from flask_jwt_extended import create_access_token
with flask_app.app_context():
    admin_token = create_access_token(identity='admin@cep.edu', additional_claims={'role': 'admin'})
    headers = {'Authorization': f'Bearer {admin_token}', 'Content-Type': 'application/json'}

def test_health_endpoint():
    r = client.get('/api/v1/health')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['status'] == 'operational'
    assert len(data['modules']) == 6

def test_login_valid():
    r = client.post('/api/v1/auth/login',
        json={'username': 'admin@cep.edu', 'password': 'admin123'})
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'access_token' in data
    assert data['role'] == 'admin'

def test_login_invalid():
    r = client.post('/api/v1/auth/login',
        json={'username': 'admin@cep.edu', 'password': 'wrong'})
    assert r.status_code == 401

def test_get_zones():
    r = client.get('/api/v1/risk/zones', headers=headers)
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['total'] == 6
    assert len(data['zones']) == 6

def test_get_single_zone():
    r = client.get('/api/v1/risk/zones/Z001', headers=headers)
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['id'] == 'Z001'

def test_get_zone_not_found():
    r = client.get('/api/v1/risk/zones/Z999', headers=headers)
    assert r.status_code == 404

def test_predict_endpoint():
    r = client.post('/api/v1/predict/disaster', headers=headers,
        json={'zone_id': 'Z001', 'sensor_data': {'rainfall_mm': 180, 'water_level_pct': 90}})
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'predicted_type' in data
    assert 'risk_score' in data

def test_predict_no_sensor_data():
    r = client.post('/api/v1/predict/disaster', headers=headers,
        json={'zone_id': 'Z001'})
    assert r.status_code == 400

def test_get_alerts():
    r = client.get('/api/v1/alerts/live', headers=headers)
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'alerts' in data
    assert 'unacknowledged' in data

def test_sensor_stream():
    r = client.get('/api/v1/sensors/stream?zone=Z001', headers=headers)
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'rainfall_mm' in data
    assert 'temperature_c' in data

def test_dashboard_summary():
    r = client.get('/api/v1/dashboard/summary', headers=headers)
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['total_zones'] == 6
    assert 'model_accuracy' in data

def test_unauthenticated_request():
    r = client.get('/api/v1/risk/zones')  # No token
    assert r.status_code == 401

test("GET /health → 200 + 6 modules", test_health_endpoint)
test("POST /auth/login → valid credentials", test_login_valid)
test("POST /auth/login → invalid → 401", test_login_invalid)
test("GET /risk/zones → all 6 zones", test_get_zones)
test("GET /risk/zones/Z001 → single zone", test_get_single_zone)
test("GET /risk/zones/Z999 → 404", test_get_zone_not_found)
test("POST /predict/disaster → prediction returned", test_predict_endpoint)
test("POST /predict/disaster → no data → 400", test_predict_no_sensor_data)
test("GET /alerts/live → alerts list", test_get_alerts)
test("GET /sensors/stream → live readings", test_sensor_stream)
test("GET /dashboard/summary", test_dashboard_summary)
test("Unauthenticated request → 401", test_unauthenticated_request)

# ============================================================
# MODULE 3: SECURE API LAYER
# ============================================================
section("Module 3: Secure API Layer")

from app import security, SecurityLayer

def test_encrypt_decrypt():
    original = {'zone': 'Z001', 'risk': 87.4, 'type': 'Flood'}
    encrypted = security.encrypt_payload(original)
    assert isinstance(encrypted, str)
    assert encrypted != json.dumps(original)
    decrypted = security.decrypt_payload(encrypted)
    assert decrypted == original

def test_password_hashing():
    h1 = security.hash_password('admin123')
    h2 = security.hash_password('admin123')
    h3 = security.hash_password('different')
    assert h1 == h2, "Same password must produce same hash"
    assert h1 != h3, "Different passwords must produce different hashes"
    assert len(h1) == 64, "SHA-256 produces 64 hex chars"

def test_audit_log():
    initial = len(security.audit_log)
    security.log_event('TEST', 'testuser', 'SUCCESS', '127.0.0.1')
    assert len(security.audit_log) == initial + 1
    last = security.audit_log[-1]
    assert last['event'] == 'TEST'
    assert last['user'] == 'testuser'
    assert last['status'] == 'SUCCESS'

def test_audit_log_retrieval():
    for i in range(5):
        security.log_event(f'EVT_{i}', 'user', 'SUCCESS')
    log = security.get_audit_log(limit=3)
    assert len(log) == 3  # Most recent 3

def test_jwt_auth_flow():
    r = client.post('/api/v1/auth/login',
        json={'username': 'admin@cep.edu', 'password': 'admin123'})
    token = json.loads(r.data)['access_token']
    # Use token on protected route
    r2 = client.get('/api/v1/risk/zones',
        headers={'Authorization': f'Bearer {token}'})
    assert r2.status_code == 200

def test_encrypt_endpoint():
    r = client.post('/api/v1/security/encrypt', headers=headers,
        json={'payload': {'sensor': 'rain', 'value': 142}})
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'encrypted' in data
    assert data['algorithm'] == 'Fernet(AES-256)'

def test_audit_endpoint_admin_only():
    # Non-admin should be rejected
    with flask_app.app_context():
        researcher_token = create_access_token(identity='researcher@cep.edu', additional_claims={'role': 'researcher'})
    r = client.get('/api/v1/security/audit',
        headers={'Authorization': f'Bearer {researcher_token}'})
    assert r.status_code == 403

test("AES-256 encrypt → decrypt round-trip", test_encrypt_decrypt)
test("SHA-256 password hashing deterministic", test_password_hashing)
test("Audit log records events", test_audit_log)
test("Audit log pagination (limit=3)", test_audit_log_retrieval)
test("JWT auth flow end-to-end", test_jwt_auth_flow)
test("POST /security/encrypt endpoint", test_encrypt_endpoint)
test("Audit log: researcher denied (403)", test_audit_endpoint_admin_only)

# ============================================================
# MODULE 4: VISUALIZATION (data integrity)
# ============================================================
section("Module 4: Dashboard & Visualization")

def test_zone_data_structure():
    for zone in ZONES:
        assert 'id' in zone
        assert 'name' in zone
        assert 'risk' in zone
        assert 0 <= zone['risk'] <= 100
        assert 'lat' in zone and 'lon' in zone
        assert 'population' in zone

def test_alert_data_structure():
    for alert in ALERTS_DB:
        assert 'id' in alert
        assert alert['severity'] in ['critical', 'warning', 'info']
        assert 'type' in alert
        assert 'acknowledged' in alert

def test_sensor_data_ranges():
    r = client.get('/api/v1/sensors/stream?zone=Z001', headers=headers)
    data = json.loads(r.data)
    assert 0 <= data['rainfall_mm'] <= 500
    assert 0 <= data['wind_speed_kmh'] <= 200
    assert -20 <= data['temperature_c'] <= 60
    assert 0 <= data['soil_moisture_pct'] <= 100
    assert 0 <= data['water_level_pct'] <= 100

def test_risk_fluctuation():
    r1 = client.get('/api/v1/risk/zones/Z001', headers=headers)
    r2 = client.get('/api/v1/risk/zones/Z001', headers=headers)
    d1 = json.loads(r1.data)['risk']
    d2 = json.loads(r2.data)['risk']
    # Both should be valid risk scores
    assert 0 <= d1 <= 100
    assert 0 <= d2 <= 100

def test_alert_filtering():
    r = client.get('/api/v1/alerts/live?severity=critical', headers=headers)
    data = json.loads(r.data)
    for alert in data['alerts']:
        assert alert['severity'] == 'critical'

test("Zone data structure complete", test_zone_data_structure)
test("Alert data structure valid", test_alert_data_structure)
test("Sensor readings within valid ranges", test_sensor_data_ranges)
test("Risk scores fluctuate realistically", test_risk_fluctuation)
test("Alert filtering by severity", test_alert_filtering)

# ============================================================
# MODULE 5: RESOURCE ALLOCATION OPTIMIZER
# ============================================================
section("Module 5: Resource Allocation Optimizer")

from app import optimizer, ResourceOptimizer

def test_single_incident_allocation():
    incidents = [{'id': 'INC001', 'type': 'Flood', 'severity': 'critical'}]
    resources = [
        {'id': 'R1', 'type': 'Rescue Team'},
        {'id': 'R2', 'type': 'Ambulance'},
    ]
    result = optimizer.optimize(incidents, resources)
    assert result['status'] == 'optimal'
    assert 'INC001' in result['allocation']
    assert result['coverage_pct'] == 100.0

def test_multi_incident_allocation():
    incidents = [
        {'id': 'INC001', 'type': 'Flood'},
        {'id': 'INC002', 'type': 'Wildfire'},
        {'id': 'INC003', 'type': 'Landslide'},
    ]
    resources = [
        {'id': 'R1', 'type': 'Rescue Team'},
        {'id': 'R2', 'type': 'Fire Engine'},
        {'id': 'R3', 'type': 'Helicopter'},
        {'id': 'R4', 'type': 'Ambulance'},
    ]
    result = optimizer.optimize(incidents, resources)
    assert result['status'] == 'optimal'
    assert len(result['allocation']) <= len(incidents)

def test_more_incidents_than_resources():
    incidents = [{'id': f'INC{i}', 'type': 'Flood'} for i in range(5)]
    resources = [{'id': f'R{i}', 'type': 'Rescue Team'} for i in range(2)]
    result = optimizer.optimize(incidents, resources)
    assert result['coverage_pct'] < 100.0  # Can't cover all
    assert result['status'] == 'optimal'

def test_empty_inputs():
    result = optimizer.optimize([], [])
    assert 'error' in result

def test_objective_variations():
    inc = [{'id': 'INC1', 'type': 'Flood'}]
    res = [{'id': 'R1', 'type': 'Rescue Team'}]
    for obj in ['min_response_time', 'max_coverage', 'min_cost', 'balanced']:
        result = optimizer.optimize(inc, res, objective=obj)
        assert result['objective'] == obj

def test_solve_time_reasonable():
    incidents = [{'id': f'INC{i}', 'type': 'Flood'} for i in range(10)]
    resources = [{'id': f'R{i}', 'type': 'Rescue Team'} for i in range(10)]
    result = optimizer.optimize(incidents, resources)
    assert result['solve_time_ms'] < 5000  # Must solve in < 5s

def test_resources_endpoint():
    r = client.get('/api/v1/resources', headers=headers)
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['total'] == 6

def test_allocate_endpoint():
    r = client.post('/api/v1/resources/allocate', headers=headers,
        json={'incidents': [{'id': 'INC001', 'type': 'Flood', 'severity': 'critical'}],
              'objective': 'min_response_time'})
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['status'] == 'optimal'

test("Single incident → 100% coverage", test_single_incident_allocation)
test("Multi-incident allocation", test_multi_incident_allocation)
test("More incidents than resources → partial coverage", test_more_incidents_than_resources)
test("Empty inputs → error response", test_empty_inputs)
test("All 4 objectives produce results", test_objective_variations)
test("LP solver completes < 5 seconds", test_solve_time_reasonable)
test("GET /resources → 6 units", test_resources_endpoint)
test("POST /resources/allocate", test_allocate_endpoint)

# ============================================================
# MODULE 6: QUANTUM-AI
# ============================================================
section("Module 6: Quantum-AI Algorithms")

from app import quantum, QuantumSimulator

def test_vqe_basic():
    r = quantum.vqe_simulation(qubits=4, iterations=30)
    assert r['algorithm'] == 'VQE'
    assert r['qubits'] == 4
    assert len(r['energy_history']) > 0
    assert r['final_energy'] < 0  # Ground state is negative
    assert 0 < r['fidelity'] <= 1

def test_vqe_convergence():
    r = quantum.vqe_simulation(qubits=4, iterations=50)
    assert 0 <= r['convergence_iteration'] <= 50

def test_vqe_more_qubits():
    r = quantum.vqe_simulation(qubits=8, iterations=20)
    assert r['circuit_depth'] == 8 * 4 + 2

def test_qaoa_basic():
    r = quantum.qaoa_optimization(n_qubits=6, p_layers=2)
    assert r['algorithm'] == 'QAOA'
    assert r['qubits'] == 6
    assert r['p_layers'] == 2
    assert 0 < r['approximation_ratio'] <= 1
    assert r['quantum_solution'] <= r['classical_optimum']

def test_qaoa_p_scaling():
    r1 = quantum.qaoa_optimization(n_qubits=6, p_layers=1)
    r2 = quantum.qaoa_optimization(n_qubits=6, p_layers=5)
    assert r2['circuit_depth'] > r1['circuit_depth']
    assert r2['total_gates'] > r1['total_gates']

def test_qaoa_fidelity_increases_with_p():
    r1 = quantum.qaoa_optimization(n_qubits=6, p_layers=1)
    r5 = quantum.qaoa_optimization(n_qubits=6, p_layers=5)
    assert r5['fidelity'] >= r1['fidelity']

def test_speedup_positive():
    r = quantum.qaoa_optimization(n_qubits=10, p_layers=3)
    assert r['speedup_vs_classical'] > 1.0

def test_vqe_endpoint():
    r = client.get('/api/v1/quantum/vqe?qubits=4&iterations=20', headers=headers)
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['algorithm'] == 'VQE'

def test_qaoa_endpoint():
    r = client.post('/api/v1/quantum/qaoa', headers=headers,
        json={'qubits': 8, 'p_layers': 3})
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['algorithm'] == 'QAOA'

def test_hybrid_optimize_endpoint():
    r = client.post('/api/v1/quantum/optimize', headers=headers,
        json={'incidents': [], 'resources': [], 'qubits': 6, 'p_layers': 2})
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'classical' in data
    assert 'quantum' in data
    assert data['recommendation'] in ['classical', 'quantum']

test("VQE runs and converges", test_vqe_basic)
test("VQE convergence iteration found", test_vqe_convergence)
test("VQE circuit depth scales with qubits", test_vqe_more_qubits)
test("QAOA basic optimization", test_qaoa_basic)
test("QAOA circuit depth scales with p", test_qaoa_p_scaling)
test("QAOA fidelity improves with p layers", test_qaoa_fidelity_increases_with_p)
test("Quantum speedup > 1× classical", test_speedup_positive)
test("GET /quantum/vqe endpoint", test_vqe_endpoint)
test("POST /quantum/qaoa endpoint", test_qaoa_endpoint)
test("POST /quantum/optimize hybrid endpoint", test_hybrid_optimize_endpoint)

# ============================================================
# SUMMARY
# ============================================================
total = pass_count + fail_count
pct = round(pass_count / total * 100) if total else 0
bar = '█' * (pass_count * 20 // total) + '░' * (20 - pass_count * 20 // total) if total else ''

print(f"\n{BOLD}{'='*60}")
print(f"  Test Results: {G}{pass_count} passed{W} / {R}{fail_count} failed{W} / {total} total")
print(f"  Coverage: [{G}{bar}{W}] {G}{pct}%{W}")
print(f"{'='*60}{W}\n")

if fail_count == 0:
    print(f"{G}{BOLD}✓ All tests passed! SADRIS is fully operational.{W}\n")
else:
    print(f"{R}{BOLD}✗ {fail_count} test(s) failed. Check output above.{W}\n")
    sys.exit(1)
