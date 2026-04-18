"""Tests unitaires — structure de l'API FastAPI"""
import ast
import pytest
from pathlib import Path

API_PATH = Path(__file__).parent.parent / "src" / "api.py"

def test_api_file_exists():
    assert API_PATH.exists(), "src/api.py introuvable"

def test_api_syntax_valid():
    """La syntaxe Python de api.py doit être valide."""
    with open(API_PATH) as f:
        source = f.read()
    try:
        ast.parse(source)
    except SyntaxError as e:
        pytest.fail(f"Erreur de syntaxe dans api.py : {e}")

def test_api_contains_required_endpoints():
    """Vérifier que les endpoints obligatoires sont définis."""
    with open(API_PATH) as f:
        source = f.read()
    required_routes = [
        '/health',
        '/models',
        '/predict/ensemble',
        '/predict/ensemble/scan',
    ]
    for route in required_routes:
        assert route in source, f"Endpoint manquant : {route}"

def test_api_uses_fastapi():
    with open(API_PATH) as f:
        source = f.read()
    assert 'FastAPI' in source
    assert 'from fastapi' in source

def test_api_has_cors():
    with open(API_PATH) as f:
        source = f.read()
    assert 'CORSMiddleware' in source, "CORSMiddleware non configuré"

def test_api_ensemble_members_defined():
    with open(API_PATH) as f:
        source = f.read()
    assert 'ENSEMBLE_MEMBERS' in source
    assert 'patchtst' in source
    assert 'lstm_attention' in source
    assert 'cnn_transformer' in source
