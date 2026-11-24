import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("=== 1. Test Existing Default Developer Credentials ===")
res = client.post("/api/auth/login", json={"username": "developer@soundsense.ai", "password": "SoundSense@Dev2026!", "role": "developer"})
assert res.status_code == 200, f"Failed: {res.text}"
data = res.json()
assert data["success"] is True and data["role"] == "developer"
print(f"  OK: Login succeeded for {data['user']['email']}, name: {data['user']['name']}")

print("\n=== 2. Test Invalid Password ===")
res_bad = client.post("/api/auth/login", json={"username": "developer@soundsense.ai", "password": "BadPassword123!", "role": "developer"})
assert res_bad.status_code == 401
print("  OK: Invalid password properly denied with 401")

print("\n=== 3. Test Create Account (Registration) ===")
reg_email = f"new_engineer_{int(time.time())}@soundsense.ai"
res_reg = client.post("/api/auth/register", json={
    "name": "Sarah Connor",
    "email": reg_email,
    "password": "NeuralDevSecret2026!",
    "role": "developer"
})
assert res_reg.status_code == 200, f"Registration failed: {res_reg.text}"
reg_data = res_reg.json()
assert reg_data["success"] is True and reg_data["role"] == "developer"
assert "Sarah Connor" in reg_data["user"]["name"]
print(f"  OK: Registered new developer: {reg_data['user']['name']} ({reg_data['user']['email']})")

print("\n=== 4. Test Login With Newly Registered Account ===")
res_new_login = client.post("/api/auth/login", json={
    "username": reg_email,
    "password": "NeuralDevSecret2026!",
    "role": "developer"
})
assert res_new_login.status_code == 200, f"New account login failed: {res_new_login.text}"
new_login_data = res_new_login.json()
assert new_login_data["success"] is True and new_login_data["role"] == "developer"
print(f"  OK: New account logged in successfully! Role: {new_login_data['role']}")

print("\n=== 5. Test Duplicate Registration Prevention ===")
res_dup = client.post("/api/auth/register", json={
    "name": "Sarah Connor Clone",
    "email": reg_email,
    "password": "AnotherPassword2026!",
    "role": "developer"
})
assert res_dup.status_code == 400
print("  OK: Duplicate account registration prevented with 400")

print("\n==========================================")
print(" ALL AUTH & REGISTRATION TESTS PASSED!   ")
print("==========================================")
