# auth_utils.py
import json
import bcrypt
import os
from typing import Optional, Dict

USERS_FILE = os.getenv("USERS_FILE", "users.json")

def load_users() -> Dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users: Dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_user(username: str, password: str, role: str = "Reviewer"):
    users = load_users()
    users[username] = {"password": hash_password(password), "role": role}
    save_users(users)

def authenticate(username: str, password: str) -> Optional[Dict]:
    users = load_users()
    u = users.get(username)
    if not u:
        return None
    if check_password(password, u["password"]):
        return {"username": username, "role": u["role"]}
    return None

def get_role(username: str) -> Optional[str]:
    users = load_users()
    u = users.get(username)
    if not u:
        return None
    return u.get("role")
