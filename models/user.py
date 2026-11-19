import hashlib
import os
import binascii
import json
from pathlib import Path
from typing import Optional, Dict

DB_PATH = Path("instance/users.json")

def hash_password(password: str) -> str:
    """
    PBKDF2-SHA256 password hashing with a cryptographically secure random salt (Section 4.11).
    Stored as: salt_hex$hash_hex
    """
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"{binascii.hexlify(salt).decode('ascii')}${binascii.hexlify(dk).decode('ascii')}"

def verify_password(stored_hash: str, provided_password: str) -> bool:
    """Verifies a password against the stored PBKDF2-SHA256 salt and hash."""
    try:
        salt_hex, hash_hex = stored_hash.split("$")
        salt = binascii.unhexlify(salt_hex)
        expected_hash = binascii.unhexlify(hash_hex)
        actual_hash = hashlib.pbkdf2_hmac("sha256", provided_password.encode("utf-8"), salt, 100_000)
        return actual_hash == expected_hash
    except Exception:
        return False

def normalize_key(username: str) -> str:
    """Normalizes username key: lowercase and removes leading @ for matching."""
    cleaned = username.strip().lower()
    if cleaned.startswith("@"):
        cleaned = cleaned[1:]
    return cleaned

class UserManager:
    """Lightweight persistent user store supporting Section 3.5 & 4.11 requirements."""
    def __init__(self, db_file: Path = DB_PATH):
        self.db_file = db_file
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_file.exists():
            self._save({})

    def _load(self) -> Dict[str, dict]:
        try:
            with open(self.db_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self, data: Dict[str, dict]):
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register_user(self, username: str, password: str, email: str = "") -> bool:
        norm_key = normalize_key(username)
        if not norm_key or not password or len(password) < 6:
            return False
        users = self._load()
        users[norm_key] = {
            "username": username.strip(), # Preserves display casing like @Rohit
            "email": email.strip(),
            "password_hash": hash_password(password)
        }
        self._save(users)
        return True

    def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        norm_key = normalize_key(username)
        users = self._load()
        user = users.get(norm_key)
        if not user:
            return None
        if verify_password(user["password_hash"], password):
            return {"username": user["username"], "email": user.get("email", "")}
        return None
