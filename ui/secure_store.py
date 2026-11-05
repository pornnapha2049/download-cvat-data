"""
Secure password storage for "Remember password" — works on Windows without extra installs.

Backend order:
1. keyring       — Windows Credential Manager / macOS Keychain / Secret Service (if installed and working)
2. Windows DPAPI — built into Windows: the password is encrypted for the current Windows user and
                   saved (encrypted, base64) in QSettings. Only this Windows account on this PC can decrypt it.
3. none          — "Remember password" is disabled (URL + username can still be remembered).

The plain password is never written to disk.
"""

from __future__ import annotations

import base64
import sys

SERVICE = "CVAT to YOLO Studio"
_SETTINGS_PREFIX = "login/secret/"


# ---------------------------------------------------------------------
# keyring (optional)
# ---------------------------------------------------------------------
def _keyring():
    try:
        import keyring
        from keyring.backends import fail
        kr = keyring.get_keyring()
        if isinstance(kr, fail.Keyring) or getattr(kr, "priority", 1) <= 0:
            return None   # installed but no usable backend
        return keyring
    except Exception:
        return None


# ---------------------------------------------------------------------
# Windows DPAPI via ctypes (no extra package)
# ---------------------------------------------------------------------
def _dpapi_available() -> bool:
    return sys.platform == "win32"


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    _crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _CRYPTPROTECT_UI_FORBIDDEN = 0x01

    _crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DATA_BLOB), wintypes.LPCWSTR, ctypes.POINTER(_DATA_BLOB),
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DATA_BLOB)]
    _crypt32.CryptProtectData.restype = wintypes.BOOL
    _crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DATA_BLOB), ctypes.POINTER(wintypes.LPWSTR), ctypes.POINTER(_DATA_BLOB),
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DATA_BLOB)]
    _crypt32.CryptUnprotectData.restype = wintypes.BOOL
    _kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    _kernel32.LocalFree.restype = ctypes.c_void_p

    def _blob(data: bytes):
        buf = ctypes.create_string_buffer(data, len(data))
        return _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf

    def _dpapi_encrypt(data: bytes) -> bytes:
        inp, _keep = _blob(data)
        out = _DATA_BLOB()
        if not _crypt32.CryptProtectData(ctypes.byref(inp), SERVICE, None, None, None,
                                         _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return ctypes.string_at(out.pbData, out.cbData)
        finally:
            _kernel32.LocalFree(ctypes.cast(out.pbData, ctypes.c_void_p))

    def _dpapi_decrypt(data: bytes) -> bytes:
        inp, _keep = _blob(data)
        out = _DATA_BLOB()
        if not _crypt32.CryptUnprotectData(ctypes.byref(inp), None, None, None, None,
                                           _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return ctypes.string_at(out.pbData, out.cbData)
        finally:
            _kernel32.LocalFree(ctypes.cast(out.pbData, ctypes.c_void_p))


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def backend_name() -> str | None:
    """'keyring', 'dpapi' or None when passwords cannot be stored securely."""
    if _keyring() is not None:
        return "keyring"
    if _dpapi_available():
        return "dpapi"
    return None


def available() -> bool:
    return backend_name() is not None


def describe() -> str:
    b = backend_name()
    if b == "keyring":
        return "Stored in the system credential store (Windows Credential Manager / Keychain), not in a file."
    if b == "dpapi":
        return "Encrypted with Windows DPAPI — only your Windows account on this PC can read it."
    return "Not available on this system: install 'keyring' (pip install keyring) to remember the password."


def _settings_key(account: str) -> str:
    return _SETTINGS_PREFIX + base64.urlsafe_b64encode(account.encode("utf-8")).decode("ascii")


def save_password(settings, account: str, password: str) -> bool:
    """Returns True when stored."""
    b = backend_name()
    try:
        if b == "keyring":
            _keyring().set_password(SERVICE, account, password)
            return True
        if b == "dpapi":
            enc = _dpapi_encrypt(password.encode("utf-8"))
            settings.setValue(_settings_key(account), base64.b64encode(enc).decode("ascii"))
            return True
    except Exception:
        pass
    return False


def load_password(settings, account: str) -> str | None:
    b = backend_name()
    try:
        if b == "keyring":
            return _keyring().get_password(SERVICE, account)
        if b == "dpapi":
            raw = settings.value(_settings_key(account), "", type=str)
            if raw:
                return _dpapi_decrypt(base64.b64decode(raw)).decode("utf-8")
    except Exception:
        pass
    return None


def delete_password(settings, account: str):
    kr = _keyring()
    if kr is not None:
        try:
            kr.delete_password(SERVICE, account)
        except Exception:
            pass
    try:
        settings.remove(_settings_key(account))
    except Exception:
        pass


def delete_all(settings):
    """Removes every DPAPI-encrypted secret written by this app (keyring entries are removed per account)."""
    try:
        settings.remove(_SETTINGS_PREFIX.rstrip("/"))
    except Exception:
        pass