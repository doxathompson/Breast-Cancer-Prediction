import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

# In a clinical setting, DO NOT hardcode credentials. Store securely in Azure Key Vault or AWS Secrets Manager.
# For demo purposes only:
VALID_USERNAME = "clinician"
VALID_PASSWORD = "secure_password_123"

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    is_user_ok = secrets.compare_digest(credentials.username, VALID_USERNAME)
    is_pass_ok = secrets.compare_digest(credentials.password, VALID_PASSWORD)
    if not (is_user_ok and is_pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
