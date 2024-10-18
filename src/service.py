import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, JSON
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from datetime import datetime, timedelta

# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

# User credentials for authentication
USERS = {
    "user123": "password123",
    "user456": "password456"
}

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/predict":
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})

            try:
                token = token.split()[1]  # Remove 'Bearer ' prefix
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})

            request.state.user = payload.get("sub")

        response = await call_next(request)
        return response

# Pydantic model to validate input data
class InputModel(BaseModel):
    GRE_Score:int
    TOEFL_Score:int
    University_Rating:int
    SOP:float
    LOR :float
    CGPA:float
    Research:float

# Get the model from the Model Store
admission_rd_runner = bentoml.sklearn.get("admission_rd:latest").to_runner()
scaler = bentoml.sklearn.load_model("admission_scaler:latest")
# Create a service API
rd_service = bentoml.Service("ridge_service", runners=[admission_rd_runner])

# Add the JWTAuthMiddleware to the service
rd_service.add_asgi_middleware(JWTAuthMiddleware)

# Create an API endpoint for the service
@rd_service.api(input=JSON(), output=JSON())
def login(credentials: dict) -> dict:
    username = credentials.get("username")
    password = credentials.get("password")

    if username in USERS and USERS[username] == password:
        token = create_jwt_token(username)
        return {"token": token}
    else:
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

# Create an API endpoint for the service
@rd_service.api(
    input=JSON(pydantic_model=InputModel),
    output=JSON(),
    route='/predict'
)
async def classify(input_data: InputModel, ctx: bentoml.Context) -> dict:
    request = ctx.request
    user = request.state.user if hasattr(request.state, 'user') else None

    # Convert the input data to a numpy array
    input_series = np.array([input_data.GRE_Score, input_data.TOEFL_Score, input_data.University_Rating, input_data.SOP,
                             input_data.LOR, input_data.CGPA, input_data.Research])
        # Normaliser les données avec le scaler
    print(input_series.reshape(1, -1))
    scaled_data = scaler.transform(input_series.reshape(1, -1))
    print(scaled_data)
    result = await admission_rd_runner.predict.async_run(scaled_data)

    return {
        "prediction": result.tolist(),
        "user": user
    }

# Function to create a JWT token
def create_jwt_token(user_id: str):
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {
        "sub": user_id,
        "exp": expiration
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token