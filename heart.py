from vitallens import VitalLens, Method
from dotenv import load_dotenv
import os

load_dotenv()

vl = VitalLens(method=Method.VITALLENS, api_key=os.getenv("VITALLENS_API_KEY"))
result = vl("henry.mp4")
print(result)
