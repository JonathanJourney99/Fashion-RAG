from together import Together
from dotenv import load_dotenv
import os


load_dotenv()
api_key= os.getenv('TOGETHER_AI')

client = Together(api_key=api_key)
response = client.images.generate(
    prompt=f"A classy googles",
    model="black-forest-labs/FLUX.1-schnell-Free",
    steps=3,
    n=4
)

image_url=response.data[0].url
print(image_url)