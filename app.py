import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import requests
import json
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

# --- 1. Define Tools ---
@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
        "romance": "Jab We Met, Sita Ramam, 96",
        "thriller": "Drishyam, Ratsasan, Andhadhun",
        "horror": "Tumbbad, Stree, Bhool Bhulaiyaa"
    }

    return movies.get(
        genre.lower(),
        "No movies found for that genre."
    )


# ------------------------------------------------------------
# Weather
# ------------------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""

    try:
        # Geocoding API
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_params = {
            "name": city,
            "count": 1
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        ).json()

        if "results" not in geo_response:
            return f"Could not find weather data for {city}."

        location = geo_response["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # Weather API
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        ).json()

        current = weather_response["current"]

        result = {
            "city": location["name"],
            "temperature_celsius": current["temperature_2m"],
            "weather_code": current["weather_code"]
        }

        return json.dumps(result)

    except Exception as e:
        return f"Weather error: {str(e)}"


# ------------------------------------------------------------
# Temperature Conversion
# ------------------------------------------------------------

@tool
def celsius_to_fahrenheit(temp_c: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""

    return (temp_c * 9 / 5) + 32


@tool
def fahrenheit_to_celsius(temp_f: float) -> float:
    """Convert Fahrenheit temperature to Celsius."""

    return (temp_f - 32) * 5 / 9


# ------------------------------------------------------------
# Calculator
# ------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Calculate a basic mathematical expression."""

    try:
        allowed_characters = "0123456789+-*/().% "

        if not all(
            character in allowed_characters
            for character in expression
        ):
            return "Invalid mathematical expression."

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:
        return "Could not calculate the expression."


# ------------------------------------------------------------
# Square Root
# ------------------------------------------------------------

@tool
def square_root(number: float) -> str:
    """Calculate square root of a number."""

    if number < 0:
        return "Cannot calculate square root of a negative number."

    return str(math.sqrt(number))


# ------------------------------------------------------------
# Even / Odd
# ------------------------------------------------------------

@tool
def check_even_odd(number: int) -> str:
    """Check whether a number is even or odd."""

    if number % 2 == 0:
        return f"{number} is an even number."

    return f"{number} is an odd number."


# ------------------------------------------------------------
# Prime Number
# ------------------------------------------------------------

@tool
def check_prime(number: int) -> str:
    """Check whether a number is prime."""

    if number < 2:
        return f"{number} is not a prime number."

    for i in range(2, int(math.sqrt(number)) + 1):

        if number % i == 0:
            return f"{number} is not a prime number."

    return f"{number} is a prime number."


# ------------------------------------------------------------
# Current Date and Time
# ------------------------------------------------------------

@tool
def get_current_datetime() -> str:
    """Get current date and time."""

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ------------------------------------------------------------
# Word Counter
# ------------------------------------------------------------

@tool
def word_counter(text: str) -> str:
    """Count words and characters in a text."""

    words = len(text.split())
    characters = len(text)

    result = {
        "words": words,
        "characters": characters
    }

    return json.dumps(result)


# ------------------------------------------------------------
# BMI Calculator
# ------------------------------------------------------------

@tool
def bmi_calculator(
    weight_kg: float,
    height_m: float
) -> str:
    """Calculate BMI using weight in kg and height in meters."""

    if height_m <= 0:
        return "Height must be greater than zero."

    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        category = "Underweight"

    elif bmi < 25:
        category = "Normal weight"

    elif bmi < 30:
        category = "Overweight"

    else:
        category = "Obesity"

    result = {
        "BMI": round(bmi, 2),
        "category": category
    }

    return json.dumps(result)


# ------------------------------------------------------------
# Percentage Calculator
# ------------------------------------------------------------

@tool
def percentage_calculator(
    obtained: float,
    total: float
) -> str:
    """Calculate percentage from obtained and total marks."""

    if total == 0:
        return "Total marks cannot be zero."

    percentage = (obtained / total) * 100

    return f"{percentage:.2f}%"


# ------------------------------------------------------------
# Simple Interest
# ------------------------------------------------------------

@tool
def simple_interest(
    principal: float,
    rate: float,
    time: float
) -> str:
    """Calculate simple interest."""

    interest = (principal * rate * time) / 100

    total = principal + interest

    result = {
        "interest": round(interest, 2),
        "total_amount": round(total, 2)
    }

    return json.dumps(result)


# ------------------------------------------------------------
# Compound Interest
# ------------------------------------------------------------

@tool
def compound_interest(
    principal: float,
    rate: float,
    time: float
) -> str:
    """Calculate compound interest annually."""

    amount = principal * (
        1 + rate / 100
    ) ** time

    interest = amount - principal

    result = {
        "interest": round(interest, 2),
        "total_amount": round(amount, 2)
    }

    return json.dumps(result)


tools = [
    get_weather,
    search_movies,
    change__to_f,
    calculator,
    get_current_datetime,
    square_root,
    check_even_odd,
    check_prime,
    word_counter,
    celsius_to_fahrenheit,
    fahrenheit_to_celsius,
    bmi_calculator,
    percentage_calculator,
    simple_interest,
    compound_interest
]

# --- 2. Initialize Model & Agent ---
# Retrieve the key from the OS environment instead of Colab's userdata
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized agent restricted ONLY to Indian weather and cinema. "
        "For any other roles, topics, questions, or general knowledge outside of Indian weather and movies, "
        "you must say exactly: 'I am not authorized to answer questions outside of Indian weather and cinema.'"
    )
)

class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")


def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}

def extract_text_response(agent_output: dict) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    # Case 1: top-level messages (normal final state)
    messages = agent_output.get("messages")

    # Case 2: nested under a node name, e.g. {"model": {"messages": [...]}}
    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))

    return str(agent_output)

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)

# --- 3. FastAPI App ---
##Need To Code
app = FastAPI()
add_routes(app, formatted_agent_chain, path="/agent")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
