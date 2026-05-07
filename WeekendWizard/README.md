# Weekend Wizard

Weekend Wizard is a small local CLI agent for the "Capstone: Build a Simple, Fun Agent (MCP + Free APIs + Local LLM)" exercise from the Level 2 - Practitioner Confluence page.

It connects a local Ollama model to a Python MCP server that exposes a few free public APIs:

- Open-Meteo for current weather
- Open Library for book recommendations
- JokeAPI for a safe one-line joke
- Dog CEO for a random dog image link
- Open-Meteo geocoding as a convenience helper for city names

The agent uses a simple tool-calling loop, gathers tool output, and then writes a short final response.

## Prerequisites

Required:

- Windows with PowerShell
- Python 3.10 or newer
- Ollama installed and running locally
- At least one local model installed in Ollama
- Internet access for the free APIs

Recommended:

- VS Code for editing
- A terminal that can run `python`

## Installed For This Project

The project uses the following Python packages:

- `mcp`
- `requests`

I intentionally did not add the optional trivia API yet, per request.

## What Was Coded

### `server_fun.py`

This is the MCP tools server. It exposes:

- `get_weather(latitude, longitude)`
- `city_to_coords(city, limit=1)`
- `book_recs(topic, limit=5)`
- `random_joke()`
- `random_dog()`

### `agent_fun.py`

This is the CLI agent client. It:

- launches the MCP server over stdio
- discovers the tools at startup
- talks to Ollama through the local HTTP API
- performs a loop of decide -> call tool -> observe -> decide
- uses a reflection pass before printing the final answer
- forces the obvious tools for weekend-plan prompts so the demo completes reliably

### `requirements.txt`

Defines the Python dependencies needed to run the project.

## Model Choice

The exercise text references `mistral:7b`, but on this machine the installed Ollama model is:

- `llama3.2:1b`

The agent defaults to that local model, but you can override it with:

- `WEEKEND_WIZARD_MODEL`

## How To Run

1. Install the dependencies:
```powershell
python -m pip install -r requirements.txt
```

2. Start Ollama if it is not already running.

3. Run the agent:
```powershell
python agent_fun.py
```

4. Enter a prompt such as:
```text
Plan a cozy Saturday in New York at (40.7128, -74.0060). Include the current weather, 2 book ideas about mystery, one joke, and a dog pic.
```

5. Type `quit` or `exit` to stop.

## Demo Transcript

For the required screen recording or terminal transcript, use this order:

1. Open PowerShell in the project root:
```powershell
cd C:\Users\i23448\source\repos\capstone-aiagent-shawn-englerth\WeekendWizard
```

2. Start the transcript:
```powershell
Start-Transcript -Path .\WeekendWizardDemo.txt
```

3. Activate the virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

4. Run the agent:
```powershell
python .\WeekendWizard\agent_fun.py
```

5. Watch for the startup banner and the available tool list.

6. Enter your demo prompt at the `You:` prompt.

7. When the agent finishes, type `quit` at the `You:` prompt.

8. Stop the transcript:
```powershell
Stop-Transcript
```

9. View the transcript:
```powershell
Get-Content .\WeekendWizardDemo.txt
```

10. If you want to end the virtual environment in that shell, run:
```powershell
deactivate
```

## What To Expect

When the agent starts, it prints the tools it discovered from the MCP server.

For a weekend-planning request, it should gather:

- weather data
- two book recommendations
- one joke
- one dog image link (you can specify breed)

Then it prints a short final response that is grounded in those results.

## Notes

- The trivia API was intentionally left out for now.
- The geocoding helper is included as a convenience, but the main exercise flow still centers on weather, books, jokes, and dog images.
- If you want a different Ollama model, set `WEEKEND_WIZARD_MODEL` before starting the agent.

## Troubleshooting

- If the agent cannot connect to Ollama, make sure the Ollama service is running on `http://localhost:11434`.
- If tool calls fail, check that your machine has internet access.
- If Python cannot import `mcp` or `requests`, rerun the dependency install step.
- If the model returns weak JSON, the agent includes a repair pass, but a stronger model may improve reliability.

## Next Steps

Possible follow-up improvements:

- add the optional trivia tool
- improve city-name handling in the agent prompt
- add retry/backoff logic for HTTP calls
- add a small test suite for the tool server
