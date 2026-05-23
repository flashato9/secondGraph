# SemanticOS

Imagine a computer that finally **speaks human**. 
- No more fighting with command lines.
- No more drowning in open tabs.
- No more hunting through endless folders for a photo you can see in your mind but can't find on your drive. 

**Stop operating a machine** and **start directing a digital partner**. 

Powered by LangGraph, this intelligent agent harness handles the entire execution—from fetching live APIs to navigating your local system—based on a single natural language request. 

You **define the what**, your **computer figures out the how**.

## How to use it?

### 1. Clone the repository

```bash
git clone https://github.com/flashato9/SemanticOS.git
```

### 2. Run the Application

From the project directory, run:

```bash
langgraph dev --allow-blocking --debug-port 5679
```

This starts the LangGraph server and required services in the background.

### 3. Go to Chat

Open your browser and navigate to:

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123
```

The chat interface will be available for you to interact with.

### 4. Start Chatting

Simply type your messages in the chat box and press Enter. The agent will respond with intelligent replies based on the configured logic and tools.

<div align="center">
  <img src="./static/README/chat_screenshot.png" alt="Chat interface. Powered by LangGraph" width="75%" />
</div>
