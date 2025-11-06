# Application Status Agent

A production-ready LangGraph-based agent that helps users check their job application status through an intelligent **Plan → Gather → Coverage → Draft** workflow with MCP (Model Context Protocol) integration.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Set up environment variables
cp env.example .env
# Edit .env with your API keys
```

### 2. Run the Agent

#### Local Testing (Recommended)
```bash
# Using Makefile (easiest)
make run_local CONV_ID=215471618006513

# With full conversation
make run_local CONV_ID=215471618006513 FULL_CONV=true

# With actual Intercom writes (careful!)
make run_local CONV_ID=215471618006513 NO_DRY_RUN=true

# Or use the script directly
PYTHONPATH=src:$PYTHONPATH python scripts/run_local.py 215471618006513 --help
```

**Features:**
- ✅ Dry run mode by default (no writes to Intercom)
- ✅ First message only mode (faster testing)
- ✅ Automatic state dumping to `local_runs/` folder
- ✅ Clean summary output

#### LangGraph Studio
```bash
# Start the development server
langgraph up --wait

# Open Studio
open https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### 3. Run Tests
```bash
pytest tests/ -v
```

## 🏗️ Architecture

```
User Input → Initialize → Plan → Gather → Coverage → Draft → Response
     ↓           ↓         ↓       ↓        ↓        ↓
  Messages    MCP Tools  Plan   Execute   Analyze  Generate
```

**Core Components:**
- **Initialize Node**: Sets up MCP client and available tools
- **Plan Node**: Creates execution plan based on conversation history
- **Gather Node**: Executes MCP tools to fetch data
- **Coverage Node**: Analyzes data sufficiency and decides next action
- **Draft Node**: Generates final response with evidence

**MCP Integration:**
- Dynamic tool discovery from MCP server
- Flexible schema handling for prototyping
- Centralized client management
- Error handling and retry logic

## 📊 Example Output

```json
{
  "response": "You have 3 active applications for Software Engineer positions...",
  "tool_data": {
    "get_user_applications_detailed": {
      "applications": [
        {"title": "Software Engineer", "status": "Interview", "company": "TechCorp"}
      ]
    }
  },
  "docs_data": {
    "withdrawal process": {
      "content": "To withdraw an application, contact your recruiter..."
    }
  },
  "hops": [
    {
      "hop_number": 1,
      "plan": {"tool_calls": [{"name": "get_user_applications_detailed"}]},
      "gather": {"successful_tools": 1, "failed_tools": 0},
      "coverage": {"coverage_analysis": "sufficient", "next_action": "continue"}
    }
  ]
}
```

## 🧪 Test Cases

✅ **Application Status**: Returns detailed status with tool citations  
✅ **Withdrawal Process**: Provides accurate withdrawal instructions  
✅ **Payment Queries**: Handles complex payment and region-specific questions  
✅ **Multi-hop Planning**: Iteratively gathers more data when needed  
✅ **Error Handling**: Gracefully handles MCP server errors  

## 📁 Project Structure

```
src/
├── agent/                    # Main agent implementation
│   ├── types.py             # State management and data models
│   ├── llm.py               # LLM client configuration
│   ├── graph.py             # LangGraph assembly and routing
│   ├── runner.py            # Main entry point
│   ├── prompts.py           # LangSmith prompt management
│   └── nodes/               # Graph nodes
│       ├── plan/            # Planning node with schemas
│       ├── gather/          # Tool execution node
│       ├── coverage/        # Data sufficiency analysis
│       └── draft/           # Response generation
└── mcp/                     # MCP integration
    ├── client.py            # MCP client implementation
    ├── factory.py           # Client factory
    ├── schemas.py           # MCP data models
    └── tools.py             # Tool wrappers
```

## 🔧 Configuration

### Required Environment Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_key
MODEL_NAME=gpt-4o-mini  # or gpt-4o, gpt-5

# LangSmith Integration
LANGSMITH_API_KEY=your_langsmith_key
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=application-status-agent

# MCP Server
MCP_SERVER_URL=https://your-mcp-server.com
MCP_API_KEY=your_mcp_key
```

### Optional Configuration
```bash
# LangSmith Project
LANGSMITH_PROJECT=application-status-agent

# Development
LANGCHAIN_DEBUG=true
```

## 🚀 Deployment

### Local Development
```bash
# Run with LangGraph
langgraph up --wait
```

### Production Deployment
See [PRODUCTION_DEPLOYMENT.md](./PRODUCTION_DEPLOYMENT.md) for detailed deployment instructions.

## 📚 Documentation

- **[MESSAGE_FORMAT.md](./MESSAGE_FORMAT.md)** - Multi-message conversation format
- **[PRODUCTION_DEPLOYMENT.md](./PRODUCTION_DEPLOYMENT.md)** - Production deployment guide
- **[LangGraph Docs](https://langchain-ai.github.io/langgraph/)** - LangGraph framework
- **[MCP Protocol](https://modelcontextprotocol.io/)** - Model Context Protocol

## 🎯 Features

**Core Capabilities:**
- [x] Multi-message conversation support
- [x] Dynamic MCP tool discovery
- [x] Intelligent planning with context awareness
- [x] Iterative data gathering (max 2 hops)
- [x] Coverage analysis for data sufficiency
- [x] Evidence-based response generation
- [x] LangSmith prompt management
- [x] Production-ready deployment

**Advanced Features:**
- [x] Conversation history awareness
- [x] Flexible schema handling
- [x] Error recovery and retry logic
- [x] Cost optimization with token tracking
- [x] Comprehensive testing suite

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.