still under development
initialize a directory
```bash
uv init mcp-client
```
initialize a venv, initialize and set the api key in .env
```bash
uv venv
#for windows
.venv\Scripts\activate
```
install libraries
```bash
uv pip install -r requirements.txt
```
run the client
```
uv run client.py <MCP_SERVER_URL>
```
