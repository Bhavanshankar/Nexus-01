"""`python -m concierge_ops_server` - the PDF section 2.3 entry point. The server is mcp_servers/concierge_ops_server.py."""

import runpy

if __name__ == "__main__":
    runpy.run_module("mcp_servers.concierge_ops_server", run_name="__main__", alter_sys=True)
