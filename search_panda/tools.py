TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_result",
            "description": "Read one search result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer"
                    }
                },
                "required": ["index"]
            }
        }
    }
]