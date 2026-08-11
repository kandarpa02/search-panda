# from fastapi import FastAPI
# from pydantic import BaseModel
# from llama_cpp import Llama
# import uvicorn
# from .models import download_model
# from pathlib import Path

# api = FastAPI()


# class ChatMessage(BaseModel):
#     role: str
#     content: str


# class ChatCompletionRequest(BaseModel):
#     model: str
#     messages: list[ChatMessage]
#     max_tokens: int | None = 512
#     temperature: float | None = 0.7


# def load_llama(model_path: str) -> Llama:
#     llm = Llama(
#         model_path=str(model_path),
#         n_ctx=4096,
#         verbose=False,
#     )
#     def llama_run(message:str, max_tokens, temperature) -> str:
#         response = llm.create_chat_completion(
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": message,
#                     }
#                 ],
#                 max_tokens=max_tokens,
#                 temperature=temperature,
#             )
#         return response["choices"][0]["message"]["content"]

#     return llama_run
    
# class LlamaAPI:
#     def __init__(self, model_path: str | Path):
#         self.model = load_llama(model_path)
#         @api.post("/v1/chat/completions")
#         async def chat(request: ChatCompletionRequest):

#             message = request.messages[-1].content

#             response = self.model(
#                 messages=[
#                     {
#                         "role": m.role,
#                         "content": m.content,
#                     }
#                     for m in request.messages
#                 ],
#                 max_tokens=request.max_tokens,
#                 temperature=request.temperature,
#             )

#             return response

#     async def serve(self):
#         config = uvicorn.Config(
#             api,
#             host="127.0.0.1",
#             port=4586,
#             reload=False,
#         )

#         server = uvicorn.Server(config)

#         await server.serve()
    
# # @api.post("/chat")
# # async def chat(request: ChatRequest):
# #     return {
# #         "response": self.model(request.message)
# #     }

from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
from pathlib import Path


api = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = 512
    temperature: float | None = 0.7


class LlamaAPI:
    def __init__(self, model_path: str | Path):
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            verbose=False,
        )

    async def serve(self):
        config = uvicorn.Config(
            api,
            host="127.0.0.1",
            port=1890,
            reload=False,
        )

        server = uvicorn.Server(config)

        await server.serve()


llama_api: LlamaAPI | None = None


@api.post("/v1/chat/completions")
async def chat(request: ChatCompletionRequest):
    if llama_api is None:
        raise RuntimeError("Llama model has not been loaded")

    response = llama_api.model.create_chat_completion(
        messages=[
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ],
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return response