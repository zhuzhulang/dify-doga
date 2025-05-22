import httpx
from collections.abc import Generator
from typing import Optional, Union, cast, List
from core.model_runtime.entities.llm_entities import LLMResult, LLMResultChunk, LLMResultChunkDelta
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.model_runtime.entities import PromptMessage, PromptMessageTool
from core.model_runtime.entities import (
    AssistantPromptMessage,
    LLMUsage
)
from core.model_runtime.errors.invoke import (
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)

class DogaError(Exception):
    pass

class DogaAPIError(DogaError):
    message: str
    request: httpx.request
    body: object | None

    def __init__(self, message, request, *, body) -> None:  # noqa: ARG002
        super().__init__(message)
        self.request = request
        self.message = message
        self.body = body

class DogaLargeLanguageModel(LargeLanguageModel):
    def _invoke(self, model: str, credentials: dict, prompt_messages: list[PromptMessage], model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None, stop: Optional[List[str]] = None,
        stream: bool = True, user: Optional[str] = None) -> Union[LLMResult, Generator]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for tool calling
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        if stream:
            response_text = ["我是个传奇的人物,","不管你信不信,","这都是个事实。","哈哈"]
            response = []
            for text in response_text:
                chunk = LLMResultChunk(
                    model=model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(
                        index=0,
                        message=AssistantPromptMessage(content=text, tool_calls=[]),
                    )
                )
                response.append(chunk)
            return self._handle_stream_response(response)
        else:
            assistant_prompt_message = AssistantPromptMessage(content="我是个传奇的人物", tool_calls=[])
            usage = LLMUsage.empty_usage()
            # usage = self._calc_response_usage(
            #     model, credentials, usage.prompt_tokens, usage.completion_tokens
            # )
            response = dict(
                model=model, prompt_messages=prompt_messages, message=assistant_prompt_message, usage=usage
            )
            return self._handle_sync_response(response)

    def get_num_tokens(self, model: str, credentials: dict, prompt_messages: list[PromptMessage],
                    tools: Optional[list[PromptMessageTool]] = None) -> int:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return:
        """
        return 0

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        pass

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """
        return {
            InvokeConnectionError: [DogaAPIError],
            InvokeServerUnavailableError: [DogaAPIError],
            InvokeRateLimitError: [DogaAPIError],
            InvokeAuthorizationError: [DogaAPIError],
            InvokeBadRequestError: [
                DogaAPIError
            ],
        }

    def _handle_stream_response(self, response) -> Generator:
        for chunk in response:
            yield chunk
    def _handle_sync_response(self, response) -> LLMResult:
        return LLMResult(**response)