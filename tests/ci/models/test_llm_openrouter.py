"""Regression tests for OpenRouter client setup and response handling."""

import inspect
from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel

from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import UserMessage
from browser_use.llm.openrouter.chat import ChatOpenRouter


class Answer(BaseModel):
	answer: str


def _completion(*, content: str | None = 'ok', choices: bool = True) -> ChatCompletion:
	return ChatCompletion(
		id='chatcmpl-test',
		choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(role='assistant', content=content))]
		if choices
		else [],
		created=0,
		model='openai/gpt-4o',
		object='chat.completion',
	)


async def test_request_params_reach_completion_not_client():
	llm = ChatOpenRouter(model='openai/gpt-4o', api_key='test-key', top_p=0.9, seed=42)

	client = llm.get_client()

	assert client.api_key == 'test-key'
	assert 'top_p' not in llm._get_client_params()
	assert 'seed' not in llm._get_client_params()

	create = AsyncMock(return_value=_completion())
	with patch.object(type(client.chat.completions), 'create', create):
		await llm.ainvoke([UserMessage(content='question')])
	request_kwargs = create.await_args_list[0].kwargs
	assert request_kwargs['top_p'] == 0.9
	assert request_kwargs['seed'] == 42


def test_provider_key_does_not_fall_back_to_openai_key(monkeypatch: pytest.MonkeyPatch):
	monkeypatch.setenv('OPENAI_API_KEY', 'wrong-provider-key')
	monkeypatch.setenv('OPENROUTER_API_KEY', 'openrouter-key')
	assert ChatOpenRouter(model='openai/gpt-4o').get_client().api_key == 'openrouter-key'

	monkeypatch.delenv('OPENROUTER_API_KEY')
	with pytest.raises(ModelProviderError, match='Missing OpenRouter API key') as exc_info:
		ChatOpenRouter(model='openai/gpt-4o').get_client()
	assert exc_info.value.status_code == 401


async def test_empty_choices_raise_provider_error():
	llm = ChatOpenRouter(model='openai/gpt-4o', api_key='test-key')
	create = AsyncMock(return_value=_completion(choices=False))

	with patch.object(type(llm.get_client().chat.completions), 'create', create):
		with pytest.raises(ModelProviderError, match='missing or empty `choices`') as exc_info:
			await llm.ainvoke([UserMessage(content='question')])

	assert exc_info.value.status_code == 502


async def test_structured_provider_error_keeps_status_code():
	llm = ChatOpenRouter(model='openai/gpt-4o', api_key='test-key')
	create = AsyncMock(return_value=_completion(content=None))

	with patch.object(type(llm.get_client().chat.completions), 'create', create):
		with pytest.raises(ModelProviderError, match='Failed to parse structured output') as exc_info:
			await llm.ainvoke([UserMessage(content='question')], Answer)

	assert exc_info.value.status_code == 500


async def test_extra_body_reaches_sdk_extra_body_param():
	extra_body = {
		'provider': {'order': ['openai']},
		'models': ['openai/gpt-4o', 'anthropic/claude-sonnet-4.5'],
		'usage': {'include': True},
	}
	llm = ChatOpenRouter(model='openai/gpt-4o', api_key='test-key', extra_body=extra_body)

	client = llm.get_client()
	# Captured before patching: chat.completions.create is keyword-only with no **kwargs, so
	# OpenRouter-only keys are only deliverable via the SDK's own extra_body= parameter.
	real_create_signature = inspect.signature(client.chat.completions.create)

	for output_format in (None, Answer):
		create = AsyncMock(return_value=_completion(content='{"answer": "ok"}'))
		with patch.object(type(client.chat.completions), 'create', create):
			await llm.ainvoke([UserMessage(content='question')], output_format)

		request_kwargs = create.await_args_list[0].kwargs
		assert request_kwargs['extra_body'] == extra_body
		for openrouter_only_key in extra_body:
			assert openrouter_only_key not in request_kwargs
		real_create_signature.bind(**request_kwargs)
