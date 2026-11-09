import { streamSse } from "@fim/fetch";
import { ChatMessage, CompletionOptions, LLMOptions } from "../../index.js";

import { ChatCompletionCreateParams } from "@fim/openai-adapters";
// Sentinel tokens Continue appended to prompts to mark apply/next-edit
// requests. Those modes don't exist in this fork, but Inception still strips
// the tokens defensively, so the values are kept here.
const APPLY_UNIQUE_TOKEN = "<|!@#IS_APPLY!@#|>";
const UNIQUE_TOKEN = "<|!@#IS_NEXT_EDIT!@#|>";
import OpenAI from "./OpenAI.js";

/**
 * Inception Labs provider
 *
 * Integrates with Inception Labs' OpenAI-compatible API endpoints.
 * Provides access to Mercury models for autocomplete and other tasks.
 *
 * Different models use different API endpoints:
 * - mercury-editor-mini-experimental: zaragoza.api.inceptionlabs.ai
 * - mercury-editor-small-experimental: copenhagen.api.inceptionlabs.ai
 *
 * More information at: https://docs.inceptionlabs.ai/
 */
/**
 * Mercury Edit 2 and later use a dedicated `fim/completions` endpoint and a
 * plain prompt/suffix body. The legacy `completions` path (and its penalty and
 * stop-word workarounds) is only for the older experimental Mercury models.
 */
export function usesDedicatedFimEndpoint(model: string | undefined): boolean {
  return /^mercury-edit-\d/i.test(model ?? "");
}

class Inception extends OpenAI {
  static providerName = "inception";
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://api.inceptionlabs.ai/v1/",
    model: "mercury-edit-2",
    completionOptions: {
      temperature: 0.0,
      model: "mercury-edit-2", // required by CompletionOptions
    },
  };

  constructor(options: LLMOptions) {
    super(options);
    // The shared OpenAI adapter would send FIM to its own path; mercury-edit
    // needs `fim/completions`, which `_streamFim` below handles.
    if (usesDedicatedFimEndpoint(this.model)) {
      this.useOpenAIAdapterFor = this.useOpenAIAdapterFor.filter(
        (t) => t !== "streamFim",
      );
    }
  }

  supportsFim(): boolean {
    return true;
  }

  // It seems like this should be inherited automatically from the parent OpenAI class, but it sometimes doesn't.
  // protected useOpenAIAdapterFor: (LlmApiRequestType | "*")[] = [
  //   "chat",
  //   "embed",
  //   "list",
  //   "rerank",
  //   "streamChat",
  //   "streamFim",
  // ];

  protected modifyChatBody(
    body: ChatCompletionCreateParams,
  ): ChatCompletionCreateParams {
    const hasNextEditCapability = this.capabilities?.nextEdit ?? false;

    // Add the nextEdit parameter for Inception-specific routing.
    (body as any).nextEdit = hasNextEditCapability;
    return body;
  }

  async *_streamFim(
    prefix: string,
    suffix: string,
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    if (usesDedicatedFimEndpoint(options.model)) {
      yield* this._streamEditFim(prefix, suffix, signal, options);
      return;
    }
    const endpoint = new URL("completions", this.apiBase);
    const resp = await this.fetch(endpoint, {
      method: "POST",
      body: JSON.stringify({
        model: options.model,
        prompt: prefix,
        suffix: suffix.trim() === "" ? "<|endoftext|>" : suffix,
        max_tokens: options.maxTokens ?? 150, // Only want this for /fim, not chat
        temperature: options.temperature,
        top_p: options.topP,
        frequency_penalty: options.frequencyPenalty,
        presence_penalty: options.presencePenalty,
        stop: [...(options.stop ?? []), "\n\n", "\n \n"],
        stream: true,
      }),
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      signal,
    });
    for await (const chunk of streamSse(resp)) {
      if (!chunk.choices[0]) {
        continue;
      }
      yield chunk.choices[0].text;
    }
  }

  private async *_streamEditFim(
    prefix: string,
    suffix: string,
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    const endpoint = new URL("fim/completions", this.apiBase);
    const resp = await this.fetch(endpoint, {
      method: "POST",
      body: JSON.stringify({
        model: options.model,
        prompt: prefix,
        suffix,
        max_tokens: options.maxTokens ?? 150,
        temperature: options.temperature,
        top_p: options.topP,
        ...(options.stop?.length ? { stop: options.stop } : {}),
        stream: true,
      }),
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      signal,
    });
    for await (const chunk of streamSse(resp)) {
      const text = chunk.choices?.[0]?.text;
      if (text) {
        yield text;
      }
    }
  }

  protected async *_streamChat(
    messages: ChatMessage[],
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<ChatMessage> {
    if (this.isNextEdit(messages)) {
      messages = this.removeToken(messages, UNIQUE_TOKEN);
      yield* this.streamSpecialEndpoint(
        "edit/completions",
        messages,
        signal,
        options,
      );
    } else if (this.isApply(messages)) {
      messages = this.removeToken(messages, APPLY_UNIQUE_TOKEN);
      yield* this.streamSpecialEndpoint(
        "apply/completions",
        messages,
        signal,
        options,
      );
    } else {
      // Use regular chat/completions endpoint - call parent OpenAI implementation.
      yield* super._streamChat(messages, signal, options);
    }
  }

  private isNextEdit(messages: ChatMessage[]): boolean {
    // Check if any message contains the unique next edit token.
    return messages.some(
      (message) =>
        typeof message.content === "string" &&
        message.content.endsWith(UNIQUE_TOKEN),
    );
  }

  private isApply(messages: ChatMessage[]): boolean {
    return messages.some(
      (message) =>
        typeof message.content === "string" &&
        message.content.endsWith(APPLY_UNIQUE_TOKEN),
    );
  }

  private removeToken(messages: ChatMessage[], token: string): ChatMessage[] {
    const lastMessage = messages[messages.length - 1];

    if (
      typeof lastMessage?.content === "string" &&
      lastMessage.content.endsWith(token)
    ) {
      const cleanedMessages = [...messages];
      cleanedMessages[cleanedMessages.length - 1] = {
        ...lastMessage,
        content: lastMessage.content.slice(0, -token.length),
      };
      return cleanedMessages;
    }

    return messages;
  }

  private async *streamSpecialEndpoint(
    path: string,
    messages: ChatMessage[],
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<ChatMessage> {
    const endpoint = new URL(path, this.apiBase);

    const resp = await this.fetch(endpoint, {
      method: "POST",
      body: JSON.stringify({
        model: options.model,
        messages,
        max_tokens: options.maxTokens,
        temperature: options.temperature,
        top_p: options.topP,
        frequency_penalty: options.frequencyPenalty,
        presence_penalty: options.presencePenalty,
        stop: options.stop,
        stream: true,
      }),
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      signal,
    });

    for await (const chunk of streamSse(resp)) {
      const content = chunk.choices?.[0]?.delta?.content;
      if (content) {
        yield { role: "assistant", content };
      }
    }
  }
}

export default Inception;
