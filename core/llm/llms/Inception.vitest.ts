import { afterEach, describe, expect, it, vi } from "vitest";

import { llmFromProviderAndOptions } from ".";
import { getTemplateForModel } from "../../autocomplete/templating/AutocompleteTemplate";

import { usesDedicatedFimEndpoint } from "./Inception";

describe("Inception mercury-edit FIM", () => {
  afterEach(() => vi.restoreAllMocks());

  it("detects models on the dedicated endpoint", () => {
    expect(usesDedicatedFimEndpoint("mercury-edit-2")).toBe(true);
    expect(usesDedicatedFimEndpoint("mercury-coder-small")).toBe(false);
    expect(usesDedicatedFimEndpoint(undefined)).toBe(false);
  });

  it("posts a plain prompt/suffix body to fim/completions", async () => {
    const llm: any = llmFromProviderAndOptions(
      "inception" as any,
      {
        model: "mercury-edit-2",
        apiKey: "k",
        uniqueId: "t",
      } as any,
    );
    let url = "";
    let body: any;
    llm.fetch = async (u: URL, init: any) => {
      url = u.toString();
      body = JSON.parse(init.body);
      return new Response(
        'data: {"choices":[{"text":"abc"}]}\n\ndata: [DONE]\n\n',
      );
    };
    const out: string[] = [];
    for await (const c of llm._streamFim(
      "pre",
      "suf",
      new AbortController().signal,
      {
        model: "mercury-edit-2",
        maxTokens: 50,
        presencePenalty: 1.5,
        stop: [],
      },
    )) {
      out.push(c);
    }
    expect(url).toBe("https://api.inceptionlabs.ai/v1/fim/completions");
    expect(body).toMatchObject({
      prompt: "pre",
      suffix: "suf",
      max_tokens: 50,
    });
    expect(body.presence_penalty).toBeUndefined();
    expect(body.stop).toBeUndefined();
    expect(out.join("")).toBe("abc");
  });

  it("bypasses the OpenAI adapter for streamFim only on mercury-edit", () => {
    const edit: any = llmFromProviderAndOptions(
      "inception" as any,
      {
        model: "mercury-edit-2",
        uniqueId: "t",
      } as any,
    );
    const legacy: any = llmFromProviderAndOptions(
      "inception" as any,
      {
        model: "mercury-coder-small",
        uniqueId: "t",
      } as any,
    );
    expect(edit.useOpenAIAdapterFor).not.toContain("streamFim");
    expect(legacy.useOpenAIAdapterFor).toContain("streamFim");
  });

  it("does not use the sentinel-token template for mercury-edit-2", () => {
    const t = getTemplateForModel("mercury-edit-2");
    expect(t).not.toBe(getTemplateForModel("mercury-coder-small"));
  });
});
