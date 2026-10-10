import { describe, expect, it } from "vitest";
import { answerToPlainText, parseAnswer } from "./answer";

const FULL = `[[STRUCTURE]]
1. Requirements
2. API design
[[ANSWER]]
I'd start by clarifying the expected scale.
Then I'd sketch the API.
[[KEY_POINTS]]
- Base62 short codes
- Cache hot links
  in Redis
[[EXAMPLE]]
At 100M links/day...
[[FOLLOW_UP]]
What read/write ratio should I assume?
[[BASIS]]
Suggested framing: generic system-design structure.
[[CAVEAT]]
Context insufficient — verify before using.`;

describe("parseAnswer", () => {
  it("parses every section", () => {
    const a = parseAnswer(FULL);
    expect(a.structure).toEqual(["Requirements", "API design"]);
    expect(a.answer).toBe("I'd start by clarifying the expected scale.\nThen I'd sketch the API.");
    expect(a.keyPoints).toEqual(["Base62 short codes", "Cache hot links in Redis"]);
    expect(a.example).toBe("At 100M links/day...");
    expect(a.followUp).toBe("What read/write ratio should I assume?");
    expect(a.basis).toContain("Suggested framing");
    expect(a.caveat).toContain("verify before using");
    expect(a.current).toBe("caveat");
  });

  it("handles partial streams and hides half-written markers", () => {
    const a = parseAnswer("[[ANSWER]]\nWe used Kafka because\n[[KEY_PO");
    expect(a.answer).toBe("We used Kafka because");
    expect(a.keyPoints).toEqual([]);
    expect(a.current).toBe("answer");
  });

  it("treats unmarked output as the answer", () => {
    const a = parseAnswer("Just a plain answer.");
    expect(a.answer).toBe("Just a plain answer.");
    expect(a.preamble).toBe("");
  });

  it("tolerates CRLF, code fences and inline marker text", () => {
    const a = parseAnswer(
      "```\r\n[[ANSWER]] Inline start\r\nmore\r\n[[CLARIFY]]\r\n- Which region?\r\n```",
    );
    expect(a.answer).toBe("Inline start\nmore");
    expect(a.clarify).toEqual(["Which region?"]);
  });

  it("renders plain text for copy", () => {
    const text = answerToPlainText(parseAnswer(FULL));
    expect(text).toContain("1. Requirements");
    expect(text).toContain("• Base62 short codes");
    expect(text).toContain("You could ask: What read/write ratio");
  });
});
