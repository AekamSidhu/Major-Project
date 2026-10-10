// Client-side validation for settings inputs. Messages are shown inline next
// to the control; the backend re-validates everything on save.

import type { PrivacyRuleType, Rect } from "@/bindings";

/** Mirror of `openmeet_core::settings::validate_base_url`. Null when valid. */
export function validateBaseUrl(url: string): string | null {
  const value = url.trim();
  const lower = value.toLowerCase();
  if (!(lower.startsWith("http://") || lower.startsWith("https://"))) {
    return "Enter a URL starting with http:// or https://";
  }
  if (/\s/.test(value)) return "The URL must not contain spaces.";
  const rest = value.slice(value.indexOf("://") + 3);
  if (rest === "" || rest.startsWith("/")) return "The URL is missing a host name.";
  if ((rest.split("/")[0] ?? "").includes("@")) {
    return "Don't put credentials in the URL. Store API keys with the key field instead.";
  }
  return null;
}

/** Base URL field validation: optional fields accept an empty value. */
export function validateOptionalBaseUrl(url: string, required: boolean): string | null {
  if (url.trim() === "") return required ? "This provider needs a server address." : null;
  return validateBaseUrl(url);
}

export function validateRequired(value: string, what: string): string | null {
  return value.trim() === "" ? `${what} is required.` : null;
}

/** Parses an integer field; returns an error message or the value. */
export function parseIntegerInRange(
  raw: string,
  min: number,
  max: number,
): { ok: true; value: number } | { ok: false; error: string } {
  const t = raw.trim();
  if (t === "") return { ok: false, error: `Enter a number between ${min} and ${max}.` };
  if (!/^-?\d+$/.test(t)) return { ok: false, error: "Enter a whole number." };
  const n = Number(t);
  if (!Number.isSafeInteger(n) || n < min || n > max) {
    return { ok: false, error: `Enter a number between ${min} and ${max}.` };
  }
  return { ok: true, value: n };
}

/** ISO-639-1/2 language code ("en", "pt", "yue"). */
export function validateLanguageCode(code: string): string | null {
  const t = code.trim().toLowerCase();
  if (t === "") return "Enter a language code such as en or de.";
  return /^[a-z]{2,3}$/.test(t)
    ? null
    : "Use a two- or three-letter ISO language code, e.g. en or de.";
}

/** API keys are opaque, but never contain whitespace. */
export function validateApiKey(key: string): string | null {
  const t = key.trim();
  if (t === "") return "Paste a key first.";
  if (/\s/.test(t)) return "API keys don't contain spaces or line breaks.";
  if (t.length < 8) return "That looks too short to be an API key.";
  return null;
}

/** Parses "x,y,w,h" (whole pixels, w/h > 0). */
export function parseRegion(raw: string): { ok: true; rect: Rect } | { ok: false; error: string } {
  const parts = raw.split(",").map((p) => p.trim());
  if (parts.length !== 4 || parts.some((p) => !/^\d+$/.test(p))) {
    return {
      ok: false,
      error: "Enter the region as x,y,width,height in pixels, e.g. 0,0,800,600.",
    };
  }
  const [x, y, width, height] = parts.map(Number) as [number, number, number, number];
  if (width <= 0 || height <= 0)
    return { ok: false, error: "Width and height must be greater than zero." };
  return { ok: true, rect: { x, y, width, height } };
}

const DOMAIN_RE = /^(\*\.)?([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{0,61}[a-z0-9]$/;

/**
 * Normalizes a privacy-rule value for its type, or explains what is wrong.
 * Domains are accepted with a scheme or path ("https://bank.com/login" →
 * "bank.com"); regions must be "x,y,w,h".
 */
export function normalizeRuleValue(
  type: PrivacyRuleType,
  raw: string,
): { ok: true; value: string } | { ok: false; error: string } {
  const t = raw.trim();
  switch (type) {
    case "application":
      if (t === "") return { ok: false, error: "Enter an application name, e.g. 1Password." };
      if (t.length > 200)
        return { ok: false, error: "Keep the application name under 200 characters." };
      return { ok: true, value: t };
    case "window":
      if (t === "") return { ok: false, error: "Enter text that appears in the window title." };
      if (t.length > 200) return { ok: false, error: "Keep the title text under 200 characters." };
      return { ok: true, value: t };
    case "domain": {
      if (t === "") return { ok: false, error: "Enter a domain, e.g. bank.example.com." };
      let host = t.toLowerCase().replace(/^[a-z][a-z0-9+.-]*:\/\//, "");
      host = (host.split(/[/?#]/)[0] ?? "").replace(/:\d+$/, "").replace(/\.$/, "");
      if (!DOMAIN_RE.test(host))
        return { ok: false, error: "That is not a valid domain, e.g. bank.example.com." };
      return { ok: true, value: host };
    }
    case "region": {
      const r = parseRegion(t);
      if (!r.ok) return r;
      const { x, y, width, height } = r.rect;
      return { ok: true, value: `${x},${y},${width},${height}` };
    }
  }
}

export const RULE_TYPE_LABELS: Record<PrivacyRuleType, string> = {
  application: "Application",
  window: "Window title",
  domain: "Website domain",
  region: "Screen region",
};

export const RULE_TYPE_PLACEHOLDERS: Record<PrivacyRuleType, string> = {
  application: "e.g. 1Password",
  window: "e.g. Online Banking",
  domain: "e.g. bank.example.com",
  region: "x,y,width,height — e.g. 0,0,800,600",
};
