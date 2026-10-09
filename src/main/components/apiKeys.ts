// Shared API-key status (never the keys themselves) so every key field on a
// page shows the same state after a save.

import { create } from "zustand";
import type { ApiKeyStatus, SecretProvider } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";

interface ApiKeyStore {
  statuses: Partial<Record<SecretProvider, ApiKeyStatus>>;
  loaded: boolean;
  loading: boolean;
  error: string | null;
  load: () => Promise<void>;
  save: (provider: SecretProvider, key: string) => Promise<void>;
  clear: (provider: SecretProvider) => Promise<void>;
}

export const useApiKeys = create<ApiKeyStore>((set, get) => ({
  statuses: {},
  loaded: false,
  loading: false,
  error: null,

  load: async () => {
    if (get().loading) return;
    set({ loading: true });
    try {
      const list = await api.getApiKeyStatus();
      const statuses: Partial<Record<SecretProvider, ApiKeyStatus>> = {};
      for (const s of list) statuses[s.provider] = s;
      set({ statuses, loaded: true, loading: false, error: null });
    } catch (e) {
      set({ loading: false, loaded: true, error: errorMessage(e) });
    }
  },

  /** Throws the backend error so the caller can show it next to the field. */
  save: async (provider, key) => {
    const status = await api.setApiKey(provider, key.trim());
    set((s) => ({ statuses: { ...s.statuses, [provider]: status } }));
  },

  clear: async (provider) => {
    const status = await api.clearApiKey(provider);
    set((s) => ({ statuses: { ...s.statuses, [provider]: status } }));
  },
}));

export const SECRET_LABELS: Record<SecretProvider, string> = {
  openai: "OpenAI",
  anthropic: "Anthropic",
  openai_compatible: "OpenAI-compatible server",
};
