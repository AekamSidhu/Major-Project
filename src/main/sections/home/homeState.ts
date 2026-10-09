// UI-only state of the Home section that must survive navigating away.

import { create } from "zustand";

interface HomeUiState {
  /** Completed meeting whose summary view the user closed ("Start a new meeting"). */
  dismissedMeetingId: string | null;
  dismissCompleted: (meetingId: string) => void;
}

export const useHomeUi = create<HomeUiState>((set) => ({
  dismissedMeetingId: null,
  dismissCompleted: (meetingId) => set({ dismissedMeetingId: meetingId }),
}));
