// Navigation state of the History section (selected meeting, list page).
// Lives outside the component so "Open in History" from Home can deep-link.

import { create } from "zustand";
import { useMain } from "@/main/store";

interface HistoryNavState {
  selectedId: string | null;
  page: number;
  openMeeting: (meetingId: string) => void;
  closeMeeting: () => void;
  setPage: (page: number) => void;
}

export const useHistoryNav = create<HistoryNavState>((set) => ({
  selectedId: null,
  page: 0,
  openMeeting: (meetingId) => set({ selectedId: meetingId }),
  closeMeeting: () => set({ selectedId: null }),
  setPage: (page) => set({ page: Math.max(0, page) }),
}));

/** Switches the main window to History with `meetingId` open. */
export function openInHistory(meetingId: string): void {
  useHistoryNav.getState().openMeeting(meetingId);
  useMain.getState().setSection("history");
}
