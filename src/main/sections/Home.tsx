// Home section: start flow, live meeting, or the summary of the meeting that
// just ended — chosen from the live meeting state in the store.

import { useMain } from "@/main/store";
import { CompletedMeeting } from "./home/CompletedMeeting";
import { homeView } from "./home/helpers";
import { useHomeUi } from "./home/homeState";
import { LiveMeeting } from "./home/LiveMeeting";
import { StartMeeting } from "./home/StartMeeting";

export function HomeSection() {
  const settings = useMain((s) => s.settings);
  const live = useMain((s) => s.live);
  const dismissedId = useHomeUi((s) => s.dismissedMeetingId);
  if (!settings) return null;
  switch (homeView(live.meeting, live.state, dismissedId)) {
    case "live":
      return <LiveMeeting live={live} settings={settings} />;
    case "completed":
      return <CompletedMeeting live={live} />;
    case "start":
      return <StartMeeting settings={settings} />;
  }
}
