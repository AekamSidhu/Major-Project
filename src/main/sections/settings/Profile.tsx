// Profile & context: what the user tells OpenMeet so suggestions never have
// to invent facts (spec §17, §29, §30). Saved automatically while typing.

import { TextAreaSetting } from "../../components/fields";
import { saveSettings } from "../../components/save";
import { useSettings } from "../../components/useSettings";
import { Banner, PageHeader, Section } from "../../ui";

function topicCount(text: string): string {
  const n = text.split("\n").filter((l) => l.trim()).length;
  return n === 1 ? "1 topic" : `${n} topics`;
}

export function ProfileSection() {
  const p = useSettings().profile;
  return (
    <>
      <PageHeader
        title="Profile & context"
        description="Stored only on this device and sent to your AI provider as part of each request."
      />
      <Banner tone="accent" title="Suggestions use only what you provide">
        In candidate mode, suggested answers use facts from here and from the conversation. OpenMeet
        never invents employers, projects, numbers or achievements; where a personal example is
        needed, it shows a placeholder for you to fill in.
      </Banner>
      <Section title="Candidate mode">
        <TextAreaSetting
          label="Background"
          description="Your resume or notes in your own words: roles, projects, technologies, results you can talk about."
          value={p.background}
          maxLength={20000}
          rows={10}
          placeholder={
            "e.g. 6 years backend engineering. At Acme I led the migration of the billing service to Kafka (cut p99 latency from 900 ms to 120 ms)…"
          }
          onCommit={(v) =>
            saveSettings((d) => {
              d.profile.background = v;
            })
          }
        />
      </Section>
      <Section title="Both modes">
        <TextAreaSetting
          label="Role, job description or agenda"
          description="The position you're interviewing for (or hiring for), or the meeting's agenda."
          value={p.roleContext}
          maxLength={10000}
          rows={6}
          onCommit={(v) =>
            saveSettings((d) => {
              d.profile.roleContext = v;
            })
          }
        />
        <TextAreaSetting
          label="Custom instructions"
          description="Standing preferences for every suggestion, e.g. “Prefer Go examples” or “Keep answers under 60 seconds spoken”."
          value={p.customInstructions}
          maxLength={4000}
          rows={4}
          onCommit={(v) =>
            saveSettings((d) => {
              d.profile.customInstructions = v;
            })
          }
        />
      </Section>
      <Section title="Interviewer mode">
        <TextAreaSetting
          label="Planned topics"
          description="One per line. They appear in the overlay's topic tracker, and uncovered ones are suggested as the next area."
          value={p.plannedTopics}
          maxLength={4000}
          rows={6}
          placeholder={"System design\nConcurrency\nTesting strategy\nTeam collaboration"}
          footer={topicCount(p.plannedTopics)}
          onCommit={(v) =>
            saveSettings((d) => {
              d.profile.plannedTopics = v;
            })
          }
        />
      </Section>
    </>
  );
}
