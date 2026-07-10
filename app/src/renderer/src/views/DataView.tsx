import { UserSummary } from '../components/UserSummary'
import type { ProfileState } from '../lib/sidecar'

interface DataViewProps {
  profile: ProfileState | undefined
  onProfileChange: (profile: ProfileState) => void
}

export function DataView({ profile, onProfileChange }: DataViewProps): React.JSX.Element {
  return (
    <div
      className="flex h-full min-h-0 flex-col gap-2.5 overflow-y-auto px-px pb-4 pt-px"
      aria-label="Data"
    >
      <UserSummary profile={profile} onProfileChange={onProfileChange} />
    </div>
  )
}
