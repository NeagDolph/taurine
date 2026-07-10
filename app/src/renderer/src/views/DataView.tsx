import { UserSummary } from '../components/UserSummary'
import type { ProfileState } from '../lib/sidecar'

interface DataViewProps {
  profile: ProfileState | undefined
  onProfileChange: (profile: ProfileState) => void
}

export function DataView({ profile, onProfileChange }: DataViewProps): React.JSX.Element {
  return (
    <div className="page data-page" aria-label="Data">
      <UserSummary profile={profile} onProfileChange={onProfileChange} />
    </div>
  )
}
